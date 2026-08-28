from __future__ import annotations

"""
JPL HORIZONS REST API client.

Provides search, orbital element retrieval, physical parameter lookup,
and SPK ephemeris file download for solar system bodies.

API documentation: https://ssd-api.jpl.nasa.gov/doc/horizons.html
SBDB documentation: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
"""

import base64
import json
import logging
import math
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

#: Mean obliquity of the ecliptic at J2000.0, degrees (IAU 2006/2000A:
#: 23 deg 26' 21.406").
J2000_OBLIQUITY_DEG = 23.439279


def ecliptic_pole_to_equatorial(lambda_deg, lat_deg,
                                obliquity_deg=J2000_OBLIQUITY_DEG):
    """``(ra_deg, dec_deg)`` from an ecliptic pole ``(lambda, beta)``.

    THE canonical implementation for this project; ``gui_core.body_properties``
    re-exports it rather than keeping a second copy.  Both published sources of
    an asteroid spin pole -- SBDB's ``pole`` physical parameter and DAMIT's
    ``spin.txt`` -- give ECLIPTIC longitude and latitude, while PTM3D's
    ``generic_body.spin.pole_ra_deg`` / ``pole_dec_deg`` are EQUATORIAL.
    Writing one straight into the other is an error of up to the obliquity,
    23.4 degrees, which moves the sub-solar latitude and the seasons and which
    no plausibility check on a spin pole would catch.

    Written with ``cos(beta)`` rather than ``tan(beta)`` so an exactly polar
    ``beta = +-90`` (the orbit-normal case) is finite.
    """
    lam = math.radians(float(lambda_deg))
    bet = math.radians(float(lat_deg))
    eps = math.radians(float(obliquity_deg))

    sin_dec = (math.sin(bet) * math.cos(eps)
               + math.cos(bet) * math.sin(eps) * math.sin(lam))
    dec = math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))

    y = (math.cos(bet) * math.sin(lam) * math.cos(eps)
         - math.sin(bet) * math.sin(eps))
    x = math.cos(bet) * math.cos(lam)
    ra = math.degrees(math.atan2(y, x)) % 360.0
    return ra, dec


# Kept private-looking for the call sites inside this module, but the same
# object, so there is exactly one implementation.
_ecliptic_pole_to_equatorial = ecliptic_pole_to_equatorial


def _parse_sbdb_pole(value):
    """``(ra_deg, dec_deg)`` from SBDB's ``pole`` string, or ``(None, None)``.

    SBDB publishes the pole as ``"RA/Dec"`` in EQUATORIAL J2000 coordinates --
    the same frame ``generic_body.spin.pole_ra_deg`` / ``pole_dec_deg`` take,
    so it needs NO conversion.  Measured against the IAU WGCCRE values
    (Archinal et al. 2018) on four bodies, all agreeing to better than 0.03
    degrees::

        (4) Vesta   SBDB 309.06110/42.232386   IAU 309.031 / 42.235
        (1) Ceres   SBDB 291.421/66.758        IAU 291.418 / 66.764
        (433) Eros  SBDB 11.37/17.22           IAU  11.35  / 17.22
        (21) Lutetia SBDB 51.8/+10.8           IAU  52.0   / 12.0

    This is worth stating because DAMIT's ``spin.txt``, the OTHER source of a
    pole on this card, is ECLIPTIC ``(lambda, beta)`` and DOES need converting
    (see ``ecliptic_pole_to_equatorial``).  The two are not interchangeable:
    converting SBDB's pole as if it were ecliptic moved Vesta's from
    309.06/42.23 to 300.41/22.83, about 20 degrees, which is the same class of
    silent error as leaving DAMIT's unconverted.

    Some records carry a leading ``+`` or extra whitespace, and a few carry two
    solutions separated by a comma; only the first is read, because picking
    silently between two spin solutions is exactly the kind of invisible choice
    this project refuses to make -- the caller reports which it used.
    """
    text = str(value or "").strip()
    if not text:
        return None, None
    text = text.split(",")[0].strip()
    if "/" not in text:
        return None, None
    ra_text, _, dec_text = text.partition("/")
    try:
        ra = float(ra_text.strip())
        dec = float(dec_text.strip())
    except ValueError:
        return None, None
    if not (-90.0 <= dec <= 90.0):
        return None, None
    return ra % 360.0, dec


@dataclass
class BodySearchResult:
    """A single body returned from a HORIZONS search."""
    name: str
    naif_id: str
    body_type: str = ""  # "asteroid", "comet", "planet", "satellite"
    designation: str = ""


@dataclass
class OrbitalElements:
    """Keplerian orbital elements from HORIZONS."""
    semi_major_axis_au: float = 0.0
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    lon_ascending_node_deg: float = 0.0
    arg_perihelion_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    epoch_jd: float = 2451545.0  # J2000.0
    epoch_iso: str = "2000-01-01T12:00:00"
    orbital_period_days: float = 0.0
    mean_motion_deg_per_day: float = 0.0


@dataclass
class PhysicalParams:
    """Physical parameters from SBDB.

    ``albedo`` is SBDB's ``albedo`` field VERBATIM, which is the visual
    GEOMETRIC albedo p_V -- NOT a Bond albedo.  The engine's
    ``generic_body.surface.albedo`` is a Bond albedo (``thermal_model.cpp``
    computes ``1 - albedo`` and scales the solar constant by it), so anything
    that moves this field into a configuration must convert it first; see
    ``gui_core.body_properties.bond_albedo``.  The field is left named
    ``albedo`` because that is SBDB's own name for it and renaming it would
    break every existing caller for no gain -- the definition is stated here
    instead, and ``albedo_is_geometric`` marks it in the data.

    ``pole_ra_deg`` / ``pole_dec_deg`` carry SBDB's ``pole`` parameter, which
    is published in EQUATORIAL J2000 and needs no conversion (verified against
    IAU WGCCRE on four bodies; see ``_parse_sbdb_pole``).  DAMIT's spin
    solution, the other source of a pole, is ECLIPTIC and does -- that
    conversion lives in ``ecliptic_pole_to_equatorial`` and is applied by the
    DAMIT path, not here.
    """
    name: str = ""
    diameter_km: float = 0.0
    albedo: float = -1.0
    rotation_period_hours: float = 0.0
    spectral_type: str = ""
    gm_km3_s2: float = 0.0
    density_g_cm3: float = 0.0
    pole_ra_deg: float = -1.0
    pole_dec_deg: float = -1.0
    #: Absolute magnitude H, and the IAU H,G slope parameter G.  G is needed
    #: to turn p_V into a Bond albedo (q = 0.290 + 0.684 G); None means SBDB
    #: published none, which is DISTINCT from a fitted G of 0.0.
    absolute_magnitude_h: float = 0.0
    slope_parameter_g: Optional[float] = None
    #: An ecliptic pole, when one is known from a source that publishes in
    #: that frame (DAMIT).  SBDB does NOT: it publishes equatorial, so these
    #: stay unset on an SBDB lookup and ``pole_ra_deg``/``pole_dec_deg`` are
    #: the published pair.
    pole_ecliptic_lon_deg: float = -999.0
    pole_ecliptic_lat_deg: float = -999.0
    #: Always True for anything this client fills in from SBDB.  Present so a
    #: caller cannot mistake the field for a Bond albedo by reading the object
    #: alone.
    albedo_is_geometric: bool = True


# ---------------------------------------------------------------------------
# What went wrong, as five different answers
# ---------------------------------------------------------------------------
#
# Every failure out of this client used to be a bare RuntimeError, and the one
# caller that mattered -- the GUI's "Fetch kernel from HORIZONS" button --
# turned all of them into "HORIZONS could not be reached".  It said that while
# HORIZONS was up, had answered in 130 ms, and had explained in the response
# body exactly what it objected to.  A message that names the wrong subsystem
# sends whoever reads it to check their network instead of their request.
#
# These all derive from RuntimeError so that existing `except RuntimeError`
# call sites keep working unchanged.


class HorizonsError(RuntimeError):
    """Base for everything this client raises."""

    def __init__(self, message, server_text=""):
        super().__init__(message)
        #: What the server itself said, verbatim and untruncated. The message
        #: carries a shortened form; this is here for a log.
        self.server_text = str(server_text or "")


class HorizonsNetworkError(HorizonsError):
    """The service could not be reached, or did not answer in time.

    THE ONLY ONE that means "check the network". Nothing else does.
    """


class HorizonsHTTPError(HorizonsError):
    """The service answered with an HTTP error status."""

    def __init__(self, message, status=0, server_text=""):
        super().__init__(message, server_text)
        self.status = int(status or 0)


class HorizonsRefusalError(HorizonsError):
    """The service understood the request, resolved the body, and said no.

    The Bennu case: HORIZONS will not generate an SPK for an object that has a
    pre-computed ephemeris in its major-body index, and says so.  That is not a
    network fault, not a bad designation, and not something a retry fixes.
    """


class HorizonsNotFoundError(HorizonsError):
    """The service answered, and nothing matched the designation."""


class HorizonsParseError(HorizonsError):
    """The service answered with something this client does not understand.

    Distinct from a refusal on purpose: a refusal is HORIZONS working as
    designed and this is the client being out of date with it.  Reporting the
    second as the first hides a client bug behind a server message.
    """


# ---------------------------------------------------------------------------
# Addressing a small body the way HORIZONS actually wants it
# ---------------------------------------------------------------------------

#: The highest IAU asteroid number HORIZONS' small-body record index accepts,
#: read out of its own refusal text ("IAU numbered asteroids: 1- 895910") on
#: 2026-08-27.  It grows as asteroids are numbered; it is used only to decide
#: whether a bare number is plausibly a record number rather than an SPK ID,
#: and the two ranges are nowhere near each other, so drift is harmless.
SMALL_BODY_RECORD_MAX = 895910

#: HORIZONS' own record boundaries for the other two small-body classes, from
#: the same refusal text.  A number inside either is passed through as a record
#: number rather than treated as an SPK ID.
UNNUMBERED_RECORD_RANGE = (50000001, 50663878)
COMET_RECORD_RANGE = (90000001, 90004975)


def small_body_spk_command(body_id, body_type="asteroid"):
    """The HORIZONS ``COMMAND`` string that selects this small body.

    THIS IS THE BUG THAT BROKE THE BUTTON, so it is worth stating precisely.

    HORIZONS has two different numeric namespaces for a small body and the
    trailing semicolon selects between them and the major-body index -- but
    only ONE of the two small-body numbers goes in front of the semicolon:

    * ``COMMAND='623;'`` -- 623 is the IAU **record number**, and this works.
    * ``COMMAND='DES=2000623;'`` -- 2000623 is the **SPK ID**, and the ``DES=``
      keyword is what makes a number mean an SPK ID (JPL's Horizons API
      documentation, "COMMAND Parameter": ``'99942;'`` and ``'DES=2099942;'``
      select the same object).
    * ``COMMAND='2000623;'`` -- an SPK ID in the record-number slot.  HORIZONS
      answers ``DXREAD: requested IOBJ= 2000623 is out of bounds``, because
      record numbers stop at 895910.

    The card's lookup fills ``generic_body.naif_id`` with the SPK ID (2101955
    for Bennu, 20000623 for Chimaera) -- correctly: that is the id SPICE needs,
    and ``_spk_success`` deliberately re-reads it out of the kernel.  The old
    ``fetch_spk`` then appended a semicolon to it, producing the third form, so
    **every numbered asteroid failed**, not only Bennu.  Measured live on
    2026-08-27: ``'20000623;'`` out of bounds, ``'623;'`` and
    ``'DES=2000623;'`` both return a kernel.

    Both SPK-ID conventions are mapped back to the IAU number here, because the
    numeric record form is documented as a guaranteed-unique match and a name
    search is not.  HORIZONS migrated small-body SPK IDs from 2000000+N to
    20000000+N and this project's own cache carries both.

    ``body_type`` is honored: anything that is not an asteroid or a comet is
    passed through untouched, so a major body reaches HORIZONS as a major body
    and is refused as one.  Appending ``;`` to a major-body id would silently
    select a DIFFERENT OBJECT -- ``'599'`` is Jupiter and ``'599;'`` is asteroid
    599 Luisa, and both return an answer.

    Raises ``ValueError`` on an empty identifier rather than sending HORIZONS a
    bare semicolon, which it reads as "user-supplied osculating elements
    follow".
    """
    text = str(body_id or "").strip().rstrip(";").strip()
    if not text:
        raise ValueError(
            "There is no body identifier to ask HORIZONS about. A NAIF id, an "
            "asteroid number, a designation or a name is required.")

    if str(body_type or "").strip().lower() not in ("asteroid", "comet"):
        # Not ours to rewrite. Pass it through and let HORIZONS answer for it.
        return str(body_id or "").strip()

    if text.upper().startswith("DES="):
        return text + ";"

    if text.isdigit():
        value = int(text)
        if 1 <= value <= SMALL_BODY_RECORD_MAX:
            return f"{value};"                      # IAU record number
        for base in (20000000, 2000000):
            # An SPK ID. Map it back to the IAU number it encodes; the
            # remainder must still be a legal record number, or this is some
            # other number that happens to be large and DES= is the honest
            # thing to send.
            if base < value < base + SMALL_BODY_RECORD_MAX:
                return f"{value - base};"
        for low, high in (UNNUMBERED_RECORD_RANGE, COMET_RECORD_RANGE):
            if low <= value <= high:
                return f"{value};"
        return f"DES={value};"

    # Text. A provisional or survey designation ("1999 RQ36", "2116 T-2",
    # "A907 BC") needs the DES= keyword; a name does not, and HORIZONS resolves
    # "Chimaera;" on its own (measured: returns 20000623).
    if re.match(r"^(?:[A-Z]?\d{4}|A\d{3})\s+[A-Za-z]{1,2}[-\d]*$", text):
        return f"DES={text};"
    return f"{text};"


#: How much of HORIZONS' own text a raised error carries. The refusals worth
#: reading are one to six lines; 600 characters holds the longest of them
#: (the DXREAD record-boundary table) whole, and the untruncated text is on
#: the exception as ``server_text`` either way.
SERVER_TEXT_LIMIT = 600


def _tidy(text):
    """HORIZONS' response text, whitespace-normalized and length-capped."""
    body = " ".join(str(text or "").split())
    if len(body) <= SERVER_TEXT_LIMIT:
        return body
    return body[:SERVER_TEXT_LIMIT] + f" [... {len(body)} characters total]"


def _spk_failure(body_id, command, data):
    """The right exception for an SPK response that carried no kernel.

    Four outcomes, and they are four different things to do about it:

    ``error``           HORIZONS resolved the request and declined it. Its own
                        words are quoted; there is nothing to retry.
    "No matches found"  the DASTCOM index search returned nothing, so the
                        designation is wrong, not the request.
    neither             the reply has a shape this client does not know. That
                        is a client bug and is labeled as one rather than
                        reported as a server refusal.

    Returns the exception; the caller raises it, so the traceback starts at
    the request rather than in here.
    """
    error = data.get("error")
    result = data.get("result", "")
    if error:
        return HorizonsRefusalError(
            f"HORIZONS resolved COMMAND='{command}' (asked for {body_id!r}) "
            f"and refused to build an SPK. HORIZONS said: {_tidy(error)}",
            server_text=str(error))
    if "No matches found" in str(result):
        return HorizonsNotFoundError(
            f"HORIZONS' small-body index has no match for COMMAND='{command}' "
            f"(asked for {body_id!r}). HORIZONS said: {_tidy(result)}",
            server_text=str(result))
    return HorizonsParseError(
        f"HORIZONS answered COMMAND='{command}' (asked for {body_id!r}) with "
        f"no 'spk' payload, no 'error' and no index-search miss, so this "
        f"client does not know what happened. Keys returned: "
        f"{sorted(data)}. HORIZONS said: {_tidy(result)}",
        server_text=str(result))


@dataclass
class HorizonsBody:
    """Combined data for a body from HORIZONS + SBDB."""
    name: str = ""
    naif_id: str = ""
    body_type: str = ""
    elements: OrbitalElements = field(default_factory=OrbitalElements)
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    raw_horizons: dict = field(default_factory=dict)
    raw_sbdb: dict = field(default_factory=dict)


class HorizonsClient:
    """Client for the JPL HORIZONS and SBDB REST APIs."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _fetch_json(self, url: str, params: dict) -> dict:
        """Fetch JSON from a URL with query parameters."""
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
        logger.debug("Fetching: %s", full_url)

        req = urllib.request.Request(full_url)
        req.add_header("User-Agent", "PTM3D-HorizonsClient/1.0")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise HorizonsHTTPError(
                f"HORIZONS API error {e.code}: {body[:500]}",
                status=e.code, server_text=body) from e
        except urllib.error.URLError as e:
            raise HorizonsNetworkError(
                f"Network error contacting HORIZONS: {e.reason}") from e
        except socket.timeout as e:                          # pragma: no cover
            raise HorizonsNetworkError(
                f"HORIZONS did not answer within {self.timeout} s") from e
        try:
            return json.loads(raw)
        except ValueError as e:
            # A 200 whose body is not JSON. That is the client and the service
            # disagreeing about the contract, not a network fault, and calling
            # it one sends the reader to check their network.
            raise HorizonsParseError(
                f"HORIZONS answered with something that is not JSON "
                f"({e}): {raw[:300]}", server_text=raw) from e

    def _fetch_raw(self, url: str, params: dict) -> str:
        """Fetch raw text response from HORIZONS API."""
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
        logger.debug("Fetching raw: %s", full_url)

        req = urllib.request.Request(full_url)
        req.add_header("User-Agent", "PTM3D-HorizonsClient/1.0")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise HorizonsHTTPError(
                f"HORIZONS API error {e.code}: {body[:500]}",
                status=e.code, server_text=body) from e
        except urllib.error.URLError as e:
            raise HorizonsNetworkError(
                f"Network error contacting HORIZONS: {e.reason}") from e

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_body(self, query: str) -> list[BodySearchResult]:
        """
        Search HORIZONS for a body by name, number, or designation.

        Returns a list of matching bodies.
        """
        params = {
            "format": "json",
            "COMMAND": f"'{query}'",
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "NO",
        }

        data = self._fetch_json(HORIZONS_API_URL, params)

        # If HORIZONS returns a unique match it puts result directly;
        # if ambiguous it returns a 'list' key with candidates.
        results = []

        if "result" in data:
            result_text = data["result"]

            # Check for "Multiple major-bodies match" or similar
            if "multiple" in result_text.lower() and "ID#" in result_text:
                results = self._parse_major_body_list(result_text)
            elif "small-body" in result_text.lower() and "Record #" in result_text:
                results = self._parse_small_body_list(result_text)
            else:
                # Single match - try several header formats
                result = self._parse_single_body(result_text)
                if result:
                    results.append(result)

        return results

    def _parse_single_body(self, text: str) -> Optional[BodySearchResult]:
        """Parse a single-body HORIZONS response.

        Handles several formats:
          - "Target body name: Mars (499)  {source...}"
          - "JPL/HORIZONS  4 Vesta (A807 FA)  ..." with "Rec #: 4"
          - Major planets with "Revised:" header
        """
        # Format 1: "Target body name: <name> {ID}" (major bodies with ephemeris)
        m = re.search(
            r"Target body name:\s*(.+?)(?:\s*\{(\d+)\}|\s*\((\d+)\))", text
        )
        if m:
            name = m.group(1).strip()
            naif_id = m.group(2) or m.group(3) or ""
            return BodySearchResult(name=name, naif_id=naif_id)

        # Format 2: "JPL/HORIZONS  <number> <name> (<designation>)"
        # with "Rec #:  <number>"  (small bodies like asteroids)
        m = re.search(
            r"JPL/HORIZONS\s+(\d+)\s+(\S.*?)\s+\(", text
        )
        rec_m = re.search(r"Rec\s*#:\s*(\d+)", text)
        if m:
            number = m.group(1)
            name = f"{number} {m.group(2).strip()}"
            naif_id = rec_m.group(1) if rec_m else number
            body_type = "asteroid"
            # Check if it's a comet
            if "comet" in text.lower():
                body_type = "comet"
            return BodySearchResult(
                name=name, naif_id=naif_id, body_type=body_type
            )

        # Format 3: "Revised:" header (major planets / barycenters)
        # e.g. "Revised: April 12, 2021        Mars Barycenter              4"
        if "Revised" in text:
            # Try "Target body name:" first
            name_m = re.search(
                r"Target body name:\s*(.+?)(?:\(|\{|\n)", text
            )
            id_m = re.search(r"\{(\d+)\}", text)
            if name_m:
                return BodySearchResult(
                    name=name_m.group(1).strip(),
                    naif_id=id_m.group(1) if id_m else "",
                )

            # Try "Revised: ... <Name>  <ID> / <barycenter>" or
            #      "Revised: ... <Name>  <ID>" format
            rev_m = re.search(
                r"Revised[^\n]*?\s{2,}(\S.*?)\s{2,}(\d+)\s*(?:/\s*\d+)?\s*$",
                text, re.MULTILINE
            )
            if rev_m:
                return BodySearchResult(
                    name=rev_m.group(1).strip(),
                    naif_id=rev_m.group(2),
                )

        return None

    def _parse_major_body_list(self, text: str) -> list[BodySearchResult]:
        """Parse the major-body multiple-match table from HORIZONS."""
        results = []
        # Lines like: "  199      Mercury                 Mercury
        for line in text.split("\n"):
            m = re.match(r"\s*(\d+)\s+(\S.*?)(?:\s{2,}|\s*$)", line)
            if m:
                naif_id = m.group(1).strip()
                name = m.group(2).strip()
                results.append(BodySearchResult(name=name, naif_id=naif_id))
        return results

    def _parse_small_body_list(self, text: str) -> list[BodySearchResult]:
        """Parse the small-body multiple-match table from HORIZONS."""
        results = []
        # Lines like: "Record #  Epoch-yr  Primary Coverage ...  >DIFFCORR"
        in_table = False
        for line in text.split("\n"):
            if "Record #" in line:
                in_table = True
                continue
            if in_table:
                if line.strip() == "" or line.startswith("----"):
                    if results:
                        break
                    continue
                m = re.match(r"\s*(\d+)\s+(.*)", line)
                if m:
                    rec = m.group(1).strip()
                    rest = m.group(2).strip()
                    # Extract name/designation from the rest of the line
                    name = rest.split("  ")[0].strip() if rest else f"Record {rec}"
                    results.append(BodySearchResult(
                        name=name,
                        naif_id=rec,
                        body_type="asteroid",
                    ))
        return results

    # ------------------------------------------------------------------
    # Orbital elements
    # ------------------------------------------------------------------

    def get_orbital_elements(
        self, body_id: str, epoch: str = "2451545.0"
    ) -> OrbitalElements:
        """
        Get osculating orbital elements for a body at a given epoch.

        Parameters
        ----------
        body_id : str
            HORIZONS command string (NAIF ID, name, or designation).
            For small bodies use 'DES=<designation>' or the record number.
        epoch : str
            Julian date for the osculating elements (default: J2000.0).
        """
        params = {
            "format": "json",
            "COMMAND": f"'{body_id}'",
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "ELEMENTS",
            "CENTER": "'500@10'",  # Sun (heliocentric)
            "START_TIME": f"JD{epoch}",
            "STOP_TIME": f"JD{float(epoch) + 1}",
            "STEP_SIZE": "1",
            "REF_PLANE": "ECLIPTIC",
            "REF_SYSTEM": "J2000",
            "OUT_UNITS": "AU-D",
        }

        data = self._fetch_json(HORIZONS_API_URL, params)
        result_text = data.get("result", "")

        return self._parse_elements(result_text, epoch)

    def _parse_elements(self, text: str, epoch: str) -> OrbitalElements:
        """Parse orbital elements from HORIZONS text output."""
        elems = OrbitalElements()
        elems.epoch_jd = float(epoch)

        # Find the data block between $$SOE and $$EOE
        soe_idx = text.find("$$SOE")
        eoe_idx = text.find("$$EOE")
        if soe_idx < 0 or eoe_idx < 0:
            logger.warning("Could not find element data in HORIZONS response")
            return elems

        data_block = text[soe_idx + 5 : eoe_idx].strip()

        # Parse key=value pairs from the element block
        patterns = {
            "EC": "eccentricity",
            "QR": None,  # perihelion distance (AU) - not directly stored
            "IN": "inclination_deg",
            "OM": "lon_ascending_node_deg",
            "W": "arg_perihelion_deg",
            "MA": "mean_anomaly_deg",
            "A": "semi_major_axis_au",
            "N": "mean_motion_deg_per_day",
            "PR": "orbital_period_days",
        }

        for line in data_block.split("\n"):
            for key, attr in patterns.items():
                # Match patterns like "EC= 7.607126566E-02" or "A = 2.767..."
                pattern = rf"\b{re.escape(key)}\s*=\s*([Ee0-9.+-]+)"
                m = re.search(pattern, line)
                if m and attr is not None:
                    try:
                        setattr(elems, attr, float(m.group(1)))
                    except ValueError:
                        pass

        return elems

    # ------------------------------------------------------------------
    # Physical parameters (SBDB)
    # ------------------------------------------------------------------

    def get_physical_params(self, body_id: str) -> PhysicalParams:
        """
        Get physical parameters from the Small Body Database (SBDB).

        Works for asteroids and comets. For major bodies, returns defaults.

        Parameters
        ----------
        body_id : str
            Body name, number, or designation.
        """
        params = {
            "sstr": body_id,
            "phys-par": "true",
        }

        try:
            data = self._fetch_json(SBDB_API_URL, params)
        except RuntimeError:
            logger.info("SBDB lookup failed for '%s' (may be a major body)", body_id)
            return PhysicalParams(name=body_id)

        phys = PhysicalParams()

        obj = data.get("object", {})
        phys.name = obj.get("fullname", obj.get("shortname", body_id))

        pp = data.get("phys_par", [])
        for entry in pp:
            name = entry.get("name", "")
            value_str = entry.get("value", "")

            # `pole` and `spec_T` are NOT numbers, so they are handled before
            # the float() gate.  `pole` in particular arrives as the string
            # "lambda/beta" -- ECLIPTIC longitude and latitude, per SBDB's own
            # field documentation -- and float() on it raises, which is why
            # this parameter was silently dropped by every earlier version of
            # this parser and no lookup has ever filled in a spin pole.
            if name == "pole":
                ra, dec = _parse_sbdb_pole(value_str)
                if ra is not None:
                    # EQUATORIAL as published; verified against IAU WGCCRE on
                    # four bodies (see _parse_sbdb_pole). NOT converted.
                    phys.pole_ra_deg, phys.pole_dec_deg = ra, dec
                continue
            if name == "spec_T":
                phys.spectral_type = value_str
                continue

            try:
                value = float(value_str) if value_str else 0.0
            except ValueError:
                continue

            if name == "diameter":
                phys.diameter_km = value
            elif name == "albedo":
                # SBDB's GEOMETRIC albedo p_V.  Stored as published; the Bond
                # conversion belongs to whoever writes it into a config.
                phys.albedo = value
            elif name == "rot_per":
                phys.rotation_period_hours = value
            elif name == "GM":
                phys.gm_km3_s2 = value
            elif name == "density":
                phys.density_g_cm3 = value
            elif name == "H":
                phys.absolute_magnitude_h = value
            elif name == "G":
                # A fitted G of 0.0 is a real value and must not read as
                # "absent"; None is the only "absent".
                phys.slope_parameter_g = value

        return phys

    # ------------------------------------------------------------------
    # SPK download
    # ------------------------------------------------------------------

    def download_spk(
        self,
        body_id: str,
        start_time: str,
        stop_time: str,
        output_path: str,
        body_type: str = "asteroid",
    ) -> str:
        """
        Download a binary SPK ephemeris file from HORIZONS.

        SPK generation is restricted to small bodies -- asteroids and comets --
        and HORIZONS refuses it for a further subset even of those: an object
        whose ephemeris it serves pre-computed out of its major-body index gets
        ``SPK creation is not available for pre-computed objects in the major
        body index``, whichever designation form is used.  (101955) Bennu, whose
        record HORIZONS marks ``*SPECIAL*`` and sources from ``ORX_merged``, is
        one; measured 2026-08-27 against ``101955;``, ``DES=101955;``,
        ``DES=2101955;`` and ``DES=1999 RQ36;``, all four refused, while
        VECTORS for the same object are served normally.  That is a
        :class:`HorizonsRefusalError`, not a network fault and not a bad
        designation, and the caller is told which.

        ``body_id`` is whatever the caller has -- an SPK ID, an IAU number, a
        designation or a name.  Turning it into the COMMAND string HORIZONS
        wants happens HERE, in :func:`small_body_spk_command`, and not in the
        caller: the old arrangement had the GUI append the semicolon itself,
        which is how an SPK ID ended up in the record-number slot and broke the
        fetch for every numbered asteroid.

        Parameters
        ----------
        body_id : str
            SPK ID, IAU number, designation or name.
        start_time, stop_time : str
            ISO dates. A "T" separator is accepted and normalized.
        output_path : str
            Local path to write the SPK file.
        body_type : str
            "asteroid"/"comet" get the small-body command form; anything else
            is passed to HORIZONS untouched.

        Returns
        -------
        str
            Path to the written SPK file.
        """
        command = small_body_spk_command(body_id, body_type)

        # HORIZONS SPK generation doesn't accept ISO "T" separator in dates.
        # Convert "2025-01-01T00:00:00" -> "2025-01-01 00:00:00".
        clean_start = start_time.replace("T", " ")
        clean_stop = stop_time.replace("T", " ")

        params = {
            "format": "json",
            "COMMAND": f"'{command}'",
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "SPK",
            "CENTER": "'500@0'",  # Solar System Barycenter
            "START_TIME": f"'{clean_start}'",
            "STOP_TIME": f"'{clean_stop}'",
        }

        logger.info("HORIZONS SPK request: COMMAND='%s' (from %r), %s to %s",
                    command, body_id, clean_start, clean_stop)
        data = self._fetch_json(HORIZONS_API_URL, params)

        spk_b64 = data.get("spk")
        if not spk_b64:
            raise _spk_failure(body_id, command, data)

        try:
            spk_bytes = base64.b64decode(spk_b64)
        except Exception as exc:                             # noqa: BLE001
            raise HorizonsParseError(
                f"HORIZONS returned an 'spk' payload for {command!r} that is "
                f"not valid base64 ({exc}). Nothing was written.") from exc

        # A payload that decodes but is not a kernel would otherwise be written
        # out with a .spk name and fail much later, inside SPICE, as somebody
        # else's problem. Every SPK begins "DAF/SPK " (NAIF DAF spec).
        if not spk_bytes.startswith(b"DAF/SPK "):
            raise HorizonsParseError(
                f"HORIZONS returned {len(spk_bytes)} bytes for {command!r} that "
                f"do not begin with the SPK file signature 'DAF/SPK ' (got "
                f"{spk_bytes[:16]!r}). Nothing was written.")

        with open(output_path, "wb") as f:
            f.write(spk_bytes)

        logger.info("Wrote SPK file (%d bytes, HORIZONS spk_file_id %s): %s",
                    len(spk_bytes), data.get("spk_file_id", "?"), output_path)
        return output_path

    def get_spk_id(self, body_id: str) -> Optional[str]:
        """
        Get the NAIF SPK ID for a body, which is needed for SPICE loading.

        For small bodies, HORIZONS assigns SPK IDs like 2000001 (Ceres),
        2000004 (Vesta), etc.
        """
        params = {
            "format": "json",
            "COMMAND": f"'{body_id}'",
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "NO",
        }

        data = self._fetch_json(HORIZONS_API_URL, params)
        result_text = data.get("result", "")

        # Look for SPK ID in the response
        m = re.search(r"SPK-ID\s*[:=]\s*(\d+)", result_text, re.IGNORECASE)
        if m:
            return m.group(1)

        # Try NAIF ID pattern
        m = re.search(r"\{(\d+)\}", result_text)
        if m:
            return m.group(1)

        return None

    # ------------------------------------------------------------------
    # Combined lookup
    # ------------------------------------------------------------------

    def get_body(
        self, query: str, epoch: str = "2451545.0"
    ) -> HorizonsBody:
        """
        Combined lookup: search + orbital elements + physical params.

        Parameters
        ----------
        query : str
            Body name, number, or designation.
        epoch : str
            Julian date for orbital elements.

        Returns
        -------
        HorizonsBody
            Combined data from HORIZONS and SBDB.
        """
        body = HorizonsBody()

        # Clean query: strip trailing semicolons for display/SBDB,
        # but keep for HORIZONS command (semicolon = small body hint)
        clean_query = query.rstrip("; ")
        horizons_cmd = query  # Keep as-is for HORIZONS

        body.name = clean_query

        # Search first to get NAIF ID (use HORIZONS command form)
        results = self.search_body(horizons_cmd)
        if results:
            body.naif_id = results[0].naif_id
            body.name = results[0].name
            body.body_type = results[0].body_type

        # Get orbital elements.
        # For small bodies (asteroids/comets), HORIZONS record numbers can
        # collide with major-body NAIF IDs (e.g. record #4 = Vesta, NAIF 4 =
        # Mars Barycenter).  Appending a semicolon tells HORIZONS to interpret
        # the number as a small-body record number.
        cmd = body.naif_id if body.naif_id else horizons_cmd
        if cmd and body.body_type in ("asteroid", "comet"):
            cmd = cmd.rstrip(";") + ";"
        try:
            body.elements = self.get_orbital_elements(cmd, epoch)
        except RuntimeError as e:
            logger.warning("Failed to get orbital elements: %s", e)

        # Get physical parameters from SBDB (use clean name, no semicolons)
        sbdb_query = clean_query
        try:
            body.physical = self.get_physical_params(sbdb_query)
            # Use SBDB name if we got a good result
            if body.physical.name and body.physical.name != sbdb_query:
                body.name = body.physical.name
        except RuntimeError as e:
            logger.warning("Failed to get physical params: %s", e)

        return body
