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
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


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
    """Physical parameters from SBDB."""
    name: str = ""
    diameter_km: float = 0.0
    albedo: float = -1.0
    rotation_period_hours: float = 0.0
    spectral_type: str = ""
    gm_km3_s2: float = 0.0
    density_g_cm3: float = 0.0
    pole_ra_deg: float = -1.0
    pole_dec_deg: float = -1.0


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
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HORIZONS API error {e.code}: {body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Network error contacting HORIZONS: {e.reason}"
            ) from e

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
            raise RuntimeError(
                f"HORIZONS API error {e.code}: {body[:500]}"
            ) from e

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
            try:
                value = float(value_str) if value_str else 0.0
            except ValueError:
                continue

            if name == "diameter":
                phys.diameter_km = value
            elif name == "albedo":
                phys.albedo = value
            elif name == "rot_per":
                phys.rotation_period_hours = value
            elif name == "spec_T":
                phys.spectral_type = value_str
            elif name == "GM":
                phys.gm_km3_s2 = value
            elif name == "density":
                phys.density_g_cm3 = value

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
    ) -> str:
        """
        Download a binary SPK ephemeris file from HORIZONS.

        SPK files are available for small bodies (asteroids, comets).
        The file is base64-encoded in the API response.

        Parameters
        ----------
        body_id : str
            HORIZONS command (NAIF ID or designation).
        start_time : str
            Start time in ISO format (e.g. "2025-01-01").
        stop_time : str
            Stop time in ISO format (e.g. "2030-01-01").
        output_path : str
            Local path to write the SPK file.

        Returns
        -------
        str
            Path to the written SPK file.
        """
        # HORIZONS SPK generation doesn't accept ISO "T" separator in dates.
        # Convert "2025-01-01T00:00:00" → "2025-01-01 00:00:00" or just the date.
        clean_start = start_time.replace("T", " ")
        clean_stop = stop_time.replace("T", " ")

        params = {
            "format": "json",
            "COMMAND": f"'{body_id}'",
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "SPK",
            "CENTER": "'500@0'",  # Solar System Barycenter
            "START_TIME": f"'{clean_start}'",
            "STOP_TIME": f"'{clean_stop}'",
        }

        data = self._fetch_json(HORIZONS_API_URL, params)

        spk_b64 = data.get("spk")
        if not spk_b64:
            # Check if the response indicates an error
            result = data.get("result", "")
            raise RuntimeError(
                f"No SPK data returned for body '{body_id}'. "
                f"SPK may not be available for this body type. "
                f"Response: {result[:300]}"
            )

        spk_bytes = base64.b64decode(spk_b64)
        with open(output_path, "wb") as f:
            f.write(spk_bytes)

        logger.info("Wrote SPK file (%d bytes): %s", len(spk_bytes), output_path)
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
