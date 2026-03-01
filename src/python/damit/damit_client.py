from __future__ import annotations

"""
DAMIT (Database of Asteroid Models from Inversion Techniques) client.

Downloads asteroid shape models and spin parameters from the DAMIT database.
DAMIT does not have a formal REST API; this client works by constructing
download URLs from the web interface.

Main site: https://damit.cuni.cz/
Documentation: https://damit.cuni.cz/pages/documentation
"""

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# DAMIT uses two domains interchangeably
DAMIT_BASE_URL = "https://damit.cuni.cz"
DAMIT_MIRROR_URL = "https://astro.troja.mff.cuni.cz/projects/damit"


@dataclass
class DAMITSpinParams:
    """Spin parameters from a DAMIT model."""
    lambda_ecl_deg: float = 0.0   # Ecliptic longitude of pole (deg)
    beta_ecl_deg: float = 0.0     # Ecliptic latitude of pole (deg)
    period_hours: float = 0.0     # Sidereal rotation period (hours)
    epoch_jd: float = 2451545.0   # Reference epoch (JD)

    @property
    def pole_ra_deg(self) -> float:
        """Convert ecliptic pole to equatorial RA (approximate)."""
        import math
        lam = math.radians(self.lambda_ecl_deg)
        bet = math.radians(self.beta_ecl_deg)
        eps = math.radians(23.4393)  # J2000 obliquity

        sin_ra = (math.sin(lam) * math.cos(eps)
                  - math.tan(bet) * math.sin(eps))
        cos_ra = math.cos(lam)
        ra = math.degrees(math.atan2(sin_ra, cos_ra)) % 360.0
        return ra

    @property
    def pole_dec_deg(self) -> float:
        """Convert ecliptic pole to equatorial Dec (approximate)."""
        import math
        lam = math.radians(self.lambda_ecl_deg)
        bet = math.radians(self.beta_ecl_deg)
        eps = math.radians(23.4393)

        sin_dec = (math.sin(bet) * math.cos(eps)
                   + math.cos(bet) * math.sin(eps) * math.sin(lam))
        dec = math.degrees(math.asin(max(-1, min(1, sin_dec))))
        return dec


@dataclass
class DAMITShapeModel:
    """A shape model from DAMIT."""
    damit_id: int = 0
    asteroid_number: int = 0
    asteroid_name: str = ""
    vertices: list[list[float]] = field(default_factory=list)  # [[x,y,z], ...]
    faces: list[list[int]] = field(default_factory=list)       # [[v1,v2,v3], ...] 0-indexed
    spin: DAMITSpinParams = field(default_factory=DAMITSpinParams)
    is_scaled: bool = False  # True if coordinates are in km
    quality_flag: int = 0

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


class DAMITClient:
    """Client for downloading DAMIT asteroid shape models."""

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        self.timeout = timeout
        self._ssl_context: Optional[ssl.SSLContext] = None
        if not verify_ssl:
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE

    def _fetch(self, url: str) -> str:
        """Fetch text content from a URL."""
        logger.debug("Fetching: %s", url)
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PTM3D-DAMITClient/1.0")

        try:
            if self._ssl_context:
                with urllib.request.urlopen(
                    req, timeout=self.timeout, context=self._ssl_context
                ) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            else:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

    def _parse_browse_page(self, html: str) -> list[dict]:
        """Parse asteroid entries from a DAMIT browse page HTML."""
        pattern = re.compile(
            r'/asteroids/view/(\d+).*?'
            r'\((\d+)\)\s*(\w[^<]*)',
            re.DOTALL,
        )
        results = []
        seen = set()
        for m in pattern.finditer(html):
            damit_id = int(m.group(1))
            if damit_id in seen:
                continue
            seen.add(damit_id)
            results.append({
                "damit_id": damit_id,
                "number": int(m.group(2)),
                "name": m.group(3).strip(),
            })
        return results

    def _browse_page(self, page: int) -> list[dict]:
        """Fetch a single paginated browse page and return parsed entries.

        Returns an empty list if the page doesn't exist (404) or on error.
        """
        url = f"{DAMIT_BASE_URL}/projects/damit/asteroids/browse/page:{page}"
        try:
            html = self._fetch(url)
            return self._parse_browse_page(html)
        except RuntimeError:
            pass
        # Fallback to mirror
        try:
            url = f"{DAMIT_MIRROR_URL}/asteroids/browse/page:{page}"
            html = self._fetch(url)
            return self._parse_browse_page(html)
        except RuntimeError:
            return []

    def search_asteroid(self, query: str) -> list[dict]:
        """
        Search DAMIT for an asteroid by name or number.

        Uses paginated browse pages (sorted by asteroid number) to find
        the correct entry, since the DAMIT search parameter is unreliable.

        Parameters
        ----------
        query : str
            Asteroid name (e.g. "Vesta") or number (e.g. "4").

        Returns
        -------
        list[dict]
            List of matches with keys: damit_id, number, name.
            Results are filtered to match the query by number or name.
        """
        # Clean query: strip trailing semicolons (HORIZONS convention)
        clean = query.strip().rstrip(";").strip()

        # Extract asteroid number from the query.  Handles formats:
        #   "4"  /  "4;"  /  "4 Vesta"  /  "4 Vesta (A807 FA)"
        query_number = None
        try:
            query_number = int(clean)
        except ValueError:
            m_num = re.match(r'^(\d+)\s', clean)
            if m_num:
                query_number = int(m_num.group(1))

        # Extract name words for fallback matching
        q_lower = clean.lower()
        name_words = re.findall(r'[a-zA-Z]{3,}', q_lower)

        # --- Page 1: always fetch first (covers low-numbered asteroids) ---
        page1 = self._browse_page(1)

        # Check page 1 for matches
        filtered = self._filter_entries(page1, query_number, q_lower, name_words)
        if filtered:
            return filtered

        # --- Paginated search for higher-numbered asteroids ---
        # Browse pages are sorted by asteroid number, ~15 entries per page.
        # Not every asteroid has a DAMIT model, so page numbers don't map
        # exactly to asteroid numbers. Use binary search.
        if query_number is not None and query_number > 15:
            result = self._search_by_page(query_number, name_words, q_lower)
            if result:
                return result

        logger.info(
            "No DAMIT model matches query '%s'", query,
        )
        return []

    def _filter_entries(
        self,
        entries: list[dict],
        query_number: "int | None",
        q_lower: str,
        name_words: list[str],
    ) -> list[dict]:
        """Filter browse page entries against query number and name."""
        if query_number is not None:
            exact = [r for r in entries if r["number"] == query_number]
            if exact:
                return exact

        # Word-level name match
        matched = []
        for r in entries:
            name_lower = r["name"].lower()
            if q_lower in name_lower or name_lower in q_lower:
                matched.append(r)
            elif any(w in name_lower for w in name_words):
                matched.append(r)
        return matched

    def _search_by_page(
        self,
        target_number: int,
        name_words: list[str],
        q_lower: str,
    ) -> list[dict]:
        """Binary search through paginated browse to find an asteroid by number."""
        per_page = 15
        # Initial guess: not all asteroids are in DAMIT, so the actual page
        # is lower than target_number / per_page.  Start at ~60% density estimate.
        lo, hi = 1, max(2, target_number // (per_page - 3))

        # Probe the upper bound to make sure it's high enough
        for _ in range(3):
            entries = self._browse_page(hi)
            if not entries:
                # Past the last page
                hi = hi - 1
                break
            max_num = max(r["number"] for r in entries)
            if max_num >= target_number:
                break
            # Double the upper bound
            lo = hi
            hi = hi * 2

        # Binary search
        best_entries: list[dict] = []
        for _ in range(12):  # At most ~12 iterations for 4000 pages
            if lo > hi:
                break
            mid = (lo + hi) // 2
            entries = self._browse_page(mid)
            if not entries:
                hi = mid - 1
                continue

            numbers = [r["number"] for r in entries]
            min_num, max_num = min(numbers), max(numbers)

            # Check if target is on this page
            exact = [r for r in entries if r["number"] == target_number]
            if exact:
                return exact

            if target_number < min_num:
                hi = mid - 1
            elif target_number > max_num:
                lo = mid + 1
            else:
                # target_number is within the range but not present —
                # this asteroid is not in DAMIT
                logger.info(
                    "Asteroid %d not in DAMIT (page %d has %d..%d)",
                    target_number, mid, min_num, max_num,
                )
                return []

        return best_entries

    def _get_model_links(self, damit_id: int) -> dict:
        """Scrape the asteroid view page / AJAX endpoint for file links.

        Returns a dict with optional keys ``shape_txt_url``,
        ``spin_txt_url``, ``shape_obj_url``, and ``model_id``.
        """
        links: dict = {}

        # Try the AJAX endpoint first (lighter than the full view page)
        ajax_url = (
            f"{DAMIT_BASE_URL}/projects/damit/asteroid_models/related"
            f"?showAsteroidColumn=0&showStoredFileColumn=1"
            f"&asteroidId={damit_id}"
        )
        html = None
        for url in [
            ajax_url,
            f"{DAMIT_BASE_URL}/projects/damit/asteroids/view/{damit_id}",
        ]:
            try:
                html = self._fetch(url)
                break
            except RuntimeError:
                continue

        if not html:
            return links

        # Extract model_id from "asteroid_models/view/{model_id}"
        model_match = re.search(r'/asteroid_models/view/(\d+)', html)
        if model_match:
            links["model_id"] = int(model_match.group(1))

        # Extract shape.txt stored-file link
        shape_match = re.search(
            r'href="(/projects/damit/stored_files/open/\d+/shape\.txt)"',
            html,
        )
        if shape_match:
            links["shape_txt_url"] = DAMIT_BASE_URL + shape_match.group(1)

        # Generated files use a predictable URL based on model_id
        if "model_id" in links:
            mid = links["model_id"]
            base = f"{DAMIT_BASE_URL}/projects/damit/generated_files/open/AsteroidModel/{mid}"
            links["spin_txt_url"] = f"{base}/spin.txt"
            links["shape_obj_url"] = f"{base}/shape.obj"

        return links

    def get_model(self, damit_id: int) -> DAMITShapeModel:
        """
        Download a shape model and spin parameters for a DAMIT entry.

        Parameters
        ----------
        damit_id : int
            DAMIT asteroid database ID (from search results).

        Returns
        -------
        DAMITShapeModel
            Shape model with vertices, faces, and spin parameters.
        """
        model = DAMITShapeModel(damit_id=damit_id)

        # Scrape the view page for actual download URLs
        links = self._get_model_links(damit_id)

        # --- Download shape ---
        shape_text = None

        # 1. Try the stored-file link parsed from the view page
        if "shape_txt_url" in links:
            try:
                shape_text = self._fetch(links["shape_txt_url"])
            except RuntimeError:
                pass

        # 2. Fallback: try the generated OBJ and convert
        #    (not needed if shape.txt worked, but acts as backup)

        if shape_text:
            self._parse_shape(shape_text, model)
        else:
            logger.warning("Could not download shape for DAMIT ID %d", damit_id)

        # --- Download spin parameters ---
        spin_loaded = False

        # 1. Try the generated-files URL
        if "spin_txt_url" in links:
            try:
                spin_text = self._fetch(links["spin_txt_url"])
                self._parse_spin(spin_text, model)
                spin_loaded = True
            except RuntimeError:
                pass

        if not spin_loaded:
            logger.warning("Could not download spin for DAMIT ID %d", damit_id)

        return model

    def get_model_by_number(self, asteroid_number: int) -> Optional[DAMITShapeModel]:
        """
        Download a shape model by asteroid catalog number.

        Parameters
        ----------
        asteroid_number : int
            IAU asteroid number (e.g. 4 for Vesta).

        Returns
        -------
        DAMITShapeModel or None
            Shape model if found, None otherwise.
        """
        results = self.search_asteroid(str(asteroid_number))

        # Find the match with the right number
        for r in results:
            if r.get("number") == asteroid_number:
                model = self.get_model(r["damit_id"])
                model.asteroid_number = asteroid_number
                model.asteroid_name = r.get("name", "")
                return model

        # If no exact match, try the first result
        if results:
            model = self.get_model(results[0]["damit_id"])
            model.asteroid_number = results[0].get("number", 0)
            model.asteroid_name = results[0].get("name", "")
            return model

        return None

    def _parse_shape(self, text: str, model: DAMITShapeModel) -> None:
        """
        Parse DAMIT shape.txt format.

        Format:
          Line 1: num_vertices num_faces
          Lines 2..nv+1: x y z  (vertex coordinates)
          Lines nv+2..end: v1 v2 v3  (face indices, 1-based)
        """
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return

        # First line: counts
        parts = lines[0].split()
        if len(parts) < 2:
            logger.warning("Invalid shape header: %s", lines[0])
            return

        nv = int(parts[0])
        nf = int(parts[1])

        # Read vertices
        vertices = []
        for i in range(1, min(nv + 1, len(lines))):
            coords = lines[i].split()
            if len(coords) >= 3:
                vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])

        # Read faces (convert from 1-based to 0-based)
        faces = []
        for i in range(nv + 1, min(nv + nf + 1, len(lines))):
            indices = lines[i].split()
            if len(indices) >= 3:
                faces.append([int(indices[0]) - 1, int(indices[1]) - 1, int(indices[2]) - 1])

        model.vertices = vertices
        model.faces = faces

        logger.info(
            "Parsed shape: %d vertices, %d faces (expected %d, %d)",
            len(vertices), len(faces), nv, nf
        )

    def _parse_spin(self, text: str, model: DAMITShapeModel) -> None:
        """
        Parse DAMIT spin.txt format.

        Typical content:
          lambda  beta  period  epoch  ...
          values...
        """
        lines = [l.strip() for l in text.strip().split("\n")
                 if l.strip() and not l.strip().startswith("#")]

        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    lam = float(parts[0])
                    bet = float(parts[1])
                    period = float(parts[2])

                    # Validate ranges
                    if 0 <= lam <= 360 and -90 <= bet <= 90 and period > 0:
                        model.spin.lambda_ecl_deg = lam
                        model.spin.beta_ecl_deg = bet
                        model.spin.period_hours = period
                        if len(parts) >= 4:
                            try:
                                model.spin.epoch_jd = float(parts[3])
                            except ValueError:
                                pass
                        break
                except ValueError:
                    continue
