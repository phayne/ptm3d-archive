from __future__ import annotations

"""
Data cache manager for PTM3D HORIZONS/DAMIT integration.

Manages local storage of downloaded ephemeris files, shape models,
and generated body configuration YAML files.

Cache structure:
    data/horizons_cache/
        index.json              # Master index of cached bodies
        bodies/<name>/
            body_config.yaml    # Complete generic_body config for C++
            ephemeris.spk       # Binary SPK file from HORIZONS
            metakernel.tm       # Auto-generated meta-kernel
            shape.obj           # DAMIT shape model (optional)
            horizons_raw.json   # Raw API data (for debugging)
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedBody:
    """Metadata for a cached body."""
    name: str
    naif_id: str = ""
    body_type: str = ""
    cache_date: str = ""
    config_path: str = ""
    spk_path: str = ""
    metakernel_path: str = ""
    shape_path: str = ""
    spk_start: str = ""
    spk_stop: str = ""
    has_damit_shape: bool = False
    damit_id: int = 0


class DataManager:
    """Manage the local cache of HORIZONS/DAMIT data."""

    def __init__(self, cache_dir: str = ""):
        """
        Parameters
        ----------
        cache_dir : str
            Path to the cache directory. Defaults to 'data/horizons_cache'
            relative to the project root.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Auto-detect project root (look for CMakeLists.txt)
            here = Path(__file__).resolve().parent
            for parent in [here] + list(here.parents):
                if (parent / "CMakeLists.txt").exists():
                    self.cache_dir = parent / "data" / "horizons_cache"
                    break
            else:
                self.cache_dir = Path("data") / "horizons_cache"

        self.bodies_dir = self.cache_dir / "bodies"
        self.index_path = self.cache_dir / "index.json"

    def initialize(self) -> None:
        """Create cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index({})

    def _load_index(self) -> dict:
        """Load the master index."""
        if not self.index_path.exists():
            return {}
        with open(self.index_path) as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        """Save the master index."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _body_dir(self, name: str) -> Path:
        """Get the cache directory for a specific body."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        return self.bodies_dir / safe_name

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_cached(self, name: str) -> bool:
        """Check if a body is in the cache."""
        index = self._load_index()
        return name.lower() in {k.lower() for k in index}

    def get_config_path(self, name: str) -> Optional[str]:
        """Get the path to a cached body's YAML config file."""
        index = self._load_index()
        for k, v in index.items():
            if k.lower() == name.lower():
                config_path = v.get("config_path", "")
                if config_path and os.path.exists(config_path):
                    return config_path
        return None

    def get_cached_body(self, name: str) -> Optional[CachedBody]:
        """Get full cache metadata for a body."""
        index = self._load_index()
        for k, v in index.items():
            if k.lower() == name.lower():
                return CachedBody(
                    name=k,
                    naif_id=v.get("naif_id", ""),
                    body_type=v.get("body_type", ""),
                    cache_date=v.get("cache_date", ""),
                    config_path=v.get("config_path", ""),
                    spk_path=v.get("spk_path", ""),
                    metakernel_path=v.get("metakernel_path", ""),
                    shape_path=v.get("shape_path", ""),
                    spk_start=v.get("spk_start", ""),
                    spk_stop=v.get("spk_stop", ""),
                    has_damit_shape=v.get("has_damit_shape", False),
                    damit_id=v.get("damit_id", 0),
                )
        return None

    def list_cached(self) -> list[CachedBody]:
        """List all cached bodies."""
        index = self._load_index()
        return [
            CachedBody(
                name=k,
                naif_id=v.get("naif_id", ""),
                body_type=v.get("body_type", ""),
                cache_date=v.get("cache_date", ""),
                config_path=v.get("config_path", ""),
                has_damit_shape=v.get("has_damit_shape", False),
            )
            for k, v in index.items()
        ]

    # ------------------------------------------------------------------
    # Store methods
    # ------------------------------------------------------------------

    def store_body(
        self,
        name: str,
        naif_id: str = "",
        body_type: str = "",
        config_yaml: str = "",
        spk_data: Optional[bytes] = None,
        metakernel_content: str = "",
        shape_obj: str = "",
        horizons_raw: Optional[dict] = None,
        spk_start: str = "",
        spk_stop: str = "",
        damit_id: int = 0,
    ) -> CachedBody:
        """
        Store a body's data in the cache.

        Parameters
        ----------
        name : str
            Body name (used as cache key).
        naif_id : str
            NAIF/SPK ID.
        body_type : str
            Body type ("asteroid", "comet", etc.).
        config_yaml : str
            Content of the body_config.yaml file.
        spk_data : bytes, optional
            Binary SPK file content.
        metakernel_content : str
            Content of the .tm meta-kernel.
        shape_obj : str
            Content of the OBJ shape file.
        horizons_raw : dict, optional
            Raw API response data for debugging.
        spk_start : str
            SPK coverage start time.
        spk_stop : str
            SPK coverage stop time.
        damit_id : int
            DAMIT database ID.

        Returns
        -------
        CachedBody
            Metadata for the stored body.
        """
        body_dir = self._body_dir(name)
        body_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "naif_id": naif_id,
            "body_type": body_type,
            "cache_date": datetime.now().isoformat(),
            "spk_start": spk_start,
            "spk_stop": spk_stop,
            "has_damit_shape": bool(shape_obj),
            "damit_id": damit_id,
        }

        # Write config YAML
        if config_yaml:
            config_path = str(body_dir / "body_config.yaml")
            with open(config_path, "w") as f:
                f.write(config_yaml)
            entry["config_path"] = config_path

        # Write SPK
        if spk_data:
            spk_path = str(body_dir / "ephemeris.spk")
            with open(spk_path, "wb") as f:
                f.write(spk_data)
            entry["spk_path"] = spk_path

        # Write meta-kernel
        if metakernel_content:
            mk_path = str(body_dir / "metakernel.tm")
            with open(mk_path, "w") as f:
                f.write(metakernel_content)
            entry["metakernel_path"] = mk_path

        # Write shape OBJ
        if shape_obj:
            shape_path = str(body_dir / "shape.obj")
            with open(shape_path, "w") as f:
                f.write(shape_obj)
            entry["shape_path"] = shape_path

        # Write raw HORIZONS data
        if horizons_raw:
            raw_path = str(body_dir / "horizons_raw.json")
            with open(raw_path, "w") as f:
                json.dump(horizons_raw, f, indent=2)

        # Update index
        index = self._load_index()
        index[name] = entry
        self._save_index(index)

        logger.info("Cached body '%s' at %s", name, body_dir)

        return CachedBody(name=name, **{k: v for k, v in entry.items() if k != "name"})

    def clear_cache(self, name: str) -> bool:
        """Remove a body from the cache."""
        index = self._load_index()
        key_to_remove = None
        for k in index:
            if k.lower() == name.lower():
                key_to_remove = k
                break

        if key_to_remove is None:
            return False

        body_dir = self._body_dir(key_to_remove)
        if body_dir.exists():
            shutil.rmtree(body_dir)

        del index[key_to_remove]
        self._save_index(index)

        logger.info("Removed '%s' from cache", name)
        return True

    def clear_all(self) -> int:
        """Remove all cached bodies. Returns count of removed entries."""
        index = self._load_index()
        count = len(index)

        if self.bodies_dir.exists():
            shutil.rmtree(self.bodies_dir)
            self.bodies_dir.mkdir(parents=True, exist_ok=True)

        self._save_index({})
        logger.info("Cleared %d cached bodies", count)
        return count

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_spk_coverage(
        self, name: str, sim_start: str, sim_stop: str
    ) -> bool:
        """
        Check if the cached SPK covers the requested simulation window.

        Parameters
        ----------
        name : str
            Body name.
        sim_start : str
            Simulation start time (ISO format).
        sim_stop : str
            Simulation stop time (ISO format).

        Returns
        -------
        bool
            True if SPK coverage includes the simulation window.
        """
        cached = self.get_cached_body(name)
        if not cached or not cached.spk_start or not cached.spk_stop:
            return False

        try:
            spk_start = datetime.fromisoformat(cached.spk_start)
            spk_stop = datetime.fromisoformat(cached.spk_stop)
            req_start = datetime.fromisoformat(sim_start)
            req_stop = datetime.fromisoformat(sim_stop)

            return spk_start <= req_start and spk_stop >= req_stop
        except ValueError:
            return False

    def check_offline_ready(self, name: str) -> dict[str, bool]:
        """
        Verify all local files exist for offline operation.

        Returns a dict of file_type -> exists_bool.
        """
        cached = self.get_cached_body(name)
        if not cached:
            return {"config": False, "spk": False, "metakernel": False, "shape": False}

        return {
            "config": bool(cached.config_path) and os.path.exists(cached.config_path),
            "spk": bool(cached.spk_path) and os.path.exists(cached.spk_path),
            "metakernel": bool(cached.metakernel_path) and os.path.exists(cached.metakernel_path),
            "shape": bool(cached.shape_path) and os.path.exists(cached.shape_path),
        }


def read_spk_body_id(
    spk_path: str, fallback_naif_id: str = ""
) -> Optional[str]:
    """Read the body id actually stored in a downloaded SPK file.

    HORIZONS writes small bodies into SPK files under either the old
    ``2000000 + number`` or the new ``20000000 + number`` convention, and the
    two are not interchangeable: spkpos/spkez fail with SPKINSUFFDATA if you
    query the wrong one.  Rather than guess, read the segment summaries from
    the SPK itself and return the body id that is genuinely present.

    Robust to both conventions: whatever id the SPK declares is returned as-is.
    The body (target) ids in HORIZONS small-body SPKs are large positive
    integers; the only other id present is normally the Sun/SSB center
    (0 or 10), which we ignore.

    Parameters
    ----------
    spk_path : str
        Path to the binary SPK file.
    fallback_naif_id : str
        Value to return if the SPK cannot be read (e.g. SPICE unavailable).

    Returns
    -------
    str or None
        The body id as a string, or ``fallback_naif_id`` (possibly empty) if
        it could not be determined.
    """
    if not spk_path or not os.path.exists(spk_path):
        return fallback_naif_id or None

    ids: list[int] = []
    try:
        import spiceypy as sp

        cell = sp.spkobj(spk_path)
        ids = [cell[i] for i in range(len(cell))]
    except Exception:
        # Fall back to a minimal DAF traversal if spiceypy is unavailable or
        # errors out.  This reads SPK segment summaries directly: each segment
        # descriptor stores the target body id as the third integer component.
        try:
            ids = _daf_spk_body_ids(spk_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read body id from SPK %s: %s", spk_path, exc)
            return fallback_naif_id or None

    # Ignore generic ephemeris centers (Solar System Barycenter / Sun).
    body_ids = [i for i in ids if i not in (0, 10)]
    if not body_ids:
        body_ids = ids

    if not body_ids:
        return fallback_naif_id or None

    if len(set(body_ids)) > 1:
        logger.warning(
            "SPK %s contains multiple body ids %s; using %s",
            spk_path, sorted(set(body_ids)), max(body_ids),
        )

    # Prefer the largest id: HORIZONS small-body ids dwarf any incidental
    # planetary center that might appear.
    return str(max(body_ids))


def _daf_spk_body_ids(spk_path: str) -> list[int]:
    """Read SPK target body ids by walking the DAF structure directly.

    This is a dependency-free fallback for read_spk_body_id used only when
    spiceypy is not importable.  It parses the DAF file/summary records and
    extracts the third integer component of each segment descriptor, which for
    SPK files is the target body id.
    """
    import struct

    with open(spk_path, "rb") as f:
        file_record = f.read(1024)
        if not file_record.startswith(b"DAF/SPK"):
            raise ValueError("not a DAF/SPK file")

        # Endianness marker lives at bytes 88..96 ("LTL-IEEE" / "BIG-IEEE").
        bff = file_record[88:96].rstrip(b"\x00 ").decode("ascii", "replace")
        endian = "<" if bff.startswith("LTL") else ">"

        nd = struct.unpack(endian + "i", file_record[8:12])[0]
        ni = struct.unpack(endian + "i", file_record[12:16])[0]
        fward = struct.unpack(endian + "i", file_record[76:80])[0]

        # Number of 8-byte words per descriptor (doubles + packed ints).
        ss = nd + (ni + 1) // 2

        body_ids: list[int] = []
        record_no = fward
        while record_no != 0:
            f.seek((record_no - 1) * 1024)
            rec = f.read(1024)
            next_rec = struct.unpack(endian + "d", rec[0:8])[0]
            nsum = int(struct.unpack(endian + "d", rec[16:24])[0])
            offset = 24
            for _ in range(nsum):
                desc = rec[offset : offset + ss * 8]
                offset += ss * 8
                # Integer components follow the nd doubles; the SPK descriptor
                # integers are [target, center, frame, type, baddr, eaddr].
                int_off = nd * 8
                target = struct.unpack(
                    endian + "i", desc[int_off : int_off + 4]
                )[0]
                body_ids.append(target)
            record_no = int(next_rec)

    return body_ids


def fetch_and_cache_body(
    query: str,
    cache_dir: str = "",
    spk_start: str = "2020-01-01",
    spk_stop: str = "2030-01-01",
    fetch_damit: bool = True,
    ephemeris_source: str = "keplerian",
    kernels_base_dir: str = "",
    equilibration_orbits: int = 10,
) -> Optional[str]:
    """
    High-level convenience function: search, download, and cache a body.

    Parameters
    ----------
    query : str
        Body name or number to search in HORIZONS.
    cache_dir : str
        Cache directory path.
    spk_start : str
        SPK ephemeris start date.
    spk_stop : str
        SPK ephemeris end date.
    fetch_damit : bool
        Whether to search DAMIT for a shape model.
    ephemeris_source : str
        "keplerian" or "spice".
    kernels_base_dir : str
        Path to SPICE kernels directory.
    equilibration_orbits : int
        Number of day/night equilibration cycles. The SPK start date
        will be extended backward to cover the equilibration period.

    Returns
    -------
    str or None
        Path to the generated body_config.yaml, or None on failure.
    """
    from horizons import HorizonsClient
    from horizons.spk_generator import SPKGenerator
    from horizons.body_config_writer import BodyConfigWriter

    dm = DataManager(cache_dir)
    dm.initialize()

    # Check cache first
    if dm.is_cached(query):
        config_path = dm.get_config_path(query)
        if config_path:
            logger.info("Using cached config for '%s': %s", query, config_path)
            return config_path

    # Fetch from HORIZONS
    client = HorizonsClient()
    print(f"Searching HORIZONS for '{query}'...")
    body = client.get_body(query)

    if not body.elements.semi_major_axis_au:
        print(f"Could not get orbital elements for '{query}'")
        return None

    print(f"Found: {body.name} (NAIF ID: {body.naif_id})")
    print(f"  a = {body.elements.semi_major_axis_au:.4f} AU, "
          f"e = {body.elements.eccentricity:.4f}, "
          f"P = {body.elements.orbital_period_days:.1f} days")
    if body.physical.diameter_km > 0:
        print(f"  Diameter: {body.physical.diameter_km:.1f} km")
    if body.physical.rotation_period_hours > 0:
        print(f"  Rotation period: {body.physical.rotation_period_hours:.3f} hours")

    # Extend SPK start backward to cover the equilibration period.
    # The C++ simulation pushes start_time backward by
    # equilibration_orbits * synodic_day (≈ rotation period for asteroids).
    if ephemeris_source == "spice" and equilibration_orbits > 0:
        rot_hours = body.physical.rotation_period_hours
        if rot_hours > 0:
            rot_days = rot_hours / 24.0
            # Add 20% margin beyond the equilibration period
            eq_days = equilibration_orbits * rot_days * 1.2
            from datetime import datetime, timedelta
            try:
                start_dt = datetime.fromisoformat(
                    spk_start.replace("T", " ").split(".")[0]
                )
                adjusted = start_dt - timedelta(days=eq_days)
                spk_start = adjusted.strftime("%Y-%m-%d")
                print(f"  SPK start extended to {spk_start} "
                      f"({eq_days:.1f} days before sim start "
                      f"for {equilibration_orbits} equil. cycles "
                      f"x {rot_hours:.1f}h rotation)")
            except Exception:
                pass  # keep original spk_start

    # Download SPK if using SPICE ephemeris
    spk_data = None
    spk_path = ""
    # NAIF id used for SPK queries.  For SPICE mode this is overwritten with
    # the id actually found inside the downloaded SPK (see read_spk_body_id),
    # so it is robust to HORIZONS' old (2000000+N) vs new (20000000+N) scheme.
    spk_naif_id = body.naif_id
    if ephemeris_source == "spice":
        try:
            print(f"Downloading SPK ephemeris ({spk_start} to {spk_stop})...")
            body_dir = dm._body_dir(body.name)
            body_dir.mkdir(parents=True, exist_ok=True)
            spk_path = str(body_dir / "ephemeris.spk")
            # For asteroids/comets, append ";" so HORIZONS interprets the
            # ID as a small-body record number (not a major-body NAIF ID).
            spk_cmd = body.naif_id or query
            if body.body_type in ("asteroid", "comet"):
                spk_cmd = spk_cmd.rstrip(";") + ";"
            client.download_spk(spk_cmd, spk_start, spk_stop, spk_path)
            with open(spk_path, "rb") as f:
                spk_data = f.read()
            print(f"  SPK downloaded: {len(spk_data)} bytes")

            # Read the body id actually stored in the SPK.  Computing it from
            # the asteroid number is unreliable because HORIZONS migrated
            # small-body SPK ids from 2000000+N to 20000000+N.
            spk_body_id = read_spk_body_id(spk_path, fallback_naif_id=body.naif_id)
            if spk_body_id:
                spk_naif_id = spk_body_id
                print(f"  SPK body id (read from file): {spk_naif_id}")
            else:
                print(f"  WARNING: could not read body id from SPK; "
                      f"using HORIZONS record number {body.naif_id}")
        except RuntimeError as e:
            print(f"  SPK download failed: {e}")
            print("  Falling back to Keplerian ephemeris")
            ephemeris_source = "keplerian"

    # Fetch DAMIT shape if requested
    shape_obj_content = ""
    shape_path = ""
    damit_id = 0
    if fetch_damit:
        try:
            from damit import DAMITClient, ShapeConverter

            print(f"Searching DAMIT for shape model...")
            damit = DAMITClient(verify_ssl=False)
            # Use the body name from HORIZONS (cleaner than raw query)
            damit_query = body.name if body.name else query
            results = damit.search_asteroid(damit_query)

            # Extract asteroid number from HORIZONS body for validation
            _body_number = None
            if body.naif_id:
                try:
                    _body_number = int(body.naif_id.rstrip(";").strip())
                except ValueError:
                    pass

            if results:
                # Verify the DAMIT result matches the queried body
                if _body_number is not None and results[0].get("number") != _body_number:
                    print(f"  DAMIT result mismatch: got ({results[0].get('number')}) "
                          f"{results[0].get('name', '')}, expected ({_body_number}). "
                          f"Skipping DAMIT shape.")
                    results = []

            if results:
                print(f"  Found {len(results)} DAMIT model(s)")
                damit_id = results[0]["damit_id"]
                print(f"  Using DAMIT asteroid ID {damit_id}: "
                      f"({results[0].get('number', '?')}) "
                      f"{results[0].get('name', '')}")
                model = damit.get_model(damit_id)

                if model.vertices:
                    radius_km = body.physical.diameter_km / 2 if body.physical.diameter_km > 0 else 0
                    body_dir = dm._body_dir(body.name)
                    body_dir.mkdir(parents=True, exist_ok=True)
                    shape_path = str(body_dir / "shape.obj")

                    ShapeConverter.damit_to_obj(
                        model.vertices, model.faces, shape_path,
                        radius_km=radius_km, body_name=body.name
                    )
                    with open(shape_path) as f:
                        shape_obj_content = f.read()
                    print(f"  Shape model: {model.num_vertices} vertices, "
                          f"{model.num_faces} faces")

                    # Use DAMIT spin params if available
                    if model.spin.period_hours > 0 and body.physical.rotation_period_hours == 0:
                        body.physical.rotation_period_hours = model.spin.period_hours
                    if model.spin.pole_ra_deg != 0:
                        body.physical.pole_ra_deg = model.spin.pole_ra_deg
                        body.physical.pole_dec_deg = model.spin.pole_dec_deg
                else:
                    print("  No shape data in DAMIT model")
            else:
                print("  No DAMIT models found")
        except Exception as e:
            print(f"  DAMIT fetch failed: {e}")

    # Generate meta-kernel
    metakernel_content = ""
    if ephemeris_source == "spice" and spk_path:
        spk_gen = SPKGenerator(kernels_base_dir)
        body_dir = dm._body_dir(body.name)
        mk_path = str(body_dir / "metakernel.tm")
        spk_gen.create_metakernel(mk_path, spk_path, body.name)
        with open(mk_path) as f:
            metakernel_content = f.read()

    # Generate body config YAML
    writer = BodyConfigWriter()
    body_dir = dm._body_dir(body.name)
    body_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(body_dir / "body_config.yaml")

    shape_type = "spherical"
    if shape_path:
        shape_type = "custom_mesh"

    mk_path_for_config = str(body_dir / "metakernel.tm") if metakernel_content else ""

    writer.write_config(
        output_path=config_path,
        name=body.name,
        elements=body.elements,
        physical=body.physical,
        ephemeris_source=ephemeris_source,
        spk_file=spk_path,
        metakernel_file=mk_path_for_config,
        shape_mesh_file=shape_path,
        shape_type=shape_type,
        naif_id=spk_naif_id,
        body_type=body.body_type,
    )

    # Read back the config content for caching
    with open(config_path) as f:
        config_yaml = f.read()

    # Store in cache
    dm.store_body(
        name=body.name,
        naif_id=spk_naif_id,
        body_type=body.body_type,
        config_yaml=config_yaml,
        spk_data=spk_data,
        metakernel_content=metakernel_content,
        shape_obj=shape_obj_content,
        spk_start=spk_start,
        spk_stop=spk_stop,
        damit_id=damit_id,
    )

    print(f"\nBody config written to: {config_path}")
    print(f"Use with PTM3D: --generic-body-config {config_path}")

    return config_path
