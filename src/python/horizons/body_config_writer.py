from __future__ import annotations

"""
Generate PTM3D-compatible YAML body configuration files.

Combines HORIZONS orbital elements, SBDB physical parameters, and
optional DAMIT shape data into the generic_body YAML format that the
C++ config_parser reads.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


def _jd_to_iso(jd: float) -> str:
    """Approximate Julian date to ISO 8601 string (good enough for config)."""
    # J2000.0 = JD 2451545.0 = 2000-01-01T12:00:00
    from datetime import datetime, timedelta
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    dt = j2000 + timedelta(days=(jd - 2451545.0))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class BodyConfigWriter:
    """Write PTM3D generic_body YAML configuration files."""

    def write_config(
        self,
        output_path: str,
        name: str,
        elements: "OrbitalElements",
        physical: "PhysicalParams",
        ephemeris_source: str = "keplerian",
        spk_file: str = "",
        metakernel_file: str = "",
        shape_mesh_file: str = "",
        shape_type: str = "spherical",
        semi_axes_km: Optional[tuple[float, float, float]] = None,
        surface_overrides: Optional[dict] = None,
        naif_id: str = "",
        body_type: str = "",
    ) -> str:
        """
        Write a complete generic_body YAML configuration.

        Parameters
        ----------
        output_path : str
            Path to write the YAML file.
        name : str
            Body name.
        elements : OrbitalElements
            Keplerian orbital elements.
        physical : PhysicalParams
            Physical parameters from SBDB.
        ephemeris_source : str
            "keplerian" or "spice".
        spk_file : str
            Path to SPK file (for SPICE mode).
        metakernel_file : str
            Path to meta-kernel (for SPICE mode).
        shape_mesh_file : str
            Path to OBJ mesh file (for custom_mesh shape).
        shape_type : str
            "spherical", "ellipsoidal", or "custom_mesh".
        semi_axes_km : tuple, optional
            (a, b, c) semi-axes for ellipsoidal shape.
        surface_overrides : dict, optional
            Override thermophysical properties (albedo, emissivity, etc.).

        Returns
        -------
        str
            Path to the written YAML file.
        """
        # Build the config dictionary
        config = {"generic_body": {}}
        gb = config["generic_body"]

        gb["name"] = name
        gb["ephemeris_source"] = ephemeris_source

        # SPICE NAIF ID for SPK queries.
        #
        # This MUST be the body id actually stored in the downloaded SPK, not
        # a value computed from the asteroid number.  HORIZONS has migrated
        # small-body SPK ids from the old "2000000 + number" scheme to the new
        # "20000000 + number" scheme, so any locally computed id is unreliable
        # and causes spkpos to fail with SPKINSUFFDATA.  The caller is expected
        # to pass the id read back from the SPK (see read_spk_body_id); we write
        # it through verbatim.
        if naif_id:
            gb["naif_id"] = str(naif_id).rstrip(";").strip()

        # Shape
        shape = {}
        shape["type"] = shape_type
        if physical.diameter_km > 0:
            shape["radius_km"] = round(physical.diameter_km / 2.0, 3)
        if semi_axes_km:
            shape["semi_axes_km"] = list(semi_axes_km)
        if shape_mesh_file:
            shape["mesh_file"] = shape_mesh_file
        gb["shape"] = shape

        # Orbit
        orbit = {}
        orbit["semi_major_axis_au"] = round(elements.semi_major_axis_au, 8)
        orbit["eccentricity"] = round(elements.eccentricity, 8)
        orbit["inclination_deg"] = round(elements.inclination_deg, 6)
        orbit["lon_ascending_node_deg"] = round(elements.lon_ascending_node_deg, 6)
        orbit["arg_perihelion_deg"] = round(elements.arg_perihelion_deg, 6)
        orbit["mean_anomaly_deg"] = round(elements.mean_anomaly_deg, 6)
        orbit["epoch"] = _jd_to_iso(elements.epoch_jd)
        gb["orbit"] = orbit

        # Spin
        spin = {}
        if physical.rotation_period_hours > 0:
            spin["period_hours"] = round(physical.rotation_period_hours, 6)
        else:
            spin["period_hours"] = 6.0  # Default for unknown
        if physical.pole_ra_deg >= 0:
            spin["pole_ra_deg"] = round(physical.pole_ra_deg, 3)
        if physical.pole_dec_deg >= -90 and physical.pole_dec_deg != -1.0:
            spin["pole_dec_deg"] = round(physical.pole_dec_deg, 3)
        spin["prime_meridian_deg"] = 0.0
        gb["spin"] = spin

        # Surface properties
        surface = self._default_surface(physical)
        if surface_overrides:
            surface.update(surface_overrides)
        gb["surface"] = surface

        # SPICE configuration
        if ephemeris_source == "spice":
            if metakernel_file:
                gb["spice_metakernel"] = metakernel_file
            if spk_file:
                gb["spk_file"] = spk_file

        # Write YAML
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if yaml is not None:
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            # Fallback: write manually formatted YAML
            self._write_yaml_manual(output_path, config)

        logger.info("Wrote body config: %s", output_path)
        return output_path

    def _default_surface(self, physical: "PhysicalParams") -> dict:
        """Estimate default thermophysical properties from physical params.

        WARNING, and it is emitted at runtime rather than only written here:
        ``physical.albedo`` is SBDB's GEOMETRIC albedo p_V, and
        ``generic_body.surface.albedo`` is a BOND albedo -- the engine computes
        ``1 - albedo`` and scales the solar constant by it
        (``thermal_model.cpp:2906``).  The two differ by the phase integral
        ``q = 0.290 + 0.684 G``: for a bright body (p_V 0.42, G 0.32) the Bond
        albedo is 0.214, and writing 0.42 instead absorbs 26 percent less
        sunlight and runs about 7 percent cold, with nothing downstream to
        catch it.

        This is NOT silently converted here, because every ``body_config.yaml``
        already in ``data/horizons_cache`` was written by this code path and
        converting now would change what those cached bodies mean without
        rewriting them.  It warns instead, and the GUI's Asteroid card does the
        conversion at the point where it can label it
        (``gui_core.body_properties.bond_albedo``).  Converting here, and
        rewriting the cached configurations that were written before it, is
        owed as its own change.
        """
        surface = {}

        # Albedo from SBDB if available
        if physical.albedo > 0:
            surface["albedo"] = round(physical.albedo, 3)
            logger.warning(
                "generic_body.surface.albedo is being written as %.3f, which "
                "is SBDB's GEOMETRIC albedo p_V, into a field the engine reads "
                "as a BOND albedo. They are not the same: A_B = p_V x "
                "(0.290 + 0.684 G). This body will absorb too little sunlight "
                "and run cold -- by 26%% in flux for a bright object. Convert "
                "it, or set the albedo by hand.",
                physical.albedo)
        else:
            surface["albedo"] = 0.1  # Default for rocky asteroid
            logger.warning(
                "No albedo was published for this body, so "
                "generic_body.surface.albedo is a PLACEHOLDER of 0.1, not a "
                "measurement. Set it before trusting a temperature.")

        surface["emissivity"] = 0.95

        # Estimate thermal properties based on spectral type
        spec = physical.spectral_type.upper() if physical.spectral_type else ""
        if spec.startswith("C") or spec.startswith("B"):
            # Carbonaceous
            surface["ks"] = 0.005
            surface["kd"] = 0.1
            surface["rhos"] = 1300.0
            surface["rhod"] = 1800.0
        elif spec.startswith("S") or spec.startswith("V"):
            # Silicaceous/Vestoid
            surface["ks"] = 0.01
            surface["kd"] = 0.5
            surface["rhos"] = 1500.0
            surface["rhod"] = 2500.0
        else:
            # Generic defaults
            surface["ks"] = 0.01
            surface["kd"] = 0.5
            surface["rhos"] = 1500.0
            surface["rhod"] = 2000.0

        surface["H"] = 0.05
        surface["cp0"] = 800.0
        surface["Qb"] = 0.0

        return surface

    def _write_yaml_manual(self, path: str, config: dict) -> None:
        """Write YAML without PyYAML dependency (simple nested dict)."""
        lines = []

        def _write_dict(d: dict, indent: int = 0) -> None:
            prefix = "  " * indent
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    _write_dict(v, indent + 1)
                elif isinstance(v, list):
                    lines.append(f"{prefix}{k}: [{', '.join(str(x) for x in v)}]")
                elif isinstance(v, str):
                    lines.append(f'{prefix}{k}: "{v}"')
                elif isinstance(v, float):
                    lines.append(f"{prefix}{k}: {v}")
                elif isinstance(v, int):
                    lines.append(f"{prefix}{k}: {v}")
                else:
                    lines.append(f"{prefix}{k}: {v}")

        _write_dict(config)

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
