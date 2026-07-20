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

        # Compute SPICE NAIF ID for SPK queries
        # Convention: numbered asteroids = 2000000 + number, comets = 1000000 + number
        if naif_id:
            try:
                num = int(naif_id.rstrip(";").strip())
                if body_type == "comet":
                    gb["naif_id"] = str(1000000 + num)
                else:
                    # Asteroids and other small bodies
                    gb["naif_id"] = str(2000000 + num)
            except ValueError:
                gb["naif_id"] = naif_id  # Use as-is if not numeric

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
        """Estimate default thermophysical properties from physical params."""
        surface = {}

        # Albedo from SBDB if available
        if physical.albedo > 0:
            surface["albedo"] = round(physical.albedo, 3)
        else:
            surface["albedo"] = 0.1  # Default for rocky asteroid

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
