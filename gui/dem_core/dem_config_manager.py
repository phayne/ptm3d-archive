"""
Configuration manager for the DEM/Mesh module.
Follows the same pattern as ConfigManager, ExoConfigManager, etc.

Uses dtm_to_ply_delaunay exclusively for DEM-to-mesh conversion.
"""

import copy
import yaml


class DEMConfigManager:
    """Manages DEM/Mesh processing configuration."""

    DEFAULT_CONFIG = {
        "input": {
            "dem_file": "",
            "body_radius_m": 1737400.0,  # Moon default
        },
        "roi": {
            "enabled": False,
            "center_lat": 0.0,
            "center_lon": 0.0,
            "radius_km": 10.0,
            "max_points": 500000,
        },
        "conversion": {
            "downsample": 10,
            "roi_core_frac": 0.15,
            "density_exp": 2.0,         # Density falloff exponent (1=linear, 2=quadratic)
            "max_edge_m": -1.0,         # -1 = auto-compute; 0 = disable filtering; >0 = threshold
            "south_pole_coords": False, # Apply south pole coordinate corrections
            "swap_axes": False,         # Swap X/Y axes in polar stereographic
            "output_file": "",
        },
        "quality_control": {
            "enabled": True,            # Post-processing QC sweep
            "sliver_angle_deg": 5.0,    # Minimum angle for sliver triangle removal
            "max_hole_edges": 20,       # Max boundary loop edges to fill (0=disable)
        },
        "refinement": {
            "enabled": False,
            "center_lat": 0.0,
            "center_lon": 0.0,
            "inner_radius_km": 1.0,
            "outer_radius_km": 20.0,
            "max_faces": 100000,
            "min_edge_m": 10.0,
            "max_edge_m": 1000.0,
            "initial_downsample": 10,
            "output_file": "",
        },
    }

    PRESETS = {
        "quick_preview": {
            "conversion": {"downsample": 50, "max_edge_m": 0},
            "quality_control": {"enabled": False},
            "roi": {"enabled": False},
            "refinement": {"enabled": False},
        },
        "standard": {
            "conversion": {"downsample": 10, "max_edge_m": -1.0},
            "quality_control": {"enabled": True},
            "roi": {"enabled": False},
            "refinement": {"enabled": False},
        },
        "high_res_roi": {
            "conversion": {"downsample": 5, "density_exp": 2.0, "max_edge_m": -1.0},
            "quality_control": {"enabled": True},
            "roi": {"enabled": True, "radius_km": 10.0, "max_points": 500000},
            "refinement": {"enabled": False},
        },
        "south_polar": {
            "conversion": {
                "downsample": 10,
                "south_pole_coords": True,
                "max_edge_m": -1.0,
            },
            "quality_control": {"enabled": True},
            "roi": {"enabled": True, "radius_km": 15.0, "max_points": 1000000},
            "refinement": {"enabled": False},
        },
        "adaptive_landing_site": {
            "conversion": {"downsample": 10, "density_exp": 1.5},
            "quality_control": {"enabled": True},
            "roi": {"enabled": True, "radius_km": 15.0, "max_points": 1000000},
            "refinement": {
                "enabled": True,
                "inner_radius_km": 1.0,
                "outer_radius_km": 20.0,
                "max_faces": 100000,
            },
        },
    }

    BODY_RADII = {
        "Moon": 1737400.0,
        "Mercury": 2439700.0,
        "Mars": 3389500.0,
        "Ceres": 473000.0,
        "Vesta": 262700.0,
        "Europa": 1560800.0,
        "Enceladus": 252100.0,
    }

    def __init__(self):
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)

    def get_config(self) -> dict:
        return copy.deepcopy(self._config)

    def get_value(self, key, default=None):
        """Get value using dot notation: 'input.dem_file'."""
        keys = key.split(".")
        obj = self._config
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj

    def set_value(self, key, value):
        """Set value using dot notation: 'input.dem_file'."""
        keys = key.split(".")
        obj = self._config
        for k in keys[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        obj[keys[-1]] = value

    def validate(self):
        """Validate configuration. Returns (is_valid, [messages])."""
        messages = []
        dem_file = self.get_value("input.dem_file", "")
        if not dem_file:
            messages.append("No DEM file specified")

        downsample = self.get_value("conversion.downsample", 1)
        if downsample < 1:
            messages.append("Downsample factor must be >= 1")

        output_file = self.get_value("conversion.output_file", "")
        if not output_file:
            messages.append("No output file specified")

        if self.get_value("roi.enabled", False):
            radius = self.get_value("roi.radius_km", 0)
            if radius <= 0:
                messages.append("ROI radius must be > 0")

        density_exp = self.get_value("conversion.density_exp", 2.0)
        if density_exp < 0.1:
            messages.append("Density exponent must be >= 0.1")

        if self.get_value("refinement.enabled", False):
            inner = self.get_value("refinement.inner_radius_km", 0)
            outer = self.get_value("refinement.outer_radius_km", 0)
            if inner >= outer:
                messages.append("Refinement inner radius must be < outer radius")
            max_faces = self.get_value("refinement.max_faces", 0)
            if max_faces < 100:
                messages.append("Max faces must be >= 100")

        is_valid = len(messages) == 0 or (
            len(messages) == 1 and messages[0] == "No DEM file specified"
        )
        return is_valid, messages

    def load_from_yaml(self, filename):
        """Load configuration from YAML file."""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        if data:
            self._deep_merge(self._config, data)

    def save_to_yaml(self, filename):
        """Save configuration to YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def load_preset(self, preset_name):
        """Load a preset configuration."""
        if preset_name in self.PRESETS:
            self._config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._deep_merge(self._config, self.PRESETS[preset_name])

    def reset_to_defaults(self):
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)

    def set_body(self, body_name):
        """Set body radius from name."""
        if body_name in self.BODY_RADII:
            self.set_value("input.body_radius_m", self.BODY_RADII[body_name])

    @staticmethod
    def _deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                DEMConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
