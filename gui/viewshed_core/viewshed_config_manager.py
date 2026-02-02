"""
Configuration Manager for Viewshed GUI.
Handles loading, saving, validating, and managing viewshed simulation configurations.
"""

import yaml
from copy import deepcopy
from pathlib import Path


class ViewshedConfigManager:
    """Manages viewshed simulation configuration."""

    DEFAULT_CONFIG = {
        "input": {
            "ptm_output_dir": "",
            "mesh_file": "",
            "waypoints_csv": "",
        },
        "mission": {
            "start_utc": "2030-10-01",
            "end_utc": "2030-10-14",
            "body_radius_m": 1737400.0,
        },
        "viewshed": {
            "max_emission_angle_deg": 85.0,
            "max_range_km": 100.0,
        },
        "ice": {
            "temperature_threshold_K": 110.0,
            "ice_fraction": 0.20,
            "gradient_scale_K": 0.0,
        },
        "output": {
            "output_dir": "",
            "write_truth_map": True,
            "write_per_waypoint": True,
        },
        "masks": [],
    }

    def __init__(self):
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def get_config(self):
        """Get the current configuration."""
        return self.config

    def set_value(self, key, value):
        """Set a configuration value using dot notation (e.g., 'mission.start_utc')."""
        keys = key.split(".")
        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get_value(self, key, default=None):
        """Get a configuration value using dot notation."""
        keys = key.split(".")
        current = self.config
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def load_from_yaml(self, filename):
        """Load configuration from YAML file."""
        with open(filename, "r") as f:
            loaded = yaml.safe_load(f)
        self.config = self._deep_merge(deepcopy(self.DEFAULT_CONFIG), loaded or {})

    def save_to_yaml(self, filename):
        """Save configuration to YAML file."""
        with open(filename, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def generate_yaml(self):
        """Generate YAML string for preview."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

    def validate(self):
        """Validate the current configuration.

        Returns:
            tuple: (is_valid, list_of_messages)
        """
        errors = []
        warnings = []

        inp = self.config["input"]
        mission = self.config["mission"]
        vs = self.config["viewshed"]
        ice = self.config["ice"]
        out = self.config["output"]

        # Required: PTM output directory
        if not inp["ptm_output_dir"]:
            errors.append("PTM output directory is required.")
        elif not Path(inp["ptm_output_dir"]).is_dir():
            errors.append(
                f"PTM output directory does not exist: {inp['ptm_output_dir']}"
            )

        # Required: waypoints CSV
        if not inp["waypoints_csv"]:
            errors.append("Waypoints CSV file is required.")
        elif not Path(inp["waypoints_csv"]).is_file():
            errors.append(
                f"Waypoints CSV file not found: {inp['waypoints_csv']}"
            )

        # Required: output directory
        if not out["output_dir"]:
            errors.append("Output directory is required.")

        # Optional mesh file
        if inp["mesh_file"] and not Path(inp["mesh_file"]).is_file():
            warnings.append(f"Mesh file not found: {inp['mesh_file']}")

        # Validate mesh.ply in PTM dir if no override
        if inp["ptm_output_dir"] and not inp["mesh_file"]:
            default_mesh = Path(inp["ptm_output_dir"]) / "mesh.ply"
            if not default_mesh.is_file():
                warnings.append(
                    f"Default mesh not found: {default_mesh}"
                )

        # Mission dates
        if not mission["start_utc"]:
            errors.append("Mission start date is required.")
        if not mission["end_utc"]:
            errors.append("Mission end date is required.")

        # Viewshed parameters
        if not (0 < vs["max_emission_angle_deg"] <= 90):
            errors.append("Max emission angle must be between 0 and 90 degrees.")
        if vs["max_range_km"] <= 0:
            errors.append("Max range must be positive.")

        # Ice parameters
        if ice["temperature_threshold_K"] <= 0:
            errors.append("Temperature threshold must be positive.")
        if not (0 <= ice["ice_fraction"] <= 1):
            errors.append("Ice fraction must be between 0 and 1.")
        if ice["gradient_scale_K"] < 0:
            errors.append("Gradient scale cannot be negative.")

        # Body radius
        if mission["body_radius_m"] <= 0:
            errors.append("Body radius must be positive.")

        all_messages = errors + warnings
        return (len(errors) == 0, all_messages)

    def _deep_merge(self, base, update):
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
