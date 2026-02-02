"""
Configuration Manager for Exosphere GUI.
Handles loading, saving, validating, and managing exosphere simulation configurations.
Maps 1:1 to ptm_exosphere_main CLI flags.
"""

import yaml
from copy import deepcopy
from pathlib import Path


class ExoConfigManager:
    """Manages exosphere simulation configuration."""

    DEFAULT_CONFIG = {
        "general": {
            "ptm_output_dir": "",
            "output_dir": "exosphere_output",
            "mesh_file": "",
            "timesteps": 144,
            "equatorial_ice_mm": 0.0,
            "polar_ice_mm": 0.0,
        },
        "sputtering": {
            "enabled": False,
            "rate_scale": 1.0,
            "hemisphere_contrast": 3.0,
        },
        "ice_layer": {
            "enabled": False,
            "dust_fraction": 0.05,
        },
        "crystallization": {
            "enabled": False,
            "initial_crystallinity": 0.0,
            "activation_energy": 62000.0,
            "amorphization_dose": 3.0,
            "equatorial_porosity": 0.5,
            "polar_porosity": 0.2,
            "dose_hemisphere_contrast": 5.0,
            "dose_polar_enhancement": 2.5,
            "base_dose_rate": 2.3e-6,
        },
        "multiscale": {
            "enabled": False,
            "epochs": 1000,
            "cycles_per_epoch": 100,
            "export_interval": 100,
            "analytical_acceleration": True,
            "cache_refresh": 0,
        },
    }

    PRESETS = {
        "sublimation_only": {
            "general": {"equatorial_ice_mm": 1.0, "polar_ice_mm": 10.0},
            "sputtering": {"enabled": False},
            "ice_layer": {"enabled": False},
            "crystallization": {"enabled": False},
            "multiscale": {"enabled": False},
        },
        "full_physics": {
            "general": {"equatorial_ice_mm": 1.0, "polar_ice_mm": 10.0},
            "sputtering": {"enabled": True, "rate_scale": 1.0, "hemisphere_contrast": 3.0},
            "ice_layer": {"enabled": True, "dust_fraction": 0.05},
            "crystallization": {"enabled": True},
            "multiscale": {"enabled": False},
        },
        "geological_evolution": {
            "general": {"equatorial_ice_mm": 1.0, "polar_ice_mm": 10.0},
            "sputtering": {"enabled": True},
            "ice_layer": {"enabled": True, "dust_fraction": 0.05},
            "crystallization": {"enabled": True},
            "multiscale": {
                "enabled": True, "epochs": 1000,
                "cycles_per_epoch": 100, "export_interval": 100,
            },
        },
    }

    def __init__(self):
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def get_config(self):
        """Get the current configuration."""
        return self.config

    def set_value(self, key, value):
        """Set a configuration value using dot notation (e.g., 'general.timesteps')."""
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

    def enforce_dependencies(self):
        """Enforce configuration dependencies.

        - crystallization.enabled -> ice_layer.enabled
        - multiscale.enabled -> ice_layer.enabled
        """
        if self.config["crystallization"]["enabled"]:
            self.config["ice_layer"]["enabled"] = True
        if self.config["multiscale"]["enabled"]:
            self.config["ice_layer"]["enabled"] = True

    def validate(self):
        """Validate the current configuration.

        Returns:
            tuple: (is_valid, list_of_messages)
        """
        errors = []
        warnings = []

        # Enforce dependencies before validating
        self.enforce_dependencies()

        gen = self.config["general"]

        # Required fields
        if not gen["ptm_output_dir"]:
            errors.append("PTM output directory is required (--ptm-output)")
        elif not Path(gen["ptm_output_dir"]).is_dir():
            errors.append(
                f"PTM output directory does not exist: {gen['ptm_output_dir']}"
            )

        if gen["timesteps"] <= 0:
            errors.append("Timesteps must be positive")

        # Ice thickness
        if gen["equatorial_ice_mm"] < 0:
            errors.append("Equatorial ice thickness cannot be negative")
        if gen["polar_ice_mm"] < 0:
            errors.append("Polar ice thickness cannot be negative")

        # Sputtering
        sput = self.config["sputtering"]
        if sput["enabled"]:
            if sput["rate_scale"] <= 0:
                errors.append("Sputtering rate scale must be positive")
            if sput["hemisphere_contrast"] <= 0:
                errors.append("Sputtering hemisphere contrast must be positive")

        # Ice layer
        ice = self.config["ice_layer"]
        if ice["enabled"]:
            if not (0 <= ice["dust_fraction"] <= 1):
                errors.append("Dust fraction must be between 0 and 1")

        # Crystallization
        cryst = self.config["crystallization"]
        if cryst["enabled"]:
            if not (0 <= cryst["initial_crystallinity"] <= 1):
                errors.append("Initial crystallinity must be between 0 and 1")
            if cryst["activation_energy"] <= 0:
                errors.append("Activation energy must be positive")
            if cryst["amorphization_dose"] <= 0:
                errors.append("Amorphization dose must be positive")
            if not (0 <= cryst["equatorial_porosity"] <= 1):
                errors.append("Equatorial porosity must be between 0 and 1")
            if not (0 <= cryst["polar_porosity"] <= 1):
                errors.append("Polar porosity must be between 0 and 1")
            if cryst["dose_hemisphere_contrast"] <= 0:
                errors.append("Dose hemisphere contrast must be positive")
            if cryst["dose_polar_enhancement"] <= 0:
                errors.append("Dose polar enhancement must be positive")
            if cryst["base_dose_rate"] <= 0:
                errors.append("Base dose rate must be positive")

        # Multi-timescale
        multi = self.config["multiscale"]
        if multi["enabled"]:
            if multi["epochs"] <= 0:
                errors.append("Number of epochs must be positive")
            if multi["cycles_per_epoch"] <= 0:
                errors.append("Cycles per epoch must be positive")
            if multi["export_interval"] < 0:
                errors.append("Export interval cannot be negative")
            if not ice["enabled"]:
                errors.append("Multi-timescale mode requires ice layer model")

        # Warnings
        if gen["equatorial_ice_mm"] == 0 and gen["polar_ice_mm"] == 0:
            if not sput["enabled"]:
                warnings.append(
                    "No initial ice and no sputtering: "
                    "only sublimation from surface temperatures will be simulated"
                )

        if gen.get("mesh_file") and not Path(gen["mesh_file"]).is_file():
            warnings.append(f"Mesh file not found: {gen['mesh_file']}")

        all_messages = errors + warnings
        return (len(errors) == 0, all_messages)

    def load_preset(self, preset_name):
        """Load a preset configuration."""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        preset = self.PRESETS[preset_name]
        self.reset_to_defaults()
        self.config = self._deep_merge(self.config, preset)

    def build_command_args(self):
        """Build CLI argument list from current config.

        Returns:
            list of str: Arguments suitable for subprocess.
        """
        args = []
        gen = self.config["general"]

        # Required
        if gen["ptm_output_dir"]:
            args.extend(["--ptm-output", gen["ptm_output_dir"]])
        if gen["output_dir"]:
            args.extend(["--output", gen["output_dir"]])
        args.extend(["--timesteps", str(gen["timesteps"])])

        # Optional general
        if gen.get("mesh_file"):
            args.extend(["--mesh", gen["mesh_file"]])
        if gen["equatorial_ice_mm"] > 0:
            args.extend(["--equatorial-ice", str(gen["equatorial_ice_mm"])])
        if gen["polar_ice_mm"] > 0:
            args.extend(["--polar-ice", str(gen["polar_ice_mm"])])

        # Sputtering
        sput = self.config["sputtering"]
        if sput["enabled"]:
            args.append("--sputtering")
            if sput["rate_scale"] != 1.0:
                args.extend(["--sputtering-rate", str(sput["rate_scale"])])
            if sput["hemisphere_contrast"] != 3.0:
                args.extend(
                    ["--sputtering-contrast", str(sput["hemisphere_contrast"])]
                )

        # Ice layer
        ice = self.config["ice_layer"]
        if ice["enabled"]:
            args.append("--ice-layer")
            if ice["dust_fraction"] != 0.05:
                args.extend(["--dust-fraction", str(ice["dust_fraction"])])

        # Crystallization
        cryst = self.config["crystallization"]
        if cryst["enabled"]:
            args.append("--crystallization")
            if cryst["initial_crystallinity"] != 0.0:
                args.extend(
                    ["--initial-crystallinity", str(cryst["initial_crystallinity"])]
                )
            if cryst["activation_energy"] != 62000.0:
                args.extend(
                    ["--activation-energy", str(cryst["activation_energy"])]
                )
            if cryst["amorphization_dose"] != 3.0:
                args.extend(
                    ["--amorphization-dose", str(cryst["amorphization_dose"])]
                )
            if cryst["equatorial_porosity"] != 0.5:
                args.extend(
                    ["--equatorial-porosity", str(cryst["equatorial_porosity"])]
                )
            if cryst["polar_porosity"] != 0.2:
                args.extend(["--polar-porosity", str(cryst["polar_porosity"])])
            if cryst["dose_hemisphere_contrast"] != 5.0:
                args.extend(
                    [
                        "--dose-hemisphere-contrast",
                        str(cryst["dose_hemisphere_contrast"]),
                    ]
                )
            if cryst["dose_polar_enhancement"] != 2.5:
                args.extend(
                    [
                        "--dose-polar-enhancement",
                        str(cryst["dose_polar_enhancement"]),
                    ]
                )
            if cryst["base_dose_rate"] != 2.3e-6:
                args.extend(["--base-dose-rate", str(cryst["base_dose_rate"])])

        # Multi-timescale
        multi = self.config["multiscale"]
        if multi["enabled"]:
            args.append("--multiscale")
            if multi["epochs"] != 1000:
                args.extend(["--epochs", str(multi["epochs"])])
            if multi["cycles_per_epoch"] != 100:
                args.extend(
                    ["--cycles-per-epoch", str(multi["cycles_per_epoch"])]
                )
            if multi["export_interval"] != 100:
                args.extend(
                    ["--export-interval", str(multi["export_interval"])]
                )
            if not multi["analytical_acceleration"]:
                args.append("--no-analytical")
            if multi["cache_refresh"] > 0:
                args.extend(["--cache-refresh", str(multi["cache_refresh"])])

        return args

    def compute_total_time(self):
        """Compute total simulated time for multi-timescale mode.

        Returns:
            dict with keys: total_days, total_years, epochs, cycles_per_epoch
            or None if multiscale is disabled.
        """
        multi = self.config["multiscale"]
        if not multi["enabled"]:
            return None
        europa_day = 3.5512  # days
        total_cycles = multi["epochs"] * multi["cycles_per_epoch"]
        total_days = total_cycles * europa_day
        total_years = total_days / 365.25
        return {
            "total_days": total_days,
            "total_years": total_years,
            "epochs": multi["epochs"],
            "cycles_per_epoch": multi["cycles_per_epoch"],
        }

    def _deep_merge(self, base, update):
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
