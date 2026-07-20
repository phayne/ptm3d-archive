"""
Configuration Manager for PTM3D GUI
Handles loading, saving, validating, and managing simulation configurations
"""

import yaml
from copy import deepcopy
from pathlib import Path


def detect_hardware_capabilities():
    """Detect GPU (OpenCL) and CPU SIMD capabilities.

    Returns:
        dict with keys: opencl_available, opencl_devices, avx2_available, avx512_available
    """
    capabilities = {
        "opencl_available": False,
        "opencl_devices": [],
        "avx2_available": False,
        "avx512_available": False
    }

    # Check OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for dev in devices:
                    mem_gb = dev.global_mem_size / (1024**3)
                    capabilities["opencl_devices"].append({
                        "name": dev.name.strip(),
                        "vendor": dev.vendor.strip(),
                        "memory_gb": round(mem_gb, 1),
                        "compute_units": dev.max_compute_units
                    })
            except cl.RuntimeError:
                pass

            try:
                cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                for dev in cpu_devices:
                    capabilities["opencl_devices"].append({
                        "name": dev.name.strip() + " (CPU)",
                        "vendor": dev.vendor.strip(),
                        "memory_gb": 0,
                        "compute_units": dev.max_compute_units
                    })
            except cl.RuntimeError:
                pass

        capabilities["opencl_available"] = len(capabilities["opencl_devices"]) > 0
    except ImportError:
        pass
    except Exception:
        pass

    # Check CPU SIMD capabilities
    try:
        import platform
        import subprocess

        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5
            )
            features = result.stdout.upper()
            capabilities["avx2_available"] = "AVX2" in features
            capabilities["avx512_available"] = "AVX512" in features or "AVX512F" in features
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().upper()
            capabilities["avx2_available"] = "AVX2" in cpuinfo
            capabilities["avx512_available"] = "AVX512" in cpuinfo or "AVX512F" in cpuinfo
        elif platform.system() == "Windows":
            capabilities["avx2_available"] = True
    except Exception:
        pass

    return capabilities


class ConfigManager:
    """Manages PTM3D simulation configuration"""

    DEFAULT_CONFIG = {
        "simulation": {
            "start_time": "2025-01-01T00:00:00",
            "duration_days": 3.55,
            "timesteps": 144,
            "body": "Europa"
        },
        "mesh": {
            "resolution": 5,
            "import_file": "",
            "scale_factor": 1.0,
        },
        "thermal": {
            "run_thermal_mesh": True,
            "threads": 0,
            "equilibration_orbits": 1.0,
            # Equilibration cadence (optional). Priority:
            # 1) equilibration_steps_per_day > 0 -> explicit override (steps/Earth-day)
            # 2) equilibration_samples_per_diurnal_cycle > 0 -> accuracy-based auto
            # 3) both <=0 -> legacy (match output cadence)
            # GUI default uses samples_per_diurnal_cycle=120 ("Balanced" tier, ~4 K max
            # accuracy on Moon ico3 reference test).
            "equilibration_steps_per_day": -1.0,
            "equilibration_samples_per_diurnal_cycle": 120.0,
            "use_gpu": False,
            "compute_statistics": True,
            # Per-face thermal property maps. The GUI presents these via two
            # synthetic fields (property_map_mode, property_map_kind) that map
            # onto the underlying YAML keys at save time:
            #   mode 0 (None)    -> use_default_property_map=False, both file paths empty
            #   mode 1 (Bundled) -> use_default_property_map=True
            #   mode 2 (Custom)  + kind 0 (H)  -> h_parameter_map = <path>
            #                    + kind 1 (TI) -> thermal_inertia_map = <path>
            "property_map_mode": 0,
            "property_map_kind": 0,
            "thermal_inertia_map": "",
            "h_parameter_map": "",
            "use_default_property_map": False,
            "property_maps_directory": "data/property_maps",
            "enable_shadow_flux": False,
            "crater_dd_ratio": 8.0,
            "integration_days": -1.0,
            "integration_steps": -1,
            "integration_offset_days": 0.0,
            "integration_offset_steps": 0,
            "geothermal_heat_flow": -1.0  # W/m², -1 = use planet default
        },
        "thermophysical": {
            # Surface properties
            "use_default_albedo": True,
            "albedo": 0.12,  # Display value (updated based on body)
            "use_default_emissivity": True,
            "emissivity": 0.95,
            "incidence_angle_albedo": True,  # Enable angle-dependent albedo
            # Thermal conductivity
            "use_default_ks": True,
            "ks": 7.4e-4,  # Surface conductivity (W/m/K)
            "use_default_kd": True,
            "kd": 3.4e-3,  # Deep conductivity (W/m/K)
            "use_default_H": True,
            "H": 0.06,  # Scale height (m)
            # Density
            "use_default_rhos": True,
            "rhos": 1100.0,  # Surface density (kg/m³)
            "use_default_rhod": True,
            "rhod": 1800.0,  # Deep density (kg/m³)
            # Heat capacity
            "use_default_cp0": True,
            "cp0": 800.0,  # Reference heat capacity (J/kg/K)
            # Internal heat
            "use_default_Qb": True,
            "Qb": 0.018,  # Geothermal heat flow (W/m²)
            # Radiative conductivity
            "use_default_chi": True,
            "chi": 2.7,  # Radiative conductivity parameter (k = kc * [1 + chi*(T/350)^3])
            # Constant thermal inertia mode
            "constant_thermal_inertia": -1.0  # >0: constant TI (chi=0, cp=const); -1: disabled
        },
        "radiation": {
            "enabled": False,
            "min_shadow_fraction": 0.3,
            "verbose": False,
            "use_gpu": True,
            "view_factor_file": "",
            "max_coupling_iterations": 10,
            "coupling_temp_threshold": 0.5,
            "coupling_flux_threshold": 0.1,
            "coupling_face_fraction": 0.01,
            "use_hmatrix": False,
            "use_aca": True,
            "aca_tolerance": 0.001,
            "aca_max_rank": 50,
            "hmatrix_threads": 0,
            # Adaptive neighbor selection (sparse view factors)
            "adaptive_neighbors": True,
            "captured_energy_target": 0.95,
            "max_neighbors": 500,
            "min_neighbors": 20
        },
        "illumination": {
            "include_primary_reflection": True,
            "use_batch_processing": True,
            "albedo_map": "",
            "albedo_maps_directory": "data/albedo_maps",
            "positions_file": ""
        },
        "output": {
            "directory": "output",
            "format": "binary",
            "vtk_mode": "both",
            "include_primary_ir": False,
            "thermal_vtk_file": "",
            "complete_vtk_file": "",
            "locations_file": "",
            "depths_file": "",
            "series_output": "",
            "location_data": ""
        },
        "advanced": {
            "verbose": False,
            "default_visible_albedo": -1.0
        },
        "generic_body": {
            "name": "",
            "config_file": "",  # Path to HORIZONS-generated body_config.yaml
            "ephemeris_source": "auto",
            "spice_metakernel": "",
            "naif_id": "",
            "shape": {
                "type": "spherical",
                "radius_km": -1.0,
                "semi_axis_a_km": -1.0,
                "semi_axis_b_km": -1.0,
                "semi_axis_c_km": -1.0,
                "mesh_file": "",
                "mesh_scale": 1.0,
            },
            "orbit": {
                "semi_major_axis_au": -1.0,
                "eccentricity": -1.0,
                "inclination_deg": -1.0,
                "lon_ascending_node_deg": -1.0,
                "arg_perihelion_deg": -1.0,
                "mean_anomaly_deg": -1.0,
                "epoch": "",
            },
            "spin": {
                "period_hours": -1.0,
                "pole_ra_deg": -1.0,
                "pole_dec_deg": -1.0,
                "prime_meridian_deg": 0.0,
            },
        }
    }

    PRESETS = {
        "europa_quick": {
            "simulation": {"body": "Europa", "duration_days": 0.5, "timesteps": 24},
            "mesh": {"resolution": 4},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 1.0},
            "output": {"directory": "output/quick_test", "vtk_mode": "thermal"}
        },
        "europa_highres": {
            "simulation": {"body": "Europa", "duration_days": 3.55, "timesteps": 144},
            "mesh": {"resolution": 7},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 1.0},
            "output": {"directory": "output/europa_high_res"},
            "advanced": {"verbose": True}
        },
        "moon_thermal": {
            "simulation": {"body": "Moon", "duration_days": 29.5, "timesteps": 708},
            "mesh": {"resolution": 6},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 1.0},
            "output": {"directory": "output/moon_thermal"}
        },
        "crater_radex": {
            "simulation": {"body": "Moon", "duration_days": 29.5, "timesteps": 96},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 1.0},
            "radiation": {
                "enabled": True, "verbose": True, "use_gpu": True,
                "min_shadow_fraction": 0.3, "max_coupling_iterations": 10
            },
            "output": {"directory": "output/crater_radex"}
        },
        "gpu_accelerated": {
            "thermal": {"use_gpu": True, "use_gpu_illumination": True},
            "radiation": {"enabled": True, "use_gpu": True}
        }
    }

    def __init__(self):
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def get_config(self):
        """Get the current configuration"""
        return self.config

    def set_value(self, key, value):
        """Set a configuration value using dot notation"""
        keys = key.split('.')
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def get_value(self, key, default=None):
        """Get a configuration value using dot notation"""
        keys = key.split('.')
        current = self.config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def load_from_yaml(self, filename):
        """Load configuration from YAML file"""
        with open(filename, 'r') as f:
            loaded = yaml.safe_load(f)

        self.config = self._deep_merge(deepcopy(self.DEFAULT_CONFIG), loaded or {})

        # Derive synthetic GUI fields property_map_mode / property_map_kind
        # from the underlying YAML keys (so the GUI radio reflects the file).
        therm = self.config.get("thermal", {})
        h_path  = therm.get("h_parameter_map", "")
        ti_path = therm.get("thermal_inertia_map", "")
        use_def = therm.get("use_default_property_map", False)
        if h_path:
            therm["property_map_mode"] = 2  # Custom
            therm["property_map_kind"] = 0  # H
        elif ti_path:
            therm["property_map_mode"] = 2  # Custom
            therm["property_map_kind"] = 1  # TI
        elif use_def:
            therm["property_map_mode"] = 1  # Bundled default
            # Kind defaults to H (which is what the bundled default is); user
            # can flip to TI bundled if the file is ti273k_*.bin (not yet
            # auto-discovered in v1).
            therm["property_map_kind"] = 0
        else:
            therm["property_map_mode"] = 0  # None
            therm["property_map_kind"] = 0
        self.config["thermal"] = therm

        # Auto-detect custom thermophysical values from loaded YAML
        # If the YAML contains a thermophysical property, set use_default_* to False
        if loaded and "thermophysical" in loaded:
            loaded_thermo = loaded["thermophysical"]
            # Map of property name to its use_default flag
            property_flags = {
                "albedo": "use_default_albedo",
                "emissivity": "use_default_emissivity",
                "ks": "use_default_ks",
                "kd": "use_default_kd",
                "H": "use_default_H",
                "rhos": "use_default_rhos",
                "rhod": "use_default_rhod",
                "cp0": "use_default_cp0",
                "Qb": "use_default_Qb",
                "chi": "use_default_chi",
            }
            for prop_name, flag_name in property_flags.items():
                if prop_name in loaded_thermo and flag_name not in loaded_thermo:
                    # YAML has the property but no explicit use_default flag —
                    # infer that the user set a custom value (backwards compat)
                    self.config["thermophysical"][flag_name] = False

    @staticmethod
    def _strip_sentinels(d):
        """Remove sentinel values (-1.0, empty strings) from nested dict for clean YAML."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned = ConfigManager._strip_sentinels(v)
                if cleaned:  # Only include non-empty sub-dicts
                    result[k] = cleaned
            elif isinstance(v, float) and v == -1.0:
                continue  # Skip sentinel
            elif isinstance(v, str) and not v:
                continue  # Skip empty strings
            else:
                result[k] = v
        return result

    def _config_for_yaml(self):
        """Return config dict suitable for YAML output.

        Preserves generic_body when it has meaningful data (config_file,
        name, or orbital elements set) so that project files survive
        save/load cycles.  Also sanitises the body name so the separator
        or ``[H] `` display prefix never leak into the YAML.

        For generic_body output:
        - Sentinel values (-1.0, empty strings) are stripped for clean YAML
        - Semi-axes a/b/c are converted to a ``semi_axes_km`` YAML sequence
          as expected by the C++ parser (config_parser.cpp:133)
        """
        from copy import deepcopy as _dc
        out = {}
        for k, v in self.config.items():
            if k == "generic_body":
                # Include generic_body when it carries real data
                gb = v if isinstance(v, dict) else {}
                has_data = (
                    gb.get("config_file") or gb.get("name")
                    or gb.get("orbit", {}).get("semi_major_axis_au", -1.0) > 0
                )
                if has_data:
                    cleaned = self._strip_sentinels(_dc(gb))
                    # Convert separate semi-axis keys to YAML sequence
                    shape = cleaned.get("shape", {})
                    a = shape.pop("semi_axis_a_km", None)
                    b = shape.pop("semi_axis_b_km", None)
                    c = shape.pop("semi_axis_c_km", None)
                    if a is not None and b is not None and c is not None:
                        shape["semi_axes_km"] = [a, b, c]

                    # For generic/custom bodies, write explicit thermophysical
                    # values to generic_body.surface so the C++ backend picks
                    # them up (it doesn't parse the GUI thermophysical section).
                    thermo = self.config.get("thermophysical", {})
                    surface = cleaned.get("surface", {})
                    for key in ("albedo", "emissivity", "ks", "kd",
                                "rhos", "rhod", "H", "cp0", "Qb", "chi"):
                        if key in surface:
                            continue  # Already set (e.g. from HORIZONS config)
                        flag = f"use_default_{key}"
                        if not thermo.get(flag, True):
                            val = thermo.get(key)
                            if val is not None:
                                surface[key] = val
                    if surface:
                        cleaned["surface"] = surface

                    out[k] = cleaned
                continue
            out[k] = v

        # Translate the synthetic property-map mode/kind into the underlying
        # YAML keys the C++ parser consumes, then strip the synthetic ones.
        # mode 0 = None ; 1 = Bundled default ; 2 = Custom file
        # kind 0 = H-parameter ; 1 = Thermal inertia
        # The GUI's QComboBox writeback stores the *label text* (see
        # add_combo_field's currentTextChanged binding), so map labels back
        # to indices defensively.
        _MODE_LABELS = {
            "None (uniform body H)": 0,
            "Bundled icosphere default for body": 1,
            "Custom file": 2,
        }
        _KIND_LABELS = {
            "H-parameter (m)": 0,
            "Thermal inertia (J m^-2 K^-1 s^-1/2)": 1,
        }
        def _to_index(raw, label_map):
            if isinstance(raw, str):
                if raw in label_map:
                    return label_map[raw]
                # numeric string fallback (e.g. "1")
                try:
                    return int(raw)
                except (ValueError, TypeError):
                    return 0
            try:
                return int(raw)
            except (ValueError, TypeError):
                return 0
        if "thermal" in out:
            therm = _dc(out["thermal"])
            mode = _to_index(therm.pop("property_map_mode", 0), _MODE_LABELS)
            kind = _to_index(therm.pop("property_map_kind", 0), _KIND_LABELS)
            # Reset all three underlying knobs first so we serialize a
            # consistent state regardless of stale fields in self.config.
            therm["use_default_property_map"] = False
            therm["thermal_inertia_map"] = ""
            therm["h_parameter_map"] = ""
            if mode == 1:  # Bundled default
                therm["use_default_property_map"] = True
            elif mode == 2:  # Custom file
                # The GUI single file widget binds to thermal_inertia_map by
                # convention; on first load from a YAML that wrote
                # h_parameter_map, the path may still live there. Pull from
                # whichever has data.
                src_thermal = self.config.get("thermal", {})
                custom_path = (
                    src_thermal.get("thermal_inertia_map", "")
                    or src_thermal.get("h_parameter_map", "")
                )
                if kind == 0:
                    therm["h_parameter_map"] = custom_path
                else:
                    therm["thermal_inertia_map"] = custom_path
            out["thermal"] = therm

        # Sanitise simulation.body
        body = out.get("simulation", {}).get("body", "")
        if body.startswith("[H] "):
            out["simulation"] = _dc(out["simulation"])
            out["simulation"]["body"] = body[4:]
        elif "\u2500" in body:
            # Separator character leaked — replace with generic body name
            out["simulation"] = _dc(out["simulation"])
            gb_name = self.config.get("generic_body", {}).get("name", "")
            out["simulation"]["body"] = gb_name or "Custom"

        return out

    def save_to_yaml(self, filename):
        """Save configuration to YAML file"""
        with open(filename, 'w') as f:
            yaml.dump(self._config_for_yaml(), f, default_flow_style=False, sort_keys=False)

    def generate_yaml(self):
        """Generate YAML string for preview"""
        return yaml.dump(self._config_for_yaml(), default_flow_style=False, sort_keys=False)

    def validate(self):
        """Validate the current configuration.

        Returns:
            tuple: (is_valid, list_of_messages)
        """
        errors = []
        warnings = []

        # Simulation parameters
        if self.config["simulation"]["duration_days"] <= 0:
            errors.append("Duration must be positive")
        if self.config["simulation"]["timesteps"] <= 0:
            errors.append("Timesteps must be positive")

        # Mesh parameters
        if self.config["mesh"]["resolution"] < 1:
            errors.append("Mesh resolution must be >= 1")
        if self.config["mesh"]["scale_factor"] <= 0:
            errors.append("Mesh scale factor must be positive")

        # Thermal parameters
        if self.config["thermal"]["threads"] < 0:
            errors.append("Thermal threads cannot be negative")
        if self.config["thermal"]["equilibration_orbits"] < 0:
            errors.append("Equilibration cycles cannot be negative")

        # Radiation parameters
        rad = self.config.get("radiation", {})
        if rad.get("enabled", False):
            if not (0 <= rad.get("min_shadow_fraction", 0) <= 1):
                errors.append("Radiation: min_shadow_fraction must be between 0 and 1")
            if rad.get("max_coupling_iterations", 0) < 1:
                errors.append("Radiation: max_coupling_iterations must be at least 1")
            if rad.get("coupling_temp_threshold", 0) <= 0:
                errors.append("Radiation: coupling_temp_threshold must be positive")
            if rad.get("aca_tolerance", 0) <= 0:
                errors.append("Radiation: aca_tolerance must be positive")

        # Output validation
        valid_formats = ["binary", "vtk"]
        if self.config["output"]["format"] not in valid_formats:
            errors.append(f"Output format must be one of: {', '.join(valid_formats)}")

        valid_vtk_modes = ["thermal", "complete", "both"]
        if self.config["output"]["vtk_mode"] not in valid_vtk_modes:
            errors.append(f"VTK mode must be one of: {', '.join(valid_vtk_modes)}")

        # GPU warnings
        thermal = self.config.get("thermal", {})
        gpu_requested = thermal.get("use_gpu") or rad.get("use_gpu")
        if gpu_requested:
            caps = detect_hardware_capabilities()
            if not caps["opencl_available"]:
                warnings.append("Warning: GPU requested but OpenCL not detected - will fall back to CPU")

        all_messages = errors + warnings
        return (len(errors) == 0, all_messages)

    def load_preset(self, preset_name):
        """Load a preset configuration"""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset = self.PRESETS[preset_name]
        self.reset_to_defaults()
        self.config = self._deep_merge(self.config, preset)

    def _deep_merge(self, base, update):
        """Deep merge two dictionaries"""
        result = deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
