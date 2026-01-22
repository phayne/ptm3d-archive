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
            "start_time": "2030-04-10T00:00:00",
            "duration_days": 3.55,
            "timesteps": 144,
            "body": "Europa"
        },
        "mesh": {
            "resolution": 5,
            "use_topographic": False,
            "topo_directory": "",
            "import_file": "",
            "scale_factor": 1.0,
            "export_basic": False
        },
        "thermal": {
            "run_thermal_mesh": True,
            "threads": 0,
            "equilibration_orbits": 2,
            "use_gpu": False,
            "use_gpu_illumination": False,
            "grid_quality": "auto",
            "compute_statistics": True,
            "thermal_inertia_map": "",
            "enable_shadow_flux": False,
            "crater_dd_ratio": 8.0,
            "integration_days": -1.0,
            "integration_steps": -1,
            "integration_offset_days": 0.0,
            "integration_offset_steps": 0
        },
        "radiation": {
            "enabled": False,
            "min_shadow_fraction": 0.3,
            "min_terrain_view_angle": 5.0,
            "max_neighbors": 2000,
            "search_radius_factor": 10.0,
            "max_iterations": 100,
            "convergence_tolerance": 0.01,
            "verbose": False,
            "use_gpu": True,
            "view_factor_file": "",
            "max_coupling_iterations": 10,
            "coupling_temp_threshold": 0.5,
            "coupling_flux_threshold": 0.1,
            "coupling_face_fraction": 0.01,
            "use_hmatrix": False,
            "hmatrix_tolerance": 0.01,
            "hmatrix_max_depth": 10,
            "hmatrix_min_block_size": 64,
            "hmatrix_max_rank": 50
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
            "no_maps": False,
            "locations_file": "",
            "depths_file": "",
            "series_output": "",
            "location_data": ""
        },
        "visualization": {
            "create": False,
            "width": 1024,
            "height": 768,
            "fps": 30,
            "camera_distance": 5.0,
            "camera_latitude": 30.0,
            "camera_longitude": 0.0,
            "emission_angle": -1.0,
            "azimuth": -1.0,
            "save_frames": False
        },
        "advanced": {
            "verbose": False,
            "default_visible_albedo": -1.0
        }
    }

    PRESETS = {
        "europa_quick": {
            "simulation": {"body": "Europa", "duration_days": 0.5, "timesteps": 24},
            "mesh": {"resolution": 4},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 2},
            "output": {"directory": "output/quick_test", "vtk_mode": "thermal"}
        },
        "europa_highres": {
            "simulation": {"body": "Europa", "duration_days": 3.55, "timesteps": 144},
            "mesh": {"resolution": 7},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 2},
            "output": {"directory": "output/europa_high_res"},
            "visualization": {"create": True, "width": 1920, "height": 1080},
            "advanced": {"verbose": True}
        },
        "moon_thermal": {
            "simulation": {"body": "Moon", "duration_days": 29.5, "timesteps": 708},
            "mesh": {"resolution": 6},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 2},
            "output": {"directory": "output/moon_thermal"}
        },
        "crater_radex": {
            "simulation": {"body": "Moon", "duration_days": 29.5, "timesteps": 96},
            "thermal": {"run_thermal_mesh": True, "equilibration_orbits": 2},
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

    def save_to_yaml(self, filename):
        """Save configuration to YAML file"""
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def generate_yaml(self):
        """Generate YAML string for preview"""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

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
            errors.append("Equilibration orbits cannot be negative")

        valid_grid_qualities = ["coarse", "standard", "fine", "auto"]
        if self.config["thermal"].get("grid_quality", "auto") not in valid_grid_qualities:
            errors.append(f"Grid quality must be one of: {', '.join(valid_grid_qualities)}")

        # Radiation parameters
        rad = self.config.get("radiation", {})
        if rad.get("enabled", False):
            if not (0 <= rad.get("min_shadow_fraction", 0) <= 1):
                errors.append("Radiation: min_shadow_fraction must be between 0 and 1")
            if not (0 <= rad.get("min_terrain_view_angle", 0) <= 90):
                errors.append("Radiation: min_terrain_view_angle must be 0-90 degrees")
            if rad.get("max_neighbors", 0) < 10:
                errors.append("Radiation: max_neighbors should be at least 10")
            if rad.get("convergence_tolerance", 0) <= 0:
                errors.append("Radiation: convergence_tolerance must be positive")
            if rad.get("max_iterations", 0) < 1:
                errors.append("Radiation: max_iterations must be at least 1")

        # Output validation
        valid_formats = ["binary", "vtk"]
        if self.config["output"]["format"] not in valid_formats:
            errors.append(f"Output format must be one of: {', '.join(valid_formats)}")

        valid_vtk_modes = ["thermal", "complete", "both"]
        if self.config["output"]["vtk_mode"] not in valid_vtk_modes:
            errors.append(f"VTK mode must be one of: {', '.join(valid_vtk_modes)}")

        # Visualization validation
        if self.config["visualization"]["width"] <= 0 or self.config["visualization"]["height"] <= 0:
            errors.append("Visualization dimensions must be positive")
        if self.config["visualization"]["fps"] <= 0:
            errors.append("FPS must be positive")

        # GPU warnings
        thermal = self.config.get("thermal", {})
        gpu_requested = (thermal.get("use_gpu") or thermal.get("use_gpu_illumination") or rad.get("use_gpu"))
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
