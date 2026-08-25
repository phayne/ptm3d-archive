"""
Configuration Manager for PTM3D GUI
Handles loading, saving, validating, and managing simulation configurations
"""

import yaml
from copy import deepcopy
from pathlib import Path

from ptm.engine_defaults import ENGINE_DEFAULTS

from gui_core.preflight import RECOMMENDED_ENGINE_ENV


def _sync_with_engine_defaults(shape, overrides, prefix=""):
    """Replace every value in `shape` that is an engine key with the engine's own.

    `shape` declares WHICH keys the GUI exposes (and carries the GUI-synthetic
    ones the engine never reads); `ptm.engine_defaults.ENGINE_DEFAULTS` is the
    generated, test-pinned table of what the engine actually defaults to. Taking
    the values from there rather than retyping them is what makes the next
    engine-side change a test failure instead of a GUI that quietly believes an
    old number. `radiation.max_coupling_iterations` sat at 10 for weeks after the
    engine raised it to 16 exactly this way.

    Keys in `overrides` keep the GUI's value; every other key that the engine
    does not read is left as written (GUI-synthetic and display-only fields).
    Mutates and returns `shape`.
    """
    for key, value in shape.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            _sync_with_engine_defaults(value, overrides, dotted + ".")
        elif dotted in ENGINE_DEFAULTS and dotted not in overrides:
            engine_value = ENGINE_DEFAULTS[dotted]
            # A key read without a literal default (a sequence) has None here;
            # that is "no default in the parser", not "defaults to null".
            if engine_value is not None:
                shape[key] = engine_value
    return shape


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

    #: Every dotted key here keeps the GUI's value; every OTHER key in
    #: DEFAULT_CONFIG that the engine reads takes the engine's own default from
    #: ptm.engine_defaults (see _sync_with_engine_defaults). So a divergence is
    #: either on this list with a reason, or it does not exist.
    #:
    #: src/python/tests/test_gui_defaults_follow_engine.py pins the list: an
    #: entry that no longer differs from the engine, or one naming a key the
    #: engine does not read, fails there.
    GUI_DEFAULT_OVERRIDES = {
        "simulation.body":
            "Moon is the best-validated built-in body (Diviner) and the least "
            "surprising starting point; the engine's own default is Europa. "
            "Paul's D1 ruling, 2026-08-02.",
        "simulation.duration_days":
            "One lunar synodic period, to match the default body. The Simulation "
            "tab re-sets this whenever the body changes.",
        "simulation.timesteps":
            "144 steps over one synodic period; the engine's 72 is for its own "
            "3.55-day Europa default.",
        "thermal.run_thermal_mesh":
            "The GUI is a thermal application: a new project runs the thermal "
            "model. The engine defaults it off because illumination-only runs "
            "are a normal CLI use.",
        "thermal.equilibration_samples_per_diurnal_cycle":
            "120 = the 'Balanced' accuracy tier the cadence selector presents "
            "(~4 K max on the Moon ico3 reference). The engine's -1 is 'legacy, "
            "match output cadence', which the tier list offers explicitly.",
        "mesh.topo_directory":
            "The built-in dataset contract: data/topo, resolved through "
            "PTM_DATA_ROOT. The engine's '' means 'no built-in topography'.",
        "generic_body.shape.type":
            "A new custom body starts spherical, which is what the Body "
            "Definition tab's first field offers.",
    }

    DEFAULT_CONFIG = {
        "simulation": {
            "start_time": "2025-01-01T00:00:00",
            # Default duration = one synodic period of the default body (Moon).
            # 29.530588 d matches gui_tooltips.BODY_SYNODIC_PERIODS["Moon"]; the
            # Simulation tab makes this body-aware (a body change re-sets the
            # duration to that body's synodic period). Keeping it here so a fresh
            # project — and its saved YAML — starts at one full lunar day.
            "duration_days": 29.530588,
            "timesteps": 144,
            # Default body is Moon: the best-validated built-in body (Diviner),
            # and the least surprising starting point for a new project. A
            # first-launch picker (ptm3d_gui.py) lets the user choose explicitly.
            "body": "Moon"
        },
        "mesh": {
            "resolution": 5,
            "import_file": "",
            "scale_factor": 1.0,
            "use_topographic": False,
            # Built-in topography ships under data/topo (engine appends
            # <body>/topo_<body>_ico<N>.bin per its search order).
            "topo_directory": "data/topo",
            # GUI-synthetic dataset-source mode: "builtin" | "custom".
            # Stripped at save time; inferred on load (see load_from_yaml).
            "topo_source": "builtin",
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
            # Both of these now have controls on the Thermal tab. The rest of
            # the old orphan set (crater_dd_ratio, integration_offset_days,
            # integration_offset_steps) is gone: written into every saved YAML
            # at the engine's own default, with no widget and no handler, so
            # they read as settings and were not.
            "enable_shadow_flux": False,
            "integration_days": -1.0,
            "integration_steps": -1,
            "geothermal_heat_flow": -1.0  # W/m², -1 = use planet default
        },
        "thermophysical": {
            # Surface properties
            "use_default_albedo": True,
            "albedo": 0.12,  # Display value (updated based on body)
            "use_default_emissivity": True,
            "emissivity": 0.95,
            # Incidence-angle albedo model: false = Hayne (2017) angular albedo
            # (the default for global bodies, per Paul's ruling); true = Hapke
            # directional model (opt-in). Maps to engine key incidence_angle_albedo
            # (config_parser.cpp: false->Hayne/constant, true->Hapke).
            "incidence_angle_albedo": False,
            # Temperature-dependent conductivity law: hayne2017 | martinez2021
            # (config_parser.cpp validates the string and names both).
            "conductivity_model": "hayne2017",
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
            # Values below that the engine reads are REPLACED by the engine's
            # own defaults at class-definition time; what is written here is the
            # key list, not the numbers. See GUI_DEFAULT_OVERRIDES.
            "enabled": False,
            "min_shadow_fraction": 0.3,
            "verbose": False,
            "use_gpu": False,
            "view_factor_file": "",
            # PTM-102: bounded (default) | fixed | legacy. coupling_face_fraction
            # is consulted ONLY under legacy; the engine says so at runtime.
            "coupling_mode": "bounded",
            "max_coupling_iterations": 16,
            "coupling_temp_threshold": 0.5,
            "coupling_flux_threshold": 0.1,
            "coupling_face_fraction": 0.01,
            # Near-field view-factor quadrature order and the builder choice.
            # Order 1 is the M1 centroid estimator (+51.7% on the touching-pair
            # closed form); the engine calls it a diagnostic setting.
            "nearfield_quadrature_order": 3,
            "use_monte_carlo_vf": False,
            "mc_rays_per_face": 100000,
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
            "use_albedo_map": True,   # engine default: auto-load body maps
            "albedo_map": "",
            "albedo_maps_directory": "data/albedo_maps",
            # GUI-synthetic dataset-source mode: "builtin" | "custom" | "none".
            # Stripped at save time; inferred on load (see load_from_yaml).
            "albedo_source": "builtin",
            "positions_file": ""
        },
        "output": {
            # thermal_vtk_file, complete_vtk_file and location_data were
            # written into every saved YAML at '' with no control anywhere.
            # output.vtk_mode already selects which VTK products a run writes;
            # the explicit filename keys are a CLI concern.
            "directory": "output",
            "format": "binary",
            "vtk_mode": "both",
            "include_primary_ir": False,
            "locations_file": "",
            "depths_file": "",
            "series_output": ""
        },
        # The `advanced:` section held exactly two keys (verbose,
        # default_visible_albedo), both at the engine's default, both with no
        # control. Dropping the section changes no run: the engine reads it
        # under `if (root["advanced"])` and falls back to the same values.
        # Engine environment flags (PTM_*) passed to ptm_thermal_sim.
        # Prefilled with the recommended-ON physics fix set; edited via the
        # Engine Flags tab. A blank value means "do not set". The C++ YAML
        # parser ignores this section -- it is consumed by the GUI runner
        # (gui_core.preflight.build_engine_env) and kept in the YAML so a
        # saved project documents exactly what the run had set.
        "engine_env": dict(RECOMMENDED_ENGINE_ENV),
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

    # There is no preset table here on purpose (GUI-PRESET-1). The five that
    # used to live at this spot -- quick, highres, moon_thermal, crater_radex,
    # gpu_accelerated -- were unreachable: `grep -rn load_preset gui/` returned
    # only the definition, and no menu, button or dialog ever called it. They
    # also carried values the rest of this file had moved past
    # (max_coupling_iterations 10, radiation.use_gpu true, an `advanced:`
    # section that no longer exists), and `gpu_accelerated` flipped
    # thermal.use_gpu to true -- one of the two candidate explanations for the
    # 2026-08-14 flip that has never been reproduced.
    #
    # docs/design/gui-update-plan.md item 2.3 designs the replacement: a
    # reachable list built from section 4.4, each preset stating which keys it
    # sets, and a preset that would violate the beaming facet-scale rule
    # refusing with the reason.

    # Properties with an engine-side *_override key for BUILT-IN bodies
    # (config_parser.cpp "Parse Thermophysical section"; the other
    # thermophysical scalars have no override path outside generic_body).
    #
    # H was missing from this tuple until 2026-08-25 even though the engine has
    # read thermophysical.H_override the whole time (config_parser.cpp:759,
    # applied at ptm_thermal_sim.cpp:14778 by setting planet.H directly, so it
    # works on a built-in body). The regolith density scale height is arguably
    # THE lunar thermal parameter, and a colleague could not change it on the
    # Moon from this GUI at all.
    ENGINE_OVERRIDE_KEYS = ("albedo", "emissivity", "ks", "kd", "H")

    def __init__(self):
        self.config = deepcopy(self.DEFAULT_CONFIG)

    def generic_body_active(self):
        """True when the config carries real generic-body data (the engine
        then reads thermophysical values from generic_body.surface)."""
        gb = self.config.get("generic_body", {}) or {}
        return bool(
            gb.get("config_file") or gb.get("name")
            or gb.get("orbit", {}).get("semi_major_axis_au", -1.0) > 0
        )

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

        # engine_env is REPLACED, not merged. A config that carries the block
        # is taken at its word -- `engine_env: {}` means "no fix flags", and a
        # block naming one variable does not silently gain the rest. Merging
        # made both of those impossible to write: every loaded config came back
        # carrying the recommended set whatever it said. A config with no
        # engine_env key at all has expressed no opinion and keeps the default.
        # gui_core.preflight.engine_env_plan applies the same rule at launch.
        # A non-mapping `engine_env:` (a truncated file) keeps the default set
        # instead, which is what engine_env_plan does at launch -- the two must
        # not disagree about what the same file means.
        if isinstance(loaded, dict) and isinstance(loaded.get("engine_env"), dict):
            self.config["engine_env"] = dict(loaded["engine_env"])

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

        # Derive the synthetic dataset-source modes from the loaded engine
        # keys, so existing configs keep their explicit paths (Custom) and
        # only canonical built-in references select Built-in. No behavior
        # change for old configs.
        loaded_mesh = (loaded or {}).get("mesh", {}) or {}
        mesh_cfg = self.config.get("mesh", {})
        topo_dir = str(loaded_mesh.get("topo_directory", "") or "")
        if "topo_directory" not in loaded_mesh or \
                topo_dir.rstrip("/") in ("", "data/topo"):
            mesh_cfg["topo_source"] = "builtin"
            mesh_cfg.setdefault("topo_directory", "data/topo")
        else:
            mesh_cfg["topo_source"] = "custom"
        self.config["mesh"] = mesh_cfg

        loaded_illum = (loaded or {}).get("illumination", {}) or {}
        illum_cfg = self.config.get("illumination", {})
        albedo_map = str(loaded_illum.get("albedo_map", "") or "")
        albedo_dir = str(loaded_illum.get("albedo_maps_directory", "") or "")
        use_map = loaded_illum.get("use_albedo_map", True)
        if use_map is False or (
                "albedo_maps_directory" in loaded_illum
                and not albedo_dir and not albedo_map):
            illum_cfg["albedo_source"] = "none"
        elif albedo_map or (
                albedo_dir and albedo_dir.rstrip("/") != "data/albedo_maps"):
            illum_cfg["albedo_source"] = "custom"
        else:
            illum_cfg["albedo_source"] = "builtin"
        self.config["illumination"] = illum_cfg

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

            # Round-trip the engine override keys (what save now writes for
            # built-in bodies): albedo_override: 0.3 -> albedo=0.3 with
            # use_default_albedo=False so the UI shows the custom value.
            for prop_name in self.ENGINE_OVERRIDE_KEYS:
                override_key = f"{prop_name}_override"
                if override_key in loaded_thermo:
                    self.config["thermophysical"][prop_name] = \
                        loaded_thermo[override_key]
                    self.config["thermophysical"][f"use_default_{prop_name}"] = False

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
        save/load cycles.  Also sanitizes the body name so the separator
        or ``[H] `` display prefix never leak into the YAML.

        For generic_body output:
        - Sentinel values (-1.0, empty strings) are stripped for clean YAML
        - Semi-axes a/b/c are converted to a ``semi_axes_km`` YAML sequence
          as expected by the C++ parser (config_parser.cpp:133)
        """
        from copy import deepcopy as _dc
        generic_active = self.generic_body_active()
        out = {}
        for k, v in self.config.items():
            if k == "mesh":
                # Translate the GUI-synthetic topo_source mode into concrete
                # engine keys, then strip it. Built-in mode writes the
                # canonical RELATIVE reference (data/topo) so the saved
                # config stays portable -- the engine resolves it through
                # its PTM_DATA_ROOT contract on whatever machine runs it.
                mesh = _dc(v) if isinstance(v, dict) else {}
                source = mesh.pop("topo_source", "builtin")
                if source == "builtin":
                    mesh["topo_directory"] = "data/topo"
                out[k] = mesh
                continue
            if k == "illumination":
                # Same translation for albedo_source:
                #   builtin -> use_albedo_map: true, directory data/albedo_maps
                #   none    -> use_albedo_map: false, both paths cleared
                #   custom  -> user paths kept as typed
                illum = _dc(v) if isinstance(v, dict) else {}
                source = illum.pop("albedo_source", "builtin")
                if source == "builtin":
                    illum["use_albedo_map"] = True
                    illum["albedo_map"] = ""
                    illum["albedo_maps_directory"] = "data/albedo_maps"
                elif source == "none":
                    illum["use_albedo_map"] = False
                    illum["albedo_map"] = ""
                    illum["albedo_maps_directory"] = ""
                else:
                    illum["use_albedo_map"] = True
                out[k] = illum
                continue
            if k == "thermophysical" and not generic_active:
                # Built-in body: the C++ parser IGNORES plain scalar keys in
                # this section (albedo, ks, ...); it reads only
                # use_default_albedo, incidence_angle_albedo / albedo_model,
                # constant_thermal_inertia, and the *_override keys
                # (config_parser.cpp "Parse Thermophysical section"). Emit
                # ONLY keys the engine (or the GUI round-trip) actually
                # consumes -- the old behavior wrote dead placeholders like
                # `cp0: 800.0` + `use_default_cp0: true` that looked like
                # live settings.
                thermo = v if isinstance(v, dict) else {}
                cleaned = {
                    "use_default_albedo": thermo.get("use_default_albedo", True),
                    "incidence_angle_albedo": thermo.get(
                        "incidence_angle_albedo", False),
                    "conductivity_model": thermo.get(
                        "conductivity_model", "hayne2017"),
                    "constant_thermal_inertia": thermo.get(
                        "constant_thermal_inertia", -1.0),
                }
                for key in self.ENGINE_OVERRIDE_KEYS:
                    if not thermo.get(f"use_default_{key}", True):
                        val = thermo.get(key)
                        if val is not None:
                            cleaned[f"{key}_override"] = val
                # The remaining properties (rhos, rhod, cp0, Qb, chi) have
                # NO engine override path for built-in bodies; the UI disables
                # them (see ThermophysicalTab) so nothing is silently dropped
                # here.
                out[k] = cleaned
                continue
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

        # Sanitize simulation.body
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

    def save_run_yaml(self, filename, project_root):
        """Write the engine-facing YAML with every path field absolutized
        against `project_root`.

        Use this whenever the YAML is written somewhere other than the
        user's project directory (e.g. the dashboard writing the run config
        beside the output dir): the engine resolves a relative
        radiation.view_factor_file against the YAML's OWN directory
        (config_parser.cpp), so a config that moves must carry absolute
        paths. The old tempfile approach silently broke the VF cache."""
        from gui_core.preflight import absolutize_config_paths
        data = absolutize_config_paths(self._config_for_yaml(), project_root)
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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

    def _deep_merge(self, base, update):
        """Deep merge two dictionaries"""
        result = deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# A typo in GUI_DEFAULT_OVERRIDES would silently do nothing -- the key would
# simply never match, and the GUI would go on taking the engine's value while
# the table claimed otherwise. Refuse at import instead; the GUI-imports step in
# CI runs this.
_unknown_overrides = sorted(
    k for k in ConfigManager.GUI_DEFAULT_OVERRIDES if k not in ENGINE_DEFAULTS)
if _unknown_overrides:
    raise ValueError(
        "ConfigManager.GUI_DEFAULT_OVERRIDES names key(s) the engine does not "
        f"read, so the override could never apply: {_unknown_overrides}")

_sync_with_engine_defaults(ConfigManager.DEFAULT_CONFIG,
                           ConfigManager.GUI_DEFAULT_OVERRIDES)
