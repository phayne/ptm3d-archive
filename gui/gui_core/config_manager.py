"""
Configuration Manager for PTM3D GUI
Handles loading, saving, validating, and managing simulation configurations
"""

import logging
import yaml
from copy import deepcopy
from pathlib import Path

from ptm.engine_defaults import ENGINE_DEFAULTS

from gui_core.preflight import RECOMMENDED_ENGINE_ENV
from gui_core import yaml_roundtrip

logger = logging.getLogger(__name__)


#: Keys where the bare words `off`/`on` are MEANINGFUL STRINGS to the engine,
#: and where PyYAML's YAML-1.1 resolver therefore hands back a boolean:
#:   thermal.eq_indirect_source  -- auto | converged | off (ptm_thermal_sim.cpp:16534)
#:   illumination.bvh_cache_file -- "off" DISABLES the cache (config_parser.h:435)
#: The engine's yaml-cpp reads the raw scalar, so it sees the string either way.
_YAML11_STRING_KEYS = (
    "thermal.eq_indirect_source",
    "illumination.bvh_cache_file",
)

#: Dotted keys this schema treats as SECTIONS (mappings), at any depth. A
#: config that puts a scalar where one of these belongs has it dropped, which is
#: what yaml-cpp does with a scalar section node under `if (root["radiation"])`.
#: Derived from the default config's own shape below, so it cannot fall behind.
_SECTION_KEYS = set()


def _collect_section_keys(node, prefix=""):
    for key, value in node.items():
        if isinstance(value, dict):
            _SECTION_KEYS.add(f"{prefix}{key}")
            _collect_section_keys(value, f"{prefix}{key}.")


def _flatten_leaves(node, prefix=""):
    """``{dotted key: leaf value}`` for a (possibly nested) section."""
    out = {}
    if not isinstance(node, dict):
        return out
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_leaves(value, dotted))
        else:
            out[dotted] = value
    return out


#: The bodies the ENGINE has built-in properties for, lower-cased for matching.
#: Generated from ``planetary_properties.h`` -- not retyped, so a body added or
#: removed in C++ moves this set rather than needing somebody to remember.
#:
#: This is the discriminator for the emission rule below. It is imported
#: defensively because `ptm.body_table` is generated, and a GUI that cannot
#: import it must still start -- it just cannot make the check, and says so.
try:
    from ptm.body_table import IMPLEMENTED_BODIES as _IMPLEMENTED_BODIES
    BUILTIN_BODY_NAMES = frozenset(str(b).strip().lower()
                                   for b in _IMPLEMENTED_BODIES)
except Exception:                                            # pragma: no cover
    logger.warning(
        "[config] ptm.body_table is unavailable, so a configuration that names "
        "a built-in body AND carries a generic_body: section cannot be "
        "detected here; the engine will run the generic body and ignore "
        "simulation.body.")
    BUILTIN_BODY_NAMES = frozenset()


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
            # Coarse occluder-only mesh (the ring of the core-plus-ring
            # architecture). Added to the illumination shadow-ray BVH only --
            # never thermally solved, radiosity-coupled or output.
            # docs/extended-occluder-architecture.md.
            "occluder_file": "",
            # Where an imported mesh sits on the body. -999 = leave it in its
            # own frame (the engine's "unset" sentinel).
            "latitude_override": -999.0,
            "longitude_override": -999.0,
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
            "geothermal_heat_flow": -1.0,  # W/m², -1 = use planet default
            # Initial condition. spinup_seed: physical | legacy; seed_anchor:
            # lumped | warm (only reachable under physical);
            # analytic_seed_cycles -1 = follow spinup_seed.
            "spinup_seed": "physical",
            "seed_anchor": "lumped",
            "analytic_seed_cycles": -1.0,
            "seed_anchor_refuse_two_timescale": False,
            # Warm start and chunk handoff. Without these a GUI run cannot
            # resume, which is the whole long-run workflow.
            "initial_state_file": "",
            "final_state_file": "",
            # --- Equilibration convergence (item 2.7) ---------------------
            # The GUI exposed the equilibration CYCLE COUNT and a cadence tier
            # and nothing about the criterion those cycles are scored against.
            "equilibration_cycles": -1,
            "equilibration_target_K": 0.1,
            "equilibration_stop_on_certificate": False,
            "eq_indirect_passes": 1,
            "eq_indirect_source": "auto",
            "eq_indirect_flux_file": "",
            # --- Depth grid (item 2.9): DISPLAY ONLY ----------------------
            # Standing instruction: do not retune thermal_b. Shown so a reader
            # can see the grid a run used; the controls are read-only.
            "thermal_b": 15,
            "thermal_m": 10,
            "thermal_n": 15,
            # --- Ice stability (item 2.6) ---------------------------------
            # A whole science product (ice_depth.bin) that had no GUI surface.
            "compute_ice_stability": False,
            "ice_stability": {
                "flux_threshold_kgm2s": 3e-14,
                "geothermal_gradient_K_per_m": 0.0,
                "grain_hop_length_um": 75.0,
                "n_depth_subsamples": -1,
                "sublimation_enthalpy_kJmol": 51.058,
            },
            # --- GPU dispatch (item 2.8) ----------------------------------
            # First-class wedge mitigations on this hardware. gpu_device_ids is
            # a SEQUENCE for the engine (config_parser.cpp gates on
            # IsSequence()), presented as "0,1" text; an empty list means "use
            # gpu_devices" and is omitted from the written YAML.
            "auto_calibrate_gpu": True,
            "gpu_devices": 1,
            "gpu_device_ids": [],
            "gpu_batch_size": 0,
            "gpu_inter_batch_pause_ms": 0,
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
            # Smith (1967) sub-facet roughness. The switch needs the knob:
            # beaming: true with rms_slope_deg == 0 is FATAL at startup. The
            # GUI additionally applies the facet-scale rule of
            # docs/radiation-exchange.md 8.4 -- see gui_core.facet_scale.
            "beaming": False,
            "rms_slope_deg": 0.0,
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
            "min_neighbors": 20,
            # Multi-GPU radiosity dispatch (item 2.8).
            "gpu_devices": 1,
            "gpu_device_ids": [],
            "gpu_batch_size": 0,
            "gpu_inter_batch_pause_ms": 0,
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
            "positions_file": "",
            # Deliberately absent until 2026-08-25, and the reasoning was
            # sound: CPU illumination on this machine is both correct AND
            # faster. Exposed now WITH that measurement in the tooltip, rather
            # than leaving a colleague to wonder why the checkbox exists for
            # thermal and radiation and not here.
            "use_gpu": False,
            "gpu_devices": 1,
            "gpu_device_ids": [],
            "gpu_inter_batch_pause_ms": 0,
            # Reusing a BVH is a large saving on a repeat run of the same mesh.
            "bvh_cache_file": "",
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
        # --- GPU safety policy (item 2.5 / DP-8) --------------------------
        # READ-MOSTLY. These are cross-cutting guardrails, not per-solver
        # tuning, and one of them is unlike every other key in the engine's
        # 184-key surface: writing `min_vram_margin_gib` AT ALL makes the VRAM
        # floor enforcing on every platform rather than only the wedge-prone
        # one (config_parser.h:826, `min_vram_margin_explicit`).
        #
        # So this section is written SELECTIVELY, not like the rest: see
        # _gpu_section_for_yaml. A value equal to the engine's own default is
        # not written, and the floor is written only when the user asks for it
        # through `margin_explicit`, which exists precisely to make the
        # presence semantics a control instead of an accident.
        "gpu": {
            "min_vram_margin_gib": 2.0,
            "placement_policy": "auto",
            "placement_margin_gib": 4.0,
            "thin_margin_policy": "separate",
            "hard_min_vram_margin_gib": 0.25,
            # GUI-synthetic: "write min_vram_margin_gib even at its default".
            # Stripped at save time; set on load when the file carried the key.
            "margin_explicit": False,
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

    #: Property-map combo labels, and the label-or-index decoder. Class-level
    #: because ThermalTab.load_from_config needs the SAME mapping: the combo's
    #: writeback stores the label text, so a second load_from_config() without
    #: an intervening load_from_yaml() handed `int()` a sentence and aborted the
    #: application. Two copies of this decoder is how that happened.
    _MODE_LABELS = {
        "None (uniform body H)": 0,
        "Bundled icosphere default for body": 1,
        "Custom file": 2,
    }
    _KIND_LABELS = {
        "H-parameter (m)": 0,
        "Thermal inertia (J m^-2 K^-1 s^-1/2)": 1,
    }

    @staticmethod
    def _to_index(raw, label_map):
        """Combo label text, a numeric string, or an int -> index. Never raises."""
        if isinstance(raw, str):
            if raw in label_map:
                return label_map[raw]
            try:                       # numeric string fallback (e.g. "1")
                return int(raw)
            except (ValueError, TypeError):
                return 0
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0

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
        #: rms_slope_deg of a loaded config that used the pre-2026-08-25
        #: IMPLICIT beaming switch (a positive slope and no `beaming` key), or
        #: None. It cannot be recovered from `self.config` afterward: the merge
        #: with DEFAULT_CONFIG always supplies `beaming`, so "the file did not
        #: mention it" is information only `load_from_yaml` ever has.
        self.legacy_implicit_beaming = None
        #: DP-1(b) round-trip state. `_source_doc` is the opened file's own
        #: text and comment-carrying document; `_loaded_engine_view` is the
        #: engine-facing view of the config AS LOADED. Saving writes the
        #: difference between that view and the current one, and nothing else,
        #: which is what makes "this key is unset" representable at all.
        self._source_doc = None
        self._loaded_engine_view = None
        #: What the last save_to_yaml actually did: "unchanged" (nothing to
        #: write, the file was not touched), "in_place" (the opened file,
        #: edited minimally), "copied_to_new_file", "replaced_other_file" (Save
        #: As over a DIFFERENT existing file -- its comments are gone), or
        #: "expanded" (the flattening fallback). The caller reports it rather
        #: than assuming; "Comments and layout preserved" over someone else's
        #: file was a false claim.
        self.last_save_mode = None
        #: Lines the last save reformatted that carry no key the user edited.
        #: Empty on every shipped config; non-empty is a layout the writer
        #: could not reproduce, and it is REPORTED rather than left to be
        #: noticed in a later diff.
        self.last_save_layout_changes = []
        #: True once the user has been told that this session's saves cannot
        #: preserve comments (ruamel.yaml missing). The GUI shows it once as a
        #: dialog and every time in the console; it is never silent.
        self.comment_loss_disclosed = False
        #: What the LAST `_config_for_yaml()` had to correct so the emitted
        #: configuration named one body rather than two, or "". Same lifetime
        #: as `last_save_layout_changes`: set on every emission. Read by
        #: `gui_core.preflight.run_preflight`, which puts it in front of the
        #: user before a run starts, and by the body cards' own report.
        self.last_body_identity_conflict = ""
        #: True once `simulation.body` is an explicit statement -- a file that
        #: sets it, or a control that wrote it -- rather than DEFAULT_CONFIG's
        #: 'Moon' showing through.
        #:
        #: The identity rule below MUST NOT fire on the default. A generic-body
        #: configuration may legitimately omit `simulation.body` (the engine
        #: ignores that key entirely when a generic body is configured, and all
        #: seven cached `body_config.yaml` files are written that way), and the
        #: merge with DEFAULT_CONFIG then supplies 'Moon'. Treating that as
        #: "the user named the Moon" would delete the whole `generic_body:`
        #: section of a perfectly good asteroid file and emit a Moon run
        #: instead -- the opposite of the defect this rule exists to fix, and
        #: with the same silence. Same shape of problem, and the same remedy,
        #: as `legacy_implicit_beaming`: "the file did not mention it" is
        #: information only the loader has.
        self.body_is_explicit = False

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = deepcopy(self.DEFAULT_CONFIG)
        self.legacy_implicit_beaming = None
        self._source_doc = None
        self._loaded_engine_view = None
        self.last_body_identity_conflict = ""
        self.body_is_explicit = False

    def generic_body_active(self):
        """True when the config carries real generic-body data (the engine
        then reads thermophysical values from generic_body.surface)."""
        gb = self.config.get("generic_body", {}) or {}
        return bool(
            gb.get("config_file") or gb.get("name")
            or self._as_number(
                (gb.get("orbit") or {}).get("semi_major_axis_au", -1.0), -1.0) > 0
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
        if key == "simulation.body":
            # Somebody chose this: a control, a preset, or a card's mirror of
            # the generic body's name. See `body_is_explicit`.
            self.body_is_explicit = bool(str(value or "").strip())

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

    def reset_to_default(self, path):
        """Reset one section or one key to its default, reporting what moved.

        `path` is a top-level section name (``"generic_body"``) or a dotted key
        (``"mesh.import_file"``). Returns ``{dotted key: previous value}`` for
        every leaf that actually changed -- empty when it was already at its
        default. The return value exists so a caller can SAY what it cleared; a
        setting silently reset is the same defect as a setting silently kept.

        This is what makes a card able to clear another card's residue.
        ``set_value`` cannot do it at section granularity: it writes one leaf,
        so clearing ``generic_body`` with it means enumerating twenty keys and
        remembering to extend the list when the schema grows. Resetting from
        ``DEFAULT_CONFIG`` instead makes the schema the single source of what
        "cleared" means.
        """
        parts = str(path).split(".")
        default_node = self.DEFAULT_CONFIG
        for part in parts:
            if not isinstance(default_node, dict) or part not in default_node:
                # Never silent: a `clears` entry naming a key this schema does
                # not have would otherwise clear nothing and report nothing.
                raise KeyError(
                    f"{path!r} is not a key or section of DEFAULT_CONFIG, so "
                    f"there is no default to reset it to.")
            default_node = default_node[part]
        default = deepcopy(default_node)

        current_parent = self.config
        for part in parts[:-1]:
            node = current_parent.get(part)
            if not isinstance(node, dict):
                node = {}
                current_parent[part] = node
            current_parent = node
        leaf = parts[-1]

        before = _flatten_leaves(current_parent.get(leaf), path) \
            if isinstance(current_parent.get(leaf), dict) \
            else {path: current_parent.get(leaf)}
        after = _flatten_leaves(default, path) if isinstance(default, dict) \
            else {path: default}
        current_parent[leaf] = default
        return {key: value for key, value in before.items()
                if key not in after or after[key] != value}

    def load_from_yaml(self, filename):
        """Load configuration from YAML file"""
        with open(filename, 'r') as f:
            loaded = yaml.safe_load(f)

        # A bare section header (`radiation:` with nothing under it) parses as
        # None, and _deep_merge then puts None where a dict belongs, so every
        # later `rad.get(...)` or `mesh[...]` raises out of a Qt slot -- an
        # application abort. yaml-cpp treats a Null node as falsy under
        # `if (root["radiation"])`, so the engine runs these files happily and
        # only the GUI died. Pre-existing (identical on cf7d97c); fixed here
        # because it is the same crash class as the rest of this branch and the
        # fix is one line at the boundary. Dropping the key is exactly what the
        # engine does with it.
        # Generalized from the None case: ANY non-mapping top-level value is
        # dropped, because every section in this schema is a mapping. A bare
        # `radiation:` (None), a `mesh: ico6` (str) or a `thermal:` followed by
        # a list all used to reach `.get(...)` on a non-dict and abort. Dropping
        # is what yaml-cpp effectively does -- `if (root["radiation"])` on a
        # scalar node reads the section as absent -- so the GUI and the engine
        # now agree about what such a file means.
        if isinstance(loaded, dict):
            # RECURSIVE, because generic_body's orbit/spin/shape are sections
            # too: `generic_body: {orbit: abc}` reached .get on a str and
            # aborted the window from BodyDefinitionTab.load_from_config.
            # Depth-1 sanitizing covered the top-level sections only.
            def _drop_non_mappings(node, prefix=""):
                for key in [k for k, v in node.items() if not isinstance(v, dict)]:
                    dotted = f"{prefix}{key}"
                    # A leaf VALUE is fine; only a key that this schema says is
                    # a section gets dropped.
                    if dotted not in _SECTION_KEYS:
                        continue
                    if node[key] is not None:
                        # `logger`, not a function-local `import logging`: that
                        # import made the name local to the WHOLE function, so
                        # any use ABOVE it raised UnboundLocalError. It did,
                        # the first time a check was added higher up.
                        logger.warning(
                            "[config] section %r is %r, not a mapping; "
                            "ignoring it (the engine reads a scalar section as "
                            "absent too).", dotted, node[key])
                    del node[key]
                for key, value in node.items():
                    if isinstance(value, dict):
                        _drop_non_mappings(value, f"{prefix}{key}.")

            _drop_non_mappings(loaded)
        elif loaded is not None:
            raise ValueError(
                f"{filename} does not contain a YAML mapping at the top level "
                f"(found {type(loaded).__name__}); PTM3D configs are mappings "
                f"of sections.")

        # Record the pre-2026-08-25 IMPLICIT beaming switch before merging.
        # config_parser.cpp:631 turns beaming ON for a config with a positive
        # rms_slope_deg and no `beaming` key, and says so. Once merged with
        # DEFAULT_CONFIG the key is always present, so this is the only moment
        # the distinction exists -- and it matters, because the GUI writes
        # `beaming` explicitly and would otherwise hand the engine
        # `beaming: false`, silently demoting that run to Lambertian.
        loaded_rad = ((loaded or {}).get("radiation", {}) or {}) \
            if isinstance(loaded, dict) else {}
        try:
            # A QUOTED value (rms_slope_deg: "20") comes back as str, which the
            # engine's getWithDefault<double> parses without complaint. A bare
            # comparison raised TypeError here and turned a loadable config into
            # "Failed to load project".
            loaded_slope = float(loaded_rad.get("rms_slope_deg") or 0.0)
        except (TypeError, ValueError):
            loaded_slope = 0.0
        self.legacy_implicit_beaming = (
            loaded_slope
            if "beaming" not in loaded_rad and loaded_slope > 0.0
            else None)

        # Did the FILE name a body? Only the loader can answer that: the merge
        # below supplies DEFAULT_CONFIG's 'Moon' either way, and the identity
        # rule must not read that default as a statement about the body. See
        # `body_is_explicit`.
        loaded_sim = ((loaded or {}).get("simulation", {}) or {}) \
            if isinstance(loaded, dict) else {}
        self.body_is_explicit = bool(
            str((loaded_sim or {}).get("body", "") or "").strip()
            if isinstance(loaded_sim, dict) else False)

        self.config = self._deep_merge(deepcopy(self.DEFAULT_CONFIG), loaded or {})
        if self.legacy_implicit_beaming is not None:
            self.config["radiation"]["beaming"] = True

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
        therm = self.config.get("thermal")
        therm = therm if isinstance(therm, dict) else {}
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
        mesh_cfg = self.config.get("mesh")
        mesh_cfg = mesh_cfg if isinstance(mesh_cfg, dict) else {}
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

            # H joined ENGINE_OVERRIDE_KEYS on 2026-08-25, which turns a plain
            # `thermophysical: {H: ...}` -- inert on a built-in body before that
            # date, and DROPPED by this writer -- into a live `H_override` the
            # engine applies by setting planet.H. That is the point of the
            # change, and it is still a change to an existing config's physics,
            # so it is announced rather than discovered later.
            if "H" in loaded_thermo and "H_override" not in loaded_thermo \
                    and not self.generic_body_active():
                logger.warning(
                    "[thermophysical] this config carries thermophysical.H = "
                    "%s with no H_override. On a built-in body that value used "
                    "to be ignored; the GUI now writes it as "
                    "thermophysical.H_override, which the engine APPLIES. The "
                    "run's regolith scale height changes from the body default "
                    "to this value. Set 'Use Default' on the H row to keep the "
                    "old behavior.", loaded_thermo["H"])

            # Round-trip the engine override keys (what save now writes for
            # built-in bodies): albedo_override: 0.3 -> albedo=0.3 with
            # use_default_albedo=False so the UI shows the custom value.
            for prop_name in self.ENGINE_OVERRIDE_KEYS:
                override_key = f"{prop_name}_override"
                if override_key in loaded_thermo:
                    self.config["thermophysical"][prop_name] = \
                        loaded_thermo[override_key]
                    self.config["thermophysical"][f"use_default_{prop_name}"] = False

        # ---- YAML 1.1 strikes: PyYAML resolves the bare word `off` as the
        # boolean False (and `on`/`yes`/`no` likewise), while the engine's
        # yaml-cpp reads the raw scalar and the values `off`/`on` are
        # MEANINGFUL STRINGS for two keys. Without this, a config spelling the
        # engine's own documented `eq_indirect_source: off` loaded as False,
        # validate() refused a config the engine accepts, and the run YAML
        # carried `false`, which the engine FATALs on. The writer now quotes
        # these on the way out; this handles every file that already exists.
        for dotted in _YAML11_STRING_KEYS:
            section, _, leaf = dotted.partition(".")
            node = self.config.get(section)
            if not isinstance(node, dict) or leaf not in node:
                continue
            if isinstance(node[leaf], bool):
                spelling = "on" if node[leaf] else "off"
                logger.warning(
                    "[config] %s was read as the boolean %r, which is YAML "
                    "1.1 resolving the bare word %r. The engine reads it as "
                    "the string; treating it as %r.",
                    dotted, node[leaf], spelling, spelling)
                node[leaf] = spelling

        # A config that WROTE gpu.min_vram_margin_gib asked for the enforcing
        # floor, whatever value it gave. That is information only this moment
        # has -- after the merge the key is always present -- and dropping it
        # would silently demote the run from "enforced everywhere" to
        # "advisory except on macOS+AMD" the first time the file was saved.
        loaded_gpu = ((loaded or {}).get("gpu", {}) or {}) \
            if isinstance(loaded, dict) else {}
        gpu_cfg = self.config.get("gpu")
        gpu_cfg = gpu_cfg if isinstance(gpu_cfg, dict) else {}
        gpu_cfg["margin_explicit"] = "min_vram_margin_gib" in loaded_gpu
        self.config["gpu"] = gpu_cfg

        # ---- DP-1(b): keep the file's own text, and the engine-facing view of
        # what it said, so a later save can write ONLY what changed.
        self._source_doc = yaml_roundtrip.RoundTripDoc.load(filename)
        self._loaded_engine_view = deepcopy(self._config_for_yaml())
        if self._source_doc is None:
            n_comments = self._count_comment_lines(filename)
            if n_comments:
                logger.warning(
                    "[config] %s carries %d comment line(s) that a save cannot "
                    "preserve. %s", filename, n_comments,
                    self.comment_preservation_status()[1])

    @staticmethod
    def _count_comment_lines(filename):
        """How many comment lines the file carries (for the loss warning)."""
        try:
            text = Path(filename).read_text()
        except Exception:
            return 0
        return sum(1 for line in text.splitlines()
                   if line.lstrip().startswith("#"))

    def comment_preservation_status(self):
        """(preserved, message) -- can this session's saves keep comments?

        A GUI that quietly flattens the file it was given is exactly the defect
        Stage 1 was about, so the answer is surfaced rather than assumed: the
        caller shows the message in the console on every save, and once as a
        dialog before the first lossy one.
        """
        if not yaml_roundtrip.available():
            return (False, yaml_roundtrip.UNAVAILABLE_MESSAGE)
        if self._source_doc is None:
            return (False,
                    "This project was not opened from a YAML file (or its YAML "
                    "could not be round-tripped), so there is no source layout "
                    "to preserve. Save writes the full engine-facing config; "
                    "the next save of that file preserves it.")
        return (True,
                "Comments and layout are preserved: saving rewrites only the "
                "keys whose values changed.")

    def plan_save_edits(self):
        """(updates, removals) a save would apply, or None if not round-tripping.

        Exposed so a test -- and a future diff dialog -- can assert what a save
        touches without writing anything.
        """
        if self._source_doc is None:
            return None
        return yaml_roundtrip.plan_edits(
            self._loaded_engine_view or {},
            self._config_for_yaml(),
            self._source_doc.file_keys())

    #: gpu: keys the engine reads, with the engine's own default beside each.
    #: Taken from ptm.engine_defaults at class-definition time (below), never
    #: retyped: a GUI copy of an engine default is how
    #: radiation.max_coupling_iterations sat at 10 for weeks.
    GPU_POLICY_KEYS = ("min_vram_margin_gib", "placement_policy",
                       "placement_margin_gib", "thin_margin_policy",
                       "hard_min_vram_margin_gib")

    def _gpu_section_for_yaml(self, gpu):
        """The `gpu:` block to write -- usually nothing at all.

        Every other section in this writer emits every key it knows. This one
        cannot: ``gpu.min_vram_margin_gib`` is the only key in the engine's
        whole config surface whose semantics depend on whether it was WRITTEN
        (``config_parser.h``: presence promotes a macOS+AMD guardrail into a
        universal one). Writing the section unconditionally would change the
        GPU policy of every config the GUI touches, on every machine, for
        nobody's reason.

        So: a key is written only when it differs from the engine's own
        default, plus ``min_vram_margin_gib`` whenever the user has ticked
        "enforce on every platform" -- which is that presence semantics made
        into a control instead of an accident.
        """
        out = {}
        explicit = bool(gpu.get("margin_explicit", False))
        for key in self.GPU_POLICY_KEYS:
            if key not in gpu:
                continue
            value = gpu[key]
            engine_value = ENGINE_DEFAULTS.get(f"gpu.{key}")
            if key == "min_vram_margin_gib" and explicit:
                out[key] = value
                continue
            # The SAME equality rule plan_edits uses. A strict type test made
            # an EDIT-FREE open+Save DELETE a line: a file spelling
            # `placement_margin_gib: 4` (int) put 4 into the loaded view, the
            # panel's set_numeric coerced it to 4.0, that now equals the
            # engine's 4.0 default, the key dropped out of the written view --
            # and plan_edits classified it as a REMOVAL. Reproduced end to end
            # through a real ConfigurationPanel.
            if not yaml_roundtrip.values_equal(engine_value, value):
                out[key] = value
        return out

    def gpu_policy_report(self):
        """(rows, writes_section) for the read-mostly display.

        Each row is (key, effective_value, source_text). `source_text` is the
        whole point of the panel: "2.0 (engine default -- enforced on macOS+AMD,
        advisory elsewhere)" and "2.0 (written -- enforced on EVERY platform)"
        are the same number and different physics.
        """
        gpu = self.config.get("gpu")
        gpu = gpu if isinstance(gpu, dict) else {}
        written = self._gpu_section_for_yaml(gpu)
        rows = []
        for key in self.GPU_POLICY_KEYS:
            value = gpu.get(key, ENGINE_DEFAULTS.get(f"gpu.{key}"))
            if key in written:
                if key == "min_vram_margin_gib":
                    source = "written to the config: ENFORCED on every platform"
                else:
                    source = "written to the config"
            elif key == "min_vram_margin_gib":
                source = ("engine default: enforced on macOS+AMD, advisory "
                          "elsewhere")
            else:
                source = "engine default"
            rows.append((key, value, source))
        return rows, dict(written)

    @staticmethod
    def machine_profile():
        """The host GPU calibration profile, or (None, reason).

        Written by `ptm_thermal_sim --calibrate-gpu`. The GUI shows it beside
        the guardrails because "conservative defaults" in a run log means this
        file was missing, and that is worth seeing before the run rather than
        in the console afterward.
        """
        import json
        import os
        path = os.environ.get("PTM_MACHINE_PROFILE")
        if not path:
            home = os.environ.get("HOME", "")
            path = f"{home}/.ptm3d/machine_profile.json" if home else ""
        if not path:
            return None, "no PTM_MACHINE_PROFILE and no HOME to resolve one"
        try:
            with open(path) as handle:
                profile = json.load(handle)
            if not isinstance(profile, dict):
                # A JSON array or scalar sailed past the except clause and was
                # returned as the profile; the caller then did profile.get(...)
                # -> AttributeError, out of GpuPolicyTab.init_ui, which aborts
                # construction of the WHOLE ConfigurationPanel rather than one
                # tab.
                return None, (f"{path} is a JSON {type(profile).__name__}, not "
                              f"an object; regenerate it with "
                              f"ptm_thermal_sim --calibrate-gpu")
            return profile, path
        except FileNotFoundError:
            return None, (
                f"{path} does not exist. The engine will use CONSERVATIVE "
                f"dispatch defaults and say so. Create one with: "
                f"ptm_thermal_sim --calibrate-gpu")
        except Exception as exc:
            return None, f"{path} could not be read ({exc})"

    @staticmethod
    def parse_device_ids(text):
        """'0, 1' -> [0, 1]; '' -> []; anything else -> None (unparseable).

        None is a real answer, not an error to swallow: the engine reads this
        key only when it is a SEQUENCE, so a malformed value would be ignored
        in silence and the run would quietly use a different device set than
        the one on screen. validate() refuses it instead.
        """
        if isinstance(text, (list, tuple)):
            try:
                return [int(v) for v in text]
            except (TypeError, ValueError):
                return None
        if text is None:
            return []
        parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
        parts = [p for p in parts if p]
        if not parts:
            return []
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None

    @staticmethod
    def format_device_ids(value):
        """[0, 1] -> '0,1'. A malformed stored value shows as it is."""
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return "" if value is None else str(value)

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

    def body_identity_conflict(self):
        """The one-line reason this configuration names two different bodies.

        Empty string when it does not. A configuration that sets
        ``simulation.body`` to a body the engine implements AND carries a
        ``generic_body:`` section names two bodies, and the engine resolves
        that in a way nothing in this application used to say out loud::

            ConfigParser::GenericBodyParams::isConfigured()   config_parser.h:717
                return !config_file.empty() || !name.empty()
                       || hasOrbitalElements() || hasSpinState();

            if (sim_config.generic_body.isConfigured()) {     ptm_thermal_sim.cpp:14626
                ...                                          // simulation.body
                planet = thermal::Planet(generic_body.name,   // is NEVER read
                                         generic_body.thermo);

        A single leftover ``generic_body.name`` therefore wins outright:
        ``simulation.body`` is not consulted, ``target_body_name`` is never set
        (it is assigned only on the built-in branch, ``:14127``), and so the
        albedo-map loader at ``:7660`` reports "No body name in config" and
        falls back to a uniform albedo. Measured on 2026-08-27: a Europa Global
        card run emitted ``simulation.body: Europa`` beside
        ``generic_body: {name: bennu}`` and the engine's own provenance.txt
        recorded ``body=bennu``, ``generic_body=true``,
        ``mesh_median_radius=99.99`` km -- a 100 km sphere at 1 AU where a
        1561 km moon at 5.2 AU was asked for.

        Every shipped configuration that carries a ``generic_body:`` block sets
        ``simulation.body`` to that same body's name (25 of 25, measured), so
        this combination is never a legitimate configuration -- it is always
        one card's leftovers under another card's body.
        """
        if not self.body_is_explicit:
            # DEFAULT_CONFIG's 'Moon' showing through, not a choice. See
            # `body_is_explicit`: firing here would delete the generic_body:
            # section of every asteroid configuration that omits the key.
            return ""
        body = str((self.config.get("simulation") or {}).get("body", "") or "")
        body = body.strip()
        if body.startswith("[H] "):
            return ""            # an explicit HORIZONS body: the generic route
        if body.lower() not in BUILTIN_BODY_NAMES:
            return ""
        if not self.generic_body_active():
            return ""
        gb = self.config.get("generic_body") or {}
        named = (gb.get("name") or gb.get("config_file")
                 or f"semi_major_axis_au "
                    f"{(gb.get('orbit') or {}).get('semi_major_axis_au')}")
        return (
            f"simulation.body is {body!r}, but the configuration also carries a "
            f"generic_body: section ({named!r}). The engine reads the "
            f"generic body and ignores simulation.body entirely, so this would "
            f"run {named!r} -- with its radius, orbit, spin and albedo -- under "
            f"the name {body!r}. The generic_body: section is left out of the "
            f"configuration written for the run.")

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

        A ``generic_body:`` section that contradicts a built-in
        ``simulation.body`` is DROPPED here and recorded in
        ``last_body_identity_conflict`` -- see :meth:`body_identity_conflict`
        for why that combination can only ever be another card's residue, and
        what the engine does with it if it is emitted. This is the last point
        before the engine, so it holds whether the residue arrived from a card,
        from the full settings view, or from a file somebody hand-edited.
        """
        from copy import deepcopy as _dc
        # Reset per call, like last_save_layout_changes: this records what THIS
        # emission did, and _config_for_yaml is called on every preview, every
        # save and twice per round-trip plan.
        self.last_body_identity_conflict = self.body_identity_conflict()
        if self.last_body_identity_conflict:
            logger.warning("[config] %s", self.last_body_identity_conflict)
        # ONE decision, used by both the generic_body: branch and the
        # thermophysical: branch below. They used to read
        # `generic_body_active()` separately, which is how a single stale
        # `generic_body.name` also suppressed the thermophysical cleanup and
        # put the whole dead 19-key block back into the file (PTM-67).
        generic_active = (self.generic_body_active()
                          and not self.last_body_identity_conflict)
        out = {}
        for k, v in self.config.items():
            if k == "gpu":
                # Selective, unlike every other section. See the DEFAULT_CONFIG
                # comment: presence is semantics here.
                gpu_out = self._gpu_section_for_yaml(
                    v if isinstance(v, dict) else {})
                if gpu_out:
                    out[k] = gpu_out
                continue
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
                # Include generic_body when it carries real data AND does not
                # contradict a built-in simulation.body. `generic_active` is
                # the single decision computed at the top of this method, so
                # this branch and the thermophysical: branch above can no
                # longer disagree about whether a generic body is in play.
                gb = v if isinstance(v, dict) else {}
                has_data = generic_active and (
                    gb.get("config_file") or gb.get("name")
                    or self._as_number(
                (gb.get("orbit") or {}).get("semi_major_axis_au", -1.0), -1.0) > 0
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
        _MODE_LABELS = self._MODE_LABELS
        _KIND_LABELS = self._KIND_LABELS
        _to_index = self._to_index
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

        # An EMPTY gpu_device_ids list means "no explicit device list; use
        # gpu_devices". The engine gates on IsSequence() and would read `[]` as
        # an explicit empty selection; more to the point, writing it into every
        # config adds three keys that look like settings and are not. Drop it.
        for section in ("thermal", "radiation", "illumination"):
            node = out.get(section)
            if isinstance(node, dict) and node.get("gpu_device_ids") == []:
                node = _dc(node)
                node.pop("gpu_device_ids")
                out[section] = node

        # Sanitize simulation.body
        # str(): a scalar that is not a string still has to be written, not
        # crash the save. yaml.safe_load can hand back a float here.
        body = str(out.get("simulation", {}).get("body", "") or "")
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
        """Save the project file, preserving its comments and layout (DP-1(b)).

        Two paths, and which one ran is reported by
        ``comment_preservation_status()`` -- never inferred by the caller:

        * a file opened from YAML and round-trippable: the loaded document is
          edited in place with ONLY the keys whose engine-facing value changed
          since it was opened, then re-emitted. Comments, ordering, blank lines
          and quoting survive, and a key the file never mentioned stays
          unmentioned unless the user changed it. That last property is what
          makes ``gpu.min_vram_margin_gib`` safe to expose (config_parser.h:
          its mere presence makes the VRAM floor enforcing on every platform).
        * otherwise (a new project, or ruamel.yaml missing): the old full
          ``yaml.dump`` of the engine-facing config.

        After either path the file on disk becomes the new baseline, so the
        next save is again a minimal edit.
        """
        doc = self._source_doc
        # Reset FIRST. It was only ever assigned in the round-trip branch, so
        # a later "unchanged" or "expanded" save re-reported the previous
        # save's reflow -- a warning about a write that did not happen.
        self.last_save_layout_changes = []
        # Save As onto a DIFFERENT existing file replaces it wholesale with
        # this project's document -- which is what Save As means, but it is not
        # "comments preserved": the target's own comments are gone. Reporting
        # "Comments and layout preserved" there was false. Record which
        # happened so the caller can say the true thing.
        target = Path(filename)
        same_file = bool(doc) and doc.path and \
            Path(doc.path).resolve(strict=False) == target.resolve(strict=False)
        if doc is None:
            self.last_save_mode = "expanded"
        elif same_file:
            self.last_save_mode = "in_place"
        elif target.exists():
            self.last_save_mode = "replaced_other_file"
        else:
            self.last_save_mode = "copied_to_new_file"
        if doc is not None:
            updates, removals = yaml_roundtrip.plan_edits(
                self._loaded_engine_view or {}, self._config_for_yaml(),
                doc.file_keys())
            if not updates and not removals and same_file:
                # NOTHING CHANGED, so touch nothing. Re-emitting is close to
                # byte-exact but not exactly: ruamel normalizes a missing
                # trailing newline, and a block sequence indented differently
                # from its own default gets reflowed. Measured across all 77
                # configs/ and examples/ files, 2 differed on a save with an
                # empty plan. Not writing is the only way "no edits is
                # byte-identical" is true for every file rather than most.
                self.last_save_mode = "unchanged"
                return self.last_save_mode
            doc.apply(updates, removals)
            text = doc.dump()
            # ANY line that changed and is not one of the edited keys is a
            # LAYOUT change, not an edit -- ruamel reflowing something. The
            # first-sequence indentation heuristic covers the common case and
            # cannot cover a document that MIXES sequence shapes, so rather
            # than hope, measure and report. Silence about a rewrite of lines
            # nobody edited is the whole failure mode this module exists to
            # remove.
            self.last_save_layout_changes = yaml_roundtrip.layout_only_changes(
                doc.text, text, set(updates) | set(removals))
            if self.last_save_layout_changes:
                logger.warning(
                    "[config] saving %s reformatted %d line(s) that carry no "
                    "key you edited (YAML layout the writer could not "
                    "reproduce exactly): %s", filename,
                    len(self.last_save_layout_changes),
                    " | ".join(self.last_save_layout_changes[:4]))
            with open(filename, 'w') as f:
                f.write(text)
        else:
            with open(filename, 'w') as f:
                yaml.dump(self._config_for_yaml(), f,
                          default_flow_style=False, sort_keys=False)

        # Re-baseline against what is now on disk. Without this a second save
        # would re-apply the first save's edits against a stale "as loaded"
        # view, and setting a key back to its original value would not be
        # written back.
        self._source_doc = yaml_roundtrip.RoundTripDoc.load(filename)
        self._loaded_engine_view = deepcopy(self._config_for_yaml())
        return self.last_save_mode

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

    @staticmethod
    def _as_number(value, fallback):
        """Coerce a YAML scalar for comparison, never raising.

        yaml.safe_load returns a str for a QUOTED number and None for a bare
        `key:`, both of which the engine's getWithDefault<double> handles. A
        bare `>` on either raises TypeError -- and validate() is called
        unguarded from a Qt slot, where an escaping exception makes PyQt6
        qFatal and takes the whole application down with any unsaved edits.
        Measured: exit 134, SIGABRT. The same spelling was already fixed one
        layer up in load_from_yaml; it has to follow the value here too.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        # NaN and the infinities pass float() and then poison everything
        # downstream: `int(inf)` raises OverflowError, `int(nan)` ValueError,
        # and every comparison against NaN is False. YAML spells them `.nan`,
        # `.inf`, `-.inf`. Treat them as "no usable number".
        if number != number or number in (float("inf"), float("-inf")):
            return fallback
        return number

    def validate(self):
        """Validate the current configuration.

        Returns:
            tuple: (is_valid, list_of_messages)
        """
        errors = []
        warnings = []

        # EVERY numeric comparison below goes through _as_number. A quoted YAML
        # scalar loads without complaint (the engine's getWithDefault<double>
        # parses it), and a bare `>` on a str or None raises TypeError out of a
        # Qt slot, which is an application abort -- exit 134. Applying the
        # coercion to only the new checks left seven of the old ones live; a
        # crash-class fix has to cover the function, not the diff.
        num = self._as_number

        # Simulation parameters
        sim = self.config.get("simulation")
        sim = sim if isinstance(sim, dict) else {}
        if num(sim.get("duration_days"), 0.0) <= 0:
            errors.append("Duration must be positive")
        if num(sim.get("timesteps"), 0.0) <= 0:
            errors.append("Timesteps must be positive")

        # Mesh parameters
        mesh_cfg = self.config.get("mesh")
        mesh_cfg = mesh_cfg if isinstance(mesh_cfg, dict) else {}
        if num(mesh_cfg.get("resolution"), 0.0) < 1:
            errors.append("Mesh resolution must be >= 1")
        if num(mesh_cfg.get("scale_factor"), 0.0) <= 0:
            errors.append("Mesh scale factor must be positive")

        # Thermal parameters
        therm = self.config.get("thermal")
        therm = therm if isinstance(therm, dict) else {}
        if num(therm.get("threads"), 0.0) < 0:
            errors.append("Thermal threads cannot be negative")
        if num(therm.get("equilibration_orbits"), 0.0) < 0:
            errors.append("Equilibration cycles cannot be negative")

        # A SPICE ephemeris needs a kernel. The engine refuses this exactly
        # (generic_body.h:94, "SPICE ephemeris source requires a metakernel for
        # body '<name>'"), and `--validate-config` does NOT reach it -- that
        # mode's own banner lists "generic-body construction" under NOT checked
        # -- so a configuration missing the kernel passes every gate this
        # application has and dies after the mesh is built.
        #
        # It is reachable without touching the kernel field at all: choosing
        # SPICE writes `ephemeris_source` alone, and `_strip_sentinels` drops
        # the empty `spice_metakernel` on the way out, so the emitted YAML
        # carries a SPICE source and no kernel key whatsoever. Reproduced
        # 2026-08-28 against a click-path config, 24 of 24 keys identical.
        gb_cfg = self.config.get("generic_body")
        gb_cfg = gb_cfg if isinstance(gb_cfg, dict) else {}
        if (self.generic_body_active()
                and str(gb_cfg.get("ephemeris_source", "") or "").strip().lower()
                == "spice"
                and not str(gb_cfg.get("spice_metakernel", "") or "").strip()):
            errors.append(
                "Positions are set to SPICE but no metakernel is set. The "
                "engine refuses this outright; nothing falls back. Either "
                "press the ephemeris button on the body card to wire one, "
                "pick a kernel file yourself, or choose Keplerian elements "
                "so the run's record says that is what was used.")

        # Radiation parameters
        rad = self.config.get("radiation")
        rad = rad if isinstance(rad, dict) else {}
        if rad.get("enabled", False):
            if not 0 <= num(rad.get("min_shadow_fraction"), 0.0) <= 1:
                errors.append("Radiation: min_shadow_fraction must be between 0 and 1")
            if num(rad.get("max_coupling_iterations"), 0.0) < 1:
                errors.append("Radiation: max_coupling_iterations must be at least 1")
            if num(rad.get("coupling_temp_threshold"), 0.0) <= 0:
                errors.append("Radiation: coupling_temp_threshold must be positive")
            if num(rad.get("aca_tolerance"), 0.0) <= 0:
                errors.append("Radiation: aca_tolerance must be positive")

        # coupling_mode: the engine compares it EXACTLY and makes a mismatch a
        # startup FATAL (ptm_thermal_sim.cpp:14184). There was no check here at
        # all, so `Bounded` passed the launch gate and died in the engine.
        mode = str(rad.get("coupling_mode", "bounded") or "")
        if mode not in ("bounded", "fixed", "legacy"):
            errors.append(
                f"Radiation: coupling_mode is {mode!r}; the engine compares it "
                f"exactly against bounded | fixed | legacy and refuses "
                f"anything else at startup, including a different case.")

        # The switch needs the knob. OUTSIDE the `enabled` branch on purpose:
        # the engine's check is unconditional (ptm_thermal_sim.cpp:13591 sits in
        # plain config application, not behind the radiation gate), and the
        # Beaming group on the tab is not gated on radiation.enabled either. So
        # a user can leave radiation off -- the GUI default -- tick beaming,
        # step the slope back to zero, and launch a run that dies at startup.
        slope = self._as_number(rad.get("rms_slope_deg", 0.0), 0.0)
        if rad.get("beaming", False) and not slope > 0:
            errors.append(
                "Radiation: beaming is on but rms_slope_deg is "
                f"{rad.get('rms_slope_deg', 0.0)!r}. The engine refuses this "
                "pair at startup, whether or not radiation exchange is "
                "enabled. Literature range 20-37 degrees; 32.5 is the "
                "validated lunar polar value (Jhoti et al. 2024).")

        # conductivity_model: martinez2021 has TWO engine-side FATALs that a
        # GUI user can reach with the new combo and never see coming. Same shape
        # as the beaming/slope pair above: the control exists, so the refusal
        # has to exist next to it.
        thermo = self.config.get("thermophysical")
        thermo = thermo if isinstance(thermo, dict) else {}
        model = str(thermo.get("conductivity_model", "hayne2017") or "")
        is_martinez = "".join(
            c for c in model.lower() if c not in "_- ").startswith("martinez")
        if is_martinez:
            c_ti = self._as_number(thermo.get("constant_thermal_inertia", -1.0),
                                   -1.0)
            if c_ti > 0:
                errors.append(
                    "Thermophysical: constant_thermal_inertia "
                    f"({c_ti} SI) and conductivity_model martinez2021 are "
                    "mutually contradictory -- constant TI derives cp from ks, "
                    "and under Martinez the conductivity is not ks but Eq. 10, "
                    "a function of temperature. The engine refuses this pair at "
                    "startup (ptm_thermal_sim.cpp:16723).")
            # Eq. 10's solid coefficient goes NEGATIVE below 1003.52 kg/m^3,
            # so the engine refuses the run. On a built-in body the density
            # rows are disabled (no override path), which means Europa
            # (rho 500/700) cannot be made valid from this GUI at all -- say so
            # here rather than let the launch fail after pre-flight.
            rho_lo = min(self._as_number(thermo.get("rhos", 0.0), 0.0),
                         self._as_number(thermo.get("rhod", 0.0), 0.0))
            if 0 < rho_lo < 1003.52:
                errors.append(
                    f"Thermophysical: conductivity_model martinez2021 is "
                    f"invalid at density {rho_lo:g} kg/m^3 -- Eq. 10 gives a "
                    f"NEGATIVE conductivity below 1003.52. The engine refuses "
                    f"this at startup. Raise the densities, or use hayne2017. "
                    f"(On a built-in body the density rows have no engine "
                    f"override path, so hayne2017 is the only option there.)")

        # The engine gates the mesh rotation on `latitude_override > -900` and
        # range-checks nothing, so -95 is fed straight to rotateMeshToLatLon,
        # where cos(lat) < 0 mirrors the target through the pole: the mesh lands
        # at +85 and longitude+180, silently, somewhere other than the field
        # says. -999 stays the documented "leave it alone" sentinel.
        mesh = self.config.get("mesh")
        mesh = mesh if isinstance(mesh, dict) else {}
        lat = self._as_number(mesh.get("latitude_override", -999.0), -999.0)
        if lat > -900.0 and not (-90.0 <= lat <= 90.0):
            errors.append(
                f"Mesh: latitude_override {lat} is outside [-90, 90]. The "
                "engine does not range-check it and would mirror the mesh "
                "through the pole. Use -999 to leave the mesh in its own frame.")
        lon = self._as_number(mesh.get("longitude_override", -999.0), -999.0)
        if lon > -900.0 and lat <= -900.0:
            # A WARNING, not an error: the Mesh tab disables the longitude
            # field once the latitude gate closes, so a hard error here locked
            # the launch behind a control the user could no longer reach.
            warnings.append(
                "Warning: mesh.longitude_override is set but "
                "latitude_override is not. The engine gates the whole rotation "
                "on latitude, so this longitude will be ignored. Set a "
                "latitude, or clear the longitude to -999.")

        # GPU device lists. The engine reads gpu_device_ids ONLY when it is a
        # YAML sequence (config_parser.cpp gates on IsSequence()), so a
        # malformed value is not rejected -- it is ignored, and the run uses a
        # different device set than the panel shows. Refuse it here, where the
        # user can still see the field.
        for section in ("thermal", "radiation", "illumination"):
            node = self.config.get(section)
            node = node if isinstance(node, dict) else {}
            if "gpu_device_ids" not in node:
                continue
            # Only when that section's GPU path is actually on. The
            # illumination field is DISABLED while illumination.use_gpu is off,
            # so an unconditional error blocked the launch over a field the
            # user could no longer edit, for a key the run would never read.
            # Reproduced: tick the box, type `abc`, untick it.
            if not node.get("use_gpu", False):
                continue
            parsed = self.parse_device_ids(node.get("gpu_device_ids"))
            if parsed is None:
                errors.append(
                    f"{section}.gpu_device_ids is "
                    f"{node.get('gpu_device_ids')!r}, which is not a list of "
                    f"device indices. The engine reads this key only as a YAML "
                    f"sequence, so it would be IGNORED and the run would use a "
                    f"different device set than shown. Use e.g. 0,1 -- or "
                    f"leave it blank to select by count.")
            elif any(v < 0 for v in parsed):
                errors.append(
                    f"{section}.gpu_device_ids contains a negative index "
                    f"({parsed}); device indices start at 0.")

        # GPU safety policy. The engine turns each of these into a startup
        # FATAL; saying so here costs nothing and saves a launch.
        gpu_cfg = self.config.get("gpu")
        gpu_cfg = gpu_cfg if isinstance(gpu_cfg, dict) else {}
        # The engine LOWERCASES both of these before comparing
        # (gpu::guardrails::toLower, ptm_thermal_sim.cpp:14327 and :14349), so
        # `Auto` and `Separate` run fine. Comparing the raw string here made
        # validate() block a valid config while asserting the engine would
        # refuse it -- false, and validate() is the public launch gate, reached
        # headlessly too, not only after set_enum has canonicalized the value.
        placement = str(gpu_cfg.get("placement_policy", "auto") or "").lower()
        if placement not in ("auto", "shared", "disjoint"):
            errors.append(
                f"gpu.placement_policy must be auto, shared or disjoint (got "
                f"{placement!r}); the engine refuses anything else at startup.")
        thin = str(gpu_cfg.get("thin_margin_policy", "separate") or "").lower()
        if thin not in ("separate", "refuse"):
            errors.append(
                f"gpu.thin_margin_policy must be separate or refuse (got "
                f"{thin!r}); the engine refuses anything else at startup.")
        # The engine's test, transcribed rather than recalled
        # (ptm_thermal_sim.cpp:14365): `thin == "separate" && hard > floor`,
        # plus `hard >= 0`, `floor >= 0` and `placement_margin >= 0`.
        # The first spelling here was `hard >= floor > 0`, and it was wrong in
        # BOTH directions -- REPRODUCED against the binary: it refused
        # hard == floor == 2.0, which the engine accepts, and it passed
        # floor = 0 with hard = 0.25, which the engine refuses. A second copy
        # of an engine rule is a second place for it to be wrong; this one
        # survives only because catching it here saves a launch.
        hard = num(gpu_cfg.get("hard_min_vram_margin_gib"), 0.25)
        floor = num(gpu_cfg.get("min_vram_margin_gib"), 2.0)
        if hard < 0.0:
            errors.append(
                f"gpu.hard_min_vram_margin_gib ({hard}) must be >= 0; the "
                f"engine refuses this at startup.")
        if floor < 0.0:
            errors.append(
                f"gpu.min_vram_margin_gib ({floor}) must be >= 0; the engine "
                f"refuses this at startup.")
        if num(gpu_cfg.get("placement_margin_gib"), 4.0) < 0.0:
            errors.append(
                "gpu.placement_margin_gib must be >= 0; the engine refuses "
                "this at startup.")
        if thin == "separate" and hard > floor:
            errors.append(
                f"gpu.hard_min_vram_margin_gib ({hard}) is above "
                f"gpu.min_vram_margin_gib ({floor}), so the 'separate' band is "
                f"empty and only the hard floor would ever apply. The engine "
                f"refuses this at startup. Lower the hard floor, or use "
                f"gpu.thin_margin_policy: refuse.")

        # Ice stability: the engine's own guard rails, stated where the
        # controls are.
        if therm.get("compute_ice_stability", False):
            # MEASURED 2026-08-25 (Moon ico2, 4 steps, engine
            # c6ec28eb2353c06d2131699583f75b5d): the ice-stability accumulator
            # is passed ONLY to runThermalModelsForAllFaces_GPU_Persistent
            # (ptm_thermal_sim.cpp:17320). On the CPU paths the run completes,
            # prints "[Ice stability] WARNING: requested but accumulator not
            # ready", and writes no ice_depth.bin. A WARNING rather than an
            # error: the run itself is valid, it just will not produce the
            # product that was asked for -- and that is exactly the kind of
            # silence worth breaking before the launch, not after.
            if not therm.get("use_gpu", False):
                warnings.append(
                    "Warning: thermal.compute_ice_stability is on but "
                    "thermal.use_gpu is off. The ice-stability accumulator is "
                    "filled only by the GPU thermal solver, so this run will "
                    "complete normally and write NO ice_depth.bin (the engine "
                    "prints '[Ice stability] WARNING: requested but "
                    "accumulator not ready'). Measured on this engine, "
                    "2026-08-25.")
            ice = therm.get("ice_stability")
            ice = ice if isinstance(ice, dict) else {}
            if num(ice.get("flux_threshold_kgm2s"), 0.0) <= 0:
                errors.append(
                    "Ice stability: flux_threshold_kgm2s must be positive; it "
                    "is the sublimation rate below which ice is called stable.")
            if num(ice.get("grain_hop_length_um"), 0.0) <= 0:
                errors.append(
                    "Ice stability: grain_hop_length_um must be positive.")
            if num(ice.get("sublimation_enthalpy_kJmol"), 0.0) <= 0:
                errors.append(
                    "Ice stability: sublimation_enthalpy_kJmol must be "
                    "positive.")

        # Equilibration convergence criterion.
        if num(therm.get("equilibration_target_K"), 0.1) <= 0:
            errors.append(
                "Thermal: equilibration_target_K must be positive; it is the "
                "temperature the equilibration residual is scored against.")
        if num(therm.get("eq_indirect_passes"), 1) < 1:
            errors.append(
                "Thermal: eq_indirect_passes must be at least 1 (the engine "
                "refuses 0 at startup).")
        # ptm_thermal_sim.cpp:16534 -- "Valid: auto | converged | off". Taken
        # from the engine's own refusal, not from memory: the first spelling of
        # this check invented (auto|none|file|internal) and would have refused
        # the one non-default value that works.
        eq_source = str(therm.get("eq_indirect_source", "auto") or "")
        if eq_source not in ("auto", "converged", "off"):
            errors.append(
                f"Thermal: eq_indirect_source must be auto, converged or off "
                f"(got {eq_source!r}); the engine refuses anything else at "
                f"startup.")

        # Output validation. .get(), not [...]: the only unguarded subscript
        # left in this function, and `output: binary` (a scalar section) made it
        # raise "string indices must be integers" out of the Run slot.
        # load_from_yaml drops non-mapping sections now, but validate() is
        # public and is also reached from configs this manager did not load.
        out_cfg = self.config.get("output")
        out_cfg = out_cfg if isinstance(out_cfg, dict) else {}
        # `none` is a REAL value for both, used by five shipped configs
        # (configs/amundsen_survey_template.yaml, amundsen_wilcoski_
        # illumination.yaml, examples/moon_global_topo_res5_flat.yaml,
        # quickstart_europa_albedo_map.yaml, quickstart_europa_diurnal.yaml),
        # and the engine accepts it -- measured, --validate-config rc 0 on
        # quickstart_europa_diurnal.yaml. There is no engine-side check at all
        # on this path: ConfigParser::validate() is DEAD CODE (the only
        # reference is the comment at ptm_thermal_sim.cpp:17063 saying it is
        # not called), and the writer is an if/else chain that simply falls
        # through. So refusing here BLOCKED configs the engine runs -- the
        # second-weaker-copy disagreement item 2.4 exists to remove.
        #
        # An unrecognized value is a WARNING, not an error, because that is
        # what it does: the chain falls through and the run writes nothing.
        # ERRORS, not warnings. The previous pass widened the valid sets
        # correctly and then also downgraded the severity, which let
        # `format: Binary` or `bianry` through the launch gate: the engine's
        # writer is an if/else chain that falls through, so a multi-hour run
        # completes having written no field output at all. The engine refuses
        # nothing here, which is exactly why the GUI must.
        valid_formats = ["binary", "vtk", "none"]
        if out_cfg.get("format") not in valid_formats:
            errors.append(
                f"output.format is {out_cfg.get('format')!r}; it must be one "
                f"of {', '.join(valid_formats)}. The engine does NOT refuse "
                f"anything else -- it falls through and writes no field "
                f"output, so a whole run can finish with nothing to show.")

        valid_vtk_modes = ["thermal", "complete", "both", "none"]
        if out_cfg.get("vtk_mode") not in valid_vtk_modes:
            errors.append(
                f"output.vtk_mode is {out_cfg.get('vtk_mode')!r}; it must be "
                f"one of {', '.join(valid_vtk_modes)}. The engine writes no "
                f"VTK for anything else and does not say so.")

        # A custom body's surface value too small to survive the trip into the
        # solver. The engine forwards it through std::to_string -- six decimal
        # places -- so anything under 5e-7 arrives as "0.000000", fails the
        # receiving positive-value check, and the run silently keeps a
        # PLACEHOLDER property instead (BUGS.md PTM-201; measured, ks = kd =
        # 4e-7 and ks = 0.0096, kd = 0.05 give byte-identical statistics -- a
        # 24000-fold conductivity error with nothing said).
        #
        # An ERROR, not a warning: the run would complete, look complete, and
        # be wrong. ks below 5e-7 W/m/K is not exotic -- it is the porous end
        # of the cometary and trans-Neptunian regolith literature, which is
        # exactly where someone modeling an insulating lag would go.
        try:
            from gui_core.body_cards import tiny_surface_refusals
            errors.extend(tiny_surface_refusals(self.config))
        except ImportError:                              # pragma: no cover
            warnings.append(
                "Warning: the custom-body surface round-trip check could not "
                "be loaded, so a conductivity or density too small to reach "
                "the solver would not be caught here.")

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
_collect_section_keys(ConfigManager.DEFAULT_CONFIG)
