"""
Tooltip documentation for PTM3D GUI options.
Each key corresponds to a configuration path (e.g., "simulation.body").
"""

TOOLTIPS = {
    # =========================================================================
    # SIMULATION TAB
    # =========================================================================
    "simulation.body": (
        "Target planetary body. Determines default orbital parameters, "
        "thermal properties, and SPICE ephemeris kernels."
    ),
    "simulation.start_time": (
        "Simulation start time in UTC (ISO 8601 format, e.g., 2030-04-10T00:00:00). "
        "Used for ephemeris calculations via SPICE to determine Sun position."
    ),
    "simulation.duration_days": (
        "Total simulation duration in Earth days. For tidally-locked bodies "
        "like Europa, one full rotation = 3.55 days."
    ),
    "simulation.timesteps": (
        "Number of output time steps. Higher values provide finer temporal "
        "resolution but increase output file size and memory usage."
    ),

    # =========================================================================
    # MESH TAB
    # =========================================================================
    "mesh.resolution": (
        "Icosphere subdivision level (0-8). Each level quadruples face count. "
        "Level 5 ≈ 10K faces, Level 6 ≈ 40K faces, Level 7 ≈ 164K faces."
    ),
    "mesh.import_file": (
        "Import a custom mesh file (PLY, OBJ, or STL format). "
        "Must be a closed, manifold triangle mesh."
    ),
    "mesh.scale_factor": (
        "Scale factor applied to imported mesh coordinates. "
        "Use to convert mesh units to meters."
    ),

    # =========================================================================
    # THERMAL TAB
    # =========================================================================
    "thermal.run_thermal_mesh": (
        "Enable thermal simulation. When disabled, only illumination is computed."
    ),
    "thermal.threads": (
        "Number of OpenMP threads for parallel computation. "
        "0 = auto-detect available CPU cores."
    ),
    "thermal.equilibration_orbits": (
        "Number of diurnal cycles to run before recording output. "
        "Ensures thermal equilibrium is reached before data collection. "
        "Default: 1. Higher values (2-5) may be needed for bodies with long thermal timescales."
    ),
    "thermal.use_gpu": (
        "Enable GPU acceleration for the thermal solver. "
        "Requires OpenCL-compatible GPU. Falls back to CPU if unavailable."
    ),
    "thermal.compute_statistics": (
        "Calculate and output temperature statistics (min, max, mean) "
        "for each timestep."
    ),
    "thermal.thermal_inertia_map": (
        "FITS file with spatially-varying thermal inertia values (SI units). "
        "Overrides default uniform thermal properties."
    ),
    "thermal.enable_shadow_flux": (
        "Enable scattered light flux in crater shadows. "
        "Accounts for light scattering from illuminated crater walls."
    ),
    "thermal.crater_dd_ratio": (
        "Crater depth-to-diameter ratio for shadow flux calculation. "
        "Typical values: 5-10 for simple craters."
    ),
    "thermal.integration_days": (
        "Duration of thermal integration in days. "
        "-1 = use simulation duration."
    ),
    "thermal.integration_steps": (
        "Number of integration time steps. "
        "-1 = automatic based on thermal time constant."
    ),
    "thermal.geothermal_heat_flow": (
        "Geothermal/internal heat flux (W/m²) at the bottom boundary. "
        "-1 = use planet default value."
    ),

    # =========================================================================
    # THERMOPHYSICAL TAB
    # =========================================================================
    "thermophysical.albedo": (
        "Bond albedo (0-1). Fraction of incident solar radiation "
        "reflected by the surface. Lower values absorb more energy."
    ),
    "thermophysical.emissivity": (
        "Thermal emissivity (0-1). Efficiency of thermal radiation emission. "
        "Most planetary surfaces have emissivity 0.9-0.95."
    ),
    "thermophysical.incidence_angle_albedo": (
        "Enable solar-incidence-angle dependent albedo (Hayne et al. 2017). "
        "Albedo increases at grazing incidence angles. "
        "Primarily relevant for the Moon."
    ),
    "thermophysical.ks": (
        "Surface thermal conductivity (W/m/K). Controls heat conduction "
        "in the upper regolith layers. Typical range: 10⁻⁴ to 10⁻² W/m/K."
    ),
    "thermophysical.kd": (
        "Deep thermal conductivity (W/m/K). Conductivity at depths much "
        "greater than the scale height H. Usually higher than surface value."
    ),
    "thermophysical.H": (
        "Density/conductivity scale height (m). The e-folding depth for "
        "exponential variation of density and conductivity with depth. "
        "Typical range: 0.02-0.1 m."
    ),
    "thermophysical.rhos": (
        "Surface bulk density (kg/m³). Affects thermal inertia. "
        "Porous regolith: 500-1500 kg/m³."
    ),
    "thermophysical.rhod": (
        "Deep bulk density (kg/m³). Density at depths >> scale height. "
        "Typically higher than surface density due to compaction."
    ),
    "thermophysical.cp0": (
        "Reference heat capacity (J/kg/K). Base value for temperature-dependent "
        "heat capacity calculation. Set to 0 for purely linear T-dependence."
    ),
    "thermophysical.Qb": (
        "Geothermal/internal heat flux (W/m²). Constant heat flux applied "
        "at the bottom boundary of the thermal domain. "
        "Europa: ~0.03, Moon: ~0.018 W/m²."
    ),

    # =========================================================================
    # RADIATION TAB
    # =========================================================================
    "radiation.enabled": (
        "Enable multi-bounce radiation exchange between mesh faces. "
        "Important for craters and rough terrain where faces see each other."
    ),
    "radiation.min_shadow_fraction": (
        "Minimum shadow fraction (0-1) for a face to participate in "
        "radiation exchange. Reduces computation for mostly-sunlit faces."
    ),
    "radiation.use_gpu": (
        "Use GPU acceleration for radiosity solver and view factor computation."
    ),
    "radiation.view_factor_file": (
        "File to save/load precomputed view factors. "
        "Saves time when re-running with same mesh geometry."
    ),
    "radiation.max_coupling_iterations": (
        "Maximum iterations for thermal-radiation coupling loop. "
        "Typically converges in 3-5 iterations."
    ),
    "radiation.coupling_temp_threshold": (
        "Temperature change threshold (K) for coupling convergence. "
        "Loop stops when max temperature change < threshold."
    ),
    "radiation.coupling_flux_threshold": (
        "Flux change threshold (W/m²) for coupling convergence."
    ),
    "radiation.use_hmatrix": (
        "Use H-matrix compression for view factors. "
        "Recommended for meshes with 50K-1M faces. "
        "For larger meshes (>1M faces), consider using sparse view factors with adaptive neighbor selection."
    ),
    "radiation.use_aca": (
        "Use Adaptive Cross Approximation for H-matrix construction. "
        "Faster than full SVD for low-rank blocks."
    ),
    "radiation.aca_tolerance": (
        "Relative tolerance for ACA compression. "
        "Lower values = more accurate but larger memory."
    ),
    "radiation.adaptive_neighbors": (
        "Enable per-face adaptive neighbor selection for sparse view factors. "
        "Selects neighbors based on captured energy fraction for each face."
    ),
    "radiation.captured_energy_target": (
        "Target fraction of radiative potential to capture (0.5-1.0). "
        "Higher values include more neighbors for better accuracy. Default: 0.95 (95%)."
    ),
    "radiation.max_neighbors": (
        "Maximum neighbors per face (hard ceiling). "
        "Limits memory usage for very rough terrain. Default: 500."
    ),
    "radiation.min_neighbors": (
        "Minimum neighbors per face. "
        "Ensures faces always consider some nearby surfaces. Default: 20."
    ),

    # =========================================================================
    # ILLUMINATION TAB
    # =========================================================================
    "illumination.include_primary_reflection": (
        "Include reflected solar radiation from the primary body (e.g., Jupiter for Europa). "
        "Adds reflected sunlight contribution from the primary to the satellite surface."
    ),
    "illumination.use_batch_processing": (
        "Process illumination in batches for better memory efficiency. "
        "Recommended for large meshes."
    ),
    "illumination.albedo_map": (
        "FITS file with spatially-varying Bond albedo values. "
        "Overrides uniform albedo from thermophysical properties."
    ),
    "illumination.albedo_maps_directory": (
        "Directory to search for albedo map files. "
        "Files are matched by resolution level."
    ),
    "illumination.positions_file": (
        "File containing pre-computed ephemeris positions. "
        "Useful for batch processing without SPICE."
    ),

    # =========================================================================
    # OUTPUT TAB
    # =========================================================================
    "output.directory": (
        "Directory for simulation output files. "
        "Created automatically if it doesn't exist."
    ),
    "output.format": (
        "Output format: 'binary' (compact, fast) or 'vtk' (visualization-ready)."
    ),
    "output.vtk_mode": (
        "VTK output mode: 'thermal' (temperature only), 'complete' (all fields), "
        "or 'both' (separate files)."
    ),
    "output.include_primary_ir": (
        "Include thermal IR radiation from the primary body in VTK output."
    ),
    "output.locations_file": (
        "File listing specific lat/lon locations for detailed output. "
        "Format: latitude longitude (one per line)."
    ),
    "output.depths_file": (
        "File listing depth levels (m) for subsurface temperature profiles."
    ),

    # =========================================================================
    # ADVANCED TAB
    # =========================================================================
    "advanced.verbose": (
        "Enable verbose console output for debugging."
    ),
    "advanced.default_visible_albedo": (
        "Override default visible albedo. "
        "-1 = use planet default from thermophysical properties."
    ),
}

# Default thermophysical properties by body (matches planetary_properties.h)
BODY_DEFAULTS = {
    "Europa": {
        "albedo": 0.5,
        "emissivity": 0.9,
        "Qb": 0.03,
        "ks": 0.0096,
        "kd": 0.05,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.07,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
    },
    "Moon": {
        "albedo": 0.12,
        "emissivity": 0.95,
        "Qb": 0.018,
        "ks": 7.4e-4,
        "kd": 3.4e-3,
        "rhos": 1100.0,
        "rhod": 1800.0,
        "H": 0.06,
        "cp0": -3.6125,
        "incidence_angle_albedo": True,
    },
    "Mercury": {
        "albedo": 0.07,
        "emissivity": 0.93,
        "Qb": 0.02,
        "ks": 0.002,
        "kd": 2.0,
        "rhos": 1800.0,
        "rhod": 2200.0,
        "H": 0.1,
        "cp0": 800.0,
        "incidence_angle_albedo": True,
    },
    # Bodies not yet fully implemented in C++ - use reasonable defaults
    "Enceladus": {
        "albedo": 0.81,
        "emissivity": 0.9,
        "Qb": 0.025,
        "ks": 0.001,
        "kd": 0.01,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.05,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
    },
    "Titan": {
        "albedo": 0.22,
        "emissivity": 0.9,
        "Qb": 0.005,
        "ks": 0.02,
        "kd": 0.2,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.1,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
    },
    "Mars": {
        "albedo": 0.25,
        "emissivity": 0.95,
        "Qb": 0.02,
        "ks": 0.02,
        "kd": 2.0,
        "rhos": 1500.0,
        "rhod": 2000.0,
        "H": 0.1,
        "cp0": 800.0,
        "incidence_angle_albedo": False,
    },
    "Ceres": {
        "albedo": 0.09,
        "emissivity": 0.95,
        "Qb": 0.01,
        "ks": 0.001,
        "kd": 0.01,
        "rhos": 1200.0,
        "rhod": 1500.0,
        "H": 0.05,
        "cp0": 800.0,
        "incidence_angle_albedo": False,
    },
}


def get_tooltip(key):
    """Get tooltip text for a configuration key."""
    return TOOLTIPS.get(key, "")


def get_body_defaults(body_name):
    """Get default thermophysical properties for a planetary body."""
    return BODY_DEFAULTS.get(body_name, BODY_DEFAULTS["Europa"])
