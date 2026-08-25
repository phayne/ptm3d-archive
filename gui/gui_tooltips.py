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
        "Total simulation duration. Use the dropdown to select display units: "
        "days, hours, Earth years, or synodic periods. "
        "When a body is selected, duration defaults to 1 synodic day (solar day). "
        "The underlying value is always stored in Earth days."
    ),
    "simulation.end_time": (
        "Computed simulation end time in UTC (ISO 8601 format). "
        "Editable: changing end time adjusts duration while keeping start time fixed."
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
        "Thermal equilibration period before recording output. "
        "Use the dropdown to select units: synodic periods, orbital periods, days, hours, or Earth years. "
        "The simulation runs this period first to reach thermal equilibrium, "
        "then records output. Default: 1 synodic period. "
        "Higher values (2-5 synodic periods or 1+ orbital periods) may be needed for "
        "bodies with long thermal timescales or high thermal inertia."
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
    "thermophysical.chi": (
        "Radiative conductivity parameter \u03c7 (chi). Controls the T\u00b3-dependent "
        "component of thermal conductivity: k = k_c \u00d7 [1 + \u03c7\u00b7(T/350)\u00b3]. "
        "Typical values: 2.7 for rocky regolith (Hayne et al. 2017), "
        "0 for icy surfaces."
    ),
    "thermophysical.thermal_inertia": (
        "Surface thermal inertia in SI units (J m\u207b\u00b2 K\u207b\u00b9 s\u207b\u00bd). "
        "TI = \u221a(k \u00d7 \u03c1 \u00d7 cp). Editing this value scales ks, kd, \u03c1s, and \u03c1d "
        "by a uniform factor to achieve the target TI, using cp evaluated "
        "at the body's equilibrium temperature. "
        "Typical values: 20\u201350 (lunar regolith), 50\u2013200 (rocky asteroid), "
        "200\u20132500 (bare rock/ice)."
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

    # =========================================================================
    # BODY DEFINITION TAB
    # =========================================================================
    "generic_body.ephemeris_source": (
        "Ephemeris source for position computation. "
        "'auto' uses SPICE if a metakernel is available, otherwise Keplerian. "
        "'keplerian' uses orbital elements only. "
        "'spice' requires a metakernel and NAIF body ID."
    ),
    "generic_body.spice_metakernel": (
        "Path to SPICE metakernel (.tm) file for this body. "
        "Required when ephemeris source is 'spice'."
    ),
    "generic_body.naif_id": (
        "NAIF/SPICE body ID (e.g. '2000269' for asteroid 269 Justitia). "
        "Required for SPICE-based ephemeris."
    ),
    "generic_body.shape_type": (
        "Body shape model: 'spherical' (single radius), "
        "'ellipsoidal' (three semi-axes a, b, c), "
        "or 'custom_mesh' (user-provided OBJ/PLY file)."
    ),
    "generic_body.shape_radius": (
        "Body mean radius in kilometers. Used for spherical bodies "
        "and as a reference for mesh scaling."
    ),
    "generic_body.orbit.sma": (
        "Orbital semi-major axis in AU. Primary parameter controlling "
        "heliocentric distance and equilibrium temperature."
    ),
    "generic_body.orbit.ecc": (
        "Orbital eccentricity (0 = circular, <1 = elliptical). "
        "Controls variation in solar flux over the orbit."
    ),
    "generic_body.orbit.inc": (
        "Orbital inclination in degrees relative to the ecliptic plane."
    ),
    "generic_body.orbit.lan": (
        "Longitude of the ascending node in degrees (ecliptic J2000)."
    ),
    "generic_body.orbit.aop": (
        "Argument of perihelion in degrees."
    ),
    "generic_body.orbit.ma": (
        "Mean anomaly at the reference epoch in degrees."
    ),
    "generic_body.orbit.epoch": (
        "Reference epoch for orbital elements in ISO 8601 format "
        "(e.g. '2000-01-01T12:00:00' for J2000.0)."
    ),
    "generic_body.spin.period": (
        "Sidereal rotation period in hours. Determines the length "
        "of the diurnal thermal cycle."
    ),
    "generic_body.spin.obliquity": (
        "Obliquity: angle between the spin axis and the orbit normal (0\u2013180\u00b0). "
        "0\u00b0 means the spin pole is aligned with the orbit normal (no seasons). "
        "Changing this adjusts pole RA/Dec to achieve the specified obliquity "
        "while preserving the tilt direction."
    ),
    "generic_body.spin.pole_ra": (
        "Right ascension of the spin pole in degrees (J2000 equatorial). "
        "Together with Dec, defines the spin axis orientation and obliquity."
    ),
    "generic_body.spin.pole_dec": (
        "Declination of the spin pole in degrees (J2000 equatorial)."
    ),
    "generic_body.spin.pm": (
        "Prime meridian angle at the reference epoch in degrees."
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
        "chi": 0.0,           # Ice — no radiative conductivity
        "rAU": 5.2,
        "cpCoeff": [7.49],    # cp = 7.49*T (Giauque & Stout 1936)
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
        "chi": 2.7,           # Hayne et al. 2017
        "rAU": 1.0,
        "cpCoeff": [2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9],
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
        "chi": 2.7,
        "rAU": 0.387,
        "cpCoeff": [0.25, 0.0],
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
        "chi": 0.0,           # Ice
        "rAU": 9.54,
        "cpCoeff": [7.49],
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
        "chi": 0.0,           # Ice
        "rAU": 9.54,
        "cpCoeff": [7.49],
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
        "chi": 2.7,
        "rAU": 1.524,
        "cpCoeff": [],        # Constant cp = cp0
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
        "chi": 0.0,           # Ice/clay mixture
        "rAU": 2.77,
        "cpCoeff": [],        # Constant cp = cp0
    },
}


def get_tooltip(key):
    """Get tooltip text for a configuration key."""
    return TOOLTIPS.get(key, "")


def get_body_defaults(body_name):
    """Get default thermophysical properties for a planetary body."""
    return BODY_DEFAULTS.get(body_name, BODY_DEFAULTS["Europa"])


# Synodic day (solar day) in Earth days for built-in bodies.
# These are sunrise-to-sunrise periods matching the C++ thermal model (thermo.day).
BODY_SYNODIC_PERIODS = {
    "Europa": 3.55,         # Synchronous rotation
    "Moon": 29.530588,      # Synodic lunar day
    "Mercury": 175.942,     # Solar day (2/3 orbital period)
    "Enceladus": 1.370218,  # Synchronous rotation
    "Titan": 15.945,        # Synchronous rotation
    "Mars": 1.02749,        # Martian sol
    "Ceres": 0.3781,        # ~9.074 hours
}

# Constants for Kepler's third law (matching C++ orbital_elements.h)
_GM_SUN = 1.32712440018e20  # m^3/s^2
_AU_METERS = 1.496e11       # meters per AU


def _as_number(value, fallback):
    """Coerce a YAML scalar for comparison, never raising.

    These helpers read straight from the config, where yaml.safe_load may have
    left a str (a quoted number, or `5e3`), None (a bare `key:`), or a
    non-finite float. A bare `<=` on any of those raises, and every caller here
    is reached from a Qt slot -- where an escaping exception is a process abort
    (PyQt6 qFatal, exit 134), not an error message.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number or number in (float("inf"), float("-inf")):
        return fallback
    return number


def compute_synodic_day_days(config_manager):
    """Compute synodic day from generic_body config data (Kepler + rotation).

    Mirrors the C++ logic in generic_body.h::updateDerivedProperties().
    Returns the synodic day in Earth days, or None if data is unavailable.
    """
    import math

    a_au = _as_number(
        config_manager.get_value("generic_body.orbit.semi_major_axis_au", -1.0),
        -1.0)
    period_hours = _as_number(
        config_manager.get_value("generic_body.spin.period_hours", -1.0), -1.0)

    if a_au <= 0 or period_hours <= 0:
        return None

    a_m = a_au * _AU_METERS
    orbital_period_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / _GM_SUN)
    sidereal_day_s = period_hours * 3600.0

    if abs(sidereal_day_s - orbital_period_s) > 1.0:
        synodic_s = 1.0 / abs(1.0 / sidereal_day_s - 1.0 / orbital_period_s)
    else:
        synodic_s = sidereal_day_s  # Synchronous

    return synodic_s / 86400.0


def get_synodic_period_days(config_manager, body_name=None):
    """Return the synodic day (solar day) in Earth days for the given body.

    Priority: compute from HORIZONS/generic_body config → fallback dict → current config.
    """
    if body_name is None:
        body_name = config_manager.get_value("simulation.body", "Europa")

    body_name = str(body_name or "")
    clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name

    computed = compute_synodic_day_days(config_manager)
    if computed is not None:
        return computed

    if clean_name in BODY_SYNODIC_PERIODS:
        return BODY_SYNODIC_PERIODS[clean_name]

    return config_manager.get_value("simulation.duration_days", 3.55)


# Orbital period (year) in Earth days for built-in bodies.
# Matches the C++ thermal model (thermo.year).
BODY_ORBITAL_PERIODS = {
    "Europa": 3.55,         # Orbital period around Jupiter (synchronous)
    "Moon": 365.25,         # Earth year
    "Mercury": 87.969,      # Mercury year
    "Enceladus": 1.370218,  # Orbital period around Saturn (synchronous)
    "Titan": 15.945,        # Orbital period around Saturn (synchronous)
    "Mars": 686.97,         # Martian year
    "Ceres": 1681.63,       # ~4.6 Earth years
}


def compute_orbital_period_days(config_manager):
    """Compute orbital period from generic_body config via Kepler's third law.

    Returns the orbital period in Earth days, or None if data is unavailable.
    """
    import math

    a_au = _as_number(
        config_manager.get_value("generic_body.orbit.semi_major_axis_au", -1.0),
        -1.0)
    if a_au <= 0:
        return None

    a_m = a_au * _AU_METERS
    orbital_period_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / _GM_SUN)
    return orbital_period_s / 86400.0


def get_orbital_period_days(config_manager, body_name=None):
    """Return the orbital period (year) in Earth days for the given body.

    Priority: compute from HORIZONS/generic_body config → fallback dict → synodic period.
    """
    if body_name is None:
        body_name = config_manager.get_value("simulation.body", "Europa")

    body_name = str(body_name or "")
    clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name

    computed = compute_orbital_period_days(config_manager)
    if computed is not None:
        return computed

    if clean_name in BODY_ORBITAL_PERIODS:
        return BODY_ORBITAL_PERIODS[clean_name]

    # Fallback: use synodic period as a rough estimate
    return get_synodic_period_days(config_manager, body_name)


# ── Obliquity utilities ──────────────────────────────────────────────
# Earth's obliquity for equatorial↔ecliptic conversion (J2000).
# Matches SpinState::OBLIQUITY_J2000_RAD in src/cpp/orbital/spin_state.h.
_OBLIQUITY_J2000_DEG = 23.4392911


def compute_obliquity_deg(pole_ra_deg, pole_dec_deg,
                          inclination_deg, lon_ascending_node_deg):
    """Compute obliquity (angle between spin pole and orbit normal) in degrees.

    Matches C++ formulas in body_fixed_frame.cpp (spin pole equatorial→ecliptic)
    and kepler_propagator.cpp (orbit normal from orbital elements).

    Returns obliquity in [0, 180] degrees, or None if inputs are invalid.
    """
    import math

    # Orbit normal in ecliptic J2000  (kepler_propagator.cpp:28-54)
    i_rad = math.radians(inclination_deg)
    Omega_rad = math.radians(lon_ascending_node_deg)
    nx = math.sin(Omega_rad) * math.sin(i_rad)
    ny = -math.cos(Omega_rad) * math.sin(i_rad)
    nz = math.cos(i_rad)

    # Spin pole: equatorial J2000 → ecliptic J2000  (body_fixed_frame.cpp:15-37)
    ra_rad = math.radians(pole_ra_deg)
    dec_rad = math.radians(pole_dec_deg)
    sx_eq = math.cos(dec_rad) * math.cos(ra_rad)
    sy_eq = math.cos(dec_rad) * math.sin(ra_rad)
    sz_eq = math.sin(dec_rad)

    eps = math.radians(_OBLIQUITY_J2000_DEG)
    ce, se = math.cos(eps), math.sin(eps)
    sx = sx_eq
    sy = sy_eq * ce + sz_eq * se
    sz = -sy_eq * se + sz_eq * ce

    # Obliquity = arccos(n · s)
    dot = nx * sx + ny * sy + nz * sz
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def compute_pole_for_obliquity(obliquity_deg, current_pole_ra_deg, current_pole_dec_deg,
                                inclination_deg, lon_ascending_node_deg):
    """Compute new pole RA/Dec to achieve the given obliquity, preserving tilt direction.

    When changing obliquity, the spin pole moves along the great circle defined by
    the orbit normal and the current tilt direction. If the current spin pole is
    aligned with the orbit normal (obliquity ≈ 0), the ascending node direction
    is used as the default tilt direction.

    Returns (pole_ra_deg, pole_dec_deg).
    """
    import math

    obl_rad = math.radians(obliquity_deg)

    # Orbit normal in ecliptic J2000
    i_rad = math.radians(inclination_deg)
    Omega_rad = math.radians(lon_ascending_node_deg)
    n = [math.sin(Omega_rad) * math.sin(i_rad),
         -math.cos(Omega_rad) * math.sin(i_rad),
         math.cos(i_rad)]

    # Current spin pole in ecliptic
    ra_rad = math.radians(current_pole_ra_deg)
    dec_rad = math.radians(current_pole_dec_deg)
    sx_eq = math.cos(dec_rad) * math.cos(ra_rad)
    sy_eq = math.cos(dec_rad) * math.sin(ra_rad)
    sz_eq = math.sin(dec_rad)

    eps = math.radians(_OBLIQUITY_J2000_DEG)
    ce, se = math.cos(eps), math.sin(eps)
    s = [sx_eq,
         sy_eq * ce + sz_eq * se,
         -sy_eq * se + sz_eq * ce]

    # Perpendicular component: tilt direction
    dot_sn = s[0] * n[0] + s[1] * n[1] + s[2] * n[2]
    e = [s[k] - dot_sn * n[k] for k in range(3)]
    e_mag = math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)

    if e_mag < 1e-10:
        # Current spin pole ≈ orbit normal → use ascending node direction as default.
        # Ascending node dir = (cos Ω, sin Ω, 0) is always perpendicular to n.
        e = [math.cos(Omega_rad), math.sin(Omega_rad), 0.0]
        e_mag = math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)

    # Normalize tilt direction
    e = [e[k] / e_mag for k in range(3)]

    # New spin pole in ecliptic
    new_s = [math.cos(obl_rad) * n[k] + math.sin(obl_rad) * e[k] for k in range(3)]

    # Ecliptic → equatorial (inverse rotation by ε about x-axis)
    new_sx_eq = new_s[0]
    new_sy_eq = new_s[1] * ce - new_s[2] * se
    new_sz_eq = new_s[1] * se + new_s[2] * ce

    dec = math.degrees(math.asin(max(-1.0, min(1.0, new_sz_eq))))
    ra = math.degrees(math.atan2(new_sy_eq, new_sx_eq)) % 360.0

    return ra, dec


# ── Thermal Inertia utilities ────────────────────────────────────────
_SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann (W/m²/K⁴), SI-2019 exact; canonical: src/cpp/constants/stefan_boltzmann.h


def compute_equilibrium_temperature(albedo, emissivity, rAU):
    """Compute global equilibrium temperature: T_eq = ((1-A)*S/(4*eps*sigma))^0.25.

    Uses S = 1361 / rAU² for solar constant at heliocentric distance rAU.
    """
    rAU = _as_number(rAU, 0.0)
    emissivity = _as_number(emissivity, 0.0)
    albedo = _as_number(albedo, 0.0)
    if rAU <= 0 or emissivity <= 0:
        return 270.0  # Fallback
    S = 1361.0 / (rAU * rAU)
    return ((1.0 - albedo) * S / (4.0 * emissivity * _SIGMA_SB)) ** 0.25


def compute_cp_at_temperature(T, cp0, cpCoeff):
    """Evaluate cp(T) = cp0 + c1*T + c2*T^2 + ... using Horner-like accumulation.

    Matches the C++ thermal model polynomial convention (planetary_properties.h).
    Returns cp, clamped to a minimum of 1.0 J/kg/K.
    """
    cp = cp0
    T_power = T
    for c in cpCoeff:
        cp += c * T_power
        T_power *= T
    return max(cp, 1.0)


def compute_thermal_inertia(ks, rhos, cp):
    """Compute thermal inertia TI = sqrt(ks * rhos * cp)."""
    import math
    return math.sqrt(max(0.0, ks * rhos * cp))
