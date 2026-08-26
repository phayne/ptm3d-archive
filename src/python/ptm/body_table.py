"""Implemented planetary bodies and their engine properties -- GENERATED.

Do not edit by hand. Derived from src/cpp/thermal/planetary_properties.h and
src/cpp/maps/albedo_manager.cpp by

    python3 -m ptm.body_capabilities --regenerate

src/python/tests/test_body_capabilities.py re-derives this from the C++ sources
on every run and fails if the two disagree, so a body added, removed or retuned
in the engine shows up as a test failure and a diff to this file rather than as
a GUI that quietly describes a different simulator.

3 implemented bodies.
"""

#: Body names `simulation.body` accepts with no `generic_body:` block. Anything
#: else is a FATAL at config parse (thermal::isImplementedBody). The GUI must
#: not offer a name that is not here.
IMPLEMENTED_BODIES = (
    'Moon',
    'Europa',
    'Mercury',
)

#: Fields every row carries. `solar_day_days` is the SOLAR (synodic) day, which
#: is what the engine forces at -- not the sidereal rotation period.
CAPABILITY_FIELDS = (
    'H',
    'Qb',
    'S',
    'albedo',
    'albedoCoef',
    'cp0',
    'cpCoeff',
    'emissivity',
    'kd',
    'ks',
    'rAU',
    'radius_km',
    'rhod',
    'rhos',
    'solar_day_days',
    'spice_metakernel',
    'year_days',
)

#: {body: {field: engine value}}. Lengths in km, periods in Earth days,
#: fluxes in W/m^2, conductivities in W/m/K, densities in kg/m^3.
BODY_CAPABILITIES = {
    'Europa': {
        'H': 0.07,
        'Qb': 0.03,
        'S': 52.0,
        'albedo': 0.5,
        'albedoCoef': (),
        'cp0': 0.0,
        'cpCoeff': (7.49,),
        'emissivity': 0.9,
        'kd': 0.05,
        'ks': 0.0096,
        'rAU': 5.2,
        'radius_km': 1560.8,
        'rhod': 700.0,
        'rhos': 500.0,
        'solar_day_days': 3.55,
        'spice_metakernel': 'europa.tm',
        'year_days': 3.55,
    },
    'Mercury': {
        'H': 0.1,
        'Qb': 0.02,
        'S': 9126.6,
        'albedo': 0.07,
        'albedoCoef': (0.02, 0.0),
        'cp0': 800.0,
        'cpCoeff': (0.25, 0.0),
        'emissivity': 0.93,
        'kd': 2.0,
        'ks': 0.002,
        'rAU': 0.387,
        'radius_km': 2439.7,
        'rhod': 2200.0,
        'rhos': 1800.0,
        'solar_day_days': 175.938,
        'spice_metakernel': 'mercury.tm',
        'year_days': 87.969,
    },
    'Moon': {
        'H': 0.06,
        'Qb': 0.018,
        'S': 1361.0,
        'albedo': 0.12,
        'albedoCoef': (0.06, 0.25),
        'cp0': -3.6125,
        'cpCoeff': (2.7431, 0.0023616, -1.234e-05, 8.9093e-09),
        'emissivity': 0.95,
        'kd': 0.0034,
        'ks': 0.00074,
        'rAU': 1.0,
        'radius_km': 1737.4,
        'rhod': 1800.0,
        'rhos': 1100.0,
        'solar_day_days': 29.530588,
        'spice_metakernel': 'moon.tm',
        'year_days': 365.25,
    },
}

#: FITS albedo-map names the engine tries, in order (AlbedoManager::tryLoadBestMap).
ALBEDO_FITS_PATTERNS = (
    '{Body}-0.5deg-BondAlbedo.fits',
    '{Body}-BondAlbedo.fits',
    '{Body}-albedo.fits',
    '{Body}-1deg-BondAlbedo.fits',
)

#: Per-face binary albedo-map name, tried at each of ALBEDO_MAP_RESOLUTIONS.
ALBEDO_BINARY_PATTERN = '{Body}_Albedo_res{res}.bin'

#: Icosphere resolutions the engine tries for a per-face binary map, in order.
ALBEDO_MAP_RESOLUTIONS = (8, 7, 6, 5, 4, 3)

#: Topography grid names (MeshUtils::checkForTopographicMesh), relative to the
#: topography directory.
TOPOGRAPHY_PATTERNS = (
    '{body}/topo_{body}_ico{res}.bin',
    'topo_{body}_ico{res}.bin',
    '{Body}_Topo_res{res}.bin',
)
