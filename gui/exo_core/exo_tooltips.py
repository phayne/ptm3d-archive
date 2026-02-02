"""
Tooltip documentation for Exosphere GUI options.
Each key corresponds to a configuration path (e.g., "general.ptm_output_dir").
"""

TOOLTIPS = {
    # =========================================================================
    # GENERAL TAB
    # =========================================================================
    "general.ptm_output_dir": (
        "Directory containing PTM3D temperature output (required). "
        "Must contain temperature_K/ subdirectory and mesh.ply."
    ),
    "general.output_dir": (
        "Output directory for exosphere simulation results. "
        "Created automatically if it doesn't exist. Default: exosphere_output."
    ),
    "general.mesh_file": (
        "Optional custom mesh file (PLY format). "
        "If not specified, uses mesh.ply from the PTM3D output directory."
    ),
    "general.timesteps": (
        "Number of diurnal timesteps to simulate (required). "
        "Must match the number of temperature files in PTM3D output. "
        "Typical: 144 for Europa (2.5-min resolution over 3.55 days)."
    ),
    "general.equatorial_ice_mm": (
        "Initial equatorial ice thickness in millimeters. "
        "Sets ice column density at equator; interpolated poleward. "
        "0.0 = no initial ice (sublimation-only mode)."
    ),
    "general.polar_ice_mm": (
        "Initial polar ice thickness in millimeters. "
        "Sets ice column density at poles. Typically thicker than equatorial."
    ),

    # =========================================================================
    # SPUTTERING TAB
    # =========================================================================
    "sputtering.enabled": (
        "Enable magnetospheric plasma sputtering (Cassidy et al. 2013). "
        "Adds ion-driven mass loss that dominates over sublimation at low latitudes."
    ),
    "sputtering.rate_scale": (
        "Scale factor for sputtering rates (default: 1.0). "
        "1.0 = Cassidy et al. 2013 calibrated rates. "
        "Values > 1 increase sputtering; values < 1 reduce it."
    ),
    "sputtering.hemisphere_contrast": (
        "Trailing/leading hemisphere sputtering ratio (default: 3.0). "
        "Europa's trailing hemisphere receives ~3x more magnetospheric flux."
    ),

    # =========================================================================
    # ICE LAYER TAB
    # =========================================================================
    "ice_layer.enabled": (
        "Enable multi-layer ice model with lag deposit tracking. "
        "Tracks ice mass, non-ice concentration, and cumulative fluxes per face. "
        "Required for crystallization and multi-timescale modes."
    ),
    "ice_layer.dust_fraction": (
        "Dust volume fraction in deposited ice (default: 0.05 = 5%). "
        "Controls lag deposit accumulation rate as ice sublimates. "
        "Higher values build protective lag layers faster."
    ),

    # =========================================================================
    # CRYSTALLIZATION TAB (sub-group of Ice Layer)
    # =========================================================================
    "crystallization.enabled": (
        "Enable ice crystallization tracking (Mitchell et al. 2017). "
        "Models competition between thermal crystallization and radiation amorphization. "
        "Auto-enables ice layer model."
    ),
    "crystallization.initial_crystallinity": (
        "Initial crystalline fraction [0-1] (default: 0.0 = fully amorphous). "
        "Fraction of ice in crystalline phase at simulation start."
    ),
    "crystallization.activation_energy": (
        "Crystallization activation energy in J/mol (default: 62000). "
        "Controls temperature sensitivity of Avrami crystallization kinetics. "
        "Mitchell et al. 2017 value: 62000 J/mol."
    ),
    "crystallization.amorphization_dose": (
        "Radiation dose threshold for amorphization in eV/molecule (default: 3.0). "
        "Dose at which crystalline ice transitions fully to amorphous."
    ),
    "crystallization.equatorial_porosity": (
        "Equatorial surface porosity [0-1] (default: 0.5). "
        "Affects radiation penetration depth and thermal conductivity."
    ),
    "crystallization.polar_porosity": (
        "Polar surface porosity [0-1] (default: 0.2). "
        "Typically lower than equatorial due to different surface processes."
    ),
    "crystallization.dose_hemisphere_contrast": (
        "Trailing/leading hemisphere radiation dose ratio (default: 5.0). "
        "Based on Nordheim et al. 2022 magnetospheric particle flux model."
    ),
    "crystallization.dose_polar_enhancement": (
        "Pole/equator radiation dose enhancement factor (default: 2.5). "
        "Accounts for charged particle focusing toward magnetic poles."
    ),
    "crystallization.base_dose_rate": (
        "Base radiation dose rate in eV/(molecule*s) (default: 2.3e-6). "
        "Equatorial leading-hemisphere value from Nordheim et al. 2022."
    ),

    # =========================================================================
    # MULTI-TIMESCALE TAB
    # =========================================================================
    "multiscale.enabled": (
        "Enable multi-timescale geological evolution. "
        "Uses operator splitting to evolve ice over thousands of years: "
        "fast diurnal cycles are averaged, slow processes advanced analytically. "
        "Auto-enables ice layer model."
    ),
    "multiscale.epochs": (
        "Number of geological epochs to simulate (default: 1000). "
        "Each epoch = cycles_per_epoch diurnal cycles. "
        "Total time = epochs * cycles_per_epoch * 3.55 days."
    ),
    "multiscale.cycles_per_epoch": (
        "Number of diurnal cycles per epoch (default: 100). "
        "100 cycles ~ 1 Europa year (355 days). "
        "Higher values improve accuracy but slow simulation."
    ),
    "multiscale.export_interval": (
        "Export VTU snapshot every N epochs (default: 100). "
        "0 = export only final state. Lower values give better temporal resolution."
    ),
    "multiscale.analytical_acceleration": (
        "Use analytical solution for thermally slow regions (default: on). "
        "Accelerates evolution of faces where sublimation is negligible. "
        "Typically safe; disable for debugging only."
    ),
    "multiscale.cache_refresh": (
        "Force diurnal flux recomputation every N epochs (default: 0 = adaptive). "
        "0 = use adaptive refresh (fastest). "
        "Non-zero forces periodic full recomputation even if fluxes haven't changed."
    ),
}


# Field metadata for visualization labels and units
FIELD_INFO = {
    # PTM3D input fields
    "temperature_K": {
        "label": "Temperature",
        "units": "K",
        "category": "ptm3d",
        "colormap": "inferno",
    },
    "illumination": {
        "label": "Solar Illumination",
        "units": "W/m\u00b2",
        "category": "ptm3d",
        "colormap": "hot",
    },

    # Core exosphere fields
    "vapor_pressure_Pa": {
        "label": "Vapor Pressure",
        "units": "Pa",
        "category": "exosphere",
        "colormap": "viridis",
    },
    "emission_flux_kg_m2_s": {
        "label": "Emission Flux",
        "units": "kg/m\u00b2/s",
        "category": "exosphere",
        "colormap": "plasma",
    },
    "deposition_flux_kg_m2_s": {
        "label": "Deposition Flux",
        "units": "kg/m\u00b2/s",
        "category": "exosphere",
        "colormap": "cividis",
    },
    "net_flux_kg_m2_s": {
        "label": "Net Mass Flux",
        "units": "kg/m\u00b2/s",
        "category": "exosphere",
        "colormap": "RdBu_r",
    },
    "escape_flux_kg_m2_s": {
        "label": "Escape Flux",
        "units": "kg/m\u00b2/s",
        "category": "exosphere",
        "colormap": "magma",
    },
    "accumulated_mass_kg_m2": {
        "label": "Accumulated Mass",
        "units": "kg/m\u00b2",
        "category": "exosphere",
        "colormap": "Blues",
    },
    "ice_thickness_m": {
        "label": "Ice Thickness",
        "units": "m",
        "category": "exosphere",
        "colormap": "YlGnBu",
    },

    # Temperature statistics
    "temperature_mean_K": {
        "label": "Mean Temperature",
        "units": "K",
        "category": "exosphere",
        "colormap": "inferno",
    },
    "temperature_min_K": {
        "label": "Min Temperature",
        "units": "K",
        "category": "exosphere",
        "colormap": "inferno",
    },
    "temperature_max_K": {
        "label": "Max Temperature",
        "units": "K",
        "category": "exosphere",
        "colormap": "inferno",
    },

    # Sputtering fields
    "sputtering_flux_kg_m2_s": {
        "label": "Sputtering Flux",
        "units": "kg/m\u00b2/s",
        "category": "sputtering",
        "colormap": "Reds",
    },
    "sputtering_yield": {
        "label": "Sputtering Yield",
        "units": "",
        "category": "sputtering",
        "colormap": "YlOrRd",
    },
    "sputtering_mass_loss_kg_m2_s": {
        "label": "Sputtering Mass Loss",
        "units": "kg/m\u00b2/s",
        "category": "sputtering",
        "colormap": "OrRd",
    },
    "sputtering_spatial_weight": {
        "label": "Sputtering Spatial Weight",
        "units": "",
        "category": "sputtering",
        "colormap": "YlOrBr",
    },

    # Ice layer fields
    "nonice_concentration": {
        "label": "Non-Ice Concentration",
        "units": "",
        "category": "ice_layer",
        "colormap": "copper",
    },
    "ice_mass_kg_m2": {
        "label": "Ice Mass",
        "units": "kg/m\u00b2",
        "category": "ice_layer",
        "colormap": "Blues",
    },
    "cumulative_sublimated_kg_m2": {
        "label": "Cumulative Sublimated",
        "units": "kg/m\u00b2",
        "category": "ice_layer",
        "colormap": "Oranges",
    },
    "cumulative_sputtered_kg_m2": {
        "label": "Cumulative Sputtered",
        "units": "kg/m\u00b2",
        "category": "ice_layer",
        "colormap": "Reds",
    },
    "cumulative_deposited_kg_m2": {
        "label": "Cumulative Deposited",
        "units": "kg/m\u00b2",
        "category": "ice_layer",
        "colormap": "Greens",
    },

    # Ice crystallization fields
    "crystalline_fraction": {
        "label": "Crystalline Fraction",
        "units": "",
        "category": "crystallization",
        "colormap": "coolwarm",
    },
    "porosity": {
        "label": "Porosity",
        "units": "",
        "category": "crystallization",
        "colormap": "bone",
    },
    "radiation_dose_rate_eV_mol_s": {
        "label": "Radiation Dose Rate",
        "units": "eV/(mol\u00b7s)",
        "category": "crystallization",
        "colormap": "hot",
    },
    "cumulative_dose_eV_mol": {
        "label": "Cumulative Radiation Dose",
        "units": "eV/mol",
        "category": "crystallization",
        "colormap": "YlOrRd",
    },
    "crystallization_timescale_yr": {
        "label": "Crystallization Timescale",
        "units": "yr",
        "category": "crystallization",
        "colormap": "viridis_r",
    },
}

# Category display order and labels
FIELD_CATEGORIES = {
    "ptm3d": "PTM3D Input",
    "exosphere": "Exosphere",
    "sputtering": "Sputtering",
    "ice_layer": "Ice Layer",
    "crystallization": "Crystallization",
}


def get_tooltip(key):
    """Get tooltip text for a configuration key."""
    return TOOLTIPS.get(key, "")


def get_field_info(field_name):
    """Get field metadata (label, units, category, colormap)."""
    return FIELD_INFO.get(field_name, {
        "label": field_name.replace("_", " ").title(),
        "units": "",
        "category": "unknown",
        "colormap": "viridis",
    })


def get_fields_by_category(category):
    """Get all field names for a given category."""
    return [name for name, info in FIELD_INFO.items()
            if info["category"] == category]
