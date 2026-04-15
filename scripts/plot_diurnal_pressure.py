#!/usr/bin/env python3
"""Diurnal surface pressure and tangent column density figures for Europa.

Computes sublimation and sputtering contributions to surface pressure from
PTM3D per-timestep temperature files, and generates:
  1. Diurnal surface pressure vs orbital phase (all species)
  2. Tangent column density limb profiles (4 sub-observer views)
  3. Scale height Mollweide map (optional)

All physics recomputed in Python from temperature data — no C++ changes needed.

Usage:
    python scripts/plot_diurnal_pressure.py \
        --ptm-dir output/europa_ico6_uniform-ice_averaged \
        --save-dir docs/figures/transport_averaged
"""

import os
import sys
import struct
import argparse
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker, patches
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'cm',
})

# ---------------------------------------------------------------------------
# Physical constants (matching C++ exosphere_physics.h and sputtering_model.h)
# ---------------------------------------------------------------------------
BOLTZMANN_K = 1.380649e-23           # J/K
H2O_MASS    = 2.99151e-26           # kg (single molecule)
H2O_MOLAR   = 0.01801528            # kg/mol
R_GAS       = 8.314462618           # J/(mol·K)
R_EUROPA    = 1.5608e6              # m
G_EUROPA    = 1.314                 # m/s²
A_EUROPA    = 3.09e13               # m² (4πR²)

# Sputtering (Cassidy et al. 2013)
GLOBAL_H2O_RATE     = 2.0e27        # molecules/s (total Europa)
HEMISPHERE_CONTRAST = 3.0            # trailing/leading ratio
POLAR_ENHANCEMENT   = 1.2           # polar enhancement factor

# Temperature-dependent yield (Teolis et al. 2017)
Q_ACTIVATION   = 0.06               # eV
Q0_PREFACTOR   = 1000.0             # dimensionless
K_BOLTZMANN_EV = 8.617333e-5        # eV/K
T_REF          = 100.0              # K (normalization temperature)
T_LOW          = 80.0               # K (below this, yield = 1)

# Species fractions and escape (sputtering_model.h)
H2O_FRAC   = 0.85;  H2O_ESCAPE = 0.10
O2_FRAC    = 0.10;  O2_ESCAPE  = 0.90;  O2_MASS = 5.3135e-26   # kg
H2_FRAC    = 0.05;  H2_ESCAPE  = 0.95;  H2_MASS = 3.3474e-27   # kg

SPECIES = {
    'H$_2$O': {'frac': H2O_FRAC, 'escape': H2O_ESCAPE, 'mass': H2O_MASS},
    'O$_2$':  {'frac': O2_FRAC,  'escape': O2_ESCAPE,  'mass': O2_MASS},
    'H$_2$':  {'frac': H2_FRAC,  'escape': H2_ESCAPE,  'mass': H2_MASS},
}


# ---------------------------------------------------------------------------
# Physics functions (NumPy-vectorized)
# ---------------------------------------------------------------------------

def vapor_pressure_h2o(T):
    """Marti & Mauersberger (1993) H₂O sublimation vapor pressure [Pa]."""
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    mask = T > 0
    out[mask] = 10.0 ** (-2663.5 / T[mask] + 12.537)
    return out


def scale_height(T, m=H2O_MASS):
    """Scale height H = kT/(mg) [m]."""
    return BOLTZMANN_K * np.asarray(T, dtype=np.float64) / (m * G_EUROPA)


def number_density_sublimation(T):
    """Surface number density from sublimation n = P/(kT) [m⁻³]."""
    T = np.asarray(T, dtype=np.float64)
    P = vapor_pressure_h2o(T)
    out = np.zeros_like(T)
    mask = T > 0
    out[mask] = P[mask] / (BOLTZMANN_K * T[mask])
    return out


def sputtering_spatial_weight(lat, lon):
    """Normalized spatial weight (mean=1). See sputtering_model.h:298-327."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Convert to west longitude (Europa convention)
    west_lon = np.fmod(-lon + 360.0, 360.0)
    west_lon[west_lon < 0] += 360.0

    # Longitude factor: cosine distribution centered on trailing apex (270°W)
    lon_rad = np.radians(west_lon)
    trailing_rad = np.radians(270.0)
    cos_angle = np.cos(lon_rad - trailing_rad)
    min_f = 1.0 / HEMISPHERE_CONTRAST
    lon_factor = min_f + (1.0 - min_f) * (cos_angle + 1.0) / 2.0

    # Latitude factor: slight polar enhancement
    lat_factor = 1.0 + (POLAR_ENHANCEMENT - 1.0) * np.abs(lat) / 90.0

    w = lon_factor * lat_factor
    # Normalize so mean = 1
    w *= len(w) / w.sum()
    return w


def temperature_yield_factor(T):
    """Teolis et al. (2017) yield enhancement. See sputtering_model.h:329-350."""
    T = np.asarray(T, dtype=np.float64)
    out = np.ones_like(T)
    m = T >= T_LOW
    kT = K_BOLTZMANN_EV * T[m]
    kT_ref = K_BOLTZMANN_EV * T_REF
    out[m] = (1.0 + Q0_PREFACTOR * np.exp(-Q_ACTIVATION / kT)) / \
             (1.0 + Q0_PREFACTOR * np.exp(-Q_ACTIVATION / kT_ref))
    return out


def sputtering_number_density(T, spatial_w, species_key='H$_2$O'):
    """Surface number density from sputtered species [m⁻³].

    Non-escaping fraction forms ballistic exosphere.
    Effusion: Φ = n·v_th/4  →  n = 4Φ/v_th.
    """
    sp = SPECIES[species_key]
    frac, escape, mass = sp['frac'], sp['escape'], sp['mass']

    T = np.asarray(T, dtype=np.float64)
    avg_mass_flux = (GLOBAL_H2O_RATE / A_EUROPA) * H2O_MASS  # kg/m²/s total
    species_flux = avg_mass_flux * spatial_w * temperature_yield_factor(T) * frac
    redeposit = species_flux * (1.0 - escape)          # kg/m²/s
    phi = redeposit / mass                              # molec/m²/s
    v_th = np.sqrt(8 * BOLTZMANN_K * T / (np.pi * mass))
    return 4.0 * phi / v_th


def sputtering_pressure(T, spatial_w, species_key='H$_2$O'):
    """Equivalent surface pressure from sputtered species [Pa]."""
    n = sputtering_number_density(T, spatial_w, species_key)
    return n * BOLTZMANN_K * np.asarray(T, dtype=np.float64)


def tangent_column_density(n0, H, h, R=R_EUROPA):
    """Tangent column density [m⁻²] at altitude h for exponential atmosphere."""
    return n0 * np.sqrt(2.0 * np.pi * R * H) * np.exp(-h / H)


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def read_ply_mesh(ply_path):
    """Read binary PLY mesh, return vertices, faces, face lat/lon."""
    with open(ply_path, 'rb') as f:
        n_vertices = 0
        n_faces = 0
        # Parse header
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line == 'end_header':
                break

        # Vertices: 6 doubles (x, y, z, nx, ny, nz)
        vdata = np.frombuffer(f.read(n_vertices * 48), dtype='<f8')
        vdata = vdata.reshape(n_vertices, 6)
        vertices = vdata[:, :3]

        # Faces: uchar(3) + 3×int32 + float lat + float lon = 21 bytes
        lats = np.zeros(n_faces)
        lons = np.zeros(n_faces)
        face_indices = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            data = f.read(21)
            idx = struct.unpack('<3i', data[1:13])
            lat, lon = struct.unpack('<ff', data[13:21])
            face_indices[i] = idx
            lats[i] = lat
            lons[i] = lon

    return vertices, face_indices, lats, lons


def compute_face_areas(vertices, face_indices):
    """Compute triangle areas from vertex positions."""
    v0 = vertices[face_indices[:, 0]]
    v1 = vertices[face_indices[:, 1]]
    v2 = vertices[face_indices[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def read_binary_field(filepath):
    """Read a binary field file: uint32 count + float32 array."""
    with open(filepath, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(count * 4), dtype='<f4')
    return data.astype(np.float64)


def discover_temperature_files(ptm_dir):
    """Find all temperature_XXXX.bin files, sorted by index."""
    pattern = os.path.join(ptm_dir, 'temperature', 'temperature_*.bin')
    files = sorted(glob.glob(pattern))
    return files


# ---------------------------------------------------------------------------
# Regridding (for Mollweide)
# ---------------------------------------------------------------------------

def regrid_to_latlon(lats, lons, values, n_lat=181, n_lon=361):
    """Regrid unstructured face data to regular lat/lon grid."""
    lat_grid = np.linspace(-90, 90, n_lat)
    lon_grid = np.linspace(0, 360, n_lon)
    LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)

    lons_norm = lons % 360
    margin = 10.0
    mask_low = lons_norm < margin
    mask_high = lons_norm > (360 - margin)

    lats_ext = np.concatenate([lats, lats[mask_low], lats[mask_high]])
    lons_ext = np.concatenate([lons_norm, lons_norm[mask_low] + 360,
                               lons_norm[mask_high] - 360])
    vals_ext = np.concatenate([values, values[mask_low], values[mask_high]])

    data_grid = griddata(
        (lons_ext, lats_ext), vals_ext,
        (LON_G, LAT_G), method='linear'
    )
    mask_nan = np.isnan(data_grid)
    if np.any(mask_nan):
        data_nn = griddata(
            (lons_ext, lats_ext), vals_ext,
            (LON_G, LAT_G), method='nearest'
        )
        data_grid[mask_nan] = data_nn[mask_nan]

    return lat_grid, lon_grid, data_grid


# ---------------------------------------------------------------------------
# Figure 1: Diurnal surface pressure
# ---------------------------------------------------------------------------

def fig_diurnal_pressure(temp_files, lats, lons, face_areas, save_path):
    """Global mean surface pressure vs orbital phase (all species)."""
    n_steps = len(temp_files)
    phase = np.linspace(0, 360, n_steps, endpoint=False)

    # Pre-compute spatial weights (static, depends on body-fixed coordinates)
    spatial_w = sputtering_spatial_weight(lats, lons)
    total_area = face_areas.sum()

    # Arrays for area-weighted global mean pressures
    P_sub_h2o = np.zeros(n_steps)
    P_sput = {key: np.zeros(n_steps) for key in SPECIES}

    print("  Computing pressures across diurnal cycle...")
    for t, fpath in enumerate(temp_files):
        T = read_binary_field(fpath)

        # Sublimation vapor pressure (H₂O only)
        P_sub = vapor_pressure_h2o(T)
        P_sub_h2o[t] = np.sum(P_sub * face_areas) / total_area

        # Sputtering equivalent pressure for each species
        for key in SPECIES:
            P_sp = sputtering_pressure(T, spatial_w, species_key=key)
            P_sput[key][t] = np.sum(P_sp * face_areas) / total_area

        if (t + 1) % 50 == 0 or t == n_steps - 1:
            print(f"    timestep {t+1}/{n_steps}")

    # Total sputtering pressure
    P_sput_total = sum(P_sput[k] for k in SPECIES)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5),
                                    gridspec_kw={'height_ratios': [3, 1],
                                                 'hspace': 0.12})

    # (a) Pressure vs orbital phase
    ax1.semilogy(phase, P_sub_h2o, 'C0-', lw=1.5, label='H$_2$O sublimation')
    style = {'H$_2$O': ('C1', '--'), 'O$_2$': ('C2', '-.'), 'H$_2$': ('C3', ':')}
    for key in SPECIES:
        c, ls = style[key]
        ax1.semilogy(phase, P_sput[key], color=c, ls=ls, lw=1.2,
                     label=f'{key} sputtering')
    ax1.semilogy(phase, P_sput_total, 'k-', lw=0.8, alpha=0.5,
                 label='Total sputtering')

    ax1.set_ylabel('Global mean surface pressure (Pa)')
    ax1.set_xlim(0, 360)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top')
    ax1.set_xticklabels([])  # shared x-axis with panel b

    # (b) Ratio: total sputtering / sublimation
    ratio = P_sput_total / np.maximum(P_sub_h2o, 1e-50)
    ax2.semilogy(phase, ratio, 'k-', lw=1.2)
    ax2.axhline(1.0, color='0.5', ls=':', lw=0.5)
    ax2.set_xlabel(r'Orbital phase ($\degree$)')
    ax2.set_ylabel(r'$P_\mathrm{sput}$ / $P_\mathrm{sub}$')
    ax2.set_xlim(0, 360)
    ax2.text(0.02, 0.92, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top')

    _save(fig, save_path)

    # Print diagnostic
    print(f"  P_sub(H₂O) range: {P_sub_h2o.min():.2e} – {P_sub_h2o.max():.2e} Pa")
    for key in SPECIES:
        print(f"  P_sput({key}) range: {P_sput[key].min():.2e} – {P_sput[key].max():.2e} Pa")


# ---------------------------------------------------------------------------
# Figure 2: Tangent column density limb profiles
# ---------------------------------------------------------------------------

def _bin_limb_latitude(species_data, lats, lons, face_areas, obs_lon,
                       n_lat_bins=90):
    """Bin n₀ and H by latitude for limb faces at ±90° from observer."""
    lat_edges = np.linspace(-90, 90, n_lat_bins + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    limb_lon_east = (obs_lon + 90) % 360
    limb_lon_west = (obs_lon - 90) % 360
    lon_norm = lons % 360

    result = {}
    for key, sd in species_data.items():
        n0_profile = np.zeros(n_lat_bins)
        H_profile = np.zeros(n_lat_bins)

        for li in range(n_lat_bins):
            lat_mask = (lats >= lat_edges[li]) & (lats < lat_edges[li + 1])
            if not np.any(lat_mask):
                continue
            d_east = np.abs(((lon_norm - limb_lon_east + 180) % 360) - 180)
            d_west = np.abs(((lon_norm - limb_lon_west + 180) % 360) - 180)
            limb_mask = (d_east < 30) | (d_west < 30)
            combined = lat_mask & limb_mask
            if not np.any(combined):
                combined = lat_mask
            areas = face_areas[combined]
            w = areas / areas.sum()
            n0_profile[li] = np.sum(sd['n0'][combined] * w)
            H_profile[li] = np.sum(sd['H'][combined] * w)

        result[key] = {'n0': n0_profile, 'H': H_profile}

    return lat_centers, result


def fig_limb_profiles(species_data, lats, lons, face_areas, save_path):
    """Tangent column density limb profiles: Europa disk + exosphere halo."""
    views = [
        (r'Sub-Jovian (0$\degree$E)', 0),
        (r'Trailing (90$\degree$E)', 90),
        (r'Anti-Jovian (180$\degree$E)', 180),
        (r'Leading (270$\degree$E)', 270),
    ]

    # Image coordinates: (x, y) in units of R_EUROPA
    # Europa disk at center with radius 1. Exosphere extends beyond.
    r_max = 1.35   # R_Europa units (= 1 + 500 km / 1561 km ≈ 1.32)
    n_pix = 401
    xy = np.linspace(-r_max, r_max, n_pix)
    X, Y = np.meshgrid(xy, xy)
    R_img = np.sqrt(X**2 + Y**2)

    # Polar angle from north pole: θ = 0 at top (north), π at bottom (south)
    # In image coords: north is +y. θ measured from +y axis.
    theta_img = np.arctan2(X, Y)  # angle from +y axis (north)
    # Convert to latitude: lat = 90 - θ_deg
    lat_img = 90.0 - np.degrees(np.arccos(np.clip(Y / np.maximum(R_img, 1e-10), -1, 1)))
    # Simpler: latitude = arcsin(y/r) for points on the limb circle
    lat_img = np.degrees(np.arcsin(np.clip(Y / np.maximum(R_img, 1e-10), -1, 1)))

    # Tangent altitude: h = (R_img - 1) * R_EUROPA  (for R_img > 1)
    h_img = (R_img - 1.0) * R_EUROPA  # meters

    # Latitude bins
    n_lat_bins = 90

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5),
                              gridspec_kw={'hspace': 0.22, 'wspace': 0.05})
    axes = axes.ravel()

    for vi, (view_label, obs_lon) in enumerate(views):
        ax = axes[vi]

        lat_centers, profiles = _bin_limb_latitude(
            species_data, lats, lons, face_areas, obs_lon, n_lat_bins)

        # For each pixel outside the disk, compute column density
        N_img = np.full_like(R_img, np.nan)
        outside = R_img > 1.0

        # For each outside pixel, find the latitude and altitude
        lat_pix = lat_img[outside]
        h_pix = h_img[outside]

        # Evaluate column density by interpolating profiles
        N_pix = np.zeros(lat_pix.shape)
        for key, prof in profiles.items():
            n0_interp = np.interp(lat_pix, lat_centers, prof['n0'],
                                  left=prof['n0'][0], right=prof['n0'][-1])
            H_interp = np.interp(lat_pix, lat_centers, prof['H'],
                                 left=prof['H'][0], right=prof['H'][-1])
            valid = (n0_interp > 0) & (H_interp > 0)
            N_pix[valid] += tangent_column_density(
                n0_interp[valid], H_interp[valid], h_pix[valid])

        # Convert to cm⁻²
        N_pix_cm2 = N_pix * 1e-4
        N_pix_cm2 = np.maximum(N_pix_cm2, 1.0)

        N_img[outside] = np.log10(N_pix_cm2)

        # Plot exosphere halo
        pcm = ax.pcolormesh(xy, xy, N_img, cmap='Blues',
                            vmin=10, vmax=13, shading='auto', rasterized=True)

        # Draw Europa's disk
        disk = patches.Circle((0, 0), 1.0, fc='0.85', ec='0.3', lw=0.8,
                               zorder=5)
        ax.add_patch(disk)

        # Equator and pole markers
        ax.plot([-1, 1], [0, 0], 'k-', lw=0.3, alpha=0.3, zorder=6)
        ax.plot([0, 0], [-1, 1], 'k-', lw=0.3, alpha=0.3, zorder=6)

        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_aspect('equal')
        ax.set_title(view_label, fontsize=9)

        # Axis labels
        if vi >= 2:
            ax.set_xlabel(r'$x$ / $R_\mathrm{Europa}$')
        else:
            ax.set_xticklabels([])
        if vi % 2 == 0:
            ax.set_ylabel(r'$y$ / $R_\mathrm{Europa}$')
        else:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=10, vmax=13),
                           cmap='Blues')
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'log$_{10}$ column density (cm$^{-2}$)', fontsize=9)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2b: Species breakdown limb profiles (longitude-averaged)
# ---------------------------------------------------------------------------

def fig_species_limb(species_data, lats, face_areas, save_path):
    """Tangent column density by species, averaged over all longitudes."""
    n_lat_bins = 90
    lat_edges = np.linspace(-90, 90, n_lat_bins + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    # Bin pre-computed n0 and H by latitude (all longitudes)
    species_profiles = {}
    for key, sd in species_data.items():
        n0_profile = np.zeros(n_lat_bins)
        H_profile = np.zeros(n_lat_bins)
        for li in range(n_lat_bins):
            mask = (lats >= lat_edges[li]) & (lats < lat_edges[li + 1])
            if not np.any(mask):
                continue
            areas = face_areas[mask]
            w = areas / areas.sum()
            n0_profile[li] = np.sum(sd['n0'][mask] * w)
            H_profile[li] = np.sum(sd['H'][mask] * w)

        species_profiles[key] = {'n0': n0_profile, 'H': H_profile,
                                 'mass': sd['mass']}

    # Image grid
    r_max = 1.35
    n_pix = 401
    xy = np.linspace(-r_max, r_max, n_pix)
    X, Y = np.meshgrid(xy, xy)
    R_img = np.sqrt(X**2 + Y**2)
    lat_img = np.degrees(np.arcsin(np.clip(
        Y / np.maximum(R_img, 1e-10), -1, 1)))
    h_img = (R_img - 1.0) * R_EUROPA
    outside = R_img > 1.0
    lat_pix = lat_img[outside]
    h_pix = h_img[outside]

    # Compute per-species column density images
    panels = [
        ('Total (H$_2$O + O$_2$ + H$_2$)', None),
        ('H$_2$O', 'H$_2$O'),
        ('O$_2$', 'O$_2$'),
        ('H$_2$', 'H$_2$'),
    ]

    species_images = {}
    for key, prof in species_profiles.items():
        N_pix = np.zeros(lat_pix.shape)
        n0_interp = np.interp(lat_pix, lat_centers, prof['n0'],
                              left=prof['n0'][0], right=prof['n0'][-1])
        H_interp = np.interp(lat_pix, lat_centers, prof['H'],
                             left=prof['H'][0], right=prof['H'][-1])
        valid = (n0_interp > 0) & (H_interp > 0)
        N_pix[valid] = tangent_column_density(
            n0_interp[valid], H_interp[valid], h_pix[valid])
        species_images[key] = N_pix

    # Total
    N_total = sum(species_images.values())

    vmin, vmax = 10, 13

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5),
                              gridspec_kw={'hspace': 0.22, 'wspace': 0.05})
    axes = axes.ravel()

    for pi, (label, sp_key) in enumerate(panels):
        ax = axes[pi]
        N_pix_sel = N_total if sp_key is None else species_images[sp_key]
        N_cm2 = np.maximum(N_pix_sel * 1e-4, 1.0)
        N_img = np.full_like(R_img, np.nan)
        N_img[outside] = np.log10(N_cm2)

        ax.pcolormesh(xy, xy, N_img, cmap='Blues',
                      vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        disk = patches.Circle((0, 0), 1.0, fc='0.85', ec='0.3', lw=0.8,
                               zorder=5)
        ax.add_patch(disk)
        ax.plot([-1, 1], [0, 0], 'k-', lw=0.3, alpha=0.3, zorder=6)
        ax.plot([0, 0], [-1, 1], 'k-', lw=0.3, alpha=0.3, zorder=6)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=9)

        if pi >= 2:
            ax.set_xlabel(r'$x$ / $R_\mathrm{Europa}$')
        else:
            ax.set_xticklabels([])
        if pi % 2 == 0:
            ax.set_ylabel(r'$y$ / $R_\mathrm{Europa}$')
        else:
            ax.set_yticklabels([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax),
                           cmap='Blues')
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'log$_{10}$ column density (cm$^{-2}$)', fontsize=9)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2c: H₂O sublimation vs sputtering (longitude-averaged)
# ---------------------------------------------------------------------------

def fig_h2o_breakdown(n_sub, n_sput, H_h2o, lats, face_areas, save_path):
    """H₂O column density: total, sublimation, and sputtering contributions.

    n_sub, n_sput, H_h2o: diurnal-averaged per-face arrays.
    """
    n_lat_bins = 90
    lat_edges = np.linspace(-90, 90, n_lat_bins + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    n_total = n_sub + n_sput

    # Bin by latitude
    components = {
        'Total H$_2$O': n_total,
        'Sublimation': n_sub,
        'Sputtering': n_sput,
    }
    profiles = {}
    H_profile = np.zeros(n_lat_bins)
    for li in range(n_lat_bins):
        mask = (lats >= lat_edges[li]) & (lats < lat_edges[li + 1])
        if not np.any(mask):
            continue
        areas = face_areas[mask]
        w = areas / areas.sum()
        H_profile[li] = np.sum(H_h2o[mask] * w)
        for name, n_arr in components.items():
            if name not in profiles:
                profiles[name] = np.zeros(n_lat_bins)
            profiles[name][li] = np.sum(n_arr[mask] * w)

    # Image grid
    r_max = 1.35
    n_pix = 401
    xy = np.linspace(-r_max, r_max, n_pix)
    X, Y = np.meshgrid(xy, xy)
    R_img = np.sqrt(X**2 + Y**2)
    lat_img = np.degrees(np.arcsin(np.clip(
        Y / np.maximum(R_img, 1e-10), -1, 1)))
    h_img = (R_img - 1.0) * R_EUROPA
    outside = R_img > 1.0
    lat_pix = lat_img[outside]
    h_pix = h_img[outside]

    # Compute column density images for each component
    col_images = {}
    for name, n0_prof in profiles.items():
        N_pix = np.zeros(lat_pix.shape)
        n0_interp = np.interp(lat_pix, lat_centers, n0_prof,
                              left=n0_prof[0], right=n0_prof[-1])
        H_interp = np.interp(lat_pix, lat_centers, H_profile,
                             left=H_profile[0], right=H_profile[-1])
        valid = (n0_interp > 0) & (H_interp > 0)
        N_pix[valid] = tangent_column_density(
            n0_interp[valid], H_interp[valid], h_pix[valid])
        col_images[name] = N_pix

    # Ratio panel: sublimation / sputtering
    ratio_pix = np.full(lat_pix.shape, np.nan)
    valid = col_images['Sputtering'] > 0
    ratio_pix[valid] = (col_images['Sublimation'][valid] /
                        col_images['Sputtering'][valid])

    panels = [
        ('Total H$_2$O', 'col', 'Total H$_2$O'),
        ('Sublimation', 'col', 'Sublimation'),
        ('Sputtering', 'col', 'Sputtering'),
        ('Sublimation / Sputtering', 'ratio', None),
    ]

    vmin, vmax = 10, 13

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5),
                              gridspec_kw={'hspace': 0.22, 'wspace': 0.05})
    axes = axes.ravel()

    for pi, (label, ptype, comp_key) in enumerate(panels):
        ax = axes[pi]

        if ptype == 'col':
            N_cm2 = np.maximum(col_images[comp_key] * 1e-4, 1.0)
            N_img = np.full_like(R_img, np.nan)
            N_img[outside] = np.log10(N_cm2)
            ax.pcolormesh(xy, xy, N_img, cmap='Blues',
                          vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        else:
            R_img_val = np.full_like(R_img, np.nan)
            R_img_val[outside] = np.log10(np.maximum(ratio_pix, 1e-6))
            ax.pcolormesh(xy, xy, R_img_val, cmap='RdBu',
                          vmin=-3, vmax=3, shading='auto', rasterized=True)

        disk = patches.Circle((0, 0), 1.0, fc='0.85', ec='0.3', lw=0.8,
                               zorder=5)
        ax.add_patch(disk)
        ax.plot([-1, 1], [0, 0], 'k-', lw=0.3, alpha=0.3, zorder=6)
        ax.plot([0, 0], [-1, 1], 'k-', lw=0.3, alpha=0.3, zorder=6)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=9)

        if pi >= 2:
            ax.set_xlabel(r'$x$ / $R_\mathrm{Europa}$')
        else:
            ax.set_xticklabels([])
        if pi % 2 == 0:
            ax.set_ylabel(r'$y$ / $R_\mathrm{Europa}$')
        else:
            ax.set_yticklabels([])

    # Column density colorbar (left side, for panels a-c)
    cbar_ax1 = fig.add_axes([0.92, 0.52, 0.015, 0.35])
    sm1 = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax),
                            cmap='Blues')
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label(r'log$_{10}$ $N$ (cm$^{-2}$)', fontsize=8)

    # Ratio colorbar (for panel d)
    cbar_ax2 = fig.add_axes([0.92, 0.10, 0.015, 0.35])
    sm2 = cm.ScalarMappable(norm=colors.Normalize(vmin=-3, vmax=3),
                            cmap='RdBu')
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
    cbar2.set_label(r'log$_{10}$ (sub / sput)', fontsize=8)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3: Scale height Mollweide
# ---------------------------------------------------------------------------

def plot_mollweide(ax, lat_grid, lon_grid, data_grid, cmap, norm,
                   lon_center=0):
    """Plot data on a Mollweide axis. Reuses logic from plot_ice_transport.py."""
    lon_shifted = lon_grid - lon_center
    lon_shifted = (lon_shifted + 180) % 360 - 180
    order = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[order]
    data_sorted = data_grid[:, order]

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_sorted)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    pcm = ax.pcolormesh(LON, LAT, data_sorted, cmap=cmap, norm=norm,
                        shading='auto', rasterized=True)
    ax.grid(True, color='0.3', linewidth=0.3, alpha=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return pcm


def add_mollweide_labels(ax, lon_center=0):
    """Add lon/lat labels. Reuses logic from plot_ice_transport.py."""
    for lat_deg in [-60, -30, 0, 30, 60]:
        ax.text(np.deg2rad(185 - lon_center), np.deg2rad(lat_deg),
                f'{abs(lat_deg)}\u00b0{"N" if lat_deg > 0 else "S" if lat_deg < 0 else ""}',
                fontsize=5.5, color='0.4', ha='left', va='center',
                transform=ax.transData)
    for lon_deg in [0, 90, 180, 270]:
        lon_plot = lon_deg - lon_center
        if lon_plot > 180:
            lon_plot -= 360
        elif lon_plot < -180:
            lon_plot += 360
        if abs(lon_plot) > 170:
            continue
        ax.text(np.deg2rad(lon_plot), np.deg2rad(-8),
                f'{lon_deg}\u00b0E',
                fontsize=5.5, color='0.4', ha='center', va='top',
                transform=ax.transData)


def fig_scale_height(T_mean, lats, lons, save_path):
    """Mollweide map of H₂O scale height."""
    H_km = scale_height(T_mean) / 1e3  # convert m → km
    lat_g, lon_g, data_g = regrid_to_latlon(lats, lons, H_km)

    vmin, vmax = np.nanpercentile(data_g, [1, 99])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(5.5, 3.0))
    ax = fig.add_subplot(111, projection='mollweide')
    pcm = plot_mollweide(ax, lat_g, lon_g, data_g, cmap, norm)
    add_mollweide_labels(ax)

    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.04, aspect=35)
    cbar.set_label('H$_2$O scale height (km)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, save_path):
    """Save figure in both PNG and PDF."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    base = save_path.rsplit('.', 1)[0]
    for ext in ['.png', '.pdf']:
        fig.savefig(base + ext)
    print(f'  Saved {base}.png / .pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diurnal surface pressure and limb column density figures')
    parser.add_argument('--ptm-dir', required=True,
                        help='PTM3D output directory with temperature/ and mesh.ply')
    parser.add_argument('--save-dir', default='docs/figures/transport_averaged',
                        help='Output directory for figures')
    parser.add_argument('--hemisphere-contrast', type=float, default=3.0,
                        help='Sputtering trailing/leading ratio (default: 3.0)')
    parser.add_argument('--sputtering-scale', type=float, default=1.0,
                        help='Scale factor for sputtering rate (default: 1.0)')
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    global HEMISPHERE_CONTRAST, GLOBAL_H2O_RATE
    HEMISPHERE_CONTRAST = args.hemisphere_contrast
    GLOBAL_H2O_RATE *= args.sputtering_scale

    os.makedirs(args.save_dir, exist_ok=True)

    # Load mesh
    print("Loading mesh...")
    vertices, face_indices, lats, lons = read_ply_mesh(
        os.path.join(args.ptm_dir, 'mesh.ply'))
    face_areas = compute_face_areas(vertices, face_indices)
    n_faces = len(lats)
    print(f"  {n_faces} faces, total area {face_areas.sum():.4e} m²")

    # Discover temperature files
    temp_files = discover_temperature_files(args.ptm_dir)
    if not temp_files:
        print("ERROR: No temperature files found in "
              f"{os.path.join(args.ptm_dir, 'temperature/')}")
        sys.exit(1)
    print(f"  {len(temp_files)} temperature timesteps found")

    # Pre-compute spatial weights
    spatial_w = sputtering_spatial_weight(lats, lons)

    # --- Figure 1: Diurnal pressure ---
    print("\nFigure 1: Diurnal surface pressure...")
    fig_diurnal_pressure(
        temp_files, lats, lons, face_areas,
        save_path=os.path.join(args.save_dir, 'diurnal_surface_pressure.png'))

    # --- Compute diurnal-averaged quantities for Figures 2 & 3 ---
    # IMPORTANT: Because vapor pressure is exponentially sensitive to T,
    # we must average n_sub(T) over timesteps, NOT compute n_sub(T_mean).
    # Jensen's inequality: <n(T)> >> n(<T>) for convex n(T).
    print("\nComputing diurnal-averaged densities and scale heights...")
    n_steps = len(temp_files)
    T_sum = np.zeros(n_faces)
    # Per-species accumulators: diurnal-mean n and H
    n_sub_sum = np.zeros(n_faces)       # H₂O sublimation
    species_n_sum = {k: np.zeros(n_faces) for k in SPECIES}
    species_H_sum = {k: np.zeros(n_faces) for k in SPECIES}

    for ti, fpath in enumerate(temp_files):
        T = read_binary_field(fpath)
        T_sum += T
        n_sub_sum += number_density_sublimation(T)
        for key, sp in SPECIES.items():
            species_n_sum[key] += sputtering_number_density(
                T, spatial_w, species_key=key)
            species_H_sum[key] += scale_height(T, m=sp['mass'])
        if (ti + 1) % 50 == 0 or ti == n_steps - 1:
            print(f"    timestep {ti+1}/{n_steps}")

    T_mean = T_sum / n_steps
    n_sub_mean = n_sub_sum / n_steps
    species_n_mean = {k: v / n_steps for k, v in species_n_sum.items()}
    species_H_mean = {k: v / n_steps for k, v in species_H_sum.items()}

    print(f"  T_mean range: {T_mean.min():.1f} – {T_mean.max():.1f} K")
    print(f"  <n_sub(T)> max: {n_sub_mean.max():.2e} m⁻³  "
          f"(vs n_sub(T_mean) max: {number_density_sublimation(T_mean).max():.2e})")

    # Build diurnal-averaged species data dict for figure functions
    diurnal_species = {}
    for key in SPECIES:
        n_sput = species_n_mean[key]
        H = species_H_mean[key]
        if key == 'H$_2$O':
            n_total = n_sub_mean + n_sput
        else:
            n_total = n_sput
        diurnal_species[key] = {'n0': n_total, 'H': H,
                                'mass': SPECIES[key]['mass']}

    # --- Figure 2: Limb profiles ---
    print("\nFigure 2: Tangent column density limb profiles...")
    fig_limb_profiles(
        diurnal_species, lats, lons, face_areas,
        save_path=os.path.join(args.save_dir, 'tangent_column_density.png'))

    # --- Figure 2b: Species breakdown ---
    print("\nFigure 2b: Species breakdown limb profiles...")
    fig_species_limb(
        diurnal_species, lats, face_areas,
        save_path=os.path.join(args.save_dir, 'tangent_column_species.png'))

    # --- Figure 2c: H₂O sublimation vs sputtering ---
    print("\nFigure 2c: H₂O sublimation vs sputtering...")
    fig_h2o_breakdown(
        n_sub_mean, species_n_mean['H$_2$O'], species_H_mean['H$_2$O'],
        lats, face_areas,
        save_path=os.path.join(args.save_dir, 'tangent_column_h2o_breakdown.png'))

    # --- Figure 3: Scale height Mollweide ---
    print("\nFigure 3: Scale height Mollweide...")
    fig_scale_height(
        T_mean, lats, lons,
        save_path=os.path.join(args.save_dir, 'scale_height_mollweide.png'))

    print(f"\nDone! Figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
