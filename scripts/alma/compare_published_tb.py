#!/usr/bin/env python3
"""
Compare PTM3D predicted brightness temperatures with published ALMA values.

For each observation epoch with published Tb:
  1. Load PTM3D depth profiles from --locations/--depths output
  2. Compute disk-averaged predicted Tb via radiative transfer
  3. Compare with published values from Thelen et al. (2024)
  4. Compute statistics (chi-squared, bias, RMS)

Usage:
  python compare_published_tb.py --ptm3d-dir output/alma   # After PTM3D runs
  python compare_published_tb.py --demo                     # Demo with synthetic T(z)
  python compare_published_tb.py --sensitivity              # Sweep emissivity, porosity
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from setup_ptm3d_runs import (get_published_tb, THELEN_GLOBAL_AVG,
                               TRUMBO_BEST_FIT)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────
# Disk-averaged Tb computation
# ─────────────────────────────────────────────────────────────────────

def emission_angle_cos(lat_deg, lon_deg, sub_obs_lat_deg, sub_obs_lon_deg):
    """Compute cosine of emission angle for a surface point.

    Parameters
    ----------
    lat_deg, lon_deg : float or array
        Surface point latitude and longitude (degrees).
    sub_obs_lat_deg, sub_obs_lon_deg : float
        Sub-observer latitude and longitude (degrees West).

    Returns
    -------
    float or array
        cos(emission_angle). Negative values indicate the far hemisphere.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat0 = np.radians(sub_obs_lat_deg)
    lon0 = np.radians(sub_obs_lon_deg)
    return (np.sin(lat0) * np.sin(lat) +
            np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0))


def synthetic_tz_profile(T_surface, z, thermal_inertia=95.0,
                         period_s=3.55 * 86400, local_time_hrs=12.0):
    """Generate a synthetic T(z) profile for a surface point.

    Uses the thermal skin depth model: temperature oscillations decay
    exponentially with depth. Below ~3 skin depths, T ≈ T_mean.

    Parameters
    ----------
    T_surface : float
        Surface temperature (K).
    z : array
        Depth grid (m).
    thermal_inertia : float
        Thermal inertia (J m⁻² K⁻¹ s⁻¹/²).
    period_s : float
        Rotation period (seconds).
    local_time_hrs : float
        Local time in hours (0=midnight, 12=noon).

    Returns
    -------
    array
        Temperature profile T(z).
    """
    # Europa ice properties
    rho = 500.0      # density (kg/m³) for porous ice
    cp = 800.0       # heat capacity (J/kg/K)
    k = thermal_inertia**2 / (rho * cp)  # thermal conductivity

    # Thermal skin depth
    omega = 2 * np.pi / period_s
    delta = np.sqrt(2 * k / (rho * cp * omega))

    # Diurnal temperature amplitude (roughly ΔT ≈ T_max - T_mean)
    # For Europa, diurnal amplitude is ~30-40 K
    T_mean = T_surface * 0.75   # rough estimate of diurnal mean
    T_amp = T_surface - T_mean

    # Phase: noon = max, midnight = min
    phase = 2 * np.pi * (local_time_hrs / 24.0 - 0.5)

    T = T_mean + T_amp * np.exp(-z / delta) * np.cos(phase - z / delta)
    return T


def compute_predicted_tb_synthetic(obs, porosity=0.5, emissivity="fresnel",
                                   thermal_inertia=95.0, n_lat=20, n_lon=36):
    """Compute predicted disk-averaged Tb using synthetic T(z) profiles.

    This is used before PTM3D runs are available, for verification and
    sensitivity studies.

    Parameters
    ----------
    obs : dict
        Observation metadata (sub_obs_lon, sub_obs_lat, freq_ghz, etc.).
    porosity : float
        Ice porosity (0-1).
    emissivity : float
        Surface emissivity.
    thermal_inertia : float
        Thermal inertia (SI units).
    n_lat, n_lon : int
        Number of latitude/longitude grid points.

    Returns
    -------
    float
        Predicted disk-averaged Tb (K).
    """
    from forward_model import radiative_transfer_emission_angle
    from ice_dielectric import absorption_coefficient, porous_ice_permittivity

    freq_ghz = obs["freq_ghz"]
    sub_lon = obs["sub_obs_lon"]
    sub_lat = obs["sub_obs_lat"]

    # Generate a grid covering the visible hemisphere
    lats = np.linspace(-85, 85, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)

    # Depth grid: 0 to 50 cm, geometric spacing near surface
    n_z = 50
    z = np.concatenate([[0], np.geomspace(0.001, 0.5, n_z - 1)])

    # Europa Bond albedo (average ~0.5, varies with hemisphere)
    A_mean = 0.50
    solar_constant = 1361.0 / 5.2**2  # W/m² at Jupiter

    weighted_tb_sum = 0.0
    weight_sum = 0.0

    for lat in lats:
        for lon in lons:
            cos_em = emission_angle_cos(lat, lon, sub_lat, sub_lon)
            if cos_em <= 0:
                continue  # Far hemisphere, not visible

            # Surface temperature from simple thermal model
            # cos(incidence) for sub-solar point
            cos_inc = np.cos(np.radians(lat)) * np.cos(np.radians(lon - sub_lon))
            cos_inc = max(cos_inc, 0)

            # Equilibrium daytime temperature (bolometric emissivity = 0.9)
            T_eq = ((solar_constant * (1 - A_mean) * cos_inc) /
                    (0.9 * 5.67e-8)) ** 0.25 if cos_inc > 0 else 50.0

            # Generate synthetic T(z)
            T_z = synthetic_tz_profile(T_eq, z, thermal_inertia=thermal_inertia)
            T_z = np.maximum(T_z, 40.0)  # Floor at ~40 K (nightside)

            # Emission angle for this point
            em_angle = np.degrees(np.arccos(cos_em))

            # Radiative transfer with emission angle correction
            Tb_face = radiative_transfer_emission_angle(
                z, T_z, freq_ghz, porosity, em_angle,
                emissivity, mixing="maxwell_garnett"
            )

            # Weight by projected area
            cos_lat = np.cos(np.radians(lat))
            weight = cos_em * cos_lat
            weighted_tb_sum += Tb_face * weight
            weight_sum += weight

    if weight_sum > 0:
        return weighted_tb_sum / weight_sum
    return 0.0


def parse_location_name(name):
    """Parse lat/lon from PTM3D location filename encoding.

    Filenames use: lat_m012d1_lon_0150d0 where m=minus, p=plus, d=decimal.

    Returns
    -------
    tuple (lat, lon) in degrees, or None if parsing fails.
    """
    import re
    # Match lat_{sign}{digits}d{digits}_lon_{digits}d{digits}
    m = re.match(r"lat_([mp])(\d+)d(\d+)_lon_(\d+)d(\d+)", name)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "m" else 1.0
    lat = sign * float(f"{m.group(2)}.{m.group(3)}")
    lon = float(f"{m.group(4)}.{m.group(5)}")
    return lat, lon


def load_ptm3d_depth_csv(csv_path, depths_file=None, orbit_steps=288):
    """Load a PTM3D location depth profile CSV.

    Format:
        # Location: lat_p000d0_lon_0225d0
        time,T(z=0.000 m),T(z=0.000 m),...,T(z=0.500 m)
        2016-09-19T12:00:00.000,103.048,...

    Note: PTM3D outputs depth profiles for all timesteps including
    equilibration orbits. We use the orbit-mean T(z) from the last orbit
    (the output window after equilibration), which has fully equilibrated
    temperatures.

    Parameters
    ----------
    csv_path : Path
        Path to the depth profile CSV.
    depths_file : Path, optional
        Path to the depths file with exact depth values. If provided,
        these are used instead of parsing from the CSV header (which
        rounds sub-mm depths to 0.000 m).
    orbit_steps : int
        Number of timesteps per orbit (default: 288).

    Returns
    -------
    z : ndarray
        Depth values in meters.
    T_z : ndarray
        Orbit-mean temperature profile (K).
    """
    with open(csv_path) as f:
        lines = f.readlines()

    # Find header line (skip comment lines starting with #)
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header_idx = i
            break

    header = lines[header_idx].strip()
    n_depth_cols = len(header.split(",")) - 1  # subtract time column

    # Get depth values from the depths file (exact) or header (rounded)
    if depths_file is not None and Path(depths_file).exists():
        z = np.loadtxt(str(depths_file))
        if len(z) != n_depth_cols:
            raise ValueError(
                f"Depths file has {len(z)} values but CSV has "
                f"{n_depth_cols} depth columns"
            )
    else:
        import re
        z = []
        for col in header.split(",")[1:]:
            m = re.search(r"z=([\d.]+)\s*m", col)
            if m:
                z.append(float(m.group(1)))
        z = np.array(z)

    # Parse temperature data (skip datetime column)
    data_lines = [l.strip() for l in lines[header_idx + 1:] if l.strip()]
    if not data_lines:
        return z, None

    # Use last orbit (output window after equilibration)
    n_use = min(orbit_steps, len(data_lines))
    temps = []
    for line in data_lines[-n_use:]:
        parts = line.split(",")
        temps.append([float(v) for v in parts[1:]])
    temps = np.array(temps)

    # Return orbit-mean T(z) profile
    return z, temps.mean(axis=0)


def compute_predicted_tb_from_ptm3d(obs, ptm3d_dir, porosity=0.5,
                                     emissivity="fresnel", depths_file=None):
    """Compute predicted disk-averaged Tb from PTM3D output.

    Loads depth profiles from PTM3D location CSV files and applies
    radiative transfer to compute disk-averaged Tb.

    Parameters
    ----------
    obs : dict
        Observation metadata.
    ptm3d_dir : Path
        PTM3D output directory for this epoch.
    porosity : float
        Ice porosity.
    emissivity : float or "fresnel"
        Surface emissivity.  "fresnel" (default) computes the
        angle-dependent Fresnel emissivity from the dielectric model.
    depths_file : Path, optional
        Path to depths file with exact depth values.

    Returns
    -------
    float or None
        Predicted Tb, or None if PTM3D output not found.
    """
    from forward_model import radiative_transfer_emission_angle

    loc_dir = ptm3d_dir / "locations"
    if not loc_dir.exists():
        return None

    freq_ghz = obs["freq_ghz"]
    sub_lon = obs["sub_obs_lon"]
    sub_lat = obs["sub_obs_lat"]

    weighted_tb_sum = 0.0
    weight_sum = 0.0

    for csv_path in sorted(loc_dir.glob("*_depths.csv")):
        name = csv_path.stem.replace("_depths", "")

        # Parse lat/lon from encoded filename
        coords = parse_location_name(name)
        if coords is None:
            continue
        lat, lon = coords

        # Load depth profile with exact depths
        z, T_z = load_ptm3d_depth_csv(csv_path, depths_file=depths_file)
        if T_z is None or len(T_z) == 0:
            continue

        cos_em = emission_angle_cos(lat, lon, sub_lat, sub_lon)
        if cos_em <= 0:
            continue

        em_angle = np.degrees(np.arccos(min(cos_em, 1.0)))
        Tb_face = radiative_transfer_emission_angle(
            z, T_z, freq_ghz, porosity, em_angle,
            emissivity, mixing="maxwell_garnett"
        )

        cos_lat = np.cos(np.radians(lat))
        weight = cos_em * cos_lat
        weighted_tb_sum += Tb_face * weight
        weight_sum += weight

    if weight_sum > 0:
        return weighted_tb_sum / weight_sum
    return None


# ─────────────────────────────────────────────────────────────────────
# Comparison statistics
# ─────────────────────────────────────────────────────────────────────

def compute_statistics(results):
    """Compute comparison statistics.

    Parameters
    ----------
    results : list of dict
        Each dict has: tb_predicted, tb_published, tb_error.

    Returns
    -------
    dict
        chi2_r, bias, rms, pearson_r, n_points
    """
    obs = np.array([r["tb_published"] for r in results])
    pred = np.array([r["tb_predicted"] for r in results])
    err = np.array([r["tb_error"] for r in results])

    residuals = pred - obs
    n = len(results)

    chi2 = np.sum(residuals**2 / err**2)
    chi2_r = chi2 / n

    bias = np.mean(residuals)
    rms = np.sqrt(np.mean(residuals**2))

    if n > 2:
        r = np.corrcoef(obs, pred)[0, 1]
    else:
        r = np.nan

    return {
        "chi2_r": chi2_r,
        "bias": bias,
        "rms": rms,
        "pearson_r": r,
        "n_points": n,
    }


def print_comparison_table(results, stats):
    """Print a formatted comparison table."""
    print(f"\n  {'Label':>6} {'Band':>5} {'Hem':>10} {'Lon':>6} "
          f"{'Tb_obs':>8} {'Tb_pred':>8} {'Resid':>7} {'σ':>5}")
    print(f"  {'─'*65}")

    for r in results:
        resid = r["tb_predicted"] - r["tb_published"]
        n_sigma = resid / r["tb_error"]
        print(f"  {r['label']:>6} {r['band']:>5} {r['hemisphere']:>10} "
              f"{r['sub_obs_lon']:>5.0f}° "
              f"{r['tb_published']:>7.1f}K {r['tb_predicted']:>7.1f}K "
              f"{resid:>+6.1f}K {n_sigma:>+4.1f}σ")

    print(f"\n  Statistics (N={stats['n_points']}):")
    print(f"    χ²ᵣ  = {stats['chi2_r']:.2f}")
    print(f"    Bias = {stats['bias']:+.1f} K")
    print(f"    RMS  = {stats['rms']:.1f} K")
    print(f"    r    = {stats['pearson_r']:.3f}")


def sensitivity_grid(observations, emissivities, porosities,
                     thermal_inertia=95.0):
    """Sweep emissivity and porosity to find best fit.

    Parameters
    ----------
    observations : list
        Observations with published Tb.
    emissivities : array
        Emissivity values to try.
    porosities : array
        Porosity values to try.
    thermal_inertia : float
        Fixed thermal inertia.

    Returns
    -------
    dict
        Grid of chi2_r values and best-fit parameters.
    """
    chi2_grid = np.zeros((len(emissivities), len(porosities)))

    for i, eps in enumerate(emissivities):
        for j, phi in enumerate(porosities):
            results = []
            for obs in observations:
                tb_pred = compute_predicted_tb_synthetic(
                    obs, porosity=phi, emissivity=eps,
                    thermal_inertia=thermal_inertia,
                    n_lat=10, n_lon=18  # Coarser grid for speed
                )
                results.append({
                    "tb_predicted": tb_pred,
                    "tb_published": obs["tb_published"],
                    "tb_error": obs["tb_error"],
                })
            stats = compute_statistics(results)
            chi2_grid[i, j] = stats["chi2_r"]

    best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    return {
        "chi2_grid": chi2_grid,
        "emissivities": emissivities,
        "porosities": porosities,
        "best_emissivity": emissivities[best_idx[0]],
        "best_porosity": porosities[best_idx[1]],
        "best_chi2_r": chi2_grid[best_idx],
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PTM3D predictions with published ALMA Tb values"
    )
    parser.add_argument("--ptm3d-dir", type=Path,
                        help="PTM3D output base directory")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic T(z) profiles")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run emissivity/porosity sensitivity grid")
    parser.add_argument("--porosity", type=float, default=0.5,
                        help="Ice porosity (default: 0.5)")
    parser.add_argument("--emissivity", default="fresnel",
                        help="Emissivity: 'fresnel' (default) or a float value")
    parser.add_argument("--thermal-inertia", type=float, default=95.0,
                        help="Thermal inertia in SI (default: 95)")
    parser.add_argument("--depths-file", type=Path,
                        default=PROJECT_ROOT / "configs" / "alma" / "alma_depths.txt",
                        help="Depths file with exact depth values")
    parser.add_argument("--save-json", type=Path, default=None,
                        help="Save results as JSON for plotting")
    args = parser.parse_args()

    # Parse emissivity: "fresnel" string or numeric value
    if args.emissivity != "fresnel":
        try:
            args.emissivity = float(args.emissivity)
        except ValueError:
            parser.error(f"--emissivity must be 'fresnel' or a number, got '{args.emissivity}'")

    observations = get_published_tb()
    print(f"\n  Published ALMA Tb values: {len(observations)} observations")
    eps_str = "Fresnel" if args.emissivity == "fresnel" else f"{args.emissivity}"
    print(f"  Parameters: ε={eps_str}, φ={args.porosity}, "
          f"Γ={args.thermal_inertia} SI")

    if args.demo or args.ptm3d_dir is None:
        print(f"\n  Mode: Synthetic T(z) profiles (demo)")
        results = []
        for obs in observations:
            tb_pred = compute_predicted_tb_synthetic(
                obs, porosity=args.porosity, emissivity=args.emissivity,
                thermal_inertia=args.thermal_inertia
            )
            results.append({
                "label": obs["label"],
                "band": obs["band"],
                "hemisphere": obs["hemisphere"],
                "sub_obs_lon": obs["sub_obs_lon"],
                "tb_predicted": tb_pred,
                "tb_published": obs["tb_published"],
                "tb_error": obs["tb_error"],
                "freq_ghz": obs["freq_ghz"],
                "reference": obs["reference"],
            })

        stats = compute_statistics(results)
        print_comparison_table(results, stats)

        if args.sensitivity:
            print(f"\n  Running sensitivity grid...")
            eps_range = np.arange(0.60, 0.95, 0.05)
            phi_range = np.arange(0.30, 0.75, 0.05)
            sens = sensitivity_grid(observations, eps_range, phi_range,
                                    args.thermal_inertia)
            print(f"  Best fit: ε={sens['best_emissivity']:.2f}, "
                  f"φ={sens['best_porosity']:.2f}, "
                  f"χ²ᵣ={sens['best_chi2_r']:.2f}")

    else:
        print(f"\n  Mode: PTM3D output from {args.ptm3d_dir}")
        results = []
        for obs in observations:
            epoch_dir = args.ptm3d_dir / f"alma_{obs['label']}"
            tb_pred = compute_predicted_tb_from_ptm3d(
                obs, epoch_dir, porosity=args.porosity,
                emissivity=args.emissivity,
                depths_file=args.depths_file
            )
            if tb_pred is None:
                print(f"  WARNING: No PTM3D output for {obs['label']}, skipping")
                continue

            results.append({
                "label": obs["label"],
                "band": obs["band"],
                "hemisphere": obs["hemisphere"],
                "sub_obs_lon": obs["sub_obs_lon"],
                "tb_predicted": tb_pred,
                "tb_published": obs["tb_published"],
                "tb_error": obs["tb_error"],
                "freq_ghz": obs["freq_ghz"],
                "reference": obs["reference"],
            })

        if results:
            stats = compute_statistics(results)
            print_comparison_table(results, stats)

    # Save results as JSON for plotting
    if args.save_json and results:
        import json
        out_dir = args.save_json.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"results": results, "stats": stats}, f, indent=2)
        print(f"\n  Saved: {args.save_json}")


if __name__ == "__main__":
    main()
