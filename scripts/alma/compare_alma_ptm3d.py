#!/usr/bin/env python3
"""
Quantitative comparison of ALMA observations with PTM3D predictions.

For each ALMA observation epoch:
  1. Load observed Tb map (from processed ALMA data)
  2. Load predicted Tb map (from PTM3D forward model output)
  3. Compute residuals, RMS, correlation, mean bias
  4. Generate comparison statistics and save results

Usage:
  python compare_alma_ptm3d.py --obs data/alma/trumbo2018/processed/ --model output/alma/
  python compare_alma_ptm3d.py --summary   # Print comparison table
"""

import argparse
import numpy as np
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup_ptm3d_runs import TRUMBO_OBS, THELEN_OBS, get_all_observations

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_alma_observation(npz_path):
    """Load processed ALMA observation from NPZ file.

    Expected contents (from alma_geometry.py process_fits):
        lat, lon, emission_angle, on_disk, tb,
        sub_obs_lon, sub_obs_lat, ang_diam, pixel_scale

    Parameters
    ----------
    npz_path : str or Path
        Path to geometry NPZ file.

    Returns
    -------
    dict
        Observation data.
    """
    data = np.load(str(npz_path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_ptm3d_prediction(npz_path):
    """Load PTM3D forward model prediction from NPZ file.

    Expected contents (from forward_model.py):
        Tb, face_lat, face_lon, freq_ghz, porosity, emissivity

    Parameters
    ----------
    npz_path : str or Path
        Path to prediction NPZ file.

    Returns
    -------
    dict
        Prediction data.
    """
    data = np.load(str(npz_path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def regrid_model_to_observation(model_tb, model_lat, model_lon,
                                obs_lat, obs_lon, method="nearest"):
    """Regrid model Tb onto the observation lat/lon grid.

    Parameters
    ----------
    model_tb : ndarray
        Model Tb values at mesh face centers.
    model_lat, model_lon : ndarray
        Model face center coordinates.
    obs_lat, obs_lon : ndarray
        Observation pixel lat/lon (2D arrays or 1D extracted).
    method : str
        Interpolation method: "nearest" or "linear".

    Returns
    -------
    ndarray
        Model Tb regridded to observation coordinates.
    """
    from scipy.interpolate import griddata

    # Flatten observation coordinates if needed
    flat_lat = obs_lat.ravel()
    flat_lon = obs_lon.ravel()

    # Filter out NaN locations (off-disk pixels)
    valid = np.isfinite(flat_lat) & np.isfinite(flat_lon)

    model_tb_regrid = np.full_like(flat_lat, np.nan)

    if np.sum(valid) > 0:
        # Handle longitude wrapping (0/360 boundary)
        # Convert to Cartesian for interpolation
        model_x = np.cos(np.radians(model_lat)) * np.cos(np.radians(model_lon))
        model_y = np.cos(np.radians(model_lat)) * np.sin(np.radians(model_lon))
        model_z = np.sin(np.radians(model_lat))

        obs_x = np.cos(np.radians(flat_lat[valid])) * np.cos(np.radians(flat_lon[valid]))
        obs_y = np.cos(np.radians(flat_lat[valid])) * np.sin(np.radians(flat_lon[valid]))
        obs_z = np.sin(np.radians(flat_lat[valid]))

        model_points = np.column_stack([model_x, model_y, model_z])
        obs_points = np.column_stack([obs_x, obs_y, obs_z])

        model_tb_regrid[valid] = griddata(
            model_points, model_tb, obs_points, method=method
        )

    return model_tb_regrid.reshape(obs_lat.shape)


def compute_comparison_stats(obs_tb, model_tb, emission_angle=None,
                             max_emission_angle=80.0):
    """Compute comparison statistics between observed and modeled Tb.

    Parameters
    ----------
    obs_tb, model_tb : ndarray
        Observed and modeled brightness temperatures (K).
    emission_angle : ndarray, optional
        Emission angles (degrees) for weighting.
    max_emission_angle : float
        Maximum emission angle to include.

    Returns
    -------
    dict
        Statistics: rms, bias, correlation, max_residual, n_valid, etc.
    """
    # Mask invalid pixels
    valid = np.isfinite(obs_tb) & np.isfinite(model_tb)
    if emission_angle is not None:
        valid &= (emission_angle < max_emission_angle)

    if np.sum(valid) < 5:
        return {"n_valid": int(np.sum(valid)), "error": "Too few valid pixels"}

    obs = obs_tb[valid]
    mod = model_tb[valid]
    residual = obs - mod

    stats = {
        "n_valid": int(np.sum(valid)),
        "obs_mean": float(np.mean(obs)),
        "obs_std": float(np.std(obs)),
        "model_mean": float(np.mean(mod)),
        "model_std": float(np.std(mod)),
        "bias": float(np.mean(residual)),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "max_residual": float(np.max(np.abs(residual))),
        "correlation": float(np.corrcoef(obs, mod)[0, 1]),
        "residual_std": float(np.std(residual)),
    }

    # Weighted statistics (weight by cos(emission_angle) if available)
    if emission_angle is not None:
        cos_em = np.cos(np.radians(emission_angle[valid]))
        w = cos_em / np.sum(cos_em)
        stats["weighted_bias"] = float(np.sum(w * residual))
        stats["weighted_rms"] = float(np.sqrt(np.sum(w * residual**2)))

    return stats


def compare_single_epoch(obs_data, model_data, obs_info, output_dir=None):
    """Compare a single ALMA observation with PTM3D prediction.

    Parameters
    ----------
    obs_data : dict
        Loaded ALMA observation data.
    model_data : dict
        Loaded PTM3D prediction data.
    obs_info : dict
        Observation metadata (label, reference, etc.).
    output_dir : Path, optional
        Directory to save comparison results.

    Returns
    -------
    dict
        Comparison results.
    """
    label = obs_info["label"]
    print(f"\n  {'='*60}")
    print(f"  Comparing {label}: {obs_info['reference']}")
    print(f"  Band {obs_info['band']} ({obs_info['freq_ghz']:.1f} GHz), "
          f"{obs_info['hemisphere']} hemisphere")
    print(f"  {'='*60}")

    obs_tb = obs_data.get("tb")
    obs_lat = obs_data.get("lat")
    obs_lon = obs_data.get("lon")
    obs_emang = obs_data.get("emission_angle")

    model_tb = model_data.get("Tb")
    model_lat = model_data.get("face_lat")
    model_lon = model_data.get("face_lon")

    if model_lat is None or model_lon is None:
        print("  ERROR: Model data missing face coordinates")
        return None

    # Regrid model to observation grid
    print("  Regridding model to observation coordinates...")
    model_tb_regrid = regrid_model_to_observation(
        model_tb, model_lat, model_lon, obs_lat, obs_lon
    )

    # Compute statistics
    stats = compute_comparison_stats(obs_tb, model_tb_regrid, obs_emang)
    stats["label"] = label
    stats["reference"] = obs_info["reference"]
    stats["band"] = obs_info["band"]
    stats["freq_ghz"] = obs_info["freq_ghz"]
    stats["hemisphere"] = obs_info["hemisphere"]

    # Print results
    print(f"  Valid pixels: {stats.get('n_valid', 0)}")
    if "error" not in stats:
        print(f"  Observed Tb:  {stats['obs_mean']:.1f} ± {stats['obs_std']:.1f} K")
        print(f"  Model Tb:     {stats['model_mean']:.1f} ± {stats['model_std']:.1f} K")
        print(f"  Bias:         {stats['bias']:+.2f} K")
        print(f"  RMS:          {stats['rms']:.2f} K")
        print(f"  Correlation:  {stats['correlation']:.4f}")
        print(f"  Max residual: {stats['max_residual']:.2f} K")

    # Save comparison results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(output_dir / f"comparison_{label}.npz"),
            obs_tb=obs_tb,
            model_tb_regrid=model_tb_regrid,
            residual=obs_tb - model_tb_regrid,
            obs_lat=obs_lat,
            obs_lon=obs_lon,
            emission_angle=obs_emang,
            **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
        )
        print(f"  Saved: {output_dir / f'comparison_{label}.npz'}")

    return stats


def compare_all(obs_dir, model_dir, output_dir=None):
    """Compare all available observation-model pairs.

    Parameters
    ----------
    obs_dir : str or Path
        Directory containing processed ALMA observations (*_geometry.npz).
    model_dir : str or Path
        Directory containing PTM3D predictions (*_Tb_*.npz).
    output_dir : str or Path, optional
        Directory for comparison outputs.

    Returns
    -------
    list
        List of comparison statistics dictionaries.
    """
    obs_dir = Path(obs_dir)
    model_dir = Path(model_dir)
    all_obs = get_all_observations()
    results = []

    for obs_info in all_obs:
        label = obs_info["label"]

        # Find observation file
        obs_files = list(obs_dir.rglob(f"*{label}*_geometry.npz"))
        if not obs_files:
            obs_files = list(obs_dir.rglob(f"*geometry*.npz"))

        # Find model file
        model_files = list(model_dir.rglob(f"*{label}*_Tb_*.npz"))

        if not obs_files:
            print(f"  {label}: No observation file found, skipping")
            continue
        if not model_files:
            print(f"  {label}: No model file found, skipping")
            continue

        obs_data = load_alma_observation(obs_files[0])
        model_data = load_ptm3d_prediction(model_files[0])

        stats = compare_single_epoch(obs_data, model_data, obs_info, output_dir)
        if stats is not None:
            results.append(stats)

    return results


def print_summary_table(results):
    """Print a summary comparison table.

    Parameters
    ----------
    results : list
        List of comparison statistics dictionaries.
    """
    if not results:
        print("  No comparison results available.")
        return

    print(f"\n  {'='*90}")
    print(f"  ALMA vs. PTM3D Comparison Summary")
    print(f"  {'='*90}")
    print(f"  {'Label':>6} {'Band':>5} {'Hemisphere':>11} "
          f"{'Obs <Tb>':>9} {'Mod <Tb>':>9} {'Bias':>7} {'RMS':>6} {'r':>6} {'N':>6}")
    print(f"  {'─'*80}")

    for s in results:
        if "error" in s:
            print(f"  {s['label']:>6} {s['band']:>5} {s['hemisphere']:>11} "
                  f"  --- error: {s['error']}")
            continue
        print(f"  {s['label']:>6} {s['band']:>5} {s['hemisphere']:>11} "
              f"{s['obs_mean']:>8.1f}K {s['model_mean']:>8.1f}K "
              f"{s['bias']:>+6.1f}K {s['rms']:>5.1f}K "
              f"{s['correlation']:>5.3f} {s['n_valid']:>6d}")

    # Overall statistics
    valid_results = [s for s in results if "error" not in s]
    if valid_results:
        avg_rms = np.mean([s["rms"] for s in valid_results])
        avg_bias = np.mean([s["bias"] for s in valid_results])
        avg_corr = np.mean([s["correlation"] for s in valid_results])
        print(f"  {'─'*80}")
        print(f"  {'MEAN':>6} {'':>5} {'':>11} "
              f"{'':>9} {'':>9} "
              f"{avg_bias:>+6.1f}K {avg_rms:>5.1f}K "
              f"{avg_corr:>5.3f}")


def sensitivity_analysis(z, T_profiles, face_lat, face_lon,
                         obs_data, obs_info, freq_ghz=233.0):
    """Run sensitivity analysis: vary emissivity, thermal inertia, porosity.

    Parameters
    ----------
    z : ndarray
        Depth grid.
    T_profiles : ndarray
        Temperature profiles for all faces.
    face_lat, face_lon : ndarray
        Face coordinates.
    obs_data : dict
        Observation data for comparison.
    obs_info : dict
        Observation metadata.
    freq_ghz : float
        ALMA frequency.

    Returns
    -------
    dict
        Sensitivity results (parameter grids and RMS values).
    """
    from forward_model import radiative_transfer_batch

    obs_tb = obs_data["tb"]
    obs_lat = obs_data["lat"]
    obs_lon = obs_data["lon"]

    # Parameter grids
    emissivities = np.arange(0.6, 0.95, 0.05)
    porosities = np.arange(0.3, 0.8, 0.05)

    rms_grid = np.zeros((len(emissivities), len(porosities)))

    for i, eps in enumerate(emissivities):
        for j, phi in enumerate(porosities):
            model_tb = radiative_transfer_batch(
                z, T_profiles, freq_ghz, phi, eps
            )
            model_regrid = regrid_model_to_observation(
                model_tb, face_lat, face_lon, obs_lat, obs_lon
            )
            stats = compute_comparison_stats(obs_tb, model_regrid)
            rms_grid[i, j] = stats.get("rms", np.nan)

    # Find best-fit parameters
    idx = np.unravel_index(np.nanargmin(rms_grid), rms_grid.shape)
    best_eps = emissivities[idx[0]]
    best_phi = porosities[idx[1]]
    best_rms = rms_grid[idx]

    print(f"\n  Sensitivity Analysis for {obs_info['label']}:")
    print(f"  Best-fit emissivity: {best_eps:.2f}")
    print(f"  Best-fit porosity:   {best_phi*100:.0f}%")
    print(f"  Minimum RMS:         {best_rms:.2f} K")

    return {
        "emissivities": emissivities,
        "porosities": porosities,
        "rms_grid": rms_grid,
        "best_emissivity": float(best_eps),
        "best_porosity": float(best_phi),
        "best_rms": float(best_rms),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare ALMA observations with PTM3D predictions"
    )
    parser.add_argument("--obs", default=str(PROJECT_ROOT / "data" / "alma"),
                        help="Directory with processed ALMA observations")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "output" / "alma"),
                        help="Directory with PTM3D predictions")
    parser.add_argument("-o", "--output",
                        default=str(PROJECT_ROOT / "data" / "alma" / "comparison"),
                        help="Output directory for comparison results")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table of existing comparisons")
    args = parser.parse_args()

    if args.summary:
        # Load existing comparison results
        comp_dir = Path(args.output)
        results = []
        for npz in sorted(comp_dir.glob("comparison_*.npz")):
            data = np.load(str(npz), allow_pickle=True)
            results.append({k: data[k].item() if data[k].ndim == 0 else data[k]
                            for k in data.files})
        print_summary_table(results)
        return

    results = compare_all(args.obs, args.model, args.output)
    print_summary_table(results)


if __name__ == "__main__":
    main()
