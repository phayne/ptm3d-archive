#!/usr/bin/env python3
"""
Batch process Trumbo et al. (2018) ALMA FITS files.

Steps:
  1. Convert PI pipeline Jy/beam images to brightness temperature (K)
  2. Project to Europa lat/lon using observation geometry
  3. Save processed NPZ files for comparison with PTM3D

Usage:
  python process_trumbo_data.py
  python process_trumbo_data.py --use-ari   # Use ARI reprocessed images instead
  python process_trumbo_data.py --planck     # Use full Planck inversion
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from alma_to_tb import convert_fits
from alma_geometry import process_fits as project_to_latlon
from setup_ptm3d_runs import TRUMBO_OBS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FITS_DIR = PROJECT_ROOT / "data" / "alma" / "trumbo2018" / "fits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "alma" / "trumbo2018" / "processed"

# Map observation labels to ALMA archive UIDs
UID_MAP = {
    "T1": "uid___A001_X2d8_X6d",
    "T2": "uid___A001_X2d8_X70",
    "T3": "uid___A001_X2d8_X73",
    "T4": "uid___A001_X2d8_X76",
}


def find_fits_file(label, use_ari=False):
    """Find the FITS file for a given observation label.

    Parameters
    ----------
    label : str
        Observation label (T1, T2, T3, T4).
    use_ari : bool
        If True, use ARI reprocessed images. Otherwise, use PI pipeline.

    Returns
    -------
    Path or None
        Path to FITS file if found.
    """
    uid = UID_MAP.get(label)
    if uid is None:
        return None

    files = sorted(FITS_DIR.glob(f"member.{uid}*"))
    if not files:
        return None

    if use_ari:
        ari_files = [f for f in files if ".ari_l." in f.name]
        if ari_files:
            return ari_files[0]
    else:
        pi_files = [f for f in files if "calibrated_final_cont" in f.name]
        if pi_files:
            return pi_files[0]

    return files[0]


def find_disk_center(data, disk_radius_pix=None):
    """Find Europa's disk center in an ALMA image.

    Strategy: find the peak pixel, then compute a flux-weighted centroid
    within a region around the peak using only bright pixels (>20% of peak).
    This avoids being pulled off-center by noise and sidelobes.

    Parameters
    ----------
    data : ndarray
        2D image data.
    disk_radius_pix : float, optional
        Expected disk radius in pixels (used to define search region).

    Returns
    -------
    tuple
        (center_x, center_y) in pixel coordinates, or (None, None).
    """
    valid = np.isfinite(data)
    if np.sum(valid) < 100:
        return None, None

    ny, nx = data.shape

    # Step 1: Find peak pixel
    work = np.where(valid, data, -np.inf)
    peak_idx = np.unravel_index(np.argmax(work), data.shape)
    peak_val = data[peak_idx]

    if peak_val <= 0:
        return None, None

    # Step 2: Compute centroid around peak using bright pixels only
    # Use a box of 3× disk radius (or 1/4 of image) around the peak
    if disk_radius_pix is not None:
        box_r = int(3 * disk_radius_pix)
    else:
        box_r = min(nx, ny) // 4

    y0 = max(0, peak_idx[0] - box_r)
    y1 = min(ny, peak_idx[0] + box_r)
    x0 = max(0, peak_idx[1] - box_r)
    x1 = min(nx, peak_idx[1] + box_r)

    sub = data[y0:y1, x0:x1]
    # Threshold at 20% of peak to select only Europa's disk emission
    bright = np.isfinite(sub) & (sub > 0.2 * peak_val)

    if np.sum(bright) < 5:
        # Fall back to peak position
        return float(peak_idx[1]), float(peak_idx[0])

    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    flux = np.where(bright, sub, 0)
    total = np.sum(flux)
    cx = np.sum(xx * flux) / total + x0
    cy = np.sum(yy * flux) / total + y0

    return float(cx), float(cy)


def process_observation(obs_info, use_ari=False, use_planck=False):
    """Process a single Trumbo observation through the full pipeline.

    Parameters
    ----------
    obs_info : dict
        Observation metadata from setup_ptm3d_runs.
    use_ari : bool
        Use ARI reprocessed images.
    use_planck : bool
        Use Planck inversion for Tb conversion.

    Returns
    -------
    dict or None
        Processing results.
    """
    label = obs_info["label"]
    fits_file = find_fits_file(label, use_ari=use_ari)

    if fits_file is None:
        print(f"\n  {label}: No FITS file found, skipping")
        return None

    print(f"\n{'='*70}")
    print(f"  Processing {label}: {obs_info['reference']}")
    print(f"  Epoch: {obs_info['epoch']}")
    print(f"  Sub-observer: lon={obs_info['sub_obs_lon']:.1f}°W, "
          f"lat={obs_info['sub_obs_lat']:.1f}°")
    print(f"  Hemisphere: {obs_info['hemisphere']}")
    print(f"  FITS: {fits_file.name}")
    print(f"{'='*70}")

    # Step 1: Convert Jy/beam to brightness temperature
    tb_path = PROCESSED_DIR / f"{label}_tb.fits"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result = convert_fits(fits_file, tb_path, use_planck=use_planck)
    if result is None or result.get("already_tb"):
        print(f"  Conversion failed for {label}")
        return None

    # Step 2: Read the Tb image and find disk center
    from astropy.io import fits as pyfits
    with pyfits.open(str(tb_path)) as hdul:
        tb_data = hdul[0].data.copy()
        hdr = hdul[0].header

    # Squeeze to 2D
    while tb_data.ndim > 2:
        tb_data = tb_data[0]

    # Step 3: Project to Europa lat/lon
    pixel_scale = abs(hdr.get("CDELT1", 0)) * 3600  # arcsec/pix
    ny, nx = tb_data.shape
    ang_diam = obs_info["ang_diam"]

    # Find disk center using peak-based centroid
    disk_radius_pix = (ang_diam / 2.0) / pixel_scale
    cx, cy = find_disk_center(tb_data, disk_radius_pix=disk_radius_pix)
    if cx is not None:
        print(f"  Disk center (bright centroid): ({cx:.1f}, {cy:.1f})")
    else:
        print(f"  WARNING: Could not find disk center, using image center")
    sub_lon = obs_info["sub_obs_lon"]
    sub_lat = obs_info["sub_obs_lat"]

    from alma_geometry import compute_disk_geometry, extract_tb_on_disk

    lat, lon, emang, on_disk = compute_disk_geometry(
        nx, ny, pixel_scale, ang_diam, sub_lon, sub_lat,
        center_x=cx, center_y=cy
    )

    n_disk = np.sum(on_disk)
    print(f"\n  Geometry projection:")
    print(f"    Image: {nx}x{ny}, pixel scale: {pixel_scale*1000:.1f} mas/pix")
    print(f"    Europa radius: {ang_diam/2:.3f} arcsec = "
          f"{ang_diam/2/pixel_scale:.1f} pixels")
    print(f"    Pixels on disk: {n_disk}")

    # Extract on-disk data
    extracted = extract_tb_on_disk(tb_data, lat, lon, emang, on_disk)
    print(f"    Valid pixels (em < 80°): {extracted['n_pixels']}")

    if extracted["n_pixels"] > 0:
        print(f"    Tb range: {np.nanmin(extracted['tb']):.1f} - "
              f"{np.nanmax(extracted['tb']):.1f} K")
        print(f"    Tb mean (disk): {np.nanmean(extracted['tb']):.1f} K")

        # Disk-averaged Tb (what Trumbo reports in Table 2)
        # Weight by cos(emission_angle) for proper average
        cos_em = np.cos(np.radians(extracted["emission_angle"]))
        disk_avg_tb = np.average(extracted["tb"], weights=cos_em)
        print(f"    Tb disk-average (cos-weighted): {disk_avg_tb:.1f} K")

    # Save processed data as NPZ
    output_npz = PROCESSED_DIR / f"{label}_geometry.npz"
    np.savez_compressed(
        str(output_npz),
        tb=tb_data,
        lat=lat,
        lon=lon,
        emission_angle=emang,
        on_disk=on_disk,
        sub_obs_lon=sub_lon,
        sub_obs_lat=sub_lat,
        ang_diam=ang_diam,
        pixel_scale=pixel_scale,
    )
    print(f"  Saved: {output_npz}")

    return {
        "label": label,
        "fits_file": str(fits_file),
        "tb_file": str(tb_path),
        "npz_file": str(output_npz),
        "n_pixels_disk": int(n_disk),
        "n_pixels_valid": extracted["n_pixels"],
        "tb_mean": float(np.nanmean(extracted["tb"])) if extracted["n_pixels"] > 0 else None,
        "disk_center": (cx, cy),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process Trumbo et al. (2018) ALMA data for PTM3D comparison"
    )
    parser.add_argument("--use-ari", action="store_true",
                        help="Use ARI reprocessed images instead of PI pipeline")
    parser.add_argument("--planck", action="store_true",
                        help="Use full Planck inversion (default: Rayleigh-Jeans)")
    parser.add_argument("--label", help="Process single observation (T1/T2/T3/T4)")
    args = parser.parse_args()

    if args.label:
        obs_list = [o for o in TRUMBO_OBS if o["label"] == args.label]
        if not obs_list:
            print(f"Unknown label: {args.label}")
            print(f"Available: {[o['label'] for o in TRUMBO_OBS]}")
            return
    else:
        obs_list = TRUMBO_OBS

    src = "ARI reprocessed" if args.use_ari else "PI pipeline"
    method = "Planck" if args.planck else "Rayleigh-Jeans"
    print(f"\n  Processing {len(obs_list)} Trumbo observations")
    print(f"  Source: {src}")
    print(f"  Tb method: {method}")

    results = []
    for obs in obs_list:
        result = process_observation(obs, use_ari=args.use_ari,
                                     use_planck=args.planck)
        if result is not None:
            results.append(result)

    # Print summary
    if results:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        print(f"  {'Label':>6} {'Pixels':>8} {'Tb mean':>8} {'Disk center':>16}")
        print(f"  {'─'*50}")
        for r in results:
            cx, cy = r["disk_center"]
            center_str = f"({cx:.0f},{cy:.0f})" if cx is not None else "image center"
            tb_str = f"{r['tb_mean']:.1f} K" if r["tb_mean"] else "N/A"
            print(f"  {r['label']:>6} {r['n_pixels_valid']:>8} {tb_str:>8} "
                  f"{center_str:>16}")


if __name__ == "__main__":
    main()
