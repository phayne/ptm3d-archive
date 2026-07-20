#!/usr/bin/env python3
"""
Automated extraction of per-spaxel band-parameter values from Cartwright et al.
(2025) Figs 3-5 (PSJ 6, 125).

For each target subplot:
  1. Crop disk region from the rendered page PNG.
  2. Detect disk center & radius (largest connected non-background blob,
     fit circle by mean & max radius).
  3. Crop the colorbar below the subplot, sample it to build a
     color → value lookup.
  4. Sample the disk on a regular (lat, lon) grid using inverse
     orthographic projection (sub-observer point assumed near the
     leading apex, calibrated against the known position of Tara
     Regio in the figure).
  5. For each (lat, lon), look up the nearest spaxel color and convert
     to a value via the colorbar lookup.
  6. Mask out "not detected" gray spaxels (Cartwright convention:
     light gray = ND).

Output: CSV (lat_deg, lon_deg_E, value, source_pixel_x, source_pixel_y)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Subplot specifications (hardcoded after per-page layout inspection at 400 DPI)
# ---------------------------------------------------------------------------

@dataclass
class MapSpec:
    name: str            # short id ("fresnel_peak_height")
    page_png: str        # path to rendered page PNG
    disk_y_center: int   # disk center pixel (y)
    disk_x_center: int   # disk center pixel (x)
    disk_radius_guess: int   # approximate disk radius in pixels
    cbar_y_top: int      # colorbar top y (pixel)
    cbar_y_bot: int      # colorbar bottom y (pixel)
    cbar_x_left: int     # colorbar left x (pixel)
    cbar_x_right: int    # colorbar right x (pixel)
    val_lo: float        # value at left of colorbar
    val_hi: float        # value at right of colorbar
    description: str
    # Some maps mark "not detected" with light-gray fill instead of color.
    # If True, we drop spaxels whose chroma is below `nd_chroma_max`.
    has_nd: bool = False
    nd_chroma_max: int = 25


# Colorbar y-ranges measured by automatic blue-row detection (see notes
# in the script run that produced these). All disks centered at x=977 (col 1),
# x=1755 (col 2), x=2520 (col 3) at 400 DPI.

# Page 7 (Fig 3) row 3 col 1: 2.0 μm Band CENTER  (1.972 - 2.028 μm)
SPEC_2UM_BANDCENTER = MapSpec(
    name="band_center_2um",
    page_png="data/jwst/cartwright2025/figures/cartwright_p7.png",
    disk_y_center=2810, disk_x_center=977, disk_radius_guess=300,
    cbar_y_top=3294, cbar_y_bot=3329,
    cbar_x_left=380, cbar_x_right=1300,
    val_lo=1.972, val_hi=2.028,
    description="2.0 μm H2O band center [μm]",
)

# Page 8 (Fig 4) row 2 col 2: 3.1 μm Fresnel Peak HEIGHT  (2.5 - 6.5 %)
SPEC_FRESNEL_HEIGHT = MapSpec(
    name="fresnel_31um",
    page_png="data/jwst/cartwright2025/figures/cartwright_p8.png",
    disk_y_center=1800, disk_x_center=1755, disk_radius_guess=300,
    cbar_y_top=2225, cbar_y_bot=2260,
    cbar_x_left=1340, cbar_x_right=2160,
    val_lo=2.5, val_hi=6.5,
    description="3.1 μm Fresnel peak height [%]",
    has_nd=True,
)

# Page 9 (Fig 5) row 1 col 2: 4.25 μm CO2 Lobe DEPTH  (3 - 27 %)
SPEC_CO2_DEPTH = MapSpec(
    name="co2_band_depth",
    page_png="data/jwst/cartwright2025/figures/cartwright_p9.png",
    disk_y_center=740, disk_x_center=1755, disk_radius_guess=300,
    cbar_y_top=1158, cbar_y_bot=1193,
    cbar_x_left=1340, cbar_x_right=2160,
    val_lo=3.0, val_hi=27.0,
    description="4.25 μm CO2 lobe band depth [%]",
)


# ---------------------------------------------------------------------------
# Disk detection
# ---------------------------------------------------------------------------

def detect_disk_circle(img: np.ndarray, y_center: int, x_center: int,
                       r_guess: int) -> tuple[int, int, int]:
    """Refine disk center & radius by finding edge pixels.

    Strategy: the disk has a thin black bounding circle visible against the
    page background. We look outward from the supplied center along radial
    rays for the dark-line transition.
    """
    H, W = img.shape[:2]
    # Coarse sweep: along 16 angles, find first dark pixel after some margin.
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    radii_found = []
    for theta in angles:
        for r in range(int(r_guess * 0.85), int(r_guess * 1.15)):
            yi = int(y_center + r * np.sin(theta))
            xi = int(x_center + r * np.cos(theta))
            if 0 <= yi < H and 0 <= xi < W:
                rgb = img[yi, xi]
                if rgb.mean() < 80:  # dark pixel = boundary line
                    radii_found.append(r)
                    break
    if not radii_found:
        return y_center, x_center, r_guess
    return y_center, x_center, int(np.median(radii_found))


# ---------------------------------------------------------------------------
# Colorbar sampling
# ---------------------------------------------------------------------------

def build_colorbar_lookup(img: np.ndarray, spec: MapSpec,
                          n_samples: int = 64) -> np.ndarray:
    """Sample the colorbar at n_samples uniformly-spaced points along its
    horizontal axis and return an (n_samples, 4) array: (R, G, B, value).
    """
    y_mid = (spec.cbar_y_top + spec.cbar_y_bot) // 2
    xs = np.linspace(spec.cbar_x_left + 4, spec.cbar_x_right - 4,
                     n_samples).astype(int)
    vals = np.linspace(spec.val_lo, spec.val_hi, n_samples)
    rgbs = img[y_mid, xs].astype(np.float64)
    return np.column_stack([rgbs, vals])  # (n, 4)


def color_to_value(rgb: np.ndarray, lookup: np.ndarray) -> float:
    """Find the colorbar entry closest to rgb in Euclidean RGB space."""
    diff = lookup[:, :3] - rgb[None, :].astype(np.float64)
    d = np.sum(diff ** 2, axis=1)
    return float(lookup[int(np.argmin(d)), 3])


# ---------------------------------------------------------------------------
# Orthographic projection
# ---------------------------------------------------------------------------

def latlon_to_pixel(lat_deg: np.ndarray, lon_deg: np.ndarray,
                    sub_lat_deg: float, sub_lon_deg: float,
                    y_c: int, x_c: int, r_pix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward orthographic projection: (lat, lon) → (px, py, visible_mask).

    visible_mask is True where the point is on the near hemisphere.
    """
    phi  = np.deg2rad(lat_deg)
    lam  = np.deg2rad(lon_deg)
    phi0 = np.deg2rad(sub_lat_deg)
    lam0 = np.deg2rad(sub_lon_deg)

    cos_c = (np.sin(phi0) * np.sin(phi)
             + np.cos(phi0) * np.cos(phi) * np.cos(lam - lam0))
    visible = cos_c > 0

    x = np.cos(phi) * np.sin(lam - lam0)
    y = (np.cos(phi0) * np.sin(phi)
         - np.sin(phi0) * np.cos(phi) * np.cos(lam - lam0))

    # Cartwright Figs 3-5 use the standard planetary-cartography convention
    # where east longitude DECREASES going right across the visible disk
    # (i.e. east is on the left). The labels in Fig 3-5 show Powys on the
    # left and Tara on the right; with sub-obs lon 270°E, Tara at 285°E is
    # +15° east of center, which must therefore be plotted to the LEFT.
    # Hence the negative sign on the x-projection.
    px = x_c - r_pix * x
    py = y_c - r_pix * y
    return px, py, visible


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_map(spec: MapSpec, sub_lat: float, sub_lon: float,
                grid_dlat: float = 6.0, grid_dlon: float = 6.0,
                spaxel_avg_radius_px: int = 6) -> list[tuple[float, float, float, int, int]]:
    """Sample one Cartwright disk map at a regular (lat, lon) grid.

    Returns list of (lat, lon, value, px, py) tuples for each visible &
    detected spaxel.
    """
    img = np.asarray(Image.open(spec.page_png).convert("RGB"))

    y_c, x_c, r_pix = detect_disk_circle(
        img, spec.disk_y_center, spec.disk_x_center, spec.disk_radius_guess)
    print(f"[{spec.name}] disk center=({y_c},{x_c}), radius={r_pix} px "
          f"(guess was {spec.disk_radius_guess})")

    lookup = build_colorbar_lookup(img, spec)

    # Build (lat, lon) grid
    lats = np.arange(-87.5, 87.6, grid_dlat)
    lons = np.arange(-180.0, 180.0, grid_dlon)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = LAT.ravel()
    lon_flat = (LON.ravel() + sub_lon) % 360  # convert relative to absolute

    px, py, vis = latlon_to_pixel(
        lat_flat, lon_flat, sub_lat, sub_lon, y_c, x_c, r_pix * 0.96)

    H, W = img.shape[:2]
    results = []
    for i in range(lat_flat.size):
        if not vis[i]:
            continue
        xi, yi = int(round(px[i])), int(round(py[i]))
        # Stay inside disk and inside image
        dr = np.hypot(xi - x_c, yi - y_c)
        if dr > 0.94 * r_pix:
            continue
        if not (spaxel_avg_radius_px <= xi < W - spaxel_avg_radius_px and
                spaxel_avg_radius_px <= yi < H - spaxel_avg_radius_px):
            continue
        # Average a small patch
        patch = img[yi - spaxel_avg_radius_px:yi + spaxel_avg_radius_px + 1,
                    xi - spaxel_avg_radius_px:xi + spaxel_avg_radius_px + 1]
        rgb = patch.reshape(-1, 3).mean(axis=0)
        # Reject text/grid pixels (very dark)
        if rgb.mean() < 25:
            continue
        # Reject background (whitish > 240)
        if rgb.mean() > 240:
            continue
        # Reject the green/yellow chaos-terrain outlines (saturated green or yellow)
        if rgb[1] > rgb[0] + 35 and rgb[1] > rgb[2] + 25:
            continue
        # Reject the orange "Craters and Ejecta" outline
        if rgb[0] > 200 and rgb[1] > 100 and rgb[1] < 180 and rgb[2] < 130:
            continue
        # Reject ND (Cartwright "not detected" spaxels are near-gray light fill,
        # R≈G≈B & moderate-to-high brightness). True valid blue colormap pixels
        # always have B > R + ~20.
        b_minus_r = float(rgb[2]) - float(rgb[0])
        if spec.has_nd:
            if b_minus_r < 18 and rgb.mean() > 180:
                continue
        val = color_to_value(rgb, lookup)
        results.append((float(lat_flat[i]), float(lon_flat[i]),
                        val, xi, yi))
    return results


def write_csv(out_path: Path, rows: Sequence[tuple],
              spec: MapSpec, sub_lat: float, sub_lon: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Automated extraction from Cartwright 2025 figure\n")
        f.write(f"# Map: {spec.description}\n")
        f.write(f"# Source: {spec.page_png}\n")
        f.write(f"# Disk center pixel = ({spec.disk_y_center}; {spec.disk_x_center}); "
                f"radius ~ {spec.disk_radius_guess} px\n")
        f.write(f"# Sub-observer assumed at lat={sub_lat:.1f} deg; lon={sub_lon:.1f} deg E\n")
        f.write(f"# Colorbar value range = [{spec.val_lo}; {spec.val_hi}]\n")
        f.write(f"# Columns: lat_deg lon_deg_E value pixel_x pixel_y\n")
        f.write("lat_deg,lon_deg_E,value,pixel_x,pixel_y\n")
        for lat, lon, val, px, py in rows:
            f.write(f"{lat:.2f},{lon % 360:.2f},{val:.4f},{px},{py}\n")
    print(f"  wrote {len(rows)} spaxels to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/jwst/cartwright2025/digitized"))
    p.add_argument("--sub-lat", type=float, default=0.0,
                   help="Sub-observer sub-Earth latitude in degrees (default 0)")
    p.add_argument("--sub-lon", type=float, default=270.0,
                   help="Sub-observer sub-Earth east-longitude in degrees "
                        "(default 270 = leading apex)")
    p.add_argument("--grid-deg", type=float, default=6.0,
                   help="Grid spacing in degrees for sampling (default 6 deg)")
    p.add_argument("--maps", nargs="+",
                   default=["fresnel_31um", "band_center_2um", "co2_band_depth"],
                   help="Which maps to extract")
    args = p.parse_args()

    specs = {
        "fresnel_31um": SPEC_FRESNEL_HEIGHT,
        "band_center_2um": SPEC_2UM_BANDCENTER,
        "co2_band_depth": SPEC_CO2_DEPTH,
    }

    print(f"Sub-observer assumed at ({args.sub_lat:.1f}, {args.sub_lon:.1f}°E)")
    for name in args.maps:
        spec = specs[name]
        print(f"\n=== {spec.name}: {spec.description} ===")
        rows = extract_map(spec, args.sub_lat, args.sub_lon,
                           grid_dlat=args.grid_deg,
                           grid_dlon=args.grid_deg)
        out_path = args.out_dir / f"{spec.name}.csv"
        write_csv(out_path, rows, spec, args.sub_lat, args.sub_lon)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
