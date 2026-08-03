#!/usr/bin/env python3
"""Re-register the lunar Bond-albedo FITS map that is offset 180 deg in longitude.

BUG (PTM-30): ``data/albedo_maps/Moon-BondAlbedo.fits`` is a mislabeled data file.
Its WCS header declares ``CRVAL1=0, CRPIX1=1, CDELT1=1`` -- i.e. image column 0 =
IAU_MOON longitude 0 (east-positive, 0..360). But the raster data is actually laid
out as a "-180..180 centered" world map: IAU longitude 0 (the mean sub-Earth prime
meridian) sits at the *center* column (col 180), not column 0. The engine's FITS
reader (``src/cpp/maps/fits_reader.cpp:193``) honors the (wrong) header, so the
albedo is sampled 180 deg off: the dark nearside maria land on the bright farside
highlands and vice-versa.

Ground truth used to prove it (see BUGS.md PTM-30 / SESSION-NOTES):
  * The topographic mesh (topo_moon_ico*.bin) is CORRECT vs IAU_MOON -- South
    Pole-Aitken basin (53S, 191E) is the deepest point (-5.9 km), every nearside
    mare is low, the farside is high.
  * As loaded, the albedo has nearside maria BRIGHT and farside highlands DARK --
    physically impossible. A +180 deg longitude roll makes the maria dark
    (Bond albedo ~0.07-0.09) and highlands bright, matching reality.
  * The Europa FITS (SAME loader, SAME CRVAL1=0 convention) registers CORRECTLY,
    so this is a data problem specific to the Moon file, NOT a loader bug. Patching
    the shared loader would misregister Europa (NOT do-no-harm).

FIX: roll the image data by half its width (nx/2 = 180 columns) so the data matches
its own 0..360-east WCS header and IAU_MOON. Latitude is untouched (it is correct).
The original is backed up to ``<name>.orig-mislabeled-180`` before writing.

This script is idempotent-guarded: it refuses to roll a map that already passes the
maria-dark physical test (so re-running cannot re-break a corrected file).

Usage:
  python3 scripts/fix_moon_albedo_lon_registration.py \
      [--fits data/albedo_maps/Moon-BondAlbedo.fits] [--dry-run]
"""
import argparse
import os
import sys

import numpy as np
from astropy.io import fits

# Authoritative IAU_MOON (mean-Earth/polar, east-positive 0..360) anchors.
# Maria are dark + low; highlands are bright. (SPA is farside-but-dark, so it is
# deliberately excluded from the highland set.)
MARIA = [  # (lat, lon_east) -- expected DARK
    (28.0, 17.5),   # Mare Serenitatis
    (32.8, 344.4),  # Mare Imbrium
    (17.0, 58.4),   # Mare Crisium
    (20.0, 302.0),  # Oceanus Procellarum
    (8.5, 31.4),    # Mare Tranquillitatis
]
HIGHLANDS = [  # (lat, lon_east) -- expected BRIGHT
    (0.0, 180.0),   # farside highlands (anti-sub-Earth)
    (-4.4, 202.6),  # Korolev region (farside highlands)
    (0.0, 160.0),
    (20.0, 150.0),
    (-43.3, 348.8),  # Tycho ejecta (bright)
]


def _sample(data, lat, lon):
    """Nearest-pixel sample honoring header CRVAL1=0/CRVAL2=-90, CDELT=1, east-lon.

    Row 0 = lat -90 (CRVAL2=-90 at CRPIX2=1); col 0 = lon 0 east (CRVAL1=0).
    """
    ny, nx = data.shape
    r = int(np.clip(round((lat + 90.0) * (ny / 180.0)), 0, ny - 1))
    c = int(round((lon % 360.0) * (nx / 360.0))) % nx
    return data[r, c]


def _contrast(data):
    """highland_mean - maria_mean. Physically must be > 0 (maria darker)."""
    m = np.mean([_sample(data, la, lo) for la, lo in MARIA])
    h = np.mean([_sample(data, la, lo) for la, lo in HIGHLANDS])
    return m, h, h - m


def _heal_constant_columns(data):
    """Replace fill columns (constant across latitude = the wrap-seam fill) with a
    cyclic linear interpolation of the nearest real columns, per row.

    The source map fills its longitude wrap seam (edge columns) with the constant
    default albedo. Those edge fills are healed by the engine at load. But a 180 deg
    roll would relocate them to the map interior, where the engine's edge-only healer
    would leave them as a visible 2-deg seam. Healing here (before rolling) keeps the
    corrected map seamless everywhere. Returns (healed_data, n_healed).
    """
    ny, nx = data.shape
    fill = np.std(data, axis=0) < 1e-9  # constant columns
    n = int(fill.sum())
    if n == 0:
        return data, 0
    real = ~fill
    if not real.any():
        return data, 0
    real_idx = np.where(real)[0]
    out = data.copy()
    for c in np.where(fill)[0]:
        # nearest real column to the left and right on the cyclic axis
        left = real_idx[real_idx < c]
        left = left[-1] if left.size else real_idx[-1]
        right = real_idx[real_idx > c]
        right = right[0] if right.size else real_idx[0]
        dl = (c - left) % nx
        dr = (right - c) % nx
        w = dl / (dl + dr)
        out[:, c] = (1.0 - w) * data[:, left] + w * data[:, right]
    return out, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits",
                    default="data/albedo_maps/Moon-BondAlbedo.fits",
                    help="path to the lunar Bond-albedo FITS to re-register")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the diagnosis but do not write any files")
    args = ap.parse_args(argv)

    path = args.fits
    if not os.path.isfile(path):
        sys.exit(f"ERROR: FITS not found: {path}")

    with fits.open(path) as hdul:
        orig_dtype = hdul[0].data.dtype  # preserve storage precision (do-no-harm)
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    ny, nx = data.shape
    print(f"[fix-moon-albedo] loaded {path}  ({nx}x{ny})")
    print(f"[fix-moon-albedo] WCS: CRVAL1={header.get('CRVAL1')} "
          f"CRPIX1={header.get('CRPIX1')} CDELT1={header.get('CDELT1')} "
          f"CRVAL2={header.get('CRVAL2')}")

    m0, h0, c0 = _contrast(data)
    print(f"[fix-moon-albedo] BEFORE: maria_mean={m0:.4f}  highland_mean={h0:.4f}  "
          f"contrast(h-m)={c0:+.4f}")

    if c0 > 0.0:
        print("[fix-moon-albedo] Map ALREADY passes the maria-dark physical test "
              "(contrast > 0). Refusing to roll -- it is either already correct or "
              "not the known-mislabeled file. No changes made.")
        return 0

    # Heal the wrap-seam fill columns BEFORE rolling so they do not land in the map
    # interior (where the engine's edge-only seam healer would leave them).
    healed, n_fill = _heal_constant_columns(data)
    if n_fill:
        print(f"[fix-moon-albedo] healed {n_fill} constant fill column(s) "
              "(wrap seam) before rolling")

    # Roll longitude by exactly half the width. For a 0..360 map mislabeled as a
    # -180..180-centered raster, the correction is a structural half-width roll.
    if nx % 2 != 0:
        print(f"[fix-moon-albedo] WARNING: width {nx} is odd; rolling by {nx // 2} "
              "columns (half-width); verify the anchor test result below.")
    rolled = np.roll(healed, nx // 2, axis=1)

    m1, h1, c1 = _contrast(rolled)
    print(f"[fix-moon-albedo] AFTER +{nx // 2}col ({360.0 * (nx // 2) / nx:.1f} deg) "
          f"roll: maria_mean={m1:.4f}  highland_mean={h1:.4f}  contrast(h-m)={c1:+.4f}")

    if c1 <= 0.0:
        sys.exit("ERROR: roll did NOT make maria darker than highlands "
                 "(contrast still <= 0). Aborting without writing -- the offset is "
                 "not a clean 180 deg; investigate before forcing a fix.")

    if args.dry_run:
        print("[fix-moon-albedo] --dry-run: not writing. Diagnosis confirms a "
              "+180 deg longitude misregistration; roll fixes it.")
        return 0

    # Back up the ORIGINAL bytes (copy, not move) so provenance is preserved.
    backup = path + ".orig-mislabeled-180"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(path, backup)
        print(f"[fix-moon-albedo] backed up original -> {backup}")
    else:
        print(f"[fix-moon-albedo] backup already exists -> {backup} (not overwritten)")

    # Preserve the header (CRVAL1=0 is now HONEST for the rolled data) and record
    # the correction in HISTORY so the provenance is not silent.
    header.add_history("PTM-30: rolled longitude axis by nx/2 (+180 deg) to correct "
                       "a -180..180-centered raster mislabeled as 0..360 east. "
                       "Now data col 0 = IAU_MOON lon 0. See BUGS.md PTM-30.")
    fits.writeto(path, rolled.astype(orig_dtype), header, overwrite=True)
    print(f"[fix-moon-albedo] WROTE corrected map -> {path} (dtype {orig_dtype}, "
          "longitude-only change)")
    print("[fix-moon-albedo] NOTE: per-face camera .bin maps (Moon_Albedo_res*.bin, "
          "Moon_Albedo_res9_camera.bin) were sampled from the OLD FITS and are also "
          "180 deg off; regenerate them from this corrected FITS before any camera "
          "render (BUGS.md PTM-30). The thermal loader collapses .bin to a uniform "
          "mean, so thermal runs are unaffected by the .bin.")
    print("[fix-moon-albedo] NOTE: the distributed copy (fetch_data.py --dataset "
          "albedo-maps / Zenodo) is STILL the mislabeled file; re-fetching will "
          "re-introduce the bug until the archive is corrected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
