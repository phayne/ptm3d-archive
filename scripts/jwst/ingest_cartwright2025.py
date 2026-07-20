#!/usr/bin/env python3
"""
Ingest Cartwright et al. (2025) JWST/NIRSpec band-parameter maps for Europa
into a uniform NPZ format that the comparison framework can consume.

Source data products (per the paper):
- 3.1 μm Fresnel peak strength map (crystalline H2O diagnostic; Figs 3, 6, 8, 9)
- 2.0 μm band center map (temperature thermometer; Fig 4)
- 4.25 μm CO2 band depth map (Fig 7)

Data availability: only raw NIRSpec cubes at MAST DOI 10.17909/ve8q-sq65
are deposited. Published spatial maps are not separately archived. We
therefore use **figure digitization** (WebPlotDigitizer or similar) to
extract per-spaxel values from the published color-coded maps in the
paper PDF, output as CSV files: lat, lon, value, [uncert].

This script reads the digitized CSV files and packages them into a single
NPZ keyed on the figure name. If no digitization is available yet, it
generates a *synthetic* placeholder that lets the comparison pipeline
run end-to-end (clearly flagged in the output as PLACEHOLDER).

Usage:
  python ingest_cartwright2025.py \
      --csv-dir data/jwst/cartwright2025/digitized \
      --out     data/jwst/cartwright2025/observations.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED = {
    "fresnel_31um.csv":     ("Fresnel peak strength at 3.1 μm",      "fresnel_31um"),
    "band_center_2um.csv":  ("2.0 μm H2O band center [μm]",          "band_center_2um"),
    "co2_band_depth.csv":   ("4.25 μm CO2 band depth [%]",           "co2_band_depth"),
}

CSV_HEADER_HELP = """\
Each digitized CSV must have columns (header optional):
   lat_deg, lon_deg_E, value [, uncertainty]
where:
  - lat_deg in [-90, 90]
  - lon_deg_E in [0, 360], IAU east-positive (matches mesh.ply convention).
    NB: Cartwright (2025) figures often use west-positive (or sub-Earth-
    centered) longitudes; the digitizer must convert to east-positive
    before saving.
  - value: the observed quantity (band depth, peak strength, etc.)
  - uncertainty: optional 1-sigma error
Lines starting with '#' are ignored.
"""


def _read_csv(path: Path) -> dict:
    """Robust CSV reader: skip '#'-prefixed comments (regardless of embedded
    commas), take the first remaining line as the header, parse the rest as
    floats. More forgiving than np.genfromtxt(names=True) which mis-counts
    columns when comment lines contain commas."""
    rows = []
    header = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if header is None:
                header = [p.lower() for p in parts]
                continue
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            rows.append(parts)
    if header is None or not rows:
        raise ValueError(f"{path} contained no header or no data rows")

    cols = {name: np.array([row[i] for row in rows], dtype=str)
            for i, name in enumerate(header)}

    def col(*names):
        for n in names:
            if n in cols:
                # blank cells become NaN
                arr = cols[n]
                out = np.full(arr.shape, np.nan, dtype=np.float64)
                mask = (arr != "") & (arr != "nan")
                out[mask] = arr[mask].astype(np.float64)
                return out
        return None

    lat = col("lat_deg", "lat", "latitude", "latitude_deg")
    lon = col("lon_deg_e", "lon_deg", "lon", "longitude", "longitude_deg_e")
    val = col("value", "v", "y")
    unc = col("uncertainty", "uncert", "sigma", "err", "error")
    if lat is None or lon is None or val is None:
        raise ValueError(f"{path} missing lat / lon / value columns "
                         f"(saw: {list(cols)})")
    out = dict(lat=lat, lon=lon % 360.0, value=val)
    if unc is not None:
        out["uncertainty"] = unc
    return out


def _make_synthetic_placeholder() -> dict:
    """Synthetic, geographically-plausible placeholder data.

    Designed so that the comparison code runs end-to-end. The pattern is
    intentionally crude; do NOT interpret as reflecting real observations.
    """
    rng = np.random.default_rng(42)
    n_lat, n_lon = 18, 36
    lat = np.repeat(np.linspace(-85, 85, n_lat), n_lon)
    lon = np.tile(np.linspace(5, 355, n_lon), n_lat)

    # Crystalline/Fresnel: strongest at warm leading-hemisphere chaos
    leading_dist = np.minimum(np.abs(lon - 270),
                              360 - np.abs(lon - 270))
    cold_factor = np.cos(np.deg2rad(lat))
    fresnel = (cold_factor * (1.0 - leading_dist / 180.0) +
               0.05 * rng.standard_normal(lat.size))
    fresnel = np.clip(fresnel, 0, 1)

    # 2.0 μm band center: 1.986 μm (warm) to 1.992 μm (cold)
    T_proxy = 130 + 25 * cold_factor * (1.0 - leading_dist / 180.0)
    band_center = 1.992 - 3.0e-4 * (T_proxy - 130)
    band_center += 1e-4 * rng.standard_normal(lat.size)

    # CO2 band depth: peak near Tara/Powys (chaos terrain)
    tara_dist = np.sqrt((lat + 7) ** 2 + ((lon - 285 + 180) % 360 - 180) ** 2)
    powys_dist = np.sqrt((lat - 5) ** 2 + ((lon - 248 + 180) % 360 - 180) ** 2)
    co2_depth = 1.5 * np.exp(-(tara_dist / 30.0) ** 2) + \
                1.0 * np.exp(-(powys_dist / 30.0) ** 2) + \
                0.05 + 0.1 * rng.uniform(size=lat.size)

    return dict(
        fresnel_31um=dict(lat=lat, lon=lon, value=fresnel),
        band_center_2um=dict(lat=lat, lon=lon, value=band_center),
        co2_band_depth=dict(lat=lat, lon=lon, value=co2_depth),
        _placeholder=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=CSV_HEADER_HELP)
    p.add_argument("--csv-dir", type=Path,
                   default=Path("data/jwst/cartwright2025/digitized"))
    p.add_argument("--out", type=Path,
                   default=Path("data/jwst/cartwright2025/observations.npz"))
    p.add_argument("--allow-placeholder", action="store_true",
                   help="If digitized CSVs are not present, write a synthetic "
                        "placeholder (clearly flagged) so downstream scripts run")
    args = p.parse_args()

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for fname, (label, key) in REQUIRED.items():
        path = args.csv_dir / fname
        if path.exists():
            found[key] = _read_csv(path)
            print(f"[ingest] loaded {fname}  ({len(found[key]['value'])} points)  : {label}")
        else:
            print(f"[ingest] MISSING {fname}  : {label}")

    if not found:
        if args.allow_placeholder:
            print("[ingest] No CSVs found; emitting synthetic placeholder data.")
            data = _make_synthetic_placeholder()
        else:
            print("[ingest] Re-run with --allow-placeholder to generate synthetic data.")
            print("\n" + CSV_HEADER_HELP)
            return 1
    else:
        data = found
        if len(found) < len(REQUIRED):
            print("[ingest] Warning: only some band products present.")

    # Pack into a single NPZ
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, d in data.items():
        if key.startswith("_"):
            flat[key] = np.array(d)
            continue
        for sub, arr in d.items():
            flat[f"{key}__{sub}"] = arr
    np.savez_compressed(args.out, **flat)
    print(f"[ingest] Wrote {args.out}  with keys: {list(flat.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
