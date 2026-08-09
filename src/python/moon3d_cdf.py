#!/usr/bin/env python3
"""Cumulative distributions (CDFs) of per-face max/mean illumination and max/mean
surface temperature for the Moon3D 500k surface patch at its nominal latitude
(-84.8 deg), WITH vs WITHOUT radiation coupling.

Four panels, each plotting the fraction of faces with value <= x:
  (a) Maximum illumination   (per-face max over the lunation)
  (b) Mean illumination      (per-face mean over the lunation)
  (c) Maximum temperature    (per-face max over the lunation)
  (d) Mean temperature       (per-face mean over the lunation)

Illumination = absorbed solar (+IR) flux:
  rad OFF = direct solar only (the saved direct_solar series)
  rad ON  = direct + indirect (radiosity IR + reflected, from the QIDT dump)
Temperature:
  rad OFF = radoff baseline run; rad ON = coupled run (statistics/temperature_*).

The CDFs intercept at x=0 for illumination give the permanently-shadowed (PSR)
face fraction directly. CDFs are by FACE COUNT (matching the companion histograms).

    python3 moon3d_cdf.py \
        --coupled  <heat3d>/moon3d_verify/coupled \
        --radoff   <heat3d>/moon3d_verify/radoff \
        --qind     <heat3d>/moon3d_verify/coupled_indirect.qidt \
        --out      docs/figures/moon3d_500k/illum_temp_cdf.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptm.pubstyle import apply_style, COLORS, DOUBLE_W, panel_label, load_face_mask


def read_qidt(path: Path):
    with open(path, "rb") as fh:
        magic, nf, ns = np.fromfile(fh, dtype=np.uint32, count=3)
        if magic != 0x51494454:
            raise ValueError(f"{path}: bad magic 0x{magic:08x} (expected 'QIDT')")
        data = np.fromfile(fh, dtype=np.float32, count=int(nf) * int(ns))
    return data.reshape(int(nf), int(ns))


def stat(run: Path, field: str, name: str) -> np.ndarray:
    return np.fromfile(run / "statistics" / f"{field}_{name}.bin", dtype=np.float32)[1:]


def illum_maxmean(run: Path, qind: np.ndarray):
    """Per-face (max, mean) of direct (OFF) and direct+indirect (ON)."""
    dsteps = sorted((run / "direct_solar").glob("direct_solar_*.bin"))
    nuse = min(len(dsteps), qind.shape[1])
    nf = qind.shape[0]
    d_max = np.zeros(nf); d_sum = np.zeros(nf)
    t_max = np.zeros(nf); t_sum = np.zeros(nf)
    for j in range(nuse):
        d = np.fromfile(dsteps[j], dtype=np.float32)[1:].astype(np.float64)
        tot = d + qind[:, j]
        np.maximum(d_max, d, out=d_max);   d_sum += d
        np.maximum(t_max, tot, out=t_max); t_sum += tot
    return (d_max, d_sum / nuse), (t_max, t_sum / nuse)


def cdf_xy(v: np.ndarray, npts: int = 2000):
    """Return (x, F) for an empirical CDF, downsampled to ~npts plotting points.
    Step-left convention: F(x) = fraction of faces with value <= x."""
    s = np.sort(v)
    n = s.size
    F = np.arange(1, n + 1) / n
    if n > npts:
        idx = np.unique(np.linspace(0, n - 1, npts).astype(int))
        # always keep the exact endpoints and the x=0 plateau edge
        s, F = s[idx], F[idx]
    return s, F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupled", required=True, help="coupled run dir (direct_solar/, statistics/)")
    ap.add_argument("--radoff", required=True, help="radoff baseline run dir (statistics/)")
    ap.add_argument("--qind", required=True, help="PTM_DUMP_QIND_SERIES .qidt for the coupled run")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/illum_temp_cdf.png")
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="exclude faces whose coupled max-T exceeds this (K) as "
                         "view-factor over-count artifacts; 0 = keep all")
    ap.add_argument("--reclassify-mask", default="",
                    help="boolean face-mask (make_psr_mask.py): spurious-lit deep-PSR faces "
                         "reclassified as PSR (illumination=0, temperature=PSR-floor)")
    a = ap.parse_args()
    coupled, radoff = Path(a.coupled), Path(a.radoff)

    qind = read_qidt(Path(a.qind))
    (i_max_off, i_avg_off), (i_max_on, i_avg_on) = illum_maxmean(coupled, qind)

    t_max_on,  t_avg_on  = stat(coupled, "temperature", "max"), stat(coupled, "temperature", "avg")
    t_max_off, t_avg_off = stat(radoff,  "temperature", "max"), stat(radoff,  "temperature", "avg")

    rc = load_face_mask(a.reclassify_mask) if a.reclassify_mask else np.zeros(len(t_max_on), bool)
    if rc.any():
        genuine_psr = (i_max_off <= 1e-6) & ~rc            # genuine PSR (cold floor) for the fill value
        for arr in (i_max_off, i_avg_off, i_max_on, i_avg_on):
            arr[rc] = 0.0                                   # spurious illumination -> PSR
        for arr in (t_max_on, t_avg_on, t_max_off, t_avg_off):
            arr[rc] = np.median(arr[genuine_psr])           # spurious heating -> PSR floor
        print(f"  reclassified {int(rc.sum())} spurious-lit deep-PSR faces as PSR")

    if a.mask_tmax > 0:
        keep = (t_max_on <= a.mask_tmax) | rc              # keep reclassified faces
        i_max_off, i_avg_off, i_max_on, i_avg_on = (i_max_off[keep], i_avg_off[keep],
                                                    i_max_on[keep], i_avg_on[keep])
        t_max_on, t_avg_on = t_max_on[keep], t_avg_on[keep]
        t_max_off, t_avg_off = t_max_off[keep], t_avg_off[keep]
        print(f"  masked {int(np.count_nonzero(~keep))} hot faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) as VF-overcount artifacts")

    panels = [
        ("(a) Maximum illumination", "absorbed solar + IR flux (W m$^{-2}$)", i_max_off, i_max_on, "illum"),
        ("(b) Mean illumination",    "absorbed solar + IR flux (W m$^{-2}$)", i_avg_off, i_avg_on, "illum"),
        ("(c) Maximum temperature",  "surface temperature (K)",               t_max_off, t_max_on, "temp"),
        ("(d) Mean temperature",     "surface temperature (K)",               t_avg_off, t_avg_on, "temp"),
    ]

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 0.82 * DOUBLE_W))
    for ax, (title, xlabel, off, on, kind) in zip(axes.ravel(), panels):
        for v, color, lab in ((off, COLORS["ref"], "rad OFF"), (on, COLORS["ptm"], "rad ON")):
            x, F = cdf_xy(v)
            ax.plot(x, F, color=color, lw=1.6, label=lab)
        ax.set_title(title.split()[0], fontsize=10, loc="left")   # panel letter only, no descriptive title
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        # clip the sparse extreme tail (e.g. a few non-convergent deep-pit faces)
        # so the bulk is readable; 99.9th pct matches the companion histograms.
        hi = max(np.percentile(off, 99.9), np.percentile(on, 99.9))
        ax.set_xlim(0, hi)
    axes[0, 0].set_ylabel("cumulative fraction of faces")
    axes[1, 0].set_ylabel("cumulative fraction of faces")
    axes[0, 0].legend(loc="lower right" if False else "center right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.suffix != ".pdf":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+pdf)")

    # console summary: key percentiles + PSR fractions
    for title, _, off, on, kind in panels:
        unit = "W/m2" if kind == "illum" else "K"
        po = np.percentile(off, [10, 50, 90]); pn = np.percentile(on, [10, 50, 90])
        extra = ""
        if kind == "illum":
            extra = f"  F(0) OFF={np.mean(off<=1e-6)*100:.1f}% ON={np.mean(on<=1e-6)*100:.1f}%"
        print(f"  {title:28s} OFF p10/50/90={np.round(po,1)} {unit}  "
              f"ON={np.round(pn,1)}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
