#!/usr/bin/env python3
"""Histograms of per-face maximum and mean absorbed solar+IR illumination for the
Moon3D 500k surface patch over one lunation, WITH vs WITHOUT radiation coupling.

"Without coupling" = direct solar flux only (the saved direct_solar series).
"With coupling"    = direct + indirect (the radiosity IR + reflected flux). The
indirect per-face per-timestep flux is read from a PTM_DUMP_QIND_SERIES dump
(magic 'QIDT': uint32 magic, uint32 n_faces, uint32 n_steps, then
n_faces*n_steps float32, face-major), aligned 1:1 with the direct_solar step
files. Total absorbed flux = direct + indirect is reduced per face to max/mean.

The grazing polar Sun gives this 1.8 m patch no true PSRs (every facet catches
some direct Sun over a month), so the dotted vertical markers indicate the median
illumination of the COLD-TRAP faces (peak surface T < --coldtrap-tmax K), with vs
without coupling -- showing how inter-face IR lifts the most-shadowed facets.

    python3 moon3d_illum_histograms.py \
        --run  <heat3d>/output/moon3d_500k_qind \
        --qind <heat3d>/output/moon3d_500k_qind_indirect.qidt \
        --out  docs/figures/moon3d_500k/illum_histograms.png
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


def load_stat(stat_dir: Path, field: str, name: str) -> np.ndarray:
    raw = np.fromfile(stat_dir / f"{field}_{name}.bin", dtype=np.float32)
    return raw[1:]  # drop the int32 count header


def read_qidt(path: Path):
    """Read the per-timestep indirect-flux dump -> (n_faces, n_steps) float32."""
    with open(path, "rb") as fh:
        magic, nf, ns = np.fromfile(fh, dtype=np.uint32, count=3)
        if magic != 0x51494454:
            raise ValueError(f"{path}: bad magic 0x{magic:08x} (expected 'QIDT')")
        data = np.fromfile(fh, dtype=np.float32, count=int(nf) * int(ns))
    return data.reshape(int(nf), int(ns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir (direct_solar/, statistics/)")
    ap.add_argument("--qind", required=True, help="PTM_DUMP_QIND_SERIES .qidt file")
    ap.add_argument("--coldtrap-tmax", type=float, default=130.0,
                    help="cold-trap faces have peak surface T below this (K)")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/illum_histograms.png")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="exclude faces whose coupled max-T exceeds this (K) as "
                         "view-factor over-count artifacts; 0 = keep all")
    ap.add_argument("--reclassify-mask", default="",
                    help="boolean face-mask file (make_psr_mask.py): spurious-lit "
                         "deep-PSR faces reclassified as PSR (illumination set to 0)")
    a = ap.parse_args()
    run = Path(a.run)

    # indirect flux per face per output-window timestep
    qind = read_qidt(Path(a.qind))                       # (nf, ns)
    nf, ns = qind.shape

    # stream the direct_solar step files; build per-face max/mean of direct and
    # of total = direct + indirect (max over time must use the per-step sum).
    dsteps = sorted((run / "direct_solar").glob("direct_solar_*.bin"))
    if len(dsteps) != ns:
        print(f"WARNING: {len(dsteps)} direct_solar steps vs {ns} indirect columns; "
              f"using min", file=sys.stderr)
    nuse = min(len(dsteps), ns)
    d_max = np.zeros(nf, np.float64); d_sum = np.zeros(nf, np.float64)
    t_max = np.zeros(nf, np.float64); t_sum = np.zeros(nf, np.float64)
    for j in range(nuse):
        d = np.fromfile(dsteps[j], dtype=np.float32)[1:].astype(np.float64)
        tot = d + qind[:, j]
        np.maximum(d_max, d, out=d_max);   d_sum += d
        np.maximum(t_max, tot, out=t_max); t_sum += tot
    d_avg = d_sum / nuse
    t_avg = t_sum / nuse

    rc = load_face_mask(a.reclassify_mask) if a.reclassify_mask else np.zeros(nf, bool)
    if rc.any():
        d_max[rc] = 0.0; d_avg[rc] = 0.0; t_max[rc] = 0.0; t_avg[rc] = 0.0   # -> PSR
        print(f"  reclassified {int(rc.sum())} spurious-lit deep-PSR faces as PSR (illum=0)")
    if a.mask_tmax > 0:
        tcmax = np.fromfile(run / "statistics" / "temperature_max.bin", dtype=np.float32)[1:]
        keep = (tcmax <= a.mask_tmax) | rc            # keep reclassified faces (don't exclude)
        d_max, d_avg, t_max, t_avg = d_max[keep], d_avg[keep], t_max[keep], t_avg[keep]
        print(f"  masked {int(np.count_nonzero(~keep))} hot faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) as VF-overcount artifacts")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 0.46 * DOUBLE_W))
    panels = [("Maximum", d_max, t_max, "a"), ("Mean", d_avg, t_avg, "b")]
    PCTS = [10, 25, 50, 75, 90]

    for ax, (title, off, on, lab) in zip(axes, panels):
        hi = max(np.percentile(on, 99.9), np.percentile(off, 99.9), 1.0)
        bins = np.linspace(0.0, hi, a.bins)
        ax.hist(off, bins=bins, color=COLORS["ref"], alpha=0.55,
                histtype="stepfilled", lw=0)
        ax.hist(on, bins=bins, color=COLORS["ptm"], alpha=0.55,
                histtype="stepfilled", lw=0)
        ax.hist(off, bins=bins, color=COLORS["ref"], histtype="step", lw=1.3)
        ax.hist(on, bins=bins, color=COLORS["ptm"], histtype="step", lw=1.3)
        ax.set_title(f"{title} illumination")
        ax.set_xlabel("absorbed solar + IR flux (W m$^{-2}$)")
        ax.set_yscale("log")
        # modest top headroom so the tallest bar clears the frame (the large 22x
        # band that reserved space for the removed on-figure percentile table is
        # no longer needed under the clean-figure standard)
        ax.set_ylim(0.8, ax.get_ylim()[1] * 3)
        panel_label(ax, f"({lab})")
    axes[0].set_ylabel("face count")

    from matplotlib.patches import Patch
    handles = [Patch(fc=COLORS["ref"], alpha=0.55, label="direct only (rad OFF)"),
               Patch(fc=COLORS["ptm"], alpha=0.55, label="direct + indirect (rad ON)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}" + (" (+pdf)" if out.suffix == ".png" else ""))
    for title, off, on, _ in panels:
        po = np.percentile(off, PCTS); pn = np.percentile(on, PCTS)
        print(f"  {title:8s} W/m2  pct {PCTS}")
        print(f"           OFF {np.round(po,1)}")
        print(f"           ON  {np.round(pn,1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
