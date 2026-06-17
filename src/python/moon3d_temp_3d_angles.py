#!/usr/bin/env python3
"""Render the Moon3D 500k 3D temperature figure (coupled mean T | coupling
difference) from several oblique azimuth/elevation viewpoints.

Loads the mesh + statistics ONCE and reuses the double-precision DEM read,
south-polar z-flip, and per-face rendering from moon3d_temp_3d.

    python3 moon3d_temp_3d_angles.py \
        --ply output/moon3d_500k_clean/mesh.ply \
        --coupled output/moon3d_500k_clean/statistics \
        --baseline output/moon3d_500k_clean_radoff/statistics \
        --field avg --outdir docs/figures/moon3d_500k/angles
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, DOUBLE_W
from moon3d_temp_3d import (
    read_ply_vertices_double, load_stat, tri_faces, render_panel,
)

# Curated (elevation, azimuth) viewpoints — a sweep around the patch at three
# elevations (low-grazing / moderate / high-oblique) and varied azimuths.
DEFAULT_ANGLES = [
    (15, -120), (15, 30),
    (30, -150), (30, -60), (30, 60),
    (50, -110), (50, 45),
    (70, -90),
]


def parse_angles(spec: str):
    out = []
    for pair in spec.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        e, a = pair.split(",")
        out.append((float(e), float(a)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--field", default="avg", choices=["min", "avg", "max"])
    ap.add_argument("--outdir", default="docs/figures/moon3d_500k/angles")
    ap.add_argument("--angles", default=None,
                    help='override views as "elev,azim;elev,azim;..."')
    ap.add_argument("--zexag", type=float, default=3.0)
    a = ap.parse_args()

    angles = parse_angles(a.angles) if a.angles else DEFAULT_ANGLES

    # ---- load once ----
    full = pv.read(a.ply)
    ff = tri_faces(full)
    vf = read_ply_vertices_double(a.ply)
    assert len(vf) == full.n_points
    vf = vf - vf.mean(0)
    vf[:, 2] *= -1.0                          # south-polar: outward (up) is -z

    coupled = load_stat(Path(a.coupled), a.field)
    baseline = load_stat(Path(a.baseline), a.field)
    assert len(coupled) == len(ff) == len(baseline)
    diff = coupled - baseline
    field_name = {"min": "minimum", "avg": "mean", "max": "maximum"}[a.field]

    nL = Normalize(np.percentile(coupled, 1), np.percentile(coupled, 99))
    dmax = max(np.percentile(np.abs(diff), 99.5), 1.0)
    if diff.min() < -0.5:
        nR, cmapR = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax), "RdBu_r"
    else:
        nR, cmapR = Normalize(0.0, dmax), "hot"

    apply_style()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for elev, azim in angles:
        fig = plt.figure(figsize=(DOUBLE_W, 0.52 * DOUBLE_W))
        axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
        axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
        render_panel(axL, vf, ff, coupled, "inferno", nL, elev, azim,
                     f"Coupled {field_name} T", "T (K)", zexag=a.zexag)
        render_panel(axR, vf, ff, diff, cmapR, nR, elev, azim,
                     f"Coupling effect (Δ{field_name} T)", "ΔT (K)", zexag=a.zexag)
        axL.text2D(0.03, 0.95, "(a)", transform=axL.transAxes,
                   fontsize=10, fontweight="bold", va="top")
        axR.text2D(0.03, 0.95, "(b)", transform=axR.transAxes,
                   fontsize=10, fontweight="bold", va="top")
        fig.suptitle(f"Moon3D 509k — {field_name} T   "
                     f"(elev {elev:g}°, azim {azim:g}°)",
                     fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = outdir / f"temp_3d_{a.field}_e{int(elev):02d}_a{int(azim):+04d}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"wrote {out}")
    print(f"\n{len(angles)} views written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
