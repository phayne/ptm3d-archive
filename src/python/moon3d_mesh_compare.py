#!/usr/bin/env python3
"""Step 3 — mesh-resolution comparison for Moon3D.

Left panel:  the original (full-resolution ~1.7M-face) mesh.
Right panel: the decimated + fold-over-cleaned 509k-face mesh used for the
             thermal run.

Both panels render the SAME spatial sub-region from the SAME oblique camera,
with triangle wireframe edges drawn and a Lambertian illumination shading
(regolith-tan, lit from an oblique sun) so the surface form and the change in
facet density are both legible.

    python3 moon3d_mesh_compare.py \
        --original .../moon3d_c_cartesian_NEW.vtk \
        --decimated .../moon3d_c_cartesian_NEW_500k_clean.vtk \
        --out docs/figures/moon3d_500k/mesh_compare.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, DOUBLE_W, panel_label

REGOLITH = np.array([0.80, 0.76, 0.70])     # warm gray lunar tone
LIGHT = np.array([0.9, 0.45, 0.55])          # oblique sun (mesh frame), normalized below
AMBIENT = 0.30


def tri_faces(mesh: pv.PolyData) -> np.ndarray:
    return mesh.faces.reshape(-1, 4)[:, 1:]


def clip_region(mesh: pv.PolyData, frac: float) -> pv.PolyData:
    """Keep cells whose centroid X,Y lie within a centered box of side
    frac*extent. Only the map-plane (X,Y) is constrained — the surface is a
    2D manifold draped in 3D, so a thin Z slab would miss it entirely."""
    b = np.array(mesh.bounds).reshape(3, 2)
    ctr = b.mean(1)
    half = frac * (b[:2, 1] - b[:2, 0]).max() / 2.0
    surf = mesh.extract_surface().triangulate()
    cc = surf.cell_centers().points
    keep = np.all((cc[:, :2] >= ctr[:2] - half) & (cc[:, :2] <= ctr[:2] + half), axis=1)
    return surf.extract_cells(np.where(keep)[0]).extract_surface().triangulate()


def shaded_colors(v, f):
    """Per-face Lambertian shading -> RGB."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn > 0, nn, 1.0)
    L = LIGHT / np.linalg.norm(LIGHT)
    lam = np.abs(n @ L)                      # abs: outward orientation may vary
    shade = AMBIENT + (1.0 - AMBIENT) * lam
    return np.clip(shade[:, None] * REGOLITH[None, :], 0, 1)


def render(ax, mesh, elev, azim, title, lw):
    v, f = mesh.points - mesh.points.mean(0), tri_faces(mesh)
    # South-polar patch: selenocentric +z points toward the north pole, so the
    # outward (topographic-up) direction is -z. Flip z so the surface renders
    # right-side-up; shade on the flipped geometry so the lighting matches.
    v[:, 2] *= -1.0
    colors = shaded_colors(v, f)
    pc = Poly3DCollection(v[f], facecolors=colors, edgecolors=(0, 0, 0, 0.55),
                          linewidths=lw, antialiased=True, rasterized=True)
    ax.add_collection3d(pc)
    lim = np.array([v.min(0), v.max(0)]); ctr = lim.mean(0)
    half = (lim[1] - lim[0]).max() / 2.0
    ax.set_xlim(ctr[0]-half, ctr[0]+half); ax.set_ylim(ctr[1]-half, ctr[1]+half)
    ax.set_zlim(ctr[2]-half, ctr[2]+half)
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"{title}\n{len(f):,} faces in view", pad=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--decimated", required=True)
    ap.add_argument("--out", default="docs/figures/moon3d_500k/mesh_compare.png")
    ap.add_argument("--clip-frac", type=float, default=0.16,
                    help="side of centered sub-region as a fraction of mesh extent")
    ap.add_argument("--elev", type=float, default=42.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    a = ap.parse_args()

    print("loading original (this is the big one)...")
    orig = clip_region(pv.read(a.original), a.clip_frac)
    deci = clip_region(pv.read(a.decimated), a.clip_frac)
    print(f"in-view faces: original={orig.n_cells:,}  decimated={deci.n_cells:,}")

    apply_style()
    fig = plt.figure(figsize=(DOUBLE_W, 0.55 * DOUBLE_W))
    axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    # thinner edges on the denser mesh so it doesn't ink-fill
    render(axL, orig, a.elev, a.azim, "Original full-resolution mesh", lw=0.25)
    render(axR, deci, a.elev, a.azim, "Decimated + cleaned 509k mesh", lw=0.5)
    axL.text2D(0.03, 0.95, "(a)", transform=axL.transAxes,
               fontsize=10, fontweight="bold", va="top")
    axR.text2D(0.03, 0.95, "(b)", transform=axR.transAxes,
               fontsize=10, fontweight="bold", va="top")
    fig.suptitle("Moon3D — mesh resolution and surface illumination",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
