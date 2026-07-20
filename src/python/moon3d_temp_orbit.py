#!/usr/bin/env python3
"""Orbit animations of the Moon3D 500k radiatively-coupled temperature field.

For each of mean / max / min temperature, render a 3D oblique view of the
coupled model and orbit the camera in azimuth (fixed elevation = fixed
emission angle, fixed distance, mesh filling the frame), writing both an MP4
and a GIF at 2 fps.

    python3 moon3d_temp_orbit.py \
        --ply output/moon3d_500k_clean/mesh.ply \
        --coupled output/moon3d_500k_clean/statistics \
        --outdir docs/figures/moon3d_500k/anim
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style
from moon3d_temp_3d import read_ply_vertices_double, load_stat, tri_faces

FIELDS = {"min": "minimum", "avg": "mean", "max": "maximum"}
CMAP = "inferno"


def centroids(v, f):
    return (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0


def build_display_mesh(ply, reduction):
    """Full-precision DEM vertices (recentered, south-polar z-flipped), then a
    clean quadric decimation for fast per-frame redraws; map full-res per-face
    temperatures onto the display faces by nearest centroid."""
    full = pv.read(ply)
    ff = tri_faces(full)
    vf = read_ply_vertices_double(ply)
    vf = vf - vf.mean(0)
    vf[:, 2] *= -1.0
    cen_full = centroids(vf, ff)

    m = pv.PolyData(vf, np.column_stack([np.full(len(ff), 3), ff]).ravel())
    disp = m.decimate(reduction) if reduction > 0.02 else m
    vd = np.asarray(disp.points)
    fd = tri_faces(disp)
    nn = cKDTree(cen_full).query(centroids(vd, fd))[1]
    return vd, fd, nn, len(ff)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--outdir", default="docs/figures/moon3d_500k/anim")
    ap.add_argument("--fields", default="avg,max,min")
    ap.add_argument("--elev", type=float, default=32.0, help="fixed emission/elevation angle (deg)")
    ap.add_argument("--daz", type=float, default=10.0, help="azimuth increment (deg)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--reduction", type=float, default=0.75)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()

    vd, fd, nn, n_full = build_display_mesh(a.ply, a.reduction)
    vd = vd.copy()
    vd[:, 2] *= a.zexag
    polys = vd[fd]
    print(f"display mesh: {len(fd)} faces (from {n_full})")

    # Fixed bounding-circle limits so the rotating patch always fills the frame
    # without clipping (square map-plane footprint -> use the radial extent).
    r_xy = float(np.sqrt(vd[:, 0] ** 2 + vd[:, 1] ** 2).max())
    z0, z1 = vd[:, 2].min(), vd[:, 2].max()
    zc = 0.5 * (z0 + z1)
    azimuths = np.arange(0.0, 360.0, a.daz)
    apply_style()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for key in [f.strip() for f in a.fields.split(",")]:
        fname = FIELDS[key]
        vals = load_stat(Path(a.coupled), key)[nn]
        norm = Normalize(np.percentile(vals, 1), np.percentile(vals, 99))
        sm = ScalarMappable(norm=norm, cmap=CMAP)

        fig = plt.figure(figsize=(6.4, 6.0))
        ax = fig.add_axes([-0.02, -0.04, 0.9, 1.05], projection="3d",
                          computed_zorder=False)
        cax = fig.add_axes([0.88, 0.30, 0.025, 0.42])
        pc = Poly3DCollection(polys, facecolors=sm.to_rgba(vals),
                              edgecolors="none", antialiased=False, rasterized=True)
        ax.add_collection3d(pc)
        ax.set_xlim(-r_xy, r_xy); ax.set_ylim(-r_xy, r_xy)
        ax.set_zlim(zc - r_xy, zc + r_xy)        # cubic box -> constant size on orbit
        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.9)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        cb = fig.colorbar(sm, cax=cax); cb.set_label("T (K)")
        ttl = fig.text(0.45, 0.93, f"Moon3D 509k — coupled {fname} temperature",
                       ha="center", fontsize=13, fontweight="bold")

        def update(az):
            ax.view_init(elev=a.elev, azim=az)
            return (pc,)

        anim = animation.FuncAnimation(fig, update, frames=azimuths, blit=False)
        mp4 = outdir / f"temp_orbit_{key}.mp4"
        gif = outdir / f"temp_orbit_{key}.gif"
        anim.save(mp4, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000), dpi=a.dpi)
        print(f"wrote {mp4}")
        anim.save(gif, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
        print(f"wrote {gif}")
        plt.close(fig)

    print(f"\n{len(azimuths)} frames/field at {a.fps} fps -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
