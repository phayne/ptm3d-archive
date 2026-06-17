#!/usr/bin/env python3
"""Diurnal-cycle animation of INSTANTANEOUS surface temperature for the Moon3D
500k radiatively-coupled run.

Fixed camera (emission angle 45 deg = 45 deg elevation from the mean surface
normal, azimuth chosen to look toward the south pole, fixed distance, mesh
filling the frame). One frame per output timestep over the ~29.5-day lunar
diurnal cycle; a single fixed color scale so frames are comparable in time.
Writes an MP4 and a GIF.

    python3 moon3d_temp_diurnal.py \
        --ply  output/moon3d_500k_clean/mesh.ply \
        --mesh-vtk .../moon3d_c_cartesian_NEW_500k_clean.vtk \
        --tdir output/moon3d_500k_clean/temperature \
        --times output/moon3d_500k_clean/time_steps.txt \
        --outdir docs/figures/moon3d_500k/anim
"""
from __future__ import annotations
import argparse
import re
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

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style
from moon3d_temp_orbit import build_display_mesh

SECONDS_PER_DAY = 86400.0
DIURNAL_DAYS = 29.53            # lunar synodic day


def load_step(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)[1:]      # drop count header


def pole_camera_azim(mesh_vtk: str) -> float:
    """matplotlib azim that makes the camera look toward the selenocentric
    south pole (0,0,-R). Camera sits opposite the pole, so azim = az_pole+180;
    the z-flip leaves the X,Y azimuth unchanged."""
    c = pv.read(mesh_vtk).points.mean(0)
    az_pole = np.degrees(np.arctan2(-c[1], -c[0]))      # XY direction to pole
    return (az_pole + 180.0) % 360.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--mesh-vtk", required=True,
                    help="source .vtk (full-precision, for the pole azimuth)")
    ap.add_argument("--tdir", required=True, help="temperature/ step-file dir")
    ap.add_argument("--times", required=True, help="time_steps.txt")
    ap.add_argument("--outdir", default="docs/figures/moon3d_500k/anim")
    ap.add_argument("--elev", type=float, default=45.0,
                    help="elevation = 90 - emission angle (deg)")
    ap.add_argument("--azim", type=float, default=None,
                    help="override camera azimuth (default: toward pole)")
    ap.add_argument("--stride", type=int, default=3, help="use every Nth timestep")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--reduction", type=float, default=0.75)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()

    vd, fd, nn, n_full = build_display_mesh(a.ply, a.reduction)
    vd = vd.copy(); vd[:, 2] *= a.zexag
    polys = vd[fd]
    r_xy = float(np.sqrt(vd[:, 0] ** 2 + vd[:, 1] ** 2).max())
    zc = 0.5 * (vd[:, 2].min() + vd[:, 2].max())

    steps = sorted(Path(a.tdir).glob("temperature_*.bin"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    steps = steps[::a.stride]
    print(f"{len(steps)} frames (stride {a.stride}) from {n_full}-face run")

    # elapsed-time axis (seconds) -> days since window start
    t_all = np.loadtxt(a.times, comments="#", usecols=0)
    t0 = t_all[0]
    idxs = list(range(0, len(t_all), a.stride))[:len(steps)]
    days = [(t_all[i] - t0) / SECONDS_PER_DAY for i in idxs]

    # fixed color scale from a temporal sample (robust global range)
    sample = steps[:: max(1, len(steps) // 24)]
    allv = np.concatenate([load_step(p) for p in sample])
    vmin, vmax = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    print(f"fixed color scale: [{vmin:.0f}, {vmax:.0f}] K")
    norm = Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap="inferno")

    azim = a.azim if a.azim is not None else pole_camera_azim(a.mesh_vtk)
    print(f"camera: elev={a.elev:g} (emission {90-a.elev:g}), azim={azim:.1f} (toward pole)")

    apply_style()
    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_axes([-0.02, -0.04, 0.9, 1.05], projection="3d", computed_zorder=False)
    cax = fig.add_axes([0.88, 0.30, 0.025, 0.42])
    pc = Poly3DCollection(polys, edgecolors="none", antialiased=False, rasterized=True)
    pc.set_facecolors(sm.to_rgba(load_step(steps[0])[nn]))
    ax.add_collection3d(pc)
    ax.set_xlim(-r_xy, r_xy); ax.set_ylim(-r_xy, r_xy); ax.set_zlim(zc - r_xy, zc + r_xy)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.9)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=a.elev, azim=azim)        # FIXED camera
    ax.set_axis_off()
    cb = fig.colorbar(sm, cax=cax); cb.set_label("T (K)")
    fig.text(0.45, 0.95, "Moon3D 509k — instantaneous surface temperature",
             ha="center", fontsize=12.5, fontweight="bold")
    clock = fig.text(0.45, 0.90, "", ha="center", fontsize=11)

    def update(k):
        vals = load_step(steps[k])[nn]
        pc.set_facecolors(sm.to_rgba(vals))
        clock.set_text(f"diurnal day {days[k]:5.1f} / {DIURNAL_DAYS:.1f}")
        return (pc, clock)

    anim = animation.FuncAnimation(fig, update, frames=len(steps), blit=False)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mp4 = outdir / "temp_diurnal.mp4"
    gif = outdir / "temp_diurnal.gif"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=a.fps, bitrate=4000), dpi=a.dpi)
    print(f"wrote {mp4}")
    anim.save(gif, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
    print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
