#!/usr/bin/env python3
"""Illumination snapshots of the Moon3D 500k surface patch over one lunation.

A grid of oblique 3D renders of the patch colored (grayscale) by instantaneous
direct solar flux (W/m^2) at several phases of the lunar day. The view looks
INTO the crater bowl: the camera sits on the low (shallow) rim and looks toward
the tall far wall, and the near lip of the patch is clipped away so the bowl
interior and far wall fill each frame. Each panel draws the Sun direction (gold)
for that instant; a shared inset shows the fixed direction to the lunar south
pole. The grazing, rotating polar illumination sweeps shadows across the relief.

    python3 moon3d_illum_snapshots.py \
        --run <heat3d>/output/moon3d_500k_clean --nframes 6 \
        --out docs/figures/moon3d_500k/illum_snapshots.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces
from ptm_pubstyle import apply_style, DOUBLE_W


def load_step(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    return raw[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run output dir (mesh.ply, illumination/, body_positions.csv)")
    ap.add_argument("--nframes", type=int, default=6)
    ap.add_argument("--local-times", default="6,9,12,15,18,21",
                    help="comma-separated target local solar times (h) to show; "
                         "the nearest step to each is selected. Empty = evenly spaced.")
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--azim", type=float, default=-52.0)
    ap.add_argument("--zexag", type=float, default=1.0)   # 1 = true (equal-axis) relief
    # cutaway: keep faces on the FAR side of a vertical plane so the near lip
    # doesn't occlude the bowl. View is from the +x,+y rim, so cut the near +x lip.
    ap.add_argument("--cut-axis", choices=["x", "y"], default="x")
    ap.add_argument("--cut-val", type=float, default=0.45)
    ap.add_argument("--cut-keep", choices=["greater", "less"], default="less")
    ap.add_argument("--relief", type=float, default=0.30,
                    help="faint hillshade floor so shadowed relief stays visible (0 = none)")
    ap.add_argument("--zoom", type=float, default=1.5,
                    help="3D box zoom (>1 crops in on the crater, less white space)")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/illum_snapshots.png")
    a = ap.parse_args()
    run = Path(a.run)

    # geometry (full precision, recentered, z-flipped so the surface is up)
    full = pv.read(str(run / "mesh.ply"))
    f = tri_faces(full)
    v = read_ply_vertices_double(str(run / "mesh.ply"))
    ctr = v.mean(0)
    v = v - ctr
    # Stand the (south-polar) surface up with a PROPER rotation -- 180 deg about
    # the x-axis (y -> -y, z -> -z, det = +1), NOT a z-only reflection. A bare
    # z-flip inverts chirality and mirror-images the scene, which makes the Sun
    # appear to rise in the west. The same rotation is applied to the Sun and
    # pole vectors below so all directions stay mutually consistent and correct.
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0

    # cutaway mask (per face, by real-space centroid)
    cent = v[f].mean(axis=1)
    ax_i = {"x": 0, "y": 1}[a.cut_axis]
    keep = (cent[:, ax_i] > a.cut_val) if a.cut_keep == "greater" \
        else (cent[:, ax_i] < a.cut_val)
    fk = f[keep]

    # face normals -> hillshade (so geometry reads even where flux = 0)
    nrm = np.cross(v[fk[:, 1]] - v[fk[:, 0]], v[fk[:, 2]] - v[fk[:, 0]])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    nrm[nrm[:, 2] < 0] *= -1.0
    light = np.array([np.cos(np.radians(a.azim + 35)), np.sin(np.radians(a.azim + 35)), 0.85])
    light /= np.linalg.norm(light)
    shade = np.clip(nrm @ light, 0.0, 1.0)        # 0..1 form-shading

    polys = v.copy(); polys[:, 2] *= a.zexag
    tris = polys[fk]

    # Sun directions (body-fixed) per timestep, mapped to figure frame (z flip)
    bp = np.genfromtxt(run / "body_positions.csv", delimiter=",", names=True)
    sun = np.column_stack([bp["sun_x"], bp["sun_y"], bp["sun_z"]]).astype(float)
    sun_fig = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    sun_fig[:, 1] *= -1.0; sun_fig[:, 2] *= -1.0     # same 180 deg-about-x rotation
    # direction to the lunar south pole (fixed), figure frame
    south_pole = np.array([0.0, 0.0, -1737400.0]) - ctr
    pole_fig = south_pole / np.linalg.norm(south_pole)
    pole_fig[1] *= -1.0; pole_fig[2] *= -1.0

    # local solar geometry at the patch, in a topocentric East-North-Up frame at
    # the (selenocentric) patch center -- uses the RAW, un-z-flipped Sun vector.
    sun_u = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    up = ctr / np.linalg.norm(ctr)                         # radial (outward)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    sun_el = np.degrees(np.arcsin(np.clip(sun_u @ up, -1.0, 1.0)))
    sun_az = np.degrees(np.arctan2(sun_u @ east, sun_u @ north)) % 360.0  # 0=N,90=E
    # local apparent solar time (one synodic day = 24 h): noon when the Sun is on
    # the site meridian; hour angle = subsolar lon - site lon (wrapped to +/-180).
    lon_site = np.arctan2(ctr[1], ctr[0])
    lon_ss = np.arctan2(sun[:, 1], sun[:, 0])
    ha = np.degrees((lon_ss - lon_site + np.pi) % (2.0 * np.pi) - np.pi)
    sun_lst = (12.0 - ha / 15.0) % 24.0

    steps = sorted((run / "illumination").glob("illumination_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    n = len(steps)
    if a.local_times.strip():
        # pick the step nearest each requested local solar time (circular in 24 h)
        targets = [float(x) for x in a.local_times.split(",")]
        m = min(n, len(sun_lst))
        idxs = []
        for tgt in targets:
            d = np.abs((sun_lst[:m] - tgt + 12.0) % 24.0 - 12.0)
            idxs.append(int(np.argmin(d)))
        idxs = np.array(idxs)
    else:
        idxs = np.linspace(0, n - 1, a.nframes).round().astype(int)
    # common color scale across frames (robust max over the sampled frames)
    vmax = max(np.percentile(load_step(steps[i])[keep], 99.5) for i in idxs)
    norm = Normalize(0, vmax)
    cmap = colormaps["gray"]

    apply_style()
    ncol = 3
    nrow = int(np.ceil(a.nframes / ncol))
    fig = plt.figure(figsize=(DOUBLE_W, 0.46 * DOUBLE_W * nrow))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    span = (hi - lo)

    for k, i in enumerate(idxs):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d", computed_zorder=False)
        vals = load_step(steps[i])[keep]
        # grayscale illumination, with a faint hillshade floor so the shadowed
        # bowl/far wall still show relief (does not brighten lit faces much).
        g = norm(vals)
        if a.relief > 0:
            g = np.clip(g + a.relief * shade * (1.0 - g), 0.0, 1.0)
        rgba = cmap(g)
        pc = Poly3DCollection(tris, facecolors=rgba, edgecolors="none",
                              antialiased=False, rasterized=True)
        ax.add_collection3d(pc)
        for sset, a0, a1 in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]),
                             (ax.set_zlim, lo[2], hi[2])):
            sset(a0, a1)
        try:
            ax.set_box_aspect((span[0], span[1], span[2]), zoom=a.zoom)
        except TypeError:
            ax.set_box_aspect((span[0], span[1], span[2]))
        except Exception:
            pass
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_axis_off()
        # Sun arrow (gold): slender, anchored just above the patch, toward the Sun
        L = 0.36 * max(span[0], span[1])
        z0 = hi[2] + 0.12 * span[2]
        s = sun_fig[i]
        ax.quiver(0, 0, z0, s[0], s[1], s[2], length=L, color="gold", lw=1.5,
                  arrow_length_ratio=0.16, zorder=10)
        ax.text(s[0]*L*1.2, s[1]*L*1.2, z0 + s[2]*L*1.2, "Sun", color="goldenrod",
                fontsize=7, fontweight="bold", ha="center", va="center")
        # to-S-pole reference (cyan): drawn in the SAME 3D axes/camera as the Sun
        # so the two share one projection and are directly comparable (a separate
        # 2D inset is not camera-consistent and misrepresents the pole bearing).
        p = pole_fig
        ax.quiver(0, 0, z0, p[0], p[1], p[2], length=L, color="tab:cyan", lw=1.5,
                  arrow_length_ratio=0.16, zorder=10)
        ax.text(p[0]*L*1.25, p[1]*L*1.25, z0 + p[2]*L*1.25, "S pole", color="tab:cyan",
                fontsize=7, fontweight="bold", ha="center", va="center")
        # title: local solar time, azimuth, elevation at the patch
        tmin = int(round(sun_lst[i] * 60.0))
        hh, mm = (tmin // 60) % 24, tmin % 60
        ax.set_title(f"LT {hh:02d}:{mm:02d}   az {sun_az[i]:.0f}°   el {sun_el[i]:+.1f}°",
                     fontsize=8.5, pad=-4)

    # shared colorbar
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("direct solar flux (W m$^{-2}$)")

    fig.subplots_adjust(left=0.01, right=0.90, top=0.99, bottom=0.02, wspace=-0.05, hspace=-0.02)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250)
    print(f"wrote {out}  (vmax={vmax:.0f} W/m2, kept {keep.sum()}/{len(f)} faces, frames {list(idxs)})")
    for i in idxs:
        print(f"  step {i:3d}: LT {sun_lst[i]:5.2f} h  az {sun_az[i]:6.1f} deg  el {sun_el[i]:+5.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
