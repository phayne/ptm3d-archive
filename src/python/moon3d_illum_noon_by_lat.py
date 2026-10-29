#!/usr/bin/env python3
"""Local-noon direct solar illumination of the Moon3D crater patch across six
latitudes.

The same 254k-vertex crater patch was placed at 60, 65, 70, 75, 80 deg S
(relocated copies) and at its native ~84.8 deg S location. This figure shows
one panel per latitude, each rendered at local solar NOON (the step nearest
LST 12:00). As latitude increases toward the pole the noon Sun elevation falls
(~+31 deg at 60 to ~+7 deg at 84.8), so the bowl's pole-facing walls drop into
ever-longer shadow. Grayscale = instantaneous direct solar flux (W/m^2), shared
scale across panels; a faint hillshade keeps shadowed relief readable.

Each patch is stood up in its own local East-North-Up frame (radial -> +z) so
all six are viewed from an identical camera and differ only in the Sun.

RUNS default was repointed 2026-07-22 (manuscript v4 review, Paul flagged this
figure as "a previous model version") from the dead pre-fix `moon3d_lat/
lat*_clean` + `moon3d_500k_clean_fixed` dirs (verified gone from disk; those
runs predate the 2026-07-17 illumination fixes AND the corrected-VF FINAL
campaign) to the FINAL campaign's own per-latitude coupled runs
(`moon3d_lat_final/{lat60..lat80,native}_coupled`) -- the same 6 latitudes
this figure has always shown, at the campaign every other current sibling
figure (temp_3d_native_twoband, fig_indirect_bands_*, tmax_vs_latitude_twoband,
etc.) is sourced from. Illumination is coupling-independent (pure geometry +
ephemeris), so the _coupled dirs are used only because they are the
convention other FINAL-campaign scripts already follow; _radoff would give
byte-identical direct-solar frames.

    python3 moon3d_illum_noon_by_lat.py \
        --out docs/figures/moon3d_500k/illum_noon_by_lat.png
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces
from ptm_pubstyle import apply_style, DOUBLE_W, savefig_pinned

HEAT3D = Path(os.environ.get("PTM_HEAT3D_ROOT", str(Path(__file__).resolve().parents[3])))   # holds output/<run>
# (run dir, latitude label in deg S). Last entry is the native patch.
RUNS = [
    (HEAT3D / "output/moon3d_lat_final/lat60_coupled", 60.0),
    (HEAT3D / "output/moon3d_lat_final/lat65_coupled", 65.0),
    (HEAT3D / "output/moon3d_lat_final/lat70_coupled", 70.0),
    (HEAT3D / "output/moon3d_lat_final/lat75_coupled", 75.0),
    (HEAT3D / "output/moon3d_lat_final/lat80_coupled", 80.0),
    (HEAT3D / "output/moon3d_lat_final/native_coupled", 84.78),  # native location
]


def load_step(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)[1:]


def enu_frame(c: np.ndarray):
    up = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up]), up, east, north


def noon_index(run: Path, c: np.ndarray, up: np.ndarray):
    bp = np.genfromtxt(run / "body_positions.csv", delimiter=",", names=True)
    sun = np.column_stack([bp["sun_x"], bp["sun_y"], bp["sun_z"]]).astype(float)
    su = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    lon_site = np.arctan2(c[1], c[0])
    lon_ss = np.arctan2(sun[:, 1], sun[:, 0])
    ha = np.degrees((lon_ss - lon_site + np.pi) % (2.0 * np.pi) - np.pi)
    lst = (12.0 - ha / 15.0) % 24.0
    steps = sorted((run / "illumination").glob("illumination_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    m = min(len(steps), len(lst))
    i = int(np.argmin(np.abs((lst[:m] - 12.0 + 12.0) % 24.0 - 12.0)))
    el = float(np.degrees(np.arcsin(np.clip(su[i] @ up, -1.0, 1.0))))
    return steps[i], el, float(lst[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/figures/moon3d_500k/illum_noon_by_lat.png")
    ap.add_argument("--run", nargs=2, action="append", metavar=("LAT", "DIR"),
                    default=None,
                    help="repeatable; overrides the built-in RUNS list "
                         "(e.g. --run 60 .../lat60_coupled ...)")
    ap.add_argument("--elev", type=float, default=68.0)
    ap.add_argument("--azim", type=float, default=-90.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=1.85)
    ap.add_argument("--relief", type=float, default=0.30)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pubclean-canvas-px", default=None, metavar="WxH",
                    help="pin the saved canvas to this exact pixel size at "
                         "--dpi so it matches a reference sibling 1:1 (this "
                         "figure has no suptitle to suppress; the flag only "
                         "controls canvas pinning).")
    a = ap.parse_args()

    runs = [(Path(d).expanduser(), float(lat)) for lat, d in a.run] if a.run else list(RUNS)

    apply_style()
    panels = []
    vmax = 0.0
    for run, lat in runs:
        v = read_ply_vertices_double(str(run / "mesh.ply"))
        f = tri_faces(pv.read(str(run / "mesh.ply")))
        c = v.mean(0)
        R, up, east, north = enu_frame(c)
        ve = (R @ (v - c).T).T
        step, el, lst = noon_index(run, c, up)
        vals = load_step(step)
        # hillshade from ENU face normals (light from the local noon Sun bearing)
        nrm = np.cross(ve[f[:, 1]] - ve[f[:, 0]], ve[f[:, 2]] - ve[f[:, 0]])
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
        nrm[nrm[:, 2] < 0] *= -1.0
        light = np.array([0.0, 0.6, 0.8]); light /= np.linalg.norm(light)
        shade = np.clip(nrm @ light, 0.0, 1.0)
        panels.append(dict(lat=lat, ve=ve, f=f, vals=vals, shade=shade, el=el))
        vmax = max(vmax, float(np.percentile(vals, 99.5)))
        print(f"  lat {lat:5.1f}: noon step {step.name}  el {el:+.1f} deg  "
              f"LST {lst:.2f}  vmax_local {np.percentile(vals,99.5):.0f}")

    norm = Normalize(0, vmax)
    cmap = colormaps["gray"]
    ncol, nrow = 3, 2
    fig = plt.figure(figsize=(DOUBLE_W, 0.60 * DOUBLE_W))
    pa_l, pa_r = 0.0, 0.885          # panel-area width (colorbar lives to the right)
    cw, ch = (pa_r - pa_l) / ncol, 1.0 / nrow

    for k, p in enumerate(panels):
        ve, f = p["ve"], p["f"]
        polys = ve.copy(); polys[:, 2] *= a.zexag
        tris = polys[f]
        rr, cc = divmod(k, ncol)
        ax = fig.add_axes([pa_l + cc * cw, 1.0 - (rr + 1) * ch, cw, ch],
                          projection="3d", computed_zorder=False)
        g = norm(p["vals"])
        if a.relief > 0:
            g = np.clip(g + a.relief * p["shade"] * (1.0 - g), 0.0, 1.0)
        pc = Poly3DCollection(tris, facecolors=cmap(g), edgecolors="none",
                              antialiased=False, rasterized=True)
        ax.add_collection3d(pc)
        lo = tris.reshape(-1, 3).min(0); hi = tris.reshape(-1, 3).max(0)
        span = hi - lo
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect((span[0], span[1], span[2]), zoom=a.zoom)
        except TypeError:
            ax.set_box_aspect((span[0], span[1], span[2]))
        except Exception:
            pass
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_axis_off()
        lab = f"{p['lat']:.0f}°S" if abs(p['lat'] - round(p['lat'])) < 1e-6 \
            else f"{p['lat']:.1f}°S"
        ax.text2D(0.5, 0.985, f"{lab}   Sun el {p['el']:+.1f}°",
                  transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                            alpha=0.80))

    cax = fig.add_axes([0.915, 0.22, 0.013, 0.56])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("direct solar flux (W m$^{-2}$)")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.pubclean_canvas_px:
        w, h = (int(v) for v in a.pubclean_canvas_px.lower().split("x"))
        written = savefig_pinned(fig, out.with_suffix(""), formats=(".png",),
                                 canvas_px=(w, h), dpi=a.dpi)
        print(f"wrote {written[0]}  (vmax={vmax:.0f} W/m2)  [pinned canvas {w}x{h}px]")
    else:
        fig.savefig(out, dpi=a.dpi)
        print(f"wrote {out}  (vmax={vmax:.0f} W/m2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
