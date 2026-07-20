#!/usr/bin/env python3
"""Moon3D regional-tilt effect at the true latitude (manuscript Section 4.3,
`fig_tilt_effect_85`).

Two top-down max-surface-temperature maps of the SAME Ames Regolith Testbed
crater patch at latitude -85, side by side on ONE identical color scale:

  (a) tilted     -- the patch retains its 10.85 deg regional tilt (az 185 deg),
                    i.e. the site as it actually sits (lat85_coupled).
  (b) de-tilted  -- the regional tilt removed, pure latitude placement
                    (dt85_coupled), the -85 end member of the latitude sweep.

Both panels are framed in their own local East-North-Up frame (up = local radial)
and viewed top-down with the shared crew camera, reusing the double-precision
vertex helpers from moon3d_temp_3d.py. The color field is per-face
temperature_max.bin (radiatively coupled). No mask: >400 K is honest
concave-focusing physics (Paul's no-mask ruling); the scale saturates via the
colorbar extend arrow.

Per the crew figure-style convention, the psr_area / psr_count / median-Tmax
contrast between the two cases is printed to stdout for the MANUSCRIPT CAPTION,
NOT drawn on the figure.

    python3 moon3d_tilt_effect.py \
        --tilted   .../lat85_coupled --detilted .../dt85_coupled --lat -85 \
        --out docs/figures/final/fig_tilt_effect_85.png
"""
from __future__ import annotations
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
from ptm_pubstyle import apply_style, DOUBLE_W

HEAT3D = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d")
PSR_THRESH = 1.0     # W/m^2
COLD_THRESH = 110.0  # K


def load_stat(run: Path, name: str) -> np.ndarray:
    return np.fromfile(run / "statistics" / f"{name}.bin", dtype=np.float32)[1:]


def enu_frame(c: np.ndarray) -> np.ndarray:
    up = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up])


def face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def load_panel(run: Path, zexag: float):
    v = read_ply_vertices_double(str(run / "mesh.ply"))
    f = tri_faces(pv.read(str(run / "mesh.ply")))
    tmax = load_stat(run, "temperature_max")
    dsm = load_stat(run, "direct_solar_max")
    if not (len(tmax) == len(f) == len(dsm)):
        raise ValueError(f"{run}: face/stat mismatch {len(f)}/{len(tmax)}/{len(dsm)}")
    area = face_areas(v, f)
    psr = dsm < PSR_THRESH
    stats = dict(
        psr_area=float(area[psr].sum() / area.sum()),
        psr_count=float(psr.mean()),
        median_tmax=float(np.median(tmax)),
        coldtrap_area=float(area[tmax < COLD_THRESH].sum() / area.sum()),
    )
    c = v.mean(0)
    ve = (enu_frame(c) @ (v - c).T).T
    ve[:, 2] *= zexag
    return ve, f, tmax, stats


def render(ax, ve, f, tmax, cmap, norm, elev, azim, zoom, title):
    tris = ve[f]
    pc = Poly3DCollection(tris, facecolors=cmap(norm(tmax)), edgecolors="none",
                          antialiased=False, rasterized=True)
    ax.add_collection3d(pc)
    lo = tris.reshape(-1, 3).min(0); hi = tris.reshape(-1, 3).max(0)
    span = hi - lo
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((span[0], span[1], span[2]), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((span[0], span[1], span[2]))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=-2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tilted", default=str(HEAT3D / "output/moon3d_lat_final/lat85_coupled"))
    ap.add_argument("--detilted", default=str(HEAT3D / "output/moon3d_lat_final/dt85_coupled"))
    ap.add_argument("--lat", type=float, default=-85.0)
    ap.add_argument("--out", default="docs/figures/final/fig_tilt_effect_85.png")
    ap.add_argument("--vmin", type=float, default=100.0)
    ap.add_argument("--vmax", type=float, default=400.0)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--elev", type=float, default=68.0)
    ap.add_argument("--azim", type=float, default=-90.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=1.85)
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()

    apply_style()
    cmap = colormaps[a.cmap]
    norm = Normalize(a.vmin, a.vmax)

    veT, fT, tmT, sT = load_panel(Path(a.tilted), a.zexag)
    veD, fD, tmD, sD = load_panel(Path(a.detilted), a.zexag)

    fig = plt.figure(figsize=(DOUBLE_W, 0.56 * DOUBLE_W))
    axL = fig.add_axes([0.02, 0.05, 0.44, 0.88], projection="3d", computed_zorder=False)
    axR = fig.add_axes([0.47, 0.05, 0.44, 0.88], projection="3d", computed_zorder=False)
    render(axL, veT, fT, tmT, cmap, norm, a.elev, a.azim, a.zoom,
           f"(a) tilted  (lat {a.lat:.0f}°, 10.85° regional tilt)")
    render(axR, veD, fD, tmD, cmap, norm, a.elev, a.azim, a.zoom,
           f"(b) de-tilted  (lat {a.lat:.0f}°, pure latitude)")

    cax = fig.add_axes([0.925, 0.18, 0.014, 0.62])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max")
    cb.set_label("maximum surface temperature [K]")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))

    print(f"wrote {out}  (shared norm {a.vmin:.0f}-{a.vmax:.0f} K, "
          f"camera elev={a.elev:.0f}/azim={a.azim:.0f})")
    print("\n--- CAPTION numbers (NOT on-figure) ---")
    print(f"  tilted   (lat85):  psr_area {sT['psr_area']:.3f}  "
          f"psr_count {sT['psr_count']:.3f}  median_Tmax {sT['median_tmax']:.2f} K  "
          f"coldtrap_area {sT['coldtrap_area']:.3f}")
    print(f"  de-tilted(dt85):   psr_area {sD['psr_area']:.3f}  "
          f"psr_count {sD['psr_count']:.3f}  median_Tmax {sD['median_tmax']:.2f} K  "
          f"coldtrap_area {sD['coldtrap_area']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
