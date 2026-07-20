#!/usr/bin/env python3
"""Moon3D max-surface-temperature perspective montage vs latitude
(manuscript Section 4.3, `tmax_vs_latitude_twoband`).

One shared-colorbar 3D perspective panel per latitude of the de-tilted crater
sweep (dt85..dt60): the same 254k-vertex Ames Regolith Testbed crater patch,
de-tilted and rigidly placed at six clean integer latitudes, each colored by the
per-face MAXIMUM surface temperature over the run (statistics/temperature_max.bin,
radiatively coupled). All panels share ONE Normalize and ONE colorbar so the
latitude trend reads directly as color; they differ only in the latitude-dependent
thermal field.

This is the tracked rebuild of the original (lost) montage generator. It adapts
render_montage() from ptm12_compare_perspective.py (1xN shared-scale perspective
panels) but raises the oblique camera to elev=52/azim=-62 (matching
temp_3d_native_twoband, per Paul's figure review -- a higher camera sees into the
crater interior) and frames each panel in its own local
East-North-Up frame (up = local radial), reusing the double-precision-vertex
helpers from moon3d_temp_3d.py, so patches at six different selenocentric
orientations are all viewed identically (a single z-flip would only be correct at
one latitude). No mask: max-T > 400 K is honest concave-focusing physics and is
shown, not grayed (Paul's standing no-mask ruling; the color scale saturates via
the colorbar `extend="max"` arrow).

    python3 moon3d_tmax_montage.py \
        --run -85 .../dt85_coupled --run -80 .../dt80_coupled ... \
        --out docs/figures/final/tmax_vs_latitude_twoband.png
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
from ptm_pubstyle import apply_style

HEAT3D = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d")

# de-tilted sweep, native-first ordering (lat -85 -> -60, left to right).
RUNS = [
    (HEAT3D / "output/moon3d_lat_final/dt85_coupled", -85.0),
    (HEAT3D / "output/moon3d_lat_final/dt80_coupled", -80.0),
    (HEAT3D / "output/moon3d_lat_final/dt75_coupled", -75.0),
    (HEAT3D / "output/moon3d_lat_final/dt70_coupled", -70.0),
    (HEAT3D / "output/moon3d_lat_final/dt65_coupled", -65.0),
    (HEAT3D / "output/moon3d_lat_final/dt60_coupled", -60.0),
]

# Oblique camera. elev=52/azim=-62 matches temp_3d_native_twoband (Paul's review:
# a higher camera sees into the crater interior). Overridable on the CLI.
ELEV_DEFAULT, AZIM_DEFAULT = 52.0, -62.0


def load_tmax(run: Path) -> np.ndarray:
    """Per-face max surface temperature (K); drop the leading count header."""
    return np.fromfile(run / "statistics" / "temperature_max.bin",
                       dtype=np.float32)[1:]


def enu_frame(c: np.ndarray) -> np.ndarray:
    """Local East-North-Up rotation at selenocentric position c (radial -> +z)."""
    up = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", metavar=("LAT", "DIR"),
                    default=None, help="repeatable; overrides the built-in RUNS list")
    ap.add_argument("--out", default="docs/figures/final/tmax_vs_latitude_twoband.png")
    ap.add_argument("--vmin", type=float, default=100.0)
    ap.add_argument("--vmax", type=float, default=400.0)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--elev", type=float, default=ELEV_DEFAULT)
    ap.add_argument("--azim", type=float, default=AZIM_DEFAULT)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=1.85)
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()

    if a.run:
        runs = [(Path(d).expanduser(), float(lat)) for lat, d in a.run]
    else:
        runs = list(RUNS)

    apply_style()
    cmap = colormaps[a.cmap]
    norm = Normalize(a.vmin, a.vmax)

    panels = []
    for run, lat in runs:
        tmax_path = run / "statistics" / "temperature_max.bin"
        if not tmax_path.exists():
            print(f"  WARNING: skipping lat {lat:+.1f} -- missing {tmax_path}",
                  file=sys.stderr)
            continue
        v = read_ply_vertices_double(str(run / "mesh.ply"))
        f = tri_faces(pv.read(str(run / "mesh.ply")))
        tmax = load_tmax(run)
        if len(tmax) != len(f):
            print(f"  WARNING: lat {lat:+.1f} face/stat mismatch "
                  f"({len(f)} faces, {len(tmax)} values) -- skipping", file=sys.stderr)
            continue
        c = v.mean(0)
        ve = (enu_frame(c) @ (v - c).T).T          # local ENU: up = +z
        ve[:, 2] *= a.zexag
        panels.append(dict(lat=lat, ve=ve, f=f, tmax=tmax))
        print(f"  lat {lat:+6.1f}: Tmax median {np.median(tmax):6.1f} K  "
              f"range {tmax.min():.0f}-{tmax.max():.0f} K  "
              f"({int((tmax>a.vmax).sum())} faces > {a.vmax:.0f} K, shown not masked)")

    if not panels:
        print("ERROR: no runs with a readable temperature_max.bin.", file=sys.stderr)
        return 1

    ncol = len(panels)
    fig = plt.figure(figsize=(3.2 * ncol, 3.6))
    pa_l, pa_r = 0.0, 0.90
    cw = (pa_r - pa_l) / ncol

    for k, p in enumerate(panels):
        ve, f = p["ve"], p["f"]
        tris = ve[f]
        ax = fig.add_axes([pa_l + k * cw, 0.0, cw, 0.86], projection="3d",
                          computed_zorder=False)
        fc = cmap(norm(p["tmax"]))
        pc = Poly3DCollection(tris, facecolors=fc, edgecolors="none",
                              linewidths=0.0, antialiased=False, rasterized=True)
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
        ax.set_title(f"lat {p['lat']:.0f}°", fontsize=10, pad=2)

    cax = fig.add_axes([0.925, 0.10, 0.013, 0.66])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max")
    cb.set_label("maximum surface temperature [K]")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}  ({len(panels)} panels, shared norm "
          f"{a.vmin:.0f}-{a.vmax:.0f} K, camera elev={a.elev:.0f}/azim={a.azim:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
