#!/usr/bin/env python3
"""Maximum surface temperature of the Moon3D crater patch across six latitudes
(manuscript Section 4.3, Figure 16).

The same 254k-vertex Ames Regolith Testbed crater patch is rigid-rotated to six
latitudes and thermally simulated with radiative coupling enabled. This figure
renders one top-down panel per latitude, each colored by the per-face MAXIMUM
surface temperature over the run (statistics/temperature_max.bin). It is the
temperature analog of Figure 15 (moon3d_illum_noon_by_lat.py, noon illumination)
and reuses the same per-patch East-North-Up framing and camera so all six panels
are viewed identically and differ only in the latitude-dependent thermal field.

Per-panel annotation reproduces the manuscript exactly:
    lat -84.8 deg
    T^med_max = 85 K, 59% < 110 K
where T^med_max is the MEDIAN of the per-face max-temperature field and the
"% < 110 K" is the face-COUNT fraction below the cold-trap threshold, taken
over the RESOLVED facets only -- a facet whose value is not finite has no value
and leaves both the numerator and the denominator (Paul's ruling, 2026-09-15).
The resolved population is printed per latitude and NEVER drawn on the figure
(Paul, 2026-09-15: disclosures belong in notes outside the paper, not on a
figure). (Note: Figure 16's per-panel
"% < 110 K" is a count fraction; Figure 17's cold-trap "area fraction" is
area-weighted -- these are deliberately different metrics. See
moon3d_psr_coldtrap_by_lat.py for the area-weighted version.)

    python3 moon3d_tmax_by_lat.py \
        --out docs/figures/moon3d_500k/tmax_by_lat.png

Point it at the forthcoming six-latitude run outputs with repeated --run flags,
each LAT DIR (a directory holding mesh.ply + statistics/temperature_max.bin):
    --run -84.79 /path/native/coupled --run -80 /path/lat80/coupled ...
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
from ptm_pubstyle import apply_style, DOUBLE_W
import ptm_population as pop

HEAT3D = Path(os.environ.get("PTM_HEAT3D_ROOT", str(Path(__file__).resolve().parents[3])))   # holds output/<run>

# (run dir, latitude in deg). Native patch first so it renders upper-left, per
# the manuscript Fig 16 ordering (upper-left to lower-right: -84.8, -80, ..., -60).
# UPDATE these to the forthcoming six-latitude two-band coupled outputs; keep
# them consistent with moon3d_illum_noon_by_lat.py's RUNS. Any/all are
# overridable on the command line with --run LAT=DIR.
RUNS = [
    (HEAT3D / "output/moon3d_500k_clean_fixed", -84.79),  # native location
    (HEAT3D / "output/moon3d_lat/lat80_clean", -80.0),
    (HEAT3D / "output/moon3d_lat/lat75_clean", -75.0),
    (HEAT3D / "output/moon3d_lat/lat70_clean", -70.0),
    (HEAT3D / "output/moon3d_lat/lat65_clean", -65.0),
    (HEAT3D / "output/moon3d_lat/lat60_clean", -60.0),
]


def load_tmax(run: Path) -> np.ndarray:
    """Per-face maximum surface temperature (K). The .bin carries a leading
    float32 face-count header, so drop the first element."""
    return np.fromfile(run / "statistics" / "temperature_max.bin",
                       dtype=np.float32)[1:]


def enu_frame(c: np.ndarray):
    """Local East-North-Up frame at selenocentric position c (radial -> +z)."""
    up = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up]), up, east, north


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", metavar=("LAT", "DIR"),
                    default=None, help="repeatable; overrides the built-in RUNS list")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/tmax_by_lat.png")
    ap.add_argument("--vmin", type=float, default=100.0, help="colorbar min [K]")
    ap.add_argument("--vmax", type=float, default=400.0, help="colorbar max [K]")
    ap.add_argument("--cold-thresh", type=float, default=110.0,
                    help="cold-trap threshold for the panel annotation [K]")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="draw faces whose max-T exceeds this (K) in neutral gray "
                         "(view-factor over-count artifacts); 0 = keep all")
    ap.add_argument("--elev", type=float, default=68.0)
    ap.add_argument("--azim", type=float, default=-90.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=1.85)
    ap.add_argument("--annotate", dest="annotate", action="store_true", default=True,
                    help="draw the manuscript per-panel lat + stat labels (default)")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false",
                    help="clean-figure variant: latitude label only, no stat callout")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    if a.run:
        runs = [(Path(d).expanduser(), float(lat_s)) for lat_s, d in a.run]
    else:
        runs = list(RUNS)

    apply_style()
    panels = []
    for run, lat in runs:
        if not (run / "statistics" / "temperature_max.bin").exists():
            print(f"  WARNING: skipping lat {lat:+.1f} -- missing "
                  f"{run/'statistics'/'temperature_max.bin'}", file=sys.stderr)
            continue
        v = read_ply_vertices_double(str(run / "mesh.ply"))
        f = tri_faces(pv.read(str(run / "mesh.ply")))
        tmax = load_tmax(run)
        if len(tmax) != len(f):
            print(f"  WARNING: lat {lat:+.1f} face/stat mismatch "
                  f"({len(f)} faces, {len(tmax)} values) -- skipping", file=sys.stderr)
            continue
        c = v.mean(0)
        R, up, east, north = enu_frame(c)
        ve = (R @ (v - c).T).T
        # Paul's ruling 2026-09-15: a facet whose temperature is not finite has
        # no value and leaves BOTH the numerator and the denominator of the
        # count fraction. `np.mean(tmax < thresh)` would instead have counted a
        # NaN facet as "not a cold trap" and a -inf facet AS one.
        resolved = pop.resolved_mask(tmax)
        pop.warn_if_unresolved(f"lat {lat:+.1f} cold-trap count (temperature_max)",
                               resolved)
        cold_ct = 100.0 * pop.count_fraction(tmax < a.cold_thresh, resolved,
                                             f"lat {lat:+.1f} cold-trap count")
        # the panel-label median leaves the unresolved facets behind too, or
        # the published panel reads "T^med_max=nan K"
        tmed = float(np.median(pop.resolved_values(tmax, resolved)))
        panels.append(dict(lat=lat, ve=ve, f=f, tmax=tmax, tmed=tmed, cold=cold_ct,
                           resolved=resolved))
        nmask = (int(np.count_nonzero((tmax > a.mask_tmax) & resolved))
                 if a.mask_tmax > 0 else 0)
        print(f"  lat {lat:+6.1f}: T^med_max {tmed:5.1f} K  "
              f"{cold_ct:5.1f}% < {a.cold_thresh:.0f} K  "
              f"[Tmax range {tmax.min():.0f}-{tmax.max():.0f} K]"
              + (f"  masked {nmask} >{a.mask_tmax:.0f}K" if a.mask_tmax > 0 else "")
              + f"  [{pop.population_note(resolved)}]")

    if not panels:
        print("ERROR: no runs with a readable temperature_max.bin were found; "
              "point --run at the six-latitude outputs.", file=sys.stderr)
        return 1

    norm = Normalize(a.vmin, a.vmax)
    cmap = colormaps[a.cmap]
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
        fc = cmap(norm(p["tmax"]))
        if a.mask_tmax > 0:
            # `& resolved` matches what the console reported: without it a
            # +inf facet is painted as a VF over-count face on the figure
            # while the printed mask count says it is not one.
            fc[(p["tmax"] > a.mask_tmax) & p["resolved"]] = (0.7, 0.7, 0.7, 1.0)
        pc = Poly3DCollection(tris, facecolors=fc,
                              edgecolors="none", antialiased=False, rasterized=True)
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
        if a.annotate:
            lab = (f"lat {p['lat']:.1f}°\n"
                   f"$T^{{\\mathrm{{med}}}}_{{\\mathrm{{max}}}}$={p['tmed']:.0f} K, "
                   f"{p['cold']:.0f}% < {a.cold_thresh:.0f} K")
            # The percentage is over the RESOLVED facets only (Paul's ruling,
            # 2026-09-15), but the panel says nothing about that. Paul,
            # 2026-09-15: "Do not restyle figures or present disclosures/
            # caveats on figures -- these can be recorded in notes outside of
            # the paper that I can incorporate as needed." The population goes
            # to stdout (above) and to the session notes, never onto the
            # figure. This is the standing crew convention, not a one-off.
        else:
            lab = f"lat {p['lat']:.1f}°"
        ax.text2D(0.5, 0.985, lab, transform=ax.transAxes, ha="center", va="top",
                  fontsize=8.5, bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                          ec="none", alpha=0.80))

    cax = fig.add_axes([0.915, 0.22, 0.013, 0.56])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max")
    cb.set_label("maximum surface temperature [K]")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}  ({len(panels)} panel(s), norm {a.vmin:.0f}-{a.vmax:.0f} K)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
