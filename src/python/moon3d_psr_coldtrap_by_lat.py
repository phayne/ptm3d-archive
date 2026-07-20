#!/usr/bin/env python3
"""Moon3D PSR and cold-trap area fraction vs latitude -- fraction driver + figure
(manuscript Section 4.3, Figure 17).

This is the reconstructed fraction-computation driver (the original lived only in
/tmp and was lost). It reads the six-latitude run outputs -- one radiatively
COUPLED run and one no-coupling (radoff) run per latitude -- and computes three
AREA-WEIGHTED fractions per latitude:

  * psr_frac         : area fraction with direct_solar_max < PSR_THRESH (permanent
                       shadow, i.e. never directly sunlit). Geometric -> taken from
                       the coupled run's statistics/direct_solar_max.bin.
  * coldtrap_coupled : area fraction with temperature_max < COLD_THRESH in the
                       radiatively-coupled run (statistics/temperature_max.bin).
  * coldtrap_radoff  : same threshold in the no-coupling run.

"Area-weighted" means sum(area of qualifying faces) / sum(all face areas), with
face areas from the double-precision PLY vertices. This reproduces the surviving
docs/figures/moon3d_500k/psr_coldtrap_vs_latitude.json to full float precision
(verified against the native-latitude gray baseline). NOTE: Figure 16's per-panel
"% < 110 K" is a face-COUNT fraction; this figure's cold-trap "area fraction" is
area-weighted -- they are deliberately different metrics.

The driver writes the JSON (same schema the legacy plotter
scripts/plot_moon3d_psr_coldtrap_vs_lat.py consumes) AND renders the manuscript's
two-panel Figure 17: (a) cold-trap area fraction vs latitude with the area lost to
coupling, (b) the absolute (percentage-point) and relative (%) reduction from
radiative coupling. PSR fraction is stored in the JSON but, per the published
figure, is not drawn as a separate curve.

    python3 moon3d_psr_coldtrap_by_lat.py \
        --out  docs/figures/moon3d_500k/psr_coldtrap_vs_latitude \
        --json docs/figures/moon3d_500k/psr_coldtrap_vs_latitude.json

Point it at the forthcoming six-latitude outputs with repeated --run flags, each
LAT COUPLED_DIR RADOFF_DIR (dirs holding mesh.ply + statistics/*.bin):
    --run -84.79 /path/native/coupled /path/native/radoff \
    --run -80    /path/lat80/coupled  /path/lat80/radoff  ...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces

PSR_THRESH = 1.0      # W m^-2  (direct-solar max below this = permanent shadow)
COLD_THRESH = 110.0   # K       (cold-trap water-ice stability threshold)
TRUE_LAT = -84.781    # Moon3D nominal latitude

HEAT3D = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d")
# (latitude deg, coupled run dir, radoff run dir). UPDATE these to the forthcoming
# six-latitude two-band outputs; all overridable on the command line with
# --run LAT COUPLED_DIR RADOFF_DIR.
RUNS = [
    (-84.79, HEAT3D / "output/moon3d_500k_clean_fixed",
             HEAT3D / "output/moon3d_500k_clean_fixed_radoff"),
    (-80.0,  HEAT3D / "output/moon3d_lat/lat80_clean",
             HEAT3D / "output/moon3d_lat/lat80_clean_radoff"),
    (-75.0,  HEAT3D / "output/moon3d_lat/lat75_clean",
             HEAT3D / "output/moon3d_lat/lat75_clean_radoff"),
    (-70.0,  HEAT3D / "output/moon3d_lat/lat70_clean",
             HEAT3D / "output/moon3d_lat/lat70_clean_radoff"),
    (-65.0,  HEAT3D / "output/moon3d_lat/lat65_clean",
             HEAT3D / "output/moon3d_lat/lat65_clean_radoff"),
    (-60.0,  HEAT3D / "output/moon3d_lat/lat60_clean",
             HEAT3D / "output/moon3d_lat/lat60_clean_radoff"),
]


def face_areas(run: Path) -> np.ndarray:
    """Per-face triangle areas from the double-precision PLY vertices."""
    v = read_ply_vertices_double(str(run / "mesh.ply"))
    f = tri_faces(pv.read(str(run / "mesh.ply")))
    tri = v[f]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def load_stat(run: Path, name: str) -> np.ndarray:
    """A statistics/<name>.bin field (drops the leading float32 count header)."""
    return np.fromfile(run / "statistics" / f"{name}.bin", dtype=np.float32)[1:]


def area_fraction(field: np.ndarray, area: np.ndarray, mask: np.ndarray) -> float:
    if len(field) != len(area):
        raise ValueError(f"field/area length mismatch: {len(field)} vs {len(area)}")
    return float(area[mask].sum() / area.sum())


def compute(runs):
    """Return dict with lat, psr_frac, coldtrap_coupled, coldtrap_radoff lists."""
    lat, psr, ctc, cto = [], [], [], []
    for latv, cdir, rdir in runs:
        cdir, rdir = Path(cdir), Path(rdir)
        need = [cdir / "statistics/direct_solar_max.bin",
                cdir / "statistics/temperature_max.bin",
                rdir / "statistics/temperature_max.bin"]
        missing = [p for p in need if not p.exists()]
        if missing:
            print(f"  WARNING: lat {latv:+.1f} skipped -- missing: "
                  f"{', '.join(str(m) for m in missing)}", file=sys.stderr)
            continue
        area_c = face_areas(cdir)
        area_r = face_areas(rdir)
        dsm = load_stat(cdir, "direct_solar_max")
        tmc = load_stat(cdir, "temperature_max")
        tmr = load_stat(rdir, "temperature_max")
        if not (len(dsm) == len(tmc) == len(area_c)) or len(tmr) != len(area_r):
            print(f"  WARNING: lat {latv:+.1f} face/stat count mismatch -- skipping",
                  file=sys.stderr)
            continue
        p = area_fraction(dsm, area_c, dsm < PSR_THRESH)
        c = area_fraction(tmc, area_c, tmc < COLD_THRESH)
        o = area_fraction(tmr, area_r, tmr < COLD_THRESH)
        lat.append(float(latv)); psr.append(p); ctc.append(c); cto.append(o)
        print(f"  lat {latv:+6.1f}:  PSR {100*p:5.1f}%   cold-trap coupled "
              f"{100*c:5.1f}%   radoff {100*o:5.1f}%   (lost {100*(o-c):+.2f} pp)")
    return dict(lat=lat, psr_frac=psr, coldtrap_coupled=ctc, coldtrap_radoff=cto)


def make_figure(d, out_base: Path, dpi: int) -> None:
    lat = np.array(d["lat"]); psr = np.array(d["psr_frac"])
    ctc = np.array(d["coldtrap_coupled"]); cto = np.array(d["coldtrap_radoff"])
    order = np.argsort(lat)
    lat, psr, ctc, cto = lat[order], psr[order], ctc[order], cto[order]
    absred = 100.0 * (cto - ctc)                                   # percentage points
    relred = np.where(cto > 0, 100.0 * (cto - ctc) / cto, 0.0)     # percent

    try:
        import pretty_plots as pp
        pp.use("planetary")
    except Exception:
        from ptm_pubstyle import apply_style
        apply_style()

    C_COUP = "#b2182b"; C_NOCOUP = "#777777"; C_BAND = "#f2b8b8"
    C_ABS = "#3b6ea5"; C_REL = "#2e7d32"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # (a) cold-trap area fraction vs latitude
    if len(lat) > 1:
        axA.fill_between(lat, 100 * ctc, 100 * cto, color=C_BAND, alpha=0.7, lw=0,
                         label="area lost to coupling")
    axA.plot(lat, 100 * cto, "--s", color=C_NOCOUP, lw=2.0, ms=7, label="no coupling")
    axA.plot(lat, 100 * ctc, "-o", color=C_COUP, lw=2.4, ms=7, label="rad-coupled")
    axA.set_xlabel("latitude [deg]")
    axA.set_ylabel("cold-trap area fraction [%]")
    axA.set_ylim(bottom=0)
    axA.set_title("(a) cold-trap area  ($T_{max}$ < %g K)" % COLD_THRESH, fontsize=15)
    axA.grid(alpha=0.3)
    axA.legend(framealpha=0.92)

    # (b) reduction from radiative coupling: absolute (bars) + relative (line)
    width = 3.0 if len(lat) > 1 else 1.0
    axB.bar(lat, absred, width=width, color=C_ABS, alpha=0.85, label="absolute (left)")
    axB.set_xlabel("latitude [deg]")
    axB.set_ylabel("reduction [percentage points]", color=C_ABS)
    axB.tick_params(axis="y", colors=C_ABS)
    axB.set_ylim(bottom=0)
    axB.set_title("(b) reduction from radiative coupling", fontsize=15)
    axBr = axB.twinx()
    axBr.plot(lat, relred, "-D", color=C_REL, lw=2.2, ms=7, label="relative (right)")
    axBr.set_ylabel("relative reduction [%]", color=C_REL)
    axBr.tick_params(axis="y", colors=C_REL)
    axBr.set_ylim(0, 100)
    hb, lb = axB.get_legend_handles_labels()
    hr, lr = axBr.get_legend_handles_labels()
    axB.legend(hb + hr, lb + lr, framealpha=0.92)

    fig.tight_layout(w_pad=3.0)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=dpi)
    fig.savefig(str(out_base) + ".pdf")
    print(f"wrote {out_base}.png / .pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", metavar=("LAT", "COUPLED", "RADOFF"),
                    default=None, help="repeatable; overrides the built-in RUNS list")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/psr_coldtrap_vs_latitude",
                    help="figure base path (.png/.pdf appended)")
    ap.add_argument("--json", default="docs/figures/moon3d_500k/psr_coldtrap_vs_latitude.json")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-figure", action="store_true", help="write JSON only")
    a = ap.parse_args()

    if a.run:
        runs = [(float(lat), Path(c).expanduser(), Path(r).expanduser())
                for lat, c, r in a.run]
    else:
        runs = list(RUNS)

    d = compute(runs)
    if not d["lat"]:
        print("ERROR: no latitudes had readable coupled+radoff outputs; point "
              "--run at the six-latitude runs.", file=sys.stderr)
        return 1

    jp = Path(a.json); jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(d, indent=1))
    print(f"wrote {jp}  ({len(d['lat'])} latitude(s))")

    if not a.no_figure:
        make_figure(d, Path(a.out), a.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
