#!/usr/bin/env python3
"""PTM3D vs Diviner SEASONAL polar temperatures (Williams et al. 2019).

The Diviner level-4 polar product gives summer/winter MAX and AVG bolometric
brightness temperature (240 m, poleward of 80 S), where summer/winter are defined
by the subsolar latitude being below/above the equator (the 1.54 deg obliquity).

We reuse the existing full-year PTM3D runs (same DEMs): split each run's output
timesteps into summer (subsolar lat < 0) and winter (>= 0) via SPICE, then per
grid cell take the MAX or MEAN over the season of the footprint bolometric T
   Tbol(cell,t) = eps^(1/4) * ( sum_i A_i T_i(t)^4 / sum_i A_i )^(1/4),
and compare to the corresponding Diviner seasonal map. Registration is by
selenographic lon/lat projected to south-polar stereographic (R=1737.4 km),
clean (the seasonal ASCII gives clon/clat directly).

Output: a per-site 4x3 figure (rows: summer-max, summer-avg, winter-max,
winter-avg; cols: Diviner | PTM3D | diff) + a summary JSON of bias/RMSE.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pyproj
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent))
from diviner_compare import read_ply, face_geometry, load_step
from ptm_pubstyle import apply_style

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
OUTROOT = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output")
SEASDIR = REPO / "data/diviner_seasonal"
METAKERNEL = str(REPO / "data/kernels/moon.tm")
EPS4 = 0.95 ** 0.25
STEREO = pyproj.CRS.from_proj4("+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +no_defs")
LL = pyproj.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
TO_XY = pyproj.Transformer.from_crs(LL, STEREO, always_xy=True)

SITES = [("Haworth", "haworth", -87.45, -2.1),
         ("Faustini", "faustini", -87.3, 84.0),
         ("Shackleton", "shackleton", -89.66, 129.2),
         ("Haworth SE rim", "sunlit", -88.181, 6.041)]
# Per-site run-directory override (relative to OUTROOT). Sites not listed here
# fall back to the production "divval_{key}_cr" (2-orbit) run. Haworth uses the
# multi-year-equilibrated run (deep column converged; fixes the off-peak warm
# bias -- see the seasonal-amplitude investigation).
RUN_OVERRIDE = {"haworth": "divval_haworth_r32c",
                "faustini": "divval_faustini_fix",
                "shackleton": "divval_shackleton_r32",
                "sunlit": "divval_sunlit_fix"}
# Explicit absolute-path run-directory overrides, keyed by site key. EMPTY by
# default (NO behavior change vs the committed script); populated only via
# --run-dir SITE=PATH. Takes precedence over RUN_OVERRIDE/OUTROOT so a run
# OUTSIDE OUTROOT (e.g. the two-band run in ~/dev/heat3d) can be wired in
# without touching the defaults above.
RUN_DIR_ABS: dict = {}
# Explicit (lat0, lon0) center overrides, keyed by site key. EMPTY by default
# (NO behavior change vs the committed script / SITES table above); populated
# only via --center SITE=LAT,LON. 2026-07-22 finding: the hardcoded "sunlit"
# entry here (-88.181, 6.041) is a DIFFERENT point (~2.5 km away) from the
# fig8_hybrid mesh's actual build center (-88.1, 6.3, per Paul's site spec
# "88.1 S 6.3 E" and the mesh session note's independently-verified 22.00 km
# distance-to-Haworth check) -- use --center sunlit=-88.1,6.3 for that mesh,
# NOT the stale table default. Flagged rather than silently edited above,
# since 3 other scripts share this same hardcoded tuple.
CENTER_OVERRIDE: dict = {}
# Small, unobtrusive white in-panel crater-name labels for the --overview LOLA
# topography panel, (name, dx, dy, ha) with dx/dy in km relative to the site
# center and `ha` the matplotlib text horizontal-alignment anchor -- keyed by
# site so this is a no-op for haworth/faustini/shackleton. Positions for
# "sunlit" derived from the real crater centers (Haworth: SITES table
# -87.45,-2.1; Shoemaker: data/locations/locations_lunar_psr_south.txt
# -88.1,50 -- the project's own PSR reference list) run through this module's
# own TO_XY projection, THEN visually cross-checked against the actual LOLA
# raster at both the 30km (--overview) and a wider 60km diagnostic extent
# (2026-07-28, manuscript v5 Fig. 8 review): Haworth is confirmed as the
# ~25km-radius basin dominating the upper-left/center of the panel (the
# "sunlit" site sits on its south rim, consistent with the site's "Haworth SE
# rim" name); Shoemaker's catalog center (dx=+38, dy=-18 km) falls just
# outside the 30km half-width, but its ~25km rim visibly clips into the
# panel's lower-right corner in the wider check -- matching Paul's own
# expectation that only a rim portion is visible there. Shoemaker's label
# uses ha="right" anchored near the panel edge (not "center") because the
# full word at fontsize 8 bold otherwise ran past the axis frame on the right
# (caught on the first render -- see session note).
CRATER_LABELS = {"sunlit": [("Haworth", -15.0, 22.0, "center"),
                            ("Shoemaker", 28.0, -19.0, "right")]}
WINDOW_KM, CELL_M, CORE_KM = 15.0, 250.0, 15.0
# Manuscript Fig. 8 site-overview row (--overview): wider-region LOLA topography +
# Diviner summer Tbol,max context, with the CORE_KM comparison window outlined.
LOLA = REPO / "data/dem/validation/_lola_20m_spstereo_pm120km.tif"
OVERVIEW_ROI_KM = 30.0

# Per-site color scales (K), tuned to each site's T_bol distribution (see
# diviner_seasonal_histograms.py): (T_lo, T_hi) for the Diviner/PTM3D panels,
# (dT_lo, dT_hi) for the model-minus-Diviner panel. T ranges are the combined
# 1-99 percentile; dT is +/- the 95th-pct |model-Diviner|, capped at +/-60 K so
# the PSR-floor structure stays visible on the bimodal (sunlit-rim) sites.
SCALES_BY_SITE = {
    "haworth": {
        ("summer", "max"):  (35, 100, -35, 35),
        ("summer", "mean"): (25,  55, -20, 20),
        ("winter", "max"):  (25,  55, -20, 20),
        ("winter", "mean"): (25,  35,  -5,  5),
    },
    "faustini": {
        ("summer", "max"):  (35, 280, -50, 50),
        ("summer", "mean"): (25, 140, -20, 20),
        ("winter", "max"):  (25, 250, -35, 35),
        ("winter", "mean"): (25,  70, -10, 10),
    },
    "shackleton": {
        ("summer", "max"):  (65, 315, -60, 60),
        ("summer", "mean"): (50, 180, -45, 45),
        ("winter", "max"):  (45, 290, -60, 60),
        ("winter", "mean"): (35, 145, -30, 30),
    },
    "sunlit": {
        ("summer", "max"):  (30, 300, -60, 60),
        ("summer", "mean"): (25, 150, -35, 35),
        ("winter", "max"):  (25, 270, -60, 60),
        ("winter", "mean"): (25,  85, -15, 15),
    },
}


def resolve_run(key):
    """Use the corrected (occluder+eq+eq-indirect) run if it exists yet,
    otherwise fall back to the 2-orbit production run. Lets the figure
    regenerate incrementally as each site's re-run completes."""
    if key in RUN_DIR_ABS:               # explicit absolute override (--run-dir)
        return Path(RUN_DIR_ABS[key])
    ov = RUN_OVERRIDE.get(key)
    if ov and (OUTROOT / ov / "mesh.ply").exists():
        return OUTROOT / ov
    return OUTROOT / f"divval_{key}_cr"


def season_masks(run_dir: Path):
    """Boolean summer/winter masks aligned to the sorted per-step temperatures."""
    utcs = []
    for ln in (run_dir / "time_steps.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        if len(p) > 1:
            utcs.append(p[1])
    lats = np.array([_subsolar_lat(u) for u in utcs])
    return lats < 0.0, lats >= 0.0, lats


def _subsolar_lat(utc):
    et = sp.str2et(utc)
    spoint, _, _ = sp.subslr("Near point: ellipsoid", "MOON", et, "IAU_MOON", "LT+S", "SUN")
    return np.degrees(sp.reclat(spoint)[2])


def model_seasonal(run: Path, lat0, lon0, mask):
    """Per-cell MAX and MEAN footprint Tbol over the masked (season) timesteps."""
    x0, y0 = TO_XY.transform(lon0, lat0)
    ex = np.arange(x0 - WINDOW_KM*1e3, x0 + WINDOW_KM*1e3 + CELL_M, CELL_M)
    ey = np.arange(y0 - WINDOW_KM*1e3, y0 + WINDOW_KM*1e3 + CELL_M, CELL_M)
    nx, ny = len(ex) - 1, len(ey) - 1
    cx, cy = 0.5*(ex[:-1]+ex[1:]), 0.5*(ey[:-1]+ey[1:])
    CX, CY = np.meshgrid(cx, cy)
    core = (np.hypot(CX - x0, CY - y0) <= CORE_KM*1e3)

    v, f, flat, flon = read_ply(str(run / "mesh.ply"))
    area, flat, flon = face_geometry(v, f, flat, flon)
    fx, fy = TO_XY.transform(flon, flat)
    rkm = np.hypot(fx - x0, fy - y0) / 1e3
    ix = np.floor((fx - ex[0]) / CELL_M).astype(int)
    iy = np.floor((fy - ey[0]) / CELL_M).astype(int)
    keep = (rkm <= CORE_KM) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = iy[keep]*nx + ix[keep]
    A = area[keep]
    Asum = np.bincount(lin, weights=A, minlength=nx*ny)

    steps = sorted((run / "temperature").glob("temperature_*.bin"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    idx = np.where(mask[:len(steps)])[0]
    best = np.zeros(nx*ny)
    ssum = np.zeros(nx*ny)
    for k in idx:
        T = load_step(steps[k])[keep]
        num = np.bincount(lin, weights=A*(T.astype(np.float64)**4), minlength=nx*ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            tb = EPS4 * np.where(Asum > 0, num/Asum, 0.0)**0.25
        best = np.maximum(best, tb)
        ssum += tb
    mx = np.where(Asum > 0, best, np.nan).reshape(ny, nx)
    mn = np.where(Asum > 0, ssum/max(1, len(idx)), np.nan).reshape(ny, nx)
    return dict(max=mx, mean=mn, core=core, ex=ex, ey=ey, x0=x0, y0=y0,
                nx=nx, ny=ny, n_steps=len(idx))


def diviner_seasonal(npy, grid):
    """Bin a Diviner seasonal product (clat,clon,tbol) onto the model grid."""
    d = np.load(npy)                                  # (N,3): clat, clon, tbol
    clat, clon, tb = d[:, 0], d[:, 1], d[:, 2]
    dx, dy = TO_XY.transform(clon, clat)
    ex, ey, nx, ny = grid["ex"], grid["ey"], grid["nx"], grid["ny"]
    ix = np.floor((dx - ex[0]) / CELL_M).astype(int)
    iy = np.floor((dy - ey[0]) / CELL_M).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(tb) & (tb > 0)
    lin = iy[ok]*nx + ix[ok]
    s = np.bincount(lin, weights=tb[ok].astype(np.float64), minlength=nx*ny)
    c = np.bincount(lin, minlength=nx*ny)
    with np.errstate(invalid="ignore"):
        g = np.where(c > 0, s/np.maximum(c, 1), np.nan).reshape(ny, nx)
    return g


def stats(model, div, core):
    valid = core & np.isfinite(model) & np.isfinite(div)
    r = (model - div)[valid]
    if r.size < 3:
        return dict(n=int(r.size), bias=np.nan, rmse=np.nan)
    return dict(n=int(r.size), bias=float(r.mean()), rmse=float(np.sqrt((r**2).mean())))


def fill_internal_gaps(arr):
    """Fill NaN holes interior to the valid data (small mesh-cell gaps that
    render as white pixels) by linear interpolation, leaving the exterior
    (outside the data hull) as NaN. Display only — never feeds the statistics."""
    valid = np.isfinite(arr)
    if valid.sum() < 4 or valid.all():
        return arr
    from scipy.interpolate import griddata
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    holes = ~valid
    out = arr.copy()
    out[holes] = griddata((yy[valid], xx[valid]), arr[valid],
                          (yy[holes], xx[holes]), method="linear")
    return out


def _context_grids(lat0, lon0):
    """LOLA elevation + Diviner summer-max binned on a +/-OVERVIEW_ROI_KM stereo
    grid — the wider-region site-overview context for the manuscript Fig. 8 top
    row. Reuses the diviner_se_rim_paper.py recipe: LOLA is nearest-sampled from
    the south-polar-stereo DEM (DN->m via x0.5), Diviner summer-max is queried by
    nearest neighbor (the level-4 product is a sparse point cloud; binning leaves
    white gaps), masked where the nearest observation is > 1.5 km away."""
    import rasterio
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    x0, y0 = TO_XY.transform(lon0, lat0)
    n = 300
    gx = np.linspace(x0 - OVERVIEW_ROI_KM * 1e3, x0 + OVERVIEW_ROI_KM * 1e3, n)
    gy = np.linspace(y0 - OVERVIEW_ROI_KM * 1e3, y0 + OVERVIEW_ROI_KM * 1e3, n)
    GX, GY = np.meshgrid(gx, gy)
    src = rasterio.open(LOLA)
    inv = ~src.transform
    elev = src.read(1).astype(float)
    col, row = inv * (GX.ravel(), GY.ravel())
    EL = map_coordinates(elev, [row, col], order=1, mode="constant",
                         cval=np.nan).reshape(GX.shape) * 0.5   # DN -> m
    d = np.load(SEASDIR / "polar_south_80_summer_max_polar.npy")
    good = np.isfinite(d[:, 2]) & (d[:, 2] > 0)
    dx, dy = TO_XY.transform(d[good, 1], d[good, 0])
    tree = cKDTree(np.column_stack([dx, dy]))
    dist, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    DT = d[good, 2][idx].reshape(GX.shape).astype(float)
    DT[dist.reshape(GX.shape) > 1500.0] = np.nan
    ext = [-OVERVIEW_ROI_KM, OVERVIEW_ROI_KM, -OVERVIEW_ROI_KM, OVERVIEW_ROI_KM]
    return EL, DT, ext


def _figure_with_overview(lat0, lon0, pubclean=False, id_labels=False, site_key=None):
    """Manuscript Fig. 8 layout: a 1x2 site-overview row (LEFT = LOLA topography,
    RIGHT = Diviner summer Tbol,max, both over a wide ROI, with the CORE_KM
    comparison window outlined by a cyan circle + center cross) ABOVE an empty 4x3
    axes grid. Returns (fig, axes4x3); the caller fills the 4x3 exactly as in the
    default layout so the comparison grid is unchanged.

    pubclean (2026-07-22, additive, default OFF): suppress the two overview-panel
    titles ("LOLA topography" / "Diviner summer Tbol,max"), matching the same
    no-on-figure-identity-text treatment already applied to the 4x3 grid's
    per-column titles under --pubclean -- axis labels and colorbar labels (which
    already say "Elevation (m)" / "T (K)") are unaffected, and the caption carries
    the panel identity instead.

    id_labels (2026-07-28, additive, default OFF -- manuscript v5 Fig. 8 review):
    Paul's content-identification labels, allowed under the no-on-figure-notes
    rule (it bars data/stat annotations, not identity): re-adds the two overview
    titles even under pubclean, using the NEW wording "LOLA topography" /
    "Diviner $T_\\mathrm{bol}$" (replacing "Diviner summer $T_{bol,max}$"), and
    draws small unobtrusive white crater-name labels (no boxes/arrows) inside the
    LOLA topography panel from CRATER_LABELS[site_key] if present."""
    import matplotlib.gridspec as gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LightSource
    EL, DT, cext = _context_grids(lat0, lon0)
    fig = plt.figure(figsize=(8.6, 14.0))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.35, 4.0], hspace=0.18)
    topgs = outer[0].subgridspec(1, 2, wspace=0.45)
    axT = fig.add_subplot(topgs[0, 0])
    ls = LightSource(azdeg=315, altdeg=30)
    hs = ls.hillshade(np.nan_to_num(EL, nan=np.nanmean(EL)), vert_exag=2.0,
                      dx=200, dy=200)
    e0, e1 = np.nanpercentile(EL, [1, 99])
    axT.imshow(hs, origin="lower", extent=cext, cmap="gray")
    axT.imshow(EL, origin="lower", extent=cext, cmap="terrain", alpha=0.55,
               vmin=e0, vmax=e1)
    fig.colorbar(ScalarMappable(Normalize(e0, e1), "terrain"), ax=axT, shrink=0.85,
                 pad=0.03, label="Elevation (m)")
    if id_labels or not pubclean:
        axT.set_title("LOLA topography", fontsize=11)
    if id_labels:
        for name, dx, dy, ha in CRATER_LABELS.get(site_key, []):
            axT.text(dx, dy, name, color="white", fontsize=8, fontweight="bold",
                     ha=ha, va="center", zorder=5)
    axD = fig.add_subplot(topgs[0, 1])
    dl, dh = np.nanpercentile(DT[np.isfinite(DT)], [2, 98])
    imD = axD.imshow(DT, origin="lower", extent=cext, cmap="inferno",
                     vmin=dl, vmax=dh)
    fig.colorbar(imD, ax=axD, shrink=0.85, pad=0.03, label="$T$ (K)")
    if id_labels:
        axD.set_title(r"Diviner $T_\mathrm{bol}$", fontsize=11)
    elif not pubclean:
        axD.set_title("Diviner summer $T_{bol,max}$", fontsize=11)
    for ax in (axT, axD):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(CORE_KM * np.cos(th), CORE_KM * np.sin(th), "c-", lw=1.4)
        ax.plot(0, 0, "c+", ms=10, mew=1.8)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_aspect("equal")
    gs = outer[1].subgridspec(4, 3, hspace=0.32, wspace=0.30)
    axes = np.empty((4, 3), dtype=object)
    for r in range(4):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
    return fig, axes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", action="append", default=[], metavar="SITE=PATH",
                    help="Explicit absolute run-directory override for a site key "
                         "(repeatable). Does NOT change the built-in defaults.")
    ap.add_argument("--center", action="append", default=[], metavar="SITE=LAT,LON",
                    help="Explicit (lat0,lon0) center override for a site key "
                         "(repeatable). Does NOT change the built-in SITES defaults; "
                         "use when the mesh's actual build center differs from the "
                         "table (e.g. --center sunlit=-88.1,6.3 for fig8_hybrid).")
    ap.add_argument("--site", action="append", default=[], metavar="KEY",
                    help="Restrict to these site key(s) (repeatable). Default: all.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: docs/figures/diviner_validation/seasonal).")
    ap.add_argument("--tag", default="",
                    help="Filename suffix for figures/JSON (avoids overwriting).")
    ap.add_argument("--figstem", default=None,
                    help="Override the per-site figure basename (single-site use only).")
    ap.add_argument("--overview", action="store_true",
                    help="Prepend the manuscript Fig. 8 site-overview row (LOLA "
                         "topography + Diviner summer Tbol,max over a wide ROI, "
                         "comparison window outlined) above the 4x3 grid. Single "
                         "site only; the 4x3 comparison grid is unchanged.")
    ap.add_argument("--pubclean", action="store_true",
                    help="Publication-clean variant (Paul's no-on-figure-notes rule, "
                         "checklist P2.8, 2026-07-22): suppress the figure suptitle, "
                         "the per-column titles (Diviner/PTM3D/model-Div), and the "
                         "bias/RMSE text box. Axis labels, colorbars, and legends are "
                         "unaffected. Default OFF -- no behavior change for existing "
                         "callers/figures.")
    ap.add_argument("--pubclean-canvas-px", default=None, metavar="WxH",
                    help="With --pubclean, pin the saved canvas to this exact pixel "
                         "size at 300 dpi (e.g. 2311x2974, the annotated sibling's own "
                         "shipped size) instead of letting the tight-bbox crop shrink "
                         "when the title/stat-box text is removed -- so the manuscript "
                         "can swap the two files 1:1 without a layout reflow.")
    ap.add_argument("--id-labels", action="store_true",
                    help="Content-IDENTIFICATION labels (Paul's manuscript v5 Fig. 8 "
                         "review, 2026-07-28) -- allowed under the no-on-figure-notes "
                         "rule, which bars data/stat annotations, not identity. Works "
                         "WITH --pubclean (keeps the suptitle/stat-box suppressed) and "
                         "adds: small per-column headers 'Diviner'/'model'/'difference' "
                         "on the 4x3 grid's top row; '--overview' panel titles 'LOLA "
                         "topography' / 'Diviner $T_\\mathrm{bol}$' (new wording, "
                         "replaces the old 'Diviner summer $T_{bol,max}$'); and small "
                         "white crater-name labels inside the LOLA panel from "
                         "CRATER_LABELS. Default OFF -- no behavior change otherwise.")
    args = ap.parse_args(argv)
    for spec in args.run_dir:
        if "=" not in spec:
            ap.error(f"--run-dir expects SITE=PATH, got {spec!r}")
        k, p = spec.split("=", 1)
        RUN_DIR_ABS[k] = p
    for spec in args.center:
        if "=" not in spec or "," not in spec:
            ap.error(f"--center expects SITE=LAT,LON, got {spec!r}")
        k, ll = spec.split("=", 1)
        latstr, lonstr = ll.split(",", 1)
        CENTER_OVERRIDE[k] = (float(latstr), float(lonstr))
    site_filter = set(args.site) or None
    tag = args.tag
    outdir = (Path(args.outdir) if args.outdir
              else REPO / "docs/figures/diviner_validation/seasonal")
    outdir.mkdir(parents=True, exist_ok=True)
    sites = [s for s in SITES if site_filter is None or s[1] in site_filter]
    if args.figstem is not None and len(sites) != 1:
        ap.error("--figstem is only valid with exactly one --site")
    if args.overview and len(sites) != 1:
        ap.error("--overview is only valid with exactly one --site")

    apply_style()
    sp.furnsh(METAKERNEL)
    summary = {}
    for disp, key, lat0, lon0 in sites:
        if key in CENTER_OVERRIDE:
            olat, olon = CENTER_OVERRIDE[key]
            print(f"[{key}] CENTER OVERRIDE: table default ({lat0},{lon0}) -> "
                  f"({olat},{olon}) via --center")
            lat0, lon0 = olat, olon
        run = resolve_run(key)
        if not (run / "mesh.ply").exists():
            print(f"skip {key}: no run"); continue
        print(f"[{key}] run = {run.name}")
        smask, wmask, lats = season_masks(run)
        print(f"[{key}] summer={int(smask.sum())} winter={int(wmask.sum())} steps")
        mods = {"summer": model_seasonal(run, lat0, lon0, smask),
                "winter": model_seasonal(run, lat0, lon0, wmask)}
        rows = []      # (season, stat, model, div, st)
        for season in ("summer", "winter"):
            g = mods[season]
            for stat_name in ("max", "mean"):
                dv = diviner_seasonal(
                    SEASDIR / f"polar_south_80_{season}_{'max' if stat_name=='max' else 'avg'}_polar.npy", g)
                model = np.where(g["core"], g[stat_name], np.nan)
                dvm = np.where(g["core"], dv, np.nan)
                st = stats(model, dvm, g["core"])
                rows.append((season, stat_name, model, dvm, st, g))
                summary.setdefault(key, {})[f"{season}_{stat_name}"] = st
                print(f"   {season:6s} {stat_name:4s}: N={st['n']} bias={st['bias']:+.1f} RMSE={st['rmse']:.1f}")

        # per-site figure: 4 rows x 3 cols, optionally with the manuscript Fig. 8
        # site-overview row (LOLA topo + Diviner summer Tbol,max) prepended above.
        if args.overview:
            fig, axes = _figure_with_overview(lat0, lon0, pubclean=args.pubclean,
                                              id_labels=args.id_labels, site_key=key)
        else:
            fig, axes = plt.subplots(4, 3, figsize=(8.0, 10.2))
        for r, (season, stat_name, model, dvm, st, g) in enumerate(rows):
            ext = [(g["ex"][0]-g["x0"])/1e3, (g["ex"][-1]-g["x0"])/1e3,
                   (g["ey"][0]-g["y0"])/1e3, (g["ey"][-1]-g["y0"])/1e3]
            # Per-site color scales tuned to each site's distribution.
            vlo, vhi, dlo, dhi = SCALES_BY_SITE[key][(season, stat_name)]
            # Fill the small interior mesh-cell gaps in the model maps so they
            # do not render as white pixels (display only; stats use raw model).
            model_disp = np.where(g["core"], fill_internal_gaps(model), np.nan)
            for c, (arr, cmap, norm) in enumerate([
                    (dvm, "inferno", Normalize(vlo, vhi)),
                    (model_disp, "inferno", Normalize(vlo, vhi)),
                    (model_disp-dvm, "RdBu_r", TwoSlopeNorm(0.0, dlo, dhi))]):
                ax = axes[r, c]
                im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, norm=norm)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="K" if c < 2 else "$\\Delta$K")
                if c > 0: ax.tick_params(labelleft=False)
                if r == 0:
                    if args.id_labels:
                        ax.set_title(["Diviner", "model", "difference"][c], fontsize=9)
                    elif not args.pubclean:
                        ax.set_title(["Diviner", "PTM3D", "model$-$Div"][c], fontsize=10)
            axes[r, 0].set_ylabel(f"{season} {stat_name}\ny (km)", fontsize=9)
            if not args.pubclean:
                axes[r, 2].text(0.03, 0.96, f"bias {st['bias']:+.1f}\nRMSE {st['rmse']:.1f}",
                                transform=axes[r, 2].transAxes, va="top", fontsize=7.5,
                                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        for c in range(3):
            axes[3, c].set_xlabel("x (km)")
        slug = disp.lower().replace(" ", "_")
        stem = args.figstem if args.figstem is not None else f"seasonal_{slug}"
        out = outdir / f"{stem}{tag}"
        if args.overview:
            # Match the paper's Fig. 8: no overall suptitle; GridSpec + tight bbox.
            # --overview never adds a suptitle itself, so this branch is safe
            # regardless of --pubclean; but --pubclean + --pubclean-canvas-px
            # still needs the same pinned-canvas treatment as the non-overview
            # path below (2026-07-22, additive: --overview + --pubclean was never
            # exercised together before this, so this is new, not a regression).
            if args.pubclean and args.pubclean_canvas_px:
                # tight_layout() is a no-op here (confirmed: matplotlib warns
                # "Axes that are not compatible with tight_layout" and the
                # rendered output was pixel-identical with/without the call) --
                # the nested outer/topgs/gs GridSpec plus the standalone
                # fig.colorbar() axes aren't the simple single-level grid
                # tight_layout knows how to repack. A naive uncropped full-Bbox
                # save at the pin target therefore keeps the ORIGINAL (title-
                # sized) top margin even though the two overview titles are
                # gone under --pubclean, leaving a large blank gap. Fix: measure
                # the actual tight content bbox directly (get_tightbbox) and
                # center a fixed-size (w_in, h_in) window on it, so the pin
                # target is filled the same way a natural tight crop would fill
                # it, instead of showing raw canvas margin.
                wpx, hpx = (int(v) for v in args.pubclean_canvas_px.lower().split("x"))
                w_in, h_in = wpx / 300.0, hpx / 300.0
                fig.set_size_inches(w_in, h_in)
                fig.canvas.draw()
                tbb = fig.get_tightbbox(fig.canvas.get_renderer())
                cx = tbb.x0 + tbb.width / 2.0
                cy = tbb.y0 + tbb.height / 2.0
                from matplotlib.transforms import Bbox
                full = Bbox([[cx - w_in / 2.0, cy - h_in / 2.0],
                             [cx + w_in / 2.0, cy + h_in / 2.0]])
                for fmt, dpi in ((".png", 300), (".pdf", None)):
                    fig.savefig(out.with_suffix(fmt), dpi=dpi, bbox_inches=full)
            else:
                for fmt, dpi in ((".png", 300), (".pdf", None)):
                    fig.savefig(out.with_suffix(fmt), dpi=dpi, bbox_inches="tight")
        elif args.pubclean:
            fig.tight_layout(rect=(0, 0, 1, 1))
            if args.pubclean_canvas_px:
                # Pin the canvas to the annotated sibling's own shipped pixel size
                # (tight-bbox alone crops SHORTER once the suptitle/titles/stat box
                # are gone, since there's less content at the top edge -- pinning
                # keeps the two files swappable 1:1 without a manuscript reflow).
                wpx, hpx = (int(v) for v in args.pubclean_canvas_px.lower().split("x"))
                w_in, h_in = wpx / 300.0, hpx / 300.0
                fig.set_size_inches(w_in, h_in)
                # rcParams["savefig.bbox"]="tight" (ptm_pubstyle) still crops by
                # default even when bbox_inches is left unset; passing None does
                # NOT disable it (None just falls back to the rcParam). Pass an
                # explicit full-canvas Bbox to force the pinned size verbatim.
                from matplotlib.transforms import Bbox
                full = Bbox([[0, 0], [w_in, h_in]])
                for fmt, dpi in ((".png", 300), (".pdf", None)):
                    fig.savefig(out.with_suffix(fmt), dpi=dpi, bbox_inches=full)
            else:
                for fmt, dpi in ((".png", 300), (".pdf", None)):
                    fig.savefig(out.with_suffix(fmt), dpi=dpi)
        else:
            fig.suptitle(f"PTM3D vs Diviner seasonal $T_{{bol}}$ — {disp}",
                         fontsize=12, fontweight="bold", y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            for fmt, dpi in ((".png", 300), (".pdf", None)):
                fig.savefig(out.with_suffix(fmt), dpi=dpi)
        plt.close(fig)
        print(f"[{key}] wrote {out.with_suffix('.png')}")

    if summary:
        (outdir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))
    else:
        print("WARNING: no runs resolved; summary.json NOT written (nothing to compare)")

    # cross-site summary bar chart (skip when fewer than two sites are selected)
    if len(summary) < 2:
        print("\n=== SEASONAL SUMMARY (bias / RMSE, K) ===")
        for k, v in summary.items():
            print(f"  {k}:")
            for kk, st in v.items():
                print(f"     {kk:14s} N={st['n']:5d} bias={st['bias']:+6.1f} "
                      f"RMSE={st['rmse']:5.1f}")
        sp.kclear()
        return summary
    order = [(k, lat) for (disp, k, lat, lon) in SITES if k in summary]
    keys = [k for k, _ in order]
    dlabel = {k: d for d, k, *_ in SITES}
    combos = [("summer_max", "summer max"), ("summer_mean", "summer mean"),
              ("winter_max", "winter max"), ("winter_mean", "winter mean")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(keys)); w = 0.2
    for i, (ck, lab) in enumerate(combos):
        b = [summary[k][ck]["bias"] for k in keys]
        r = [summary[k][ck]["rmse"] for k in keys]
        ax1.bar(x + (i - 1.5)*w, b, w, label=lab)
        ax2.bar(x + (i - 1.5)*w, r, w, label=lab)
    for ax, ttl, yl in [(ax1, r"Bias (model $-$ Diviner)", "bias (K)"),
                        (ax2, "RMSE", "RMSE (K)")]:
        ax.set_xticks(x); ax.set_xticklabels([dlabel[k] for k in keys], rotation=20,
                                             ha="right", fontsize=8)
        ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10); ax.axhline(0, color="k", lw=0.8)
    ax1.legend(fontsize=7, ncol=2)
    fig.suptitle(r"PTM3D vs Diviner seasonal $T_{bol}$ (Williams et al. 2019) — summary",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(outdir / (f"seasonal_summary{tag}" + fmt), dpi=dpi)
    plt.close(fig)

    print("\n=== SEASONAL SUMMARY (bias / RMSE, K) ===")
    for k, v in summary.items():
        print(f"  {k}:")
        for kk, st in v.items():
            print(f"     {kk:14s} N={st['n']:5d} bias={st['bias']:+6.1f} RMSE={st['rmse']:5.1f}")
    sp.kclear()
    return summary


if __name__ == "__main__":
    main()
