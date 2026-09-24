#!/usr/bin/env python3
"""ema_pretty_figs.py — reusable publication-quality figure/animation styling
for EMA small-body thermal+camera products, built on Paul Hayne's
``pretty_plots`` package (https://github.com/phayne/pretty-plots, the
``planetary`` family).

Written for the 623 Chimaera conference-talk figure/animation revamp
(2026-07-04); designed to be re-run for the other EMA flyby targets and for
Justitia by just changing ``target``/``EMA_ROOT`` below.

Data contract (PTM3D outputs, empirically confirmed against on-disk data,
NOT assumed):

* Per-face binary arrays (``statistics/*.bin``, ``temperature/temperature_%04d.bin``,
  etc.): 4-byte little-endian int32 face count header, followed by
  ``float32[n_faces]`` — see :func:`read_face_bin`.
* ``mesh.ply``: PLY with a ``face`` element carrying ``vertex_indices`` PLUS
  precomputed per-face ``latitude``/``longitude`` (degrees, body-fixed frame
  that is static across the run — the sun is what rotates through it) — see
  :func:`read_mesh_lonlat`.
* ``body_positions.csv``: one row per timestep, columns
  ``sun_x, sun_y, sun_z`` = Sun position in that SAME body-fixed rotating
  frame (confirmed empirically: the (x, y) position angle advances by
  exactly 360°/period_hours per timestep, i.e. it is NOT inertial/J2000).
  Local solar time is therefore ``mod((lon_face - atan2(sun_y, sun_x))/15 + 12, 24)``
  — sign fixed empirically by requiring the diurnal peak lag AFTER local
  noon (thermal inertia), not before.
* Camera FITS frames (``frame_%04d.fits``): multi-extension, ``RADIANCE``
  (primary), ``THERMAL``, ``FACE_IDX`` (int32, ``-1`` = off-body background
  pixel), ``INC_ANG``, ``EMI_ANG``, ``PHA_ANG``, ``TEMP_K``, ``DIST_KM``,
  ``LATITUDE``, ``LONGITUDE``, plus WCS + camera-geometry header keys
  (``UTC_TIME``, ``CAM_X/Y/Z``, ``FOV_X/Y``, ...).

Reconstruction note (2026-07-04, Rocky): the ORIGINAL scripts that produced
the pre-revamp figures are gone (were /tmp scratch files from an earlier,
now-ended session) and did not cache their intermediate curve arrays for
``tvslocaltime.png`` or the on-grid ``sweep_grid_fit_curves.png`` panel. I
validated a straightforward, fully-documented reduction — mean +/- 1 sigma
per 1-hour local-solar-time bin over ALL faces with |lat| <= 25 deg and ALL
timesteps in a run — directly against the ``lst``/``clean_curve`` arrays
that ARE cached for the off-grid TI=75/200 forward sims
(``ti_fit_offgrid/fit_results_full.json``); it reproduces the correct phase
(peak lagging local noon) but not the original's larger, jaggier amplitude,
which looks like it tracked a single representative facet rather than a
spatial average (correlation ~0.75, not exact). Rather than fabricate an
unverifiable near-match, this module recomputes tvslocaltime.png and the
sweep-grid figure as this clean, fully-defensible band-average statistic
from the real per-face data (a legitimate, different but equally valid "T
vs local time" diagnostic), and reserves exact reuse for the figures whose
underlying arrays ARE cached on disk (ti_recovery_fit, ti_recovery_offgrid,
offgrid_noisy_curves — those are redrawn byte-faithful to the cached JSON).
See SESSION-NOTES / memory for the full writeup.
"""

from __future__ import annotations

import csv
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pretty_plots  # noqa: F401  (side-effect: registers style families)
from pretty_plots import planetary

try:
    from plyfile import PlyData
except ImportError:  # pragma: no cover
    PlyData = None

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None

try:
    from scipy.spatial import ConvexHull
except ImportError:  # pragma: no cover
    ConvexHull = None

# --------------------------------------------------------------------------
# Paths (EMA project map conventions — see ~/.claude/ptm-project-knowledge.md)
# --------------------------------------------------------------------------

EMA_SCIENCE_ROOT = Path(
    "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science"
)


def thermal_dir(target: str) -> Path:
    return EMA_SCIENCE_ROOT / "thermal" / target


def camera_dir(target: str) -> Path:
    return EMA_SCIENCE_ROOT / "camera" / target


# --------------------------------------------------------------------------
# Instrument label (2026-07-04 conference revamp, round 2)
# --------------------------------------------------------------------------
# The IR/thermal camera product (ECAM-IR3S, incl. its own reflected channel)
# is branded "EMACS IR Cam" in all pub titles — this is the single source of
# truth; every title-building call site below reads this constant rather than
# spelling out the instrument name. Applies to the IR3S-derived animations
# (thermal + IR3S's own reflected channel) only. CNAC (ECAM-C50) is a
# different instrument and keeps its own clean label — do NOT apply this
# constant there. The diurnal-cycle animation (`render_diurnal_animation`) is
# a DIRECT thermal-model surface-temperature render, not an instrument
# simulation, so it intentionally does not carry any camera/instrument label.
IR_CAM_LABEL = "EMACS/IR Cam"

# --------------------------------------------------------------------------
# Smooth-thermal (Gouraud) rendering support (2026-07-04, Rocky/Ada)
# --------------------------------------------------------------------------
# ptm_camera_sim (binary md5 7004db7169839bc76557967a0ee569c4+) writes an
# ADDITIVE ``TEMP_K_SMOOTH`` FITS extension: the same facet-mean temperature
# field, but barycentric-interpolated (Gouraud) from area-weighted per-vertex
# temperatures, recomputed every rendered epoch. The raw flat ``TEMP_K`` is
# always still present and unchanged. Consumers below prefer TEMP_K_SMOOTH
# when present and fall back to flat TEMP_K on older FITS (do-no-harm for any
# camera sequence rendered on a pre-smooth-thermal binary).
HONESTY_CAPTION = (
    "Facet temperatures Gouraud-interpolated for display; native model resolution 1144 facets.\n"
    "Shadow/terminator edges rendered sharp."
)


def _pick_thermal_ext(frame_files, preferred="TEMP_K_SMOOTH", fallback="TEMP_K"):
    """Return `preferred` if present in the first frame's FITS, else `fallback`."""
    if not frame_files:
        return fallback
    with fits.open(frame_files[0]) as hdul:
        names = {h.name for h in hdul}
    return preferred if preferred in names else fallback


def _format_utc_clean(utc_str) -> str:
    """Format an ISO-8601 UTC timestamp (with or without fractional seconds)
    as a clean, fixed-width ``YYYY-MM-DD HH:MM:SS UTC`` string for on-frame
    overlays."""
    if not utc_str:
        return ""
    from datetime import datetime

    dt = datetime.fromisoformat(str(utc_str))
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# --------------------------------------------------------------------------
# pretty_plots styling — planetary family + slide-tuned overrides
# --------------------------------------------------------------------------

PUB_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # NOTE: matplotlib's tight-bbox savefig crop can shave a few px off tall
    # title ascenders/parentheses at these larger publication font sizes if
    # pad_inches is too small — 0.15 was empirically verified not to clip
    # (tested down to 0.12) on the longest title used here. Don't reduce
    # this without re-checking for title clipping.
    "savefig.pad_inches": 0.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
}


def setup_style() -> None:
    """Apply pretty_plots' ``planetary`` family, then layer slide-tuned
    overrides (title ~19pt, labels 16pt, ticks 13.5pt, legend 13pt, 300 DPI,
    vector-safe PDF fonts) on top."""
    planetary.use()
    plt.rcParams.update(PUB_RC)


def save_pub(fig, out_stub, formats=("png", "pdf"), dpi=300):
    """Save ``fig`` to ``out_stub.<fmt>`` for each format; returns the list
    of written Paths. Non-destructive: never overwrites anything outside
    the given ``out_stub`` directory."""
    out_stub = Path(out_stub)
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_stub.with_suffix("." + fmt)
        fig.savefig(p, dpi=dpi if fmt == "png" else None)
        written.append(p)
    return written


# --------------------------------------------------------------------------
# Raw PTM3D per-face binary + mesh + geometry readers
# --------------------------------------------------------------------------


def read_face_bin(path) -> np.ndarray:
    """Read a PTM3D per-face binary (4-byte int32 face-count header +
    float32[n_faces])."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[:4], dtype="<i4")[0])
    return np.frombuffer(raw[4 : 4 + 4 * n], dtype="<f4").copy()


def read_mesh_lonlat(mesh_ply_path):
    """Return (lon_deg, lat_deg) per face from a PTM3D mesh.ply (body-fixed,
    static — the sun rotates through this frame, not the mesh)."""
    if PlyData is None:
        raise RuntimeError("plyfile is required (pip install plyfile)")
    ply = PlyData.read(str(mesh_ply_path))
    faces = ply["face"]
    lon = np.asarray(faces["longitude"], dtype=float)
    lat = np.asarray(faces["latitude"], dtype=float)
    return lon, lat


def read_mesh_verts_faces(mesh_ply_path):
    """Return (verts (N,3) float, faces (M,3) int) from a PTM3D mesh.ply —
    the body-fixed vertex positions and triangle connectivity, for direct 3D
    rendering (as opposed to :func:`read_mesh_lonlat`, which only returns the
    precomputed per-face lon/lat used for local-solar-time statistics)."""
    if PlyData is None:
        raise RuntimeError("plyfile is required (pip install plyfile)")
    ply = PlyData.read(str(mesh_ply_path))
    v = ply["vertex"]
    verts = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(float)
    faces = np.stack(ply["face"]["vertex_indices"]).astype(int)
    return verts, faces


def read_subsolar_lon_deg(body_positions_csv) -> np.ndarray:
    """Per-timestep subsolar body-fixed longitude (deg) from body_positions.csv."""
    lons = []
    with open(body_positions_csv) as f:
        for row in csv.DictReader(f):
            lons.append(np.degrees(np.arctan2(float(row["sun_y"]), float(row["sun_x"]))))
    return np.asarray(lons)


def lst_bin_diurnal(rundir, lat_max=25.0, nbins=24):
    """Body-fixed local-solar-time diurnal statistic for one PTM3D run.

    Mean +/- std of surface temperature over ALL faces with |lat| <= lat_max
    and ALL timesteps in the run, binned into ``nbins`` equal local-solar-time
    bins spanning 0-24h. See the module docstring for the LST convention and
    the reconstruction-fidelity caveat.

    Returns (bin_centers_h, mean_K, std_K, counts).
    """
    rundir = Path(rundir)
    lon, lat = read_mesh_lonlat(rundir / "mesh.ply")
    subsolar_lon = read_subsolar_lon_deg(rundir / "body_positions.csv")
    tfiles = sorted(glob.glob(str(rundir / "temperature" / "temperature_*.bin")), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    if len(tfiles) != len(subsolar_lon):
        raise RuntimeError(
            f"{rundir}: {len(tfiles)} temperature files but "
            f"{len(subsolar_lon)} body_positions rows"
        )

    mask = np.abs(lat) <= lat_max
    lon_m = lon[mask]
    binw = 24.0 / nbins

    sums = np.zeros(nbins)
    sumsq = np.zeros(nbins)
    counts = np.zeros(nbins)
    for i, fp in enumerate(tfiles):
        T = read_face_bin(fp)[mask]
        lst = np.mod((lon_m - subsolar_lon[i]) / 15.0 + 12.0, 24.0)
        idx = np.clip((lst / binw).astype(int), 0, nbins - 1)
        sums += np.bincount(idx, weights=T, minlength=nbins)
        sumsq += np.bincount(idx, weights=T * T, minlength=nbins)
        counts += np.bincount(idx, minlength=nbins)

    mean = np.divide(sums, counts, out=np.full(nbins, np.nan), where=counts > 0)
    var = np.divide(sumsq, counts, out=np.full(nbins, np.nan), where=counts > 0) - mean**2
    std = np.sqrt(np.clip(var, 0, None))
    centers = (np.arange(nbins) + 0.5) * binw
    return centers, mean, std, counts


# --------------------------------------------------------------------------
# Static figures
# --------------------------------------------------------------------------

FOOTNOTE_STYLE = dict(fontsize=8.5, color="0.35", ha="center")


def _add_footnote(fig, text):
    if not text:
        return
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.16))
    fig.text(0.5, 0.02, text, **FOOTNOTE_STYLE)


def fig_tvslocaltime(flyby_dir, out_stub, body_label="(623) Chimaera",
                      footnote=None, lat_max=25.0):
    centers, mean, std, counts = lst_bin_diurnal(flyby_dir, lat_max=lat_max)
    setup_style()
    fig, ax = planetary.subplots(figsize=(7.5, 5.2))
    ax.errorbar(centers, mean, yerr=std, fmt="o-", color="#3778bf",
                ecolor="#9ec3e0", elinewidth=1.6, capsize=3, ms=6, zorder=3)
    ax.set_xlabel("Local solar time (h)")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title(f"{body_label} — surface temperature vs. local solar time")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(alpha=0.25, linewidth=0.7)
    _add_footnote(fig, footnote)
    return save_pub(fig, out_stub), fig


def fig_sweep_grid(sweep_root, ti_list, alb_suffix, fit_results_json,
                    out_stub, body_label="(623) Chimaera", lat_max=25.0,
                    footnote=None):
    setup_style()
    fig, ax = planetary.subplots(figsize=(8.0, 5.6))
    norm = matplotlib.colors.LogNorm(vmin=min(ti_list), vmax=max(ti_list))
    cmap = cm.get_cmap("viridis")
    for ti in ti_list:
        rundir = Path(sweep_root) / f"ti{ti}_alb{alb_suffix}"
        centers, mean, std, _ = lst_bin_diurnal(rundir, lat_max=lat_max)
        color = cmap(norm(ti))
        ax.plot(centers, mean, "-o", ms=4, color=color, label=f"TI={ti} SI", zorder=3)
        ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.12,
                         linewidth=0)
    ax.set_xlabel("Local solar time (h)")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title(f"{body_label} — TI sweep grid diurnal curves (|lat| ≤ {lat_max:g}°)")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", frameon=True, ncol=2)

    if fit_results_json is not None and Path(fit_results_json).exists():
        entries = json.load(open(fit_results_json))
        lines = ["On-grid TI recovery (independent fit):"]
        for e in entries:
            lines.append(
                f"  truth {e['ti_true']:.0f} SI  →  fit {e['ti']:.0f} "
                f"[{e['ti_lo']:.0f},{e['ti_hi']:.0f}]  "
                f"(χ²/dof={e['chi2']/e['ndof']:.2f})"
            )
        ax.text(0.98, 0.03, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    _add_footnote(fig, footnote)
    return save_pub(fig, out_stub), fig


def fig_ti_recovery_ongrid(fit_results_json, out_stub,
                            body_label="(623) Chimaera", footnote=None):
    entries = json.load(open(fit_results_json))
    setup_style()
    fig, ax = planetary.subplots(figsize=(6.4, 6.0))
    lo = min(e["ti_true"] for e in entries) * 0.6
    hi = max(e["ti_true"] for e in entries) * 1.6
    ax.plot([lo, hi], [lo, hi], "k:", lw=1.5, label="1:1", zorder=1)
    for e in entries:
        yerr = np.array([[e["ti"] - e["ti_lo"]], [e["ti_hi"] - e["ti"]]])
        ax.errorbar(e["ti_true"], e["ti"], yerr=yerr, fmt="o", ms=10,
                    capsize=4, elinewidth=1.8, zorder=3,
                    label=f"truth TI={e['ti_true']:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True thermal inertia (SI)")
    ax.set_ylabel("Recovered thermal inertia (SI)")
    ax.set_title(f"{body_label} — TI recovery (Δχ² = 1 errors)")
    ax.grid(alpha=0.25, which="both", linewidth=0.6)
    ax.legend(loc="upper left", frameon=True)
    _add_footnote(fig, footnote)
    return save_pub(fig, out_stub), fig


def fig_ti_recovery_offgrid(fit_summary_json, out_stub, grid_ti=(32, 56, 100, 178, 316),
                             body_label="(623) Chimaera", footnote=None):
    summary = json.load(open(fit_summary_json))
    setup_style()
    fig, ax = planetary.subplots(figsize=(6.6, 6.2))
    all_ti = list(grid_ti) + [v["truth_ti"] for v in summary.values()]
    lo = min(all_ti) * 0.6
    hi = max(all_ti) * 1.6
    for g in grid_ti:
        ax.axvline(g, color="0.75", ls="--", lw=1.1, zorder=0)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1.5, label="1:1", zorder=1)
    colors = cm.get_cmap("tab10").colors
    for i, (key, v) in enumerate(summary.items()):
        yerr = np.array([[v["median"] - v["p16"]], [v["p84"] - v["median"]]])
        ax.errorbar(v["truth_ti"], v["median"], yerr=yerr, fmt="o", ms=11,
                    color=colors[i % 10], capsize=4, elinewidth=1.8, zorder=3,
                    label=f"TI={v['truth_ti']:.0f} → {v['median']:.0f} "
                          f"[{v['p16']:.0f},{v['p84']:.0f}]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True TI (SI, off-grid forward sim)")
    ax.set_ylabel("Recovered TI (SI, median +16/-84%)")
    ax.set_title(f"{body_label} — off-grid TI recovery under IR NEDT noise")
    ax.grid(alpha=0.2, which="both", linewidth=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=11.5)
    _add_footnote(fig, footnote)
    return save_pub(fig, out_stub), fig


def fig_offgrid_noisy_curves(fit_full_json, out_stub, body_label="(623) Chimaera",
                              footnote=None):
    full = json.load(open(fit_full_json))
    setup_style()
    keys = list(full.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6 * len(keys), 5.4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        v = full[key]
        lst = np.asarray(v["lst"])
        clean = np.asarray(v["clean_curve"])
        for real in v["sample_realizations"]:
            ax.plot(lst, real["obs"], "-", color="#e08283", alpha=0.55, lw=1.3, zorder=2)
        ax.plot(lst, clean, "k-", lw=2.6, zorder=3, label=f"clean (true TI={v['truth_ti']:.0f})")
        ax.set_xlabel("Local solar time (h)")
        ax.set_ylabel("T (K)")
        ax.set_title(f"TI={v['truth_ti']:.0f}: clean + {len(v['sample_realizations'])} "
                     f"noisy IR realizations")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(loc="upper right", frameon=True)
    fig.suptitle(f"{body_label} — off-grid diurnal curves under IR noise", y=1.02)
    _add_footnote(fig, footnote)
    return save_pub(fig, out_stub), fig


# --------------------------------------------------------------------------
# Snapshot stills (camera FITS, direct render — no upscaling)
# --------------------------------------------------------------------------


def _nearest_frame_index(files, target_iso):
    from datetime import datetime
    tgt = datetime.fromisoformat(target_iso)
    best_i, best_dt = None, None
    for i, fp in enumerate(files):
        t = datetime.fromisoformat(fits.getheader(fp)["UTC_TIME"])
        dt = abs((t - tgt).total_seconds())
        if best_dt is None or dt < best_dt:
            best_dt, best_i = dt, i
    return best_i


def _masked_channel(fp, ext):
    with fits.open(fp) as hdul:
        data = hdul[ext].data.astype(float).copy()
        face_idx = hdul["FACE_IDX"].data
        header = hdul[0].header
    data[face_idx < 0] = np.nan
    return data, header


def fig_snapshot_stills(anim_dir, ext, cmap, cbar_label, out_stub, title,
                         ca_utc, offset_min=2.0, vmin=None, vmax=None):
    files = sorted(glob.glob(str(Path(anim_dir) / "frame_*.fits")))
    files = [f for f in files if "noisy" not in f]
    from datetime import datetime, timedelta

    ca_dt = datetime.fromisoformat(ca_utc)
    targets = {
        "CA − 2 min": (ca_dt - timedelta(minutes=offset_min)).isoformat(),
        "CA (closest approach)": ca_dt.isoformat(),
        "CA + 2 min": (ca_dt + timedelta(minutes=offset_min)).isoformat(),
    }

    panels = []
    for label, iso in targets.items():
        idx = _nearest_frame_index(files, iso)
        data, header = _masked_channel(files[idx], ext)
        panels.append((label, data, header))

    if vmin is None or vmax is None:
        allvals = np.concatenate([p[1][np.isfinite(p[1])].ravel() for p in panels])
        vmin = 0.0 if vmin is None else vmin
        vmax = np.nanpercentile(allvals, 99.5) if vmax is None else vmax

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="black")
    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    for ax, (label, data, header) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=15)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color="white", fontsize=13)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=11)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(title, color="white", fontsize=17, y=1.02)
    fig.patch.set_facecolor("black")
    return save_pub(fig, out_stub, formats=("png", "pdf")), fig


def _noisy_ir_temp_field(clean_fp, noisy_fp):
    """Reconstruct the GENUINELY noise-injected brightness-temperature field
    for one IR frame (2026-07-06/07, Rocky — root-cause fix).

    Root cause: ``ema_camera_noise.py`` injects IR noise into the ``THERMAL``
    (spectral radiance) HDU only; it never touches/recomputes ``TEMP_K``.
    Every consumer that read ``TEMP_K`` directly from a "noisy" FITS
    (:func:`fig_noisy_snapshot_montage`'s IR channel, and
    ``render_mp4_from_fits``'s ``noisy_ir_pub.mp4`` call in
    ``make_ousha_animations``/``make_chimaera_animations``) was therefore
    displaying the bit-identical CLEAN temperature field — confirmed
    numerically (``TEMP_K`` clean vs. noisy: max abs diff == 0.0 across all
    on-body pixels of every frame checked).

    Fix: recompute brightness temperature from the ALREADY-noised THERMAL
    radiance via the true (nonlinear) Planck inversion, using a per-pixel
    ADDITIVE calibration correction derived from the paired CLEAN frame so
    the reconstruction exactly reproduces the real ``TEMP_K`` in the
    zero-noise limit (reconciles this approximate monochromatic-Planck
    formula with whatever exact bandpass/emissivity convention
    ``ptm_camera_sim`` itself used to write ``TEMP_K`` — validated: mean
    abs discrepancy 0.42 K, max 28.9 K on a representative frame, before
    the correction is applied):

        recon_clean(px) = invplanck(THERMAL_clean(px) / EPS)
        correction(px)  = TEMP_K_clean(px) - recon_clean(px)
        T_noisy(px)     = invplanck(THERMAL_noisy(px) / EPS) + correction(px)

    A first attempt used the LINEARIZED model already implicit in
    ``ema_camera_noise.ir_sigma_T`` (``T_noisy = TEMP_K_clean +
    dL/(EPS*dplanck_dT(TEMP_K_clean))``) — it reproduces the documented
    NEDT model beautifully in the well-behaved regime (e.g. empirical
    sigma 4.15 K vs. model's 4.10 K at T~185 K) but DIVERGES catastrophically
    at cold/low-signal pixels where dPlanck/dT -> 0 (max reconstructed
    "temperature" 75,295 K on one frame, plus negative values) — a real
    artifact of extrapolating a local linear Taylor term through a strongly
    curved, asymptotically-flattening function. The full NONLINEAR
    invplanck inversion used here does not have this failure mode (Planck's
    law is intrinsically bounded/monotonic; a zero-radiance floor maps to a
    finite, very-cold temperature, not a runaway) — confirmed: reconstructed
    T is bounded to [20.5, 203.3] K across all 49 frames of this run, zero
    negative values, vs. the linear approach's [-45, 75295] K.

    Returns ``(field, mask, header)`` where ``field`` is NaN off-body (for
    ``cmap.set_bad`` masking) and ``mask`` is the on-body boolean array,
    matching :func:`_masked_channel`'s contract.
    """
    import ema_camera_noise as ecn

    with fits.open(clean_fp) as h:
        fidx_c = np.asarray(h["FACE_IDX"].data)
        tempk_c = h["TEMP_K"].data.astype(float)
        thermal_c = h["THERMAL"].data.astype(float)
        header = h[0].header
    with fits.open(noisy_fp) as h:
        fidx_n = np.asarray(h["FACE_IDX"].data)
        thermal_n = h["THERMAL"].data.astype(float)

    mask = (fidx_c >= 0) & (fidx_n >= 0)

    # Shared reconstruction kernel -- lives in ema_camera_noise so the noiser
    # (ema_camera_noise.process_fits, which now writes the recomputed TEMP_K HDU
    # directly) and this figure/movie path use ONE implementation and cannot
    # drift: recompute brightness-T from the already-noised THERMAL via the
    # nonlinear Planck inversion + per-pixel additive clean-frame calibration
    # (exact at zero noise). See ema_camera_noise.recompute_temp_k for the math.
    recon = ecn.recompute_temp_k(tempk_c, thermal_c, thermal_n, mask)

    field = np.full(tempk_c.shape, np.nan)
    field[mask] = recon[mask]
    return field, mask, header


def _bbox_crop_window(mask, fill_frac=0.8):
    """Return an integer pixel crop window ``(r0, r1, c0, c1)`` (a TRUE
    sub-array selection, no resampling) sized so the on-body bounding box
    fills ``fill_frac`` of the crop's longer side, centered on the bounding
    box, clipped to stay inside ``mask``'s own shape."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bbox_h = int(rows.max() - rows.min() + 1)
    bbox_w = int(cols.max() - cols.min() + 1)
    H, W = mask.shape
    crop = int(np.ceil(max(bbox_h, bbox_w) / fill_frac))
    crop = min(crop, H, W)
    cy = (int(rows.min()) + int(rows.max())) // 2
    cx = (int(cols.min()) + int(cols.max())) // 2
    r0 = max(0, min(H - crop, cy - crop // 2))
    c0 = max(0, min(W - crop, cx - crop // 2))
    return r0, r0 + crop, c0, c0 + crop


def _panel_gsd_m(header):
    """True ground-sample-distance (m/px) for a frame, from its own header
    (range from CAM_X/Y/Z, IFOV from FOV_X/NAXIS1) — independent of any
    crop (cropping selects a sub-array, it does not change per-pixel
    angular resolution)."""
    rng_km = float(np.sqrt(header["CAM_X"] ** 2 + header["CAM_Y"] ** 2 + header["CAM_Z"] ** 2))
    ifov_rad = np.radians(float(header["FOV_X"])) / float(header.get("NAXIS1", 640))
    return rng_km * ifov_rad * 1000.0


def fig_noisy_snapshot_montage(channel, noisy_dir, cmap, cbar_label, out_png, title,
                                ca_utc, offset_min=2.0, fill_frac=0.8, dpi=300,
                                clean_dir=None):
    """Pub-styled 3-panel (CA-2min / CA / CA+2min) snapshot montage built
    DIRECTLY from an existing noisy-frame directory (2026-07-06, Rocky;
    noise-fix + crop added 2026-07-06/07 per Paul's review).

    ``channel`` is ``"ir"`` or ``"vis"`` — since Ada's 2026-07-06/07 source
    fix landed in ``ema_camera_noise.py`` (it now writes a correctly-noised
    ``TEMP_K`` itself, via the exact shared kernel
    :func:`_noisy_ir_temp_field` implements independently — the two were
    cross-validated to agree to 7.6e-6 K), BOTH channels now read their
    named HDU directly from the noisy FITS via :func:`_masked_channel`
    (``TEMP_K`` for ``"ir"``, ``RADIANCE`` for ``"vis"``) — no
    reconstruction workaround needed anymore. ``clean_dir`` is accepted for
    backward compatibility but unused (the noisy FITS is now
    self-consistent). :func:`_noisy_ir_temp_field` is kept, unused, as the
    validated reference implementation / historical record of the
    root-cause bug it fixed (see its own docstring) — do not delete.

    Differs from :func:`fig_snapshot_stills` in that ``noisy_dir`` IS the
    noisy set itself (no "exclude noisy" filename filter — that filter
    would wrongly exclude everything here, since every path under a
    directory named ``.../noisy/`` contains the substring "noisy").
    ``vmin``/``vmax`` are computed over ALL frames in ``noisy_dir`` (fixed
    0.0 floor, 99.5th-percentile ceiling — the exact same rule
    :func:`render_mp4_from_fits` uses), not just the 3 selected panel
    frames, so the montage's color scale is an EXACT match to what the
    corresponding (once similarly fixed) ``*_pub.mp4`` animation's own
    scale would be.

    Each panel keeps the existing "CA - 2 min" / "CA (closest approach)" /
    "CA + 2 min" relative-time titles (matches Chimaera's pre-existing
    montage LAYOUT) but adds a clean per-frame
    ``YYYY-MM-DD HH:MM:SS UTC`` timestamp + true per-panel GSD underneath
    each panel, matching the pub animations' footer convention. Each panel
    is CROPPED (true integer-pixel sub-array selection, NOT resampled) so
    the body fills ~``fill_frac`` of the panel, displayed with
    ``interpolation='nearest'`` so no pixel is invented/smoothed and no
    finer resolution than the real GSD is implied.
    """
    noisy_dir = Path(noisy_dir)
    if clean_dir is None:
        clean_dir = noisy_dir.parent
    else:
        clean_dir = Path(clean_dir)

    files = sorted(glob.glob(str(noisy_dir / "frame_*.fits")))
    if not files:
        raise RuntimeError(f"no frames under {noisy_dir}")

    ext_by_channel = {"ir": "TEMP_K", "vis": "RADIANCE"}
    if channel not in ext_by_channel:
        raise ValueError(f"channel must be 'ir' or 'vis', got {channel!r}")

    datas, masks, headers = [], [], []
    for fp in files:
        d, h = _masked_channel(fp, ext_by_channel[channel])
        m = np.isfinite(d)
        datas.append(d)
        masks.append(m)
        headers.append(h)
    allvals = np.concatenate([d[np.isfinite(d)].ravel() for d in datas])
    vmin = 0.0
    vmax = np.nanpercentile(allvals, 99.5)

    from datetime import datetime, timedelta

    ca_dt = datetime.fromisoformat(ca_utc)
    targets = {
        "CA − 2 min": ca_dt - timedelta(minutes=offset_min),
        "CA (closest approach)": ca_dt,
        "CA + 2 min": ca_dt + timedelta(minutes=offset_min),
    }

    panels = []
    for label, tgt_dt in targets.items():
        idx = _nearest_frame_index(files, tgt_dt.isoformat())
        r0, r1, c0, c1 = _bbox_crop_window(masks[idx], fill_frac=fill_frac)
        crop_data = datas[idx][r0:r1, c0:c1]
        gsd_m = _panel_gsd_m(headers[idx])
        panels.append((label, crop_data, headers[idx], gsd_m, (r1 - r0, c1 - c0)))

    setup_style()
    # NOTE: unlike render_mp4_from_fits/render_diurnal_animation, this is a
    # SINGLE static image, not a video-frame sequence needing a consistent
    # pixel canvas across many saves — there is no ffmpeg-assembly reason to
    # override savefig.bbox='tight' here. The suptitle below is deliberately
    # placed at y=1.02 (matching fig_snapshot_stills' existing, working
    # convention), which relies on the *tight* bbox to expand the saved
    # canvas to include it; forcing bbox=None (tried first) clips the title
    # against a fixed canvas top edge instead. Leave setup_style()'s
    # tight+pad_inches=0.15 rcParams in effect for this save.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="black")
    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    for ax, (label, data, header, gsd_m, shape) in zip(axes, panels):
        # interpolation='nearest': the crop is a true pixel sub-array (no
        # resampling) — this guarantees matplotlib's on-screen upscaling of
        # that small array stays blocky/honest rather than smoothing in
        # values that were never measured.
        im = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="upper",
                        interpolation="nearest")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=15)
        utc_clean = _format_utc_clean(header.get("UTC_TIME", ""))
        footer = f"{utc_clean}\nGSD {gsd_m:.1f} m/px  ({shape[1]}×{shape[0]} px crop)"
        ax.text(0.5, -0.06, footer, color="white", fontsize=10.5,
                family="monospace", ha="center", va="top",
                transform=ax.transAxes)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color="white", fontsize=13)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=11)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(title, color="white", fontsize=17, y=1.02)
    fig.patch.set_facecolor("black")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="black")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# Animations — re-assembled from FITS frames, pretty-plots styled overlays
# --------------------------------------------------------------------------


def render_mp4_from_fits(frame_files, ext, cmap, cbar_label, out_mp4, title,
                          fps=12, vmin=None, vmax=None, crf=16,
                          target_width_px=1920, caption=None):
    """Render each FITS frame's `ext` channel with a pretty_plots-styled
    overlay (colorbar+units, title, per-frame UTC/range label) and assemble
    into a high-resolution, high-bitrate H.264 MP4 via ffmpeg.

    `caption`, if given, is drawn as a small persistent honesty/methods note
    in the top-left corner of every frame (distinct from the per-frame
    UTC/range footer) — e.g. `HONESTY_CAPTION` when `ext` is a Gouraud-
    smoothed field."""
    setup_style()
    # setup_style()'s planetary/PUB_RC styling sets savefig.bbox='tight',
    # which is right for the static PNG/PDF figures (save_pub) but silently
    # content-crops every per-frame PNG here to less than the intended
    # target_width_px/1080p canvas and can vary the crop given differing
    # per-frame content extents. Force a fixed, un-cropped canvas for the
    # video frames specifically (restored after, so it doesn't leak into any
    # later save_pub() call in the same process).
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    frame_files = list(frame_files)
    if not frame_files:
        raise RuntimeError("no frames given")

    # First pass: global color scale + background mask reuse.
    datas, headers = [], []
    for fp in frame_files:
        d, h = _masked_channel(fp, ext)
        datas.append(d)
        headers.append(h)
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        allvals = np.concatenate([d[np.isfinite(d)].ravel() for d in datas])
        vmax = np.nanpercentile(allvals, 99.5)

    h0, w0 = datas[0].shape
    aspect = h0 / w0
    fig_w = target_width_px / 200.0
    fig_h = fig_w * aspect * 1.02 + 0.9  # small margin for title/footer
    dpi = 200

    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_pub_frames_"))
    try:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
        im = ax.imshow(datas[0], cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color="white", fontsize=14)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        title_txt = fig.suptitle(title, color="white", fontsize=15, y=0.99)
        footer_txt = fig.text(0.02, 0.015, "", color="white", fontsize=11.5,
                               ha="left", va="bottom", family="monospace")
        # Honesty caption is appended as a second, smaller line UNDER the
        # per-frame UTC/range label (not a separate fixed text block) so it
        # can never collide with the title regardless of title length —
        # both live in the same bottom-left footer text object.

        for i, (d, h) in enumerate(zip(datas, headers)):
            im.set_data(d)
            utc = h.get("UTC_TIME", "")
            noised = h.get("NOISED", False)
            label = _format_utc_clean(utc)
            if all(k in h for k in ("CAM_X", "CAM_Y", "CAM_Z")):
                rng = float(np.sqrt(h["CAM_X"]**2 + h["CAM_Y"]**2 + h["CAM_Z"]**2))
                label += f"   range {rng:.0f} km"
            if noised:
                label += "   [noise-injected]"
            if caption:
                label += f"\n{caption}"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            # libx264/yuv420p requires even width & height; matplotlib's
            # fractional-inch figsize * dpi can round to an odd pixel count,
            # so force-crop to the nearest even dimensions.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_mp4
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


# --------------------------------------------------------------------------
# Diurnal-cycle animation — direct 3D thermal-model render (NOT camera-sim)
# --------------------------------------------------------------------------
#
# Reconstruction note (2026-07-04, Rocky): the original diurnal generator that
# made ``chimaera_diurnal_OLD-BUGGY.mp4`` is gone (another lost /tmp scratch
# script, same pattern as the tvslocaltime/sweep-grid figures — see module
# docstring). Framing was recovered empirically, NOT guessed, by diffing the
# 292 cached PNG frames under
# ``camera/623chimaera/diurnal/frames/`` against each other: the silhouette
# mask (which pixels belong to the body vs. background) is pixel-identical
# (0-2 px, anti-aliasing noise) across ALL sampled frames spanning the full
# ~1.5-rotation window, while the per-face color pattern is periodic with the
# body's rotation period (mean abs RGB diff vs frame 0 returns to ~0.1 near
# frame 194, one full rotation later, and re-peaks by frame 291). This proves
# the camera is BODY-FIXED (co-rotating with the body — the same vantage
# point every frame), not inertial: you watch the local-solar-time /
# temperature pattern sweep across a fixed silhouette, rather than watching
# the body visibly spin. Replicated below with a static ``view_init`` across
# all frames on the (already body-fixed, static) mesh vertex positions. The
# OLD-BUGGY video's own two-panel diagnostic content (a cyan "spurious
# self-shadow" overlay) was itself a screenshot of the terminator-wedge bug
# this data supersedes, so it is not reproduced — only the framing
# (single 3D body-fixed render, inferno, vmax-labeled colorbar) is kept.


def render_diurnal_animation(
    target,
    thermal_subdir="flyby",
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Surface temperature (K)",
    cmap="inferno",
    vmin=None,
    vmax=None,
    elev=15,
    azim=-45,
    fps=15,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
):
    """Render a PTM3D per-timestep temperature field as a body-fixed 3D
    diurnal-cycle animation (see the module note above for the framing
    justification) and assemble it into a pub-quality H.264 MP4.

    Parametrized by ``target`` (+ ``thermal_subdir``, default ``"flyby"``)
    so the same function serves every EMA body and Justitia — just point it
    at ``thermal_dir(target)/thermal_subdir``. Reads:
      * ``mesh.ply`` — body-fixed vertex positions + triangle connectivity
        (static across the run; see :func:`read_mesh_verts_faces`).
      * ``temperature/temperature_%04d.bin`` — per-timestep per-face T (K).
      * ``body_positions.csv`` — per-timestep UTC (``time`` column), for the
        per-frame timestamp overlay.

    ``vmin``/``vmax`` default to the run's own global min/max, rounded
    outward to the nearest 5 K, if not given explicitly.
    """
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = camera_dir(target) / "diurnal" / f"{target}_diurnal_pub.mp4"
    if title is None:
        title = f"{body_label} — surface thermal-IR temperature"

    verts, faces = read_mesh_verts_faces(tdir / "mesh.ply")
    tri_verts = verts[faces]

    tfiles = sorted(glob.glob(str(tdir / "temperature" / "temperature_*.bin")), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    if not tfiles:
        raise RuntimeError(f"no temperature_*.bin files under {tdir/'temperature'}")

    utcs = []
    with open(tdir / "body_positions.csv") as f:
        for row in csv.DictReader(f):
            utcs.append(row["time"])
    if len(utcs) != len(tfiles):
        raise RuntimeError(
            f"{tdir}: {len(tfiles)} temperature files but "
            f"{len(utcs)} body_positions rows"
        )

    Ts = [read_face_bin(fp) for fp in tfiles]
    if vmin is None or vmax is None:
        allT = np.concatenate(Ts)
        if vmin is None:
            vmin = 5.0 * np.floor(allT.min() / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(allT.max() / 5.0)

    setup_style()
    # See the matching note in render_mp4_from_fits: setup_style()'s
    # savefig.bbox='tight' would content-crop these per-frame PNGs away from
    # the intended fixed 1920x1080-class canvas. Force it off for the frame
    # loop, restore after.
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8  # empirically-tuned fill factor

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_diurnal_frames_"))
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
        ax = fig.add_axes([-0.08, 0.0, 0.95, 0.92], projection="3d")
        ax.set_facecolor("black")
        pc = Poly3DCollection(
            tri_verts, facecolors=cmap_obj(norm(Ts[0])), edgecolor="none",
            antialiased=True,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, color="white", fontsize=14)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        fig.suptitle(title, color="white", fontsize=17, y=0.97)
        footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=13,
                               ha="left", va="bottom", family="monospace")

        for i, (T, utc) in enumerate(zip(Ts, utcs)):
            pc.set_facecolor(cmap_obj(norm(T)))
            footer_txt.set_text(_format_utc_clean(utc))
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_mp4
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


# --------------------------------------------------------------------------
# CLI driver — Chimaera
# --------------------------------------------------------------------------

def _sorted_fits(dirpath):
    return sorted(glob.glob(str(Path(dirpath) / "frame_*.fits")))


def make_chimaera_static_figures():
    target = "623chimaera"
    tdir = thermal_dir(target)
    cdir = camera_dir(target)
    out = tdir / "figures_pub"
    body_label = "(623) Chimaera"
    footnote = ("CA epoch 2030-06-14T12:42:40  ·  binary 79f091e (terminator-fix)  ·  "
                "CPU  ·  χ=2.7  ·  ~15–16 K deferred night-side cold bias (see memory)")

    written = {}

    p, fig = fig_tvslocaltime(tdir / "flyby", out / "tvslocaltime",
                               body_label=body_label, footnote=footnote)
    written["tvslocaltime"] = p
    plt.close(fig)

    p, fig = fig_sweep_grid(tdir / "sweep", [32, 56, 100, 178, 316], "0.0371",
                             tdir / "ti_fit" / "fit_results.json",
                             out / "sweep_grid_fit_curves",
                             body_label=body_label, footnote=footnote)
    written["sweep_grid_fit_curves"] = p
    plt.close(fig)

    p, fig = fig_ti_recovery_ongrid(tdir / "ti_fit" / "fit_results.json",
                                     out / "ti_recovery_fit",
                                     body_label=body_label, footnote=footnote)
    written["ti_recovery_fit"] = p
    plt.close(fig)

    p, fig = fig_ti_recovery_offgrid(tdir / "ti_fit_offgrid" / "fit_results_summary.json",
                                      out / "ti_recovery_offgrid",
                                      body_label=body_label, footnote=footnote)
    written["ti_recovery_offgrid"] = p
    plt.close(fig)

    p, fig = fig_offgrid_noisy_curves(tdir / "ti_fit_offgrid" / "fit_results_full.json",
                                       out / "offgrid_noisy_curves",
                                       body_label=body_label, footnote=footnote)
    written["offgrid_noisy_curves"] = p
    plt.close(fig)

    p, fig = fig_snapshot_stills(cdir / "flyby" / "anim", "TEMP_K", "inferno",
                                  "Brightness T (K) @ 10.5 µm",
                                  out / "snapshot_stills_ir",
                                  f"{body_label} — {IR_CAM_LABEL} thermal-IR",
                                  "2030-06-14T12:42:40")
    written["snapshot_stills_ir"] = p
    plt.close(fig)

    p, fig = fig_snapshot_stills(cdir / "flyby" / "anim", "RADIANCE", "gray",
                                  "Reflected radiance (W m⁻² sr⁻¹)",
                                  out / "snapshot_stills_reflected",
                                  f"{body_label} — {IR_CAM_LABEL} reflected",
                                  "2030-06-14T12:42:40")
    written["snapshot_stills_reflected"] = p
    plt.close(fig)

    p, fig = fig_snapshot_stills(cdir / "flyby" / "visible" / "anim", "RADIANCE", "gray",
                                  "Reflected radiance (W m⁻² sr⁻¹)",
                                  out / "snapshot_stills_cnac",
                                  f"{body_label} — EMA CNAC (ECAM-C50) visible/grayscale",
                                  "2030-06-14T12:42:40")
    written["snapshot_stills_cnac"] = p
    plt.close(fig)

    return written


def make_chimaera_animations():
    target = "623chimaera"
    cdir = camera_dir(target)
    body_label = "(623) Chimaera"
    written = {}

    written["ir"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim"), "TEMP_K", "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "flyby" / "chimaera_flyby_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} thermal-IR flyby")

    written["reflected"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "chimaera_flyby_reflected_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} reflected flyby")

    written["cnac"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "visible" / "anim"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "visible" / "chimaera_cnac_visible_pub.mp4",
        f"{body_label} — EMA CNAC visible flyby")

    written["noisy_ir"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim" / "noisy"), "TEMP_K", "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "flyby" / "noisy" / "noisy_ir_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} thermal-IR flyby (noise-injected)")

    written["noisy_reflected"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim" / "noisy"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "noisy" / "noisy_visible_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} reflected flyby (noise-injected)")

    written["noisy_cnac"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "visible" / "anim" / "noisy"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "visible" / "noisy" / "noisy_visible_pub.mp4",
        f"{body_label} — EMA CNAC visible flyby (noise-injected)")

    return written


def make_ousha_animations():
    """(23871) Ousha parallel of :func:`make_chimaera_animations` — same
    renderer function, same params (cmap, channel, fps/crf/width defaults,
    title wording), just re-pointed at Ousha's own existing flyby FITS
    frames. Render/encode only, no re-noise/re-render of sims. ``vmin``/
    ``vmax`` are left at :func:`render_mp4_from_fits`'s own defaults
    (fixed floor 0.0 K, ceiling = that run's own 99.5th-percentile) —
    i.e. per-body auto-scaling, matching exactly what Chimaera's own call
    below does (Chimaera does NOT use a shared fixed T-range either)."""
    target = "23871ousha"
    cdir = camera_dir(target)
    body_label = "(23871) Ousha"
    written = {}

    written["ir"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim"), "TEMP_K", "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "flyby" / "ousha_flyby_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} thermal-IR flyby")

    written["reflected"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "ousha_flyby_reflected_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} reflected flyby")

    written["cnac"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "visible" / "anim"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "visible" / "ousha_cnac_visible_pub.mp4",
        f"{body_label} — EMA CNAC visible flyby")

    written["noisy_ir"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim" / "noisy"), "TEMP_K", "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "flyby" / "noisy" / "noisy_ir_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} thermal-IR flyby (noise-injected)")

    written["noisy_reflected"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "anim" / "noisy"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "noisy" / "noisy_visible_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} reflected flyby (noise-injected)")

    written["noisy_cnac"] = render_mp4_from_fits(
        _sorted_fits(cdir / "flyby" / "visible" / "anim" / "noisy"), "RADIANCE", "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "visible" / "noisy" / "noisy_visible_pub.mp4",
        f"{body_label} — EMA CNAC visible flyby (noise-injected)")

    return written


def make_ousha_noisy_snapshots():
    """(23871) Ousha ``flyby/noisy/{noisy_ir,noisy_visible}_snapshots.png``
    — pub-styled replacements for the pre-existing raw montages, matching
    (623) Chimaera's 3-panel CA-2min/CA/CA+2min grid layout (see
    ``.../623chimaera/flyby/noisy/noisy_ir_snapshots.png``) but with the
    CURRENT ``EMACS/IR Cam`` label + clean per-panel UTC stamps (Chimaera's
    own copy of this file is layout-reference only — it is itself
    label-stale, still reading "EMACS IR Cam" with no slash; do not copy
    that wording).

    Sourced from the SAME ``flyby/anim/noisy/`` frames the
    ``noisy_ir_pub.mp4``/``noisy_visible_pub.mp4`` animations were built
    from (see :func:`make_ousha_animations`) — both channels come from this
    one (IR-camera-sized) noisy directory, matching the project's existing
    ``flyby/noisy/noisy_visible.mp4`` = "EMACS/IR Cam's own reflected
    channel, NOT CNAC" convention. ``flyby/visible/noisy/`` (CNAC) is a
    separate product, untouched here.

    ``ca_utc`` is the empirically measured minimum-range (closest-approach)
    frame — frame_0024, range 520.5 km, found by scanning all 49 frames'
    ``CAM_X/Y/Z`` headers (same recipe :func:`render_mp4_from_fits` uses for
    the on-frame range label) — NOT assumed/guessed.

    2026-07-06/07 (Paul review, ADDENDUM): the IR panel is built via
    ``channel="ir"`` (:func:`_noisy_ir_temp_field`) rather than reading
    ``TEMP_K`` straight from the noisy FITS — the latter is bit-identical to
    the clean field (root cause of "the noisy IR snapshots don't look
    noisy"), see that function's docstring. Also each panel is now cropped
    to ~80% body fill (true pixel sub-array, nearest-neighbor display, real
    per-panel GSD reported) instead of showing the full 640x480 frame.
    """
    target = "23871ousha"
    cdir = camera_dir(target)
    body_label = "(23871) Ousha"
    ca_utc = "2032-12-25T16:47:20"
    noisy_dir = cdir / "flyby" / "anim" / "noisy"
    clean_dir = cdir / "flyby" / "anim"
    written = {}

    written["noisy_ir_snapshots"] = fig_noisy_snapshot_montage(
        "ir", noisy_dir, "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "flyby" / "noisy" / "noisy_ir_snapshots.png",
        f"{body_label} — {IR_CAM_LABEL} thermal-IR flyby (noise-injected)",
        ca_utc, clean_dir=clean_dir)

    written["noisy_visible_snapshots"] = fig_noisy_snapshot_montage(
        "vis", noisy_dir, "gray",
        "Reflected radiance (W m⁻² sr⁻¹)",
        cdir / "flyby" / "noisy" / "noisy_visible_snapshots.png",
        f"{body_label} — {IR_CAM_LABEL} reflected flyby (noise-injected)",
        ca_utc, clean_dir=clean_dir)

    return written


# --------------------------------------------------------------------------
# Coverage-accumulation animation — INERTIAL fixed view of a rotating body,
# each facet showing the MOST-RECENT camera-imaged IR brightness-T
# --------------------------------------------------------------------------
#
# New tooling (2026-07-04, Justitia proxops product). Unlike
# render_diurnal_animation (body-fixed / co-rotating camera — the body's
# silhouette stays fixed and only the temperature pattern sweeps across it),
# this renders an INERTIAL-anchored fixed viewpoint: the body visibly SPINS
# under a constant vantage direction, while each facet's color is the most
# recent thermal-IR brightness-T from whichever rendered camera frame last
# had that facet in view (most-recent-wins accumulation over time), with
# never-yet-imaged facets left neutral gray.
#
# Implementation notes:
#  * Per-facet "was this facet imaged, and at what brightness-T" for each
#    input camera frame is read DIRECTLY from that frame's own FITS
#    FACE_IDX (-1 = off-body/background, else the facet id actually hit by
#    the camera, already FOV + occlusion + smooth-shading aware) and TEMP_K
#    extensions — no separate FOV/visibility projection is reimplemented;
#    this reuses the camera simulator's own, already-validated visibility
#    determination rather than re-deriving it (a facet is "touched" this
#    frame iff >=1 of its pixels appears with FACE_IDX==that facet).
#  * "sunlit/visible" (per the task spec) is interpreted as "geometrically
#    visible to the camera" (the FACE_IDX gate), NOT an additional day-side
#    filter — this is a thermal-IR channel, which images the night side
#    with a physically valid (cold) temperature just as validly as the day
#    side; gating on solar illumination would arbitrarily blank out roughly
#    half of every frame's real IR coverage. Flagged as a judgment call.
#  * The "inertial fixed view" is built by de-spinning the (body-fixed,
#    static) mesh vertices frame-by-frame via SPICE: R(t) = pxform(J2000,
#    naif_frame, t0) @ pxform(naif_frame, J2000, t) maps body-fixed-at-t
#    vertex coordinates into the frame anchored at the first frame's
#    body-fixed orientation (R(t0) = identity by construction). The camera
#    view itself (``view_init``) is then held static across the animation,
#    same recipe as :func:`render_diurnal_animation` but applied to
#    per-frame ROTATED vertices instead of a static mesh — this is what
#    makes the body visibly rotate under a fixed vantage point.

# --------------------------------------------------------------------------
# Sub-facet swath accumulation geometry (2026-07-05, Rocky) — REWORK of the
# coverage-accumulation animation from per-FACE_IDX to per-pixel-footprint.
#
# Problem (Paul Hayne): at proxops range the pixel/FOV footprint is far
# finer than a DAMIT facet (this 1144-face Justitia mesh has facet
# sqrt(area) ranging 0.04-13 km!), so marking coverage per FACE_IDX makes a
# whole triangle "pop" fully covered the instant ANY part of it is imaged —
# reads as blocky discrete triangles, not a real imaging swath.
#
# Fix: accumulate on a per-facet ADAPTIVELY subdivided sub-triangle mesh,
# using the camera FITS' own per-pixel LATITUDE/LONGITUDE (already the
# camera sim's own ray-traced body-fixed surface coordinates, empirically
# confirmed to match mesh.ply's precomputed per-face lat/lon convention —
# direct spherical (arcsin(z/r), atan2(y,x)) on the body-fixed frame — to
# ~1e-5 deg, and to have real, continuous sub-facet spatial variation
# WITHIN a facet's footprint, not a repeated centroid value) to determine,
# per frame, which sub-triangles a frame's pixels actually swept over.
#
# Two accumulation strategies were prototyped and compared (see session
# notes for the visual comparison + numbers):
#   (a) "nearest sub-triangle centroid" (per-pixel argmax dot-product
#       against its own facet's candidate sub-triangle directions) — is
#       CORRECT (never leaks across the FACE_IDX gate) but produces a
#       sparse, speckled scatter rather than a contiguous swath, because
#       the 640x480 camera raster under-samples a fine sub-triangle grid
#       relative to a facet's true swept footprint. REJECTED for the
#       shipped animation on visual-quality grounds (kept as
#       :func:`_pixel_nearest_subtri_DEPRECATED` is NOT kept — the
#       exact-barycentric approach below fully supersedes it).
#   (b) "exact per-pixel barycentric position + per-facet-per-frame
#       bounding-box fill" — SHIPPED. Ray-plane-intersects each pixel's
#       (lat,lon)-implied body-fixed direction with its own known FACE_IDX
#       facet's plane to get an exact (u,v,w) barycentric position, then
#       marks every sub-triangle whose own (v,w) center falls within that
#       frame's own pixel (v,w) bounding box (percentile-trimmed for
#       robustness against ~0.1-1% grazing/near-limb outlier pixels) as
#       covered. Dense + contiguous + still genuinely partial (verified:
#       for the 4 proxops phases, 72-92% of ever-imaged facets are marked
#       only PARTIALLY covered, not fully — see session notes for the
#       per-phase facet partial/full/empty breakdown and a zoomed
#       before/after still).
# --------------------------------------------------------------------------


def _adaptive_subdivide_mesh(verts, faces, target_km=0.3, n_min=2, n_max=16):
    """Per-facet ADAPTIVE barycentric subdivision of a PTM3D mesh into
    sub-triangles, for sub-facet-resolution rendering/accumulation.

    Each facet f is split into ``n_facet[f]**2`` sub-triangles, with
    ``n_facet[f] = clip(round(sqrt(facet_area_km2[f]) / target_km), n_min,
    n_max)`` — i.e. roughly constant ABSOLUTE sub-triangle size
    (``target_km``), not a constant subdivision COUNT per facet. This
    matters because real small-body DAMIT meshes are far from
    uniform-facet-size: Justitia's 1144-face mesh spans sqrt(area) from
    0.04 to 13 km (a ~300x range) — a fixed global N under-resolves the
    largest outlier facets (empirically: N=12 uniform left a 13x7.6x10.6 km
    sliver facet resolved into only 5/144 candidate cells for a real
    partial-coverage frame) while wasting triangles on the many small ones.
    ``n_max`` bounds the largest facets' cost; ``target_km=0.3`` (with
    ``n_max=16``) keeps the TOTAL sub-triangle count for this mesh in the
    tens of thousands (tractable for a 66-frame ``Poly3DCollection``
    animation — NOT anywhere near true camera GSD [~5-42 m across the 4
    proxops phases], which would need tens of millions of sub-triangles
    mesh-wide and is intractable for this renderer; see the module note
    above for the explicit rendering-tech tradeoff).

    Returns:
      sub_tri_xyz: (n_sub, 3, 3) float — sub-triangle vertex positions
        (NOT a shared index buffer; matches how the existing renderers
        already feed `Poly3DCollection` directly, e.g. ``verts[faces]``).
      face_ids: (n_sub,) int — parent facet index of each sub-triangle.
      sub_vw: (n_sub, 2) float — each sub-triangle's own barycentric
        CENTER (v toward vertex B, w toward vertex C — same convention as
        :func:`_pixel_barycentric`); kept for diagnostics/plotting only —
        accumulation itself uses ``sub_vw_bounds`` (see below), NOT this.
      sub_vw_bounds: (n_sub, 4) float — each sub-triangle's own axis-aligned
        (v, w) bounding box as ``(v_lo, v_hi, w_lo, w_hi)``. **Load-bearing
        for correctness, not just an optimization**: an earlier version of
        this accumulation tested whether a sub-triangle's CENTER fell
        inside a frame's pixel (v, w) box, which silently FAILED whenever
        an imaged patch didn't happen to contain any sub-triangle's exact
        center point — empirically this dropped 350/417 (84%) of genuine
        facet-frame imaging events on the Justitia orbit_low sequence
        (facet-count coverage undercounted at 4.7% vs the true 15.4%).
        Testing AABB-overlap against each sub-triangle's own (v, w) extent
        instead is what a full tiling of the barycentric simplex actually
        guarantees will hit at least one sub-triangle (verified: recovers
        the true 15.4%/176-facets figure exactly on orbit_low, while still
        leaving all but 1 of those 176 facets marked only PARTIALLY, not
        fully, covered — see session notes).
      local_offset: (n_faces+1,) int — sub-triangles for facet f live at
        ``sub_tri_xyz[local_offset[f]:local_offset[f+1]]`` (face_ids is
        sorted to make this a simple slice, not a boolean mask, at every
        per-frame lookup).
      n_facet: (n_faces,) int — the chosen subdivision level per facet
        (diagnostic / for area-weighted coverage bookkeeping).
    """
    tri = verts[faces]
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    F = len(faces)
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    size_km = np.sqrt(area)
    n_facet = np.clip(np.round(size_km / target_km).astype(int), n_min, n_max)

    sub_tri_chunks = []
    face_id_chunks = []
    vw_chunks = []
    bounds_chunks = []
    # Group facets by n (a few dozen unique values at most) so each
    # subdivision level is built with one vectorized op across its facet
    # subset, rather than looping per-facet.
    for n in np.unique(n_facet):
        idx = np.where(n_facet == n)[0]
        Ag, Bg, Cg = A[idx], B[idx], C[idx]
        Fg = len(idx)
        # Barycentric lattice P(i,j) = A + (i/n)(B-A) + (j/n)(C-A), i+j<=n.
        grid = np.full((Fg, n + 1, n + 1, 3), np.nan)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                grid[:, i, j, :] = Ag + (i / n) * (Bg - Ag) + (j / n) * (Cg - Ag)
        for i in range(n):
            for j in range(n - i):
                # Both the "upward" and "downward" sub-triangles at lattice
                # cell (i, j) share the SAME (v, w) AABB: v in [i/n,
                # (i+1)/n], w in [j/n, (j+1)/n] (the two halves of one
                # parallelogram-shaped cell).
                vlo, vhi, wlo, whi = i / n, (i + 1) / n, j / n, (j + 1) / n
                bounds_row = np.column_stack([
                    np.full(Fg, vlo), np.full(Fg, vhi), np.full(Fg, wlo), np.full(Fg, whi)])
                # "upward" sub-triangle: P(i,j), P(i+1,j), P(i,j+1)
                t = np.stack([grid[:, i, j], grid[:, i + 1, j], grid[:, i, j + 1]], axis=1)
                sub_tri_chunks.append(t)
                face_id_chunks.append(idx)
                vw_chunks.append(np.column_stack([
                    np.full(Fg, (i + 1.0 / 3) / n), np.full(Fg, (j + 1.0 / 3) / n)]))
                bounds_chunks.append(bounds_row)
                if j < n - i - 1:
                    # "downward" sub-triangle: P(i+1,j), P(i+1,j+1), P(i,j+1)
                    t2 = np.stack([grid[:, i + 1, j], grid[:, i + 1, j + 1], grid[:, i, j + 1]], axis=1)
                    sub_tri_chunks.append(t2)
                    face_id_chunks.append(idx)
                    vw_chunks.append(np.column_stack([
                        np.full(Fg, (i + 2.0 / 3) / n), np.full(Fg, (j + 2.0 / 3) / n)]))
                    bounds_chunks.append(bounds_row)

    sub_tri_xyz = np.concatenate(sub_tri_chunks, axis=0)
    face_ids = np.concatenate(face_id_chunks, axis=0)
    sub_vw = np.concatenate(vw_chunks, axis=0)
    sub_vw_bounds = np.concatenate(bounds_chunks, axis=0)

    order = np.argsort(face_ids, kind="stable")
    sub_tri_xyz = sub_tri_xyz[order]
    face_ids = face_ids[order]
    sub_vw = sub_vw[order]
    sub_vw_bounds = sub_vw_bounds[order]
    local_offset = np.searchsorted(face_ids, np.arange(F + 1))
    return sub_tri_xyz, face_ids, sub_vw, sub_vw_bounds, local_offset, n_facet


def _pixel_barycentric(verts, faces, face_ids_px, plat_rad, plon_rad):
    """Exact ray-plane-intersection barycentric (u, v, w) of each pixel's
    surface direction within its OWN already-known ``FACE_IDX`` facet.

    The ray originates at the body-fixed-frame origin (body center) along
    the unit direction implied by (lat, lon) — this convention is
    empirically confirmed to match ``mesh.ply``'s own precomputed per-face
    lat/lon (direct spherical conversion of the 3D centroid) to ~1e-5 deg,
    so it is the SAME convention the camera sim itself used to write out
    per-pixel LATITUDE/LONGITUDE. ``v`` = barycentric weight toward vertex
    B, ``w`` = weight toward vertex C, ``u = 1 - v - w`` = weight toward A.

    A small fraction of pixels (empirically ~0.03-0.3% "wild": |v| or
    |w| > ~2; ~0.08-1.1% mildly outside [0,1], across the 4 Justitia
    proxops phases) are grazing/near-limb pixels where this ray-from-center
    reconstruction is numerically ill-conditioned (the ray nearly parallel
    to the facet's plane) — callers should guard against these (see
    ``wild_outlier_thresh``/clipping in :func:`render_coverage_accumulation_animation`),
    not trust them at face value.
    """
    tri0 = verts[faces]
    A0, B0, C0 = tri0[:, 0], tri0[:, 1], tri0[:, 2]
    normal0 = np.cross(B0 - A0, C0 - A0)

    A = A0[face_ids_px]
    Bv = B0[face_ids_px]
    C = C0[face_ids_px]
    nrm = normal0[face_ids_px]

    d = np.stack([np.cos(plat_rad) * np.cos(plon_rad),
                  np.cos(plat_rad) * np.sin(plon_rad),
                  np.sin(plat_rad)], axis=1)
    denom = np.einsum("ij,ij->i", nrm, d)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = np.einsum("ij,ij->i", nrm, A) / denom
    p = t[:, None] * d

    v0 = Bv - A
    v1 = C - A
    v2 = p - A
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return u, v, w


def _archive_existing(path, suffix="_facetbased_preSwathFix_"):
    """If ``path`` already exists, rename it (non-destructive) to a
    timestamped ``*<suffix><UTC stamp>.<ext>`` name in the same directory,
    and return the archived Path — else return None. Used so regenerating a
    coverage-accum animation never silently overwrites the prior product.
    ``suffix`` names WHY the predecessor was superseded (default keeps the
    original swath-fix convention; the round-6 parallel re-render passes its
    own suffix so the archived round-5 serial MP4s are self-describing)."""
    from datetime import datetime, timezone

    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}{suffix}{stamp}{path.suffix}")
    path.rename(archived)
    return archived


# --------------------------------------------------------------------------
# Texture-mapped (PyVista/VTK) rendering support (2026-07-05, round 2)
# --------------------------------------------------------------------------
# Paul Hayne reviewed the sub-facet-swath ANIMATION (built by the
# Poly3DCollection-based ``render_coverage_accumulation_animation`` above,
# now renamed ``..._subfacet_polygon_legacy``) and found it still reads as
# "scattered sliver-like triangles filling in, rather than a continuous
# swath" — subdividing keeps the faceted look, and gets WORSE with more
# (thinner/sliver) polygons because ``matplotlib``'s ``mplot3d`` does only
# an approximate per-polygon depth-sort. The fix is a genuinely different
# RENDERING technology, not more subdivision: bake the accumulated
# coverage into a single equirectangular RASTER TEXTURE and map it onto
# the whole body mesh in ONE draw call via a real texture-capable,
# true-z-buffered renderer (PyVista/VTK) — no per-polygon facecolors, no
# per-polygon depth-sort, so no sliver artifact is even possible.
#
# The already-validated sub-facet ACCUMULATION math (``_adaptive_subdivide_mesh``,
# ``_pixel_barycentric``, the per-facet-per-frame percentile-trimmed (v, w)
# AABB-overlap fill) is REUSED UNCHANGED — only the rendering back end
# changes. This matters because that accumulation logic was already
# debugged (the center-containment -> AABB-overlap fix) and cross-checked
# against ground truth; there was no reason to re-derive it.
#
# A real, load-bearing discovery made while building this (see session
# notes for the full write-up): the camera FITS' per-pixel LATITUDE/
# LONGITUDE fields are NOT finely continuous within a single facet in a
# single frame — empirically, ~92-98% of (facet, frame) pairs across all
# 4 Justitia proxops phases have <= 8 DISTINCT (lat, lon) values
# REGARDLESS of how many thousands of pixels belong to that facet (e.g. a
# 232,175-pixel facet-frame reduced to exactly 4 distinct (lat, lon)
# pairs). This contradicts the round-1 swath-fix memory's claim of "real
# continuous spatial variation" (that finding likely came from comparing
# lat/lon ACROSS frames for the same facet, where genuine parallax as the
# spacecraft moves does shift the value slightly frame-to-frame, not
# WITHIN one frame). Practically: there is no genuine finer-than-sub-facet
# position information in a single frame to exploit, so this renderer does
# NOT attempt to bin raw per-pixel LAT/LON directly into texels (that
# would look SPARSE/speckled, not continuous, exactly like the previously
# REJECTED nearest-centroid prototype from round 1). Instead, each
# ALREADY-COVERED sub-triangle (from the existing, validated AABB-overlap
# accumulation) is rasterized into the texture via ITS OWN precomputed
# axis-aligned texel bounding box (:func:`_subtri_texel_bounds`) — i.e.
# the texture raster is the SAME sub-facet-resolution estimate as before,
# just displayed as a continuous baked texture instead of individual
# ``Poly3DCollection`` polygons. The texture resolution chosen below
# (``texture_width=4096`` -> ~38.9 m/texel at the equator) is deliberately
# finer than the ~150-350 m sub-triangle scale (for a smooth, antialiased
# look) but does NOT claim to resolve the true nominal camera GSD
# (~2-42 m across the 4 phases) — that finer number is not actually
# achievable from this data at single-frame resolution; see the module
# note and session notes for the full honest accounting.


def _build_equirect_uv_mesh(verts, faces):
    """Build an UNWELDED (each triangle owns its own 3 corner points, so
    texture coordinates can differ vertex-by-vertex across a shared
    physical edge — required for the antimeridian-seam fix below) whole-body
    equirectangular-UV mesh, as plain numpy arrays ready to hand to
    ``pyvista.PolyData``.

    UV convention (verified against a real ``pv.Texture`` with a labeled-
    quadrant test image before use, NOT assumed): ``U = (lon_unwrapped +
    180) / 360`` (0 at the -180 meridian, increasing eastward), ``V = (lat
    + 90) / 180`` (0 at the south pole, 1 at the north pole). A texture
    IMAGE array paired with this V convention must be built with ``row =
    (90 - lat) / 180 * H`` (row 0 = north pole, ordinary image-top
    convention) — see :func:`_subtri_texel_bounds`, which uses the same
    convention for the texel raster itself.

    Per-face longitude UNWRAP is the actual antimeridian-seam fix: each
    face's 3 corner longitudes are corrected by whole multiples of 360 deg
    relative to its OWN first corner (minimizing the corner spread) before
    computing U. Without this, any face whose true footprint straddles
    +-180 deg would have its U-interpolation take the "long way around"
    the texture map (a highly visible smeared band). ~6% of this 1144-face
    Justitia mesh's facets need the correction (measured). The corrected U
    can end up slightly outside ``[0, 1)`` (by at most one period) — the
    paired ``pv.Texture`` MUST have ``.wrap = 1`` (REPEAT) set so those
    samples wrap correctly instead of clamping to the texture edge.

    Returns ``(pts_unwelded (3*n_faces, 3), faces_vtk, uv_unwelded
    (3*n_faces, 2))`` — ``faces_vtk`` is the flat VTK cell-connectivity
    array PyVista's ``PolyData(points, faces)`` constructor expects.
    """
    A_i, B_i, C_i = faces[:, 0], faces[:, 1], faces[:, 2]
    r = np.linalg.norm(verts, axis=1)
    lon_v = np.degrees(np.arctan2(verts[:, 1], verts[:, 0]))
    lat_v = np.degrees(np.arcsin(np.clip(verts[:, 2] / r, -1.0, 1.0)))

    lon_tri = np.stack([lon_v[A_i], lon_v[B_i], lon_v[C_i]], axis=1)
    ref = lon_tri[:, 0:1]
    correction = np.round((lon_tri - ref) / 360.0) * 360.0
    lon_unwrapped = lon_tri - correction
    lat_tri = np.stack([lat_v[A_i], lat_v[B_i], lat_v[C_i]], axis=1)

    U = (lon_unwrapped + 180.0) / 360.0
    V = (lat_tri + 90.0) / 180.0

    pts_unwelded = verts[faces].reshape(-1, 3)
    uv_unwelded = np.stack([U, V], axis=2).reshape(-1, 2)

    n_pts = pts_unwelded.shape[0]
    tri_conn = np.arange(n_pts).reshape(-1, 3)
    faces_vtk = np.hstack(
        [np.full((len(tri_conn), 1), 3), tri_conn]
    ).astype(np.int64).ravel()
    return pts_unwelded, faces_vtk, uv_unwelded


def _subtri_texel_bounds(sub_tri_xyz, W, H):
    """Per-sub-triangle axis-aligned texel bounding box ``(row_lo, row_hi,
    col_lo, col_hi)`` in an ``(H, W)`` equirectangular texture, computed
    once from each sub-triangle's own 3 corner positions (from
    :func:`_adaptive_subdivide_mesh`), using the SAME longitude-unwrap
    convention as :func:`_build_equirect_uv_mesh` (unwrapped relative to
    the sub-triangle's own first corner — always safe: a sub-triangle is
    always far smaller than 180 deg of longitude).

    ``col_lo``/``col_hi`` may be negative or >= ``W`` (by at most one
    period) for the rare sub-triangle whose own 3 corners straddle the
    antimeridian; callers must fill with wraparound (``% W``), matching
    the texture's ``wrap = REPEAT`` setting. This is a coarse
    AXIS-ALIGNED bounding box, not an exact triangle rasterization —
    consistent with (and exactly as approximate as) the existing sub-facet
    accumulation's own (v, w) AABB-overlap logic; a real, small,
    over-covering tradeoff, reported honestly (see the module docstring
    above and the session notes).
    """
    r = np.linalg.norm(sub_tri_xyz, axis=2)
    lon = np.degrees(np.arctan2(sub_tri_xyz[:, :, 1], sub_tri_xyz[:, :, 0]))
    lat = np.degrees(np.arcsin(np.clip(sub_tri_xyz[:, :, 2] / r, -1.0, 1.0)))
    ref = lon[:, 0:1]
    correction = np.round((lon - ref) / 360.0) * 360.0
    lon_unwrapped = lon - correction

    U = (lon_unwrapped + 180.0) / 360.0
    V = (lat + 90.0) / 180.0
    col_f = U * W
    row_f = (1.0 - V) * H

    col_lo = np.floor(col_f.min(axis=1)).astype(np.int64)
    col_hi = np.ceil(col_f.max(axis=1)).astype(np.int64)
    row_lo = np.clip(np.floor(row_f.min(axis=1)).astype(np.int64), 0, H - 1)
    row_hi = np.clip(np.ceil(row_f.max(axis=1)).astype(np.int64), 0, H - 1)
    return row_lo, row_hi, col_lo, col_hi


def _fill_texel_image(H, W, subtri_bounds, touched_idx, touched_vals):
    """Build a fresh ``(H, W)`` float32 texel image (NaN = never imaged)
    by filling each of ``touched_idx``'s precomputed axis-aligned texel
    bounding box (:func:`_subtri_texel_bounds`) with its paired value from
    ``touched_vals``. Column ranges that fall outside ``[0, W)``
    (antimeridian-straddling sub-triangles) wrap around, matching the
    ``texture.wrap = REPEAT`` setting used when this image is uploaded as
    a texture."""
    row_lo, row_hi, col_lo, col_hi = subtri_bounds
    img = np.full((H, W), np.nan, dtype=np.float32)
    for si, val in zip(touched_idx, touched_vals):
        r0, r1 = int(row_lo[si]), int(row_hi[si])
        c0, c1 = int(col_lo[si]), int(col_hi[si])
        if c1 - c0 >= W:
            img[r0:r1 + 1, :] = val
            continue
        cc0, cc1 = c0 % W, c1 % W
        if cc0 <= cc1:
            img[r0:r1 + 1, cc0:cc1 + 1] = val
        else:
            img[r0:r1 + 1, cc0:] = val
            img[r0:r1 + 1, :cc1 + 1] = val
    return img


def render_coverage_accumulation_animation_facet_legacy(
    target,
    camera_subdir,
    thermal_subdir,
    naif_frame,
    metakernel,
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0),
    vmin=None,
    vmax=None,
    elev=20,
    azim=25,
    fps=8,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
    thermal_ext=None,
    caption=None,
):
    """LEGACY (pre-2026-07-05) per-FACE_IDX coverage-accumulation animation
    — preserved verbatim (renamed from ``render_coverage_accumulation_animation``)
    for reference/regression only. A whole facet "pops" fully covered the
    instant any part of it is imaged (blocky at proxops range where the
    pixel/FOV footprint is much finer than a facet) — superseded by the
    sub-facet swath version above for all current call sites. See the
    module note above :func:`_adaptive_subdivide_mesh` for the rework
    rationale.

    Build the coverage-accumulation animation described above.

    Parametrized (target/camera_subdir/thermal_subdir/naif_frame/metakernel)
    so it is reusable for any EMA body's proxops-style camera sequence, not
    just Justitia. Reads:
      * ``camera_dir(target)/camera_subdir/frame_*.fits`` — the rendered
        camera sequence (already time-ordered by filename), for per-frame
        FACE_IDX + <thermal_ext> + UTC_TIME.
      * ``thermal_dir(target)/thermal_subdir/mesh.ply`` — body-fixed vertex
        positions + triangle connectivity (the SAME mesh the camera frames
        were rendered against).
      * SPICE ``metakernel`` (body ephemeris + spin PCK) + ``naif_frame``
        (e.g. ``"IAU_JUSTITIA_269"``) to compute the per-frame de-spin
        rotation.

    `thermal_ext`: which FITS extension to accumulate/paint. `None` (default)
    auto-picks ``TEMP_K_SMOOTH`` (Gouraud-interpolated) if present in the
    frames, else falls back to the flat ``TEMP_K`` — do-no-harm for camera
    sequences rendered on a pre-smooth-thermal binary.

    `caption`, if given, is drawn as a small persistent honesty/methods note
    on every frame (e.g. `HONESTY_CAPTION` when using TEMP_K_SMOOTH).
    """
    import spiceypy as sp

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")

    cdir = camera_dir(target) / camera_subdir
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = camera_dir(target) / camera_subdir / f"{target}_coverage_accum_pub.mp4"
    if title is None:
        title = f"{body_label} — accumulating IR coverage (most-recent-imaged)"

    fits_files = sorted(glob.glob(str(cdir / "frame_*.fits")))
    if not fits_files:
        raise RuntimeError(f"no frame_*.fits under {cdir}")

    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    verts, faces = read_mesh_verts_faces(tdir / "mesh.ply")
    n_faces = len(faces)

    # --- harvest per-frame FACE_IDX / <thermal_ext> / UTC, and accumulate ---
    utcs = []
    coverage_T = np.full(n_faces, np.nan)
    accum_snapshots = []  # per-output-frame copy of the coverage_T state
    all_valid_T = []
    for fp in fits_files:
        with fits.open(fp) as hdul:
            face_idx = np.asarray(hdul["FACE_IDX"].data)
            temp_k = np.asarray(hdul[thermal_ext].data, dtype=float)
            utc = hdul[0].header["UTC_TIME"]
        utcs.append(utc)
        valid = face_idx >= 0
        ids = face_idx[valid].astype(int)
        vals = temp_k[valid]
        all_valid_T.append(vals)
        if ids.size:
            sums = np.bincount(ids, weights=vals, minlength=n_faces)
            counts = np.bincount(ids, minlength=n_faces)
            touched = counts > 0
            coverage_T[touched] = sums[touched] / counts[touched]
        accum_snapshots.append(coverage_T.copy())

    n_touched_final = int(np.sum(~np.isnan(accum_snapshots[-1])))

    if vmin is None or vmax is None:
        allT = np.concatenate(all_valid_T)
        if vmin is None:
            vmin = 5.0 * np.floor(allT.min() / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(allT.max() / 5.0)

    # --- per-frame de-spin rotation (inertial-anchored view) ---
    sp.furnsh(str(metakernel))
    et_list = [sp.str2et(u) for u in utcs]
    et0 = et_list[0]
    R0 = sp.pxform("J2000", naif_frame, et0)  # J2000 -> body-fixed-at-t0
    rotated_tris = []
    for et in et_list:
        Rt = sp.pxform(naif_frame, "J2000", et)  # body-fixed-at-t -> J2000
        R = R0 @ Rt  # body-fixed-at-t -> (anchored) body-fixed-at-t0 frame
        rv = verts @ R.T
        rotated_tris.append(rv[faces])
    sp.unload(str(metakernel))

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_coverage_frames_"))
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
        ax = fig.add_axes([-0.08, 0.0, 0.95, 0.92], projection="3d")
        ax.set_facecolor("black")

        def facecolors_for(snapshot):
            colors = np.tile(np.array(neutral_rgba), (n_faces, 1))
            touched = ~np.isnan(snapshot)
            colors[touched] = cmap_obj(norm(snapshot[touched]))
            return colors

        pc = Poly3DCollection(
            rotated_tris[0], facecolors=facecolors_for(accum_snapshots[0]),
            edgecolor="none", antialiased=True,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, color="white", fontsize=14)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        fig.suptitle(title, color="white", fontsize=15, y=0.97)
        footer_fontsize = 11.0 if caption else 13
        footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=footer_fontsize,
                               ha="left", va="bottom", family="monospace")
        # Honesty caption appended as extra line(s) UNDER the per-frame
        # coverage label (bottom-left), not a separate fixed top text block —
        # avoids any collision with the title regardless of caption length.

        for i, (tri, snap, utc) in enumerate(zip(rotated_tris, accum_snapshots, utcs)):
            pc.set_verts(tri)
            pc.set_facecolor(facecolors_for(snap))
            n_cov = int(np.sum(~np.isnan(snap)))
            label = (f"{_format_utc_clean(utc)}   "
                     f"coverage: {n_cov}/{n_faces} faces ({100.0*n_cov/n_faces:.0f}%)")
            if caption:
                label += f"\n{caption}"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_mp4, n_touched_final, n_faces, thermal_ext
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


def render_coverage_accumulation_animation_subfacet_polygon_legacy(
    target,
    camera_subdir,
    thermal_subdir,
    naif_frame,
    metakernel,
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0),
    vmin=None,
    vmax=None,
    elev=20,
    azim=25,
    fps=8,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
    thermal_ext=None,
    caption=None,
    target_km=0.3,
    n_min=2,
    n_max=16,
    bbox_percentile=(2.0, 98.0),
    wild_outlier_thresh=2.0,
    archive_existing=True,
):
    """SUB-FACET SWATH coverage-accumulation animation, POLYGON-RENDERED
    (2026-07-05 rework of the per-``FACE_IDX`` legacy version above).
    SUPERSEDED 2026-07-05 (round 2) — Paul Hayne reviewed this version's
    output and found it still reads as "scattered sliver-like triangles
    filling in, rather than a continuous swath": subdividing into more
    polygons cannot fix a fundamentally polygon-based renderer, and
    ``matplotlib``'s ``mplot3d`` does only an approximate per-polygon
    depth-sort that gets visibly worse with more (and thinner/sliver)
    triangles. Preserved verbatim here (renamed from
    ``render_coverage_accumulation_animation``) for reference/regression
    only — the live name now points to
    :func:`render_coverage_accumulation_animation_texture`'s texture-mapped
    VTK/PyVista renderer (true z-buffer, no polygon depth-sort, and the
    swath is a continuous raster rather than discrete facecolored
    triangles). See the module note above :func:`_adaptive_subdivide_mesh`
    for the sub-facet accumulation motivation (unchanged, reused as-is by
    the new renderer) and the session notes for the side-by-side
    before/after comparison that motivated this second rework.

    Same parametrization and same inertial-fixed-vantage / honesty-caption
    styling as the legacy version, but accumulates on an adaptively
    subdivided sub-triangle mesh (:func:`_adaptive_subdivide_mesh`) using
    each frame's own per-pixel ``LATITUDE``/``LONGITUDE`` (exact
    ray-plane barycentric position within the pixel's own known
    ``FACE_IDX`` facet, :func:`_pixel_barycentric`) rather than the whole
    facet. Per facet-per-frame, sub-triangles whose own barycentric center
    falls within that frame's own (percentile-trimmed, outlier-guarded)
    pixel (v, w) bounding box are marked covered — dense/contiguous, and
    empirically confirmed still genuinely PARTIAL per facet (not
    whole-triangle fill; see the returned ``n_facets_partial`` et al. and
    the session notes' zoomed before/after stills).

    Returns a dict (NOT the legacy 4-tuple — there is a genuinely
    different, additional coverage metric to report now):
      ``out_mp4``, ``thermal_ext``,
      ``n_sub_touched``/``n_sub_total`` (sub-triangle counts),
      ``area_pct`` (AREA-weighted coverage, the honest swath-area metric —
      will generally be SMALLER than the legacy facet-count fraction,
      since the legacy metric marks an entire multi-km facet "covered" the
      instant even one pixel touches it),
      ``facet_pct_legacy`` (the old-style "touched-facet-count / n_faces"
      fraction, recomputed from this SAME run for direct comparison),
      ``n_facets_partial``/``n_facets_full``/``n_facets_empty``/``n_facets_total``
      (the partial-vs-whole-triangle breakdown — the direct evidence this
      is genuinely sub-facet), and
      ``n_wild_pixels_dropped``/``n_pixels_total`` (data-quality
      transparency for the small fraction of grazing/near-limb pixels
      whose ray-from-center barycentric reconstruction was ill-conditioned
      and excluded).
    """
    import spiceypy as sp

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")

    cdir = camera_dir(target) / camera_subdir
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = camera_dir(target) / camera_subdir / f"{target}_coverage_accum_pub.mp4"
    if title is None:
        title = f"{body_label} — accumulating IR coverage (most-recent-imaged)"

    fits_files = sorted(glob.glob(str(cdir / "frame_*.fits")))
    if not fits_files:
        raise RuntimeError(f"no frame_*.fits under {cdir}")

    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    verts, faces = read_mesh_verts_faces(tdir / "mesh.ply")
    n_faces = len(faces)

    sub_tri_xyz, sub_face_ids, sub_vw, sub_vw_bounds, local_offset, n_facet = _adaptive_subdivide_mesh(
        verts, faces, target_km=target_km, n_min=n_min, n_max=n_max)
    n_sub = sub_tri_xyz.shape[0]

    # --- harvest per-frame FACE_IDX / LATITUDE / LONGITUDE / <thermal_ext>
    # / UTC, and accumulate at sub-facet resolution via per-facet bbox-fill.
    utcs = []
    coverage_T = np.full(n_sub, np.nan)
    accum_snapshots = []
    all_valid_T = []
    n_wild_total = 0
    n_px_total = 0
    plo, phi = bbox_percentile
    for fp in fits_files:
        with fits.open(fp) as hdul:
            face_idx = np.asarray(hdul["FACE_IDX"].data)
            lat = np.asarray(hdul["LATITUDE"].data, dtype=float)
            lon = np.asarray(hdul["LONGITUDE"].data, dtype=float)
            temp_k = np.asarray(hdul[thermal_ext].data, dtype=float)
            utc = hdul[0].header["UTC_TIME"]
        utcs.append(utc)
        valid = face_idx >= 0
        ids = face_idx[valid].astype(int)
        vals = temp_k[valid]
        all_valid_T.append(vals)
        n_px_total += ids.size

        if ids.size:
            plat = np.radians(lat[valid])
            plon = np.radians(lon[valid])
            u, v, w = _pixel_barycentric(verts, faces, ids, plat, plon)
            wild = (np.abs(v) > wild_outlier_thresh) | (np.abs(w) > wild_outlier_thresh)
            n_wild_total += int(wild.sum())
            keep = ~wild
            ids_k = ids[keep]
            v_k = np.clip(v[keep], -0.05, 1.05)
            w_k = np.clip(w[keep], -0.05, 1.05)
            vals_k = vals[keep]

            for f in np.unique(ids_k):
                m = ids_k == f
                if not m.any():
                    continue
                vmin_f, vmax_f = np.percentile(v_k[m], [plo, phi])
                wmin_f, wmax_f = np.percentile(w_k[m], [plo, phi])
                lo, hi = local_offset[f], local_offset[f + 1]
                # AABB-OVERLAP against each sub-triangle's own (v, w) extent
                # — NOT "is the sub-triangle's centroid inside the pixel
                # box" (that test silently drops the large majority of
                # genuine imaging events whenever a narrow/off-center
                # imaged patch doesn't happen to contain any sub-triangle's
                # exact center; see the docstring note on `sub_vw_bounds`
                # in `_adaptive_subdivide_mesh` for the empirical numbers).
                sub_vlo = sub_vw_bounds[lo:hi, 0]
                sub_vhi = sub_vw_bounds[lo:hi, 1]
                sub_wlo = sub_vw_bounds[lo:hi, 2]
                sub_whi = sub_vw_bounds[lo:hi, 3]
                overlap = ((sub_vlo <= vmax_f) & (sub_vhi >= vmin_f)
                           & (sub_wlo <= wmax_f) & (sub_whi >= wmin_f))
                if overlap.any():
                    coverage_T[lo:hi][overlap] = vals_k[m].mean()
        accum_snapshots.append(coverage_T.copy())

    n_sub_touched = int(np.sum(~np.isnan(accum_snapshots[-1])))

    if vmin is None or vmax is None:
        allT = np.concatenate(all_valid_T)
        if vmin is None:
            vmin = 5.0 * np.floor(allT.min() / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(allT.max() / 5.0)

    # --- coverage-quality bookkeeping (computed from the FINAL snapshot) ---
    tri = verts[faces]
    Av, Bv, Cv = tri[:, 0], tri[:, 1], tri[:, 2]
    facet_area = 0.5 * np.linalg.norm(np.cross(Bv - Av, Cv - Av), axis=1)
    total_area = facet_area.sum()
    final_touched = ~np.isnan(accum_snapshots[-1])
    sub_area = facet_area[sub_face_ids] / (n_facet[sub_face_ids] ** 2)
    area_pct = 100.0 * sub_area[final_touched].sum() / total_area

    n_facets_partial = n_facets_full = n_facets_empty = 0
    for f in range(n_faces):
        lo, hi = local_offset[f], local_offset[f + 1]
        nt = int(np.sum(final_touched[lo:hi]))
        if nt == 0:
            n_facets_empty += 1
        elif nt == (hi - lo):
            n_facets_full += 1
        else:
            n_facets_partial += 1
    facet_pct_legacy = 100.0 * (n_facets_partial + n_facets_full) / n_faces

    # --- per-frame de-spin rotation (inertial-anchored view) — applied
    # directly to the flat sub-triangle vertex array (no shared index
    # buffer needed; a rigid rotation is well-defined per-point). ---
    sp.furnsh(str(metakernel))
    et_list = [sp.str2et(u) for u in utcs]
    et0 = et_list[0]
    R0 = sp.pxform("J2000", naif_frame, et0)
    rotated_subs = []
    flat = sub_tri_xyz.reshape(-1, 3)
    for et in et_list:
        Rt = sp.pxform(naif_frame, "J2000", et)
        R = R0 @ Rt
        rotated_subs.append((flat @ R.T).reshape(sub_tri_xyz.shape))
    sp.unload(str(metakernel))

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    if archive_existing:
        _archived = _archive_existing(out_mp4)
    else:
        _archived = None

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_coverage_swath_frames_"))
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
        ax = fig.add_axes([-0.08, 0.0, 0.95, 0.92], projection="3d")
        ax.set_facecolor("black")

        def facecolors_for(snapshot):
            colors = np.tile(np.array(neutral_rgba), (n_sub, 1))
            touched = ~np.isnan(snapshot)
            colors[touched] = cmap_obj(norm(snapshot[touched]))
            return colors

        pc = Poly3DCollection(
            rotated_subs[0], facecolors=facecolors_for(accum_snapshots[0]),
            edgecolor="none", antialiased=True,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, color="white", fontsize=14)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        fig.suptitle(title, color="white", fontsize=15, y=0.97)
        footer_fontsize = 11.0 if caption else 13
        footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=footer_fontsize,
                               ha="left", va="bottom", family="monospace")

        for i, (tri_r, snap, utc) in enumerate(zip(rotated_subs, accum_snapshots, utcs)):
            pc.set_verts(tri_r)
            pc.set_facecolor(facecolors_for(snap))
            area_pct_i = 100.0 * sub_area[~np.isnan(snap)].sum() / total_area
            label = (f"{_format_utc_clean(utc)}   "
                     f"coverage: {area_pct_i:.1f}% of surface area (sub-facet)")
            if caption:
                label += f"\n{caption}"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return {
            "out_mp4": out_mp4,
            "archived_legacy_mp4": _archived,
            "thermal_ext": thermal_ext,
            "n_sub_touched": n_sub_touched,
            "n_sub_total": n_sub,
            "area_pct": area_pct,
            "facet_pct_legacy": facet_pct_legacy,
            "n_facets_partial": n_facets_partial,
            "n_facets_full": n_facets_full,
            "n_facets_empty": n_facets_empty,
            "n_facets_total": n_faces,
            "n_wild_pixels_dropped": n_wild_total,
            "n_pixels_total": n_px_total,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


def render_coverage_accumulation_animation_texture_subfacet_legacy(
    target,
    camera_subdir,
    thermal_subdir,
    naif_frame,
    metakernel,
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0),
    vmin=None,
    vmax=None,
    elev=20,
    azim=25,
    fps=8,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
    thermal_ext=None,
    caption=None,
    target_km=0.3,
    n_min=2,
    n_max=16,
    bbox_percentile=(2.0, 98.0),
    wild_outlier_thresh=2.0,
    archive_existing=True,
    texture_width=4096,
):
    """TEXTURE-MAPPED coverage-accumulation animation (2026-07-05, round 2)
    — PRESERVED VERBATIM for reference/regression only (renamed from
    ``render_coverage_accumulation_animation_texture``; superseded 2026-07-05
    round 3 by :func:`render_coverage_accumulation_animation_texture_truegsd`,
    which bins REAL per-pixel ``SURF_X_KM/SURF_Y_KM/SURF_Z_KM`` hit points
    directly into texels instead of this function's sub-triangle-AABB
    approximation — see that function's docstring for why). Replaces the
    ``Poly3DCollection``/``mplot3d`` sub-facet-swath renderer (now
    ``render_coverage_accumulation_animation_subfacet_polygon_legacy``)
    with a true texture-mapped, true-z-buffered PyVista/VTK render. See the
    module note above :func:`_build_equirect_uv_mesh` for the full
    motivation, the antimeridian-seam fix, and the real discovery (per-pixel
    LATITUDE/LONGITUDE is NOT finely continuous within one frame) that
    shaped this design.

    Same parametrization, same inertial-fixed vantage, same honesty-caption
    styling, and the SAME underlying sub-facet coverage estimate (reuses
    :func:`_adaptive_subdivide_mesh` / :func:`_pixel_barycentric`
    UNCHANGED) as the polygon-rendered predecessor — only the rendering
    back end changes: accumulated coverage is baked into a single
    equirectangular raster texture (:func:`_subtri_texel_bounds` /
    :func:`_fill_texel_image`) and mapped onto the WHOLE body mesh in one
    draw call per frame, instead of drawing ~62k individually
    depth-sorted, flat-facecolored polygons.

    ``texture_width`` (texels around the full 360 deg of longitude;
    height is always ``texture_width // 2``) defaults to 4096 (~38.9
    m/texel at this body's mean radius) — deliberately finer than the
    ~150-350 m sub-triangle accumulation scale (for a smooth/antialiased
    look) but NOT a claim of resolving the true nominal camera GSD; see
    the returned ``texel_km`` and the module docstring for the honest
    accounting.

    Returns the same dict keys as the polygon-rendered predecessor
    (``out_mp4``, ``archived_legacy_mp4``, ``thermal_ext``,
    ``n_sub_touched``, ``n_sub_total``, ``area_pct``, ``facet_pct_legacy``,
    ``n_facets_partial/full/empty/total``, ``n_wild_pixels_dropped``,
    ``n_pixels_total``) PLUS new texture-specific diagnostics:
    ``renderer`` (``"pyvista_texture"``, for provenance), ``texture_width``,
    ``texture_height``, ``texel_km`` (approximate texel edge length at the
    mean radius), ``n_texels_touched``, ``n_texels_total``.
    """
    import spiceypy as sp

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")
    if pv is None:
        raise RuntimeError("pyvista is required (pip install pyvista)")

    cdir = camera_dir(target) / camera_subdir
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = camera_dir(target) / camera_subdir / f"{target}_coverage_accum_pub.mp4"
    if title is None:
        title = f"{body_label} — accumulating IR coverage (most-recent-imaged)"

    fits_files = sorted(glob.glob(str(cdir / "frame_*.fits")))
    if not fits_files:
        raise RuntimeError(f"no frame_*.fits under {cdir}")

    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    verts, faces = read_mesh_verts_faces(tdir / "mesh.ply")
    n_faces = len(faces)

    sub_tri_xyz, sub_face_ids, sub_vw, sub_vw_bounds, local_offset, n_facet = _adaptive_subdivide_mesh(
        verts, faces, target_km=target_km, n_min=n_min, n_max=n_max)
    n_sub = sub_tri_xyz.shape[0]

    W = int(texture_width)
    H = W // 2
    subtri_bounds = _subtri_texel_bounds(sub_tri_xyz, W, H)
    circumference_km = 2.0 * np.pi * float(np.mean(np.linalg.norm(verts, axis=1)))
    texel_km = circumference_km / W

    body_pts_unwelded, body_faces_vtk, body_uv = _build_equirect_uv_mesh(verts, faces)

    # --- harvest per-frame FACE_IDX / LATITUDE / LONGITUDE / <thermal_ext>
    # / UTC, and accumulate at sub-facet resolution via per-facet bbox-fill
    # — IDENTICAL to the polygon-rendered predecessor (unchanged, reused). ---
    utcs = []
    coverage_T = np.full(n_sub, np.nan)
    accum_snapshots = []  # small (n_sub,)-sized snapshots — cheap to keep all of
    all_valid_T = []
    n_wild_total = 0
    n_px_total = 0
    plo, phi = bbox_percentile
    for fp in fits_files:
        with fits.open(fp) as hdul:
            face_idx = np.asarray(hdul["FACE_IDX"].data)
            lat = np.asarray(hdul["LATITUDE"].data, dtype=float)
            lon = np.asarray(hdul["LONGITUDE"].data, dtype=float)
            temp_k = np.asarray(hdul[thermal_ext].data, dtype=float)
            utc = hdul[0].header["UTC_TIME"]
        utcs.append(utc)
        valid = face_idx >= 0
        ids = face_idx[valid].astype(int)
        vals = temp_k[valid]
        all_valid_T.append(vals)
        n_px_total += ids.size

        if ids.size:
            plat = np.radians(lat[valid])
            plon = np.radians(lon[valid])
            u, v, w = _pixel_barycentric(verts, faces, ids, plat, plon)
            wild = (np.abs(v) > wild_outlier_thresh) | (np.abs(w) > wild_outlier_thresh)
            n_wild_total += int(wild.sum())
            keep = ~wild
            ids_k = ids[keep]
            v_k = np.clip(v[keep], -0.05, 1.05)
            w_k = np.clip(w[keep], -0.05, 1.05)
            vals_k = vals[keep]

            for f in np.unique(ids_k):
                m = ids_k == f
                if not m.any():
                    continue
                vmin_f, vmax_f = np.percentile(v_k[m], [plo, phi])
                wmin_f, wmax_f = np.percentile(w_k[m], [plo, phi])
                lo, hi = local_offset[f], local_offset[f + 1]
                sub_vlo = sub_vw_bounds[lo:hi, 0]
                sub_vhi = sub_vw_bounds[lo:hi, 1]
                sub_wlo = sub_vw_bounds[lo:hi, 2]
                sub_whi = sub_vw_bounds[lo:hi, 3]
                overlap = ((sub_vlo <= vmax_f) & (sub_vhi >= vmin_f)
                           & (sub_wlo <= wmax_f) & (sub_whi >= wmin_f))
                if overlap.any():
                    coverage_T[lo:hi][overlap] = vals_k[m].mean()
        accum_snapshots.append(coverage_T.copy())

    n_sub_touched = int(np.sum(~np.isnan(accum_snapshots[-1])))

    if vmin is None or vmax is None:
        allT = np.concatenate(all_valid_T)
        if vmin is None:
            vmin = 5.0 * np.floor(allT.min() / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(allT.max() / 5.0)

    # --- coverage-quality bookkeeping (identical to the predecessor) ---
    tri = verts[faces]
    Av, Bv, Cv = tri[:, 0], tri[:, 1], tri[:, 2]
    facet_area = 0.5 * np.linalg.norm(np.cross(Bv - Av, Cv - Av), axis=1)
    total_area = facet_area.sum()
    final_touched = ~np.isnan(accum_snapshots[-1])
    sub_area = facet_area[sub_face_ids] / (n_facet[sub_face_ids] ** 2)
    area_pct = 100.0 * sub_area[final_touched].sum() / total_area

    n_facets_partial = n_facets_full = n_facets_empty = 0
    for f in range(n_faces):
        lo, hi = local_offset[f], local_offset[f + 1]
        nt = int(np.sum(final_touched[lo:hi]))
        if nt == 0:
            n_facets_empty += 1
        elif nt == (hi - lo):
            n_facets_full += 1
        else:
            n_facets_partial += 1
    facet_pct_legacy = 100.0 * (n_facets_partial + n_facets_full) / n_faces

    final_texel_img = _fill_texel_image(
        H, W, subtri_bounds, np.where(final_touched)[0], accum_snapshots[-1][final_touched])
    n_texels_touched = int(np.sum(~np.isnan(final_texel_img)))

    # --- per-frame de-spin rotation (inertial-anchored view), applied
    # directly to the flat unwelded body point array. ---
    sp.furnsh(str(metakernel))
    et_list = [sp.str2et(u) for u in utcs]
    et0 = et_list[0]
    R0 = sp.pxform("J2000", naif_frame, et0)
    rotated_body_pts = []
    for et in et_list:
        Rt = sp.pxform(naif_frame, "J2000", et)
        R = R0 @ Rt
        rotated_body_pts.append(body_pts_unwelded @ R.T)
    sp.unload(str(metakernel))

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    neutral = np.array(neutral_rgba)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    if archive_existing:
        _archived = _archive_existing(out_mp4)
    else:
        _archived = None

    # --- PyVista off-screen plotter: one static texture-mapped body mesh,
    # re-textured + re-posed (de-spin rotation) every frame. NOTE (real,
    # load-bearing quirk found while building this): PyVista/VTK's
    # off-screen ``Plotter.screenshot()`` does NOT itself force a fresh
    # render pass after `camera_position`/`actor.texture`/`mesh.points` are
    # reassigned — verified directly (two screenshots taken after two
    # different camera positions came back byte-IDENTICAL without an
    # explicit `.render()` in between). Every per-frame update below is
    # therefore followed by an explicit `pl.render()` before
    # `pl.screenshot()` — omitting this would silently ship a frozen
    # first-frame video.
    d = np.array([
        np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
        np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
        np.sin(np.radians(elev)),
    ])
    campos = center + d * half * 6

    pv_win_w = max(64, int(round(0.83 * target_width_px)))
    pv_win_h = max(64, int(round(0.90 * target_height_px)))

    body_mesh = pv.PolyData(rotated_body_pts[0].copy(), body_faces_vtk)
    body_mesh.active_texture_coordinates = body_uv

    pl = pv.Plotter(off_screen=True, window_size=[pv_win_w, pv_win_h])
    pl.set_background("black")
    actor = pl.add_mesh(body_mesh, texture=pv.Texture(np.zeros((2, 2, 3), dtype=np.uint8)),
                         lighting=False)
    pl.camera_position = [tuple(campos), tuple(center), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = half

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 0.83, 0.92])
    ax.set_facecolor("black")
    ax.set_axis_off()
    im = ax.imshow(np.zeros((pv_win_h, pv_win_w, 3), dtype=np.uint8), aspect="auto")

    cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, color="white", fontsize=14)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(title, color="white", fontsize=15, y=0.97)
    footer_fontsize = 11.0 if caption else 13
    footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=footer_fontsize,
                           ha="left", va="bottom", family="monospace")

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_coverage_texture_frames_"))
    try:
        running_snapshot_touched = np.zeros(n_sub, dtype=bool)
        for i, (snap, utc, pts_i) in enumerate(zip(accum_snapshots, utcs, rotated_body_pts)):
            touched_i = ~np.isnan(snap)
            touched_idx = np.where(touched_i)[0]
            texel_img = _fill_texel_image(H, W, subtri_bounds, touched_idx, snap[touched_idx])
            rgba = np.tile(neutral, (H, W, 1))
            tmask = ~np.isnan(texel_img)
            rgba[tmask] = cmap_obj(norm(texel_img[tmask]))
            img_u8 = (rgba[:, :, :3] * 255).astype(np.uint8)
            tex = pv.Texture(img_u8)
            tex.wrap = 1  # REPEAT — required for antimeridian-unwrapped UV/texel coords
            actor.texture = tex
            body_mesh.points = pts_i
            pl.render()
            frame_rgb = pl.screenshot(return_img=True)

            area_pct_i = 100.0 * sub_area[touched_i].sum() / total_area
            im.set_data(frame_rgb)
            label = (f"{_format_utc_clean(utc)}   "
                     f"coverage: {area_pct_i:.1f}% of surface area (sub-facet)")
            if caption:
                label += f"\n{caption}"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)
        pl.close()

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return {
            "out_mp4": out_mp4,
            "archived_legacy_mp4": _archived,
            "thermal_ext": thermal_ext,
            "renderer": "pyvista_texture",
            "texture_width": W,
            "texture_height": H,
            "texel_km": texel_km,
            "n_texels_touched": n_texels_touched,
            "n_texels_total": W * H,
            "n_sub_touched": n_sub_touched,
            "n_sub_total": n_sub,
            "area_pct": area_pct,
            "facet_pct_legacy": facet_pct_legacy,
            "n_facets_partial": n_facets_partial,
            "n_facets_full": n_facets_full,
            "n_facets_empty": n_facets_empty,
            "n_facets_total": n_faces,
            "n_wild_pixels_dropped": n_wild_total,
            "n_pixels_total": n_px_total,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


# --------------------------------------------------------------------------
# TRUE-PER-PIXEL coverage-accumulation render (2026-07-05, round 3)
# --------------------------------------------------------------------------
# ``ptm_camera_sim`` (binary md5 25866a03673bbf743f635be1b51fca13+) now
# writes an ADDITIVE per-pixel FITS extension trio ``SURF_X_KM``/
# ``SURF_Y_KM``/``SURF_Z_KM``: the EXACT body-fixed 3D point where that
# pixel's ray hit the real mesh surface, resolved at the camera's true GSD
# (confirmed empirically: a 54,201-pixel Justitia facet now yields 54,201
# DISTINCT (x, y, z) triples, ~110 m apart at High-orbit range — see session
# notes for the full measurement). This removes the root cause of round 2's
# residual approximation: the pre-existing FITS ``LATITUDE``/``LONGITUDE``
# extensions are NOT finely continuous within a frame (round 2's real
# finding, still true — confirmed unchanged, <=8 distinct values per
# facet-frame regardless of pixel count), but ``SURF_X_KM``/``Y``/``Z`` ARE,
# because they are the camera's own ray-mesh intersection point rather than
# a coarser per-facet-interpolated lat/lon field. Consequently, this
# renderer no longer needs ANY of round 1/2's sub-triangle machinery
# (``_adaptive_subdivide_mesh``/``_pixel_barycentric``/
# ``_subtri_texel_bounds``/``_fill_texel_image``, all still present above,
# UNCHANGED, for the legacy functions only) — every on-body pixel is binned
# directly into its own equirectangular texel via
# ``r = sqrt(x^2+y^2+z^2); lat = degrees(asin(z/r)); lon =
# degrees(atan2(y, x))`` (the exact formula specified for this task), most-
# recent-frame-wins. No ray-plane barycentric solve, no AABB-overlap
# approximation, no "wild outlier" pixel-dropping (all real failure modes
# of the earlier approximate methods) — the hit point IS the answer.
#
# Texture resolution is now chosen PER PHASE to track each phase's own
# measured true GSD (median real-space distance between horizontally
# adjacent on-body pixels, measured directly from ``SURF_*``, not the
# coarser angular pre-estimate from the smooth-thermal campaign — see
# :func:`make_justitia_proxops_phase_animations` / session notes for the
# measured numbers and the honest per-phase texel-vs-GSD accounting,
# including where the texel is deliberately coarser than the finest GSD
# encountered (Low orbit, and especially the Descent phase's ~2 m
# closest-approach minimum, which is NOT resolvable in a single fixed-
# resolution texture without an impractically large array).
#
# Memory/perf note: for the larger per-phase textures (up to 16384x8192),
# this renderer does NOT store one full-resolution snapshot per frame (that
# would be tens of GB across 66 frames). Instead it maintains ONE
# persistent ``(H, W, 3)`` uint8 RGBA raster + one ``(H, W)`` touched-mask,
# updated in place frame-by-frame (only the touched pixels' texels are
# rewritten each frame — bounded by ~pixels-per-frame, not texture size),
# and renders+screenshots immediately after each frame's update rather than
# in a second pass. This is a genuinely different (simpler, cheaper) loop
# structure than round 1/2's two-pass design, made possible because there
# is no longer any per-facet/per-sub-triangle bookkeeping to do first.
#
# --------------------------------------------------------------------------
# FOOTPRINT-FILL (2026-07-05, round 5, Fix 1, Paul-approved): each imaged
# pixel now fills its ACTUAL ground footprint (a ~GSD x GSD patch centered
# on the pixel's own SURF hit point), not just the single texel nearest that
# hit point. This closes two real, previously-flagged gaps: (a) the
# WITHIN-frame texel-vs-native-pixel-spacing speckle documented in round 3
# (~10.6% local hole fraction at swath boundaries, where pixel pitch and
# texel pitch are only marginally matched); and (b) the full-period
# product's round-4-flagged under-credit of the coarser High/Mid-orbit
# passes on the finer shared 16384-wide grid (a single ~38-40 m High-orbit
# pixel legitimately spans ~5 texels of that 9.73 m grid — round 4 measured
# "~4x" from first principles; direct per-pixel measurement here confirms
# mean ~4.6-5.0 texels-across, occasionally to ~15x at grazing emission).
# GSD itself uses the already-established task formula (GSD = |SURF-CAM| *
# IFOV, see the IR3S_* constants below). Foreshortening: at grazing
# emission the true projected footprint elongates along the local emission-
# angle gradient to ~GSD/cos(emission_angle); rather than resolve the exact
# anisotropic ellipse orientation (would need the local surface-tilt
# direction, not just its magnitude), this fills an ISOTROPIC footprint
# sized to that ELONGATED (long-axis) diameter in every direction — a
# conservative choice per the task's explicit "at least an emission-scaled
# one" allowance: it never UNDER-fills a grazing pixel, at the cost of a
# modest over-fill in the cross-elongation direction. Emission angle is
# clipped at ``FOOTPRINT_EMI_CLIP_DEG`` before the cos(.) division to bound
# near-limb blowup (a pixel at emi_ang -> 90 deg would otherwise imply an
# infinite footprint); pixels beyond the clip are honestly under-filled,
# not further fabricated. The resulting footprint half-width (in texels) is
# ALSO capped at ``FOOTPRINT_MAX_TEXELS_HALF`` as a practical, documented
# compute-cost ceiling — measured on real frames this caps <2% of pixels
# even on the most demanding case (High orbit on the fine full-period
# grid), so it does not materially change the delivered coverage numbers.
# No new camera binary is required: ``EMI_ANG`` is already an extension on
# the existing SURF-writing binary's (md5 25866a03673bbf743f635be1b51fca13)
# output.

FOOTPRINT_EMI_CLIP_DEG = 80.0   # emission-angle floor for the 1/cos(.) foreshortening scale
FOOTPRINT_MAX_TEXELS_HALF = 5   # practical cap on footprint-fill half-width, in texels


def _footprint_fill_radii(rng_km, emi_ang_deg, lat_deg, texel_km):
    """Per-pixel true-footprint half-width, in texel units, for the
    footprint-fill accumulation (round 5, Fix 1). Returns ``(rad_row,
    rad_col)`` int64 arrays (same shape as the inputs): ``rad_row`` is the
    half-width in the latitude (row) direction; ``rad_col`` is the
    half-width in the longitude (column) direction, additionally corrected
    for the equirectangular projection's cos(latitude) column-narrowing
    (a column-texel is physically ``texel_km * cos(lat)`` wide, not
    ``texel_km``, away from the equator). ``rng_km`` is the true per-pixel
    camera-to-surface range (``|SURF - CAM|``); ``emi_ang_deg`` is the
    pixel's own ``EMI_ANG`` FITS value. See the module note above for the
    foreshortening/clip/cap rationale.
    """
    cos_emi = np.clip(np.cos(np.radians(emi_ang_deg)),
                       np.cos(np.radians(FOOTPRINT_EMI_CLIP_DEG)), 1.0)
    gsd_km = rng_km * IR3S_IFOV_RAD
    footprint_diam_km = gsd_km / cos_emi
    n_row = np.maximum(1, np.round(footprint_diam_km / texel_km)).astype(np.int64)
    cos_lat = np.clip(np.cos(np.radians(lat_deg)), np.cos(np.radians(85.0)), 1.0)
    n_col = np.maximum(1, np.round(footprint_diam_km / (texel_km * cos_lat))).astype(np.int64)
    rad_row = np.clip((n_row - 1) // 2, 0, FOOTPRINT_MAX_TEXELS_HALF)
    rad_col = np.clip((n_col - 1) // 2, 0, FOOTPRINT_MAX_TEXELS_HALF)
    return rad_row, rad_col


def _footprint_splat_indices(row0, col0, rad_row, rad_col, H, W):
    """Expand each pixel's texel CENTER ``(row0[i], col0[i])`` into every
    flat texel index within its footprint-fill box (``+-rad_row[i]`` rows,
    ``+-rad_col[i]`` cols, column index wrapped mod ``W`` for the
    antimeridian, row index clipped to ``[0, H)``), grouped by unique
    ``(rad_row, rad_col)`` pairs for vectorized (no per-pixel Python loop)
    processing. Returns ``(flat_expanded, src_expanded)``: ``flat_expanded``
    is the flat texel index for every (pixel, footprint-texel) pair;
    ``src_expanded`` is the ORIGINAL pixel index (into the caller's own
    parallel arrays, e.g. color/GSD/temperature) that produced it — the
    caller looks up its own per-pixel value via ``src_expanded``, this
    function only computes geometry. A ``rad_row=rad_col=0`` pixel expands
    to exactly its own single center texel (byte-identical to the pre-Fix-1
    point-binning behavior), so this is a strict, non-fabricating
    generalization of the legacy single-texel bin.
    """
    row0 = np.asarray(row0, dtype=np.int64)
    col0 = np.asarray(col0, dtype=np.int64)
    rad_row = np.asarray(rad_row, dtype=np.int64)
    rad_col = np.asarray(rad_col, dtype=np.int64)
    pairs = np.stack([rad_row, rad_col], axis=1)
    uniq_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    flat_chunks = []
    src_chunks = []
    for gi in range(uniq_pairs.shape[0]):
        rr, rc = int(uniq_pairs[gi, 0]), int(uniq_pairs[gi, 1])
        sel = np.flatnonzero(inverse == gi)
        if sel.size == 0:
            continue
        dr = np.arange(-rr, rr + 1)
        dc = np.arange(-rc, rc + 1)
        DR, DC = np.meshgrid(dr, dc, indexing="ij")
        DR = DR.ravel()
        DC = DC.ravel()
        rows = np.clip(row0[sel][:, None] + DR[None, :], 0, H - 1)
        cols = (col0[sel][:, None] + DC[None, :]) % W
        flat_chunks.append((rows * W + cols).ravel())
        src_chunks.append(np.repeat(sel, DR.size))
    if not flat_chunks:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    return np.concatenate(flat_chunks), np.concatenate(src_chunks)


# IR3S optics (defined here, ahead of both coverage-accumulation renderers
# that now need them for the true-GSD footprint-fill calc above; previously
# only the full-period renderer needed these, defined further down — moved
# up 2026-07-05 round 5, values unchanged).
IR3S_PITCH_M = 17e-6
IR3S_EFL_M = 102.8e-3
IR3S_IFOV_RAD = IR3S_PITCH_M / IR3S_EFL_M  # 165.37 urad; cross-check vs FOV_X/640 in the returned dict


# --------------------------------------------------------------------------
# Round 6 (2026-07-05, Rocky) — two view/overlay improvements on TOP of the
# round-5 true-GSD/footprint-fill pipeline. NEITHER changes the coverage
# accumulation itself (texel binning, footprint-fill, highest-res-wins) —
# both are pure VIEW/overlay additions, applied identically to the per-phase
# (`_texture_truegsd`) and full-period (`_fullperiod_truegsd`) renderers.
#
# Improvement 1 — sub-spacecraft-tracking camera (fixes "imaged region
# rotates behind the limb"). Round 1-5 held the RENDER camera fixed (a
# static elev/azim offset) while de-spinning the body mesh into an
# inertial-anchored frame -- correct for showing the body's true orientation
# history, but means the region actually being imaged (which moves across
# the body as the real spacecraft orbits, even with body-rotation removed)
# periodically swings onto the far side, out of view. Fix: recompute the
# render camera's ANGULAR position every frame so its boresight always
# passes through the real spacecraft's nadir/sub-spacecraft point (the
# surface point where the real CAM->body-center line intersects the body),
# expressed in the SAME per-frame de-spin rotation as the mesh -- i.e. the
# render camera now "follows" the real imaging geometry instead of holding
# a fixed inertial vantage. Camera DISTANCE/parallel_scale (hence body
# apparent size) and the accumulation itself are UNCHANGED -- only the
# camera's position/pointing on its orbit around the (still de-spun) body.
# The `elev`/`azim` parameters are kept for call-signature compatibility
# but are now used only as a harmless placeholder/fallback direction (e.g.
# if the real spacecraft position is degenerate); they no longer set the
# final view for any real frame.
#
# Improvement 2 — green current-frame footprint outline. Each frame, the
# TRUE imaged-pixel boundary (a 2D convex hull computed in IMAGE (row, col)
# pixel space -- cheap, robust, guaranteed a simple non-self-intersecting
# polygon even at grazing angles/partial fill) is mapped back to the real
# 3D SURF_X/Y/Z hit points at those specific hull pixels (not an idealized
# rectangle), nudged a small distance outward along the local radial
# direction (to avoid z-fighting with the textured surface it sits on), and
# drawn as a closed green polyline actor, replaced every frame. On frames
# that are 100%-frame-filling (measured true for essentially all Justitia
# proxops frames to date -- the IR3S FOV is smaller than the body's angular
# size at these ranges), the hull correctly degenerates to the image's own
# 4 corners projected onto the body -- not a bug, the honest answer when
# every pixel is on-body.
# --------------------------------------------------------------------------


def _tracked_camera_pose(cam_bodyfixed_xyz, R, center, distance, fallback_dir):
    """Per-frame render-camera pose that keeps the REAL spacecraft's
    sub-spacecraft (nadir) point centered in view (round 6, Improvement 1).

    ``cam_bodyfixed_xyz`` is the real camera/spacecraft position (``CAM_X/
    Y/Z``) in the body-fixed-at-time-t frame (same convention as the mesh
    vertices/SURF pixels BEFORE de-spin rotation); ``R`` is that exact
    frame's own de-spin rotation matrix (``pts_render = pts_bodyfixed @
    R.T``, identical convention used for the rotating mesh). ``center`` is
    the (fixed, static) render focal point; ``distance`` is the fixed
    render-camera standoff (``half*6``, unchanged frame to frame so body
    apparent size stays constant). ``fallback_dir`` is used only in the
    degenerate case where the spacecraft is exactly at ``center`` (never
    expected on real data).

    Returns ``(campos, focal_point, up)`` ready for PyVista's
    ``camera_position = [campos, focal_point, up]``. The "up" vector is
    switched away from global +Z to global +Y whenever the tracked
    direction comes within ~26 deg of the pole, to avoid the classic
    view-up-parallel-to-view-direction degeneracy (a known simplification —
    this can still produce a single visible frame-to-frame roll jump right
    at that crossover; not smoothed here, flagged for Ada if a smoother
    full-length render shows it).
    """
    cam_render = np.asarray(cam_bodyfixed_xyz, dtype=np.float64) @ R.T
    d = cam_render - np.asarray(center, dtype=np.float64)
    dn = np.linalg.norm(d)
    if dn < 1e-9:
        d = np.asarray(fallback_dir, dtype=np.float64)
        dn = np.linalg.norm(d)
    d = d / dn
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    campos = np.asarray(center, dtype=np.float64) + d * distance
    return campos, np.asarray(center, dtype=np.float64), up


def _frame_footprint_hull_points(face_idx_2d, sx_2d, sy_2d, sz_2d):
    """Body-fixed 3D points tracing the boundary of the CURRENT frame's
    imaged (``FACE_IDX>=0``) pixel footprint, for the green outline overlay
    (round 6, Improvement 2).

    Computes a 2D convex hull in IMAGE (row, col) pixel space (fast, robust,
    always a simple/non-self-intersecting polygon regardless of grazing
    angle or partial fill), then looks up the REAL ``SURF_X_KM/Y_KM/Z_KM``
    hit point at each hull pixel — the returned ring is the true imaged
    footprint's outline, not a geometric idealization.

    Returns an ``(M, 3)`` array (in hull order, NOT yet closed — caller
    closes the loop, e.g. via ``pv.lines_from_points(..., close=True)``), or
    ``None`` if there are fewer than 3 imaged pixels or the imaged pixels
    are degenerate/collinear (no well-defined hull; both are honestly
    skipped, not fabricated).
    """
    if ConvexHull is None:
        raise RuntimeError(
            "scipy is required for the footprint-outline overlay (pip install scipy)"
        )
    valid = face_idx_2d >= 0
    rows, cols = np.nonzero(valid)
    if rows.size < 3:
        return None
    pts2d = np.stack([rows, cols], axis=1).astype(np.float64)
    try:
        hull = ConvexHull(pts2d)
    except Exception:
        return None
    hr = rows[hull.vertices]
    hc = cols[hull.vertices]
    return np.stack([sx_2d[hr, hc], sy_2d[hr, hc], sz_2d[hr, hc]], axis=1)


def _footprint_hull_actor_points(hull_pts_bodyfixed, R, center, half):
    """Rotate a frame's hull points (from
    :func:`_frame_footprint_hull_points`) into the render frame and nudge
    them a small distance outward along their own local radial direction
    (from ``center``) so the drawn green polyline sits just above the
    textured surface instead of z-fighting with it. ``half`` is the body's
    display half-size (same quantity used for ``parallel_scale``); the
    nudge is a small, fixed fraction of it (~1%), well under a facet's own
    scale for this mesh but comfortably above typical VTK z-buffer jitter.
    """
    hull_render = np.asarray(hull_pts_bodyfixed, dtype=np.float64) @ R.T
    center = np.asarray(center, dtype=np.float64)
    radial = hull_render - center
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm[radial_norm < 1e-9] = 1.0
    return hull_render + (radial / radial_norm) * (0.01 * half)


def render_coverage_accumulation_animation_texture_truegsd(
    target,
    camera_subdir,
    thermal_subdir,
    naif_frame,
    metakernel,
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0),
    vmin=None,
    vmax=None,
    elev=20,
    azim=25,
    fps=8,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
    thermal_ext=None,
    caption=None,
    texture_width=4096,
    archive_existing=True,
    track_subspacecraft=True,
    draw_footprint_box=True,
    footprint_box_color="lime",
    footprint_box_line_width=5,
):
    """TRUE-PER-PIXEL coverage-accumulation animation (2026-07-05, round 3;
    footprint-fill added round 5, Fix 1; sub-spacecraft camera tracking +
    green current-frame footprint outline added round 6, Improvements 1+2).

    Bins each frame's real ``SURF_X_KM``/``SURF_Y_KM``/``SURF_Z_KM`` pixel
    hit points into an equirectangular texel raster (most-recent-frame-
    wins), then maps that raster onto the whole body mesh via the SAME
    PyVista/VTK true-z-buffer pipeline as round 2
    (:func:`_build_equirect_uv_mesh`, the antimeridian-seam fix, the
    ``pl.render()``-before-``screenshot()`` requirement). Requires a
    ``ptm_camera_sim`` build that writes ``SURF_X_KM``/``SURF_Y_KM``/
    ``SURF_Z_KM`` (md5 25866a03673bbf743f635be1b51fca13 or newer) — raises
    ``RuntimeError`` up front if the first frame lacks them, rather than
    silently falling back to a coarser method.

    **Round 5 (Fix 1):** each pixel now fills its true ~GSD x GSD ground
    footprint (emission-angle-elongated per :func:`_footprint_fill_radii`),
    not just its own single nearest texel — see the module note above
    ``_footprint_fill_radii``/``_footprint_splat_indices`` for the full
    rationale (closes within-frame texel-vs-pixel-spacing speckle honestly,
    without fabricating coverage beyond the real observed area). A
    ``rad_row=rad_col=0`` pixel (footprint <= ~1 texel) degrades to exactly
    the pre-Fix-1 single-texel bin, so this is a strict generalization, not
    a behavior change for well-matched texel/GSD phases.

    ``texture_width`` (texels around 360 deg of longitude; height is always
    ``texture_width // 2``) should be tuned PER PHASE to roughly track that
    phase's own measured true GSD (see :func:`make_justitia_proxops_phase_animations`)
    — there is no longer a universal "good enough" default the way round
    2's fixed 4096 was, since true GSD now spans ~2-43 m across the 4
    Justitia proxops phases. Returns ``texel_km`` for honest per-phase
    reporting.

    Returns a dict: ``out_mp4``, ``archived_legacy_mp4``, ``thermal_ext``,
    ``renderer`` (``"pyvista_texture_truegsd"``), ``texture_width``,
    ``texture_height``, ``texel_km``, ``n_texels_touched``,
    ``n_texels_total``, ``area_pct`` (NEW definition: latitude-weighted
    fraction of an assumed MEAN-RADIUS-SPHERE's surface area covered by
    touched texels — an equirectangular-projection approximation, honestly
    NOT the true irregular-body area; see caveat below), ``facet_pct_legacy``
    (simple count fraction of mesh facets with >=1 pixel EVER imaged,
    matching round 1/2's definition exactly, for direct comparison),
    ``n_facets_touched``, ``n_facets_total``, ``n_pixels_total``.

    Caveat (reported honestly, not silently): ``area_pct`` approximates the
    body as a sphere of its own mean vertex radius for the cos(latitude)
    equirectangular area-weighting. Justitia's real DAMIT shape is
    irregular (facet sqrt-area spans 0.04-13 km, i.e. real elongation), so
    this is a genuine but standard cartographic-projection approximation,
    not an exact computation on the true mesh. ``facet_pct_legacy`` (which
    IS computed on the true mesh facet areas/topology) is the exact
    companion statistic for that reason.

    **Round 6** (see the module note above ``_tracked_camera_pose``):
    ``track_subspacecraft`` (default ``True``) re-points the render camera
    every frame so the real spacecraft's sub-spacecraft/nadir point stays
    centered — the imaged region no longer swings behind the limb.
    ``draw_footprint_box`` (default ``True``) overlays a green outline of
    the CURRENT frame's actually-imaged pixel footprint. Both are pure
    view/overlay changes — the coverage accumulation itself (texel
    binning, footprint-fill, highest-res-wins) is byte-identical whether
    these are on or off; set either to ``False`` to reproduce the exact
    round-5 view/frames for regression comparison.
    """
    import spiceypy as sp

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")
    if pv is None:
        raise RuntimeError("pyvista is required (pip install pyvista)")
    if draw_footprint_box and ConvexHull is None:
        raise RuntimeError(
            "scipy is required for draw_footprint_box=True (pip install scipy), "
            "or pass draw_footprint_box=False"
        )

    cdir = camera_dir(target) / camera_subdir
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = camera_dir(target) / camera_subdir / f"{target}_coverage_accum_pub.mp4"
    if title is None:
        title = f"{body_label} — accumulating IR coverage (most-recent-imaged)"

    fits_files = sorted(glob.glob(str(cdir / "frame_*.fits")))
    if not fits_files:
        raise RuntimeError(f"no frame_*.fits under {cdir}")

    with fits.open(fits_files[0]) as hdul0:
        names0 = {h.name for h in hdul0}
    missing = [k for k in ("SURF_X_KM", "SURF_Y_KM", "SURF_Z_KM") if k not in names0]
    if missing:
        raise RuntimeError(
            f"{fits_files[0]} is missing {missing} — re-render with ptm_camera_sim "
            "md5 25866a03673bbf743f635be1b51fca13 (or newer) before using the "
            "true-per-pixel renderer; see render_coverage_accumulation_animation_"
            "texture_subfacet_legacy for the pre-SURF fallback."
        )

    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    verts, faces = read_mesh_verts_faces(tdir / "mesh.ply")
    n_faces = len(faces)
    body_pts_unwelded, body_faces_vtk, body_uv = _build_equirect_uv_mesh(verts, faces)

    W = int(texture_width)
    H = W // 2
    mean_r = float(np.mean(np.linalg.norm(verts, axis=1)))
    circumference_km = 2.0 * np.pi * mean_r
    texel_km = circumference_km / W

    tri = verts[faces]
    Av, Bv, Cv = tri[:, 0], tri[:, 1], tri[:, 2]
    facet_area = 0.5 * np.linalg.norm(np.cross(Bv - Av, Cv - Av), axis=1)
    total_area = facet_area.sum()

    # --- pass 1 (cheap: FACE_IDX + one thermal extension only, no SURF
    # arrays) — global vmin/vmax + the UTC list, same convention as round 2. ---
    utcs = []
    all_valid_T = []
    for fp in fits_files:
        with fits.open(fp) as hdul:
            face_idx = np.asarray(hdul["FACE_IDX"].data)
            temp_k = np.asarray(hdul[thermal_ext].data, dtype=float)
            utc = hdul[0].header["UTC_TIME"]
        utcs.append(utc)
        valid = face_idx >= 0
        all_valid_T.append(temp_k[valid])

    if vmin is None or vmax is None:
        allT = np.concatenate(all_valid_T)
        if vmin is None:
            vmin = 5.0 * np.floor(allT.min() / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(allT.max() / 5.0)

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    neutral_u8 = (np.array(neutral_rgba[:3]) * 255).astype(np.uint8)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    if archive_existing:
        _archived = _archive_existing(out_mp4)
    else:
        _archived = None

    # --- per-frame de-spin rotation (inertial-anchored view) — identical
    # convention to round 1/2. R_list is saved (round 6) so the tracked
    # sub-spacecraft-camera pose can rotate the REAL spacecraft position
    # into this exact same render frame every frame. ---
    sp.furnsh(str(metakernel))
    et_list = [sp.str2et(u) for u in utcs]
    et0 = et_list[0]
    R0 = sp.pxform("J2000", naif_frame, et0)
    rotated_body_pts = []
    R_list = []
    for et in et_list:
        Rt = sp.pxform(naif_frame, "J2000", et)
        R = R0 @ Rt
        rotated_body_pts.append(body_pts_unwelded @ R.T)
        R_list.append(R)
    sp.unload(str(metakernel))

    # Fixed elev/azim placeholder direction — used only to initialize the
    # Plotter's camera before the per-frame loop runs, and as the
    # `fallback_dir` for the (never expected on real data) degenerate case
    # where the real spacecraft sits exactly at the body center. When
    # `track_subspacecraft=True` (default), every REAL frame overrides this
    # with the true sub-spacecraft-tracking pose (round 6, Improvement 1).
    d_vec = np.array([
        np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
        np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
        np.sin(np.radians(elev)),
    ])
    campos = center + d_vec * half * 6
    cam_distance = half * 6

    pv_win_w = max(64, int(round(0.83 * target_width_px)))
    pv_win_h = max(64, int(round(0.90 * target_height_px)))

    body_mesh = pv.PolyData(rotated_body_pts[0].copy(), body_faces_vtk)
    body_mesh.active_texture_coordinates = body_uv

    pl = pv.Plotter(off_screen=True, window_size=[pv_win_w, pv_win_h])
    pl.set_background("black")
    actor = pl.add_mesh(body_mesh, texture=pv.Texture(np.zeros((2, 2, 3), dtype=np.uint8)),
                         lighting=False)
    pl.camera_position = [tuple(campos), tuple(center), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = half
    footprint_line_actor = None

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 0.83, 0.92])
    ax.set_facecolor("black")
    ax.set_axis_off()
    im = ax.imshow(np.zeros((pv_win_h, pv_win_w, 3), dtype=np.uint8), aspect="auto")

    cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, color="white", fontsize=14)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(title, color="white", fontsize=15, y=0.97)
    footer_fontsize = 11.0 if caption else 13
    footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=footer_fontsize,
                           ha="left", va="bottom", family="monospace")

    # --- persistent whole-run rasters (updated in place; NOT one copy per
    # frame — see the module note above on why). ---
    rgba_img = np.empty((H, W, 3), dtype=np.uint8)
    rgba_img[:, :] = neutral_u8
    touched_ever = np.zeros((H, W), dtype=bool)
    facet_touched = np.zeros(n_faces, dtype=bool)
    lat_row_centers = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    row_weight = np.cos(np.radians(lat_row_centers))  # equirect area-weighting (mean-sphere approx)
    total_weight = row_weight.sum() * W
    n_px_total = 0

    rgba_flat = rgba_img.reshape(-1, 3)
    touched_flat = touched_ever.reshape(-1)

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_coverage_truegsd_frames_"))
    try:
        for i, (fp, utc, pts_i) in enumerate(zip(fits_files, utcs, rotated_body_pts)):
            R_i = R_list[i]
            with fits.open(fp) as hdul:
                face_idx = np.asarray(hdul["FACE_IDX"].data)
                sx = np.asarray(hdul["SURF_X_KM"].data, dtype=np.float64)
                sy = np.asarray(hdul["SURF_Y_KM"].data, dtype=np.float64)
                sz = np.asarray(hdul["SURF_Z_KM"].data, dtype=np.float64)
                emi_ang = np.asarray(hdul["EMI_ANG"].data, dtype=np.float64)
                vals_full = np.asarray(hdul[thermal_ext].data, dtype=float)
                h0 = hdul[0].header
                camx, camy, camz = h0["CAM_X"], h0["CAM_Y"], h0["CAM_Z"]

            valid = face_idx >= 0
            n_px_total += int(valid.sum())
            xs, ys, zs = sx[valid], sy[valid], sz[valid]
            vals = vals_full[valid]
            emi = emi_ang[valid]

            # --- Round 6, Improvement 1: re-point the render camera every
            # frame so the real sub-spacecraft point stays centered. ---
            if track_subspacecraft:
                campos_i, focal_i, up_i = _tracked_camera_pose(
                    (camx, camy, camz), R_i, center, cam_distance, d_vec)
                pl.camera_position = [tuple(campos_i), tuple(focal_i), tuple(up_i)]
                pl.camera.parallel_projection = True
                pl.camera.parallel_scale = half

            # Task-specified per-pixel lat/lon, computed DIRECTLY from the
            # real hit point — no barycentric solve, no facet lookup.
            r = np.sqrt(xs * xs + ys * ys + zs * zs)
            lat = np.degrees(np.arcsin(np.clip(zs / r, -1.0, 1.0)))
            lon = np.degrees(np.arctan2(ys, xs))

            col = np.floor((lon + 180.0) / 360.0 * W).astype(np.int64) % W
            row = np.clip(np.floor((90.0 - lat) / 180.0 * H).astype(np.int64), 0, H - 1)

            # --- Footprint-fill (round 5, Fix 1): each pixel paints its
            # TRUE ground footprint (GSD x GSD, emission-angle-elongated),
            # not just its own single texel — see the module note above
            # ``_footprint_fill_radii``/``_footprint_splat_indices``. ---
            rng_km = np.sqrt((xs - camx) ** 2 + (ys - camy) ** 2 + (zs - camz) ** 2)
            rad_row, rad_col = _footprint_fill_radii(rng_km, emi, lat, texel_km)
            flat_exp, src_exp = _footprint_splat_indices(row, col, rad_row, rad_col, H, W)

            colors = (cmap_obj(norm(vals))[:, :3] * 255).astype(np.uint8)
            rgba_flat[flat_exp] = colors[src_exp]   # most-recent-wins (this frame overwrites)
            touched_flat[flat_exp] = True
            facet_touched[np.unique(face_idx[valid].astype(int))] = True

            tex = pv.Texture(rgba_img)
            tex.wrap = 1  # REPEAT — antimeridian columns wrap correctly (lon already in [-180,180))
            actor.texture = tex
            body_mesh.points = pts_i

            # --- Round 6, Improvement 2: green outline of THIS frame's
            # actually-imaged footprint, replaced every frame (never
            # accumulated — that's what the texture raster is for). ---
            if footprint_line_actor is not None:
                pl.remove_actor(footprint_line_actor)
                footprint_line_actor = None
            if draw_footprint_box:
                hull_bf = _frame_footprint_hull_points(face_idx, sx, sy, sz)
                if hull_bf is not None and len(hull_bf) >= 3:
                    hull_render = _footprint_hull_actor_points(hull_bf, R_i, center, half)
                    line_poly = pv.lines_from_points(hull_render, close=True)
                    footprint_line_actor = pl.add_mesh(
                        line_poly, color=footprint_box_color,
                        line_width=footprint_box_line_width,
                        render_lines_as_tubes=True, lighting=False)

            pl.render()
            frame_rgb = pl.screenshot(return_img=True)

            area_pct_i = 100.0 * float((row_weight[:, None] * touched_ever).sum()) / total_weight
            im.set_data(frame_rgb)
            # Trimmed 2026-07-05 (round 4, Hayne): show ONLY the UTC timestamp
            # and the % coverage on-frame — no explanatory wording, no
            # honesty caption unless the caller explicitly passes one (the
            # Justitia proxops phase driver no longer does, see
            # make_justitia_proxops_phase_animations).
            label = f"{_format_utc_clean(utc)}    {area_pct_i:.2f}% coverage"
            if caption:
                label += f"\n{caption}"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)
        pl.close()

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        n_facets_touched = int(facet_touched.sum())
        area_pct_final = 100.0 * float((row_weight[:, None] * touched_ever).sum()) / total_weight
        return {
            "out_mp4": out_mp4,
            "archived_legacy_mp4": _archived,
            "thermal_ext": thermal_ext,
            "renderer": "pyvista_texture_truegsd",
            "footprint_fill": True,
            "track_subspacecraft": track_subspacecraft,
            "footprint_box": draw_footprint_box,
            "texture_width": W,
            "texture_height": H,
            "texel_km": texel_km,
            "n_texels_touched": int(touched_ever.sum()),
            "n_texels_total": W * H,
            "area_pct": area_pct_final,
            "facet_pct_legacy": 100.0 * n_facets_touched / n_faces,
            "n_facets_touched": n_facets_touched,
            "n_facets_total": n_faces,
            "n_pixels_total": n_px_total,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


# The true-per-pixel renderer is now the LIVE implementation behind the
# bare name (matching this module's existing convention: the un-suffixed
# name always points at the current-best approach; superseded approaches
# are kept under an explicit `_..._legacy` name for reference/regression).
render_coverage_accumulation_animation = render_coverage_accumulation_animation_texture_truegsd


# --------------------------------------------------------------------------
# CLI driver — (269) Justitia proxops
# --------------------------------------------------------------------------

def make_justitia_proxops_animations():
    """Both Justitia proxops animations: (1) camera-view IR3S sequence,
    (2) the new inertial-fixed-view accumulating-coverage render."""
    target = "269justitia"
    camera_subdir = "proxops"
    thermal_subdir = "proxops"
    body_label = "(269) Justitia"
    naif_frame = "IAU_JUSTITIA_269"
    metakernel = (
        "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/"
        "data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel_camera.tm"
    )
    cdir = camera_dir(target) / camera_subdir
    written = {}

    written["camera_view"] = render_mp4_from_fits(
        _sorted_fits(cdir), "TEMP_K", "inferno",
        "Brightness T (K) @ 10.5 µm",
        cdir / "justitia_proxops_camera_view_pub.mp4",
        f"{body_label} — {IR_CAM_LABEL} proximity-ops thermal-IR")

    cov = render_coverage_accumulation_animation(
        target, camera_subdir, thermal_subdir, naif_frame, metakernel,
        out_mp4=cdir / "justitia_proxops_coverage_accum_pub.mp4",
        body_label=body_label,
        title=f"{body_label} — accumulating {IR_CAM_LABEL} coverage (most-recent-imaged)")
    written["coverage_accum"] = cov["out_mp4"]
    written["coverage_accum_area_pct"] = cov["area_pct"]
    written["coverage_accum_facet_pct_legacy"] = cov["facet_pct_legacy"]
    written["coverage_accum_thermal_ext"] = cov["thermal_ext"]
    written["coverage_accum_detail"] = cov

    return written


# --------------------------------------------------------------------------
# CLI driver — (269) Justitia proxops, PER-PHASE (2026-07-04 smooth-thermal
# campaign): 3 gravity-mapping-file altitude tiers ("Orbit A/B/C", i.e.
# High/Mid/Low) + the close-approach descent, all within the single
# available mid-GM trajectory file. See memory
# ema-justitia-proxops-trajectory-inventory-2026-07-04 for why these are
# NOT three separate GM-assumption trajectories.
# --------------------------------------------------------------------------

# ``texture_width`` (2026-07-05, round 3): tuned PER PHASE to track each
# phase's own measured TRUE per-pixel GSD (median real-space distance
# between horizontally-adjacent on-body SURF_X/Y/Z pixels, measured
# directly post-re-render — NOT the coarser angular pre-estimate from the
# 2026-07-04 smooth-thermal campaign, which this measurement supersedes for
# orbit_mid/orbit_low/descent):
#   orbit_high: measured ~40-43 m across the window -> 4096 (39.9 m/texel,
#               reuses round 2's exact default — already a close match).
#   orbit_mid:  measured ~12.3-13.3 m -> 12800 (12.45 m/texel, closely
#               matched).
#   orbit_low:  measured ~6.3-7.2 m -> capped at 16384 (9.73 m/texel,
#               DELIBERATELY coarser than the true GSD — see the honest
#               tradeoff note in the session notes / task deliverable).
#   descent:    measured 2.08 m (closest approach, frame ~32) to 8.2 m
#               (window edges) -> capped at 16384 (9.73 m/texel) same as
#               Low — this resolves the coarser parts of the window but
#               explicitly does NOT resolve the ~2 m closest-approach
#               minimum (would need a ~76,600-wide texture, i.e. ~23x the
#               texel count, impractical here).
JUSTITIA_PROXOPS_PHASES = {
    "orbit_high":   dict(camera_subdir="orbit_high",   thermal_subdir="orbit_high",
                          label_suffix="High orbit (~254 km)", texture_width=4096),
    "orbit_mid":    dict(camera_subdir="orbit_mid",    thermal_subdir="",
                          label_suffix="Mid orbit (~97 km, MIDGM epoch)", texture_width=12800),
    "orbit_low":    dict(camera_subdir="orbit_low",    thermal_subdir="orbit_low",
                          label_suffix="Low orbit (~62 km)", texture_width=16384),
    "descent":      dict(camera_subdir="descent",      thermal_subdir="descent",
                          label_suffix="Close-approach descent (min 33.5 km)", texture_width=16384),
}


def make_justitia_proxops_phase_animations(phase_key, parallel_workers=8,
                                            coverage_only=True,
                                            target_width_px=1600,
                                            target_height_px=900,
                                            crf=20, max_mb=95.0):
    """Per-phase camera-view + coverage-accumulation animations for the
    2026-07-04 smooth-thermal (TEMP_K_SMOOTH) re-render campaign. `phase_key`
    is one of `JUSTITIA_PROXOPS_PHASES` ("orbit_high", "orbit_mid",
    "orbit_low", "descent"). Reuses the same tooling as
    `make_justitia_proxops_animations` (the original flat MIDGM product,
    left untouched) but: (1) auto-picks TEMP_K_SMOOTH over flat TEMP_K via
    `_pick_thermal_ext`, (2) stamps `HONESTY_CAPTION` on every frame of both
    animations, (3) reads/writes the new per-phase subdirectories rather
    than the flat `proxops/` dir. `thermal_subdir=""` (Mid orbit) means
    "reuse the existing flat thermal output" (its thermal solve was not
    re-run — only its camera frames were, with the new smooth binary)."""
    target = "269justitia"
    phase = JUSTITIA_PROXOPS_PHASES[phase_key]
    camera_subdir = f"proxops/{phase['camera_subdir']}"
    thermal_subdir = f"proxops/{phase['thermal_subdir']}" if phase["thermal_subdir"] else "proxops"
    body_label = "(269) Justitia"
    naif_frame = "IAU_JUSTITIA_269"
    metakernel = (
        "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/"
        "data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel_camera.tm"
    )
    cdir = camera_dir(target) / camera_subdir
    label = f"{body_label} {phase['label_suffix']}"
    written = {}

    frame_files = _sorted_fits(cdir)
    ext = _pick_thermal_ext(frame_files)
    written["thermal_ext_used"] = ext

    # Shorter on-frame titles than `label` (which stays available for the
    # body_label param / other metadata) — long titles run off the frame
    # edges at the shared render functions' fixed font size, worst on the
    # narrower 640x480-based camera-view frames.
    suffix = phase["label_suffix"]
    # The camera-VIEW product is unchanged by round 6 (only the coverage-accum
    # animations get the #1/#2 overlays + the parallel re-render) — skip it by
    # default (`coverage_only=True`) so the final round-6 pass doesn't
    # needlessly re-encode it.
    if not coverage_only:
        written["camera_view"] = render_mp4_from_fits(
            frame_files, ext, "inferno",
            "Brightness T (K) @ 10.5 µm",
            cdir / f"justitia_{phase_key}_camera_view_pub.mp4",
            f"{suffix} — {IR_CAM_LABEL}",
            caption=HONESTY_CAPTION)

    # 2026-07-05 (round 4, Hayne): trimmed on-animation text for the
    # coverage-accum render specifically — drop the honesty caption and
    # shorten the title to just the phase label; the per-frame footer now
    # shows ONLY the UTC timestamp + % coverage (see the label-construction
    # comment inside render_coverage_accumulation_animation_texture_truegsd).
    # camera_view (above) is unchanged/out of scope for this trim.
    #
    # Round 6 (Ada): dispatch to the PARALLEL two-stage renderer when
    # `parallel_workers>1` (byte-identical frame content — see the module note
    # above the parallel core); fall back to the serial oracle for
    # `parallel_workers<=1`. Default resolution reduced 1920x1080 -> 1600x900
    # with a size-target CRF fallback so every MP4 lands < ~95 MB.
    cov_kwargs = dict(
        out_mp4=cdir / f"justitia_{phase_key}_coverage_accum_pub.mp4",
        body_label=label, title=suffix,
        texture_width=phase.get("texture_width", 4096))
    if parallel_workers and parallel_workers > 1:
        cov = render_coverage_accumulation_animation_texture_truegsd_parallel(
            target, camera_subdir, thermal_subdir, naif_frame, metakernel,
            target_width_px=target_width_px, target_height_px=target_height_px,
            crf=crf, max_mb=max_mb, n_workers=parallel_workers, **cov_kwargs)
    else:
        cov = render_coverage_accumulation_animation(
            target, camera_subdir, thermal_subdir, naif_frame, metakernel,
            **cov_kwargs)
    written["coverage_accum"] = cov["out_mp4"]
    written["coverage_accum_area_pct"] = cov["area_pct"]
    written["coverage_accum_facet_pct_legacy"] = cov["facet_pct_legacy"]
    written["coverage_accum_thermal_ext"] = cov["thermal_ext"]
    written["coverage_accum_detail"] = cov

    return written


def make_justitia_proxops_all_phase_animations(parallel_workers=8, **kwargs):
    """Run `make_justitia_proxops_phase_animations` for all 4 phases.
    `parallel_workers`/`kwargs` are forwarded (round 6: parallel two-stage
    coverage renderer, 1600x900, size-target CRF)."""
    return {k: make_justitia_proxops_phase_animations(
        k, parallel_workers=parallel_workers, **kwargs)
        for k in JUSTITIA_PROXOPS_PHASES}


# --------------------------------------------------------------------------
# Full-proxops-period coverage animation — HIGHEST-RESOLUTION-WINS across
# all 4 mission phases + a coverage-vs-resolution histogram (2026-07-05,
# round 4 refinements: Rocky, from Iris/Hayne).
# --------------------------------------------------------------------------
#
# Unlike the per-phase render_coverage_accumulation_animation_texture_truegsd
# (most-recent-frame-wins, one phase at a time), this combines ALL 4 proxops
# phases into ONE texel raster animated over the FULL mission timeline, with
# HIGHEST-RESOLUTION-WINS per-texel selection: a later, FINER-GSD observation
# overwrites an earlier coarser one, but a later COARSER observation of an
# already finely-imaged texel does NOT regress it (the opposite failure mode
# from most-recent-wins, which would let the wide, coarse High-orbit pass
# silently overwrite fine Descent/Low patches if it were replayed after them
# in time — it isn't, mission order is High->Mid->Low->Descent, but
# highest-res-wins is the semantically correct rule regardless of order).
#
# Mission-time ordering (confirmed 2026-07-05, not assumed): each phase's
# first/last UTC_TIME are strictly increasing and non-overlapping — High
# 2035-08-25..28, Mid 2035-10-01, Low 2036-01-28, Descent 2036-02-14..15 —
# so the natural JUSTITIA_PROXOPS_PHASES dict order IS mission-chronological
# order; no separate re-sort needed.
#
# GSD (this round's task spec): GSD_pixel = |SURF_pixel - CAM| * IFOV,
# IFOV = detector pitch / EFL = 17 um / 102.8 mm = 165.37 urad (IR3S optics,
# examples/cameras/ema_ecam_ir3s_rockox.yaml). Cross-checked against the
# FITS header's own FOV_X / 640: 165.22 urad on real frames (0.09% apart —
# consistent with a small-angle pitch/EFL approximation vs. the header's
# exact tan-based FOV_X, not a bug). CAM_X/Y/Z (header) and SURF_X/Y/Z
# (per-pixel) are BOTH already in the same body-fixed frame, so the vector
# subtraction needs no rotation. Verified against real frames (High/Mid/Low/
# Descent, several frames each): the resulting per-pixel range reproduces
# each phase's already-measured true GSD almost exactly (e.g. Descent frame
# ~32, the closest-approach frame: computed median 2.077 m vs. the
# previously-measured "2.08 m" — same number, independently reproduced) and
# is BYTE-IDENTICAL (<1.2e-4 km max diff, float32 rounding) to the camera
# sim's own DIST_KM field — i.e. DIST_KM already WAS the true per-pixel
# range even before this task. Using the explicit SURF/CAM formula here per
# the task spec (self-documenting, matches the task's own derivation), not
# silently substituting DIST_KM — but flagging the equivalence as a
# validated cross-check, not a coincidence.
#
# Practical resolution ceiling (same honest tradeoff as round 3): the
# combined accumulation/histogram uses texture_width=16384 (9.73 m/texel,
# the same practical ceiling already used for Low/Descent) — genuinely
# resolving the Descent closest-approach ~2 m minimum would need a
# ~76,600-wide texture (~22x the texel count), which round 3 already found
# impractical for this PyVista/VTK pipeline. This means: the reported GSD
# VALUES themselves (used for highest-res-wins selection and the histogram
# x-axis) are the true per-pixel physical numbers (not texel-size-derived),
# but the per-texel "best GSD ever recorded here" bookkeeping merges
# multiple distinct real pixel footprints that happen to land in the same
# 9.73 m texel — so a texel's credited area can slightly OVER-claim how
# much of it was truly imaged at its single best-recorded GSD value. This is
# the same texel-vs-true-GSD granularity caveat already accepted for the
# per-phase renders, carried through honestly here rather than re-litigated
# or hidden.
#
# 2026-07-05 (round 5, Fix 1): this is now MUCH less true of the round-4
# "62.4% never imaged" area_pct specifically — footprint-fill (see the
# module note above the per-phase renderer, ``_footprint_fill_radii``/
# ``_footprint_splat_indices``) means a coarse High/Mid-orbit pixel now
# correctly paints its full true ~5-texel footprint on this grid instead of
# a single point, closing most of round 4's flagged grid-mismatch
# under-credit rather than merely explaining it away.
#
# IR3S_PITCH_M/EFL_M/IFOV_RAD are now defined ABOVE (ahead of the per-phase
# renderer, which needs them too as of round 5) — no longer redefined here.


def _reduce_best_per_texel(flat_idx, metric):
    """For each unique value in ``flat_idx``, return ``(uniq_flat, best_pos)``
    where ``best_pos`` indexes into the ORIGINAL arrays at the position with
    the SMALLEST ``metric`` among all entries sharing that flat index —
    exact (via a stable lexsort), not an approximation. Used to reduce a
    single frame's (possibly many-pixels-per-texel) hits down to one
    "this frame's best GSD per touched texel" value before comparing against
    the running whole-mission best."""
    order = np.lexsort((metric, flat_idx))  # sort by flat_idx, then metric ascending
    sorted_flat = flat_idx[order]
    uniq_flat, first_pos = np.unique(sorted_flat, return_index=True)
    best_pos = order[first_pos]
    return uniq_flat, best_pos


def render_coverage_accumulation_animation_fullperiod_truegsd(
    target,
    phase_keys=("orbit_high", "orbit_mid", "orbit_low", "descent"),
    naif_frame="IAU_JUSTITIA_269",
    metakernel=None,
    out_mp4=None,
    body_label=None,
    title=None,
    cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0),
    vmin=None,
    vmax=None,
    elev=20,
    azim=25,
    fps=8,
    target_width_px=1920,
    target_height_px=1080,
    crf=16,
    dpi=200,
    thermal_ext=None,
    texture_width=16384,
    archive_existing=True,
    save_raster_npz=None,
    track_subspacecraft=True,
    draw_footprint_box=True,
    footprint_box_color="lime",
    footprint_box_line_width=5,
):
    """Single coverage-accumulation animation spanning the ENTIRE proxops
    imaging campaign (all 4 mission phases, in strict mission-time order),
    with HIGHEST-RESOLUTION-WINS per-texel selection — see the module note
    above for the full rationale, the mission-time-ordering confirmation,
    and the GSD formula/cross-check/practical-ceiling caveats.

    Returns the same-shaped dict as
    :func:`render_coverage_accumulation_animation_texture_truegsd`, plus
    ``n_frames``, ``ifov_rad``, ``ifov_check_fov_over_nx_rad`` (the task's
    own cross-check, measured from the first frame's header), and
    ``raster_npz`` (path to the saved per-texel ``best_gsd_m``/``touched``
    raster, if ``save_raster_npz`` was given — consumed directly by
    :func:`fig_coverage_resolution_histogram`, no recomputation needed).

    **Round 6** (identical to the per-phase renderer — see the module note
    above ``_tracked_camera_pose``): ``track_subspacecraft`` (default
    ``True``) keeps the real spacecraft's sub-spacecraft point centered
    every frame; ``draw_footprint_box`` (default ``True``) overlays a green
    outline of the current frame's actually-imaged footprint. Both are
    view/overlay-only — the highest-resolution-wins accumulation itself is
    unchanged whether these are on or off.
    """
    import spiceypy as sp

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")
    if pv is None:
        raise RuntimeError("pyvista is required (pip install pyvista)")
    if draw_footprint_box and ConvexHull is None:
        raise RuntimeError(
            "scipy is required for draw_footprint_box=True (pip install scipy), "
            "or pass draw_footprint_box=False"
        )

    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = (camera_dir(target) / "proxops" / "fullperiod"
                   / f"{target}_fullperiod_coverage_accum_pub.mp4")
    if title is None:
        title = f"{body_label} — full proximity-ops campaign"
    if metakernel is None:
        metakernel = (
            "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/"
            "data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel_camera.tm"
        )

    # --- resolve each phase's camera dir + fits files, in mission-time
    # (== dict declaration) order. ---
    fits_files = []
    for pk in phase_keys:
        ph = JUSTITIA_PROXOPS_PHASES[pk]
        cdir = camera_dir(target) / "proxops" / ph["camera_subdir"]
        pf = _sorted_fits(cdir)
        if not pf:
            raise RuntimeError(f"no frame_*.fits under {cdir}")
        fits_files.extend(pf)

    with fits.open(fits_files[0]) as hdul0:
        names0 = {h.name for h in hdul0}
    missing = [k for k in ("SURF_X_KM", "SURF_Y_KM", "SURF_Z_KM") if k not in names0]
    if missing:
        raise RuntimeError(
            f"{fits_files[0]} is missing {missing} — need the SURF-writing "
            "ptm_camera_sim binary (md5 25866a03673bbf743f635be1b51fca13 or newer)."
        )

    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    # Mesh is IDENTICAL (md5-verified 2026-07-05) across all 4 phases' own
    # thermal dirs — read once from the first phase.
    first_ph = JUSTITIA_PROXOPS_PHASES[phase_keys[0]]
    first_thermal_subdir = (
        f"proxops/{first_ph['thermal_subdir']}" if first_ph["thermal_subdir"] else "proxops"
    )
    verts, faces = read_mesh_verts_faces(thermal_dir(target) / first_thermal_subdir / "mesh.ply")
    n_faces = len(faces)
    body_pts_unwelded, body_faces_vtk, body_uv = _build_equirect_uv_mesh(verts, faces)

    W = int(texture_width)
    H = W // 2
    mean_r = float(np.mean(np.linalg.norm(verts, axis=1)))
    texel_km = (2.0 * np.pi * mean_r) / W

    # --- pass 1: cheap header/FACE_IDX/thermal-only scan for global vmin/
    # vmax + the UTC list + the IFOV cross-check (measured once). ---
    utcs = []
    all_valid_T = []
    ifov_check = None
    for fp in fits_files:
        with fits.open(fp) as hdul:
            face_idx = np.asarray(hdul["FACE_IDX"].data)
            temp_k = np.asarray(hdul[thermal_ext].data, dtype=float)
            h = hdul[0].header
            utc = h["UTC_TIME"]
            if ifov_check is None:
                nx = face_idx.shape[1]
                ifov_check = np.radians(float(h["FOV_X"])) / nx
        utcs.append(utc)
        valid = face_idx >= 0
        all_valid_T.append(temp_k[valid].astype(np.float32))

    if vmin is None or vmax is None:
        allT = np.concatenate(all_valid_T)
        if vmin is None:
            vmin = 5.0 * np.floor(float(allT.min()) / 5.0)
        if vmax is None:
            vmax = 5.0 * np.ceil(float(np.percentile(allT, 99.5)) / 5.0)
    del all_valid_T

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    cmap_obj = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    neutral_u8 = (np.array(neutral_rgba[:3]) * 255).astype(np.uint8)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8

    fig_w = target_width_px / dpi
    fig_h = target_height_px / dpi

    if archive_existing:
        _archived = _archive_existing(out_mp4)
    else:
        _archived = None

    # --- per-frame de-spin rotation, ONE global et0 anchor (first frame of
    # the WHOLE mission, i.e. Orbit-High frame 0) — same pxform convention as
    # the per-phase renderers, just a single shared anchor across all 264
    # frames instead of a fresh one per phase. R_list saved (round 6) for
    # sub-spacecraft-camera tracking, same convention as the per-phase
    # renderer. ---
    sp.furnsh(str(metakernel))
    et_list = [sp.str2et(u) for u in utcs]
    et0 = et_list[0]
    R0 = sp.pxform("J2000", naif_frame, et0)
    rotated_body_pts = []
    R_list = []
    for et in et_list:
        Rt = sp.pxform(naif_frame, "J2000", et)
        R = R0 @ Rt
        rotated_body_pts.append(body_pts_unwelded @ R.T)
        R_list.append(R)
    sp.unload(str(metakernel))

    # Fixed elev/azim placeholder — Plotter-init value + degenerate-case
    # fallback only; every real frame is overridden by the tracked
    # sub-spacecraft pose when `track_subspacecraft=True` (default).
    d_vec = np.array([
        np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
        np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
        np.sin(np.radians(elev)),
    ])
    campos = center + d_vec * half * 6
    cam_distance = half * 6

    pv_win_w = max(64, int(round(0.83 * target_width_px)))
    pv_win_h = max(64, int(round(0.90 * target_height_px)))

    body_mesh = pv.PolyData(rotated_body_pts[0].copy(), body_faces_vtk)
    body_mesh.active_texture_coordinates = body_uv

    pl = pv.Plotter(off_screen=True, window_size=[pv_win_w, pv_win_h])
    pl.set_background("black")
    actor = pl.add_mesh(body_mesh, texture=pv.Texture(np.zeros((2, 2, 3), dtype=np.uint8)),
                         lighting=False)
    pl.camera_position = [tuple(campos), tuple(center), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = half
    footprint_line_actor = None

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 0.83, 0.92])
    ax.set_facecolor("black")
    ax.set_axis_off()
    im = ax.imshow(np.zeros((pv_win_h, pv_win_w, 3), dtype=np.uint8), aspect="auto")

    cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, color="white", fontsize=14)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(title, color="white", fontsize=15, y=0.97)
    # Trimmed on-animation text (round 4, same convention as the per-phase
    # renderer): ONLY the UTC timestamp + % coverage, no honesty caption.
    footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=13,
                           ha="left", va="bottom", family="monospace")

    # --- persistent whole-mission rasters. best_gsd_m starts at +inf so the
    # FIRST observation of any texel always "wins" (matches the per-phase
    # renderer's first-touch behavior); later frames only overwrite when
    # STRICTLY finer (highest-resolution-wins, not most-recent-wins). ---
    rgba_img = np.empty((H, W, 3), dtype=np.uint8)
    rgba_img[:, :] = neutral_u8
    touched_ever = np.zeros((H, W), dtype=bool)
    best_gsd_m = np.full((H, W), np.inf, dtype=np.float64)
    facet_touched = np.zeros(n_faces, dtype=bool)
    lat_row_centers = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    row_weight = np.cos(np.radians(lat_row_centers))
    total_weight = row_weight.sum() * W

    rgba_flat = rgba_img.reshape(-1, 3)
    touched_flat = touched_ever.reshape(-1)
    best_gsd_flat = best_gsd_m.reshape(-1)

    tmpdir = Path(tempfile.mkdtemp(prefix="ema_coverage_fullperiod_frames_"))
    try:
        for i, (fp, utc, pts_i) in enumerate(zip(fits_files, utcs, rotated_body_pts)):
            R_i = R_list[i]
            with fits.open(fp) as hdul:
                face_idx = np.asarray(hdul["FACE_IDX"].data)
                sx = np.asarray(hdul["SURF_X_KM"].data, dtype=np.float64)
                sy = np.asarray(hdul["SURF_Y_KM"].data, dtype=np.float64)
                sz = np.asarray(hdul["SURF_Z_KM"].data, dtype=np.float64)
                emi_ang = np.asarray(hdul["EMI_ANG"].data, dtype=np.float64)
                vals_full = np.asarray(hdul[thermal_ext].data, dtype=float)
                h = hdul[0].header
                camx, camy, camz = h["CAM_X"], h["CAM_Y"], h["CAM_Z"]

            valid = face_idx >= 0
            xs, ys, zs = sx[valid], sy[valid], sz[valid]
            vals = vals_full[valid]
            emi = emi_ang[valid]

            # --- Round 6, Improvement 1: re-point the render camera every
            # frame so the real sub-spacecraft point stays centered. ---
            if track_subspacecraft:
                campos_i, focal_i, up_i = _tracked_camera_pose(
                    (camx, camy, camz), R_i, center, cam_distance, d_vec)
                pl.camera_position = [tuple(campos_i), tuple(focal_i), tuple(up_i)]
                pl.camera.parallel_projection = True
                pl.camera.parallel_scale = half

            # Task's exact per-pixel range + GSD formula (see module note).
            rng_km = np.sqrt((xs - camx) ** 2 + (ys - camy) ** 2 + (zs - camz) ** 2)
            gsd_m = rng_km * 1000.0 * IR3S_IFOV_RAD

            r = np.sqrt(xs * xs + ys * ys + zs * zs)
            lat = np.degrees(np.arcsin(np.clip(zs / r, -1.0, 1.0)))
            lon = np.degrees(np.arctan2(ys, xs))
            col = np.floor((lon + 180.0) / 360.0 * W).astype(np.int64) % W
            row = np.clip(np.floor((90.0 - lat) / 180.0 * H).astype(np.int64), 0, H - 1)

            # --- Footprint-fill (round 5, Fix 1): expand each pixel into
            # every texel of its true (emission-elongated) footprint BEFORE
            # the best-per-texel reduction, so a coarse High/Mid-orbit pixel
            # correctly claims its full ~5-texel span on this fine shared
            # grid instead of a single point (directly closes the round-4
            # flagged 37.6%-vs-81.9% grid-mismatch under-credit). Every
            # footprint texel inherits its SOURCE pixel's own GSD value
            # (the true resolution of that observation), not a
            # distance-to-texel-center recomputation. ---
            rad_row, rad_col = _footprint_fill_radii(rng_km, emi, lat, texel_km)
            flat_exp, src_exp = _footprint_splat_indices(row, col, rad_row, rad_col, H, W)
            gsd_exp = gsd_m[src_exp]

            # Per-frame "best GSD per touched texel" reduction, THEN compare
            # against the running whole-mission best (highest-res-wins).
            uniq_flat, best_pos = _reduce_best_per_texel(flat_exp, gsd_exp)
            frame_best_gsd = gsd_exp[best_pos]
            existing_best = best_gsd_flat[uniq_flat]
            better = frame_best_gsd < existing_best
            sel_flat = uniq_flat[better]
            sel_src = src_exp[best_pos[better]]

            if sel_flat.size:
                colors = (cmap_obj(norm(vals[sel_src]))[:, :3] * 255).astype(np.uint8)
                rgba_flat[sel_flat] = colors
                touched_flat[sel_flat] = True
                best_gsd_flat[sel_flat] = frame_best_gsd[better]

            facet_touched[np.unique(face_idx[valid].astype(int))] = True

            tex = pv.Texture(rgba_img)
            tex.wrap = 1  # REPEAT — antimeridian columns wrap correctly
            actor.texture = tex
            body_mesh.points = pts_i

            # --- Round 6, Improvement 2: green outline of THIS frame's
            # actually-imaged footprint, replaced every frame. ---
            if footprint_line_actor is not None:
                pl.remove_actor(footprint_line_actor)
                footprint_line_actor = None
            if draw_footprint_box:
                hull_bf = _frame_footprint_hull_points(face_idx, sx, sy, sz)
                if hull_bf is not None and len(hull_bf) >= 3:
                    hull_render = _footprint_hull_actor_points(hull_bf, R_i, center, half)
                    line_poly = pv.lines_from_points(hull_render, close=True)
                    footprint_line_actor = pl.add_mesh(
                        line_poly, color=footprint_box_color,
                        line_width=footprint_box_line_width,
                        render_lines_as_tubes=True, lighting=False)

            pl.render()
            frame_rgb = pl.screenshot(return_img=True)

            area_pct_i = 100.0 * float((row_weight[:, None] * touched_ever).sum()) / total_weight
            im.set_data(frame_rgb)
            label = f"{_format_utc_clean(utc)}    {area_pct_i:.2f}% coverage"
            footer_txt.set_text(label)
            out_png = tmpdir / f"frame_{i:04d}.png"
            fig.savefig(out_png, dpi=dpi, facecolor="black")
        plt.close(fig)
        pl.close()

        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        n_facets_touched = int(facet_touched.sum())
        area_pct_final = 100.0 * float((row_weight[:, None] * touched_ever).sum()) / total_weight

        raster_path = None
        if save_raster_npz is not None:
            raster_path = Path(save_raster_npz)
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raster_path,
                best_gsd_m=np.where(touched_ever, best_gsd_m, np.nan).astype(np.float32),
                touched=touched_ever,
                row_weight=row_weight,
                texture_width=W,
                texture_height=H,
                mean_r_km=mean_r,
            )

        return {
            "out_mp4": out_mp4,
            "archived_legacy_mp4": _archived,
            "thermal_ext": thermal_ext,
            "renderer": "pyvista_texture_fullperiod_truegsd",
            "footprint_fill": True,
            "track_subspacecraft": track_subspacecraft,
            "footprint_box": draw_footprint_box,
            "texture_width": W,
            "texture_height": H,
            "texel_km": texel_km,
            "n_texels_touched": int(touched_ever.sum()),
            "n_texels_total": W * H,
            "area_pct": area_pct_final,
            "facet_pct_legacy": 100.0 * n_facets_touched / n_faces,
            "n_facets_touched": n_facets_touched,
            "n_facets_total": n_faces,
            "n_frames": len(fits_files),
            "ifov_rad": IR3S_IFOV_RAD,
            "ifov_check_fov_over_nx_rad": ifov_check,
            "raster_npz": raster_path,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        matplotlib.rcParams["savefig.bbox"] = _prev_bbox


# ==========================================================================
# PARALLEL coverage-accumulation renderer (2026-07-05, round 6 — Ada)
# ==========================================================================
# The serial renderers above (``render_coverage_accumulation_animation_
# texture_truegsd`` and ``..._fullperiod_truegsd``) take multiple tens of
# minutes at N>1000 frames (round 5 flagged: descent 2801 s / 1072 f,
# full-period 3506 s / 1336 f). Measured per-frame cost breakdown at the
# 16384-wide texture (Ada, 2026-07-05): pv.Texture upload+assign ~940 ms,
# accumulation (footprint splat) ~260 ms, matplotlib savefig ~190 ms,
# pl.render ~230 ms, pl.screenshot ~90 ms, FITS read ~40 ms. The render half
# (texture/render/screenshot/savefig, ~1.5 s) is the dominant cost and is
# embarrassingly parallel across frames; only the coverage ACCUMULATION is
# sequential — but it is CHEAP once the per-frame paint operations are
# precomputed (measured replay-by-scatter cost 0.5-16 ms/frame vs. 260 ms to
# recompute from FITS).
#
# IMPORTANT macOS GL FINDING (Ada, 2026-07-05, verified by repeated crash):
# VTK here uses a real Cocoa/NSOpenGL render window (this env has NO OSMesa),
# and creating multiple off-screen GL contexts CONCURRENTLY across processes
# SEGFAULTS in GL context creation (`glGetString` -> KERN_INVALID_ADDRESS in
# vtkCocoaRenderWindow). So the OpenGL render itself CANNOT be parallelized
# across processes on this machine — an initial 4-worker attempt crashed the
# GPU contexts repeatedly. A SINGLE off-screen context is stable (exactly what
# the serial renderers already do). Therefore the render stays single-context
# and the speedup comes from (a) parallelizing everything that is NOT GL and
# (b) removing the round-5 per-frame texture bottleneck.
#
# Three-stage design (keeps every frame's rendered CONTENT byte-identical to
# the serial path — accumulation math, footprint-fill, most-recent-wins /
# highest-resolution-wins combine, camera tracking, green footprint box, AND
# the exact PyVista texture pixels are reproduced EXACTLY; verified array-for-
# array and render-for-render against the serial renderer before use):
#
#   Stage 1 (PARALLEL, no OpenGL — pure numpy/astropy pool): each frame's
#   compact "paint-list" is extracted independently — the unique texels that
#   frame paints, the per-texel source temperature (colorization DEFERRED to
#   stage 2 so this stage needs no global vmin/vmax), the per-texel GSD
#   (full-period only), the camera position, the green-footprint-hull points,
#   the touched facet ids, and the valid-pixel temperatures. Saved to a LOCAL
#   $TMPDIR .npz per frame (~3-5 MB — the footprint only covers the sensor's
#   on-body pixels, bounded regardless of texture width). This is the
#   ~300 ms/frame accumulation work, spread across all cores.
#
#   Stage 2 (SERIAL, single OpenGL context in the PARENT): replays the paint-
#   lists in order (cheap ~ms scatter) to rebuild the exact accumulated raster
#   per frame, then renders body + tracking camera + footprint box and writes
#   the raw RGB screenshot to a .npy. The round-5 bottleneck was `pv.Texture(
#   rgba)` at 16384 (~940 ms/frame: pyvista's Fortran-order flip/reshape of a
#   402 MB array); `_coverage_fast_pv_texture` produces a BYTE-IDENTICAL
#   texture ~3x faster, so the unavoidable single-context GL floor is just
#   render+screenshot (~270 ms/frame) plus the fast texture build.
#
#   Stage 3 (PARALLEL, no OpenGL — matplotlib Agg pool): composites each saved
#   RGB screenshot into the titled/colorbarred/footer'd publication figure and
#   saves the PNG. Pure Agg, safe to parallelize (spawn children never touch
#   GL). Overlaps nothing but is cheap and off the single-context critical path.
#
#   Parent: aggregates Stage-1 vmin/vmax, runs Stage 2, dispatches Stage 3,
#   ffmpeg-assembles the PNGs (size-target CRF fallback so each MP4 lands under
#   a byte ceiling), and returns the coverage statistics straight from Stage
#   2's own final raster (no extra replay) plus the full-period best-GSD raster.
#
# The serial functions are intentionally LEFT UNCHANGED as regression oracles.
# On any single-worker request (n_workers<=1) the callers fall back to them.


def _coverage_colorize(vals, cmap_obj, norm):
    """RGB uint8 for a temperature array — the single colorization used
    identically by the serial renderers and the parallel replay
    (``(cmap(norm(vals))[:, :3] * 255).astype(uint8)``)."""
    return (cmap_obj(norm(vals))[:, :3] * 255).astype(np.uint8)


def _coverage_extract_one(fp, W, H, texel_km, thermal_ext, mode, draw_footprint_box):
    """Compute one frame's compact paint-list — the EXACT set of texel writes
    the serial renderer would perform for this frame, deferred into a compact,
    order-independent form (see the module note above). ``mode`` is
    ``"recent"`` (per-phase most-recent-wins) or ``"fullperiod"`` (highest-
    resolution-wins). Returns a dict of numpy arrays + scalars; NO OpenGL."""
    with fits.open(fp) as hdul:
        face_idx = np.asarray(hdul["FACE_IDX"].data)
        sx = np.asarray(hdul["SURF_X_KM"].data, dtype=np.float64)
        sy = np.asarray(hdul["SURF_Y_KM"].data, dtype=np.float64)
        sz = np.asarray(hdul["SURF_Z_KM"].data, dtype=np.float64)
        emi_ang = np.asarray(hdul["EMI_ANG"].data, dtype=np.float64)
        vals_full = np.asarray(hdul[thermal_ext].data, dtype=float)
        h0 = hdul[0].header
        camx, camy, camz = h0["CAM_X"], h0["CAM_Y"], h0["CAM_Z"]
        utc = h0["UTC_TIME"]

    valid = face_idx >= 0
    n_valid = int(valid.sum())
    xs, ys, zs = sx[valid], sy[valid], sz[valid]
    vals = vals_full[valid]
    emi = emi_ang[valid]

    # --- identical per-pixel lat/lon + footprint-fill splat as the serial
    # renderer (see render_coverage_accumulation_animation_texture_truegsd). ---
    r = np.sqrt(xs * xs + ys * ys + zs * zs)
    lat = np.degrees(np.arcsin(np.clip(zs / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(ys, xs))
    col = np.floor((lon + 180.0) / 360.0 * W).astype(np.int64) % W
    row = np.clip(np.floor((90.0 - lat) / 180.0 * H).astype(np.int64), 0, H - 1)
    rng_km = np.sqrt((xs - camx) ** 2 + (ys - camy) ** 2 + (zs - camz) ** 2)
    rad_row, rad_col = _footprint_fill_radii(rng_km, emi, lat, texel_km)
    flat_exp, src_exp = _footprint_splat_indices(row, col, rad_row, rad_col, H, W)

    if mode == "fullperiod":
        gsd_m = rng_km * 1000.0 * IR3S_IFOV_RAD
        gsd_exp = gsd_m[src_exp]
        # per-frame best GSD per touched texel (same reduction as serial).
        uniq_flat, best_pos = _reduce_best_per_texel(flat_exp, gsd_exp)
        flat_store = uniq_flat.astype(np.int64)
        val_store = vals[src_exp[best_pos]].astype(np.float64)
        gsd_store = gsd_exp[best_pos].astype(np.float64)
    else:
        # most-recent-wins: reproduce numpy fancy-assign's last-write-wins by
        # keeping, per unique texel, the value of the LAST occurrence in
        # flat_exp order (byte-identical to rgba_flat[flat_exp]=colors[src_exp]).
        n = flat_exp.size
        if n:
            uniq, first_in_rev = np.unique(flat_exp[::-1], return_index=True)
            pos_last = n - 1 - first_in_rev
            flat_store = uniq.astype(np.int64)
            val_store = vals[src_exp[pos_last]].astype(np.float64)
        else:
            flat_store = np.empty(0, dtype=np.int64)
            val_store = np.empty(0, dtype=np.float64)
        gsd_store = None

    face_unique = np.unique(face_idx[valid].astype(np.int64)) if n_valid else np.empty(0, np.int64)

    hull_bf = None
    if draw_footprint_box:
        hb = _frame_footprint_hull_points(face_idx, sx, sy, sz)
        if hb is not None and len(hb) >= 3:
            hull_bf = np.asarray(hb, dtype=np.float64)

    return dict(
        flat=flat_store, val=val_store, gsd=gsd_store,
        cam=np.array([camx, camy, camz], dtype=np.float64), utc=str(utc),
        hull_bf=hull_bf, face_unique=face_unique, n_valid=n_valid,
        valid_T=vals.astype(np.float32),
    )


def _coverage_extract_worker(a):
    """Stage-1 pool worker: extract one frame's paint-list and save it to a
    local .npz; return a light summary (index, tmin, tmax, n_valid, utc)."""
    (idx, fp, W, H, texel_km, thermal_ext, mode, draw_box, npz_path) = a
    d = _coverage_extract_one(fp, W, H, texel_km, thermal_ext, mode, draw_box)
    save = dict(
        flat=d["flat"], val=d["val"],
        cam=d["cam"], face_unique=d["face_unique"],
        valid_T=d["valid_T"],
        has_gsd=np.array([d["gsd"] is not None]),
        has_hull=np.array([d["hull_bf"] is not None]),
    )
    save["gsd"] = d["gsd"] if d["gsd"] is not None else np.zeros(0, np.float64)
    save["hull_bf"] = d["hull_bf"] if d["hull_bf"] is not None else np.zeros((0, 3), np.float64)
    np.savez(npz_path, **save)
    vt = d["valid_T"]
    tmin = float(vt.min()) if vt.size else np.inf
    tmax = float(vt.max()) if vt.size else -np.inf
    return (idx, tmin, tmax, d["n_valid"], d["utc"])


def _coverage_apply_paintlist(z, mode, rgba_flat, touched_flat, best_gsd_flat,
                               cmap_obj, norm):
    """Apply one loaded paint-list ``z`` (an npz mapping) to the running
    rasters, using the SAME combine rule as the serial renderer for ``mode``.
    Returns the set of touched facet ids for this frame (for facet stats)."""
    flat = z["flat"]
    if flat.size:
        val = z["val"]
        if mode == "fullperiod":
            gsd = z["gsd"]
            existing = best_gsd_flat[flat]
            better = gsd < existing
            if better.any():
                sel = flat[better]
                rgba_flat[sel] = _coverage_colorize(val[better], cmap_obj, norm)
                touched_flat[sel] = True
                best_gsd_flat[sel] = gsd[better]
        else:
            rgba_flat[flat] = _coverage_colorize(val, cmap_obj, norm)
            touched_flat[flat] = True
    return z["face_unique"]


def _coverage_fast_pv_texture(rgba_img):
    """Build a ``pv.Texture`` from an ``(H, W, 3)`` uint8 raster that is
    BYTE-IDENTICAL to ``pv.Texture(rgba_img); .wrap=1`` but ~3x faster at the
    16384 texture size. ``pv.Texture(ndarray)`` internally does
    ``np.flip(image.swapaxes(0, 1), axis=1).reshape((-1, 3), order='F')``
    (pyvista/plotting/texture.py ``_from_array``) — a Fortran-order reshape of
    a transposed+flipped VIEW, which materializes the whole 402 MB array with
    cache-hostile strides (~940 ms/frame, the round-5 bottleneck). The
    contiguous ``np.flip(rgba, axis=0).reshape(-1, 3)`` produces the exact same
    scalar array (verified ``array_equal``), so feeding it as a pre-built
    ``ImageData`` takes pyvista's fast ``_from_image_data`` path with identical
    texture settings → pixel-for-pixel identical renders (verified) at ~3x."""
    H, W = rgba_img.shape[0], rgba_img.shape[1]
    grid = pv.ImageData(dimensions=(W, H, 1))
    grid.point_data["Image"] = np.ascontiguousarray(
        np.flip(rgba_img, axis=0).reshape(-1, 3))
    grid.set_active_scalars("Image")
    tex = pv.Texture(grid)
    tex.wrap = 1  # REPEAT — same as the serial renderer's tex.wrap = 1
    return tex


def _coverage_serial_gl_pass(cfg):
    """Stage 2 — SINGLE off-screen PyVista context (run in the PARENT; the
    only stable option on this Mac: concurrent Cocoa/NSOpenGL contexts across
    processes crash in GL context creation — verified). Replays the Stage-1
    paint-lists IN ORDER (cheap scatter) to reconstruct the exact accumulated
    raster at every frame, renders the textured body + tracking camera + green
    footprint box with the fast byte-identical texture, and writes each frame's
    raw RGB screenshot to ``frame_rgb_dir/rgb_<i>.npy`` for the parallel
    savefig stage. Returns ``(touched2d, facet_touched, best_gsd_flat,
    n_texels_touched, frame_meta)`` where ``frame_meta`` is the ordered list of
    ``(i, utc, area_pct)`` — so the parent needs no separate replay for the
    returned statistics or the footer labels."""
    import spiceypy as sp

    W, H = cfg["W"], cfg["H"]
    mode = cfg["mode"]
    npz_dir = Path(cfg["npz_dir"])
    rgb_dir = Path(cfg["frame_rgb_dir"])
    utcs = cfg["utcs"]
    n_frames = len(utcs)
    center = np.asarray(cfg["center"], dtype=np.float64)
    half = cfg["half"]
    cam_distance = cfg["cam_distance"]
    neutral_u8 = np.asarray(cfg["neutral_u8"], dtype=np.uint8)
    cmap_obj = cm.get_cmap(cfg["cmap"])
    norm = matplotlib.colors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])

    rgba_img = np.empty((H, W, 3), dtype=np.uint8)
    rgba_img[:, :] = neutral_u8
    rgba_flat = rgba_img.reshape(-1, 3)
    touched_ever = np.zeros((H, W), dtype=bool)
    touched_flat = touched_ever.reshape(-1)
    best_gsd_flat = (np.full(H * W, np.inf, dtype=np.float64)
                     if mode == "fullperiod" else None)
    facet_touched = np.zeros(cfg["n_faces"], dtype=bool)
    lat_row_centers = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    row_weight = np.cos(np.radians(lat_row_centers))
    total_weight = row_weight.sum() * W

    sp.furnsh(str(cfg["metakernel"]))
    et0 = sp.str2et(utcs[0])
    R0 = sp.pxform("J2000", cfg["naif_frame"], et0)
    R_list = [R0 @ sp.pxform(cfg["naif_frame"], "J2000", sp.str2et(u)) for u in utcs]
    sp.unload(str(cfg["metakernel"]))

    body_pts_unwelded = np.asarray(cfg["body_pts_unwelded"], dtype=np.float64)
    body_faces_vtk = np.asarray(cfg["body_faces_vtk"])
    body_uv = np.asarray(cfg["body_uv"], dtype=np.float64)
    d_vec = np.array([
        np.cos(np.radians(cfg["elev"])) * np.cos(np.radians(cfg["azim"])),
        np.cos(np.radians(cfg["elev"])) * np.sin(np.radians(cfg["azim"])),
        np.sin(np.radians(cfg["elev"])),
    ])
    campos0 = center + d_vec * cam_distance
    pv_win_w = max(64, int(round(0.83 * cfg["target_width_px"])))
    pv_win_h = max(64, int(round(0.90 * cfg["target_height_px"])))

    body_mesh = pv.PolyData(body_pts_unwelded.copy(), body_faces_vtk)
    body_mesh.active_texture_coordinates = body_uv
    pl = pv.Plotter(off_screen=True, window_size=[pv_win_w, pv_win_h])
    pl.set_background("black")
    actor = pl.add_mesh(body_mesh, texture=pv.Texture(np.zeros((2, 2, 3), dtype=np.uint8)),
                         lighting=False)
    pl.camera_position = [tuple(campos0), tuple(center), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = half
    footprint_line_actor = None

    frame_meta = []
    for i in range(n_frames):
        with np.load(npz_dir / f"pl_{i:05d}.npz") as z:
            fu = _coverage_apply_paintlist(z, mode, rgba_flat, touched_flat,
                                           best_gsd_flat, cmap_obj, norm)
            cam = z["cam"]
            hull_bf = z["hull_bf"] if bool(z["has_hull"][0]) else None
        if fu.size:
            facet_touched[fu] = True
        R_i = R_list[i]
        body_mesh.points = body_pts_unwelded @ R_i.T

        if cfg["track_subspacecraft"]:
            campos_i, focal_i, up_i = _tracked_camera_pose(
                tuple(cam), R_i, center, cam_distance, d_vec)
            pl.camera_position = [tuple(campos_i), tuple(focal_i), tuple(up_i)]
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = half

        actor.texture = _coverage_fast_pv_texture(rgba_img)

        if footprint_line_actor is not None:
            pl.remove_actor(footprint_line_actor)
            footprint_line_actor = None
        if cfg["draw_footprint_box"] and hull_bf is not None and len(hull_bf) >= 3:
            hull_render = _footprint_hull_actor_points(hull_bf, R_i, center, half)
            line_poly = pv.lines_from_points(hull_render, close=True)
            footprint_line_actor = pl.add_mesh(
                line_poly, color=cfg["footprint_box_color"],
                line_width=cfg["footprint_box_line_width"],
                render_lines_as_tubes=True, lighting=False)

        pl.render()
        frame_rgb = pl.screenshot(return_img=True)
        np.save(rgb_dir / f"rgb_{i:05d}.npy", frame_rgb)
        area_pct_i = 100.0 * float((row_weight[:, None] * touched_ever).sum()) / total_weight
        frame_meta.append((i, utcs[i], area_pct_i))

    pl.close()
    return (touched_ever, facet_touched, best_gsd_flat, int(touched_ever.sum()),
            frame_meta)


def _coverage_savefig_worker(args):
    """Stage 3 — parallel matplotlib (Agg, NO OpenGL, safe to parallelize):
    composite each frame's saved RGB screenshot into the titled/colorbarred/
    footer'd publication figure and save a PNG. One figure per worker,
    reused across its assigned frames."""
    frames = args["frames"]           # list of (i, utc, area_pct)
    cfg = args["cfg"]
    rgb_dir = Path(cfg["frame_rgb_dir"])
    png_dir = Path(cfg["frame_dir"])
    dpi = cfg["dpi"]
    pv_win_w = max(64, int(round(0.83 * cfg["target_width_px"])))
    pv_win_h = max(64, int(round(0.90 * cfg["target_height_px"])))
    cmap_obj = cm.get_cmap(cfg["cmap"])
    norm = matplotlib.colors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
    caption = cfg.get("caption")

    setup_style()
    _prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    fig = plt.figure(figsize=(cfg["target_width_px"] / dpi, cfg["target_height_px"] / dpi),
                     dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 0.83, 0.92])
    ax.set_facecolor("black")
    ax.set_axis_off()
    im = ax.imshow(np.zeros((pv_win_h, pv_win_w, 3), dtype=np.uint8), aspect="auto")
    cax = fig.add_axes([0.86, 0.12, 0.022, 0.68])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=cax)
    cbar.set_label(cfg["cbar_label"], color="white", fontsize=14)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.suptitle(cfg["title"], color="white", fontsize=15, y=0.97)
    footer_fontsize = 11.0 if caption else 13
    footer_txt = fig.text(0.02, 0.02, "", color="white", fontsize=footer_fontsize,
                           ha="left", va="bottom", family="monospace")

    n = 0
    for (i, utc, area_pct_i) in frames:
        frame_rgb = np.load(rgb_dir / f"rgb_{i:05d}.npy")
        im.set_data(frame_rgb)
        label = f"{_format_utc_clean(utc)}    {area_pct_i:.2f}% coverage"
        if caption:
            label += f"\n{caption}"
        footer_txt.set_text(label)
        fig.savefig(png_dir / f"frame_{i:04d}.png", dpi=dpi, facecolor="black")
        n += 1
    plt.close(fig)
    matplotlib.rcParams["savefig.bbox"] = _prev_bbox
    return n


def _coverage_encode_mp4(frame_dir, fps, out_mp4, crf, max_mb=None, crf_step=3,
                          crf_cap=32):
    """ffmpeg-assemble ``frame_dir/frame_%04d.png`` into ``out_mp4`` (same
    libx264/yuv420p/faststart settings as the serial renderers). If
    ``max_mb`` is set and the encode exceeds it, RE-ENCODE from the retained
    PNGs at a higher CRF (cheap — no re-render) until it fits or ``crf_cap``
    is reached. Returns ``(final_crf, size_mb)``."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cur = crf
    while True:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(Path(frame_dir) / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(cur),
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        size_mb = out_mp4.stat().st_size / 1e6
        if max_mb is None or size_mb <= max_mb or cur >= crf_cap:
            return cur, size_mb
        cur = min(cur + crf_step, crf_cap)


def _coverage_parallel_core(
    fits_files, mesh_path, out_mp4, mode, *,
    naif_frame, metakernel, body_label, title, cbar_label, cmap, neutral_rgba,
    vmin, vmax, elev, azim, fps, target_width_px, target_height_px, crf, dpi,
    thermal_ext, texture_width, caption, archive_existing, archive_suffix,
    save_raster_npz, track_subspacecraft, draw_footprint_box,
    footprint_box_color, footprint_box_line_width, n_workers, max_mb,
):
    """Shared parallel core for both coverage renderers. Produces a dict of
    the same shape as the serial renderers plus ``n_workers``/``final_crf``/
    ``final_size_mb``/``wall_*`` timing keys."""
    import multiprocessing as mp
    import time as _time

    if fits is None:
        raise RuntimeError("astropy is required (pip install astropy)")
    if pv is None:
        raise RuntimeError("pyvista is required (pip install pyvista)")
    if draw_footprint_box and ConvexHull is None:
        raise RuntimeError("scipy is required for draw_footprint_box=True")

    if not fits_files:
        raise RuntimeError("no input frame_*.fits resolved for the parallel render")

    with fits.open(fits_files[0]) as hdul0:
        names0 = {h.name for h in hdul0}
    missing = [k for k in ("SURF_X_KM", "SURF_Y_KM", "SURF_Z_KM") if k not in names0]
    if missing:
        raise RuntimeError(
            f"{fits_files[0]} is missing {missing} — the SURF-writing "
            "ptm_camera_sim binary (md5 25866a03673bbf743f635be1b51fca13+) is required.")
    if thermal_ext is None:
        thermal_ext = _pick_thermal_ext(fits_files)

    verts, faces = read_mesh_verts_faces(mesh_path)
    n_faces = len(faces)
    body_pts_unwelded, body_faces_vtk, body_uv = _build_equirect_uv_mesh(verts, faces)

    W = int(texture_width)
    H = W // 2
    mean_r = float(np.mean(np.linalg.norm(verts, axis=1)))
    texel_km = (2.0 * np.pi * mean_r) / W

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    half = (bbox_max - bbox_min).max() / 2.0 * 0.8
    cam_distance = half * 6
    neutral_u8 = (np.array(neutral_rgba[:3]) * 255).astype(np.uint8)

    n_frames = len(fits_files)
    ctx = mp.get_context("spawn")
    npz_dir = Path(tempfile.mkdtemp(prefix="ema_coverage_paintlists_"))
    frame_dir = Path(tempfile.mkdtemp(prefix="ema_coverage_parframes_"))

    t_all0 = _time.time()
    try:
        # ---- Stage 1: parallel paint-list extraction (no OpenGL). ----
        t0 = _time.time()
        ex_args = [
            (i, fp, W, H, texel_km, thermal_ext, mode, draw_footprint_box,
             str(npz_dir / f"pl_{i:05d}.npz"))
            for i, fp in enumerate(fits_files)
        ]
        summaries = [None] * n_frames
        with ctx.Pool(processes=n_workers) as pool:
            for (idx, tmin, tmax, nv, utc) in pool.imap_unordered(
                    _coverage_extract_worker, ex_args, chunksize=1):
                summaries[idx] = (tmin, tmax, nv, utc)
        wall_extract = _time.time() - t0

        utcs = [summaries[i][3] for i in range(n_frames)]
        n_px_total = int(sum(summaries[i][2] for i in range(n_frames)))

        # ---- global vmin/vmax (matches the serial rule for each mode). ----
        if vmin is None or vmax is None:
            tmin_all = min(summaries[i][0] for i in range(n_frames))
            if vmin is None:
                vmin = 5.0 * np.floor(tmin_all / 5.0)
            if vmax is None:
                if mode == "fullperiod":
                    allT = np.concatenate([
                        np.load(npz_dir / f"pl_{i:05d}.npz")["valid_T"]
                        for i in range(n_frames)])
                    vmax = 5.0 * np.ceil(float(np.percentile(allT, 99.5)) / 5.0)
                    del allT
                else:
                    tmax_all = max(summaries[i][1] for i in range(n_frames))
                    vmax = 5.0 * np.ceil(tmax_all / 5.0)

        # ---- shared config for the GL pass + the savefig pool. ----
        rgb_dir = Path(tempfile.mkdtemp(prefix="ema_coverage_framergb_"))
        cfg = dict(
            W=W, H=H, n_faces=n_faces, mode=mode, npz_dir=str(npz_dir),
            frame_dir=str(frame_dir), frame_rgb_dir=str(rgb_dir),
            utcs=utcs, center=center.tolist(), half=float(half),
            cam_distance=float(cam_distance), neutral_u8=neutral_u8.tolist(),
            cmap=cmap, vmin=float(vmin), vmax=float(vmax),
            metakernel=str(metakernel), naif_frame=naif_frame,
            body_pts_unwelded=body_pts_unwelded, body_faces_vtk=body_faces_vtk,
            body_uv=body_uv, elev=elev, azim=azim,
            target_width_px=target_width_px, target_height_px=target_height_px,
            dpi=dpi, cbar_label=cbar_label, title=title, caption=caption,
            track_subspacecraft=track_subspacecraft,
            draw_footprint_box=draw_footprint_box,
            footprint_box_color=footprint_box_color,
            footprint_box_line_width=footprint_box_line_width,
        )

        # ---- Stage 2: SINGLE-context serial GL pass in the PARENT (concurrent
        # Cocoa/GL contexts crash on this Mac — verified; the fast byte-
        # identical texture makes a single context ~3x faster than round 5).
        # Produces per-frame raw RGB screenshots + the final raster/stats. ----
        try:
            t0 = _time.time()
            (touched2d, facet_touched, best_gsd_flat, n_tex_touched, frame_meta) = \
                _coverage_serial_gl_pass(cfg)
            wall_render = _time.time() - t0

            # ---- Stage 3: parallel matplotlib savefig (Agg, no OpenGL — safe
            # to run across processes; spawn children never touch GL). ----
            t0 = _time.time()
            n_eff = max(1, min(n_workers, n_frames))
            bounds = np.linspace(0, n_frames, n_eff + 1).astype(int)
            sf_args = [{"frames": frame_meta[bounds[k]:bounds[k + 1]], "cfg": cfg}
                       for k in range(n_eff) if bounds[k + 1] > bounds[k]]
            with ctx.Pool(processes=len(sf_args)) as pool:
                list(pool.imap_unordered(_coverage_savefig_worker, sf_args, chunksize=1))
            wall_savefig = _time.time() - t0
        finally:
            shutil.rmtree(rgb_dir, ignore_errors=True)

        # ---- archive predecessor + encode with size-target CRF fallback. ----
        _archived = (_archive_existing(out_mp4, suffix=archive_suffix)
                     if archive_existing else None)
        t0 = _time.time()
        final_crf, final_size_mb = _coverage_encode_mp4(
            frame_dir, fps, out_mp4, crf, max_mb=max_mb)
        wall_encode = _time.time() - t0

        # ---- final coverage stats: from the GL pass's own final raster
        # (no separate parent replay needed). ----
        touched = touched2d.reshape(-1)
        best_gsd = best_gsd_flat
        n_facets_touched = int(facet_touched.sum())
        lat_row_centers = 90.0 - (np.arange(H) + 0.5) / H * 180.0
        row_weight = np.cos(np.radians(lat_row_centers))
        total_weight = row_weight.sum() * W
        area_pct_final = 100.0 * float(
            (row_weight[:, None] * touched2d).sum()) / total_weight

        raster_path = None
        if save_raster_npz is not None and mode == "fullperiod":
            raster_path = Path(save_raster_npz)
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            best_gsd_2d = best_gsd.reshape(H, W)
            np.savez_compressed(
                raster_path,
                best_gsd_m=np.where(touched2d, best_gsd_2d, np.nan).astype(np.float32),
                touched=touched2d, row_weight=row_weight,
                texture_width=W, texture_height=H, mean_r_km=mean_r,
            )

        wall_total = _time.time() - t_all0
        result = {
            "out_mp4": Path(out_mp4),
            "archived_legacy_mp4": _archived,
            "thermal_ext": thermal_ext,
            "renderer": ("pyvista_texture_fullperiod_truegsd_parallel"
                         if mode == "fullperiod" else "pyvista_texture_truegsd_parallel"),
            "footprint_fill": True,
            "track_subspacecraft": track_subspacecraft,
            "footprint_box": draw_footprint_box,
            "texture_width": W, "texture_height": H, "texel_km": texel_km,
            "n_texels_touched": int(touched.sum()), "n_texels_total": W * H,
            "area_pct": area_pct_final,
            "facet_pct_legacy": 100.0 * n_facets_touched / n_faces,
            "n_facets_touched": n_facets_touched, "n_facets_total": n_faces,
            "n_pixels_total": n_px_total,
            "n_frames": n_frames,
            "n_workers": n_workers,
            "final_crf": final_crf, "final_size_mb": final_size_mb,
            "target_width_px": target_width_px, "target_height_px": target_height_px,
            "wall_extract_s": wall_extract, "wall_render_s": wall_render,
            "wall_savefig_s": wall_savefig, "wall_encode_s": wall_encode,
            "wall_total_s": wall_total,
        }
        if mode == "fullperiod":
            result["ifov_rad"] = IR3S_IFOV_RAD
            result["raster_npz"] = raster_path
        return result
    finally:
        shutil.rmtree(npz_dir, ignore_errors=True)
        shutil.rmtree(frame_dir, ignore_errors=True)


def render_coverage_accumulation_animation_texture_truegsd_parallel(
    target, camera_subdir, thermal_subdir, naif_frame, metakernel,
    out_mp4=None, body_label=None, title=None,
    cbar_label="Brightness T (K) @ 10.5 µm", cmap="inferno",
    neutral_rgba=(0.55, 0.55, 0.55, 1.0), vmin=None, vmax=None, elev=20, azim=25,
    fps=8, target_width_px=1600, target_height_px=900, crf=20, dpi=200,
    thermal_ext=None, caption=None, texture_width=4096, archive_existing=True,
    track_subspacecraft=True, draw_footprint_box=True, footprint_box_color="lime",
    footprint_box_line_width=5, n_workers=8, max_mb=95.0,
    archive_suffix="_preRound6ParallelReRender_",
):
    """Parallel drop-in for :func:`render_coverage_accumulation_animation_
    texture_truegsd` (per-phase, most-recent-wins). Byte-identical frame
    CONTENT (fast byte-identical texture); faster at the given resolution/CRF.
    See the module note above for the three-stage design."""
    cdir = camera_dir(target) / camera_subdir
    tdir = thermal_dir(target) / thermal_subdir
    if body_label is None:
        body_label = target
    if out_mp4 is None:
        out_mp4 = cdir / f"{target}_coverage_accum_pub.mp4"
    if title is None:
        title = f"{body_label} — accumulating IR coverage (most-recent-imaged)"
    fits_files = sorted(glob.glob(str(cdir / "frame_*.fits")))
    return _coverage_parallel_core(
        fits_files, tdir / "mesh.ply", out_mp4, "recent",
        naif_frame=naif_frame, metakernel=metakernel, body_label=body_label,
        title=title, cbar_label=cbar_label, cmap=cmap, neutral_rgba=neutral_rgba,
        vmin=vmin, vmax=vmax, elev=elev, azim=azim, fps=fps,
        target_width_px=target_width_px, target_height_px=target_height_px,
        crf=crf, dpi=dpi, thermal_ext=thermal_ext, texture_width=texture_width,
        caption=caption, archive_existing=archive_existing,
        archive_suffix=archive_suffix, save_raster_npz=None,
        track_subspacecraft=track_subspacecraft, draw_footprint_box=draw_footprint_box,
        footprint_box_color=footprint_box_color,
        footprint_box_line_width=footprint_box_line_width,
        n_workers=n_workers, max_mb=max_mb)


def render_coverage_accumulation_animation_fullperiod_truegsd_parallel(
    target, phase_keys=("orbit_high", "orbit_mid", "orbit_low", "descent"),
    naif_frame="IAU_JUSTITIA_269", metakernel=None, out_mp4=None,
    body_label=None, title=None, cbar_label="Brightness T (K) @ 10.5 µm",
    cmap="inferno", neutral_rgba=(0.55, 0.55, 0.55, 1.0), vmin=None, vmax=None,
    elev=20, azim=25, fps=8, target_width_px=1600, target_height_px=900,
    crf=20, dpi=200, thermal_ext=None, texture_width=16384, archive_existing=True,
    save_raster_npz=None, track_subspacecraft=True, draw_footprint_box=True,
    footprint_box_color="lime", footprint_box_line_width=5, n_workers=8,
    max_mb=95.0, archive_suffix="_preRound6ParallelReRender_",
):
    """Parallel drop-in for :func:`render_coverage_accumulation_animation_
    fullperiod_truegsd` (all 4 phases, highest-resolution-wins). Byte-
    identical frame CONTENT (fast byte-identical texture). See the module note above."""
    if body_label is None:
        body_label = target
    if metakernel is None:
        metakernel = (
            "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/"
            "data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel_camera.tm")
    if out_mp4 is None:
        out_mp4 = (camera_dir(target) / "proxops" / "fullperiod"
                   / f"{target}_fullperiod_coverage_accum_pub.mp4")
    if title is None:
        title = f"{body_label} — full proximity-ops campaign"

    fits_files = []
    for pk in phase_keys:
        ph = JUSTITIA_PROXOPS_PHASES[pk]
        pf = _sorted_fits(camera_dir(target) / "proxops" / ph["camera_subdir"])
        if not pf:
            raise RuntimeError(f"no frame_*.fits for phase {pk}")
        fits_files.extend(pf)
    first_ph = JUSTITIA_PROXOPS_PHASES[phase_keys[0]]
    first_thermal_subdir = (
        f"proxops/{first_ph['thermal_subdir']}" if first_ph["thermal_subdir"] else "proxops")
    mesh_path = thermal_dir(target) / first_thermal_subdir / "mesh.ply"

    return _coverage_parallel_core(
        fits_files, mesh_path, out_mp4, "fullperiod",
        naif_frame=naif_frame, metakernel=metakernel, body_label=body_label,
        title=title, cbar_label=cbar_label, cmap=cmap, neutral_rgba=neutral_rgba,
        vmin=vmin, vmax=vmax, elev=elev, azim=azim, fps=fps,
        target_width_px=target_width_px, target_height_px=target_height_px,
        crf=crf, dpi=dpi, thermal_ext=thermal_ext, texture_width=texture_width,
        caption=None, archive_existing=archive_existing,
        archive_suffix=archive_suffix, save_raster_npz=save_raster_npz,
        track_subspacecraft=track_subspacecraft, draw_footprint_box=draw_footprint_box,
        footprint_box_color=footprint_box_color,
        footprint_box_line_width=footprint_box_line_width,
        n_workers=n_workers, max_mb=max_mb)


def make_justitia_proxops_fullperiod_animation(parallel_workers=8,
                                               target_width_px=1600,
                                               target_height_px=900,
                                               crf=20, max_mb=95.0):
    """Single coverage-accumulation animation spanning the ENTIRE proxops
    imaging campaign (all 4 phases, mission-time order, highest-resolution-
    wins per texel) — see
    :func:`render_coverage_accumulation_animation_fullperiod_truegsd`.

    Round 6 (Ada): dispatches to the PARALLEL two-stage renderer when
    `parallel_workers>1` (byte-identical frame content, faster; 1600x900 with
    a size-target CRF fallback so the MP4 lands < ~95 MB — the round-5 serial
    1920x1080 product was 264 MB), else the serial oracle."""
    target = "269justitia"
    body_label = "(269) Justitia"
    naif_frame = "IAU_JUSTITIA_269"
    metakernel = (
        "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/"
        "data/horizons_cache/bodies/269_Justitia_(A887_SA)/metakernel_camera.tm"
    )
    out_dir = camera_dir(target) / "proxops" / "fullperiod"
    common = dict(
        naif_frame=naif_frame, metakernel=metakernel,
        out_mp4=out_dir / "justitia_fullperiod_coverage_accum_pub.mp4",
        body_label=body_label,
        title=f"{body_label} — full proximity-ops campaign ({IR_CAM_LABEL})",
        texture_width=16384,
        save_raster_npz=out_dir / "fullperiod_coverage_raster_truegsd.npz",
    )
    if parallel_workers and parallel_workers > 1:
        return render_coverage_accumulation_animation_fullperiod_truegsd_parallel(
            target, n_workers=parallel_workers,
            target_width_px=target_width_px, target_height_px=target_height_px,
            crf=crf, max_mb=max_mb, **common)
    return render_coverage_accumulation_animation_fullperiod_truegsd(
        target, **common)


def fig_coverage_resolution_histogram(raster_npz, out_stub,
                                       body_label="(269) Justitia",
                                       n_bins=14, footnote=None):
    """Conference science figure: % of TOTAL surface area covered vs. the
    BEST (finest) achieved ground-sample-distance, from the full-proxops-
    period highest-resolution-wins accumulation
    (:func:`render_coverage_accumulation_animation_fullperiod_truegsd`).

    Differential bars = surface-area fraction per log-spaced GSD bin;
    overlaid line = cumulative "% of surface imaged at resolution <= X".
    Both use the SAME latitude-weighted mean-sphere area convention as the
    animations' own ``area_pct`` (an equirectangular-projection
    approximation of the real, irregular DAMIT body — see the coverage-
    animation module notes) for direct numeric consistency with the
    reported per-phase/full-period coverage percentages. Never-imaged
    surface is reported in the footnote, not silently folded into the bars.
    """
    data = np.load(raster_npz)
    best_gsd = data["best_gsd_m"]  # (H, W) float32, NaN where never imaged
    touched = data["touched"]      # (H, W) bool
    row_weight = data["row_weight"]  # (H,)
    H, W = touched.shape
    total_weight = row_weight.sum() * W

    gsd_touched = best_gsd[touched].astype(float)
    w_touched = np.broadcast_to(row_weight[:, None], (H, W))[touched]

    lo, hi = float(np.nanmin(gsd_touched)), float(np.nanmax(gsd_touched))
    bins = np.logspace(np.log10(lo * 0.95), np.log10(hi * 1.05), n_bins + 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean, for log bins

    hist_w, _ = np.histogram(gsd_touched, bins=bins, weights=w_touched)
    diff_pct = 100.0 * hist_w / total_weight       # differential: % of TOTAL surface
    cum_pct = np.cumsum(diff_pct)                   # cumulative: resolution <= bin edge

    total_imaged_pct = 100.0 * float(w_touched.sum()) / total_weight
    never_imaged_pct = 100.0 - total_imaged_pct

    setup_style()
    fig, ax = planetary.subplots(figsize=(8.0, 5.6))
    widths = np.diff(bins)
    ax.bar(bin_centers, diff_pct, width=widths, align="center",
           color="#3778bf", alpha=0.75, edgecolor="white", linewidth=0.4,
           label="% of surface area (differential)", zorder=2)
    ax.plot(bin_centers, cum_pct, "-o", color="#e08283", ms=5, lw=2.2,
            label="cumulative: % imaged at resolution ≤ X", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Spatial resolution, GSD (m/px)")
    ax.set_ylabel("% of total surface area")
    ax.set_title(f"{body_label} — {IR_CAM_LABEL} coverage vs. spatial resolution\n"
                 "(full proximity-ops campaign, highest-resolution-wins)")
    ax.grid(alpha=0.25, which="both", linewidth=0.6)
    ax.legend(loc="upper left", frameon=True)

    note = (f"{total_imaged_pct:.1f}% of surface imaged overall "
            f"({never_imaged_pct:.1f}% never imaged — not shown in bars)")
    if footnote:
        note = footnote + "\n" + note
    _add_footnote(fig, note)

    stats = {
        "bins_m": bins, "bin_centers_m": bin_centers,
        "diff_pct": diff_pct, "cum_pct": cum_pct,
        "total_imaged_pct": total_imaged_pct, "never_imaged_pct": never_imaged_pct,
    }
    return save_pub(fig, out_stub), fig, stats


def make_justitia_proxops_resolution_histogram(raster_npz=None):
    """Build the coverage-vs-resolution histogram figure from the
    full-period accumulation's saved raster (regenerating it first via
    :func:`make_justitia_proxops_fullperiod_animation` if ``raster_npz`` is
    not given / doesn't exist yet)."""
    target = "269justitia"
    out_dir = camera_dir(target) / "proxops" / "fullperiod"
    if raster_npz is None:
        raster_npz = out_dir / "fullperiod_coverage_raster_truegsd.npz"
    if not Path(raster_npz).exists():
        make_justitia_proxops_fullperiod_animation()
    # Kept to ONE short line (2026-07-05, round 4): an earlier 2-line
    # version (full methodology sentence + coverage stat) visually
    # overlapped the x-axis label at this figure's aspect ratio — the
    # methodology (GSD formula, 9.73 m/texel binning ceiling, 264-frame/
    # 4-phase mission span) belongs in the paper caption/memory, not baked
    # into the slide; `fig_coverage_resolution_histogram`'s own auto
    # coverage-stat note (imaged/never-imaged %) is left as the only footnote.
    return fig_coverage_resolution_histogram(
        raster_npz, out_dir / "coverage_resolution_histogram",
        body_label="(269) Justitia")


def make_chimaera_diurnal_animation():
    """Regenerate the Chimaera diurnal-cycle animation from the CURRENT
    (terminator-fix, binary 79f091e) thermal output — supersedes
    ``chimaera_diurnal_OLD-BUGGY.mp4`` (made on the pre-fix terminator-wedge
    binary; see the module note above :func:`render_diurnal_animation`)."""
    return render_diurnal_animation(
        "623chimaera",
        thermal_subdir="flyby",
        body_label="(623) Chimaera",
        out_mp4=camera_dir("623chimaera") / "diurnal" / "chimaera_diurnal_pub.mp4",
    )


if __name__ == "__main__":
    figs = make_chimaera_static_figures()
    for k, v in figs.items():
        print("FIGURE", k, "->", v)
    anims = make_chimaera_animations()
    for k, v in anims.items():
        print("ANIM", k, "->", v)
    diurnal = make_chimaera_diurnal_animation()
    print("ANIM diurnal ->", diurnal)
