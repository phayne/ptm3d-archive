#!/usr/bin/env python3
"""Moon3D year-long campaign — annual mean/max temperature & volatile-stability
figure set (Paul's overnight mandate, 2026-07-20/21, executed by Luna).

Inputs (see SESSION-NOTES-2026-07-21-moon3d-yearlong-figures.md for full
provenance/verification trail):
  outputs/moon3d_native_yearlong/{coupled,radoff}/statistics/            ANNUAL
      window (steps 1-1484, the full 365.25-day run; merged by the campaign's
      own ptm_merge_chunks, re-verified byte-identical here by an independent
      Python re-implementation of the merge).
  outputs/moon3d_native_yearlong/{coupled,radoff}/first_lunation/statistics/
      FIRST-LUNATION window (steps 1-140, i.e. chunk_000's real steps 1-20
      plus a supplemental 120-step continuation run from the chunk_000
      checkpoint -- NOT part of the original completed campaign; launched by
      Luna to fill a genuine gap, see caption below and the session note).

Cadence: both windows are the SAME 120-samples/lunation cadence (the merged
per-face statistics are single-window min/avg/max/sum/count aggregates over
each window's *entire* step range -- there is no per-timestep archive in
either run, so no true time-resolved seasonal series exists; see
fig_seasonal_psr_floor.BLOCKED.md).

No masking (Paul's standing instruction for this deliverable): the small
radiative-coupling hot-face population (ΣF<=1 corrected VF, still ~0.2% of
faces > 400 K in the coupled run, absent in radoff) is shown as-is; color
limits use robust (1st/99th percentile) bounds with an extended colorbar so
the rare outliers don't wash out the scale, but no data is hidden or altered.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv

sys.path.insert(0, "/Users/paha3326/projects/ptm/src/python")
sys.path.insert(0, "/Users/paha3326/projects/ptm/scripts/yearlong_analysis")
from moon3d_temp_3d import read_ply_vertices_double, tri_faces
from ptm_pubstyle import apply_style, DOUBLE_W, COLORS
from ptm_stats_io import read_f32_array

BASE = Path("/Users/paha3326/projects/moon3d/outputs/moon3d_native_yearlong")
OUT = Path("/Users/paha3326/projects/moon3d/figures_yearlong")
OUT.mkdir(parents=True, exist_ok=True)
MESH_PLY = BASE / "coupled" / "chunk_000" / "mesh.ply"

COLD_THRESH = 110.0   # K, standard lunar cold-trap threshold (project convention)
CO2_THRESH = 54.0     # K, CO2 cold-trap threshold

WINDOWS = {
    "annual": "statistics",
    "first_lunation": "first_lunation/statistics",
}


def load_run(run: str):
    d = {}
    for win, sub in WINDOWS.items():
        s = BASE / run / sub
        d[win] = {
            name: read_f32_array(s / f"{name}_{stat}.bin")
            for name in ["temperature"]
            for stat in ["max", "avg", "min"]
        }
        d[win]["tmax"] = read_f32_array(s / "temperature_max.bin")
        d[win]["tavg"] = read_f32_array(s / "temperature_avg.bin")
        d[win]["tmin"] = read_f32_array(s / "temperature_min.bin")
    return d


def enu_frame(c: np.ndarray):
    up = c / np.linalg.norm(c)
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up])


def top_down_panel(ax, ve, f, vals, cmap, norm, zexag=3.0, zoom=1.85,
                   elev=68.0, azim=-90.0, title="", cbar_label="", extend="neither",
                   title_fontsize=11, title_pad=None, title_xy=None,
                   cbar_shrink=0.62, cbar_pad=0.02, cbar_aspect=18,
                   cbar_label_fontsize=None, cbar_tick_labelsize=None):
    """All style-override kwargs default to values that reproduce the
    original call signature exactly (byte-identical output for any existing
    caller that doesn't pass them) -- added 2026-07-22 for the manuscript-v4
    fig_annual_tmax_map cleanup (panel-title / colorbar-label overlap +
    right-edge colorbar clipping fix), see annual_map_figure_v2().

    title_xy=(x, y), if given (figure-fraction coords), draws the title as a
    free-floating fig.text() at that fixed position instead of ax.set_title().
    Needed because fig.colorbar(ax=ax, ...) SHRINKS `ax` in place to make
    room for itself -- ax.set_title() centers on that post-shrink (narrower)
    axes, so a long title can get squeezed into the colorbar's own column
    once the colorbar takes a big enough bite (found empirically on the
    right-hand panel here, whose longer title got clipped by its own
    colorbar). Pinning the title to a fixed figure-fraction x/y decouples it
    from that per-panel shrink entirely."""
    polys = ve.copy(); polys[:, 2] *= zexag
    tris = polys[f]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fc = sm.to_rgba(vals)
    pc = Poly3DCollection(tris, facecolors=fc, edgecolors="none",
                          antialiased=False, rasterized=True)
    ax.add_collection3d(pc)
    lo = tris.reshape(-1, 3).min(0); hi = tris.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    span = hi - lo
    try:
        ax.set_box_aspect((span[0], span[1], span[2]), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((span[0], span[1], span[2]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title_xy is None:
        title_kw = {} if title_pad is None else {"pad": title_pad}
        ax.set_title(title, fontsize=title_fontsize, **title_kw)
    else:
        tx, ty = title_xy
        ax.figure.text(tx, ty, title, ha="center", va="bottom",
                       fontsize=title_fontsize,
                       fontweight=matplotlib.rcParams.get("axes.titleweight", "bold"))
    cb = ax.figure.colorbar(sm, ax=ax, shrink=cbar_shrink, pad=cbar_pad,
                            aspect=cbar_aspect, extend=extend)
    label_kw = {} if cbar_label_fontsize is None else {"fontsize": cbar_label_fontsize}
    cb.set_label(cbar_label, **label_kw)
    if cbar_tick_labelsize is not None:
        cb.ax.tick_params(labelsize=cbar_tick_labelsize)
    return pc


def annual_map_figure(ve, f, coupled, field: str, label: str, out_stub: str):
    """Main annual map (percentile-limited, unmasked) + small companion delta
    (annual - first_lunation) panel for the same field."""
    apply_style()
    annual = coupled["annual"][field]
    lunation = coupled["first_lunation"][field]
    delta = annual - lunation

    fig = plt.figure(figsize=(DOUBLE_W, 0.52 * DOUBLE_W))
    axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

    vlo, vhi = np.percentile(annual, 1), np.percentile(annual, 99)
    nL = Normalize(vlo, vhi)
    top_down_panel(axL, ve, f, annual, "inferno", nL,
                   title=f"Annual {label} (coupled, full year)",
                   cbar_label=f"{label} (K)", extend="both")

    dmax = max(np.percentile(np.abs(delta), 99), 1.0)
    nR = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    top_down_panel(axR, ve, f, delta, "RdBu_r", nR,
                   title="Annual − first-lunation (same field)",
                   cbar_label=f"Δ{label} (K)", extend="both")

    fig.tight_layout()
    fig.savefig(OUT / f"{out_stub}.png", dpi=250)
    fig.savefig(OUT / f"{out_stub}.pdf")
    plt.close(fig)
    print(f"wrote {out_stub}.png/.pdf  "
          f"(annual {label}: 1-99pct [{vlo:.1f},{vhi:.1f}] K; "
          f"delta 99pct |.|={dmax:.1f} K)")


def annual_map_figure_v2(ve, f, coupled, field: str, label: str, out_stub: str):
    """Manuscript-v4-review cleanup of annual_map_figure() (Paul, 2026-07-22).

    Fixes applied to the ORIGINAL fig_annual_tmax_map, all additive (new
    function, original untouched/still byte-reproducible):
      (a) panel-title / colorbar-label overlap: the original's rotated
          colorbar label (e.g. "...temperature (K)") ran up past the top of
          its own colorbar and directly over the panel title text above it
          (confirmed by pixel inspection, 2026-07-22 -- this IS the AUDIT-
          NOTES P2.8 "colorbar-edge" issue's twin, both symptoms of the same
          cramped-vertical-layout root cause). Root cause: the colorbar's
          rotated label string is long enough that a short colorbar (the
          original's shrink=0.62) forces the label text to extend well past
          the bar's own top/bottom. Fixed with (i) a taller colorbar
          (shrink=0.90) so the label has its own room, (ii) titles pinned to
          a FIXED figure-fraction position via a free-floating fig.text()
          (see title_xy on top_down_panel) instead of ax.set_title(), so the
          title lives in its own reserved margin (subplots_adjust top=0.82)
          entirely above where the colorbar can ever reach, regardless of
          shrink -- this decoupling is what made the fix robust rather than
          a fragile pad/shrink balancing act (three earlier iterations of
          just tuning pad/shrink/fontsize kept trading one overlap for
          another; see wspace/two-line-title note below for the same lesson
          applied to the horizontal axis).
      (b) right-edge colorbar-label clipping (AUDIT-NOTES P2.8, pre-existing,
          confirmed in-file): the shared apply_style() rcParam
          savefig.pad_inches=0.06 is not enough margin for a long rotated
          label sitting at the canvas edge. Fixed with an explicit pad_inches
          on THIS figure's own savefig calls only (does not touch the shared
          rcParam / any other figure) -- kept modest (0.08in, close to the
          shared default) rather than generous, because bloating the saved
          PDF's native page size dilutes fix (c) below (see next note).
      (c) fonts bumped (title 11->14pt, colorbar label 14(rcParam-default,
          accidentally LARGER than its own 11pt title -- part of why it
          visually dominated/overlapped)->12pt, colorbar ticks 12->11pt) so
          labels stay legible once the manuscript places this figure ~10-15%
          smaller. This print-width detail matters quantitatively: the
          manuscript's own \\includegraphics[width=0.95\\linewidth] call
          (Moon3D_Paper_v4_revised.tex) means EFFECTIVE on-page font size is
          proportional to (native pt size / native PDF page width) -- so a
          fontsize bump can be entirely canceled out (or worse) by growing
          the saved canvas, which two earlier iterations did (wspace=0.95 +
          generous pad_inches ballooned the native PDF width from the
          original's 7.73in to over 9in). The two-line titles below +
          pad_inches=0.08 keep the FINAL native width to 7.81in (+1% vs the
          original 7.73in), so the fontsize bump actually reaches the page:
          verified numerically (not just asserted) -- at an assumed 0.95->
          0.81\\linewidth (~15% smaller) placement, effective title size is
          10.1pt vs the original's 9.5pt (+7%), and colorbar-label size is
          8.7pt (comfortably legible for a subordinate label, well below the
          original's inflated/buggy 12.0pt-effective).
      (d) no suptitle -- unchanged from the original (this script has never
          called fig.suptitle here, confirmed by grep); per-panel titles are
          kept per Paul's standing rule.

    GUARD: Tmax-only. The annual TAVG field was adjudicated transient-
    contaminated (AUDIT-NOTES R3-AMENDMENT, 2026-07-21) and pulled from the
    manuscript -- this function refuses any other field so a future call
    site cannot silently reintroduce it here.
    """
    assert field == "tmax", (
        "annual_map_figure_v2 is Tmax-only (annual TAVG is adjudicated "
        f"transient-contaminated and out of scope) -- refused field={field!r}"
    )
    apply_style()
    annual = coupled["annual"][field]
    lunation = coupled["first_lunation"][field]
    delta = annual - lunation

    fig = plt.figure(figsize=(DOUBLE_W, 0.60 * DOUBLE_W))
    axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    # Fully manual layout -- deliberately NOT using fig.tight_layout() here.
    # mplot3d axes + a separately-attached fig.colorbar(ax=...) mini-axes are
    # a known-unreliable combination for matplotlib's automatic layout
    # managers (same "Axes that are not compatible with tight_layout"
    # warning already documented for the indirect-bands overview figure,
    # 2026-07-22 session note) -- confirmed empirically here too: tight_layout
    # fought with an explicit GridSpec wspace and made both the title/title
    # and title/colorbar-label overlaps WORSE on early iterations.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.82, bottom=0.02, wspace=0.5)
    # Snapshot each axes' PRE-colorbar bbox center now -- fig.colorbar(ax=ax)
    # shrinks `ax` in place, and a title pinned to the post-shrink center
    # would drift left of true panel-center (see title_xy note on
    # top_down_panel). Fixed y sits well above the top=0.82 axes boundary so
    # it can't collide with either panel's colorbar regardless of shrink.
    titleY = 0.94
    titleX_L = (axL.get_position().x0 + axL.get_position().x1) / 2.0
    titleX_R = (axR.get_position().x0 + axR.get_position().x1) / 2.0

    # Two-line titles: at a legible print-compensated fontsize (14pt native,
    # see docstring (c)), these long single-line strings need more horizontal
    # gutter than is affordable without bloating the whole canvas (found
    # empirically -- widening wspace enough to fit them on one line inflated
    # the saved PDF's native page width enough to give BACK most of the
    # font-size legibility gain once the manuscript's width-driven
    # \includegraphics[width=0.95\linewidth] (Moon3D_Paper_v4_revised.tex)
    # rescales it -- a WIDER native PDF at the SAME declared width fraction
    # means a SMALLER effective on-page font, not a larger one. Wrapping
    # preserves the exact same words/content, just breaks the line -- a
    # standard, non-lossy fix for a long panel title over a half-width panel.
    titleL_text = f"Annual {label}\n(coupled, full year)"
    titleR_text = "Annual − first-lunation\n(same field)"

    vlo, vhi = np.percentile(annual, 1), np.percentile(annual, 99)
    nL = Normalize(vlo, vhi)
    top_down_panel(axL, ve, f, annual, "inferno", nL,
                   title=titleL_text,
                   cbar_label=f"{label} (K)", extend="both",
                   title_fontsize=14, title_xy=(titleX_L, titleY),
                   # shrink bumped UP from the original 0.62 (not down): the
                   # rotated label string ("...temperature (K)", ~30 chars)
                   # is long enough that a SHORT colorbar forces the label to
                   # extend well past the bar's own top/bottom -- that
                   # overflow, not the title's own position, was the real
                   # source of the original overlap. A taller bar gives the
                   # label enough of its own vertical room to sit inside.
                   cbar_shrink=0.90, cbar_pad=0.05, cbar_aspect=22,
                   cbar_label_fontsize=12, cbar_tick_labelsize=11)

    dmax = max(np.percentile(np.abs(delta), 99), 1.0)
    nR = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    top_down_panel(axR, ve, f, delta, "RdBu_r", nR,
                   title=titleR_text,
                   cbar_label=f"Δ{label} (K)", extend="both",
                   title_fontsize=14, title_xy=(titleX_R, titleY),
                   cbar_shrink=0.90, cbar_pad=0.05, cbar_aspect=22,
                   cbar_label_fontsize=12, cbar_tick_labelsize=11)

    # Explicit, generous pad_inches (rather than the shared apply_style
    # rcParam savefig.pad_inches=0.06, which is not enough margin for a long
    # rotated colorbar label sitting near the canvas edge -- AUDIT-NOTES
    # P2.8's pre-existing right-edge clipping, confirmed in-file by pixel
    # inspection 2026-07-22). Applied only to this figure's own savefig
    # calls; the shared rcParam/other figures are untouched.
    fig.savefig(OUT / f"{out_stub}.png", dpi=250, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{out_stub}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {out_stub}.png/.pdf  "
          f"(annual {label}: 1-99pct [{vlo:.1f},{vhi:.1f}] K; "
          f"delta 99pct |.|={dmax:.1f} K)")


def coldtrap_area_figure(coupled_areas_frac, radoff_areas_frac, thresholds):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.plot(thresholds, 100 * coupled_areas_frac["annual"], "-", lw=2.6,
           color=COLORS["bad"], label="annual max (coupled)")
    ax.plot(thresholds, 100 * coupled_areas_frac["first_lunation"], "--", lw=2.6,
           color=COLORS["bad"], label="first-lunation max (coupled)")
    ax.plot(thresholds, 100 * radoff_areas_frac["annual"], "-", lw=1.8,
           color=COLORS["neutral"], label="annual max (no coupling)")
    ax.plot(thresholds, 100 * radoff_areas_frac["first_lunation"], "--", lw=1.8,
           color=COLORS["neutral"], label="first-lunation max (no coupling)")
    for th in (CO2_THRESH, COLD_THRESH):
        ax.axvline(th, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("$T_{max}$ threshold (K)")
    ax.set_ylabel("area fraction below threshold (%)")
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig_annual_coldtrap_area.png", dpi=250)
    fig.savefig(OUT / "fig_annual_coldtrap_area.pdf")
    plt.close(fig)
    print("wrote fig_annual_coldtrap_area.png/.pdf")


def hist_figure(coupled, radoff):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 0.42 * DOUBLE_W), sharey=True)
    bins = np.linspace(50, 470, 85)
    for ax, d, name in zip(axes, [coupled, radoff], ["coupled", "no coupling (radoff)"]):
        ax.hist(d["annual"]["tmax"], bins=bins, histtype="step", lw=2.0,
               color=COLORS["ptm"], label="year (full 365.25 d)")
        ax.hist(d["first_lunation"]["tmax"], bins=bins, histtype="step", lw=2.0,
               color=COLORS["ref"], label="first lunation (29.5 d)")
        ax.set_yscale("log")
        ax.set_xlabel("$T_{max}$ per face (K)")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("face count")
    axes[0].legend(fontsize=10, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig_year_vs_lunation_hist.png", dpi=250)
    fig.savefig(OUT / "fig_year_vs_lunation_hist.pdf")
    plt.close(fig)
    print("wrote fig_year_vs_lunation_hist.png/.pdf")


def hist_figure_v2(coupled, radoff):
    """Legend-only fix (manuscript-v4 review finding M4, 2026-07-22): the
    "first_lunation" window plotted here is steps 1-140, NOT one true
    lunation. Verified from the run's own real timestamps (chunk_000's step 1
    = 2030-09-01T00:00:00.442; first_lunation_tail's step 120 [[= combined
    step 140]] = 2030-10-05T05:04:26.264): the window spans 34.21 days =
    1.158 lunations (lunation length 29.535040431266847 d, from
    first_lunation_tail/config.yaml's own duration_days) -- NOT the 29.5 d
    the old legend claimed.

    CHOICE: fix the legend text to describe the actual window (do NOT
    truncate the comparison to a true 1-lunation/steps-1-120 window), because
    truncation is not reproducible from existing data: both source runs
    (chunk_000, steps 1-20; first_lunation_tail, steps 21-140) only carry
    pre-aggregated min/avg/max/sum/count STATISTICS bins over their own
    entire step range each -- there is no per-timestep temperature archive
    for either (confirmed by directory listing, 2026-07-22), so a Tmax
    aggregate restricted to just steps 1-120 cannot be recomputed from what
    exists on disk without a new production run (out of scope here). This
    also means `summary_yearlong.json`'s own provenance text ("29.53 d" for
    the first_lunation window) carries the same mislabeling -- flagged here,
    NOT fixed in this pass (out of scope: that file is an already-delivered,
    cited provenance record; a correction there is a separate, deliberate
    edit for Paul/Sagan to green-light, not a silent piggyback on a figure
    legend fix).
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 0.42 * DOUBLE_W), sharey=True)
    bins = np.linspace(50, 470, 85)
    for ax, d, name in zip(axes, [coupled, radoff], ["coupled", "no coupling (radoff)"]):
        ax.hist(d["annual"]["tmax"], bins=bins, histtype="step", lw=2.0,
               color=COLORS["ptm"], label="year (full 365.25 d)")
        ax.hist(d["first_lunation"]["tmax"], bins=bins, histtype="step", lw=2.0,
               color=COLORS["ref"], label="first 34.2 d (1.16 lunations)")
        ax.set_yscale("log")
        ax.set_xlabel("$T_{max}$ per face (K)")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("face count")
    axes[0].legend(fontsize=10, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig_year_vs_lunation_hist_v2.png", dpi=250)
    fig.savefig(OUT / "fig_year_vs_lunation_hist_v2.pdf")
    plt.close(fig)
    print("wrote fig_year_vs_lunation_hist_v2.png/.pdf (legend fix only, finding M4)")


def area_frac_below(tmax, area, thresholds):
    total = area.sum()
    return np.array([area[tmax < th].sum() / total for th in thresholds])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", action="store_true",
                    help="Additionally write the manuscript-v4-review cleaned-up "
                         "fig_annual_tmax_map_v2.{png,pdf} (title/colorbar-label "
                         "overlap + right-edge colorbar clipping fix + legibility "
                         "bump for ~10-15%% smaller print size; Tmax-only, see "
                         "annual_map_figure_v2()). Additive -- default OFF; does "
                         "not change any existing figure's output.")
    ap.add_argument("--only-v2", action="store_true",
                    help="With --v2, skip regenerating the original figures/summary "
                         "JSON entirely (mesh+run load, then ONLY the v2 outputs) -- "
                         "for iterating on the v2 layout without re-touching any of "
                         "the already-delivered original files. No effect without "
                         "--v2.")
    a = ap.parse_args()

    v = read_ply_vertices_double(str(MESH_PLY))
    f = tri_faces(pv.read(str(MESH_PLY)))
    area = np.load(BASE / "_face_area_m2.npy")
    assert len(area) == len(f) == v[f[:, 0]].shape[0]

    c = v.mean(0)
    R = enu_frame(c)
    ve = (R @ (v - c).T).T

    coupled = load_run("coupled")
    radoff = load_run("radoff")

    if a.only_v2:
        assert a.v2, "--only-v2 requires --v2"
        annual_map_figure_v2(ve, f, coupled, "tmax", "maximum surface temperature",
                             "fig_annual_tmax_map_v2")
        hist_figure_v2(coupled, radoff)
        return

    # ---- Figure 1: annual Tmax / Tavg maps + first-lunation delta ----
    annual_map_figure(ve, f, coupled, "tmax", "maximum surface temperature", "fig_annual_tmax_map")
    annual_map_figure(ve, f, coupled, "tavg", "mean surface temperature", "fig_annual_tavg_map")

    if a.v2:
        annual_map_figure_v2(ve, f, coupled, "tmax", "maximum surface temperature",
                             "fig_annual_tmax_map_v2")

    # ---- Figure 3: cold-trap area fraction vs threshold ----
    thresholds = np.linspace(40, 200, 161)
    coupled_frac = {w: area_frac_below(coupled[w]["tmax"], area, thresholds) for w in WINDOWS}
    radoff_frac = {w: area_frac_below(radoff[w]["tmax"], area, thresholds) for w in WINDOWS}
    coldtrap_area_figure(coupled_frac, radoff_frac, thresholds)

    # ---- Figure 4: Tmax histograms, year vs first-lunation ----
    hist_figure(coupled, radoff)
    if a.v2:
        hist_figure_v2(coupled, radoff)

    # ---- Summary JSON ----
    def stats_block(run_d, win):
        tmax, tavg = run_d[win]["tmax"], run_d[win]["tavg"]
        area_tot = area.sum()
        def af(th):
            return float(area[tmax < th].sum() / area_tot)
        def cf(th):
            return float(np.mean(tmax < th))
        return dict(
            median_tmax_K=float(np.median(tmax)), mean_tmax_K=float(tmax.mean()),
            median_tavg_K=float(np.median(tavg)), mean_tavg_K=float(tavg.mean()),
            max_tmax_K=float(tmax.max()), min_tmax_K=float(tmax.min()),
            area_frac_below_110K=af(COLD_THRESH), area_frac_below_54K=af(CO2_THRESH),
            count_frac_below_110K=cf(COLD_THRESH), count_frac_below_54K=cf(CO2_THRESH),
            n_faces_gt_400K=int((tmax > 400).sum()),
        )

    # deep-cold-trap population: defined on FIRST-LUNATION Tmax < 110 K (coupled)
    floor_mask = coupled["first_lunation"]["tmax"] < COLD_THRESH
    n_floor = int(floor_mask.sum())
    floor_annual_tmax = coupled["annual"]["tmax"][floor_mask]
    floor_lunation_tmax = coupled["first_lunation"]["tmax"][floor_mask]
    pct = [10, 50, 90]
    floor_block = dict(
        n_faces=n_floor,
        area_frac_of_mesh=float(area[floor_mask].sum() / area.sum()),
        first_lunation_tmax_percentiles_K={f"p{p}": float(np.percentile(floor_lunation_tmax, p)) for p in pct},
        annual_tmax_percentiles_K={f"p{p}": float(np.percentile(floor_annual_tmax, p)) for p in pct},
        max_swing_at_percentile_K={
            f"p{p}": float(np.percentile(floor_annual_tmax, p) - np.percentile(floor_lunation_tmax, p))
            for p in pct
        },
        NOTE=("2-POINT BRACKET only (annual-max vs first-lunation-max of the SAME "
              "floor population) -- NOT a time-resolved seasonal curve. No per-timestep "
              "archive exists in either run to build fig_seasonal_psr_floor; see "
              "fig_seasonal_psr_floor.BLOCKED.md."),
    )

    summary = dict(
        provenance=dict(
            binary="/Users/paha3326/projects/ptm-bin/ptm_thermal_sim",
            binary_md5="141922269e74424019e436e259c24a26",
            binary_built="2026-07-19T18:27:32-06:00 (predates all Phase-R portability commits, "
                          "which start 2026-07-21T00:41 -- confirmed by absence of Phase-R log "
                          "banners in every run log)",
            merge_binary="/Users/paha3326/dev/heat3d/ptm-build/bin/ptm_merge_chunks (2026-06-23)",
            mesh=str(MESH_PLY), n_faces=int(len(f)),
            vf_file="/Users/paha3326/projects/moon3d/vf/moon3d_c_cartesian_NEW_500k_clean.vf",
            vf_replace_csr="/Users/paha3326/projects/moon3d/vf/moon3d_c_cartesian_NEW_500k_clean_corrected_pp.csr",
        ),
        windows=dict(
            annual="steps 1-1484 (2030-09-01 -> 2031-09-01, 365.25 d, 120 samples/lunation, ORIGINAL completed campaign)",
            first_lunation="steps 1-140 (2030-09-01 -> 2030-10-05, 29.53 d, SAME 120/lunation cadence; "
                            "steps 1-20 = chunk_000 of the original campaign, steps 21-140 = a supplemental "
                            "120-step continuation Luna ran from the chunk_000 checkpoint, same pinned binary/"
                            "batch/env, to fill a gap the original campaign did not produce (see caveats)",
        ),
        cadence_caveat=("Both windows share the SAME 120-samples/lunation cadence (verified: every face has "
                        "exactly 1484 / 140 samples respectively). The separate OneDrive lunation campaign "
                        "(native_{coupled,radoff}, 360/lunation) is NOT used here -- cross-campaign comparison "
                        "was descoped (secondary per the mandate); only the year run's own internal windows "
                        "are compared."),
        coupled=dict(annual=stats_block(coupled, "annual"), first_lunation=stats_block(coupled, "first_lunation")),
        radoff=dict(annual=stats_block(radoff, "annual"), first_lunation=stats_block(radoff, "first_lunation")),
        deep_cold_trap_floor_population=floor_block,
        known_artifact=dict(
            description="Coupled-run-only hot-face population (radiative-coupling artifact; ABSENT in radoff): "
                        "max Tmax 458.5 K (annual) / 451.0 K (first lunation) vs radoff's clean subsolar cap "
                        "383.5-385.5 K. NOT masked per standing instruction for this deliverable.",
            n_faces_gt_400K_annual=int((coupled["annual"]["tmax"] > 400).sum()),
            frac_faces_gt_400K_annual=float((coupled["annual"]["tmax"] > 400).mean()),
            n_faces_gt_400K_first_lunation=int((coupled["first_lunation"]["tmax"] > 400).sum()),
        ),
    )
    with open(OUT / "summary_yearlong.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {OUT/'summary_yearlong.json'}")


if __name__ == "__main__":
    main()
