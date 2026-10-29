#!/usr/bin/env python3
"""Moon3D band-separated INDIRECT flux — first data products from the two new
per-band radiosity outputs (indirect_solar, indirect_ir).

The coupled two-band run now writes the ABSORBED indirect flux split into its
two source bands (visible multiply-scattered solar + thermal IR) as per-timestep
frames + statistics, in the same layout as direct_solar/ (see the 2026-07-17
band-separated-output work). This script reads the per-face MAXIMUM-over-time of
each band from statistics/indirect_{solar,ir}_max.bin and renders:

  (a) fig_indirect_bands_map  — two 3D perspective panels (same camera as
      moon3d_temp_3d, elev 24 / azim -62), left = max indirect SOLAR, right =
      max indirect IR, each with its OWN colorbar (the two bands differ by
      roughly an order of magnitude, so a shared scale would wash out the solar
      band).
  (b) fig_indirect_bands_hist — BOTH bands overlaid on one set of shared axes:
      histograms of the per-face max indirect flux over the coupled faces, log
      count (y) over log-spaced flux bins (x, shared across both bands so the
      ~order-of-magnitude offset between them is directly legible), two colored
      series with a legend, because the distribution spans several decades.
      Emitted twice: `fig_indirect_bands_hist` (log count) and
      `fig_indirect_bands_hist_linear` (linear count), same x/series/styling.
  (c) fig_indirect_bands_hist_ratio — same overlaid log-x/log-count histogram
      as (b), but the flux axis is normalized to the solar constant S (derived
      from the run as the peak direct_solar = normal-incidence insolation).
  (d) fig_indirect_vs_temp_max — scatter of per-face MAX indirect flux (log y,
      floored at NOISE_FLOOR) vs per-face MAX temperature (linear x), both bands
      overlaid as rasterized low-alpha point clouds.
  (e) fig_indirect_vs_temp_mean — same scatter for the MEANS (indirect_*_avg vs
      temperature_avg).

The `--figs {map,hist,hist_ratio,scatter_max,scatter_mean,both,all}` selector
picks which figure(s) to (re)write; the newer options are additive so existing
figures are never rewritten unless explicitly requested.

The statistics binaries carry a leading int32 face-count header (face f at index
f+1 when read as float32), identical to temperature_*/direct_solar_* .bin.

Following the crew figure convention there are NO on-figure data tables or
numeric annotations; headline numbers are printed to stdout for the caller.

Example:
    python3 moon3d_indirect_bands.py \
        --run  .../moon3d_lat_fixed/native_coupled \
        --outdir docs/figures/fixed --mask-tmax 415
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces, render_panel
from ptm_pubstyle import apply_style, DOUBLE_W, savefig_pinned

BANDS = [("indirect_solar", "indirect solar (visible scattered)"),
         ("indirect_ir", "indirect IR (thermal)")]

# Lower edge of the shared log-flux histogram axis. The prior 1e-4 W/m^2 edge was
# not a physical floor: the per-band NONZERO minima reach into float32 denormals
# (solar min 1.4e-45, IR min 8.4e-40; the IR band also carries a handful of tiny
# NEGATIVE round-off values), so the raw nonzero minimum is pure accumulation
# noise, not signal — binning down to it would waste ~40 decades of empty axis.
# But 1e-4 also CLIPS real data: the IR band's physical population is continuous
# from ~1e-8 upward and 1e-4 dropped ~4.8k genuine faces. Empirically the IR band
# shows a clean ~10-decade EMPTY gap ([1e-18, 1e-8) holds zero faces) between its
# denormal round-off cluster (<1e-18) and the start of its physical population
# (~1e-8); we adopt the bottom of that gap as a stated float32 round-off floor. It
# captures the full physical IR population and the bulk of the solar tail, while
# excluding the deep-denormal round-off that runs down to 1e-45.
NOISE_FLOOR = 1e-8  # W/m^2 — stated float32 round-off floor (see note above)


def load_stat(stat_dir: Path, name: str) -> np.ndarray:
    """Per-face statistic; leading int32/float32 count header dropped."""
    return np.fromfile(stat_dir / f"{name}.bin", dtype=np.float32)[1:]


def render_panel_shaded(ax, v, f, fc_vals, cmap, norm, elev, azim, title, cbar_label,
                        zexag=3.0, mask=None, mask_color=(0.7, 0.7, 0.7, 1.0),
                        light_elev=35.0, light_azim=-45.0, light_strength=0.35):
    """Same rendering as `moon3d_temp_3d.render_panel`, plus a FAINT
    directional hillshade -- added 2026-07-22 (Paul's manuscript v4 ask: tilt
    the map camera more top-down and add a faint light source so topography
    still reads once the oblique view is gone).

    The hillshade is blended ADDITIVELY into the NORMALIZED scalar value
    before the colormap is applied (`g -> g + light_strength*shade*(1-g)`),
    the same formula `moon3d_illum_noon_by_lat.py` uses for its own hillshade
    -- NOT a multiplicative darkening of the post-colormap RGB. That
    distinction matters here: most of this map's faces carry indirect flux
    ~0 (colormap-black), and a multiplicative darkening of black stays black,
    so the crater relief would stay invisible outside the lit patch. Blending
    into the normalized value instead lets zero-flux (fully shadowed/unlit)
    faces still pick up a faint relief-dependent brightness, while the
    colorbar itself keeps mapping the TRUE (unshaded) fc_vals -- the physical
    flux scale is unaffected, only the rendered face brightness is.

    The hillshade is computed from TRUE, pre-exaggeration face normals (not the
    zexag-inflated geometry used only for the visual relief boost), so it
    reflects the real ~0.2 depth/diameter crater slope, not an exaggerated one.
    "Faint" means keeping `light_strength` small (default 0.35).

    Kept as a LOCAL variant, not merged into `moon3d_temp_3d.render_panel`, so
    that shared function -- used unmodified by Fig. 12 (temp_3d_native_twoband)
    and the latitude montage -- is untouched by this change.
    """
    v = v.copy()
    polys_flat = v[f]  # (nf, 3, 3), PRE-exaggeration -> true-relief normals
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)   # colorbar stays on TRUE fc_vals

    nrm = np.cross(polys_flat[:, 1] - polys_flat[:, 0], polys_flat[:, 2] - polys_flat[:, 0])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    # locally up-facing patch convention (matches moon3d_illum_noon_by_lat.py's
    # own hillshade): flip any normal that points "down" in this recentered,
    # z-up frame so shading reflects relief, not winding direction.
    nrm[nrm[:, 2] < 0] *= -1.0
    el, az = np.radians(light_elev), np.radians(light_azim)
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    shade = np.clip(nrm @ light, 0.0, 1.0)

    g = np.clip(norm(fc_vals), 0.0, 1.0)
    g = np.clip(g + light_strength * shade * (1.0 - g), 0.0, 1.0)
    colors = cmap_obj(g)

    if mask is not None:
        colors[mask] = mask_color

    v[:, 2] *= zexag
    polys = v[f]
    pc = Poly3DCollection(polys, facecolors=colors, edgecolors="none",
                          linewidths=0.0, antialiased=False, rasterized=True)
    ax.add_collection3d(pc)
    lo, hi = v.min(0), v.max(0)
    d = hi - lo
    for setlim, a0, a1 in ((ax.set_xlim, lo[0], hi[0]),
                           (ax.set_ylim, lo[1], hi[1]),
                           (ax.set_zlim, lo[2], hi[2])):
        setlim(a0, a1)
    try:
        ax.set_box_aspect((d[0], d[1], d[2]))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, pad=-2)
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.5, pad=-0.02, aspect=16)
    cb.set_label(cbar_label)
    return pc


def scatter_flux_vs_temp(outdir: Path, temp: np.ndarray, arrays: dict, colors: dict,
                         stat: str, temp_label: str, fname: str, title: str,
                         dpi: int) -> None:
    """Per-face absorbed indirect flux (log y, floored at NOISE_FLOOR) vs
    temperature (linear x), BOTH bands overlaid as rasterized low-alpha point
    clouds (~509k faces/band). The scatter layer is rasterized so the vector
    PDF stays small; ALL faces are shown (no over-count masking) so the known
    the VF over-count population appears honestly as high-T outliers."""
    from matplotlib.lines import Line2D
    figs, ax = plt.subplots(1, 1, figsize=(0.80 * DOUBLE_W, 0.52 * DOUBLE_W))
    ymax = 0.0
    for key, label in BANDS:
        v = arrays[key]
        m = v >= NOISE_FLOOR
        ax.scatter(temp[m], v[m], s=1.5, c=colors[key], alpha=0.03,
                   linewidths=0.0, edgecolors="none", rasterized=True)
        ymax = max(ymax, float(v[m].max()))
        print(f"  scatter {key}: plotted {int(m.sum())}/{len(v)} faces "
              f"(>= {NOISE_FLOOR:.0e} W/m^2; {int((v < NOISE_FLOOR).sum())} below floor hidden)")
    ax.set_yscale("log")
    # half-decade top headroom so the upper-left legend clears the point cloud
    ax.set_ylim(bottom=NOISE_FLOOR, top=ymax * 3.0)
    ax.set_xlabel(temp_label)
    ax.set_ylabel(f"{stat} absorbed indirect flux [W/m$^2$]")
    # opaque, enlarged proxy markers so the low-alpha cloud reads in the legend
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                      color=colors[k], label=f"{stat} {lbl}") for k, lbl in BANDS]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)
    figs.suptitle(title, fontsize=13, fontweight="bold")
    figs.tight_layout(rect=(0, 0, 1, 0.95))
    o = outdir / f"{fname}.png"
    figs.savefig(o, dpi=dpi); figs.savefig(o.with_suffix(".pdf"))
    print(f"wrote {o} (+pdf)  [rasterized scatter, 2x509k pts]")
    plt.close(figs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="coupled run dir (holds mesh.ply + statistics/)")
    ap.add_argument("--outdir", default="docs/figures/fixed")
    ap.add_argument("--mask-tmax", type=float, default=415.0,
                    help="draw faces whose coupled MAX-T exceeds this (K) in neutral "
                         "gray on the MAP and drop them from the map color scale "
                         "(VF over-count artifacts); 0 = keep all. Histograms are "
                         "over all coupled faces regardless.")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--zexag", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figs",
                    choices=["map", "hist", "hist_ratio", "scatter_max",
                             "scatter_mean", "both", "all"],
                    default="both",
                    help="which figure(s) to (re)write; 'both'=map+hist (legacy "
                         "default), 'all'=every figure. The scatter/ratio options "
                         "are additive so existing figures stay untouched.")
    ap.add_argument("--pubclean", action="store_true",
                    help="Publication-clean variant (Paul's no-on-figure-notes rule, "
                         "manuscript v4 review, 2026-07-22): suppress the upper "
                         "suptitle on the map/hist/hist_ratio figures. Colorbars/axis "
                         "labels/legends unaffected. Additive, default OFF.")
    ap.add_argument("--pubclean-canvas-px", default=None, metavar="WxH",
                    help="With --pubclean, pin the saved canvas to this exact pixel "
                         "size at --dpi so it matches an annotated (titled) sibling's "
                         "own shipped size 1:1. Apply per single --figs call (map and "
                         "hist have different natural aspect ratios).")
    ap.add_argument("--tag", default="",
                    help="suffix appended to the map/hist output filenames before "
                         "the extension (e.g. '_v2'), so a regenerated figure lands "
                         "alongside the original without overwriting it. (hist_ratio/"
                         "scatter filenames are untouched by this flag.)")
    ap.add_argument("--light-elev", type=float, default=None,
                    help="If set, enables a FAINT directional hillshade on the MAP "
                         "figure only (added 2026-07-22): sun elevation in deg for "
                         "shading. Does NOT move the camera -- --elev/--azim do that. "
                         "Off by default (map renders byte-identically without it).")
    ap.add_argument("--light-azim", type=float, default=-45.0,
                    help="Hillshade light azimuth in deg; only used with --light-elev.")
    ap.add_argument("--light-strength", type=float, default=0.35,
                    help="Hillshade darkening fraction (0=off, 1=fully dark) on faces "
                         "facing away from the light; only used with --light-elev.")
    a = ap.parse_args()

    run = Path(a.run)
    stat = run / "statistics"
    ply = run / "mesh.ply"

    # geometry (match moon3d_temp_3d preprocessing: recenter + z-flip up)
    ff = tri_faces(pv.read(str(ply)))
    vf = read_ply_vertices_double(str(ply))
    vf = vf - vf.mean(0)
    vf[:, 2] *= -1.0

    bands = {key: load_stat(stat, f"{key}_max") for key, _ in BANDS}
    nf = len(ff)
    for key, arr in bands.items():
        if len(arr) != nf:
            print(f"ERROR: {key}_max has {len(arr)} values, mesh has {nf} faces",
                  file=sys.stderr)
            return 1

    if a.mask_tmax > 0:
        hot = load_stat(stat, "temperature_max") > a.mask_tmax
        print(f"map: masking {int(hot.sum())} over-count faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) as neutral gray")
    else:
        hot = np.zeros(nf, bool)
    good = ~hot

    for key, _ in BANDS:
        v = bands[key]
        vg = v[good]
        print(f"  {key}_max [W/m^2]: n={len(v)}  nonzero={int((v > 0).sum())}  "
              f"max={v.max():.4g}  median={np.median(v):.4g}  "
              f"p90={np.percentile(v, 90):.4g}  p99={np.percentile(v, 99):.4g}  "
              f"(good-only max={vg.max():.4g})")

    apply_style()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- (a) map: two perspective panels, own colorbar each --------------
    if a.figs in ("map", "both", "all"):
        figm = plt.figure(figsize=(DOUBLE_W, 0.52 * DOUBLE_W))
        axL = figm.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
        axR = figm.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
        for ax, (key, label), tag in ((axL, BANDS[0], "(a)"), (axR, BANDS[1], "(b)")):
            vg = bands[key][good]
            vmax = float(np.percentile(vg, 99.0))
            vmax = vmax if vmax > 0 else float(vg.max() or 1.0)
            norm = Normalize(0.0, vmax)
            if a.light_elev is not None:
                render_panel_shaded(ax, vf, ff, bands[key], "inferno", norm, a.elev, a.azim,
                                    f"max {label}", "flux [W/m$^2$]", zexag=a.zexag, mask=hot,
                                    light_elev=a.light_elev, light_azim=a.light_azim,
                                    light_strength=a.light_strength)
            else:
                render_panel(ax, vf, ff, bands[key], "inferno", norm, a.elev, a.azim,
                             f"max {label}", "flux [W/m$^2$]", zexag=a.zexag, mask=hot)
            ax.text2D(0.03, 0.95, tag, transform=ax.transAxes,
                      fontsize=10, fontweight="bold", va="top")
        if not a.pubclean:
            figm.suptitle("Moon3D 509k — max absorbed indirect flux by band (coupled)",
                          fontsize=13, fontweight="bold", y=0.98)
        figm.tight_layout(rect=(0, 0, 1, 1) if a.pubclean else (0, 0, 1, 0.95))
        om = outdir / f"fig_indirect_bands_map{a.tag}.png"
        if a.pubclean_canvas_px:
            w, h = (int(v) for v in a.pubclean_canvas_px.lower().split("x"))
            written = savefig_pinned(figm, om.with_suffix(""), canvas_px=(w, h), dpi=a.dpi)
            print(f"wrote {written[0]} (+pdf)  [pinned canvas {w}x{h}px]")
        else:
            figm.savefig(om, dpi=a.dpi); figm.savefig(om.with_suffix(".pdf"))
            print(f"wrote {om} (+pdf)")

    # ---- (b) histogram: both bands on shared log-flux axes ---------------
    # Two variants (log-count and linear-count y); same x, series, and styling.
    if a.figs in ("hist", "both", "all"):
        pos = {key: bands[key][bands[key] > 0] for key, _ in BANDS}
        # Report the floor decision explicitly so no silent clipping goes
        # unrecorded (min nonzero per band, faces below the prior 1e-4 edge, and
        # faces below the adopted float32 round-off floor that we exclude).
        for key, _ in BANDS:
            p = pos[key]
            print(f"  {key}: min-nonzero={p.min():.3e}  "
                  f"faces<1e-4(prior edge)={int((p < 1e-4).sum())}  "
                  f"faces<{NOISE_FLOOR:.0e}(new floor, excluded as noise)="
                  f"{int((p < NOISE_FLOOR).sum())}")
        lo = NOISE_FLOOR
        hi = max(p.max() for p in pos.values())
        print(f"  histogram shared lower edge = {lo:.0e} W/m^2 (was 1e-4)")
        bins = np.logspace(np.log10(lo), np.log10(hi), 70)
        colors = {"indirect_solar": "#b2182b", "indirect_ir": "#2166ac"}
        # wider + taller canvas (not just a y-limit stretch) so the legend has
        # room to sit clear of the data
        figsize = (0.80 * DOUBLE_W, 0.52 * DOUBLE_W)

        for yscale, suffix in (("log", ""), ("linear", "_linear")):
            figh, ax = plt.subplots(1, 1, figsize=figsize)
            ymax = 0.0
            for key, label in BANDS:
                n, _, _ = ax.hist(pos[key], bins=bins, color=colors[key], alpha=0.55,
                                  edgecolor=colors[key], linewidth=0.8,
                                  label=f"max {label}")
                ymax = max(ymax, float(n.max()))
            ax.set_xscale("log"); ax.set_yscale(yscale)
            # top headroom so the legend clears the tallest bar (~1.3e4); the
            # multiplier differs by scale (a decade on log, ~30% on linear)
            ax.set_ylim(top=ymax * (10.0 if yscale == "log" else 1.55))
            ax.set_xlabel("max absorbed indirect flux [W/m$^2$]")
            ax.set_ylabel("face count")
            ax.legend(loc="upper left", framealpha=0.95)
            if not a.pubclean:
                figh.suptitle("Moon3D 509k — per-face max indirect flux distribution (coupled)",
                              fontsize=13, fontweight="bold")
            figh.tight_layout(rect=(0, 0, 1, 1) if a.pubclean else (0, 0, 1, 0.95))
            oh = outdir / f"fig_indirect_bands_hist{suffix}{a.tag}.png"
            if a.pubclean_canvas_px:
                w, h = (int(v) for v in a.pubclean_canvas_px.lower().split("x"))
                written = savefig_pinned(figh, oh.with_suffix(""), canvas_px=(w, h), dpi=a.dpi)
                print(f"wrote {written[0]} (+pdf)  [{yscale} y, pinned canvas {w}x{h}px]")
            else:
                figh.savefig(oh, dpi=a.dpi); figh.savefig(oh.with_suffix(".pdf"))
                print(f"wrote {oh} (+pdf)  [{yscale} y, {figsize[0]:.2f}x{figsize[1]:.2f} in]")

    # ---- (c) ratio histogram: flux / solar constant ----------------------
    # Same overlaid log-x/log-count histogram as (b), but the flux axis is
    # normalized by the solar constant S, derived self-consistently from the run
    # as the PEAK direct_solar over all faces & timesteps (= normal-incidence
    # insolation the model actually delivered). Cross-checked against the sun
    # distance in body_positions.csv (0.99964-1.00941 AU -> 1336-1362 W/m^2).
    if a.figs in ("hist_ratio", "all"):
        S = float(load_stat(stat, "direct_solar_max").max())
        print(f"  solar constant S = peak direct_solar over run = {S:.2f} W/m^2 "
              f"(normal-incidence; ~1.005 AU; body_positions cross-check 1336-1362)")
        rpos = {key: bands[key][bands[key] > 0] / S for key, _ in BANDS}
        lo = NOISE_FLOOR / S
        hi = max(p.max() for p in rpos.values())
        bins = np.logspace(np.log10(lo), np.log10(hi), 70)
        colors = {"indirect_solar": "#b2182b", "indirect_ir": "#2166ac"}
        figr, ax = plt.subplots(1, 1, figsize=(0.80 * DOUBLE_W, 0.52 * DOUBLE_W))
        ymax = 0.0
        for key, label in BANDS:
            n, _, _ = ax.hist(rpos[key], bins=bins, color=colors[key], alpha=0.55,
                              edgecolor=colors[key], linewidth=0.8, label=f"max {label}")
            ymax = max(ymax, float(n.max()))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(top=ymax * 10.0)
        ax.set_xlabel(f"max absorbed indirect flux / solar constant  "
                      f"(S = {S:.0f} W m$^{{-2}}$)")
        ax.set_ylabel("face count")
        ax.legend(loc="upper left", framealpha=0.95)
        figr.suptitle("Moon3D 509k — per-face max indirect flux / solar constant (coupled)",
                      fontsize=13, fontweight="bold")
        figr.tight_layout(rect=(0, 0, 1, 0.95))
        oh = outdir / "fig_indirect_bands_hist_ratio.png"
        figr.savefig(oh, dpi=a.dpi); figr.savefig(oh.with_suffix(".pdf"))
        print(f"wrote {oh} (+pdf)  [log y, ratio x, S={S:.1f} W/m^2]")
        plt.close(figr)

    # ---- (d) scatter: per-face MAX indirect flux vs MAX temperature -------
    colors = {"indirect_solar": "#b2182b", "indirect_ir": "#2166ac"}
    if a.figs in ("scatter_max", "all"):
        tmax = load_stat(stat, "temperature_max")
        scatter_flux_vs_temp(
            outdir, tmax, bands, colors, "max",
            "per-face max temperature [K]", "fig_indirect_vs_temp_max",
            "Moon3D 509k — per-face max indirect flux vs max temperature (coupled)",
            a.dpi)

    # ---- (e) scatter: per-face MEAN indirect flux vs MEAN temperature -----
    if a.figs in ("scatter_mean", "all"):
        tavg = load_stat(stat, "temperature_avg")
        avg = {key: load_stat(stat, f"{key}_avg") for key, _ in BANDS}
        scatter_flux_vs_temp(
            outdir, tavg, avg, colors, "mean",
            "per-face mean temperature [K]", "fig_indirect_vs_temp_mean",
            "Moon3D 509k — per-face mean indirect flux vs mean temperature (coupled)",
            a.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
