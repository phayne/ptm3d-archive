#!/usr/bin/env python3
"""Map-view comparison + difference figure for the Lunar-VISE camera
DEM-resolution error study.

Four panels (north-up, site local ENU, meters):
  (a) channel ortho-projected onto the FINE (truth) mesh
  (b) same pixels projected onto the COARSE DEM
  (c) difference (coarse - fine)
  (d) per-pixel error map (geolocation offset for radiance; |Δphase| for
      phase), with summary stats

--channel radiance : radiance channel + horizontal geolocation error (m)
--channel phase    : solar phase angle (deg) + |Δ phase| error (deg)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from ptm_pubstyle import apply_style, DOUBLE_W  # noqa: E402

CH = {
    "radiance": dict(
        fine="fine_img", coarse="coarse_img", err="disp_img",
        stats="disp_stats", cmap="gray",
        clabel="Radiance (W m$^{-2}$ sr$^{-1}$)",
        ename="Horizontal geolocation error", elabel="Offset (m)",
        eunit="m", efmt="{:.2f}", title="radiance"),
    "phase": dict(
        fine="fine_pha_img", coarse="coarse_pha_img", err="pha_err_img",
        stats="pha_stats", cmap="viridis",
        clabel="Solar phase angle (deg)",
        ename="Phase-angle error", elabel="|$\\Delta$ phase| (deg)",
        eunit="deg", efmt="{:.3f}", title="solar phase angle"),
    "incidence": dict(
        fine="fine_inc_img", coarse="coarse_inc_img", err="inc_err_img",
        stats="inc_stats", cmap="cividis",
        clabel="Solar incidence angle (deg)",
        ename="Incidence-angle error", elabel="|$\\Delta i$| (deg)",
        eunit="deg", efmt="{:.3f}",
        title="solar incidence angle (drives I/F)"),
    "temperature": dict(
        fine="fine_temp_img", coarse="coarse_temp_img",
        err="temp_err_img", stats="temp_stats", cmap="inferno",
        clabel="Surface temperature (K)",
        ename="Temperature error", elabel="|$\\Delta T$| (K)",
        eunit="K", efmt="{:.2f}", title="surface temperature"),
    "thermal": dict(
        fine="fine_therm_img", coarse="coarse_therm_img",
        err="therm_err_img", stats="therm_stats", cmap="magma",
        clabel="Thermal radiance (W m$^{-2}$ sr$^{-1}$)",
        ename="Thermal-radiance error",
        elabel="|$\\Delta L$| (W m$^{-2}$ sr$^{-1}$)",
        eunit="W m$^{-2}$ sr$^{-1}$", efmt="{:.3f}",
        title="thermal radiance (EPF $\\propto\\cos^{0.5}\\theta_e$)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", choices=list(CH), default="radiance")
    ap.add_argument("--dem-label", default="coarse DEM",
                    help="Label for panel (b), e.g. '1 m-equiv DEM'.")
    ap.add_argument("--lst-label", default="10:00 LST",
                    help="Local-solar-time label for the suptitle.")
    args = ap.parse_args()

    c = CH[args.channel]
    d = np.load(args.npz)
    fine = d[c["fine"]]
    coarse = d[c["coarse"]]
    err = d[c["err"]]
    ext = d["extent"]
    mean_e, p95_e, max_e = d[c["stats"]]
    res = float(d["res_m"])

    apply_style()
    plt.rcParams.update({"axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE_W, DOUBLE_W * 0.66),
                            layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06,
                                wspace=0.10, hspace=0.10)
    both = np.concatenate([fine[np.isfinite(fine)],
                           coarse[np.isfinite(coarse)]])
    vmin, vmax = np.percentile(both, [1, 99])
    imk = dict(origin="lower", extent=ext, cmap=c["cmap"],
               vmin=vmin, vmax=vmax, interpolation="nearest")
    cbk = dict(fraction=0.046, pad=0.025)

    def cbar(im, ax, label):
        cb = fig.colorbar(im, ax=ax, label=label, **cbk)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.label.set_size(7)
        cb.ax.yaxis.set_offset_position("left")
        return cb

    ax = axs[0, 0]
    im = ax.imshow(fine, **imk)
    ax.set_title("(a) Fine DEM (truth)")
    cbar(im, ax, c["clabel"])

    ax = axs[0, 1]
    im = ax.imshow(coarse, **imk)
    ax.set_title(f"(b) {args.dem_label}")
    cbar(im, ax, c["clabel"])

    ax = axs[1, 0]
    diff = coarse - fine
    lim = max(np.nanpercentile(np.abs(diff), 99), 1e-9)
    im = ax.imshow(diff, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title("(c) Difference (coarse - fine)")
    cbar(im, ax, "$\\Delta$")

    ax = axs[1, 1]
    im = ax.imshow(err, origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=max(np.nanpercentile(err, 99), 1e-9),
                   interpolation="nearest")
    ax.set_title(f"(d) {c['ename']}")
    cbar(im, ax, c["elabel"])
    f = c["efmt"]
    ax.text(0.04, 0.05,
            (f"mean {f.format(mean_e)} {c['eunit']}\n"
             f"p95 {f.format(p95_e)} {c['eunit']}\n"
             f"max {f.format(max_e)} {c['eunit']}"),
            transform=ax.transAxes, fontsize=8, color="w", va="bottom",
            bbox=dict(facecolor="k", alpha=0.45, edgecolor="none",
                      boxstyle="round,pad=0.25"))

    # Outer-only axis labels to avoid label/colorbar collisions.
    for r in range(2):
        for col in range(2):
            ax = axs[r, col]
            ax.set_aspect("equal")
            ax.set_xlabel("East (m)" if r == 1 else "")
            ax.set_ylabel("North (m)" if col == 0 else "")
            if r == 0:
                ax.tick_params(labelbottom=False)

    fig.suptitle(
        f"Lunar-VISE camera: DEM-resolution {c['title']} error  "
        f"(36.5$\\degree$N, -40.7$\\degree$E; {args.lst_label}; "
        f"{res:g} m grid)", fontsize=9)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    print(f"{args.channel} error: mean={mean_e:.3f} p95={p95_e:.3f} "
          f"max={max_e:.3f} {c['eunit']}")


if __name__ == "__main__":
    main()
