#!/usr/bin/env python3
"""
F5b — Ingersoll bowl-crater validation of the PRODUCTION PTM3D radiation module.

Reads the .npy outputs of `bin/bowl_crater_ptm` (spherical-cap bowl + square flat
apron; Embree ray-traced occluded view factors via ViewFactorCalculator;
ray-traced Sun shadow; per-face radiative-equilibrium temperature from the
production RadiositySolver) and draws:

  rows 1-2 : 3D equilibrium-temperature renders at β = 1° and β = 10°, each for
             d/D = 0.01, 0.1, 0.2, with subtle directional (Sun) shading.
  row 3    : floor (shadow) mean T and floor indirect flux Q_ind vs β/β_c, PTM3D
             mesh radiosity (markers) vs the Ingersoll, Svitek & Murray (1992)
             exact two-zone formula (smooth dashed, finely sampled in β).

Run the solver first:
    ./build/bin/bowl_crater_ptm output/bowl_ptm 0.01 0.1 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import PchipInterpolator

from ptm.pubstyle import apply_style, DOUBLE_W, panel_label

CMAP = plt.cm.RdBu_r
SIGMA = 5.670374e-8
EMISS = 0.95
RENDER_BETAS = (1.0, 10.0)


def load_dd(data_dir: Path, idx: int) -> dict:
    p = data_dir / f"dd{idx}"
    meta = {}
    for line in (data_dir / f"dd{idx}_meta.txt").read_text().splitlines():
        k, v = line.split()
        meta[k] = float(v)
    return dict(
        verts=np.load(f"{p}_verts.npy"), faces=np.load(f"{p}_faces.npy"),
        Tr1=np.load(f"{p}_Trender1.npy"), Tr10=np.load(f"{p}_Trender10.npy"),
        bsweep=np.load(f"{p}_bsweep.npy"), Tfmesh=np.load(f"{p}_Tfmesh.npy"),
        Tfing=np.load(f"{p}_Tfing.npy"), Tw=np.load(f"{p}_Tw.npy"),
        Qm=np.load(f"{p}_Qm.npy"), Qa=np.load(f"{p}_Qa.npy"), meta=meta)


def face_shade(verts, faces, sun_dir, ambient=0.62):
    """Subtle Lambertian shading factor in [ambient, 1] from the Sun direction."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    n[n[:, 2] < 0] *= -1.0                       # up-facing (illuminated side)
    return ambient + (1.0 - ambient) * np.clip(n @ sun_dir, 0.0, 1.0)


def render_crater_3d(ax, verts, faces, T, vmin, vmax, beta_deg):
    b = np.deg2rad(beta_deg)
    sun = np.array([np.cos(b), 0.0, np.sin(b)])
    rgba = CMAP(LogNorm(vmin, vmax)(np.clip(T, vmin, vmax)))
    rgba[:, :3] *= face_shade(verts, faces, sun)[:, None]
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=rgba,
                        edgecolors="none", linewidths=0.0, antialiased=False))
    H = float(np.abs(verts[:, :2]).max())
    d = float(-verts[:, 2].min())
    ax.set_xlim(-H, H); ax.set_ylim(-H, H); ax.set_zlim(-d, 0.02)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.42), zoom=1.0)   # full square apron in view
    except TypeError:
        ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=46, azim=-65)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/bowl_ptm")
    ap.add_argument("--out", default="docs/figures/v2/bowl_crater_3d.png")
    ap.add_argument("--ndd", type=int, default=3)
    a = ap.parse_args()
    data_dir = Path(a.data)
    res = [load_dd(data_dir, i) for i in range(a.ndd)]

    apply_style()
    vmin = 40.0     # log scale: bring out the cold shadow as well as the hot walls
    vmax = float(np.ceil(max(max(r["Tr1"].max(), r["Tr10"].max()) for r in res) / 10.0) * 10.0)
    colors = ["#0072B2", "#E69F00", "#009E73"]

    fig = plt.figure(figsize=(DOUBLE_W * 1.55, DOUBLE_W * 1.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05],
                          left=0.11, right=0.975, top=0.95, bottom=0.065,
                          hspace=0.10, wspace=0.08)

    row_axes = [[], []]
    for ri, (Tkey, beta) in enumerate(zip(("Tr1", "Tr10"), RENDER_BETAS)):
        for k, r in enumerate(res):
            ax = fig.add_subplot(gs[ri, k], projection="3d")
            render_crater_3d(ax, r["verts"], r["faces"], r[Tkey], vmin, vmax, beta)
            row_axes[ri].append(ax)
            if ri == 0:
                ax.set_title(fr"$d/D = {r['meta']['dd']:g}$", fontsize=13, pad=0)
    # panel label (a) for the whole render block
    fig.text(0.10, 0.965, "(a)", fontsize=13, fontweight="bold", va="top", ha="left")

    # row labels (β) at the left, vertically centered on each render row
    fig.canvas.draw()
    for ri, beta in enumerate(RENDER_BETAS):
        boxes = [ax.get_position() for ax in row_axes[ri]]
        yc = 0.5 * (max(bx.y1 for bx in boxes) + min(bx.y0 for bx in boxes))
        fig.text(0.085, yc, fr"$\beta = {beta:g}^\circ$", rotation=90,
                 va="center", ha="center", fontsize=13, fontweight="bold")

    # bottom row: line plots in a nested gridspec with generous spacing
    bgs = gs[2, :].subgridspec(1, 3, wspace=0.42)
    ax_T = fig.add_subplot(bgs[0, :2])
    ax_Q = fig.add_subplot(bgs[0, 2])

    bfine = np.linspace(min(r["bsweep"][0] for r in res),
                        max(r["bsweep"][-1] for r in res), 400)
    rms_all, max_rel = [], 0.0
    for r, c in zip(res, colors):
        bc = r["meta"]["bcrit"]
        x = r["bsweep"] / bc
        # smooth analytic curves: spline the wall T_eff(β), then apply the
        # Ingersoll closed form on a fine β grid.
        sel = bfine <= r["bsweep"][-1]
        bf = bfine[sel]
        Tw_f = PchipInterpolator(r["bsweep"], r["Tw"])(bf)
        Tfi_f = Tw_f * (1.0 - r["meta"]["fsky_analytic"]) ** 0.25
        Qa_f = EMISS * SIGMA * Tw_f ** 4 * r["meta"]["floor_Fterrain"]
        ax_T.plot(bf / bc, Tfi_f, color=c, lw=2.0, ls="--")
        ax_T.plot(x, r["Tfmesh"], "o", color=c, mfc="white", mew=1.6, ms=6.5,
                  label=fr"$d/D = {r['meta']['dd']:g}$")
        ax_Q.plot(bf / bc, Qa_f, color=c, lw=2.0, ls="--")
        ax_Q.plot(x, r["Qm"], "s", color=c, mfc="white", mew=1.6, ms=6,
                  label=fr"$d/D = {r['meta']['dd']:g}$")
        rms_all.append(np.sqrt(np.mean((r["Tfmesh"] - r["Tfing"]) ** 2)))
        max_rel = max(max_rel, float((np.abs(r["Qm"] - r["Qa"])
                                      / np.maximum(1e-9, r["Qa"])).max()))

    ax_T.set_xlim(0, 1)
    ax_T.set_ylim(top=ax_T.get_ylim()[1] * 1.18)
    ax_T.set_xlabel(r"Solar elevation $\beta\,/\,\beta_c$")
    ax_T.set_ylabel(r"Floor (shadow) mean $T$ (K)")
    ax_T.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), fontsize=10,
                title="markers: PTM3D mesh   dashed: Ingersoll", title_fontsize=8.5)
    panel_label(ax_T, "(b)")

    ax_Q.set_xlim(0, 1)
    ax_Q.set_xlabel(r"$\beta\,/\,\beta_c$")
    ax_Q.set_ylabel(r"Floor $Q_\mathrm{ind}$ (W m$^{-2}$)")
    ax_Q.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9)
    panel_label(ax_Q, "(c)")

    sm = plt.cm.ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap=CMAP)
    cax = fig.add_axes([0.035, 0.42, 0.013, 0.46])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(r"Equilibrium $T$ (K)")
    ticks = [t for t in (40, 60, 100, 150, 200, 300) if vmin <= t <= vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"  overall RMS = {np.sqrt(np.mean(np.square(rms_all))):.3f} K, "
          f"max Q rel err = {100*max_rel:.3f}%\n  wrote {out}")


if __name__ == "__main__":
    main()
