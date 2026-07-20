#!/usr/bin/env python3
"""
F5b — 3D bowl-crater equilibrium temperature for several depth/diameter ratios,
each embedded in a flat surrounding level surface, compared to the Ingersoll,
Svitek & Murray (1992) two-zone analytic solution.

For each crater a spherical-cap bowl is meshed together with a SQUARE flat
apron at the rim level (z = 0) representing the surrounding terrain. Per-face
view factors F_ij are computed in the differential-area limit; bowl<->apron
pairs are gated by the spherical-cap escape test so the shadowed floor exchanges
IR only with the walls it can actually see (the rim occludes the apron), which
keeps the floor energy balance physical. A single-bounce radiosity balance then
gives T_i at each solar elevation:

    ε σ T_i⁴  =  (1 − A) S μ_i V_i  +  Σ_j ρ ε σ T_j⁴ F_ij

with μ_i = max(0, n̂_i · ŝ) and V_i the rim-shadow Sun visibility at face i.

Shadow-regime validation (β < β_c = arctan(2 d/D)): the shadowed floor absorbs

    Q_ind^floor  =  ε σ <T_wall^4> (1 − F_sky_floor),     F_sky_floor = sin²α,

which the lower panels compare to the mesh radiosity for each d/D.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ingersoll_crater import critical_elevation, sky_view_factor
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, panel_label


SIGMA = 5.670374e-8
SOLAR_CONSTANT = 1361.0
ALBEDO = 0.12
EMISSIVITY = 0.95
CMAP = plt.cm.RdBu_r            # diverging map, reversed so hot = red, cold = blue


# ---------------------------------------------------------------------------
# Mesh: spherical-cap bowl + square flat apron at the rim level

def build_crater_mesh(depth_to_diameter: float = 0.1,
                      diameter: float = 1.0,
                      n_rings: int = 50,
                      n_az: int = 120,
                      flat_extent: float = 1.7,
                      n_flat_rings: int = 12):
    """Bowl crater + square flat surround. Returns (verts, faces, masks, geom).

    masks: dict with boolean per-face arrays is_floor / is_wall / is_flat / is_bowl.
    The apron morphs from the rim circle out to a square of half-width
    flat_extent * r_rim, all at z = 0.
    """
    d = depth_to_diameter * diameter
    r_rim = diameter / 2.0
    R = (r_rim * r_rim + d * d) / (2.0 * d)
    z_c = R - d
    theta_max = float(np.arcsin(r_rim / R))

    verts = [[0.0, 0.0, -d]]                    # apex (floor centre)
    thetas = np.linspace(0.0, theta_max, n_rings + 1)
    for k in range(1, n_rings + 1):
        th = thetas[k]
        rho = R * np.sin(th)
        z = z_c - R * np.cos(th)
        for j in range(n_az):
            phi = 2.0 * np.pi * j / n_az
            verts.append([rho * np.cos(phi), rho * np.sin(phi), z])
    rim0 = 1 + (n_rings - 1) * n_az             # first vertex of the rim ring (z=0)

    # square flat apron: morph the rim circle out to a square at z = 0
    H = flat_extent * r_rim
    flat_ring_starts = []
    for s_idx in range(1, n_flat_rings + 1):
        s = s_idx / n_flat_rings
        flat_ring_starts.append(len(verts))
        for j in range(n_az):
            phi = 2.0 * np.pi * j / n_az
            c, sn = np.cos(phi), np.sin(phi)
            sc = H / max(abs(c), abs(sn))       # ray->square boundary scale
            x = (1.0 - s) * r_rim * c + s * sc * c
            y = (1.0 - s) * r_rim * sn + s * sc * sn
            verts.append([x, y, 0.0])
    verts = np.asarray(verts)

    faces = []
    for j in range(n_az):                       # inner fan around the apex
        faces.append([0, 1 + j, 1 + (j + 1) % n_az])
    for k in range(1, n_rings):                 # bowl rings
        base_in = 1 + (k - 1) * n_az
        base_out = 1 + k * n_az
        for j in range(n_az):
            j1 = (j + 1) % n_az
            faces.append([base_in + j, base_out + j, base_out + j1])
            faces.append([base_in + j, base_out + j1, base_in + j1])
    n_bowl_faces = len(faces)

    ring_starts = [rim0] + flat_ring_starts     # apron rings (rim -> square)
    for r in range(len(ring_starts) - 1):
        a0, b0 = ring_starts[r], ring_starts[r + 1]
        for j in range(n_az):
            j1 = (j + 1) % n_az
            faces.append([a0 + j, b0 + j, b0 + j1])
            faces.append([a0 + j, b0 + j1, a0 + j1])
    faces = np.asarray(faces, dtype=int)

    is_flat = np.arange(len(faces)) >= n_bowl_faces
    cent_xy = verts[faces].mean(axis=1)[:, :2]
    is_floor = (np.hypot(cent_xy[:, 0], cent_xy[:, 1]) < 0.18 * r_rim) & ~is_flat
    masks = {"is_floor": is_floor, "is_flat": is_flat, "is_bowl": ~is_flat,
             "is_wall": (~is_flat) & ~is_floor}
    geom = {"d": d, "r_rim": r_rim, "R": R, "z_c": z_c, "H": H,
            "theta_max": theta_max, "n_rings": n_rings, "n_az": n_az}
    return verts, faces, masks, geom


def face_geometry(verts, faces):
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cent = (p0 + p1 + p2) / 3.0
    cross = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    n = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    n[n[:, 2] < 0] *= -1.0                       # outward / up-facing normals
    return cent, n, area


# ---------------------------------------------------------------------------
# View factors (bowl<->apron occlusion via the spherical-cap escape test)

def compute_view_factors(cent, n, area, is_bowl, geom):
    N = len(cent)
    F = np.zeros((N, N), dtype=np.float64)
    z_c = geom["z_c"]
    C = np.array([0.0, 0.0, z_c])
    is_flat = ~is_bowl
    eps = 1e-6
    for i in range(N):
        rij = cent - cent[i]
        d = np.linalg.norm(rij, axis=1)
        d[i] = 1.0
        rh = rij / d[:, None]
        cos_i = np.maximum(0.0, rh @ n[i])
        cos_j = np.maximum(0.0, -np.einsum("ij,ij->i", rh, n))
        Frow = cos_i * cos_j * area / (np.pi * d * d)
        Frow[i] = 0.0
        if is_bowl[i]:
            # gate the apron columns: bowl face i sees flat face j only if the
            # ray i->j escapes the spherical cap above the rim plane.
            PC = C - cent[i]
            rdotPC = rh @ PC
            p_exit_z = cent[i, 2] + 2.0 * rdotPC * rh[:, 2]
            escapes = (rdotPC <= eps) | (p_exit_z >= 0.0)
            Frow[is_flat & ~escapes] = 0.0
            F[i, :] = Frow
        else:
            Frow[is_bowl] = 0.0                  # filled below by reciprocity
            F[i, :] = Frow
    # apron->bowl by reciprocity from the gated bowl->apron block
    b = np.where(is_bowl)[0]
    f = np.where(is_flat)[0]
    if f.size and b.size:
        F[np.ix_(f, b)] = (F[np.ix_(b, f)].T) * area[b][None, :] / area[f][:, None]
    row_sum = F.sum(axis=1)
    bad = row_sum > 1.0
    if np.any(bad):
        F[bad] = F[bad] / row_sum[bad, None]
    return F


# ---------------------------------------------------------------------------
# Sun visibility

def _visible_point(P, sun_dir, z_c, R, eps=1e-6):
    C = np.array([0.0, 0.0, z_c])
    rdotPC = ((C - P) * sun_dir).sum(axis=-1)
    P_exit = P + 2.0 * rdotPC[..., None] * sun_dir
    return (rdotPC <= eps) | (P_exit[..., 2] >= 0.0)


def sun_visibility(verts, faces, cent, n, geom, sun_dir, is_bowl):
    """Per-face Sun visibility in [0, 1]. Apron faces (level surface) are fully
    lit when the Sun is up; bowl faces use the anti-aliased rim-shadow test."""
    if sun_dir[2] <= 0:
        return np.zeros(len(cent))
    front = (n @ sun_dir) > 0.0
    vis = np.zeros(len(cent))
    is_flat = ~is_bowl
    vis[is_flat & front] = 1.0
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    samples = np.stack([cent, 0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v0 + v2)],
                       axis=1)
    vb = _visible_point(samples, sun_dir, geom["z_c"], geom["R"]).mean(axis=1)
    bowl_front = is_bowl & front
    vis[bowl_front] = vb[bowl_front]
    return vis


# ---------------------------------------------------------------------------
# Solver

def solve_radiative_equilibrium(F, mu, vis, T_init=200.0, albedo=ALBEDO,
                                emissivity=EMISSIVITY, solar=SOLAR_CONSTANT,
                                tol=1e-3, max_iter=600):
    D = (1.0 - albedo) * solar * mu * vis / (emissivity * SIGMA)
    T = np.full(len(F), T_init, dtype=np.float64)
    for _ in range(max_iter):
        T_new = (D + F @ (T ** 4)) ** 0.25
        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break
        T = 0.5 * (T + T_new)
    return T


# ---------------------------------------------------------------------------
# Rendering

def render_crater_3d(ax, verts, faces, T, vmin, vmax):
    coll = Poly3DCollection(verts[faces], facecolors=CMAP(Normalize(vmin, vmax)(T)),
                            edgecolors="none", linewidths=0.0, antialiased=False)
    ax.add_collection3d(coll)
    H = float(np.abs(verts[:, :2]).max())
    d = float(-verts[0, 2])
    ax.set_xlim(-H, H)
    ax.set_ylim(-H, H)
    ax.set_zlim(-d, 0.02)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.42), zoom=1.35)
    except TypeError:
        ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=46, azim=-65)              # tilted up to see into the bowl
    ax.set_axis_off()
    return coll


# ---------------------------------------------------------------------------

def run_one(dd, n_rings, n_az):
    """Mesh, view factors, render solve, and β-sweep for one d/D ratio."""
    verts, faces, masks, geom = build_crater_mesh(
        depth_to_diameter=dd, diameter=1.0, n_rings=n_rings, n_az=n_az)
    cent, n, area = face_geometry(verts, faces)
    is_bowl, is_floor = masks["is_bowl"], masks["is_floor"]
    is_wall = masks["is_wall"]
    F = compute_view_factors(cent, n, area, is_bowl, geom)

    A_wall = area[is_wall]; A_wall_sum = A_wall.sum()
    A_floor = area[is_floor]; A_floor_sum = A_floor.sum()
    F_floor_terrain = float(np.average(F[is_floor].sum(axis=1), weights=A_floor))
    fsky_analytic = sky_view_factor(dd)
    bcrit = float(np.rad2deg(critical_elevation(dd)))

    def solve_at(beta_deg):
        b = np.deg2rad(beta_deg)
        sd = np.array([np.cos(b), 0.0, np.sin(b)])
        v = sun_visibility(verts, faces, cent, n, geom, sd, is_bowl)
        mu = np.maximum(0.0, n @ sd)
        return solve_radiative_equilibrium(F, mu, v)

    def t_wall_eff(T):
        return float((np.sum(A_wall * T[is_wall] ** 4) / A_wall_sum) ** 0.25)

    def q_meas(T):
        Q = EMISSIVITY * SIGMA * (F @ T ** 4)
        return float(np.sum(A_floor * Q[is_floor]) / A_floor_sum)

    def q_anal(T):
        return EMISSIVITY * SIGMA * (t_wall_eff(T) ** 4) * F_floor_terrain

    beta_render = 0.5 * bcrit
    T_render = solve_at(beta_render)

    bsweep = np.linspace(0.05, bcrit * 0.92, 22)
    Tf_mesh = np.empty_like(bsweep); Tf_ing = np.empty_like(bsweep)
    Qm = np.empty_like(bsweep); Qa = np.empty_like(bsweep)
    for i, bd in enumerate(bsweep):
        Tb = solve_at(bd)
        Tw = t_wall_eff(Tb)
        Tf_mesh[i] = float(Tb[is_floor].mean())
        Tf_ing[i] = Tw * (1.0 - fsky_analytic) ** 0.25
        Qm[i] = q_meas(Tb); Qa[i] = q_anal(Tb)

    print(f"  d/D={dd:<5}: {len(faces):6d} faces (floor {int(is_floor.sum())}, "
          f"apron {int(masks['is_flat'].sum())}), β_c={bcrit:5.2f}°, "
          f"F_sky floor mesh={1-F_floor_terrain:.4f} vs sin²α={fsky_analytic:.4f}")
    return dict(dd=dd, verts=verts, faces=faces, T_render=T_render,
                beta_render=beta_render, bcrit=bcrit, bsweep=bsweep,
                Tf_mesh=Tf_mesh, Tf_ing=Tf_ing, Qm=Qm, Qa=Qa)


def make_figure(out_path, dd_values=(0.01, 0.1, 0.2), n_rings=50, n_az=120):
    apply_style()
    res = [run_one(dd, n_rings, n_az) for dd in dd_values]

    vmin = 80.0
    vmax = float(np.ceil(max(r["T_render"].max() for r in res) / 10.0) * 10.0)
    colors = ["#0072B2", "#E69F00", "#009E73"]

    fig = plt.figure(figsize=(DOUBLE_W * 1.55, DOUBLE_W * 1.05))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.28, wspace=0.12,
                          left=0.07, right=0.97, top=0.96, bottom=0.09)

    for k, r in enumerate(res):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        render_crater_3d(ax, r["verts"], r["faces"], r["T_render"], vmin, vmax)
        ax.set_title(fr"$d/D = {r['dd']:g}$   ($\beta = {r['beta_render']:.1f}^\circ$)",
                     fontsize=13, pad=2)

    # ---- (a) floor T vs β/β_c ------------------------------------------------
    ax_T = fig.add_subplot(gs[1, :2])
    rms_all = []
    for r, c in zip(res, colors):
        x = r["bsweep"] / r["bcrit"]
        ax_T.plot(x, r["Tf_ing"], color=c, lw=2.2, ls="--")
        ax_T.plot(x, r["Tf_mesh"], "o", color=c, mfc="white", mew=1.6, ms=6.5,
                  label=fr"$d/D = {r['dd']:g}$")
        rms_all.append(np.sqrt(np.mean((r["Tf_mesh"] - r["Tf_ing"]) ** 2)))
    ax_T.set_xlim(0, 1)
    ax_T.set_ylim(top=ax_T.get_ylim()[1] * 1.18)
    ax_T.set_xlabel(r"Solar elevation $\beta\,/\,\beta_c$")
    ax_T.set_ylabel(r"Floor (shadow) mean $T$ (K)")
    ax_T.legend(loc="upper left", fontsize=10, title="markers: mesh   dashed: Ingersoll",
                title_fontsize=8.5)
    ax_T.text(0.97, 0.06, fr"RMS $T_\mathrm{{mesh}}-T_\mathrm{{ing}}$ = "
              fr"{np.sqrt(np.mean(np.square(rms_all))):.2f} K",
              transform=ax_T.transAxes, va="bottom", ha="right", fontsize=10)
    panel_label(ax_T, "(a)")

    # ---- (b) floor Q_ind vs β/β_c -------------------------------------------
    ax_Q = fig.add_subplot(gs[1, 2])
    max_rel = 0.0
    for r, c in zip(res, colors):
        x = r["bsweep"] / r["bcrit"]
        ax_Q.plot(x, r["Qa"], color=c, lw=2.2, ls="--")
        ax_Q.plot(x, r["Qm"], "s", color=c, mfc="white", mew=1.6, ms=6,
                  label=fr"$d/D = {r['dd']:g}$")
        max_rel = max(max_rel, float((np.abs(r["Qm"] - r["Qa"])
                                      / np.maximum(1e-12, r["Qa"])).max()))
    ax_Q.set_xlim(0, 1)
    ax_Q.set_xlabel(r"$\beta\,/\,\beta_c$")
    ax_Q.set_ylabel(r"Floor $Q_\mathrm{ind}$ (W m$^{-2}$)")
    ax_Q.legend(loc="upper left", fontsize=9)
    ax_Q.text(0.97, 0.04, f"max rel err = {100 * max_rel:.2f}%",
              transform=ax_Q.transAxes, va="bottom", ha="right", fontsize=9)
    panel_label(ax_Q, "(b)")

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP)
    cax = fig.add_axes([0.028, 0.46, 0.012, 0.46])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(r"Equilibrium $T$ (K)")
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"\n  overall RMS(T_mesh−T_ing) = {np.sqrt(np.mean(np.square(rms_all))):.3f} K, "
          f"max Q rel err = {100*max_rel:.3f}%\n  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/figures/v2/bowl_crater_3d.png")
    p.add_argument("--dd", type=float, nargs="+", default=[0.01, 0.1, 0.2])
    p.add_argument("--n-rings", type=int, default=50)
    p.add_argument("--n-az", type=int, default=120)
    args = p.parse_args()
    make_figure(Path(args.out), dd_values=tuple(args.dd),
                n_rings=args.n_rings, n_az=args.n_az)


if __name__ == "__main__":
    main()
