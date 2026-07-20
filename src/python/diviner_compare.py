#!/usr/bin/env python3
"""Compare a PTM3D year-long run against Diviner max bolometric brightness T.

Methodology
-----------
Diviner Tbol is the brightness temperature of a blackbody emitting the surface's
bolometric flux. For a graybody at kinetic T with emissivity eps the footprint
bolometric temperature is

    Tbol(footprint) = eps^(1/4) * ( sum_i A_i T_i^4 / sum_i A_i )^(1/4)

i.e. an AREA-WEIGHTED T^4 average over the model faces in the footprint (the
anisothermal mixing Diviner sees), scaled by eps^(1/4).

We bin model faces into a regular comparison grid (cell ~= a fraction of the
Diviner footprint), and compute the MAX OVER TIME of the footprint bolometric
temperature (rigorous: max of the per-timestep footprint Tbol, from the kept
per-step temperatures). A cheaper proxy (footprint of the per-face max) is also
available; it biases warm at shadow-sweeping PSR edges.

Diviner T_max_bol is block-averaged onto the same grid. We report difference
maps, a model-vs-Diviner scatter, residual stats (bias, RMSE, R2) overall and
split by regime (PSR vs sunlit), in the pretty-plots style.

    python3 diviner_compare.py \
        --diviner .../Tbols_SP_*.mat \
        --run-dir output/divval_shackleton \
        --center-lat -89.93 --center-lon 20.2 \
        --window-km 5 --cell-m 120 --out docs/figures/diviner_validation/shackleton
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import distance_transform_edt


# Fixed publication temperature scale for the bolometric-T maps (K).
T_LO, T_HI = 25.0, 225.0


def fill_nan_nearest(grid, mask):
    """Fill NaN cells of `grid` that lie inside `mask` with the nearest valid
    value (nearest-neighbor), for gap-free display of the sparsely-sampled model
    grid. Returns a copy; metrics are computed on the original, unfilled grid."""
    out = np.asarray(grid, float).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = out[tuple(idx)]
    holes = (~valid) & mask
    out[holes] = filled[holes]
    return out

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, panel_label
from diviner_data import load_diviner, latlon_to_diviner_xy, KM_PER_DEG

_PLY_T = {b"double": "<f8", b"float": "<f4", b"int": "<i4", b"uint": "<u4",
          b"uchar": "<u1", b"char": "<i1", b"short": "<i2", b"ushort": "<u2"}


def read_ply(path: str):
    """Binary PLY -> (verts double Nx3, faces Nx3 int, face_lat, face_lon|None)."""
    with open(path, "rb") as fh:
        assert fh.readline().strip() == b"ply"
        assert b"binary_little_endian" in fh.readline()
        elements, cur = [], None
        while True:
            tok = fh.readline().split()
            if not tok:
                continue
            if tok[0] == b"element":
                cur = {"name": tok[1], "count": int(tok[2]), "props": []}
                elements.append(cur)
            elif tok[0] == b"property":
                if tok[1] == b"list":
                    cur["props"].append(("list", tok[2], tok[3], tok[4]))
                else:
                    cur["props"].append(("scalar", tok[1], tok[2]))
            elif tok[0] == b"end_header":
                break
        verts = faces = flat = flon = None
        for el in elements:
            if el["name"] == b"vertex":
                dt = np.dtype([(p[2].decode(), _PLY_T[p[1]]) for p in el["props"]])
                rec = np.frombuffer(fh.read(el["count"] * dt.itemsize), dt, el["count"])
                verts = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
            elif el["name"] == b"face":
                # list prop (uchar count, int idx) + optional scalar lat/lon
                scalars = [p for p in el["props"] if p[0] == "scalar"]
                f = np.empty((el["count"], 3), np.int64)
                sc = {p[2].decode(): np.empty(el["count"], _PLY_T[p[1]]) for p in scalars}
                cnt_t = np.dtype("<u1"); idx_t = np.dtype("<i4")
                for i in range(el["count"]):
                    n = np.frombuffer(fh.read(1), cnt_t, 1)[0]
                    f[i] = np.frombuffer(fh.read(4 * n), idx_t, n)[:3]
                    for p in scalars:
                        nm = p[2].decode()
                        sc[nm][i] = np.frombuffer(fh.read(np.dtype(_PLY_T[p[1]]).itemsize),
                                                  _PLY_T[p[1]], 1)[0]
                faces = f
                flat = sc.get("latitude"); flon = sc.get("longitude")
    return verts, faces, flat, flon


def face_geometry(verts, faces, flat, flon):
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    if flat is None:
        cen = (p0 + p1 + p2) / 3.0
        r = np.linalg.norm(cen, axis=1)
        flat = np.degrees(np.arcsin(np.clip(cen[:, 2] / r, -1, 1)))
        flon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
    return area, np.asarray(flat, float), np.asarray(flon, float)


def load_step(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)[1:]


def block_average(a, fx, fy):
    """Average a (ny,nx) onto (ny//fy, nx//fx) ignoring NaN."""
    ny, nx = a.shape
    a = a[: ny // fy * fy, : nx // fx * fx]
    m = np.isfinite(a)
    s = np.where(m, a, 0).reshape(a.shape[0] // fy, fy, a.shape[1] // fx, fx).sum((1, 3))
    c = m.reshape(a.shape[0] // fy, fy, a.shape[1] // fx, fx).sum((1, 3))
    out = np.full(s.shape, np.nan, float)
    np.divide(s, c, out=out, where=c > 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diviner", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--window-km", type=float, default=5.0)
    ap.add_argument("--cell-m", type=float, default=120.0)
    ap.add_argument("--core-radius-km", type=float, default=5.0)
    ap.add_argument("--emissivity", type=float, default=0.95)
    ap.add_argument("--max-mode", choices=["rigorous", "perface"], default="rigorous")
    ap.add_argument("--psr-threshold", type=float, default=200.0,
                    help="Diviner Tmax below this = PSR regime (K)")
    ap.add_argument("--out", default="docs/figures/diviner_validation/compare")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    eps4 = a.emissivity ** 0.25
    label = a.label or run.name

    # --- comparison grid in the Diviner azimuthal-equidistant frame (deg) ---
    xd0, yd0 = latlon_to_diviner_xy(a.center_lat, a.center_lon)
    cell_deg = (a.cell_m / 1000.0) / KM_PER_DEG
    W_deg = a.window_km / KM_PER_DEG
    edges_x = np.arange(xd0 - W_deg, xd0 + W_deg + cell_deg, cell_deg)
    edges_y = np.arange(yd0 - W_deg, yd0 + W_deg + cell_deg, cell_deg)
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    CX, CY = np.meshgrid(cx, cy)
    cell_r = np.sqrt((CX - xd0) ** 2 + (CY - yd0) ** 2) * KM_PER_DEG   # km from center
    core_cell = cell_r <= a.core_radius_km

    # --- model: faces -> bins (projected into the Diviner deg frame) --------
    verts, faces, flat, flon = read_ply(str(run / "mesh.ply"))
    area, flat, flon = face_geometry(verts, faces, flat, flon)
    fxd, fyd = latlon_to_diviner_xy(flat, flon)
    rkm = np.sqrt((fxd - xd0) ** 2 + (fyd - yd0) ** 2) * KM_PER_DEG
    ix = np.floor((fxd - edges_x[0]) / cell_deg).astype(int)
    iy = np.floor((fyd - edges_y[0]) / cell_deg).astype(int)
    keep = (rkm <= a.core_radius_km) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = (iy[keep] * nx + ix[keep])
    A = area[keep]
    Asum = np.bincount(lin, weights=A, minlength=nx * ny)
    print(f"[{label}] {keep.sum()} core faces -> {int((Asum>0).sum())} populated cells "
          f"({a.cell_m:.0f} m); window +/-{a.window_km} km")

    model_tbol = np.full(nx * ny, np.nan)
    if a.max_mode == "rigorous":
        steps = sorted((run / "temperature").glob("temperature_*.bin"),
                       key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
        assert steps, "no per-step temperatures (run with skip_time_series:false)"
        best = np.zeros(nx * ny)
        for k, sp in enumerate(steps):
            T = load_step(sp)[keep]
            num = np.bincount(lin, weights=A * (T.astype(np.float64) ** 4), minlength=nx * ny)
            with np.errstate(invalid="ignore", divide="ignore"):
                tbol = eps4 * np.where(Asum > 0, num / Asum, 0.0) ** 0.25
            best = np.maximum(best, tbol)
        model_tbol = np.where(Asum > 0, best, np.nan)
        print(f"[{label}] rigorous max over {len(steps)} timesteps")
    else:
        Tmax = load_step(run / "statistics" / "temperature_max.bin")[keep]
        num = np.bincount(lin, weights=A * (Tmax.astype(np.float64) ** 4), minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            model_tbol = eps4 * np.where(Asum > 0, num / Asum, np.nan) ** 0.25
    model_tbol = model_tbol.reshape(ny, nx)

    # --- Diviner: bin its native pixels (own x,y deg) into the same cells ---
    g = load_diviner(a.diviner)
    jx = np.searchsorted(g.x_deg, [edges_x[0], edges_x[-1]])
    jy = np.searchsorted(g.y_deg, [edges_y[0], edges_y[-1]])
    sx = slice(max(0, int(jx[0]) - 1), min(len(g.x_deg), int(jx[1]) + 1))
    sy = slice(max(0, int(jy[0]) - 1), min(len(g.y_deg), int(jy[1]) + 1))
    DX, DY = np.meshgrid(g.x_deg[sx], g.y_deg[sy])
    DT = g.t_max[sy, sx]
    dok = np.isfinite(DT)
    dix = np.floor((DX[dok] - edges_x[0]) / cell_deg).astype(int)
    diy = np.floor((DY[dok] - edges_y[0]) / cell_deg).astype(int)
    din = (diy >= 0) & (diy < ny) & (dix >= 0) & (dix < nx)
    dlin = diy[din] * nx + dix[din]
    dsum = np.bincount(dlin, weights=DT[dok][din].astype(np.float64), minlength=nx * ny)
    dcnt = np.bincount(dlin, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        div = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), np.nan).reshape(ny, nx)

    # --- residuals + metrics ------------------------------------------------
    valid = core_cell & np.isfinite(model_tbol) & np.isfinite(div)
    res = (model_tbol - div)[valid]
    dv = div[valid]
    def stats(mask):
        r = res[mask]
        if r.size < 3: return dict(n=int(r.size), bias=np.nan, rmse=np.nan, r2=np.nan)
        return dict(n=int(r.size), bias=float(r.mean()), rmse=float(np.sqrt((r**2).mean())),
                    r2=float(1 - np.var(r) / max(np.var((model_tbol[valid])[mask]), 1e-9)))
    overall = stats(np.ones(res.shape, bool))
    psr = stats(dv < a.psr_threshold); sun = stats(dv >= a.psr_threshold)
    print(f"[{label}] N={overall['n']}  bias={overall['bias']:+.1f} K  RMSE={overall['rmse']:.1f} K")
    print(f"   PSR  (Div<{a.psr_threshold:.0f}K): n={psr['n']} bias={psr['bias']:+.1f} RMSE={psr['rmse']:.1f}")
    print(f"   sunlit          : n={sun['n']} bias={sun['bias']:+.1f} RMSE={sun['rmse']:.1f}")

    # --- figure -------------------------------------------------------------
    apply_style()
    ext = [(edges_x[0]-xd0)*KM_PER_DEG, (edges_x[-1]-xd0)*KM_PER_DEG,
           (edges_y[0]-yd0)*KM_PER_DEG, (edges_y[-1]-yd0)*KM_PER_DEG]
    # Gap-free display copies (metrics already computed on the unfilled grids).
    model_disp = fill_nan_nearest(model_tbol, core_cell)
    diff_disp = np.where(np.isfinite(div), model_disp - div, np.nan)
    dm = np.nanpercentile(np.abs((model_tbol - div)[valid]), 98)
    # Per-site robust color scale (2-98th pct of combined Diviner+PTM3D in the
    # core), shared by both temperature panels to enhance PSR contrast.
    both = np.concatenate([div[core_cell & np.isfinite(div)],
                           model_tbol[core_cell & np.isfinite(model_tbol)]])
    vlo, vhi = (np.nanpercentile(both, [2, 98]) if both.size else (T_LO, T_HI))
    if not (vhi > vlo):
        vlo, vhi = T_LO, T_HI
    tnorm = Normalize(vlo, vhi)
    panels = [(div, "Diviner $T_{bol,max}$", "inferno", tnorm, "$T$ (K)"),
              (model_disp, "PTM3D $T_{bol,max}$", "inferno", tnorm, "$T$ (K)"),
              (diff_disp, "model $-$ Diviner", "RdBu_r", Normalize(-dm, dm), "$\\Delta T$ (K)")]
    fig = plt.figure(figsize=(7.6, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.46, wspace=0.42)
    for j, (arr, ttl, cmap, norm, clab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(np.where(core_cell, arr, np.nan), origin="lower", extent=ext,
                       cmap=cmap, norm=norm)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("x (km)")
        if j == 0:
            ax.set_ylabel("y (km)")
        else:
            ax.tick_params(labelleft=False)          # avoid colorbar/tick collisions
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clab, fontsize=9, labelpad=2)
        cb.ax.tick_params(labelsize=8)
        panel_label(ax, f"({'abc'[j]})")
    # scatter (tight data range, not the map scale)
    axs = fig.add_subplot(gs[1, 0:2])
    axs.hexbin(dv, model_tbol[valid], gridsize=45, cmap="viridis", mincnt=1, bins="log")
    vmin_d = float(np.nanmin([dv.min(), model_tbol[valid].min()]))
    vmax_d = float(np.nanmax([dv.max(), model_tbol[valid].max()]))
    pad = 0.04 * (vmax_d - vmin_d)
    lo, hi = vmin_d - pad, vmax_d + pad
    axs.plot([lo, hi], [lo, hi], "k--", lw=1)
    axs.set_xlabel("Diviner $T_{bol,max}$ (K)")
    axs.set_ylabel("PTM3D $T_{bol,max}$ (K)")
    axs.set_xlim(lo, hi); axs.set_ylim(lo, hi); panel_label(axs, "(d)")
    axs.text(0.04, 0.96, f"N={overall['n']}\nbias={overall['bias']:+.1f} K\nRMSE={overall['rmse']:.1f} K",
             transform=axs.transAxes, va="top", fontsize=9,
             bbox=dict(fc="white", ec="0.7", alpha=0.85))
    # residual histogram
    axh = fig.add_subplot(gs[1, 2])
    axh.hist(res, bins=40, color="#1f77b4", alpha=0.85)
    axh.axvline(0, color="k", lw=1); axh.axvline(overall["bias"], color="r", ls="--", lw=1.2)
    axh.set_xlabel("residual (K)"); axh.set_ylabel("cells", labelpad=2)
    panel_label(axh, "(e)")
    fig.suptitle(f"PTM3D vs Diviner max bolometric T — {label} "
                 f"({a.cell_m:.0f} m cells, {a.max_mode})", fontsize=12, fontweight="bold")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext_fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(ext_fmt), dpi=dpi, bbox_inches="tight")
    print(f"[{label}] wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
