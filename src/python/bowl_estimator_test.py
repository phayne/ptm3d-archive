#!/usr/bin/env python3
"""Decompose the floor view-factor error on the analytic bowl harness across
estimator choices, to design the production fix. Known answer (smooth bowl):
F_terrain(center) = sin^2(beta_c); rim-band F_far ~ that minus the near share.

Estimators compared (floor -> terrain, per floor face, summed):
  form factor:  M1 (point)   vs  M2 (Rezac & Zhao arctan, bounded near-field)
  occlusion:    OFF  vs  ON (heightfield running-max-angle = ray-equivalent)
Run on smooth & rough bowls at coarse & fine resolution.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_bowl import build  # noqa: E402

DIAM = 20000.0
CELL = 60.0
DSTEP = 60.0
N_AZ = 180


def faces(dd, nr, na, rough):
    V, Fc, vn, info = build(dd, DIAM, nr, na, 14, 1.7, rough, 0)
    p0, p1, p2 = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    c = (p0 + p1 + p2) / 3.0
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    fn = vn[Fc[:, 0]] + vn[Fc[:, 1]] + vn[Fc[:, 2]]
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30
    return c, fn, A, info


def heightfield(c):
    emin, nmin = c[:, 0].min(), c[:, 1].min()
    ne = int((c[:, 0].max() - emin) / CELL) + 2
    nn = int((c[:, 1].max() - nmin) / CELL) + 2
    H = np.full((ne, nn), -1e9)
    ie = ((c[:, 0] - emin) / CELL).astype(int)
    iN = ((c[:, 1] - nmin) / CELL).astype(int)
    np.maximum.at(H, (ie, iN), c[:, 2])
    return H, emin, nmin, ne, nn


def run(dd, nr, na, rough):
    c, fn, A, info = faces(dd, nr, na, rough)
    nf = len(A)
    H, emin, nmin, ne, nn = heightfield(c)
    sgrid = np.arange(DSTEP, 11000.0, DSTEP)
    rho = np.hypot(c[:, 0], c[:, 1])
    r_rim = DIAM / 2.0
    floor = np.where(rho < 0.15 * r_rim)[0]
    probes = floor[:: max(1, len(floor) // 60)]
    rim = rho > 0.6 * r_rim                                   # "far" rim band

    out = {k: [] for k in ["M1_off", "M2_off", "M1_on", "M2_on",
                           "M1_on_far", "M2_on_far"]}
    for pf in probes:
        ci, ni, Ai = c[pf], fn[pf], A[pf]
        de = c[:, 0] - ci[0]; dn = c[:, 1] - ci[1]; du = c[:, 2] - ci[2]
        hd = np.hypot(de, dn)
        d = np.sqrt(de * de + du * du + dn * dn)
        sel = d > 1.0
        rh = np.stack([de, dn, du], 1) / (d[:, None] + 1e-30)
        cos_i = rh @ ni
        cos_j = -(rh * fn).sum(1)
        cull = sel & (cos_i > 1e-3) & (cos_j > 1e-3)
        # form factors
        M1 = np.zeros(nf); M2 = np.zeros(nf)
        cc = cull
        M1[cc] = (A[cc] / np.pi) * (cos_i[cc] * cos_j[cc]) / (d[cc] ** 2)
        ti = np.arctan(np.sqrt(np.pi * Ai) * cos_i[cc] / (2 * d[cc]))
        tj = np.arctan(np.sqrt(np.pi * A[cc]) * cos_j[cc] / (2 * d[cc]))
        M2[cc] = (4 * np.sqrt(Ai * A[cc]) / (np.pi ** 2 * Ai)) * ti * tj
        # occlusion: per-azimuth running-max terrain angle, then per-target lookup
        hor = np.full((N_AZ, len(sgrid)), -np.pi / 2)
        azs = np.linspace(0, 2 * np.pi, N_AZ, endpoint=False)
        for ai, az in enumerate(azs):
            xs = ci[0] + sgrid * np.cos(az); ys = ci[1] + sgrid * np.sin(az)
            cei = ((xs - emin) / CELL).astype(int); cni = ((ys - nmin) / CELL).astype(int)
            ok = (cei >= 0) & (cei < ne) & (cni >= 0) & (cni < nn)
            ang = np.full(len(sgrid), -np.pi / 2)
            hz = np.full(ok.sum(), -1e9); hz[:] = H[cei[ok], cni[ok]]
            good = hz > -1e8
            aa = np.full(ok.sum(), -np.pi / 2)
            aa[good] = np.arctan2(hz[good] - ci[2], sgrid[ok][good])
            ang[ok] = aa
            hor[ai] = np.maximum.accumulate(ang)
        azt = np.arctan2(dn, de) % (2 * np.pi)
        tht = np.arctan2(du, hd + 1e-30)
        ait = (azt / (2 * np.pi) * N_AZ).astype(int) % N_AZ
        dit = np.clip((hd / DSTEP).astype(int) - 1, 0, len(sgrid) - 1)
        occluded = tht < (hor[ait, dit] - np.deg2rad(0.25))
        vis = cull & ~occluded
        out["M1_off"].append(M1[cull].sum()); out["M2_off"].append(M2[cull].sum())
        out["M1_on"].append(M1[vis].sum());  out["M2_on"].append(M2[vis].sum())
        out["M1_on_far"].append(M1[vis & rim].sum()); out["M2_on_far"].append(M2[vis & rim].sum())
    m = {k: float(np.mean(v)) for k, v in out.items()}
    return m, info


def main():
    print(f"{'case':28s} {'F_an':>7s} {'M1off':>8s} {'M2off':>8s} {'M1on':>8s} {'M2on':>8s} {'M1on_far':>9s} {'M2on_far':>9s}")
    for rough in (0.0, 10.0):
        for nr, na, lbl in ((36, 96, "coarse"), (144, 384, "fine")):
            m, info = run(0.05, nr, na, rough)
            print(f"dd0.05 rough{rough:>4.0f} {lbl:7s}     {info['F_terrain']:7.4f} "
                  f"{m['M1_off']:8.4f} {m['M2_off']:8.4f} {m['M1_on']:8.4f} {m['M2_on']:8.4f} "
                  f"{m['M1_on_far']:9.4f} {m['M2_on_far']:9.4f}")
    print("F_an = smooth analytic floor F_terrain (sin^2 beta_c). Rim-band ~0.006 of it.")


if __name__ == "__main__":
    main()
