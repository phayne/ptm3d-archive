#!/usr/bin/env python3
"""The falsifying test for the smith_beaming.h chi-table parametrization.

QUESTION: is the chi lookup table accurate on the geometry that actually delivers
wall->floor IR flux at a lunar pole?

This ports src/cpp/radiation/smith_beaming.h verbatim (Nq=33 Gaussian-slope
quadrature, G1 masking, Z normalization over Nt=24 x Np=48, clamp [0,4]) and
compares, on a MEASURED flux-weighted (i, e, phi) pair distribution:
  * EXACT   chi(cos_sun, cos_e, cos_psi) evaluated pointwise, against
  * LOOKUP  trilinear interpolation of the flat table evalChi() reads,
for BOTH third-axis parametrizations -- the retired cos_psi one and the cos_phi
one the header now uses -- at several grid resolutions.

ANSWER (w = 32.5 deg, 5.3M measured pairs): exact 1.657; cos_psi-indexed
24x24x33 reads 1.595 (-3.8%, p95 pointwise 24%) and does NOT converge with
refinement (64x128x129 still reads 1.617), because most cells of a regular
(cos_sun, cos_e, cos_psi) grid are geometrically unreachable and hold chi = 1,
and the polar wall->floor flux sits right on that boundary.  Re-indexed on
cos_phi the SAME 24x24x33 reads 1.652 (-0.33%) and 48x48x65 reads 1.655
(-0.12%).  That is why the header changed axis, not resolution.

INPUT: --pairs is an .npy of shape (N, >=6) with columns
  2 = the IR flux the pair delivers (the weight),
  3 = cos(incidence on the emitting wall),
  4 = cos(emission angle from the wall toward the receiver),
  5 = cos(phase angle) = d . s_hat,
produced by the physics review's calc10_geom.py from a run's own view-factor matrix and
temperature field.  It is a multi-hundred-MB measurement, NOT in the repo; the
default path is where the 2026-08-24 review left it.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from scipy.special import erfc

DEFAULT_PAIRS = Path(os.environ.get("PTM_CHI_PAIRS", "calc10_pairs.npy"))
PAIRS = DEFAULT_PAIRS
W_DEG = 32.5
NQ = 33
NT, NP = 24, 48

U = np.linspace(-4.0, 4.0, NQ)
GW = np.exp(-0.5 * U * U)
GW /= GW.sum()


def lam(a):
    a = np.maximum(a, 1e-6)
    out = 0.5 * (np.exp(-a * a) / (a * np.sqrt(np.pi)) - erfc(a))
    return np.where(a > 8.0, 0.0, out)


def G1(ct, w):
    ct = np.clip(ct, 0.0, 1.0)
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    return np.where(ct <= 0.0, 0.0,
                    np.where(ct >= 1.0, 1.0,
                             1.0 / (1.0 + lam(ct / np.maximum(st, 1e-9) / (np.sqrt(2.0) * w)))))


def micro(w):
    """Micro-facet normals m and their Gaussian weights (the header's `integral`)."""
    P, Q = np.meshgrid(w * U, w * U, indexing="ij")
    Wt = np.outer(GW, GW)
    inv = 1.0 / np.sqrt(1.0 + P * P + Q * Q)
    M = np.stack([-P * inv, -Q * inv, inv], -1).reshape(-1, 3)
    return M, Wt.ravel()


def I_of(s, d, M, Wt):
    ms = M @ s
    md = M @ d
    return float((Wt * np.maximum(ms, 0.0) * np.maximum(md, 0.0)).sum())


def Z_of(cs, w, M, Wt):
    ss = np.sqrt(max(0.0, 1.0 - cs * cs))
    s = np.array([ss, 0.0, cs])
    Z = 0.0
    dt = (np.pi / 2) / NT
    dp = 2 * np.pi / NP
    for it in range(NT):
        te = (it + 0.5) * dt
        ce, se = np.cos(te), np.sin(te)
        g = G1(ce, w)
        for ip in range(NP):
            ph = (ip + 0.5) * dp
            d = np.array([se * np.cos(ph), se * np.sin(ph), ce])
            Z += g * I_of(s, d, M, Wt) * se * dt * dp
    return max(Z, 1e-12)


def chi_point(cs, ce, cp, w, M, Wt, Zc):
    """smith::buildChiTable's inner expression, at an arbitrary (cs, ce, cp)."""
    ss = np.sqrt(max(0.0, 1.0 - cs * cs))
    se = np.sqrt(max(0.0, 1.0 - ce * ce))
    den = ss * se
    if den <= 1e-9:
        return 1.0
    cphi = (cp - cs * ce) / den
    if cphi < -1.0 or cphi > 1.0:
        return 1.0
    sphi = np.sqrt(max(0.0, 1.0 - cphi * cphi))
    d = np.array([se * cphi, se * sphi, ce])
    s = np.array([ss, 0.0, cs])
    x = np.pi * G1(ce, w) * I_of(s, d, M, Wt) / (ce * Zc)
    return float(min(4.0, max(0.0, x)))


def build_table(ns, ne, npsi, w, M, Wt):
    """smith::buildChiTable(w, ns, ne, npsi)."""
    tab = np.ones((ns, ne, npsi))
    for i_s in range(ns):
        cs = (i_s + 0.5) / ns
        Zc = Z_of(cs, w, M, Wt)
        for ie in range(ne):
            ce = (ie + 0.5) / ne
            for ip in range(npsi):
                cp = -1.0 + 2.0 * (ip + 0.5) / npsi
                tab[i_s, ie, ip] = chi_point(cs, ce, cp, w, M, Wt, Zc)
    return tab


def eval_table(tab, cs, ce, cp):
    """smith::evalChi -- trilinear, clamped to cell centers (vectorized)."""
    ns, ne, npsi = tab.shape
    out = np.ones_like(cs)
    act = (cs > 0.0) & (ce > 0.0)
    fs = np.clip(cs * ns - 0.5, 0.0, ns - 1.0)
    fe = np.clip(ce * ne - 0.5, 0.0, ne - 1.0)
    fp = np.clip((cp + 1.0) * 0.5 * npsi - 0.5, 0.0, npsi - 1.0)
    is0 = fs.astype(int); ie0 = fe.astype(int); ip0 = fp.astype(int)
    is1 = np.minimum(is0 + 1, ns - 1)
    ie1 = np.minimum(ie0 + 1, ne - 1)
    ip1 = np.minimum(ip0 + 1, npsi - 1)
    ds = fs - is0; de = fe - ie0; dp = fp - ip0
    c00 = tab[is0, ie0, ip0] * (1 - dp) + tab[is0, ie0, ip1] * dp
    c01 = tab[is0, ie1, ip0] * (1 - dp) + tab[is0, ie1, ip1] * dp
    c10 = tab[is1, ie0, ip0] * (1 - dp) + tab[is1, ie0, ip1] * dp
    c11 = tab[is1, ie1, ip0] * (1 - dp) + tab[is1, ie1, ip1] * dp
    c0 = c00 * (1 - de) + c01 * de
    c1 = c10 * (1 - de) + c11 * de
    return np.where(act, c0 * (1 - ds) + c1 * ds, out)


def main(nsamp=4000):
    w = np.radians(W_DEG)
    M, Wt = micro(w)
    R = np.load(PAIRS, mmap_mode="r")
    wflux = np.asarray(R[:, 2], float)
    P = wflux / wflux.sum()
    rng = np.random.default_rng(7)
    idx = rng.choice(len(R), size=nsamp, replace=True, p=P)
    cs = np.asarray(R[idx, 3], float)
    ce = np.asarray(R[idx, 4], float)
    cp = np.asarray(R[idx, 5], float)
    keep = (cs > 0) & (ce > 0)
    cs, ce, cp = cs[keep], ce[keep], cp[keep]
    n = len(cs)
    print(f"# flux-weighted draws with cos_sun>0 and cos_e>0: {n} of {nsamp}")
    print(f"# cos_sun  p05={np.percentile(cs,5):.4f} p50={np.percentile(cs,50):.4f} "
          f"p95={np.percentile(cs,95):.4f}")
    print(f"# cos_e    p05={np.percentile(ce,5):.4f} p50={np.percentile(ce,50):.4f} "
          f"p95={np.percentile(ce,95):.4f}")
    print(f"# cos_psi  p05={np.percentile(cp,5):.4f} p50={np.percentile(cp,50):.4f} "
          f"p95={np.percentile(cp,95):.4f}")

    # EXACT: Z on a fine cos_sun grid (Z is smooth; 200 nodes is well past converged)
    csg = np.linspace(0.0025, 0.9975, 200)
    Zt = np.array([Z_of(c, w, M, Wt) for c in csg])
    exact = np.empty(n)
    for k in range(n):
        exact[k] = chi_point(cs[k], ce[k], cp[k], w, M, Wt,
                             float(np.interp(cs[k], csg, Zt)))
    print(f"\nEXACT flux-weighted mean chi (w={W_DEG} deg) = {exact.mean():.4f}")

    print(f"\n{'grid ns x ne x npsi':>22}  {'mean chi':>9}  {'bias vs exact':>13}  "
          f"{'|err| p50':>9}  {'|err| p95':>9}  {'|err| max':>9}")
    for (ns, ne, npsi) in [(24, 24, 33), (24, 48, 33), (48, 48, 65),
                           (48, 96, 65), (48, 96, 129), (64, 128, 129)]:
        tab = build_table(ns, ne, npsi, w, M, Wt)
        got = eval_table(tab, cs, ce, cp)
        err = got - exact
        rel = np.abs(err) / np.maximum(exact, 1e-9)
        print(f"{ns:6d} x{ne:4d} x{npsi:4d}  {got.mean():9.4f}  "
              f"{got.mean()-exact.mean():+13.4f}  "
              f"{np.percentile(rel,50)*100:8.3f}%  {np.percentile(rel,95)*100:8.3f}%  "
              f"{rel.max()*100:8.3f}%")




# ---------------------------------------------------------------------------
# Diagnosis arm: the shipped table is indexed by cos_psi, but chi is only
# DEFINED where |cphi| = |(cos_psi - cos_sun cos_e)/(sin_sun sin_e)| <= 1.
# On a regular (cos_sun, cos_e, cos_psi) grid most cells are outside that
# admissible wedge and are filled with chi = 1, so every trilinear stencil that
# straddles the boundary is dragged toward 1 -- and the measured wall->floor
# flux sits right ON that boundary (near-zero phase, cphi ~ +1).  Re-index the
# SAME chi on (cos_sun, cos_e, cphi): the domain becomes the full box, there are
# no dead cells, and chi is smooth in cphi.
# ---------------------------------------------------------------------------
def build_table_cphi(ns, ne, nph, w, M, Wt):
    tab = np.ones((ns, ne, nph))
    for i_s in range(ns):
        cs = (i_s + 0.5) / ns
        ss = np.sqrt(max(0.0, 1.0 - cs * cs))
        Zc = Z_of(cs, w, M, Wt)
        s = np.array([ss, 0.0, cs])
        for ie in range(ne):
            ce = (ie + 0.5) / ne
            se = np.sqrt(max(0.0, 1.0 - ce * ce))
            g1 = G1(ce, w)
            for ip in range(nph):
                cphi = -1.0 + 2.0 * (ip + 0.5) / nph
                sphi = np.sqrt(max(0.0, 1.0 - cphi * cphi))
                d = np.array([se * cphi, se * sphi, ce])
                x = np.pi * g1 * I_of(s, d, M, Wt) / (ce * Zc)
                tab[i_s, ie, ip] = min(4.0, max(0.0, x))
    return tab


def eval_table_cphi(tab, cs, ce, cp):
    ss = np.sqrt(np.maximum(0.0, 1.0 - cs * cs))
    se = np.sqrt(np.maximum(0.0, 1.0 - ce * ce))
    den = ss * se
    cphi = np.where(den > 1e-9, (cp - cs * ce) / np.maximum(den, 1e-30), 0.0)
    ok = (den > 1e-9) & (cphi >= -1.0) & (cphi <= 1.0) & (cs > 0) & (ce > 0)
    cphi = np.clip(cphi, -1.0, 1.0)
    # same trilinear kernel, third axis now cphi in [-1, 1]
    v = eval_table(tab, cs, ce, cphi)
    return np.where(ok, v, 1.0)


def arm2(nsamp=4000):
    w = np.radians(W_DEG)
    M, Wt = micro(w)
    R = np.load(PAIRS, mmap_mode="r")
    wflux = np.asarray(R[:, 2], float); P = wflux / wflux.sum()
    rng = np.random.default_rng(7)
    idx = rng.choice(len(R), size=nsamp, replace=True, p=P)
    cs = np.asarray(R[idx, 3], float); ce = np.asarray(R[idx, 4], float)
    cp = np.asarray(R[idx, 5], float)
    keep = (cs > 0) & (ce > 0); cs, ce, cp = cs[keep], ce[keep], cp[keep]
    csg = np.linspace(0.0025, 0.9975, 200)
    Zt = np.array([Z_of(c, w, M, Wt) for c in csg])
    exact = np.array([chi_point(cs[k], ce[k], cp[k], w, M, Wt,
                                float(np.interp(cs[k], csg, Zt)))
                      for k in range(len(cs))])
    print(f"\n--- re-indexed on cphi (azimuth cosine) --- exact = {exact.mean():.4f}")
    print(f"{'grid ns x ne x nph':>22}  {'mean chi':>9}  {'bias vs exact':>13}  "
          f"{'|err| p50':>9}  {'|err| p95':>9}  {'|err| max':>9}")
    for (ns, ne, nph) in [(24, 24, 33), (24, 48, 33), (48, 48, 65), (48, 96, 65)]:
        tab = build_table_cphi(ns, ne, nph, w, M, Wt)
        got = eval_table_cphi(tab, cs, ce, cp)
        err = got - exact
        rel = np.abs(err) / np.maximum(exact, 1e-9)
        print(f"{ns:6d} x{ne:4d} x{nph:4d}  {got.mean():9.4f}  "
              f"{got.mean()-exact.mean():+13.4f}  "
              f"{np.percentile(rel,50)*100:8.3f}%  {np.percentile(rel,95)*100:8.3f}%  "
              f"{rel.max()*100:8.3f}%")


def _cli():
    global PAIRS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                    help="measured flux-weighted pair distribution (.npy); see module docstring")
    ap.add_argument("--samples", type=int, default=4000,
                    help="flux-weighted draws (the exact arm is O(samples) integrals)")
    a = ap.parse_args()
    if not a.pairs.exists():
        raise SystemExit(f"FATAL: pair distribution not found: {a.pairs}\n"
                         "It is a measurement, not a repo artifact -- regenerate it with "
                         "a pair-geometry script (calc10_geom.py) against the run you want to characterize, "
                         "or point --pairs at an existing copy. Refusing to invent one.")
    PAIRS = a.pairs
    main(a.samples)
    arm2(a.samples)


if __name__ == "__main__":
    _cli()
