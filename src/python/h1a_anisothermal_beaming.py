#!/usr/bin/env python3
"""H1a simple model: anisothermal-wall thermal beaming toward a PSR crater floor.

Hypothesis (see docs/psr-floor-warmbias-hypotheses.md, H1): the sunlit far inner
wall of a polar crater is NOT isothermal. Sub-resolution rough micro-facets that
face the grazing Sun run far hotter than the wall's mean; shadowed micro-facets are
cold. Because the floor views the wall at a small phase angle (~10 deg, backscatter /
opposition), it preferentially sees the HOT Sun-facing micro-facets -> the wall's
effective brightness temperature toward the floor exceeds its mean. The production
Lambertian radiosity uses the facet MEAN T and misses this; the Smith (1967) term
captures the angular redistribution but assumes ISOTHERMAL facets, so it gets only
the weak +0.5 K, not the (T_hot/T_mean)^4 contrast.

This script quantifies the enhancement with a Gaussian micro-slope ensemble:
  - micro-facet normals m ~ Gaussian slope distribution (RMS slope w) about the
    macroscopic wall normal n_hat;
  - each facet absorbs direct sun (1-A) S max(0, m.s) * G1(m.s; w)  [Smith masking]
    plus a uniform background Q_bg (crater IR keeping shadowed facets warm);
  - facet temperature from instantaneous radiative equilibrium: eps sigma T^4 = absorbed;
  - directional emission toward the floor weights each facet by max(0, m.d) * G1(m.d; w);
  - ENHANCEMENT E = <T^4 * w_e> / (<T^4> * <w_e>)  (anisothermal vs isothermal-at-mean
    directional brightness toward the floor); E>1 means the floor sees the wall hotter
    than its mean.

Geometry is the Haworth deep floor at summer peak: Sun elevation e=4.1 deg, crater
d/D=0.051 -> rim/floor phase ~10 deg. We also sweep wall slope, RMS slope, background
fraction, and phase angle. A separate block estimates the reflected-SOLAR opposition
contribution (currently Lambertian + under-fed in the model).

Pure numpy; no PTM3D output needed. Run:  python3 h1a_anisothermal_beaming.py
"""
from __future__ import annotations
import numpy as np
try:
    from scipy.special import erfc as _erfc_v
except Exception:                       # fallback: vectorized math.erfc
    from math import erfc as _erfc_s
    _erfc_v = np.vectorize(_erfc_s)

SIGMA = 5.670374e-8
S = 1361.0          # solar constant [W/m^2]
EPS = 0.95
A_BOND = 0.12       # solar Bond albedo


# ---- Smith (1967) Gaussian-slope masking/shadowing G1 (ported from smith_beaming.h)
def smith_lambda(a):
    a = np.asarray(a, float)
    out = np.zeros_like(a)
    m = a < 8.0
    aa = np.maximum(a[m], 1e-6)
    out[m] = 0.5 * (np.exp(-aa * aa) / (aa * np.sqrt(np.pi)) - _erfc_v(aa))
    return out


def G1(cos_t, w):
    """Monostatic Smith masking G1(theta)=1/(1+Lambda), cos_t in (0,1]."""
    cos_t = np.asarray(cos_t, float)
    out = np.zeros_like(cos_t)
    good = cos_t > 0.0
    ct = np.clip(cos_t[good], 1e-9, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    cot = ct / np.maximum(1e-9, sin_t)
    a = cot / (np.sqrt(2.0) * w)
    out[good] = 1.0 / (1.0 + smith_lambda(a))
    out[cos_t >= 1.0] = 1.0
    return out


def ortho_basis(n):
    """Two unit tangents orthogonal to unit vector n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = a - n * np.dot(a, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def geometry(sun_elev_deg, wall_slope_deg, d_over_D):
    """Return (n_hat wall normal, s_hat to-Sun, d_hat wall->floor), all global,
    plus the macroscopic incidence/emission angles and the phase angle.

    Global frame: +x = Sun azimuth (horizontal), +z = up. Sun in the x-z plane.
    Far inner wall faces +x (toward Sun); its normal is tilted `wall_slope` from
    vertical toward +x. Floor is at crater center, the wall at the rim; the
    wall->floor direction drops by atan(2 d/D) over the radius."""
    e = np.radians(sun_elev_deg)
    s_hat = np.array([np.cos(e), 0.0, np.sin(e)])              # toward Sun
    ws = np.radians(wall_slope_deg)
    n_hat = np.array([np.sin(ws), 0.0, np.cos(ws)])            # inner-wall normal (+x, up)
    drop = np.arctan(2.0 * d_over_D)                            # rim->floor depression
    d_hat = np.array([np.cos(drop), 0.0, -np.sin(drop)])       # wall->floor (+x, down)
    mu_s = float(np.dot(n_hat, s_hat))
    mu_e = float(np.dot(n_hat, d_hat))
    phase = np.degrees(np.arccos(np.clip(np.dot(s_hat, d_hat), -1, 1)))
    return n_hat, s_hat, d_hat, mu_s, mu_e, phase


def micro_facets(n_hat, w, ngrid=121, umax=4.0):
    """Gaussian micro-slope ensemble about n_hat. Returns global facet normals m
    (M,3) and area-true probability weights P (M,) summing to 1.

    Slopes (p,q)=w*(u,v), u,v ~ N(0,1). Facet local normal (-p,-q,1)/|.|. Weight =
    Gaussian PDF (per the unit-Gaussian in u) times the true-area factor sqrt(1+p^2+q^2)
    (a facet of slope tilt presents more true area per unit base area)."""
    t1, t2 = ortho_basis(n_hat)
    u = np.linspace(-umax, umax, ngrid)
    gu = np.exp(-0.5 * u * u)
    U, V = np.meshgrid(u, u, indexing="ij")
    GW = np.outer(gu, gu)
    p = w * U.ravel()
    q = w * V.ravel()
    norm = np.sqrt(1.0 + p * p + q * q)
    ml = np.stack([-p / norm, -q / norm, 1.0 / norm], axis=1)   # local
    m = ml[:, [0]] * t1 + ml[:, [1]] * t2 + ml[:, [2]] * n_hat  # global (M,3)
    P = GW.ravel() * norm           # true-area weight
    P /= P.sum()
    return m, P


def enhancement(sun_elev_deg=4.1, wall_slope_deg=11.0, d_over_D=0.051,
                w_deg=20.0, beta=0.2, phase_override_deg=None,
                correlated_shadow=False, ngrid=161):
    """Anisothermal directional-brightness enhancement E toward the floor.

    beta = Q_bg / [(1-A) S mu_s_macro], the uniform background as a fraction of the
    macroscopic direct insolation (keeps shadowed facets warm; beta=0 -> max contrast).
    correlated_shadow: at small phase, source-shadow and view-mask are correlated; if
    True use a single G1 (min angle) instead of the product (brackets the hot spot)."""
    n_hat, s_hat, d_hat, mu_s_macro, mu_e_macro, phase = geometry(
        sun_elev_deg, wall_slope_deg, d_over_D)
    if phase_override_deg is not None:
        # rebuild d_hat in the x-z plane at the requested phase from the Sun
        ph = np.radians(phase_override_deg)
        # rotate s_hat by -ph about y (toward the wall normal side / downward)
        c, sn = np.cos(ph), np.sin(ph)
        d_hat = np.array([s_hat[0] * c + s_hat[2] * sn, 0.0,
                          -s_hat[0] * sn + s_hat[2] * c])
        mu_e_macro = float(np.dot(n_hat, d_hat))
        phase = phase_override_deg

    w = np.radians(w_deg)
    m, P = micro_facets(n_hat, w, ngrid=ngrid)
    mu_s = m @ s_hat
    mu_e = m @ d_hat

    lit = mu_s > 0.0
    vis = mu_e > 0.0
    g_s = np.where(lit, G1(np.clip(mu_s, 0, 1), w), 0.0)
    g_e = np.where(vis, G1(np.clip(mu_e, 0, 1), w), 0.0)

    # direct absorbed (normalized units; the S,(1-A) prefactor cancels in E)
    absorbed_direct = np.where(lit, mu_s * g_s, 0.0)
    Q_bg = beta * mu_s_macro                      # uniform background, same units
    T4 = absorbed_direct + Q_bg                    # ~ eps sigma T^4 / [(1-A)S]

    if correlated_shadow:
        # joint shadow+mask -> single G1 at the more grazing of the two directions
        g_joint = np.minimum(np.where(lit, G1(np.clip(mu_s, 0, 1), w), 0.0),
                             np.where(vis, G1(np.clip(mu_e, 0, 1), w), 0.0))
        w_e = np.where(vis, np.maximum(mu_e, 0) * g_joint, 0.0)
        # recompute T4 with the joint masking on the source side for consistency
        T4 = np.where(lit, mu_s * g_joint, 0.0) + Q_bg
    else:
        w_e = np.where(vis, mu_e * g_e, 0.0)

    Pn = P / P.sum()
    mean_T4 = np.sum(Pn * T4)
    mean_we = np.sum(Pn * w_e)
    aniso = np.sum(Pn * T4 * w_e)
    if mean_T4 * mean_we <= 0:
        return dict(E=np.nan, phase=phase, mu_s_macro=mu_s_macro,
                    mu_e_macro=mu_e_macro, T_b_ratio=np.nan, true_ratio=np.nan)
    E = aniso / (mean_T4 * mean_we)
    # true_ratio = (rough-anisothermal floor brightness) / (flat-Lambertian-at-T_j),
    # the factor the PRODUCTION code applies to eps*sigma*T_j^4. = E * (iso angular
    # factor). T_j^4 = mean_T4 (energy conservation). Baseline projection = n.d = mu_e_macro.
    iso_factor = mean_we / max(mu_e_macro, 1e-9)
    true_ratio = aniso / (mean_T4 * max(mu_e_macro, 1e-9))
    return dict(E=float(E), phase=float(phase), mu_s_macro=mu_s_macro,
                mu_e_macro=mu_e_macro,
                iso_factor=float(iso_factor), true_ratio=float(true_ratio),
                T_b_ratio=float(true_ratio ** 0.25),  # brightness-T ratio toward floor
                lit_frac=float(np.sum(Pn * (mu_s > 0))),
                vis_frac=float(np.sum(Pn * (mu_e > 0))))


def smith_chi_at(n_hat, s_hat, d_hat, w):
    """Exact isothermal-vs-... wait: the production Smith chi (smith_beaming.h), i.e.
    chi = pi * G1(cos_e) * I(s,d) / (cos_e * Z), with I(s,d)=<max(0,m.s)max(0,m.d)>,
    Z=Int g1 I dOmega. This is what the code currently multiplies eps*sigma*T_j^4 by.
    Evaluated by direct Gaussian-slope + hemisphere quadrature (no table)."""
    cos_e = float(np.dot(n_hat, d_hat))
    if cos_e <= 0:
        return 1.0
    t1, t2 = ortho_basis(n_hat)
    # local s,d
    s_l = np.array([np.dot(s_hat, t1), np.dot(s_hat, t2), np.dot(s_hat, n_hat)])
    d_l = np.array([np.dot(d_hat, t1), np.dot(d_hat, t2), np.dot(d_hat, n_hat)])
    Nq = 41
    u = np.linspace(-4, 4, Nq)
    gu = np.exp(-0.5 * u * u)
    U, V = np.meshgrid(u, u, indexing="ij")
    GW = (np.outer(gu, gu)).ravel()
    GW /= GW.sum()
    p = w * U.ravel(); q = w * V.ravel()
    inv = 1.0 / np.sqrt(1 + p * p + q * q)
    m = np.stack([-p * inv, -q * inv, inv], axis=1)

    def I_of(dl):
        ms = m @ s_l; md = m @ dl
        return np.sum(GW * np.maximum(0, ms) * np.maximum(0, md))

    # Z = Int over emission hemisphere of G1(theta') I(d') dOmega'
    Nt, Np = 24, 48
    Z = 0.0
    for it in range(Nt):
        te = (it + 0.5) * (np.pi / 2) / Nt
        ce, se = np.cos(te), np.sin(te)
        g1 = float(G1(np.array([ce]), w)[0])
        for ip in range(Np):
            ph = (ip + 0.5) * 2 * np.pi / Np
            dl = np.array([se * np.cos(ph), se * np.sin(ph), ce])
            Z += g1 * I_of(dl) * se * (np.pi / 2 / Nt) * (2 * np.pi / Np)
    g1e = float(G1(np.array([cos_e]), w)[0])
    chi = np.pi * g1e * I_of(d_l) / (cos_e * max(Z, 1e-12))
    return float(np.clip(chi, 0, 4))


def floor_temp_from_flux(flux, T_ref=64.6, flux_ref=0.94):
    """Map a delivered floor flux to a radiative-equilibrium floor temperature using
    the Ingersoll calibration (0.94 W/m^2 -> 64.6 K)."""
    return T_ref * (flux / flux_ref) ** 0.25


def combined_budget(w_deg=20.0, beta=0.2, wall_slope_deg=11.0, d_over_D=0.051,
                    T_mean=290.0, ir_flat=0.47, rho_model=0.05, A=A_BOND,
                    B_opp=1.3):
    """Combined floor-flux budget: anisothermal wall IR + reflected-solar opposition.

    Anchored to the model's delivered ~0.49 W/m^2 (ir_flat = 0.47 IR + the rest
    reflected at rho_model=0.05, Lambertian-flat). Enhancements from the micro-facet
    ensemble:
      IR :       ir_flat * E_IR                       (E_IR with thermal background beta)
      solar:     ir_flat * ratio_flat * (A/rho) * E_sol * B_opp
        ratio_flat = (rho S mu_s_macro) / (eps sigma T_mean^4)   [reflected/IR, flat model]
        A/rho      = albedo fix (Bond 0.12 vs thermal reflectivity 0.05) ~ 2.4x
        E_sol      = shadow-hiding directional enhancement (beta=0; reflection has no
                     thermal background) -- the geometric opposition (lit facets are the
                     visible ones at backscatter)
        B_opp      = intrinsic particle/coherent backscatter surge (~1.3 at 10 deg phase)
    """
    r_ir = enhancement(wall_slope_deg=wall_slope_deg, w_deg=w_deg, beta=beta,
                       d_over_D=d_over_D)
    r_sol = enhancement(wall_slope_deg=wall_slope_deg, w_deg=w_deg, beta=0.0,
                        d_over_D=d_over_D)
    E_IR, E_sol = r_ir["E"], r_sol["E"]
    mu_s_macro = r_ir["mu_s_macro"]
    ir_emiss_flat = EPS * SIGMA * T_mean ** 4
    ratio_flat = (rho_model * S * mu_s_macro) / ir_emiss_flat
    ir_aniso = ir_flat * E_IR
    refl_aniso = ir_flat * ratio_flat * (A / rho_model) * E_sol * B_opp
    total = ir_aniso + refl_aniso
    base = ir_flat + ir_flat * ratio_flat              # model's flat delivered total
    return dict(E_IR=E_IR, E_sol=E_sol, ratio_flat=ratio_flat,
                ir_aniso=ir_aniso, refl_aniso=refl_aniso, total=total, base=base,
                T_eq=floor_temp_from_flux(total),
                T_cons=45.0 * (total / base) ** 0.25)


# --------------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("H1a: anisothermal-wall thermal beaming toward the Haworth PSR floor")
    print("=" * 78)
    n_hat, s_hat, d_hat, mu_s, mu_e, phase = geometry(4.1, 11.0, 0.051)
    print(f"Geometry (Sun 4.1 deg, wall slope 11 deg, d/D=0.051):")
    print(f"  macroscopic Sun incidence theta_s = {np.degrees(np.arccos(mu_s)):.1f} deg "
          f"(mu_s={mu_s:.3f})")
    print(f"  macroscopic emission theta_e        = {np.degrees(np.arccos(mu_e)):.1f} deg "
          f"(mu_e={mu_e:.3f})")
    print(f"  phase angle (Sun<->floor)           = {phase:.1f} deg  (backscatter)")
    print()

    print("Anisothermal enhancement E (=floor-IR multiplier) and brightness-T ratio:")
    print("  product shadow/mask (standard Smith; underestimates the hot spot)")
    print(f"  {'w_RMS':>6} {'beta':>5} {'wall':>5} | {'E':>6} {'E^1/4':>6} "
          f"{'litf':>5} {'visf':>5}")
    for wslope in (8.0, 11.0, 20.0):
        for w_deg in (10.0, 20.0, 30.0):
            for beta in (0.0, 0.2, 0.5):
                r = enhancement(wall_slope_deg=wslope, w_deg=w_deg, beta=beta)
                print(f"  {w_deg:6.0f} {beta:5.2f} {wslope:5.0f} | "
                      f"{r['E']:6.2f} {r['T_b_ratio']:6.3f} "
                      f"{r['lit_frac']:5.2f} {r['vis_frac']:5.2f}")
        print()

    print("Hot-spot bracket (correlated shadow/mask, single G1) at the nominal point:")
    for w_deg in (10.0, 20.0, 30.0):
        rp = enhancement(w_deg=w_deg, beta=0.2, correlated_shadow=False)
        rc = enhancement(w_deg=w_deg, beta=0.2, correlated_shadow=True)
        print(f"  w={w_deg:4.0f} deg : E_product={rp['E']:5.2f}  "
              f"E_correlated={rc['E']:5.2f}")
    print()

    print("Crater-depth (d/D) dependence (w=20, beta=0.2, wall=11): real phase varies")
    for ddD in (0.03, 0.051, 0.08, 0.12):
        r = enhancement(w_deg=20.0, beta=0.2, d_over_D=ddD)
        print(f"  d/D={ddD:5.3f}  phase={r['phase']:4.1f} deg : E={r['E']:5.2f}")
    print()

    print("Implied floor temperature, two calibrations (nominal E~2.0):")
    print("  The model floor reaches only 45 K for the 0.49 W/m^2 it is *delivered*")
    print("  (it does not equilibrate: inertia smooths the brief peak + conduction to")
    print("  the cold deep column). So H1a raises the delivered flux x E; the realized")
    print("  floor gain is bracketed by:")
    print(f"  {'E':>4} {'flux':>5} | {'T_conservative':>14} {'T_equilibrium':>14}")
    print(f"       {'(W/m2)':>5} | {'(45*E^.25)':>14} {'(Ingersoll)':>14}")
    for E in (1.0, 1.5, 2.0, 2.5):
        t_cons = 45.0 * E ** 0.25
        t_eq = floor_temp_from_flux(0.49 * E)
        print(f"  {E:4.1f} {0.49*E:5.2f} | {t_cons:14.1f} {t_eq:14.1f}")
    print("  Diviner summer-max target: 64.6 K (Ingersoll-required flux ~0.94, E~1.9)")
    print()
    print("  => H1a (E~2) supplies the Ingersoll-required flux; the realized floor gain")
    print("     is +8 K (if the floor's current under-response persists) up to the full")
    print("     +20 K (if it equilibrates) -- the under-response is H2's domain (a more")
    print("     sustained warm ring lets the floor equilibrate). H1a and H2 are")
    print("     complementary, and together plausibly close the gap.")
    print()

    print("=" * 78)
    print("Combined floor-flux budget: anisothermal IR + reflected-solar opposition")
    print("=" * 78)
    print("  (anchored to the model's delivered ~0.49 W/m^2; B_opp = intrinsic surge)")
    print(f"  {'w':>3} {'B_opp':>5} | {'E_IR':>5} {'E_sol':>5} | "
          f"{'IR':>5} {'+solar':>6} {'=total':>6} | {'T_eq':>6} {'T_cons':>6}")
    for w_deg in (15.0, 20.0, 30.0):
        for B_opp in (1.0, 1.3, 1.6):
            b = combined_budget(w_deg=w_deg, B_opp=B_opp)
            print(f"  {w_deg:3.0f} {B_opp:5.1f} | {b['E_IR']:5.2f} {b['E_sol']:5.2f} | "
                  f"{b['ir_aniso']:5.2f} {b['refl_aniso']:6.2f} {b['total']:6.2f} | "
                  f"{b['T_eq']:6.1f} {b['T_cons']:6.1f}")
    print()
    bn = combined_budget(w_deg=20.0, B_opp=1.3)
    print(f"  Nominal (w=20, B_opp=1.3): floor flux {bn['base']:.2f} -> {bn['total']:.2f} "
          f"W/m^2  ({bn['total']/bn['base']:.2f}x)")
    print(f"    IR anisothermal contributes {bn['ir_aniso']:.2f}, "
          f"reflected-solar opposition {bn['refl_aniso']:.2f} "
          f"({100*bn['refl_aniso']/bn['total']:.0f}% of total)")
    print(f"    floor T: equilibrium {bn['T_eq']:.1f} K, conservative {bn['T_cons']:.1f} K "
          f"(Diviner 64.6 K)")
    print("  => solar opposition is a real but SECONDARY add-on (~+2 K eq); the")
    print("     anisothermal IR (E~2) is the dominant lever.")


if __name__ == "__main__":
    main()
