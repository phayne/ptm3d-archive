#!/usr/bin/env python3
"""Quantify the Diviner depth-weighting bias on PSR-floor MAX bolometric T.

Hypothesis (user): Diviner's bolometric brightness T in a PSR is biased WARM
because its longer thermal channels (~50-400 um) sense emission from mm-deep
regolith that is warmer than the surface, so the model (surface kinetic T) is
unfairly compared to a subsurface-weighted observation.

Method: a standalone 1-D regolith heat-conduction model (properties matched to
the PTM3D C++ Moon defaults) forced by a realistic PSR-floor absorbed-flux
history (near-zero direct + a synodic-periodic indirect peak). Solve T(z,t) to a
periodic steady state on a grid fine enough near the surface (10 um) to resolve
the thermal-IR emission depth (~10 um - 2 mm). Then compute, for each candidate
emission depth d, the emission-weighted brightness temperature
   T_b,d(t) = ( integral_0^inf sigma T(z,t)^4 * (1/d) e^{-z/d} dz / sigma )^(1/4)
and compare its MAX over time to the surface kinetic MAX. A positive
(T_b,d max - T_surface max) supports the user's hypothesis.

The sign hinges on skin-depth vs emission-depth: for the slow synodic/multi-day
variations that set the PSR max, the thermal skin depth is cm-scale >> the mm
emission depth, so the mm regolith TRACKS the surface and the depth bias is the
small thermal-rectification (time-mean subsurface warming) seen at mm depth.
"""
from __future__ import annotations
import numpy as np

SIGMA = 5.670374419e-8
# --- PTM3D Moon-default regolith (config_parser.h / planetary_properties.h) ---
KS, KD = 7.4e-4, 3.4e-3      # surface/deep conductivity W/m/K
H = 0.06                      # density/conductivity e-folding depth m
RHOS, RHOD = 1100.0, 1800.0   # kg/m^3
CHI = 2.7                     # radiative-conductivity coefficient
EPS = 0.95
QB = 0.018                    # geothermal W/m^2
P_SYN = 29.53 * 86400.0       # synodic period s


def cp_hayne(T):
    """Hayne et al. (2017) lunar regolith specific heat cp(T) [J/kg/K]."""
    c = [-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9]
    T = np.asarray(T, float)
    return c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4


def kc(z):
    return KD + (KS - KD) * np.exp(-z / H)


def rho(z):
    return RHOD + (RHOS - RHOD) * np.exp(-z / H)


def build_grid(z_max=0.6, dz0=1e-5, growth=1.06):
    """Geometric grid: 10 um at the surface growing to z_max."""
    z = [0.0]
    dz = dz0
    while z[-1] < z_max:
        z.append(z[-1] + dz)
        dz *= growth
    return np.array(z)


def forcing(t, q_base, q_peak, peak_width_days, period=P_SYN):
    """Absorbed indirect flux on a PSR floor face: small baseline + one broad
    synodic peak per cycle (the lit rim sweeps past the face over a few days)."""
    phase = (t % period) / period            # 0..1
    d = np.minimum(np.abs(phase - 0.5), 1 - np.abs(phase - 0.5))  # dist to peak center, wrapped
    w = (peak_width_days * 86400.0) / period
    return q_base + q_peak * np.exp(-0.5 * (d / w) ** 2)


def solve(q_base=0.5, q_peak=30.0, peak_width_days=3.0,
          n_cycles=40, steps_per_cycle=2000, z_max=0.6):
    """Backward-Euler 1-D conduction to periodic steady state. Returns z, and
    T(z,t) over the final cycle."""
    z = build_grid(z_max)
    n = len(z)
    dz = np.diff(z)
    rho_z = rho(z)
    dt = P_SYN / steps_per_cycle
    T = np.full(n, 90.0)                       # initial guess
    # midpoint spacings for the conservative Laplacian
    dz_m = 0.5 * (np.concatenate([dz, [dz[-1]]]) + np.concatenate([[dz[0]], dz]))

    n_save = steps_per_cycle
    Tsave = np.zeros((n_save, n))
    tsave = np.zeros(n_save)
    total_steps = n_cycles * steps_per_cycle
    for s in range(total_steps):
        t = s * dt
        Told = T.copy()
        # Picard on the nonlinear k(T), cp(T), and the radiative surface BC.
        for _ in range(3):
            kz = kc(z) * (1.0 + CHI * (T / 350.0) ** 3)
            kf = 0.5 * (kz[:-1] + kz[1:])      # face conductivities (n-1)
            cp = cp_hayne(T)
            C = rho_z * cp * dz_m / dt          # heat capacity / dt per node
            A = np.zeros((n, n))
            b = np.zeros(n)
            # interior nodes
            for i in range(1, n - 1):
                aw = kf[i - 1] / dz[i - 1]
                ae = kf[i] / dz[i]
                A[i, i - 1] = -aw
                A[i, i + 1] = -ae
                A[i, i] = C[i] + aw + ae
                b[i] = C[i] * Told[i]
            # surface node (i=0): eps sigma T^4 + conduction in = q(t)
            ae = kf[0] / dz[0]
            qabs = forcing(t, q_base, q_peak, peak_width_days)
            # linearize eps sigma T^4 ~ eps sigma (4 Ts^3 T - 3 Ts^4)
            Ts = T[0]
            rad = EPS * SIGMA
            A[0, 0] = C[0] + ae + rad * 4.0 * Ts ** 3
            A[0, 1] = -ae
            b[0] = C[0] * Told[0] + qabs + rad * 3.0 * Ts ** 4
            # bottom node: geothermal flux in
            aw = kf[-1] / dz[-1]
            A[-1, -1] = C[-1] + aw
            A[-1, -2] = -aw
            b[-1] = C[-1] * Told[-1] + QB
            T = np.linalg.solve(A, b)
        if s >= total_steps - n_save:
            k = s - (total_steps - n_save)
            Tsave[k] = T
            tsave[k] = t
    return z, tsave, Tsave


def brightness_max(z, Tsave, depths_mm):
    """For each emission depth d, T_b,d(t) = (<sigma T^4>_w / sigma)^1/4 with an
    exponential emission kernel w(z)=e^{-z/d}/d; return max over the cycle."""
    out = {}
    surf_max = Tsave[:, 0].max()
    out["surface"] = surf_max
    for dmm in depths_mm:
        d = dmm * 1e-3
        w = np.exp(-z / d)
        w = w / np.trapz(w, z)
        T4 = Tsave ** 4
        Tb = (np.trapz(T4 * w, z, axis=1)) ** 0.25      # brightness T(t)
        out[f"{dmm:g}mm"] = (Tb.max(), Tb.max() - surf_max)
    return out


def main():
    print("=== PSR-floor depth-weighting: warm-fringe face (q_peak=30 W/m^2, 3-day peak) ===")
    z, ts, Ts = solve(q_base=0.5, q_peak=30.0, peak_width_days=3.0)
    # report skin depths and the T(z) structure at the surface-max time
    imax = Ts[:, 0].argmax()
    print(f"  surface T: mean={Ts[:,0].mean():.1f} K, max={Ts[:,0].max():.1f} K, min={Ts[:,0].min():.1f} K")
    for dmm in [0.05, 0.1, 0.2, 0.5]:
        zi = np.argmin(np.abs(z - dmm*1e-3))
        print(f"  T at {dmm:g} mm: mean={Ts[:,zi].mean():.1f} K, max={Ts[:,zi].max():.1f} K "
              f"(at surface-max time: surf {Ts[imax,0]:.1f} vs depth {Ts[imax,zi]:.1f})")
    print("  emission-depth-weighted brightness-T MAX vs surface MAX:")
    bm = brightness_max(z, Ts, [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    print(f"    surface max = {bm['surface']:.2f} K")
    for k, v in bm.items():
        if k != "surface":
            print(f"    d={k:>5s}: Tb_max={v[0]:.2f} K  bias(Tb_max - surf_max) = {v[1]:+.2f} K")

    # Where does the warm subsurface (rectification) actually appear? It must be
    # BELOW the thermal-IR emission depth (mm) for the depth bias to vanish.
    print("\n=== time-mean T vs depth (rectification): warm subsurface is a cm-m effect ===")
    Tmean = Ts.mean(axis=0)
    for dd in [0.0, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 0.2, 0.5]:
        zi = np.argmin(np.abs(z - dd))
        print(f"   z={dd*1000:7.2f} mm: T_mean={Tmean[zi]:.2f} K  (dT vs surface {Tmean[zi]-Tmean[0]:+.2f} K)")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path
    try:
        from ptm.pubstyle import apply_style
        apply_style()
    except Exception:
        pass
    depths_mm = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    bm = brightness_max(z, Ts, list(depths_mm))
    bias = np.array([bm[f"{d:g}mm"][1] for d in depths_mm])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax1.semilogx(z * 1000, Tmean, "C0-")
    ax1.axvspan(0.01, 2.0, color="C1", alpha=0.18, label="thermal-IR emission depth\n(~10 um - 2 mm)")
    ax1.axvspan(20, 600, color="C3", alpha=0.12, label="microwave / Apollo depth\n(cm - m, warm subsurface)")
    ax1.set_xlabel("depth (mm)"); ax1.set_ylabel("time-mean T (K)")
    ax1.set_title("Rectification is a cm-m effect")
    ax1.legend(fontsize=7, loc="upper left")
    ax2.axhline(0, color="0.5", lw=1, ls="--")
    ax2.plot(depths_mm, bias, "C3o-")
    ax2.set_xlabel("emission depth d (mm)")
    ax2.set_ylabel("Diviner Tb_max - surface max (K)")
    ax2.set_title("Depth-weighting COOLS the max")
    fig.suptitle("Diviner depth-weighting on PSR-floor MAX bolometric T — wrong sign to explain the cold bias",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("docs/figures/diviner_validation/depth_weighting.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")

    print("\n=== sensitivity: brief vs broad peak (does peak duration change the sign?) ===")
    for pw in [0.5, 1.0, 3.0, 7.0]:
        z, ts, Ts = solve(q_base=0.5, q_peak=30.0, peak_width_days=pw)
        bm = brightness_max(z, Ts, [0.2, 1.0])
        print(f"  peak_width={pw:>4g} d: surf_max={bm['surface']:.1f}  "
              f"bias@0.2mm={bm['0.2mm'][1]:+.2f}  bias@1mm={bm['1mm'][1]:+.2f} K")


if __name__ == "__main__":
    main()
