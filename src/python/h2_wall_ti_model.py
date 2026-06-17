#!/usr/bin/env python3
"""H2 simple model: does higher crater-WALL thermal inertia warm the PSR floor?

Two-component setup (user's framing): crater floor = low TI (regolith); crater
walls/rim = high TI (e.g. 10x). The question is whether high-TI walls, which stay
warm longer as the grazing Sun circles in azimuth, raise the floor's summer-max --
against two headwinds: (a) high TI LOWERS each wall's peak T (Jensen: <T^4> drops),
and (b) the floor's summer-max is held down by conductive drain to its cold (~30 K)
deep column, set by the ANNUAL-MEAN indirect flux.

Model:
  - 1D heat conduction (implicit, linearized radiative BC) for N_SECTORS azimuthal
    wall columns + 1 floor column, over multiple years at sub-daily steps.
  - Polar insolation: subsolar lat oscillates +/-1.54 deg/yr; the Sun circles in
    azimuth once per synodic month (rise/set at lat -87.45). Each wall sector j is
    lit by S*max(0, n_j . s_hat); n_j = inner-wall normal at azimuth phi_j.
  - Floor indirect flux = F_terrain * mean_j eps*sigma*T_j^4  (the warm-ring sum).
  - Compare floor summer-max for wall TI = 1x vs 10x (floor TI fixed low).

Pure numpy. Run: python3 h2_wall_ti_model.py
"""
from __future__ import annotations
import numpy as np

SIGMA = 5.670374e-8
S = 1361.0
A = 0.12
EPS = 0.95
QB = 0.018                      # geothermal [W/m^2]
LAT = np.radians(-87.45)
OBLIQ = np.radians(1.54)
P_SYN = 29.53 * 86400.0        # synodic month [s]
P_YEAR = 365.25 * 86400.0
F_TERRAIN = 0.0055             # floor view factor to the rim ring (total)
WALL_SLOPE = np.radians(11.0)
RIM_ELEV = np.radians(6.0)     # floor's rim horizon

# Regolith properties; TI = sqrt(k*rho*c). Floor = baseline; wall scaled by TI ratio.
RHO = 1500.0
CP_REF = 640.0                 # reference cp for the nominal TI label
K_FLOOR = 1.0e-3
TI_FLOOR = np.sqrt(K_FLOOR * RHO * CP_REF)


def cp_of_T(T):
    """Heat capacity cp(T) [J/kg/K], ~Hayne (2017): low at low T (cp(46)~126,
    cp(300)~771). Linear-ish approximation cp ~ 2.6 T, clamped."""
    return np.clip(2.6 * np.asarray(T, float), 50.0, 800.0)


def depth_grid(n=44, z_max=2.0, r=1.18):
    """Exponentially spaced node depths [m], fine at the surface, to z_max."""
    z = np.cumsum(r ** np.arange(n))
    z = z - z[0]
    return z / z[-1] * z_max


def sun_dir(t):
    """Unit Sun direction (x,y,z) in a local frame (z=up) at time t [s] array."""
    phi_sun = -OBLIQ * np.cos(2 * np.pi * t / P_YEAR)        # subsolar lat
    h = 2 * np.pi * t / P_SYN                                # hour angle (azimuth)
    sin_e = np.sin(LAT) * np.sin(phi_sun) + np.cos(LAT) * np.cos(phi_sun) * np.cos(h)
    sin_e = np.clip(sin_e, -1, 1)
    e = np.arcsin(sin_e)
    cos_e = np.cos(e)
    # azimuth measured by the hour angle h
    return np.stack([cos_e * np.cos(h), cos_e * np.sin(h), np.sin(e)], axis=-1), e


def wall_insolation(t, n_sectors):
    """S * max(0, n_j . s_hat) for each azimuthal wall sector. Returns (Nt, Nsec)."""
    s, e = sun_dir(t)                              # (Nt,3)
    phis = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    # inner-wall normal at azimuth phi: (-cos phi sinb, -sin phi sinb, cosb)
    sb, cb = np.sin(WALL_SLOPE), np.cos(WALL_SLOPE)
    nx = -np.cos(phis) * sb; ny = -np.sin(phis) * sb; nz = np.full(n_sectors, cb)
    mu = (s[:, 0:1] * nx + s[:, 1:2] * ny + s[:, 2:3] * nz)    # (Nt, Nsec)
    return S * np.clip(mu, 0, None)


def conduct(flux_of_t, dt, k, z, T0=90.0):
    """Implicit 1D conduction for ONE column driven by surface absorbed-flux series
    flux_of_t (W/m^2, the non-radiative absorbed flux). Linearized radiative BC,
    temperature-dependent cp(T) (lagged one step). Returns T_surface(t)."""
    n = len(z)
    T = np.full(n, T0)
    nt = flux_of_t.shape[0]
    Tsurf = np.empty(nt)
    dzm = np.empty(n); dzp = np.empty(n)
    dzm[1:] = z[1:] - z[:-1]; dzm[0] = dzm[1]
    dzp[:-1] = z[1:] - z[:-1]; dzp[-1] = dzp[-2]
    vol = 0.5 * (dzm + dzp)
    idx = np.arange(1, n - 1)
    for it in range(nt):
        rc = RHO * cp_of_T(T)                         # rho*cp per node (lagged)
        kap = k / rc                                  # diffusivity per node
        a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
        aw = kap[idx] / (dzm[idx] * vol[idx])
        ae = kap[idx] / (dzp[idx] * vol[idx])
        a[idx] = -aw * dt; c[idx] = -ae * dt
        b[idx] = 1 + (aw + ae) * dt; d[idx] = T[idx]
        # surface node: -k dT/dz = F_abs - eps sigma T^4 (radiation linearized)
        ae0 = kap[0] / (dzp[0] * vol[0])
        rad = 4 * EPS * SIGMA * T[0] ** 3
        b[0] = 1 + ae0 * dt + dt * rad / (rc[0] * vol[0])
        c[0] = -ae0 * dt
        d[0] = T[0] + dt / (rc[0] * vol[0]) * (flux_of_t[it]
                                               - EPS * SIGMA * T[0] ** 4 + rad * T[0])
        # bottom node: geothermal flux in
        aw_b = kap[-1] / (dzm[-1] * vol[-1])
        a[-1] = -aw_b * dt; b[-1] = 1 + aw_b * dt
        d[-1] = T[-1] + dt / (rc[-1] * vol[-1]) * QB
        # Thomas solve
        for i in range(1, n):
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]; d[i] -= m * d[i - 1]
        T[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            T[i] = (d[i] - c[i] * T[i + 1]) / b[i]
        Tsurf[it] = T[0]
    return Tsurf


def run(ti_ratio, n_sectors=24, years=4, steps_per_syn=80):
    """Return (t, Tfloor, Twall_sectors, e) for wall TI = ti_ratio * floor TI."""
    dt = P_SYN / steps_per_syn
    nt = int(years * P_YEAR / dt)
    t = np.arange(nt) * dt
    z = depth_grid()
    k_wall = K_FLOOR * ti_ratio ** 2               # TI ~ sqrt(k) -> k ~ TI^2

    # wall sectors
    Qw = wall_insolation(t, n_sectors)             # (nt, Nsec) insolation
    Twall = np.empty((nt, n_sectors))
    for j in range(n_sectors):
        Twall[:, j] = conduct((1 - A) * Qw[:, j], dt, k_wall, z, T0=120.0)

    # floor: indirect flux = F_terrain * mean over sectors of eps sigma T_wall^4
    qind = F_TERRAIN * EPS * SIGMA * np.mean(Twall ** 4, axis=1)   # (nt,) absorbed-ready
    Tfloor = conduct(EPS * qind, dt, K_FLOOR, z, T0=40.0)
    _, e = sun_dir(t)
    return t, Tfloor, Twall, e


def summer_mask(t, e):
    """Summer = last full year, near peak subsolar (Sun highest). Use the top-decile
    elevation window of the final year as the summer season."""
    last = t > (t[-1] - P_YEAR)
    thr = np.percentile(e[last], 80)
    return last & (e > thr)


def main():
    print("H2 simple model: wall thermal inertia -> PSR floor temperature")
    print(f"  floor TI = {TI_FLOOR:.0f}; comparing wall TI = 1x, 3x, 10x floor")
    print(f"  {'wall TI':>8} | {'floor summer-max':>16} {'floor ann-mean':>15} "
          f"{'wall summer-max':>15} {'wall Tmin(off-peak)':>19}")
    base = None
    for ratio in (1.0, 3.0, 10.0):
        t, Tf, Tw, e = run(ratio)
        sm = summer_mask(t, e)
        last = t > (t[-1] - P_YEAR)
        f_smax = Tf[sm].max()
        f_mean = Tf[last].mean()
        w_smax = Tw[sm].max()
        w_min = np.percentile(Tw[last], 5)
        if base is None:
            base = f_smax
        d = f_smax - base
        print(f"  {ratio:6.0f}x | {f_smax:10.1f} ({d:+.1f}) {f_mean:15.1f} "
              f"{w_smax:15.1f} {w_min:19.1f}")
    print()
    print("Read: does higher wall TI raise the floor summer-max (col 2, delta)?")
    print("Watch the competition: wall summer-max should DROP (lower peaks) while wall")
    print("off-peak Tmin RISES (stays warm) -- the floor wins only if sustained-ring")
    print("beats lower-peaks.")


if __name__ == "__main__":
    main()
