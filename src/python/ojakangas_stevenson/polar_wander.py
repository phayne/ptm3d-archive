"""Polar wander of a decoupled ice shell (Ojakangas & Stevenson 1989b, "OSPW").

THE polar-wander physics of this package; ``scripts/ospw_polar_wander.py``,
``scripts/polar_wander_cycle.py`` and ``scripts/coupled_shell_simulation.py``
import from here. Equation and page numbers refer to OSPW (Icarus 81,
242-270).

Contents
--------
* Eq. 2.11  Willemann compensation C_w (``willemann_cw``)
* Eq. 2.18  gravity/inertia of bottom-loaded thickness variations, including
            the iceberg effect (``iceberg_bracket``, ``shell_c2m``,
            ``compensation_factor``)
* Eqs. 3.8-3.10  the body-fixed hydrostatic fossil bulge
            (``fossil_c20_c22``, ``fossil_inertia_tensor``)
* Eqs. 3.11/3.12  quasi-static stability test (``stability_test``)
* Eq. 5.29  I_0, Eq. 5.15 C_f, Eq. 5.32 (t/tau)_equil, Eq. 5.34 epsilon,
            Eq. 5.41 tau_f, Eq. 4.22 undamped time scale
* Eq. 5.39  overdamped reorientation (``overdamped_step``)

Conventions: body frame x toward Jupiter (sub-Jovian point), z along the
spin axis at the epoch the fossil froze. C20, C22 are unnormalized
(Eqs. 2.9, 3.7); the inertia tensor follows Eqs. 2.12-2.14 without the
isotropic I'.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants as C

YEAR_S = 365.25 * 86400.0


@dataclass
class OSPWParams:
    """Shell, lithosphere and dissipation parameters (OSPW Sections 2-5).

    ``None`` for radius / mean_density / gravity / mass means "take it from
    the active constants set" (constants.use_parameter_set).
    """
    shell_thickness: float = 20e3        # b_c [m]
    lithosphere_thickness: float = 2e3   # d [m] (b_c/d ~ 10, p. 250)
    ice_density: float = C.ICE_DENSITY   # rho_c
    water_density: float = C.WATER_DENSITY  # rho_m
    rigidity: float = C.ICE_RIGIDITY     # mu [Pa] (4e10 dyn/cm^2)
    poisson_ratio: float = 0.33          # nu (p. 247)
    love_h: float = 1.26                 # h, Ross & Schubert 1987 (p. 249)
    # Eq. 5.2 viscosity below ~190 K: eta = eta0 exp[l (T_m/T - 1)],
    # eta0 ~ 3e15 P = 3e14 Pa s, l ~ 15 (p. 256)
    melting_viscosity: float = 3e14
    activation_l: float = 15.0
    melting_temp: float = C.T_MELT
    surface_temp: float = 100.0          # T_s for Eq. 5.15 (p. 263)
    fracture_temp: float = 124.0         # T_f, base of the fractured layer (p. 264)
    # Eq. 5.29: I_0 ~ (8 pi/3) rho b_0 a^4 with b_0 ~ 100 km (shell + ocean)
    # and rho ~ 1 g/cm^3 (p. 260)
    i0_thickness: float = 100e3
    i0_density: float = 1000.0
    # tau in t/tau: the growth parameter, ~1e12 yr (p. 262), NOT tau_th
    tau_growth_yr: float = 1e12
    tau_th_yr: float = 1e7               # thermal diffusion time (p. 261)
    radius: Optional[float] = None
    mean_density: Optional[float] = None
    gravity: Optional[float] = None
    mass: Optional[float] = None

    def __post_init__(self):
        if self.radius is None:
            self.radius = C.EUROPA_RADIUS
        if self.mean_density is None:
            self.mean_density = C.EUROPA_DENSITY
        if self.gravity is None:
            self.gravity = C.EUROPA_GRAVITY
        if self.mass is None:
            self.mass = C.EUROPA_MASS

    @property
    def delta_rho(self):
        return self.water_density - self.ice_density


def gamma_tidal(p):
    """gamma = n^2 a^3 / (G M) (Eq. 3.7 text)."""
    return C.MEAN_MOTION**2 * p.radius**3 / (C.G_GRAV * p.mass)


def willemann_cw(p):
    """Willemann compensation parameter C_w, Eq. 2.11 (with tau_0 of Eq. 2.7).

    C_w = [1 + tau_0 / (rho_m (1 - (3/5) rho_m/rho_bar))]^-1,
    tau_0 = 8 (1+nu)/(5+nu) mu' (d/a) rho_bar,  mu' = mu/(rho_bar g a).
    ~0.994 for the nominal Europa values (the membrane-only estimate used
    before 2026-10 gave 0.980).
    """
    mu_p = p.rigidity / (p.mean_density * p.gravity * p.radius)
    tau0 = (8.0 * (1.0 + p.poisson_ratio) / (5.0 + p.poisson_ratio)
            * mu_p * (p.lithosphere_thickness / p.radius) * p.mean_density)
    return 1.0 / (1.0 + tau0 / (p.water_density
                                * (1.0 - 0.6 * p.water_density / p.mean_density)))


def iceberg_bracket(p, cw=None, shell_thickness=None):
    """The Eq. 2.18 bracket  -(1 - C_w) + 4 (b_c/a) (1 - (drho/rho_m) C_w).

    First term: elastic support of a bottom load (negative); second: the
    iceberg effect (surface rise of a thickened shell).
    """
    cw = willemann_cw(p) if cw is None else cw
    bc = p.shell_thickness if shell_thickness is None else shell_thickness
    return (-(1.0 - cw)
            + 4.0 * (bc / p.radius) * (1.0 - (p.delta_rho / p.water_density) * cw))


def compensation_factor(p, cw=None, shell_thickness=None):
    """Ratio of the Eq. 2.18 C_2m to the uncompensated-load value Q_2m.

    C_2m = Q_2m (drho/rho_c) [bracket], Q_2m = 3 rho_c t_2m/(5 rho_bar a)
    (Eq. 2.9). Multiply the geometric inertia tensor of an ice layer of
    thickness t (density rho_c, rigidly at the surface) by this factor to get
    the inertia of the bottom-loaded, buoyantly compensated shell. ~0.004 for
    nominal Europa: 14-21x smaller than the pre-2026-10 code's factor, which
    multiplied the iceberg term by rho_c/drho and used the membrane C_w.
    """
    return (p.delta_rho / p.ice_density) * iceberg_bracket(p, cw, shell_thickness)


def shell_c2m(t2m, p, cw=None, shell_thickness=None):
    """Gravity coefficient C_2m of a thickness harmonic t_2m [m] (Eq. 2.18)."""
    q2m = 3.0 * p.ice_density * t2m / (5.0 * p.mean_density * p.radius)
    return q2m * compensation_factor(p, cw, shell_thickness)


def fossil_c20_c22(p, cw=None):
    """Fossil bulge gravity coefficients, Eq. 3.10.

    C20 = -(1/2) h gamma (rho_m/rho_bar)(1 - C_w),
    C22 = (3/20) h gamma (rho_m/rho_bar)(1 - C_w).
    The bulge is hydrostatic in shape (Eqs. 3.5-3.6) but attenuated by the
    elastic lithosphere to (1 - C_w) (Eq. 3.8).
    """
    cw = willemann_cw(p) if cw is None else cw
    f = p.love_h * gamma_tidal(p) * (p.water_density / p.mean_density) * (1.0 - cw)
    return -0.5 * f, 0.15 * f


def inertia_from_c2(c20, c22, p):
    """Deviatoric inertia tensor [kg m^2] of (C20, C22), Eqs. 2.12-2.14.

    I_xx = Ma^2 (C20/3 - 2 C22), I_yy = Ma^2 (C20/3 + 2 C22),
    I_zz = -(2/3) Ma^2 C20 (the isotropic I' is dropped).
    """
    ma2 = p.mass * p.radius**2
    return ma2 * np.diag([c20 / 3.0 - 2.0 * c22, c20 / 3.0 + 2.0 * c22,
                          -2.0 / 3.0 * c20])


def fossil_inertia_tensor(p, cw=None, rotation=None):
    """Body-fixed fossil-bulge inertia tensor [kg m^2] (Eqs. 3.8-3.10).

    ``rotation`` (3x3) maps the frame the bulge froze in to the body frame;
    default identity (frozen with x toward Jupiter, z along the spin axis).
    The tensor is FIXED in the body (shell) frame: it rotates with the shell
    during polar wander, which is what makes it a restoring term.
    """
    I = inertia_from_c2(*fossil_c20_c22(p, cw), p)
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        I = R @ I @ R.T
    return I


def shell_inertia_tensor(b, theta, lam, p, cw=None, shell_thickness=None):
    """Physical (Eq. 2.18) deviatoric inertia tensor of a thickness field [kg m^2].

    The geometric tensor of an ice layer b (density rho_c at radius a) is
    scaled by ``compensation_factor``. Only the degree-2 part of b reaches a
    deviatoric inertia tensor, so the scaling is exact for the l = 2
    harmonics that Eq. 2.18 describes.
    """
    from .grid import thin_shell_inertia_tensor, deviatoric
    geo = thin_shell_inertia_tensor(b, theta, lam) * p.ice_density * p.radius**4
    return deviatoric(geo) * compensation_factor(p, cw, shell_thickness)


def stability_test(t20_2t22_m, p, cw=None):
    """Quasi-static I_yy > I_zz test (OSPW Eqs. 3.4, 3.11, 3.12).

    Parameters
    ----------
    t20_2t22_m : float
        Thickness combination t20 + 2 t22 [m] of the thermal-equilibrium shell.

    Returns
    -------
    dict with
        'unstable' : bool -- exact test, (C20 + 2 C22)_shell+fossil > 0 (Eq. 3.4)
        'ratio_eq311' : float -- left side of Eq. 3.11 (> 1 means unstable)
        'threshold_km_eq311' : t20 + 2 t22 needed by Eq. 3.11 for these params
        'threshold_km_eq312' : the same from the printed Eq. 3.12 numbers
        'threshold_km_exact' : the same from the exact C_w chain
        'sum_shell', 'sum_fossil' : C20 + 2 C22 of each part
    """
    cw = willemann_cw(p) if cw is None else cw
    a, h, gam = p.radius, p.love_h, gamma_tidal(p)
    rc, rm, rb = p.ice_density, p.water_density, p.mean_density
    nu, mu = p.poisson_ratio, p.rigidity
    bc, d = p.shell_thickness, p.lithosphere_thickness
    # Eq. 3.11
    pre = 3.0 * (rm - rc) / (h * gam * rm) / a
    brk = -1.0 + 0.5 * (5.0 + nu) / (1.0 + nu) * (rc / rb) * (1.0 - 0.6 * rm / rb) \
        * (rb * p.gravity * a / mu) * (bc / d)
    ratio = t20_2t22_m * pre * brk
    thr311 = np.inf if brk <= 0 else 1.0 / (pre * brk) / 1e3
    # Eq. 3.12 as printed: (x/1 km)(0.243)[-1 + 0.75 (4e10 dyn cm^-2/mu)(b_c/d)] > 1
    brk312 = -1.0 + 0.75 * (4e9 / mu) * (bc / d)
    thr312 = np.inf if brk312 <= 0 else 1.0 / (0.243 * brk312)
    # Exact chain: shell (Eq. 2.18) + fossil (Eq. 3.10)
    c20f, c22f = fossil_c20_c22(p, cw)
    sum_fossil = c20f + 2.0 * c22f
    sum_shell = shell_c2m(t20_2t22_m, p, cw)
    per_m = shell_c2m(1.0, p, cw)
    thr_exact = np.inf if per_m <= 0 else -sum_fossil / per_m / 1e3
    return {
        'unstable': bool(sum_shell + sum_fossil > 0),
        'ratio_eq311': float(ratio),
        'threshold_km_eq311': float(thr311),
        'threshold_km_eq312': float(thr312),
        'threshold_km_exact': float(thr_exact),
        'sum_shell': float(sum_shell),
        'sum_fossil': float(sum_fossil),
        'Cw': float(cw),
    }


def moment_I0(p):
    """I_0 ~ (8 pi/3) rho b_0 a^4, Eq. 5.29 (shell + ocean, b_0 ~ 100 km)."""
    return 8.0 * np.pi / 3.0 * p.i0_density * p.i0_thickness * p.radius**4


def viscosity(T, p):
    """Eq. 5.2: eta = eta0 exp[l (T_m/T - 1)] [Pa s]."""
    return p.melting_viscosity * np.exp(p.activation_l * (p.melting_temp / T - 1.0))


def dissipation_cf(p, Tf=None, shell_thickness=None):
    """Damping coefficient C_f of a fractured shell, Eq. 5.15 [kg m^2 / s].

    C_f = 4 pi a^2 b_c (ln 10 / l) (T_m/(T_m - T_s)) (T_f/T_m)^2
          eta0 exp[l (T_m/T_f - 1)] gamma^2.
    (The ln 10 comes from Eq. 5.14; the pre-2026-10 code omitted it.)
    """
    Tf = p.fracture_temp if Tf is None else Tf
    bc = p.shell_thickness if shell_thickness is None else shell_thickness
    Tm, Ts, l = p.melting_temp, p.surface_temp, p.activation_l
    return (4.0 * np.pi * p.radius**2 * bc * (np.log(10.0) / l) * (Tm / (Tm - Ts))
            * (Tf / Tm)**2 * viscosity(Tf, p) * gamma_tidal(p)**2)


def t_over_tau_equilibrium(t20_2t22_m, p, cw=None, fossil=True):
    """(t/tau)_equil = (I_yy - I_zz)/I_0, Eqs. 5.31-5.32 (exact chain).

    Eq. 5.32's approximation reads 1e-5 [(t20+2t22)/1 km] - 0.6e-5 (mu/4e10)(d/2 km).
    """
    cw = willemann_cw(p) if cw is None else cw
    s = shell_c2m(t20_2t22_m, p, cw)
    if fossil:
        c20f, c22f = fossil_c20_c22(p, cw)
        s += c20f + 2.0 * c22f
    return p.mass * p.radius**2 * s / moment_I0(p)


def epsilon(t20_m, t22_m, p, cw=None, fossil=True):
    """epsilon = (I_zz - I_xx)/I_0, Eqs. 5.33-5.34."""
    cw = willemann_cw(p) if cw is None else cw
    c20 = shell_c2m(t20_m, p, cw)
    c22 = shell_c2m(t22_m, p, cw)
    if fossil:
        c20f, c22f = fossil_c20_c22(p, cw)
        c20, c22 = c20 + c20f, c22 + c22f
    return p.mass * p.radius**2 * (-c20 + 2.0 * c22) / moment_I0(p)


def tau_f(p, Tf=None):
    """Overdamped polar-wander time scale tau_f = [C_f tau/(n^2 I_0)]^(1/2), Eq. 5.41 [yr].

    tau is the t/tau growth parameter (~1e12 yr, p. 262), not tau_th.

    Note: integrating the overdamped Eq. 5.39,
    psi_dot = (t/tau)(n^2 I_0/2C_f) sin 2psi, gives
    tan psi = tan psi_0 exp[t^2 / (2 tau_f^2)] with tau_f as defined here,
    i.e. half the exponent printed in Eq. 5.41. The figure and Eq. 5.42 use
    the printed form. ``psi_overdamped`` integrates Eq. 5.39 itself.
    """
    tau_s = p.tau_growth_yr * YEAR_S
    return np.sqrt(dissipation_cf(p, Tf) * tau_s / (C.MEAN_MOTION**2 * moment_I0(p))) / YEAR_S


def psi_overdamped(t_yr, psi0, p, Tf=None):
    """Closed-form solution of Eq. 5.39 with t/tau = t/tau_growth [rad]."""
    k = (C.MEAN_MOTION**2 * moment_I0(p) / dissipation_cf(p, Tf)) / (p.tau_growth_yr * YEAR_S)
    t_s = np.asarray(t_yr, dtype=float) * YEAR_S
    return np.arctan(np.tan(psi0) * np.exp(0.5 * k * t_s**2))


def undamped_time_scale(eps, psi0, p):
    """Eq. 4.22: T ~ [tau/(n^2 eps)]^(1/3) (-ln psi0)^(1/3) [yr]."""
    tau_s = p.tau_growth_yr * YEAR_S
    return (tau_s / (C.MEAN_MOTION**2 * abs(eps)))**(1.0 / 3.0) \
        * (-np.log(psi0))**(1.0 / 3.0) / YEAR_S


def overdamped_step(spin, I_dev, cf, dt_s):
    """Advance the spin axis by dt under the overdamped Eq. 5.39 (exact for fixed I).

    Eq. 5.39 in the overdamped limit is psi_dot = (n^2/2C_f)(I_b - I_a) sin 2psi
    for a pole at angle psi from principal axis a toward axis b (OSPW write
    I_b - I_a = (t/tau) I_0). Its vector form is the gradient flow
    s_dot = (n^2/C_f) [I s - (s.I s) s], whose solution for constant I is
    s(t) = exp(k I t) s0 / |exp(k I t) s0|, k = n^2/C_f. That closed form is
    used here, so the step is exact and unconditionally stable for the frozen
    tensor (no explicit-Euler growth error, no step-size cap needed for
    accuracy).

    Parameters
    ----------
    spin : (3,) unit vector, body frame
    I_dev : (3, 3) physical inertia tensor [kg m^2] (deviatoric is enough)
    cf : float, C_f [kg m^2/s]
    dt_s : float, step [s]

    Returns
    -------
    new_spin : (3,) unit vector
    info : dict with the moment pair (I_spin, I_target), psi_dot [rad/s]
           at the start of the step, and the principal axes.
    """
    k = C.MEAN_MOTION**2 / cf
    vals, vecs = np.linalg.eigh(I_dev)
    c = vecs.T @ spin
    expo = k * dt_s * (vals - vals.max())
    v = vecs @ (np.exp(expo) * c)
    new = v / np.linalg.norm(v)
    Is = spin @ I_dev @ spin
    g = I_dev @ spin - Is * spin                      # projected gradient
    gnorm = np.linalg.norm(g)
    a = int(np.argmax(np.abs(c)))                     # axis the pole sits on
    info = {
        'eigenvalues': vals,
        'axes': vecs,
        'i_axis': a,
        'I_spin': float(vals[a]),
        'I_target': float(vals[-1]) if a != len(vals) - 1 else float(vals[a]),
        'psi_dot': float(k * gnorm),
        'unstable': bool(a != len(vals) - 1),
    }
    return new, info
