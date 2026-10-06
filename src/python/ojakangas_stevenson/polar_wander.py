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
* Eqs. 5.19-5.21 / 5.26 / 5.38  the same with the tidal torque from Jupiter
            (``overdamped_step_tidal``; stiffness 4 n^2 eps toward Jupiter),
            and the quasi-static sub-Jovian meridian (``meridian_min_moment``)
* Stefan-limited shell relaxation time f rho L b / q, f = exact eigenmode
            factor or 1 + S/L (``stefan_relaxation_time``, ``stefan_eigen_factor``,
            ``sensible_heat_ratio``)

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


def projected_meridian(spin, reference=(1.0, 0.0, 0.0)):
    """Unit vector of ``reference`` projected into the plane perpendicular to spin.

    The pre-2026-10-05 sub-Jovian meridian: the body x axis carried into the
    current equatorial plane. Falls back to body y when spin ~ reference.
    """
    s = np.asarray(spin, dtype=float)
    r = np.asarray(reference, dtype=float)
    e = r - (r @ s) * s
    n = np.linalg.norm(e)
    if n < 1e-10:
        e = np.array([0.0, 1.0, 0.0]) - s[1] * s
        n = np.linalg.norm(e)
    return e / n


def meridian_min_moment(I_dev, spin, reference, rel_tol=1e-12):
    """Sub-Jovian direction = minimum-moment axis in the plane perpendicular to spin.

    In synchronous rotation the tidal torque -3 n^2 (I_j - I_k) (OSPW
    Eqs. 5.19-5.21) turns the shell about its spin axis until the smallest
    in-plane moment faces Jupiter (stiffness 3 n^2 (B - A)). This returns that
    axis, assuming the libration about the spin axis is quasi-static.

    ``reference`` fixes the sign (continuity with the previous meridian:
    sub-Jovian vs anti-Jovian, hence which hemisphere trails) and is
    returned (projected) when the in-plane moments are degenerate to
    ``rel_tol`` of the largest tensor element.
    """
    s = np.asarray(spin, dtype=float)
    u = projected_meridian(s, reference)
    v = np.cross(s, u)
    I = np.asarray(I_dev, dtype=float)
    M = np.array([[u @ I @ u, u @ I @ v], [v @ I @ u, v @ I @ v]])
    vals, vecs = np.linalg.eigh(M)
    if vals[1] - vals[0] <= rel_tol * max(np.abs(I).max(), 1e-300):
        return u
    a, b = vecs[:, 0]
    e = a * u + b * v
    if e @ u < 0:
        e = -e
    return e / np.linalg.norm(e)


def _tidal_rhs(s, I, k, meridian_fn):
    """s_dot of the overdamped flow with centrifugal and tidal torques.

    Torque on the shell (body frame), for spin s and sub-Jovian unit vector
    e (perpendicular to s):  tau = n^2 (I s) x s - 3 n^2 (I e) x e
    (OSPW Eqs. 5.19-5.21: the rotational potential -(n^2/2) s.I.s plus the
    tidal potential (3 n^2/2) e.I.e). Overdamped, C_f omega = tau, and
    s_dot = s x omega gives
        s_dot = (n^2/C_f) [I s - (s.I s) s + 3 (e.I s) e].
    """
    e = meridian_fn(s)
    Is = I @ s
    return k * (Is - (s @ Is) * s + 3.0 * (e @ Is) * e)


def overdamped_step_tidal(spin, I_dev, cf, dt_s, meridian_fn, max_growth=0.1):
    """Overdamped reorientation (Eq. 5.39) WITH the tidal torque from Jupiter.

    ``overdamped_step`` keeps only the rotational (centrifugal) torque, so a
    tilt of the pole toward Jupiter is resisted with stiffness n^2 (C - A)
    instead of the 4 n^2 (C - A) of OSPW Eqs. 5.26/5.38. Here the torque is
    the full one (see ``_tidal_rhs``); the tilt toward the trailing apex
    keeps stiffness n^2 (C - B) (the tidal term vanishes for it).

    ``meridian_fn(s)`` returns the sub-Jovian unit vector for pole s
    (``projected_meridian`` or a frozen ``meridian_min_moment``).

    Integration: classical RK4 with substeps chosen so that the largest
    possible growth factor per substep, 4 k (I_max - I_min) dt_sub, is
    <= max_growth (k = n^2/C_f), and renormalization after each substep.
    No closed form exists because e depends on s.

    Returns (new_spin, info) like ``overdamped_step``.
    """
    k = C.MEAN_MOTION**2 / cf
    I = np.asarray(I_dev, dtype=float)
    vals, vecs = np.linalg.eigh(I)
    spread = vals[-1] - vals[0]
    n_sub = max(1, int(np.ceil(4.0 * k * spread * dt_s / max_growth)))
    h = dt_s / n_sub
    s = np.asarray(spin, dtype=float).copy()
    rate0 = np.linalg.norm(_tidal_rhs(s, I, k, meridian_fn))
    for _ in range(n_sub):
        k1 = _tidal_rhs(s, I, k, meridian_fn)
        k2 = _tidal_rhs(_unit(s + 0.5 * h * k1), I, k, meridian_fn)
        k3 = _tidal_rhs(_unit(s + 0.5 * h * k2), I, k, meridian_fn)
        k4 = _tidal_rhs(_unit(s + h * k3), I, k, meridian_fn)
        s = _unit(s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    c = vecs.T @ np.asarray(spin, dtype=float)
    a = int(np.argmax(np.abs(c)))
    info = {
        'eigenvalues': vals,
        'axes': vecs,
        'i_axis': a,
        'I_spin': float(vals[a]),
        'I_target': float(vals[-1]) if a != len(vals) - 1 else float(vals[a]),
        'psi_dot': float(rate0),
        'unstable': bool(a != len(vals) - 1),
        'n_substeps': n_sub,
    }
    return s, info


def _unit(v):
    return v / np.linalg.norm(v)


def tidal_frame_t20_2t22(I_shell, spin, sub_jovian, p, cw=None):
    """Thickness-equivalent t20 + 2 t22 [m] of a shell tensor in the tidal frame.

    (I_trail - I_spin) of the Eq. 2.18 shell tensor, with trail = spin x
    sub_jovian, divided by M a^2 C_2m(1 m). Equals the harmonic t20 + 2 t22
    when the tidal frame is the body frame (I_yy - I_zz = M a^2 (C20 + 2 C22))
    and is defined for any pole, so it tracks the quantity of the Eq. 3.12
    test along a polar-wander path.
    """
    s = np.asarray(spin, dtype=float)
    e = projected_meridian(s, sub_jovian)
    tr = np.cross(s, e)
    d = tr @ I_shell @ tr - s @ I_shell @ s
    return d / (p.mass * p.radius**2 * shell_c2m(1.0, p, cw))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def sensible_heat_ratio(T_s, T_m=None, latent_heat=None):
    """S/L for the steady conductive profile of a shell with surface T_s, exactly.

    When the base advances by db the whole profile, which scales with b for
    the k = a1/T + a0 shell without internal heating (z/b = int_Ts^T k dT / F),
    must also be re-equilibrated. The energy per unit area to make a shell of
    thickness b from melt at T_m is rho b [L + S], with
        S = int_0^1 [h(T_m) - h(T(zeta))] dzeta
          = (1/F) int_Ts^Tm [h(T_m) - h(T)] k(T) dT,   h(T) = int c_p dT,
    c_p = 90 + 7.49 T (constants.ice_heat_capacity), F = int_Ts^Tm k dT.
    Evaluated by 48-point Gauss-Legendre in T (smooth integrand; exact to
    ~1e-12). Modern k = 651/T: S/L = 0.530 / 0.503 / 0.477 at T_s = 90 / 100
    / 110 K (Fermi, 2026-10-05). Returns an array shaped like T_s.
    """
    T_m = C.T_MELT if T_m is None else T_m
    L = C.ICE_LATENT_HEAT if latent_heat is None else latent_heat
    Ts = np.asarray(T_s, dtype=float)
    c0, c1 = C.ICE_CP_COEFFS
    h = lambda T: c0 * T + 0.5 * c1 * T * T                     # noqa: E731
    k = lambda T: C.A1_CONDUCTIVITY / T + C.A0_CONDUCTIVITY     # noqa: E731
    half = 0.5 * (T_m - Ts)[..., None]
    T = 0.5 * (T_m + Ts)[..., None] + half * _GL_X
    num = np.sum(_GL_W * (h(T_m) - h(T)) * k(T), axis=-1) * half[..., 0]
    F = C.A1_CONDUCTIVITY * np.log(T_m / Ts) + C.A0_CONDUCTIVITY * (T_m - Ts)
    return num / F / L


_TABLES = {}
_T_TABLE = np.arange(30.0, 272.0 + 1e-9, 0.02)        # T_s grid for the S/L table [K]
_T_EIG = np.arange(30.0, 272.0 + 1e-9, 1.0)           # T_s grid for the eigenmode table [K]


def _table_key(kind):
    return (kind, C.A1_CONDUCTIVITY, C.A0_CONDUCTIVITY, tuple(C.ICE_CP_COEFFS), C.ICE_LATENT_HEAT,
            C.T_MELT, C.ICE_DENSITY)


def sensible_heat_ratio_tabulated(T_s):
    """``sensible_heat_ratio`` interpolated from a 0.02 K table (built once per constant set).

    Agrees with the quadrature to < 1e-9 (tested); removes the per-cell
    quadrature that cost 45% of a coupled run (Fermi F10).
    """
    key = _table_key('S/L')
    if key not in _TABLES:
        _TABLES[key] = sensible_heat_ratio(_T_TABLE)
    return np.interp(np.asarray(T_s, dtype=float), _T_TABLE, _TABLES[key])


def _kirchhoff_profile(Ts, zeta):
    """T(zeta) of the steady k = a1/T + a0 profile: a1 ln(T/Ts) + a0 (T - Ts) = zeta F."""
    T_m = C.T_MELT
    a1, a0 = C.A1_CONDUCTIVITY, C.A0_CONDUCTIVITY
    F = a1 * np.log(T_m / Ts) + a0 * (T_m - Ts)
    T = Ts * (T_m / Ts) ** zeta                          # exact for a0 = 0; Newton otherwise
    for _ in range(50 if a0 else 0):
        g = a1 * np.log(T / Ts) + a0 * (T - Ts) - zeta * F
        T = T - g / (a1 / T + a0)
    return T, F


def stefan_eigen_factor(T_s, n=600):
    """Exact slowest-mode relaxation time of the conductive Stefan shell, in units of rho L b^2 / F.

    Linearized about the steady conductive profile, a perturbation of the
    Kirchhoff variable u = int k dT obeys (rho c_p / k) du/dt = u'' with u = 0
    at the surface and the Stefan condition rho L db/dt = -(flux jump) at the
    base. Its slowest decay rate lambda solves the symmetric generalized
    eigenproblem  int v' phi' dz = lambda [ int (rho c_p/k) v phi dz +
    (rho L b / F) v(b) phi(b) ]  (linear finite elements, n elements). Returns
    1 / (lambda rho L b^2 / F), independent of b. Modern k = 651/T: ~1.40 at
    100 K (Fermi's shooting, stefan_eig2.py), vs 1 + S/L = 1.50 (the
    sensible-heat bound, ~8% slow). Validated against the constant-k, c_p
    case tau/tau_L = St/x^2, x tan x = St (test).
    """
    from scipy.linalg import eigh
    Ts = float(T_s)
    zeta_n = np.linspace(0.0, 1.0, n + 1)
    zeta_m = 0.5 * (zeta_n[1:] + zeta_n[:-1])
    T_m, F = _kirchhoff_profile(Ts, zeta_m)
    c0, c1 = C.ICE_CP_COEFFS
    w = C.ICE_DENSITY * (c0 + c1 * T_m) / (C.A1_CONDUCTIVITY / T_m + C.A0_CONDUCTIVITY)
    # b = 1 m (the factor is b-independent); element length h
    h = 1.0 / n
    K = np.zeros((n + 1, n + 1))
    M = np.zeros((n + 1, n + 1))
    for e in range(n):
        i = slice(e, e + 2)
        K[i, i] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        M[i, i] += w[e] * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    M[n, n] += C.ICE_DENSITY * C.ICE_LATENT_HEAT * 1.0 / F
    lam = eigh(K[1:, 1:], M[1:, 1:], eigvals_only=True, subset_by_index=[0, 0])[0]
    tau_L = C.ICE_DENSITY * C.ICE_LATENT_HEAT * 1.0 / F
    return 1.0 / (lam * tau_L)


def stefan_eigen_factor_tabulated(T_s):
    """``stefan_eigen_factor`` interpolated from a 1 K table (built once per constant set, ~1 s)."""
    key = _table_key('eig')
    if key not in _TABLES:
        _TABLES[key] = np.array([stefan_eigen_factor(T, n=300) for T in _T_EIG])
    return np.interp(np.asarray(T_s, dtype=float), _T_EIG, _TABLES[key])


def stefan_relaxation_time(b, T_s, T_m=None, latent_heat=None, density=None,
                           sensible='eigen'):
    """Stefan-limited relaxation time of the shell thickness [s].

    tau = f rho L b^2 / F,  F = a1 ln(T_m/T_s) + a0 (T_m - T_s)  (q_cond = F/b).
    A thickness anomaly decays only as fast as the basal flux imbalance can
    melt or freeze ice and re-equilibrate the profile. Linearized about
    equilibrium (dq/db = -q/b) with latent heat alone, f = 1; the factor f
    carries the sensible heat:

    ``sensible``:
      'eigen' (default since 2026-10-05, Fermi F6): f = the exact slowest
          eigenmode of the conductive Stefan problem (``stefan_eigen_factor``,
          tabulated in T_s); ~1.40 at 100 K, ~19 Myr for 30 km.
      'exact': f = 1 + S/L, S/L exact for the steady profile
          (``sensible_heat_ratio``, tabulated). An UPPER estimate: it puts the
          whole profile's sensible heat on the base, ~8% slower than 'eigen'.
      'none': f = 1 (latent heat only, the 679af84 form, ~13 Myr for 30 km).
    O&S/OSPW quote "<~ a few x 1e7 yr" (OSPW p. 247). Not included: the change
    of internal tidal heating with b, which modifies dq/db.
    """
    T_m = C.T_MELT if T_m is None else T_m
    L = C.ICE_LATENT_HEAT if latent_heat is None else latent_heat
    rho = C.ICE_DENSITY if density is None else density
    T_s = np.asarray(T_s, dtype=float)
    F = C.A1_CONDUCTIVITY * np.log(T_m / T_s) + C.A0_CONDUCTIVITY * (T_m - T_s)
    if sensible == 'eigen':
        fac = stefan_eigen_factor_tabulated(T_s)
    elif sensible == 'exact':
        fac = 1.0 + sensible_heat_ratio_tabulated(T_s)
    elif sensible == 'none':
        fac = 1.0
    else:
        raise ValueError(f"sensible must be 'eigen', 'exact' or 'none', got {sensible!r}")
    return fac * rho * L * np.asarray(b, dtype=float) ** 2 / F
