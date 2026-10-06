"""Tests for the O&S ice-shell thickness and polar-wander code (2026-10 fixes).

Run either way (CPU only, ~1-2 minutes):
    python -m pytest src/python/tests/test_ojakangas_stevenson.py
    python src/python/tests/test_ojakangas_stevenson.py

Each test names the defect it would have caught (physics review, 2026-10-03).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src', 'python'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from ojakangas_stevenson import constants as C                      # noqa: E402
from ojakangas_stevenson import grid as G                           # noqa: E402
from ojakangas_stevenson import polar_wander as PW                  # noqa: E402
from ojakangas_stevenson.rheology import MaxwellRheology, FlowLawRheology  # noqa: E402
from ojakangas_stevenson.shell_thickness import (                   # noqa: E402
    equilibrium_thickness, dissipation_field, table1_case, IceShellModel)
from ojakangas_stevenson.surface_temperature import OSSurfaceTemperature  # noqa: E402
from ojakangas_stevenson.ptm3d_coupling import PTM3DSurfaceTemperature    # noqa: E402
from ojakangas_stevenson import validate as V                       # noqa: E402


# --------------------------------------------------------------------------- thickness law
def test_c_r_is_paper_value():
    assert C.C_R == 0.5


def test_eq28_closed_form_and_not_legacy_quadratic():
    """Eq. 2.8: b = ln(TM/Ts)/sqrt(2D/a1 + (H/a1)^2). The legacy solver
    (b^2 D/ln + b H = a1 ln) is sqrt(2) thicker at H = 0."""
    Ts, D, a1 = 100.0, 2.0e-3, 488.0
    ln = np.log(273.0 / Ts)
    for H in (0.0, 0.01):
        b = equilibrium_thickness(Ts, D, H, a1=a1)
        assert np.isclose(b, ln / np.sqrt(2 * D / a1 + (H / a1) ** 2), rtol=1e-14)
    legacy = ln * np.sqrt(a1 / D)
    assert np.isclose(legacy / equilibrium_thickness(Ts, D, 0.0, a1=a1), np.sqrt(2.0))


def test_thickness_refuses_bad_input():
    for bad in (273.0, 300.0, np.nan, 0.0):
        try:
            equilibrium_thickness(bad, 1e-3, 0.0)
        except ValueError:
            continue
        raise AssertionError(f"T_s={bad} accepted")
    try:
        equilibrium_thickness(100.0, 0.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("D = H = 0 accepted")


def _exact_by_quadrature(Ts, D, H, F, a1=488.0, TM=273.0):
    """Independent check of the exact first integral (Eqs. 2.4/2.5) by scipy.quad."""
    from scipy.integrate import quad
    yp = lambda T: np.sqrt(2.0 * D / a1 * F(T) + (H / a1) ** 2)  # noqa: E731
    pts = [TM * 0.9, TM * 0.97, TM * 0.995]
    return quad(lambda T: 1.0 / (T * yp(T)), Ts, TM, limit=500, points=pts)[0]


def test_exact_first_integral_matches_quadrature():
    for rheology in (MaxwellRheology(), FlowLawRheology()):
        F = lambda T, r=rheology: r.dissipation_fraction_above(T, 273.0)  # noqa: E731
        for Ts in (52.0, 108.8):
            for H in (0.0, 0.01):
                for D in (2e-7, 4e-7, 1e-6):     # b ~ 15-60 km: the physical range
                    b = equilibrium_thickness(Ts, D, H, method='exact', rheology=rheology)
                    ref = _exact_by_quadrature(Ts, D, H, F)
                    assert abs(b - ref) / ref < 1e-9, (rheology, Ts, H, D, b, ref)
                    # exact is thicker than Eq. 2.8 (y' grows toward the surface)
                    assert b > equilibrium_thickness(Ts, D, H)


def test_exact_needs_rheology():
    try:
        equilibrium_thickness(100.0, 1e-3, method='exact')
    except ValueError:
        return
    raise AssertionError("method='exact' without rheology accepted")


def test_table1_gate_paper_mode():
    """All twelve Table I entries within 5% or 0.1 km (paper constants, C_R=0.5)."""
    with C.parameter_set('paper'):
        models = V.build_models()
        ok, res = V.validate_thickness(models, n_theta=181, n_lam=360)
    assert ok, res


def test_table1_gate_rejects_legacy_formula():
    """The gate is discriminating: the legacy quadratic fails it at C_R=0.5."""
    with C.parameter_set('paper'):
        theta, lam = G.periodic_grid(91, 180)
        TH, LA = np.meshgrid(theta, lam, indexing='ij')
        Ts = OSSurfaceTemperature().temperature(TH)
        D = dissipation_field(MaxwellRheology(), TH, LA)
        legacy = np.log(273.0 / Ts) * np.sqrt(C.A1_CONDUCTIVITY / D)
        avg = G.area_mean(legacy, theta, lam) / 1e3
    assert not V.entry_passes(avg, 23.0), avg


# --------------------------------------------------------------------------- quadrature
def test_periodic_grid_has_no_seam_duplicate():
    theta, lam = G.periodic_grid(181, 360)
    assert lam[0] == -np.pi and lam[-1] < np.pi
    assert np.isclose(G.area_weights(theta, lam).sum(), 4 * np.pi, rtol=1e-14)
    try:
        G.check_periodic(np.linspace(-np.pi, np.pi, 361))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicated-seam grid accepted")


def test_pure_tidal_forcing_has_no_odd_harmonics():
    """Pure O&S tidal forcing is symmetric under lam -> lam + pi and N-S, so
    b11c (and b11s, b10, b21c, b21s, b22s) must vanish. The old grid (seam
    counted twice) gave b11c = -0.08 km."""
    with C.parameter_set('paper'):
        model = IceShellModel(MaxwellRheology())
        theta, lam, b = model.thickness_grid(181, 360)
    h = G.harmonics(b, theta, lam)
    for k in ('b11c', 'b11s', 'b10', 'b21c', 'b21s', 'b22s'):
        assert abs(h[k]) / 1e3 < 1e-12, (k, h[k])
    # and the legacy rectangle sum over the duplicated grid does not vanish
    th_old = np.linspace(0.001, np.pi - 0.001, 181)
    lm_old = np.linspace(-np.pi, np.pi, 361)
    TH, LA = np.meshgrid(th_old, lm_old, indexing='ij')
    with C.parameter_set('paper'):
        b_old = equilibrium_thickness(OSSurfaceTemperature().temperature(TH),
                                      dissipation_field(MaxwellRheology(), TH, LA))
    st = np.sin(th_old)[:, None]
    b11c_old = 3 / (4 * np.pi) * np.sum(b_old * st * np.cos(LA) * st) \
        * (th_old[1] - th_old[0]) * (lm_old[1] - lm_old[0])
    assert abs(b11c_old) > 10.0, b11c_old           # tens of meters


def test_projection_recovers_known_harmonics():
    theta, lam = G.periodic_grid(91, 180)
    Y = G.basis_functions(theta, lam)
    coef = {'b00': 20e3, 'b10': 3.0, 'b11c': -7.0, 'b11s': 11.0, 'b20': -500.0,
            'b21c': 2.0, 'b21s': -4.0, 'b22c': 1600.0, 'b22s': 9.0}
    field = sum(c * Y[k] for k, c in coef.items())
    h = G.harmonics(field, theta, lam)
    for k, c in coef.items():
        assert abs(h[k] - c) < 1e-9, (k, h[k], c)


def test_uniform_shell_is_isotropic():
    """A uniform shell has no deviatoric inertia (the midpoint rule faked one)."""
    theta, lam = G.periodic_grid(46, 90)
    b = np.full((46, 90), 23.7e3)
    I = G.thin_shell_inertia_tensor(b, theta, lam)
    assert np.abs(G.deviatoric(I)).max() / np.trace(I) < 1e-14
    assert abs(G.harmonics(b, theta, lam)['t20']) < 1e-9


# --------------------------------------------------------------------------- seam padding
def _fibonacci_faces(n):
    k = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * k / n))
    lon = np.degrees((np.pi * (1 + 5 ** 0.5) * k) % (2 * np.pi))   # [0, 360)
    return lat, lon


def test_temperature_2d_is_seam_free():
    """A smooth field interpolated by temperature_2d has no seam artifact."""
    lat, lon = _fibonacci_faces(40000)
    col, lr = np.radians(90 - lat), np.radians(lon)
    f = 100 + 10 * np.sin(col) * np.cos(lr) + 5 * np.sin(col) ** 2 * np.sin(2 * lr)
    obj = PTM3DSurfaceTemperature.from_arrays(f, lat, lon)
    theta, lam = G.periodic_grid(91, 180)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    truth = 100 + 10 * np.sin(TH) * np.cos(LA) + 5 * np.sin(TH) ** 2 * np.sin(2 * LA)
    err = np.abs(obj.temperature_2d(theta, lam) - truth)
    interior = err[5:-5][:, 10:-10].max()
    seam = err[5:-5][:, np.r_[0:3, -3:0]].max()
    assert seam < 3 * interior, (seam, interior)
    # without padding the seam columns are much worse
    obj_nopad = obj.temperature_2d(theta, lam, seam_pad_deg=0.0)
    seam_nopad = np.abs(obj_nopad - truth)[5:-5][:, np.r_[0:3, -3:0]].max()
    assert seam_nopad > 3 * seam, (seam_nopad, seam)


# --------------------------------------------------------------------------- surface T
def test_polar_flux_is_eq_A15():
    m = OSSurfaceTemperature()
    i = m.obliquity
    assert np.isclose(m.annual_average_flux(0.0), 50 * i / (np.sqrt(2) * np.pi))
    assert np.isclose(m.annual_average_flux(np.pi), 50 * i / (np.sqrt(2) * np.pi))
    # A.15 meets A.9 at phi = i (to O(i^3))
    lo, hi = m.annual_average_flux(i - 1e-9), m.annual_average_flux(i + 1e-9)
    assert abs(lo - hi) / hi < i ** 2
    assert np.isclose(float(m.temperature(0.0)), 47.74, atol=0.01)
    legacy = OSSurfaceTemperature(polar_flux='legacy-max')
    assert np.isclose(float(legacy.temperature(0.0)), 52.06, atol=0.01)


# --------------------------------------------------------------------------- constants
def test_parameter_sets():
    assert C.active_parameter_set().name == 'paper'
    with C.parameter_set('modern') as ps:
        assert (C.ECCENTRICITY, C.EUROPA_RADIUS, C.EUROPA_DENSITY, C.EUROPA_GRAVITY,
                C.A1_CONDUCTIVITY) == (0.0094, 1.5608e6, 3013.0, 1.315, 651.0)
        assert ps.as_dict()['name'] == 'modern'
        # self-consistent: rho and g from the same GM
        assert np.isclose(3 * C.EUROPA_MASS / (4 * np.pi * C.EUROPA_RADIUS ** 3),
                          C.EUROPA_DENSITY, rtol=1e-4)
        assert np.isclose(C.G_GRAV * C.EUROPA_MASS / C.EUROPA_RADIUS ** 2,
                          C.EUROPA_GRAVITY, rtol=1e-3)
    with C.parameter_set('modern-klinger'):
        assert C.A1_CONDUCTIVITY == 567.0
    assert C.active_parameter_set().name == 'paper'
    assert (C.ECCENTRICITY, C.EUROPA_RADIUS, C.A1_CONDUCTIVITY) == (0.01, 1.569e6, 488.0)
    for ps in C.PARAMETER_SETS.values():
        assert all(len(src) == 3 for src in ps.sources)
    try:
        C.use_parameter_set('bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown parameter set accepted')


# --------------------------------------------------------------------------- OSPW
def _p():
    with C.parameter_set('paper'):
        return PW.OSPWParams()


def test_willemann_cw_is_eq_2_11():
    p = _p()
    cw = PW.willemann_cw(p)
    assert np.isclose(cw, 0.99387, atol=2e-5), cw


def test_iceberg_bracket_is_eq_2_18():
    """Eq. 2.18 factor; the old code was 14-21x larger."""
    p = _p()
    cw = PW.willemann_cw(p)
    f = PW.compensation_factor(p)
    expect = (80.0 / 920.0) * (-(1 - cw) + 4 * (20e3 / p.radius) * (1 - (80.0 / 1000.0) * cw))
    assert np.isclose(f, expect, rtol=1e-12)
    legacy = (80.0 / 920.0) * (-(1 - cw) + 4 * cw * (20e3 / p.radius) * (920.0 / 80.0))
    assert 10 < legacy / f < 25, legacy / f


def test_epsilon_chain_reproduces_eq_5_32():
    """(t/tau) ~ 1e-5 per km of (t20 + 2 t22); the fossil contributes -0.6e-5."""
    p = _p()
    per_km = PW.t_over_tau_equilibrium(1000.0, p, fossil=False)
    fossil = PW.t_over_tau_equilibrium(0.0, p, fossil=True)
    assert abs(per_km / 1e-5 - 1) < 0.05, per_km
    assert abs(fossil / -0.6e-5 - 1) < 0.05, fossil
    # Eq. 5.34: fossil epsilon ~ 2e-5 (the paper rounds)
    assert 1.8e-5 < PW.epsilon(0.0, 0.0, p) < 2.6e-5


def test_stability_test_eq_3_12():
    p = _p()
    r = PW.stability_test(1000.0, p)
    assert abs(r['threshold_km_eq312'] - 0.633) < 0.002
    assert abs(r['threshold_km_exact'] - r['threshold_km_eq312']) < 0.05
    assert abs(r['threshold_km_eq311'] - r['threshold_km_eq312']) < 0.05
    assert r['unstable'] and r['ratio_eq311'] > 1
    assert not PW.stability_test(500.0, p)['unstable']


def test_fossil_tensor_is_hydrostatic_and_stabilizing():
    p = _p()
    I = PW.fossil_inertia_tensor(p)
    c20, c22 = PW.fossil_c20_c22(p)
    assert np.isclose(c22 / c20, -0.3)                      # (3/20)/(-1/2)
    assert np.allclose(I, np.diag(np.diag(I)))
    assert I[2, 2] > I[1, 1] > I[0, 0]                      # C > B > A
    # body-fixed: a rotation of the body rotates the tensor
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    assert np.allclose(PW.fossil_inertia_tensor(p, rotation=R), R @ I @ R.T)


def test_shell_tensor_is_compensated_geometric_tensor():
    """A t20 P20 layer: geometric Izz - Ixx = -(4 pi/5) rho_c a^4 t20 (Eqs. 2.9,
    2.12-2.14); the physical tensor is that times the Eq. 2.18 factor."""
    p = _p()
    theta, lam = G.periodic_grid(91, 180)
    t20 = 1000.0
    b = 20e3 + t20 * G.basis_functions(theta, lam)['b20']
    I = PW.shell_inertia_tensor(b, theta, lam, p)
    geo = -(4 * np.pi / 5) * p.ice_density * p.radius ** 4 * t20
    assert np.isclose(I[2, 2] - I[0, 0], geo * PW.compensation_factor(p), rtol=1e-10)
    # and matches Ma^2 (-C20) with C20 from Eq. 2.18
    assert np.isclose(I[2, 2] - I[0, 0],
                      -p.mass * p.radius ** 2 * PW.shell_c2m(t20, p), rtol=1e-10)


def test_overdamped_step_is_eq_5_39():
    """tan psi grows as exp(n^2 (I_b - I_a) t / C_f) for a frozen tensor, and one
    big exact step equals many small ones."""
    dI, Cf = 1e26, 1e32
    I = np.diag([0.0, dI, 0.0])                             # target y, pole near z
    psi0 = 1e-6
    s0 = np.array([0.0, np.sin(psi0), np.cos(psi0)])
    t = 3.0e13
    s1, info = PW.overdamped_step(s0, I, Cf, t)
    s2 = s0.copy()
    for _ in range(1000):
        s2, _ = PW.overdamped_step(s2, I, Cf, t / 1000)
    psi_expect = np.arctan(np.tan(psi0) * np.exp(C.MEAN_MOTION ** 2 * dI / Cf * t))
    assert np.isclose(np.arctan2(s1[1], s1[2]), psi_expect, rtol=1e-10)
    assert np.allclose(s1, s2, atol=1e-12)
    # initial rate: (n^2/2C_f) dI sin 2psi0
    assert np.isclose(info['psi_dot'], C.MEAN_MOTION ** 2 / (2 * Cf) * dI * np.sin(2 * psi0),
                      rtol=1e-6)
    # a pole exactly on the axis does not move (the balanced pendulum)
    s3, _ = PW.overdamped_step(np.array([0.0, 0.0, 1.0]), I, Cf, t)
    assert np.allclose(s3, [0, 0, 1])


def test_cf_has_ln10_and_tau_f_uses_growth_tau():
    p = _p()
    Tf = 124.0
    Cf = PW.dissipation_cf(p, Tf)
    no_ln10 = Cf / np.log(10.0)
    eta = 3e14 * np.exp(15 * (273 / Tf - 1))
    expect = (4 * np.pi * p.radius ** 2 * 20e3 * (1 / 15) * (273 / 173) * (Tf / 273) ** 2
              * eta * PW.gamma_tidal(p) ** 2)
    assert np.isclose(no_ln10, expect, rtol=1e-12)
    tf = PW.tau_f(p, Tf)
    assert 1e6 < tf < 3e6, tf              # OSPW Fig. 6: ~2 Myr near 124 K
    assert np.isclose((tf * PW.YEAR_S) ** 2,
                      Cf * 1e12 * PW.YEAR_S / (C.MEAN_MOTION ** 2 * PW.moment_I0(p)),
                      rtol=1e-12)


def test_ospw_evolution_numerical_matches_closed_form():
    import ospw_polar_wander as opw
    with C.parameter_set('paper'):
        cfg = opw.OSPWConfig()
        evo = opw.OSPWEvolution(cfg)
        tf = PW.tau_f(evo.p)
        out = evo.run(3 * tf, psi0_deg=np.degrees(1e-3), n_points=200)
        ana = np.degrees(evo.analytical_solution(out['t_yr'], 1e-3))
    assert np.max(np.abs(out['psi_deg'] - ana)) < 1e-4


# --------------------------------------------------------------------------- coupled sim
def _small_coupled(**kw):
    import coupled_shell_simulation as css
    cfg = css.CoupledConfig(use_ptm3d_surrogate=False, n_theta=24, n_lam=48,
                            n_snapshots=2, n_history=50, n_coarse_steps=200, **kw)
    return css, cfg


def test_coupled_defaults():
    import coupled_shell_simulation as css
    import polar_wander_cycle as pwc
    assert css.CoupledConfig().A_dark == 0.3814
    assert pwc.PolarWanderConfig().A_dark == 0.3814
    assert not hasattr(pwc.PolarWanderConfig(), 'fossil_bulge_factor')


def test_tau_dark_up_to_1e7():
    """Darkening follows 1 - exp(-t/tau_dark) for tau_dark = 1e4 .. 1e7 yr and
    the fine step never exceeds the coarse step."""
    for tau in (1e4, 1e5, 1e6, 1e7):
        css, cfg = _small_coupled(tau_dark=tau, t_max_years=2e6)
        evo = css.CoupledEvolution(cfg)
        evo.run()
        assert np.isclose(evo.f_dark, 1 - np.exp(-evo.t / tau), rtol=1e-9, atol=1e-12), tau
        assert np.isfinite(evo.b).all()


def test_psi0_seed_fires_once_on_instability():
    css, cfg = _small_coupled(t_max_years=1e3, psi0_rad=1e-3)
    evo = css.CoupledEvolution(cfg)
    evo.spin_pole = np.array([0.0, 0.0, 1.0])
    I0 = evo._ospw_I0
    stable = np.diag([-2e-6, -1e-6, 3e-6]) * I0
    unstable = np.diag([-2e-6, 3e-6, -1e-6]) * I0
    evo._apply_psi0_seed(stable)
    assert np.allclose(evo.spin_pole, [0, 0, 1]) and evo._seed_armed
    evo._apply_psi0_seed(unstable)
    assert np.isclose(np.arcsin(abs(evo.spin_pole[1])), 1e-3, rtol=1e-9)
    assert len(evo.seed_events) == 1
    before = evo.spin_pole.copy()
    evo._apply_psi0_seed(unstable)                         # disarmed: no second kick
    assert np.allclose(evo.spin_pole, before) and len(evo.seed_events) == 1
    # round-off-level gaps never fire
    evo2 = css.CoupledEvolution(cfg)
    evo2.spin_pole = np.array([0.0, 0.0, 1.0])
    evo2._apply_psi0_seed(np.diag([0.0, 1e-15, 0.0]) * I0)
    assert np.allclose(evo2.spin_pole, [0, 0, 1]) and not evo2.seed_events


def test_tidal_heating_follows_small_pole_tilts():
    """D must be re-rotated for any pole motion; a 1-degree update threshold
    left it body-fixed for small tilts and drove a spurious pole oscillation."""
    import polar_wander_cycle as pwc
    css, cfg = _small_coupled(t_max_years=1e3)
    evo = css.CoupledEvolution(cfg)
    D0 = evo.D.copy()
    tilt = np.radians(0.1)
    evo.spin_pole = np.array([0.0, np.sin(tilt), np.cos(tilt)])
    evo._update_dissipation()
    D_direct, _ = pwc._build_dissipation_for_spin_pole(
        cfg.rheology_case, evo.theta, evo.lam, evo.spin_pole)
    assert np.allclose(evo.D, D_direct, rtol=1e-12, atol=0)
    assert np.max(np.abs(evo.D / D0 - 1)) > 1e-4


def test_flow_law_D_table_matches_direct_evaluation():
    import polar_wander_cycle as pwc
    with C.parameter_set('paper'):
        theta, lam = G.periodic_grid(24, 48)
        pole = np.array([0.0, np.sin(0.3), np.cos(0.3)])
        D_tab, _ = pwc._build_dissipation_for_spin_pole('flow_ng3', theta, lam, pole)
        th_t, lm_t = pwc._tidal_coordinates(theta, lam, pole)
        rheology, _ = table1_case('flow_ng3')
        D_dir = dissipation_field(rheology, th_t, lm_t)
    assert np.max(np.abs(D_tab / D_dir - 1)) < 5e-3


def test_tau_diff_uses_ice_heat_capacity():
    """cp(T) = 90 + 7.49 T (heat1d planets.py Europa cpCoeff) in kappa = k/(rho cp)."""
    assert np.isclose(C.ice_heat_capacity(100.0), 839.0)
    css, cfg = _small_coupled(t_max_years=1e3)
    assert cfg.cp_mode == 'ice'
    evo = css.CoupledEvolution(cfg)
    T = np.full_like(evo.b, 100.0)
    b = np.full_like(evo.b, 20e3)
    k = C.A1_CONDUCTIVITY / 100.0 + C.A0_CONDUCTIVITY
    expect = b ** 2 / (k / (C.ICE_DENSITY * 839.0)) / (365.25 * 86400.0)
    assert np.allclose(evo._tau_diff_years(b, T), expect, rtol=1e-12)


def test_static_albedo_map_is_body_fixed(tmp_path=None):
    """static_map: T = T_an(A) + dT(tidal) + R(body); A and R stay with the body."""
    import tempfile
    css, cfg = _small_coupled(t_max_years=1e3)
    theta, lam = G.periodic_grid(cfg.n_theta, cfg.n_lam)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A = 0.55 + 0.2 * np.sin(TH) * np.sin(LA)
    R = 0.5 * np.cos(TH) * np.cos(LA)
    th_t, lm_t = np.linspace(0, np.pi, 31), np.linspace(-np.pi, np.pi, 61)
    dT = np.full((31, 61), 1.5)
    path = os.path.join(tempfile.mkdtemp(), 'forcing.npz')
    np.savez(path, theta=theta, lam=lam, A_meas=A, A_unif=0.55, R_meas=R,
             R_unif=np.zeros_like(R), dT_theta=th_t, dT_lam=lm_t, dT=dT)
    css2, cfg = _small_coupled(t_max_years=1e3, albedo_mode='static_map',
                               forcing_file=path, init_b_mode='equilibrium')
    cfg.use_ptm3d_surrogate = True
    evo = css2.CoupledEvolution(cfg)
    z = np.array([0.0, 0.0, 1.0])
    T = evo.surface_temperature_coupled(evo.model.albedo_field(0.0, z), z)
    T_an = ((1 - A) * OSSurfaceTemperature().annual_average_flux(TH) / C.STEFAN_BOLTZMANN) ** 0.25
    assert np.allclose(T, T_an + R + 1.5, rtol=1e-12)
    tilted = np.array([0.0, np.sin(0.3), np.cos(0.3)])
    assert np.array_equal(evo.model.albedo_field(0.0, tilted), A)
    assert np.allclose(evo.b, evo._b_eq(T), rtol=1e-12)            # equilibrium start
    # a forcing file built on another grid is refused
    cfg.n_theta = 26
    try:
        css2.CoupledEvolution(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('grid mismatch accepted')


def test_psi0_default_is_obliquity():
    import polar_wander_cycle as pwc
    assert np.isclose(pwc.PolarWanderConfig().psi0_rad, np.radians(0.1))


def test_coupled_initial_state_is_isotropic():
    css, cfg = _small_coupled(t_max_years=1e3)
    evo = css.CoupledEvolution(cfg)
    assert np.abs(evo.total_inertia_tensor()).max() / evo._ospw_I0 < 1e-15



# --------------------------------------------------------------------------- 2026-10-05 dynamics fixes
def _diag_I(ixx, iyy, izz):
    return np.diag([ixx, iyy, izz])


def test_tidal_torque_stiffness_is_4n2eps_toward_jupiter():
    """OSPW Eqs. 5.26/5.38: a pole tilted toward Jupiter (x) is pulled back with
    stiffness 4 n^2 (C - A); toward the trailing apex (y) with n^2 (C - B) only.
    The rotational-only overdamped_step (pre-2026-10-05) gives 1x toward Jupiter."""
    Cf, I0 = 1e32, 1e30
    I = _diag_I(-2e-6, -1e-6, 3e-6) * I0             # z max, x min: stable pole
    k = C.MEAN_MOTION ** 2 / Cf
    psi = 1e-3
    for axis, dI, factor in ((0, (3e-6 + 2e-6) * I0, 4.0), (1, (3e-6 + 1e-6) * I0, 1.0)):
        s0 = np.array([0.0, 0.0, np.cos(psi)])
        s0[axis] = np.sin(psi)
        _, info = PW.overdamped_step_tidal(s0, I, Cf, 1.0, PW.projected_meridian)
        expect = factor * k / 2.0 * dI * np.sin(2 * psi)
        assert np.isclose(info['psi_dot'], expect, rtol=1e-6), (axis, info['psi_dot'], expect)
        _, old = PW.overdamped_step(s0, I, Cf, 1.0)
        assert np.isclose(old['psi_dot'], k / 2.0 * dI * np.sin(2 * psi), rtol=1e-6)
    # integrated: tilt toward Jupiter decays 4x faster in the exponent
    t = 0.2 / (k * 5e-6 * I0)
    s0 = np.array([np.sin(psi), 0.0, np.cos(psi)])
    s1, info = PW.overdamped_step_tidal(s0, I, Cf, t, PW.projected_meridian)
    psi1 = np.arctan2(s1[0], s1[2])
    assert np.isclose(psi1, np.arctan(np.tan(psi) * np.exp(-4 * k * 5e-6 * I0 * t)), rtol=1e-6)
    assert info['n_substeps'] >= 8
    # the substep budget (max_growth 0.1) is converged: 10x finer agrees to 1e-9
    s_fine, _ = PW.overdamped_step_tidal(s0, I, Cf, t, PW.projected_meridian, max_growth=0.01)
    assert np.allclose(s1, s_fine, atol=1e-9)
    big = np.array([np.sin(0.6), 0.3, np.cos(0.6)]); big /= np.linalg.norm(big)
    Iu = _diag_I(-2e-6, 4e-6, -2e-6) * I0                 # unstable: large excursion
    a, _ = PW.overdamped_step_tidal(big, Iu, Cf, 2 * t, PW.projected_meridian)
    b, _ = PW.overdamped_step_tidal(big, Iu, Cf, 2 * t, PW.projected_meridian, max_growth=0.01)
    assert np.allclose(a, b, atol=1e-7)


def test_coupled_ospw_step_uses_tidal_torque():
    """The coupled model's pole step carries the tidal torque (rate toward Jupiter
    4x the rotational-only rate); tidal_torque=False restores the old step."""
    rates = {}
    for tidal in (True, False):
        css, cfg = _small_coupled(t_max_years=1e3, tidal_torque=tidal)
        evo = css.CoupledEvolution(cfg)
        I = _diag_I(-2e-6, -1e-6, 3e-6) * evo._ospw_I0
        evo.spin_pole = np.array([np.sin(1e-3), 0.0, np.cos(1e-3)])
        evo._ospw_step(1.0, I)
        rates[tidal] = evo._last_omega_deg_per_Myr
    assert np.isclose(rates[True] / rates[False], 4.0, rtol=1e-6)


def test_tidal_torque_leaves_trailing_instability_unchanged():
    """The unstable direction relevant to the exogenic mechanism (pole toward the
    trailing apex, I_yy > I_zz) has no tidal term: same growth as Eq. 5.39."""
    dI, Cf = 1e26, 1e32
    I = np.diag([0.0, dI, 0.0])
    psi0 = 1e-6
    s0 = np.array([0.0, np.sin(psi0), np.cos(psi0)])
    t = 3.0e13
    s1, _ = PW.overdamped_step_tidal(s0, I, Cf, t, PW.projected_meridian)
    s2, _ = PW.overdamped_step(s0, I, Cf, t)
    assert np.allclose(s1, s2, atol=1e-9)


def test_meridian_min_moment():
    """Quasi-static libration: the sub-Jovian axis is the in-plane minimum-moment
    axis, sign kept by continuity, projection returned when degenerate."""
    th = np.radians(25.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    I = R @ np.diag([-2.0, 1.0, 1.0]) @ R.T
    e = PW.meridian_min_moment(I, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    assert np.allclose(e, [np.cos(th), np.sin(th), 0], atol=1e-12)
    e2 = PW.meridian_min_moment(I, np.array([0, 0, 1.0]), np.array([-1.0, 0, 0]))
    assert np.allclose(e2, -e, atol=1e-12)
    iso = np.diag([1.0, 1.0, -2.0])
    e3 = PW.meridian_min_moment(iso, np.array([0, 0, 1.0]), np.array([1.0, 1.0, 0]))
    assert np.allclose(e3, [2 ** -0.5, 2 ** -0.5, 0])


def test_min_moment_meridian_moves_tidal_heating():
    """meridian_mode='min_moment': D follows the rotated meridian."""
    import polar_wander_cycle as pwc
    css, cfg = _small_coupled(t_max_years=1e3, meridian_mode='min_moment')
    evo = css.CoupledEvolution(cfg)
    D0 = evo.D.copy()
    th = np.radians(10.0)
    evo.sub_jovian = np.array([np.cos(th), np.sin(th), 0.0])
    evo._update_dissipation()
    D_dir, _ = pwc._build_dissipation_for_spin_pole(cfg.rheology_case, evo.theta, evo.lam,
                                                    evo.spin_pole, evo.sub_jovian)
    assert np.allclose(evo.D, D_dir, rtol=1e-12, atol=0)
    assert np.max(np.abs(evo.D / D0 - 1)) > 1e-3
    # the projection path never passes a meridian (legacy bytes)
    css, cfg = _small_coupled(t_max_years=1e3)
    assert css.CoupledEvolution(cfg)._meridian_arg() is None


def _saddle_evo(mode):
    css, cfg = _small_coupled(t_max_years=1e3, psi0_rad=1e-3, seed_rearm=mode)
    evo = css.CoupledEvolution(cfg)
    evo.spin_pole = np.array([0.0, 0.0, 1.0])
    # state after an earlier kick far away: disarmed, last kick 60 deg off
    evo._seed_armed = False
    evo._last_seed_pole = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
    return evo


def test_psi0_reseeds_a_pole_parked_on_a_saddle():
    """A pole that reached the MID axis (a saddle) after an earlier kick must be
    kicked again ('any_axis'); the pre-2026-10-05 rule re-armed only on the max
    axis and left it parked (Fermi: flow n5.2 'settled' at 90 deg)."""
    saddle = np.diag([-3e-6, 2e-6, 1e-6])               # z is the MID axis
    evo = _saddle_evo('max_axis')
    evo._apply_psi0_seed(saddle * evo._ospw_I0)
    assert np.allclose(evo.spin_pole, [0, 0, 1]) and not evo.seed_events
    evo = _saddle_evo('any_axis')
    evo._apply_psi0_seed(saddle * evo._ospw_I0)
    assert len(evo.seed_events) == 1
    assert np.isclose(np.arcsin(abs(evo.spin_pole[1])), 1e-3, rtol=1e-9)
    # a shell that re-centers on the kicked pole cannot get a kick per step:
    # rotate the tensor so its mid axis follows the pole -> no second kick
    s = evo.spin_pole
    ax = np.cross([0, 0, 1.0], s); ang = np.arcsin(np.linalg.norm(ax)); ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    evo._apply_psi0_seed(R @ saddle @ R.T * evo._ospw_I0)
    assert len(evo.seed_events) == 1


def test_settled_requires_max_axis_alignment():
    """A quiet pole parked on a saddle is 'parked', not 'settled'; misalignment is
    measured against the MAX axis, not the nearest one."""
    from ojakangas_stevenson import tpw_analysis as TA
    t = np.linspace(0, 3e7, 301)
    psi = np.radians(np.clip((t - 1e6) / 1e7, 0, 1) * 90.0)
    spin = np.stack([np.zeros_like(t), np.sin(psi), np.cos(psi)], 1)
    vmax_far = np.tile([1.0, 0.0, 0.0], (len(t), 1))       # max axis 90 deg away
    r = TA.summarize(t, spin, vmax_far)
    assert r['final_state'] == 'parked' and r['settle_Myr'] is None
    assert np.isclose(r['misalign_end_deg'], 90.0)
    r = TA.summarize(t, spin, spin)                          # converged on the max axis
    assert r['final_state'] == 'settled' and r['history'] == 'single'
    assert 10.0 < r['settle_Myr'] < 11.2 and 1.0 < r['onset_Myr'] < 2.0
    # two separated swings = episodic
    psi2 = np.radians(np.clip((t - 1e6) / 2e6, 0, 1) * 30 + np.clip((t - 1.5e7) / 2e6, 0, 1) * 30)
    spin2 = np.stack([np.zeros_like(t), np.sin(psi2), np.cos(psi2)], 1)
    r = TA.summarize(t, spin2, spin2)
    assert r['n_episodes'] == 2 and r['history'] == 'episodic'


def test_stefan_relaxation_time():
    """tau = rho L b / q = rho L b^2 / F, ~13 Myr for 30 km (modern k = 651/T);
    the coupled model uses it by default and 'diffusive' keeps b^2/kappa."""
    with C.parameter_set('modern'):
        tau = PW.stefan_relaxation_time(30e3, 100.0, sensible='none') / PW.YEAR_S
        F = C.A1_CONDUCTIVITY * np.log(273.0 / 100.0) + C.A0_CONDUCTIVITY * 173.0
        assert np.isclose(tau * PW.YEAR_S, 920.0 * 3.3355e5 * 30e3 ** 2 / F, rtol=1e-12)
        assert 12e6 < tau < 15e6, tau
    css, cfg = _small_coupled(t_max_years=1e3, relaxation_mode='stefan')
    evo = css.CoupledEvolution(cfg)
    T = np.full_like(evo.b, 100.0)
    b = np.full_like(evo.b, 20e3)
    st = evo._tau_relax_years(b, T)
    assert np.allclose(st, PW.stefan_relaxation_time(b, T, sensible='none') / PW.YEAR_S, rtol=1e-12)
    assert np.all(st > 3.0 * evo._tau_diff_years(b, T))      # slower than b^2/kappa
    evo.cfg.relaxation_mode = 'diffusive'
    assert np.array_equal(evo._tau_relax_years(b, T), evo._tau_diff_years(b, T))


# --------------------------------------------------------------------------- split albedo
def _split_forcing(theta, lam, A):
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'forcing.npz')
    np.savez(path, theta=theta, lam=lam, A_meas=A, A_unif=0.55, R_meas=np.zeros_like(A),
             R_unif=np.zeros_like(A), dT_theta=np.linspace(0, np.pi, 31),
             dT_lam=np.linspace(-np.pi, np.pi, 61), dT=np.zeros((31, 61)))
    return path


def test_exogenic_template_fit_and_split():
    """bh2021 = c1 cos(gamma) about the trailing apex; the fit recovers it, the
    template is N-S symmetric with zero mean, and A_exo + A_intr == A."""
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(46, 90)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    body = 0.03 * np.cos(TH) * np.cos(LA)                   # N-S antisymmetric, body-fixed
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA) + body
    fit = EA.fit_template(A, theta, lam, 'bh2021')
    assert np.isclose(fit['coeffs'][0], -0.12, atol=1e-10)
    assert np.isclose(fit['const'], 0.55, atol=1e-10)
    A_exo, A_intr = EA.split_map(A, theta, lam, fit)
    assert np.allclose(A_exo + A_intr, A, atol=1e-15)
    assert np.allclose(A_exo, A_exo[::-1, :], atol=1e-14)    # N-S symmetric
    assert abs(G.area_mean(A_exo, theta, lam)) < 1e-12
    for form in ('apex_l2', 'bh2021_zonal'):
        f = EA.fit_template(A, theta, lam, form)
        assert abs(G.area_mean(EA.template_field(f, TH, LA), theta, lam)) < 1e-12


def test_split_albedo_follows_tidal_frame():
    """tau_exo = 0: A_exo is the template at TIDAL coordinates of the current pole;
    tau_exo = inf: body-fixed; finite: exact exponential relaxation; and the
    total is clamped to [0.3814, 0.80]."""
    import polar_wander_cycle as pwc
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA) + 0.05 * np.cos(TH)
    path = _split_forcing(theta, lam, A)
    tilt = np.array([0.0, np.sin(0.4), np.cos(0.4)])
    out = {}
    for tau in (0.0, 1e6, np.inf):
        css, cfg = _small_coupled(t_max_years=1e3, albedo_mode='split', forcing_file=path,
                                  split_component='exogenic', tau_exo_yr=tau,
                                  init_b_mode='equilibrium')
        cfg.use_ptm3d_surrogate = True
        evo = css.CoupledEvolution(cfg)
        A0 = evo.model.A_exo.copy()
        assert np.allclose(evo.model.A_intr, 0.55)
        evo.model.advance(5e5, tilt)
        out[tau] = (A0, evo.model.A_exo.copy(), evo)
    th_t, lm_t = pwc._tidal_coordinates(theta, lam, tilt)
    eq = EA.template_field(out[0.0][2].exo_fit, th_t, lm_t)
    assert np.allclose(out[0.0][1], eq, atol=1e-15)
    assert np.array_equal(out[np.inf][1], out[np.inf][0])
    A0 = out[1e6][0]
    assert np.allclose(out[1e6][1], eq + (A0 - eq) * np.exp(-0.5), atol=1e-15)
    evo = out[0.0][2]
    evo.model.A_exo = np.full_like(evo.model.A_exo, 0.5)        # total 1.05
    assert np.allclose(evo.model.albedo_field(0.0), 0.80)
    evo.model.A_exo = np.full_like(evo.model.A_exo, -0.4)       # total 0.15
    assert np.allclose(evo.model.albedo_field(0.0), 0.3814)


def test_split_components_and_initial_conditions():
    """full = A at t=0 (clamped); intrinsic = A - template; blend / perturbation /
    pole offset initial conditions do what they say."""
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA) + 0.05 * np.cos(TH)
    path = _split_forcing(theta, lam, A)
    kw = dict(t_max_years=1e3, albedo_mode='split', forcing_file=path, tau_exo_yr=0.0)
    css, cfg = _small_coupled(split_component='full', init_b_mode='equilibrium', **kw)
    cfg.use_ptm3d_surrogate = True
    full = css.CoupledEvolution(cfg)
    assert np.allclose(full.model.albedo_field(0.0), np.clip(A, 0.3814, 0.80), atol=1e-14)
    css, cfg = _small_coupled(split_component='intrinsic', init_b_mode='equilibrium', **kw)
    cfg.use_ptm3d_surrogate = True
    intr = css.CoupledEvolution(cfg)
    exo0, _ = EA.split_map(A, theta, lam, intr.exo_fit)
    assert np.allclose(intr.model.albedo_field(0.0), np.clip(A - exo0, 0.3814, 0.80), atol=1e-14)
    shells = {}
    for mode in ('uniform_equilibrium', 'blend', 'equilibrium'):
        css, cfg = _small_coupled(split_component='exogenic', init_b_mode=mode, **kw)
        cfg.use_ptm3d_surrogate = True
        shells[mode] = css.CoupledEvolution(cfg)
    bu, bb, be = (shells[m].b for m in ('uniform_equilibrium', 'blend', 'equilibrium'))
    assert np.allclose(bb, 0.5 * (bu + be), rtol=1e-12)
    assert shells['blend']._ospw_p.shell_thickness == shells['equilibrium']._ospw_p.shell_thickness
    css, cfg = _small_coupled(split_component='exogenic', init_b_mode='equilibrium',
                              init_l2_perturb_km=(0.2, 0.2), init_perturb_seed=3,
                              init_pole_offset_deg=3.0, init_pole_azimuth_deg=90.0, **kw)
    cfg.use_ptm3d_surrogate = True
    pert = css.CoupledEvolution(cfg)
    db = pert.b - be
    rms = np.sqrt(G.area_mean(db ** 2, theta, lam))
    assert np.isclose(rms, 200.0, rtol=0.02), rms
    assert abs(G.area_mean(db, theta, lam)) < 1e-6 * 200          # pure degree 2
    assert np.allclose(pert.spin_pole, [0, np.sin(np.radians(3)), np.cos(np.radians(3))])


def test_tidal_frame_t20_2t22_matches_harmonics():
    css, cfg = _small_coupled(t_max_years=1e3)
    evo = css.CoupledEvolution(cfg)
    theta, lam = evo.theta, evo.lam
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    b = 20e3 + 300.0 * (1.5 * np.cos(TH) ** 2 - 0.5) + 80.0 * 3 * np.sin(TH) ** 2 * np.cos(2 * LA)
    p = evo._ospw_p
    I = PW.shell_inertia_tensor(b, theta, lam, p)
    h = G.harmonics(b, theta, lam)
    val = PW.tidal_frame_t20_2t22(I, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), p)
    assert np.isclose(val, h['t20_2t22'], rtol=1e-9), (val, h['t20_2t22'])



# --------------------------------------------------------------------------- review fixes (2026-10-05)
def test_uniform_equilibrium_ic_uses_uniform_residual():
    """static_map 'measured' + uniform_equilibrium: the uniform-albedo shell must
    use R_unif, not the measured-map residual R_meas."""
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA)
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'forcing.npz')
    R_meas = 3.0 * np.cos(TH)                                # large, N-S antisymmetric
    np.savez(path, theta=theta, lam=lam, A_meas=A, A_unif=0.55, R_meas=R_meas,
             R_unif=np.zeros_like(A), dT_theta=np.linspace(0, np.pi, 31),
             dT_lam=np.linspace(-np.pi, np.pi, 61), dT=np.zeros((31, 61)))
    css, cfg = _small_coupled(t_max_years=1e3, albedo_mode='static_map', static_field='measured',
                              forcing_file=path, init_b_mode='uniform_equilibrium')
    cfg.use_ptm3d_surrogate = True
    evo = css.CoupledEvolution(cfg)
    T_u = ((1 - 0.55) * OSSurfaceTemperature().annual_average_flux(TH) / C.STEFAN_BOLTZMANN) ** 0.25
    assert np.allclose(evo.b, evo._b_eq(T_u), rtol=1e-12)
    assert evo._R is not None and np.array_equal(evo._R, R_meas)   # the run keeps R_meas


def test_min_moment_meridian_jump_is_reported():
    css, cfg = _small_coupled(t_max_years=1e3, meridian_mode='min_moment')
    evo = css.CoupledEvolution(cfg)
    I0 = evo._ospw_I0
    evo._update_meridian(np.diag([-1e-6, 0.0, 1e-6]) * I0)       # x is the in-plane min
    assert not evo.meridian_jumps and np.allclose(evo.sub_jovian, [1, 0, 0])
    evo._update_meridian(np.diag([0.0, -1e-6, 1e-6]) * I0)       # now y: a 90 deg jump
    assert len(evo.meridian_jumps) == 1 and abs(evo.meridian_jumps[0][1] - 90.0) < 1e-6


def test_min_moment_requires_ospw():
    for model in ('exponential', 'coupled'):
        css, cfg = _small_coupled(t_max_years=1e3, meridian_mode='min_moment', tpw_model=model)
        try:
            css.CoupledEvolution(cfg)
        except ValueError:
            continue
        raise AssertionError(f'min_moment accepted with tpw_model={model}')


def test_continuous_only_if_episode_runs_to_the_end():
    from ojakangas_stevenson import tpw_analysis as TA
    t = np.linspace(0, 3e7, 301)
    psi = np.radians(np.clip((t - 1e6) / 9e6, 0, 1) * 30.0)         # 30 deg by 10 Myr
    psi = psi + np.radians(np.clip((t - 2.7e7) / 3e6, 0, 1) * 4.0)  # 4 deg drift, 1.3 deg/Myr
    spin = np.stack([np.zeros_like(t), np.sin(psi), np.cos(psi)], 1)
    r = TA.summarize(t, spin, spin)
    assert r['n_episodes'] == 1 and r['history'] == 'single' and r['final_state'] == 'moving', r
    psi2 = np.radians(np.clip((t - 1e6) / 2.9e7, 0, 1) * 60.0)     # one episode to the end
    spin2 = np.stack([np.zeros_like(t), np.sin(psi2), np.cos(psi2)], 1)
    assert TA.summarize(t, spin2, spin2)['history'] == 'continuous'
    try:
        TA.summarize(t[:1], spin[:1], spin[:1])
    except ValueError:
        pass
    else:
        raise AssertionError('one-record track accepted')


def test_pole0_is_the_offset_pole():
    css, cfg = _small_coupled(t_max_years=1e3, init_pole_offset_deg=2.0, init_pole_azimuth_deg=30.0)
    evo = css.CoupledEvolution(cfg)
    assert np.array_equal(evo._pole0, evo.spin_pole) and evo.spin_pole[2] < 1.0



# --------------------------------------------------------------------------- 2026-10-05 rulings: S/L, convection proxy, exosphere template
def test_sensible_heat_ratio_exact():
    """S/L for the steady conductive profile: Fermi's 0.53/0.50/0.48 at 90/100/110 K
    (modern k = 651/T, c_p = 90 + 7.49 T), and the closed form for k = a1/T."""
    with C.parameter_set('modern'):
        r = PW.sensible_heat_ratio(np.array([90.0, 100.0, 110.0]))
        assert np.allclose(r, [0.530, 0.503, 0.477], atol=1.5e-3), r
        Ts, Tm, L = 100.0, 273.0, C.ICE_LATENT_HEAT
        lnr = np.log(Tm / Ts)
        h = lambda T: 90.0 * T + 3.745 * T * T                      # noqa: E731
        closed = (h(Tm) - 90.0 * (Tm - Ts) / lnr - 3.745 * (Tm ** 2 - Ts ** 2) / (2 * lnr)) / L
        assert np.isclose(PW.sensible_heat_ratio(Ts), closed, rtol=1e-10)
        tau = PW.stefan_relaxation_time(30e3, 100.0, sensible='exact')
        assert np.isclose(tau / PW.stefan_relaxation_time(30e3, 100.0, sensible='none'),
                          1.0 + closed, rtol=1e-8)
        # the 0.02 K table reproduces the quadrature (Fermi F10)
        T = np.linspace(60.0, 200.0, 777)
        assert np.max(np.abs(PW.sensible_heat_ratio_tabulated(T) - PW.sensible_heat_ratio(T))) < 1e-9
    css, cfg = _small_coupled(t_max_years=1e3, relaxation_mode='stefan_sensible')
    evo = css.CoupledEvolution(cfg)
    T = np.linspace(90.0, 130.0, evo.b.size).reshape(evo.b.shape)  # per location
    b = np.full_like(evo.b, 25e3)
    assert np.allclose(evo._tau_relax_years(b, T),
                       PW.stefan_relaxation_time(b, T, sensible='exact') / PW.YEAR_S, rtol=1e-12)


def test_convection_proxy_knobs():
    """conv_s scales every degree >= 1 of the equilibrium anomaly by s (mean kept);
    conv_fD weights only the tidal (D) part; s = f_D = 1 is the plain O&S field."""
    css, cfg = _small_coupled(t_max_years=1e3)
    evo = css.CoupledEvolution(cfg)
    TH, LA = np.meshgrid(evo.theta, evo.lam, indexing='ij')
    T = 100.0 + 5.0 * np.sin(TH) * np.sin(LA) + 3.0 * np.cos(TH) ** 2
    b1 = evo._b_eq(T)
    import polar_wander_cycle as pwc
    # s = f_D = 1 takes the plain O&S path. rtol 1e-14, not array_equal: repeated
    # in-process evaluations of the same expression intermittently differ by
    # ~2 ULP on this machine (cause not identified; ledger PTM-709).
    assert np.allclose(b1, pwc._equilibrium_thickness(T, evo.D, evo.H, cfg.thickness_method,
                                                      evo.rheology), rtol=1e-14, atol=0)
    h1 = G.harmonics(b1, evo.theta, evo.lam)
    for s_ in (0.4, 0.03):
        evo.cfg.conv_s, evo.cfg.conv_fD = s_, 1.0
        bs = evo._b_eq(T)
        hs = G.harmonics(bs, evo.theta, evo.lam)
        assert np.isclose(G.area_mean(bs, evo.theta, evo.lam), G.area_mean(b1, evo.theta, evo.lam))
        for k in ('t20_2t22', 'b11s', 't22', 'b20'):
            assert np.isclose(hs[k], s_ * h1[k], rtol=1e-9, atol=1e-9), k
    # f_D = 0: no tidal contribution -> b depends on T only (D replaced by its mean)
    evo.cfg.conv_s, evo.cfg.conv_fD = 1.0, 0.0
    D_ref = np.full_like(evo.D, G.area_mean(evo.D, evo.theta, evo.lam))
    assert np.allclose(evo._b_eq(T), pwc._equilibrium_thickness(T, D_ref, evo.H,
                       cfg.thickness_method, evo.rheology), rtol=1e-12)
    evo.cfg.conv_fD = 0.4
    b_ref = pwc._equilibrium_thickness(T, D_ref, evo.H, cfg.thickness_method, evo.rheology)
    assert np.allclose(evo._b_eq(T), b_ref + 0.4 * (b1 - b_ref), rtol=1e-12)
    evo.cfg.conv_s = 0.0
    try:
        evo._b_eq(T)
    except ValueError:
        pass
    else:
        raise AssertionError('conv_s = 0 accepted')


def _exo_npz(theta, lam):
    import tempfile
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    S = -0.02 + 0.01 * (1.5 * np.cos(theta) ** 2 - 0.5)
    c = np.array([-0.02, -0.01, -0.002, 0.003])
    cg = np.sin(TH) * np.sin(LA)
    T2 = sum(cn * np.polynomial.legendre.legval(cg, [0.0] * n + [1.0]) for n, cn in enumerate(c, 1))
    path = os.path.join(tempfile.mkdtemp(), 'exo.npz')
    np.savez(path, theta=theta, lam=lam, high_1M_dA_spin_theta=S, high_1M_tidal_Pn_coeffs=c,
             high_1M_dA_spin_2d=np.broadcast_to(S[:, None], TH.shape), high_1M_dA_tidal_2d=T2,
             mergny_dA_spin_theta=0.05 * (1.5 * np.cos(theta) ** 2 - 0.5))
    return path, S, c, T2


def test_exo_profile_template_variants():
    """E1 at t=0 = S + T of the exosphere file (S zero-mean); E2 total cos(gamma)
    coefficient = the map's; E3 spin part = the map's zonal profile; split stays exact."""
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    path, S, c, T2 = _exo_npz(theta, lam)
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA) + 0.04 * (1.5 * np.cos(TH) ** 2 - 0.5)
    w = G.theta_weights(theta)
    S0 = S - np.sum(S * w) / np.sum(w)
    f1 = EA.exo_profile_fit(path, 'high_1M', 'E1', A, theta, lam)
    assert np.allclose(EA.template_field(f1, TH, LA), S0[:, None] + T2, atol=1e-14)
    f2 = EA.exo_profile_fit(path, 'high_1M', 'E2', A, theta, lam)
    assert np.isclose(f2['coeffs'][0], -0.12, atol=1e-10) and np.isclose(f2['c_impl'], -0.10, atol=1e-10)
    f3 = EA.exo_profile_fit(path, 'high_1M', 'E3', A, theta, lam)
    M = 0.05 * (1.5 * np.cos(theta) ** 2 - 0.5)
    assert np.allclose(f3['S'], M - np.sum(M * w) / np.sum(w))
    fs = EA.exo_profile_fit(path, 'high_1M', 'E1', A, theta, lam, scale_zonal=2.0, scale_odd=3.0)
    assert np.allclose(fs['coeffs'], [3 * c[0], 2 * c[1], 3 * c[2], 2 * c[3]])
    for f in (f1, f2, f3):
        exo, intr = EA.split_map(A, theta, lam, f)
        assert np.allclose(exo + intr, A, atol=1e-15)
        assert abs(G.area_mean(exo, theta, lam)) < 1e-3
    # wired into the coupled model, following the spin pole and the tidal frame
    path_f = _split_forcing(theta, lam, A)
    css, cfg = _small_coupled(t_max_years=1e3, albedo_mode='split', forcing_file=path_f,
                              split_component='exogenic', tau_exo_yr=0.0, init_b_mode='equilibrium',
                              exo_template='exo_file', exo_forcing_file=path, exo_case='high_1M',
                              exo_variant='E2')
    cfg.use_ptm3d_surrogate = True
    evo = css.CoupledEvolution(cfg)
    assert np.allclose(evo.model.A_exo, EA.template_field(f2, TH, LA))
    import polar_wander_cycle as pwc
    tilt = np.array([0.0, np.sin(0.4), np.cos(0.4)])
    evo.model.advance(1e3, tilt)
    th_t, lm_t = pwc._tidal_coordinates(theta, lam, tilt)
    assert np.allclose(evo.model.A_exo, EA.template_field(evo.exo_fit, th_t, lm_t))



# --------------------------------------------------------------------------- Fermi round-2 fixes (F1, F2, F4, F6, F10)
def test_stefan_eigen_factor():
    """Exact eigenmode factor: Fermi's shooting 1.422/1.396/1.372 at 90/100/110 K
    (modern), the constant-k analytic case St/x^2 (x tan x = St), the table,
    ~8% below 1 + S/L, and the coupled default."""
    with C.parameter_set('modern'):
        f = [PW.stefan_eigen_factor(T) for T in (90.0, 100.0, 110.0)]
        assert np.allclose(f, [1.422, 1.396, 1.372], atol=1.5e-3), f
        r = np.array(f) / (1.0 + PW.sensible_heat_ratio(np.array([90.0, 100.0, 110.0])))
        assert np.all((0.92 < r) & (r < 0.94)), r
        T = np.array([73.3, 100.0, 141.7])
        assert np.allclose(PW.stefan_eigen_factor_tabulated(T),
                           [PW.stefan_eigen_factor(x, n=300) for x in T], rtol=2e-4)
        assert np.isclose(PW.stefan_relaxation_time(30e3, 100.0)
                          / PW.stefan_relaxation_time(30e3, 100.0, sensible='none'),
                          PW.stefan_eigen_factor_tabulated(100.0), rtol=1e-12)
    # constant k, c_p: tau/tau_L = St/x^2, x tan x = St, St = c dT / L
    from scipy.optimize import brentq
    saved = (C.A1_CONDUCTIVITY, C.A0_CONDUCTIVITY, C.ICE_CP_COEFFS)
    try:
        C.A1_CONDUCTIVITY, C.A0_CONDUCTIVITY, C.ICE_CP_COEFFS = 0.0, 2.5, (1500.0, 0.0)
        St = 1500.0 * 173.0 / C.ICE_LATENT_HEAT
        x = brentq(lambda x: x * np.tan(x) - St, 1e-6, np.pi / 2 - 1e-9)
        assert np.isclose(PW.stefan_eigen_factor(100.0), St / x ** 2, rtol=1e-4)
    finally:
        C.A1_CONDUCTIVITY, C.A0_CONDUCTIVITY, C.ICE_CP_COEFFS = saved
    css, cfg = _small_coupled(t_max_years=1e3)
    assert cfg.relaxation_mode == 'stefan_eigen'
    evo = css.CoupledEvolution(cfg)
    T = np.linspace(90.0, 130.0, evo.b.size).reshape(evo.b.shape)
    b = np.full_like(evo.b, 25e3)
    assert np.allclose(evo._tau_relax_years(b, T), PW.stefan_relaxation_time(b, T) / PW.YEAR_S,
                       rtol=1e-12)
    # conv_flow_tau (default on): tau_eff = s tau_th
    evo.cfg.conv_s = 0.4
    assert np.allclose(evo._tau_relax_years(b, T),
                       0.4 * PW.stefan_relaxation_time(b, T) / PW.YEAR_S, rtol=1e-12)


def test_e3map_is_the_maps_axisymmetric_content():
    """E3map = map zonal profile (N-S symmetrized) + map c1..c4; no model input."""
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    path, S, c, T2 = _exo_npz(theta, lam)
    d = dict(np.load(path))
    M = 0.05 * (1.5 * np.cos(theta) ** 2 - 0.5) + 0.02 * np.cos(theta)    # N-S asymmetric
    d['mergny_dA_spin_theta'] = M
    d['mergny_tidal_Pn_coeffs'] = np.array([-0.124, -0.006, -0.015, 0.007])
    np.savez(path, **d)
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA)
    f = EA.exo_profile_fit(path, 'high_1M', 'E3map', A, theta, lam)
    assert np.allclose(f['coeffs'], [-0.124, -0.006, -0.015, 0.007]) and f['sym_S']
    w = G.theta_weights(theta)
    Ms = 0.05 * (1.5 * np.cos(theta) ** 2 - 0.5)
    assert np.allclose(f['S'], Ms - np.sum(Ms * w) / np.sum(w), atol=1e-15)
    assert np.allclose(EA.exo_profile_fit(path, 'high_1M', 'E3map', A, theta, lam, 2.0, 3.0)['coeffs'],
                       f['coeffs'])                                          # case/scale-independent
    g = EA.exo_profile_fit(path, 'high_1M', 'E3map', A, theta, lam, sym_S=False)
    assert not np.allclose(g['S'], f['S'])


def test_fD_warns():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        css, cfg = _small_coupled(t_max_years=1e3, conv_fD=0.4)
        css.CoupledEvolution(cfg)
    assert 'not a physical convection proxy' in buf.getvalue()



def test_e4_dose_term():
    """E4 = E1 + kappa u, u = -(D/<D> - 1) zero-mean, evaluated in TIDAL coordinates by a
    periodic bilinear interpolation (exact at the nodes, accurate off them), and wired
    into the coupled model."""
    import tempfile
    from ojakangas_stevenson import exogenic_albedo as EA
    theta, lam = G.periodic_grid(24, 48)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    path, S, c, T2 = _exo_npz(theta, lam)
    Dn = 1.0 + 0.5 * np.sin(TH) ** 4 * (1.0 + 0.6 * np.sin(TH) * np.sin(LA))  # lens-like
    dpath = os.path.join(tempfile.mkdtemp(), 'dose.npz')
    np.savez(dpath, theta=theta, lam=lam, dose_norm_EN_1mm=Dn)
    th_u, lm_u, u = EA.dose_unit_field(dpath, 'EN_1mm')
    assert abs(G.area_mean(u, theta, lam)) < 1e-14
    A = 0.55 - 0.12 * np.sin(TH) * np.sin(LA)
    e1 = EA.exo_profile_fit(path, 'high_1M', 'E4', A, theta, lam)
    e4 = EA.add_dose(e1, dpath, 'EN_1mm', 0.2)
    assert np.allclose(EA.template_field(e4, TH, LA), EA.template_field(e1, TH, LA) + 0.2 * u, atol=1e-14)
    # off the nodes: smooth analytic field of the tidal coordinates, 46 x 90 grid
    th2, lm2 = G.periodic_grid(46, 90)
    TH2, LA2 = np.meshgrid(th2, lm2, indexing='ij')
    Fx = lambda t, l: np.sin(t) ** 2 * np.cos(2 * l) + np.cos(t)                  # noqa: E731
    tq, lq = np.random.default_rng(1).uniform(0, np.pi, 500), np.random.default_rng(2).uniform(-4, 4, 500)
    # bilinear bound h^2 max|F''| / 8 with h = 4 deg and |F''| <= 4 (cos 2 lam): ~2.4e-3 per axis
    h = 2 * np.pi / len(lm2)
    assert np.max(np.abs(EA._periodic_interp(th2, lm2, Fx(TH2, LA2), tq, lq) - Fx(tq, lq))) < 2 * h * h * 4 / 8
    # wired: the coupled exogenic albedo at t = 0 is A_unif + E4
    path_f = _split_forcing(theta, lam, A)
    css, cfg = _small_coupled(t_max_years=1e3, albedo_mode='split', forcing_file=path_f,
                              split_component='exogenic', tau_exo_yr=0.0, init_b_mode='equilibrium',
                              exo_template='exo_file', exo_forcing_file=path, exo_case='high_1M',
                              exo_variant='E4', exo_dose_file=dpath, exo_dose_kappa=0.2)
    cfg.use_ptm3d_surrogate = True
    evo = css.CoupledEvolution(cfg)
    assert np.allclose(evo.model.A_exo, EA.template_field(e4, TH, LA), atol=1e-14)
    css, cfg = _small_coupled(t_max_years=1e3, albedo_mode='split', forcing_file=path_f,
                              split_component='exogenic', exo_template='exo_file', exo_forcing_file=path,
                              exo_variant='E4')
    cfg.use_ptm3d_surrogate = True
    try:
        css.CoupledEvolution(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('E4 without a dose file accepted')



# --------------------------------------------------------------------------- simple demo + NSR (2026-10-06)
def test_moment_differences_exact_chain():
    """moment_differences uses the Eq. 2.18 shell tensor; the O&S Eq. 5.9 rounding
    1e-7 (4 t22 / km) is 12-18x low against it in the whole-body normalization."""
    with C.parameter_set('modern'):
        rh, H = table1_case('maxwell_h0')
        m = IceShellModel(rh, H_flux=H)
        r = m.moment_differences(91, 180)
        theta, lam, b = m.thickness_grid(91, 180)
        bc = G.area_mean(b, theta, lam)
        p = PW.OSPWParams(shell_thickness=bc, lithosphere_thickness=bc / 10.0)
        I = PW.shell_inertia_tensor(b, theta, lam, p)
        Cb = C.C_MA2_EUROPA * p.mass * p.radius ** 2
        assert np.isclose(r['delta_BA_C'], (I[1, 1] - I[0, 0]) / Cb, rtol=1e-12)
        ratio = r['delta_BA_C'] / r['delta_BA_C_os_eq59']
        assert 12.0 < ratio < 19.0, ratio
        # B - A = 4 M a^2 C22, B - C = M a^2 (C20 + 2 C22): the ratio is 4 t22/(t20 + 2 t22)
        assert np.isclose(r['delta_BA'] / r['delta_BC'], r['4t22_km'] / r['t20_plus_2t22_km'], rtol=1e-6)
        assert np.isclose(PW.tidal_frame_B_minus_A(I, np.array([0, 0, 1.0]), np.array([1.0, 0, 0])),
                          r['delta_BA'], rtol=1e-12)
        Cs = PW.nsr_normalizations(p, bc)
        assert Cs['C_body'] > Cs['I_0'] > Cs['C_s']


def test_kappa_star_gp():
    """B&H 2023 Eq. 7: 5.80e-7 at e = 0.0094; 4.64e-4 at e = 0.05 (their ms p. 21)."""
    assert np.isclose(PW.kappa_star_gp(0.0094), 5.80e-7, rtol=2e-3)
    assert np.isclose(PW.kappa_star_gp(0.05), 4.64e-4, rtol=2e-3)


def test_demo_forcing():
    """Demo forcing: every term zero-mean; Bhat has unit band contrast; the cos(gamma)
    and Y22 terms are the code's b11s and b22c bases; it follows the tidal frame."""
    from ojakangas_stevenson import exogenic_albedo as EA
    import polar_wander_cycle as pwc
    theta, lam = G.periodic_grid(91, 180)
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    for shape, kw in (('cos2', {}), ('cos2', {'delta': 0.0}), ('gauss', {}), ('p2', {})):
        f = EA.demo_fit(theta, lam, a_b=-0.03, shape=shape, **kw)
        A = EA.template_field(f, TH, LA)
        assert abs(G.area_mean(A, theta, lam)) < 1e-14
        assert np.isclose(EA.band_contrast(A, theta, lam), -0.03, rtol=1e-12), shape
        assert np.allclose(A, A[::-1], atol=1e-15) and np.allclose(A, A[:, :1], atol=1e-15)
    f = EA.demo_fit(theta, lam, a1=-0.12, a22=0.006)
    h = G.harmonics(EA.template_field(f, TH, LA), theta, lam)
    assert np.isclose(h['b11s'], -0.12, rtol=1e-9) and np.isclose(h['b22c'], 0.006, rtol=1e-9)
    assert abs(h['b20']) < 1e-12
    fb = EA.demo_fit(theta, lam, a_b=1.0)
    hb = G.harmonics(EA.template_field(fb, TH, LA), theta, lam)
    assert -1.62 < hb['b20'] < -1.52, hb['b20']          # spec: b20 per unit beta = -1.567
    tilt = np.array([0.0, np.sin(0.3), np.cos(0.3)])
    th_t, lm_t = pwc._tidal_coordinates(theta, lam, tilt)
    A_t = EA.template_field(f, th_t, lm_t)
    assert np.isclose(np.max(np.abs(A_t)), np.max(np.abs(EA.template_field(f, TH, LA))), rtol=0.02)


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as exc:          # report every failure, then exit 1
                failures += 1
                print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    print(f'{failures} failure(s)')
    sys.exit(1 if failures else 0)
