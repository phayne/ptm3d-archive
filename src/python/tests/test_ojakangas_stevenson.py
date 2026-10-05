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


def test_coupled_initial_state_is_isotropic():
    css, cfg = _small_coupled(t_max_years=1e3)
    evo = css.CoupledEvolution(cfg)
    assert np.abs(evo.total_inertia_tensor()).max() / evo._ospw_I0 < 1e-15


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
