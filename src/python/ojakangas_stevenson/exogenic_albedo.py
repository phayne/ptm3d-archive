"""Exogenic (tidal-frame) albedo template for the split albedo boundary condition.

The Bond albedo of Europa is split into
    A(x, t) = A_intr(x_body) + A_exo(x, t),
    dA_exo/dt = [A_exo^eq(tidal-frame coordinates of x) - A_exo] / tau_exo,
where A_exo^eq is the pattern set by the magnetospheric plasma flow. That flow
is fixed to the orbit (it overtakes Europa from the trailing side), so the
pattern is fixed in the TIDAL frame, not to the surface, and it is
north-south symmetric by construction.

Functional form (Burnett & Hayne 2021, Icarus 364, 114438, Sec. 2 and Eq. 7,
after Nelson et al. 1986): the alteration varies "as the cosine of the angle
from the antapex of motion", i.e. along the equator as
r(lambda) = a0/2 + a1 cos(lambda) + b1 sin(lambda), with a1 ~ 0 for
synchronous rotation (their a1/b1 ~ 1e-3). On the sphere this is
    A_exo^eq = c1 cos(gamma),   cos(gamma) = sin(theta_t) sin(lambda_t),
gamma = angle from the trailing apex (lambda_t = +90 deg). One term gives
both the trailing-apex darkening and the leading-hemisphere brightening.
This is ``form='bh2021'`` (the default). Options, for sensitivity tests only:
  * 'apex_l2'     adds P2(cos gamma): a sharper (or flatter) apex bullseye
                  (Fermi's apex-axisymmetric degree <= 2 fit);
  * 'bh2021_zonal' adds P2(cos theta_t): a latitude (polar-brightening)
                  term that follows the spin pole. It is NOT exogenic in
                  origin; it is offered because, left out, it ends up in the
                  body-fixed A_intr.

Anchoring: the template is fitted to the measured (eq36, Mergny et al. 2025)
map at t = 0, when the tidal frame equals the body frame (spin = body z,
sub-Jovian = body x, longitude east-positive, trailing apex at +90 E), by
area-weighted least squares on [1, basis...]. The constant goes to A_intr;
every template basis function has zero area mean, so A_exo^eq carries no
global-mean albedo. A_intr = A_meas - A_exo^eq(t = 0) exactly.
"""
import numpy as np

from . import grid as G

TEMPLATE_FORMS = {
    'bh2021': ('cos_gamma',),
    'apex_l2': ('cos_gamma', 'p2_apex'),
    'bh2021_zonal': ('cos_gamma', 'p2_spin'),
}


def _p2(x):
    return 1.5 * x * x - 0.5


def basis(form, theta_t, lam_t):
    """Template basis functions (zero area mean) at tidal coordinates."""
    if form not in TEMPLATE_FORMS:
        raise ValueError(f'unknown template form {form!r}; '
                         f'choose from {sorted(TEMPLATE_FORMS)}')
    cos_g = np.sin(theta_t) * np.sin(lam_t)
    out = []
    for name in TEMPLATE_FORMS[form]:
        if name == 'cos_gamma':
            out.append(cos_g)
        elif name == 'p2_apex':
            out.append(_p2(cos_g))
        elif name == 'p2_spin':
            out.append(_p2(np.cos(theta_t)))
    return out


def fit_template(A, theta, lam, form='bh2021'):
    """Fit the template to a body-frame map A(theta, lam) at the initial pole.

    Returns dict: form, names, coeffs (template only), const (goes to
    A_intr), r2 (area-weighted variance explained), rms_residual.
    """
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    cols = [np.ones_like(TH)] + basis(form, TH, LA)
    w = G.area_weights(theta, lam).ravel()
    sw = np.sqrt(w)
    M = np.stack([c.ravel() for c in cols], axis=1)
    coef, *_ = np.linalg.lstsq(M * sw[:, None], A.ravel() * sw, rcond=None)
    res = A.ravel() - M @ coef
    mean = lambda f: np.sum(f * w) / np.sum(w)  # noqa: E731
    tot = mean((A.ravel() - mean(A.ravel())) ** 2)
    r = mean((res - mean(res)) ** 2)
    return {'form': form, 'names': list(TEMPLATE_FORMS[form]),
            'coeffs': [float(c) for c in coef[1:]], 'const': float(coef[0]),
            'r2': float(1.0 - r / tot), 'rms_residual': float(np.sqrt(r))}


def _legendre(n, x):
    return np.polynomial.legendre.legval(x, [0.0] * n + [1.0])


def exo_profile_fit(path, case, variant, A_meas, theta, lam,
                    scale_zonal=1.0, scale_odd=1.0, sym_S=None):
    """Exosphere-derived exogenic template (Halo, 2026-10-05), as a fit dict.

    ``path``: exo_albedo_forcing_91x180.npz. It decomposes the exosphere
    model's albedo change Delta A (uniform start, 8-phase mean) into
    S(theta) about the SPIN pole (``<case>_dA_spin_theta``) and
    T(gamma) = sum_n c_n P_n(cos gamma) in TIDAL coordinates
    (``<case>_tidal_Pn_coeffs``, gamma from the trailing apex). Its body-fixed
    residual is not exogenic and is left in A_intr.

    case: e.g. 'base_1M', 'high_1M'. scale_zonal multiplies S and the even
    c_n, scale_odd the odd c_n (e.g. a saturated-amplitude estimate).
    variant:
      'E1' exosphere only: S + T;
      'E2' E1 plus the B&H cos(gamma) implantation term, amplitude chosen so
           the total cos(gamma) coefficient equals the measured map's
           (the bh2021 fit of A_meas, -0.124 for eq36);
      'E3' (42318df; SUPERSEDED, double-counts degree 2, Fermi F1): E2 with
           the spin part replaced by the map's zonal profile but the MODEL's
           tidal-even c2..c4 kept (2.5-3.7x the map's c2);
      'E4' is E1 plus the dose-darkening term (``add_dose``; set up by the
           coupled model from exo_dose_* ; no B&H term: the dose term
           carries its own c1);
      'E3map' (E3-fair): the MAP's own axisymmetric degree <= 4 content as
           exogenic: the map's zonal profile S_map(theta) about the spin pole
           plus the map's c1..c4 (``mergny_tidal_Pn_coeffs``) in tidal
           coordinates. Case- and scale-independent (no model input). The
           map's S is N-S symmetrized (``sym_S``, default for E3map, Fermi F2:
           its south cap, -0.198 vs -0.040 north, sits in the gap-fill
           annulus). NOTE: ~65% of the map's zonal b20 is a DARK LOW-LATITUDE
           BAND (|lat| < 25 deg), not bright poles, and that band is not
           longitude-uniform; E3map assumes it follows the spin pole.
    S is made zero-mean (the global darkening changes <b>, not the driving).
    """
    d = np.load(path)
    if not (np.allclose(d['theta'], theta) and np.allclose(d['lam'], lam)):
        raise ValueError(f'{path} was built on a different grid')
    S = np.asarray(d[f'{case}_dA_spin_theta'], dtype=float) * scale_zonal
    c = np.asarray(d[f'{case}_tidal_Pn_coeffs'], dtype=float).copy()
    c[1::2] *= scale_zonal                      # c2, c4 (even)
    c[0::2] *= scale_odd                        # c1, c3 (odd)
    c_impl = 0.0
    if variant in ('E2', 'E3'):
        c_map = fit_template(A_meas, theta, lam, 'bh2021')['coeffs'][0]
        c_impl = c_map - c[0]
        c[0] = c_map
    elif variant == 'E4':
        pass                                    # E1 here; the dose term is added by add_dose
    elif variant == 'E3map':
        c = np.asarray(d['mergny_tidal_Pn_coeffs'], dtype=float).copy()
        c_impl = 0.0
    elif variant != 'E1':
        raise ValueError(f"variant must be E1, E2, E3, E3map or E4: {variant!r}")
    if variant in ('E3', 'E3map'):
        S = np.asarray(d['mergny_dA_spin_theta'], dtype=float)
    if sym_S is None:
        sym_S = variant == 'E3map'
    if sym_S:
        S = 0.5 * (S + S[::-1])                 # GL nodes are symmetric about the equator
    w = G.theta_weights(theta)
    S = S - np.sum(S * w) / np.sum(w)
    return {'form': 'exo_profiles', 'names': ['S(theta_spin)'] + [f'P{n}(cos gamma)' for n in range(1, 5)],
            'coeffs': [float(x) for x in c], 'const': 0.0, 'theta_S': np.asarray(theta, float),
            'S': S, 'c_impl': float(c_impl), 'case': case, 'variant': variant,
            'scale_zonal': scale_zonal, 'scale_odd': scale_odd, 'sym_S': bool(sym_S),
            'source': str(path)}


def dose_unit_field(path, scenario):
    """u = -(D/<D> - 1) of Halo's dose-darkening forcing, on its (theta, lam) grid.

    The dose-darkening term is dA_dose = kappa u (zero mean), with D/<D> the
    normalized deposited dose of ``scenario`` (``dose_norm_<scenario>``, e.g.
    'EN_1mm': Nordheim-lens electrons in the top 1 mm, heavy-ion skin dose
    discounted). The whole field is EXTERNAL frame: tidal coordinates.
    """
    d = np.load(path)
    u = -(np.asarray(d[f'dose_norm_{scenario}'], dtype=float) - 1.0)
    th, lm = np.asarray(d['theta'], float), np.asarray(d['lam'], float)
    return th, lm, u - G.area_mean(u, th, lm)


def add_dose(fit, path, scenario, kappa):
    """Return a copy of an exo_profiles fit with kappa x dose_unit_field added (variant E4)."""
    th, lm, u = dose_unit_field(path, scenario)
    out = dict(fit)
    out.update({'dose_theta': th, 'dose_lam': lm, 'dose_u': u, 'dose_kappa': float(kappa),
                'dose_scenario': scenario, 'dose_source': str(path)})
    return out


def _periodic_interp(th, lm, F, theta_t, lam_t):
    """Bilinear interpolation of F(theta, lam) (GL rows, periodic lam) at (theta_t, lam_t).

    Pole rows are added as the mean of the nearest row (a single-valued pole).
    """
    from scipy.interpolate import RegularGridInterpolator
    lam_c = np.append(lm, lm[0] + 2.0 * np.pi)
    Fc = np.column_stack([F, F[:, 0]])
    th_c = np.concatenate([[0.0], th, [np.pi]])
    Fc = np.vstack([np.full(Fc.shape[1], F[0].mean()), Fc, np.full(Fc.shape[1], F[-1].mean())])
    rgi = RegularGridInterpolator((th_c, lam_c), Fc)
    tt = np.asarray(theta_t, dtype=float)
    ll = np.mod(np.asarray(lam_t, dtype=float) - lm[0], 2.0 * np.pi) + lm[0]
    tt, ll = np.broadcast_arrays(tt, ll)
    return rgi(np.stack([tt.ravel(), ll.ravel()], -1)).reshape(tt.shape)


def template_field(fit, theta_t, lam_t):
    """A_exo^eq at tidal coordinates (zero area mean; no constant)."""
    if fit['form'] == 'exo_profiles':
        th = np.asarray(theta_t, dtype=float)
        cos_g = np.sin(th) * np.sin(lam_t)
        out = np.interp(th, fit['theta_S'], fit['S'])   # constant beyond the GL end nodes
        for n, cn in enumerate(fit['coeffs'], start=1):
            out = out + cn * _legendre(n, cos_g)
        if fit.get('dose_kappa'):
            out = out + fit['dose_kappa'] * _periodic_interp(fit['dose_theta'], fit['dose_lam'],
                                                           fit['dose_u'], th, lam_t)
        return out
    out = np.zeros(np.broadcast(theta_t, lam_t).shape)
    for c, f in zip(fit['coeffs'], basis(fit['form'], theta_t, lam_t)):
        out = out + c * f
    return out


def split_map(A, theta, lam, fit):
    """(A_exo^eq at t = 0, A_intr) on the body grid; A_exo + A_intr == A."""
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A_exo = template_field(fit, TH, LA)
    return A_exo, A - A_exo
