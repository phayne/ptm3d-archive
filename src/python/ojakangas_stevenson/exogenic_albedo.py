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


def template_field(fit, theta_t, lam_t):
    """A_exo^eq at tidal coordinates (zero area mean; no constant)."""
    out = np.zeros(np.broadcast(theta_t, lam_t).shape)
    for c, f in zip(fit['coeffs'], basis(fit['form'], theta_t, lam_t)):
        out = out + c * f
    return out


def split_map(A, theta, lam, fit):
    """(A_exo^eq at t = 0, A_intr) on the body grid; A_exo + A_intr == A."""
    TH, LA = np.meshgrid(theta, lam, indexing='ij')
    A_exo = template_field(fit, TH, LA)
    return A_exo, A - A_exo
