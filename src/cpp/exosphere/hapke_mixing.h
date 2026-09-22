/*
 * Hapke intimate-mixture albedo model for the exosphere ice-fraction coupling
 * (--ice-fraction-from-albedo=hapke-intimate).
 *
 * Scattering model: the thermal engine's own closed form, i.e. isotropic
 * scatterers with the Hapke H-function approximation, no opposition effect and
 * no macroscopic roughness (ptm_thermal_sim.cpp applyIncidenceAlbedo,
 * thermal_model.cpp Planet::albedoWithIncidenceFromA0):
 *
 *   gamma      = sqrt(1 - w)
 *   H(mu)      = (1 + 2 mu) / (1 + 2 mu gamma)
 *   A_h(mu0)   = 1 - gamma H(mu0)                   directional-hemispheric
 *   A_B(w)     = 2 * int_0^1 mu A_h(mu) dmu          spherical (Bond) albedo
 *
 * A_B is evaluated with 64-point Gauss-Legendre quadrature; sphericalAlbedoAnalytic()
 * is the exact integral, kept as the test reference.
 *
 * Anisotropic grains (w, g) enter through the similarity relation
 *   w* = w (1 - g) / (1 - w g),   A_B(w, g) = A_B(w*).
 *
 * Intimate mixing of an ice and a non-ice endmember. Let f be the ice share of
 * the extinction cross section (the projected-area fraction in the geometric-optics
 * limit, where Q_ext = 2 for both grain types). Then
 *   w_mix = f w_ice + (1 - f) w_non                       (extinction-weighted)
 *   g_mix = [f w_ice g_ice + (1 - f) w_non g_non] / w_mix  (scattering-weighted,
 *                                                           sigma_sca = w sigma_ext)
 *
 * Exposed ice area versus volume. With equal Q_ext, sigma_ext per unit volume
 * scales as 1/D, so for an ice volume fraction phi (of the solids) and grain-size
 * ratio r = D_ice / D_nonice:
 *   f   = phi / (phi + r (1 - phi))
 *   phi = f r / (f r + 1 - f)
 * Sublimation scales with the exposed ice AREA f. The ice-layer model keeps its
 * bookkeeping in VOLUME (nonice_concentration c = 1 - phi).
 */

#ifndef PTM_EXOSPHERE_HAPKE_MIXING_H
#define PTM_EXOSPHERE_HAPKE_MIXING_H

#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ptm {
namespace exosphere {
namespace hapke {

// 64-point Gauss-Legendre nodes/weights mapped to [0, 1]. Computed once by
// Newton iteration on P_64 (converges to machine precision in a few steps).
struct GaussLegendre64 {
    static constexpr int N = 64;
    std::array<double, N> mu{};
    std::array<double, N> wt{};
    GaussLegendre64() {
        for (int i = 0; i < N; ++i) {
            double x = std::cos(M_PI * (i + 0.75) / (N + 0.5));
            double dp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p0 = 1.0, p1 = x;
                for (int k = 2; k <= N; ++k) {
                    double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                    p0 = p1;
                    p1 = p2;
                }
                dp = N * (x * p1 - p0) / (x * x - 1.0);
                double dx = p1 / dp;
                x -= dx;
                if (std::abs(dx) < 1e-16) break;
            }
            // Recompute the derivative at the converged node for the weight.
            double p0 = 1.0, p1 = x;
            for (int k = 2; k <= N; ++k) {
                double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = N * (x * p1 - p0) / (x * x - 1.0);
            mu[i] = 0.5 * (x + 1.0);
            wt[i] = 0.5 * 2.0 / ((1.0 - x * x) * dp * dp);
        }
    }
};

inline const GaussLegendre64& gaussLegendre64() {
    static const GaussLegendre64 gl;
    return gl;
}

// Engine closed form: A_h(mu0) = 1 - gamma*H(mu0), isotropic scatterers.
inline double directionalHemisphericAlbedo(double w, double mu0) {
    const double gamma = std::sqrt(std::max(0.0, 1.0 - std::clamp(w, 0.0, 1.0)));
    const double H = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
    return 1.0 - gamma * H;
}

// Spherical (Bond) albedo A_B(w) = 2 int_0^1 mu A_h(mu) dmu, 64-point Gauss-Legendre.
inline double sphericalAlbedo(double w) {
    const auto& gl = gaussLegendre64();
    double s = 0.0;
    for (int i = 0; i < GaussLegendre64::N; ++i) {
        s += gl.wt[i] * gl.mu[i] * directionalHemisphericAlbedo(w, gl.mu[i]);
    }
    return 2.0 * s;
}

// Exact integral of the same expression (test reference only):
//   int_0^1 mu (1+2mu)/(1+a mu) dmu = 1/a + c - (c/a) ln(1+a),  a = 2 gamma, c = (a-2)/a^2
//   A_B = 1 - 2 gamma [1/a + c - (c/a) ln(1+a)]
// Loses precision as gamma -> 0 (w -> 1); valid for gamma >= ~1e-3.
inline double sphericalAlbedoAnalytic(double w) {
    const double gamma = std::sqrt(std::max(0.0, 1.0 - std::clamp(w, 0.0, 1.0)));
    const double a = 2.0 * gamma;
    const double c = (a - 2.0) / (a * a);
    const double I = 1.0 / a + c - (c / a) * std::log1p(a);
    return 1.0 - 2.0 * gamma * I;
}

// Similarity relation: anisotropic (w, g) -> isotropic-equivalent w*.
inline double similarityW(double w, double g) {
    return w * (1.0 - g) / (1.0 - w * g);
}

// Inverse of the similarity relation: w = w* / (1 - g + w* g).
inline double wFromSimilarityW(double w_star, double g) {
    return w_star / (1.0 - g + w_star * g);
}

inline double bondAlbedo(double w, double g) {
    return sphericalAlbedo(similarityW(w, g));
}

// Invert A_B(w*) = A for w* in [0, 1] by bisection (A_B is strictly increasing in w*).
inline double invertSphericalAlbedo(double A) {
    const double A_max = sphericalAlbedo(1.0);
    if (!(A >= 0.0) || A > A_max) {
        throw std::invalid_argument("invertSphericalAlbedo: Bond albedo outside [0, A_B(w=1)]");
    }
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 200 && hi - lo > 1e-16; ++it) {
        const double m = 0.5 * (lo + hi);
        if (sphericalAlbedo(m) < A) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
}

// Single-scattering albedo w of an endmember with Bond albedo A and asymmetry g.
inline double singleScatteringAlbedoFromBond(double A, double g) {
    return wFromSimilarityW(invertSphericalAlbedo(A), g);
}

// Exposed-area (extinction) fraction <-> volume fraction, r = D_ice / D_nonice.
inline double exposedAreaFromVolume(double phi, double r) {
    phi = std::clamp(phi, 0.0, 1.0);
    const double den = phi + r * (1.0 - phi);
    return (den > 0.0) ? phi / den : 0.0;
}

inline double volumeFromExposedArea(double f, double r) {
    f = std::clamp(f, 0.0, 1.0);
    const double den = f * r + 1.0 - f;
    return (den > 0.0) ? f * r / den : 0.0;
}

// Two-endmember intimate mixture.
struct IntimateMixture {
    double w_ice = 0.0, g_ice = 0.0;
    double w_non = 0.0, g_non = 0.0;

    // Bond albedo of the mixture with ice extinction fraction f.
    double bond(double f) const {
        f = std::clamp(f, 0.0, 1.0);
        const double w = f * w_ice + (1.0 - f) * w_non;
        const double s = f * w_ice * g_ice + (1.0 - f) * w_non * g_non;
        const double g = (w > 0.0) ? s / w : 0.0;
        return bondAlbedo(w, g);
    }

    // Solve bond(f) = A for f in [0, 1] by bisection, clamping at the endmembers.
    // clamp: -1 if A <= bond(0) (f = 0), +1 if A >= bond(1) (f = 1), else 0.
    double invert(double A, int* clamp = nullptr) const {
        const double A0 = bond(0.0), A1 = bond(1.0);
        if (A <= A0) { if (clamp) *clamp = -1; return 0.0; }
        if (A >= A1) { if (clamp) *clamp = +1; return 1.0; }
        if (clamp) *clamp = 0;
        double lo = 0.0, hi = 1.0;
        for (int it = 0; it < 200 && hi - lo > 1e-16; ++it) {
            const double m = 0.5 * (lo + hi);
            if (bond(m) < A) lo = m; else hi = m;
        }
        return 0.5 * (lo + hi);
    }
};

// Build the mixture from endmember Bond albedos and asymmetry parameters.
inline IntimateMixture makeIntimateMixture(double A_ice, double g_ice,
                                           double A_non, double g_non) {
    IntimateMixture m;
    m.g_ice = g_ice;
    m.g_non = g_non;
    m.w_ice = singleScatteringAlbedoFromBond(A_ice, g_ice);
    m.w_non = singleScatteringAlbedoFromBond(A_non, g_non);
    return m;
}

} // namespace hapke
} // namespace exosphere
} // namespace ptm

#endif // PTM_EXOSPHERE_HAPKE_MIXING_H
