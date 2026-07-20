#pragma once

// Smith (1967) statistical-roughness anisotropic thermal-emission model.
//
// A rough surface (Gaussian micro-slope distribution, RMS slope w) does NOT
// emit isotropically: its hot Sun-facing micro-facets beam thermal IR
// preferentially toward the Sun. At the lunar pole, where the Sun grazes the
// horizon and a crater wall is sloped toward the floor, this redistributes IR
// from the wall toward the floor -- a Sun-direction-dependent enhancement of
// wall->floor heating that the Lambertian radiosity misses.
//
// This header provides the directional emission factor
//   chi(cos_sun, cos_e, cos_psi; w)
//     = (rough-surface radiance toward d) / (smooth-Lambertian radiance toward d),
// energy-normalized so the cos_e-weighted hemispherical mean of chi is exactly 1
// (pure angular redistribution; the wall still emits eps*sigma*T^4 in total).
//   cos_sun = n . s_hat   (mean normal . Sun),        in [0,1]
//   cos_e   = n . d        (mean normal . emission dir), in [0,1]
//   cos_psi = d . s_hat    (emission dir . Sun),        in [-1,1]
//
// chi -> 1 everywhere as w -> 0. Toward the Sun (small psi) chi > 1; away, < 1.
//
// The math here is the single source of truth shared by the production radiosity
// manager (radiation_exchange.cpp) and the standalone validator
// (test_beam_chi.cpp).

#include <vector>
#include <cmath>
#include <algorithm>

namespace radiation {
namespace smith {

// Smith Lambda for a Gaussian (Beckmann) slope distribution, RMS slope w.
// a = cot(theta) / (sqrt(2) w). Lambda = 0.5 [ exp(-a^2)/(a sqrt(pi)) - erfc(a) ].
inline double lambda(double a) {
    if (a > 8.0) return 0.0;          // near-normal incidence: no masking
    if (a < 1e-6) a = 1e-6;           // grazing guard
    return 0.5 * (std::exp(-a * a) / (a * std::sqrt(M_PI)) - std::erfc(a));
}

// Monostatic Smith masking/shadowing G1(theta) = 1/(1+Lambda) for a direction at
// polar angle theta from the mean normal (cos_t in (0,1]).
inline double G1(double cos_t, double w) {
    if (cos_t <= 0.0) return 0.0;
    if (cos_t >= 1.0) return 1.0;
    double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
    double cot = cos_t / std::max(1e-9, sin_t);
    double a = cot / (std::sqrt(2.0) * w);
    return 1.0 / (1.0 + lambda(a));
}

// I(d) = integral over the Gaussian micro-slope distribution of
//   max(0, m.s_hat) * max(0, m.d),  m = (-p,-q,1)/|.|,  p = w*u, q = w*v.
// Hot Sun-facing facets (m.s_hat) projected toward d (m.d). The per-dim weights
// gw carry the unit-Gaussian normalization in u = p/w.
inline double integral(const double s[3], const double d[3], double w,
                       const std::vector<double>& nodes,
                       const std::vector<double>& gw) {
    double acc = 0.0;
    const int N = static_cast<int>(nodes.size());
    for (int iu = 0; iu < N; ++iu) {
        double p = w * nodes[iu];
        for (int iv = 0; iv < N; ++iv) {
            double q = w * nodes[iv];
            double inv = 1.0 / std::sqrt(1.0 + p * p + q * q);
            double mx = -p * inv, my = -q * inv, mz = inv;
            double ms = mx * s[0] + my * s[1] + mz * s[2];
            if (ms <= 0.0) continue;
            double md = mx * d[0] + my * d[1] + mz * d[2];
            if (md <= 0.0) continue;
            acc += gw[iu] * gw[iv] * ms * md;
        }
    }
    return acc;
}

// Flat, energy-normalized chi table on a regular grid:
//   index(is,ie,ipsi) = (is*ne + ie)*npsi + ipsi
// cos_sun cell centers (is+0.5)/ns in (0,1); cos_e cell centers (ie+0.5)/ne in
// (0,1); cos_psi cell centers -1 + 2(ipsi+0.5)/npsi in (-1,1).
inline std::vector<float> buildChiTable(double w_rad, int ns, int ne, int npsi) {
    std::vector<float> table(static_cast<size_t>(ns) * ne * npsi, 1.0f);
    const double w = w_rad;

    // Gaussian-slope quadrature (unit Gaussian in u = p/w), normalized to sum 1.
    const int Nq = 33;
    std::vector<double> nodes(Nq), gw(Nq);
    const double umin = -4.0, umax = 4.0, du = (umax - umin) / (Nq - 1);
    double wsum = 0.0;
    for (int k = 0; k < Nq; ++k) {
        double u = umin + k * du;
        nodes[k] = u;
        gw[k] = std::exp(-0.5 * u * u);
        wsum += gw[k];
    }
    for (int k = 0; k < Nq; ++k) gw[k] /= wsum;

    // Hemisphere quadrature for the energy-normalization integral Z(theta_sun).
    const int Nt = 24, Np = 48;
    const double dtheta = (M_PI / 2.0) / Nt;
    const double dphi = (2.0 * M_PI) / Np;

    #pragma omp parallel for schedule(dynamic)
    for (int is = 0; is < ns; ++is) {
        double cos_sun = (is + 0.5) / ns;
        double sin_sun = std::sqrt(std::max(0.0, 1.0 - cos_sun * cos_sun));
        double s[3] = { sin_sun, 0.0, cos_sun };

        // Z = integral over the emission hemisphere of G1(theta') I(d') dOmega'.
        double Z = 0.0;
        for (int it = 0; it < Nt; ++it) {
            double te = (it + 0.5) * dtheta;
            double cos_e = std::cos(te), sin_e = std::sin(te);
            double g1 = G1(cos_e, w);
            for (int ip = 0; ip < Np; ++ip) {
                double phi = (ip + 0.5) * dphi;
                double d[3] = { sin_e * std::cos(phi), sin_e * std::sin(phi), cos_e };
                double I = integral(s, d, w, nodes, gw);
                Z += g1 * I * sin_e * dtheta * dphi;
            }
        }
        if (Z < 1e-12) Z = 1e-12;

        for (int ie = 0; ie < ne; ++ie) {
            double cos_e = (ie + 0.5) / ne;
            double sin_e = std::sqrt(std::max(0.0, 1.0 - cos_e * cos_e));
            double g1 = G1(cos_e, w);
            for (int ipsi = 0; ipsi < npsi; ++ipsi) {
                double cos_psi = -1.0 + 2.0 * (ipsi + 0.5) / npsi;
                double chi = 1.0;
                double denom = sin_sun * sin_e;
                if (denom > 1e-9) {
                    double cphi = (cos_psi - cos_sun * cos_e) / denom;
                    if (cphi >= -1.0 && cphi <= 1.0) {
                        double sphi = std::sqrt(std::max(0.0, 1.0 - cphi * cphi));
                        double d[3] = { sin_e * cphi, sin_e * sphi, cos_e };
                        double I = integral(s, d, w, nodes, gw);
                        chi = M_PI * g1 * I / (cos_e * Z);
                    }
                }
                if (!std::isfinite(chi)) chi = 1.0;
                chi = std::min(4.0, std::max(0.0, chi));
                size_t idx = (static_cast<size_t>(is) * ne + ie) * npsi + ipsi;
                table[idx] = static_cast<float>(chi);
            }
        }
    }
    return table;
}

// Trilinear lookup into a table from buildChiTable(). Returns 1.0 outside the
// active domain (cos_sun <= 0 or cos_e <= 0).
inline float evalChi(const std::vector<float>& table, int ns, int ne, int npsi,
                     double cos_sun, double cos_e, double cos_psi) {
    if (table.empty()) return 1.0f;
    if (cos_sun <= 0.0 || cos_e <= 0.0) return 1.0f;
    double fs = cos_sun * ns - 0.5;
    double fe = cos_e * ne - 0.5;
    double fp = (cos_psi + 1.0) * 0.5 * npsi - 0.5;
    auto clampf = [](double f, int n) {
        return std::min(static_cast<double>(n - 1), std::max(0.0, f));
    };
    fs = clampf(fs, ns); fe = clampf(fe, ne); fp = clampf(fp, npsi);
    int is0 = static_cast<int>(fs), ie0 = static_cast<int>(fe), ip0 = static_cast<int>(fp);
    int is1 = std::min(is0 + 1, ns - 1);
    int ie1 = std::min(ie0 + 1, ne - 1);
    int ip1 = std::min(ip0 + 1, npsi - 1);
    double ds = fs - is0, de = fe - ie0, dp = fp - ip0;
    auto at = [&](int a, int b, int c) -> double {
        return table[(static_cast<size_t>(a) * ne + b) * npsi + c];
    };
    double c00 = at(is0, ie0, ip0) * (1 - dp) + at(is0, ie0, ip1) * dp;
    double c01 = at(is0, ie1, ip0) * (1 - dp) + at(is0, ie1, ip1) * dp;
    double c10 = at(is1, ie0, ip0) * (1 - dp) + at(is1, ie0, ip1) * dp;
    double c11 = at(is1, ie1, ip0) * (1 - dp) + at(is1, ie1, ip1) * dp;
    double c0 = c00 * (1 - de) + c01 * de;
    double c1 = c10 * (1 - de) + c11 * de;
    return static_cast<float>(c0 * (1 - ds) + c1 * ds);
}

} // namespace smith
} // namespace radiation
