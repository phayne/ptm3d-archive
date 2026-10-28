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
//   index(is,ie,iphi) = (is*ne + ie)*nphi + iphi
// cos_sun cell centers (is+0.5)/ns in (0,1); cos_e cell centers (ie+0.5)/ne in
// (0,1); AZIMUTH cosine cos_phi cell centers -1 + 2(iphi+0.5)/nphi in (-1,1).
//
// ⛔ THE THIRD AXIS IS cos_phi, NOT cos_psi (changed 2026-08-24). chi is a
// function of (theta_sun, theta_e, phi), and cos_psi = d.s_hat is a DERIVED
// quantity: cos_psi = cos_sun cos_e + sin_sun sin_e cos_phi. On a regular
// (cos_sun, cos_e, cos_psi) grid most cells are geometrically UNREACHABLE
// (|cos_phi| > 1) and were filled with chi = 1, so every trilinear stencil
// straddling that curved boundary was dragged toward 1 -- and at the lunar pole
// the wall->floor flux sits right ON the boundary (near-zero phase, cos_phi ~ +1),
// which is where chi is largest and varies fastest.
//
// MEASURED on the real flux-weighted wall->floor pair distribution (the physics review's
// calc10_pairs.npy, 5.3M pairs, w = 32.5 deg): exact flux-weighted mean chi
// 1.657; the old cos_psi-indexed 24x24x33 table read 1.595 (-3.8%, p95 pointwise
// error 24%), and refining that grid to 64x128x129 only reached 1.617 -- the bias
// is the parametrization, not the resolution. Re-indexed on cos_phi at the SAME
// 24x24x33 the error drops to -0.33% (p95 1.6%), and at 48x48x65 to -0.12%
// (p95 1.2%). Reproduce with scripts/chi_table_resolution.py (its --pairs input is
// a measurement, not a repo artifact; the script says so and refuses to invent one).
//
// The evalChi() signature is unchanged: it still takes cos_psi and does the
// conversion, so no caller has to know this.
inline std::vector<float> buildChiTable(double w_rad, int ns, int ne, int nphi) {
    std::vector<float> table(static_cast<size_t>(ns) * ne * nphi, 1.0f);
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
            for (int iphi = 0; iphi < nphi; ++iphi) {
                // Every cell is reachable now: cos_phi spans its own full range.
                double cphi = -1.0 + 2.0 * (iphi + 0.5) / nphi;
                double sphi = std::sqrt(std::max(0.0, 1.0 - cphi * cphi));
                double d[3] = { sin_e * cphi, sin_e * sphi, cos_e };
                double I = integral(s, d, w, nodes, gw);
                double chi = M_PI * g1 * I / (cos_e * Z);
                if (!std::isfinite(chi)) chi = 1.0;
                chi = std::min(4.0, std::max(0.0, chi));
                size_t idx = (static_cast<size_t>(is) * ne + ie) * nphi + iphi;
                table[idx] = static_cast<float>(chi);
            }
        }
    }
    return table;
}

// The cos_sun axis of the chi lookup, pre-resolved.
//
// WHY THIS EXISTS (perf/beaming-overhead, 2026-08-25): cos_sun = n_j . s_hat
// depends only on the EMITTER and on the timestep's Sun direction -- never on
// which receiver the pair points at. In the production beam pass every emitter
// appears in hundreds to thousands of pairs, so evalChi() was recomputing one
// sqrt, one multiply-add and the whole cos_sun index/weight triple once per PAIR
// instead of once per FACE. MEASURED on the r50 beam pass (193.6 M pairs, 48
// threads): the chi lookup was 0.775 ns of the kernel's 1.195 ns/pair (65% of
// it), and hoisting this axis out of the pair loop took the whole kernel to
// 1.001 ns/pair.
//
// The fields are exactly the quantities evalChi() used to compute inline, with
// the same expressions in the same order, so the hoisted path is bit-identical
// to the per-pair one -- verified, not assumed (see the beaming gates).
struct SunAxis {
    double sin_sun;   // sqrt(max(0, 1 - cos_sun^2))
    double ds;        // trilinear weight on the cos_sun axis
    int    is0, is1;  // bracketing chi-table slabs
    bool   active;    // cos_sun > 0; false => chi is identically 1
};

inline SunAxis resolveSunAxis(double cos_sun, int ns) {
    SunAxis ax{};
    ax.active = (cos_sun > 0.0);
    if (!ax.active) { ax.sin_sun = 0.0; ax.ds = 0.0; ax.is0 = ax.is1 = 0; return ax; }
    ax.sin_sun = std::sqrt(std::max(0.0, 1.0 - cos_sun * cos_sun));
    double fs = cos_sun * ns - 0.5;
    fs = std::min(static_cast<double>(ns - 1), std::max(0.0, fs));
    ax.is0 = static_cast<int>(fs);
    ax.is1 = std::min(ax.is0 + 1, ns - 1);
    ax.ds  = fs - ax.is0;
    return ax;
}

// Trilinear lookup with the cos_sun axis already resolved by the caller. This is
// THE implementation; evalChi() below is a thin wrapper that resolves the axis
// first, so the hoisted production path and the reference path cannot drift
// apart. The caller still supplies cos_psi = d.s_hat, and the azimuth cosine the
// table is indexed on is derived here. Returns 1.0 outside the active domain
// (cos_sun <= 0, cos_e <= 0) and for a geometrically UNREACHABLE (cos_sun,
// cos_e, cos_psi) triple (|cos_phi| > 1), which is what the exact expression
// does too. The degenerate slabs sin_sun -> 0 / sin_e -> 0 are azimuth-
// independent, so any cos_phi reads the same value; 0 is used.
inline float evalChiAxis(const std::vector<float>& table, int ns, int ne, int nphi,
                         const SunAxis& ax, double cos_sun, double cos_e,
                         double cos_psi) {
    if (table.empty()) return 1.0f;
    if (!ax.active || cos_e <= 0.0) return 1.0f;
    const double sin_e   = std::sqrt(std::max(0.0, 1.0 - cos_e * cos_e));
    const double denom = ax.sin_sun * sin_e;
    double cphi = 0.0;
    if (denom > 1e-9) {
        cphi = (cos_psi - cos_sun * cos_e) / denom;
        // ⛔ CLAMP AT THE BOUNDARY, DO NOT REJECT THERE. |cos_phi| = 1 is the
        // principal plane -- the azimuth that carries essentially all of the
        // grazing-Sun wall->floor flux -- and a caller that forms cos_psi from
        // its own sin_sun/sin_e lands there at 1 +/- a few ulp. Rejecting on a
        // bare `> 1.0` sends exactly that geometry to chi = 1, i.e. beaming
        // silently OFF where it matters most; it cost one build to find, under
        // -O3 without -ffast-math (with -ffast-math the reassociation happened to
        // land on the other side, so the failure was not even reproducible
        // across optimization settings). Reject only genuinely unreachable
        // triples, which no geometric caller can produce.
        constexpr double kReach = 1e-6;
        if (cphi < -1.0 - kReach || cphi > 1.0 + kReach) return 1.0f;
        cphi = std::min(1.0, std::max(-1.0, cphi));
    }
    double fe = cos_e * ne - 0.5;
    double fp = (cphi + 1.0) * 0.5 * nphi - 0.5;
    auto clampf = [](double f, int n) {
        return std::min(static_cast<double>(n - 1), std::max(0.0, f));
    };
    fe = clampf(fe, ne); fp = clampf(fp, nphi);
    int ie0 = static_cast<int>(fe), ip0 = static_cast<int>(fp);
    int ie1 = std::min(ie0 + 1, ne - 1);
    int ip1 = std::min(ip0 + 1, nphi - 1);
    double de = fe - ie0, dp = fp - ip0;
    auto at = [&](int a, int b, int c) -> double {
        return table[(static_cast<size_t>(a) * ne + b) * nphi + c];
    };
    double c00 = at(ax.is0, ie0, ip0) * (1 - dp) + at(ax.is0, ie0, ip1) * dp;
    double c01 = at(ax.is0, ie1, ip0) * (1 - dp) + at(ax.is0, ie1, ip1) * dp;
    double c10 = at(ax.is1, ie0, ip0) * (1 - dp) + at(ax.is1, ie0, ip1) * dp;
    double c11 = at(ax.is1, ie1, ip0) * (1 - dp) + at(ax.is1, ie1, ip1) * dp;
    double c0 = c00 * (1 - de) + c01 * de;
    double c1 = c10 * (1 - de) + c11 * de;
    return static_cast<float>(c0 * (1 - ax.ds) + c1 * ax.ds);
}

// Trilinear lookup into a table from buildChiTable(). Unchanged signature and
// unchanged result: it resolves the cos_sun axis and defers to evalChiAxis().
inline float evalChi(const std::vector<float>& table, int ns, int ne, int nphi,
                     double cos_sun, double cos_e, double cos_psi) {
    if (table.empty()) return 1.0f;
    if (cos_sun <= 0.0 || cos_e <= 0.0) return 1.0f;
    return evalChiAxis(table, ns, ne, nphi, resolveSunAxis(cos_sun, ns),
                       cos_sun, cos_e, cos_psi);
}

} // namespace smith
} // namespace radiation
