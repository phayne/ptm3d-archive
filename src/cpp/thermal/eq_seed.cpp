#include "eq_seed.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "thermal_model.h"   // ABSOLUTE_ZERO_K

namespace thermal {

namespace {
std::string normalizeKey(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '_' || c == '-' || c == ' ') continue;
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return v;
}
}  // namespace

bool parseEqSeedMode(const std::string& s, EqSeedMode& out) {
    const std::string v = normalizeKey(s);
    if (v.empty() || v == "physical" || v == "default" || v == "perface") {
        out = EqSeedMode::Physical;
        return true;
    }
    if (v == "legacy" || v == "historical") {
        out = EqSeedMode::Legacy;
        return true;
    }
    return false;
}

bool parseEqSeedAnchor(const std::string& s, EqSeedAnchor& out) {
    const std::string v = normalizeKey(s);
    if (v.empty() || v == "lumped" || v == "default" || v == "derived") {
        out = EqSeedAnchor::Lumped;
        return true;
    }
    if (v == "warm" || v == "warmbracket" || v == "t4" || v == "legacy") {
        out = EqSeedAnchor::Warm;
        return true;
    }
    return false;
}

const char* eqSeedAnchorStatusName(EqSeedAnchorStatus s) {
    switch (s) {
        case EqSeedAnchorStatus::WarmRequested:            return "warm-requested";
        case EqSeedAnchorStatus::NoForcingSeries:          return "no-forcing-series";
        case EqSeedAnchorStatus::Lumped:                   return "lumped";
        case EqSeedAnchorStatus::LumpedInputLimited:       return "lumped-input-limited";
        case EqSeedAnchorStatus::LumpedTwoTimescale:       return "lumped-two-timescale";
        case EqSeedAnchorStatus::ZeroSwing:                return "zero-swing";
        case EqSeedAnchorStatus::FallbackTwoTimescale:     return "fallback-two-timescale";
        case EqSeedAnchorStatus::FallbackRadiationCoupled: return "fallback-radiation-coupled";
        case EqSeedAnchorStatus::FallbackLatentHeat:       return "fallback-latent-heat";
        case EqSeedAnchorStatus::FallbackNonUniformSamples: return "fallback-nonuniform-samples";
        case EqSeedAnchorStatus::FallbackFewLitSamples:    return "fallback-few-lit-samples";
        case EqSeedAnchorStatus::FallbackShortSpan:        return "fallback-short-span";
        case EqSeedAnchorStatus::FallbackBadProperties:    return "fallback-bad-properties";
        case EqSeedAnchorStatus::FallbackNotConverged:     return "fallback-not-converged";
    }
    return "unknown";
}

// ===========================================================================
// Real one-sided power spectrum, arbitrary length.
//
// A generic length is not optional here: a forcing series is however many
// samples the run happens to have (118 = 2*59 on the shipped lunar crater,
// 1440 on polar sites). Radix-2 when n is a power of two; Bluestein (chirp-z) over
// the next power of two >= 2n-1 otherwise, so the cost is O(n log n) for every
// n including primes. Plans are cached per thread, keyed on n, because every
// face in a mesh shares the same n.
// ===========================================================================
namespace {

using cplx = std::complex<double>;

void fftPow2(cplx* a, int m, bool inverse) {
    // bit reversal
    for (int i = 1, j = 0; i < m; ++i) {
        int bit = m >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= m; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
        const cplx wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < m; i += len) {
            cplx w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const cplx u = a[i + j];
                const cplx v = a[i + j + len / 2] * w;
                a[i + j]             = u + v;
                a[i + j + len / 2]   = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) {
        const double s = 1.0 / static_cast<double>(m);
        for (int i = 0; i < m; ++i) a[i] *= s;
    }
}

/// Bluestein plan: the chirp and the FFT of the kernel, both independent of the
/// data, so they are built once per distinct n.
struct BluesteinPlan {
    int n = 0, m = 0;
    std::vector<cplx> chirp;   // w[j] = exp(-i*pi*j^2/n), j < n
    std::vector<cplx> bfft;    // FFT of the kernel
};

const BluesteinPlan& bluesteinPlan(int n) {
    static thread_local BluesteinPlan plan;
    if (plan.n == n) return plan;
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    plan.n = n;
    plan.m = m;
    plan.chirp.assign(n, cplx(0.0, 0.0));
    for (int j = 0; j < n; ++j) {
        // j*j can overflow for large n and, more importantly, loses precision
        // in the angle. (j*j) mod 2n keeps both exact.
        const long long jj = (static_cast<long long>(j) * j) % (2LL * n);
        const double ang = -M_PI * static_cast<double>(jj) / static_cast<double>(n);
        plan.chirp[j] = cplx(std::cos(ang), std::sin(ang));
    }
    plan.bfft.assign(m, cplx(0.0, 0.0));
    plan.bfft[0] = std::conj(plan.chirp[0]);
    for (int j = 1; j < n; ++j) {
        plan.bfft[j]     = std::conj(plan.chirp[j]);
        plan.bfft[m - j] = std::conj(plan.chirp[j]);
    }
    fftPow2(plan.bfft.data(), m, /*inverse=*/false);
    return plan;
}

/// X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n), k = 0..n-1.
void dftAny(const double* x, std::size_t stride, int n, std::vector<cplx>& X) {
    X.assign(n, cplx(0.0, 0.0));
    if (n <= 0) return;
    if (n == 1) { X[0] = cplx(x[0], 0.0); return; }
    if ((n & (n - 1)) == 0) {                       // power of two
        for (int j = 0; j < n; ++j) X[j] = cplx(x[static_cast<std::size_t>(j) * stride], 0.0);
        fftPow2(X.data(), n, /*inverse=*/false);
        return;
    }
    const BluesteinPlan& p = bluesteinPlan(n);
    static thread_local std::vector<cplx> a;
    a.assign(p.m, cplx(0.0, 0.0));
    for (int j = 0; j < n; ++j)
        a[j] = cplx(x[static_cast<std::size_t>(j) * stride], 0.0) * p.chirp[j];
    fftPow2(a.data(), p.m, /*inverse=*/false);
    for (int j = 0; j < p.m; ++j) a[j] *= p.bfft[j];
    fftPow2(a.data(), p.m, /*inverse=*/true);
    for (int k = 0; k < n; ++k) X[k] = a[k] * p.chirp[k];
}

}  // namespace

void eqSeedOneSidedPower(const double* x, std::size_t stride, int n,
                         std::vector<double>& P_out) {
    const int nb = (n > 0) ? (n / 2 + 1) : 0;
    P_out.assign(std::max(0, nb), 0.0);
    if (n <= 1 || x == nullptr) return;
    static thread_local std::vector<cplx> X;
    dftAny(x, stride, n, X);
    const double inv = 1.0 / static_cast<double>(n);
    for (int k = 1; k < nb; ++k) {
        const cplx c = X[k] * inv;
        P_out[k] = 2.0 * std::norm(c);
    }
    // A real series of EVEN length has an unpaired Nyquist bin: it is counted
    // once, not twice, or the variance identity sum_{k>=1} P_k = var(x) fails.
    if (n % 2 == 0) P_out[nb - 1] *= 0.5;
    P_out[0] = 0.0;   // DC carries no information the anchor can use
}

// ===========================================================================
// Layered surface admittance.
// ===========================================================================
namespace {

/// tanh of a complex argument, guarded. For |Re(g*d)| >~ 20 the layer is
/// electrically infinite and tanh saturates at +-1; evaluating std::tanh there
/// overflows the intermediate exponentials for no gain. Deep cells on a real
/// grid are routinely in that regime (z_max is tens of skin depths).
cplx guardedTanh(const cplx& v) {
    const double re = v.real();
    if (re > 20.0)  return cplx(1.0, 0.0);
    if (re < -20.0) return cplx(-1.0, 0.0);
    return std::tanh(v);
}

double heatCapacityAt(const EqSeedForcing& fc, double T) {
    double cp = fc.cp0;
    double Tp = T;
    for (int j = 0; j < fc.n_cp_coeff; ++j) {
        cp += fc.cp_coeff[j] * Tp;
        Tp *= T;
    }
    return cp;
}

double conductivityAt(const EqSeedInput& in, int node, double T) {
    if (in.cond == ConductivityModel::Martinez2021)
        return martinezK(T, in.rho ? in.rho[node] : 0.0);
    return hayneK(T, in.kc ? in.kc[node] : 0.0, in.R350);
}

}  // namespace

double eqSeedKirchhoffGInv(double G, double R4) {
    if (!(R4 > 0.0)) return G;                 // exact identity: the chi==0 arm
    if (!(G > 0.0)) return G;
    // f(T) = T + R4*T^4 - G is increasing and convex, and f(G) = R4*G^4 >= 0,
    // so Newton started at T = G decreases monotonically onto the root. No
    // bracketing, no step limiter, and it cannot overshoot below the root.
    double T = G;
    for (int it = 0; it < 60; ++it) {
        const double T2 = T * T;
        const double f  = T + R4 * T2 * T2 - G;
        const double df = 1.0 + 4.0 * R4 * T2 * T;
        const double dT = f / df;
        T -= dT;
        if (std::fabs(dT) <= 1e-15 * std::fabs(T)) break;
    }
    return T;
}

double eqSeedGammaEff(const EqSeedInput& in, const EqSeedForcing& fc,
                      double omega, double T_ref) {
    if (in.n < 2 || in.z == nullptr || !(omega > 0.0)) return -1.0;
    const double cp = heatCapacityAt(fc, T_ref);
    if (!(cp > 0.0)) return -1.0;               // loud: caller turns this into a fallback

    // Cell j spans nodes [j, j+1]. Properties are taken at the CELL CENTER, as
    // the arithmetic mean of the two bounding nodes -- the second-order
    // quadrature for the exponentially graded lunar column. (The conductive
    // profile above uses the SHALLOWER node instead, matching the CN interface
    // conductance it has to be consistent with; that choice is upwind by
    // design and would bias Gamma_eff toward the surface value here.)
    const int ncell = in.n - 1;
    cplx Y(0.0, 0.0);
    bool have = false;
    for (int j = ncell - 1; j >= 0; --j) {
        const double d = in.z[j + 1] - in.z[j];
        if (!(d > 0.0)) continue;
        const double k_lo = conductivityAt(in, j, T_ref);
        const double k_hi = conductivityAt(in, j + 1, T_ref);
        const double k    = 0.5 * (k_lo + k_hi);
        const double rho  = in.rho ? 0.5 * (in.rho[j] + in.rho[j + 1]) : 0.0;
        if (!(k > 0.0) || !(rho > 0.0)) continue;
        const cplx g  = std::sqrt(cplx(0.0, omega * rho * cp / k));
        const cplx Yl = k * g;                    // = sqrt(i*omega*rho*cp*k)
        if (!have) { Y = Yl; have = true; }       // semi-infinite termination
        const cplx t   = guardedTanh(g * d);
        const cplx num = Y + Yl * t;
        const cplx den = Yl + Y * t;
        if (std::abs(den) <= 0.0) continue;
        Y = Yl * num / den;
    }
    if (!have) return -1.0;
    const double gam = std::abs(Y) / std::sqrt(omega);
    return std::isfinite(gam) ? gam : -1.0;
}

// ===========================================================================
// The lumped-substrate anchor.
// ===========================================================================
namespace {

/// Shared implementation. `u` carries the per-sample deviations T_i - T_d; when
/// `warm_start` is true it is reused from the previous (G, T_d) property pass,
/// which is what keeps the closure's cost at one full Newton solve plus a few
/// cheap refinements rather than six full solves.
double lumpedAnchorImpl(const double* F, std::size_t stride, int n,
                        double Qb, double eps, double sigma, double G,
                        double* T_warm_out, double* T_cold_out, int* iters_out,
                        std::vector<double>& u, bool warm_start, double Td0) {
    if (iters_out) *iters_out = -1;
    if (F == nullptr || n <= 0) return -1.0;
    const double a = eps * sigma;
    if (!(a > 0.0)) return -1.0;
    if (!(Qb >= 0.0)) Qb = 0.0;
    if (!(G >= 0.0))  G  = 0.0;

    const double inv_n = 1.0 / static_cast<double>(n);
    double sumF = 0.0, sumTc = 0.0;
    for (int i = 0; i < n; ++i) {
        double Fi = F[static_cast<std::size_t>(i) * stride];
        if (!(Fi >= 0.0)) Fi = 0.0;            // also catches NaN
        sumF  += Fi;
        sumTc += std::sqrt(std::sqrt((Fi + Qb) / a));
    }
    const double T_warm = std::sqrt(std::sqrt((sumF * inv_n + Qb) / a));
    const double T_cold = sumTc * inv_n;
    if (T_warm_out) *T_warm_out = T_warm;
    if (T_cold_out) *T_cold_out = T_cold;
    if (!std::isfinite(T_warm)) return -1.0;

    // Newton on the scalar T_d. The per-sample quartic is solved for the
    // DEVIATION u_i = T_i - T_d:
    //     f(u)  = a*(T_d+u)^4 + G*u - (F_i + Qb) = 0
    //     f'(u) = 4*a*(T_d+u)^3 + G
    // and the outer residual/derivative are
    //     g(T_d)  = <u_i>
    //     g'(T_d) = -<h_i/(h_i+G)>,   h_i = 4*a*T_i^3
    // Both vanish like 1/(h+G) as G -> inf, so the step g/g' stays O(1) and
    // the warm limit is reached to machine precision instead of dividing 0/0.
    // Solving for u (never T_i - T_d) is what keeps <u> free of cancellation.
    if (!warm_start || u.size() != static_cast<std::size_t>(n))
        u.assign(static_cast<std::size_t>(n), 0.0);

    double Td = warm_start ? Td0 : T_warm;
    if (!(Td > 0.0)) Td = T_warm;
    const double tol = 1e-12 * std::max(1.0, T_warm);
    int it = 0;
    bool converged = false;
    for (it = 0; it < 60; ++it) {
        double sum_u = 0.0, sum_dg = 0.0;
        for (int i = 0; i < n; ++i) {
            double Fi = F[static_cast<std::size_t>(i) * stride];
            if (!(Fi >= 0.0)) Fi = 0.0;
            const double rhs = Fi + Qb;
            double ui = u[i];
            double Ti = Td + ui;
            if (!(Ti > 0.0)) { Ti = 0.0; ui = -Td; }
            for (int k = 0; k < 60; ++k) {
                const double T3 = Ti * Ti * Ti;
                const double f  = a * T3 * Ti + G * ui - rhs;
                const double df = 4.0 * a * T3 + G;
                if (!(df > 0.0)) break;
                const double du = f / df;
                ui -= du;
                Ti  = Td + ui;
                if (Ti < 0.0) { Ti = 0.0; ui = -Td; }
                if (std::fabs(du) <= 1e-14 * std::max(1.0, std::fabs(Ti))) break;
            }
            u[i] = ui;
            const double hi = 4.0 * a * Ti * Ti * Ti;
            sum_u  += ui;
            sum_dg += hi / (hi + G);
        }
        const double gres = sum_u * inv_n;
        const double dg   = -(sum_dg * inv_n);
        if (!(dg < 0.0)) break;                 // degenerate; leave Td as is
        const double step = -gres / dg;         // = <u>/<h/(h+G)>
        Td += step;
        if (!(Td > 0.0)) Td = 0.0;
        if (std::fabs(step) <= tol) { converged = true; break; }
    }
    if (iters_out) *iters_out = converged ? (it + 1) : -1;
    return std::isfinite(Td) ? Td : -1.0;
}

}  // namespace

double eqSeedLumpedAnchor(const double* F, std::size_t stride, int n,
                          double Qb, double eps, double sigma, double G,
                          double* T_warm_out, double* T_cold_out,
                          int* iters_out) {
    static thread_local std::vector<double> u;
    return lumpedAnchorImpl(F, stride, n, Qb, eps, sigma, G,
                            T_warm_out, T_cold_out, iters_out,
                            u, /*warm_start=*/false, /*Td0=*/0.0);
}

// ===========================================================================
// The closure: pick G by variance match, then re-solve T_d, to a fixed point.
// Returns the anchor temperature and fills the diagnostics in `r`. On any
// fallback it returns `T_warm` and records WHY in r.anchor_status.
// ===========================================================================
namespace {

/// (G, T_d) property passes. Gamma_eff and h = 4*eps*sigma*T_d^3 both depend on
/// T_d, so the closure is a fixed point. It is iterated to a TOLERANCE, not to
/// a fixed count -- `kAnchorPropertyMax` is the refusal limit, not the plan.
/// The reference implementation converged in 4; measured here, 3-4 passes reach
/// 1e-7 K on every lunar and asteroid face.
constexpr int    kAnchorPropertyMax   = 12;
constexpr double kAnchorPropertyTolK  = 1e-7;

/// The surface-temperature power spectrum the Kirchhoff depth profile needs to
/// decay W(z), carried OUT of the anchor rather than recomputed. Recomputing it
/// would mean a second transform and, worse, a second chance to disagree with
/// the quadrature the anchor actually closed on.
struct KirchhoffWave {
    std::vector<double> weight;    ///< Q_n / sum(Q), index = harmonic n, n >= 1
    double omega1 = 0.0;           ///< fundamental; harmonic n decays at sqrt(n)
    bool   valid  = false;
};

double anchorSolve(const EqSeedInput& in, double T_warm_fallback,
                   EqSeedResult& r, KirchhoffWave* wave) {
    const EqSeedForcing& fc = *in.forcing;
    r.n_samples = fc.n_samples;
    r.span_s    = fc.span_s;
    r.cycle_s   = fc.cycle_s;

    const double a = in.emissivity * in.sigma;
    double Qb = in.Qb;
    if (!(Qb >= 0.0)) Qb = 0.0;

    if (fc.flux == nullptr || fc.n_samples <= 0 || !(a > 0.0)) {
        r.anchor_status = EqSeedAnchorStatus::NoForcingSeries;
        return T_warm_fallback;
    }
    if (fc.radiation_coupled) {
        r.anchor_status = EqSeedAnchorStatus::FallbackRadiationCoupled;
        return T_warm_fallback;
    }
    if (fc.latent_heat) {
        r.anchor_status = EqSeedAnchorStatus::FallbackLatentHeat;
        return T_warm_fallback;
    }
    if (fc.nonuniform_samples) {
        r.anchor_status = EqSeedAnchorStatus::FallbackNonUniformSamples;
        return T_warm_fallback;
    }

    // --- scan the series: swing, lit-sample count, finiteness, brackets -----
    // The two brackets are computed HERE, before any gate, so they are on the
    // series' own quadrature and are reported even on a fallback. Acceptance
    // scaling needs T_w, T_c and the anchor on the SAME quadrature, and a
    // fallback that reported T_c == T_w would silently make the bracket look
    // like zero on exactly the faces worth auditing.
    const std::size_t st = (fc.stride > 0) ? fc.stride : 1;
    double fmin = fc.flux[0], fmax = fc.flux[0];
    int n_lit = 0;
    bool bad = false;
    double sumF = 0.0, sumTc = 0.0;
    for (int i = 0; i < fc.n_samples; ++i) {
        const double v = fc.flux[static_cast<std::size_t>(i) * st];
        if (!std::isfinite(v)) { bad = true; break; }
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
        if (v > 0.0) ++n_lit;
        const double vp = (v > 0.0) ? v : 0.0;
        sumF  += vp;
        sumTc += std::sqrt(std::sqrt((vp + Qb) / a));
    }
    r.n_lit_samples = n_lit;
    if (bad) {
        r.anchor_status = EqSeedAnchorStatus::FallbackBadProperties;
        return T_warm_fallback;
    }
    {
        const double inv_n = 1.0 / static_cast<double>(fc.n_samples);
        r.T_warm = std::sqrt(std::sqrt((sumF * inv_n + Qb) / a));
        r.T_cold = sumTc * inv_n;
    }

    // ZERO SWING is exact, and it is held exactly: a flat F makes the cold and
    // warm brackets coincide, so the anchor has nothing to add and the seed
    // must stay bit-for-bit what it was. Bypassing here (rather than trusting
    // the Newton to land on T_warm to the last ulp) is what makes that a
    // guarantee instead of a measurement.
    if (fmin == fmax) {
        r.anchor_status = EqSeedAnchorStatus::ZeroSwing;
        return T_warm_fallback;
    }

    // --- INPUT-LIMITED: too few lit samples to know the forcing shape -------
    // A face the Sun grazes for 1-3 samples has no resolved shape at all, so
    // the variance the closure would match is quadrature noise. That is a
    // QUADRATURE limit on the accuracy claim, and it is disclosed below as
    // LumpedInputLimited -- but it is NOT by itself a reason to substitute
    // T_w, which is measurably worse on that population. n_lit == 0 is not
    // this case at all: an unlit face is flat-ish and the anchor is exact.
    const bool input_limited =
        (n_lit > 0 && n_lit < fc.input_limited_lit_samples);
    if (n_lit > 0 && n_lit < fc.min_lit_samples) {
        r.anchor_status = EqSeedAnchorStatus::FallbackFewLitSamples;
        return T_warm_fallback;
    }

    // --- the span has to be a cycle mean ------------------------------------
    if (!(fc.span_s > 0.0)) {
        r.anchor_status = EqSeedAnchorStatus::FallbackShortSpan;
        return T_warm_fallback;
    }
    if (fc.cycle_s > 0.0 && fc.span_s < 0.9 * fc.cycle_s) {
        r.anchor_status = EqSeedAnchorStatus::FallbackShortSpan;
        return T_warm_fallback;
    }
    const double omega1 = 2.0 * M_PI / fc.span_s;   // bin spacing = fundamental

    // --- forcing spectrum ---------------------------------------------------
    static thread_local std::vector<double> P;
    eqSeedOneSidedPower(fc.flux, st, fc.n_samples, P);
    const int nb = static_cast<int>(P.size());
    double S0 = 0.0;
    for (int k = 1; k < nb; ++k) S0 += P[k];
    if (!(S0 > 0.0)) {                              // no resolved variance
        r.anchor_status = EqSeedAnchorStatus::ZeroSwing;
        return T_warm_fallback;
    }

    // --- TWO-TIMESCALE detection ---------------------------------------------
    // Bin n is the physical frequency n*omega1. If the span covers N_cyc
    // diurnal cycles, the diurnal fundamental IS bin N_cyc, and every bin below
    // it is a LONGER period than a day -- seasonal. A single lumped G is being
    // asked to cover two regimes at once there, because the describing function
    // matches ONE variance.
    //
    // DETECTED ALWAYS, REFUSED ONLY ON REQUEST. Refusing means substituting
    // T_w, and T_w is the estimator with the +45 K Jensen bias: measured on the
    // 47 non-dark faces this fires on in the shipped 80N crater run, refusing
    // costs 70.46 K median / 83.94 K max against a 384-lunation reference where
    // the anchor costs 3.67 / 13.42. See
    // thermal.seed_anchor_refuse_two_timescale.
    bool two_timescale = false;
    if (fc.cycle_s > 0.0) {
        const int n_diurnal =
            static_cast<int>(std::lround(fc.span_s / fc.cycle_s));
        if (n_diurnal >= 2) {
            double sub = 0.0;
            for (int k = 1; k < std::min(n_diurnal, nb); ++k) sub += P[k];
            r.subdiurnal_variance_fraction = sub / S0;
            two_timescale = r.subdiurnal_variance_fraction >
                            fc.max_subdiurnal_variance_fraction;
            if (two_timescale && fc.refuse_two_timescale) {
                r.anchor_status = EqSeedAnchorStatus::FallbackTwoTimescale;
                return T_warm_fallback;
            }
        }
    }

    // --- the closure --------------------------------------------------------
    static thread_local std::vector<double> u_ws;
    // Start from the SERIES' own warm bracket, not the caller's mean-flux one:
    // the two are the same number whenever the caller averaged the same series,
    // and where they are not, the closure has to be self-consistent with the
    // quadrature it is about to analyze.
    double Td = r.T_warm;
    double G = 0.0, gam = 0.0;
    int iters = -1, pass = 0;
    bool closed = false;
    // The plain substitution Td <- Phi(Td) converges only LINEARLY (9 passes to
    // 1e-7 K on a lit lunar face). A scalar secant on the residual
    // Phi(Td) - Td closes it in 3-4, and is safeguarded back to plain
    // substitution whenever it would leave the physical bracket.
    double Td_prev = 0.0, res_prev = 0.0;
    bool have_prev = false;
    for (pass = 0; pass < kAnchorPropertyMax; ++pass) {
        gam = eqSeedGammaEff(in, fc, omega1, Td);
        if (!(gam >= 0.0)) {
            r.anchor_status = EqSeedAnchorStatus::FallbackBadProperties;
            return T_warm_fallback;
        }
        const double h = 4.0 * a * Td * Td * Td;
        double D = 0.0;
        for (int k = 1; k < nb; ++k) {
            if (P[k] <= 0.0) continue;
            const double A = gam * std::sqrt(static_cast<double>(k) * omega1);
            const double den = h * h + std::sqrt(2.0) * h * A + A * A;
            if (den > 0.0) D += P[k] / den;
        }
        G = (D > 0.0) ? (std::sqrt(S0 / D) - h) : 0.0;
        if (!(G > 0.0) || !std::isfinite(G)) G = 0.0;

        double Tw = 0.0, Tc = 0.0;
        const double Phi = lumpedAnchorImpl(fc.flux, st, fc.n_samples, Qb,
                                            in.emissivity, in.sigma, G,
                                            &Tw, &Tc, &iters,
                                            u_ws, /*warm_start=*/pass > 0, Td);
        r.T_warm = Tw;
        r.T_cold = Tc;
        if (!(Phi >= 0.0) || iters < 0) {
            r.anchor_status = EqSeedAnchorStatus::FallbackNotConverged;
            return T_warm_fallback;
        }
        const double res = Phi - Td;
        if (std::fabs(res) <= kAnchorPropertyTolK) {
            Td = Phi;
            closed = true;
            break;
        }
        double Td_next = Phi;                        // plain substitution
        if (have_prev) {
            const double dres = res - res_prev;
            if (dres != 0.0) {
                const double s = Td - res * (Td - Td_prev) / dres;
                if (std::isfinite(s) && s >= Tc && s <= Tw) Td_next = s;
            }
        }
        Td_prev  = Td;
        res_prev = res;
        have_prev = true;
        Td = Td_next;
    }
    if (!closed) {
        r.anchor_status = EqSeedAnchorStatus::FallbackNotConverged;
        return T_warm_fallback;
    }
    r.G = G;
    r.gamma_eff = gam;
    r.anchor_iterations = pass + 1;

    // The truth is bracketed by the two limits BY CONSTRUCTION (G >= 0 finite),
    // so an answer outside the bracket means the closure produced something the
    // physics forbids. Refuse it rather than seed it.
    const double lo = std::min(r.T_cold, r.T_warm) - 1e-6;
    const double hi = std::max(r.T_cold, r.T_warm) + 1e-6;
    if (!(Td >= lo && Td <= hi) || !std::isfinite(Td)) {
        r.anchor_status = EqSeedAnchorStatus::FallbackNotConverged;
        return T_warm_fallback;
    }
    // --- the surface temperature spectrum, for the Kirchhoff depth profile ---
    // Q_n = P_n / |h + Gamma sqrt(n omega) e^(i pi/4)|^2 is the half-space
    // temperature response to this face's own forcing -- the same denominator
    // the variance match above already forms, evaluated once more at the
    // CONVERGED T_d so the profile and the anchor describe one column.
    if (wave != nullptr) {
        const double h_f = 4.0 * a * Td * Td * Td;
        wave->weight.assign(static_cast<std::size_t>(nb), 0.0);
        double sum = 0.0;
        for (int k = 1; k < nb; ++k) {
            if (P[k] <= 0.0) continue;
            const double A   = gam * std::sqrt(static_cast<double>(k) * omega1);
            const double den = h_f * h_f + std::sqrt(2.0) * h_f * A + A * A;
            if (!(den > 0.0)) continue;
            const double q = P[k] / den;
            wave->weight[static_cast<std::size_t>(k)] = q;
            sum += q;
        }
        if (sum > 0.0 && std::isfinite(sum)) {
            const double inv = 1.0 / sum;
            for (double& w : wave->weight) w *= inv;
            wave->omega1 = omega1;
            wave->valid  = true;
        }
    }

    r.anchor_status = two_timescale ? EqSeedAnchorStatus::LumpedTwoTimescale
                    : input_limited ? EqSeedAnchorStatus::LumpedInputLimited
                                    : EqSeedAnchorStatus::Lumped;
    return Td;
}

}  // namespace

EqSeedResult eqSeedColumn(const EqSeedInput& in, double* T_out) {
    EqSeedResult r;

    // --- surface anchor: eps*sigma*T0^4 = <F_abs> + Qb ----------------------
    double F  = in.mean_absorbed_flux;
    double Qb = in.Qb;
    if (!(F  >= 0.0)) F  = 0.0;   // also catches NaN
    if (!(Qb >= 0.0)) Qb = 0.0;

    const double denom = in.emissivity * in.sigma;
    double T0 = (denom > 0.0) ? std::sqrt(std::sqrt((F + Qb) / denom)) : 0.0;
    r.T_warm   = T0;
    r.T_cold   = T0;
    r.T_anchor = T0;

    // --- lit-face seed anchor -----------------------------------------------
    // The line above is the WARM BRACKET, exact for <T^4> and biased +45 K in
    // the mean on a lit lunar face. Where the caller supplies the face's own
    // forcing series, replace it with the derived lumped-substrate anchor.
    static thread_local KirchhoffWave wave;
    wave.valid = false;
    if (in.forcing != nullptr) {
        if (in.anchor == EqSeedAnchor::Warm) {
            r.anchor_status = EqSeedAnchorStatus::WarmRequested;
            r.n_samples     = in.forcing->n_samples;
            r.span_s        = in.forcing->span_s;
            r.cycle_s       = in.forcing->cycle_s;
        } else {
            T0 = anchorSolve(in, T0, r, &wave);
            r.T_anchor = T0;
        }
    } else {
        r.anchor_status = EqSeedAnchorStatus::NoForcingSeries;
    }

    // Lower guard only, and it is physics, not tuning: cp(T) is a polynomial
    // fit (Moon cp0 = -3.6125, Europa cp = 7.49*T with cp0 = 0), so a 0 K node
    // gives cp <= 0 and an infinite or negative diffusivity. Every other
    // temperature site in the solver already floors at ABSOLUTE_ZERO_K; the
    // seed did not. Reachable: a Qb = 0 body (rockox_generic-test.yaml) with a
    // face whose mean absorbed flux is exactly 0.
    if (!(T0 > ABSOLUTE_ZERO_K)) {
        T0 = ABSOLUTE_ZERO_K;
        r.clamped_low = true;
    }
    // NO upper clamp. The old seeds clamped at 400 K; Mercury's subsolar
    // CYCLE MEAN is ~468 K and its peak is 633 K, so that clamp was a silent
    // 68-233 K substitution on a body the engine lets you select. A radiative
    // equilibrium temperature is physical by construction, and as of
    // 2026-08-26 no solver path bounds it from above either, so there is
    // nothing left for the seed to warn about.

    r.T_surface = T0;
    r.T_deep    = T0;

    if (in.n <= 0 || in.z == nullptr) {
        if (T_out && in.n > 0) {
            for (int i = 0; i < in.n; ++i) T_out[i] = T0;
        }
        return r;
    }
    if (T_out) T_out[0] = T0;

    // --- Kirchhoff (fourth-moment) depth profile ----------------------------
    // Runs ONLY where the derived anchor ran: while the anchor is the warm
    // bracket the correction below is identically zero (see the header note),
    // so gating on the anchor is what makes the chi == 0 and zero-swing
    // reductions exact by control flow rather than by arithmetic. Hayne only --
    // Martinez2021 does not separate as k_c(z)*g(T), so the potential G does
    // not exist for it.
    const double R4 = 0.25 * in.R350;
    // Note there is NO Qb > 0 condition. With chi > 0 and a diurnal swing the
    // zero-basal-flux column is NOT isothermal: d<G>/dz = 0 makes <G> constant,
    // so as the wave dies <T^4> falls toward <T>^4 and <T> must RISE to hold <G>
    // fixed. That is the rectification of the wave by the temperature-dependent
    // conductivity, and suppressing it would be a silent substitution on every
    // Qb = 0 body. The docstring's "Qb <= 0 gives an isothermal column" is only
    // true at chi == 0.
    if (R4 > 0.0 && in.cond == ConductivityModel::Hayne2017 &&
        eqSeedAnchorRan(r.anchor_status) && wave.valid && in.kc != nullptr &&
        in.rho != nullptr && in.forcing != nullptr) {
        // W(0) = <T_0^4> - <T_0>^4, both moments on the anchor's own quadrature.
        // The bracket check in anchorSolve guarantees T_anchor <= T_warm, so
        // this is non-negative; the guard is for the degenerate equal case.
        const double Tw4 = r.T_warm * r.T_warm * r.T_warm * r.T_warm;
        const double Ta4 = T0 * T0 * T0 * T0;
        const double W0  = (Tw4 > Ta4) ? (Tw4 - Ta4) : 0.0;
        const double dG0 = R4 * W0;              // G-space size of the correction
        r.kirchhoff_surface_offset = dG0;

        const double Gsurf = eqSeedKirchhoffG(T0, R4);
        const int    nb    = static_cast<int>(wave.weight.size());
        const double omega = wave.omega1;

        double Rk   = 0.0;   // INT dz/k_c  -- k_c, not k(T)
        double zeta = 0.0;   // INT dz/delta
        double T    = T0;
        bool   stalled = false;
        for (int i = 1; i < in.n; ++i) {
            const double dz   = in.z[i] - in.z[i - 1];
            const double kc_i = in.kc[i - 1];
            if (!stalled && kc_i > 0.0 && dz > 0.0) {
                Rk += dz / kc_i;
                // zeta accumulates on the LOCAL skin depth of the layered,
                // temperature-dependent medium: delta = sqrt(2k/(rho cp omega)).
                const double k_i   = hayneK(T, kc_i, in.R350);
                const double cp_i  = heatCapacityAt(*in.forcing, T);
                const double rho_i = in.rho[i - 1];
                if (k_i > 0.0 && cp_i > 0.0 && rho_i > 0.0 && omega > 0.0) {
                    const double delta =
                        std::sqrt(2.0 * k_i / (rho_i * cp_i * omega));
                    if (delta > 0.0) zeta += dz / delta;
                }
            } else {
                // A non-positive k_c or dz would give an infinite gradient.
                // Hold the profile below, exactly as the plain branch does.
                stalled = true;
            }
            // d(z): the surface variance propagated harmonic by harmonic.
            // Harmonic n has skin depth delta/sqrt(n), so it attenuates as
            // exp(-sqrt(n)*zeta) in amplitude and exp(-2 sqrt(n) zeta) in power.
            double d = 0.0;
            if (dG0 > 0.0) {
                for (int k = 1; k < nb; ++k) {
                    const double w = wave.weight[static_cast<std::size_t>(k)];
                    if (w <= 0.0) continue;
                    d += w * std::exp(-2.0 * std::sqrt(static_cast<double>(k)) * zeta);
                }
            }
            const double G = Gsurf + Qb * Rk + dG0 * (1.0 - d);
            T = eqSeedKirchhoffGInv(G, R4);
            if (!std::isfinite(T)) { T = T0; stalled = true; }
            if (T_out) T_out[i] = T;
        }
        r.kirchhoff_profile = true;
        r.T_deep = T;
        return r;
    }

    // --- conductive steady gradient: dT/dz = Qb / k(T, z) -------------------
    // Forward integration with k evaluated at the SHALLOWER node, which is the
    // same upwind choice the CN interface conductances make, so the seed is
    // consistent with the stencil that will step it.
    double T = T0;
    for (int i = 1; i < in.n; ++i) {
        const double dz = in.z[i] - in.z[i - 1];
        double k;
        if (in.cond == ConductivityModel::Martinez2021) {
            const double rho_i = in.rho ? in.rho[i - 1] : 0.0;
            k = martinezK(T, rho_i);
        } else {
            const double kc_i = in.kc ? in.kc[i - 1] : 0.0;
            k = hayneK(T, kc_i, in.R350);
        }
        if (k > 0.0 && dz > 0.0) {
            T += Qb * dz / k;
        }
        // k <= 0 (a Martinez density below 1004 kg/m^3, or a zero kc) would
        // give an infinite gradient. Hold the profile instead of producing
        // Inf; martinezDensityIsValid() is the caller's loud guard for that
        // configuration.
        if (T_out) T_out[i] = T;
    }
    r.T_deep = T;
    return r;
}

double analyticSeedCyclesFromPeriods(double planet_day_s, double planet_year_s,
                                     int nyearseq) {
    if (!(planet_day_s > 0.0) || nyearseq <= 0) return 0.0;
    const double raw = static_cast<double>(nyearseq) *
                       std::round(planet_year_s / planet_day_s);
    if (raw >= 1.0) return raw;
    // year < day/2 -> round() gives 0 and the pre-conditioner would be silently
    // switched off. One whole diurnal cycle is the shortest length for which
    // "integrate the analytic clear-sky forcing" means anything, and it is said
    // out loud rather than substituted quietly. Reachable today only for
    // Mercury (2 years = 1 solar day) under thermal.spinup_seed: legacy.
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
        std::cerr << "[eq-seed] NOTE: nyearseq=" << nyearseq << " x round(year/day) = "
                  << raw << " for a body whose solar day (" << (planet_day_s / 86400.0)
                  << " d) is longer than its year (" << (planet_year_s / 86400.0)
                  << " d). The legacy analytic pre-conditioner is run for ONE whole "
                     "diurnal cycle instead of zero; set thermal.analytic_seed_cycles "
                     "explicitly to choose a different length." << std::endl;
    }
    return 1.0;
}

double analyticSeedCycles(EqSeedMode mode, double explicit_cycles,
                          double planet_day_s, double planet_year_s,
                          int nyearseq) {
    if (explicit_cycles >= 0.0) return explicit_cycles;
    if (mode == EqSeedMode::Physical) return 0.0;
    if (!(planet_day_s > 0.0)) return 0.0;
    return analyticSeedCyclesFromPeriods(planet_day_s, planet_year_s, nyearseq);
}

// ---------------------------------------------------------------------------
// EqSeedRecorder
// ---------------------------------------------------------------------------
EqSeedRecorder& EqSeedRecorder::instance() {
    static EqSeedRecorder r;
    return r;
}

EqSeedRecorder::EqSeedRecorder() {
    const char* p = std::getenv("PTM_DUMP_SEED");
    if (p && *p != '\0') {
        enabled_ = true;
        path_    = p;
        std::cerr << "[eq-seed] PTM_DUMP_SEED=" << path_
                  << " -- the per-face seed will be written for cross-path"
                     " comparison." << std::endl;
    }
}

void EqSeedRecorder::reserve(std::size_t n_faces) {
    if (!enabled_) return;
    if (rows_.size() < n_faces) rows_.resize(n_faces);
}

void EqSeedRecorder::record(std::size_t face, const EqSeedInput& in,
                            const EqSeedResult& res) {
    if (!enabled_) return;
    if (face >= rows_.size()) return;   // reserve() was not called; drop, quietly
    Row& r   = rows_[face];
    r.mean_flux = in.mean_absorbed_flux;
    r.Qb        = in.Qb;
    r.T_surf    = res.T_surface;
    r.T_deep    = res.T_deep;
    r.T_lowinertia = in.mean_inst_radeq_T;
    r.T_warm       = res.T_warm;
    r.T_cold       = res.T_cold;
    r.G            = res.G;
    r.gamma_eff    = res.gamma_eff;
    r.n_samples    = res.n_samples;
    r.anchor_status = res.anchor_status;
    r.kirchhoff    = res.kirchhoff_profile;
    r.kirch_offset = res.kirchhoff_surface_offset;
    r.set       = true;
}

void EqSeedRecorder::write(const std::string& tag) const {
    if (!enabled_ || rows_.empty()) return;
    // PTM_DUMP_SEED_ALL=1: keep EVERY seeding pass (a coupled run seeds again in its
    // re-thermal passes, and the default single file keeps only the last one), as
    // <path>.<tag>.<pass>. Default (unset): one file at <path>, exactly as before.
    std::string out = path_;
    if (const char* all = std::getenv("PTM_DUMP_SEED_ALL"); all && all[0] == '1') {
        static int pass = 0;
        out = path_ + "." + tag + "." + std::to_string(pass++);
    }
    std::ofstream fh(out);
    if (!fh) {
        std::cerr << "[eq-seed] WARNING: cannot open '" << out
                  << "' for writing; NO seed dump produced." << std::endl;
        return;
    }
    fh << std::setprecision(17);
    fh << "# PTM3D shared equilibration seed, path=" << tag << "\n";
    fh << "face,mean_absorbed_flux_Wm2,Qb_Wm2,T_surface_K,T_deep_K,"
          "T_lowinertia_K,T_warm_K,T_cold_K,G_Wm2K,gamma_eff,"
          "n_samples,anchor_status,kirchhoff,kirchhoff_offset_K\n";
    std::size_t written = 0;
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        if (!rows_[i].set) continue;
        fh << i << ',' << rows_[i].mean_flux << ',' << rows_[i].Qb << ','
           << rows_[i].T_surf << ',' << rows_[i].T_deep << ','
           << rows_[i].T_lowinertia << ',' << rows_[i].T_warm << ','
           << rows_[i].T_cold << ',' << rows_[i].G << ','
           << rows_[i].gamma_eff << ',' << rows_[i].n_samples << ','
           << eqSeedAnchorStatusName(rows_[i].anchor_status) << ','
           << (rows_[i].kirchhoff ? 1 : 0) << ','
           << rows_[i].kirch_offset << '\n';
        ++written;
    }
    std::cerr << "[eq-seed] wrote " << written << " seed rows to " << out
              << std::endl;
}

}  // namespace thermal
