/*
 * Multi-Timescale Ice Evolution via Operator Splitting
 *
 * Enables simulation of ice redistribution over geological timescales (kyr to Myr)
 * by separating the fast diurnal dynamics from slow concentration evolution.
 *
 * Algorithm:
 * 1. Run one full diurnal cycle to compute time-averaged fluxes
 * 2. Use averaged fluxes to evolve non-ice concentration over many diurnal cycles (one "epoch")
 * 3. Repeat for desired number of epochs
 *
 * Key physics:
 * - Ice availability is continuous: sublimation scales with ice fraction (1 - nonice_concentration),
 *   or, in the exposed-area mode (--ice-fraction-from-albedo=hapke-intimate), with the exposed
 *   ice area fraction f(1 - nonice_concentration); see solveExposedAreaStep()
 * - Self-limiting depletion: as ice is consumed, sublimation rate decreases proportionally
 * - Analytical exponential evolution: d(nonice)/dt = A*(1-nonice) - B is solved exactly,
 *   enabling accurate super-stepping over many epochs without numerical error
 * - Each face reaches equilibrium where reduced sublimation balances incoming deposition
 *
 * Key approximations:
 * - Transport matrix is fixed (computed once at start)
 * - Deposition is constant within a step at the global scale (per source group) whose
 *   deposition equals what the kernel retains from the step's own emission (PTM-644)
 *
 * References:
 * - Operator splitting: Strang (1968), LeVeque (2002)
 */

#ifndef MULTITIMESCALE_EVOLUTION_H
#define MULTITIMESCALE_EVOLUTION_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include "ice_layer_model.h"

namespace ptm {
namespace exosphere {

// Physical constants for timescale calculations
namespace timescale_constants {
    // Europa diurnal period [seconds]
    constexpr double EUROPA_DIURNAL_PERIOD = 3.5512 * 24.0 * 3600.0;  // 3.55 days

    // Seconds per Earth year
    constexpr double SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0;

    // Default diurnal cycles per epoch (~ 1 Earth year)
    constexpr int DEFAULT_CYCLES_PER_EPOCH = 100;

    // Minimum epochs between progress reports (1 = every epoch, updates are cheap)
    constexpr int PROGRESS_INTERVAL = 1;

    // Convergence threshold for non-ice concentration change per epoch
    constexpr double CONVERGENCE_THRESHOLD = 1e-6;  // dimensionless

    // Flux imbalance threshold for mass balance convergence
    constexpr double FLUX_CONVERGENCE_THRESHOLD = 0.01;  // 1% imbalance
}

/**
 * One closed-form step of the exposed-area ice ODE
 * (--ice-fraction-from-albedo=hapke-intimate).
 *
 * With ice volume fraction phi = 1 - nonice_concentration and exposed ice area
 * fraction f(phi) = phi / (phi + r (1 - phi)), r = D_ice / D_nonice:
 *   dphi/dt = B - A f(phi)
 * A = base emission / max ice mass, B = deposition / max ice mass [1/s].
 * Separable: dt = (r - (r-1) phi) dphi / (E - D phi), D = B (r - 1) + A, E = B r,
 * integrated in closed form (see t_of below; finite for any sign of D, so r < 1
 * is covered). The interior equilibrium is phi_eq = E / D (A f(phi_eq) = B) when
 * D > E; otherwise phi rises to 1. phi_end is found by bisection on the
 * monotone t(phi). For r = 1 this reduces to the legacy
 * exponential solution. phi is clamped at 1 (deposition overflow is ignored,
 * as in the legacy model).
 *
 * mean_f is the time-mean exposed fraction, so that base_emission * mean_f is
 * the mean emission over the step: int f dt, from whichever of the exact balance
 * (B t - dphi)/A or the closed-form int phi/(E - D phi) dphi is well conditioned
 * (the trapezoid rule when phi moves by < 1e-8).
 */
struct ExposedAreaStep {
    double phi_end = 0.0;
    double mean_f = 0.0;
};

inline ExposedAreaStep solveExposedAreaStep(double phi0, double A, double B,
                                            double r, double dt) {
    ExposedAreaStep out;
    phi0 = std::clamp(phi0, 0.0, 1.0);
    auto f_of = [r](double p) {
        const double den = p + r * (1.0 - p);
        return (den > 0.0) ? p / den : 0.0;
    };
    const double f0 = f_of(phi0);
    if (dt <= 0.0) { out.phi_end = phi0; out.mean_f = f0; return out; }
    if (A <= 1e-30) {
        // No emission: pure deposition (linear), as in the legacy branch.
        out.phi_end = std::min(1.0, phi0 + B * dt);
        out.mean_f = 0.5 * (f0 + f_of(out.phi_end));
        return out;
    }

    const double rate0 = B - A * f0;           // dphi/dt at phi0
    if (rate0 == 0.0) { out.phi_end = phi0; out.mean_f = f0; return out; }

    // t(phi) = int_{phi0}^{phi} (r - (r-1)p) / (E - D p) dp,  D = B (r-1) + A,  E = B r.
    // Written with s = p - phi0, eps = D / g0 (g0 = E - D phi0, same sign as rate0) as
    //   t = [n0 L(s) - (r-1) M(s)] / g0,   n0 = r - (r-1) phi0,
    //   L = int ds/(1 - eps s),  M = int s ds/(1 - eps s),
    // which stays finite and accurate for any sign of D, including D -> 0 (r < 1).
    const double D = B * (r - 1.0) + A;
    const double E = B * r;
    const double g0 = E - D * phi0;
    const double n0 = r - (r - 1.0) * phi0;
    const double eps = D / g0;
    // L(s) = int_0^s du/(1 - eps u), M(s) = int_0^s u du/(1 - eps u).
    auto LM = [&](double sd, double& L, double& M) {
        const double y = eps * sd;             // < 1 along the trajectory
        if (std::abs(y) < 1e-3) {
            L = sd * (1.0 + y / 2.0 + y * y / 3.0 + y * y * y / 4.0);
            M = sd * sd * (0.5 + y / 3.0 + y * y / 4.0 + y * y * y / 5.0);
        } else {
            L = -std::log1p(-y) / eps;
            M = (L - sd) / eps;
        }
    };
    auto t_of = [&](double p) {
        double L, M;
        LM(p - phi0, L, M);
        return (n0 * L - (r - 1.0) * M) / g0;
    };
    // Exposure integral: f dt = f (dt/dphi) dphi = phi / (E - D phi) dphi, so
    //   int_0^t f dt' = (phi0 L(s) + M(s)) / g0.
    auto f_integral = [&](double p) {
        double L, M;
        LM(p - phi0, L, M);
        return (phi0 * L + M) / g0;
    };
    // Interior equilibrium phi_eq = E / D (A f(phi_eq) = B) exists only for D > E.
    const bool interior_eq = (D > 0.0) && (E < D);

    double lo, hi;
    bool clamped_at_one = false;
    double t_one = 0.0;
    if (rate0 > 0.0) {
        // phi increases toward phi_eq, or toward 1 when there is no interior equilibrium.
        lo = phi0;
        if (interior_eq) {
            hi = E / D;
        } else {
            hi = 1.0;
            if (E - D > 0.0) {                 // dphi/dt > 0 at phi = 1: reached in finite time
                t_one = t_of(1.0);
                if (dt >= t_one) clamped_at_one = true;
            }
        }
    } else {
        // phi decreases toward phi_eq (rate0 < 0 implies D > E / phi0 > 0).
        lo = std::max(0.0, E / D); hi = phi0;
    }

    // int f dt over [0, t] to phi: two exact, complementary forms.
    //   balance: (B t - (phi - phi0)) / A   -- cancels when A f << B (deposition-
    //            dominated growth, far from equilibrium);
    //   direct:  f_integral(phi)            -- log-singular at phi_eq.
    // Use the balance unless it loses more than 3 digits to cancellation; in that
    // regime phi is far from phi_eq and the direct form is well conditioned.
    auto exposure = [&](double p, double t) {
        const double dphi = p - phi0;
        const double bal = B * t - dphi;
        if (std::abs(bal) > 1e-3 * std::max(std::abs(B * t), std::abs(dphi))) return bal / A;
        return f_integral(p);
    };

    if (clamped_at_one) {
        out.phi_end = 1.0;
        // Interior segment [0, t_one], then f = 1 for the rest of the step.
        out.mean_f = std::clamp((exposure(1.0, t_one) + (dt - t_one)) / dt, f0, 1.0);
        return out;
    }

    for (int it = 0; it < 200 && hi - lo > 1e-16; ++it) {
        const double m = 0.5 * (lo + hi);
        const bool before = t_of(m) < dt;      // m is reached before dt
        if (rate0 > 0.0) { if (before) lo = m; else hi = m; }
        else             { if (before) hi = m; else lo = m; }
    }
    out.phi_end = 0.5 * (lo + hi);

    const double f1 = f_of(out.phi_end);
    const double f_lo = std::min(f0, f1), f_hi = std::max(f0, f1);
    // When phi barely moves, its bisection-resolved change (~1e-16) cannot carry
    // the integral; f is then constant to < r * 1e-8 and the trapezoid is exact
    // to that order.
    if (std::abs(out.phi_end - phi0) < 1e-8) {
        out.mean_f = 0.5 * (f0 + f1);
    } else {
        out.mean_f = std::clamp(exposure(out.phi_end, dt) / dt, f_lo, f_hi);
    }
    return out;
}

/**
 * One step of the lag-barrier face model (IceLayerModel::enableLagBarrier), with
 * deposition B held constant over the step (PTM-644 solves for B).
 *
 * Inputs [SI]: Js, Jp = pure-ice potential sublimation and gross sputter fluxes,
 * B = deposition [kg/(m² s)]; d0 [m], mf0 (frost veneer), M0 (substrate ice) [kg/m²];
 * f = exposed ice area fraction of the mixture; beta [m³/kg]; ell = l [m].
 *
 * Regimes (a = Js f, p = Jp f, u = 1 + d/l, E_S = a/u + p):
 *  V  veneer (m_f > 0): emission Js + Jp (unbarriered, whole face), dm_f/dt = B - Js - Jp.
 *  S  substrate (m_f = 0, B <= E_S): emission a/u (sublimation) + p (sputter); the net
 *     loss E_S - B comes out of the substrate and thickens the lag,
 *       dd/dt = beta (a/u + p - B).
 *     With q = p - B, g0 = a + q u0, eps = q/g0, s = u - u0 this separates to
 *       t(s) = l / (beta g0) [u0 L(s) + M(s)],  L = int_0^s dx/(1+eps x), M = int_0^s x dx/(1+eps x),
 *     and the sublimation emitted is int a/u dt = l a / (beta g0) L(s). For B = p = 0 this
 *     is d + d^2/(2l) = beta a t. The equilibrium u* = a/(B - p) (Fermi's
 *     d* = l (Js f / B - 1) for p = 0) is approached, never crossed.
 *  sliding (m_f = 0, E_S < B < Js + Jp): frost that lands sublimates again as fast as
 *     it arrives (Filippov: the face is frost-free a fraction lambda = (Jv - B)/(Jv - E_S)
 *     of the time); emission = B, d, m_f and the substrate are constant.
 *  V also when B >= Js + Jp at m_f = 0: the veneer grows.
 * The step is integrated exactly (no hold-d approximation): at most V -> (S | sliding),
 * and S -> depleted (substrate exhausted) -> (sliding | V).
 * sub_int, sput_int are the emitted masses over the step [kg/m²]; the substrate
 * change is booked from them, so the per-face inventory closes exactly.
 */
struct LagFaceStep {
    double d_end = 0.0, mf_end = 0.0, M_end = 0.0;
    double sub_int = 0.0, sput_int = 0.0;   // [kg/m²] over the step
    int regime_end = 0;                     // 0 = S, 1 = sliding, 2 = veneer, 3 = no ice and no deposition
};

inline void lagLM(double eps, double s, double& L, double& M) {
    const double y = eps * s;
    if (std::abs(y) < 1e-4) {
        L = s * (1.0 - y / 2.0 + y * y / 3.0 - y * y * y / 4.0);
        M = s * s * (0.5 - y / 3.0 + y * y / 4.0 - y * y * y / 5.0);
    } else {
        L = std::log1p(y) / eps;
        M = (s - L) / eps;
    }
}

inline LagFaceStep solveLagFaceStep(double Js, double Jp, double B, double d0, double mf0,
                                    double M0, double f, double beta, double ell, double dt) {
    LagFaceStep out;
    double d = d0, mf = std::max(0.0, mf0), M = std::max(0.0, M0), t_left = std::max(0.0, dt);
    const double Jv = Js + Jp;
    double sub = 0.0, sput = 0.0;
    int regime = 0;
    // Veneer phase.
    if (mf > 0.0 && t_left > 0.0) {
        const double rate = B - Jv;
        const double tv = (rate < 0.0) ? mf / (-rate) : std::numeric_limits<double>::infinity();
        const double tau = std::min(tv, t_left);
        sub += Js * tau;
        sput += Jp * tau;
        mf = (tau == tv) ? 0.0 : std::max(0.0, mf + rate * tau);
        t_left -= tau;
        regime = 2;
    }
    for (int phase = 0; phase < 3 && t_left > 0.0; ++phase) {
        const bool has = M > 0.0;
        const double a = has ? Js * f : 0.0, p = has ? Jp * f : 0.0;
        const double u0 = 1.0 + d / ell;
        const double ES = a / u0 + p;
        if (has && ES > 0.0 && B <= ES) {
            regime = 0;
            const double net = ES - B;                 // >= 0
            const double g0 = u0 * net;                // a + (p - B) u0
            if (!(beta > 0.0) || !(g0 > 0.0)) {
                // No lag can form (x_n = 0) or exact equilibrium: constant rates.
                const double t_dep = (net > 0.0) ? M / net : std::numeric_limits<double>::infinity();
                const double tau = std::min(t_dep, t_left);
                sub += (a / u0) * tau;
                sput += p * tau;
                M = (tau == t_dep) ? 0.0 : std::max(0.0, M - net * tau);
                if (beta > 0.0) d += beta * net * tau;
                t_left -= tau;
                continue;
            }
            const double q = p - B;
            const double eps = q / g0;
            const double k = ell / (beta * g0);        // t = k [u0 L + M]
            auto t_of = [&](double sv) { double L, Mm; lagLM(eps, sv, L, Mm); return k * (u0 * L + Mm); };
            // Substrate exhausted at s_dep (net loss l s / beta = M).
            const double s_dep = beta * M / ell;
            const double t_dep = t_of(s_dep);
            double s_end;
            double tau;
            if (t_dep <= t_left) {
                s_end = s_dep;
                tau = t_dep;
            } else {
                // Root of t(s) = t_left on (0, s_dep); t is increasing, convex-free; safeguarded Newton.
                double lo = 0.0, hi = s_dep;
                if (eps < 0.0) hi = std::min(hi, -1.0 / eps);   // equilibrium, t -> infinity
                const double rate0 = beta * net / ell;          // ds/dt at s = 0
                double sv = std::min(0.5 * (lo + hi), rate0 * t_left);
                for (int it = 0; it < 200; ++it) {
                    const double tv = t_of(sv);
                    if (tv < t_left) lo = sv; else hi = sv;
                    const double dtds = k * (u0 + sv) / (1.0 + eps * sv);
                    double sn = sv - (tv - t_left) / dtds;
                    if (!(sn > lo && sn < hi)) sn = 0.5 * (lo + hi);
                    if (std::abs(sn - sv) <= 1e-15 * std::max(sv, 1e-300) || hi - lo <= 1e-15 * hi) { sv = sn; break; }
                    sv = sn;
                }
                s_end = sv;
                tau = t_left;
            }
            // Sublimation emitted over the segment, two exact forms:
            //   balance: l s / beta - (p - B) tau   (the lag grew by beta x the net loss);
            //   direct:  l a / (beta g0) L(s)       (log-singular at the equilibrium u*).
            // Use the balance unless it loses more than 3 digits to cancellation (then
            // the face is far from u* and the direct form is well conditioned).
            const double grown = ell * s_end / beta, other = (B - p) * tau;
            double sub_seg = grown + other;
            if (!(std::abs(sub_seg) > 1e-3 * std::max(std::abs(grown), std::abs(other)))) {
                double L, Mm;
                lagLM(eps, s_end, L, Mm);
                sub_seg = ell * a / (beta * g0) * L;
            }
            sub += sub_seg;
            sput += p * tau;
            const double loss = sub_seg + p * tau - B * tau;   // booked net loss, >= 0
            M = (tau == t_dep) ? 0.0 : std::max(0.0, M - loss);
            d += ell * s_end;
            t_left -= tau;
            continue;
        }
        if (B < Jv) {
            // Sliding (or no ice and B < Jv): emission = B, split by the frost-free fraction.
            const double lam = (Jv - B) / (Jv - ES);
            sub += (lam * (a / u0) + (1.0 - lam) * Js) * t_left;
            sput += (lam * p + (1.0 - lam) * Jp) * t_left;
            regime = (B > 0.0 || ES > 0.0) ? 1 : 3;
            t_left = 0.0;
        } else {
            // Deposition at or above the free-frost loss: the veneer grows.
            sub += Js * t_left;
            sput += Jp * t_left;
            mf += (B - Jv) * t_left;
            regime = (B > 0.0) ? 2 : 3;
            t_left = 0.0;
        }
    }
    out.d_end = d;
    out.mf_end = mf;
    out.M_end = M;
    out.sub_int = sub;
    out.sput_int = sput;
    out.regime_end = regime;
    return out;
}

/**
 * Garden mode (PTM-669): one step of the lag face model with the sputter split and
 * micrometeoroid gardening, deposition B held constant (Fermi's formulation,
 * Paul's rulings 2026-09-24).
 *
 * Inputs: Js sublimation, Jw H2O-ejection, Jr radiolytic (O2 + H2, = 36/32 x O2)
 * pure-ice potentials [kg/(m² s)]; shield = false leaves the H2O ejection unbarriered
 * (with the radiolytic part); B deposition; d0 [m], mf0 (veneer), M0 (box ice) [kg/m²];
 * f exposed ice fraction of the box; beta = beta_c of the CURRENT box c [m³/kg];
 * ell = l [m]; vm = gardening rate v_m [m/s]; gv = rho_ice (1 - phi) v_m, the veneer
 * gardening rate [kg/(m² s)]; dt [s].
 *
 * With a = Js f, pw = Jw f, pr = Jr f, u = 1 + d/l, E_S = (a + pw)/u + pr:
 *  V   (m_f > 0): emission Js + Jw + Jr, dm_f/dt = B - Jv - gv; the gardened frost
 *      (gv) goes into the box ice; the lag under the veneer is untouched.
 *  S   (m_f = 0, B <= E_S): dd/dt = beta [(a + pw)/u + pr - B] - v_m 1(d > 0), the
 *      box loses E_S - B. This is the PTM-659 closed form with a -> a + pw and
 *      p -> pr - v_m/beta. d = 0 is absorbing when beta (a + pw + pr - B) <= v_m
 *      (Spencer criterion): the box then loses ice with no lag and its c rises.
 *  sliding (m_f = 0, E_S < B < Jv + gv): the face is frost-free a fraction
 *      lambda = (Jv + gv - B)/(Jv + gv - E_S) of the time (Filippov); emission
 *      lambda E_S + (1 - lambda) Jv, the box gains (1 - lambda) gv, and the lag is
 *      gardened at lambda v_m. lambda is held at its segment-start value (the lag
 *      moves by < 10% of l + d per step under the super-step bound); mass is exact
 *      for any lambda.
 *  V   also when B >= Jv + gv at m_f = 0 (the veneer grows).
 * Box non-ice: the caller moves (1 - phi)(d_end - d0) of non-ice volume from the box
 * to the lag (negative when the lag is gardened back), which keeps c fixed while
 * d > 0 and leaves the freed non-ice in the box while d = 0.
 */
/** Garden-mode mass audit residual (kept out of line; see EvolutionStatistics::print). */
__attribute__((noinline)) inline double gardenAuditResidual(double inv, double inv0, double net, double entrained) {
    return (inv - inv0) + net - entrained;
}

struct GardenFaceStep {
    double d_end = 0.0, mf_end = 0.0, M_end = 0.0;
    double sub_int = 0.0, sputw_int = 0.0, sputr_int = 0.0;   // emitted [kg/m²]
    double box_gain = 0.0;                                     // gardened veneer frost into the box [kg/m²]
    int regime_end = 0;   // 0 substrate (lag or absorbing), 1 sliding, 2 veneer, 3 none
    bool absorbing = false;  // ended in the Spencer-absorbing d = 0 state
};

inline GardenFaceStep solveGardenFaceStep(double Js, double Jw, double Jr, double B, double d0, double mf0,
                                          double M0, double f, double beta, double ell, double vm, double gv,
                                          double dt, bool shield = true) {
    GardenFaceStep out;
    double d = std::max(0.0, d0), mf = std::max(0.0, mf0), M = std::max(0.0, M0), t_left = std::max(0.0, dt);
    const double Jv = Js + Jw + Jr, Vt = Jv + gv;
    const double inf = std::numeric_limits<double>::infinity();
    // Set when a recycling segment ends where E_S = B: the next segment is the
    // substrate regime at L = 0 (rounding must not send it back to recycling).
    bool force_S = false;
    // Once the box runs dry within a step it stays dry for the rest of the step: frost
    // gardened in meanwhile is held for the next step (otherwise the step would chatter
    // between re-sublimating each speck of gardened frost and recycling).
    bool dry = false;
    for (int phase = 0; phase < 16 && t_left > 0.0; ++phase) {
        if (mf > 0.0) {
            const double rate = B - Vt;
            const double tv = (rate < 0.0) ? mf / (-rate) : inf;
            const double tau = std::min(tv, t_left);
            out.sub_int += Js * tau; out.sputw_int += Jw * tau; out.sputr_int += Jr * tau;
            out.box_gain += gv * tau; M += gv * tau;
            mf = (tau == tv) ? 0.0 : std::max(0.0, mf + rate * tau);
            t_left -= tau;
            out.regime_end = 2;
            continue;
        }
        const bool has = !dry && M > 0.0;
        const double a = has ? Js * f : 0.0, pw_all = has ? Jw * f : 0.0;
        // H2O ejection: barriered when shielded, else with the radiolytic part.
        const double pw = shield ? pw_all : 0.0, pwu = shield ? 0.0 : pw_all;
        const double pr_rad = has ? Jr * f : 0.0;   // the radiolytic part alone (output split)
        const double pr = pr_rad + pwu;
        const double ap = a + pw;
        const double u0 = 1.0 + d / ell;
        const double ES = ap / u0 + pr;
        const double L = force_S ? std::max(0.0, ES - B) : ES - B;
        const bool into_S = force_S;
        force_S = false;
        if (has && ES > 0.0 && (L >= 0.0 || into_S)) {
            out.regime_end = 0;
            out.absorbing = false;
            // Spencer criterion: at d = 0 a lag forms only if beta L exceeds the gardening rate.
            const double dd0 = beta * L - vm;
            if (!(beta > 0.0) || (d <= 0.0 && dd0 <= 0.0) || dd0 == 0.0) {
                // Constant rates: no lag can form (beta = 0: pure ice), the Spencer-absorbing
                // d = 0 state, or an exact balance. With beta = 0 a lag left over is gardened.
                const double t_dep = (L > 0.0) ? M / L : inf;
                const double t_zero = (d > 0.0 && !(beta > 0.0) && vm > 0.0) ? d / vm : inf;
                const double tau = std::min(t_left, std::min(t_dep, t_zero));
                out.sub_int += (a / u0) * tau; out.sputw_int += (pw / u0 + pwu) * tau; out.sputr_int += pr_rad * tau;
                M = (tau == t_dep) ? 0.0 : std::max(0.0, M - L * tau);
                if (tau == t_dep) dry = true;
                if (d > 0.0 && !(beta > 0.0)) d = (tau == t_zero) ? 0.0 : std::max(0.0, d - vm * tau);
                out.absorbing = (d <= 0.0 && beta > 0.0);
                t_left -= tau;
                continue;
            }
            const double q = pr - B - vm / beta;
            const double g0 = ap + q * u0;
            const double eps = q / g0;
            const double k = ell / (beta * g0);
            auto t_of = [&](double sv) { double Ll, Mm; lagLM(eps, sv, Ll, Mm); return k * (u0 * Ll + Mm); };
            // Bracket of reachable s: growth toward u* (eps < 0), shrink toward d = 0 or u* > 1.
            double s_lim;           // the attractor or the d = 0 boundary
            double t_zero = inf;    // time to reach d = 0 (shrinking only)
            if (g0 > 0.0) {
                s_lim = (eps < 0.0) ? -1.0 / eps : inf;
            } else {
                const double s_star = -1.0 / eps;           // eps > 0 when shrinking
                const double s_zero = 1.0 - u0;
                if (s_star > s_zero) s_lim = s_star;         // u* > 1: approached, never d = 0
                else { s_lim = s_zero; t_zero = t_of(s_zero); }
            }
            auto s_at = [&](double tt) {
                // Root of t(s) = tt between 0 and s_lim; t is monotone in |s|.
                double lo = 0.0, hi = std::isfinite(s_lim) ? s_lim : (beta * std::max(L, 0.0) / ell + 1.0) * tt;
                if (!std::isfinite(s_lim)) { while (t_of(hi) < tt) hi *= 2.0; }
                for (int it = 0; it < 200; ++it) {
                    const double m = 0.5 * (lo + hi);
                    if (t_of(m) < tt) lo = m; else hi = m;
                    if (std::abs(hi - lo) <= 1e-15 * std::max(std::abs(hi), 1e-300)) break;
                }
                return 0.5 * (lo + hi);
            };
            double tau = std::min(t_left, t_zero);
            double s_end = (tau == t_zero) ? 1.0 - u0 : s_at(tau);
            // Ice lost over tau: (l s + v_m tau)/beta; the box may run out first.
            auto lost = [&](double sv, double tt) { return (ell * sv + vm * tt) / beta; };
            bool depleted = false;
            if (lost(s_end, tau) >= M) {
                double lo = 0.0, hi = tau;
                for (int it = 0; it < 100; ++it) {
                    const double m = 0.5 * (lo + hi);
                    if (lost(s_at(m), m) < M) lo = m; else hi = m;
                }
                tau = hi; s_end = s_at(tau); depleted = true;
            }
            // Barriered emission int (a + pw)/u dt: balance or direct form (conditioning).
            const double grown = lost(s_end, tau), other = (B - pr) * tau;
            double I = grown + other;
            if (!(std::abs(I) > 1e-3 * std::max(std::abs(grown), std::abs(other)))) {
                double Ll, Mm;
                lagLM(eps, s_end, Ll, Mm);
                I = ell * ap / (beta * g0) * Ll;
            }
            I = std::max(0.0, I);
            out.sub_int += (ap > 0.0 ? a / ap : 0.0) * I;
            out.sputw_int += (ap > 0.0 ? pw / ap : 0.0) * I + pwu * tau;
            out.sputr_int += pr_rad * tau;
            const double loss = I + pr * tau - B * tau;
            M = depleted ? 0.0 : std::max(0.0, M - loss);
            if (depleted) dry = true;
            d = (tau == t_zero) ? 0.0 : std::max(0.0, d + ell * s_end);
            t_left -= tau;
            continue;
        }
        if (B < Vt) {
            const double lam = (Vt - B) / (Vt - ES);
            double tau = t_left, d_next = d;
            if (d > 0.0 && vm > 0.0 && lam > 0.0) {
                const double er = lam * vm;
                double t_ev = d / er, d_ev = 0.0;
                bool crosses = false;
                if (has && ap > 0.0 && B > pr) {
                    const double dc = ell * (ap / (B - pr) - 1.0);   // E_S(dc) = B
                    if (dc > 0.0 && dc < d) {
                        if (dc >= d * (1.0 - 1e-12)) { force_S = true; continue; }   // already there
                        t_ev = (d - dc) / er; d_ev = dc; crosses = true;
                    }
                }
                if (t_ev < t_left) { tau = t_ev; d_next = d_ev; force_S = crosses; }
                else d_next = std::max(0.0, d - er * tau);
            }
            out.sub_int += (lam * a / u0 + (1.0 - lam) * Js) * tau;
            out.sputw_int += (lam * (pw / u0 + pwu) + (1.0 - lam) * Jw) * tau;
            out.sputr_int += (lam * pr_rad + (1.0 - lam) * Jr) * tau;
            const double g = (1.0 - lam) * gv * tau;
            out.box_gain += g; M += g;
            d = d_next;
            out.regime_end = (B > 0.0 || ES > 0.0) ? 1 : 3;
            t_left -= tau;
            continue;
        }
        out.sub_int += Js * t_left; out.sputw_int += Jw * t_left; out.sputr_int += Jr * t_left;
        mf += (B - Vt) * t_left;
        out.box_gain += gv * t_left; M += gv * t_left;
        out.regime_end = (B > 0.0) ? 2 : 3;
        t_left = 0.0;
    }
    if (t_left > 0.0) {
        std::ostringstream m;
        m << std::scientific << std::setprecision(17) << "solveGardenFaceStep: too many regime changes in one step"
          << " (Js " << Js << ", Jw " << Jw << ", Jr " << Jr << ", B " << B << ", d0 " << d0 << ", mf0 " << mf0
          << ", M0 " << M0 << ", f " << f << ", beta " << beta << ", l " << ell << ", vm " << vm << ", gv " << gv
          << ", dt " << dt << ", shield " << shield << "; left " << t_left << " s at d " << d << ", mf " << mf
          << ", M " << M << ")";
        throw std::logic_error(m.str());
    }
    out.d_end = d; out.mf_end = mf; out.M_end = M;
    return out;
}

/**
 * PTM-676: one garden face over dt in nsub substeps (see
 * MultiTimescaleConfig::garden_subcycle_kappa). Each substep is one
 * solveGardenFaceStep() call; between substeps the gardening entrainment of the box
 * growth ((1 - c0) rho dh of ice, c0 dh of non-ice) is added and the box composition
 * (c, f, beta) is recomputed, so a face whose box turns over within the step follows
 * its own time scale. Emission integrals are summed. N_end, M_end: the box after the
 * step INCLUDING the entrainment (the single call leaves that to the caller).
 */
inline GardenFaceStep solveGardenFaceSubcycled(const IceLayerModel& m, double Js, double Jw, double Jr, double B,
                                               double d0, double mf0, double M0, double N0, double f0, double beta0,
                                               double c0, double t0, double dt, int nsub,
                                               double& N_end, double& M_end) {
    const auto& lc = m.getLagConfig();
    const double phi = lc.lag_porosity, rho = layer_constants::ICE_DENSITY, ell = m.getLagBarrierLength();
    const double vm = lc.gardening ? lc.gardening_rate : 0.0;
    const double gv = lc.gardening ? rho * (1.0 - lc.lag_porosity) * lc.gardening_rate : 0.0;
    double d = d0, mf = mf0, M = M0, N = N0, f = f0, beta = beta0, t = t0;
    const int n = std::max(1, nsub);
    const double h = dt / n;
    GardenFaceStep out;
    for (int k = 0; k < n; ++k) {
        const GardenFaceStep s = solveGardenFaceStep(Js, Jw, Jr, B, d, mf, M, f, beta, ell, vm, gv, h, lc.sputter_shield);
        out.sub_int += s.sub_int; out.sputw_int += s.sputw_int; out.sputr_int += s.sputr_int;
        out.box_gain += s.box_gain;
        out.regime_end = s.regime_end; out.absorbing = s.absorbing;
        const double dh = lc.gardening ? lc.gardenBoxDepth(t + h) - lc.gardenBoxDepth(t) : 0.0;
        N = std::max(0.0, N - (1.0 - phi) * (s.d_end - d)) + c0 * dh;
        M = s.M_end + (1.0 - c0) * rho * dh;
        d = s.d_end; mf = s.mf_end; t += h;
        double c;
        m.boxCompositionOf(N, M, c, f, beta);
    }
    out.d_end = d; out.mf_end = mf; out.M_end = M;
    N_end = N; M_end = M;
    return out;
}

/**
 * PTM-676: substeps a frost-free face with box ice M needs over dt. Subcycling applies
 * only where the single call is step-size dependent: the box gardening entrains more
 * than kappa x M of ice within the step (S dt > kappa M, S = entrainment rate), which
 * the single call books after the step while the loss runs through it. Such a face gets
 * substeps of at most kappa x its turnover time M / max(|E_S - B|, S). A face that
 * merely drains within the step is left to the single call (its closed form handles the
 * depletion; its exposed fraction is held over the step, as before). 1 = single call.
 */
inline int gardenSubcycleSubsteps(double kappa, int cap, double M, double mf, double ES, double B, double S,
                                  double dt, bool* capped = nullptr) {
    if (capped) *capped = false;
    if (!(kappa > 0.0) || !(dt > 0.0) || !(M > 0.0) || mf > 0.0) return 1;
    if (!(S * dt > kappa * M)) return 1;
    const double rate = std::max(std::abs(ES - B), S);
    if (!(rate > 0.0)) return 1;
    const double n = std::ceil(dt * rate / (kappa * M));
    if (!(n > 1.0)) return 1;
    const double c = static_cast<double>(std::max(1, cap));
    if (capped) *capped = n > c;   // the substeps would have been longer than kappa x tau
    return static_cast<int>(std::min<double>(n, c));
}

/**
 * Garden mode: instantaneous availabilities, the first instant of solveGardenFaceStep.
 * as: sublimation; apw, apr: the H2O-ejection and radiolytic parts; ap: the gross sputter
 * (w apw + (1 - w) apr, w = H2O share of the gross potential). Without a deposition value
 * the recycling regime cannot be identified and the plain rule is used.
 */
inline void gardenInstantAvailability(double Js, double Jp, double w, double B, bool have_B, double d,
                                      double mf, double M, double f, double ell, double gv, bool shield,
                                      double& as, double& apw, double& apr, double& ap) {
    if (mf > 0.0) { as = apw = apr = ap = 1.0; return; }
    const bool has = M > 0.0;
    const double fb = has ? ((d > 0.0) ? f / (1.0 + d / ell) : f) : 0.0;
    const double fu = has ? f : 0.0;
    double fs = fb, fw = shield ? fb : fu, fr = fu;
    auto gross = [&](double x, double y) { return (x == y) ? x : w * x + (1.0 - w) * y; };
    as = fs; apw = fw; apr = fr; ap = gross(fw, fr);
    if (!have_B) return;
    const double ES = Js * fs + Jp * ap, Vt = Js + Jp + gv;
    if (has && ES > 0.0 && B <= ES) return;
    if (B < Vt) {
        const double lam = (Vt - B) / (Vt - ES);
        as = lam * fs + (1.0 - lam); apw = lam * fw + (1.0 - lam); apr = lam * fr + (1.0 - lam);
        ap = gross(apw, apr);
        return;
    }
    as = apw = apr = ap = (B > 0.0) ? 1.0 : 0.0;
}

/**
 * Lag barrier: the instantaneous availabilities (sublimation, sputter) of a face,
 * i.e. its current emission / pure-ice potential, matching the first instant of
 * solveLagFaceStep. Without a deposition value (have_B = false: before the first
 * step) the recycling regime cannot be identified and the plain rule is used
 * (veneer: 1, 1; no substrate ice: 0, 0; else f/(1 + d/l), f), which is
 * IceLayerModel::getSublimationAvailability / getSputterAvailability bit for bit.
 */
inline void lagInstantAvailability(double Js, double Jp, double B, bool have_B, double d, double mf,
                                   double M, double f, double ell, double& as, double& ap) {
    if (mf > 0.0) { as = ap = 1.0; return; }
    const bool has = M > 0.0;
    const double fs = has ? ((d > 0.0) ? f / (1.0 + d / ell) : f) : 0.0;
    const double fp = has ? f : 0.0;
    if (!have_B) { as = fs; ap = fp; return; }
    const double ES = Js * fs + Jp * fp, Jv = Js + Jp;
    if (has && ES > 0.0 && B <= ES) { as = fs; ap = fp; return; }
    if (B < Jv) {
        const double lam = (Jv - B) / (Jv - ES);
        as = lam * fs + (1.0 - lam);
        ap = lam * fp + (1.0 - lam);
        return;
    }
    as = ap = (B > 0.0) ? 1.0 : 0.0;
}

/**
 * Configuration for multi-timescale evolution
 */
struct MultiTimescaleConfig {
    // Number of diurnal cycles per epoch
    int cycles_per_epoch = timescale_constants::DEFAULT_CYCLES_PER_EPOCH;

    // Total number of epochs to simulate
    int num_epochs = 1000;  // ~1000 years with default cycles_per_epoch

    // Export interval (epochs between VTU exports, 0 = final only)
    int export_interval = 100;

    // Progress reporting
    bool show_progress = true;

    // Convergence threshold for early termination (max concentration change per epoch)
    double convergence_threshold = timescale_constants::CONVERGENCE_THRESHOLD;

    // Flux imbalance threshold for mass balance convergence
    // Converges when |sublimation + sputtering - deposition| / (sublimation + sputtering) < threshold
    double flux_convergence_threshold = timescale_constants::FLUX_CONVERGENCE_THRESHOLD;

    // PTM-644: deposition over a step follows the emission produced over that step
    // (see MultiTimescaleEvolution::evolveEpoch). false reproduces the pre-fix
    // operator splitting: deposition held at its start-of-step value while the
    // sources follow the ice availability, and (linear availability mode) the
    // trapezoid mean of the availability.
    bool superstep_consistent_deposition = true;

    // Upper bound on one super-step, in epochs (0 = none). Used for step-size
    // refinement checks; the physical bound is computeSafeEpochCount().
    int max_superstep_epochs = 0;

    // PTM-676: garden-step subcycling. A frost-free face whose box gardening entrains
    // more than kappa of its box ice within the step is integrated in substeps of at
    // most kappa x its turnover time (tau = M / max(|E_S - B|, entrainment rate)), with
    // the entrainment and the box composition (c, f, beta) updated between substeps.
    // Without it the equilibrium of such a face depends on the super-step length (the
    // entrainment is booked after the step while the loss runs through it).
    // 0 = off (the pre-PTM-676 single call).
    double garden_subcycle_kappa = 0.1;
    int garden_subcycle_cap = 1024;

    // Constructor with defaults
    MultiTimescaleConfig() = default;

    // Convenience: compute total simulation time
    double totalTimeYears() const {
        double epoch_duration = cycles_per_epoch * timescale_constants::EUROPA_DIURNAL_PERIOD;
        return num_epochs * epoch_duration / timescale_constants::SECONDS_PER_YEAR;
    }

    // Print configuration
    void print() const {
        std::cout << "\n=== Multi-Timescale Evolution Configuration ===" << std::endl;
        std::cout << "  Cycles per epoch: " << cycles_per_epoch << std::endl;
        std::cout << "  Number of epochs: " << num_epochs << std::endl;
        std::cout << "  Total simulation time: " << std::scientific << std::setprecision(2)
                  << totalTimeYears() << " years" << std::endl;
        std::cout << "  Export interval: " << export_interval << " epochs" << std::endl;
        std::cout << "  Convergence threshold: " << convergence_threshold << std::endl;
        std::cout << "  Flux convergence threshold: " << (flux_convergence_threshold * 100.0) << "%" << std::endl;
        std::cout << "  Step deposition: "
                  << (superstep_consistent_deposition
                      ? "consistent with the step's emission"
                      : "LEGACY start-of-step value (not applied; not mass-conserving)")
                  << std::endl;
        if (max_superstep_epochs > 0) {
            std::cout << "  Max super-step: " << max_superstep_epochs << " epochs" << std::endl;
        }
        std::cout << std::fixed;
    }
};

/**
 * Time-averaged flux data from diurnal cycle
 * These are the "slow" variables that drive long-term evolution
 *
 * Units documented for each field.
 */
struct DiurnalAverages {
    // Current effective fluxes (diurnal averages)
    std::vector<double> avg_sublimation_flux;   // Time-averaged emission [kg/(m²·s)]
    std::vector<double> avg_deposition_flux;    // Time-averaged deposition [kg/(m²·s)]
    std::vector<double> avg_sputtering_flux;    // Time-averaged sputtering [kg/(m²·s)]
    std::vector<double> avg_net_flux;           // Time-averaged net change [kg/(m²·s)]
    std::vector<double> avg_escape_flux;        // Time-averaged escape [kg/(m²·s)]
    std::vector<double> avg_temperature;        // Time-averaged temperature [K]

    // Time-averaged crystallization rate <k_cryst(T(t))> [s^-1].
    // Distinct from k_cryst(<T>) because k_cryst is exponential in T; using
    // <k_cryst(T)> avoids a Jensen's-inequality bias of many orders of
    // magnitude near the crystallization knife-edge (~125 K). Populated only
    // when the crystallization model is active; left at zero otherwise.
    std::vector<double> avg_crystallization_rate;

    // Emission-weighted temperature T_w,i = sum_t Phi_i(t) T_i(t) / sum_t Phi_i(t) [K]
    // (M1). The ballistic kernel's hop length scales with the temperature of
    // the molecules actually launched (d_char = 2 k T / (m g)), and sublimation
    // is exponential in T, so the launch-weighted temperature is far above the
    // diurnal mean (146 K vs 117 K on the delivered ico6 product). Using
    // avg_temperature for transport shortened every hop by ~1.25x. Same
    // Jensen's-inequality correction as avg_crystallization_rate above.
    // Faces with zero emission over the whole cycle launch nothing; for them
    // the diurnal mean is used and counted in emission_weighted_fallback_count.
    std::vector<double> emission_weighted_temperature;
    std::vector<double> emission_weight_sum;        // accumulator: sum_t Phi_i(t)
    size_t emission_weighted_fallback_count = 0;

    // Cached base fluxes (theoretical full-rate, normalized by ice fraction at cache time)
    // evolveEpoch() uses these directly with analytical exponential evolution
    std::vector<double> base_sublimation_flux;  // Full-rate sublimation [kg/(m²·s)]
    std::vector<double> base_sputtering_flux;   // Full-rate sputtering [kg/(m²·s)]
    std::vector<double> base_deposition_flux;   // Base deposition [kg/(m²·s)]

    // Cache for ice availability (ice fraction) at last full diurnal computation
    std::vector<double> cached_ice_availability;

    // Peak values (for diagnostics)
    std::vector<double> max_temperature;  // [K]
    std::vector<double> min_temperature;  // [K]

    // Cache status
    bool cache_valid = false;
    int cache_epoch = -1;

    // Total emission at cache time (for scaling deposition)
    double total_base_emission = 0.0;  // [kg/s] (summed over all faces)

    // PTM-642 sputter mass accounting. When sputter_transport is set,
    // avg_sputtering_flux / base_sputtering_flux are the GROSS sputter removal,
    // a fraction sputter_transport_share of it (the non-escaping H2O) enters the
    // ballistic transport, and deposition is kept as two parts, each rescaled by
    // the ratio of ITS OWN source (sublimation, sputter) to its cache-time value.
    // When it is clear (sputtering off, or --sputter-legacy-escape-only), the
    // original single (sublimation + sputtering) rescale runs unchanged.
    bool sputter_transport = false;
    double sputter_transport_share = 0.0;   // of gross: non-escaping H2O -> transport
    double sputter_escape_share = 0.0;      // of gross: escaping, all products
    double sputter_reservoir_share = 0.0;   // of gross: non-escaping O2 + H2
    std::vector<double> avg_deposition_sublimation_flux;  // [kg/(m²·s)] from sublimation
    std::vector<double> avg_deposition_sputter_flux;      // [kg/(m²·s)] from sputtered H2O
    std::vector<double> base_deposition_sublimation_flux;
    std::vector<double> base_deposition_sputter_flux;
    double total_base_sublimation = 0.0;  // [kg/s] sublimation source at cache time
    double total_base_sputter = 0.0;      // [kg/s] GROSS sputter removal at cache time

    // PTM-649. Per-face fate fractions of the gross sputter flux (diurnal means),
    // set when the composition varies with temperature (current parameter set:
    // tau(T) on the radiolytic part only). Empty = the global shares above.
    std::vector<double> sputter_face_transport_frac;
    std::vector<double> sputter_face_escape_frac;
    std::vector<double> sputter_face_reservoir_frac;
    double total_base_sputter_transport = 0.0;  // [kg/s] sputter transport source at cache time
    bool perFaceSputter() const { return !sputter_face_transport_frac.empty(); }
    double sputterTransportFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_transport_frac[i] : sputter_transport_share;
    }
    double sputterEscapeFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_escape_frac[i] : sputter_escape_share;
    }
    double sputterReservoirFrac(size_t i) const {
        return perFaceSputter() ? sputter_face_reservoir_frac[i] : sputter_reservoir_share;
    }

    // PTM-644. Deposition = base deposition x a global scale per source group
    // (one group, total emission, without sputter transport; sublimation and
    // sputter with it). The scales below are the CURRENT (start-of-step) values
    // set by cacheBaseFluxesUnscaled() (1) and applyScaledFluxes(); evolveEpoch()
    // replaces them over a step by the scales consistent with the step's
    // emission. The base deposition totals (area-weighted, summed in face order)
    // give each group's retention: deposited / emitted at cache time (sublimation:
    // 1 - Jeans - unscattered fraction; sputter: the non-escaping H2O share).
    double deposition_scale_total = 1.0;
    double deposition_scale_sub = 1.0;
    double deposition_scale_sput = 1.0;
    double total_base_deposition = 0.0;      // [kg/s]
    double total_base_deposition_sub = 0.0;  // [kg/s]
    double total_base_deposition_sput = 0.0; // [kg/s]

    explicit DiurnalAverages(size_t num_faces) {
        avg_sublimation_flux.resize(num_faces, 0.0);
        avg_deposition_flux.resize(num_faces, 0.0);
        avg_sputtering_flux.resize(num_faces, 0.0);
        avg_net_flux.resize(num_faces, 0.0);
        avg_escape_flux.resize(num_faces, 0.0);
        avg_temperature.resize(num_faces, 0.0);
        avg_crystallization_rate.resize(num_faces, 0.0);
        emission_weighted_temperature.resize(num_faces, 0.0);
        emission_weight_sum.resize(num_faces, 0.0);

        base_sublimation_flux.resize(num_faces, 0.0);
        base_sputtering_flux.resize(num_faces, 0.0);
        base_deposition_flux.resize(num_faces, 0.0);

        cached_ice_availability.resize(num_faces, 1.0);

        avg_deposition_sublimation_flux.resize(num_faces, 0.0);
        avg_deposition_sputter_flux.resize(num_faces, 0.0);
        base_deposition_sublimation_flux.resize(num_faces, 0.0);
        base_deposition_sputter_flux.resize(num_faces, 0.0);

        max_temperature.resize(num_faces, 0.0);
        min_temperature.resize(num_faces, 1000.0);
    }

    void reset() {
        std::fill(avg_sublimation_flux.begin(), avg_sublimation_flux.end(), 0.0);
        std::fill(avg_deposition_flux.begin(), avg_deposition_flux.end(), 0.0);
        std::fill(avg_sputtering_flux.begin(), avg_sputtering_flux.end(), 0.0);
        std::fill(avg_net_flux.begin(), avg_net_flux.end(), 0.0);
        std::fill(avg_escape_flux.begin(), avg_escape_flux.end(), 0.0);
        std::fill(avg_temperature.begin(), avg_temperature.end(), 0.0);
        std::fill(avg_crystallization_rate.begin(), avg_crystallization_rate.end(), 0.0);
        std::fill(emission_weighted_temperature.begin(), emission_weighted_temperature.end(), 0.0);
        std::fill(emission_weight_sum.begin(), emission_weight_sum.end(), 0.0);
        emission_weighted_fallback_count = 0;
        std::fill(max_temperature.begin(), max_temperature.end(), 0.0);
        std::fill(min_temperature.begin(), min_temperature.end(), 1000.0);
        // Note: base_* and cached_* are NOT reset - they persist across epochs
    }

    // Finalize averages after accumulation
    void finalize(int num_timesteps) {
        double inv_n = 1.0 / static_cast<double>(num_timesteps);
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            avg_sublimation_flux[i] *= inv_n;
            avg_deposition_flux[i] *= inv_n;
            avg_sputtering_flux[i] *= inv_n;
            avg_net_flux[i] *= inv_n;
            avg_escape_flux[i] *= inv_n;
            avg_temperature[i] *= inv_n;
            avg_crystallization_rate[i] *= inv_n;
        }
        // Emission-weighted temperature (ratio of sums; no 1/n needed).
        size_t fallback = 0;
        for (size_t i = 0; i < emission_weighted_temperature.size(); ++i) {
            if (emission_weight_sum[i] > 0.0) {
                emission_weighted_temperature[i] /= emission_weight_sum[i];
            } else {
                emission_weighted_temperature[i] = avg_temperature[i];
                ++fallback;
            }
        }
        emission_weighted_fallback_count = fallback;
    }

    // Cache base (pure-ice potential) fluxes. On entry the avg_* fields have
    // ALREADY been scaled by the per-face ice availability, and
    // avg_deposition_flux comes from that scaled emission. The unscaled pure-ice
    // potential fluxes are stored directly as base_* (no availability floor), so
    // the exposure factor is applied dynamically and a face that holds no ice
    // today resumes sublimating once deposition gives it ice. total_base_emission
    // is the emission that produced base_deposition_flux, so the deposition scale
    // is 1 at cache time.
    //
    // (This replaced cacheBaseFluxes(), which divided the pure-ice diurnal
    // emission by the availability as though it were already scaled. That made
    // the first-epoch rate the pure-ice rate for any initial composition and left
    // deposition and sublimation out of step by the availability factor.)
    void cacheBaseFluxesUnscaled(const std::vector<double>& unscaled_sublimation,
                                 const std::vector<double>& unscaled_sputtering,
                                 const std::vector<double>& ice_availability,
                                 const std::vector<double>& face_areas) {
        total_base_emission = 0.0;
        total_base_sublimation = 0.0;
        total_base_sputter = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            base_sublimation_flux[i] = unscaled_sublimation[i];
            base_sputtering_flux[i] = unscaled_sputtering.empty() ? 0.0 : unscaled_sputtering[i];
            base_deposition_flux[i] = avg_deposition_flux[i];
            total_base_emission += (base_sublimation_flux[i] + base_sputtering_flux[i])
                                   * ice_availability[i] * face_areas[i];
            cached_ice_availability[i] = ice_availability[i];
            if (sputter_transport) {
                base_deposition_sublimation_flux[i] = avg_deposition_sublimation_flux[i];
                base_deposition_sputter_flux[i] = avg_deposition_sputter_flux[i];
                total_base_sublimation += base_sublimation_flux[i] * ice_availability[i] * face_areas[i];
                total_base_sputter += base_sputtering_flux[i] * ice_availability[i] * face_areas[i];
            }
        }
        total_base_sputter_transport = 0.0;
        if (sputter_transport && perFaceSputter()) {
            for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
                total_base_sputter_transport += base_sputtering_flux[i] * sputter_face_transport_frac[i]
                                                * ice_availability[i] * face_areas[i];
            }
        }
        deposition_scale_total = deposition_scale_sub = deposition_scale_sput = 1.0;
        total_base_deposition = total_base_deposition_sub = total_base_deposition_sput = 0.0;
        for (size_t i = 0; i < avg_sublimation_flux.size(); ++i) {
            total_base_deposition += base_deposition_flux[i] * face_areas[i];
            total_base_deposition_sub += base_deposition_sublimation_flux[i] * face_areas[i];
            total_base_deposition_sput += base_deposition_sputter_flux[i] * face_areas[i];
        }
        cache_valid = true;
    }

    // Apply ice fraction scaling to cached base fluxes (fast operation).
    // Sets avg_* fields for diagnostics/export. evolveEpoch() uses base_* directly
    // with analytical exponential evolution for the actual concentration update.
    void applyScaledFluxes(const std::vector<double>& ice_availability,
                           const std::vector<double>& face_areas) {
        size_t n = avg_sublimation_flux.size();

        if (sputter_transport) {
            // PTM-642 (c): each deposition part follows the source that produced
            // it. The old single (sublimation + sputtering) ratio let deposition
            // exceed sublimation whenever the two sources lost ice availability
            // at different rates (sublimation is equatorial, sputtering trailing).
            // Face-order (deterministic) sums: an OpenMP reduction here made the
            // deposition scale, and so every face's state, vary by an ulp with the
            // thread count and run to run (PTM-645; flaked the PTM-644 equilibrium test).
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                avg_sublimation_flux[i] = base_sublimation_flux[i] * ice_availability[i];
                avg_sputtering_flux[i] = base_sputtering_flux[i] * ice_availability[i];
            }
            double s_now = 0.0, p_now = 0.0;
            for (size_t i = 0; i < n; ++i) {
                s_now += avg_sublimation_flux[i] * face_areas[i];
                p_now += avg_sputtering_flux[i] * face_areas[i];
            }
            const double scale_s = (total_base_sublimation > 1e-30) ? s_now / total_base_sublimation : 1.0;
            double scale_p = (total_base_sputter > 1e-30) ? p_now / total_base_sputter : 1.0;
            if (perFaceSputter()) {
                // The sputter deposition follows ITS source, the per-face transported share.
                double c_now = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    c_now += avg_sputtering_flux[i] * sputter_face_transport_frac[i] * face_areas[i];
                }
                scale_p = (total_base_sputter_transport > 1e-30) ? c_now / total_base_sputter_transport : 1.0;
            }
            deposition_scale_sub = scale_s;
            deposition_scale_sput = scale_p;
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                avg_deposition_sublimation_flux[i] = base_deposition_sublimation_flux[i] * scale_s;
                avg_deposition_sputter_flux[i] = base_deposition_sputter_flux[i] * scale_p;
                avg_deposition_flux[i] = avg_deposition_sublimation_flux[i] + avg_deposition_sputter_flux[i];
                avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
            }
            return;
        }

        // Pass 1: Compute scaled emission per face and global total (fused with reduction)
        double total_current_emission = 0.0;

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_sublimation_flux[i] = base_sublimation_flux[i] * ice_availability[i];
            avg_sputtering_flux[i] = base_sputtering_flux[i] * ice_availability[i];
        }
        for (size_t i = 0; i < n; ++i) {  // face order: deterministic (PTM-645)
            total_current_emission += (avg_sublimation_flux[i] + avg_sputtering_flux[i]) * face_areas[i];
        }

        // Deposition scales with total emission (both now area-weighted [kg/s])
        double deposition_scale = (total_base_emission > 1e-30)
                                ? total_current_emission / total_base_emission
                                : 1.0;
        deposition_scale_total = deposition_scale;

        // Pass 2: Apply deposition scaling and compute net flux
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_deposition_flux[i] = base_deposition_flux[i] * deposition_scale;
            avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
        }
    }

    // Lag barrier: sublimation and sputtering have separate availabilities (the
    // barrier acts on sublimation only). The same bookkeeping as cacheBaseFluxesUnscaled /
    // applyScaledFluxes, deliberately NOT shared with them: routing the legacy path
    // through these functions changed its double-precision rounding under -ffast-math
    // and broke the --no-lag-barrier byte identity (measured, 1 thread).
    std::vector<double> cached_sub_availability;   // lag only: availabilities at cache time
    std::vector<double> cached_sput_availability;

    // Garden mode (PTM-669): per-face composition of the gross sputter potential.
    // w = H2O-ejection share; tw, ew = transported / escaping share of the H2O part;
    // er = escaping share of the radiolytic part (the rest is atmospheric loss).
    // Set by the integration from the diurnal fate accumulation (empty otherwise).
    std::vector<double> sput_w, sput_tw, sput_ew, sput_er;
    bool hasSputterSplit() const { return !sput_w.empty(); }

    void cacheBaseFluxesUnscaledLag(const std::vector<double>& unscaled_sublimation,
                                    const std::vector<double>& unscaled_sputtering,
                                    const std::vector<double>& avail_sub,
                                    const std::vector<double>& avail_sput,
                                    const std::vector<double>& face_areas) {
        const size_t n = avg_sublimation_flux.size();
        total_base_emission = total_base_sublimation = total_base_sputter = 0.0;
        cached_sub_availability = avail_sub;
        cached_sput_availability = avail_sput;
        for (size_t i = 0; i < n; ++i) {
            base_sublimation_flux[i] = unscaled_sublimation[i];
            base_sputtering_flux[i] = unscaled_sputtering.empty() ? 0.0 : unscaled_sputtering[i];
            base_deposition_flux[i] = avg_deposition_flux[i];
            const double es = base_sublimation_flux[i] * avail_sub[i] * face_areas[i];
            const double ep = base_sputtering_flux[i] * avail_sput[i] * face_areas[i];
            // Equal availabilities (every face at t = 0): the legacy expression, bit for bit.
            total_base_emission += (avail_sub[i] == avail_sput[i])
                ? (base_sublimation_flux[i] + base_sputtering_flux[i]) * avail_sub[i] * face_areas[i]
                : es + ep;
            cached_ice_availability[i] = avail_sub[i];
            if (sputter_transport) {
                base_deposition_sublimation_flux[i] = avg_deposition_sublimation_flux[i];
                base_deposition_sputter_flux[i] = avg_deposition_sputter_flux[i];
                total_base_sublimation += es;
                total_base_sputter += ep;
            }
        }
        total_base_sputter_transport = 0.0;
        if (sputter_transport && perFaceSputter()) {
            for (size_t i = 0; i < n; ++i) {
                total_base_sputter_transport += base_sputtering_flux[i] * sputter_face_transport_frac[i]
                                                * avail_sput[i] * face_areas[i];
            }
        }
        deposition_scale_total = deposition_scale_sub = deposition_scale_sput = 1.0;
        total_base_deposition = total_base_deposition_sub = total_base_deposition_sput = 0.0;
        for (size_t i = 0; i < n; ++i) {
            total_base_deposition += base_deposition_flux[i] * face_areas[i];
            total_base_deposition_sub += base_deposition_sublimation_flux[i] * face_areas[i];
            total_base_deposition_sput += base_deposition_sputter_flux[i] * face_areas[i];
        }
        cache_valid = true;
    }

    void applyScaledFluxesLag(const std::vector<double>& avail_sub,
                              const std::vector<double>& avail_sput,
                              const std::vector<double>& face_areas) {
        const size_t n = avg_sublimation_flux.size();
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_sublimation_flux[i] = base_sublimation_flux[i] * avail_sub[i];
            avg_sputtering_flux[i] = base_sputtering_flux[i] * avail_sput[i];
        }
        if (sputter_transport) {
            double s_now = 0.0, p_now = 0.0;
            for (size_t i = 0; i < n; ++i) {
                s_now += avg_sublimation_flux[i] * face_areas[i];
                p_now += avg_sputtering_flux[i] * face_areas[i];
            }
            const double scale_s = (total_base_sublimation > 1e-30) ? s_now / total_base_sublimation : 1.0;
            double scale_p = (total_base_sputter > 1e-30) ? p_now / total_base_sputter : 1.0;
            if (perFaceSputter()) {
                double c_now = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    c_now += avg_sputtering_flux[i] * sputter_face_transport_frac[i] * face_areas[i];
                }
                scale_p = (total_base_sputter_transport > 1e-30) ? c_now / total_base_sputter_transport : 1.0;
            }
            deposition_scale_sub = scale_s;
            deposition_scale_sput = scale_p;
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                avg_deposition_sublimation_flux[i] = base_deposition_sublimation_flux[i] * scale_s;
                avg_deposition_sputter_flux[i] = base_deposition_sputter_flux[i] * scale_p;
                avg_deposition_flux[i] = avg_deposition_sublimation_flux[i] + avg_deposition_sputter_flux[i];
                avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
            }
            return;
        }
        double total_current_emission = 0.0;
        for (size_t i = 0; i < n; ++i) {
            total_current_emission += (avg_sublimation_flux[i] + avg_sputtering_flux[i]) * face_areas[i];
        }
        const double deposition_scale = (total_base_emission > 1e-30)
                                      ? total_current_emission / total_base_emission : 1.0;
        deposition_scale_total = deposition_scale;
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            avg_deposition_flux[i] = base_deposition_flux[i] * deposition_scale;
            avg_net_flux[i] = avg_deposition_flux[i] - avg_sublimation_flux[i] - avg_sputtering_flux[i];
        }
    }

    // Check if cache refresh is needed based on ice fraction changes.
    // Uses emission-weighted mean change: a face that contributes 0.1% of global
    // emission can deplete without triggering a global recomputation.
    // With continuous ice fractions, changes are gradual, so refresh triggers
    // less frequently than with the old binary model.
    bool needsRefresh(const std::vector<double>& ice_availability,
                      const std::vector<double>& face_areas,
                      double threshold) const {
        if (!cache_valid) return true;

        double weighted_change = 0.0;
        double total_weight = 0.0;
        int faces_changed = 0;

        for (size_t i = 0; i < ice_availability.size(); ++i) {
            double delta = std::abs(ice_availability[i] - cached_ice_availability[i]);
            // Weight by base emission contribution (area × flux)
            double weight = (base_sublimation_flux[i] + base_sputtering_flux[i]) * face_areas[i];
            weighted_change += delta * weight;
            total_weight += weight;
            if (delta > 0.01) faces_changed++;
        }

        double mean_change = (total_weight > 1e-30) ? weighted_change / total_weight : 0.0;

        // Refresh if emission-weighted mean change exceeds threshold
        // OR if >5% of faces changed significantly (structural change)
        return (mean_change > threshold) ||
               (faces_changed > static_cast<int>(0.05 * ice_availability.size()));
    }
};

/**
 * Statistics for long-term evolution tracking
 *
 * Units documented for each field.
 */
struct EvolutionStatistics {
    int current_epoch = 0;
    double elapsed_time_years = 0.0;  // [years]

    // Global mass balance
    double total_sublimated_mass = 0.0;    // [kg]
    double total_deposited_mass = 0.0;     // [kg]
    double total_sputtered_mass = 0.0;     // [kg]
    double total_escaped_mass = 0.0;       // [kg]

    // Non-ice concentration statistics
    double max_nonice_concentration = 0.0;   // [dimensionless, 0-1]
    double mean_nonice_concentration = 0.0;  // [dimensionless, 0-1]
    size_t faces_with_ice = 0;               // count
    size_t faces_depleted = 0;               // count (concentration = 1)

    // Convergence tracking
    double max_concentration_change = 0.0;  // [dimensionless] max change this epoch
    bool converged = false;

    // Mass balance rates (per epoch, for convergence diagnostics)
    double sublimation_rate = 0.0;          // [kg/s] global sublimation rate this epoch
    double deposition_rate = 0.0;           // [kg/s] global deposition rate this epoch
    double sputtering_rate = 0.0;           // [kg/s] global sputtering rate this epoch
    double net_mass_loss_rate = 0.0;        // [kg/s] net loss (sublim + sput - depos)
    double flux_imbalance_fraction = 0.0;   // |net| / (sublim + sput), dimensionless

    // PTM-642 sputter fates (set only when the new accounting is active; then
    // sputtering_rate is the GROSS removal). Rates are per step [kg/s]; the
    // redeposited rate is the start-of-step deposition the step used, while
    // sputtering_rate is averaged over the step's ice availability, so the two
    // differ by the operator-splitting lag of a super-step.
    bool sputter_partition_active = false;
    double sputter_redeposited_rate = 0.0;
    double sputter_escaped_rate = 0.0;
    double sputter_reservoir_rate = 0.0;
    double sublimation_deposition_rate = 0.0;
    double total_sputter_redeposited_mass = 0.0;  // [kg]
    double total_sputter_escaped_mass = 0.0;      // [kg]
    double total_sputter_reservoir_mass = 0.0;    // [kg]
    // Decomposition of the cumulative balance (new accounting only):
    // sublimated + sputtered - deposited = sputter escaped + reservoir
    //   + sublimation Jeans escape - super-step lag.
    // The lag is deposition in excess of the step-mean sources: a super-step
    // holds deposition at its start-of-step value while sublimation and
    // sputtering follow the step-mean ice availability (PTM-644).
    double total_sublimation_jeans_mass = 0.0;    // [kg]
    double total_superstep_lag_mass = 0.0;        // [kg]

    // PTM-644 step conservation audit (all multiscale modes; it throws only in
    // the consistent mode). Per step: deposited = retained emission, where each
    // source group retains rho_k = kernel deposition / kernel source at cache time.
    // Cumulative identity: sublimated + sputtered - deposited
    //   = not_redeposited - step_residual.
    bool superstep_consistent = false;
    double total_step_residual_mass = 0.0;     // [kg] deposited - retained emission
    double total_not_redeposited_mass = 0.0;   // [kg] emitted - retained (Jeans, unscattered, sputter escape + reservoir)
    double max_step_residual_rel = 0.0;        // max |residual| / emitted over steps
    double last_step_residual_rel = 0.0;
    double total_layer_overflow_mass = 0.0;    // [kg] deposition onto pure-ice faces beyond the layer
    double total_external_deposition_mass = 0.0;  // [kg] deposition with no cached source (synthetic only)
    double total_inventory_error_mass = 0.0;   // [kg] sum |state change - booked fluxes| (numerical)
    double max_inventory_error_rel = 0.0;      // per step, / emitted mass
    double last_step_scale_sub = 1.0, last_step_scale_sput = 0.0;
    long long step_audit_count = 0;
    long long steps_evolved = 0;
    long long step_solver_evaluations = 0;
    int step_solver_max_iterations = 0;
    // PTM-640 albedo step limiter (predictions do not change the state).
    long long step_solver_slope_fallbacks = 0;  // FD slope outside [-1, 0): used -1
    long long step_solver_loose = 0;            // accepted at |residual| <= 1e-8 after stagnation
    long long predictions = 0;
    long long prediction_evaluations = 0;
    long long steps_albedo_limited = 0;

    // Lag-layer diffusion barrier (lag_active only when IceLayerModel::enableLagBarrier ran).
    // The inventory is sum (substrate ice + frost veneer) x area; the global audit is
    //   inventory_now - inventory_0 = -(sublimated + sputtered - deposited).
    bool lag_active = false;
    double lag_inventory_initial_mass = 0.0;   // [kg]
    double lag_inventory_mass = 0.0;           // [kg]
    double lag_veneer_mass = 0.0;              // [kg] sum m_f x area
    double lag_max_thickness = 0.0;            // [m]
    size_t lag_faces_over_warn = 0;            // faces with d above the warning thickness
    size_t lag_regime_count[4] = {0, 0, 0, 0}; // last step: S, sliding, veneer, none
    // Garden mode (PTM-669): ice brought in by the growing box; the audit becomes
    //   inventory_now - inventory_0 = entrained - (sublimated + sputtered - deposited).
    double garden_entrained_mass = 0.0;        // [kg]
    size_t garden_absorbing_faces = 0;         // last step: Spencer-absorbing d = 0 faces
    // PTM-676 subcycling (commits only; the evaluate/predict passes repeat it)
    long long garden_subcycled_face_steps = 0; // face-steps integrated in substeps
    int garden_subcycle_max = 0;               // most substeps in one face-step
    long long garden_subcycle_capped = 0;      // face-steps that hit the substep cap

    // Albedo coupling statistics
    double max_albedo_change = 0.0;       // [dimensionless] max albedo change this epoch
    double mean_albedo = 0.0;             // [dimensionless] mean albedo across all faces
    int albedo_temp_refreshes = 0;        // count of temperature refreshes triggered by albedo

    void print() const {
        std::cout << "\n--- Evolution Statistics (Epoch " << current_epoch << ") ---" << std::endl;
        std::cout << "  Elapsed time: " << std::scientific << std::setprecision(3)
                  << elapsed_time_years << " years" << std::endl;
        std::cout << "  Non-ice concentration:" << std::endl;
        std::cout << "    Max: " << std::fixed << std::setprecision(4)
                  << (max_nonice_concentration * 100) << "%" << std::endl;
        std::cout << "    Mean: " << (mean_nonice_concentration * 100) << "%" << std::endl;
        std::cout << "  Faces with ice: " << faces_with_ice << std::endl;
        std::cout << "  Faces depleted (no ice): " << faces_depleted << std::endl;
        std::cout << "  Max concentration change: " << std::scientific
                  << max_concentration_change << " /epoch" << std::endl;
        if (sublimation_rate > 0.0 || deposition_rate > 0.0) {
            std::cout << "  Mass balance:" << std::endl;
            std::cout << "    Sublimation: " << std::scientific << std::setprecision(3)
                      << sublimation_rate << " kg/s" << std::endl;
            std::cout << "    Deposition:  " << deposition_rate << " kg/s" << std::endl;
            if (sputtering_rate > 0.0) {
                std::cout << "    Sputtering:  " << sputtering_rate << " kg/s"
                          << (sputter_partition_active ? " (gross removal)" : " (escaping share only; legacy)")
                          << std::endl;
            }
            if (sputter_partition_active) {
                std::cout << "      deposition from sublimation: " << sublimation_deposition_rate << " kg/s" << std::endl;
                std::cout << "      sputter redeposited:         " << sputter_redeposited_rate << " kg/s" << std::endl;
                std::cout << "      sputter escaped:             " << sputter_escaped_rate << " kg/s" << std::endl;
                std::cout << "      sputter reservoir (O2+H2):   " << sputter_reservoir_rate << " kg/s" << std::endl;
            }
            std::cout << "    Net loss:    " << net_mass_loss_rate << " kg/s" << std::endl;
            std::cout << "    Flux imbalance: " << std::fixed << std::setprecision(2)
                      << (flux_imbalance_fraction * 100.0) << "%" << std::endl;
            std::cout << "  Cumulative mass balance:" << std::endl;
            std::cout << "    Total sublimated: " << std::scientific << std::setprecision(3)
                      << total_sublimated_mass << " kg" << std::endl;
            std::cout << "    Total deposited:  " << total_deposited_mass << " kg" << std::endl;
            if (total_sputtered_mass > 0.0) {
                std::cout << "    Total sputtered:  " << total_sputtered_mass << " kg" << std::endl;
            }
            if (sputter_partition_active) {
                std::cout << "      sputter redeposited: " << total_sputter_redeposited_mass << " kg" << std::endl;
                std::cout << "      sputter escaped:     " << total_sputter_escaped_mass << " kg" << std::endl;
                std::cout << "      sputter reservoir:   " << total_sputter_reservoir_mass << " kg" << std::endl;
                std::cout << "    Sublimation Jeans escape: " << total_sublimation_jeans_mass << " kg" << std::endl;
                std::cout << "    Super-step lag (deposition in excess of step-mean sources): "
                          << total_superstep_lag_mass << " kg" << std::endl;
                std::cout << "    (sublimated + sputtered - deposited = sputter escaped + reservoir + "
                          << "Jeans - lag)" << std::endl;
            }
            std::cout << "    Total escaped:    " << total_escaped_mass << " kg" << std::endl;
            if (lag_active) {
                const double net = total_sublimated_mass + total_sputtered_mass - total_deposited_mass;
                double resid = (lag_inventory_mass - lag_inventory_initial_mass) + net;
                // Garden mode: the entrained supply, in a separate non-inlined function so that
                // -ffast-math cannot fuse it into the PTM-659 expression above (measured: a fused
                // "- 0.0" moved the gardening-off audit line by an ulp).
                if (garden_entrained_mass > 0.0) {
                    resid = gardenAuditResidual(lag_inventory_mass, lag_inventory_initial_mass, net,
                                                garden_entrained_mass);
                }
                if (garden_entrained_mass > 0.0) {
                    std::cout << "  Gardening: box growth entrained " << garden_entrained_mass << " kg of ice; "
                              << garden_absorbing_faces << " faces Spencer-absorbing at d = 0" << std::endl;
                }
                std::cout << "  Lag barrier: ice inventory (substrate + frost veneer) " << lag_inventory_mass
                          << " kg, change " << (lag_inventory_mass - lag_inventory_initial_mass)
                          << " kg vs -(sublimated + sputtered - deposited) " << -net << " kg; residual "
                          << resid << " kg (" << (lag_inventory_initial_mass > 0.0
                                                  ? resid / lag_inventory_initial_mass : 0.0)
                          << " of the inventory)" << std::endl;
                std::cout << "    frost veneer " << lag_veneer_mass << " kg; max lag thickness "
                          << lag_max_thickness << " m; last step regimes: substrate " << lag_regime_count[0]
                          << ", recycling (frost-free fraction < 1) " << lag_regime_count[1]
                          << ", veneer " << lag_regime_count[2] << ", no ice " << lag_regime_count[3]
                          << std::endl;
            }
            if (step_audit_count > 0) {
                std::cout << "  Step conservation" << (superstep_consistent ? "" : " (LEGACY, not enforced)")
                          << ": not redeposited " << total_not_redeposited_mass << " kg, residual "
                          << total_step_residual_mass << " kg (max step " << max_step_residual_rel
                          << " rel), layer overflow " << total_layer_overflow_mass << " kg" << std::endl;
            }
        }
        if (mean_albedo > 0.0) {
            std::cout << "  Albedo coupling:" << std::endl;
            std::cout << "    Mean albedo: " << std::fixed << std::setprecision(4)
                      << mean_albedo << std::endl;
            std::cout << "    Max albedo change: " << std::scientific
                      << max_albedo_change << " /epoch" << std::endl;
            std::cout << "    Temperature refreshes: " << albedo_temp_refreshes << std::endl;
        }
        std::cout << "  Converged: " << (converged ? "yes" : "no") << std::endl;
        std::cout << std::fixed;
    }
};

/**
 * Multi-timescale evolution controller
 *
 * Manages the operator splitting between fast diurnal dynamics
 * and slow non-ice concentration evolution.
 */
class MultiTimescaleEvolution {
private:
    MultiTimescaleConfig config_;
    size_t num_faces_;

    // Diurnal averages (computed once per epoch refresh)
    std::unique_ptr<DiurnalAverages> diurnal_avg_;

    // Evolution statistics
    EvolutionStatistics stats_;

    // Concentration change rates for each face [1/s]
    std::vector<double> concentration_change_rate_;

    // Reusable buffer for ice availability queries (avoids per-epoch allocation)
    mutable std::vector<double> ice_availability_buf_;

    // Per-face scratch for evolveEpoch()'s face-order sums (PTM-644).
    struct StepBuffers {
        std::vector<double> emit_sub, emit_sput, dep_sub, dep_sput, deposition, overflow, inventory_err;
        std::vector<double> sput_transport, sput_escape, sput_reservoir;
        void resize(size_t n) {
            if (emit_sub.size() == n) return;
            for (auto* v : {&emit_sub, &emit_sput, &dep_sub, &dep_sput, &deposition, &overflow, &inventory_err,
                            &sput_transport, &sput_escape, &sput_reservoir}) {
                v->assign(n, 0.0);
            }
        }
    };
    StepBuffers step_buffers_;
    bool predict_skip_solve_ = false;
    std::vector<int> lag_regime_;      // lag barrier: regime at the end of the last step (see LagFaceStep)
    std::vector<double> lag_change_;   // lag barrier: |change of sublimation availability| over the last step
    std::vector<double> garden_entrained_;  // garden mode: ice entrained by the box growth, last step [kg]

public:
    /**
     * Constructor
     * @param num_faces Number of mesh faces
     * @param config Multi-timescale configuration
     */
    explicit MultiTimescaleEvolution(size_t num_faces,
                                      const MultiTimescaleConfig& config = MultiTimescaleConfig())
        : config_(config), num_faces_(num_faces) {
        diurnal_avg_ = std::make_unique<DiurnalAverages>(num_faces);
        concentration_change_rate_.resize(num_faces, 0.0);
        ice_availability_buf_.resize(num_faces, 0.0);
    }

    /**
     * Accumulate flux data from a single diurnal timestep
     * Call this during the diurnal cycle simulation
     *
     * @param emission_flux Current emission flux [kg/(m²·s)]
     * @param deposition_flux Current deposition flux [kg/(m²·s)]
     * @param sputtering_flux Current sputtering flux [kg/(m²·s)] (can be empty)
     * @param escape_flux Current escape flux [kg/(m²·s)]
     * @param temperatures Current surface temperatures [K]
     */
    void accumulateDiurnalStep(const std::vector<double>& emission_flux,
                                const std::vector<double>& deposition_flux,
                                const std::vector<double>& sputtering_flux,
                                const std::vector<double>& escape_flux,
                                const std::vector<double>& temperatures) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            diurnal_avg_->avg_sublimation_flux[i] += emission_flux[i];
            diurnal_avg_->avg_deposition_flux[i] += deposition_flux[i];

            if (!sputtering_flux.empty()) {
                diurnal_avg_->avg_sputtering_flux[i] += sputtering_flux[i];
            }

            diurnal_avg_->avg_escape_flux[i] += escape_flux[i];
            double sput = (!sputtering_flux.empty()) ? sputtering_flux[i] : 0.0;
            diurnal_avg_->avg_net_flux[i] += (deposition_flux[i] - emission_flux[i] - sput);
            diurnal_avg_->avg_temperature[i] += temperatures[i];
            diurnal_avg_->emission_weighted_temperature[i] += emission_flux[i] * temperatures[i];
            diurnal_avg_->emission_weight_sum[i] += emission_flux[i];

            diurnal_avg_->max_temperature[i] = std::max(diurnal_avg_->max_temperature[i], temperatures[i]);
            diurnal_avg_->min_temperature[i] = std::min(diurnal_avg_->min_temperature[i], temperatures[i]);
        }
    }

    /**
     * Finalize diurnal averages after one complete diurnal cycle
     * @param num_timesteps Number of timesteps in the diurnal cycle
     */
    void finalizeDiurnalAverages(int num_timesteps) {
        diurnal_avg_->finalize(num_timesteps);
    }

    /**
     * Accumulate per-face crystallization rate at one diurnal timestep.
     *
     * Must be called once per diurnal timestep (and once per stride if
     * subsampling is in use), exactly in lock-step with accumulateDiurnalStep,
     * so that the averaging denominator matches. After the diurnal cycle
     * completes, finalizeDiurnalAverages divides by the timestep count to
     * yield <k_cryst(T(t))> per face.
     *
     * @param k_cryst_per_face Pre-computed k_cryst(T(t), porosity) [s^-1]
     */
    void accumulateCrystallizationRate(const std::vector<double>& k_cryst_per_face) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            diurnal_avg_->avg_crystallization_rate[i] += k_cryst_per_face[i];
        }
    }

    /**
     * Reset diurnal accumulator for next epoch
     */
    void resetDiurnalAccumulator() {
        diurnal_avg_->reset();
    }

    /**
     * Check if flux cache is valid
     */
    bool isCacheValid() const {
        return diurnal_avg_->cache_valid;
    }

    /**
     * Cache the pure-ice potential base fluxes (all modes). The diurnal averages
     * must already be scaled by the ice availability factor.
     */
    void cacheBaseFluxesUnscaled(const IceLayerModel& ice_model,
                                 const IceLayerState& ice_state,
                                 const std::vector<double>& unscaled_sublimation,
                                 const std::vector<double>& unscaled_sputtering,
                                 const std::vector<double>& face_areas,
                                 int epoch) {
        if (lagOn(ice_model, ice_state)) {
            std::vector<double> as, ap;
            lagAvailabilities(ice_model, ice_state, unscaled_sublimation, unscaled_sputtering, as, ap);
            diurnal_avg_->cacheBaseFluxesUnscaledLag(unscaled_sublimation, unscaled_sputtering, as, ap,
                                                     face_areas);
            diurnal_avg_->cache_epoch = epoch;
            return;
        }
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability_buf_[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }
        diurnal_avg_->cacheBaseFluxesUnscaled(unscaled_sublimation, unscaled_sputtering,
                                              ice_availability_buf_, face_areas);
        diurnal_avg_->cache_epoch = epoch;
    }

    /**
     * Apply scaled fluxes using cached base values
     * Much faster than recomputing diurnal cycle
     */
    void applyScaledFluxes(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state,
                           const std::vector<double>& face_areas) {
        if (lagOn(ice_model, ice_state)) {
            std::vector<double> as, ap;
            lagAvailabilities(ice_model, ice_state, diurnal_avg_->base_sublimation_flux,
                              diurnal_avg_->base_sputtering_flux, as, ap);
            diurnal_avg_->applyScaledFluxesLag(as, ap, face_areas);
            return;
        }
        // Compute current ice availability factors
        for (size_t i = 0; i < num_faces_; ++i) {
            ice_availability_buf_[i] = ice_model.getIceAvailabilityFactor(ice_state, i);
        }

        diurnal_avg_->applyScaledFluxes(ice_availability_buf_, face_areas);
    }

    /** Garden mode: veneer gardening rate rho_ice (1 - phi) v_m [kg/(m² s)]. */
    static double gardenVeneerRate(const IceLayerModel& m) {
        const auto& lc = m.getLagConfig();
        return lc.gardening ? layer_constants::ICE_DENSITY * (1.0 - lc.lag_porosity) * lc.gardening_rate : 0.0;
    }

    /**
     * Garden mode: after the diurnal fate accumulation and the current availabilities,
     * rewrite the per-face gross-sputter fate fractions (transport / escape / reservoir)
     * for the CURRENT shielding, so that the transport source, the cache totals and the
     * audits see the H2O and radiolytic parts in their current proportion.
     */
    void setGardenSputterFractions(const IceLayerModel& ice_model, const IceLayerState& st,
                                   const std::vector<double>& Js, const std::vector<double>& Jp) {
        DiurnalAverages& dv = *diurnal_avg_;
        if (!dv.hasSputterSplit()) return;
        const auto& lc = ice_model.getLagConfig();
        const bool have_B = step_buffers_.deposition.size() == num_faces_ && stats_.steps_evolved > 0;
        const double ell = ice_model.getLagBarrierLength(), gv = gardenVeneerRate(ice_model);
        dv.sputter_face_transport_frac.resize(num_faces_);
        dv.sputter_face_escape_frac.resize(num_faces_);
        dv.sputter_face_reservoir_frac.resize(num_faces_);
        for (size_t i = 0; i < num_faces_; ++i) {
            double as, apw, apr, ap;
            gardenInstantAvailability(Js[i], Jp[i], dv.sput_w[i], have_B ? step_buffers_.deposition[i] : 0.0,
                                      have_B, st.lag_thickness[i], st.frost_veneer[i], st.substrate_ice[i],
                                      st.lag_exposed_fraction[i], ell, gv, lc.sputter_shield, as, apw, apr, ap);
            const double w = dv.sput_w[i];
            // Shares of the current gross: H2O part w apw / ap, radiolytic (1 - w) apr / ap.
            const double hw = (ap > 0.0) ? w * apw / ap : w, hr = 1.0 - hw;
            dv.sputter_face_transport_frac[i] = hw * dv.sput_tw[i];
            dv.sputter_face_escape_frac[i] = hw * dv.sput_ew[i] + hr * dv.sput_er[i];
            dv.sputter_face_reservoir_frac[i] = 1.0 - dv.sputter_face_transport_frac[i]
                                                - dv.sputter_face_escape_frac[i];
        }
    }

    /**
     * Lag barrier: per-face instantaneous availabilities (lagInstantAvailability) for
     * the pure-ice potentials Js, Jp, with the last step's deposition once a step has run.
     */
    // The lag model and its state go together; a lag-enabled model paired with a
    // state that enableLagBarrier() did not initialize is a programming error.
    static bool lagOn(const IceLayerModel& ice_model, const IceLayerState& st) {
        if (!ice_model.usesLagBarrier()) return false;
        if (!st.hasLag()) {
            throw std::logic_error("MultiTimescaleEvolution: the ice model has the lag barrier enabled but "
                                   "the ice state was not initialized by enableLagBarrier()");
        }
        return true;
    }

    void lagAvailabilities(const IceLayerModel& ice_model, const IceLayerState& st,
                           const std::vector<double>& Js, const std::vector<double>& Jp,
                           std::vector<double>& as, std::vector<double>& ap) const {
        lagOn(ice_model, st);
        as.resize(num_faces_);
        ap.resize(num_faces_);
        const bool have_B = step_buffers_.deposition.size() == num_faces_ && stats_.steps_evolved > 0;
        const double ell = ice_model.getLagBarrierLength();
        if (ice_model.usesGardenMode()) {
            const auto& lc = ice_model.getLagConfig();
            const double gv = gardenVeneerRate(ice_model);
            const DiurnalAverages& dv = *diurnal_avg_;
            for (size_t i = 0; i < num_faces_; ++i) {
                double apw, apr;
                const double w = dv.hasSputterSplit() ? dv.sput_w[i] : 1.0;
                gardenInstantAvailability(Js[i], Jp.empty() ? 0.0 : Jp[i], w,
                                          have_B ? step_buffers_.deposition[i] : 0.0, have_B,
                                          st.lag_thickness[i], st.frost_veneer[i], st.substrate_ice[i],
                                          st.lag_exposed_fraction[i], ell, gv, lc.sputter_shield,
                                          as[i], apw, apr, ap[i]);
            }
            return;
        }
        for (size_t i = 0; i < num_faces_; ++i) {
            lagInstantAvailability(Js[i], Jp.empty() ? 0.0 : Jp[i], have_B ? step_buffers_.deposition[i] : 0.0,
                                   have_B, st.lag_thickness[i], st.frost_veneer[i], st.substrate_ice[i],
                                   st.lag_exposed_fraction[i], ell, as[i], ap[i]);
        }
    }

    /**
     * Lag barrier: how far the emission PATTERN has moved since the transport was
     * cached, as the total-variation distance between the normalized current and
     * cached per-face emission (max over sublimation and sputtering; 0..1). A uniform
     * decay is not a pattern change (the PTM-644 scale absorbs it); a change here is,
     * and the cached deposition pattern no longer matches its sources.
     */
    double lagSourcePatternDrift(const IceLayerModel& ice_model, const IceLayerState& ice_state,
                                 const std::vector<double>& face_areas) const {
        const DiurnalAverages& d = *diurnal_avg_;
        if (!d.cache_valid || d.cached_sub_availability.size() != num_faces_) return 0.0;
        std::vector<double> as, ap;
        lagAvailabilities(ice_model, ice_state, d.base_sublimation_flux, d.base_sputtering_flux, as, ap);
        double worst = 0.0;
        for (int g = 0; g < 2; ++g) {
            const std::vector<double>& base = g == 0 ? d.base_sublimation_flux : d.base_sputtering_flux;
            const std::vector<double>& ref = g == 0 ? d.cached_sub_availability : d.cached_sput_availability;
            const std::vector<double>& now = g == 0 ? as : ap;
            double s_now = 0.0, s_ref = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) {
                s_now += base[i] * now[i] * face_areas[i];
                s_ref += base[i] * ref[i] * face_areas[i];
            }
            if (!(s_now > 0.0) || !(s_ref > 0.0)) continue;
            double tv = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) {
                tv += std::abs(base[i] * now[i] * face_areas[i] / s_now - base[i] * ref[i] * face_areas[i] / s_ref);
            }
            worst = std::max(worst, 0.5 * tv);
        }
        return worst;
    }

    /**
     * Check if cache refresh is needed.
     *
     * The transport product depends on TEMPERATURE (kernel shape), not ice
     * availability. Ice availability changes are handled cheaply by
     * applyScaledFluxes(), which scales cached base fluxes. A full diurnal
     * + transport recomputation (~200s for 1.3M faces) is only needed when:
     *   (a) Cache is not yet valid (first epoch)
     *   (b) Temperatures changed (albedo coupling — handled via force_cache_refresh
     *       in the caller)
     *
     * Ice availability changes do NOT require transport recomputation.
     *
     * NOTE (M3): this returns true only while the cache is empty. There is no
     * periodic or concentration-threshold refresh; the CLI refuses
     * --cache-refresh and --no-analytical. Appendix D.7.4/D.7.6 describing an
     * "adaptive cache refresh" is inaccurate (doc fix owed).
     */
    bool needsCacheRefresh(const IceLayerModel& ice_model,
                           const IceLayerState& ice_state,
                           const std::vector<double>& face_areas,
                           int current_epoch) const {
        if (!diurnal_avg_->cache_valid) return true;

        // Transport product depends on temperature, not ice availability.
        // applyScaledFluxes() handles ice depletion correctly by scaling
        // emission to 0 and reducing deposition proportionally.
        // Temperature changes are signaled via force_cache_refresh from
        // the albedo coupling logic in the caller.
        return false;
    }

    /**
     * Compute the maximum number of epochs that can be safely super-stepped.
     *
     * Uses the emission-weighted criterion: faces are sorted by time-to-depletion,
     * and we accumulate the weighted impact of each depletion until the cumulative
     * emission-weighted change exceeds the cache refresh threshold (10%).
     *
     * This allows jumping past many low-impact depletion events (cold polar faces
     * with negligible emission) while stopping before a high-impact face depletes.
     *
     * Also limits step size to avoid overshooting concentration bounds.
     *
     * @param ice_state Current ice layer state
     * @param face_areas Per-face areas [m²]
     * @param epoch_duration Duration of one epoch [s]
     * @param max_epochs Maximum epochs to step (remaining in simulation)
     * @return Number of safe epochs [1, max_epochs]
     */
    int computeSafeEpochCount(const IceLayerState& ice_state,
                              const std::vector<double>& face_areas,
                              double epoch_duration,
                              int max_epochs) const {
        if (max_epochs <= 1 || epoch_duration <= 0.0) return 1;

        // Compute total base emission for weighting
        double total_emission_weight = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            total_emission_weight += (diurnal_avg_->base_sublimation_flux[i] +
                                      diurnal_avg_->base_sputtering_flux[i]) * face_areas[i];
        }

        if (total_emission_weight < 1e-30) {
            // No emission at all → safe to step to the end
            return max_epochs;
        }

        // Build list of (time_to_event, emission_weight) for faces with significant rate changes
        struct FaceEvent {
            double time;      // Time until ice fraction changes significantly
            double weight;    // Emission weight of this face
        };
        std::vector<FaceEvent> events;
        events.reserve(num_faces_);

        double max_time = max_epochs * epoch_duration;

        for (size_t i = 0; i < num_faces_; ++i) {
            double rate = concentration_change_rate_[i];
            double conc = ice_state.nonice_concentration[i];

            if (rate > 0.0) {
                // Face losing ice → will deplete
                double room = layer_constants::MAX_NONICE_CONCENTRATION - conc;
                if (room > 1e-10) {
                    double t = room / rate;
                    if (t < max_time) {
                        double w = (diurnal_avg_->base_sublimation_flux[i] +
                                    diurnal_avg_->base_sputtering_flux[i]) * face_areas[i];
                        events.push_back({t, w});
                    }
                }
            } else if (rate < 0.0) {
                // Face gaining ice → ice fraction increasing
                // Track when concentration drops enough to significantly change sublimation rate
                if (conc > 0.01) {
                    // Time for concentration to drop by 0.01 (1% ice fraction change)
                    double t = 0.01 / (-rate);
                    if (t > 0.0 && t < max_time) {
                        double w = (diurnal_avg_->base_deposition_flux[i]) * face_areas[i];
                        events.push_back({t, w});
                    }
                }
            }
        }

        if (events.empty()) {
            // No events within the time window → safe to step all the way
            return max_epochs;
        }

        // Sort events by time
        std::sort(events.begin(), events.end(),
                  [](const FaceEvent& a, const FaceEvent& b) { return a.time < b.time; });

        // Walk through events, accumulating weighted change until threshold
        double cumulative_weighted_change = 0.0;
        double threshold = 0.1;  // 10% emission-weighted change
        double safe_time = max_time;

        for (const auto& event : events) {
            cumulative_weighted_change += event.weight / total_emission_weight;

            if (cumulative_weighted_change > threshold) {
                // This event pushes us over the threshold
                safe_time = event.time;
                break;
            }

            // Also check: if >5% of faces changed, stop (structural change)
            // (Approximate: count events so far)
        }

        // Convert safe time to epoch count (conservative: floor)
        int safe_epochs = static_cast<int>(safe_time / epoch_duration);

        // Clamp to [1, max_epochs]
        safe_epochs = std::max(1, std::min(safe_epochs, max_epochs));

        return safe_epochs;
    }

    /**
     * Lag barrier: the super-step bound. The face state no longer moves the
     * concentration; the events are, per face, (a) the barrier factor 1/(1 + d/l)
     * changing by 10% (d grows by 0.1 (l + d) at the current rate), weighted by the
     * face's sublimation; (b) the substrate running out; (c) a veneer vanishing,
     * weighted by (an upper bound on) the emission jump. As in computeSafeEpochCount, the step ends when
     * the events passed carry 10% of the current emission. The per-face step is
     * exact; this bounds the within-step change of the emission the PTM-644
     * deposition solve averages over, and of the pattern the cached transport holds.
     */
    int computeSafeEpochCountLag(const IceLayerModel& ice_model, const IceLayerState& st,
                                 const std::vector<double>& face_areas, double epoch_duration,
                                 int max_epochs) const {
        if (max_epochs <= 1 || epoch_duration <= 0.0) return 1;
        lagOn(ice_model, st);
        if (ice_model.usesGardenMode()) {
            return computeSafeEpochCountGarden(ice_model, st, face_areas, epoch_duration, max_epochs);
        }
        const DiurnalAverages& dv = *diurnal_avg_;
        const double ell = ice_model.getLagBarrierLength();
        const bool have_step_dep = step_buffers_.deposition.size() == num_faces_;
        struct FaceEvent { double time, weight; };
        std::vector<FaceEvent> events;
        events.reserve(num_faces_);
        const double max_time = max_epochs * epoch_duration;
        std::vector<double> as_now, ap_now;
        lagAvailabilities(ice_model, st, dv.base_sublimation_flux, dv.base_sputtering_flux, as_now, ap_now);
        double total = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            const double Js = dv.base_sublimation_flux[i], Jp = dv.base_sputtering_flux[i];
            total += (Js * as_now[i] + Jp * ap_now[i]) * face_areas[i];
            const double B = have_step_dep ? step_buffers_.deposition[i] : dv.avg_deposition_flux[i];
            const double M = st.substrate_ice[i], f = st.lag_exposed_fraction[i];
            const double u = 1.0 + st.lag_thickness[i] / ell;
            const double a = (M > 0.0) ? Js * f : 0.0, p = (M > 0.0) ? Jp * f : 0.0;
            const double ES = a / u + p;
            if (st.frost_veneer[i] > 0.0) {
                const double Jv = Js + Jp;
                if (B < Jv) {
                    const double t = st.frost_veneer[i] / (Jv - B);
                    // Weight: Jv - min(ES, B) bounds the jump Jv - max(ES, B) from above
                    // (conservative: the step ends no later than the documented criterion).
                    if (t < max_time) events.push_back({t, std::abs(Jv - std::min(ES, B)) * face_areas[i]});
                }
                continue;
            }
            if (!(M > 0.0) || !(ES > B)) continue;       // recycling, veneer growth or no ice: steady
            const double net = ES - B;
            const double t_dep = M / net;
            if (t_dep < max_time) events.push_back({t_dep, ES * face_areas[i]});
            const double beta = st.lag_beta[i];
            if (beta > 0.0 && a > 0.0) {
                const double t10 = 0.1 * u * ell / (beta * net);
                if (t10 < max_time) events.push_back({t10, (a / u) * face_areas[i]});
            }
        }
        if (!(total > 1e-30) || events.empty()) return max_epochs;
        std::sort(events.begin(), events.end(),
                  [](const FaceEvent& x, const FaceEvent& y) { return x.time < y.time; });
        double cum = 0.0, safe_time = max_time;
        for (const auto& e : events) {
            cum += e.weight / total;
            if (cum > 0.1) { safe_time = e.time; break; }
        }
        const int n = static_cast<int>(safe_time / epoch_duration);
        return std::max(1, std::min(n, max_epochs));
    }

    /**
     * Garden mode (PTM-669) super-step bound, the same 10%-of-emission rule as
     * computeSafeEpochCountLag with the garden ODE's rates: a 10% change of the
     * barrier factor (growing or gardened lag), the lag gardened away, the box of an
     * absorbing d = 0 face losing 10% of its ice (its c and albedo move), substrate
     * exhaustion, a veneer vanishing (not recycling faces: their emission is B). Globally the box depth may grow by <= 10%.
     */
    int computeSafeEpochCountGarden(const IceLayerModel& ice_model, const IceLayerState& st,
                                    const std::vector<double>& face_areas, double epoch_duration,
                                    int max_epochs) const {
        const DiurnalAverages& dv = *diurnal_avg_;
        const auto& lc = ice_model.getLagConfig();
        const double ell = ice_model.getLagBarrierLength();
        const double vm = lc.gardening ? lc.gardening_rate : 0.0, gv = gardenVeneerRate(ice_model);
        const bool have_B = step_buffers_.deposition.size() == num_faces_;
        struct FaceEvent { double time, weight; int kind; };
        std::vector<FaceEvent> events;
        events.reserve(num_faces_);
        double max_time = max_epochs * epoch_duration;
        if (lc.gardening && lc.box_K99 > 0.0) {
            const double h = lc.gardenBoxDepth(st.lag_time);
            max_time = std::min(max_time, std::max(epoch_duration, 0.21 * h * h / lc.box_K99));
        }
        std::vector<double> as_now, ap_now;
        lagAvailabilities(ice_model, st, dv.base_sublimation_flux, dv.base_sputtering_flux, as_now, ap_now);
        double total = 0.0;
        for (size_t i = 0; i < num_faces_; ++i) {
            const double Js = dv.base_sublimation_flux[i], Jp = dv.base_sputtering_flux[i];
            const double A = face_areas[i];
            total += (Js * as_now[i] + Jp * ap_now[i]) * A;
            const double B = have_B ? step_buffers_.deposition[i] : dv.avg_deposition_flux[i];
            const double w = dv.hasSputterSplit() ? dv.sput_w[i] : 1.0;
            const double M = st.substrate_ice[i], f = st.lag_exposed_fraction[i], d = st.lag_thickness[i];
            const double u = 1.0 + d / ell;
            const bool has = M > 0.0;
            const double a = has ? Js * f : 0.0;
            const double pw = has ? w * Jp * f : 0.0, pr = has ? (1.0 - w) * Jp * f : 0.0;
            const double ap = a + (lc.sputter_shield ? pw : 0.0);
            const double pu = pr + (lc.sputter_shield ? 0.0 : pw);
            const double ES = ap / u + pu, Vt = Js + Jp + gv;
            if (st.frost_veneer[i] > 0.0) {
                if (B < Vt) {
                    const double t = st.frost_veneer[i] / (Vt - B);
                    if (t < max_time) events.push_back({t, std::abs(Js + Jp - std::min(ES, B)) * A, 0});
                }
                continue;
            }
            const double beta = st.lag_beta[i];
            if (has && ES > B) {
                const double L = ES - B;
                const double t_dep = M / L;
                if (t_dep < max_time) events.push_back({t_dep, ES * A, 1});
                const double dd = beta * L - (d > 0.0 ? vm : 0.0);
                if (d <= 0.0 && dd <= 0.0) {
                    const double t = 0.1 * M / L;            // absorbing: the box composition moves
                    if (t < max_time) events.push_back({t, ES * A, 2});
                } else if (dd != 0.0 && beta > 0.0) {
                    const double t10 = 0.1 * u * ell / std::abs(dd);
                    if (t10 < max_time) events.push_back({t10, (ap / u) * A, dd > 0.0 ? 3 : 4});
                    if (dd < 0.0) {
                        const double t0 = d / (-dd);
                        if (t0 < max_time) events.push_back({t0, ES * A, 2});
                    }
                }
            }
            // Recycling faces emit B whatever their lag; gardening their lag changes only the
            // emission's split, and the hand-over to the substrate regime is continuous
            // (E_S = B there). No event.
        }
        if (!(total > 1e-30) || events.empty()) {
            return std::max(1, std::min(static_cast<int>(max_time / epoch_duration), max_epochs));
        }
        std::sort(events.begin(), events.end(),
                  [](const FaceEvent& x, const FaceEvent& y) { return x.time < y.time; });
        double cum = 0.0, safe_time = max_time;
        for (const auto& e : events) {
            cum += e.weight / total;
            if (cum > 0.1) { safe_time = e.time; break; }
        }
        if (std::getenv("PTM669_DEBUG_STEP")) {
            double wk[5] = {0, 0, 0, 0, 0};
            for (const auto& e : events) if (e.time <= safe_time) wk[e.kind] += e.weight / total;
            std::cerr << "\n[PTM669 step] safe " << safe_time / epoch_duration << " epochs (cap " << max_time / epoch_duration
                      << "); weight by kind veneer/deplete/absorb-or-lag-gone/barrier10-grow/barrier10-shrink: " << wk[0] << " " << wk[1]
                      << " " << wk[2] << " " << wk[3] << " " << wk[4] << std::endl;
        }
        const int n = static_cast<int>(safe_time / epoch_duration);
        return std::max(1, std::min(n, max_epochs));
    }

    /**
     * One face over one step with deposition held at B (PTM-644 helper).
     * conc_before: non-ice concentration at the start; A, B [1/s] as in
     * evolveEpoch(). mean_avail is the step-mean ice availability, so that
     * base_emission * mean_avail is the step-mean emission.
     */
    struct FaceStep {
        double nonice_new = 0.0;   // unclamped end-of-step concentration
        double mean_avail = 0.0;   // step-mean availability
        double net_rate = 0.0;     // d(nonice)/dt at the start [1/s]
    };
    FaceStep stepFace(const IceLayerModel& ice_model, double conc_before, double A, double B,
                      double dt, bool exact_linear_mean) const {
        FaceStep out;
        if (ice_model.usesExposedAreaFraction()) {
            // Exposed-area mode: d(phi)/dt = B - A f(phi), solved in closed form.
            const double phi0 = 1.0 - conc_before;
            const ExposedAreaStep st =
                solveExposedAreaStep(phi0, A, B, ice_model.getExposedAreaGrainRatio(), dt);
            out.nonice_new = 1.0 - st.phi_end;
            out.mean_avail = st.mean_f;
            out.net_rate = A * ice_model.exposedAreaFraction(phi0) - B;
            return out;
        }
        if (exact_linear_mean) {
            // Well-conditioned closed form (PTM-644). With f = 1 - nonice, y = A t and
            // rate0 = B - A f0 (df/dt at the start):
            //   f(t) - f0 = rate0 t phi1(y),               phi1(y) = (1 - e^-y) / y
            //   int_0^t f = t [f0 + rate0 t psi(y)],       psi(y) = (y - 1 + e^-y) / y^2
            // The legacy form nonice_eq - (nonice_eq - c0) e^-y loses ulp(B/A) when a
            // face barely emits (A t << 1, B/A >> 1: every cold receiving face), and
            // its trapezoid mean is not the mean of an exponential. f is clamped at 1
            // (pure ice; deposition beyond it overflows the layer).
            auto phi1 = [](double y) {
                return (y < 1e-8) ? 1.0 - 0.5 * y : -std::expm1(-y) / y;
            };
            auto psi = [](double y) {
                if (y < 1e-2) return 0.5 - y / 6.0 + y * y / 24.0 - y * y * y / 120.0 + y * y * y * y / 720.0;
                return (y + std::expm1(-y)) / (y * y);
            };
            const double f0 = 1.0 - conc_before;
            const double rate0 = B - A * f0;
            double nonice_new;
            if (A > 1e-30 && B <= 2.0 * A) {
                // |nonice_eq| <= 1: the legacy closed form is well conditioned; kept so
                // that faces at (or near) equilibrium evolve bit for bit as before.
                const double nonice_eq = 1.0 - B / A;
                nonice_new = nonice_eq - (nonice_eq - conc_before) * std::exp(-A * dt);
            } else {
                nonice_new = conc_before - rate0 * dt * phi1(A * dt);
            }
            double f_end = 1.0 - std::clamp(nonice_new, 0.0, 1.0);
            out.nonice_new = nonice_new;
            out.net_rate = A * (1.0 - conc_before) - B;
            if (rate0 == 0.0) {
                // Exact equilibrium: f is constant; the trapezoid of the end points keeps
                // such faces bit-identical to the legacy step. (An absolute |df| < 1e-8
                // switch here made the step mean discontinuous in B on nearly depleted
                // faces, f ~ 1e-6, and stalled the deposition solve; PTM-640 work.)
                out.mean_avail = 0.5 * (f0 + f_end);
                return out;
            }
            double integral;
            if (nonice_new >= 0.0) {
                integral = dt * (f0 + rate0 * dt * psi(A * dt));
            } else {
                // f reaches 1 at t1 (rate0 > 0 and B > A there), then stays at 1.
                const double t1 = (A > 0.0) ? -std::log1p(-A * (1.0 - f0) / rate0) / A
                                            : (1.0 - f0) / rate0;
                integral = t1 * (f0 + rate0 * t1 * psi(A * t1)) + (dt - t1);
            }
            const double lo = std::min(f0, f_end), hi = std::max(f0, f_end);
            out.mean_avail = std::clamp(integral / dt, lo, hi);
            return out;
        }
        if (A > 1e-30) {
            double nonice_eq = 1.0 - B / A;
            double decay = std::exp(-A * dt);
            out.nonice_new = nonice_eq - (nonice_eq - conc_before) * decay;
        } else {
            // No emission: pure deposition (linear)
            out.nonice_new = conc_before - B * dt;
        }
        double ice_frac_before = 1.0 - conc_before;
        double ice_frac_after = 1.0 - std::clamp(out.nonice_new, 0.0, 1.0);
        out.mean_avail = 0.5 * (ice_frac_before + ice_frac_after);
        out.net_rate = A * (1.0 - conc_before) - B;
        return out;
    }

    /**
     * Evolve non-ice concentration using analytical exponential solution
     *
     * Solves the ODE: d(nonice)/dt = A*(1-nonice) - B exactly, where
     * A = base_emission/max_ice_mass (emission rate constant) and
     * B = deposition/max_ice_mass (deposition rate). This provides
     * accurate evolution over arbitrarily long super-steps.
     *
     * Each face approaches equilibrium nonice_eq = 1 - B/A, where
     * sublimation (reduced by ice fraction) balances deposition.
     *
     * Deposition over the step (PTM-644). Deposition is the transport kernel's
     * output at cache time times a global scale per source group (s_sub, s_sput;
     * one group without sputter transport), because the kernel is linear in its
     * source and the pattern is fixed until the next full diurnal. Pre-fix, the
     * scale was frozen at its start-of-step value while every face's emission
     * followed its own availability trajectory, so a step deposited more than it
     * emitted whenever availability fell (Europa uniform-hapke phase 0, sputtering
     * off: +0.24% cumulative, 2.1e11 kg of ice from nothing; the lag halved with
     * each halving of the step). Now B is held at the scale whose deposition equals what the
     * kernel retains from the step's actual emission:
     *     s_k * S_k,base = sum_i a_i E_k,i <f_i>(B(s))      for each group k,
     * where <f_i> is the exact step-mean availability for that constant B. This
     * is the implicit (self-consistent) coupling of the source and its deposition
     * over the step; within the step the timing of deposition is approximated by
     * its mean, so the pattern error is O(the step's change in emission), which
     * computeSafeEpochCount() bounds, and it vanishes under step refinement
     * (--max-superstep-epochs). (Temperatures and the albedo-coupling refresh are
     * still evaluated only at step boundaries; that lag is PTM-640, not this.)
     * The mass balance is exact for every step:
     *     deposited_k = rho_k * emitted_k,  rho_k = D_k,base / S_k,base
     * (sublimation: 1 - Jeans - unscattered; sputter: non-escaping H2O share).
     * The scalar system is solved by Newton on s (1-2 unknowns; finite-difference
     * Jacobian; it is affine in s in the linear availability mode). All sums that
     * feed the solve are taken in face order, so the result is reproducible.
     * The step audit throws above 1e-6 relative. When the start-of-step scale is
     * already consistent (constant availability), it is kept bit for bit.
     *
     * @param ice_model Ice layer model
     * @param ice_state Ice layer state (modified in place)
     * @param face_areas Face areas [m²]
     * @param epoch_duration Duration of one epoch [s] (may be multiple epochs for super-stepping)
     */
    void evolveEpoch(const IceLayerModel& ice_model,
                     IceLayerState& ice_state,
                     const std::vector<double>& face_areas,
                     double epoch_duration) {
        evolveEpochImpl(ice_model, ice_state, face_areas, epoch_duration, nullptr, nullptr);
    }

    /**
     * PTM-640: the non-ice concentrations evolveEpoch() WOULD produce over
     * epoch_duration from the current state (same deposition solve and closed
     * form), without changing the state or the statistics.
     */
    void predictConcentrations(const IceLayerModel& ice_model,
                               const IceLayerState& ice_state,
                               const std::vector<double>& face_areas,
                               double epoch_duration,
                               std::vector<double>& nonice_out,
                               bool solve_deposition = true,
                               std::vector<double>* lag_thickness_out = nullptr) {
        // solve_deposition = false (PTM-657): hold deposition at its start-of-step
        // scale instead of solving the step-consistent scale; ~7x cheaper, and
        // adequate for the albedo step limiter, which only needs where the albedo
        // is heading (the step itself is still solved).
        predict_skip_solve_ = !solve_deposition;
        const EvolutionStatistics saved = stats_;
        nonice_out.resize(num_faces_);
        if (lag_thickness_out) lag_thickness_out->resize(num_faces_);
        evolveEpochImpl(ice_model, const_cast<IceLayerState&>(ice_state), face_areas, epoch_duration,
                        &nonice_out, lag_thickness_out);
        const long long evals = stats_.step_solver_evaluations - saved.step_solver_evaluations;
        stats_ = saved;
        stats_.prediction_evaluations += evals;
        ++stats_.predictions;
        predict_skip_solve_ = false;
    }

private:
    // ---- Garden mode (PTM-669) step pieces, kept out of evolveEpochImpl (see there). ----
    GardenFaceStep gardenStepAt(const IceLayerModel& m, const IceLayerState& st, size_t i, double B,
                                double dt) const {
        const DiurnalAverages& d = *diurnal_avg_;
        const auto& lc = m.getLagConfig();
        const double w = d.hasSputterSplit() ? d.sput_w[i] : 1.0;
        const double Jp = d.base_sputtering_flux[i];
        return solveGardenFaceStep(d.base_sublimation_flux[i], w * Jp, (1.0 - w) * Jp, B,
                                   st.lag_thickness[i], st.frost_veneer[i], st.substrate_ice[i],
                                   st.lag_exposed_fraction[i], st.lag_beta[i], m.getLagBarrierLength(),
                                   lc.gardening ? lc.gardening_rate : 0.0, gardenVeneerRate(m), dt, lc.sputter_shield);
    }
    // ---- PTM-676: garden-step subcycling --------------------------------------
    // Number of substeps face i needs over dt (1 = the plain single call).
    int gardenSubcycleCount(const IceLayerModel& m, const IceLayerState& st, size_t i, double B,
                            double dt, bool* capped = nullptr) const {
        if (capped) *capped = false;
        if (!(config_.garden_subcycle_kappa > 0.0)) return 1;
        const DiurnalAverages& d = *diurnal_avg_;
        const auto& lc = m.getLagConfig();
        const double w = d.hasSputterSplit() ? d.sput_w[i] : 1.0;
        const double Js = d.base_sublimation_flux[i], Jp = d.base_sputtering_flux[i], f = st.lag_exposed_fraction[i];
        const double ap = (Js + (lc.sputter_shield ? w * Jp : 0.0)) * f;
        const double pu = ((1.0 - w) * Jp + (lc.sputter_shield ? 0.0 : w * Jp)) * f;
        const double ES = ap / (1.0 + st.lag_thickness[i] / m.getLagBarrierLength()) + pu;
        double S = 0.0;
        if (lc.gardening && st.box_c0.size() == num_faces_ && dt > 0.0) {
            const double dh = lc.gardenBoxDepth(st.lag_time + dt) - lc.gardenBoxDepth(st.lag_time);
            S = (1.0 - st.box_c0[i]) * layer_constants::ICE_DENSITY * dh / dt;
        }
        return gardenSubcycleSubsteps(config_.garden_subcycle_kappa, config_.garden_subcycle_cap,
                                      st.substrate_ice[i], st.frost_veneer[i], ES, B, S, dt, capped);
    }
    GardenFaceStep gardenStepSubcycled(const IceLayerModel& m, const IceLayerState& st, size_t i, double B,
                                       double dt, int nsub, double& N_end, double& M_end) const {
        const DiurnalAverages& dv = *diurnal_avg_;
        const double w = dv.hasSputterSplit() ? dv.sput_w[i] : 1.0;
        const double Jp = dv.base_sputtering_flux[i];
        return solveGardenFaceSubcycled(m, dv.base_sublimation_flux[i], w * Jp, (1.0 - w) * Jp, B,
                                        st.lag_thickness[i], st.frost_veneer[i], st.substrate_ice[i],
                                        st.box_nonice_volume[i], st.lag_exposed_fraction[i], st.lag_beta[i],
                                        st.box_c0[i], st.lag_time, dt, nsub, N_end, M_end);
    }

    double gardenDepAt(size_t i, const double* sc, bool two) const {
        const DiurnalAverages& d = *diurnal_avg_;
        return two ? d.base_deposition_sublimation_flux[i] * sc[0] + d.base_deposition_sputter_flux[i] * sc[1]
                   : d.base_deposition_flux[i] * sc[0];
    }
    __attribute__((noinline)) void gardenEvaluate(const IceLayerModel& ice_model, const IceLayerState& ice_state,
                                                  const std::vector<double>& face_areas, double epoch_duration,
                                                  const double* sc, bool two, bool per_face) {
        const DiurnalAverages& d = *diurnal_avg_;
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const double B = gardenDepAt(i, sc, two);
            const int ns = gardenSubcycleCount(ice_model, ice_state, i, B, epoch_duration);
            double Nx, Mx;
            const GardenFaceStep st = (ns > 1)
                ? gardenStepSubcycled(ice_model, ice_state, i, B, epoch_duration, ns, Nx, Mx)
                : gardenStepAt(ice_model, ice_state, i, B, epoch_duration);
            const double g1 = (per_face && d.hasSputterSplit()) ? st.sputw_int * d.sput_tw[i]
                                                                : st.sputw_int + st.sputr_int;
            step_buffers_.emit_sub[i] = st.sub_int / epoch_duration * face_areas[i];
            step_buffers_.emit_sput[i] = g1 / epoch_duration * face_areas[i];
        }
    }
    __attribute__((noinline)) void gardenPredict(const IceLayerModel& ice_model, const IceLayerState& ice_state,
                                                 double epoch_duration, const double* s, bool two,
                                                 std::vector<double>& c_out, std::vector<double>* d_out) const {
        const double phi = ice_model.getLagConfig().lag_porosity;
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const double B = gardenDepAt(i, s, two);
            const int ns = gardenSubcycleCount(ice_model, ice_state, i, B, epoch_duration);
            if (ns > 1) {
                // The subcycled face's predicted c includes the step's entrainment: that is
                // its actual step (PTM-676). The single call below leaves it out as before;
                // for those faces the entrainment is < kappa of their box ice.
                double Nx, Mx;
                const GardenFaceStep sx = gardenStepSubcycled(ice_model, ice_state, i, B, epoch_duration, ns, Nx, Mx);
                double c, fx, bx;
                ice_model.boxCompositionOf(Nx, Mx, c, fx, bx);
                c_out[i] = c;
                if (d_out) (*d_out)[i] = sx.d_end;
                continue;
            }
            const GardenFaceStep st = gardenStepAt(ice_model, ice_state, i, B, epoch_duration);
            const double N = ice_state.box_nonice_volume[i] - (1.0 - phi) * (st.d_end - ice_state.lag_thickness[i]);
            const double V = std::max(0.0, N) + st.M_end / layer_constants::ICE_DENSITY;
            c_out[i] = (V > 0.0) ? std::clamp(std::max(0.0, N) / V, 0.0, 1.0) : 1.0;
            if (d_out) (*d_out)[i] = st.d_end;
        }
    }
    __attribute__((noinline)) void gardenApplyStep(const IceLayerModel& ice_model, IceLayerState& ice_state,
                                                   const std::vector<double>& face_areas, double epoch_duration,
                                                   const double* s, bool two, bool per_face, double& max_change) {
        const DiurnalAverages& d = *diurnal_avg_;
        auto dep_at = [&](size_t i, const double* sc) { return gardenDepAt(i, sc, two); };
        auto gStep = [&](size_t i, double B) { return gardenStepAt(ice_model, ice_state, i, B, epoch_duration); };
            if (lag_regime_.size() != num_faces_) lag_regime_.assign(num_faces_, 0);
            if (lag_change_.size() != num_faces_) lag_change_.assign(num_faces_, 0.0);
            if (garden_entrained_.size() != num_faces_) garden_entrained_.assign(num_faces_, 0.0);
            if (!stats_.lag_active) {
                double inv0 = 0.0;
                for (size_t i = 0; i < num_faces_; ++i) {
                    inv0 += (ice_state.substrate_ice[i] + ice_state.frost_veneer[i]) * face_areas[i];
                }
                stats_.lag_inventory_initial_mass = inv0;
                stats_.lag_active = true;
            }
            const auto& lc = ice_model.getLagConfig();
            const double t0 = ice_state.lag_time, t1 = t0 + epoch_duration;
            const double dh = lc.gardenBoxDepth(t1) - lc.gardenBoxDepth(t0);   // entrained column [m]
            const double phi = lc.lag_porosity, rho = layer_constants::ICE_DENSITY;
            long long sub_n = 0, sub_capped = 0;
            int sub_max = 0;
            #pragma omp parallel for reduction(+:sub_n,sub_capped) reduction(max:sub_max)
            for (size_t i = 0; i < num_faces_; ++i) {
                const double deposition = dep_at(i, s);
                const double avail_before = ice_model.getSublimationAvailability(ice_state, i);
                const double inv_before = ice_state.substrate_ice[i] + ice_state.frost_veneer[i];
                const double d_before = ice_state.lag_thickness[i];
                // PTM-676: a face whose box turns over within the step is integrated in substeps
                // (entrainment and box composition inside the step); every other face takes the
                // single call exactly as before.
                bool capped = false;
                const int ns = gardenSubcycleCount(ice_model, ice_state, i, deposition, epoch_duration, &capped);
                double N_sub = 0.0, M_sub = 0.0;
                const GardenFaceStep st = (ns > 1)
                    ? gardenStepSubcycled(ice_model, ice_state, i, deposition, epoch_duration, ns, N_sub, M_sub)
                    : gStep(i, deposition);
                if (ns > 1) { ++sub_n; sub_max = std::max(sub_max, ns); if (capped) ++sub_capped; }
                const double sput_int = st.sputw_int + st.sputr_int;
                ice_state.lag_thickness[i] = st.d_end;
                ice_state.frost_veneer[i] = st.mf_end;
                // Box: non-ice moves between box and lag with the lag; gardening entrains
                // column material at the t = 0 composition c_0 (external supply, booked).
                const double c0 = ice_state.box_c0[i];
                const double entrained = (1.0 - c0) * rho * dh;
                if (ns > 1) {
                    ice_state.box_nonice_volume[i] = N_sub;
                    ice_state.substrate_ice[i] = M_sub;
                } else {
                    ice_state.box_nonice_volume[i] = std::max(0.0,
                        ice_state.box_nonice_volume[i] - (1.0 - phi) * (st.d_end - d_before)) + c0 * dh;
                    ice_state.substrate_ice[i] = st.M_end + entrained;
                }
                garden_entrained_[i] = entrained * face_areas[i];
                ice_model.updateBoxComposition(ice_state, i);
                ice_model.updateIceMass(ice_state, i);
                lag_regime_[i] = st.absorbing ? 4 : st.regime_end;
                lag_change_[i] = std::abs(ice_model.getSublimationAvailability(ice_state, i) - avail_before);
                concentration_change_rate_[i] = 0.0;
                ice_state.cumulative_sublimated[i] += st.sub_int;
                ice_state.cumulative_sputtered[i] += sput_int;
                ice_state.cumulative_deposited[i] += deposition * epoch_duration;
                step_buffers_.deposition[i] = deposition;
                step_buffers_.emit_sub[i] = st.sub_int / epoch_duration * face_areas[i];
                step_buffers_.emit_sput[i] = sput_int / epoch_duration * face_areas[i];
                if (per_face) {
                    const double tw = d.sput_tw[i], ew = d.sput_ew[i], er = d.sput_er[i];
                    const double k = face_areas[i] / epoch_duration;
                    step_buffers_.sput_transport[i] = st.sputw_int * tw * k;
                    step_buffers_.sput_escape[i] = (st.sputw_int * ew + st.sputr_int * er) * k;
                    step_buffers_.sput_reservoir[i] = (st.sputw_int * (1.0 - tw - ew) + st.sputr_int * (1.0 - er)) * k;
                }
                step_buffers_.dep_sub[i] = (two ? d.base_deposition_sublimation_flux[i] * s[0] : deposition)
                                           * face_areas[i];
                step_buffers_.dep_sput[i] = two ? deposition * face_areas[i] - step_buffers_.dep_sub[i] : 0.0;
                step_buffers_.overflow[i] = 0.0;
                const double booked = deposition * epoch_duration - st.sub_int - sput_int + entrained;
                const double inv_after = ice_state.substrate_ice[i] + ice_state.frost_veneer[i];
                step_buffers_.inventory_err[i] = std::abs((inv_after - inv_before) - booked) * face_areas[i];
            }
            ice_state.lag_time = t1;
            stats_.garden_subcycled_face_steps += sub_n;
            stats_.garden_subcycle_capped += sub_capped;
            stats_.garden_subcycle_max = std::max(stats_.garden_subcycle_max, sub_max);
            double inv = 0.0, ven = 0.0, dmax = 0.0, ent = 0.0;
            size_t over = 0;
            size_t rc[4] = {0, 0, 0, 0};
            size_t absorbing = 0;
            const double warn = lc.warn_thickness;
            for (size_t i = 0; i < num_faces_; ++i) {
                inv += (ice_state.substrate_ice[i] + ice_state.frost_veneer[i]) * face_areas[i];
                ven += ice_state.frost_veneer[i] * face_areas[i];
                ent += garden_entrained_[i];
                dmax = std::max(dmax, ice_state.lag_thickness[i]);
                if (ice_state.lag_thickness[i] > warn) ++over;
                if (lag_regime_[i] == 4) ++absorbing;
                ++rc[lag_regime_[i] == 4 ? 0 : std::clamp(lag_regime_[i], 0, 3)];
                max_change = std::max(max_change, lag_change_[i]);
            }
            stats_.lag_inventory_mass = inv;
            stats_.lag_veneer_mass = ven;
            stats_.lag_max_thickness = dmax;
            stats_.lag_faces_over_warn = over;
            stats_.garden_entrained_mass += ent;
            stats_.garden_absorbing_faces = absorbing;
            for (int k = 0; k < 4; ++k) stats_.lag_regime_count[k] = rc[k];
    }

    void evolveEpochImpl(const IceLayerModel& ice_model,
                         IceLayerState& ice_state,
                         const std::vector<double>& face_areas,
                         double epoch_duration,
                         std::vector<double>* predict_only,
                         std::vector<double>* predict_lag_d) {
        const DiurnalAverages& d = *diurnal_avg_;
        // Lag barrier (IceLayerModel::enableLagBarrier): the per-face step is
        // solveLagFaceStep, with separate sublimation and sputter emission.
        const bool lag = lagOn(ice_model, ice_state);
        const double ell = lag ? ice_model.getLagBarrierLength() : 0.0;
        if (lag && !config_.superstep_consistent_deposition) {
            throw std::logic_error("evolveEpoch: the lag barrier requires the step-consistent deposition");
        }
        // Garden mode (PTM-669): solveGardenFaceStep, the sputter split and the box.
        const bool garden = lag && ice_model.usesGardenMode();
        if (garden && d.sputter_transport && !d.perFaceSputter()) {
            throw std::logic_error("evolveEpoch: garden mode needs the per-face sputter composition "
                                   "(current parameter set); --sputter-legacy-params is not supported");
        }
        auto lagStep = [&](size_t i, double B) {
            return solveLagFaceStep(d.base_sublimation_flux[i], d.base_sputtering_flux[i], B,
                                    ice_state.lag_thickness[i], ice_state.frost_veneer[i],
                                    ice_state.substrate_ice[i], ice_state.lag_exposed_fraction[i],
                                    ice_state.lag_beta[i], ell, epoch_duration);
        };
        double max_ice_mass = ice_model.getMaxIceMassPerArea();
        const bool consistent = config_.superstep_consistent_deposition;
        const bool two = d.sputter_transport;
        double max_change = 0.0;
        double epoch_sublimated = 0.0, epoch_deposited = 0.0, epoch_sputtered = 0.0;
        step_buffers_.resize(num_faces_);

        // Source groups: 0 = sublimation (+ sputtering when it is not transported), 1 = sputter.
        // Group 1 is the sputter transport source: the gross, or (per-face composition,
        // PTM-649) the per-face transported share of it.
        const bool per_face = two && d.perFaceSputter();
        const double S_base[2] = {two ? d.total_base_sublimation : d.total_base_emission,
                                  two ? (per_face ? d.total_base_sputter_transport : d.total_base_sputter) : 0.0};
        auto w1 = [&](size_t i) { return per_face ? d.sputter_face_transport_frac[i] : 1.0; };
        const double D_base[2] = {two ? d.total_base_deposition_sub : d.total_base_deposition,
                                  two ? d.total_base_deposition_sput : 0.0};
        const bool active[2] = {S_base[0] > 1e-30, two && S_base[1] > 1e-30};
        double s[2] = {two ? d.deposition_scale_sub : d.deposition_scale_total,
                       two ? d.deposition_scale_sput : 0.0};
        auto dep_at = [&](size_t i, const double* sc) {
            return two ? d.base_deposition_sublimation_flux[i] * sc[0] + d.base_deposition_sputter_flux[i] * sc[1]
                       : d.base_deposition_flux[i] * sc[0];
        };

        // Step-mean emission per group [kg/s] for deposition scales sc (face-order sums).
        auto evaluate = [&](const double* sc, double* G) {
            if (garden) {
                gardenEvaluate(ice_model, ice_state, face_areas, epoch_duration, sc, two, per_face);
            } else if (lag) {
                #pragma omp parallel for
                for (size_t i = 0; i < num_faces_; ++i) {
                    const LagFaceStep st = lagStep(i, dep_at(i, sc));
                    step_buffers_.emit_sub[i] = st.sub_int / epoch_duration * face_areas[i];
                    step_buffers_.emit_sput[i] = st.sput_int / epoch_duration * w1(i) * face_areas[i];
                }
            } else {
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                const double bs = d.base_sublimation_flux[i], bp = d.base_sputtering_flux[i];
                const FaceStep st = stepFace(ice_model, ice_state.nonice_concentration[i],
                                             (bs + bp) / max_ice_mass, dep_at(i, sc) / max_ice_mass,
                                             epoch_duration, true);
                step_buffers_.emit_sub[i] = bs * st.mean_avail * face_areas[i];
                step_buffers_.emit_sput[i] = bp * w1(i) * st.mean_avail * face_areas[i];
            }
            }
            G[0] = G[1] = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) G[0] += step_buffers_.emit_sub[i];
            for (size_t i = 0; i < num_faces_; ++i) G[1] += step_buffers_.emit_sput[i];
            if (!two) { G[0] += G[1]; G[1] = 0.0; }
            ++stats_.step_solver_evaluations;
        };
        auto residual = [&](const double* G, const double* sc, double* g) {
            for (int k = 0; k < 2; ++k) g[k] = active[k] ? G[k] / S_base[k] - sc[k] : 0.0;
        };
        auto converged_at = [&](const double* g, const double* sc) {
            for (int k = 0; k < 2; ++k) {
                if (active[k] && !(std::abs(g[k]) <= 1e-12 * std::max(sc[k], 1e-6))) return false;
            }
            return true;
        };
        auto gnorm = [](const double* g) { return std::max(std::abs(g[0]), std::abs(g[1])); };

        // The kernel cannot retain more than its source: rho_sub <= 1 (Jeans and
        // unscattered losses only), rho_sput <= the non-escaping H2O share.
        if (consistent) {
            const double rho_max[2] = {1.0, (two && !per_face) ? d.sputter_transport_share : 1.0};
            for (int k = 0; k < 2; ++k) {
                if (active[k] && D_base[k] > S_base[k] * rho_max[k] * (1.0 + 1e-6)) {
                    std::ostringstream m;
                    m << std::scientific << std::setprecision(6) << "evolveEpoch: cached deposition "
                      << D_base[k] << " kg/s exceeds its source's retainable share " << S_base[k] * rho_max[k]
                      << " kg/s (group " << k << ")";
                    throw std::runtime_error(m.str());
                }
            }
        }

        int iterations = 0;
        if (consistent && (active[0] || active[1]) && !(predict_only && predict_skip_solve_)) {
            double G[2], g[2];
            evaluate(s, G);
            residual(G, s, g);
            while (!converged_at(g, s)) {
                // Stagnation: the scale no longer moves (a kink at the root). Accept
                // when the residual is far inside the 1e-6 step audit, count it.
                if (iterations >= 8 && gnorm(g) <= 1e-8) {
                    ++stats_.step_solver_loose;
                    break;
                }
                if (++iterations > 40) {
                    std::ostringstream m;
                    m << std::scientific << std::setprecision(6)
                      << "evolveEpoch: step deposition scale did not converge in 40 Newton iterations"
                      << " (s = " << s[0] << ", " << s[1] << "; residual " << g[0] << ", " << g[1] << ")";
                    throw std::runtime_error(m.str());
                }
                // Finite-difference Jacobian of g (inactive groups: identity, g = 0).
                double J[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
                for (int k = 0; k < 2; ++k) {
                    if (!active[k]) continue;
                    double sp[2] = {s[0], s[1]}, Gp[2], gp[2];
                    const double h = 1e-6 * std::max(s[k], 1e-3);
                    sp[k] += h;
                    evaluate(sp, Gp);
                    residual(Gp, sp, gp);
                    for (int j = 0; j < 2; ++j) J[j][k] = active[j] ? (gp[j] - g[j]) / h : (j == k ? 1.0 : 0.0);
                }
                // g_k = G_k/S_k - s_k decreases in s_k with slope in [-1, 0) (more
                // deposition, more ice, more emission, but never more than deposited).
                // A finite difference across a kink in the per-face closed forms can
                // violate that; fall back to the fixed-point slope -1 for that group.
                for (int k = 0; k < 2; ++k) {
                    if (active[k] && !(J[k][k] < 0.0 && J[k][k] >= -1.0 - 1e-6)) {
                        J[k][k] = -1.0;
                        J[k][1 - k] = 0.0;
                        ++stats_.step_solver_slope_fallbacks;
                    }
                }
                const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
                    throw std::runtime_error("evolveEpoch: singular Jacobian in the step deposition solve");
                }
                const double dx0 = (-g[0] * J[1][1] + g[1] * J[0][1]) / det;
                const double dx1 = (-g[1] * J[0][0] + g[0] * J[1][0]) / det;
                // Damped update (g is monotone and nearly affine; damping only guards clamps).
                double lambda = 1.0, sn[2], Gn[2], gn[2];
                for (int ls = 0; ; ++ls) {
                    sn[0] = active[0] ? std::max(0.0, s[0] + lambda * dx0) : s[0];
                    sn[1] = active[1] ? std::max(0.0, s[1] + lambda * dx1) : s[1];
                    evaluate(sn, Gn);
                    residual(Gn, sn, gn);
                    if (gnorm(gn) < gnorm(g) || converged_at(gn, sn) || ls >= 30) break;
                    lambda *= 0.5;
                }
                s[0] = sn[0]; s[1] = sn[1]; g[0] = gn[0]; g[1] = gn[1];
            }
        }
        if (predict_only && garden) {
            gardenPredict(ice_model, ice_state, epoch_duration, s, two, *predict_only, predict_lag_d);
            return;
        }
        if (predict_only && lag) {
            // Prediction: the end-of-step lag thickness (the concentration is frozen).
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                (*predict_only)[i] = ice_state.nonice_concentration[i];
                if (predict_lag_d) (*predict_lag_d)[i] = lagStep(i, dep_at(i, s)).d_end;
            }
            return;
        }
        if (predict_only) {
            // Prediction: the end-of-step concentrations only (state untouched).
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                const double deposition = consistent ? dep_at(i, s) : d.avg_deposition_flux[i];
                const double A = (d.base_sublimation_flux[i] + d.base_sputtering_flux[i]) / max_ice_mass;
                const FaceStep st = stepFace(ice_model, ice_state.nonice_concentration[i], A,
                                             deposition / max_ice_mass, epoch_duration, consistent);
                (*predict_only)[i] = std::clamp(st.nonice_new, 0.0, layer_constants::MAX_NONICE_CONCENTRATION);
            }
            return;
        }
        stats_.step_solver_max_iterations = std::max(stats_.step_solver_max_iterations, iterations);
        ++stats_.steps_evolved;

        if (garden) {
            // Separate, non-inlined function: keeps the codegen (and so the -ffast-math
            // rounding) of the PTM-659 lag branch below identical to 254bb03.
            gardenApplyStep(ice_model, ice_state, face_areas, epoch_duration, s, two, per_face, max_change);
        } else if (lag) {
            if (lag_regime_.size() != num_faces_) lag_regime_.assign(num_faces_, 0);
            if (lag_change_.size() != num_faces_) lag_change_.assign(num_faces_, 0.0);
            if (!stats_.lag_active) {
                double inv0 = 0.0;
                for (size_t i = 0; i < num_faces_; ++i) {
                    inv0 += (ice_state.substrate_ice[i] + ice_state.frost_veneer[i]) * face_areas[i];
                }
                stats_.lag_inventory_initial_mass = inv0;
                stats_.lag_active = true;
            }
            #pragma omp parallel for
            for (size_t i = 0; i < num_faces_; ++i) {
                const double deposition = dep_at(i, s);
                const double avail_before = ice_model.getSublimationAvailability(ice_state, i);
                const double inv_before = ice_state.substrate_ice[i] + ice_state.frost_veneer[i];
                const LagFaceStep st = lagStep(i, deposition);
                const double eff_sublimation = st.sub_int / epoch_duration;
                const double eff_sputtering = st.sput_int / epoch_duration;
                ice_state.lag_thickness[i] = st.d_end;
                ice_state.frost_veneer[i] = st.mf_end;
                ice_state.substrate_ice[i] = st.M_end;
                ice_model.updateIceMass(ice_state, i);
                lag_regime_[i] = st.regime_end;
                lag_change_[i] = std::abs(ice_model.getSublimationAvailability(ice_state, i) - avail_before);
                concentration_change_rate_[i] = 0.0;   // the concentration is frozen
                ice_state.cumulative_sublimated[i] += st.sub_int;
                ice_state.cumulative_sputtered[i] += st.sput_int;
                ice_state.cumulative_deposited[i] += deposition * epoch_duration;
                step_buffers_.deposition[i] = deposition;
                step_buffers_.emit_sub[i] = eff_sublimation * face_areas[i];
                step_buffers_.emit_sput[i] = eff_sputtering * face_areas[i];
                if (per_face) {
                    step_buffers_.sput_transport[i] = eff_sputtering * d.sputter_face_transport_frac[i] * face_areas[i];
                    step_buffers_.sput_escape[i] = eff_sputtering * d.sputter_face_escape_frac[i] * face_areas[i];
                    step_buffers_.sput_reservoir[i] = eff_sputtering * d.sputter_face_reservoir_frac[i] * face_areas[i];
                }
                step_buffers_.dep_sub[i] = (two ? d.base_deposition_sublimation_flux[i] * s[0] : deposition)
                                           * face_areas[i];
                step_buffers_.dep_sput[i] = two ? deposition * face_areas[i] - step_buffers_.dep_sub[i] : 0.0;
                // Deposition the substrate cannot take goes to the veneer: no overflow.
                step_buffers_.overflow[i] = 0.0;
                const double booked = deposition * epoch_duration - st.sub_int - st.sput_int;
                const double inv_after = ice_state.substrate_ice[i] + ice_state.frost_veneer[i];
                step_buffers_.inventory_err[i] = std::abs((inv_after - inv_before) - booked) * face_areas[i];
            }
            double inv = 0.0, ven = 0.0, dmax = 0.0;
            size_t over = 0;
            size_t rc[4] = {0, 0, 0, 0};
            const double warn = ice_model.getLagConfig().warn_thickness;
            for (size_t i = 0; i < num_faces_; ++i) {
                inv += (ice_state.substrate_ice[i] + ice_state.frost_veneer[i]) * face_areas[i];
                ven += ice_state.frost_veneer[i] * face_areas[i];
                dmax = std::max(dmax, ice_state.lag_thickness[i]);
                if (ice_state.lag_thickness[i] > warn) ++over;
                ++rc[std::clamp(lag_regime_[i], 0, 3)];
                max_change = std::max(max_change, lag_change_[i]);
            }
            stats_.lag_inventory_mass = inv;
            stats_.lag_veneer_mass = ven;
            stats_.lag_max_thickness = dmax;
            stats_.lag_faces_over_warn = over;
            for (int k = 0; k < 4; ++k) stats_.lag_regime_count[k] = rc[k];
        } else {
        #pragma omp parallel for reduction(max:max_change) \
            reduction(+:epoch_sublimated,epoch_deposited,epoch_sputtered)
        for (size_t i = 0; i < num_faces_; ++i) {
            double conc_before = ice_state.nonice_concentration[i];

            // Base (unscaled) emission rates and deposition
            double base_sublim = d.base_sublimation_flux[i];
            double base_sput = d.base_sputtering_flux[i];
            double deposition = consistent ? dep_at(i, s) : d.avg_deposition_flux[i];

            // Analytical exponential evolution:
            // The ODE is: d(nonice)/dt = [base_emission * (1-nonice) - deposition] / max_ice_mass
            // where sublimation/sputtering scale with ice fraction (1-nonice).
            // Let A = (base_sublim + base_sput) / max_ice_mass  [emission rate constant]
            //     B = deposition / max_ice_mass                  [deposition rate]
            // ODE: d(nonice)/dt = A*(1-nonice) - B = A - B - A*nonice
            // Solution: nonice(t) = nonice_eq - (nonice_eq - nonice_0) * exp(-A*t)
            //   where nonice_eq = 1 - B/A  (equilibrium concentration)
            // (Exposed-area mode: d(phi)/dt = B - A f(phi); see stepFace().)
            double A = (base_sublim + base_sput) / max_ice_mass;
            double B = deposition / max_ice_mass;

            const FaceStep st = stepFace(ice_model, conc_before, A, B, epoch_duration, consistent);
            const double nonice_new = st.nonice_new;
            const double eff_sublimation = base_sublim * st.mean_avail;
            const double eff_sputtering = base_sput * st.mean_avail;

            // Accumulate area-weighted mass balance
            epoch_sublimated += eff_sublimation * face_areas[i];
            epoch_deposited += deposition * face_areas[i];
            epoch_sputtered += eff_sputtering * face_areas[i];

            // Store concentration change rate (instantaneous, for super-step estimation)
            concentration_change_rate_[i] = st.net_rate;

            // Clamp to valid range [0, 1]
            ice_state.nonice_concentration[i] = std::clamp(nonice_new, 0.0,
                layer_constants::MAX_NONICE_CONCENTRATION);

            // Update ice mass from concentration
            ice_model.updateIceMass(ice_state, i);

            // Track cumulative fluxes
            ice_state.cumulative_sublimated[i] += eff_sublimation * epoch_duration;
            ice_state.cumulative_sputtered[i] += eff_sputtering * epoch_duration;
            ice_state.cumulative_deposited[i] += deposition * epoch_duration;

            // Track maximum change
            double change = std::abs(ice_state.nonice_concentration[i] - conc_before);
            max_change = std::max(max_change, change);

            // PTM-644 audit terms, summed in face order below.
            step_buffers_.deposition[i] = deposition;
            step_buffers_.emit_sub[i] = eff_sublimation * face_areas[i];
            step_buffers_.emit_sput[i] = eff_sputtering * face_areas[i];
            if (per_face) {
                step_buffers_.sput_transport[i] = eff_sputtering * d.sputter_face_transport_frac[i] * face_areas[i];
                step_buffers_.sput_escape[i] = eff_sputtering * d.sputter_face_escape_frac[i] * face_areas[i];
                step_buffers_.sput_reservoir[i] = eff_sputtering * d.sputter_face_reservoir_frac[i] * face_areas[i];
            }
            step_buffers_.dep_sub[i] = (two ? (consistent ? d.base_deposition_sublimation_flux[i] * s[0]
                                                          : d.avg_deposition_sublimation_flux[i])
                                            : deposition) * face_areas[i];
            step_buffers_.dep_sput[i] = two ? deposition * face_areas[i] - step_buffers_.dep_sub[i] : 0.0;
            // Ice inventory: the state change must equal (deposition - emission) x dt,
            // except deposition that overflows a pure-ice face.
            const double dphi = conc_before - ice_state.nonice_concentration[i];
            const double booked = (B - A * st.mean_avail) * epoch_duration;
            const bool pure_ice_end = (ice_state.nonice_concentration[i] <= 0.0);
            const double over = (pure_ice_end && booked > dphi) ? booked - dphi : 0.0;
            step_buffers_.overflow[i] = over * max_ice_mass * face_areas[i];
            step_buffers_.inventory_err[i] = std::abs(booked - over - dphi) * max_ice_mass * face_areas[i];
        }
        }

        // Face-order (reproducible) sums of the step.
        // Gs, Gp, Ds, Dp [kg/s]; over, inv_err [kg] over the step.
        double Gs = 0.0, Gp = 0.0, Ds = 0.0, Dp = 0.0, over = 0.0, inv_err = 0.0;
        double Gc = 0.0, Ge = 0.0, Gr = 0.0;  // per-face sputter fates [kg/s] (PTM-649)
        if (per_face) {
            for (size_t i = 0; i < num_faces_; ++i) {
                Gc += step_buffers_.sput_transport[i];
                Ge += step_buffers_.sput_escape[i];
                Gr += step_buffers_.sput_reservoir[i];
            }
        }
        for (size_t i = 0; i < num_faces_; ++i) {
            Gs += step_buffers_.emit_sub[i];
            Gp += step_buffers_.emit_sput[i];
            Ds += step_buffers_.dep_sub[i];
            Dp += step_buffers_.dep_sput[i];
            over += step_buffers_.overflow[i];
            inv_err += step_buffers_.inventory_err[i];
        }
        if (consistent) {   // (the lag barrier requires it; see the top of this function)
            // Stats from the reproducible sums (the OpenMP reductions above vary at
            // ~1e-16 run to run; PTM-645).
            epoch_sublimated = Gs;
            epoch_sputtered = Gp;
            epoch_deposited = Ds + Dp;
        }

        // Step conservation audit (PTM-644): deposited = retained share of what
        // the step emitted, per group.
        const double rho0 = active[0] ? D_base[0] / S_base[0] : 0.0;
        const double rho1 = active[1] ? D_base[1] / S_base[1] : 0.0;
        const double emitted = Gs + Gp;
        const double retained = two ? rho0 * Gs + rho1 * (per_face ? Gc : Gp) : rho0 * emitted;
        // A group with deposition but no source at cache time (only in synthetic
        // setups: the kernel conserves its source) is an external supply; it is
        // reported separately and excluded from the audit, never silently.
        const double external = ((!active[0] && D_base[0] > 0.0) ? Ds : 0.0) +
                                ((!active[1] && D_base[1] > 0.0) ? Dp : 0.0);
        if (external > 0.0 && stats_.total_external_deposition_mass == 0.0) {
            std::cerr << "WARNING: deposition with no cached source (external supply, "
                      << std::scientific << std::setprecision(3) << external
                      << " kg/s); it is excluded from the step conservation audit and reported as "
                      << "external deposition" << std::fixed << std::endl;
        }
        stats_.total_external_deposition_mass += external * epoch_duration;
        const double deposited = Ds + Dp - external;
        const double step_residual = deposited - retained;
        const double step_rel = (emitted > 0.0) ? step_residual / emitted
                              : (deposited > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
        const double inv_rel = (emitted > 0.0) ? inv_err / (emitted * epoch_duration) : 0.0;
        stats_.superstep_consistent = consistent;
        stats_.last_step_residual_rel = step_rel;
        stats_.max_step_residual_rel = std::max(stats_.max_step_residual_rel, std::abs(step_rel));
        stats_.max_inventory_error_rel = std::max(stats_.max_inventory_error_rel, inv_rel);
        stats_.total_step_residual_mass += step_residual * epoch_duration;
        stats_.total_not_redeposited_mass += (emitted - retained) * epoch_duration;
        stats_.total_layer_overflow_mass += over;      // already [kg] over the step
        stats_.total_inventory_error_mass += inv_err;  // already [kg] over the step
        stats_.last_step_scale_sub = s[0];
        stats_.last_step_scale_sput = s[1];
        stats_.step_audit_count++;
        if (consistent && !(std::abs(step_rel) <= 1e-6)) {
            std::ostringstream m;
            m << std::scientific << std::setprecision(6)
              << "evolveEpoch: step conservation audit failed: deposited " << deposited
              << " kg/s vs retained emission " << retained << " kg/s (emitted " << emitted
              << "), residual " << step_rel << " relative (limit 1e-6)";
            throw std::runtime_error(m.str());
        }

        // Update statistics
        stats_.max_concentration_change = max_change;
        stats_.total_sublimated_mass += epoch_sublimated * epoch_duration;
        stats_.total_deposited_mass += epoch_deposited * epoch_duration;
        stats_.total_sputtered_mass += epoch_sputtered * epoch_duration;

        // Net surface mass loss = sublimation + sputtering - deposition. With the PTM-642
        // accounting this is escape + atmospheric reservoir (+ Jeans), minus the super-step lag.
        double epoch_net_loss = epoch_sublimated + epoch_sputtered - epoch_deposited;
        stats_.total_escaped_mass += std::max(0.0, epoch_net_loss) * epoch_duration;

        // Mass balance rates (for convergence diagnostics)
        stats_.sublimation_rate = epoch_sublimated;
        stats_.deposition_rate = epoch_deposited;
        stats_.sputtering_rate = epoch_sputtered;
        stats_.net_mass_loss_rate = epoch_net_loss;

        double total_emission = epoch_sublimated + epoch_sputtered;
        stats_.flux_imbalance_fraction = (total_emission > 1e-30)
            ? std::abs(total_emission - epoch_deposited) / total_emission
            : 0.0;

        if (d.sputter_transport && consistent) {
            // Deposition is the retained share of the step's own sources, so the
            // Jeans term is (1 - rho_sub) x sublimated and the lag is the audit residual.
            stats_.total_sublimation_jeans_mass += (1.0 - rho0) * Gs * epoch_duration;
            stats_.total_superstep_lag_mass += step_residual * epoch_duration;
            stats_.sputter_redeposited_rate = Dp;
            stats_.sublimation_deposition_rate = Ds;
            stats_.sputter_escaped_rate = per_face ? Ge : d.sputter_escape_share * epoch_sputtered;
            stats_.sputter_reservoir_rate = per_face ? Gr : d.sputter_reservoir_share * epoch_sputtered;
            stats_.total_sputter_redeposited_mass += Dp * epoch_duration;
            stats_.total_sputter_escaped_mass += stats_.sputter_escaped_rate * epoch_duration;
            stats_.total_sputter_reservoir_mass += stats_.sputter_reservoir_rate * epoch_duration;
        } else if (d.sputter_transport) {
            // Legacy (pre-PTM-644) diagnostics, unchanged. Separate pass so the reductions above are untouched.
            double dep_sput = 0.0, dep_sub = 0.0, s0 = 0.0;
            for (size_t i = 0; i < num_faces_; ++i) {
                dep_sput += d.avg_deposition_sputter_flux[i] * face_areas[i];
                dep_sub += d.avg_deposition_sublimation_flux[i] * face_areas[i];
                s0 += d.avg_sublimation_flux[i] * face_areas[i];  // start of step
            }
            // Deposition from sublimation had the step followed the step-mean source.
            const double dep_sub_mean = (s0 > 0.0) ? dep_sub * (epoch_sublimated / s0) : 0.0;
            const double lag_rate = (dep_sub - dep_sub_mean)
                + (dep_sput - d.sputter_transport_share * epoch_sputtered);
            stats_.total_sublimation_jeans_mass += (epoch_sublimated - dep_sub_mean) * epoch_duration;
            stats_.total_superstep_lag_mass += lag_rate * epoch_duration;
            stats_.sputter_redeposited_rate = dep_sput;
            stats_.sublimation_deposition_rate = dep_sub;
            stats_.sputter_escaped_rate = d.sputter_escape_share * epoch_sputtered;
            stats_.sputter_reservoir_rate = d.sputter_reservoir_share * epoch_sputtered;
            stats_.total_sputter_redeposited_mass += dep_sput * epoch_duration;
            stats_.total_sputter_escaped_mass += stats_.sputter_escaped_rate * epoch_duration;
            stats_.total_sputter_reservoir_mass += stats_.sputter_reservoir_rate * epoch_duration;
        }

        // Multi-criteria convergence:
        // (a) Concentration change below threshold, OR
        // (b) Flux imbalance small AND concentration change rate is low
        stats_.converged = (max_change < config_.convergence_threshold)
            || (stats_.flux_imbalance_fraction < config_.flux_convergence_threshold
                && max_change < 1e-4);
    }

public:
    /**
     * Deposition flux [kg/(m²·s)] that the last evolveEpoch() step used, per face
     * (the step-consistent value; the start-of-step value in legacy mode).
     */
    const std::vector<double>& getStepDepositionFlux() const { return step_buffers_.deposition; }

    /** Lag barrier: regime of each face at the end of the last step (empty before the first). */
    const std::vector<int>& getLagRegime() const { return lag_regime_; }

    /**
     * Update statistics after epoch evolution
     * @param epoch_count Number of epochs advanced (for super-stepping, default 1)
     */
    void updateStatistics(const IceLayerModel& ice_model,
                          const IceLayerState& ice_state,
                          double epoch_duration,
                          int epoch_count = 1) {
        stats_.current_epoch += epoch_count;
        stats_.elapsed_time_years += (epoch_duration * epoch_count) / timescale_constants::SECONDS_PER_YEAR;

        double max_conc = 0.0, total_conc = 0.0;
        size_t with_ice = 0, depleted = 0;

        for (size_t i = 0; i < num_faces_; ++i) {
            double conc = ice_state.nonice_concentration[i];
            max_conc = std::max(max_conc, conc);
            total_conc += conc;

            if (ice_model.hasIce(ice_state, i)) {
                with_ice++;
            } else {
                depleted++;
            }
        }

        stats_.max_nonice_concentration = max_conc;
        stats_.mean_nonice_concentration = total_conc / num_faces_;
        stats_.faces_with_ice = with_ice;
        stats_.faces_depleted = depleted;
    }

    /**
     * Get current statistics
     */
    const EvolutionStatistics& getStatistics() const { return stats_; }

    /**
     * Get diurnal averages (for export/analysis)
     */
    const DiurnalAverages& getDiurnalAverages() const { return *diurnal_avg_; }
    DiurnalAverages& getMutableDiurnalAverages() { return *diurnal_avg_; }

    /**
     * Activate the PTM-642 sputter mass accounting (call before the first
     * epoch). Shares are fractions of the gross sputtered mass and must sum to 1.
     */
    void setSputterPartition(double transport_share, double escape_share, double reservoir_share) {
        const double sum = transport_share + escape_share + reservoir_share;
        if (!(std::abs(sum - 1.0) <= 1e-12) || transport_share < 0.0 || escape_share < 0.0 ||
            reservoir_share < 0.0) {
            throw std::invalid_argument("setSputterPartition: shares must be >= 0 and sum to 1 (sum = " +
                                        std::to_string(sum) + ")");
        }
        diurnal_avg_->sputter_transport = true;
        diurnal_avg_->sputter_transport_share = transport_share;
        diurnal_avg_->sputter_escape_share = escape_share;
        diurnal_avg_->sputter_reservoir_share = reservoir_share;
        stats_.sputter_partition_active = true;
    }

    /**
     * Global sputter mass audit of the CURRENT flux state [kg/s].
     * gross = redeposited + escaped + reservoir must close to 1e-6 (the caller throws);
     * redeposited is MEASURED from the deposition field (the kernel output after
     * rescaling), not assumed. Also checks the PTM-642 (c) regression:
     * deposition <= sublimation + sputter transport source.
     */
    struct SputterAudit {
        double gross = 0.0, transport_source = 0.0, redeposited = 0.0, escaped = 0.0, reservoir = 0.0;
        double sublimation = 0.0, deposition = 0.0, deposition_from_sublimation = 0.0;
        double closure_rel = 0.0;       // |gross - (redeposited + escaped + reservoir)| / gross
        double transport_rel = 0.0;     // |transport_source - redeposited| / transport_source
        double dep_excess_rel = 0.0;    // (deposition - sublimation - transport_source) / (sublimation + transport_source)
    };
    SputterAudit sputterFluxAudit(const std::vector<double>& face_areas) const {
        SputterAudit a;
        const auto& d = *diurnal_avg_;
        for (size_t i = 0; i < num_faces_; ++i) {
            a.gross += d.avg_sputtering_flux[i] * face_areas[i];
            a.redeposited += d.avg_deposition_sputter_flux[i] * face_areas[i];
            a.sublimation += d.avg_sublimation_flux[i] * face_areas[i];
            a.deposition += d.avg_deposition_flux[i] * face_areas[i];
            a.deposition_from_sublimation += d.avg_deposition_sublimation_flux[i] * face_areas[i];
        }
        if (d.perFaceSputter()) {
            for (size_t i = 0; i < num_faces_; ++i) {
                const double g = d.avg_sputtering_flux[i] * face_areas[i];
                a.transport_source += d.sputter_face_transport_frac[i] * g;
                a.escaped += d.sputter_face_escape_frac[i] * g;
                a.reservoir += d.sputter_face_reservoir_frac[i] * g;
            }
        } else {
            a.transport_source = d.sputter_transport_share * a.gross;
            a.escaped = d.sputter_escape_share * a.gross;
            a.reservoir = d.sputter_reservoir_share * a.gross;
        }
        if (a.gross > 0.0) {
            a.closure_rel = std::abs(a.gross - (a.redeposited + a.escaped + a.reservoir)) / a.gross;
        }
        if (a.transport_source > 0.0) {
            a.transport_rel = std::abs(a.transport_source - a.redeposited) / a.transport_source;
        }
        const double src = a.sublimation + a.transport_source;
        if (src > 0.0) a.dep_excess_rel = (a.deposition - src) / src;
        return a;
    }

    /**
     * Get configuration
     */
    const MultiTimescaleConfig& getConfig() const { return config_; }
    void setConfig(const MultiTimescaleConfig& config) { config_ = config; }

    /**
     * Get concentration change rate for a face [1/s]
     */
    double getConcentrationChangeRate(size_t face_idx) const {
        return (face_idx < concentration_change_rate_.size()) ? concentration_change_rate_[face_idx] : 0.0;
    }

    /**
     * Print progress with visual progress bar
     */
    void printProgress() const {
        if (!config_.show_progress) return;

        int epoch = stats_.current_epoch;
        int total = config_.num_epochs;
        double pct = 100.0 * epoch / total;
        double years = stats_.elapsed_time_years;

        // Build progress bar (30 chars wide)
        const int bar_width = 30;
        int filled = static_cast<int>(bar_width * epoch / total);
        std::string bar(filled, '=');
        if (filled < bar_width) bar += '>';
        bar.resize(bar_width, ' ');

        // Format years nicely
        std::string years_str;
        if (years < 1e-3) {
            years_str = std::to_string(static_cast<int>(years * 365.25 * 24)) + " hr";
        } else if (years < 1.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years * 365.25) << " days";
            years_str = oss.str();
        } else if (years < 1000.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << years << " yr";
            years_str = oss.str();
        } else if (years < 1e6) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years / 1000.0) << " kyr";
            years_str = oss.str();
        } else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (years / 1e6) << " Myr";
            years_str = oss.str();
        }

        // Format concentration
        std::string conc_str;
        double conc_pct = stats_.max_nonice_concentration * 100.0;
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << conc_pct << "%";
            conc_str = oss.str();
        }

        // Format flux imbalance
        std::string flux_str;
        {
            double flux_pct = stats_.flux_imbalance_fraction * 100.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << flux_pct << "%";
            flux_str = oss.str();
        }

        std::cerr << "\r[" << bar << "] "
                  << std::setw(5) << std::fixed << std::setprecision(1) << pct << "% | "
                  << "Epoch " << std::setw(5) << epoch << "/" << total << " | "
                  << std::setw(10) << years_str << " | "
                  << "nonice: " << std::setw(8) << conc_str << " | "
                  << "flux: " << std::setw(7) << flux_str
                  << "   " << std::flush;
    }

    /**
     * Reset evolution state for new simulation
     */
    void reset() {
        diurnal_avg_->reset();
        std::fill(concentration_change_rate_.begin(), concentration_change_rate_.end(), 0.0);
        stats_ = EvolutionStatistics();
    }
};

} // namespace exosphere
} // namespace ptm

#endif // MULTITIMESCALE_EVOLUTION_H
