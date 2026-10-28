// eq_extrapolation.h — pass-stride RRE extrapolation of the equilibration state.
//
// WHAT THIS IS
// ------------
// The GPU equilibration phase replays ONE uploaded forcing array P times
// (thermal_opencl.cpp: `total_cycles = n_equil_cycles + 1`). Because the same
// block is replayed byte-identically, the map
//
//     S_{k+1} = Phi(S_k)          S_k = the whole T(z) field at the top of pass k
//
// is EXACTLY autonomous, for every face, on every body, independent of how
// seasonal the forcing inside the block is. That is the entire reason this
// works at PASS stride and does not work at lunation stride: the measured
// seasonal modulation index of the moon3d eq block is a_f = 2.32 (median), 232x
// the 0.01 bar at which a one-lunation stride would have been legitimate
// (af_measurement/RESULTS.md, 2026-08-08). There is deliberately NO stride knob
// here — pass stride is hard-coded, and a body that genuinely wants a one-cycle
// stride gets it by setting `equilibration_orbits: 1`.
//
// Vector extrapolation (RRE) on S_0..S_{p} therefore estimates the fixed point
// of a genuinely autonomous map, which is the only situation in which it is
// sound.
//
// ALGORITHM — RRE, order p, over the last p+1 states
// --------------------------------------------------
//     u_j   = X_{j+1} - X_j                        j = 0..p-1
//     min || sum_j gamma_j u_j ||_2   s.t.  sum_j gamma_j = 1
//     T*    = sum_{j=0}^{p-1} gamma_j X_j
//
// solved in the unconstrained parameterization gamma_0 = 1 - sum_{j>=1} beta_j,
// i.e. min_beta || u_0 + sum_{j>=1} beta_j (u_j - u_0) ||. That reduction is
// what makes it robust: when fewer than p modes are live the normal-equation
// matrix is singular, and a MINIMUM-NORM least-squares solve returns the small
// solution instead of amplifying noise through a regularized inverse. This
// mirrors, line for line, the reference implementation the GO decision was
// measured on (`eq_compare_2026-08-08/code/modeext.py:_rre`, rcond 1e-12) — the
// numbers in the physics review's A1 table are that function's output, so any
// deviation here would invalidate the acceptance thresholds.
//
// For the default p = 4 the per-face solve is a 50x3 least-squares problem
// (50 = depth layers, 3 = p-1). It is done in FLOAT64: measured (ADDENDUM A3e)
// to be unnecessary at pass stride, and kept anyway because it costs
// microseconds and removes a class of future argument.
//
// CERTIFICATE (mandatory, and free in wall-clock)
// -----------------------------------------------
//     rhat  = clamp( ||S_{N-1}-S_{N-2}||_2 / ||S_{N-2}-S_{N-3}||_2, 1e-6, r_cap )
//     T_out = Phi(T*)        <-- pass N, which we were running anyway
//     ehat  = safety * ||T_out - T*||_inf * rhat/(1-rhat)
//
// The verification pass IS the polish pass IS the delivered state, so the
// certificate costs zero wall-clock. On ehat > tol_K the state still ships (it
// is a contraction step and strictly better than S_{N-1}); the face is counted
// `cert_refused` and the distribution is printed. Everything is per face.
//
// SCOPE / NON-GOALS
// -----------------
// - Default OFF. With `method: off` every function here is unreachable and the
//   solver takes literally the pre-existing code path.
// - No kernel change, no new dispatch, no sub-pass machinery.
// - Header-only on purpose: it compiles identically under -DPTM_ENABLE_GPU=ON
//   and =OFF, and needs no CMake edit. The 2026-08-07 CPU-only CI break came
//   from putting a helper behind the GPU guard while calling it outside.
#ifndef PTM_THERMAL_EQ_EXTRAPOLATION_H
#define PTM_THERMAL_EQ_EXTRAPOLATION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace ptm {
namespace eqextrap {

// Matches modeext.py's RCOND exactly. Singular values <= rcond * sigma_max are
// treated as zero, which is numpy.linalg.lstsq's own convention.
constexpr double kRcond = 1e-12;

// Smallest polynomial order we will ever run. p=2 is not merely weak, it is
// HARMFUL: measured 54.82 K worst-column error versus plain 2-pass cycling's
// 31.88 K (ADDENDUM A1). The DC seed sits outside the linear regime and a
// degree-1 annihilator cannot remove it. Refuse, never silently reduce.
constexpr int kMinOrderP = 3;
// Validated range upper bound. Above this we still run, but we say so.
constexpr int kMaxValidatedOrderP = 4;
// Column limit for the tiny stack-allocated solver below (p-1 columns).
constexpr int kMaxCols = 8;

// Level-2 hold threshold (ADDENDUM NOTE 8) -- declared here because Options
// uses it as a default. Full rationale at level2Hold() below.
constexpr double kHoldKDefault = 1e-3;  // K per pass

// Certificate constants, folded in from what were user knobs. Neither was ever
// set to anything but these values in any validated run, and neither has a
// defensible reason to vary per config:
//   kSafety -- the conservatism factor in ehat. Measured recall over 72 cases
//     (4 columns x 3 profiles x 6 variants): 0 missed detections at both the
//     0.1 K and 0.5 K thresholds, median conservatism 3.42x. 3.0 is the value
//     that recall was established at; changing it invalidates that evidence.
//   kRCap -- clamp on the measured contraction ratio, so that ehat stays finite
//     when a sequence stalls and rhat -> 1. At pass stride the measured rhat is
//     0.26-0.59, four orders below the cap, so it is a guard rail rather than a
//     tuning parameter.
constexpr double kSafety = 3.0;
constexpr double kRCap   = 0.9995;

struct Options {
    bool   enabled  = false;   // method: rre
    int    order_p  = 4;
    double tol_K    = 0.5;
    double hold_K   = kHoldKDefault;  // Level-2 hold threshold, K per pass  // Level-2 hold threshold, K per pass
    // Radiation-coupling iteration that owns this eq phase. Reported in the
    // banner. The ring buffer is function-local and therefore structurally
    // cleared at every call, which is what satisfies ADDENDUM A5's mandatory
    // condition: on coupling iterations >= 2 the eq forcing gains
    // eq_indirect_flux[f], the fixed point MOVES, and stale snapshots from the
    // previous iteration's map must never be extrapolated across.
    int    coupling_iteration = 0;
};

// ---------------------------------------------------------------------------
// ADDENDUM NOTE 6 guard constants.
//
// The defect the first validation caught is an ILL-POSEDNESS, not a bug: RRE
// fits one gamma per column by minimizing a COLUMN-WIDE 2-norm, so a deep layer
// whose window differences sit at the float32 noise floor contributes nothing
// to the fit and yet receives the full weight vector. In exact arithmetic
// sum(gamma)=1 protects it; in float32 the residual is
// ||gamma||_1 * eps32 * |S(z)| -- amplified roundoff, with the amplification
// set by layers that carry signal and applied to layers that do not.
//
// MEASURED separation (N6.2): worst ACTIVE column ||gamma||_1 = 5.3; the faces
// that failed need ||gamma||_1 ~ 2.6e5. Five orders. `cond` is a POOR
// discriminator (an active column measured cond 3.2e5 and was fine) --
// ||gamma||_1 is the right diagnostic.
//
// Rejected by the physics review, do not reintroduce: per-layer gamma (the scalar-Aitken
// family, measured diverging to 1e14 K), per-depth-block gamma (more
// parameters, arbitrary boundaries, deep block still fitted to noise), and
// "extrapolate only above the last active layer" (assumes activity is monotone
// in depth -- it is NOT: the PSR/polar-night population has a LOCKED surface
// over a still-relaxing deep column, exactly inverted).
// LEVEL 2 threshold (ADDENDUM NOTE 8). ABSOLUTE, in Kelvin, and deliberately
// NOT coupled to ||gamma||_1: that coupling was justified by a roundoff premise
// that the measurement refuted (N8.0), and keeping it would carry a refuted
// rationale into shipped code. Physically interpretable instead: a layer moving
// less than a millikelvin per pass is converged. Sized on the contraction bound,
// not on noise -- s_z = 1e-3 K/pass at r_pass = 0.348 implies a remaining error
// of 5.3e-4 K, an order of magnitude below any threshold in the gate table.
// No ULP floor arm: 1e-3 K is 556 ULP at 30 K and 43 ULP at 390 K, so such an
// arm would be dead code.
constexpr double kHeldDispTol  = 1e-4;  // K, T*-based assertion (structurally 0)
constexpr double kHeldPostTol  = 1e-2;  // K, POST-POLISH tripwire on held layers
constexpr int    kNBlocks     = 3;      // certificate blocks [0,15) [15,30) [30,n)
inline int blockOf(int layer) {
    return (layer < 15) ? 0 : (layer < 30 ? 1 : 2);
}

// LEVEL 2 — hold one layer whose LAST-PASS increment is below HOLD_K.
//
// s_z = |S_{N-1}(z) - S_{N-2}(z)|, the LAST difference only. The seed transition
// is excluded by construction, which is the whole point: including it (the
// N6.2 wording said max_j|u_j|) makes a layer bit-locked since pass 1 still look
// active, because the seed sits ~5.85 K away at depth. That is what made the
// first guard inert.
//
// LAST difference, not max of the last two, because the two error directions are
// NOT symmetric:
//   * a MISSED hold (converged layer judged active) -> displacement
//     gamma_0*(seed - converged), up to 0.5 K -- a WRONG ANSWER;
//   * a SPURIOUS hold (active layer judged quiet)  -> that layer converges at the
//     plain rate instead of the accelerated one -- graceful, never wrong.
// The predicate must therefore be biased toward holding, and the last difference
// is the smallest candidate statistic, so it holds most. The usual objection to a
// single difference -- an increment passing through zero on an active layer --
// lands entirely in the benign direction under that asymmetry.
inline bool level2Hold(double s_layer, double hold_K) {
    return s_layer <= hold_K;
}

// NEAR-DEGENERATE WINDOW — a REPORTED diagnostic, not a guard. Nothing acts on
// it, deliberately.
//
// A window is near-degenerate when the fit's weight on the seed snapshot could
// move a converged layer by more than the reporting tolerance:
//
//     |gamma_0| * max_z |S_0(z) - S_last(z)|  >  tol_K
//
// This is the physics review's bound (N8.0) evaluated at extrapolation time from quantities
// already in hand. It identifies the population that produced the 73 faces in
// the 2026-08-09 validation: columns whose upper layers are still active (so
// the fit assigns a non-zero seed weight) while deeper layers have converged to
// within hold_K of a value the seed is far from. Counting it makes that
// population visible on any body without re-deriving the analysis; acting on it
// would need evidence this round did not produce.
inline bool nearDegenerateWindow(double gamma0, double seed_distance, double tol_K) {
    return std::fabs(gamma0) * seed_distance > tol_K;
}

struct Stats {
    long long certified        = 0;  // ehat <= tol_K
    long long cert_refused     = 0;  // ehat >  tol_K  (state still shipped)
    long long budget_exhausted = 0;  // faces that never got p+1 passes
    long long faces_seen       = 0;
    int       passes_spent     = 0;
    long long solve_failed     = 0;  // non-finite lstsq -> fell back to S_{N-1}
    // ---- NOTE 6 additions ----
    long long converged_window  = 0; // whole column below the noise floor (vacuous)
    long long near_degenerate   = 0; // window where the seed weight could displace > tol_K
    long long layers_held       = 0; // Level 2: individual layers held
    long long faces_partly_held = 0; // faces with >=1 held layer but not all
    double    d_held_max        = 0.0;  // T*-based ASSERTION that the guard ran (structurally 0)
    long long layers_total      = 0;    // for the held-layer fraction
    std::vector<float> d_held_post;     // POST-POLISH |Phi(T*)-S_last| on HELD layers, per face
    std::vector<float> ehat;         // one entry per extrapolated face (max over blocks)

    void merge(const Stats& o) {
        certified        += o.certified;
        cert_refused     += o.cert_refused;
        budget_exhausted += o.budget_exhausted;
        faces_seen       += o.faces_seen;
        solve_failed     += o.solve_failed;
        converged_window += o.converged_window;
        near_degenerate  += o.near_degenerate;
        layers_held      += o.layers_held;
        faces_partly_held += o.faces_partly_held;
        d_held_max        = std::max(d_held_max, o.d_held_max);
        layers_total     += o.layers_total;
        d_held_post.insert(d_held_post.end(), o.d_held_post.begin(), o.d_held_post.end());
        passes_spent      = std::max(passes_spent, o.passes_spent);
        ehat.insert(ehat.end(), o.ehat.begin(), o.ehat.end());
    }
};

// ---------------------------------------------------------------------------
// Minimum-norm least squares for a tall, very thin system, via one-sided Jacobi
// SVD. Chosen over normal equations because rank deficiency is EXPECTED here
// (a face whose column has fewer than p live modes) and the minimum-norm
// behavior on the deficient subspace is the property that keeps the
// extrapolant sane. numpy.linalg.lstsq has the same semantics.
//
//   A : m x n, COLUMN-MAJOR, DESTROYED (overwritten with U*Sigma).
//   b : m
//   x : n   (output)
// Returns the numerical rank; *cond_out gets sigma_max/sigma_min.
// ---------------------------------------------------------------------------
inline int minNormLstsq(double* A, int m, int n, const double* b, double* x,
                        double rcond, double* cond_out) {
    if (n <= 0 || n > kMaxCols || m <= 0) {
        if (cond_out) *cond_out = HUGE_VAL;
        return 0;
    }
    double V[kMaxCols * kMaxCols];
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) V[j * n + k] = (j == k) ? 1.0 : 0.0;

    // One-sided Jacobi sweeps. n <= 3 in practice, so this converges in 2-3.
    for (int sweep = 0; sweep < 30; ++sweep) {
        bool rotated = false;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                const double* Ai = A + static_cast<size_t>(i) * m;
                const double* Aj = A + static_cast<size_t>(j) * m;
                for (int r = 0; r < m; ++r) {
                    alpha += Ai[r] * Ai[r];
                    beta  += Aj[r] * Aj[r];
                    gamma += Ai[r] * Aj[r];
                }
                if (gamma == 0.0) continue;
                if (std::fabs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) continue;
                // Classical Jacobi: with the column update
                //   A_i <- c*A_i - s*A_j ,  A_j <- s*A_i + c*A_j
                // the new off-diagonal is c*s*(alpha-beta) + (c^2-s^2)*gamma,
                // which vanishes for t = s/c solving t^2 + 2*zeta*t - 1 = 0 with
                // zeta = (beta-alpha)/(2*gamma). Take the SMALLER root.
                // The sign here is load-bearing: (alpha-beta) instead of
                // (beta-alpha) makes the sweep ANTI-diagonalize, and the solve
                // then disagrees with numpy.linalg.lstsq by ~20 % on a
                // full-rank system while still returning finite numbers.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t    = (zeta >= 0.0 ? 1.0 : -1.0)
                                  / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c   = 1.0 / std::sqrt(1.0 + t * t);
                const double s   = c * t;
                double* Aim = A + static_cast<size_t>(i) * m;
                double* Ajm = A + static_cast<size_t>(j) * m;
                for (int r = 0; r < m; ++r) {
                    const double ai = Aim[r], aj = Ajm[r];
                    Aim[r] = c * ai - s * aj;
                    Ajm[r] = s * ai + c * aj;
                }
                for (int r = 0; r < n; ++r) {
                    const double vi = V[static_cast<size_t>(i) * n + r];
                    const double vj = V[static_cast<size_t>(j) * n + r];
                    V[static_cast<size_t>(i) * n + r] = c * vi - s * vj;
                    V[static_cast<size_t>(j) * n + r] = s * vi + c * vj;
                }
                rotated = true;
            }
        }
        if (!rotated) break;
    }

    double sigma[kMaxCols];
    double smax = 0.0;
    for (int j = 0; j < n; ++j) {
        double sq = 0.0;
        const double* Aj = A + static_cast<size_t>(j) * m;
        for (int r = 0; r < m; ++r) sq += Aj[r] * Aj[r];
        sigma[j] = std::sqrt(sq);
        smax = std::max(smax, sigma[j]);
    }
    double smin = HUGE_VAL;
    for (int j = 0; j < n; ++j) smin = std::min(smin, sigma[j]);
    if (cond_out) *cond_out = (smin > 0.0) ? (smax / smin) : HUGE_VAL;

    const double cutoff = rcond * smax;
    double y[kMaxCols];
    int rank = 0;
    for (int j = 0; j < n; ++j) {
        if (sigma[j] > cutoff && sigma[j] > 0.0) {
            double ub = 0.0;
            const double* Aj = A + static_cast<size_t>(j) * m;
            for (int r = 0; r < m; ++r) ub += Aj[r] * b[r];
            y[j] = ub / (sigma[j] * sigma[j]);   // (u_j^T b)/sigma_j with u_j = A_j/sigma_j
            ++rank;
        } else {
            y[j] = 0.0;
        }
    }
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += V[static_cast<size_t>(j) * n + k] * y[j];
        x[k] = acc;
    }
    return rank;
}

// ---------------------------------------------------------------------------
// RRE over p+1 states of a single face's depth column.
//   X   : (p+1) x n_layers, ROW-MAJOR by snapshot -> X[s * n_layers + l]
//         X[0] is the OLDEST, X[p] the NEWEST.
//   out : n_layers  (the extrapolated state T*)
// Returns false (and leaves `out` = X[p], the newest state) if the solve is
// not usable. `out` is therefore ALWAYS a valid state -- never garbage.
// ---------------------------------------------------------------------------
// `gamma0_out` receives gamma_0, the weight the fit assigns to the OLDEST
// snapshot (the seed). It is the quantity that bounds the known failure mode:
// for a layer converged since pass 1 (S_0 = a, S_1.. = b), sum(gamma) = 1 gives
//     T*(z) = b + gamma_0 * (a - b),
// so the displacement is gamma_0 times that layer's seed error -- LINEAR, not
// amplified. Measured gamma_0 ~ 0.09 on the affected faces.
// `s_layer_out`, if non-null, receives
// max_j |u_j(z)| per layer: that layer's window SIGNAL, which Level 2 compares
// against the local noise floor.
inline bool rreColumn(const double* X, int p, int n_layers, double* out,
                      double* cond_out, double rcond = kRcond,
                      double* gamma0_out = nullptr,
                      double* s_layer_out = nullptr) {
    const double* newest = X + static_cast<size_t>(p) * n_layers;
    if (s_layer_out) {
        // LAST difference only: |S_{N-1} - S_{N-2}| = |X[p] - X[p-1]|.
        const double* sl  = X + static_cast<size_t>(p)     * n_layers;
        const double* sl1 = X + static_cast<size_t>(p - 1) * n_layers;
        for (int l = 0; l < n_layers; ++l) s_layer_out[l] = std::fabs(sl[l] - sl1[l]);
    }
    auto fallback = [&]() {
        for (int l = 0; l < n_layers; ++l) out[l] = newest[l];
        if (cond_out) *cond_out = HUGE_VAL;
        if (gamma0_out) *gamma0_out = HUGE_VAL;
        return false;
    };
    if (p < 2 || n_layers <= 0 || p - 1 > kMaxCols) return fallback();

    // u_j = X_{j+1} - X_j, j = 0..p-1. Only u_0 and (u_j - u_0) are needed.
    // A is n_layers x (p-1), column-major: column (j-1) is u_j - u_0.
    std::vector<double> A(static_cast<size_t>(n_layers) * (p - 1));
    std::vector<double> b(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        const double u0 = X[static_cast<size_t>(1) * n_layers + l] - X[l];
        b[l] = -u0;
        for (int j = 1; j <= p - 1; ++j) {
            const double uj = X[static_cast<size_t>(j + 1) * n_layers + l]
                            - X[static_cast<size_t>(j) * n_layers + l];
            A[static_cast<size_t>(j - 1) * n_layers + l] = uj - u0;
        }
    }
    double beta[kMaxCols];
    minNormLstsq(A.data(), n_layers, p - 1, b.data(), beta, rcond, cond_out);

    double gam[kMaxCols + 1];
    double bsum = 0.0;
    for (int j = 0; j < p - 1; ++j) {
        if (!std::isfinite(beta[j])) return fallback();
        gam[j + 1] = beta[j];
        bsum += beta[j];
    }
    gam[0] = 1.0 - bsum;
    if (!std::isfinite(gam[0])) return fallback();

    if (gamma0_out) *gamma0_out = gam[0];

    // T* = sum_{j=0}^{p-1} gamma_j X_j  (note: X_p is used only via u_{p-1}).
    for (int l = 0; l < n_layers; ++l) {
        double acc = 0.0;
        for (int j = 0; j < p; ++j) acc += gam[j] * X[static_cast<size_t>(j) * n_layers + l];
        out[l] = acc;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (!std::isfinite(out[l])) return fallback();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Ring-buffer addressing. Snapshot S_k is written to slot k % depth, so at the
// top of pass `newest_cycle` the p+1 live snapshots, OLDEST FIRST, are at
//   slot(s) = (newest_cycle - p + s) % depth,   s = 0..p
// depth == p+1 by construction, and newest_cycle >= p is guaranteed by the
// caller's `total_cycles >= p+1` test, so the argument never goes negative.
//
// This lives here rather than inline at the call site because an off-by-one
// would feed RRE a ROTATED sequence: still finite, still plausible, and wrong
// in a way no physics gate would flag. Shared with the unit test.
inline int ringSlot(int newest_cycle, int p, int s, int depth) {
    const int k = newest_cycle - p + s;
    return ((k % depth) + depth) % depth;
}

inline double clampRhat(double num, double den, double r_cap) {
    if (!(den > 0.0) || !std::isfinite(num) || !std::isfinite(den)) return r_cap;
    double r = num / den;
    if (!std::isfinite(r)) return r_cap;
    return std::min(r_cap, std::max(1e-6, r));
}

inline double percentile(std::vector<float>& v, double q) {
    if (v.empty()) return 0.0;
    const size_t k = static_cast<size_t>(q * (v.size() - 1) + 0.5);
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return static_cast<double>(v[k]);
}

}  // namespace eqextrap
}  // namespace ptm

#endif  // PTM_THERMAL_EQ_EXTRAPOLATION_H
