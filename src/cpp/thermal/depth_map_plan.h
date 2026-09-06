// depth_map_plan.h -- how a requested subsurface depth maps onto the solved
// depth grid, and what that mapping costs in accuracy.
//
// This header is deliberately PURE: no mesh, no Planet, no solver, no I/O. It
// takes a depth request and a z[] grid and returns either a plan or a refusal.
// That is what makes it unit-testable without a run (tools/test_depth_map_plan)
// and what lets `test_thermal_wave_depth` score the SAME arithmetic the engine
// uses against an analytic reference.
//
// ---------------------------------------------------------------------------
// WHY A PLAN AND NOT JUST AN INTERPOLATION
// ---------------------------------------------------------------------------
// Averaging commutes with linear interpolation in z exactly; max and min do
// not:
//     max_t[(1-w)T_j + w T_{j+1}]  <=  (1-w) max_t T_j + w max_t T_{j+1}
// so interpolating an already-reduced max field is an UPPER BOUND whose gap is
// set by the phase lag between adjacent layers. Measured on the real lunar
// adaptive grid (test_thermal_wave_depth, 2026-09-04) that post-hoc route
// ("architecture B") costs 1.229e-3 * A and is WORST AT LAYER 0, i.e. 0.197 K
// at a lunar-equatorial amplitude A = 160 K -- over the project's 0.15 K floor
// by itself. Accumulating max/min in the engine at the requested depth
// ("architecture A") costs 4.8e-4 * A. Hence the shipped architecture is the
// hybrid: avg is exact either way, max/min are accumulated in-engine.
//
// A separate number, and the one most easily mis-budgeted: the INSTANTANEOUS
// interpolation error at an off-node depth is 1.217e-3 * A = 0.195 K at
// A = 160 K, independent of architecture, because the profile is curved and
// the interpolant is straight. It is LARGER than the error in the max (0.085 K)
// -- pointwise errors partly cancel at the peak. So a time series at depth and
// a max/min map at the same depth have DIFFERENT error budgets, differing by
// ~2.3x and in opposite directions from the naive expectation. Never quote one
// for the other. Both are reported per depth by this module, separately.
//
// Only a requested depth that lands ON a node avoids either. That is why
// snap-to-node is the DEFAULT (D15) and why the residual is always reported.
//
// ---------------------------------------------------------------------------
// WHAT THIS MODULE REFUSES
// ---------------------------------------------------------------------------
// Every one of these is a refusal, not a warning and not a silent fix-up,
// because each of them would otherwise ship a map that is physics over part of
// its area and something else over the rest, distinguishable only by a log
// line:
//   * a depth below the deepest solved node (naming thermal.grid_b);
//   * a negative or non-finite depth;
//   * duplicate requested depths (refuse, do NOT dedupe -- a dedupe hides a
//     config error and changes the file set the caller expected);
//   * two requests that resolve to the same output file name;
//   * an empty or unrecognized statistic list;
//   * a grid that is not strictly increasing, or shorter than two nodes;
//   * a projected product size above the caller's ceiling.
//
#ifndef PTM_THERMAL_DEPTH_MAP_PLAN_H
#define PTM_THERMAL_DEPTH_MAP_PLAN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace thermal {
namespace depthmap {

// The statistics a depth map can carry. Deliberately a closed set: these are
// the three the surface products already ship, so the depth files mirror
// statistics/temperature_{avg,max,min}.bin exactly.
enum class Stat { Avg, Max, Min };

inline const char* statName(Stat s) {
    switch (s) {
        case Stat::Avg: return "avg";
        case Stat::Max: return "max";
        case Stat::Min: return "min";
    }
    return "?";
}

inline bool statFromName(const std::string& name, Stat& out) {
    if (name == "avg") { out = Stat::Avg; return true; }
    if (name == "max") { out = Stat::Max; return true; }
    if (name == "min") { out = Stat::Min; return true; }
    return false;
}

// What the user asked for.
struct Request {
    std::vector<double> depths_m;
    bool snap_to_node = true;                 // D15: default ON
    std::vector<Stat> stats = { Stat::Avg, Stat::Max, Stat::Min };
    // D10 size guard on the STATISTICS product. 0 disables the check. The
    // default is generous for a statistics product (5 depths x 3 stats at
    // F = 2.4M is 137 MiB) and still refuses the class of request that is a
    // disk event rather than a product.
    std::uint64_t max_bytes = 8ull << 30;     // 8 GiB
};

// One resolved depth.
struct Entry {
    double requested_m = 0.0;
    double resolved_m  = 0.0;   // what is actually sampled
    double residual_m  = 0.0;   // resolved - requested (0 when not snapping)
    int    j           = 0;     // lower bracketing node index
    double w           = 0.0;   // weight on node j+1; 0 => exactly on node j
    bool   on_node     = false;
    std::string field; // e.g. "temperature_z0p050m"

    // Error budget, PER HARMONIC, per unit amplitude of that harmonic at the
    // surface. Index h-1 is harmonic h = 1..kBiasHarmonics.
    //
    // Two different quantities, and they are not interchangeable:
    //   bias_inst_per_A[]  the INSTANTANEOUS interpolation error. Summed over
    //                      harmonics weighted by their surface amplitudes it is
    //                      a RIGOROUS upper bound on the error in a time series
    //                      at this depth AND on the error in its max or min,
    //                      since |max f - max g| <= max|f - g|.
    //   bias_max_per_A[]   the error in the MAX specifically, which is smaller
    //                      because the pointwise errors partly cancel at the
    //                      peak. A scale, not a bound.
    //
    // Empty when no skin depth was supplied: the budget is not computable from
    // the grid alone, and a fabricated number is worse than none.
    //
    // ONE HARMONIC IS NOT ENOUGH, and this is the trap the arrays exist to
    // close: harmonic h propagates with skin depth delta/sqrt(h), so it sees an
    // effectively COARSER grid and its bias coefficient is larger. A
    // radiatively forced surface series is far from a pure cosine, so a
    // fundamental-only figure UNDER-predicts -- measured at ~20x on the
    // engine's own series (test_thermal_wave_depth section 6).
    std::vector<double> bias_max_per_A;
    std::vector<double> bias_inst_per_A;
};

// How many harmonics the per-depth budget is tabulated for. Four covers the
// harmonic content that carries essentially all of a lunar surface series'
// energy; the coefficients grow with h, so a user with more can extrapolate
// from the printed trend rather than be handed a silent truncation.
constexpr int kBiasHarmonics = 4;

struct Plan {
    std::vector<Entry>  entries;
    std::vector<double> z;              // the grid this plan was built against
    std::vector<Stat>   stats;
    bool   snap_to_node  = true;
    double skin_depth_m  = 0.0;         // 0 = not supplied, budget unavailable

    bool   empty() const { return entries.empty(); }
    size_t size()  const { return entries.size(); }
};

// ---------------------------------------------------------------------------
// Naming. `z<meters with '.' -> 'p'>m`, three decimals: sortable, filesystem
// safe, and reversible. Three decimals is a millimeter, which is finer than any
// shipped grid's top cell, but two distinct RESOLVED depths CAN still collide
// (adjacent deep nodes are far apart, adjacent shallow ones are not) -- and a
// collision would silently overwrite one map with another. buildPlan refuses
// on collision rather than trusting the resolution.
// ---------------------------------------------------------------------------
inline std::string depthTag(double z_m) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", z_m);
    std::string s(buf);
    for (char& c : s) if (c == '.') c = 'p';
    return "z" + s + "m";
}

inline std::string fieldName(double z_m) {
    return "temperature_" + depthTag(z_m);
}

// ---------------------------------------------------------------------------
// The error budget, from the damped thermal wave T(z,t) = A e^-zeta cos(wt-zeta)
// with zeta = z/delta.
//
//   truth (amplitude at the probe depth)     : e^-zeta_q
//   architecture A (max of the interpolant)  : |(1-w) E(zeta1) + w E(zeta2)|
//   instantaneous interpolation error        : |(1-w)E(zeta1) + w E(zeta2) - E(zeta_q)|
//   with E(zeta) = e^-(1+i)zeta
//
// delta is the SURFACE (lattice) diurnal skin depth, so zeta over-counts at
// depth where the local skin depth is larger; the reported bias is therefore
// conservative (an over-estimate) below the top cell, which is the safe
// direction for an error budget. Stated rather than assumed.
// ---------------------------------------------------------------------------
// `harmonic` h enters only through the skin depth: d_h = delta/sqrt(h), so
// every zeta is scaled by sqrt(h). h = 1 is the fundamental.
inline void waveBias(double zeta1, double zeta2, double zeta_q, double w,
                     int harmonic,
                     double& bias_max_per_A, double& bias_inst_per_A) {
    const double sh = std::sqrt(static_cast<double>(harmonic));
    const double z1 = zeta1 * sh, z2 = zeta2 * sh, zq = zeta_q * sh;
    const double a1 = std::exp(-z1), a2 = std::exp(-z2), aq = std::exp(-zq);
    const double re = (1.0 - w) * a1 * std::cos(z1) + w * a2 * std::cos(z2);
    const double im = -((1.0 - w) * a1 * std::sin(z1) + w * a2 * std::sin(z2));
    const double interp_amp = std::sqrt(re * re + im * im);
    bias_max_per_A = interp_amp - aq;

    const double dre = re - aq * std::cos(zq);
    const double dim = im + aq * std::sin(zq);
    bias_inst_per_A = std::sqrt(dre * dre + dim * dim);
}

// ---------------------------------------------------------------------------
// Size of the statistics product this plan will write, in bytes: for each
// depth and each statistic one uint32 face count plus one float32 per face,
// plus (when raw side files are requested) a float64 sum and a uint32 count per
// depth.
// ---------------------------------------------------------------------------
inline std::uint64_t estimateBytes(const Plan& plan, std::size_t n_faces,
                                   bool with_raw) {
    const std::uint64_t F = static_cast<std::uint64_t>(n_faces);
    const std::uint64_t per_field = 4ull + 4ull * F;
    std::uint64_t total = static_cast<std::uint64_t>(plan.entries.size()) *
                          static_cast<std::uint64_t>(plan.stats.size()) * per_field;
    if (with_raw) {
        const std::uint64_t per_raw = (4ull + 8ull * F) + (4ull + 4ull * F);
        total += static_cast<std::uint64_t>(plan.entries.size()) * per_raw;
    }
    return total;
}

// ---------------------------------------------------------------------------
// buildPlan -- the whole refusal surface lives here.
//
//   z, n_layers    the SOLVED depth grid this plan will be evaluated on
//   skin_depth_m   surface diurnal skin depth, or <= 0 for "not supplied"
//   n_faces        for the size guard; 0 disables it
//
// Returns false and fills `refusal` with an actionable message. `warnings` is
// appended to, never cleared, and is non-fatal by construction.
// ---------------------------------------------------------------------------
// The clauses that do NOT need the depth grid. Split out so the engine can run
// them BEFORE the illumination phase -- a misspelled statistic or a duplicated
// depth then costs seconds instead of hours -- while the grid-dependent clauses
// wait until the solver's real grid exists. buildPlan calls this first, so
// there is exactly one copy of each rule.
inline bool validateRequest(const Request& req, std::string& refusal) {
    refusal.clear();
    if (req.depths_m.empty()) return true;   // off, and byte-neutral

    if (req.stats.empty()) {
        refusal = "output.map_depth_stats is empty; a depth map with no statistic "
                  "would write nothing. Use [avg, max, min] or remove "
                  "output.map_depths_m.";
        return false;
    }
    for (std::size_t a = 0; a < req.stats.size(); ++a) {
        for (std::size_t b = a + 1; b < req.stats.size(); ++b) {
            if (req.stats[a] == req.stats[b]) {
                refusal = std::string("output.map_depth_stats lists '") +
                          statName(req.stats[a]) + "' twice.";
                return false;
            }
        }
    }
    for (double d : req.depths_m) {
        if (!std::isfinite(d)) {
            refusal = "output.map_depths_m contains a non-finite depth.";
            return false;
        }
        if (d < 0.0) {
            std::ostringstream os;
            os << "output.map_depths_m contains a negative depth (" << d
               << " m). Depth is measured downward from the surface; 0.0 is the "
                  "surface node.";
            refusal = os.str();
            return false;
        }
    }
    // Duplicate REQUESTED depths: refuse, do not dedupe (a dedupe hides a
    // config error and silently changes the file set the caller expects).
    for (std::size_t a = 0; a < req.depths_m.size(); ++a) {
        for (std::size_t b = a + 1; b < req.depths_m.size(); ++b) {
            if (req.depths_m[a] == req.depths_m[b]) {
                std::ostringstream os;
                os << "output.map_depths_m lists " << req.depths_m[a]
                   << " m twice (entries " << a << " and " << b
                   << "). Remove the duplicate; this is not deduplicated, because "
                      "a silent dedupe would change the set of files written.";
                refusal = os.str();
                return false;
            }
        }
    }
    return true;
}

inline bool buildPlan(const Request& req,
                      const double* z, int n_layers,
                      double skin_depth_m,
                      std::size_t n_faces, bool with_raw,
                      Plan& out, std::string& refusal,
                      std::vector<std::string>& warnings) {
    out = Plan{};
    if (!validateRequest(req, refusal)) return false;

    if (req.depths_m.empty()) {
        // Not a refusal: "no depths requested" is the default and must remain
        // byte-neutral. The caller checks plan.empty().
        return true;
    }

    if (!z || n_layers < 2) {
        refusal = "output.map_depths_m needs a solved depth grid with at least two "
                  "nodes; this run has " + std::to_string(n_layers) + ".";
        return false;
    }
    for (int i = 1; i < n_layers; ++i) {
        if (!(z[i] > z[i - 1])) {
            refusal = "the solved depth grid is not strictly increasing at node " +
                      std::to_string(i) + " (z[" + std::to_string(i - 1) + "]=" +
                      std::to_string(z[i - 1]) + ", z[" + std::to_string(i) + "]=" +
                      std::to_string(z[i]) + "); a depth map cannot be interpolated "
                      "on it.";
            return false;
        }
    }

    const double z_top = z[0];
    const double z_bot = z[n_layers - 1];

    out.z.assign(z, z + n_layers);
    out.stats = req.stats;
    out.snap_to_node = req.snap_to_node;
    out.skin_depth_m = (skin_depth_m > 0.0) ? skin_depth_m : 0.0;

    for (double d : req.depths_m) {
        if (d < z_top) {
            // Only reachable if a grid ever starts below 0; kept so the branch
            // cannot become a silent extrapolation later.
            std::ostringstream os;
            os << "requested depth " << d << " m is ABOVE the shallowest solved node ("
               << z_top << " m). Extrapolating above the grid is refused.";
            refusal = os.str();
            return false;
        }
        // D9: below the deepest solved node is a REFUSAL, not a Fourier
        // extrapolation. The location export (locations/*_depths.csv) does
        // extrapolate with a warning; a MAP must not, because a map that is
        // physics over part of its area and extrapolation over the rest is
        // indistinguishable to every downstream consumer.
        if (d > z_bot) {
            std::ostringstream os;
            os << "requested depth " << d << " m is below the deepest solved node ("
               << z_bot << " m). Increase thermal.grid_b (skin depths to the bottom "
                           "node) so the column reaches that depth, or drop the "
                           "request; the engine will not extrapolate a map.";
            refusal = os.str();
            return false;
        }

        Entry e;
        e.requested_m = d;

        // Bracket: largest j with z[j] <= d, capped so j+1 is in range.
        int j = static_cast<int>(std::upper_bound(z, z + n_layers, d) - z) - 1;
        if (j < 0) j = 0;
        if (j > n_layers - 2) j = n_layers - 2;
        const double h = z[j + 1] - z[j];
        double w = (h > 0.0) ? (d - z[j]) / h : 0.0;
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;

        if (req.snap_to_node) {
            const int jn = (w <= 0.5) ? j : (j + 1);
            e.j = std::min(jn, n_layers - 2);
            e.w = (jn == e.j) ? 0.0 : 1.0;      // jn == n_layers-1 -> j = n-2, w = 1
            e.resolved_m = z[jn];
            e.on_node = true;
        } else {
            e.j = j;
            e.w = w;
            e.resolved_m = d;
            e.on_node = (w == 0.0);
        }
        e.residual_m = e.resolved_m - e.requested_m;
        e.field = fieldName(e.resolved_m);

        if (out.skin_depth_m > 0.0) {
            const double inv = 1.0 / out.skin_depth_m;
            e.bias_max_per_A.assign(kBiasHarmonics, 0.0);
            e.bias_inst_per_A.assign(kBiasHarmonics, 0.0);
            for (int h = 1; h <= kBiasHarmonics; ++h) {
                waveBias(z[e.j] * inv, z[e.j + 1] * inv, e.resolved_m * inv, e.w, h,
                         e.bias_max_per_A[h - 1], e.bias_inst_per_A[h - 1]);
            }
        }

        out.entries.push_back(e);
    }

    // Two requests resolving to the same output file would silently overwrite
    // one another. This is reachable in normal use: snap-to-node maps a whole
    // interval of requests onto one node.
    for (std::size_t a = 0; a < out.entries.size(); ++a) {
        for (std::size_t b = a + 1; b < out.entries.size(); ++b) {
            if (out.entries[a].field == out.entries[b].field) {
                std::ostringstream os;
                os << "requested depths " << out.entries[a].requested_m << " m and "
                   << out.entries[b].requested_m << " m both resolve to "
                   << out.entries[a].resolved_m << " m (file '"
                   << out.entries[a].field << "_*.bin')";
                if (req.snap_to_node) {
                    os << " after snapping to the nearest grid node. Set "
                          "output.map_depth_snap_to_node: false to sample them "
                          "separately (and read the interpolation bias reported in "
                          "statistics/depth_map_z.json first), or drop one.";
                } else {
                    os << ". Depth file names carry three decimals (1 mm).";
                }
                refusal = os.str();
                out = Plan{};
                return false;
            }
        }
    }

    if (n_faces > 0 && req.max_bytes > 0) {
        const std::uint64_t bytes = estimateBytes(out, n_faces, with_raw);
        if (bytes > req.max_bytes) {
            std::ostringstream os;
            os << "the requested depth product is " << (bytes / (1024.0 * 1024.0))
               << " MiB (" << out.entries.size() << " depths x " << out.stats.size()
               << " statistics x " << n_faces << " faces"
               << (with_raw ? " + merge-friendly raw side files" : "")
               << "), over the " << (req.max_bytes / (1024.0 * 1024.0))
               << " MiB ceiling. Request fewer depths or fewer statistics.";
            refusal = os.str();
            out = Plan{};
            return false;
        }
    }

    if (out.skin_depth_m <= 0.0) {
        warnings.push_back(
            "depth map: no skin depth was available, so the per-depth "
            "interpolation error budget could not be computed and is reported as "
            "null in statistics/depth_map_z.json.");
    }
    if (!req.snap_to_node) {
        warnings.push_back(
            "depth map: output.map_depth_snap_to_node is false, so off-node depths "
            "are linearly interpolated. See the per-depth bias in "
            "statistics/depth_map_z.json -- at lunar-equatorial amplitude the "
            "instantaneous interpolation error alone is ~0.195 K, over the 0.15 K "
            "floor by itself.");
    }
    return true;
}

// ---------------------------------------------------------------------------
// Sampling. THE one place the interpolation is written down; the OpenCL kernel
// is a device mirror of it (like conductivity_model.h) and is held to it by the
// CPU-vs-GPU agreement gate, not by inspection.
//
// `stride` is the distance in elements between consecutive layers of one
// face's column: 1 for a contiguous column, THERMAL_BATCH_SIZE for the SIMD
// batch's layer-major SoA.
// ---------------------------------------------------------------------------
template <typename T>
inline double sampleColumn(const Entry& e, const T* column, std::ptrdiff_t stride) {
    const double a = static_cast<double>(column[static_cast<std::ptrdiff_t>(e.j) * stride]);
    if (e.on_node && e.w == 0.0) return a;   // exact: no rounding introduced
    const double b = static_cast<double>(column[static_cast<std::ptrdiff_t>(e.j + 1) * stride]);
    if (e.w == 1.0) return b;                // exact
    return (1.0 - e.w) * a + e.w * b;
}

// ---------------------------------------------------------------------------
// Accumulator -- per depth x per face avg/max/min over an output-step window.
// Face-owned, so it needs no atomics; the CPU driver fills disjoint face ranges
// from separate OpenMP threads exactly as the GPU kernel fills disjoint work
// items.
// ---------------------------------------------------------------------------
struct Accumulator {
    std::size_t n_depths = 0;
    std::size_t n_faces  = 0;
    std::vector<double> sum;          // n_depths * n_faces, depth-major
    std::vector<double> vmax;
    std::vector<double> vmin;
    // PER FACE, not per run: this is what statistics/_raw/*_count.bin carries
    // and what ptm_merge_chunks divides by, and a run-wide scalar could not
    // express a face whose contribution came from a different number of frames.
    // It is also the only thing that makes "this face was never folded in"
    // distinguishable from "this face was folded in and its sum is 0".
    std::vector<std::uint32_t> count;

    bool ready() const {
        return n_depths > 0 && n_faces > 0 &&
               sum.size() == n_depths * n_faces &&
               vmax.size() == sum.size() && vmin.size() == sum.size() &&
               count.size() == n_faces;
    }

    void allocate(std::size_t depths, std::size_t faces) {
        n_depths = depths;
        n_faces  = faces;
        sum.assign(depths * faces, 0.0);
        // ⛔ FINITE sentinels, not +/-infinity. The OpenCL mirror of this
        // accumulator is compiled `-cl-fast-relaxed-math`, which implies
        // `-cl-finite-math-only` -- the compiler may assume no operand is
        // infinite, and there is no OpenCL equivalent of the host build's
        // `-fno-finite-math-only` to keep the sentinel alive. Every max map
        // could then ship as -inf with nothing to catch it. Keeping the host
        // and device sentinels finite AND identical in kind removes the class.
        // "Never folded in" is answered by `count`, not by the sentinel.
        vmax.assign(depths * faces, -std::numeric_limits<double>::max());
        vmin.assign(depths * faces,  std::numeric_limits<double>::max());
        count.assign(faces, 0u);
    }

    // Reset ONE face to the pristine state. Needed by the radiation-coupling
    // re-thermal, which re-integrates a selected face and must REPLACE that
    // face's contribution, not add a second one on top of iteration 0's.
    void resetFace(std::size_t face) {
        for (std::size_t d = 0; d < n_depths; ++d) {
            const std::size_t idx = d * n_faces + face;
            sum[idx]  = 0.0;
            vmax[idx] = -std::numeric_limits<double>::max();
            vmin[idx] =  std::numeric_limits<double>::max();
        }
        count[face] = 0u;
    }

    // Fold one face's column at one epoch. A non-null `surface_override`
    // replaces node 0: the massless surface node is RE-EVALUATED for the output
    // frame (thermal::outputSurfaceTempBatch) rather than read out of the
    // solved column, so a depth product that read the column's node 0 would not
    // reproduce statistics/temperature_*.bin and would be reporting a different
    // quantity at z = 0 without saying so.
    //
    // ⛔ A POINTER, not a NaN sentinel. libthermal is compiled
    // `-ffast-math -fno-finite-math-only` (thermal/CMakeLists.txt), and this
    // header is deliberately usable from any target: under plain -ffast-math
    // an `std::isnan(x)` sentinel folds to constant false, the override would
    // be silently dropped, and z = 0 would quietly become a different quantity.
    // A null pointer cannot be optimized away.
    template <typename T>
    void addColumn(const Plan& plan, std::size_t face,
                   const T* column, std::ptrdiff_t stride,
                   const double* surface_override = nullptr) {
        const bool have_override = (surface_override != nullptr);
        for (std::size_t d = 0; d < plan.entries.size(); ++d) {
            const Entry& e = plan.entries[d];
            double v;
            if (have_override && e.j == 0) {
                const double a = *surface_override;
                if (e.on_node && e.w == 0.0) {
                    v = a;
                } else {
                    const double b = static_cast<double>(column[stride]);
                    v = (e.w == 1.0) ? b : (1.0 - e.w) * a + e.w * b;
                }
            } else {
                v = sampleColumn(e, column, stride);
            }
            const std::size_t idx = d * n_faces + face;
            sum[idx] += v;
            if (v > vmax[idx]) vmax[idx] = v;
            if (v < vmin[idx]) vmin[idx] = v;
        }
        ++count[face];
    }
};

// ---------------------------------------------------------------------------
// The z-manifest. Carries what was asked for, what was sampled, the weights
// actually used, and the PREDICTED error budget per depth -- so a reviewer sees
// the budget without re-deriving it, and so a snapped depth can never be
// mistaken for the requested one.
// ---------------------------------------------------------------------------
inline std::string toJson(const Plan& plan,
                          const std::string& grid_fingerprint,
                          std::size_t n_faces,
                          const std::string& accumulation_window,
                          const std::string& solver_path) {
    auto num = [](double v) {
        if (!std::isfinite(v)) return std::string("null");
        std::ostringstream os;
        os.precision(17);
        os << v;
        return os.str();
    };
    auto arr = [&num](const std::vector<double>& v) {
        if (v.empty()) return std::string("null");
        std::ostringstream os;
        os << "[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) os << ", ";
            os << num(v[i]);
        }
        os << "]";
        return os.str();
    };
    std::ostringstream os;
    os << "{\n";
    os << "  \"schema\": \"ptm3d-depth-map/1\",\n";
    os << "  \"grid_fingerprint\": \"" << grid_fingerprint << "\",\n";
    os << "  \"solver_path\": \"" << solver_path << "\",\n";
    os << "  \"n_faces\": " << n_faces << ",\n";
    os << "  \"accumulation_window\": \"" << accumulation_window << "\",\n";
    os << "  \"snap_to_node\": " << (plan.snap_to_node ? "true" : "false") << ",\n";
    os << "  \"skin_depth_m\": " << num(plan.skin_depth_m) << ",\n";
    os << "  \"statistics\": [";
    for (std::size_t i = 0; i < plan.stats.size(); ++i) {
        if (i) os << ", ";
        os << "\"" << statName(plan.stats[i]) << "\"";
    }
    os << "],\n";
    os << "  \"native_z_m\": [";
    for (std::size_t i = 0; i < plan.z.size(); ++i) {
        if (i) os << ", ";
        os << num(plan.z[i]);
    }
    os << "],\n";
    os << "  \"bias_harmonics\": [";
    for (int h = 1; h <= kBiasHarmonics; ++h) { if (h > 1) os << ", "; os << h; }
    os << "],\n";
    os << "  \"bias_units\": \"Each array is indexed by bias_harmonics and is in "
          "kelvin per kelvin of THAT harmonic's amplitude in the surface "
          "temperature series. Combine as sum_h A_h * coefficient_h. "
          "'bias_instantaneous_per_A_by_harmonic' is a RIGOROUS upper bound on the "
          "error of a time series at this depth and, since |max f - max g| <= "
          "max|f-g|, on the error of the max/min maps too. "
          "'bias_max_per_A_by_harmonic' is the SMALLER typical error in the max, "
          "where the pointwise errors partly cancel; it is a scale, not a bound. "
          "Do not substitute one for the other, and do not use the fundamental "
          "alone -- harmonic h propagates with skin depth delta/sqrt(h) and its "
          "coefficient is larger, so a fundamental-only figure under-predicts (~20x "
          "on the engine's own radiatively forced series). All zero at an "
          "on-node depth, which is why snap-to-node is the default.\",\n";
    os << "  \"depths\": [\n";
    for (std::size_t i = 0; i < plan.entries.size(); ++i) {
        const Entry& e = plan.entries[i];
        os << "    {\"requested_m\": " << num(e.requested_m)
           << ", \"resolved_m\": " << num(e.resolved_m)
           << ", \"residual_m\": " << num(e.residual_m)
           << ", \"node_j\": " << e.j
           << ", \"weight_on_j_plus_1\": " << num(e.w)
           << ", \"on_node\": " << (e.on_node ? "true" : "false")
           << ", \"field\": \"" << e.field << "\""
           << ", \"bias_max_per_A_by_harmonic\": " << arr(e.bias_max_per_A)
           << ", \"bias_instantaneous_per_A_by_harmonic\": " << arr(e.bias_inst_per_A)
           << "}";
        if (i + 1 < plan.entries.size()) os << ",";
        os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
    return os.str();
}

}  // namespace depthmap
}  // namespace thermal

#endif  // PTM_THERMAL_DEPTH_MAP_PLAN_H
