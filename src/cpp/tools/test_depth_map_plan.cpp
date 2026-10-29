// test_depth_map_plan -- the refusal surface and the arithmetic of
// thermal/depth_map_plan.h, with no mesh, no solver and no GPU.
//
// This is the gate that a depth request either resolves to something a
// reviewer can check or is refused with a message naming the fix. Every
// refusal listed in the header gets its OWN arm here, because a refusal that
// is never fired is a refusal that does not exist -- the class of defect that
// let a Hapke checkpoint and a Hayne checkpoint hash equal inside the guard
// that existed to separate them.
//
// The interpolation arithmetic is checked two ways that can disagree:
//   * against hand-computed weights on a known grid, and
//   * against the damped-wave closed form, which is also what
//     test_thermal_wave_depth scores the ENGINE's solution against. If the
//     sampler and the bias predictor ever disagree about what "the value at
//     depth" means, the second check moves and the first does not.
//
// Build: part of the standalone C++ test set (no ctest, no pytest).

#include "../thermal/depth_map_plan.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace thermal::depthmap;

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << "\n";
    } else {
        std::cout << "  ok    " << what << "\n";
    }
}

void checkNear(double got, double want, double tol, const std::string& what) {
    const bool ok = std::fabs(got - want) <= tol;
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s (got %.17g, want %.17g, tol %.3g)\n",
                    what.c_str(), got, want, tol);
    } else {
        std::printf("  ok    %s (%.6g)\n", what.c_str(), got);
    }
}

// A refusal arm: the request must fail AND the message must name the fix.
void checkRefused(const Request& req, const std::vector<double>& z,
                  const std::string& must_contain, const std::string& what) {
    Plan p;
    std::string refusal;
    std::vector<std::string> warn;
    const bool ok = buildPlan(req, z.data(), static_cast<int>(z.size()), 0.0,
                              0, false, p, refusal, warn);
    ++g_checks;
    if (ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- was ACCEPTED, expected a refusal\n";
        return;
    }
    if (refusal.find(must_contain) == std::string::npos) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- refused, but the message does not "
                     "name '" << must_contain << "': " << refusal << "\n";
        return;
    }
    std::cout << "  ok    " << what << "\n        -> " << refusal << "\n";
}

// The negative half of a refusal arm: the request must still fail, and the
// message must NOT contain a string. Needed because "names the fix" is only
// half the property -- a message can name the right knob AND the wrong one in
// the same sentence, and `find(must_contain) != npos` cannot see that.
void checkRefusalLacks(const Request& req, const std::vector<double>& z,
                       const std::string& must_not_contain,
                       const std::string& what) {
    Plan p;
    std::string refusal;
    std::vector<std::string> warn;
    const bool ok = buildPlan(req, z.data(), static_cast<int>(z.size()), 0.0,
                              0, false, p, refusal, warn);
    ++g_checks;
    if (ok) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- was ACCEPTED, expected a refusal\n";
        return;
    }
    if (refusal.find(must_not_contain) != std::string::npos) {
        ++g_failed;
        std::cout << "  FAIL  " << what << " -- the message still contains '"
                  << must_not_contain << "': " << refusal << "\n";
        return;
    }
    std::cout << "  ok    " << what << "\n";
}

// A small, deliberately NON-uniform grid: uniform spacing would make several
// of these checks pass for the wrong reason (the g1-from-dz[i] falsification in
// test_thermal_wave_depth is exactly that trap).
std::vector<double> demoGrid() {
    return {0.0, 0.01, 0.025, 0.05, 0.09, 0.15, 0.25, 0.40, 0.65, 1.00};
}

Request baseRequest(std::vector<double> depths, bool snap) {
    Request r;
    r.depths_m = std::move(depths);
    r.snap_to_node = snap;
    return r;
}

}  // namespace

int main() {
    std::cout << "=== test_depth_map_plan ===\n";
    const std::vector<double> z = demoGrid();

    // ---------------------------------------------------------------- empty
    {
        std::cout << "\n[A] no depths requested is not a refusal (byte-neutral default)\n";
        Request r;
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        const bool ok = buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, p,
                                  refusal, warn);
        check(ok, "empty request accepted");
        check(p.empty(), "empty request yields an empty plan");
        check(refusal.empty(), "empty request produces no refusal text");
    }

    // ------------------------------------------------------- exact-node math
    {
        std::cout << "\n[B] on-node depths resolve exactly, with zero residual\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.0, 0.05, 1.0}, /*snap=*/true);
        const bool ok = buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, p,
                                  refusal, warn);
        check(ok, std::string("built: ") + refusal);
        check(p.size() == 3, "three entries");
        checkNear(p.entries[0].resolved_m, 0.0, 0.0, "z=0 resolves to 0");
        check(p.entries[0].j == 0 && p.entries[0].w == 0.0, "z=0 -> (j=0, w=0)");
        check(p.entries[0].on_node, "z=0 is on-node");
        checkNear(p.entries[1].resolved_m, 0.05, 0.0, "z=0.05 resolves to 0.05");
        check(p.entries[1].j == 3 && p.entries[1].w == 0.0, "z=0.05 -> (j=3, w=0)");
        // The deepest node is the special case: j is capped at n-2 and w = 1.
        check(p.entries[2].j == (int)z.size() - 2 && p.entries[2].w == 1.0,
              "deepest node -> (j=n-2, w=1)");
        checkNear(p.entries[2].resolved_m, 1.0, 0.0, "deepest node resolves to 1.0");
        check(p.entries[0].field == "temperature_z0p000m", "z=0 field name");
        check(p.entries[1].field == "temperature_z0p050m", "z=0.05 field name");
        check(p.entries[2].field == "temperature_z1p000m", "z=1.0 field name");
        for (const auto& e : p.entries) {
            checkNear(e.residual_m, 0.0, 0.0,
                      "residual is exactly zero for " + e.field);
        }
    }

    // --------------------------------------------------------------- snapping
    {
        std::cout << "\n[C] snap-to-node reports the snapped depth AND the residual\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        // 0.031 sits between nodes 0.025 (j=2) and 0.05 (j=3); w = 0.24 -> snaps
        // DOWN to 0.025. 0.20 sits between 0.15 and 0.25, i.e. a tie in exact
        // arithmetic -- and in binary floating point (0.20-0.15)/(0.25-0.15)
        // evaluates to 0.5000000000000003, so it snaps UP. That is not a defect
        // to hide: a request that lands within an ulp of a cell midpoint has no
        // meaningful "nearer" node, which is exactly why the file name carries
        // the SNAPPED depth and the residual is always reported rather than the
        // user being left to assume they got what they typed.
        auto r = baseRequest({0.031, 0.20}, /*snap=*/true);
        const bool ok = buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, p,
                                  refusal, warn);
        check(ok, std::string("built: ") + refusal);
        checkNear(p.entries[0].resolved_m, 0.025, 0.0, "0.031 snaps to 0.025");
        checkNear(p.entries[0].residual_m, -0.006, 1e-15, "residual -6 mm reported");
        check(p.entries[0].on_node, "snapped entry is marked on-node");
        checkNear(p.entries[1].resolved_m, 0.25, 0.0,
                  "a near-tie snaps to whichever node the fp weight names (0.25 here)");
        checkNear(p.entries[1].residual_m, 0.05, 1e-15,
                  "and reports the +50 mm residual, so the snap cannot pass unnoticed");
        check(p.entries[0].field == "temperature_z0p025m",
              "the file name carries the SNAPPED depth, not the requested one");

        // An EXACT tie (all values representable in binary) resolves to the
        // shallower node, deterministically: w <= 0.5 snaps down.
        const std::vector<double> zt = {0.0, 0.25, 0.5};
        Plan pt;
        std::string ref_t;
        std::vector<std::string> warn_t;
        auto rt = baseRequest({0.125}, /*snap=*/true);
        check(buildPlan(rt, zt.data(), (int)zt.size(), 0.0, 0, false, pt, ref_t,
                        warn_t),
              "exact-tie grid builds");
        checkNear(pt.entries[0].w, 0.0, 0.0, "exact w=0.5 tie snaps DOWN (w -> 0)");
        checkNear(pt.entries[0].resolved_m, 0.0, 0.0, "exact tie resolves to 0.0");
    }

    // ------------------------------------------------------------ interpolate
    {
        std::cout << "\n[D] snap off: weights are the plain linear ones, and the "
                     "sampler reproduces them\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.031}, /*snap=*/false);
        const bool ok = buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, p,
                                  refusal, warn);
        check(ok, std::string("built: ") + refusal);
        const Entry& e = p.entries[0];
        check(e.j == 2, "0.031 brackets on j=2");
        checkNear(e.w, (0.031 - 0.025) / (0.05 - 0.025), 1e-15, "w = 0.24");
        check(!e.on_node, "off-node entry is not marked on-node");
        // A column that is linear in z must be reproduced EXACTLY by the sampler.
        std::vector<double> col(z.size());
        for (size_t i = 0; i < z.size(); ++i) col[i] = 100.0 + 37.0 * z[i];
        checkNear(sampleColumn(e, col.data(), 1), 100.0 + 37.0 * 0.031, 1e-12,
                  "linear column is interpolated exactly");
        // Strided (SoA) access must give the same answer as contiguous.
        const std::ptrdiff_t S = 8;
        std::vector<double> soa(z.size() * S, -1.0);
        for (size_t i = 0; i < z.size(); ++i) soa[i * S + 3] = col[i];
        checkNear(sampleColumn(e, soa.data() + 3, S),
                  sampleColumn(e, col.data(), 1), 0.0,
                  "SoA stride gives the identical value");
        check(!warn.empty(), "snap-off emits the interpolation-bias warning");
    }

    // ------------------------------------------------------- the error budget
    {
        std::cout << "\n[E] the reported bias matches the damped-wave closed form, "
                     "and is exactly zero on a node\n";
        const double delta = 0.06;   // m, a plausible lunar diurnal skin depth
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.031, 0.05}, /*snap=*/false);
        const bool ok = buildPlan(r, z.data(), (int)z.size(), delta, 0, false, p,
                                  refusal, warn);
        check(ok, std::string("built: ") + refusal);

        // Independent re-derivation with std::complex, i.e. a different code
        // path from the header's hand-rolled real/imaginary split.
        const Entry& e = p.entries[0];
        check(e.bias_max_per_A.size() == (size_t)kBiasHarmonics &&
              e.bias_inst_per_A.size() == (size_t)kBiasHarmonics,
              "the budget is tabulated per harmonic, not once");
        const std::complex<double> I(0.0, 1.0);
        for (int h = 1; h <= kBiasHarmonics; ++h) {
            const double dh = delta / std::sqrt((double)h);
            auto E = [&](double zz) { return std::exp(-(1.0 + I) * (zz / dh)); };
            const std::complex<double> interp =
                (1.0 - e.w) * E(z[e.j]) + e.w * E(z[e.j + 1]);
            const double want_max  = std::abs(interp) - std::exp(-e.resolved_m / dh);
            const double want_inst = std::abs(interp - E(e.resolved_m));
            checkNear(e.bias_max_per_A[h - 1], want_max, 1e-14,
                      "h=" + std::to_string(h) +
                      " bias_max matches std::complex re-derivation");
            checkNear(e.bias_inst_per_A[h - 1], want_inst, 1e-14,
                      "h=" + std::to_string(h) +
                      " bias_instantaneous matches std::complex re-derivation");
            // The bound relation that the JSON's wording rests on: the
            // instantaneous coefficient must dominate the max coefficient, at
            // EVERY harmonic. If it ever did not, the manifest would be telling
            // a user that a bound is a bound when it is not.
            check(e.bias_inst_per_A[h - 1] >= std::fabs(e.bias_max_per_A[h - 1]),
                  "h=" + std::to_string(h) +
                  ": the instantaneous coefficient bounds the max coefficient");
        }

        // The trap the arrays exist to close: the coefficient GROWS with
        // harmonic number, so a fundamental-only budget under-predicts.
        check(e.bias_inst_per_A[kBiasHarmonics - 1] > e.bias_inst_per_A[0],
              "the bias coefficient grows with harmonic number (h=4 > h=1), which "
              "is why a fundamental-only figure would under-predict");

        // On a node every coefficient is identically zero, at every harmonic.
        const Entry& en = p.entries[1];
        check(en.on_node, "0.05 is on a node");
        for (int h = 0; h < kBiasHarmonics; ++h) {
            checkNear(en.bias_max_per_A[h], 0.0, 1e-15,
                      "on-node bias_max is exactly 0 at h=" + std::to_string(h + 1));
            checkNear(en.bias_inst_per_A[h], 0.0, 1e-15,
                      "on-node bias_inst is exactly 0 at h=" + std::to_string(h + 1));
        }
    }

    // ------------------------------------------------------ the accumulator
    {
        std::cout << "\n[F] the accumulator reduces avg/max/min over a window, and "
                     "the surface override reaches z=0 only\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.0, 0.05}, /*snap=*/true);
        buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, p, refusal, warn);

        Accumulator acc;
        acc.allocate(p.size(), /*faces=*/2);
        // Face 0: node 0 sweeps 100,200,300 while the override says 111,222,333.
        // Node 3 (z=0.05) holds 50,60,70.
        const double node0[3] = {100.0, 200.0, 300.0};
        const double ovr[3]   = {111.0, 222.0, 333.0};
        const double node3[3] = { 50.0,  60.0,  70.0};
        for (int t = 0; t < 3; ++t) {
            std::vector<double> col(z.size(), 0.0);
            col[0] = node0[t];
            col[3] = node3[t];
            acc.addColumn(p, 0, col.data(), 1, &ovr[t]);
            acc.addColumn(p, 1, col.data(), 1);   // no override
        }
        check(acc.ready(), "accumulator is allocated and consistent");
        check(acc.count[0] == 3 && acc.count[1] == 3,
              "the per-face sample count is what the raw merge files carry");
        checkNear(acc.vmax[0 * 2 + 0], 333.0, 0.0,
                  "z=0 max on face 0 uses the OVERRIDE, not the column's node 0");
        checkNear(acc.vmin[0 * 2 + 0], 111.0, 0.0, "z=0 min uses the override");
        checkNear(acc.sum[0 * 2 + 0] / 3.0, 222.0, 1e-13, "z=0 avg uses the override");
        checkNear(acc.vmax[0 * 2 + 1], 300.0, 0.0,
                  "with no override, z=0 reads the column's node 0");
        checkNear(acc.vmax[1 * 2 + 0], 70.0, 0.0,
                  "z=0.05 is untouched by the surface override");
        checkNear(acc.vmin[1 * 2 + 1], 50.0, 0.0, "z=0.05 min on face 1");
        checkNear(acc.sum[1 * 2 + 1] / 3.0, 60.0, 1e-13, "z=0.05 avg on face 1");

        // resetFace is what the radiation-coupling re-thermal uses: a
        // re-integrated face must REPLACE its contribution, never add a second
        // one on top of iteration 0's. Check that it leaves the face pristine
        // and its neighbor untouched.
        acc.resetFace(0);
        check(acc.count[0] == 0u, "resetFace zeroes that face's sample count");
        check(acc.sum[0 * 2 + 0] == 0.0, "resetFace zeroes that face's sum");
        check(acc.vmax[0 * 2 + 0] == -std::numeric_limits<double>::max(),
              "resetFace restores the max sentinel");
        check(acc.vmin[1 * 2 + 0] == std::numeric_limits<double>::max(),
              "resetFace covers every depth of that face");
        // ⛔ The sentinels must stay FINITE. The OpenCL mirror is compiled
        // `-cl-fast-relaxed-math`, which implies `-cl-finite-math-only`: a
        // device compiler may assume no operand of fmax/fmin is infinite, and
        // there is no OpenCL `-fno-finite-math-only`. An infinite seed could
        // then ship every max map as -inf with a correct-looking sample count.
        // Pinned here so a "tidy-up" back to infinity fails a test instead of a
        // production run.
        Accumulator fresh;
        fresh.allocate(1, 1);
        check(std::isfinite(fresh.vmax[0]) && std::isfinite(fresh.vmin[0]),
              "the freshly allocated max/min sentinels are FINITE (the OpenCL "
              "mirror is built -cl-fast-relaxed-math; an infinity would be "
              "assumed away)");
        checkNear(acc.vmax[0 * 2 + 1], 300.0, 0.0,
                  "resetFace leaves the OTHER face alone");
        check(acc.count[1] == 3u, "resetFace leaves the other face's count alone");
    }

    // ------------------------------------------------------------- size guard
    {
        std::cout << "\n[G] the size guard refuses a product that is a disk event\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.0, 0.05, 0.15, 0.25, 1.0}, true);
        const bool ok = buildPlan(r, z.data(), (int)z.size(), 0.0,
                                  2397890, /*with_raw=*/false, p, refusal, warn);
        check(ok, "5 depths x 3 stats at 2.4M faces is accepted");
        const double mib = estimateBytes(p, 2397890, false) / (1024.0 * 1024.0);
        checkNear(mib, 137.2, 0.5, "projected size (MiB) matches the plan's table");

        Request tight = r;
        tight.max_bytes = 1ull << 20;   // 1 MiB
        Plan p2;
        std::string ref2;
        std::vector<std::string> w2;
        const bool ok2 = buildPlan(tight, z.data(), (int)z.size(), 0.0,
                                   2397890, false, p2, ref2, w2);
        check(!ok2, "a 1 MiB ceiling refuses the same request");
        check(ref2.find("ceiling") != std::string::npos,
              "the refusal names the ceiling");
        check(p2.empty(), "a refused plan is left empty, not half-built");
    }

    // ---------------------------------------------------------- the refusals
    {
        std::cout << "\n[H] every refusal in the header fires, and names the fix\n";
        checkRefused(baseRequest({-0.1}, true), z, "negative",
                     "negative depth");
        checkRefused(baseRequest({std::nan("")}, true), z, "non-finite",
                     "non-finite depth");
        // ⛔ `thermal.thermal_b`, NOT `thermal.grid_b`. The C++ FIELD is
        // `config.thermal.grid_b`; the YAML KEY the parser reads is
        // `thermal_b` (config_parser.cpp: getWithDefault(thermal, "thermal_b",
        // config.thermal.grid_b)). The refusal shipped naming the field, so it
        // told the reader to increase a key that does not exist and that the
        // engine would silently ignore if they added it.
        // Both directions are
        // asserted: the old spelling is not a substring of the new one, so the
        // negative arm is what fails if it ever comes back.
        checkRefused(baseRequest({1.5}, true), z, "thermal.thermal_b",
                     "depth below the deepest node names the YAML key "
                     "thermal.thermal_b (D9)");
        checkRefusalLacks(baseRequest({1.5}, true), z, "thermal.grid_b",
                          "that refusal does NOT name the C++ field "
                          "thermal.grid_b, which is not a config key");
        checkRefused(baseRequest({0.05, 0.05}, true), z, "twice",
                     "duplicate requested depths are refused, not deduped");
        checkRefused(baseRequest({0.051, 0.052}, true), z, "resolve to",
                     "two requests snapping to one node collide on the file name");
        {
            Request r = baseRequest({0.05}, true);
            r.stats.clear();
            checkRefused(r, z, "output.map_depth_stats", "empty statistic list");
        }
        {
            Request r = baseRequest({0.05}, true);
            r.stats = {Stat::Avg, Stat::Avg};
            checkRefused(r, z, "twice", "duplicate statistic");
        }
        {
            const std::vector<double> bad = {0.0, 0.05, 0.04, 0.10};
            checkRefused(baseRequest({0.05}, true), bad, "strictly increasing",
                         "a non-monotone grid is refused");
        }
        {
            const std::vector<double> one = {0.0};
            checkRefused(baseRequest({0.0}, true), one, "at least two",
                         "a one-node grid is refused");
        }
        {
            Stat s;
            check(!statFromName("mean", s), "'mean' is not a recognized statistic");
            check(statFromName("avg", s) && s == Stat::Avg, "'avg' parses");
            check(statFromName("max", s) && s == Stat::Max, "'max' parses");
            check(statFromName("min", s) && s == Stat::Min, "'min' parses");
        }
    }

    // ------------------------------------------------------------- the manifest
    {
        std::cout << "\n[I] the z-manifest carries the snapped depth, the weights "
                     "and the budget\n";
        Plan p;
        std::string refusal;
        std::vector<std::string> warn;
        auto r = baseRequest({0.031}, /*snap=*/false);
        buildPlan(r, z.data(), (int)z.size(), 0.06, 0, false, p, refusal, warn);
        const std::string js = toJson(p, "grid-fingerprint-here", 1234,
                                      "[0,720)", "gpu_persistent");
        check(js.find("\"requested_m\": 0.031") != std::string::npos,
              "manifest states the REQUESTED depth");
        check(js.find("\"node_j\": 2") != std::string::npos,
              "manifest states the node index actually used");
        check(js.find("bias_instantaneous_per_A_by_harmonic") != std::string::npos,
              "manifest carries the time-series bias separately from the max bias");
        check(js.find("\"bias_harmonics\": [1, 2, 3, 4]") != std::string::npos,
              "manifest states which harmonics the coefficients are indexed by");
        check(js.find("\"solver_path\": \"gpu_persistent\"") != std::string::npos,
              "manifest records which solver path produced it");
        check(js.find("\"accumulation_window\": \"[0,720)\"") != std::string::npos,
              "manifest records the accumulation window");

        // A plan with no skin depth must emit null, not a fabricated number.
        Plan q;
        std::vector<std::string> w2;
        buildPlan(r, z.data(), (int)z.size(), 0.0, 0, false, q, refusal, w2);
        const std::string js2 = toJson(q, "fp", 1, "[0,1)", "cpu_simd_batch");
        check(js2.find("\"bias_max_per_A_by_harmonic\": null") != std::string::npos,
              "an unavailable budget is null, never a made-up value");
        check(!w2.empty(), "an unavailable budget also warns");
    }

    std::cout << "\n=== " << (g_checks - g_failed) << "/" << g_checks
              << " checks passed ===\n";
    if (g_failed) {
        std::cout << "FAILED: " << g_failed << " check(s)\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
