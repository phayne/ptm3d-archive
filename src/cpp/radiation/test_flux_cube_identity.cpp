// test_flux_cube_identity.cpp
// ---------------------------------------------------------------------------
// STAGE 0 of the flux-cube implementation: the test that can KILL decision D3.
//
// D3 says the flux-cube pass reconstructs the terrain's outgoing radiosity
// B_vis / B_ir on the HOST, from fields the engine already publishes, instead
// of reading B back out of the solver (which on the multi-GPU pool route would
// mean a new device->host transfer in a kernel path that is currently
// bit-identical to the CPU).
//
// That decision is worth exactly as much as the claim "the reconstruction is
// EXACT, not approximate". This file tests that claim and nothing else. If the
// identity does not close to 1e-12 relative, D3 is dead, the readback is
// unavoidable, and the schedule moves by half a day. Run this FIRST.
//
// WHAT MAKES IT DISCRIMINATING
// ----------------------------
// A test that closes the identity on a uniform enclosure proves almost nothing:
// with one emissivity, one albedo and A_src == A_bond, six different wrong
// formulas all give the right answer. So:
//
//   * Every face gets a DISTINCT emissivity, a DISTINCT Bond albedo and a
//     DISTINCT PTM-106 source albedo, with A_src != A_bond everywhere. Swapping
//     the two albedos, or dropping the (1-eps)/eps weighting, then moves the
//     answer far outside 1e-12.
//   * Arm B runs an enclosure with Sigma_j F_ij = 1.5 > 1, so vf_row_norm_ is
//     active inside the iteration. The identity must be blind to it (the row
//     norm is absorbed into the absorbed flux), and this is the arm that proves
//     it rather than assuming it.
//   * Four NEGATIVE CONTROLS deliberately mis-spell the reconstruction and are
//     REQUIRED to fail the same 1e-12 bar. A test whose negative controls pass
//     is not measuring anything; if any control comes in under tolerance this
//     file reports FAIL, not PASS.
//
// The reconstruction under test is the SHIPPED one -- radiosity_reconstruction.h
// -- not a copy written for the test. That is the point: the flux-cube pass and
// this test call the same two inline functions, so they cannot drift.
//
// Exit code = number of failures. 0 == the identity holds and D3 stands.
// ---------------------------------------------------------------------------

#include "radiation_exchange.h"
#include "radiosity_reconstruction.h"
#include "view_factor.h"
#include "radiation_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cstddef>

using radiation::RadiationConfig;
using radiation::RadiositySolver;
using radiation::ViewFactorMatrix;

namespace {

// Symmetric, fully-coupled N-face enclosure, every row summing to s.
ViewFactorMatrix buildEnclosure(int N, double s) {
    const double c = s / static_cast<double>(N - 1);
    std::vector<int> rows, cols;
    std::vector<double> vals;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i != j) { rows.push_back(i); cols.push_back(j); vals.push_back(c); }
    ViewFactorMatrix F;
    F.buildFromTriplets(static_cast<size_t>(N), rows, cols, vals);
    return F;
}

struct ArmResult {
    bool   ran = false;
    double rel_ir = 0.0;      // max|B_ir_recon  - B_ir_retained | / max|B_ir |
    double rel_vis = 0.0;     // max|B_vis_recon - B_vis_retained| / max|B_vis|
    double max_b_ir = 0.0;
    double max_b_vis = 0.0;
    // Negative controls, same normalization. Each MUST exceed the tolerance.
    double ctl_swap_albedo = 0.0;   // A_src <-> A_bond
    double ctl_bond_as_src = 0.0;   // source albedo replaced by the Bond value
    double ctl_drop_refl = 0.0;     // IR: drop the (1-eps) reflected term
    double ctl_eps_weight = 0.0;    // IR: divide by (1-eps) instead of eps
};

// One arm: build heterogeneous properties, solve two-band on the CPU with B
// retained, then reconstruct from the PUBLISHED quantities only.
ArmResult runArm(const char* label, int N, double s, bool shadow_face0) {
    ArmResult r;

    // --- Heterogeneous, all-distinct properties. eps spans lunar regolith
    // (0.95) to a dark rock analog; A_bond spans the measured site A range
    // (0.137 - 0.290) widened both ways; A_src is deliberately offset from
    // A_bond by a face-dependent amount so the two can never be confused.
    std::vector<double> areas(N, 1.0), emiss(N), bond(N), a_src(N), T(N), solar(N);
    for (int i = 0; i < N; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(N - 1);
        emiss[i] = 0.80 + 0.18 * u;                 // 0.80 .. 0.98, all distinct
        bond[i]  = 0.10 + 0.35 * u;                 // 0.10 .. 0.45, all distinct
        a_src[i] = std::min(0.90, bond[i] + 0.12 + 0.20 * (1.0 - u));  // != bond
        T[i]     = 90.0 + 260.0 * u;                // 90 K .. 350 K (PSR .. noon)
        solar[i] = 200.0 + 900.0 * u;               // 200 .. 1100 W/m^2
    }
    if (shadow_face0) { solar[0] = 0.0; T[0] = 24.5; }   // a PSR floor face

    ViewFactorMatrix F = buildEnclosure(N, s);

    // Activate two-band the way PRODUCTION does -- config key band_separated,
    // i.e. enableTwoBand() AFTER a per-face Bond albedo is installed. The
    // PTM_RAD_TWO_BAND env route is evaluated during construction, before any
    // per-face albedo can exist, so it would fall back to the gray path here
    // and silently test nothing.
    unsetenv("PTM_RAD_TWO_BAND");
    unsetenv("PTM_SOLAR_ALBEDO");
    unsetenv("PTM_RAD_EMISSIVITY");

    RadiositySolver solver(F, emiss, areas);
    RadiationConfig cfg;
    cfg.max_iterations = 5000;
    cfg.convergence_tolerance = 1e-14;   // drive it to float64 round-off
    cfg.use_acceleration = false;        // CPU fused two-band Jacobi
    cfg.use_gpu = false;
    cfg.verbose = false;
    solver.setConfig(cfg);
    solver.setSolarAlbedo(bond);              // OPERATOR (diffuse) reflectance
    solver.enableTwoBand();                   // band separation, config route
    solver.setSolarSourceAlbedo(a_src);       // PTM-106 SOURCE albedo
    solver.setRetainRadiosityB(true);         // stage-0 hook, default off

    solver.solve(T, solar);

    const std::vector<double>& q_ir  = solver.lastIndirectIR();
    const std::vector<double>& q_vis = solver.lastIndirectVisible();
    const std::vector<double>& b_ir_ref  = solver.retainedBIR();
    const std::vector<double>& b_vis_ref = solver.retainedBVis();
    // operativeEmissivity() is what the solver actually used (PTM_RAD_EMISSIVITY
    // aware); reconstructing against the REQUESTED value instead would be a
    // second, silent way to be wrong.
    const std::vector<double>& eps_op = solver.operativeEmissivity();

    if (b_ir_ref.size() != static_cast<size_t>(N) ||
        b_vis_ref.size() != static_cast<size_t>(N) ||
        q_ir.size() != static_cast<size_t>(N) ||
        q_vis.size() != static_cast<size_t>(N)) {
        std::printf("  [%s] FAIL: solver did not publish the arrays this test needs "
                    "(B_ir=%zu B_vis=%zu q_ir=%zu q_vis=%zu, want %d each)\n",
                    label, b_ir_ref.size(), b_vis_ref.size(),
                    q_ir.size(), q_vis.size(), N);
        return r;
    }

    std::string reason;
    if (!radiation::checkReconstructible(eps_op, bond, reason)) {
        std::printf("  [%s] FAIL: checkReconstructible refused: %s\n", label, reason.c_str());
        return r;
    }

    double num_ir = 0.0, num_vis = 0.0;
    double n_swap = 0.0, n_bond_src = 0.0, n_drop = 0.0, n_epsw = 0.0;
    for (int i = 0; i < N; ++i) {
        r.max_b_ir  = std::max(r.max_b_ir,  std::abs(b_ir_ref[i]));
        r.max_b_vis = std::max(r.max_b_vis, std::abs(b_vis_ref[i]));

        // --- the shipped reconstruction, the thing under test
        const double b_ir  = radiation::reconstructBIR(eps_op[i], T[i], q_ir[i]);
        const double b_vis = radiation::reconstructBVis(a_src[i], solar[i],
                                                        bond[i], q_vis[i]);
        num_ir  = std::max(num_ir,  std::abs(b_ir  - b_ir_ref[i]));
        num_vis = std::max(num_vis, std::abs(b_vis - b_vis_ref[i]));

        // --- negative controls: each must move the answer, or the test is blind
        const double c_swap = radiation::reconstructBVis(bond[i], solar[i],
                                                          a_src[i], q_vis[i]);
        n_swap = std::max(n_swap, std::abs(c_swap - b_vis_ref[i]));

        const double c_bond_src = radiation::reconstructBVis(bond[i], solar[i],
                                                              bond[i], q_vis[i]);
        n_bond_src = std::max(n_bond_src, std::abs(c_bond_src - b_vis_ref[i]));

        const double T2 = T[i] * T[i];
        const double emission = eps_op[i] * radiation::kReconSigmaSB * T2 * T2;
        const double c_drop = emission;                       // no reflected IR term
        n_drop = std::max(n_drop, std::abs(c_drop - b_ir_ref[i]));

        const double c_epsw = emission + (1.0 - eps_op[i]) *
                              (q_ir[i] / (1.0 - eps_op[i]));  // wrong denominator
        n_epsw = std::max(n_epsw, std::abs(c_epsw - b_ir_ref[i]));
    }

    const double den_ir  = std::max(r.max_b_ir,  1e-30);
    const double den_vis = std::max(r.max_b_vis, 1e-30);
    r.rel_ir  = num_ir  / den_ir;
    r.rel_vis = num_vis / den_vis;
    r.ctl_swap_albedo = n_swap / den_vis;
    r.ctl_bond_as_src = n_bond_src / den_vis;
    r.ctl_drop_refl   = n_drop / den_ir;
    r.ctl_eps_weight  = n_epsw / den_ir;
    r.ran = true;

    std::printf("  [%s] N=%d  Sigma_j F_ij=%.2f%s\n", label, N, s,
                s > 1.0 ? "  (> 1: vf_row_norm_ ACTIVE)" : "");
    std::printf("      max|B_ir | = %10.4f W/m^2   relative identity error = %.3e\n",
                r.max_b_ir, r.rel_ir);
    std::printf("      max|B_vis| = %10.4f W/m^2   relative identity error = %.3e\n",
                r.max_b_vis, r.rel_vis);
    std::printf("      negative controls (each MUST exceed 1e-12):\n");
    std::printf("        A_src <-> A_bond swapped          : %.3e\n", r.ctl_swap_albedo);
    std::printf("        Bond albedo used as source albedo : %.3e\n", r.ctl_bond_as_src);
    std::printf("        IR reflected term dropped         : %.3e\n", r.ctl_drop_refl);
    std::printf("        IR weighted by (1-eps) not eps    : %.3e\n", r.ctl_eps_weight);
    return r;
}

}  // namespace

int main() {
    std::printf("\n=== STAGE 0: host reconstruction of B_vis / B_ir == the solver's own B ===\n");
    std::printf("Decision under test: D3 (reconstruct on the host; no GPU readback).\n");
    std::printf("Bar: max|B_recon - B_retained| / max|B| < 1e-12 on the CPU two-band route.\n\n");

    const double TOL = 1e-12;
    int fail = 0;

    // Arm A: open scene (Sigma F < 1, some flux escapes to sky), face 0 a PSR floor.
    ArmResult a = runArm("A open, Sigma F = 0.70", 48, 0.70, true);
    std::printf("\n");
    // Arm B: concave scene with Sigma F > 1, so the row normalization is live.
    ArmResult b = runArm("B concave, Sigma F = 1.50", 48, 1.50, true);
    std::printf("\n");

    const ArmResult* arms[2] = {&a, &b};
    const char* names[2] = {"A", "B"};
    for (int k = 0; k < 2; ++k) {
        const ArmResult& x = *arms[k];
        if (!x.ran) { ++fail; continue; }
        if (!(x.rel_ir < TOL)) {
            std::printf("  FAIL arm %s: IR identity error %.3e >= %.1e\n", names[k], x.rel_ir, TOL);
            ++fail;
        }
        if (!(x.rel_vis < TOL)) {
            std::printf("  FAIL arm %s: visible identity error %.3e >= %.1e\n", names[k], x.rel_vis, TOL);
            ++fail;
        }
        // A negative control that PASSES means the arm cannot tell right from wrong.
        struct Ctl { const char* what; double v; };
        const Ctl ctls[4] = {
            {"A_src<->A_bond swap", x.ctl_swap_albedo},
            {"Bond-as-source",      x.ctl_bond_as_src},
            {"IR reflected dropped", x.ctl_drop_refl},
            {"IR (1-eps) weight",   x.ctl_eps_weight},
        };
        for (const Ctl& c : ctls) {
            if (!(c.v > TOL)) {
                std::printf("  FAIL arm %s: negative control '%s' came in at %.3e, under the "
                            "tolerance -- this arm does not discriminate and its PASS is empty.\n",
                            names[k], c.what, c.v);
                ++fail;
            }
        }
    }

    std::ofstream out("output/flux_cubes/stage0_identity.csv");
    if (out) {
        out << "arm,rel_err_B_ir,rel_err_B_vis,max_B_ir_Wm2,max_B_vis_Wm2,"
               "ctl_swap_albedo,ctl_bond_as_src,ctl_drop_refl,ctl_eps_weight\n";
        for (int k = 0; k < 2; ++k) {
            const ArmResult& x = *arms[k];
            out << names[k] << "," << x.rel_ir << "," << x.rel_vis << ","
                << x.max_b_ir << "," << x.max_b_vis << "," << x.ctl_swap_albedo << ","
                << x.ctl_bond_as_src << "," << x.ctl_drop_refl << "," << x.ctl_eps_weight << "\n";
        }
        out.close();
        std::printf("\n  artifact: output/flux_cubes/stage0_identity.csv\n");
    }

    std::printf("\n  STAGE 0 VERDICT: %s\n", fail == 0
        ? "PASS -- the reconstruction is exact; D3 stands, no GPU readback needed."
        : "FAIL -- D3 is dead; B must be read back from the solver.");
    std::printf("  (%d failure%s)\n\n", fail, fail == 1 ? "" : "s");
    return fail;
}
