// test_two_band_rownorm.cpp
// ---------------------------------------------------------------------------
// Decisive severity test for BUGS.md PTM-08 (vf_row_norm_ applied in the Jacobi
// radiosity iteration but NOT in Gauss-Seidel), in the two-band VISIBLE regime
// where it bites (Bond albedo A up to ~0.5 Europa / ~0.9 frost).
//
// A synthetic, fully-coupled concave enclosure is built with a controlled
// row sum s = Σ_j F_ij > 1 (equal areas, symmetric F, so it is reciprocal).
// The M1 centroid view-factor estimator produces exactly this non-physical
// Σ > 1 overshoot on fine/concave meshes, which is why vf_row_norm_ (=1/max(1,s))
// exists. RadiositySolver::initCommon reports "N/N rows > 1 renormalized".
//
// The two-band visible band solves (I - A F) B_vis = A Q_solar. Its contraction
// rate is bounded by ρ(A F) ≤ A·s:
//   * Jacobi applies vf_row_norm_ inside the iteration → effective A·F̂ with
//     row sums ≤ 1 → gain ≤ A < 1 → contraction guaranteed (derivation Eq 7).
//   * Gauss-Seidel (production default use_acceleration=true, sparse path) sums
//     Σ_j F_ij B_j RAW → gain A·s. For A·s ≥ 1 it DIVERGES; for A·s < 1 it
//     converges but OVER-COUPLES to a different (wrong) radiosity field.
//
// This test drives the PUBLIC RadiositySolver::solve() (two-band via env) with
// use_acceleration true (GS) and false (Jacobi) and compares the absorbed
// visible indirect flux on the shadowed floor face.
//
//   BEFORE the A1 fix : GS diverges (A·s≥1) or over-couples (A·s<1) → mismatch.
//   AFTER  the A1 fix : GS converges AND matches Jacobi to machine precision.
//
// Exit code = number of "should-converge" configs where GS failed to converge
// or disagreed with Jacobi beyond tolerance. 0 == fixed.
//
// Artifact: output/two_band_prod/a0_gs_vs_jacobi.csv
// ---------------------------------------------------------------------------

#include "radiation_exchange.h"
#include "view_factor.h"
#include "radiation_config.h"

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

using radiation::RadiositySolver;
using radiation::ViewFactorMatrix;
using radiation::RadiationConfig;

namespace {

// Build a symmetric, fully-coupled N-face enclosure with F_ij = c (i != j),
// so every row sums to s = (N-1)*c. Equal areas => A_i F_ij = A_j F_ji (reciprocal).
ViewFactorMatrix buildUniformEnclosure(int N, double s) {
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

struct Result {
    bool converged;
    int  iters;
    double q_vis_floor;   // absorbed indirect VISIBLE flux on the shadowed floor face 0
    double q_ir_floor;    // absorbed indirect IR flux on face 0 (diagnostic)
    double b_vis_max;     // max |visible radiosity absorbed| across faces (blowup detector)
};

// Run one config with a FRESH solver (the visible cache is per-instance and keyed
// on solar_flux; a fresh instance guarantees an independent GS vs Jacobi solve).
Result runOne(const ViewFactorMatrix& F, const std::vector<double>& areas,
              const std::vector<double>& emiss, const std::vector<double>& solar,
              const std::vector<double>& T, bool gauss_seidel) {
    RadiositySolver solver(F, emiss, areas);
    RadiationConfig cfg;
    cfg.max_iterations = 2000;
    cfg.convergence_tolerance = 1e-10;
    cfg.use_acceleration = gauss_seidel;   // true => Gauss-Seidel, false => Jacobi
    cfg.use_gpu = false;
    cfg.verbose = false;
    solver.setConfig(cfg);

    std::vector<double> ind = solver.solve(T, solar);
    const auto& qvis = solver.lastIndirectVisible();
    const auto& qir  = solver.lastIndirectIR();

    Result r;
    r.converged = solver.didConverge();
    r.iters     = solver.getLastIterations();
    r.q_vis_floor = qvis.empty() ? std::nan("") : qvis[0];
    r.q_ir_floor  = qir.empty()  ? std::nan("") : qir[0];
    double m = 0.0;
    for (double v : qvis) m = std::max(m, std::abs(v));
    r.b_vis_max = m;
    return r;
}

// Build a symmetric enclosure with a chosen row sum (F_sky = 1 - s > 0 when s < 1
// so some scattered visible escapes to the sky — a physical open scene).
ViewFactorMatrix buildOpenEnclosure(int N, double s) { return buildUniformEnclosure(N, s); }

// ---------------------------------------------------------------------------
// A2: per-face incidence-consistent albedo energy closure. The surface BC absorbs
// (1 - A_BC[i]) Q_i directly; the radiosity reflects A_solver[i] Q_i into the
// enclosure. The SYSTEM closes iff A_solver[i] == A_BC[i] per face:
//   residual = P_incident - P_direct_abs(A_BC) - P_launched(A_solver)
//            = Σ (A_BC[i] - A_solver[i]) Q_i area_i.
// With the per-face API (setSolarAlbedo(A_BC)) this is 0; with a UNIFORM env scalar
// A_bar it leaks Σ (A_BC[i] - A_bar) Q_i area_i on a non-uniform-albedo case. The
// bowl closed to 1e-14 ONLY because it used one uniform A for both sides.
// Returns number of A2 failures.
int runA2() {
    const int    N   = 6;
    const double EPS = 0.96;
    const double s   = 0.7;                 // F_sky = 0.30 (open scene)

    // Face 0 = shadowed floor (Q=0). Sunlit walls carry DECREASING flux while their
    // incidence albedo INCREASES (grazing walls: less flux, brighter) so the
    // flux-weighted mean albedo != the plain mean => a uniform scalar leaks net.
    std::vector<double> areas(N, 1.0), emiss(N, EPS), T(N, 100.0);
    std::vector<double> Q      = {   0.0, 1000.0, 900.0, 800.0, 700.0, 600.0};
    std::vector<double> A_BC   = {   0.5,    0.2,  0.35,   0.5,  0.65,   0.8}; // per-face incidence albedo
    double A_bar = 0.0; int nlit = 0;
    for (int i = 1; i < N; ++i) { A_bar += A_BC[i]; ++nlit; }
    A_bar /= nlit;                          // plain mean of the sunlit albedos = 0.5
    std::vector<double> A_uniform(N, A_bar);

    ViewFactorMatrix F = buildOpenEnclosure(N, s);

    // Activate two-band (env), then override the albedo per-face via the A2 API.
    setenv("PTM_SOLAR_ALBEDO", "0.5", 1);
    setenv("PTM_RAD_TWO_BAND", "1", 1);

    auto measure = [&](const std::vector<double>& A_solver) {
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg;
        cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-12;
        cfg.use_acceleration = true; cfg.use_gpu = false; cfg.verbose = false;
        solver.setConfig(cfg);
        solver.setSolarAlbedo(A_solver);          // A2: per-face (time-updatable) albedo
        solver.solve(T, Q);
        const auto& qvis = solver.lastIndirectVisible();
        double P_inc = 0, P_dir_abs = 0, P_launched = 0, P_absorbed = 0;
        for (int i = 0; i < N; ++i) {
            P_inc      += areas[i] * Q[i];
            P_dir_abs  += areas[i] * (1.0 - A_BC[i]) * Q[i];     // BC absorbs with A_BC
            P_launched += areas[i] * A_solver[i]    * Q[i];      // solver reflects with A_solver
            P_absorbed += areas[i] * qvis[i];                    // multiply-scattered visible absorbed
        }
        const double P_escaped = P_launched - P_absorbed;        // to sky (internal closure)
        const double sys_resid = P_inc - P_dir_abs - P_launched; // == Σ(A_BC-A_solver)Q a
        return std::array<double,5>{{P_inc, P_dir_abs, P_absorbed, P_escaped, sys_resid}};
    };

    auto per   = measure(A_BC);       // A2 fix: A_solver == A_BC per face
    auto unif  = measure(A_uniform);  // legacy: uniform env scalar A_bar

    const double leak_per_pct  = 100.0 * per[4]  / per[0];
    const double leak_unif_pct = 100.0 * unif[4] / unif[0];

    std::printf("\n=== A2: per-face incidence-consistent albedo energy closure ===\n");
    std::printf("  N=%d  eps=%.3f  s=Σ_jF_ij=%.2f (F_sky=%.2f)  A_bar(uniform)=%.4f\n",
                N, EPS, s, 1.0 - s, A_bar);
    std::printf("  A_BC (per-face) = {0.5(floor),0.20,0.35,0.50,0.65,0.80}  Q decreasing 1000..600\n\n");
    std::printf("  [PER-FACE  A_solver=A_BC ]  P_inc=%.4f P_dir_abs=%.4f P_absorbed_vis=%.4f P_escaped=%.4f\n",
                per[0], per[1], per[2], per[3]);
    std::printf("        system energy leak = %.4f W  (%.4e %% of incident)  <-- must be ~0\n", per[4], leak_per_pct);
    std::printf("  [UNIFORM   A_solver=A_bar]  P_inc=%.4f P_dir_abs=%.4f P_absorbed_vis=%.4f P_escaped=%.4f\n",
                unif[0], unif[1], unif[2], unif[3]);
    std::printf("        system energy leak = %.4f W  (%.4f %% of incident)  <-- LEAKS (A_solver!=A_BC)\n\n",
                unif[4], leak_unif_pct);

    std::ofstream out("output/two_band_prod/a2_energy_closure.csv");
    out << "case,P_incident,P_direct_abs,P_absorbed_vis,P_escaped_sky,sys_leak_W,sys_leak_pct\n";
    out << "per_face_A_solver=A_BC," << per[0] << "," << per[1] << "," << per[2] << ","
        << per[3] << "," << per[4] << "," << leak_per_pct << "\n";
    out << "uniform_A_solver=A_bar," << unif[0] << "," << unif[1] << "," << unif[2] << ","
        << unif[3] << "," << unif[4] << "," << leak_unif_pct << "\n";
    out.close();

    int fail = 0;
    // A2 passes iff per-face closes (~0) AND the uniform scalar demonstrably leaks
    // (so the test proves the fix does something on a non-uniform case).
    if (std::abs(leak_per_pct) > 1e-9) { std::printf("  A2 FAIL: per-face did not close.\n"); ++fail; }
    if (std::abs(leak_unif_pct) < 0.5) { std::printf("  A2 FAIL: uniform case did not leak (test not exercising A2).\n"); ++fail; }
    std::printf("  A2 verdict: per-face closes to %.2e %%; uniform leaks %.3f %%.  %s\n\n",
                leak_per_pct, leak_unif_pct, fail == 0 ? "PASS" : "FAIL");
    std::printf("    artifact: output/two_band_prod/a2_energy_closure.csv\n\n");
    return fail;
}

// ---------------------------------------------------------------------------
// B1: GPU-vs-CPU two-band equivalence on a SMALL mesh (Σ_j F_ij < 1 so the GPU
// two-band gate is satisfied). The GPU path (solveGPUTwoBand) reuses the verified
// gray kernels twice; float32 => match to GPU precision, not bit-identical.
// Returns 0 on PASS or SKIP (GPU unavailable), 1 on a real GPU!=CPU mismatch.
int runB1() {
    const int    N   = 64;
    const double EPS = 0.96, S = 1361.0, s = 0.7, A = 0.5;
    std::vector<double> areas(N, 1.0), emiss(N, EPS), T(N, 250.0);
    std::vector<double> solar(N, S); solar[0] = 0.0;      // face 0 = shadowed floor
    ViewFactorMatrix F = buildOpenEnclosure(N, s);

    setenv("PTM_SOLAR_ALBEDO", "0.5", 1);
    setenv("PTM_RAD_TWO_BAND", "1", 1);

    std::printf("\n=== B1: GPU-vs-CPU two-band equivalence (N=%d, s=%.2f, A=%.2f) ===\n", N, s, A);

    // CPU two-band reference.
    std::vector<double> cpu_ind, cpu_vis, cpu_ir;
    {
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg; cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-8;
        cfg.use_acceleration = false; cfg.use_gpu = false; cfg.verbose = false;
        solver.setConfig(cfg);
        cpu_ind = solver.solve(T, solar);
        cpu_vis = solver.lastIndirectVisible();
        cpu_ir  = solver.lastIndirectIR();
    }

#ifdef PTM_GPU_OPENCL
    try {
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg; cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-6;
        cfg.use_acceleration = false; cfg.use_gpu = true; cfg.verbose = false;
        solver.setConfig(cfg);
        solver.initGPU();
        if (!solver.isGPUTwoBand()) {
            std::printf("  SKIP: GPU two-band did not engage (no GPU / init fell back to CPU).\n\n");
            return 0;
        }
        std::vector<double> gpu_ind = solver.solve(T, solar);
        std::vector<double> gpu_vis = solver.lastIndirectVisible();
        std::vector<double> gpu_ir  = solver.lastIndirectIR();

        double max_rel_tot = 0.0, max_rel_vis = 0.0, max_rel_ir = 0.0;
        for (int i = 0; i < N; ++i) {
            auto rel = [](double a, double b){ return std::abs(b) > 1e-12 ? std::abs(a-b)/std::abs(b) : std::abs(a-b); };
            max_rel_tot = std::max(max_rel_tot, rel(gpu_ind[i], cpu_ind[i]));
            if (i < (int)gpu_vis.size() && i < (int)cpu_vis.size()) max_rel_vis = std::max(max_rel_vis, rel(gpu_vis[i], cpu_vis[i]));
            if (i < (int)gpu_ir.size()  && i < (int)cpu_ir.size())  max_rel_ir  = std::max(max_rel_ir,  rel(gpu_ir[i],  cpu_ir[i]));
        }
        std::printf("  floor(face0)  CPU: total=%.6g vis=%.6g ir=%.6g\n", cpu_ind[0], cpu_vis[0], cpu_ir[0]);
        std::printf("  floor(face0)  GPU: total=%.6g vis=%.6g ir=%.6g\n", gpu_ind[0], gpu_vis[0], gpu_ir[0]);
        std::printf("  max rel diff (all faces): total=%.3e vis=%.3e ir=%.3e\n", max_rel_tot, max_rel_vis, max_rel_ir);

        std::ofstream out("output/two_band_prod/b1_gpu_vs_cpu.csv");
        out << "quantity,cpu_floor,gpu_floor,max_rel_diff\n";
        out << "total," << cpu_ind[0] << "," << gpu_ind[0] << "," << max_rel_tot << "\n";
        out << "visible," << cpu_vis[0] << "," << gpu_vis[0] << "," << max_rel_vis << "\n";
        out << "ir," << cpu_ir[0] << "," << gpu_ir[0] << "," << max_rel_ir << "\n";
        out.close();

        // float32 SpMV: 1e-3 relative is the same tolerance band as the gray GPU-vs-CPU path.
        const bool ok = (max_rel_tot < 1e-3);
        std::printf("  B1 verdict: %s (tol 1e-3, float32)   artifact: output/two_band_prod/b1_gpu_vs_cpu.csv\n\n",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  SKIP: GPU two-band threw (%s); CPU path is the safe fallback.\n\n", e.what());
        return 0;
    }
#else
    std::printf("  SKIP: built without PTM_GPU_OPENCL.\n\n");
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// B1c: GPU-vs-CPU two-band equivalence on a mesh that DOES carry Σ_j F_ij > 1.
//
// This is the case the old gate refused outright: initGPU() used to scan
// vf_row_norm_ and force the CPU two-band solve if ANY row had been normalized,
// because the GPU Jacobi kernel omitted the row norm. The GPU kernel now applies
// vf_row_norm_ inside the iteration (uploadRowNorm), so the two paths run the
// same operator ρ·(norm⊙F) and must agree.
//
// DISCRIMINATING in three independent ways:
//   * If the GPU path does not engage (old gate, or a gate that still keys on
//     Σ F > 1), isGPUTwoBand() is false and this FAILS -- it does not SKIP the
//     way B1 does, because a silent CPU fallback is precisely the defect.
//   * If the row norm were applied on the device AND again on the host, the GPU
//     flux would be low by 1/s ~ 33% here -- far outside the 1e-3 float32 band.
//   * If the row norm were NOT applied on the device at all, the GPU visible
//     band would iterate gain A·s = 0.75 instead of A·min(s,1) = 0.5 and
//     over-couple to a different fixed point (the PTM-08 error, ~2x on the floor).
// Only "applied exactly once, inside the iteration" passes.
//
// Returns 0 on PASS or on a genuine no-GPU SKIP, 1 on mismatch or non-engagement.
int runB1c() {
#ifdef PTM_GPU_OPENCL
    const int    N   = 64;
    const double EPS = 0.96, S = 1361.0;
    const double s   = 1.5;   // Σ_j F_ij > 1: every row gets normalized to 1
    const double A   = 0.5;   // A·s = 0.75 raw vs A·min(s,1) = 0.50 normalized
    std::vector<double> areas(N, 1.0), emiss(N, EPS), T(N, 250.0);
    std::vector<double> solar(N, S); solar[0] = 0.0;      // face 0 = shadowed floor
    ViewFactorMatrix F = buildUniformEnclosure(N, s);

    setenv("PTM_SOLAR_ALBEDO", "0.5", 1);
    setenv("PTM_RAD_TWO_BAND", "1", 1);

    std::printf("=== B1c: GPU-vs-CPU two-band on Σ_j F_ij = %.2f > 1 (N=%d, A=%.2f) ===\n",
                s, N, A);

    // CPU two-band reference (Jacobi; row norm applied inside the iteration).
    std::vector<double> cpu_ind, cpu_vis, cpu_ir;
    {
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg; cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-8;
        cfg.use_acceleration = false; cfg.use_gpu = false; cfg.verbose = false;
        solver.setConfig(cfg);
        cpu_ind = solver.solve(T, solar);
        cpu_vis = solver.lastIndirectVisible();
        cpu_ir  = solver.lastIndirectIR();
    }

    try {
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg; cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-6;
        cfg.use_acceleration = false; cfg.use_gpu = true; cfg.verbose = false;
        solver.setConfig(cfg);
        solver.initGPU();
        if (!solver.isGPUTwoBand()) {
            // Distinguish "no GPU on this host" (a legitimate skip, which B1
            // above would already have reported) from "a gate refused a Σ F > 1
            // mesh" (the defect this test exists to catch).
            std::printf("  B1c FAIL: GPU two-band did NOT engage on a Σ_j F_ij > 1 mesh. "
                        "Either no GPU is present (B1 would have skipped too) or a gate is "
                        "still vetoing normalized rows.\n\n");
            return 1;
        }
        std::vector<double> gpu_ind = solver.solve(T, solar);
        std::vector<double> gpu_vis = solver.lastIndirectVisible();
        std::vector<double> gpu_ir  = solver.lastIndirectIR();

        double max_rel_tot = 0.0, max_rel_vis = 0.0, max_rel_ir = 0.0;
        auto rel = [](double a, double b){
            return std::abs(b) > 1e-12 ? std::abs(a - b) / std::abs(b) : std::abs(a - b);
        };
        for (int i = 0; i < N; ++i) {
            max_rel_tot = std::max(max_rel_tot, rel(gpu_ind[i], cpu_ind[i]));
            if (i < (int)gpu_vis.size() && i < (int)cpu_vis.size())
                max_rel_vis = std::max(max_rel_vis, rel(gpu_vis[i], cpu_vis[i]));
            if (i < (int)gpu_ir.size() && i < (int)cpu_ir.size())
                max_rel_ir = std::max(max_rel_ir, rel(gpu_ir[i], cpu_ir[i]));
        }
        std::printf("  floor(face0)  CPU: total=%.6g vis=%.6g ir=%.6g\n",
                    cpu_ind[0], cpu_vis[0], cpu_ir[0]);
        std::printf("  floor(face0)  GPU: total=%.6g vis=%.6g ir=%.6g\n",
                    gpu_ind[0], gpu_vis[0], gpu_ir[0]);
        std::printf("  max rel diff (all faces): total=%.3e vis=%.3e ir=%.3e\n",
                    max_rel_tot, max_rel_vis, max_rel_ir);

        std::ofstream out("output/two_band_prod/b1c_gpu_vs_cpu_rowsum_over_1.csv");
        out << "quantity,cpu_floor,gpu_floor,max_rel_diff\n";
        out << "total," << cpu_ind[0] << "," << gpu_ind[0] << "," << max_rel_tot << "\n";
        out << "visible," << cpu_vis[0] << "," << gpu_vis[0] << "," << max_rel_vis << "\n";
        out << "ir," << cpu_ir[0] << "," << gpu_ir[0] << "," << max_rel_ir << "\n";
        out.close();

        const bool ok = (max_rel_tot < 1e-3);
        std::printf("  B1c verdict: %s (tol 1e-3, float32)   artifact: "
                    "output/two_band_prod/b1c_gpu_vs_cpu_rowsum_over_1.csv\n\n",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  SKIP: GPU two-band threw (%s); CPU path is the safe fallback.\n\n",
                    e.what());
        return 0;
    }
#else
    std::printf("=== B1c: SKIP (built without PTM_GPU_OPENCL). ===\n\n");
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// B1d: with a multi-GPU pool active, the state flags and the per-timestep
// solve() must agree with the pool being up.
//
// initGPU()'s pool rung used to return without ever setting use_gpu_ or
// two_band_gpu_ok_ (both are set inside the single-GPU lambda). Two consequences,
// both silent: isGPUTwoBand() reported false on an ACTIVE GPU pool, and the
// per-timestep solve() -- whose guard is `use_gpu_ && gpu_solver_`, and
// gpu_solver_ is null on the pool rung -- fell through to the float64 CPU
// solver. So `radiation.gpu_devices: 2` quietly changed the solver path for
// every caller that is not the coupled batch sweep.
//
// DISCRIMINATING: the pool worker is arithmetically the same code as
// solveGPUTwoBand, so a pooled single solve must be BIT-IDENTICAL to the
// single-GPU one. A CPU fallback is float64 and is not (it differs in the
// float32 ulps this test prints). Requires >= 2 GPUs; SKIPs otherwise, and
// SKIPs if the pool declines to build (that is a capability fact, not a defect).
int runB1d() {
#ifdef PTM_GPU_OPENCL
    const int    N   = 64;
    const double EPS = 0.96, S = 1361.0, s = 0.7, A = 0.5;
    std::vector<double> areas(N, 1.0), emiss(N, EPS), T(N, 250.0);
    std::vector<double> solar(N, S); solar[0] = 0.0;
    ViewFactorMatrix F = buildOpenEnclosure(N, s);

    setenv("PTM_SOLAR_ALBEDO", "0.5", 1);
    setenv("PTM_RAD_TWO_BAND", "1", 1);

    std::printf("=== B1d: multi-GPU pool state + per-timestep solve() (N=%d, s=%.2f, A=%.2f) ===\n",
                N, s, A);

    auto make_cfg = [](int devices) {
        RadiationConfig cfg;
        cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-6;
        cfg.use_acceleration = false; cfg.use_gpu = true; cfg.verbose = false;
        cfg.gpu_devices = devices;
        return cfg;
    };

    std::vector<double> single_ind;
    try {
        RadiositySolver solver(F, emiss, areas);
        solver.setConfig(make_cfg(1));
        solver.initGPU();
        if (!solver.isGPUTwoBand()) {
            std::printf("  SKIP: single-GPU two-band did not engage (no GPU here).\n\n");
            return 0;
        }
        single_ind = solver.solve(T, solar);
    } catch (const std::exception& e) {
        std::printf("  SKIP: single-GPU reference threw (%s).\n\n", e.what());
        return 0;
    }

    try {
        RadiositySolver solver(F, emiss, areas);
        solver.setConfig(make_cfg(2));
        solver.initGPU();
        if (!solver.hasGPUPool()) {
            std::printf("  SKIP: the pool did not build (fewer than 2 GPUs on this host).\n\n");
            return 0;
        }
        const bool flag_ok = solver.isGPUTwoBand();
        std::vector<double> pool_ind = solver.solve(T, solar);

        double maxabs = 0.0;
        int n_bitdiff = 0;
        for (int i = 0; i < N; ++i) {
            maxabs = std::max(maxabs, std::abs(pool_ind[i] - single_ind[i]));
            if (pool_ind[i] != single_ind[i]) ++n_bitdiff;
        }
        std::printf("  pool active: isGPUTwoBand()=%s (want true)\n", flag_ok ? "true" : "false");
        std::printf("  per-timestep solve(): max|pool - singleGPU| = %.3e over %d faces "
                    "(%d not bit-identical; want 0)\n", maxabs, N, n_bitdiff);

        int fail = 0;
        if (!flag_ok) {
            std::printf("  B1d FAIL: isGPUTwoBand() reports CPU while the GPU pool is active.\n");
            ++fail;
        }
        if (n_bitdiff != 0) {
            std::printf("  B1d FAIL: the pooled per-timestep solve is not bit-identical to the "
                        "single-GPU solve -- it ran a different solver (CPU fallback?).\n");
            ++fail;
        }
        std::printf("  B1d verdict: %s\n\n", fail == 0 ? "PASS" : "FAIL");
        return fail ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("  SKIP: pool init threw (%s).\n\n", e.what());
        return 0;
    }
#else
    std::printf("=== B1d: SKIP (built without PTM_GPU_OPENCL). ===\n\n");
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// B1b (BUGS.md PTM-11): the single-GPU two-band gate is ORDERING-SENSITIVE.
// two_band_gpu_ok_ is set inside initGPU() ONLY if two_band_ is already true at that
// moment (radiation_exchange.cpp:359-360). So the per-face Bond albedo + enableTwoBand()
// MUST be sourced BEFORE initGPU() — which is exactly what RadiationExchangeManager::
// prepare() sequences (setSolarAlbedo -> enableTwoBand -> initGPU), and what the
// ptm_thermal_sim caller now does by sourcing the albedo BEFORE prepare(). If they are
// sourced AFTER initGPU() (the PTM-11 bug), the gate never engages and solve() falls to
// the CPU two-band path (isGPUTwoBand() == false; enableTwoBand() warns).
//
// DISCRIMINATING test for the fix's mechanism: the SAME solver, SAME small Σ_j F_ij ≤ 1
// mesh, run in BOTH orderings. isGPUTwoBand() must be TRUE for the correct order and
// FALSE for the buggy order. Two-band is driven ONLY by the explicit API here (the env
// vars are unset), so the constructor does not set two_band_ before initGPU and the buggy
// ordering is faithfully reproduced. If GPU two-band cannot engage at all (no GPU / init
// fell back), we SKIP (as B1 does). Returns 0 on PASS or SKIP, 1 if the ordering did NOT
// discriminate (mechanism broken).
int runB1b() {
#ifdef PTM_GPU_OPENCL
    unsetenv("PTM_RAD_TWO_BAND");
    unsetenv("PTM_SOLAR_ALBEDO");

    const int    N = 64;
    const double EPS = 0.96, s = 0.7, A = 0.5;
    std::vector<double> areas(N, 1.0), emiss(N, EPS), A_vec(N, A);
    ViewFactorMatrix F = buildOpenEnclosure(N, s);

    std::printf("=== B1b: PTM-11 init-ordering gate discrimination (N=%d, s=%.2f, A=%.2f) ===\n",
                N, s, A);

    auto make_cfg = []() {
        RadiationConfig cfg;
        cfg.max_iterations = 2000; cfg.convergence_tolerance = 1e-6;
        cfg.use_acceleration = false; cfg.use_gpu = true; cfg.verbose = false;
        return cfg;
    };

    bool correct_engaged = false, buggy_engaged = false;

    try {
        // CORRECT order (the fix): setSolarAlbedo -> enableTwoBand -> initGPU.
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg = make_cfg();
        solver.setConfig(cfg);
        solver.setSolarAlbedo(A_vec);
        solver.enableTwoBand();
        solver.initGPU();
        correct_engaged = solver.isGPUTwoBand();
    } catch (const std::exception& e) {
        std::printf("  SKIP: GPU threw in correct-order init (%s).\n\n", e.what());
        return 0;
    }

    if (!correct_engaged) {
        std::printf("  SKIP: GPU two-band did not engage even in the correct order "
                    "(no GPU / init fell back).\n\n");
        return 0;
    }

    try {
        // BUGGY order (PTM-11): initGPU -> setSolarAlbedo -> enableTwoBand.
        RadiositySolver solver(F, emiss, areas);
        RadiationConfig cfg = make_cfg();
        solver.setConfig(cfg);
        solver.initGPU();          // two_band_ still false here -> gray GPU init
        solver.setSolarAlbedo(A_vec);
        solver.enableTwoBand();    // sets two_band_ AFTER init -> gate missed (warns)
        buggy_engaged = solver.isGPUTwoBand();
    } catch (const std::exception& e) {
        std::printf("  SKIP: GPU threw in buggy-order init (%s).\n\n", e.what());
        return 0;
    }

    const bool ok = correct_engaged && !buggy_engaged;
    std::printf("  correct order (albedo/two-band BEFORE initGPU): isGPUTwoBand=%s (want true)\n",
                correct_engaged ? "true" : "false");
    std::printf("  buggy   order (albedo/two-band AFTER  initGPU): isGPUTwoBand=%s (want false)\n",
                buggy_engaged ? "true" : "false");
    std::printf("  B1b verdict: %s (ordering %s the GPU two-band gate)\n\n",
                ok ? "PASS" : "FAIL",
                ok ? "discriminates" : "does NOT discriminate");
    return ok ? 0 : 1;
#else
    std::printf("=== B1b: SKIP (built without PTM_GPU_OPENCL) ===\n\n");
    return 0;
#endif
}

} // namespace

int main() {
    const int    N   = 6;
    const double EPS = 0.96;   // Europa emissivity
    const double S   = 1361.0; // solar constant [W/m^2]

    // (s = Σ_j F_ij, A = Bond albedo, label)
    struct Cfg { double s; double A; const char* tag; bool expect_converge; };
    const std::vector<Cfg> cfgs = {
        {1.5, 0.05, "gray-like_A0.05_s1.5",  true},   // dormant: A*s=0.075 << 1
        {1.5, 0.50, "europa_A0.50_s1.5",     true},   // over-couple: A*s=0.75 < 1
        {2.5, 0.50, "europa_A0.50_s2.5",     false},  // DIVERGE: A*s=1.25 >= 1 (GS)
        {2.5, 0.90, "frost_A0.90_s2.5",      false},  // DIVERGE hard: A*s=2.25 (GS)
    };

    std::vector<double> areas(N, 1.0), emiss(N, EPS), T(N, 100.0);
    // Face 0 = shadowed floor (no direct sun); faces 1..N-1 = sunlit walls.
    std::vector<double> solar(N, S);
    solar[0] = 0.0;

    const std::string csv = "output/two_band_prod/a0_gs_vs_jacobi.csv";
    std::ofstream out(csv);
    out << "tag,s_rowsum,A,eps,As_gs_gain,"
           "gs_converged,gs_iters,gs_qvis_floor,gs_qvismax,"
           "jac_converged,jac_iters,jac_qvis_floor,jac_qvismax,"
           "rel_diff_qvis,verdict\n";

    std::printf("\n=== A0: two-band visible-band GS-vs-Jacobi on Σ_j F_ij > 1 (PTM-08) ===\n");
    std::printf("  N=%d  eps=%.3f  S=%.1f  (face 0 = shadowed floor, Q_solar=0)\n\n", N, EPS, S);

    int n_fail = 0;
    // BUGS.md PTM-102: jac_ok was computed, printed as "JACOBI FAILED (unexpected)"
    // and then dropped on the floor -- the exit code was n_fail + a2 + b1 + b1b and
    // never saw it, so a run in which the protected Jacobi reference failed on all
    // 4 configs still exited 0. Count it separately (so the GS!=JAC summary keeps
    // meaning what it says) and add it to the exit code.
    int n_jac_fail = 0;
    for (const auto& c : cfgs) {
        // Two-band requires PTM_SOLAR_ALBEDO (Bond albedo) + PTM_RAD_TWO_BAND, read
        // once in initCommon at construction. Set per-config before constructing.
        char abuf[32];
        std::snprintf(abuf, sizeof(abuf), "%.10g", c.A);
        setenv("PTM_SOLAR_ALBEDO", abuf, 1);
        setenv("PTM_RAD_TWO_BAND", "1", 1);

        ViewFactorMatrix F = buildUniformEnclosure(N, c.s);

        std::printf("[%s]  s=%.3f  A=%.3f   GS gain A*s=%.3f %s\n",
                    c.tag, c.s, c.A, c.A * c.s,
                    (c.A * c.s >= 1.0 ? "(>=1 -> GS must DIVERGE)" : "(<1)"));

        Result gs  = runOne(F, areas, emiss, solar, T, /*gauss_seidel=*/true);
        Result jac = runOne(F, areas, emiss, solar, T, /*gauss_seidel=*/false);

        double rel = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(gs.q_vis_floor) && std::isfinite(jac.q_vis_floor) &&
            std::abs(jac.q_vis_floor) > 1e-30) {
            rel = std::abs(gs.q_vis_floor - jac.q_vis_floor) / std::abs(jac.q_vis_floor);
        }

        // Jacobi is the protected reference; it must always converge.
        const bool jac_ok = jac.converged;
        // Pass iff GS converged AND matches Jacobi (this is the post-A1 contract).
        const bool agree = gs.converged && std::isfinite(rel) && rel < 1e-6;
        const char* verdict = agree ? "GS==JAC(fixed)"
                              : (!gs.converged ? "GS_DIVERGED(bug)" : "GS_OVERCOUPLED(bug)");
        if (!agree) ++n_fail;
        if (!jac_ok) ++n_jac_fail;

        std::printf("   GS : conv=%d iters=%d  q_vis_floor=%.6g  q_vis_max=%.6g\n",
                    gs.converged, gs.iters, gs.q_vis_floor, gs.b_vis_max);
        std::printf("   JAC: conv=%d iters=%d  q_vis_floor=%.6g  q_vis_max=%.6g%s\n",
                    jac.converged, jac.iters, jac.q_vis_floor, jac.b_vis_max,
                    jac_ok ? "" : "  <-- JACOBI FAILED (unexpected)");
        std::printf("   rel_diff(q_vis_floor) = %.3e   verdict=%s\n\n", rel, verdict);

        out << c.tag << "," << c.s << "," << c.A << "," << EPS << "," << (c.A * c.s) << ","
            << gs.converged << "," << gs.iters << "," << gs.q_vis_floor << "," << gs.b_vis_max << ","
            << jac.converged << "," << jac.iters << "," << jac.q_vis_floor << "," << jac.b_vis_max << ","
            << rel << "," << verdict << "\n";
    }
    out.close();

    std::printf("=== SUMMARY: %d/%zu configs where GS != Jacobi (0 == PTM-08 fixed) ===\n",
                n_fail, cfgs.size());
    std::printf("    JACOBI reference failures: %d/%zu (must be 0 -- Jacobi is the "
                "protected reference)\n", n_jac_fail, cfgs.size());
    std::printf("    artifact: %s\n\n", csv.c_str());

    int a2_fail = runA2();
    int b1_fail = runB1();
    int b1c_fail = runB1c();
    int b1d_fail = runB1d();
    int b1b_fail = runB1b();

    std::printf("=== OVERALL: A0/A1 fails=%d  JACOBI fails=%d  A2 fails=%d  B1 fails=%d  "
                "B1c fails=%d  B1d fails=%d  B1b fails=%d ===\n",
                n_fail, n_jac_fail, a2_fail, b1_fail, b1c_fail, b1d_fail, b1b_fail);
    return n_fail + n_jac_fail + a2_fail + b1_fail + b1c_fail + b1d_fail + b1b_fail;
}
