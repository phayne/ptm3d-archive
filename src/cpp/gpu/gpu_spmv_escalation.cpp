// Stage 1 (GPU-radiosity project): stepwise nnz escalation of the single-GPU
// radiosity SpMV/Jacobi, to RETIRE the 100M-nnz folklore with data. Builds a
// synthetic SigmaF<=1 CSR at a target scale, runs RadiosityOpenCLSolver::solve()
// (the production GPU Jacobi), and reports STABILITY (completes, no crash/wedge),
// finiteness, convergence, timing, and -- at small scale -- a CPU cross-check.
//
// Wedge discipline: run under the cputime stall guard; escalate one rung per
// invocation so a crash/wedge at a big rung never loses the smaller-rung data.
//
// Usage: gpu_spmv_escalation <n_faces> <nnz_per_row> [n_solves=3] [seed=1] [cpu_check=auto]
#ifdef PTM_GPU_OPENCL
#include "radiosity_opencl.h"
#include "opencl_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <stdexcept>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

using gpu::opencl::RadiosityOpenCLSolver;
using gpu::opencl::RadiosityOpenCLConfig;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: gpu_spmv_escalation <n_faces> <nnz_per_row> [n_solves=3] [seed=1]\n"); return 2; }
    const size_t n   = std::strtoull(argv[1], nullptr, 10);
    const int    npr = std::atoi(argv[2]);
    const int    n_solves = (argc > 3) ? std::atoi(argv[3]) : 3;
    const unsigned seed = (argc > 4) ? static_cast<unsigned>(std::atoi(argv[4])) : 1u;
    const bool cpu_check = (n <= 200000);  // CPU mirror only at small scale

    const long long nnz = static_cast<long long>(n) * npr;
    std::printf("=== RUNG n_faces=%zu nnz_per_row=%d nnz=%lld (%.2fB)  CSR~%.1f GB ===\n",
                n, npr, nnz, nnz / 1e9, (nnz * 8.0 + (n + 1) * 4.0) / 1e9);
    std::fflush(stdout);

    // --- synthetic SigmaF<=1 CSR (double values, int cols/rowptrs) ---
    std::printf("building synthetic CSR ...\n"); std::fflush(stdout);
    double t0 = now_ms();
    std::vector<int64_t> rowptr(n + 1); std::vector<int> cols; std::vector<double> vals;
    cols.reserve(nnz); vals.reserve(nnz);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long long> col_d(0, static_cast<long long>(n) - 1);
    std::uniform_real_distribution<double> v_d(0.0, 1.0);
    rowptr[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        double rowsum = 0.0; size_t start = cols.size();
        for (int k = 0; k < npr; ++k) { double v = v_d(rng); vals.push_back(v); cols.push_back(static_cast<int>(col_d(rng))); rowsum += v; }
        // normalize so Sum_j F_ij = 0.5 (<=1, matches the capped VF regime)
        const double scale = (rowsum > 0) ? 0.5 / rowsum : 0.0;
        for (size_t k = start; k < cols.size(); ++k) vals[k] *= scale;
        rowptr[i + 1] = static_cast<int64_t>(cols.size());
    }
    std::printf("  CSR built (%.0f ms). host mem ~%.1f GB\n", now_ms() - t0,
                (vals.size() * 8.0 + cols.size() * 4.0 + rowptr.size() * 4.0) / 1e9);
    std::fflush(stdout);

    std::vector<float> emiss(n, 0.95f), refl(n, 0.05f);
    std::vector<float> temps(n), solar(n);
    { std::mt19937 r2(seed + 7); std::uniform_real_distribution<float> Td(40.f, 380.f), Sd(0.f, 1361.f);
      for (size_t i = 0; i < n; ++i) { temps[i] = Td(r2); solar[i] = Sd(r2); } }

    // --- GPU solve ---
    RadiosityOpenCLConfig cfg; cfg.verbose = false; cfg.warm_start = false;
    RadiosityOpenCLSolver solver(n, cfg);
    if (!solver.initialize()) { std::printf("RESULT rung=%zu STATUS=INIT_FAIL\n", n); return 1; }
    std::printf("  GPU: %s\n", solver.getDeviceInfo().name.c_str()); std::fflush(stdout);
    std::vector<float> gpu_out(n, 0.f);
    double best = 1e300, tsum = 0;
    try {
        t0 = now_ms();
        solver.uploadViewFactors(vals, cols, rowptr);
        solver.uploadFaceProperties(emiss, refl);
        std::printf("  upload done (%.0f ms)\n", now_ms() - t0); std::fflush(stdout);

        // SpMV micro-benchmark: isolate kernel time (no Jacobi, no transfers).
        // Single-buffer only. GPU float(values+cols)=8 B/nnz; CPU double+int=12.
        if (std::getenv("PTM_ESC_BENCH")) {
            const int reps = 20;
            const double gpu_bytes = (double)nnz * 8.0;      // float vals + int cols
            const double cpu_bytes = (double)nnz * 12.0;     // double vals + int cols
            double ms_s = solver.benchmarkSpMV(reps, 0) / reps;
            double ms_v = solver.benchmarkSpMV(reps, 1) / reps;
            std::printf("  BENCH GPU scalar SpMV: %.2f ms  %.1f GB/s\n", ms_s, gpu_bytes/(ms_s/1e3)/1e9);
            std::printf("  BENCH GPU vector SpMV: %.2f ms  %.1f GB/s  (speedup %.2fx)\n",
                        ms_v, gpu_bytes/(ms_v/1e3)/1e9, ms_s/ms_v);
            // CPU matvec, OpenMP, same CSR (double values). x = temps (arbitrary).
            std::vector<double> y(n);
            double c0 = now_ms();
            for (int r = 0; r < reps; ++r) {
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < (long long)n; ++i) {
                    double s = 0.0;
                    for (long long k = rowptr[i]; k < rowptr[i+1]; ++k) s += vals[k] * temps[cols[k]];
                    y[i] = s;
                }
            }
            double ms_c = (now_ms() - c0) / reps;
            std::printf("  BENCH CPU matvec (OMP): %.2f ms  %.1f GB/s  (GPU-vector vs CPU: %.2fx)\n",
                        ms_c, cpu_bytes/(ms_c/1e3)/1e9, ms_c/ms_v);
            std::fflush(stdout);
        }

        for (int s = 0; s < n_solves; ++s) {
            double ts = now_ms();
            solver.solve(temps, solar, gpu_out);
            solver.synchronize();
            double dt = now_ms() - ts; tsum += dt; if (dt < best) best = dt;
            std::printf("  solve %d: %.0f ms  iters=%d resid=%.2e\n", s, dt, solver.getLastIterations(), solver.getLastResidual());
            std::fflush(stdout);
        }
    } catch (const std::exception& ex) {
        std::printf("RESULT rung n_faces=%zu nnz=%lld STATUS=CL_ERROR msg=\"%s\"\n", n, nnz, ex.what());
        std::fflush(stdout);
        return 1;
    }
    // finiteness / physical sanity
    size_t nbad = 0; float omin = 1e30f, omax = -1e30f;
    for (size_t i = 0; i < n; ++i) { float v = gpu_out[i]; if (!std::isfinite(v)) ++nbad; if (v < omin) omin = v; if (v > omax) omax = v; }

    // Optional GPU-output dump (bit-identity oracle: diff single vs chunked path).
    if (const char* dump = std::getenv("PTM_ESC_DUMP")) {
        std::FILE* f = std::fopen(dump, "wb");
        if (f) { std::fwrite(gpu_out.data(), sizeof(float), n, f); std::fclose(f);
                 std::printf("  dumped %zu floats to %s\n", n, dump); std::fflush(stdout); }
    }

    // --- small-scale CPU cross-check (mirror: E=eps sig T^4, src=rho Q, B=E+src+rho F B) ---
    double max_absdiff = -1.0;
    if (cpu_check) {
        const double SIG = ptm::kStefanBoltzmann;
        std::vector<double> E(n), src(n), B(n), Bn(n), FB(n);
        for (size_t i = 0; i < n; ++i) { double T = temps[i]; E[i] = 0.95 * SIG * T*T*T*T; src[i] = 0.05 * solar[i]; B[i] = E[i] + src[i]; }
        for (int it = 0; it < 100; ++it) {
            for (size_t i = 0; i < n; ++i) { double s = 0; for (int64_t k = rowptr[i]; k < rowptr[i+1]; ++k) s += vals[k] * B[cols[k]]; FB[i] = s; }
            double mc = 0, mv = 0;
            for (size_t i = 0; i < n; ++i) { Bn[i] = E[i] + src[i] + 0.05 * FB[i]; mc = std::max(mc, std::fabs(Bn[i]-B[i])); mv = std::max(mv, std::fabs(Bn[i])); }
            std::swap(B, Bn);
            if ((mv > 1e-10 ? mc/mv : mc) < 1e-4) break;
        }
        // GPU output convention: absorbed indirect ~ eps * (F B). Compare the shape.
        for (size_t i = 0; i < n; ++i) { double s = 0; for (int64_t k = rowptr[i]; k < rowptr[i+1]; ++k) s += vals[k] * B[cols[k]]; double cpu = 0.95 * s; max_absdiff = std::max(max_absdiff, std::fabs(cpu - (double)gpu_out[i])); }
    }

    std::printf("RESULT rung n_faces=%zu nnz=%lld STATUS=%s converged_iters=%d resid=%.2e "
                "out_range=[%.3g,%.3g] nonfinite=%zu best_ms=%.0f mean_ms=%.0f cpu_maxdiff=%s\n",
                n, nnz, (nbad == 0 ? "STABLE_FINITE" : "NONFINITE"),
                solver.getLastIterations(), solver.getLastResidual(), omin, omax, nbad, best, tsum / n_solves,
                cpu_check ? (std::string(std::to_string(max_absdiff)).c_str()) : "skipped(large)");
    std::fflush(stdout);
    return (nbad == 0) ? 0 : 1;
}
#else
int main() { std::fprintf(stderr, "built without PTM_GPU_OPENCL\n"); return 2; }
#endif
