#include "radiation_exchange.h"
#include "smith_beaming.h"              // Smith (1967) anisotropic emission chi table
#include "mc_view_factor.h"             // Monte Carlo view factor builder
#include "../thermal/thermal_model.h"  // For ProgressBar

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#ifdef PTM_GPU_OPENCL
#include "../gpu/radiosity_opencl.h"
#include "../gpu/radiosity_gpu_pool.h"
#endif

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <atomic>
#include <omp.h>

namespace radiation {

int resolvePoolBatchSize(int configured, int n_steps, int n_devices,
                         size_t n_faces, bool announce) {
    if (n_steps <= 0) return 1;
    const int floor_batch = std::min(n_steps, 4 * std::max(1, n_devices));

    int batch;
    const char* why;
    if (configured > 0) {
        batch = std::min(configured, n_steps);
        why = "radiation.gpu_batch_size";
    } else {
        // Host-memory budget: ~6 vectors of n_faces doubles per timestep live
        // across the batched call (temps, solar, source albedo, indirect, and
        // the two per-band batches).
        constexpr size_t kVectorsPerStep = 6;
        size_t budget_mb = 4096;
        // Upper bound keeps budget_mb * 1024 * 1024 from wrapping size_t. A wrap
        // would silently collapse `affordable` to a tiny number and drop the
        // batch to the floor -- a bad value must be refused out loud, not
        // turned into a quietly different answer.
        constexpr size_t kMaxBudgetMB = size_t{1} << 30;   // 1 PiB
        if (const char* b = std::getenv("PTM_RAD_POOL_BATCH_BUDGET_MB")) {
            const long long v = std::atoll(b);
            if (v > 0 && static_cast<unsigned long long>(v) <= kMaxBudgetMB) {
                budget_mb = static_cast<size_t>(v);
            } else {
                std::cerr << "  [Multi-GPU radiosity] WARNING: ignoring "
                             "PTM_RAD_POOL_BATCH_BUDGET_MB='" << b
                          << "' (must be a positive integer <= " << kMaxBudgetMB
                          << "); using " << budget_mb << " MB.\n";
            }
        }
        const size_t bytes_per_step =
            std::max<size_t>(1, kVectorsPerStep * n_faces * sizeof(double));
        const size_t affordable = (budget_mb * 1024ull * 1024ull) / bytes_per_step;
        batch = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(n_steps), std::max<size_t>(1, affordable)));
        why = (batch >= n_steps) ? "auto = whole sweep in one dispatch"
                                 : "auto, capped by the host batch-memory budget";
    }
    // Never worse than the previous default.
    if (batch < floor_batch) batch = floor_batch;

    if (announce) {
        const int n_dispatch = (n_steps + batch - 1) / batch;
        std::cout << "  [Multi-GPU radiosity] dispatching " << batch
                  << " timesteps/batch across " << n_devices << " GPUs ("
                  << why << "; " << n_dispatch << " dispatch"
                  << (n_dispatch == 1 ? "" : "es") << " for " << n_steps
                  << " timesteps)" << std::endl;
    }
    return batch;
}

// ============================================================================
// RadiositySolver Implementation
// ============================================================================

void RadiositySolver::initCommon(const std::vector<double>& emissivity,
                                  const std::vector<double>& areas) {
    emissivity_ = emissivity;
    areas_ = areas;

    // Optional radiation-emissivity override (PTM_RAD_EMISSIVITY) for consistency
    // studies: forces the radiosity emissivity (and hence reflectivity = 1-eps below)
    // to a single value, independent of the body/config-propagation path.
    if (const char* re = std::getenv("PTM_RAD_EMISSIVITY")) {
        double e = std::atof(re);
        if (e > 0.0 && e <= 1.0 && !emissivity_.empty()) {
            std::fill(emissivity_.begin(), emissivity_.end(), e);
            std::cout << "  [PTM_RAD_EMISSIVITY] radiosity emissivity forced to " << e
                      << " (reflectivity 1-eps=" << (1.0 - e) << ")" << std::endl;
            std::cerr << "  [PTM_RAD_EMISSIVITY] WARNING: this moves the RADIOSITY emissivity "
                         "only. The surface boundary condition still emits with the body's "
                         "emissivity and the thermal path still absorbs infrared with it, so "
                         "absorption, emission and scattering no longer agree (Kirchhoff). "
                         "Use thermophysical.emissivity_override to move all three together; "
                         "the engine's [kirchhoff] check will fail this run." << std::endl;
        }
    }

    size_t n = emissivity.size();
    reflectivity_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        reflectivity_[i] = 1.0 - emissivity_[i];
    }

    // Albedo fix (PTM_SOLAR_ALBEDO): the reflected-SOLAR source should scatter the
    // solar Bond albedo (~0.12), not the thermal reflectivity 1-eps (~0.05). When the
    // env is set we fill a separate solar_reflectivity_ used only for the solar source
    // (computeScatteredSolar + the solar_source term in solve); the thermal multiple-
    // scattering rho*F*B keeps 1-eps. Unset => empty => byte-identical legacy behavior.
    if (const char* sa = std::getenv("PTM_SOLAR_ALBEDO")) {
        double a = std::atof(sa);
        if (a > 0.0 && a < 1.0) {
            solar_reflectivity_.assign(n, a);
            std::cout << "  [PTM_SOLAR_ALBEDO] reflected-solar source uses Bond albedo "
                      << a << " (was 1-eps=" << (n ? reflectivity_[0] : 0.0) << ")\n";
        }
    }

    // Band-separated (visible + thermal-IR) radiosity (PTM_RAD_TWO_BAND). See
    // docs/two-band-radiosity-derivation.tex. OFF by default => the legacy single
    // gray band runs, byte-identical to the committed behavior. When ON, the visible
    // channel scatters/absorbs with the Bond albedo A (from PTM_SOLAR_ALBEDO above),
    // the IR channel is unchanged, and their two absorbed indirect fluxes are summed.
    // A Bond albedo is REQUIRED: if none was supplied we warn LOUDLY and fall back to
    // gray rather than silently inventing one (there is no universal default albedo).
    if (const char* tb = std::getenv("PTM_RAD_TWO_BAND")) {
        if (tb[0] && tb[0] != '0') {
            if (solar_reflectivity_.empty()) {
                std::cerr << "  [PTM_RAD_TWO_BAND] WARNING: two-band radiosity requested but no "
                             "Bond albedo supplied (PTM_SOLAR_ALBEDO unset); FALLING BACK to the "
                             "gray path. Set PTM_SOLAR_ALBEDO=A to enable band separation.\n";
            } else {
                two_band_ = true;
                std::cout << "  [PTM_RAD_TWO_BAND] band-separated radiosity ACTIVE: visible band "
                             "reflects Bond A=" << solar_reflectivity_[0] << " / absorbs 1-A="
                          << (1.0 - solar_reflectivity_[0]) << ";  IR band reflects 1-eps="
                          << reflectivity_[0] << " / absorbs eps=" << emissivity_[0]
                          << ".  Q_ind = Q_ind^vis + Q_ind^ir.\n";
            }
        }
    }

    // Pre-allocate work arrays
    radiosity_.resize(n, 0.0);
    radiosity_new_.resize(n, 0.0);
    temp_vec_.resize(n, 0.0);

    // PTM-12 per-partner corrected VF (full replacement). If PTM_HMATRIX_VF_REPLACE points
    // at a symmetric CSR built from the occlusion-exact referee (assemble_vf2.py), REPLACE the
    // stored H-matrix operator with it: fixes the near-field over-count AND the floor
    // under-coupling in one reciprocal, ΣF<=1 matrix. Format: int32 n, float64 geomhash,
    // int64 indptr[n+1], int64 nnz, int32 cols[nnz], float data[nnz]. FAIL-LOUD (abort) on any
    // invariant violation (mesh-identity mismatch, rows>1, negatives) — a corrupt/mismatched
    // correction must NEVER run silently (mirrors the Sinkhorn gate). Unset => byte-identical.
    if (const char* vfr = std::getenv("PTM_HMATRIX_VF_REPLACE")) {
        // Mesh-identity hash from the face areas (matches assemble_vf2.py GEOMHASH).
        double meshHash = 0.0;
        for (size_t i = 0; i < n; ++i) meshHash += areas_[i] + static_cast<double>(i + 1) * areas_[i] * 1e-9;
        std::ifstream vf(vfr, std::ios::binary);
        int32_t nn = 0; double geomhash = 0.0;
        if (vf) { vf.read(reinterpret_cast<char*>(&nn), sizeof(nn));
                  vf.read(reinterpret_cast<char*>(&geomhash), sizeof(geomhash)); }
        if (!vf) {
            std::cerr << "  [PTM_HMATRIX_VF_REPLACE] FATAL: cannot read CSR header from '" << vfr
                      << "'. Refusing to run uncorrected when a correction was requested.\n";
            std::exit(3);
        }
        if (static_cast<size_t>(nn) != n) {
            std::cerr << "  [PTM_HMATRIX_VF_REPLACE] FATAL: CSR face count " << nn << " != mesh "
                      << n << " (wrong mesh). Aborting.\n";
            std::exit(3);
        }
        if (std::fabs(geomhash - meshHash) > 1e-3 * std::fabs(meshHash) + 1e-9) {
            std::cerr << "  [PTM_HMATRIX_VF_REPLACE] FATAL: geometry-hash mismatch (CSR " << geomhash
                      << " vs mesh " << meshHash << "). The corrected VF was built for a DIFFERENT mesh; "
                      << "NEVER apply a foreign VF. Aborting.\n";
            std::exit(3);
        }
        std::vector<int64_t> indptr(static_cast<size_t>(nn) + 1);
        vf.read(reinterpret_cast<char*>(indptr.data()),
                static_cast<std::streamsize>(indptr.size()) * sizeof(int64_t));
        int64_t nnz = 0;
        vf.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
        std::vector<int32_t> cols(static_cast<size_t>(nnz));
        vf.read(reinterpret_cast<char*>(cols.data()), static_cast<std::streamsize>(nnz) * sizeof(int32_t));
        std::vector<float> data(static_cast<size_t>(nnz));
        vf.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nnz) * sizeof(float));
        if (!vf) {
            std::cerr << "  [PTM_HMATRIX_VF_REPLACE] FATAL: CSR read truncated. Aborting.\n";
            std::exit(3);
        }
        double maxrs = 0.0; size_t nover = 0; double neg = 0.0;
        for (int32_t i = 0; i < nn; ++i) {
            double s = 0.0;
            for (int64_t k = indptr[i]; k < indptr[i + 1]; ++k) { s += data[k]; if (data[k] < neg) neg = data[k]; }
            if (s > 1.0 + 1e-6) ++nover;
            maxrs = std::max(maxrs, s);
        }
        if (nover > 0 || neg < -1e-9) {
            std::cerr << "  [PTM_HMATRIX_VF_REPLACE] FATAL: corrected VF violates invariants (rows>1="
                      << nover << ", min entry=" << neg << "). A non-physical VF (ΣF>1 or negative F) "
                      << "must not run — re-assemble. Aborting.\n";
            std::exit(3);
        }
        // Route the corrected CSR: default = CPU SparseVFOperator (unchanged);
        // PTM_HMATRIX_VF_REPLACE_GPU = build a ViewFactorMatrix and set F_ptr_ so
        // initGPU + the multi-GPU pool + the chunked-vector kernel run it. The
        // invariant checks above (mesh-hash, ΣF<=1, non-negativity) are identical
        // for both routes. Gate unset => this branch is not taken => byte-identical.
        if (std::getenv("PTM_HMATRIX_VF_REPLACE_GPU")) {
            std::vector<double> data_d(data.begin(), data.end());  // float -> double
            data.clear(); data.shrink_to_fit();
            owned_vf_ = std::make_unique<ViewFactorMatrix>();
            owned_vf_->setFromCSR(static_cast<size_t>(nn),
                                  std::move(indptr), std::move(cols), std::move(data_d));
            F_ptr_ = owned_vf_.get();
            F_op_ = std::make_unique<ViewFactorOperator>(*owned_vf_);  // CPU fallback, same matrix
            use_hmatrix_ = false;  // corrected VF is now a sparse ViewFactorMatrix, not H-matrix
            std::cout << "  [PTM_HMATRIX_VF_REPLACE_GPU] Routed corrected VF to the GPU F_ptr_ path "
                         "(nnz=" << nnz << ", max ΣF=" << maxrs << ", rows>1=0, min entry=" << neg
                      << ", mesh-hash OK); initGPU will use the chunked-vector radiosity pool."
                      << std::endl;
        } else {
            F_op_ = std::make_unique<SparseVFOperator>(static_cast<size_t>(nn),
                        std::move(indptr), std::move(cols), std::move(data));
            // Remember that the corrected matrix went to the CPU-only operator.
            // initGPU() reports it if the config also asked for GPU radiosity —
            // otherwise `radiation.use_gpu: true` reads as a GPU run in the
            // config and runs entirely on the CPU with no signal at all.
            vf_replace_cpu_route_ = true;
            std::cout << "  [PTM_HMATRIX_VF_REPLACE] REPLACED H-matrix with symmetric per-partner referee "
                         "VF from '" << vfr << "': nnz=" << nnz << ", max ΣF=" << maxrs << ", rows>1=0, "
                         "min entry=" << neg << ", mesh-hash OK. Occlusion-exact; fixes over-count + floor "
                         "under-coupling; reciprocity by construction. Routed to the CPU SparseVFOperator "
                         "(set PTM_HMATRIX_VF_REPLACE_GPU to route it to the GPU radiosity solver instead)."
                      << std::endl;
        }
        // If a radiance-prune receipt sidecar sits next to the CSR, surface a
        // one-line provenance summary (eps + max discarded flux). The CSR was
        // pruned at ASSEMBLY time (assemble_vf_haworth.py); this just reports the
        // trade that was accepted. Minimal line-based parse; absent or
        // unparseable => silent (the CSR itself remains authoritative).
        {
            const std::string rcpt = std::string(vfr) + ".prune_receipt.json";
            std::ifstream rf2(rcpt);
            if (rf2) {
                auto num_after_colon = [](const std::string& s) -> double {
                    size_t c = s.find(':');
                    return (c == std::string::npos) ? 0.0 : std::atof(s.c_str() + c + 1);
                };
                double eps = 0.0, dmax = 0.0;
                bool have_eps = false, have_dmax = false, in_disc = false;
                std::string line;
                while (std::getline(rf2, line)) {
                    if (line.find("\"radiance_prune_eps_Wm2\"") != std::string::npos) {
                        eps = num_after_colon(line); have_eps = true;
                    } else if (line.find("\"discarded_contribution_Wm2\"") != std::string::npos) {
                        in_disc = true;  // its "max" is the very next occurrence
                    } else if (in_disc && line.find("\"max\"") != std::string::npos) {
                        dmax = num_after_colon(line); have_dmax = true; in_disc = false;
                    }
                }
                if (have_eps) {
                    std::cout << "  [PTM_HMATRIX_VF_REPLACE] radiance-prune receipt: eps=" << eps
                              << " W/m^2, max discarded contribution=";
                    if (have_dmax) std::cout << dmax << " W/m^2"; else std::cout << "n/a";
                    std::cout << " (from " << rcpt << ")." << std::endl;
                }
            }
        }
    }

    // PTM-12 row-sum correction (occlusion-aware referee). Install a symmetric
    // diagonal-congruence wrapper D F D over F_op_ so the effective row sums match a
    // reference (true occluded) row sum supplied via PTM_HMATRIX_ROWFIX (a raw file of
    // n float64 = referee Sum_j F_ij per face, e.g. from scripts/ptm12_diag/mc_ref_raw).
    //   PTM_HMATRIX_ROWFIX_MODE=damp (default): target_i = min(stored_i, ref_i) => reductions
    //     only (fixes the near-field over-weight + the Sum F > 1 renorm runaway).
    //   =full: target_i = ref_i => signed (also restores mild under-coupling).
    // The wrapper preserves reciprocity and restores F_sky>0; the vf_row_norm_ block
    // below then recomputes from the corrected operator (=> a no-op where target<=1).
    // Unset => wrapper never installed => byte-identical.
    if (const char* rf = std::getenv("PTM_HMATRIX_ROWFIX")) {
        std::ifstream tf(rf, std::ios::binary);
        std::vector<double> ref(n, 0.0);
        if (tf) tf.read(reinterpret_cast<char*>(ref.data()),
                        static_cast<std::streamsize>(n) * sizeof(double));
        if (!tf) {
            std::cerr << "  [PTM_HMATRIX_ROWFIX] WARNING: could not read " << n
                      << " float64 referee row sums from '" << rf
                      << "'; row-fix DISABLED, running UNCORRECTED (not silent).\n";
        } else if (!F_op_) {
            std::cerr << "  [PTM_HMATRIX_ROWFIX] WARNING: no view-factor operator; row-fix skipped.\n";
        } else {
            const char* modeEnv = std::getenv("PTM_HMATRIX_ROWFIX_MODE");
            const bool full = (modeEnv && std::string(modeEnv) == "full");
            const bool damp = !full;
            std::vector<double> ones(n, 1.0), stored(n, 0.0);
            F_op_->matvec(ones, stored);
            // Targets. DAMP: reductions only (target=min(stored,ref)) -> a stable contraction.
            // FULL: also RESTORE under-coupling (target=ref), but cap near-zero-stored rows at
            // RCAP*stored (partial restoration) so d does not blow up as ~sqrt(ref/stored); the
            // over-coupled rows still reduce to ref. target <= ref <= 1 ALWAYS (restoration never
            // exceeds the referee truth).
            const double RCAP  = std::getenv("PTM_HMATRIX_ROWFIX_RCAP")  ? std::atof(std::getenv("PTM_HMATRIX_ROWFIX_RCAP"))  : 4.0;
            const double DMAX  = std::getenv("PTM_HMATRIX_ROWFIX_DMAX")  ? std::atof(std::getenv("PTM_HMATRIX_ROWFIX_DMAX"))  : 4.0;
            const double RELAX = std::getenv("PTM_HMATRIX_ROWFIX_RELAX") ? std::atof(std::getenv("PTM_HMATRIX_ROWFIX_RELAX")) : (full ? 0.25 : 0.5);
            std::vector<double> target(n, 0.0);
            size_t n_capped = 0;
            for (size_t i = 0; i < n; ++i) {
                double t = ref[i];
                if (t < 0.0) t = 0.0;
                if (damp) {
                    target[i] = std::min(stored[i], t);
                } else {
                    double cap = RCAP * stored[i];        // partial-restoration cap for near-zero rows
                    if (t > cap) { t = cap; ++n_capped; }
                    target[i] = t;
                }
            }
            // Symmetric Sinkhorn: rowsum(D F D)_i = d_i (F d)_i -> target_i. Stabilizations (FULL):
            // (a) PTM-14 negative/zero guard (skip update where (F d)_i<=0), (b) under-relaxation,
            // (c) clamp d in [DMIN, DMAX] to bound amplification (DMIN small so the over-count rows
            // that need d~0.15 still reduce). The fail-loud gate below is the arbiter.
            std::vector<double> d(n, 1.0), Fd(n, 0.0);
            const int SINK_ITERS = full ? 300 : 50;
            const double DMIN = 1e-3;
            for (int it = 0; it < SINK_ITERS; ++it) {
                F_op_->matvec(d, Fd);
                for (size_t i = 0; i < n; ++i) {
                    if (stored[i] <= 1e-12 || target[i] <= 0.0) { d[i] = 1.0; continue; }
                    double g = Fd[i];
                    if (g <= 1e-30) continue;             // negative/zero row-sum guard (ACA PTM-14)
                    double rs = d[i] * g;
                    if (rs > 1e-30) d[i] *= std::pow(target[i] / rs, RELAX);
                    if (d[i] > DMAX) d[i] = DMAX; else if (d[i] < DMIN) d[i] = DMIN;
                }
            }
            F_op_->matvec(d, Fd);
            double maxres = 0.0; size_t nfix = 0; bool dfinite = true;
            for (size_t i = 0; i < n; ++i) {
                if (target[i] > 0.0) maxres = std::max(maxres, std::fabs(d[i] * Fd[i] - target[i]));
                if (std::fabs(d[i] - 1.0) > 1e-6) ++nfix;
                if (!std::isfinite(d[i])) dfinite = false;
            }
            // Fail-loud convergence GATE. The FULL-mode congruence can DIVERGE when it
            // amplifies near-zero-stored rows (d_i>1 compounding via d_i d_j) or hits ACA
            // far-field negative entries (PTM-14) -> row sums blow up to ~1e19, mean<0. A
            // diverged D would make the coupled solve produce garbage (observed 2026-07-19:
            // FULL ran ~14h to T=4.8e9 K). NEVER install a diverged correction: abort so the
            // caller must switch to DAMP or fix the inputs. Tolerance explicit + logged.
            const char* tolEnv = std::getenv("PTM_HMATRIX_ROWFIX_TOL");
            const double tol = tolEnv ? std::atof(tolEnv) : 1e-2;
            if (!dfinite || !std::isfinite(maxres) || maxres > tol) {
                std::cerr << "  [PTM_HMATRIX_ROWFIX] FATAL: Sinkhorn did NOT converge for "
                          << (damp ? "DAMP" : "FULL") << " mode (max|rowsum-target|=" << maxres
                          << ", finite-d=" << (dfinite ? "yes" : "NO") << ", tolerance=" << tol
                          << "). The corrected view-factor matrix is INVALID (near-zero-stored-row "
                          << "amplification and/or ACA negative entries diverge the FULL congruence). "
                          << "REFUSING to run on a diverged VF. Use PTM_HMATRIX_ROWFIX_MODE=damp, or "
                          << "override PTM_HMATRIX_ROWFIX_TOL (NOT recommended)." << std::endl;
                std::exit(2);
            }
            F_op_ = std::make_unique<CorrectedRowFixOperator>(std::move(F_op_), std::move(d));
            std::cout << "  [PTM_HMATRIX_ROWFIX] " << (damp ? "DAMP" : "FULL")
                      << " symmetric-congruence row-fix to referee '" << rf << "': "
                      << nfix << "/" << n << " faces rescaled, Sinkhorn max|rowsum-target|="
                      << maxres << " <= tol " << tol
                      << " (converged; reciprocity-preserving; F_sky restored)." << std::endl;
            if (full) {
                std::cout << "  [PTM_HMATRIX_ROWFIX] FULL restoration: " << n_capped << "/" << n
                          << " rows capped at " << RCAP << "*stored (partial restoration; referee "
                          << "exceeds " << RCAP << "*stored on these near-zero-stored rows). d clamp ["
                          << DMIN << "," << DMAX << "], under-relax " << RELAX
                          << ". Corrected rowsum = target <= referee <= 1 everywhere." << std::endl;
            }
        }
    }

    // View-factor row normalization (energy conservation / radiosity stability).
    // Compute the row sums Σ_j F_ij once via a single F·1 matvec. The M1 point
    // estimator overshoots in the near field, so on fine/concave meshes some rows
    // sum to > 1, which makes the radiosity gain > 1 and the coupling diverge.
    // Store 1/max(1, Σ_j F_ij) and apply it after every F·x in the solve.
    {
        std::vector<double> ones(n, 1.0), rowsum(n, 0.0);
        if (F_op_) F_op_->matvec(ones, rowsum);
        vf_row_norm_.assign(n, 1.0);
        size_t n_over = 0;
        double max_rs = 0.0, sum_rs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double rs = rowsum[i];
            if (rs > 1.0) { vf_row_norm_[i] = 1.0 / rs; ++n_over; }
            max_rs = std::max(max_rs, rs);
            sum_rs += rs;
        }
        vf_row_sum_max_ = max_rs;
        vf_row_norm_count_ = n_over;
        std::cout << "[Radiosity] View-factor row sums Σ_j F_ij: max=" << max_rs
                  << ", mean=" << (n ? sum_rs / static_cast<double>(n) : 0.0)
                  << " (" << n_over << "/" << n
                  << " rows > 1 renormalized for energy conservation)" << std::endl;
        // Optional diagnostic: dump per-face VF row sums for completeness checks
        // (e.g. is the floor<->distant-wall coupling captured?). Set the env var
        // PTM_DUMP_VF_ROWSUMS to a path. Format: uint32 n, then n float64.
        if (const char* dp = std::getenv("PTM_DUMP_VF_ROWSUMS")) {
            std::ofstream ro(dp, std::ios::binary);
            if (ro) {
                uint32_t nn = static_cast<uint32_t>(n);
                ro.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
                ro.write(reinterpret_cast<const char*>(rowsum.data()),
                         static_cast<std::streamsize>(n) * sizeof(double));
                std::cout << "[Radiosity] Dumped per-face VF row sums -> " << dp << std::endl;
            }
        }
    }
}

void RadiositySolver::setSolarAlbedo(const std::vector<double>& solar_albedo) {
    // Per-face, time-updatable visible Bond albedo A_i (A2). Sourced from the SAME
    // albedo the surface BC absorbs with (per-face angular/Hapke), so the reflected
    // A_i·Q_solar the enclosure receives matches the (1-A_i)·Q_solar the BC absorbs
    // and energy closes per face. Size must match the face count.
    if (solar_albedo.size() != emissivity_.size()) {
        std::cerr << "  [RadiositySolver::setSolarAlbedo] WARNING: albedo size "
                  << solar_albedo.size() << " != face count " << emissivity_.size()
                  << "; ignored (keeping previous solar albedo)." << std::endl;
        return;
    }
    solar_reflectivity_ = solar_albedo;
    // No cache to invalidate (the visible solve is re-run each call). last_indirect_*
    // are recomputed on the next solve().
}

// --- PTM-106: direct-beam (source) albedo, distinct from the diffuse operator ---

void RadiositySolver::setSolarSourceAlbedo(const std::vector<double>& source_albedo) {
    if (source_albedo.size() != emissivity_.size()) {
        std::cerr << "  [RadiositySolver::setSolarSourceAlbedo] WARNING: source-albedo size "
                  << source_albedo.size() << " != face count " << emissivity_.size()
                  << "; IGNORED (the visible-band source keeps the STATIC base albedo "
                     "-- BUGS.md PTM-106 remains in force for this run)." << std::endl;
        return;
    }
    solar_albedo_source_ = source_albedo;
    solar_albedo_source_batch_.clear();
    checkSourceAlbedoGPURepresentable();
}

void RadiositySolver::setSolarSourceAlbedoBatch(
    const std::vector<std::vector<double>>& source_albedo) {
    for (const auto& v : source_albedo) {
        if (v.size() != emissivity_.size()) {
            std::cerr << "  [RadiositySolver::setSolarSourceAlbedoBatch] WARNING: a per-timestep "
                         "source-albedo vector has size " << v.size() << " != face count "
                      << emissivity_.size() << "; the WHOLE batch override is IGNORED (the "
                         "visible-band source keeps the STATIC base albedo -- PTM-106)."
                      << std::endl;
            return;
        }
    }
    solar_albedo_source_batch_ = source_albedo;
    solar_albedo_source_.clear();
    checkSourceAlbedoGPURepresentable();
}

void RadiositySolver::clearSolarSourceAlbedo() {
    solar_albedo_source_.clear();
    solar_albedo_source_batch_.clear();
    src_albedo_gpu_ok_ = true;
}

// The GPU kernel computes solar_source[i] = reflectivity[i] * solar_flux[i] with
// reflectivity = the OPERATOR albedo A_op. A source albedo A_src != A_op is therefore
// expressed by pre-scaling the uploaded flux by A_src/A_op, which is undefined where
// A_op -> 0. Rather than silently dropping those faces' reflected sunlight (the exact
// archetype PTM-106 is an instance of), refuse the GPU two-band path and let the CPU
// two-band solve -- which multiplies A_src in directly and needs no ratio -- run.
void RadiositySolver::checkSourceAlbedoGPURepresentable() {
    src_albedo_gpu_ok_ = true;
    if (solar_albedo_source_.empty() && solar_albedo_source_batch_.empty()) return;
    constexpr double A_OP_MIN = 1e-6;
    size_t n_bad = 0;
    for (size_t i = 0; i < solar_reflectivity_.size(); ++i) {
        if (solar_reflectivity_[i] < A_OP_MIN) ++n_bad;
    }
    if (n_bad == 0) return;
    src_albedo_gpu_ok_ = false;
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::cerr << "  [radiation.PTM-106] WARNING: " << n_bad
                  << " face(s) carry an operator Bond albedo < " << A_OP_MIN
                  << ", so the incidence-dependent visible SOURCE albedo cannot be "
                     "represented on the GPU (which hard-wires source = A_op * Q). "
                     "Falling back to the exact CPU two-band radiosity for this run "
                     "(correct results, slower)." << std::endl;
    }
}

void RadiositySolver::enableTwoBand() {
    // D1: config-driven activation (no env). Never invents an albedo — the archetype.
    if (solar_reflectivity_.empty()) {
        std::cerr << "  [radiation.band_separated] WARNING: two-band requested via config but no "
                     "per-face Bond albedo was supplied (setSolarAlbedo); staying GRAY.\n";
        return;
    }
    if (!two_band_) {
        two_band_ = true;
        std::cout << "  [radiation.band_separated] band-separated radiosity ACTIVE (config): "
                     "visible band absorbs 1-A_i, IR band absorbs eps_i; Q_ind = vis + ir.\n";
#ifdef PTM_GPU_OPENCL
        // Never substitute CPU for a GPU request silently. If the GPU solver was
        // already initialized (initGPU ran) before two-band was enabled, the GPU
        // two-band gate (two_band_gpu_ok_, set inside initGPU) never engaged, so
        // every solve falls back to the CPU two-band path. Warn once so this perf
        // substitution is visible. To engage the single-GPU two-band mirror,
        // enable two-band BEFORE initGPU (source the per-face albedo before
        // RadiationExchangeManager::prepare()); production radiosity is CPU-only
        // for two-band until that ordering fix / the multi-GPU + H-matrix two-band
        // mirror lands.
        if (use_gpu_ && !two_band_gpu_ok_) {
            std::cerr << "  [radiation.band_separated] WARNING: two-band was enabled AFTER GPU "
                         "initialization; the single-GPU two-band path is NOT engaged -> running "
                         "CPU two-band radiosity (correct results, slower). Enable two-band before "
                         "initGPU to use the GPU two-band mirror.\n";
        }
#endif
    }
}

RadiositySolver::RadiositySolver(const ViewFactorMatrix& F,
                                 const std::vector<double>& emissivity,
                                 const std::vector<double>& areas)
    : F_op_(std::make_unique<ViewFactorOperator>(F)),
      F_ptr_(&F),
      use_hmatrix_(false) {
    initCommon(emissivity, areas);
}

RadiositySolver::RadiositySolver(const HMatrixViewFactor& H,
                                 const std::vector<double>& emissivity,
                                 const std::vector<double>& areas)
    : F_op_(std::make_unique<HMatrixOperator>(H)),
      F_ptr_(nullptr),
      use_hmatrix_(true) {
    initCommon(emissivity, areas);
}

// Default destructor in the .cpp where forward-declared GPU types are
// fully visible.
RadiositySolver::~RadiositySolver() = default;

void RadiositySolver::initGPU() {
#ifdef PTM_GPU_OPENCL
    if (config_.verbose) {
        std::cout << "[RadiositySolver] initGPU called: use_gpu=" << config_.use_gpu
                  << ", use_hmatrix=" << use_hmatrix_
                  << ", F_ptr_=" << (F_ptr_ ? "valid" : "null") << std::endl;
    }

    if (!config_.use_gpu) {
        if (config_.verbose) {
            std::cout << "[RadiositySolver] GPU not requested, using CPU path" << std::endl;
        }
        return;
    }

    // Reachability, stated once and unmissably. PTM_HMATRIX_VF_REPLACE without
    // PTM_HMATRIX_VF_REPLACE_GPU installs the corrected CSR as a CPU-only
    // SparseVFOperator, leaving F_ptr_ null -- so every gate below refuses the
    // GPU and the whole radiosity solve runs on the CPU while the config still
    // reads `radiation.use_gpu: true`. Every production launcher audited on
    // 2026-08-15 set only the former, which is why the flagship 509k radiosity
    // has always been CPU. Name the cause and the fix here, not three fallbacks
    // later where the message blames the H-matrix.
    if (vf_replace_cpu_route_) {
        std::cout << "[RadiositySolver] NOTE: radiation.use_gpu is true, but "
                     "PTM_HMATRIX_VF_REPLACE was set WITHOUT PTM_HMATRIX_VF_REPLACE_GPU, so the "
                     "corrected view-factor matrix was installed as a CPU-only SparseVFOperator "
                     "and the GPU radiosity solver cannot see it. This run's radiosity will be "
                     "CPU. Set PTM_HMATRIX_VF_REPLACE_GPU=1 to run the same corrected matrix on "
                     "the GPU.\n";
    }

    // Two-band radiosity on the GPU (B1). The GPU path mirrors the CPU two-band by
    // running the two verified gray kernels twice per solve (IR: solar=0, normal
    // ε/1-ε; visible: temps=0, properties ε←1-A, ρ←A, source=A·Q) and summing —
    // see solveGPU().
    //
    // The GPU Jacobi kernel now applies vf_row_norm_ INSIDE its iteration (the
    // row_norm buffer uploaded below), exactly as the CPU Jacobi does, so both
    // paths iterate the same operator ρ·(norm⊙F). A Σ_j F_ij > 1 row is therefore
    // no longer a reason to refuse the GPU.
    //
    // What remains is the physics: Jacobi on B = S + ρ⊙(norm⊙F)B converges iff the
    // iteration matrix has spectral radius < 1, and the row-sum bound on that is
    //     gain = ρ_max · max_i (norm_i · Σ_j F_ij) = ρ_max · max_i min(Σ_j F_ij, 1).
    // The row norm caps the second factor at 1, so the gate can only fire for a
    // genuinely pathological enclosure — a perfect reflector (Bond albedo A ≥ 1 or
    // emissivity ε ≤ 0), for which the radiosity series does not converge at all
    // and no tolerance would make it. That is the check below; there is no ε-band
    // on Σ_j F_ij, so it cannot disagree with the CSR loader's 1 + 1e-6 acceptance
    // the way the old any_over test did (it vetoed a 509,337-face mesh over a
    // single row 2.03e-9 above 1 — pure float32 accumulation noise).
    if (two_band_) {
        // Effective row sum after normalization; by construction ≤ 1.
        const double sigma_eff = std::min(vf_row_sum_max_, 1.0);
        double rho_max = 0.0;
        for (double r : reflectivity_) rho_max = std::max(rho_max, r);          // IR: 1-ε
        for (double a : solar_reflectivity_) rho_max = std::max(rho_max, a);    // visible: A
        const double gain = rho_max * sigma_eff;
        if (!std::isfinite(gain) || gain >= 1.0) {
            std::cout << "[RadiositySolver] PTM_RAD_TWO_BAND: Jacobi gain ρ_max·max_i min(Σ_j F_ij,1)"
                      << " = " << rho_max << " × " << sigma_eff << " = " << gain
                      << " ≥ 1 — the radiosity series does not converge on this enclosure "
                         "(perfect reflector: A ≥ 1 or ε ≤ 0). Forcing the CPU two-band solve, "
                         "which carries the same divergence but in float64 with a residual "
                         "report. Fix the optical properties.\n";
            return;
        }
        if (use_hmatrix_ || !F_ptr_) {
            std::cout << "[RadiositySolver] PTM_RAD_TWO_BAND: GPU two-band needs a sparse VF "
                         "(H-matrix here); using the CPU two-band solve.\n";
            return;
        }
        // Fall through to initialization below (multi-GPU pool first, then
        // single-GPU); both rungs run the band-separated orchestration —
        // solveBatch() for the pool, solveGPUTwoBand() for single-GPU.
    }

    // GPU solver works best with sparse view factors, not H-matrix
    if (use_hmatrix_) {
        std::cerr << "Warning: GPU radiosity solver not supported with H-matrix; "
                  << "using CPU path\n";
        return;
    }

    if (!F_ptr_) {
        std::cerr << "Warning: GPU radiosity solver requires sparse view factor matrix\n";
        return;
    }

    size_t n = emissivity_.size();
    size_t nnz = F_ptr_->numNonZeros();

    // The multi-GPU pool handles both gray and two-band (2d): the per-device
    // solvers do the IR + visible solve with per-band capture in solveBatch().
    const bool want_pool = (config_.gpu_devices > 1)
                            || !config_.gpu_device_ids.empty();

    // Device footprint. The single-GPU solver and every pool device each hold a
    // FULL replica of the CSR. The 32-bit addressing wall (PTM-25) is handled by
    // the solver's chunked sub-buffers, so there is NO per-buffer nnz ceiling
    // here (the old 100M-nnz "soft limit" is retired) -- the only question is
    // whether the replica + working buffers fit a device's VRAM.
    const size_t csr_bytes = nnz * sizeof(float)      // values (float on device)
                           + nnz * sizeof(int)        // column indices
                           + (n + 1) * sizeof(int);   // row pointers (int32 on device)
    const size_t working_bytes = 12 * n * sizeof(float);  // ~11 state buffers + margin
    const size_t needed_bytes = csr_bytes + working_bytes;

    // Real VRAM-fit check (replaces the old fixed 50%-of-total folklore, which
    // wrongly rejected multi-GB CSRs that fit comfortably): the replica must fit
    // with a reserve for the driver + kernel scratch (15% of VRAM, min 2 GB).
    auto deviceReserve = [](size_t global_mem) -> size_t {
        return std::max<size_t>((global_mem / 100) * 15,
                                static_cast<size_t>(2) * 1024 * 1024 * 1024);
    };
    auto fitsDevice = [&](size_t global_mem) -> bool {
        return global_mem > deviceReserve(global_mem)
            && needed_bytes <= global_mem - deviceReserve(global_mem);
    };
    auto gb = [](size_t b) -> double { return b / (1024.0 * 1024.0 * 1024.0); };

    gpu::opencl::RadiosityOpenCLConfig gpu_config;
    gpu_config.max_iterations = config_.max_iterations;
    gpu_config.convergence_tolerance = config_.convergence_tolerance;
    gpu_config.convergence_check_interval = config_.convergence_check_interval;
    gpu_config.verbose = config_.verbose;
    // Two-band runs two solves per call with different sources; a cold start each
    // solve keeps the visible band from warm-starting off the IR radiosity and makes
    // the GPU result deterministic + comparable to the always-cold-started CPU path.
    if (two_band_) gpu_config.warm_start = false;

    std::vector<float> emissivity_f(n);
    std::vector<float> reflectivity_f(n);
    for (size_t i = 0; i < n; ++i) {
        emissivity_f[i] = static_cast<float>(emissivity_[i]);
        reflectivity_f[i] = static_cast<float>(reflectivity_[i]);
    }
    // The row normalization travels with the operator, not with the band
    // properties: upload it once per device, alongside the CSR. All-ones (the
    // common case) is the kernel's identity seed, so uploading it is a no-op
    // bitwise — we upload unconditionally anyway so there is no second code
    // path whose behavior depends on the mesh.
    std::vector<float> row_norm_f(n, 1.0f);
    for (size_t i = 0; i < n && i < vf_row_norm_.size(); ++i) {
        row_norm_f[i] = static_cast<float>(vf_row_norm_[i]);
    }

    // Single-GPU rung of the degrade ladder. Returns true if it engaged the GPU.
    // Every failure is loud (never a silent drop to CPU); the CSR upload is
    // wrapped because uploadViewFactors can throw (chunk/VRAM fail-loud, PTM-25).
    auto try_single_gpu = [&]() -> bool {
        gpu_solver_ = std::make_unique<gpu::opencl::RadiosityOpenCLSolver>(n, gpu_config);
        if (!gpu_solver_->initialize()) {
            std::cerr << "[Radiosity] single-GPU solver failed to initialize.\n";
            gpu_solver_.reset();
            return false;
        }
        // Fault injection (test-only, fail-loud): force the single-GPU rung to
        // fail so the degrade to CPU can be exercised deliberately.
        if (std::getenv("PTM_RAD_FORCE_SINGLE_FAIL")) {
            std::cerr << "[Radiosity] PTM_RAD_FORCE_SINGLE_FAIL set: forcing single-GPU "
                         "rung failure (test); degrading to CPU radiosity.\n";
            gpu_solver_.reset();
            return false;
        }
        const size_t dev_mem = gpu_solver_->getDeviceInfo().global_mem_size;
        if (!fitsDevice(dev_mem)) {
            std::cout << "[Radiosity] single-GPU: CSR replica + working "
                      << gb(needed_bytes) << " GB does not fit device ("
                      << gb(dev_mem) << " GB VRAM, reserve "
                      << gb(deviceReserve(dev_mem)) << " GB).\n";
            gpu_solver_.reset();
            return false;
        }
        try {
            gpu_solver_->uploadViewFactors(F_ptr_->values(),
                                           F_ptr_->colIndices(),
                                           F_ptr_->rowPointers());
            gpu_solver_->uploadFaceProperties(emissivity_f, reflectivity_f);
            gpu_solver_->uploadRowNorm(row_norm_f);
        } catch (const std::exception& e) {
            std::cerr << "[Radiosity] single-GPU CSR upload failed: " << e.what() << "\n";
            gpu_solver_.reset();
            return false;
        }
        use_gpu_ = true;
        if (two_band_) {
            two_band_gpu_ok_ = true;
            std::cout << "[RadiositySolver] PTM_RAD_TWO_BAND: single-GPU band-separated "
                         "radiosity ENABLED (IR solar=0 + visible ε←1-A/ρ←A; row norm applied "
                         "in-kernel on " << vf_row_norm_count_ << "/" << emissivity_.size()
                      << " faces).\n";
        }
        std::cout << "[Radiosity] ACTIVE RUNG: single-GPU radiosity ("
                  << gb(needed_bytes) << " GB CSR+working on a " << gb(dev_mem)
                  << " GB device).\n";
        return true;
    };

    // ---- Degrade ladder: multi-GPU pool -> single-GPU -> CPU ----
    // Every downgrade warns (never silent) and the ACTIVE RUNG is always named.

    // Rung 1: multi-GPU pool (only when gpu_devices > 1 / device_ids set, and not
    // two-band -- 2d). On any failure we fall through to the single-GPU rung.
    if (want_pool) {
        // Resolve which physical devices to use (mirrors the thermal pool).
        std::vector<gpu::opencl::OpenCLDeviceInfo> all_gpus =
            gpu::opencl::OpenCLContext::enumerateGPUs(true);
        std::vector<gpu::opencl::OpenCLDeviceInfo> chosen;
        if (!config_.gpu_device_ids.empty()) {
            for (int id : config_.gpu_device_ids) {
                if (id >= 0 && static_cast<size_t>(id) < all_gpus.size()) {
                    chosen.push_back(all_gpus[id]);
                } else {
                    std::cerr << "Warning: rad-gpu-device-id " << id
                              << " out of range (have " << all_gpus.size()
                              << " GPUs); skipping\n";
                }
            }
        } else {
            const int n_req = (config_.gpu_devices == 0)
                ? static_cast<int>(all_gpus.size())
                : std::min<int>(config_.gpu_devices,
                                static_cast<int>(all_gpus.size()));
            for (int i = 0; i < n_req; ++i) chosen.push_back(all_gpus[i]);
        }

        bool pool_ok = chosen.size() >= 2;
        if (!pool_ok) {
            std::cout << "[Radiosity] multi-GPU pool: only " << chosen.size()
                      << " device(s) selected; degrading to single-GPU.\n";
        }
        // Real per-device fit: each device replicates the full CSR.
        size_t min_dev = pool_ok ? chosen.front().global_mem_size : 0;
        if (pool_ok) {
            for (const auto& d : chosen) min_dev = std::min(min_dev, d.global_mem_size);
            if (!fitsDevice(min_dev)) {
                pool_ok = false;
                std::cout << "[Radiosity] multi-GPU pool: CSR replica " << gb(needed_bytes)
                          << " GB does not fit the smallest device (" << gb(min_dev)
                          << " GB VRAM, reserve " << gb(deviceReserve(min_dev))
                          << " GB); degrading to single-GPU.\n";
            }
        }

        // Fault injection (test-only, fail-loud): force the multi-GPU pool rung
        // to fail so the degrade to single-GPU can be exercised deliberately.
        if (pool_ok && std::getenv("PTM_RAD_FORCE_POOL_FAIL")) {
            std::cout << "[Radiosity] PTM_RAD_FORCE_POOL_FAIL set: forcing multi-GPU "
                         "pool failure (test); degrading to single-GPU.\n";
            pool_ok = false;
        }

        if (pool_ok) {
            gpu::opencl::RadiosityGPUPoolConfig pool_cfg;
            pool_cfg.devices = chosen;
            pool_cfg.inter_batch_pause_ms = config_.gpu_inter_batch_pause_ms;
            pool_cfg.verbose = config_.verbose;

            gpu::opencl::RadiosityOpenCLConfig solver_cfg;
            solver_cfg.max_iterations = config_.max_iterations;
            solver_cfg.convergence_tolerance = config_.convergence_tolerance;
            solver_cfg.convergence_check_interval = config_.convergence_check_interval;
            solver_cfg.verbose = config_.verbose;
            // Cross-batch warm-start has no meaning across devices — disable it.
            solver_cfg.warm_start = false;

            gpu_pool_ = std::make_unique<gpu::opencl::RadiosityGPUDevicePool>(
                std::move(pool_cfg));
            bool ok = gpu_pool_->initialize(n, solver_cfg);
            if (ok) {
                try {
                    ok = gpu_pool_->uploadStaticData(F_ptr_->values(),
                                                     F_ptr_->colIndices(),
                                                     F_ptr_->rowPointers(),
                                                     emissivity_f, reflectivity_f,
                                                     row_norm_f);
                } catch (const std::exception& e) {
                    std::cerr << "[Radiosity] multi-GPU pool CSR upload failed: "
                              << e.what() << "\n";
                    ok = false;
                }
            }
            if (!ok) {
                std::cout << "[Radiosity] multi-GPU pool init/upload failed; "
                             "degrading to single-GPU.\n";
                gpu_pool_.reset();
            } else {
                // The pool IS a GPU rung: publish it in the same state the
                // single-GPU rung publishes. Without this use_gpu_ and
                // two_band_gpu_ok_ stayed false while the pool ran, so
                // isGPUTwoBand() reported "CPU" on an active GPU pool and the
                // per-timestep solve() below fell to the CPU solver -- i.e.
                // setting radiation.gpu_devices: 2 silently changed the solver
                // path for every non-batch caller. gpu_solver_ stays null; the
                // dispatch checks distinguish the two rungs.
                use_gpu_ = true;
                if (two_band_) two_band_gpu_ok_ = true;
                // Advisory only: the dispatching caller resolves the real batch
                // size per sweep with resolvePoolBatchSize(), which needs the
                // timestep count this function does not have. Keep this as the
                // configured value so the two cannot silently disagree -- the
                // duplicated `4 * n_devices` formula that used to live here was
                // a second source of truth for the same number.
                pool_batch_size_ = config_.gpu_batch_size;
                std::cout << "[Radiosity] ACTIVE RUNG: multi-GPU pool, "
                          << gpu_pool_->numDevices() << " device(s); CSR replica "
                          << gb(needed_bytes) << " GB/device (smallest device "
                          << gb(min_dev) << " GB VRAM); gpu_batch_size="
                          << (pool_batch_size_ > 0 ? std::to_string(pool_batch_size_)
                                                   : std::string("auto"))
                          << " (resolved per sweep); row norm applied in-kernel on "
                          << vf_row_norm_count_ << "/" << n << " faces.\n";
                return;  // pool active -- done
            }
        }
        // Fall through: any pool failure degrades to the single-GPU rung.
    }

    // Rung 2: single-GPU. Rung 3 (CPU) is implicit when this returns false.
    if (!try_single_gpu()) {
        std::cout << "[Radiosity] ACTIVE RUNG: CPU radiosity (no GPU rung available; "
                     "normal for very large meshes or headless devices).\n";
    }
#endif
}

bool RadiositySolver::hasGPUPool() const {
#ifdef PTM_GPU_OPENCL
    return gpu_pool_ != nullptr;
#else
    return false;
#endif
}

#ifdef PTM_GPU_OPENCL
std::vector<double> RadiositySolver::solveGPUTwoBand(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux)
{
    // B1: band-separated radiosity on the GPU by reusing the verified gray kernels.
    //   IR band      : normal properties (ε, 1-ε), solar = 0   -> Q_ind^ir = ε·(F·B_ir)
    //   Visible band : properties (1-A, A), temps = 0, solar=Q -> Q_ind^vis = (1-A)·(F·B_vis)
    // The row-norm is applied to the returned flux below (== 1 under the two-band GPU
    // gate). The two solves cold-start (warm_start off), matching the CPU path.
    const size_t n = temperatures.size();
    std::vector<float> emiss_f(n), refl_f(n), visE_f(n), visR_f(n);
    std::vector<float> temps_f(n), flux_f(n), zeros_f(n, 0.0f);
    // PTM-106: the kernel forms solar_source = visR * flux, so a direct-beam source
    // albedo A_src != the operator A is applied by uploading the EFFECTIVE flux
    // Q_i * A_src_i / A_i. Reduces to Q_i exactly when no source albedo is installed
    // (visSourceAlbedo returns solar_reflectivity_ -> ratio == 1 bitwise).
    const std::vector<double>& src_alb = visSourceAlbedoSingle();
    for (size_t i = 0; i < n; ++i) {
        emiss_f[i] = static_cast<float>(emissivity_[i]);
        refl_f[i]  = static_cast<float>(reflectivity_[i]);
        const double A = (i < solar_reflectivity_.size()) ? solar_reflectivity_[i] : 0.0;
        visR_f[i]  = static_cast<float>(A);
        visE_f[i]  = static_cast<float>(1.0 - A);
        temps_f[i] = static_cast<float>(temperatures[i]);
        const double As = (i < src_alb.size()) ? src_alb[i] : A;
        flux_f[i]  = static_cast<float>((As == A) ? solar_flux[i]
                                                  : solar_flux[i] * (As / A));
    }

    // IR band (normal properties already uploaded), no solar source.
    std::vector<float> ir_f;
    gpu_solver_->solve(temps_f, zeros_f, ir_f);

    // Visible band: swap to (1-A, A), no thermal emission (temps = 0), solar = Q.
    gpu_solver_->uploadFaceProperties(visE_f, visR_f);
    std::vector<float> vis_f;
    gpu_solver_->solve(zeros_f, flux_f, vis_f);
    // Restore the thermal/IR properties for the next call.
    gpu_solver_->uploadFaceProperties(emiss_f, refl_f);

    last_iterations_ = gpu_solver_->getLastIterations();
    last_residual_   = gpu_solver_->getLastResidual();
    converged_       = gpu_solver_->didConverge();

    // vf_row_norm_ is NOT re-applied here: indirect_flux_kernel already folded it
    // into ir_f/vis_f on the device (uploadRowNorm). Multiplying again would square
    // the correction on any Σ_j F_ij > 1 face.
    last_indirect_ir_.assign(n, 0.0);
    last_indirect_vis_.assign(n, 0.0);
    std::vector<double> indirect_flux(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        last_indirect_ir_[i]  = static_cast<double>(ir_f[i]);
        last_indirect_vis_[i] = static_cast<double>(vis_f[i]);
        indirect_flux[i] = last_indirect_ir_[i] + last_indirect_vis_[i];
    }
    return indirect_flux;
}

std::vector<double> RadiositySolver::solveGPU(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux)
{
    // Band-separated GPU solve when enabled (B1); otherwise the gray GPU path below.
    // src_albedo_gpu_ok_ (PTM-106) can veto it when A_src/A_op is not representable.
    if (two_band_gpu_ok_ && src_albedo_gpu_ok_) {
        return solveGPUTwoBand(temperatures, solar_flux);
    }

    size_t n = temperatures.size();

    // Convert to float
    std::vector<float> temps_f(n);
    std::vector<float> flux_f(n);
    for (size_t i = 0; i < n; ++i) {
        temps_f[i] = static_cast<float>(temperatures[i]);
        flux_f[i] = static_cast<float>(solar_flux[i]);
    }

    // Solve on GPU
    std::vector<float> indirect_f;
    gpu_solver_->solve(temps_f, flux_f, indirect_f);

    // Get convergence info
    last_iterations_ = gpu_solver_->getLastIterations();
    last_residual_ = gpu_solver_->getLastResidual();
    converged_ = gpu_solver_->didConverge();

    // Convert back to double. The view-factor row normalization (Σ_j F_ij ≤ 1) was
    // applied on the device — inside the Jacobi iteration AND in
    // indirect_flux_kernel — so it must NOT be applied a second time here.
    std::vector<double> indirect_flux(n);
    for (size_t i = 0; i < n; ++i) {
        indirect_flux[i] = static_cast<double>(indirect_f[i]);
    }

    return indirect_flux;
}
#endif

std::vector<std::vector<double>> RadiositySolver::solveBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep)
{
    const size_t n_steps = temperatures_per_timestep.size();
    std::vector<std::vector<double>> indirect_per_timestep(n_steps);

    // PTM-106: a per-timestep source albedo must be paired with the RIGHT timestep.
    // If the batch override is present but mis-sized, refuse the pool (the serial
    // fallback below re-installs the correct per-step vector on each solve()).
    const bool src_batch_ok =
        solar_albedo_source_batch_.empty()
        || solar_albedo_source_batch_.size() == n_steps;
    if (!src_batch_ok) {
        std::cerr << "  [radiation.PTM-106] WARNING: per-timestep source-albedo batch has "
                  << solar_albedo_source_batch_.size() << " step(s) but the radiosity batch has "
                  << n_steps << "; running the serial per-timestep path instead of the pool "
                     "(correct results, slower) rather than pairing the wrong albedo."
                  << std::endl;
    }

    // Single-device or CPU fallback: serial loop over solve(). Semantically
    // identical to the caller running its own per-timestep loop.
#ifdef PTM_GPU_OPENCL
    // A PTM-106 veto sends a POOL run down the serial path; that run still expects
    // per-band batch output, so populate it here (legacy no-pool callers keep the
    // exact old behavior -- band batches untouched).
    const bool ptm106_serial_veto =
        (gpu_pool_ != nullptr) && (!src_albedo_gpu_ok_ || !src_batch_ok);
    if (!gpu_pool_ || ptm106_serial_veto) {
#else
    const bool ptm106_serial_veto = false;
#endif
        // Per-timestep source albedo (if any) must follow the timestep into solve(),
        // which reads the single-vector slot. Save/restore so the caller's batch
        // override survives the loop.
        std::vector<std::vector<double>> saved_batch;
        saved_batch.swap(solar_albedo_source_batch_);
        const std::vector<double> saved_single = solar_albedo_source_;
        // Re-entrancy guard: solve() may route a single timestep back through
        // the pool, and this loop calls solve(). The flag makes that impossible
        // to turn into recursion regardless of how the veto conditions change.
        BatchSerialGuard serial_guard(in_batch_serial_);
        if (ptm106_serial_veto && two_band_) {
            last_indirect_ir_batch_.assign(n_steps, {});
            last_indirect_vis_batch_.assign(n_steps, {});
        }
        for (size_t k = 0; k < n_steps; ++k) {
            if (!saved_batch.empty() && k < saved_batch.size()) {
                solar_albedo_source_ = saved_batch[k];
            }
            indirect_per_timestep[k] =
                solve(temperatures_per_timestep[k], solar_flux_per_timestep[k]);
            if (ptm106_serial_veto && two_band_) {
                last_indirect_ir_batch_[k]  = last_indirect_ir_;
                last_indirect_vis_batch_[k] = last_indirect_vis_;
            }
        }
        solar_albedo_source_ = saved_single;
        solar_albedo_source_batch_.swap(saved_batch);
        return indirect_per_timestep;
#ifdef PTM_GPU_OPENCL
    }

    // Multi-GPU dispatch: split the n_steps timesteps round-robin across
    // pool devices. Timesteps are striped WHOLE (never split), so each device
    // solves an independent set of timesteps against its full CSR replica -->
    // the per-timestep result is identical to the single-GPU path.
    const size_t n_dev = gpu_pool_->numDevices();
    const size_t n = emissivity_.size();
    const bool tb = two_band_;

    // Two-band per-band batch storage (pool path). Pre-size to n_steps so the
    // device workers write disjoint timestep slots without racing.
    if (tb) {
        last_indirect_ir_batch_.assign(n_steps, std::vector<double>(n, 0.0));
        last_indirect_vis_batch_.assign(n_steps, std::vector<double>(n, 0.0));
    } else {
        last_indirect_ir_batch_.clear();
        last_indirect_vis_batch_.clear();
    }

    // Two property sets, built once (read-only, shared across workers): IR uses
    // (eps, 1-eps); visible uses (1-A, A) from the per-face Bond albedo. Mirrors
    // solveGPUTwoBand exactly so the pool per-band == single-GPU per-band.
    std::vector<float> emiss_f, refl_f, visE_f, visR_f;
    if (tb) {
        emiss_f.resize(n); refl_f.resize(n); visE_f.resize(n); visR_f.resize(n);
        for (size_t i = 0; i < n; ++i) {
            emiss_f[i] = static_cast<float>(emissivity_[i]);
            refl_f[i]  = static_cast<float>(reflectivity_[i]);
            const double A = (i < solar_reflectivity_.size()) ? solar_reflectivity_[i] : 0.0;
            visR_f[i] = static_cast<float>(A);
            visE_f[i] = static_cast<float>(1.0 - A);
        }
    }

    std::vector<std::future<bool>> futures;
    futures.reserve(n_dev);

    // Round-robin assignment: timestep k goes to device k % n_dev. Keeps
    // load balanced when n_steps is not a multiple of n_dev.
    for (size_t d = 0; d < n_dev; ++d) {
        // Build the per-device list of timestep indices to process.
        std::vector<size_t> device_steps;
        for (size_t k = d; k < n_steps; k += n_dev) device_steps.push_back(k);
        if (device_steps.empty()) continue;

        if (tb) {
            futures.push_back(gpu_pool_->submit(
                [&temperatures_per_timestep, &solar_flux_per_timestep,
                 &indirect_per_timestep, device_steps, n,
                 &emiss_f, &refl_f, &visE_f, &visR_f,
                 &op_alb = solar_reflectivity_,
                 &src_single = solar_albedo_source_,
                 &src_batch = solar_albedo_source_batch_,
                 &ir_batch = last_indirect_ir_batch_,
                 &vis_batch = last_indirect_vis_batch_](
                    gpu::opencl::RadiosityOpenCLSolver& solver, size_t /*idx*/) -> bool {
                    std::vector<float> temps_f(n), solar_f(n), zeros_f(n, 0.0f), ir_f, vis_f;
                    for (size_t k : device_steps) {
                        const auto& temps_d = temperatures_per_timestep[k];
                        const auto& solar_d = solar_flux_per_timestep[k];
                        // PTM-106: per-timestep direct-beam source albedo, applied by
                        // pre-scaling the uploaded flux (the kernel forms
                        // source = A_op * flux). No override -> ratio 1, bitwise legacy.
                        const std::vector<double>* src_alb =
                            !src_batch.empty() ? &src_batch[k]
                                               : (!src_single.empty() ? &src_single : nullptr);
                        for (size_t i = 0; i < n; ++i) {
                            temps_f[i] = static_cast<float>(temps_d[i]);
                            double q = solar_d[i];
                            if (src_alb) {
                                const double A  = (i < op_alb.size()) ? op_alb[i] : 0.0;
                                const double As = (i < src_alb->size()) ? (*src_alb)[i] : A;
                                if (As != A) q *= As / A;
                            }
                            solar_f[i] = static_cast<float>(q);
                        }
                        // IR band (device holds IR props), then visible, then
                        // restore IR -- identical sequence to solveGPUTwoBand.
                        solver.solve(temps_f, zeros_f, ir_f);
                        solver.uploadFaceProperties(visE_f, visR_f);
                        solver.solve(zeros_f, solar_f, vis_f);
                        solver.uploadFaceProperties(emiss_f, refl_f);
                        auto& out = indirect_per_timestep[k]; out.resize(n);
                        auto& obi = ir_batch[k];
                        auto& obv = vis_batch[k];
                        // Row norm already applied on-device (uploadStaticData ->
                        // uploadRowNorm); do not re-apply.
                        for (size_t i = 0; i < n; ++i) {
                            const double vir  = static_cast<double>(ir_f[i]);
                            const double vvis = static_cast<double>(vis_f[i]);
                            obi[i] = vir;
                            obv[i] = vvis;
                            out[i] = vir + vvis;
                        }
                    }
                    return true;
                }));
        } else {
            futures.push_back(gpu_pool_->submit(
                [&temperatures_per_timestep, &solar_flux_per_timestep,
                 &indirect_per_timestep, device_steps, n](
                    gpu::opencl::RadiosityOpenCLSolver& solver, size_t /*idx*/) -> bool {
                    std::vector<float> temps_f(n), solar_f(n), indirect_f;
                    for (size_t k : device_steps) {
                        const auto& temps_d = temperatures_per_timestep[k];
                        const auto& solar_d = solar_flux_per_timestep[k];
                        for (size_t i = 0; i < n; ++i) {
                            temps_f[i] = static_cast<float>(temps_d[i]);
                            solar_f[i] = static_cast<float>(solar_d[i]);
                        }
                        solver.solve(temps_f, solar_f, indirect_f);
                        auto& out = indirect_per_timestep[k];
                        out.resize(n);
                        // Row norm already applied on-device; do not re-apply.
                        for (size_t i = 0; i < n; ++i) {
                            out[i] = static_cast<double>(indirect_f[i]);
                        }
                    }
                    return true;
                }));
        }
    }

    bool all_ok = true;
    for (auto& f : futures) {
        if (!f.get()) all_ok = false;
    }
    if (!all_ok) {
        std::cerr << "Warning: one or more multi-GPU radiosity batches "
                     "returned failure; results may be incomplete\n";
    }

    // Aggregate convergence stats from any one solver — they are not strictly
    // tracked across batches, but the caller uses these only for diagnostics.
    last_iterations_ = 0;     // not aggregated; caller treats as advisory
    last_residual_ = 0.0;
    converged_ = all_ok;

    return indirect_per_timestep;
#endif
}

std::vector<double> RadiositySolver::solve(const std::vector<double>& temperatures,
                                            const std::vector<double>& solar_flux) {
#ifdef PTM_GPU_OPENCL
    // Use GPU path if available. Guard: if two-band was activated but the GPU
    // two-band path is NOT engaged (e.g. band separation enabled AFTER initGPU, or
    // a Σ>1 mesh), fall through to the CPU two-band solve rather than silently
    // running the gray GPU kernel for a two-band request.
    if (use_gpu_ && gpu_solver_
        && ((two_band_gpu_ok_ && src_albedo_gpu_ok_) || !two_band_)) {
        return solveGPU(temperatures, solar_flux);
    }
    // Multi-GPU pool rung: there is no single gpu_solver_, so dispatch this one
    // timestep as a one-step batch. The pool worker is arithmetically the same
    // code as solveGPUTwoBand, so this keeps a per-timestep caller on the GPU
    // instead of silently switching it to the float64 CPU solver just because
    // radiation.gpu_devices > 1. The extra conditions are exactly those under
    // which solveBatch takes its pool branch, and in_batch_serial_ makes the
    // solve -> solveBatch -> serial -> solve cycle unreachable.
    if (use_gpu_ && !gpu_solver_ && gpu_pool_ && !in_batch_serial_
        && src_albedo_gpu_ok_ && solar_albedo_source_batch_.empty()
        && (two_band_gpu_ok_ || !two_band_)) {
        auto out = solveBatch({temperatures}, {solar_flux});
        if (two_band_) {
            // Republish the per-band results in the single-solve slots the
            // two-band consumers read (lastIndirectIR/Visible).
            if (!last_indirect_ir_batch_.empty())  last_indirect_ir_  = last_indirect_ir_batch_[0];
            if (!last_indirect_vis_batch_.empty()) last_indirect_vis_ = last_indirect_vis_batch_[0];
        }
        return out.empty() ? std::vector<double>(temperatures.size(), 0.0)
                           : std::move(out[0]);
    }
    // A pool is up but this solve cannot use it: say so once rather than change
    // solver path in silence.
    if (use_gpu_ && !gpu_solver_ && gpu_pool_ && !in_batch_serial_
        && !pool_single_fallback_warned_) {
        pool_single_fallback_warned_ = true;
        std::cerr << "  [Radiosity] NOTE: the multi-GPU pool is active but this per-timestep "
                     "solve is running on the CPU (src_albedo_gpu_ok="
                  << src_albedo_gpu_ok_ << ", per-step source-albedo batch installed="
                  << (!solar_albedo_source_batch_.empty())
                  << ", two_band_gpu_ok=" << two_band_gpu_ok_
                  << "). Results are correct but come from the float64 CPU solver, not the GPU."
                  << std::endl;
    }
#endif

    size_t n = temperatures.size();

    // Band-separated (visible + thermal-IR) radiosity when requested. ON by
    // default since the D2 flip (radiation_config.h: band_separated = true);
    // set `radiation.band_separated: false` to recover the gray path below,
    // which is byte-identical to pre-two-band behavior at A = 1 - eps.
    if (two_band_) {
        return solveTwoBand(temperatures, solar_flux);
    }

    // Compute thermal emission: E_i = ε_i σ T_i^4
    auto emission = computeThermalEmission(temperatures);

    // Compute reflected solar source: ρ_solar_i Q_solar_i (Bond albedo if set, else 1-eps)
    std::vector<double> solar_source(n);
    const bool use_solar_alb = !solar_reflectivity_.empty();
    for (size_t i = 0; i < n; ++i) {
        double rs = use_solar_alb ? solar_reflectivity_[i] : reflectivity_[i];
        solar_source[i] = rs * solar_flux[i];
    }

    // Solve radiosity equation: B = E + ρ F B + ρ Q_solar
    // Rearranged: (I - ρ F) B = E + ρ Q_solar
    std::vector<double> radiosity;
    // Use Jacobi for H-matrix (no row access), Gauss-Seidel for sparse
    if (config_.use_acceleration && !use_hmatrix_) {
        radiosity = solveGaussSeidel(emission, solar_source, reflectivity_);
    } else {
        radiosity = solveJacobi(emission, solar_source, reflectivity_);
    }

    // Compute absorbed indirect flux at each face
    // Q_indirect_i = ε_i * Σ_j F_ij * B_j
    std::vector<double> indirect_flux(n, 0.0);
    std::vector<double> incoming(n, 0.0);

    // incoming_i = Σ_j F_ij B_j  (renormalized so Σ_j F_ij ≤ 1)
    F_op_->matvec(radiosity, incoming);

    for (size_t i = 0; i < n; ++i) {
        // Absorbed flux = emissivity * incoming radiosity
        // (by Kirchhoff's law, absorptivity = emissivity)
        const double norm = (i < vf_row_norm_.size()) ? vf_row_norm_[i] : 1.0;
        indirect_flux[i] = emissivity_[i] * incoming[i] * norm;
    }

    return indirect_flux;
}

// ---------------------------------------------------------------------------
// Band-separated (visible + thermal-IR) radiosity. docs/two-band-radiosity-
// derivation.tex, Eqs (4)-(6):
//   Visible:    (I - A F) B_vis = A Q_solar        (no emission source, T-indep)
//               Q_ind^vis_i = (1 - A_i) Σ_j F_ij B_vis_j
//   Thermal-IR: (I - R F) B_ir  = ε σ T^4,  R=1-ε  (the existing gray channel)
//               Q_ind^ir_i  = ε_i     Σ_j F_ij B_ir_j
//   Surface BC receives Q_ind = Q_ind^vis + Q_ind^ir (the direct (1-A)Q_solar term
//   stays in the caller's BC, so there is no double counting).
// Same F operator, same Jacobi/GS kernel, same vf_row_norm renormalization as the
// gray path. Reduces EXACTLY to the gray solve when A = 1-ε (limiting check (ii)).
// ---------------------------------------------------------------------------
std::vector<double> RadiositySolver::solveTwoBand(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux) {
    size_t n = temperatures.size();
    const std::vector<double> zero_source(n, 0.0);

    // --- Solve both bands ---
    //   IR:      (I - R F) B_ir  = E          (R = 1-eps, no solar source)
    //   Visible: (I - A F) B_vis = A Q_solar  (A = solar_reflectivity_, no emission)
    // vis_source (A_i Q_solar) is needed by both the solve and the identity below.
    auto emission = computeThermalEmission(temperatures);
    // PTM-106: the FIRST bounce scatters the collimated direct beam, so it uses the
    // direct-beam albedo A_src (== the BC's A(theta) when the caller installs it);
    // the operator/absorptivity below keep the diffuse Bond value. Falls back to
    // solar_reflectivity_ when no source albedo was installed -> legacy path,
    // byte-identical.
    const std::vector<double>& src_alb = visSourceAlbedoSingle();
    std::vector<double> vis_source(n);
    for (size_t i = 0; i < n; ++i) {
        vis_source[i] = src_alb[i] * solar_flux[i];
    }
    std::vector<double> B_ir, B_vis;
    // Gauss-Seidel is the effective solver ONLY when it can actually access rows
    // (F_ptr_ set); the SparseVFOperator VF_REPLACE path has F_ptr_==null so GS
    // falls back to Jacobi -- there we use the Tier-2 fused two-band Jacobi (one
    // width-2 matvec per iteration + per-band freeze), byte-identical to two
    // separate solveJacobi() calls but sharing the CSR traversal.
    const bool use_gs = config_.use_acceleration && !use_hmatrix_ && (F_ptr_ != nullptr);
    if (use_gs) {
        B_ir  = solveGaussSeidel(emission, zero_source, reflectivity_);
        last_iterations_ir_ = last_iterations_;
        B_vis = solveGaussSeidel(zero_source, vis_source, solar_reflectivity_);
        last_iterations_vis_ = last_iterations_;
    } else {
        solveTwoBandJacobiFused(emission, vis_source, B_ir, B_vis);
    }
    // Per-band sweep counts (Fermi's decisive measurement). Logged once/process to
    // avoid per-timestep spam; representative of the fused solve's band balance.
    static bool s_logged_band_sweeps = false;
    if (!s_logged_band_sweeps) {
        s_logged_band_sweeps = true;
        std::cout << "[two-band] per-band Jacobi sweeps (first solve): IR="
                  << last_iterations_ir_ << " visible=" << last_iterations_vis_
                  << (use_gs ? " (Gauss-Seidel, separate)"
                             : " (fused width-2 matvec while both iterate; freeze after)")
                  << std::endl;
    }

    // Fixed-point identity (Tier 1 #2), applied to each converged band: the
    // normalized incoming F·B·vf_row_norm = (B − E − source)/refl (avoids a second
    // SpMV). Guard refl→0 (blackbody) with the last Jacobi normalized incoming.
    constexpr double REFL_GUARD = 1e-9;
    last_indirect_ir_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double refl = reflectivity_[i];
        const double inc_norm = (refl > REFL_GUARD)
            ? (B_ir[i] - emission[i]) / refl
            : ((i < last_jacobi_incoming_.size()) ? last_jacobi_incoming_[i] : 0.0);
        last_indirect_ir_[i] = emissivity_[i] * inc_norm;
    }
    last_indirect_vis_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double sr = solar_reflectivity_[i];
        const double inc_norm = (sr > REFL_GUARD)
            ? (B_vis[i] - vis_source[i]) / sr
            : ((i < last_jacobi_incoming_.size()) ? last_jacobi_incoming_[i] : 0.0);
        // Absorb scattered visible with the visible absorptivity (1 - A_i).
        last_indirect_vis_[i] = (1.0 - solar_reflectivity_[i]) * inc_norm;
    }

    // --- Sum the two absorbed indirect fluxes ---
    std::vector<double> indirect_flux(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        indirect_flux[i] = last_indirect_vis_[i] + last_indirect_ir_[i];
    }
    return indirect_flux;
}

std::vector<double> RadiositySolver::computeThermalEmission(
    const std::vector<double>& temperatures) const {

    size_t n = temperatures.size();
    std::vector<double> emission(n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double T = temperatures[i];
        double T2 = T * T;
        double T4 = T2 * T2;
        emission[i] = emissivity_[i] * SIGMA_SB * T4;
    }

    return emission;
}

std::vector<double> RadiositySolver::computeScatteredSolar(
    const std::vector<double>& solar_flux) const {

    size_t n = solar_flux.size();

    // First bounce: ρ_solar_i Q_solar_i leaves face i (Bond albedo if set, else 1-eps)
    std::vector<double> scattered_source(n);
    const bool use_solar_alb = !solar_reflectivity_.empty();
    for (size_t i = 0; i < n; ++i) {
        double rs = use_solar_alb ? solar_reflectivity_[i] : reflectivity_[i];
        scattered_source[i] = rs * solar_flux[i];
    }

    // Received scattered solar: F^T * (ρ Q_solar)  (renormalized so Σ_j F_ij ≤ 1)
    std::vector<double> received(n, 0.0);
    F_op_->matvec(scattered_source, received);

    // Absorbed scattered solar
    std::vector<double> absorbed(n);
    for (size_t i = 0; i < n; ++i) {
        const double norm = (i < vf_row_norm_.size()) ? vf_row_norm_[i] : 1.0;
        absorbed[i] = emissivity_[i] * received[i] * norm;
    }

    return absorbed;
}

std::vector<double> RadiositySolver::solveJacobi(const std::vector<double>& emission,
                                                  const std::vector<double>& solar_source,
                                                  const std::vector<double>& scatter_refl) {
    size_t n = emission.size();

    // Use local arrays instead of member arrays for thread safety
    // This allows multiple threads to call solveJacobi concurrently
    std::vector<double> radiosity(n);
    std::vector<double> radiosity_new(n);
    std::vector<double> temp_vec(n);

    // Initialize radiosity with emission + solar source
    for (size_t i = 0; i < n; ++i) {
        radiosity[i] = emission[i] + solar_source[i];
    }

    double local_residual = std::numeric_limits<double>::max();
    bool local_converged = false;
    int local_iterations = 0;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // B_new = E + ρ Q_solar + ρ F B
        // Note: matvec is internally parallelized over H-matrix blocks
        F_op_->matvec(radiosity, temp_vec);
        // Renormalize Σ_j F_ij ≤ 1 so the radiosity gain ρ·Σ stays < 1 (stable).
        for (size_t i = 0; i < n && i < vf_row_norm_.size(); ++i) {
            temp_vec[i] *= vf_row_norm_[i];
        }

        double max_change = 0.0;
        double max_val = 0.0;

        // This loop is fast, no need to parallelize (would conflict with outer parallelism)
        for (size_t i = 0; i < n; ++i) {
            radiosity_new[i] = emission[i] + solar_source[i] + scatter_refl[i] * temp_vec[i];

            double change = std::abs(radiosity_new[i] - radiosity[i]);
            max_change = std::max(max_change, change);
            max_val = std::max(max_val, std::abs(radiosity_new[i]));
        }

        // Swap buffers
        std::swap(radiosity, radiosity_new);

        // Check convergence
        double rel_residual = (max_val > 1e-10) ? max_change / max_val : max_change;
        local_residual = rel_residual;
        local_iterations = iter + 1;

        if (rel_residual < config_.convergence_tolerance) {
            local_converged = true;
            break;
        }
    }

    // Update member variables (for diagnostics) - not thread-safe but only for reporting
    last_residual_ = local_residual;
    last_iterations_ = local_iterations;
    converged_ = local_converged;
    // temp_vec holds F·B·vf_row_norm from the final iteration -- the normalized
    // incoming. solveTwoBand uses the (B−E−source)/refl identity for this, and
    // falls back to this array where refl→0 (see the header note).
    last_jacobi_incoming_ = temp_vec;

    return radiosity;
}

// Tier 2: fused two-band Jacobi. Byte-identical to
//   B_ir  = solveJacobi(emission,   0,          reflectivity_)
//   B_vis = solveJacobi(0,          vis_source, solar_reflectivity_)
// run separately -- same per-band init/update/normalization/convergence and the
// same per-band sweep count -- but while BOTH bands are still iterating it drives
// them through one width-2 matvec (matvec2), reading the CSR once for both. The
// instant a band's rel-residual drops below tolerance it is FROZEN (its converged
// radiosity is kept, its matvec skipped), so the slower band finishes on a single-
// band matvec exactly as it would alone. Only the effective-Jacobi path uses this
// (see solveTwoBand); the arithmetic here mirrors solveJacobi() line-for-line.
void RadiositySolver::solveTwoBandJacobiFused(
    const std::vector<double>& emission,
    const std::vector<double>& vis_source,
    std::vector<double>& B_ir, std::vector<double>& B_vis) {
    const size_t n = emission.size();
    std::vector<double> r_ir(n), rn_ir(n), tv_ir(n);
    std::vector<double> r_vis(n), rn_vis(n), tv_vis(n);
    for (size_t i = 0; i < n; ++i) {
        r_ir[i]  = emission[i];    // IR init  = emission + 0 (solveJacobi: E + solar_source)
        r_vis[i] = vis_source[i];  // vis init = 0 + vis_source
    }
    bool conv_ir = false, conv_vis = false;
    int it_ir = 0, it_vis = 0;
    // BUGS.md PTM-102: this path never wrote converged_ / last_residual_, so
    // didConverge() and getLastResidual() reported whatever the previous solve (or
    // the constructor) left behind. test_two_band_rownorm printed "JACOBI FAILED
    // (unexpected)" on all 4 configs while the numbers were in fact correct to
    // 1e-10, and production stats_.solver_residual was stale for the DEFAULT fused
    // CPU two-band path. Track the per-band relative residual and publish both.
    double rel_ir  = std::numeric_limits<double>::max();
    double rel_vis = std::numeric_limits<double>::max();
    const double tol = config_.convergence_tolerance;
    for (int iter = 0; iter < config_.max_iterations && !(conv_ir && conv_vis); ++iter) {
        if (!conv_ir && !conv_vis) {
            F_op_->matvec2(r_ir, r_vis, tv_ir, tv_vis);   // shared traversal
        } else if (!conv_ir) {
            F_op_->matvec(r_ir, tv_ir);
        } else {
            F_op_->matvec(r_vis, tv_vis);
        }
        if (!conv_ir) {
            for (size_t i = 0; i < n && i < vf_row_norm_.size(); ++i) tv_ir[i] *= vf_row_norm_[i];
            double max_change = 0.0, max_val = 0.0;
            for (size_t i = 0; i < n; ++i) {
                rn_ir[i] = emission[i] + reflectivity_[i] * tv_ir[i];
                max_change = std::max(max_change, std::abs(rn_ir[i] - r_ir[i]));
                max_val    = std::max(max_val,    std::abs(rn_ir[i]));
            }
            std::swap(r_ir, rn_ir);
            it_ir = iter + 1;
            const double rel = (max_val > 1e-10) ? max_change / max_val : max_change;
            rel_ir = rel;
            if (rel < tol) conv_ir = true;
        }
        if (!conv_vis) {
            for (size_t i = 0; i < n && i < vf_row_norm_.size(); ++i) tv_vis[i] *= vf_row_norm_[i];
            double max_change = 0.0, max_val = 0.0;
            for (size_t i = 0; i < n; ++i) {
                rn_vis[i] = vis_source[i] + solar_reflectivity_[i] * tv_vis[i];
                max_change = std::max(max_change, std::abs(rn_vis[i] - r_vis[i]));
                max_val    = std::max(max_val,    std::abs(rn_vis[i]));
            }
            std::swap(r_vis, rn_vis);
            it_vis = iter + 1;
            const double rel = (max_val > 1e-10) ? max_change / max_val : max_change;
            rel_vis = rel;
            if (rel < tol) conv_vis = true;
        }
    }
    // Dead-path fallback source for the #2 identity (refl→0 only): match
    // solveJacobi()'s final-call state (the visible band was solved last).
    last_jacobi_incoming_ = tv_vis;
    last_iterations_ir_  = it_ir;
    last_iterations_vis_ = it_vis;
    last_iterations_     = it_vis;  // aggregate getter: last band solved, as before
    // PTM-102: a TWO-band solve has converged only when BOTH bands have, and its
    // residual is the worse of the two. This is deliberately STRICTER than the
    // "two separate solveJacobi calls" aggregate it otherwise mirrors -- that one
    // would report only the visible band (the last call), hiding an unconverged IR
    // band entirely. Diagnostic only: nothing branches on these (consumers are
    // test_two_band_rownorm and stats_.solver_residual).
    converged_     = conv_ir && conv_vis;
    last_residual_ = std::max(rel_ir, rel_vis);
    B_ir  = std::move(r_ir);
    B_vis = std::move(r_vis);
}

std::vector<double> RadiositySolver::solveGaussSeidel(const std::vector<double>& emission,
                                                       const std::vector<double>& solar_source,
                                                       const std::vector<double>& scatter_refl) {
    size_t n = emission.size();

    // Gauss-Seidel requires row access; fall back to Jacobi if not available
    if (!F_ptr_) {
        return solveJacobi(emission, solar_source, scatter_refl);
    }

    // Initialize radiosity with emission + solar source
    radiosity_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        radiosity_[i] = emission[i] + solar_source[i];
    }

    last_residual_ = std::numeric_limits<double>::max();
    converged_ = false;

    std::vector<int> cols;
    std::vector<double> vals;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        double max_change = 0.0;
        double max_val = 0.0;

        // Gauss-Seidel: update each row using latest values
        for (size_t i = 0; i < n; ++i) {
            // Get row i of view factor matrix
            F_ptr_->getRow(static_cast<int>(i), cols, vals);

            // Compute sum F_ij * B_j using latest B values
            double sum = 0.0;
            for (size_t k = 0; k < cols.size(); ++k) {
                sum += vals[k] * radiosity_[cols[k]];
            }

            // Renormalize Σ_j F_ij ≤ 1 (BUGS.md PTM-08). The M1 estimator can
            // overshoot to Σ_j F_ij > 1 on fine/concave meshes; without this the
            // radiosity gain scatter_refl·Σ can exceed 1 and Gauss-Seidel diverges
            // (severe in the two-band visible channel, A up to ~0.5 Europa/~0.9
            // frost). Mirrors the Jacobi renormalization above (temp_vec *= norm),
            // so the two kernels agree and ρ(A·F̂) < 1 holds for both bands.
            if (i < vf_row_norm_.size()) sum *= vf_row_norm_[i];

            // Update B_i
            double B_new = emission[i] + solar_source[i] + scatter_refl[i] * sum;

            double change = std::abs(B_new - radiosity_[i]);
            max_change = std::max(max_change, change);
            max_val = std::max(max_val, std::abs(B_new));

            radiosity_[i] = B_new;
        }

        // Check convergence
        double rel_residual = (max_val > 1e-10) ? max_change / max_val : max_change;
        last_residual_ = rel_residual;
        last_iterations_ = iter + 1;

        if (rel_residual < config_.convergence_tolerance) {
            converged_ = true;
            break;
        }
    }

    return radiosity_;
}

std::array<double, 3> RadiositySolver::checkEnergyBalance(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux,
    const std::vector<double>& indirect_flux) const {

    size_t n = temperatures.size();

    double total_emission = 0.0;
    double total_solar_absorbed = 0.0;
    double total_indirect = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double T = temperatures[i];
        double T4 = T * T * T * T;
        total_emission += emissivity_[i] * SIGMA_SB * T4 * areas_[i];
        total_solar_absorbed += emissivity_[i] * solar_flux[i] * areas_[i];
        total_indirect += indirect_flux[i] * areas_[i];
    }

    double imbalance = std::abs(total_emission - total_solar_absorbed - total_indirect);

    return {{total_emission, total_solar_absorbed + total_indirect, imbalance}};
}

// ============================================================================
// FaceSelector Implementation
// ============================================================================

FaceSelector::FaceSelector(const std::shared_ptr<ptm::Mesh>& mesh,
                           const RadiationConfig& config)
    : mesh_(mesh), config_(config) {

    // Use pre-computed face geometry from mesh if available
    // PTM3D's MeshUtils ensures normals point outward (away from body center)
    size_t n = mesh_->faces.size();
    face_normals_.resize(n);
    face_centers_.resize(n);

    bool use_mesh_data = (mesh_->face_centers.size() == n &&
                          mesh_->face_normals.size() == n);

    if (use_mesh_data) {
        // Use pre-computed outward-oriented normals from mesh
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            face_centers_[i] = mesh_->face_centers[i];
            face_normals_[i] = mesh_->face_normals[i];
        }
    } else {
        // Compute from scratch and ensure outward orientation
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];

            // Center
            face_centers_[i] = {{
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            }};

            // Normal (cross product of edges)
            std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
            std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};

            face_normals_[i] = {{
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            }};

            double len = std::sqrt(face_normals_[i][0] * face_normals_[i][0] +
                                   face_normals_[i][1] * face_normals_[i][1] +
                                   face_normals_[i][2] * face_normals_[i][2]);
            if (len > 1e-10) {
                face_normals_[i][0] /= len;
                face_normals_[i][1] /= len;
                face_normals_[i][2] /= len;
            }

            // Ensure outward-pointing normals (away from body center at origin)
            // PTM3D convention: normal · center > 0 means outward
            const auto& c = face_centers_[i];
            double radial_dot = face_normals_[i][0] * c[0] +
                                face_normals_[i][1] * c[1] +
                                face_normals_[i][2] * c[2];
            if (radial_dot < 0) {
                face_normals_[i][0] = -face_normals_[i][0];
                face_normals_[i][1] = -face_normals_[i][1];
                face_normals_[i][2] = -face_normals_[i][2];
            }
        }
    }
}

FaceClassification FaceSelector::classify(
    const std::vector<std::vector<double>>& illumination_series,
    const std::vector<std::vector<double>>& temperatures) {

    size_t n = mesh_->faces.size();
    FaceClassification result;
    result.resize(n);

    // Always show progress messages for these potentially slow operations
    // Start on new line since caller may have printed without newline
    std::cout << std::endl;
    std::cout << "  Computing shadow fractions..." << std::flush;

    // Compute shadow fractions
    result.shadow_fractions = computeShadowFractions(illumination_series);

    std::cout << " done" << std::endl;

    // Extract minimum temperatures if provided
    if (!temperatures.empty() && temperatures.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            if (!temperatures[i].empty()) {
                result.min_temperatures[i] = *std::min_element(
                    temperatures[i].begin(), temperatures[i].end());
            }
        }
    }

    // PTM-98 — which faces keep the indirect flux the radiosity solve computes.
    //
    // View factors and the radiosity solve itself cover ALL faces regardless of
    // this classification; the set below is applied afterwards as a copy mask
    // (computeIndirectFlux / computeIndirectFluxBatch). Filtering for thermal
    // re-runs is a SEPARATE, downstream decision made on the computed indirect
    // flux (PTM-89 coupling_mode).
    //
    // The legacy Shadow gate is window-dependent: shadow_fractions comes from
    // this run's own illumination series, whose length is
    // n_equilibration_steps + n_output_steps, and a warm-started chunk has no
    // equilibration samples. Splitting a window therefore reclassifies faces
    // OUT of radiation exchange and zeroes their flux. Default is All.
    const bool select_all =
        (config_.exchange_selection == RadiationConfig::ExchangeSelection::All);
    for (size_t i = 0; i < n; ++i) {
        result.needs_exchange[i] =
            select_all || (result.shadow_fractions[i] >= config_.min_shadow_fraction);
    }

    result.finalize();

    // Never drop faces silently: say how many the legacy gate is zeroing.
    if (!select_all && result.exchange_faces < result.total_faces) {
        std::cout << "  WARNING [PTM-98] radiation.exchange_selection='shadow': "
                  << (result.total_faces - result.exchange_faces) << " / "
                  << result.total_faces << " faces are EXCLUDED from radiation"
                     " exchange and will receive exactly 0 W/m^2 indirect flux."
                     " This gate is measured over THIS window ("
                  << (illumination_series.empty() ? 0 : illumination_series[0].size())
                  << " illumination samples), so the result depends on how the run"
                     " was chunked. Use exchange_selection: all (default) unless you"
                     " are reproducing a pre-PTM-98 run." << std::endl;
    }

    if (config_.verbose) {
        std::cout << "Face classification: " << result.exchange_faces << " / "
                  << result.total_faces << " faces need radiation exchange ("
                  << (100.0 * result.exchange_faces / result.total_faces) << "%)"
                  << std::endl;
        std::cout << "  Mean shadow fraction: " << result.mean_shadow_fraction << std::endl;
    }

    return result;
}

FaceClassification FaceSelector::classifyGeometric() {
    // Quick classification based on geometry only
    std::vector<std::vector<double>> empty_illum;
    return classify(empty_illum);
}

std::vector<double> FaceSelector::computeShadowFractions(
    const std::vector<std::vector<double>>& illumination_series) const {

    size_t n = mesh_->faces.size();
    std::vector<double> fractions(n, 0.0);

    if (illumination_series.empty()) return fractions;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (i < illumination_series.size() && !illumination_series[i].empty()) {
            int shadow_count = 0;
            for (double flux : illumination_series[i]) {
                if (flux < 1e-6) {  // Effectively zero flux = in shadow
                    shadow_count++;
                }
            }
            fractions[i] = static_cast<double>(shadow_count) / illumination_series[i].size();
        }
    }

    return fractions;
}

// ============================================================================
// RadiationExchangeManager Implementation
// ============================================================================

RadiationExchangeManager::RadiationExchangeManager(
    const std::shared_ptr<ptm::Mesh>& mesh,
    const RadiationConfig& config)
    : mesh_(mesh), config_(config) {

    // Automatic H-matrix selection for large meshes
    // Dense view factors require O(n^2) memory which is impractical for large meshes
    // Threshold: 100K faces -> 10B entries -> 80GB for dense matrix (infeasible)
    const size_t HMATRIX_AUTO_THRESHOLD = 100000;
    size_t n_faces = mesh_->faces.size();

    if (n_faces > HMATRIX_AUTO_THRESHOLD && !config_.use_hmatrix) {
        std::cout << "Large mesh detected (" << n_faces << " faces)" << std::endl;
        std::cout << "  Automatically enabling H-matrix compression (dense would require ~"
                  << (n_faces * n_faces * 8ULL) / (1024ULL * 1024 * 1024) << " GB)" << std::endl;
        config_.use_hmatrix = true;
    }

    if (config_.verbose) {
        std::cout << "[RadiationExchangeManager] Config received:" << std::endl;
        std::cout << "  use_hmatrix=" << (config_.use_hmatrix ? "true" : "false") << std::endl;
        std::cout << "  use_gpu=" << (config_.use_gpu ? "true" : "false") << std::endl;
        std::cout << std::scientific << "  visibility_threshold=" << config_.visibility_threshold
                  << std::defaultfloat << std::endl;
        if (config_.use_hmatrix) {
            std::cout << "  hmatrix_tolerance=" << config_.hmatrix_tolerance << std::endl;
            std::cout << "  hmatrix_max_rank=" << config_.hmatrix_max_rank << std::endl;
        }
    }

    size_t n = mesh_->faces.size();
    emissivity_.assign(n, 0.95);  // Default emissivity
    areas_.resize(n);

    // Compute face areas
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& face = mesh_->faces[i];
        const auto& v0 = mesh_->vertices[face[0]];
        const auto& v1 = mesh_->vertices[face[1]];
        const auto& v2 = mesh_->vertices[face[2]];

        std::array<double, 3> e1 = {{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]}};
        std::array<double, 3> e2 = {{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]}};
        std::array<double, 3> cross = {{
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        }};
        areas_[i] = 0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    }

    indirect_flux_.resize(n, 0.0);
}

RadiationExchangeManager::~RadiationExchangeManager() {
    cleanup();
}

void RadiationExchangeManager::cleanup() {
    if (scene_) {
        rtcReleaseScene(static_cast<RTCScene>(scene_));
        scene_ = nullptr;
    }
    if (device_) {
        rtcReleaseDevice(static_cast<RTCDevice>(device_));
        device_ = nullptr;
    }
    scene_initialized_ = false;
}

bool RadiationExchangeManager::initializeScene() {
    if (scene_initialized_) return true;

    // Create Embree device
    device_ = rtcNewDevice(nullptr);
    if (!device_) {
        std::cerr << "Error: Failed to create Embree device" << std::endl;
        return false;
    }

    // Create scene
    scene_ = rtcNewScene(static_cast<RTCDevice>(device_));
    if (!scene_) {
        std::cerr << "Error: Failed to create Embree scene" << std::endl;
        cleanup();
        return false;
    }

    RTCScene rtc_scene = static_cast<RTCScene>(scene_);
    RTCDevice rtc_device = static_cast<RTCDevice>(device_);

    // Create geometry
    RTCGeometry geom = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Set vertices
    size_t n_verts = mesh_->vertices.size();
    float* vertices = static_cast<float*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, 3 * sizeof(float), n_verts));

    // Recenter to the mesh bounding-box center before float32 conversion. At
    // planetary-scale absolute coordinates (e.g. lunar selenocentric, |Z|~1.7e6 m)
    // a float32 ulp (~0.1 m) exceeds a meter-scale patch's features, collapsing
    // Embree's BVH and making visibility ray tracing pathologically slow. The
    // H-matrix builder (HMatrixViewFactor::build) computes the IDENTICAL shift
    // from the same mesh_->vertices and applies it to its ray origins, so rays
    // and scene share one small-coordinate space. View factors depend only on
    // relative geometry, so the constant shift is exact.
    std::array<double,3> bbmin = {{ mesh_->vertices[0][0], mesh_->vertices[0][1], mesh_->vertices[0][2] }};
    std::array<double,3> bbmax = bbmin;
    for (size_t i = 1; i < n_verts; ++i) {
        for (int k = 0; k < 3; ++k) {
            bbmin[k] = std::min(bbmin[k], mesh_->vertices[i][k]);
            bbmax[k] = std::max(bbmax[k], mesh_->vertices[i][k]);
        }
    }
    const std::array<double,3> origin_shift = {{
        0.5 * (bbmin[0] + bbmax[0]),
        0.5 * (bbmin[1] + bbmax[1]),
        0.5 * (bbmin[2] + bbmax[2])
    }};
    std::cout << "  Radiation Embree scene recentered by ("
              << origin_shift[0] << ", " << origin_shift[1] << ", " << origin_shift[2]
              << ") for float32 precision" << std::endl;

    for (size_t i = 0; i < n_verts; ++i) {
        vertices[3 * i + 0] = static_cast<float>(mesh_->vertices[i][0] - origin_shift[0]);
        vertices[3 * i + 1] = static_cast<float>(mesh_->vertices[i][1] - origin_shift[1]);
        vertices[3 * i + 2] = static_cast<float>(mesh_->vertices[i][2] - origin_shift[2]);
    }

    // Set indices
    size_t n_faces = mesh_->faces.size();
    unsigned int* indices = static_cast<unsigned int*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), n_faces));

    for (size_t i = 0; i < n_faces; ++i) {
        indices[3 * i + 0] = static_cast<unsigned int>(mesh_->faces[i][0]);
        indices[3 * i + 1] = static_cast<unsigned int>(mesh_->faces[i][1]);
        indices[3 * i + 2] = static_cast<unsigned int>(mesh_->faces[i][2]);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(rtc_scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(rtc_scene);

    // Check for Embree errors after scene commit
    // Large meshes can fail during BVH construction
    RTCError err = rtcGetDeviceError(static_cast<RTCDevice>(device_));
    if (err != RTC_ERROR_NONE) {
        std::cerr << "Embree scene commit error (code " << err << ")" << std::endl;
        std::cerr << "Mesh may be too large (" << mesh_->faces.size() << " faces)" << std::endl;
        cleanup();
        return false;
    }

    scene_initialized_ = true;
    return true;
}

bool RadiationExchangeManager::computeViewFactorsOnly() {
    // Compute view factors without face classification or solver setup.
    // This is used for early computation before illumination data is available.
    // The view factors can be saved to disk and loaded later during prepare().

    if (!config_.enabled) {
        return false;
    }

    size_t n_faces = mesh_->faces.size();
    std::cout << "Computing view factors for " << n_faces << " faces..." << std::endl;

    if (n_faces > 1000000) {
        size_t embree_estimate = n_faces * 80;
        std::cout << "  Estimated Embree BVH: ~" << embree_estimate / (1024*1024) << " MB" << std::endl;
    }

    // Initialize Embree scene
    if (!initializeScene()) {
        std::cerr << "Error: Failed to initialize scene for view factor computation" << std::endl;
        return false;
    }

    // Method selection: Monte Carlo overrides other paths when enabled.
    // MC was introduced because H-matrix build memory is infeasible on 4M+ meshes
    // on this machine (killed by jetsam at ~250 GB RSS) and sparse-VF hits the
    // max_neighbors cap for 99.5% of faces before reaching the energy target.
    bool use_hmatrix = config_.use_hmatrix;
    const bool use_mc = config_.use_monte_carlo_vf;
    std::cout << "  [DEBUG] View factor method selection:" << std::endl;
    std::cout << "    use_monte_carlo_vf = " << (use_mc ? "true" : "false") << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;

    if (use_hmatrix && !use_mc && n_faces > 1000000) {
        std::cout << "  Large mesh (>1M faces): H-matrix will use out-of-core storage." << std::endl;
    }

    if (use_mc) {
        std::cout << "Using Monte Carlo ray-gather for view factors..." << std::endl;
        view_factors_ = computeMonteCarlo(mesh_,
                                          static_cast<RTCDevice>(device_),
                                          static_cast<RTCScene>(scene_),
                                          config_,
                                          config_.mc_rays_per_face);
        using_hmatrix_ = false;
        std::cout << "Monte Carlo view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    } else if (use_hmatrix) {
        std::cout << "Using H-matrix compression for view factors..." << std::endl;

        HMatrixConfig hconfig;
        hconfig.compression_tolerance = config_.hmatrix_tolerance;
        hconfig.max_tree_depth = config_.hmatrix_max_depth;
        hconfig.min_block_size = config_.hmatrix_min_block_size;
        hconfig.max_svd_rank = config_.hmatrix_max_rank;
        hconfig.verbose = config_.verbose;

        // ACA configuration (Adaptive Cross Approximation - much faster)
        hconfig.compression_method = config_.use_aca ? CompressionMethod::ACA : CompressionMethod::SVD;
        hconfig.aca_tolerance = config_.aca_tolerance;
        hconfig.aca_max_rank = config_.aca_max_rank;
        hconfig.subdivide_large_admissible = config_.hmatrix_subdivide_admissible;
        hconfig.hmatrix_threads = config_.hmatrix_threads;

        if (config_.use_aca) {
            // Reset float formatting for this diagnostic. An upstream caller can
            // leave std::cout in a std::fixed / low-precision state, which
            // truncated the configured tolerance (e.g. 0.001) to a misleading
            // "0.0" in the VF-build log. The value handed to the H-matrix builder
            // (hconfig.aca_tolerance = config_.aca_tolerance, above) was always
            // honored — proven: the compression ratio is identical whether the log
            // shows "0.0" or "0.001". This is display-only; see BUGS.md PTM-09.
            // Save/restore so downstream formatting is left unchanged.
            const std::ios::fmtflags saved_fmt = std::cout.flags();
            const std::streamsize saved_prec = std::cout.precision();
            std::cout.unsetf(std::ios::floatfield);
            std::cout.precision(6);
            std::cout << "  Using ACA compression (tolerance=" << config_.aca_tolerance
                      << ", max_rank=" << config_.aca_max_rank << ")" << std::endl;
            std::cout.flags(saved_fmt);
            std::cout.precision(saved_prec);
        } else {
            std::cout << "  Using SVD compression" << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();
        hmatrix_view_factors_->build(mesh_,
                                     static_cast<RTCDevice>(device_),
                                     static_cast<RTCScene>(scene_),
                                     config_,
                                     hconfig);

        using_hmatrix_ = true;

        std::cout << "H-matrix view factors computed:" << std::endl;
        std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "  Compression ratio: " << hmatrix_view_factors_->compressionRatio() << "x" << std::endl;
        if (config_.verbose) {
            hmatrix_view_factors_->printStatistics();
        }
    } else {
        std::cout << "Using sparse view factor matrix..." << std::endl;

        ViewFactorCalculator calculator(mesh_,
                                        static_cast<RTCDevice>(device_),
                                        static_cast<RTCScene>(scene_),
                                        config_);

        view_factors_ = calculator.computeAll();
        using_hmatrix_ = false;

        std::cout << "Sparse view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }

    // Note: We don't set view_factors_loaded_ = true here because we want
    // prepare() to recognize these as freshly computed (for saving to disk).
    // The solver is NOT created here - that happens in prepare() when
    // illumination data and face classification are available.

    return true;
}

bool RadiationExchangeManager::prepare(
    const std::vector<std::vector<double>>& illumination_series) {

    if (!config_.enabled) {
        is_active_ = false;
        return false;
    }

    // Note: Auto H-matrix selection is done in constructor

    size_t n_faces = mesh_->faces.size();
    if (n_faces > 1000000) {
        // Additional info for very large meshes
        size_t embree_estimate = n_faces * 80;  // ~80 bytes per face for BVH
        std::cout << "  Estimated Embree BVH: ~" << embree_estimate / (1024*1024) << " MB" << std::endl;
    }

    // Initialize Embree scene
    if (!initializeScene()) {
        std::cerr << "Error: Failed to initialize scene for radiation exchange" << std::endl;
        is_active_ = false;
        return false;
    }

    // Classify faces
    FaceSelector selector(mesh_, config_);
    classification_ = selector.classify(illumination_series);

    // PTM-101: the beamed-emission receiver/emitter sets are still shadow-fraction
    // based and therefore window-dependent, even though the exchange set no longer
    // is. Beaming is off by default (rms_slope_deg == 0); say so when it is on.
    if (config_.rms_slope_deg > 0.0) {
        std::cout << "  WARNING [PTM-101] rms_slope_deg=" << config_.rms_slope_deg
                  << " > 0: the Smith beamed-emission receiver and emitter sets are"
                     " selected from shadow fractions measured over THIS window, so a"
                     " beaming run is still chunk-decomposition dependent. The PTM-98"
                     " fix covers the radiation-exchange set only. REMEDY: run beaming"
                     " single-shot (one window, no chunking), or accept and state the"
                     " chunk dependence of the result." << std::endl;
    }

    if (classification_.exchange_faces == 0) {
        if (config_.verbose) {
            std::cout << "No faces require radiation exchange" << std::endl;
        }
        is_active_ = false;
        return false;
    }

    // Check if view factors were pre-loaded from file
    if (view_factors_loaded_) {
        if (config_.verbose) {
            std::cout << "Using pre-loaded view factors (skipping computation)" << std::endl;
        }

        // Create radiosity solver with the loaded view factors
        if (using_hmatrix_) {
            solver_ = std::make_unique<RadiositySolver>(*hmatrix_view_factors_, emissivity_, areas_);
            if (config_.verbose) {
                std::cout << "Radiation exchange prepared (loaded H-matrix):" << std::endl;
                std::cout << "  Exchange faces: " << classification_.exchange_faces << std::endl;
                std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        } else {
            solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
            if (config_.verbose) {
                std::cout << "Radiation exchange prepared (loaded sparse):" << std::endl;
                std::cout << "  Exchange faces: " << classification_.exchange_faces << std::endl;
                std::cout << "  View factor matrix size: " << view_factors_.numNonZeros()
                          << " non-zeros" << std::endl;
                std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        }

        // CRITICAL: Set config and initialize GPU for loaded view factors
        // Without this, GPU memory check is skipped and GPU crashes on large matrices
        solver_->setConfig(config_);
        // A2: apply the per-face Bond albedo (if set) before initGPU so the GPU
        // path (once two-band-aware) sees it too.
        if (!solar_albedo_.empty()) solver_->setSolarAlbedo(solar_albedo_);
        // D1: config-driven two-band activation (before initGPU so the GPU gate sees it).
        if (config_.band_separated) solver_->enableTwoBand();
        solver_->initGPU();

        is_active_ = true;
        if (config_.rms_slope_deg > 0.0) {
            chi_w_rad_ = config_.rms_slope_deg * M_PI / 180.0;
            buildChiTable();
            buildBeamPairs();
            beam_active_ = !beam_pairs_.empty();
            configureSolarBeaming();
        }
        return true;
    }

    // Compute view factors for ALL faces, not just exchange faces.
    // This is critical because:
    // 1. Exchange faces (shadowed floor) need to RECEIVE radiation from illuminated walls
    // 2. Illuminated wall faces are NOT exchange faces (they're well-lit)
    // 3. For floor to receive radiation from wall, we need F_floor,wall which requires
    //    computing view factors FROM floors TO walls (and walls back to floors)
    // 4. The radiosity solver needs B_wall = ε*σ*T_wall^4 to propagate to floors
    //
    // Computing view factors for all faces ensures that warm illuminated surfaces
    // can contribute thermal radiation to cold shadowed surfaces.
    if (config_.verbose) {
        std::cout << "Computing view factors for ALL " << mesh_->faces.size()
                  << " faces (not just " << classification_.exchange_faces
                  << " exchange faces)..." << std::endl;
        std::cout << "  This ensures illuminated walls can radiate to shadowed floors." << std::endl;
    }

    // Method selection: Monte Carlo overrides when enabled. See
    // computeViewFactorsOnly() for the rationale.
    bool use_hmatrix = config_.use_hmatrix;
    const bool use_mc = config_.use_monte_carlo_vf;
    std::cout << "  [DEBUG] View factor method selection (prepare):" << std::endl;
    std::cout << "    use_monte_carlo_vf = " << (use_mc ? "true" : "false") << std::endl;
    std::cout << "    config_.use_hmatrix = " << (config_.use_hmatrix ? "true" : "false") << std::endl;
    std::cout << "    n_faces = " << n_faces << std::endl;

    if (use_hmatrix && !use_mc && n_faces > 1000000) {
        std::cout << "  Large mesh (>1M faces): H-matrix will use out-of-core storage." << std::endl;
    }

    if (use_mc) {
        std::cout << "Using Monte Carlo ray-gather for view factors..." << std::endl;
        view_factors_ = computeMonteCarlo(mesh_,
                                          static_cast<RTCDevice>(device_),
                                          static_cast<RTCScene>(scene_),
                                          config_,
                                          config_.mc_rays_per_face);
        solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
        using_hmatrix_ = false;
        std::cout << "Monte Carlo view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    } else if (use_hmatrix) {
        std::cout << "Using H-matrix compression for view factors..." << std::endl;

        // Create H-matrix configuration
        HMatrixConfig hconfig;
        hconfig.compression_tolerance = config_.hmatrix_tolerance;
        hconfig.max_tree_depth = config_.hmatrix_max_depth;
        hconfig.min_block_size = config_.hmatrix_min_block_size;
        hconfig.max_svd_rank = config_.hmatrix_max_rank;
        hconfig.verbose = config_.verbose;

        // ACA configuration (Adaptive Cross Approximation - much faster)
        hconfig.compression_method = config_.use_aca ? CompressionMethod::ACA : CompressionMethod::SVD;
        hconfig.aca_tolerance = config_.aca_tolerance;
        hconfig.aca_max_rank = config_.aca_max_rank;
        hconfig.subdivide_large_admissible = config_.hmatrix_subdivide_admissible;
        hconfig.hmatrix_threads = config_.hmatrix_threads;

        if (config_.use_aca) {
            // Reset float formatting for this diagnostic. An upstream caller can
            // leave std::cout in a std::fixed / low-precision state, which
            // truncated the configured tolerance (e.g. 0.001) to a misleading
            // "0.0" in the VF-build log. The value handed to the H-matrix builder
            // (hconfig.aca_tolerance = config_.aca_tolerance, above) was always
            // honored — proven: the compression ratio is identical whether the log
            // shows "0.0" or "0.001". This is display-only; see BUGS.md PTM-09.
            // Save/restore so downstream formatting is left unchanged.
            const std::ios::fmtflags saved_fmt = std::cout.flags();
            const std::streamsize saved_prec = std::cout.precision();
            std::cout.unsetf(std::ios::floatfield);
            std::cout.precision(6);
            std::cout << "  Using ACA compression (tolerance=" << config_.aca_tolerance
                      << ", max_rank=" << config_.aca_max_rank << ")" << std::endl;
            std::cout.flags(saved_fmt);
            std::cout.precision(saved_prec);
        } else {
            std::cout << "  Using SVD compression" << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();
        hmatrix_view_factors_->build(mesh_,
                                     static_cast<RTCDevice>(device_),
                                     static_cast<RTCScene>(scene_),
                                     config_,
                                     hconfig);

        // Create radiosity solver with H-matrix
        solver_ = std::make_unique<RadiositySolver>(*hmatrix_view_factors_, emissivity_, areas_);
        using_hmatrix_ = true;

        // Always show H-matrix summary (memory savings are important info)
        std::cout << "H-matrix view factors computed:" << std::endl;
        std::cout << "  Memory usage: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "  Compression ratio: " << hmatrix_view_factors_->compressionRatio() << "x" << std::endl;
        if (config_.verbose) {
            hmatrix_view_factors_->printStatistics();
        }
    } else {
        // Standard sparse view factor computation
        std::cout << "Using sparse view factor matrix..." << std::endl;

        ViewFactorCalculator calculator(mesh_,
                                        static_cast<RTCDevice>(device_),
                                        static_cast<RTCScene>(scene_),
                                        config_);

        // Compute for ALL faces so that wall->floor radiation is captured
        view_factors_ = calculator.computeAll();

        // Create radiosity solver
        solver_ = std::make_unique<RadiositySolver>(view_factors_, emissivity_, areas_);
        using_hmatrix_ = false;

        // Always show sparse matrix summary
        std::cout << "Sparse view factors computed:" << std::endl;
        std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
        std::cout << "  Memory usage: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }

    solver_->setConfig(config_);
    // A2: apply the per-face Bond albedo (if set) before initGPU.
    if (!solar_albedo_.empty()) solver_->setSolarAlbedo(solar_albedo_);
    // D1: config-driven two-band activation (before initGPU so the GPU gate sees it).
    if (config_.band_separated) solver_->enableTwoBand();

    // Initialize GPU solver if requested (must be after setConfig)
    solver_->initGPU();

    is_active_ = true;

    // Smith (1967) anisotropic beamed-emission setup (only when rms_slope_deg>0).
    if (config_.rms_slope_deg > 0.0) {
        chi_w_rad_ = config_.rms_slope_deg * M_PI / 180.0;
        buildChiTable();
        buildBeamPairs();
        beam_active_ = !beam_pairs_.empty();
        configureSolarBeaming();
    }

    return true;
}

bool RadiationExchangeManager::hasGPUPool() const {
    return solver_ ? solver_->hasGPUPool() : false;
}

void RadiationExchangeManager::setSolarAlbedo(const std::vector<double>& solar_albedo) {
    // A2: store the per-face Bond albedo and forward to the solver if it exists.
    // If called before prepare() (solver_ not yet created) it is applied at solver
    // creation; if called per-timestep after prepare() it retargets immediately.
    solar_albedo_ = solar_albedo;
    if (solver_ && !solar_albedo_.empty()) {
        solver_->setSolarAlbedo(solar_albedo_);
    }
}

// PTM-106 forwarders. Unlike setSolarAlbedo these are NOT cached across prepare():
// the source albedo is a per-timestep quantity that only means anything once the
// solver exists and the caller is inside the coupling loop. Silently dropping it
// would reinstate the bug, so a missing solver WARNS.
void RadiationExchangeManager::setSolarSourceAlbedo(const std::vector<double>& source_albedo) {
    if (!solver_) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "  [radiation.PTM-106] WARNING: setSolarSourceAlbedo() called before the "
                         "radiosity solver exists; the incidence-dependent visible SOURCE albedo "
                         "was DROPPED for this call (static base albedo in force)." << std::endl;
        }
        return;
    }
    solver_->setSolarSourceAlbedo(source_albedo);
}

void RadiationExchangeManager::setSolarSourceAlbedoBatch(
    const std::vector<std::vector<double>>& source_albedo) {
    if (!solver_) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "  [radiation.PTM-106] WARNING: setSolarSourceAlbedoBatch() called before "
                         "the radiosity solver exists; DROPPED (static base albedo in force)."
                      << std::endl;
        }
        return;
    }
    solver_->setSolarSourceAlbedoBatch(source_albedo);
}

void RadiationExchangeManager::enableTwoBand() {
    if (solver_) solver_->enableTwoBand();
}

bool RadiationExchangeManager::isTwoBand() const {
    return solver_ ? solver_->isTwoBand() : false;
}

std::vector<std::vector<double>>
RadiationExchangeManager::computeIndirectFluxBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep) {
    // Delegate to the Sun-aware overload with no Sun directions (beaming off).
    return computeIndirectFluxBatch(temperatures_per_timestep,
                                    solar_flux_per_timestep, {});
}

std::vector<std::vector<double>>
RadiationExchangeManager::computeIndirectFluxBatch(
    const std::vector<std::vector<double>>& temperatures_per_timestep,
    const std::vector<std::vector<double>>& solar_flux_per_timestep,
    const std::vector<std::array<double, 3>>& sun_dirs) {

    const size_t n_steps = temperatures_per_timestep.size();
    const size_t n = mesh_->faces.size();
    std::vector<std::vector<double>> indirect_per_timestep(n_steps,
                                                            std::vector<double>(n, 0.0));

    if (!is_active_ || !solver_ || n_steps == 0) {
        return indirect_per_timestep;
    }

    auto raw_per_step = solver_->solveBatch(temperatures_per_timestep,
                                             solar_flux_per_timestep);

    const bool do_beam = beam_active_ && (sun_dirs.size() == n_steps);

    // Per-band batch (two-band pool path): the solver captured absorbed IR and
    // visible per timestep. Map them to exchange faces, mirroring the summed
    // mapping. NO beaming is applied to the bands (matches the single-GPU
    // solveGPUTwoBand bands, which are pure radiosity). Empty on the gray path.
    const auto& ir_raw  = solver_->lastIndirectIRBatch();
    const auto& vis_raw = solver_->lastIndirectVisibleBatch();
    const bool have_band_batch = (ir_raw.size() == n_steps && vis_raw.size() == n_steps);
    if (have_band_batch) {
        indirect_ir_batch_.assign(n_steps, std::vector<double>(n, 0.0));
        indirect_vis_batch_.assign(n_steps, std::vector<double>(n, 0.0));
    } else {
        indirect_ir_batch_.clear();
        indirect_vis_batch_.clear();
    }

    // Map back to all faces using the exchange-face index list (same as
    // computeIndirectFlux). Update aggregate stats from the last timestep.
    double sum = 0.0;
    double max_flux = 0.0;
    for (size_t k = 0; k < n_steps; ++k) {
        const auto& flux = raw_per_step[k];
        if (flux.empty()) continue;
        // PTM-98: full write-back (see computeIndirectFlux).
        writeBackFlux(flux, indirect_per_timestep[k]);
        if (have_band_batch && !ir_raw[k].empty() && !vis_raw[k].empty()) {
            writeBackFlux(ir_raw[k],  indirect_ir_batch_[k]);
            writeBackFlux(vis_raw[k], indirect_vis_batch_[k]);
        }
        // Smith beamed-emission correction (per timestep, Sun-dependent).
        if (do_beam) {
            const auto& s = sun_dirs[k];
            if (s[0] != 0.0 || s[1] != 0.0 || s[2] != 0.0) {
                addBeamedFlux(temperatures_per_timestep[k], solar_flux_per_timestep[k],
                              s, indirect_per_timestep[k]);
            }
        }
    }
    if (!raw_per_step.empty() && !raw_per_step.back().empty()) {
        for (int idx : classification_.exchange_face_indices) {
            sum += indirect_per_timestep.back()[idx];
            max_flux = std::max(max_flux, indirect_per_timestep.back()[idx]);
        }
        stats_.mean_indirect_flux = sum / std::max(size_t(1),
                                                    classification_.exchange_faces);
        stats_.max_indirect_flux = max_flux;
        stats_.solver_iterations = solver_->getLastIterations();
        stats_.solver_residual = solver_->getLastResidual();
    }

    return indirect_per_timestep;
}

std::vector<double> RadiationExchangeManager::computeIndirectFlux(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux) {
    // Delegate to the Sun-aware overload with no Sun direction (beaming off).
    return computeIndirectFlux(temperatures, solar_flux, {0.0, 0.0, 0.0});
}

std::vector<double> RadiationExchangeManager::computeIndirectFlux(
    const std::vector<double>& temperatures,
    const std::vector<double>& solar_flux,
    const std::array<double, 3>& sun_dir) {

    size_t n = mesh_->faces.size();
    indirect_flux_.assign(n, 0.0);

    if (!is_active_ || !solver_) {
        return indirect_flux_;
    }

    // Solve radiosity equation (isotropic/Lambertian thermal emission)
    auto flux = solver_->solve(temperatures, solar_flux);

    // PTM-98: write the solve's result back in FULL. `flux` is already indexed by
    // global face id and covers every face; the legacy path copied only the
    // classified subset and left the rest at 0, discarding computed flux.
    // `flux` is ours, so the total path steals its buffer rather than copying.
    writeBackFlux(std::move(flux), indirect_flux_);

    // C1: per-band absorbed indirect flux (reflected-solar vis / thermal-IR),
    // mapped to all faces. Only populated when the solver ran two-band.
    if (solver_->isTwoBand()) {
        const auto& qvis = solver_->lastIndirectVisible();
        const auto& qir  = solver_->lastIndirectIR();
        indirect_flux_vis_.assign(n, 0.0);
        indirect_flux_ir_.assign(n, 0.0);
        writeBackFlux(qvis, indirect_flux_vis_);
        writeBackFlux(qir,  indirect_flux_ir_);
    } else {
        indirect_flux_vis_.clear();
        indirect_flux_ir_.clear();
    }

    // Smith (1967) anisotropic beamed-emission correction (Sun-dependent).
    // Redistributes each sunlit wall's fixed total emission toward the floor.
    const bool sun_valid = (sun_dir[0] != 0.0 || sun_dir[1] != 0.0 || sun_dir[2] != 0.0);
    if (beam_active_ && sun_valid) {
        addBeamedFlux(temperatures, solar_flux, sun_dir, indirect_flux_);
    }

    // Update statistics (after beaming so they reflect the corrected flux)
    double sum = 0.0;
    double max_flux = 0.0;
    for (int idx : classification_.exchange_face_indices) {
        sum += indirect_flux_[idx];
        max_flux = std::max(max_flux, indirect_flux_[idx]);
    }

    stats_.mean_indirect_flux = sum / std::max(size_t(1), classification_.exchange_faces);
    stats_.max_indirect_flux = max_flux;
    stats_.solver_iterations = solver_->getLastIterations();
    stats_.solver_residual = solver_->getLastResidual();

    return indirect_flux_;
}

// ============================================================================
// Smith (1967) anisotropic beamed thermal emission
// The directional-emission math lives in smith_beaming.h (shared with the
// standalone validator test_beam_chi.cpp).
// ============================================================================

void RadiationExchangeManager::buildChiTable() {
    chi_ns_ = 24; chi_ne_ = 24; chi_npsi_ = 33;
    chi_table_ = smith::buildChiTable(chi_w_rad_, chi_ns_, chi_ne_, chi_npsi_);
}

float RadiationExchangeManager::lookupChi(double cos_sun, double cos_e,
                                          double cos_psi) const {
    return smith::evalChi(chi_table_, chi_ns_, chi_ne_, chi_npsi_,
                          cos_sun, cos_e, cos_psi);
}

void RadiationExchangeManager::buildBeamPairs() {
    beam_pairs_.clear();
    if (classification_.exchange_face_indices.empty()) return;

    // Receivers = the shadowed-floor faces. PTM-98: this is deliberately NOT
    // classification_.exchange_face_indices any more. That set is now every face
    // by default, and the Smith beamed-emission correction is specifically a
    // sunlit-wall -> shadowed-floor redistribution: widening the receiver set
    // would both change the physics and make buildBeamPairs recompute view
    // factors for the whole mesh. Identical to the old list when
    // exchange_selection == Shadow.
    //
    // KNOWN RESIDUAL (PTM-101): this receiver set, and the shadow[j] >= 0.99
    // emitter test below, are still measured over THIS window, so a beaming run
    // (rms_slope_deg > 0) remains chunk-decomposition dependent. Beaming is off
    // by default; the warning is emitted in prepare().
    // No reserve hint: since PTM-98 the exchange list is the whole mesh, so it is
    // no longer an estimate of the receiver count. Built once per run.
    beam_receivers_.clear();
    for (size_t i = 0; i < classification_.shadow_fractions.size(); ++i) {
        if (classification_.shadow_fractions[i] >= config_.min_shadow_fraction) {
            beam_receivers_.push_back(static_cast<int>(i));
        }
    }
    if (beam_receivers_.empty()) return;
    const std::vector<int>& receivers = beam_receivers_;

    // Geometric VF + Embree visibility for receiver rows. Backend-independent
    // (works for H-matrix runs where per-row matrix access is unavailable), and
    // uses the same F formula and row-normalization as the radiosity.
    ViewFactorCalculator calc(mesh_, static_cast<RTCDevice>(device_),
                              static_cast<RTCScene>(scene_), config_);
    ViewFactorMatrix rows = calc.computeSubset(receivers);  // global row indexing

    // Cache per-face mean normals (float) for the run-time beamed pass.
    const auto& normals = calc.getFaceNormals();
    const auto& centers = calc.getFaceCenters();
    beam_normals_.resize(normals.size());
    for (size_t f = 0; f < normals.size(); ++f) {
        beam_normals_[f] = { static_cast<float>(normals[f][0]),
                             static_cast<float>(normals[f][1]),
                             static_cast<float>(normals[f][2]) };
    }

    const auto& shadow = classification_.shadow_fractions;
    const double thresh = config_.beam_vf_threshold;

    std::vector<int> cols;
    std::vector<double> vals;
    for (int i : receivers) {
        rows.getRow(i, cols, vals);     // global receiver index
        if (cols.empty()) continue;
        double rowsum = 0.0;
        for (double v : vals) rowsum += v;
        double rownorm = 1.0 / std::max(1.0, rowsum);   // matches radiosity vf_row_norm
        const auto& ci = centers[i];
        for (size_t m = 0; m < cols.size(); ++m) {
            int j = cols[m];
            if (j < 0 || j >= static_cast<int>(shadow.size())) continue;
            if (shadow[j] >= 0.99) continue;            // permanently shadowed -> not an emitter
            double Fhat = vals[m] * rownorm;
            if (Fhat <= thresh) continue;
            const auto& cj = centers[j];
            double dx = ci[0] - cj[0], dy = ci[1] - cj[1], dz = ci[2] - cj[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) continue;
            double inv = 1.0 / std::sqrt(r2);
            double dirx = dx * inv, diry = dy * inv, dirz = dz * inv;
            const auto& nj = normals[j];
            double cos_e = nj[0] * dirx + nj[1] * diry + nj[2] * dirz;
            if (cos_e <= 0.0) continue;                 // wall not emitting toward floor
            BeamPair bp;
            bp.i = i;
            bp.j = j;
            bp.Fhat_ij = static_cast<float>(Fhat);
            bp.cos_e = static_cast<float>(cos_e);
            bp.dir_ji = { static_cast<float>(dirx),
                          static_cast<float>(diry),
                          static_cast<float>(dirz) };
            beam_pairs_.push_back(bp);
        }
    }

    std::cout << "  Smith beamed emission: " << beam_pairs_.size()
              << " wall->floor beam pairs (rms_slope = " << config_.rms_slope_deg
              << " deg)" << std::endl;
}

void RadiationExchangeManager::configureSolarBeaming() {
    beam_solar_ = false;
    solar_albedo_beam_ = 0.0;
    if (const char* bs = std::getenv("PTM_BEAM_SOLAR")) beam_solar_ = (std::atoi(bs) != 0);
    if (const char* sa = std::getenv("PTM_SOLAR_ALBEDO")) {
        double a = std::atof(sa);
        if (a > 0.0 && a < 1.0) solar_albedo_beam_ = a;
    }
    if (beam_solar_ && solar_albedo_beam_ <= 0.0) solar_albedo_beam_ = 0.12;  // default Bond
    if (beam_solar_) {
        std::cout << "  [PTM_BEAM_SOLAR] beaming reflected solar (Bond albedo "
                  << solar_albedo_beam_ << ") toward the floor, in addition to thermal IR"
                  << std::endl;
    }
}

void RadiationExchangeManager::addBeamedFlux(const std::vector<double>& temperatures,
                                             const std::vector<double>& solar_flux,
                                             const std::array<double, 3>& sun_dir,
                                             std::vector<double>& flux) const {
    // This is the ISOTHERMAL Smith-chi radiosity beaming. The anisothermal
    // sub-facet hook (radiation/anisothermal_beaming.h aniso::enhancementBol) is
    // DESIGNED but NOT wired here. HARD PREREQUISITE before wiring it (Fermi,
    // 2026-07-05): first divide E_bol by a per-cos_sun closure table
    // C(cos_sun) = int chi*E_bol cos dOmega / pi to close the ~+9%-at-grazing
    // energy leak -- REQUIRED for energy balance. Do NOT re-fold E with chi's
    // (m.s)(m.d) weight (Fermi: worse, +85% at grazing). See enhancementBol.
    if (beam_pairs_.empty()) return;
    const float sx = static_cast<float>(sun_dir[0]);
    const float sy = static_cast<float>(sun_dir[1]);
    const float sz = static_cast<float>(sun_dir[2]);

    // Accumulate the signed (chi-1) corrections, then apply once per receiver so
    // the result stays non-negative (absorbed flux). OpenMP-safe via atomic add
    // on the receiver bucket would race the final clamp, so reduce into a local.
    const size_t np = beam_pairs_.size();
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < np; ++k) {
        const BeamPair& bp = beam_pairs_[k];
        const std::array<float, 3>& nj = beam_normals_[bp.j];
        double cos_sun = static_cast<double>(nj[0] * sx + nj[1] * sy + nj[2] * sz);
        if (cos_sun <= 0.0) continue;   // wall mean in shadow this timestep -> no beaming
        double cos_psi = static_cast<double>(bp.dir_ji[0] * sx + bp.dir_ji[1] * sy
                                             + bp.dir_ji[2] * sz);
        double chi = lookupChi(cos_sun, static_cast<double>(bp.cos_e), cos_psi);
        double delta = chi - 1.0;
        if (delta == 0.0) continue;
        double Tj = temperatures[bp.j];
        double Ej = emissivity_[bp.j] * SIGMA_SB * Tj * Tj * Tj * Tj;   // thermal IR source
        // Solar opposition beaming (PTM_BEAM_SOLAR): the reflected-solar source
        // (albedo * incident sunlight) leaving the sunlit wall has the same m.s facet
        // structure as its IR emission, so the floor (backscatter) gets the same chi
        // enhancement. Absorbed at the floor with eps_i (consistent with the radiosity's
        // solar handling). Off by default => byte-identical IR-only beaming.
        double Sj = 0.0;
        if (beam_solar_ && bp.j < static_cast<int>(solar_flux.size())) {
            Sj = solar_albedo_beam_ * solar_flux[bp.j];
        }
        double dq = emissivity_[bp.i] * (Ej + Sj) * static_cast<double>(bp.Fhat_ij) * delta;
        #pragma omp atomic
        flux[bp.i] += dq;
    }

    // Keep absorbed indirect flux non-negative on the receiver set. PTM-98: the
    // beam receiver list, not the (now all-faces) exchange list — only receivers
    // can be driven negative by the beamed correction, and clamping the rest
    // would silently touch faces this pass never wrote.
    for (int i : beam_receivers_) {
        if (flux[i] < 0.0) flux[i] = 0.0;
    }
}

void RadiationExchangeManager::setEmissivity(double emissivity) {
    std::fill(emissivity_.begin(), emissivity_.end(), emissivity);
}

void RadiationExchangeManager::setEmissivity(const std::vector<double>& emissivity) {
    emissivity_ = emissivity;
}

const std::vector<double>& RadiationExchangeManager::operativeEmissivity() const {
    // Prefer the solver's copy: it is the one an override can have moved.
    static const std::vector<double> kEmpty;
    if (solver_) return solver_->operativeEmissivity();
    return emissivity_.empty() ? kEmpty : emissivity_;
}

void RadiationExchangeManager::setConfig(const RadiationConfig& config) {
    config_ = config;
    if (solver_) {
        solver_->setConfig(config_);
    }
}

bool RadiationExchangeManager::loadViewFactors(const std::string& filename) {
    // Try to detect file format from magic number
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (config_.verbose) {
            std::cout << "View factor file not found: " << filename << std::endl;
        }
        return false;
    }

    // Read magic number to detect format
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.close();

    bool success = false;

    // H-matrix format: magic = 0x484D4154 ("HMAT")
    // Sparse format: magic = 0x5646464D ("VFFM")
    if (magic == 0x484D4154) {
        // H-matrix format
        // For large meshes, use memory-mapped loading to avoid loading entire matrix into RAM.
        //
        // KNOWN BUG (2026-04): parallel matvec on a mmap-loaded H-matrix SIGSEGVs
        // during the first radiosity iteration on >1M-face meshes. Root cause
        // under investigation. Until it's fixed, auto-mmap is effectively
        // disabled by raising the threshold far above any expected mesh size.
        // Full-RAM load is safe and the matrix typically fits comfortably
        // (probe: 1.57M faces → 2.1 GB .hmat ≪ 384 GB RAM).
        size_t n_faces = mesh_->faces.size();
        const size_t MMAP_THRESHOLD_FACES = 1'000'000'000;  // effectively off

        bool use_mmap = (n_faces >= MMAP_THRESHOLD_FACES);

        if (use_mmap) {
            std::cout << "Large mesh detected (" << n_faces << " faces)" << std::endl;
            std::cout << "  Using memory-mapped H-matrix loading to minimize RAM usage" << std::endl;
        }

        if (config_.verbose) {
            std::cout << "Loading H-matrix view factors from: " << filename
                      << (use_mmap ? " (mmap)" : " (RAM)") << std::endl;
        }

        hmatrix_view_factors_ = std::make_unique<HMatrixViewFactor>();

        if (use_mmap) {
            success = hmatrix_view_factors_->readBinaryMmap(filename);
        } else {
            success = hmatrix_view_factors_->readBinary(filename);
        }

        if (success) {
            using_hmatrix_ = true;
            config_.use_hmatrix = true;  // Ensure config reflects loaded format

            if (config_.verbose) {
                std::cout << "Loaded H-matrix view factors:" << std::endl;
                std::cout << "  Size: " << hmatrix_view_factors_->numRows() << " x "
                          << hmatrix_view_factors_->numCols() << std::endl;
                std::cout << "  Memory: " << hmatrix_view_factors_->memoryBytes() / (1024.0 * 1024.0)
                          << " MB" << (use_mmap ? " (index only, data on disk)" : "") << std::endl;
                if (use_mmap) {
                    std::cout << "  Mode: Memory-mapped (on-demand block access)" << std::endl;
                }
                hmatrix_view_factors_->printStatistics();
            }
        } else {
            hmatrix_view_factors_.reset();
            std::cerr << "Error: Failed to load H-matrix view factors from: " << filename << std::endl;
        }
    } else if (magic == 0x5646464D) {
        // Sparse format
        if (config_.verbose) {
            std::cout << "Loading sparse view factors from: " << filename << std::endl;
        }

        success = view_factors_.readBinary(filename);

        if (success) {
            using_hmatrix_ = false;
            config_.use_hmatrix = false;

            if (config_.verbose) {
                std::cout << "Loaded sparse view factors:" << std::endl;
                std::cout << "  Size: " << view_factors_.numRows() << " x "
                          << view_factors_.numCols() << std::endl;
                std::cout << "  Non-zeros: " << view_factors_.numNonZeros() << std::endl;
                std::cout << "  Memory: " << view_factors_.memoryUsage() / (1024.0 * 1024.0)
                          << " MB" << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to load sparse view factors from: " << filename << std::endl;
        }
    } else {
        std::cerr << "Error: Unknown view factor file format (magic: 0x"
                  << std::hex << magic << std::dec << ")" << std::endl;
        return false;
    }

    if (success) {
        view_factors_loaded_ = true;

        // Verify matrix dimensions match mesh
        size_t n_faces = mesh_->faces.size();
        size_t matrix_size = using_hmatrix_ ? hmatrix_view_factors_->numRows() : view_factors_.numRows();

        if (matrix_size != n_faces) {
            std::cerr << "Warning: View factor matrix size (" << matrix_size
                      << ") does not match mesh face count (" << n_faces << ")" << std::endl;
            std::cerr << "  View factors will be recomputed." << std::endl;
            view_factors_loaded_ = false;
            hmatrix_view_factors_.reset();
            return false;
        }
    }

    return success;
}

} // namespace radiation
