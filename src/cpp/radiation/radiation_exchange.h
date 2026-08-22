#pragma once

#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <string>

#include "radiation_config.h"
#include "view_factor.h"
#include "hmatrix_view_factor.h"
#include "../mesh/mesh_utils.h"
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

// Forward declaration for GPU radiosity solver
#ifdef PTM_GPU_OPENCL
namespace gpu {
namespace opencl {
    class RadiosityOpenCLSolver;
    class RadiosityGPUDevicePool;
}
}
#endif

namespace radiation {

// Stefan-Boltzmann constant [W m^-2 K^-4]
constexpr double SIGMA_SB = ptm::kStefanBoltzmann;

// Resolve how many timesteps the multi-GPU radiosity pool dispatches per batch.
//
// `configured` is radiation.gpu_batch_size (0 = auto). Auto now means ONE
// DISPATCH PER SWEEP, capped only by the host memory the batch buffers need:
// the old auto of 4 x n_devices split a sweep into dozens of std::async
// round-trips and measurably cost ~30% of the pool's speedup (1.37x measured at
// batch 8 vs ~1.94x at whole-sweep, 2 devices, 472-timestep crater).
//
// The cap exists because the batched path holds ~6 vectors of n_faces doubles
// per timestep in HOST memory (temperatures, solar, source albedo, indirect, and
// the two per-band batches). At 509,337 faces that is ~24 MB per timestep, so a
// several-hundred-timestep sweep would otherwise reserve many GB. The budget
// defaults to 4 GB and is overridable with PTM_RAD_POOL_BATCH_BUDGET_MB.
//
// Never returns less than 4 x n_devices (the previous default), so this cannot
// make any run slower than it was, and never more than n_steps.
//
// @param announce  print the resolved value and the reason (once per sweep).
int resolvePoolBatchSize(int configured, int n_steps, int n_devices,
                         size_t n_faces, bool announce);

// Abstract interface for view factor matrix-vector product
// Allows RadiositySolver to work with either ViewFactorMatrix or HMatrixViewFactor
class IViewFactorOperator {
public:
    virtual ~IViewFactorOperator() = default;
    virtual void matvec(const std::vector<double>& x, std::vector<double>& y) const = 0;
    // Fused width-2 matvec: y1 = F·x1, y2 = F·x2 in ONE structural traversal (each
    // row's indices/values are read once, feeding two dot products) -- halves the
    // operator's memory-bandwidth reads for the two-band radiosity while both bands
    // iterate. Default = two separate matvecs, so it is bit-identical for every
    // operator; SparseVFOperator overrides it with the single-traversal kernel.
    virtual void matvec2(const std::vector<double>& x1, const std::vector<double>& x2,
                         std::vector<double>& y1, std::vector<double>& y2) const {
        matvec(x1, y1);
        matvec(x2, y2);
    }
    virtual size_t numRows() const = 0;
    virtual size_t numCols() const = 0;
    virtual void getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const = 0;
};

// Wrapper for ViewFactorMatrix
class ViewFactorOperator : public IViewFactorOperator {
public:
    explicit ViewFactorOperator(const ViewFactorMatrix& F) : F_(F) {}
    void matvec(const std::vector<double>& x, std::vector<double>& y) const override {
        F_.matvec(x, y);
    }
    size_t numRows() const override { return F_.numRows(); }
    size_t numCols() const override { return F_.numCols(); }
    void getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const override {
        F_.getRow(row, cols, vals);
    }
private:
    const ViewFactorMatrix& F_;
};

// Wrapper for HMatrixViewFactor
class HMatrixOperator : public IViewFactorOperator {
public:
    explicit HMatrixOperator(const HMatrixViewFactor& H) : H_(H) {}
    void matvec(const std::vector<double>& x, std::vector<double>& y) const override {
        H_.matvec(x, y);
    }
    size_t numRows() const override { return H_.numRows(); }
    size_t numCols() const override { return H_.numCols(); }
    void getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const override {
        // H-matrix doesn't support row access directly; use Jacobi only
        cols.clear();
        vals.clear();
    }
private:
    const HMatrixViewFactor& H_;
};

// PTM-12 row-sum correction wrapper. Wraps any base VF operator and applies the
// symmetric diagonal congruence y = D * (F * (D * x)), D = diag(d_i), where d_i is
// chosen (Sinkhorn, at construction) so the corrected row sums Sum_j (d_i F_ij d_j)
// match an occlusion-aware referee's true row sums. The symmetric congruence PRESERVES
// reciprocity (A_i F_ij d_i d_j == A_j F_ji d_j d_i because the stored F is reciprocal)
// and restores F_sky = 1 - Sum_j F_ij > 0, curing BOTH the near-field M1 over-weight
// (PTM-12) and the Sum F > 1 renorm runaway. Installed ONLY when env PTM_HMATRIX_ROWFIX
// is set; default OFF => this wrapper never exists => byte-identical legacy behavior.
class CorrectedRowFixOperator : public IViewFactorOperator {
public:
    CorrectedRowFixOperator(std::unique_ptr<IViewFactorOperator> base,
                            std::vector<double> d)
        : base_(std::move(base)), d_(std::move(d)) {}
    void matvec(const std::vector<double>& x, std::vector<double>& y) const override {
        const size_t n = x.size();
        xs_.resize(n);
        for (size_t i = 0; i < n; ++i) xs_[i] = (i < d_.size() ? d_[i] : 1.0) * x[i];
        base_->matvec(xs_, y);
        for (size_t i = 0; i < n && i < d_.size(); ++i) y[i] *= d_[i];
    }
    size_t numRows() const override { return base_->numRows(); }
    size_t numCols() const override { return base_->numCols(); }
    void getRow(int, std::vector<int>& cols, std::vector<double>& vals) const override {
        cols.clear();  // congruence does not expose sparse rows; Jacobi/matvec path only
        vals.clear();
    }
    const std::vector<double>& scale() const { return d_; }
private:
    std::unique_ptr<IViewFactorOperator> base_;
    std::vector<double> d_;
    mutable std::vector<double> xs_;
};

// PTM-12 per-partner corrected VF (full replacement). A SYMMETRIC sparse (CSR) view-factor
// matrix built from the occlusion-exact referee (raw-winding hemisphere MC, per-partner F),
// symmetrized for EXACT reciprocity (A_i F_ij = A_j F_ji by construction) and capped to
// ΣF <= 1. REPLACES the stored H-matrix operator, fixing BOTH the near-field over-count
// (walls) AND the floor under-coupling (missing floor->wall IR) in one consistent matrix
// that per-row scaling could not. Installed only when env PTM_HMATRIX_VF_REPLACE is set;
// default OFF => never constructed => byte-identical.
class SparseVFOperator : public IViewFactorOperator {
public:
    SparseVFOperator(size_t n, std::vector<int64_t> indptr,
                     std::vector<int32_t> cols, std::vector<float> data)
        : n_(n), indptr_(std::move(indptr)), cols_(std::move(cols)), data_(std::move(data)) {}
    void matvec(const std::vector<double>& x, std::vector<double>& y) const override {
        // Every y[i] is fully overwritten below, so the old y.assign(n_,0.0) was a
        // dead full-vector memset; resize() only guarantees the size (no-op when the
        // caller pre-sized y). schedule(guided) balances the highly variable per-row
        // nnz (floor faces couple to ~1000s of partners, rims to a handful) that
        // schedule(static) left imbalanced. Both are bit-identical: each y[i] is the
        // same in-order row sum regardless of which thread computes it.
        y.resize(n_);
        #pragma omp parallel for schedule(guided)
        for (long i = 0; i < static_cast<long>(n_); ++i) {
            double s = 0.0;
            const int64_t e = indptr_[i + 1];
            for (int64_t k = indptr_[i]; k < e; ++k) s += static_cast<double>(data_[k]) * x[cols_[k]];
            y[i] = s;
        }
    }
    // Fused width-2 traversal: read each row's (col,val) once, accumulate two dot
    // products. Byte-identical to two matvec() calls -- per-row sums are the same
    // in-order accumulation; only the CSR bandwidth read is halved.
    void matvec2(const std::vector<double>& x1, const std::vector<double>& x2,
                 std::vector<double>& y1, std::vector<double>& y2) const override {
        y1.resize(n_); y2.resize(n_);
        #pragma omp parallel for schedule(guided)
        for (long i = 0; i < static_cast<long>(n_); ++i) {
            double s1 = 0.0, s2 = 0.0;
            const int64_t e = indptr_[i + 1];
            for (int64_t k = indptr_[i]; k < e; ++k) {
                const double v = static_cast<double>(data_[k]);
                const int32_t c = cols_[k];
                s1 += v * x1[c];
                s2 += v * x2[c];
            }
            y1[i] = s1; y2[i] = s2;
        }
    }
    size_t numRows() const override { return n_; }
    size_t numCols() const override { return n_; }
    void getRow(int, std::vector<int>& c, std::vector<double>& v) const override { c.clear(); v.clear(); }
    // The stored CSR, so initGPU() can promote this operator to a GPU
    // ViewFactorMatrix without re-reading the file or keeping a second copy
    // alive from load time. Read-only: the operator stays the owner and stays
    // installed, which is what makes the GPU's CPU fallback correct by
    // construction rather than by repair.
    const std::vector<int64_t>& rowPointers() const { return indptr_; }
    const std::vector<int32_t>& colIndices() const { return cols_; }
    const std::vector<float>&   values() const { return data_; }

private:
    size_t n_;
    std::vector<int64_t> indptr_;
    std::vector<int32_t> cols_;
    std::vector<float>   data_;
};

// Result of a power iteration on the loaded view-factor operator F.
// ADVISORY ONLY -- nothing in the solve consumes it. Linearizing the outer
// radiation<->thermal Picard loop gives M = G·ε·F·(I−ρ_r F)^-1·D with
// D = diag(4εσT³) and G ≤ D^-1, so M is similar to ε·F(I−ρ_r F)^-1: the T³
// factors cancel and the asymptotic sweep-to-sweep rate is bounded by a purely
// geometric number, r ≤ ε·ρ(F)/(1−(1−ε)·ρ(F)). Validated at two meshes to three
// digits (509k lunar flagship ρ=0.7766 → r=0.7676, measured 0.745; 12.8k 80 N
// crater ρ≈0.081 → r=0.0773, measured 0.077–0.128).
struct SpectralRadiusEstimate {
    bool   ok         = false;  // false = no operator, or a degenerate/zero F
    double rho        = 0.0;    // Rayleigh quotient at the last iterate
    int    iterations = 0;      // power iterations actually performed
    double seconds    = 0.0;    // wall clock
    bool   converged  = false;  // relative Rayleigh change fell below rel_tol
    size_t n          = 0;      // operator dimension
};

// Radiosity solver for thermal radiation exchange
// Solves: B = ε σ T^4 + ρ F B + ρ Q_solar
// Where B is radiosity, ε is emissivity, ρ = 1-ε is reflectivity,
// F is the view factor matrix, and Q_solar is incident solar flux
class RadiositySolver {
public:
    // Constructor with view factor matrix and face properties
    RadiositySolver(const ViewFactorMatrix& F,
                    const std::vector<double>& emissivity,
                    const std::vector<double>& areas);

    // Constructor with H-matrix view factor
    RadiositySolver(const HMatrixViewFactor& H,
                    const std::vector<double>& emissivity,
                    const std::vector<double>& areas);

    // Explicit destructor: required because unique_ptr<RadiosityOpenCLSolver>
    // and unique_ptr<RadiosityGPUDevicePool> are forward-declared, so the
    // implicit destructor would be instantiated at every dependent's
    // translation unit (e.g. test_bowl_crater.cpp). Defining it in the .cpp
    // keeps the GPU types out of the public header.
    ~RadiositySolver();

    // Set configuration
    void setConfig(const RadiationConfig& config) { config_ = config; }

    // Solve for indirect radiation flux at each face
    // temperatures: current surface temperatures [K]
    // solar_flux: direct solar flux at each face [W/m^2]
    // Returns: total absorbed indirect flux at each face [W/m^2]
    std::vector<double> solve(const std::vector<double>& temperatures,
                              const std::vector<double>& solar_flux);

    // Batched solve: each row is one timestep's (T, Q_solar). Returns one row
    // of indirect flux per input timestep. When gpu_devices > 1 and the GPU
    // pool was successfully initialized, the batch is partitioned across the
    // pool's devices and solved concurrently. Otherwise falls back to a
    // serial loop over single-timestep solve() — semantically identical to
    // the caller running its own per-timestep loop.
    std::vector<std::vector<double>> solveBatch(
        const std::vector<std::vector<double>>& temperatures_per_timestep,
        const std::vector<std::vector<double>>& solar_flux_per_timestep);

    // True iff the GPU pool was successfully initialized (gpu_devices > 1).
    bool hasGPUPool() const;

    // Devices in the ACTIVE radiosity pool, 0 when no pool is active. This is
    // the only honest source for "how many GPUs is radiosity using": the
    // configured radiation.gpu_devices ignores gpu_device_ids, the degrade
    // ladder, and the placement policy, and the dispatch announce used to
    // print it as if it were fact.
    int gpuPoolDeviceCount() const;

    // Power-iterate the view-factor operator the SOLVE applies -- that is
    // diag(vf_row_norm_)·F, not raw F, because solveJacobi and every GPU/
    // two-band sibling renormalize each row of F·x by 1/max(1, Σ_j F_ij)
    // immediately after the matvec. Identical to ρ(F) whenever no row sums
    // above 1, which is the usual case; on a mesh with over-unity rows, raw F
    // would over-state ρ and could claim "not a contraction" about a solve that
    // contracts.
    // ADVISORY ONLY: read-only on F, allocates two n-vectors, and touches no
    // solver state, so it cannot change a result. F is entrywise non-negative,
    // so by Perron-Frobenius a non-negative start converges to the dominant
    // eigenpair; the start vector is drawn from a FIXED seed so the reported
    // number is reproducible run to run. Cost is one matvec per iteration --
    // measured 3.0 s for 60 iterations at 495M nnz on this machine's 48 cores,
    // i.e. well under 1% of ONE coupling sweep on that mesh.
    SpectralRadiusEstimate estimateSpectralRadius(int max_iterations = 60,
                                                  double rel_tol = 1e-6,
                                                  uint64_t seed = 0) const;

    // True iff a GPU two-band path is active after initGPU() — either the
    // single-GPU solveGPUTwoBand or the multi-GPU pool's two-band solveBatch.
    // Lets a test confirm the GPU path actually engaged rather than silently
    // falling back to CPU. Before the pool set use_gpu_/two_band_gpu_ok_ this
    // reported false on an active pool.
    bool isGPUTwoBand() const {
#ifdef PTM_GPU_OPENCL
        return two_band_gpu_ok_ && use_gpu_;
#else
        return false;
#endif
    }

    // Compute thermal emission from faces (ε σ T^4)
    std::vector<double> computeThermalEmission(const std::vector<double>& temperatures) const;

    // Compute scattered solar radiation (single bounce)
    std::vector<double> computeScatteredSolar(const std::vector<double>& solar_flux) const;

    // True iff band-separated (visible + thermal-IR) radiosity is active
    // (PTM_RAD_TWO_BAND set and a Bond albedo supplied via PTM_SOLAR_ALBEDO).
    bool isTwoBand() const { return two_band_; }

    // Set the per-face visible Bond albedo A_i used by the reflected-solar source
    // and the two-band visible channel (scatter A_i / absorb 1-A_i). Replaces the
    // uniform env scalar so A_rad == A_BC per face, and is TIME-UPDATABLE: call it
    // before each solve() when the boundary condition uses an incidence-dependent
    // albedo (Hayne/Hapke), so the reflected-solar energy the enclosure receives
    // matches the (1-A_i) the surface BC absorbs (no per-face energy leak). The
    // size must equal the face count; a mismatch warns and is ignored. See A2 in
    // TICKET-two-band-radiosity-production.md and docs/two-band-radiosity-derivation.tex.
    void setSolarAlbedo(const std::vector<double>& solar_albedo);

    // PTM-106: per-face DIRECT-BEAM source albedo A_src,i for the visible band.
    //
    // The visible band has TWO distinct reflectances and they are not the same
    // physical quantity:
    //   (1) SOURCE      -- the fraction of the collimated DIRECT solar beam that
    //                      face i scatters into the enclosure. The surface BC
    //                      absorbs (1 - A(theta_i)) Q_i, so by construction the
    //                      enclosure must receive A(theta_i) Q_i. When the BC uses
    //                      an incidence-dependent albedo (Hayne 2017 / Hapke),
    //                      A(theta_i) is per-face AND per-timestep.
    //   (2) OPERATOR    -- the fraction of DIFFUSE inter-face flux that face i
    //                      re-scatters, and (1 - A) the fraction it absorbs. Diffuse
    //                      incidence has no single theta, so the hemispheric/Bond
    //                      value (solar_reflectivity_, set by setSolarAlbedo) is the
    //                      correct reflectance here and is left UNCHANGED.
    //
    // Before PTM-106 the source used (2) -- the static base albedo -- so at grazing
    // incidence a Hayne body scattered with 0.12 while its BC absorbed with ~0.60:
    // (A(theta) - A0) Q of reflected sunlight vanished from the budget instead of
    // reaching the shadowed floors. Call this per timestep with the SAME albedo the
    // BC used. Empty (the default, or after clearSolarSourceAlbedo()) => the source
    // falls back to solar_reflectivity_, i.e. the exact legacy behavior.
    void setSolarSourceAlbedo(const std::vector<double>& source_albedo);

    // Per-timestep source albedo for solveBatch() (multi-GPU pool path), indexed
    // [timestep][face]. Must have the same outer length as the batch handed to
    // solveBatch(); a mismatch WARNS and is ignored (never silently mis-pairs).
    void setSolarSourceAlbedoBatch(const std::vector<std::vector<double>>& source_albedo);

    // Drop any source-albedo override (restores the legacy static-albedo source).
    void clearSolarSourceAlbedo();

    // True iff a direct-beam source albedo is currently installed (single-timestep
    // or batch). Public observable for the banner / regression tests.
    bool hasSolarSourceAlbedo() const {
        return !solar_albedo_source_.empty() || !solar_albedo_source_batch_.empty();
    }

    // Activate band-separated (two-band) radiosity from config (D1), independent of
    // the PTM_RAD_TWO_BAND env. Requires a per-face Bond albedo already supplied via
    // setSolarAlbedo(); if none is set this WARNS and stays gray (never invents one).
    void enableTwoBand();

    // Band-separated absorbed indirect fluxes from the last two-band solve()
    // (Paul's new product). Empty in the legacy gray path. lastIndirect() (the
    // solve() return) is their sum; these expose each band separately:
    //   Q_ind^vis_i = (1-A_i) Σ_j F_ij B^vis_j   (multiply-scattered solar)
    //   Q_ind^ir_i  = ε_i     Σ_j F_ij B^ir_j    (thermal IR, unchanged)
    const std::vector<double>& lastIndirectVisible() const { return last_indirect_vis_; }
    const std::vector<double>& lastIndirectIR() const { return last_indirect_ir_; }

    // Per-band batch results from the last two-band solveBatch() (multi-GPU pool
    // path), indexed [timestep][face]. Empty on the gray path and on the
    // single-timestep two-band path (use lastIndirect{Visible,IR}() there).
    const std::vector<std::vector<double>>& lastIndirectVisibleBatch() const { return last_indirect_vis_batch_; }
    const std::vector<std::vector<double>>& lastIndirectIRBatch() const { return last_indirect_ir_batch_; }

    // Get convergence information from last solve
    int getLastIterations() const { return last_iterations_; }
    // Per-band sweep counts from the last two-band solve (Fermi's measurement).
    int getLastIterationsIR() const { return last_iterations_ir_; }
    int getLastIterationsVis() const { return last_iterations_vis_; }
    // CSR traversals (matvecs) the last solve actually dispatched. This is the
    // measured cost of a sweep and it is NOT the iteration count: a solve also
    // pays a final matvec to turn the converged radiosity into absorbed flux, and
    // the fused two-band routes (CPU and GPU) share one traversal between the
    // bands while both iterate.
    int getLastCsrTraversals() const { return last_csr_traversals_; }
    double getLastResidual() const { return last_residual_; }
    bool didConverge() const { return converged_; }

    // Compute energy balance check
    // Returns: [total_emission, total_absorption, imbalance]
    std::array<double, 3> checkEnergyBalance(const std::vector<double>& temperatures,
                                              const std::vector<double>& solar_flux,
                                              const std::vector<double>& indirect_flux) const;

    // Initialize GPU solver if enabled (call after setConfig)
    void initGPU();

    // Promote a PTM_HMATRIX_VF_REPLACE matrix from its CPU SparseVFOperator to a
    // GPU-capable ViewFactorMatrix (sets F_ptr_ for the upload). Called from
    // initGPU(), which is the FIRST point at which config_ holds the user's
    // settings -- initCommon() runs from the constructor, before setConfig().
    // The CPU operator stays installed throughout, so declining the GPU needs no
    // repair; releaseVfReplaceGpuOperator() just drops F_ptr_ and the float64
    // replica. Both are no-ops on every other path.
    void promoteVfReplaceToGpuOperator();
    void releaseVfReplaceGpuOperator(const char* why);

    // The emissivity actually in force, i.e. after initCommon applied any
    // PTM_RAD_EMISSIVITY override. reflectivity_ is exactly 1 - this.
    const std::vector<double>& operativeEmissivity() const { return emissivity_; }

private:
    // Initialize common data
    void initCommon(const std::vector<double>& emissivity,
                    const std::vector<double>& areas);

    // Jacobi iteration solver. scatter_refl is the per-face reflectance applied to
    // the multiple-scatter term ρ·F·B: reflectivity_ (=1-ε) for the gray/IR band,
    // the Bond albedo A for the two-band visible channel. Passing it explicitly
    // (rather than reading reflectivity_) lets the same kernel drive both bands;
    // the gray path passes reflectivity_ so its arithmetic is byte-identical.
    std::vector<double> solveJacobi(const std::vector<double>& emission,
                                    const std::vector<double>& solar_source,
                                    const std::vector<double>& scatter_refl);

    // Fused two-band Jacobi (Tier 2): runs the IR band (source=emission, refl=
    // reflectivity_) and visible band (source=vis_source, refl=solar_reflectivity_)
    // together, sharing ONE width-2 matvec (matvec2) per iteration while both are
    // unconverged, and FREEZING each band the instant it converges so each runs
    // exactly its own sweep count. BYTE-IDENTICAL to two separate solveJacobi()
    // calls; only the shared CSR traversal is saved. Sets last_iterations_ir_/vis_.
    void solveTwoBandJacobiFused(const std::vector<double>& emission,
                                 const std::vector<double>& vis_source,
                                 std::vector<double>& B_ir, std::vector<double>& B_vis);

    // Gauss-Seidel iteration (faster convergence). scatter_refl as in solveJacobi.
    std::vector<double> solveGaussSeidel(const std::vector<double>& emission,
                                         const std::vector<double>& solar_source,
                                         const std::vector<double>& scatter_refl);

    // Band-separated (visible + thermal-IR) radiosity solve. See the derivation in
    // docs/two-band-radiosity-derivation.tex, Eqs (4)-(6). Reuses the SAME view-
    // factor operator and the SAME Jacobi/GS kernel; the visible channel is
    // T-independent and is cached across Picard sweeps at fixed solar_flux.
    std::vector<double> solveTwoBand(const std::vector<double>& temperatures,
                                     const std::vector<double>& solar_flux);

#ifdef PTM_GPU_OPENCL
    // GPU-accelerated solve
    std::vector<double> solveGPU(const std::vector<double>& temperatures,
                                 const std::vector<double>& solar_flux);

    // GPU band-separated (two-band) solve (B1). Mirrors solveTwoBand() on the GPU by
    // running the verified gray kernels twice: IR band (solar=0, ε/1-ε) + visible band
    // (temps=0, properties ε←1-A / ρ←A, source A·Q), summed. Only reached when
    // two_band_gpu_ok_ (all Σ_j F_ij ≤ 1). Float32 => matches the CPU two-band to GPU
    // float precision, not bit-identically (same contract as the gray GPU path).
    std::vector<double> solveGPUTwoBand(const std::vector<double>& temperatures,
                                        const std::vector<double>& solar_flux);
#endif

    // Data members
    std::unique_ptr<IViewFactorOperator> F_op_;
    const ViewFactorMatrix* F_ptr_ = nullptr;  // For row access in Gauss-Seidel
    // Owns a ViewFactorMatrix built from a VF_REPLACE corrected CSR when routed
    // to the GPU pool (PTM_HMATRIX_VF_REPLACE_GPU); F_ptr_ points into it.
    std::unique_ptr<ViewFactorMatrix> owned_vf_;
    bool use_hmatrix_ = false;
    std::vector<double> emissivity_;
    std::vector<double> reflectivity_;  // 1 - emissivity (thermal reflectivity)
    // Solar Bond albedo for the reflected-SOLAR source term only (empty = fall back
    // to reflectivity_ = 1-eps, the legacy behavior). Set from PTM_SOLAR_ALBEDO so
    // a sunlit wall scatters ~0.12 of its insolation rather than ~0.05. See initCommon.
    // In two-band mode this same per-face value is the visible Bond albedo A for the
    // whole visible channel (scatter A, absorb 1-A), not just the first-bounce source.
    std::vector<double> solar_reflectivity_;
    // PTM-106: per-face DIRECT-BEAM source albedo A_src (see setSolarSourceAlbedo).
    // Empty => the source falls back to solar_reflectivity_ (legacy, byte-identical).
    std::vector<double> solar_albedo_source_;
    std::vector<std::vector<double>> solar_albedo_source_batch_;
    // PTM-106 GPU representability guard. The GPU kernel hard-wires
    //   solar_source[i] = reflectivity[i] * solar_flux[i],
    // so a source albedo different from the operator albedo is expressed by
    // pre-scaling the uploaded flux by A_src/A_op. That factor is undefined where
    // A_op -> 0. false => the GPU two-band path is REFUSED (loudly) and the exact
    // CPU two-band solve runs instead; never a silent wrong-albedo GPU solve.
    bool src_albedo_gpu_ok_ = true;
    void checkSourceAlbedoGPURepresentable();
    // Per-face visible-band SOURCE reflectance for a SINGLE-timestep solve: the
    // installed direct-beam albedo, else the operator albedo (legacy fallback --
    // returns solar_reflectivity_ ITSELF so callers can compare by address/value
    // and take a bitwise-legacy path). Deliberately does NOT consult the batch: a
    // single-timestep solve has no timestep index, and silently handing it
    // batch[0] would pair one timestep's albedo with another's flux.
    const std::vector<double>& visSourceAlbedoSingle() const {
        if (!solar_albedo_source_.empty()) return solar_albedo_source_;
        return solar_reflectivity_;
    }
    std::vector<double> areas_;

    // --- Band-separated (two-band) radiosity state -----------------------------
    // Enabled by PTM_RAD_TWO_BAND (requires PTM_SOLAR_ALBEDO for the Bond albedo A).
    // OFF by default => the legacy single gray-band path runs, byte-identical.
    bool two_band_ = false;
    // True iff the GPU is running the two-band solve (B1). The GPU Jacobi kernel
    // now applies vf_row_norm_ inside its iteration (uploadRowNorm), so it runs the
    // same operator ρ·(norm⊙F) as the CPU and a Σ_j F_ij > 1 row is no longer a
    // reason to refuse the GPU. What initGPU still checks is the actual divergence
    // criterion: the Jacobi gain ρ_max · max_i min(Σ_j F_ij, 1) must be < 1.
    bool two_band_gpu_ok_ = false;
    // Absorbed indirect flux from each band, from the last solveTwoBand() (the new
    // per-band product). Their sum is what solve() returns and the surface BC consumes.
    // Written per solve() call; read by the caller before the next call. solve() is
    // NOT re-entrant on a single instance (the multi-GPU pool uses one solver per
    // device); do not call concurrently on the same RadiositySolver.
    std::vector<double> last_indirect_vis_;
    std::vector<double> last_indirect_ir_;
    std::vector<std::vector<double>> last_indirect_vis_batch_;  // [step][face], pool two-band
    std::vector<std::vector<double>> last_indirect_ir_batch_;   // [step][face], pool two-band
    // Final normalized incoming (F·B·vf_row_norm) from the last solveJacobi call.
    // solveTwoBand recovers the incoming from the fixed-point identity
    // (B−E−source)/refl instead of a second SpMV; this is the fallback where
    // refl→0 (blackbody) makes that division ill-posed. Written non-thread-safely
    // like last_iterations_/last_residual_ (the two-band solves are sequential).
    std::vector<double> last_jacobi_incoming_;
    // NOTE (B2/B3): the previous single-slot visible cache (keyed on the last
    // solar_flux) was REMOVED. In production the coupling loop nests for iter { for t },
    // so consecutive solve() calls carry different solar_flux and the cache MISSED
    // every call while paying an O(n) vector compare + copy, and it was mutable
    // cross-call state hostile to concurrency. The visible system (I - A F) B_vis =
    // A Q_solar converges in 1-2 SpMV (ρ(A F) ≤ A_max f_max < 1; derivation §3), so
    // re-solving it each call is the intended, cheap cost — not the cache.
    // Per-face view-factor row normalization = 1/max(1, Σ_j F_ij). The M1
    // centroid view-factor estimator overshoots in the near field, so on fine
    // or concave meshes Σ_j F_ij can exceed 1 — non-physical for an open scene
    // (must be ≤ 1). Uncorrected, this drives the radiosity gain > 1 and makes
    // the thermal–radiation coupling diverge (indirect flux blew up past 10× the
    // solar constant, clamping temperatures to MIN/MAX). Multiplying every F·x
    // result by this factor renormalizes to an energy-conserving F̂·x.
    std::vector<double> vf_row_norm_;
    // max_i Σ_j F_ij and the count of rows the normalization touched, kept from the
    // one F·1 matvec above so the GPU gate can score the real divergence criterion
    // (gain = ρ_max · max_i min(Σ_j F_ij, 1)) instead of re-deriving it.
    double vf_row_sum_max_ = 0.0;
    size_t vf_row_norm_count_ = 0;
    // Which route PTM_HMATRIX_VF_REPLACE took for the corrected CSR. Exactly one
    // is true once it has run (both false when it did not run at all).
    //   _cpu_ : installed as a CPU-only SparseVFOperator. initGPU() reports it if
    //           the config also asked for GPU radiosity.
    //   _gpu_ : installed as a ViewFactorMatrix with F_ptr_ set, for initGPU and
    //           the pool. If no GPU rung then engages,
    //           restoreCpuVfOperatorAfterGpuDecline() converts it back so the
    //           fallback is the production CPU solver rather than Gauss-Seidel.
    bool vf_replace_cpu_route_ = false;
    bool vf_replace_gpu_route_ = false;
    // Path the corrected CSR came from, for the promotion message.
    std::string vf_replace_path_;
    // use_hmatrix_ as it stood before a promotion, so releasing restores it.
    bool vf_replace_hmatrix_saved_ = false;
    RadiationConfig config_;

#ifdef PTM_GPU_OPENCL
    // GPU radiosity solver — single-device path (used when gpu_devices == 1).
    std::unique_ptr<gpu::opencl::RadiosityOpenCLSolver> gpu_solver_;
    // Multi-GPU dispatch pool (used when gpu_devices > 1). When set, each
    // batch picks the next-idle device's persistent solver and calls solve().
    std::unique_ptr<gpu::opencl::RadiosityGPUDevicePool> gpu_pool_;
    // True when ANY GPU rung is active. gpu_solver_ != nullptr means the
    // single-GPU rung; gpu_pool_ != nullptr means the multi-GPU pool rung.
    bool use_gpu_ = false;
    int  pool_batch_size_ = 0;  // resolved from config: 4 * n_devices when 0/auto
    bool pool_single_fallback_warned_ = false;
#endif

    // Set while solveBatch runs its serial per-timestep fallback, so solve()
    // cannot bounce that timestep back into solveBatch. Declared outside the
    // GPU #ifdef because solveBatch's serial loop is compiled in both builds.
    bool in_batch_serial_ = false;
    struct BatchSerialGuard {
        bool& flag;
        explicit BatchSerialGuard(bool& f) : flag(f) { flag = true; }
        ~BatchSerialGuard() { flag = false; }
        BatchSerialGuard(const BatchSerialGuard&) = delete;
        BatchSerialGuard& operator=(const BatchSerialGuard&) = delete;
    };

    // Convergence tracking
    int last_iterations_ = 0;
    int last_iterations_ir_ = 0;
    int last_iterations_vis_ = 0;
    double last_residual_ = 0.0;
    bool converged_ = false;

    // Cost accounting (PTM_RAD_COST_REPORT): traversals of the view-factor matrix
    // charged by the last solve(). Each solver path sets this itself, because the
    // relationship between iteration counts and traversals differs per path -- the
    // fused CPU two-band does ONE width-2 traversal per outer iteration, while the
    // GPU two-band runs the bands as two separate solves and pays IR + visible.
    int last_csr_traversals_ = 0;

    // Work arrays (reused across calls)
    mutable std::vector<double> radiosity_;
    mutable std::vector<double> radiosity_new_;
    mutable std::vector<double> temp_vec_;
};

// Face selection for determining which faces need radiation exchange
class FaceSelector {
public:
    FaceSelector(const std::shared_ptr<ptm::Mesh>& mesh,
                 const RadiationConfig& config);

    // Classify faces based on illumination history and geometry
    // illumination_series: [n_faces x n_times] illumination values
    // temperatures: optional [n_faces x n_times] temperature history
    FaceClassification classify(
        const std::vector<std::vector<double>>& illumination_series,
        const std::vector<std::vector<double>>& temperatures = {});

    // Quick classification based on geometry only
    FaceClassification classifyGeometric();

    // Compute shadow fraction for each face from illumination series
    std::vector<double> computeShadowFractions(
        const std::vector<std::vector<double>>& illumination_series) const;

private:
    std::shared_ptr<ptm::Mesh> mesh_;
    RadiationConfig config_;
    std::vector<std::array<double, 3>> face_normals_;
    std::vector<std::array<double, 3>> face_centers_;
};

// One emitter->receiver beam pair for the Smith (1967) anisotropic thermal-
// emission pass. Built once in prepare() (only when rms_slope_deg > 0), pruned
// to sunlit-wall (emitter j) -> shadowed-floor (receiver i) pairs. The beamed
// correction redistributes each wall's fixed total emission (epsilon sigma T^4)
// angularly: hot Sun-facing micro-facets beam IR preferentially toward the Sun,
// which at the lunar pole (grazing Sun, wall sloped toward floor) enhances
// wall->floor IR. See addBeamedFlux().
struct BeamPair {
    int i;                       // receiver face (shadowed floor)
    int j;                       // emitter face (sunlit wall)
    float Fhat_ij;               // renormalized VF weight (matches radiosity)
    float cos_e;                 // emission-angle cosine n_j . dir_ji (fixed)
    std::array<float, 3> dir_ji; // unit emitter->receiver direction (body frame)
};

// Main radiation exchange manager
// Coordinates view factor computation, radiosity solving, and thermal integration
class RadiationExchangeManager {
public:
    RadiationExchangeManager(const std::shared_ptr<ptm::Mesh>& mesh,
                             const RadiationConfig& config);

    ~RadiationExchangeManager();

    // Initialize Embree scene for visibility testing
    bool initializeScene();

    // Classify faces and compute view factors
    // illumination_series: illumination history for face selection
    // Returns true if radiation exchange is needed
    bool prepare(const std::vector<std::vector<double>>& illumination_series);

    // Check if radiation exchange is active
    bool isActive() const { return is_active_; }

    // Get number of faces with radiation exchange
    size_t numExchangeFaces() const {
        return classification_.exchange_face_indices.size();
    }

    // Compute indirect radiation for a single timestep
    // temperatures: current surface temperatures for all faces [K]
    // solar_flux: current solar flux for all faces [W/m^2]
    // Returns: indirect flux for all faces (0 for non-exchange faces) [W/m^2]
    std::vector<double> computeIndirectFlux(
        const std::vector<double>& temperatures,
        const std::vector<double>& solar_flux);

    // Overload with the current Sun direction (unit, body-fixed frame). Required
    // for the Smith anisotropic-emission pass (rms_slope_deg > 0); the beamed
    // wall->floor enhancement is Sun-direction-dependent. When rms_slope_deg == 0
    // this is byte-identical to the 2-arg overload. The 2-arg overload delegates
    // here with sun_dir = {0,0,0} (beaming inactive).
    std::vector<double> computeIndirectFlux(
        const std::vector<double>& temperatures,
        const std::vector<double>& solar_flux,
        const std::array<double, 3>& sun_dir);

    // Compute indirect radiation for a batch of timesteps (multi-GPU path).
    // Inputs are [n_timesteps][n_faces]; output mirrors the shape.
    // When the underlying RadiositySolver has a GPU pool, the batch is
    // dispatched across all pool devices in parallel. Otherwise loops
    // computeIndirectFlux() per timestep. The caller is responsible for
    // chunking the full timestep range into reasonably-sized batches.
    std::vector<std::vector<double>> computeIndirectFluxBatch(
        const std::vector<std::vector<double>>& temperatures_per_timestep,
        const std::vector<std::vector<double>>& solar_flux_per_timestep);

    // Batch overload with per-timestep Sun directions (unit, body-fixed). When
    // rms_slope_deg > 0 the Smith beamed pass is applied per timestep after the
    // (multi-GPU) radiosity solve. sun_dirs.size() must equal the batch length.
    // Empty sun_dirs (or rms_slope_deg == 0) => identical to the 2-arg overload.
    std::vector<std::vector<double>> computeIndirectFluxBatch(
        const std::vector<std::vector<double>>& temperatures_per_timestep,
        const std::vector<std::vector<double>>& solar_flux_per_timestep,
        const std::vector<std::array<double, 3>>& sun_dirs);

    // True iff multi-GPU radiosity dispatch is active.
    bool hasGPUPool() const;

    // Devices in the ACTIVE radiosity pool (0 = no pool). See the solver's
    // twin: the configured count is not evidence of what is running.
    int gpuPoolDeviceCount() const;

    // Set the per-face visible Bond albedo A_i for the two-band radiosity (A2).
    // Stored and forwarded to the underlying RadiositySolver. May be called before
    // prepare() (applied at solver creation) or per-timestep before
    // computeIndirectFlux() when the surface-BC albedo is incidence-dependent, so
    // A_rad == A_BC per face and reflected-solar energy closes. Size must equal the
    // mesh face count.
    void setSolarAlbedo(const std::vector<double>& solar_albedo);

    // PTM-106: per-face DIRECT-BEAM source albedo for the visible band, forwarded to
    // the solver. Call per timestep with the SAME effective albedo A(theta) the
    // surface BC absorbed with; the enclosure then receives exactly the A(theta) Q
    // the BC did not absorb. Not stored across prepare() -- the solver must exist.
    // See RadiositySolver::setSolarSourceAlbedo for the source/operator distinction.
    void setSolarSourceAlbedo(const std::vector<double>& source_albedo);

    // Per-timestep source albedo for the batched (multi-GPU pool) path.
    void setSolarSourceAlbedoBatch(const std::vector<std::vector<double>>& source_albedo);

    // Activate band-separated (two-band) radiosity on the underlying solver (D1),
    // e.g. after per-face albedo has been sourced post-prepare(). Two-band enabled
    // this late uses the CPU two-band solve (the GPU two-band gate is set at
    // initGPU time). No-op if the solver does not exist yet.
    void enableTwoBand();

    // True iff the underlying solver is running band-separated (two-band) radiosity.
    bool isTwoBand() const;

    // Per-band absorbed indirect flux from the last computeIndirectFlux[Batch]()
    // (C1, Paul's product): absorbed reflected-SOLAR and absorbed thermal-IR as
    // SEPARATE full-face vectors (0 on non-exchange faces). Empty in the gray path.
    // Their sum equals the total indirect flux the surface BC consumes.
    const std::vector<double>& lastIndirectVisible() const { return indirect_flux_vis_; }
    const std::vector<double>& lastIndirectIR() const { return indirect_flux_ir_; }

    // Per-band batch results from the last computeIndirectFluxBatch() (multi-GPU
    // pool two-band path), indexed [timestep][face], 0 on non-exchange faces.
    // Empty on the gray pool path. Their per-timestep sum equals the batch total.
    const std::vector<std::vector<double>>& lastIndirectVisibleBatch() const { return indirect_vis_batch_; }
    const std::vector<std::vector<double>>& lastIndirectIRBatch() const { return indirect_ir_batch_; }

    // Get face classification
    const FaceClassification& getClassification() const { return classification_; }

    // Get view factor matrix (valid after prepare())
    const ViewFactorMatrix& getViewFactorMatrix() const { return view_factors_; }

    // Get H-matrix view factors (valid after prepare() with use_hmatrix=true)
    const HMatrixViewFactor* getHMatrixViewFactors() const { return hmatrix_view_factors_.get(); }

    // Load pre-computed view factors from file
    // Returns true if successfully loaded and skips computation in prepare()
    // File format is auto-detected (sparse .vfm or H-matrix .hmat)
    bool loadViewFactors(const std::string& filename);

    // Check if view factors have been loaded from file
    bool hasLoadedViewFactors() const { return view_factors_loaded_; }

    // Compute view factors only (no face classification or solver setup)
    // Use this for early computation before illumination is available.
    // View factors will be saved to disk and can be loaded later during prepare().
    // Returns true on success.
    bool computeViewFactorsOnly();

    // Set emissivity for all faces (or use per-face values)
    void setEmissivity(double emissivity);
    void setEmissivity(const std::vector<double>& emissivity);

    // The emissivity the solver OPERATIVELY holds -- i.e. after
    // RadiositySolver::initCommon has applied any PTM_RAD_EMISSIVITY override,
    // which is the one way the radiosity's 1-eps can come apart from the
    // emissivity the thermal path absorbs and emits with. Empty before
    // prepare(). Read by the engine's Kirchhoff cross-check; do not use it as
    // a configuration source.
    const std::vector<double>& operativeEmissivity() const;

    // Get/set configuration
    const RadiationConfig& getConfig() const { return config_; }
    void setConfig(const RadiationConfig& config);

    // ρ(F) of the operator the radiosity ACTUALLY holds after prepare() --
    // H-matrix, dense, a PTM_HMATRIX_VF_REPLACE corrected CSR, or a congruence
    // rescaling of one of those. Advisory only; see
    // RadiositySolver::estimateSpectralRadius. Returns ok = false before
    // prepare() or when no operator was loaded.
    SpectralRadiusEstimate estimateVFSpectralRadius(int max_iterations = 60,
                                                    double rel_tol = 1e-6) const;

    // Statistics
    struct Statistics {
        double mean_indirect_flux = 0.0;
        double max_indirect_flux = 0.0;
        double mean_sky_view_factor = 0.0;
        int solver_iterations = 0;
        double solver_residual = 0.0;
    };
    Statistics getStatistics() const { return stats_; }

private:
    // Cleanup Embree resources
    void cleanup();

    // Data members
    std::shared_ptr<ptm::Mesh> mesh_;
    RadiationConfig config_;

    // Embree resources
    void* device_ = nullptr;  // RTCDevice
    void* scene_ = nullptr;   // RTCScene
    bool scene_initialized_ = false;

    // View factor data
    ViewFactorMatrix view_factors_;
    std::unique_ptr<HMatrixViewFactor> hmatrix_view_factors_;
    std::unique_ptr<RadiositySolver> solver_;
    bool using_hmatrix_ = false;
    bool view_factors_loaded_ = false;  // True if loaded from file (skip computation)

    // Face classification
    FaceClassification classification_;
    bool is_active_ = false;

    // PTM-98: true when every face keeps the flux the radiosity solve computed
    // for it, i.e. the post-solve copy mask is the identity. Then the write-back
    // is a whole-vector move instead of a scatter, and no face can be left
    // holding a stale or zero value that the solver actually filled in.
    bool writeBackAll() const {
        return classification_.exchange_faces == classification_.total_faces;
    }

    // PTM-98: copy one solver result (indexed by GLOBAL face id, length >= dst)
    // into a destination already sized and zero-filled. Total when every face
    // participates; otherwise only the classified subset survives, which is the
    // pre-PTM-98 behavior kept for reproduction. Single definition so the total
    // and per-band write-backs cannot drift apart.
    void writeBackFlux(const std::vector<double>& src,
                       std::vector<double>& dst) const {
        const size_t n = dst.size();
        if (writeBackAll() && src.size() >= n) {
            std::copy(src.begin(), src.begin() + n, dst.begin());
            return;
        }
        for (int face_idx : classification_.exchange_face_indices) {
            if (face_idx >= 0 && static_cast<size_t>(face_idx) < src.size()) {
                dst[face_idx] = src[face_idx];
            }
        }
    }

    // Overload for a source we own: steals the buffer instead of copying it.
    void writeBackFlux(std::vector<double>&& src, std::vector<double>& dst) const {
        if (writeBackAll() && src.size() == dst.size()) {
            dst = std::move(src);
            return;
        }
        writeBackFlux(static_cast<const std::vector<double>&>(src), dst);
    }

    // Receivers for the Smith (1967) beamed-emission pass: the shadowed-floor
    // faces. Kept separate from classification_.exchange_face_indices, which is
    // all faces by default since PTM-98. Built in buildBeamPairs().
    std::vector<int> beam_receivers_;

    // Face properties
    std::vector<double> emissivity_;
    std::vector<double> areas_;
    // Per-face visible Bond albedo A_i for the two-band radiosity (A2). Empty =>
    // the solver keeps its env-sourced albedo. Forwarded to solver_ on creation
    // (prepare) and whenever setSolarAlbedo() is called.
    std::vector<double> solar_albedo_;

    // Statistics
    Statistics stats_;

    // Work arrays
    std::vector<double> indirect_flux_;
    // Per-band absorbed indirect flux (C1), mapped to all faces. Populated by
    // computeIndirectFlux[Batch]() only when the solver runs two-band.
    std::vector<double> indirect_flux_vis_;
    std::vector<double> indirect_flux_ir_;
    std::vector<std::vector<double>> indirect_vis_batch_;  // [step][face], pool two-band
    std::vector<std::vector<double>> indirect_ir_batch_;   // [step][face], pool two-band

    // --- Smith (1967) anisotropic beamed thermal emission ---------------------
    // All inactive unless config_.rms_slope_deg > 0 (beam_active_ == false).
    bool beam_active_ = false;
    double chi_w_rad_ = 0.0;                       // RMS micro-slope [rad]
    // Also beam the reflected-SOLAR source (opposition/backscatter) toward the floor,
    // not just thermal IR. Gated by PTM_BEAM_SOLAR; solar_albedo_beam_ is the Bond
    // albedo applied to the per-face incident solar flux. See addBeamedFlux().
    bool beam_solar_ = false;
    double solar_albedo_beam_ = 0.0;
    std::vector<BeamPair> beam_pairs_;             // sunlit-wall -> shadowed-floor
    std::vector<std::array<float, 3>> beam_normals_; // per-face mean normals (float)
    // Directional emission factor chi(cos_sun, cos_e, cos_psi) on a regular grid.
    // cos_sun = n_j.s_hat in [0,1]; cos_e = n_j.dir_ji in [0,1]; cos_psi = dir_ji.s_hat in [-1,1].
    // Energy-normalized: cos_e-weighted hemispherical mean of chi == 1.
    std::vector<float> chi_table_;
    int chi_ns_ = 0, chi_ne_ = 0, chi_npsi_ = 0;

    // Build the chi lookup table by Gaussian quadrature over the Smith facet-
    // slope distribution (called once in prepare() when rms_slope_deg > 0).
    void buildChiTable();
    // Build the pruned sunlit-wall -> shadowed-floor beam-pair list (geometry +
    // Embree visibility), renormalized to match the radiosity VF weights.
    void buildBeamPairs();
    // Add the per-timestep beamed correction: for each pair,
    //   flux[i] += eps_i * eps_j * sigma * T_j^4 * Fhat_ij * (chi - 1).
    void addBeamedFlux(const std::vector<double>& temperatures,
                       const std::vector<double>& solar_flux,
                       const std::array<double, 3>& sun_dir,
                       std::vector<double>& flux) const;
    // Read PTM_BEAM_SOLAR / PTM_SOLAR_ALBEDO to enable beaming the reflected-solar
    // source toward the floor (opposition), in addition to thermal IR. Called from
    // prepare() when beaming is active.
    void configureSolarBeaming();
    // Trilinear lookup into chi_table_; returns 1.0 outside the active domain.
    float lookupChi(double cos_sun, double cos_e, double cos_psi) const;
};

} // namespace radiation
