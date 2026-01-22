#pragma once

#include <vector>
#include <array>
#include <memory>
#include <functional>

#include "radiation_config.h"
#include "view_factor.h"
#include "hmatrix_view_factor.h"
#include "../mesh/mesh_utils.h"

// Forward declaration for GPU radiosity solver
#ifdef PTM_GPU_OPENCL
namespace gpu {
namespace opencl {
    class RadiosityOpenCLSolver;
}
}
#endif

namespace radiation {

// Stefan-Boltzmann constant [W m^-2 K^-4]
constexpr double SIGMA_SB = 5.67051196e-8;

// Abstract interface for view factor matrix-vector product
// Allows RadiositySolver to work with either ViewFactorMatrix or HMatrixViewFactor
class IViewFactorOperator {
public:
    virtual ~IViewFactorOperator() = default;
    virtual void matvec(const std::vector<double>& x, std::vector<double>& y) const = 0;
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

    // Set configuration
    void setConfig(const RadiationConfig& config) { config_ = config; }

    // Solve for indirect radiation flux at each face
    // temperatures: current surface temperatures [K]
    // solar_flux: direct solar flux at each face [W/m^2]
    // Returns: total absorbed indirect flux at each face [W/m^2]
    std::vector<double> solve(const std::vector<double>& temperatures,
                              const std::vector<double>& solar_flux);

    // Compute thermal emission from faces (ε σ T^4)
    std::vector<double> computeThermalEmission(const std::vector<double>& temperatures) const;

    // Compute scattered solar radiation (single bounce)
    std::vector<double> computeScatteredSolar(const std::vector<double>& solar_flux) const;

    // Get convergence information from last solve
    int getLastIterations() const { return last_iterations_; }
    double getLastResidual() const { return last_residual_; }
    bool didConverge() const { return converged_; }

    // Compute energy balance check
    // Returns: [total_emission, total_absorption, imbalance]
    std::array<double, 3> checkEnergyBalance(const std::vector<double>& temperatures,
                                              const std::vector<double>& solar_flux,
                                              const std::vector<double>& indirect_flux) const;

    // Initialize GPU solver if enabled (call after setConfig)
    void initGPU();

private:
    // Initialize common data
    void initCommon(const std::vector<double>& emissivity,
                    const std::vector<double>& areas);

    // Jacobi iteration solver
    std::vector<double> solveJacobi(const std::vector<double>& emission,
                                    const std::vector<double>& solar_source);

    // Gauss-Seidel iteration (faster convergence)
    std::vector<double> solveGaussSeidel(const std::vector<double>& emission,
                                         const std::vector<double>& solar_source);

#ifdef PTM_GPU_OPENCL
    // GPU-accelerated solve
    std::vector<double> solveGPU(const std::vector<double>& temperatures,
                                 const std::vector<double>& solar_flux);
#endif

    // Data members
    std::unique_ptr<IViewFactorOperator> F_op_;
    const ViewFactorMatrix* F_ptr_ = nullptr;  // For row access in Gauss-Seidel
    bool use_hmatrix_ = false;
    std::vector<double> emissivity_;
    std::vector<double> reflectivity_;  // 1 - emissivity
    std::vector<double> areas_;
    RadiationConfig config_;

#ifdef PTM_GPU_OPENCL
    // GPU radiosity solver
    std::unique_ptr<gpu::opencl::RadiosityOpenCLSolver> gpu_solver_;
    bool use_gpu_ = false;
#endif

    // Convergence tracking
    int last_iterations_ = 0;
    double last_residual_ = 0.0;
    bool converged_ = false;

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

    // Get/set configuration
    const RadiationConfig& getConfig() const { return config_; }
    void setConfig(const RadiationConfig& config);

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

    // Face properties
    std::vector<double> emissivity_;
    std::vector<double> areas_;

    // Statistics
    Statistics stats_;

    // Work arrays
    std::vector<double> indirect_flux_;
};

} // namespace radiation
