#pragma once
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <map>
#include <functional>
#include <limits>
#include <iostream>
#include "../constants/physics_constants.h"
#include "../thermal/thermal_model.h"
#include "../mesh/mesh_utils.h"
#include "../radiation/horizon_field.h"

// Add these Embree headers
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

// Forward declarations for Embree
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;

// GPU BVH ray tracer forward declarations
#ifdef PTM_GPU_OPENCL
namespace gpu {
namespace opencl {
    class BVHBuilder;
    class BVHRayTracer;
    class BVHGPUPool;
}
}
#endif

// Forward declarations for orbital mechanics
namespace orbital {
    class KeplerianPropagator;
    class BodyFixedFrame;
    struct GenericBody;
}

namespace ptm {

class IlluminationModel {
public:
    // Constructor and destruction
    IlluminationModel(bool debug_output = false);
    ~IlluminationModel();

    void setUseGeometricHorizon(bool enable) { use_geometric_horizon_ = enable; }

    // PTM-13: local-horizon (skyline) culling of the grazing DIRECT-sun shadow. Default OFF
    // (byte-identical). Mirrors setUseGeometricHorizon. Also settable via env
    // PTM_ILLUM_HORIZON_CULL=1 (+ PTM_ILLUM_HORIZON_SLACK_DEG / _CELL_SCALE tunables), read in
    // initializeScene. CPU illumination path only (warns if GPU ray tracing is active).
    void setHorizonCull(bool enable) { use_horizon_cull_ = enable; }
    void setHorizonSlackDeg(double deg) { horizon_slack_deg_ = deg; }
    void setHorizonCellScale(double s) { horizon_cell_scale_ = s; }

    // PTM-13 P1: topological self-skip for the grazing DIRECT-sun shadow ray. When enabled, the CPU
    // occlusion test starts the ray AT the face (eps_n=eps_d=0, tnear=0) and skips the source face +
    // its <= selfskip_rings_ edge-neighbors by primID (an iterative rtcIntersect1 re-trace) instead
    // of relying on a large geometric ray-origin offset, so a real mm-scale local rim occluder is
    // detected rather than hopped by the 1 cm offset. Default OFF -> byte-identical. Also settable
    // via env PTM_ILLUM_TOPO_SELFSKIP=1 (+ PTM_ILLUM_SELFSKIP_RINGS=<n>), read in initializeScene.
    // CPU illumination path only (warns if GPU ray tracing is active). See BUGS.md PTM-13.
    void setTopoSelfSkip(bool enable) { use_topo_selfskip_ = enable; }
    void setSelfSkipRings(int rings) { if (rings >= 1) selfskip_rings_ = rings; }

    // PTM-13 rise-epsilon: the physical self-relief discriminator for the grazing DIRECT-sun shadow
    // ray. When enabled, the CPU occlusion test marches from the TRUE face centroid (eps_n=eps_d=0,
    // tnear~0, NO metric offset, NO ring skip) and rejects a hit as self-relief iff its out-of-plane
    // rise dz = (P_hit - c_src)*n_src is below a facet-scaled floor h_eps = rise_keps_ * local_edge(src);
    // the FIRST hit rising >= h_eps out of the source tangent plane is a real occluder. Separates the
    // ~0.12 mm coplanar self-relief from the ~2.47 mm real rim occluder the 1 cm offset hops (BUGS.md
    // PTM-13). Resolution-independent (h_eps scales with the facet, so A3GT m-DEM occluders are always
    // kept). Independent of the topological self-skip flag. Default OFF -> byte-identical. Also settable
    // via env PTM_ILLUM_RISE_EPS=1 (+ PTM_ILLUM_RISE_KEPS=<k>), read in initializeScene. CPU path only
    // (warns if GPU ray tracing is active). See BUGS.md PTM-13.
    void setRiseEps(bool enable) { use_rise_eps_ = enable; }
    void setRiseKeps(double k) { if (k > 0.0) rise_keps_ = k; }

    // PTM-18: interior-hole seal for the grazing DIRECT-sun shadow. Small INTERIOR boundary-edge loops
    // in the core mesh (missing triangles / T-junction cracks -- NOT the outer patch perimeter) let a
    // low grazing sun thread through to shadowed floor faces via the finite-disk penumbra path, a
    // spurious "flash" the rise-epsilon / true-normal fixes cannot touch (it is a mesh topology defect,
    // not an occlusion-verdict error). When enabled, initializeScene() triangulates each small interior
    // boundary loop and appends the fill triangles to the shadow BVH (occluder-only, like the ring);
    // the standard occlusion then blocks the threaded rays. Default OFF -> byte-identical (no fill).
    // Also settable via env PTM_ILLUM_SEAL_HOLES=1, read in initializeScene. See BUGS.md PTM-18.
    void setSealInteriorHoles(bool enable) { seal_interior_holes_ = enable; }

    // Setup and configuration
    void setMeshResolution(int resolution);
    void setTimeSteps(int steps);
    void setAlbedo(double albedo);

    // Supply an externally-built, possibly non-uniform ephemeris-time grid.
    // When set (non-empty), calculateIllumination / calculateIlluminationFromMesh
    // populate times_, time_strings_, sun_positions_, primary_positions_ from
    // these ETs instead of the internal uniform grid. Size must equal time_steps_.
    // Used for decoupled equilibration/output cadences.
    void setExternalTimeGrid(const std::vector<double>& ets_tdb);
    void clearExternalTimeGrid();
    bool hasExternalTimeGrid() const { return !external_ets_.empty(); }
    void setKernelsPath(const std::string& path) {
        kernels_path_ = path;
    }

    // Set the target body (e.g., "EUROPA", "MOON")
    void setTargetBody(const std::string& body_name) {
        target_body_name_ = body_name;
        // Update reference frame automatically
        target_body_frame_ = "IAU_" + body_name;
        // Update SPICE ID via lookup function
        target_body_id_ = getSpiceIdForBody(body_name);
    }

    // Set the target body using a planetary body enum
    void setMetaKernel(const std::string& filename) {
        meta_kernel_filename_ = filename;
    }
    
    // Add getter for meta-kernel filename
    std::string getMetaKernel() const {
        return meta_kernel_filename_;
    }

    // Set the primary body (e.g., "JUPITER", "EARTH") 
    void setPrimaryBody(const std::string& body_name) {
        primary_body_name_ = body_name;
        
        // CRITICAL: Update the primary_is_sun_ flag and radius
        if (body_name == "SUN") {
            primary_is_sun_ = true;
            primary_radius_ = constants::SUN_RADIUS;
        } else {
            primary_is_sun_ = false;
            
            // Set the appropriate radius for non-Sun primaries
            if (body_name == "JUPITER") {
                primary_radius_ = constants::JUPITER_RADIUS;
            } else if (body_name == "EARTH") {
                primary_radius_ = constants::EARTH_RADIUS;
            } else if (body_name == "SATURN") {
                primary_radius_ = constants::SATURN_RADIUS;
            } else if (body_name == "MARS") {
                primary_radius_ = constants::MARS_RADIUS;
            } else {
                // Default to Earth for unknown bodies
                primary_radius_ = constants::EARTH_RADIUS;
                std::cerr << "Warning: Unknown primary body '" << body_name 
                        << "', using Earth radius as default" << std::endl;
            }
        }
        
        // // Debug output to confirm
        // std::cout << "setPrimaryBody: " << body_name 
        //         << " (is_sun=" << (primary_is_sun_ ? "true" : "false")
        //         << ", radius=" << primary_radius_ << ")" << std::endl;
    }

    // Set both target and primary bodies using a planetary body enum
    void setPlanetaryBody(thermal::PlanetaryBody body);

    // Configure from a generic body definition (uses Keplerian propagation)
    void setGenericBody(const orbital::GenericBody& body);

    // Check if using Keplerian propagation (vs SPICE)
    bool isKeplerian() const { return use_keplerian_; }

    // Check if a primary body is configured
    bool hasPrimaryBody() const { return has_primary_body_; }

    // Get the albedo for the specific planetary body
    double getAlbedo() const { return albedo_; }
    
    // Helper function to get SPICE ID for a body
    static std::string getSpiceIdForBody(const std::string& body_name);
    
    // Get current target and primary body names
    std::string getTargetBodyName() const { return target_body_name_; }
    std::string getPrimaryBodyName() const { return primary_body_name_; }

    /**
     * Calculate illumination for a mesh
     * @param mesh_vertices Vector of 3D point coordinates (x,y,z)
     * @param mesh_faces Vector of face indices (triplets)
     * @param start_time Starting time in ISO format
     * @param duration_days Simulation duration in Earth days
     * @return True if calculation was successful
     */

    bool calculateIlluminationFromMesh(
        const std::shared_ptr<Mesh>& mesh,
        const std::string& start_time,
        double duration_days);

    bool calculateIllumination(
        const std::vector<std::array<double, 3>>& mesh_vertices,
        const std::vector<std::array<int, 3>>& mesh_faces,
        const std::string& start_time,
        double duration_days
    );
    
    bool calculateIlluminationOptimized(
        const std::vector<std::array<double, 3>>& mesh_vertices,
        const std::vector<std::array<int, 3>>& mesh_faces,
        const std::string& start_time,
        double duration_days);
    
    /**
     * Calculate illumination for specific locations
     * @param locations Vector of latitude/longitude pairs (degrees)
     * @param start_time Starting time in ISO format
     * @param duration_days Simulation duration in Earth days
     * @return Map of location index to illumination time series
     */
    std::map<int, std::vector<double>> calculateLocationIllumination(
        const std::vector<std::pair<double, double>>& locations,
        const std::string& start_time,
        double duration_days
    );
    
    // Access results
    const std::vector<double>& getIllumination() const;
    const std::vector<double>& getTimes() const;
    const std::vector<std::string>& getTimeStrings() const { return time_strings_; }
    std::vector<std::pair<double, double>> getIlluminationForFace(int face_idx) const;
    std::vector<std::array<double, 3>> getSunPositions() const;
    std::vector<std::array<double, 3>> getPrimaryPositions() const;
    
    bool loadPositionsFromFile(const std::string& filename);
    bool initialize_spice(bool verbose = false);
    double convertTimeToET(const std::string& time_str);
    std::array<double, 3> getSunPosition(double et);
    std::array<double, 3> getPrimaryPosition(double et);
    static std::array<double, 3> cartesianFromLatLon(double lat, double lon, double radius = 1.0);
    static std::pair<double, double> latLonFromCartesian(double x, double y, double z);

    const std::vector<double>& getPrimaryIRFlux() const {
        return primary_ir_values_;
    }

    std::vector<std::pair<double, double>> getPrimaryIRFluxForFace(int face_idx) const;

    // Add these to the public methods
    void setIncludePrimaryReflection(bool include) { include_primary_reflection_ = include; }
    bool getIncludePrimaryReflection() const { return include_primary_reflection_; }

    // Getter methods for component values
    const std::vector<double>& getDirectSolarValues() const { return direct_solar_values_; }
    const std::vector<double>& getPrimaryReflectedValues() const { return primary_reflected_values_; }

    // Add a setter for debug output
    void setDebugOutput(bool enabled) { debug_output_enabled_ = enabled; }

    // New setter method for controlling progress bar visibility
    void setShowProgress(bool show) { show_progress_ = show; }

    // GPU ray tracing support
#ifdef PTM_GPU_OPENCL
    /**
     * @brief Enable GPU-accelerated ray tracing for illumination
     * @param enable If true, use GPU BVH ray tracer instead of CPU Embree
     * @return true if GPU was successfully initialized, false otherwise
     */
    bool enableGPURayTracing(bool enable = true);

    /**
     * @brief Configure multi-GPU ray tracing.
     *
     * Must be called BEFORE initializeScene() so the device list is set
     * when the BVH/mesh upload happens. Defaults preserve single-GPU
     * behaviour.
     *
     * @param gpu_devices N (>=1) or 0 for "all available GPUs".
     * @param gpu_device_ids If non-empty, exactly those device indices
     *        (overrides gpu_devices). Indices come from
     *        OpenCLContext::enumerateGPUs().
     * @param inter_batch_pause_ms Optional sleep between batches on the
     *        same device (watchdog mitigation knob).
     */
    void configureGPURayTracingDevices(int gpu_devices,
                                       const std::vector<int>& gpu_device_ids,
                                       int inter_batch_pause_ms = 0);

    /**
     * @brief Set the BVH disk cache path. Empty string disables caching.
     * If the file exists at scene-init time, the BVH is loaded from it
     * (skipping the SAH build, which dominates first-run startup for
     * 5 M-face meshes). After the build, the BVH is written back to
     * the file for future runs to consume.
     */
    void setBVHCacheFile(const std::string& path) { bvh_cache_file_ = path; }
    const std::string& getBVHCacheFile() const { return bvh_cache_file_; }

    /**
     * Register occluder-only geometry (e.g. a coarse LOLA ring/annulus
     * surrounding the SfS core). These triangles are appended to the
     * ray-tracing BVH so distant topography shadows the core, but are
     * NOT thermally solved, NOT in the radiosity matrix, and NOT output —
     * they exist only to block direct-solar rays. Must be called BEFORE
     * initializeScene()/calculateIlluminationFromMesh(). Empty = no-op
     * (the core-only path is then byte-identical to before).
     */
    void setOccluderGeometry(
        const std::vector<std::array<double, 3>>& vertices,
        const std::vector<std::array<int, 3>>& faces) {
        occluder_vertices_ = vertices;
        occluder_faces_ = faces;
    }

    /**
     * @brief Check if GPU ray tracing is currently enabled and available
     */
    bool isGPURayTracingEnabled() const { return use_gpu_ray_tracer_; }

    /**
     * @brief Get GPU ray tracer statistics
     */
    struct GPURayTracerStats {
        size_t rays_traced = 0;
        double total_time_ms = 0;
        double kernel_time_ms = 0;
    };
    GPURayTracerStats getGPURayTracerStats() const;
#endif

    // Accessor methods that work with chunked storage
    double getIlluminationValue(size_t face_idx, size_t time_idx) const;
    double getDirectSolarValue(size_t face_idx, size_t time_idx) const;
    double getPrimaryReflectedValue(size_t face_idx, size_t time_idx) const;
    double getPrimaryIRValue(size_t face_idx, size_t time_idx) const;

    // Enable/disable shadow flux calculation
    void enableShadowFlux(bool enable, double diameter_depth_ratio = 8.0) {
        enable_shadow_flux_ = enable;
        crater_diameter_depth_ratio_ = diameter_depth_ratio;
        if (diameter_depth_ratio < 1.0) {
            std::cerr << "Warning: diameter/depth ratio should be > 1. Using inverse." << std::endl;
            crater_diameter_depth_ratio_ = 1.0 / diameter_depth_ratio;
        }
    }
    
    // Set body optical properties for shadow flux calculation
    void setBodyOpticalProperties(double albedo, double emissivity) {
        body_albedo_ = albedo;
        body_emissivity_ = emissivity;
    }
    
    // Calculate shadow flux based on Ingersoll et al. (1992)
    double calculateShadowFlux(
        const std::array<double, 3>& face_center,
        const std::array<double, 3>& face_normal,
        const std::array<double, 3>& sun_position
    ) const;

private:
    // Thread-local ray batch structure
    struct RayBatch {
        std::vector<std::array<double, 3>> origins;
        std::vector<std::array<double, 3>> directions;
        std::vector<int> results;
        std::vector<size_t> indices;
        size_t current_size = 0;
        bool processed = false;
    };
    // Ray tracing methods
    std::vector<uint8_t> traceOccluded(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        bool coherent = false);

    // PTM-13 P1: occlusion test with topological self-skip. Per ray, iteratively rtcIntersect1 and
    // skip any hit on the source face or its <= selfskip_rings_ edge-neighbors (face_skip_off_/
    // face_skip_nbr_), returning blocked only on the first NON-neighbor hit. Callers pass origins AT
    // the face (no geometric offset) so the near local occluder is detected. CPU path; used only
    // when use_topo_selfskip_ is set. src_faces[i] = source core-face id for ray i (-1 = no skip).
    std::vector<uint8_t> traceOccludedSelfSkip(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        const std::vector<int>& src_faces);

    // Build the per-core-face self-skip CSR (BFS over <= selfskip_rings_ edge rings of the merged
    // mesh). Called from initializeScene when use_topo_selfskip_ is set. n_core = # illuminated
    // (core) faces; faces_all = merged core+occluder connectivity (primID space).
    void buildSelfSkipAdjacency(const std::vector<std::array<int, 3>>& faces_all, size_t n_core);

    // PTM-13 rise-epsilon occlusion: march from the true centroid (tnear~0, no offset, no ring skip),
    // rejecting a hit as self-relief iff its out-of-plane rise dz = (P_hit - c_src)*n_src < h_eps
    // (per-face, = rise_keps_ * face_edge_local_[src]); the first hit with dz >= h_eps is a real
    // occluder. Callers pass origins AT the face (eps_n=eps_d=0) so c_src = origin and dz =
    // t_hit*(dir*n_src). The origin primID is also skipped (dz~0) for float robustness. CPU path; used
    // when use_rise_eps_ is set. src_faces[i]/src_normals[i] = source core-face id / effective outward
    // normal for ray i.
    std::vector<uint8_t> traceOccludedRiseEps(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions,
        const std::vector<int>& src_faces,
        const std::vector<std::array<double, 3>>& src_normals);

    // PTM-18: build fan-triangulated fill for the SMALL INTERIOR boundary-edge loops of the core mesh
    // (missing-triangle holes / T-junction cracks), EXCLUDING the outer patch perimeter (the boundary
    // component whose spatial extent spans the mesh). Returns occluder-only triangles in CORE vertex
    // index space (appended to the shadow BVH by initializeScene). Only well-formed closed loops are
    // filled; open/non-manifold components are skipped with a warning (silence is not an option).
    // Called from initializeScene when seal_interior_holes_ is set. See BUGS.md PTM-18.
    std::vector<std::array<int, 3>> buildInteriorHoleFill(
        const std::vector<std::array<double, 3>>& vertices,
        const std::vector<std::array<int, 3>>& faces) const;

    bool checkLunarHorizon(
        const std::array<double, 3>& ray_origin,
        const std::array<double, 3>& ray_direction,
        double lunar_radius = 1737400.0) const;
    
    bool use_geometric_horizon_ = true; // if true, re-enable geometric horizon clipping
    
    // Thread-local batch storage
    static thread_local RayBatch tls_batch;
    
    // Thread-local batching vectors for optimized ray tracing
    static thread_local std::vector<std::array<double, 3>> tls_ray_origins;
    static thread_local std::vector<std::array<double, 3>> tls_ray_directions;
    static thread_local std::vector<int> tls_ray_results;
    static thread_local size_t tls_batch_size;
    
    // Batch collection parameters
    static constexpr size_t COLLECTION_BATCH_SIZE = 64;   // Size to collect before processing
    static constexpr size_t MAX_BATCH_SIZE = 16;          // Optimal for AVX-512

    // Chunked storage for very large meshes
    bool using_chunked_storage_ = false;
    size_t chunk_size_ = 1000000; // 1 million faces per chunk
    std::vector<std::vector<double>> illumination_chunks_;
    std::vector<std::vector<double>> direct_solar_chunks_;
    std::vector<std::vector<double>> primary_reflected_chunks_;
    std::vector<std::vector<double>> primary_ir_chunks_;
    
    // Helper methods for chunked storage
    void initializeChunkedStorage(size_t total_faces, int time_steps);
    void storeChunkedValue(std::vector<std::vector<double>>& chunks, size_t face_idx, size_t time_idx, double value);
    double getChunkedValue(const std::vector<std::vector<double>>& chunks, size_t face_idx, size_t time_idx) const;

public:
    // Release chunked storage memory after extraction to flat arrays.
    void releaseChunkedStorage() {
        for (auto& c : illumination_chunks_)       std::vector<double>().swap(c);
        for (auto& c : direct_solar_chunks_)        std::vector<double>().swap(c);
        for (auto& c : primary_reflected_chunks_)   std::vector<double>().swap(c);
        for (auto& c : primary_ir_chunks_)          std::vector<double>().swap(c);
        illumination_chunks_.clear();
        direct_solar_chunks_.clear();
        primary_reflected_chunks_.clear();
        primary_ir_chunks_.clear();
        using_chunked_storage_ = false;
    }
private:
    
    // Mutex for thread synchronization when processing batches
    std::mutex batch_mutex_;
    std::vector<RayBatch*> shared_batches_;
    bool batches_being_processed_ = false;
    
    // Ray batch handling methods
    void initRayBatch();
    void initRayBatchCollection();
    size_t addRayToBatch(const std::array<double, 3>& origin, const std::array<double, 3>& direction);
    void processRayBatches();
    void processRayBatch();
    bool getRayHitResult(size_t batch_index);

    // Shadow flux calculation parameters
    bool enable_shadow_flux_ = false;
    double crater_diameter_depth_ratio_ = 8.0;  // D = diameter/depth (>1)
    double body_albedo_ = 0.1;  // Will be set from body parameters
    double body_emissivity_ = 0.98;  // Will be set from body parameters
    
    // prototype for SIMD debugging
    void printSIMDInfoTemporary();
    
    // Set default target body and primary body
    std::string target_body_name_ = "EUROPA";       // Default target body
    std::string primary_body_name_ = "JUPITER";     // Default primary body
    std::string target_body_frame_ = "IAU_EUROPA";  // Default reference frame
    std::string target_body_id_ = "502";            // Default SPICE ID for Europa
    std::string meta_kernel_filename_ = "europa.tm";  // Default meta-kernel

    // Primary body; flag to indicate if primary is the Sun
    bool primary_is_sun_ = false;
    double primary_radius_ = constants::JUPITER_RADIUS;  // Default, updated when body changes
    bool has_primary_body_ = true;  // True for satellites (Europa→Jupiter), false for heliocentric

    // Generic body / Keplerian propagation support
    bool use_keplerian_ = false;
    std::unique_ptr<orbital::KeplerianPropagator> keplerian_propagator_;
    std::unique_ptr<orbital::BodyFixedFrame> body_fixed_frame_;
    double keplerian_epoch_et_ = 0.0;  // ET offset for Keplerian epoch

    // Embree ray tracing methods
    bool initializeEmbree();
    void cleanupEmbree();
    bool initializeScene(
        const std::vector<std::array<double, 3>>& vertices,
        const std::vector<std::array<int, 3>>& faces);
    
    // Ray tracing and intersection detection
    std::vector<int> intersectsFirst(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions);
    std::vector<int> intersectsFirstSIMD(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions);
    std::vector<int> intersectsFirstAVX512(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions);
    std::vector<int> traceRays(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions);
    bool isAVX512Supported();
    double calculateSafeRayOffset(const std::array<double, 3>& face_center, 
                             const std::array<double, 3>& face_normal) const;
    
    // SPICE methods for ephemeris calculations
    void cleanupSpice();
    
    // SPICE error checking helper
    static void checkSpiceError(const std::string& operation);
    
    // Illumination physics calculations

    double calculatePrimaryHeating(
        const std::array<double, 3>& face_normal,
        const std::array<double, 3>& primary_direction,
        double primary_distance);

    double calculatePrimaryReflectedLight(
        const std::array<double, 3>& face_normal,
        const std::array<double, 3>& primary_direction,
        const std::array<double, 3>& sun_position,
        const std::array<double, 3>& primary_position);

    double calculateSolarIlluminationWithFiniteDisk(
        const std::array<double, 3>& face_center,
        const std::array<double, 3>& face_normal,
        const std::array<double, 3>& sun_position,
        const std::array<double, 3>& primary_position,
        int num_samples = 9,     // Default to 9 samples (3x3 grid)
        int src_face = -1);      // PTM-13 P1: source core-face id for topological self-skip (-1 = off)

    void calculatePrimaryIRBatch(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<std::array<double, 3>>& face_normals,
        const std::array<double, 3>& primary_position,
        std::vector<double>& ir_values);

    void calculateDirectSolarBatchWithFiniteDisk(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<std::array<double, 3>>& face_normals,
        const std::array<double, 3>& sun_position,
        const std::array<double, 3>& primary_position,
        std::vector<double>& direct_solar_values,
        int num_samples = 9);

    void calculatePrimaryReflectedBatch(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<std::array<double, 3>>& face_normals,
        const std::array<double, 3>& sun_position,
        const std::array<double, 3>& primary_position,
        std::vector<double>& reflected_values);

    // Vector math utilities
    inline double dotProduct(const std::array<double, 3>& v1, const std::array<double, 3>& v2) const;
    inline std::array<double, 3> normalize(const std::array<double, 3>& v) const;
    inline double vectorLength(const std::array<double, 3>& v) const;
    inline std::array<double, 3> crossProduct(const std::array<double, 3>& v1, const std::array<double, 3>& v2) const;
    
    // Vector utilities
    static std::array<double, 3> vectorSubtract(
        const std::array<double, 3>& v1,
        const std::array<double, 3>& v2);
    
    // Ray-sphere intersection test
    static bool raySphereIntersect(
        const std::array<double, 3>& ray_origin,
        const std::array<double, 3>& ray_direction,
        const std::array<double, 3>& sphere_center,
        double sphere_radius,
        double& t1,
        double& t2);

    // Create orthogonal basis for a direction vector
    static void createCoordinateSystem(
        const std::array<double, 3>& direction,
        std::array<double, 3>& u,
        std::array<double, 3>& v);
    
    // Configuration parameters
    int mesh_resolution_ = 32;
    int time_steps_ = 24;
    double albedo_ = 0.5;
    std::string kernels_path_ = "/Users/paha3326/software/cspice/data";
    
    // Embree ray tracing resources
    RTCDevice embree_device_ = nullptr;
    RTCScene embree_scene_ = nullptr;

    // Scene origin shift to improve numerical robustness (applied in Embree space)
    std::array<double,3> scene_origin_offset_{0.0, 0.0, 0.0};
    bool scene_has_offset_ = false;

    // Mesh topology flag: true if the CORE mesh is an OPEN patch (has boundary
    // edges belonging to exactly one triangle), false if watertight/closed (0
    // boundary edges). Computed once in initializeScene() over the core faces
    // (occluders excluded). Gates the analytic body-occlusion pre-check:
    //   - CLOSED mesh (e.g. 623 Chimaera): the Embree BVH holds the full body and
    //     self-occludes, so the analytic pre-check is DISABLED (it over-culls
    //     sunward grazing facets on shaped bodies). This matches the GPU path,
    //     which has never had a pre-check and is correct on closed bodies.
    //   - OPEN mesh (e.g. the lunar/Europa micro-patch): the BVH has no far-side
    //     geometry so shadow rays pass through the bulk body; the analytic body-
    //     sphere is the SOLE self-occlusion guard, so the pre-check runs. It culls
    //     a face only when the body sphere lies BETWEEN the face and the sun
    //     (forward ray-sphere hit at t>0) -- NOT gated by mu, and NOT a bare
    //     |fc x sun| < R line test, which would wrongly cull sunward-tilted facets
    //     just past the terminator. Applied to the CPU AND GPU paths identically.
    // See docs/illumination-model.md and memory
    // ptm-mu-gate-open-mesh-selfocclusion-fail.
    bool mesh_is_open_ = false;

    // Body-level occluding-sphere radius (NATIVE mesh units, same units as the
    // runtime face-center radius |fc|). Used by the OPEN-mesh analytic body-
    // occlusion pre-check as the sphere the shadow ray must clear. Computed once
    // in initializeScene() as min over core-face centroid radii -- the solid bulk
    // of the body reaches down to its lowest face, so a shadow ray whose closest
    // approach falls inside this sphere is blocked by the body. This REPLACES the
    // former per-face 0.998*|fc| inset, which put the occluder ~0.2% (thousands of
    // m on the Moon) BELOW the surface and left a ~3.6 deg blind band: on a near-
    // tangent micro-patch the grazing night-side ray cleared the sunken sphere and
    // leaked. min(|fc|) keys the sphere to the true body radius (relief-adaptive:
    // it is the lowest crater floor) so tangent grazing rays are caught, while no
    // face is ever culled by a sphere larger than its own radius (0 daytime loss).
    // 0 until the scene is built.
    double body_radius_occ_native_ = 0.0;

    // PTM-13: local-horizon (skyline) direct-shadow cull. horizon_direct_ is a macroscopic
    // running-max terrain-elevation grid built once in initializeScene() when use_horizon_cull_
    // is set (env PTM_ILLUM_HORIZON_CULL=1 or setHorizonCull(true)); it culls the grazing-sun
    // leak (spurious DIRECT sun inside PSRs) resolution-independently, where the single Embree
    // shadow ray hops the mm rim. Azimuth-specific. Default OFF -> byte-identical. See
    // BUGS.md PTM-13 and RUNBOOK-ptm13-validation.md.
    radiation::HorizonField horizon_direct_;
    bool   use_horizon_cull_ = false;
    double horizon_slack_deg_ = 0.25;   // skyline slack (tunable; acne<->leak borderline)
    double horizon_cell_scale_ = 1.0;   // height-grid cell = scale * median face pitch

    // PTM-13 P1: topological self-skip adjacency (CSR). face_skip_off_[c]..[c+1] index into
    // face_skip_nbr_ the sorted primIDs to skip for CORE face c (itself + <= selfskip_rings_ edge
    // neighbors). Built in initializeScene when use_topo_selfskip_ is set; empty otherwise. Default
    // OFF -> byte-identical (no build, no re-trace). See BUGS.md PTM-13, RUNBOOK-ptm13-validation.md.
    std::vector<uint32_t> face_skip_off_;
    std::vector<int>      face_skip_nbr_;
    bool use_topo_selfskip_ = false;
    int  selfskip_rings_ = 2;

    // PTM-13 rise-epsilon: per-core-face local edge length [m] (mean of the 3 triangle edges), the
    // facet scale for the out-of-plane-rise self-relief floor h_eps = rise_keps_ * face_edge_local_[c].
    // Built in initializeScene when use_rise_eps_ is set; empty otherwise. Default OFF -> byte-identical.
    // See BUGS.md PTM-13.
    std::vector<float> face_edge_local_;
    bool   use_rise_eps_ = false;
    double rise_keps_    = 0.2;   // h_eps = rise_keps_ * local edge (~0.5 mm at a 2.5 mm facet)
    // PTM-13 Class-2 fix: absolute self-relief floor override (meters). When >= 0 (env
    // PTM_ILLUM_RISE_HABS), traceOccludedRiseEps uses h_eps = rise_habs_ INSTEAD of the
    // edge-proportional rise_keps_*face_edge_local_[src]. Paul's scale ruling (2026-07-17): spatial
    // scale is arbitrary (no lateral conduction), so resolved mesh relief SHOULD shadow and an
    // edge-proportional floor is conceptually wrong; an absolute floor at numerical-noise scale (or 0
    // = exact source-triangle-only self-exclusion) kills the grazing occlusion leak with no acne.
    // Phase-2 knob table (scripts/ptm_illum_deepdive/phase2_knob_table_grazing.csv): 0.2*edge kills
    // 0.9% of 8854 float64-confirmed grazing occlusions; h_abs=0 kills 100% with 0/778 grazing acne.
    // Sentinel -1.0 => override OFF -> byte-identical (edge-proportional floor as before).
    double rise_habs_    = -1.0;

    // PTM-13 rise-epsilon MULTI-HIT refinement (OPTION 3, sub-mode of rise-eps): the single hit-POINT
    // rise dz misses a near-coplanar occluder that is CROSSED near its shared edge (dz<h_eps) yet whose
    // TRIANGLE rises >= h_eps at a farther vertex. When use_rise_eps2_ is set, traceOccludedRiseEps adds
    // an OR layer: a hit ALSO counts as a real occluder iff it is a genuine FLOAT64-INTERIOR crossing
    // (barycentric min > rise_bary_eps_, which rejects the float32 edge-clip artifacts that are the
    // entire acne source) AND the hit TRIANGLE's max-vertex out-of-plane rise >= h_eps. Strict superset
    // of the base rise-eps blocks (never removes a block), so ON only ever darkens more; requires the
    // per-core-face recentered triangle vertices (face_verts_local_), built only when set. Implies
    // use_rise_eps_ (base build + CPU routing). Default OFF -> byte-identical. STEP-1 measurement
    // (scripts/ptm_riseeps2_multihit/): near-coplanar leak recall 24%->84%, ZERO incremental acne on
    // open terrain AND steep grazing-lit walls. See BUGS.md PTM-13.
    std::vector<std::array<std::array<double, 3>, 3>> face_verts_local_;  // per core face: 3 recentered verts
    bool   use_rise_eps2_ = false;
    double rise_bary_eps_ = 0.0;  // interior-crossing barycentric floor (>0 => strictly inside the tri)

    // Occluder-only geometry (shadow ray-tracing); appended to the BVH in
    // initializeScene(). Never thermally solved / output. See
    // docs/extended-occluder-architecture.md.
    std::vector<std::array<double, 3>> occluder_vertices_;
    std::vector<std::array<int, 3>>    occluder_faces_;

    // PTM-18: seal small interior mesh holes for the grazing-sun flash (env PTM_ILLUM_SEAL_HOLES=1 or
    // setSealInteriorHoles). Fill triangles are appended to the shadow BVH like the occluder ring;
    // never thermally solved / output. Default OFF -> byte-identical. See BUGS.md PTM-18.
    bool seal_interior_holes_ = false;
    
    // Results storage
    std::vector<double> times_;  // Ephemeris times
    std::vector<std::string> time_strings_;  // Human-readable times
    // External ET grid (optional). When non-empty, overrides uniform grid construction.
    std::vector<double> external_ets_;
    std::vector<std::array<double, 3>> sun_positions_;
    std::vector<std::array<double, 3>> primary_positions_;
    std::vector<double> illumination_values_;  // Flattened: [face_idx * time_steps + time_idx]
    std::vector<double> primary_ir_values_;  // Flattened: [face_idx * time_steps + time_idx]

    // Add these to the private member variables
    bool include_primary_reflection_ = true;
    std::vector<double> direct_solar_values_;      // Direct solar only
    std::vector<double> primary_reflected_values_;  // Primary reflection only

    bool debugSpice();
    bool precomputed_positions_loaded_ = false;

    struct Ray {
        std::array<double, 3> origin;
        std::array<double, 3> direction;
        
        RTCRayHit toRTCRayHit() const {
            RTCRayHit rayhit;
            rayhit.ray.org_x = origin[0];
            rayhit.ray.org_y = origin[1];
            rayhit.ray.org_z = origin[2];
            rayhit.ray.dir_x = direction[0];
            rayhit.ray.dir_y = direction[1];
            rayhit.ray.dir_z = direction[2];
            rayhit.ray.tnear = 1.0f;
            rayhit.ray.tfar = 1.0e30f; // Far distance (replaced std::numeric_limits<float>::infinity())
            rayhit.ray.mask = 0xFFFFFFFF; // Use a mask to allow all rays
            rayhit.ray.flags = 0;
            rayhit.ray.time = 0.0f;
            
            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            
            return rayhit;
        }
    };

    // Add a helper method to trace a single ray
    bool traceRaySingle(const Ray& ray);

    // Face normal and center calculation
    std::array<double, 3> calculateFaceNormal(
        const std::array<double, 3>& v1,
        const std::array<double, 3>& v2,
        const std::array<double, 3>& v3);

    // Replace the simple debugPrint method with a template version that supports streaming
    template<typename T>
    void debugPrint(const T& message) const {
        if (debug_output_enabled_) {
            std::cout << message << std::endl;
        }
    }

    // Variadic template version to support multiple arguments with << operators
    template<typename T, typename... Args>
    void debugPrint(const T& first, const Args&... args) const {
        if (debug_output_enabled_) {
            std::cout << first;
            debugPrint(args...);
        }
    }

    // Add a new private member
    bool debug_output_enabled_ = false;

    // Flag to control progress bar visibility
    bool show_progress_ = true;

    // GPU ray tracing members
#ifdef PTM_GPU_OPENCL
    std::unique_ptr<gpu::opencl::BVHBuilder> gpu_bvh_;
    std::unique_ptr<gpu::opencl::BVHGPUPool> gpu_ray_pool_;
    bool use_gpu_ray_tracer_ = false;
    bool gpu_mesh_uploaded_ = false;

    // Multi-GPU dispatch config (defaults preserve single-GPU behaviour).
    int                gpu_devices_count_         = 1;
    std::vector<int>   gpu_device_ids_;
    int                gpu_inter_batch_pause_ms_  = 0;
    // BVH disk cache path (empty = disabled).
    std::string        bvh_cache_file_;

    // Initialize GPU BVH ray tracer
    bool initializeGPURayTracer();

    // Upload mesh to GPU for ray tracing
    void uploadMeshToGPU();

    // GPU-accelerated occlusion test (serialized, for compatibility)
    std::vector<uint8_t> traceOccludedGPU(
        const std::vector<std::array<double, 3>>& origins,
        const std::vector<std::array<double, 3>>& directions);

    // Batched GPU ray tracing for maximum throughput
    // Traces all rays in a single GPU kernel launch
    void traceOccludedBatchedGPU(
        const std::vector<float>& ray_org_x,
        const std::vector<float>& ray_org_y,
        const std::vector<float>& ray_org_z,
        const std::vector<float>& ray_dir_x,
        const std::vector<float>& ray_dir_y,
        const std::vector<float>& ray_dir_z,
        std::vector<uint8_t>& occluded);

    // GPU-optimized version of calculateDirectSolarBatchWithFiniteDisk
    // Uses batched GPU ray tracing for all faces
    void calculateDirectSolarBatchWithFiniteDiskGPU(
        const std::vector<std::array<double, 3>>& face_centers,
        const std::vector<std::array<double, 3>>& face_normals,
        const std::array<double, 3>& sun_position,
        const std::array<double, 3>& primary_position,
        std::vector<double>& direct_solar_values,
        int num_samples = 9);
#endif
};

/**
 * The core functions for illumination calculations are:
 * 
 * - calculateIlluminationOptimized: Main entry point for optimized calculations
 * - calculateDirectSolarBatchWithFiniteDisk: High performance batch calculation of direct solar flux
 * - calculatePrimaryReflectedBatch: Calculates primary-reflected solar flux
 * - calculatePrimaryIRBatch: Calculates primary's thermal IR flux
 *
 * Vector math utilities have been standardized for consistency.
 * Ray tracing uses Embree with SIMD optimizations where appropriate.
 */

} // namespace ptm