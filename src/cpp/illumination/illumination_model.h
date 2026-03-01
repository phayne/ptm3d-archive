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
    
    // Setup and configuration
    void setMeshResolution(int resolution);
    void setTimeSteps(int steps);
    void setAlbedo(double albedo);
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
        int num_samples = 9);  // Default to 9 samples (3x3 grid)

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
    
    // Results storage
    std::vector<double> times_;  // Ephemeris times
    std::vector<std::string> time_strings_;  // Human-readable times
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
    std::unique_ptr<gpu::opencl::BVHRayTracer> gpu_ray_tracer_;
    mutable std::mutex gpu_ray_tracer_mutex_;  // Protects GPU ray tracer calls (not thread-safe)
    bool use_gpu_ray_tracer_ = false;
    bool gpu_mesh_uploaded_ = false;

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