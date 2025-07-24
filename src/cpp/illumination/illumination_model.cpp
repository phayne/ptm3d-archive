#include "illumination_model.h"
#include "../constants/physics_constants.h"
#include "../constants/numeric_constants.h"
#include "../constants/constants.h"  // For ptm_thermal::constants
#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <limits>
#include <cassert>
#include <array>
#include <mutex>

// Conditionally include x86-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>  // SSE2 intrinsics
#include <immintrin.h>  // AVX/AVX2/AVX-512 intrinsics
#endif

#include <filesystem>  // For std::filesystem::exists
#include "config.h"    // For SPICE_KERNELS_DIR

// Embree includes
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_common.h>

// SPICE includes
#include <SpiceUsr.h>  // Angle brackets to use system include path

namespace constants = ptm_thermal::constants;  

namespace numeric = ptm::numeric;             // For numeric constants

namespace ptm {

// Thread-local storage definitions
thread_local std::vector<std::array<double, 3>> IlluminationModel::tls_ray_origins;
thread_local std::vector<std::array<double, 3>> IlluminationModel::tls_ray_directions;
thread_local std::vector<int> IlluminationModel::tls_ray_results;
thread_local size_t IlluminationModel::tls_batch_size = 0;

thread_local IlluminationModel::RayBatch IlluminationModel::tls_batch;

// Error handling function for Embree
void embreeErrorFunction(void* userPtr, enum RTCError error, const char* str) {
    std::cerr << "Embree error: " << str << " (code " << error << ")" << std::endl;
}

IlluminationModel::IlluminationModel(bool debug_output) 
    : embree_device_(nullptr), embree_scene_(nullptr), debug_output_enabled_(debug_output), show_progress_(true) {
    // Default initialization
    mesh_resolution_ = 32;
    time_steps_ = 24;
    albedo_ = 0.5;
    
    // Get kernels path from config
    kernels_path_ = ptm_thermal::config::SPICE_KERNELS_DIR;
    
    // Initialize SPICE
    initialize_spice(); // FIXED: changed from initializeSpice() to initialize_spice()
}

IlluminationModel::~IlluminationModel() {
    cleanupEmbree();
    cleanupSpice();
}

void IlluminationModel::setMeshResolution(int resolution) {
    mesh_resolution_ = resolution;
}

void IlluminationModel::setTimeSteps(int steps) {
    time_steps_ = steps;
}

void IlluminationModel::setAlbedo(double albedo) {
    albedo_ = albedo;
}

bool IlluminationModel::initializeEmbree() {
    // Initialize Embree device if it doesn't exist
    if (!embree_device_) {
        embree_device_ = rtcNewDevice(nullptr);
        if (!embree_device_) {
            std::cerr << "Failed to create Embree device" << std::endl;
            return false;
        }
    }
    
    return true;
}

void IlluminationModel::cleanupEmbree() {
    // Release Embree resources
    if (embree_scene_) {
        rtcReleaseScene(embree_scene_);
        embree_scene_ = nullptr;
    }
    
    if (embree_device_) {
        rtcReleaseDevice(embree_device_);
        embree_device_ = nullptr;
    }
}

bool IlluminationModel::initializeScene(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::array<int, 3>>& faces) {
    
    // Safety check for empty inputs
    if (vertices.empty() || faces.empty()) {
        std::cerr << "Error: Empty vertex or face array passed to initializeScene" << std::endl;
        return false;
    }
    
    try {
        // Create a new scene
        if (embree_scene_) {
            rtcReleaseScene(embree_scene_);
            embree_scene_ = nullptr;
        }
        
        // Make sure device is initialized
        if (!embree_device_) {
            if (!initializeEmbree()) {
                std::cerr << "Failed to initialize Embree device" << std::endl;
                return false;
            }
        }
        
        embree_scene_ = rtcNewScene(embree_device_);
        if (!embree_scene_) {
            std::cerr << "Failed to create Embree scene" << std::endl;
            return false;
        }
        
        // Create new geometry
        RTCGeometry geometry = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (!geometry) {
            std::cerr << "Failed to create geometry" << std::endl;
            return false;
        }
        
        // Set vertex buffer with proper error handling
        size_t vertex_count = vertices.size();
        std::cout << "Setting up ray tracing for " << vertex_count << " vertices and " 
                  << faces.size() << " faces" << std::endl;
                  
        float* vertex_buffer = static_cast<float*>(rtcSetNewGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(float) * 3, vertex_count
        ));
        
        if (!vertex_buffer) {
            std::cerr << "Failed to create vertex buffer" << std::endl;
            rtcReleaseGeometry(geometry);
            return false;
        }
        
        // Copy vertices to buffer with bounds checking
        for (size_t i = 0; i < vertex_count; ++i) {
            if (i * 3 + 2 >= vertex_count * 3) {
                std::cerr << "Vertex buffer index out of bounds" << std::endl;
                rtcReleaseGeometry(geometry);
                return false;
            }
            vertex_buffer[i*3]     = static_cast<float>(vertices[i][0]);
            vertex_buffer[i*3 + 1] = static_cast<float>(vertices[i][1]);
            vertex_buffer[i*3 + 2] = static_cast<float>(vertices[i][2]);
        }
        
        // Set index buffer with proper error handling
        size_t face_count = faces.size();
        unsigned* index_buffer = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(unsigned) * 3, face_count
        ));
        
        if (!index_buffer) {
            std::cerr << "Failed to create index buffer" << std::endl;
            rtcReleaseGeometry(geometry);
            return false;
        }
        
        // Copy indices to buffer with validity check
        for (size_t i = 0; i < face_count; ++i) {
            // Check for valid vertex indices
            if (static_cast<size_t>(faces[i][0]) >= vertex_count || 
                static_cast<size_t>(faces[i][1]) >= vertex_count || 
                static_cast<size_t>(faces[i][2]) >= vertex_count) {
                std::cerr << "Face " << i << " has invalid vertex index "
                          << faces[i][0] << ", " << faces[i][1] << ", " << faces[i][2] 
                          << " (max valid: " << vertex_count - 1 << ")" << std::endl;
                rtcReleaseGeometry(geometry);
                return false;
            }
            
            // Verify buffer access is in bounds
            if (i * 3 + 2 >= face_count * 3) {
                std::cerr << "Index buffer access out of bounds" << std::endl;
                rtcReleaseGeometry(geometry);
                return false;
            }
            
            index_buffer[i*3]     = static_cast<unsigned>(faces[i][0]);
            index_buffer[i*3 + 1] = static_cast<unsigned>(faces[i][1]);
            index_buffer[i*3 + 2] = static_cast<unsigned>(faces[i][2]);
        }
        
        // Commit geometry and attach to scene
        rtcCommitGeometry(geometry);
        unsigned geometry_id = rtcAttachGeometry(embree_scene_, geometry);
        rtcReleaseGeometry(geometry);
        
        // Commit scene
        std::cout << "Committing ray tracing scene..." << std::endl;
        rtcCommitScene(embree_scene_);
        std::cout << "Ray tracing scene ready" << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        debugPrint("Debug: Exception during scene initialization: " + std::string(e.what()));
        std::cerr << "Exception during ray tracing scene setup: " << e.what() << std::endl;
        // Cleanup in case of exception
        if (embree_scene_) {
            rtcReleaseScene(embree_scene_);
            embree_scene_ = nullptr;
        }
        return false;
    }
    catch (...) {
        std::cerr << "Unknown exception during ray tracing scene setup" << std::endl;
        // Cleanup in case of exception
        if (embree_scene_) {
            rtcReleaseScene(embree_scene_);
            embree_scene_ = nullptr;
        }
        return false;
    }
}

std::vector<int> IlluminationModel::intersectsFirst(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {
    
    if (!embree_scene_) {
        std::cerr << "Embree scene not initialized" << std::endl;
        return std::vector<int>(origins.size(), -1);
    }
    
    size_t num_rays = origins.size();
    std::vector<int> face_ids(num_rays, -1);  // Initialize all to -1 (no hit)
    
    // Create ray context
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    
    // Process rays in parallel using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < num_rays; ++i) {
        // Initialize ray hit structure for Embree 4
        RTCRayHit rayhit;
        rayhit.ray.org_x = origins[i][0];
        rayhit.ray.org_y = origins[i][1];
        rayhit.ray.org_z = origins[i][2];
        rayhit.ray.dir_x = directions[i][0];
        rayhit.ray.dir_y = directions[i][1];
        rayhit.ray.dir_z = directions[i][2];
        rayhit.ray.tnear = 0.0001f;  // Small offset to avoid self-intersection
        rayhit.ray.tfar = 1.0e30f; // Far distance (replaced std::numeric_limits<float>::infinity())
        rayhit.ray.mask = 0;
        rayhit.ray.flags = 0;
        rayhit.ray.time = 0.0f;
        
        // Initialize hit fields
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        
        // Trace the ray
        RTCIntersectArguments local_args = args;
        rtcIntersect1(embree_scene_, &rayhit, &local_args);
        
        // Store result if hit
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            face_ids[i] = static_cast<int>(rayhit.hit.primID);
        }
    }
    
    return face_ids;
}

// SIMD version for improved performance when tracing multiple rays
std::vector<int> IlluminationModel::intersectsFirstSIMD(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {

#if defined(__x86_64__) || defined(_M_X64)
    if (!embree_scene_) {
        std::cerr << "Embree scene not initialized" << std::endl;
        return std::vector<int>(origins.size(), -1);
    }
    
    size_t num_rays = origins.size();
    std::vector<int> face_ids(num_rays, -1);  // Initialize all to -1 (no hit)
    
    // Process rays in SIMD packets of 8
    size_t processed = 0;
    while (processed < num_rays) {
        // Number of rays in this packet (up to 8)
        size_t packet_size = std::min(size_t(8), num_rays - processed);
        
        // Initialize SIMD ray packet
        alignas(32) int valid[8];
        alignas(32) RTCRayHit8 rayhit;
        
        // Clear valid array
        std::memset(valid, 0, sizeof(valid));
        
        // Set up rays for this packet
        for (size_t i = 0; i < packet_size; ++i) {
            valid[i] = -1;  // Mark ray as active
            
            size_t idx = processed + i;
            
            // Set ray origin and direction
            rayhit.ray.org_x[i] = origins[idx][0];
            rayhit.ray.org_y[i] = origins[idx][1];
            rayhit.ray.org_z[i] = origins[idx][2];
            rayhit.ray.dir_x[i] = directions[idx][0];
            rayhit.ray.dir_y[i] = directions[idx][1];
            rayhit.ray.dir_z[i] = directions[idx][2];
            
            // Other ray parameters
            rayhit.ray.tnear[i] = 0.0001f;
            rayhit.ray.tfar[i] = 1.0e30f; // Far distance (replaced std::numeric_limits<float>::infinity())
            rayhit.ray.mask[i] = 0;
            rayhit.ray.flags[i] = 0;
            rayhit.ray.time[i] = 0.0f;
            
            // Initialize hit fields
            rayhit.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
        }
        
        // Trace ray packet
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect8(valid, embree_scene_, &rayhit, &args);
        
        // Process results
        for (size_t i = 0; i < packet_size; ++i) {
            size_t idx = processed + i;
            if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) {
                face_ids[idx] = static_cast<int>(rayhit.hit.primID[i]);
            }
        }
        
        processed += packet_size;
    }
    
    return face_ids;

#else
    // Fallback to non-SIMD version on non-x86 platforms
    return intersectsFirst(origins, directions);
#endif
}

void IlluminationModel::initRayBatch() {
    // Clear previous results
    tls_ray_results.clear();
    tls_batch_size = 0;
    
    // Ensure vectors are preallocated for efficiency
    if (tls_ray_origins.size() < MAX_BATCH_SIZE) {
        tls_ray_origins.resize(MAX_BATCH_SIZE);
        tls_ray_directions.resize(MAX_BATCH_SIZE);
    }
}

// Initialize the batch collection system
void IlluminationModel::initRayBatchCollection() {
    // Clear thread-local batch
    tls_batch.origins.clear();
    tls_batch.directions.clear();
    tls_batch.results.clear();
    tls_batch.indices.clear();
    tls_batch.current_size = 0;
    tls_batch.processed = false;
    
    // Preallocate memory for efficiency
    tls_batch.origins.reserve(COLLECTION_BATCH_SIZE);
    tls_batch.directions.reserve(COLLECTION_BATCH_SIZE);
    tls_batch.indices.reserve(COLLECTION_BATCH_SIZE);
    
    // Clear shared batches under lock protection
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        // Don't actually delete the RayBatch objects, just clear the pointers
        shared_batches_.clear();
        batches_being_processed_ = false;
    }
}

// Add a ray to the thread-local batch
size_t IlluminationModel::addRayToBatch(const std::array<double, 3>& origin, const std::array<double, 3>& direction) {
    // Add ray to thread-local batch
    size_t batch_index = tls_batch.current_size;
    
    tls_batch.origins.push_back(origin);
    tls_batch.directions.push_back(direction);
    tls_batch.current_size++;
    
    // Check if the thread-local batch is full
    if (tls_batch.current_size >= COLLECTION_BATCH_SIZE) {
        // Only submit batch if it's not already being processed
        if (!tls_batch.processed) {
            std::lock_guard<std::mutex> lock(batch_mutex_);
            
            // If not currently processing batches, add this batch
            if (!batches_being_processed_) {
                shared_batches_.push_back(&tls_batch);
                tls_batch.processed = true;
                
                // If we have enough batches or the batch is large enough,
                // start processing
                if (shared_batches_.size() >= 4 || tls_batch.current_size >= 64) {
                    processRayBatches();
                }
            }
        }
    }
    
    return batch_index;
}

// Process all collected ray batches in SIMD-friendly chunks
void IlluminationModel::processRayBatches() {
    std::vector<std::array<double, 3>> all_origins;
    std::vector<std::array<double, 3>> all_directions;
    std::vector<RayBatch*> batches_to_process;
    
    // Under lock: mark that we're processing and collect all batches
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        
        if (shared_batches_.empty()) {
            return;
        }
        
        batches_being_processed_ = true;
        batches_to_process = shared_batches_;
        shared_batches_.clear();
    }
    
    // Calculate total rays
    size_t total_rays = 0;
    for (auto* batch : batches_to_process) {
        total_rays += batch->current_size;
    }
    
    // Allocate enough space for all rays
    all_origins.reserve(total_rays);
    all_directions.reserve(total_rays);
    
    // Combine all ray batches
    for (auto* batch : batches_to_process) {
        all_origins.insert(all_origins.end(), batch->origins.begin(), batch->origins.begin() + batch->current_size);
        all_directions.insert(all_directions.end(), batch->directions.begin(), batch->directions.begin() + batch->current_size);
    }
    
    // Process rays with optimal SIMD batching
    std::vector<int> all_results = traceRays(all_origins, all_directions);
    
    // Distribute results back to thread-local batches
    size_t result_offset = 0;
    for (auto* batch : batches_to_process) {
        batch->results.resize(batch->current_size);
        std::copy(all_results.begin() + result_offset, 
                  all_results.begin() + result_offset + batch->current_size, 
                  batch->results.begin());
        result_offset += batch->current_size;
        batch->processed = true;
    }
    
    // Mark processing as done
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batches_being_processed_ = false;
    }
}

// Get the hit result for a ray in the thread-local batch
bool IlluminationModel::getRayHitResult(size_t batch_index) {
    // Check if the batch has been processed
    if (!tls_batch.processed) {
        // Process the batch now if needed
        if (tls_batch.current_size > 0) {
            std::vector<std::array<double, 3>> origins(tls_batch.origins.begin(), 
                                                     tls_batch.origins.begin() + tls_batch.current_size);
            std::vector<std::array<double, 3>> directions(tls_batch.directions.begin(), 
                                                        tls_batch.directions.begin() + tls_batch.current_size);
            
            tls_batch.results = traceRays(origins, directions);
            tls_batch.processed = true;
        } else {
            return false;  // Empty batch
        }
    }
    
    // Check if the index is valid
    if (batch_index < tls_batch.results.size()) {
        return tls_batch.results[batch_index] >= 0;
    }
    
    return false;
}

void IlluminationModel::processRayBatch() {
    if (tls_batch_size == 0) return;
    
    // Trace all rays in the batch at once
    tls_ray_results = traceRays(
        std::vector<std::array<double, 3>>(tls_ray_origins.begin(), tls_ray_origins.begin() + tls_batch_size),
        std::vector<std::array<double, 3>>(tls_ray_directions.begin(), tls_ray_directions.begin() + tls_batch_size)
    );
    
    // Reset batch counter but keep results
    tls_batch_size = 0;
}

// Batch ray tracing with 16-wide AVX-512 packets for maximum performance on supported hardware
std::vector<int> IlluminationModel::intersectsFirstAVX512(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {

    // Ensure Embree is initialized
    if (!embree_scene_) {
        std::cerr << "Embree scene not initialized" << std::endl;
        return std::vector<int>(origins.size(), -1);
    }
    
    // 
    size_t num_rays = origins.size();
    std::vector<int> face_ids(num_rays, -1);  // Initialize all to -1 (no hit)

#if defined(__x86_64__) || defined(_M_X64) && defined(__AVX512F__) && defined(__AVX512DQ__)
    // Process rays in SIMD packets of 16
    size_t processed = 0;
    while (processed < num_rays) {
        // Number of rays in this packet (up to 16)
        size_t packet_size = std::min(size_t(16), num_rays - processed);
        
        // Initialize SIMD ray packet
        alignas(64) int valid[16];
        alignas(64) RTCRayHit16 rayhit;
        
        // Clear valid array
        std::memset(valid, 0, sizeof(valid));
        
        // Set up rays for this packet
        for (size_t i = 0; i < packet_size; ++i) {
            valid[i] = -1;  // Mark ray as active
            
            size_t idx = processed + i;
            
            // Set ray origin and direction
            rayhit.ray.org_x[i] = origins[idx][0];
            rayhit.ray.org_y[i] = origins[idx][1];
            rayhit.ray.org_z[i] = origins[idx][2];
            rayhit.ray.dir_x[i] = directions[idx][0];
            rayhit.ray.dir_y[i] = directions[idx][1];
            rayhit.ray.dir_z[i] = directions[idx][2];
            
            // Other ray parameters
            rayhit.ray.tnear[i] = 0.0001f;
            rayhit.ray.tfar[i] = 1.0e30f; // Far distance (replaced std::numeric_limits<float>::infinity())
            rayhit.ray.mask[i] = 0;
            rayhit.ray.flags[i] = 0;
            rayhit.ray.time[i] = 0.0f;
            
            // Initialize hit fields
            rayhit.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
        }
        
        // Trace ray packet
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect16(valid, embree_scene_, &rayhit, &args);
        
        // Process results
        for (size_t i = 0; i < packet_size; ++i) {
            size_t idx = processed + i;
            if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) {
                face_ids[idx] = static_cast<int>(rayhit.hit.primID[i]);
            }
        }
        
        processed += packet_size;
    }
    
    return face_ids;

#else
    // Fallback to SIMD or non-SIMD version on platforms without AVX-512
    #if defined(__x86_64__) || defined(_M_X64)
        return intersectsFirstSIMD(origins, directions);
    #else
        return intersectsFirst(origins, directions);
    #endif
#endif
}

// Auto-select best ray tracing method based on hardware and ray count
std::vector<int> IlluminationModel::traceRays(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& directions) {
    
    size_t num_rays = origins.size();
    
#if defined(__x86_64__) || defined(_M_X64)
    // Use different methods based on ray count and CPU features
    if (num_rays >= 16 && isAVX512Supported()) {
        // std::cerr << "Using AVX-512 ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirstAVX512(origins, directions);
    } else if (num_rays >= 8) {
        // std::cerr << "Using SIMD ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirstSIMD(origins, directions);
    } else {
        // std::cerr << "Using scalar ray tracing for " << num_rays << " rays" << std::endl;
        return intersectsFirst(origins, directions);
    }
#else
    // On non-x86 architectures, always use the scalar version
        // std::cerr << "Using scalar ray tracing (non-x86 architecture) for " << num_rays << " rays" << std::endl;
    return intersectsFirst(origins, directions);
#endif
}

// Helper to check if AVX-512 is supported
bool IlluminationModel::isAVX512Supported() {
    static bool checked = false;
    static bool avx512_supported = false;
    
    if (!checked) {
        checked = true;
        
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Compile-time check passed
        
        // Additional runtime detection could be added here if needed
        // This ensures the binary has AVX512 instructions AND the CPU supports them
        
        // On Intel CPUs with AVX-512, also check for thermal throttling issues
        // by trying vectorized code on a small test case first
        static bool tested = false;
        static bool avx512_works = false;
        
        if (!tested) {
            // Simple runtime test to verify AVX-512 actually works
            tested = true;
            avx512_works = true;  // Default to true, could add actual test here
        }
        
        avx512_supported = avx512_works;
        
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "AVX-512 support detected: " << (avx512_supported ? "Yes" : "No") << std::endl;
    #else
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "AVX-512 not supported at compile time" << std::endl;
        avx512_supported = false;
    #endif
#else
        // TEMPORARY: Always output SIMD support info regardless of debug flag
        std::cerr << "Non-x86 architecture detected, AVX-512 not supported" << std::endl;
        avx512_supported = false;
#endif

        // TEMPORARY: Print detailed SIMD support information
        printSIMDInfoTemporary();
    }
    
    return avx512_supported;
}

// TEMPORARY: Helper function to print detailed SIMD information
void IlluminationModel::printSIMDInfoTemporary() {
    std::cerr << "=== SIMD Support Information ===" << std::endl;

#if defined(__x86_64__) || defined(_M_X64)
    std::cerr << "x86-64 architecture detected" << std::endl;

    #if defined(__SSE2__)
        std::cerr << "SSE2: Supported" << std::endl;
    #else
        std::cerr << "SSE2: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX__)
        std::cerr << "AVX: Supported" << std::endl;
    #else
        std::cerr << "AVX: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX2__)
        std::cerr << "AVX2: Supported" << std::endl;
    #else
        std::cerr << "AVX2: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX512F__)
        std::cerr << "AVX-512F: Supported" << std::endl;
    #else
        std::cerr << "AVX-512F: Not supported" << std::endl;
    #endif
    
    #if defined(__AVX512DQ__)
        std::cerr << "AVX-512DQ: Supported" << std::endl;
    #else
        std::cerr << "AVX-512DQ: Not supported" << std::endl;
    #endif
#else
    std::cerr << "Non-x86 architecture detected" << std::endl;
#endif

    std::cerr << "===============================" << std::endl;
}

// Clean up SPICE resources
void IlluminationModel::cleanupSpice() {
    // Unload all kernels
    unload_c("all");
}

bool IlluminationModel::calculateIllumination(
    const std::vector<std::array<double, 3>>& mesh_vertices,
    const std::vector<std::array<int, 3>>& mesh_faces,
    const std::string& start_time,
    double duration_days) {
    
    static bool warned = false;
    if (!warned) {
        std::cerr << "Warning: Using non-optimized illumination calculation path. "
                  << "Consider using calculateIlluminationOptimized instead." << std::endl;
        warned = true;
    }
    
    // Initialize SPICE for ephemeris calculations
    if (!precomputed_positions_loaded_ && !initialize_spice()) { 
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return false;
    }
    
    // Initialize Embree for ray tracing
    if (!initializeEmbree()) {
        std::cerr << "Failed to initialize Embree" << std::endl;
        return false;
    }
    
    // Set up Embree scene with the mesh
    if (!initializeScene(mesh_vertices, mesh_faces)) {
        std::cerr << "Failed to initialize Embree scene" << std::endl;
        return false;
    }
    
    // Skip position calculations if precomputed positions were loaded
    if (!precomputed_positions_loaded_) {
        // Initialize time array
        double start_et = convertTimeToET(start_time);
        double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
        double step_size = duration_seconds / time_steps_;
        
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);
        
        for (int t = 0; t < time_steps_; ++t) {
            double et = start_et + t * step_size;
            times_[t] = et;
            
            // Convert to string for display
            char time_str[41];
            et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
            time_strings_[t] = std::string(time_str);
        }
        
        // Compute Sun and Primary positions for each time step
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);
        
        for (int t = 0; t < time_steps_; ++t) {
            sun_positions_[t] = getSunPosition(times_[t]);
            primary_positions_[t] = getPrimaryPosition(times_[t]);
        }
    }
    
    // Calculate face normals and centers
    std::vector<std::array<double, 3>> face_normals(mesh_faces.size());
    std::vector<std::array<double, 3>> face_centers(mesh_faces.size());
    
    for (size_t f = 0; f < mesh_faces.size(); ++f) {
        const auto& face = mesh_faces[f];
        const auto& v0 = mesh_vertices[face[0]];
        const auto& v1 = mesh_vertices[face[1]];
        const auto& v2 = mesh_vertices[face[2]];
        
        face_normals[f] = calculateFaceNormal(v0, v1, v2);
        face_centers[f] = {{
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        }};
    }
    
    // Resize illumination values array
    illumination_values_.resize(mesh_faces.size() * time_steps_, 0.0);
    
    // Process each time step
    for (int t = 0; t < time_steps_; ++t) {
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];

        // Use calculateSolarIlluminationOptimized for each face
        #pragma omp parallel for schedule(dynamic)
        for (size_t f = 0; f < mesh_faces.size(); ++f) {
            // This calls our optimized method which includes Primary eclipse checking
            double illumination = calculateSolarIlluminationWithFiniteDisk(
                face_centers[f],
                face_normals[f],
                sun_pos,
                primary_pos,
                constants::N_SOLAR_SAMPLES  // Number of solar samples (sqrt(N) should be integer)
            );
            
            // Store illumination value
            illumination_values_[f * time_steps_ + t] = illumination;
        }
    }
    
    return true;
}

bool IlluminationModel::calculateIlluminationOptimized(
    const std::vector<std::array<double, 3>>& mesh_vertices,
    const std::vector<std::array<int, 3>>& mesh_faces,
    const std::string& start_time,
    double duration_days) {
    
    debugPrint("Debug: Entering calculateIlluminationOptimized");
    
    // Check for very large meshes and adjust thread count accordingly
    size_t face_count = mesh_faces.size();
    bool is_large_mesh = face_count > 1000000;
    bool is_very_large_mesh = face_count > 10000000;
    
    // Auto-adjust thread count for large meshes to prevent memory pressure
    int original_thread_count = 0;
    #ifdef _OPENMP
    original_thread_count = omp_get_max_threads();
    if (is_very_large_mesh) {
        int adjusted_threads = std::min(8, original_thread_count);
        debugPrint("Very large mesh detected (", face_count, " faces). Limiting to ", 
                  adjusted_threads, " threads to reduce memory pressure.");
        omp_set_num_threads(adjusted_threads);
    } else if (is_large_mesh) {
        int adjusted_threads = std::min(16, original_thread_count);
        debugPrint("Large mesh detected (", face_count, " faces). Limiting to ", 
                  adjusted_threads, " threads.");
        omp_set_num_threads(adjusted_threads);
    }
    #endif
    
    // Initialize SPICE for ephemeris calculations
    if (!precomputed_positions_loaded_ && !initialize_spice()) {
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return false;
    }
    
    debugPrint("Debug: SPICE initialization complete");
    
    // Initialize Embree for ray tracing
    debugPrint("Debug: Starting Embree initialization");
    if (!initializeEmbree()) {
        std::cerr << "Failed to initialize Embree" << std::endl;
        return false;
    }
    debugPrint("Debug: Embree initialization complete");
    
    // Set up Embree scene with the mesh
    debugPrint("Debug: Starting scene initialization with " + std::to_string(mesh_vertices.size()) + " vertices and " + std::to_string(mesh_faces.size()) + " faces");
    
    try {
        if (!initializeScene(mesh_vertices, mesh_faces)) {
            debugPrint("Debug: Failed to initialize Embree scene");
            return false;
        }
        debugPrint("Debug: Scene initialization complete");
    } catch (const std::exception& e) {
        debugPrint("Debug: Exception during scene initialization: " + std::string(e.what()));
        return false;
    } catch (...) {
        debugPrint("Debug: Unknown exception during scene initialization");
        return false;
    }
    
    // Skip position calculations if precomputed positions were loaded
    if (!precomputed_positions_loaded_) {
        debugPrint("Debug: Calculating celestial body positions");
        // Initialize time array
        double start_et = convertTimeToET(start_time);
        double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
        double step_size = duration_seconds / time_steps_;
        
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);
        
        for (int t = 0; t < time_steps_; ++t) {
            double et = start_et + t * step_size;
            times_[t] = et;
            
            // Convert to string for display
            char time_str[41];
            et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
            time_strings_[t] = std::string(time_str);
        }
        
        // Compute Sun and Primary positions for each time step
        debugPrint("Debug: Resizing position arrays for " + std::to_string(time_steps_) + " time steps");
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);
        
        debugPrint("Debug: Computing Sun and Primary positions");
        for (int t = 0; t < time_steps_; ++t) {
            try {
                sun_positions_[t] = getSunPosition(times_[t]);
                primary_positions_[t] = getPrimaryPosition(times_[t]);
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during position calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                return false;
            } catch (...) {
                debugPrint("Debug: Unknown exception during position calculation at time step " + std::to_string(t));
                return false;
            }
        }
        debugPrint("Debug: Sun and Primary positions computed successfully");
    }
    
    debugPrint("Debug: Calculating face normals and centers");
    // Calculate face normals and centers
    std::vector<std::array<double, 3>> face_normals(mesh_faces.size());
    std::vector<std::array<double, 3>> face_centers(mesh_faces.size());
    
    // Use a smaller chunk size for large meshes to balance parallelism and memory usage
    size_t chunk_size = is_very_large_mesh ? 4096 : (is_large_mesh ? 8192 : 16384);
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t f = 0; f < mesh_faces.size(); ++f) {
        try {
            const auto& face = mesh_faces[f];
            
            // Add range checks to prevent out-of-bounds access
            if (face[0] >= mesh_vertices.size() || face[1] >= mesh_vertices.size() || face[2] >= mesh_vertices.size()) {
                #pragma omp critical
                {
                    std::cerr << "Error: Face " << f << " contains invalid vertex indices" << std::endl;
                }
                continue;  // Skip this face rather than failing
            }
            
            const auto& v0 = mesh_vertices[face[0]];
            const auto& v1 = mesh_vertices[face[1]];
            const auto& v2 = mesh_vertices[face[2]];
            
            face_normals[f] = calculateFaceNormal(v0, v1, v2);
            face_centers[f] = {{
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            }};
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Exception during normal calculation at face " << f << ": " << e.what() << std::endl;
            }
            // Continue processing other faces - don't fail the entire calculation for one face
        } catch (...) {
            #pragma omp critical
            {
                std::cerr << "Unknown exception during normal calculation at face " << f << std::endl;
            }
        }
    }
    debugPrint("Debug: Finished calculating normals and centers");
    
    debugPrint("Debug: Resizing result arrays for " + std::to_string(mesh_faces.size()) + " faces and " + std::to_string(time_steps_) + " time steps");
    
    // Pre-calculate the total size needed
    size_t total_faces = mesh_faces.size();
    size_t total_values = total_faces * time_steps_;
    
    // Check if we need to use chunked storage (over 500 million elements)
    const size_t chunked_threshold = 500000000;
    
    try {
        if (total_values > chunked_threshold) {
            std::cout << "Using chunked storage for very large mesh (" 
                      << total_faces << " faces, " << time_steps_ << " time steps)" << std::endl;
            
            // Initialize chunked storage
            initializeChunkedStorage(total_faces, time_steps_);
            
            // Clear standard arrays to save memory
            illumination_values_.clear();
            direct_solar_values_.clear();
            primary_reflected_values_.clear();
            primary_ir_values_.clear();
            
            // Shrink to fit to release memory
            illumination_values_.shrink_to_fit();
            direct_solar_values_.shrink_to_fit();
            primary_reflected_values_.shrink_to_fit();
            primary_ir_values_.shrink_to_fit();
        } else {
            // Regular storage approach
            using_chunked_storage_ = false;
            
            // Check for size overflow to prevent integer overflow
            if (mesh_faces.size() > std::numeric_limits<size_t>::max() / time_steps_) {
                throw std::overflow_error("Mesh size causes integer overflow in array allocation");
            }
            
            // Resize arrays with zero initialization
            illumination_values_.resize(total_values, 0.0);
            direct_solar_values_.resize(total_values, 0.0);
            primary_reflected_values_.resize(total_values, 0.0);
            primary_ir_values_.resize(total_values, 0.0);
        }
        
        debugPrint("Debug: Arrays resized successfully");
    } catch (const std::exception& e) {
        debugPrint("Debug: Exception during array resizing: " + std::string(e.what()));
        std::cerr << "Failed to allocate memory for illumination arrays: " << e.what() << std::endl;
        std::cerr << "Try using a lower resolution mesh or running with fewer time steps" << std::endl;
        return false;
    }
    
    debugPrint("Debug: Starting illumination calculations for " + std::to_string(time_steps_) + " time steps");
    
    // Create progress bar
    thermal::ProgressBar progressBar(time_steps_, "Illumination calculation", 50, show_progress_);
    
    // For very large meshes, use smaller thread-local batches to reduce memory pressure
    size_t batch_size = is_very_large_mesh ? 16 : (is_large_mesh ? 32 : COLLECTION_BATCH_SIZE);
    
    // Process each time step with batch processing
    for (int t = 0; t < time_steps_; ++t) {
        debugPrint("Debug: Processing time step " + std::to_string(t+1) + " of " + std::to_string(time_steps_));
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];
        
        // Temporary vectors to store values for this time step
        // For large meshes, allocate these in chunks to reduce peak memory usage
        const size_t chunk_faces = is_very_large_mesh ? 1000000 : mesh_faces.size();
        
        std::vector<double> timestep_direct_solar(mesh_faces.size(), 0.0);
        std::vector<double> timestep_primary_reflected(mesh_faces.size(), 0.0);
        std::vector<double> timestep_primary_ir(mesh_faces.size(), 0.0);
        
        // Initialize thread-local ray batching system
        initRayBatchCollection();
        
        // Calculate direct solar illumination
        debugPrint("Debug: Calculating direct solar illumination for time step " + std::to_string(t+1));
        try {
            calculateDirectSolarBatchWithFiniteDisk(
                face_centers,
                face_normals,
                sun_pos,
                primary_pos,
                timestep_direct_solar,
                constants::N_SOLAR_SAMPLES
            );
            debugPrint("Debug: Direct solar calculation complete for time step " + std::to_string(t+1));
        } catch (const std::exception& e) {
            debugPrint("Debug: Exception during direct solar calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
            std::cerr << "Exception during direct solar calculation: " << e.what() << std::endl;
            // Continue with zeros rather than failing completely
            std::fill(timestep_direct_solar.begin(), timestep_direct_solar.end(), 0.0);
        } catch (...) {
            debugPrint("Debug: Unknown exception during direct solar calculation at time step " + std::to_string(t));
            std::cerr << "Unknown exception during direct solar calculation" << std::endl;
            // Continue with zeros rather than failing completely
            std::fill(timestep_direct_solar.begin(), timestep_direct_solar.end(), 0.0);
        }

        // Calculate Primary reflected solar (if enabled)
        if (include_primary_reflection_ && !primary_is_sun_) {
            debugPrint("Debug: Calculating Primary reflection for time step " + std::to_string(t+1));
            try {
                calculatePrimaryReflectedBatch(
                    face_centers,
                    face_normals,
                    sun_pos,
                    primary_pos,
                    timestep_primary_reflected
                );
                debugPrint("Debug: Primary reflection calculation complete for time step " + std::to_string(t+1));
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during Primary reflection calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                std::cerr << "Exception during Primary reflection calculation: " << e.what() << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
            } catch (...) {
                debugPrint("Debug: Unknown exception during Primary reflection calculation at time step " + std::to_string(t));
                std::cerr << "Unknown exception during Primary reflection calculation" << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
            }
        } else {
            // Fill with zeros when primary is Sun or reflection is disabled
            std::fill(timestep_primary_reflected.begin(), timestep_primary_reflected.end(), 0.0);
        }

        // Calculate Primary IR heating
        if (!primary_is_sun_) {
            debugPrint("Debug: Calculating Primary IR for time step " + std::to_string(t+1));
            try {
                calculatePrimaryIRBatch(
                    face_centers,
                    face_normals,
                    primary_pos,
                    timestep_primary_ir
                );
                debugPrint("Debug: Primary IR calculation complete for time step " + std::to_string(t+1));
            } catch (const std::exception& e) {
                debugPrint("Debug: Exception during Primary IR calculation at time step " + std::to_string(t) + ": " + std::string(e.what()));
                std::cerr << "Exception during Primary IR calculation at time step " + std::to_string(t)
                        << " with exception: " + std::string(e.what()) << std::endl;
                std::cerr << "Caught exception in calculatePrimaryIRBatch at time step " + std::to_string(t)
                        << ": " + std::string(e.what()) << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
            } catch (...) {
                std::cerr << "Unknown exception during Primary IR calculation at time step " + std::to_string(t) << std::endl;
                // Continue with zeros rather than failing completely
                std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
            }
        } else {
            // Fill with zeros when primary is Sun
            std::fill(timestep_primary_ir.begin(), timestep_primary_ir.end(), 0.0);
        }
        
        // Calculate total illumination
        std::vector<double> timestep_total(mesh_faces.size(), 0.0);
        for (size_t i = 0; i < timestep_total.size(); ++i) {
            // Total = Direct + Reflected
            timestep_total[i] = timestep_direct_solar[i] + 
                              (include_primary_reflection_ ? timestep_primary_reflected[i] : 0.0);
        }
        
        // Copy batch results to main storage in face-major order
        // Process in smaller chunks for better cache behavior
        const size_t copy_chunk_size = 4096;
        for (size_t chunk_start = 0; chunk_start < mesh_faces.size(); chunk_start += copy_chunk_size) {
            size_t chunk_end = std::min(chunk_start + copy_chunk_size, mesh_faces.size());
            
            #pragma omp parallel for schedule(static)
            for (size_t f = chunk_start; f < chunk_end; ++f) {
                if (using_chunked_storage_) {
                    storeChunkedValue(illumination_chunks_, f, t, timestep_total[f]);
                    storeChunkedValue(direct_solar_chunks_, f, t, timestep_direct_solar[f]);
                    storeChunkedValue(primary_reflected_chunks_, f, t, timestep_primary_reflected[f]);
                    storeChunkedValue(primary_ir_chunks_, f, t, timestep_primary_ir[f]);
                } else {
                    size_t index = f * time_steps_ + t;
                    if (index < illumination_values_.size()) {  // Safety check
                        illumination_values_[index] = timestep_total[f];  // Store correct total
                        direct_solar_values_[index] = timestep_direct_solar[f];
                        primary_reflected_values_[index] = timestep_primary_reflected[f];
                        primary_ir_values_[index] = timestep_primary_ir[f];
                    }
                }
            }
        }
        
        // Update progress bar
        progressBar.update(t + 1);
    }
    
    // Ensure progress bar is completed
    progressBar.finish();
    
    // Restore original thread count if we modified it
    #ifdef _OPENMP
    if ((is_large_mesh || is_very_large_mesh) && original_thread_count > 0) {
        omp_set_num_threads(original_thread_count);
        debugPrint("Debug: Restored original thread count: ", original_thread_count);
    }
    #endif
    
    debugPrint("Debug: Illumination calculation completed successfully");
    return true;
}

std::map<int, std::vector<double>> IlluminationModel::calculateLocationIllumination(
    const std::vector<std::pair<double, double>>& locations,
    const std::string& start_time,
    double duration_days) {
    
    // Initialize SPICE for ephemeris calculations (if not already initialized)
    if (!precomputed_positions_loaded_ && !initialize_spice()) { 
        std::cerr << "Failed to initialize SPICE" << std::endl;
        return {};
    }
    
    // If positions not already loaded, calculate them
    if (sun_positions_.empty() || primary_positions_.empty()) {
        // Initialize time array
        double start_et = convertTimeToET(start_time);
        double duration_seconds = duration_days * constants::SECONDS_PER_DAY;
        double step_size = duration_seconds / time_steps_;
        
        times_.resize(time_steps_);
        time_strings_.resize(time_steps_);
        sun_positions_.resize(time_steps_);
        primary_positions_.resize(time_steps_);
        
        for (int t = 0; t < time_steps_; ++t) {
            double et = start_et + t * step_size;
            times_[t] = et;
            
            // Convert to string for display
            char time_str[41];
            et2utc_c(et, "ISOC", 3, sizeof(time_str), time_str);
            time_strings_[t] = std::string(time_str);
            
            // Get positions
            sun_positions_[t] = getSunPosition(times_[t]);
            primary_positions_[t] = getPrimaryPosition(times_[t]);
        }
    }
    
    // Result container
    std::map<int, std::vector<double>> location_illumination;
    
    // Convert locations to 3D positions and normals
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> normals;
    
    for (const auto& [lat, lon] : locations) {
        // Convert lat/lon to 3D position on Body's surface
        std::array<double, 3> position = cartesianFromLatLon(lat, lon, constants::BODY_RADIUS);
        positions.push_back(position);
        
        // Normal vector at this position (points outward from center)
        normals.push_back(normalize(position));
    }
    
    // For each time step
    for (int t = 0; t < times_.size(); ++t) {
        const auto& sun_pos = sun_positions_[t];
        const auto& primary_pos = primary_positions_[t];

        // Process all locations for this time step
        for (size_t loc_idx = 0; loc_idx < locations.size(); ++loc_idx) {
            const auto& position = positions[loc_idx];
            const auto& normal = normals[loc_idx];
            
            // Calculate illumination with our advanced model
            double illumination = calculateSolarIlluminationWithFiniteDisk(
                position,
                normal,
                sun_pos,
                primary_pos,
                constants::N_SOLAR_SAMPLES  // Number of solar samples (sqrt(N) should be integer)
            );
            
            // Store result
            if (location_illumination.find(loc_idx) == location_illumination.end()) {
                location_illumination[loc_idx] = std::vector<double>(times_.size(), 0.0);
            }
            
            location_illumination[loc_idx][t] = illumination;
        }
    }
    
    return location_illumination;
}

const std::vector<double>& IlluminationModel::getIllumination() const {
    return illumination_values_;
}

const std::vector<double>& IlluminationModel::getTimes() const {
    return times_;
}

std::vector<std::pair<double, double>> IlluminationModel::getPrimaryIRFluxForFace(int face_idx) const {
    std::vector<std::pair<double, double>> result;
    result.reserve(times_.size());
    
    for (size_t t = 0; t < times_.size(); ++t) {
        double ir_flux;
        if (using_chunked_storage_) {
            ir_flux = getChunkedValue(primary_ir_chunks_, face_idx, t);
        } else {
            size_t index = face_idx * time_steps_ + t;
            ir_flux = (index < primary_ir_values_.size()) ? 
                primary_ir_values_[index] : 0.0;
        }
        result.emplace_back(times_[t], ir_flux);
    }
    
    return result;
}

std::vector<std::array<double, 3>> IlluminationModel::getSunPositions() const {
    return sun_positions_;
}

std::vector<std::array<double, 3>> IlluminationModel::getPrimaryPositions() const {
    return primary_positions_;
}

// Static utility methods
std::array<double, 3> IlluminationModel::cartesianFromLatLon(double lat, double lon, double radius) {
    double lat_rad = lat * constants::DEG_TO_RAD;
    double lon_rad = lon * constants::DEG_TO_RAD;
    
    return {{
        radius * std::cos(lat_rad) * std::cos(lon_rad),
        radius * std::cos(lat_rad) * std::sin(lon_rad),
        radius * std::sin(lat_rad)
    }};
}

std::pair<double, double> IlluminationModel::latLonFromCartesian(double x, double y, double z) {
    double radius = std::sqrt(x*x + y*y + z*z);
    double lat_rad = std::asin(z / radius);
    double lon_rad = std::atan2(y, x);
    
    return {
        lat_rad * constants::RAD_TO_DEG,
        lon_rad * constants::RAD_TO_DEG
    };
}

bool IlluminationModel::initialize_spice(bool verbose) {
    static bool spice_initialized = false;
    
    if (!spice_initialized) {
        try {
            // Reset error handling
            erract_c("SET", 0, const_cast<char*>("RETURN"));
            errprt_c("SET", 0, const_cast<char*>("NONE"));
            
            std::cout << "Initializing SPICE with meta-kernel" << std::endl;
            
            // Use the meta-kernel filename from properties
            std::string meta_kernel = kernels_path_ + "/" + meta_kernel_filename_;
            
            if (!std::filesystem::exists(meta_kernel)) {
                std::cerr << "Error: SPICE meta-kernel not found: " << meta_kernel << std::endl;
                // Try falling back to default meta-kernel if different
                if (meta_kernel_filename_ != "europa.tm") {
                    std::cerr << "Attempting to fall back to default europa.tm" << std::endl;
                    meta_kernel = kernels_path_ + "/europa.tm";
                    
                    if (!std::filesystem::exists(meta_kernel)) {
                        std::cerr << "Error: Default SPICE meta-kernel not found: " << meta_kernel << std::endl;
                        return false;
                    }
                } else {
                    return false;
                }
            }
            
            furnsh_c(meta_kernel.c_str());
            // Only print detailed messages if verbose mode is enabled
            if (verbose) {
                std::cout << "Loaded SPICE meta-kernel: " << meta_kernel_filename_ << std::endl;
                std::cout << "Set planetary body to " << target_body_name_ 
                        << " with primary " << primary_body_name_ 
                        << " using meta-kernel " << meta_kernel_filename_ << std::endl;
            }
            // Only run diagnostic checks if verbose mode is enabled
            if (verbose) {
                debugSpice();
            }
            
            spice_initialized = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "SPICE initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    return true;
}

double IlluminationModel::convertTimeToET(const std::string& time_str) {
    // Convert time string to SPICE ephemeris time
    double et;
    str2et_c(time_str.c_str(), &et);
    return et;
}

std::array<double, 3> IlluminationModel::getSunPosition(double et) {
    double position[3];
    double lt;
    
    // Use the configurable target body and frame
    spkpos_c("SUN", et, target_body_frame_.c_str(), "LT+S", 
             target_body_name_.c_str(), position, &lt);
    checkSpiceError("getSunPosition");
    
    return {{position[0], position[1], position[2]}};
}

std::array<double, 3> IlluminationModel::getPrimaryPosition(double et) {
    double position[3];
    double lt;
    
    // Use the configurable primary and target bodies
    spkpos_c(primary_body_name_.c_str(), et, target_body_frame_.c_str(), "LT+S", 
             target_body_name_.c_str(), position, &lt);
    checkSpiceError("getPrimaryPosition");
    
    return {{position[0], position[1], position[2]}};
}

// Helper method for SPICE in IlluminationModel class
void IlluminationModel::checkSpiceError(const std::string& operation) {
    if (failed_c()) {
        SpiceChar short_msg[81];
        SpiceChar long_msg[241];
        
        getmsg_c("SHORT", 80, short_msg);
        getmsg_c("LONG", 240, long_msg);
        
        std::cerr << "SPICE error in " << operation << ": " << short_msg << std::endl;
        std::cerr << long_msg << std::endl;
        
        // Clear the error state
        reset_c();
    }
}

bool IlluminationModel::debugSpice() {
    std::cout << "==== SPICE Debug Information ====" << std::endl;
    
    // Test body ID lookups
    SpiceInt sunID, europaID, jupiterID;
    SpiceBoolean found;
    
    // Try all possible name formats
    bodn2c_c("SUN", &sunID, &found);
    std::cout << "Body lookup 'SUN': " << (found ? "Found (ID: " + std::to_string(sunID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("10", &sunID, &found);
    std::cout << "Body lookup '10': " << (found ? "Found (ID: " + std::to_string(sunID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("EUROPA", &europaID, &found);
    std::cout << "Body lookup 'EUROPA': " << (found ? "Found (ID: " + std::to_string(europaID) + ")" : "Not found") << std::endl;
    
    bodn2c_c("502", &europaID, &found);
    std::cout << "Body lookup '502': " << (found ? "Found (ID: " + std::to_string(europaID) + ")" : "Not found") << std::endl;
    
    // Test frame lookups
    SpiceInt j2000ID, iauEuropaID;
    
    namfrm_c("J2000", &j2000ID);
    std::cout << "Frame lookup 'J2000': " << (j2000ID != 0 ? "Found (ID: " + std::to_string(j2000ID) + ")" : "Not found") << std::endl;
    
    namfrm_c("IAU_EUROPA", &iauEuropaID);
    std::cout << "Frame lookup 'IAU_EUROPA': " << (iauEuropaID != 0 ? "Found (ID: " + std::to_string(iauEuropaID) + ")" : "Not found") << std::endl;
    
    std::cout << "==== End Debug Information ====" << std::endl;
    return true;
}

bool IlluminationModel::loadPositionsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open positions file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    // Skip header lines that start with #
    while (std::getline(file, line)) {
        line_number++;
        if (!line.empty() && line[0] != '#') {
            break;
        }
    }
    
    // Reset file position
    file.clear();
    file.seekg(0);
    
    // Skip header lines again
    for (int i = 0; i < line_number - 1; i++) {
        std::getline(file, line);
    }
    
    // Parse data
    int expected_time_steps = time_steps_;
    sun_positions_.clear();
    primary_positions_.clear();
    times_.clear();
    
    int time_index;
    double et, sun_x, sun_y, sun_z, primary_x, primary_y, primary_z;
    char comma;

    while (file >> time_index >> comma >> et >> comma >> sun_x >> comma >> sun_y >> comma >>
           sun_z >> comma >> primary_x >> comma >> primary_y >> comma >> primary_z) {

        times_.push_back(et);
        sun_positions_.push_back({{sun_x, sun_y, sun_z}});
        primary_positions_.push_back({{primary_x, primary_y, primary_z}});
    }
    
    if (sun_positions_.size() != expected_time_steps) {
        std::cerr << "Warning: Expected " << expected_time_steps << " time steps, but loaded " 
                  << sun_positions_.size() << std::endl;
    }
    
    // Convert ephemeris times to strings
    time_strings_.resize(times_.size());
    for (size_t i = 0; i < times_.size(); i++) {
        // Format time as ISO string (simplified approach)
        std::stringstream ss;
        ss << "Time " << i;
        time_strings_[i] = ss.str();
    }
    
    precomputed_positions_loaded_ = true;
    std::cout << "Loaded " << sun_positions_.size() << " positions from file." << std::endl;
    return true;
}

double IlluminationModel::dotProduct(const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

std::array<double, 3> IlluminationModel::normalize(const std::array<double, 3>& v) {
    double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    
    if (length < constants::EPSILON) {
        return {{0.0, 0.0, 0.0}}; // Return zero vector if the input is too short
    }
    
    return {{
        v[0] / length,
        v[1] / length,
        v[2] / length
    }};
}

std::array<double, 3> IlluminationModel::calculateFaceNormal(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2,
    const std::array<double, 3>& v3) {
    
    // Calculate two edges of the triangle
    std::array<double, 3> edge1 = {{
        v2[0] - v1[0],
        v2[1] - v1[1],
        v2[2] - v1[2]
    }};
    
    std::array<double, 3> edge2 = {{
        v3[0] - v1[0],
        v3[1] - v1[1],
        v3[2] - v1[2]
    }};
    
    // Calculate cross product of edges
    std::array<double, 3> normal = {{
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0]
    }};
    
    // Normalize the resulting vector
    return normalize(normal);
}

// Batch processing for direct solar illumination with finite disk model
void IlluminationModel::calculateDirectSolarBatchWithFiniteDisk(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    std::vector<double>& direct_solar_values,
    int num_samples) {
    
    size_t num_faces = face_centers.size();
    direct_solar_values.resize(num_faces, 0.0);
    
    // Calculate direct solar illumination for each face in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_faces; ++i) {
        // Use the existing single-face calculation function
        direct_solar_values[i] = calculateSolarIlluminationWithFiniteDisk(
            face_centers[i],
            face_normals[i],
            sun_position,
            primary_position,
            num_samples
        );
    }
}

// Implementation of the finite solar disk model
double IlluminationModel::calculateSolarIlluminationWithFiniteDisk(
    const std::array<double, 3>& face_center,
    const std::array<double, 3>& face_normal,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    int num_samples) {
    
    // First part remains unchanged
    // Calculate sun direction and distance
    std::array<double, 3> sun_rel = {
        sun_position[0] - face_center[0],
        sun_position[1] - face_center[1],
        sun_position[2] - face_center[2]
    };
    
    double sun_dist = vectorLength(sun_rel);
    
    // Early exit if distance is too small
    if (sun_dist < constants::EPSILON) {
        return 0.0;
    }
    
    std::array<double, 3> sun_dir = {
        sun_rel[0] / sun_dist,
        sun_rel[1] / sun_dist,
        sun_rel[2] / sun_dist
    };
    
    // Calculate cosine of angle between normal and sun
    double cos_angle = dotProduct(face_normal, sun_dir);
    
    // If surface is facing away from sun, no illumination
    if (cos_angle <= 0.0) {
        return 0.0;
    }
    
    // Create ray system for sampling points on solar disk
    std::array<double, 3> u, v;
    createCoordinateSystem(sun_dir, u, v);
    
    // Create grid sampling of solar disk
    int grid_size = static_cast<int>(std::sqrt(num_samples));
    if (grid_size < 1) grid_size = 1;
    
    double sun_dist_au = sun_dist / constants::KM_PER_AU;
    double solar_irradiance = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);
    
    // Simplified batching approach - directly collect rays in thread-local arrays
    std::vector<std::array<double, 3>> ray_origins;
    std::vector<std::array<double, 3>> ray_directions;
    std::vector<bool> primary_eclipsed;
    int total_samples = 0;
    
    // Reserve space based on expected sample count to avoid reallocations
    ray_origins.reserve(grid_size * grid_size);
    ray_directions.reserve(grid_size * grid_size);
    primary_eclipsed.reserve(grid_size * grid_size);
    
    // Loop over grid points on solar disk - collect all rays first
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            // Generate point on unit disk
            double x = (2.0 * ((i + 0.5) / grid_size) - 1.0) * constants::SUN_RADIUS / sun_dist;
            double y = (2.0 * ((j + 0.5) / grid_size) - 1.0) * constants::SUN_RADIUS / sun_dist;
            
            // Only use points inside unit disk
            if (x*x + y*y <= (constants::SUN_RADIUS*constants::SUN_RADIUS)/(sun_dist*sun_dist)) {
                total_samples++;
                
                // Calculate direction to this point on solar disk
                std::array<double, 3> sample_dir = {
                    sun_dir[0] + x * u[0] + y * v[0],
                    sun_dir[1] + x * u[1] + y * v[1],
                    sun_dir[2] + x * u[2] + y * v[2]
                };
                
                // Normalize direction
                double len = vectorLength(sample_dir);
                sample_dir[0] /= len;
                sample_dir[1] /= len;
                sample_dir[2] /= len;
                
                // Adjust ray origin slightly above surface to prevent self-intersection
                std::array<double, 3> ray_origin = {
                    face_center[0] + face_normal[0] * 0.001,
                    face_center[1] + face_normal[1] * 0.001,
                    face_center[2] + face_normal[2] * 0.001
                };
                
                // First check for Primary eclipse using fast ray-sphere test
                bool primary_eclipse = false;
                
                // Skip primary eclipse check if primary is the Sun!
                if (!primary_is_sun_) {
                    double t1, t2;
                    if (raySphereIntersect(
                            ray_origin,
                            sample_dir,
                            primary_position,
                            primary_radius_,
                            t1, t2)) {
                        // Only count eclipse if Primary intersection is before the sun
                        if ((t1 > 0.0 && t1 < sun_dist) || (t2 > 0.0 && t2 < sun_dist)) {
                            primary_eclipse = true;
                        }
                    }
                }
                
                // Add this sample to tracking
                primary_eclipsed.push_back(primary_eclipse);
                
                // Only add rays to batch if not already eclipsed by Primary
                if (!primary_eclipse) {
                    ray_origins.push_back(ray_origin);
                    ray_directions.push_back(sample_dir);
                }
            }
        }
    }
    
    // Process rays directly with one traceRays call
    std::vector<int> ray_results;
    if (!ray_origins.empty()) {
        ray_results = traceRays(ray_origins, ray_directions);
    }
    
    // Count visible samples
    int visible_samples = 0;
    size_t result_idx = 0;
    
    for (int i = 0; i < total_samples; i++) {
        if (primary_eclipsed[i]) {
            // Already know it's eclipsed by Primary
            continue;
        }
        
        // Check if this ray hit any geometry (terrain)
        bool body_eclipse = (result_idx < ray_results.size()) && (ray_results[result_idx] >= 0);
        result_idx++;
        
        // Sample is visible if not eclipsed by either Primary or Body terrain
        if (!body_eclipse) {
            visible_samples++;
        }
    }
    
    // Calculate illumination based on fraction of visible samples
    double visibility = static_cast<double>(visible_samples) / static_cast<double>(total_samples);
    double illumination = solar_irradiance * cos_angle * visibility;
    
    return illumination;
}

std::array<double, 3> IlluminationModel::vectorSubtract(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2) {
    return {{
        v1[0] - v2[0],
        v1[1] - v2[1],
        v1[2] - v2[2]
    }};
}

std::array<double, 3> IlluminationModel::crossProduct(
    const std::array<double, 3>& v1,
    const std::array<double, 3>& v2) {
    return {{
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    }};
}

double IlluminationModel::vectorLength(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool IlluminationModel::raySphereIntersect(
    const std::array<double, 3>& ray_origin,
    const std::array<double, 3>& ray_direction,
    const std::array<double, 3>& sphere_center,
    double sphere_radius,
    double& t1,
    double& t2) {
    
    // Compute relative position of sphere to ray origin
    std::array<double, 3> rel_pos = {{
        ray_origin[0] - sphere_center[0],
        ray_origin[1] - sphere_center[1],
        ray_origin[2] - sphere_center[2]
    }};
    
    // Ray-sphere intersection coefficients
    double b = 2.0 * (ray_direction[0] * rel_pos[0] + 
                     ray_direction[1] * rel_pos[1] + 
                     ray_direction[2] * rel_pos[2]);
    
    double c = rel_pos[0] * rel_pos[0] +
               rel_pos[1] * rel_pos[1] + 
               rel_pos[2] * rel_pos[2] - 
               sphere_radius * sphere_radius;
    
    double discriminant = b*b - 4.0*c;
    
    if (discriminant > 0.0) {
        t1 = (-b - std::sqrt(discriminant)) / 2.0;
        t2 = (-b + std::sqrt(discriminant)) / 2.0;
        return true;
    }
    
    t1 = t2 = 0.0;
    return false;
}

void IlluminationModel::createCoordinateSystem(
    const std::array<double, 3>& direction,
    std::array<double, 3>& u,
    std::array<double, 3>& v) {
    
    // Choose a suitable up vector for creating the basis
    std::array<double, 3> up = {0.0, 0.0, 1.0};
    
    // If direction is nearly parallel to up, use a different up vector
    if (std::abs(direction[2]) > 0.9) {
        up = {0.0, 1.0, 0.0};
    }
    
    // Implement cross product directly instead of calling member function
    u[0] = up[1] * direction[2] - up[2] * direction[1];
    u[1] = up[2] * direction[0] - up[0] * direction[2];
    u[2] = up[0] * direction[1] - up[1] * direction[0];
    
    // Normalize directly
    double len = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (len > constants::EPSILON) {
        u[0] /= len;
        u[1] /= len;
        u[2] /= len;
    }
    
    // Second cross product directly
    v[0] = direction[1] * u[2] - direction[2] * u[1];
    v[1] = direction[2] * u[0] - direction[0] * u[2];
    v[2] = direction[0] * u[1] - direction[1] * u[0];
}

void IlluminationModel::calculatePrimaryIRBatch(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& primary_position,
    std::vector<double>& ir_values) {
    
    size_t num_faces = face_centers.size();
    ir_values.resize(num_faces, 0.0);
    
    // Skip IR calculation if primary is the Sun
    if (primary_is_sun_) {
        return;
    }
    
    // Process each face in parallel directly
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_faces; ++i) {
        const auto& face_center = face_centers[i];
        const auto& face_normal = face_normals[i];
        
        // Calculate Primary direction relative to face
        std::array<double, 3> primary_rel = {
            primary_position[0] - face_center[0],
            primary_position[1] - face_center[1],
            primary_position[2] - face_center[2]
        };
        
        double primary_dist = vectorLength(primary_rel);
        
        // Only consider faces that potentially face Primary
        if (primary_dist <= constants::EPSILON) {
            continue;
        }
        
        std::array<double, 3> primary_dir = {
            primary_rel[0] / primary_dist,
            primary_rel[1] / primary_dist,
            primary_rel[2] / primary_dist
        };
        
        double cos_angle = dotProduct(face_normal, primary_dir);
        
        // Skip faces not facing Primary
        if (cos_angle <= 0.0) {
            continue;
        }
        
        // Pre-calculate heating value
        double ir_flux = calculatePrimaryHeating(
            face_normal, primary_dir, primary_dist);
        
        // Create ray for visibility check
        std::array<double, 3> ray_origin = {
            face_center[0] + face_normal[0] * 0.001,
            face_center[1] + face_normal[1] * 0.001,
            face_center[2] + face_normal[2] * 0.001
        };
        
        // Create a single ray directly
        std::vector<std::array<double, 3>> origins = {ray_origin};
        std::vector<std::array<double, 3>> directions = {primary_dir};
        
        // Use optimized traceRays function with reduced distance
        // This avoids the need for custom ray setup and manual tfar setting
        std::vector<int> results = traceRays(origins, directions);
        
        // If ray doesn't hit anything, Primary is visible
        // We need to verify that the hit point is beyond the primary
        if (!results.empty() && results[0] < 0) {
            ir_values[i] = ir_flux;
        }
    }
}

void IlluminationModel::calculatePrimaryReflectedBatch(
    const std::vector<std::array<double, 3>>& face_centers,
    const std::vector<std::array<double, 3>>& face_normals,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position,
    std::vector<double>& reflected_values) {
    
    size_t num_faces = face_centers.size();
    reflected_values.resize(num_faces, 0.0);
    
    // Skip calculation if primary is the Sun
    if (primary_is_sun_) {
        return;
    }
    
    // Perform calculation for each face in parallel directly
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_faces; ++i) {
        const auto& face_center = face_centers[i];
        const auto& face_normal = face_normals[i];
        
        // Calculate Primary direction relative to face
        std::array<double, 3> primary_rel = {
            primary_position[0] - face_center[0],
            primary_position[1] - face_center[1],
            primary_position[2] - face_center[2]
        };
        
        double primary_dist = vectorLength(primary_rel);
        
        // Only consider faces that potentially face Primary
        if (primary_dist <= constants::EPSILON) {
            continue;
        }
        
        std::array<double, 3> primary_dir = {
            primary_rel[0] / primary_dist,
            primary_rel[1] / primary_dist,
            primary_rel[2] / primary_dist
        };
        
        double cos_angle = dotProduct(face_normal, primary_dir);
        
        // Skip faces not facing Primary
        if (cos_angle <= 0.0) {
            continue;
        }
        
        // Pre-calculate reflection value
        double reflection_value = calculatePrimaryReflectedLight(
            face_normal, primary_dir, sun_position, primary_position);
        
        // Create ray for visibility check
        std::array<double, 3> ray_origin = {
            face_center[0] + face_normal[0] * 0.001,
            face_center[1] + face_normal[1] * 0.001,
            face_center[2] + face_normal[2] * 0.001
        };
        
        // Create a single ray directly with origin and direction
        std::vector<std::array<double, 3>> origins = {ray_origin};
        std::vector<std::array<double, 3>> directions = {primary_dir};
        
        // Trace this single ray
        std::vector<int> results = traceRays(origins, directions);
        
        // If ray doesn't hit anything, Primary is visible
        if (!results.empty() && results[0] < 0) {
            reflected_values[i] = reflection_value;
        }
    }
}

bool IlluminationModel::traceRaySingle(const Ray& ray) {
    // Create vectors with a single ray
    std::vector<std::array<double, 3>> origins = {ray.origin};
    std::vector<std::array<double, 3>> directions = {ray.direction};
    
    // Use existing ray tracing infrastructure to perform the intersection test
    std::vector<int> result = traceRays(origins, directions);
    
    // If the result indicates a hit (not -1), there was an intersection
    return !result.empty() && result[0] >= 0;
}

double IlluminationModel::calculatePrimaryHeating(
    const std::array<double, 3>& face_normal,
    const std::array<double, 3>& primary_direction,
    double primary_distance) {

    // Calculate cosine of angle between normal and primary direction
    double cos_angle = dotProduct(face_normal, primary_direction);

    // If Primary is below horizon, no heating
    if (cos_angle <= 0.0) {
        return 0.0;
    }

    // Use the PRIMARY_IR_FLUX constant directly as requested
    // Calculate the angular size of Primary as seen from Body
    double primary_angular_radius = std::asin(constants::PRIMARY_RADIUS / primary_distance);
    double solid_angle = 2 * M_PI * (1.0 - std::cos(primary_angular_radius));

    // View factor calculation using the solid angle and Lambert's cosine law
    double view_factor = solid_angle / (4 * M_PI) * cos_angle;
    
    // Apply the constant IR flux, scaled by view factor
    return constants::PRIMARY_IR_FLUX * view_factor;
}

double IlluminationModel::calculatePrimaryReflectedLight(
    const std::array<double, 3>& face_normal,
    const std::array<double, 3>& primary_direction,
    const std::array<double, 3>& sun_position,
    const std::array<double, 3>& primary_position) {

    // Calculate cosine of angle between normal and primary direction
    double cos_angle = dotProduct(face_normal, primary_direction);

    // If Primary is below horizon, no reflection
    if (cos_angle <= 0.0) {
        return 0.0;
    }

    // Get vector from Primary to Sun
    std::array<double, 3> primary_to_sun = {
        sun_position[0] - primary_position[0],
        sun_position[1] - primary_position[1],
        sun_position[2] - primary_position[2]
    };
    double sun_dist = vectorLength(primary_to_sun);

    // Normalize
    std::array<double, 3> sun_direction = {
        primary_to_sun[0] / sun_dist,
        primary_to_sun[1] / sun_dist,
        primary_to_sun[2] / sun_dist
    };

    // Calculate Primary distance
    double primary_distance = vectorLength(primary_position);

    // Phase angle between sun and Body as seen from Primary
    // Note the negative sign because primary_direction points *toward* Primary
    double phase_angle = std::acos(-dotProduct(sun_direction, primary_direction));

    // Phase function for Primary's reflectance (Lambert sphere approximation)
    double phase_function = (1.0 + std::cos(phase_angle)) / 2.0;

    // Primary's albedo
    const double primary_albedo = 0.52; // Bond albedo

    // Calculate angular size of Primary as seen from Body
    double primary_angular_radius = std::asin(constants::PRIMARY_RADIUS / primary_distance);
    double solid_angle = 2 * M_PI * (1.0 - std::cos(primary_angular_radius));

    // Calculate solar flux at Primary
    double sun_dist_au = sun_dist / constants::KM_PER_AU;
    double solar_flux_at_primary = constants::SOLAR_CONSTANT / (sun_dist_au * sun_dist_au);

    // Calculate reflected flux
    double reflected_flux = solar_flux_at_primary * primary_albedo * phase_function *
                           (solid_angle / (4.0 * M_PI)) * cos_angle;
    
    return reflected_flux;
}

// Helper function to map body names to SPICE IDs
std::string IlluminationModel::getSpiceIdForBody(const std::string& body_name) {
    // Common SPICE IDs for solar system bodies
    static const std::map<std::string, std::string> spice_ids = {
        {"SUN", "10"},
        {"MERCURY", "199"},
        {"VENUS", "299"},
        {"EARTH", "399"},
        {"MOON", "301"},
        {"MARS", "499"},
        {"JUPITER", "599"},
        {"EUROPA", "502"},
        {"GANYMEDE", "503"},
        {"CALLISTO", "504"},
        {"IO", "501"},
        {"SATURN", "699"},
        {"TITAN", "606"},
        {"ENCELADUS", "602"},
        {"URANUS", "799"},
        {"NEPTUNE", "899"},
        {"PLUTO", "999"}
    };
    
    auto it = spice_ids.find(body_name);
    if (it != spice_ids.end()) {
        return it->second;
    }
    
    // Default to Europa if not found
    // Replace debugPrint with std::cout since this is a static method
    std::cout << "Warning: No SPICE ID found for body: " << body_name 
              << ". Using Europa's ID as default." << std::endl;
    return "502";
}

// Set planetary body from enum
void IlluminationModel::setPlanetaryBody(thermal::PlanetaryBody body) {
    // Get properties for this body to access meta-kernel info
    thermal::ThermophysicalProperties props = thermal::PlanetaryData::getProperties(body);
    
    // Set meta-kernel from properties
    meta_kernel_filename_ = props.spice_metakernel;

    // Set albedo from planetary properties for consistency
    setAlbedo(props.albedo);
    
    switch (body) {
        case thermal::PlanetaryBody::EUROPA:
            setTargetBody("EUROPA");
            setPrimaryBody("JUPITER");
            primary_radius_ = constants::JUPITER_RADIUS;
            primary_is_sun_ = false;
            break;
        case thermal::PlanetaryBody::MOON:
            setTargetBody("MOON");
            setPrimaryBody("EARTH");
            primary_radius_ = constants::EARTH_RADIUS;
            primary_is_sun_ = false;
            break;
        case thermal::PlanetaryBody::MERCURY:
            setTargetBody("MERCURY");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            break;
        case thermal::PlanetaryBody::ENCELADUS:
            setTargetBody("ENCELADUS");
            setPrimaryBody("SATURN");
            primary_radius_ = constants::SATURN_RADIUS;
            primary_is_sun_ = false;
            break;
        case thermal::PlanetaryBody::TITAN:
            setTargetBody("TITAN");
            setPrimaryBody("SATURN");
            primary_radius_ = constants::SATURN_RADIUS;
            primary_is_sun_ = false;
            break;
        case thermal::PlanetaryBody::MARS:
            setTargetBody("MARS");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            break;
        case thermal::PlanetaryBody::CERES:
            setTargetBody("CERES");
            setPrimaryBody("SUN");
            primary_is_sun_ = true;
            break;
        default:
            setTargetBody("EUROPA"); // Default
            setPrimaryBody("JUPITER");
            break;
    }
    
    // Use standard cout since debugPrint is non-static
    std::cout << "Set planetary body to " << target_body_name_ 
              << " with primary " << primary_body_name_
              << " using meta-kernel " << meta_kernel_filename_ << std::endl;
}

void IlluminationModel::initializeChunkedStorage(size_t total_faces, int time_steps) {
    using_chunked_storage_ = true;
    
    // Calculate number of chunks needed
    size_t num_chunks = (total_faces + chunk_size_ - 1) / chunk_size_;
    
    // Initialize chunk vectors
    illumination_chunks_.resize(num_chunks);
    direct_solar_chunks_.resize(num_chunks);
    primary_reflected_chunks_.resize(num_chunks);
    primary_ir_chunks_.resize(num_chunks);
    
    // Initialize each chunk
    for (size_t i = 0; i < num_chunks; i++) {
        // Calculate size of this chunk
        size_t this_chunk_size = (i == num_chunks - 1) ? 
            (total_faces - i * chunk_size_) * time_steps : 
            chunk_size_ * time_steps;
            
        illumination_chunks_[i].resize(this_chunk_size, 0.0);
        direct_solar_chunks_[i].resize(this_chunk_size, 0.0);
        primary_reflected_chunks_[i].resize(this_chunk_size, 0.0);
        primary_ir_chunks_[i].resize(this_chunk_size, 0.0);
    }
    
    std::cout << "Initialized chunked storage with " << num_chunks 
              << " chunks of up to " << chunk_size_ << " faces each" << std::endl;
}

void IlluminationModel::storeChunkedValue(std::vector<std::vector<double>>& chunks, 
                                         size_t face_idx, size_t time_idx, double value) {
    size_t chunk_idx = face_idx / chunk_size_;
    size_t local_face_idx = face_idx % chunk_size_;
    size_t local_idx = local_face_idx * time_steps_ + time_idx;
    
    // Bounds checking
    if (chunk_idx >= chunks.size()) {
        debugPrint("Warning: chunk_idx out of bounds in storeChunkedValue: ", 
                  chunk_idx, " >= ", chunks.size());
        return;
    }
    
    if (local_idx >= chunks[chunk_idx].size()) {
        debugPrint("Warning: local_idx out of bounds in storeChunkedValue: ", 
                  local_idx, " >= ", chunks[chunk_idx].size());
        return;
    }
    
    chunks[chunk_idx][local_idx] = value;
}

double IlluminationModel::getChunkedValue(const std::vector<std::vector<double>>& chunks, 
                                         size_t face_idx, size_t time_idx) const {
    if (!using_chunked_storage_) {
        // If we're not using chunked storage, this is an error
        return 0.0;
    }
    
    size_t chunk_idx = face_idx / chunk_size_;
    size_t local_face_idx = face_idx % chunk_size_;
    size_t local_idx = local_face_idx * time_steps_ + time_idx;
    
    // Bounds checking
    if (chunk_idx >= chunks.size()) {
        return 0.0;
    }
    
    if (local_idx >= chunks[chunk_idx].size()) {
        return 0.0;
    }
    
    return chunks[chunk_idx][local_idx];
}

// New accessor methods implementation
double IlluminationModel::getIlluminationValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(illumination_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < illumination_values_.size()) {
        return illumination_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getDirectSolarValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(direct_solar_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < direct_solar_values_.size()) {
        return direct_solar_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getPrimaryReflectedValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(primary_reflected_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < primary_reflected_values_.size()) {
        return primary_reflected_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

double IlluminationModel::getPrimaryIRValue(size_t face_idx, size_t time_idx) const {
    if (using_chunked_storage_) {
        return getChunkedValue(primary_ir_chunks_, face_idx, time_idx);
    } else if (face_idx * time_steps_ + time_idx < primary_ir_values_.size()) {
        return primary_ir_values_[face_idx * time_steps_ + time_idx];
    }
    return 0.0;
}

} // namespace ptm