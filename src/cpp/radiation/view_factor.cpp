#include "view_factor.h"
#include "../thermal/thermal_model.h"  // For ProgressBar

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>
#include <atomic>
#include <iomanip>
#include <chrono>

// Platform-specific includes for memory query
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace radiation {

// ============================================================================
// Memory Query Functions (for adaptive neighbor selection)
// ============================================================================

static size_t getAvailableMemory() {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stats), &count) == KERN_SUCCESS) {
        vm_size_t page_size;
        host_page_size(host, &page_size);
        // Available = free + inactive (can be reclaimed)
        return (vm_stats.free_count + vm_stats.inactive_count) * page_size;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
#else
    return 0;  // Unknown platform
#endif
}

static size_t getTotalMemory() {
#ifdef __APPLE__
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
        return static_cast<size_t>(memsize);
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

// ============================================================================
// SIMD Batch Visibility Testing (AVX-512 / AVX2)
// ============================================================================

// Check for AVX-512 support at runtime
static bool hasAVX512() {
#if defined(__x86_64__) || defined(_M_X64)
    // Check CPUID for AVX-512F support
    unsigned int eax, ebx, ecx, edx;
    #ifdef _MSC_VER
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    ebx = cpuInfo[1];
    #else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    #endif
    return (ebx & (1 << 16)) != 0;  // AVX-512F bit
#else
    return false;
#endif
}

// Ray candidate for batch visibility testing
struct RayCandidate {
    std::array<double, 3> origin;
    std::array<double, 3> dir;
    float tfar;
    int neighbor_idx;  // Track which neighbor this ray corresponds to
};

// AVX-512 16-wide ray packet visibility test
static void batchedVisibilityTest16(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible,
    size_t start_idx,
    size_t end_idx) {

    constexpr int BATCH_SIZE = 16;

    // 64-byte aligned arrays for AVX-512
    alignas(64) float org_x[BATCH_SIZE];
    alignas(64) float org_y[BATCH_SIZE];
    alignas(64) float org_z[BATCH_SIZE];
    alignas(64) float dir_x[BATCH_SIZE];
    alignas(64) float dir_y[BATCH_SIZE];
    alignas(64) float dir_z[BATCH_SIZE];
    alignas(64) float tnear[BATCH_SIZE];
    alignas(64) float tfar_arr[BATCH_SIZE];
    alignas(64) int valid[BATCH_SIZE];

    size_t i = start_idx;
    while (i < end_idx) {
        int batch_count = 0;
        size_t batch_start = i;

        // Fill batch
        while (batch_count < BATCH_SIZE && i < end_idx) {
            const auto& c = candidates[i];
            org_x[batch_count] = static_cast<float>(c.origin[0]);
            org_y[batch_count] = static_cast<float>(c.origin[1]);
            org_z[batch_count] = static_cast<float>(c.origin[2]);
            dir_x[batch_count] = static_cast<float>(c.dir[0]);
            dir_y[batch_count] = static_cast<float>(c.dir[1]);
            dir_z[batch_count] = static_cast<float>(c.dir[2]);
            tnear[batch_count] = 0.0f;
            tfar_arr[batch_count] = c.tfar;
            valid[batch_count] = -1;  // -1 means valid ray
            batch_count++;
            i++;
        }

        // Mark remaining slots as invalid
        for (int j = batch_count; j < BATCH_SIZE; ++j) {
            valid[j] = 0;
            org_x[j] = org_y[j] = org_z[j] = 0.0f;
            dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
            tnear[j] = 0.0f;
            tfar_arr[j] = -1.0f;
        }

        // Create 16-wide ray packet
        RTCRayHit16 rayhit16;
        for (int j = 0; j < BATCH_SIZE; ++j) {
            rayhit16.ray.org_x[j] = org_x[j];
            rayhit16.ray.org_y[j] = org_y[j];
            rayhit16.ray.org_z[j] = org_z[j];
            rayhit16.ray.dir_x[j] = dir_x[j];
            rayhit16.ray.dir_y[j] = dir_y[j];
            rayhit16.ray.dir_z[j] = dir_z[j];
            rayhit16.ray.tnear[j] = tnear[j];
            rayhit16.ray.tfar[j] = tfar_arr[j];
            rayhit16.ray.mask[j] = 0xFFFFFFFF;
            rayhit16.ray.flags[j] = 0;
            rayhit16.ray.time[j] = 0.0f;
            rayhit16.ray.id[j] = static_cast<unsigned int>(batch_start + j);
            rayhit16.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit16.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit16.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
        }

        // Execute 16-wide ray trace
        rtcIntersect16(valid, scene, &rayhit16);

        // Extract results
        for (int j = 0; j < batch_count; ++j) {
            visible[batch_start + j] = (rayhit16.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID);
        }
    }
}

// AVX2 8-wide ray packet visibility test
static void batchedVisibilityTest8(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible,
    size_t start_idx,
    size_t end_idx) {

    constexpr int BATCH_SIZE = 8;

    // 32-byte aligned arrays for AVX2
    alignas(32) float org_x[BATCH_SIZE];
    alignas(32) float org_y[BATCH_SIZE];
    alignas(32) float org_z[BATCH_SIZE];
    alignas(32) float dir_x[BATCH_SIZE];
    alignas(32) float dir_y[BATCH_SIZE];
    alignas(32) float dir_z[BATCH_SIZE];
    alignas(32) float tnear[BATCH_SIZE];
    alignas(32) float tfar_arr[BATCH_SIZE];
    alignas(32) int valid[BATCH_SIZE];

    size_t i = start_idx;
    while (i < end_idx) {
        int batch_count = 0;
        size_t batch_start = i;

        // Fill batch
        while (batch_count < BATCH_SIZE && i < end_idx) {
            const auto& c = candidates[i];
            org_x[batch_count] = static_cast<float>(c.origin[0]);
            org_y[batch_count] = static_cast<float>(c.origin[1]);
            org_z[batch_count] = static_cast<float>(c.origin[2]);
            dir_x[batch_count] = static_cast<float>(c.dir[0]);
            dir_y[batch_count] = static_cast<float>(c.dir[1]);
            dir_z[batch_count] = static_cast<float>(c.dir[2]);
            tnear[batch_count] = 0.0f;
            tfar_arr[batch_count] = c.tfar;
            valid[batch_count] = -1;  // -1 means valid ray
            batch_count++;
            i++;
        }

        // Mark remaining slots as invalid
        for (int j = batch_count; j < BATCH_SIZE; ++j) {
            valid[j] = 0;
            org_x[j] = org_y[j] = org_z[j] = 0.0f;
            dir_x[j] = dir_y[j] = dir_z[j] = 0.0f;
            tnear[j] = 0.0f;
            tfar_arr[j] = -1.0f;
        }

        // Create 8-wide ray packet
        RTCRayHit8 rayhit8;
        for (int j = 0; j < BATCH_SIZE; ++j) {
            rayhit8.ray.org_x[j] = org_x[j];
            rayhit8.ray.org_y[j] = org_y[j];
            rayhit8.ray.org_z[j] = org_z[j];
            rayhit8.ray.dir_x[j] = dir_x[j];
            rayhit8.ray.dir_y[j] = dir_y[j];
            rayhit8.ray.dir_z[j] = dir_z[j];
            rayhit8.ray.tnear[j] = tnear[j];
            rayhit8.ray.tfar[j] = tfar_arr[j];
            rayhit8.ray.mask[j] = 0xFFFFFFFF;
            rayhit8.ray.flags[j] = 0;
            rayhit8.ray.time[j] = 0.0f;
            rayhit8.ray.id[j] = static_cast<unsigned int>(batch_start + j);
            rayhit8.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit8.hit.primID[j] = RTC_INVALID_GEOMETRY_ID;
            rayhit8.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
        }

        // Execute 8-wide ray trace
        rtcIntersect8(valid, scene, &rayhit8);

        // Extract results
        for (int j = 0; j < batch_count; ++j) {
            visible[batch_start + j] = (rayhit8.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID);
        }
    }
}

// Thread-safe initialization of SIMD mode (called once before parallel region)
static bool g_simd_initialized = false;
static bool g_use_avx512 = false;

static void initializeSIMDVisibility() {
    if (!g_simd_initialized) {
        g_use_avx512 = hasAVX512();
        if (g_use_avx512) {
            std::cout << "  Using AVX-512 16-wide ray packets for visibility testing" << std::endl;
        } else {
            std::cout << "  Using AVX2 8-wide ray packets for visibility testing" << std::endl;
        }
        g_simd_initialized = true;
    }
}

// Batched visibility test using Embree ray packets
// Automatically selects AVX-512 (16-wide) or AVX2 (8-wide) based on CPU support
// NOTE: initializeSIMDVisibility() must be called before using this from parallel code
static void batchedVisibilityTest(
    RTCScene scene,
    const std::vector<RayCandidate>& candidates,
    std::vector<bool>& visible) {

    visible.resize(candidates.size());
    if (candidates.empty()) return;

    const size_t n_candidates = candidates.size();

    if (g_use_avx512) {
        batchedVisibilityTest16(scene, candidates, visible, 0, n_candidates);
    } else {
        batchedVisibilityTest8(scene, candidates, visible, 0, n_candidates);
    }
}

// Prepare ray from two points with appropriate offsets to avoid self-intersection
static RayCandidate prepareRay(
    const std::array<double, 3>& p1,
    const std::array<double, 3>& p2,
    int neighbor_idx) {

    RayCandidate ray;
    ray.neighbor_idx = neighbor_idx;

    // Compute direction
    std::array<double, 3> dir = {{p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]}};
    double dist = std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);

    if (dist < 1e-10) {
        // Same point - mark as invalid ray
        ray.origin = p1;
        ray.dir = {{1.0, 0.0, 0.0}};
        ray.tfar = -1.0f;
        return ray;
    }

    // Normalize
    dir[0] /= dist;
    dir[1] /= dist;
    dir[2] /= dist;

    // Compute offset (same logic as isVisible)
    double relative_offset = dist * 0.001;
    double absolute_offset = 0.5;
    double offset = std::max(relative_offset, absolute_offset);
    offset = std::min(offset, dist * 0.1);

    // Compute ray origin with offset
    ray.origin = {{
        p1[0] + offset * dir[0],
        p1[1] + offset * dir[1],
        p1[2] + offset * dir[2]
    }};
    ray.dir = dir;

    // Compute tfar
    double end_margin = std::max(dist * 0.001, absolute_offset);
    end_margin = std::min(end_margin, dist * 0.1);
    double tfar = dist - offset - end_margin;

    ray.tfar = (tfar < 0.1f) ? -1.0f : static_cast<float>(tfar);  // -1 means skip (too close)

    return ray;
}

// ============================================================================
// ViewFactorMatrix Implementation
// ============================================================================

void ViewFactorMatrix::initialize(size_t n_faces, size_t estimated_nnz_per_row) {
    n_rows_ = n_faces;
    n_cols_ = n_faces;
    row_ptrs_.resize(n_faces + 1, 0);
    values_.reserve(n_faces * estimated_nnz_per_row);
    col_indices_.reserve(n_faces * estimated_nnz_per_row);
}

void ViewFactorMatrix::buildFromTriplets(size_t n_faces,
                                          const std::vector<int>& row_indices,
                                          const std::vector<int>& col_indices,
                                          const std::vector<double>& values) {
    n_rows_ = n_faces;
    n_cols_ = n_faces;
    size_t nnz = values.size();

    if (nnz == 0) {
        row_ptrs_.assign(n_faces + 1, 0);
        values_.clear();
        col_indices_.clear();
        return;
    }

    // Count entries per row
    std::vector<int> row_counts(n_faces, 0);
    for (size_t k = 0; k < nnz; ++k) {
        row_counts[row_indices[k]]++;
    }

    // Compute row pointers
    row_ptrs_.resize(n_faces + 1);
    row_ptrs_[0] = 0;
    for (size_t i = 0; i < n_faces; ++i) {
        row_ptrs_[i + 1] = row_ptrs_[i] + row_counts[i];
    }

    // Allocate values and column indices
    values_.resize(nnz);
    col_indices_.resize(nnz);

    // Fill in values (using row_counts as temporary write positions)
    std::fill(row_counts.begin(), row_counts.end(), 0);
    for (size_t k = 0; k < nnz; ++k) {
        int row = row_indices[k];
        int64_t pos = row_ptrs_[row] + row_counts[row];
        values_[pos] = values[k];
        col_indices_[pos] = col_indices[k];
        row_counts[row]++;
    }

    // Sort each row by column index for efficient access
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_faces; ++i) {
        int64_t start = row_ptrs_[i];
        int64_t end = row_ptrs_[i + 1];
        if (end - start > 1) {
            // Simple insertion sort for small arrays
            for (int64_t j = start + 1; j < end; ++j) {
                int col = col_indices_[j];
                double val = values_[j];
                int64_t k = j - 1;
                while (k >= start && col_indices_[k] > col) {
                    col_indices_[k + 1] = col_indices_[k];
                    values_[k + 1] = values_[k];
                    k--;
                }
                col_indices_[k + 1] = col;
                values_[k + 1] = val;
            }
        }
    }
}

double ViewFactorMatrix::get(int row, int col) const {
    if (row < 0 || row >= static_cast<int>(n_rows_)) return 0.0;

    int64_t start = row_ptrs_[row];
    int64_t end = row_ptrs_[row + 1];

    // Binary search for column
    auto it = std::lower_bound(col_indices_.begin() + start,
                               col_indices_.begin() + end, col);
    if (it != col_indices_.begin() + end && *it == col) {
        return values_[it - col_indices_.begin()];
    }
    return 0.0;
}

void ViewFactorMatrix::getRow(int row, std::vector<int>& cols, std::vector<double>& vals) const {
    cols.clear();
    vals.clear();

    if (row < 0 || row >= static_cast<int>(n_rows_)) return;

    int64_t start = row_ptrs_[row];
    int64_t end = row_ptrs_[row + 1];

    cols.reserve(end - start);
    vals.reserve(end - start);

    for (int64_t i = start; i < end; ++i) {
        cols.push_back(col_indices_[i]);
        vals.push_back(values_[i]);
    }
}

int ViewFactorMatrix::getRowNNZ(int row) const {
    if (row < 0 || row >= static_cast<int>(n_rows_)) return 0;
    return static_cast<int>(row_ptrs_[row + 1] - row_ptrs_[row]);
}

void ViewFactorMatrix::matvec(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(n_rows_, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_rows_; ++i) {
        double sum = 0.0;
        int64_t start = row_ptrs_[i];
        int64_t end = row_ptrs_[i + 1];
        for (int64_t k = start; k < end; ++k) {
            sum += values_[k] * x[col_indices_[k]];
        }
        y[i] = sum;
    }
}

void ViewFactorMatrix::matvecTranspose(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(n_cols_, 0.0);

    // This is less efficient but correct for sparse transpose
    for (size_t i = 0; i < n_rows_; ++i) {
        int64_t start = row_ptrs_[i];
        int64_t end = row_ptrs_[i + 1];
        for (int64_t k = start; k < end; ++k) {
            #pragma omp atomic
            y[col_indices_[k]] += values_[k] * x[i];
        }
    }
}

std::vector<double> ViewFactorMatrix::rowSums() const {
    std::vector<double> sums(n_rows_, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_rows_; ++i) {
        double sum = 0.0;
        int64_t start = row_ptrs_[i];
        int64_t end = row_ptrs_[i + 1];
        for (int64_t k = start; k < end; ++k) {
            sum += values_[k];
        }
        sums[i] = sum;
    }
    return sums;
}

void ViewFactorMatrix::clear() {
    values_.clear();
    col_indices_.clear();
    row_ptrs_.clear();
    n_rows_ = 0;
    n_cols_ = 0;
}

size_t ViewFactorMatrix::memoryUsage() const {
    return values_.size() * sizeof(double) +
           col_indices_.size() * sizeof(int) +
           row_ptrs_.size() * sizeof(int64_t);
}

// Magic number for sparse view factor file format: "VFFM" = View Factor File Matrix
static constexpr uint32_t SPARSE_VF_MAGIC = 0x5646464D;
static constexpr uint32_t SPARSE_VF_VERSION = 1;

bool ViewFactorMatrix::writeBinary(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write magic number and version for format detection
    file.write(reinterpret_cast<const char*>(&SPARSE_VF_MAGIC), sizeof(SPARSE_VF_MAGIC));
    file.write(reinterpret_cast<const char*>(&SPARSE_VF_VERSION), sizeof(SPARSE_VF_VERSION));

    // Write header
    uint64_t nrows = n_rows_;
    uint64_t ncols = n_cols_;
    uint64_t nnz = values_.size();

    file.write(reinterpret_cast<const char*>(&nrows), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&ncols), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(uint64_t));

    // Write row pointers as int32 (the on-disk .vf format is v1/int32 for
    // backward compatibility with every existing .vf file). row_ptrs_ is int64
    // in memory; fail loud rather than silently truncate if this CSR is too
    // large to serialize -- a > 2^31-nnz matrix is an in-memory GPU/VF_REPLACE
    // object, never written to the legacy .vf format.
    {
        std::vector<int32_t> rp32(row_ptrs_.size());
        for (size_t i = 0; i < row_ptrs_.size(); ++i) {
            if (row_ptrs_[i] > static_cast<int64_t>(INT32_MAX)) {
                std::cerr << "ViewFactorMatrix::writeBinary: row pointer "
                          << row_ptrs_[i] << " exceeds INT32_MAX; this CSR ("
                          << values_.size() << " nnz) is too large for the int32 "
                          << ".vf format. Aborting write." << std::endl;
                return false;
            }
            rp32[i] = static_cast<int32_t>(row_ptrs_[i]);
        }
        file.write(reinterpret_cast<const char*>(rp32.data()),
                   rp32.size() * sizeof(int32_t));
    }

    // Write column indices
    file.write(reinterpret_cast<const char*>(col_indices_.data()),
               col_indices_.size() * sizeof(int));

    // Write values
    file.write(reinterpret_cast<const char*>(values_.data()),
               values_.size() * sizeof(double));

    return file.good();
}

bool ViewFactorMatrix::readBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    // Read and verify magic number and version
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != SPARSE_VF_MAGIC) {
        std::cerr << "Invalid sparse view factor file format (magic: 0x"
                  << std::hex << magic << std::dec << ")" << std::endl;
        return false;
    }
    if (version != SPARSE_VF_VERSION) {
        std::cerr << "Unsupported sparse view factor file version: " << version << std::endl;
        return false;
    }

    // Read header
    uint64_t nrows, ncols, nnz;
    file.read(reinterpret_cast<char*>(&nrows), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&ncols), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(uint64_t));

    n_rows_ = nrows;
    n_cols_ = ncols;

    // Read row pointers (on-disk v1 format stores them as int32) and widen to
    // the int64 in-memory representation. Byte-for-byte compatible with every
    // existing .vf file.
    {
        std::vector<int32_t> rp32(nrows + 1);
        file.read(reinterpret_cast<char*>(rp32.data()),
                  rp32.size() * sizeof(int32_t));
        row_ptrs_.resize(nrows + 1);
        for (size_t i = 0; i < rp32.size(); ++i)
            row_ptrs_[i] = static_cast<int64_t>(rp32[i]);
    }

    // Read column indices
    col_indices_.resize(nnz);
    file.read(reinterpret_cast<char*>(col_indices_.data()),
              col_indices_.size() * sizeof(int));

    // Read values
    values_.resize(nnz);
    file.read(reinterpret_cast<char*>(values_.data()),
              values_.size() * sizeof(double));

    return file.good();
}

// ============================================================================
// ViewFactorCalculator Implementation
// ============================================================================

ViewFactorCalculator::ViewFactorCalculator(const std::shared_ptr<ptm::Mesh>& mesh,
                                           RTCDevice device,
                                           RTCScene scene,
                                           const RadiationConfig& config)
    : mesh_(mesh), device_(device), scene_(scene), config_(config) {
    if (config_.verbose) {
        std::cout << "ViewFactorCalculator initialized with visibility_threshold="
                  << config_.visibility_threshold << std::endl;
    }
    precomputeFaceData();
}

ViewFactorCalculator::~ViewFactorCalculator() {
    // Note: We don't own the Embree device/scene, so don't release them
}

void ViewFactorCalculator::precomputeFaceData() {
    if (face_data_computed_) return;

    size_t n_faces = mesh_->faces.size();
    face_centers_.resize(n_faces);
    face_normals_.resize(n_faces);
    face_areas_.resize(n_faces);

    // Check if mesh already has pre-computed face data with outward-oriented normals
    // PTM3D's MeshUtils ensures normals point outward (away from body center)
    bool use_mesh_data = (mesh_->face_centers.size() == n_faces &&
                          mesh_->face_normals.size() == n_faces);

    if (use_mesh_data) {
        // Use pre-computed face centers and outward-oriented normals from mesh
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_faces; ++i) {
            face_centers_[i] = mesh_->face_centers[i];
            face_normals_[i] = mesh_->face_normals[i];

            // Still need to compute areas
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];
            auto e1 = subtract(v1, v0);
            auto e2 = subtract(v2, v0);
            auto n = cross(e1, e2);
            face_areas_[i] = length(n) / 2.0;
        }

        if (config_.verbose) {
            std::cout << "Using pre-computed face data from mesh (outward-oriented normals)" << std::endl;
        }
    } else {
        // Compute face data from scratch and orient normals outward
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_faces; ++i) {
            const auto& face = mesh_->faces[i];
            const auto& v0 = mesh_->vertices[face[0]];
            const auto& v1 = mesh_->vertices[face[1]];
            const auto& v2 = mesh_->vertices[face[2]];

            // Compute center
            face_centers_[i] = {{
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0
            }};

            // Compute normal and area
            auto e1 = subtract(v1, v0);
            auto e2 = subtract(v2, v0);
            auto n = cross(e1, e2);
            double area2 = length(n);
            face_areas_[i] = area2 / 2.0;

            if (area2 > 0) {
                face_normals_[i] = {{n[0] / area2, n[1] / area2, n[2] / area2}};
            } else {
                face_normals_[i] = {{0.0, 0.0, 1.0}};
            }

            // Ensure outward-pointing normals (away from body center at origin)
            // This matches PTM3D convention: normal · center > 0 means outward
            const auto& c = face_centers_[i];
            double radial_dot = face_normals_[i][0] * c[0] +
                                face_normals_[i][1] * c[1] +
                                face_normals_[i][2] * c[2];
            if (radial_dot < 0) {
                // Flip to point outward
                face_normals_[i][0] = -face_normals_[i][0];
                face_normals_[i][1] = -face_normals_[i][1];
                face_normals_[i][2] = -face_normals_[i][2];
            }
        }

        if (config_.verbose) {
            std::cout << "Computed face data with outward-oriented normals" << std::endl;
        }
    }

    face_data_computed_ = true;

    // Build spatial grid
    double avg_area = 0.0;
    for (double a : face_areas_) avg_area += a;
    avg_area /= n_faces;
    double cell_hint = std::sqrt(avg_area) * 10.0;
    spatial_grid_.build(face_centers_, cell_hint);
    spatial_grid_built_ = true;

    // Build lightweight cluster-pair visibility cull tree (conservative pre-filter
    // before per-face neighbor ranking). Gated by config flag.
    if (config_.use_cluster_cull) {
        cluster_tree_ = std::make_unique<ClusterTree>();
        cluster_tree_->build(face_centers_, face_normals_);
        if (config_.verbose) {
            std::cout << "Cluster cull tree: " << cluster_tree_->numLeaves()
                      << " leaves, ~"
                      << (cluster_tree_->memoryBytes() / (1024 * 1024))
                      << " MB" << std::endl;
        }
    }
}

double ViewFactorCalculator::computeFaceArea(int face_idx) {
    if (face_data_computed_) {
        return face_areas_[face_idx];
    }
    const auto& face = mesh_->faces[face_idx];
    const auto& v0 = mesh_->vertices[face[0]];
    const auto& v1 = mesh_->vertices[face[1]];
    const auto& v2 = mesh_->vertices[face[2]];
    auto e1 = subtract(v1, v0);
    auto e2 = subtract(v2, v0);
    return length(cross(e1, e2)) / 2.0;
}

ViewFactorMatrix ViewFactorCalculator::computeAll() {
    std::vector<int> all_faces(mesh_->faces.size());
    std::iota(all_faces.begin(), all_faces.end(), 0);
    return computeSubset(all_faces);
}

// Static debug counters for view factor computation (centroid method)
static std::atomic<int> g_angle_reject{0};
static std::atomic<int> g_vis_reject{0};
static std::atomic<int> g_accepted{0};
static std::atomic<bool> g_first_accept_printed{false};

// Static debug counters for subdivision method
static std::atomic<int> g_subdiv_angle_reject{0};
static std::atomic<int> g_subdiv_vis_reject{0};
static std::atomic<int> g_subdiv_accepted{0};
static std::atomic<int> g_subdiv_weight_zero{0};

ViewFactorMatrix ViewFactorCalculator::computeSubset(const std::vector<int>& face_indices) {
    size_t n_subset = face_indices.size();
    size_t n_total = mesh_->faces.size();

    if (config_.verbose) {
        std::cout << "Computing view factors for " << n_subset << " faces..." << std::endl;
    }

    // Compute search radius based on mesh extent, not just face size.
    // For crater/terrain meshes, we need to search far enough to find
    // faces on opposite walls that can exchange radiation.
    //
    // Use the larger of:
    // 1. search_radius_factor * sqrt(avg_area) - for fine meshes
    // 2. mesh_extent / 2 - to ensure we can reach across the crater
    double avg_area = 0.0;
    for (int idx : face_indices) {
        avg_area += face_areas_[idx];
    }
    avg_area /= n_subset;
    double face_based_radius = config_.search_radius_factor * std::sqrt(avg_area);

    // Compute mesh extent from bounding box
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();

    for (int idx : face_indices) {
        const auto& c = face_centers_[idx];
        min_x = std::min(min_x, c[0]); max_x = std::max(max_x, c[0]);
        min_y = std::min(min_y, c[1]); max_y = std::max(max_y, c[1]);
        min_z = std::min(min_z, c[2]); max_z = std::max(max_z, c[2]);
    }

    double extent_x = max_x - min_x;
    double extent_y = max_y - min_y;
    double extent_z = max_z - min_z;
    double mesh_extent = std::sqrt(extent_x * extent_x + extent_y * extent_y + extent_z * extent_z);
    double extent_based_radius = mesh_extent;  // Full diagonal to reach any face

    double search_radius = std::max(face_based_radius, extent_based_radius);

    if (config_.verbose) {
        std::cout << "Search radius: " << search_radius << " m "
                  << "(face-based: " << face_based_radius << ", extent-based: " << extent_based_radius << ")"
                  << std::endl;
    }

    // Collect triplets in thread-local storage
    std::vector<std::vector<int>> local_rows(omp_get_max_threads());
    std::vector<std::vector<int>> local_cols(omp_get_max_threads());
    std::vector<std::vector<double>> local_vals(omp_get_max_threads());

    // Compute memory budget for adaptive selection
    size_t memory_budget = 0;
    size_t max_total_entries = 0;
    if (config_.adaptive_neighbors) {
        memory_budget = computeMemoryBudget(n_total);
        max_total_entries = memory_budget / 12;  // 12 bytes per CSR entry
        std::cout << "  Using adaptive neighbor selection:" << std::endl;
        std::cout << "    Target captured energy: " << (config_.captured_energy_target * 100) << "%" << std::endl;
        std::cout << "    Min neighbors/face: " << config_.min_neighbors << std::endl;
        std::cout << "    Max neighbors/face: " << config_.max_neighbors << std::endl;
    }

    // Statistics tracking (atomic for thread safety)
    std::atomic<size_t> global_entry_count{0};
    std::atomic<size_t> total_neighbors{0};
    std::atomic<size_t> faces_at_min{0};
    std::atomic<size_t> faces_at_max{0};
    std::atomic<size_t> faces_at_target{0};

    // Initialize SIMD visibility testing BEFORE parallel region (thread-safe)
    if (config_.adaptive_neighbors) {
        initializeSIMDVisibility();
        std::cout << std::flush;  // Ensure output is visible before parallel region
    }

    // Timing for diagnostics
    auto start_time = std::chrono::high_resolution_clock::now();

    // Progress tracking with atomic counter for thread-safe updates
    // Always show progress bar (not gated by verbose) since this is a long operation
    std::atomic<size_t> progress_counter{0};
    // For large meshes, update more frequently (every 1000 faces or 0.1%)
    const size_t progress_interval = std::min(
        std::max(n_subset / 1000, static_cast<size_t>(1)),   // Update ~1000 times
        static_cast<size_t>(1000)                             // But at least every 1000 faces
    );
    thermal::ProgressBar progress_bar(static_cast<int>(n_subset), "View factor computation: ", 50, true);
    // Show initial 0% progress immediately
    progress_bar.update(0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        local_rows[tid].reserve(n_subset * 50 / omp_get_max_threads());
        local_cols[tid].reserve(n_subset * 50 / omp_get_max_threads());
        local_vals[tid].reserve(n_subset * 50 / omp_get_max_threads());

        // Thread-local scratch buffer for candidates (avoids per-face allocation)
        std::vector<int> candidates;
        candidates.reserve(10000);  // Pre-allocate for typical crater geometry

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < n_subset; ++i) {
            int face_i = face_indices[i];

            // Find candidate neighbor faces (using by-reference overload)
            findNeighbors(face_i, search_radius, candidates);

            // Select neighbors and compute view factors
            // Adaptive method returns precomputed geometry to avoid redundant calculations
            if (config_.adaptive_neighbors) {
                auto neighbors_with_geom = selectNeighborsAdaptive(face_i, candidates,
                                                                    global_entry_count,
                                                                    max_total_entries);

                // Track statistics
                size_t n_neighbors = neighbors_with_geom.size();
                total_neighbors.fetch_add(n_neighbors, std::memory_order_relaxed);

                if (n_neighbors == static_cast<size_t>(config_.min_neighbors)) {
                    faces_at_min.fetch_add(1, std::memory_order_relaxed);
                } else if (n_neighbors == static_cast<size_t>(config_.max_neighbors)) {
                    faces_at_max.fetch_add(1, std::memory_order_relaxed);
                } else {
                    faces_at_target.fetch_add(1, std::memory_order_relaxed);
                }

                // Compute view factors using SIMD batch visibility testing
                auto view_factors = computeViewFactorsBatched(face_i, neighbors_with_geom);

                for (const auto& [face_j, F_ij] : view_factors) {
                    local_rows[tid].push_back(face_i);
                    local_cols[tid].push_back(face_j);
                    local_vals[tid].push_back(F_ij);
                }
            } else {
                // Fixed method: use original computeAdaptive (no precomputed geometry)
                auto neighbors = selectNeighborsFixed(face_i, candidates);

                // Track statistics
                size_t n_neighbors = neighbors.size();
                total_neighbors.fetch_add(n_neighbors, std::memory_order_relaxed);

                if (n_neighbors == static_cast<size_t>(config_.min_neighbors)) {
                    faces_at_min.fetch_add(1, std::memory_order_relaxed);
                } else if (n_neighbors == static_cast<size_t>(config_.max_neighbors)) {
                    faces_at_max.fetch_add(1, std::memory_order_relaxed);
                } else {
                    faces_at_target.fetch_add(1, std::memory_order_relaxed);
                }

                // Compute view factors (full computation for each pair)
                for (int face_j : neighbors) {
                    double F_ij = computeAdaptive(face_i, face_j);

                    if (F_ij > config_.visibility_threshold) {
                        local_rows[tid].push_back(face_i);
                        local_cols[tid].push_back(face_j);
                        local_vals[tid].push_back(F_ij);
                    }
                }
            }

            // Update progress counter (all threads increment)
            size_t current = ++progress_counter;

            // Only one thread updates the display to avoid contention
            // Use modulo on current count so updates happen regularly regardless of which thread
            if (current % progress_interval == 0) {
                #pragma omp critical(progress_update)
                {
                    progress_bar.update(static_cast<int>(current));
                }
            }
        }
    }

    // Finalize progress bar
    progress_bar.finish();

    // Report computation time
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << std::endl << "  View factor computation time: " << std::fixed << std::setprecision(1)
              << elapsed_sec << " seconds" << std::endl;

    // Print neighbor selection statistics
    if (config_.adaptive_neighbors) {
        double avg_neighbors = static_cast<double>(total_neighbors.load()) / n_subset;
        std::cout << std::endl << "  Adaptive neighbor selection complete:" << std::endl;
        std::cout << "    Average neighbors/face: " << std::fixed << std::setprecision(1)
                  << avg_neighbors << std::endl;
        std::cout << "    Faces at minimum (" << config_.min_neighbors << "): "
                  << faces_at_min.load() << std::endl;
        std::cout << "    Faces at target energy: " << faces_at_target.load() << std::endl;
        std::cout << "    Faces at maximum (" << config_.max_neighbors << "): "
                  << faces_at_max.load() << std::endl;
        std::cout << "    Total neighbor selections: " << global_entry_count.load() << std::endl;

        // Calculate actual vs theoretical max memory
        size_t actual_entries = global_entry_count.load();
        size_t max_possible = n_subset * static_cast<size_t>(config_.max_neighbors);
        double savings_pct = 100.0 * (1.0 - static_cast<double>(actual_entries) / max_possible);
        std::cout << "    Memory savings vs fixed max: " << std::fixed << std::setprecision(1)
                  << savings_pct << "%" << std::endl;
    }

    // Print debug summary after progress bar completes (only in verbose mode)
    if (config_.verbose) {
        std::cout << "View factor computation complete." << std::endl;
    }

    // Merge thread-local data
    std::vector<int> all_rows, all_cols;
    std::vector<double> all_vals;

    size_t total_nnz = 0;
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        total_nnz += local_rows[t].size();
    }
    all_rows.reserve(total_nnz);
    all_cols.reserve(total_nnz);
    all_vals.reserve(total_nnz);

    for (int t = 0; t < omp_get_max_threads(); ++t) {
        all_rows.insert(all_rows.end(), local_rows[t].begin(), local_rows[t].end());
        all_cols.insert(all_cols.end(), local_cols[t].begin(), local_cols[t].end());
        all_vals.insert(all_vals.end(), local_vals[t].begin(), local_vals[t].end());
    }

    // Build sparse matrix
    ViewFactorMatrix F;
    F.buildFromTriplets(n_total, all_rows, all_cols, all_vals);

    if (config_.verbose) {
        std::cout << "View factor matrix: " << F.numRows() << " x " << F.numCols()
                  << ", " << F.numNonZeros() << " non-zeros ("
                  << (100.0 * F.numNonZeros()) / (F.numRows() * F.numCols())
                  << "% dense)" << std::endl;
        std::cout << "Memory usage: " << F.memoryUsage() / (1024.0 * 1024.0)
                  << " MB" << std::endl;

        // Print debug rejection statistics
        std::cout << "View factor rejection statistics (centroid method):" << std::endl;
        std::cout << "  Rejected by angle check: " << g_angle_reject.load() << std::endl;
        std::cout << "  Rejected by visibility: " << g_vis_reject.load() << std::endl;
        std::cout << "  Accepted: " << g_accepted.load() << std::endl;

        std::cout << "View factor rejection statistics (subdivision method):" << std::endl;
        std::cout << "  Rejected (weight_sum=0, all angles bad): " << g_subdiv_weight_zero.load() << std::endl;
        std::cout << "  Rejected by visibility: " << g_subdiv_vis_reject.load() << std::endl;
        std::cout << "  Accepted: " << g_subdiv_accepted.load() << std::endl;

        // Reset for next run
        g_angle_reject = 0;
        g_vis_reject = 0;
        g_accepted = 0;
        g_first_accept_printed = false;
        g_subdiv_weight_zero = 0;
        g_subdiv_vis_reject = 0;
        g_subdiv_accepted = 0;

        // Diagnostic: Check row sum variation (indicates view factor consistency)
        auto row_sums = F.rowSums();
        double min_sum = 1e9, max_sum = 0.0, sum_total = 0.0;
        int zero_rows = 0;
        for (size_t i = 0; i < row_sums.size(); ++i) {
            if (row_sums[i] > 0) {
                min_sum = std::min(min_sum, row_sums[i]);
                max_sum = std::max(max_sum, row_sums[i]);
                sum_total += row_sums[i];
            } else {
                zero_rows++;
            }
        }
        size_t nonzero_rows = row_sums.size() - zero_rows;
        double mean_sum = (nonzero_rows > 0) ? sum_total / nonzero_rows : 0.0;
        std::cout << "Row sum statistics (terrain view factors):" << std::endl;
        std::cout << "  Zero rows (no view factors): " << zero_rows << std::endl;
        std::cout << "  Non-zero rows: " << nonzero_rows << std::endl;
        std::cout << "  Row sum range: [" << min_sum << ", " << max_sum << "]" << std::endl;
        std::cout << "  Mean row sum: " << mean_sum << std::endl;

        // Check variation in row sums for neighboring faces
        // High variation among neighbors indicates potential speckling source
        double max_neighbor_variation = 0.0;
        int high_var_face = -1;
        for (size_t i = 0; i < n_total && i < 5000; ++i) {
            if (row_sums[i] == 0) continue;
            // Check adjacent faces (assuming roughly sequential numbering)
            for (int offset = 1; offset <= 5; ++offset) {
                if (i + offset < n_total && row_sums[i + offset] > 0) {
                    double ratio = row_sums[i] / row_sums[i + offset];
                    if (ratio < 1.0) ratio = 1.0 / ratio;
                    if (ratio > max_neighbor_variation) {
                        max_neighbor_variation = ratio;
                        high_var_face = static_cast<int>(i);
                    }
                }
            }
        }
        if (high_var_face >= 0) {
            std::cout << "  Max neighbor row sum ratio: " << max_neighbor_variation
                      << " at face " << high_var_face << std::endl;
            std::cout << "    Face " << high_var_face << " row sum: " << row_sums[high_var_face] << std::endl;
            for (int offset = 1; offset <= 3 && high_var_face + offset < static_cast<int>(n_total); ++offset) {
                std::cout << "    Face " << (high_var_face + offset) << " row sum: "
                          << row_sums[high_var_face + offset] << std::endl;
            }
        }
    }

    return F;
}

double ViewFactorCalculator::computePair(int face_i, int face_j) {
    return computeAdaptive(face_i, face_j);
}

double ViewFactorCalculator::computeSkyViewFactor(int face_idx) {
    // Sky view factor = 1 - sum of terrain view factors
    // For now, compute by summing all view factors to other faces
    double terrain_sum = 0.0;

    // Find all neighbors within search radius
    double avg_area = 0.0;
    for (double a : face_areas_) avg_area += a;
    avg_area /= face_areas_.size();
    double search_radius = config_.search_radius_factor * std::sqrt(avg_area);

    auto neighbors = findNeighbors(face_idx, search_radius);

    for (int j : neighbors) {
        if (j == face_idx) continue;
        terrain_sum += computeAdaptive(face_idx, j);
    }

    return std::max(0.0, 1.0 - terrain_sum);
}

double ViewFactorCalculator::computeCentroid(int face_i, int face_j) {
    // M1: Centroid approximation
    // F_ij ≈ (A_j / π) * (cos θ_i * cos θ_j) / r²
    //
    // For two faces to exchange radiation:
    // - Face i must "look toward" face j: the component of n_i along r_ij should
    //   have the right sign. For outward normals, this means dot(n_i, r_hat) > 0.
    //   For inward normals, it would be dot(n_i, r_hat) < 0.
    // - Similarly for face j looking back at face i.
    //
    // To handle arbitrary normal orientations, we check if the faces are
    // geometrically oriented toward each other: normals should point in
    // "opposite" directions relative to the connecting vector.
    // This is captured by: dot(n_i, r_hat) * dot(n_j, r_hat) < 0

    const auto& c_i = face_centers_[face_i];
    const auto& c_j = face_centers_[face_j];
    const auto& n_i = face_normals_[face_i];
    const auto& n_j = face_normals_[face_j];
    double A_j = face_areas_[face_j];

    // Vector from i to j
    auto r_vec = subtract(c_j, c_i);
    double r2 = dot(r_vec, r_vec);
    if (r2 < 1e-20) return 0.0;

    double r = std::sqrt(r2);
    std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

    // Compute dot products with connecting vector r_hat (pointing from i to j)
    double dot_i = dot(n_i, r_hat);   // n_i · r_hat
    double dot_j = dot(n_j, r_hat);   // n_j · r_hat

    // View factor formula with OUTWARD normals (PTM3D convention):
    //
    // For face i to "see" face j:
    //   - Face i's outward normal should have a component toward face j
    //     → cos_i = n_i · r_hat > 0 (face i looks toward j)
    //   - Face j's outward normal should have a component toward face i
    //     → cos_j = n_j · (-r_hat) = -dot_j > 0 (face j looks toward i)
    //
    // For mutual visibility: dot_i > 0 AND dot_j < 0
    //
    // The view factor formula is:
    //   F_ij = A_j * cos_i * cos_j / (π * r²)
    // where cos_i = dot_i and cos_j = -dot_j (both positive for visible faces)

    double cos_i = dot_i;       // n_i · r_hat (positive if i faces toward j)
    double cos_j = -dot_j;      // n_j · (-r_hat) (positive if j faces toward i)

    // Skip nearly-grazing angles (avoid numerical issues)
    if (cos_i < 1e-6 || cos_j < 1e-6) {
        g_angle_reject++;
        return 0.0;
    }

    // Check visibility using ray tracing
    if (!isVisible(c_i, c_j)) {
        g_vis_reject++;
        return 0.0;
    }

    // View factor formula
    double F_ij = (A_j / M_PI) * (cos_i * cos_j) / r2;

    g_accepted++;

    return std::max(0.0, F_ij);
}

double ViewFactorCalculator::computeSubdivision(int face_i, int face_j, int level) {
    // M2: Single-point to area integration method
    // Use centroid of face_i, but integrate over subdivided face_j
    // This is more accurate than M1 for nearby faces and more stable than M3
    //
    // F_ij = (1/π) * ∫∫_{A_j} (cos_i * cos_j / r²) dA_j
    //
    // We approximate this by summing over subdivision points of face_j,
    // where each point represents an equal area element A_j/n_points.

    std::vector<std::array<double, 3>> points_j;
    std::vector<double> weights_j;

    getSubdivisionPoints(face_j, level, points_j, weights_j);

    const auto& c_i = face_centers_[face_i];
    const auto& n_i = face_normals_[face_i];
    const auto& n_j = face_normals_[face_j];
    double A_j = face_areas_[face_j];

    // First check center-to-center visibility
    if (!isVisible(c_i, face_centers_[face_j])) {
        g_subdiv_vis_reject++;
        return 0.0;
    }

    double F_sum = 0.0;
    int n_valid = 0;
    int angle_rejects = 0;

    // Integrate over receiving face (face_j)
    for (size_t pj = 0; pj < points_j.size(); ++pj) {
        const auto& p_j = points_j[pj];

        auto r_vec = subtract(p_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) continue;

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0] / r, r_vec[1] / r, r_vec[2] / r}};

        // Compute dot products with connecting vector r_hat (pointing from i to j)
        double dot_i = dot(n_i, r_hat);
        double dot_j = dot(n_j, r_hat);

        // With outward normals:
        //   cos_i = dot_i (positive if face i looks toward point)
        //   cos_j = -dot_j (positive if face j looks toward face i)
        double cos_i = dot_i;
        double cos_j = -dot_j;

        // Skip faces that don't face each other (cos < 0) or nearly-grazing angles
        if (cos_i < 1e-6 || cos_j < 1e-6) {
            angle_rejects++;
            continue;
        }

        // Each subdivision point represents equal area A_j / n_points
        // F contribution = (cos_i * cos_j / r²) * (A_j / n_points) / π
        F_sum += (cos_i * cos_j) / r2;
        n_valid++;
    }

    if (n_valid == 0) {
        g_subdiv_weight_zero++;
        return 0.0;
    }

    // Final formula: F_ij = (A_j / π) * (1/n_points) * sum
    // where sum = Σ (cos_i * cos_j / r²)
    double n_points = static_cast<double>(points_j.size());
    double F_ij = (A_j / M_PI) * F_sum / n_points;

    // Debug: print first few computed view factors with detailed cosine info
    if (F_ij > 0) {
        g_subdiv_accepted++;
    }
    return std::max(0.0, F_ij);
}

double ViewFactorCalculator::computeAdaptive(int face_i, int face_j) {
    // Adaptive method selection based on dimensionless separation Ns
    // Ns = r / sqrt(A) where r is distance and A is face area

    const auto& c_i = face_centers_[face_i];
    const auto& c_j = face_centers_[face_j];

    auto r_vec = subtract(c_j, c_i);
    double r = std::sqrt(dot(r_vec, r_vec));

    double A_eff = std::sqrt(face_areas_[face_i] * face_areas_[face_j]);
    double Ns = r / std::sqrt(A_eff);

    if (Ns > config_.adaptive_ns_threshold) {
        // Far apart: use centroid method
        return computeCentroid(face_i, face_j);
    } else {
        // Close: use subdivision method
        return computeSubdivision(face_i, face_j, config_.subdivision_level);
    }
}

double ViewFactorCalculator::computeWithPrecomputed(int face_i, int face_j,
                                                     const PrecomputedGeometry& geom) {
    // Compute view factor using precomputed geometry from neighbor selection
    // This avoids redundant distance/angle calculations for the centroid method
    //
    // For far faces (Ns > threshold), we can use the precomputed values directly.
    // For nearby faces requiring subdivision, we fall back to full computation
    // since subdivision requires computing at multiple sample points.

    double A_eff = std::sqrt(face_areas_[face_i] * face_areas_[face_j]);
    double Ns = geom.r / std::sqrt(A_eff);

    if (Ns <= config_.adaptive_ns_threshold) {
        // Close faces: use subdivision method (needs full computation at sample points)
        return computeSubdivision(face_i, face_j, config_.subdivision_level);
    }

    // Far faces: use centroid method with precomputed geometry
    // View factor formula with OUTWARD normals (PTM3D convention):
    //   cos_i = dot(n_i, r_hat)   (positive if face i looks toward j)
    //   cos_j = -dot(n_j, r_hat)  (positive if face j looks toward i)
    double cos_i = geom.dot_ni_rhat;
    double cos_j = -geom.dot_nj_rhat;

    // Skip faces that don't face each other or nearly-grazing angles
    if (cos_i < 1e-6 || cos_j < 1e-6) {
        return 0.0;
    }

    // Check visibility using ray tracing
    if (!isVisible(face_centers_[face_i], face_centers_[face_j])) {
        return 0.0;
    }

    // View factor formula: F_ij = (A_j / π) * (cos_i * cos_j) / r²
    double A_j = face_areas_[face_j];
    double F_ij = (A_j / M_PI) * (cos_i * cos_j) / geom.r2;

    return std::max(0.0, F_ij);
}

// Compute view factors for all neighbors of a face using SIMD batch visibility testing
// Returns vector of (neighbor_idx, F_ij) pairs for non-zero view factors
std::vector<std::pair<int, double>> ViewFactorCalculator::computeViewFactorsBatched(
    int face_i,
    const std::vector<NeighborData>& neighbors) {

    std::vector<std::pair<int, double>> results;
    if (neighbors.empty()) return results;

    results.reserve(neighbors.size());

    // Separate neighbors into far (centroid) and near (subdivision) based on Ns threshold
    struct FarNeighbor {
        int idx;              // Index in neighbors vector
        int face_j;           // Face index
        double cos_i;
        double cos_j;
        double r2;
        double A_j;
    };
    std::vector<FarNeighbor> far_neighbors;
    std::vector<size_t> near_neighbor_indices;

    far_neighbors.reserve(neighbors.size());
    near_neighbor_indices.reserve(neighbors.size() / 10);  // Usually few near neighbors

    for (size_t idx = 0; idx < neighbors.size(); ++idx) {
        const auto& nd = neighbors[idx];
        int face_j = nd.neighbor_idx;

        // Compute Ns to decide method
        double A_eff = std::sqrt(face_areas_[face_i] * face_areas_[face_j]);
        double Ns = nd.geom.r / std::sqrt(A_eff);

        if (Ns <= config_.adaptive_ns_threshold) {
            // Near face: needs subdivision method
            near_neighbor_indices.push_back(idx);
        } else {
            // Far face: can use centroid with batch visibility
            double cos_i = nd.geom.dot_ni_rhat;
            double cos_j = -nd.geom.dot_nj_rhat;

            // Skip faces that don't face each other (angle filter)
            if (cos_i >= 1e-6 && cos_j >= 1e-6) {
                far_neighbors.push_back({
                    static_cast<int>(idx), face_j,
                    cos_i, cos_j, nd.geom.r2, face_areas_[face_j]
                });
            }
        }
    }

    // Process far neighbors with batch visibility testing
    if (!far_neighbors.empty()) {
        // Prepare rays for batch visibility
        std::vector<RayCandidate> rays;
        rays.reserve(far_neighbors.size());

        const auto& c_i = face_centers_[face_i];

        for (const auto& fn : far_neighbors) {
            const auto& c_j = face_centers_[fn.face_j];
            RayCandidate ray = prepareRay(c_i, c_j, fn.idx);
            rays.push_back(ray);
        }

        // Batch visibility test
        std::vector<bool> visible;
        batchedVisibilityTest(scene_, rays, visible);

        // Compute view factors for visible pairs
        for (size_t r = 0; r < far_neighbors.size(); ++r) {
            // Skip invalid rays (tfar < 0 means too close, treat as visible)
            bool is_visible = (rays[r].tfar < 0.0f) ? true : visible[r];

            if (is_visible) {
                const auto& fn = far_neighbors[r];
                double F_ij = (fn.A_j / M_PI) * (fn.cos_i * fn.cos_j) / fn.r2;
                if (F_ij > config_.visibility_threshold) {
                    results.push_back({fn.face_j, F_ij});
                }
            }
        }
    }

    // Process near neighbors with subdivision method (individual visibility)
    for (size_t idx : near_neighbor_indices) {
        const auto& nd = neighbors[idx];
        double F_ij = computeSubdivision(face_i, nd.neighbor_idx, config_.subdivision_level);
        if (F_ij > config_.visibility_threshold) {
            results.push_back({nd.neighbor_idx, F_ij});
        }
    }

    return results;
}

bool ViewFactorCalculator::isVisible(const std::array<double, 3>& p1,
                                     const std::array<double, 3>& p2) {
    // Ray from p1 toward p2
    auto dir = subtract(p2, p1);
    double dist = length(dir);
    if (dist < 1e-10) return true;  // Same point

    // Normalize direction
    dir = {{dir[0] / dist, dir[1] / dist, dir[2] / dist}};

    // Use BOTH relative and absolute offsets to avoid self-intersection.
    // The relative offset handles long-distance rays, while the absolute
    // offset (based on typical face size) handles short-distance rays
    // where a tiny relative offset might still hit the source face.
    //
    // For crater meshes with ~20m face edges, use 0.5m as minimum offset.
    double relative_offset = dist * 0.001;  // 0.1% of distance
    double absolute_offset = 0.5;           // 0.5m minimum
    double offset = std::max(relative_offset, absolute_offset);

    // Clamp offset to at most 10% of distance to avoid overshooting for nearby faces
    offset = std::min(offset, dist * 0.1);

    std::array<double, 3> origin = {{
        p1[0] + offset * dir[0],
        p1[1] + offset * dir[1],
        p1[2] + offset * dir[2]
    }};

    // Similarly, stop the ray before reaching p2 with adequate margin
    double end_margin = std::max(dist * 0.001, absolute_offset);
    end_margin = std::min(end_margin, dist * 0.1);
    double tfar = dist - offset - end_margin;

    // If the effective ray length is too short, consider it visible
    // (faces are essentially touching)
    if (tfar < 0.1) return true;

    // Create ray
    RTCRayHit rayhit;
    rayhit.ray.org_x = static_cast<float>(origin[0]);
    rayhit.ray.org_y = static_cast<float>(origin[1]);
    rayhit.ray.org_z = static_cast<float>(origin[2]);
    rayhit.ray.dir_x = static_cast<float>(dir[0]);
    rayhit.ray.dir_y = static_cast<float>(dir[1]);
    rayhit.ray.dir_z = static_cast<float>(dir[2]);
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = static_cast<float>(tfar);
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.ray.time = 0.0f;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Trace ray
    rtcIntersect1(scene_, &rayhit);

    // Visible if no hit (geomID remains invalid)
    return rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID;
}

std::vector<bool> ViewFactorCalculator::checkVisibilityBatch(
    const std::vector<std::array<double, 3>>& origins,
    const std::vector<std::array<double, 3>>& targets) {

    size_t n = origins.size();
    std::vector<bool> visible(n, false);

    // Process in batches for cache efficiency
    constexpr size_t BATCH_SIZE = 64;

    for (size_t start = 0; start < n; start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, n);

        for (size_t i = start; i < end; ++i) {
            visible[i] = isVisible(origins[i], targets[i]);
        }
    }

    return visible;
}

std::vector<int> ViewFactorCalculator::findNeighbors(int face_idx, double radius) {
    std::vector<int> result;
    findNeighbors(face_idx, radius, result);
    return result;
}

// By-reference overload: avoids allocation when caller provides pre-allocated buffer
void ViewFactorCalculator::findNeighbors(int face_idx, double radius, std::vector<int>& result) {
    result.clear();  // Clear but keep capacity

    if (!spatial_grid_built_) {
        // Fallback: linear search (slow)
        const auto& c = face_centers_[face_idx];
        double r2 = radius * radius;

        for (size_t j = 0; j < face_centers_.size(); ++j) {
            auto d = subtract(face_centers_[j], c);
            if (dot(d, d) <= r2) {
                result.push_back(static_cast<int>(j));
            }
        }
        return;
    }

    spatial_grid_.query(face_centers_[face_idx], radius, result);
}

void ViewFactorCalculator::getSubdivisionPoints(int face_idx, int level,
                                                 std::vector<std::array<double, 3>>& points,
                                                 std::vector<double>& weights) {
    const auto& face = mesh_->faces[face_idx];
    const auto& v0 = mesh_->vertices[face[0]];
    const auto& v1 = mesh_->vertices[face[1]];
    const auto& v2 = mesh_->vertices[face[2]];

    // Generate barycentric coordinates for subdivision
    // Level 0: 1 point (centroid)
    // Level 1: 4 points
    // Level 2: 16 points, etc.

    int n_per_edge = 1 << level;  // 2^level
    int n_points = 0;

    // Count total points
    for (int i = 0; i <= n_per_edge; ++i) {
        for (int j = 0; j <= n_per_edge - i; ++j) {
            n_points++;
        }
    }

    points.resize(n_points);
    weights.resize(n_points, 1.0 / n_points);

    int idx = 0;
    double step = 1.0 / n_per_edge;

    for (int i = 0; i <= n_per_edge; ++i) {
        double u = i * step;
        for (int j = 0; j <= n_per_edge - i; ++j) {
            double v = j * step;
            double w = 1.0 - u - v;

            points[idx] = {{
                u * v0[0] + v * v1[0] + w * v2[0],
                u * v0[1] + v * v1[1] + w * v2[1],
                u * v0[2] + v * v1[2] + w * v2[2]
            }};
            idx++;
        }
    }
}

// Vector utilities
std::array<double, 3> ViewFactorCalculator::subtract(const std::array<double, 3>& a,
                                                      const std::array<double, 3>& b) {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<double, 3> ViewFactorCalculator::cross(const std::array<double, 3>& a,
                                                   const std::array<double, 3>& b) {
    return {{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    }};
}

double ViewFactorCalculator::dot(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double ViewFactorCalculator::length(const std::array<double, 3>& v) {
    return std::sqrt(dot(v, v));
}

std::array<double, 3> ViewFactorCalculator::normalize(const std::array<double, 3>& v) {
    double len = length(v);
    if (len < 1e-20) return {{0.0, 0.0, 1.0}};
    return {{v[0] / len, v[1] / len, v[2] / len}};
}

// ============================================================================
// SpatialGrid Implementation
// ============================================================================

void ViewFactorCalculator::SpatialGrid::build(
    const std::vector<std::array<double, 3>>& centers, double cell_size_hint) {

    if (centers.empty()) return;

    // Find bounding box
    min_bound = centers[0];
    max_bound = centers[0];

    for (const auto& c : centers) {
        for (int d = 0; d < 3; ++d) {
            min_bound[d] = std::min(min_bound[d], c[d]);
            max_bound[d] = std::max(max_bound[d], c[d]);
        }
    }

    // Add small margin
    for (int d = 0; d < 3; ++d) {
        double margin = (max_bound[d] - min_bound[d]) * 0.01 + cell_size_hint;
        min_bound[d] -= margin;
        max_bound[d] += margin;
    }

    // Determine cell size and grid dimensions
    for (int d = 0; d < 3; ++d) {
        double extent = max_bound[d] - min_bound[d];
        grid_dims[d] = std::max(1, static_cast<int>(extent / cell_size_hint) + 1);
        cell_size[d] = extent / grid_dims[d];
    }

    // Allocate cells
    int total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    cells.resize(total_cells);

    // Insert centers into cells
    for (size_t i = 0; i < centers.size(); ++i) {
        int cell_idx = getCellIndex(centers[i]);
        if (cell_idx >= 0 && cell_idx < total_cells) {
            cells[cell_idx].push_back(static_cast<int>(i));
        }
    }
}

std::vector<int> ViewFactorCalculator::SpatialGrid::query(
    const std::array<double, 3>& center, double radius) const {

    std::vector<int> result;
    query(center, radius, result);
    return result;
}

// By-reference overload: avoids allocation when caller provides pre-allocated buffer
void ViewFactorCalculator::SpatialGrid::query(
    const std::array<double, 3>& center, double radius, std::vector<int>& result) const {

    result.clear();  // Clear but keep capacity

    // Find cell range to search
    std::array<int, 3> min_cell, max_cell;
    for (int d = 0; d < 3; ++d) {
        min_cell[d] = std::max(0, static_cast<int>((center[d] - radius - min_bound[d]) / cell_size[d]));
        max_cell[d] = std::min(grid_dims[d] - 1,
                               static_cast<int>((center[d] + radius - min_bound[d]) / cell_size[d]));
    }

    // Search relevant cells
    for (int iz = min_cell[2]; iz <= max_cell[2]; ++iz) {
        for (int iy = min_cell[1]; iy <= max_cell[1]; ++iy) {
            for (int ix = min_cell[0]; ix <= max_cell[0]; ++ix) {
                int cell_idx = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
                // Note: We return all faces in range; caller will check actual distance
                result.insert(result.end(), cells[cell_idx].begin(), cells[cell_idx].end());
            }
        }
    }
}

int ViewFactorCalculator::SpatialGrid::getCellIndex(const std::array<double, 3>& point) const {
    int ix = static_cast<int>((point[0] - min_bound[0]) / cell_size[0]);
    int iy = static_cast<int>((point[1] - min_bound[1]) / cell_size[1]);
    int iz = static_cast<int>((point[2] - min_bound[2]) / cell_size[2]);

    ix = std::max(0, std::min(ix, grid_dims[0] - 1));
    iy = std::max(0, std::min(iy, grid_dims[1] - 1));
    iz = std::max(0, std::min(iz, grid_dims[2] - 1));

    return ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
}

// ============================================================================
// Adaptive Neighbor Selection
// ============================================================================

size_t ViewFactorCalculator::computeMemoryBudget(size_t n_faces) const {
    size_t available = getAvailableMemory();
    size_t total = getTotalMemory();

    // If we can't query memory, use a conservative default
    if (available == 0) {
        std::cout << "  Warning: Could not query available memory, using conservative estimate" << std::endl;
        // Assume 16 GB available, use 50%
        available = 8ULL * 1024 * 1024 * 1024;
    }

    // Reserve 50% of available memory for safety
    size_t budget = available / 2;

    // Subtract estimated overhead
    size_t embree_bvh = n_faces * 80;      // ~80 bytes per face for BVH
    size_t working_arrays = n_faces * 100;  // Thread-local buffers
    size_t overhead = embree_bvh + working_arrays;

    if (budget <= overhead) {
        std::cerr << "  Warning: Very low available memory" << std::endl;
        // Minimum budget: enough for 50 neighbors per face
        return n_faces * 50 * 12;
    }

    size_t sparse_budget = budget - overhead;

    std::cout << "  Memory budget for sparse view factors:" << std::endl;
    std::cout << "    Available RAM: " << available / (1024ULL * 1024 * 1024) << " GB" << std::endl;
    std::cout << "    Total RAM: " << total / (1024ULL * 1024 * 1024) << " GB" << std::endl;
    std::cout << "    Budget (50% of available): " << budget / (1024ULL * 1024 * 1024) << " GB" << std::endl;
    std::cout << "    After overhead: " << sparse_budget / (1024ULL * 1024 * 1024) << " GB" << std::endl;
    std::cout << "    Max sparse entries: " << sparse_budget / 12 << std::endl;

    return sparse_budget;
}

std::vector<NeighborData> ViewFactorCalculator::selectNeighborsAdaptive(
    int face_i,
    const std::vector<int>& candidates,
    std::atomic<size_t>& global_entry_count,
    size_t max_total_entries) const {

    if (candidates.empty()) return {};

    // Compute potential view factor for each candidate and store geometry
    // We store the raw dot products so they can be reused in view factor computation
    struct CandidateData {
        int j;
        double r;
        double r2;
        double dot_ni_rhat;  // Raw dot product (signed)
        double dot_nj_rhat;  // Raw dot product (signed)
        double potential;    // For sorting (uses absolute cosines)
    };
    std::vector<CandidateData> candidate_data;
    candidate_data.reserve(candidates.size());

    const auto& c_i = face_centers_[face_i];
    const auto& n_i = face_normals_[face_i];

    // Cluster-pair visibility pre-cull (conservative). Source leaf looked up once;
    // for each candidate the test is an O(1) dot-product check against precomputed
    // per-leaf average center/normal. Safe no-op when tree was not built.
    const ClusterTree* cluster = cluster_tree_.get();
    const int src_leaf = (cluster != nullptr) ? cluster->leafId(face_i) : -1;

    double total_potential = 0.0;

    for (int j : candidates) {
        if (j == face_i) continue;  // Skip self

        // Conservative cluster-pair cull BEFORE computing per-face geometry.
        // Saves subtract + sqrt + two dot products per culled candidate and,
        // more importantly, removes them from the ranking set.
        if (cluster != nullptr && src_leaf >= 0) {
            int j_leaf = cluster->leafId(j);
            if (j_leaf >= 0 && j_leaf != src_leaf &&
                !cluster->canLeavesSeeEachOther(src_leaf, j_leaf)) {
                continue;
            }
        }

        const auto& c_j = face_centers_[j];
        const auto& n_j = face_normals_[j];
        double A_j = face_areas_[j];

        auto r_vec = subtract(c_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) continue;

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0]/r, r_vec[1]/r, r_vec[2]/r}};

        // Store raw dot products (signed) for later view factor computation
        double dot_ni = dot(n_i, r_hat);
        double dot_nj = dot(n_j, r_hat);

        // EARLY BACKFACE CULLING: Skip if faces cannot see each other
        // Face i must see j (dot_ni > 0) and face j must face toward i (dot_nj < 0)
        // Note: dot_nj is dot(n_j, r_hat) where r_hat points from i to j
        // So for j to face i, n_j should point toward i, meaning dot_nj < 0
        if (dot_ni <= 0.0 || dot_nj >= 0.0) continue;

        // Potential uses absolute cosines (for ranking purposes)
        // Since we passed backface culling: dot_ni > 0, -dot_nj > 0
        double potential = A_j * dot_ni * (-dot_nj) / r2;
        total_potential += potential;

        candidate_data.push_back({j, r, r2, dot_ni, dot_nj, potential});
    }

    if (candidate_data.empty()) return {};

    // Use std::nth_element for O(K) partial selection instead of O(K log K) full sort
    // We only need the top max_neighbors candidates sorted
    auto cmp = [](const CandidateData& a, const CandidateData& b) {
        return a.potential > b.potential;  // Descending order (highest first)
    };

    size_t max_needed = std::min(candidate_data.size(),
                                  static_cast<size_t>(config_.max_neighbors));

    if (max_needed < candidate_data.size()) {
        // Partial sort: O(K) to partition, then O(N log N) to sort only top N
        std::nth_element(candidate_data.begin(),
                         candidate_data.begin() + max_needed,
                         candidate_data.end(),
                         cmp);

        // Now sort only the top portion for greedy selection
        std::sort(candidate_data.begin(),
                  candidate_data.begin() + max_needed,
                  cmp);
    } else {
        // Small number of candidates: full sort is fine
        std::sort(candidate_data.begin(), candidate_data.end(), cmp);
    }

    // Select neighbors until target captured energy is reached
    std::vector<NeighborData> selected;
    selected.reserve(std::min(candidate_data.size(), static_cast<size_t>(config_.max_neighbors)));

    double captured = 0.0;
    double target = config_.captured_energy_target * total_potential;

    for (const auto& cand : candidate_data) {
        // Check global memory ceiling
        size_t current_count = global_entry_count.load(std::memory_order_relaxed);
        if (current_count >= max_total_entries) {
            break;  // Global memory limit reached
        }

        // Check hard per-face ceiling
        if (static_cast<int>(selected.size()) >= config_.max_neighbors) {
            break;
        }

        // Store neighbor with precomputed geometry
        NeighborData nd;
        nd.neighbor_idx = cand.j;
        nd.geom.r = cand.r;
        nd.geom.r2 = cand.r2;
        nd.geom.dot_ni_rhat = cand.dot_ni_rhat;
        nd.geom.dot_nj_rhat = cand.dot_nj_rhat;
        nd.geom.potential = cand.potential;
        selected.push_back(nd);

        captured += cand.potential;
        global_entry_count.fetch_add(1, std::memory_order_relaxed);

        // Check if we've captured enough energy AND have minimum neighbors
        if (captured >= target &&
            static_cast<int>(selected.size()) >= config_.min_neighbors) {
            break;
        }
    }

    return selected;
}

std::vector<int> ViewFactorCalculator::selectNeighborsFixed(
    int face_i,
    const std::vector<int>& candidates) const {

    if (candidates.empty()) return {};

    // If we have fewer candidates than max, return all (except self)
    std::vector<int> neighbors;
    neighbors.reserve(candidates.size());
    for (int j : candidates) {
        if (j != face_i) {
            neighbors.push_back(j);
        }
    }

    if (static_cast<int>(neighbors.size()) <= config_.max_neighbors) {
        return neighbors;
    }

    // Sort by potential view factor and keep top max_neighbors
    std::vector<std::pair<double, int>> potential_vf;
    potential_vf.reserve(neighbors.size());

    const auto& c_i = face_centers_[face_i];
    const auto& n_i = face_normals_[face_i];

    for (int j : neighbors) {
        const auto& c_j = face_centers_[j];
        const auto& n_j = face_normals_[j];
        double A_j = face_areas_[j];

        auto r_vec = subtract(c_j, c_i);
        double r2 = dot(r_vec, r_vec);
        if (r2 < 1e-20) {
            potential_vf.push_back({0.0, j});
            continue;
        }

        double r = std::sqrt(r2);
        std::array<double, 3> r_hat = {{r_vec[0]/r, r_vec[1]/r, r_vec[2]/r}};

        double cos_i = std::abs(dot(n_i, r_hat));
        double cos_j = std::abs(dot(n_j, r_hat));

        double potential = A_j * cos_i * cos_j / r2;
        potential_vf.push_back({-potential, j});  // Negative for descending sort
    }

    // Partial sort to get top max_neighbors
    std::partial_sort(potential_vf.begin(),
                      potential_vf.begin() + config_.max_neighbors,
                      potential_vf.end());

    neighbors.resize(config_.max_neighbors);
    for (int k = 0; k < config_.max_neighbors; ++k) {
        neighbors[k] = potential_vf[k].second;
    }

    return neighbors;
}

} // namespace radiation
