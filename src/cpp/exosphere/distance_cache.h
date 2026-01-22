/*
 * Distance Cache for Exosphere Transport
 *
 * Provides O(1) lookup of angular distances between mesh faces,
 * avoiding repeated acos() computations during transport matrix builds.
 *
 * Memory: Uses upper triangular storage with float precision for cos(theta),
 * reducing memory from O(N²) doubles to O(N²/2) floats.
 *
 * Key optimization: Stores cos(theta) directly since:
 *   - Comparisons can use cos(theta) (monotonic for theta in [0,π])
 *   - Only compute acos() when actual distance is needed
 *   - Kernel cutoff checks use cos(theta) threshold
 */

#ifndef DISTANCE_CACHE_H
#define DISTANCE_CACHE_H

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace ptm {
namespace exosphere {

class DistanceCache {
private:
    size_t num_faces_;
    double body_radius_;

    // Store cos(theta) in upper triangular form (float for memory efficiency)
    // For N faces, we need N*(N-1)/2 entries
    std::vector<float> cos_theta_upper_;

    // Also cache theta values to avoid repeated acos() during matrix builds
    std::vector<float> theta_upper_;
    bool theta_cached_;

    // Precomputed unit vectors for each face center
    std::vector<std::array<float, 3>> unit_centers_;

    bool is_valid_;

    // Convert (i,j) pair to upper triangular index
    // Assumes i < j
    inline size_t triangularIndex(size_t i, size_t j) const {
        // Upper triangular index formula: i*N - i*(i+1)/2 + (j-i-1)
        // Simplified: row_offset + column_within_row
        return i * num_faces_ - (i * (i + 1)) / 2 + (j - i - 1);
    }

public:
    DistanceCache() : num_faces_(0), body_radius_(0), theta_cached_(false), is_valid_(false) {}

    explicit DistanceCache(double body_radius)
        : num_faces_(0), body_radius_(body_radius), theta_cached_(false), is_valid_(false) {}

    // Build the cache from face centers
    // This is O(N²) but only done once
    void build(const std::vector<std::array<double, 3>>& face_centers) {
        auto start = std::chrono::high_resolution_clock::now();

        num_faces_ = face_centers.size();
        if (num_faces_ == 0) {
            is_valid_ = false;
            return;
        }

        // Allocate storage
        size_t upper_size = (num_faces_ * (num_faces_ - 1)) / 2;
        cos_theta_upper_.resize(upper_size);
        theta_upper_.resize(upper_size);
        unit_centers_.resize(num_faces_);

        // Precompute unit vectors (single precision for cache efficiency)
        #pragma omp parallel for
        for (size_t i = 0; i < num_faces_; ++i) {
            const auto& fc = face_centers[i];
            double r = std::sqrt(fc[0]*fc[0] + fc[1]*fc[1] + fc[2]*fc[2]);
            if (r > 0.0) {
                double inv_r = 1.0 / r;
                unit_centers_[i] = {
                    static_cast<float>(fc[0] * inv_r),
                    static_cast<float>(fc[1] * inv_r),
                    static_cast<float>(fc[2] * inv_r)
                };
            } else {
                unit_centers_[i] = {0.0f, 0.0f, 1.0f};
            }
        }

        // Compute all pairwise cos(theta) and theta values
        // Parallelized by row for good cache behavior
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < num_faces_ - 1; ++i) {
            const auto& ui = unit_centers_[i];
            for (size_t j = i + 1; j < num_faces_; ++j) {
                const auto& uj = unit_centers_[j];
                // Dot product gives cos(theta)
                float cos_t = ui[0]*uj[0] + ui[1]*uj[1] + ui[2]*uj[2];
                // Clamp to [-1, 1] for numerical safety
                cos_t = std::max(-1.0f, std::min(1.0f, cos_t));
                size_t idx = triangularIndex(i, j);
                cos_theta_upper_[idx] = cos_t;
                // Also compute and cache theta (the expensive acos call)
                theta_upper_[idx] = std::acos(cos_t);
            }
        }

        is_valid_ = true;
        theta_cached_ = true;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Distance cache built: " << num_faces_ << " faces, "
                  << upper_size << " pairs, " << duration.count() << " ms, "
                  << (upper_size * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    // Get cos(theta) between faces i and j (O(1) lookup)
    inline float getCosTheta(size_t i, size_t j) const {
        if (i == j) return 1.0f;  // Same face, theta = 0
        if (i > j) std::swap(i, j);  // Ensure i < j
        return cos_theta_upper_[triangularIndex(i, j)];
    }

    // Get angular distance theta in radians (O(1) lookup from cache)
    inline float getThetaCached(size_t i, size_t j) const {
        if (i == j) return 0.0f;  // Same face, theta = 0
        if (i > j) std::swap(i, j);  // Ensure i < j
        return theta_upper_[triangularIndex(i, j)];
    }

    // Get angular distance theta in radians (computes acos if not cached)
    inline double getTheta(size_t i, size_t j) const {
        if (theta_cached_) {
            return static_cast<double>(getThetaCached(i, j));
        }
        return std::acos(static_cast<double>(getCosTheta(i, j)));
    }

    // Get great circle distance in meters
    inline double getDistance(size_t i, size_t j) const {
        return body_radius_ * getTheta(i, j);
    }

    // Check if angular distance is less than threshold (without computing acos)
    // Uses cos(theta) > cos(theta_max) since cos is decreasing on [0,π]
    inline bool isWithinAngle(size_t i, size_t j, double theta_max) const {
        float cos_max = static_cast<float>(std::cos(theta_max));
        return getCosTheta(i, j) > cos_max;
    }

    // Get precomputed unit vector for face
    inline const std::array<float, 3>& getUnitCenter(size_t i) const {
        return unit_centers_[i];
    }

    // Accessors
    bool isValid() const { return is_valid_; }
    size_t numFaces() const { return num_faces_; }
    double bodyRadius() const { return body_radius_; }

    // Memory usage in bytes
    size_t memoryUsage() const {
        return cos_theta_upper_.size() * sizeof(float) +
               theta_upper_.size() * sizeof(float) +
               unit_centers_.size() * sizeof(std::array<float, 3>);
    }

    // Check if theta values are cached
    bool isThetaCached() const { return theta_cached_; }

    // Invalidate cache (e.g., if mesh changes)
    void invalidate() { is_valid_ = false; }

    // Clear all data
    void clear() {
        cos_theta_upper_.clear();
        theta_upper_.clear();
        unit_centers_.clear();
        num_faces_ = 0;
        is_valid_ = false;
        theta_cached_ = false;
    }
};

// Adaptive rebuild controller for transport matrix
class AdaptiveRebuildController {
private:
    std::vector<double> last_temperatures_;
    double relative_threshold_;
    double absolute_threshold_;
    int rebuild_count_;
    int skip_count_;
    bool force_next_rebuild_;

public:
    AdaptiveRebuildController(double rel_thresh = 0.05, double abs_thresh = 2.0)
        : relative_threshold_(rel_thresh),
          absolute_threshold_(abs_thresh),
          rebuild_count_(0),
          skip_count_(0),
          force_next_rebuild_(true) {}

    // Check if rebuild is needed based on temperature changes
    bool needsRebuild(const std::vector<double>& temperatures) {
        if (force_next_rebuild_) {
            force_next_rebuild_ = false;
            rebuild_count_++;
            last_temperatures_ = temperatures;
            return true;
        }

        if (last_temperatures_.empty() ||
            last_temperatures_.size() != temperatures.size()) {
            rebuild_count_++;
            last_temperatures_ = temperatures;
            return true;
        }

        // Check maximum relative and absolute changes
        double max_rel_change = 0.0;
        double max_abs_change = 0.0;

        #pragma omp parallel for reduction(max:max_rel_change,max_abs_change)
        for (size_t i = 0; i < temperatures.size(); ++i) {
            double T_old = last_temperatures_[i];
            double T_new = temperatures[i];

            if (T_old > 0.0) {
                double rel = std::abs(T_new - T_old) / T_old;
                max_rel_change = std::max(max_rel_change, rel);
            }
            double abs_diff = std::abs(T_new - T_old);
            max_abs_change = std::max(max_abs_change, abs_diff);
        }

        bool needs_rebuild = (max_rel_change > relative_threshold_) ||
                            (max_abs_change > absolute_threshold_);

        if (needs_rebuild) {
            rebuild_count_++;
            last_temperatures_ = temperatures;
        } else {
            skip_count_++;
        }

        return needs_rebuild;
    }

    // Force rebuild on next check
    void forceRebuild() { force_next_rebuild_ = true; }

    // Get statistics
    int getRebuildCount() const { return rebuild_count_; }
    int getSkipCount() const { return skip_count_; }
    double getSkipRatio() const {
        int total = rebuild_count_ + skip_count_;
        return total > 0 ? static_cast<double>(skip_count_) / total : 0.0;
    }

    // Configuration
    void setRelativeThreshold(double thresh) { relative_threshold_ = thresh; }
    void setAbsoluteThreshold(double thresh) { absolute_threshold_ = thresh; }
    double getRelativeThreshold() const { return relative_threshold_; }
    double getAbsoluteThreshold() const { return absolute_threshold_; }

    // Print statistics
    void printStats() const {
        std::cout << "Adaptive rebuild stats: "
                  << rebuild_count_ << " rebuilds, "
                  << skip_count_ << " skipped ("
                  << std::fixed << std::setprecision(1)
                  << (getSkipRatio() * 100.0) << "% skip rate)" << std::endl;
    }
};

} // namespace exosphere
} // namespace ptm

#endif // DISTANCE_CACHE_H
