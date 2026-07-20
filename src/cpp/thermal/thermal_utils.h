#ifndef THERMAL_UTILS_H
#define THERMAL_UTILS_H

#include "thermal_model.h"
#include "thermal_adaptive.h"
#include "thermal_adaptive_optimized.h"
#include <memory>
#include <vector>
#include <iostream>

namespace thermal {

// Configuration for adaptive thermal model
struct AdaptiveConfig {
    bool use_optimized = true;        // Use optimized implementation
    bool use_wavelets = true;         // Use wavelet analysis
    bool use_ml_predictor = true;     // Use ML time step prediction
    bool enable_work_stealing = true; // Enable work stealing for parallel processing
    int max_zones = 20;               // Maximum number of adaptive zones
    double stability_factor = 0.95;   // Stability factor for time stepping
    bool show_performance = false;    // Show performance metrics
    int report_interval = 1000;       // Performance reporting interval
};

// Factory function to create appropriate adaptive stepper
std::unique_ptr<AdaptiveStepper> createAdaptiveStepper(
    const Profile& profile,
    const AdaptiveConfig& config = AdaptiveConfig()) {
    
    if (config.use_optimized) {
        auto stepper = std::make_unique<OptimizedAdaptiveStepper>(profile);
        
        // Configure the optimized stepper
        // Note: These would be public methods you'd add to the class
        // stepper->setMaxZones(config.max_zones);
        // stepper->setStabilityFactor(config.stability_factor);
        
        if (config.show_performance) {
            std::cout << "Using OPTIMIZED adaptive thermal model with:" << std::endl;
            std::cout << "  - Wavelet analysis: " << (config.use_wavelets ? "ON" : "OFF") << std::endl;
            std::cout << "  - ML predictor: " << (config.use_ml_predictor ? "ON" : "OFF") << std::endl;
            std::cout << "  - Work stealing: " << (config.enable_work_stealing ? "ON" : "OFF") << std::endl;
            std::cout << "  - Max zones: " << config.max_zones << std::endl;
#ifdef AVX512_ENABLED
            std::cout << "  - AVX-512: ENABLED" << std::endl;
#elif defined(AVX2_ENABLED)
            std::cout << "  - AVX2: ENABLED" << std::endl;
#else
            std::cout << "  - SIMD: Scalar fallback" << std::endl;
#endif
        }
        
        return stepper;
    } else {
        // Fall back to standard adaptive stepper
        if (config.show_performance) {
            std::cout << "Using STANDARD adaptive thermal model" << std::endl;
        }
        return std::make_unique<AdaptiveStepper>(profile);
    }
}

} // namespace thermal

#endif // THERMAL_UTILS_H