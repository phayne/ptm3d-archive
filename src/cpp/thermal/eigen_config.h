#ifndef PTM_THERMAL_EIGEN_CONFIG_H
#define PTM_THERMAL_EIGEN_CONFIG_H

// Define this before including any Eigen headers
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG // Disable Eigen assertions that might use infinity
#endif

// Define safe large values to use instead of infinity
#define EIGEN_SAFE_INFINITY_FLOAT 1.0e30f
#define EIGEN_SAFE_INFINITY_DOUBLE 1.0e30

// Include this pragma to temporarily disable the specific warning
// This is cleaner than trying to patch Eigen's functions
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#endif

// Include Eigen core headers
#include <Eigen/Dense>

// Restore warning state after Eigen headers
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif // PTM_THERMAL_EIGEN_CONFIG_H