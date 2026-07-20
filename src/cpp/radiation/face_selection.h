#pragma once

#include "radiation_config.h"
#include "radiation_exchange.h"

// This header provides the FaceSelector class which is implemented
// in radiation_exchange.cpp. It's included here for completeness
// and to match the module structure defined in the implementation plan.

namespace radiation {

// FaceSelector is declared in radiation_exchange.h
// See that file for the full interface.

// Additional utility functions for face selection

// Compute horizon angles for each face using ray tracing
// Returns vector of maximum horizon elevation angles [degrees]
std::vector<double> computeHorizonAngles(
    const std::shared_ptr<ptm::Mesh>& mesh,
    void* embree_device,
    void* embree_scene,
    int num_azimuth_samples = 36);

// Identify permanently shadowed regions (PSR)
// Returns indices of faces that are never directly illuminated
std::vector<int> findPermanentlyShadowedFaces(
    const std::vector<std::vector<double>>& illumination_series,
    double threshold = 1e-6);

// Identify crater-like depressions based on geometry
// Returns indices of faces in topographic depressions
std::vector<int> findDepressionFaces(
    const std::shared_ptr<ptm::Mesh>& mesh,
    double depth_threshold = 0.1);

} // namespace radiation
