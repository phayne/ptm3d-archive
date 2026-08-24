#pragma once

#include <array>

/**
 * @file mc_view_factor.h
 * @brief Monte-Carlo ray-gather view factor construction for large meshes.
 *
 * For each source face, shoot N cosine-weighted hemispherical rays and count
 * first-hit primIDs. F_{ij} ≈ hits[j] / N. Output is a sparse CSR
 * ViewFactorMatrix that plugs into the existing radiosity solver unchanged.
 *
 * This path exists to scale beyond the H-matrix limit on 4M+ meshes where
 * H-matrix build memory exceeds available RAM and sparse-VF + adaptive
 * neighbors truncates accuracy at the max_neighbors cap.
 */

#include <memory>

#include "view_factor.h"

// Forward declarations for Embree (pulled in from view_factor.h)
typedef struct RTCDeviceTy* RTCDevice;
typedef struct RTCSceneTy* RTCScene;

namespace radiation {

/**
 * @brief Build view factor matrix via Monte Carlo ray-gather.
 *
 * @param mesh          Triangulated mesh (faces are the source/target set).
 * @param device        Committed Embree device.
 * @param scene         Committed Embree scene built from @p mesh.
 * @param config        Radiation configuration. Fields consumed:
 *                        - max_neighbors: per-face top-K cap on unique targets
 *                        - visibility_threshold: min F_ij to retain (pre-top-K)
 * @param rays_per_face Number of cosine-weighted rays per source face.
 * @param scene_origin_shift  The constant translation the CALLER applied to
 *                      @p scene's vertices before their float32 conversion.
 *                      Ray origins are translated by it. Default zero, for a
 *                      scene built at mesh coordinates.
 *
 *                      This used to be recomputed here from the mesh bounding
 *                      box on the assumption that the caller had recentered the
 *                      scene the same way. A caller who had not -- any harness,
 *                      any gate -- got ray origins displaced by the whole
 *                      bounding-box offset and a matrix built from rays that
 *                      mostly missed the geometry, with no error anywhere. It
 *                      is a parameter now for the same reason
 *                      ViewFactorCalculator::setSceneOriginShift is.
 * @return Sparse CSR view factor matrix F with row_i · 1 ≤ 1, indexed by face.
 */
ViewFactorMatrix computeMonteCarlo(
    const std::shared_ptr<ptm::Mesh>& mesh,
    RTCDevice device,
    RTCScene scene,
    const RadiationConfig& config,
    int rays_per_face,
    const std::array<double, 3>& scene_origin_shift = {{0.0, 0.0, 0.0}});

}  // namespace radiation
