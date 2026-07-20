#pragma once

#include "camera_simulator.h"

#include <string>

namespace ptm {

// Load a camera-definition YAML file and merge its contents into `config`.
// Fields not mentioned in the YAML are left at their existing values (defaults
// or previously-applied CLI flags), so the call order is: defaults → YAML
// (loadCameraYaml) → CLI overrides.
//
// Returns true on successful parse. Prints a diagnostic on failure.
//
// Expected structure (all keys optional):
//
//   camera:
//     focal_length_mm: 700.0
//     sensor_mm: [36.0, 24.0]          # or
//     pixel_pitch_mm: [0.012, 0.012]   # [px, py]; overrides sensor_mm when > 0
//     fov_x_deg: 10.0                  # overrides focal length if > 0
//     nx: 2048
//     ny: 2048
//     principal_point: [1024.0, 1024.0]
//     distortion:
//       model: brown_conrady
//       k1: 0.0
//       k2: 0.0
//       k3: 0.0
//       p1: 0.0
//       p2: 0.0
//     pointing:
//       mode: look_at      # look_at | quaternion | rotation_matrix | spice_ck
//       look_at: [0, 0, 0]
//       up:      [0, 0, 1]
//       # quaternion: [w, x, y, z]
//       # rotation_matrix: [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
//       # instrument_name: ARTEMIS_II_CAM   (for spice_ck)
//     spice:
//       instrument_name: ARTEMIS_II_CAM
//       ik_kernel:  kernels/ik/artemis_cam.ti
//       ck_kernel:  kernels/ck/artemis_nominal.bc
//     photometry:
//       reflectance: lommel_seeliger
//       albedo:      0.12
//       emissivity:  0.95
bool loadCameraYaml(const std::string& filepath, CameraSimConfig& config);

} // namespace ptm
