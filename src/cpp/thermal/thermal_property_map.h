// thermal_property_map.h
//
// Per-face thermophysical-property map loader.
//
// Supports two input modes:
//   * H_PARAMETER     — values are e-folding depths H (m) of the regolith
//                       density/conductivity profile (Hayne et al. 2017 Diviner fits).
//   * THERMAL_INERTIA — values are TI(273K) in J m^-2 K^-1 s^-1/2; the loader
//                       inverts to an effective H per face via bisection so that
//                       the rest of the thermal model (which is parameterized by H)
//                       can consume the result directly.
//
// File layout (binary, little-endian, host-order; matches the offline tool
//   `lunar_property_map_generator`):
//   * <path>           — float32, n_faces values, in mesh-face-index order
//   * <path>.mask      — uint8,  n_faces values (1 = sampled from raster, 0 = default)
//                        (optional; if absent, all faces are flagged valid)
//
// Bundled-default discovery:
//   loadDefault(body, ico_level, root, mode) probes
//     <root>/<body>/{hpar|ti273k}_<body>_ico<level>.bin
//   The companion mask file is loaded if present.

#ifndef PTM_THERMAL_PROPERTY_MAP_H
#define PTM_THERMAL_PROPERTY_MAP_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "planetary_properties.h"

namespace thermal {

class ThermalPropertyMap {
public:
    enum class Mode { H_PARAMETER, THERMAL_INERTIA };

    // Auto-discover the bundled icosphere map for (body, ico_level, mode).
    // Returns std::nullopt if the file does not exist or fails to load.
    static std::optional<ThermalPropertyMap> loadDefault(
        const std::string& body,
        int ico_level,
        const std::string& maps_root,
        Mode mode);

    // Load a user-supplied custom file. expected_n_faces must equal the mesh
    // face count or loading fails.
    static std::optional<ThermalPropertyMap> loadCustom(
        const std::string& path,
        std::size_t expected_n_faces,
        Mode mode);

    // After construction, the H values (always — TI inputs are inverted to H).
    const std::vector<double>& H_values() const { return H_; }

    // Per-face validity (1 = sampled from raster; 0 = body-default fallback).
    const std::vector<std::uint8_t>& valid_mask() const { return mask_; }

    // Fraction of faces using the body-default fallback, in [0, 1].
    double fraction_default() const;

    // Always H_PARAMETER after construction (TI inputs are converted in-place).
    Mode mode() const { return Mode::H_PARAMETER; }

    Mode source_mode() const { return source_mode_; }

    const std::string& source_path() const { return source_path_; }

private:
    ThermalPropertyMap() = default;

    // Loaders read the on-disk float32 array into raw_, optionally a sidecar
    // mask file into mask_, and finalize H_ depending on Mode.
    bool readBinaryFloat32(const std::string& path,
                           std::size_t expected_n,
                           std::vector<float>& out);
    bool readBinaryUInt8(const std::string& path,
                         std::size_t expected_n,
                         std::vector<std::uint8_t>& out);

    // For TI mode: invert each TI to an effective H using the same forward
    // integral as `lunar_property_map_generator::depthWeightedTI`.
    // Body defaults supply ks/kd/rho_s/rho_d/cp(273)/day.
    void invertTIToH(const ThermophysicalProperties& body);

    std::vector<double>          H_;
    std::vector<std::uint8_t>    mask_;
    Mode                          source_mode_ = Mode::H_PARAMETER;
    std::string                   source_path_;

    // Body defaults used during inversion + polar fallback (set on load).
    ThermophysicalProperties      body_defaults_;
    bool                          body_defaults_set_ = false;
};

// Helper exposed for unit-tests / Phase-4 round-trip checks.
double depthWeightedTI(const ThermophysicalProperties& body, double H_face);

}  // namespace thermal

#endif  // PTM_THERMAL_PROPERTY_MAP_H
