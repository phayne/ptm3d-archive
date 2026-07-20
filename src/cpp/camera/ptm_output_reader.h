#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <array>
#include "../mesh/mesh_utils.h"

namespace ptm {

class PTMOutputReader {
public:
    PTMOutputReader() = default;

    // Load all metadata from a PTM3D binary output directory
    bool loadFromDirectory(const std::string& output_dir);

    // Load mesh separately
    bool loadMesh(const std::string& mesh_file);

    // Load time steps separately
    bool loadTimeSteps(const std::string& time_file);

    // Get per-face data at a given ET, with temporal interpolation
    enum class Interpolation { NEAREST, LINEAR };
    std::vector<float> getTemperatureAtET(double et,
                                          Interpolation mode = Interpolation::LINEAR) const;
    std::vector<float> getIlluminationAtET(double et,
                                           Interpolation mode = Interpolation::LINEAR) const;
    std::vector<float> getDirectSolarAtET(double et,
                                          Interpolation mode = Interpolation::LINEAR) const;
    std::vector<float> getPrimaryReflectedAtET(double et,
                                               Interpolation mode = Interpolation::LINEAR) const;

    // Direct timestep access (no interpolation)
    std::vector<float> loadTimestepData(const std::string& field, int step_index) const;

    // Accessors
    std::shared_ptr<Mesh> getMesh() const { return mesh_; }
    const std::vector<double>& getTimesET() const { return times_et_; }
    const std::vector<std::string>& getTimeStrings() const { return time_strings_; }
    int getNumFaces() const;
    int getNumTimeSteps() const { return static_cast<int>(times_et_.size()); }
    const std::string& outputDir() const { return output_dir_; }

    // Check which data fields are available
    bool hasTemperature() const { return has_temperature_; }
    bool hasIllumination() const { return has_illumination_; }
    bool hasDirectSolar() const { return has_direct_solar_; }
    bool hasPrimaryReflected() const { return has_primary_reflected_; }

private:
    std::shared_ptr<Mesh> mesh_;
    std::vector<double> times_et_;
    std::vector<std::string> time_strings_;
    std::string output_dir_;

    bool has_temperature_ = false;
    bool has_illumination_ = false;
    bool has_direct_solar_ = false;
    bool has_primary_reflected_ = false;

    // LRU cache for loaded timestep data
    mutable std::map<std::string, std::vector<float>> cache_;
    static constexpr size_t MAX_CACHE_ENTRIES = 20;
    void evictCache() const;

    // Read a single binary file: uint32 count + float[count]
    static std::vector<float> readBinaryFile(const std::string& filepath);

    // Build filename for a given field and step index
    std::string buildFilename(const std::string& field, int step_index) const;

    // Find bracketing timestep indices for interpolation
    std::pair<int, int> findBracketingSteps(double et) const;

    // Generic interpolated data retrieval
    std::vector<float> getFieldAtET(const std::string& field, double et,
                                    Interpolation mode) const;
};

} // namespace ptm
