#include "ptm_output_reader.h"
#include "../fileio/frame_naming.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

namespace ptm {

bool PTMOutputReader::loadFromDirectory(const std::string& output_dir) {
    output_dir_ = output_dir;

    if (!fs::exists(output_dir)) {
        std::cerr << "[PTMOutputReader] Directory not found: " << output_dir << std::endl;
        return false;
    }

    // Load mesh
    std::string mesh_path = output_dir + "/mesh.ply";
    if (!loadMesh(mesh_path)) {
        std::cerr << "[PTMOutputReader] Failed to load mesh from " << mesh_path << std::endl;
        return false;
    }

    // Load time steps
    std::string time_path = output_dir + "/time_steps.txt";
    if (!loadTimeSteps(time_path)) {
        std::cerr << "[PTMOutputReader] Failed to load time steps from " << time_path << std::endl;
        return false;
    }

    // Check which data fields are available
    has_temperature_ = fs::exists(output_dir + "/temperature");
    has_illumination_ = fs::exists(output_dir + "/illumination");
    has_direct_solar_ = fs::exists(output_dir + "/direct_solar");
    has_primary_reflected_ = fs::exists(output_dir + "/primary_reflected");

    std::cout << "[PTMOutputReader] Loaded " << getNumFaces() << " faces, "
              << getNumTimeSteps() << " time steps" << std::endl;
    std::cout << "[PTMOutputReader] Available: "
              << (has_temperature_ ? "temperature " : "")
              << (has_illumination_ ? "illumination " : "")
              << (has_direct_solar_ ? "direct_solar " : "")
              << (has_primary_reflected_ ? "primary_reflected " : "")
              << std::endl;

    return true;
}

bool PTMOutputReader::loadMesh(const std::string& mesh_file) {
    MeshUtils mesh_utils;
    if (!mesh_utils.loadMeshFromFile(mesh_file)) {
        return false;
    }
    mesh_ = mesh_utils.getMesh();
    if (!mesh_) return false;

    // Ensure face properties are computed
    if (mesh_->face_normals.empty() || mesh_->face_centers.empty()) {
        mesh_utils.calculateFaceProperties(true);
        mesh_ = mesh_utils.getMesh();
    }

    return true;
}

bool PTMOutputReader::loadTimeSteps(const std::string& time_file) {
    std::ifstream file(time_file);
    if (!file.is_open()) {
        std::cerr << "[PTMOutputReader] Cannot open " << time_file << std::endl;
        return false;
    }

    times_et_.clear();
    time_strings_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double et;
        std::string utc;
        if (iss >> et >> utc) {
            times_et_.push_back(et);
            time_strings_.push_back(utc);
        }
    }

    if (times_et_.empty()) {
        std::cerr << "[PTMOutputReader] No time steps found in " << time_file << std::endl;
        return false;
    }

    return true;
}

int PTMOutputReader::getNumFaces() const {
    if (!mesh_) return 0;
    return static_cast<int>(mesh_->faces.size());
}

std::vector<float> PTMOutputReader::readBinaryFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[PTMOutputReader] Cannot open binary file: " << filepath << std::endl;
        return {};
    }

    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
    if (!file.good() || count == 0 || count > 100000000) {
        std::cerr << "[PTMOutputReader] Invalid count in " << filepath << ": " << count << std::endl;
        return {};
    }

    std::vector<float> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!file.good()) {
        std::cerr << "[PTMOutputReader] Read error in " << filepath << std::endl;
        return {};
    }

    return data;
}

std::string PTMOutputReader::buildFilename(const std::string& field, int step_index) const {
    // Either naming generation (BUGS.md PTM-664): >= 6-digit indices from
    // current engines, 4-digit from runs written before. When neither exists
    // the current-rule name is returned so the caller's "cannot open" message
    // names a real candidate.
    const std::string dir = output_dir_ + "/" + field;
    const std::string found = ptm::resolveFrameFile(dir, field, step_index);
    if (!found.empty()) return found;
    return dir + "/" + field + "_" + ptm::frameIndexString(step_index, 0) + ".bin";
}

std::vector<float> PTMOutputReader::loadTimestepData(const std::string& field,
                                                      int step_index) const {
    std::string key = field + "_" + std::to_string(step_index);

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    // Load from disk
    std::string filepath = buildFilename(field, step_index);
    auto data = readBinaryFile(filepath);

    if (!data.empty()) {
        evictCache();
        cache_[key] = data;
    }

    return data;
}

void PTMOutputReader::evictCache() const {
    while (cache_.size() >= MAX_CACHE_ENTRIES) {
        cache_.erase(cache_.begin());
    }
}

std::pair<int, int> PTMOutputReader::findBracketingSteps(double et) const {
    if (times_et_.empty()) return {-1, -1};

    // Before first timestep
    if (et <= times_et_.front()) return {0, 0};
    // After last timestep
    if (et >= times_et_.back()) {
        int last = static_cast<int>(times_et_.size()) - 1;
        return {last, last};
    }

    // Binary search for bracketing indices
    auto it = std::lower_bound(times_et_.begin(), times_et_.end(), et);
    int upper = static_cast<int>(std::distance(times_et_.begin(), it));
    int lower = upper - 1;
    return {lower, upper};
}

std::vector<float> PTMOutputReader::getFieldAtET(const std::string& field, double et,
                                                  Interpolation mode) const {
    auto [lo, hi] = findBracketingSteps(et);
    if (lo < 0) return {};

    auto data_lo = loadTimestepData(field, lo);
    if (data_lo.empty()) return {};

    if (mode == Interpolation::NEAREST || lo == hi) {
        // Snap to nearest
        if (lo != hi) {
            double dt_lo = std::abs(et - times_et_[lo]);
            double dt_hi = std::abs(et - times_et_[hi]);
            if (dt_hi < dt_lo) {
                return loadTimestepData(field, hi);
            }
        }
        return data_lo;
    }

    // Linear interpolation
    auto data_hi = loadTimestepData(field, hi);
    if (data_hi.empty() || data_hi.size() != data_lo.size()) return data_lo;

    double t_lo = times_et_[lo];
    double t_hi = times_et_[hi];
    double dt = t_hi - t_lo;
    if (dt <= 0.0) return data_lo;

    double alpha = (et - t_lo) / dt;

    std::vector<float> result(data_lo.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<float>((1.0 - alpha) * data_lo[i] + alpha * data_hi[i]);
    }

    return result;
}

std::vector<float> PTMOutputReader::getTemperatureAtET(double et, Interpolation mode) const {
    return getFieldAtET("temperature", et, mode);
}

std::vector<float> PTMOutputReader::getIlluminationAtET(double et, Interpolation mode) const {
    return getFieldAtET("illumination", et, mode);
}

std::vector<float> PTMOutputReader::getDirectSolarAtET(double et, Interpolation mode) const {
    return getFieldAtET("direct_solar", et, mode);
}

std::vector<float> PTMOutputReader::getPrimaryReflectedAtET(double et, Interpolation mode) const {
    return getFieldAtET("primary_reflected", et, mode);
}

} // namespace ptm
