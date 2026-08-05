/**
 * PRISM Dark Horse Viewshed Analysis Example
 * 
 * Demonstrates using the ViewshedAnalyzer module for:
 * 1. Loading traverse waypoints from CSV file
 * 2. Computing viewsheds along traverse paths
 * 3. Integrating thermal model results for enhanced analysis
 * 4. Generating cumulative coverage maps
 */

#include "viewshed_analyzer.h"
#include "../thermal/thermal_model.h"
// Removed missing VTK include; example will skip VTK-based thermal loading
// #include "../fileio/vtk_io.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <set>
#include <algorithm>
#include <cctype>
#ifdef HAS_HDF5
#include <H5Cpp.h>
#include <hdf5.h>
#endif
#ifdef PTM_USE_VTK_READER
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkStringArray.h>
#include <vtkPoints.h>
#endif

namespace fs = std::filesystem;
using namespace ptm;

// Structure to hold thermal data at each waypoint
struct ThermalData {
    double temperature_min;
    double temperature_max;
    double temperature_mean;
    double solar_flux_mean;
    bool has_data;
    
    ThermalData() : temperature_min(0), temperature_max(0), 
                   temperature_mean(0), solar_flux_mean(0), has_data(false) {}
};

// Series of temperatures across timesteps or statistic types
struct ThermalSeries {
    // temperatures[time_index][face_index]
    std::vector<std::vector<double>> temperatures;
    // labels per time index (e.g., "mean","max","min" or ISO timestamps)
    std::vector<std::string> labels;
    // true if represents statistics (mean/max/min) rather than time series
    bool is_statistics = false;
};

// Helper function to print results summary
void printViewshedSummary(const ViewshedResult& result, 
                          const std::string& location_name,
                          const ThermalData& thermal = ThermalData()) {
    std::cout << "\n=== Viewshed Analysis: " << location_name << " ===" << std::endl;
    std::cout << "Location: Lat=" << std::fixed << std::setprecision(3) 
              << result.location.latitude << "°, Lon=" 
              << result.location.longitude << "°" << std::endl;
    std::cout << "Height above surface: " << result.location.height << " m" << std::endl;
    
    std::cout << "\nVisibility Statistics:" << std::endl;
    std::cout << "  Visible faces: " << result.num_visible_faces 
              << " (" << std::setprecision(1) 
              << (100.0 * result.num_visible_faces / result.visibility.size()) 
              << "%)" << std::endl;
    std::cout << "  Visible area: " << std::setprecision(2) 
              << result.visible_area_km2 << " km²" << std::endl;
    std::cout << "  Max range: " << result.max_visible_range_km << " km" << std::endl;
    std::cout << "  Mean range: " << result.mean_visible_range_km << " km" << std::endl;
    
    // Display thermal data if available
    if (thermal.has_data) {
        std::cout << "\nThermal Environment:" << std::endl;
        std::cout << "  Temperature range: " << std::setprecision(1) 
                  << thermal.temperature_min << " - " 
                  << thermal.temperature_max << " K" << std::endl;
        std::cout << "  Mean temperature: " << thermal.temperature_mean << " K" << std::endl;
        std::cout << "  Mean solar flux: " << std::setprecision(2) 
                  << thermal.solar_flux_mean << " W/m^2" << std::endl;
    }
    
    std::cout << "\nComputation time: " << std::setprecision(2) 
              << result.computation_time_ms << " ms" << std::endl;
}

// Load traverse waypoints from CSV file
std::vector<ViewLocation> loadTraverseWaypoints(const std::string& filename) {
    std::vector<ViewLocation> waypoints;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open waypoints file: " << filename << std::endl;
        return waypoints;
    }
    
    std::string line;
    int line_num = 0;
    
    std::cout << "Loading traverse waypoints from: " << filename << std::endl;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Skip header if present
        if (line.find("label") != std::string::npos || 
            line.find("lat") != std::string::npos) {
            continue;
        }
        
        std::istringstream iss(line);
        std::string label;
        double lat, lon, height = 2.0;  // Default height for rover
        
        // Parse CSV: label,lat,lon[,height]
        if (std::getline(iss, label, ',')) {
            std::string lat_str, lon_str, height_str;
            
            if (std::getline(iss, lat_str, ',') && std::getline(iss, lon_str, ',')) {
                try {
                    lat = std::stod(lat_str);
                    lon = std::stod(lon_str);
                    
                    // Optional height
                    if (std::getline(iss, height_str, ',')) {
                        height = std::stod(height_str);
                    }

                    // Optional UTC timestamp
                    std::string timestamp_str;
                    if (std::getline(iss, timestamp_str, ',')) {
                        // Trim leading/trailing whitespace
                        size_t start = timestamp_str.find_first_not_of(" \t\r\n");
                        size_t end = timestamp_str.find_last_not_of(" \t\r\n");
                        if (start != std::string::npos)
                            timestamp_str = timestamp_str.substr(start, end - start + 1);
                        else
                            timestamp_str.clear();
                    }

                    // Remove quotes from label if present
                    if (!label.empty() && label.front() == '"') {
                        label = label.substr(1);
                    }
                    if (!label.empty() && label.back() == '"') {
                        label.pop_back();
                    }

                    ViewLocation loc(lat, lon, height, label);
                    loc.utc_timestamp = timestamp_str;
                    waypoints.push_back(loc);

                    std::cout << "  Waypoint " << waypoints.size() << ": "
                              << label << " (Lat=" << lat << "°, Lon=" << lon
                              << "°, H=" << height << "m"
                              << (timestamp_str.empty() ? "" : ", T=" + timestamp_str)
                              << ")" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Error parsing line " << line_num 
                              << ": " << e.what() << std::endl;
                }
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << waypoints.size() << " waypoints" << std::endl;
    return waypoints;
}

// Load thermal model results from VTU/PVD or HDF5
// Load thermal model results; prefer HDF5 multi-dataset to support timesteps/statistics
static ThermalSeries loadThermalSeries(const std::string& thermal_file,
                                       const std::shared_ptr<Mesh>& mesh) {
    ThermalSeries series;

    if (thermal_file.empty() || !fs::exists(thermal_file)) {
        std::cout << "No thermal model file found, proceeding without thermal data" << std::endl;
        return series;
    }

    std::cout << "Loading thermal model results from: " << thermal_file << std::endl;

    // Determine file format
    std::string ext = fs::path(thermal_file).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".vtu" || ext == ".pvd" || ext == ".vtk") {
#ifdef PTM_USE_VTK_READER
        auto readArrayToVector = [](vtkDataArray* arr) {
            std::vector<double> v;
            if (!arr) return v;
            const vtkIdType n = arr->GetNumberOfTuples();
            v.resize(static_cast<size_t>(n));
            for (vtkIdType i = 0; i < n; ++i) {
                v[static_cast<size_t>(i)] = arr->GetComponent(i, 0);
            }
            return v;
        };

        auto loadVTU = [&](const std::string& vtu_path, const std::string& label_hint){
            vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
            reader->SetFileName(vtu_path.c_str());
            reader->Update();
            vtkUnstructuredGrid* ug = reader->GetOutput();
            if (!ug) return;
            vtkCellData* cd = ug->GetCellData();
            if (!cd) return;
            // Prefer selecting exactly one array per VTU based on label_hint
            vtkDataArray* chosen = nullptr;
            std::string out_label = label_hint;

            if (label_hint == "avg" || label_hint == "statistics_avg") {
                chosen = cd->GetArray("temperature_avg");
                if (!chosen) chosen = cd->GetArray("temperature_mean");
                if (!chosen) chosen = cd->GetArray("temperature");
                out_label = "avg";
                series.is_statistics = true;
            } else if (label_hint == "max" || label_hint == "statistics_max") {
                chosen = cd->GetArray("temperature_max");
                if (!chosen) chosen = cd->GetArray("temperature");
                out_label = "max";
                series.is_statistics = true;
            } else if (label_hint == "min" || label_hint == "statistics_min") {
                chosen = cd->GetArray("temperature_min");
                if (!chosen) chosen = cd->GetArray("temperature");
                out_label = "min";
                series.is_statistics = true;
            } else {
                // No hint: fallback to a reasonable default order
                chosen = cd->GetArray("temperature");
                if (!chosen) chosen = cd->GetArray("temperature_avg");
                if (!chosen) chosen = cd->GetArray("temperature_mean");
                if (!chosen) chosen = cd->GetArray("temperature_max");
                if (!chosen) chosen = cd->GetArray("temperature_min");
                out_label = label_hint.empty() ? fs::path(vtu_path).stem().string() : label_hint;
            }

            if (chosen) {
                series.temperatures.push_back(readArrayToVector(chosen));
                series.labels.push_back(out_label);
            }
        };

        if (ext == ".vtu") {
            loadVTU(thermal_file, "");
            return series;
        }

        if (ext == ".pvd") {
            // Parse PVD XML manually to collect referenced VTU files
            std::ifstream fin(thermal_file);
            if (!fin.is_open()) return series;
            std::string xml((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            fin.close();
            // Heuristic parse: look for file="...vtu" and timestep value
            std::vector<std::pair<std::string,std::string>> files; // (path,label)
            size_t pos = 0;
            while (true) {
                auto fpos = xml.find("file=\"", pos);
                if (fpos == std::string::npos) break;
                fpos += 6;
                auto fend = xml.find("\"", fpos);
                if (fend == std::string::npos) break;
                std::string rel = xml.substr(fpos, fend - fpos);
                // timestep attribute
                std::string label;
                auto tpos = xml.rfind("timestep=\"", fpos);
                if (tpos != std::string::npos) {
                    tpos += 10; auto tend = xml.find("\"", tpos); if (tend != std::string::npos) label = xml.substr(tpos, tend - tpos);
                }
                std::string vtu_path = fs::path(thermal_file).parent_path().string() + "/" + rel;
                files.emplace_back(vtu_path, label);
                pos = fend + 1;
            }

            // Determine type: statistics vs simulation
            bool is_statistics = fs::path(thermal_file).stem().string().find("statistics") != std::string::npos;
            for (auto& pr : files) {
                std::string label_hint = pr.second;
                if (is_statistics && label_hint.empty()) {
                    // Infer from filename: statistics_avg.vtu, statistics_max.vtu, statistics_min.vtu
                    std::string stem = fs::path(pr.first).stem().string();
                    if (stem.find("avg") != std::string::npos) label_hint = "avg";
                    else if (stem.find("max") != std::string::npos) label_hint = "max";
                    else if (stem.find("min") != std::string::npos) label_hint = "min";
                }
                loadVTU(pr.first, label_hint);
            }
            return series;
        }

        return series;
#else
        std::cout << "VTU/PVD requested, but VTK reader is not enabled in this build." << std::endl;
        return series;
#endif
    }
#ifdef HAS_HDF5
    else if (ext == ".h5" || ext == ".hdf5") {
        try {
            H5::H5File file(thermal_file, H5F_ACC_RDONLY);
            // Collect dataset names at root
            std::vector<std::string> dataset_names;
            // Iterate names (HDF5 API to list objects)
            hsize_t idx = 0;
            auto lister = [](hid_t /*grp*/, const char* name, const H5L_info_t* /*info*/, void* op_data) -> herr_t {
                auto vec = static_cast<std::vector<std::string>*>(op_data);
                vec->emplace_back(name);
                return 0;
            };
            H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_INC, &idx, lister, &dataset_names);

            auto has_ds = [&](const std::string& n){ return std::find(dataset_names.begin(), dataset_names.end(), n) != dataset_names.end(); };

            // Case 1: statistics datasets mean/min/max
            if (has_ds("temperature_mean") || has_ds("temperature_min") || has_ds("temperature_max")) {
                series.is_statistics = true;
                std::vector<std::pair<std::string,std::string>> stats = {
                    {"temperature_mean","mean"},
                    {"temperature_max","max"},
                    {"temperature_min","min"}
                };
                for (const auto& [dsname,label] : stats) {
                    if (!has_ds(dsname)) continue;
                    H5::DataSet ds = file.openDataSet(dsname);
                    H5::DataSpace space = ds.getSpace();
                    if (space.getSimpleExtentNdims() != 1) continue;
                    hsize_t n = 0; space.getSimpleExtentDims(&n, nullptr);
                    std::vector<double> temps(n, 0.0);
                    ds.read(temps.data(), H5::PredType::NATIVE_DOUBLE);
                    series.temperatures.push_back(std::move(temps));
                    series.labels.push_back(label);
                }
                std::cout << "  Loaded statistics datasets: " << series.labels.size() << std::endl;
            }

            // Case 2: simulation timesteps in a group or datasets like temperature_tXXXX
            // Heuristic: any dataset with prefix "temperature_t" or group "temperature"
            std::vector<std::string> timestep_ds;
            for (const auto& n : dataset_names) {
                if (n.rfind("temperature_t", 0) == 0) timestep_ds.push_back(n);
            }
            // Also check group "temperature"
            if (file.nameExists("temperature")) {
                H5::Group g = file.openGroup("temperature");
                hsize_t gi = 0;
                std::vector<std::string> group_names;
                H5Literate(g.getId(), H5_INDEX_NAME, H5_ITER_INC, &gi, lister, &group_names);
                for (const auto& n : group_names) {
                    // Expect datasets inside group
                    H5O_info_t info; H5Oget_info_by_name(g.getId(), n.c_str(), &info, H5P_DEFAULT);
                    if (info.type == H5O_TYPE_DATASET) {
                        timestep_ds.push_back("temperature/" + n);
                    }
                }
            }

            if (!timestep_ds.empty()) {
                // Sort for consistent order
                std::sort(timestep_ds.begin(), timestep_ds.end());
                for (const auto& name : timestep_ds) {
                    H5::DataSet ds = (name.find('/') != std::string::npos)
                        ? file.openDataSet(name)
                        : file.openDataSet(name);
                    H5::DataSpace space = ds.getSpace();
                    if (space.getSimpleExtentNdims() != 1) continue;
                    hsize_t n = 0; space.getSimpleExtentDims(&n, nullptr);
                    std::vector<double> temps(n, 0.0);
                    ds.read(temps.data(), H5::PredType::NATIVE_DOUBLE);
                    series.temperatures.push_back(std::move(temps));
                    // Derive label
                    std::string lbl = name;
                    auto pos = lbl.find_last_of('/'); if (pos != std::string::npos) lbl = lbl.substr(pos+1);
                    series.labels.push_back(lbl);
                }
                std::cout << "  Loaded simulation timesteps: " << series.labels.size() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "HDF5 load failed: " << e.what() << std::endl;
        }
    }
#endif
    else {
        std::cerr << "Warning: Unknown thermal file format: " << ext << std::endl;
    }

    return series;
}

// Compute temperature histogram for a set of visible faces
static void computeAndWriteTemperatureHistogram(
        const ViewshedResult& result,
        const std::map<int, ThermalData>& thermal_map,
        const std::string& output_dir,
        double min_k = 0.0,
        double max_k = 400.0,
        double bin_size_k = 1.0,
        const std::string& series_label = "") {
    if (thermal_map.empty()) {
        std::cout << "No thermal data available; skipping histogram for "
                  << result.location.label << std::endl;
        return;
    }

    if (bin_size_k <= 0.0 || max_k <= min_k) {
        std::cerr << "Invalid histogram parameters; min=" << min_k
                  << ", max=" << max_k << ", bin=" << bin_size_k << std::endl;
        return;
    }

    const int num_bins = static_cast<int>(std::ceil((max_k - min_k) / bin_size_k));
    std::vector<size_t> bins(static_cast<size_t>(num_bins), 0);
    size_t samples = 0;

    for (size_t face_idx = 0; face_idx < result.visibility.size(); ++face_idx) {
        if (result.visibility[face_idx] <= 0) continue;
        auto it = thermal_map.find(static_cast<int>(face_idx));
        if (it == thermal_map.end() || !it->second.has_data) continue;

        const double t = it->second.temperature_mean;
        if (t < min_k || t >= max_k) continue;

        const int bin = static_cast<int>((t - min_k) / bin_size_k);
        if (bin >= 0 && bin < num_bins) {
            bins[static_cast<size_t>(bin)]++;
            samples++;
        }
    }

    // Sanitize label for file name
    std::string label = result.location.label;
    for (char& c : label) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }

    std::string filename = output_dir + "/temperature_histogram_" + label + ".txt";
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to write histogram file: " << filename << std::endl;
        return;
    }

    out << "# Temperature histogram for waypoint: " << result.location.label << "\n";
    {
        std::string pretty_series = series_label;
        if (series_label == "avg" || series_label == "mean" || series_label == "statistics_avg") pretty_series = "Tmean";
        else if (series_label == "max" || series_label == "statistics_max") pretty_series = "Tmax";
        else if (series_label == "min" || series_label == "statistics_min") pretty_series = "Tmin";
        if (!pretty_series.empty()) {
            out << "# Series: " << pretty_series << "\n";
        }
    }
    out << "# Latitude: " << result.location.latitude << " deg\n";
    out << "# Longitude: " << result.location.longitude << " deg\n";
    out << "# Height: " << result.location.height << " m\n";
    out << "# Parameters: min_k=" << min_k << ", max_k=" << max_k
        << ", bin_size_k=" << bin_size_k << "\n";
    out << "# Samples (visible faces with thermal data): " << samples << "\n";
    out << "# Columns: bin_start_K, count\n";

    for (int i = 0; i < num_bins; ++i) {
        const double bin_start = min_k + i * bin_size_k;
        out << bin_start << ", " << bins[static_cast<size_t>(i)] << "\n";
    }

    out.close();
    std::cout << "Temperature histogram written: " << filename << std::endl;
}

// Main traverse analysis function
void analyzeTraverse(ViewshedAnalyzer& analyzer, 
                    const std::string& waypoints_file,
                    const std::string& thermal_file = "",
                    const std::string& output_dir = "viewshed_output") {
    std::cout << "\n################################################################" << std::endl;
    std::cout << "# TRAVERSE ANALYSIS FOR DARK HORSE MISSION" << std::endl;
    std::cout << "################################################################" << std::endl;
    
    // Load traverse waypoints from CSV
    auto waypoints = loadTraverseWaypoints(waypoints_file);
    if (waypoints.empty()) {
        std::cerr << "Error: No waypoints loaded" << std::endl;
        return;
    }
    
    // Load thermal model results if available
    auto mesh = analyzer.getMesh();
    auto thermal_series = loadThermalSeries(thermal_file, mesh);
    
    
    // Set up viewshed parameters
    ViewshedParams params;
    params.max_range_km = 100.0;  // 100 km range
    params.compute_horizon = true;
    params.num_threads = 0;  // Auto-detect
    
    // Compute viewsheds for all waypoints
    std::cout << "\nComputing viewsheds for " << waypoints.size() 
              << " traverse waypoints..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = analyzer.computeMultipleViewsheds(waypoints, params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double compute_duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Viewshed computation completed in " << std::fixed 
              << std::setprecision(2) << compute_duration << " seconds" << std::endl;
    
    // Analyze results with thermal context
    std::cout << "\n=== TRAVERSE ANALYSIS RESULTS ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Waypoint" 
              << std::setw(15) << "Visible Area" 
              << std::setw(15) << "Max Range"
              << std::setw(20) << "Histogram Samples" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::set<int> all_visible_faces;
    double total_distance = 0.0;
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // For each available series/timestep, compute and write histogram
        const double HIST_MIN_K = 0.0;
        const double HIST_MAX_K = 400.0;
        const double HIST_BIN_K = 1.0;
        size_t histogram_samples_total = 0;
        if (!thermal_series.temperatures.empty()) {
            for (size_t t = 0; t < thermal_series.temperatures.size(); ++t) {
                // Build a transient map view for reuse of histogram function
                std::map<int, ThermalData> thermal_map;
                const auto& temps = thermal_series.temperatures[t];
                const size_t faces = std::min<size_t>(temps.size(), result.visibility.size());
                for (size_t f = 0; f < faces; ++f) {
                    ThermalData td; td.temperature_mean = temps[f]; td.temperature_min = temps[f]; td.temperature_max = temps[f]; td.has_data = true;
                    thermal_map[static_cast<int>(f)] = td;
                }

                // Append timestep/label suffix to output directory filename by temporarily adjusting label
                ViewshedResult labeled_result = result;
                std::string suffix = (t < thermal_series.labels.size()) ? ("_" + thermal_series.labels[t]) : ("_t" + std::to_string(t));
                labeled_result.location.label = result.location.label + suffix;
                computeAndWriteTemperatureHistogram(labeled_result, thermal_map, output_dir,
                                                    HIST_MIN_K, HIST_MAX_K, HIST_BIN_K,
                                                    (t < thermal_series.labels.size() ? thermal_series.labels[t] : std::string("")));

                // Count samples for summary row
                for (size_t face_idx = 0; face_idx < faces; ++face_idx) {
                    if (result.visibility[face_idx] > 0) {
                        const double tt = temps[face_idx];
                        if (tt >= HIST_MIN_K && tt < HIST_MAX_K) histogram_samples_total++;
                    }
                }
            }
        }
        
        std::cout << std::left << std::setw(25) << result.location.label
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << std::to_string(result.visible_area_km2) + " km²"
                  << std::setw(15) << std::to_string(result.max_visible_range_km) + " km";
        
        // Print total samples across all series/timesteps for this waypoint
        std::cout << std::setw(20) << std::to_string(histogram_samples_total);
        std::cout << std::endl;
        
        // Track cumulative coverage
        for (size_t j = 0; j < result.visibility.size(); ++j) {
            if (result.visibility[j] > 0) {
                all_visible_faces.insert(j);
            }
        }
        
        // Calculate distance to next waypoint
        if (i < results.size() - 1) {
            const auto& loc1 = result.location;
            const auto& loc2 = results[i + 1].location;
            
            // Approximate distance calculation (simplified)
            double dlat = (loc2.latitude - loc1.latitude) * M_PI / 180.0;
            double dlon = (loc2.longitude - loc1.longitude) * M_PI / 180.0;
            double a = sin(dlat/2) * sin(dlat/2) + 
                      cos(loc1.latitude * M_PI / 180.0) * cos(loc2.latitude * M_PI / 180.0) * 
                      sin(dlon/2) * sin(dlon/2);
            double c = 2 * atan2(sqrt(a), sqrt(1-a));
            double distance = 1737.4 * c;  // Moon radius in km
            total_distance += distance;
        }
    }
    
    // Print cumulative statistics
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "\nCUMULATIVE COVERAGE:" << std::endl;
    std::cout << "  Total traverse distance: " << std::fixed << std::setprecision(2) 
              << total_distance << " km" << std::endl;
    std::cout << "  Total unique faces visible: " << all_visible_faces.size() 
              << " / " << mesh->faces.size() << std::endl;
    
    // Note: Cold regions analysis removed in favor of per-waypoint histograms.
    
    // Export results - create output directory
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Warning: Could not create output directory '" << output_dir 
                  << "': " << ec.message() << std::endl;
    } else {
        std::cout << "\nOutput directory: " << output_dir << std::endl;
    }
    
    // Export traverse viewsheds as time series
    std::string pvd_filename = output_dir + "/traverse_viewsheds.pvd";
    if (analyzer.exportToPVD(pvd_filename, results, true)) {
        std::cout << "\nTraverse viewsheds exported to: " << pvd_filename << std::endl;
    }
    
    // Generate combined coverage map
    ViewshedResult combined;
    combined.visibility.resize(results[0].visibility.size(), 0.0f);
    combined.location.label = "Combined_Traverse_Coverage";
    
    for (const auto& r : results) {
        for (size_t i = 0; i < r.visibility.size(); ++i) {
            if (r.visibility[i] > 0) {
                combined.visibility[i] = 1.0f;
            }
        }
    }
    
    std::string combined_file = output_dir + "/traverse_combined_coverage.vtk";
    analyzer.exportToVTK(combined_file, combined);
    std::cout << "Combined coverage map exported to: " << combined_file << std::endl;
    
    // Export horizon profiles
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].horizon_profile.empty()) {
            std::string horizon_file = output_dir + "/horizon_" + 
                                      std::to_string(i+1) + "_" + 
                                      results[i].location.label + ".csv";
            std::ofstream hfile(horizon_file);
            if (hfile.is_open()) {
                hfile << "azimuth_deg,elevation_deg\n";
                for (const auto& [az, el] : results[i].horizon_profile) {
                    hfile << az << "," << el << "\n";
                }
                hfile.close();
            }
        }
    }
    
#ifdef HAS_HDF5
    // Export to HDF5 for detailed analysis
    std::string hdf5_file = output_dir + "/traverse_analysis.h5";
    if (analyzer.exportToHDF5(hdf5_file, results)) {
        std::cout << "Detailed results exported to HDF5: " << hdf5_file << std::endl;
    }
#endif
}

// Main function
int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "    PRISM DARK HORSE - VIEWSHED ANALYSIS TOOL" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Purpose: Analyze viewsheds along rover traverse routes" << std::endl;
    std::cout << "Version: 2.0 (Simplified waypoint-based analysis)" << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    // Check command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dem_file> <waypoints_csv> [options]" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  dem_file       - DEM mesh (PLY, OBJ, VTK, or GeoTIFF)" << std::endl;
        std::cerr << "  waypoints_csv  - CSV file with traverse waypoints" << std::endl;
        std::cerr << "\nOptions:" << std::endl;
        std::cerr << "  --thermal <file>     - Thermal model results (VTU/PVD/HDF5)" << std::endl;
        std::cerr << "  --output-dir <dir>   - Output directory (default: viewshed_output)" << std::endl;
        std::cerr << "\nCSV Format:" << std::endl;
        std::cerr << "  label,latitude,longitude[,height]" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " lunar_south.ply traverse.csv" << std::endl;
        std::cerr << "  " << argv[0] << " lunar_south.ply traverse.csv --thermal thermal_results.vtu" << std::endl;
        std::cerr << "  " << argv[0] << " lunar_south.ply traverse.csv --output-dir results/run1" << std::endl;
        return 1;
    }
    
    std::string dem_file = argv[1];
    std::string waypoints_file = argv[2];
    std::string thermal_file = "";
    std::string output_dir = "viewshed_output";  // Default output directory
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--thermal" && i + 1 < argc) {
            thermal_file = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " <dem_file> <waypoints_csv> [options]" << std::endl;
            std::cerr << "\nArguments:" << std::endl;
            std::cerr << "  dem_file       - DEM mesh (PLY, OBJ, VTK, or GeoTIFF)" << std::endl;
            std::cerr << "  waypoints_csv  - CSV file with traverse waypoints" << std::endl;
            std::cerr << "\nOptions:" << std::endl;
            std::cerr << "  --thermal <file>     - Thermal model results (VTU/PVD/HDF5)" << std::endl;
            std::cerr << "  --output-dir <dir>   - Output directory (default: viewshed_output)" << std::endl;
            return 0;
        } else {
            std::cerr << "Warning: Unknown option '" << arg << "'" << std::endl;
        }
    }
    
    // Validate input files
    if (!fs::exists(dem_file)) {
        std::cerr << "Error: DEM file not found: " << dem_file << std::endl;
        return 1;
    }
    
    if (!fs::exists(waypoints_file)) {
        std::cerr << "Error: Waypoints file not found: " << waypoints_file << std::endl;
        return 1;
    }
    
    // Initialize viewshed analyzer
    std::cout << "Loading DEM/mesh: " << dem_file << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ViewshedAnalyzer analyzer;
    if (!analyzer.initialize(dem_file)) {
        std::cerr << "Failed to initialize viewshed analyzer" << std::endl;
        return 1;
    }
    
    auto load_time = std::chrono::high_resolution_clock::now();
    double load_duration = std::chrono::duration<double>(load_time - start_time).count();
    
    // Print mesh statistics
    auto mesh = analyzer.getMesh();
    std::cout << "DEM loaded in " << std::fixed << std::setprecision(2) 
              << load_duration << " seconds" << std::endl;
    std::cout << "Mesh statistics:" << std::endl;
    std::cout << "  Vertices: " << mesh->vertices.size() << std::endl;
    std::cout << "  Faces: " << mesh->faces.size() << std::endl;
    
    // Perform traverse analysis
    analyzeTraverse(analyzer, waypoints_file, thermal_file, output_dir);
    
    // Final summary
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Analysis complete!" << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) 
              << total_duration << " seconds" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    return 0;
}