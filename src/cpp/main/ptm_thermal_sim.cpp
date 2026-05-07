#include "../mesh/mesh_utils.h"
#include "../illumination/illumination_model.h"
#include "../constants/constants.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream> 
#include <filesystem>
#include <numeric>
#include "config.h"
#include <cstdlib>
#include <string>
#include <SpiceUsr.h>  // Include SPICE functions
#include <functional>
#include <atomic>
#include <mutex>
#include <tuple>
#include <cctype>
#include <algorithm>

// DEBUGGING INCLUDES
#include <csignal>
#include <execinfo.h>
#include <cxxabi.h>
#include <unistd.h>   // for STDERR_FILENO

#include "../tools/config_parser.h"
#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"
#include "../thermal/thermal_property_map.h"
#include "../maps/spatial_map.h"
#include "../ice/ice_table.h"

// Radiation exchange module
#include "../radiation/radiation_config.h"
#include "../radiation/radiation_exchange.h"

// Orbital mechanics module (generic bodies)
#include "../orbital/generic_body.h"

// GPU thermal solver (optional)
#ifdef PTM_GPU_OPENCL
#include "../gpu/thermal_opencl.h"
#include "../gpu/thermal_gpu_pool.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ptm;
using namespace ptm_thermal::constants;
using namespace ptm_thermal::config;

using namespace thermal;

// ================= CRASH HANDLING UTILITIES =====================
namespace {
// Minimal backtrace printer
void print_backtrace() {
    void* buf[128];
    int n = backtrace(buf, 128);
    backtrace_symbols_fd(buf, n, STDERR_FILENO);
}

void on_signal(int sig) {
    std::cerr << "\n[FATAL] Caught signal " << sig << " — backtrace:\n";
    print_backtrace();
    std::_Exit(128 + sig);
}

void on_terminate() {
    if (auto ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            std::cerr << "\n[FATAL] Unhandled exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "\n[FATAL] Unhandled non-std exception\n";
        }
    } else {
        std::cerr << "\n[FATAL] std::terminate() without active exception\n";
    }
    std::cerr << "[FATAL] Backtrace:\n";
    print_backtrace();
    std::_Exit(1);
}

// RAII joiner for std::thread (C++17)
struct JoiningThread {
    std::thread t;
    template <class F>
    explicit JoiningThread(F&& f) : t(std::forward<F>(f)) {}
    JoiningThread(JoiningThread&&) = default;
    JoiningThread& operator=(JoiningThread&&) = default;
    ~JoiningThread() { if (t.joinable()) t.join(); }
    void join() { if (t.joinable()) t.join(); }
};

// OpenMP-safe barrier helper: only barrier if inside a parallel region
#ifdef _OPENMP
inline void barrier_if_parallel() {
    if (omp_in_parallel()) {
        #pragma omp barrier
    }
}
#else
inline void barrier_if_parallel() {}
#endif

struct CrashGuard {
    CrashGuard() {
        std::set_terminate(on_terminate);
        std::signal(SIGABRT, on_signal);
        std::signal(SIGSEGV, on_signal);
        std::signal(SIGBUS,  on_signal);
        std::signal(SIGILL,  on_signal);
    }
} crash_guard;
} // namespace
// ================================================================

// ================= PLY ASCII FALLBACK LOADER (DTM meshes) =====================
// Supports PLY files produced by dtm_to_ply_converter.cpp:
//  format ascii 1.0
//  element vertex N
//    properties (at least x y z; may include nx ny nz latitude longitude elevation)
//  element face M
//    property list uchar int vertex_indices
//  faces: "3 v0 v1 v2"
static bool loadAsciiDTMPLY(
    const std::string& filename,
    std::shared_ptr<Mesh>& mesh_out)
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "[PLY-Fallback] Cannot open file: " << filename << "\n";
        return false;
    }

    auto trim = [](const std::string& s) -> std::string {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
        return s.substr(a, b - a);
    };

    std::string line;
    bool header_done = false;
    size_t num_vertices = 0;
    size_t num_faces = 0;
    std::vector<std::string> vprops;
    bool format_ascii = false;
    std::string current_element;

    while (!header_done && std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream ls(line);
        std::string tok;
        ls >> tok;

        if (tok == "ply") continue;
        if (tok == "comment") continue;

        if (tok == "format") {
            std::string fmt; ls >> fmt;
            if (fmt.rfind("ascii", 0) == 0) format_ascii = true;
            else {
                std::cerr << "[PLY-Fallback] Non-ASCII format (" << fmt << ") unsupported by fallback\n";
                return false;
            }
        } else if (tok == "element") {
            std::string name; size_t count = 0;
            ls >> name >> count;
            current_element = name;
            if (name == "vertex") num_vertices = count;
            else if (name == "face") num_faces = count;
        } else if (tok == "property") {
            if (current_element == "vertex") {
                std::string type, name;
                ls >> type >> name;
                if (!name.empty()) vprops.push_back(name);
            } else if (current_element == "face") {
                // property list uchar int vertex_indices (ignore extras)
            }
        } else if (tok == "end_header") {
            header_done = true;
            break;
        }
    }

    if (!format_ascii || num_vertices == 0 || num_faces == 0) {
        std::cerr << "[PLY-Fallback] Invalid header (format_ascii=" << format_ascii
                  << ", vertices=" << num_vertices << ", faces=" << num_faces << ")\n";
        return false;
    }

    auto findProp = [&](const std::string& name)->int {
        for (size_t i=0;i<vprops.size();++i)
            if (vprops[i]==name) return static_cast<int>(i);
        return -1;
    };

    int ix = findProp("x"), iy = findProp("y"), iz = findProp("z");
    if (ix < 0 || iy < 0 || iz < 0) {
        std::cerr << "[PLY-Fallback] Missing required x/y/z properties\n";
        return false;
    }

    // Read vertices
    std::vector<std::array<double,3>> vertices;
    vertices.reserve(num_vertices);

    size_t v_read = 0;
    while (v_read < num_vertices && std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream ls(line);
        std::vector<double> vals(vprops.size());
        for (size_t k=0;k<vprops.size();++k) {
            if (!(ls >> vals[k])) {
                std::cerr << "[PLY-Fallback] Malformed vertex line " << v_read << "\n";
                return false;
            }
        }
        vertices.push_back({ vals[ix], vals[iy], vals[iz] });
        ++v_read;
    }

    if (v_read != num_vertices) {
        std::cerr << "[PLY-Fallback] Vertex count mismatch read=" << v_read
                  << " expected=" << num_vertices << "\n";
        return false;
    }

    // Read faces
    std::vector<std::array<int,3>> faces;
    faces.reserve(num_faces);
    size_t f_read = 0;
    while (f_read < num_faces && std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream ls(line);
        int n;
        if (!(ls >> n)) {
            std::cerr << "[PLY-Fallback] Malformed face line " << f_read << "\n";
            return false;
        }
        if (n != 3) {
            for (int k = 0; k < n; ++k) { int dummy; ls >> dummy; }
            continue;
        }
        int a,b,c;
        if (!(ls >> a >> b >> c)) {
            std::cerr << "[PLY-Fallback] Face indices parse error at face " << f_read << "\n";
            return false;
        }
        if (a<0||b<0||c<0||a>=static_cast<int>(num_vertices)||b>=static_cast<int>(num_vertices)||c>=static_cast<int>(num_vertices)) {
            std::cerr << "[PLY-Fallback] Face index out of range at face " << f_read << "\n";
            return false;
        }
        faces.push_back({a,b,c});
        ++f_read;
    }

    if (faces.size() < num_faces * 0.9) {
        std::cerr << "[PLY-Fallback] Parsed only " << faces.size() << "/" << num_faces
                  << " faces (<90%) aborting\n";
        return false;
    }

    // Build mesh
    mesh_out = std::make_shared<Mesh>();
    mesh_out->vertices = std::move(vertices);
    mesh_out->faces    = std::move(faces);

    // Compute face centers & normals and lat/lon
    const size_t F = mesh_out->faces.size();
    mesh_out->face_centers.resize(F);
    mesh_out->face_normals.resize(F);
    mesh_out->latitudes.resize(F);
    mesh_out->longitudes.resize(F);

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<F; ++i) {
        const auto& tri = mesh_out->faces[i];
        const auto& v1 = mesh_out->vertices[tri[0]];
        const auto& v2 = mesh_out->vertices[tri[1]];
        const auto& v3 = mesh_out->vertices[tri[2]];

        std::array<double,3> c = {
            (v1[0]+v2[0]+v3[0])/3.0,
            (v1[1]+v2[1]+v3[1])/3.0,
            (v1[2]+v2[2]+v3[2])/3.0
        };
        mesh_out->face_centers[i] = c;

        std::array<double,3> e1 = { v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2] };
        std::array<double,3> e2 = { v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2] };
        std::array<double,3> n = {
            e1[1]*e2[2]-e1[2]*e2[1],
            e1[2]*e2[0]-e1[0]*e2[2],
            e1[0]*e2[1]-e1[1]*e2[0]
        };
        double L = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (L>0) { n[0]/=L; n[1]/=L; n[2]/=L; }
        // outward
        double dot = c[0]*n[0]+c[1]*n[1]+c[2]*n[2];
        if (dot < 0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        mesh_out->face_normals[i] = n;

        double r = std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
        double lat = (r > 0) ? std::asin(c[2]/r) * RAD_TO_DEG : 0.0;
        double lon = std::atan2(c[1], c[0]) * RAD_TO_DEG;
        mesh_out->latitudes[i]  = lat;
        mesh_out->longitudes[i] = lon;
    }

    mesh_out->setProperty("is_custom_mesh", "true");
    mesh_out->setProperty("source_file", filename);
    std::cout << "[PLY-Fallback] Loaded ASCII PLY: vertices=" << mesh_out->vertices.size()
              << " faces=" << mesh_out->faces.size() << "\n";
    return true;
}

// Wrapper: try standard MeshUtils import; if it fails and file is .ply ASCII DTM, use fallback
static bool importMeshWithFallback(const std::string& filename,
                                   std::shared_ptr<Mesh>& mesh_out,
                                   double scale_factor = 1.0)
{
    MeshUtils mu;
    if (mu.importTriangulatedMesh(filename, scale_factor)) {
        mesh_out = mu.getMesh();
        return true;
    }

    // fallback only for .ply
    std::string ext;
    auto p = filename.find_last_of('.');
    if (p != std::string::npos) {
        ext = filename.substr(p + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }
    if (ext == "ply") {
        std::cerr << "[Mesh Import] Standard loader failed; attempting ASCII fallback...\n";
        if (loadAsciiDTMPLY(filename, mesh_out)) {
            if (scale_factor != 1.0 && mesh_out) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < mesh_out->vertices.size(); ++i) {
                    mesh_out->vertices[i][0] *= scale_factor;
                    mesh_out->vertices[i][1] *= scale_factor;
                    mesh_out->vertices[i][2] *= scale_factor;
                }
                // recompute centers only (normals unchanged length-1)
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < mesh_out->faces.size(); ++i) {
                    const auto& f = mesh_out->faces[i];
                    const auto& a = mesh_out->vertices[f[0]];
                    const auto& b = mesh_out->vertices[f[1]];
                    const auto& c = mesh_out->vertices[f[2]];
                    mesh_out->face_centers[i] = {
                        (a[0]+b[0]+c[0])/3.0,
                        (a[1]+b[1]+c[1])/3.0,
                        (a[2]+b[2]+c[2])/3.0
                    };
                }
            }
            return true;
        }
        std::cerr << "[Mesh Import] Fallback also failed.\n";
    }
    return false;
}
// ================= END PLY ASCII FALLBACK LOADER =====================

// Constants for albedo processing
namespace {
    // Minimum physically plausible albedo for icy surfaces
    // Ice has albedo typically 0.5+, but allowing down to 0.05 for "dirty" ice
    constexpr double MIN_VALID_ALBEDO = 0.01;
    
    // Maximum physically plausible albedo (pure white ice/snow)
    constexpr double MAX_VALID_ALBEDO = 0.99;
    
    // Default visible albedo if no valid value is found
    constexpr double DEFAULT_ICE_ALBEDO = 0.55;
    
    // Default IR albedo used for thermal calculations
    constexpr double DEFAULT_IR_ALBEDO = 0.1;
}

// Thread-safe output suppressor class
class OutputSuppressor {
private:
    std::streambuf* original_buffer;
    bool active;
    static std::ofstream null_stream;
    static int ref_count;
    static std::mutex mtx;

public:
    OutputSuppressor() : active(false), original_buffer(nullptr) {
        suppress();
    }
    
    void suppress() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!active) {
            if (ref_count == 0) {
                // Only open the null stream once
                null_stream.open("/dev/null");
            }
            ref_count++;
            original_buffer = std::cout.rdbuf();
            std::cout.rdbuf(null_stream.rdbuf());
            active = true;
        }
    }
    
    void resume() {
        std::lock_guard<std::mutex> lock(mtx);
        if (active) {
            std::cout.rdbuf(original_buffer);
            ref_count--;
            if (ref_count == 0) {
                null_stream.close();
            }
            active = false;
        }
    }
    
    ~OutputSuppressor() {
        resume();
    }
};

// Initialize static members
std::ofstream OutputSuppressor::null_stream;
int OutputSuppressor::ref_count = 0;
std::mutex OutputSuppressor::mtx;

// Thread-safe stderr progress bar class
class StderrProgressBar {
private:
    int total;
    int barWidth;
    std::string prefix;
    bool showProgress;
    std::mutex mtx; // Add mutex for thread safety
    int last_percentage; // Track last percentage to reduce updates

public:
    StderrProgressBar(int total, const std::string& prefix, int barWidth, bool showProgress)
        : total(total), barWidth(barWidth), prefix(prefix), showProgress(showProgress), last_percentage(-1) {}
    
    void update(int current) {
        if (!showProgress) return;
        
        // Calculate current percentage
        int percentage = static_cast<int>((static_cast<float>(current) / total) * 100.0);
        
        // Only update if percentage has changed to reduce mutex contention
        if (percentage == last_percentage && percentage != 100)
            return;
        
        // Lock mutex for thread-safe stderr access
        std::lock_guard<std::mutex> lock(mtx);
        
        // Only update if percentage has changed (double-check after lock)
        if (percentage != last_percentage || percentage == 100) {
            last_percentage = percentage;
            
            float progress = static_cast<float>(current) / total;
            int pos = static_cast<int>(barWidth * progress);
            
            std::cerr << "\r" << prefix << " [";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cerr << "=";
                else if (i == pos) std::cerr << ">";
                else std::cerr << " ";
            }
            std::cerr << "] " << percentage << "%";
            std::cerr.flush();
        }
    }
    
    void finish() {
        if (!showProgress) return;
        
        // Lock mutex for thread-safe stderr access
        std::lock_guard<std::mutex> lock(mtx);
        
        std::cerr << "\r" << prefix << " [";
        for (int i = 0; i < barWidth; ++i) {
            std::cerr << "=";
        }
        std::cerr << "] 100%" << std::endl;
    }
};

// ================= FUNCTION DECLARATIONS =====================

static std::vector<std::tuple<std::string,double,double>> readLocationsCSV(const std::string& filename);
static std::vector<double> readDepthsFile(const std::string& filename);
static int findNearestFaceIndexByLatLon(const std::shared_ptr<Mesh>& mesh, double lat_deg, double lon_deg);
static bool writeLocationDepthCSV(const std::string& out_path,
                           const std::string& location_name,
                           const std::vector<std::string>& time_strings,
                           const std::vector<double>& depths_m,
                           const std::vector<std::vector<double>>& T_depths);
static std::vector<std::vector<double>> runThermalModelDepthSeries(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude_deg,
    const thermal::Config& config,
    const std::vector<double>& depths_m,
    int equilibration_steps,
    const thermal::Planet& planet);
static bool generateDepthProfilePlots(const std::string& csv_dir, const std::string& executable_path);

bool exportLocationTemperatureOnly(const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<std::string>& time_strings,
    int time_steps);

bool exportLocationIllumination(const std::string&, const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<std::string>&, int);
    
bool exportBodyPositions(const std::string&, const std::vector<std::array<double, 3>>&,
    const std::vector<std::array<double, 3>>&, const std::vector<std::string>&);

bool exportDataAsBinary(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    bool is_custom_mesh);
    
struct Statistics;
bool writeStatisticsBinary(
    const std::string& output_dir,
    const std::string& data_type,
    const Statistics& stats,
    size_t num_faces);

bool exportIlluminationStatisticsStreamed(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    ptm::IlluminationModel& model,
    int start_step,
    int steps);

bool processLocations(const std::string&, const std::string&, IlluminationModel&);

bool runThermalModel(const std::string&, const std::vector<double>&, const std::vector<double>&,
    double, double, const std::string&, int, const Config&, bool);
    
std::vector<double> runThermalModelForFace(const std::vector<double>&, const std::vector<double>&,
    double, const thermal::Config&, int);
    
bool runThermalModelsForAllFaces(const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, int, const thermal::Config&, double, double,
    std::vector<std::vector<double>>*, int, int, int,
    const thermal::Planet* = nullptr,
    const std::vector<double>* face_Qb = nullptr);  // Per-face heat flow override

// SIMD batch thermal solver (SIMD-over-pixels)
bool runThermalModelsForAllFacesBatchSoA(
    const std::string& thermal_output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    int equilibration_steps,
    int total_time_steps,
    int thermal_threads,
    const std::vector<double>* face_Qb = nullptr,
    const std::vector<double>* face_H_overrides = nullptr);

// GPU persistent kernel thermal solver
// equilibration_steps: number of steps in the equilibration phase (0 = all steps
// are output). When > 0 and the eq phase has a different dt from the output phase,
// the solver runs in two passes: equilibration cycles with eq_dt, then one output
// cycle with output_dt.
bool runThermalModelsForAllFaces_GPU_Persistent(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int total_time_steps,
    int equilibration_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    const std::vector<double>* face_Qb = nullptr,
    const std::vector<double>* face_H_overrides = nullptr,
    // Optional Schorghofer & Taylor (2007) accumulator. When non-null,
    // ⟨E(T(z,t))⟩ per face × native depth node is filled over the
    // post-equilibration output window. Must be preallocated to
    // n_faces × n_layers via IceStabilityAccumulator::allocate(); the
    // function fills sum_E and z, and sets n_samples to output_steps.
    ice::IceStabilityAccumulator* ice_acc = nullptr,
    // Optional thermal-state checkpoint paths used by the chunked-run
    // orchestrator. When initial_state_file is non-empty, the per-face
    // T(z) profile is loaded from that file BEFORE each batch's GPU
    // kernel runs. When final_state_file is non-empty, the per-face
    // T(z) profile is dumped to that file AFTER all batches finish.
    // Format: uint32 magic 'PTMT' (0x50544D54), uint32 n_faces, uint32
    // n_layers, then n_layers*n_faces float32 (layer-major SoA, the
    // GPU buf_T_ native layout).
    const std::string& initial_state_file = "",
    const std::string& final_state_file   = "",
    // Multi-GPU controls (see config_parser.h thermal.gpu_*).
    //   gpu_devices == 1 (default) AND gpu_device_ids empty → legacy
    //     single-device path, byte-identical to pre-multi-GPU behaviour.
    //   gpu_devices == 0 / "auto" → use every enumerated GPU.
    //   gpu_device_ids non-empty → use exactly those device indices.
    //   gpu_batch_size = 0 keeps the legacy 6 GB / 500 K-face cap.
    int gpu_devices = 1,
    const std::vector<int>& gpu_device_ids = {},
    size_t gpu_batch_size = 0,
    int gpu_inter_batch_pause_ms = 0);

// GPU thermal solver for selected faces with pre-combined flux arrays
// Used in radiation-thermal coupling iterations where flux includes indirect radiation
bool runThermalForSelectedFaces_GPU(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<int>& face_indices,           // Faces to process
    const std::vector<std::vector<double>>& flux,   // Pre-combined flux per face [face_idx][timestep]
    const std::vector<double>& times,
    int total_time_steps,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps); // In/out: updated for selected faces

void saveCheckpoint(const std::string&, const std::vector<size_t>&,
    const std::vector<std::vector<double>>&);
    
bool addThermalDataToVTK(const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

using FluxFn = std::function<double(size_t,int)>;
bool runThermalModelsForAllFacesStreamed(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& times,
    int total_time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    int thermal_threads,
    FluxFn get_direct,
    FluxFn get_reflected,
    FluxFn get_ir,
    std::vector<std::vector<double>>* surface_temps_out,
    const std::vector<double>* face_Qb = nullptr,
    const std::vector<double>* face_H_overrides = nullptr);

std::vector<double> computeFaceAlbedos(
    const std::shared_ptr<Mesh>& mesh,
    std::shared_ptr<ptm::SpatialMap> albedo_map,
    double default_visible_albedo);

void fixOutOfRangeAlbedos(
    std::vector<double>& face_albedos,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo);

std::string calculateAdjustedStartTime(const std::string& iso_time, double days_offset);

/**
 * @brief Rotate mesh vertices so mesh center moves to (target_lat, target_lon).
 *
 * Uses Rodrigues' rotation formula. After rotating vertices, recomputes
 * face_centers, face_normals, latitudes, and longitudes from scratch.
 * View factors depend only on relative geometry (preserved by rotation),
 * so pre-computed VFs remain valid.
 */
static void rotateMeshToLatLon(std::shared_ptr<ptm::Mesh>& mesh,
                                double target_lat_deg,
                                double target_lon_deg)
{
    if (!mesh || mesh->vertices.empty()) return;

    constexpr double DEG_TO_RAD = M_PI / 180.0;
    constexpr double RAD_TO_DEG = 180.0 / M_PI;

    // 1. Compute current mesh center (mean of all vertices)
    double cx = 0, cy = 0, cz = 0;
    for (const auto& v : mesh->vertices) {
        cx += v[0]; cy += v[1]; cz += v[2];
    }
    double N = static_cast<double>(mesh->vertices.size());
    cx /= N; cy /= N; cz /= N;

    double r = std::sqrt(cx*cx + cy*cy + cz*cz);
    double cur_lat = std::asin(cz / r);            // radians
    double cur_lon = std::atan2(cy, cx);            // radians

    double tgt_lat = target_lat_deg * DEG_TO_RAD;
    double tgt_lon = (target_lon_deg > -900.0) ? target_lon_deg * DEG_TO_RAD : cur_lon;

    // 2. Source and target unit vectors on the sphere
    auto to_cart = [](double lat, double lon) -> std::array<double,3> {
        return {{ std::cos(lat)*std::cos(lon),
                  std::cos(lat)*std::sin(lon),
                  std::sin(lat) }};
    };
    auto src = to_cart(cur_lat, cur_lon);
    auto tgt = to_cart(tgt_lat, tgt_lon);

    // Rotation axis = cross(src, tgt), angle = acos(dot(src, tgt))
    double dot = src[0]*tgt[0] + src[1]*tgt[1] + src[2]*tgt[2];
    dot = std::max(-1.0, std::min(1.0, dot));
    double angle = std::acos(dot);

    if (angle < 1e-10) {
        std::cout << "[Mesh Rotation] No rotation needed (already at target)" << std::endl;
        return;
    }

    std::array<double,3> axis = {{
        src[1]*tgt[2] - src[2]*tgt[1],
        src[2]*tgt[0] - src[0]*tgt[2],
        src[0]*tgt[1] - src[1]*tgt[0]
    }};
    double axis_len = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (axis_len < 1e-12) {
        // src and tgt are anti-parallel — pick any perpendicular axis
        axis = {{ -src[1], src[0], 0.0 }};
        axis_len = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1]);
        if (axis_len < 1e-12) { axis = {{0, 0, 1}}; axis_len = 1.0; }
    }
    axis[0] /= axis_len; axis[1] /= axis_len; axis[2] /= axis_len;

    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);

    std::cout << "[Mesh Rotation] Rotating mesh center from ("
              << cur_lat * RAD_TO_DEG << " deg, " << cur_lon * RAD_TO_DEG
              << " deg) to (" << target_lat_deg << " deg, "
              << ((target_lon_deg > -900.0) ? target_lon_deg : cur_lon * RAD_TO_DEG)
              << " deg)" << std::endl;
    std::cout << "  Rotation angle: " << angle * RAD_TO_DEG << " deg" << std::endl;

    // 3. Apply Rodrigues' rotation to all vertices
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh->vertices.size(); ++i) {
        auto& v = mesh->vertices[i];
        double kxv0 = axis[1]*v[2] - axis[2]*v[1];
        double kxv1 = axis[2]*v[0] - axis[0]*v[2];
        double kxv2 = axis[0]*v[1] - axis[1]*v[0];
        double kdv  = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2];

        v[0] = v[0]*cos_a + kxv0*sin_a + axis[0]*kdv*(1.0 - cos_a);
        v[1] = v[1]*cos_a + kxv1*sin_a + axis[1]*kdv*(1.0 - cos_a);
        v[2] = v[2]*cos_a + kxv2*sin_a + axis[2]*kdv*(1.0 - cos_a);
    }

    // 4. Recompute face_centers, face_normals, latitudes, longitudes
    const size_t F = mesh->faces.size();
    mesh->face_centers.resize(F);
    mesh->face_normals.resize(F);
    mesh->latitudes.resize(F);
    mesh->longitudes.resize(F);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < F; ++i) {
        const auto& tri = mesh->faces[i];
        const auto& v1 = mesh->vertices[tri[0]];
        const auto& v2 = mesh->vertices[tri[1]];
        const auto& v3 = mesh->vertices[tri[2]];

        std::array<double,3> c = {{
            (v1[0]+v2[0]+v3[0]) / 3.0,
            (v1[1]+v2[1]+v3[1]) / 3.0,
            (v1[2]+v2[2]+v3[2]) / 3.0
        }};
        mesh->face_centers[i] = c;

        std::array<double,3> e1 = {{ v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2] }};
        std::array<double,3> e2 = {{ v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2] }};
        std::array<double,3> n = {{
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0]
        }};
        double L = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (L > 0) { n[0]/=L; n[1]/=L; n[2]/=L; }
        // Ensure outward-pointing normals (dot with center vector > 0)
        double dot_cn = c[0]*n[0] + c[1]*n[1] + c[2]*n[2];
        if (dot_cn < 0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        mesh->face_normals[i] = n;

        double rr = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        mesh->latitudes[i]  = (rr > 0) ? std::asin(c[2]/rr) * RAD_TO_DEG : 0.0;
        mesh->longitudes[i] = std::atan2(c[1], c[0]) * RAD_TO_DEG;
    }

    // 5. Verify
    cx = cy = cz = 0;
    for (const auto& v : mesh->vertices) { cx += v[0]; cy += v[1]; cz += v[2]; }
    cx /= N; cy /= N; cz /= N;
    r = std::sqrt(cx*cx + cy*cy + cz*cz);
    double new_lat = std::asin(cz/r) * RAD_TO_DEG;
    double new_lon = std::atan2(cy, cx) * RAD_TO_DEG;
    std::cout << "  New mesh center: (" << new_lat << " deg, " << new_lon << " deg)" << std::endl;
}

// ================= FUNCTION DEFINITIONS =====================

class StdoutRedirector {
private:
    std::streambuf* original_buffer;
    std::ostringstream null_stream;
    bool active;

public:
    StdoutRedirector() : active(false), original_buffer(nullptr) {}
    
    void redirect() {
        if (!active) {
            original_buffer = std::cout.rdbuf();
            std::cout.rdbuf(null_stream.rdbuf());
            active = true;
        }
    }
    
    void restore() {
        if (active && original_buffer) {
            std::cout.rdbuf(original_buffer);
            active = false;
        }
    }
    
    ~StdoutRedirector() {
        restore(); // Always restore on destruction
    }
}; 

double getSynodicDayFromPlanet(const thermal::Planet& planet) {
    // The planet object stores the rotational period in seconds in the 'day' field
    double rotation_period_seconds = planet.day;
    double rotation_period_days = rotation_period_seconds / (24.0 * 3600.0);
    
    return rotation_period_days;
}

std::vector<double> importAlbedoMapFromFile(const std::string& filename) {
    try {
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            std::cerr << "File does not exist: " << filename << std::endl;
            return {};
        }
        
        // Get file size for validation
        std::uintmax_t fileSize = std::filesystem::file_size(filename);
        if (fileSize < sizeof(size_t)) {
            std::cerr << "File too small to be valid: " << filename << std::endl;
            return {};
        }
        
        // Open file with explicit error checking
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return {};
        }
        
        // Read the number of elements
        size_t num_faces = 0;
        file.read(reinterpret_cast<char*>(&num_faces), sizeof(num_faces));
        
        // Validate num_faces against file size
        size_t expected_size = sizeof(size_t) + num_faces * sizeof(double);
        if (expected_size != fileSize) {
            std::cerr << "File size mismatch. Expected: " << expected_size 
                      << ", Actual: " << fileSize << std::endl;
            return {};
        }
        
        // Allocate memory for albedo values
        std::vector<double> albedos(num_faces);
        
        // Read all albedo values
        file.read(reinterpret_cast<char*>(albedos.data()), num_faces * sizeof(double));
        
        // Check if read succeeded
        if (!file.good() && !file.eof()) {
            std::cerr << "Error reading data from file: " << filename << std::endl;
            return {};
        }
        
        std::cout << "Successfully read " << num_faces << " albedo values from " << filename << std::endl;
        return albedos;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during file import: " << e.what() << std::endl;
        return {};
    }
}

std::vector<double> calculateAbsorbedFlux(
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    int time_steps,
    const std::shared_ptr<ptm::SpatialMap>& albedo_map,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo,
    double default_ir_albedo,
    const thermal::Config& config,
    const thermal::Planet* planet = nullptr,  // Optional planet for incidence-dependent albedo
    double solar_constant = 0.0) {            // Solar constant for computing incidence angle
    
    // Calculate IR emissivity from IR albedo
    double ir_emissivity = 1.0 - default_ir_albedo;
    
    // Derive time_steps if the provided value doesnâ€™t match array sizes
    const size_t n_values = direct_solar_values.size();
    const size_t face_count = mesh ? mesh->faces.size() : 0;
    if (face_count > 0 && n_values % face_count == 0) {
        int inferred_ts = static_cast<int>(n_values / face_count);
        if (inferred_ts != time_steps) {
            std::cout << "Warning: time_steps=" << time_steps
                      << " mismatches illumination arrays (inferred=" << inferred_ts
                      << "). Using inferred value." << std::endl;
            time_steps = inferred_ts;
        }
    }
    
    // Validate input arrays have expected size
    if (n_values == 0 || primary_reflected_values.size() != n_values || primary_ir_values.size() != n_values) {
        std::cerr << "Error: Input arrays must have matching sizes in calculateAbsorbedFlux" << std::endl;
        std::cerr << "direct_solar: " << direct_solar_values.size() 
                  << ", primary_reflected: " << primary_reflected_values.size()
                  << ", primary_ir: " << primary_ir_values.size() << std::endl;
        return std::vector<double>(); // Return empty vector to signal error
    }
    
    size_t num_faces = n_values / time_steps;
    
    // Validate face count makes sense
    if (num_faces * time_steps != n_values) {
        std::cerr << "Error: Input array size not divisible by time_steps in calculateAbsorbedFlux" << std::endl;
        return std::vector<double>();
    }
    
    // Validate mesh has expected face count
    if (mesh->faces.size() < num_faces) {
        std::cerr << "Error: Mesh has fewer faces (" << mesh->faces.size() 
                  << ") than expected (" << num_faces << ")" << std::endl;
        return std::vector<double>();
    }
    
    // Safety check: ensure latitude/longitude arrays are correctly sized
    if (mesh->latitudes.size() < num_faces || mesh->longitudes.size() < num_faces) {
        std::cerr << "Error: Mesh latitude/longitude arrays too small. "
                  << "Expected: " << num_faces 
                  << ", Got: " << mesh->latitudes.size() << "/" << mesh->longitudes.size() << std::endl;
        return std::vector<double>();
    }
    
    std::vector<double> absorbed_flux(n_values);
    
    // Pre-calculate albedo for each face
    std::vector<double> face_albedos(num_faces);

    // Check if we're using a custom mesh
    bool is_custom_mesh = false;
    try {
        auto it = config.custom_properties.find("is_custom_mesh");
        if (it != config.custom_properties.end()) {
            is_custom_mesh = (it->second == "1" || it->second == "true");
        }
    } catch (const std::exception& e) {
        std::cout << "WARNING: Error checking custom mesh status: " << e.what() << std::endl;
    }

    std::cout << "Using " << (is_custom_mesh ? "custom" : "standard") << " mesh mode" << std::endl;
    std::cout << "Default albedo: " << default_visible_albedo << std::endl;

    // Initialize all faces with default albedo
    std::fill(face_albedos.begin(), face_albedos.end(), default_visible_albedo);

    // Create a local non-const pointer for the albedo map to use
    std::shared_ptr<ptm::SpatialMap> albedo_map_to_use = albedo_map;  // Copy the const pointer

    // Check if user explicitly provided an albedo map file
    bool explicit_albedo_map = (albedo_map_to_use && albedo_map_to_use->isValid());
    bool albedo_successfully_loaded = false;

    // Check if albedo map loading is disabled via use_albedo_map config option
    bool albedo_map_disabled = (config.getProperty("use_albedo_map") == "0");

    if (explicit_albedo_map) {
        std::cout << "Using provided albedo map" << std::endl;
        albedo_successfully_loaded = true;
    }
    else if (albedo_map_disabled) {
        std::cout << "Albedo map loading disabled (use_albedo_map: false), "
                  << "using uniform albedo " << default_visible_albedo << " for all faces" << std::endl;
    }
    // For custom mesh without explicit albedo map, try to load global albedo map
    // Skip if: (a) user explicitly set a uniform albedo (use_default_albedo: false), or
    //          (b) albedo_maps_directory is explicitly set to empty
    else if (is_custom_mesh) {
        bool albedo_explicit = (config.getProperty("thermophysical_albedo_explicit") == "1");

        // Get the albedo maps directory
        std::string albedo_maps_dir = "data/albedo_maps";
        bool albedo_dir_explicitly_set = false;
        if (config.custom_properties.count("albedo_maps_directory") > 0) {
            albedo_maps_dir = config.getProperty("albedo_maps_directory");
            albedo_dir_explicitly_set = true;
        } else if (const char* env_dir = std::getenv("PTM_ALBEDO_MAPS_DIR")) {
            albedo_maps_dir = env_dir;
            albedo_dir_explicitly_set = true;
        } else if (const char* env_dir = std::getenv("EUROPA_ALBEDO_MAPS_DIR")) {
            // Backward compatibility
            albedo_maps_dir = env_dir;
            albedo_dir_explicitly_set = true;
        }

        // Skip FITS search if user explicitly set albedo or albedo_maps_directory is empty
        bool skip_fits = albedo_explicit || (albedo_dir_explicitly_set && albedo_maps_dir.empty());

        if (skip_fits) {
            if (albedo_explicit) {
                std::cout << "Using explicit uniform albedo " << default_visible_albedo
                          << " (thermophysical override, FITS auto-search disabled)" << std::endl;
            } else {
                std::cout << "Albedo maps directory is empty, skipping FITS auto-search. "
                          << "Using uniform albedo " << default_visible_albedo << std::endl;
            }
        } else {
            // Ensure directory has trailing slash
            if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/' && albedo_maps_dir.back() != '\\') {
                albedo_maps_dir += '/';
            }

            // Get the target body name from config
            std::string body_name = config.getProperty("target_body_name");
            if (body_name.empty()) {
                std::cerr << "ERROR: Target body name is not specified in config." << std::endl;
                return std::vector<double>();
            }

            // Convert to proper case
            if (!body_name.empty()) {
                std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
                body_name[0] = std::toupper(body_name[0]);
            }

            // Try loading the global albedo map for THIS BODY (not hardcoded Europa!)
            std::string albedo_fits_name = body_name + "-0.5deg-BondAlbedo.fits";
            std::vector<std::string> search_paths;
            search_paths.push_back(albedo_maps_dir + albedo_fits_name);  // User-specified or default path
            // Only add hardcoded fallback paths if directory was NOT explicitly configured
            if (!albedo_dir_explicitly_set) {
                search_paths.push_back("../data/albedo_maps/" + albedo_fits_name);  // From build/ directory
                search_paths.push_back("data/albedo_maps/" + albedo_fits_name);     // From project root
            }

            auto global_albedo_map = std::make_shared<ptm::SpatialMap>();
            bool found = false;
            for (const auto& path : search_paths) {
                std::cout << "Loading global albedo map from: " << path << std::endl;
                if (global_albedo_map->loadFromFITS(path)) {
                    std::cout << "Successfully loaded global albedo map for " << body_name << std::endl;
                    std::cout << "Map dimensions: " << global_albedo_map->getWidth()
                            << "x" << global_albedo_map->getHeight() << std::endl;
                    std::cout << "Albedo range in map: " << global_albedo_map->getMin()
                            << " to " << global_albedo_map->getMax() << std::endl;
                    std::cout << "NOTE: Global albedo map overrides default uniform albedo ("
                            << default_visible_albedo << "). To disable, set "
                            << "albedo_maps_directory to empty in config YAML." << std::endl;
                    albedo_map_to_use = global_albedo_map;
                    albedo_successfully_loaded = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "No global albedo map found for " << body_name
                        << ", using default albedo " << default_visible_albedo << std::endl;
            }
        }
    }
    else {
        // For standard icosphere meshes, try to load pre-computed binary albedo files
        // (Skip for generic bodies — no pre-computed maps exist for arbitrary bodies)
        bool is_generic_body = (config.getProperty("using_generic_body") == "true");
        if (is_generic_body) {
            std::cout << "Generic body: skipping binary albedo map search, using default albedo "
                      << default_visible_albedo << std::endl;
        }

        if (!is_generic_body) {
        // Get the albedo maps directory
        std::string albedo_maps_dir = "data/albedo_maps";
        if (config.custom_properties.count("albedo_maps_directory") > 0) {
            albedo_maps_dir = config.getProperty("albedo_maps_directory");
        } else if (const char* env_dir = std::getenv("PTM_ALBEDO_MAPS_DIR")) {
            albedo_maps_dir = env_dir;
        } else if (const char* env_dir = std::getenv("EUROPA_ALBEDO_MAPS_DIR")) {
            // Backward compatibility
            albedo_maps_dir = env_dir;
        }

        // Ensure directory has trailing slash
        if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/' && albedo_maps_dir.back() != '\\') {
            albedo_maps_dir += '/';
        }

        // Get the target body name from config
        std::string body_name = config.getProperty("target_body_name");
        if (!body_name.empty()) {
            // Convert to proper case
            std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
            body_name[0] = std::toupper(body_name[0]);
            
            // Get mesh resolution from config
            int mesh_resolution = 5; // Default
            try {
                auto it = config.custom_properties.find("mesh_resolution");
                if (it != config.custom_properties.end()) {
                    mesh_resolution = std::stoi(it->second);
                }
            } catch (const std::exception& e) {
                std::cout << "Warning: Could not parse mesh resolution, using default 5" << std::endl;
            }
            
            // Construct binary albedo file name: e.g., "Moon_Albedo_res6.bin"
            std::string albedo_filename = body_name + "_Albedo_res" + std::to_string(mesh_resolution) + ".bin";

            // Try multiple paths for the albedo file (supports running from build/ directory)
            std::vector<std::string> search_paths = {
                albedo_maps_dir + albedo_filename,           // User-specified or default path
                "../data/albedo_maps/" + albedo_filename,    // Relative from build/ directory
                "data/albedo_maps/" + albedo_filename        // Relative from project root
            };

            std::vector<double> loaded_albedos;
            std::string found_path;

            for (const auto& path : search_paths) {
                std::cout << "Attempting to load binary albedo file: " << path << std::endl;
                loaded_albedos = importAlbedoMapFromFile(path);
                if (!loaded_albedos.empty()) {
                    found_path = path;
                    break;
                }
            }

            if (!loaded_albedos.empty() && loaded_albedos.size() == num_faces) {
                std::cout << "Successfully loaded " << loaded_albedos.size() << " albedo values from: " << found_path << std::endl;
                std::cout << "Albedo range: " << *std::min_element(loaded_albedos.begin(), loaded_albedos.end())
                        << " to " << *std::max_element(loaded_albedos.begin(), loaded_albedos.end()) << std::endl;

                // Copy the loaded albedo values to face_albedos
                std::copy(loaded_albedos.begin(), loaded_albedos.end(), face_albedos.begin());
                albedo_successfully_loaded = true;
            } else {
                if (loaded_albedos.empty()) {
                    std::cout << "Failed to load binary albedo file from any of the search paths" << std::endl;
                } else {
                    std::cout << "Binary albedo file size mismatch: expected " << num_faces
                            << " faces, got " << loaded_albedos.size() << " values" << std::endl;
                }
                std::cout << "Using default albedo " << default_visible_albedo << " for all faces" << std::endl;
            }
        }
        } // end if (!is_generic_body)
    }

    // Now interpolate the albedo map to each face if we have a FITS map to interpolate
    // (Binary albedo files are already loaded directly into face_albedos above)
    if (albedo_map_to_use && albedo_map_to_use->isValid()) {
        std::cout << "Interpolating FITS albedo map to " << num_faces << " mesh faces..." << std::endl;

        size_t valid_interpolations = 0;
        size_t invalid_interpolations = 0;

        // Use OpenMP for parallel processing
        #pragma omp parallel for reduction(+:valid_interpolations,invalid_interpolations)
        for (size_t f = 0; f < num_faces; f++) {
            // Ensure we have valid lat/lon data
            if (f < mesh->latitudes.size() && f < mesh->longitudes.size()) {
                double face_lat = mesh->latitudes[f];
                double face_lon = mesh->longitudes[f];

                // Normalize longitude to 0-360 range for map lookup
                while (face_lon < 0) face_lon += 360.0;
                while (face_lon >= 360.0) face_lon -= 360.0;

                try {
                    // Get albedo from the map at THIS FACE's location
                    double map_albedo = albedo_map_to_use->getValue(face_lon, face_lat);

                    // Check if we got a valid value
                    if (map_albedo != albedo_map_to_use->getMissingValue() &&
                        map_albedo >= MIN_VALID_ALBEDO &&
                        map_albedo <= MAX_VALID_ALBEDO) {
                        face_albedos[f] = map_albedo;
                        valid_interpolations++;
                    } else {
                        // Keep default value for this face
                        invalid_interpolations++;
                    }
                } catch (const std::exception& e) {
                    // Keep default value on error
                    invalid_interpolations++;
                }
            } else {
                invalid_interpolations++;
            }
        }

        // Report statistics
        double min_albedo = *std::min_element(face_albedos.begin(), face_albedos.end());
        double max_albedo = *std::max_element(face_albedos.begin(), face_albedos.end());
        double avg_albedo = std::accumulate(face_albedos.begin(), face_albedos.end(), 0.0) / face_albedos.size();

        std::cout << "FITS albedo interpolation complete:" << std::endl;
        std::cout << "  Valid interpolations: " << valid_interpolations << " faces" << std::endl;
        std::cout << "  Invalid/default values: " << invalid_interpolations << " faces" << std::endl;
        std::cout << "  Albedo range: " << min_albedo << " to " << max_albedo << std::endl;
        std::cout << "  Average albedo: " << avg_albedo << std::endl;
        albedo_successfully_loaded = true;
    } else if (albedo_successfully_loaded) {
        // Binary albedo file was already loaded - just report statistics
        double min_albedo = *std::min_element(face_albedos.begin(), face_albedos.end());
        double max_albedo = *std::max_element(face_albedos.begin(), face_albedos.end());
        double avg_albedo = std::accumulate(face_albedos.begin(), face_albedos.end(), 0.0) / face_albedos.size();
        std::cout << "Using binary albedo map:" << std::endl;
        std::cout << "  Albedo range: " << min_albedo << " to " << max_albedo << std::endl;
        std::cout << "  Average albedo: " << avg_albedo << std::endl;
    } else {
        std::cout << "Using constant albedo " << default_visible_albedo << " for all faces" << std::endl;
    }

    // Final validation
    size_t corrected_faces = 0;
    for (size_t f = 0; f < num_faces; f++) {
        if (std::isnan(face_albedos[f]) || std::isinf(face_albedos[f]) || 
            face_albedos[f] < MIN_VALID_ALBEDO || face_albedos[f] > MAX_VALID_ALBEDO) {
            face_albedos[f] = default_visible_albedo;
            corrected_faces++;
        }
    }

    if (corrected_faces > 0) {
        std::cout << "Corrected " << corrected_faces << " invalid albedo values" << std::endl;
    }

    // Final validation to ensure all albedo values are valid
    int invalid_count = 0;
    for (size_t f = 0; f < num_faces; f++) {
        if (std::isnan(face_albedos[f]) || 
            std::isinf(face_albedos[f]) || 
            face_albedos[f] < MIN_VALID_ALBEDO || 
            face_albedos[f] > MAX_VALID_ALBEDO) {
            
            face_albedos[f] = default_visible_albedo;
            invalid_count++;
        }
    }
    
    if (invalid_count > 0) {
        std::cout << "Warning: Fixed " << invalid_count << " invalid albedo values" << std::endl;
    }
    
    // Check if we should use incidence-dependent albedo.
    // Skip if a spatially-varying map was already loaded (binary or FITS) — the per-face
    // map values must not be overridden by the uniform planet->albedo + incidence formula.
    bool use_incidence_albedo = (planet != nullptr && solar_constant > 0.0 &&
                                 !planet->using_albedo_map &&
                                 !albedo_successfully_loaded &&
                                 planet->albedoCoef.size() >= 2);

    if (use_incidence_albedo) {
        std::cout << "Using incidence-angle-dependent albedo (Hayne et al. 2017 formula)" << std::endl;
        std::cout << "  Base albedo A0 = " << planet->albedo << std::endl;
        std::cout << "  Coefficients: a = " << planet->albedoCoef[0]
                  << ", b = " << planet->albedoCoef[1] << std::endl;
    }

    // Apply albedo to calculate absorbed flux for each face at each time step
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t f = 0; f < num_faces; f++) {
        double base_albedo = face_albedos[f];
        double face_lat = mesh->latitudes[f];
        double face_lon = mesh->longitudes[f];

        for (size_t t = 0; t < time_steps; t++) {
            size_t idx = f * time_steps + t;

            // Add explicit bounds checking
            if (idx >= n_values) {
                #pragma omp critical
                {
                    std::cerr << "Error: Index out of bounds in calculateAbsorbedFlux: "
                              << idx << " >= " << n_values << std::endl;
                }
                continue;
            }

            double direct_flux = direct_solar_values[idx];
            double visible_albedo = base_albedo;

            // Compute incidence-dependent albedo if enabled and face is illuminated
            if (use_incidence_albedo && direct_flux > 0.0) {
                // Compute incidence angle from direct flux
                // direct_flux = S * cos(theta), so cos(theta) = direct_flux / S
                double cos_theta = direct_flux / solar_constant;
                // Clamp to valid range [0, 1] to handle numerical issues
                cos_theta = std::min(1.0, std::max(0.0, cos_theta));
                double incidence_angle = std::acos(cos_theta);

                // Get incidence-dependent albedo
                visible_albedo = planet->getAlbedoWithIncidence(
                    face_lon * M_PI / 180.0,  // Convert to radians
                    face_lat * M_PI / 180.0,
                    incidence_angle);
            }

            // Apply appropriate albedo to each component
            double absorbed_direct = direct_flux * (1.0 - visible_albedo);
            double absorbed_reflected = primary_reflected_values[idx] * (1.0 - visible_albedo);
            double absorbed_ir = primary_ir_values[idx] * ir_emissivity;

            // Sum all components
            absorbed_flux[idx] = absorbed_direct + absorbed_reflected + absorbed_ir;
        }
    }

    return absorbed_flux;
}

std::vector<double> runThermalModelWithMatchedTimes(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps,
    double Qb_override = -1.0,
    double H_override  = -1.0) {

    // Validate inputs
    std::vector<double> temps_out;
    if (times.empty() || flux_values.empty()) {
        std::cerr << "runThermalModelWithMatchedTimes: empty input" << std::endl;
        return temps_out;
    }

    const int N = static_cast<int>(times.size());
    
    // CRITICAL FIX: Create matrix with correct dimensions (2 rows, N columns)
    // Row 0: times, Row 1: flux values
    Eigen::MatrixXd flux_arr(2, N);  // NOT (N, 2)!
    
    // Check size matching
    if (flux_values.size() != times.size()) {
        std::cerr << "WARNING: flux_values size (" << flux_values.size() 
                  << ") != times size (" << times.size() << ")" << std::endl;
        // Use the minimum size
        int min_size = std::min(static_cast<int>(flux_values.size()), N);
        flux_arr.resize(2, min_size);
        for (int i = 0; i < min_size; ++i) {
            flux_arr(0, i) = times[i];      // Row 0: times
            flux_arr(1, i) = flux_values[i]; // Row 1: flux
        }
    } else {
        // Sizes match, fill the matrix correctly
        for (int i = 0; i < N; ++i) {
            flux_arr(0, i) = times[i];      // Row 0: times
            flux_arr(1, i) = flux_values[i]; // Row 1: flux
        }
    }

    try {
        const std::string planet_name = config.getProperty("planet", "Moon");  // Falls back to Moon if not set
        thermal::Planet planet(planet_name);

        // Override basal heat flow if specified (per-face or global)
        if (Qb_override >= 0.0) {
            planet.Qb = Qb_override;
        }

        // Override H-parameter if a per-face value was supplied (Diviner /
        // Hayne+2017 map or user custom). This drives the rho(z), k(z) profile
        // in Profile::Profile() and changes the diurnal thermal inertia.
        if (H_override > 0.0) {
            planet.H = H_override;
        }

        // Create thermal model
        thermal::Model model(planet,
                            latitude * M_PI/180.0,  // Convert to radians
                            times,
                            flux_arr,
                            config,
                            false);

        // CRITICAL: Don't use internal equilibration - it messes up time alignment
        model.use_full_time_series = true;
        // Don't set equilibration_steps in the model - handle it externally
        model.setStartTime(times.front());

        // Run the model
        model.run();

        // Extract surface temperatures
        temps_out.resize(N, 0.0);
        
        // The model's T matrix should be N_steps x N_z (time steps x depth layers)
        if (model.T.rows() == N && model.T.cols() > 0) {
            for (int i = 0; i < N; ++i) {
                temps_out[i] = model.T(i, 0);  // Column 0 is surface
            }
        } else {
            std::cerr << "ERROR: Unexpected thermal model output size: " 
                      << model.T.rows() << "x" << model.T.cols() 
                      << " (expected " << N << "xN_z)" << std::endl;
            // Fill with default values
            std::fill(temps_out.begin(), temps_out.end(), 200.0);
        }
        
        return temps_out;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in thermal model: " << e.what() << std::endl;
        temps_out.resize(N, 200.0);  // Default temperature
        return temps_out;
    }
}

bool exportLocationIllumination(
    const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::string>& time_strings,
    int time_steps) {
    // Basic validations
    if (!mesh) {
        std::cerr << "exportLocationIllumination: mesh is null" << std::endl;
        return false;
    }
    const size_t F = mesh->faces.size();
    if (F == 0) {
        std::cerr << "exportLocationIllumination: mesh has no faces" << std::endl;
        return false;
    }
    if (time_steps <= 0) {
        std::cerr << "exportLocationIllumination: invalid time_steps" << std::endl;
        return false;
    }

    // Read locations (try CSV first: name,lat,lon)
    std::vector<std::string> location_names;
    std::vector<std::pair<double,double>> locations;
    std::cout << "Attempting to open locations file: " << locations_file << std::endl;

    auto loc_csv = readLocationsCSV(locations_file);
    if (!loc_csv.empty()) {
        for (const auto& tup : loc_csv) {
            const std::string& name = std::get<0>(tup);
            double lat = std::get<1>(tup);
            double lon = std::get<2>(tup);
            location_names.push_back(name);
            locations.emplace_back(lat, lon);
        }
    } else {
        // Fallback: space-delimited "lat lon [name…]"
        std::ifstream input(locations_file);
        if (!input.is_open()) {
            std::cerr << "Failed to open locations file: " << locations_file << std::endl;
            return false;
        }
        std::cout << "Successfully opened locations file" << std::endl;

        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            // Trim leading spaces
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            if (line[start] == '#') continue;

            std::istringstream iss(line);
            double lat = 0.0, lon = 0.0;
            if (!(iss >> lat >> lon)) {
                // Try CSV style: supports both name,lat,lon and lat,lon,name
                std::string token;
                std::vector<std::string> parts;
                std::istringstream iss2(line);
                while (std::getline(iss2, token, ',')) {
                    // trim token
                    size_t a = token.find_first_not_of(" \t\r\n");
                    size_t b = token.find_last_not_of(" \t\r\n");
                    parts.push_back((a == std::string::npos) ? "" : token.substr(a, b - a + 1));
                }
                if (parts.size() >= 3) {
                    // Try format: name,lat,lon
                    try {
                        const std::string name = parts[0];
                        lat = std::stod(parts[1]);
                        lon = std::stod(parts[2]);
                        locations.emplace_back(lat, lon);
                        location_names.push_back(name);
                        continue;
                    } catch (...) {
                        // Try format: lat,lon,name
                        try {
                            lat = std::stod(parts[0]);
                            lon = std::stod(parts[1]);
                            const std::string name = parts[2];
                            locations.emplace_back(lat, lon);
                            location_names.push_back(name);
                            continue;
                        } catch (...) {
                            std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                            continue;
                        }
                    }
                } else if (parts.size() == 2) {
                    // Try format: lat,lon (no name)
                    try {
                        lat = std::stod(parts[0]);
                        lon = std::stod(parts[1]);
                        std::ostringstream name_ss;
                        name_ss << "Location_" << std::fixed << std::setprecision(2) << lat << "_" << lon;
                        locations.emplace_back(lat, lon);
                        location_names.push_back(name_ss.str());
                        continue;
                    } catch (...) {
                        std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                        continue;
                    }
                }
                std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                continue;
            }
            // Rest of line is name (optional)
            std::string name;
            std::getline(iss >> std::ws, name);
            if (name.empty()) {
                std::ostringstream name_ss;
                name_ss << "Location_" << std::fixed << std::setprecision(6) << lat << "_" << lon;
                name = name_ss.str();
            }
            locations.emplace_back(lat, lon);
            location_names.push_back(name);
        }
    }

    if (locations.empty()) {
        std::cerr << "[ERROR] No valid locations found in file" << std::endl;
        return false;
    }

    // Map each location to nearest face (robust helper)
    // Location CSV uses East longitude (IAU convention), same as mesh (atan2)
    std::vector<size_t> facet_indices;
    facet_indices.reserve(locations.size());
    for (size_t i = 0; i < locations.size(); ++i) {
        const auto& ll = locations[i];
        int face_idx = findNearestFaceIndexByLatLon(mesh, ll.first, ll.second);
        if (face_idx < 0 || static_cast<size_t>(face_idx) >= F) {
            std::cerr << "Warning: Could not map location '" << location_names[i]
                      << "' (" << std::fixed << std::setprecision(9) << ll.first
                      << ", " << ll.second << ") to a mesh face; skipping." << std::endl;
            continue;
        }
        facet_indices.push_back(static_cast<size_t>(face_idx));
        std::cout << "Location " << location_names[i] << " (" << std::fixed << std::setprecision(9)
                  << ll.first << ", " << ll.second << ") mapped to facet " << face_idx << " at ("
                  << std::fixed << std::setprecision(12)
                  << mesh->latitudes[static_cast<size_t>(face_idx)] << ", "
                  << mesh->longitudes[static_cast<size_t>(face_idx)] << ")" << std::endl;
    }

    if (facet_indices.empty()) {
        std::cerr << "No locations could be mapped to faces" << std::endl;
        return false;
    }

    // Open output file
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    // Header
    output << "Time";
    for (const auto& name : location_names) {
        output << "," << name << "_Total";
        output << "," << name << "_DirectSolar";
        output << "," << name << "_PrimaryReflected";
        output << "," << name << "_PrimaryIR";
    }
    output << "\n";

    // Safe value fetcher
    auto get_val = [&](const std::vector<double>& v, size_t f, int t) -> double {
        if (v.empty()) return 0.0;
        const size_t stride = static_cast<size_t>(time_steps);
        const size_t idx = f * stride + static_cast<size_t>(t);
        return (idx < v.size()) ? v[idx] : 0.0;
    };

    // Rows
    for (int t = 0; t < time_steps; ++t) {
        const std::string ts = (static_cast<size_t>(t) < time_strings.size())
                               ? time_strings[static_cast<size_t>(t)]
                               : std::to_string(t);
        output << ts;

        for (size_t i = 0; i < facet_indices.size(); ++i) {
            const size_t f = facet_indices[i];
            if (f >= F) {
                output << ",0,0,0,0";
                continue;
            }
            const double tot = get_val(illumination_values, f, t);
            const double dir = get_val(direct_solar_values, f, t);
            const double rfl = get_val(primary_reflected_values, f, t);
            const double pir = get_val(primary_ir_values, f, t);
            output << "," << tot << "," << dir << "," << rfl << "," << pir;
        }
        output << "\n";
    }

    std::cout << "Location illumination data written to " << output_file << std::endl;
    return true;
}

// Add this function to export body positions
bool exportBodyPositions(
    const std::string& output_file,
    const std::vector<std::array<double, 3>>& sun_positions,
    const std::vector<std::array<double, 3>>& primary_positions,
    const std::vector<std::string>& time_strings) {
    
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open body positions file: " << output_file << std::endl;
        return false;
    }
    
    // Write CSV header
    output << "time,sun_x,sun_y,sun_z,primary_x,primary_y,primary_z" << std::endl;

    // Write positions for each time step
    for (size_t i = 0; i < time_strings.size(); i++) {
        const auto& sun_pos = sun_positions[i];
        const auto& primary_pos = primary_positions[i];
        
        output << time_strings[i] << ","
               << sun_pos[0] << "," << sun_pos[1] << "," << sun_pos[2] << ","
               << primary_pos[0] << "," << primary_pos[1] << "," << primary_pos[2]
               << std::endl;
    }
    
    std::cout << "Body positions written to " << output_file << std::endl;
    return true;
}

std::vector<double> calculateAbsorbedFlux(
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    int time_steps,
    const std::shared_ptr<ptm::SpatialMap>& albedo_map,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo,
    double default_ir_albedo,
    const thermal::Config& config,
    const thermal::Planet* planet,
    double solar_constant);

bool exportLocationTemperatureOnly(
    const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,   // filtered temps (faces x steps)
    const std::vector<std::string>& time_strings,            // filtered times (size = steps)
    int time_steps)
{
    // Validate inputs
    if (!mesh || mesh->faces.empty() || surface_temps.empty() || time_strings.size() < static_cast<size_t>(time_steps)) {
        std::cerr << "exportLocationTemperatureOnly: invalid inputs\n";
        return false;
    }

    // Read locations
    std::ifstream input(locations_file);
    if (!input.is_open()) {
        std::cerr << "Failed to open locations file: " << locations_file << std::endl;
        return false;
    }

    std::vector<std::pair<double,double>> locations;
    std::vector<std::string> names;

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double lat, lon;
        std::string name;
        if (!(iss >> lat >> lon)) continue;
        std::getline(iss >> std::ws, name);
        if (name.empty()) {
            std::ostringstream os; os << std::fixed << std::setprecision(6) << lat << "_" << lon;
            name = os.str();
        }
        locations.emplace_back(lat, lon);
        names.push_back(name);
    }
    if (locations.empty()) {
        std::cerr << "No valid locations found in " << locations_file << "\n";
        return false;
    }

    // Map each location to nearest face center by lat/lon
    const size_t F = mesh->faces.size();
    std::vector<size_t> face_idx(locations.size(), 0);
    for (size_t i = 0; i < locations.size(); ++i) {
        double qlat = locations[i].first;
        double qlon = locations[i].second;
        // Normalize longitude diffs to [-180,180]
        auto dlon = [](double a, double b)->double {
            double d = a - b;
            while (d > 180.0) d -= 360.0;
            while (d < -180.0) d += 360.0;
            return d;
        };
        double best = std::numeric_limits<double>::max();
        size_t best_f = 0;
        for (size_t f = 0; f < F; ++f) {
            double dlat = mesh->latitudes[f] - qlat;
            double dl = dlon(mesh->longitudes[f], qlon);
            double dist2 = dlat*dlat + dl*dl;
            if (dist2 < best) { best = dist2; best_f = f; }
        }
        face_idx[i] = best_f;
        std::cout << "Location " << names[i] << " (" << std::fixed << std::setprecision(9)
                  << locations[i].first << ", " << locations[i].second << ") mapped to facet "
                  << best_f << " at (" << mesh->latitudes[best_f] << ", "
                  << mesh->longitudes[best_f] << ")\n";
    }

    // Open output CSV and write temperature only
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    out << "Time";
    for (const auto& n : names) out << "," << n << "_Temperature";
    out << "\n";

    // Write per-step temps
    for (int t = 0; t < time_steps; ++t) {
        out << time_strings[t];
        for (size_t i = 0; i < face_idx.size(); ++i) {
            const size_t f = face_idx[i];
            double val = 0.0;
            if (f < surface_temps.size() && static_cast<size_t>(t) < surface_temps[f].size()) {
                val = surface_temps[f][t];
            }
            out << "," << std::fixed << std::setprecision(6) << val;
        }
        out << "\n";
    }

    std::cout << "Location temperatures written to " << output_file << std::endl;
    return true;
}

// Example of using the improved function with a location file
bool processLocations(const std::string& location_file, 
                       const std::string& output_file,
                       IlluminationModel& model) {
    // Read locations from file
    std::vector<std::pair<double, double>> locations;
    std::vector<std::string> location_names;
    
    std::ifstream file(location_file);
    if (!file) {
        std::cerr << "Error: Could not open location file: " << location_file << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip comments and empty lines

        std::istringstream iss(line);
        double lat, lon;
        if (iss >> lat >> lon) {
            locations.push_back({lat, lon});

            // Get name if present
            std::string name;
            std::getline(iss >> std::ws, name);  // Read rest of line as name
            if (name.empty()) {
                name = "Location_" + std::to_string(locations.size());
            }
            location_names.push_back(name);
        } else {
            // Try CSV format: lat,lon,name or name,lat,lon
            std::string token;
            std::vector<std::string> parts;
            std::istringstream iss2(line);
            while (std::getline(iss2, token, ',')) {
                size_t a = token.find_first_not_of(" \t\r\n");
                size_t b = token.find_last_not_of(" \t\r\n");
                parts.push_back((a == std::string::npos) ? "" : token.substr(a, b - a + 1));
            }
            if (parts.size() >= 3) {
                try {
                    lat = std::stod(parts[0]);
                    lon = std::stod(parts[1]);
                    locations.push_back({lat, lon});
                    location_names.push_back(parts[2]);
                } catch (...) {
                    try {
                        lat = std::stod(parts[1]);
                        lon = std::stod(parts[2]);
                        locations.push_back({lat, lon});
                        location_names.push_back(parts[0]);
                    } catch (...) {
                        // Skip invalid lines
                    }
                }
            } else if (parts.size() == 2) {
                try {
                    lat = std::stod(parts[0]);
                    lon = std::stod(parts[1]);
                    locations.push_back({lat, lon});
                    location_names.push_back("Location_" + std::to_string(locations.size()));
                } catch (...) {
                    // Skip invalid lines
                }
            }
        }
    }

    if (locations.empty()) {
        std::cerr << "[ERROR] No valid locations found in file" << std::endl;
        return false;
    }

    // Calculate illumination for all locations
    auto results = model.calculateLocationIllumination(
        locations, "2020-01-01T00:00:00", 1.0);
    
    // Write results to CSV file
    std::ofstream out_file(output_file);
    if (!out_file) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return false;
    }
    
    // Write header
    out_file << "Location,Latitude,Longitude";
    for (size_t t = 0; t < model.getTimes().size(); t++) {
        out_file << ",Time_" << t;
    }
    out_file << std::endl;
    
    // Write data
    for (size_t i = 0; i < locations.size(); i++) {
        out_file << location_names[i] << "," 
                << locations[i].first << "," 
                << locations[i].second;
        
        for (double value : results[i]) {
            out_file << "," << value;
        }
        out_file << std::endl;
    }
    
    return true;
}

bool runThermalModel(
    const std::string& output_file,
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    double longitude,
    const std::string& location_name,
    int days,  // We'll keep this parameter for backwards compatibility
    const Config& config,
    bool verbose = false) {
    
    try {
        std::cout << "Running thermal model for location: " 
                  << location_name << " (Lat: " << latitude << ", Lon: " << longitude << ")" << std::endl;
        
        // Create the planet model (Europa by default)
        Planet europa("Europa");
        
        // Create a flux array in Eigen format
        Eigen::MatrixXd flux_arr(flux_values.size(), 2);
        for (size_t i = 0; i < flux_values.size(); i++) {
            flux_arr(i, 0) = times[i];           // Time (seconds)
            flux_arr(i, 1) = flux_values[i];      // Flux (W/mÂ²)
        }
        
        // Use vector of times instead of integer days parameter
        Model thermalModel(europa, latitude * M_PI/180.0, times, flux_arr, config, verbose);
        
        // Run the thermal model
        auto start_time = std::chrono::high_resolution_clock::now();
        thermalModel.run();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        std::cout << "Thermal model completed in " << duration_ms << " ms" << std::endl;
        
        // Write results to CSV file
        std::ofstream outfile(output_file);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return false;
        }
        
        // Write header
        outfile << "Time(hours),";
        for (int i = 0; i < thermalModel.profile.z().size(); i++) {
            outfile << "T_" << thermalModel.profile.z()[i] << "m";
            if (i < thermalModel.profile.z().size() - 1) outfile << ",";
        }
        outfile << std::endl;
        
        // Write data
        for (int i = 0; i < thermalModel.T.rows(); i++) {
            outfile << thermalModel.lt(i) / 3600.0 << ","; // Time in hours
            for (int j = 0; j < thermalModel.T.cols(); j++) {
                outfile << thermalModel.T(i, j);
                if (j < thermalModel.T.cols() - 1) outfile << ",";
            }
            outfile << std::endl;
        }
        outfile.close();
        
        // Print summary statistics
        double min_temp = thermalModel.T.col(0).minCoeff();
        double max_temp = thermalModel.T.col(0).maxCoeff();
        double avg_temp = thermalModel.T.col(0).mean();

        std::cout << "Surface Temperature Summary for " << location_name << ":" << std::endl;
        std::cout << "  Minimum: " << min_temp << " K" << std::endl;
        std::cout << "  Maximum: " << max_temp << " K" << std::endl;
        std::cout << "  Average: " << avg_temp << " K" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in thermal model: " << e.what() << std::endl;
        return false;
    }
}

// Fix the runThermalModelForFace function to ensure time alignment
std::vector<double> runThermalModelForFace(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps) {
    
    try {
        // Create planet model
        Planet europa("Europa");
        
        // Create flux array in the expected format
        Eigen::MatrixXd flux_arr(flux_values.size(), 2);
        for (size_t i = 0; i < flux_values.size(); i++) {
            flux_arr(i, 0) = times[i];
            flux_arr(i, 1) = flux_values[i];
        }
        
        // SIMPLIFIED APPROACH: Create thermal model with all time points
        Model thermalModel(europa, latitude * M_PI/180.0, times, flux_arr, config, false);
        
        // Do NOT use internal equilibration filtering - let the model run continuously
        // for all time points, just like the illumination model did
        thermalModel.run();
        
        // Return ALL temperatures for manual filtering later
        std::vector<double> surface_temps(thermalModel.T.rows());
        for (int i = 0; i < thermalModel.T.rows(); i++) {
            surface_temps[i] = thermalModel.T(i, 0);  // Column 0 = surface
        }
        
        return surface_temps;
    } catch (const std::exception& e) {
        return {};
    }
}

// Streamed thermal path: read illumination on demand; no giant buffers
bool runThermalModelsForAllFacesStreamed(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& times,
    int total_time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    int thermal_threads,
    FluxFn get_direct,
    FluxFn get_reflected,
    FluxFn get_ir,
    std::vector<std::vector<double>>* surface_temps_out,
    const std::vector<double>* face_Qb,
    const std::vector<double>* face_H_overrides)
{
    if (!mesh || mesh->faces.empty() || total_time_steps <= 0) {
        std::cerr << "Streamed thermal: invalid mesh or time steps\n";
        return false;
    }

    // Load per-face albedo: explicit FITS path > binary file > FITS auto-discovery > uniform default
    std::shared_ptr<ptm::SpatialMap> albedo_map;
    const size_t n_faces = mesh->faces.size();
    std::vector<double> face_albedos(n_faces, default_visible_albedo);
    bool albedo_loaded = false;

    // Check if albedo map loading is disabled
    bool albedo_map_disabled = (config.getProperty("use_albedo_map") == "0");

    // 1. Try explicit FITS path from config
    if (!albedo_map_disabled && config.custom_properties.count("albedo_map")) {
        const std::string path = config.getProperty("albedo_map");
        if (!path.empty()) {
            auto m = std::make_shared<ptm::SpatialMap>();
            if (m->loadFromFITS(path)) {
                albedo_map = m;
                std::cout << "[Streamed] Loaded explicit albedo map: " << path << std::endl;
            } else {
                std::cout << "[Streamed] Failed to load albedo map " << path
                          << " (" << m->getLastError() << ")\n";
            }
        }
    }

    // 2. Try pre-computed binary albedo file (for standard icosphere meshes)
    if (!albedo_map_disabled && !albedo_map) {
        std::string body_name = config.getProperty("target_body_name");
        if (!body_name.empty()) {
            std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
            body_name[0] = std::toupper(body_name[0]);

            int mesh_resolution = 5;
            auto it = config.custom_properties.find("mesh_resolution");
            if (it != config.custom_properties.end()) {
                try { mesh_resolution = std::stoi(it->second); } catch (...) {}
            }

            std::string albedo_maps_dir = "data/albedo_maps";
            if (config.custom_properties.count("albedo_maps_directory") > 0) {
                albedo_maps_dir = config.getProperty("albedo_maps_directory");
            }
            if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/') {
                albedo_maps_dir += '/';
            }

            // Try binary file first
            std::string bin_filename = body_name + "_Albedo_res" + std::to_string(mesh_resolution) + ".bin";
            std::vector<std::string> bin_paths = {
                albedo_maps_dir + bin_filename,
                "../data/albedo_maps/" + bin_filename,
                "data/albedo_maps/" + bin_filename
            };
            for (const auto& bp : bin_paths) {
                auto loaded = importAlbedoMapFromFile(bp);
                if (!loaded.empty() && loaded.size() == n_faces) {
                    std::copy(loaded.begin(), loaded.end(), face_albedos.begin());
                    albedo_loaded = true;
                    std::cout << "[Streamed] Loaded binary albedo from: " << bp
                              << "  range=[" << *std::min_element(loaded.begin(), loaded.end())
                              << ", " << *std::max_element(loaded.begin(), loaded.end()) << "]" << std::endl;
                    break;
                }
            }

            // If binary not found, try FITS auto-discovery
            if (!albedo_loaded) {
                std::string fits_name = body_name + "-0.5deg-BondAlbedo.fits";
                std::vector<std::string> fits_paths = {
                    albedo_maps_dir + fits_name,
                    "../data/albedo_maps/" + fits_name,
                    "data/albedo_maps/" + fits_name
                };
                for (const auto& fp : fits_paths) {
                    auto m = std::make_shared<ptm::SpatialMap>();
                    if (m->loadFromFITS(fp)) {
                        albedo_map = m;
                        std::cout << "[Streamed] Loaded FITS albedo map: " << fp << std::endl;
                        break;
                    }
                }
            }
        }
    }

    // Apply FITS albedo map if loaded (interpolate to face centers)
    if (albedo_map && albedo_map->isValid() && !albedo_loaded) {
        face_albedos = computeFaceAlbedos(mesh, albedo_map, default_visible_albedo);
        albedo_loaded = true;
    }

    if (!albedo_loaded) {
        if (albedo_map_disabled) {
            std::cout << "[Streamed] Albedo map disabled, using uniform albedo "
                      << default_visible_albedo << std::endl;
        } else {
            std::cout << "[Streamed] No albedo map found, using uniform albedo "
                      << default_visible_albedo << std::endl;
        }
    }
    const double emiss_ir = 1.0 - ir_albedo;

    const size_t CHUNK_SIZE = 100000;
    const size_t NUM_CHUNKS = (n_faces + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (surface_temps_out) {
        surface_temps_out->clear();
        surface_temps_out->resize(n_faces);
    }

    std::cerr << "\n[Streamed] Running thermal models for " << n_faces
              << " faces over " << total_time_steps << " steps..." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    thermal::ProgressBar progressBar(n_faces, "Thermal Simulation", 50, true);

    std::atomic<int> completed_faces{0};
    std::atomic<bool> progress_done{false};
    JoiningThread progress_thread([&](){
        using namespace std::chrono_literals;
        int last = -1;
        while (!progress_done.load(std::memory_order_relaxed)) {
            int cur = completed_faces.load(std::memory_order_relaxed);
            if (cur != last) {
                progressBar.update(std::min(cur, static_cast<int>(n_faces)));
                last = cur;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        progressBar.update(static_cast<int>(n_faces));
    });

    for (size_t chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        size_t f0 = chunk * CHUNK_SIZE;
        size_t f1 = std::min(f0 + CHUNK_SIZE, n_faces);
        size_t localN = f1 - f0;

        #pragma omp parallel for schedule(dynamic,1) num_threads(thermal_threads)
        for (size_t i = 0; i < localN; ++i) {
            size_t f = f0 + i;

            // Build on-the-fly absorbed flux for this face across all times
            std::vector<double> face_flux(total_time_steps);
            const double A_vis = face_albedos[f];

            for (int t = 0; t < total_time_steps; ++t) {
                const double Fd  = get_direct(f, t);
                const double Fr  = get_reflected(f, t);
                const double Fir = get_ir(f, t);
                face_flux[t] = (1.0 - A_vis) * (Fd + Fr) + emiss_ir * Fir;
            }

            // Solve thermal for this face with matched times
            double face_qb = (face_Qb && f < face_Qb->size()) ? (*face_Qb)[f] : -1.0;
            // Per-face H from the property map (-1 = use planet default).
            double face_H = (face_H_overrides && f < face_H_overrides->size())
                          ? (*face_H_overrides)[f]
                          : -1.0;
            std::vector<double> temps = runThermalModelWithMatchedTimes(
                face_flux, times, mesh->latitudes[f], config, /*equilibration_steps*/0,
                face_qb, face_H);

            if (surface_temps_out) {
                (*surface_temps_out)[f].swap(temps);
            }

            // Increment atomic counter (no direct progress bar writes in parallel region)
            completed_faces.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Stop poller and finish bar
    progress_done.store(true, std::memory_order_relaxed);
    progress_thread.join();
    progressBar.finish();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cerr << "[Streamed] Thermal computation completed in "
              << std::fixed << std::setprecision(3)
              << (ms/1000.0) << " s (" << ms << " ms)" << std::endl;

    return true;
}

// Function to run thermal models for all mesh faces
bool runThermalModelsForAllFaces(
    const std::string& thermal_output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    int equilibration_steps,
    int total_time_steps,
    int thermal_threads,
    const thermal::Planet* planet,
    const std::vector<double>* face_Qb) {
    
    // Load albedo map if specified
    std::shared_ptr<ptm::SpatialMap> albedo_map;
    std::string albedo_map_path;

    // Check if the property exists in the config
    if (config.custom_properties.find("albedo_map") != config.custom_properties.end()) {
        albedo_map_path = config.getProperty("albedo_map");
        
        if (!albedo_map_path.empty()) {
            albedo_map = std::make_shared<ptm::SpatialMap>();
            if (albedo_map->loadFromFITS(albedo_map_path)) {
                std::cout << "Using albedo map: " << albedo_map_path << std::endl;
                
                // Print some statistics about the map
                std::cout << "  - Map resolution: " << albedo_map->getWidth() << "x" << albedo_map->getHeight() << std::endl;
                std::cout << "  - Albedo range: " << albedo_map->getMin() << " to " << albedo_map->getMax() << std::endl;
                std::cout << "  - Coordinate range: Lon [" << albedo_map->getLonMin() << "," << albedo_map->getLonMax() 
                          << "], Lat [" << albedo_map->getLatMin() << "," << albedo_map->getLatMax() << "]" << std::endl;
            } else {
                std::cout << "Warning: Failed to load albedo map: " << albedo_map_path << std::endl;
                std::cout << "Error: " << albedo_map->getLastError() << std::endl;
                albedo_map.reset();
            }
        }
    }
    
    // Use illumination model's albedo instead of constants::BODY_ALBEDO
    //double visible_albedo = illumination_model->getAlbedo();
    ir_albedo = DEFAULT_IR_ALBEDO;

    // Calculate total absorbed flux at each face and time
    // Pass planet for incidence-dependent albedo if available
    double solar_constant = planet ? planet->S : 0.0;
    std::vector<double> absorbed_flux = calculateAbsorbedFlux(
        direct_solar_values,            // from illumination_model or extracted accessors
        primary_reflected_values,
        primary_ir_values,
        total_time_steps,               // FIX: use total_time_steps, not filtered time_steps
        albedo_map,
        mesh,
        default_visible_albedo,
        ir_albedo,
        config,
        planet,                         // Pass planet for incidence-dependent albedo
        solar_constant);
    
    size_t n_faces = mesh->faces.size();
    
    // Print initial message
    std::cerr << "\nRunning thermal models for " << n_faces << " mesh faces..." << std::endl;
    
    // Start timing - Add timer here
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // CHUNKED PROCESSING - for large meshes
    // Process faces in chunks to limit peak memory usage
    const size_t CHUNK_SIZE = 100000; // Process 100k faces at a time 
    const size_t NUM_CHUNKS = (n_faces + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Initialize results container if output is requested
    if (surface_temps_out) {
        surface_temps_out->clear();
        surface_temps_out->resize(n_faces);
    }
    
    // Atomic counter for progress tracking
    std::atomic<int> completed_faces_total(0);
    
    // Process in chunks to reduce memory pressure
    bool success = true;
    
    // Create progress bar for thermal calculation - using cerr for output
    thermal::ProgressBar progressBar(n_faces, "Thermal Simulation", 50, true);

    std::atomic<bool> progress_done{false};
    JoiningThread progress_thread([&](){
        using namespace std::chrono_literals;
        int last = -1;
        while (!progress_done.load(std::memory_order_relaxed)) {
            int cur = completed_faces_total.load(std::memory_order_relaxed);
            if (cur != last) {
                progressBar.update(std::min(cur, static_cast<int>(n_faces)));
                last = cur;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Final update to 100%
        progressBar.update(static_cast<int>(n_faces));
    });

    for (size_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        size_t start_face = chunk * CHUNK_SIZE;
        size_t end_face = std::min((chunk + 1) * CHUNK_SIZE, n_faces);
        size_t chunk_size = end_face - start_face;
        
        // Show chunk info only if there are multiple chunks
        if (NUM_CHUNKS > 1) {
            std::cerr << "Processing chunk " << (chunk + 1) << " of " << NUM_CHUNKS 
                      << " (faces " << start_face << " to " << (end_face - 1) << ")" << std::endl;
        }
        
        // Temporary storage for this chunk
        std::vector<std::vector<double>> chunk_results(chunk_size);
        
        // Initialize each temperature array with default value of 50.0K (same as original code)
        for (size_t i = 0; i < chunk_size; i++) {
            chunk_results[i].resize(total_time_steps, 50.0);
        }
        
        // Reset counter for this chunk
        // completed_faces_atomic.store(0);
        
        // Process faces in this chunk in parallel
        #pragma omp parallel for schedule(dynamic, 1) num_threads(thermal_threads)
        for (size_t local_idx = 0; local_idx < chunk_size; local_idx++) {
            size_t f = start_face + local_idx;
            
            try {
                // Process each face
                std::vector<double> face_flux(total_time_steps);
                for (int t = 0; t < total_time_steps; t++) {
                    size_t idx = f * total_time_steps + t;
                    if (idx < absorbed_flux.size()) {
                        face_flux[t] = absorbed_flux[idx];
                    } else {
                        face_flux[t] = 0.0; // Safety fallback
                    }
                }

                // Run thermal model for this face
                double face_qb = (face_Qb && f < face_Qb->size()) ? (*face_Qb)[f] : -1.0;
                // Per-face H from the property map (Diviner / user-supplied);
                // -1.0 means "use planet default".
                double face_H = (planet && f < planet->face_H_overrides.size())
                              ? planet->face_H_overrides[f]
                              : -1.0;
                std::vector<double> temps = runThermalModelWithMatchedTimes(
                    face_flux,
                    times,
                    mesh->latitudes[f],
                    config,
                    equilibration_steps,
                    face_qb,
                    face_H
                );

                // Store results locally in the chunk
                chunk_results[local_idx] = std::move(temps);
                
                // Update global progress (one face completed)
                completed_faces_total.fetch_add(1, std::memory_order_relaxed);
                
                // Update progress
                // int completed = ++completed_faces_atomic;
                // if (completed % 100 == 0 || completed == static_cast<int>(chunk_size)) {
                //     // Update the progress bar with global progress
                //     progressBar.update(start_face + completed);
                // }

                // Stop poller and finish bar
                // progress_done.store(true, std::memory_order_relaxed);
                // progress_thread.join();
                // progressBar.finish();

            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Error processing face " << f << ": " << e.what() << std::endl;
                    success = false;
                }
            } catch (...) {
                #pragma omp critical
                {
                    std::cerr << "Unknown exception in thermal model for face " << f << std::endl;
                    success = false;
                }
            }
        }
        
        // Copy chunk results to main output if requested
        if (surface_temps_out) {
            for (size_t local_idx = 0; local_idx < chunk_size; local_idx++) {
                size_t f = start_face + local_idx;
                (*surface_temps_out)[f] = std::move(chunk_results[local_idx]);
                
                // Verify the data was copied correctly
                if ((*surface_temps_out)[f].size() != total_time_steps) {
                    std::cerr << "Warning: Face " << f << " has " 
                            << (*surface_temps_out)[f].size() 
                            << " time steps instead of " << total_time_steps << std::endl;
                }
            }
            
            // Additional verification: Check for constant temperatures
            if (chunk == 0 && start_face == 0) {  // Only check first chunk's first few faces
                for (size_t check_idx = 0; check_idx < std::min(size_t(5), chunk_size); check_idx++) {
                    const auto& temps = (*surface_temps_out)[check_idx];
                    if (temps.size() > 1) {
                        bool all_same = true;
                        double first_temp = temps[0];
                        for (size_t t = 1; t < temps.size(); t++) {
                            if (std::abs(temps[t] - first_temp) > 0.01) {
                                all_same = false;
                                break;
                            }
                        }
                        if (all_same) {
                            std::cerr << "Warning: Face " << check_idx 
                                    << " has constant temperature across all time steps: " 
                                    << first_temp << " K" << std::endl;
                        }
                    }
                }
            }
        }
        
        // Force cleanup of chunk memory before starting next chunk
        chunk_results.clear();
        chunk_results.shrink_to_fit();
        
        // Add a newline between chunks if there are multiple
        if (NUM_CHUNKS > 1 && chunk < NUM_CHUNKS - 1) {
            std::cerr << std::endl;
        }
    }
    
    // Stop poller and finish bar AFTER all chunks complete
    progress_done.store(true, std::memory_order_relaxed);
    progress_thread.join();
    progressBar.finish();
    
    // End timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;
    
    // Print timing information using cerr to match illumination code
    std::cerr << "Thermal computation completed in " << std::fixed << std::setprecision(3) 
              << seconds << " seconds (" << duration.count() << " ms)" << std::endl;
    
    if (success) {
        std::cerr << "\nThermal models completed successfully!" << std::endl;
    } else {
        std::cerr << "\nThermal models completed with errors!" << std::endl;
    }

    return success;
}

// =============================================================================
// SIMD Batch Thermal Solver (SIMD-over-pixels)
// Processes 8 faces at once using AVX-512 (or 4 with AVX2)
// =============================================================================
bool runThermalModelsForAllFacesBatchSoA(
    const std::string& thermal_output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int time_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    int equilibration_steps,
    int total_time_steps,
    int thermal_threads,
    const std::vector<double>* face_Qb,
    const std::vector<double>* face_H_overrides)
{
    const size_t n_faces = mesh->faces.size();
    constexpr int BS = thermal::THERMAL_BATCH_SIZE;
    const size_t n_batches = (n_faces + BS - 1) / BS;

    std::cerr << "\n=== Running SIMD Batch Thermal Models (SIMD-over-Pixels) ===" << std::endl;
    std::cerr << "Processing " << n_faces << " faces in " << n_batches << " batches of " << BS << std::endl;
#if defined(__AVX512F__)
    std::cerr << "Using AVX-512 (8 faces per SIMD operation)" << std::endl;
#elif defined(__AVX2__)
    std::cerr << "Using AVX2 (4 faces per SIMD operation)" << std::endl;
#else
    std::cerr << "Using scalar fallback" << std::endl;
#endif

    // Build planet & optional explicit FITS albedo map up front so we can
    // route the per-face absorbed flux through calculateAbsorbedFlux — that
    // reuses the standard path's spatial-albedo + incidence-angle-albedo
    // logic. Without this, the batch path previously hardcoded a uniform
    // albedo and drifted from the standard path by up to ~115 K at
    // albedo-sensitive faces (spatial map + high-incidence terminator).
    thermal::Planet planet(config.getProperty("planet", "Moon"));

    std::shared_ptr<ptm::SpatialMap> batch_albedo_map;
    if (config.custom_properties.find("albedo_map") != config.custom_properties.end()) {
        const std::string path = config.getProperty("albedo_map");
        if (!path.empty()) {
            batch_albedo_map = std::make_shared<ptm::SpatialMap>();
            if (!batch_albedo_map->loadFromFITS(path)) {
                std::cout << "[SIMD Batch] Failed to load explicit albedo_map: " << path
                          << " (" << batch_albedo_map->getLastError() << ")" << std::endl;
                batch_albedo_map.reset();
            }
        }
    }

    std::cerr << "Computing absorbed flux (per-face spatial + incidence-angle albedo)..." << std::endl;
    std::vector<double> absorbed_flux_rowmajor = calculateAbsorbedFlux(
        direct_solar_values,
        primary_reflected_values,
        primary_ir_values,
        total_time_steps,
        batch_albedo_map,
        mesh,
        default_visible_albedo,
        ir_albedo,
        config,
        &planet,
        planet.S);
    if (absorbed_flux_rowmajor.empty() ||
        absorbed_flux_rowmajor.size() != n_faces * static_cast<size_t>(total_time_steps)) {
        std::cerr << "[SIMD Batch] calculateAbsorbedFlux returned unexpected size -- aborting.\n";
        return false;
    }

    // Transpose to COLUMN-MAJOR for SIMD efficiency.
    // Layout: flux_column_major[t * n_faces + f] - timestep-major ordering.
    std::vector<double> flux_column_major(n_faces * total_time_steps, 0.0);
    constexpr size_t BLOCK_F = 64;
    constexpr size_t BLOCK_T = 64;
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t f_block = 0; f_block < n_faces; f_block += BLOCK_F) {
        for (int t_block = 0; t_block < total_time_steps; t_block += static_cast<int>(BLOCK_T)) {
            size_t f_end = std::min(f_block + BLOCK_F, n_faces);
            int t_end = std::min(t_block + static_cast<int>(BLOCK_T), total_time_steps);
            for (size_t f = f_block; f < f_end; ++f) {
                for (int t = t_block; t < t_end; ++t) {
                    size_t src_idx = f * static_cast<size_t>(total_time_steps) + t;
                    size_t dst_idx = static_cast<size_t>(t) * n_faces + f;
                    flux_column_major[dst_idx] = absorbed_flux_rowmajor[src_idx];
                }
            }
        }
    }
    std::vector<double>().swap(absorbed_flux_rowmajor);

    // Initialize output
    if (surface_temps_out) {
        surface_temps_out->resize(n_faces);
        for (auto& v : *surface_temps_out) {
            v.resize(total_time_steps, 0.0);
        }
    }

    // Create reference profile for grid parameters (reuses `planet` above).
    thermal::Profile ref_profile(planet, 0.0, config, false);
    const int n_layers = ref_profile.nlayers;

    std::cerr << "Using " << n_layers << " depth layers per profile" << std::endl;

    // Progress tracking
    std::atomic<size_t> completed_batches{0};
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create progress bar
    thermal::ProgressBar progressBar(static_cast<int>(n_faces), "SIMD Batch thermal", 50, true);

    bool success = true;

    #pragma omp parallel num_threads(thermal_threads)
    {
        // Thread-local batch buffer
        thermal::ThermalBatchSoA batch;
        batch.initialize(n_layers, BS, planet, config);
        batch.copyGridFromProfile(ref_profile);

        #pragma omp for schedule(dynamic, 1)
        for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
            size_t start_face = batch_idx * BS;
            size_t end_face = std::min(start_face + BS, n_faces);
            int faces_in_batch = static_cast<int>(end_face - start_face);

            batch.n_faces = faces_in_batch;

            // Per-face H-parameter override: when the planet has a per-face H
            // map loaded (Diviner Hayne+2017 or user-supplied), refresh kc/rho
            // for THIS batch using each face's local H. The grid (z, dz, g1,
            // g2) stays as the ref-profile grid — built from the body-default
            // skin depth — which is fine because individual face H values vary
            // by < 10x and the geometric stretching keeps every layer well-
            // resolved across that span.
            std::array<double, thermal::THERMAL_BATCH_SIZE> H_batch{};
            const bool have_face_H = (face_H_overrides != nullptr) &&
                                     !face_H_overrides->empty();
            if (have_face_H) {
                for (int f = 0; f < faces_in_batch; ++f) {
                    const size_t gf = start_face + f;
                    H_batch[f] = (gf < face_H_overrides->size())
                                ? (*face_H_overrides)[gf]
                                : planet.H;
                }
                for (int f = faces_in_batch; f < thermal::THERMAL_BATCH_SIZE; ++f) {
                    H_batch[f] = planet.H;
                }
                batch.copyGridFromProfile(ref_profile, H_batch.data());
            }

            // Override per-face Qb if mask is provided
            if (face_Qb) {
                for (int f = 0; f < faces_in_batch; ++f) {
                    size_t gf = start_face + f;
                    if (gf < face_Qb->size()) {
                        batch.Qb[f] = (*face_Qb)[gf];
                    }
                }
            }

            // Initialize temperatures with a depth-dependent profile, mirroring
            // Profile::init_T (thermal_model.cpp:706). A uniform initial T at
            // all depths takes many lunar days of forcing to relax into the
            // correct geothermal gradient; for high-latitude faces it can
            // leave ~100 K surface-vs-deep residual bias even after 24 lunar
            // days of equilibration. Using the same exponential surface->deep
            // blend as the standard path gets the batch into the same
            // accuracy class as Model::run.
            for (int f = 0; f < faces_in_batch; ++f) {
                size_t global_f = start_face + f;

                // Per-face surface estimate from the face's actual average flux
                // (includes shadowing, reflected, IR — already baked into
                // flux_column_major by calculateAbsorbedFlux above).
                double avg_flux = 0.0;
                const int flux_avg_window = std::min(100, total_time_steps);
                for (int t = 0; t < flux_avg_window; ++t) {
                    avg_flux += flux_column_major[t * n_faces + global_f];
                }
                avg_flux /= std::max(1, flux_avg_window);

                double T_surf_est = 200.0;
                if (avg_flux > 0) {
                    double ratio = avg_flux / (planet.emissivity * SIGMA_SB);
                    T_surf_est = std::sqrt(std::sqrt(ratio));
                    T_surf_est = std::max(50.0, std::min(400.0, T_surf_est));
                }

                // Deep temperature: geothermal gradient on top of surface avg,
                // capped to prevent runaway (same formula as Profile::init_T).
                const double R350 = config.getR350();
                const double k_deep = planet.kd * (1.0 + R350 * T_surf_est * T_surf_est * T_surf_est);
                const double depth = ref_profile.z().back();
                const double gradient = (k_deep > 0.0) ? (planet.Qb / k_deep) : 0.0;
                const double delta_T_geo = std::min(gradient * depth, 50.0);
                const double T_deep = T_surf_est + delta_T_geo;

                // Exponential skin-depth blend from T_surf_est at surface to T_deep at depth.
                double cp_for_skin = planet.cp0;
                double T_repr = 200.0;
                double T_pow = T_repr;
                for (size_t j = 0; j < planet.cpCoeff.size(); ++j) {
                    cp_for_skin += planet.cpCoeff[j] * T_pow;
                    T_pow *= T_repr;
                }
                if (cp_for_skin < 100.0) cp_for_skin = 600.0;
                const double kappa0 = planet.ks / (planet.rhos * cp_for_skin);
                const double skin_d = thermal::skinDepth(planet.day, kappa0);
                const double inv_skin = (skin_d > 0.0) ? (1.0 / skin_d) : 0.0;

                batch.T_eq_init[f] = T_surf_est;
                for (int layer = 0; layer < n_layers; ++layer) {
                    const double z_layer = ref_profile.z()[layer];
                    const double blend = std::exp(-z_layer * inv_skin);
                    batch.T[layer * BS + f] = T_deep + (T_surf_est - T_deep) * blend;
                }
            }

            // Zero-pad unused face slots (use a benign mid-range temperature).
            for (int f = faces_in_batch; f < BS; ++f) {
                batch.T_eq_init[f] = 200.0;
                for (int layer = 0; layer < n_layers; ++layer) {
                    batch.T[layer * BS + f] = 200.0;
                }
            }

            // Equilibration phase:
            // Mirror Model::fastEquilibrate — run NYEARSEQ * round(year/day) lunar days
            // of diurnal forcing before the main output loop. Cycle the precomputed
            // flux trajectory as a quasi-periodic signal and substep it down to
            // dt <= planet.day / 192 so CN accuracy stays within ~1 K worst case
            // (see src/cpp/tools/test_equilibration_dt.cpp for the accuracy study).
            const double equil_days_needed = static_cast<double>(config.NYEARSEQ) *
                                             std::round(planet.year / planet.day);
            const double equil_duration_s = equil_days_needed * planet.day;

            if (equil_duration_s > 0.0 && total_time_steps >= 2) {
                // Match Model::fastEquilibrate (thermal_model.cpp:2024) exactly:
                // analytical clear-sky diurnal forcing using planet.albedo (not
                // per-face spatial albedo, not incidence-angle albedo). The
                // standard path runs this simplified equilibration first, then
                // reconciles to full-physics fluxes during the main loop. If
                // the batch path instead pre-equilibrates under full physics
                // the two paths converge to different steady states and the
                // short main loop isn't enough to bridge the gap (observed up
                // to ~115 K max residual).
                //
                // dt choice: planet.day / 192 gives <1 K worst-case accuracy
                // per src/cpp/tools/test_equilibration_dt.cpp. Much finer than
                // that (and thus closer to fastEquilibrate's Fourier-stable dt)
                // buys accuracy < 0.5 K at ~2× substep count.
                const double target_dt = planet.day / 768.0;
                const int steps_per_day = std::max(1,
                    static_cast<int>(std::round(planet.day / target_dt)));
                const double dt_sub = planet.day / static_cast<double>(steps_per_day);
                const int total_substeps = static_cast<int>(std::llround(
                    equil_duration_s / dt_sub));

                alignas(64) double flux_vec[thermal::THERMAL_BATCH_SIZE];
                for (int f = 0; f < thermal::THERMAL_BATCH_SIZE; ++f) flux_vec[f] = 0.0;

                // Pre-compute per-face cos(lat) and the planetary absorbed
                // pre-factor so the per-substep inner loop is lean.
                double cos_lat[thermal::THERMAL_BATCH_SIZE];
                for (int f = 0; f < thermal::THERMAL_BATCH_SIZE; ++f) cos_lat[f] = 0.0;
                for (int f = 0; f < faces_in_batch; ++f) {
                    const size_t gf = start_face + f;
                    const double lat_rad = mesh->latitudes[gf] * (M_PI / 180.0);
                    cos_lat[f] = std::cos(lat_rad);
                }
                const double planetary_absorption = planet.S * (1.0 - planet.albedo);

                for (int step = 0; step < total_substeps; ++step) {
                    const double day_fraction = static_cast<double>(step % steps_per_day)
                                                / static_cast<double>(steps_per_day);
                    const double hour_angle = 2.0 * M_PI * day_fraction + M_PI;
                    const double cos_hour = std::cos(hour_angle);
                    for (int f = 0; f < faces_in_batch; ++f) {
                        const double cos_zen = cos_lat[f] * cos_hour;
                        const double flux_factor = (cos_zen > 0.0) ? cos_zen : 0.0;
                        flux_vec[f] = planetary_absorption * flux_factor;
                    }
                    batch.setFluxContiguous(flux_vec, faces_in_batch);
                    batch.advanceTimestep(dt_sub);
                }
            }

            // Main simulation time-stepping loop.
            // Convention (matches Model::run in thermal_model.cpp:1842):
            //   surface_temps_out[f][t] == state at times[t]
            // So we store the initial state at index 0 before the first advance,
            // then after each advanceTimestep(dt = times[t+1] - times[t]) we store
            // the post-advance state at index t+1. Without this the previous
            // code stored the post-advance state at index t (off-by-one) and
            // duplicated the final state into indices total_time_steps-2 and -1.
            if (surface_temps_out) {
                for (int f = 0; f < faces_in_batch; ++f) {
                    (*surface_temps_out)[start_face + f][0] = batch.getSurfaceTemp(f);
                }
            }
            // Main-loop substepping with linear flux interpolation.
            // The effective dt ceiling is set by thermal-diffusion accuracy at
            // the finest depth layer (Fourier number ~0.5 to avoid CN's
            // low-order oscillatory error), not by flux-transition detection.
            // An earlier experiment with flux-transition-based adaptive
            // substepping gave only ~7% speedup at ~3x worse max error, so we
            // use a uniform target dt here.
            const double main_target_dt = planet.day / 768.0;
            alignas(64) double flux_step[thermal::THERMAL_BATCH_SIZE];
            for (int f = 0; f < thermal::THERMAL_BATCH_SIZE; ++f) flux_step[f] = 0.0;
            for (int t = 0; t < total_time_steps - 1; ++t) {
                const double dt_output = times[t + 1] - times[t];
                const double* flux_a = &flux_column_major[t       * n_faces + start_face];
                const double* flux_b = &flux_column_major[(t + 1) * n_faces + start_face];

                const int n_sub = std::max(1,
                    static_cast<int>(std::ceil(dt_output / main_target_dt)));
                const double dt_sub = dt_output / n_sub;

                for (int s = 0; s < n_sub; ++s) {
                    const double alpha = (s + 0.5) / static_cast<double>(n_sub);
                    const double one_m = 1.0 - alpha;
                    for (int f = 0; f < faces_in_batch; ++f) {
                        flux_step[f] = one_m * flux_a[f] + alpha * flux_b[f];
                    }
                    batch.setFluxContiguous(flux_step, faces_in_batch);
                    batch.advanceTimestep(dt_sub);
                }
                if (surface_temps_out) {
                    for (int f = 0; f < faces_in_batch; ++f) {
                        (*surface_temps_out)[start_face + f][t + 1] = batch.getSurfaceTemp(f);
                    }
                }
            }

            // Update progress
            size_t done = ++completed_batches;
            #pragma omp critical
            {
                progressBar.update(static_cast<int>(done * BS));
            }
        }
    }

    progressBar.finish();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;

    std::cerr << "SIMD Batch thermal computation completed in " << std::fixed << std::setprecision(3)
              << seconds << " seconds (" << duration.count() << " ms)" << std::endl;

    if (success) {
        std::cerr << "\nSIMD Batch thermal models completed successfully!" << std::endl;
    }

    return success;
}

// Function to export thermal results to VTK format - complete rewrite
bool addThermalDataToVTK(
    const std::string& vtk_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times) {

    // Read the entire original file
    std::ifstream infile(vtk_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open VTK file: " << vtk_file << std::endl;
        return false;
    }

    std::vector<std::string> content;
    std::string line;
    bool found_cell_data = false;
    int cell_data_line = -1;
    int cell_count = 0;
    
    // Read and parse the file
    while (std::getline(infile, line)) {
        content.push_back(line);
        if (line.find("CELL_DATA") != std::string::npos) {
            found_cell_data = true;
            cell_data_line = content.size() - 1;
            std::istringstream iss(line);
            std::string dummy;
            iss >> dummy >> cell_count;
        }
    }
    infile.close();
    
    // Make sure we found the CELL_DATA section
    if (!found_cell_data) {
        std::cerr << "Could not find CELL_DATA section in VTK file" << std::endl;
        return false;
    }
    
    // Create temperature data blocks
    std::vector<std::string> temp_data;
    for (size_t t = 0; t < times.size(); t++) {
        temp_data.push_back(std::string("SCALARS temperature_") + std::to_string(t) + " float 1");
        temp_data.push_back("LOOKUP_TABLE default");
        
        // Write one value per face
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            double temp = 50.0;  // Default value
            if (f < surface_temps.size() && t < surface_temps[f].size()) {
                temp = surface_temps[f][t];
                if (std::isnan(temp) || std::isinf(temp)) {
                    temp = 50.0;  // Safety for visualization
                }
            }
            temp_data.push_back(std::to_string(temp));
        }
    }
    
    // Find the end of all existing CELL_DATA sections to insert our data
    int insert_pos = cell_data_line + 1;
    bool in_scalars = false;
    for (size_t i = cell_data_line + 1; i < content.size(); i++) {
        if (content[i].find("SCALARS") != std::string::npos) {
            in_scalars = true;
            continue;
        }
        
        if (in_scalars && content[i].find("LOOKUP_TABLE") != std::string::npos) {
            // Skip the next cell_count lines (the data values)
            i += cell_count;
            in_scalars = false;
            insert_pos = i + 1;  // Insert after this block
            continue;
        }
        
        // If we find a new section, stop
        if (content[i].find("POINT_DATA") != std::string::npos ||
            content[i].find("FIELD") != std::string::npos) {
            insert_pos = i;
            break;
        }
    }
    
    // Insert our temperature data at the proper position
    content.insert(content.begin() + insert_pos, temp_data.begin(), temp_data.end());
    
    // Write the modified file
    std::ofstream outfile(vtk_file);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output VTK file for writing" << std::endl;
        return false;
    }
    
    for (const auto& line : content) {
        outfile << line << std::endl;
    }
    
    outfile.close();
    std::cout << "Temperature data added to VTK file: " << vtk_file << std::endl;
    std::cout << "Thermal VTK file generated successfully" << std::endl;
    return true;
}

// ========== GPU PERSISTENT THERMAL SOLVER ==========
#ifdef PTM_GPU_OPENCL
/**
 * @brief GPU-accelerated thermal solver using persistent kernel
 *
 * Processes ALL timesteps in a single GPU kernel launch for maximum efficiency.
 * Data transfer is minimized: upload flux once, download surface temps once.
 *
 * For large meshes (>100K faces), processes faces in batches to fit in GPU memory.
 */
bool runThermalModelsForAllFaces_GPU_Persistent(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<double>& times,
    int total_time_steps,
    int equilibration_steps,
    const thermal::Config& config,
    double default_visible_albedo,
    double ir_albedo,
    std::vector<std::vector<double>>* surface_temps_out,
    const std::vector<double>* face_Qb,
    const std::vector<double>* face_H_overrides,
    ice::IceStabilityAccumulator* ice_acc,
    const std::string& initial_state_file,
    const std::string& final_state_file,
    int gpu_devices,
    const std::vector<int>& gpu_device_ids,
    size_t gpu_batch_size_override,
    int gpu_inter_batch_pause_ms)
{
    using namespace gpu::opencl;

    // Debug output control - set to true for troubleshooting
    constexpr bool GPU_DEBUG_OUTPUT = false;

    const size_t n_faces = mesh->faces.size();
    std::cout << "[GPU Thermal] Starting persistent kernel solver for " << n_faces
              << " faces, " << total_time_steps << " timesteps" << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    // Initialize output. In two-phase mode, only the output phase temperatures
    // are stored (not equilibration), saving ~eq_steps/total * memory.
    // Initialize output
    if (surface_temps_out) {
        surface_temps_out->resize(n_faces);
        for (auto& v : *surface_temps_out) {
            v.resize(total_time_steps, 0.0);
        }
    }

    // Get thermal model parameters
    thermal::Planet planet(config.getProperty("planet", "Moon"));
    thermal::Profile reference_profile(planet, 0.0, config, false);
    const auto& z_grid = reference_profile.z();
    const size_t n_layers = z_grid.size();

    // ---- Optional initial / final thermal-state checkpoint -------------
    // Magic bytes 'PTMT' (0x50544D54) identify the format.
    constexpr uint32_t kThermalStateMagic = 0x50544D54;

    // Load initial T(z) once; we'll slice per-batch below.
    // Layout in file: layer-major across ALL faces, float32, size
    // n_layers * n_faces. Matches the GPU SoA in buf_T_.
    std::vector<float> loaded_initial_T;
    if (!initial_state_file.empty()) {
        std::ifstream is(initial_state_file, std::ios::binary);
        if (!is) {
            std::cerr << "[GPU Thermal] ERROR: cannot open initial_state_file '"
                      << initial_state_file << "'\n";
            return false;
        }
        uint32_t magic = 0, file_n_faces = 0, file_n_layers = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(&file_n_faces), sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(&file_n_layers), sizeof(uint32_t));
        if (!is || magic != kThermalStateMagic) {
            std::cerr << "[GPU Thermal] ERROR: bad magic in initial_state_file '"
                      << initial_state_file << "' (got 0x" << std::hex << magic
                      << ", expected 0x" << kThermalStateMagic << std::dec << ")\n";
            return false;
        }
        if (file_n_faces != static_cast<uint32_t>(n_faces) ||
            file_n_layers != static_cast<uint32_t>(n_layers)) {
            std::cerr << "[GPU Thermal] ERROR: initial_state_file shape mismatch "
                      << "(file=" << file_n_faces << "x" << file_n_layers
                      << ", live=" << n_faces << "x" << n_layers << ")\n";
            return false;
        }
        const size_t expected = static_cast<size_t>(file_n_faces) * file_n_layers;
        loaded_initial_T.resize(expected);
        is.read(reinterpret_cast<char*>(loaded_initial_T.data()),
                expected * sizeof(float));
        if (!is) {
            std::cerr << "[GPU Thermal] ERROR: short read on initial_state_file\n";
            return false;
        }
        std::cout << "[GPU Thermal] Loaded initial T(z) checkpoint: "
                  << initial_state_file << " ("
                  << expected * sizeof(float) / (1024.0*1024.0) << " MiB)\n";
    }

    // Whole-mesh buffer to receive the final T(z) profile, if requested.
    // Filled per-batch and dumped at the end.
    std::vector<float> assembled_final_T;
    const bool dump_final_state = !final_state_file.empty();
    if (dump_final_state) {
        assembled_final_T.assign(n_layers * n_faces, 0.0f);
    }
    // -------------------------------------------------------------------

    // Calculate dt for each phase. When equilibration uses a different cadence
    // from the output phase, we run the GPU solver in two passes.
    const int output_steps = total_time_steps - equilibration_steps;
    float eq_dt = 0.0f;
    float out_dt = 0.0f;
    bool two_phase = false;

    if (equilibration_steps > 1) {
        eq_dt = static_cast<float>(times[1] - times[0]);
    }
    if (output_steps > 1 && equilibration_steps < total_time_steps) {
        out_dt = static_cast<float>(times[equilibration_steps + 1] - times[equilibration_steps]);
    } else if (equilibration_steps > 1) {
        out_dt = eq_dt;
    }
    if (equilibration_steps <= 0) eq_dt = out_dt;

    two_phase = (equilibration_steps > 0 && output_steps > 0 &&
                 std::abs(eq_dt - out_dt) > 1e-3f);

    // For single-phase (legacy), use a single dt.
    float dt = two_phase ? eq_dt : (total_time_steps > 1 ? static_cast<float>(times[1] - times[0]) : 0.0f);

    if (two_phase) {
        std::cout << "[GPU Thermal] Two-phase execution: eq_dt="
                  << std::fixed << std::setprecision(1) << eq_dt << "s ("
                  << equilibration_steps << " steps), out_dt="
                  << out_dt << "s (" << output_steps << " steps)" << std::endl;
    }

    // GPU memory constraint: estimate ~50 bytes per face per layer for arrays
    // Plus flux and output: ~8 bytes per face per timestep * 2
    // Conservative batch size based on 6 GB GPU memory budget.
    // Cap at 500K faces/batch: large enough to process typical res-7 meshes in
    // one batch (fewer per-batch setup/CL-buffer creations + persistent-kernel
    // launches), small enough to fit comfortably on 4–8 GB integrated GPUs.
    const size_t bytes_per_face = n_layers * 50 + total_time_steps * 8 * 2;
    const size_t gpu_memory_limit = 6ULL * 1024 * 1024 * 1024;  // 6 GB conservative
    const size_t max_batch_size_legacy = std::min(n_faces, gpu_memory_limit / bytes_per_face);
    // gpu_batch_size_override > 0 lets the user shrink batches (watchdog mitigation
    // on macOS+AMD; see memory/gpu_hang_amd_macos.md). 0 keeps legacy behaviour.
    const size_t BATCH_SIZE = (gpu_batch_size_override > 0)
        ? std::min(n_faces, std::min(max_batch_size_legacy, gpu_batch_size_override))
        : std::min(max_batch_size_legacy, static_cast<size_t>(500000));

    std::cout << "[GPU Thermal] Using batch size: " << BATCH_SIZE
              << " (n_layers=" << n_layers << ")" << std::endl;

    // Prepare face albedos - handle custom meshes (FITS interpolation) vs standard icospheres (binary files)
    std::vector<double> face_albedos(n_faces, default_visible_albedo);
    bool albedo_loaded = false;

    // Check if we're using a custom mesh
    bool is_custom_mesh = false;
    try {
        auto it = config.custom_properties.find("is_custom_mesh");
        if (it != config.custom_properties.end()) {
            is_custom_mesh = (it->second == "1" || it->second == "true");
        }
    } catch (const std::exception& e) {
        // Use default (not custom mesh)
    }

    // Get body name from config
    std::string body_name = config.getProperty("target_body_name");
    if (!body_name.empty()) {
        // Convert to proper case
        std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
        body_name[0] = std::toupper(body_name[0]);
    }

    // Check if albedo map loading is disabled via use_albedo_map config option
    bool albedo_map_disabled = (config.getProperty("use_albedo_map") == "0");
    if (albedo_map_disabled) {
        std::cout << "[GPU Thermal] Albedo map loading disabled (use_albedo_map: false), "
                  << "using uniform albedo " << default_visible_albedo << " for all faces" << std::endl;
    }

    // Get albedo maps directory from config
    std::string albedo_maps_dir = "data/albedo_maps";
    bool albedo_dir_explicitly_set = false;
    if (config.custom_properties.count("albedo_maps_directory") > 0) {
        albedo_maps_dir = config.getProperty("albedo_maps_directory");
        albedo_dir_explicitly_set = true;
    }

    // Check if user explicitly set albedo (skip FITS auto-search)
    bool albedo_explicit = (config.custom_properties.count("thermophysical_albedo_explicit") > 0 &&
                            config.getProperty("thermophysical_albedo_explicit") == "1");

    // Skip FITS search if user explicitly set albedo or albedo_maps_directory is empty
    bool skip_fits = albedo_explicit || (albedo_dir_explicitly_set && albedo_maps_dir.empty());

    // Ensure trailing slash on albedo maps directory (used by all search paths below)
    if (!albedo_maps_dir.empty() && albedo_maps_dir.back() != '/' && albedo_maps_dir.back() != '\\') {
        albedo_maps_dir += '/';
    }

    if (!albedo_map_disabled && is_custom_mesh && !body_name.empty() && !skip_fits) {
        // For custom meshes, load FITS file and interpolate to face centers
        std::cout << "[GPU Thermal] Custom mesh detected - loading FITS albedo map for interpolation" << std::endl;

        std::string albedo_fits_name = body_name + "-0.5deg-BondAlbedo.fits";
        std::vector<std::string> search_paths;
        search_paths.push_back(albedo_maps_dir + albedo_fits_name);  // User-specified or default path
        // Only add hardcoded fallback paths if directory was NOT explicitly configured
        if (!albedo_dir_explicitly_set) {
            search_paths.push_back("../data/albedo_maps/" + albedo_fits_name);  // From build/ directory
            search_paths.push_back("data/albedo_maps/" + albedo_fits_name);     // From project root
        }

        auto albedo_map = std::make_shared<ptm::SpatialMap>();
        bool fits_loaded = false;
        std::string found_path;
        for (const auto& path : search_paths) {
            if (albedo_map->loadFromFITS(path)) {
                found_path = path;
                fits_loaded = true;
                break;
            }
        }

        if (fits_loaded) {
            std::cout << "[GPU Thermal] Loaded FITS albedo map from: " << found_path << std::endl;
            std::cout << "[GPU Thermal] Map dimensions: " << albedo_map->getWidth()
                      << "x" << albedo_map->getHeight() << std::endl;
            std::cout << "[GPU Thermal] Albedo range in map: " << albedo_map->getMin()
                      << " to " << albedo_map->getMax() << std::endl;
            std::cout << "[GPU Thermal] NOTE: Global albedo map overrides default uniform albedo ("
                      << default_visible_albedo << "). To disable, set "
                      << "albedo_maps_directory to empty in config YAML." << std::endl;

            // Interpolate FITS map to each face center
            size_t valid_count = 0;
            #pragma omp parallel for reduction(+:valid_count)
            for (size_t f = 0; f < n_faces; f++) {
                if (f < mesh->latitudes.size() && f < mesh->longitudes.size()) {
                    double face_lat = mesh->latitudes[f];
                    double face_lon = mesh->longitudes[f];

                    // Normalize longitude to 0-360 range
                    while (face_lon < 0) face_lon += 360.0;
                    while (face_lon >= 360.0) face_lon -= 360.0;

                    // Get albedo from map
                    double map_albedo = albedo_map->getValue(face_lon, face_lat);

                    if (map_albedo != albedo_map->getMissingValue() &&
                        map_albedo >= 0.01 && map_albedo <= 1.0) {
                        face_albedos[f] = map_albedo;
                        valid_count++;
                    }
                }
            }

            double min_albedo = *std::min_element(face_albedos.begin(), face_albedos.end());
            double max_albedo = *std::max_element(face_albedos.begin(), face_albedos.end());
            double avg_albedo = std::accumulate(face_albedos.begin(), face_albedos.end(), 0.0) / n_faces;
            std::cout << "[GPU Thermal] FITS interpolation complete: " << valid_count << "/" << n_faces << " faces" << std::endl;
            std::cout << "[GPU Thermal] Albedo range: " << min_albedo << " to " << max_albedo
                      << " (mean: " << avg_albedo << ")" << std::endl;
            albedo_loaded = true;
        } else {
            std::cout << "[GPU Thermal] Could not load FITS albedo map for " << body_name
                      << ", using default albedo " << default_visible_albedo << std::endl;
        }
    }
    else if (!albedo_map_disabled && is_custom_mesh && skip_fits) {
        // User explicitly set albedo or albedo_maps_directory is empty — skip FITS search
        if (albedo_explicit) {
            std::cout << "[GPU Thermal] Using explicit uniform albedo " << default_visible_albedo
                      << " (thermophysical override, FITS auto-search disabled)" << std::endl;
        } else {
            std::cout << "[GPU Thermal] Albedo maps directory is empty, skipping FITS auto-search. "
                      << "Using uniform albedo " << default_visible_albedo << std::endl;
        }
    }
    else if (!albedo_map_disabled && !body_name.empty() && config.getProperty("using_generic_body") != "true") {
        // For standard icosphere meshes, try to load pre-computed binary albedo files
        // (skipped for generic bodies — no pre-computed maps exist)
        int mesh_resolution = 5; // Default
        try {
            auto it = config.custom_properties.find("mesh_resolution");
            if (it != config.custom_properties.end()) {
                mesh_resolution = std::stoi(it->second);
            }
        } catch (const std::exception& e) {
            // Use default
        }

        // Construct binary albedo file name: e.g., "Europa_Albedo_res6.bin"
        std::string albedo_filename = body_name + "_Albedo_res" + std::to_string(mesh_resolution) + ".bin";

        // Try multiple paths for the albedo file
        std::vector<std::string> search_paths = {
            albedo_maps_dir + albedo_filename,
            "../data/albedo_maps/" + albedo_filename,
            "data/albedo_maps/" + albedo_filename
        };

        std::vector<double> loaded_albedos;
        for (const auto& path : search_paths) {
            loaded_albedos = importAlbedoMapFromFile(path);
            if (!loaded_albedos.empty()) {
                if (loaded_albedos.size() == n_faces) {
                    std::cout << "[GPU Thermal] Loaded binary albedo map from: " << path << std::endl;
                    std::cout << "[GPU Thermal] Albedo range: "
                              << *std::min_element(loaded_albedos.begin(), loaded_albedos.end())
                              << " to " << *std::max_element(loaded_albedos.begin(), loaded_albedos.end()) << std::endl;
                    face_albedos = std::move(loaded_albedos);
                    albedo_loaded = true;
                } else {
                    std::cout << "[GPU Thermal] Albedo file size mismatch: expected " << n_faces
                              << ", got " << loaded_albedos.size() << std::endl;
                }
                break;
            }
        }

        if (!albedo_loaded) {
            std::cout << "[GPU Thermal] No binary albedo map found for " << body_name
                      << " res" << mesh_resolution << ", using default albedo " << default_visible_albedo << std::endl;
        }
    } else {
        std::cout << "[GPU Thermal] No body name in config, using default albedo " << default_visible_albedo << std::endl;
    }

    // Process faces in batches
    size_t n_batches = (n_faces + BATCH_SIZE - 1) / BATCH_SIZE;
    std::cout << "[GPU Thermal] Processing " << n_faces << " faces in " << n_batches << " batch(es)..." << std::endl;

    // ----- Schorghofer & Taylor (2007) ice-stability accumulator setup -----
    // Pre-size the global accumulator and copy the depth grid. Each batch
    // contributes its slice of faces to ice_acc->sum_E[layer * n_faces + face].
    if (ice_acc) {
        ice_acc->allocate(static_cast<int>(n_faces), static_cast<int>(n_layers));
        for (size_t i = 0; i < n_layers; ++i) ice_acc->z[i] = z_grid[i];
        // n_samples will be set after the batch loop, once we know how many
        // output timesteps were accumulated.
    }

    // ----- Per-batch worker (shared by serial and multi-GPU paths) -----
    // The lambda contains everything that runs after a ThermalOpenCLSolver is
    // initialized for one batch: host-side prep + uploads + persistent
    // kernel + result download + writebacks. Single-GPU calls it inline
    // from the serial for-loop below; multi-GPU dispatches it via
    // ThermalGPUDevicePool.submit().
    //
    // Bit-equivalence note: capturing all relevant state by reference and
    // calling the lambda in the serial path is functionally identical to
    // the previous inline body (same kernel inputs, same execution order on
    // the same device).
    auto run_batch_on_solver = [&](ThermalOpenCLSolver& solver,
                                    size_t batch_idx,
                                    size_t batch_start,
                                    size_t batch_n) -> bool {
        // Prepare mesh data for this batch (SoA layout: [layer * batch_n + face_in_batch])
        std::vector<float> z_grid_f(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            z_grid_f[i] = static_cast<float>(z_grid[i]);
        }

        std::vector<float> rho_batch(batch_n * n_layers);
        std::vector<float> kc_batch(batch_n * n_layers);
        std::vector<float> emissivity_batch(batch_n);
        std::vector<float> T_deep_batch(batch_n);
        std::vector<float> Qb_batch(batch_n);

        const auto& rho_ref = reference_profile.rho();
        const auto& kc_ref = reference_profile.kc();
        const auto& T_ref = reference_profile.T();
        double T_deep_val = T_ref.back();
        double R350_val = reference_profile.R350_val;

        const bool use_per_face_H = (face_H_overrides != nullptr) &&
                                    !face_H_overrides->empty();

        for (size_t f = 0; f < batch_n; ++f) {
            size_t global_f = batch_start + f;
            emissivity_batch[f] = static_cast<float>(planet.emissivity);
            T_deep_batch[f] = static_cast<float>(T_deep_val);
            if (face_Qb && global_f < face_Qb->size()) {
                Qb_batch[f] = static_cast<float>((*face_Qb)[global_f]);
            } else {
                Qb_batch[f] = static_cast<float>(planet.Qb);
            }

            if (use_per_face_H) {
                const double H_face = (global_f < face_H_overrides->size())
                                    ? (*face_H_overrides)[global_f]
                                    : planet.H;
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    const double depth = z_grid[layer];
                    const double df    = std::exp(-depth / H_face);
                    rho_batch[idx] = static_cast<float>(planet.rhod + (planet.rhos - planet.rhod) * df);
                    kc_batch[idx]  = static_cast<float>(planet.kd  + (planet.ks  - planet.kd ) * df);
                }
            } else {
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    rho_batch[idx] = static_cast<float>(rho_ref[layer]);
                    kc_batch[idx]  = static_cast<float>(kc_ref[layer]);
                }
            }
        }

        solver.uploadMeshData(z_grid_f, rho_batch, kc_batch, emissivity_batch, T_deep_batch, Qb_batch);

        if (GPU_DEBUG_OUTPUT && batch_idx == 0) {
            std::cerr << "[GPU Thermal DEBUG] z_grid (first 10): ";
            for (size_t i = 0; i < std::min(n_layers, (size_t)10); ++i) {
                std::cerr << z_grid_f[i] << " ";
            }
            std::cerr << "\n";
            std::cerr << "[GPU Thermal DEBUG] rho[0] (first face): " << rho_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] kc[0] (first face): " << kc_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] emissivity (first face): " << emissivity_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] T_deep (first face): " << T_deep_batch[0] << "\n";
        }

        std::vector<float> cp_coeffs_f;
        for (double c : planet.cpCoeff) {
            cp_coeffs_f.push_back(static_cast<float>(c));
        }
        solver.setMaterialParams(
            static_cast<float>(planet.cp0),
            cp_coeffs_f,
            static_cast<float>(R350_val),
            static_cast<float>(5.67051196e-8)
        );

        std::vector<float> T_initial(batch_n * n_layers);
        for (size_t f = 0; f < batch_n; ++f) {
            size_t global_f = batch_start + f;
            double albedo = face_albedos[global_f];
            double avg_flux = 0.0;
            int n_samples = std::min(10, total_time_steps);
            for (int t = 0; t < n_samples; ++t) {
                size_t flux_idx = global_f * total_time_steps + t;
                double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;
                avg_flux += (1.0 - albedo) * direct + (1.0 - albedo) * reflected + (1.0 - ir_albedo) * ir;
            }
            avg_flux /= n_samples;

            double T_eq = std::sqrt(std::sqrt(avg_flux / (planet.emissivity * 5.67051196e-8)));
            if (T_eq < 40.0) T_eq = 100.0;
            if (T_eq > 400.0) T_eq = 400.0;

            for (size_t layer = 0; layer < n_layers; ++layer) {
                size_t idx = layer * batch_n + f;
                double depth_frac = z_grid[layer] / z_grid[n_layers - 1];
                T_initial[idx] = static_cast<float>(T_eq * (1.0 - depth_frac) + T_deep_val * depth_frac);
            }
        }
        solver.uploadInitialTemperature(T_initial);

        if (!loaded_initial_T.empty()) {
            std::vector<float> T_chunked(batch_n * n_layers);
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float* src = loaded_initial_T.data()
                                 + layer * n_faces + batch_start;
                float*       dst = T_chunked.data() + layer * batch_n;
                std::copy(src, src + batch_n, dst);
            }
            solver.uploadAllTemps(T_chunked);
        }

        if (GPU_DEBUG_OUTPUT && batch_idx == 0) {
            std::cerr << "[GPU Thermal DEBUG] Initial T profile for face 0: ";
            for (size_t layer = 0; layer < std::min(n_layers, (size_t)10); ++layer) {
                std::cerr << T_initial[layer * batch_n + 0] << " ";
            }
            std::cerr << "\n";
            std::cerr << "[GPU Thermal DEBUG] T_deep_val = " << T_deep_val << ", R350_val = " << R350_val << "\n";
            std::cerr << "[GPU Thermal DEBUG] dt = " << dt << " seconds\n";

            double T_repr = 200.0;
            double kc = kc_batch[0];
            double k_repr = kc * (1.0 + R350_val * T_repr * T_repr * T_repr);
            double cp_repr = planet.cp0;
            for (size_t c = 0; c < planet.cpCoeff.size(); ++c) {
                cp_repr += planet.cpCoeff[c] * std::pow(T_repr, c + 1);
            }
            double rho_repr = rho_batch[0];
            double kappa_repr = k_repr / (rho_repr * cp_repr);
            double dz_min = z_grid[1] - z_grid[0];
            double Fo = kappa_repr * dt / (dz_min * dz_min);
            std::cerr << "[GPU Thermal DEBUG] Stability check: k=" << k_repr << ", cp=" << cp_repr
                      << ", rho=" << rho_repr << ", kappa=" << kappa_repr
                      << ", dz_min=" << dz_min << ", Fo=" << Fo << "\n";
            if (Fo > 0.5) {
                std::cerr << "[GPU Thermal WARNING] Fourier number " << Fo << " > 0.5, solver may be unstable!\n";
            }
        }

        auto prepare_flux = [&](int t_start, int n_steps) {
            std::vector<float> flux(static_cast<size_t>(n_steps) * batch_n);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < n_steps; ++t) {
                for (size_t f = 0; f < batch_n; ++f) {
                    size_t global_f = batch_start + f;
                    size_t flux_idx = global_f * total_time_steps + (t_start + t);

                    double albedo = face_albedos[global_f];
                    double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                    double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                    double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;

                    double absorbed = (1.0 - albedo) * direct + (1.0 - albedo) * reflected + (1.0 - ir_albedo) * ir;
                    flux[t * batch_n + f] = static_cast<float>(absorbed);
                }
            }
            return flux;
        };

        int equil_cycles = 1;
        std::vector<float> output_temps;

        std::vector<float> batch_avg_E;
        std::vector<float>* batch_avg_E_ptr = nullptr;
        if (ice_acc) {
            batch_avg_E.assign(static_cast<size_t>(n_layers) * batch_n, 0.0f);
            batch_avg_E_ptr = &batch_avg_E;
        }

        if (two_phase) {
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Phase 1: " << equil_cycles
                          << " eq cycles × " << equilibration_steps
                          << " steps (dt=" << std::fixed << std::setprecision(1)
                          << eq_dt << "s)" << std::endl;
            }
            auto flux_eq = prepare_flux(0, equilibration_steps);
            std::vector<float> eq_output_discard;
            solver.solvePersistentMultipleCycles(flux_eq, eq_output_discard,
                                                 equilibration_steps, eq_dt, equil_cycles);

            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Phase 2: 1 output cycle × "
                          << output_steps << " steps (dt="
                          << out_dt << "s)" << std::endl;
            }
            auto flux_out = prepare_flux(equilibration_steps, output_steps);
            solver.solvePersistentMultipleCycles(flux_out, output_temps,
                                                 output_steps, out_dt, 0,
                                                 batch_avg_E_ptr,
                                                 /*accum_start*/ 0,
                                                 /*accum_end*/   output_steps);
        } else {
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Running " << equil_cycles
                          << " equilibration cycles + 1 output cycle..." << std::endl;
            }
            auto flux_all = prepare_flux(0, total_time_steps);
            solver.solvePersistentMultipleCycles(flux_all, output_temps,
                                                 total_time_steps, dt, equil_cycles,
                                                 batch_avg_E_ptr,
                                                 /*accum_start*/ equilibration_steps,
                                                 /*accum_end*/   total_time_steps);
        }

        if (ice_acc) {
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float* src = batch_avg_E.data() + layer * batch_n;
                float*       dst = ice_acc->sum_E.data() + layer * n_faces + batch_start;
                std::copy(src, src + batch_n, dst);
            }
        }

        if (dump_final_state) {
            std::vector<float> batch_T_final;
            try {
                solver.downloadAllTemps(batch_T_final);
            } catch (const std::exception& e) {
                std::cerr << "[GPU Thermal] WARNING: downloadAllTemps failed: "
                          << e.what() << "\n";
                batch_T_final.clear();
            }
            if (batch_T_final.size() == batch_n * n_layers) {
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const float* src = batch_T_final.data() + layer * batch_n;
                    float*       dst = assembled_final_T.data() + layer * n_faces + batch_start;
                    std::copy(src, src + batch_n, dst);
                }
            }
        }

        if (GPU_DEBUG_OUTPUT) {
            std::cerr << "[GPU Thermal DEBUG] output_temps size: " << output_temps.size()
                      << ", expected: " << (static_cast<size_t>(total_time_steps) * batch_n) << "\n";
            if (!output_temps.empty()) {
                std::cerr << "[GPU Thermal DEBUG] Host output_temps[0..4]: ";
                for (size_t i = 0; i < std::min(size_t(5), output_temps.size()); ++i) {
                    std::cerr << output_temps[i] << " ";
                }
                std::cerr << "\n";

                for (int check_t : {0, 10, 100, 500, 832}) {
                    size_t idx = check_t * batch_n;
                    if (idx < output_temps.size()) {
                        std::cerr << "[GPU Thermal DEBUG] output_temps at t=" << check_t << " (face 0): "
                                  << output_temps[idx] << " K\n";
                    }
                }
            }
        }

        if (surface_temps_out) {
            if (GPU_DEBUG_OUTPUT) {
                std::cerr << "[GPU Thermal DEBUG] Copying batch " << (batch_idx + 1)
                          << " results to surface_temps_out (batch_n=" << batch_n
                          << ", output_temps.size=" << output_temps.size() << ")\n";

                if (!output_temps.empty()) {
                    float sample_val = output_temps[0];
                    std::cerr << "[GPU Thermal DEBUG] output_temps[0] = " << sample_val << "\n";
                }
            }

            const int copy_n = two_phase ? (total_time_steps - equilibration_steps) : total_time_steps;
            const int copy_offset = two_phase ? equilibration_steps : 0;

            #pragma omp parallel for schedule(static)
            for (size_t f = 0; f < batch_n; ++f) {
                size_t global_f = batch_start + f;
                for (int t = 0; t < copy_n; ++t) {
                    (*surface_temps_out)[global_f][copy_offset + t] =
                        static_cast<double>(output_temps[t * batch_n + f]);
                }
            }

            if (GPU_DEBUG_OUTPUT) {
                std::cerr << "[GPU Thermal DEBUG] After copy: surface_temps_out[0][0] = "
                          << (*surface_temps_out)[batch_start][0] << "\n";
            }
        }

        return true;
    };

    // ----- Multi-GPU device-pool dispatch (opt-in) -----
    // gpu_devices == 1 (default) AND gpu_device_ids empty → fall through to
    // the single-device serial loop below (byte-identical to legacy code).
    // gpu_devices == 0 / "auto" → use every enumerated GPU.
    // gpu_device_ids non-empty → use exactly those device indices.
    const bool use_multi_gpu_thermal =
        (gpu_devices != 1) || !gpu_device_ids.empty();
    if (use_multi_gpu_thermal) {
        auto all_gpus = OpenCLContext::enumerateGPUs(/*only_available=*/true);
        std::vector<OpenCLDeviceInfo> selected;
        if (!gpu_device_ids.empty()) {
            for (int id : gpu_device_ids) {
                if (id < 0 || static_cast<size_t>(id) >= all_gpus.size()) {
                    std::cerr << "[GPU Thermal] Invalid gpu_device_id " << id
                              << " (only " << all_gpus.size() << " GPUs enumerated)"
                              << std::endl;
                    return false;
                }
                selected.push_back(all_gpus[id]);
            }
        } else if (gpu_devices == 0) {
            selected = all_gpus;
        } else {
            const int n = std::min(gpu_devices,
                                   static_cast<int>(all_gpus.size()));
            for (int i = 0; i < n; ++i) selected.push_back(all_gpus[i]);
        }
        if (selected.empty()) {
            std::cerr << "[GPU Thermal] No GPUs selected for multi-GPU dispatch"
                      << std::endl;
            return false;
        }

        std::cout << "[GPU Thermal] Multi-GPU dispatch across " << selected.size()
                  << " device(s):" << std::endl;
        for (size_t i = 0; i < selected.size(); ++i) {
            std::cout << "  [" << i << "] " << selected[i].name << " ("
                      << (selected[i].global_mem_size / (1024 * 1024)) << " MB)"
                      << std::endl;
        }
        if (gpu_inter_batch_pause_ms > 0) {
            std::cout << "[GPU Thermal] Inter-batch pause: "
                      << gpu_inter_batch_pause_ms << " ms (per device)" << std::endl;
        }

        ThermalGPUPoolConfig pool_cfg;
        pool_cfg.devices = selected;
        pool_cfg.inter_batch_pause_ms = gpu_inter_batch_pause_ms;
        pool_cfg.verbose = false;
        ThermalGPUDevicePool pool(std::move(pool_cfg));

        ThermalOpenCLConfig opencl_config;
        opencl_config.verbose = false;
        opencl_config.work_group_size = 64;

        std::vector<std::future<bool>> futures;
        futures.reserve(n_batches);
        for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
            const size_t batch_start = batch_idx * BATCH_SIZE;
            const size_t batch_end   = std::min(batch_start + BATCH_SIZE, n_faces);
            const size_t batch_n     = batch_end - batch_start;
            futures.push_back(pool.submit(batch_n, n_layers, opencl_config,
                [&run_batch_on_solver, batch_idx, batch_start, batch_n](
                    ThermalOpenCLSolver& solver, size_t /*device_idx*/) -> bool {
                    return run_batch_on_solver(solver, batch_idx, batch_start, batch_n);
                }));
        }

        bool all_ok = true;
        size_t completed = 0;
        for (auto& f : futures) {
            const bool ok = f.get();
            ++completed;
            std::cout << "\r[GPU Thermal Pool] " << completed << "/" << n_batches
                      << " batches done" << std::flush;
            all_ok = ok && all_ok;
        }
        std::cout << std::endl;

        if (!all_ok) {
            std::cerr << "[GPU Thermal] Multi-GPU dispatch had at least one batch failure"
                      << std::endl;
            return false;
        }

        // Skip the single-device serial loop — work is done.
        goto post_batch_loop;
    }

    for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
        size_t batch_start = batch_idx * BATCH_SIZE;
        size_t batch_end = std::min(batch_start + BATCH_SIZE, n_faces);
        size_t batch_n = batch_end - batch_start;

        // Progress bar with elapsed-time and ETA
        double progress = 100.0 * static_cast<double>(batch_start) / static_cast<double>(n_faces);
        std::cout << "\r[GPU Thermal] Progress: " << std::fixed << std::setprecision(1)
                  << progress << "% (batch " << (batch_idx + 1) << "/" << n_batches << ")";
        {
            double elapsed_s = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_total).count();
            auto fmt = [](double s) -> std::string {
                if (s < 0.0 || !std::isfinite(s)) return "?";
                if (s < 60.0)   { std::ostringstream o; o << std::fixed << std::setprecision(0) << s << "s"; return o.str(); }
                if (s < 3600.0) { int m = (int)(s/60.0); int r = (int)(s - m*60); std::ostringstream o; o << m << "m " << r << "s"; return o.str(); }
                int h = (int)(s/3600.0); int m = (int)((s - h*3600.0)/60.0);
                std::ostringstream o; o << h << "h" << m << "m"; return o.str();
            };
            std::cout << " | elapsed " << fmt(elapsed_s);
            if (batch_idx > 0) {
                double frac = static_cast<double>(batch_idx) / static_cast<double>(n_batches);
                if (frac > 0.01) {
                    double eta_s = elapsed_s * (1.0 - frac) / frac;
                    std::cout << ", ETA " << fmt(eta_s);
                }
            }
            std::cout << "       ";  // pad for shrinking ETA strings
        }
        std::cout << std::flush;

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        // Initialize solver for this batch
        ThermalOpenCLConfig opencl_config;
        opencl_config.verbose = false;
        opencl_config.work_group_size = 64;

        ThermalOpenCLSolver solver(batch_n, n_layers, opencl_config);
        if (!solver.initialize()) {
            std::cerr << "[GPU Thermal] Failed to initialize OpenCL solver" << std::endl;
            return false;
        }

        // Prepare mesh data for this batch (SoA layout: [layer * batch_n + face_in_batch])
        std::vector<float> z_grid_f(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            z_grid_f[i] = static_cast<float>(z_grid[i]);
        }

        // Material properties - use reference profile values for now
        // These are depth-dependent but uniform across faces (could be extended for spatially-varying)
        std::vector<float> rho_batch(batch_n * n_layers);
        std::vector<float> kc_batch(batch_n * n_layers);
        std::vector<float> emissivity_batch(batch_n);
        std::vector<float> T_deep_batch(batch_n);
        std::vector<float> Qb_batch(batch_n);

        // Get reference values from profile
        const auto& rho_ref = reference_profile.rho();
        const auto& kc_ref = reference_profile.kc();
        const auto& T_ref = reference_profile.T();
        double T_deep_val = T_ref.back();  // Deep temperature from initialized profile
        double R350_val = reference_profile.R350_val;

        // When a per-face H map is loaded, override rho(z) and k_c(z) per face
        // using each face's own H. Without a map, every face shares the
        // reference profile (legacy behavior). The depth grid (z_grid_f) stays
        // shared — same caveat as the CPU batch path: built for the body
        // default skin depth.
        const bool use_per_face_H = (face_H_overrides != nullptr) &&
                                    !face_H_overrides->empty();

        for (size_t f = 0; f < batch_n; ++f) {
            size_t global_f = batch_start + f;
            emissivity_batch[f] = static_cast<float>(planet.emissivity);
            T_deep_batch[f] = static_cast<float>(T_deep_val);
            // Use per-face Qb if mask is provided, otherwise planet default
            if (face_Qb && global_f < face_Qb->size()) {
                Qb_batch[f] = static_cast<float>((*face_Qb)[global_f]);
            } else {
                Qb_batch[f] = static_cast<float>(planet.Qb);
            }

            if (use_per_face_H) {
                const double H_face = (global_f < face_H_overrides->size())
                                    ? (*face_H_overrides)[global_f]
                                    : planet.H;
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    const double depth = z_grid[layer];
                    const double df    = std::exp(-depth / H_face);
                    rho_batch[idx] = static_cast<float>(planet.rhod + (planet.rhos - planet.rhod) * df);
                    kc_batch[idx]  = static_cast<float>(planet.kd  + (planet.ks  - planet.kd ) * df);
                }
            } else {
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    rho_batch[idx] = static_cast<float>(rho_ref[layer]);
                    kc_batch[idx]  = static_cast<float>(kc_ref[layer]);
                }
            }
        }

        solver.uploadMeshData(z_grid_f, rho_batch, kc_batch, emissivity_batch, T_deep_batch, Qb_batch);

        // Debug: print z grid
        if (GPU_DEBUG_OUTPUT && batch_idx == 0) {
            std::cerr << "[GPU Thermal DEBUG] z_grid (first 10): ";
            for (size_t i = 0; i < std::min(n_layers, (size_t)10); ++i) {
                std::cerr << z_grid_f[i] << " ";
            }
            std::cerr << "\n";
            std::cerr << "[GPU Thermal DEBUG] rho[0] (first face): " << rho_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] kc[0] (first face): " << kc_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] emissivity (first face): " << emissivity_batch[0] << "\n";
            std::cerr << "[GPU Thermal DEBUG] T_deep (first face): " << T_deep_batch[0] << "\n";
        }

        // Set material parameters
        std::vector<float> cp_coeffs_f;
        for (double c : planet.cpCoeff) {
            cp_coeffs_f.push_back(static_cast<float>(c));
        }
        solver.setMaterialParams(
            static_cast<float>(planet.cp0),
            cp_coeffs_f,
            static_cast<float>(R350_val),
            static_cast<float>(5.67051196e-8)  // Stefan-Boltzmann
        );

        // Initialize temperature (equilibrium temperature for all layers)
        std::vector<float> T_initial(batch_n * n_layers);
        for (size_t f = 0; f < batch_n; ++f) {
            // Get initial flux to estimate equilibrium temperature
            // Include ALL flux components (direct, reflected, IR) for accurate initialization
            size_t global_f = batch_start + f;
            double albedo = face_albedos[global_f];
            double avg_flux = 0.0;
            int n_samples = std::min(10, total_time_steps);
            for (int t = 0; t < n_samples; ++t) {
                size_t flux_idx = global_f * total_time_steps + t;
                double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;
                // Absorbed flux = (1-albedo)*direct + (1-albedo)*reflected + (1-ir_albedo)*ir
                avg_flux += (1.0 - albedo) * direct + (1.0 - albedo) * reflected + (1.0 - ir_albedo) * ir;
            }
            avg_flux /= n_samples;

            // Equilibrium temperature from Stefan-Boltzmann
            double T_eq = std::sqrt(std::sqrt(avg_flux / (planet.emissivity * 5.67051196e-8)));
            if (T_eq < 40.0) T_eq = 100.0;  // Minimum reasonable temperature
            if (T_eq > 400.0) T_eq = 400.0;

            for (size_t layer = 0; layer < n_layers; ++layer) {
                size_t idx = layer * batch_n + f;
                // Linear interpolation between surface and deep temperature
                double depth_frac = z_grid[layer] / z_grid[n_layers - 1];
                T_initial[idx] = static_cast<float>(T_eq * (1.0 - depth_frac) + T_deep_val * depth_frac);
            }
        }
        solver.uploadInitialTemperature(T_initial);

        // If a chunked-run initial-state file was loaded, overwrite the
        // (computed-from-flux) initial T with the saved per-face profile
        // from the previous chunk. Slice the global layer-major buffer
        // into this batch's local SoA layout: dst[layer*batch_n+f] =
        // src[layer*n_faces + batch_start + f].
        if (!loaded_initial_T.empty()) {
            std::vector<float> T_chunked(batch_n * n_layers);
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float* src = loaded_initial_T.data()
                                 + layer * n_faces + batch_start;
                float*       dst = T_chunked.data() + layer * batch_n;
                std::copy(src, src + batch_n, dst);
            }
            solver.uploadAllTemps(T_chunked);
        }

        // Debug: show initial temperature profile
        if (GPU_DEBUG_OUTPUT && batch_idx == 0) {
            std::cerr << "[GPU Thermal DEBUG] Initial T profile for face 0: ";
            for (size_t layer = 0; layer < std::min(n_layers, (size_t)10); ++layer) {
                std::cerr << T_initial[layer * batch_n + 0] << " ";
            }
            std::cerr << "\n";
            std::cerr << "[GPU Thermal DEBUG] T_deep_val = " << T_deep_val << ", R350_val = " << R350_val << "\n";
            std::cerr << "[GPU Thermal DEBUG] dt = " << dt << " seconds\n";

            // Check stability: Fourier number Fo = kappa * dt / dz^2 should be < 0.5
            // Use representative values
            double T_repr = 200.0;
            double kc = kc_batch[0];
            double k_repr = kc * (1.0 + R350_val * T_repr * T_repr * T_repr);
            double cp_repr = planet.cp0;
            for (size_t c = 0; c < planet.cpCoeff.size(); ++c) {
                cp_repr += planet.cpCoeff[c] * std::pow(T_repr, c + 1);
            }
            double rho_repr = rho_batch[0];
            double kappa_repr = k_repr / (rho_repr * cp_repr);
            double dz_min = z_grid[1] - z_grid[0];
            double Fo = kappa_repr * dt / (dz_min * dz_min);
            std::cerr << "[GPU Thermal DEBUG] Stability check: k=" << k_repr << ", cp=" << cp_repr
                      << ", rho=" << rho_repr << ", kappa=" << kappa_repr
                      << ", dz_min=" << dz_min << ", Fo=" << Fo << "\n";
            if (Fo > 0.5) {
                std::cerr << "[GPU Thermal WARNING] Fourier number " << Fo << " > 0.5, solver may be unstable!\n";
            }
        }

        // Helper: prepare flux data in timestep-major layout for a range of timesteps.
        auto prepare_flux = [&](int t_start, int n_steps) {
            std::vector<float> flux(static_cast<size_t>(n_steps) * batch_n);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < n_steps; ++t) {
                for (size_t f = 0; f < batch_n; ++f) {
                    size_t global_f = batch_start + f;
                    size_t flux_idx = global_f * total_time_steps + (t_start + t);

                    double albedo = face_albedos[global_f];
                    double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                    double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                    double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;

                    double absorbed = (1.0 - albedo) * direct + (1.0 - albedo) * reflected + (1.0 - ir_albedo) * ir;
                    flux[t * batch_n + f] = static_cast<float>(absorbed);
                }
            }
            return flux;
        };

        // Number of equilibration cycles (each replays the flux sequence).
        int equil_cycles = 1;
        std::vector<float> output_temps;

        // Per-batch Schorghofer accumulator buffer. Sized n_layers × batch_n.
        // The OpenCL solver fills this when ice_acc is enabled.
        std::vector<float> batch_avg_E;
        std::vector<float>* batch_avg_E_ptr = nullptr;
        if (ice_acc) {
            batch_avg_E.assign(static_cast<size_t>(n_layers) * batch_n, 0.0f);
            batch_avg_E_ptr = &batch_avg_E;
        }

        if (two_phase) {
            // ---- Two-phase GPU execution ----
            // Phase 1: equilibration with coarse dt (temperatures persist on GPU)
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Phase 1: " << equil_cycles
                          << " eq cycles × " << equilibration_steps
                          << " steps (dt=" << std::fixed << std::setprecision(1)
                          << eq_dt << "s)" << std::endl;
            }
            auto flux_eq = prepare_flux(0, equilibration_steps);
            std::vector<float> eq_output_discard;
            solver.solvePersistentMultipleCycles(flux_eq, eq_output_discard,
                                                 equilibration_steps, eq_dt, equil_cycles);

            // Phase 2: output with fine dt (1 cycle, download results).
            // Accumulator runs over the entire output cycle: [0, output_steps).
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Phase 2: 1 output cycle × "
                          << output_steps << " steps (dt="
                          << out_dt << "s)" << std::endl;
            }
            auto flux_out = prepare_flux(equilibration_steps, output_steps);
            solver.solvePersistentMultipleCycles(flux_out, output_temps,
                                                 output_steps, out_dt, 0,
                                                 batch_avg_E_ptr,
                                                 /*accum_start*/ 0,
                                                 /*accum_end*/   output_steps);
        } else {
            // ---- Single-phase (legacy) ----
            // Accumulator runs only on the FINAL cycle, and only for relative
            // timesteps in the output portion of the trajectory:
            //   [equilibration_steps, total_time_steps).
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Running " << equil_cycles
                          << " equilibration cycles + 1 output cycle..." << std::endl;
            }
            auto flux_all = prepare_flux(0, total_time_steps);
            solver.solvePersistentMultipleCycles(flux_all, output_temps,
                                                 total_time_steps, dt, equil_cycles,
                                                 batch_avg_E_ptr,
                                                 /*accum_start*/ equilibration_steps,
                                                 /*accum_end*/   total_time_steps);
        }

        // Copy this batch's accumulator into the global ice_acc buffer.
        if (ice_acc) {
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float* src = batch_avg_E.data() + layer * batch_n;
                float*       dst = ice_acc->sum_E.data() + layer * n_faces + batch_start;
                std::copy(src, src + batch_n, dst);
            }
        }

        // If a final-state checkpoint was requested, download this batch's
        // final T(z) and merge it into the whole-mesh layer-major buffer.
        if (dump_final_state) {
            std::vector<float> batch_T_final;
            try {
                solver.downloadAllTemps(batch_T_final);
            } catch (const std::exception& e) {
                std::cerr << "[GPU Thermal] WARNING: downloadAllTemps failed: "
                          << e.what() << "\n";
                batch_T_final.clear();
            }
            if (batch_T_final.size() == batch_n * n_layers) {
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const float* src = batch_T_final.data() + layer * batch_n;
                    float*       dst = assembled_final_T.data() + layer * n_faces + batch_start;
                    std::copy(src, src + batch_n, dst);
                }
            }
        }

        // Debug: verify output_temps matches kernel output
        if (GPU_DEBUG_OUTPUT) {
            std::cerr << "[GPU Thermal DEBUG] output_temps size: " << output_temps.size()
                      << ", expected: " << (static_cast<size_t>(total_time_steps) * batch_n) << "\n";
            if (!output_temps.empty()) {
                std::cerr << "[GPU Thermal DEBUG] Host output_temps[0..4]: ";
                for (size_t i = 0; i < std::min(size_t(5), output_temps.size()); ++i) {
                    std::cerr << output_temps[i] << " ";
                }
                std::cerr << "\n";

                // Check several timesteps to see temperature evolution
                for (int check_t : {0, 10, 100, 500, 832}) {
                    size_t idx = check_t * batch_n;
                    if (idx < output_temps.size()) {
                        std::cerr << "[GPU Thermal DEBUG] output_temps at t=" << check_t << " (face 0): "
                                  << output_temps[idx] << " K\n";
                    }
                }
            }
        }

        // Copy results to output (transpose from timestep-major to face-major)
        if (surface_temps_out) {
            if (GPU_DEBUG_OUTPUT) {
                std::cerr << "[GPU Thermal DEBUG] Copying batch " << (batch_idx + 1)
                          << " results to surface_temps_out (batch_n=" << batch_n
                          << ", output_temps.size=" << output_temps.size() << ")\n";

                // Debug: verify output_temps has valid data before copy
                if (!output_temps.empty()) {
                    float sample_val = output_temps[0];
                    std::cerr << "[GPU Thermal DEBUG] output_temps[0] = " << sample_val << "\n";
                }
            }

            // In two-phase mode, output_temps has output_steps entries and
            // surface_temps_out is also sized to output_steps (no eq data stored).
            // In single-phase, both have total_time_steps entries.
            // In two-phase mode, output_temps has output_steps entries.
            // Copy into the output portion of surface_temps (after eq slots).
            const int copy_n = two_phase ? (total_time_steps - equilibration_steps) : total_time_steps;
            const int copy_offset = two_phase ? equilibration_steps : 0;

            #pragma omp parallel for schedule(static)
            for (size_t f = 0; f < batch_n; ++f) {
                size_t global_f = batch_start + f;
                for (int t = 0; t < copy_n; ++t) {
                    (*surface_temps_out)[global_f][copy_offset + t] =
                        static_cast<double>(output_temps[t * batch_n + f]);
                }
            }

            if (GPU_DEBUG_OUTPUT) {
                // Debug: verify copy worked
                std::cerr << "[GPU Thermal DEBUG] After copy: surface_temps_out[0][0] = "
                          << (*surface_temps_out)[batch_start][0] << "\n";
            }
        }

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            batch_end_time - batch_start_time);
        (void)batch_duration;  // Suppress unused warning when not in verbose mode
    }

post_batch_loop:
    // Complete the progress bar
    std::cout << "\r[GPU Thermal] Progress: 100.0% (complete)                    " << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total);
    std::cout << "[GPU Thermal] Completed " << n_faces << " faces x " << total_time_steps
              << " timesteps in " << total_duration.count() << " seconds" << std::endl;

    // Finalize Schorghofer accumulator. n_samples is the number of output
    // timesteps over which the GPU kernel summed E(T(z,t)). For both two-phase
    // and single-phase modes, the output window is `output_steps` long.
    if (ice_acc) {
        ice_acc->n_samples = output_steps;
        std::cout << "[Ice stability] Accumulated <E(z)> over " << output_steps
                  << " output samples × " << n_layers << " depth nodes × "
                  << n_faces << " faces" << std::endl;
    }

    // Dump final thermal state if requested. PTMT magic + dims + raw float32.
    if (dump_final_state) {
        std::ofstream os(final_state_file, std::ios::binary);
        if (!os) {
            std::cerr << "[GPU Thermal] WARNING: cannot open final_state_file '"
                      << final_state_file << "' for write\n";
        } else {
            uint32_t magic = kThermalStateMagic;
            uint32_t nf = static_cast<uint32_t>(n_faces);
            uint32_t nl = static_cast<uint32_t>(n_layers);
            os.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(&nf),    sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(&nl),    sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(assembled_final_T.data()),
                     assembled_final_T.size() * sizeof(float));
            if (!os.good()) {
                std::cerr << "[GPU Thermal] WARNING: short write on final_state_file\n";
            } else {
                std::cout << "[GPU Thermal] Wrote final T(z) checkpoint: "
                          << final_state_file << " ("
                          << (assembled_final_T.size() * sizeof(float)) / (1024.0*1024.0)
                          << " MiB)\n";
            }
        }
    }

    return true;
}

/**
 * @brief GPU thermal solver for selected faces with pre-combined flux arrays
 *
 * Used in radiation-thermal coupling iterations where flux already includes
 * direct solar, reflected, IR, and indirect radiation components.
 *
 * @param mesh Mesh with latitudes
 * @param face_indices List of faces to process
 * @param flux Pre-combined absorbed flux per face [idx in face_indices -> timesteps]
 * @param times Time array
 * @param total_time_steps Number of timesteps
 * @param config Thermal configuration
 * @param surface_temps In/out: temperature results (updated for selected faces only)
 * @return true if successful
 */
bool runThermalForSelectedFaces_GPU(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<int>& face_indices,
    const std::vector<std::vector<double>>& flux,
    const std::vector<double>& times,
    int total_time_steps,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps)
{
    using namespace gpu::opencl;

    const size_t n_selected = face_indices.size();
    if (n_selected == 0) {
        return true;  // Nothing to do
    }

    // Get thermal model parameters
    thermal::Planet planet(config.getProperty("planet", "Moon"));
    thermal::Profile reference_profile(planet, 0.0, config, false);
    const auto& z_grid = reference_profile.z();
    const size_t n_layers = z_grid.size();

    // Calculate dt from times array
    float dt = 0.0f;
    if (total_time_steps > 1) {
        dt = static_cast<float>(times[1] - times[0]);
    }

    // GPU memory constraint
    const size_t bytes_per_face = n_layers * 50 + total_time_steps * 8 * 2;
    const size_t gpu_memory_limit = 6ULL * 1024 * 1024 * 1024;
    const size_t max_batch_size = std::min(n_selected, gpu_memory_limit / bytes_per_face);
    const size_t BATCH_SIZE = std::min(max_batch_size, static_cast<size_t>(50000));

    // Get reference material properties
    const auto& rho_ref = reference_profile.rho();
    const auto& kc_ref = reference_profile.kc();
    const auto& T_ref = reference_profile.T();
    double T_deep_val = T_ref.back();
    double R350_val = reference_profile.R350_val;

    // Process faces in batches
    size_t n_batches = (n_selected + BATCH_SIZE - 1) / BATCH_SIZE;

    for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
        size_t batch_start = batch_idx * BATCH_SIZE;
        size_t batch_end = std::min(batch_start + BATCH_SIZE, n_selected);
        size_t batch_n = batch_end - batch_start;

        // Initialize solver for this batch
        ThermalOpenCLConfig opencl_config;
        opencl_config.verbose = false;
        opencl_config.work_group_size = 64;

        ThermalOpenCLSolver solver(batch_n, n_layers, opencl_config);
        if (!solver.initialize()) {
            std::cerr << "[GPU Thermal] Failed to initialize OpenCL solver for selected faces" << std::endl;
            return false;
        }

        // Prepare mesh data in SoA layout: [layer * batch_n + face_in_batch]
        std::vector<float> z_grid_f(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            z_grid_f[i] = static_cast<float>(z_grid[i]);
        }

        std::vector<float> rho_batch(batch_n * n_layers);
        std::vector<float> kc_batch(batch_n * n_layers);
        std::vector<float> emissivity_batch(batch_n);
        std::vector<float> T_deep_batch(batch_n);
        std::vector<float> Qb_batch(batch_n);

        // Per-face H map (selected-faces variant): map slot f -> global face id
        // via face_indices[batch_start + f], then look up planet.face_H_overrides.
        const bool use_per_face_H = !planet.face_H_overrides.empty();

        for (size_t f = 0; f < batch_n; ++f) {
            emissivity_batch[f] = static_cast<float>(planet.emissivity);
            T_deep_batch[f] = static_cast<float>(T_deep_val);
            Qb_batch[f] = static_cast<float>(planet.Qb);

            if (use_per_face_H) {
                const int   global_f = face_indices[batch_start + f];
                const double H_face  = (global_f >= 0 &&
                                         static_cast<size_t>(global_f) < planet.face_H_overrides.size())
                                     ? planet.face_H_overrides[global_f]
                                     : planet.H;
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    const double depth = z_grid[layer];
                    const double df    = std::exp(-depth / H_face);
                    rho_batch[idx] = static_cast<float>(planet.rhod + (planet.rhos - planet.rhod) * df);
                    kc_batch[idx]  = static_cast<float>(planet.kd  + (planet.ks  - planet.kd ) * df);
                }
            } else {
                for (size_t layer = 0; layer < n_layers; ++layer) {
                    const size_t idx = layer * batch_n + f;
                    rho_batch[idx] = static_cast<float>(rho_ref[layer]);
                    kc_batch[idx]  = static_cast<float>(kc_ref[layer]);
                }
            }
        }

        solver.uploadMeshData(z_grid_f, rho_batch, kc_batch, emissivity_batch, T_deep_batch, Qb_batch);

        // Set material parameters
        std::vector<float> cp_coeffs_f;
        for (double c : planet.cpCoeff) {
            cp_coeffs_f.push_back(static_cast<float>(c));
        }
        solver.setMaterialParams(
            static_cast<float>(planet.cp0),
            cp_coeffs_f,
            static_cast<float>(R350_val),
            static_cast<float>(5.67051196e-8)  // Stefan-Boltzmann
        );

        // Initialize temperature from existing surface_temps (warm start)
        std::vector<float> T_initial(batch_n * n_layers);
        for (size_t f = 0; f < batch_n; ++f) {
            size_t idx = batch_start + f;
            int face_idx = face_indices[idx];

            // Use existing surface temperature for initial estimate
            double T_surface = (surface_temps[face_idx].empty()) ? T_deep_val : surface_temps[face_idx][0];
            if (T_surface < 40.0) T_surface = 100.0;
            if (T_surface > 400.0) T_surface = 400.0;

            for (size_t layer = 0; layer < n_layers; ++layer) {
                size_t tidx = layer * batch_n + f;
                // Linear interpolation between surface and deep temperature
                double depth_frac = z_grid[layer] / z_grid[n_layers - 1];
                T_initial[tidx] = static_cast<float>(T_surface * (1.0 - depth_frac) + T_deep_val * depth_frac);
            }
        }
        solver.uploadInitialTemperature(T_initial);

        // Prepare flux data in timestep-major layout: [timestep * batch_n + face_in_batch]
        std::vector<float> flux_all(static_cast<size_t>(total_time_steps) * batch_n);
        for (int t = 0; t < total_time_steps; ++t) {
            for (size_t f = 0; f < batch_n; ++f) {
                size_t idx = batch_start + f;
                // flux[idx] is the pre-combined flux for this face (indexed by position in face_indices)
                flux_all[t * batch_n + f] = static_cast<float>(flux[idx][t]);
            }
        }

        // Equilibration phase: run multiple cycles to establish correct subsurface
        // temperature profile before output. This matches the initial GPU thermal run
        // behavior (runThermalModelsForAllFaces_GPU_Persistent) which also runs 5 cycles.
        // Without this, the crude linear interpolation initial profile biases faces
        // with higher daytime temperatures to retain excess subsurface heat at night.
        int equil_cycles = 5;
        if (equil_cycles > 0 && total_time_steps > 1) {
            std::vector<float> equil_output_dummy;
            for (int eq_cycle = 0; eq_cycle < equil_cycles; ++eq_cycle) {
                solver.solvePersistentAllTimesteps(flux_all, equil_output_dummy, total_time_steps, dt);
            }
        }

        // Output phase (temperatures now equilibrated)
        std::vector<float> output_temps;
        solver.solvePersistentAllTimesteps(flux_all, output_temps, total_time_steps, dt);

        // Copy results back to surface_temps (transpose from timestep-major to face-major)
        for (size_t f = 0; f < batch_n; ++f) {
            size_t idx = batch_start + f;
            int face_idx = face_indices[idx];
            for (int t = 0; t < total_time_steps; ++t) {
                surface_temps[face_idx][t] = static_cast<double>(output_temps[t * batch_n + f]);
            }
        }
    }

    return true;
}

#else
// Stubs when OpenCL is not available
bool runThermalModelsForAllFaces_GPU_Persistent(
    const std::shared_ptr<Mesh>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    int,
    int,
    const thermal::Config&,
    double,
    double,
    std::vector<std::vector<double>>*,
    const std::vector<double>*)
{
    std::cerr << "[GPU Thermal] Error: OpenCL support not compiled in. "
              << "Rebuild with -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL" << std::endl;
    return false;
}

bool runThermalForSelectedFaces_GPU(
    const std::shared_ptr<Mesh>&,
    const std::vector<int>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int,
    const thermal::Config&,
    std::vector<std::vector<double>>&)
{
    std::cerr << "[GPU Thermal] Error: OpenCL support not compiled in. "
              << "Rebuild with -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL" << std::endl;
    return false;
}
#endif  // PTM_GPU_OPENCL

// Add checkpoint saving
void saveCheckpoint(
    const std::string& checkpoint_file,
    const std::vector<size_t>& completed_indices,
    const std::vector<std::vector<double>>& surface_temps) {
    
    std::ofstream out(checkpoint_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to save checkpoint" << std::endl;
        return;
    }
    
    // Write number of completed faces
    size_t count = completed_indices.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write completed face indices
    for (size_t idx : completed_indices) {
        out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        
        // Write temperature data for this face
        size_t temp_count = surface_temps[idx].size();
        out.write(reinterpret_cast<const char*>(&temp_count), sizeof(temp_count));
        
        out.write(reinterpret_cast<const char*>(surface_temps[idx].data()), 
                 temp_count * sizeof(double));
    }
    
    std::cout << "Checkpoint saved with " << count << " faces" << std::endl;
}

// ===== Helper implementations =====
static std::vector<std::tuple<std::string,double,double>> readLocationsCSV(const std::string& filename) {
    std::vector<std::tuple<std::string,double,double>> out;
    std::ifstream in(filename);
    if (!in.is_open()) return out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        // normalize delimiters
        for (char& c : line) { if (c=='\t') c = ','; }
        std::stringstream ss(line);
        std::string a,b,c;
        if (!std::getline(ss,a,',')) continue;
        if (!std::getline(ss,b,',')) {
            // maybe "lat lon"
            std::stringstream ws(line);
            double lat, lon;
            if (ws >> lat >> lon) out.emplace_back("", lat, lon);
            continue;
        }
        if (!std::getline(ss,c,',')) {
            // assume a=lat, b=lon
            try {
                double lat = std::stod(a);
                double lon = std::stod(b);
                out.emplace_back("", lat, lon);
            } catch (...) {}
            continue;
        }
        // Try format: name,lat,lon
        try {
            double lat = std::stod(b);
            double lon = std::stod(c);
            out.emplace_back(a, lat, lon);
            continue;
        } catch (...) {}
        // Try format: lat,lon,name
        try {
            double lat = std::stod(a);
            double lon = std::stod(b);
            out.emplace_back(c, lat, lon);
        } catch (...) {}
    }
    return out;
}

static std::vector<double> readDepthsFile(const std::string& filename) {
    std::vector<double> d;
    std::ifstream in(filename);
    if (!in.is_open()) return d;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        try {
            double z = std::stod(line);
            if (z >= 0.0) d.push_back(z);
        } catch (...) {}
    }
    return d;
}

static int findNearestFaceIndexByLatLon(const std::shared_ptr<Mesh>& mesh, double lat_deg, double lon_deg) {
    if (!mesh || mesh->faces.empty() || mesh->latitudes.size() != mesh->faces.size()) return -1;
    auto normLon = [](double L) {
        while (L >= 180.0) L -= 360.0;
        while (L <  -180.0) L += 360.0;
        return L;
    };
    lon_deg = normLon(lon_deg);
    double best = 1e300; int best_idx = -1;
    for (size_t i = 0; i < mesh->faces.size(); ++i) {
        double dlat = (mesh->latitudes[i] - lat_deg);
        double dlon = normLon(mesh->longitudes[i] - lon_deg);
        double dist2 = dlat*dlat + dlon*dlon;
        if (dist2 < best) { best = dist2; best_idx = static_cast<int>(i); }
    }
    return best_idx;
}

static std::vector<std::vector<double>> runThermalModelDepthSeries(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude_deg,
    const thermal::Config& config,
    const std::vector<double>& depths_m,
    int equilibration_steps,
    const thermal::Planet& planet)
{
    // Build flux array [2 x N]: times, flux
    Eigen::MatrixXd flux_arr(2, static_cast<int>(times.size()));
    for (int i = 0; i < static_cast<int>(times.size()); ++i) {
        flux_arr(0, i) = times[i];
        flux_arr(1, i) = (i < static_cast<int>(flux_values.size())) ? flux_values[i] : 0.0;
    }
    // Create and run model
    thermal::Model mdl(
        planet,
        latitude_deg * M_PI / 180.0,
        times,
        flux_arr,
        config,
        /*show_progress=*/false
    );
    if (equilibration_steps > 0) {
        mdl.setEquilibrationSteps(equilibration_steps);
    }
    mdl.run();

    const auto& z_layers = mdl.profile.getGrid().z;
    const int Nz = static_cast<int>(z_layers.size());
    const int Nt = static_cast<int>(times.size());
    const Eigen::MatrixXd& Tmat = mdl.T;
    const bool rows_are_layers = (Tmat.rows() == Nz && Tmat.cols() == Nt);
    const bool cols_are_layers = (Tmat.cols() == Nz && Tmat.rows() == Nt);

    std::vector<std::vector<double>> T_depths;
    T_depths.resize(depths_m.size(), std::vector<double>(Nt, 0.0));

    if (!rows_are_layers && !cols_are_layers) {
        // Unexpected shape; fallback to surface only
        for (size_t k = 0; k < depths_m.size(); ++k) {
            for (int t = 0; t < Nt; ++t) {
                double Ts = rows_are_layers ? Tmat(0, t) : (cols_are_layers ? Tmat(t, 0) : 0.0);
                T_depths[k][t] = Ts;
            }
        }
        return T_depths;
    }

    // For each depth, interpolate between adjacent layers
    for (size_t k = 0; k < depths_m.size(); ++k) {
        const double zq = depths_m[k];
        if (zq <= z_layers.front()) {
            for (int t = 0; t < Nt; ++t) T_depths[k][t] = rows_are_layers ? Tmat(0, t) : Tmat(t, 0);
            continue;
        }
        if (zq >= z_layers.back()) {
            // Extrapolate using Fourier's law: dT/dz = Qb / k
            // T(z) = T_bottom + Qb * (z - z_bottom) / k_bottom
            int j = Nz - 1;
            const double z_bottom = z_layers.back();
            const double Qb = planet.Qb;  // Interior heat flux (W/m²)
            const auto& k_vec = mdl.profile.getGrid().k;
            const double k_bottom = (k_vec.size() > 0) ? k_vec.back() : planet.kd;
            const double dz_extra = zq - z_bottom;
            const double dT_extrapolate = (k_bottom > 0.0) ? (Qb * dz_extra / k_bottom) : 0.0;

            for (int t = 0; t < Nt; ++t) {
                double T_bottom = rows_are_layers ? Tmat(j, t) : Tmat(t, j);
                T_depths[k][t] = T_bottom + dT_extrapolate;
            }
            continue;
        }
        int j = 0;
        while (j + 1 < Nz && z_layers[j + 1] < zq) ++j;
        int jp = std::min(j + 1, Nz - 1);
        const double tfrac = (z_layers[jp] > z_layers[j]) ? ((zq - z_layers[j]) / (z_layers[jp] - z_layers[j])) : 0.0;
        for (int t = 0; t < Nt; ++t) {
            const double Tj  = rows_are_layers ? Tmat(j,  t) : Tmat(t, j);
            const double Tjp = rows_are_layers ? Tmat(jp, t) : Tmat(t, jp);
            T_depths[k][t] = Tj + (Tjp - Tj) * tfrac;
        }
    }
    return T_depths;
}

static bool writeLocationDepthCSV(const std::string& out_path,
                           const std::string& location_name,
                           const std::vector<std::string>& time_strings,
                           const std::vector<double>& depths_m,
                           const std::vector<std::vector<double>>& T_depths)
{
    if (depths_m.size() != T_depths.size()) return false;
    std::ofstream out(out_path);
    if (!out.is_open()) return false;

    out << "# Location: " << (location_name.empty() ? "unnamed" : location_name) << "\n";
    out << "time";
    out << std::fixed;
    for (double z : depths_m) out << ",T(z=" << std::setprecision(3) << z << " m)";
    out << "\n";

    const size_t Nt = time_strings.size();
    for (size_t t = 0; t < Nt; ++t) {
        out << time_strings[t];
        for (size_t k = 0; k < depths_m.size(); ++k) {
            double Tv = (t < T_depths[k].size()) ? T_depths[k][t] : 0.0;
            out << "," << std::setprecision(6) << Tv;
        }
        out << "\n";
    }
    out.flush();
    return true;
}

// Helper: generate depth profile plots using Python script
static bool generateDepthProfilePlots(const std::string& csv_dir, const std::string& executable_path) {
    // Find the Python script relative to the executable
    // Expected location: <project_root>/src/python/plot_depth_profiles.py
    // The executable is typically in <project_root>/build/bin/

    std::filesystem::path exe_path(executable_path);
    std::filesystem::path project_root;

    // Try to find project root by looking for src/python directory
    std::filesystem::path search_path = exe_path.parent_path();
    for (int i = 0; i < 5; ++i) {  // Search up to 5 levels
        if (std::filesystem::exists(search_path / "src" / "python" / "plot_depth_profiles.py")) {
            project_root = search_path;
            break;
        }
        search_path = search_path.parent_path();
        if (search_path.empty() || search_path == search_path.root_path()) break;
    }

    if (project_root.empty()) {
        std::cerr << "Warning: Could not find plot_depth_profiles.py script" << std::endl;
        return false;
    }

    std::filesystem::path script_path = project_root / "src" / "python" / "plot_depth_profiles.py";

    // Determine which Python to use
    // Priority: 1) PYTHON_EXECUTABLE env var (set by conda/venv)
    //           2) python3 if available
    //           3) python as fallback
    std::string python_cmd;
    const char* python_env = std::getenv("PYTHON_EXECUTABLE");
    if (python_env && std::strlen(python_env) > 0) {
        // Use environment-specified Python (respects conda/venv activation)
        python_cmd = python_env;
    } else {
        // Fallback: check for python3, then python
        int python_check = std::system("python3 --version > /dev/null 2>&1");
        python_cmd = (python_check == 0) ? "python3" : "python";
    }

    // Build command
    std::ostringstream cmd;
    cmd << python_cmd << " \"" << script_path.string() << "\" "
        << "--directory \"" << csv_dir << "\" "
        << "--format png --dpi 150 --style paper";

    std::cout << "Generating depth profile plots..." << std::endl;

    int result = std::system(cmd.str().c_str());
    if (result != 0) {
        std::cerr << "Warning: Plot generation failed (exit code " << result << ")" << std::endl;
        std::cerr << "  You can manually run: " << cmd.str() << std::endl;
        return false;
    }

    return true;
}

// Helper: compute per-face visible albedos once (uses map if available)
std::vector<double> computeFaceAlbedos(
    const std::shared_ptr<Mesh>& mesh,
    std::shared_ptr<ptm::SpatialMap> albedo_map,
    double default_visible_albedo)
{
    const size_t num_faces = mesh->faces.size();
    std::vector<double> face_albedos(num_faces, default_visible_albedo);

    if (albedo_map && albedo_map->isValid()
        && mesh->latitudes.size() == num_faces
        && mesh->longitudes.size() == num_faces)
    {
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < num_faces; ++f) {
            double lon = mesh->longitudes[f];
            while (lon < 0) lon += 360.0;
            while (lon >= 360.0) lon -= 360.0;
            double lat = mesh->latitudes[f];
            double v = albedo_map->getValue(lon, lat);
            if (v == albedo_map->getMissingValue() || !std::isfinite(v) || v <= 0.0 || v >= 1.0) {
                v = default_visible_albedo;
            }
            face_albedos[f] = v;
        }
    }
    return face_albedos;
}

// Helper function to adjust start time
std::string calculateAdjustedStartTime(const std::string& iso_time, double days_offset) {
    // Parse the ISO time string to get seconds since J2000
    double et;
    str2et_c(iso_time.c_str(), &et);
    
    // Apply the offset in seconds
    double adjusted_et = et + days_offset * 86400.0;
    
    // Convert back to ISO format
    char time_string[51];
    et2utc_c(adjusted_et, "ISOC", 0, 50, time_string);
    
    return std::string(time_string);
}

bool exportLocationIlluminationAndTemperature(
    const std::string& output_file,
    const std::string& locations_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<std::string>& time_strings,
    int time_steps,
    const std::vector<std::vector<double>>& indirect_flux = {}) {
    // Basic validations
    if (!mesh) {
        std::cerr << "exportLocationIlluminationAndTemperature: mesh is null" << std::endl;
        return false;
    }
    const size_t F = mesh->faces.size();
    if (F == 0) {
        std::cerr << "exportLocationIlluminationAndTemperature: mesh has no faces" << std::endl;
        return false;
    }
    if (time_steps <= 0) {
        std::cerr << "exportLocationIlluminationAndTemperature: invalid time_steps" << std::endl;
        return false;
    }

    // Read locations first via CSV; fallback to whitespace format
    std::vector<std::string> location_names;
    std::vector<std::pair<double,double>> locations;
    std::cout << "Attempting to open locations file: " << locations_file << std::endl;

    auto loc_csv = readLocationsCSV(locations_file);
    if (!loc_csv.empty()) {
        for (const auto& tup : loc_csv) {
            location_names.push_back(std::get<0>(tup));
            locations.emplace_back(std::get<1>(tup), std::get<2>(tup));
        }
    } else {
        std::ifstream input(locations_file);
        if (!input.is_open()) {
            std::cerr << "Failed to open locations file: " << locations_file << std::endl;
            return false;
        }
        std::cout << "Successfully opened locations file" << std::endl;

        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            if (line[start] == '#') continue;

            std::istringstream iss(line);
            double lat = 0.0, lon = 0.0;
            if (!(iss >> lat >> lon)) {
                // Try CSV: supports both name,lat,lon and lat,lon,name
                std::string token;
                std::vector<std::string> parts;
                std::istringstream iss2(line);
                while (std::getline(iss2, token, ',')) {
                    size_t a = token.find_first_not_of(" \t\r\n");
                    size_t b = token.find_last_not_of(" \t\r\n");
                    parts.push_back((a == std::string::npos) ? "" : token.substr(a, b - a + 1));
                }
                if (parts.size() >= 3) {
                    // Try format: name,lat,lon
                    try {
                        const std::string name = parts[0];
                        lat = std::stod(parts[1]);
                        lon = std::stod(parts[2]);
                        locations.emplace_back(lat, lon);
                        location_names.push_back(name);
                        continue;
                    } catch (...) {
                        // Try format: lat,lon,name
                        try {
                            lat = std::stod(parts[0]);
                            lon = std::stod(parts[1]);
                            const std::string name = parts[2];
                            locations.emplace_back(lat, lon);
                            location_names.push_back(name);
                            continue;
                        } catch (...) {
                            std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                            continue;
                        }
                    }
                } else if (parts.size() == 2) {
                    // Try format: lat,lon (no name)
                    try {
                        lat = std::stod(parts[0]);
                        lon = std::stod(parts[1]);
                        std::ostringstream name_ss;
                        name_ss << "Location_" << std::fixed << std::setprecision(2) << lat << "_" << lon;
                        locations.emplace_back(lat, lon);
                        location_names.push_back(name_ss.str());
                        continue;
                    } catch (...) {
                        std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                        continue;
                    }
                }
                std::cerr << "[ERROR] Error parsing location: " << line << std::endl;
                continue;
            }
            std::string name;
            std::getline(iss >> std::ws, name);
            if (name.empty()) {
                std::ostringstream name_ss;
                name_ss << "Location_" << std::fixed << std::setprecision(6) << lat << "_" << lon;
                name = name_ss.str();
            }
            locations.emplace_back(lat, lon);
            location_names.push_back(name);
        }
    }

    if (locations.empty()) {
        std::cerr << "[ERROR] No valid locations found in file" << std::endl;
        return false;
    }

    // Map locations
    // Location CSV uses East longitude (IAU convention), same as mesh (atan2)
    std::vector<size_t> facet_indices;
    facet_indices.reserve(locations.size());
    for (size_t i = 0; i < locations.size(); ++i) {
        const auto& ll = locations[i];
        int face_idx = findNearestFaceIndexByLatLon(mesh, ll.first, ll.second);
        if (face_idx < 0 || static_cast<size_t>(face_idx) >= F) {
            std::cerr << "Warning: Could not map location '" << location_names[i]
                      << "' (" << std::fixed << std::setprecision(9) << ll.first
                      << ", " << ll.second << ") to a mesh face; skipping." << std::endl;
            continue;
        }
        facet_indices.push_back(static_cast<size_t>(face_idx));
        std::cout << "Location " << location_names[i] << " (" << std::fixed << std::setprecision(9)
                  << ll.first << ", " << ll.second << ") mapped to facet " << face_idx << " at ("
                  << std::fixed << std::setprecision(12)
                  << mesh->latitudes[static_cast<size_t>(face_idx)] << ", "
                  << mesh->longitudes[static_cast<size_t>(face_idx)] << ")" << std::endl;
    }

    if (facet_indices.empty()) {
        std::cerr << "No locations could be mapped to faces" << std::endl;
        return false;
    }

    // Open output
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    // Header
    const bool have_indirect = !indirect_flux.empty();
    output << "Time";
    for (const auto& name : location_names) {
        output << "," << name << "_Total";
        output << "," << name << "_DirectSolar";
        output << "," << name << "_PrimaryReflected";
        output << "," << name << "_PrimaryIR";
        if (have_indirect) output << "," << name << "_IndirectFlux";
        output << "," << name << "_Ts";
    }
    output << "\n";

    auto get_val = [&](const std::vector<double>& v, size_t f, int t) -> double {
        if (v.empty()) return 0.0;
        const size_t stride = static_cast<size_t>(time_steps);
        const size_t idx = f * stride + static_cast<size_t>(t);
        return (idx < v.size()) ? v[idx] : 0.0;
    };

    // Rows
    for (int t = 0; t < time_steps; ++t) {
        const std::string ts = (static_cast<size_t>(t) < time_strings.size())
                               ? time_strings[static_cast<size_t>(t)]
                               : std::to_string(t);
        output << ts;

        for (size_t i = 0; i < facet_indices.size(); ++i) {
            const size_t f = facet_indices[i];
            if (f >= F) {
                output << ",0,0,0,0";
                if (have_indirect) output << ",0";
                output << ",0";
                continue;
            }
            const double tot = get_val(illumination_values, f, t);
            const double dir = get_val(direct_solar_values, f, t);
            const double rfl = get_val(primary_reflected_values, f, t);
            const double pir = get_val(primary_ir_values, f, t);

            double ind = 0.0;
            if (have_indirect && f < indirect_flux.size()) {
                const auto& irow = indirect_flux[f];
                if (static_cast<size_t>(t) < irow.size()) {
                    ind = irow[static_cast<size_t>(t)];
                }
            }

            double Ts = 0.0;
            if (!surface_temps.empty() && f < surface_temps.size()) {
                const auto& row = surface_temps[f];
                if (static_cast<size_t>(t) < row.size()) {
                    Ts = row[static_cast<size_t>(t)];
                }
            }
            output << "," << tot << "," << dir << "," << rfl << "," << pir;
            if (have_indirect) output << "," << ind;
            output << "," << Ts;
        }
        output << "\n";
    }

    std::cout << "Location illumination and temperature data written to " << output_file << std::endl;
    return true;
}

// Run the thermal model for a face with matched times
std::vector<double> runThermalModelWithMatchedTimes(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps,
    double Qb_override,
    double H_override);

// Write a complete VTK file with both illumination and thermal data
bool writeCompleteVTK(
    const std::string& output_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    const std::string& source_mesh_file,
    bool include_primary_ir = false) {
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_file << std::endl;
        return false;
    }
    
    // Write VTK header with custom mesh info
    std::string vtk_header = "Planetary Thermal Simulation";
    if (!source_mesh_file.empty()) {
        vtk_header += " - Using custom mesh: " + 
            std::filesystem::path(source_mesh_file).filename().string();
    }
    
    file << "# vtk DataFile Version 3.0\n";
    file << vtk_header << "\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Write vertices
    file << "POINTS " << mesh->vertices.size() << " double\n";
    file << std::setprecision(17);
    for (const auto& v : mesh->vertices) {
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    file << std::setprecision(6);

    // Write cells (faces)
    file << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
    for (const auto& f : mesh->faces) {
        file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }

    // Write cell types (all triangles)
    file << "CELL_TYPES " << mesh->faces.size() << "\n";
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        file << "5\n"; // VTK_TRIANGLE
    }

    // Write FIELD data for time values
    file << "FIELD FieldData 2\n";

    // Write time values
    file << "TIME_STEPS " << 1 << " " << time_steps << " float\n";
    for (int t = 0; t < time_steps; t++) {
        file << times[t] << "\n"; 
    }
    
    // Write time strings
    file << "TIME_STRINGS " << 1 << " " << time_steps << " string\n";
    for (int t = 0; t < time_steps; t++) {
        file << time_strings[t] << "\n";
    }
    
    // Write cell data
    file << "CELL_DATA " << mesh->faces.size() << "\n";
    
    // Add latitude/longitude data first (same as illumination file)
    file << "SCALARS latitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->latitudes[f] << "\n";
    }
    
    file << "SCALARS longitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->longitudes[f] << "\n";
    }

    // Write illumination data for each time step
    for (int t = 0; t < time_steps; t++) {
        // Total illumination
        file << "SCALARS illumination_" << t << " float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            size_t idx = f * time_steps + t;
            if (idx < illumination_values.size()) {
                file << illumination_values[idx] << "\n";
            } else {
                file << "0.0\n";
            }
        }

        // Only write primary IR data if explicitly requested
        if (include_primary_ir) {
            file << "SCALARS primary_ir_" << t << " float 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                size_t idx = f * time_steps + t;
                if (idx < primary_ir_values.size()) {
                    file << primary_ir_values[idx] << "\n";
                } else {
                    file << "0.0\n";
                }
            }
        }
        
        // Surface temperatures (if available)
        if (!surface_temps.empty() && t < time_steps) {
            file << "SCALARS temperature_" << t << " float 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t f = 0; f < mesh->faces.size(); f++) {
                if (f < surface_temps.size() && t < surface_temps[f].size()) {
                    file << surface_temps[f][t] << "\n";
                } else {
                    file << "0.0\n";
                }
            }
        }
    }
    
    file.close();
    std::cout << "Complete VTK file with " << time_steps << " time steps written to: " << output_file << std::endl;
    return true;
}

// Write temperature data to a separate VTK file in the same format as illumination data
bool writeThermalVTK(
    const std::string& output_file,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    bool include_equilibration = false) {

    // Determine actual number of time steps to write
    int start_idx = include_equilibration ? 0 : (times.size() - time_steps);
    int output_time_steps = include_equilibration ? times.size() : time_steps;
    
    // Status message - no verbose check, always print
    std::cout << "Generating thermal VTK file: " << output_file << std::endl;
    std::cout << "Writing " << output_time_steps << " time steps" 
              << (include_equilibration ? " (including equilibration)" : " (excluding equilibration)") << std::endl;
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open thermal output file: " << output_file << std::endl;
        return false;
    }
    
    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Planetary Thermal Simulation\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Write vertices
    file << "POINTS " << mesh->vertices.size() << " double\n";
    file << std::setprecision(17);
    for (const auto& v : mesh->vertices) {
        file << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    file << std::setprecision(6);

    // Write faces
    file << "CELLS " << mesh->faces.size() << " " << mesh->faces.size() * 4 << "\n";
    for (const auto& f : mesh->faces) {
        file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    
    // Cell types (5 = VTK_TRIANGLE)
    file << "CELL_TYPES " << mesh->faces.size() << "\n";
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        file << "5\n";
    }
    
    // Write FIELD data for time values
    file << "FIELD FieldData 2\n";
    
    // Write time values - use output_time_steps and start_idx for correct selection
    file << "TIME_STEPS " << 1 << " " << output_time_steps << " float\n";
    for (int t = 0; t < output_time_steps; t++) {
        file << times[start_idx + t] << "\n"; 
    }
    
    // Write time strings - use output_time_steps and start_idx for correct selection
    file << "TIME_STRINGS " << 1 << " " << output_time_steps << " string\n";
    for (int t = 0; t < output_time_steps; t++) {
        file << time_strings[start_idx + t] << "\n";
    }
    
    // Write cell data
    file << "CELL_DATA " << mesh->faces.size() << "\n";
    
    // Add latitude/longitude data first (same as illumination file)
    file << "SCALARS latitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->latitudes[f] << "\n";
    }
    
    file << "SCALARS longitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t f = 0; f < mesh->faces.size(); f++) {
        file << mesh->longitudes[f] << "\n";
    }
    
    // For detection of temperature range
    double min_temp = std::numeric_limits<double>::max();
    double max_temp = std::numeric_limits<double>::lowest();
    int valid_temp_count = 0;
    int nan_inf_count = 0;
    
    // For each time step, write a temperature array - use output_time_steps and start_idx
    for (int t = 0; t < output_time_steps; t++) {
        // Calculate the actual time index using start_idx
        int time_idx = start_idx + t;
        
        // Temperature for this time step
        file << "SCALARS temperature_" << t << " float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t f = 0; f < mesh->faces.size(); f++) {
            // Use -1.0 as marker for invalid data
            double temp = -1.0;  
            
            if (f < surface_temps.size() && time_idx < surface_temps[f].size()) {
                temp = surface_temps[f][time_idx];
                
                // Only filter out non-numeric values
                if (std::isnan(temp) || std::isinf(temp)) {
                    temp = -1.0;
                    nan_inf_count++;
                } else {
                    // Track temperature statistics for valid temperatures only
                    min_temp = std::min(min_temp, temp);
                    max_temp = std::max(max_temp, temp);
                    valid_temp_count++;
                }
            }
            
            // Write temperature value
            file << std::fixed << std::setprecision(6) << temp << "\n";
        }
    }
    
    file.close();
    
    // Print diagnostic information about temperature data
    if (valid_temp_count > 0) {
        std::cout << "Temperature data statistics:" << std::endl;
        std::cout << "  Valid values: " << valid_temp_count << std::endl;
        std::cout << "  NaN/Inf values replaced: " << nan_inf_count << std::endl;
        std::cout << "  Temperature range: " << min_temp << "K to " << max_temp << "K" << std::endl;
    } else {
        std::cout << "WARNING: No valid temperature values were written to VTK!" << std::endl;
    }
    
    std::cout << "Temperature data written to: " << output_file << std::endl;
    std::cout << "Thermal VTK file generated successfully" << std::endl;
    return true;
}

// Validate the temperature matrix dimensions
static bool validateTemperatureMatrix(const std::vector<std::vector<double>>& temps,
                                      size_t expected_faces,
                                      int expected_steps)
{
    if (temps.size() != expected_faces) {
        std::cerr << "[Error] surface_temps rows (" << temps.size()
                  << ") != faces (" << expected_faces << ")\n";
        return false;
    }
    for (size_t f = 0; f < temps.size(); ++f) {
        if ((int)temps[f].size() != expected_steps) {
            std::cerr << "[Error] surface_temps[" << f << "] length ("
                      << temps[f].size() << ") != expected steps ("
                      << expected_steps << ")\n";
            return false;
        }
    }
    return true;
}

// Fix unreasonable albedo values
void fixOutOfRangeAlbedos(
    std::vector<double>& face_albedos,
    const std::shared_ptr<Mesh>& mesh,
    double default_visible_albedo) {  // This should be planet.albedo
    
    const size_t num_faces = face_albedos.size();
    if (num_faces == 0) return;
    
    // Define reasonable bounds for albedo
    const double min_reasonable_albedo = MIN_VALID_ALBEDO;
    const double max_reasonable_albedo = MAX_VALID_ALBEDO;
    
    // Identify faces with unreasonable albedo values
    std::vector<size_t> problematic_faces;
    for (size_t f = 0; f < num_faces; f++) {
        if (face_albedos[f] < min_reasonable_albedo || 
            face_albedos[f] > max_reasonable_albedo) {
            problematic_faces.push_back(f);
        }
    }
    
    if (problematic_faces.empty()) {
        std::cout << "No unreasonable albedo values found." << std::endl;
        return;
    }
    
    std::cout << "Found " << problematic_faces.size() << " faces with unreasonable albedo values ("
              << (100.0 * problematic_faces.size() / num_faces) << "%)." << std::endl;
    
    // Fix by spatial interpolation from nearby valid faces
    for (size_t idx : problematic_faces) {
        // For now, just use the default albedo
        // In future, could implement spatial interpolation from neighbors
        face_albedos[idx] = default_visible_albedo;
    }
    
    std::cout << "Fixed albedo values using planet default: " << default_visible_albedo << std::endl;
}

// Add this function to your declarations section
bool validateMesh(const std::shared_ptr<ptm::Mesh>& mesh);

// Add the implementation at an appropriate location
bool validateMesh(const std::shared_ptr<ptm::Mesh>& mesh) {
    if (mesh->vertices.empty() || mesh->faces.empty()) {
        std::cerr << "Error: Mesh has no vertices or faces" << std::endl;
        return false;
    }
    
    // Check if face normals were calculated
    if (mesh->face_normals.size() != mesh->faces.size()) {
        std::cerr << "Error: Face normals were not properly calculated" << std::endl;
        return false;
    }
    
    // Check if lat/lon mappings were calculated
    if (mesh->latitudes.size() != mesh->faces.size() || 
        mesh->longitudes.size() != mesh->faces.size()) {
        std::cerr << "Error: Latitude/longitude mappings were not properly calculated" << std::endl;
        return false;
    }
    
    // Mesh appears valid
    return true;
}

// Display usage information
void showUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Configuration File Options (NEW!)\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --config FILE              Load parameters from YAML config file\n";
    std::cout << "  --generic-body-config FILE Load generic body config from YAML\n";
    std::cout << "  --generate-template [FILE] Generate template config file\n";
    std::cout << "                             (default: ptm_config_template.yaml)\n";
    std::cout << "  --help                     Show this help message\n\n";
    
    std::cout << "Note: Command-line arguments OVERRIDE config file values.\n";
    std::cout << "Example usage:\n";
    std::cout << "  " << programName << " --config my_sim.yaml\n";
    std::cout << "  " << programName << " --config my_sim.yaml --duration 7.1 --timesteps 144\n";
    std::cout << "  " << programName << " --generate-template europa_template.yaml\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Time and Simulation Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --body, -b <name>          Set planetary body (EUROPA, MOON, MERCURY, etc.)\n";
    std::cout << "  --start TIME               Start time in ISO format (default: 2025-01-01T00:00:00)\n";
    std::cout << "  --duration DAYS            Simulation duration in days (default: 3.55)\n";
    std::cout << "  --timesteps N              Number of time steps (default: 72)\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Mesh Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --resolution <n>           Set mesh resolution for global models (default: 5)\n";
    std::cout << "  --mesh-file <file>         Import triangulated mesh from file (OBJ, PLY, VTK)\n";
    std::cout << "  --mesh-scale <factor>      Scale factor for imported mesh (default: 1.0)\n";
    std::cout << "  --export-basic-mesh        Export basic mesh to VTK format\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Thermal Model Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --thermal-mesh             Run thermal model for all mesh faces\n";
    std::cout << "  --thermal-threads N        Number of parallel threads (default: auto)\n";
    std::cout << "  --use-simd-batch           Use SIMD batch solver (default on; flag kept for compatibility)\n";
    std::cout << "  --no-simd-batch            Force legacy Model::run scalar thermal path\n";
    std::cout << "  --use-gpu-thermal          Use GPU persistent kernel solver (OpenCL)\n";
    std::cout << "  --list-gpu-devices         Print enumerated OpenCL GPUs and exit\n";
    std::cout << "  --gpu-devices N|auto       Number of GPUs for the thermal pool (default: 1).\n";
    std::cout << "                             'auto'/'all' = use every enumerated GPU.\n";
    std::cout << "  --gpu-device-ids 0,1,...   Explicit GPU indices (overrides --gpu-devices).\n";
    std::cout << "                             Indices come from --list-gpu-devices.\n";
    std::cout << "  --gpu-batch-size N         Per-device max faces/batch (default: 0 = legacy 6 GB cap).\n";
    std::cout << "                             Smaller batches reduce macOS+AMD watchdog hang risk.\n";
    std::cout << "  --gpu-inter-batch-pause-ms N  Sleep ms between batches on the same device (default: 0).\n";
    std::cout << "  --use-gpu-illumination     Use GPU BVH ray tracer for shadow rays (default: CPU Embree)\n";
    std::cout << "  --illum-gpu-devices N|auto Number of GPUs for the illumination pool (default: 1)\n";
    std::cout << "  --illum-gpu-device-ids 0,1,...  Explicit illumination GPU indices\n";
    std::cout << "  --illum-gpu-inter-batch-pause-ms N  Per-device sleep between illumination batches\n";
    std::cout << "  --bvh-cache-file FILE      Disk cache for the BVH (default: <output>/bvh_cache.bvh1)\n";
    std::cout << "  --equilibration-orbits N   Day/night cycles for equilibration (default: 1)\n";
    std::cout << "  --equilibration-steps-per-day N   Explicit equilibration cadence (steps/Earth-day);\n";
    std::cout << "                             -1 (default) = auto (use --equilibration-samples-per-diurnal).\n";
    std::cout << "                             > 0 = explicit override for the equilibration segment.\n";
    std::cout << "  --equilibration-samples-per-diurnal N  Accuracy-based default:\n";
    std::cout << "                             target illumination samples per diurnal cycle on the\n";
    std::cout << "                             target body (24=<2 K accuracy, 48=<1 K, 12=marginal,\n";
    std::cout << "                             <=0=legacy/match output cadence).\n";
    std::cout << "  --thermal-inertia-map FILE Path to thermal inertia map file\n";
    std::cout << "  --no-statistics            Disable computation of statistics\n";
    std::cout << "  --integration-days DAYS    Integration window in days\n";
    std::cout << "  --integration-steps N      Integration steps override\n";
    std::cout << "  --integration-offset-days  Offset for integration window (days)\n";
    std::cout << "  --integration-offset-steps Offset for integration window (steps)\n";
    std::cout << "  --thermal-debug            Enable thermal debug output for equatorial face\n";
    std::cout << "  --thermal-debug-lat DEG    Max latitude for debug face selection (default: 10)\n";
    std::cout << "  --thermal-m N              Layers in upper skin depth (default: 10)\n";
    std::cout << "  --thermal-n N              Layer growth factor (default: 15)\n";
    std::cout << "  --thermal-b N              Skin depths to bottom layer (default: 8)\n";
    std::cout << "  --geothermal-heat-flow Q   Geothermal heat flow [W/m²] (overrides planet default)\n";
    std::cout << "  --heat-flow-mask FILE      Face index file for elevated Qb (one index per line)\n";
    std::cout << "  --heat-flow-mask-value Q   Qb value [W/m²] for masked faces (default: 1.0)\n\n";

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Radiation Exchange Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --radiation-exchange       Enable inter-face radiation exchange\n";
    std::cout << "  --rad-hmatrix              Use H-matrix compression for large meshes\n";
    std::cout << "  --rad-use-gpu              Use GPU for radiosity solver (OpenCL)\n";
    std::cout << "  --rad-verbose              Print detailed radiation progress\n";
    std::cout << "  --rad-view-factor-file F   Save/load view factors from file\n";
    std::cout << "  --rad-coupling-iter N      Max coupling iterations (default: 10)\n";
    std::cout << "  --rad-coupling-temp-tol K  Temperature convergence threshold [K] (default: 2.0)\n";
    std::cout << "  --rad-coupling-flux-tol W  Flux change threshold [W/m2] (default: 0.1)\n";
    std::cout << "  --rad-coupling-face-frac F Face fraction still changing to stop (default: 0.01)\n";
    std::cout << "  --rad-use-svd              Use SVD compression instead of ACA (slower)\n";
    std::cout << "  --rad-aca-tolerance T      ACA relative tolerance (default: 0.001)\n";
    std::cout << "  --rad-aca-max-rank R       ACA maximum rank before fallback (default: 50)\n";
    std::cout << "  --rad-hmatrix-threads N    Limit H-matrix build threads (0 = auto)\n";
    std::cout << "  --rad-captured-energy F    Target captured energy fraction (default: 0.99 = 99%)\n";
    std::cout << "  --rad-max-neighbors N      Override max neighbors per face (disables adaptive)\n\n";

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Illumination Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --no-primary-reflection    Disable reflected solar radiation from primary\n";
    std::cout << "  --no-batch                 Disable SIMD batch processing for illumination\n";
    std::cout << "  --albedo-map <file>        Custom albedo FITS map file\n";
    std::cout << "  --albedo-maps-directory    Directory containing albedo maps\n";
    std::cout << "  --shadow-flux              Enable shadow/crater flux enhancement\n";
    std::cout << "  --crater-dd-ratio          Crater diameter/depth ratio (default: 8.0)\n";
    std::cout << "  --positions FILE           Export body positions to specified file\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Output Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --output DIR               Output directory (default: ./output)\n";
    std::cout << "  --output-format FORMAT     Output format: 'binary' (default) or 'vtk'\n";
    std::cout << "  --include-primary-ir       Include primary IR data in VTK files\n";
    std::cout << "  --vtk-mode MODE            VTK mode: 'both', 'thermal', or 'complete'\n";
    std::cout << "  --thermal-vtk-file FILE    Custom filename for thermal VTK output\n";
    std::cout << "  --complete-vtk-file FILE   Custom filename for complete VTK output\n";
    std::cout << "  --locations FILE           Extract illumination for specific lat/lon points\n";
    std::cout << "  --depths FILE              Depths file for thermal profile output (meters, one per line)\n";
    std::cout << "  --no-depth-plots           Skip automatic plot generation for depth profiles\n";
    std::cout << "  --location-data FILE       Specify custom location data file\n";
    std::cout << "  --series-output FILE       Location illumination time series output\n\n";

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Advanced Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --verbose                  Enable verbose output\n\n";
    
    std::cout << "For detailed documentation and examples, see the README or run:\n";
    std::cout << "  " << programName << " --generate-template\n\n";
}

std::string createDefaultLocationsFile(const std::string& output_dir) {
    // Ensure output directory exists
    try {
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }
        
        // Create a default locations file with some useful points
        std::string filename = output_dir + "/default_locations.txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            return "";
        }
        
        // Format: latitude longitude location_name (space-separated)
        file << "# lat lon name\n";
        file << "0.0 0.0 Sub_Jupiter\n";      // Equator at 0Â° longitude (sub-Jupiter point)
        file << "0.0 90.0 Trailing\n";        // Equator at 90Â°E (trailing hemisphere)
        file << "0.0 180.0 Anti_Jupiter\n";   // Equator at 180Â° (anti-Jupiter point)
        file << "0.0 270.0 Leading\n";        // Equator at 270Â°E (leading hemisphere)
        file << "45.0 0.0 North_45\n";        // 45Â°N at 0Â° longitude
        file << "-45.0 0.0 South_45\n";       // 45Â°S at 0Â° longitude
        file << "85.0 0.0 North_Pole\n";      // Near North Pole
        file << "-85.0 0.0 South_Pole\n";     // Near South Pole
        
        file.close();
        return filename;
    } catch (const std::exception& e) {
        std::cerr << "Error creating default locations file: " << e.what() << std::endl;
        return "";
    }
}

std::string generateSimulationFilename(
    const std::string& output_dir,
    const std::string& body_name,
    const std::string& start_time,
    const std::string& suffix) {
    
    // Clean the start time to be filesystem-friendly
    std::string clean_time = start_time;
    std::replace(clean_time.begin(), clean_time.end(), ':', '-');
    std::replace(clean_time.begin(), clean_time.end(), ' ', '_');
    std::replace(clean_time.begin(), clean_time.end(), 'T', '_');
    
    // Trim to a reasonable length (first 10 chars)
    if (clean_time.length() > 10) {
        clean_time = clean_time.substr(0, 10);
    }
    
    // Format: output_dir/ptm_BODY_STARTTIME_suffix.vtk
    return output_dir + "/ptm_" + body_name + "_" + clean_time + "_" + suffix + ".vtk";
}

// Helper function to write binary data for a single time step
bool writeBinaryTimeStepFile(
    const std::string& filename,
    const std::vector<float>& data,
    bool write_header)
{
    try {
        const uint32_t count = static_cast<uint32_t>(data.size());
        const size_t payload_bytes = data.size() * sizeof(float);
        const size_t expected_size = (write_header ? sizeof(uint32_t) : 0) + payload_bytes;

        // Ensure directory exists
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());

        const int max_retries = 6;
        for (int attempt = 1; attempt <= max_retries; ++attempt) {
            std::string tmp = filename + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                if (!out.is_open()) {
                    std::cerr << "[writeBinaryTimeStepFile] open fail attempt " << attempt
                              << " for " << tmp << " errno=" << errno
                              << " (" << std::strerror(errno) << ")\n";
                    goto retry;
                }
                if (write_header) {
                    out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
                }
                if (payload_bytes) {
                    out.write(reinterpret_cast<const char*>(data.data()),
                              static_cast<std::streamsize>(payload_bytes));
                }
                out.flush();
                if (!out.good()) {
                    std::cerr << "[writeBinaryTimeStepFile] stream bad attempt " << attempt
                              << " for " << tmp << "\n";
                    goto retry;
                }
            }
            {
                std::error_code ec;
                auto sz = std::filesystem::file_size(tmp, ec);
                if (ec || sz != expected_size) {
                    std::cerr << "[writeBinaryTimeStepFile] size mismatch attempt " << attempt
                              << " got=" << (ec ? -1LL : (long long)sz)
                              << " exp=" << expected_size << " file=" << tmp << "\n";
                    std::filesystem::remove(tmp, ec);
                    goto retry;
                }
                std::error_code rn;
                std::filesystem::rename(tmp, filename, rn);
                if (rn) {
                    std::cerr << "[writeBinaryTimeStepFile] rename fail attempt " << attempt
                              << " : " << rn.message() << "\n";
                    std::filesystem::remove(tmp);
                    goto retry;
                }
            }
            // Final check
            {
                std::error_code ec;
                auto sz = std::filesystem::file_size(filename, ec);
                if (!ec && sz == expected_size) return true;
            }
        retry:
            std::this_thread::sleep_for(std::chrono::milliseconds(40 * attempt));
        }
        std::cerr << "[writeBinaryTimeStepFile] FAIL writing " << filename << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[writeBinaryTimeStepFile] EXCEPTION: " << e.what() << "\n";
        return false;
    } catch (...) {
        std::cerr << "[writeBinaryTimeStepFile] UNKNOWN EXCEPTION\n";
        return false;
    }
}

// Function to write time steps metadata file
bool writeTimeStepsFile(
    const std::string& filename,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[writeTimeStepsFile] ERROR: Cannot open " << filename << "\n";
        return false;
    }
    
    if (times.size() != time_strings.size()) {
        std::cerr << "[writeTimeStepsFile] ERROR: times/time_strings size mismatch\n";
        return false;
    }

    file << "# Time steps metadata\n";
    file << "# Column 1: Time elapsed from start (seconds)\n";
    file << "# Column 2: UTC time\n";
    
    for (size_t i = 0; i < times.size(); i++) {
        file << std::fixed << std::setprecision(6) << times[i] << "\t" << time_strings[i] << "\n";
    }
    
    return file.good();
}

// Function to export mesh to binary PLY format
bool exportMeshToPLY(
    const std::string& filename,
    const std::shared_ptr<Mesh>& mesh) 
{
    try {
        if (!mesh || mesh->vertices.empty() || mesh->faces.empty()) {
            std::cerr << "[exportMeshToPLY] ERROR: mesh invalid/empty\n";
            return false;
        }

        // Ensure face centers exist for per-face lat/lon
        if (mesh->face_centers.size() != mesh->faces.size()) {
            auto& centers = const_cast<std::vector<std::array<double,3>>&>(mesh->face_centers);
            centers.resize(mesh->faces.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < mesh->faces.size(); ++i) {
                const auto& f = mesh->faces[i];
                const auto& a = mesh->vertices[f[0]];
                const auto& b = mesh->vertices[f[1]];
                const auto& c = mesh->vertices[f[2]];
                centers[i] = {
                    (a[0]+b[0]+c[0])/3.0,
                    (a[1]+b[1]+c[1])/3.0,
                    (a[2]+b[2]+c[2])/3.0
                };
            }
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[exportMeshToPLY] ERROR: Cannot open " << filename << "\n";
            return false;
        }

        const size_t V = mesh->vertices.size();
        const size_t F = mesh->faces.size();

        // Header: binary_little_endian, vertices in double precision; faces keep float lat/lon for compatibility
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
        file << "element vertex " << V << "\n";
        file << "property double x\n";
        file << "property double y\n";
        file << "property double z\n";
        file << "property double nx\n";
        file << "property double ny\n";
        file << "property double nz\n";
        file << "element face " << F << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "property float latitude\n";
        file << "property float longitude\n";
        file << "end_header\n";

        // Vertices: write xyz and zero normals as float64 (double)
        const double zero_normals_d[3] = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < V; ++i) {
            const double xyz[3] = {
                static_cast<double>(mesh->vertices[i][0]),
                static_cast<double>(mesh->vertices[i][1]),
                static_cast<double>(mesh->vertices[i][2])
            };
            file.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
            file.write(reinterpret_cast<const char*>(zero_normals_d), sizeof(zero_normals_d));
        }

        // Faces: 3 indices + latitude/longitude as float (unchanged layout for the Python reader)
        for (size_t i = 0; i < F; ++i) {
            unsigned char count = 3;
            uint32_t tri[3] = {
                static_cast<uint32_t>(mesh->faces[i][0]),
                static_cast<uint32_t>(mesh->faces[i][1]),
                static_cast<uint32_t>(mesh->faces[i][2])
            };

            const auto& c = mesh->face_centers[i];
            double r = std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
            float lat = (r > 0) ? static_cast<float>(std::asin(c[2]/r) * RAD_TO_DEG) : 0.0f;
            float lon = static_cast<float>(std::atan2(c[1], c[0]) * RAD_TO_DEG);

            file.write(reinterpret_cast<const char*>(&count), sizeof(count));
            file.write(reinterpret_cast<const char*>(tri), sizeof(tri));
            file.write(reinterpret_cast<const char*>(&lat), sizeof(lat));
            file.write(reinterpret_cast<const char*>(&lon), sizeof(lon));
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[exportMeshToPLY] EXCEPTION: " << e.what() << "\n";
        return false;
    }
}

// Helper function to export mesh in binary PLY format
void export_binary_mesh(const std::string& filename, const std::shared_ptr<Mesh>& mesh) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[export_binary_mesh] ERROR: Cannot open " << filename << "\n";
        return;
    }

    const size_t V = mesh->vertices.size();
    const size_t F = mesh->faces.size();

    // Header (double precision vertices/normals; float lat/lon on faces)
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << V << "\n";
    file << "property double x\n";
    file << "property double y\n";
    file << "property double z\n";
    file << "property double nx\n";
    file << "property double ny\n";
    file << "property double nz\n";
    file << "element face " << F << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "property float latitude\n";
    file << "property float longitude\n";
    file << "end_header\n";

    // Vertices (double)
    const double zero_normals_d[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < V; ++i) {
        const double xyz[3] = {
            static_cast<double>(mesh->vertices[i][0]),
            static_cast<double>(mesh->vertices[i][1]),
            static_cast<double>(mesh->vertices[i][2])
        };
        file.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
        file.write(reinterpret_cast<const char*>(zero_normals_d), sizeof(zero_normals_d));
    }

    // Ensure face_centers exist
    if (mesh->face_centers.size() != F) {
        auto& centers = const_cast<std::vector<std::array<double,3>>&>(mesh->face_centers);
        centers.resize(F);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < F; ++i) {
            const auto& f = mesh->faces[i];
            const auto& a = mesh->vertices[f[0]];
            const auto& b = mesh->vertices[f[1]];
            const auto& c = mesh->vertices[f[2]];
            centers[i] = { (a[0]+b[0]+c[0])/3.0, (a[1]+b[1]+c[1])/3.0, (a[2]+b[2]+c[2])/3.0 };
        }
    }

    // Faces (lat/lon float)
    for (size_t i = 0; i < F; ++i) {
        unsigned char count = 3;
        uint32_t tri[3] = {
            static_cast<uint32_t>(mesh->faces[i][0]),
            static_cast<uint32_t>(mesh->faces[i][1]),
            static_cast<uint32_t>(mesh->faces[i][2])
        };

        const auto& ctr = mesh->face_centers[i];
        const double r = std::sqrt(ctr[0]*ctr[0] + ctr[1]*ctr[1] + ctr[2]*ctr[2]);
        float lat = (r > 0.0) ? static_cast<float>(std::asin(ctr[2]/r) * RAD_TO_DEG) : 0.0f;
        float lon = static_cast<float>(std::atan2(ctr[1], ctr[0]) * RAD_TO_DEG);

        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(tri), sizeof(tri));
        file.write(reinterpret_cast<const char*>(&lat), sizeof(lat));
        file.write(reinterpret_cast<const char*>(&lon), sizeof(lon));
    }

    file.flush();
    file.close();
}

// Helper function to write binary data file for a specific time step
void write_binary_data_file(const std::string& filename, 
                           const std::vector<double>& data_array,
                           size_t num_faces,
                           int time_step,
                           int total_time_steps) {
    // Always write a header with face count, then num_faces float32 values
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Header: uint32 count = num_faces
    const uint32_t nfaces = static_cast<uint32_t>(num_faces);
    file.write(reinterpret_cast<const char*>(&nfaces), sizeof(nfaces));

    // Payload: one float32 per face for this time_step (face-major source layout)
    for (size_t f = 0; f < num_faces; ++f) {
        const size_t idx = f * static_cast<size_t>(total_time_steps) + static_cast<size_t>(time_step);
        float value = 0.0f;
        if (idx < data_array.size()) {
            value = static_cast<float>(data_array[idx]);
        }
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    file.flush();
    file.close();
}

// Helper function for safe array indexing
static inline size_t safeIndex(size_t face, int t, int stride, size_t total_size) {
    size_t idx = face * static_cast<size_t>(stride) + static_cast<size_t>(t);
    return (idx < total_size) ? idx : total_size; // returns total_size as sentinel
}

struct FilteredData {
    std::vector<double> illumination;
    std::vector<double> direct;
    std::vector<double> reflected;
    std::vector<double> ir;
    std::vector<std::vector<double>> temps;
    std::vector<double> times;
    std::vector<std::string> time_strings;
    std::vector<std::array<double,3>> sun_pos;
    std::vector<std::array<double,3>> primary_pos;
};

static FilteredData filterFaceMajorStable(
    const std::vector<double>& illum_full,
    const std::vector<double>& direct_full,
    const std::vector<double>& refl_full,
    const std::vector<double>& ir_full,
    const std::vector<std::vector<double>>& temps_full,
    const std::vector<double>& times_full,
    const std::vector<std::string>& time_str_full,
    const std::vector<std::array<double,3>>& sun_pos_full,
    const std::vector<std::array<double,3>>& primary_pos_full,
    size_t face_count,
    int total_time_steps,
    int skip_equil_steps,
    int output_time_steps,
    bool verbose = false)
{
    FilteredData fd;

    // Time/meta slice
    int start = skip_equil_steps;
    int end   = std::min(start + output_time_steps, total_time_steps);
    fd.times.reserve(end - start);
    fd.time_strings.reserve(end - start);
    fd.sun_pos.reserve(end - start);
    fd.primary_pos.reserve(end - start);
    for (int t = start; t < end; ++t) {
        fd.times.push_back(times_full[t]);
        fd.time_strings.push_back(time_str_full[t]);
        fd.sun_pos.push_back(sun_pos_full[t]);
        fd.primary_pos.push_back(primary_pos_full[t]);
    }
    const int filtered_steps = static_cast<int>(fd.times.size());

    // Allocate flattened face-major arrays
    size_t out_size = face_count * static_cast<size_t>(filtered_steps);
    fd.illumination.assign(out_size, 0.0);
    fd.direct.assign(out_size, 0.0);
    fd.reflected.assign(out_size, 0.0);
    fd.ir.assign(out_size, 0.0);

    // Copy (face-major stride)
    #pragma omp parallel for schedule(static)
    for (size_t f = 0; f < face_count; ++f) {
        size_t src_base = f * static_cast<size_t>(total_time_steps);
        size_t dst_base = f * static_cast<size_t>(filtered_steps);
        for (int ot = 0; ot < filtered_steps; ++ot) {
            int st = start + ot;
            size_t si = src_base + static_cast<size_t>(st);
            size_t di = dst_base + static_cast<size_t>(ot);
            // Bounds check: use the largest available source array.
            // illum_full may be empty (freed to save memory) while component
            // arrays (direct, reflected, ir) still have the data.
            size_t max_src = std::max({illum_full.size(), direct_full.size(),
                                       refl_full.size(), ir_full.size()});
            if (si >= max_src) continue;

            double ds = direct_full.empty() ? 0.0 : direct_full[si];
            double rf = refl_full.empty()   ? 0.0 : refl_full[si];
            double ir = ir_full.empty()     ? 0.0 : ir_full[si];
            double il = illum_full.empty()  ? (ds + rf) : illum_full[si];

            // Reconstruct illumination if zero but components exist
            if (il == 0.0 && (ds > 0.0 || rf > 0.0))
                il = ds + rf;

            fd.direct[di]       = ds;
            fd.reflected[di]    = rf;
            fd.ir[di]           = ir;
            fd.illumination[di] = il;
        }
    }

    // Temperatures
    fd.temps.resize(face_count);

    #pragma omp parallel for schedule(static)
    for (size_t f = 0; f < face_count; ++f) {
        if (f >= temps_full.size()) continue;
        const auto& src = temps_full[f];
        std::vector<double> dst(filtered_steps, 50.0);
        for (int ot = 0; ot < filtered_steps; ++ot) {
            int st = start + ot;
            if (st < static_cast<int>(src.size()))
                dst[ot] = src[st];
        }
        fd.temps[f].swap(dst);
    }

    return fd;
}

static void diagnosticsIlluminationCoverage(
    const std::vector<double>& illum,
    const std::vector<double>& direct,
    size_t face_count,
    int time_steps,
    const std::string& label,
    bool verbose = false)
{
    if (!verbose) return;  // Skip diagnostics unless verbose mode is enabled

    std::cout << "\n[Diagnostics] " << label << " timestep illumination coverage:\n";
    size_t all_zero_direct = 0;
    for (int t = 0; t < time_steps; ++t) {
        size_t illuminated = 0;
        size_t direct_nonzero = 0;
        double sum_direct = 0.0;
        for (size_t f = 0; f < face_count; ++f) {
            size_t idx = f * static_cast<size_t>(time_steps) + static_cast<size_t>(t);
            if (illum[idx] > 0.0) illuminated++;
            if (direct[idx] > 0.0) {
                direct_nonzero++;
                sum_direct += direct[idx];
            }
        }
        if (direct_nonzero == 0) all_zero_direct++;

        if (t % 10 == 0 || t == time_steps - 1) {
            double avg_d = (direct_nonzero > 0) ? sum_direct / direct_nonzero : 0.0;
            std::cout << "  Timestep " << std::setw(3) << t
                      << ": illum_faces=" << std::setw(8) << illuminated
                      << ", direct_faces=" << std::setw(8) << direct_nonzero
                      << ", avg_direct=" << std::fixed << std::setprecision(2) << avg_d << "\n";
        }
    }
    if (all_zero_direct > 0) {
        std::cout << "[Diagnostics] Timesteps with zero direct solar on all faces: "
                  << all_zero_direct << "/" << time_steps << "\n";
    }
    std::cout.flush();
}

// Corrected function to write time step data with proper stride handling
bool writeTimeStepDataChunked(
    const std::string& filename,
    const std::vector<double>& data_array,
    size_t num_faces,
    int time_step,
    int time_stride,  // This is the stride in the source array (filtered_time_steps)
    const size_t CHUNK_SIZE = 100000)
{
    try {
        std::ofstream out(filename, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "[writeTimeStepDataChunked] ERROR: Cannot open " << filename << "\n";
            return false;
        }
        
        // Write header: number of faces
        uint32_t nfaces = static_cast<uint32_t>(num_faces);
        out.write(reinterpret_cast<const char*>(&nfaces), sizeof(uint32_t));
        
        const size_t total_size = data_array.size();
        std::vector<float> buffer;
        buffer.reserve(std::min(CHUNK_SIZE, num_faces));
        
        size_t zero_count = 0;
        size_t valid_count = 0;
        double sum_values = 0.0;
        double max_value = 0.0;
        
        // Process in chunks
        for (size_t start = 0; start < num_faces; start += CHUNK_SIZE) {
            size_t end = std::min(start + CHUNK_SIZE, num_faces);
            buffer.clear();
            buffer.resize(end - start);
            
            #pragma omp parallel for reduction(+:zero_count,valid_count,sum_values) reduction(max:max_value)
            for (size_t i = 0; i < (end - start); ++i) {
                size_t f = start + i;
                size_t idx = safeIndex(f, time_step, time_stride, total_size);
                float v = 0.0f;
                
                if (idx < total_size) {
                    v = static_cast<float>(data_array[idx]);
                    if (v == 0.0f) {
                        zero_count++;
                    } else {
                        valid_count++;
                        sum_values += v;
                        max_value = std::max(max_value, static_cast<double>(v));
                    }
                } else {
                    zero_count++;
                    // Debug message for out-of-bounds access
                    if (f < 5 || f == num_faces - 1) {  // Log first few and last
                        std::cerr << "[writeTimeStepDataChunked] WARNING: Out of bounds access for face " 
                                  << f << " at timestep " << time_step 
                                  << " (idx=" << idx << " >= " << total_size << ")\n";
                    }
                }
                buffer[i] = v;
            }
            
            // Write chunk to file
            out.write(reinterpret_cast<const char*>(buffer.data()),
                      buffer.size() * sizeof(float));
                      
            if (!out.good()) {
                std::cerr << "[writeTimeStepDataChunked] ERROR: Write failed for " 
                          << filename << "\n";
                return false;
            }
        }
        
        out.flush();
        if (!out.good()) {
            std::cerr << "[writeTimeStepDataChunked] ERROR: Final flush failed for " 
                      << filename << "\n";
            return false;
        }
        
        // Report statistics for debugging
        if (valid_count == 0) {
            std::cerr << "[writeTimeStepDataChunked] WARNING: Timestep " << time_step
                      << " written with ALL ZEROS (" << filename << ")\n";
        } else if (time_step % 10 == 0) {  // Report every 10th timestep
            double avg = sum_values / valid_count;
            std::cout << "  [Debug] Timestep " << time_step << ": " 
                      << valid_count << " non-zero values, avg=" << avg 
                      << ", max=" << max_value << "\n";
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[writeTimeStepDataChunked] EXCEPTION: " << e.what() 
                  << " for " << filename << "\n";
        return false;
    }
}

// Corrected temperature writer with proper bounds checking
bool writeTemperatureTimeStepChunked(
    const std::string& filename,
    const std::vector<std::vector<double>>& surface_temps,
    int time_step,
    const size_t CHUNK_SIZE = 100000)
{
    try {
        if (surface_temps.empty()) {
            std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Empty temperature data\n";
            return false;
        }
        if (time_step < 0) {
            std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Negative time_step " << time_step << "\n";
            return false;
        }

        // Ensure output directory exists and write atomically to .tmp then rename
        const auto parent = std::filesystem::path(filename).parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Cannot create directories for "
                          << parent << " : " << ec.message() << "\n";
                return false;
            }
        }
        const std::string tmpfile = filename + ".tmp";

        std::ofstream out(tmpfile, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Cannot open " << tmpfile << "\n";
            return false;
        }
        
        const size_t num_faces = surface_temps.size();
        const uint32_t nfaces = static_cast<uint32_t>(num_faces);
        out.write(reinterpret_cast<const char*>(&nfaces), sizeof(uint32_t));
        
        size_t zero_count = 0;
        size_t valid_count = 0;
        double sum_temp = 0.0;
        double min_temp = 1e10;
        double max_temp = -1e10;
        std::vector<float> buffer;
        buffer.reserve(std::min(CHUNK_SIZE, num_faces));
        
        // Process in chunks
        for (size_t start = 0; start < num_faces; start += CHUNK_SIZE) {
            const size_t end = std::min(start + CHUNK_SIZE, num_faces);
            buffer.clear();
            buffer.resize(end - start);
            
            for (size_t i = 0; i < (end - start); ++i) {
                const size_t f = start + i;
                float val = 50.0f;  // Default temperature
                
                if (f < surface_temps.size() &&
                    time_step < static_cast<int>(surface_temps[f].size())) {
                    val = static_cast<float>(surface_temps[f][static_cast<size_t>(time_step)]);
                    
                    // Validate temperature value
                    if (!std::isfinite(val) || val < 0.0f || val > 10000.0f) {
                        val = 50.0f;
                        zero_count++;
                    } else if (val == 0.0f) {
                        zero_count++;
                    } else {
                        valid_count++;
                        sum_temp += val;
                        min_temp = std::min(min_temp, static_cast<double>(val));
                        max_temp = std::max(max_temp, static_cast<double>(val));
                    }
                } else {
                    zero_count++;
                    if (f < 5) {  // Log first few misses
                        std::cerr << "[writeTemperatureTimeStepChunked] WARNING: No data for face " 
                                  << f << " at timestep " << time_step << "\n";
                    }
                }
                buffer[i] = val;
            }
            
            // Write chunk
            out.write(reinterpret_cast<const char*>(buffer.data()),
                      static_cast<std::streamsize>(buffer.size() * sizeof(float)));
                      
            if (!out.good()) {
                std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Write failed\n";
                out.close();
                std::error_code rm;
                std::filesystem::remove(tmpfile, rm);
                return false;
            }
        }
        
        out.flush();
        out.close();

        // Validate file size: header (u32) + num_faces * float32
        const uint64_t expected = sizeof(uint32_t) + static_cast<uint64_t>(num_faces) * sizeof(float);
        std::error_code szec;
        const auto written = std::filesystem::file_size(tmpfile, szec);
        if (szec || written != expected) {
            std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Size mismatch for "
                      << tmpfile << " got=" << (szec ? -1LL : static_cast<long long>(written))
                      << " expected=" << static_cast<long long>(expected) << "\n";
            std::error_code rm;
            std::filesystem::remove(tmpfile, rm);
            return false;
        }

        // Atomic rename to final path
        {
            std::error_code rm;
            std::filesystem::remove(filename, rm); // ok if it doesn't exist
        }
        std::error_code rn;
        std::filesystem::rename(tmpfile, filename, rn);
        if (rn) {
            std::cerr << "[writeTemperatureTimeStepChunked] ERROR: Rename failed: " << rn.message() << "\n";
            std::error_code rm;
            std::filesystem::remove(tmpfile, rm);
            return false;
        }
        
        // Report statistics
        if (valid_count == 0) {
            std::cerr << "[writeTemperatureTimeStepChunked] WARNING: Timestep " 
                      << time_step << " all zeros/invalid\n";
        } else if (time_step == 0 || time_step % 10 == 0) {
            const double avg = sum_temp / static_cast<double>(valid_count);
            std::cout << "  [Temp Debug] Timestep " << time_step << ": " 
                      << valid_count << " valid temps, range=[" << min_temp 
                      << "," << max_temp << "]K, avg=" << avg << "K\n";
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[writeTemperatureTimeStepChunked] EXCEPTION: " << e.what() << "\n";
        // Best-effort cleanup of temp file
        std::error_code rm;
        std::filesystem::remove(std::string(filename + ".tmp"), rm);
        return false;
    }
}

// Statistics computation structure
struct Statistics {
    std::vector<double> avg;    // Average across time
    std::vector<double> max;    // Maximum across time
    std::vector<double> min;    // Minimum across time
    // Optional pre-reduction state used by the chunked-run merge path
    // (output.write_raw_stats=true). When populated, sum[f] is the
    // running sum of valid samples and count[f] is the number of valid
    // samples — allowing N independent chunks to be combined exactly
    // via sum-of-sums and sum-of-counts for the global avg.
    std::vector<double>   sum;
    std::vector<uint32_t> count;
};

// Function to compute statistics for a data array
Statistics computeStatistics(
    const std::vector<double>& data_array,
    size_t num_faces,
    int time_steps) {
    
    Statistics stats;
    stats.avg.resize(num_faces, 0.0);
    stats.max.resize(num_faces, std::numeric_limits<double>::lowest());
    stats.min.resize(num_faces, std::numeric_limits<double>::max());
    stats.sum.assign(num_faces, 0.0);
    stats.count.assign(num_faces, 0u);

    // If data array is empty, return zeros
    if (data_array.empty()) {
        std::fill(stats.max.begin(), stats.max.end(), 0.0);
        std::fill(stats.min.begin(), stats.min.end(), 0.0);
        return stats;
    }

    // Compute statistics for each face
    for (size_t f = 0; f < num_faces; f++) {
        double sum = 0.0;
        int valid_count = 0;

        for (int t = 0; t < time_steps; t++) {
            size_t idx = f * time_steps + t;
            if (idx < data_array.size()) {
                double value = data_array[idx];
                sum += value;
                valid_count++;

                stats.max[f] = std::max(stats.max[f], value);
                stats.min[f] = std::min(stats.min[f], value);
            }
        }

        if (valid_count > 0) {
            stats.avg[f] = sum / valid_count;
        }
        stats.sum[f]   = sum;
        stats.count[f] = static_cast<uint32_t>(valid_count);

        // Handle case where no valid data found
        if (stats.max[f] == std::numeric_limits<double>::lowest()) {
            stats.max[f] = 0.0;
        }
        if (stats.min[f] == std::numeric_limits<double>::max()) {
            stats.min[f] = 0.0;
        }
    }

    return stats;
}

// Function to compute statistics for temperature data (different structure)
Statistics computeTemperatureStatistics(
    const std::vector<std::vector<double>>& surface_temps,
    size_t num_faces,
    int time_steps) {
    
    Statistics stats;
    stats.avg.resize(num_faces, 0.0);
    stats.max.resize(num_faces, std::numeric_limits<double>::lowest());
    stats.min.resize(num_faces, std::numeric_limits<double>::max());
    stats.sum.assign(num_faces, 0.0);
    stats.count.assign(num_faces, 0u);

    // If temperature data is empty, return zeros
    if (surface_temps.empty() || surface_temps.size() != num_faces) {
        std::fill(stats.max.begin(), stats.max.end(), 0.0);
        std::fill(stats.min.begin(), stats.min.end(), 0.0);
        return stats;
    }

    // Compute statistics for each face
    for (size_t f = 0; f < num_faces; f++) {
        double sum = 0.0;
        int valid_count = 0;

        if (f < surface_temps.size()) {
            const auto& face_temps = surface_temps[f];

            for (size_t t = 0; t < face_temps.size() && t < static_cast<size_t>(time_steps); t++) {
                double value = face_temps[t];
                sum += value;
                valid_count++;

                stats.max[f] = std::max(stats.max[f], value);
                stats.min[f] = std::min(stats.min[f], value);
            }
        }

        if (valid_count > 0) {
            stats.avg[f] = sum / valid_count;
        }
        stats.sum[f]   = sum;
        stats.count[f] = static_cast<uint32_t>(valid_count);

        // Handle case where no valid data found
        if (stats.max[f] == std::numeric_limits<double>::lowest()) {
            stats.max[f] = 0.0;
        }
        if (stats.min[f] == std::numeric_limits<double>::max()) {
            stats.min[f] = 0.0;
        }
    }

    return stats;
}

// Function to write statistics binary files
bool writeStatisticsBinary(
    const std::string& output_dir,
    const std::string& data_type,
    const Statistics& stats,
    size_t num_faces) {
    
    try {
        // Create statistics subdirectory
        std::string stats_dir = output_dir + "/statistics";
        std::filesystem::create_directories(stats_dir);
        
        // Write average
        std::string avg_file = stats_dir + "/" + data_type + "_avg.bin";
        std::ofstream avg_out(avg_file, std::ios::binary);
        if (!avg_out.is_open()) {
            std::cerr << "Failed to open " << avg_file << std::endl;
            return false;
        }
        uint32_t count = static_cast<uint32_t>(num_faces);
        avg_out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        for (size_t f = 0; f < num_faces; f++) {
            float val = static_cast<float>(stats.avg[f]);
            avg_out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        avg_out.close();
        
        // Write maximum
        std::string max_file = stats_dir + "/" + data_type + "_max.bin";
        std::ofstream max_out(max_file, std::ios::binary);
        if (!max_out.is_open()) {
            std::cerr << "Failed to open " << max_file << std::endl;
            return false;
        }
        max_out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        for (size_t f = 0; f < num_faces; f++) {
            float val = static_cast<float>(stats.max[f]);
            max_out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        max_out.close();
        
        // Write minimum
        std::string min_file = stats_dir + "/" + data_type + "_min.bin";
        std::ofstream min_out(min_file, std::ios::binary);
        if (!min_out.is_open()) {
            std::cerr << "Failed to open " << min_file << std::endl;
            return false;
        }
        min_out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        for (size_t f = 0; f < num_faces; f++) {
            float val = static_cast<float>(stats.min[f]);
            min_out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        min_out.close();
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception writing statistics: " << e.what() << std::endl;
        return false;
    }
}

// Write merge-friendly raw statistics into <output_dir>/statistics/_raw/.
// Produces:
//   <data_type>_sum.bin    : uint32 N + N float64 (per-face sum)
//   <data_type>_count.bin  : uint32 N + N uint32  (per-face sample count)
// Used by the chunked-run orchestrator + ptm_merge_chunks tool to combine
// N independent chunks losslessly.
bool writeRawStatsBinary(
    const std::string& output_dir,
    const std::string& data_type,
    const Statistics& stats,
    size_t num_faces) {

    if (stats.sum.size() != num_faces || stats.count.size() != num_faces) {
        std::cerr << "[raw-stats] " << data_type
                  << ": sum/count not populated (likely a streamed-stats path); "
                     "skipping raw write.\n";
        return false;
    }
    try {
        std::string raw_dir = output_dir + "/statistics/_raw";
        std::filesystem::create_directories(raw_dir);

        // <data_type>_sum.bin
        {
            std::ofstream os(raw_dir + "/" + data_type + "_sum.bin",
                             std::ios::binary);
            if (!os) return false;
            uint32_t N = static_cast<uint32_t>(num_faces);
            os.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(stats.sum.data()),
                     stats.sum.size() * sizeof(double));
            if (!os.good()) return false;
        }
        // <data_type>_count.bin
        {
            std::ofstream os(raw_dir + "/" + data_type + "_count.bin",
                             std::ios::binary);
            if (!os) return false;
            uint32_t N = static_cast<uint32_t>(num_faces);
            os.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(stats.count.data()),
                     stats.count.size() * sizeof(uint32_t));
            if (!os.good()) return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception writing raw stats: " << e.what() << "\n";
        return false;
    }
}

// Write the Schorghofer ⟨E(z)⟩ accumulator (raw form) so a chunked-run
// merge step can sum sum_E and n_samples across chunks before solving
// for stability depth on the merged accumulator.
//   ice_depth_sum_E.bin     : uint32 n_layers,n_faces + n_layers*n_faces float32 (layer-major)
//   ice_depth_n_samples.txt : single integer line
//   depth_grid.bin          : uint32 n_layers + n_layers float64 (z values)
bool writeIceStabilityRaw(
    const std::string& output_dir,
    const ice::IceStabilityAccumulator& acc) {
    if (!acc.ready()) {
        std::cerr << "[raw-stats] ice-stability accumulator not ready; skipping raw write.\n";
        return false;
    }
    try {
        std::string raw_dir = output_dir + "/statistics/_raw";
        std::filesystem::create_directories(raw_dir);

        {
            std::ofstream os(raw_dir + "/ice_depth_sum_E.bin", std::ios::binary);
            if (!os) return false;
            uint32_t nl = static_cast<uint32_t>(acc.n_layers);
            uint32_t nf = static_cast<uint32_t>(acc.n_faces);
            os.write(reinterpret_cast<const char*>(&nl), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(&nf), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(acc.sum_E.data()),
                     acc.sum_E.size() * sizeof(float));
            if (!os.good()) return false;
        }
        {
            std::ofstream os(raw_dir + "/ice_depth_n_samples.txt");
            if (!os) return false;
            os << acc.n_samples << "\n";
        }
        {
            std::ofstream os(raw_dir + "/depth_grid.bin", std::ios::binary);
            if (!os) return false;
            uint32_t nl = static_cast<uint32_t>(acc.n_layers);
            os.write(reinterpret_cast<const char*>(&nl), sizeof(uint32_t));
            os.write(reinterpret_cast<const char*>(acc.z.data()),
                     acc.z.size() * sizeof(double));
            if (!os.good()) return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception writing ice-stability raw stats: " << e.what() << "\n";
        return false;
    }
}

// Implement streamed illumination statistics export (no large allocations)
bool exportIlluminationStatisticsStreamed(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    ptm::IlluminationModel& model,
    int start_step,
    int steps)
{
    if (!mesh || mesh->faces.empty() || steps <= 0) {
        std::cerr << "[Statistics] Streamed: invalid mesh or steps\n";
        return false;
    }
    const size_t F = mesh->faces.size();

    auto initStats = [&](Statistics& s){
        s.avg.assign(F, 0.0);
        s.max.assign(F, std::numeric_limits<double>::lowest());
        s.min.assign(F, std::numeric_limits<double>::max());
    };

    Statistics s_il, s_ds, s_rf, s_ir;
    initStats(s_il); initStats(s_ds); initStats(s_rf); initStats(s_ir);

    std::cout << "[Statistics] Computing illumination stats in streamed mode (faces=" << F
              << ", steps=" << steps << ", start=" << start_step << ")...\n";

    #pragma omp parallel for schedule(static)
    for (size_t f = 0; f < F; ++f) {
        double sum_il = 0.0, sum_ds = 0.0, sum_rf = 0.0, sum_ir = 0.0;
        double max_il = std::numeric_limits<double>::lowest();
        double max_ds = max_il, max_rf = max_il, max_ir = max_il;
        double min_il = std::numeric_limits<double>::max();
        double min_ds = min_il, min_rf = min_il, min_ir = min_il;

        for (int k = 0; k < steps; ++k) {
            int t = start_step + k;
            const double ds = model.getDirectSolarValue(f, t);
            const double rf = model.getPrimaryReflectedValue(f, t);
            const double ir = model.getPrimaryIRValue(f, t);
            const double il = ds + rf; // definition of illumination used elsewhere

            sum_ds += ds; sum_rf += rf; sum_ir += ir; sum_il += il;

            if (ds > max_ds) max_ds = ds;
            if (rf > max_rf) max_rf = rf;
            if (ir > max_ir) max_ir = ir;
            if (il > max_il) max_il = il;

            if (ds < min_ds) min_ds = ds;
            if (rf < min_rf) min_rf = rf;
            if (ir < min_ir) min_ir = ir;
            if (il < min_il) min_il = il;
        }

        const double invN = 1.0 / static_cast<double>(steps);
        s_ds.avg[f] = sum_ds * invN; s_ds.max[f] = max_ds; s_ds.min[f] = min_ds;
        s_rf.avg[f] = sum_rf * invN; s_rf.max[f] = max_rf; s_rf.min[f] = min_rf;
        s_ir.avg[f] = sum_ir * invN; s_ir.max[f] = max_ir; s_ir.min[f] = min_ir;
        s_il.avg[f] = sum_il * invN; s_il.max[f] = max_il; s_il.min[f] = min_il;
    }

    bool ok = true;
    ok &= writeStatisticsBinary(output_dir, "illumination",      s_il, F);
    ok &= writeStatisticsBinary(output_dir, "direct_solar",      s_ds, F);
    ok &= writeStatisticsBinary(output_dir, "primary_reflected", s_rf, F);
    ok &= writeStatisticsBinary(output_dir, "primary_ir",        s_ir, F);
    if (!ok) {
        std::cerr << "[Statistics] Streamed write failed\n";
    } else {
        std::cout << "[Statistics] Streamed illumination statistics exported\n";
    }
    return ok;
}

// Function to export statistics for all data types
bool exportStatisticsBinary(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    int time_steps,
    bool write_raw = false) {

    size_t face_count = mesh->faces.size();

    std::cout << "\n[exportStatisticsBinary] Computing and exporting statistics..." << std::endl;
    std::cout << "  Face count: " << face_count << std::endl;
    std::cout << "  Time steps: " << time_steps << std::endl;
    if (write_raw) {
        std::cout << "  Writing merge-friendly raw stats (statistics/_raw/)" << std::endl;
    }

    bool all_success = true;

    // Local helper to write per-datatype stats + optional raw side files.
    auto write_one = [&](const std::string& name, const Statistics& s) {
        bool ok = writeStatisticsBinary(output_dir, name, s, face_count);
        if (!ok) {
            std::cerr << "  Failed to write " << name << " statistics" << std::endl;
            all_success = false;
        }
        if (ok && write_raw) {
            if (!writeRawStatsBinary(output_dir, name, s, face_count)) {
                std::cerr << "  Failed to write " << name << " raw stats" << std::endl;
                all_success = false;
            }
        }
    };

    // Compute and write illumination statistics
    if (!illumination_values.empty()) {
        std::cout << "  Computing illumination statistics..." << std::endl;
        write_one("illumination",
                  computeStatistics(illumination_values, face_count, time_steps));
    }

    // Compute and write direct solar statistics
    if (!direct_solar_values.empty()) {
        std::cout << "  Computing direct solar statistics..." << std::endl;
        write_one("direct_solar",
                  computeStatistics(direct_solar_values, face_count, time_steps));
    }

    // Compute and write primary reflected statistics
    if (!primary_reflected_values.empty()) {
        std::cout << "  Computing primary reflected statistics..." << std::endl;
        write_one("primary_reflected",
                  computeStatistics(primary_reflected_values, face_count, time_steps));
    }

    // Compute and write primary IR statistics
    if (!primary_ir_values.empty()) {
        std::cout << "  Computing primary IR statistics..." << std::endl;
        write_one("primary_ir",
                  computeStatistics(primary_ir_values, face_count, time_steps));
    }

    // Compute and write temperature statistics
    if (!surface_temps.empty()) {
        std::cout << "  Computing temperature statistics..." << std::endl;
        write_one("temperature",
                  computeTemperatureStatistics(surface_temps, face_count, time_steps));
        // Note: ice-depth statistics are computed separately by the main
        // driver from the Schorghofer & Taylor (2007) IceStabilityAccumulator
        // (only when thermal.compute_ice_stability is enabled). The legacy
        // surface-T-only Newton solver was removed in favour of the
        // depth-resolved time-averaged formulation.
    }

    if (all_success) {
        std::cout << "\n[exportStatisticsBinary] Statistics export completed successfully!" << std::endl;
    } else {
        std::cerr << "\n[exportStatisticsBinary] Some statistics failed to export!" << std::endl;
    }

    return all_success;
}

// Main export function with improved temperature handling
bool exportDataAsBinary(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<double>& illumination_values,
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& times,
    const std::vector<std::string>& time_strings,
    int time_steps,
    bool is_custom_mesh,
    bool compute_statistics,
    bool skip_mesh = false)
{
    using std::cerr;
    using std::cout;
    if (!mesh || mesh->faces.empty()) {
        cerr << "[exportDataAsBinary] ERROR: mesh empty\n";
        return false;
    }
    if (time_steps <= 0) {
        cerr << "[exportDataAsBinary] ERROR: time_steps invalid\n";
        return false;
    }
    if (times.size() < static_cast<size_t>(time_steps) ||
        time_strings.size() < static_cast<size_t>(time_steps)) {
        cerr << "[exportDataAsBinary] ERROR: insufficient time metadata\n";
        return false;
    }

    const size_t F = mesh->faces.size();
    const size_t expected_flat = F * static_cast<size_t>(time_steps);
    auto chk = [&](const char* n, const std::vector<double>& v){
        if (!v.empty() && v.size()!=expected_flat) {
            cerr << "[exportDataAsBinary] WARNING: " << n << " size="
                 << v.size() << " expected=" << expected_flat << "\n";
        }
    };
    chk("illumination", illumination_values);
    chk("direct_solar", direct_solar_values);
    chk("primary_reflected", primary_reflected_values);
    chk("primary_ir", primary_ir_values);

    // Directories
    try {
        std::filesystem::create_directories(output_dir);
        if (!illumination_values.empty())      std::filesystem::create_directories(output_dir + "/illumination");
        if (!direct_solar_values.empty())      std::filesystem::create_directories(output_dir + "/direct_solar");
        if (!primary_reflected_values.empty()) std::filesystem::create_directories(output_dir + "/primary_reflected");
        if (!primary_ir_values.empty())        std::filesystem::create_directories(output_dir + "/primary_ir");
        if (!surface_temps.empty())            std::filesystem::create_directories(output_dir + "/temperature");
    } catch (const std::exception& e) {
        cerr << "[exportDataAsBinary] ERROR mkdir: " << e.what() << "\n";
        return false;
    }

    cout << "\n[exportDataAsBinary] Starting binary export\n"
         << "  Faces=" << F << " time_steps=" << time_steps
         << " components:"
         << (illumination_values.empty()?"":" illumination")
         << (direct_solar_values.empty()?"":" direct_solar")
         << (primary_reflected_values.empty()?"":" primary_reflected")
         << (primary_ir_values.empty()?"":" primary_ir")
         << (surface_temps.empty()?"":" temperature") << "\n";

    // Mesh PLY (binary with per-face lat/lon to satisfy downstream python).
    // Skipped on chunked-run chunks 1+ (the orchestrator propagates the
    // mesh from chunk 0 to the parent dir, so per-chunk copies are wasteful).
    if (!skip_mesh) {
        if (!exportMeshToPLY(output_dir + "/mesh.ply", mesh)) {
            cerr << "[exportDataAsBinary] ERROR exporting mesh\n";
            return false;
        }
    }
    if (!writeTimeStepsFile(output_dir + "/time_steps.txt", times, time_strings)) {
        cerr << "[exportDataAsBinary] ERROR writing time metadata\n";
        return false;
    }

    struct AWriter {
        static bool write(const std::string& fn,
                          uint32_t count,
                          const float* data,
                          size_t n,
                          int retries=6)
        {
            namespace fs=std::filesystem;
            const size_t expected = sizeof(uint32_t) + n*sizeof(float);
            for (int a=1;a<=retries;++a) {
                std::string tmp = fn + ".tmp";
                {
                    std::ofstream out(tmp, std::ios::binary|std::ios::trunc);
                    if (!out.is_open()) {
                        std::cerr << "[IO] open fail attempt " << a << " " << tmp << "\n";
                        goto retry;
                    }
                    out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
                    if (n) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n*sizeof(float)));
                    out.flush();
                    if (!out.good()) {
                        std::cerr << "[IO] stream bad attempt " << a << "\n";
                        goto retry;
                    }
                }
                {
                    std::error_code ec;
                    auto sz = fs::file_size(tmp, ec);
                    if (ec || sz != expected) {
                        std::cerr << "[IO] size mismatch attempt " << a
                                  << " got=" << (ec?-1LL:(long long)sz)
                                  << " exp=" << expected << "\n";
                        fs::remove(tmp, ec);
                        goto retry;
                    }
                    std::error_code rn;
                    fs::rename(tmp, fn, rn);
                    if (rn) {
                        std::cerr << "[IO] rename fail attempt " << a << " : " << rn.message() << "\n";
                        fs::remove(tmp);
                        goto retry;
                    }
                }
                return true;
            retry:
                std::this_thread::sleep_for(std::chrono::milliseconds(40*a));
            }
            std::cerr << "[IO] FAIL " << fn << "\n";
            return false;
        }
    };

    // Buffers
    std::vector<float> bIllum, bDirect, bRefl, bIR, bTemp;
    if (!illumination_values.empty())      bIllum.resize(F);
    if (!direct_solar_values.empty())      bDirect.resize(F);
    if (!primary_reflected_values.empty()) bRefl.resize(F);
    if (!primary_ir_values.empty())        bIR.resize(F);
    if (!surface_temps.empty())            bTemp.resize(F);

    const uint32_t Fu32 = static_cast<uint32_t>(F);
    int success_steps=0, fail_steps=0;
    int report_interval = std::max(1, time_steps/20);

    for (int t=0;t<time_steps;++t) {
        if ((t%report_interval)==0 || t==time_steps-1) {
            cout << "\r  Progress " << (t+1) << "/" << time_steps << std::flush;
        }
        size_t baseOffset = static_cast<size_t>(t);
        // Fill component buffers
        if (!illumination_values.empty()) {
            #pragma omp parallel for schedule(static)
            for (size_t f=0; f<F; ++f) {
                size_t idx = f*static_cast<size_t>(time_steps)+baseOffset;
                bIllum[f] = (idx < illumination_values.size()) ?
                    static_cast<float>(illumination_values[idx]) : 0.0f;
            }
        }
        if (!direct_solar_values.empty()) {
            #pragma omp parallel for schedule(static)
            for (size_t f=0; f<F; ++f) {
                size_t idx = f*static_cast<size_t>(time_steps)+baseOffset;
                bDirect[f] = (idx < direct_solar_values.size()) ?
                    static_cast<float>(direct_solar_values[idx]) : 0.0f;
            }
        }
        if (!primary_reflected_values.empty()) {
            #pragma omp parallel for schedule(static)
            for (size_t f=0; f<F; ++f) {
                size_t idx = f*static_cast<size_t>(time_steps)+baseOffset;
                bRefl[f] = (idx < primary_reflected_values.size()) ?
                    static_cast<float>(primary_reflected_values[idx]) : 0.0f;
            }
        }
        if (!primary_ir_values.empty()) {
            #pragma omp parallel for schedule(static)
            for (size_t f=0; f<F; ++f) {
                size_t idx = f*static_cast<size_t>(time_steps)+baseOffset;
                bIR[f] = (idx < primary_ir_values.size()) ?
                    static_cast<float>(primary_ir_values[idx]) : 0.0f;
            }
        }
        if (!surface_temps.empty()) {
            #pragma omp parallel for schedule(static)
            for (size_t f=0; f<F; ++f) {
                float v=50.0f;
                if (f<surface_temps.size() && t < static_cast<int>(surface_temps[f].size())) {
                    double tv=surface_temps[f][t];
                    if (std::isfinite(tv) && tv>0.0 && tv<10000.0) v = static_cast<float>(tv);
                }
                bTemp[f]=v;
            }
        }

        std::ostringstream oss; oss << std::setw(4) << std::setfill('0') << t;
        std::string ts=oss.str();
        bool ok=true;
        auto writeComp=[&](const char* name, const std::vector<float>& buf){
            if (buf.empty()) return true;
            std::string fn = output_dir + "/" + name + "/" + name + "_" + ts + ".bin";
            return AWriter::write(fn, Fu32, buf.data(), buf.size());
        };
        if (!illumination_values.empty())      ok &= writeComp("illumination", bIllum);
        if (!direct_solar_values.empty())      ok &= writeComp("direct_solar", bDirect);
        if (!primary_reflected_values.empty()) ok &= writeComp("primary_reflected", bRefl);
        if (!primary_ir_values.empty())        ok &= writeComp("primary_ir", bIR);
        if (!surface_temps.empty())            ok &= writeComp("temperature", bTemp);

        if (ok) ++success_steps; else { ++fail_steps; cerr << "\n[exportDataAsBinary] timestep " << t << " write failure\n"; }
    }
    cout << "\n  Export complete: " << success_steps << "/" << time_steps
         << (fail_steps? " (failures)":"") << "\n";

    // Metadata
    {
        std::ofstream meta(output_dir + "/metadata.txt");
        if (meta.is_open()) {
            meta << "faces=" << F << "\n";
            meta << "time_steps=" << time_steps << "\n";
            meta << "successful_timesteps=" << success_steps << "\n";
            meta << "failed_timesteps=" << fail_steps << "\n";
            meta << "is_custom_mesh=" << (is_custom_mesh?"true":"false") << "\n";
            meta << "components=";
            bool first=true;
            auto add=[&](const char* n,const std::vector<double>& v){
                if (v.empty()) return;
                meta << (first?"":",") << n; first=false;
            };
            add("illumination", illumination_values);
            add("direct_solar", direct_solar_values);
            add("primary_reflected", primary_reflected_values);
            add("primary_ir", primary_ir_values);
            if (!surface_temps.empty()) meta << (first?"":",") << "temperature";
            meta << "\n";
        } else {
            cerr << "[exportDataAsBinary] WARNING metadata write failed\n";
        }
    }

    std::cout << "\nBinary export completed successfully!\n";
    
    // Export statistics if enabled
    if (compute_statistics) {
        bool stats_success = exportStatisticsBinary(
            output_dir, mesh,
            illumination_values, direct_solar_values,
            primary_reflected_values, primary_ir_values,
            surface_temps, time_steps
        );
        if (!stats_success) {
            std::cerr << "Warning: Statistics export had errors\n";
        }
    }

    return fail_steps==0;
}

void printBanner() {
    std::cout << "\n";
    std::cout << "    ╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "    ║    *      .    *     \\   ||    //      * .   .      *         ║\n";
    std::cout << "    ║       .      *     \\  \\  ||   //    //       .             .  ║\n";
    std::cout << "    ║  *      .       \\   \\  \\  ☀  //    //      .      *           ║\n";
    std::cout << "    ║     .         *  \\   ...-'''''-...//    *         .           ║\n";
    std::cout << "    ║           _ ... .-'''''   PTM3D     ``''-.....,__      .      ║\n";
    std::cout << "    ║   __,.-''    Planetary Thermophysical Model 3D     `'-..., _  ║\n";
    std::cout << "    ║  ═══════════════════════════════════════════════════════════  ║\n";
    std::cout << "    ║      Coupled Thermal-Illumination for Planetary Bodies        ║\n";
    std::cout << "    ╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    printBanner();

    // ========================================================================
    // PHASE 1: Check for config file and template generation BEFORE anything else
    // ========================================================================
    std::string config_file = "";
    bool has_config_file = false;
    
    // Quick scan for --config and --generate-template flags
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[i + 1];
            has_config_file = true;
        } else if (arg == "--generate-template") {
            // Handle template generation and exit
            std::string template_file = "ptm_config_template.yaml";
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                template_file = argv[i + 1];
            }
            ptm::ConfigParser::generateTemplate(template_file);
            return 0;
        } else if (arg == "--help") {
            showUsage(argv[0]);
            return 0;
        } else if (arg == "--list-gpu-devices") {
#ifdef PTM_GPU_OPENCL
            auto devs = gpu::opencl::OpenCLContext::enumerateGPUs(/*only_available=*/false);
            if (devs.empty()) {
                std::cout << "No OpenCL GPU devices detected." << std::endl;
            } else {
                std::cout << "Detected " << devs.size()
                          << " OpenCL GPU device(s):\n";
                for (size_t k = 0; k < devs.size(); ++k) {
                    std::cout << "  GPU " << k << ": " << devs[k].name
                              << " (" << (devs[k].global_mem_size / (1024 * 1024))
                              << " MB, " << devs[k].max_compute_units << " CUs"
                              << (devs[k].available ? "" : ", UNAVAILABLE")
                              << ")" << std::endl;
                }
            }
#else
            std::cout << "Built without OpenCL support; --list-gpu-devices is a no-op." << std::endl;
#endif
            return 0;
        }
    }

    // ========================================================================
    // PHASE 2: Initialize configuration with defaults
    // ========================================================================
    ptm::SimulationConfig sim_config;  // Starts with default values
    
    // If config file specified, load it
    if (has_config_file) {
        try {
            ptm::ConfigParser parser;
            sim_config = parser.loadFromYAML(config_file);
            std::cout << "✓ Configuration loaded from: " << config_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config file: " << e.what() << std::endl;
            std::cerr << "Continuing with default configuration..." << std::endl;
            // Continue with defaults - don't exit
        }
    }

    // ========================================================================
    // PHASE 3: Map config to local variables (PRESERVED: All original variables)
    // ========================================================================
    // These maintain the exact same variable names and types as the original code
    
    // Simulation parameters (from config or defaults)
    std::string start_time = sim_config.time.start_time;
    double duration_days = sim_config.time.duration_days;
    int mesh_resolution = sim_config.mesh.resolution;
    int time_steps = sim_config.time.timesteps;
    std::string output_dir = sim_config.output.directory;
    double integration_days = sim_config.thermal.integration_days;
    int integration_steps_cli = sim_config.thermal.integration_steps;
    double integration_offset_days = sim_config.thermal.integration_offset_days;
    int integration_offset_steps_cli = sim_config.thermal.integration_offset_steps;
    std::string positions_file = sim_config.illumination.positions_file;
    bool use_batch_processing = sim_config.illumination.use_batch_processing;
    bool export_locations = !sim_config.output.locations_file.empty();
    std::string locations_file = sim_config.output.locations_file;
    std::string series_output = sim_config.output.series_output.empty()
        ? "illumination_series.csv" : sim_config.output.series_output;
    std::string depths_file = sim_config.output.depths_file;
    if (!depths_file.empty()) {
        std::cout << "Depths file configured: " << depths_file << std::endl;
    }
    std::string location_plots_dir = "";
    bool generate_depth_plots = true;  // Generate plots for depth profiles by default
    std::string executable_path = argv[0];  // Store for finding Python scripts
    bool include_primary_reflection = sim_config.illumination.include_primary_reflection;
    bool include_primary_ir_in_vtk = sim_config.output.include_primary_ir;
    std::string vtk_output_mode = sim_config.output.vtk_mode;
    std::string output_format = sim_config.output.format;
    bool enable_shadow_flux = sim_config.thermal.enable_shadow_flux;
    double crater_diameter_depth_ratio = sim_config.thermal.crater_dd_ratio;
    
    // Create the thermal config instance (used for non-mesh thermal runs)
    // Note: thermal_debug settings are applied later after CLI parsing
    thermal::Config config;

    bool compute_statistics = sim_config.thermal.compute_statistics;

    // Schorghofer & Taylor (2007) ice-stability accumulator. Persists from
    // the thermal solve through the statistics-export phase. Empty / unused
    // unless sim_config.thermal.compute_ice_stability is true.
    ice::IceStabilityAccumulator ice_stability_accum;

    // Mesh input options
    std::string mesh_file = sim_config.mesh.import_file;
    double mesh_scale_factor = sim_config.mesh.scale_factor;
    
    // Parameter to control verbosity
    bool verbose = sim_config.advanced.verbose;
    
    // Thermal model parameters
    bool run_thermal_mesh = sim_config.thermal.run_thermal_mesh;
    bool thermal_debug = sim_config.thermal.debug;
    double thermal_debug_lat = sim_config.thermal.debug_lat_threshold;

    // Thermal multiprocessing parameters
    int thermal_threads = sim_config.thermal.threads;

    // Command-line parameter for equilibration orbits (fractional allowed)
    double equilibration_orbits = sim_config.thermal.equilibration_orbits;

    // Optional: separate (coarser) cadence for the equilibration period.
    // Priority order:
    //   1. equilibration_steps_per_day > 0  => explicit override
    //   2. equilibration_samples_per_diurnal_cycle > 0 => auto-compute from body's synodic day
    //   3. otherwise => legacy (match output cadence)
    double eq_steps_per_day_cli = sim_config.thermal.equilibration_steps_per_day;
    double eq_samples_per_diurnal_cli = sim_config.thermal.equilibration_samples_per_diurnal_cycle;

    // GPU thermal solver from YAML config (can be overridden by --use-gpu-thermal CLI flag)
    if (sim_config.thermal.use_gpu) {
        config.setProperty("use_gpu_thermal", true);
        std::cout << "Using GPU persistent kernel thermal solver (OpenCL)" << std::endl;
    }

    // Thermal grid parameters
    int thermal_m = sim_config.thermal.grid_m;
    int thermal_n = sim_config.thermal.grid_n;
    int thermal_b = sim_config.thermal.grid_b;
    
    // Command line options for maps
    std::string albedo_map_file = sim_config.illumination.albedo_map;
    std::string thermal_inertia_map_file = sim_config.thermal.thermal_inertia_map;
    std::string h_parameter_map_file     = sim_config.thermal.h_parameter_map;
    bool        use_default_property_map = sim_config.thermal.use_default_property_map;
    std::string property_maps_directory  = sim_config.thermal.property_maps_directory;
    std::string albedo_maps_directory = sim_config.illumination.albedo_maps_directory;
    bool use_albedo_map = sim_config.illumination.use_albedo_map;

    // VTK files
    std::string thermal_vtk_file = sim_config.output.thermal_vtk_file;
    std::string complete_vtk_file = sim_config.output.complete_vtk_file;

    // New command line option
    bool export_basic_mesh = sim_config.mesh.export_basic;

    // Planet (default to Europa, but can be overridden)
    thermal::Planet planet(sim_config.time.body);

    // Geothermal heat flow override (from config or command line)
    double geothermal_heat_flow = sim_config.thermal.geothermal_heat_flow;

    std::shared_ptr<ptm::IlluminationModel> illumination_model;
    
    // Location data file argument
    std::string location_data_arg = sim_config.output.location_data;

    // Radiation exchange configuration (defaults, overridden by YAML then CLI)
    bool enable_radiation_exchange = false;
    bool rad_use_hmatrix = false;
    bool rad_use_gpu = false;
    bool rad_verbose = false;
    std::string rad_view_factor_file = "";
    int rad_max_coupling_iter = 10;
    double rad_coupling_temp_tol = 2.0;     // Temperature convergence threshold [K]
    double rad_coupling_flux_tol = 0.1;     // Flux change threshold [W/m2]
    double rad_coupling_face_frac = 0.01;   // Face fraction still changing to stop

    // ACA (Adaptive Cross Approximation) configuration for H-matrix
    bool rad_use_aca = true;                // Use ACA by default (much faster than SVD)
    double rad_aca_tolerance = 1e-3;        // ACA relative tolerance
    int rad_aca_max_rank = 50;              // Maximum rank before falling back to sparse
    int rad_hmatrix_threads = 0;            // 0 = auto, >0 = manual thread limit

    // Adaptive neighbor selection for sparse view factors
    bool rad_adaptive_neighbors = true;          // Enable per-face adaptive selection
    double rad_captured_energy_target = 0.95;    // Target fraction of radiative potential to capture
    int rad_max_neighbors = 2000;                // Hard ceiling per face
    int rad_min_neighbors = 20;                  // Minimum neighbors per face
    int rad_max_neighbors_override = -1;         // -1 = use adaptive, >0 = fixed override
    bool rad_use_cluster_cull = true;            // Conservative octree cluster-pair cull before ranking
    bool rad_force_hmatrix = false;              // Bypass historical H-matrix auto-disable (vestigial)
    bool rad_use_monte_carlo_vf = false;         // Monte Carlo ray-gather view factors
    int  rad_mc_rays_per_face = 100000;          // Rays per source face for MC method

    // Apply radiation settings from YAML config (before CLI override)
    enable_radiation_exchange = sim_config.radiation.enabled;
    rad_use_hmatrix = sim_config.radiation.use_hmatrix;
    rad_use_gpu = sim_config.radiation.use_gpu;
    rad_verbose = sim_config.radiation.verbose;
    rad_view_factor_file = sim_config.radiation.view_factor_file;
    rad_max_coupling_iter = sim_config.radiation.max_coupling_iterations;
    rad_coupling_temp_tol = sim_config.radiation.coupling_temp_threshold;
    rad_coupling_flux_tol = sim_config.radiation.coupling_flux_threshold;
    rad_coupling_face_frac = sim_config.radiation.coupling_face_fraction;
    rad_use_aca = sim_config.radiation.use_aca;
    rad_aca_tolerance = sim_config.radiation.aca_tolerance;
    rad_aca_max_rank = sim_config.radiation.aca_max_rank;
    rad_hmatrix_threads = sim_config.radiation.hmatrix_threads;
    // Adaptive neighbor selection parameters
    rad_adaptive_neighbors = sim_config.radiation.adaptive_neighbors;
    rad_captured_energy_target = sim_config.radiation.captured_energy_target;
    rad_max_neighbors = sim_config.radiation.max_neighbors;
    rad_min_neighbors = sim_config.radiation.min_neighbors;
    rad_use_cluster_cull = sim_config.radiation.use_cluster_cull;
    rad_use_monte_carlo_vf = sim_config.radiation.use_monte_carlo_vf;
    rad_mc_rays_per_face = sim_config.radiation.mc_rays_per_face;

    if (enable_radiation_exchange) {
        std::cout << "Radiation exchange enabled from config file" << std::endl;
        if (rad_use_hmatrix) {
            std::cout << "  H-matrix compression: enabled" << std::endl;
        }
    }

    // ========================================================================
    // PHASE 4: Command-line argument parsing (PRESERVED: Exact original logic)
    // Command-line arguments OVERRIDE config file values
    // ========================================================================
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Skip --config and its argument (already processed)
        if (arg == "--config") {
            i++; // Skip the filename argument
            continue;
        }
        
        // Skip --generate-template (already handled)
        if (arg == "--generate-template") {
            // Skip optional filename if present
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                i++;
            }
            continue;
        }
        
        // ===== ALL ORIGINAL COMMAND-LINE ARGUMENTS BELOW THIS LINE =====
        
        if (arg == "--start" && i + 1 < argc) {
            start_time = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_days = std::stod(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            mesh_resolution = std::stoi(argv[++i]);
        } else if (arg == "--timesteps" && i + 1 < argc) {
            time_steps = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--include-primary-ir") {
            include_primary_ir_in_vtk = true;
            std::cout << "Including primary IR data in VTK files" << std::endl;
        } else if (arg == "--vtk-mode" && i + 1 < argc) {
            vtk_output_mode = argv[++i];
            // Validate the mode
            if (vtk_output_mode != "both" && vtk_output_mode != "thermal" && vtk_output_mode != "complete") {
                std::cerr << "Error: vtk-mode must be 'both', 'thermal', or 'complete'" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
            std::cout << "VTK output mode set to: " << vtk_output_mode << std::endl;
        } else if (arg == "--output-format" && i + 1 < argc) {
            std::string format = argv[++i];
            if (format == "binary" || format == "vtk") {
                output_format = format;
                std::cout << "Output format set to: " << output_format << std::endl;
            } else {
                std::cerr << "Error: output-format must be 'binary' or 'vtk'" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--thermal-vtk-file" && i + 1 < argc) {
            thermal_vtk_file = argv[++i];
            std::cout << "Using custom thermal VTK file: " << thermal_vtk_file << std::endl;
        } else if (arg == "--complete-vtk-file" && i + 1 < argc) {
            complete_vtk_file = argv[++i];
            std::cout << "Using custom complete VTK file: " << complete_vtk_file << std::endl;
        } else if (arg == "--positions" && i + 1 < argc) {
            positions_file = argv[++i];
        } else if (arg == "--no-batch") {
            use_batch_processing = false;
        } else if (arg == "--locations" && i + 1 < argc) {
            export_locations = true;
            locations_file = argv[++i];
        } else if (arg == "--depths" && i + 1 < argc) {
            depths_file = argv[++i];
            std::cout << "Using depths file for thermal profiles: " << depths_file << std::endl;
        } else if (arg == "--no-depth-plots") {
            generate_depth_plots = false;
        } else if (arg == "--series-output" && i + 1 < argc) {
            series_output = argv[++i];
        } else if (arg == "--no-primary-reflection") {
            include_primary_reflection = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--thermal-mesh") {
            run_thermal_mesh = true;
        } else if (arg == "--thermal-debug") {
            thermal_debug = true;
        } else if (arg == "--thermal-debug-lat" && i + 1 < argc) {
            thermal_debug_lat = std::stod(argv[++i]);
        } else if (arg == "--thermal-m" && i + 1 < argc) {
            thermal_m = std::stoi(argv[++i]);
        } else if (arg == "--thermal-n" && i + 1 < argc) {
            thermal_n = std::stoi(argv[++i]);
        } else if (arg == "--thermal-b" && i + 1 < argc) {
            thermal_b = std::stoi(argv[++i]);
        } else if (arg == "--thermal-threads" && i + 1 < argc) {
            thermal_threads = std::stoi(argv[++i]);
        } else if (arg == "--use-simd-batch") {
            // The SIMD batch solver is now on by default; this flag is kept
            // as a no-op for backwards-compatible scripts. Use --no-simd-batch
            // to force the legacy scalar Model::run path.
            config.setProperty("use_simd_batch", true);
        } else if (arg == "--no-simd-batch") {
            config.setProperty("use_simd_batch", false);
            std::cout << "SIMD batch solver disabled; using legacy Model::run scalar path" << std::endl;
        } else if (arg == "--use-gpu-thermal") {
            config.setProperty("use_gpu_thermal", true);
            std::cout << "Using GPU persistent kernel thermal solver (OpenCL)" << std::endl;
        } else if (arg == "--gpu-devices" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "all") {
                sim_config.thermal.gpu_devices = 0;  // 0 = all available
            } else {
                sim_config.thermal.gpu_devices = std::stoi(val);
            }
            std::cout << "[Multi-GPU] gpu_devices = "
                      << (sim_config.thermal.gpu_devices == 0 ? "auto"
                                                              : std::to_string(sim_config.thermal.gpu_devices))
                      << std::endl;
        } else if (arg == "--gpu-device-ids" && i + 1 < argc) {
            sim_config.thermal.gpu_device_ids.clear();
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) sim_config.thermal.gpu_device_ids.push_back(std::stoi(token));
            }
            std::cout << "[Multi-GPU] gpu_device_ids =";
            for (int id : sim_config.thermal.gpu_device_ids) std::cout << " " << id;
            std::cout << std::endl;
        } else if (arg == "--gpu-batch-size" && i + 1 < argc) {
            long long v = std::stoll(argv[++i]);
            sim_config.thermal.gpu_batch_size = static_cast<size_t>(std::max<long long>(0, v));
            std::cout << "[Multi-GPU] gpu_batch_size = "
                      << sim_config.thermal.gpu_batch_size
                      << (sim_config.thermal.gpu_batch_size == 0 ? " (=legacy formula)" : "")
                      << std::endl;
        } else if (arg == "--gpu-inter-batch-pause-ms" && i + 1 < argc) {
            sim_config.thermal.gpu_inter_batch_pause_ms = std::stoi(argv[++i]);
            std::cout << "[Multi-GPU] gpu_inter_batch_pause_ms = "
                      << sim_config.thermal.gpu_inter_batch_pause_ms << std::endl;
        } else if (arg == "--use-gpu-illumination") {
            sim_config.illumination.use_gpu = true;
            std::cout << "Using GPU BVH ray tracer for illumination (OpenCL)" << std::endl;
        } else if (arg == "--illum-gpu-devices" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "all") {
                sim_config.illumination.gpu_devices = 0;
            } else {
                sim_config.illumination.gpu_devices = std::stoi(val);
            }
            std::cout << "[Illum GPU] gpu_devices = "
                      << (sim_config.illumination.gpu_devices == 0 ? "auto"
                          : std::to_string(sim_config.illumination.gpu_devices))
                      << std::endl;
        } else if (arg == "--illum-gpu-device-ids" && i + 1 < argc) {
            sim_config.illumination.gpu_device_ids.clear();
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty())
                    sim_config.illumination.gpu_device_ids.push_back(std::stoi(token));
            }
            std::cout << "[Illum GPU] gpu_device_ids =";
            for (int id : sim_config.illumination.gpu_device_ids) std::cout << " " << id;
            std::cout << std::endl;
        } else if (arg == "--illum-gpu-inter-batch-pause-ms" && i + 1 < argc) {
            sim_config.illumination.gpu_inter_batch_pause_ms = std::stoi(argv[++i]);
            std::cout << "[Illum GPU] gpu_inter_batch_pause_ms = "
                      << sim_config.illumination.gpu_inter_batch_pause_ms << std::endl;
        } else if (arg == "--bvh-cache-file" && i + 1 < argc) {
            sim_config.illumination.bvh_cache_file = argv[++i];
            std::cout << "[Illum GPU] bvh_cache_file = "
                      << sim_config.illumination.bvh_cache_file << std::endl;
        } else if (arg == "--equilibration-orbits" && i + 1 < argc) {
            equilibration_orbits = std::stoi(argv[++i]);
        } else if (arg == "--equilibration-steps-per-day" && i + 1 < argc) {
            eq_steps_per_day_cli = std::stod(argv[++i]);
        } else if (arg == "--equilibration-samples-per-diurnal" && i + 1 < argc) {
            eq_samples_per_diurnal_cli = std::stod(argv[++i]);
        } else if (arg == "--geothermal-heat-flow" && i + 1 < argc) {
            geothermal_heat_flow = std::stod(argv[++i]);
            std::cout << "Setting geothermal heat flow to: " << geothermal_heat_flow << " W/m²" << std::endl;
        } else if (arg == "--heat-flow-mask" && i + 1 < argc) {
            sim_config.thermal.heat_flow_mask_file = argv[++i];
            std::cout << "Heat flow mask file: " << sim_config.thermal.heat_flow_mask_file << std::endl;
        } else if (arg == "--heat-flow-mask-value" && i + 1 < argc) {
            sim_config.thermal.heat_flow_mask_value = std::stod(argv[++i]);
            std::cout << "Heat flow mask value: " << sim_config.thermal.heat_flow_mask_value << " W/m²" << std::endl;
        } else if (arg == "--albedo-map" && i + 1 < argc) {
            albedo_map_file = argv[++i];
            config.setProperty("albedo_map", albedo_map_file);
            std::cout << "Using custom albedo map: " << albedo_map_file << std::endl;
        } else if (arg == "--albedo-maps-directory" && i + 1 < argc) {
            albedo_maps_directory = argv[++i];
            config.setProperty("albedo_maps_directory", albedo_maps_directory);
            std::cout << "Using albedo maps directory: " << albedo_maps_directory << std::endl;
        } else if (arg == "--thermal-inertia-map" && i + 1 < argc) {
            thermal_inertia_map_file = argv[++i];
            continue;
        } else if (arg == "--mesh-file" && i + 1 < argc) {
            mesh_file = argv[++i];
        } else if (arg == "--mesh-scale" && i + 1 < argc) {
            mesh_scale_factor = std::stod(argv[++i]);
        } else if (arg == "--export-basic-mesh") {
            export_basic_mesh = true;
        } else if (arg == "--location-data" && i + 1 < argc) {
            location_data_arg = argv[++i];
        } else if (arg == "--body" || arg == "-b") {
            if (i + 1 < argc) {
                std::string body_name = argv[++i];
                std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::toupper);
                
                // Set the planetary body in config for thermal properties
                thermal::PlanetaryBody body_enum = thermal::PlanetaryBody::EUROPA; // Default
                
                // Map string to enum
                if (body_name == "MOON") body_enum = thermal::PlanetaryBody::MOON;
                else if (body_name == "MERCURY") body_enum = thermal::PlanetaryBody::MERCURY;
                else if (body_name == "EUROPA") body_enum = thermal::PlanetaryBody::EUROPA;
                else if (body_name == "ENCELADUS") body_enum = thermal::PlanetaryBody::ENCELADUS;
                else if (body_name == "TITAN") body_enum = thermal::PlanetaryBody::TITAN;
                else if (body_name == "MARS") body_enum = thermal::PlanetaryBody::MARS;
                else if (body_name == "CERES") body_enum = thermal::PlanetaryBody::CERES;
                else {
                    std::cerr << "Warning: Unrecognized body name '" << body_name 
                            << "'. Using Europa as default." << std::endl;
                }
                
                // Update both thermal model planet and illumination model settings
                planet = thermal::Planet(body_enum);
                
                // Create illumination model instance if not created yet
                if (!illumination_model) {
                    illumination_model = std::make_shared<ptm::IlluminationModel>();
                }

                // Enable shadow flux calculation if requested
                if (enable_shadow_flux) {
                    illumination_model->enableShadowFlux(true, crater_diameter_depth_ratio);
                    
                    // Pass the body's optical properties to the illumination model
                    illumination_model->setBodyOpticalProperties(
                        planet.albedo,       // Use the planet's albedo
                        planet.emissivity    // Use the planet's emissivity
                    );
                    
                    std::cout << "Shadow flux calculation enabled with:" << std::endl;
                    std::cout << "  Diameter/depth ratio D = " << crater_diameter_depth_ratio << std::endl;
                    std::cout << "  Body albedo = " << planet.albedo << std::endl;
                    std::cout << "  Body emissivity = " << planet.emissivity << std::endl;
                }
                
                // Set the planetary body in the illumination model
                illumination_model->setPlanetaryBody(body_enum);
                std::string target_body_name = illumination_model->getTargetBodyName();
                config.setProperty("target_body_name", target_body_name);
                
                // Convert to proper case (first letter uppercase, rest lowercase)
                if (!target_body_name.empty()) {
                    std::transform(target_body_name.begin(), target_body_name.end(), target_body_name.begin(), ::tolower);
                    target_body_name[0] = std::toupper(target_body_name[0]);
                }
                config.setProperty("default_albedo", std::to_string(illumination_model->getAlbedo()));

                std::cerr << "Using planetary body: " << body_name << std::endl;
                std::cerr << "  Target: " << illumination_model->getTargetBodyName() << std::endl;
                std::cerr << "  Primary: " << illumination_model->getPrimaryBodyName() << std::endl;
                std::cerr << "  Albedo: " << illumination_model->getAlbedo() << std::endl;
            } else {
                std::cerr << "Error: --body requires a parameter" << std::endl;
                showUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--shadow-flux") {
            enable_shadow_flux = true;
            std::cout << "Enabling shadow/crater flux enhancement (default D/h ratio=" 
                      << crater_diameter_depth_ratio << ")\n";
        } else if (arg == "--crater-dd-ratio" && i + 1 < argc) {
            crater_diameter_depth_ratio = std::stod(argv[++i]);
            if (crater_diameter_depth_ratio <= 0.1) crater_diameter_depth_ratio = 8.0;
            std::cout << "Setting crater diameter/depth ratio to " 
                      << crater_diameter_depth_ratio << std::endl;
        } else if (arg == "--no-statistics") {
            compute_statistics = false;
            std::cout << "Statistics computation disabled" << std::endl;
        } else if (arg == "--integration-days" && i + 1 < argc) {
            integration_days = std::stod(argv[++i]);
            std::cout << "Using integration window: " << integration_days << " days\n";
        } else if (arg.rfind("--integration-days=", 0) == 0) {
            integration_days = std::stod(arg.substr(std::string("--integration-days=").size()));
            std::cout << "Using integration window: " << integration_days << " days\n";
        } else if (std::string(argv[i]) == "--integration-steps" && i + 1 < argc) {
            integration_steps_cli = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--integration-offset-days" && i + 1 < argc) {
            integration_offset_days = std::stod(argv[++i]);
        } else if (std::string(argv[i]) == "--integration-offset-steps" && i + 1 < argc) {
            integration_offset_steps_cli = std::stoi(argv[++i]);
        }
        // ===== Radiation Exchange Options =====
        else if (arg == "--radiation-exchange") {
            enable_radiation_exchange = true;
            std::cout << "Radiation exchange enabled" << std::endl;
        } else if (arg == "--rad-hmatrix") {
            rad_use_hmatrix = true;
            std::cout << "Using H-matrix compression for view factors" << std::endl;
        } else if (arg == "--rad-use-gpu") {
            rad_use_gpu = true;
            std::cout << "Using GPU for radiosity solver (OpenCL)" << std::endl;
        } else if (arg == "--rad-verbose") {
            rad_verbose = true;
        } else if (arg == "--rad-view-factor-file" && i + 1 < argc) {
            rad_view_factor_file = argv[++i];
            std::cout << "View factor file: " << rad_view_factor_file << std::endl;
        } else if (arg == "--rad-coupling-iter" && i + 1 < argc) {
            rad_max_coupling_iter = std::stoi(argv[++i]);
            std::cout << "Max radiation coupling iterations: " << rad_max_coupling_iter << std::endl;
        } else if (arg == "--rad-coupling-temp-tol" && i + 1 < argc) {
            rad_coupling_temp_tol = std::stod(argv[++i]);
            std::cout << "Radiation coupling temperature tolerance: " << rad_coupling_temp_tol << " K" << std::endl;
        } else if (arg == "--rad-coupling-flux-tol" && i + 1 < argc) {
            rad_coupling_flux_tol = std::stod(argv[++i]);
        } else if (arg == "--rad-coupling-face-frac" && i + 1 < argc) {
            rad_coupling_face_frac = std::stod(argv[++i]);
        } else if (arg == "--rad-use-svd") {
            rad_use_aca = false;
            std::cout << "Using SVD compression for H-matrix (slower but more accurate)" << std::endl;
        } else if (arg == "--rad-aca-tolerance" && i + 1 < argc) {
            rad_aca_tolerance = std::stod(argv[++i]);
            std::cout << "ACA tolerance: " << rad_aca_tolerance << std::endl;
        } else if (arg == "--rad-aca-max-rank" && i + 1 < argc) {
            rad_aca_max_rank = std::stoi(argv[++i]);
            std::cout << "ACA max rank: " << rad_aca_max_rank << std::endl;
        } else if (arg == "--rad-hmatrix-threads" && i + 1 < argc) {
            rad_hmatrix_threads = std::stoi(argv[++i]);
            std::cout << "H-matrix threads limit: " << rad_hmatrix_threads << std::endl;
        } else if (arg == "--rad-captured-energy" && i + 1 < argc) {
            rad_captured_energy_target = std::stod(argv[++i]);
            std::cout << "Captured energy target: " << (rad_captured_energy_target * 100) << "%" << std::endl;
        } else if (arg == "--rad-max-neighbors" && i + 1 < argc) {
            rad_max_neighbors_override = std::stoi(argv[++i]);
            std::cout << "Max neighbors override: " << rad_max_neighbors_override << " (adaptive disabled)" << std::endl;
        } else if (arg == "--rad-cluster-cull") {
            rad_use_cluster_cull = true;
            std::cout << "Cluster-pair visibility cull: enabled" << std::endl;
        } else if (arg == "--rad-no-cluster-cull") {
            rad_use_cluster_cull = false;
            std::cout << "Cluster-pair visibility cull: disabled" << std::endl;
        } else if (arg == "--rad-force-hmatrix") {
            rad_force_hmatrix = true;
            rad_use_hmatrix = true;  // forcing implies using
            std::cout << "H-matrix: FORCED (vestigial flag; large-mesh auto-disable already removed)"
                      << std::endl;
        } else if (arg == "--rad-vf-monte-carlo") {
            rad_use_monte_carlo_vf = true;
            std::cout << "View factor method: Monte Carlo ray-gather" << std::endl;
        } else if (arg == "--rad-mc-rays" && i + 1 < argc) {
            rad_mc_rays_per_face = std::stoi(argv[++i]);
            std::cout << "MC rays per face: " << rad_mc_rays_per_face << std::endl;
        }
        // ===== Generic Body Options =====
        else if (arg == "--generic-body-config" && i + 1 < argc) {
            sim_config.generic_body.config_file = argv[++i];
            std::cout << "Generic body config file: " << sim_config.generic_body.config_file << std::endl;
        } else if (arg == "--help") {
            showUsage(argv[0]);
            return 0;
        }
    }

    // Override planet geothermal heat flow if specified in config or command line
    if (geothermal_heat_flow > 0) {
        planet.Qb = geothermal_heat_flow;
        std::cout << "Overriding planet geothermal heat flow: " << planet.Qb << " W/m²" << std::endl;
    }

    // ========================================================================
    // PHASE 4b: Generic body construction (if configured)
    // ========================================================================
    bool using_generic_body = false;
    double chi_override = -1.0;  // Radiative conductivity chi (-1 = use default 2.7)
    orbital::GenericBody generic_body;

    if (sim_config.generic_body.isConfigured()) {
        const auto& gb = sim_config.generic_body;

        // If a separate body config file is specified, load it first as a base,
        // then overlay any values that were explicitly set via CLI / main YAML.
        if (!gb.config_file.empty()) {
            try {
                ptm::ConfigParser bodyParser;
                auto bodyConfig = bodyParser.loadFromYAML(gb.config_file);
                const auto& loaded = bodyConfig.generic_body;

                // Use loaded values as defaults; main config / CLI overrides win
                // (sentinel -1.0 or empty string means "not set")
                auto& dst = sim_config.generic_body;
                if (dst.shape_type.empty()) dst.shape_type = loaded.shape_type;
                if (dst.radius_km < 0) dst.radius_km = loaded.radius_km;
                if (dst.mesh_file.empty()) dst.mesh_file = loaded.mesh_file;
                if (dst.mesh_file.empty() ? false : dst.mesh_scale == 1.0) dst.mesh_scale = loaded.mesh_scale;
                if (dst.semi_axis_a_km < 0) dst.semi_axis_a_km = loaded.semi_axis_a_km;
                if (dst.semi_axis_b_km < 0) dst.semi_axis_b_km = loaded.semi_axis_b_km;
                if (dst.semi_axis_c_km < 0) dst.semi_axis_c_km = loaded.semi_axis_c_km;
                if (dst.semi_major_axis_au < 0) dst.semi_major_axis_au = loaded.semi_major_axis_au;
                if (dst.eccentricity < 0) dst.eccentricity = loaded.eccentricity;
                if (dst.inclination_deg < 0) dst.inclination_deg = loaded.inclination_deg;
                if (dst.lon_ascending_node_deg < 0) dst.lon_ascending_node_deg = loaded.lon_ascending_node_deg;
                if (dst.arg_perihelion_deg < 0) dst.arg_perihelion_deg = loaded.arg_perihelion_deg;
                if (dst.mean_anomaly_deg < 0) dst.mean_anomaly_deg = loaded.mean_anomaly_deg;
                if (dst.orbital_epoch.empty()) dst.orbital_epoch = loaded.orbital_epoch;
                if (dst.spin_period_hours < 0) dst.spin_period_hours = loaded.spin_period_hours;
                if (dst.pole_ra_deg < 0) dst.pole_ra_deg = loaded.pole_ra_deg;
                if (dst.pole_dec_deg < 0) dst.pole_dec_deg = loaded.pole_dec_deg;
                if (dst.albedo < 0) dst.albedo = loaded.albedo;
                if (dst.emissivity < 0) dst.emissivity = loaded.emissivity;
                if (dst.ks < 0) dst.ks = loaded.ks;
                if (dst.kd < 0) dst.kd = loaded.kd;
                if (dst.rhos < 0) dst.rhos = loaded.rhos;
                if (dst.rhod < 0) dst.rhod = loaded.rhod;
                if (dst.H_scale < 0) dst.H_scale = loaded.H_scale;
                if (dst.cp0 < 0) dst.cp0 = loaded.cp0;
                if (dst.Qb < 0) dst.Qb = loaded.Qb;
                if (dst.ephemeris_source == "auto") dst.ephemeris_source = loaded.ephemeris_source;
                if (dst.spice_metakernel.empty()) dst.spice_metakernel = loaded.spice_metakernel;
                if (dst.naif_id.empty()) dst.naif_id = loaded.naif_id;
                if (dst.name.empty()) dst.name = loaded.name;

                std::cout << "Loaded generic body config from: " << gb.config_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error loading generic body config file '" << gb.config_file
                          << "': " << e.what() << std::endl;
                return 1;
            }
        }

        // Default missing orbital/spin parameters to sensible values.
        // This handles the common case where the user provides SMA but omits
        // eccentricity etc. — defaults to a circular equatorial orbit.
        {
            auto& dst = sim_config.generic_body;
            if (dst.semi_major_axis_au > 0) {
                if (dst.eccentricity < 0) {
                    dst.eccentricity = 0.0;
                    std::cout << "  Note: Defaulting eccentricity to 0.0 (circular orbit)" << std::endl;
                }
                if (dst.inclination_deg < 0) dst.inclination_deg = 0.0;
                if (dst.lon_ascending_node_deg < 0) dst.lon_ascending_node_deg = 0.0;
                if (dst.arg_perihelion_deg < 0) dst.arg_perihelion_deg = 0.0;
                if (dst.mean_anomaly_deg < 0) dst.mean_anomaly_deg = 0.0;
                if (dst.orbital_epoch.empty()) dst.orbital_epoch = "2000-01-01T12:00:00";
            }
            if (dst.spin_period_hours < 0) {
                dst.spin_period_hours = 6.0;
                std::cout << "  Note: Defaulting spin period to 6.0 hours" << std::endl;
            }
        }

        // Re-alias after potential merge
        const auto& cfg = sim_config.generic_body;

        generic_body.name = cfg.name.empty() ? "CustomBody" : cfg.name;

        // --- Shape ---
        if (cfg.shape_type == "custom_mesh" && !cfg.mesh_file.empty()) {
            generic_body.shape.type = orbital::BodyShape::CUSTOM_MESH;
            generic_body.shape.custom_mesh_file = cfg.mesh_file;
            generic_body.shape.mesh_scale = cfg.mesh_scale;
            if (cfg.radius_km > 0)
                generic_body.shape.radius_km = cfg.radius_km;
            // Wire the custom mesh file into the top-level mesh config so
            // the existing mesh loading code picks it up.
            if (mesh_file.empty()) {
                mesh_file = cfg.mesh_file;
                mesh_scale_factor = cfg.mesh_scale;
                std::cout << "Generic body custom mesh: " << mesh_file << std::endl;
            }
        } else if (cfg.shape_type == "ellipsoidal" &&
            cfg.semi_axis_a_km > 0 && cfg.semi_axis_b_km > 0 && cfg.semi_axis_c_km > 0) {
            generic_body.shape.type = orbital::BodyShape::ELLIPSOIDAL;
            generic_body.shape.semi_axis_a_km = cfg.semi_axis_a_km;
            generic_body.shape.semi_axis_b_km = cfg.semi_axis_b_km;
            generic_body.shape.semi_axis_c_km = cfg.semi_axis_c_km;
        } else {
            generic_body.shape.type = orbital::BodyShape::SPHERICAL;
            if (cfg.radius_km > 0)
                generic_body.shape.radius_km = cfg.radius_km;
        }

        // --- Orbital elements ---
        if (cfg.hasOrbitalElements()) {
            generic_body.orbit.semi_major_axis_au = cfg.semi_major_axis_au;
            if (cfg.eccentricity >= 0)
                generic_body.orbit.eccentricity = cfg.eccentricity;
            if (cfg.inclination_deg >= 0)
                generic_body.orbit.inclination_deg = cfg.inclination_deg;
            if (cfg.lon_ascending_node_deg >= 0)
                generic_body.orbit.lon_ascending_node_deg = cfg.lon_ascending_node_deg;
            if (cfg.arg_perihelion_deg >= 0)
                generic_body.orbit.arg_perihelion_deg = cfg.arg_perihelion_deg;
            if (cfg.mean_anomaly_deg >= 0)
                generic_body.orbit.mean_anomaly_at_epoch_deg = cfg.mean_anomaly_deg;
        }

        // --- Spin state ---
        if (cfg.hasSpinState()) {
            generic_body.spin.spin_period_s = cfg.spin_period_hours * 3600.0;
            if (cfg.pole_ra_deg >= 0)
                generic_body.spin.pole_ra_deg = cfg.pole_ra_deg;
            if (cfg.pole_dec_deg >= -90.0 && cfg.pole_dec_deg != -1.0)
                generic_body.spin.pole_dec_deg = cfg.pole_dec_deg;
        }

        // --- Thermophysical properties ---
        // Start from reasonable rocky asteroid defaults, then apply any
        // explicit overrides from generic_body.surface in the config YAML.
        {
            auto defaults = orbital::GenericBody::defaultRockyAsteroid();
            generic_body.thermo = defaults.thermo;
        }
        if (cfg.albedo >= 0)     generic_body.thermo.albedo = cfg.albedo;
        if (cfg.emissivity > 0)  generic_body.thermo.emissivity = cfg.emissivity;
        if (cfg.ks > 0)          generic_body.thermo.ks = cfg.ks;
        if (cfg.kd > 0)          generic_body.thermo.kd = cfg.kd;
        if (cfg.rhos > 0)        generic_body.thermo.rhos = cfg.rhos;
        if (cfg.rhod > 0)        generic_body.thermo.rhod = cfg.rhod;
        if (cfg.H_scale > 0)     generic_body.thermo.H = cfg.H_scale;
        if (cfg.cp0 >= 0)        generic_body.thermo.cp0 = cfg.cp0;
        if (cfg.Qb >= 0)         generic_body.thermo.Qb = cfg.Qb;
        if (cfg.chi >= 0)        chi_override = cfg.chi;

        // Body radius in thermophysical properties
        generic_body.thermo.body_radius_km = generic_body.shape.meanRadiusKm();

        // --- Ephemeris source ---
        if (cfg.ephemeris_source == "spice") {
            generic_body.ephemeris_source = orbital::GenericBody::SPICE;
            generic_body.spice_metakernel = cfg.spice_metakernel;
            generic_body.spice_body_id = cfg.naif_id;
        } else {
            generic_body.ephemeris_source = orbital::GenericBody::KEPLERIAN;
        }

        // Compute derived thermophysical properties (synodic day, S, year)
        generic_body.updateDerivedProperties();

        // Validate
        std::string error_msg;
        if (!generic_body.validate(error_msg)) {
            std::cerr << "Error: Invalid generic body configuration: " << error_msg << std::endl;
            return 1;
        }

        // Create thermal Planet from generic body properties
        planet = thermal::Planet(generic_body.name, generic_body.thermo);

        // Apply geothermal override if specified
        if (geothermal_heat_flow > 0) {
            planet.Qb = geothermal_heat_flow;
        }

        using_generic_body = true;

        std::cout << "Using generic body: " << generic_body.name << std::endl;
        std::cout << "  Semi-major axis: " << generic_body.orbit.semi_major_axis_au << " AU" << std::endl;
        std::cout << "  Eccentricity: " << generic_body.orbit.eccentricity << std::endl;
        std::cout << "  Spin period: " << generic_body.spin.spin_period_s / 3600.0 << " hours" << std::endl;
        std::cout << "  Solar constant: " << generic_body.thermo.S << " W/m²" << std::endl;
        std::cout << "  Synodic day: " << generic_body.thermo.day / 3600.0 << " hours" << std::endl;
        std::cout << "  Albedo: " << generic_body.thermo.albedo << std::endl;
        std::cout << "  Emissivity: " << generic_body.thermo.emissivity << std::endl;
        std::cout << "  Body radius: " << generic_body.shape.meanRadiusKm() << " km" << std::endl;
        std::cout << "  Ephemeris: " << (generic_body.ephemeris_source == orbital::GenericBody::KEPLERIAN
                                          ? "Keplerian" : "SPICE") << std::endl;
        if (generic_body.ephemeris_source == orbital::GenericBody::SPICE && !generic_body.spice_metakernel.empty()) {
            std::cout << "  Metakernel: " << generic_body.spice_metakernel << std::endl;
        }
    }

    // ========================================================================
    // PHASE 4c: Constant thermal inertia mode
    // When thermophysical.constant_thermal_inertia > 0, override properties
    // so that TI = sqrt(k * rho * cp) is constant in temperature AND depth:
    //   chi = 0       (k = ks, constant in T)
    //   cpCoeff = {}  (cp = cp0, constant in T)
    //   cp0 = TI^2 / (ks * rhos)
    //   kd  = ks      (collapse the regolith two-layer profile)
    //   rhod = rhos   (so k(z) and rho(z) are uniform with depth)
    // Without the kd/rhod override, the per-layer interpolation
    //   k(z)   = kd  + (ks  - kd ) * exp(-z/H)
    //   rho(z) = rhod + (rhos - rhod) * exp(-z/H)
    // (see ~line 3410) leaves TI(z) varying from `constant_ti` at z=0 to
    // sqrt(kd * rhod * cp_const) at z >> H, defeating the user's intent.
    // ========================================================================
    double constant_ti = sim_config.thermophysical.constant_thermal_inertia;
    if (constant_ti > 0) {
        double ks_val = planet.ks;
        double rhos_val = planet.rhos;
        if (ks_val > 0 && rhos_val > 0) {
            double cp_const = (constant_ti * constant_ti) / (ks_val * rhos_val);
            double old_kd = planet.kd;
            double old_rhod = planet.rhod;
            planet.cp0 = cp_const;
            planet.cpCoeff.clear();  // No T-dependent cp
            chi_override = 0.0;      // No radiative conductivity
            // Force depth-invariance: deep == surface for k and rho so the
            // regolith profile collapses and TI(z) = constant_ti at every
            // depth, not just z=0.
            planet.kd = ks_val;
            planet.rhod = rhos_val;
            std::cout << "Constant thermal inertia mode: TI = " << constant_ti
                      << " SI (constant in temperature AND depth)" << std::endl;
            std::cout << "  ks = " << ks_val << " W/m/K, rhos = " << rhos_val
                      << " kg/m^3, cp = " << cp_const << " J/kg/K" << std::endl;
            std::cout << "  chi = 0 (no radiative conductivity)" << std::endl;
            if (old_kd != ks_val || old_rhod != rhos_val) {
                std::cout << "  Collapsing regolith profile: kd "
                          << old_kd << " -> " << ks_val << " W/m/K, rhod "
                          << old_rhod << " -> " << rhos_val << " kg/m^3"
                          << std::endl;
            }
        }
    }

    // ========================================================================
    // PHASE 5: Configuration summary (optional, can be enabled with --verbose)
    // ========================================================================
    if (verbose && has_config_file) {
        std::cout << "\n=== Configuration Summary ===" << std::endl;
        std::cout << "Config file: " << config_file << std::endl;
        std::cout << "Body: " << sim_config.time.body << std::endl;
        std::cout << "Duration: " << duration_days << " days" << std::endl;
        std::cout << "Time steps: " << time_steps << std::endl;
        std::cout << "Thermal mesh: " << (run_thermal_mesh ? "enabled" : "disabled") << std::endl;
        std::cout << "============================\n" << std::endl;
    }
    
    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    
    // Start timing
    auto start_time_point = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------------
    // Create / import mesh (with ASCII PLY fallback for large DTM exports)
    // ------------------------------------------------------------------
    ptm::MeshUtils meshUtils;
    std::shared_ptr<Mesh> fallback_mesh;
    bool using_fallback_mesh = false;

    // Set body radius for procedural mesh generation
    if (using_generic_body && generic_body.shape.meanRadiusKm() > 0.0) {
        meshUtils.setBodyRadius(generic_body.shape.meanRadiusKm());
    } else if (planet.body_radius_km > 0.0) {
        meshUtils.setBodyRadius(planet.body_radius_km);
    }

    if (!mesh_file.empty()) {
        std::cout << "Importing triangulated mesh from: " << mesh_file << std::endl;

        bool imported = meshUtils.importTriangulatedMesh(mesh_file, mesh_scale_factor);
        if (!imported) {
            std::cerr << "Primary mesh import failed. Attempting ASCII PLY fallback..." << std::endl;

            std::string ext;
            if (auto p = mesh_file.find_last_of('.'); p != std::string::npos) {
                ext = mesh_file.substr(p + 1);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            }
            if (ext == "ply") {
                if (loadAsciiDTMPLY(mesh_file, fallback_mesh)) {
                    using_fallback_mesh = true;
                    std::cout << "Fallback PLY load succeeded: "
                              << fallback_mesh->vertices.size() << " vertices, "
                              << fallback_mesh->faces.size() << " faces" << std::endl;

                    if (mesh_scale_factor != 1.0) {
                        std::cout << "Applying scale factor to fallback mesh: "
                                  << mesh_scale_factor << std::endl;
                        #pragma omp parallel for schedule(static)
                        for (size_t i = 0; i < fallback_mesh->vertices.size(); ++i) {
                            fallback_mesh->vertices[i][0] *= mesh_scale_factor;
                            fallback_mesh->vertices[i][1] *= mesh_scale_factor;
                            fallback_mesh->vertices[i][2] *= mesh_scale_factor;
                        }
                        // Recompute centers after scaling
                        #pragma omp parallel for schedule(static)
                        for (size_t i = 0; i < fallback_mesh->faces.size(); ++i) {
                            const auto& f = fallback_mesh->faces[i];
                            const auto& a = fallback_mesh->vertices[f[0]];
                            const auto& b = fallback_mesh->vertices[f[1]];
                            const auto& c = fallback_mesh->vertices[f[2]];
                            fallback_mesh->face_centers[i] = {
                                (a[0]+b[0]+c[0])/3.0,
                                (a[1]+b[1]+c[1])/3.0,
                                (a[2]+b[2]+c[2])/3.0
                            };
                        }
                    }

                    if (!validateMesh(fallback_mesh)) {
                        std::cerr << "Warning: Fallback mesh validation reported issues." << std::endl;
                    }
                } else {
                    std::cerr << "Fallback ASCII PLY load failed. Reverting to default icosphere." << std::endl;
                }
            } else {
                std::cerr << "File is not a PLY; skipping fallback loader." << std::endl;
            }
        } else {
            auto imported_mesh = meshUtils.getMesh();
            /*
            std::cout << "Successfully imported mesh with "
                      << imported_mesh->vertices.size() << " vertices and "
                      << imported_mesh->faces.size()    << " faces" << std::endl;
            */

            if (mesh_scale_factor != 1.0) {
                std::cout << "Applied scale factor: " << mesh_scale_factor << std::endl;
            }
            if (!validateMesh(imported_mesh)) {
                std::cerr << "Warning: Imported mesh has issues that may affect simulation quality." << std::endl;
            }
        }

        // If both primary + fallback failed → build default icosphere
        if (!using_fallback_mesh && (!meshUtils.getMesh() || meshUtils.getMesh()->faces.empty())) {
            std::cerr << "Using default icosphere mesh." << std::endl;
            meshUtils.setMeshResolution(mesh_resolution);
            meshUtils.setupMesh();
        }
    } else {
        // Procedural mesh path
        meshUtils.setMeshResolution(mesh_resolution);

        // Try loading a pre-computed topographic mesh if enabled
        bool topo_loaded = false;
        if (sim_config.mesh.use_topographic) {
            std::string topo_dir = sim_config.mesh.topo_directory;
            if (topo_dir.empty()) {
                topo_dir = "data/topo";
            }
            topo_loaded = meshUtils.checkForTopographicMesh(
                topo_dir, mesh_resolution, sim_config.time.body);
        }

        if (!topo_loaded) {
            if (using_generic_body &&
                generic_body.shape.type == orbital::BodyShape::ELLIPSOIDAL &&
                generic_body.shape.semi_axis_a_km > 0 &&
                generic_body.shape.semi_axis_b_km > 0 &&
                generic_body.shape.semi_axis_c_km > 0) {
                // Generate tri-axial ellipsoidal mesh
                meshUtils.createEllipsoidMesh(mesh_resolution,
                    generic_body.shape.semi_axis_a_km,
                    generic_body.shape.semi_axis_b_km,
                    generic_body.shape.semi_axis_c_km);
            } else {
                meshUtils.setupMesh();
            }
        }
    }

    // Resolve final mesh pointer:
    std::shared_ptr<Mesh> mesh = using_fallback_mesh ? fallback_mesh : meshUtils.getMesh();

    if (!mesh || mesh->faces.empty()) {
        std::cerr << "ERROR: No valid mesh available after import/setup." << std::endl;
        return 1;
    }

    if (using_fallback_mesh) {
        mesh->setProperty("is_custom_mesh", "true");
        mesh->setProperty("source_file", mesh_file);
    }
    /*
    std::cout << "Final mesh in use: "
              << mesh->vertices.size() << " vertices, "
              << mesh->faces.size()    << " faces"
              << (using_fallback_mesh ? " (fallback loader)" : "") << std::endl;
    */

    if (!validateMesh(mesh)) {
        std::cerr << "Warning: Final mesh validation failed; proceeding anyway." << std::endl;
    }
    
    // Validate the mesh
    if (!validateMesh(mesh)) {
        std::cerr << "Mesh validation failed. Exiting." << std::endl;
        return 1;
    }

    // ----------------------------------------------------------------------
    // Per-face thermophysical-property map (optional)
    //
    // Three trigger conditions, in priority order:
    //   1. h_parameter_map     -> custom per-face H file (binary float32)
    //   2. thermal_inertia_map -> custom per-face TI file (binary float32);
    //                              loader inverts to effective H per face
    //   3. use_default_property_map -> auto-discover bundled icosphere map
    //                                   <property_maps_directory>/<body>/...
    //
    // The loaded H values land in planet.face_H_overrides; from there the
    // thermal solver uses them via Planet::H_for_face() (single-face) and the
    // batch path (per-face refresh of kc/rho in copyGridFromProfile).
    // ----------------------------------------------------------------------
    {
        const size_t n_faces_for_map = mesh->faces.size();
        std::optional<thermal::ThermalPropertyMap> tpm;
        std::string source_label;
        std::string map_mode_label;

        auto set_planet_overrides = [&](const thermal::ThermalPropertyMap& m) {
            planet.face_H_overrides = m.H_values();
            planet.face_H_valid     = m.valid_mask();
        };

        if (!h_parameter_map_file.empty()) {
            map_mode_label = "H_PARAMETER (custom)";
            tpm = thermal::ThermalPropertyMap::loadCustom(
                h_parameter_map_file, n_faces_for_map,
                thermal::ThermalPropertyMap::Mode::H_PARAMETER);
            source_label = h_parameter_map_file;
        } else if (!thermal_inertia_map_file.empty()) {
            map_mode_label = "THERMAL_INERTIA (custom; inverted to H)";
            tpm = thermal::ThermalPropertyMap::loadCustom(
                thermal_inertia_map_file, n_faces_for_map,
                thermal::ThermalPropertyMap::Mode::THERMAL_INERTIA);
            source_label = thermal_inertia_map_file;
        } else if (use_default_property_map) {
            // Derive ico level from face count: faces = 20 * 4^level
            int detected_level = -1;
            {
                size_t f = n_faces_for_map;
                if (f >= 20 && (f % 20) == 0) {
                    size_t r = f / 20;
                    int lvl = 0;
                    while (r > 1 && (r & 1) == 0 && (r & 2) == 0) {
                        // r should be 4^lvl (only bits at positions 0, 2, 4, ... allowed)
                        r >>= 2;
                        ++lvl;
                    }
                    if (r == 1) detected_level = lvl;
                }
            }
            if (detected_level < 0) {
                std::cerr << "[property-map] mesh face count " << n_faces_for_map
                          << " is not an icosphere (20*4^N). "
                             "Custom h_parameter_map / thermal_inertia_map required.\n";
            } else {
                map_mode_label = "H_PARAMETER (bundled default)";
                tpm = thermal::ThermalPropertyMap::loadDefault(
                    sim_config.time.body, detected_level,
                    property_maps_directory,
                    thermal::ThermalPropertyMap::Mode::H_PARAMETER);
                if (tpm) {
                    source_label = tpm->source_path();
                }
            }
        }

        if (tpm) {
            set_planet_overrides(*tpm);
            std::cout << "[property-map] " << map_mode_label
                      << "  source=" << source_label
                      << "  faces=" << tpm->H_values().size()
                      << "  default_fraction="
                      << (tpm->fraction_default() * 100.0) << "%"
                      << "  H[range]=" << *std::min_element(tpm->H_values().begin(), tpm->H_values().end())
                      << ".." << *std::max_element(tpm->H_values().begin(), tpm->H_values().end())
                      << " m\n";
        } else if (!h_parameter_map_file.empty() ||
                   !thermal_inertia_map_file.empty() ||
                   use_default_property_map) {
            std::cerr << "[property-map] WARNING: requested map could not be loaded;"
                         " falling back to single body-default H="
                      << planet.H << " m\n";
        }
    }

    // Export mesh to VTK for visualization
    if (export_basic_mesh) {
        std::string mesh_filename = output_dir + "/ptm_mesh.vtk";
        std::cout << "Exporting mesh to VTK: " << mesh_filename << std::endl;
        if (!meshUtils.exportMeshToVTK(mesh_filename)) {
            std::cerr << "Failed to export mesh to VTK" << std::endl;
        } else {
            std::cout << "Mesh exported to " << mesh_filename << std::endl;
        }
    }

    // ========================================================================
    // Mesh latitude/longitude override: rotate mesh to target position
    // ========================================================================
    if (sim_config.mesh.latitude_override > -900.0) {
        rotateMeshToLatLon(mesh,
                           sim_config.mesh.latitude_override,
                           sim_config.mesh.longitude_override);
    }

    // ========================================================================
    // Resolve view_factor_file to an absolute path.
    //
    // The GUI and CLI both pass this value as a raw string, which may be a
    // relative path (e.g. "data/vf/amundsen.vfm"). Relative paths resolved
    // against the subprocess cwd silently broke the cache when the GUI launched
    // the binary from a cwd that didn't have the target subdirectory. Resolve
    // once here so every downstream save/load uses an unambiguous absolute path.
    // Preference order: relative to the YAML config file's directory (if any),
    // else relative to the current working directory.
    // ========================================================================
    if (!rad_view_factor_file.empty()) {
        std::filesystem::path vf_path(rad_view_factor_file);
        if (vf_path.is_relative()) {
            std::error_code abs_ec;
            std::filesystem::path base;
            if (has_config_file) {
                base = std::filesystem::absolute(config_file, abs_ec).parent_path();
            }
            if (base.empty() || abs_ec) {
                base = std::filesystem::current_path(abs_ec);
            }
            if (!abs_ec) {
                rad_view_factor_file =
                    (base / vf_path).lexically_normal().string();
                std::cout << "Resolved view factor file: "
                          << rad_view_factor_file << std::endl;
            }
        }
    }

    // ========================================================================
    // EARLY VIEW FACTOR COMPUTATION (if radiation exchange enabled)
    //
    // IMPORTANT: Compute view factors BEFORE illumination to avoid OOM.
    // View factor computation needs lots of working memory (for large meshes,
    // H-matrix block building can use 2-4 GB per thread). The illumination
    // arrays for large meshes can use 60+ GB. By computing view factors first:
    //   1. Full system RAM is available for working memory
    //   2. View factors are saved to disk immediately after computation
    //   3. Working memory is freed before illumination starts
    //   4. Later, only the compressed view factor file needs to be loaded
    // ========================================================================
    bool view_factors_precomputed = false;
    if (enable_radiation_exchange && !rad_view_factor_file.empty()) {
        // Check if view factor file already exists
        std::ifstream vf_check(rad_view_factor_file, std::ios::binary);
        bool vf_file_exists = vf_check.good();
        vf_check.close();

        if (!vf_file_exists) {
            std::cout << "\n========== Early View Factor Computation ==========" << std::endl;
            std::cout << "Computing view factors BEFORE illumination to avoid memory contention." << std::endl;
            std::cout << "This allows full system RAM for view factor working memory." << std::endl;

            // Configure radiation exchange for view factor computation only
            radiation::RadiationConfig early_rad_config;
            early_rad_config.enabled = true;
            early_rad_config.use_hmatrix = rad_use_hmatrix;
            early_rad_config.use_gpu = rad_use_gpu;
            early_rad_config.verbose = rad_verbose;

            // ACA configuration (Adaptive Cross Approximation - much faster)
            early_rad_config.use_aca = rad_use_aca;
            early_rad_config.aca_tolerance = rad_aca_tolerance;
            early_rad_config.aca_max_rank = rad_aca_max_rank;
            early_rad_config.hmatrix_threads = rad_hmatrix_threads;

            // Adaptive neighbor selection configuration
            early_rad_config.adaptive_neighbors = rad_adaptive_neighbors;
            early_rad_config.captured_energy_target = rad_captured_energy_target;
            early_rad_config.max_neighbors = rad_max_neighbors;
            early_rad_config.min_neighbors = rad_min_neighbors;
            early_rad_config.use_cluster_cull = rad_use_cluster_cull;
            early_rad_config.force_hmatrix = rad_force_hmatrix;
            early_rad_config.use_monte_carlo_vf = rad_use_monte_carlo_vf;
            early_rad_config.mc_rays_per_face = rad_mc_rays_per_face;
            if (rad_max_neighbors_override > 0) {
                early_rad_config.max_neighbors = rad_max_neighbors_override;
                early_rad_config.adaptive_neighbors = false;  // Use fixed if override specified
            }

            // Create temporary radiation exchange manager for view factor computation
            radiation::RadiationExchangeManager early_rad_manager(mesh, early_rad_config);
            early_rad_manager.setEmissivity(planet.emissivity);

            // For early computation, we don't have illumination data yet.
            // Use computeViewFactorsOnly() which doesn't need illumination.
            // The face classification will be done later when illumination is available.
            std::cout << "Computing view factors for " << mesh->faces.size() << " faces..." << std::endl;

            bool vf_success = early_rad_manager.computeViewFactorsOnly();

            if (vf_success) {
                // Ensure parent directory exists before writing (silent-failure guard).
                std::error_code mkdir_ec;
                std::filesystem::create_directories(
                    std::filesystem::path(rad_view_factor_file).parent_path(), mkdir_ec);

                // Save view factors to disk immediately and honour the return value.
                // Note: Check getHMatrixViewFactors() directly since H-matrix may have been
                // auto-enabled by the manager for large meshes, even if --rad-hmatrix wasn't passed
                bool wrote = false;
                if (early_rad_manager.getHMatrixViewFactors()) {
                    wrote = early_rad_manager.getHMatrixViewFactors()->writeBinary(rad_view_factor_file);
                } else {
                    wrote = early_rad_manager.getViewFactorMatrix().writeBinary(rad_view_factor_file);
                }
                if (wrote) {
                    std::cout << "Saved view factors to: " << rad_view_factor_file << std::endl;
                    view_factors_precomputed = true;
                } else {
                    std::cerr << "ERROR: Failed to save view factors to: " << rad_view_factor_file
                              << " (check path, permissions, disk space)" << std::endl;
                    // Don't mark precomputed so the later phase will rebuild instead of
                    // trying to load a non-existent file.
                }
            } else {
                std::cerr << "Warning: Early view factor computation failed." << std::endl;
                std::cerr << "Will retry during radiation coupling phase." << std::endl;
            }

            std::cout << "=================================================\n" << std::endl;

            // The early_rad_manager goes out of scope here, freeing all working memory
            // before the illumination model allocates its large arrays.
        } else {
            std::cout << "View factor file exists: " << rad_view_factor_file << std::endl;
            std::cout << "Will load during radiation coupling phase." << std::endl;
            view_factors_precomputed = true;  // Mark as available for later loading
        }
    }

    // Initialize illumination model
    std::cout << "Setting up illumination model..." << std::endl;
    if (!illumination_model) {
        illumination_model = std::make_shared<ptm::IlluminationModel>();
    }

    if (using_generic_body) {
        // Configure illumination model for generic body (Keplerian or SPICE)
        illumination_model->setGenericBody(generic_body);
        std::cout << "Illumination model configured for generic body: " << generic_body.name
                  << " (" << (generic_body.ephemeris_source == orbital::GenericBody::KEPLERIAN
                              ? "Keplerian" : "SPICE") << " ephemeris)" << std::endl;

        // For Keplerian bodies, disable primary reflection (no parent body)
        if (generic_body.ephemeris_source == orbital::GenericBody::KEPLERIAN) {
            illumination_model->setIncludePrimaryReflection(false);
        }
    } else if (planet.name != "Custom") {
        // Standard built-in body path
        // Determine body enum from name
        thermal::PlanetaryBody body_enum = thermal::PlanetaryBody::EUROPA; // Default

        if (planet.name == "Moon") body_enum = thermal::PlanetaryBody::MOON;
        else if (planet.name == "Mercury") body_enum = thermal::PlanetaryBody::MERCURY;
        else if (planet.name == "Europa") body_enum = thermal::PlanetaryBody::EUROPA;
        else if (planet.name == "Enceladus") body_enum = thermal::PlanetaryBody::ENCELADUS;
        else if (planet.name == "Titan") body_enum = thermal::PlanetaryBody::TITAN;
        else if (planet.name == "Mars") body_enum = thermal::PlanetaryBody::MARS;
        else if (planet.name == "Ceres") body_enum = thermal::PlanetaryBody::CERES;

        // Set the illumination model to use the same body
        illumination_model->setPlanetaryBody(body_enum);
    }

    double default_visible_albedo = using_generic_body
        ? planet.albedo
        : illumination_model->getAlbedo();
    // Ensure we're using the planet's albedo, not a hardcoded value
    if (std::abs(default_visible_albedo - planet.albedo) > 0.001) {
        std::cout << "Warning: Mismatch between illumination albedo (" << default_visible_albedo
                << ") and planet albedo (" << planet.albedo << "). Using planet value." << std::endl;
        default_visible_albedo = planet.albedo;
    }
    illumination_model->initialize_spice(verbose);
    illumination_model->setAlbedo(default_visible_albedo);
    illumination_model->setKernelsPath(SPICE_KERNELS_DIR);
    if (!using_generic_body) {
        illumination_model->setIncludePrimaryReflection(include_primary_reflection);
    }

#ifdef PTM_GPU_OPENCL
    // Wire up GPU BVH ray tracer if requested. Must happen BEFORE
    // calculateIlluminationFromMesh / initializeScene so the BVH is built
    // and replicated across the pool's devices on first scene init.
    if (sim_config.illumination.use_gpu) {
        illumination_model->configureGPURayTracingDevices(
            sim_config.illumination.gpu_devices,
            sim_config.illumination.gpu_device_ids,
            sim_config.illumination.gpu_inter_batch_pause_ms);

        // BVH disk cache: explicit path wins; otherwise auto-derive from output dir.
        // "off" disables caching entirely.
        std::string bvh_cache = sim_config.illumination.bvh_cache_file;
        if (bvh_cache.empty()) {
            bvh_cache = sim_config.output.directory + "/bvh_cache.bvh1";
        }
        illumination_model->setBVHCacheFile(bvh_cache);
        if (bvh_cache != "off") {
            std::cout << "[Illum GPU] BVH cache: " << bvh_cache << std::endl;
        }

        if (!illumination_model->enableGPURayTracing(true)) {
            std::cerr << "Warning: GPU illumination requested but pool init failed; "
                         "continuing on CPU Embree" << std::endl;
        }
    }
#endif

    // Load positions if available
    if (!positions_file.empty()) {
        if (!illumination_model->loadPositionsFromFile(positions_file)) {
            std::cerr << "Failed to load positions from file: " << positions_file << std::endl;
            return 1;
        }
    }
    double synodic_day = getSynodicDayFromPlanet(planet);

    // When loading a thermal-state checkpoint (chunked-run workflow),
    // the previous chunk's final T(z) is already equilibrated — we
    // disable equilibration entirely so the chunk's flux/time vectors
    // span only the requested output window. Without this, the
    // chunk's `total_time_steps` would still include the full
    // equilibration prefix, the GPU would integrate over all of it,
    // and the per-chunk stats would mix in equilibration samples.
    if (!sim_config.thermal.initial_state_file.empty() &&
        equilibration_orbits > 0.0) {
        std::cout << "[chunked] thermal.initial_state_file is set; "
                     "forcing equilibration_orbits=0 (was "
                  << equilibration_orbits << ")\n";
        equilibration_orbits = 0.0;
    }

    // Calculate equilibration period using the correct synodic day
    // Note: equilibration_orbits represents number of day/night cycles for equilibration
    double equilibration_days = std::max(0.0, equilibration_orbits) * synodic_day;

    // The output (export) window is the user-requested duration
    const double output_days = duration_days;

    // Derive desired integration window (stats) in days from CLI
    // Default: integrate over the output window length if not specified
    double days_per_step_output = (time_steps > 0 && output_days > 0.0)
        ? (output_days / static_cast<double>(time_steps))
        : 0.0;

    double requested_integ_days = output_days;
    if (integration_steps_cli > 0 && days_per_step_output > 0.0) {
        requested_integ_days = static_cast<double>(integration_steps_cli) * days_per_step_output;
    } else if (integration_days > 0.0) {
        requested_integ_days = integration_days;
    }

    double requested_integ_offset_days = 0.0;
    if (integration_offset_steps_cli >= 0 && days_per_step_output > 0.0) {
        requested_integ_offset_days = static_cast<double>(integration_offset_steps_cli) * days_per_step_output;
    } else if (integration_offset_days > 0.0) {
        requested_integ_offset_days = integration_offset_days;
    }

    // Post-equilibration days needed to cover BOTH output and integration windows
    // We simulate at least max(output_days, integ_offset + integ_days) after equilibration.
    const double post_eq_days_needed = std::max(output_days, requested_integ_offset_days + requested_integ_days);

    // Final total simulated span (equilibration + max(output, integration))
    double total_simulation_days = equilibration_days + post_eq_days_needed;

    // Log the calculation for verification
    std::cout << "\n========== Thermal Equilibration Configuration ==========" << std::endl;
    std::cout << "Body: " << planet.name << std::endl;
    std::cout << "Synodic day (solar day): " << synodic_day << " Earth days" << std::endl;
    std::cout << "Equilibration cycles: " << std::fixed << std::setprecision(2) << equilibration_orbits << " day/night cycles" << std::endl;
    std::cout << "Total equilibration period: " << equilibration_days << " Earth days" << std::endl;
    std::cout << "Output window (for export): " << output_days << " Earth days" << std::endl;
    std::cout << "Requested integration (stats): " << requested_integ_days << " days"
            << " with offset " << requested_integ_offset_days << " days" << std::endl;
    std::cout << "Post-equilibration simulated span: " << post_eq_days_needed << " Earth days" << std::endl;
    std::cout << "Total simulation time: " << total_simulation_days << " Earth days" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    // Calculate adjusted start time that includes equilibration period
    std::string adjustedStartTime = calculateAdjustedStartTime(start_time, -equilibration_days);

    // Decoupled equilibration / output cadences.
    //
    // The OUTPUT cadence is derived from the user-requested (timesteps, duration_days)
    // and controls the sampling density of the output window.
    //
    // The EQUILIBRATION cadence defaults to the output cadence (legacy behavior) but
    // can be overridden via sim_config.thermal.equilibration_steps_per_day (or the
    // --equilibration-steps-per-day CLI flag). Using a coarser cadence during
    // equilibration dramatically reduces illumination-array size and ray-tracing cost
    // when the output cadence is very fine (e.g., 1-min for camera simulations).
    const double output_steps_per_day = (output_days > 0.0 && time_steps > 0)
        ? (static_cast<double>(time_steps) / output_days)
        : 0.0;

    // Determine equilibration cadence using priority:
    //   1. explicit steps_per_day > 0
    //   2. accuracy-based auto-compute from samples_per_diurnal_cycle > 0
    //   3. legacy (match output cadence)
    double eq_steps_per_day = output_steps_per_day;
    std::string eq_cadence_source = "legacy (matches output cadence)";
    bool user_set_eq_cadence = false;
    if (eq_steps_per_day_cli > 0.0) {
        eq_steps_per_day = eq_steps_per_day_cli;
        eq_cadence_source = "explicit (equilibration_steps_per_day)";
        user_set_eq_cadence = true;
    } else if (eq_samples_per_diurnal_cli > 0.0 && synodic_day > 0.0) {
        eq_steps_per_day = eq_samples_per_diurnal_cli / synodic_day;
        std::ostringstream oss;
        oss << "auto (" << eq_samples_per_diurnal_cli
            << " samples/diurnal cycle; synodic_day=" << synodic_day << " Earth-days)";
        eq_cadence_source = oss.str();
        user_set_eq_cadence = true;
    }

    // Note: GPU thermal solver now supports two-phase execution with different dt
    // for equilibration vs output. No fallback needed — the solver is called twice
    // with equilibration_steps passed to split the flux array.

    // Guard: fast-rotator warning (less than 12 samples per diurnal cycle during eq).
    if (user_set_eq_cadence) {
        double samples_per_diurnal = eq_steps_per_day * synodic_day;
        if (samples_per_diurnal < 12.0) {
            std::cerr << "WARNING: equilibration cadence resolves only "
                      << std::fixed << std::setprecision(2) << samples_per_diurnal
                      << " samples per diurnal cycle on " << planet.name
                      << "; consider >= 24 samples/diurnal-cycle ("
                      << (24.0 / synodic_day) << " steps/Earth-day).\n";
        }
    }

    // Compute equilibration / output / total step counts
    int equilibration_steps = (equilibration_days > 0.0 && eq_steps_per_day > 0.0)
        ? static_cast<int>(std::llround(eq_steps_per_day * equilibration_days))
        : 0;
    int output_steps = static_cast<int>(std::llround(output_steps_per_day * post_eq_days_needed));
    int total_time_steps = equilibration_steps + output_steps;

    // For sanity, output_steps_expected should match user time_steps (modulo rounding)
    int output_steps_expected = static_cast<int>(std::llround(output_steps_per_day * output_days));

    // Verify the calculation
    std::cout << "Time step calculation:" << std::endl;
    std::cout << "  Requested output time steps: " << time_steps << std::endl;
    std::cout << "  Output window (days): " << output_days << std::endl;
    std::cout << "  Output cadence:        " << std::fixed << std::setprecision(6)
              << output_steps_per_day << " steps/Earth-day" << std::endl;
    std::cout << "  Equilibration cadence: " << std::fixed << std::setprecision(6)
              << eq_steps_per_day << " steps/Earth-day"
              << " [" << eq_cadence_source << "]"
              << std::endl;
    if (user_set_eq_cadence) {
        std::cout << "    -> " << std::fixed << std::setprecision(2)
                  << (eq_steps_per_day * synodic_day)
                  << " samples per diurnal cycle on " << planet.name << std::endl;
    }
    std::cout << "  Total simulation days: " << total_simulation_days << std::endl;
    std::cout << "  Equilibration steps: " << equilibration_steps << std::endl;
    std::cout << "  Output steps:        " << output_steps << std::endl;
    std::cout << "  Total time steps (including equilibration and integration coverage): "
              << total_time_steps << std::endl;
    if (output_steps_expected != time_steps) {
        std::cout << "  Note: output steps after rounding: " << output_steps_expected << " (requested " << time_steps << ")\n";
    }

    // Sanity check
    if (equilibration_steps < 0 || total_time_steps < 1) {
        std::cerr << "ERROR: Invalid time step configuration! Check parameters." << std::endl;
        // Fallback to original behavior (legacy uniform cadence)
        total_simulation_days = equilibration_days + output_days;
        total_time_steps = std::max(1, time_steps + static_cast<int>(
            std::llround(output_steps_per_day * equilibration_days)));
        equilibration_steps = std::max(0, total_time_steps - time_steps);
        eq_steps_per_day = output_steps_per_day;
        user_set_eq_cadence = false;
    }

    // Build the ET grid: coarse equilibration block followed by fine output block.
    // This is passed to the illumination model via setExternalTimeGrid() so that
    // ray-tracing + SPICE position queries happen only at these sampled times.
    std::vector<double> external_ets;
    if (total_time_steps > 0) {
        external_ets.reserve(total_time_steps);
        double et0 = illumination_model->convertTimeToET(adjustedStartTime);
        // Equilibration block: uniform cadence strictly inside [et0, et0 + eq_days)
        if (equilibration_steps > 0) {
            double eq_seconds = equilibration_days * ptm_thermal::constants::SECONDS_PER_DAY;
            for (int i = 0; i < equilibration_steps; ++i) {
                // multiply-before-divide for robust floating-point
                external_ets.push_back(et0 + (static_cast<double>(i) * eq_seconds) /
                                              static_cast<double>(equilibration_steps));
            }
        }
        // Output block: uniform fine cadence starting exactly at et0 + equilibration_days
        double et_out_start = et0 + equilibration_days * ptm_thermal::constants::SECONDS_PER_DAY;
        double out_seconds = post_eq_days_needed * ptm_thermal::constants::SECONDS_PER_DAY;
        for (int i = 0; i < output_steps; ++i) {
            external_ets.push_back(et_out_start + (static_cast<double>(i) * out_seconds) /
                                                   static_cast<double>(output_steps));
        }
    }

    // Update the illumination model with the total time steps and external grid
    illumination_model->setTimeSteps(total_time_steps);
    if (user_set_eq_cadence &&
        std::abs(eq_steps_per_day - output_steps_per_day) > 1e-6) {
        // Only install the external grid when we actually want non-uniform sampling
        // (keeps legacy behavior byte-identical when user leaves default).
        illumination_model->setExternalTimeGrid(external_ets);
    }

    // Run illumination model for the full period but keep track of where the requested interval starts
    std::cout << "Running illumination simulation for " << total_simulation_days 
            << " days (including " << equilibration_days << " days equilibration)"
            << " starting at " << adjustedStartTime << std::endl;

    // Add ray tracing timer start
    auto ray_tracing_start = std::chrono::high_resolution_clock::now();

    bool success;
    if (use_batch_processing) {
        // Use the mesh-based optimized method - avoids recalculating face properties
        success = illumination_model->calculateIlluminationFromMesh(
            mesh,
            adjustedStartTime,
            total_simulation_days
        );
    } else {
        // Use the regular method - slower but may be more compatible
        success = illumination_model->calculateIllumination(
            mesh->vertices,
            mesh->faces,
            adjustedStartTime,
            total_simulation_days
        );
    }

    // Add ray tracing timer end and reporting
    auto ray_tracing_end = std::chrono::high_resolution_clock::now();
    auto ray_tracing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        ray_tracing_end - ray_tracing_start).count();
    double ray_tracing_sec = ray_tracing_ms / 1000.0;

    std::cout << "Ray tracing computation completed in " << std::fixed << std::setprecision(3) 
            << ray_tracing_sec << " seconds (" << ray_tracing_ms << " ms)" << std::endl;

    if (!success) {
        std::cerr << "Illumination simulation failed!" << std::endl;
        return 1;
    }

    // Get arrays from illumination model - may be empty if using chunked storage
    std::vector<double> illumination_values = illumination_model->getIllumination();
    std::vector<double> direct_solar_values = illumination_model->getDirectSolarValues();
    std::vector<double> primary_reflected_values = illumination_model->getPrimaryReflectedValues();
    std::vector<double> primary_ir_values = illumination_model->getPrimaryIRFlux();

    // Get time arrays (these are never chunked)
    const auto& times = illumination_model->getTimes();
    const auto& time_strings = illumination_model->getTimeStrings();
    const auto& sun_positions = illumination_model->getSunPositions();
    const auto& primary_positions = illumination_model->getPrimaryPositions();

    // Calculate the number of faces
    size_t num_faces = mesh->faces.size();

    // Check if arrays are empty (indicates chunked storage is being used)
    bool using_chunked_storage = direct_solar_values.empty() || 
                                primary_reflected_values.empty() || 
                                primary_ir_values.empty();

    const size_t total_values_est = num_faces * static_cast<size_t>(total_time_steps);
    const size_t bytes_per_value = sizeof(double);
    const size_t arrays_count = 3; // direct, reflected, ir (illum is reconstructed as direct+reflected)
    const size_t bytes_needed = total_values_est * bytes_per_value * arrays_count;
    const double memory_required_gb_est = static_cast<double>(bytes_needed) / (1024.0 * 1024.0 * 1024.0);
    const double STREAM_THRESHOLD_GB = 10.0;
    bool use_streamed_thermal = using_chunked_storage && (memory_required_gb_est > STREAM_THRESHOLD_GB);

    // IMPORTANT: Radiation exchange requires full illumination arrays for iterative coupling
    // Cannot use streaming mode when radiation exchange is enabled
    if (enable_radiation_exchange && use_streamed_thermal) {
        std::cout << "Warning: Radiation exchange requires ~" << std::fixed << std::setprecision(1)
                  << memory_required_gb_est << " GB RAM" << std::endl;
        std::cout << "  Disabling streaming mode to extract full illumination arrays." << std::endl;
        use_streamed_thermal = false;
    }

    // IMPORTANT: GPU thermal solver requires full illumination arrays upfront
    // Cannot use streaming mode when GPU thermal is enabled
    if (config.getPropertyBool("use_gpu_thermal", false) && use_streamed_thermal) {
        std::cout << "Warning: GPU thermal solver requires ~" << std::fixed << std::setprecision(1)
                  << memory_required_gb_est << " GB RAM" << std::endl;
        std::cout << "  Disabling streaming mode to extract full illumination arrays." << std::endl;
        use_streamed_thermal = false;
    }

    // If using chunked storage, extract the data now (SKIP if we plan to stream)
    if (using_chunked_storage && !use_streamed_thermal) {
        std::cout << "Detected chunked storage. Extracting illumination data..." << std::endl;
        
        size_t total_values = num_faces * static_cast<size_t>(total_time_steps);
        
        // Resize arrays to hold the extracted data
        illumination_values.resize(total_values);
        direct_solar_values.resize(total_values);
        primary_reflected_values.resize(total_values);
        primary_ir_values.resize(total_values);
        
        // Extract from chunked storage
        size_t report_interval = std::max(size_t(1), num_faces / size_t(20));
        
        for (size_t f = 0; f < num_faces; f++) {
            if (f % report_interval == 0 || f == num_faces - 1) {
                std::cout << "\rExtracting data: " 
                        << std::fixed << std::setprecision(1) 
                        << (100.0 * (f + 1) / num_faces) << "%" << std::flush;
            }
            
            for (size_t t = 0; t < static_cast<size_t>(total_time_steps); t++) {
                size_t idx = f * static_cast<size_t>(total_time_steps) + t;
                
                // Use the per-face/per-time accessor methods
                direct_solar_values[idx] = illumination_model->getDirectSolarValue(f, t);
                primary_reflected_values[idx] = illumination_model->getPrimaryReflectedValue(f, t);
                primary_ir_values[idx] = illumination_model->getPrimaryIRValue(f, t);
                
                // Calculate total illumination (sum of direct and reflected)
                illumination_values[idx] = direct_solar_values[idx] + primary_reflected_values[idx];
            }
        }
        std::cout << std::endl;
    } else if (use_streamed_thermal) {
        std::cout << "Detected chunked storage. Skipping full illumination extraction (~"
                  << std::fixed << std::setprecision(1) << memory_required_gb_est
                  << " GB). Will stream illumination to the thermal solver." << std::endl;
    }

    // Release chunked storage after extraction — the flat arrays now own the data.
    // This frees ~180+ GB of duplicated illumination data that would otherwise
    // coexist with the flat arrays + surface_temps during the thermal solve.
    if (!use_streamed_thermal && using_chunked_storage) {
        illumination_model->releaseChunkedStorage();
        std::cout << "[Memory] Released chunked illumination storage after extraction" << std::endl;
    }

    // NOW verify the extracted data (only meaningful if we actually extracted)
    if (!use_streamed_thermal) {
        std::cout << "\nVerifying extracted illumination data:" << std::endl;
        std::cout << "  Expected array size: " << num_faces << " * " << total_time_steps 
                  << " = " << (num_faces * static_cast<size_t>(total_time_steps)) << std::endl;
        std::cout << "  Illumination array size: " << illumination_values.size() << std::endl;
        std::cout << "  Direct solar array size: " << direct_solar_values.size() << std::endl;
        std::cout << "  Times array size: " << times.size() << std::endl;
    } else {
        std::cout << "\nStreaming mode: full illumination arrays are intentionally not allocated in RAM." << std::endl;
        std::cout << "Values are accessed on-demand via getDirectSolarValue(f,t)/... during thermal solve." << std::endl;
    }

    // Check for non-zero data
    size_t non_zero_illumination = 0; // FIX: use size_t for very large counts
    double total_illumination = 0.0;
    for (size_t i = 0; i < illumination_values.size(); i++) {
        if (illumination_values[i] > 0.0) {
            non_zero_illumination++;
            total_illumination += illumination_values[i];
        }
    }

    if (non_zero_illumination == 0 && !illumination_values.empty()) {
        std::cerr << "WARNING: All illumination values are zero!" << std::endl;
    } else if (!illumination_values.empty()) {
        std::cout << "  Non-zero illumination values: " << non_zero_illumination << " out of " 
                << illumination_values.size() << std::endl;
        std::cout << "  Average non-zero illumination: " << (total_illumination / non_zero_illumination) << " W/mÂ²" << std::endl;
    }

    // DON'T filter illumination yet - run thermal model with full dataset first
    std::vector<std::vector<double>> surface_temps;

    // Time-varying indirect flux per face [face][timestep]
    // Declared at this scope so it's accessible for location export later
    std::vector<std::vector<double>> indirect_flux_series;

    if (run_thermal_mesh) {
        // Configure OpenMP and setup thermal model
        #ifdef _OPENMP
        if (thermal_threads > 0) {
            omp_set_num_threads(thermal_threads);
            std::cout << "Using " << thermal_threads << " OpenMP threads for thermal calculations" << std::endl;
        } else {
            std::cout << "Using " << omp_get_max_threads() << " OpenMP threads (auto)" << std::endl;
        }
        #else
        std::cout << "OpenMP not available, running in single-threaded mode" << std::endl;
        #endif
        
        // Set up thermal model configuration
        thermal::Config thermal_config;
        thermal_config.m = thermal_m;   // Layers in upper skin depth (CLI: --thermal-m)
        thermal_config.n = thermal_n;   // Layer growth factor (CLI: --thermal-n)
        thermal_config.b = thermal_b;   // Skin depths to bottom (CLI: --thermal-b)
        thermal_config.F = 0.5;
        thermal_config.NPERDAY = 24;
        thermal_config.NYEARSEQ = 2;
        thermal_config.sigma = 5.67e-8;
        thermal_config.chi = (chi_override >= 0) ? chi_override : 2.7;  // Hayne et al. (2017) default

        // Set thermal debug options from CLI (use local variables set during arg parsing)
        thermal_config.THERMAL_DEBUG = thermal_debug;
        thermal_config.DEBUG_LAT_THRESHOLD = thermal_debug_lat;
        thermal_config.DEBUG_OUTPUT_DIR = output_dir;


        // Set body name in configuration (both property names for compatibility)
        std::string body_name_for_config = illumination_model->getTargetBodyName();
        thermal_config.setProperty("target_body_name", body_name_for_config);
        thermal_config.setProperty("planet", body_name_for_config);  // GPU solver uses this property

        // Add albedo maps directory to config properties
        thermal_config.setProperty("albedo_maps_directory", albedo_maps_directory);
        thermal_config.setProperty("use_albedo_map", std::string(use_albedo_map ? "1" : "0"));

        // Flag generic bodies so albedo map code skips built-in body search
        if (using_generic_body) {
            thermal_config.setProperty("using_generic_body", "true");
        }

        // If using thermal model, add the maps to configuration
        if (!albedo_map_file.empty()) {
            thermal_config.setProperty("albedo_map", albedo_map_file);
        }
        if (!thermal_inertia_map_file.empty()) {
            thermal_config.setProperty("thermal_inertia_map", thermal_inertia_map_file);
        }
        
        // Add information about mesh type and resolution to thermal_config
        thermal_config.custom_properties["is_custom_mesh"] = (mesh_file.empty()) ? "0" : "1";
        thermal_config.custom_properties["mesh_resolution"] = std::to_string(mesh_resolution);

        // Pass explicit albedo flag to prevent auto-loading FITS albedo maps
        // when the user has explicitly set a uniform albedo value
        if (sim_config.thermophysical_albedo_explicit) {
            thermal_config.custom_properties["thermophysical_albedo_explicit"] = "1";
        }

        // Copy GPU thermal flag from CLI parsing config to thermal_config
        if (config.getPropertyBool("use_gpu_thermal", false)) {
            thermal_config.setProperty("use_gpu_thermal", true);
            std::cout << "[GPU Thermal] Flag detected - GPU thermal solver enabled" << std::endl;
        }

        // Copy SIMD batch flag from CLI parsing config to thermal_config.
        // The batch path is on by default (see dispatch-site defaults in this
        // file). --no-simd-batch explicitly stores "false" here, which gets
        // propagated so the dispatch picks up the override.
        if (config.custom_properties.count("use_simd_batch")) {
            thermal_config.setProperty(
                "use_simd_batch",
                config.getPropertyBool("use_simd_batch", true));
        }

        // Debug output to verify properties were set correctly
        if (verbose) {
            std::cout << "Thermal config properties:" << std::endl;
            std::cout << " - is_custom_mesh: " << ((mesh_file.empty()) ? "0" : "1") << std::endl;
            std::cout << " - mesh_resolution: " << mesh_resolution << std::endl;
            std::cout << " - Standard mesh: " << (mesh_file.empty() ? "YES" : "NO") << std::endl;
        }

        bool thermal_success = false;

        // Per-face geothermal heat flow mask
        std::vector<double> face_Qb;
        if (!sim_config.thermal.heat_flow_mask_file.empty()) {
            std::ifstream mask_file(sim_config.thermal.heat_flow_mask_file);
            if (!mask_file.is_open()) {
                std::cerr << "Error: Cannot open heat flow mask file: "
                          << sim_config.thermal.heat_flow_mask_file << std::endl;
                return 1;
            }
            face_Qb.resize(num_faces, planet.Qb);
            int mask_count = 0;
            double qb_min = std::numeric_limits<double>::max();
            double qb_max = 0.0;
            bool has_per_face_values = false;
            std::string line;
            while (std::getline(mask_file, line)) {
                if (line.empty() || line[0] == '#') continue;
                try {
                    std::istringstream iss(line);
                    int face_idx;
                    double qb_val;
                    if (!(iss >> face_idx)) continue;
                    if (face_idx < 0 || face_idx >= static_cast<int>(num_faces)) continue;
                    if (iss >> qb_val) {
                        // Format: "face_idx qb_value"
                        face_Qb[face_idx] = qb_val;
                        has_per_face_values = true;
                    } else {
                        // Format: "face_idx" only — use global mask value
                        face_Qb[face_idx] = sim_config.thermal.heat_flow_mask_value;
                        qb_val = sim_config.thermal.heat_flow_mask_value;
                    }
                    if (qb_val < qb_min) qb_min = qb_val;
                    if (qb_val > qb_max) qb_max = qb_val;
                    mask_count++;
                } catch (...) {
                    // Skip malformed lines
                }
            }
            if (has_per_face_values) {
                std::cout << "Loaded heat flow mask: " << mask_count
                          << " faces with Qb range " << qb_min << " - "
                          << qb_max << " W/m^2" << std::endl;
            } else {
                std::cout << "Loaded heat flow mask: " << mask_count << " faces with Qb = "
                          << sim_config.thermal.heat_flow_mask_value << " W/m^2" << std::endl;
            }
            std::cout << "  Remaining " << (num_faces - mask_count) << " faces use Qb = "
                      << planet.Qb << " W/m^2" << std::endl;
        }
        const std::vector<double>* face_Qb_ptr = face_Qb.empty() ? nullptr : &face_Qb;

        // Reuse illumination arrays from the outer-scope extraction if already
        // populated. This avoids a second ~180 GB allocation that would OOM.
        std::vector<double> direct_solar_values_full;
        std::vector<double> primary_reflected_values_full;
        std::vector<double> primary_ir_values_full;

        const bool already_extracted = !direct_solar_values.empty();

        // Detect chunked storage (arrays empty => chunked)
        bool using_chunked_storage = !already_extracted &&
            (illumination_model->getDirectSolarValues().empty() ||
             illumination_model->getPrimaryReflectedValues().empty() ||
             illumination_model->getPrimaryIRFlux().empty());

        const size_t total_values_est = num_faces * static_cast<size_t>(total_time_steps);
        const size_t bytes_per_value = sizeof(double);
        const size_t arrays_count = 3; // direct, reflected, ir
        const size_t bytes_needed = total_values_est * bytes_per_value * arrays_count;
        const double memory_required_gb_est = static_cast<double>(bytes_needed) / (1024.0 * 1024.0 * 1024.0);
        const double STREAM_THRESHOLD_GB = 10.0;
        bool use_streamed_thermal = using_chunked_storage && (memory_required_gb_est > STREAM_THRESHOLD_GB);

        // IMPORTANT: Radiation exchange requires full illumination arrays for iterative coupling
        // Cannot use streaming mode when radiation exchange is enabled
        if (enable_radiation_exchange && use_streamed_thermal) {
            std::cout << "Warning: Radiation exchange requires ~" << std::fixed << std::setprecision(1)
                      << memory_required_gb_est << " GB RAM" << std::endl;
            std::cout << "  Disabling streaming mode to extract full illumination arrays." << std::endl;
            use_streamed_thermal = false;
        }

        // IMPORTANT: GPU thermal solver requires full illumination arrays upfront
        // Cannot use streaming mode when GPU thermal is enabled
        bool use_gpu_thermal_check = thermal_config.getPropertyBool("use_gpu_thermal", false);
        if (use_gpu_thermal_check && use_streamed_thermal) {
            std::cout << "Warning: GPU thermal solver requires ~" << std::fixed << std::setprecision(1)
                      << memory_required_gb_est << " GB RAM" << std::endl;
            std::cout << "  Disabling streaming mode to extract full illumination arrays." << std::endl;
            use_streamed_thermal = false;
        }

        // If using chunked storage, extract the data now (SKIP if we plan to stream)
        if (using_chunked_storage && !use_streamed_thermal) {
            std::cout << "Detected chunked storage. Extracting illumination data..." << std::endl;
            
            size_t total_values = num_faces * static_cast<size_t>(total_time_steps);
            
            // Resize arrays to hold the extracted data
            illumination_values.resize(total_values);
            direct_solar_values.resize(total_values);
            primary_reflected_values.resize(total_values);
            primary_ir_values.resize(total_values);
            
            // Extract from chunked storage
            size_t report_interval = std::max(size_t(1), num_faces / size_t(20));
            
            for (size_t f = 0; f < num_faces; f++) {
                if (f % report_interval == 0 || f == num_faces - 1) {
                    std::cout << "\rExtracting data: " 
                            << std::fixed << std::setprecision(1) 
                            << (100.0 * (f + 1) / num_faces) << "%" << std::flush;
                }
                for (size_t t = 0; t < static_cast<size_t>(total_time_steps); t++) {
                    size_t idx = f * static_cast<size_t>(total_time_steps) + t;
                    direct_solar_values[idx]      = illumination_model->getDirectSolarValue(f, t);
                    primary_reflected_values[idx] = illumination_model->getPrimaryReflectedValue(f, t);
                    primary_ir_values[idx]        = illumination_model->getPrimaryIRValue(f, t);
                    illumination_values[idx]      = direct_solar_values[idx] + primary_reflected_values[idx];
                }
            }
            std::cout << std::endl;
        } else if (use_streamed_thermal) {
            std::cout << "Detected chunked storage. Skipping full illumination extraction (~"
                    << std::fixed << std::setprecision(1) << memory_required_gb_est
                    << " GB). Will stream illumination to the thermal solver." << std::endl;
        }

        // Verify extracted data (only meaningful if arrays exist)
        if (!use_streamed_thermal) {
            std::cout << "\nVerifying extracted illumination data:" << std::endl;
            std::cout << "  Expected array size: " << num_faces << " * " << total_time_steps 
                    << " = " << (num_faces * static_cast<size_t>(total_time_steps)) << std::endl;
            std::cout << "  Illumination array size: " << illumination_values.size() << std::endl;
            std::cout << "  Direct solar array size: " << direct_solar_values.size() << std::endl;
            std::cout << "  Times array size: " << times.size() << std::endl;
        } else {
            std::cout << "\nStreaming mode: full illumination arrays are intentionally not allocated in RAM." << std::endl;
            std::cout << "Values are accessed on-demand via getDirectSolarValue(f,t)/... during thermal solve." << std::endl;
        }

        // Non-zero check (safe for empty vectors)
        size_t non_zero_illumination = 0;
        double total_illumination = 0.0;
        for (size_t i = 0; i < illumination_values.size(); i++) {
            if (illumination_values[i] > 0.0) {
                non_zero_illumination++;
                total_illumination += illumination_values[i];
            }
        }
        if (non_zero_illumination == 0 && !illumination_values.empty()) {
            std::cerr << "WARNING: All illumination values are zero!" << std::endl;
        } else if (!illumination_values.empty()) {
            std::cout << "  Non-zero illumination values: " << non_zero_illumination << " out of " 
                    << illumination_values.size() << std::endl;
            std::cout << "  Average non-zero illumination: " << (total_illumination / non_zero_illumination) << " W/mÂ²" << std::endl;
        }
        
        // Execute thermal path
        if (using_chunked_storage && use_streamed_thermal) {
            // On-demand getters from illumination model (no large buffers)
            FluxFn get_direct = [&](size_t f, int t) {
                return illumination_model->getDirectSolarValue(f, t);
            };
            FluxFn get_reflected = [&](size_t f, int t) {
                return illumination_model->getPrimaryReflectedValue(f, t);
            };
            FluxFn get_ir = [&](size_t f, int t) {
                return illumination_model->getPrimaryIRValue(f, t);
            };

            // Streamed thermal: no face×time allocations.
            // Pass planet.face_H_overrides if loaded so each face gets its
            // own H-driven rho(z), k(z) profile.
            const std::vector<double>* face_H_ptr =
                planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
            bool thermal_success_streamed = runThermalModelsForAllFacesStreamed(
                mesh,
                times,
                total_time_steps,
                thermal_config,
                default_visible_albedo,
                /*ir_albedo*/0.02,
                thermal_threads,
                get_direct,
                get_reflected,
                get_ir,
                &surface_temps,
                face_Qb_ptr,
                face_H_ptr
            );

            if (!thermal_success_streamed) {
                std::cerr << "Streamed thermal mesh simulation failed" << std::endl;
                return 1;
            }
            thermal_success = true;
        } else if (using_chunked_storage) {
            // Smaller dataset: extract to arrays and run standard path
            size_t num_faces_local = mesh->faces.size();
            size_t total_values    = num_faces_local * static_cast<size_t>(total_time_steps);
            direct_solar_values_full.resize(total_values);
            primary_reflected_values_full.resize(total_values);
            primary_ir_values_full.resize(total_values);

            size_t report_interval = std::max<size_t>(1, num_faces_local / 20);
            for (size_t f = 0; f < num_faces_local; f++) {
                if (f % report_interval == 0 || f == num_faces_local - 1) {
                    std::cout << "\rExtracting data from chunked storage: " 
                            << std::fixed << std::setprecision(1) 
                            << (100.0 * f / num_faces_local) << "%" << std::flush;
                }
                for (size_t t = 0; t < static_cast<size_t>(total_time_steps); t++) {
                    size_t idx = f * static_cast<size_t>(total_time_steps) + t;
                    direct_solar_values_full[idx]      = illumination_model->getDirectSolarValue(f, t);
                    primary_reflected_values_full[idx] = illumination_model->getPrimaryReflectedValue(f, t);
                    primary_ir_values_full[idx]        = illumination_model->getPrimaryIRValue(f, t);
                }
            }
            std::cout << std::endl;

            // Check which thermal solver to use
            bool use_simd_batch = thermal_config.getPropertyBool("use_simd_batch", true);
            bool use_gpu_thermal = thermal_config.getPropertyBool("use_gpu_thermal", false);

            // Schorghofer & Taylor (2007) accumulator pointer. Null when the
            // user has not requested ice-stability output → zero overhead path.
            ice::IceStabilityAccumulator* ice_acc_ptr_a =
                sim_config.thermal.compute_ice_stability ? &ice_stability_accum : nullptr;

            if (use_gpu_thermal) {
                const std::vector<double>* face_H_ptr_g =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                // When loading a thermal-state checkpoint, skip equilibration
                // entirely — the loaded T(z) is already equilibrated by the
                // previous chunk.
                int eq_steps_eff = equilibration_steps;
                if (!sim_config.thermal.initial_state_file.empty()) {
                    if (eq_steps_eff > 0) {
                        std::cout << "[GPU Thermal] thermal.initial_state_file is set; "
                                     "forcing equilibration_steps=0 (was "
                                  << eq_steps_eff << ")" << std::endl;
                    }
                    eq_steps_eff = 0;
                }
                thermal_success = runThermalModelsForAllFaces_GPU_Persistent(
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    eq_steps_eff,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    face_Qb_ptr,
                    face_H_ptr_g,
                    ice_acc_ptr_a,
                    sim_config.thermal.initial_state_file,
                    sim_config.thermal.final_state_file,
                    sim_config.thermal.gpu_devices,
                    sim_config.thermal.gpu_device_ids,
                    sim_config.thermal.gpu_batch_size,
                    sim_config.thermal.gpu_inter_batch_pause_ms
                );
            } else if (use_simd_batch) {
                const std::vector<double>* face_H_ptr_b =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                thermal_success = runThermalModelsForAllFacesBatchSoA(
                    output_dir,
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    equilibration_steps,
                    total_time_steps,
                    thermal_threads,
                    face_Qb_ptr,
                    face_H_ptr_b
                );
            } else {
                std::cout << "[Thermal] Using CPU parallel solver with " << thermal_threads << " threads" << std::endl;
                thermal_success = runThermalModelsForAllFaces(
                    output_dir,
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    equilibration_steps,
                    total_time_steps,
                    thermal_threads,
                    &planet,  // Pass planet for incidence-dependent albedo
                    face_Qb_ptr
                );
            }

            // Validate the returned temperature matrix shape (faces x total_time_steps)
            if (thermal_success) {
                if (!validateTemperatureMatrix(surface_temps, mesh->faces.size(), total_time_steps)) {
                    std::cerr << "[FATAL] Invalid temperature matrix shape after non-streamed thermal solve; aborting.\n";
                    return 1;
                }
            }

            if (!thermal_success) {
                std::cerr << "Error during thermal mesh simulation (extracted arrays path)" << std::endl;
                return 1;
            }
        } else {
            // Not chunked (or already extracted in outer scope): use existing arrays.
            // Use swap (not move/copy) to temporarily transfer ownership to _full
            // for the thermal solve. After the solve, swap back so the outer-scope
            // vectors are restored for the export phase.
            // For both chunked-already-extracted and non-chunked cases, the outer-scope
            // arrays have the data. Reference them for the thermal solver.
            // Use swap to avoid a copy — the swap-back below restores them for export.
            direct_solar_values_full.swap(direct_solar_values);
            primary_reflected_values_full.swap(primary_reflected_values);
            primary_ir_values_full.swap(primary_ir_values);

            // Check which thermal solver to use
            bool use_simd_batch = thermal_config.getPropertyBool("use_simd_batch", true);
            bool use_gpu_thermal = thermal_config.getPropertyBool("use_gpu_thermal", false);

            // Schorghofer & Taylor (2007) accumulator pointer. Null when the
            // user has not requested ice-stability output → zero overhead path.
            ice::IceStabilityAccumulator* ice_acc_ptr_b =
                sim_config.thermal.compute_ice_stability ? &ice_stability_accum : nullptr;

            if (use_gpu_thermal) {
                const std::vector<double>* face_H_ptr_g =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                int eq_steps_eff = equilibration_steps;
                if (!sim_config.thermal.initial_state_file.empty()) {
                    if (eq_steps_eff > 0) {
                        std::cout << "[GPU Thermal] thermal.initial_state_file is set; "
                                     "forcing equilibration_steps=0 (was "
                                  << eq_steps_eff << ")" << std::endl;
                    }
                    eq_steps_eff = 0;
                }
                thermal_success = runThermalModelsForAllFaces_GPU_Persistent(
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    eq_steps_eff,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    face_Qb_ptr,
                    face_H_ptr_g,
                    ice_acc_ptr_b,
                    sim_config.thermal.initial_state_file,
                    sim_config.thermal.final_state_file,
                    sim_config.thermal.gpu_devices,
                    sim_config.thermal.gpu_device_ids,
                    sim_config.thermal.gpu_batch_size,
                    sim_config.thermal.gpu_inter_batch_pause_ms
                );
            } else if (use_simd_batch) {
                const std::vector<double>* face_H_ptr_b =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                thermal_success = runThermalModelsForAllFacesBatchSoA(
                    output_dir,
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    equilibration_steps,
                    total_time_steps,
                    thermal_threads,
                    face_Qb_ptr,
                    face_H_ptr_b
                );
            } else {
                std::cout << "[Thermal] Using CPU parallel solver with " << thermal_threads << " threads" << std::endl;
                thermal_success = runThermalModelsForAllFaces(
                    output_dir,
                    mesh,
                    direct_solar_values_full,
                    primary_reflected_values_full,
                    primary_ir_values_full,
                    times,
                    total_time_steps,
                    thermal_config,
                    default_visible_albedo,
                    0.02,
                    &surface_temps,
                    equilibration_steps,
                    total_time_steps,
                    thermal_threads,
                    &planet,  // Pass planet for incidence-dependent albedo
                    face_Qb_ptr
                );
            }

            // Validate the returned temperature matrix shape (faces x total_time_steps)
            if (thermal_success) {
                if (!validateTemperatureMatrix(surface_temps, mesh->faces.size(), total_time_steps)) {
                    std::cerr << "[FATAL] Invalid temperature matrix shape after non-streamed thermal solve; aborting.\n";
                    return 1;
                }
            }

            if (!thermal_success) {
                std::cerr << "Error during thermal mesh simulation" << std::endl;
                return 1;
            }

            // Swap illumination data back to outer scope for the export phase.
            direct_solar_values.swap(direct_solar_values_full);
            primary_reflected_values.swap(primary_reflected_values_full);
            primary_ir_values.swap(primary_ir_values_full);
        }

        if (!thermal_success && surface_temps.empty()) {
            std::cerr << "Error during thermal mesh simulation" << std::endl;
            return 1;
        }

        // Free the combined illumination_values array (~61 GB for ico9×1560).
        // The export filter reconstructs it from direct+reflected if needed
        // (filterFaceMajorStable line: il = illum_full.empty() ? (ds+rf) : illum_full[si]).
        // The per-component arrays (direct_solar, reflected, ir) must stay alive for export.
        {
            std::vector<double>().swap(illumination_values);
            std::cout << "[Memory] Released combined illumination array ("
                      << "keeping per-component arrays for export)" << std::endl;
        }

        // If the GPU solver used two-phase execution, surface_temps contains only
        // the output_steps (not eq + output). Adjust the bookkeeping so downstream
        // export code (filterFaceMajorStable) doesn't try to skip past data that
        // isn't there.
        // surface_temps has total_time_steps entries per face. In two-phase mode,
        // the eq slots (0..equilibration_steps-1) are zero-filled since those
        // temperatures were evolved on GPU but not downloaded. The existing
        // export filter skips them via skip_eq = equilibration_steps.
        // IMPORTANT: do NOT modify total_time_steps or equilibration_steps here —
        // they define the stride for indexing into the illumination arrays.

        // ========================================================================
        // RADIATION EXCHANGE COUPLING (Optional)
        // ========================================================================
        if (enable_radiation_exchange && thermal_success) {
            std::cout << "\n========== Radiation Exchange Coupling ==========" << std::endl;

            // Check memory requirements for radiation coupling
            const double rad_memory_gb = (num_faces * total_time_steps * 24.0) / (1024.0 * 1024.0 * 1024.0);
            if (rad_memory_gb > 16.0) {
                std::cerr << "Warning: Radiation exchange requires ~" << std::fixed << std::setprecision(1)
                          << rad_memory_gb << " GB RAM" << std::endl;
            }

            // Configure radiation exchange
            radiation::RadiationConfig rad_config;
            rad_config.enabled = true;
            rad_config.use_hmatrix = rad_use_hmatrix;
            rad_config.use_gpu = rad_use_gpu;
            rad_config.verbose = rad_verbose;
            rad_config.max_iterations = 100;           // Radiosity solver iterations
            rad_config.convergence_tolerance = 0.001;  // Radiosity solver tolerance

            // ACA configuration (Adaptive Cross Approximation - much faster)
            rad_config.use_aca = rad_use_aca;
            rad_config.aca_tolerance = rad_aca_tolerance;
            rad_config.aca_max_rank = rad_aca_max_rank;
            rad_config.hmatrix_threads = rad_hmatrix_threads;

            // Adaptive neighbor selection configuration
            rad_config.adaptive_neighbors = rad_adaptive_neighbors;
            rad_config.captured_energy_target = rad_captured_energy_target;
            rad_config.max_neighbors = rad_max_neighbors;
            rad_config.min_neighbors = rad_min_neighbors;
            rad_config.use_cluster_cull = rad_use_cluster_cull;
            rad_config.force_hmatrix = rad_force_hmatrix;
            rad_config.use_monte_carlo_vf = rad_use_monte_carlo_vf;
            rad_config.mc_rays_per_face = rad_mc_rays_per_face;
            if (rad_max_neighbors_override > 0) {
                rad_config.max_neighbors = rad_max_neighbors_override;
                rad_config.adaptive_neighbors = false;  // Use fixed if override specified
            }

            // Create radiation exchange manager
            radiation::RadiationExchangeManager rad_manager(mesh, rad_config);

            // Try to load pre-computed view factors. Distinguish "file missing" from
            // "file present but failed to load" so cache-miss bugs are visible.
            if (!rad_view_factor_file.empty()) {
                std::error_code exists_ec;
                bool file_exists =
                    std::filesystem::exists(rad_view_factor_file, exists_ec);
                if (file_exists) {
                    if (rad_manager.loadViewFactors(rad_view_factor_file)) {
                        std::cout << "Loaded view factors from: " << rad_view_factor_file << std::endl;
                    } else {
                        std::cerr << "WARN: View factor file exists but failed to load: "
                                  << rad_view_factor_file
                                  << " — recomputing." << std::endl;
                    }
                } else {
                    std::cout << "View factor file does not exist at: "
                              << rad_view_factor_file
                              << " — computing from scratch." << std::endl;
                }
            }

            // Set emissivity from planet properties
            rad_manager.setEmissivity(planet.emissivity);

            // Prepare illumination series for face classification
            std::vector<std::vector<double>> illumination_series(num_faces);
            for (size_t f = 0; f < num_faces; f++) {
                illumination_series[f].resize(total_time_steps);
                for (int t = 0; t < total_time_steps; t++) {
                    size_t idx = f * static_cast<size_t>(total_time_steps) + t;
                    illumination_series[f][t] = direct_solar_values[idx];
                }
            }

            // Prepare (classify faces, compute view factors if needed)
            if (!rad_manager.prepare(illumination_series)) {
                std::cout << "No faces need radiation exchange - skipping coupling loop" << std::endl;
            } else {
                std::cout << "Faces requiring radiation exchange: " << rad_manager.numExchangeFaces()
                          << " / " << num_faces << std::endl;

                // Save view factors if computed (not loaded)
                // Note: Check getHMatrixViewFactors() directly since H-matrix may have been
                // auto-enabled by the manager for large meshes, even if --rad-hmatrix wasn't passed
                if (!rad_view_factor_file.empty() && !rad_manager.hasLoadedViewFactors()) {
                    std::error_code mkdir_ec;
                    std::filesystem::create_directories(
                        std::filesystem::path(rad_view_factor_file).parent_path(), mkdir_ec);

                    bool wrote = false;
                    if (rad_manager.getHMatrixViewFactors()) {
                        wrote = rad_manager.getHMatrixViewFactors()->writeBinary(rad_view_factor_file);
                    } else {
                        wrote = rad_manager.getViewFactorMatrix().writeBinary(rad_view_factor_file);
                    }
                    if (wrote) {
                        std::cout << "Saved view factors to: " << rad_view_factor_file << std::endl;
                    } else {
                        std::cerr << "ERROR: Failed to save view factors to: "
                                  << rad_view_factor_file
                                  << " (check path, permissions, disk space)" << std::endl;
                    }
                }

                std::cout << "\nStarting radiation-thermal coupling iterations..." << std::endl;
                std::cout << "  Max iterations: " << rad_max_coupling_iter << std::endl;
                std::cout << "  Temperature tolerance: " << rad_coupling_temp_tol << " K" << std::endl;

                // Store previous iteration temperatures for convergence check
                std::vector<std::vector<double>> prev_temps = surface_temps;

                // Time-varying indirect flux per face [face][timestep]
                // This is the key fix: indirect flux varies with time as wall temperatures change
                // (indirect_flux_series declared at outer scope for location export)
                indirect_flux_series.resize(num_faces);
                for (size_t f = 0; f < num_faces; f++) {
                    indirect_flux_series[f].resize(total_time_steps, 0.0);
                }

                // Load per-face albedos for radiation coupling re-runs
                // Must match the albedos used in the initial thermal solve
                std::vector<double> face_albedos_rad(num_faces, planet.albedo);
                {
                    std::string body_name_rad = thermal_config.getProperty("target_body_name");
                    if (!body_name_rad.empty() && std::islower(body_name_rad[0])) {
                        body_name_rad[0] = std::toupper(body_name_rad[0]);
                    }
                    int mesh_res_rad = 5;
                    try {
                        auto it = thermal_config.custom_properties.find("mesh_resolution");
                        if (it != thermal_config.custom_properties.end()) {
                            mesh_res_rad = std::stoi(it->second);
                        }
                    } catch (...) {}

                    std::string albedo_filename = body_name_rad + "_Albedo_res" + std::to_string(mesh_res_rad) + ".bin";
                    std::string albedo_dir = thermal_config.getProperty("albedo_maps_directory");
                    if (albedo_dir.empty()) albedo_dir = "data/albedo_maps";
                    if (!albedo_dir.empty() && albedo_dir.back() != '/' && albedo_dir.back() != '\\') {
                        albedo_dir += '/';
                    }

                    std::vector<std::string> albedo_search_paths = {
                        albedo_dir + albedo_filename,
                        "../data/albedo_maps/" + albedo_filename,
                        "data/albedo_maps/" + albedo_filename
                    };

                    for (const auto& apath : albedo_search_paths) {
                        auto loaded = importAlbedoMapFromFile(apath);
                        if (!loaded.empty() && loaded.size() == num_faces) {
                            face_albedos_rad = std::move(loaded);
                            std::cout << "  Loaded per-face albedos for radiation coupling from: " << apath << std::endl;
                            break;
                        }
                    }
                }

                for (int iter = 0; iter < rad_max_coupling_iter; iter++) {
                    std::cout << "\n--- Radiation Iteration " << (iter + 1) << " ---" << std::endl;

                    // Step 1: Compute indirect flux at EVERY timestep
                    // This captures time-varying thermal emission from illuminated walls
                    double max_indirect = 0.0;
                    double total_indirect_sum = 0.0;

                    thermal::ProgressBar flux_progress(total_time_steps, "Computing indirect flux", 50, true);

                    // Sequential timestep loop - each solve uses all threads for H-matrix matvec
                    // The H-matrix matvec is parallelized over ~2M blocks internally
                    for (int t = 0; t < total_time_steps; t++) {
                        // Extract temperatures at timestep t
                        std::vector<double> temps_t(num_faces);
                        std::vector<double> solar_t(num_faces);

                        #pragma omp parallel for schedule(static)
                        for (size_t f = 0; f < num_faces; f++) {
                            temps_t[f] = surface_temps[f][t];
                            solar_t[f] = direct_solar_values[f * total_time_steps + t];
                        }

                        // Compute indirect flux at this timestep
                        std::vector<double> indirect_t = rad_manager.computeIndirectFlux(temps_t, solar_t);

                        // Store time-varying indirect flux and accumulate statistics
                        #pragma omp parallel for schedule(static) reduction(max:max_indirect) reduction(+:total_indirect_sum)
                        for (size_t f = 0; f < num_faces; f++) {
                            indirect_flux_series[f][t] = indirect_t[f];
                            if (indirect_t[f] > max_indirect) {
                                max_indirect = indirect_t[f];
                            }
                            total_indirect_sum += indirect_t[f];
                        }

                        if ((t + 1) % 10 == 0 || t == total_time_steps - 1) {
                            flux_progress.update(t + 1);
                        }
                    }
                    flux_progress.finish();

                    double mean_indirect = total_indirect_sum / (num_faces * total_time_steps);
                    std::cout << "  Max indirect flux: " << max_indirect << " W/m2" << std::endl;
                    std::cout << "  Mean indirect flux: " << mean_indirect << " W/m2" << std::endl;

                    // Diagnostic: compare indirect flux for high vs low topographic elevation
                    // Uses radial distance from body center as elevation proxy (works on spherical bodies)
                    // Low-elevation faces (valleys/craters) should receive MORE indirect flux
                    {
                        std::vector<double> radii(num_faces);
                        for (size_t f = 0; f < num_faces; f++) {
                            const auto& c = mesh->face_centers[f];
                            radii[f] = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
                        }
                        std::vector<double> sorted_radii = radii;
                        std::sort(sorted_radii.begin(), sorted_radii.end());
                        double r_median = sorted_radii[num_faces / 2];

                        double sum_high = 0.0, sum_low = 0.0;
                        int n_high = 0, n_low = 0;
                        for (size_t f = 0; f < num_faces; f++) {
                            double max_face_ind = 0.0;
                            for (int t = 0; t < total_time_steps; t++) {
                                max_face_ind = std::max(max_face_ind, indirect_flux_series[f][t]);
                            }
                            if (radii[f] > r_median) {
                                sum_high += max_face_ind; n_high++;
                            } else {
                                sum_low += max_face_ind; n_low++;
                            }
                        }
                        if (n_high > 0 && n_low > 0) {
                            std::cout << "  Avg max indirect (high elev): "
                                      << std::fixed << std::setprecision(4) << (sum_high / n_high) << " W/m2" << std::endl;
                            std::cout << "  Avg max indirect (low elev):  "
                                      << std::fixed << std::setprecision(4) << (sum_low / n_low) << " W/m2" << std::endl;
                        }
                    }

                    // Step 2: Identify faces that need thermal re-run
                    // Include faces where max indirect flux over time exceeds threshold
                    std::vector<int> faces_to_rerun;
                    for (size_t f = 0; f < num_faces; f++) {
                        double max_face_indirect = 0.0;
                        for (int t = 0; t < total_time_steps; t++) {
                            if (indirect_flux_series[f][t] > max_face_indirect) {
                                max_face_indirect = indirect_flux_series[f][t];
                            }
                        }
                        // Include face if max indirect flux over time exceeds threshold
                        if (max_face_indirect > rad_coupling_flux_tol) {
                            faces_to_rerun.push_back(static_cast<int>(f));
                        }
                    }

                    std::cout << "  Faces with significant indirect flux: " << faces_to_rerun.size()
                              << " / " << num_faces << std::endl;

                    if (faces_to_rerun.empty()) {
                        std::cout << "  No faces need re-run. Converged." << std::endl;
                        break;
                    }

                    // Check convergence from previous iteration
                    if (iter > 0) {
                        double max_temp_change = 0.0;
                        int faces_changed = 0;

                        for (size_t f = 0; f < num_faces; f++) {
                            for (int t = 0; t < total_time_steps; t++) {
                                double dT = std::abs(surface_temps[f][t] - prev_temps[f][t]);
                                if (dT > max_temp_change) max_temp_change = dT;
                                if (dT > rad_coupling_temp_tol) faces_changed++;
                            }
                        }

                        double face_frac = static_cast<double>(faces_changed) / (num_faces * total_time_steps);
                        std::cout << "  Max temperature change: " << max_temp_change << " K" << std::endl;
                        std::cout << "  Samples still changing: " << std::scientific << std::setprecision(2)
                                  << (face_frac * 100) << "%" << std::fixed << std::endl;

                        // Converge when BOTH conditions are met:
                        // 1. Max temperature change is below threshold
                        // 2. Fraction of samples still changing is below threshold
                        if (max_temp_change < rad_coupling_temp_tol && face_frac < rad_coupling_face_frac) {
                            std::cout << "  Converged!" << std::endl;
                            break;
                        }
                    }

                    // Save previous temps for next iteration comparison
                    prev_temps = surface_temps;

                    // Step 3: Re-run thermal for selected faces with TIME-VARYING indirect flux
                    std::cout << "  Re-running thermal model for " << faces_to_rerun.size() << " faces..." << std::endl;

                    // Build flux arrays with TIME-VARYING indirect contribution for all selected faces
                    // Use per-face albedos to match the initial thermal solve
                    std::vector<std::vector<double>> combined_flux(faces_to_rerun.size());
                    for (size_t idx = 0; idx < faces_to_rerun.size(); idx++) {
                        int f = faces_to_rerun[idx];
                        double face_albedo = face_albedos_rad[f];
                        combined_flux[idx].resize(total_time_steps);
                        for (int t = 0; t < total_time_steps; t++) {
                            size_t flux_idx = static_cast<size_t>(f) * total_time_steps + t;
                            // Direct + reflected solar (absorbed) with per-face albedo
                            double solar_absorbed = (direct_solar_values[flux_idx] +
                                                    primary_reflected_values[flux_idx]) * (1.0 - face_albedo);
                            // Add absorbed IR
                            double ir_absorbed = primary_ir_values[flux_idx] * planet.emissivity;
                            // Add TIME-VARYING indirect radiation flux at this timestep
                            combined_flux[idx][t] = solar_absorbed + ir_absorbed + indirect_flux_series[f][t];
                        }
                    }

                    // Check if GPU thermal is enabled
                    bool use_gpu_thermal_rerun = thermal_config.getPropertyBool("use_gpu_thermal", false);

                    if (use_gpu_thermal_rerun && faces_to_rerun.size() >= 100) {
                        // Use GPU thermal solver for radiation coupling re-runs
                        std::cout << "  Using GPU thermal solver..." << std::endl;
                        bool gpu_success = runThermalForSelectedFaces_GPU(
                            mesh,
                            faces_to_rerun,
                            combined_flux,
                            times,
                            total_time_steps,
                            thermal_config,
                            surface_temps
                        );
                        if (!gpu_success) {
                            std::cerr << "  GPU thermal solver failed, falling back to CPU" << std::endl;
                            use_gpu_thermal_rerun = false;
                        }
                    }

                    if (!use_gpu_thermal_rerun) {
                        // CPU path: OpenMP parallel loop
                        thermal::ProgressBar rerun_progress(faces_to_rerun.size(), "Thermal re-run", 50, true);
                        std::atomic<int> rerun_completed{0};

                        #pragma omp parallel for schedule(dynamic, 10) num_threads(thermal_threads > 0 ? thermal_threads : omp_get_max_threads())
                        for (size_t idx = 0; idx < faces_to_rerun.size(); idx++) {
                            int f = faces_to_rerun[idx];

                            // Run thermal model using the standard function
                            double latitude = mesh->latitudes[f];
                            std::vector<double> new_temps = runThermalModelWithMatchedTimes(
                                combined_flux[idx],
                                times,
                                latitude,
                                thermal_config,
                                equilibration_steps
                            );

                            // Store results
                            #pragma omp critical
                            {
                                if (!new_temps.empty()) {
                                    surface_temps[f] = std::move(new_temps);
                                }
                            }

                            int cur = ++rerun_completed;
                            if (cur % 100 == 0) {
                                #pragma omp critical
                                rerun_progress.update(cur);
                            }
                        }
                        rerun_progress.finish();
                    }
                }

                // Print final statistics
                auto stats = rad_manager.getStatistics();
                std::cout << "\n=== Radiation Exchange Complete ===" << std::endl;
                std::cout << "  Mean indirect flux: " << stats.mean_indirect_flux << " W/m2" << std::endl;
                std::cout << "  Max indirect flux: " << stats.max_indirect_flux << " W/m2" << std::endl;
            }

            std::cout << "=================================================\n" << std::endl;
        }
    }

    // Full arrays exist iff not streaming or we explicitly extracted
    bool have_full_illum_arrays =
        !(using_chunked_storage && use_streamed_thermal) &&
        !direct_solar_values.empty() &&
        !primary_reflected_values.empty() &&
        !primary_ir_values.empty();
    // Independent flag: did the stride-based export already write per-step files?
    // (Distinct from have_full_illum_arrays so the stats path can still use stats_fd.)
    bool stride_export_already_done = false;

    // ===== INTEGRATION (statistics) interval =====
    int stats_start_step = 0;
    int stats_steps = 0;
    FilteredData stats_fd;
    {
        const double steps_per_day_full = total_simulation_days > 0.0
            ? static_cast<double>(total_time_steps) / total_simulation_days
            : 0.0;

        double integ_days_used = (integration_days > 0.0 ? integration_days : output_days);
        int integration_steps = (integration_steps_cli > 0)
            ? integration_steps_cli
            : static_cast<int>(std::llround(integ_days_used * steps_per_day_full));
        if (integration_steps < 1) integration_steps = 1;

        int integration_offset_steps = (integration_offset_steps_cli >= 0)
            ? integration_offset_steps_cli
            : static_cast<int>(std::llround(std::max(0.0, integration_offset_days) * steps_per_day_full));
        if (integration_offset_steps < 0) integration_offset_steps = 0;

        stats_start_step = equilibration_steps + integration_offset_steps;
        if (stats_start_step >= total_time_steps) stats_start_step = std::max(0, total_time_steps - 1);

        stats_steps = std::min(integration_steps, std::max(0, total_time_steps - stats_start_step));
        if (stats_steps < 1) stats_steps = std::min(1, total_time_steps);

        std::cout << "\n[Integration] Statistics interval\n"
                << "  Start (abs step): " << stats_start_step << "\n"
                << "  Duration (steps): " << stats_steps << "\n";

        if (have_full_illum_arrays) {
            // Normal path: slice all arrays
            stats_fd = filterFaceMajorStable(
                illumination_values,
                direct_solar_values,
                primary_reflected_values,
                primary_ir_values,
                surface_temps,
                times,
                time_strings,
                sun_positions,
                primary_positions,
                num_faces,
                total_time_steps,
                stats_start_step,
                stats_steps
            );
        } else {
            // Meta-only slice for stats; illumination arrays remain empty (skipped)
            stats_fd.times.assign(times.begin() + stats_start_step, times.begin() + stats_start_step + stats_steps);
            stats_fd.time_strings.assign(time_strings.begin() + stats_start_step, time_strings.begin() + stats_start_step + stats_steps);
            stats_fd.sun_pos.assign(sun_positions.begin() + stats_start_step, sun_positions.begin() + stats_start_step + stats_steps);
            stats_fd.primary_pos.assign(primary_positions.begin() + stats_start_step, primary_positions.begin() + stats_start_step + stats_steps);

            // Slice temperatures for the statistics interval (only if thermal was run)
            if (!surface_temps.empty() && surface_temps.size() >= num_faces) {
                stats_fd.temps.resize(num_faces);
                #pragma omp parallel for schedule(static)
                for (size_t f = 0; f < num_faces; ++f) {
                    const auto& src = surface_temps[f];
                    if (src.size() >= static_cast<size_t>(total_time_steps) &&
                        stats_start_step + stats_steps <= static_cast<int>(src.size())) {
                        stats_fd.temps[f].assign(
                            src.begin() + stats_start_step,
                            src.begin() + stats_start_step + stats_steps
                        );
                    }
                }
            }
        }
    }

    // ========== BEGIN DATA FILTERING SECTION ==========

    // Filtered datasets (output interval)
    std::vector<double> filtered_illumination_values;
    std::vector<double> filtered_direct_solar_values;
    std::vector<double> filtered_primary_reflected_values;
    std::vector<double> filtered_primary_ir_values;
    std::vector<std::vector<double>> filtered_surface_temps;
    std::vector<double> filtered_times;
    std::vector<std::string> filtered_time_strings;
    std::vector<std::array<double,3>> filtered_sun_positions;
    std::vector<std::array<double,3>> filtered_primary_positions;

    {
        const size_t face_count = num_faces;
        const int total_ts      = total_time_steps;
        int skip_eq             = equilibration_steps;
        int requested_out_ts    = time_steps;

        if (skip_eq + requested_out_ts > total_ts) {
            requested_out_ts = std::max(0, total_ts - skip_eq);
        }

        // Diagnostics only if arrays exist
        if (have_full_illum_arrays && !illumination_values.empty() && !direct_solar_values.empty()) {
            diagnosticsIlluminationCoverage(
                illumination_values,
                direct_solar_values,
                face_count,
                total_ts,
                "Pre-filter (full series)",
                verbose);
        }

        // Illumination arrays are available for the export filter.
        if (have_full_illum_arrays) {
            // For large arrays (>10 GB), filter in-place to avoid doubling memory.
            // filterFaceMajorStable allocates full copies which would OOM for ico9.
            const double array_gb = static_cast<double>(face_count) * total_ts * 8.0 / (1024.0*1024.0*1024.0);

            if (array_gb > 10.0 && skip_eq > 0) {
                // Large arrays: skip the filter entirely. Instead, pass the
                // original arrays directly to the export function with the
                // correct stride (total_ts) and offset (skip_eq). The export
                // reads `arr[f * stride + offset + t]` for output timestep t.
                // This avoids allocating any intermediate filtered copies.
                std::cout << "[Memory] Large arrays (~" << std::fixed
                          << std::setprecision(0) << array_gb * 3
                          << " GB); exporting with stride=" << total_ts
                          << " offset=" << skip_eq << " (no intermediate copy)"
                          << std::endl;

                // Slice only the meta arrays (tiny)
                filtered_times.assign(times.begin() + skip_eq,
                                      times.begin() + skip_eq + requested_out_ts);
                filtered_time_strings.assign(time_strings.begin() + skip_eq,
                                             time_strings.begin() + skip_eq + requested_out_ts);
                filtered_sun_positions.assign(sun_positions.begin() + skip_eq,
                                              sun_positions.begin() + skip_eq + requested_out_ts);
                filtered_primary_positions.assign(primary_positions.begin() + skip_eq,
                                                   primary_positions.begin() + skip_eq + requested_out_ts);

                // The illumination and temperature arrays keep their original
                // layout (stride = total_ts). We pass skip_eq and total_ts to
                // the export function so it reads the right slice.
                // Use empty filtered_* arrays to signal the export to use
                // the originals with stride/offset.
                time_steps = requested_out_ts;

                // Export directly from original arrays with stride
                {
                    using std::cout;
                    const size_t F = face_count;
                    const int out_steps = requested_out_ts;
                    const int stride = total_ts;
                    const int offset = skip_eq;

                    // Always write mesh + time metadata (downstream stats consumers need them).
                    // Skip mesh.ply when output.skip_mesh is set (chunked-run chunks 1+).
                    namespace fs = std::filesystem;
                    fs::create_directories(output_dir);
                    if (!sim_config.output.skip_mesh &&
                        !exportMeshToPLY(output_dir + "/mesh.ply", mesh)) {
                        std::cerr << "ERROR exporting mesh\n";
                    }
                    writeTimeStepsFile(output_dir + "/time_steps.txt",
                                       filtered_times, filtered_time_strings);

                    if (sim_config.output.skip_time_series) {
                        cout << "[output] skip_time_series=true (stride-based path) — "
                                "per-step binary files NOT written; only mesh + statistics.\n";
                        // Fall through to the metadata + skip-the-loop block below.
                    } else {
                    // Create per-step output directories
                    fs::create_directories(output_dir + "/illumination");
                    fs::create_directories(output_dir + "/direct_solar");
                    fs::create_directories(output_dir + "/primary_reflected");
                    fs::create_directories(output_dir + "/primary_ir");
                    fs::create_directories(output_dir + "/temperature");

                    std::cout << "\n[exportDataAsBinary] Stride-based export\n"
                              << "  Faces=" << F << " time_steps=" << out_steps
                              << " stride=" << stride << " offset=" << offset << "\n";

                    std::vector<float> buf(F);
                    int report = std::max(1, out_steps / 20);

                    for (int t = 0; t < out_steps; ++t) {
                        if (t % report == 0 || t == out_steps - 1) {
                            cout << "\r  Progress " << (t+1) << "/" << out_steps << std::flush;
                        }

                        auto writeComponent = [&](const std::vector<double>& arr,
                                                   const std::string& dir,
                                                   const std::string& prefix) {
                            if (arr.empty()) return;
                            #pragma omp parallel for schedule(static)
                            for (size_t f = 0; f < F; ++f) {
                                size_t idx = f * static_cast<size_t>(stride) + offset + t;
                                buf[f] = (idx < arr.size())
                                    ? static_cast<float>(arr[idx]) : 0.0f;
                            }
                            std::ostringstream fn;
                            fn << output_dir << "/" << dir << "/" << prefix
                               << "_" << std::setw(4) << std::setfill('0') << t << ".bin";

                            std::ofstream out(fn.str(), std::ios::binary);
                            uint32_t count = static_cast<uint32_t>(F);
                            out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
                            out.write(reinterpret_cast<const char*>(buf.data()), F * sizeof(float));
                        };

                        // Illumination = direct + reflected (reconstruct on the fly)
                        {
                            #pragma omp parallel for schedule(static)
                            for (size_t f = 0; f < F; ++f) {
                                size_t idx = f * static_cast<size_t>(stride) + offset + t;
                                float ds = (idx < direct_solar_values.size())
                                    ? static_cast<float>(direct_solar_values[idx]) : 0.0f;
                                float rf = (idx < primary_reflected_values.size())
                                    ? static_cast<float>(primary_reflected_values[idx]) : 0.0f;
                                buf[f] = ds + rf;
                            }
                            std::ostringstream fn;
                            fn << output_dir << "/illumination/illumination_"
                               << std::setw(4) << std::setfill('0') << t << ".bin";
                            std::ofstream out(fn.str(), std::ios::binary);
                            uint32_t count = static_cast<uint32_t>(F);
                            out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
                            out.write(reinterpret_cast<const char*>(buf.data()), F * sizeof(float));
                        }

                        writeComponent(direct_solar_values, "direct_solar", "direct_solar");
                        writeComponent(primary_reflected_values, "primary_reflected", "primary_reflected");
                        writeComponent(primary_ir_values, "primary_ir", "primary_ir");

                        // Temperature (face-major vector<vector<double>>)
                        if (!surface_temps.empty()) {
                            #pragma omp parallel for schedule(static)
                            for (size_t f = 0; f < F; ++f) {
                                int src_t = offset + t;
                                float v = 50.0f;
                                if (f < surface_temps.size() &&
                                    src_t < static_cast<int>(surface_temps[f].size())) {
                                    double tv = surface_temps[f][src_t];
                                    if (std::isfinite(tv) && tv > 0.0 && tv < 10000.0)
                                        v = static_cast<float>(tv);
                                }
                                buf[f] = v;
                            }
                            std::ostringstream fn;
                            fn << output_dir << "/temperature/temperature_"
                               << std::setw(4) << std::setfill('0') << t << ".bin";
                            std::ofstream out(fn.str(), std::ios::binary);
                            uint32_t count = static_cast<uint32_t>(F);
                            out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
                            out.write(reinterpret_cast<const char*>(buf.data()), F * sizeof(float));
                        }
                    }
                    cout << "\n  Export complete: " << out_steps << "/" << out_steps << "\n";
                    cout << "  Binary export completed successfully!\n";
                    } // end else (per-step files written)
                }

                // Write metadata
                {
                    std::ofstream meta(output_dir + "/metadata.txt");
                    meta << "faces=" << face_count << "\n"
                         << "time_steps=" << requested_out_ts << "\n"
                         << "successful_timesteps=" << requested_out_ts << "\n"
                         << "failed_timesteps=0\n"
                         << "is_custom_mesh=false\n"
                         << "components=illumination,direct_solar,primary_reflected,primary_ir,temperature\n";
                }

                // Signal the caller to skip the normal binary export below.
                stride_export_already_done = true;

            } else {
                // Normal path (small arrays): allocate filtered copies
                FilteredData fd = filterFaceMajorStable(
                    illumination_values,
                    direct_solar_values,
                    primary_reflected_values,
                    primary_ir_values,
                    surface_temps,
                    times,
                    time_strings,
                    sun_positions,
                    primary_positions,
                    face_count,
                    total_ts,
                    skip_eq,
                    requested_out_ts,
                    verbose
                );

                const int filtered_steps = static_cast<int>(fd.times.size());
                diagnosticsIlluminationCoverage(
                    fd.illumination,
                    fd.direct,
                    face_count,
                    filtered_steps,
                    "Post-filter (output interval)",
                    verbose);

                filtered_illumination_values.swap(fd.illumination);
                filtered_direct_solar_values.swap(fd.direct);
                filtered_primary_reflected_values.swap(fd.reflected);
                filtered_primary_ir_values.swap(fd.ir);
                filtered_surface_temps.swap(fd.temps);
                filtered_times.swap(fd.times);
                filtered_time_strings.swap(fd.time_strings);
                filtered_sun_positions.swap(fd.sun_pos);
                filtered_primary_positions.swap(fd.primary_pos);

                time_steps = filtered_steps;
            }
        } else {
            // Chunked/streamed path: slice meta and temperatures, AND extract illumination for output window only
            const int start = skip_eq;
            const int end   = std::min(start + requested_out_ts, total_ts);
            const int filtered_steps = std::max(0, end - start);

            // Meta slices
            if (filtered_steps > 0) {
                filtered_times.assign(times.begin() + start, times.begin() + end);
                filtered_time_strings.assign(time_strings.begin() + start, time_strings.begin() + end);
                filtered_sun_positions.assign(sun_positions.begin() + start, sun_positions.begin() + end);
                filtered_primary_positions.assign(primary_positions.begin() + start, primary_positions.begin() + end);
            }

            // Temperatures slice for output interval (only if thermal was run)
            if (!surface_temps.empty() && surface_temps.size() >= face_count) {
                filtered_surface_temps.resize(face_count);
                #pragma omp parallel for schedule(static)
                for (size_t f = 0; f < face_count; ++f) {
                    const auto& src = surface_temps[f];
                    if (filtered_steps > 0 &&
                        src.size() >= static_cast<size_t>(total_ts) &&
                        end <= static_cast<int>(src.size())) {
                        filtered_surface_temps[f].assign(src.begin() + start, src.begin() + end);
                    }
                }
            }

            // Extract illumination arrays for just the output window to enable export
            if (filtered_steps > 0 && illumination_model) {
                const size_t out_size = face_count * static_cast<size_t>(filtered_steps);
                filtered_direct_solar_values.assign(out_size, 0.0);
                filtered_primary_reflected_values.assign(out_size, 0.0);
                filtered_primary_ir_values.assign(out_size, 0.0);
                filtered_illumination_values.assign(out_size, 0.0);

                // Fill via on-demand getters; write face-major contiguous blocks
                #pragma omp parallel for schedule(dynamic, 1024)
                for (size_t f = 0; f < face_count; ++f) {
                    const size_t dst_base = f * static_cast<size_t>(filtered_steps);
                    for (int ot = 0; ot < filtered_steps; ++ot) {
                        const int t = start + ot;

                        const double d  = illumination_model->getDirectSolarValue(f, t);
                        const double r  = illumination_model->getPrimaryReflectedValue(f, t);
                        const double ir = illumination_model->getPrimaryIRValue(f, t);

                        filtered_direct_solar_values[dst_base + ot]        = d;
                        filtered_primary_reflected_values[dst_base + ot]    = r;
                        filtered_primary_ir_values[dst_base + ot]           = ir;
                        filtered_illumination_values[dst_base + ot]         = d + r; // illumination = direct + reflected
                    }
                }

                diagnosticsIlluminationCoverage(
                    filtered_illumination_values,
                    filtered_direct_solar_values,
                    face_count,
                    filtered_steps,
                    "Post-filter (output interval, chunked)",
                    verbose);
            } else {
                // If no illumination_model or zero steps, leave arrays empty (nothing to export)
                filtered_direct_solar_values.clear();
                filtered_primary_reflected_values.clear();
                filtered_primary_ir_values.clear();
                filtered_illumination_values.clear();
            }

            time_steps = filtered_steps;
        }

        // Optional spot check (only if we actually have direct solar and verbose mode)
        if (verbose && !filtered_direct_solar_values.empty() && time_steps > 0) {
            std::cout << "[Filter] Spot check (face 0) direct solar:";
            for (int k : {0, time_steps/4, time_steps/2, (3*time_steps)/4, time_steps-1}) {
                const size_t idx = 0 * static_cast<size_t>(time_steps) + static_cast<size_t>(k);
                if (idx < filtered_direct_solar_values.size())
                    std::cout << " " << filtered_direct_solar_values[idx];
            }
            std::cout << "\n";
        }
    }
    // ========== END DATA FILTERING SECTION ==========

    // Now filter thermal data only if we ran the thermal model AND temps not already filtered
    if (run_thermal_mesh && !surface_temps.empty() && filtered_surface_temps.empty()) {
        std::cout << "Filtering thermal data..." << std::endl;

        const size_t face_count = surface_temps.size();
        const int full_steps = total_time_steps;          // full (incl. equilibration)
        const int skip_eq = equilibration_steps;
        const int out_steps = time_steps;                 // already updated to filtered illumination steps

        if (skip_eq + out_steps > full_steps) {
            std::cerr << "Thermal filter: adjusting output steps (skip+out exceeds full)." << std::endl;
        }

        filtered_surface_temps.resize(face_count);

        size_t report_interval = std::max<size_t>(size_t(1), face_count / 20);

        // Quick diagnostic on source temps
        std::cout << "Source temperature data check (few reference times):" << std::endl;
        for (int probe : {0, full_steps / 2, std::max(0, full_steps - 1)}) {
            if (probe < full_steps && !surface_temps[0].empty() && probe < (int)surface_temps[0].size()) {
                double mn = 1e9, mx = -1e9;
                for (size_t f = 0; f < std::min<size_t>(100, face_count); ++f) {
                    if (probe < (int)surface_temps[f].size()) {
                        mn = std::min(mn, surface_temps[f][probe]);
                        mx = std::max(mx, surface_temps[f][probe]);
                    }
                }
                std::cout << "  t=" << probe << ": min=" << mn << "K max=" << mx << "K" << std::endl;
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < face_count; ++f) {
            const auto& src = surface_temps[f];
            std::vector<double> dst(out_steps, 50.0);
            for (int ot = 0; ot < out_steps; ++ot) {
                int st = skip_eq + ot;
                if (st >= 0 && st < (int)src.size()) {
                    dst[ot] = src[st];
                }
            }
            filtered_surface_temps[f].swap(dst);

            if (f < 1) { // face 0 diagnostic once
                double mn = *std::min_element(filtered_surface_temps[f].begin(), filtered_surface_temps[f].end());
                double mx = *std::max_element(filtered_surface_temps[f].begin(), filtered_surface_temps[f].end());
                #pragma omp critical
                std::cout << "  Face 0 filtered temp range: [" << mn << "," << mx << "]K" << std::endl;
            }
            if (report_interval > 0 && f % report_interval == 0) {
                #pragma omp critical
                std::cout << "\rFiltering thermal faces: "
                        << std::fixed << std::setprecision(1)
                        << (100.0 * (f + 1) / face_count) << "%" << std::flush;
            }
        }
        std::cout << "\rFiltering thermal faces: 100.0%\n";
    } else if (run_thermal_mesh && !filtered_surface_temps.empty()) {
        if (verbose) {
            std::cout << "Thermal data already filtered (skipping duplicate filtering)." << std::endl;
        }
    }

    if (run_thermal_mesh) {
        if (!validateTemperatureMatrix(filtered_surface_temps, mesh->faces.size(), time_steps)) {
            std::cerr << "[FATAL] Filtered temperature matrix shape invalid (faces x output_steps). Aborting.\n";
            return 1;
        }
    }

    // Now write body positions file with filtered data
    std::string positions_output = output_dir + "/body_positions.csv";
    exportBodyPositions(
        positions_output,
        filtered_sun_positions,
        filtered_primary_positions,
        filtered_time_strings
    );

    // Update the VTK output section with this more efficient approach
    std::string illumination_vtk_file = output_dir + "/ptm_illumination.vtk";
    if (thermal_vtk_file.empty()) {
        thermal_vtk_file = output_dir + "/ptm_thermal.vtk";
    }
    if (complete_vtk_file.empty()) {
        complete_vtk_file = output_dir + "/ptm_complete.vtk";
    }

    bool data_write_success = false;

    // Handle output based on format selection (binary vs VTK)
    if (output_format == "vtk") {
        // VTK format output - use existing code paths
        if (!run_thermal_mesh) {
            // For illumination-only runs, write just the illumination VTK
            std::cout << "Writing illumination data to " << illumination_vtk_file << "..." << std::endl;
            MeshUtils mesh_utils;
            data_write_success = mesh_utils.exportIlluminationToVTK(
                illumination_vtk_file,
                mesh,
                filtered_illumination_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                filtered_times,
                filtered_time_strings,
                time_steps
            );
            
            if (data_write_success) {
                std::cout << "Illumination data written to " << illumination_vtk_file << std::endl;
            }
        } else {
            // For thermal runs, create a combined VTK file with all data
            std::cout << "Writing complete data to VTK..." << std::endl;
            MeshUtils mesh_utils;
            
            std::string vtk_file = (vtk_output_mode == "thermal") ? thermal_vtk_file : complete_vtk_file;
            
            data_write_success = mesh_utils.exportIlluminationToVTK(
                vtk_file,
                mesh,
                filtered_illumination_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                filtered_times,
                filtered_time_strings,
                time_steps
            );
            
            if (data_write_success && !filtered_surface_temps.empty()) {
                std::cout << "Adding thermal data to VTK file..." << std::endl;
                data_write_success = addThermalDataToVTK(
                    vtk_file,
                    mesh,
                    filtered_surface_temps,
                    filtered_times
                );
            }
            
            if (data_write_success) {
                std::cout << "Complete data written to " << vtk_file << std::endl;
            }
        }
    } else if (output_format == "binary" && !sim_config.output.skip_time_series && !stride_export_already_done) {
        std::cout << "Using binary output format for data files" << std::endl;

        // IMPORTANT: Disable internal stats here; we'll export stats for the integration window below.
        const bool compute_statistics_for_output = false;
        if (!run_thermal_mesh) {
            // For illumination-only runs, pass the filtered arrays directly
            data_write_success = exportDataAsBinary(
                output_dir,
                mesh,
                filtered_illumination_values,
                filtered_direct_solar_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                std::vector<std::vector<double>>(), // Empty temperature data
                filtered_times,
                filtered_time_strings,
                time_steps,
                !mesh_file.empty(), // is_custom_mesh
                compute_statistics_for_output,
                sim_config.output.skip_mesh
            );
        } else {
            // For thermal runs, pass all filtered data including temperatures
            data_write_success = exportDataAsBinary(
                output_dir,
                mesh,
                filtered_illumination_values,
                filtered_direct_solar_values,
                filtered_primary_reflected_values,
                filtered_primary_ir_values,
                filtered_surface_temps,
                filtered_times,
                filtered_time_strings,
                time_steps,
                !mesh_file.empty(), // is_custom_mesh
                compute_statistics_for_output,
                sim_config.output.skip_mesh
            );
        }
        
        if (data_write_success) {
            std::cout << "Data successfully written in binary format to: " << output_dir << std::endl;
        } else {
            std::cerr << "ERROR: Failed to export data in binary format!" << std::endl;
        }
    } else if (output_format == "binary" && sim_config.output.skip_time_series) {
        std::cout << "[output] skip_time_series=true — per-step binary files NOT written; "
                     "writing mesh + time metadata + statistics only." << std::endl;
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::exception& e) {
            std::cerr << "[output] WARNING: cannot create output dir: " << e.what() << "\n";
        }
        bool mesh_ok = true;
        if (!sim_config.output.skip_mesh) {
            mesh_ok = exportMeshToPLY(output_dir + "/mesh.ply", mesh);
            if (!mesh_ok) std::cerr << "[output] WARNING: mesh.ply write failed\n";
        }
        bool time_ok = writeTimeStepsFile(output_dir + "/time_steps.txt",
                                          filtered_times, filtered_time_strings);
        if (!time_ok) std::cerr << "[output] WARNING: time_steps.txt write failed\n";
        data_write_success = mesh_ok && time_ok;
    }

    // Export statistics over the INTEGRATION window only
    if (compute_statistics) {
        try {
            std::cout << "\n[Statistics] Exporting statistics over integration window..." << std::endl;

            const bool write_raw = sim_config.output.write_raw_stats;
            bool stats_ok = false;
            if (have_full_illum_arrays) {
                // Arrays are present: use existing path
                stats_ok = exportStatisticsBinary(
                    output_dir,
                    mesh,
                    stats_fd.illumination,
                    stats_fd.direct,
                    stats_fd.reflected,
                    stats_fd.ir,
                    stats_fd.temps,
                    stats_steps,
                    write_raw
                );
            } else {
                // Streamed mode: compute illumination stats on-the-fly via getters
                bool illum_ok = false;
                if (illumination_model) {
                    illum_ok = exportIlluminationStatisticsStreamed(
                        output_dir,
                        mesh,
                        *illumination_model,
                        stats_start_step,
                        stats_steps
                    );
                } else {
                    std::cerr << "[Statistics] No illumination_model available for streamed stats\n";
                }

                // Always export temperature stats (we already have stats_fd.temps)
                bool temp_ok = exportStatisticsBinary(
                    output_dir,
                    mesh,
                    /*illum*/ std::vector<double>(),
                    /*dir*/   std::vector<double>(),
                    /*refl*/  std::vector<double>(),
                    /*ir*/    std::vector<double>(),
                    stats_fd.temps,
                    stats_steps,
                    write_raw
                );

                stats_ok = illum_ok && temp_ok;
            }

            if (!stats_ok) {
                std::cerr << "WARNING: Statistics export failed." << std::endl;
            } else {
                std::cout << "[Statistics] Exported to " << (output_dir + "/statistics") << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "WARNING: Exception during statistics export: " << e.what() << std::endl;
        }

        // Schorghofer & Taylor (2007) ice-stability output. Computed only
        // when the user asked for it (zero overhead otherwise) and only
        // when the GPU/CPU thermal solver actually filled the accumulator.
        if (sim_config.thermal.compute_ice_stability && ice_stability_accum.ready()) {
            const auto& isp = sim_config.thermal.ice_stability;
            const double l_m = isp.grain_hop_length_um * 1e-6;
            std::cout << "[Ice stability] Solving J(z)=μℓ⟨E⟩/(2z)=J0 per face... "
                      << "(N_samples=" << ice_stability_accum.n_samples
                      << ", layers=" << ice_stability_accum.n_layers
                      << ", J0=" << std::scientific << std::setprecision(2)
                      << isp.flux_threshold_kgm2s
                      << " kg/m²/s, ℓ=" << std::fixed << std::setprecision(1)
                      << isp.grain_hop_length_um << " μm)" << std::endl;
            std::vector<float> ice_depth = ice::computeIceStabilityDepths(
                ice_stability_accum,
                isp.flux_threshold_kgm2s,
                l_m,
                /* z_max_m */ -1.0,
                /* use_openmp */ true);

            // If raw-stats output is enabled, dump the per-face × per-layer
            // ⟨E(z)⟩ accumulator alongside the solved depth so a chunked
            // run's merge step can sum-then-solve.
            if (sim_config.output.write_raw_stats) {
                if (writeIceStabilityRaw(output_dir, ice_stability_accum)) {
                    std::cout << "[Ice stability] Wrote raw accumulator to "
                              << output_dir << "/statistics/_raw/" << std::endl;
                }
            }

            // Write a single ice_depth.bin (replaces the legacy three-suffix
            // ice_depth_avg/max/min.bin). Format matches writeStatisticsBinary:
            // uint32 N, then N float32 values.
            try {
                std::filesystem::create_directories(output_dir + "/statistics");
            } catch (...) {}
            const std::string fn = output_dir + "/statistics/ice_depth.bin";
            std::ofstream of(fn, std::ios::binary);
            if (of) {
                uint32_t N = static_cast<uint32_t>(ice_depth.size());
                of.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));
                of.write(reinterpret_cast<const char*>(ice_depth.data()),
                         ice_depth.size() * sizeof(float));
                if (of.good()) {
                    // Quick summary
                    size_t n_at_zero = 0, n_capped = 0, n_finite = 0;
                    double sum = 0.0, lo = 1e30, hi = 0.0;
                    const double cap = ice_stability_accum.z.empty() ? 0.0 :
                                        ice_stability_accum.z.back();
                    for (float v : ice_depth) {
                        if (std::isfinite(v)) {
                            ++n_finite;
                            sum += v;
                            if (v < lo) lo = v;
                            if (v > hi) hi = v;
                            if (v == 0.0f) ++n_at_zero;
                            if (cap > 0.0 && v >= static_cast<float>(cap) - 1e-6f) ++n_capped;
                        }
                    }
                    const double mean = (n_finite > 0) ? sum / n_finite : 0.0;
                    std::cout << "[Ice stability] Wrote " << fn << "  ("
                              << n_finite << " finite, range " << lo << ".." << hi
                              << " m, mean " << mean << " m, "
                              << n_at_zero << " at surface, "
                              << n_capped << " at z_max=" << cap << " m)" << std::endl;
                } else {
                    std::cerr << "[Ice stability] WARNING: write failed for " << fn << "\n";
                }
            } else {
                std::cerr << "[Ice stability] WARNING: cannot open " << fn << " for write\n";
            }
        } else if (sim_config.thermal.compute_ice_stability) {
            std::cerr << "[Ice stability] WARNING: requested but accumulator not ready "
                         "(thermal solver path may not yet implement the hook)." << std::endl;
        }
    }

    // Common error message for both formats
    if (!data_write_success) {
        std::cerr << "Warning: Failed to write output data files" << std::endl;
    }

    // ========== CALCULATE AND PRINT STATISTICS (ONCE) ==========
    // Calculate elapsed time
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - start_time_point);

    double total_elapsed_seconds = total_duration.count() / 1000.0;
    long long total_elapsed_ms = total_duration.count();

    std::cout << "Simulation completed in " << std::fixed << std::setprecision(3) 
            << total_elapsed_seconds << " seconds (" << total_elapsed_ms << " ms)" << std::endl;

    // Calculate statistics from filtered arrays (output interval)
    // Note: Temperature statistics exported above use the integration window.
    double avg_flux = 0.0;
    double avg_direct = 0.0;
    double avg_reflected = 0.0;
    double avg_ir = 0.0;
    double max_flux = 0.0;
    double max_direct = 0.0;
    double max_reflected = 0.0;
    double max_ir = 0.0;
    double min_flux = std::numeric_limits<double>::max();
    size_t non_zero_count = 0;

    // Use filtered arrays for statistics (output interval)
    for (size_t i = 0; i < filtered_illumination_values.size(); i++) {
        double direct = filtered_direct_solar_values[i];
        double reflected = filtered_primary_reflected_values[i];
        double ir = filtered_primary_ir_values[i];
        double total = filtered_illumination_values[i] + reflected + ir;
        
        if (total > 0.0) {
            avg_flux += total;
            avg_direct += direct;
            avg_reflected += reflected;
            avg_ir += ir;
            non_zero_count++;
            
            max_flux = std::max(max_flux, total);
            max_direct = std::max(max_direct, direct);
            max_reflected = std::max(max_reflected, reflected);
            max_ir = std::max(max_ir, ir);
            
            if (total < min_flux && total > 0.0) {
                min_flux = total;
            }
        }
    }

    if (non_zero_count > 0) {
        avg_flux /= non_zero_count;
        avg_direct /= non_zero_count;
        avg_reflected /= non_zero_count;
        avg_ir /= non_zero_count;
    } else {
        min_flux = 0.0;  // Set to 0 if no non-zero values found
    }

    std::cout << "Results:" << std::endl;
    std::cout << "  Average total flux: " << std::fixed << std::setprecision(9) 
            << avg_flux << " W/mÂ²" << std::endl;
    std::cout << "    - Direct solar: " << avg_direct << " W/mÂ²" << std::endl;
    std::cout << "    - Primary reflected: " << avg_reflected << " W/mÂ²" << std::endl;
    std::cout << "    - Primary IR: " << avg_ir << " W/mÂ²" << std::endl;
    std::cout << "  Maximum total flux: " << max_flux << " W/mÂ²" << std::endl;
    std::cout << "    - Direct solar: " << max_direct << " W/mÂ²" << std::endl;
    std::cout << "    - Primary reflected: " << max_reflected << " W/mÂ²" << std::endl;
    std::cout << "    - Primary IR: " << max_ir << " W/mÂ²" << std::endl;
    std::cout << "  Minimum flux (non-zero): " << min_flux << " W/mÂ²" << std::endl;
    std::cout << "  Output written to " << output_dir << std::endl;


    // ========== LOCATION EXPORT (if requested) ==========
    if (export_locations && !locations_file.empty()) {
        try {
            const std::string series_out_path =
                (output_dir.empty() ? series_output
                                    : (output_dir.back() == '/' ? output_dir + series_output
                                                                : output_dir + "/" + series_output));
            if (!exportLocationIlluminationAndTemperature(
                    series_out_path,
                    locations_file,
                    mesh,
                    illumination_values,
                    direct_solar_values,
                    primary_reflected_values,
                    primary_ir_values,
                    surface_temps,
                    time_strings,
                    time_steps,
                    indirect_flux_series)) {
                std::cerr << "Warning: Failed to export location data to "
                          << series_out_path << std::endl;
            } else {
                std::cout << "Location illumination/temperature/indirect data written: "
                          << series_out_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Location illumination export failed: "
                      << e.what() << std::endl;
        }
    }

    // Optional per-location depth profiles
    if (export_locations && !locations_file.empty() && !depths_file.empty()) {
        try {
            auto locs = readLocationsCSV(locations_file);
            auto depths_m = readDepthsFile(depths_file);
            if (depths_m.empty()) {
                std::cerr << "Warning: Depths file parsed empty; skipping depth profiles." << std::endl;
            } else {
                const std::string loc_out_dir =
                    output_dir + (output_dir.back() == '/' ? "locations" : "/locations");
                std::filesystem::create_directories(loc_out_dir);
                std::cout << "Depth profile export: " << locs.size()
                          << " locations, " << depths_m.size()
                          << " depths -> " << loc_out_dir << std::endl;

                // Load per-face albedo from binary map (same as thermal solver)
                std::vector<double> loc_face_albedos(mesh->faces.size(), default_visible_albedo);
                {
                    std::string body_name = planet.name;
                    std::transform(body_name.begin(), body_name.end(), body_name.begin(), ::tolower);
                    if (!body_name.empty()) body_name[0] = std::toupper(body_name[0]);
                    std::string albedo_filename = body_name + "_Albedo_res"
                        + std::to_string(mesh_resolution) + ".bin";
                    std::string adir = albedo_maps_directory;
                    if (!adir.empty() && adir.back() != '/' && adir.back() != '\\') adir += '/';
                    std::vector<std::string> search_paths = {
                        adir + albedo_filename,
                        "../data/albedo_maps/" + albedo_filename,
                        "data/albedo_maps/" + albedo_filename
                    };
                    for (const auto& path : search_paths) {
                        auto loaded = importAlbedoMapFromFile(path);
                        if (!loaded.empty() && loaded.size() == mesh->faces.size()) {
                            std::copy(loaded.begin(), loaded.end(), loc_face_albedos.begin());
                            std::cout << "Location export: using per-face albedo from " << path << std::endl;
                            break;
                        }
                    }
                }

                auto sanitize = [](std::string s) {
                    for (char& c : s) {
                        if (!(std::isalnum(static_cast<unsigned char>(c)) || c=='-' || c=='_')) c = '_';
                    }
                    return s;
                };

                for (size_t i = 0; i < locs.size(); ++i) {
                    const auto& tup = locs[i];
                    const std::string name = std::get<0>(tup);
                    const double lat_deg = std::get<1>(tup);
                    const double lon_deg = std::get<2>(tup);

                    // Location CSV uses East longitude (IAU convention), same as mesh (atan2)
                    int face_idx = findNearestFaceIndexByLatLon(mesh, lat_deg, lon_deg);
                    if (face_idx < 0) {
                        std::cerr << "Warning: Could not map location '"
                                  << (name.empty() ? ("loc_"+std::to_string(i)) : name)
                                  << "' to a mesh face; skipping." << std::endl;
                        continue;
                    }

                    // Absorbed flux: (1 - A_vis)*(direct+reflected) + primary_ir
                    // Use total_time_steps (including equilibration) so the location
                    // thermal model receives the full cycled illumination, matching
                    // how mesh faces are handled.
                    std::vector<double> flux_sum(static_cast<size_t>(total_time_steps), 0.0);
                    double A_vis = loc_face_albedos[static_cast<size_t>(face_idx)];
                    A_vis = std::clamp(A_vis, 0.0, 1.0);

                    for (int t = 0; t < total_time_steps; ++t) {
                        const size_t flat = static_cast<size_t>(face_idx) * static_cast<size_t>(total_time_steps)
                                          + static_cast<size_t>(t);
                        const double direct = (flat < direct_solar_values.size()) ? direct_solar_values[flat] : 0.0;
                        const double refl   = (flat < primary_reflected_values.size()) ? primary_reflected_values[flat] : 0.0;
                        const double pir    = (flat < primary_ir_values.size()) ? primary_ir_values[flat] : 0.0;
                        flux_sum[static_cast<size_t>(t)] = (1.0 - A_vis) * (direct + refl) + pir;
                    }

                    auto T_depths = runThermalModelDepthSeries(
                        flux_sum, times, lat_deg, config, depths_m, equilibration_steps, planet);

                    std::string base = sanitize(name.empty() ? ("loc_" + std::to_string(i)) : name);
                    const std::string out_path = loc_out_dir + "/" + base + "_depths.csv";
                    if (!writeLocationDepthCSV(out_path, name, time_strings, depths_m, T_depths)) {
                        std::cerr << "Warning: Failed to write depth series for location '" << name << "'" << std::endl;
                    } else {
                        std::cout << "Wrote depth temperature series: " << out_path << std::endl;
                    }
                }
                // Generate plots if enabled
                if (generate_depth_plots) {
                    generateDepthProfilePlots(loc_out_dir, executable_path);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Depth profile export failed: " << e.what() << std::endl;
        }
    }

    // ========== IMMEDIATE MEMORY CLEANUP (ALWAYS) ==========
    // Clean up unfiltered arrays immediately to free memory
    std::cout << "Performing memory cleanup..." << std::endl;

    // Clear the UNFILTERED arrays (no longer needed)
    illumination_values.clear();
    illumination_values.shrink_to_fit();
    direct_solar_values.clear(); 
    direct_solar_values.shrink_to_fit();
    primary_reflected_values.clear();
    primary_reflected_values.shrink_to_fit();
    primary_ir_values.clear();
    primary_ir_values.shrink_to_fit();

    // Clear unfiltered temperature data
    if (!surface_temps.empty()) {
        for (auto& temps : surface_temps) {
            temps.clear();
            temps.shrink_to_fit();
        }
        surface_temps.clear();
        surface_temps.shrink_to_fit();
    }

    // Clear illumination model (has large internal arrays)
    if (illumination_model) {
        illumination_model.reset();
    }

    // Force garbage collection
    //#pragma omp barrier
    barrier_if_parallel();

    std::cout << "Memory cleanup completed." << std::endl;

    // ========== FINAL CLEANUP ==========
    // Clean up filtered arrays and mesh before program exit
    std::cout << "\nCleaning up remaining memory..." << std::endl;

    // Clear filtered data arrays
    filtered_illumination_values.clear();
    filtered_illumination_values.shrink_to_fit();

    filtered_direct_solar_values.clear();
    filtered_direct_solar_values.shrink_to_fit();

    filtered_primary_reflected_values.clear();
    filtered_primary_reflected_values.shrink_to_fit();

    filtered_primary_ir_values.clear();
    filtered_primary_ir_values.shrink_to_fit();

    // Clear filtered temperature data
    if (!filtered_surface_temps.empty()) {
        for (auto& temps : filtered_surface_temps) {
            temps.clear();
            temps.shrink_to_fit();
        }
        filtered_surface_temps.clear();
        filtered_surface_temps.shrink_to_fit();
    }

    // Clear the mesh
    if (mesh) {
        mesh->vertices.clear();
        mesh->vertices.shrink_to_fit();
        mesh->faces.clear();
        mesh->faces.shrink_to_fit();
        mesh->face_normals.clear();
        mesh->face_normals.shrink_to_fit();
        mesh->face_centers.clear();
        mesh->face_centers.shrink_to_fit();
        mesh->latitudes.clear();
        mesh->latitudes.shrink_to_fit();
        mesh->longitudes.clear();
        mesh->longitudes.shrink_to_fit();
        mesh.reset();
    }

    // Ensure all OpenMP threads are finished
    //#pragma omp barrier
    barrier_if_parallel();

    std::cout << "Cleanup completed successfully" << std::endl;

    return 0;
    }  // end of main
