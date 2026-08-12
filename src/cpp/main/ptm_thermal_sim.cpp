#include "../mesh/mesh_utils.h"
#include "../illumination/illumination_model.h"
#include "../constants/constants.h"
// Header-only, and deliberately NOT behind the GPU guard: the eq_extrapolation
// knob has to be parsed and refused in CPU-only builds too.
#include "../thermal/eq_extrapolation.h"
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
#include <unistd.h>   // for STDERR_FILENO, fsync
#include <cstdio>     // for std::fopen / fprintf / fflush / fclose (heartbeat)
#include <ctime>      // for gmtime_r / strftime (heartbeat)

#include "../tools/config_parser.h"
#include "../tools/run_provenance.h"
#include "../thermal/thermal_model.h"
#include "../thermal/eq_certificate.h"
#include <memory>
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

// Per-host GPU dispatch profile (Phase R, R1). Dependency-free; safe to
// include even in CPU-only builds (the calibration entry point is guarded by
// PTM_GPU_OPENCL, but the profile record + loader are pure I/O).
#include "../gpu/machine_profile.h"
#include <unistd.h>   // gethostname
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann
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
    
// Apply main()-forwarded thermophysical properties to a solver's local Planet
// (defined below near the solver bodies; forward-declared for the per-face path
// above the definition). See the definition for why this is needed.
static void applyConfiguredThermo(thermal::Planet& planet, const thermal::Config& config);

// PTM-53 rule-7 tripwire. applyConfiguredThermo() bumps this every time it
// applies a user-configured scalar override to a solver-local Planet; main()
// FAILS the run if an override was configured and this stayed at zero, so an
// override can never again be announced in the log and then silently ignored.
static std::atomic<int> g_thermo_override_applied{0};

// PTM-53: the SINGLE definition of "the user asked for a thermophysical
// override". The forwarding block and the tripwire both call this, so a knob
// added to one can never be missing from the other -- which is the failure mode
// that would quietly re-open PTM-53 for the next property someone adds.
// `geothermal_heat_flow` lives on SimulationConfig::thermal, not on
// ThermophysicalParams, but it reaches the solvers through the SAME ovr_*
// channel and must therefore arm the SAME tripwire (PTM-161). It is passed
// explicitly rather than read from a global so the one definition still owns
// the whole predicate.
static bool thermoOverrideRequested(const ptm::SimulationConfig::ThermophysicalParams& tp,
                                    double geothermal_heat_flow) {
    return (tp.ks_override              > 0.0)
        || (tp.kd_override              > 0.0)
        || (tp.emissivity_override      > 0.0)
        || (tp.albedo_override         >= 0.0)
        || (tp.constant_thermal_inertia > 0.0)
        || (geothermal_heat_flow        > 0.0);
}

// Parse a numeric CLI argument, or refuse the run naming the flag.
//
// The rest of this parser calls std::stod(argv[++i]) bare, so `--duration abc`
// leaves main() through an uncaught std::invalid_argument and the operator sees
// only `libc++abi: terminating`. New flags do not inherit that: a value that is
// not a finite number, or that has trailing junk ("0.12kg"), stops the run with
// a line that says which flag and what it received.
static bool parseCliDouble(const std::string& flag, const char* text, double& out) {
    try {
        std::size_t consumed = 0;
        out = std::stod(text, &consumed);
        if (text[consumed] != '\0') throw std::invalid_argument("trailing characters");
    } catch (const std::exception&) {
        std::cerr << "FATAL: " << flag << " expects a number, got '" << text << "'."
                  << std::endl;
        return false;
    }
    if (!std::isfinite(out)) {
        std::cerr << "FATAL: " << flag << " expects a finite number, got '" << text
                  << "'." << std::endl;
        return false;
    }
    return true;
}

// Heat capacity at temperature T for a body whose cp(T) is the
// planetary_properties polynomial cp = cp0 + sum_j cpCoeff[j] * T^(j+1).
//
// This is the same evaluation, at the same 273 K reference, that the engine's
// only other body-level thermal-inertia definition already uses
// (thermal_property_map.cpp: cpAtT / depthWeightedTI, i.e. the TI(273 K) maps
// of Hayne et al. 2017). It is duplicated rather than exported because that
// one is file-local in an anonymous namespace; if either moves, they must move
// together, and this comment is the contract.
//
// For every body with a constant heat capacity -- which is every generic body,
// cpCoeff empty -- this returns cp0 exactly.
static double heatCapacityAtT(const thermal::Planet& planet, double T) {
    double cp = planet.cp0;
    double T_pow = T;
    for (double c : planet.cpCoeff) { cp += c * T_pow; T_pow *= T; }
    return cp;
}

// Parse a comma-separated list of doubles ("" => empty). Shared by the
// generic-body cpCoeff forwarding and the override cpCoeff forwarding.
static std::vector<double> parseCsvDoubles(const std::string& csv) {
    std::vector<double> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stod(tok));
    return out;
}

// Format a double for a Config property WITHOUT the 6-decimal truncation that
// Config::setProperty(key, double) inflicts via std::to_string -- harmless for
// 7.4e-4, silently zero for 1e-8 (BUGS.md PTM-60).
static std::string fullPrecision(double v) {
    std::ostringstream os;
    os.precision(17);
    os << v;
    return os.str();
}

bool runThermalModelsForAllFaces(const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, int, const thermal::Config&, double, double,
    std::vector<std::vector<double>>*, int, int, int,
    const thermal::Planet* = nullptr,
    const std::vector<double>* face_Qb = nullptr,   // Per-face heat flow override
    // Per-GLOBAL-face time-mean indirect flux added to the equilibration-phase
    // absorbed flux only (absolute steps [0, equilibration_steps)). Empty =
    // legacy. Same contract as the GPU and SIMD-batch solvers.
    const std::vector<double>& eq_indirect_flux = {});

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
    const std::vector<double>* face_H_overrides = nullptr,
    // Per-GLOBAL-face time-mean indirect flux <q_ind>[f] (W/m^2) ADDED to the
    // EQUILIBRATION-phase absorbed flux only, i.e. to absolute steps
    // [0, equilibration_steps). Empty (default) = legacy: equilibration sees
    // no indirect term. Same contract as the GPU persistent solver's
    // eq_indirect_flux argument.
    const std::vector<double>& eq_indirect_flux = {});

// Apply the Hayne et al. (2017) incidence-angle correction to a base
// normal-incidence Bond albedo A0. Composes:
//
//     A(θ) = A0 + a·(θ/(π/4))^3 + b·(θ/(π/2))^8
//
// where θ is the solar incidence angle. cos_theta is recovered from the
// direct flux via cos(θ) = direct / S_solar (clamped to [0,1]). When the
// body has no albedoCoef (e.g., Europa), the base albedo is returned
// unchanged. Used by both CPU and GPU thermal flux pipelines so per-face
// map values from a spatial albedo map still receive the angular
// correction (otherwise high-θ faces are biased ~kelvin-warm in
// nighttime cooling because too much daytime flux is absorbed).
//
// Directional-hemispheric (Hapke) branch: opt-in via use_hapke (default
// false → byte-identical to the pre-Hapke Hayne-only path). When on, the
// analytic closed form A_h(mu0) = 1 - gamma*H(mu0,w) supersedes the Hayne
// polynomial entirely — this is the SAME closed form as
// Planet::getAlbedoWithIncidence() (thermal_model.cpp:579-589), replicated
// here because the GPU flux build runs host-side per-timestep and cannot
// call a Planet method inside the hot OpenMP loop cheaply. Guaranteed in
// [0,1] for w in [0,1], mu0 in [0,1].
//
// PTM-29 fix (2026-07-30): the Hapke branch is now CONTINUOUS at cos_theta>=1,
// matching the CPU reference Planet::getAlbedoWithIncidence (thermal_model.cpp:
// 579-589), which clamps mu0 to [0,1] and ALWAYS evaluates A_h in the Hapke
// branch. The old `cos_theta>=1 -> return A0` short-circuit returned the
// PER-FACE A0, but A_h(mu0->1) is anchored to the BODY-average A0 by the
// config-gate w — so for any face whose local albedo != the body A0 it was a
// step discontinuity. PTM-28 Fix A pushed cos_theta to 1.00002 at the diurnal
// insolation peak, firing that branch for exactly one output step per cycle and
// spiking the absorbed flux (measured +70 W/m^2 on a dark face) into the
// multi-K noon notch the GPU flux path showed but the CPU never did. Evaluating
// A_h at mu0=1 removes the discontinuity and makes the GPU flux albedo match the
// CPU reference. Hayne branch is already continuous there (theta=0 -> A0), so it
// is unchanged.
//
// Per-face albedo scaling (2026-07-30): the caller now passes a PER-FACE w
// (hapkeWFromA0 on that face's map albedo), so A_h(mu0=1;w_face) == that face's
// A0 by construction. This makes the PTM_LEGACY_ALBEDO_NORMAL_A0 hatch VESTIGIAL
// — the continuous branch already returns the per-face A0 at normal incidence,
// so the hatch changes nothing (kept only as a labeled no-op / documentation).
static inline bool legacyAlbedoNormalA0() {
    static bool legacy = []() {
        const char* e = std::getenv("PTM_LEGACY_ALBEDO_NORMAL_A0");
        bool on = (e && std::string(e) != "0" && e[0] != '\0');
        if (on) {
            std::cerr << "[Thermal] PTM_LEGACY_ALBEDO_NORMAL_A0=1: Hapke normal-incidence"
                         " short-circuit (VESTIGIAL under per-face w — A_h(1;w_face)==A0_face,"
                         " so this is a no-op)\n";
        }
        return on;
    }();
    return legacy;
}

// Per-face Hapke single-scattering albedo w from a normal-incidence Bond albedo
// A0, in EXACT CLOSED FORM (the inverse of A_h(mu0=1;w) = (1-gamma)/(1+2*gamma)
// = A0, with gamma = sqrt(1-w)): gamma = (1-A0)/(1+2*A0), w = 1 - gamma^2. This
// replaces the body-level bisection and lets every face carry its OWN w so the
// Hapke directional curve is anchored to that face's map albedo (not the body
// average). Well-posed for A0 in [0.05,0.5] (w in ~[0.25,0.94]); A0 clamped to
// (0,1) so gamma stays real and w in [0,1] even for a degenerate map value.
static inline double hapkeWFromA0(double A0) {
    if (A0 <= 0.0) A0 = 1e-6;
    if (A0 >= 1.0) A0 = 1.0 - 1e-6;   // A0 -> 1 => gamma -> 0 => w -> 1
    const double gamma = (1.0 - A0) / (1.0 + 2.0 * A0);
    double w = 1.0 - gamma * gamma;
    if (w < 0.0) w = 0.0;
    if (w > 1.0) w = 1.0;
    return w;
}

static inline double applyIncidenceAlbedo(double A0,
                                          double direct_flux,
                                          double solar_constant,
                                          double a_coef,
                                          double b_coef,
                                          bool use_hapke = false,
                                          double hapke_w = -1.0) {
    if (solar_constant <= 0.0 || direct_flux <= 0.0) return A0;
    double cos_theta = direct_flux / solar_constant;
    if (use_hapke && hapke_w >= 0.0) {
        // Analytic directional-hemispheric albedo A_h(mu0) = 1 - gamma*H(mu0,w).
        // PTM-29: clamp mu0 to [0,1] and evaluate A_h here for ALL cos_theta
        // (continuous with the CPU reference). The legacy escape hatch restores
        // the old per-face-A0 short-circuit at cos_theta>=1.
        if (legacyAlbedoNormalA0() && cos_theta >= 1.0) return A0;
        double mu0 = cos_theta;
        if (mu0 > 1.0) mu0 = 1.0;
        if (mu0 < 0.0) mu0 = 0.0;
        const double w     = std::max(0.0, std::min(1.0, hapke_w));
        const double gamma = std::sqrt(std::max(0.0, 1.0 - w));
        const double H     = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
        double A_h = 1.0 - gamma * H;
        if (A_h < 0.0) A_h = 0.0;
        if (A_h > 1.0) A_h = 1.0;
        return A_h;
    }
    if (cos_theta >= 1.0) return A0;
    if (cos_theta <  0.0) cos_theta = 0.0;
    const double theta = std::acos(cos_theta);
    const double t45 = theta / (M_PI / 4.0);
    const double t90 = theta / (M_PI / 2.0);
    double A = A0 + a_coef * t45 * t45 * t45 + b_coef * std::pow(t90, 8);
    if (A < 0.0) A = 0.0;
    if (A > 1.0) A = 1.0;
    return A;
}

// ---- Fix A (BUGS.md PTM-28): instantaneous heliocentric solar constant --------
// applyIncidenceAlbedo() and the CPU incidence block recover the solar incidence
// as cos(theta) = direct_flux / S. direct_solar was formed by the illumination
// model as S_inst * <cos>, with S_inst = SOLAR_CONSTANT / dist_au^2
// (illumination_model.cpp:3812). Recovering with a FIXED planet.S (the 1-AU
// value) makes cos_theta exceed 1 whenever the instantaneous insolation exceeds
// planet.S (perihelion + near-normal incidence), pinning the albedo to A0 across
// a super-solar-constant plateau; under the Hapke branch that plateau edge is a
// C1 (derivative) kink in absorbed flux which the GPU coarse time-integrator
// amplifies into a non-physical diurnal sawtooth (PTM-28). Feeding the per-
// timestep S_inst makes cos_theta <= 1 by construction so the plateau/kink never
// forms. g_solar_irr_per_step is filled ONCE in main() from the sun positions;
// left EMPTY (or env PTM_LEGACY_FIXED_SOLARCONST=1) every site falls back to the
// fixed planet.S == the legacy path (byte-identical). Read-only during the
// thermal dispatch (safe under the OpenMP flux-prep loops).
static std::vector<double> g_solar_irr_per_step;   // per-timestep S_inst (W/m^2)

static inline bool legacyFixedSolarConst() {
    static bool legacy = []() {
        const char* e = std::getenv("PTM_LEGACY_FIXED_SOLARCONST");
        return (e && std::string(e) != "0" && e[0] != '\0');
    }();
    return legacy;
}

// Instantaneous solar constant for ABSOLUTE timestep t; falls back to `fixed`
// (planet.S, the legacy 1-AU value) when the per-step table is unset/short.
static inline double solarConstForStep(int t, double fixed) {
    if (t >= 0 && static_cast<size_t>(t) < g_solar_irr_per_step.size())
        return g_solar_irr_per_step[t];
    return fixed;
}

// ---- BUGS.md PTM-107(a): observable effective surface albedo -----------------
// The BC's operative visible albedo A(theta) was computed per face per timestep and
// then thrown away, so no external energy-closure audit could reconstruct it -- an
// auditor had to treat it as a free parameter (that is what made the PTM-105
// investigation and Fermi's 11.4% closure residual un-decidable). Accumulate the
// flux-weighted mean per face here, at the ONE place the BC actually decides it,
// and write it beside the other statistics.
//   A_eff,f = sum_t A_f(t) Q_f(t) / sum_t Q_f(t)     (Q = direct solar, W/m^2)
// Faces that never see the Sun have sum_t Q = 0 and are written as NaN (they have
// no operative direct-beam albedo -- writing A0 there would invent a number).
static std::vector<double> g_eff_albedo_num;   // sum_t A*Q per face
static std::vector<double> g_eff_albedo_den;   // sum_t Q   per face
static bool g_eff_albedo_valid = false;

// ---- Accumulation WINDOW (Fermi condition C2) --------------------------------
// Every other statistics/*.bin is reduced over the INTEGRATION window
// [stats_start_step, stats_start_step + stats_steps). The albedo accumulators
// originally swept the whole flux table -- equilibration steps included -- so
// effective_albedo.bin described a different window from every file next to it.
// Measured on lunar_crater_80N (coupled, Hayne): global flux-weighted A_eff over
// eq+output was 0.5433 at 4 equilibration orbits and 0.5297 at 1, against 0.5269
// for the integration window in both -- so the file's value drifted with a knob
// that cannot change it, and injected -3.09 pp (4 orbits) / -0.53 pp (1 orbit) of
// spurious residual into an area-weighted closure audit that used it. With the
// window aligned, A_eff is identical at 1 and 4 orbits (max per-face difference
// 1.1e-6, below the run's own ET-round-trip floor of 5e-5 in the direct beam).
// Both the BC-side accumulator (below)
// and the radiosity source-side accumulator (the PTM-106 self-check, in the
// coupling loop) now reduce over exactly this window, so the file is window-
// consistent AND the self-check still compares like with like.
// Set once in main() from the SAME helper the statistics export uses; never a
// second copy of the recipe.
static int  g_aeff_win_start = 0;
static int  g_aeff_win_steps = 0;
static bool g_aeff_win_set   = false;

// The integration (statistics) window, in ABSOLUTE step indices. Extracted from
// the statistics-export site so the albedo accumulators can be told the window
// BEFORE the thermal solve without a second copy of the recipe (the PTM-106c
// class: two spellings of one formula, agreeing today, drifting later). The
// export site now calls this same function; its printed banner is unchanged.
struct IntegrationWindow { int start = 0; int steps = 0; };

static IntegrationWindow computeIntegrationWindow(int equilibration_steps,
                                                  int total_time_steps,
                                                  double output_steps_per_day,
                                                  double total_simulation_days,
                                                  double integration_days,
                                                  double output_days,
                                                  int integration_steps_cli,
                                                  int integration_offset_steps_cli,
                                                  double integration_offset_days) {
    // PTM-15 fix: the integration (statistics) window lives entirely in the
    // post-equilibration OUTPUT segment, which is sampled at the OUTPUT cadence.
    // The blended full-run cadence total_time_steps/total_simulation_days mixes
    // the (possibly coarse) equilibration cadence with the fine output cadence,
    // so converting integration/offset DAYS -> steps with it under-counts the
    // window for any decoupled-cadence run (equilibration_steps_per_day or
    // eq_samples_per_diurnal_cycle set). That silently reduced the statistics
    // _max/_min/_avg window below the written output frames: e.g. Moon3D 240 eq
    // (4.06/day) + 360 out (12.19/day) gave 29.53*(600/88.59)=200 stats steps
    // instead of 29.53*12.19=360, so direct_solar_max.bin was a max over only
    // output frames [0,200) and under-reported the true per-frame max (which
    // peaks near frame 215) for ~23% of faces. Use the output cadence; falls
    // back to the blended cadence only if the output cadence is unavailable.
    const double integ_steps_per_day = (output_steps_per_day > 0.0)
        ? output_steps_per_day
        : (total_simulation_days > 0.0
            ? static_cast<double>(total_time_steps) / total_simulation_days
            : 0.0);

    double integ_days_used = (integration_days > 0.0 ? integration_days : output_days);
    int integration_steps = (integration_steps_cli > 0)
        ? integration_steps_cli
        : static_cast<int>(std::llround(integ_days_used * integ_steps_per_day));
    if (integration_steps < 1) integration_steps = 1;

    int integration_offset_steps = (integration_offset_steps_cli >= 0)
        ? integration_offset_steps_cli
        : static_cast<int>(std::llround(std::max(0.0, integration_offset_days) * integ_steps_per_day));
    if (integration_offset_steps < 0) integration_offset_steps = 0;

    IntegrationWindow w;
    w.start = equilibration_steps + integration_offset_steps;
    if (w.start >= total_time_steps) w.start = std::max(0, total_time_steps - 1);

    w.steps = std::min(integration_steps, std::max(0, total_time_steps - w.start));
    if (w.steps < 1) w.steps = std::min(1, total_time_steps);
    return w;
}

static void setEffectiveAlbedoWindow(int start_step, int n_steps) {
    g_aeff_win_start = start_step;
    g_aeff_win_steps = n_steps;
    g_aeff_win_set   = true;
}

// True when ABSOLUTE timestep t belongs to the integration window. Before the
// window is published (it is derived after the step counts are known) nothing
// accumulates: a silent "accumulate everything" default is exactly the eq+output
// contamination this condition exists to remove, so refuse rather than guess --
// writeEffectiveAlbedo() warns loudly if that ever happens.
static inline bool aeffStepInWindow(int t) {
    return g_aeff_win_set && t >= g_aeff_win_start
                          && t <  g_aeff_win_start + g_aeff_win_steps;
}

// ---- Fermi condition C1: the same accumulation on the GPU thermal path --------
// The CPU accumulates A_eff inside calculateAbsorbedFlux, which the GPU solver
// never calls -- it builds its absorbed flux in prepare_flux() instead (two copies:
// the multi-GPU batch worker and the single-GPU serial loop). Every production
// lunar/Europa run is GPU, so both effective_albedo.bin and the PTM-106 self-check
// were absent exactly where they matter.
//
// This is deliberately a SEPARATE read-only pass rather than a hook inside
// prepare_flux: prepare_flux is parallel over TIMESTEPS, so per-face accumulation
// there would be a data race (or need atomics on the hot path), and any edit inside
// it risks perturbing the flux array the solver integrates. Here nothing is written
// except the two accumulators, the recipe is character-for-character prepare_flux's
// (same applyIncidenceAlbedo, same per-face w, same solarConstForStep), and the loop
// is parallel over FACES so face f is touched by exactly one thread.
//
// Called once per whole-mesh GPU thermal solve, before the batch dispatch, so it
// covers the single-GPU serial path and the multi-GPU pool path identically and
// exactly once. The selected-face re-thermal deliberately does NOT accumulate: it
// re-integrates a SUBSET with the same albedos, so accumulating there would
// double-weight those faces.
static void accumulateEffectiveAlbedoGPU(size_t n_faces,
                                         int total_time_steps,
                                         const std::vector<double>& direct_solar_values,
                                         const std::vector<double>& face_albedos,
                                         const std::vector<double>& face_w,
                                         bool use_incidence,
                                         bool use_hapke,
                                         double a_coef,
                                         double b_coef,
                                         double S_solar) {
    if (n_faces == 0 || total_time_steps <= 0) return;
    if (!g_aeff_win_set) {
        std::cerr << "[GPU Thermal] WARNING: integration window not published before the "
                     "thermal solve; skipping the PTM-107(a) effective-albedo accumulation "
                     "(statistics/effective_albedo.bin will be absent and the PTM-106 "
                     "self-check will report SKIPPED)." << std::endl;
        return;
    }
    const int t0 = std::max(0, g_aeff_win_start);
    const int t1 = std::min(total_time_steps, g_aeff_win_start + g_aeff_win_steps);
    if (t1 <= t0) return;

    std::vector<double> num(n_faces, 0.0), den(n_faces, 0.0);
    #pragma omp parallel for schedule(static)
    for (long long fi = 0; fi < static_cast<long long>(n_faces); ++fi) {
        const size_t f = static_cast<size_t>(fi);
        const double base_A = (f < face_albedos.size()) ? face_albedos[f] : 0.0;
        const double w_f    = (f < face_w.size()) ? face_w[f] : -1.0;
        double a_num = 0.0, a_den = 0.0;
        for (int t = t0; t < t1; ++t) {
            const size_t idx = f * static_cast<size_t>(total_time_steps)
                             + static_cast<size_t>(t);
            const double direct = (idx < direct_solar_values.size())
                                ? direct_solar_values[idx] : 0.0;
            if (direct <= 0.0) continue;
            const double albedo = use_incidence
                ? applyIncidenceAlbedo(base_A, direct, solarConstForStep(t, S_solar),
                                       a_coef, b_coef, use_hapke, w_f)
                : base_A;
            a_num += albedo * direct;
            a_den += direct;
        }
        num[f] = a_num;
        den[f] = a_den;
    }
    g_eff_albedo_num = std::move(num);
    g_eff_albedo_den = std::move(den);
    g_eff_albedo_valid = true;
}

// PTM-106 self-check. The whole claim of the fix is one identity:
//   the albedo the radiosity SCATTERS the direct beam with
//     == the albedo the surface BC did NOT absorb with.
// Both sides are now computed, so verify rather than assert by eye: the coupling
// loop accumulates its installed source albedo with the SAME flux weighting the BC
// uses above, and the two per-face vectors are differenced at the end of the run.
// A wrong recipe (Hayne where the BC runs Hapke, planet.S where the BC uses the
// per-step S_inst, a stale per-face A0) shows up here as a large residual instead
// of as a plausible-looking temperature shift nobody can falsify.
static std::vector<double> g_src_albedo_num;   // sum_t A_src*Q per face
static std::vector<double> g_src_albedo_den;
static bool g_src_albedo_valid = false;
// Set by the coupling loop when the PTM-106 source override is actually ACTIVE
// (two-band + incidence-dependent BC + no legacy hatch). Without it the report
// below cannot tell "this run has no identity to verify" (radiation off, gray
// radiosity, constant albedo, legacy hatch -- each of which prints its own
// banner) from "the identity exists and the check silently did not run", which
// is precisely the GPU defect this file's C1 change fixes.
static bool g_ptm106_expected = false;

static void reportSourceAlbedoConsistency() {
    // Rule 7: a self-check that cannot run must SAY SO. This used to return in
    // silence, which is how the whole check went missing on the GPU thermal path
    // for every production run (Fermi, 2026-08-05): g_eff_albedo_valid was set
    // only inside the CPU calculateAbsorbedFlux, so a GPU run printed nothing at
    // all and looked exactly like a run where the check had passed.
    if (!g_src_albedo_valid || !g_eff_albedo_valid) {
        // Nothing to verify when the PTM-106 override was never active (radiation
        // off, gray radiosity, no incidence-dependent albedo, legacy hatch): the
        // source IS the operator by construction, and each of those paths already
        // announces which albedo feeds the radiosity.
        if (!g_ptm106_expected) return;
        std::cerr << "  [radiation.PTM-106] WARNING: source-vs-BC albedo self-check SKIPPED "
                     "although the PTM-106 source override IS active -- "
                  << (g_src_albedo_valid
                        ? "the surface-BC effective-albedo accumulator is EMPTY (no thermal "
                          "flux path published it; streamed thermal storage does not)"
                        : "the radiosity source-albedo accumulator is EMPTY (the coupling loop "
                          "installed no per-timestep A(theta) inside the integration window)")
                  << ". The fix is installed but UNVERIFIED for this run -- do not quote it as "
                     "consistent." << std::endl;
        return;
    }
    const size_t n = std::min(g_src_albedo_den.size(), g_eff_albedo_den.size());
    double max_abs = 0.0;
    size_t worst = 0, n_cmp = 0;
    for (size_t f = 0; f < n; ++f) {
        if (g_src_albedo_den[f] <= 0.0 || g_eff_albedo_den[f] <= 0.0) continue;
        const double a_src = g_src_albedo_num[f] / g_src_albedo_den[f];
        const double a_bc  = g_eff_albedo_num[f] / g_eff_albedo_den[f];
        const double d = std::abs(a_src - a_bc);
        if (d > max_abs) { max_abs = d; worst = f; }
        ++n_cmp;
    }
    if (n_cmp == 0) {
        std::cerr << "  [radiation.PTM-106] WARNING: source-vs-BC albedo self-check had NO "
                     "comparable faces (accumulators empty). The fix may be installed but "
                     "UNVERIFIED for this run -- do not quote it as consistent." << std::endl;
        return;
    }
    std::ios_base::fmtflags fl(std::cout.flags());
    std::streamsize pr = std::cout.precision();
    std::cout << "[albedo] PTM-106 source-vs-BC albedo self-check over " << n_cmp
              << " ever-lit faces: max |A_radiosity - A_BC| = "
              << std::scientific << std::setprecision(3) << max_abs
              << " (face " << worst << ")";
    if (max_abs > 1e-9) {
        std::cout << "  <-- NOT CONSISTENT; the visible source is NOT the BC albedo."
                  << std::endl;
        std::cerr << "  [radiation.PTM-106] WARNING: radiosity source albedo disagrees with the "
                     "surface-BC albedo by up to " << max_abs
                  << "; reflected-solar energy does not close per face." << std::endl;
    } else {
        std::cout << "  -> consistent." << std::endl;
    }
    std::cout.flags(fl);
    std::cout.precision(pr);
}

// Write statistics/effective_albedo.bin (uint32 n + float32[n]) and report the
// global flux-weighted scalar. Returns false if there is nothing to write.
static bool writeEffectiveAlbedo(const std::string& output_dir) {
    if (!g_eff_albedo_valid || g_eff_albedo_den.empty()) {
        // Rule 7: an output the caller asked for that silently does not appear is
        // indistinguishable from one that appeared and read zero. Say which.
        std::cerr << "WARNING: statistics/effective_albedo.bin NOT written -- the per-face "
                     "effective-albedo accumulator was never filled (no thermal flux path "
                     "published it; streamed thermal storage does not). Any energy-closure "
                     "audit of this run has to treat the BC albedo as a free parameter again "
                     "(BUGS.md PTM-107a)." << std::endl;
        return false;
    }
    if (!g_aeff_win_set) {
        std::cerr << "WARNING: statistics/effective_albedo.bin NOT written -- the integration "
                     "window was never published, so the accumulation window cannot be stated "
                     "and would not be comparable with the other statistics/*.bin files."
                  << std::endl;
        return false;
    }
    const size_t n = g_eff_albedo_den.size();
    std::vector<double> a_eff(n);
    double num_tot = 0.0, den_tot = 0.0;
    size_t n_lit = 0;
    for (size_t f = 0; f < n; ++f) {
        if (g_eff_albedo_den[f] > 0.0) {
            a_eff[f] = g_eff_albedo_num[f] / g_eff_albedo_den[f];
            num_tot += g_eff_albedo_num[f];
            den_tot += g_eff_albedo_den[f];
            ++n_lit;
        } else {
            a_eff[f] = std::numeric_limits<double>::quiet_NaN();
        }
    }
    try {
        std::filesystem::create_directories(output_dir + "/statistics");
        const std::string fn = output_dir + "/statistics/effective_albedo.bin";
        std::ofstream out(fn, std::ios::binary);
        if (!out) {
            std::cerr << "WARNING: could not open " << fn
                      << " for the PTM-107(a) effective-albedo output." << std::endl;
            return false;
        }
        // SAME on-disk convention as every other statistics/*.bin: uint32 face
        // count, then float32[n]. Writing float64 here would silently mis-parse in
        // every existing reader (scripts/ptm_analyze, the GUI) at exactly half the
        // face count -- the PTM-67 class.
        const uint32_t n32 = static_cast<uint32_t>(n);
        out.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
        std::vector<float> a_eff_f(n);
        for (size_t f = 0; f < n; ++f) a_eff_f[f] = static_cast<float>(a_eff[f]);
        out.write(reinterpret_cast<const char*>(a_eff_f.data()),
                  static_cast<std::streamsize>(n * sizeof(float)));
        out.close();
        std::ios_base::fmtflags fl(std::cout.flags());
        std::streamsize pr = std::cout.precision();
        std::cout << "[albedo] PTM-107(a) effective surface albedo (flux-weighted over the "
                     "DIRECT beam, the value the BC absorbed with): global A_eff = "
                  << std::defaultfloat << std::setprecision(6)
                  << (den_tot > 0.0 ? num_tot / den_tot
                                    : std::numeric_limits<double>::quiet_NaN())
                  << " over " << n_lit << "/" << n << " ever-lit faces; per-face -> "
                  << fn << " (uint32 n + float32[" << n
                  << "], NaN where never lit).\n"
                  << "          Reduced over the INTEGRATION window, abs steps ["
                  << g_aeff_win_start << ","
                  << (g_aeff_win_start + g_aeff_win_steps)
                  << ") -- the same window as every other statistics/*.bin, so a closure "
                     "audit may divide the two directly." << std::endl;
        std::cout.flags(fl);
        std::cout.precision(pr);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "WARNING: PTM-107(a) effective-albedo output failed: "
                  << e.what() << std::endl;
        return false;
    }
}

// ---- BUGS.md PTM-106: radiosity visible-band SOURCE albedo -------------------
// The surface BC absorbs (1 - A(theta)) Q with the incidence-dependent Hayne/Hapke
// albedo, so the enclosure must receive A(theta) Q. Before this fix the two-band
// radiosity built its visible source from the STATIC base albedo A0 instead (0.12
// on the Moon vs ~0.6 at 80N grazing incidence), so (A(theta) - A0) Q of reflected
// sunlight left the budget entirely rather than reaching the shadowed floors.
// DEFAULT = FIXED. PTM_LEGACY_RADIOSITY_STATIC_ALBEDO=1 restores the pre-fix source
// for A/B work; it is announced loudly because it reinstates a known energy leak.
static inline bool legacyRadiosityStaticAlbedo() {
    static bool legacy = []() {
        const char* e = std::getenv("PTM_LEGACY_RADIOSITY_STATIC_ALBEDO");
        bool on = (e && std::string(e) != "0" && e[0] != '\0');
        if (on) {
            std::cerr << "[radiation.PTM-106] PTM_LEGACY_RADIOSITY_STATIC_ALBEDO=1: the two-band "
                         "visible SOURCE reverts to the STATIC base albedo. This REINSTATES the "
                         "PTM-106 energy leak at grazing incidence (reflected sunlight is lost "
                         "from the budget). Use for A/B comparison only.\n";
        }
        return on;
    }();
    return legacy;
}

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
    // When non-null, the per-face T(z) state AT THE END OF THE
    // EQUILIBRATION PHASE (before the output cycle starts) is COPIED
    // INTO this vector — layer-major SoA matching ThermalOpenCLSolver::buf_T_,
    // size n_layers × n_faces. The rad-coupling iter loop uses this so
    // re-thermal can upload exactly iter-0's post-eq state, skip its own
    // eq phase entirely, and run only the output cycle with the updated
    // (indirect-augmented) flux BC. This avoids re-running eq from a
    // cold-start in every rad iter, which numerically diverges from
    // iter-0's converged state. In single-phase configs (eq_dt == out_dt,
    // which includes EVERY warm-start chunk) there is no separate eq phase,
    // so the buffer is filled immediately BEFORE the output cycle — the state
    // that cycle actually starts from. (It previously was not filled at all on
    // that path, despite this comment claiming otherwise: BUGS.md PTM-78.)
    std::vector<float>* post_eq_T_out = nullptr,
    // Multi-GPU controls (see config_parser.h thermal.gpu_*).
    //   gpu_devices == 1 (default) AND gpu_device_ids empty → legacy
    //     single-device path, byte-identical to pre-multi-GPU behavior.
    //   gpu_devices == 0 / "auto" → use every enumerated GPU.
    //   gpu_device_ids non-empty → use exactly those device indices.
    //   gpu_batch_size = 0 keeps the legacy 6 GB / 500 K-face cap.
    int gpu_devices = 1,
    const std::vector<int>& gpu_device_ids = {},
    size_t gpu_batch_size = 0,
    int gpu_inter_batch_pause_ms = 0,
    // Per-GLOBAL-face time-mean indirect flux <q_ind>[f] (W/m^2) ADDED to the
    // EQUILIBRATION-phase absorbed flux only (deep radiation-coupling, pass 2).
    // Empty (default) = legacy: equilibration sees no indirect term. Indexed by
    // global face id, matching prepare_flux's global_f = batch_start + f.
    const std::vector<double>& eq_indirect_flux = {});

// GPU thermal solver for selected faces with pre-combined flux arrays
// Used in radiation-thermal coupling iterations where flux includes indirect radiation
//
// shared_thermal_pool: optional pre-built ThermalGPUDevicePool*. When the
// caller passes one (typed as void* to keep this declaration GPU-build
// agnostic), the function uses it instead of constructing a fresh pool.
// Hoisting the pool above the radiation-coupling iter loop eliminates the
// per-iter OpenCL context create/destroy churn that wedged the macOS+AMD
// driver during long re-thermal runs (kernel hang → SMC -108 forced reset).
// Pass nullptr to keep the legacy build-pool-per-call behavior.
bool runThermalForSelectedFaces_GPU(
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<int>& face_indices,
    // Raw illumination flux arrays (face-major: f * total_time_steps + t).
    // Same source data iter-0 uses — re-thermal applies Hayne + IR
    // multiplier internally, identical to iter-0's prepare_flux.
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    // Per-face per-timestep indirect flux from the radiation-coupling
    // module (already ε-scaled by RadiositySolver). Added on top of
    // iter-0's flux composition; produces the only physical difference
    // between iter-0 and rad-coupling iter N.
    const std::vector<std::vector<double>>& indirect_flux_series,
    // iter-0's POST-EQUILIBRATION T(z) state per face — same starting
    // point iter-0 used for its output cycle. Layer-major SoA matching
    // ThermalOpenCLSolver::buf_T_ (size n_layers × n_faces). When sized
    // for the mesh, re-thermal uploads this and SKIPS its own eq phase.
    const std::vector<float>& post_eq_T_state,
    const std::vector<double>& times,
    int total_time_steps,
    int equilibration_steps,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps,
    // Per-GLOBAL-face visible Bond albedo A0 (per-face albedo scaling). The
    // re-thermal composes Hayne on this per-face A0 and derives the per-face
    // Hapke single-scattering w from it (hapkeWFromA0), matching iter-0's flux
    // build instead of the body-average albedo. Indexed by global face id.
    const std::vector<double>& face_albedos,
    // Multi-GPU dispatch knobs (mirrors the all-faces path). Defaults preserve
    // legacy single-GPU serial-batches behavior.
    int gpu_devices = 1,
    const std::vector<int>& gpu_device_ids = {},
    int gpu_inter_batch_pause_ms = 0,
    void* shared_thermal_pool = nullptr,
    // Optional POST-COUPLING state hand-back (BUGS.md PTM-57). When non-null,
    // the T(z) column AT THE END of this call's output cycle is written into
    // final_T_inout (layer-major, GLOBAL face indexing, size n_layers ×
    // n_faces) for every face this call re-thermal'd, and the corresponding
    // final_T_touched[global_f] byte is set to 1. The chunked-run orchestrator
    // uses it to archive the coupled end-of-window state instead of iter-0's
    // un-coupled one. Faces never re-thermal'd keep iter-0's column, which is
    // correct for them — indirect did not move their temperature.
    std::vector<float>* final_T_inout = nullptr,
    std::vector<uint8_t>* final_T_touched = nullptr);

void saveCheckpoint(const std::string&, const std::vector<size_t>&,
    const std::vector<std::vector<double>>&);

bool addThermalDataToVTK(const std::string&, const std::shared_ptr<Mesh>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

// ---------------------------------------------------------------------------
// Heartbeat: single-line "where are we right now" file in <output_dir>.
// Rewritten before each long-running phase with explicit fsync(). Survives
// hard kernel hangs where stdout/stderr buffers never reach disk — the
// macOS+AMD OpenCL driver has wedged the box badly enough during multi-hour
// rad-coupling runs that the unified log captured nothing pre-reboot. The
// heartbeat is small enough that open-write-fsync-close is cheap (~ms) yet
// pinpoints the in-flight phase after the box comes back up. Best-effort;
// silently ignores I/O errors.
// ---------------------------------------------------------------------------
namespace {
std::string g_heartbeat_dir;
std::mutex  g_heartbeat_mtx;

void setHeartbeatDir(const std::string& dir) {
    std::lock_guard<std::mutex> lk(g_heartbeat_mtx);
    g_heartbeat_dir = dir;
}

void writeHeartbeat(const std::string& msg) {
    std::string dir;
    {
        std::lock_guard<std::mutex> lk(g_heartbeat_mtx);
        dir = g_heartbeat_dir;
    }
    if (dir.empty()) return;
    const std::string path = dir + "/heartbeat.txt";
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return;

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    std::fprintf(fp, "%s %s\n", ts, msg.c_str());
    std::fflush(fp);
    int fd = ::fileno(fp);
    if (fd >= 0) ::fsync(fd);
    std::fclose(fp);
}
}  // namespace

using FluxFn = std::function<double(size_t,int)>;
// No default arguments: `planet` (the angular albedo model) and
// `eq_indirect_flux` are exactly the two inputs this path used to drop on the
// floor, and a defaulted-away argument is how they went missing in the first
// place. The single caller passes all of them explicitly.
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
    const std::vector<double>* face_H_overrides,
    const thermal::Planet* planet,
    int equilibration_steps,
    const std::vector<double>& eq_indirect_flux);

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

// thermal::Config::NYEARSEQ as main() sets it. Named because the CPU analytic
// pre-conditioner length is reported in the pre-flight banner and computed by
// the SIMD equilibrator, and the two must not drift apart.
static constexpr int kCpuAnalyticSeedYears = 2;

// Length of the CPU paths' analytic clear-sky pre-conditioner, in diurnal
// cycles, for a body with a real seasonal year (Moon 24, Europa 2, Mercury 4).
// SINGLE SOURCE: the SIMD equilibrator sizes its seed with this and the
// pre-flight banner reports it with this, so the reported length cannot drift
// away from the integrated one. Generic bodies do NOT use it on the SIMD path
// (an asteroid's year is ~10^4 rotations); see the "using_generic_body" branch
// in runThermalModelsForAllFacesBatchSoA.
static double cpuAnalyticSeedCycles(const thermal::Planet& planet, int nyearseq) {
    if (!(planet.day > 0.0)) return 0.0;
    return static_cast<double>(nyearseq) * std::round(planet.year / planet.day);
}

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

// ---------------------------------------------------------------------------
// AbsorbedFluxComposer -- the ONE implementation of
//     Q_abs(f,t) = (1 - A_vis(f,t)) * (F_direct + F_reflected) + (1 - A_ir) * F_ir
// shared by every CPU thermal path.
//
// The recipe used to live only inside the array-shaped calculateAbsorbedFlux()
// below, so the STREAMED path -- which exists precisely so that face x timestep
// arrays are never materialized -- could not call it and carried an inline copy
// built on a constant per-face albedo. That copy silently dropped the
// incidence-angle albedo model (Hayne 2017 / Hapke A_h(i)), never published the
// per-face effective-albedo statistic, and had nowhere to add a supplied
// equilibration indirect flux. Splitting the routine in two removes the second
// implementation WITHOUT giving up the memory bound streaming exists for:
//   * build()    resolves the per-face albedo map and the angular model ONCE.
//                O(faces) memory -- three double vectors, no time dimension.
//   * absorbed() is the per-(face, timestep) kernel.
// The array caller loops over absorbed(); the streamed caller feeds it one
// (face, timestep) at a time straight from the chunked illumination store.
//
// THREADING: absorbed() accumulates the effective-albedo numerator/denominator
// into PER-FACE slots without atomics, exactly as the original array loop did.
// Every caller must therefore be parallel over FACES so that face f is touched
// by one thread only. Both callers are.
// ---------------------------------------------------------------------------
class AbsorbedFluxComposer {
public:
    // Resolve the per-face albedo and the angular albedo model. Returns false,
    // having said why, rather than silently substituting a uniform albedo.
    bool build(const std::shared_ptr<ptm::SpatialMap>& albedo_map,
               const std::shared_ptr<Mesh>& mesh,
               size_t num_faces,
               double default_visible_albedo,
               double default_ir_albedo,
               const thermal::Config& config,
               const thermal::Planet* planet,
               double solar_constant);

    // The operative visible albedo for (face, ABSOLUTE timestep). Verbatim from
    // the array loop this was extracted out of.
    double visibleAlbedo(size_t f, int t, double direct_flux) const {
        const double base_albedo = face_albedos_[f];
        double visible_albedo = base_albedo;

        // Compute incidence-dependent albedo if enabled and face is illuminated.
        // PTM-28 Fix A: recover cos(theta) against the per-timestep
        // instantaneous solar constant (falls back to the fixed value when
        // the table is legacy).
        if (use_incidence_albedo_ && direct_flux > 0.0) {
            double cos_theta = direct_flux / solarConstForStep(t, solar_constant_);
            cos_theta = std::min(1.0, std::max(0.0, cos_theta));
            double incidence_angle = std::acos(cos_theta);

            // Per-face albedo scaling (defect #2). When a per-face albedo map
            // was actually loaded, compose the incidence albedo on THIS face's
            // own A0 (base_albedo) -- Hapke via applyIncidenceAlbedo (the SAME
            // function + per-face w the GPU flux path uses, so CPU==GPU on
            // mapped faces), Hayne via the per-face getAlbedoWithIncidence
            // overload (identical pow polynomial). With NO map loaded, every
            // face has the uniform default albedo, so we take the EXACT prior
            // lon/lat path -- byte-identical (paper/polar protection).
            if (albedo_successfully_loaded_) {
                if (use_hapke_albedo_) {
                    visible_albedo = applyIncidenceAlbedo(
                        base_albedo, direct_flux,
                        solarConstForStep(t, solar_constant_),
                        0.0, 0.0, /*use_hapke=*/true, hapkeWFromA0(base_albedo));
                } else {
                    visible_albedo = planet_->getAlbedoWithIncidence(
                        base_albedo, incidence_angle);
                }
            } else {
                visible_albedo = planet_->getAlbedoWithIncidence(
                    mesh_->longitudes[f] * M_PI / 180.0,  // Convert to radians
                    mesh_->latitudes[f] * M_PI / 180.0,
                    incidence_angle);
            }
        }
        return visible_albedo;
    }

    // The per-(face, timestep) kernel. `t` is the ABSOLUTE timestep index.
    double absorbed(size_t f, int t, double direct_flux,
                    double reflected_flux, double ir_flux) {
        const double visible_albedo = visibleAlbedo(f, t, direct_flux);

        // PTM-107(a): record the OPERATIVE albedo the BC just decided on, flux-
        // weighted by the direct beam. Per-face accumulators under a loop that
        // is parallel over FACES, so face f is touched by one thread only.
        // C2: only over the INTEGRATION window (t is the ABSOLUTE step index --
        // the table spans equilibration + output). Sweeping the equilibration
        // block too made this file describe a different window from every other
        // statistics/*.bin and injected an equilibration-length-dependent
        // residual into any closure audit that used it.
        if (direct_flux > 0.0 && aeffStepInWindow(t)) {
            eff_alb_num_[f] += visible_albedo * direct_flux;
            eff_alb_den_[f] += direct_flux;
        }

        // Apply appropriate albedo to each component
        const double absorbed_direct    = direct_flux * (1.0 - visible_albedo);
        const double absorbed_reflected = reflected_flux * (1.0 - visible_albedo);
        const double absorbed_ir        = ir_flux * ir_emissivity_;

        // Sum all components
        return absorbed_direct + absorbed_reflected + absorbed_ir;
    }

    // PTM-107(a): publish the operative effective albedo for the statistics
    // writer. Call once, after every face has been composed.
    void publishEffectiveAlbedo() {
        g_eff_albedo_num = std::move(eff_alb_num_);
        g_eff_albedo_den = std::move(eff_alb_den_);
        g_eff_albedo_valid = true;
    }

private:
    const Mesh*            mesh_   = nullptr;
    const thermal::Planet* planet_ = nullptr;
    std::vector<double> face_albedos_;
    std::vector<double> eff_alb_num_;
    std::vector<double> eff_alb_den_;
    double ir_emissivity_  = 1.0;
    double solar_constant_ = 0.0;
    bool use_incidence_albedo_       = false;
    bool use_hapke_albedo_           = false;
    bool albedo_successfully_loaded_ = false;
};

// The explicit thermophysical `albedo_map` FITS key is opened by the CALLER on
// every path -- AbsorbedFluxComposer::build() consumes an already-loaded map, it
// never opens that key itself. Shared by the CPU SIMD-batch and streamed callers
// so the two cannot drift; `tag` is only the log prefix.
// (runThermalModelsForAllFaces keeps its own copy, which additionally prints the
// map's resolution/range/extent; it is left alone rather than have this helper
// grow a verbosity flag for one caller.)
static std::shared_ptr<ptm::SpatialMap> loadExplicitAlbedoMap(
    const thermal::Config& config, const char* tag) {
    std::shared_ptr<ptm::SpatialMap> map;
    if (config.custom_properties.find("albedo_map") == config.custom_properties.end()) {
        return map;
    }
    const std::string path = config.getProperty("albedo_map");
    if (path.empty()) return map;
    map = std::make_shared<ptm::SpatialMap>();
    if (!map->loadFromFITS(path)) {
        std::cout << tag << " Failed to load explicit albedo_map: " << path
                  << " (" << map->getLastError() << ")" << std::endl;
        map.reset();
    }
    return map;
}

bool AbsorbedFluxComposer::build(
    const std::shared_ptr<ptm::SpatialMap>& albedo_map,
    const std::shared_ptr<Mesh>& mesh,
    size_t num_faces,
    double default_visible_albedo,
    double default_ir_albedo,
    const thermal::Config& config,
    const thermal::Planet* planet,
    double solar_constant) {

    // Calculate IR emissivity from IR albedo
    ir_emissivity_  = 1.0 - default_ir_albedo;
    planet_         = planet;
    solar_constant_ = solar_constant;
    mesh_           = mesh ? mesh.get() : nullptr;

    // Validate mesh has expected face count
    if (!mesh || mesh->faces.size() < num_faces) {
        std::cerr << "Error: Mesh has fewer faces (" << (mesh ? mesh->faces.size() : 0)
                  << ") than expected (" << num_faces << ")" << std::endl;
        return false;
    }

    // Safety check: ensure latitude/longitude arrays are correctly sized
    if (mesh->latitudes.size() < num_faces || mesh->longitudes.size() < num_faces) {
        std::cerr << "Error: Mesh latitude/longitude arrays too small. "
                  << "Expected: " << num_faces
                  << ", Got: " << mesh->latitudes.size() << "/" << mesh->longitudes.size() << std::endl;
        return false;
    }

    // Pre-calculate albedo for each face. `face_albedos` is a reference so the
    // resolution logic below is character-for-character what it always was.
    face_albedos_.assign(num_faces, 0.0);
    std::vector<double>& face_albedos = face_albedos_;

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
                return false;
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
    
    // Apply the Hayne et al. (2017) incidence-angle correction whenever the
    // body's albedoCoef are set. When a spatially-varying albedo map is in
    // use, the per-face map value supersedes the body default A0; the angular
    // correction is *still* applied on top. Previously the gate also required
    // !using_albedo_map which silently dropped the high-θ correction for
    // map-using runs and biased nighttime temperatures at high latitudes by
    // several K.
    // Incidence-angle-dependent albedo is active when either (a) the body has
    // Hayne (2017) polynomial coefficients, or (b) the Hapke directional-
    // hemispheric gate is on. getAlbedoWithIncidence() internally routes to the
    // correct model (Hapke branch first when use_hapke_albedo). When neither is
    // set the constant Bond albedo is used and the path is byte-identical.
    bool use_hapke_albedo = (planet != nullptr && planet->use_hapke_albedo &&
                             planet->hapke_albedo_w >= 0.0);
    bool use_incidence_albedo = (planet != nullptr && solar_constant > 0.0 &&
                                 (planet->albedoCoef.size() >= 2 || use_hapke_albedo));

    if (use_incidence_albedo) {
        // PTM-55: report the OPERATIVE base albedo (the value face_albedos was
        // initialized to), not the Planet's, and say so when they differ. On
        // the CPU path the two agree today -- `planet` here IS the main-scope
        // Planet -- but the GPU path rebuilds its Planet, and a banner that can
        // silently mean two things is exactly what PTM-55 is.
        if (std::abs(planet->albedo - default_visible_albedo) > 1e-9) {
            // PTM-09 class: std::cerr/std::cout inherit whatever precision the
            // caller left set (measured: 0.07 and 0.120 both printed as "0.1"),
            // so a diagnostic that exists to show a DISAGREEMENT would print
            // two identical numbers. Format explicitly, then restore.
            std::ios_base::fmtflags f(std::cerr.flags());
            std::streamsize pr = std::cerr.precision();
            std::cerr << std::defaultfloat << std::setprecision(6)
                      << "[albedo] WARNING: Planet(\"" << planet->name
                      << "\") carries albedo " << planet->albedo
                      << ", which DISAGREES with the operative "
                      << default_visible_albedo
                      << " used to build face_albedos. See docs/KNOWN_LIMITATIONS.md (PTM-55)."
                      << std::endl;
            std::cerr.flags(f);
            std::cerr.precision(pr);
        }
        if (use_hapke_albedo) {
            std::cout << "Using directional-hemispheric (Hapke) albedo A_h(i) = "
                         "1 - gamma*H(mu0,w)" << std::endl;
            std::cout << "  Base albedo A0 = " << std::defaultfloat
                      << std::setprecision(6) << default_visible_albedo
                      << " (operative), single-scattering w = " << planet->hapke_albedo_w
                      << std::endl;
        } else {
            std::cout << "Using incidence-angle-dependent albedo (Hayne et al. 2017 formula)" << std::endl;
            std::cout << "  Base albedo A0 = " << std::defaultfloat
                      << std::setprecision(6) << default_visible_albedo
                      << " (operative)" << std::endl;
            std::cout << "  Coefficients: a = " << planet->albedoCoef[0]
                      << ", b = " << planet->albedoCoef[1] << std::endl;
        }
    }

    use_hapke_albedo_           = use_hapke_albedo;
    use_incidence_albedo_       = use_incidence_albedo;
    albedo_successfully_loaded_ = albedo_successfully_loaded;

    // PTM-107(a): per-face flux-weighted effective albedo accumulators. Sized
    // here (never inside a parallel region) and published to the file-scope
    // globals by publishEffectiveAlbedo(), so a run that composes flux more
    // than once reports the LAST (operative) composition, not a silent mixture.
    eff_alb_num_.assign(num_faces, 0.0);
    eff_alb_den_.assign(num_faces, 0.0);
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
    const thermal::Planet* planet = nullptr,  // Optional planet for incidence-dependent albedo
    double solar_constant = 0.0) {            // Solar constant for computing incidence angle

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

    AbsorbedFluxComposer composer;
    if (!composer.build(albedo_map, mesh, num_faces, default_visible_albedo,
                        default_ir_albedo, config, planet, solar_constant)) {
        return std::vector<double>();
    }

    std::vector<double> absorbed_flux(n_values);

    // Apply albedo to calculate absorbed flux for each face at each time step.
    // Parallel over FACES -- required by the composer's per-face accumulators.
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t f = 0; f < num_faces; f++) {
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

            absorbed_flux[idx] = composer.absorbed(
                f, static_cast<int>(t),
                direct_solar_values[idx],
                primary_reflected_values[idx],
                primary_ir_values[idx]);
        }
    }

    composer.publishEffectiveAlbedo();

    return absorbed_flux;
}

std::vector<double> runThermalModelWithMatchedTimes(
    const std::vector<double>& flux_values,
    const std::vector<double>& times,
    double latitude,
    const thermal::Config& config,
    int equilibration_steps,
    double Qb_override = -1.0,
    double H_override  = -1.0,
    // Global face index for the cross-path seed dump (PTM_DUMP_SEED) and the
    // seed guards. -1 = not a mesh face (point mode / standalone), record
    // nothing. `seed_flags` accumulates {floor_hits, above_500K_hits}.
    long long seed_record_face = -1,
    std::atomic<size_t>* seed_floor_faces = nullptr,
    std::atomic<size_t>* seed_hot_faces   = nullptr) {

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
        applyConfiguredThermo(planet, config);   // generic-body props + user scalar overrides (PTM-53)

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

        // Shared spin-up seed diagnostics. The constructor has already applied
        // the seed; record it against this face so the CSV can be diffed
        // against the SIMD-batch and GPU dumps (gate G1), and count the two
        // guards so they can be reported instead of swallowed.
        if (model.seed_applied) {
            if (model.seed_result.clamped_low && seed_floor_faces) ++(*seed_floor_faces);
            if (model.seed_result.above_500K  && seed_hot_faces)   ++(*seed_hot_faces);
            if (seed_record_face >= 0) {
                thermal::EqSeedInput rec;
                rec.mean_absorbed_flux = model.seed_mean_flux;
                rec.Qb                 = planet.Qb;
                thermal::EqSeedRecorder::instance().record(
                    static_cast<size_t>(seed_record_face), rec, model.seed_result);
            }
        }

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
            flux_arr(i, 1) = flux_values[i];      // Flux (W/m^2)
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

// Apply generic-body thermophysical properties that main() forwarded via the
// thermal_config "gb_*" properties. The thermal solvers below each rebuild a
// LOCAL Planet from the body NAME (config "planet"); for a non-built-in generic
// body (e.g. an asteroid) that name resolves to Europa PLACEHOLDER defaults
// (planetary_properties.h getPropertiesByName), so the configured surface
// conductivity ks / the ks->kd depth profile / cp0 / rho / H / cpCoeff (and the
// constant_thermal_inertia collapse, already baked into the main Planet) never
// reached the conduction -- thermal inertia was silently ignored for all EMA
// asteroid runs. Gated on "gb_thermo": built-in Moon/Europa runs (which do not
// set it) are byte-identical untouched. Call right after the local Planet is
// built and BEFORE reference_profile, so the depth grid / cp / kc all pick it up.
//
// PTM-53: the SECOND block below fixes the defect this gate created. The
// gb_thermo gate is correct for what it does -- it repairs a generic body whose
// name-resolved Planet is a placeholder -- but it also meant that the explicit
// scalar overrides (thermophysical.{ks,kd,emissivity,albedo}_override and the
// constant_thermal_inertia collapse), which main() resolves onto the MAIN-scope
// Planet and announces in the log, never reached a NAMED body's solver at all.
// A Moon run printed "Property override: ks 0.00074 -> 0.00148 W/m/K" and then
// solved with 0.00074: byte-identical output, announced success, zero effect.
// Those overrides are therefore forwarded per-knob and applied here on EVERY
// path. A config that sets no override forwards nothing, so baseline runs stay
// byte-identical.
static void applyConfiguredThermo(thermal::Planet& planet, const thermal::Config& config) {
    double v;
    // ---- generic bodies: repair the placeholder-resolved Planet -------------
    if (!config.getProperty("gb_thermo", "").empty()) {
        if ((v = config.getPropertyDouble("gb_ks",   -1.0)) >  0.0) planet.ks   = v;
        if ((v = config.getPropertyDouble("gb_kd",   -1.0)) >  0.0) planet.kd   = v;
        if ((v = config.getPropertyDouble("gb_cp0",  -1.0)) >= 0.0) planet.cp0  = v;
        if ((v = config.getPropertyDouble("gb_rhos", -1.0)) >  0.0) planet.rhos = v;
        if ((v = config.getPropertyDouble("gb_rhod", -1.0)) >  0.0) planet.rhod = v;
        if ((v = config.getPropertyDouble("gb_H",    -1.0)) >  0.0) planet.H    = v;
        if ((v = config.getPropertyDouble("gb_albedo",     -1.0)) >= 0.0) planet.albedo     = v;
        if ((v = config.getPropertyDouble("gb_emissivity", -1.0)) >  0.0) planet.emissivity = v;
        // Orbital/rotational scalars (see main): the name-resolved local Planet
        // otherwise carries Europa placeholder values for these on the generic path.
        if ((v = config.getPropertyDouble("gb_day",  -1.0)) >  0.0) planet.day  = v;
        if ((v = config.getPropertyDouble("gb_S",    -1.0)) >  0.0) planet.S    = v;
        if ((v = config.getPropertyDouble("gb_year", -1.0)) >  0.0) planet.year = v;
        if ((v = config.getPropertyDouble("gb_Qb",   -2.0)) >= 0.0) planet.Qb   = v;
        // cpCoeff forwarded as a comma-separated list; empty string => constant cp.
        planet.cpCoeff = parseCsvDoubles(config.getProperty("gb_cpCoeff", ""));
    }

    // ---- PTM-53: explicit user overrides, on EVERY body -------------------
    // "ovr_thermo" is the one-lookup fast path (mirroring the gb_thermo gate
    // above): this function runs PER FACE on the scalar CPU path, so the
    // overwhelmingly common no-override case must not pay for seven map probes
    // and an ostringstream.
    if (config.getProperty("ovr_thermo", "").empty()) return;

    // Presence of a key IS the request (main forwards a key only when the
    // matching config knob is set), so there is no sentinel to mis-read and an
    // override of any magnitude survives -- including one that would round to
    // zero in a sentinel comparison.
    static const struct { const char* key; double thermal::Planet::* field;
                          const char* label; const char* unit; } kScalarOverrides[] = {
        {"ovr_ks",         &thermal::Planet::ks,         "ks",         " W/m/K"},
        {"ovr_kd",         &thermal::Planet::kd,         "kd",         " W/m/K"},
        {"ovr_rhod",       &thermal::Planet::rhod,       "rhod",       " kg/m^3"},
        {"ovr_cp0",        &thermal::Planet::cp0,        "cp0",        " J/kg/K"},
        {"ovr_emissivity", &thermal::Planet::emissivity, "emissivity", ""},
        {"ovr_albedo",     &thermal::Planet::albedo,     "albedo A0",  ""},
        // PTM-161: `thermal.geothermal_heat_flow` / `--geothermal-heat-flow`
        // was parsed, applied to the main-scope Planet and announced, and then
        // discarded by every one of the four solver entry points, which each
        // rebuild a local Planet from the body NAME. On a built-in body the
        // knob did nothing on either backend while the log claimed success.
        // Gated by examples/moon_ico3_qb_falsifier.yaml.
        {"ovr_Qb",         &thermal::Planet::Qb,         "Qb",         " W/m^2"},
    };

    // Only build the echo string on the one call that will actually print it.
    static std::atomic<bool> logged{false};
    const bool want_echo = !logged.load(std::memory_order_relaxed);
    std::ostringstream echo;
    int applied = 0;
    const auto& props = config.custom_properties;

    for (const auto& ovr : kScalarOverrides) {
        auto it = props.find(ovr.key);
        if (it == props.end()) continue;
        v = std::stod(it->second);
        if (want_echo) {
            if (applied) echo << "; ";
            echo << ovr.label << ' ' << planet.*ovr.field << " -> " << v << ovr.unit;
        }
        planet.*ovr.field = v;
        ++applied;
    }
    // cpCoeff: an EMPTY forwarded list is meaningful (constant-TI mode wants
    // cp(T) collapsed to cp0), so the key's presence -- not its content -- is
    // the trigger.
    {
        auto it = props.find("ovr_cpCoeff");
        if (it != props.end()) {
            std::vector<double> coeff = parseCsvDoubles(it->second);
            if (want_echo && coeff != planet.cpCoeff) {
                if (applied) echo << "; ";
                echo << "cpCoeff[" << planet.cpCoeff.size() << "] -> [" << coeff.size() << "]";
            }
            planet.cpCoeff = std::move(coeff);
            ++applied;
        }
    }

    if (applied > 0) {
        g_thermo_override_applied.fetch_add(applied, std::memory_order_relaxed);
        // Echo ONCE per process. This line is the proof the override reached the
        // SOLVER -- the banner main() prints only proves it reached main.
        if (want_echo && !logged.exchange(true, std::memory_order_relaxed)) {
            std::cout << "[Thermo override] applied at solver: " << echo.str() << std::endl;
        }
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
    const std::vector<double>* face_H_overrides,
    const thermal::Planet* planet,
    int equilibration_steps,
    const std::vector<double>& eq_indirect_flux)
{
    if (!mesh || mesh->faces.empty() || total_time_steps <= 0) {
        std::cerr << "Streamed thermal: invalid mesh or time steps\n";
        return false;
    }

    const size_t n_faces = mesh->faces.size();

    // Absorbed-flux composition runs through the SAME AbsorbedFluxComposer every
    // array path uses. This block used to be a second, lesser implementation:
    // its own albedo-map search followed by an inline
    //     face_flux[t] = (1 - A_vis) * (Fd + Fr) + emiss_ir * Fir
    // with A_vis a per-face CONSTANT. That silently discarded the incidence-angle
    // albedo model, never published statistics/effective_albedo.bin, and had
    // nowhere to add a supplied equilibration indirect flux. build() is O(faces)
    // -- no time dimension -- so the memory bound this whole path exists to hold
    // is untouched; only the composition moved.
    //
    // The explicit `albedo_map` FITS key is opened by the caller on every path;
    // same helper the CPU SIMD-batch caller uses.
    std::shared_ptr<ptm::SpatialMap> streamed_albedo_map =
        loadExplicitAlbedoMap(config, "[Streamed]");

    std::cerr << "[Streamed] Composing absorbed flux "
                 "(per-face spatial + incidence-angle albedo)..." << std::endl;
    AbsorbedFluxComposer composer;
    if (!composer.build(streamed_albedo_map, mesh, n_faces, default_visible_albedo,
                        ir_albedo, config, planet, planet ? planet->S : 0.0)) {
        std::cerr << "[Streamed] Absorbed-flux composition could not be built -- "
                     "refusing to run against an unresolved albedo.\n";
        return false;
    }
    // Rule 7: the angular albedo model is unavailable without a Planet. Every
    // other CPU path is handed one; say so rather than quietly reverting to the
    // constant-albedo behavior this fix exists to remove.
    if (!planet) {
        std::cerr << "[Streamed] WARNING: no Planet was supplied, so the incidence-angle "
                     "albedo model A(theta) cannot be applied and a constant per-face "
                     "albedo is used. Every other CPU thermal path applies it.\n";
    }

    // Deep radiation-coupling: per-face time-mean indirect flux added to the
    // EQUILIBRATION-phase forcing only, i.e. absolute steps [0, equilibration_steps).
    // Same gate and same message as the array paths. Empty vector = legacy:
    // not one element is touched.
    if (!eq_indirect_flux.empty() && eq_indirect_flux.size() != n_faces) {
        std::cerr << "[Streamed] FATAL: eq_indirect_flux has "
                  << eq_indirect_flux.size() << " entries for " << n_faces
                  << " faces; refusing to equilibrate against a mismatched forcing.\n";
        return false;
    }
    const bool add_eq_indirect = (!eq_indirect_flux.empty() && equilibration_steps > 0);
    const int  eq_hi = std::min(equilibration_steps, total_time_steps);
    if (add_eq_indirect) {
        std::cerr << "[Streamed] Equilibration forcing carries the per-face mean"
                     " indirect flux over steps [0, " << eq_hi << ").\n";
    }

    const size_t CHUNK_SIZE = 100000;
    const size_t NUM_CHUNKS = (n_faces + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (surface_temps_out) {
        surface_temps_out->clear();
        surface_temps_out->resize(n_faces);
    }

    std::cerr << "\n[Streamed] Running thermal models for " << n_faces
              << " faces over " << total_time_steps << " steps..." << std::endl;

    // This path calls the CPU SCALAR solver, not the SIMD batch the banner
    // elsewhere implies -- so it gets the same shared seed through
    // runThermalModelWithMatchedTimes.
    std::atomic<size_t> seed_floor_faces{0};
    std::atomic<size_t> seed_hot_faces{0};
    if (config.spinup_seed == thermal::EqSeedMode::Physical) {
        thermal::EqSeedRecorder::instance().reserve(n_faces);
        std::cerr << "[eq-seed] equilibration seed: SHARED per-face conductive "
                     "steady profile [thermal.spinup_seed: physical; streamed "
                     "path uses the CPU SCALAR solver]" << std::endl;
    }

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

            // Build on-the-fly absorbed flux for this face across all times.
            // Same kernel the array paths call; the loop is parallel over FACES,
            // which is what the composer's per-face accumulators require.
            std::vector<double> face_flux(total_time_steps);
            for (int t = 0; t < total_time_steps; ++t) {
                face_flux[t] = composer.absorbed(f, t,
                                                 get_direct(f, t),
                                                 get_reflected(f, t),
                                                 get_ir(f, t));
            }
            if (add_eq_indirect) {
                for (int t = 0; t < eq_hi; ++t) face_flux[t] += eq_indirect_flux[f];
            }

            // Solve thermal for this face with matched times
            double face_qb = (face_Qb && f < face_Qb->size()) ? (*face_Qb)[f] : -1.0;
            // Per-face H from the property map (-1 = use planet default).
            double face_H = (face_H_overrides && f < face_H_overrides->size())
                          ? (*face_H_overrides)[f]
                          : -1.0;
            std::vector<double> temps = runThermalModelWithMatchedTimes(
                face_flux, times, mesh->latitudes[f], config, /*equilibration_steps*/0,
                face_qb, face_H, static_cast<long long>(f),
                &seed_floor_faces, &seed_hot_faces);

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

    // PTM-107(a): hand the per-face effective albedo to the statistics writer.
    // Without this a streamed run silently shipped no effective_albedo.bin.
    composer.publishEffectiveAlbedo();

    if (seed_floor_faces.load() > 0) {
        std::cerr << "[Streamed] WARNING: " << seed_floor_faces.load() << " of "
                  << n_faces << " faces seeded at the " << thermal::ABSOLUTE_ZERO_K
                  << " K CMB floor (Qb <= 0 AND zero mean absorbed flux).\n";
    }
    if (seed_hot_faces.load() > 0) {
        std::cerr << "[Streamed] WARNING: " << seed_hot_faces.load() << " of "
                  << n_faces << " faces seed above " << thermal::MAX_PHYSICAL_TEMP_K
                  << " K, the CPU solver's hard clamp.\n";
    }
    thermal::EqSeedRecorder::instance().write("cpu-streamed");

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
    const std::vector<double>* face_Qb,
    const std::vector<double>& eq_indirect_flux) {
    
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

    // Deep radiation-coupling: per-face time-mean indirect flux on the
    // EQUILIBRATION-phase forcing only (absolute steps [0, equilibration_steps)).
    // Empty vector = legacy, not one element touched.
    if (!eq_indirect_flux.empty() && equilibration_steps > 0) {
        if (eq_indirect_flux.size() != n_faces) {
            std::cerr << "[Thermal] FATAL: eq_indirect_flux has "
                      << eq_indirect_flux.size() << " entries for " << n_faces
                      << " faces; refusing to equilibrate against a mismatched"
                         " forcing.\n";
            return false;
        }
        const int eq_hi = std::min(equilibration_steps, total_time_steps);
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < n_faces; ++f) {
            const size_t base = f * static_cast<size_t>(total_time_steps);
            for (int t = 0; t < eq_hi; ++t) {
                if (base + t < absorbed_flux.size())
                    absorbed_flux[base + t] += eq_indirect_flux[f];
            }
        }
        std::cerr << "[Thermal] Equilibration forcing carries the per-face mean"
                     " indirect flux over steps [0, " << eq_hi << ").\n";
    }
    
    // Print initial message
    std::cerr << "\nRunning thermal models for " << n_faces << " mesh faces..." << std::endl;

    // Shared spin-up seed (thermal/eq_seed.h). This path used to seed every
    // column from ONE global T_avg = T_ss/sqrt(2) -- flux-blind AND
    // latitude-blind -- and then integrate 24 lunations of analytic clear-sky
    // forcing on top. Both are gone under spinup_seed: physical.
    std::atomic<size_t> seed_floor_faces{0};
    std::atomic<size_t> seed_hot_faces{0};
    if (config.spinup_seed == thermal::EqSeedMode::Physical) {
        thermal::EqSeedRecorder::instance().reserve(n_faces);
        std::cerr << "[eq-seed] equilibration seed: SHARED per-face conductive "
                     "steady profile -- T_0=((<F_abs>+Qb)/(eps*sigma))^0.25 with "
                     "<F_abs> the mean over all " << total_time_steps
                  << " forcing steps, then T(z)=T_0+Qb*integral dz/k "
                     "[thermal.spinup_seed: physical; identical on every solver "
                     "path]" << std::endl;
    } else {
        std::cerr << "[eq-seed] equilibration seed: LEGACY CPU scalar global "
                     "T_avg = T_ss/sqrt(2), flux-blind and latitude-blind, "
                     "blended surface->deep over the skin depth "
                     "[thermal.spinup_seed: legacy]" << std::endl;
    }

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
                    face_H,
                    static_cast<long long>(f),
                    &seed_floor_faces,
                    &seed_hot_faces
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
    
    // Shared-seed guards, reported rather than swallowed.
    if (seed_floor_faces.load() > 0) {
        std::cerr << "WARNING: " << seed_floor_faces.load() << " of " << n_faces
                  << " faces seeded at the " << thermal::ABSOLUTE_ZERO_K
                  << " K CMB floor (Qb <= 0 AND zero mean absorbed flux)."
                  << std::endl;
    }
    if (seed_hot_faces.load() > 0) {
        std::cerr << "WARNING: " << seed_hot_faces.load() << " of " << n_faces
                  << " faces have a cycle-mean radiative-equilibrium seed above "
                  << thermal::MAX_PHYSICAL_TEMP_K << " K, the CPU solver's hard "
                     "clamp; the solution WILL be truncated there." << std::endl;
    }
    thermal::EqSeedRecorder::instance().write("cpu-scalar");

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

// Phase R / Fermi PORT-EQ-01: per-face mean-flux equilibration seed.
// The legacy Moon/Europa/A3GT seed derives T(z,0) from the INSTANTANEOUS
// absorbed flux over the first 10 equilibration steps with a hard 100 K
// nightside floor — so a PSR floor (which never sees the sun) is stamped at
// ~100 K and, with no radiation to drain it, its deep column holds ~50 K across
// the whole run (the 2026-07-21 radoff-PSR-warmth finding). When
// PTM_EQ_PERFACE_FLUX is set, each column is instead seeded from ITS OWN
// time-mean absorbed flux (the DC term of the forcing provider — the degenerate
// case of the Fourier-seed architecture) plus Qb, so a PSR floor cold-starts at
// the geothermal steady state T_geo=(Qb/(eps*sigma))^0.25 (~24.7 K on the Moon),
// while sunlit faces (mean flux ~ clear-sky) equilibrate to ~the same state as
// before. Default OFF => byte-identical to every established baseline.
static bool eqPerfaceFluxMode() {
    static bool mode = []() {
        const char* e = std::getenv("PTM_EQ_PERFACE_FLUX");
        const bool on = (e && std::string(e) != "0" && e[0] != '\0');
        std::cerr << "[eq-seed] LEGACY GPU equilibration seed: "
                  << (on ? "PER-FACE time-mean absorbed flux (PSR floors cold-start "
                           "at the geothermal steady state) [PTM_EQ_PERFACE_FLUX=1]"
                         : "first-10-step instantaneous flux with 100 K "
                           "nightside floor [PTM_EQ_PERFACE_FLUX unset]")
                  << std::endl;
        return on;
    }();
    return mode;
}

// ---------------------------------------------------------------------------
// SHARED SEED, GPU host side. Fills the persistent solver's initial-temperature
// buffer from thermal::eqSeedColumn() -- the same function Profile::
// seedFromMeanFlux (CPU scalar/streamed/point) and the CPU SIMD batch call.
//
// The seed is computed in float64 and cast once, so the only CPU/GPU difference
// in the initial condition is the float32 storage of the final value.
//
// What it REPLACES on this path (all three of these are gone under
// thermal.spinup_seed: physical):
//   * `if (T_eq < 40) T_eq = 100.0` -- a warm substitution that stamped every
//     permanently shadowed face at 100 K and, with no radiation to drain it,
//     held its deep column ~50 K for the whole run;
//   * a 10-step INSTANTANEOUS flux mean, which is a body-fixed day/night
//     snapshot -- on a spinning asteroid that is exactly the longitude
//     discontinuity that shows up as the "seam";
//   * the `using_generic_body` branch's flux-blind isothermal reference
//     profile, which removed the seam by removing the per-face information.
//     The full-span TIME mean is longitude-independent on a spinning body by
//     construction, so it does not reintroduce the seam -- verified by
//     measuring the seed's longitude spread at fixed latitude.
// ---------------------------------------------------------------------------
struct GpuSharedSeedContext {
    const thermal::Planet* planet = nullptr;
    const double* z               = nullptr;   // reference grid, n_layers nodes
    size_t n_layers               = 0;
    // Material properties are taken in DOUBLE from the reference profile, not
    // read back out of the float32 upload buffers. Reading the float32 copies
    // would make the GPU seed differ from the CPU seed by the rounding of kc
    // and rho -- i.e. it would break gate G1 for no reason. The final T is
    // cast to float32 once, which is the GPU's storage precision and is the
    // only remaining CPU/GPU difference in the initial condition.
    const double* kc_ref          = nullptr;   // reference profile kc(z)
    const double* rho_ref         = nullptr;   // reference profile rho(z)
    const std::vector<double>* face_H_overrides = nullptr;  // may be null/empty
    const std::vector<double>* face_Qb          = nullptr;  // may be null
    size_t batch_start            = 0;
    size_t batch_n                = 0;
    /// The absorbed flux for (global face, absolute step), composed by the
    /// SAME expression the solver's own prepare_flux uses -- including the
    /// incidence-angle (Hayne / Hapke) albedo. Passing a callable rather than
    /// the raw direct/reflected/IR arrays is deliberate: the GPU seed used to
    /// re-derive absorbed flux inline with a FLAT albedo, twenty lines from a
    /// prepare_flux that applies A(i), so the seed disagreed with the forcing
    /// the same function then uploaded (measured: up to 39.5 K on the seed of
    /// a lunar 80N crater face).
    const std::function<double(size_t, int)>* absorbed = nullptr;
    int    total_time_steps = 0;
    double R350             = 0.0;
    thermal::ConductivityModel cond = thermal::ConductivityModel::Hayne2017;
};

static void gpuFillSharedSeed(const GpuSharedSeedContext& c,
                              std::vector<float>& T_initial,
                              size_t& floor_faces, size_t& hot_faces) {
    std::vector<double> z_col(c.n_layers), kc_col(c.n_layers),
                        rho_col(c.n_layers), T_col(c.n_layers);
    for (size_t i = 0; i < c.n_layers; ++i) z_col[i] = c.z[i];
    const thermal::Planet& pl = *c.planet;
    const bool use_per_face_H = (c.face_H_overrides != nullptr) &&
                                !c.face_H_overrides->empty();

    for (size_t f = 0; f < c.batch_n; ++f) {
        const size_t global_f = c.batch_start + f;

        // Time mean of THIS face's absorbed flux over the WHOLE forcing span,
        // and the zero-thermal-inertia bracket on the cycle-mean surface
        // temperature (report-only; see EqSeedInput::mean_inst_radeq_T).
        const double qb_f = (c.face_Qb && global_f < c.face_Qb->size())
                                ? (*c.face_Qb)[global_f] : pl.Qb;
        double avg_flux = 0.0, sum_inst_T = 0.0;
        const int n_samples = std::max(1, c.total_time_steps);
        for (int t = 0; t < n_samples; ++t) {
            const double Ft = (*c.absorbed)(global_f, t);
            avg_flux += Ft;
            const double q = (Ft > 0.0 ? Ft : 0.0) + qb_f;
            sum_inst_T += (q > 0.0)
                ? std::sqrt(std::sqrt(q / (pl.emissivity * thermal::SIGMA_SB)))
                : 0.0;
        }
        avg_flux /= n_samples;
        sum_inst_T /= n_samples;

        // Same expression as the float32 upload loop above, in double.
        if (use_per_face_H) {
            const double H_face = (global_f < c.face_H_overrides->size())
                                ? (*c.face_H_overrides)[global_f] : pl.H;
            for (size_t layer = 0; layer < c.n_layers; ++layer) {
                const double df = std::exp(-z_col[layer] / H_face);
                rho_col[layer] = pl.rhod + (pl.rhos - pl.rhod) * df;
                kc_col[layer]  = pl.kd   + (pl.ks   - pl.kd  ) * df;
            }
        } else {
            for (size_t layer = 0; layer < c.n_layers; ++layer) {
                kc_col[layer]  = c.kc_ref[layer];
                rho_col[layer] = c.rho_ref[layer];
            }
        }

        thermal::EqSeedInput in;
        in.mean_absorbed_flux = avg_flux;
        in.Qb                 = qb_f;
        in.emissivity         = pl.emissivity;
        in.mean_inst_radeq_T  = sum_inst_T;   // report-only
        // The OpenCL kernels carry this value as a literal (they cannot include
        // a C++ header); keeping the host seed on the same constant is what
        // makes gate G1 exact between the CPU and GPU paths.
        in.sigma              = thermal::SIGMA_SB;
        in.z                  = z_col.data();
        in.kc                 = kc_col.data();
        in.rho                = rho_col.data();
        in.n                  = static_cast<int>(c.n_layers);
        in.R350               = c.R350;
        in.cond               = c.cond;

        const thermal::EqSeedResult res = thermal::eqSeedColumn(in, T_col.data());
        if (res.clamped_low) ++floor_faces;
        if (res.above_500K)  ++hot_faces;
        thermal::EqSeedRecorder::instance().record(global_f, in, res);

        for (size_t layer = 0; layer < c.n_layers; ++layer)
            T_initial[layer * c.batch_n + f] = static_cast<float>(T_col[layer]);
    }
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
    const std::vector<double>* face_H_overrides,
    const std::vector<double>& eq_indirect_flux)
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
    applyConfiguredThermo(planet, config);   // generic-body props + user scalar overrides (PTM-53)
    // Per-face albedo scaling (defect #1): propagate the Hapke directional-albedo
    // gate from config to THIS mesh-flux planet. Without this the CPU SIMD-batch
    // path left use_hapke_albedo default-OFF and always ran Hayne, even when the
    // config/GPU selected Hapke — so CPU and GPU used different albedo models.
    planet.use_hapke_albedo = config.getPropertyBool("use_hapke_albedo", false)
                              && config.getPropertyDouble("hapke_albedo_w", -1.0) >= 0.0;
    planet.hapke_albedo_w   = config.getPropertyDouble("hapke_albedo_w", -1.0);

    std::shared_ptr<ptm::SpatialMap> batch_albedo_map =
        loadExplicitAlbedoMap(config, "[SIMD Batch]");

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

    // Deep radiation-coupling: add the per-face time-mean indirect flux to the
    // EQUILIBRATION-phase forcing only, i.e. absolute steps
    // [0, equilibration_steps). Those steps are the real-ephemeris spin-up
    // prefix this solver integrates before the output window (the analytic
    // clear-sky pre-equilibration above is a separate, flux-independent seed
    // and is deliberately left alone). The output window gets the time-varying
    // indirect from the coupling loop, so this cannot double-count.
    // Empty vector = legacy: not a single element is touched.
    if (!eq_indirect_flux.empty() && equilibration_steps > 0) {
        if (eq_indirect_flux.size() != n_faces) {
            std::cerr << "[SIMD Batch] FATAL: eq_indirect_flux has "
                      << eq_indirect_flux.size() << " entries for " << n_faces
                      << " faces; refusing to equilibrate against a mismatched"
                         " forcing.\n";
            return false;
        }
        const int eq_hi = std::min(equilibration_steps, total_time_steps);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < eq_hi; ++t) {
            double* col = &flux_column_major[static_cast<size_t>(t) * n_faces];
            for (size_t f = 0; f < n_faces; ++f) col[f] += eq_indirect_flux[f];
        }
        std::cerr << "[SIMD Batch] Equilibration forcing carries the per-face mean"
                     " indirect flux over steps [0, " << eq_hi << ").\n";
    }

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
    // Faces whose time-mean absorbed flux came back negative or non-finite. The
    // seed below substitutes 0 for those so it cannot emit a NaN column; that
    // substitution is reported after the loop rather than applied silently.
    std::atomic<size_t> bad_flux_faces{0};
    // Same contract for the per-face interior heat flux: a negative or
    // non-finite Qb is substituted with 0 in the seed, and counted, so the
    // substitution is reported instead of silently swallowed.
    std::atomic<size_t> bad_qb_faces{0};
    // Seed guards, reported after the loop (never silent):
    //   seed_floor_faces -- columns pinned at ABSOLUTE_ZERO_K (Qb <= 0 AND no
    //                       absorbed flux; cp(T) is undefined below that);
    //   seed_hot_faces   -- columns whose radiative-equilibrium seed exceeds
    //                       the CPU solver's own MAX_PHYSICAL_TEMP_K clamp, so
    //                       the run WILL be truncated by that clamp (Mercury).
    std::atomic<size_t> seed_floor_faces{0};
    std::atomic<size_t> seed_hot_faces{0};
    auto start_time = std::chrono::high_resolution_clock::now();

    const thermal::EqSeedMode seed_mode = config.spinup_seed;
    thermal::EqSeedRecorder::instance().reserve(n_faces);

    // State the seed every run, and warn if a caller set a toggle we are not
    // going to honor. PTM_EQ_PERFACE_FLUX used to gate the GPU path ONLY, so a
    // config could set it in engine_env, run on CPU, and silently get a
    // different initial condition than it asked for.
    if (seed_mode == thermal::EqSeedMode::Physical) {
        std::cerr << "[eq-seed] equilibration seed: SHARED per-face conductive "
                     "steady profile -- T_0=((<F_abs>+Qb)/(eps*sigma))^0.25 "
                     "with <F_abs> the mean over all " << total_time_steps
                  << " forcing steps, then T(z)=T_0+Qb*integral dz/k "
                     "[thermal.spinup_seed: physical; identical on every solver "
                     "path]" << std::endl;
    } else {
        const int seed_window = std::min(100, total_time_steps);
        std::cerr << "[eq-seed] equilibration seed: LEGACY CPU SIMD-batch "
                     "per-face steady state T=((<F_abs>+Qb)/(eps*sigma))^0.25, "
                     "<F_abs> averaged over the first " << seed_window
                  << " forcing step(s), blended surface->deep over the skin "
                     "depth [thermal.spinup_seed: legacy]" << std::endl;
    }
    {
        const char* e_perface = std::getenv("PTM_EQ_PERFACE_FLUX");
        const char* e_legacy  = std::getenv("PTM_EQ_LEGACY_SEED");
        if ((e_perface && e_perface[0] != '\0') ||
            (e_legacy  && e_legacy[0]  != '\0')) {
            std::cerr << "[eq-seed] WARNING: ";
            if (e_perface && e_perface[0] != '\0') {
                std::cerr << "PTM_EQ_PERFACE_FLUX is set but is superseded by "
                             "thermal.spinup_seed (the per-face seed is now the "
                             "default on EVERY path); ";
            }
            if (e_legacy && e_legacy[0] != '\0') {
                std::cerr << "PTM_EQ_LEGACY_SEED is set but is read by no code "
                             "path in this build -- use thermal.spinup_seed: "
                             "legacy; ";
            }
            std::cerr << "NO effect on this run. The seed named above is the "
                         "one actually used." << std::endl;
        }
    }

    // Loop-invariant radiative denominator for the seed below. Guard it ONCE and
    // loudly: with emissivity <= 0 the old seed divided by zero, got +inf, and
    // silently clamped every face to 400 K. Refuse the run instead.
    const double sb_denom = planet.emissivity * SIGMA_SB;
    if (!(sb_denom > 0.0)) {
        std::cerr << "ERROR: emissivity must be > 0 for the thermal seed "
                     "(got " << planet.emissivity << "). Refusing to run: the "
                     "surface energy balance eps*sigma*T^4 = F_abs + Qb has no "
                     "solution." << std::endl;
        return false;
    }

    // Create progress bar
    thermal::ProgressBar progressBar(static_cast<int>(n_faces), "SIMD Batch thermal", 50, true);

    // ------------------------------------------------------------------
    // PTM-51 diagnostic (env PTM_DUMP_EQ_SEED=<path.csv>, default OFF).
    //
    // PTM-51 documents a real, measured residual: Moon topographic faces with
    // EXACTLY zero absorbed flux equilibrate to ~63.6 K against T_geo = 24.04 K.
    // Its stated MECHANISM -- an active 50 K cap on the seed's deep column --
    // was disputed (Fermi, 2026-08-04) as ~20x too small to supply that heat,
    // with the analytic clear-sky EQUILIBRATION FORCING proposed instead. The
    // two hypotheses make opposite, checkable predictions for a zero-flux face:
    //   PTM-51 as written : T_deep - T_surf_est = 50 K   and eq forcing = 0
    //   Fermi             : T_deep - T_surf_est ~ 2.5 K  and eq forcing > 0
    // This dump writes both quantities per face so the question is settled by
    // measurement rather than by arithmetic on either side. Zero cost and
    // byte-identical when the variable is unset.
    // ------------------------------------------------------------------
    const char* eq_seed_dump_path = std::getenv("PTM_DUMP_EQ_SEED");
    const bool dump_eq_seed = (eq_seed_dump_path != nullptr && *eq_seed_dump_path != '\0');
    std::vector<double> dbg_avg_flux, dbg_T_surf_est, dbg_dT_geo,
                        dbg_eqflux_sum, dbg_eqflux_max;
    std::vector<long long> dbg_eqflux_n;
    if (dump_eq_seed) {
        dbg_avg_flux.assign(n_faces, 0.0);
        dbg_T_surf_est.assign(n_faces, 0.0);
        dbg_dT_geo.assign(n_faces, 0.0);
        dbg_eqflux_sum.assign(n_faces, 0.0);
        dbg_eqflux_max.assign(n_faces, 0.0);
        dbg_eqflux_n.assign(n_faces, 0);
        std::cout << "[PTM-51 dump] PTM_DUMP_EQ_SEED=" << eq_seed_dump_path
                  << " -- per-face equilibration seed + forcing will be written"
                  << std::endl;
    }

    bool success = true;

    // Equilibration duration. Hoisted out of the batch loop (it depends only
    // on config/planet/times, never on the batch) so it is computed once and
    // the spin-up banner can quote the configured cycle count.
    // The analytic clear-sky pre-conditioner. Read thermal/eq_seed.h before
    // touching this: it is a FORCING with no topography, no shadowing, no IR
    // and no indirect flux, it is NOT counted by equilibration_orbits, the GPU
    // has no analog, and it destroys the per-face initial condition computed
    // above (Fermi measured a permanently shadowed lunar column go 24.043 K ->
    // 64.375 K surface / +60 K at 1 m under 24 lunations of it). Under
    // spinup_seed: physical it is OFF unless a config asks for it by name.
    double equil_duration_s_shared;
    const double analytic_seed_cycles_cfg = thermal::analyticSeedCycles(
        seed_mode, config.analytic_seed_cycles, planet.day, planet.year,
        config.NYEARSEQ);
    if (analytic_seed_cycles_cfg <= 0.0) {
        equil_duration_s_shared = 0.0;
        if (seed_mode == thermal::EqSeedMode::Physical) {
            std::cerr << "[eq-seed] analytic clear-sky pre-conditioner: OFF "
                         "(thermal.analytic_seed_cycles: 0). Spin-up is the "
                         "configured equilibration under the REAL forcing."
                      << std::endl;
        }
    } else if (config.analytic_seed_cycles >= 0.0) {
        // Explicitly requested, whatever the mode.
        equil_duration_s_shared = analytic_seed_cycles_cfg * planet.day;
        std::cerr << "[eq-seed] WARNING: analytic clear-sky pre-conditioner ON "
                     "by request -- " << analytic_seed_cycles_cfg
                  << " diurnal cycles of cos(lat)*cos(h) insolation with NO "
                     "topography, NO shadowing, NO IR and NO indirect flux, "
                     "integrated BEFORE the real forcing and NOT counted by "
                     "equilibration_orbits. It overwrites the per-face seed."
                  << std::endl;
    } else if (config.getProperty("using_generic_body") == "true") {
        // Generic (fast-rotating) body: equilibrate for the CONFIGURED number of
        // DIURNAL cycles, not NYEARSEQ *seasonal* years. equilibration_steps was
        // sized in main as (output cadence)x(equilibration_orbits)x(diurnal
        // period), so equilibration_steps x output-cadence dt = the intended
        // equilibration duration in diurnal cycles. NYEARSEQ*round(year/day) here
        // would be ~10^4 four-hour cycles (an asteroid's year is that many diurnal
        // periods) and never finish. Gated -> Moon/Europa untouched.
        const double dt_out = (total_time_steps > 1) ? (times[1] - times[0]) : planet.day;
        const double eq_secs = static_cast<double>(equilibration_steps) * dt_out;
        equil_duration_s_shared = (eq_secs > 0.0) ? eq_secs : (40.0 * planet.day);
    } else {
        equil_duration_s_shared =
            cpuAnalyticSeedCycles(planet, config.NYEARSEQ) * planet.day;
    }

    // ---- Equilibration spin-up estimate (FIX 0) ----------------------------
    // Report-only and DEFAULT ON: it changes no solver state and the
    // statistics are byte-identical either way, so gating it off would only
    // hide the one number a user benefits from. PTM_EQ_CERTIFICATE=0 opts out.
    const double eq_target_K = config.getPropertyDouble("eq_target_K", 0.1);
    std::unique_ptr<thermal::EqSpinupCensus> g_spinup_census;
    if (thermal::eqCertificateEnabled()) {
        g_spinup_census.reset(new thermal::EqSpinupCensus(
            planet.day, eq_target_K,
            /*precision_floor_K=*/0.0,
            /*seconds_per_cycle=*/0.0));   // nothing measures it; banner says so
    }

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
                //
                // The window is the WHOLE forcing span under the shared seed.
                // The legacy window was the first min(100, N) steps, which on
                // the shipped lunar crater is 0.83 of a lunation starting at an
                // arbitrary phase — a phase-biased sample, not a cycle mean,
                // and a different quantity from the GPU path's full-span mean.
                // Same span on every path is what makes gate G1 exact.
                double avg_flux = 0.0;
                // REPORT-ONLY companion: the zero-thermal-inertia estimate of
                // the cycle-mean surface temperature. The seed's anchor is the
                // infinite-inertia limit (<T^4>)^0.25; this is the other
                // bracket. Dumped, never used by the solver. See the note on
                // EqSeedInput::mean_inst_radeq_T.
                double sum_inst_T = 0.0;
                const int flux_avg_window =
                    (seed_mode == thermal::EqSeedMode::Physical)
                        ? total_time_steps
                        : std::min(100, total_time_steps);
                for (int t = 0; t < flux_avg_window; ++t) {
                    const double Ft = flux_column_major[t * n_faces + global_f];
                    avg_flux += Ft;
                    const double q = (Ft > 0.0 ? Ft : 0.0) + batch.Qb[f];
                    sum_inst_T += (q > 0.0)
                        ? std::sqrt(std::sqrt(q / (batch.emissivity[f] * batch.sigma_sb)))
                        : 0.0;
                }
                avg_flux /= std::max(1, flux_avg_window);
                sum_inst_T /= std::max(1, flux_avg_window);

                // PTM-46: seed the column at the steady state this face can
                // actually hold, eps*sigma*T^4 = <F_abs> + Qb. Including the
                // interior heat flux makes the expression CONTINUOUS as
                // avg_flux -> 0 and lands a never-lit face exactly on the
                // geothermal floor the solver itself enforces every step
                // (ThermalBatchSoA::advanceTimestep, thermal_model.cpp:2426,
                // T_geo = (Qb/(eps*sigma))^0.25) -- so the seed and the floor
                // now agree instead of fighting.
                //
                // What this replaces: a `T_surf_est = 200.0` zero-flux fallback
                // plus a 50 K floor, i.e. a 150 K step at avg_flux -> 0+. A face
                // with 1e-30 W/m^2 seeded at 50 K; a face with exactly 0.0
                // seeded at 200 K. On Europa that stamped every polar-night face
                // that cannot see Jupiter at 200 K, and ice drains it so slowly
                // that a whole orbit of equilibration still left an 85-face
                // 119 K patch embedded in a 47-105 K polar night. The 50 K floor
                // was the same defect one step down: it pinned the rest of the
                // polar night at 50 K against a physical 27.7 K (PTM-48).
                //
                // Qb is taken per face (batch.Qb[f], already overridden above
                // from face_Qb) so the seed matches the per-face floor the
                // solver will apply; with no face_Qb map this is identically
                // planet.Qb. The GPU per-face branch uses the same radiative
                // balance but with the GLOBAL planet.Qb and an isothermal
                // column fill, so this is close kin, not the same seed.
                double qb_for_seed = batch.Qb[f];
                if (!(qb_for_seed >= 0.0)) {     // also catches NaN
                    qb_for_seed = 0.0;
                    ++bad_qb_faces;
                }
                double flux_for_seed = avg_flux;
                if (!(flux_for_seed >= 0.0)) {   // also catches NaN
                    flux_for_seed = 0.0;
                    ++bad_flux_faces;
                }

                // ---- SHARED SEED (thermal/eq_seed.h) -----------------------
                // The same function the CPU scalar path (Profile::
                // seedFromMeanFlux) and the GPU host seed call. What changes
                // versus the block below it: the deep column is now the
                // conductive steady profile T(z) = T_0 + Qb*integral dz/k
                // instead of an exponential blend to a 50 K-capped T_deep. The
                // blend put ~63 % of the whole geothermal rise inside ONE skin
                // depth (3 cm on the Moon) where the true rise is Qb*delta/k_s
                // = 0.68 K; the rest of the column was then flat, which is the
                // part that takes hundreds of cycles to correct.
                if (seed_mode == thermal::EqSeedMode::Physical) {
                    alignas(64) double kc_col[thermal::MAX_THERMAL_LAYERS];
                    alignas(64) double rho_col[thermal::MAX_THERMAL_LAYERS];
                    alignas(64) double T_col[thermal::MAX_THERMAL_LAYERS];
                    for (int layer = 0; layer < n_layers; ++layer) {
                        kc_col[layer]  = batch.kc [layer * BS + f];
                        rho_col[layer] = batch.rho[layer * BS + f];
                    }
                    thermal::EqSeedInput sin_;
                    sin_.mean_absorbed_flux = flux_for_seed;
                    sin_.Qb                 = qb_for_seed;
                    sin_.emissivity         = batch.emissivity[f];
                    sin_.sigma              = batch.sigma_sb;
                    sin_.z                  = batch.z;
                    sin_.kc                 = kc_col;
                    sin_.rho                = rho_col;
                    sin_.n                  = n_layers;
                    sin_.R350               = batch.R350_val;
                    sin_.cond               = batch.cond_model;
                    sin_.mean_inst_radeq_T  = sum_inst_T;   // report-only
                    const thermal::EqSeedResult sres =
                        thermal::eqSeedColumn(sin_, T_col);
                    if (sres.clamped_low) ++seed_floor_faces;
                    if (sres.above_500K)  ++seed_hot_faces;
                    thermal::EqSeedRecorder::instance().record(global_f, sin_, sres);

                    batch.T_eq_init[f] = sres.T_surface;
                    for (int layer = 0; layer < n_layers; ++layer)
                        batch.T[layer * BS + f] = T_col[layer];

                    if (dump_eq_seed) {                   // PTM-51 diagnostic
                        dbg_avg_flux[global_f]   = avg_flux;
                        dbg_T_surf_est[global_f] = sres.T_surface;
                        dbg_dT_geo[global_f]     = sres.T_deep - sres.T_surface;
                    }
                    continue;
                }

                const double ratio = (flux_for_seed + qb_for_seed) / sb_denom;
                // Clamp the seed to the physical range [T_CMB, 400 K]. The
                // LOWER bound is load-bearing for a body with Qb <= 0, which is
                // reachable: `gb_Qb` accepts 0.0 (:2357) and
                // examples/campaigns/rockox_generic-test.yaml sets Qb: 0.0.
                // With Qb = 0 and a face whose mean absorbed flux is exactly 0
                // this expression returns exactly 0 K, and nothing downstream
                // repairs it -- the solver's per-face geothermal floor bails
                // out first (thermal_model.cpp:2425 `if (qb <= 0.0) continue;`).
                // A 0 K column is outside the range of the cp(T) polynomials
                // actually shipped: Europa is cp = 7.49*T with cp0 = 0
                // (planetary_properties.h:120,124) -> cp(0) = 0 -> kappa =
                // k/(rho*cp) = inf; the Moon's cp0 = -3.6125
                // (planetary_properties.h:161) -> cp(0) < 0 -> negative
                // diffusivity. Both are silent. Every other temperature site in
                // the solver already clamps at ABSOLUTE_ZERO_K
                // (thermal_model.cpp:849, 2641, 2678, 2704); the seed did not.
                // No validated arm reaches the bound (minimum seed in the
                // PTM-46 validation set is 24.04 K = Moon T_geo), so this is a
                // guard, not a behavior change -- verified byte-identical.
                double T_surf_est = std::min(400.0,
                        std::max(thermal::ABSOLUTE_ZERO_K,
                                 std::sqrt(std::sqrt(ratio))));

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

                if (dump_eq_seed) {                       // PTM-51 diagnostic
                    dbg_avg_flux[global_f]   = avg_flux;
                    dbg_T_surf_est[global_f] = T_surf_est;
                    dbg_dT_geo[global_f]     = delta_T_geo;   // == T_deep - T_surf_est
                }

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

            // Spin-up estimate (FIX 0, report-only). Scored after the FIRST
            // substep of the run -- late enough that advanceTimestep has filled
            // batch.k / batch.cp (initialize() memsets them to zero and scoring
            // a zero-conductivity column would report nonsense), early enough
            // that it lands before any real time is spent. Reading only; writes
            // no solver state.
            //
            // It is a LAMBDA because the analytic pre-conditioner it used to
            // live inside is now off by default: with no pre-conditioner the
            // first substep is the first step of the MAIN loop, and the
            // estimator has to be scored there instead or the run silently
            // stops reporting it.
            bool census_scored = false;
            auto score_spinup_census = [&]() {
                if (census_scored || !g_spinup_census) return;
                census_scored = true;
                const int BS2 = thermal::THERMAL_BATCH_SIZE;
                std::vector<double> cz(n_layers), cT(n_layers),
                                    ck(n_layers), cr(n_layers), cc(n_layers);
                for (int i = 0; i < n_layers; ++i) cz[i] = batch.z[i];
                for (int f = 0; f < faces_in_batch; ++f) {
                    for (int i = 0; i < n_layers; ++i) {
                        cT[i] = batch.T  [i * BS2 + f];
                        ck[i] = batch.k  [i * BS2 + f];
                        cr[i] = batch.rho[i * BS2 + f];
                        cc[i] = batch.cp [i * BS2 + f];
                    }
                    thermal::EqSpinupInput ein;
                    ein.z = cz.data(); ein.T = cT.data(); ein.k = ck.data();
                    ein.rho = cr.data(); ein.cp = cc.data();
                    ein.n = n_layers;
                    ein.Qb  = batch.Qb[f];
                    ein.eps = batch.emissivity[f];
                    ein.sigma = thermal::SIGMA_SB;
                    ein.P = planet.day;
                    ein.target_K = eq_target_K;
                    // This path is float64 and carries no arithmetic floor;
                    // the float32 arrest is a GPU-path property.
                    ein.precision_floor_K = 0.0;
                    // Dark iff the face never sees sun in the REAL
                    // illumination. Using the analytic equilibration forcing
                    // instead (cos_lat*cos_hour) would call every face lit --
                    // it has no topography and no shadowing, so cos_lat > 0
                    // everywhere off the poles -- and the split that exists to
                    // expose the frozen dark population would report no dark
                    // population at all.
                    double fmax = 0.0;
                    for (int t = 0; t < total_time_steps; ++t) {
                        const double v = flux_column_major[
                            static_cast<size_t>(t) * n_faces + start_face + f];
                        if (v > fmax) fmax = v;
                    }
                    const bool dark = !(fmax > 0.0);
                    g_spinup_census->add(thermal::eqEstimateSpinup(ein), dark);
                }
            };

            // Equilibration phase:
            // Mirror Model::fastEquilibrate — run NYEARSEQ * round(year/day) lunar days
            // of diurnal forcing before the main output loop. Cycle the precomputed
            // flux trajectory as a quasi-periodic signal and substep it down to
            // dt <= planet.day / 192 so CN accuracy stays within ~1 K worst case
            // (see src/cpp/tools/test_equilibration_dt.cpp for the accuracy study).
            // Computed once above the parallel region (it depends only on
            // config/planet/times, never on the batch); see equil_duration_s_shared.
            const double& equil_duration_s = equil_duration_s_shared;

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

                // This dt is fixed by steps-per-day and knows nothing about the
                // depth grid, so a finer --thermal-m raises the top-cell
                // Fourier number as m^2 with no automatic response. Warn rather
                // than clamp (the removed 1 mm thickness floor was the clamp).
                thermal::warnIfTopCellFourierHigh(ref_profile, dt_sub,
                                                  "CPU equilibration");

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
                    if (dump_eq_seed) {                   // PTM-51 diagnostic
                        for (int f = 0; f < faces_in_batch; ++f) {
                            const size_t gf = start_face + f;
                            dbg_eqflux_sum[gf] += flux_vec[f];
                            if (flux_vec[f] > dbg_eqflux_max[gf])
                                dbg_eqflux_max[gf] = flux_vec[f];
                            ++dbg_eqflux_n[gf];
                        }
                    }
                    batch.setFluxContiguous(flux_vec, faces_in_batch);
                    batch.advanceTimestep(dt_sub);
                    if (step == 0) score_spinup_census();
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
            // Same caveat as the equilibration loop above: dt here is set by a
            // fixed steps-per-day, not by the grid.
            thermal::warnIfTopCellFourierHigh(ref_profile, main_target_dt,
                                              "CPU main loop");
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
                    // Fires only when the analytic pre-conditioner was off (the
                    // default), in which case this is the run's first substep.
                    if (!census_scored) score_spinup_census();
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

    // Equilibration spin-up estimate (FIX 0). Emitted once, after every batch
    // has contributed. The numbers are a prediction made one substep into a
    // ~30,000-substep equilibration, not a measurement of what it achieved --
    // the retrospective certificate is a separate, not-yet-implemented piece.
    if (g_spinup_census) {
        // Report the spin-up this run ACTUALLY integrates, which is the analytic
        // clear-sky seed PLUS the real-ephemeris equilibration prefix. Passing
        // the seed alone (as this did) printed "Config requests: 24 cycles" for
        // a config that says equilibration_orbits: 3 — a number that is neither
        // the configured value nor the integrated one.
        const double seed_cycles = (planet.day > 0.0)
                                 ? (equil_duration_s_shared / planet.day) : 0.0;
        double real_eq_cycles = 0.0;
        if (planet.day > 0.0 && equilibration_steps > 0 &&
            equilibration_steps < total_time_steps &&
            static_cast<size_t>(equilibration_steps) < times.size()) {
            real_eq_cycles = (times[equilibration_steps] - times[0]) / planet.day;
        }
        g_spinup_census->report(std::cout, seed_cycles + real_eq_cycles);
    }

    // PTM-51 diagnostic: write the per-face seed terms and the analytic
    // equilibration forcing that actually drove the column.
    if (dump_eq_seed) {
        std::ofstream fh(eq_seed_dump_path);
        if (!fh) {
            std::cerr << "[PTM-51 dump] WARNING: cannot open '" << eq_seed_dump_path
                      << "' for writing; NO dump produced." << std::endl;
        } else {
            fh << std::setprecision(10);
            fh << "# PTM-51 equilibration-seed diagnostic\n"
               << "# body=" << planet.name
               << " emissivity=" << planet.emissivity
               << " Qb=" << planet.Qb
               << " kd=" << planet.kd
               << " albedo=" << planet.albedo
               << " S=" << planet.S
               << " grid_depth_m=" << ref_profile.z().back()
               << " n_layers=" << n_layers
               << " T_geo=" << std::pow(planet.Qb / (planet.emissivity * SIGMA_SB), 0.25)
               << "\n";
            fh << "face,lat_deg,lon_deg,avg_flux_Wm2,T_surf_est_K,"
                  "T_deep_minus_T_surf_K,eq_forcing_mean_Wm2,eq_forcing_max_Wm2,"
                  "eq_substeps,T_surf_final_K\n";
            for (size_t gf = 0; gf < n_faces; ++gf) {
                const double eqmean = (dbg_eqflux_n[gf] > 0)
                    ? dbg_eqflux_sum[gf] / static_cast<double>(dbg_eqflux_n[gf]) : 0.0;
                double t_final = 0.0;
                if (surface_temps_out && gf < surface_temps_out->size() &&
                    !(*surface_temps_out)[gf].empty())
                    t_final = (*surface_temps_out)[gf].back();
                fh << gf << ','
                   << (gf < mesh->latitudes.size()  ? mesh->latitudes[gf]  : 0.0) << ','
                   << (gf < mesh->longitudes.size() ? mesh->longitudes[gf] : 0.0) << ','
                   << dbg_avg_flux[gf] << ','
                   << dbg_T_surf_est[gf] << ','
                   << dbg_dT_geo[gf] << ','
                   << eqmean << ','
                   << dbg_eqflux_max[gf] << ','
                   << dbg_eqflux_n[gf] << ','
                   << t_final << '\n';
            }
            std::cout << "[PTM-51 dump] wrote " << n_faces << " rows to "
                      << eq_seed_dump_path << std::endl;
        }
    }

    // Never substitute silently (see bad_flux_faces above).
    if (bad_flux_faces.load() > 0) {
        std::cerr << "WARNING: " << bad_flux_faces.load() << " of " << n_faces
                  << " faces had a negative or non-finite time-mean absorbed "
                     "flux; their equilibration seed used 0 W/m^2 instead, so "
                     "they cold-started at the geothermal steady state. "
                     "Absorbed flux should never be negative -- treat this as "
                     "an upstream flux/albedo-map defect, not a seed issue."
                  << std::endl;
    }

    // Same rule for the interior heat flux (see bad_qb_faces above).
    if (bad_qb_faces.load() > 0) {
        std::cerr << "WARNING: " << bad_qb_faces.load() << " of " << n_faces
                  << " faces had a negative or non-finite interior heat flux "
                     "Qb; their equilibration seed used 0 W/m^2 instead, so "
                     "they cold-started at the radiative steady state of the "
                     "absorbed flux alone (floored at "
                  << thermal::ABSOLUTE_ZERO_K
                  << " K). A negative Qb is not modeled -- treat this as a "
                     "defective geothermal_heat_flow value or Qb map, not a "
                     "seed issue."
                  << std::endl;
    }

    // Shared-seed guards. Both are physics statements, not tuning knobs, and
    // both are reported rather than applied silently.
    if (seed_floor_faces.load() > 0) {
        std::cerr << "WARNING: " << seed_floor_faces.load() << " of " << n_faces
                  << " faces seeded at the " << thermal::ABSOLUTE_ZERO_K
                  << " K CMB floor -- they have Qb <= 0 AND zero mean absorbed "
                     "flux, so the steady-state balance has no positive root. "
                     "cp(T) is not defined below that."
                  << std::endl;
    }
    if (seed_hot_faces.load() > 0) {
        std::cerr << "WARNING: " << seed_hot_faces.load() << " of " << n_faces
                  << " faces have a cycle-mean radiative-equilibrium seed above "
                  << thermal::MAX_PHYSICAL_TEMP_K
                  << " K, which is the CPU solver's own hard temperature clamp. "
                     "The seed is physical (Mercury's subsolar cycle mean is "
                     "~468 K); the CLAMP will truncate the solution. Results on "
                     "those faces are not trustworthy."
                  << std::endl;
    }
    thermal::EqSeedRecorder::instance().write("cpu-simd-batch");

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

// ---- Equilibration cycling (thermal.equilibration_cycles) ------------------
//
// PLACEMENT: this block must stay OUTSIDE the `#ifdef PTM_GPU_OPENCL` that
// opens just below it. resolveEquilibrationPasses() is called from main() to
// REFUSE the knob on every non-GPU path, and that refusal has to compile in
// CPU-only builds — which are precisely the builds where the knob can never be
// honored. It first shipped inside the guard, which built fine with GPU=ON here
// and broke the CPU-only CI build outright ("use of undeclared identifier
// resolveEquilibrationPasses" at the main() call site). Guarding the call site
// instead would compile, but would silently delete the refusal from the only
// configuration that needs it.
//
// The unit here is PASSES over the equilibration forcing array, not "extra"
// cycles. The GPU persistent solver already implements the loop
// (thermal_opencl.cpp: `const int total_cycles = n_equil_cycles + 1;`): each
// pass replays the SAME uploaded flux array while buf_T_ persists across
// passes, so P passes over a B-cycle forcing block integrate P*B diurnal
// cycles at 1x illumination cost and 1x flux RAM. That is the only affordable
// route past the RAM wall — a 25-cycle eq array is ~6.1 GB on the 509k-face
// lunar mesh, so 100+ cycles in ONE array is hard-blocked.
//
// ---- SEMANTICS (changed 2026-08-11; BUGS.md PTM-146) ----------------------
// `thermal.equilibration_orbits` is the INTEGRATED number of diurnal cycles of
// real, ephemeris-driven spin-up. `thermal.equilibration_cycles` (passes) is a
// pure MEMORY/COST dial that says how many replays are used to reach it, so
// the assembled forcing block is
//
//     block_cycles = equilibration_orbits / equilibration_cycles
//     integrated   = equilibration_cycles * block_cycles = equilibration_orbits
//
// and the configured number is the integrated number on every path and at
// every pass count. Before this change `equilibration_orbits` was the BLOCK
// length and the integrated total was orbits x passes, with the pass count
// defaulting to 2 — so `equilibration_orbits: 25` silently integrated 50
// lunations on the GPU and 25 on the CPU (measured, see below). The default
// pass count is now 1, which is what makes "configured == integrated" true for
// a config that does not mention passes at all.
//
// MEASURED, `equilibration_orbits: 3.0`, Moon, res-2 icosphere, before/after
// the change (integrated diurnal cycles of real forcing before the output
// window opens):
//     CPU scalar   27.00 -> 27.00   (3.00 configured + 24.00 analytic seed)
//     CPU SIMD     27.00 -> 27.00   (3.00 configured + 24.00 analytic seed)
//     GPU 2-phase   6.00 ->  3.00
//     GPU 1-phase   7.07 ->  3.04
//
// ⚠️ WHAT THIS DOES NOT FIX. Both CPU paths ALSO run a fixed analytic
// clear-sky pre-conditioner before the real forcing — Model::fastEquilibrate
// (scalar) and its mirror in the SIMD batch path — of
// `NYEARSEQ * round(planet.year / planet.day)` diurnal cycles: 24 on the Moon,
// 2 on Europa, 4 on Mercury, with NYEARSEQ hard-wired to 2 in main(). It is
// not user-configurable, it is not counted by equilibration_orbits, and the
// GPU path has no equivalent. So "configured == integrated" is exact on the
// GPU and is "configured == integrated real-forcing spin-up, plus a fixed
// analytic seed" on the CPU. The banner below prints the seed length rather
// than letting it stay invisible. Removing or exposing it is a separate change
// (it would move every CPU baseline) — BUGS.md row proposed.
//
// Granularity: we cycle the EQUILIBRATION forcing array as assembled
// (`prepare_flux(0, equilibration_steps)`). On the shipped lunar year-long
// config the block is 25 lunations = 738.3 d = 2.02 Earth-years, so the
// seasonal (declination) content is inside the cycled block and is preserved
// by construction; cycling a single lunation would discard it. Cycling a
// *separately assembled* annual array is not done here: it would need a second
// illumination pass on a second time grid and a second flux buffer, and would
// not reduce to the current path exactly.
static constexpr int kDefaultEquilibrationPasses = 1;
// The pre-2026-08-11 default, kept only so messages can name it: every run
// before that date integrated TWICE what its config said on the GPU.
static constexpr int kHistoricalEquilibrationPasses = 2;

// Resolve the requested number of equilibration passes from the raw config
// value (-1 = unset). Precedence: PTM_EQUILIBRATION_CYCLES env > config
// `equilibration_cycles` > default. Returns 0 on a malformed request —
// callers MUST treat that as fatal rather than silently continuing with the
// default (a silent substitution here would misreport the equilibration
// length of every run).
static int resolveEquilibrationPassesValue(int cfg,
                                           bool* explicitly_set = nullptr,
                                           std::string* source = nullptr) {
    int passes = kDefaultEquilibrationPasses;
    bool explicit_set = false;
    std::string src = "default";

    if (cfg > 0) {
        passes = cfg;
        explicit_set = true;
        src = "config thermal.equilibration_cycles";
    } else if (cfg != -1) {
        std::cerr << "[Thermal] ERROR: equilibration_cycles must be >= 1 "
                     "(or -1 for the default); got " << cfg << "\n";
        if (explicitly_set) *explicitly_set = true;
        if (source) *source = "invalid";
        return 0;
    }

    if (const char* e = std::getenv("PTM_EQUILIBRATION_CYCLES")) {
        int v = 0;
        try {
            size_t consumed = 0;
            v = std::stoi(std::string(e), &consumed);
            if (consumed != std::string(e).size()) v = 0;
        } catch (const std::exception&) {
            v = 0;
        }
        if (v < 1) {
            std::cerr << "[Thermal] ERROR: PTM_EQUILIBRATION_CYCLES='" << e
                      << "' is not an integer >= 1\n";
            if (explicitly_set) *explicitly_set = true;
            if (source) *source = "invalid";
            return 0;
        }
        if (explicit_set && v != passes) {
            std::cout << "[Thermal] PTM_EQUILIBRATION_CYCLES=" << v
                      << " overrides config equilibration_cycles=" << passes
                      << "\n";
        }
        passes = v;
        explicit_set = true;
        src = "env PTM_EQUILIBRATION_CYCLES";
    }

    if (explicitly_set) *explicitly_set = explicit_set;
    if (source) *source = src;
    return passes;
}

// Config-object overload. main() resolves the pass count long before
// thermal_config exists (it needs it to size the equilibration window), so the
// raw-value function above is the primitive and this is the thin wrapper.
static int resolveEquilibrationPasses(const thermal::Config& config,
                                      bool* explicitly_set = nullptr,
                                      std::string* source = nullptr) {
    return resolveEquilibrationPassesValue(
        config.getPropertyInt("equilibration_cycles", -1), explicitly_set, source);
}

// ---- Pass-stride RRE extrapolation (thermal.eq_extrapolation) --------------
//
// PLACEMENT: this stays OUTSIDE the `#ifdef PTM_GPU_OPENCL` below, for exactly
// the reason spelled out above resolveEquilibrationPasses. The 2026-08-07
// CPU-only CI break was a helper defined inside the guard and called outside
// it; the refusal path here is needed most in the build that cannot honor the
// knob at all.
//
// Returns false only on a request we refuse to run. A request we cannot honor
// (too few passes) is NOT a refusal: it warns and falls back to plain cycling,
// which is what Fermi's ADDENDUM A3(b) specifies.
//
// `eq_passes` <= 0 means "the pass budget is not known at this call site" and
// skips ONLY the budget check, so main() can validate the knob itself long
// before the solver has resolved a pass count.
static bool resolveEqExtrapolation(const thermal::Config& config,
                                   int eq_passes,
                                   ptm::eqextrap::Options* out,
                                   bool* requested = nullptr) {
    ptm::eqextrap::Options o;
    const std::string method = config.getProperty("eq_extrap_method", "off");
    if (requested) *requested = (method != "off");
    if (method == "off") {
        if (out) *out = o;                       // enabled = false
        return true;
    }
    if (method != "rre") {
        std::cerr << "[Thermal] ERROR: thermal.eq_extrapolation.method must be "
                     "'off' or 'rre'; got '" << method << "'\n";
        return false;
    }
    o.order_p = config.getPropertyInt("eq_extrap_order_p", 4);
    o.tol_K   = config.getPropertyDouble("eq_extrap_tol_K", 0.5);
    o.hold_K  = config.getPropertyDouble("eq_extrap_hold_K", ptm::eqextrap::kHoldKDefault);

    if (o.order_p < ptm::eqextrap::kMinOrderP) {
        std::cerr << "[Thermal] ERROR: eq_extrapolation.order_p=" << o.order_p
                  << " is refused. order_p=2 is not merely weak, it is HARMFUL "
                     "(54.82 K worst-column error vs 31.88 K for plain 2-pass "
                     "cycling): the DC seed sits outside the linear regime and a "
                     "degree-1 annihilator cannot remove it. Minimum is "
                  << ptm::eqextrap::kMinOrderP
                  << ". Refusing rather than silently raising it.\n";
        return false;
    }
    if (o.order_p > ptm::eqextrap::kMaxValidatedOrderP) {
        std::cout << "[Thermal] WARNING: eq_extrapolation.order_p=" << o.order_p
                  << " is above the validated range (3-4). It will run, but no "
                     "acceptance threshold has been measured for it.\n";
    }
    if (!(o.hold_K > 0.0)) {
        std::cerr << "[Thermal] ERROR: eq_extrapolation.hold_K must be > 0; got "
                  << o.hold_K << "\n";
        return false;
    }
    if (!(o.tol_K > 0.0)) {
        std::cerr << "[Thermal] ERROR: eq_extrapolation.tol_K must be > 0; got "
                  << o.tol_K << "\n";
        return false;
    }

    // Budget. Extrapolating at the top of the last pass needs order_p+1
    // snapshots, i.e. order_p+1 passes. Warn and run plain; never silently
    // reduce the order (a reduction to p=2 would be worse than not
    // extrapolating at all).
    if (eq_passes > 0 && eq_passes < o.order_p + 1) {
        std::cout << "[Thermal] WARNING: eq_extrapolation is ON with order_p="
                  << o.order_p << ", which needs equilibration_cycles >= "
                  << (o.order_p + 1) << ", but only " << eq_passes
                  << " pass(es) are budgeted. Running PLAIN cycling for this run "
                     "(the order is NOT reduced: order_p=2 is harmful).\n";
        o.enabled = false;
        if (out) *out = o;
        return true;
    }

    o.enabled = true;
    if (out) *out = o;
    return true;
}

// ---- Equilibration forcing dump (PTM_DUMP_EQ_LUNATION_MEAN) ----------------
//
// DIAGNOSTIC, DEFAULT OFF. When PTM_DUMP_EQ_LUNATION_MEAN names a directory the
// run assembles the equilibration absorbed-flux block exactly as the thermal
// solver does (the reduction consumes the solver's OWN prepare_flux output, it
// does not re-derive illumination), reduces it to a per-face, per-diurnal-cycle
// mean, writes it, and EXITS before any thermal solve. With the variable unset
// every function below is unreachable and no state is allocated, so the default
// path is byte-identical.
//
// WHY: the engine's equilibration forcing is `equilibration_orbits` diurnal
// cycles of REAL ephemeris illumination (25 cycles = 738.26 d on the shipped
// moon3d config) replayed as one block, so the solver's Poincare period is one
// PASS over that block, not one diurnal cycle. Any equilibration accelerator
// that samples the state at a ONE-CYCLE stride assumes the block is one-cycle
// periodic. The seasonal modulation index
//     a_f = (max_j Qbar_j - min_j Qbar_j) / (2 * mean_j Qbar_j),  j = 1..n_lun
// measures how false that assumption is, per face. See the 2026-08-08 audit of
// the restarted-RRE equilibration accelerator, section "DECISIVE NEXT
// MEASUREMENT".
//
// PLACEMENT is deliberately OUTSIDE `#ifdef PTM_GPU_OPENCL`. The reduction
// hangs off the GPU persistent solver, so a CPU-only build can never produce
// the dump — which is exactly why its REFUSAL (in main()) must compile there.
// A GPU-gated definition with a CPU-visible call site is the CPU-only CI break
// of 2026-08-07, verbatim; do not move this inside the guard.
namespace eqlundump {

// Empty string = not requested. Any other value is the output DIRECTORY.
inline std::string requestedDir() {
    const char* e = std::getenv("PTM_DUMP_EQ_LUNATION_MEAN");
    if (!e) return std::string();
    std::string v(e);
    if (v.empty() || v == "0" || v == "false" || v == "off") return std::string();
    return v;
}

struct Buffers {
    bool        enabled   = false;
    std::string dir;
    size_t      n_faces   = 0;
    size_t      n_lun     = 0;   // diurnal cycles inside the eq forcing block
    size_t      spl       = 0;   // samples per diurnal cycle
    int         eq_steps  = 0;
    double      eq_dt_s   = 0.0;
    double      synodic_d = 0.0;

    std::vector<double>   sum_abs;  // n_faces * n_lun  (sum of absorbed W/m^2)
    std::vector<double>   sum_dir;  // n_faces * n_lun  (sum of direct  W/m^2)
    std::vector<uint32_t> lit_cnt;  // n_faces * n_lun  (samples with direct > 0)
    std::vector<float>    max_dir;  // n_faces          (max direct over block)
};

// Configure, or REFUSE. Every condition that would silently make the dump mean
// something other than "per-diurnal-cycle mean of the equilibration forcing
// block" is fatal: a wrong-but-plausible dump is worse than no dump.
inline void configure(Buffers& b, const std::string& dir, size_t n_faces,
                      int equilibration_steps, bool two_phase,
                      double eq_dt_s, double synodic_day_days) {
    if (dir.empty()) return;

    if (!two_phase || equilibration_steps <= 0) {
        std::cerr << "[eq-lun-dump] FATAL: PTM_DUMP_EQ_LUNATION_MEAN is set but this "
                     "run has no separate equilibration phase (two_phase="
                  << (two_phase ? 1 : 0) << ", equilibration_steps="
                  << equilibration_steps << "). There is no equilibration forcing "
                     "block to reduce. Refusing rather than dumping the wrong array.\n";
        std::exit(1);
    }
    const double synodic_s = synodic_day_days * 86400.0;
    if (!(synodic_s > 0.0) || !(eq_dt_s > 0.0)) {
        std::cerr << "[eq-lun-dump] FATAL: cannot resolve the diurnal period "
                     "(synodic_day=" << synodic_day_days << " d, eq_dt="
                  << eq_dt_s << " s).\n";
        std::exit(1);
    }

    // n_lun from the physics (block length / synodic day), then cross-checked
    // against exact divisibility of the sample count. Env override exists for
    // configs whose cadence is not an integer number of samples per cycle.
    const double eq_seconds = static_cast<double>(equilibration_steps) * eq_dt_s;
    long long n_lun = static_cast<long long>(eq_seconds / synodic_s + 0.5);
    if (const char* e = std::getenv("PTM_DUMP_EQ_LUNATIONS")) {
        long long v = std::atoll(e);
        if (v < 2) {
            std::cerr << "[eq-lun-dump] FATAL: PTM_DUMP_EQ_LUNATIONS='" << e
                      << "' is not an integer >= 2.\n";
            std::exit(1);
        }
        if (v != n_lun) {
            std::cout << "[eq-lun-dump] PTM_DUMP_EQ_LUNATIONS=" << v
                      << " overrides the derived " << n_lun << "\n";
        }
        n_lun = v;
    }
    if (n_lun < 2) {
        std::cerr << "[eq-lun-dump] FATAL: the equilibration block spans only "
                  << (eq_seconds / synodic_s) << " diurnal cycles; a "
                     "cycle-to-cycle modulation index is undefined.\n";
        std::exit(1);
    }
    if (equilibration_steps % static_cast<int>(n_lun) != 0) {
        std::cerr << "[eq-lun-dump] FATAL: equilibration_steps="
                  << equilibration_steps << " is not divisible by n_lunations="
                  << n_lun << "; the per-cycle bins would be ragged and the "
                     "modulation index would pick up a binning artifact. Set "
                     "PTM_DUMP_EQ_LUNATIONS to a divisor, or use a cadence that "
                     "divides evenly.\n";
        std::exit(1);
    }

    b.enabled   = true;
    b.dir       = dir;
    b.n_faces   = n_faces;
    b.n_lun     = static_cast<size_t>(n_lun);
    b.spl       = static_cast<size_t>(equilibration_steps) / b.n_lun;
    b.eq_steps  = equilibration_steps;
    b.eq_dt_s   = eq_dt_s;
    b.synodic_d = synodic_day_days;
    b.sum_abs.assign(b.n_faces * b.n_lun, 0.0);
    b.sum_dir.assign(b.n_faces * b.n_lun, 0.0);
    b.lit_cnt.assign(b.n_faces * b.n_lun, 0u);
    b.max_dir.assign(b.n_faces, 0.0f);

    std::cout << "[eq-lun-dump] ENABLED -> " << dir << "\n"
              << "[eq-lun-dump]   " << n_faces << " faces x " << b.n_lun
              << " diurnal cycles x " << b.spl << " samples/cycle = "
              << equilibration_steps << " eq steps (dt=" << eq_dt_s
              << " s, synodic=" << synodic_day_days << " d)\n"
              << "[eq-lun-dump]   the run will EXIT after the dump; NO thermal "
                 "solve is performed.\n";
}

// Reduce one batch of the solver's OWN eq forcing array. flux_eq is laid out
// [t * batch_n + f] exactly as prepare_flux returns it; direct_solar_values is
// the whole-mesh illumination table laid out [global_f * total_time_steps + t].
// Batches cover disjoint face ranges, so concurrent pool workers need no lock.
inline void accumulateBatch(Buffers& b, const std::vector<float>& flux_eq,
                            const std::vector<double>& direct_solar_values,
                            size_t batch_start, size_t batch_n,
                            int equilibration_steps, int total_time_steps) {
    if (!b.enabled) return;
    const size_t n_lun = b.n_lun, spl = b.spl;

    for (int t = 0; t < equilibration_steps; ++t) {
        const size_t j = static_cast<size_t>(t) / spl;
        if (j >= n_lun) break;
        const float* row = flux_eq.data() + static_cast<size_t>(t) * batch_n;
        for (size_t f = 0; f < batch_n; ++f) {
            b.sum_abs[(batch_start + f) * n_lun + j] += static_cast<double>(row[f]);
        }
    }
    for (size_t f = 0; f < batch_n; ++f) {
        const size_t gf   = batch_start + f;
        const size_t base = gf * static_cast<size_t>(total_time_steps);
        double  mx = 0.0;
        for (int t = 0; t < equilibration_steps; ++t) {
            const size_t idx = base + static_cast<size_t>(t);
            const double d = (idx < direct_solar_values.size())
                           ? direct_solar_values[idx] : 0.0;
            const size_t j = static_cast<size_t>(t) / spl;
            if (j >= n_lun) break;
            b.sum_dir[gf * n_lun + j] += d;
            if (d > 0.0) { b.lit_cnt[gf * n_lun + j] += 1u; mx = std::max(mx, d); }
        }
        b.max_dir[gf] = static_cast<float>(mx);
    }
}

// Write the products. Returns false on ANY write failure -- the caller must
// treat that as fatal; a partially written dump that later gets analyzed is a
// silent-wrong-answer generator.
inline bool write(const Buffers& b,
                  const std::vector<double>& latitudes,
                  const std::vector<double>& longitudes,
                  const std::string& planet_name) {
    if (!b.enabled) return true;
    std::error_code ec;
    std::filesystem::create_directories(b.dir, ec);

    auto path = [&](const char* leaf) { return b.dir + "/" + leaf; };
    auto put  = [&](const std::string& p, const void* data, size_t bytes) {
        std::ofstream os(p, std::ios::binary);
        if (!os) { std::cerr << "[eq-lun-dump] ERROR: cannot open " << p << "\n"; return false; }
        os.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!os) { std::cerr << "[eq-lun-dump] ERROR: short write on " << p << "\n"; return false; }
        return true;
    };

    const size_t nf = b.n_faces, nl = b.n_lun;
    const double inv = 1.0 / static_cast<double>(b.spl);

    std::vector<float> mean_abs(nf * nl), mean_dir(nf * nl);
    for (size_t i = 0; i < nf * nl; ++i) {
        mean_abs[i] = static_cast<float>(b.sum_abs[i] * inv);
        mean_dir[i] = static_cast<float>(b.sum_dir[i] * inv);
    }
    std::vector<float> geom(nf * 3);
    for (size_t f = 0; f < nf; ++f) {
        geom[f * 3 + 0] = static_cast<float>(f < latitudes.size()  ? latitudes[f]  : 0.0);
        geom[f * 3 + 1] = static_cast<float>(f < longitudes.size() ? longitudes[f] : 0.0);
        geom[f * 3 + 2] = b.max_dir[f];
    }

    bool ok = true;
    ok = put(path("eq_lunation_mean_absorbed.f32"), mean_abs.data(), mean_abs.size() * sizeof(float)) && ok;
    ok = put(path("eq_lunation_mean_direct.f32"),   mean_dir.data(), mean_dir.size() * sizeof(float)) && ok;
    ok = put(path("eq_lunation_lit_count.u32"),     b.lit_cnt.data(), b.lit_cnt.size() * sizeof(uint32_t)) && ok;
    ok = put(path("eq_face_geom.f32"),              geom.data(),      geom.size() * sizeof(float)) && ok;

    std::ofstream js(path("eq_lunation_dump.json"));
    if (!js) { std::cerr << "[eq-lun-dump] ERROR: cannot open the JSON sidecar\n"; ok = false; }
    else {
        js << "{\n"
           << "  \"n_faces\": " << nf << ",\n"
           << "  \"n_lunations\": " << nl << ",\n"
           << "  \"samples_per_lunation\": " << b.spl << ",\n"
           << "  \"equilibration_steps\": " << b.eq_steps << ",\n"
           << "  \"eq_dt_s\": " << std::setprecision(12) << b.eq_dt_s << ",\n"
           << "  \"synodic_day_days\": " << b.synodic_d << ",\n"
           << "  \"planet\": \"" << planet_name << "\",\n"
           << "  \"dtype_mean\": \"float32\",\n"
           << "  \"dtype_lit_count\": \"uint32\",\n"
           << "  \"layout\": \"face-major: value[f, j] at offset f*n_lunations + j\",\n"
           << "  \"eq_face_geom\": \"n_faces x 3 float32 (lat_deg, lon_deg, max_direct_W_m2)\",\n"
           << "  \"units\": \"W/m^2\",\n"
           << "  \"source\": \"prepare_flux(0, equilibration_steps) of the GPU persistent solver\",\n"
           << "  \"env\": \"PTM_DUMP_EQ_LUNATION_MEAN\"\n"
           << "}\n";
        js.close();
        if (!js) { std::cerr << "[eq-lun-dump] ERROR: short write on the JSON sidecar\n"; ok = false; }
    }

    std::cout << "[eq-lun-dump] wrote " << nf << " x " << nl
              << " float32 (" << (nf * nl * sizeof(float)) << " bytes) to "
              << path("eq_lunation_mean_absorbed.f32") << (ok ? "" : "  [WITH ERRORS]")
              << std::endl;
    return ok;
}

}  // namespace eqlundump

// ========== GPU PERSISTENT THERMAL SOLVER ==========
#ifdef PTM_GPU_OPENCL

// ============================================================================
// Phase R (auto-calibrate): representative, in-run GPU dispatch calibration.
//
// The dispatch self-chunk ceiling (max_timesteps_per_dispatch) is a WEDGE knob,
// not a physics knob: it only bounds how many timesteps one clEnqueueNDRangeKernel
// runs before a host clFinish, and enqueuePersistentSelfChunked keeps that split
// NUMERICALLY INERT (same CN updates, same order). A conservative default is safe
// but slow; a hand-run `--calibrate-gpu` on a generic grid is fast but can be
// non-representative (F3). The auto path below probes THIS run's actual grid and
// thermophysical props right before the main solve, injects the measured ceiling
// in-process (setDispatchCeilingOverride, which the consumer honors ahead of its
// static cache), and caches it grid-keyed so alternating bodies/resolutions do
// not re-probe every run. Default ON; opt out with PTM_NO_AUTO_CALIBRATE or
// config auto_calibrate_gpu:false; PTM_MAX_TIMESTEPS_PER_DISPATCH always wins.
// ============================================================================

// Restore std::cout's format flags/precision on scope exit. The GPU thermal path
// leaves std::cout in std::fixed + setprecision(1) (the two-phase dt banner), which
// would otherwise round the calibration's small grid numbers (dz_min~1e-3 m,
// kappa~1e-9) to "0.0" in the log. Setting defaultfloat + precision(6) for the
// duration of a calibration log keeps the printed values honest.
struct CoutFmtGuard {
    std::ios::fmtflags f;
    std::streamsize    p;
    CoutFmtGuard() : f(std::cout.flags()), p(std::cout.precision()) {
        std::cout << std::defaultfloat << std::setprecision(6);
    }
    ~CoutFmtGuard() { std::cout.flags(f); std::cout.precision(p); }
};

// Probe description passed to calibrateGpuCore. The manual --calibrate-gpu path
// synthesizes a generic grid (is_run_grid=false); the auto path fills it from the
// LOADED run's actual grid + near-surface props (is_run_grid=true), which is what
// makes the ceiling representative and stops the F3 clamp from firing.
struct CalibrationProbe {
    std::vector<float> z_grid;        // actual depth grid z(nl) [m], z_grid[0]=0
    int    n_layers      = 0;
    int    batch         = 0;         // faces per single dispatch (per-device load)
    double dt_output_s   = 0.0;       // worst-case dt the run will pass to the kernel
    double kc            = 0.0;       // near-surface conductivity [W/m/K]
    double rho           = 0.0;       // near-surface density [kg/m^3]
    double cp0           = 0.0;       // BASE heat capacity [J/kg/K] (may be 0 when
                                      // cp is fully given by cp_coeffs, e.g. Moon)
    std::vector<float> cp_coeffs;     // cp(T) polynomial coeffs (production kernel)
    double R350          = 2.7f;      // radiative-conductivity coeff (Hayne 2017)
    int    cond_model    = 0;         // conductivity law: 0 = Hayne 2017, 1 = Martinez 2021.
                                      // The probe must integrate the SAME law as the run,
                                      // or its measured substep count (and hence the safe
                                      // dispatch ceiling) is unrepresentative.
    double cp_eff        = 0.0;       // effective near-surface cp for the kappa
                                      // signature (caller evaluates cp(T_rep));
                                      // must be > 0 (never divide kappa by 0)
    double emissivity    = 0.95;
    double T_deep        = 100.0;     // deep boundary temp [K] (probe only)
    double T_initial     = 120.0;     // uniform initial temp [K] (probe only)
    double ceiling_s     = 25.0;      // safe per-dispatch wall budget
    double safety_factor = 4.0;       // F3 clamp (generic path only)
    int    probe_steps_1 = 30;
    int    probe_steps_2 = 120;
    double probe_budget_s = 15.0;     // wedge guard: abort if 30-step wall exceeds this
    bool   is_run_grid   = false;     // true relaxes the F3 4x clamp -> INFO
};

// Measure this host's safe dispatch ceiling for the given probe, filling `out`.
// Runs the SAME validated persistent-kernel path a production solve uses, so the
// timing reflects real dispatch cost. Return codes: 0 ok; 2 bad params; 3 OpenCL
// init fail; 4 non-positive slope (probe too noisy); 6 budget abort (grid too
// slow to probe safely). NUMERICALLY INERT: nothing here changes a production
// computed value — it only measures wall time and sets a dispatch-batching ceiling.
static int calibrateGpuCore(const CalibrationProbe& pr, gpu::MachineProfile& out) {
    using namespace gpu::opencl;
    CoutFmtGuard _cout_fmt;   // honest small-number logging; restored on return
    if (pr.batch < 1 || pr.n_layers < 2 || pr.ceiling_s <= 0.0 ||
        pr.dt_output_s <= 0.0 || pr.probe_steps_1 < 1 ||
        pr.probe_steps_2 <= pr.probe_steps_1 || pr.safety_factor <= 0.0 ||
        static_cast<int>(pr.z_grid.size()) != pr.n_layers ||
        pr.kc <= 0.0 || pr.rho <= 0.0 || pr.cp_eff <= 0.0 ||
        (pr.cp0 <= 0.0 && pr.cp_coeffs.empty())) {
        std::cerr << "[calibrate-gpu] invalid probe parameters (need batch>=1, "
                     "n_layers>=2 == z_grid size, ceiling>0, dt>0, "
                     "0<steps1<steps2, safety>0, kc/rho/cp_eff>0, and cp0>0 OR "
                     "cp_coeffs non-empty)\n";
        return 2;
    }

    const size_t nf = static_cast<size_t>(pr.batch);
    const size_t nl = static_cast<size_t>(pr.n_layers);

    ThermalOpenCLConfig cfg;
    ThermalOpenCLSolver solver(nf, nl, cfg);
    if (!solver.initialize()) {
        std::cerr << "[calibrate-gpu] OpenCL init FAILED — cannot profile this "
                     "host (no usable GPU). No profile written.\n";
        return 3;
    }

    std::vector<float> rho(nl * nf, static_cast<float>(pr.rho));
    std::vector<float> kc(nl * nf, static_cast<float>(pr.kc));
    std::vector<float> emissivity(nf, static_cast<float>(pr.emissivity));
    std::vector<float> T_deep(nf, static_cast<float>(pr.T_deep));
    std::vector<float> Qb(nf, 0.0f);
    std::vector<float> T_initial(nl * nf, static_cast<float>(pr.T_initial));
    std::vector<float> out_temps;

    solver.uploadMeshData(pr.z_grid, rho, kc, emissivity, T_deep, Qb);
    // Mirror the production kernel's heat-capacity model: base cp0 + cp(T)
    // polynomial + R350, so the probe's per-step substep count is representative.
    solver.setMaterialParams(static_cast<float>(pr.cp0), pr.cp_coeffs,
                             static_cast<float>(pr.R350),
                             static_cast<float>(ptm::kStefanBoltzmann), pr.cond_model);

    // Two-point regression: wall(P) ~= overhead + P * per_step. Two point counts
    // separate the fixed per-dispatch overhead (INTERCEPT) from the per-timestep
    // kernel cost (SLOPE). The slope is transfer-inclusive and thus conservative
    // (safe) for the ceiling.
    auto run_probe = [&](int steps) -> double {
        solver.uploadInitialTemperature(T_initial);   // same start state each time
        std::vector<float> flux(static_cast<size_t>(steps) * nf, 5.0f);  // faint cold flux
        auto w0 = std::chrono::high_resolution_clock::now();
        solver.solvePersistentAllTimesteps(flux, out_temps, steps,
                                           static_cast<float>(pr.dt_output_s));
        auto w1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(w1 - w0).count();
    };

    double wall1 = run_probe(pr.probe_steps_1);
    // Wedge budget guard: if the SMALL point already blew the budget, the LARGE
    // point (~4x steps) would be far worse — abort before running it.
    if (wall1 > pr.probe_budget_s) {
        std::cerr << "[calibrate-gpu] WARNING: probe budget exceeded ("
                  << pr.probe_steps_1 << "-step wall " << wall1 << "s > "
                  << pr.probe_budget_s << "s budget); aborting the probe to avoid a "
                     "slow/wedge-prone launch. Falling back to the conservative "
                     "ceiling.\n";
        return 6;
    }
    double wall2 = run_probe(pr.probe_steps_2);

    double per_step_s = (wall2 - wall1) /
                        static_cast<double>(pr.probe_steps_2 - pr.probe_steps_1);
    if (per_step_s <= 0.0) {
        std::cerr << "[calibrate-gpu] non-positive per-step slope (wall1=" << wall1
                  << "s wall2=" << wall2 << "s); probe too noisy/small — refusing "
                     "to write a bogus profile. Increase --probe-steps-2.\n";
        return 4;
    }
    double overhead_per_batch_s = std::max(0.0, wall1 - pr.probe_steps_1 * per_step_s);
    double per_step_ms = per_step_s * 1000.0;

    // Ceiling: keep ONE dispatch under safe_ceiling_s of PURE stepping time.
    long max_ts_raw   = static_cast<long>(pr.ceiling_s / per_step_s);
    int  conservative = gpu::MachineProfile::conservativeDefault().max_timesteps_per_dispatch;
    int  max_ts = 0;
    long cap = 0;
    if (pr.is_run_grid) {
        // Representative grid: max_ts_raw already keeps ONE dispatch under
        // ceiling_s for THIS grid — that IS the binding safety bound, so the 4x
        // clamp is redundant. Keep only a LOOSE absolute sanity cap to catch a
        // degenerate (near-zero-slope) measurement, and log at INFO (F3 must not
        // fire on a representative grid — that is the acceptance criterion).
        constexpr long kAbsSanityCap = 50000;
        cap = kAbsSanityCap;
        max_ts = static_cast<int>(std::min<long>(max_ts_raw, cap));
        if (max_ts < 1) max_ts = 1;
        if (max_ts_raw > cap) {
            std::cout << "[calibrate-gpu] INFO: representative-grid ceiling "
                      << max_ts_raw << " exceeds the absolute sanity cap " << cap
                      << " — clamping (measurement may be degenerate).\n";
        } else {
            std::cout << "[calibrate-gpu] INFO (F3): representative grid — ceiling "
                      << max_ts << " keeps one dispatch under " << pr.ceiling_s
                      << " s; 4x clamp not applied.\n";
        }
    } else {
        // Generic/synthetic probe grid: cannot be trusted to represent
        // production, so keep the conservative 4x clamp + loud WARNING.
        cap = static_cast<long>(pr.safety_factor * conservative);
        max_ts = static_cast<int>(std::min<long>(max_ts_raw, cap));
        if (max_ts < 1) max_ts = 1;
        if (max_ts_raw > cap) {
            std::cerr << "[calibrate-gpu] WARNING (F3): calibrated ceiling "
                      << max_ts_raw << " timesteps/dispatch is >> " << pr.safety_factor
                      << "x the conservative default (" << conservative << "); this "
                         "often means the probe grid is NOT representative of your "
                         "production run (too-coarse near-surface dz => too-few "
                         "substeps). CLAMPING to " << cap << ". Re-probe with --dz0 / "
                         "--n-layers / --dt-output-s matched to your production config, "
                         "or use the automatic in-run calibration.\n";
        }
    }

    out = gpu::MachineProfile();
    out.valid                        = true;
    out.safe_dispatch_ceiling_s      = pr.ceiling_s;
    out.max_timesteps_per_dispatch   = max_ts;
    out.recommended_batch_size       = pr.batch;
    out.kernel_ms_per_step_per_batch = per_step_ms;
    out.overhead_per_batch_s         = overhead_per_batch_s;
    out.n_layers_probe               = pr.n_layers;
    out.batch_size_probe             = pr.batch;
    double dz_min = 0.0;
    for (size_t i = 1; i < pr.z_grid.size(); ++i) {
        double d = static_cast<double>(pr.z_grid[i]) - static_cast<double>(pr.z_grid[i - 1]);
        if (d > 0.0 && (dz_min == 0.0 || d < dz_min)) dz_min = d;
    }
    out.dz_min_probe                 = dz_min;
    out.dt_output_probe_s            = pr.dt_output_s;
    out.kappa_probe                  = pr.kc / (pr.rho * pr.cp_eff);
    out.is_run_grid                  = pr.is_run_grid ? 1 : 0;
    {
        char hn[256] = {0};
        if (gethostname(hn, sizeof(hn) - 1) == 0) out.host = hn;
        out.device_name = solver.getDeviceInfo().name;
        std::time_t t = std::time(nullptr);
        char ts[32] = {0};
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
        out.created_utc = ts;
    }

    std::cout << "[calibrate-gpu] two-point: wall(" << pr.probe_steps_1 << ")=" << wall1
              << "s wall(" << pr.probe_steps_2 << ")=" << wall2 << "s => per-step="
              << per_step_ms << " ms, overhead=" << overhead_per_batch_s
              << " s/dispatch\n";
    std::cout << "[calibrate-gpu] => max_timesteps_per_dispatch=" << max_ts
              << " (raw " << max_ts_raw << ", cap " << cap << ", keeps one dispatch "
              << "under " << pr.ceiling_s << " s of stepping)\n";
    return 0;
}

// Ensure THIS run's grid has a representative dispatch ceiling injected before
// the main GPU solve. Idempotent per process/grid. Reaching this code implies
// use_gpu==true, so "GPU requested" holds by construction. Never throws; never
// crashes the run — on any failure it fails LOUD and leaves the conservative
// ceiling in force. See the module header above for the policy.
static void ensureGpuCalibrated(const thermal::Config& config,
                                const thermal::Planet& planet,
                                const thermal::Profile& reference_profile,
                                size_t batch, double dt_worst_s) {
    using namespace gpu::opencl;
    CoutFmtGuard _cout_fmt;   // honest small-number logging; restored on return

    // (1) Operator's explicit ceiling always wins — skip auto entirely; the
    //     static consumer path honors PTM_MAX_TIMESTEPS_PER_DISPATCH with its
    //     own log line.
    if (const char* e = std::getenv("PTM_MAX_TIMESTEPS_PER_DISPATCH")) {
        if (e[0] != '\0') return;
    }

    // Grid signature from the LIVE run.
    const auto& z      = reference_profile.z();
    const auto& kc_ref = reference_profile.kc();
    const auto& rho_ref = reference_profile.rho();
    const auto& T_ref  = reference_profile.T();
    if (z.size() < 2 || kc_ref.empty() || rho_ref.empty()) {
        std::cerr << "[auto-calibrate] WARNING: reference grid is empty/degenerate; "
                     "skipping auto-calibration (conservative ceiling in force).\n";
        return;
    }
    const int    n_layers = static_cast<int>(z.size());
    const double kc0 = kc_ref[0], rho0 = rho_ref[0];
    // Effective near-surface heat capacity. planet.cp0 is only the BASE term and
    // is 0 for bodies whose cp is a cp(T) polynomial (e.g. the Moon), so evaluate
    // the real cp(T) at a representative near-surface temperature. This drives the
    // stability ratio S in the grid signature; the probe itself uses the full
    // cp0 + cp_coeffs model below so its per-step substep count is representative.
    const double T_rep = (!T_ref.empty() && T_ref[0] > 0.0) ? T_ref[0] : 250.0;
    // cp(T) = cp0 + T*(c0 + T*(c1 + ...)) via Horner — the SAME evaluation the GPU
    // kernel's compute_cp uses (thermal_opencl.cpp) and thermal_model.cpp's
    // representative estimate.
    double cp_eff = planet.cp0;
    if (!planet.cpCoeff.empty()) {
        double result = planet.cpCoeff.back();
        for (int j = static_cast<int>(planet.cpCoeff.size()) - 2; j >= 0; --j)
            result = result * T_rep + planet.cpCoeff[j];
        cp_eff = planet.cp0 + T_rep * result;
    }
    if (cp_eff <= 0.0) cp_eff = (planet.cp0 > 0.0) ? planet.cp0 : 600.0;
    if (kc0 <= 0.0 || rho0 <= 0.0 || cp_eff <= 0.0 || dt_worst_s <= 0.0) {
        std::cerr << "[auto-calibrate] WARNING: non-positive grid property (kc0="
                  << kc0 << ", rho0=" << rho0 << ", cp_eff=" << cp_eff << ", dt="
                  << dt_worst_s << "); skipping auto-calibration (conservative "
                     "ceiling in force).\n";
        return;
    }
    double dz_min = 0.0;
    for (int i = 1; i < n_layers; ++i) {
        double d = static_cast<double>(z[i]) - static_cast<double>(z[i - 1]);
        if (d > 0.0 && (dz_min == 0.0 || d < dz_min)) dz_min = d;
    }
    if (dz_min <= 0.0) {
        std::cerr << "[auto-calibrate] WARNING: non-increasing depth grid "
                     "(dz_min<=0); skipping auto-calibration (conservative "
                     "ceiling in force).\n";
        return;
    }
    const double kappa  = kc0 / (rho0 * cp_eff);
    const int    ibatch = static_cast<int>(batch);

    // (2) Opt-out: env or config. Restore today's exact behavior (the static
    //     profile/conservative consumer path, banner and all).
    bool opted_out = false;
    if (const char* e = std::getenv("PTM_NO_AUTO_CALIBRATE")) {
        if (e[0] != '\0') opted_out = true;
    }
    if (!config.getPropertyBool("auto_calibrate_gpu", true)) opted_out = true;
    if (opted_out) {
        std::cout << "[auto-calibrate] disabled (opt-out); using host machine "
                     "profile or conservative default for the dispatch ceiling.\n";
        return;
    }

    // (3) In-process idempotency: once calibrated for THIS grid this process,
    //     reuse the already-set override (the selective-rethermal solve benefits
    //     without a redundant probe). Key on the grid only — batch is excluded so
    //     a smaller-batch rethermal reuses the larger-batch ceiling (safe:
    //     per-step cost is ~flat above GPU saturation, so a bigger-batch ceiling
    //     is conservative for a smaller batch).
    static std::string g_calibrated_key;
    char keybuf[160];
    std::snprintf(keybuf, sizeof(keybuf), "nl%d_dz%.4e_dt%.2f_k%.4e",
                  n_layers, dz_min, dt_worst_s, kappa);
    const std::string key = keybuf;
    if (g_calibrated_key == key) return;

    const int cons = gpu::MachineProfile::conservativeDefault().max_timesteps_per_dispatch;

    // Reuse a matching profile: inject its ceiling in-process and mark this grid
    // done. Shared by every no-re-probe path below (explicit / default / sidecar).
    auto reuse_profile = [&](int max_ts, const std::string& how) {
        std::cout << "[auto-calibrate] reusing " << how << ": ceiling " << max_ts
                  << " timesteps/dispatch.\n";
        setDispatchCeilingOverride(max_ts);
        g_calibrated_key = key;
    };

    // (4) Explicit hand-tuned profile (PTM_MACHINE_PROFILE): authoritative.
    const char* explicit_profile = std::getenv("PTM_MACHINE_PROFILE");
    const bool  has_explicit = explicit_profile && explicit_profile[0] != '\0';
    if (has_explicit) {
        gpu::MachineProfile hp;
        if (gpu::loadMachineProfile(hp, explicit_profile)) {
            std::string reason;
            if (gpu::gridMatchesProfile(n_layers, dz_min, dt_worst_s, kappa, ibatch,
                                        hp, reason)) {
                reuse_profile(hp.max_timesteps_per_dispatch,
                              "explicit PTM_MACHINE_PROFILE (grid matches)");
                return;
            }
            // Present but mismatched: DO NOT overwrite a hand-tuned file.
            std::cerr << "[auto-calibrate] WARNING: explicit PTM_MACHINE_PROFILE "
                         "grid mismatch (" << reason << "); NOT overwriting a "
                         "hand-tuned profile. Using the conservative ceiling "
                      << cons << " timesteps/dispatch for safety.\n";
            setDispatchCeilingOverride(cons);
            g_calibrated_key = key;
            return;
        }
        // Explicit path set but missing/corrupt: fall through and CREATE it.
    }

    // (5a) Default/host profile: reuse if it is a representative match.
    {
        gpu::MachineProfile dp;
        std::string reason;
        if (gpu::loadMachineProfile(dp, "") &&
            gpu::gridMatchesProfile(n_layers, dz_min, dt_worst_s, kappa, ibatch,
                                    dp, reason)) {
            reuse_profile(dp.max_timesteps_per_dispatch,
                          "host profile for this grid (no re-probe)");
            return;
        }
    }
    // (5b) Sidecar cache: reuse a prior probe for a matching grid (no re-probe).
    {
        gpu::MachineProfile cpf;
        std::string matched;
        if (gpu::loadMatchingCachedProfile(n_layers, dz_min, dt_worst_s, kappa,
                                           ibatch, cpf, matched)) {
            reuse_profile(cpf.max_timesteps_per_dispatch,
                          "cached host profile for this grid (no re-probe) [" +
                              matched + "]");
            return;
        }
    }

    // (6) No usable/matching profile -> calibrate this run's grid in-process.
    //     Pre-seed the override to the conservative ceiling BEFORE the probe:
    //     this (a) suppresses the alarming loadMachineProfileOrDefault stderr
    //     banner the static cache would otherwise emit when the probe dispatches,
    //     and (b) governs the probe's own launches at the safe conservative
    //     ceiling.
    setDispatchCeilingOverride(cons);

    std::cout << "[auto-calibrate] no matching host profile; measuring this run's "
                 "grid (n_layers=" << n_layers << ", dz_min=" << dz_min
              << " m, dt=" << dt_worst_s << " s, kappa=" << kappa
              << " m^2/s, batch=" << ibatch << ") before the main solve ...\n";

    CalibrationProbe pr;
    pr.z_grid.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) pr.z_grid[i] = static_cast<float>(z[i]);
    pr.n_layers   = n_layers;
    pr.batch      = ibatch;
    pr.dt_output_s = dt_worst_s;
    pr.kc = kc0; pr.rho = rho0;
    pr.cp0 = planet.cp0;             // base term (may be 0 for cp(T) bodies)
    pr.cp_coeffs.assign(planet.cpCoeff.begin(), planet.cpCoeff.end());
    pr.R350 = reference_profile.R350_val;
    pr.cond_model = static_cast<int>(config.conductivity_model);
    pr.cp_eff = cp_eff;              // effective cp used for the S signature
    pr.emissivity = (planet.emissivity > 0.0) ? planet.emissivity : 0.95;
    pr.ceiling_s  = 25.0;             // documented survivable per-dispatch window
    pr.is_run_grid = true;
    pr.probe_budget_s = 15.0;

    gpu::MachineProfile out;
    int rc = 4;
    try {
        rc = calibrateGpuCore(pr, out);
    } catch (const std::exception& ex) {
        std::cerr << "[auto-calibrate] WARNING: probe threw (" << ex.what()
                  << "); using conservative ceiling " << cons << ".\n";
        rc = 7;
    } catch (...) {
        std::cerr << "[auto-calibrate] WARNING: probe threw; using conservative "
                     "ceiling " << cons << ".\n";
        rc = 7;
    }

    if (rc != 0) {
        std::cerr << "[auto-calibrate] WARNING: calibration failed (rc=" << rc
                  << "); the run continues with the conservative ceiling " << cons
                  << " timesteps/dispatch (slower, but safe).\n";
        g_calibrated_key = key;   // leave override at the conservative value
        return;
    }

    // Success: inject the representative ceiling in-process and persist it.
    setDispatchCeilingOverride(out.max_timesteps_per_dispatch);
    g_calibrated_key = key;

    std::string cache_file = gpu::calibCacheFileForGrid(n_layers, dz_min,
                                                        dt_worst_s, kappa, ibatch);
    bool saved_cache = !cache_file.empty() && gpu::saveMachineProfile(out, cache_file);
    std::string default_path = has_explicit ? std::string(explicit_profile)
                                            : std::string("");
    bool saved_default = gpu::saveMachineProfile(out, default_path);

    std::cout << "[auto-calibrate] representative ceiling = "
              << out.max_timesteps_per_dispatch << " timesteps/dispatch (per-step "
              << out.kernel_ms_per_step_per_batch << " ms, keeps one dispatch under "
              << out.safe_dispatch_ceiling_s << " s)"
              << (saved_cache ? "" : " [cache write skipped]")
              << (saved_default ? "" : " [profile write skipped]") << "\n";
}

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
    std::vector<float>* post_eq_T_out,
    int gpu_devices,
    const std::vector<int>& gpu_device_ids,
    size_t gpu_batch_size_override,
    int gpu_inter_batch_pause_ms,
    const std::vector<double>& eq_indirect_flux)
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
    applyConfiguredThermo(planet, config);   // generic-body props + user scalar overrides (PTM-53)
    thermal::Profile reference_profile(planet, 0.0, config, false);
    const auto& z_grid = reference_profile.z();
    const size_t n_layers = z_grid.size();

    // Shared-seed guards, reported after the batch loop (never silent).
    size_t gpu_seed_floor_faces = 0, gpu_seed_hot_faces = 0;
    if (config.spinup_seed == thermal::EqSeedMode::Physical) {
        thermal::EqSeedRecorder::instance().reserve(mesh->faces.size());
        std::cerr << "[eq-seed] equilibration seed: SHARED per-face conductive "
                     "steady profile -- T_0=((<F_abs>+Qb)/(eps*sigma))^0.25 with "
                     "<F_abs> the mean over all forcing steps, then "
                     "T(z)=T_0+Qb*integral dz/k [thermal.spinup_seed: physical; "
                     "identical on every solver path]" << std::endl;
    }

    // Directional-hemispheric (Hapke) albedo gate. The gate is resolved on the
    // main-scope Planet and forwarded as thermal_config properties (see the
    // thermal_config build site). This locally-rebuilt Planet does not inherit
    // it, so re-read it here and drive the host-side flux build below. Default
    // OFF → byte-identical Hayne-only path preserved for all non-Hapke runs.
    const bool   use_hapke = config.getPropertyBool("use_hapke_albedo", false)
                             && config.getPropertyDouble("hapke_albedo_w", -1.0) >= 0.0;
    const double hapke_w   = config.getPropertyDouble("hapke_albedo_w", -1.0);
    planet.use_hapke_albedo = use_hapke;
    planet.hapke_albedo_w   = hapke_w;
    if (use_hapke) {
        // Mirror the CPU reference log (calculateAbsorbedFlux, ~line 1442) so a
        // gate-ON run is verifiable from stdout — proves the branch FIRES.
        std::cout << "[GPU Thermal] Using directional-hemispheric (Hapke) albedo "
                     "A_h(i) = 1 - gamma*H(mu0,w)" << std::endl;
        // PTM-55: this banner used to print `planet.albedo` -- the albedo of the
        // Planet REBUILT two lines above from the body name, which inherits the
        // body default and never re-reads the configured value. That is not the
        // number the physics uses: face_albedos is initialized from
        // default_visible_albedo (below), and each face's Hapke w is derived
        // from ITS OWN face_albedos entry. A lunar campaign log therefore
        // carried two different values for one quantity ("A0 = 0.12" from the
        // main scope, "A0 = 0.1" here) with no way for a reader to tell which
        // the solver saw. Report the operative value, and say where it came
        // from -- per PTM-05.
        std::cout << "[GPU Thermal]   Base albedo A0 = " << std::defaultfloat
                  << std::setprecision(6) << default_visible_albedo
                  << " (operative: face_albedos is initialized to this; a loaded "
                     "albedo map supersedes it per face), single-scattering w = "
                  << hapke_w << std::endl;
        if (std::abs(planet.albedo - default_visible_albedo) > 1e-9) {
            // Explicit precision -- see the CPU sibling: the inherited stream
            // state can print two different values identically (PTM-09 class).
            std::ios_base::fmtflags f(std::cerr.flags());
            std::streamsize pr = std::cerr.precision();
            std::cerr << std::defaultfloat << std::setprecision(6)
                      << "[GPU Thermal] WARNING: the solver-scope Planet(\""
                      << planet.name << "\") carries albedo " << planet.albedo
                      << ", which DISAGREES with the operative "
                      << default_visible_albedo
                      << ". The operative value is what the flux uses; the "
                         "solver-scope Planet's albedo is used for nothing else. "
                         "See docs/KNOWN_LIMITATIONS.md (PTM-55)." << std::endl;
            std::cerr.flags(f);
            std::cerr.precision(pr);
        }
    }

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

    // A warm-start chunk is one that resumes from a checkpoint. It forces
    // equilibration_orbits=0 upstream (main, "[chunked] thermal.initial_state_file
    // is set"), so it ALWAYS lands on the single-phase solver branch below.
    const bool warm_start_run = !initial_state_file.empty();

    // Whole-mesh buffer to receive the final T(z) profile, if requested.
    // Filled per-batch and dumped at the end.
    std::vector<float> assembled_final_T;
    const bool dump_final_state = !final_state_file.empty();
    if (dump_final_state) {
        assembled_final_T.assign(n_layers * n_faces, 0.0f);
    }

    // Whole-mesh buffer to receive iter-0's POST-EQUILIBRATION T(z), if
    // requested by the caller via post_eq_T_out. Filled per-batch right
    // after the eq-phase solver call (two-phase configs) or at the end
    // of the single-phase cycle. Used by the radiation-coupling iter
    // loop so re-thermal can upload the same starting state iter-0 used
    // for its output cycle.
    std::vector<float> assembled_post_eq_T;
    if (post_eq_T_out) {
        assembled_post_eq_T.assign(n_layers * n_faces, 0.0f);
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

    // ---- Equilibration cycling knob ------------------------------------
    // eq_passes = number of passes over the equilibration forcing array.
    // solvePersistentMultipleCycles runs (n_equil_cycles + 1) passes, so the
    // value handed to it is eq_passes - 1. main() has ALREADY divided
    // thermal.equilibration_orbits by this pass count when it sized the
    // equilibration window, so eq_passes x (this block) == the configured
    // number of integrated diurnal cycles. See kDefaultEquilibrationPasses.
    bool   eq_cycles_explicit = false;
    std::string eq_cycles_source;
    const int eq_passes = resolveEquilibrationPasses(config, &eq_cycles_explicit,
                                                     &eq_cycles_source);
    if (eq_passes <= 0) {
        std::cerr << "[GPU Thermal] FATAL: invalid equilibration_cycles request; "
                     "refusing to run rather than silently using the default.\n";
        return false;
    }
    {
        const int    cycled_steps = two_phase ? equilibration_steps : total_time_steps;
        const double cycled_dt    = two_phase ? eq_dt : dt;
        const double synodic_s    = getSynodicDayFromPlanet(planet) * 86400.0;
        const double eq_seconds   = static_cast<double>(eq_passes)
                                  * static_cast<double>(cycled_steps) * cycled_dt;
        std::ios_base::fmtflags bf(std::cout.flags());
        std::streamsize bp = std::cout.precision();
        std::cout << "[GPU Thermal] Equilibration cycling: " << eq_passes
                  << " passes x " << cycled_steps << " steps x "
                  << std::fixed << std::setprecision(1) << cycled_dt << " s = "
                  << (static_cast<long long>(eq_passes) * cycled_steps)
                  << " integrated steps";
        if (synodic_s > 0.0) {
            std::cout << " = " << std::setprecision(2) << (eq_seconds / synodic_s)
                      << " equivalent diurnal cycles";
        }
        std::cout << "  [" << eq_cycles_source << "; default = "
                  << kDefaultEquilibrationPasses << " pass, pre-2026-08-11 = "
                  << kHistoricalEquilibrationPasses << "]" << std::endl;
        std::cout.flags(bf);
        std::cout.precision(bp);
    }
    // Paths that do NOT honor the knob must say so — a silently ignored
    // equilibration request is exactly the class of failure that makes a run
    // unreproducible.
    if (eq_cycles_explicit && warm_start_run) {
        std::cerr << "[GPU Thermal] WARNING: equilibration_cycles=" << eq_passes
                  << " requested, but this is a WARM-START chunk "
                     "(initial_state_file set). The loaded T(z) is already "
                     "equilibrated and the window is integrated EXACTLY ONCE "
                     "(PTM-54/57/78); the knob is IGNORED here.\n";
    } else if (eq_cycles_explicit && !two_phase) {
        std::cerr << "[GPU Thermal] WARNING: equilibration_cycles=" << eq_passes
                  << " on a SINGLE-PHASE run: the cycled array is the FULL "
                     "window (equilibration + output, " << total_time_steps
                  << " steps), not an equilibration-only prefix.\n";
    }
    if (eq_cycles_explicit && equilibration_steps <= 0 && !warm_start_run) {
        std::cerr << "[GPU Thermal] WARNING: equilibration_cycles=" << eq_passes
                  << " requested with equilibration_steps=0; there is no "
                     "equilibration phase to cycle.\n";
    }

    // ---- Pass-stride RRE extrapolation ---------------------------------
    // Default OFF. When off, `eq_extrap.enabled == false`, the eq call sites
    // below hand the solver a disabled Options and it takes the pre-existing
    // path with no readback, no upload and no extra allocation.
    ptm::eqextrap::Options eq_extrap;
    bool eq_extrap_requested = false;
    if (!resolveEqExtrapolation(config, eq_passes, &eq_extrap, &eq_extrap_requested)) {
        std::cerr << "[GPU Thermal] FATAL: invalid eq_extrapolation request; "
                     "refusing to run.\n";
        return false;
    }
    // The extrapolation is defined on the EQUILIBRATION pass map. Anything that
    // is not a two-phase, cold-start equilibration must say out loud that the
    // knob is inert rather than let the config imply it was applied.
    if (eq_extrap.enabled && warm_start_run) {
        std::cerr << "[GPU Thermal] WARNING: eq_extrapolation is ON but this is a "
                     "WARM-START chunk; there is no equilibration pass map to "
                     "extrapolate. The knob is IGNORED here.\n";
        eq_extrap.enabled = false;
    }
    if (eq_extrap.enabled && !two_phase) {
        std::cerr << "[GPU Thermal] WARNING: eq_extrapolation is ON but this is a "
                     "SINGLE-PHASE run: the cycled array is the full window, not "
                     "an equilibration prefix. The knob is IGNORED here.\n";
        eq_extrap.enabled = false;
    }
    if (eq_extrap.enabled) {
        std::cout << "[eq-extrap] pass-stride RRE ARMED: order_p=" << eq_extrap.order_p
                  << " over " << eq_passes << " passes (extrapolate after "
                  << (eq_passes - 1) << ", pass " << eq_passes
                  << " = polish + certificate + delivered state), tol_K="
                  << eq_extrap.tol_K << " hold_K=" << eq_extrap.hold_K
                  << " K/pass" << std::endl;
    }
    ptm::eqextrap::Stats eq_extrap_stats;
    std::mutex eq_extrap_mutex;
    // ONE place that knows the locking rule. The multi-GPU pool runs eq batches
    // concurrently, so every merge into eq_extrap_stats must hold the mutex;
    // keeping that discipline at the call sites would mean a future third eq
    // call site could copy the solver call and quietly omit the lock.
    auto merge_extrap_stats = [&](const ptm::eqextrap::Stats& s) {
        if (!eq_extrap.enabled) return;
        std::lock_guard<std::mutex> lk(eq_extrap_mutex);
        eq_extrap_stats.merge(s);
    };

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
    // on macOS+AMD; see memory/gpu_hang_amd_macos.md). 0 keeps legacy behavior.
    const size_t BATCH_SIZE = (gpu_batch_size_override > 0)
        ? std::min(n_faces, std::min(max_batch_size_legacy, gpu_batch_size_override))
        : std::min(max_batch_size_legacy, static_cast<size_t>(500000));

    std::cout << "[GPU Thermal] Using batch size: " << BATCH_SIZE
              << " (n_layers=" << n_layers << ")" << std::endl;

    // Phase R (auto-calibrate): inject a representative dispatch ceiling for THIS
    // run's grid before the main solve. Bind on the WORST-CASE (largest) dt the
    // run will pass to the persistent kernel (more substeps => slower step =>
    // smaller, safer ceiling). No-op / cheap cache-hit when already calibrated,
    // opted out, or an env ceiling is set. Numerically inert (batching only).
    ensureGpuCalibrated(config, planet, reference_profile, BATCH_SIZE,
                        static_cast<double>(std::max(eq_dt, out_dt)));

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

    // Per-face Hapke single-scattering albedo (per-face albedo scaling):
    // precompute w for each face from its map albedo A0 so the Hapke branch is
    // anchored to that face's OWN albedo rather than the body average. Closed
    // form, O(n_faces); reused across all timesteps in the prepare_flux loops
    // below. Inert when Hapke is off (Hayne/constant-albedo paths ignore it).
    std::vector<double> face_w(n_faces);
    for (size_t f = 0; f < n_faces; ++f) face_w[f] = hapkeWFromA0(face_albedos[f]);

    // PTM-107(a) on the GPU path (Fermi condition C1). OUTPUT-ONLY: reads the flux
    // table and face_albedos/face_w, writes only the two file-scope accumulators.
    // The incidence gate is reproduced here from the SAME planet fields the batch
    // workers use for their own `use_incidence` (see prepare_flux below) so the
    // recorded albedo is the one the solver actually absorbed with.
    {
        const double aeff_a = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[0] : 0.0;
        const double aeff_b = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[1] : 0.0;
        const double aeff_S = planet.S;
        const bool   aeff_incidence = (planet.albedoCoef.size() >= 2 && aeff_S > 0.0)
                                    || (use_hapke && aeff_S > 0.0);
        accumulateEffectiveAlbedoGPU(n_faces, total_time_steps, direct_solar_values,
                                     face_albedos, face_w, aeff_incidence, use_hapke,
                                     aeff_a, aeff_b, aeff_S);
    }

    // Process faces in batches
    size_t n_batches = (n_faces + BATCH_SIZE - 1) / BATCH_SIZE;
    std::cout << "[GPU Thermal] Processing " << n_faces << " faces in " << n_batches << " batch(es)..." << std::endl;

    // Equilibration forcing dump (diagnostic, default OFF). Declared here so it
    // is in scope on BOTH the multi-GPU pool path and the serial loop, and ahead
    // of the `goto post_batch_loop` below. configure() is a no-op when the env
    // var is unset and fatal when the request cannot be honored exactly.
    eqlundump::Buffers eq_lun_dump;
    eqlundump::configure(eq_lun_dump, eqlundump::requestedDir(), n_faces,
                         equilibration_steps, two_phase,
                         static_cast<double>(eq_dt),
                         getSynodicDayFromPlanet(planet));

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
            static_cast<float>(ptm::kStefanBoltzmann),
            static_cast<int>(config.conductivity_model)
        );

        // ONE absorbed-flux composition for this batch, used by BOTH the
        // spin-up seed below and prepare_flux further down. It used to be
        // written out twice, and the two copies disagreed: prepare_flux
        // applied the incidence-angle (Hayne / Hapke) albedo A(i) while the
        // seed used the flat per-face Bond albedo, so the seed was built from
        // a different forcing than the one the same function then uploaded.
        // Measured on the lunar 80N crater: up to 48.9 W/m^2 of mean flux and
        // 39.5 K of seed temperature between the CPU and GPU seeds, all of it
        // this. Same expression, one place.
        const double a_coef = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[0] : 0.0;
        const double b_coef = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[1] : 0.0;
        const double S_solar = planet.S;
        const bool   has_angular = (planet.albedoCoef.size() >= 2 && S_solar > 0.0);
        // Incidence-dependent albedo fires for Hayne (has_angular) OR the
        // Hapke gate. Both need S>0 to recover cos(theta) from flux. When both
        // are off this is false and applyIncidenceAlbedo is bypassed ->
        // byte-identical constant-albedo path.
        const bool   use_incidence = has_angular || (use_hapke && S_solar > 0.0);
        const std::function<double(size_t, int)> absorbed_at =
            [&](size_t global_f, int t_abs) -> double {
            const size_t flux_idx =
                global_f * static_cast<size_t>(total_time_steps) + t_abs;
            const double base_A    = face_albedos[global_f];
            const double direct    = (flux_idx < direct_solar_values.size())
                                        ? direct_solar_values[flux_idx] : 0.0;
            const double reflected = (flux_idx < primary_reflected_values.size())
                                        ? primary_reflected_values[flux_idx] : 0.0;
            const double ir        = (flux_idx < primary_ir_values.size())
                                        ? primary_ir_values[flux_idx] : 0.0;
            // Hayne (2017): A(theta) = A0 + a*(theta/(pi/4))^3 +
            // b*(theta/(pi/2))^8. Hapke branch (opt-in): A_h(mu0) =
            // 1 - gamma*H(mu0,w), which supersedes Hayne and ignores base_A.
            const double albedo = use_incidence
                ? applyIncidenceAlbedo(base_A, direct,
                                       solarConstForStep(t_abs, S_solar),
                                       a_coef, b_coef, use_hapke, face_w[global_f])
                : base_A;
            // Kept as two separate products, NOT (1-A)*(direct+reflected):
            // a*x + a*y is not bit-identical to a*(x+y) in floating point, and
            // this expression has to stay byte-identical to the pre-refactor
            // GPU path under thermal.spinup_seed: legacy.
            return (1.0 - albedo) * direct + (1.0 - albedo) * reflected
                 + planet.emissivity * ir;
        };

        // Initial subsurface temperature seed.
        //
        // Generic-body path: seed every face with the smooth, longitude-independent
        // diurnal-mean radiative-equilibrium reference profile (Profile::init_T,
        // T_avg = T_ss/sqrt(2); docs/equilibration-paper.md §5), identical to the
        // CPU path's starting state. The legacy per-face seed (else branch) derived
        // T(z,0) from the *instantaneous* absorbed flux over the first 10
        // equilibration steps, with a hard 100 K nightside floor, which stamped a
        // body-fixed day/night (terminator) discontinuity into the subsurface. Once
        // the configured (higher) thermal inertia is applied on the generic-body
        // path a finite equilibration no longer erases that imprint, and it
        // surfaces as the body-fixed "seam" in the spinning-asteroid flybys. The
        // smooth seed removes it at the source and matches the CPU path.
        //
        // Gated on "using_generic_body" (mirrors applyConfiguredThermo / 710db0f):
        // built-in Moon/Europa/A3GT runs keep the legacy seed and stay
        // byte-identical to their established baselines.
        std::vector<float> T_initial(batch_n * n_layers);
        if (config.spinup_seed == thermal::EqSeedMode::Physical) {
            GpuSharedSeedContext sc;
            sc.planet = &planet;
            sc.z = z_grid.data();
            sc.n_layers = n_layers;
            sc.kc_ref = kc_ref.data();
            sc.rho_ref = rho_ref.data();
            sc.face_H_overrides = face_H_overrides;
            sc.face_Qb = face_Qb;
            sc.batch_start = batch_start;
            sc.batch_n = batch_n;
            sc.absorbed = &absorbed_at;
            sc.total_time_steps = total_time_steps;
            sc.R350 = R350_val;
            sc.cond = config.conductivity_model;
            gpuFillSharedSeed(sc, T_initial, gpu_seed_floor_faces, gpu_seed_hot_faces);
        } else if (config.getProperty("using_generic_body") == "true") {
            const auto& T_seed = reference_profile.T();  // aligned_vector<double>, nlayers
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float Tl = static_cast<float>(T_seed[layer]);
                for (size_t f = 0; f < batch_n; ++f) {
                    T_initial[layer * batch_n + f] = Tl;
                }
            }
        } else {
            const bool perface = eqPerfaceFluxMode();
            for (size_t f = 0; f < batch_n; ++f) {
                size_t global_f = batch_start + f;
                double albedo = face_albedos[global_f];
                double avg_flux = 0.0;
                int n_samples = perface ? total_time_steps : std::min(10, total_time_steps);
                for (int t = 0; t < n_samples; ++t) {
                    size_t flux_idx = global_f * total_time_steps + t;
                    double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                    double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                    double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;
                    avg_flux += (1.0 - albedo) * direct + (1.0 - albedo) * reflected + planet.emissivity * ir;
                }
                avg_flux /= n_samples;

                if (perface) {
                    // Fermi PORT-EQ-01: seed each column from ITS OWN time-mean
                    // absorbed flux + Qb. PSR floors (mean ~0) cold-start at the
                    // geothermal steady state ~24.7 K; sunlit ~unchanged after eq.
                    // Isothermal at the diurnal-mean equilibrium (no warm floor).
                    double T_eq = std::sqrt(std::sqrt(
                        (avg_flux + planet.Qb) / (planet.emissivity * ptm::kStefanBoltzmann)));
                    for (size_t layer = 0; layer < n_layers; ++layer)
                        T_initial[layer * batch_n + f] = static_cast<float>(T_eq);
                } else {
                    double T_eq = std::sqrt(std::sqrt(avg_flux / (planet.emissivity * ptm::kStefanBoltzmann)));
                    if (T_eq < 40.0) T_eq = 100.0;
                    if (T_eq > 400.0) T_eq = 400.0;
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        size_t idx = layer * batch_n + f;
                        double depth_frac = z_grid[layer] / z_grid[n_layers - 1];
                        T_initial[idx] = static_cast<float>(T_eq * (1.0 - depth_frac) + T_deep_val * depth_frac);
                    }
                }
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

                    (void)flux_idx;
                    double absorbed = absorbed_at(global_f, t_start + t);
                    // Deep radiation-coupling: add the per-face time-mean indirect
                    // flux to the EQUILIBRATION-phase forcing only. The output
                    // window already gets the time-varying indirect via the
                    // coupling loop, so this cannot double-count there.
                    //
                    // The test is on the ABSOLUTE step index, not on the shape of
                    // the call. It used to read `t_start == 0 && n_steps ==
                    // equilibration_steps`, which identifies the eq call only in
                    // TWO-PHASE mode; the single-phase branch prepares one array
                    // with prepare_flux(0, total_time_steps) and therefore dropped
                    // the term entirely -- silently, after printing the "Loaded
                    // eq-indirect flux" banner. Absolute-step selection covers both
                    // and is byte-identical on the two-phase path (its eq call has
                    // t_start = 0 and every step < equilibration_steps; its output
                    // call has t_start = equilibration_steps and no step qualifies).
                    if (!eq_indirect_flux.empty()
                            && (t_start + t) < equilibration_steps
                            && global_f < eq_indirect_flux.size()) {
                        absorbed += eq_indirect_flux[global_f];
                    }
                    flux[t * batch_n + f] = static_cast<float>(absorbed);
                }
            }
            return flux;
        };

        // Passes over the eq forcing array, minus the +1 that
        // solvePersistentMultipleCycles adds internally (total_cycles =
        // n_equil_cycles + 1). eq_passes defaults to 1 (pre-2026-08-11: 2).
        int equil_cycles = eq_passes - 1;
        std::vector<float> output_temps;

        std::vector<float> batch_avg_E;
        std::vector<float>* batch_avg_E_ptr = nullptr;
        if (ice_acc) {
            batch_avg_E.assign(static_cast<size_t>(n_layers) * batch_n, 0.0f);
            batch_avg_E_ptr = &batch_avg_E;
        }

        if (two_phase) {
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                // Report PASSES (equil_cycles + 1) — the number the kernel
                // actually runs and logs. The old banner printed equil_cycles
                // and so understated the eq phase by one full pass.
                std::cout << "[GPU Thermal] Phase 1: " << (equil_cycles + 1)
                          << " eq passes × " << equilibration_steps
                          << " steps (dt=" << std::fixed << std::setprecision(1)
                          << eq_dt << "s)" << std::endl;
            }
            auto flux_eq = prepare_flux(0, equilibration_steps);
            // Diagnostic dump of the equilibration forcing (default OFF).
            // Reduces the solver's OWN flux array, then abandons the batch
            // before any kernel dispatch — the dump path must never run a solve.
            if (eq_lun_dump.enabled) {
                eqlundump::accumulateBatch(eq_lun_dump, flux_eq, direct_solar_values,
                                           batch_start, batch_n,
                                           equilibration_steps, total_time_steps);
                return true;
            }
            std::vector<float> eq_output_discard;
            // Pass-stride RRE. Batch-local Stats: a shared accumulator would be
            // a data race on every counter under the multi-GPU pool.
            ptm::eqextrap::Stats batch_extrap_stats;
            solver.solvePersistentMultipleCycles(flux_eq, eq_output_discard,
                                                 equilibration_steps, eq_dt, equil_cycles,
                                                 nullptr, 0, -1,
                                                 &eq_extrap, &batch_extrap_stats);
            merge_extrap_stats(batch_extrap_stats);

            // Snapshot the post-eq T(z) profile here — this is exactly
            // the initial state the output cycle is about to run from.
            // The rad-coupling iter loop reuses this so re-thermal can
            // skip its own eq phase and run with iter-0's converged
            // depth profile (instead of a cold-start that may converge
            // to a different fixed point).
            if (post_eq_T_out) {
                std::vector<float> batch_T_post_eq;
                try {
                    solver.downloadAllTemps(batch_T_post_eq);
                } catch (const std::exception& e) {
                    std::cerr << "[GPU Thermal] WARNING: downloadAllTemps (post-eq) failed: "
                              << e.what() << "\n";
                    batch_T_post_eq.clear();
                }
                // downloadAllTemps packs data with stride = activeFaceCount(),
                // which EXCEEDS batch_n when the persistent solver was allocated
                // to a larger capacity (e.g. a shared pool sized to the
                // re-thermal cap). The old guard compared the size to
                // batch_n*n_layers and the copy assumed a batch_n stride; both
                // are wrong in that case, so the snapshot silently captured
                // NOTHING and the rad-coupling re-thermal later started every
                // selected face from a zero T(z) column. That was masked by the
                // old [3,500] K clamp (the zero column was clamped to the 3 K
                // floor — the long-standing "3 K PSR" artifact) and surfaces as
                // a 0 K -> runaway divergence once the clamp is removed. Use the
                // actual active-face stride, and warn loudly if capture fails.
                const size_t active = solver.activeFaceCount();
                if (active >= batch_n && n_layers > 0 &&
                    batch_T_post_eq.size() == active * n_layers) {
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        const float* src = batch_T_post_eq.data() + layer * active;
                        float*       dst = assembled_post_eq_T.data() + layer * n_faces + batch_start;
                        std::copy(src, src + batch_n, dst);
                    }
                } else {
                    std::cerr << "[GPU Thermal] WARNING: post-eq snapshot NOT captured"
                                 " (download size " << batch_T_post_eq.size()
                              << ", active=" << active << ", batch_n=" << batch_n
                              << ", n_layers=" << n_layers << "); rad-coupling"
                                 " re-thermal would start from a zero column\n";
                }
            }

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
            // ---- Single-phase (legacy dt, and EVERY warm-start chunk) ----
            // Two defects used to live on exactly this branch (BUGS.md PTM-78,
            // and the mechanism behind PTM-54/PTM-57):
            //
            //  1. equil_cycles == 1 made solvePersistentMultipleCycles run
            //     total_cycles = n_equil_cycles + 1 = 2 passes over the chunk's
            //     OWN flux sequence and report the second. A chunked
            //     continuation therefore replayed its window instead of
            //     continuing the trajectory. A warm start's loaded T(z) is
            //     already equilibrated — that is the entire premise of the
            //     checkpoint — so it integrates the window exactly once.
            //
            //  2. post_eq_T_out was filled ONLY in the two_phase branch above,
            //     so on this branch the rad-coupling re-thermal was handed an
            //     all-zero T(z) column for every selected face. The function's
            //     own doc comment already promised the single-phase fill; the
            //     code never did it.
            const int sp_equil_cycles = warm_start_run ? 0 : equil_cycles;
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Running " << sp_equil_cycles
                          << " equilibration cycles + 1 output cycle"
                          << (warm_start_run
                                  ? " (warm start: window integrated ONCE)" : "")
                          << "..." << std::endl;
            }
            auto flux_all = prepare_flux(0, total_time_steps);
            // Pre-cycles, output discarded. Splitting one (n+1)-cycle call into
            // an n-cycle call plus a 1-cycle call is numerically inert: buf_T_
            // persists across calls, kernel/flux/dt are identical, and only the
            // final cycle ever accumulated. It buys the snapshot point below.
            if (sp_equil_cycles > 0) {
                std::vector<float> sp_discard;
                solver.solvePersistentMultipleCycles(flux_all, sp_discard,
                                                     total_time_steps, dt,
                                                     sp_equil_cycles - 1);
            }
            // Snapshot the state the OUTPUT cycle is about to start from — the
            // single-phase analog of the two-phase post-eq snapshot above.
            if (post_eq_T_out) {
                std::vector<float> batch_T_pre_out;
                try {
                    solver.downloadAllTemps(batch_T_pre_out);
                } catch (const std::exception& e) {
                    std::cerr << "[GPU Thermal] WARNING: downloadAllTemps (single-phase "
                                 "pre-output) failed: " << e.what() << "\n";
                    batch_T_pre_out.clear();
                }
                const size_t active = solver.activeFaceCount();
                if (active >= batch_n && n_layers > 0 &&
                    batch_T_pre_out.size() == active * n_layers) {
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        const float* src = batch_T_pre_out.data() + layer * active;
                        float*       dst = assembled_post_eq_T.data() + layer * n_faces + batch_start;
                        std::copy(src, src + batch_n, dst);
                    }
                } else {
                    std::cerr << "[GPU Thermal] WARNING: single-phase pre-output snapshot"
                                 " NOT captured (download size " << batch_T_pre_out.size()
                              << ", active=" << active << ", batch_n=" << batch_n
                              << ", n_layers=" << n_layers << "); rad-coupling"
                                 " re-thermal would start from a zero column\n";
                }
            }
            solver.solvePersistentMultipleCycles(flux_all, output_temps,
                                                 total_time_steps, dt, 0,
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

        ThermalOpenCLConfig opencl_config;
        opencl_config.verbose = false;
        opencl_config.work_group_size = 64;

        // Persistent-solver pool: one ThermalOpenCLSolver per device, sized
        // to BATCH_SIZE capacity, reused across every batch on its device.
        // Constructor eagerly calls solver.initialize() per device, so any
        // OpenCL init failure surfaces here (not mid-loop).
        ThermalGPUPoolConfig pool_cfg;
        pool_cfg.devices = selected;
        pool_cfg.max_n_faces = BATCH_SIZE;
        pool_cfg.n_layers = n_layers;
        pool_cfg.solver_config = opencl_config;
        pool_cfg.inter_batch_pause_ms = gpu_inter_batch_pause_ms;
        pool_cfg.verbose = false;
        std::unique_ptr<ThermalGPUDevicePool> pool;
        try {
            pool = std::make_unique<ThermalGPUDevicePool>(std::move(pool_cfg));
        } catch (const std::exception& e) {
            std::cerr << "[GPU Thermal] Multi-GPU pool construction failed: "
                      << e.what() << std::endl;
            return false;
        }

        std::vector<std::future<bool>> futures;
        futures.reserve(n_batches);
        for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
            const size_t batch_start = batch_idx * BATCH_SIZE;
            const size_t batch_end   = std::min(batch_start + BATCH_SIZE, n_faces);
            const size_t batch_n     = batch_end - batch_start;
            futures.push_back(pool->submit(batch_n,
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
            static_cast<float>(ptm::kStefanBoltzmann),
            static_cast<int>(config.conductivity_model)  // Stefan-Boltzmann
        );

        // ONE absorbed-flux composition for this batch, used by BOTH the
        // spin-up seed below and prepare_flux further down. It used to be
        // written out twice, and the two copies disagreed: prepare_flux
        // applied the incidence-angle (Hayne / Hapke) albedo A(i) while the
        // seed used the flat per-face Bond albedo, so the seed was built from
        // a different forcing than the one the same function then uploaded.
        // Measured on the lunar 80N crater: up to 48.9 W/m^2 of mean flux and
        // 39.5 K of seed temperature between the CPU and GPU seeds, all of it
        // this. Same expression, one place.
        const double a_coef2 = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[0] : 0.0;
        const double b_coef2 = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[1] : 0.0;
        const double S_solar2 = planet.S;
        const bool   has_angular2 = (planet.albedoCoef.size() >= 2 && S_solar2 > 0.0);
        // Incidence-dependent albedo fires for Hayne (has_angular) OR the
        // Hapke gate. Both need S>0 to recover cos(theta) from flux. When both
        // are off this is false and applyIncidenceAlbedo is bypassed ->
        // byte-identical constant-albedo path.
        const bool   use_incidence2 = has_angular2 || (use_hapke && S_solar2 > 0.0);
        const std::function<double(size_t, int)> absorbed_at =
            [&](size_t global_f, int t_abs) -> double {
            const size_t flux_idx =
                global_f * static_cast<size_t>(total_time_steps) + t_abs;
            const double base_A    = face_albedos[global_f];
            const double direct    = (flux_idx < direct_solar_values.size())
                                        ? direct_solar_values[flux_idx] : 0.0;
            const double reflected = (flux_idx < primary_reflected_values.size())
                                        ? primary_reflected_values[flux_idx] : 0.0;
            const double ir        = (flux_idx < primary_ir_values.size())
                                        ? primary_ir_values[flux_idx] : 0.0;
            // Hayne (2017): A(theta) = A0 + a*(theta/(pi/4))^3 +
            // b*(theta/(pi/2))^8. Hapke branch (opt-in): A_h(mu0) =
            // 1 - gamma*H(mu0,w), which supersedes Hayne and ignores base_A.
            const double albedo = use_incidence2
                ? applyIncidenceAlbedo(base_A, direct,
                                       solarConstForStep(t_abs, S_solar2),
                                       a_coef2, b_coef2, use_hapke, face_w[global_f])
                : base_A;
            // Kept as two separate products, NOT (1-A)*(direct+reflected):
            // a*x + a*y is not bit-identical to a*(x+y) in floating point, and
            // this expression has to stay byte-identical to the pre-refactor
            // GPU path under thermal.spinup_seed: legacy.
            return (1.0 - albedo) * direct + (1.0 - albedo) * reflected
                 + planet.emissivity * ir;
        };

        // Initial subsurface temperature seed. Under thermal.spinup_seed:
        // physical (the default) every path calls the SAME eqSeedColumn; the
        // two legacy branches below are kept verbatim so `legacy` reproduces
        // the pre-2026-08-11 baselines exactly.
        std::vector<float> T_initial(batch_n * n_layers);
        if (config.spinup_seed == thermal::EqSeedMode::Physical) {
            GpuSharedSeedContext sc;
            sc.planet = &planet;
            sc.z = z_grid.data();
            sc.n_layers = n_layers;
            sc.kc_ref = kc_ref.data();
            sc.rho_ref = rho_ref.data();
            sc.face_H_overrides = face_H_overrides;
            sc.face_Qb = face_Qb;
            sc.batch_start = batch_start;
            sc.batch_n = batch_n;
            sc.absorbed = &absorbed_at;
            sc.total_time_steps = total_time_steps;
            sc.R350 = R350_val;
            sc.cond = config.conductivity_model;
            gpuFillSharedSeed(sc, T_initial, gpu_seed_floor_faces, gpu_seed_hot_faces);
        } else if (config.getProperty("using_generic_body") == "true") {
            const auto& T_seed = reference_profile.T();  // aligned_vector<double>, nlayers
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const float Tl = static_cast<float>(T_seed[layer]);
                for (size_t f = 0; f < batch_n; ++f) {
                    T_initial[layer * batch_n + f] = Tl;
                }
            }
        } else {
            const bool perface = eqPerfaceFluxMode();
            for (size_t f = 0; f < batch_n; ++f) {
                // Get initial flux to estimate equilibrium temperature
                // Include ALL flux components (direct, reflected, IR) for accurate initialization
                size_t global_f = batch_start + f;
                double albedo = face_albedos[global_f];
                double avg_flux = 0.0;
                int n_samples = perface ? total_time_steps : std::min(10, total_time_steps);
                for (int t = 0; t < n_samples; ++t) {
                    size_t flux_idx = global_f * total_time_steps + t;
                    double direct = (flux_idx < direct_solar_values.size()) ? direct_solar_values[flux_idx] : 0.0;
                    double reflected = (flux_idx < primary_reflected_values.size()) ? primary_reflected_values[flux_idx] : 0.0;
                    double ir = (flux_idx < primary_ir_values.size()) ? primary_ir_values[flux_idx] : 0.0;
                    // Absorbed flux = (1-albedo)*direct + (1-albedo)*reflected + (1-ir_albedo)*ir
                    avg_flux += (1.0 - albedo) * direct + (1.0 - albedo) * reflected + planet.emissivity * ir;
                }
                avg_flux /= n_samples;

                if (perface) {
                    // Fermi PORT-EQ-01: seed each column from ITS OWN time-mean
                    // absorbed flux + Qb (PSR floors cold-start at the geothermal
                    // steady state ~24.7 K; sunlit ~unchanged after equilibration).
                    double T_eq = std::sqrt(std::sqrt(
                        (avg_flux + planet.Qb) / (planet.emissivity * ptm::kStefanBoltzmann)));
                    for (size_t layer = 0; layer < n_layers; ++layer)
                        T_initial[layer * batch_n + f] = static_cast<float>(T_eq);
                } else {
                    // Equilibrium temperature from Stefan-Boltzmann
                    double T_eq = std::sqrt(std::sqrt(avg_flux / (planet.emissivity * ptm::kStefanBoltzmann)));
                    if (T_eq < 40.0) T_eq = 100.0;  // Minimum reasonable temperature
                    if (T_eq > 400.0) T_eq = 400.0;

                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        size_t idx = layer * batch_n + f;
                        // Linear interpolation between surface and deep temperature
                        double depth_frac = z_grid[layer] / z_grid[n_layers - 1];
                        T_initial[idx] = static_cast<float>(T_eq * (1.0 - depth_frac) + T_deep_val * depth_frac);
                    }
                }
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

                    (void)flux_idx;
                    double absorbed = absorbed_at(global_f, t_start + t);
                    // Deep radiation-coupling: equilibration-phase indirect flux,
                    // selected on the ABSOLUTE step index. See the sibling
                    // prepare_flux above for why the test is not on the shape of
                    // the call.
                    if (!eq_indirect_flux.empty()
                            && (t_start + t) < equilibration_steps
                            && global_f < eq_indirect_flux.size()) {
                        absorbed += eq_indirect_flux[global_f];
                    }
                    flux[t * batch_n + f] = static_cast<float>(absorbed);
                }
            }
            return flux;
        };

        // Number of equilibration cycles (each replays the flux sequence).
        // eq_passes = passes over the array; solvePersistentMultipleCycles
        // runs n_equil_cycles + 1, so hand it eq_passes - 1. Default = 1 pass
        // (pre-2026-08-11 it was 2, which doubled every GPU run's spin-up).
        int equil_cycles = eq_passes - 1;
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
                // Report PASSES (equil_cycles + 1) — the number the kernel
                // actually runs and logs. The old banner printed equil_cycles
                // and so understated the eq phase by one full pass.
                std::cout << "[GPU Thermal] Phase 1: " << (equil_cycles + 1)
                          << " eq passes × " << equilibration_steps
                          << " steps (dt=" << std::fixed << std::setprecision(1)
                          << eq_dt << "s)" << std::endl;
            }
            auto flux_eq = prepare_flux(0, equilibration_steps);
            // Diagnostic dump of the equilibration forcing (default OFF); see
            // the multi-GPU copy above. Abandons the batch before any dispatch.
            if (eq_lun_dump.enabled) {
                eqlundump::accumulateBatch(eq_lun_dump, flux_eq, direct_solar_values,
                                           batch_start, batch_n,
                                           equilibration_steps, total_time_steps);
                continue;
            }
            std::vector<float> eq_output_discard;
            // Pass-stride RRE. Batch-local Stats: a shared accumulator would be
            // a data race on every counter under the multi-GPU pool.
            ptm::eqextrap::Stats batch_extrap_stats;
            solver.solvePersistentMultipleCycles(flux_eq, eq_output_discard,
                                                 equilibration_steps, eq_dt, equil_cycles,
                                                 nullptr, 0, -1,
                                                 &eq_extrap, &batch_extrap_stats);
            merge_extrap_stats(batch_extrap_stats);

            // Snapshot the post-eq T(z) for the rad-coupling re-thermal hand-off
            // (same purpose as the multi-GPU run_batch_on_solver path). This
            // single-GPU inline loop previously had NO snapshot, so on
            // single-GPU runs assembled_post_eq_T stayed all-zero and every
            // re-thermal'd face started from a 0 K column — masked by the old
            // [3,500] K clamp (the "3 K PSR" artifact), exposed as 0 K->runaway
            // divergence once the clamp was removed. downloadAllTemps packs with
            // stride = activeFaceCount(), so use that (not batch_n).
            if (post_eq_T_out) {
                std::vector<float> batch_T_post_eq;
                try {
                    solver.downloadAllTemps(batch_T_post_eq);
                } catch (const std::exception& e) {
                    std::cerr << "[GPU Thermal] WARNING: downloadAllTemps (post-eq) failed: "
                              << e.what() << "\n";
                    batch_T_post_eq.clear();
                }
                const size_t active = solver.activeFaceCount();
                if (active >= batch_n && n_layers > 0 &&
                    batch_T_post_eq.size() == active * n_layers) {
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        const float* src = batch_T_post_eq.data() + layer * active;
                        float*       dst = assembled_post_eq_T.data() + layer * n_faces + batch_start;
                        std::copy(src, src + batch_n, dst);
                    }
                } else {
                    std::cerr << "[GPU Thermal] WARNING: post-eq snapshot NOT captured"
                                 " (download size " << batch_T_post_eq.size()
                              << ", active=" << active << ", batch_n=" << batch_n
                              << ", n_layers=" << n_layers << "); rad-coupling"
                                 " re-thermal would start from a zero column\n";
                }
            }

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
            // ---- Single-phase (legacy dt, and EVERY warm-start chunk) ----
            // Accumulator runs only on the FINAL cycle, and only for relative
            // timesteps in the output portion of the trajectory:
            //   [equilibration_steps, total_time_steps).
            // See the identical block in the multi-GPU worker above for why a
            // warm start uses 0 pre-cycles and why the snapshot is taken here.
            const int sp_equil_cycles = warm_start_run ? 0 : equil_cycles;
            if (GPU_DEBUG_OUTPUT || batch_idx == 0) {
                std::cout << "[GPU Thermal] Running " << sp_equil_cycles
                          << " equilibration cycles + 1 output cycle"
                          << (warm_start_run
                                  ? " (warm start: window integrated ONCE)" : "")
                          << "..." << std::endl;
            }
            auto flux_all = prepare_flux(0, total_time_steps);
            if (sp_equil_cycles > 0) {
                std::vector<float> sp_discard;
                solver.solvePersistentMultipleCycles(flux_all, sp_discard,
                                                     total_time_steps, dt,
                                                     sp_equil_cycles - 1);
            }
            if (post_eq_T_out) {
                std::vector<float> batch_T_pre_out;
                try {
                    solver.downloadAllTemps(batch_T_pre_out);
                } catch (const std::exception& e) {
                    std::cerr << "[GPU Thermal] WARNING: downloadAllTemps (single-phase "
                                 "pre-output) failed: " << e.what() << "\n";
                    batch_T_pre_out.clear();
                }
                const size_t active = solver.activeFaceCount();
                if (active >= batch_n && n_layers > 0 &&
                    batch_T_pre_out.size() == active * n_layers) {
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        const float* src = batch_T_pre_out.data() + layer * active;
                        float*       dst = assembled_post_eq_T.data() + layer * n_faces + batch_start;
                        std::copy(src, src + batch_n, dst);
                    }
                } else {
                    std::cerr << "[GPU Thermal] WARNING: single-phase pre-output snapshot"
                                 " NOT captured (download size " << batch_T_pre_out.size()
                              << ", active=" << active << ", batch_n=" << batch_n
                              << ", n_layers=" << n_layers << "); rad-coupling"
                                 " re-thermal would start from a zero column\n";
                }
            }
            solver.solvePersistentMultipleCycles(flux_all, output_temps,
                                                 total_time_steps, dt, 0,
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

    // Equilibration forcing dump: every batch reduced its own eq flux array and
    // returned before dispatching a kernel, so nothing downstream of here has
    // valid state. Write and stop -- continuing would emit statistics from
    // uninitialized temperatures.
    if (eq_lun_dump.enabled) {
        const bool ok = eqlundump::write(eq_lun_dump, mesh->latitudes,
                                         mesh->longitudes,
                                         config.getProperty("planet", "Moon"));
        std::cout << "[eq-lun-dump] NO THERMAL SOLVE WAS RUN; exiting with "
                  << (ok ? 0 : 1) << std::endl;
        std::cout.flush();
        std::exit(ok ? 0 : 1);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total);
    std::cout << "[GPU Thermal] Completed " << n_faces << " faces x " << total_time_steps
              << " timesteps in " << total_duration.count() << " seconds" << std::endl;

    // ---- eq-extrapolation banner --------------------------------------
    // Printed unconditionally whenever the knob was REQUESTED, including when
    // it could not be honored. This line IS the guard-rate measurement: it is
    // the first time a PTM3D run has reported how converged its own deep
    // column is, per face, without running a second simulation to find out.
    if (eq_extrap_requested) {
        std::ios_base::fmtflags bf(std::cout.flags());
        std::streamsize bp = std::cout.precision();
        const double med = ptm::eqextrap::percentile(eq_extrap_stats.ehat, 0.50);
        const double p95 = ptm::eqextrap::percentile(eq_extrap_stats.ehat, 0.95);
        // Five numbers. Trimmed deliberately: the earlier banner reported
        // counters for guards that never fired and a diagnostic from a theory
        // the measurements refuted.
        //   extrapolated  = faces where the extrapolation actually did work
        //                   (certified count minus the vacuous converged windows)
        //   declined      = faces the certificate could not certify at tol_K;
        //                   their state still ships, it is a contraction step
        //   near-degenerate = REPORTED ONLY, never acted on: windows where the
        //                   seed weight could displace a converged layer by more
        //                   than tol_K (|gamma_0| * max|S_0 - S_last| > tol_K)
        const long long extrapolated =
            eq_extrap_stats.certified - eq_extrap_stats.converged_window;
        const double held_frac = eq_extrap_stats.layers_total > 0
            ? static_cast<double>(eq_extrap_stats.layers_held)
                  / static_cast<double>(eq_extrap_stats.layers_total)
            : 0.0;
        std::cout << "[eq-extrap] extrapolated " << extrapolated
                  << " | declined "        << eq_extrap_stats.cert_refused
                  << " | p95 e " << std::fixed << std::setprecision(4) << p95
                  << " K | held_fraction " << std::setprecision(6) << held_frac
                  << " | near_degenerate " << eq_extrap_stats.near_degenerate
                  << std::endl;
        const double q50 = ptm::eqextrap::percentile(eq_extrap_stats.d_held_post, 0.50);
        const double q95 = ptm::eqextrap::percentile(eq_extrap_stats.d_held_post, 0.95);
        const double qmx = ptm::eqextrap::percentile(eq_extrap_stats.d_held_post, 1.00);
        std::cout << "[eq-extrap] post-polish |Phi(T*)-S_last| on HELD layers: p50 "
                  << std::scientific << std::setprecision(3) << q50
                  << " | p95 " << q95 << " | max " << qmx
                  << " K (bar " << ptm::eqextrap::kHeldPostTol << " K)" << std::endl;
        if (eq_extrap_stats.d_held_max > 0.0) {
            std::cerr << "[eq-extrap] ERROR: held layers were modified -- the Level-2 guard "
                         "did not execute as claimed.\n";
        }
        if (qmx > ptm::eqextrap::kHeldPostTol) {
            std::cerr << "[eq-extrap] WARNING: post-polish deviation on a held layer exceeds "
                      << ptm::eqextrap::kHeldPostTol << " K; the delivered state moved a layer "
                         "the guard declared converged.\n";
        }
        if (eq_extrap_stats.budget_exhausted > 0) {
            std::cout << "[eq-extrap] budget_exhausted " << eq_extrap_stats.budget_exhausted
                      << " (too few passes for order_p; ran plain)" << std::endl;
        }
        if (eq_extrap_stats.solve_failed > 0) {
            std::cout << "[eq-extrap] solve_failed " << eq_extrap_stats.solve_failed
                      << " (non-finite lstsq; fell back to the plain state)" << std::endl;
        }
        std::cout.flags(bf);
        std::cout.precision(bp);
    }

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

    // Hand the assembled post-eq T(z) profile to the caller (rad-coupling
    // iter loop). Safety net: a face with an all-zero column was never
    // equilibrated, and the rad-coupling re-thermal would start it from 0 K
    // and ramp it to a runaway (the bug that the missing single-GPU snapshot
    // and the [3,500] K clamp together hid). Warn loudly if any survive so
    // this class of bug can never silently corrupt a coupled run again.
    if (post_eq_T_out) {
        // The `two_phase` gate that used to wrap this check is REMOVED: the
        // single-phase branch never filled assembled_post_eq_T at all, so the
        // one configuration where every column was zero was also the one
        // configuration the tripwire declined to inspect. That is how a
        // warm-start chunk shipped a coupled run whose re-thermal started from
        // 0 K (BUGS.md PTM-78).
        size_t n_zero = 0;
        for (size_t fc = 0; fc < n_faces; ++fc) {
            bool all_zero = true;
            for (size_t l = 0; l < n_layers; ++l) {
                if (assembled_post_eq_T[l * n_faces + fc] != 0.0f) { all_zero = false; break; }
            }
            if (all_zero) ++n_zero;
        }
        if (n_zero > 0) {
            std::cerr << "[GPU Thermal] WARNING: " << n_zero << "/" << n_faces
                      << " faces have an all-zero post-eq column; rad-coupling"
                         " re-thermal of these faces would diverge"
                      << (two_phase ? "" : " (single-phase path)") << "\n";
        }
        *post_eq_T_out = assembled_post_eq_T;
    }

    // Shared-seed guards. Physics statements, reported rather than swallowed.
    if (gpu_seed_floor_faces > 0) {
        std::cerr << "[GPU Thermal] WARNING: " << gpu_seed_floor_faces << " of "
                  << n_faces << " faces seeded at the " << thermal::ABSOLUTE_ZERO_K
                  << " K CMB floor -- Qb <= 0 AND zero mean absorbed flux, so "
                     "the steady-state balance has no positive root.\n";
    }
    if (gpu_seed_hot_faces > 0) {
        std::cerr << "[GPU Thermal] WARNING: " << gpu_seed_hot_faces << " of "
                  << n_faces << " faces have a cycle-mean radiative-equilibrium "
                     "seed above " << thermal::MAX_PHYSICAL_TEMP_K
                  << " K, the CPU solver's hard clamp. The seed is physical; a "
                     "CPU run of the same case would be truncated there.\n";
    }
    thermal::EqSeedRecorder::instance().write("gpu-persistent");

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
    const std::vector<double>& direct_solar_values,
    const std::vector<double>& primary_reflected_values,
    const std::vector<double>& primary_ir_values,
    const std::vector<std::vector<double>>& indirect_flux_series,
    const std::vector<float>& post_eq_T_state,
    const std::vector<double>& times,
    int total_time_steps,
    int equilibration_steps,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps,
    const std::vector<double>& face_albedos,
    int gpu_devices,
    const std::vector<int>& gpu_device_ids,
    int gpu_inter_batch_pause_ms,
    void* shared_thermal_pool,
    std::vector<float>* final_T_inout,
    std::vector<uint8_t>* final_T_touched)
{
    using namespace gpu::opencl;

    const size_t n_selected = face_indices.size();
    if (n_selected == 0) {
        return true;  // Nothing to do
    }

    // Get thermal model parameters
    thermal::Planet planet(config.getProperty("planet", "Moon"));
    applyConfiguredThermo(planet, config);   // generic-body props + user scalar overrides (PTM-53)
    thermal::Profile reference_profile(planet, 0.0, config, false);
    const auto& z_grid = reference_profile.z();
    const size_t n_layers = z_grid.size();
    // No seed banner here: this path never seeds. It is warm-started from
    // post_eq_T_state (uploadAllTemps), so the spin-up seed does not reach it.
    const size_t n_faces_total = mesh->faces.size();

    // Directional-hemispheric (Hapke) albedo gate (re-read from config; this
    // path rebuilds its own Planet). Kept consistent with iter-0's flux build
    // so the rad-coupling re-thermal uses the SAME absorbed-solar composition.
    // Default OFF → byte-identical Hayne-only path.
    const bool   use_hapke = config.getPropertyBool("use_hapke_albedo", false)
                             && config.getPropertyDouble("hapke_albedo_w", -1.0) >= 0.0;
    // Per-face albedo scaling: the Hapke single-scattering w is derived per
    // face from face_albedos[global_f] (hapkeWFromA0) at each call site below,
    // so the scalar body-level hapke_w is no longer read here.

    // Two-phase detection (mirrors iter-0 setup at lines 3199-3217 of
    // runThermalModelsForAllFaces_GPU_Persistent). When eq_dt != out_dt
    // we run two solver calls: equilibration cycles at eq_dt, then one
    // output cycle at out_dt — same as iter-0.
    const int output_steps = total_time_steps - equilibration_steps;
    float eq_dt = 0.0f;
    float out_dt = 0.0f;
    if (equilibration_steps > 1) {
        eq_dt = static_cast<float>(times[1] - times[0]);
    }
    if (output_steps > 1 && equilibration_steps < total_time_steps) {
        out_dt = static_cast<float>(times[equilibration_steps + 1] - times[equilibration_steps]);
    } else if (equilibration_steps > 1) {
        out_dt = eq_dt;
    }
    if (equilibration_steps <= 0) eq_dt = out_dt;
    const bool two_phase = (equilibration_steps > 0 && output_steps > 0 &&
                            std::abs(eq_dt - out_dt) > 1e-3f);
    const float dt_single = two_phase
        ? eq_dt
        : (total_time_steps > 1 ? static_cast<float>(times[1] - times[0]) : 0.0f);
    if (two_phase) {
        std::cout << "  [Re-thermal] Post-eq state-handoff: 1 output cycle x "
                  << output_steps << " steps at out_dt="
                  << std::fixed << std::setprecision(1) << out_dt
                  << "s, +indirect_flux" << std::endl;
    } else {
        std::cout << "  [Re-thermal] Post-eq state-handoff: 1 cycle x "
                  << total_time_steps << " steps at dt="
                  << std::fixed << std::setprecision(1) << dt_single
                  << "s, +indirect_flux" << std::endl;
    }
    // Per-face albedo scaling: the caller supplies per-face A0 (face_albedos,
    // global-indexed) so the re-thermal composes Hayne / derives Hapke w on each
    // face's own albedo. Warn LOUDLY rather than silently body-averaging if it is
    // missing/short (faces beyond it fall back to planet.albedo per face below).
    if (face_albedos.size() < n_faces_total) {
        std::cerr << "[GPU Thermal] WARNING: re-thermal face_albedos size "
                  << face_albedos.size() << " < n_faces " << n_faces_total
                  << "; faces beyond it fall back to the body-average albedo\n";
    }

    // GPU memory constraint
    const size_t bytes_per_face = n_layers * 50 + total_time_steps * 8 * 2;
    const size_t gpu_memory_limit = 6ULL * 1024 * 1024 * 1024;
    const size_t max_batch_size = std::min(n_selected, gpu_memory_limit / bytes_per_face);
    size_t BATCH_SIZE = std::min(max_batch_size, static_cast<size_t>(50000));
    // Re-equilibration with indirect flux (PTM_EQ_WITH_INDIRECT) folds long
    // multi-cycle equilibration kernels into each batch. At the 50000-face
    // default that blows past the macOS+AMD GPU command-buffer timeout
    // (gpusKillClientExt). Cap the batch to iter-0's proven-safe width so each
    // 750-step eq kernel matches the all-faces path's regime.
    if (const char* e = std::getenv("PTM_EQ_WITH_INDIRECT");
        e && std::atoi(e) != 0) {
        size_t cap = 4000;
        if (const char* b = std::getenv("PTM_EQ_RETHERMAL_BATCH")) {
            const int v = std::atoi(b);
            if (v > 0) cap = static_cast<size_t>(v);
        }
        BATCH_SIZE = std::min(BATCH_SIZE, cap);
    }

    // Phase R (auto-calibrate): ensure a representative dispatch ceiling is in
    // force for the selective re-thermal solve too. In a coupled run the main
    // solve above has already calibrated this grid, so the grid-keyed idempotency
    // guard makes this a cheap no-op (no redundant probe); a standalone re-thermal
    // self-calibrates. Numerically inert (batching only).
    ensureGpuCalibrated(config, planet, reference_profile, BATCH_SIZE,
                        static_cast<double>(std::max(eq_dt, out_dt)));

    // Get reference material properties
    const auto& rho_ref = reference_profile.rho();
    const auto& kc_ref = reference_profile.kc();
    const auto& T_ref = reference_profile.T();
    double T_deep_val = T_ref.back();
    double R350_val = reference_profile.R350_val;

    // Process faces in batches
    size_t n_batches = (n_selected + BATCH_SIZE - 1) / BATCH_SIZE;

    // Per-batch worker: encapsulates the upload + equilibration + output-phase
    // logic that was previously inline in the for-loop. Pulled out so it can
    // be invoked from either the legacy single-GPU loop or the multi-GPU
    // pool's submit() callback.
    auto run_batch_on_solver = [&](ThermalOpenCLSolver& solver,
                                   size_t batch_idx,
                                   size_t batch_start,
                                   size_t batch_n) -> bool {
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
            static_cast<float>(ptm::kStefanBoltzmann),
            static_cast<int>(config.conductivity_model)  // Stefan-Boltzmann
        );

        // -----------------------------------------------------------------
        // Upload iter-0's post-eq T(z) for this batch's faces, then run
        // ONLY the output cycle with combined_flux = iter-0 prepare_flux
        // composition + indirect_flux. No eq phase here — the eq phase
        // was iter-0's job, and re-running it from cold-start in each
        // rad iter has been shown empirically to diverge from iter-0's
        // converged state (May 2026 zero-indirect diagnostic). Reusing
        // iter-0's post-eq state guarantees the output cycle starts from
        // exactly the same point iter-0 did, so any temperature
        // difference comes solely from the indirect-flux delta.
        // -----------------------------------------------------------------
        if (post_eq_T_state.size() != n_layers * n_faces_total) {
            std::cerr << "[Re-thermal] FATAL: post_eq_T_state size "
                      << post_eq_T_state.size() << " != n_layers * n_faces "
                      << (n_layers * n_faces_total)
                      << " — cannot proceed without iter-0's post-eq state."
                      << std::endl;
            return false;
        }

        // Gather per-batch slice of iter-0's post-eq T(z): layer-major
        // SoA, indexed by global face id from face_indices.
        std::vector<float> T_chunked(batch_n * n_layers);
        for (size_t f = 0; f < batch_n; ++f) {
            const int face_idx = face_indices[batch_start + f];
            for (size_t layer = 0; layer < n_layers; ++layer) {
                T_chunked[layer * batch_n + f] =
                    post_eq_T_state[layer * n_faces_total
                                    + static_cast<size_t>(face_idx)];
            }
        }
        solver.uploadAllTemps(T_chunked);

        // Hayne (2017) parameters — same source as iter-0 (line 3597-3602).
        const double a_coef = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[0] : 0.0;
        const double b_coef = (planet.albedoCoef.size() >= 2) ? planet.albedoCoef[1] : 0.0;
        const double S_solar = planet.S;
        const bool   has_angular = (planet.albedoCoef.size() >= 2 && S_solar > 0.0);
        // Hayne OR Hapke gate (both need S>0 to recover cos(theta) from flux).
        const bool   use_incidence = has_angular || (use_hapke && S_solar > 0.0);

        // ----------------------------------------------------------------
        // Optional: re-equilibrate the deep column WITH the indirect flux
        // before the output cycle (env PTM_EQ_WITH_INDIRECT=1, cycles via
        // PTM_EQ_INDIRECT_CYCLES, default 2).
        //
        // The post_eq_T_state handed in by iter-0 was equilibrated with ZERO
        // indirect (the eq phase of runThermalModelsForAllFaces_GPU_Persistent
        // omits it — the indirect is only known after the radiosity). For a
        // permanently-shadowed floor whose surface flux is almost entirely
        // indirect IR, that means the deep column settles to the bare
        // direct+geothermal floor (~25 K) and then DRAINS the indirect-warmed
        // surface downward during the output cycle, capping the summer max.
        // Re-running a few equilibration cycles here with the indirect folded
        // into the eq-window forcing lets the column settle to the correct
        // (warmer) indirect-equilibrated state, starting from iter-0's already
        // reasonable column (no cold-start divergence).
        //
        // The eq window [start-eq_days, start) precedes the output window and
        // covers earlier orbits of the same (stationary) seasonal cycle, so
        // each eq timestep's indirect is phase-mapped from the output-window
        // indirect_flux_series at the matching annual phase.
        static const bool eq_with_indirect = []() {
            const char* e = std::getenv("PTM_EQ_WITH_INDIRECT");
            return e && std::atoi(e) != 0;
        }();
        static const int eq_indirect_cycles = []() {
            const char* e = std::getenv("PTM_EQ_INDIRECT_CYCLES");
            const int v = e ? std::atoi(e) : 2;
            return v > 0 ? v : 2;
        }();
        if (eq_with_indirect && two_phase && equilibration_steps > 0
                && !indirect_flux_series.empty()) {
            const double out_t0  = times[equilibration_steps];
            const double cyc_sec = (total_time_steps > equilibration_steps + 1)
                ? (times[total_time_steps - 1] - out_t0) : 0.0;
            std::vector<float> eq_flux(static_cast<size_t>(equilibration_steps) * batch_n);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < equilibration_steps; ++t) {
                // Phase-map this eq timestep onto an output-window timestep.
                int map_out = 0;
                if (cyc_sec > 0.0) {
                    double frac = (times[t] - out_t0) / cyc_sec;   // < 0 for eq steps
                    frac -= std::floor(frac);                      // wrap into [0,1)
                    map_out = std::min(output_steps - 1,
                                       std::max(0, static_cast<int>(
                                           std::lround(frac * (output_steps - 1)))));
                }
                for (size_t f = 0; f < batch_n; ++f) {
                    const int    global_f = face_indices[batch_start + f];
                    const size_t flux_idx = static_cast<size_t>(global_f)
                                          * total_time_steps + t;
                    const double base_A   = (static_cast<size_t>(global_f) < face_albedos.size()) ? face_albedos[global_f] : planet.albedo;
                    const double direct   = (flux_idx < direct_solar_values.size())
                                          ? direct_solar_values[flux_idx] : 0.0;
                    const double reflected = (flux_idx < primary_reflected_values.size())
                                          ? primary_reflected_values[flux_idx] : 0.0;
                    const double ir       = (flux_idx < primary_ir_values.size())
                                          ? primary_ir_values[flux_idx] : 0.0;
                    const double albedo = use_incidence
                        ? applyIncidenceAlbedo(base_A, direct,
                                               solarConstForStep(t, S_solar),
                                               a_coef, b_coef, use_hapke, hapkeWFromA0(base_A))
                        : base_A;
                    double absorbed = (1.0 - albedo) * direct
                                    + (1.0 - albedo) * reflected
                                    + planet.emissivity * ir;
                    const auto& irow = indirect_flux_series[global_f];
                    const int oidx = equilibration_steps + map_out;
                    if (oidx < static_cast<int>(irow.size())) absorbed += irow[oidx];
                    eq_flux[t * batch_n + f] = static_cast<float>(absorbed);
                }
            }
            // Run the cycles as SEPARATE single-cycle dispatches (each
            // equilibration_steps long, matching iter-0's safe kernel size)
            // with an inter-cycle pause, instead of one (cycles+1)*steps
            // mega-kernel that trips the macOS+AMD watchdog.
            for (int c = 0; c < eq_indirect_cycles; ++c) {
                std::vector<float> eq_discard;
                solver.solvePersistentMultipleCycles(eq_flux, eq_discard,
                                                     equilibration_steps, eq_dt,
                                                     /*n_equil_cycles=*/0);
                if (gpu_inter_batch_pause_ms > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(gpu_inter_batch_pause_ms));
                }
            }
            if (batch_idx == 0) {
                std::cout << "  [Re-thermal] Re-equilibrated WITH indirect flux ("
                          << eq_indirect_cycles << " cycle(s) x " << equilibration_steps
                          << " steps, phase-mapped) — PTM_EQ_WITH_INDIRECT" << std::endl;
            }
        }

        // Build output-window flux. Same composition as iter-0's
        // prepare_flux (Hayne on direct/reflected, ε on primary_ir)
        // plus indirect_flux_series for this iter.
        const int t_offset = two_phase ? equilibration_steps : 0;
        const int n_out    = two_phase ? output_steps : total_time_steps;
        const float dt_out = two_phase ? out_dt : dt_single;

        std::vector<float> flux_out(static_cast<size_t>(n_out) * batch_n);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < n_out; ++t) {
            for (size_t f = 0; f < batch_n; ++f) {
                const int    global_f = face_indices[batch_start + f];
                const size_t flux_idx = static_cast<size_t>(global_f)
                                      * total_time_steps + (t_offset + t);

                const double base_A   = (static_cast<size_t>(global_f) < face_albedos.size()) ? face_albedos[global_f] : planet.albedo;
                const double direct   = (flux_idx < direct_solar_values.size())
                                      ? direct_solar_values[flux_idx] : 0.0;
                const double reflected = (flux_idx < primary_reflected_values.size())
                                      ? primary_reflected_values[flux_idx] : 0.0;
                const double ir       = (flux_idx < primary_ir_values.size())
                                      ? primary_ir_values[flux_idx] : 0.0;

                const double albedo = use_incidence
                    ? applyIncidenceAlbedo(base_A, direct,
                                           solarConstForStep(t_offset + t, S_solar),
                                           a_coef, b_coef, use_hapke, hapkeWFromA0(base_A))
                    : base_A;

                double absorbed = (1.0 - albedo) * direct
                                + (1.0 - albedo) * reflected
                                + planet.emissivity * ir;
                if (!indirect_flux_series.empty()
                    && static_cast<size_t>(global_f) < indirect_flux_series.size()
                    && (t_offset + t) < static_cast<int>(indirect_flux_series[global_f].size())) {
                    absorbed += indirect_flux_series[global_f][t_offset + t];
                }
                flux_out[t * batch_n + f] = static_cast<float>(absorbed);
            }
        }

        // Single output cycle, no equilibration. The persistent kernel
        // takes the just-uploaded post-eq state, integrates forward
        // n_out timesteps at dt_out, and captures surface T per step.
        // TODO(ice-stability): rad-coupling re-thermal does not update
        // ice_acc->sum_E for selected faces; ice_depth.bin reflects
        // only iter-0's flux BC on the re-thermal'd faces.
        std::vector<float> output_temps;
        solver.solvePersistentMultipleCycles(flux_out, output_temps,
                                             n_out, dt_out,
                                             /*n_equil_cycles=*/0);

        // Hand the POST-COUPLING end-of-window column back to the caller so a
        // chunk checkpoint archives the coupled state, not iter-0's un-coupled
        // one (BUGS.md PTM-57). downloadAllTemps strides by activeFaceCount(),
        // which EXCEEDS batch_n whenever the shared re-thermal pool was
        // allocated wider than this batch — using batch_n as the stride there
        // reads the wrong faces, and gating on `size == batch_n*n_layers`
        // silently captures nothing. Use the active stride and say so on failure.
        if (final_T_inout && final_T_touched) {
            std::vector<float> batch_T_final;
            try {
                solver.downloadAllTemps(batch_T_final);
            } catch (const std::exception& e) {
                std::cerr << "[Re-thermal] WARNING: downloadAllTemps (post-coupling "
                             "state) failed: " << e.what() << "\n";
                batch_T_final.clear();
            }
            const size_t active = solver.activeFaceCount();
            if (active >= batch_n && n_layers > 0 &&
                batch_T_final.size() == active * n_layers) {
                for (size_t f = 0; f < batch_n; ++f) {
                    const size_t gf = static_cast<size_t>(face_indices[batch_start + f]);
                    for (size_t layer = 0; layer < n_layers; ++layer) {
                        (*final_T_inout)[layer * n_faces_total + gf] =
                            batch_T_final[layer * active + f];
                    }
                    (*final_T_touched)[gf] = 1;
                }
            } else {
                std::cerr << "[Re-thermal] WARNING: post-coupling state NOT captured for"
                             " batch " << batch_idx << " (download size "
                          << batch_T_final.size() << ", active=" << active
                          << ", batch_n=" << batch_n << ", n_layers=" << n_layers
                          << "); the chunk checkpoint keeps iter-0's un-coupled column"
                             " for these faces (BUGS.md PTM-57)\n";
            }
        }

        // Copy output surface T into surface_temps at the output-window
        // slots. In two-phase mode, eq-window slots stay at 0 (matches
        // iter-0). In single-phase, the full range is written.
        //
        // RE-THERMAL WARMUP SKIP (PTM-54). This used to be a hard-coded 2:
        // the first 2 recorded steps of every re-thermal were LEFT at iter-0's
        // UN-COUPLED values, which is exactly the "first two frames of every
        // chunk are un-coupled" artifact. Its stated justification was a
        // "spurious cold, MIN_TEMP-clamped transient" — but the kernel
        // recomputes k/cp/kappa from T_local every substep and starts the
        // surface Newton solve from a physics guess (equilibrium T, or T1 at
        // night), so it carries NO state across uploadAllTemps that could need
        // 2 steps to settle. The transient it was masking was the ZERO T(z)
        // seed (post-eq snapshot never captured; see PTM-78) being clamped to
        // the old [3,500] K floor — a seed bug, symptomatically patched here.
        // With a correct seed there is nothing to skip. Default 0; set
        // PTM_RETHERMAL_WARMUP_SKIP=2 to restore the legacy mask.
        static const int kReThermalWarmupSkip = []() {
            const char* e = std::getenv("PTM_RETHERMAL_WARMUP_SKIP");
            const int v = e ? std::atoi(e) : 0;
            return v > 0 ? v : 0;
        }();
        if (batch_idx == 0 && kReThermalWarmupSkip > 0) {
            std::cout << "  [Re-thermal] PTM_RETHERMAL_WARMUP_SKIP="
                      << kReThermalWarmupSkip << ": the first "
                      << kReThermalWarmupSkip << " output step(s) keep iter-0's"
                         " UN-COUPLED temperature (legacy PTM-54 behavior)"
                      << std::endl;
        }
        for (size_t f = 0; f < batch_n; ++f) {
            const int face_idx = face_indices[batch_start + f];
            for (int t = kReThermalWarmupSkip; t < n_out; ++t) {
                surface_temps[face_idx][t_offset + t] =
                    static_cast<double>(output_temps[t * batch_n + f]);
            }
        }

        return true;
    };

    // Multi-GPU dispatch (opt-in): mirrors the all-faces path. When enabled,
    // the per-batch lambda above runs concurrently across N devices via the
    // ThermalGPUDevicePool (~3x speedup measured on the 4-die W6800X Duo).
    const bool use_multi_gpu =
        (gpu_devices != 1) || !gpu_device_ids.empty();
    if (use_multi_gpu) {
        auto all_gpus = OpenCLContext::enumerateGPUs(/*only_available=*/true);
        std::vector<OpenCLDeviceInfo> selected;
        if (!gpu_device_ids.empty()) {
            for (int id : gpu_device_ids) {
                if (id < 0 || static_cast<size_t>(id) >= all_gpus.size()) {
                    std::cerr << "[GPU Thermal] Invalid gpu_device_id " << id
                              << " (only " << all_gpus.size()
                              << " GPUs enumerated)" << std::endl;
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
            std::cerr << "[GPU Thermal] No GPUs selected for selected-faces "
                         "multi-GPU dispatch" << std::endl;
            return false;
        }

        // Two paths: caller-supplied long-lived pool (preferred — eliminates
        // per-iter context churn during radiation-thermal coupling) or
        // function-local pool (legacy when caller passes nullptr).
        ThermalGPUDevicePool* pool_ptr = nullptr;
        std::unique_ptr<ThermalGPUDevicePool> local_pool;

        if (shared_thermal_pool != nullptr) {
            pool_ptr = static_cast<ThermalGPUDevicePool*>(shared_thermal_pool);
            std::cout << "  [GPU Thermal Pool] Selected-faces multi-GPU dispatch "
                         "(shared pool, " << pool_ptr->numDevices()
                      << " device(s))" << std::endl;
            if (BATCH_SIZE > pool_ptr->capacity()) {
                std::cerr << "  [GPU Thermal Pool] Shared pool capacity "
                          << pool_ptr->capacity() << " < required BATCH_SIZE "
                          << BATCH_SIZE << "; falling back to local pool"
                          << std::endl;
                pool_ptr = nullptr;
            }
        }

        if (pool_ptr == nullptr) {
            std::cout << "  [GPU Thermal Pool] Selected-faces multi-GPU dispatch "
                         "across " << selected.size() << " device(s)" << std::endl;

            ThermalOpenCLConfig opencl_config;
            opencl_config.verbose = false;
            opencl_config.work_group_size = 64;

            ThermalGPUPoolConfig pool_cfg;
            pool_cfg.devices = selected;
            pool_cfg.max_n_faces = BATCH_SIZE;
            pool_cfg.n_layers = n_layers;
            pool_cfg.solver_config = opencl_config;
            pool_cfg.inter_batch_pause_ms = gpu_inter_batch_pause_ms;
            pool_cfg.verbose = false;
            try {
                local_pool = std::make_unique<ThermalGPUDevicePool>(std::move(pool_cfg));
            } catch (const std::exception& e) {
                std::cerr << "  [GPU Thermal Pool] Construction failed: "
                          << e.what() << std::endl;
                return false;
            }
            pool_ptr = local_pool.get();
        }

        std::vector<std::future<bool>> futures;
        futures.reserve(n_batches);
        for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
            const size_t batch_start = batch_idx * BATCH_SIZE;
            const size_t batch_end   = std::min(batch_start + BATCH_SIZE, n_selected);
            const size_t batch_n     = batch_end - batch_start;
            futures.push_back(pool_ptr->submit(batch_n,
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
            std::cout << "\r  [GPU Thermal Pool] " << completed << "/" << n_batches
                      << " re-thermal batches done" << std::flush;
            // Heartbeat every batch — the re-thermal phase is exactly where
            // the macOS+AMD driver wedged. Pinpointing the last completed
            // batch index after a hang lets us correlate with GPU temps
            // and decide whether to drop concurrency further.
            writeHeartbeat("re_thermal batch " + std::to_string(completed)
                           + "/" + std::to_string(n_batches)
                           + (ok ? " ok" : " FAILED"));
            all_ok = ok && all_ok;
        }
        std::cout << std::endl;
        return all_ok;
    }

    // Legacy single-GPU serial-batches path (byte-identical to pre-multi-GPU
    // behavior when gpu_devices == 1).
    for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
        size_t batch_start = batch_idx * BATCH_SIZE;
        size_t batch_end = std::min(batch_start + BATCH_SIZE, n_selected);
        size_t batch_n = batch_end - batch_start;

        // Initialize solver for this batch (single-device default path)
        ThermalOpenCLConfig opencl_config;
        opencl_config.verbose = false;
        opencl_config.work_group_size = 64;

        ThermalOpenCLSolver solver(batch_n, n_layers, opencl_config);
        if (!solver.initialize()) {
            std::cerr << "[GPU Thermal] Failed to initialize OpenCL solver for selected faces" << std::endl;
            return false;
        }

        if (!run_batch_on_solver(solver, batch_idx, batch_start, batch_n)) {
            return false;
        }
        // Inter-batch pause lets the macOS+AMD driver state settle between
        // consecutive batches on the same device — important when each batch
        // now carries multi-cycle eq kernels (PTM_EQ_WITH_INDIRECT).
        if (gpu_inter_batch_pause_ms > 0 && batch_idx + 1 < n_batches) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(gpu_inter_batch_pause_ms));
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
    const std::vector<double>*,
    const std::vector<double>*,
    ice::IceStabilityAccumulator*,
    const std::string&,
    const std::string&,
    std::vector<float>*,
    int,
    const std::vector<int>&,
    size_t,
    int,
    const std::vector<double>&)
{
    std::cerr << "[GPU Thermal] Error: OpenCL support not compiled in. "
              << "Rebuild with -DENABLE_GPU=ON -DGPU_BACKEND=OPENCL" << std::endl;
    return false;
}

bool runThermalForSelectedFaces_GPU(
    const std::shared_ptr<Mesh>&,
    const std::vector<int>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<float>&,
    const std::vector<double>&,
    int,
    int,
    const thermal::Config&,
    std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int,
    const std::vector<int>&,
    int,
    void*,
    std::vector<float>*,
    std::vector<uint8_t>*)
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

// Helper function to adjust start time
std::string calculateAdjustedStartTime(const std::string& iso_time, double days_offset) {
    // Parse the ISO time string to get seconds since J2000
    double et;
    str2et_c(iso_time.c_str(), &et);

    // Fail loud rather than propagate an uninitialized epoch. If the leapseconds
    // kernel (or the whole metakernel) never loaded, str2et_c is a silent no-op
    // under erract=RETURN and `et` is left uninitialized; printing/using it would
    // produce a garbled start time and an epoch driven by stack garbage instead
    // of the requested start_time (see BUGS.md PTM-27).
    if (failed_c()) {
        SpiceChar short_msg[81] = {0};
        getmsg_c("SHORT", 81, short_msg);
        reset_c();
        std::cerr << "\nFATAL: SPICE str2et_c failed for start_time '" << iso_time << "': "
                  << short_msg << "\n"
                  << "  The SPICE leapseconds/ephemeris kernels are not loaded, so the absolute\n"
                  << "  epoch cannot be resolved. Populate the kernel directory with:\n"
                  << "      python3 scripts/fetch_kernels.py\n"
                  << "  (see INSTALL.md, section \"SPICE kernel data\")." << std::endl;
        std::exit(1);
    }

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
    double H_override,
    long long seed_record_face,
    std::atomic<size_t>* seed_floor_faces,
    std::atomic<size_t>* seed_hot_faces);

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

// ============================================================================
// PTM-37: state the mesh length unit, and REFUSE to run when it is a guess.
//
// PTM3D mesh units are not uniform and nothing on Mesh records which it has:
// global icosphere meshes are built in KILOMETERS (mesh_utils.cpp scales
// vertices to body_radius_km_) while DEM/site meshes are in METERS. Three
// separate places in illumination_model.cpp re-derive the unit with the same
// ad-hoc test, `|face_center| > 10000` -> meters (kmToNativeScale :74,
// calculatePrimaryIRBatch :4070, calculatePrimaryReflectedBatch :4218). That
// heuristic is a 1000x error waiting to happen: getting it backwards is what
// PTM-36 was (Jupiter's shadow shrank 1000x and every Europa eclipse vanished).
//
// This does not fix the substrate -- Mesh still carries no unit -- but it
// makes the guess VISIBLE and refuses the cases where it is unsafe:
//
//   1. The mesh STRADDLES the 10,000 discriminator (some faces above, some
//      below). Every consumer classifies from ONE face, so different call
//      sites can reach opposite conclusions about the same mesh. Fatal.
//   2. The body radius CONTRADICTS the heuristic. This is the gas-giant
//      landmine named in the ledger: a Jupiter-centered mesh in km has
//      |face_center| ~ 71,492 > 10,000, so the heuristic calls it meters and
//      is wrong by 1000x. Fatal.
//   3. The median sits in the ambiguous band around the discriminator
//      (10,000/50 .. 10,000*50) and NO body radius confirms it. Fatal.
//
// A body radius that CONFIRMS the heuristic resolves case 3: the Moon and
// Europa icospheres (1737.4 km, 1560.8 km) both land in the ambiguous band by
// magnitude alone, and are only unambiguous because the radius agrees to
// ~0.1%. That confirmation is why this guard can be strict without refusing
// the two most common meshes in the project.
// ============================================================================
namespace {

struct MeshUnitReport {
    bool fatal = false;
    bool meters = false;         // heuristic verdict (the one consumers use)
    double median = 0.0;
    double lo = 0.0, hi = 0.0;
    std::string message;         // reason, when fatal
};

MeshUnitReport inspectMeshLengthUnit(const std::shared_ptr<ptm::Mesh>& mesh,
                                     double body_radius_km) {
    // The discriminator every consumer already uses, and the margin within
    // which we refuse to trust it unaided.
    constexpr double kDiscriminator = 10000.0;
    constexpr double kAmbiguousFactor = 50.0;
    // A body-centered mesh's |face_center| is the body radius to within
    // topography and figure flattening; a factor of 2 is generous for both.
    constexpr double kRadiusTolLo = 0.5, kRadiusTolHi = 2.0;

    MeshUnitReport r;
    const auto& fc = mesh->face_centers;
    if (fc.empty()) return r;   // nothing to say; validateMesh already objected

    std::vector<double> mag;
    mag.reserve(fc.size());
    for (const auto& c : fc)
        mag.push_back(std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]));
    const size_t mid = mag.size() / 2;
    std::nth_element(mag.begin(), mag.begin() + mid, mag.end());
    r.median = mag[mid];
    r.lo = *std::min_element(mag.begin(), mag.end());
    r.hi = *std::max_element(mag.begin(), mag.end());
    r.meters = (r.median > kDiscriminator);

    // Median |vertex|, as an independent witness of the same geometry. These
    // two must agree: face centers ARE vertex averages. They disagree when
    // something rescaled one array and not the other -- see PTM-65, where
    // mesh.scale_factor scales the vertices and leaves face_centers /
    // face_areas / face_normals at the loader's values, then prints "Applied
    // scale factor". Checking it here turns a silent 1000x geometry error
    // into a refusal.
    double vmedian = 0.0;
    if (!mesh->vertices.empty()) {
        std::vector<double> vmag;
        vmag.reserve(mesh->vertices.size());
        for (const auto& v : mesh->vertices)
            vmag.push_back(std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));
        const size_t vmid = vmag.size() / 2;
        std::nth_element(vmag.begin(), vmag.begin() + vmid, vmag.end());
        vmedian = vmag[vmid];
    }

    std::ostringstream unit_line;
    unit_line << "[mesh-units] inferred " << (r.meters ? "METERS" : "KILOMETERS")
              << ": median |face_center| = " << std::setprecision(10) << r.median
              << " (range " << r.lo << " .. " << r.hi << "), "
              << mesh->faces.size() << " faces";
    if (body_radius_km > 0.0)
        unit_line << ", R_body = " << body_radius_km << " km";
    else
        unit_line << ", R_body UNKNOWN";
    std::cout << unit_line.str() << std::endl;

    // -- 0. face centers vs vertices ---------------------------------------
    // Tolerance is deliberately loose: a triangle centroid sits INSIDE the
    // sphere its vertices lie on, so the ratio is coarseness-dependent --
    // measured on icospheres, 0.795 (res 0, 20 faces), 0.944 (res 1),
    // 0.985 (res 2), 0.999 (res 4). A tight gate would refuse coarse meshes.
    // [0.5, 2.0] cannot false-positive on any real surface mesh and still
    // catches every unit-scale (>=1000x) mismatch with three orders of
    // margin. It is NOT meant to catch a small scale slip.
    if (vmedian > 0.0 && r.median > 0.0) {
        const double ratio = r.median / vmedian;
        if (ratio < 0.5 || ratio > 2.0) {
            std::ostringstream m;
            m << "the mesh is INTERNALLY INCONSISTENT: median |face_center| = "
              << r.median << " but median |vertex| = " << vmedian
              << " (ratio " << ratio << "). Face centers are vertex averages, so "
                 "these cannot legitimately differ. Something rescaled one array "
                 "and not the other -- if mesh.scale_factor is set, that is PTM-65 "
                 "(the vertices are scaled, the face centers/areas/normals are not, "
                 "and the run prints 'Applied scale factor' anyway).";
            r.fatal = true; r.message = m.str();
            return r;
        }
    }

    // -- 1. straddle -------------------------------------------------------
    if (r.lo <= kDiscriminator && r.hi > kDiscriminator) {
        std::ostringstream m;
        m << "mesh face centers STRADDLE the " << kDiscriminator
          << " unit discriminator (min " << r.lo << ", max " << r.hi
          << "). Consumers classify the whole mesh from a single face, so "
             "different code paths would reach opposite conclusions about "
             "this one mesh.";
        r.fatal = true; r.message = m.str();
        return r;
    }

    // -- 2/3. body-radius cross-check --------------------------------------
    const bool have_R = (body_radius_km > 0.0);
    bool radius_says_km = false, radius_says_m = false;
    if (have_R) {
        const double ratio_km = r.median / body_radius_km;
        const double ratio_m  = (r.median / 1000.0) / body_radius_km;
        radius_says_km = (ratio_km >= kRadiusTolLo && ratio_km <= kRadiusTolHi);
        radius_says_m  = (ratio_m  >= kRadiusTolLo && ratio_m  <= kRadiusTolHi);
        if (radius_says_km || radius_says_m) {
            std::cout << "[mesh-units]   confirmed by R_body: |face_center| / R_body = "
                      << (radius_says_km ? ratio_km : ratio_m) << " assuming "
                      << (radius_says_km ? "KILOMETERS" : "METERS") << std::endl;
        }
    }

    if ((radius_says_km || radius_says_m) && (radius_says_m != r.meters)) {
        std::ostringstream m;
        m << "the body radius CONTRADICTS the |face_center| > " << kDiscriminator
          << " heuristic. R_body = " << body_radius_km << " km and the median "
          << "|face_center| is " << r.median << ", which is "
          << (radius_says_km ? "KILOMETERS" : "METERS") << " -- but the heuristic "
          << "every consumer uses says " << (r.meters ? "METERS" : "KILOMETERS")
          << ". Running would scale SPICE positions and shadow radii by 1000x in "
             "the wrong direction (this is PTM-36).";
        r.fatal = true; r.message = m.str();
        return r;
    }

    const bool in_ambiguous_band =
        (r.median > kDiscriminator / kAmbiguousFactor) &&
        (r.median < kDiscriminator * kAmbiguousFactor);
    if (in_ambiguous_band && !(radius_says_km || radius_says_m)) {
        std::ostringstream m;
        m << "median |face_center| = " << r.median << " sits in the ambiguous band ("
          << (kDiscriminator / kAmbiguousFactor) << " .. " << (kDiscriminator * kAmbiguousFactor)
          << ") around the " << kDiscriminator << " unit discriminator, and ";
        if (have_R)
            m << "the body radius (" << body_radius_km << " km) does not confirm either "
                 "unit (|face_center|/R_body = " << (r.median / body_radius_km)
              << " as km, " << ((r.median / 1000.0) / body_radius_km) << " as m). ";
        else
            m << "no body radius is available to confirm it. ";
        m << "PTM3D cannot tell kilometers from meters here, and a wrong guess is a "
             "1000x error in every SPICE-derived length.";
        r.fatal = true; r.message = m.str();
        return r;
    }

    if (!(radius_says_km || radius_says_m)) {
        // Outside the band, so the magnitude alone is decisive -- but nothing
        // corroborates it. Say so rather than implying more confidence than
        // we have (a locally recentered patch would land here).
        std::cout << "[mesh-units]   WARNING: unit inferred from magnitude ALONE; "
                     "no body radius corroborates it";
        if (have_R)
            std::cout << " (R_body = " << body_radius_km << " km gives ratio "
                      << (r.meters ? (r.median / 1000.0) / body_radius_km
                                   : r.median / body_radius_km)
                      << ", outside " << kRadiusTolLo << ".." << kRadiusTolHi << ")";
        std::cout << ". If this mesh is not body-centered, check it." << std::endl;
    }
    return r;
}

}  // namespace

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
    std::cout << "  --equilibration-orbits N   INTEGRATED day/night cycles of spin-up (default: 1).\n";
    std::cout << "                             This is what the solver integrates, on every path.\n";
    std::cout << "  --equilibration-cycles N   Replays of the equilibration forcing block used to\n";
    std::cout << "                             reach it (GPU persistent solver only; default: 1).\n";
    std::cout << "                             The block is equilibration-orbits / N cycles long,\n";
    std::cout << "                             so the integrated total is unchanged and only the\n";
    std::cout << "                             flux RAM and illumination cost fall (by N).\n";
    std::cout << "                             Before 2026-08-11 the default was 2 and the total\n";
    std::cout << "                             was N x equilibration-orbits (BUGS.md PTM-146).\n";
    std::cout << "                             Env override: PTM_EQUILIBRATION_CYCLES.\n";
    std::cout << "  --equilibration-steps-per-day N   Explicit equilibration cadence (steps/Earth-day);\n";
    std::cout << "                             -1 (default) = auto (use --equilibration-samples-per-diurnal).\n";
    std::cout << "                             > 0 = explicit override for the equilibration segment.\n";
    std::cout << "  --equilibration-samples-per-diurnal N  Accuracy-based default:\n";
    std::cout << "                             target illumination samples per diurnal cycle on the\n";
    std::cout << "                             target body (24=<2 K accuracy, 48=<1 K, 12=marginal,\n";
    std::cout << "                             <=0=legacy/match output cadence).\n";
    std::cout << "  --thermal-inertia-map FILE Path to thermal inertia map file\n";
    std::cout << "  --thermal-inertia TI       Set the nominal SURFACE thermal inertia [SI:\n";
    std::cout << "                             J m^-2 K^-1 s^-1/2] on any body, built-in or generic.\n";
    std::cout << "                             TI = sqrt(ks * rhos * cp(273 K)): the densities and the\n";
    std::cout << "                             heat capacity are held at the body's values and the\n";
    std::cout << "                             conductivity carries the change, ks = TI^2/(rhos*cp);\n";
    std::cout << "                             kd is scaled by the SAME factor, so the kd/ks depth\n";
    std::cout << "                             contrast is preserved. This is the convention already\n";
    std::cout << "                             used by scripts/ema_campaign_driver.py, divval_sweep.py\n";
    std::cout << "                             and examples/quickstart_asteroid_diurnal_ti200_check.yaml.\n";
    std::cout << "                             cp(T) and chi are NOT altered, so on a body that has them\n";
    std::cout << "                             the delivered TI varies with temperature about TI; the\n";
    std::cout << "                             number is nominal, at the Hayne et al. (2017) 273 K\n";
    std::cout << "                             reference. Equivalent to setting ks_override/kd_override.\n";
    std::cout << "  --constant-thermal-inertia TI  The OTHER reading: TI exactly constant in temperature\n";
    std::cout << "                             AND depth. Derives cp = TI^2/(ks*rhos) and forces chi=0,\n";
    std::cout << "                             kd=ks, rhod=rhos. Same as thermophysical.\n";
    std::cout << "                             constant_thermal_inertia. Mutually exclusive with\n";
    std::cout << "                             --thermal-inertia.\n";
    std::cout << "  --albedo A0                Bond albedo A0 in [0,1), on any body. Same as\n";
    std::cout << "                             thermophysical.albedo_override. A per-face albedo MAP\n";
    std::cout << "                             out-ranks it for absorbed solar flux -- the run warns if\n";
    std::cout << "                             one is configured; set illumination.use_albedo_map: false\n";
    std::cout << "                             to sweep albedo directly. With incidence_angle_albedo the\n";
    std::cout << "                             angular model is re-anchored on the new A0.\n";
    std::cout << "  --no-statistics            Disable computation of statistics\n";
    std::cout << "  --integration-days DAYS    Integration window in days\n";
    std::cout << "  --integration-steps N      Integration steps override\n";
    std::cout << "  --integration-offset-days  Offset for integration window (days)\n";
    std::cout << "  --integration-offset-steps Offset for integration window (steps)\n";
    std::cout << "  --thermal-debug            Enable thermal debug output for equatorial face\n";
    std::cout << "  --thermal-debug-lat DEG    Max latitude for debug face selection (default: 10)\n";
    std::cout << "  --thermal-m N              Layers in upper skin depth (default: 10)\n";
    std::cout << "  --thermal-n N              Layer growth factor (default: 15)\n";
    std::cout << "  --thermal-b N              Skin depths to bottom layer (default: 15)\n";
    std::cout << "  --geothermal-heat-flow Q   Geothermal heat flow [W/m^2] (overrides planet default)\n";
    std::cout << "  --heat-flow-mask FILE      Face index file for elevated Qb (one index per line)\n";
    std::cout << "  --heat-flow-mask-value Q   Qb value [W/m^2] for masked faces (default: 1.0)\n\n";

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Radiation Exchange Options\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  --radiation-exchange       Enable inter-face radiation exchange\n";
    std::cout << "  --rad-hmatrix              Use H-matrix compression for large meshes\n";
    std::cout << "  --rad-use-gpu              Use GPU for radiosity solver (OpenCL)\n";
    std::cout << "  --rad-verbose              Print detailed radiation progress\n";
    std::cout << "  --rad-view-factor-file F   Save/load view factors from file\n";
    std::cout << "  --rad-coupling-iter N      Max coupling iterations (default: 10)\n";
    std::cout << "  --rad-coupling-temp-tol K  Temperature convergence threshold [K] (default: 0.5)\n";
    std::cout << "  --rad-coupling-flux-tol W  Flux change threshold [W/m2] for LEGACY mode (default: 0.1)\n";
    std::cout << "  --rad-coupling-face-frac F Face fraction still changing to stop (default: 0.01)\n";
    std::cout << "                                   LEGACY coupling mode only (see --rad-coupling-mode).\n";
    std::cout << "  --rad-coupling-mode MODE   PTM-89 chunk-decomposition independence:\n";
    std::cout << "                                   bounded (default): absolute selection threshold, no\n";
    std::cout << "                                     face-fraction exit -- a chunked run reproduces the\n";
    std::cout << "                                     single-shot field.\n";
    std::cout << "                                   fixed:   as bounded, and always runs coupling-iter\n";
    std::cout << "                                     sweeps (iteration count cannot depend on the window).\n";
    std::cout << "                                   legacy:  pre-2026-08-05 behavior, bit-for-bit. Use ONLY\n";
    std::cout << "                                     to reproduce a published run; it re-introduces the\n";
    std::cout << "                                     chunk dependence. Env: PTM_COUPLING_MODE.\n";
    std::cout << "  --rad-coupling-select-dt K Absolute predicted-ΔT selection threshold for bounded/fixed\n";
    std::cout << "                                   (default: 0 = every face carrying indirect flux). >0\n";
    std::cout << "                                   trades accuracy for speed and warns.\n";
    std::cout << "  --rad-coupling-select-mode MODE  Face-selection mode: 'physics' (default) or 'legacy'.\n";
    std::cout << "                                   physics: ΔT predictor (1−α)·ΔF/(4εσT³) > temp_tol·safety.\n";
    std::cout << "                                   legacy:  absolute |max indirect flux| > flux_tol.\n";
    std::cout << "  --rad-coupling-select-safety F   Safety factor on predicted-ΔT threshold (default: 0.5).\n";
    std::cout << "  --rad-use-svd              Use SVD compression instead of ACA (slower)\n";
    std::cout << "  --rad-aca-tolerance T      ACA relative tolerance (default: 0.001)\n";
    std::cout << "  --rad-aca-max-rank R       ACA maximum rank before fallback (default: 50)\n";
    std::cout << "  --rad-hmatrix-threads N    Limit H-matrix build threads (0 = auto)\n";
    std::cout << "  --rad-captured-energy F    Target captured energy fraction (default: 0.95 = 95%)\n";
    std::cout << "  --rad-max-neighbors N      Override max neighbors per face (disables adaptive)\n";
    std::cout << "  --rad-exchange-selection S Which faces exchange: 'all' (default) or 'shadow'\n";
    std::cout << "                             (shadow = only faces at least --rad-min-shadow-fraction shadowed).\n";
    std::cout << "  --rad-min-shadow-fraction F  Shadow fraction a face must reach to be included when\n";
    std::cout << "                             --rad-exchange-selection shadow is in force (default: 0.3).\n";
    std::cout << "  --rad-cluster-cull         Enable cluster-pair visibility culling during VF assembly\n";
    std::cout << "  --rad-no-cluster-cull      Disable it (default: enabled)\n";
    std::cout << "  --rad-force-hmatrix        Force the H-matrix path (vestigial: the large-mesh\n";
    std::cout << "                             auto-disable it defeated no longer exists)\n";
    std::cout << "  --rad-vf-monte-carlo       Compute view factors by Monte Carlo ray-gather instead of\n";
    std::cout << "                             the analytic double-area integral\n";
    std::cout << "  --rad-mc-rays N            Rays per face for --rad-vf-monte-carlo (default: 100000)\n";
    std::cout << "  --rad-gpu-devices N|auto   Number of GPUs for the radiosity pool (default: 1).\n";
    std::cout << "                             1 = legacy single-GPU; auto = all enumerated GPUs.\n";
    std::cout << "  --rad-gpu-device-ids 0,1,...  Explicit radiosity GPU indices (overrides --rad-gpu-devices).\n";
    std::cout << "  --rad-gpu-batch-size N     Timesteps dispatched per pool round (default: auto = 4 * N_devices).\n";
    std::cout << "  --rad-gpu-inter-batch-pause-ms N  Sleep between consecutive batches on same device (default: 0).\n\n";

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
        file << "0.0 0.0 Sub_Jupiter\n";      // Equator at 0 deg longitude (sub-Jupiter point)
        file << "0.0 90.0 Trailing\n";        // Equator at 90 deg E (trailing hemisphere)
        file << "0.0 180.0 Anti_Jupiter\n";   // Equator at 180 deg (anti-Jupiter point)
        file << "0.0 270.0 Leading\n";        // Equator at 270 deg E (leading hemisphere)
        file << "45.0 0.0 North_45\n";        // 45 deg N at 0 deg longitude
        file << "-45.0 0.0 South_45\n";       // 45 deg S at 0 deg longitude
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

// ---------------------------------------------------------------------------
// Band-separated INDIRECT flux outputs (Paul's product, 2026-07-17 — Ada).
//
// Writes the two indirect radiosity bands as per-timestep frames + statistics,
// mirroring the direct_solar / primary_ir components exactly (uint32 face-count
// header + float32 payload; PTM-15-corrected statistics window):
//   indirect_solar/ : absorbed multiply-scattered SOLAR indirect  (1-A)·Σ_j F_ij B^vis_j
//   indirect_ir/    : absorbed thermal-IR indirect                   ε·Σ_j F_ij B^ir_j
//
// CONVENTION — ABSORBED flux at the face (W/m^2), NOT incident. This DIFFERS
// from direct_solar/primary_reflected/primary_ir (which are stored incident,
// with absorption applied at the surface BC). Rationale: the two-band radiosity
// solve produces exactly these absorbed band fluxes; their SUM equals
// indirect_flux_series — the value coupled into the thermal BC and dumped by
// PTM_DUMP_QIND_SERIES ('QIDT'). Converting to incident would require a fragile
// per-face division by (1-A_i) and ε_i that also breaks that sum-equality
// cross-check. (No Smith beaming here — with rms_slope_deg > 0 the beamed IR
// correction is added to the total but not the split, so vis+ir < total then.)
//
//   *_series : [face][ABSOLUTE step] absorbed band flux (exactly as coupled).
//   frames   : output window [eq_steps, eq_steps+out_steps) -> _0000.._{out-1}.bin
//   stats    : [stats_start, stats_start+stats_steps) (same window as the others)
// Returns true on full success; warns and returns false on any failure.
bool writeIndirectBandOutputs(
    const std::string& output_dir,
    const std::shared_ptr<Mesh>& mesh,
    const std::vector<std::vector<double>>& indirect_solar_series,
    const std::vector<std::vector<double>>& indirect_ir_series,
    int eq_steps,
    int out_steps,
    int stats_start,
    int stats_steps,
    bool write_frames,
    bool write_stats)
{
    if (!mesh || mesh->faces.empty()) {
        std::cerr << "[indirect-bands] ERROR: empty mesh; skipping.\n";
        return false;
    }
    const size_t F = mesh->faces.size();
    if (indirect_solar_series.size() != F || indirect_ir_series.size() != F) {
        std::cerr << "[indirect-bands] ERROR: series size ("
                  << indirect_solar_series.size() << "/" << indirect_ir_series.size()
                  << ") != face count (" << F << "); bands NOT written.\n";
        return false;
    }

    bool ok = true;

    // --- Per-timestep frames ------------------------------------------------
    if (write_frames && out_steps > 0) {
        try {
            std::filesystem::create_directories(output_dir + "/indirect_solar");
            std::filesystem::create_directories(output_dir + "/indirect_ir");
        } catch (const std::exception& e) {
            std::cerr << "[indirect-bands] ERROR mkdir: " << e.what() << "\n";
            return false;
        }

        // Atomic tmp+rename writer (OneDrive-safe). Same on-disk format as the
        // other component frames: uint32 face-count, then F float32 values.
        auto write_frame = [&](const std::string& path,
                               const std::vector<float>& buf) -> bool {
            const uint32_t cnt = static_cast<uint32_t>(F);
            const std::string tmp = path + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                if (!out.is_open()) return false;
                out.write(reinterpret_cast<const char*>(&cnt), sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(buf.data()),
                          static_cast<std::streamsize>(F) * sizeof(float));
                out.flush();
                if (!out.good()) return false;
            }
            std::error_code ec;
            std::filesystem::rename(tmp, path, ec);
            if (ec) { std::error_code rm; std::filesystem::remove(tmp, rm); return false; }
            return true;
        };

        std::vector<float> buf_vis(F), buf_ir(F);
        int fail = 0;
        for (int t = 0; t < out_steps; ++t) {
            const int at = eq_steps + t;  // absolute step
            #pragma omp parallel for schedule(static)
            for (size_t f = 0; f < F; ++f) {
                const auto& rv = indirect_solar_series[f];
                const auto& ri = indirect_ir_series[f];
                buf_vis[f] = (at >= 0 && at < static_cast<int>(rv.size()))
                    ? static_cast<float>(rv[at]) : 0.0f;
                buf_ir[f]  = (at >= 0 && at < static_cast<int>(ri.size()))
                    ? static_cast<float>(ri[at]) : 0.0f;
            }
            std::ostringstream ss; ss << std::setw(4) << std::setfill('0') << t;
            const std::string ts = ss.str();
            if (!write_frame(output_dir + "/indirect_solar/indirect_solar_" + ts + ".bin", buf_vis)) ++fail;
            if (!write_frame(output_dir + "/indirect_ir/indirect_ir_" + ts + ".bin", buf_ir)) ++fail;
        }
        if (fail) {
            std::cerr << "[indirect-bands] WARNING: " << fail << " frame write failure(s).\n";
            ok = false;
        } else {
            std::cout << "[indirect-bands] Wrote " << out_steps
                      << " frame(s) each to indirect_solar/ and indirect_ir/.\n";
        }
    }

    // --- Statistics over the (PTM-15-corrected) integration window ----------
    if (write_stats && stats_steps > 0) {
        auto window_stats = [&](const std::vector<std::vector<double>>& series) {
            Statistics s;
            s.avg.assign(F, 0.0);
            s.max.assign(F, std::numeric_limits<double>::lowest());
            s.min.assign(F, std::numeric_limits<double>::max());
            s.sum.assign(F, 0.0);
            s.count.assign(F, 0u);
            const int lo = stats_start;
            const int hi = stats_start + stats_steps;  // one past end
            #pragma omp parallel for schedule(static)
            for (size_t f = 0; f < F; ++f) {
                const auto& row = series[f];
                double sum = 0.0; int vc = 0;
                double mx = std::numeric_limits<double>::lowest();
                double mn = std::numeric_limits<double>::max();
                for (int t = std::max(0, lo); t < hi && t < static_cast<int>(row.size()); ++t) {
                    const double v = row[t];
                    sum += v; ++vc;
                    if (v > mx) mx = v;
                    if (v < mn) mn = v;
                }
                if (vc > 0) s.avg[f] = sum / vc;
                s.sum[f] = sum; s.count[f] = static_cast<uint32_t>(vc);
                s.max[f] = (mx == std::numeric_limits<double>::lowest()) ? 0.0 : mx;
                s.min[f] = (mn == std::numeric_limits<double>::max()) ? 0.0 : mn;
            }
            return s;
        };
        if (!writeStatisticsBinary(output_dir, "indirect_solar", window_stats(indirect_solar_series), F)) ok = false;
        if (!writeStatisticsBinary(output_dir, "indirect_ir",    window_stats(indirect_ir_series),    F)) ok = false;
        if (ok) std::cout << "[indirect-bands] Wrote indirect_solar/indirect_ir statistics.\n";
    }

    // --- Patch metadata.txt components= (advisory; readers also glob dirs) ---
    if (write_frames) {
        const std::string meta_path = output_dir + "/metadata.txt";
        std::ifstream mi(meta_path);
        if (mi) {
            std::vector<std::string> lines; std::string line;
            bool patched = false;
            while (std::getline(mi, line)) {
                if (line.rfind("components=", 0) == 0
                        && line.find("indirect_solar") == std::string::npos) {
                    if (!line.empty() && line.back() != '=') line += ",";
                    line += "indirect_solar,indirect_ir";
                    patched = true;
                }
                lines.push_back(line);
            }
            mi.close();
            if (patched) {
                std::ofstream mo(meta_path, std::ios::trunc);
                if (mo) { for (auto& l : lines) mo << l << "\n"; }
                else { std::cerr << "[indirect-bands] WARNING: cannot rewrite metadata.txt\n"; ok = false; }
            }
        } else {
            std::cerr << "[indirect-bands] WARNING: metadata.txt not found; "
                         "components= not updated (dirs still written).\n";
        }
    }

    return ok;
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
        // surface-T-only Newton solver was removed in favor of the
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

// ============================================================================
// Phase R (R1): `--calibrate-gpu` — measure this host's safe GPU dispatch
// limits and write ~/.ptm3d/machine_profile.json (or $PTM_MACHINE_PROFILE).
//
// The measurement runs the SAME validated persistent-kernel path a production
// solve uses (solvePersistentAllTimesteps) on a synthetic uniform workload, so
// the timing reflects the real dispatch cost. The cost model is the FIXED one
// ported from scripts/run_yearlong_wedge_safe.py mode calibrate:
//   per-timestep kernel wall = kernel_ms / probe_steps   (one batch)
//   dispatch of N timesteps   ~ N * per-timestep kernel wall
//   max_timesteps_per_dispatch = floor(ceiling_s * 1000 / per_timestep_ms)
// Host-side per-dispatch overhead (buffer create/upload/read) is measured as
// (wall - kernel) around the solve call and recorded for the runner's
// launch-time consistency check.
//
// Returns a process exit code (0 = profile written).
// ============================================================================
#ifdef PTM_GPU_OPENCL
static int runGpuCalibration(int argc, char** argv) {
    // Fermi F3: the probe MUST be production-representative or the ceiling is
    // unsafe. The kernel's von-Neumann substep count (Fo<0.49) is set by the
    // MINIMUM near-surface dz and kappa at the OUTPUT dt: a uniform 1 cm grid
    // @250 K yields ~2 substeps/step, but production cold-PSR fine grids hit
    // 29-200x -- so a ceiling calibrated on a coarse grid could run ~100x
    // longer in production and WEDGE. Defaults below are lunar-cold-PSR-ish
    // (0.5 mm first layer, geometric grid, low TI props, hourly dt). For a
    // guaranteed-representative ceiling, prefer the automatic in-run calibration
    // (default ON) which probes the LOADED run's own grid instead of these.
    int    batch_size    = 8000;
    int    n_layers      = 50;
    double ceiling_s     = 25.0;
    double dz0           = 5.0e-4;   // first-layer thickness [m] (fine near-surface)
    double grid_ratio    = 1.2;      // geometric growth of dz with depth
    double dt_output_s   = 3600.0;   // OUTPUT timestep [s]
    double kc_probe      = 2.0e-3;   // near-surface conductivity [W/m/K]
    double rho_probe     = 1100.0;   // density [kg/m^3]
    double cp0_probe     = 600.0;    // heat capacity [J/kg/K]
    int    probe_steps_1 = 30;       // two-point regression (F4): small point
    int    probe_steps_2 = 120;      //                            large point
    double safety_factor = 4.0;      // clamp: max_ts <= factor x conservative default
    std::string profile_out;         // empty => resolved default path

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--batch-size"      && i + 1 < argc) batch_size    = std::stoi(argv[++i]);
        else if (a == "--gpu-batch-size"  && i + 1 < argc) batch_size    = std::stoi(argv[++i]);
        else if (a == "--n-layers"        && i + 1 < argc) n_layers      = std::stoi(argv[++i]);
        else if (a == "--safe-ceiling-s"  && i + 1 < argc) ceiling_s     = std::stod(argv[++i]);
        else if (a == "--dz0"             && i + 1 < argc) dz0           = std::stod(argv[++i]);
        else if (a == "--grid-ratio"      && i + 1 < argc) grid_ratio    = std::stod(argv[++i]);
        else if (a == "--dt-output-s"     && i + 1 < argc) dt_output_s   = std::stod(argv[++i]);
        else if (a == "--kc"              && i + 1 < argc) kc_probe      = std::stod(argv[++i]);
        else if (a == "--rho"             && i + 1 < argc) rho_probe     = std::stod(argv[++i]);
        else if (a == "--cp0"             && i + 1 < argc) cp0_probe     = std::stod(argv[++i]);
        else if (a == "--probe-steps-1"   && i + 1 < argc) probe_steps_1 = std::stoi(argv[++i]);
        else if (a == "--probe-steps-2"   && i + 1 < argc) probe_steps_2 = std::stoi(argv[++i]);
        else if (a == "--safety-factor"   && i + 1 < argc) safety_factor = std::stod(argv[++i]);
        else if (a == "--profile-out"     && i + 1 < argc) profile_out   = argv[++i];
    }
    if (batch_size < 1 || n_layers < 2 || ceiling_s <= 0.0 || dz0 <= 0.0 ||
        grid_ratio < 1.0 || dt_output_s <= 0.0 || probe_steps_1 < 1 ||
        probe_steps_2 <= probe_steps_1 || safety_factor <= 0.0) {
        std::cerr << "[calibrate-gpu] invalid parameters (need batch>=1, "
                     "n_layers>=2, ceiling>0, dz0>0, ratio>=1, dt>0, "
                     "0<steps1<steps2, safety>0)\n";
        return 2;
    }

    std::cout << "[calibrate-gpu] representative probe: batch=" << batch_size
              << " n_layers=" << n_layers << " dz0=" << dz0 << " m ratio="
              << grid_ratio << " dt=" << dt_output_s << " s (kc=" << kc_probe
              << " rho=" << rho_probe << " cp0=" << cp0_probe << ")\n";

    // Build the manual probe: synthesized geometric grid, is_run_grid=false (keeps
    // the conservative 4x F3 clamp). No wedge budget guard on the manual path —
    // the operator explicitly asked to probe — so keep prior behavior.
    CalibrationProbe pr;
    pr.n_layers    = n_layers;
    pr.batch       = batch_size;
    pr.dt_output_s = dt_output_s;
    pr.kc = kc_probe; pr.rho = rho_probe; pr.cp0 = cp0_probe;
    pr.cp_eff = cp0_probe;           // constant cp (no polynomial) on the manual path
    pr.R350 = 2.7;                   // cp_coeffs left empty
    pr.emissivity  = 0.95;
    pr.ceiling_s   = ceiling_s;
    pr.safety_factor = safety_factor;
    pr.probe_steps_1 = probe_steps_1;
    pr.probe_steps_2 = probe_steps_2;
    pr.is_run_grid = false;
    pr.probe_budget_s = 1e300;      // effectively no budget guard for manual probe
    pr.z_grid.resize(static_cast<size_t>(n_layers));
    pr.z_grid[0] = 0.0f;
    {
        double dz = dz0;
        for (int i = 1; i < n_layers; ++i) {
            pr.z_grid[i] = pr.z_grid[i - 1] + static_cast<float>(dz);
            dz *= grid_ratio;
        }
    }

    gpu::MachineProfile p;
    int rc = calibrateGpuCore(pr, p);
    if (rc != 0) return rc;

    if (!gpu::saveMachineProfile(p, profile_out)) {
        std::cerr << "[calibrate-gpu] FAILED to write profile.\n";
        return 5;
    }
    std::string where = profile_out.empty()
        ? gpu::machine_profile_detail::defaultProfilePath() : profile_out;
    std::cout << "[calibrate-gpu] wrote profile to " << where << "\n";
    return 0;
}
#endif  // PTM_GPU_OPENCL

// ---------------------------------------------------------------------------
// Equilibration-phase indirect flux: the run record.
// ---------------------------------------------------------------------------
// A default that changes the deep column of every radiation-coupled run has to
// say in the log whether it fired, where the numbers came from, and how big
// they were. These travel together, so they are one object rather than six
// loose locals that can drift apart. `provenance == "none"` with an empty
// `reason` is the state before anything has decided, and is never printed.
struct EqIndirectRecord {
    std::string provenance = "none";  // "internal pass" | "file: <path>" | "none"
    std::string reason;               // why nothing was applied, when nothing was
    double mean_Wm2   = 0.0;
    double median_Wm2 = 0.0;
    double max_Wm2    = 0.0;
    size_t n_nonzero  = 0;

    // Census of a per-face <q_ind> vector. Median as well as mean because the
    // distribution is strongly skewed -- a handful of crater floors carry most
    // of the flux, so the mean alone reads as "negligible" for a correction
    // that is tens of kelvin on the faces that matter.
    void census(const std::vector<double>& q) {
        std::vector<double> sorted = q;
        std::sort(sorted.begin(), sorted.end());
        median_Wm2 = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
        double sum = 0.0;
        max_Wm2 = 0.0;
        n_nonzero = 0;
        for (double v : q) {
            sum += v;
            max_Wm2 = std::max(max_Wm2, v);
            if (v > 0.0) ++n_nonzero;
        }
        mean_Wm2 = sum / std::max<size_t>(1, q.size());
    }
};

int main(int argc, char** argv) {
    // Line-buffer stdout so progress prints reach disk before a kernel hang
    // can swallow them. The macOS+AMD OpenCL driver has wedged the box hard
    // enough during multi-hour rad-coupling runs that the unified log
    // captures nothing pre-reboot — line-buffering at least keeps the
    // user-visible run log fresh up to the moment of the hang.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

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
        } else if (arg == "--calibrate-gpu") {
            // Phase R (R1): measure this host's safe GPU dispatch limits and
            // write ~/.ptm3d/machine_profile.json (or $PTM_MACHINE_PROFILE).
#ifdef PTM_GPU_OPENCL
            return runGpuCalibration(argc, argv);
#else
            std::cerr << "--calibrate-gpu requires an OpenCL-enabled build "
                         "(ENABLE_GPU=ON). CPU-only builds do not need a GPU "
                         "profile." << std::endl;
            return 1;
#endif
        }
    }

    // ========================================================================
    // PHASE 2: Initialize configuration with defaults
    // ========================================================================
    ptm::SimulationConfig sim_config;  // Starts with default values

    // PTM-42: apply the config's `engine_env:` block BEFORE anything reads the
    // environment. This must precede loadFromYAML (which reads PTM_DATA_ROOT)
    // and every solver/illumination construction (which latch their gates on
    // first call). The process environment wins on a conflict, and every
    // variable is echoed with its disposition -- see
    // ConfigParser::applyEngineEnv.
    if (has_config_file) {
        ptm::ConfigParser::applyEngineEnv(config_file);
    }

    // If config file specified, load it
    if (has_config_file) {
        try {
            ptm::ConfigParser parser;
            sim_config = parser.loadFromYAML(config_file);
            std::cout << "✓ Configuration loaded from: " << config_file << std::endl;
        } catch (const std::exception& e) {
            // PTM-66: this used to warn and CONTINUE. `sim_config` is still at
            // its constructor defaults here, so continuing ran the whole
            // simulation on built-in defaults -- default body, mesh, timesteps,
            // thermophysical properties and output directory -- and exited 0.
            // Demonstrated 2026-08-04: a config whose only defect was one
            // rejected enum value produced a complete 12-step Moon icosphere
            // run, wrote statistics to the DEFAULT output/ directory rather
            // than the config's, and returned 0, so no wrapper or orchestrator
            // could detect it. Defaults exist for the NO-CONFIG case; an
            // operator who passed --config asked for that file specifically.
            std::cerr << "FATAL: failed to load config file '" << config_file
                      << "': " << e.what() << "\n"
                      << "  Refusing to run. Continuing on built-in defaults would "
                         "silently replace the body, mesh, timesteps, thermophysical "
                         "properties and output directory you configured.\n"
                      << "  See docs/KNOWN_LIMITATIONS.md (PTM-66)." << std::endl;
            return 1;
        }
    }

    // Validate the conductivity-law selector HERE, before anything else runs.
    // An unrecognized law must never be silently substituted with Hayne -- that
    // would attribute an entire campaign to the wrong physics.
    //
    // Historically this could not be done by throwing from the parser: the catch
    // above downgraded any parse exception to a warning and then ran the whole
    // simulation on built-in defaults with exit code 0 (BUGS.md PTM-66). PTM-66
    // is fixed in this same release, so a throw would now be fatal -- but this
    // check stays, because it names the offending value and the valid set
    // instead of emitting a generic "failed to load config file".
    {
        thermal::ConductivityModel cm_probe;
        if (!thermal::parseConductivityModel(
                sim_config.thermophysical.conductivity_model, cm_probe)) {
            std::cerr << "FATAL: unknown thermophysical.conductivity_model '"
                      << sim_config.thermophysical.conductivity_model
                      << "'. Valid values: hayne2017 (default) | martinez2021.\n";
            return 1;
        }
    }

    // Phase R (R4): mid-run checkpoint cadence. The config key exists so the
    // knob is first-class, but true per-output-step full-mesh checkpointing is
    // architecturally incompatible with the batch-outer/time-inner GPU
    // persistent solver (see config_parser.h + docs/PORTABILITY-AUDIT.md).
    // Fail loud rather than silently no-op if a user sets it.
    if (sim_config.thermal.checkpoint_every_output_steps > 0) {
        std::cerr << "[Phase R/R4] WARNING: thermal.checkpoint_every_output_steps="
                  << sim_config.thermal.checkpoint_every_output_steps
                  << " is set, but per-output-step full-mesh checkpointing is NOT "
                     "active on the GPU persistent solver (batch-outer/time-inner: "
                     "no wall-clock moment holds all faces at the same output step; "
                     "a partial dump is a resume footgun). Use the runner's chunk "
                     "split (run_yearlong_wedge_safe.py --stage-b-nchunks) for "
                     "blast-radius bounding. This knob is reserved for the CPU-first "
                     "solver path (portability Phase 1)." << std::endl;
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
    // Wire heartbeat to the output dir so post-mortem readers can see the
    // last in-flight phase after a kernel hang / forced reboot.
    setHeartbeatDir(output_dir);
    writeHeartbeat("startup output_dir=" + output_dir);
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

    // Scalar physical-property overrides taken straight from argv. `--albedo`
    // and `--constant-thermal-inertia` land directly on their config keys in
    // the argv loop; `--thermal-inertia` cannot, because resolving it needs the
    // finished Planet's rhos / cp / ks / kd. It is parked here and resolved in
    // PHASE 4b' below, before anything reads the override keys.
    double cli_thermal_inertia = -1.0;

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
    double rad_coupling_flux_tol = 0.1;     // Flux change threshold [W/m2] (legacy mode only)
    double rad_coupling_face_frac = 0.01;   // Face fraction still changing to stop
    std::string rad_coupling_select_mode = "physics";  // "physics" (default) or "legacy"
    double rad_coupling_select_safety = 0.5;            // Safety factor on predicted-ΔT threshold
    // PTM-89: "bounded" (default) | "fixed" | "legacy". See config_parser.h.
    std::string rad_coupling_mode = "bounded";
    double rad_coupling_select_dt = 0.0;    // Absolute predicted-ΔT selection threshold [K]
    // PTM-98: "all" (default) | "shadow". Radiation-exchange face set. See config_parser.h.
    std::string rad_exchange_selection = "all";
    double rad_min_shadow_fraction = 0.3;   // "shadow" mode only (PTM-99: now plumbed)

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

    // Anisotropic (roughness-beamed) thermal emission (Smith 1967). 0 = OFF.
    double rad_rms_slope_deg = 0.0;              // RMS micro-slope [deg]
    double rad_beam_vf_threshold = 1e-7;         // Min renormalized VF to keep a beam pair
    double rad_radiance_prune_eps = 0.0;         // Radiance-prune eps [W/m^2] (assembly-time; 0 = off)

    // Multi-GPU radiosity dispatch (mirrors thermal pool semantics).
    int rad_gpu_devices = 1;                     // 1 = legacy single-GPU; >1 = pool
    std::vector<int> rad_gpu_device_ids;         // Explicit indices (overrides rad_gpu_devices)
    int rad_gpu_batch_size = 0;                  // 0 = auto = 4 * rad_gpu_devices
    int rad_gpu_inter_batch_pause_ms = 0;        // Watchdog mitigation

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
    rad_coupling_select_mode = sim_config.radiation.coupling_select_mode;
    rad_coupling_select_safety = sim_config.radiation.coupling_select_safety;
    rad_coupling_mode = sim_config.radiation.coupling_mode;
    rad_coupling_select_dt = sim_config.radiation.coupling_select_dt;
    rad_exchange_selection = sim_config.radiation.exchange_selection;
    rad_min_shadow_fraction = sim_config.radiation.min_shadow_fraction;
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
    rad_rms_slope_deg = sim_config.radiation.rms_slope_deg;
    rad_beam_vf_threshold = sim_config.radiation.beam_vf_threshold;
    rad_radiance_prune_eps = sim_config.radiation.radiance_prune_eps;
    // Multi-GPU radiosity (only consulted if rad_use_gpu == true).
    rad_gpu_devices = sim_config.radiation.gpu_devices;
    rad_gpu_device_ids = sim_config.radiation.gpu_device_ids;
    rad_gpu_batch_size = sim_config.radiation.gpu_batch_size;
    rad_gpu_inter_batch_pause_ms = sim_config.radiation.gpu_inter_batch_pause_ms;

    if (enable_radiation_exchange) {
        std::cout << "Radiation exchange enabled from config file" << std::endl;
        if (rad_use_hmatrix) {
            std::cout << "  H-matrix compression: enabled" << std::endl;
        }
        // Radiance-weighted VF pruning is applied at VF ASSEMBLY, not in the sim;
        // we only surface the effective epsilon for provenance. An env override
        // (PTM_RADIANCE_PRUNE_EPS) wins over the config at assembly time, so if it
        // is set AND differs, report that it overrides -- never silently.
        {
            const char* eps_env = std::getenv("PTM_RADIANCE_PRUNE_EPS");
            double eps_cfg = rad_radiance_prune_eps;
            double eps_eff = eps_cfg;
            bool from_env = false;
            if (eps_env && *eps_env) {
                eps_eff = std::atof(eps_env);
                from_env = (eps_eff != eps_cfg);
            }
            std::cout << "  radiance-prune epsilon: " << eps_eff
                      << " W/m^2 (assembly-time; 0=off)";
            if (from_env) {
                std::cout << " [env PTM_RADIANCE_PRUNE_EPS overrides config "
                          << eps_cfg << "]";
            }
            std::cout << std::endl;
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
        } else if (arg == "--equilibration-cycles" && i + 1 < argc) {
            sim_config.thermal.equilibration_cycles = std::stoi(argv[++i]);
            std::cout << "Equilibration passes over the eq forcing array: "
                      << sim_config.thermal.equilibration_cycles
                      << " (legacy = 2)" << std::endl;
        } else if (arg == "--equilibration-steps-per-day" && i + 1 < argc) {
            eq_steps_per_day_cli = std::stod(argv[++i]);
        } else if (arg == "--equilibration-samples-per-diurnal" && i + 1 < argc) {
            eq_samples_per_diurnal_cli = std::stod(argv[++i]);
        } else if (arg == "--geothermal-heat-flow" && i + 1 < argc) {
            geothermal_heat_flow = std::stod(argv[++i]);
            std::cout << "Setting geothermal heat flow to: " << geothermal_heat_flow << " W/m^2" << std::endl;
        } else if (arg == "--heat-flow-mask" && i + 1 < argc) {
            sim_config.thermal.heat_flow_mask_file = argv[++i];
            std::cout << "Heat flow mask file: " << sim_config.thermal.heat_flow_mask_file << std::endl;
        } else if (arg == "--heat-flow-mask-value" && i + 1 < argc) {
            sim_config.thermal.heat_flow_mask_value = std::stod(argv[++i]);
            std::cout << "Heat flow mask value: " << sim_config.thermal.heat_flow_mask_value << " W/m^2" << std::endl;
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
        } else if (arg == "--thermal-inertia" && i + 1 < argc) {
            // Resolved in PHASE 4b' -- it needs the finished Planet.
            if (!parseCliDouble(arg, argv[++i], cli_thermal_inertia)) return 1;
            if (!(cli_thermal_inertia > 0.0)) {
                std::cerr << "FATAL: --thermal-inertia must be > 0 (got "
                          << cli_thermal_inertia
                          << "). Thermal inertia is sqrt(k*rho*cp); zero or negative "
                             "has no property assignment." << std::endl;
                return 1;
            }
        } else if (arg == "--constant-thermal-inertia" && i + 1 < argc) {
            double v = -1.0;
            if (!parseCliDouble(arg, argv[++i], v)) return 1;
            if (!(v > 0.0)) {
                std::cerr << "FATAL: --constant-thermal-inertia must be > 0 (got " << v
                          << "). Use --thermal-inertia, or omit both, for the body default."
                          << std::endl;
                return 1;
            }
            sim_config.thermophysical.constant_thermal_inertia = v;
            std::cout << "[cli] constant_thermal_inertia = " << v << " SI" << std::endl;
        } else if (arg == "--albedo" && i + 1 < argc) {
            double v = -1.0;
            if (!parseCliDouble(arg, argv[++i], v)) return 1;
            if (!(v >= 0.0 && v < 1.0)) {
                std::cerr << "FATAL: --albedo must be in [0, 1) (got " << v
                          << "). This is the Bond albedo A0, a reflected fraction."
                          << std::endl;
                return 1;
            }
            // Same key the YAML `thermophysical.albedo_override` sets, so the
            // flag inherits the PTM-53 solver forwarding, the albedo-map
            // out-ranking warning and the rule-7 tripwire unchanged. CLI beats
            // config, per the documented precedence -- say so when it does.
            double prior = sim_config.thermophysical.albedo_override;
            if (prior >= 0.0 && prior != v) {
                std::cout << "[cli] --albedo " << v
                          << " overrides thermophysical.albedo_override " << prior
                          << " from the config file" << std::endl;
            }
            sim_config.thermophysical.albedo_override = v;
            std::cout << "[cli] albedo_override = " << v << std::endl;
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
        } else if (arg == "--rad-coupling-select-mode" && i + 1 < argc) {
            rad_coupling_select_mode = argv[++i];
            if (rad_coupling_select_mode != "physics" && rad_coupling_select_mode != "legacy") {
                std::cerr << "Invalid --rad-coupling-select-mode '"
                          << rad_coupling_select_mode
                          << "': must be 'physics' or 'legacy'." << std::endl;
                return 1;
            }
            std::cout << "Radiation coupling face-selection mode: "
                      << rad_coupling_select_mode << std::endl;
        } else if (arg == "--rad-coupling-select-safety" && i + 1 < argc) {
            rad_coupling_select_safety = std::stod(argv[++i]);
            std::cout << "Radiation coupling select safety factor: "
                      << rad_coupling_select_safety << std::endl;
        } else if (arg == "--rad-coupling-mode" && i + 1 < argc) {
            rad_coupling_mode = argv[++i];
            if (rad_coupling_mode != "bounded" && rad_coupling_mode != "fixed"
                    && rad_coupling_mode != "legacy") {
                std::cerr << "Invalid --rad-coupling-mode '" << rad_coupling_mode
                          << "': must be 'bounded', 'fixed' or 'legacy'." << std::endl;
                return 1;
            }
            std::cout << "Radiation coupling mode: " << rad_coupling_mode << std::endl;
        } else if (arg == "--rad-coupling-select-dt" && i + 1 < argc) {
            rad_coupling_select_dt = std::stod(argv[++i]);
            std::cout << "Radiation coupling selection threshold: "
                      << rad_coupling_select_dt << " K" << std::endl;
        } else if (arg == "--rad-exchange-selection" && i + 1 < argc) {
            rad_exchange_selection = argv[++i];
            if (rad_exchange_selection != "all" && rad_exchange_selection != "shadow") {
                std::cerr << "Invalid --rad-exchange-selection '" << rad_exchange_selection
                          << "': must be 'all' or 'shadow'." << std::endl;
                return 1;
            }
            std::cout << "Radiation exchange selection: " << rad_exchange_selection << std::endl;
        } else if (arg == "--rad-min-shadow-fraction" && i + 1 < argc) {
            rad_min_shadow_fraction = std::stod(argv[++i]);
            std::cout << "Radiation min shadow fraction: " << rad_min_shadow_fraction
                      << " (exchange_selection: shadow only)" << std::endl;
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
        } else if (arg == "--rad-gpu-devices" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto" || val == "all") {
                rad_gpu_devices = 0;  // 0 = all available
            } else {
                rad_gpu_devices = std::stoi(val);
            }
            std::cout << "[Multi-GPU radiosity] gpu_devices = "
                      << (rad_gpu_devices == 0 ? "auto"
                                                : std::to_string(rad_gpu_devices))
                      << std::endl;
        } else if (arg == "--rad-gpu-device-ids" && i + 1 < argc) {
            rad_gpu_device_ids.clear();
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) rad_gpu_device_ids.push_back(std::stoi(token));
            }
            std::cout << "[Multi-GPU radiosity] gpu_device_ids =";
            for (int id : rad_gpu_device_ids) std::cout << " " << id;
            std::cout << std::endl;
        } else if (arg == "--rad-gpu-batch-size" && i + 1 < argc) {
            rad_gpu_batch_size = std::stoi(argv[++i]);
            std::cout << "[Multi-GPU radiosity] gpu_batch_size = "
                      << rad_gpu_batch_size << " timesteps/dispatch"
                      << std::endl;
        } else if (arg == "--rad-gpu-inter-batch-pause-ms" && i + 1 < argc) {
            rad_gpu_inter_batch_pause_ms = std::stoi(argv[++i]);
            std::cout << "[Multi-GPU radiosity] inter-batch pause = "
                      << rad_gpu_inter_batch_pause_ms << " ms" << std::endl;
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

    // ----------------------------------------------------------------------
    // PTM-89 coupling-mode resolution, AT CONFIG TIME.
    //
    // Resolved and validated here — before the illumination pass — so a typo in
    // the mode costs nothing instead of surfacing after minutes to hours of ray
    // tracing (the failure shape of the *other* PTM-80). The env override lets a
    // published run be reproduced without editing its config.
    // ----------------------------------------------------------------------
    {
        if (const char* e = std::getenv("PTM_COUPLING_MODE")) {
            const std::string v(e);
            if (v != rad_coupling_mode) {
                std::cout << "[coupling] PTM_COUPLING_MODE overrides coupling mode '"
                          << rad_coupling_mode << "' -> '" << v << "'" << std::endl;
            }
            rad_coupling_mode = v;
        }
        if (rad_coupling_mode != "bounded" && rad_coupling_mode != "fixed"
                && rad_coupling_mode != "legacy") {
            std::cerr << "FATAL [coupling]: radiation.coupling_mode='" << rad_coupling_mode
                      << "' is not one of bounded|fixed|legacy.\n"
                         "  bounded (default) = PTM-89 fix: absolute selection threshold, no"
                         " face-fraction exit.\n"
                         "  fixed             = as bounded, plus a fixed iteration count.\n"
                         "  legacy            = pre-PTM-89 behavior, for reproducing published"
                         " runs." << std::endl;
            return 1;
        }
        if (rad_coupling_select_dt < 0.0) {
            std::cerr << "FATAL [coupling]: radiation.coupling_select_dt="
                      << rad_coupling_select_dt << " K must be >= 0." << std::endl;
            return 1;
        }
    }

    // ----------------------------------------------------------------------
    // PTM-98 exchange-selection resolution, AT CONFIG TIME (same rationale as
    // the coupling-mode block above: a typo must cost nothing).
    // ----------------------------------------------------------------------
    {
        if (const char* e = std::getenv("PTM_EXCHANGE_SELECTION")) {
            const std::string v(e);
            if (v != rad_exchange_selection) {
                std::cout << "[exchange] PTM_EXCHANGE_SELECTION overrides exchange selection '"
                          << rad_exchange_selection << "' -> '" << v << "'" << std::endl;
            }
            rad_exchange_selection = v;
        }
        if (rad_exchange_selection != "all" && rad_exchange_selection != "shadow") {
            std::cerr << "FATAL [exchange]: radiation.exchange_selection='"
                      << rad_exchange_selection << "' is not one of all|shadow.\n"
                         "  all (default) = PTM-98 fix: every face keeps the indirect flux the\n"
                         "                  radiosity solve computed for it. Window-independent.\n"
                         "  shadow        = pre-PTM-98 shadow-fraction gate, for reproducing\n"
                         "                  published runs. Window-DEPENDENT." << std::endl;
            return 1;
        }
        if (rad_exchange_selection == "shadow") {
            std::cout << "[exchange] WARNING: radiation.exchange_selection='shadow' re-introduces"
                         " PTM-98 — the set of faces that receive indirect flux is measured over"
                         " THIS window (equilibration + output steps), so the result depends on"
                         " how the run was chunked. Use it only to reproduce a pre-PTM-98 run,"
                         " and pair it with coupling_mode: legacy." << std::endl;
            if (rad_coupling_mode != "legacy") {
                std::cout << "[exchange] NOTE: exchange_selection='shadow' with coupling_mode='"
                          << rad_coupling_mode << "' reproduces NEITHER the pre-PTM-98 nor the"
                             " current behavior exactly; pre-PTM-98 reproduction needs"
                             " coupling_mode: legacy as well." << std::endl;
            }
            if (!(rad_min_shadow_fraction >= 0.0 && rad_min_shadow_fraction <= 1.0)) {
                std::cerr << "FATAL [exchange]: radiation.min_shadow_fraction="
                          << rad_min_shadow_fraction << " must be in [0, 1]." << std::endl;
                return 1;
            }
        } else if (rad_min_shadow_fraction != 0.3) {
            std::cout << "[exchange] WARNING: radiation.min_shadow_fraction="
                      << rad_min_shadow_fraction << " is IGNORED under"
                         " exchange_selection='all' (every face participates). Set"
                         " exchange_selection: shadow to make it live." << std::endl;
        }
    }

    // ----------------------------------------------------------------------
    // GPU-availability gate, AT CONFIG TIME.
    //
    // The compiled-in check already existed, but only inside the GPU thermal
    // entry point -- i.e. AFTER the whole illumination pass. A CPU-only build
    // handed a `use_gpu: true` config therefore ray-traced the entire run
    // (minutes to hours on a production mesh) and only then printed
    // "[GPU Thermal] Error: OpenCL support not compiled in" and gave up, having
    // written nothing. Measured on the clean-room CPU-only build: full
    // illumination for a 1280-face ico3, then the error. 47 of the 55 configs
    // under configs/ set use_gpu: true, so this is the default way a new user
    // meets a CPU-only build.
    //
    // The downstream check in the GPU entry point is deliberately LEFT IN PLACE
    // (it guards paths that can enable the GPU after this point); this one only
    // moves the common failure forward to where it costs nothing.
    {
        const bool want_gpu_thermal = config.getPropertyBool("use_gpu_thermal", false);
        const bool want_gpu_illum   = sim_config.illumination.use_gpu;
        const bool want_gpu_rad     = rad_use_gpu;
#ifndef PTM_GPU_OPENCL
        if (want_gpu_thermal) {
            std::cerr << "FATAL [gpu]: this binary was built WITHOUT GPU support, but the run "
                         "requests the GPU thermal solver.\n"
                         "  Refusing to run: the illumination pass would complete and the "
                         "thermal stage would then fail, wasting the whole run.\n"
                         "  Either set `thermal.use_gpu: false` in the config (or drop "
                         "--use-gpu-thermal) to use the CPU solver, which is correct and on "
                         "many hosts faster,\n"
                         "  or rebuild with -DPTM_ENABLE_GPU=ON -DGPU_BACKEND=OPENCL."
                      << std::endl;
            return 1;
        }
        if (want_gpu_illum || want_gpu_rad) {
            // Not fatal: both of these already fall back to CPU. Say so instead
            // of falling back in silence.
            std::cerr << "[gpu] WARNING: this binary was built WITHOUT GPU support; "
                      << (want_gpu_illum ? "illumination.use_gpu " : "")
                      << (want_gpu_illum && want_gpu_rad ? "and " : "")
                      << (want_gpu_rad ? "radiation.use_gpu " : "")
                      << "will be IGNORED and the CPU path used. Set the flag(s) false to "
                         "silence this, or rebuild with -DPTM_ENABLE_GPU=ON "
                         "-DGPU_BACKEND=OPENCL." << std::endl;
        }
#else
        (void)want_gpu_thermal; (void)want_gpu_illum; (void)want_gpu_rad;
#endif
    }

    // Override planet geothermal heat flow if specified in config or command line
    if (geothermal_heat_flow > 0) {
        planet.Qb = geothermal_heat_flow;
        std::cout << "Overriding planet geothermal heat flow: " << planet.Qb << " W/m^2" << std::endl;
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
        std::cout << "  Solar constant: " << generic_body.thermo.S << " W/m^2" << std::endl;
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
    // PHASE 4b': --thermal-inertia  ->  ks / kd on the RESOLVED body
    // ========================================================================
    // Thermal inertia is not a stored property of this engine. The primaries
    // are ks / kd / rhos / rhod / cp0 / chi, which is very likely why 126 CLI
    // flags existed and none of them set TI. Rather than define a new
    // inversion, this resolves the flag onto the SAME two knobs every existing
    // thermal-inertia sweep in the tree already turns -- ks_override and
    // kd_override -- so the request inherits, unchanged, the PTM-53 forwarding
    // to every solver, the martinez2021 composition guard, and the rule-7
    // tripwire that fails a run whose override no solver applied.
    //
    // THE CONVENTION, and where it comes from. Three places in this repository
    // already map a thermal-inertia NUMBER onto properties this way:
    //   scripts/ema_campaign_driver.py   ti_to_ks(ti) = ti^2 / (rhos * cp0),
    //                                    "TI is set PHYSICALLY via ks"
    //   scripts/divval_sweep.py          ks_override and kd_override scaled by
    //                                    the same factor, "TI ~ sqrt(k)"
    //   examples/quickstart_asteroid_diurnal{,_ti200_check}.yaml
    //                                    ks,kd x4 shipped as "TI: 100 -> 200"
    // So: TI = sqrt(ks * rhos * cp) at the SURFACE, with the densities and the
    // heat capacity held fixed and the conductivity carrying the whole change;
    // kd moves by the same FACTOR as ks, so the kd/ks depth contrast the body
    // was configured with survives instead of being flattened. That factor rule
    // is divval's ("depth-profile-preserving"), and it is a no-op on the
    // asteroid quickstarts, where ks == kd already.
    //
    // Varying k rather than cp is not a free choice: cp of a silicate regolith
    // is known to within a factor well under two, while k spans four orders of
    // magnitude with porosity and contact area. It is also the only reading
    // under which the flag reproduces the two-point sweep the repository ships
    // as a tutorial.
    //
    // cp IS EVALUATED AT 273 K, by the same polynomial and at the same
    // reference temperature as the engine's only other body-level thermal-
    // inertia definition (thermal_property_map.cpp: cpAtT / depthWeightedTI --
    // the TI(273 K) maps of Hayne et al. 2017). On a body whose cp is constant,
    // i.e. every generic body, cp(273) == cp0 exactly and this reduces to the
    // EMA/quickstart formula with no residual. On the Moon it is the only
    // workable choice: `cp0` there is the CONSTANT TERM of the cp polynomial
    // and equals -3.6125 J/kg/K, so sqrt(ks*rhos*cp0) is not a real number.
    //
    // WHAT THIS FLAG DOES NOT DO. It does not make thermal inertia constant in
    // temperature or in depth. chi (k's T^3 term) and the cp(T) polynomial are
    // left alone, so on a body that has them the delivered TI moves with
    // temperature about the requested value, and the requested number is a
    // nominal surface TI at 273 K. `--constant-thermal-inertia` is the knob for
    // the other reading -- exactly constant, at the cost of collapsing cp(T),
    // chi and the two-layer profile -- and the banner below names it.
    // ========================================================================
    if (cli_thermal_inertia > 0.0) {
        auto& tpw = sim_config.thermophysical;

        // Rule 7: two knobs that both claim to set thermal inertia cannot both
        // be honored, and picking one silently is how a sweep measures the
        // wrong axis for a week. Refuse, and name the choice.
        if (tpw.constant_thermal_inertia > 0.0) {
            std::cerr << "\nFATAL: --thermal-inertia " << cli_thermal_inertia
                      << " and constant_thermal_inertia "
                      << tpw.constant_thermal_inertia
                      << " both set the thermal inertia, by different conventions.\n"
                      << "  --thermal-inertia          scales k, keeps cp(T), chi and the "
                         "kd/ks depth contrast;\n"
                         "                             the delivered TI varies with "
                         "temperature about the request.\n"
                      << "  --constant-thermal-inertia derives cp instead, and forces "
                         "chi=0, kd=ks, rhod=rhos so\n"
                         "                             that TI is exactly constant in "
                         "temperature AND depth.\n"
                      << "  Pick one." << std::endl;
            return 1;
        }
        if (tpw.ks_override > 0.0 || tpw.kd_override > 0.0) {
            std::cerr << "\nFATAL: --thermal-inertia " << cli_thermal_inertia
                      << " resolves to ks_override and kd_override, but the config "
                         "already sets"
                      << (tpw.ks_override > 0.0 ? " ks_override" : "")
                      << (tpw.kd_override > 0.0 ? " kd_override" : "")
                      << ".\n  Refusing to overwrite an explicit conductivity with a "
                         "derived one. Remove one of them." << std::endl;
            return 1;
        }
        if (!thermal_inertia_map_file.empty()) {
            std::cerr << "\nFATAL: --thermal-inertia " << cli_thermal_inertia
                      << " is a single body-wide value, but a per-face thermal-inertia "
                         "map is also configured\n  ("
                      << thermal_inertia_map_file
                      << "). The map's TI values are inverted to per-face H against the "
                         "body's ks/kd;\n  rescaling ks/kd underneath it would silently "
                         "move every face off the TI the map asked for.\n"
                         "  Use one or the other." << std::endl;
            return 1;
        }

        const double rhos_b = planet.rhos;
        const double ks_b   = planet.ks;
        const double kd_b   = planet.kd;
        const double cp_ref = heatCapacityAtT(planet, 273.0);
        if (!(rhos_b > 0.0) || !(ks_b > 0.0) || !(cp_ref > 0.0)) {
            std::cerr << "\nFATAL: --thermal-inertia " << cli_thermal_inertia
                      << " needs rhos > 0, ks > 0 and cp(273 K) > 0 to solve "
                         "ks = TI^2 / (rhos * cp),\n  but the resolved body '"
                      << planet.name << "' has rhos = " << rhos_b
                      << ", ks = " << ks_b << ", cp(273 K) = " << cp_ref << ".\n"
                      << "  Refusing to run with the body defaults under a "
                         "thermal-inertia banner." << std::endl;
            return 1;
        }

        const double ks_new = (cli_thermal_inertia * cli_thermal_inertia)
                              / (rhos_b * cp_ref);
        const double factor = ks_new / ks_b;
        const double kd_new = kd_b * factor;
        const double ti_before = std::sqrt(ks_b * rhos_b * cp_ref);

        tpw.ks_override = ks_new;
        if (kd_new > 0.0) tpw.kd_override = kd_new;

        std::cout << "Thermal inertia (requested): " << cli_thermal_inertia
                  << " SI = sqrt(ks * rhos * cp(273 K)), surface, nominal\n"
                  << "  body '" << planet.name << "': rhos = " << rhos_b
                  << " kg/m^3, cp(273 K) = " << cp_ref << " J/kg/K"
                  << (planet.cpCoeff.empty() ? " (constant)"
                                             : " (cp is T-dependent)") << "\n"
                  << "  ks " << ks_b << " -> " << ks_new << " W/m/K  (x" << factor
                  << ", from TI " << ti_before << " SI)\n"
                  << "  kd " << kd_b << " -> " << kd_new
                  << " W/m/K  (same factor: kd/ks depth contrast preserved)"
                  << std::endl;
        // chi does not live on Planet: it is thermal::Config::chi, resolved at
        // the forwarding site as (chi_override >= 0 ? chi_override : 2.7).
        const double chi_eff = (chi_override >= 0.0) ? chi_override : 2.7;
        const bool cp_varies  = !planet.cpCoeff.empty();
        const bool chi_active = (chi_eff > 0.0);
        if (cp_varies || chi_active) {
            std::cout << "  NOTE: this body has "
                      << (cp_varies ? "a temperature-dependent cp" : "")
                      << ((cp_varies && chi_active) ? " and " : "")
                      << (chi_active ? "a radiative conductivity term chi = " : "");
            if (chi_active) std::cout << chi_eff;
            std::cout << ", so the delivered thermal inertia varies with\n"
                         "        temperature about the requested value. Use "
                         "--constant-thermal-inertia for a TI that is exactly\n"
                         "        constant in temperature and depth."
                      << std::endl;
        }
    }

    // ========================================================================
    // PHASE 4c: Constant thermal inertia mode
    // ========================================================================
    // Optional scalar property overrides (sensitivity sweeps). Applied to the
    // finalized Planet (Moon/SPICE path included) BEFORE the constant-TI block
    // and BEFORE illumination albedo sync (~line 8814), so albedo flows to
    // face_albedos / radiosity reflectance and ks/kd feed the constant-TI cp.
    // Sentinel <0 leaves the body default untouched -> baseline runs unchanged.
    //
    // PTM-53: these banners describe the REQUEST, not the outcome. What the
    // solver ends up using is echoed separately by applyConfiguredThermo()
    // ("[Thermo override] applied at solver: ..."), and a run that requests an
    // override no solver ever applies is FAILED at the tripwire below. Do not
    // reword these back into a claim of success.
    // ========================================================================
    {
        const auto& tp = sim_config.thermophysical;
        if (tp.albedo_override >= 0.0) {
            std::cout << "Property override (requested): albedo A0 " << planet.albedo
                      << " -> " << tp.albedo_override << std::endl;
            planet.albedo = tp.albedo_override;
        }
        if (tp.emissivity_override > 0.0) {
            std::cout << "Property override (requested): emissivity " << planet.emissivity
                      << " -> " << tp.emissivity_override << std::endl;
            planet.emissivity = tp.emissivity_override;
        }
        if (tp.ks_override > 0.0) {
            std::cout << "Property override (requested): ks " << planet.ks
                      << " -> " << tp.ks_override << " W/m/K" << std::endl;
            planet.ks = tp.ks_override;
        }
        if (tp.kd_override > 0.0) {
            std::cout << "Property override (requested): kd " << planet.kd
                      << " -> " << tp.kd_override << " W/m/K" << std::endl;
            planet.kd = tp.kd_override;
        }

        // Directional-hemispheric (Hapke) albedo gate. Opt-in; default OFF ->
        // untouched baseline. When on, the single-scattering albedo w is taken
        // from config (albedo_hapke_w) or derived from the body Bond albedo A0
        // so the hemispherically-averaged A_h reproduces A0 at normal incidence
        // (A_h(mu0=1) = 1 - gamma*3/(1+2*gamma) = A0). That inverse is exact in
        // closed form (hapkeWFromA0), so the 60-step bisection is gone. This
        // BODY-level w now only sets the gate flag and the log line: per-face w
        // (hapkeWFromA0 on each face's map albedo) supersedes it at every flux
        // site, anchoring each face's Hapke curve to its OWN albedo.
        if (tp.albedo_hapke) {
            double w = tp.albedo_hapke_w;
            if (w < 0.0) {
                double A0 = planet.albedo;
                w = hapkeWFromA0(A0);
                std::cout << "Hapke albedo gate: body single-scattering w = "
                          << w << " from Bond albedo A0 = " << A0
                          << " (per-face w supersedes at each flux site)" << std::endl;
            } else {
                std::cout << "Hapke albedo gate: single-scattering w = " << w
                          << " (from config)" << std::endl;
            }
            planet.use_hapke_albedo = true;
            planet.hapke_albedo_w = std::max(0.0, std::min(1.0, w));
        }
    }

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
        } else {
            // Rule 7: this used to fall through silently, leaving the run at the
            // body defaults while the user believed a constant TI was in force.
            std::cerr << "\nFATAL: thermophysical.constant_thermal_inertia = " << constant_ti
                      << " requires ks > 0 and rhos > 0 to derive cp = TI^2/(ks*rhos),\n"
                      << "  but the resolved body has ks = " << ks_val
                      << " and rhos = " << rhos_val << ".\n"
                      << "  Refusing to run with the body defaults under a constant-TI banner."
                      << std::endl;
            return 1;
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
    // Hoisted out of the procedural branch below so the provenance record can
    // say whether the topographic mesh was actually LOADED, not merely asked
    // for -- `use_topographic: true` silently falls back to a plain icosphere.
    bool topo_loaded = false;

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

        // PTM-17: a REQUESTED mesh import that fails is FATAL. Never silently substitute
        // a default icosphere for a specified import_file -- that runs the simulation on
        // the wrong geometry without warning. (The legitimate procedural-icosphere path is
        // the `else` branch below, taken only when no import_file is given.)
        if (!using_fallback_mesh && (!meshUtils.getMesh() || meshUtils.getMesh()->faces.empty())) {
            std::cerr << "FATAL: Failed to load mesh from file '" << mesh_file
                      << "'. Refusing to substitute a default icosphere for a requested "
                         "import. Check the path/format (legacy VTK 5.1 BINARY is not "
                         "supported -- convert to PLY)." << std::endl;
            return 1;
        }
    } else {
        // Procedural mesh path
        meshUtils.setMeshResolution(mesh_resolution);

        // Try loading a pre-computed topographic mesh if enabled
        if (sim_config.mesh.use_topographic) {
            std::string topo_dir = sim_config.mesh.topo_directory;
            if (topo_dir.empty()) {
                topo_dir = "data/topo";
            }
            topo_loaded = meshUtils.checkForTopographicMesh(
                topo_dir, mesh_resolution, sim_config.time.body);
            if (!topo_loaded) {
                // Fail loud rather than silently substituting a smooth sphere:
                // a topography-requested run on the wrong geometry is a wasted
                // (and misleading) simulation. Name the fix.
                std::string body_lc = sim_config.time.body;
                for (auto& c : body_lc) c = std::tolower(c);
                std::cerr << "ERROR: mesh.use_topographic is set, but no topography grid for "
                          << sim_config.time.body << " ico" << mesh_resolution
                          << " was found under '" << topo_dir << "' (expected e.g. "
                          << topo_dir << "/" << body_lc << "/topo_" << body_lc
                          << "_ico" << mesh_resolution << ".bin).\n";
                if (body_lc == "moon")
                    std::cerr << "  Fetch it with: python3 scripts/fetch_data.py --dataset moon-topo\n";
                std::cerr << "  Data location contract: data/README.md, docs/ENVIRONMENT.md ($PTM_DATA_ROOT)."
                          << std::endl;
                return 1;
            }
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

    // PTM-37: state the inferred mesh length unit (and the body radius that
    // does or does not corroborate it), and refuse to run when the inference
    // is a coin flip. See inspectMeshLengthUnit for the three fatal cases.
    std::string mesh_length_unit = "unknown";
    double mesh_median_radius = 0.0;
    {
        const double R_for_units = using_generic_body
            ? generic_body.shape.meanRadiusKm()
            : planet.body_radius_km;
        MeshUnitReport units = inspectMeshLengthUnit(mesh, R_for_units);
        mesh_length_unit = units.meters ? "meters" : "kilometers";
        mesh_median_radius = units.median;
        if (units.fatal) {
            // Do NOT recommend mesh.scale_factor here. It scales the vertices
            // and leaves face_centers/face_areas/face_normals at the loader's
            // values (PTM-65), which is precisely the internal inconsistency
            // case 0 above refuses -- so the advice would send the operator
            // straight back into this same FATAL, one case earlier.
            std::cerr << "FATAL [mesh-units]: " << units.message << "\n"
                      << "  Refusing to run. Rescale the mesh FILE so its face centers "
                         "are unambiguously meters or kilometers (regenerate it, or "
                         "rewrite the vertices with a mesh tool), or supply the body "
                         "radius via the generic-body shape block.\n"
                      << "  mesh.scale_factor will NOT fix this: it scales the vertices "
                         "only and leaves face centers/areas/normals unscaled, which is "
                         "itself a fatal inconsistency here (PTM-65).\n"
                      << "  See docs/KNOWN_LIMITATIONS.md (PTM-37, PTM-36, PTM-65)."
                      << std::endl;
            return 1;
        }
    }

    // ========================================================================
    // Run provenance
    // ========================================================================
    // Until this existed, metadata.txt was the ONLY thing the engine wrote
    // about a run, and it named no version, no commit, no checksum, no body,
    // no mesh unit and no configuration -- so which binary produced a given
    // output directory was not answerable from the directory. Written HERE, at
    // the first point where every recorded field is known and before any of
    // the expensive work, so that a run which dies in the solver still leaves
    // a record of what it was.
    //
    // Two new files, no existing format touched. See tools/run_provenance.h.
    {
        ptm::RunProvenance prov;
        prov.output_dir  = output_dir;
        prov.config_file = has_config_file ? config_file : std::string();
        prov.argv.assign(argv, argv + argc);
        prov.body = using_generic_body ? generic_body.name : sim_config.time.body;
        prov.generic_body = using_generic_body;
        if (!mesh_file.empty()) {
            prov.mesh_source = mesh_file;
        } else if (topo_loaded) {
            prov.mesh_source = "topographic sphere, resolution "
                             + std::to_string(mesh_resolution);
        } else {
            prov.mesh_source = "icosphere, resolution "
                             + std::to_string(mesh_resolution);
        }
        prov.mesh_faces        = mesh->faces.size();
        prov.mesh_length_unit  = mesh_length_unit;
        prov.mesh_median_radius = mesh_median_radius;
        ptm::writeRunProvenance(prov);
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
            early_rad_config.rms_slope_deg = rad_rms_slope_deg;
            early_rad_config.beam_vf_threshold = rad_beam_vf_threshold;
            early_rad_config.radiance_prune_eps = rad_radiance_prune_eps;
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

                // Save view factors to disk immediately and honor the return value.
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
    // Load SPICE kernels for the CONFIGURED body/metakernel. This must run after
    // setPlanetaryBody()/setGenericBody() above so the correct per-body metakernel
    // (e.g. moon.tm, not the europa.tm default) is furnsh'd. A failure here (missing
    // or unresolved kernels) is fatal: without an ephemeris the absolute epoch and
    // Sun/primary positions cannot be computed. Do not proceed with rc=0 (BUGS.md PTM-27).
    if (!illumination_model->initialize_spice(verbose)) {
        std::cerr << "\nFATAL: SPICE initialization failed for body '" << planet.name << "'.\n"
                  << "  Cannot resolve the absolute epoch/ephemeris without SPICE kernels.\n"
                  << "  Populate the kernel directory with:  python3 scripts/fetch_kernels.py\n"
                  << "  (see INSTALL.md, section \"SPICE kernel data\")." << std::endl;
        return 1;
    }
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

    // ---- Equilibration length: configured == integrated ---------------------
    // thermal.equilibration_orbits is the INTEGRATED number of diurnal cycles
    // of real, ephemeris-driven spin-up. thermal.equilibration_cycles is the
    // number of replays used to reach it (GPU persistent solver only), so the
    // assembled forcing block is orbits/passes and passes*block == orbits at
    // every pass count. See kDefaultEquilibrationPasses for the full note and
    // the before/after measurements.
    std::string eq_passes_source;
    const int   eq_passes_requested = resolveEquilibrationPassesValue(
        sim_config.thermal.equilibration_cycles, nullptr, &eq_passes_source);
    if (eq_passes_requested <= 0) {
        std::cerr << "Error: invalid thermal.equilibration_cycles; aborting rather "
                     "than silently equilibrating for a different length than the "
                     "config states.\n";
        return 1;
    }
    const bool eq_gpu_thermal = config.getPropertyBool("use_gpu_thermal", false);
    // Clamped once, then used for every derived quantity and every message, so
    // the block length, the notices and the banner cannot disagree about what
    // the configured length was.
    const double eq_orbits_configured = std::max(0.0, equilibration_orbits);
    // Only the GPU persistent solver has a pass loop. The CPU scalar and SIMD
    // equilibrators integrate their forcing block exactly once, so on those
    // paths the SAME integrated length is delivered by one block of the full
    // configured size. That is a larger flux array, not a different spin-up —
    // warn (it costs RAM and illumination time) but never change the length.
    int eq_passes_effective = eq_passes_requested;
    if (!eq_gpu_thermal && eq_passes_requested > 1) {
        std::cout << "[Equilibration] NOTE: equilibration_cycles="
                  << eq_passes_requested << " is a GPU-persistent-solver memory "
                     "optimization (replay a short forcing block). This run uses a "
                     "CPU solver, which has no pass loop, so the same "
                  << eq_orbits_configured << " integrated diurnal cycles are "
                     "delivered as ONE block of that length instead of "
                  << eq_passes_requested << " replays of "
                  << (eq_orbits_configured / eq_passes_requested)
                  << ". The spin-up LENGTH is unchanged; the equilibration flux "
                     "array is " << eq_passes_requested << "x larger and the "
                     "forcing is continuous rather than replayed, so results will "
                     "differ from the GPU run of this config.\n";
        eq_passes_effective = 1;
    }
    const double eq_block_cycles = eq_orbits_configured
                                 / static_cast<double>(eq_passes_effective);
    const double eq_block_frac =
        std::fabs(eq_block_cycles - std::round(eq_block_cycles));
    if (eq_passes_effective > 1 && eq_block_frac > 1e-9) {
        std::cerr << "[Equilibration] WARNING: equilibration_orbits="
                  << eq_orbits_configured << " / equilibration_cycles="
                  << eq_passes_effective << " gives a forcing block of "
                  << eq_block_cycles << " diurnal cycles, which is not a whole "
                     "number. Each replay then restarts at a different point in "
                     "the diurnal cycle than the one it ended on, which is a "
                     "discontinuity in the forcing the solver sees. Prefer an "
                     "equilibration_orbits that is a multiple of "
                     "equilibration_cycles.\n";
    }
    // Calculate equilibration period using the correct synodic day.
    double equilibration_days = eq_block_cycles * synodic_day;

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

    // ---- Spin-up seed, resolved ONCE for every solver path ------------------
    // Precedence: PTM_SPINUP_SEED env > thermal.spinup_seed > "physical".
    // An unparseable value is FATAL: the seed is worth 46.7 K on a permanently
    // shadowed column, so it must never be chosen for the user by a fallback.
    thermal::EqSeedMode resolved_seed_mode = thermal::EqSeedMode::Physical;
    {
        std::string seed_str = sim_config.thermal.spinup_seed;
        std::string seed_src = "thermal.spinup_seed";
        if (const char* e = std::getenv("PTM_SPINUP_SEED")) {
            if (*e != '\0') {
                if (seed_str != std::string(e)) {
                    std::cerr << "[eq-seed] PTM_SPINUP_SEED=" << e
                              << " overrides thermal.spinup_seed: " << seed_str
                              << std::endl;
                }
                seed_str = e;
                seed_src = "PTM_SPINUP_SEED";
            }
        }
        if (!thermal::parseEqSeedMode(seed_str, resolved_seed_mode)) {
            std::cerr << "FATAL: unknown " << seed_src << " '" << seed_str
                      << "'. Valid: physical | legacy.\n";
            return 1;
        }
    }
    double resolved_analytic_cycles_cfg = sim_config.thermal.analytic_seed_cycles;
    if (const char* e = std::getenv("PTM_ANALYTIC_SEED_CYCLES")) {
        if (*e != '\0') {
            try {
                const double v = std::stod(e);
                if (v < -1.0) throw std::invalid_argument("negative");
                std::cerr << "[eq-seed] PTM_ANALYTIC_SEED_CYCLES=" << v
                          << " overrides thermal.analytic_seed_cycles: "
                          << resolved_analytic_cycles_cfg << std::endl;
                resolved_analytic_cycles_cfg = v;
            } catch (const std::exception&) {
                std::cerr << "FATAL: PTM_ANALYTIC_SEED_CYCLES='" << e
                          << "' is not a number >= -1.\n";
                return 1;
            }
        }
    }
    const double resolved_analytic_cycles = thermal::analyticSeedCycles(
        resolved_seed_mode, resolved_analytic_cycles_cfg,
        planet.day, planet.year, kCpuAnalyticSeedYears);
    config.spinup_seed          = resolved_seed_mode;
    config.analytic_seed_cycles = resolved_analytic_cycles_cfg;

    // Log the calculation for verification
    std::cout << "\n========== Thermal Equilibration Configuration ==========" << std::endl;
    std::cout << "Body: " << planet.name << std::endl;
    std::cout << "Synodic day (solar day): " << synodic_day << " Earth days" << std::endl;
    // Equilibration length, stated so the configured number, the integrated
    // number and the forcing block can be read off one banner on EVERY solver
    // path. Before 2026-08-11 this printed only equilibration_orbits, which on
    // the GPU was half of what the run actually integrated (BUGS.md PTM-146).
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Equilibration (configured): " << eq_orbits_configured
              << " day/night cycles  (thermal.equilibration_orbits)" << std::endl;
    std::cout << "Equilibration (integrated): "
              << (eq_block_cycles * eq_passes_effective)
              << " day/night cycles = " << eq_passes_effective << " pass(es) x "
              << eq_block_cycles << "-cycle forcing block  ["
              << eq_passes_source
              << (eq_passes_effective != eq_passes_requested
                      ? ", downgraded to 1 pass on this CPU path" : "")
              << "]" << std::endl;
    std::cout << "Spin-up seed: " << thermal::eqSeedModeName(resolved_seed_mode)
              << (resolved_seed_mode == thermal::EqSeedMode::Physical
                      ? "  (shared per-face conductive steady profile, identical"
                        " on every solver path)"
                      : "  (pre-2026-08-11 per-path seeds + analytic"
                        " pre-conditioner)")
              << std::endl;
    if (!eq_gpu_thermal) {
        // The CPU scalar and SIMD equilibrators can additionally run an
        // analytic clear-sky pre-conditioner that equilibration_orbits does not
        // count and the GPU path does not have. It is OFF by default now; print
        // its length whenever it is on, because it dominates the real prefix on
        // short lunar configs and it OVERWRITES the per-face seed.
        // The two CPU paths do NOT agree on its length for a generic body: the
        // SIMD batch path uses the equilibration window (ptm_thermal_sim.cpp,
        // "using_generic_body" branch) while the scalar Model::run always uses
        // NYEARSEQ * round(year/day), which for an asteroid is thousands of
        // rotations. Report whichever this run will take.
        const bool simd_path = config.getPropertyBool("use_simd_batch", true);
        double seed_cycles = resolved_analytic_cycles;
        if (resolved_analytic_cycles_cfg < 0.0 && simd_path && using_generic_body
            && resolved_seed_mode == thermal::EqSeedMode::Legacy) {
            seed_cycles = eq_block_cycles;  // generic + SIMD: the eq window
        }
        if (seed_cycles > 0.0) {
            std::cout << "Equilibration (analytic seed, CPU paths only): "
                      << seed_cycles << " day/night cycles  (clear-sky forcing, "
                         "no topography or shadowing; thermal.analytic_seed_cycles)"
                      << std::endl;
        } else {
            std::cout << "Equilibration (analytic seed, CPU paths only): 0 "
                         "day/night cycles  (OFF -- spin-up is the configured "
                         "equilibration under the REAL forcing)" << std::endl;
        }
        if (!simd_path && using_generic_body && seed_cycles > 1000.0) {
            std::cerr << "[Equilibration] WARNING: the CPU scalar path's analytic "
                         "seed for this generic body is " << seed_cycles
                      << " rotations (NYEARSEQ x round(year/day)), which is a "
                         "body-year of spin-up, not a spin-up length anyone asked "
                         "for. The SIMD batch path uses the equilibration window "
                         "instead. Prefer the SIMD path for generic bodies.\n";
        }
    }
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

    // Publish the integration (statistics) window NOW, before any flux table is
    // built, so the effective-albedo accumulators on both the CPU and the GPU
    // thermal paths reduce over exactly the window every other statistics/*.bin
    // uses (Fermi condition C2). Same helper the statistics export calls below;
    // the export cross-checks the two and warns if they ever disagree.
    {
        const IntegrationWindow aeff_iw = computeIntegrationWindow(
            equilibration_steps, total_time_steps, output_steps_per_day,
            total_simulation_days, integration_days, output_days,
            integration_steps_cli, integration_offset_steps_cli,
            integration_offset_days);
        setEffectiveAlbedoWindow(aeff_iw.start, aeff_iw.steps);
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

    // Occluder-only geometry: a coarse ring/annulus (e.g. LOLA out to the
    // physics horizon) added to the shadow-ray BVH so distant topography
    // shadows the SfS core. NOT thermally solved / radiosity-coupled /
    // output — the sim still runs entirely on `mesh` (the core). When
    // mesh.occluder_file is empty this is skipped and behavior is
    // byte-identical. See docs/extended-occluder-architecture.md.
    if (!sim_config.mesh.occluder_file.empty()) {
        const std::string& occ_file = sim_config.mesh.occluder_file;
        std::cout << "Loading occluder mesh (shadow-only): " << occ_file << std::endl;
        std::shared_ptr<Mesh> occ_mesh;
        MeshUtils occMU;
        if (occMU.importTriangulatedMesh(occ_file, mesh_scale_factor)) {
            occ_mesh = occMU.getMesh();
        } else {
            occ_mesh = std::make_shared<Mesh>();
            if (!loadAsciiDTMPLY(occ_file, occ_mesh)) {
                occ_mesh.reset();
            } else if (mesh_scale_factor != 1.0) {
                for (auto& v : occ_mesh->vertices) {
                    v[0] *= mesh_scale_factor;
                    v[1] *= mesh_scale_factor;
                    v[2] *= mesh_scale_factor;
                }
            }
        }
        if (occ_mesh && !occ_mesh->vertices.empty() && !occ_mesh->faces.empty()) {
            illumination_model->setOccluderGeometry(occ_mesh->vertices,
                                                    occ_mesh->faces);
            std::cout << "Occluder mesh registered: "
                      << occ_mesh->vertices.size() << " verts, "
                      << occ_mesh->faces.size() << " tris (shadow BVH only)\n";
        } else {
            std::cerr << "WARNING: occluder_file set but could not be loaded ("
                      << occ_file << "); proceeding WITHOUT distant-terrain "
                      << "shadowing.\n";
        }
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

    // Fix A (BUGS.md PTM-28): build the per-timestep instantaneous solar constant
    // table used by the incidence-angle albedo (Hayne + Hapke). direct_solar was
    // formed as (SOLAR_CONSTANT/dist_au^2)*<cos>, so recovering cos with the same
    // instantaneous S keeps cos<=1 and removes the super-solar-constant albedo
    // plateau + Hapke C1-kink -> GPU sawtooth. Default ON; the legacy env leaves
    // the table EMPTY so every incidence site falls back to the fixed planet.S.
    g_solar_irr_per_step.clear();
    if (legacyFixedSolarConst()) {
        std::cerr << "[solar-const] PTM_LEGACY_FIXED_SOLARCONST=1: incidence albedo "
                     "uses the FIXED 1-AU solar constant (legacy; PTM-28 sawtooth "
                     "may reappear near perihelion under GPU+Hapke)\n";
    } else {
        g_solar_irr_per_step.assign(static_cast<size_t>(total_time_steps), 0.0);
        size_t n_ok = 0; double s_min = 1e30, s_max = 0.0;
        for (int t = 0; t < total_time_steps; ++t) {
            if (static_cast<size_t>(t) < sun_positions.size()) {
                const auto& sp = sun_positions[t];
                const double d_km = std::sqrt(sp[0]*sp[0] + sp[1]*sp[1] + sp[2]*sp[2]);
                const double d_au = d_km / constants::KM_PER_AU;
                if (d_au > 0.0) {
                    const double s = constants::SOLAR_CONSTANT / (d_au * d_au);
                    g_solar_irr_per_step[t] = s;
                    s_min = std::min(s_min, s); s_max = std::max(s_max, s);
                    ++n_ok;
                }
            }
        }
        if (n_ok != static_cast<size_t>(total_time_steps)) {
            std::cerr << "[solar-const] WARNING: only " << n_ok << "/" << total_time_steps
                      << " timesteps have a valid sun distance; the rest fall back to the "
                         "fixed 1-AU solar constant (PTM-28 incidence fix partially inactive)\n";
        }
        if (n_ok > 0) {
            std::cerr << "[solar-const] incidence albedo uses the INSTANTANEOUS heliocentric "
                         "solar constant (PTM-28 fix, default ON); S_inst range ["
                      << std::fixed << std::setprecision(1) << s_min << ", " << s_max
                      << "] W/m^2\n";
        }
    }

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
    // Max illumination footprint (GB) to EXTRACT into RAM; above this the thermal
    // solver streams per-(face,timestep) from the in-RAM chunk store, which re-reads
    // through every equilibration cycle and loses the flat-array fast path (~100x
    // slower). Default 10 GB keeps very large meshes (e.g. lunar 500k, ~180 GB) on
    // the streaming path. Override with PTM_RAM_ILLUM_MAX_GB to force smaller (ico6
    // asteroid, ~15 GB) meshes to RAM-extract on high-memory boxes. Default unchanged.
    double STREAM_THRESHOLD_GB = 10.0;
    if (const char* ev = std::getenv("PTM_RAM_ILLUM_MAX_GB")) {
        double ev_gb = std::atof(ev);
        if (ev_gb > 0.0) STREAM_THRESHOLD_GB = ev_gb;
    }
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

    // BUGS.md PTM-76: the T(z) checkpoint is read and written ONLY by the GPU
    // persistent solver. On any CPU path both keys are dropped — no file, no
    // error — so a chunked / warm-start run silently degrades into a set of
    // independent COLD-STARTED runs that still exit 0. Refuse: a request this
    // consequential must not be answered with a different simulation.
    if (!config.getPropertyBool("use_gpu_thermal", false)
        && (!sim_config.thermal.initial_state_file.empty()
            || !sim_config.thermal.final_state_file.empty())) {
        std::cerr << "\nFATAL: thermal.initial_state_file / thermal.final_state_file are set,"
                     " but this run will not take the GPU persistent solver.\n"
                  << "  Only that solver reads or writes the T(z) checkpoint (BUGS.md PTM-76);"
                     " on a CPU path the request is silently dropped and every chunk"
                     " cold-starts.\n"
                  << "  Set thermal.use_gpu: true, or drop the checkpoint keys and run the"
                     " window in one piece.\n"
                  << "  Refusing to run a chunked workflow that cannot hand state over."
                  << std::endl;
        return 1;
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
        std::cout << "  Average non-zero illumination: " << (total_illumination / non_zero_illumination) << " W/m^2" << std::endl;
    }

    // DON'T filter illumination yet - run thermal model with full dataset first
    std::vector<std::vector<double>> surface_temps;

    // ======================================================================
    // Equilibration-phase indirect flux (docs/radiation-exchange.md 7.7)
    // ======================================================================
    // Per-global-face time-mean indirect flux <q_ind>[f] added to the
    // EQUILIBRATION-phase forcing, so the deep column equilibrates consistent
    // with the inter-face infrared it actually receives during the output
    // window. Without it, a face whose entire forcing is neighbor infrared
    // equilibrates to the bare geothermal floor and then warms through the
    // output window, biasing the window mean cold.
    //
    // Two sources, in precedence order:
    //   1. thermal.eq_indirect_flux_file -- an explicit 'QIND' file from a
    //      prior pass-1 run. Always wins when set.
    //   2. thermal.eq_indirect_source: auto (default) -- the engine estimates
    //      it internally inside the radiation-coupling block below.
    // "off" restores the pre-2026-08 behavior on both.
    std::string eq_indirect_source = sim_config.thermal.eq_indirect_source;
    if (const char* e = std::getenv("PTM_EQ_INDIRECT_SOURCE")) {
        if (*e != '\0' && eq_indirect_source != std::string(e)) {
            std::cout << "[Deep coupling] PTM_EQ_INDIRECT_SOURCE=" << e
                      << " overrides thermal.eq_indirect_source: "
                      << eq_indirect_source << std::endl;
            eq_indirect_source = e;
        }
    }
    if (eq_indirect_source != "auto" && eq_indirect_source != "off") {
        std::cerr << "FATAL: unknown thermal.eq_indirect_source '"
                  << eq_indirect_source << "'. Valid: auto | off.\n";
        return 1;
    }
    // Provenance for the end-of-run banner. Never left as a guess: whichever
    // branch supplies the vector rewrites it, and a branch that declines says
    // why. Reported even when nothing was applied.
    EqIndirectRecord eq_indirect_record;
    if (!sim_config.thermal.eq_indirect_flux_file.empty()
            && eq_indirect_source == "off") {
        std::cout << "[Deep coupling] thermal.eq_indirect_flux_file is set AND "
                     "eq_indirect_source: off. The explicit file WINS; the "
                     "equilibration will carry its indirect flux." << std::endl;
    }
    std::vector<double> eq_indirect_vec;
    if (!sim_config.thermal.eq_indirect_flux_file.empty()) {
        const std::string& path = sim_config.thermal.eq_indirect_flux_file;
        std::ifstream qin(path, std::ios::binary);
        if (!qin) {
            std::cerr << "Warning: cannot open eq_indirect_flux_file '" << path
                      << "' — equilibration will not include indirect flux." << std::endl;
        } else {
            uint32_t magic = 0, nf = 0;
            qin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            qin.read(reinterpret_cast<char*>(&nf), sizeof(nf));
            if (magic != 0x51494E44u) {  // 'QIND'
                std::cerr << "Warning: eq_indirect_flux_file '" << path
                          << "' bad magic (0x" << std::hex << magic << std::dec
                          << "); ignoring." << std::endl;
            } else if (nf != mesh->faces.size()) {
                std::cerr << "Warning: eq_indirect_flux_file face count " << nf
                          << " != mesh faces " << mesh->faces.size()
                          << "; ignoring." << std::endl;
            } else {
                eq_indirect_vec.resize(nf);
                qin.read(reinterpret_cast<char*>(eq_indirect_vec.data()),
                         static_cast<std::streamsize>(nf) * sizeof(double));
                double s = 0.0, mx = 0.0;
                for (double q : eq_indirect_vec) { s += q; mx = std::max(mx, q); }
                std::cout << "[Deep coupling] Loaded eq-indirect flux for " << nf
                          << " faces from " << path << " (mean "
                          << (s / std::max<uint32_t>(1, nf)) << ", max " << mx
                          << " W/m^2) — added to equilibration forcing." << std::endl;
                eq_indirect_record.census(eq_indirect_vec);
                eq_indirect_record.provenance = "file: " + path;
            }
        }
    }

    // Time-varying indirect flux per face [face][timestep]
    // Declared at this scope so it's accessible for location export later
    std::vector<std::vector<double>> indirect_flux_series;

    // Band-separated ABSORBED indirect radiosity flux (Paul's product): the
    // multiply-scattered SOLAR (visible) and thermal-IR channels of the total
    // held in indirect_flux_series, captured per timestep from the two-band
    // solve. Same [face][ABSOLUTE step] layout as indirect_flux_series; left
    // empty for gray / radoff runs (so those runs' outputs stay byte-identical).
    // With no Smith beaming, indirect_solar_series + indirect_ir_series ==
    // indirect_flux_series exactly (== the PTM_DUMP_QIND_SERIES 'QIDT' total).
    std::vector<std::vector<double>> indirect_solar_series;  // (1-A)·Σ F·B_vis
    std::vector<std::vector<double>> indirect_ir_series;     //   ε·Σ F·B_ir
    // Set true once a two-band solve populated the per-timestep bands.
    bool indirect_bands_available = false;
    // Set true (once, with a warning) if any timestep went through the multi-GPU
    // pool path, which runs GRAY radiosity and cannot produce the band split —
    // the bands are then OMITTED rather than written as misleading zeros.
    bool indirect_bands_batch_gap = false;

    // Whole-mesh per-face T(z) at the END OF ITER-0'S EQUILIBRATION
    // PHASE — i.e., the initial state iter-0 used for its output cycle.
    // Filled by iter-0 when rad coupling is enabled; consumed by every
    // rad-coupling re-thermal iter to skip its own eq phase and start
    // from exactly iter-0's converged state. Layer-major SoA matching
    // ThermalOpenCLSolver::buf_T_: [layer * n_faces + face_idx]. Stays
    // empty for rad-off runs (never allocated).
    std::vector<float> rad_post_eq_T;

    // TRUE only after the whole-mesh solve that produced `surface_temps` was
    // runThermalModelsForAllFaces_GPU_Persistent WITH rad coupling on — i.e.
    // the one and only code path that fills `rad_post_eq_T`. Set at the call
    // sites below, next to the `&rad_post_eq_T` argument itself, so the flag
    // cannot drift from the mechanism it records.
    //
    // The PTM-78 seed tripwire is gated on THIS, not on `radiation.enabled`
    // (a user-set flag) and not on `want_post_coupling_state` (a GPU coupled
    // run without a checkpoint still needs the guard). BUGS.md PTM-94.
    bool rad_post_eq_T_expected = false;


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
        // Length of the CPU analytic clear-sky pre-conditioner, in body years:
        // NYEARSEQ * round(year/day) diurnal cycles (Moon 24, Europa 2). It is
        // NOT counted by thermal.equilibration_orbits and the GPU path has no
        // equivalent; the pre-flight banner prints it for that reason.
        // NYEARSEQ is now only the LEGACY length -- whether the pre-conditioner
        // runs at all is thermal.analytic_seed_cycles / thermal.spinup_seed.
        thermal_config.NYEARSEQ = kCpuAnalyticSeedYears;
        // Stefan-Boltzmann for every mesh run. This used to be 5.67e-8, a value
        // no other subsystem used, so the surface energy balance and the
        // radiosity that feeds it did not conserve energy with each other.
        thermal_config.sigma = ptm::kStefanBoltzmann;
        // The one shared spin-up seed. Resolved once above (env > config), and
        // carried on the config object that every solver entry point receives,
        // so no path can pick a different initial condition. The seed reads
        // config.sigma, so it now shares the single constant above -- which is
        // what drives the CPU-scalar-vs-SIMD seed residual to zero.
        thermal_config.spinup_seed          = resolved_seed_mode;
        thermal_config.analytic_seed_cycles = resolved_analytic_cycles_cfg;
        thermal_config.chi = (chi_override >= 0) ? chi_override : 2.7;  // Hayne et al. (2017) default

        // ---- Regolith conductivity law (thermophysical.conductivity_model) ----
        // Placed on thermal_config, which is built ONCE here and PASSED to every
        // solver entry point. It is deliberately NOT placed on `planet`: every
        // mesh/GPU solver entry point rebuilds its own thermal::Planet(body) and
        // discards main-scope Planet edits, which is exactly how ks_override/
        // kd_override became silent no-ops on the Moon path (BUGS.md PTM-53).
        {
            thermal::ConductivityModel cm = thermal::ConductivityModel::Hayne2017;
            if (!thermal::parseConductivityModel(
                    sim_config.thermophysical.conductivity_model, cm)) {
                std::cerr << "FATAL: unknown thermophysical.conductivity_model '"
                          << sim_config.thermophysical.conductivity_model
                          << "'. Valid: hayne2017 | martinez2021.\n";
                return 1;
            }
            thermal_config.conductivity_model = cm;

            // Print through a LOCAL stream. std::cout arrives here carrying
            // `std::fixed << std::setprecision(1)` from an earlier banner
            // (:10725 and friends), which rendered ks = 7.4e-4 as "0.0" and
            // H = 0.06 as "0.1" -- a run reporting numbers that are not the
            // numbers it used, i.e. exactly the PTM-05/PTM-55 archetype. A
            // local ostringstream starts in the default format and cannot be
            // affected by whatever std::cout's state happens to be.
            std::ostringstream cmb;
            if (cm == thermal::ConductivityModel::Martinez2021) {
                cmb << "Conductivity model: MARTINEZ2021 "
                             "(Martinez & Siegler 2021, JGR Planets 126, e2021JE006829, Eq. 10)\n"
                    << "  k(T,rho) = (A1*rho - A2)*k_am(T) + (B1*rho - B2)*T^3, "
                       "A1=5.0821e-6 A2=5.1e-3 B1=2.0022e-13 B2=1.953e-10\n"
                    << "  NOTE ks (" << planet.ks << " W/m/K) and kd (" << planet.kd
                    << " W/m/K) are INERT under this law: Eq. 10 depends on rho(z) only.\n"
                    << "  NOTE chi (" << thermal_config.chi
                    << ") is INERT too; the T^3 term uses the paper's own B(rho).\n"
                    << "  rho(z) profile still comes from the config: rhos="
                    << planet.rhos << " rhod=" << planet.rhod
                    << " H=" << planet.H << " m.\n";
                // The paper re-tuned H to 5.6 cm together with A(rho)/B(rho).
                // Running with a different H is legitimate (it isolates the
                // conductivity law) but is NOT a reproduction of their figures.
                if (std::abs(planet.H - 0.056) > 1e-9) {
                    cmb << "  NOTE H=" << planet.H << " m, not the paper's retuned "
                           "0.056 m. This isolates the conductivity law but does NOT "
                           "reproduce Martinez et al.'s published curves.\n";
                }
                // Eq. 10 returns a NEGATIVE conductivity below rho ~ 1003.5
                // kg/m^3. Refuse rather than integrate a negative diffusivity.
                const double rho_lo = std::min(planet.rhos, planet.rhod);
                if (!thermal::martinezDensityIsValid(rho_lo)) {
                    std::ostringstream err;
                    err << "FATAL: conductivity_model martinez2021 is invalid for "
                           "density " << rho_lo << " kg/m^3 -- Eq. 10 gives a NEGATIVE "
                           "conductivity below " << thermal::kMartinezRhoMin
                        << " kg/m^3. Raise rhos/rhod or use hayne2017.\n";
                    std::cout << cmb.str();
                    std::cerr << err.str();
                    return 1;
                }
                // ---- composition guard 1: constant_thermal_inertia (FATAL) ----
                // `constant_thermal_inertia` means "TI is constant in temperature
                // AND depth". It delivers that by SOLVING for cp from ks:
                //   cp = TI^2 / (ks * rhos)   (:9777)
                // and zeroing chi so that k == ks at every T. Under martinez2021
                // k is NOT ks -- it is Eq. 10, a function of T -- so that cp is
                // derived from a conductivity the run will not use, and the
                // delivered TI = sqrt(k_mart(T)*rho*cp) both misses the request
                // and varies with temperature. The two options contradict each
                // other by construction; there is no value of cp that satisfies
                // both. Refuse rather than silently deliver a different TI than
                // the one the operator asked for and the banner announced.
                const double c_ti = sim_config.thermophysical.constant_thermal_inertia;
                if (c_ti > 0.0) {
                    const double cp_c = (planet.ks > 0.0 && planet.rhos > 0.0)
                        ? (c_ti * c_ti) / (planet.ks * planet.rhos) : 0.0;
                    std::ostringstream err;
                    err.precision(4);
                    err << "FATAL: thermophysical.constant_thermal_inertia ("
                        << c_ti << " SI) and conductivity_model martinez2021 are "
                           "mutually contradictory.\n"
                           "  constant_thermal_inertia derives cp = TI^2/(ks*rhos) = "
                        << cp_c << " J/kg/K and zeroes chi so that k == ks = "
                        << planet.ks << " W/m/K at every temperature.\n"
                           "  martinez2021 ignores ks entirely: k = Eq.10(T, rho). The run "
                           "would therefore integrate TI = sqrt(k_mart(T)*rho*cp), which is "
                           "neither constant nor equal to the requested value:\n";
                    if (cp_c > 0.0) {
                        for (double Tp : {50.0, 100.0, 200.0, 380.0}) {
                            const double kg = thermal::martinezK(Tp, planet.rhos);
                            const double G  = std::sqrt(kg * planet.rhos * cp_c);
                            err << "      T = " << Tp << " K  ->  TI = " << G
                                << " SI  (" << ((G / c_ti - 1.0) * 100.0)
                                << " % vs the requested " << c_ti << ")\n";
                        }
                    }
                    err << "  Choose one: drop constant_thermal_inertia (set it to 0 and "
                           "give ks/kd/rhos/rhod/cp0 directly), or use "
                           "conductivity_model: hayne2017.\n";
                    std::cout << cmb.str();
                    std::cerr << err.str();
                    return 1;
                }

                // ---- composition guard 2: ks/kd overrides (WARNING) ----
                // ks_override/kd_override DO reach the solver (that is the PTM-53
                // fix) and DO print "Property override: ks ... -> ...", but Eq. 10
                // never reads ks or kd. Left unsaid, a conductivity sweep driven by
                // ks_override under martinez2021 would run N identical simulations
                // and report them as a sweep -- the PTM-53 archetype re-entering by
                // the other door.
                {
                    const auto& tpo = sim_config.thermophysical;
                    if (tpo.ks_override > 0.0 || tpo.kd_override > 0.0) {
                        cmb << "  WARNING: thermophysical.";
                        if (tpo.ks_override > 0.0 && tpo.kd_override > 0.0)
                            cmb << "ks_override (" << tpo.ks_override << ") and kd_override ("
                                << tpo.kd_override << ") are";
                        else if (tpo.ks_override > 0.0)
                            cmb << "ks_override (" << tpo.ks_override << ") is";
                        else
                            cmb << "kd_override (" << tpo.kd_override << ") is";
                        cmb << " set, but INERT under conductivity_model martinez2021:\n"
                               "           Eq. 10 depends on rho(z) and T only, so this run is "
                               "IDENTICAL to one without the override.\n"
                               "           A conductivity sweep parameterized by ks/kd will "
                               "produce N identical results here. Sweep rhos/rhod instead, or "
                               "use conductivity_model: hayne2017.\n";
                    }
                }

                // Outside the paper's fitted density span the law is an
                // extrapolation. Say so; do not stop.
                if (rho_lo < 1100.0 || std::max(planet.rhos, planet.rhod) > 1800.0) {
                    cmb << "  WARNING: rho span [" << rho_lo << ", "
                        << std::max(planet.rhos, planet.rhod)
                        << "] kg/m^3 extends outside the paper's fitted range "
                           "1100-1800 kg/m^3; Eq. 10 is extrapolated there.\n";
                }
            } else {
                cmb << "Conductivity model: HAYNE2017 (default) "
                       "k = kc(z)*(1 + chi*(T/350)^3), chi="
                    << thermal_config.chi << "\n";
            }
            std::cout << cmb.str() << std::flush;
        }

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
        // Phase R (auto-calibrate): forward the opt-out so the GPU solvers can read
        // it via getPropertyBool. Default true (on); PTM_NO_AUTO_CALIBRATE also opts
        // out at the solver.
        thermal_config.setProperty("auto_calibrate_gpu", sim_config.thermal.auto_calibrate_gpu);

        // Accuracy the spin-up ESTIMATE targets (report-only; the solvers
        // rebuild a local Config, so like every other engine knob it has to be
        // forwarded or it silently reverts to the built-in default).
        thermal_config.setProperty("eq_target_K",
                                   fullPrecision(sim_config.thermal.equilibration_target_K));

        // Flag generic bodies so albedo map code skips built-in body search
        if (using_generic_body) {
            thermal_config.setProperty("using_generic_body", "true");

            // Forward the fully-configured main Planet's thermophysical properties
            // so the solvers (which rebuild a local Planet from the body NAME and
            // otherwise fall back to Europa placeholder defaults) apply the real
            // ks / ks->kd depth profile / cp0 / rho / H / cpCoeff (and any
            // constant_thermal_inertia collapse already baked into `planet`).
            // Applied by applyConfiguredThermo() at each solver entry point;
            // gated on "gb_thermo" so built-in Moon/Europa runs stay byte-identical.
            // (User-requested scalar overrides are forwarded separately, below,
            //  on EVERY body -- see PTM-53.)
            thermal_config.setProperty("gb_thermo", "1");
            thermal_config.setProperty("gb_ks",         planet.ks);
            thermal_config.setProperty("gb_kd",         planet.kd);
            thermal_config.setProperty("gb_cp0",        planet.cp0);
            thermal_config.setProperty("gb_rhos",       planet.rhos);
            thermal_config.setProperty("gb_rhod",       planet.rhod);
            thermal_config.setProperty("gb_H",          planet.H);
            thermal_config.setProperty("gb_albedo",     planet.albedo);
            thermal_config.setProperty("gb_emissivity", planet.emissivity);
            // Also forward the orbital/rotational scalars. The solver entry points
            // rebuild a LOCAL Planet from the body NAME, which for a generic body
            // resolves to Europa placeholder defaults -- so without these the
            // subsurface grid and analytical equilibration used Europa's rotation
            // period (day), solar constant (S), orbital period (year) and Qb
            // instead of the generic body's. day drives the diurnal skin depth
            // (grid resolution) and the equilibration diurnal period; S and the
            // Bond albedo drive the equilibration forcing; year sets its duration.
            thermal_config.setProperty("gb_day",  planet.day);
            thermal_config.setProperty("gb_S",    planet.S);
            thermal_config.setProperty("gb_year", planet.year);
            thermal_config.setProperty("gb_Qb",   planet.Qb);
            std::string gb_cpc;
            for (size_t i = 0; i < planet.cpCoeff.size(); ++i) {
                if (i) gb_cpc += ",";
                gb_cpc += std::to_string(planet.cpCoeff[i]);
            }
            thermal_config.setProperty("gb_cpCoeff", gb_cpc);
        }

        // ====================================================================
        // PTM-53: forward the user's EXPLICIT scalar overrides to the solvers,
        // on named bodies (Moon/Europa/Mercury) as well as generic ones.
        // ====================================================================
        // Every mesh solver entry point rebuilds a local Planet from the body
        // NAME, so anything main() resolved onto `planet` is lost unless it is
        // forwarded. Before this block that forwarding existed only inside the
        // `using_generic_body` branch above, which made ks_override/kd_override/
        // emissivity_override -- and the whole constant_thermal_inertia collapse
        // -- silent no-ops on the Moon while the log announced them. A lunar
        // conductivity sweep measured nothing and said it had worked.
        //
        // Forwarded PER KNOB and only when the user actually set that knob, so a
        // config with no overrides forwards no keys and stays byte-identical.
        // The value sent is the RESOLVED planet.* (post constant-TI collapse),
        // not the raw request, so the solver sees exactly what main() computed.
        //
        // Values go through fullPrecision() rather than the Config double setter,
        // which truncates at 6 decimals (BUGS.md PTM-60).
        {
            const auto& tp = sim_config.thermophysical;
            auto fwd = [&thermal_config](const char* key, double value) {
                thermal_config.setProperty(key, fullPrecision(value));
            };
            if (thermoOverrideRequested(tp, geothermal_heat_flow)) {
                thermal_config.setProperty("ovr_thermo", "1");  // solver-side fast-path gate
            }
            // PTM-161. `geothermal_heat_flow` is a `thermal.*` knob, not a
            // `thermophysical.*` one, but it suffers the identical defect and
            // rides the identical channel. `planet.Qb` here is post-resolution
            // (main applies the CLI/YAML value at :11647/:11833), so the solver
            // sees exactly what main computed. Forwarded ONLY when the user set
            // it (> 0): the tracked lunar configs all carry -1.0, so they
            // forward no key and stay byte-identical.
            if (geothermal_heat_flow   > 0.0) fwd("ovr_Qb",         planet.Qb);
            if (tp.ks_override         > 0.0) fwd("ovr_ks",         planet.ks);
            if (tp.kd_override         > 0.0) fwd("ovr_kd",         planet.kd);
            if (tp.emissivity_override > 0.0) fwd("ovr_emissivity", planet.emissivity);
            if (tp.albedo_override    >= 0.0) fwd("ovr_albedo",     planet.albedo);
            // Rule 7: albedo_override is the one knob a per-face albedo MAP can
            // out-rank. When a map loads, its per-face value governs the
            // absorbed-solar flux and the override survives only as the
            // equilibration seed / uncovered-face fallback (<=0.51 K on Moon
            // ico5). Say so rather than let an albedo sweep quietly flatline.
            if (tp.albedo_override >= 0.0 && (use_albedo_map || !albedo_map_file.empty())) {
                std::cout << "WARNING: thermophysical.albedo_override = " << tp.albedo_override
                          << " is set AND a per-face albedo map is configured.\n"
                          << "  If the map loads, ITS per-face albedo governs the absorbed solar "
                             "flux; the override then only\n"
                          << "  sets the equilibration seed and the fallback for uncovered faces. "
                             "Check the albedo-load lines\n"
                          << "  below, and set use_albedo_map: false to sweep albedo directly "
                             "(docs/KNOWN_LIMITATIONS.md)." << std::endl;
            }
            if (tp.constant_thermal_inertia > 0.0) {
                // The collapse is cp0 + an emptied cp(T) polynomial + kd->ks +
                // rhod->rhos. chi=0 already reaches the solver via
                // thermal_config.chi; these four did not.
                fwd("ovr_cp0",  planet.cp0);
                fwd("ovr_kd",   planet.kd);
                fwd("ovr_rhod", planet.rhod);
                std::string cpc;
                for (size_t i = 0; i < planet.cpCoeff.size(); ++i) {
                    if (i) cpc += ",";
                    cpc += fullPrecision(planet.cpCoeff[i]);
                }
                thermal_config.setProperty("ovr_cpCoeff", cpc);
            }
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

        // Equilibration cycling (thermal.equilibration_cycles). Only set the
        // property when the pass count will actually be used, i.e. on the GPU
        // persistent solver → thermal_config is unchanged for every run that
        // does not mention the knob, and the solver falls through to
        // kDefaultEquilibrationPasses.
        //
        // On a CPU run main() has already downgraded the pass count to 1 and
        // said so out loud when it sized the equilibration window (the block is
        // then the full configured length, so the INTEGRATED spin-up is the
        // same). Setting the property here as well would leave a value in
        // thermal_config that nothing honors.
        //
        // ConfigParser::validate() is NOT called on this executable's path, so
        // the range check has to live here: a `> 0` guard alone would DROP an
        // out-of-range request (0, -5) without a word and run the default length
        // while the config claimed otherwise.
        if (sim_config.thermal.equilibration_cycles != -1) {
            if (sim_config.thermal.equilibration_cycles < 1) {
                std::cerr << "Error: thermal.equilibration_cycles must be >= 1 "
                             "(or -1 for the solver default); got "
                          << sim_config.thermal.equilibration_cycles
                          << ". Aborting rather than silently equilibrating for "
                             "a different length than the config states.\n";
                return 1;
            }
            if (config.getPropertyBool("use_gpu_thermal", false)) {
                thermal_config.setProperty("equilibration_cycles",
                                           sim_config.thermal.equilibration_cycles);
            }
        }
        // Pass-stride RRE extrapolation. Same "only when armed" rule as the
        // pass knob above: with method == "off" no property is set at all, so
        // thermal_config is byte-identical for every legacy run and the
        // resolver's default (disabled) applies.
        if (sim_config.thermal.eq_extrapolation.method != "off") {
            thermal_config.setProperty("eq_extrap_method",
                                       sim_config.thermal.eq_extrapolation.method);
            thermal_config.setProperty("eq_extrap_order_p",
                                       sim_config.thermal.eq_extrapolation.order_p);
            thermal_config.setProperty("eq_extrap_tol_K",
                                       sim_config.thermal.eq_extrapolation.tol_K);
            thermal_config.setProperty("eq_extrap_hold_K",
                                       sim_config.thermal.eq_extrapolation.hold_K);
        }
        {
            // The extrapolation lives in the GPU persistent solver's pass loop.
            // The CPU/SIMD equilibrators have no pass loop at all and would
            // ignore the request without a word. Refuse instead of pretending.
            bool extrap_requested = false;
            ptm::eqextrap::Options probe;
            // -1 = pass budget not known here, so this validates the KNOB only;
            // the solver re-resolves with the real budget and owns the
            // "too few passes" warning.
            if (!resolveEqExtrapolation(thermal_config, /*eq_passes=*/-1, &probe,
                                        &extrap_requested)) {
                std::cerr << "Error: invalid thermal.eq_extrapolation; aborting.\n";
                return 1;
            }
            if (extrap_requested &&
                !config.getPropertyBool("use_gpu_thermal", false)) {
                std::cerr << "Error: thermal.eq_extrapolation is implemented only "
                             "for the GPU persistent thermal solver "
                             "(thermal.use_gpu: true / --use-gpu-thermal). The CPU "
                             "and SIMD equilibrators have no pass loop and would "
                             "ignore it silently. Aborting rather than reporting an "
                             "equilibration that did not happen.\n";
                return 1;
            }
        }
        // NOTE: equilibration_cycles needs no check here. main() resolved and
        // validated it far above (where it sized the equilibration window) and
        // aborted on a malformed value, and on a CPU run it already downgraded
        // the pass count to 1 and said so — under the old semantics that was a
        // hard refusal, which was right when equilibration_orbits was the BLOCK
        // length, but the knob can no longer change the integrated length.
        // Re-resolving here would also re-print the PTM_EQUILIBRATION_CYCLES
        // override notice a second time.

        // The equilibration forcing dump reduces the GPU persistent solver's own
        // prepare_flux output. The CPU/SIMD equilibrators build their forcing
        // elsewhere and would produce no file at all, so refuse instead of
        // letting the run proceed to a full thermal solve the caller did not
        // want and then exit 0 with nothing written.
        if (!eqlundump::requestedDir().empty() &&
            !config.getPropertyBool("use_gpu_thermal", false)) {
            std::cerr << "Error: PTM_DUMP_EQ_LUNATION_MEAN is implemented only for "
                         "the GPU persistent thermal solver (thermal.use_gpu: true / "
                         "--use-gpu-thermal). The CPU and SIMD paths would ignore it "
                         "silently and run a full simulation instead. Aborting.\n";
            return 1;
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

        // Propagate the directional-hemispheric (Hapke) albedo gate into
        // thermal_config. The gate is resolved imperatively onto the main-scope
        // `planet` (single-scattering w derived/validated at the config block
        // ~line 8548); the GPU thermal solver rebuilds its OWN Planet from
        // config properties and would otherwise never see the gate. Only set
        // these when the gate is on → thermal_config stays byte-identical for
        // every non-Hapke run (default OFF path unchanged for Moon3D/A3GT/
        // Europa/gate-off EMA).
        if (planet.use_hapke_albedo && planet.hapke_albedo_w >= 0.0) {
            thermal_config.setProperty("use_hapke_albedo", true);
            thermal_config.setProperty("hapke_albedo_w", planet.hapke_albedo_w);
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

        // ------------------------------------------------------------------
        // Whole-mesh initial thermal solve, callable a second time.
        // ------------------------------------------------------------------
        // The internal equilibration-indirect pass (in the radiation block
        // below) needs to repeat exactly this solve with an extra per-face
        // constant on the equilibration forcing. It reads the OUTER-scope
        // illumination arrays rather than the `_full` aliases, because both
        // extraction branches below leave the outer arrays populated by the
        // time the radiation block runs, while `_full` is left empty by the
        // swap-back on the non-chunked branch.
        //
        // `eqi` empty reproduces the legacy call exactly on every solver.
        auto solveWholeMeshThermal =
            [&](const std::vector<double>& eqi,
                const std::vector<double>& ds,
                const std::vector<double>& pr,
                const std::vector<double>& pi) -> bool {
            const bool use_simd_batch_l  = thermal_config.getPropertyBool("use_simd_batch", true);
            const bool use_gpu_thermal_l = thermal_config.getPropertyBool("use_gpu_thermal", false);
            ice::IceStabilityAccumulator* ice_acc_l =
                sim_config.thermal.compute_ice_stability ? &ice_stability_accum : nullptr;
            if (use_gpu_thermal_l) {
                const std::vector<double>* face_H_ptr_l =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                // A loaded thermal-state checkpoint is already equilibrated by
                // the previous chunk, so there is no equilibration phase to
                // correct (and eqi is a measured no-op: no step qualifies).
                int eq_steps_eff = equilibration_steps;
                if (!sim_config.thermal.initial_state_file.empty()) {
                    if (eq_steps_eff > 0) {
                        std::cout << "[GPU Thermal] thermal.initial_state_file is set; "
                                     "forcing equilibration_steps=0 (was "
                                  << eq_steps_eff << ")" << std::endl;
                    }
                    eq_steps_eff = 0;
                }
                const bool ok = runThermalModelsForAllFaces_GPU_Persistent(
                    mesh, ds, pr, pi, times, total_time_steps, eq_steps_eff,
                    thermal_config, default_visible_albedo, 0.02, &surface_temps,
                    face_Qb_ptr, face_H_ptr_l, ice_acc_l,
                    sim_config.thermal.initial_state_file,
                    sim_config.thermal.final_state_file,
                    enable_radiation_exchange ? &rad_post_eq_T : nullptr,
                    sim_config.thermal.gpu_devices,
                    sim_config.thermal.gpu_device_ids,
                    sim_config.thermal.gpu_batch_size,
                    sim_config.thermal.gpu_inter_batch_pause_ms,
                    eqi);
                // This solver — and only this solver — fills the post-eq T(z)
                // hand-off (argument above). Record it for the PTM-78 tripwire.
                rad_post_eq_T_expected = enable_radiation_exchange;
                return ok;
            }
            if (use_simd_batch_l) {
                const std::vector<double>* face_H_ptr_l =
                    planet.face_H_overrides.empty() ? nullptr : &planet.face_H_overrides;
                return runThermalModelsForAllFacesBatchSoA(
                    output_dir, mesh, ds, pr, pi, times, total_time_steps,
                    thermal_config, default_visible_albedo, 0.02, &surface_temps,
                    equilibration_steps, total_time_steps, thermal_threads,
                    face_Qb_ptr, face_H_ptr_l, eqi);
            }
            std::cout << "[Thermal] Using CPU parallel solver with " << thermal_threads
                      << " threads" << std::endl;
            return runThermalModelsForAllFaces(
                output_dir, mesh, ds, pr, pi, times, total_time_steps,
                thermal_config, default_visible_albedo, 0.02, &surface_temps,
                equilibration_steps, total_time_steps, thermal_threads,
                &planet, face_Qb_ptr, eqi);
        };

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
        // Max illumination footprint (GB) to EXTRACT into RAM; above this we stream
        // per-(face,timestep) (~100x slower, see the other STREAM_THRESHOLD_GB site).
        // Default 10 GB unchanged; PTM_RAM_ILLUM_MAX_GB overrides for ico6 asteroids.
        double STREAM_THRESHOLD_GB = 10.0;
        if (const char* ev = std::getenv("PTM_RAM_ILLUM_MAX_GB")) {
            double ev_gb = std::atof(ev);
            if (ev_gb > 0.0) STREAM_THRESHOLD_GB = ev_gb;
        }
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
            std::cout << "  Average non-zero illumination: " << (total_illumination / non_zero_illumination) << " W/m^2" << std::endl;
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
                face_H_ptr,
                // The angular albedo model and the supplied equilibration
                // indirect flux: the same two inputs every other CPU path
                // already receives. This path used to get neither.
                &planet,
                equilibration_steps,
                eq_indirect_vec
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

            thermal_success = solveWholeMeshThermal(
                eq_indirect_vec,
                direct_solar_values_full,
                primary_reflected_values_full,
                primary_ir_values_full);

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

            thermal_success = solveWholeMeshThermal(
                eq_indirect_vec,
                direct_solar_values_full,
                primary_reflected_values_full,
                primary_ir_values_full);

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

        // Surface-Newton globalization counters and, under PTM_SOLVER_EQFIX,
        // the T_geo-floor release census. Both are no-ops unless something
        // fired; a counter nobody prints is the same as no counter.
        thermal::reportSurfaceNewtonStats();

        // ====================================================================
        // PTM-53 rule-7 tripwire: an override that was requested but never
        // applied must FAIL, not sail through with a reassuring banner.
        // ====================================================================
        // This is what PTM-53 was: the log said "Property override: ks 0.00074
        // -> 0.00148 W/m/K" and the temperature field came out byte-identical to
        // the un-overridden run. It fires before any statistics are exported, so
        // a run that cannot honor the request produces no misleading output.
        // It will also catch the regression of a FUTURE fifth solver entry point
        // that forgets to call applyConfiguredThermo().
        {
            const auto& tp = sim_config.thermophysical;
            if (thermoOverrideRequested(tp, geothermal_heat_flow)
                && g_thermo_override_applied.load(std::memory_order_relaxed) == 0) {
                std::cerr << "\nFATAL: a thermophysical override was configured but NO thermal "
                             "solver applied it.\n"
                          << "  Requested:"
                          << (tp.ks_override              > 0.0 ? " ks_override"         : "")
                          << (tp.kd_override              > 0.0 ? " kd_override"         : "")
                          << (tp.emissivity_override      > 0.0 ? " emissivity_override" : "")
                          << (tp.albedo_override         >= 0.0 ? " albedo_override"     : "")
                          << (tp.constant_thermal_inertia > 0.0 ? " constant_thermal_inertia" : "")
                          << (geothermal_heat_flow        > 0.0 ? " geothermal_heat_flow" : "")
                          << "\n"
                          << "  The run would have reported the override and silently used the "
                             "body defaults (BUGS.md PTM-53).\n"
                          << "  Refusing to write results." << std::endl;
                return 1;
            }
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
            rad_config.rms_slope_deg = rad_rms_slope_deg;
            rad_config.beam_vf_threshold = rad_beam_vf_threshold;
            rad_config.radiance_prune_eps = rad_radiance_prune_eps;
            // PTM-98 / PTM-99: radiation-exchange face set. min_shadow_fraction
            // parsed but was never bridged here before PTM-99, so the hard-coded
            // 0.3 always won; it is now forwarded and only consulted in "shadow".
            rad_config.exchange_selection =
                (rad_exchange_selection == "shadow")
                    ? radiation::RadiationConfig::ExchangeSelection::Shadow
                    : radiation::RadiationConfig::ExchangeSelection::All;
            rad_config.min_shadow_fraction = rad_min_shadow_fraction;
            // Band-separated (visible + thermal-IR) radiosity (D1). Default ON
            // (D2, 2026-07-12); gray recovered exactly with band_separated:false.
            // When ON, the per-face Bond albedo is sourced
            // into the manager BEFORE prepare() (below), so prepare()'s initGPU can
            // engage the single-GPU two-band gate for sparse Sum_j F_ij <= 1 meshes
            // (PTM-11); H-matrix / Sum_j F_ij > 1 use the CPU two-band path.
            rad_config.band_separated = sim_config.radiation.band_separated;
            // Multi-GPU radiosity dispatch — pool is materialized inside
            // RadiositySolver::initGPU() when gpu_devices > 1 and use_gpu = true.
            rad_config.gpu_devices = rad_gpu_devices;
            rad_config.gpu_device_ids = rad_gpu_device_ids;
            rad_config.gpu_batch_size = rad_gpu_batch_size;
            rad_config.gpu_inter_batch_pause_ms = rad_gpu_inter_batch_pause_ms;
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

            // Load per-face albedos for radiation coupling BEFORE prepare() (PTM-11).
            // Must match the albedos used in the initial thermal solve. The per-face
            // Bond albedo is a pure function of the config, the on-disk albedo map, and
            // the mesh face count — it does NOT depend on prepare()'s Embree scene,
            // face classification, or view-factor build — so sourcing it here is safe.
            // Reused below by the physics-mode selector (ΔT/ΔF).
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

            // D1/PTM-11: band-separated (two-band) radiosity. Source the SAME per-face
            // Bond albedo the surface BC uses (face_albedos_rad) into the manager BEFORE
            // prepare(). prepare() sequences setSolarAlbedo -> enableTwoBand -> initGPU
            // (radiation_exchange.cpp:1471-1476 / 1613-1620), so with the albedo present
            // at prepare() time two_band_ is set BEFORE initGPU and the single-GPU
            // two-band gate (two_band_gpu_ok_) engages for sparse Sum_j F_ij <= 1 meshes.
            // H-matrix / Sum_j F_ij > 1 meshes keep the row-norm-protected CPU two-band
            // (initGPU emits the CPU-fallback notice). enableTwoBand is driven by
            // config_.band_separated inside prepare() — no explicit call needed here (it
            // would be a no-op before the solver exists). Default ON (D2, 2026-07-12);
            // gray recovered exactly with band_separated:false (this block skipped).
            if (rad_config.band_separated) {
                rad_manager.setSolarAlbedo(face_albedos_rad);
                std::cout << "  [radiation.band_separated] two-band radiosity ON: visible band "
                             "uses per-face Bond albedo (scatter A_i / absorb 1-A_i), IR band 1-eps/eps."
                          << std::endl;
                // BUGS.md PTM-106. The albedo above is the STATIC base Bond albedo A0. It
                // remains the correct reflectance for the DIFFUSE inter-face exchange
                // (the operator and the absorptivity), but NOT for the first bounce of
                // the collimated direct beam, which the surface BC reflects with the
                // incidence-dependent A(theta). The coupling loop below now installs
                // A(theta) per timestep as the visible-band SOURCE albedo. Say which
                // albedo feeds the radiosity, every run -- never leave it inferable only
                // from source.
                const bool incidence_on =
                    (planet.albedoCoef.size() >= 2 && planet.S > 0.0) || planet.use_hapke_albedo;
                if (incidence_on) {
                    if (legacyRadiosityStaticAlbedo()) {
                        std::cout << "  [radiation.band_separated] PTM-106 source albedo: LEGACY "
                                     "(static base A0) -- incidence-dependent albedo is active, so "
                                     "reflected sunlight (A(theta)-A0)*Q is LOST from the budget at "
                                     "grazing incidence. Diffuse operator/absorptivity = A0."
                                  << std::endl;
                    } else {
                        std::cout << "  [radiation.band_separated] PTM-106 source albedo: FIXED -- "
                                     "the visible-band SOURCE uses the same per-face, per-timestep "
                                     "A(theta) the surface BC absorbs with ("
                                  << (planet.use_hapke_albedo ? "Hapke" : "Hayne 2017")
                                  << "); the diffuse operator/absorptivity keep the Bond value A0."
                                  << std::endl;
                    }
                } else {
                    std::cout << "  [radiation.band_separated] PTM-106 source albedo: base A0 "
                                 "(no incidence-dependent albedo on this body -- source == "
                                 "operator == A0 by construction)." << std::endl;
                }
            } else {
                // Fermi condition C3. The PTM-106 fix lives entirely inside the
                // two-band solve; on the GRAY path the visible source is not even
                // formed from a Bond albedo. setSolarAlbedo() is called only in the
                // branch above, so solar_reflectivity_ stays EMPTY here and
                // RadiositySolver::solve() falls back to
                //   solar_source[i] = reflectivity_[i] * Q_i = (1 - eps_i) * Q_i,
                // i.e. ~0.05 on the Moon -- not A0 = 0.12, and certainly not
                // A(theta) ~ 0.6 at grazing incidence. Do not let that be inferable
                // only from the source.
                const bool incidence_on_gray =
                    (planet.albedoCoef.size() >= 2 && planet.S > 0.0) || planet.use_hapke_albedo;
                if (incidence_on_gray) {
                    std::cerr << "  [radiation.band_separated] WARNING: GRAY radiosity "
                                 "(band_separated: false) with an incidence-dependent surface BC ("
                              << (planet.use_hapke_albedo ? "Hapke" : "Hayne 2017")
                              << "). The PTM-106 visible-source consistency fix does NOT apply on "
                                 "the gray path: the gray solve has no separate visible band, and "
                                 "with no per-face Bond albedo installed its reflected-solar "
                                 "source falls back to the THERMAL reflectivity 1-eps = "
                              << std::defaultfloat << std::setprecision(6)
                              << (1.0 - planet.emissivity)
                              << " (not A0 = " << default_visible_albedo
                              << ", and not the BC's A(theta)). Reflected sunlight in the "
                                 "enclosure is therefore understated for this run. Set "
                                 "band_separated: true for the incidence-consistent source."
                              << std::endl;
                }
            }

            // Prepare (classify faces, compute view factors if needed). two_band_ is now
            // activated BEFORE initGPU (via config_.band_separated inside prepare()), so
            // initGPU can set two_band_gpu_ok_ for sparse Sum_j F_ij <= 1 meshes (PTM-11).
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

                // ---- PTM-89: decomposition (chunking) independence ----------
                // Mode was resolved and validated at config time (above).
                const bool rad_coupling_legacy_mode = (rad_coupling_mode == "legacy");
                const bool rad_coupling_fixed_iters = (rad_coupling_mode == "fixed");
                // std::cout arrives here carrying a `fixed`/`setprecision(1)`
                // left over from an earlier progress line (BUGS.md PTM-69), which
                // would print a 0.05 threshold as "0.1". A diagnostic that lies
                // about the number it is warning you about is worse than none, so
                // pin the format for this block and restore it afterwards.
                const std::ios_base::fmtflags rad_banner_flags = std::cout.flags();
                const std::streamsize rad_banner_prec = std::cout.precision();
                std::cout << std::defaultfloat << std::setprecision(6);
                std::cout << "  Coupling mode: " << rad_coupling_mode
                          << (rad_coupling_legacy_mode
                                  ? "  (PRE-PTM-89: window-dependent selection + unbounded"
                                    " face-fraction exit -- results depend on how the window"
                                    " was chunked)"
                                  : "  (PTM-89: absolute selection threshold, no face-fraction"
                                    " exit)")
                          << std::endl;
                if (!rad_coupling_legacy_mode) {
                    std::cout << "  Selection threshold: " << rad_coupling_select_dt
                              << " K (absolute predicted ΔT; 0 = every face carrying"
                                 " indirect flux)" << std::endl;
                    if (rad_coupling_select_dt > 0.0) {
                        std::cout << "  WARNING [PTM-89]: coupling_select_dt > 0 re-introduces"
                                     " chunk-decomposition dependence at up to ~"
                                  << rad_coupling_select_dt
                                  << " K per face. Set 0 for decomposition independence."
                                  << std::endl;
                    }
                    if (rad_coupling_fixed_iters) {
                        std::cout << "  Early exit on coupling_temp_threshold DISABLED: every"
                                     " window runs exactly " << rad_max_coupling_iter
                                  << " sweeps." << std::endl;
                    }
                    // These two knobs are inert here; say so rather than letting a
                    // config author believe they are doing something. Compared
                    // against the struct defaults so the test cannot drift if a
                    // default is ever retuned.
                    const auto rad_defaults = decltype(sim_config.radiation){};
                    if (std::abs(rad_coupling_face_frac
                                 - rad_defaults.coupling_face_fraction) > 1e-12) {
                        std::cout << "  WARNING [PTM-89]: radiation.coupling_face_fraction="
                                  << rad_coupling_face_frac
                                  << " is IGNORED in coupling_mode='" << rad_coupling_mode
                                  << "' (the face-fraction exit has no error bound). Use"
                                     " coupling_mode: legacy to restore it." << std::endl;
                    }
                    if (std::abs(rad_coupling_select_safety
                                 - rad_defaults.coupling_select_safety) > 1e-12) {
                        std::cout << "  WARNING [PTM-89]: radiation.coupling_select_safety="
                                  << rad_coupling_select_safety
                                  << " is IGNORED in coupling_mode='" << rad_coupling_mode
                                  << "' (the selection threshold is the absolute"
                                     " coupling_select_dt there, not a fraction of"
                                     " coupling_temp_threshold)." << std::endl;
                    }
                    if (rad_coupling_select_mode != "physics") {
                        std::cout << "  WARNING [PTM-89]: coupling_select_mode='"
                                  << rad_coupling_select_mode
                                  << "' selects on an absolute flux threshold taken over the"
                                     " window, which remains chunk-dependent. Only"
                                     " coupling_select_mode: physics is decomposition-"
                                     "independent." << std::endl;
                    }
                }
                std::cout.flags(rad_banner_flags);
                std::cout.precision(rad_banner_prec);

                // ---- Re-thermal SEED tripwire (BUGS.md PTM-78) --------------
                // Every GPU coupling iteration re-integrates the selected faces
                // from rad_post_eq_T. If that hand-off is empty or all-zero the
                // whole coupled field is garbage, and it used to happen in
                // SILENCE on every warm-start chunk. Refuse to produce results
                // rather than write a plausible-looking wrong answer.
                //
                // GATED ON THE MECHANISM, not on radiation.enabled (BUGS.md
                // PTM-94). The hand-off exists on exactly one path: it is
                // written by runThermalModelsForAllFaces_GPU_Persistent and
                // read by runThermalForSelectedFaces_GPU. The CPU whole-mesh
                // solvers (runThermalModelsForAllFacesBatchSoA,
                // runThermalModelsForAllFaces) never write it, and the CPU
                // re-thermal (runThermalModelWithMatchedTimes) takes no seed
                // argument at all — it re-equilibrates each selected face from
                // scratch. Ungated, this guard therefore aborted EVERY CPU
                // radiation-coupled run while protecting nothing on that path.
                // It keeps its full force on the GPU path, which is where
                // PTM-78 actually corrupted results.
                if (rad_post_eq_T_expected) {
                    if (rad_post_eq_T.empty()) {
                        std::cerr << "\nFATAL: radiation coupling is enabled but the thermal solver"
                                     " returned no post-equilibration T(z) hand-off.\n"
                                  << "  Every coupling iteration would re-thermal from an undefined"
                                     " state (BUGS.md PTM-78).\n"
                                  << "  Refusing to write results." << std::endl;
                        return 1;
                    }
                    const size_t nl_seed = rad_post_eq_T.size() / std::max<size_t>(1, num_faces);
                    size_t n_cold_seed = 0;
                    for (size_t f = 0; f < num_faces; ++f) {
                        if (rad_post_eq_T[f] < 1.0f) ++n_cold_seed;   // layer 0
                    }
                    if (n_cold_seed > 0) {
                        std::cerr << "\nFATAL: " << n_cold_seed << "/" << num_faces
                                  << " faces carry a sub-1 K surface in the post-equilibration"
                                     " hand-off (n_layers=" << nl_seed << ").\n"
                                  << "  The radiation re-thermal would integrate them from a dead"
                                     " column and report the ramp as physics (BUGS.md PTM-78).\n"
                                  << "  Refusing to write results." << std::endl;
                        return 1;
                    }
                    std::cout << "  Re-thermal seed: OK (" << num_faces
                              << " faces, no dead columns)" << std::endl;
                } else {
                    // Not an error — a different, self-contained integration.
                    // Say it out loud so nobody compares a CPU coupled field to
                    // a GPU one and calls the difference physics (BUGS.md PTM-94).
                    std::cerr << "  [Re-thermal] WARNING: CPU thermal solver — each coupling"
                                 " iteration re-equilibrates the selected faces from scratch"
                                 " instead of resuming the GPU path's saved post-equilibration"
                                 " temperatures, so CPU and GPU coupled results can differ.\n";
                }

                // Post-coupling end-of-window T(z), assembled across coupling
                // iterations from whichever faces each iteration re-thermal'd.
                // Only allocated when a checkpoint was actually requested.
                const bool want_post_coupling_state =
                    !sim_config.thermal.final_state_file.empty();
                std::vector<float>  rad_final_T;
                std::vector<uint8_t> rad_final_touched;
                if (want_post_coupling_state) {
                    const size_t nl_state = rad_post_eq_T.size() / std::max<size_t>(1, num_faces);
                    rad_final_T.assign(nl_state * num_faces, 0.0f);
                    rad_final_touched.assign(num_faces, 0);
                }

                // Store previous iteration temperatures for convergence check
                std::vector<std::vector<double>> prev_temps = surface_temps;

                // Per-face max indirect flux from the previous iter — used by
                // the physics-mode selector to compute ΔF (the change in
                // indirect flux) rather than absolute flux. On iter 0 there
                // is no prior, so the selector falls back to absolute flux.
                std::vector<double> prev_max_indirect(num_faces, 0.0);

                // ---- Internal equilibration-indirect pass: arming -----------
                // The predicate is deliberately NOT keyed on the coupling mode,
                // the band count, or how many non-zeros the view-factor operator
                // holds. Reaching this point already means radiation.enabled is
                // true, an operator loaded, and rad_manager.prepare() found faces
                // that exchange. What remains is only: is there an equilibration
                // phase whose forcing omits a term the output window will carry?
                //
                // A mesh with no meaningful inter-face view therefore needs no
                // special case: the pass runs, measures <q_ind> = 0, says so, and
                // skips the re-solve. That is a MEASURED no-op, which is worth
                // more than a coded one.
                const bool eq_indirect_warm_start =
                    !sim_config.thermal.initial_state_file.empty();
                const bool eq_indirect_two_phase_window =
                    equilibration_steps > 0 && equilibration_steps < total_time_steps;
                const bool eq_indirect_internal_armed =
                    (eq_indirect_source == "auto")
                    && sim_config.thermal.eq_indirect_flux_file.empty()
                    && eq_indirect_two_phase_window
                    && !eq_indirect_warm_start;
                bool eq_indirect_internal_done = false;

                if (eq_indirect_internal_armed) {
                    std::cout << "  Equilibration indirect flux: INTERNAL PASS (thermal."
                                 "eq_indirect_source: auto). One extra radiosity sweep"
                                 " and one extra whole-mesh thermal solve." << std::endl;
                } else if (!sim_config.thermal.eq_indirect_flux_file.empty()) {
                    std::cout << "  Equilibration indirect flux: FILE (thermal."
                                 "eq_indirect_flux_file overrides the internal pass)."
                              << std::endl;
                } else if (eq_indirect_source != "auto") {
                    std::cout << "  Equilibration indirect flux: OFF (thermal."
                                 "eq_indirect_source: " << eq_indirect_source
                              << "). The deep column equilibrates without the"
                                 " inter-face infrared it receives in the output"
                                 " window." << std::endl;
                    eq_indirect_record.reason = "eq_indirect_source: " + eq_indirect_source;
                } else if (eq_indirect_warm_start) {
                    std::cout << "  Equilibration indirect flux: not applicable —"
                                 " thermal.initial_state_file is set, so this window has"
                                 " no equilibration phase. The column it resumes carries"
                                 " whatever the first chunk equilibrated." << std::endl;
                    eq_indirect_record.reason = "warm start (no equilibration phase)";
                } else {
                    std::cout << "  Equilibration indirect flux: not applicable —"
                                 " equilibration_steps=" << equilibration_steps
                              << " of " << total_time_steps
                              << " leaves no separate equilibration phase to correct."
                              << std::endl;
                    eq_indirect_record.reason = "no equilibration phase";
                }

                // Whether the physics-mode selector is active for this run.
                const bool rad_coupling_physics_mode =
                    (rad_coupling_select_mode == "physics");
                std::cout << "  Face-selection mode: "
                          << rad_coupling_select_mode << std::endl;

                // Time-varying indirect flux per face [face][timestep]
                // This is the key fix: indirect flux varies with time as wall temperatures change
                // (indirect_flux_series declared at outer scope for location export)
                indirect_flux_series.resize(num_faces);
                for (size_t f = 0; f < num_faces; f++) {
                    indirect_flux_series[f].resize(total_time_steps, 0.0);
                }

                // Parallel band-separated series, allocated only when the solver
                // runs two-band (avoids the ~2x indirect RAM otherwise). Same
                // shape as indirect_flux_series; captured per timestep in Step 1.
                if (rad_manager.isTwoBand()) {
                    indirect_solar_series.resize(num_faces);
                    indirect_ir_series.resize(num_faces);
                    for (size_t f = 0; f < num_faces; f++) {
                        indirect_solar_series[f].resize(total_time_steps, 0.0);
                        indirect_ir_series[f].resize(total_time_steps, 0.0);
                    }
                    std::cout << "  [indirect-bands] Two-band radiosity active: "
                                 "capturing band-separated absorbed indirect flux "
                                 "(indirect_solar + indirect_ir).\n";
                } else {
                    std::cout << "  [indirect-bands] Gray radiosity (not two-band): "
                                 "band-separated indirect outputs OMITTED.\n";
                }

                // (Per-face albedos + two-band activation were sourced BEFORE prepare()
                // above — PTM-11 — so the single-GPU two-band gate can engage for sparse
                // Sum_j F_ij <= 1 meshes. The face_albedos_rad declared there is reused
                // below by the physics-mode DeltaT/DeltaF selector.)

                // Hoisted thermal GPU pool for radiation-coupling re-thermal.
                // Created once before the iter loop and reused across every
                // selected-faces re-thermal call. Eliminates per-iter OpenCL
                // context create/destroy churn that wedged the macOS+AMD
                // driver during multi-hour rad-on runs (SMC -108 forced
                // reset; see memory/multi_gpu_thermal.md). Sized to the
                // re-thermal hard cap (50000 faces) — actual batches narrow
                // via setActiveFaceCount.
                void* shared_rerun_pool = nullptr;
#ifdef PTM_GPU_OPENCL
                std::unique_ptr<gpu::opencl::ThermalGPUDevicePool> rerun_pool;
                if (thermal_config.getPropertyBool("use_gpu_thermal", false)
                    && (sim_config.thermal.gpu_devices != 1
                        || !sim_config.thermal.gpu_device_ids.empty())) {
                    auto all_gpus = gpu::opencl::OpenCLContext::enumerateGPUs();
                    std::vector<gpu::opencl::OpenCLDeviceInfo> rerun_devs;
                    if (!sim_config.thermal.gpu_device_ids.empty()) {
                        for (int id : sim_config.thermal.gpu_device_ids) {
                            if (id >= 0
                                && static_cast<size_t>(id) < all_gpus.size()) {
                                rerun_devs.push_back(all_gpus[id]);
                            }
                        }
                    } else if (sim_config.thermal.gpu_devices == 0) {
                        rerun_devs = all_gpus;
                    } else {
                        const int n_req =
                            std::min(sim_config.thermal.gpu_devices,
                                     static_cast<int>(all_gpus.size()));
                        for (int i = 0; i < n_req; ++i) {
                            rerun_devs.push_back(all_gpus[i]);
                        }
                    }

                    // Cap re-thermal concurrency to 1 GPU by default.
                    // Rationale (May 12, 2026 production campaign):
                    //  * At N=2 (the previous default), ~25% of high-load
                    //    winter windows (~640 K selected faces) hit the
                    //    `gpusKillClientExt` hang and required the chunked
                    //    runner's 3-attempt retry budget; 3 of 88 windows
                    //    failed outright and needed a follow-up --resume.
                    //  * At N=1, the May 12 recovery cleared all 3 failed
                    //    windows in 81 minutes with zero retries.
                    //  * Per-window wall at N=1 (~23 min) measured within
                    //    noise of clean N=2 (~25 min) — the re-thermal
                    //    phase isn't 2× slower because the inter-batch
                    //    pause overhead is preserved across N and the
                    //    kernel time is not the dominant cost. Net: N=1
                    //    is the same speed and removes the failure mode.
                    // Skip the cap when the user passed explicit
                    // gpu_device_ids — that's an opt-in to expert config.
                    // Override the default with PTM_RAD_RERUN_MAX_GPUS=N.
                    {
                        int rerun_max = 1;
                        if (const char* env =
                                std::getenv("PTM_RAD_RERUN_MAX_GPUS")) {
                            int v = std::atoi(env);
                            if (v > 0) rerun_max = v;
                        }
                        if (sim_config.thermal.gpu_device_ids.empty()
                            && static_cast<int>(rerun_devs.size()) > rerun_max) {
                            std::cout << "  [GPU Thermal Pool] Capping "
                                         "re-thermal concurrency at "
                                      << rerun_max << " device(s) (was "
                                      << rerun_devs.size()
                                      << ") to reduce macOS+AMD driver "
                                         "pressure during rad-coupling. "
                                         "Override: PTM_RAD_RERUN_MAX_GPUS env."
                                      << std::endl;
                            rerun_devs.resize(rerun_max);
                        }
                    }

                    if (rerun_devs.size() >= 2) {
                        // Match the n_layers and BATCH_SIZE that
                        // runThermalForSelectedFaces_GPU computes internally.
                        // BATCH_SIZE is hardcoded to min(memory-limit, 50000)
                        // there; we size for the 50000 cap so the same pool
                        // covers every iter's re-thermal regardless of the
                        // selected-face count.
                        thermal::Profile rerun_profile(planet, 0.0,
                                                       thermal_config, false);
                        const size_t rerun_n_layers = rerun_profile.z().size();
                        constexpr size_t kRerunBatchCap = 50000;

                        gpu::opencl::ThermalOpenCLConfig solver_cfg;
                        solver_cfg.verbose = false;
                        solver_cfg.work_group_size = 64;

                        gpu::opencl::ThermalGPUPoolConfig pool_cfg;
                        pool_cfg.devices = rerun_devs;
                        pool_cfg.max_n_faces = kRerunBatchCap;
                        pool_cfg.n_layers = rerun_n_layers;
                        pool_cfg.solver_config = solver_cfg;
                        pool_cfg.inter_batch_pause_ms =
                            sim_config.thermal.gpu_inter_batch_pause_ms;
                        pool_cfg.verbose = false;
                        try {
                            rerun_pool = std::make_unique<
                                gpu::opencl::ThermalGPUDevicePool>(
                                std::move(pool_cfg));
                            shared_rerun_pool = rerun_pool.get();
                            std::cout << "  [GPU Thermal Pool] Hoisted "
                                         "re-thermal pool: "
                                      << rerun_devs.size() << " device(s), "
                                      << "capacity=" << kRerunBatchCap
                                      << " faces, n_layers="
                                      << rerun_n_layers << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "  [GPU Thermal Pool] Hoisted pool "
                                         "construction failed: " << e.what()
                                      << " — re-thermal will build per-iter "
                                         "pools instead" << std::endl;
                            rerun_pool.reset();
                            shared_rerun_pool = nullptr;
                        }
                    }
                }
#endif  // PTM_GPU_OPENCL

                // Picard under-relaxation for the thermal–radiation coupling.
                // Each iteration blends the freshly-computed indirect flux with
                // the previous iterate: indirect = (1-ω)·prev + ω·new. Without
                // it, clusters of mutually-facing faces ratchet their indirect
                // flux up every iteration (1105→3046 W/m² over 5 iters, never
                // converging) and a few peg the 500 K MAX_TEMP clamp. The fixed
                // point is unchanged — ω only damps the path to it. With the
                // near-field-accurate view factors (computeViewFactorRefined),
                // Σ_j F_ij ≤ ~1 holds and the coupling already converges in one
                // iteration, so under-relaxation is OFF by default (ω=1, the
                // full Picard step) and retained only as an env-gated safety
                // knob for pathological meshes. Tunable via PTM_RAD_COUPLING_RELAX.
                double rad_relax = 1.0;
                if (const char* e = std::getenv("PTM_RAD_COUPLING_RELAX")) {
                    const double v = std::atof(e);
                    if (v > 0.0 && v <= 1.0) rad_relax = v;
                }
                std::cout << "  Coupling under-relaxation ω = " << rad_relax << std::endl;

                for (int iter = 0; iter < rad_max_coupling_iter; iter++) {
                    std::cout << "\n--- Radiation Iteration " << (iter + 1) << " ---" << std::endl;
                    writeHeartbeat("rad_iter " + std::to_string(iter + 1)
                                   + " start (max=" + std::to_string(rad_max_coupling_iter) + ")");

                    // Step 1: Compute indirect flux at EVERY timestep
                    // This captures time-varying thermal emission from illuminated walls
                    double max_indirect = 0.0;
                    double total_indirect_sum = 0.0;

                    thermal::ProgressBar flux_progress(total_time_steps, "Computing indirect flux", 50, true);

                    // Multi-GPU radiosity dispatch: when the pool is active,
                    // process timesteps in batches sized to amortize per-batch
                    // overhead across N GPUs concurrently. Single-GPU and CPU
                    // paths use batch_size = 1 for byte-identical legacy behavior.
                    const bool use_pool_dispatch = rad_manager.hasGPUPool();
                    int batch_size = 1;
                    if (use_pool_dispatch) {
                        batch_size = (rad_gpu_batch_size > 0)
                            ? rad_gpu_batch_size
                            : 4 * std::max(1, rad_gpu_devices);
                        std::cout << "  [Multi-GPU radiosity] dispatching "
                                  << batch_size << " timesteps/batch across "
                                  << rad_gpu_devices << " GPUs" << std::endl;
                    }

                    // Hoisted out of the per-timestep loop: these two num_faces-sized
                    // buffers (~38 MB at 2.4M faces) were re-allocated + freed every
                    // timestep on the single-timestep fast path. They are fully
                    // overwritten each step below, so reuse is behavior-identical.
                    std::vector<double> temps_t(num_faces);
                    std::vector<double> solar_t(num_faces);

                    // --- PTM-106: per-timestep visible-band SOURCE albedo ------------
                    // A(theta) is the albedo the surface BC reflects the DIRECT beam
                    // with, so it is exactly what the enclosure must receive. Recipe is
                    // character-for-character the one the coupled CPU re-thermal uses to
                    // rebuild combined_flux below, so source == 1 - BC absorptivity by
                    // construction on every face and every timestep.
                    const double ptm106_a_coef = (planet.albedoCoef.size() >= 2)
                                               ? planet.albedoCoef[0] : 0.0;
                    const double ptm106_b_coef = (planet.albedoCoef.size() >= 2)
                                               ? planet.albedoCoef[1] : 0.0;
                    const double ptm106_S      = planet.S;
                    const bool   ptm106_hapke  = planet.use_hapke_albedo
                                               && planet.hapke_albedo_w >= 0.0;
                    const bool   ptm106_incidence =
                        (planet.albedoCoef.size() >= 2 && ptm106_S > 0.0)
                        || (ptm106_hapke && ptm106_S > 0.0);
                    // Active only where it can mean anything: two-band radiosity ON and
                    // an incidence-dependent BC. Otherwise source == operator == A0 and
                    // installing it would be a no-op that only risks drift.
                    const bool ptm106_active = rad_config.band_separated
                                             && ptm106_incidence
                                             && !legacyRadiosityStaticAlbedo();
                    std::vector<double> srcalb_t;
                    if (ptm106_active) srcalb_t.resize(num_faces);
                    // Self-check accumulators (see reportSourceAlbedoConsistency).
                    // Filled on coupling iteration 0 only -- every iteration sweeps the
                    // same timesteps, so accumulating them all would just re-weight the
                    // identical sum and hide nothing.
                    // NOTE: this block runs once per COUPLING ITERATION, not once per
                    // run -- clearing unconditionally here would wipe iteration 0's
                    // accumulation on iteration 1 and leave the self-check with nothing
                    // to compare (it silently reported nothing until this was caught).
                    if (ptm106_active && iter == 0) {
                        g_src_albedo_num.assign(num_faces, 0.0);
                        g_src_albedo_den.assign(num_faces, 0.0);
                        // Record that the identity EXISTS for this run, so a
                        // self-check that fails to run is reported as a defect
                        // rather than as silence (see reportSourceAlbedoConsistency).
                        g_ptm106_expected = true;
                    }
                    auto ptm106_fill = [&](std::vector<double>& out, int t,
                                           const std::vector<double>& q,
                                           bool accumulate) {
                        const double S_t = solarConstForStep(t, ptm106_S);
                        #pragma omp parallel for schedule(static)
                        for (size_t f = 0; f < num_faces; f++) {
                            const double base_A = (f < face_albedos_rad.size())
                                                ? face_albedos_rad[f] : planet.albedo;
                            out[f] = applyIncidenceAlbedo(base_A, q[f], S_t,
                                                          ptm106_a_coef, ptm106_b_coef,
                                                          ptm106_hapke,
                                                          hapkeWFromA0(base_A));
                            // C2: same INTEGRATION window as the BC-side accumulator.
                            // The coupling loop sweeps every absolute timestep,
                            // equilibration included; restricting one side and not
                            // the other would turn a 1-ulp identity into a spurious
                            // several-percent "inconsistency".
                            if (accumulate && q[f] > 0.0 && aeffStepInWindow(t)) {
                                g_src_albedo_num[f] += out[f] * q[f];
                                g_src_albedo_den[f] += q[f];
                            }
                        }
                        if (accumulate) g_src_albedo_valid = true;
                    };
                    for (int t_start = 0; t_start < total_time_steps;
                         t_start += batch_size) {
                        const int t_end =
                            std::min(t_start + batch_size, total_time_steps);
                        const int n_batch = t_end - t_start;

                        if (n_batch == 1 && !use_pool_dispatch) {
                            // Legacy fast path: single timestep, no batch
                            // wrapping. Preserves byte-identical behavior
                            // for CPU and single-GPU radiation paths.
                            const int t = t_start;
                            #pragma omp parallel for schedule(static)
                            for (size_t f = 0; f < num_faces; f++) {
                                temps_t[f] = surface_temps[f][t];
                                solar_t[f] = direct_solar_values[f * total_time_steps + t];
                            }
                            // Smith beamed emission needs the Sun direction.
                            std::array<double, 3> s_hat = {0.0, 0.0, 0.0};
                            if (rad_rms_slope_deg > 0.0 &&
                                t < static_cast<int>(sun_positions.size())) {
                                const auto& sp = sun_positions[t];
                                double sl = std::sqrt(sp[0]*sp[0] + sp[1]*sp[1] + sp[2]*sp[2]);
                                if (sl > 0.0) s_hat = {sp[0]/sl, sp[1]/sl, sp[2]/sl};
                            }
                            // PTM-106: install A(theta) for THIS timestep before solving.
                            if (ptm106_active) {
                                ptm106_fill(srcalb_t, t, solar_t, iter == 0);
                                rad_manager.setSolarSourceAlbedo(srcalb_t);
                            }
                            std::vector<double> indirect_t =
                                rad_manager.computeIndirectFlux(temps_t, solar_t, s_hat);
                            const double w = (iter == 0) ? 1.0 : rad_relax;
                            // Band-separated absorbed indirect (two-band only). The
                            // manager refreshed these on the computeIndirectFlux call
                            // above; capture the pointers before the parallel region.
                            const bool have_bands =
                                !indirect_solar_series.empty()
                                && rad_manager.isTwoBand()
                                && rad_manager.lastIndirectVisible().size() == num_faces
                                && rad_manager.lastIndirectIR().size() == num_faces;
                            const std::vector<double>* qvis =
                                have_bands ? &rad_manager.lastIndirectVisible() : nullptr;
                            const std::vector<double>* qir =
                                have_bands ? &rad_manager.lastIndirectIR() : nullptr;
                            if (have_bands) indirect_bands_available = true;
                            #pragma omp parallel for schedule(static) reduction(max:max_indirect) reduction(+:total_indirect_sum)
                            for (size_t f = 0; f < num_faces; f++) {
                                indirect_flux_series[f][t] =
                                    (1.0 - w) * indirect_flux_series[f][t] + w * indirect_t[f];
                                if (have_bands) {
                                    indirect_solar_series[f][t] =
                                        (1.0 - w) * indirect_solar_series[f][t] + w * (*qvis)[f];
                                    indirect_ir_series[f][t] =
                                        (1.0 - w) * indirect_ir_series[f][t] + w * (*qir)[f];
                                }
                                if (indirect_t[f] > max_indirect) {
                                    max_indirect = indirect_t[f];
                                }
                                total_indirect_sum += indirect_t[f];
                            }
                        } else {
                            // Batched path: build [n_batch][num_faces] of
                            // (T, Q_solar), dispatch to multi-GPU pool,
                            // write back to indirect_flux_series.
                            std::vector<std::vector<double>> temps_batch(n_batch);
                            std::vector<std::vector<double>> solar_batch(n_batch);
                            #pragma omp parallel for schedule(static)
                            for (int j = 0; j < n_batch; j++) {
                                const int t = t_start + j;
                                temps_batch[j].resize(num_faces);
                                solar_batch[j].resize(num_faces);
                                for (size_t f = 0; f < num_faces; f++) {
                                    temps_batch[j][f] = surface_temps[f][t];
                                    solar_batch[j][f] = direct_solar_values[f * total_time_steps + t];
                                }
                            }

                            // Per-timestep Sun directions for Smith beamed emission.
                            std::vector<std::array<double, 3>> sun_dirs_batch;
                            if (rad_rms_slope_deg > 0.0) {
                                sun_dirs_batch.assign(n_batch, {0.0, 0.0, 0.0});
                                for (int j = 0; j < n_batch; j++) {
                                    const int t = t_start + j;
                                    if (t >= static_cast<int>(sun_positions.size())) continue;
                                    const auto& sp = sun_positions[t];
                                    double sl = std::sqrt(sp[0]*sp[0] + sp[1]*sp[1] + sp[2]*sp[2]);
                                    if (sl > 0.0) sun_dirs_batch[j] = {sp[0]/sl, sp[1]/sl, sp[2]/sl};
                                }
                            }
                            // PTM-106: per-timestep A(theta) for the batched (pool) path.
                            // Built here so timestep j pairs with solar_batch[j]; the
                            // solver refuses the pool rather than mis-pair on a mismatch.
                            if (ptm106_active) {
                                std::vector<std::vector<double>> srcalb_batch(n_batch);
                                for (int j = 0; j < n_batch; j++) {
                                    srcalb_batch[j].resize(num_faces);
                                    ptm106_fill(srcalb_batch[j], t_start + j,
                                                solar_batch[j], iter == 0);
                                }
                                rad_manager.setSolarSourceAlbedoBatch(srcalb_batch);
                            }
                            std::vector<std::vector<double>> indirect_batch =
                                rad_manager.computeIndirectFluxBatch(temps_batch, solar_batch,
                                                                     sun_dirs_batch);

                            // Two-band pool (2d): per-band batch results, mapped to
                            // full faces. indirect_solar = visible band, indirect_ir
                            // = IR band (same convention as the single-timestep path).
                            const auto& irb  = rad_manager.lastIndirectIRBatch();
                            const auto& visb = rad_manager.lastIndirectVisibleBatch();
                            const bool have_bands_batch =
                                !indirect_solar_series.empty()
                                && rad_manager.isTwoBand()
                                && irb.size() == static_cast<size_t>(n_batch)
                                && visb.size() == static_cast<size_t>(n_batch);
                            if (have_bands_batch) {
                                indirect_bands_available = true;
                            } else if (!indirect_solar_series.empty()
                                       && rad_manager.isTwoBand()
                                       && !indirect_bands_batch_gap) {
                                // Two-band requested but the pool returned no bands:
                                // OMIT rather than write misleading zeros (fail-loud).
                                indirect_bands_batch_gap = true;
                                std::cerr << "  [indirect-bands] WARNING: multi-GPU pool "
                                             "radiosity produced no band separation; "
                                             "indirect_solar/indirect_ir will be OMITTED.\n";
                            }

                            for (int j = 0; j < n_batch; j++) {
                                const int t = t_start + j;
                                const auto& indirect_t = indirect_batch[j];
                                const double w = (iter == 0) ? 1.0 : rad_relax;
                                const std::vector<double>* qir =
                                    have_bands_batch ? &irb[j] : nullptr;
                                const std::vector<double>* qvis =
                                    have_bands_batch ? &visb[j] : nullptr;
                                #pragma omp parallel for schedule(static) reduction(max:max_indirect) reduction(+:total_indirect_sum)
                                for (size_t f = 0; f < num_faces; f++) {
                                    indirect_flux_series[f][t] =
                                        (1.0 - w) * indirect_flux_series[f][t] + w * indirect_t[f];
                                    if (have_bands_batch) {
                                        indirect_solar_series[f][t] =
                                            (1.0 - w) * indirect_solar_series[f][t] + w * (*qvis)[f];
                                        indirect_ir_series[f][t] =
                                            (1.0 - w) * indirect_ir_series[f][t] + w * (*qir)[f];
                                    }
                                    if (indirect_t[f] > max_indirect) {
                                        max_indirect = indirect_t[f];
                                    }
                                    total_indirect_sum += indirect_t[f];
                                }
                            }
                        }

                        if (t_end % 10 == 0 || t_end == total_time_steps) {
                            flux_progress.update(t_end);
                        }
                    }
                    flux_progress.finish();

                    double mean_indirect = total_indirect_sum / (num_faces * total_time_steps);
                    std::cout << "  Max indirect flux: " << max_indirect << " W/m2" << std::endl;
                    std::cout << "  Mean indirect flux: " << mean_indirect << " W/m2" << std::endl;

                    // ==========================================================
                    // Internal equilibration-indirect pass (thermal.eq_indirect_source: auto)
                    // ==========================================================
                    // The sweep just above is pass 1: it produced Q_indirect(i,t)
                    // from temperatures whose deep column was equilibrated with
                    // NO indirect term at all. Reduce it to the output-window
                    // time mean <q_ind>[i] -- the same estimator that
                    // thermal.indirect_mean_out_file writes -- and repeat the
                    // whole-mesh thermal solve with <q_ind> on the equilibration
                    // forcing. Then restart the coupling loop, which now runs
                    // against a column that was equilibrated against the heat it
                    // actually receives.
                    //
                    // Cost: exactly one extra radiosity sweep (this one) plus one
                    // extra whole-mesh thermal solve. It fires ONCE per run.
                    //
                    // Why the time mean and not the time series: equilibration is
                    // a DC problem -- it sets the deep column, whose response time
                    // is many diurnal cycles -- so the cycle-mean forcing is the
                    // quantity that determines it. Dropping the diurnal modulation
                    // of the indirect term is correct at leading order and is what
                    // the manual two-pass file workflow has always done.
                    //
                    // One pass UNDER-corrects: pass 1's own emitters are colder
                    // than the converged state, so <q_ind> is a lower bound. The
                    // natural place to iterate is right here -- loop this block
                    // until <q_ind> stops moving -- but no iteration is built
                    // until the single-pass recovery has been measured.
                    if (eq_indirect_internal_armed && !eq_indirect_internal_done) {
                        eq_indirect_internal_done = true;

                        const int out_lo = equilibration_steps;
                        const int out_hi = total_time_steps;
                        const int out_n  = std::max(1, out_hi - out_lo);
                        std::vector<double> q_ind_mean(num_faces, 0.0);
                        #pragma omp parallel for schedule(static)
                        for (size_t f = 0; f < num_faces; ++f) {
                            const auto& row = indirect_flux_series[f];
                            double acc = 0.0;
                            for (int t = out_lo; t < out_hi
                                     && t < static_cast<int>(row.size()); ++t)
                                acc += row[t];
                            q_ind_mean[f] = acc / out_n;
                        }
                        eq_indirect_record.census(q_ind_mean);

                        const std::ios_base::fmtflags eqi_flags = std::cout.flags();
                        const std::streamsize eqi_prec = std::cout.precision();
                        std::cout << std::defaultfloat << std::setprecision(6);
                        std::cout << "\n  [Eq indirect] Internal pass 1 complete: per-face"
                                     " output-window mean indirect flux over "
                                  << out_n << " steps\n"
                                  << "    mean " << eq_indirect_record.mean_Wm2
                                  << ", median " << eq_indirect_record.median_Wm2
                                  << ", max " << eq_indirect_record.max_Wm2 << " W/m^2; "
                                  << eq_indirect_record.n_nonzero << "/" << num_faces
                                  << " faces receive any" << std::endl;

                        if (eq_indirect_record.max_Wm2 <= 0.0) {
                            // A mesh with no inter-face view lands here. Skipping
                            // the re-solve keeps it bit-identical to legacy AND
                            // costs nothing, but it must be said out loud.
                            std::cout << "    No face receives indirect flux — the"
                                         " equilibration forcing is unchanged and this"
                                         " run is identical to eq_indirect_source: off."
                                      << std::endl;
                            eq_indirect_record.provenance = "internal pass (measured zero)";
                            std::cout.flags(eqi_flags);
                            std::cout.precision(eqi_prec);
                        } else {
                            std::cout << "    Re-solving the whole-mesh thermal model with"
                                         " this on the equilibration forcing, then"
                                         " restarting the coupling loop." << std::endl;
                            std::cout.flags(eqi_flags);
                            std::cout.precision(eqi_prec);

                            eq_indirect_vec = q_ind_mean;
                            const bool ok = solveWholeMeshThermal(
                                eq_indirect_vec, direct_solar_values,
                                primary_reflected_values, primary_ir_values);
                            if (!ok || !validateTemperatureMatrix(
                                    surface_temps, num_faces, total_time_steps)) {
                                std::cerr << "[FATAL] Internal equilibration-indirect"
                                             " pass 2 thermal solve failed; refusing to"
                                             " write results.\n";
                                return 1;
                            }
                            eq_indirect_record.provenance = "internal pass";

                            // The coupling loop's own state refers to the pass-1
                            // temperature field, which no longer exists. Reset it
                            // so iteration 1 below is a genuine first iteration.
                            prev_temps = surface_temps;
                            std::fill(prev_max_indirect.begin(),
                                      prev_max_indirect.end(), 0.0);
                            for (auto& row : indirect_flux_series)
                                std::fill(row.begin(), row.end(), 0.0);
                            for (auto& row : indirect_solar_series)
                                std::fill(row.begin(), row.end(), 0.0);
                            for (auto& row : indirect_ir_series)
                                std::fill(row.begin(), row.end(), 0.0);

                            iter = -1;   // the ++ below makes the next pass iter 0
                            continue;
                        }
                    }

                    // Diagnostic: compare indirect flux for high vs low topographic elevation
                    // Uses radial distance from body center as elevation proxy (works on spherical bodies)
                    // Low-elevation faces (valleys/craters) should receive MORE indirect flux.
                    // Gated on verbose: it std::sorts all num_faces radii every coupling
                    // iteration purely for the two printed averages -- skip when quiet.
                    if (rad_verbose) {
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

                    // Step 2: Per-face statistics + selection.
                    //
                    // Single pass over (face, timestep) computes everything
                    // both the selector and the convergence check need:
                    //   - max_indirect[f]: max over t of indirect_flux_series
                    //   - max_T[f]:        max over t of surface_temps
                    //   - observed_dT[f]:  max over t of |T_iter − T_iter-1|
                    //                      (zero on iter 0 since prev_temps == surface_temps)
                    //   - global max_temp_change (= max over f of observed_dT)
                    //
                    // Physics mode (default):
                    //   ΔF[f]    = max(0, max_indirect[f] − prev_max_indirect[f])  (iter > 0)
                    //              max_indirect[f]                                  (iter == 0)
                    //   T_eff[f] = max(max_T[f], T_floor=40 K)
                    //   pred_dT  = (1−α)·ΔF / (4εσ·T_eff³)   ← linearized radiative response
                    //   select if max(pred_dT, observed_dT) > temp_tol·safety
                    //
                    //   This combines Phase A (predict future change from new
                    //   indirect flux) and Phase B (don't drop a face that's
                    //   still moving even if its flux didn't change). A face
                    //   with stable flux AND stable past T is naturally frozen.
                    //
                    // Legacy mode:
                    //   select if max_indirect[f] > rad_coupling_flux_tol
                    //   (reproduces pre-2026-05-09 behavior byte-for-byte).
                    std::vector<double> face_max_indirect(num_faces, 0.0);
                    std::vector<double> observed_dT(num_faces, 0.0);
                    std::vector<double> face_max_T(num_faces, 0.0);
                    double max_temp_change = 0.0;
                    double total_indirect_per_face_sum = 0.0;

                    #pragma omp parallel for schedule(static) \
                        reduction(max:max_temp_change) \
                        reduction(+:total_indirect_per_face_sum)
                    for (size_t f = 0; f < num_faces; f++) {
                        double mi = 0.0, mT = 0.0, odT = 0.0;
                        for (int t = 0; t < total_time_steps; t++) {
                            const double v = indirect_flux_series[f][t];
                            if (v > mi) mi = v;
                            const double T = surface_temps[f][t];
                            if (T > mT) mT = T;
                            const double dT = std::abs(T - prev_temps[f][t]);
                            if (dT > odT) odT = dT;
                        }
                        face_max_indirect[f] = mi;
                        face_max_T[f]        = mT;
                        observed_dT[f]       = odT;
                        total_indirect_per_face_sum += mi;
                        if (odT > max_temp_change) max_temp_change = odT;
                    }

                    // Pick faces to re-run.
                    std::vector<int> faces_to_rerun;
                    faces_to_rerun.reserve(num_faces / 4);
                    int n_frozen_by_dT = 0;
                    int n_frozen_by_freeze = 0;

                    if (rad_coupling_physics_mode) {
                        const double epsilon = planet.emissivity;
                        const double sigma = ptm::kStefanBoltzmann;
                        const double T_floor = 40.0;
                        // PTM-89: in legacy mode the threshold is a fraction of
                        // the convergence tolerance -- ~1 K -- and both of the
                        // quantities compared against it are maxima OVER THE
                        // WINDOW, so a half-length window scores every face
                        // lower and freezes faces the full window re-thermals.
                        // In bounded/fixed mode the threshold is absolute
                        // (default 0), so the decision is "does this face carry
                        // any indirect flux", which no longer depends on where
                        // the window boundaries fell.
                        const double pred_dT_threshold = rad_coupling_legacy_mode
                            ? rad_coupling_temp_tol * rad_coupling_select_safety
                            : rad_coupling_select_dt;

                        for (size_t f = 0; f < num_faces; f++) {
                            const double dF = (iter == 0)
                                ? face_max_indirect[f]
                                : std::max(0.0, face_max_indirect[f] - prev_max_indirect[f]);
                            const double T_eff = std::max(face_max_T[f], T_floor);
                            const double dTdF = (1.0 - face_albedos_rad[f])
                                              / (4.0 * epsilon * sigma * T_eff * T_eff * T_eff);
                            const double pred_dT = dTdF * dF;

                            // Phase A criterion: does the new indirect flux
                            // imply a future T change above the threshold?
                            // Phase B criterion: did the face's T already
                            // change meaningfully in the previous re-thermal?
                            // Comparison is strict in bounded/fixed mode so that
                            // a zero threshold still freezes the faces that carry
                            // no flux at all (>= 0 would select all of them).
                            const bool select = rad_coupling_legacy_mode
                                ? ((pred_dT >= pred_dT_threshold)
                                   || (iter > 0 && observed_dT[f] >= pred_dT_threshold))
                                : ((pred_dT > pred_dT_threshold)
                                   || (iter > 0 && observed_dT[f] > pred_dT_threshold));
                            if (select) {
                                faces_to_rerun.push_back(static_cast<int>(f));
                            } else {
                                if (iter > 0 && observed_dT[f] < pred_dT_threshold) {
                                    ++n_frozen_by_freeze;
                                }
                                if (pred_dT < pred_dT_threshold) {
                                    ++n_frozen_by_dT;
                                }
                            }
                        }
                        std::cout << "  Faces above predicted ΔT threshold ("
                                  << pred_dT_threshold << " K): "
                                  << faces_to_rerun.size() << " / " << num_faces
                                  << "  (frozen: " << (num_faces - faces_to_rerun.size())
                                  << ")" << std::endl;
                    } else {
                        for (size_t f = 0; f < num_faces; f++) {
                            if (face_max_indirect[f] > rad_coupling_flux_tol) {
                                faces_to_rerun.push_back(static_cast<int>(f));
                            }
                        }
                        std::cout << "  Faces with significant indirect flux: "
                                  << faces_to_rerun.size() << " / " << num_faces
                                  << std::endl;
                    }

                    if (faces_to_rerun.empty()) {
                        std::cout << "  No faces need re-run. Converged." << std::endl;
                        break;
                    }

                    // NOTE: the convergence check MOVED to AFTER the re-thermal
                    // (end of this loop body) so a converged window stops BEFORE
                    // running one more redundant indirect (radiosity) pass whose
                    // result the skipped re-thermal would ignore. observed_dT here
                    // is still computed above and drives face SELECTION (Phase B).

                    // Save previous-iter snapshots for the NEXT iter.
                    prev_temps = surface_temps;
                    prev_max_indirect = face_max_indirect;
                    (void)total_indirect_per_face_sum;
                    (void)n_frozen_by_dT;
                    (void)n_frozen_by_freeze;
                    (void)max_temp_change;  // consumer (old pre-radiosity check) moved
                                            // post-re-thermal; kept in the stats reduction

                    // Step 3: Re-run thermal for selected faces with the updated
                    // flux BC (= iter-0 flux composition + indirect_flux). The GPU
                    // path mirrors iter-0's prepare_flux + cold-start + 2-eq+1-out
                    // internally and consumes the raw arrays directly. The CPU
                    // fallback still takes a pre-built combined_flux per face,
                    // so we build that here with the SAME composition as iter-0
                    // (Hayne for direct/reflected, ε for primary_ir, + indirect)
                    // — this ensures both paths give physics-consistent results.
                    std::cout << "  Re-running thermal model for " << faces_to_rerun.size() << " faces..." << std::endl;

                    // Check if GPU thermal is enabled
                    bool use_gpu_thermal_rerun = thermal_config.getPropertyBool("use_gpu_thermal", false);

                    if (use_gpu_thermal_rerun && faces_to_rerun.size() >= 100) {
                        // The seed guard, restated at the point of consumption:
                        // the GPU re-thermal is the ONLY consumer of the
                        // hand-off, so it must never be entered without one.
                        // Unreachable today (the same use_gpu_thermal property
                        // picks both solvers), but it is what keeps the gate
                        // above honest if a future CPU->GPU re-thermal
                        // escalation is added (BUGS.md PTM-78, PTM-94).
                        if (!rad_post_eq_T_expected || rad_post_eq_T.empty()) {
                            std::cerr << "\nFATAL: the GPU re-thermal was selected but no solver"
                                         " produced the post-equilibration T(z) hand-off it seeds"
                                         " from (BUGS.md PTM-78, PTM-94).\n"
                                      << "  Refusing to write results." << std::endl;
                            return 1;
                        }
                        std::cout << "  Using GPU thermal solver..." << std::endl;
                        bool gpu_success = runThermalForSelectedFaces_GPU(
                            mesh,
                            faces_to_rerun,
                            direct_solar_values,
                            primary_reflected_values,
                            primary_ir_values,
                            indirect_flux_series,
                            rad_post_eq_T,
                            times,
                            total_time_steps,
                            equilibration_steps,
                            thermal_config,
                            surface_temps,
                            face_albedos_rad,   // per-face A0 (per-face albedo scaling)
                            sim_config.thermal.gpu_devices,
                            sim_config.thermal.gpu_device_ids,
                            sim_config.thermal.gpu_inter_batch_pause_ms,
                            shared_rerun_pool,
                            want_post_coupling_state ? &rad_final_T       : nullptr,
                            want_post_coupling_state ? &rad_final_touched : nullptr
                        );
                        if (!gpu_success) {
                            std::cerr << "  GPU thermal solver failed, falling back to CPU" << std::endl;
                            use_gpu_thermal_rerun = false;
                        }
                    }

                    // CPU path needs pre-built combined_flux per face. Build it
                    // here with iter-0's composition + indirect_flux. Hayne is
                    // applied per-timestep. Only built when GPU path is unused
                    // or failed, to avoid wasting RAM (~5.2 GB for 1.8M faces ×
                    // 960 steps × 8 bytes × 0.27 selection fraction).
                    std::vector<std::vector<double>> combined_flux;
                    if (!use_gpu_thermal_rerun) {
                        // Hayne (2017) parameters — match iter-0.
                        const double a_coef = (planet.albedoCoef.size() >= 2)
                                            ? planet.albedoCoef[0] : 0.0;
                        const double b_coef = (planet.albedoCoef.size() >= 2)
                                            ? planet.albedoCoef[1] : 0.0;
                        const double S_solar = planet.S;
                        const bool   has_angular = (planet.albedoCoef.size() >= 2 && S_solar > 0.0);
                        // Hapke gate (main-scope planet already carries the gate).
                        const bool   use_hapke = planet.use_hapke_albedo
                                               && planet.hapke_albedo_w >= 0.0;
                        // Per-face w derived from face_albedos_rad[f] at the call
                        // site (hapkeWFromA0); body-level hapke_w no longer read.
                        const bool   use_incidence = has_angular
                                                   || (use_hapke && S_solar > 0.0);

                        combined_flux.resize(faces_to_rerun.size());
                        for (size_t idx = 0; idx < faces_to_rerun.size(); idx++) {
                            int f = faces_to_rerun[idx];
                            const double base_A = (static_cast<size_t>(f) < face_albedos_rad.size()) ? face_albedos_rad[f] : planet.albedo;
                            combined_flux[idx].resize(total_time_steps);
                            for (int t = 0; t < total_time_steps; t++) {
                                const size_t flux_idx = static_cast<size_t>(f)
                                                      * total_time_steps + t;
                                const double direct = direct_solar_values[flux_idx];
                                const double reflected = primary_reflected_values[flux_idx];
                                const double ir = primary_ir_values[flux_idx];
                                const double albedo = use_incidence
                                    ? applyIncidenceAlbedo(base_A, direct,
                                                           solarConstForStep(t, S_solar),
                                                           a_coef, b_coef, use_hapke, hapkeWFromA0(base_A))
                                    : base_A;
                                combined_flux[idx][t] = (1.0 - albedo) * direct
                                                      + (1.0 - albedo) * reflected
                                                      + planet.emissivity * ir
                                                      + indirect_flux_series[f][t];
                            }
                        }
                    }

                    if (!use_gpu_thermal_rerun) {
                        // The CPU re-thermal returns surface temperatures only —
                        // runThermalModelWithMatchedTimes keeps no T(z) column —
                        // so a chunk checkpoint written after a CPU re-thermal
                        // still carries iter-0's UN-COUPLED column on these faces.
                        // Say so; do not let it pass in silence (BUGS.md PTM-57).
                        if (want_post_coupling_state) {
                            std::cerr << "  [Re-thermal] WARNING: CPU re-thermal path — no T(z)"
                                         " column is available, so the chunk checkpoint keeps"
                                         " iter-0's UN-COUPLED column for these "
                                      << faces_to_rerun.size() << " faces (BUGS.md PTM-57)."
                                         " The next chunk will cold-start them.\n";
                        }
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

                    // ---- Post-re-thermal convergence check (MOVED here from before
                    // the radiosity pass). Measures the temperature change this
                    // re-thermal actually produced (surface_temps vs the prev_temps
                    // snapshot taken just before it) -- the same quantity the old
                    // pre-radiosity check measured one iteration later, so the
                    // converge decision is identical but we stop BEFORE the next
                    // iteration's redundant indirect pass. Fermi verified that skipped
                    // pass changes 0 temperatures; the band-separated indirect OUTPUT
                    // arrays (indirect_solar/ir) are thus one Picard step stale, within
                    // rad_coupling_temp_tol. Iteration cap (max_coupling_iterations)
                    // unchanged. ----
                    {
                        double max_change_post = 0.0;
                        size_t count_changing = 0;  // faces (physics) / samples (legacy)
                        for (size_t f = 0; f < num_faces; f++) {
                            double odT = 0.0;
                            for (int t = 0; t < total_time_steps; t++) {
                                const double dT = std::abs(surface_temps[f][t] - prev_temps[f][t]);
                                if (dT > odT) odT = dT;
                                if (!rad_coupling_physics_mode && dT > rad_coupling_temp_tol) ++count_changing;
                            }
                            if (odT > max_change_post) max_change_post = odT;
                            if (rad_coupling_physics_mode && odT > rad_coupling_temp_tol) ++count_changing;
                        }
                        const double frac_denom = rad_coupling_physics_mode
                            ? static_cast<double>(num_faces)
                            : static_cast<double>(num_faces) * total_time_steps;
                        const double face_frac = static_cast<double>(count_changing) / frac_denom;
                        std::cout << "  Max temperature change: " << max_change_post << " K" << std::endl;
                        std::cout << (rad_coupling_physics_mode
                                       ? "  Faces still changing:   "
                                       : "  Samples still changing: ")
                                  << std::scientific << std::setprecision(2)
                                  << (face_frac * 100) << "%" << std::fixed << std::endl;
                        // PTM-89: the max-temperature exit bounds the residual on
                        // EVERY face by rad_coupling_temp_tol. "fixed" mode drops
                        // it as well, because any data-driven stopping rule is a
                        // function of the window and so can stop a short window
                        // an iteration earlier than a long one.
                        if (max_change_post < rad_coupling_temp_tol
                                && !rad_coupling_fixed_iters) {
                            std::cout << "  Converged on max-temperature criterion ("
                                      << max_change_post << " K < " << rad_coupling_temp_tol
                                      << " K)" << std::endl;
                            break;
                        }
                        // PTM-89: the face-fraction exit has NO error bound -- it
                        // stops while up to coupling_face_fraction of the mesh is
                        // still moving by an arbitrary amount, and the count is a
                        // max over the window, so a short window reaches it first.
                        // Retained only under coupling_mode: legacy.
                        if (rad_coupling_legacy_mode
                                && face_frac < rad_coupling_face_frac) {
                            std::cout << "  Converged on face-fraction criterion ("
                                      << std::scientific << std::setprecision(2)
                                      << (face_frac * 100) << "% < "
                                      << (rad_coupling_face_frac * 100) << "%)"
                                      << std::fixed << std::endl;
                            break;
                        }
                    }
                }

                // ---- PTM-57: re-archive the checkpoint POST-coupling --------
                // thermal.final_state_file was already written by the iter-0
                // solver, which returns BEFORE this block runs — so on disk it
                // holds the UN-COUPLED column, and the next chunk cold-starts
                // its whole 30-layer column on exactly the faces where indirect
                // radiation sets the temperature (measured: 24.36 K archived vs
                // 39.5 K in the concurrent output frame on a PSR floor).
                // Overwrite it with iter-0's column patched by the coupled
                // end-of-window column on every face the re-thermal touched.
                // Faces never re-thermal'd keep iter-0's column, which is the
                // right answer for them. Writing here (rather than deferring
                // the first write) keeps an un-coupled run byte-identical and
                // leaves a usable checkpoint if the coupling stage dies.
                if (want_post_coupling_state) {
                    const std::string& spath = sim_config.thermal.final_state_file;
                    size_t n_touched = 0;
                    for (uint8_t b : rad_final_touched) n_touched += (b != 0);

                    std::ifstream sin(spath, std::ios::binary);
                    uint32_t magic = 0, f_nf = 0, f_nl = 0;
                    std::vector<float> merged;
                    bool ok = false;
                    if (sin) {
                        sin.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
                        sin.read(reinterpret_cast<char*>(&f_nf),  sizeof(uint32_t));
                        sin.read(reinterpret_cast<char*>(&f_nl),  sizeof(uint32_t));
                        if (sin && magic == 0x50544D54u
                                && f_nf == static_cast<uint32_t>(num_faces)
                                && static_cast<size_t>(f_nl) * f_nf == rad_final_T.size()) {
                            merged.resize(rad_final_T.size());
                            sin.read(reinterpret_cast<char*>(merged.data()),
                                     static_cast<std::streamsize>(merged.size()) * sizeof(float));
                            ok = static_cast<bool>(sin);
                        }
                    }
                    if (!ok) {
                        std::cerr << "[PTM-57] WARNING: cannot re-read the iter-0 checkpoint '"
                                  << spath << "' (magic=0x" << std::hex << magic << std::dec
                                  << ", faces=" << f_nf << ", layers=" << f_nl
                                  << "); the archived state stays PRE-COUPLING and the next"
                                     " chunk will cold-start its radiation-fed faces.\n";
                    } else if (n_touched == 0) {
                        std::cout << "[PTM-57] No face was re-thermal'd; the checkpoint is"
                                     " already the coupled state." << std::endl;
                    } else {
                        for (size_t f = 0; f < num_faces; ++f) {
                            if (!rad_final_touched[f]) continue;
                            for (uint32_t l = 0; l < f_nl; ++l) {
                                merged[static_cast<size_t>(l) * num_faces + f] =
                                    rad_final_T[static_cast<size_t>(l) * num_faces + f];
                            }
                        }
                        std::ofstream so(spath, std::ios::binary | std::ios::trunc);
                        if (!so) {
                            std::cerr << "[PTM-57] WARNING: cannot reopen '" << spath
                                      << "' for the post-coupling rewrite; the archived state"
                                         " stays PRE-COUPLING.\n";
                        } else {
                            const uint32_t m = 0x50544D54u;
                            const uint32_t nf = static_cast<uint32_t>(num_faces);
                            so.write(reinterpret_cast<const char*>(&m),    sizeof(uint32_t));
                            so.write(reinterpret_cast<const char*>(&nf),   sizeof(uint32_t));
                            so.write(reinterpret_cast<const char*>(&f_nl), sizeof(uint32_t));
                            so.write(reinterpret_cast<const char*>(merged.data()),
                                     static_cast<std::streamsize>(merged.size()) * sizeof(float));
                            if (!so.good()) {
                                std::cerr << "[PTM-57] WARNING: short write on the post-coupling"
                                             " checkpoint rewrite.\n";
                            } else {
                                std::cout << "[PTM-57] Re-archived POST-coupling T(z) checkpoint: "
                                          << n_touched << "/" << num_faces
                                          << " faces carry the coupled column -> " << spath
                                          << std::endl;
                            }
                        }
                    }
                }

                // Print final statistics
                auto stats = rad_manager.getStatistics();
                std::cout << "\n=== Radiation Exchange Complete ===" << std::endl;
                std::cout << "  Mean indirect flux: " << stats.mean_indirect_flux << " W/m2" << std::endl;
                std::cout << "  Max indirect flux: " << stats.max_indirect_flux << " W/m2" << std::endl;

                // Equilibration-phase indirect flux: the run record. A default
                // that changes the deep column of every radiation-coupled run
                // has to say, in the log, whether it fired and on what numbers.
                {
                    const std::ios_base::fmtflags fl = std::cout.flags();
                    const std::streamsize pr = std::cout.precision();
                    std::cout << std::defaultfloat << std::setprecision(6);
                    std::cout << "  Equilibration indirect flux: "
                              << eq_indirect_record.provenance;
                    if (!eq_indirect_record.reason.empty())
                        std::cout << " (" << eq_indirect_record.reason << ")";
                    std::cout << std::endl;
                    if (eq_indirect_record.provenance != "none") {
                        std::cout << "    per-face <q_ind>: mean "
                                  << eq_indirect_record.mean_Wm2 << ", median "
                                  << eq_indirect_record.median_Wm2 << ", max "
                                  << eq_indirect_record.max_Wm2 << " W/m^2 over "
                                  << eq_indirect_record.n_nonzero << "/" << num_faces
                                  << " faces receiving any" << std::endl;
                    }
                    std::cout.flags(fl);
                    std::cout.precision(pr);
                }

                // Deep radiation-coupling (pass 1): dump the per-face OUTPUT-window
                // time-mean indirect flux <q_ind>[f] so a follow-up run can feed it
                // into equilibration (thermal.eq_indirect_flux_file). Format:
                // uint32 magic 'QIND', uint32 n_faces, n_faces float64.
                if (!sim_config.thermal.indirect_mean_out_file.empty()
                        && !indirect_flux_series.empty()) {
                    const std::string& opath = sim_config.thermal.indirect_mean_out_file;
                    const uint32_t nf = static_cast<uint32_t>(indirect_flux_series.size());
                    const int out_lo = equilibration_steps;
                    const int out_hi = total_time_steps;
                    const int out_n = std::max(1, out_hi - out_lo);
                    std::vector<double> q_ind_mean(nf, 0.0);
                    #pragma omp parallel for schedule(static)
                    for (uint32_t f = 0; f < nf; ++f) {
                        const auto& row = indirect_flux_series[f];
                        double s = 0.0;
                        for (int t = out_lo; t < out_hi && t < static_cast<int>(row.size()); ++t)
                            s += row[t];
                        q_ind_mean[f] = s / out_n;
                    }
                    std::ofstream qo(opath, std::ios::binary);
                    if (qo) {
                        const uint32_t magic = 0x51494E44u;  // 'QIND'
                        qo.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                        qo.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
                        qo.write(reinterpret_cast<const char*>(q_ind_mean.data()),
                                 static_cast<std::streamsize>(nf) * sizeof(double));
                        double s = 0.0, mx = 0.0;
                        for (double q : q_ind_mean) { s += q; mx = std::max(mx, q); }
                        std::cout << "[Deep coupling] Wrote per-face mean indirect flux ("
                                  << nf << " faces, mean " << (s / std::max<uint32_t>(1, nf))
                                  << ", max " << mx << " W/m^2) -> " << opath << std::endl;
                    } else {
                        std::cerr << "Warning: cannot write indirect_mean_out_file '"
                                  << opath << "'." << std::endl;
                    }
                }

                // Per-timestep indirect-flux dump (env-gated) for offline flux
                // analysis: PTM_DUMP_QIND_SERIES=<path>. Writes the per-face
                // OUTPUT-window indirect (IR + reflected) radiosity flux series,
                // aligned 1:1 with the saved direct_solar/illumination step files,
                // so a script can form the total absorbed flux = direct + indirect
                // and reduce it to per-face max/mean (the coupling effect on flux,
                // which is otherwise visible only in temperature). Format:
                // uint32 magic 'QIDT', uint32 n_faces, uint32 n_steps, then
                // n_faces*n_steps float32 (face-major: all steps of face 0, ...).
                if (const char* qsp = std::getenv("PTM_DUMP_QIND_SERIES")) {
                    if (!indirect_flux_series.empty()) {
                        const uint32_t nf = static_cast<uint32_t>(indirect_flux_series.size());
                        const int out_lo = equilibration_steps;
                        const int out_hi = total_time_steps;
                        const uint32_t ns = static_cast<uint32_t>(std::max(0, out_hi - out_lo));
                        std::ofstream qo(qsp, std::ios::binary);
                        if (qo) {
                            const uint32_t magic = 0x51494454u;  // 'QIDT'
                            qo.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                            qo.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
                            qo.write(reinterpret_cast<const char*>(&ns), sizeof(ns));
                            std::vector<float> rowbuf(ns);
                            for (uint32_t f = 0; f < nf; ++f) {
                                const auto& row = indirect_flux_series[f];
                                for (uint32_t j = 0; j < ns; ++j) {
                                    const int t = out_lo + static_cast<int>(j);
                                    rowbuf[j] = (t < static_cast<int>(row.size()))
                                        ? static_cast<float>(row[t]) : 0.0f;
                                }
                                qo.write(reinterpret_cast<const char*>(rowbuf.data()),
                                         static_cast<std::streamsize>(ns) * sizeof(float));
                            }
                            std::cout << "[Deep coupling] Wrote per-timestep indirect flux series ("
                                      << nf << " faces x " << ns << " steps) -> " << qsp << std::endl;
                        } else {
                            std::cerr << "Warning: cannot write PTM_DUMP_QIND_SERIES '"
                                      << qsp << "'." << std::endl;
                        }
                    }
                }

                // Reflected-IR diagnostic (Lucey & Hayne 2021): for each face,
                // the peak surface T over the output window and the radiosity
                // indirect flux AT that instant. A follow-up script forms the
                // apparent bolometric T = [(eps*sigma*T^4 + (1-eps)*I_inc)/sigma]^1/4
                // (I_inc ~ Q_indirect/eps) to estimate the wall-IR reflected
                // toward Diviner but absent from the emission-only comparison.
                // Env-gated: PTM_DUMP_REFLIR=<path>. Format: uint32 magic 'RFIR',
                // uint32 n_faces, then n_faces*(float64 Tpeak, float64 q_at_peak).
                if (const char* rp = std::getenv("PTM_DUMP_REFLIR")) {
                    if (!indirect_flux_series.empty() && !surface_temps.empty()) {
                        const uint32_t nf = static_cast<uint32_t>(indirect_flux_series.size());
                        const int out_lo = equilibration_steps;
                        const int out_hi = total_time_steps;
                        std::vector<double> pk(static_cast<size_t>(nf) * 2, 0.0);
                        #pragma omp parallel for schedule(static)
                        for (uint32_t f = 0; f < nf; ++f) {
                            const auto& Trow = surface_temps[f];
                            const auto& Qrow = indirect_flux_series[f];
                            double tmax = 0.0, qat = 0.0;
                            for (int t = out_lo; t < out_hi && t < static_cast<int>(Trow.size()); ++t) {
                                if (Trow[t] > tmax) {
                                    tmax = Trow[t];
                                    qat = (t < static_cast<int>(Qrow.size())) ? Qrow[t] : 0.0;
                                }
                            }
                            pk[2 * f] = tmax;
                            pk[2 * f + 1] = qat;
                        }
                        std::ofstream ro(rp, std::ios::binary);
                        if (ro) {
                            const uint32_t magic = 0x52464952u;  // 'RFIR'
                            ro.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                            ro.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
                            ro.write(reinterpret_cast<const char*>(pk.data()),
                                     static_cast<std::streamsize>(nf) * 2 * sizeof(double));
                            std::cout << "[Reflected-IR] Wrote per-face (Tpeak, q@peak) for "
                                      << nf << " faces -> " << rp << std::endl;
                        }
                    }
                }
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
        // The window math lives in computeIntegrationWindow() (see its PTM-15 note),
        // shared with the early publication that drives the effective-albedo
        // accumulators. Same inputs, same call, so the two can never diverge.
        const IntegrationWindow iw = computeIntegrationWindow(
            equilibration_steps, total_time_steps, output_steps_per_day,
            total_simulation_days, integration_days, output_days,
            integration_steps_cli, integration_offset_steps_cli,
            integration_offset_days);
        stats_start_step = iw.start;
        stats_steps      = iw.steps;

        // The accumulators were told this window before the solve. If anything
        // upstream changed the step counts in between, the file would silently
        // describe a different window from its neighbors -- fail loudly instead.
        if (g_aeff_win_set &&
            (g_aeff_win_start != stats_start_step || g_aeff_win_steps != stats_steps)) {
            std::cerr << "WARNING: the effective-albedo accumulation window ["
                      << g_aeff_win_start << ","
                      << (g_aeff_win_start + g_aeff_win_steps)
                      << ") does NOT match the statistics window [" << stats_start_step
                      << "," << (stats_start_step + stats_steps)
                      << "); statistics/effective_albedo.bin is NOT comparable with the "
                         "other statistics/*.bin files for this run.\n";
        }

        std::cout << "\n[Integration] Statistics interval\n"
                << "  Start (abs step): " << stats_start_step << "\n"
                << "  Duration (steps): " << stats_steps << "\n";

        // Make silence impossible (PTM-15 archetype): the per-timestep output frames
        // cover abs steps [equilibration_steps, equilibration_steps+time_steps); if the
        // statistics window does NOT cover all of them, the _max/_min/_avg files are
        // reduced over a STRICT SUBSET of the written frames — a consumer reading
        // direct_solar_max.bin as a global per-face max would be wrong. Warn loudly.
        {
            const int out_first = equilibration_steps;
            const int out_last  = equilibration_steps + time_steps; // one past end
            const int stat_last = stats_start_step + stats_steps;   // one past end
            if (stats_start_step > out_first || stat_last < out_last) {
                std::cerr << "WARNING: statistics window [" << stats_start_step << ","
                          << stat_last << ") does NOT cover all written output frames ["
                          << out_first << "," << out_last << "); statistics/_max, _min and"
                             " _avg are reduced over that SUBSET only and can under-report"
                             " the per-frame extrema (PTM-15). Set integration_days >= the"
                             " output span (and integration_offset_days=0) for a full-window"
                             " reduction.\n";
            }
        }

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

        // PTM-107(a): the operative per-face effective albedo, so an external energy-
        // closure audit no longer has to treat the BC albedo as a free parameter.
        // Only meaningful for a run that actually integrated a thermal model --
        // an illumination-only run has no surface BC and therefore no operative
        // absorbed-beam albedo to report.
        if (run_thermal_mesh) {
            writeEffectiveAlbedo(output_dir);
            reportSourceAlbedoConsistency();
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

    // ===== Band-separated INDIRECT flux outputs (Paul's product) =====
    // indirect_solar (absorbed multiply-scattered solar) + indirect_ir (absorbed
    // thermal-IR), per-timestep frames + statistics, mirroring direct_solar /
    // primary_ir. ABSORBED convention (see writeIndirectBandOutputs). Emitted only
    // for coupled two-band runs where the bands were captured; radoff / gray /
    // multi-GPU-pool runs OMIT them (no radiosity band split available), which
    // keeps those runs' pre-existing outputs byte-identical.
    if (output_format == "binary" && run_thermal_mesh
            && !indirect_solar_series.empty()) {
        if (indirect_bands_available && !indirect_bands_batch_gap) {
            const bool write_frames = !sim_config.output.skip_time_series;
            bool ib_ok = writeIndirectBandOutputs(
                output_dir, mesh,
                indirect_solar_series, indirect_ir_series,
                equilibration_steps, time_steps,   // frame window (output)
                stats_start_step, stats_steps,     // stats window (PTM-15)
                write_frames, compute_statistics);
            if (!ib_ok) {
                std::cerr << "[indirect-bands] WARNING: one or more indirect-band "
                             "outputs failed to write.\n";
            }
        } else {
            std::cerr << "[indirect-bands] Band-separated indirect outputs OMITTED "
                         "(bands_available=" << indirect_bands_available
                      << ", batch_gap=" << indirect_bands_batch_gap
                      << "). The indirect_flux_series total is unaffected.\n";
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
            << avg_flux << " W/m^2" << std::endl;
    std::cout << "    - Direct solar: " << avg_direct << " W/m^2" << std::endl;
    std::cout << "    - Primary reflected: " << avg_reflected << " W/m^2" << std::endl;
    std::cout << "    - Primary IR: " << avg_ir << " W/m^2" << std::endl;
    std::cout << "  Maximum total flux: " << max_flux << " W/m^2" << std::endl;
    std::cout << "    - Direct solar: " << max_direct << " W/m^2" << std::endl;
    std::cout << "    - Primary reflected: " << max_reflected << " W/m^2" << std::endl;
    std::cout << "    - Primary IR: " << max_ir << " W/m^2" << std::endl;
    std::cout << "  Minimum flux (non-zero): " << min_flux << " W/m^2" << std::endl;
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
