// lunar_property_map_generator.cpp
//
// Offline tool: bilinear-sample the Diviner-derived 128 ppd H-parameter raster
// onto icosphere face centers, compute exp(-z/zs)-weighted depth-averaged thermal
// inertia at 273 K, and write per-face binary maps for use by ptm_thermal_sim.
//
// Raster format (from /Users/paha3326/.../moon/hpar/dataset/test-file-io.ipynb):
//   dtype:           float32, little-endian, row-major
//   pixels-per-deg:  128
//   samples/line:    360 * 128 + 1     = 46081  (both 0 and 360 endpoints)
//   lines:           140 * 128 + 1     = 17921  (both +70 and -70 endpoints)
//   upper-left:      (lat = +70 deg, lon = 0 deg)
//   total bytes:     17921 * 46081 * 4 = 3,303,270,404
//
// Outputs (per ico level N):
//   <out>/hpar_<body>_ico<N>.bin     float32 x n_faces  (H per face, m)
//   <out>/ti273k_<body>_ico<N>.bin   float32 x n_faces  (TI per face, J m^-2 K^-1 s^-1/2)
//   <out>/mask_<body>_ico<N>.bin     uint8   x n_faces  (1 = sampled from raster)
//
// Usage:
//   lunar_property_map_generator --input <hpar.bin> --output-dir <dir>
//                                [--ico-level N | --all-levels A:B]
//                                [--body moon]

#include "../mesh/mesh_utils.h"
#include "../thermal/planetary_properties.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// 128 ppd raster geometry from the Diviner H-par dataset
constexpr int    kPPD             = 128;
constexpr int    kSamplesPerLine  = 360 * kPPD + 1;   // 46081
constexpr int    kLines           = 140 * kPPD + 1;   // 17921
constexpr double kLat0            = 70.0;             // upper-left latitude
constexpr double kLatMin          = -70.0;
constexpr double kDl              = 1.0 / kPPD;       // degrees per pixel

// Memory-mapped (or fully loaded) H raster
struct HRaster {
    std::vector<float> data;  // size = kLines * kSamplesPerLine
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            std::cerr << "ERROR: cannot open " << path << "\n";
            return false;
        }
        std::streamsize bytes = f.tellg();
        const std::streamsize expected =
            static_cast<std::streamsize>(kLines) *
            static_cast<std::streamsize>(kSamplesPerLine) * 4;
        if (bytes != expected) {
            std::cerr << "ERROR: " << path << " size " << bytes
                      << " != expected " << expected << "\n";
            return false;
        }
        f.seekg(0, std::ios::beg);
        data.resize(static_cast<size_t>(kLines) * kSamplesPerLine);

        // Read in 64 MB chunks. Single fread > 2 GB can return short on
        // OneDrive-mounted files on macOS; chunked reads are robust.
        char* buf = reinterpret_cast<char*>(data.data());
        constexpr std::streamsize kChunk = 64 * 1024 * 1024;
        std::streamsize remaining = bytes;
        std::streamsize read_total = 0;
        while (remaining > 0) {
            const std::streamsize this_chunk = std::min(kChunk, remaining);
            f.read(buf + read_total, this_chunk);
            const std::streamsize got = f.gcount();
            if (got <= 0) {
                std::cerr << "ERROR: short read from " << path
                          << " at offset " << read_total
                          << " (got " << got << " of " << this_chunk << ")\n";
                return false;
            }
            read_total += got;
            remaining  -= got;
            if (read_total % (1024LL * 1024LL * 1024LL) == 0) {
                std::cerr << "  read " << (read_total / (1024 * 1024)) << " MB\n";
            }
        }
        loaded = true;
        return true;
    }

    // Box-average the raster over a square of half-width `half_box_deg` in
    // latitude and longitude (cosine-corrected for longitude convergence near
    // the poles). When half_box_deg = 0 this falls back to a single bilinear
    // sample. Otherwise we average all valid pixels in the box, matching what
    // Hayne et al. (2017) do at 1/10° in their published map: per-face values
    // become representative of the face's solid-angle footprint instead of a
    // single noisy raster pixel.
    bool sample(double lat, double lon, double& out_value,
                double half_box_deg = 0.0) const {
        if (!loaded) return false;
        if (lat > kLat0 || lat < kLatMin) return false;

        // Normalize longitude to [0, 360)
        double lon_n = std::fmod(lon, 360.0);
        if (lon_n < 0.0) lon_n += 360.0;

        // Box-average path -------------------------------------------------
        if (half_box_deg > 0.0) {
            // Latitude rows
            const double row_lo = (kLat0 - (lat + half_box_deg)) / kDl;
            const double row_hi = (kLat0 - (lat - half_box_deg)) / kDl;
            int rmin = std::max(0,         static_cast<int>(std::floor(row_lo)));
            int rmax = std::min(kLines-1,  static_cast<int>(std::ceil (row_hi)));
            if (rmax < rmin) return false;

            // Longitude columns (wrap-aware): cos(lat) compensates for
            // converging meridians so high-latitude box samples cover
            // approximately the same solid angle as the equator.
            const double cos_lat = std::cos(lat * M_PI / 180.0);
            const double lon_half = half_box_deg / std::max(0.05, cos_lat);
            const double col_lo = (lon_n - lon_half) / kDl;
            const double col_hi = (lon_n + lon_half) / kDl;
            int cmin = static_cast<int>(std::floor(col_lo));
            int cmax = static_cast<int>(std::ceil (col_hi));

            double sum = 0.0;
            std::size_t n = 0;
            for (int r = rmin; r <= rmax; ++r) {
                for (int cc = cmin; cc <= cmax; ++cc) {
                    int c = cc % (kSamplesPerLine - 1);  // exclude the 360° dup
                    if (c < 0) c += (kSamplesPerLine - 1);
                    const float v = data[static_cast<std::size_t>(r) *
                                          kSamplesPerLine + c];
                    if (std::isfinite(v) && v > 0.0) {
                        sum += v;
                        ++n;
                    }
                }
            }
            if (n == 0) return false;
            out_value = sum / static_cast<double>(n);
            return true;
        }

        // Bilinear point-sample fallback (legacy) --------------------------
        // Pixel coordinates: row 0 is at lat0=+70, row increases southward.
        double row_f = (kLat0 - lat) / kDl;
        double col_f = lon_n / kDl;

        int r0 = static_cast<int>(std::floor(row_f));
        int c0 = static_cast<int>(std::floor(col_f));
        double rt = row_f - r0;
        double ct = col_f - c0;

        if (r0 < 0) { r0 = 0; rt = 0.0; }
        if (r0 >= kLines - 1) { r0 = kLines - 2; rt = 1.0; }

        // Wrap longitude column at 360 -> 0 (column kSamplesPerLine-1 == column 0)
        int c1 = c0 + 1;
        if (c0 < 0) c0 = 0;
        if (c1 >= kSamplesPerLine) c1 = 0;

        const float v00 = data[static_cast<size_t>(r0)     * kSamplesPerLine + c0];
        const float v01 = data[static_cast<size_t>(r0)     * kSamplesPerLine + c1];
        const float v10 = data[static_cast<size_t>(r0 + 1) * kSamplesPerLine + c0];
        const float v11 = data[static_cast<size_t>(r0 + 1) * kSamplesPerLine + c1];

        // Bilinear; treat any negative or NaN samples as missing (skip the corner).
        double w00 = (1.0 - rt) * (1.0 - ct);
        double w01 = (1.0 - rt) * ct;
        double w10 = rt * (1.0 - ct);
        double w11 = rt * ct;

        double sum_v = 0.0, sum_w = 0.0;
        auto add = [&](double v, double w) {
            if (std::isfinite(v) && v > 0.0) {
                sum_v += v * w;
                sum_w += w;
            }
        };
        add(v00, w00);
        add(v01, w01);
        add(v10, w10);
        add(v11, w11);

        if (sum_w < 0.5) return false;  // mostly missing pixels; treat as out-of-band
        out_value = sum_v / sum_w;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Heat-capacity polynomial (Moon: Hayne et al. 2017 Table A1)
// cp(T) = cp0 + sum_j cpCoeff[j] * T^(j+1)
// ---------------------------------------------------------------------------
double cpAtT(const thermal::ThermophysicalProperties& props, double T) {
    double cp = props.cp0;
    double T_pow = T;
    for (size_t j = 0; j < props.cpCoeff.size(); ++j) {
        cp += props.cpCoeff[j] * T_pow;
        T_pow *= T;
    }
    if (cp < 100.0) cp = 600.0;  // clamp like thermal_model.cpp:651
    return cp;
}

// ---------------------------------------------------------------------------
// Diurnal skin depth from H-derived effective properties.
//
// We emulate the same convention used by thermal_model.cpp (single-skin formula
// `skinDepth(P, kappa) = sqrt(kappa * P / pi)`), but for a layered profile we
// take the *exp(-z/H)-weighted depth average* of k(z) and rho(z), since H is
// what controls the regolith stratification. cp is taken at the reference
// temperature (273 K) consistent with the TI definition.
//
// k_avg = ∫(kd + (ks-kd)·e^{-z/H}) e^{-z/zs}/zs dz, etc.
// To avoid a circular dependency on zs, we use H itself as the weight scale
// for the *initial* estimate, which closely matches the diurnal skin depth
// for typical lunar regolith (zs ~ 0.05–0.2 m, H ~ 0.06 m). We then iterate
// once for self-consistency. In practice one iteration is enough.
// ---------------------------------------------------------------------------
struct EffectiveProps {
    double k_eff;    // W/m/K
    double rho_eff;  // kg/m^3
    double cp;       // J/kg/K
    double zs;       // m, diurnal skin depth
};

EffectiveProps effectiveProps(const thermal::ThermophysicalProperties& body, double H_face) {
    const double cp273 = cpAtT(body, 273.0);

    // k(z) and rho(z) integrated from 0 to infinity with weight exp(-z/L)/L.
    // ∫(kd + (ks-kd) e^{-z/H}) e^{-z/L}/L dz = kd + (ks - kd) * H / (H + L)
    // ∫(rhod + (rhos-rhod) e^{-z/H}) e^{-z/L}/L dz = rhod + (rhos - rhod) * H / (H + L)
    auto avg_k   = [&](double L) { return body.kd  + (body.ks  - body.kd ) * H_face / (H_face + L); };
    auto avg_rho = [&](double L) { return body.rhod + (body.rhos - body.rhod) * H_face / (H_face + L); };

    // Iterate to self-consistency: zs depends on the (k, rho) average that
    // depends on zs. Two-fixed-point iterations are sufficient.
    double zs = H_face;  // initial guess
    for (int it = 0; it < 4; ++it) {
        const double k   = avg_k(zs);
        const double rho = avg_rho(zs);
        const double kappa = k / (rho * cp273);
        zs = std::sqrt(kappa * body.day / M_PI);
        if (!(zs > 1e-4)) zs = 1e-4;
    }
    EffectiveProps r;
    r.k_eff   = avg_k(zs);
    r.rho_eff = avg_rho(zs);
    r.cp      = cp273;
    r.zs      = zs;
    return r;
}

// ---------------------------------------------------------------------------
// Depth-weighted TI(273 K):
//   TI(z) = sqrt( k(z) * rho(z) * cp(273) )
//   <TI> = ∫_0^{10·zs} TI(z) e^{-z/zs} dz / zs   (Simpson, 200 panels)
// ---------------------------------------------------------------------------
double depthWeightedTI(const thermal::ThermophysicalProperties& body, double H_face) {
    const EffectiveProps ep = effectiveProps(body, H_face);
    const double cp = ep.cp;
    const double zs = ep.zs;
    const double zmax = 10.0 * zs;
    const int n = 200;  // even
    const double dz = zmax / n;

    auto integrand = [&](double z) {
        const double df = std::exp(-z / H_face);
        const double kz = body.kd  + (body.ks  - body.kd ) * df;
        const double rz = body.rhod + (body.rhos - body.rhod) * df;
        const double TI = std::sqrt(kz * rz * cp);
        return TI * std::exp(-z / zs);
    };

    // Simpson's rule
    double sum = integrand(0.0) + integrand(zmax);
    for (int i = 1; i < n; ++i) {
        const double z = i * dz;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * integrand(z);
    }
    const double integral = sum * dz / 3.0;
    return integral / zs;
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
struct Args {
    std::string input;
    std::string output_dir = "data/property_maps/moon";
    std::string body = "moon";
    int level_lo = -1;
    int level_hi = -1;  // inclusive; if -1 -> single level == level_lo
};

void usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " --input <hpar.bin> --output-dir <dir>\n"
        "       (--ico-level N | --all-levels A:B)\n"
        "       [--body moon]\n";
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: " << s << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if      (s == "--input")        a.input = next();
        else if (s == "--output-dir")   a.output_dir = next();
        else if (s == "--body")         a.body = next();
        else if (s == "--ico-level")    { a.level_lo = std::stoi(next()); a.level_hi = a.level_lo; }
        else if (s == "--all-levels")   {
            std::string r = next();
            auto colon = r.find(':');
            if (colon == std::string::npos) { std::cerr << "ERROR: --all-levels A:B\n"; return false; }
            a.level_lo = std::stoi(r.substr(0, colon));
            a.level_hi = std::stoi(r.substr(colon + 1));
        }
        else if (s == "--help" || s == "-h") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "ERROR: unknown arg " << s << "\n"; return false; }
    }
    if (a.input.empty() || a.level_lo < 0) {
        usage(argv[0]);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    // Load body defaults (used to fill polar fallback faces and to compute TI)
    thermal::ThermophysicalProperties body;
    {
        std::string lower = args.body;
        for (auto& c : lower) c = std::tolower(c);
        if      (lower == "moon")    body = thermal::PlanetaryData::getProperties(thermal::PlanetaryBody::MOON);
        else { std::cerr << "ERROR: only --body moon is supported in v1\n"; return 1; }
    }

    std::cout << "Body: " << args.body
              << "  ks=" << body.ks << "  kd=" << body.kd
              << "  rho_s=" << body.rhos << "  rho_d=" << body.rhod
              << "  H_default=" << body.H << " m\n";

    // Load the H-par raster once; reused across all ico levels.
    HRaster raster;
    std::cout << "Loading H-par raster from " << args.input << " ("
              << kLines << " x " << kSamplesPerLine << " float32)...\n";
    if (!raster.load(args.input)) return 1;
    std::cout << "  loaded " << raster.data.size() << " floats\n";

    std::filesystem::create_directories(args.output_dir);

    for (int level = args.level_lo; level <= args.level_hi; ++level) {
        std::cout << "\n--- ico level " << level << " ---\n";

        ptm::MeshUtils mu;
        mu.setBodyRadius(body.body_radius_km);
        mu.setMeshResolution(level);    // also calls setupMesh() -> mapFacesToLatLon()
        auto mesh = mu.getMesh();
        const size_t n = mesh->faces.size();
        std::cout << "  faces: " << n << "\n";

        // Per-face spatial averaging size: roughly half the angular face
        // spacing. An icosphere at level N has typical inter-face arc length
        // ~60°/2^N along the long edge of each spherical triangle. We average
        // raster pixels within a half-box equal to that, so that each face
        // value reflects the mean H over the patch it represents — matching
        // the 1/10° spatial binning convention of Hayne et al. (2017) when
        // the face spacing happens to be ~0.1°. For coarser meshes (e.g.
        // ico6 at 0.5°), we average more pixels and recover the spatial
        // smoothing. For finer meshes (ico10+ at < 0.05°), we approach a
        // single-pixel sample.
        const double half_box_deg = 30.0 / std::pow(2.0, level);
        std::cout << "  raster averaging half-box: " << half_box_deg
                  << " deg (≈ " << static_cast<int>(2 * half_box_deg / kDl)
                  << " pixels per axis)\n";

        std::vector<float> hpar(n);
        std::vector<float> ti(n);
        std::vector<uint8_t> mask(n);

        size_t n_default = 0;
        double h_min =  1e30, h_max = -1e30, h_sum = 0.0;
        double ti_min = 1e30, ti_max = -1e30, ti_sum = 0.0;

        for (size_t f = 0; f < n; ++f) {
            const double lat = mesh->latitudes[f];
            const double lon = mesh->longitudes[f];
            double H_value;
            const bool ok = raster.sample(lat, lon, H_value, half_box_deg);
            if (!ok || !(H_value > 0.0) || !std::isfinite(H_value)) {
                H_value = body.H;
                mask[f] = 0;
                ++n_default;
            } else {
                mask[f] = 1;
            }
            hpar[f] = static_cast<float>(H_value);
            const double TI = depthWeightedTI(body, H_value);
            ti[f] = static_cast<float>(TI);

            h_min = std::min(h_min, H_value); h_max = std::max(h_max, H_value); h_sum += H_value;
            ti_min = std::min(ti_min, TI);    ti_max = std::max(ti_max, TI);    ti_sum += TI;
        }

        const double frac_default = static_cast<double>(n_default) / static_cast<double>(n);

        std::cout << "  H:  min=" << h_min << "  max=" << h_max
                  << "  mean=" << (h_sum / n) << " m\n"
                  << "  TI: min=" << ti_min << "  max=" << ti_max
                  << "  mean=" << (ti_sum / n) << " J m^-2 K^-1 s^-1/2\n"
                  << "  fallback faces (outside ±70°): " << n_default
                  << " (" << (frac_default * 100.0) << "%)\n";

        auto writeBin = [&](const std::string& fname, const void* data, size_t bytes) {
            const auto path = std::filesystem::path(args.output_dir) / fname;
            std::ofstream out(path, std::ios::binary);
            if (!out) {
                std::cerr << "ERROR: cannot open for write: " << path << "\n";
                return false;
            }
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            if (!out) {
                std::cerr << "ERROR: short write to " << path << "\n";
                return false;
            }
            std::cout << "  wrote " << path << " (" << bytes << " bytes)\n";
            return true;
        };

        const std::string lvl = std::to_string(level);
        if (!writeBin("hpar_"   + args.body + "_ico" + lvl + ".bin",
                      hpar.data(), hpar.size() * sizeof(float))) return 1;
        if (!writeBin("ti273k_" + args.body + "_ico" + lvl + ".bin",
                      ti.data(), ti.size() * sizeof(float))) return 1;
        if (!writeBin("mask_"   + args.body + "_ico" + lvl + ".bin",
                      mask.data(), mask.size() * sizeof(uint8_t))) return 1;
    }

    return 0;
}
