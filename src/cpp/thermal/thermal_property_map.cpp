// thermal_property_map.cpp — see thermal_property_map.h for the contract.

#include "thermal_property_map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace thermal {

namespace {

// cp(T) polynomial evaluation matching planetary_properties.h convention
// (cp = cp0 + sum_j cpCoeff[j] * T^(j+1)).
double cpAtT(const ThermophysicalProperties& props, double T) {
    double cp = props.cp0;
    double T_pow = T;
    for (size_t j = 0; j < props.cpCoeff.size(); ++j) {
        cp += props.cpCoeff[j] * T_pow;
        T_pow *= T;
    }
    if (cp < 100.0) cp = 600.0;
    return cp;
}

// Diurnal skin depth from H-derived effective k(z), rho(z) at T=273K.
// Closed-form analytic depth average:
//   <k>  = kd + (ks - kd) * H/(H + L)
//   <rho>= rhod + (rhos - rhod) * H/(H + L)
// Iterate a fixed-point on L = zs(<k>, <rho>, cp(273)) since L itself depends
// on the average. Two iterations are sufficient.
double effectiveSkinDepth(const ThermophysicalProperties& body, double H_face) {
    const double cp273 = cpAtT(body, 273.0);
    auto avg_k   = [&](double L) { return body.kd  + (body.ks  - body.kd ) * H_face / (H_face + L); };
    auto avg_rho = [&](double L) { return body.rhod + (body.rhos - body.rhod) * H_face / (H_face + L); };
    double zs = H_face;
    for (int it = 0; it < 4; ++it) {
        const double k   = avg_k(zs);
        const double rho = avg_rho(zs);
        const double kappa = k / (rho * cp273);
        zs = std::sqrt(kappa * body.day / M_PI);
        if (!(zs > 1e-4)) zs = 1e-4;
    }
    return zs;
}

}  // namespace

double depthWeightedTI(const ThermophysicalProperties& body, double H_face) {
    const double cp273 = cpAtT(body, 273.0);
    const double zs    = effectiveSkinDepth(body, H_face);
    const double zmax  = 10.0 * zs;
    constexpr int n    = 200;
    const double dz    = zmax / n;
    auto integrand = [&](double z) {
        const double df = std::exp(-z / H_face);
        const double kz = body.kd  + (body.ks  - body.kd ) * df;
        const double rz = body.rhod + (body.rhos - body.rhod) * df;
        return std::sqrt(kz * rz * cp273) * std::exp(-z / zs);
    };
    double sum = integrand(0.0) + integrand(zmax);
    for (int i = 1; i < n; ++i) {
        sum += (i % 2 == 0 ? 2.0 : 4.0) * integrand(i * dz);
    }
    return (sum * dz / 3.0) / zs;
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------
bool ThermalPropertyMap::readBinaryFloat32(const std::string& path,
                                           std::size_t expected_n,
                                           std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "ThermalPropertyMap: cannot open " << path << "\n";
        return false;
    }
    const std::streamsize bytes = f.tellg();
    const std::streamsize need  = static_cast<std::streamsize>(expected_n) * 4;
    if (bytes != need) {
        std::cerr << "ThermalPropertyMap: " << path << " size " << bytes
                  << " bytes, expected " << need
                  << " (n_faces=" << expected_n << ")\n";
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize(expected_n);
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!f) {
        std::cerr << "ThermalPropertyMap: short read of " << path << "\n";
        return false;
    }
    return true;
}

bool ThermalPropertyMap::readBinaryUInt8(const std::string& path,
                                         std::size_t expected_n,
                                         std::vector<std::uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize bytes = f.tellg();
    if (bytes != static_cast<std::streamsize>(expected_n)) return false;
    f.seekg(0, std::ios::beg);
    out.resize(expected_n);
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return static_cast<bool>(f);
}

void ThermalPropertyMap::invertTIToH(const ThermophysicalProperties& body) {
    // For each face, find H such that depthWeightedTI(body, H) ≈ TI[f] via
    // bisection on H ∈ [H_LO, H_HI]. The forward map TI(H) is monotone
    // *decreasing* in H over this range (more diffusive surface layer ⇒
    // lower bulk TI), confirmed by spot-checks at H = 0.001..1.0 m.
    constexpr double kHLo = 1e-3;
    constexpr double kHHi = 1.0;
    constexpr double kTol = 1e-4;
    constexpr int    kMaxIter = 40;

    const double TI_at_lo = depthWeightedTI(body, kHLo);  // largest TI
    const double TI_at_hi = depthWeightedTI(body, kHHi);  // smallest TI
    const bool decreasing = (TI_at_lo > TI_at_hi);

    int n_clamped = 0;
    for (size_t f = 0; f < H_.size(); ++f) {
        const double TI_target = H_[f];   // currently storing TI; about to overwrite with H
        if (!std::isfinite(TI_target) || TI_target <= 0.0) {
            H_[f] = body.H;
            mask_[f] = 0;
            ++n_clamped;
            continue;
        }
        // Out-of-range clamp
        if ((decreasing && TI_target >= TI_at_lo) ||
            (!decreasing && TI_target <= TI_at_lo)) {
            H_[f] = kHLo;
            mask_[f] = 0;
            ++n_clamped;
            continue;
        }
        if ((decreasing && TI_target <= TI_at_hi) ||
            (!decreasing && TI_target >= TI_at_hi)) {
            H_[f] = kHHi;
            mask_[f] = 0;
            ++n_clamped;
            continue;
        }

        double lo = kHLo, hi = kHHi;
        double H_mid = 0.5 * (lo + hi);
        for (int it = 0; it < kMaxIter; ++it) {
            H_mid = 0.5 * (lo + hi);
            const double TI_mid = depthWeightedTI(body, H_mid);
            if (std::abs(TI_mid - TI_target) < kTol) break;
            const bool target_above = (TI_target > TI_mid);
            // If TI is decreasing in H, larger TI -> smaller H -> shrink hi to mid
            if (decreasing == target_above) hi = H_mid;
            else                            lo = H_mid;
        }
        H_[f] = H_mid;
    }
    if (n_clamped > 0) {
        std::cerr << "ThermalPropertyMap: " << n_clamped
                  << " face(s) had TI outside achievable range; clamped to body default\n";
    }
}

// ---------------------------------------------------------------------------
// Public loaders
// ---------------------------------------------------------------------------
std::optional<ThermalPropertyMap> ThermalPropertyMap::loadDefault(
    const std::string& body,
    int ico_level,
    const std::string& maps_root,
    Mode mode) {

    std::string body_lower = body;
    for (auto& c : body_lower) c = std::tolower(c);

    const std::string prefix = (mode == Mode::H_PARAMETER) ? "hpar_" : "ti273k_";
    const std::string fname  = prefix + body_lower + "_ico"
                             + std::to_string(ico_level) + ".bin";

    // Accept either a "root" path (data/property_maps) or a directory that
    // already ends with the body subdir (data/property_maps/moon). The latter
    // is what the GUI's directory picker often saves when the user navigates
    // into the body folder before clicking "Open".
    std::filesystem::path root(maps_root);
    std::filesystem::path path = root / body_lower / fname;
    if (!std::filesystem::exists(path)) {
        std::filesystem::path alt = root / fname;
        if (std::filesystem::exists(alt)) {
            path = alt;
        }
    }

    if (!std::filesystem::exists(path)) {
        std::cerr << "ThermalPropertyMap: bundled file not found at "
                  << (root / body_lower / fname) << " or " << (root / fname) << "\n";
        return std::nullopt;
    }
    const std::size_t n_faces = static_cast<std::size_t>(20) *
                                static_cast<std::size_t>(std::pow(4, ico_level));
    return loadCustom(path.string(), n_faces, mode);
}

std::optional<ThermalPropertyMap> ThermalPropertyMap::loadCustom(
    const std::string& path,
    std::size_t expected_n_faces,
    Mode mode) {

    ThermalPropertyMap m;
    std::vector<float> raw;
    if (!m.readBinaryFloat32(path, expected_n_faces, raw)) {
        return std::nullopt;
    }

    // Try to read sidecar mask (optional; defaults to all-valid)
    const std::string mask_path = path + ".mask";
    std::vector<std::uint8_t> mask;
    bool had_sidecar = m.readBinaryUInt8(mask_path, expected_n_faces, mask);
    if (!had_sidecar) {
        // Try the generator's mask file naming (mask_<body>_ico<N>.bin alongside)
        std::filesystem::path p(path);
        std::string fname = p.filename().string();
        std::string body_part;
        // strip leading "hpar_" or "ti273k_"
        for (const std::string& pref : {"hpar_", "ti273k_"}) {
            if (fname.rfind(pref, 0) == 0) {
                body_part = fname.substr(pref.size());
                break;
            }
        }
        if (!body_part.empty()) {
            std::filesystem::path companion = p.parent_path() / ("mask_" + body_part);
            had_sidecar = m.readBinaryUInt8(companion.string(),
                                            expected_n_faces, mask);
        }
    }
    if (!had_sidecar) {
        mask.assign(expected_n_faces, 1);
    }

    m.H_.assign(raw.begin(), raw.end());  // float -> double
    m.mask_ = std::move(mask);
    m.source_mode_ = mode;
    m.source_path_ = path;

    // For TI mode, invert each TI to H using body defaults. The caller can
    // override the body via the helper below if needed (Moon hardcoded for now).
    if (mode == Mode::THERMAL_INERTIA) {
        // Load Moon defaults; v1 only supports Moon. Future: dispatch on body
        // detected from the filename / config (already in .cpp tool).
        const ThermophysicalProperties body =
            PlanetaryData::getProperties(PlanetaryBody::MOON);
        m.body_defaults_ = body;
        m.body_defaults_set_ = true;
        m.invertTIToH(body);
    } else {
        m.body_defaults_ = PlanetaryData::getProperties(PlanetaryBody::MOON);
        m.body_defaults_set_ = true;
    }

    // Sanity-check the H values: clamp insane values to defaults.
    int n_clamped = 0;
    for (std::size_t f = 0; f < m.H_.size(); ++f) {
        if (!std::isfinite(m.H_[f]) || m.H_[f] <= 0.0 || m.H_[f] > 5.0) {
            m.H_[f] = m.body_defaults_.H;
            m.mask_[f] = 0;
            ++n_clamped;
        }
    }
    if (n_clamped > 0) {
        std::cerr << "ThermalPropertyMap: " << n_clamped
                  << " face(s) had non-finite/out-of-range H; clamped to body default\n";
    }

    return m;
}

double ThermalPropertyMap::fraction_default() const {
    if (mask_.empty()) return 0.0;
    std::size_t n_default = 0;
    for (auto v : mask_) if (v == 0) ++n_default;
    return static_cast<double>(n_default) / static_cast<double>(mask_.size());
}

}  // namespace thermal
