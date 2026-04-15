#include "photometric_model.h"
#include <algorithm>
#include <cmath>

namespace ptm {

// Physical constants
static constexpr double STEFAN_BOLTZMANN = 5.670374419e-8;  // W/m^2/K^4
static constexpr double PLANCK_H = 6.62607015e-34;          // J·s
static constexpr double SPEED_C = 2.99792458e8;             // m/s
static constexpr double BOLTZMANN_K = 1.380649e-23;         // J/K

PhotometricModel::PhotometricModel()
    : type_(ReflectanceType::LOMMEL_SEELIGER), params_() {}

PhotometricModel::PhotometricModel(ReflectanceType type, const Params& params)
    : type_(type), params_(params) {}

std::string PhotometricModel::reflectanceTypeName(ReflectanceType t) {
    switch (t) {
        case ReflectanceType::LAMBERTIAN: return "Lambertian";
        case ReflectanceType::LOMMEL_SEELIGER: return "Lommel-Seeliger";
        case ReflectanceType::MINNAERT: return "Minnaert";
        case ReflectanceType::HAPKE: return "Hapke";
    }
    return "Unknown";
}

// ============================================================================
// Reflected radiance
// ============================================================================

double PhotometricModel::reflectedRadiance(double cos_i, double cos_e,
                                            double alpha_rad,
                                            double solar_flux) const {
    if (cos_i <= 0.0 || cos_e <= 0.0) return 0.0;

    double r = 0.0;
    switch (type_) {
        case ReflectanceType::LAMBERTIAN:
            r = lambertian(cos_i, cos_e);
            break;
        case ReflectanceType::LOMMEL_SEELIGER:
            r = lommelSeeliger(cos_i, cos_e);
            break;
        case ReflectanceType::MINNAERT:
            r = minnaert(cos_i, cos_e);
            break;
        case ReflectanceType::HAPKE:
            r = hapke(cos_i, cos_e, alpha_rad);
            break;
    }

    // Radiance = reflectance * cos(i) * solar_flux
    // For Lambertian: L = (A/pi) * cos(i) * F_sun
    // For others: L = r * cos(i) * F_sun (some already include cos(i) dependence)
    return r * cos_i * solar_flux;
}

double PhotometricModel::reflectedIF(double cos_i, double cos_e,
                                      double alpha_rad) const {
    if (cos_i <= 0.0 || cos_e <= 0.0) return 0.0;

    double r = 0.0;
    switch (type_) {
        case ReflectanceType::LAMBERTIAN:
            r = lambertian(cos_i, cos_e);
            break;
        case ReflectanceType::LOMMEL_SEELIGER:
            r = lommelSeeliger(cos_i, cos_e);
            break;
        case ReflectanceType::MINNAERT:
            r = minnaert(cos_i, cos_e);
            break;
        case ReflectanceType::HAPKE:
            r = hapke(cos_i, cos_e, alpha_rad);
            break;
    }

    // I/F = pi * r * cos(i)  (normalized to Lambertian disk)
    return M_PI * r * cos_i;
}

// ============================================================================
// Individual reflectance models
// ============================================================================

double PhotometricModel::lambertian(double /*cos_i*/, double /*cos_e*/) const {
    // Bidirectional reflectance of a Lambertian surface: r = A / pi
    return params_.albedo / M_PI;
}

double PhotometricModel::lommelSeeliger(double cos_i, double cos_e) const {
    // r = (w / 4pi) * 1 / (cos_i + cos_e)
    // Models single scattering from a semi-infinite particulate medium
    double denom = cos_i + cos_e;
    if (denom < 1e-10) return 0.0;
    return (params_.ls_w / (4.0 * M_PI)) / denom;
}

double PhotometricModel::minnaert(double cos_i, double cos_e) const {
    // r = (A / pi) * cos_i^(k-1) * cos_e^(k-1)
    // k=1 is Lambertian; k=0.5 is typical for the Moon
    double k = params_.minnaert_k;
    return (params_.albedo / M_PI) * std::pow(cos_i, k - 1.0) * std::pow(cos_e, k - 1.0);
}

double PhotometricModel::hapke(double cos_i, double cos_e, double alpha_rad) const {
    // Hapke (1993) simplified model:
    // r = (w / 4pi) * (1 / (cos_i + cos_e)) *
    //     [(1 + B(alpha)) * P(alpha) + H(cos_i)*H(cos_e) - 1]
    double w = params_.hapke_w;
    double denom = cos_i + cos_e;
    if (denom < 1e-10) return 0.0;

    double cos_alpha = std::cos(alpha_rad);
    double P = henyeyGreenstein(cos_alpha);
    double B = oppositionSurge(alpha_rad);
    double Hi = chandrasekharH(cos_i, w);
    double He = chandrasekharH(cos_e, w);

    double bracket = (1.0 + B) * P + Hi * He - 1.0;
    return (w / (4.0 * M_PI)) * bracket / denom;
}

double PhotometricModel::henyeyGreenstein(double cos_alpha) const {
    // P(alpha) = (1 - g^2) / (1 + 2g*cos(alpha) + g^2)^(3/2)
    double g = params_.hapke_g;
    double g2 = g * g;
    double denom = 1.0 + 2.0 * g * cos_alpha + g2;
    if (denom <= 0.0) return 1.0;
    return (1.0 - g2) / std::pow(denom, 1.5);
}

double PhotometricModel::oppositionSurge(double alpha_rad) const {
    // B(alpha) = B0 / (1 + tan(alpha/2) / h)
    double B0 = params_.hapke_B0;
    double h = params_.hapke_h;
    if (h <= 0.0) return 0.0;

    double half_alpha = alpha_rad / 2.0;
    if (half_alpha >= M_PI / 2.0 - 0.01) return 0.0;  // avoid tan singularity
    double tan_half = std::tan(half_alpha);
    return B0 / (1.0 + tan_half / h);
}

double PhotometricModel::chandrasekharH(double x, double w) const {
    // Approximate H-function: H(x) = (1 + 2x) / (1 + 2x * sqrt(1-w))
    double gamma = std::sqrt(std::max(0.0, 1.0 - w));
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma);
}

// ============================================================================
// Thermal emission
// ============================================================================

double PhotometricModel::emissionPhase(double cos_e) const {
    switch (params_.emission_phase) {
        case EmissionPhaseType::LAMBERTIAN:
            return std::max(0.0, cos_e);
        case EmissionPhaseType::FLAT:
            return (cos_e > 0.0) ? 1.0 : 0.0;
        case EmissionPhaseType::POLYNOMIAL: {
            if (cos_e <= 0.0) return 0.0;
            double result = 0.0;
            double cos_power = 1.0;
            for (double coeff : params_.emission_poly_coeffs) {
                result += coeff * cos_power;
                cos_power *= cos_e;
            }
            return std::max(0.0, result);
        }
    }
    return std::max(0.0, cos_e);
}

double PhotometricModel::planckFunction(double lambda_m, double T) {
    // B(lambda, T) = (2hc^2 / lambda^5) / (exp(hc / (lambda * k_B * T)) - 1)
    // Returns W/m^2/sr/m
    if (T <= 0.0 || lambda_m <= 0.0) return 0.0;
    double c1 = 2.0 * PLANCK_H * SPEED_C * SPEED_C;  // 2hc^2
    double c2 = PLANCK_H * SPEED_C / BOLTZMANN_K;     // hc/k_B
    double exponent = c2 / (lambda_m * T);
    if (exponent > 500.0) return 0.0;  // avoid overflow
    double lambda5 = lambda_m * lambda_m * lambda_m * lambda_m * lambda_m;
    return c1 / (lambda5 * (std::exp(exponent) - 1.0));
}

double PhotometricModel::thermalSpectralRadiance(double T_kelvin, double cos_e,
                                                  double lambda_um) const {
    // Returns W/m^2/sr/um
    if (T_kelvin <= 0.0 || cos_e <= 0.0) return 0.0;
    double lambda_m = lambda_um * 1e-6;
    double B = planckFunction(lambda_m, T_kelvin);
    // Convert from W/m^2/sr/m to W/m^2/sr/um
    B *= 1e-6;
    return params_.emissivity * B * emissionPhase(cos_e);
}

double PhotometricModel::thermalBroadbandRadiance(double T_kelvin, double cos_e) const {
    // L = epsilon * sigma * T^4 / pi (Lambertian thermal emission per steradian)
    // Modified by emission phase function
    if (T_kelvin <= 0.0 || cos_e <= 0.0) return 0.0;
    double T4 = T_kelvin * T_kelvin * T_kelvin * T_kelvin;
    double L = params_.emissivity * STEFAN_BOLTZMANN * T4 / M_PI;
    return L * emissionPhase(cos_e);
}

} // namespace ptm
