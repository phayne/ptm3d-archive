#pragma once

#include <cmath>
#include <vector>
#include <string>

namespace ptm {

class PhotometricModel {
public:
    enum class ReflectanceType { LAMBERTIAN, LOMMEL_SEELIGER, MINNAERT, HAPKE };
    enum class EmissionPhaseType { LAMBERTIAN, FLAT, POLYNOMIAL };

    struct Params {
        // Reflectance common
        double albedo = 0.12;

        // Lommel-Seeliger
        double ls_w = 0.25;  // single-scattering albedo

        // Minnaert
        double minnaert_k = 0.5;

        // Hapke (simplified 5-parameter)
        double hapke_w = 0.25;       // single-scattering albedo
        double hapke_B0 = 1.0;       // opposition surge amplitude
        double hapke_h = 0.05;       // opposition surge angular width
        double hapke_g = -0.3;       // Henyey-Greenstein asymmetry (negative = backscattering)
        double hapke_theta_bar = 20.0; // mean topographic slope angle (degrees)

        // Thermal emission
        double emissivity = 0.95;
        double thermal_wavelength_um = 10.0;  // for spectral mode

        // Emission phase function
        EmissionPhaseType emission_phase = EmissionPhaseType::LAMBERTIAN;
        std::vector<double> emission_poly_coeffs;  // for polynomial mode
    };

    PhotometricModel();
    PhotometricModel(ReflectanceType type, const Params& params);

    // Compute reflected radiance (W/m^2/sr) for a surface element
    // cos_i = cos(incidence angle), cos_e = cos(emission angle)
    // alpha_rad = phase angle in radians
    // solar_flux = solar irradiance at body's heliocentric distance (W/m^2)
    double reflectedRadiance(double cos_i, double cos_e, double alpha_rad,
                             double solar_flux) const;

    // Same, but with per-face albedo override (used when an albedo map is loaded)
    double reflectedRadiance(double cos_i, double cos_e, double alpha_rad,
                             double solar_flux, double face_albedo) const;

    // Compute I/F (radiance factor) — dimensionless
    double reflectedIF(double cos_i, double cos_e, double alpha_rad) const;

    // Compute thermal spectral radiance (W/m^2/sr/um) at wavelength lambda_um
    double thermalSpectralRadiance(double T_kelvin, double cos_e, double lambda_um) const;

    // Compute thermal broadband radiance (W/m^2/sr)
    double thermalBroadbandRadiance(double T_kelvin, double cos_e) const;

    ReflectanceType reflectanceType() const { return type_; }
    const Params& params() const { return params_; }

    static std::string reflectanceTypeName(ReflectanceType t);

private:
    ReflectanceType type_;
    Params params_;

    // Individual reflectance models — return bidirectional reflectance r (sr^-1)
    double lambertian(double cos_i, double cos_e) const;
    double lommelSeeliger(double cos_i, double cos_e) const;
    double minnaert(double cos_i, double cos_e) const;
    double hapke(double cos_i, double cos_e, double alpha_rad) const;

    // Hapke sub-functions
    double henyeyGreenstein(double cos_alpha) const;
    double oppositionSurge(double alpha_rad) const;
    double chandrasekharH(double x, double w) const;

    // Emission phase function evaluation
    double emissionPhase(double cos_e) const;

    // Planck function B(lambda, T) in W/m^2/sr/m
    static double planckFunction(double lambda_m, double T);
};

} // namespace ptm
