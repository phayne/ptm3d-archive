/**
 * @file conductivity_model.h
 * @brief Selectable regolith thermal-conductivity laws k(T, rho).
 *
 * PTM3D's historical (and still DEFAULT) law is Hayne et al. (2017):
 *
 *     k(T, z) = kc(z) * [1 + chi * (T/350)^3]
 *             = kc(z) + kc(z)*chi/350^3 * T^3
 *
 * i.e. a temperature-INDEPENDENT solid-conduction term plus a T^3 radiative
 * term whose coefficient is tied to the solid term through chi. That is the
 * "Watson equation" form, k = A + B*T^3, and it is fitted to Apollo-fines
 * laboratory data measured only down to ~150 K. Below that it is an
 * extrapolation, which is exactly where PSR floors live.
 *
 * The optional alternative is Martinez & Siegler (2021), "A Global Thermal
 * Conductivity Model for Lunar Regolith at Low Temperatures", JGR Planets 126,
 * e2021JE006829, doi:10.1029/2021JE006829. It replaces the constant solid term
 * with the amorphous-endmember temperature dependence of Woods-Robinson et al.
 * (2019) and re-fits the density dependence of both terms:
 *
 *     Eq. 10:  k(T, rho) = (A1*rho - A2) * k_am(T)  +  (B1*rho - B2) * T^3
 *     Eq.  9:  k_am(T)   = Ac + Bc*T^-4 + Cc*T^-3 + Dc*T^-2 + Ec*T^-1
 *                             + Fc*T + Gc*T^2 + Hc*T^3 + Ic*T^4
 *
 * with k in W m^-1 K^-1, rho in kg m^-3, T in K.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS PORT DOES AND DOES NOT INCLUDE  (read before interpreting a run)
 * ---------------------------------------------------------------------------
 * INCLUDED: the conductivity law, Eqs. 9 + 10, verbatim with the paper's
 *           published coefficients.
 *
 * NOT INCLUDED (deliberately - each is a separate config knob here, and
 * changing them would confound the conductivity comparison):
 *   - The paper's retuned density scale height H = 5.6 cm (PTM3D keeps the
 *     configured H; the lunar default is 6.0 cm, Hayne et al. 2017). The paper
 *     retuned H *after* fixing A(rho)/B(rho), to recover the equatorial diurnal
 *     curve. A faithful reproduction of their Figures 4-11 therefore also needs
 *     `thermophysical.H: 0.056`.
 *   - The Feng et al. (2020) bolometric Bond albedo (their Eq. 4) and Feng's
 *     revised ks = 8.0e-4 / kd = 3.8e-3. PTM3D's Moon carries Hayne (2017)
 *     ks = 7.4e-4 / kd = 3.4e-3. Under Martinez, ks/kd become INERT for the
 *     conduction law anyway (see below), so only the albedo law differs.
 *   - Their cp(T): the paper uses Hayne et al. (2017) cp unchanged, which is
 *     what PTM3D already uses. Nothing to do.
 *
 * CONSEQUENCE, MEASURED, NOT ASSUMED: under Eq. 10 the conductivity depends on
 * the LOCAL DENSITY rho(z) only. `ks` and `kd` no longer enter k at all. At the
 * lunar surface node (rho = 1100 kg m^-3) Martinez gives 0.63-0.69x the Hayne
 * conductivity at ALL temperatures 150-400 K, not merely at low T (the two laws
 * cross near rho ~ 1300-1400). This is the paper's own behavior - see its
 * Figure 3, where the "New 1100" curve sits below "Hayne 1100" across 0-400 K -
 * and it means selecting this model changes DAYTIME thermal inertia too.
 *
 * ---------------------------------------------------------------------------
 * VALIDITY (from the paper, plus limits derived here)
 * ---------------------------------------------------------------------------
 *   - k_am fit accuracy vs amorphous SiO2 (Touloukian & Ho 1970): +-3% over
 *     200-500 K, +-8% at 50 K and 900 K, +-15% below 10 K.
 *   - Lab data underpinning the density fit: 15-205 K (Zhong et al. 2016,
 *     NU-LHT-2M), rescaled across 1100-1800 kg m^-3.
 *   - DERIVED: (A1*rho - A2) changes sign at rho = 1003.52 kg m^-3 and
 *     (B1*rho - B2) at rho = 975.43 kg m^-3, so Eq. 10 returns a NEGATIVE
 *     conductivity for rho below ~1004 kg m^-3. Lunar rho_s = 1100 is above
 *     this, but nothing in the equation prevents a config from going under it.
 *     `martinezDensityIsValid()` exists so callers can refuse such a config
 *     loudly instead of silently integrating a negative diffusivity.
 *   - DERIVED: k_am(T) -> -inf as T -> 0 (the Bc*T^-4 term, Bc < 0). It is
 *     positive for T > 0.99 K; k_am(2.7 K) = 0.0605. PTM3D clamps every node at
 *     ABSOLUTE_ZERO_K = 2.7 K, so the polynomial is never evaluated in its
 *     divergent region on a well-formed run. `kMartinezTMin` re-imposes that
 *     same 2.7 K floor locally so the law is safe even if a caller hands it a
 *     colder T.
 *   - DERIVED: k_am(T) is NON-MONOTONIC between ~6.5 K (local max 0.1297) and
 *     ~11.9 K (local min 0.1240) - a 4.4% dip, inside the paper's own +-15%
 *     error band below 10 K. Left exactly as published: k stays positive, so
 *     the diffusion operator remains well posed, and "fixing" it would stop
 *     being the paper's model.
 *
 * The OpenCL device mirror of these functions lives in
 * `src/cpp/gpu/thermal_opencl.cpp::getEmbeddedKernelSource()` and MUST be kept
 * in step with this header; `src/cpp/tools/test_conductivity_model.cpp` checks
 * both against the paper's published values.
 */

#ifndef PTM_THERMAL_CONDUCTIVITY_MODEL_H
#define PTM_THERMAL_CONDUCTIVITY_MODEL_H

#include <cctype>
#include <string>

namespace thermal {

/// Selectable conductivity laws. Hayne2017 is the default and must stay the
/// default: every existing result in the repository was produced with it.
enum class ConductivityModel : int {
    Hayne2017    = 0,  ///< k = kc(z) * (1 + chi*(T/350)^3)   [PTM3D default]
    Martinez2021 = 1   ///< k = (A1*rho-A2)*k_am(T) + (B1*rho-B2)*T^3
};

// --- Martinez & Siegler (2021) Eq. 9 coefficients (amorphous endmember) ------
constexpr double kMartinezAc = -0.203297;
constexpr double kMartinezBc = -11.472;
constexpr double kMartinezCc =  22.5793;
constexpr double kMartinezDc = -14.3084;
constexpr double kMartinezEc =   3.41742;
constexpr double kMartinezFc =   0.01101;
constexpr double kMartinezGc =  -2.8049e-5;
constexpr double kMartinezHc =   3.35837e-8;
constexpr double kMartinezIc =  -1.40021e-11;

// --- Martinez & Siegler (2021) Eq. 10 density coefficients ------------------
constexpr double kMartinezA1 = 5.0821e-6;
constexpr double kMartinezA2 = 5.1e-3;
constexpr double kMartinezB1 = 2.0022e-13;
constexpr double kMartinezB2 = 1.953e-10;

/// Lowest temperature at which Eq. 9 is evaluated. Equal to PTM3D's own
/// physical temperature floor (ABSOLUTE_ZERO_K, the CMB), below which the
/// solver does not let a node go anyway. Not a fudge: a documented domain
/// restriction on a polynomial that diverges at T -> 0.
constexpr double kMartinezTMin = 2.7;

/// Densities below this make Eq. 10's solid coefficient (A1*rho - A2) negative.
constexpr double kMartinezRhoMin = kMartinezA2 / kMartinezA1;  // 1003.52 kg/m^3

/// True when Eq. 10 yields a positive conductivity for this density, i.e. when
/// BOTH fitted coefficients are positive. Callers should refuse (or loudly
/// warn about) a configuration that fails this rather than integrate a
/// negative diffusivity.
inline bool martinezDensityIsValid(double rho) {
    return (kMartinezA1 * rho - kMartinezA2) > 0.0 &&
           (kMartinezB1 * rho - kMartinezB2) > 0.0;
}

/// Eq. 9 evaluated WITHOUT the domain clamp. Callers must pass t >= kMartinezTMin.
/// Split out so `martinezK` does not re-clamp a value it has already clamped --
/// this runs per node, per substep, per timestep, on both CPU and GPU.
inline double martinezKamAtOrAbove(double t) {
    const double it = 1.0 / t;
    const double it2 = it * it;
    const double it3 = it2 * it;
    const double it4 = it2 * it2;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t2 * t2;
    return kMartinezAc
         + kMartinezBc * it4
         + kMartinezCc * it3
         + kMartinezDc * it2
         + kMartinezEc * it
         + kMartinezFc * t
         + kMartinezGc * t2
         + kMartinezHc * t3
         + kMartinezIc * t4;
}

/// Martinez & Siegler (2021) Eq. 9: amorphous-endmember solid conductivity
/// k_am(T) [W m^-1 K^-1], evaluated at max(T, kMartinezTMin).
inline double martinezKam(double T) {
    return martinezKamAtOrAbove((T < kMartinezTMin) ? kMartinezTMin : T);
}

/// Martinez & Siegler (2021) Eq. 10: effective conductivity [W m^-1 K^-1] at
/// temperature T [K] and local bulk density rho [kg m^-3].
inline double martinezK(double T, double rho) {
    const double t  = (T < kMartinezTMin) ? kMartinezTMin : T;
    const double a  = kMartinezA1 * rho - kMartinezA2;
    const double b  = kMartinezB1 * rho - kMartinezB2;
    return a * martinezKamAtOrAbove(t) + b * (t * t * t);
}

/// Hayne et al. (2017): k = kc * (1 + R350*T^3), R350 = chi/350^3.
/// Written out so both laws are visible side by side; bit-for-bit the same
/// expression the solvers have always used.
inline double hayneK(double T, double kc, double R350) {
    return kc * (1.0 + R350 * (T * T * T));
}

/// Reference dispatch form: both laws written as one expression.
///
/// The solvers do NOT call this. They branch on the model OUTSIDE their inner
/// loops, so that the Hayne path stays the original SIMD code, textually and
/// bit-for-bit. This exists as the single written-down statement of the two
/// laws side by side, and it is what `test_conductivity_model.cpp` pins against
/// the historical inline expression.
inline double conductivityK(ConductivityModel model,
                            double T, double kc, double rho, double R350) {
    return (model == ConductivityModel::Martinez2021) ? martinezK(T, rho)
                                                      : hayneK(T, kc, R350);
}

/// Parse the `thermophysical.conductivity_model` config string.
/// Returns false (leaving `out` untouched) on an unrecognized name so the
/// caller can fail loudly rather than silently falling back to the default -
/// a silent fallback here would misattribute a whole campaign.
inline bool parseConductivityModel(const std::string& s, ConductivityModel& out) {
    std::string v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '_' || c == '-' || c == ' ') continue;
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v.empty() || v == "hayne" || v == "hayne2017" || v == "default") {
        out = ConductivityModel::Hayne2017;
        return true;
    }
    if (v == "martinez" || v == "martinez2021" || v == "martinezsiegler2021") {
        out = ConductivityModel::Martinez2021;
        return true;
    }
    return false;
}

inline const char* conductivityModelName(ConductivityModel m) {
    return (m == ConductivityModel::Martinez2021) ? "martinez2021" : "hayne2017";
}

}  // namespace thermal

#endif  // PTM_THERMAL_CONDUCTIVITY_MODEL_H
