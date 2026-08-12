#ifndef THERMAL_EQ_SEED_H
#define THERMAL_EQ_SEED_H

// ONE spin-up seed, used by every solver path.
//
// WHAT WENT WRONG. Before this file the engine shipped four different initial
// conditions and a CPU-only pre-conditioner, and no test compared any two of
// them. Measured on tutorial04_crater80N_radoff (1,402 permanently shadowed
// faces, identical illumination on every arm, geothermal floor T_geo =
// 24.043 K), the mean temperature of the dark population after the run was:
//
//     CPU scalar / streamed          69.150 K   Profile::init_T, one global
//                                               T_avg = T_ss/sqrt(2): flux-blind
//                                               AND latitude-blind
//     CPU SIMD batch (the default)   66.246 K   per-face radiative balance
//                                               (PTM-46) -- correct -- then
//                                               destroyed by the pre-conditioner
//     GPU persistent (default)       71.028 K   10-step mean flux, then
//                                               `if (T_eq < 40) T_eq = 100`
//     GPU + PTM_EQ_PERFACE_FLUX=1    24.281 K   per-face mean flux + Qb
//
// 46.7 K of spread on a column whose physics fixes it at 24.043 K. Two of the
// three shipped defaults sit 42-47 K above the floor; only the opt-in per-face
// seed lands on it. That is the largest single divergence in the model, and it
// lands on PSR / cold-trap / volatile-stability science.
//
// THE PHYSICS THIS FILE IMPLEMENTS. At the cycle mean the column is in steady
// state, so the conducted flux is constant and equal to the basal flux Qb.
// Then
//
//     dT/dz = Qb / k(T, z)        =>   T(z) = T_0 + Qb * integral_0^z dz'/k
//
// anchored at the surface by radiative balance against THAT FACE'S OWN mean
// absorbed flux,
//
//     eps*sigma*T_0^4 = <F_abs> + Qb   =>   T_0 = ((<F_abs> + Qb)/(eps*sigma))^(1/4).
//
// The diurnal wave rides on top of this and has zero cycle mean, so the cycle
// mean IS the profile above. Including Qb in the surface balance makes T_0
// continuous as <F_abs> -> 0 and lands a never-lit face exactly on the
// geothermal floor T_geo = (Qb/(eps*sigma))^(1/4) that the solver itself
// enforces every step -- seed and floor agree instead of fighting.
//
// This is the same analytic profile the R1 study accepted as the deviation-form
// reference T_ref (memory r1-deviation-form-arrest-gate-2026-08-10): T_0 =
// (Qb/eps*sigma)^(1/4), T_{i+1} = T_i + Qb*dz_i/k_i. Generalizing T_0 to
// include <F_abs> is the only change, so the seed and the deviation reference
// are the same object and stay consistent by construction.
//
// WHAT IS DELIBERATELY NOT HERE: the damped diurnal wave
// exp(-z/delta)*cos(omega t - z/delta). Its amplitude is set by the FIRST
// HARMONIC of the per-face forcing, and it decays with the top-cell relaxation
// time, which is ~ one diurnal period by construction -- so omitting it costs
// at most ~1 cycle of spin-up, while the deep column (the actual bottleneck)
// costs hundreds. Adding it would also require getting the phase right per
// face. Measured attribution is in the session notes; if a future study shows
// the surface term binds, add it here so every path gets it at once.
//
// WHAT IS NOT A SEED AT ALL. Both CPU paths used to integrate
// NYEARSEQ*round(year/day) diurnal cycles (24 on the Moon) of ANALYTIC
// CLEAR-SKY forcing -- no topography, no shadowing, no IR, no indirect, body
// average albedo -- before the real forcing, and the GPU had no analog. An
// initial condition is free: a correct solver forgets it. A FORCING is not. It
// destroyed the correct per-face IC the same code path had just computed
// (Fermi measured 24.043 K -> 64.375 K surface, +60 K at 1 m on a face that
// never sees the Sun). It is OFF by default now, and reported when on.
// See analyticSeedCycles() below.

#include <cstddef>
#include <string>
#include <vector>

#include "conductivity_model.h"

namespace thermal {

// ---------------------------------------------------------------------------
// Which seed a run uses. ONE knob, so that "reproduce the old baseline" is one
// word rather than a set of per-path environment variables that can disagree.
// ---------------------------------------------------------------------------
enum class EqSeedMode {
    Physical,  ///< shared eqSeedColumn(); analytic pre-conditioner OFF (default)
    Legacy     ///< each path's pre-2026-08-11 seed; pre-conditioner ON as before
};

/// Parse "physical" | "legacy". Returns false (leaving `out` untouched) on an
/// unrecognized name so the caller fails loudly instead of silently picking a
/// seed the user did not ask for.
bool parseEqSeedMode(const std::string& s, EqSeedMode& out);

inline const char* eqSeedModeName(EqSeedMode m) {
    return (m == EqSeedMode::Legacy) ? "legacy" : "physical";
}

// ---------------------------------------------------------------------------
// The seed itself.
// ---------------------------------------------------------------------------
struct EqSeedInput {
    /// Time mean of THIS face's absorbed flux over the available forcing span
    /// [W/m^2]. Whatever the caller's own absorbed-flux composition produces --
    /// including the two-pass mean indirect flux, which is what makes a face
    /// warmed only by its neighbors seed above the floor.
    double mean_absorbed_flux = 0.0;
    double Qb                 = 0.0;    ///< THIS face's basal flux [W/m^2]
    double emissivity         = 0.95;
    double sigma              = 5.67051196e-8;

    const double* z   = nullptr;        ///< depth grid [m], n nodes, z[0] == 0
    const double* kc  = nullptr;        ///< conductive k_c(z) [W/m/K], n nodes
    const double* rho = nullptr;        ///< bulk density [kg/m^3]; Martinez only
    int           n   = 0;

    double            R350 = 0.0;       ///< chi/350^3; 0 disables the T^3 term
    ConductivityModel cond = ConductivityModel::Hayne2017;

    /// REPORT-ONLY diagnostic: the caller's zero-thermal-inertia estimate of
    /// the cycle-mean surface temperature,
    ///     <T>_0 = mean_t ( (F(t) + Qb) / (eps*sigma) )^0.25 ,
    /// over the SAME span as mean_absorbed_flux. It changes no solver state;
    /// it is dumped alongside the seed so the anchor question can be settled
    /// with numbers.
    ///
    /// WHY IT IS HERE. The conductive part of the seed describes the CYCLE
    /// MEAN, and at steady state the exact statement the energy balance makes
    /// is about the FOURTH moment: <eps*sigma*T^4> = <F> + Qb. By Jensen,
    ///     <T>  <=  (<T^4>)^0.25 = T_0 ,
    /// with equality only when T(t) is flat. Measured on the shipped lunar
    /// 80N crater against a 384-lunation reference, T_0 minus the converged
    /// cycle mean is +0.000 K (median) on the permanently shadowed population
    /// -- where T(t) IS flat, and where the seed is consequently exact -- but
    /// +36 K on lit faces (mean diurnal swing 105 K) and +69 K on grazing
    /// faces (swing 156 K). The two limits bracket the truth:
    ///     mean_inst_radeq_T  <=  <T>  <=  T_0
    /// (zero thermal inertia and infinite thermal inertia respectively).
    /// 0 = not supplied.
    double mean_inst_radeq_T = 0.0;
};

struct EqSeedResult {
    double T_surface = 0.0;   ///< T(z=0)
    double T_deep    = 0.0;   ///< T(z=z_max)
    bool   clamped_low  = false;   ///< hit the ABSOLUTE_ZERO_K guard
    bool   above_500K   = false;   ///< physical, but outside the CPU solver clamp
};

/// Fill T_out[0..n-1] with the cycle-mean conductive steady profile described
/// above. T_out may be null (then only the summary is returned).
///
/// Degenerate inputs are handled explicitly, never silently:
///   * non-finite or negative <F_abs> / Qb are treated as 0 and FLAGGED via the
///     return value so the caller can count and report them;
///   * Qb <= 0 gives an isothermal column at T_0, which is the correct steady
///     state for an insulated base (and matches the Dirichlet/Neumann split the
///     bottom BC already makes at Qb == 0);
///   * a non-positive k at any node stops the integration there and holds the
///     profile constant below, rather than producing an infinite gradient.
EqSeedResult eqSeedColumn(const EqSeedInput& in, double* T_out);

// ---------------------------------------------------------------------------
// Length of the CPU analytic clear-sky pre-conditioner, in diurnal cycles.
//
//   mode == Physical  -> 0 unless explicitly requested
//   mode == Legacy    -> nyearseq * round(year/day)   (Moon 24, Europa 2)
//
// `explicit_cycles` is the resolved thermal.analytic_seed_cycles: < 0 means
// "unset, follow the mode", >= 0 means "use exactly this many". A caller that
// gets a non-zero answer MUST print it (see the banner in ptm_thermal_sim).
// ---------------------------------------------------------------------------
double analyticSeedCycles(EqSeedMode mode, double explicit_cycles,
                          double planet_day_s, double planet_year_s,
                          int nyearseq);

// ---------------------------------------------------------------------------
// Cross-path seed recorder -- the instrument for gate G1 ("every path produces
// the same seed for the same face given the same inputs").
//
// Enabled by PTM_DUMP_SEED=<path.csv>. Every path records into the same slot
// array (one write per face, preallocated, so no locking), and the CSV is
// written once at the end of the run. Diffing two paths' CSVs is the gate.
// ---------------------------------------------------------------------------
class EqSeedRecorder {
public:
    static EqSeedRecorder& instance();

    bool enabled() const { return enabled_; }
    /// Size the slot array. Safe to call repeatedly; only grows.
    void reserve(std::size_t n_faces);
    void record(std::size_t face, const EqSeedInput& in, const EqSeedResult& r);
    /// Write the CSV. No-op when disabled or when nothing was recorded.
    void write(const std::string& tag) const;

private:
    EqSeedRecorder();
    struct Row {
        double mean_flux = 0.0, Qb = 0.0, T_surf = 0.0, T_deep = 0.0;
        double T_lowinertia = 0.0;
        bool   set = false;
    };
    bool             enabled_ = false;
    std::string      path_;
    mutable std::vector<Row> rows_;
};

}  // namespace thermal

#endif  // THERMAL_EQ_SEED_H
