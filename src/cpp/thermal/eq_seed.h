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
// ALSO NOT HERE: the Kirchhoff (fourth-moment) depth profile. The exact 1-D
// periodic steady state for k(z,T) = k_c(z)*g(T) integrates the POTENTIAL
// G(T) = INT_0^T g(s) ds, not T, so the surface constant is
// <G(T_0)> = <T_0> + chi*<T_0^4>/(4*350^3) and the extra term over today's
// profile is R4*W(0) with W(0) = <T_0^4> - T_anchor^4. While the anchor is the
// warm bracket T_w that term is IDENTICALLY ZERO by construction, because T_w
// is DEFINED by eps*sigma*T_w^4 = <F>+Qb = eps*sigma*<T_0^4> -- measured, arm
// A' minus arm A is max 9.16e-4 K over the whole seed field and the cycle
// counts are bit-for-bit the same 40/56/60. It becomes non-zero the moment the
// anchor stops being T_w, and is then worth a further 20 -> 3 cycles on lit
// faces. Separable, and deliberately NOT bundled with the anchor below.
// (kirchprofile_2026-08-12/e2_cycles.log, e5_universal.log.)
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
#include "../constants/stefan_boltzmann.h"   // ptm::kStefanBoltzmann

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
// WHICH SURFACE ANCHOR the conductive profile hangs from.
//
// THE PROBLEM. Steady state pins the FOURTH moment exactly,
// <eps*sigma*T^4> = <F> + Qb, so T_w = ((<F>+Qb)/(eps*sigma))^(1/4) is the
// exact <T^4>^(1/4). But the conductive profile is anchored at the CYCLE MEAN
// <T>, and by Jensen (<T^4>)^(1/4) >= <T> with equality only where T(t) is
// flat. On the shipped lunar 80N crater that is +45.06 K mean / 83.15 K max
// against a converged reference. Dark faces are flat, so T_w is already exact
// there and must STAY exact.
//
// THE FIX (derived, no fitted constant). Replace the half-space under the
// surface by a lumped substrate at T_d coupled through a real conductance G:
//
//     eps*sigma*T(t)^4 + G*T(t) = F(t) + Qb + G*T_d ,      T_d = <T(t)> .
//
// The samples decouple -- there is no time stepping and no history -- so this
// is a scalar fixed point on T_d solved by Newton. Its two limits are exact
// boundary conditions, not approximations:
//     G -> 0    =>  T_d = < ((F+Qb)/(eps*sigma))^(1/4) >   (the cold bracket)
//     G -> inf  =>  T_d = T_w                              (the warm bracket)
// and a flat F makes the two coincide, so a zero-swing face is exact.
//
// G IS NOT FITTED. It is fixed by a describing-function variance match: choose
// the single G whose lumped response reproduces the LINEARIZED total variance
// of the true half-space response to THIS face's own forcing spectrum.
//
//     h    = 4*eps*sigma*T_d^3                          (radiative conductance)
//     P_n  = one-sided power of F at bin n >= 1, from this face's own rFFT
//     |T_n|_halfspace = |F_n| / |h + Gamma_eff*sqrt(n*omega)*e^(i*pi/4)|
//     |T_n|_lumped    = |F_n| / (h + G)
//     (h+G)^2 = SUM_n P_n / SUM_n [ P_n / (h^2 + sqrt2*h*A_n + A_n^2) ],
//                                              A_n = Gamma_eff*sqrt(n*omega)
//     G = max(0, sqrt(...) - h)
//
// with Gamma_eff = |Y(omega)|/sqrt(omega) the equivalent thermal inertia of the
// LAYERED column, from the transmission-line admittance recursion below (it
// degenerates to sqrt(k*rho*cp) on a homogeneous column). Gamma_eff and h both
// depend on T_d, so the closure is iterated to a fixed point -- 4 passes.
//
// WHERE IT DOES NOT HOLD, it falls back to T_w and SAYS SO (EqSeedAnchorStatus):
// two-timescale forcing (diurnal + annual -- one omega is the wrong object),
// radiation exchange ON (F depends on the neighbors, so the anchor has to be
// re-evaluated inside the Picard loop), latent heat, and faces with too few lit
// samples to have a resolved forcing shape at all.
// ---------------------------------------------------------------------------
enum class EqSeedAnchor {
    Lumped,  ///< derived lumped-substrate anchor T_d (default)
    Warm     ///< T_w only -- the pre-2026-08-12 anchor, bit-for-bit
};

/// Parse "lumped" | "warm". Returns false (leaving `out` untouched) on an
/// unrecognized name so the caller fails loudly.
bool parseEqSeedAnchor(const std::string& s, EqSeedAnchor& out);

inline const char* eqSeedAnchorName(EqSeedAnchor a) {
    return (a == EqSeedAnchor::Warm) ? "warm" : "lumped";
}

/// What the anchor actually did for THIS face. Every value other than `Lumped`
/// and `ZeroSwing` means the seed fell back to T_w, and the reason is reported
/// rather than swallowed.
enum class EqSeedAnchorStatus {
    WarmRequested = 0,        ///< thermal.seed_anchor: warm
    NoForcingSeries,          ///< caller supplied no series (e.g. point mode)
    Lumped,                   ///< the derived anchor ran
    ZeroSwing,                ///< F(t) exactly flat: T_d == T_w, T_w kept bitwise
    LumpedInputLimited,       ///< the anchor ran, but on < 4 lit samples
    FallbackTwoTimescale,     ///< power below the diurnal fundamental
    FallbackRadiationCoupled, ///< F depends on neighbors this iteration
    FallbackLatentHeat,       ///< a phase change is in the surface budget
    FallbackNonUniformSamples,///< the caller's samples are not evenly spaced
    FallbackFewLitSamples,    ///< < min_lit_samples samples with F > 0
    FallbackShortSpan,        ///< span < ~1 cycle, or span_s <= 0
    FallbackBadProperties,    ///< cp <= 0, k <= 0, non-finite Gamma_eff
    FallbackNotConverged      ///< Newton did not reach tolerance
};

const char* eqSeedAnchorStatusName(EqSeedAnchorStatus s);

/// True when the status means "the derived anchor was used".
inline bool eqSeedAnchorRan(EqSeedAnchorStatus s) {
    return s == EqSeedAnchorStatus::Lumped ||
           s == EqSeedAnchorStatus::LumpedInputLimited;
}

// ---------------------------------------------------------------------------
// The per-face forcing series the anchor needs. Optional: without it the seed
// behaves exactly as it did before this file grew an anchor.
//
// QUADRATURE DISCLOSURE. `n_samples` is the ONLY information the anchor has
// about F(t). Production equilibration samples at 48/cycle, and on a
// shadow-edge face that carries several kelvin of IRREDUCIBLE input error --
// the anchor cannot know about a shadow entry that falls between two samples.
// EqSeedResult::n_samples is therefore reported, and the run banner must state
// it: an accuracy claim without its quadrature is meaningless on those faces.
// ---------------------------------------------------------------------------
struct EqSeedForcing {
    /// Absorbed flux samples [W/m^2], uniformly spaced in time, sample i at
    /// flux[i*stride]. The SAME composition the solver's own forcing uses.
    const double* flux   = nullptr;
    std::size_t   stride = 1;
    int           n_samples = 0;
    /// Total duration the samples cover [s] = n_samples * dt. The rFFT treats
    /// the series as periodic with this period, so bin n is the physical
    /// angular frequency n*2*pi/span_s.
    double        span_s  = 0.0;
    /// The body's diurnal period [s]. Used ONLY to detect the two-timescale
    /// regime and a sub-cycle span; 0 disables both of those gates.
    double        cycle_s = 0.0;

    /// cp(T) = cp0 + sum_j cp_coeff[j] * T^(j+1)  -- planet.cp0 / planet.cpCoeff.
    /// Needed because Gamma_eff = sqrt(k*rho*cp) in the homogeneous limit; the
    /// seed's conductive profile never needed cp, the admittance does.
    double        cp0        = 0.0;
    const double* cp_coeff   = nullptr;
    int           n_cp_coeff = 0;

    /// Set by the caller when the regime invalidates the closure.
    bool radiation_coupled = false;  ///< F depends on neighbor temperatures
    bool latent_heat       = false;  ///< a phase change is in the surface budget
    /// The rFFT assumes evenly spaced samples. A caller whose time base is not
    /// uniform (an event-driven or resampled series) must say so; the anchor
    /// then stands down instead of computing a spectrum of the wrong signal.
    bool nonuniform_samples = false;

    /// Fraction of the forcing variance allowed to sit BELOW the diurnal
    /// fundamental before the face is declared two-timescale. Reported either
    /// way (EqSeedResult::subdiurnal_variance_fraction) so the choice is
    /// auditable rather than invisible.
    double max_subdiurnal_variance_fraction = 0.05;
    /// Fewest samples with F > 0 before a face is REFUSED the anchor outright.
    /// DEFAULT 0 = never refuse, because refusing means substituting T_w, and
    /// T_w is +28 K on exactly this population: measured on the 24 grazing
    /// faces of the 80N crater that carry 1-3 lit samples, refusing moves the
    /// grazing p95 from 3.6 K to 26.9 K. The condition is real -- with 1-3 lit
    /// samples the forcing shape is quadrature noise and the anchor's accuracy
    /// claim does not hold -- so it is DISCLOSED (status LumpedInputLimited)
    /// rather than acted on. Set to 4 to restore the refusal.
    int    min_lit_samples = 0;
    /// Lit-sample count below which the face is flagged input-limited. Report
    /// only; the anchor still runs.
    int    input_limited_lit_samples = 4;
};

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
    /// The seed MUST use the same sigma as the surface BC on its own path, or
    /// it anchors the column somewhere the BC will not hold it. Since
    /// fix/surface-bc-unification there is exactly one value engine-wide, so
    /// this default and every caller's explicit value are the same number.
    double sigma              = ptm::kStefanBoltzmann;

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

    /// Which anchor to use. Only consulted when `forcing` is supplied.
    EqSeedAnchor anchor = EqSeedAnchor::Lumped;
    /// THIS face's forcing series. Null => the anchor is T_w, as before.
    const EqSeedForcing* forcing = nullptr;
};

struct EqSeedResult {
    double T_surface = 0.0;   ///< T(z=0)
    double T_deep    = 0.0;   ///< T(z=z_max)
    bool   clamped_low  = false;   ///< hit the ABSOLUTE_ZERO_K guard
    bool   above_500K   = false;   ///< physical, but outside the CPU solver clamp

    // --- anchor diagnostics (all report-only; the solver reads T_surface) ---
    EqSeedAnchorStatus anchor_status = EqSeedAnchorStatus::NoForcingSeries;
    /// The anchor BEFORE the ABSOLUTE_ZERO_K guard that T_surface carries.
    /// Reported separately so a study can score the anchor itself rather than
    /// the seed's floor -- they differ only where the physics is below 2.7 K.
    double T_anchor = 0.0;
    double T_warm = 0.0;      ///< ((<F>+Qb)/(eps*sigma))^0.25   -- infinite-inertia
    double T_cold = 0.0;      ///< <((F+Qb)/(eps*sigma))^0.25>   -- zero-inertia
    double G      = 0.0;      ///< the matched lumped conductance [W/m^2/K]
    double gamma_eff = 0.0;   ///< |Y(omega)|/sqrt(omega)        [J/m^2/K/s^0.5]
    double subdiurnal_variance_fraction = 0.0;
    int    n_samples     = 0; ///< QUADRATURE DISCLOSURE -- report this
    int    n_lit_samples = 0;
    int    anchor_iterations = 0;
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
// Anchor internals, exposed because the two limits are UNIT TESTS, not
// statistics: they are boundary conditions, so they hold or the code is wrong.
// ---------------------------------------------------------------------------

/// Solve   eps*sigma*T(t)^4 + G*T(t) = F(t) + Qb + G*T_d ,  T_d = <T(t)>
/// for the scalar T_d, with G GIVEN. Newton on T_d; the per-sample quartic is
/// solved for the DEVIATION u = T - T_d so the G -> inf limit stays
/// well-conditioned (both <u> and d<u>/dT_d vanish at the same rate).
///
///   G = 0    -> <((F+Qb)/(eps*sigma))^0.25>  exactly (one Newton step)
///   G -> inf -> ((<F>+Qb)/(eps*sigma))^0.25  exactly
///
/// Returns < 0 on a malformed input (n <= 0, eps*sigma <= 0, non-finite F).
/// `iters_out` (optional) receives the outer Newton count; a negative count
/// means the tolerance was not reached.
double eqSeedLumpedAnchor(const double* F, std::size_t stride, int n,
                          double Qb, double eps, double sigma, double G,
                          double* T_warm_out, double* T_cold_out,
                          int* iters_out);

/// One-sided power spectrum P[0..n/2] of the real series x, normalized so that
/// sum_{k>=1} P[k] is the sample variance of x (P[0] is zeroed by the caller's
/// convention here: the DC bin carries no information the anchor can use).
/// Arbitrary n: radix-2 when n is a power of two, Bluestein otherwise.
void eqSeedOneSidedPower(const double* x, std::size_t stride, int n,
                         std::vector<double>& P_out);

/// Equivalent thermal inertia Gamma_eff = |Y(omega)|/sqrt(omega) of the LAYERED
/// column (z, kc, rho, n from `in`), linearized about T_ref, from the standard
/// transmission-line recursion
///     Y_j = Yl_j * (Y_{j+1} + Yl_j*tanh(g_j d_j)) / (Yl_j + Y_{j+1}*tanh(g_j d_j))
/// with Yl_j = k_j*g_j, g_j = sqrt(i*omega*rho_j*cp/k_j), terminated at the
/// bottom with the local semi-infinite value. Reduces to sqrt(k*rho*cp) on a
/// homogeneous column. Returns < 0 if cp <= 0 or every k <= 0.
double eqSeedGammaEff(const EqSeedInput& in, const EqSeedForcing& fc,
                      double omega, double T_ref);

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
        double T_warm = 0.0, T_cold = 0.0, G = 0.0, gamma_eff = 0.0;
        int    n_samples = 0;
        EqSeedAnchorStatus anchor_status = EqSeedAnchorStatus::NoForcingSeries;
        bool   set = false;
    };
    bool             enabled_ = false;
    std::string      path_;
    mutable std::vector<Row> rows_;
};

}  // namespace thermal

#endif  // THERMAL_EQ_SEED_H
