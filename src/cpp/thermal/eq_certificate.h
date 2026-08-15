#ifndef THERMAL_EQ_CERTIFICATE_H
#define THERMAL_EQ_CERTIFICATE_H

// Equilibration: predict before, certify after. REPORT-ONLY (stage R-1).
//
// PROSPECTIVE (eqEstimateSpinup) answers, at grid build and before any
// expensive work, "how many cycles does THIS problem need for the accuracy I
// asked for?" -- from tau1, the slowest eigenvalue of the actual discrete
// column, rather than from an inherited default.
//
// RETROSPECTIVE (eqCertifyColumn) answers, at the end of equilibration, "did it
// get there, and if not will more cycles help?" Both are in kelvin and both use
// the same tau1, so they can be compared at the end of a run.
//
// Neither writes any solver state. A run with these on must be BYTE-IDENTICAL
// to the same run with them off; that is the whole acceptance test.
//
// WHY EVERY NUMBER IS SPLIT BY POPULATION. At the shipped 50-lunation default
// the LIT faces are ~6 K from settled (median drift 50->100 lunations -6.12 K,
// max -19.5 K) while the DARK faces are frozen -- and the all-face MEDIAN drift
// is exactly 0.0 K. A mesh-wide aggregate hides the largest real error in the
// model behind a frozen majority. So: two populations, always, never one line.

#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <vector>
#include "../constants/stefan_boltzmann.h"  // ptm::kStefanBoltzmann (the single definition)

namespace thermal {

// DEFAULT ON; PTM_EQ_CERTIFICATE=0 opts out. Report-only, so the default
// is that every run reports what its spin-up bought.
bool eqCertificateEnabled();

// ---------------------------------------------------------------------------
// Shared machinery: the slowest time constant of the ACTUAL discrete column.
//
// Generalized symmetric tridiagonal eigenproblem G x = lambda C x, with G the
// interface conductances (k at the shallower node, matching the solver's own
// stencil), a Robin top (the linearized radiative leak h_rad = 4 eps sigma T^3)
// and a Neumann bottom. Solved by THREE inverse iterations on a Thomas solve --
// measured relative error 6.2e-6 at three iterations, verified to 6.2e-14
// against a dense eigensolve. tau1 = 1/lambda_min, in seconds.
//
// Returns <= 0 if the column is degenerate.
// ---------------------------------------------------------------------------
double eqColumnTau1(const double* z, const double* T, const double* k,
                    const double* rho, const double* cp, int n,
                    double eps, double sigma);

// ---------------------------------------------------------------------------
// PROSPECTIVE estimate for one column.
// ---------------------------------------------------------------------------
struct EqSpinupInput {
    const double* z = nullptr, *T = nullptr, *k = nullptr;
    const double* rho = nullptr, *cp = nullptr;
    int    n = 0;
    double Qb = 0.0, eps = 0.95, sigma = ptm::kStefanBoltzmann;
    double P  = 0.0;             // diurnal period [s]
    double target_K = 0.1;       // the user's requested accuracy
    // Arithmetic floor of the solver path this run will use. Below it no cycle
    // count helps, because the limit is arithmetic and not physics.
    double precision_floor_K = 0.0;
};

struct EqSpinupResult {
    double tau1        = 0.0;    // [s]
    double tau1_cycles = 0.0;    // tau1 / P
    double E0          = 0.0;    // seed's departure from its own conductive profile [K]
    double n_req       = 0.0;    // (tau1/P) * ln(E0/target)
    bool   unreachable = false;  // target below the path's arithmetic floor
    bool   valid       = false;
};

EqSpinupResult eqEstimateSpinup(const EqSpinupInput& in);

// ---------------------------------------------------------------------------
// Population-split accumulator + banner. `zero_forcing` separates the dark
// population from the forced one.
// ---------------------------------------------------------------------------
class EqSpinupCensus {
public:
    EqSpinupCensus(double P, double target_K, double precision_floor_K,
                   double seconds_per_cycle_estimate);
    void add(const EqSpinupResult& r, bool zero_forcing);
    // Prints the banner. `n_configured` is what the config actually asks for,
    // so the user can compare before spending the time. Returns false if
    // nothing was scored.
    bool report(std::ostream& os, double n_configured) const;
private:
    struct Bin {
        long long n = 0, unreachable = 0, invalid = 0;
        std::vector<double> n_req, tau1_cycles, E0;
    };
    Bin zero_, forced_;
    mutable std::mutex m_;   // scored once per face, from an OpenMP batch loop
    double P_, target_K_, floor_K_, sec_per_cycle_;
    static void line(std::ostream& os, const char* label, const Bin& b,
                     double sec_per_cycle);
};

// ===========================================================================
// RETROSPECTIVE certificate -- "did the column actually settle?"
// ===========================================================================
//
// THE QUANTITY, and why it is this one and not another.
//
//   r_c  = qbar_c - Qb              [W/m^2]   at a cut depth z_c
//   E    = |r_c| / h_obs            [K]
//
// qbar_c is the CYCLE-MEAN upward interface heat flux at z_c and Qb is the
// geothermal flux the column is driven by. At a settled state the two are
// equal at every depth below the forced layer, because nothing between the cut
// and the base stores or releases net heat over a cycle. r_c is therefore a
// CONSERVATION LAW ON A SPATIAL GRADIENT, and that is the whole point:
//
//   AN ARRESTED COLUMN CANNOT FOOL IT. When float32 rounding freezes a column,
//   T stops changing -- so "run until it stops moving", pass-to-pass bit
//   identity, and every increment-based test report SUCCESS at exactly the
//   moment they have failed. r_c is computed from the state's own spatial
//   gradient, not from any increment, so a frozen column keeps reporting its
//   frozen gradient deficit forever. Measured: the banked 8-pass moon3d state,
//   bit-frozen across 100 further lunations, still carries |r_c| worth ~0.27 K
//   at the surface -- nearly 3x the 0.1 K target it is asked to meet.
//
// h_obs CONVERTS W/m^2 TO KELVIN, AND WHICH ONE DEPENDS ON THE OBSERVABLE:
//   diurnal MAXIMUM:  h = 4 eps sigma T_max^3
//   diurnal MEAN:     h = [ <(4 eps sigma T0(t)^3)^-1> ]^-1, the HARMONIC
//                     cycle mean. Not the arithmetic mean: the surface
//                     displacement caused by a given flux residual is largest
//                     exactly where the radiative stiffness is smallest, so
//                     the two are anti-correlated and the arithmetic mean
//                     UNDERSTATES the error -- by up to 8.8x, in the
//                     non-conservative direction. When the caller cannot
//                     supply the harmonic mean it supplies the rigorous
//                     bracket 4 eps sigma T_min^3 <= h <= 4 eps sigma T_max^3
//                     instead, and the verdict comes back INDETERMINATE rather
//                     than being decided on a substitute.
//   DEEP products:    E_deep = |r_c| * tau1 / C_below.  CALIBRATED, and NOT
//                     conservative. On freely relaxing columns the POINTWISE
//                     ratio E_deep/dT_true has median 0.91-1.09 over six
//                     datasets -- i.e. UNBIASED, not high.
//                       An earlier version of this comment claimed 2.1-3.8x
//                     HIGH. That factor is real but it is not a calibration:
//                     it was measured on a population that was 100 % ARRESTED
//                     (the run's own result3.json records
//                     arrest_fraction_of_E_true = 1.0, delivered_total = 0 K),
//                     so it scored this estimator's IMMUNITY TO ARREST -- the
//                     feature described above -- and read it as conservatism.
//                     Trusting it and gating on E_deep at S = 1 would release
//                     70-77 % of faces early.
//                       Scored against the observable that actually matters --
//                     the MAX |dT| over nodes below the cut, not the
//                     C-weighted mean -- a safe gate needs S = 2.0 held for
//                     K = 5 CONSECUTIVE cycles (100 % safe on 12 of 12
//                     grid x cut x bar combinations). K = 1 is unsafe by
//                     17-23 K: at the clamped z_max/2 cut |r_c| passes through
//                     zero, and only the hysteresis filters that crossing.
//                     Measurement: project memory note
//                     edeep-recalibration-arrest-not-physics-2026-08-15.
//                     Still reported here, and still never used to PASS
//                     anything.
//
// CUT DEPTH z_c = min(5*delta_ref, z_max/2), delta_ref the skin depth of the
// LONGEST forcing period present. Five skin depths attenuates that period's
// wave by e^-5 = 0.7%; z_max/2 keeps the cut clear of the base. When the
// z_max/2 branch binds, the cut buys fewer than five skin depths and the
// result says so out loud (it binds on the shipped moon3d grid: 5*delta_annual
// = 0.552 m against z_max/2 = 0.502 m, i.e. 4.55 skin depths).
//
// WHAT THIS DOES NOT CATCH, stated so nobody assumes otherwise: it is a
// SELF-CONSISTENCY test. A column driven by the wrong Qb (BUGS.md PTM-161) or
// pinned on a clamp (PTM-159) is perfectly self-consistent and certifies.
// eqQbFromState() exists so the Qb actually in force is printed next to the
// configured one every time, which is the only defense against the first.

// Sign convention, stated once. z increases DOWNWARD, T normally increases
// with depth, and q below is the UPWARD heat flux, so a settled column has
// q == +Qb at every interface below the forced layer.

// Upward interface heat flux on the engine's own stencil:
//     q[j] = k[j] * (T[j+1] - T[j]) / (z[j+1] - z[j])
// with k at the SHALLOWER node -- which is what the Crank-Nicolson assembly
// uses (thermal_model.cpp: the node-i equation couples downward through
// g2[i-1]*k[i]). Writes n-1 values into q_out. False on a degenerate grid.
bool eqInterfaceFlux(const double* z, const double* T, const double* k, int n,
                     double* q_out);

// The geothermal flux THE STATE ITSELF CARRIES, read out of the engine's own
// bottom boundary condition
//     T[n-1] = T[n-2] + Qb * dz[n-2] / k[n-1]      (botTemp, thermal_model.cpp)
// Note k at the DEEPER node here -- the bottom BC and the interior stencil do
// not use the same one, and getting that backwards misreads Qb by the k ratio.
//
// This is measured, not configured, and that matters: `geothermal_heat_flow`
// was inert on every solver entry point for the whole moon3d campaign
// (BUGS.md PTM-161), so a certificate that trusted the config would have
// scored the residual against a Qb 22% off the one the state was integrated
// with. Returns <= 0 on a degenerate column.
double eqQbFromState(const double* z, const double* T, const double* k, int n);

// The skin depth THE DEPTH GRID ITSELF was built on, recovered from the grid:
// the engine sets z_max = b * delta_0 on both grid paths (thermal_model.cpp
// generateStableGrid / generateAdaptiveGrid), so delta_0 = z_max / b. Exact to
// the bottom layer's overshoot past z_max (1e-7 relative on the shipped moon3d
// grid). Scale to any other period with sqrt(P/P_diurnal).
//
// WHY NOT THE STATE'S OWN SURFACE kappa. It is a different number, by a lot:
// cp(T) collapses at low temperature, so kappa at a 24 K dark surface is ~8.8x
// the value at the 200 K representative temperature the grid was built from,
// and the implied skin depth is ~3x larger. A cut rule fed by the state would
// therefore land at a different depth on every face and would not be
// commensurate with the grid the column lives on. Which one is used is a
// choice, so it is the CALLER's -- eqCertifyColumn refuses a column with no
// delta_ref rather than quietly picking one.
double eqGridSkinDepth(const double* z, int n, int grid_b);

struct EqCut {
    int    j        = -1;     // interface index; the cut is between node j and j+1
    double z        = 0.0;    // z[j], the depth the cut actually landed on
    double z_target = 0.0;    // min(5*delta_ref, z_max/2) before snapping to a node
    double n_delta  = 0.0;    // z / delta_ref -- how many skin depths were bought
    bool   clamped  = false;  // the z_max/2 branch bound (fewer than 5 skin depths)
    bool   shallow  = false;  // fewer than 2 skin depths: the cut is inside the
                              // forced layer and the residual there is forcing,
                              // not equilibration error
};

// Choose the cut. Never returns the bottom-most interface (n-2): that one is
// the bottom-BC identity itself, q == Qb by construction, and reporting it
// would certify every column ever written.
EqCut eqChooseCut(const double* z, int n, double delta_ref);

enum class EqVerdict { NotScored = 0, Pass, Fail, Indeterminate };
const char* eqVerdictName(EqVerdict v);

struct EqCertifyInput {
    // The column. z/k/rho/cp as the engine built them; T the state under audit.
    const double *z = nullptr, *T = nullptr, *k = nullptr;
    const double *rho = nullptr, *cp = nullptr;
    int    n = 0;

    // Optional: a TRUE cycle-mean interface-flux profile (n-1 values), which
    // supersedes the one computed from T. The inline stopping condition
    // accumulates exactly this; the retrospective tool averages it over
    // however many banked states it was given.
    const double* qbar = nullptr;

    double Qb    = -1.0;                    // < 0 => eqQbFromState()
    double eps   = 0.95;
    double sigma = ptm::kStefanBoltzmann;

    double P_ref     = 0.0;   // longest forcing period present [s]
    double delta_ref = 0.0;   // its skin depth [m]. REQUIRED (> 0): see
                              // eqGridSkinDepth above for why there is no
                              // default. A column without one is not scored.
    double tau1      = -1.0;  // < 0 => eqColumnTau1()
    double target_K  = 0.1;

    // Observables. Each is scored only if its input is supplied; an absent
    // input yields NotScored, never a substituted one.
    double T_max      = 0.0;  // cycle-max SURFACE temperature [K]
    // Cycle-MIN surface temperature. The coldest instant is the STIFFEST-
    // observable-per-watt: h = 4 eps sigma T^3 is smallest there, so a given
    // flux residual displaces T_min more than it displaces either T_max or the
    // cycle mean. MEASURED on the lunar ico3 quickstart: stopping spin-up when
    // the cycle-mean observable reached 0.1 K left temperature_min 0.69 K from
    // its full-length answer while temperature_max moved 0.03 K -- a factor of
    // 24 between the two ends of the same cycle. An observable set without the
    // minimum in it certifies a run that is wrong where the PSR science is.
    double T_min      = 0.0;
    double h_harmonic = 0.0;  // harmonic cycle mean of 4 eps sigma T0^3
    double h_lower    = 0.0;  // rigorous lower bound on h (e.g. from T_min)
    double h_upper    = 0.0;  // rigorous upper bound on h (e.g. from T_max)

    // PROVENANCE OF qbar -- the honesty knobs. A snapshot is not a cycle mean.
    // Exactly one of these makes a verdict definitive:
    //   q_is_cycle_mean      the caller integrated qbar over whole cycles;
    //   no_periodic_forcing  the surface forcing does not vary, so there is no
    //                        periodic component for a snapshot to miss;
    //   q_phase_bound > 0    a MEASURED bound [W/m^2] on the un-averaged part.
    // With none of them the flux is bracketed [0, inf) and every verdict comes
    // back INDETERMINATE. That is the correct answer, not a failure of the
    // certificate: one snapshot of a forced column does not contain a cycle
    // mean, and pretending otherwise is how a 27%-of-seasonal-phase error
    // becomes a PASS.
    bool   q_is_cycle_mean     = false;
    bool   no_periodic_forcing = false;
    double q_phase_bound       = 0.0;

    // T came out of a 'PTMT' checkpoint, whose temperature block is float32 on
    // EVERY path including the float64 CPU solvers (thermal_state_io.h). The
    // gradient across the cut is then quantized, which puts a floor under
    // |r_c| that has nothing to do with equilibration. Set this and the result
    // carries that floor so a target below it is called out instead of being
    // silently "reached". Measured on the shipped lunar grid: ~1e-4 K.
    bool   T_is_float32 = false;
};

struct EqCertifyResult {
    bool   valid = false;
    EqCut  cut;

    double Qb_used       = 0.0;   // what the residual was scored against
    double Qb_from_state = 0.0;   // what the state's own bottom BC carries
    double q_cut         = 0.0;   // W/m^2, upward
    double r_cut         = 0.0;   // q_cut - Qb_used; SIGNED
    // Sign, corrected. The sub-column below the cut obeys
    //     C_below * dTbar/dt = Qb - qbar_c = -r_c,
    // so a column with q > Qb (r > 0) is LOSING heat below the cut and cooling.
    // The design note (NOTE 11.4) wrote d_hat = r*P/C with the opposite sign,
    // which made every faithfully relaxing column look arrested and mislabeled
    // 19.4% of the moon3d mesh. Direction lives here so no caller re-derives it.
    double dTdt_below    = 0.0;   // K/s
    double C_below       = 0.0;   // J/m^2/K, nodes strictly below the cut
    double tau1          = 0.0;   // s

    // Kelvin, per observable. lo/hi bracket the flux uncertainty (phase) and
    // the h uncertainty (harmonic mean unknown) together; they are equal when
    // both are known exactly.
    double E_max_lo = 0.0,  E_max_hi = 0.0;
    double E_min_lo = 0.0,  E_min_hi = 0.0;
    double E_mean_lo = 0.0, E_mean_hi = 0.0;
    double E_deep_lo = 0.0, E_deep_hi = 0.0;

    // Arithmetic floor from a float32 checkpoint, in the same kelvin as E_max.
    // Zero when T_is_float32 was not set. Below it no amount of spin-up shows
    // up in a banked state, because the limit is the FORMAT, not the physics.
    double q_quant  = 0.0;   // W/m^2
    double E_floor  = 0.0;   // K

    EqVerdict v_max  = EqVerdict::NotScored;
    EqVerdict v_min  = EqVerdict::NotScored;
    EqVerdict v_mean = EqVerdict::NotScored;
    EqVerdict v_deep = EqVerdict::NotScored;
};

EqCertifyResult eqCertifyColumn(const EqCertifyInput& in);

// ---------------------------------------------------------------------------
// Population-split census + banner for the retrospective certificate.
// Same two-population rule as EqSpinupCensus, and for the same measured
// reason: the frozen dark majority hides the lit faces in any mesh-wide
// aggregate.
// ---------------------------------------------------------------------------
class EqCertifyCensus {
public:
    EqCertifyCensus(double target_K, double P_ref_seconds);
    void add(const EqCertifyResult& r, bool zero_forcing);
    // `cycles_used` is what the run actually integrated (or -1 if unknown).
    // Returns false if nothing was scored.
    bool report(std::ostream& os, double cycles_used) const;

    // Mesh-wide roll-up, for a caller that needs to act on the verdict rather
    // than print it. Fail wins over Indeterminate wins over Pass.
    EqVerdict worstVerdict() const;

private:
    struct Obs {
        long long pass = 0, fail = 0, indet = 0, not_scored = 0;
        std::vector<double> E_hi;
        // E_lo over the FAILING faces: the phase-stripped residual a FAIL
        // actually rests on. Reported because "is the reported E inflated by
        // the un-averaged forcing?" is the first question anyone asks of a
        // FAIL, and E_lo answers it -- it is the number with the whole
        // measured phase bound already subtracted.
        std::vector<double> E_lo_fail;
    };
    struct Bin {
        long long n = 0, invalid = 0, cut_clamped = 0, cut_shallow = 0;
        Obs mx, mn, mean, deep;
        std::vector<double> r_abs, Qb_state, E_floor;
    };
    Bin zero_, forced_;
    mutable std::mutex m_;
    double target_K_, P_ref_;
    static void line(std::ostream& os, const char* label, const Bin& b,
                     double target_K);
    static void obsLine(std::ostream& os, const char* label, const Obs& o,
                        double target_K);
};

}  // namespace thermal

#endif  // THERMAL_EQ_CERTIFICATE_H
