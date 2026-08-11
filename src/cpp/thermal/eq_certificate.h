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
    double Qb = 0.0, eps = 0.95, sigma = 5.67051196e-8;
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

}  // namespace thermal

#endif  // THERMAL_EQ_CERTIFICATE_H
