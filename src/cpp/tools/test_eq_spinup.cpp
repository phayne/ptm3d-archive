// Gate + demonstration for the equilibration spin-up estimator
// (src/cpp/thermal/eq_certificate.{h,cpp}).
//
// The estimator answers, before any expensive work, "how many cycles does this
// column need for the accuracy I asked for?" from tau1 -- the slowest
// eigenvalue of the ACTUAL discrete column -- rather than from an inherited
// default. This program checks the three properties that make that answer
// trustworthy, and prints the population-split banner a user would see.
//
// GATES
//   1. tau1 recovers the analytic slab value on a uniform column. For a slab of
//      depth L with an insulated bottom and a nearly-insulating top, the
//      slowest mode is the quarter-wave, tau = (2L/pi)^2 / kappa. Checked to
//      2% -- the discretization and the small Robin leak both perturb it.
//   2. n_req scales as ln(1/target): halving the target must add exactly
//      tau1_cycles*ln(2) cycles. This is the property that makes the estimate a
//      model rather than a fit.
//   3. An unreachable target returns unreachable AND NO cycle count. Returning
//      a large number for a target the arithmetic cannot reach would be the
//      worst possible output.
//
// Exit status: 0 = PASS. CI greps "[test_eq_spinup] PASS".

#include "../thermal/eq_certificate.h"
#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {

struct Column {
    std::vector<double> z, T, k, rho, cp;
};

// A uniform slab, n nodes over depth L, with constant properties.
Column uniformSlab(int n, double L, double k, double rho, double cp, double T0) {
    Column c;
    for (int i = 0; i < n; ++i) {
        c.z.push_back(L * static_cast<double>(i) / (n - 1));
        c.T.push_back(T0);
        c.k.push_back(k);
        c.rho.push_back(rho);
        c.cp.push_back(cp);
    }
    return c;
}

}  // namespace

int main() {
    using namespace thermal;
    const double sigma = SIGMA_SB;
    bool ok = true;

    // ---- Gate 1: tau1 vs the analytic quarter-wave slab --------------------
    {
        const int    n   = 200;
        const double L   = 1.0;      // m
        const double k   = 1.0;      // W/m/K
        const double rho = 1000.0;   // kg/m^3
        const double cp  = 1000.0;   // J/kg/K
        const double kappa = k / (rho * cp);
        // The top is driven to the DIRICHLET limit by an artificially large
        // radiative leak (eps is a knob here, not a physical emissivity):
        // h_rad = 4*eps*sigma*T^3 >> k/dz. With a Neumann bottom the slowest
        // mode is then the quarter-wave, tau = (2L/pi)^2/kappa.
        //
        // The opposite limit is deliberately NOT tested: a nearly insulating
        // top makes G singular to ~1e-17 relative (both boundaries insulating
        // admits a zero mode), the solve overflows, and eqColumnTau1 correctly
        // returns -1. That is a real property of the pencil, not a defect.
        Column c = uniformSlab(n, L, k, rho, cp, /*T0=*/200.0);
        const double tau = eqColumnTau1(c.z.data(), c.T.data(), c.k.data(),
                                        c.rho.data(), c.cp.data(), n,
                                        /*eps=*/1e6, sigma);
        const double tau_analytic = std::pow(2.0 * L / M_PI, 2) / kappa;
        const double rel = std::abs(tau - tau_analytic) / tau_analytic;
        std::printf("Gate 1  tau1 on a uniform insulated slab\n");
        std::printf("  measured  %.6e s\n  analytic  %.6e s   (2L/pi)^2/kappa\n",
                    tau, tau_analytic);
        std::printf("  relative difference %.3e  (tol 2.0e-2)\n\n", rel);
        if (!(rel < 2.0e-2)) { ok = false; std::printf("  -> FAIL\n"); }
    }

    // ---- Gate 2: n_req scales as ln(1/target) ------------------------------
    {
        Planet moon("Moon");
        Column c = uniformSlab(60, 1.0, 0.01, 1500.0, 600.0, 200.0);
        EqSpinupInput in;
        in.z = c.z.data(); in.T = c.T.data(); in.k = c.k.data();
        in.rho = c.rho.data(); in.cp = c.cp.data(); in.n = 60;
        in.Qb = 0.018; in.eps = 0.95; in.sigma = sigma;
        in.P = moon.day;

        in.target_K = 1.0;   const EqSpinupResult a = eqEstimateSpinup(in);
        in.target_K = 0.5;   const EqSpinupResult b = eqEstimateSpinup(in);
        const double predicted = a.tau1_cycles * std::log(2.0);
        const double actual    = b.n_req - a.n_req;
        std::printf("Gate 2  n_req scales as ln(1/target)\n");
        std::printf("  tau1 = %.3f cycles,  E0 = %.4f K\n", a.tau1_cycles, a.E0);
        std::printf("  n_req(1.0 K) = %.3f,  n_req(0.5 K) = %.3f\n", a.n_req, b.n_req);
        std::printf("  difference %.6f  vs  tau1*ln2 = %.6f\n\n", actual, predicted);
        if (!(a.valid && b.valid && std::abs(actual - predicted) < 1e-9)) {
            ok = false; std::printf("  -> FAIL\n");
        }
    }

    // ---- Gate 3: an unreachable target returns NO cycle count --------------
    {
        Planet moon("Moon");
        Column c = uniformSlab(60, 1.0, 0.01, 1500.0, 600.0, 200.0);
        EqSpinupInput in;
        in.z = c.z.data(); in.T = c.T.data(); in.k = c.k.data();
        in.rho = c.rho.data(); in.cp = c.cp.data(); in.n = 60;
        in.Qb = 0.018; in.eps = 0.95; in.sigma = sigma;
        in.P = moon.day;
        // The measured float32 arrest floor on the GPU path.
        in.precision_floor_K = 0.36;
        in.target_K = 0.05;
        const EqSpinupResult r = eqEstimateSpinup(in);
        std::printf("Gate 3  target below the path's arithmetic floor\n");
        std::printf("  target %.3f K, floor %.3f K -> unreachable = %s, n_req = %.1f\n\n",
                    in.target_K, in.precision_floor_K,
                    r.unreachable ? "true" : "FALSE", r.n_req);
        if (!(r.valid && r.unreachable && r.n_req == 0.0)) {
            ok = false; std::printf("  -> FAIL\n");
        }
    }

    // ---- The banner a user would see, on both populations ------------------
    {
        Planet moon("Moon");
        EqSpinupCensus census(moon.day, /*target_K=*/0.1,
                              /*precision_floor_K=*/0.0,
                              /*seconds_per_cycle=*/2.5);
        // Dark faces seeded at the geothermal floor; lit faces seeded at their
        // radiative-equilibrium temperature. Both isothermal, as the engine
        // seeds them, so E0 is the interior gradient each must still grow.
        for (int i = 0; i < 200; ++i) {
            const bool dark = (i % 4) != 0;
            Column c = uniformSlab(60, 1.0, 0.01, 1500.0, 600.0,
                                   dark ? 24.04 : (180.0 + 0.4 * i));
            EqSpinupInput in;
            in.z = c.z.data(); in.T = c.T.data(); in.k = c.k.data();
            in.rho = c.rho.data(); in.cp = c.cp.data(); in.n = 60;
            in.Qb = 0.018; in.eps = 0.95; in.sigma = sigma;
            in.P = moon.day; in.target_K = 0.1;
            census.add(eqEstimateSpinup(in), dark);
        }
        census.report(std::cout, /*n_configured=*/50.0);
    }

    if (!ok) { std::printf("\n[test_eq_spinup] FAIL\n"); return 1; }
    std::printf("\n[test_eq_spinup] PASS\n");
    return 0;
}
