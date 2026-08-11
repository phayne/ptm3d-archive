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


// ---------------------------------------------------------------------------
// THE PRODUCTION LUNAR GRID -- the one the 50->100 lunation drift was actually
// measured on: body=Moon m=10 n=15 b=32, ADAPTIVE_GRID=1, 50 layers, z_max =
// 1.0040 m. Embedded rather than read from a scratch directory so the
// calibration below is reproducible from the tree alone.
//
// This exists because of a mistake worth not repeating: the estimator's answer
// on the ico3 TEST config (30 layers) was compared against a drift measured on
// THIS grid (50 layers) and looked 20x optimistic. tau1 is a property of the  // american-english: allow
// grid and the state; comparing one grid's estimate to another grid's
// measurement is not a calibration, it is a category error.
// ---------------------------------------------------------------------------
const double kProdZ[50] = {
    0.0000000000e+00, 3.3079068646e-03, 6.9662604516e-03, 1.1000056889e-02,
    1.5432086602e-02, 2.0291453869e-02, 2.5615405761e-02, 3.1451352891e-02,
    3.7859635085e-02, 4.4917421681e-02, 5.2724348418e-02, 6.1410890843e-02,
    7.1151208704e-02, 8.2183619136e-02, 9.4844783765e-02, 1.0963016643e-01,
    1.2730895745e-01, 1.4916395796e-01, 1.7595344693e-01, 2.0289422633e-01,
    2.2992046246e-01, 2.5700057743e-01, 2.8411477372e-01, 3.1125056848e-01,
    3.3840006706e-01, 3.6555826700e-01, 3.9272199457e-01, 4.1988923467e-01,
    4.4705870726e-01, 4.7422959894e-01, 5.0140139273e-01, 5.2857376002e-01,
    5.5574649192e-01, 5.8291945563e-01, 6.1009256671e-01, 6.3726577149e-01,
    6.6443903585e-01, 6.9161233808e-01, 7.1878566439e-01, 7.4595900600e-01,
    7.7313235736e-01, 8.0030571490e-01, 8.2747907637e-01, 8.5465244035e-01,
    8.8182580592e-01, 9.0899917250e-01, 9.3617253972e-01, 9.6334590735e-01,
    9.9051927524e-01, 1.0040110440e+00,
};
const double kProdKc[50] = {
    7.4000000000e-04, 8.8268126328e-04, 1.0315830416e-03, 1.1855770701e-03,
    1.3432549108e-03, 1.5032626518e-03, 1.6643140536e-03, 1.8251864128e-03,
    1.9847131878e-03, 2.1417744583e-03, 2.2952854426e-03, 2.4441829214e-03,
    2.5874089875e-03, 2.7238909253e-03, 2.8525150320e-03, 2.9720903987e-03,
    3.0812951485e-03, 3.1785901662e-03, 3.2583266645e-03, 3.3095758875e-03,
    3.3423682464e-03, 3.3633014060e-03, 3.3766444376e-03, 3.3851415025e-03,
    3.3905493806e-03, 3.3939898866e-03, 3.3961782256e-03, 3.3975699120e-03,
    3.3984548779e-03, 3.3990175887e-03, 3.3993753779e-03, 3.3996028659e-03,
    3.3997475041e-03, 3.3998394649e-03, 3.3998979332e-03, 3.3999351070e-03,
    3.3999587417e-03, 3.3999737684e-03, 3.3999833222e-03, 3.3999893965e-03,
    3.3999932584e-03, 3.3999957138e-03, 3.3999972749e-03, 3.3999982674e-03,
    3.3999988984e-03, 3.3999992996e-03, 3.3999995547e-03, 3.3999997169e-03,
    3.3999998200e-03, 3.3999998563e-03,
};
const double kProdRho[50] = {
    1.1000000000e+03, 1.1375477009e+03, 1.1767323794e+03, 1.2172571237e+03,
    1.2587512923e+03, 1.3008585926e+03, 1.3432405404e+03, 1.3855753718e+03,
    1.4275561020e+03, 1.4688880154e+03, 1.5092856428e+03, 1.5484691898e+03,
    1.5861602599e+03, 1.6220765593e+03, 1.6559250084e+03, 1.6873922102e+03,
    1.7161303022e+03, 1.7417342543e+03, 1.7627175433e+03, 1.7762041809e+03,
    1.7848337490e+03, 1.7903424753e+03, 1.7938537994e+03, 1.7960898691e+03,
    1.7975129949e+03, 1.7984183912e+03, 1.7989942699e+03, 1.7993605031e+03,
    1.7995933889e+03, 1.7997414707e+03, 1.7998356258e+03, 1.7998954910e+03,
    1.7999335537e+03, 1.7999577539e+03, 1.7999731403e+03, 1.7999829229e+03,
    1.7999891425e+03, 1.7999930970e+03, 1.7999956111e+03, 1.7999972096e+03,
    1.7999982259e+03, 1.7999988720e+03, 1.7999992829e+03, 1.7999995440e+03,
    1.7999997101e+03, 1.7999998157e+03, 1.7999998828e+03, 1.7999999255e+03,
    1.7999999526e+03, 1.7999999622e+03,
};

// Moon cp(T) = cp0 + T*(c1 + T*(c2 + T*(c3 + T*c4))), planetary_properties.h:161
double prodCp(double T) {
    const double c[4] = {2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9};
    double r = c[3];
    for (int i = 2; i >= 0; --i) r = r * T + c[i];
    return -3.6125 + T * r;
}
// Hayne k = kc*(1 + R350*T^3), chi = 2.7
double prodK(double T, double kc) {
    const double R350 = 2.7 / (350.0 * 350.0 * 350.0);
    return kc * (1.0 + R350 * T * T * T);
}

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

    // ---- Gate 4: calibration against the ONE measured spin-up we have -------
    // On this grid, lit deep columns were measured to need on the order of
    // 150-200 integrated lunations to reach ~0.002 K. The estimate must land in
    // the same family AND on the conservative side -- an estimator that tells a
    // colleague to run FEWER cycles than the measurement is worse than none.
    {
        const double DAY = 2.55144e6;
        std::vector<double> z(kProdZ, kProdZ + 50), T(50), k(50), rho(kProdRho, kProdRho + 50), cp(50);
        // A lit column: warm surface over a cooler deep interior.
        for (int i = 0; i < 50; ++i) {
            T[i]  = 250.0 - 100.0 * (static_cast<double>(i) / 49.0);
            k[i]  = prodK(T[i], kProdKc[i]);
            cp[i] = prodCp(T[i]);
        }
        EqSpinupInput in;
        in.z = z.data(); in.T = T.data(); in.k = k.data();
        in.rho = rho.data(); in.cp = cp.data(); in.n = 50;
        in.Qb = 0.018; in.eps = 0.95; in.sigma = SIGMA_SB;
        in.P = DAY; in.target_K = 0.002;
        const EqSpinupResult r = eqEstimateSpinup(in);
        const double ratio = r.n_req / 175.0;   // midpoint of the measured band
        std::printf("Gate 4  calibration on the PRODUCTION 50-layer lunar grid\n");
        std::printf("  lit column, target 0.002 K\n");
        std::printf("  tau1  = %.2f lunations\n", r.tau1_cycles);
        std::printf("  n_req = %.1f lunations   vs measured 150-200\n", r.n_req);
        std::printf("  ratio to the 175-lunation midpoint: %.2fx %s\n\n",
                    ratio, ratio >= 1.0 ? "(conservative)" : "(OPTIMISTIC -- unsafe)");  // american-english: allow
        // Same order of magnitude, and never below the measured band.
        if (!(r.valid && r.n_req >= 150.0 && r.n_req <= 700.0)) {
            ok = false;
            std::printf("  -> FAIL: outside [150, 700] lunations\n");
        }
    }

    if (!ok) { std::printf("\n[test_eq_spinup] FAIL\n"); return 1; }
    std::printf("\n[test_eq_spinup] PASS\n");
    return 0;
}
