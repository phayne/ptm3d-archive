// Convergence gate for the scalar 1-D surface boundary condition
// `thermal::surfTemp` (src/cpp/thermal/thermal_model.cpp).
//
// WHAT IS UNDER TEST
// ------------------
// `surfTemp` solves the surface energy balance
//
//     f(T) = Qs + k0*(T1 - T)/dz0 - eps*sigma*T^4 = 0
//
// with a relaxed, step-limited Newton iteration (relaxation 0.8, step cap
// 0.5*T0, max_iter 10, tolerance 1e-3 K). f is strictly decreasing in T, so
// the root is unique; this program measures how far the iterate actually lands
// from that root, against a float64 bisection solve of the SAME f.
//
// WHY IT EXISTS (PTM-116, the scalar twin of PTM-115)
// ---------------------------------------------------
// The pre-fix seed was `T_eq = (Qs/(eps*sigma))^0.25` whenever `Qs > 0`, and
// `T1` only when Qs was EXACTLY zero. As Qs -> 0+ the seed -> 0 K, while the
// step limiter caps growth at 1.5x per iteration, so over 10 iterations the
// iterate can travel at most 1.5^10 = 57.665x from the seed. Every face with
// 0 < Qs < eps*sigma*(T_root/57.665)^4 therefore returned a spuriously COLD,
// under-converged surface temperature -- exactly 1.5^10 * T_eq. That band is
// reached by the 1-D Model path (Profile::update_T -> surfTemp), which feeds
// runThermalModelForFace and runThermalModelDepthSeries, i.e. the
// depth-profile science products. The CPU MESH path (applySurfaceBCBatch)
// warm-starts from the previous surface T and is immune.
//
// The fix (identical to PTM-115's in the GPU kernel) seeds at the UPPER end of
// the root bracket, max(T_eq, T1).
//
// HOW THE "BEFORE" COLUMN IS PRODUCED
// -----------------------------------
// A single binary cannot call two versions of the shipping function, so this
// program carries a local, clearly-labelled re-implementation of the Newton
// loop with a selectable seed (`newton_emulated`). Its fidelity is not assumed:
// every case also runs the REAL shipping `surfTemp`, and the run FAILS if the
// emulation with the new seed disagrees with it by more than 1e-9 K. The
// emulation was additionally validated against the actual pre-fix binary --
// see the recorded numbers in the header of the gate section below.
//
// Exit status: 0 = PASS. CI/INSTALL greps "[test_surface_newton] PASS".

#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Reference root: bisection on the same f, in float64, to machine precision.
// f is strictly decreasing, f(0) = Qs + k0*T1/dz0 > 0 and
// f(max(T_eq,T1)) <= 0, so [0, max(T_eq,T1)] always brackets the root.
// ---------------------------------------------------------------------------
double reference_root(double Qs, double k0, double dz0, double T1,
                      double eps, double sigma)
{
    auto f = [&](double T) {
        return Qs + k0 * (T1 - T) / dz0 - eps * sigma * T * T * T * T;
    };
    double lo = 0.0;
    double hi = T1;
    if (Qs > 0.0) {
        const double T_eq = std::sqrt(std::sqrt(Qs / (eps * sigma)));
        if (T_eq > hi) hi = T_eq;
    }
    if (hi <= 0.0) return 0.0;
    // Widen defensively; f(hi) must be <= 0.
    int guard = 0;
    while (f(hi) > 0.0 && guard++ < 200) hi *= 2.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;
        if (f(mid) > 0.0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// Local re-implementation of the shipping Newton loop with a selectable seed.
// `new_seed=false` reproduces the PTM-116 defect; `new_seed=true` reproduces
// the fix. Verified against the real surfTemp on every case (see gate below).
// Keep this textually in step with thermal_model.cpp::surfTemp.
// ---------------------------------------------------------------------------
// `max_iter` / `relaxation` default to the shipping values; overriding them is
// used ONLY by the diagnostic that attributes the residual (see main).
double newton_emulated(double Qs, double k0, double dz0, double T1,
                       double eps, double sigma, bool new_seed,
                       int max_iter = 10, double relaxation = 0.8)
{
    double Tnew;
    if (new_seed) {
        Tnew = T1;
        if (Qs > 0.0) {
            const double T_eq = std::sqrt(std::sqrt(Qs / (eps * sigma)));
            if (T_eq > Tnew) Tnew = T_eq;
        }
    } else {
        if (Qs > 0.0) Tnew = std::sqrt(std::sqrt(Qs / (eps * sigma)));
        else          Tnew = T1;
    }

    double dT = 1.0;
    const double tolerance = 0.001;

    for (int iter = 0; iter < max_iter && std::abs(dT) > tolerance; iter++) {
        const double T0 = Tnew;
        const double T0_cubed = T0 * T0 * T0;
        const double T0_fourth = T0 * T0_cubed;

        double f = Qs + k0 * (T1 - T0) / dz0 - eps * sigma * T0_fourth;
        double df = -k0 / dz0 - 4.0 * eps * sigma * T0_cubed;
        if (std::abs(df) < 1e-10) df = -1e-10;

        dT = relaxation * f / df;
        const double max_step = 0.5 * T0;
        if (dT >  max_step) dT =  max_step;
        if (dT < -max_step) dT = -max_step;
        Tnew = T0 - dT;
    }
    return Tnew;
}

struct Case {
    double Qs, k0, dz0, T1, eps;
    double T_ref, T_ship, T_old;
};

// ---------------------------------------------------------------------------
// GATES.
//
// kGateMaxResidual_K -- worst |T_shipping - T_bisection| over the WHOLE grid,
//   including deliberately adversarial corners (T1 = 400 K under a 5 cm layer
//   of k = 7.4e-4 W/m/K receiving 1e-14 W/m^2). Measured 2026-08-06 at the fix:
//   0.092907 K. That residual is NOT a seed problem -- the attribution block
//   printed by this program shows it collapsing to 3.7e-3 K at max_iter = 12,
//   1.5e-4 K at max_iter = 20 and 1.0e-11 K at relaxation = 1.0, i.e. it is the
//   pre-existing truncation of a 10-iteration, 0.8-relaxed loop that has to
//   descend 4x from the seed. Neither max_iter nor the relaxation factor is
//   touched by the PTM-116 fix. Set to 0.15 K: comfortably above the measured
//   worst, far below the 399 K the pre-fix seed produced on the same case.
//   (Fermi's bit-faithful float32 emulation of the GPU twin gave <= 0.022 K on
//   a narrower grid, with the same max_iter=12 -> 8.7e-4 K signature.)
//
// kGatePhysicalResidual_K -- the same worst residual restricted to T1 <= 150 K,
//   which is where every science-relevant case lives (PSR floors, night side,
//   ice-depth profiles). Measured 0.000486 K, so this is gated at the Newton
//   loop's OWN declared tolerance, 1e-3 K. This is the number that matters;
//   the 0.15 K above is a ceiling on a corner nobody occupies.
//
// kGateMinDefectDemo_K -- the pre-fix seed MUST still visibly fail on this
//   grid. If this ever stops holding, the grid no longer covers the defect and
//   the "post-fix residual is small" result is vacuous.
// ---------------------------------------------------------------------------
constexpr double kGateMaxResidual_K      = 0.150;
constexpr double kGatePhysicalResidual_K = 0.001;
constexpr double kGateMinDefectDemo_K    = 10.0;
constexpr double kEmulationTol_K         = 1e-9;

}  // namespace

int main(int argc, char** argv)
{
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: %s [--verbose]\n", argv[0]);
            return 0;
        }
    }

    using thermal::Planet;
    using thermal::Profile;
    using thermal::Config;

    // A real Profile so the SHIPPING surfTemp is what gets measured. Its grid
    // arrays are overwritten per case; only k[0], dz[0], T[1], emissivity and
    // config->sigma are read by surfTemp.
    Planet planet("Moon");
    Config config;
    config.sigma = 5.670374419e-8;
    Profile prof(planet, 0.0, config, /*verbose=*/false);

    // --- Parameter space -----------------------------------------------------
    // Fluxes: the tiny-but-positive band that arms the defect (1e-15 .. 1),
    // plus Qs == 0 exactly (seed unchanged by the fix -> pre-existing case),
    // plus sunlit magnitudes for do-no-harm coverage.
    std::vector<double> Qs_list;
    Qs_list.push_back(0.0);
    for (int e = -15; e <= 0; ++e) {
        for (double m : {1.0, 2.5, 5.0}) Qs_list.push_back(m * std::pow(10.0, e));
    }
    for (double q : {3.0, 10.0, 50.0, 200.0, 600.0, 1361.0}) Qs_list.push_back(q);

    // Subsurface temperature spanning PSR floors to hot sunlit columns.
    const std::vector<double> T1_list =
        {20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0};

    // (k0, dz0) pairs: lunar regolith surface layer through solid water ice.
    // k0 in W/m/K, dz0 in m.
    const std::vector<double> k0_list  = {7.4e-4, 3.4e-3, 1.0e-2, 5.0e-1, 2.5};
    const std::vector<double> dz0_list = {1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2};

    // Lunar regolith and Europa ice emissivities.
    const std::vector<double> eps_list = {0.95, 0.90};

    double worst_ship = -1.0,  worst_old = -1.0,  worst_emul = -1.0;
    Case  worst_ship_case{}, worst_old_case{};
    long  n_cases = 0, n_old_bad_1K = 0, n_old_bad_001K = 0, n_ship_bad_001K = 0;

    // Per-band bookkeeping: "tiny" = 0 < Qs <= 1 W/m^2, the armed band.
    double worst_ship_tiny = -1.0, worst_old_tiny = -1.0;
    // "Physical" subset: T1 <= 150 K. A subsurface node at 400 K under an
    // essentially unilluminated surface is an adversarial construction, not a
    // state any real column reaches; the science-relevant cases (PSR floors,
    // night side, ice-depth profiles) all sit here.
    double worst_ship_phys = -1.0;
    Case worst_ship_phys_case{};

    std::vector<Case> offenders;

    for (double eps : eps_list) {
        prof.emissivity = eps;
        for (double k0 : k0_list) {
            for (double dz0 : dz0_list) {
                for (double T1 : T1_list) {
                    for (double Qs : Qs_list) {
                        // Load the case into the real Profile.
                        prof.k()[0]  = k0;
                        prof.dz()[0] = dz0;
                        prof.T()[1]  = T1;
                        prof.T()[0]  = 0.0;   // surfTemp must not depend on it

                        thermal::surfTemp(prof, Qs);
                        const double T_ship = prof.T()[0];

                        const double T_ref = reference_root(Qs, k0, dz0, T1, eps,
                                                            config.sigma);
                        const double T_old = newton_emulated(Qs, k0, dz0, T1, eps,
                                                             config.sigma, /*new_seed=*/false);
                        const double T_new = newton_emulated(Qs, k0, dz0, T1, eps,
                                                             config.sigma, /*new_seed=*/true);

                        const double d_ship = std::abs(T_ship - T_ref);
                        const double d_old  = std::abs(T_old  - T_ref);
                        const double d_emul = std::abs(T_new  - T_ship);

                        ++n_cases;
                        if (d_emul > worst_emul) worst_emul = d_emul;
                        if (d_old  > 1.0)   ++n_old_bad_1K;
                        if (d_old  > 0.01)  ++n_old_bad_001K;
                        if (d_ship > 0.01)  ++n_ship_bad_001K;

                        if (d_ship > worst_ship) {
                            worst_ship = d_ship;
                            worst_ship_case = {Qs, k0, dz0, T1, eps, T_ref, T_ship, T_old};
                        }
                        if (d_old > worst_old) {
                            worst_old = d_old;
                            worst_old_case = {Qs, k0, dz0, T1, eps, T_ref, T_ship, T_old};
                        }
                        if (Qs > 0.0 && Qs <= 1.0) {
                            if (d_ship > worst_ship_tiny) worst_ship_tiny = d_ship;
                            if (d_old  > worst_old_tiny)  worst_old_tiny  = d_old;
                        }
                        if (T1 <= 150.0 && d_ship > worst_ship_phys) {
                            worst_ship_phys = d_ship;
                            worst_ship_phys_case = {Qs, k0, dz0, T1, eps, T_ref, T_ship, T_old};
                        }
                        if (verbose && d_old > 1.0 && offenders.size() < 20) {
                            offenders.push_back({Qs, k0, dz0, T1, eps, T_ref, T_ship, T_old});
                        }
                    }
                }
            }
        }
    }

    std::printf("\n=== Scalar surface-Newton convergence (thermal::surfTemp) ===\n");
    std::printf("Cases: %ld  (Qs x T1 x k0 x dz0 x eps)\n", n_cases);
    std::printf("Reference: float64 bisection root of the same f(T), 200 iterations\n\n");

    std::printf("  shipping surfTemp   worst |dT| vs root : %10.6f K\n", worst_ship);
    std::printf("    ... restricted to 0 < Qs <= 1 W/m^2  : %10.6f K\n", worst_ship_tiny);
    std::printf("    ... restricted to T1 <= 150 K        : %10.6f K\n", worst_ship_phys);
    std::printf("    ... cases with |dT| > 0.01 K         : %ld / %ld\n",
                n_ship_bad_001K, n_cases);
    std::printf("  PRE-FIX seed (emulated, PTM-116 defect):\n");
    std::printf("    worst |dT| vs root                   : %10.6f K\n", worst_old);
    std::printf("    ... restricted to 0 < Qs <= 1 W/m^2  : %10.6f K\n", worst_old_tiny);
    std::printf("    ... cases with |dT| > 1 K            : %ld / %ld\n",
                n_old_bad_1K, n_cases);
    std::printf("    ... cases with |dT| > 0.01 K         : %ld / %ld\n",
                n_old_bad_001K, n_cases);
    std::printf("  emulation-vs-shipping agreement (new seed): %.3e K (tol %.1e)\n\n",
                worst_emul, kEmulationTol_K);

    std::printf("  worst shipping case: Qs=%.3e k0=%.3e dz0=%.3e T1=%.1f eps=%.2f\n"
                "      root=%.6f  shipping=%.6f  pre-fix=%.6f\n",
                worst_ship_case.Qs, worst_ship_case.k0, worst_ship_case.dz0,
                worst_ship_case.T1, worst_ship_case.eps,
                worst_ship_case.T_ref, worst_ship_case.T_ship, worst_ship_case.T_old);
    std::printf("  worst pre-fix case : Qs=%.3e k0=%.3e dz0=%.3e T1=%.1f eps=%.2f\n"
                "      root=%.6f  shipping=%.6f  pre-fix=%.6f\n\n",
                worst_old_case.Qs, worst_old_case.k0, worst_old_case.dz0,
                worst_old_case.T1, worst_old_case.eps,
                worst_old_case.T_ref, worst_old_case.T_ship, worst_old_case.T_old);

    std::printf("  worst T1<=150 K case: Qs=%.3e k0=%.3e dz0=%.3e T1=%.1f eps=%.2f\n"
                "      root=%.6f  shipping=%.6f  pre-fix=%.6f\n\n",
                worst_ship_phys_case.Qs, worst_ship_phys_case.k0, worst_ship_phys_case.dz0,
                worst_ship_phys_case.T1, worst_ship_phys_case.eps,
                worst_ship_phys_case.T_ref, worst_ship_phys_case.T_ship,
                worst_ship_phys_case.T_old);

    // ---- Residual attribution ----------------------------------------------
    // The post-fix residual is NOT a seed problem: it is the loop running out
    // of iterations. Demonstrate that on the worst case by relaxing ONLY
    // max_iter / the relaxation factor (neither is changed in the shipping
    // code -- this is a measurement, not a proposal). If the residual collapses
    // when the iterate is allowed to finish, the seed has done its job and what
    // remains is the pre-existing (relaxation 0.8, max_iter 10) truncation.
    {
        const Case& c = worst_ship_case;
        std::printf("  residual attribution on the worst case "
                    "(seed = upper bracket in all four rows):\n");
        struct Variant { const char* label; int iters; double omega; };
        const Variant variants[] = {
            {"max_iter=10, w=0.8  (SHIPPING)", 10, 0.8},
            {"max_iter=12, w=0.8           ", 12, 0.8},
            {"max_iter=20, w=0.8           ", 20, 0.8},
            {"max_iter=10, w=1.0           ", 10, 1.0},
        };
        for (const auto& v : variants) {
            const double T = newton_emulated(c.Qs, c.k0, c.dz0, c.T1, c.eps,
                                             config.sigma, /*new_seed=*/true,
                                             v.iters, v.omega);
            std::printf("    %s -> |dT| = %.3e K\n", v.label, std::abs(T - c.T_ref));
        }
        std::printf("\n");
    }

    if (verbose) {
        std::printf("  sample pre-fix failures (|dT| > 1 K):\n");
        for (const auto& c : offenders) {
            std::printf("    Qs=%.3e k0=%.3e dz0=%.3e T1=%6.1f eps=%.2f | root=%9.4f "
                        "pre-fix=%9.4f shipping=%9.4f\n",
                        c.Qs, c.k0, c.dz0, c.T1, c.eps, c.T_ref, c.T_old, c.T_ship);
        }
        std::printf("\n");
    }

    // ---- Verdict -----------------------------------------------------------
    if (!(worst_emul <= kEmulationTol_K)) {
        std::printf("[test_surface_newton] FAIL: the local Newton emulation no longer "
                    "reproduces the shipping surfTemp (max disagreement %.3e K > %.1e K). "
                    "The 'pre-fix' column is therefore not trustworthy -- re-sync "
                    "newton_emulated() with thermal_model.cpp::surfTemp.\n",
                    worst_emul, kEmulationTol_K);
        return 1;
    }
    if (!(worst_old >= kGateMinDefectDemo_K)) {
        std::printf("[test_surface_newton] FAIL: the pre-fix seed no longer fails on this "
                    "grid (worst %.6f K < %.1f K expected). The parameter grid no longer "
                    "covers the PTM-116 band, so the post-fix result below is vacuous.\n",
                    worst_old, kGateMinDefectDemo_K);
        return 1;
    }
    if (!(worst_ship <= kGateMaxResidual_K)) {
        std::printf("[test_surface_newton] FAIL: shipping surfTemp worst |dT| vs the "
                    "bisection root = %.6f K over the full grid, tolerance %.3f K "
                    "(PTM-116).\n", worst_ship, kGateMaxResidual_K);
        return 1;
    }
    if (!(worst_ship_phys <= kGatePhysicalResidual_K)) {
        std::printf("[test_surface_newton] FAIL: shipping surfTemp worst |dT| vs the "
                    "bisection root = %.6f K on the science-relevant T1 <= 150 K "
                    "subset, tolerance %.3f K (the Newton loop's own declared "
                    "tolerance) (PTM-116).\n",
                    worst_ship_phys, kGatePhysicalResidual_K);
        return 1;
    }
    std::printf("[test_surface_newton] PASS: shipping surfTemp worst |dT| = %.6f K "
                "full grid (tol %.3f K) / %.6f K for T1 <= 150 K (tol %.3f K); the "
                "pre-fix seed would have been off by %.3f K on the same grid "
                "(%ld/%ld cases > 1 K).\n",
                worst_ship, kGateMaxResidual_K, worst_ship_phys,
                kGatePhysicalResidual_K, worst_old, n_old_bad_1K, n_cases);
    return 0;
}
