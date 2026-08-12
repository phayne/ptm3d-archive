// Gate for the ONE shared spin-up seed (src/cpp/thermal/eq_seed.h).
//
// Why this file exists: before 2026-08-11 the engine shipped four different
// initial conditions spanning 46.7 K on a permanently shadowed lunar column,
// and NO test compared any two of them. The divergence was found by hand.
// Everything below is a property the seed must have on EVERY solver path, so
// that a future edit to one path cannot silently reintroduce a private seed.
//
// Build: cmake --build build --target test_eq_seed
// Run:   ./build/bin/test_eq_seed        (exit 0 = pass)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../thermal/eq_seed.h"
#include "../thermal/thermal_model.h"

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!ok) ++g_fail;
}

// A lunar-like 40-node geometric column, ks/kd/rhos/rhod/H from the Moon.
struct Column {
    std::vector<double> z, kc, rho;
    double ks = 7.4e-4, kd = 3.4e-3, rhos = 1100.0, rhod = 1800.0, H = 0.06;
    explicit Column(int n = 40, double z_max = 1.2) {
        z.resize(n); kc.resize(n); rho.resize(n);
        // Geometric stretching, z[0] = 0 exactly (the surface node).
        const double r = std::pow(z_max / 1e-3, 1.0 / (n - 2));
        double dz = 1e-3;
        z[0] = 0.0;
        for (int i = 1; i < n; ++i) { z[i] = z[i - 1] + dz; dz *= r; }
        const double scale = z_max / z[n - 1];
        for (int i = 0; i < n; ++i) z[i] *= scale;
        for (int i = 0; i < n; ++i) {
            const double df = std::exp(-z[i] / H);
            kc[i]  = kd   + (ks   - kd  ) * df;
            rho[i] = rhod + (rhos - rhod) * df;
        }
    }
};

thermal::EqSeedInput moonInput(const Column& c, double F, double Qb = 0.018) {
    thermal::EqSeedInput in;
    in.mean_absorbed_flux = F;
    in.Qb                 = Qb;
    in.emissivity         = 0.95;
    in.sigma              = thermal::SIGMA_SB;
    in.z    = c.z.data();
    in.kc   = c.kc.data();
    in.rho  = c.rho.data();
    in.n    = static_cast<int>(c.z.size());
    in.R350 = thermal::R350(2.7);
    return in;
}

}  // namespace

int main() {
    std::printf("=== shared equilibration seed (thermal/eq_seed.h) ===\n");
    const Column col;
    const int n = static_cast<int>(col.z.size());
    std::vector<double> T(n);

    // -----------------------------------------------------------------------
    // 1. A permanently shadowed column seeds EXACTLY on the geothermal floor.
    //    This is gate G2. The floor the solver enforces every step is
    //    T_geo = (Qb/(eps*sigma))^0.25; the seed must land on it, not 42-47 K
    //    above it (CPU SIMD shipped 66.246 K, GPU 71.028 K against 24.043 K).
    // -----------------------------------------------------------------------
    {
        auto in = moonInput(col, 0.0);
        const auto r = thermal::eqSeedColumn(in, T.data());
        const double T_geo = std::pow(in.Qb / (in.emissivity * in.sigma), 0.25);
        check(std::fabs(r.T_surface - T_geo) <= 1e-12,
              "dark column seeds at T_geo",
              "T_surf = " + std::to_string(r.T_surface) +
              " K, T_geo = " + std::to_string(T_geo) + " K");
        check(r.T_deep > r.T_surface,
              "dark column carries the geothermal gradient",
              "T_deep - T_surf = " + std::to_string(r.T_deep - r.T_surface) + " K");
    }

    // -----------------------------------------------------------------------
    // 2. The profile satisfies the steady-state PHYSICS residual, not an
    //    increment test: the conducted flux at every interface equals Qb.
    //    (An arrested column stops changing; only a residual against physics
    //    can tell "converged" from "stuck".)
    // -----------------------------------------------------------------------
    {
        auto in = moonInput(col, 0.0);
        thermal::eqSeedColumn(in, T.data());
        double worst = 0.0;
        int worst_i = -1;
        for (int i = 1; i < n; ++i) {
            const double k = thermal::hayneK(T[i - 1], col.kc[i - 1], in.R350);
            const double q = k * (T[i] - T[i - 1]) / (col.z[i] - col.z[i - 1]);
            const double e = std::fabs(q - in.Qb) / in.Qb;
            if (e > worst) { worst = e; worst_i = i; }
        }
        // The reconstruction differences two nearby temperatures, so the bound
        // is set by cancellation in T, not by the integration: ulp(T)/dT per
        // interface. 1e-9 relative is ~5 orders above that and ~9 orders below
        // anything physical.
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "worst |q - Qb|/Qb = %.3e at node %d", worst, worst_i);
        check(worst <= 1e-9, "interface conducted flux == Qb everywhere", buf);
    }

    // -----------------------------------------------------------------------
    // 3. CONTINUITY as F -> 0. The legacy GPU seed had `if (T_eq < 40) T_eq =
    //    100.0`, a 60 K jump at an arbitrary flux threshold; the legacy CPU
    //    seed had a 200 K zero-flux fallback and a 50 K floor. A face with
    //    1e-30 W/m^2 must not seed 60 K away from a face with exactly 0.
    // -----------------------------------------------------------------------
    {
        const auto a = thermal::eqSeedColumn(moonInput(col, 0.0), nullptr);
        const auto b = thermal::eqSeedColumn(moonInput(col, 1e-12), nullptr);
        check(std::fabs(b.T_surface - a.T_surface) < 1e-9,
              "seed is continuous at F -> 0+",
              "|dT| = " + std::to_string(std::fabs(b.T_surface - a.T_surface)) + " K");

        // The falsifier for the legacy substitutions. Sweep the flux ACROSS
        // the legacy GPU threshold (T_eq < 40 K -> 100 K, i.e. F ~ 0.14 W/m^2)
        // and require the closed form everywhere. The legacy branch would miss
        // by tens of kelvin here; a re-introduced clamp anywhere in the seed
        // shows up as a finite worst-case error.
        double worst = 0.0, worst_F = 0.0;
        for (double F = 0.0; F <= 1400.0; F = (F == 0.0 ? 1e-6 : F * 1.5)) {
            const auto r = thermal::eqSeedColumn(moonInput(col, F), nullptr);
            const double want = std::pow((F + 0.018) / (0.95 * thermal::SIGMA_SB), 0.25);
            const double e = std::fabs(r.T_surface - want);
            if (e > worst) { worst = e; worst_F = F; }
        }
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "worst |T - ((F+Qb)/(eps*sigma))^0.25| = %.3e K at F = %.3g W/m^2",
                      worst, worst_F);
        check(worst <= 1e-12,
              "surface anchor is the closed form at every flux, no thresholds", buf);
    }

    // -----------------------------------------------------------------------
    // 4. Qb <= 0 (an insulated base -- reachable: rockox_generic-test.yaml
    //    sets Qb: 0.0) gives an ISOTHERMAL column, and Qb == 0 with F == 0
    //    hits the documented CMB floor with the flag set, not a silent 0 K
    //    column that makes cp(T) negative.
    // -----------------------------------------------------------------------
    {
        auto in = moonInput(col, 40.0, 0.0);
        const auto r = thermal::eqSeedColumn(in, T.data());
        double spread = 0.0;
        for (int i = 0; i < n; ++i) spread = std::max(spread, std::fabs(T[i] - T[0]));
        check(spread == 0.0, "Qb = 0 gives an isothermal column",
              "spread = " + std::to_string(spread) + " K");

        const auto z = thermal::eqSeedColumn(moonInput(col, 0.0, 0.0), nullptr);
        check(z.T_surface == thermal::ABSOLUTE_ZERO_K && z.clamped_low,
              "Qb = 0 AND F = 0 lands on the CMB floor and SAYS so",
              "T = " + std::to_string(z.T_surface) +
              " K, clamped_low = " + std::to_string(int(z.clamped_low)));
    }

    // -----------------------------------------------------------------------
    // 5. NO 400 K clamp. Mercury's subsolar cycle mean absorbed flux is
    //    ~2557 W/m^2, i.e. ~468 K -- the old seeds clamped that to 400 K
    //    silently. Above the CPU solver's own MAX_PHYSICAL_TEMP_K the seed
    //    must say so, because THAT clamp will truncate the run.
    // -----------------------------------------------------------------------
    {
        const auto m = thermal::eqSeedColumn(moonInput(col, 2557.0), nullptr);
        check(m.T_surface > 400.0,
              "Mercury-scale mean flux is not clamped to 400 K",
              "T_surf = " + std::to_string(m.T_surface) + " K");
        check(!m.above_500K, "468 K does not trip the MAX_PHYSICAL_TEMP_K flag");
        const auto h = thermal::eqSeedColumn(moonInput(col, 6000.0), nullptr);
        check(h.above_500K, "a seed above 500 K trips the flag",
              "T_surf = " + std::to_string(h.T_surface) + " K");
    }

    // -----------------------------------------------------------------------
    // 6. DETERMINISM / cross-path identity. Every call site marshals its own
    //    arrays; the guarantee this file pins is that identical inputs give a
    //    bit-identical column no matter how they were assembled. The GPU host
    //    seed deliberately reads the DOUBLE reference kc/rho rather than the
    //    float32 upload buffers so that this holds there too.
    // -----------------------------------------------------------------------
    {
        std::vector<double> A(n), B(n);
        auto in = moonInput(col, 137.0);
        thermal::eqSeedColumn(in, A.data());
        // Re-marshal the same numbers through fresh storage, as a different
        // call site would.
        std::vector<double> z2(col.z), kc2(col.kc), rho2(col.rho);
        auto in2 = in;
        in2.z = z2.data(); in2.kc = kc2.data(); in2.rho = rho2.data();
        thermal::eqSeedColumn(in2, B.data());
        bool same = true;
        for (int i = 0; i < n; ++i) if (A[i] != B[i]) same = false;
        check(same, "identical inputs give a BIT-identical column");
    }

    // -----------------------------------------------------------------------
    // 7. Martinez 2021 conductivity: the seed must integrate the SELECTED law,
    //    not silently fall back to Hayne, and must stay finite and monotone.
    // -----------------------------------------------------------------------
    {
        auto in = moonInput(col, 0.0);
        in.cond = thermal::ConductivityModel::Martinez2021;
        const auto r = thermal::eqSeedColumn(in, T.data());
        bool finite = true, monotone = true;
        for (int i = 0; i < n; ++i) if (!std::isfinite(T[i])) finite = false;
        for (int i = 1; i < n; ++i) if (T[i] < T[i - 1]) monotone = false;
        check(finite && monotone, "Martinez column is finite and monotone",
              "T_deep = " + std::to_string(r.T_deep) + " K");
        const auto hay = thermal::eqSeedColumn(moonInput(col, 0.0), nullptr);
        check(std::fabs(r.T_deep - hay.T_deep) > 1e-6,
              "Martinez actually changes the profile (the law is not ignored)",
              "Martinez T_deep - Hayne T_deep = " +
              std::to_string(r.T_deep - hay.T_deep) + " K");
    }

    // -----------------------------------------------------------------------
    // 8. The analytic clear-sky pre-conditioner is OFF under the physical
    //    seed and ON at its historic length under legacy. It is a forcing,
    //    not a seed; nothing may turn it on by accident.
    // -----------------------------------------------------------------------
    {
        const double day = 29.53 * 86400.0, year = 365.25 * 86400.0;
        check(thermal::analyticSeedCycles(thermal::EqSeedMode::Physical, -1.0,
                                          day, year, 2) == 0.0,
              "physical mode: pre-conditioner OFF");
        check(thermal::analyticSeedCycles(thermal::EqSeedMode::Legacy, -1.0,
                                          day, year, 2) == 24.0,
              "legacy mode: 24 lunar cycles, the historic length");
        check(thermal::analyticSeedCycles(thermal::EqSeedMode::Physical, 3.0,
                                          day, year, 2) == 3.0,
              "an explicit request wins over the mode");
        thermal::EqSeedMode m;
        check(!thermal::parseEqSeedMode("perfaceflux", m),
              "an unknown seed name is REFUSED, not defaulted");
    }

    std::printf("=== %s (%d failure%s) ===\n", g_fail ? "FAILED" : "OK",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
