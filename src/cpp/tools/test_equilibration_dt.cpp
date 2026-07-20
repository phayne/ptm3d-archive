// Accuracy test: compare fastEquilibrate at the standard path's small
// Fourier-stable dt against the same physics driven with a large dt (as
// the SIMD-batch path does). For several latitudes we run 24 lunar days
// of diurnal flux forcing with each dt and report the resulting
// temperature-profile differences.

#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace {

// Re-implementation of Model::fastEquilibrate that takes an explicit dt
// override so we can test alternative step sizes without touching the
// shipping code. Exactly mirrors thermal_model.cpp:2024-2066.
void equilibrate(thermal::Profile& profile,
                 const thermal::Planet& planet,
                 double lat, int days, double dt)
{
    const int steps_per_day = std::max(1, static_cast<int>(planet.day / dt));

    for (int day = 0; day < days; ++day) {
        for (int step = 0; step < steps_per_day; ++step) {
            const double day_fraction = static_cast<double>(step) / steps_per_day;
            const double hour_angle = 2.0 * M_PI * day_fraction + M_PI;
            const double cos_zenith = std::cos(lat) * std::cos(hour_angle);
            const double flux_factor = std::max(0.0, cos_zenith);
            const double local_flux = planet.S * flux_factor * (1.0 - planet.albedo);
            profile.update_T(dt, local_flux, planet.Qb);
        }
    }
}

struct ProfileStats {
    double surf;                  // surface T
    double max_abs_diff;          // max |dT| across all layers vs reference
    double mean_abs_diff;         // mean |dT| across all layers vs reference
    double deep_T;                // T at the deepest layer
};

ProfileStats compare(const thermal::Profile& ref, const thermal::Profile& cand)
{
    const auto& Tr = ref.T();
    const auto& Tc = cand.T();
    const size_t n = std::min(Tr.size(), Tc.size());
    double maxd = 0.0, sumd = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::abs(Tr[i] - Tc[i]);
        if (d > maxd) maxd = d;
        sumd += d;
    }
    return { Tc[0], maxd, sumd / std::max<size_t>(1, n), Tc[n - 1] };
}

double compute_fourier_dt(const thermal::Profile& p, double planet_day)
{
    // Replicate thermal_model.cpp:1296 getTimeStep behaviour: dt_stable =
    // 0.49 * dz^2 / kappa at the most restrictive layer (typically layer 1
    // because surface dz is smallest). Start from dt_min = planet.day and
    // shrink.
    const auto& z = p.z();
    const auto& kappa = p.kappa();
    const auto& k_arr = p.k();
    const auto& rho_arr = p.rho();
    const auto& cp_arr = p.cp();
    double dt_min = planet_day;
    for (size_t i = 1; i < z.size(); ++i) {
        const double kappa_val = (kappa.size() == k_arr.size())
                                   ? kappa[i]
                                   : k_arr[i] / (rho_arr[i] * cp_arr[i]);
        const double dz = (i < z.size() - 1) ? (z[i + 1] - z[i])
                                             : (z[i] - z[i - 1]);
        const double dt_stable = 0.49 * dz * dz / kappa_val;
        if (dt_stable < dt_min) dt_min = dt_stable;
    }
    return dt_min;
}

}  // namespace

int main()
{
    using thermal::Planet;
    using thermal::Profile;
    using thermal::Config;

    Planet planet("Moon");
    Config config;
    // Match the production settings used by ptm_thermal_sim.cpp:7253-7261.
    config.m = 10;
    config.n = 5;
    config.b = 10;
    config.F = 0.5;
    config.NPERDAY = 24;
    config.NYEARSEQ = 2;
    config.sigma = 5.67e-8;
    config.chi = 2.7;

    const int days = 24;  // matches NYEARSEQ * round(year/day) for Moon

    const double latitudes_deg[] = { 0.0, 30.0, 60.0, 85.0 };
    const int candidate_steps_per_day[] = { 768, 384, 192, 96, 48, 24 };

    std::printf("\n=== Equilibration accuracy test (Moon, %d days) ===\n", days);
    std::printf("Reference: Fourier-stable dt (mirrors fastEquilibrate)\n");
    std::printf("Candidate: dt = planet.day / N\n\n");

    for (double lat_deg : latitudes_deg) {
        const double lat = lat_deg * M_PI / 180.0;

        // Reference: small Fourier-stable dt.
        Profile ref_profile(planet, lat, config, /*verbose=*/false);
        const double dt_fourier = compute_fourier_dt(ref_profile, planet.day);
        const int ref_steps_per_day = static_cast<int>(planet.day / dt_fourier);

        std::printf("Latitude %5.1f deg | Fourier dt = %.1f s (%d steps/day)\n",
                    lat_deg, dt_fourier, ref_steps_per_day);
        equilibrate(ref_profile, planet, lat, days, dt_fourier);
        std::printf("  Reference final: surf=%.3f K  deep=%.3f K\n",
                    ref_profile.T()[0],
                    ref_profile.T()[ref_profile.T().size() - 1]);

        for (int N : candidate_steps_per_day) {
            Profile cand(planet, lat, config, /*verbose=*/false);
            const double dt_large = planet.day / N;
            equilibrate(cand, planet, lat, days, dt_large);
            auto s = compare(ref_profile, cand);
            std::printf("  N=%3d steps/day (dt=%8.1f s): surf=%7.3f K  deep=%7.3f K  "
                        "max|dT|=%7.3f  mean|dT|=%6.3f\n",
                        N, dt_large, s.surf, s.deep_T, s.max_abs_diff, s.mean_abs_diff);
        }
        std::printf("\n");
    }

    return 0;
}
