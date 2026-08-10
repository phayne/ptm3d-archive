// Accuracy test: compare fastEquilibrate at the standard path's small
// Fourier-stable dt against the same physics driven with a large dt (as
// the SIMD-batch path does). For several latitudes we run 24 lunar days
// of diurnal flux forcing with each dt and report the resulting
// temperature-profile differences.

#include "../thermal/thermal_model.h"
#include "../thermal/planetary_properties.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace {

// Optional recording of the surface-T limit cycle during equilibration. When
// `record` is non-null, surface temperature is appended at every sub-step
// during the LAST `record_last_days` days of equilibration, paired with the
// time-in-cycle in hours.
struct CycleRecorder {
    std::vector<double> t_hours;   // time in hours within the recorded window
    std::vector<double> T_surf;
};

// Re-implementation of Model::fastEquilibrate that takes an explicit dt
// override so we can test alternative step sizes without touching the
// shipping code. Exactly mirrors thermal_model.cpp:2024-2066.
void equilibrate(thermal::Profile& profile,
                 const thermal::Planet& planet,
                 double lat, int days, double dt,
                 CycleRecorder* recorder = nullptr,
                 int record_last_days = 0)
{
    const int steps_per_day = std::max(1, static_cast<int>(planet.day / dt));
    const int record_first_day = (recorder && record_last_days > 0)
                                   ? std::max(0, days - record_last_days) : days;

    for (int day = 0; day < days; ++day) {
        for (int step = 0; step < steps_per_day; ++step) {
            const double day_fraction = static_cast<double>(step) / steps_per_day;
            const double hour_angle = 2.0 * M_PI * day_fraction + M_PI;
            const double cos_zenith = std::cos(lat) * std::cos(hour_angle);
            const double flux_factor = std::max(0.0, cos_zenith);
            const double local_flux = planet.S * flux_factor * (1.0 - planet.albedo);
            profile.update_T(dt, local_flux, planet.Qb);

            if (recorder && day >= record_first_day) {
                const double rel_day = static_cast<double>(day - record_first_day) + day_fraction;
                recorder->t_hours.push_back(rel_day * 24.0);
                recorder->T_surf.push_back(profile.T()[0]);
            }
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

void write_profile_csv(const std::string& path,
                       const std::vector<double>& z,
                       const std::vector<double>& T,
                       const std::vector<double>* T_ref = nullptr)
{
    std::ofstream f(path);
    f << "layer_idx,z_m,T_K";
    if (T_ref) f << ",T_ref_K,abs_dT_K";
    f << "\n";
    const size_t n = std::min(z.size(), T.size());
    for (size_t i = 0; i < n; ++i) {
        f << i << "," << z[i] << "," << T[i];
        if (T_ref && i < T_ref->size()) {
            const double dd = std::abs(T[i] - (*T_ref)[i]);
            f << "," << (*T_ref)[i] << "," << dd;
        }
        f << "\n";
    }
}

void write_cycle_csv(const std::string& path, const CycleRecorder& rec)
{
    std::ofstream f(path);
    f << "t_hours,T_surf_K\n";
    const size_t n = std::min(rec.t_hours.size(), rec.T_surf.size());
    for (size_t i = 0; i < n; ++i) {
        f << rec.t_hours[i] << "," << rec.T_surf[i] << "\n";
    }
}

// The one number this program is a GATE on.
//
// docs/thermal-model.md states the production default
// EQUILIBRATION_STEPS_PER_DAY = 192 was chosen against a "1 K production
// target": at that cadence the worst-case per-layer error against the
// Fourier-stable reference, over all four test latitudes, must stay below 1 K.
// Until 2026-08 this program computed that number, printed it, and returned 0
// unconditionally -- so a regression that doubled the error would still have
// exited 0 and been read as a pass by INSTALL.md's battery and by CI. The
// remaining cadences (768/384/96/48/24) are reported for the convergence trend
// and are NOT gated: they exist to show the dt scaling, and 24 steps/day is
// knowingly far outside the target.
//
// RE-BLESS, 2026-08-06 (PTM-116 scalar surface-Newton seed fix). The gate value
// MOVED relative to the pre-fix binary -- 0.677 K at latitude 0 -> 0.987 K at
// latitude 85 -- but it moved BACK ONTO the recorded documentation, which was
// never updated: docs/thermal-model.md L204, docs/equilibration-paper.md L193
// and docs/ptm3d-thermal-radiation-methods-paper.md L957 all state 0.99 K at
// N=192 "at the 85 deg polar slope", 2.2 K at N=96, 11.9 K at N=24 and 0.21 K
// at N=768. Post-fix this program prints 0.987 / 2.168 / 11.931 / 0.207 at
// latitude 85 -- all four match. Pre-fix it printed 0.442 / 1.111 / 13.097 /
// 0.125 and put the worst latitude at 0 deg, matching none of them. So no
// expected value here or in docs/ needed updating: PTM-116 was a regression
// away from the documented behavior, and the fix restores it.
//
// Root cause of the movement, measured (probe in
// ~/projects/ptm116_2026-08-06/probe_eqdt_arms.cpp): exactly ONE step per
// simulated day. At every candidate cadence N (all divisible by 4) the step at
// day_fraction 0.75 has hour_angle = 2.5*pi, where cos() returns ~3e-16 instead
// of 0, i.e. a "sunlit" step with Qs ~ 1e-13 W/m^2 -- squarely inside PTM-116's
// collapse band. Pre-fix that single step returned a surface T up to 140 K
// below the true root, cold-biasing the coarse-dt arms and partly canceling
// their large-dt warm bias. The Fourier reference arm (4979 steps/day, NOT
// divisible by 4) never lands on that step, which is why it barely moved.
// Zeroing fluxes below 1e-6 W/m^2 makes pre-fix and post-fix agree to <=1e-4 K
// at every cadence -- the whole delta is that one step. Do NOT "fix" the
// forcing: tiny-but-positive surface fluxes are real (Jupiter-shine on Europa,
// PTM-44 fractional-disk visibility), and this step is the canary.
//
// ATTENTION, not resolved here: the docs quoted above also claim the error
// scales as dt^2 (second-order Crank-Nicolson). The recorded numbers themselves
// contradict that -- 0.21 K at N=768 to 0.99 K at N=192 is a factor 4.7 for a
// factor 4 in dt, i.e. order ~1.1, and the post-fix series measured here
// (0.207 / 0.456 / 0.987 / 2.168 / 4.961 / 11.931 at latitude 85) is a clean
// first-order power law. That discrepancy pre-dates PTM-116 and is a claim in
// two manuscripts; it is flagged, not edited.
//
// The 1 K tolerance below is deliberately NOT widened. At latitude 85 the
// production cadence now sits at 0.987 K against it -- a 1.3% margin, where the
// pre-fix binary appeared to have 56%. That margin is real and is a decision
// for the author, not a tolerance to relax.
constexpr int    kGateStepsPerDay = 192;
constexpr double kGateMaxAbsDT_K  = 1.0;

}  // namespace

int main(int argc, char** argv)
{
    using thermal::Planet;
    using thermal::Profile;
    using thermal::Config;

    std::string csv_dir;
    int record_last_days = 3;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv-output") == 0 && i + 1 < argc) {
            csv_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--record-last-days") == 0 && i + 1 < argc) {
            record_last_days = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: %s [--csv-output DIR] [--record-last-days N]\n", argv[0]);
            std::printf("  --csv-output DIR        Write summary.csv, profile_*.csv, cycle_*.csv to DIR\n");
            std::printf("  --record-last-days N    Record surface-T trajectory over last N lunar days (default 3)\n");
            return 0;
        }
    }

    if (!csv_dir.empty()) {
        std::filesystem::create_directories(csv_dir);
    }

    Planet planet("Moon");
    Config config;
    config.m = 10;
    config.n = 5;
    config.b = 10;
    config.F = 0.5;
    config.NPERDAY = 24;
    config.NYEARSEQ = 2;
    config.sigma = 5.67e-8;
    config.chi = 2.7;

    const int days = 24;

    const double latitudes_deg[] = { 0.0, 30.0, 60.0, 85.0 };
    const int candidate_steps_per_day[] = { 768, 384, 192, 96, 48, 24 };

    // Gate bookkeeping: worst max|dT| at the production cadence, over latitudes.
    double gate_worst_dT = -1.0;
    double gate_worst_lat = 0.0;
    int    gate_samples = 0;

    std::printf("\n=== Equilibration accuracy test (Moon, %d days) ===\n", days);
    std::printf("Reference: Fourier-stable dt (mirrors fastEquilibrate)\n");
    std::printf("Candidate: dt = planet.day / N\n");
    std::printf("GATE: max|dT| at N=%d steps/day must be < %.3f K at every latitude\n",
                kGateStepsPerDay, kGateMaxAbsDT_K);
    if (!csv_dir.empty()) {
        std::printf("CSV output: %s\n", csv_dir.c_str());
    }
    std::printf("\n");

    std::ofstream summary;
    if (!csv_dir.empty()) {
        summary.open(csv_dir + "/summary.csv");
        summary << "lat_deg,N_steps_per_day,dt_s,surf_T_K,deep_T_K,"
                   "max_abs_dT_K,mean_abs_dT_K\n";
    }

    for (double lat_deg : latitudes_deg) {
        const double lat = lat_deg * M_PI / 180.0;

        Profile ref_profile(planet, lat, config, /*verbose=*/false);
        const double dt_fourier = compute_fourier_dt(ref_profile, planet.day);
        const int ref_steps_per_day = static_cast<int>(planet.day / dt_fourier);

        std::printf("Latitude %5.1f deg | Fourier dt = %.1f s (%d steps/day)\n",
                    lat_deg, dt_fourier, ref_steps_per_day);

        CycleRecorder ref_rec;
        equilibrate(ref_profile, planet, lat, days, dt_fourier,
                    csv_dir.empty() ? nullptr : &ref_rec, record_last_days);
        std::printf("  Reference final: surf=%.3f K  deep=%.3f K\n",
                    ref_profile.T()[0],
                    ref_profile.T()[ref_profile.T().size() - 1]);

        if (!csv_dir.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s/profile_ref_lat%02d.csv",
                          csv_dir.c_str(), static_cast<int>(lat_deg));
            write_profile_csv(buf,
                              std::vector<double>(ref_profile.z().begin(), ref_profile.z().end()),
                              std::vector<double>(ref_profile.T().begin(), ref_profile.T().end()));
            std::snprintf(buf, sizeof buf, "%s/cycle_ref_lat%02d.csv",
                          csv_dir.c_str(), static_cast<int>(lat_deg));
            write_cycle_csv(buf, ref_rec);
            summary << lat_deg << "," << ref_steps_per_day << "," << dt_fourier << ","
                    << ref_profile.T()[0] << "," << ref_profile.T().back() << ","
                    << 0.0 << "," << 0.0 << "\n";
        }

        for (int N : candidate_steps_per_day) {
            Profile cand(planet, lat, config, /*verbose=*/false);
            const double dt_large = planet.day / N;
            CycleRecorder rec;
            equilibrate(cand, planet, lat, days, dt_large,
                        csv_dir.empty() ? nullptr : &rec, record_last_days);
            auto s = compare(ref_profile, cand);
            std::printf("  N=%3d steps/day (dt=%8.1f s): surf=%7.3f K  deep=%7.3f K  "
                        "max|dT|=%7.3f  mean|dT|=%6.3f%s\n",
                        N, dt_large, s.surf, s.deep_T, s.max_abs_diff, s.mean_abs_diff,
                        N == kGateStepsPerDay ? "   <-- gated" : "");

            if (N == kGateStepsPerDay) {
                ++gate_samples;
                if (s.max_abs_diff > gate_worst_dT) {
                    gate_worst_dT = s.max_abs_diff;
                    gate_worst_lat = lat_deg;
                }
            }

            if (!csv_dir.empty()) {
                char buf[256];
                std::vector<double> Tref(ref_profile.T().begin(), ref_profile.T().end());
                std::snprintf(buf, sizeof buf, "%s/profile_lat%02d_N%04d.csv",
                              csv_dir.c_str(), static_cast<int>(lat_deg), N);
                write_profile_csv(buf,
                                  std::vector<double>(cand.z().begin(), cand.z().end()),
                                  std::vector<double>(cand.T().begin(), cand.T().end()),
                                  &Tref);
                std::snprintf(buf, sizeof buf, "%s/cycle_lat%02d_N%04d.csv",
                              csv_dir.c_str(), static_cast<int>(lat_deg), N);
                write_cycle_csv(buf, rec);
                summary << lat_deg << "," << N << "," << dt_large << ","
                        << s.surf << "," << s.deep_T << ","
                        << s.max_abs_diff << "," << s.mean_abs_diff << "\n";
            }
        }
        std::printf("\n");
    }

    // ---- Verdict ---------------------------------------------------------
    // Printed on ONE greppable line, and reflected in the exit status. CI greps
    // for "[test_equilibration_dt] PASS".
    const int expected_samples =
        static_cast<int>(sizeof(latitudes_deg) / sizeof(latitudes_deg[0]));
    if (gate_samples != expected_samples) {
        std::printf("[test_equilibration_dt] FAIL: the gated cadence N=%d was evaluated "
                    "at %d of %d latitudes -- the candidate list no longer contains the "
                    "production cadence, so this program is not testing what it claims.\n",
                    kGateStepsPerDay, gate_samples, expected_samples);
        return 1;
    }
    if (!(gate_worst_dT < kGateMaxAbsDT_K)) {
        std::printf("[test_equilibration_dt] FAIL: worst max|dT| at N=%d steps/day = "
                    "%.3f K at latitude %.1f deg, tolerance %.3f K "
                    "(docs/thermal-model.md, 1 K production target).\n",
                    kGateStepsPerDay, gate_worst_dT, gate_worst_lat, kGateMaxAbsDT_K);
        return 1;
    }
    std::printf("[test_equilibration_dt] PASS: worst max|dT| at N=%d steps/day = %.3f K "
                "at latitude %.1f deg, tolerance %.3f K (%d/%d latitudes checked).\n",
                kGateStepsPerDay, gate_worst_dT, gate_worst_lat, kGateMaxAbsDT_K,
                gate_samples, expected_samples);
    return 0;
}
