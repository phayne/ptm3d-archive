// Phase R (R1) CPU unit test for the per-host GPU machine profile record.
//
// Exercises the parts of gpu/machine_profile.h that need NO GPU: JSON
// round-trip, load-or-default fallback + fail-loud warning, corrupt-profile
// rejection, and $PTM_MACHINE_PROFILE path resolution. The actual GPU
// measurement (`ptm_thermal_sim --calibrate-gpu`) is validated separately.
//
// Build/run standalone (no cmake needed):
//   clang++ -std=c++17 src/cpp/tests/test_machine_profile.cpp -o /tmp/tmp_test && /tmp/tmp_test

#include "../gpu/machine_profile.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; } \
    else         { std::fprintf(stdout, "ok:   %s\n", msg); } } while (0)

static bool approx(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    using namespace gpu;

    const std::string path = "/tmp/ptm_test_machine_profile.json";
    std::remove(path.c_str());

    // 1. Conservative default has the documented safe values.
    {
        MachineProfile d = MachineProfile::conservativeDefault();
        CHECK(!d.valid, "conservativeDefault marked invalid (not calibrated)");
        CHECK(d.max_timesteps_per_dispatch == 50, "default ceiling == 50 timesteps");
        CHECK(approx(d.safe_dispatch_ceiling_s, 2.0), "default ceiling == 2 s");
        CHECK(d.recommended_batch_size == 8000, "default batch == 8000");
    }

    // 2. Round-trip: save a measured profile, load it back, fields survive.
    {
        MachineProfile p;
        p.valid = true;
        p.safe_dispatch_ceiling_s = 22.5;
        p.max_timesteps_per_dispatch = 417;
        p.recommended_batch_size = 16000;
        p.kernel_ms_per_step_per_batch = 0.0539;
        p.overhead_per_batch_s = 43.6;
        p.n_layers_probe = 50;
        p.batch_size_probe = 8000;
        p.dz_min_probe = 5.0e-4;
        p.dt_output_probe_s = 3600.0;
        p.kappa_probe = 3.03e-9;
        p.is_run_grid = 1;
        p.host = "TestHost.local";
        p.device_name = "Synthetic GPU";
        p.created_utc = "2026-07-21T05:00:00Z";
        CHECK(saveMachineProfile(p, path), "saveMachineProfile succeeds");

        MachineProfile q;
        CHECK(loadMachineProfile(q, path), "loadMachineProfile succeeds");
        CHECK(q.valid, "loaded profile is valid");
        CHECK(q.max_timesteps_per_dispatch == 417, "round-trip max_timesteps");
        CHECK(approx(q.safe_dispatch_ceiling_s, 22.5), "round-trip ceiling_s");
        CHECK(q.recommended_batch_size == 16000, "round-trip batch_size");
        CHECK(approx(q.kernel_ms_per_step_per_batch, 0.0539), "round-trip kernel cost");
        CHECK(approx(q.overhead_per_batch_s, 43.6), "round-trip overhead");
        CHECK(q.n_layers_probe == 50, "round-trip n_layers_probe");
        CHECK(approx(q.dz_min_probe, 5.0e-4), "round-trip dz_min_probe (v2)");
        CHECK(approx(q.dt_output_probe_s, 3600.0), "round-trip dt_output_probe_s (v2)");
        CHECK(approx(q.kappa_probe, 3.03e-9), "round-trip kappa_probe (v2)");
        CHECK(q.is_run_grid == 1, "round-trip is_run_grid (v2)");
        CHECK(q.host == "TestHost.local", "round-trip host string");
        CHECK(q.device_name == "Synthetic GPU", "round-trip device string");
    }

    // 2b. gridMatchesProfile: the reuse/re-calibrate decision (auto-calibrate).
    {
        MachineProfile s;                 // a representative stored profile
        s.n_layers_probe    = 50;
        s.batch_size_probe  = 8000;
        s.dz_min_probe      = 5.0e-4;     // S = dt*kappa/dz^2 = 3600*3.0e-9/2.5e-7 = 43.2
        s.dt_output_probe_s = 3600.0;
        s.kappa_probe       = 3.0e-9;
        s.is_run_grid       = 1;
        std::string why;

        // Exact grid => match.
        CHECK(gridMatchesProfile(50, 5.0e-4, 3600.0, 3.0e-9, 8000, s, why),
              "gridMatchesProfile: exact grid matches");
        // S within +20% (dt 3600->4320) => still a match (tolerance is 25%).
        CHECK(gridMatchesProfile(50, 5.0e-4, 4320.0, 3.0e-9, 8000, s, why),
              "gridMatchesProfile: S +20% within tolerance matches");
        // S +50% (dt 3600->5400) => mismatch.
        CHECK(!gridMatchesProfile(50, 5.0e-4, 5400.0, 3.0e-9, 8000, s, why),
              "gridMatchesProfile: S +50% is a mismatch");
        // n_layers differs => mismatch (must be exact).
        CHECK(!gridMatchesProfile(48, 5.0e-4, 3600.0, 3.0e-9, 8000, s, why),
              "gridMatchesProfile: n_layers mismatch");
        // batch differs > 2x => mismatch.
        CHECK(!gridMatchesProfile(50, 5.0e-4, 3600.0, 3.0e-9, 20000, s, why),
              "gridMatchesProfile: batch >2x is a mismatch");
        // batch within 2x => match (16000 = 2x of 8000, at the boundary).
        CHECK(gridMatchesProfile(50, 5.0e-4, 3600.0, 3.0e-9, 16000, s, why),
              "gridMatchesProfile: batch 2x within tolerance matches");
        // A pre-v2 profile (no grid provenance) never matches => safe re-calibrate.
        MachineProfile old = s; old.is_run_grid = 0; old.dz_min_probe = 0.0;
        CHECK(!gridMatchesProfile(50, 5.0e-4, 3600.0, 3.0e-9, 8000, old, why),
              "gridMatchesProfile: pre-v2 profile never matches (safe re-calibrate)");
    }

    // 3. Missing file => load fails; load-or-default returns conservative default.
    {
        std::string missing = "/tmp/ptm_test_does_not_exist_ptm.json";
        std::remove(missing.c_str());
        MachineProfile q;
        CHECK(!loadMachineProfile(q, missing), "loadMachineProfile fails on missing file");
        MachineProfile d = loadMachineProfileOrDefault(missing);  // WARNS to stderr
        CHECK(!d.valid && d.max_timesteps_per_dispatch == 50,
              "load-or-default returns conservative default on miss");
    }

    // 3b. Stale ceiling (calibrated against the pre-2026-08-16 25 s budget) is
    //     re-derived DOWN on load from its own measured per-step cost, so a
    //     cached profile cannot keep a ~25 s launch alive. This is the exact
    //     record that killed examples/europa_high_res.yaml: 2986 timesteps at
    //     8.371841 ms/step = 25.0 s per dispatch.
    {
        const std::string stale = "/tmp/ptm_test_stale_profile.json";
        MachineProfile p;
        p.valid = true;
        p.safe_dispatch_ceiling_s = 25.0;
        p.max_timesteps_per_dispatch = 2986;
        p.kernel_ms_per_step_per_batch = 8.371841;
        p.n_layers_probe = 21;
        p.batch_size_probe = 327680;
        p.dz_min_probe = 3.762306e-03;
        p.dt_output_probe_s = 2556.0;
        p.kappa_probe = 2.412985e-08;
        p.is_run_grid = 1;
        CHECK(saveMachineProfile(p, stale), "stale profile written");

        MachineProfile q;
        CHECK(loadMachineProfile(q, stale), "stale profile still loads");
        CHECK(q.max_timesteps_per_dispatch == 238,
              "stale 25 s ceiling re-derived to 238 timesteps (2 s budget)");
        CHECK(approx(q.safe_dispatch_ceiling_s, 2.0),
              "stale profile's declared budget updated to 2 s");
        const double implied = q.max_timesteps_per_dispatch *
                               q.kernel_ms_per_step_per_batch / 1000.0;
        CHECK(implied <= 2.0, "re-derived ceiling implies a <= 2 s launch");
        // The grid signature must survive the downgrade untouched, or the
        // profile would stop matching its own run grid and re-probe forever.
        CHECK(q.n_layers_probe == 21 && q.is_run_grid == 1,
              "downgrade preserves grid provenance");

        // A profile already inside the budget is left exactly as stored.
        MachineProfile ok = p;
        ok.max_timesteps_per_dispatch = 100;   // 100 * 8.37 ms = 0.84 s
        ok.safe_dispatch_ceiling_s = 2.0;
        CHECK(saveMachineProfile(ok, stale), "in-budget profile written");
        MachineProfile r;
        CHECK(loadMachineProfile(r, stale), "in-budget profile loads");
        CHECK(r.max_timesteps_per_dispatch == 100,
              "in-budget ceiling is NOT touched");

        // No measured per-step cost + a laxer declared budget => clamp to the
        // uncalibrated default rather than trust it.
        FILE* f = std::fopen(stale.c_str(), "w");
        std::fputs("{ \"schema_version\": 1, \"max_timesteps_per_dispatch\": 7523, "
                   "\"safe_dispatch_ceiling_s\": 25.0 }", f);
        std::fclose(f);
        MachineProfile s1;
        CHECK(loadMachineProfile(s1, stale), "schema-1 profile loads");
        CHECK(s1.max_timesteps_per_dispatch == 50,
              "schema-1 profile with no per-step cost clamped to 50");
        std::remove(stale.c_str());
    }

    // 4. Corrupt profile (non-positive ceiling) is rejected, not silently used.
    {
        std::string bad = "/tmp/ptm_test_bad_profile.json";
        FILE* f = std::fopen(bad.c_str(), "w");
        std::fputs("{ \"schema_version\": 1, \"max_timesteps_per_dispatch\": -5, "
                   "\"safe_dispatch_ceiling_s\": 25.0 }", f);
        std::fclose(f);
        MachineProfile q;
        CHECK(!loadMachineProfile(q, bad), "corrupt (negative ceiling) profile rejected");
        std::remove(bad.c_str());
    }

    // 5. $PTM_MACHINE_PROFILE path resolution.
    {
        setenv("PTM_MACHINE_PROFILE", path.c_str(), 1);
        std::string resolved = machine_profile_detail::defaultProfilePath();
        CHECK(resolved == path, "PTM_MACHINE_PROFILE overrides default path");
        MachineProfile q;
        CHECK(loadMachineProfile(q, ""), "empty path uses PTM_MACHINE_PROFILE");
        CHECK(q.max_timesteps_per_dispatch == 417, "env-resolved profile loads");
        unsetenv("PTM_MACHINE_PROFILE");
    }

    std::remove(path.c_str());
    if (failures == 0) { std::fprintf(stdout, "\nALL MACHINE-PROFILE TESTS PASSED\n"); return 0; }
    std::fprintf(stderr, "\n%d MACHINE-PROFILE TEST(S) FAILED\n", failures);
    return 1;
}
