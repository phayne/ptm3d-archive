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
        CHECK(d.max_timesteps_per_dispatch == 300, "default ceiling == 300 timesteps");
        CHECK(approx(d.safe_dispatch_ceiling_s, 25.0), "default ceiling == 25 s");
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
        CHECK(q.host == "TestHost.local", "round-trip host string");
        CHECK(q.device_name == "Synthetic GPU", "round-trip device string");
    }

    // 3. Missing file => load fails; load-or-default returns conservative default.
    {
        std::string missing = "/tmp/ptm_test_does_not_exist_ptm.json";
        std::remove(missing.c_str());
        MachineProfile q;
        CHECK(!loadMachineProfile(q, missing), "loadMachineProfile fails on missing file");
        MachineProfile d = loadMachineProfileOrDefault(missing);  // WARNS to stderr
        CHECK(!d.valid && d.max_timesteps_per_dispatch == 300,
              "load-or-default returns conservative default on miss");
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
