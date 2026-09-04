// Standalone logic test for the GPU guardrail policy decisions.
//
// Build (no OpenCL, no CMake target needed -- the header is dependency-free):
//   c++ -std=c++17 -I src/cpp src/cpp/tests/test_gpu_guardrails.cpp -o test_gpu_guardrails
// Run ALL of these; each process can only exercise ONE platform-gate state,
// because wedgeProneDriver() memoizes PTM_GPU_FORCE_WEDGE_PLATFORM on first use
// (deliberately: the gate must not change under a run's feet):
//   ./test_gpu_guardrails native
//   PTM_GPU_FORCE_WEDGE_PLATFORM=1 ./test_gpu_guardrails wedgeprone
//   PTM_GPU_FORCE_WEDGE_PLATFORM=0 ./test_gpu_guardrails notwedgeprone
//
// Why this exists: the enforcing branches are reachable on the project's
// macOS+AMD host with a large view-factor matrix, but the ADVISORY branches
// (below the floor on a platform where the wedge was never observed) are not
// reachable there at all, and no other platform is available. Without this the
// gating logic would ship exercised in one direction only.

#include "gpu/gpu_guardrails.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace gg = gpu::guardrails;

static int g_failures = 0;

static void check(const std::string& what, bool ok) {
    std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

static void resetPolicy(const std::string& placement, double placement_margin,
                        double min_margin, bool explicit_margin,
                        const std::string& thin_margin_policy = "separate",
                        double hard_floor = 0.25) {
    gg::Policy& p = gg::policy();
    p.placement = placement;
    p.placement_margin_gib = placement_margin;
    p.min_vram_margin_gib = min_margin;
    p.min_vram_margin_explicit = explicit_margin;
    p.thin_margin_policy = thin_margin_policy;
    p.hard_min_vram_margin_gib = hard_floor;
    gg::PlacementDecision& d = gg::placementDecision();
    d = gg::PlacementDecision();
}

static void setThermalIntent(bool use_gpu, int devices, std::vector<int> ids) {
    gg::ThermalIntent& t = gg::thermalIntent();
    t.published = true;
    t.use_gpu = use_gpu;
    t.devices = devices;
    t.device_ids = std::move(ids);
}

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "native";
    const std::string amd = "AMD (Advanced Micro Devices, Inc.)";
    const std::string nvidia = "NVIDIA Corporation";
    const bool gate_amd = gg::wedgeProneDriver(amd);
    std::cout << "-- mode " << mode << ": wedgeProneDriver(AMD) = "
              << (gate_amd ? "true" : "false") << "\n";

    // ---- vendor detection (independent of the platform gate) --------------
    check("vendorIsAmd recognizes the AMD OpenCL vendor string",
          gg::vendorIsAmd(amd));
    check("vendorIsAmd recognizes a bare ATI string", gg::vendorIsAmd("ATI"));
    check("vendorIsAmd rejects NVIDIA", !gg::vendorIsAmd(nvidia));
    check("vendorIsAmd rejects Intel", !gg::vendorIsAmd("Intel(R) Corporation"));

    // ---- the VRAM-margin floor: enforce or advise -------------------------
    resetPolicy("auto", 4.0, 2.0, /*explicit=*/false);
    check("default floor enforces IFF the driver is wedge-prone",
          gg::enforceVramMargin(amd) == gate_amd);
    resetPolicy("auto", 4.0, 2.0, /*explicit=*/true);
    check("an EXPLICIT floor enforces regardless of platform",
          gg::enforceVramMargin(amd) && gg::enforceVramMargin(nvidia));
    resetPolicy("auto", 4.0, 2.0, /*explicit=*/false);
    check("default floor on a non-AMD device follows the platform gate too",
          gg::enforceVramMargin(nvidia) == gg::wedgeProneDriver(nvidia));

    // ---- thin-margin ORDERING: separate (default) vs refuse ---------------
    // lunar polar site B's real geometry: the replica fits by +0.64 GiB against a 2.0
    // GiB floor. Under the DEFAULT ("separate", Paul's ruling 2026-08-22) that
    // site keeps the GPU radiosity rung behind disjoint dies at a derived
    // +7-10 min/window; under the explicit "refuse" it loses the rung, which
    // MEASURED 108.9 min/window against 31.4 on the GPU pool. These cases pin
    // which verdict each ordering produces.
    using V = gg::MarginVerdict;
    const double site_b_margin_gib = 0.64;

    // The BUILT-IN default, before anything sets it. config_parser.h carries
    // the same string; the generated engine-config table is what keeps those
    // two honest (src/python/ptm/engine_defaults.py).
    check("built-in default thin_margin_policy is 'separate'",
          gg::Policy().thin_margin_policy == "separate");
    check("built-in default hard floor is 0.25 GiB",
          gg::Policy().hard_min_vram_margin_gib == 0.25);

    resetPolicy("auto", 4.0, 2.0, true);          // default thin-margin policy
    check("DEFAULT: site B's +0.64 GiB KEEPS the GPU rung when separated",
          gg::judgeVramMargin(site_b_margin_gib, /*separated=*/true, false, amd)
              == V::WaiveSeparation);
    check("DEFAULT: the same margin still refuses when NOT separated",
          gg::judgeVramMargin(site_b_margin_gib, false, false, amd) == V::Refuse);

    resetPolicy("auto", 4.0, 2.0, true, "refuse");
    check("explicit refuse: site B's +0.64 GiB loses the GPU rung even when "
          "separated",
          gg::judgeVramMargin(site_b_margin_gib, /*separated=*/true, false, amd) == V::Refuse);

    resetPolicy("auto", 4.0, 2.0, true, "separate", 0.25);
    check("separate: site B's +0.64 GiB KEEPS the GPU rung when separated",
          gg::judgeVramMargin(site_b_margin_gib, true, false, amd) == V::WaiveSeparation);
    check("separate: the same margin still refuses when NOT separated",
          gg::judgeVramMargin(site_b_margin_gib, false, false, amd) == V::Refuse);
    check("separate: below the hard floor with no host history -> refuse",
          gg::judgeVramMargin(0.10, true, false, amd) == V::Refuse);
    check("separate: below the hard floor but this host carried it before "
          "-> waive on the measurement",
          gg::judgeVramMargin(0.10, true, true, amd) == V::WaiveHostRecord);
    check("separate: host history does NOT waive without the separation",
          gg::judgeVramMargin(0.10, false, true, amd) == V::Refuse);
    check("either ordering: a fat margin just passes",
          gg::judgeVramMargin(27.18, false, false, amd) == V::Pass);

    resetPolicy("auto", 4.0, 0.0, false);
    check("floor of 0 disables the test entirely",
          gg::judgeVramMargin(0.0, false, false, amd) == V::Pass);

    resetPolicy("auto", 4.0, 2.0, false);         // default thin-margin policy
    check("DEFAULT: a thin margin on a non-enforcing platform is ADVISORY, "
          "not a refusal",
          gg::judgeVramMargin(site_b_margin_gib, false, false, amd)
              == (gate_amd ? V::Refuse : V::Advisory));
    resetPolicy("auto", 4.0, 2.0, false, "refuse");
    check("explicit refuse: the platform gate still comes first",
          gg::judgeVramMargin(site_b_margin_gib, false, false, amd)
              == (gate_amd ? V::Refuse : V::Advisory));

    // ---- placement: shared / disjoint / auto ------------------------------
    const std::vector<int> four = {0, 1, 2, 3};

    resetPolicy("shared", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        auto out = gg::decidePlacement(four, 4, 0.1, amd, false);
        check("shared: never separates, even at 0.1 GiB of margin",
              out.size() == 4 && !gg::placementDecision().disjoint);
    }

    resetPolicy("disjoint", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        auto out = gg::decidePlacement(four, 4, 99.0, amd, false);
        const auto& d = gg::placementDecision();
        check("disjoint: separates on ANY platform and any margin",
              d.disjoint && out == std::vector<int>({0, 1})
                  && d.thermal_ids == std::vector<int>({2, 3}));
    }

    resetPolicy("auto", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        auto out = gg::decidePlacement(four, 4, 27.2, amd, false);
        check("auto: a fat margin stays shared",
              out.size() == 4 && !gg::placementDecision().disjoint);
    }

    resetPolicy("auto", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        auto out = gg::decidePlacement(four, 4, 0.64, amd, false);
        const auto& d = gg::placementDecision();
        check("auto: a thin margin separates IFF the driver is wedge-prone",
              d.disjoint == gate_amd
                  && (!gate_amd || (out == std::vector<int>({0, 1})
                                    && d.thermal_ids == std::vector<int>({2, 3}))));
    }

    resetPolicy("auto", 4.0, 2.0, false);
    setThermalIntent(false, 4, {});
    {
        auto out = gg::decidePlacement(four, 4, 0.1, amd, false);
        check("auto: nothing to separate when thermal is not on the GPU",
              out.size() == 4 && !gg::placementDecision().disjoint);
    }

    resetPolicy("auto", 4.0, 2.0, false);
    setThermalIntent(true, 0, {2, 3});
    {
        auto out = gg::decidePlacement({0, 1}, 4, 0.1, amd, true);
        check("auto: already-disjoint configuration is left alone",
              out == std::vector<int>({0, 1})
                  && !gg::placementDecision().disjoint);
    }

    resetPolicy("auto", 4.0, 2.0, false);
    setThermalIntent(true, 0, {0, 1, 2, 3});
    {
        auto out = gg::decidePlacement(four, 4, 0.1, amd, true);
        check("auto: REFUSES to overrule explicitly pinned ids on both sides",
              out.size() == 4 && !gg::placementDecision().disjoint);
    }

    resetPolicy("disjoint", 4.0, 2.0, false);
    setThermalIntent(true, 1, {});
    {
        auto out = gg::decidePlacement({0}, 1, 0.1, amd, false);
        check("one GPU cannot be split: stays shared and says so",
              out == std::vector<int>({0}) && !gg::placementDecision().disjoint);
    }

    resetPolicy("disjoint", 4.0, 2.0, false);
    setThermalIntent(true, 2, {});
    {
        auto out = gg::decidePlacement({0, 1}, 2, 0.1, amd, false);
        const auto& d = gg::placementDecision();
        check("two GPUs split 1/1",
              d.disjoint && out == std::vector<int>({0})
                  && d.thermal_ids == std::vector<int>({1}));
    }

    // ---- a decision the caller cannot honor must be undone ----------------
    resetPolicy("disjoint", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        (void)gg::decidePlacement(four, 4, 0.1, amd, false);
        check("precondition: the decision is disjoint before the revert",
              gg::placementDecision().disjoint);
        gg::revertPlacementToShared("test");
        const auto& d = gg::placementDecision();
        check("revertPlacementToShared clears BOTH the flag and the thermal ids",
              !d.disjoint && d.thermal_ids.empty()
                  && d.summary.find("reverted") != std::string::npos);
    }

    resetPolicy("disjoint", 4.0, 2.0, false);
    setThermalIntent(true, 4, {});
    {
        (void)gg::decidePlacement(four, 4, 0.1, amd, false);
        gg::advisePlacementUnavailable("single-GPU radiosity rung", 0.1, amd);
        check("a rung that cannot select its device does not inherit a stale "
              "disjoint decision",
              !gg::placementDecision().disjoint);
    }

    // ---- power-settings advisory ------------------------------------------
    {
        // Written to the CWD; the caller runs this from a scratch directory.
        const char* risky = "guardrails_test_pmset_risky.txt";
        const char* safe = "guardrails_test_pmset_safe.txt";
        FILE* f = std::fopen(risky, "w");
        std::fputs("AC Power:\n displaysleep         30\n sleep                60\n", f);
        std::fclose(f);
        f = std::fopen(safe, "w");
        std::fputs("AC Power:\n displaysleep         0\n sleep                0\n"
                   " disablesleep         1\n", f);
        std::fclose(f);

        setenv("PTM_GPU_POWER_PROBE_FILE", risky, 1);
        check("power advisory warns on displaysleep 30 / sleep 60",
              gg::powerSettingsAdvisory(true));
        check("power advisory is silent when no GPU is requested",
              !gg::powerSettingsAdvisory(false));
        setenv("PTM_GPU_POWER_PROBE_FILE", safe, 1);
        check("power advisory is silent when sleep is fully disabled",
              !gg::powerSettingsAdvisory(true));
        unsetenv("PTM_GPU_POWER_PROBE_FILE");
        std::remove(risky);
        std::remove(safe);
    }

    std::cout << (g_failures ? "\nGUARDRAILS LOGIC: FAIL\n"
                             : "\nGUARDRAILS LOGIC: PASS\n");
    return g_failures ? 1 : 0;
}
