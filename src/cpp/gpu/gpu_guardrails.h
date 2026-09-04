#pragma once

/**
 * @file gpu_guardrails.h
 * @brief Cross-cutting GPU safety policy: VRAM margin, pool placement, host
 *        power settings.
 *
 * These are the campaign-script guardrails from the 2026-08-20/21 GPU wedges
 * (SMC -128 and -108 on the A3GT gen-2 run) folded into the engine, so a CLI
 * user, a GUI user and a colleague's machine get them without copying a
 * driver script. See docs/gpu-wedge-guardrails.md for the full rationale,
 * the platform-gating table and the cost of each guardrail.
 *
 * PLATFORM GATING is the design rule here. The wedge is a macOS + AMD
 * (Navi21) driver failure; the guardrails that cost wall time or change a
 * solver rung ENFORCE only there and are ADVISORY elsewhere, so a colleague on
 * Linux + NVIDIA pays nothing and is still told what the engine noticed. An
 * explicit setting (config key present, or the environment variable set)
 * enforces everywhere -- an operator who asks for a policy gets the policy.
 *
 * Header-only and OpenCL-free on purpose: radiation_exchange.cpp, the main
 * program and a CPU-only build all include it, and none of them should pull
 * the OpenCL headers in to read a threshold.
 *
 * State lives in function-local statics (policy(), thermalIntent(),
 * placementDecision()). That is a deliberate, small, process-global handshake
 * between two subsystems that do not otherwise know about each other: the
 * radiosity solver owns the VRAM arithmetic, the main program owns both device
 * configurations, and neither can decide placement alone. It is written once
 * during configuration and read afterwards -- not a general-purpose global.
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gpu {
namespace guardrails {

// ---------------------------------------------------------------------------
// Platform gate
// ---------------------------------------------------------------------------

/// True when compiled for macOS. The pmset advisory and the AMD/macOS wedge
/// mitigations key on this.
inline bool hostIsMacOS() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

inline std::string toLower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
    return out;
}

inline bool vendorIsAmd(const std::string& vendor) {
    const std::string v = toLower(vendor);
    return v.find("amd") != std::string::npos
        || v.find("advanced micro devices") != std::string::npos
        || v.find("ati ") != std::string::npos
        || v == "ati";
}

/// True on the host/driver combination the wedge guardrails were measured on:
/// macOS with an AMD GPU. Everywhere else the enforcing guardrails stay inert
/// and only advise.
///
/// Test hook: PTM_GPU_FORCE_WEDGE_PLATFORM=1 forces "yes", =0 forces "no".
/// It announces itself the first time it is consulted, because a forced gate
/// that changed a run's solver rung without saying so would be exactly the
/// silent-substitution failure these guardrails exist to remove.
inline bool wedgeProneDriver(const std::string& vendor) {
    static int forced = -2;   // -2 = not yet read, -1 = unset, 0/1 = forced
    if (forced == -2) {
        forced = -1;
        if (const char* e = std::getenv("PTM_GPU_FORCE_WEDGE_PLATFORM")) {
            if (e[0] != '\0') {
                const std::string v = toLower(e);
                if (v == "1" || v == "true" || v == "yes") forced = 1;
                else if (v == "0" || v == "false" || v == "no") forced = 0;
                else {
                    std::cerr << "[gpu-guardrails] WARNING: "
                                 "PTM_GPU_FORCE_WEDGE_PLATFORM=\"" << e
                              << "\" is not 0/1; ignoring it and detecting the "
                                 "platform normally.\n";
                }
                if (forced >= 0) {
                    std::cout << "[gpu-guardrails] TEST HOOK: "
                                 "PTM_GPU_FORCE_WEDGE_PLATFORM=" << e
                              << " -- treating this host as "
                              << (forced == 1 ? "macOS+AMD (wedge-prone: "
                                                "guardrails ENFORCE)"
                                              : "not wedge-prone (guardrails "
                                                "ADVISORY only)")
                              << ".\n";
                }
            }
        }
    }
    if (forced >= 0) return forced == 1;
    return hostIsMacOS() && vendorIsAmd(vendor);
}

// ---------------------------------------------------------------------------
// Policy (populated from the YAML `gpu:` section by the main program)
// ---------------------------------------------------------------------------

struct Policy {
    /// Floor on the per-device VRAM headroom a resident radiosity CSR replica
    /// is allowed to leave behind, in GiB. Below it the GPU rung is refused
    /// (macOS+AMD, or when set explicitly) or flagged (elsewhere).
    double min_vram_margin_gib = 2.0;
    /// True when the run asked for a specific value (YAML key present, or
    /// PTM_GPU_MIN_VRAM_MARGIN_GIB set). An explicit ask enforces on every
    /// platform.
    bool min_vram_margin_explicit = false;

    /// "auto" | "shared" | "disjoint" -- see decidePlacement().
    std::string placement = "auto";
    /// Margin below which `auto` separates the radiation and thermal pools
    /// onto disjoint dies, in GiB.
    double placement_margin_gib = 4.0;

    /// What a margin below min_vram_margin_gib should DO. See judgeVramMargin().
    ///   "separate" -- keep the GPU rung but only behind disjoint placement,
    ///                 refusing only below hard_min_vram_margin_gib (DEFAULT)
    ///   "refuse"   -- give up the GPU rung (radiosity runs on the CPU)
    /// Default is "separate" (Paul's ruling, 2026-08-22): both orderings remove
    /// the co-residency the wedges died in, but on a large-matrix site the
    /// refusal costs a MEASURED +77.5 min/window (A3GT lunar polar site B: 108.9 min on
    /// the CPU radiosity rung against 31.4 on the GPU pool, 8 windows each)
    /// while the separation costs a derived +7-10 min. See
    /// docs/gpu-wedge-guardrails.md "Two ways to treat a thin margin".
    std::string thin_margin_policy = "separate";
    /// The floor below which even "separate" refuses, in GiB.
    double hard_min_vram_margin_gib = 0.25;
};

inline Policy& policy() {
    static Policy p;
    return p;
}

/// Resolve the effective VRAM-margin floor, applying the environment override
/// on top of the config. Memoized so the warning for a malformed value is
/// printed once, not once per device probed.
inline double minVramMarginGib() {
    static bool resolved = false;
    static double value = 0.0;
    if (resolved) return value;
    resolved = true;
    value = policy().min_vram_margin_gib;
    if (const char* env = std::getenv("PTM_GPU_MIN_VRAM_MARGIN_GIB")) {
        if (env[0] != '\0') {
            char* end = nullptr;
            const double gib = std::strtod(env, &end);
            if (end == env || !(gib >= 0.0)) {
                std::cerr << "[gpu-guardrails] WARNING: "
                             "PTM_GPU_MIN_VRAM_MARGIN_GIB=\"" << env
                          << "\" is not a non-negative number; ignoring it and "
                             "using gpu.min_vram_margin_gib=" << value << ".\n";
            } else {
                if (gib != value) {
                    std::cout << "[gpu-guardrails] PTM_GPU_MIN_VRAM_MARGIN_GIB "
                                 "overrides gpu.min_vram_margin_gib: " << value
                              << " -> " << gib << " GiB.\n";
                }
                value = gib;
                policy().min_vram_margin_explicit = true;
            }
        }
    }
    return value;
}

/// Whether a margin below the floor REFUSES the GPU rung (true) or only warns.
inline bool enforceVramMargin(const std::string& vendor) {
    return policy().min_vram_margin_explicit || wedgeProneDriver(vendor);
}

/// What to do about a given per-device margin.
enum class MarginVerdict {
    Pass,              ///< at or above the floor (or no floor set)
    Advisory,          ///< below the floor, but this platform only advises
    Refuse,            ///< below the floor: give up this GPU rung
    WaiveSeparation,   ///< below the floor but above the hard floor, and the
                       ///< radiosity replica and thermal pool are on separate
                       ///< dies, which is the mitigation the floor stands in for
    WaiveHostRecord    ///< below even the hard floor, separated, and THIS host
                       ///< has already carried a replica this large
};

/**
 * Decide what a thin per-device VRAM margin means for this rung.
 *
 * Two orderings, selected by gpu.thin_margin_policy:
 *
 *   "refuse"   (default, and what shipped on 2026-08-22) -- below
 *              gpu.min_vram_margin_gib the GPU rung is given up and radiosity
 *              falls to the CPU. Simple and safe, and on a large-matrix site
 *              very expensive: measured on A3GT lunar polar site B, the CPU radiosity
 *              rung cost 108.9 min/window against 31.4 min on the GPU pool
 *              (8 windows each, gen-1 vs gen-2) -- about +78 min/window.
 *
 *   "separate" -- the thin-margin risk is co-residency (both 2026-08 wedges
 *              died in a GPU thermal dispatch while the replica was resident
 *              on the same dies), and disjoint placement addresses exactly
 *              that at a derived +7-10 min/window. So: keep the GPU rung, but
 *              ONLY when the separation is actually in force, and refuse below
 *              a much lower hard floor. Below the hard floor there is one
 *              further escape, and it is a measurement rather than a guess:
 *              if this host's own VRAM record shows it has already committed a
 *              replica at least this large, the hardware has demonstrated it.
 *
 * The waiver is deliberately conditional on `separation_in_force`. A floor
 * waived because a mitigation "would" apply, on a rung where it could not be
 * applied (the single-GPU rung cannot choose its device; a one-GPU host cannot
 * split), would be a guardrail that reports itself as active while doing
 * nothing -- the exact failure this branch exists to remove.
 */
inline MarginVerdict judgeVramMargin(double margin_gib,
                                     bool separation_in_force,
                                     bool host_carried_before,
                                     const std::string& vendor) {
    const Policy& p = policy();
    if (!(p.min_vram_margin_gib > 0.0)) return MarginVerdict::Pass;
    if (margin_gib >= p.min_vram_margin_gib) return MarginVerdict::Pass;
    if (!enforceVramMargin(vendor)) return MarginVerdict::Advisory;
    if (toLower(p.thin_margin_policy) != "separate") return MarginVerdict::Refuse;
    if (!separation_in_force) return MarginVerdict::Refuse;
    if (margin_gib >= p.hard_min_vram_margin_gib) {
        return MarginVerdict::WaiveSeparation;
    }
    return host_carried_before ? MarginVerdict::WaiveHostRecord
                               : MarginVerdict::Refuse;
}

// ---------------------------------------------------------------------------
// Placement: which dies the radiation and thermal pools may share
// ---------------------------------------------------------------------------

/// What the thermal solver intends to do, published by the main program before
/// the radiosity solver initializes. The radiosity solver is the only place
/// that knows the replica's per-device footprint; the main program is the only
/// place that knows both device configurations. This is the handshake.
struct ThermalIntent {
    bool published = false;
    bool use_gpu = false;
    int devices = 1;               ///< 0 = all enumerated
    std::vector<int> device_ids;   ///< non-empty = explicitly pinned
};

inline ThermalIntent& thermalIntent() {
    static ThermalIntent t;
    return t;
}

struct PlacementDecision {
    bool evaluated = false;
    bool disjoint = false;
    std::vector<int> radiation_ids;
    std::vector<int> thermal_ids;
    /// One line, printed on the ACTIVE RUNG line so the decision is on the
    /// record next to the rung it applies to.
    std::string summary = "shared";
};

inline PlacementDecision& placementDecision() {
    static PlacementDecision d;
    return d;
}

inline std::string idsToString(const std::vector<int>& ids) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) o << ",";
        o << ids[i];
    }
    o << "]";
    return o.str();
}

/// Resolve a (count, explicit ids) device request against the enumerated list.
inline std::vector<int> resolveDeviceIds(int devices,
                                         const std::vector<int>& device_ids,
                                         int n_enumerated) {
    std::vector<int> out;
    if (!device_ids.empty()) {
        for (int id : device_ids) {
            if (id >= 0 && id < n_enumerated) out.push_back(id);
        }
        return out;
    }
    const int n_req = (devices == 0) ? n_enumerated
                                     : std::min(devices, n_enumerated);
    for (int i = 0; i < n_req; ++i) out.push_back(i);
    return out;
}

/**
 * Decide whether the radiosity replica and the GPU thermal pool may live on
 * the same dies, and return the device list the radiosity pool should use.
 *
 * The rule (gpu.placement_policy):
 *   shared    -- never split. The behavior every release before this one had.
 *   disjoint  -- always split when both pools want the GPU and there are >= 2
 *                devices. Overrides explicitly pinned ids, loudly.
 *   auto      -- split only when ALL of: the thermal solver is on the GPU; the
 *                two device sets overlap; the post-fit margin is below
 *                gpu.placement_margin_gib; the driver is wedge-prone
 *                (macOS+AMD, or the test hook); and NEITHER side pinned
 *                explicit device ids. An explicit pin is an operator decision
 *                and `auto` will not overrule it -- it warns instead.
 *
 * Why it exists: both 2026-08 wedges died inside a GPU *thermal* dispatch
 * while a multi-GB radiosity CSR replica was resident on all four dies. That
 * co-residency is the one condition common to both events. Splitting the dies
 * removes it, and costs ~+7-10 min on a ~30 min A3GT window (derived from the
 * N=4 -> N=2 scaling of the radiosity sweep and the thermal phases; NOT
 * measured end to end -- see docs/gpu-wedge-guardrails.md).
 *
 * @param rad_ids        device ids the radiosity pool would use
 * @param n_enumerated   number of enumerated (available) GPUs
 * @param margin_gib     per-device VRAM headroom left after the replica
 * @param vendor         vendor string of the device the margin was computed on
 * @param rad_ids_explicit  true if radiation.gpu_device_ids was set
 * @return the device ids the radiosity pool should actually use
 */
inline std::vector<int> decidePlacement(const std::vector<int>& rad_ids,
                                        int n_enumerated,
                                        double margin_gib,
                                        const std::string& vendor,
                                        bool rad_ids_explicit) {
    PlacementDecision& d = placementDecision();
    d.evaluated = true;
    d.disjoint = false;
    d.radiation_ids = rad_ids;
    d.thermal_ids.clear();

    const Policy& p = policy();
    const std::string mode = toLower(p.placement);

    if (mode == "shared") {
        d.summary = "shared (gpu.placement_policy: shared)";
        return rad_ids;
    }
    const bool force = (mode == "disjoint");
    if (!force && mode != "auto") {
        // Unknown spelling: refuse to guess. Validated at config time too, so
        // this is belt and braces for a caller that skipped validation.
        std::cerr << "[gpu-guardrails] WARNING: gpu.placement_policy=\""
                  << p.placement << "\" is not auto|shared|disjoint; treating "
                     "it as auto.\n";
    }

    const ThermalIntent& t = thermalIntent();
    if (!t.published) {
        d.summary = "shared (thermal intent not published; nothing to separate)";
        return rad_ids;
    }
    if (!t.use_gpu) {
        d.summary = "shared (thermal solver is not on the GPU)";
        return rad_ids;
    }

    const std::vector<int> thermal_ids =
        resolveDeviceIds(t.devices, t.device_ids, n_enumerated);
    d.thermal_ids = thermal_ids;

    bool overlaps = false;
    for (int r : rad_ids) {
        if (std::find(thermal_ids.begin(), thermal_ids.end(), r)
            != thermal_ids.end()) {
            overlaps = true;
            break;
        }
    }
    if (!overlaps) {
        d.summary = "already disjoint by configuration (radiation "
                    + idsToString(rad_ids) + " / thermal "
                    + idsToString(thermal_ids) + ")";
        return rad_ids;
    }

    if (!force) {
        if (!wedgeProneDriver(vendor)) {
            std::ostringstream o;
            o << "shared (auto: co-residency separation is a macOS+AMD "
                 "guardrail; this device reports vendor \"" << vendor << "\")";
            d.summary = o.str();
            if (margin_gib < p.placement_margin_gib) {
                std::cout << "[gpu-guardrails] NOTE: the radiosity replica and "
                             "the GPU thermal pool will share dies with only "
                          << margin_gib << " GiB of VRAM margin (below the "
                          << p.placement_margin_gib
                          << " GiB gpu.placement_margin_gib threshold). On "
                             "macOS+AMD this run would be split onto disjoint "
                             "dies; here it is advisory only. Set "
                             "gpu.placement_policy: disjoint to force it.\n";
            }
            return rad_ids;
        }
        if (margin_gib >= p.placement_margin_gib) {
            std::ostringstream o;
            o << "shared (auto: margin " << margin_gib << " GiB >= "
              << p.placement_margin_gib << " GiB threshold)";
            d.summary = o.str();
            return rad_ids;
        }
        if (rad_ids_explicit || !t.device_ids.empty()) {
            std::cout << "[gpu-guardrails] WARNING: the radiosity replica "
                         "leaves only " << margin_gib
                      << " GiB of margin and the GPU thermal pool shares the "
                         "same dies -- the condition common to both 2026-08 "
                         "GPU wedges. gpu.placement_policy: auto will NOT "
                         "overrule explicitly pinned device ids (radiation "
                      << idsToString(rad_ids) << ", thermal "
                      << idsToString(thermal_ids)
                      << "). Set gpu.placement_policy: disjoint, or pin "
                         "non-overlapping ids yourself.\n";
            d.summary = "shared (auto declined: device ids pinned explicitly)";
            return rad_ids;
        }
    }

    if (n_enumerated < 2) {
        std::cout << "[gpu-guardrails] WARNING: gpu.placement_policy asks for "
                     "disjoint radiation/thermal dies but only " << n_enumerated
                  << " GPU is enumerated, so they must share it. Consider "
                     "radiation.use_gpu: false for this run.\n";
        d.summary = "shared (only one GPU available)";
        return rad_ids;
    }

    // Split. Radiation keeps the first half of what it asked for (at least one
    // device, never more than half the box) and the thermal pool gets every
    // enumerated device radiation did not keep. This reproduces the hand-built
    // G3 mitigation -- radiation [0,1] / thermal [2,3] on the 4-die W6800X Duo
    // -- for the case that motivated it, and degrades sensibly elsewhere.
    const int keep = std::max(1, std::min(static_cast<int>(rad_ids.size()),
                                          n_enumerated / 2));
    std::vector<int> rad_final(rad_ids.begin(), rad_ids.begin() + keep);
    std::vector<int> thermal_final;
    for (int i = 0; i < n_enumerated; ++i) {
        if (std::find(rad_final.begin(), rad_final.end(), i) == rad_final.end()) {
            thermal_final.push_back(i);
        }
    }

    d.disjoint = true;
    d.radiation_ids = rad_final;
    d.thermal_ids = thermal_final;
    std::ostringstream o;
    o << "DISJOINT (";
    if (force) {
        o << "gpu.placement_policy: disjoint";
    } else {
        o << "auto: margin " << margin_gib << " GiB < "
          << p.placement_margin_gib << " GiB threshold";
    }
    o << "): radiation " << idsToString(rad_final) << ", thermal "
      << idsToString(thermal_final);
    d.summary = o.str();

    std::cout << "[gpu-guardrails] GPU PLACEMENT: separating the radiosity "
                 "replica from the thermal pool -- radiation "
              << idsToString(rad_final) << ", thermal "
              << idsToString(thermal_final) << ". "
              << (force ? "Forced by gpu.placement_policy: disjoint."
                        : "Automatic: the post-fit VRAM margin is below "
                          "gpu.placement_margin_gib on a macOS+AMD host.")
              << " No die then holds the CSR replica and a thermal working set "
                 "at the same time -- the condition common to both 2026-08 GPU "
                 "wedges. Expect a wall-time cost: both pools lose devices "
                 "(derived, not measured: ~+7-10 min on a ~30 min A3GT "
                 "window). Set gpu.placement_policy: shared to opt out.\n";
    return rad_final;
}

/// Undo a placement decision that the caller cannot actually honor, and say
/// why. A decision that is announced but only half-applied is worse than no
/// decision at all: the main program would move the thermal pool off dies the
/// radiosity solver never left.
inline void revertPlacementToShared(const std::string& reason) {
    PlacementDecision& d = placementDecision();
    if (!d.disjoint) return;
    std::cout << "[gpu-guardrails] GPU PLACEMENT REVERTED to shared dies: "
              << reason << "\n";
    d.disjoint = false;
    d.thermal_ids.clear();
    d.summary = "shared (placement reverted: " + reason + ")";
}

/// For rungs that cannot select their own device (the single-GPU radiosity
/// solver takes whatever OpenCL hands it): say that the separation WOULD have
/// engaged, and how to get it, instead of silently sharing the die.
inline void advisePlacementUnavailable(const std::string& rung,
                                       double margin_gib,
                                       const std::string& vendor) {
    PlacementDecision& d = placementDecision();
    // A decision taken for a rung that then failed must not survive into the
    // rung that replaced it (pool -> single-GPU degrade).
    revertPlacementToShared("the " + rung + " replaced the rung the decision "
                            "was taken for");
    d.evaluated = true;
    d.disjoint = false;
    const Policy& p = policy();
    const std::string mode = toLower(p.placement);
    const ThermalIntent& t = thermalIntent();
    const bool would_engage =
        (mode == "disjoint")
        || (mode == "auto" && t.published && t.use_gpu
            && margin_gib < p.placement_margin_gib && wedgeProneDriver(vendor));
    if (!would_engage) {
        d.summary = "shared (" + rung + " selects its own device)";
        return;
    }
    std::cout << "[gpu-guardrails] WARNING: gpu.placement_policy asks to keep "
                 "the radiosity replica off the GPU thermal pool's dies, but "
                 "the " << rung << " does not choose its device (OpenCL does), "
                 "so the separation CANNOT be applied here. The replica leaves "
              << margin_gib << " GiB of margin. To get the separation, set "
                 "radiation.gpu_devices: 2 (the multi-GPU rung honors it) or "
                 "pin non-overlapping radiation.gpu_device_ids / "
                 "thermal.gpu_device_ids yourself.\n";
    d.summary = "shared (" + rung + " cannot select a device; separation NOT applied)";
}

// ---------------------------------------------------------------------------
// Host power settings (macOS)
// ---------------------------------------------------------------------------

/// Read `pmset -g custom` (unprivileged) and return its text. Test hook:
/// PTM_GPU_POWER_PROBE_FILE=<path> reads that file instead, so the advisory's
/// two branches can be exercised without touching the real power settings.
/// Returns false when neither source is readable.
inline bool readPowerSettings(std::string& out) {
    out.clear();
    if (const char* f = std::getenv("PTM_GPU_POWER_PROBE_FILE")) {
        if (f[0] != '\0') {
            std::FILE* fh = std::fopen(f, "r");
            if (!fh) {
                std::cerr << "[gpu-guardrails] WARNING: "
                             "PTM_GPU_POWER_PROBE_FILE=\"" << f
                          << "\" could not be opened; skipping the power-"
                             "settings advisory.\n";
                return false;
            }
            char buf[512];
            while (std::fgets(buf, sizeof(buf), fh)) out += buf;
            std::fclose(fh);
            std::cout << "[gpu-guardrails] TEST HOOK: power settings read from "
                      << f << " instead of pmset.\n";
            return true;
        }
    }
    if (!hostIsMacOS()) return false;
    std::FILE* pipe = ::popen("/usr/bin/pmset -g custom 2>/dev/null", "r");
    if (!pipe) return false;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
    const int rc = ::pclose(pipe);
    return rc == 0 && !out.empty();
}

/// Largest value of `key` across the blocks pmset printed, or `missing` when
/// the key does not appear (a laptop prints an AC and a Battery block; the
/// risky one is whichever sleeps soonest, i.e. the largest timeout is the one
/// that must be zero for the setting to be safe everywhere).
inline int powerSettingValue(const std::string& text, const std::string& key,
                             int missing) {
    int best = missing;
    size_t pos = 0;
    while ((pos = text.find(key, pos)) != std::string::npos) {
        const bool word_start =
            (pos == 0) || std::isspace(static_cast<unsigned char>(text[pos - 1]));
        size_t after = pos + key.size();
        pos = after;
        if (!word_start || after >= text.size()) continue;
        if (!std::isspace(static_cast<unsigned char>(text[after]))) continue;
        while (after < text.size()
               && std::isspace(static_cast<unsigned char>(text[after]))) ++after;
        size_t end = after;
        while (end < text.size()
               && std::isdigit(static_cast<unsigned char>(text[end]))) ++end;
        if (end == after) continue;
        const int v = std::atoi(text.substr(after, end - after).c_str());
        best = (best == missing) ? v : std::max(best, v);
    }
    return best;
}

/**
 * Warn when a GPU run is requested on macOS with sleep or display sleep still
 * enabled.
 *
 * Evidence, not superstition: wedge #1's LAST logged system event of any kind
 * was `Display is turned off` -- a WindowServer modeset on the Navi21
 * controller landing on a GPU whose compute side had been frozen for 36
 * minutes -- and a display modeset is the documented trigger of the 2026-06-23
 * SMC -128 wedge on this box. The mitigation
 * (`sudo pmset -a disablesleep 1 displaysleep 0`) had never been applied.
 *
 * Advisory only: it never changes a setting, it needs no privileges, and it is
 * a no-op off macOS. @return true if it warned.
 */
inline bool powerSettingsAdvisory(bool gpu_requested) {
    if (!gpu_requested) return false;
    std::string text;
    if (!readPowerSettings(text)) return false;

    const int displaysleep = powerSettingValue(text, "displaysleep", -1);
    const int sleep_min = powerSettingValue(text, "sleep", -1);
    const int disablesleep = powerSettingValue(text, "disablesleep", -1);

    const bool risky = (displaysleep > 0) || (sleep_min > 0);
    if (risky) {
        std::cout << "\n[gpu-guardrails] ******** WARNING: macOS power "
                     "settings are NOT safe for a long GPU run ********\n"
                     "  displaysleep = "
                  << (displaysleep < 0 ? std::string("unknown")
                                       : std::to_string(displaysleep))
                  << " min, sleep = "
                  << (sleep_min < 0 ? std::string("unknown")
                                    : std::to_string(sleep_min))
                  << " min. A display or system sleep transition issues a GPU "
                     "modeset, which is the documented trigger of the SMC -128 "
                     "wedge on this hardware; the last system event before the "
                     "2026-08-20 power-off was \"Display is turned off\".\n"
                     "  Fix (one command, needs sudo, survives reboot):\n"
                     "      sudo pmset -a disablesleep 1 displaysleep 0\n"
                     "  Revert after the campaign:\n"
                     "      sudo pmset -a disablesleep 0 displaysleep 30\n"
                     "  See the GPU checklist in INSTALL.md and "
                     "docs/gpu-wedge-guardrails.md.\n\n";
        return true;
    }
    if (disablesleep != 1) {
        std::cout << "[gpu-guardrails] NOTE: sleep and displaysleep are both "
                     "disabled, but `disablesleep` is not set. That is enough "
                     "for an idle machine; `sudo pmset -a disablesleep 1` also "
                     "blocks a sleep requested by software or a closed lid. "
                     "See INSTALL.md.\n";
    }
    return false;
}

}  // namespace guardrails
}  // namespace gpu
