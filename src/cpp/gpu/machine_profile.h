#pragma once

/**
 * @file machine_profile.h
 * @brief Per-host GPU dispatch profile (Phase R robustness, R1).
 *
 * A small, dependency-free JSON record written by `ptm_thermal_sim
 * --calibrate-gpu` and consumed by (a) the in-solver dispatch self-chunking
 * (R2, thermal_opencl.cpp) and (b) the wedge-safe runners (R3, python, which
 * parse the same file). The record captures the machine-specific limits that
 * the 07-16→07-20 crash taxonomy showed are the difference between a healthy
 * run and a macOS+AMD gpusKillClientExt command-buffer watchdog kill:
 *
 *   - safe_dispatch_ceiling_s : max wall-clock a SINGLE clEnqueueNDRangeKernel
 *                               may take before the driver watchdog risk.
 *                               See kSafeDispatchCeilingSeconds below — the
 *                               older "~15-30 s launches survive" reading of
 *                               memory/gpu_hang_amd_macos.md was WRONG and cost
 *                               a run on 2026-08-16.
 *   - max_timesteps_per_dispatch : the derived ceiling the solver actually
 *                               enforces — the persistent kernel loops this
 *                               many timesteps per launch at most. Derived from
 *                               the measured per-timestep kernel cost so that
 *                               one launch stays under safe_dispatch_ceiling_s.
 *                               (This is the direct wedge dimension: the
 *                               yearlong wedge was 1560 timesteps in ONE
 *                               unbroken dispatch vs the proven-safe ~240.)
 *   - kernel_ms_per_step_per_batch, overhead_per_batch_s : the two terms of
 *                               the FIXED cost model ported from
 *                               scripts/run_yearlong_wedge_safe.py mode
 *                               calibrate (n_batches x per_batch + overhead).
 *   - recommended_batch_size : the proven-safe gpu_batch_size for this host.
 *
 * DESIGN RULE (Phase R family): "compute what you're about to do, check it's
 * sane, refuse loudly if not." When no profile exists the consumers fall back
 * to conservativeDefault() and WARN — they never silently assume this
 * machine's 384 GB / W6800X limits on a colleague's laptop.
 *
 * The file format is a flat JSON object of scalars; it is written by this
 * header and by scripts/ptm_machine_profile.py, and both parse it. Kept flat
 * and hand-serialized on purpose: no yaml-cpp / nlohmann dependency is pulled
 * into the GPU translation unit for one tiny record.
 */

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

namespace gpu {

/// Wall-clock budget for ONE clEnqueueNDRangeKernel of pure stepping, in
/// seconds. Everything else in this header (and the calibrator) derives the
/// enforced timestep ceiling from this number and the measured per-step cost.
///
/// 2.0 s, not the 25 s this file shipped with through 2026-08-16. The 25 s came
/// from reading memory/gpu_hang_amd_macos.md as "~15-30 s launches survive
/// repeatedly"; the surviving-run record on this host (macOS + AMD Radeon PRO
/// W6800X Duo) says the driver's real patience is ~2-3 s:
///
///   - 2026-08-16: `examples/europa_high_res.yaml` (ico7, 327,680 faces) was
///     auto-calibrated to 2986 timesteps/dispatch under the 25 s target and
///     dispatched all 600 equilibration steps in one ~5.0 s launch. The driver
///     killed the client (`gpusKillClientExt`, SIGABRT). A ~5 s launch is not
///     survivable here.
///   - Every long production run that DID survive kept dispatches under ~2.5 s:
///     the standing mitigation is `PTM_MAX_TIMESTEPS_PER_DISPATCH=50`, which is
///     ~2.5 s at 509k faces, and the 2026-08-14 MS3 lunar run's single
///     dispatches measured 2.0-2.9 s.
///   - The historical wedge trio (2026-07-16..20) were all multi-thousand-step
///     single dispatches (1560-8000 steps), i.e. tens of seconds.
///
/// This is a HOST-driver property, not a physics knob: the self-chunk split is
/// numerically inert (same CN updates, same order — see thermal_opencl.cpp
/// enqueuePersistentSelfChunked), and the measured cost of an extra chunk
/// boundary on this host is ~0.24 ms (overhead_per_batch_s), so a 2 s budget
/// costs ~0.012% wall time versus 25 s.
///
/// Escape hatch for a host with a more patient driver: set
/// PTM_MAX_TIMESTEPS_PER_DISPATCH (honored ahead of everything here), or
/// `--calibrate-gpu --safe-ceiling-s <s>` for the hand-run profile path.
///
/// TODO(PTM-276): source this per-host from the machine profile / a probed
/// watchdog measurement rather than a compile-time constant.
constexpr double kSafeDispatchCeilingSeconds = 2.0;

struct MachineProfile {
    int         schema_version          = 2;
    bool        valid                   = false;   // true only if loaded from a real file
    std::string source                  = "conservative-default";
    std::string host                    = "";
    std::string device_name             = "";
    std::string created_utc             = "";

    // The enforced limits.
    double      safe_dispatch_ceiling_s = kSafeDispatchCeilingSeconds;  // per launch
    int         max_timesteps_per_dispatch = 50;   // the R2 self-chunking ceiling
    int         recommended_batch_size  = 8000;

    // The measured cost-model terms (informational; used to (re)derive the
    // ceiling and by the runner's launch-time consistency check).
    double      kernel_ms_per_step_per_batch = 0.0;
    double      overhead_per_batch_s    = 0.0;
    int         n_layers_probe          = 0;
    int         batch_size_probe        = 0;

    // Representative-grid provenance (schema v2). Written by the in-run
    // auto-calibrate path so a later run can decide whether a stored profile
    // matches its production grid before reusing its dispatch ceiling instead
    // of re-probing. The per-dispatch kernel cost is driven by n_layers and the
    // von-Neumann substep count (substeps ∝ dt·kappa/dz_min²), so those are the
    // grid fields we store and compare. Zero/false on schema-1 profiles ⇒ the
    // grid is "unknown" ⇒ a mismatch ⇒ a safe re-calibrate (never a silent
    // reuse of a ceiling measured for a different grid).
    double      dz_min_probe            = 0.0;   // finest near-surface layer thickness [m]
    double      dt_output_probe_s       = 0.0;   // worst-case dt the ceiling was bound on [s]
    double      kappa_probe             = 0.0;   // near-surface diffusivity kc/(rho·cp0) [m²/s]
    int         is_run_grid             = 0;     // 1 = calibrated on a representative run grid

    /// A deliberately conservative profile for hosts with no calibration.
    /// 50 timesteps/dispatch is the standing production mitigation on this host
    /// (`PTM_MAX_TIMESTEPS_PER_DISPATCH=50`, ~2.5 s at 509k faces) — the largest
    /// uncalibrated step count with a clean survival record here. It replaces
    /// the old 300, which was only ever safe when the per-step cost was small:
    /// at the 8.4 ms/step measured on Europa ico7 (327,680 faces) a 300-step
    /// dispatch is 2.5 s, and at 509k lunar faces (~50 ms/step) it is 15 s —
    /// i.e. the old "conservative" default was itself above the watchdog.
    /// This value also bounds the auto-calibrate PROBE's own launches, which
    /// run before any per-step cost is known.
    static MachineProfile conservativeDefault() {
        MachineProfile p;
        p.valid = false;
        p.source = "conservative-default";
        return p;
    }
};

namespace machine_profile_detail {

/// Resolve the profile path: $PTM_MACHINE_PROFILE if set, else
/// $HOME/.ptm3d/machine_profile.json. Empty string if neither is resolvable.
inline std::string defaultProfilePath() {
    if (const char* env = std::getenv("PTM_MACHINE_PROFILE")) {
        if (env[0] != '\0') return std::string(env);
    }
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') return "";
    return std::string(home) + "/.ptm3d/machine_profile.json";
}

/// Extremely small tolerant reader for our own flat JSON: finds "key" then the
/// next ':' then reads the token up to the next ',' or '}'. Handles quoted
/// strings and bare numbers/booleans. Returns false if key absent.
inline bool jsonFind(const std::string& text, const std::string& key,
                     std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t k = text.find(needle);
    if (k == std::string::npos) return false;
    size_t colon = text.find(':', k + needle.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                               text[i] == '\n' || text[i] == '\r')) ++i;
    if (i >= text.size()) return false;
    if (text[i] == '"') {
        size_t end = text.find('"', i + 1);
        if (end == std::string::npos) return false;
        out = text.substr(i + 1, end - i - 1);
        return true;
    }
    size_t end = i;
    while (end < text.size() && text[end] != ',' && text[end] != '}' &&
           text[end] != '\n' && text[end] != '\r') ++end;
    out = text.substr(i, end - i);
    // trim trailing whitespace
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return true;
}

inline bool getDouble(const std::string& t, const std::string& k, double& v) {
    std::string s; if (!jsonFind(t, k, s)) return false;
    try { v = std::stod(s); return true; } catch (...) { return false; }
}
inline bool getInt(const std::string& t, const std::string& k, int& v) {
    std::string s; if (!jsonFind(t, k, s)) return false;
    try { v = static_cast<int>(std::stod(s)); return true; } catch (...) { return false; }
}
inline bool getStr(const std::string& t, const std::string& k, std::string& v) {
    return jsonFind(t, k, v);
}

inline bool getSize(const std::string& t, const std::string& k, size_t& v) {
    std::string s; if (!jsonFind(t, k, s)) return false;
    try { v = static_cast<size_t>(std::stoull(s)); return true; } catch (...) { return false; }
}
inline bool getLL(const std::string& t, const std::string& k, long long& v) {
    std::string s; if (!jsonFind(t, k, s)) return false;
    try { v = std::stoll(s); return true; } catch (...) { return false; }
}

/// mkdir -p the parent directory of `full`. Walks the components so a nested
/// profile path works; EEXIST is ignored.
inline void mkdirParents(const std::string& full) {
    size_t slash = full.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;
    const std::string dir = full.substr(0, slash);
    const bool absolute = (!dir.empty() && dir[0] == '/');
    std::string acc;
    std::stringstream ds(dir);
    std::string part;
    while (std::getline(ds, part, '/')) {
        if (part.empty()) { if (absolute && acc.empty()) acc = ""; continue; }
        acc += "/" + part;
        ::mkdir(acc.c_str(), 0755);   // ignore EEXIST
    }
}

inline std::string hostName() {
    char buf[256] = {0};
    if (::gethostname(buf, sizeof(buf) - 1) != 0) return "unknown-host";
    return std::string(buf);
}

inline std::string utcNow() {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

}  // namespace machine_profile_detail

/// Re-derive a loaded profile's dispatch ceiling against the CURRENT watchdog
/// budget (kSafeDispatchCeilingSeconds) and downgrade it if the stored ceiling
/// implies a launch longer than that budget.
///
/// Why this exists: profiles written before 2026-08-16 were calibrated against
/// a 25 s target and are on disk right now with ceilings up to 7523
/// timesteps/dispatch (~25 s launches). Bumping the constant alone would leave
/// every one of those caches live and the machine just as wedge-prone, because
/// the reuse paths in ensureGpuCalibrated take the stored ceiling verbatim
/// without re-probing. Re-deriving from the profile's own measured per-step
/// cost repairs them in place, with no re-probe (the measurement is still good;
/// only the budget it was divided into changed).
///
/// Only ever LOWERS the ceiling — a hand-written profile that asks for a
/// stricter budget is left alone. Never silent: any downgrade warns once per
/// path, per the Phase R fail-loud rule.
inline void enforceWatchdogCeiling(MachineProfile& mp, const std::string& path) {
    if (mp.max_timesteps_per_dispatch <= 0) return;
    const double budget_s = kSafeDispatchCeilingSeconds;
    const double per_step_s = mp.kernel_ms_per_step_per_batch / 1000.0;

    if (per_step_s > 0.0) {
        // Measured cost available: the honest test is the implied launch time.
        const double implied_s = mp.max_timesteps_per_dispatch * per_step_s;
        if (implied_s <= budget_s) return;              // already within budget
        long derived = static_cast<long>(budget_s / per_step_s);
        if (derived < 1) derived = 1;
        const int old_ts = mp.max_timesteps_per_dispatch;
        mp.max_timesteps_per_dispatch = static_cast<int>(derived);
        mp.safe_dispatch_ceiling_s    = budget_s;
        std::cerr << "[machine-profile] NOTE: profile at " << path
                  << " was calibrated against a " << implied_s
                  << " s per-dispatch launch (" << old_ts
                  << " timesteps at " << mp.kernel_ms_per_step_per_batch
                  << " ms/step), above this host's " << budget_s
                  << " s watchdog budget; re-derived ceiling to "
                  << mp.max_timesteps_per_dispatch
                  << " timesteps/dispatch from its own measured per-step cost "
                     "(no re-probe needed).\n";
        return;
    }

    // No measured per-step cost (schema-1 / hand-written profile): we cannot
    // re-derive, so fall back to the uncalibrated step ceiling if the stored
    // budget is laxer than ours.
    if (mp.safe_dispatch_ceiling_s > budget_s) {
        const int cons = MachineProfile::conservativeDefault().max_timesteps_per_dispatch;
        if (mp.max_timesteps_per_dispatch > cons) {
            const int old_ts = mp.max_timesteps_per_dispatch;
            mp.max_timesteps_per_dispatch = cons;
            std::cerr << "[machine-profile] WARNING: profile at " << path
                      << " declares a " << mp.safe_dispatch_ceiling_s
                      << " s dispatch budget (this host's is " << budget_s
                      << " s) and carries no measured per-step cost to re-derive "
                         "from; clamping its ceiling " << old_ts << " -> " << cons
                      << " timesteps/dispatch. Re-run `ptm_thermal_sim "
                         "--calibrate-gpu` to replace it.\n";
        }
        mp.safe_dispatch_ceiling_s = budget_s;
    }
}

/// Load the profile from `path` (or the resolved default path if empty).
/// Returns true and fills `out` (with out.valid=true) on success.
inline bool loadMachineProfile(MachineProfile& out, const std::string& path = "") {
    std::string p = path.empty() ? machine_profile_detail::defaultProfilePath() : path;
    if (p.empty()) return false;
    std::ifstream f(p);
    if (!f.good()) return false;
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();
    if (text.find('{') == std::string::npos) return false;

    MachineProfile mp;                       // start from conservative defaults
    using namespace machine_profile_detail;
    getInt(text, "schema_version", mp.schema_version);
    getStr(text, "source", mp.source);
    getStr(text, "host", mp.host);
    getStr(text, "device_name", mp.device_name);
    getStr(text, "created_utc", mp.created_utc);
    getDouble(text, "safe_dispatch_ceiling_s", mp.safe_dispatch_ceiling_s);
    getInt(text, "max_timesteps_per_dispatch", mp.max_timesteps_per_dispatch);
    getInt(text, "recommended_batch_size", mp.recommended_batch_size);
    getDouble(text, "kernel_ms_per_step_per_batch", mp.kernel_ms_per_step_per_batch);
    getDouble(text, "overhead_per_batch_s", mp.overhead_per_batch_s);
    getInt(text, "n_layers_probe", mp.n_layers_probe);
    getInt(text, "batch_size_probe", mp.batch_size_probe);
    // Schema v2 grid-provenance fields (absent on v1 profiles → left at 0/false).
    getDouble(text, "dz_min_probe", mp.dz_min_probe);
    getDouble(text, "dt_output_probe_s", mp.dt_output_probe_s);
    getDouble(text, "kappa_probe", mp.kappa_probe);
    getInt(text, "is_run_grid", mp.is_run_grid);

    // A profile with a non-positive ceiling is corrupt; refuse it loudly.
    if (mp.max_timesteps_per_dispatch <= 0 || mp.safe_dispatch_ceiling_s <= 0.0) {
        std::cerr << "[machine-profile] WARNING: profile at " << p
                  << " has a non-positive dispatch ceiling; ignoring it and "
                     "using conservative defaults.\n";
        return false;
    }
    // Stale-cache guard: a ceiling calibrated against an older, laxer
    // per-dispatch budget must not survive a load. See enforceWatchdogCeiling.
    enforceWatchdogCeiling(mp, p);

    mp.valid = true;
    if (mp.source == "conservative-default") mp.source = p;
    out = mp;
    return true;
}

/// Load-or-default. On miss, returns conservativeDefault() and WARNS once so
/// the fallback is never silent (Phase R design rule).
inline MachineProfile loadMachineProfileOrDefault(const std::string& path = "") {
    MachineProfile mp;
    if (loadMachineProfile(mp, path)) return mp;
    std::string p = path.empty() ? machine_profile_detail::defaultProfilePath() : path;
    std::cerr << "[machine-profile] no usable GPU profile at "
              << (p.empty() ? "<unresolved path>" : p)
              << "; using CONSERVATIVE defaults (max_timesteps_per_dispatch="
              << mp.max_timesteps_per_dispatch << ", ceiling="
              << mp.safe_dispatch_ceiling_s << " s). Run "
                 "`ptm_thermal_sim --calibrate-gpu` to profile this host.\n";
    return mp;
}

/// Serialize and write the profile (creates the parent directory). Returns
/// false and warns on any failure.
inline bool saveMachineProfile(const MachineProfile& p, const std::string& path = "") {
    std::string full = path.empty() ? machine_profile_detail::defaultProfilePath() : path;
    if (full.empty()) {
        std::cerr << "[machine-profile] cannot resolve a profile path "
                     "(set $PTM_MACHINE_PROFILE or $HOME).\n";
        return false;
    }
    machine_profile_detail::mkdirParents(full);
    std::ofstream f(full);
    if (!f.good()) {
        std::cerr << "[machine-profile] failed to open " << full << " for writing.\n";
        return false;
    }
    f.setf(std::ios::fixed);
    f << "{\n";
    f << "  \"schema_version\": " << p.schema_version << ",\n";
    f << "  \"source\": \"calibrate-gpu\",\n";
    f << "  \"host\": \"" << p.host << "\",\n";
    f << "  \"device_name\": \"" << p.device_name << "\",\n";
    f << "  \"created_utc\": \"" << p.created_utc << "\",\n";
    f << "  \"safe_dispatch_ceiling_s\": " << p.safe_dispatch_ceiling_s << ",\n";
    f << "  \"max_timesteps_per_dispatch\": " << p.max_timesteps_per_dispatch << ",\n";
    f << "  \"recommended_batch_size\": " << p.recommended_batch_size << ",\n";
    f << "  \"kernel_ms_per_step_per_batch\": " << p.kernel_ms_per_step_per_batch << ",\n";
    f << "  \"overhead_per_batch_s\": " << p.overhead_per_batch_s << ",\n";
    f << "  \"n_layers_probe\": " << p.n_layers_probe << ",\n";
    f << "  \"batch_size_probe\": " << p.batch_size_probe << ",\n";
    f << std::scientific;
    f << "  \"dz_min_probe\": " << p.dz_min_probe << ",\n";
    f << "  \"kappa_probe\": " << p.kappa_probe << ",\n";
    f << std::fixed;
    f << "  \"dt_output_probe_s\": " << p.dt_output_probe_s << ",\n";
    f << "  \"is_run_grid\": " << p.is_run_grid << "\n";
    f << "}\n";
    return f.good();
}

// ---------------------------------------------------------------------------
// Grid-keyed sidecar cache (auto-calibrate). Rather than overwrite one profile
// file on every grid change (which makes alternating bodies/resolutions re-probe
// every run), the auto path stores one small profile per representative grid in
// <profile-dir>/calib_cache/ and, on load, reuses the first cached profile whose
// grid matches the current run within tolerance.
// ---------------------------------------------------------------------------

/// True if a stored profile's grid matches the given run grid closely enough to
/// reuse its dispatch ceiling without re-probing. Rules:
///   - n_layers must match EXACTLY (it sets the Thomas-solve length linearly),
///   - stability ratio S = dt·kappa/dz_min² within ±25% (S drives the substep
///     count, which dominates per-dispatch cost; 25% is well inside the 4×
///     safety headroom yet catches the 2–100× coarse-vs-fine-grid errors),
///   - batch within a factor of 2 (per-step cost is ~flat above GPU saturation).
/// Fills `reason` with a human-readable mismatch description when it returns
/// false. A profile without v2 grid provenance never matches (safe re-calibrate).
inline bool gridMatchesProfile(int run_n_layers, double run_dz_min,
                               double run_dt_s, double run_kappa, int run_batch,
                               const MachineProfile& stored, std::string& reason) {
    if (stored.n_layers_probe != run_n_layers) {
        reason = "n_layers " + std::to_string(stored.n_layers_probe) + " != run " +
                 std::to_string(run_n_layers);
        return false;
    }
    if (stored.dz_min_probe <= 0.0 || stored.dt_output_probe_s <= 0.0 ||
        stored.kappa_probe <= 0.0 || stored.is_run_grid == 0) {
        reason = "stored profile has no representative-grid provenance";
        return false;
    }
    if (run_dz_min <= 0.0 || run_kappa <= 0.0 || run_dt_s <= 0.0) {
        reason = "run grid parameters non-positive";
        return false;
    }
    const double S_run = run_dt_s * run_kappa / (run_dz_min * run_dz_min);
    const double S_sto = stored.dt_output_probe_s * stored.kappa_probe /
                         (stored.dz_min_probe * stored.dz_min_probe);
    if (S_sto <= 0.0) { reason = "stored stability ratio non-positive"; return false; }
    const double rel = std::fabs(S_run - S_sto) / S_sto;
    if (rel > 0.25) {
        char b[96];
        std::snprintf(b, sizeof(b), "stability ratio S differs by %.0f%% (>25%%)",
                      rel * 100.0);
        reason = b;
        return false;
    }
    if (stored.batch_size_probe > 0 && run_batch > 0) {
        const double r = static_cast<double>(run_batch) /
                         static_cast<double>(stored.batch_size_probe);
        if (r > 2.0 || r < 0.5) {
            reason = "batch " + std::to_string(run_batch) + " differs >2× from stored " +
                     std::to_string(stored.batch_size_probe);
            return false;
        }
    }
    reason.clear();
    return true;
}

/// Directory holding the sidecar per-grid cache (beside the resolved profile).
inline std::string calibCacheDir() {
    std::string base = machine_profile_detail::defaultProfilePath();
    if (base.empty()) return "";
    size_t slash = base.find_last_of('/');
    std::string dir = (slash == std::string::npos) ? std::string(".")
                                                   : base.substr(0, slash);
    return dir + "/calib_cache";
}

/// Deterministic per-grid cache file path (the SAME grid re-calibrates into the
/// same file, so re-probing overwrites rather than accumulating stale entries).
inline std::string calibCacheFileForGrid(int n_layers, double dz_min,
                                         double dt_s, double kappa, int batch) {
    std::string dir = calibCacheDir();
    if (dir.empty()) return "";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "/profile_nl%d_dz%.3e_dt%.1f_k%.3e_b%d.json",
                  n_layers, dz_min, dt_s, kappa, batch);
    return dir + buf;
}

/// Scan the sidecar cache for a profile whose grid matches this run. Returns
/// true and fills `out` + `matched_path` on the first match.
inline bool loadMatchingCachedProfile(int run_n_layers, double run_dz_min,
                                      double run_dt_s, double run_kappa,
                                      int run_batch, MachineProfile& out,
                                      std::string& matched_path) {
    std::string dir = calibCacheDir();
    if (dir.empty()) return false;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return false;
    bool found = false;
    struct dirent* e = nullptr;
    while ((e = ::readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name.size() < 6 || name.substr(name.size() - 5) != ".json") continue;
        std::string full = dir + "/" + name;
        MachineProfile mp;
        if (!loadMachineProfile(mp, full)) continue;
        std::string reason;
        if (gridMatchesProfile(run_n_layers, run_dz_min, run_dt_s, run_kappa,
                               run_batch, mp, reason)) {
            out = mp;
            matched_path = full;
            found = true;
            break;
        }
    }
    ::closedir(d);
    return found;
}

// ---------------------------------------------------------------------------
// Per-host VRAM record (schema 1)
//
// The dispatch profile above answers "how long may one kernel launch be here".
// This one answers "how much device memory does this host have, and what is
// the largest radiosity CSR replica it has ever committed successfully". Both
// are host facts that a threshold should come from instead of a constant
// written for one 32 GiB Navi21 die.
//
// It lives in its OWN file (<profile dir>/vram_profile.json), not in
// machine_profile.json: that file is written by --calibrate-gpu and is keyed to
// a representative grid, and a read-modify-write from an unrelated code path is
// how a calibration record gets clobbered.
//
// Written after a SUCCESSFUL upload only. A refusal or a failed upload teaches
// nothing about what this host can carry, and recording an attempt as if it had
// succeeded would turn the file into a source of false confidence.
// ---------------------------------------------------------------------------

struct VramProfile {
    int schema_version = 1;
    bool valid = false;
    std::string host;
    std::string device_name;
    std::string vendor;
    std::string updated_utc;
    int n_devices = 0;                             ///< enumerated at last write
    size_t vram_total_bytes = 0;                   ///< smallest device seen
    size_t largest_committed_replica_bytes = 0;    ///< per device
    long long committed_margin_bytes = 0;          ///< margin left by that commit
    int committed_devices = 0;                     ///< devices that commit spanned
};

inline std::string vramProfilePath() {
    if (const char* env = std::getenv("PTM_VRAM_PROFILE")) {
        if (env[0] != '\0') return std::string(env);
    }
    std::string base = machine_profile_detail::defaultProfilePath();
    if (base.empty()) return "";
    size_t slash = base.find_last_of('/');
    std::string dir = (slash == std::string::npos) ? std::string(".")
                                                   : base.substr(0, slash);
    return dir + "/vram_profile.json";
}

inline bool loadVramProfile(VramProfile& out) {
    const std::string p = vramProfilePath();
    if (p.empty()) return false;
    std::ifstream f(p);
    if (!f.good()) return false;
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();
    if (text.find('{') == std::string::npos) return false;

    VramProfile v;
    using namespace machine_profile_detail;
    getInt(text, "schema_version", v.schema_version);
    getStr(text, "host", v.host);
    getStr(text, "device_name", v.device_name);
    getStr(text, "vendor", v.vendor);
    getStr(text, "updated_utc", v.updated_utc);
    getInt(text, "n_devices", v.n_devices);
    getSize(text, "vram_total_bytes", v.vram_total_bytes);
    getSize(text, "largest_committed_replica_bytes",
            v.largest_committed_replica_bytes);
    getLL(text, "committed_margin_bytes", v.committed_margin_bytes);
    getInt(text, "committed_devices", v.committed_devices);
    v.valid = true;
    out = v;
    return true;
}

inline bool saveVramProfile(const VramProfile& v) {
    const std::string p = vramProfilePath();
    if (p.empty()) return false;
    machine_profile_detail::mkdirParents(p);
    // Write-then-rename: this file is updated by EVERY GPU run, so two
    // concurrent runs can write it at once. A torn write would leave
    // unparsable JSON and silently discard the host's whole history; rename(2)
    // is atomic within a filesystem, so a reader sees either the old file or
    // the new one. The temp name carries the pid so the two writers do not
    // collide in the temp file either.
    const std::string tmp = p + ".tmp." + std::to_string(static_cast<long>(::getpid()));
    std::ofstream f(tmp);
    if (!f.good()) return false;
    f << "{\n";
    f << "  \"schema_version\": " << v.schema_version << ",\n";
    f << "  \"host\": \"" << v.host << "\",\n";
    f << "  \"device_name\": \"" << v.device_name << "\",\n";
    f << "  \"vendor\": \"" << v.vendor << "\",\n";
    f << "  \"updated_utc\": \"" << v.updated_utc << "\",\n";
    f << "  \"n_devices\": " << v.n_devices << ",\n";
    f << "  \"vram_total_bytes\": " << v.vram_total_bytes << ",\n";
    f << "  \"largest_committed_replica_bytes\": "
      << v.largest_committed_replica_bytes << ",\n";
    f << "  \"committed_margin_bytes\": " << v.committed_margin_bytes << ",\n";
    f << "  \"committed_devices\": " << v.committed_devices << "\n";
    f << "}\n";
    const bool ok = f.good();
    f.close();
    if (!ok || std::rename(tmp.c_str(), p.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

/// Record what this host's smallest participating device has. Cheap and
/// idempotent; only grows the record's knowledge (the device fields track the
/// SMALLEST device seen, because that is the one every fit check is bound by).
inline void noteVramDevice(size_t total_bytes, int n_devices,
                           const std::string& device_name,
                           const std::string& vendor) {
    if (total_bytes == 0) return;
    VramProfile v;
    loadVramProfile(v);
    if (v.vram_total_bytes == 0 || total_bytes < v.vram_total_bytes) {
        v.vram_total_bytes = total_bytes;
        v.device_name = device_name;
        v.vendor = vendor;
    }
    v.n_devices = std::max(v.n_devices, n_devices);
    v.host = machine_profile_detail::hostName();
    v.updated_utc = machine_profile_detail::utcNow();
    saveVramProfile(v);
}

/// Record a replica this host actually carried, with the margin it left.
/// Only ever raises the high-water mark. Call AFTER the upload succeeded.
inline void noteCommittedReplica(size_t bytes_per_device,
                                 long long margin_bytes,
                                 int n_devices) {
    if (bytes_per_device == 0) return;
    VramProfile v;
    loadVramProfile(v);
    if (bytes_per_device > v.largest_committed_replica_bytes) {
        v.largest_committed_replica_bytes = bytes_per_device;
        v.committed_margin_bytes = margin_bytes;
        v.committed_devices = n_devices;
    }
    v.host = machine_profile_detail::hostName();
    v.updated_utc = machine_profile_detail::utcNow();
    saveVramProfile(v);
}

}  // namespace gpu
