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
 *                               may take before the driver watchdog risk. From
 *                               memory/gpu_hang_amd_macos.md: ~15-30 s launches
 *                               survive repeatedly on this hardware.
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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>

namespace gpu {

struct MachineProfile {
    int         schema_version          = 2;
    bool        valid                   = false;   // true only if loaded from a real file
    std::string source                  = "conservative-default";
    std::string host                    = "";
    std::string device_name             = "";
    std::string created_utc             = "";

    // The enforced limits.
    double      safe_dispatch_ceiling_s = 25.0;    // per single kernel launch
    int         max_timesteps_per_dispatch = 300;  // the R2 self-chunking ceiling
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
    /// 300 timesteps/dispatch sits safely between the proven-safe ~240 and
    /// the yearlong wedge at 1560; 25 s ceiling matches the documented
    /// survivable-launch window.
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

}  // namespace machine_profile_detail

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
    // mkdir -p the parent directory (single-level create is enough for
    // ~/.ptm3d, but walk components to be safe).
    size_t slash = full.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        std::string dir = full.substr(0, slash);
        std::string acc;
        std::stringstream ds(dir);
        std::string part;
        bool absolute = (!dir.empty() && dir[0] == '/');
        while (std::getline(ds, part, '/')) {
            if (part.empty()) { if (absolute && acc.empty()) acc = ""; continue; }
            acc += "/" + part;
            ::mkdir(acc.c_str(), 0755);   // ignore EEXIST
        }
    }
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

}  // namespace gpu
