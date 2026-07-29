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
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>

namespace gpu {

struct MachineProfile {
    int         schema_version          = 1;
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
    f << "  \"batch_size_probe\": " << p.batch_size_probe << "\n";
    f << "}\n";
    return f.good();
}

}  // namespace gpu
