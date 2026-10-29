// run_timing.h -- per-component wall / CPU / peak-RSS timing for the engine.
//
// Moon3D R1 change C2: Reviewer 1 asked for measured cost curves per component
// (illumination, view factors, radiosity, thermal). Whole-run wall clock cannot
// separate them, so the engine records them itself and writes one JSON file:
//
//     PTM_TIMING_JSON=<path>  ptm_thermal_sim --config ...
//
// INERT unless PTM_TIMING_JSON is set: every entry point returns after one
// branch, nothing is printed to stdout, and no output file is touched. Timing
// must never move a result; the R1 gate runs the same config with the variable
// set and unset and requires byte-identical outputs.
//
// Model: named INTERVALS (accumulated; begin/end pairs or RAII Scope), named
// COUNTERS, a per-coupling-iteration record list, and MARKS (a timeline of
// wall/CPU/peak-RSS snapshots). Peak RSS is ru_maxrss at the moment of the mark
// or the end of the interval, i.e. the process high-water mark so far: the
// phase that raises it is the phase whose end value first exceeds the previous.
#pragma once

#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ptm {
namespace timing {

struct Snapshot {
    double wall = 0.0;     // s since the registry was created (process start, in practice)
    double cpu = 0.0;      // user + system CPU s of the whole process
    long long rss = 0;     // ru_maxrss in BYTES (macOS reports bytes, Linux KiB)
};

struct Interval {
    double wall = 0.0, cpu = 0.0;
    long long rss_end = 0;
    long calls = 0;
};

struct Registry {
    bool enabled = false;
    std::string path;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::mutex mu;
    std::map<std::string, Interval> intervals;
    std::vector<std::string> interval_order;
    std::map<std::string, double> counters;
    std::vector<std::pair<std::string, Snapshot>> marks;
    std::vector<std::map<std::string, double>> iterations;
    std::map<std::string, Snapshot> open;   // begin() without end() yet
    bool written = false;

    Registry() {
        if (const char* p = std::getenv("PTM_TIMING_JSON")) {
            if (p[0] != '\0') { enabled = true; path = p; }
        }
    }
};

inline Registry& reg() {
    static Registry r;
    return r;
}

inline bool enabled() { return reg().enabled; }

inline Snapshot now() {
    Snapshot s;
    s.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - reg().t0).count();
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        s.cpu = ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec
              + ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
#ifdef __APPLE__
        s.rss = static_cast<long long>(ru.ru_maxrss);
#else
        s.rss = static_cast<long long>(ru.ru_maxrss) * 1024LL;
#endif
    }
    return s;
}

inline void mark(const std::string& name) {
    if (!enabled()) return;
    Snapshot s = now();
    std::lock_guard<std::mutex> g(reg().mu);
    reg().marks.emplace_back(name, s);
}

inline void begin(const std::string& name) {
    if (!enabled()) return;
    Snapshot s = now();
    std::lock_guard<std::mutex> g(reg().mu);
    reg().open[name] = s;
}

inline void end(const std::string& name) {
    if (!enabled()) return;
    Snapshot s = now();
    std::lock_guard<std::mutex> g(reg().mu);
    auto& r = reg();
    auto it = r.open.find(name);
    if (it == r.open.end()) return;   // end() without begin(): recorded as nothing
    auto ins = r.intervals.emplace(name, Interval{});
    if (ins.second) r.interval_order.push_back(name);
    Interval& iv = ins.first->second;
    iv.wall += s.wall - it->second.wall;
    iv.cpu += s.cpu - it->second.cpu;
    iv.rss_end = s.rss;
    ++iv.calls;
    r.open.erase(it);
}

// RAII interval for scopes with several exits. The calls MUST be qualified:
// unqualified begin(name)/end(name) on a std::string resolve by ADL to
// std::begin/std::end (a better match for a non-const lvalue), and the scope
// then silently records nothing -- measured, not hypothetical.
struct Scope {
    std::string name;
    explicit Scope(std::string n) : name(std::move(n)) { ptm::timing::begin(name); }
    ~Scope() { ptm::timing::end(name); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

inline void counter(const std::string& name, double value) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> g(reg().mu);
    reg().counters[name] = value;
}

inline void add(const std::string& name, double value) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> g(reg().mu);
    reg().counters[name] += value;
}

inline void iteration(const std::map<std::string, double>& rec) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> g(reg().mu);
    reg().iterations.push_back(rec);
}

namespace detail {
inline std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { o += ' '; }
        else o += c;
    }
    return o;
}
inline std::string num(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.9g", v);
    return b;
}
}  // namespace detail

// Write the JSON. Called once at the end of a successful run; `extra` carries
// run metadata (strings) supplied by the caller. Idempotent.
inline void write(const std::map<std::string, std::string>& extra = {}) {
    if (!enabled()) return;
    Snapshot fin = now();
    std::lock_guard<std::mutex> g(reg().mu);
    auto& r = reg();
    if (r.written) return;
    r.written = true;
    using detail::esc;
    using detail::num;
    std::ofstream f(r.path);
    if (!f) {
        std::fprintf(stderr, "[timing] WARNING: cannot write PTM_TIMING_JSON=%s\n", r.path.c_str());
        return;
    }
    f << "{\n  \"schema\": \"ptm.run_timing/1\",\n";
    for (const auto& kv : extra) f << "  \"" << esc(kv.first) << "\": \"" << esc(kv.second) << "\",\n";
    f << "  \"total\": {\"wall_s\": " << num(fin.wall) << ", \"cpu_s\": " << num(fin.cpu)
      << ", \"peak_rss_bytes\": " << fin.rss << "},\n";
    f << "  \"intervals\": {";
    bool first = true;
    for (const auto& name : r.interval_order) {
        const Interval& iv = r.intervals[name];
        f << (first ? "\n" : ",\n") << "    \"" << esc(name) << "\": {\"wall_s\": " << num(iv.wall)
          << ", \"cpu_s\": " << num(iv.cpu) << ", \"calls\": " << iv.calls
          << ", \"peak_rss_bytes_at_end\": " << iv.rss_end << "}";
        first = false;
    }
    f << "\n  },\n  \"unclosed_intervals\": [";
    first = true;
    for (const auto& kv : r.open) { f << (first ? "" : ", ") << "\"" << esc(kv.first) << "\""; first = false; }
    f << "],\n  \"counters\": {";
    first = true;
    for (const auto& kv : r.counters) {
        f << (first ? "\n" : ",\n") << "    \"" << esc(kv.first) << "\": " << num(kv.second);
        first = false;
    }
    f << "\n  },\n  \"coupling_iterations\": [";
    first = true;
    for (const auto& rec : r.iterations) {
        f << (first ? "\n    {" : ",\n    {");
        bool f2 = true;
        for (const auto& kv : rec) {
            f << (f2 ? "" : ", ") << "\"" << esc(kv.first) << "\": " << num(kv.second);
            f2 = false;
        }
        f << "}";
        first = false;
    }
    f << "\n  ],\n  \"marks\": [";
    first = true;
    for (const auto& m : r.marks) {
        f << (first ? "\n" : ",\n") << "    {\"name\": \"" << esc(m.first) << "\", \"wall_s\": "
          << num(m.second.wall) << ", \"cpu_s\": " << num(m.second.cpu)
          << ", \"peak_rss_bytes\": " << m.second.rss << "}";
        first = false;
    }
    f << "\n  ]\n}\n";
    std::fprintf(stderr, "[timing] wrote %s\n", r.path.c_str());
}

}  // namespace timing
}  // namespace ptm
