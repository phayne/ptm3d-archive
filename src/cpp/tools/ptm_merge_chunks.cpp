// ptm_merge_chunks.cpp
//
// Merges per-chunk statistics from a chunked PTM3D run (produced by
// scripts/run_chunked_simulation.py) into a single set of final
// statistics, mathematically equivalent to a hypothetical single-shot
// run over the concatenated time window.
//
// Each input chunk dir is expected to contain a statistics/_raw/
// subdirectory with:
//   <datatype>_sum.bin     uint32 N + N float64
//   <datatype>_count.bin   uint32 N + N uint32
//   ice_depth_sum_E.bin    uint32 n_layers + uint32 n_faces + n_layers*n_faces float32
//   ice_depth_n_samples.txt   single integer line
//   depth_grid.bin         uint32 n_layers + n_layers float64
// plus the existing statistics/<datatype>_max.bin and _min.bin files.
//
// Output: a target dir <outdir>/statistics/<datatype>_{max,avg,min}.bin
// + ice_depth.bin (Schorghofer stability depth from the merged
// accumulator).
//
// CLI:
//   ptm_merge_chunks --chunks DIR1 DIR2 ... --output OUTDIR
//                    [--J0 3.0e-14] [--ell-um 75]

#include "../ice/ice_table.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---- I/O helpers ----------------------------------------------------------

bool read_uint32(std::ifstream& is, uint32_t& v) {
    is.read(reinterpret_cast<char*>(&v), sizeof(uint32_t));
    return is.good();
}

bool read_float32_array(const fs::path& p, uint32_t& N_out,
                        std::vector<float>& data) {
    std::ifstream is(p, std::ios::binary);
    if (!is) { std::cerr << "ERROR opening " << p << "\n"; return false; }
    if (!read_uint32(is, N_out)) { std::cerr << "ERROR reading N from " << p << "\n"; return false; }
    data.resize(N_out);
    is.read(reinterpret_cast<char*>(data.data()), N_out * sizeof(float));
    return is.good();
}

bool read_double_array(const fs::path& p, uint32_t& N_out,
                       std::vector<double>& data) {
    std::ifstream is(p, std::ios::binary);
    if (!is) { std::cerr << "ERROR opening " << p << "\n"; return false; }
    if (!read_uint32(is, N_out)) { std::cerr << "ERROR reading N from " << p << "\n"; return false; }
    data.resize(N_out);
    is.read(reinterpret_cast<char*>(data.data()), N_out * sizeof(double));
    return is.good();
}

bool read_uint32_array(const fs::path& p, uint32_t& N_out,
                       std::vector<uint32_t>& data) {
    std::ifstream is(p, std::ios::binary);
    if (!is) { std::cerr << "ERROR opening " << p << "\n"; return false; }
    if (!read_uint32(is, N_out)) { std::cerr << "ERROR reading N from " << p << "\n"; return false; }
    data.resize(N_out);
    is.read(reinterpret_cast<char*>(data.data()), N_out * sizeof(uint32_t));
    return is.good();
}

bool write_float32_array(const fs::path& p, const std::vector<float>& data) {
    std::ofstream os(p, std::ios::binary);
    if (!os) return false;
    uint32_t N = static_cast<uint32_t>(data.size());
    os.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(data.data()),
             data.size() * sizeof(float));
    return os.good();
}

// ---- Per-datatype merge ---------------------------------------------------

// Combine max/min/sum/count across N chunks for one datatype.
// Writes <outdir>/statistics/<name>_{max,avg,min}.bin.
//
// `strict` changes what a MISSING component means.
//   false (the five fixed datatypes): all four side files or nothing. A
//         thermal-only run has no illumination statistics at all, and skipping
//         that quietly is the documented behavior.
//   true  (subsurface depth maps): the components present are whatever
//         output.map_depth_stats asked for, so "avg is absent" is normal and
//         "avg is present in chunk 0 and absent in chunk 3" is a defect. Each
//         component is merged independently and any disagreement between chunks
//         is an ERROR, because the alternative is a merged product that quietly
//         drops a map the chunks all carried.
bool merge_one_datatype(const std::string& name,
                        const std::vector<fs::path>& chunk_dirs,
                        const fs::path& outdir,
                        uint32_t expected_N,
                        bool strict = false) {
    std::vector<float>   merged_max, merged_min;
    std::vector<double>  merged_sum;
    std::vector<uint64_t> merged_count;
    merged_max.assign(expected_N, std::numeric_limits<float>::lowest());
    merged_min.assign(expected_N, std::numeric_limits<float>::max());
    merged_sum.assign(expected_N, 0.0);
    merged_count.assign(expected_N, 0u);

    auto components = [&](const fs::path& cd, bool& has_avg, bool& has_max,
                          bool& has_min) {
        const fs::path stats = cd / "statistics";
        const fs::path raw   = stats / "_raw";
        has_max = fs::exists(stats / (name + "_max.bin"));
        has_min = fs::exists(stats / (name + "_min.bin"));
        has_avg = fs::exists(raw / (name + "_sum.bin")) &&
                  fs::exists(raw / (name + "_count.bin"));
    };

    bool want_avg = false, want_max = false, want_min = false;
    components(chunk_dirs.front(), want_avg, want_max, want_min);
    if (strict && !(want_avg || want_max || want_min)) {
        std::cerr << "[merge] ERROR: " << name << " has no mergeable component in "
                  << chunk_dirs.front() << "\n";
        return false;
    }
    // An `_avg.bin` MAP with no `_raw` sum/count behind it is a chunked run
    // that was not given `output.write_raw_stats: true`. Averages cannot be
    // merged from the per-chunk averages alone (the chunks need not carry equal
    // sample counts), so the honest answer is to refuse -- not to merge max and
    // min, drop the average, and report success.
    if (strict && !want_avg &&
        fs::exists(chunk_dirs.front() / "statistics" / (name + "_avg.bin"))) {
        std::cerr << "[merge] ERROR: " << name << " has a per-chunk _avg.bin but no "
                     "statistics/_raw/" << name << "_{sum,count}.bin to merge it "
                     "from. Re-run the chunks with output.write_raw_stats: true; an "
                     "average cannot be recovered from per-chunk averages alone.\n";
        return false;
    }

    bool any_loaded = false;
    for (const auto& cd : chunk_dirs) {
        const fs::path stats = cd / "statistics";
        const fs::path raw   = stats / "_raw";
        const fs::path p_max   = stats / (name + "_max.bin");
        const fs::path p_min   = stats / (name + "_min.bin");
        const fs::path p_sum   = raw   / (name + "_sum.bin");
        const fs::path p_count = raw   / (name + "_count.bin");

        bool has_avg, has_max, has_min;
        components(cd, has_avg, has_max, has_min);
        if (!strict) {
            // If any of the four files is missing in any chunk, the datatype
            // wasn't computed (e.g. illumination on a thermal-only run);
            // skip this datatype for the whole merge.
            if (!has_max || !has_min || !has_avg) {
                return true;  // not an error — just nothing to merge
            }
        } else if (has_avg != want_avg || has_max != want_max ||
                   has_min != want_min) {
            std::cerr << "[merge] ERROR: " << name << " carries a different set of "
                         "statistics in " << cd << " than in "
                      << chunk_dirs.front() << " (avg/max/min present: "
                      << has_avg << has_max << has_min << " vs "
                      << want_avg << want_max << want_min
                      << "). The chunks were not run with the same "
                         "output.map_depth_stats; refusing to merge.\n";
            return false;
        }

        uint32_t N;
        std::vector<float>   max_arr, min_arr;
        std::vector<double>  sum_arr;
        std::vector<uint32_t> count_arr;
        if (want_max && (!read_float32_array(p_max,  N, max_arr) || N != expected_N)) return false;
        if (want_min && (!read_float32_array(p_min,  N, min_arr) || N != expected_N)) return false;
        if (want_avg && (!read_double_array (p_sum,  N, sum_arr) || N != expected_N)) return false;
        if (want_avg && (!read_uint32_array (p_count,N, count_arr) || N != expected_N)) return false;

        for (uint32_t i = 0; i < expected_N; ++i) {
            if (want_max && max_arr[i] > merged_max[i]) merged_max[i] = max_arr[i];
            if (want_min && min_arr[i] < merged_min[i]) merged_min[i] = min_arr[i];
            if (want_avg) {
                merged_sum[i]   += sum_arr[i];
                merged_count[i] += count_arr[i];
            }
        }
        any_loaded = true;
    }

    if (!any_loaded) return true;

    std::vector<float> merged_avg(expected_N, 0.0f);
    for (uint32_t i = 0; i < expected_N; ++i) {
        if (merged_count[i] > 0) {
            merged_avg[i] = static_cast<float>(merged_sum[i] / static_cast<double>(merged_count[i]));
        } else {
            merged_avg[i] = std::numeric_limits<float>::quiet_NaN();
        }
        // Scrub sentinels for faces with no contribution at all.
        if (merged_max[i] == std::numeric_limits<float>::lowest())
            merged_max[i] = 0.0f;
        if (merged_min[i] == std::numeric_limits<float>::max())
            merged_min[i] = 0.0f;
    }

    fs::path stats_out = outdir / "statistics";
    fs::create_directories(stats_out);
    bool ok = true;
    if (want_max) ok &= write_float32_array(stats_out / (name + "_max.bin"), merged_max);
    if (want_avg) ok &= write_float32_array(stats_out / (name + "_avg.bin"), merged_avg);
    if (want_min) ok &= write_float32_array(stats_out / (name + "_min.bin"), merged_min);
    if (ok) {
        std::cout << "[merge] " << name << ": merged " << chunk_dirs.size()
                  << " chunks  (" << expected_N << " faces)\n";
    }
    return ok;
}

// ---- Subsurface depth-map discovery ---------------------------------------
//
// The depth field names encode the resolved depth (`temperature_z0p050m`), so
// they are a property of the RUN, not a compile-time list. Discover them from a
// chunk's own statistics directory rather than re-deriving them from a config
// the merge tool does not read.
std::vector<std::string> discover_depth_fields(const fs::path& chunk_dir) {
    std::vector<std::string> out;
    const fs::path stats = chunk_dir / "statistics";
    std::error_code ec;
    if (!fs::is_directory(stats, ec)) return out;
    const std::string prefix = "temperature_z";
    // ⛔ ALL THREE suffixes, de-duplicated -- not just `_max.bin`.
    // `output.map_depth_stats` is a free subset, so a run asked for
    // `[avg, min]` writes no `_max.bin` at all. Anchoring on one suffix made
    // discovery return nothing for that run, the whole depth product vanish
    // from the merged directory, and the tool still print success.
    for (const auto& entry : fs::directory_iterator(stats, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        for (const char* suffix : { "_max.bin", "_min.bin", "_avg.bin" }) {
            const std::string suf(suffix);
            if (name.size() <= prefix.size() + suf.size()) continue;
            if (name.compare(name.size() - suf.size(), suf.size(), suf) != 0)
                continue;
            std::string field = name.substr(0, name.size() - suf.size());
            if (std::find(out.begin(), out.end(), field) == out.end())
                out.push_back(field);
            break;
        }
    }
    return out;
}

// ---- Ice-stability merge --------------------------------------------------

bool merge_ice_stability(const std::vector<fs::path>& chunk_dirs,
                         const fs::path& outdir,
                         uint32_t expected_N,
                         double J0, double ell_m) {
    // Probe chunk 0 for n_layers + depth grid.
    const fs::path raw0 = chunk_dirs.front() / "statistics" / "_raw";
    const fs::path p_grid0    = raw0 / "depth_grid.bin";
    const fs::path p_sumE0    = raw0 / "ice_depth_sum_E.bin";
    const fs::path p_n0       = raw0 / "ice_depth_n_samples.txt";
    if (!fs::exists(p_sumE0) || !fs::exists(p_grid0) || !fs::exists(p_n0)) {
        std::cout << "[merge] ice_depth: raw accumulator missing in chunk[0]; skipping.\n";
        return true;
    }

    uint32_t n_layers = 0;
    std::vector<double> z_grid;
    {
        uint32_t N;
        if (!read_double_array(p_grid0, N, z_grid)) return false;
        n_layers = N;
    }

    ice::IceStabilityAccumulator acc;
    acc.allocate(static_cast<int>(expected_N), static_cast<int>(n_layers));
    for (int i = 0; i < (int)n_layers; ++i) acc.z[i] = z_grid[i];
    long long n_samples_total = 0;

    for (const auto& cd : chunk_dirs) {
        const fs::path raw = cd / "statistics" / "_raw";
        const fs::path p_sumE = raw / "ice_depth_sum_E.bin";
        const fs::path p_n    = raw / "ice_depth_n_samples.txt";
        const fs::path p_grid = raw / "depth_grid.bin";
        if (!fs::exists(p_sumE) || !fs::exists(p_n) || !fs::exists(p_grid)) {
            std::cerr << "[merge] ERROR: missing ice_depth raw files in " << cd << "\n";
            return false;
        }

        // Verify depth grid matches.
        std::vector<double> z_chunk;
        uint32_t Nl;
        if (!read_double_array(p_grid, Nl, z_chunk)) return false;
        if (Nl != n_layers || z_chunk != z_grid) {
            std::cerr << "[merge] ERROR: depth-grid mismatch between chunks (chunk "
                      << cd << ")\n";
            return false;
        }

        // Read sum_E for this chunk.
        std::ifstream is(p_sumE, std::ios::binary);
        if (!is) { std::cerr << "ERROR opening " << p_sumE << "\n"; return false; }
        uint32_t nl_h, nf_h;
        if (!read_uint32(is, nl_h) || !read_uint32(is, nf_h)) return false;
        if (nl_h != n_layers || nf_h != expected_N) {
            std::cerr << "[merge] ERROR: ice_depth sum_E shape mismatch in " << cd
                      << " (got " << nl_h << "x" << nf_h << ", expected "
                      << n_layers << "x" << expected_N << ")\n";
            return false;
        }
        const size_t total = static_cast<size_t>(nl_h) * nf_h;
        std::vector<float> sumE_chunk(total);
        is.read(reinterpret_cast<char*>(sumE_chunk.data()), total * sizeof(float));
        if (!is) { std::cerr << "ERROR reading sum_E from " << p_sumE << "\n"; return false; }

        for (size_t i = 0; i < total; ++i) acc.sum_E[i] += sumE_chunk[i];

        // Read n_samples.
        long long n;
        std::ifstream is2(p_n);
        if (!is2 || !(is2 >> n)) { std::cerr << "ERROR reading " << p_n << "\n"; return false; }
        n_samples_total += n;
    }

    acc.n_samples = static_cast<int>(n_samples_total);
    std::cout << "[merge] ice_depth: total samples = " << n_samples_total
              << " across " << chunk_dirs.size() << " chunks; solving Schorghofer J(z)=J0...\n";

    std::vector<float> ice_depth = ice::computeIceStabilityDepths(
        acc, J0, ell_m, /*z_max_m*/ -1.0, /*use_openmp*/ true);

    fs::path stats_out = outdir / "statistics";
    fs::create_directories(stats_out);
    if (!write_float32_array(stats_out / "ice_depth.bin", ice_depth)) {
        std::cerr << "[merge] ERROR writing ice_depth.bin\n";
        return false;
    }
    std::cout << "[merge] Wrote " << (stats_out / "ice_depth.bin") << "\n";
    return true;
}

void usage(const char* argv0) {
    std::cerr <<
"Usage: " << argv0 << " --chunks DIR1 DIR2 ... --output OUTDIR\n"
"                  [--J0 3.0e-14] [--ell-um 75.0]\n"
"\n"
"Combine per-chunk statistics from a chunked PTM3D run into a single set\n"
"of merged statistics. Reads each chunk's statistics/_raw/ to combine\n"
"max/min/sum/count and the Schorghofer ice-stability accumulator.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<fs::path> chunk_dirs;
    fs::path outdir;
    double J0 = 3.0e-14;
    double ell_um = 75.0;

    bool reading_chunks = false;
    bool allow_nonstandard_names = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--chunks") { reading_chunks = true; continue; }
        if (a == "--allow-nonstandard-names") {
            allow_nonstandard_names = true; reading_chunks = false; continue;
        }
        if (a == "--output" && i + 1 < argc) {
            outdir = argv[++i]; reading_chunks = false; continue;
        }
        if (a == "--J0" && i + 1 < argc) {
            J0 = std::stod(argv[++i]); reading_chunks = false; continue;
        }
        if (a == "--ell-um" && i + 1 < argc) {
            ell_um = std::stod(argv[++i]); reading_chunks = false; continue;
        }
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        if (reading_chunks) { chunk_dirs.emplace_back(a); continue; }
        std::cerr << "Unknown argument: " << a << "\n"; usage(argv[0]); return 2;
    }

    if (chunk_dirs.empty() || outdir.empty()) {
        usage(argv[0]); return 2;
    }
    for (const auto& cd : chunk_dirs) {
        if (!fs::is_directory(cd)) {
            std::cerr << "Not a directory: " << cd << "\n"; return 2;
        }
    }

    // Defensive guard (2026-07-21): refuse chunk dirs whose name isn't the
    // canonical chunk_<NNN> the writers emit. This tool takes an EXPLICIT
    // chunk list (no glob), but a shell glob `chunk_*` in a re-merge command
    // would sweep in QUARANTINED dirs such as chunk_001_NOVF_GARBAGE and
    // silently corrupt the time-mean statistics (a max survives a cold garbage
    // chunk, a mean does not — the 2026-07-21 near-miss). Fail loud instead.
    // --allow-nonstandard-names overrides for a deliberate non-canonical merge.
    if (!allow_nonstandard_names) {
        auto is_canonical = [](const std::string& name) {
            if (name.rfind("chunk_", 0) != 0 || name.size() <= 6) return false;
            for (size_t k = 6; k < name.size(); ++k)
                if (!std::isdigit(static_cast<unsigned char>(name[k]))) return false;
            return true;
        };
        bool bad = false;
        for (const auto& cd : chunk_dirs) {
            const std::string base = cd.filename().string();
            if (!is_canonical(base)) {
                std::cerr << "ERROR: chunk dir '" << cd.string()
                          << "' has a non-canonical name (expected chunk_<NNN>). "
                             "This is the shell-glob-swept-a-quarantine-dir footgun "
                             "(e.g. chunk_*_GARBAGE); refusing to merge it into the "
                             "statistics. Pass an explicit canonical chunk list, or "
                             "--allow-nonstandard-names to override.\n";
                bad = true;
            }
        }
        if (bad) return 2;
    }

    fs::create_directories(outdir / "statistics");

    // Determine N (face count) from chunk 0.
    uint32_t expected_N = 0;
    {
        const fs::path probe = chunk_dirs.front() / "statistics" / "temperature_max.bin";
        if (!fs::exists(probe)) {
            std::cerr << "ERROR: cannot find " << probe
                      << " — is chunk 0 complete? was write_raw_stats=true set?\n";
            return 1;
        }
        std::ifstream is(probe, std::ios::binary);
        if (!is || !read_uint32(is, expected_N) || expected_N == 0) {
            std::cerr << "ERROR probing N from " << probe << "\n";
            return 1;
        }
    }
    std::cout << "[merge] face count: " << expected_N
              << "  chunks: " << chunk_dirs.size() << "\n";

    // Merge each per-face datatype that has the necessary _raw side files.
    static const std::vector<std::string> DATATYPES = {
        "temperature", "illumination", "direct_solar",
        "primary_reflected", "primary_ir"
    };
    bool ok = true;
    for (const auto& dt : DATATYPES) {
        if (!merge_one_datatype(dt, chunk_dirs, outdir, expected_N)) {
            std::cerr << "[merge] ERROR merging " << dt << "\n";
            ok = false;
        }
    }

    // Subsurface temperature maps (output.map_depths_m). The field names are
    // per-run, not a fixed list, so they are DISCOVERED from chunk 0 and then
    // required to be identical in every other chunk. A chunk that asked for a
    // different depth set is a refusal, not a partial merge: averaging one
    // window at 0.05 m with another at 0.06 m would produce a file whose name
    // is true of neither.
    {
        std::vector<std::string> depth_fields = discover_depth_fields(chunk_dirs.front());
        std::sort(depth_fields.begin(), depth_fields.end());
        for (size_t i = 1; i < chunk_dirs.size(); ++i) {
            std::vector<std::string> other = discover_depth_fields(chunk_dirs[i]);
            std::sort(other.begin(), other.end());
            if (other != depth_fields) {
                std::cerr << "[merge] ERROR: chunk " << chunk_dirs[i]
                          << " carries a DIFFERENT set of subsurface depth maps from "
                             "chunk 0 (" << other.size() << " vs "
                          << depth_fields.size() << " fields). The chunks were not "
                             "run with the same output.map_depths_m; refusing to "
                             "merge them into one product.\n";
                ok = false;
                depth_fields.clear();
                break;
            }
        }
        for (const auto& df : depth_fields) {
            if (!merge_one_datatype(df, chunk_dirs, outdir, expected_N,
                                    /*strict=*/true)) {
                std::cerr << "[merge] ERROR merging " << df << "\n";
                ok = false;
            }
        }
        if (!depth_fields.empty()) {
            // Carry the z-manifest through. It describes the depths, weights and
            // error budget, all of which are chunk-invariant; without it the
            // merged directory would hold maps whose depths are only readable
            // from their file names, at three decimals.
            const fs::path src = chunk_dirs.front() / "statistics" / "depth_map_z.json";
            if (fs::exists(src)) {
                std::error_code ec;
                fs::copy_file(src, outdir / "statistics" / "depth_map_z.json",
                              fs::copy_options::overwrite_existing, ec);
                if (ec) {
                    std::cerr << "[merge] WARNING: could not copy depth_map_z.json: "
                              << ec.message() << "\n";
                }
            } else {
                std::cerr << "[merge] WARNING: chunk 0 has subsurface depth maps but "
                             "no statistics/depth_map_z.json; the merged product will "
                             "not say which grid nodes they came from.\n";
            }
            std::cout << "[merge] subsurface depth maps: " << depth_fields.size()
                      << " field(s) merged\n";
        }
    }

    // Merge ice-stability accumulator (optional).
    if (!merge_ice_stability(chunk_dirs, outdir, expected_N,
                             J0, ell_um * 1e-6)) {
        ok = false;
    }

    return ok ? 0 : 1;
}
