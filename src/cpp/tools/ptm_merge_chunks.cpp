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
bool merge_one_datatype(const std::string& name,
                        const std::vector<fs::path>& chunk_dirs,
                        const fs::path& outdir,
                        uint32_t expected_N) {
    std::vector<float>   merged_max, merged_min;
    std::vector<double>  merged_sum;
    std::vector<uint64_t> merged_count;
    merged_max.assign(expected_N, std::numeric_limits<float>::lowest());
    merged_min.assign(expected_N, std::numeric_limits<float>::max());
    merged_sum.assign(expected_N, 0.0);
    merged_count.assign(expected_N, 0u);

    bool any_loaded = false;
    for (const auto& cd : chunk_dirs) {
        const fs::path stats = cd / "statistics";
        const fs::path raw   = stats / "_raw";
        const fs::path p_max   = stats / (name + "_max.bin");
        const fs::path p_min   = stats / (name + "_min.bin");
        const fs::path p_sum   = raw   / (name + "_sum.bin");
        const fs::path p_count = raw   / (name + "_count.bin");

        // If any of the four files is missing in any chunk, the datatype
        // wasn't computed (e.g. illumination on a thermal-only run);
        // skip this datatype for the whole merge.
        if (!fs::exists(p_max) || !fs::exists(p_min) ||
            !fs::exists(p_sum) || !fs::exists(p_count)) {
            return true;  // not an error — just nothing to merge
        }

        uint32_t N;
        std::vector<float>   max_arr, min_arr;
        std::vector<double>  sum_arr;
        std::vector<uint32_t> count_arr;
        if (!read_float32_array(p_max,  N, max_arr) || N != expected_N) return false;
        if (!read_float32_array(p_min,  N, min_arr) || N != expected_N) return false;
        if (!read_double_array (p_sum,  N, sum_arr) || N != expected_N) return false;
        if (!read_uint32_array (p_count,N, count_arr) || N != expected_N) return false;

        for (uint32_t i = 0; i < expected_N; ++i) {
            if (max_arr[i] > merged_max[i]) merged_max[i] = max_arr[i];
            if (min_arr[i] < merged_min[i]) merged_min[i] = min_arr[i];
            merged_sum[i]   += sum_arr[i];
            merged_count[i] += count_arr[i];
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
    ok &= write_float32_array(stats_out / (name + "_max.bin"), merged_max);
    ok &= write_float32_array(stats_out / (name + "_avg.bin"), merged_avg);
    ok &= write_float32_array(stats_out / (name + "_min.bin"), merged_min);
    if (ok) {
        std::cout << "[merge] " << name << ": merged " << chunk_dirs.size()
                  << " chunks  (" << expected_N << " faces)\n";
    }
    return ok;
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
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--chunks") { reading_chunks = true; continue; }
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

    // Merge ice-stability accumulator (optional).
    if (!merge_ice_stability(chunk_dirs, outdir, expected_N,
                             J0, ell_um * 1e-6)) {
        ok = false;
    }

    return ok ? 0 : 1;
}
