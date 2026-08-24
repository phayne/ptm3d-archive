// run_provenance.cpp -- see run_provenance.h for the contract.

#include "run_provenance.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "ptm_build_info.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace ptm {
namespace {

// ---------------------------------------------------------------------------
// MD5 (RFC 1321). Written out rather than pulled from a dependency because the
// engine links no crypto library and this project identifies binaries by md5 --
// the value is checked against `md5 build/bin/ptm_thermal_sim` in the tests, so
// a transcription error cannot survive.
// ---------------------------------------------------------------------------
struct Md5 {
    std::uint32_t a = 0x67452301u, b = 0xefcdab89u, c = 0x98badcfeu, d = 0x10325476u;
    std::uint64_t length = 0;                 // total message length in bytes
    std::array<std::uint8_t, 64> block{};
    std::size_t block_used = 0;

    static std::uint32_t rotl(std::uint32_t x, int s) {
        return (x << s) | (x >> (32 - s));
    }

    void transform(const std::uint8_t* p) {
        static const std::uint32_t K[64] = {
            0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu,
            0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
            0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
            0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
            0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
            0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
            0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
            0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
            0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
            0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
            0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u,
            0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
            0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u,
            0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
            0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
            0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};
        static const int S[64] = {
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
            5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

        std::uint32_t M[16];
        for (int i = 0; i < 16; ++i) {
            M[i] = static_cast<std::uint32_t>(p[i * 4])
                 | (static_cast<std::uint32_t>(p[i * 4 + 1]) << 8)
                 | (static_cast<std::uint32_t>(p[i * 4 + 2]) << 16)
                 | (static_cast<std::uint32_t>(p[i * 4 + 3]) << 24);
        }

        std::uint32_t A = a, B = b, C = c, D = d;
        for (int i = 0; i < 64; ++i) {
            std::uint32_t F;
            int g;
            if (i < 16)      { F = (B & C) | (~B & D);          g = i; }
            else if (i < 32) { F = (D & B) | (~D & C);          g = (5 * i + 1) % 16; }
            else if (i < 48) { F = B ^ C ^ D;                   g = (3 * i + 5) % 16; }
            else             { F = C ^ (B | ~D);                g = (7 * i) % 16; }
            F = F + A + K[i] + M[g];
            A = D; D = C; C = B;
            B = B + rotl(F, S[i]);
        }
        a += A; b += B; c += C; d += D;
    }

    void update(const std::uint8_t* p, std::size_t n) {
        length += n;
        while (n > 0) {
            const std::size_t take = std::min(n, std::size_t{64} - block_used);
            std::memcpy(block.data() + block_used, p, take);
            block_used += take; p += take; n -= take;
            if (block_used == 64) { transform(block.data()); block_used = 0; }
        }
    }

    std::string finish() {
        const std::uint64_t bits = length * 8;
        std::uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0x00;
        while (block_used != 56) update(&pad, 1);
        std::uint8_t tail[8];
        for (int i = 0; i < 8; ++i) tail[i] = static_cast<std::uint8_t>((bits >> (8 * i)) & 0xff);
        // update() would add these 8 bytes to `length`, which no longer
        // matters: the padded length is already fixed.
        std::memcpy(block.data() + block_used, tail, 8);
        transform(block.data());

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::uint32_t w : {a, b, c, d})
            for (int i = 0; i < 4; ++i)
                out << std::setw(2) << ((w >> (8 * i)) & 0xffu);
        return out.str();
    }
};

std::string isoUtcNow() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

// argv, re-joined for the record. Any element containing whitespace is quoted
// so the line can be pasted back into a shell; a value like an output path with
// a space would otherwise reconstruct as two arguments.
std::string joinArgv(const std::vector<std::string>& argv) {
    std::ostringstream out;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i) out << ' ';
        const bool needs_quotes =
            argv[i].empty() ||
            argv[i].find_first_of(" \t\n\"'\\") != std::string::npos;
        if (!needs_quotes) { out << argv[i]; continue; }
        out << '"';
        for (char ch : argv[i]) {
            if (ch == '"' || ch == '\\') out << '\\';
            out << ch;
        }
        out << '"';
    }
    return out.str();
}

}  // namespace

std::string md5File(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    Md5 md5;
    std::vector<char> buf(64 * 1024);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = f.gcount();
        if (got > 0)
            md5.update(reinterpret_cast<const std::uint8_t*>(buf.data()),
                       static_cast<std::size_t>(got));
    }
    return md5.finish();
}

std::string selfExecutablePath(const char* argv0) {
#if defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (size > 0 && _NSGetExecutablePath(buf.data(), &size) == 0) {
        buf.resize(std::strlen(buf.c_str()));
        std::error_code ec;
        auto canon = std::filesystem::weakly_canonical(buf, ec);
        return ec ? buf : canon.string();
    }
#elif defined(__linux__)
    std::error_code ec;
    auto link = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return link.string();
#endif
    if (argv0 && *argv0) {
        std::error_code ec;
        auto canon = std::filesystem::weakly_canonical(argv0, ec);
        if (!ec) return canon.string();
        return argv0;
    }
    return "";
}

bool writeRunProvenance(const RunProvenance& p) {
    std::error_code ec;
    std::filesystem::create_directories(p.output_dir, ec);

    // --- verbatim copy of the configuration --------------------------------
    std::string config_copy;
    std::string config_md5;
    std::string config_abs;
    if (!p.config_file.empty()) {
        config_abs = std::filesystem::weakly_canonical(p.config_file, ec).string();
        if (ec) { config_abs = p.config_file; ec.clear(); }
        const std::string dest = p.output_dir + "/config_used.yaml";
        // copy_file with overwrite: a re-run into the same directory must
        // replace the previous copy, not fail and leave the older config
        // sitting next to newer outputs.
        std::filesystem::copy_file(config_abs, dest,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "[provenance] WARNING: could not copy " << config_abs
                      << " to " << dest << ": " << ec.message() << std::endl;
            ec.clear();
        } else {
            config_copy = "config_used.yaml";
            config_md5 = md5File(dest);
        }
    }

    const std::string exe = selfExecutablePath(p.argv.empty() ? nullptr : p.argv[0].c_str());
    const std::string exe_md5 = exe.empty() ? std::string() : md5File(exe);

    const std::string path = p.output_dir + "/provenance.txt";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[provenance] WARNING: cannot write " << path
                  << "; this run will not be reconstructable from its outputs."
                  << std::endl;
        return false;
    }

    auto kv = [&out](const char* k, const std::string& v) { out << k << '=' << v << '\n'; };

    kv("schema", "ptm3d-provenance-1");
    kv("written_by", "ptm_thermal_sim");
    kv("run_started_utc", isoUtcNow());

    kv("engine_version", PTM_BUILD_VERSION);
    kv("engine_git_sha", PTM_BUILD_GIT_SHA);
    kv("engine_build_type", PTM_BUILD_TYPE);
    kv("engine_gpu_enabled", std::string(PTM_BUILD_GPU) == "ON" ? "true" : "false");
    kv("engine_compiler", PTM_BUILD_COMPILER);
    kv("engine_binary", exe.empty() ? "unknown" : exe);
    kv("engine_binary_md5", exe_md5.empty() ? "unknown" : exe_md5);
    // Rule 7: name the limit of the checksum instead of letting a reader
    // assume it covers the whole engine. With static libraries -- the default
    // -- the executable IS the engine, and its md5 is a complete identity.
    // Under BUILD_SHARED_LIBS a change confined to (say) libthermal moves the
    // library and leaves this md5 unchanged, so the reader has to be told.
    kv("engine_shared_libraries", PTM_BUILD_SHARED_LIBS);
    // Embree is the one dependency that is ALWAYS a dynamic library and is NOT
    // results-neutral, so the md5 above cannot stand in for it: `brew upgrade
    // embree`, or rebuilding the same prefix with a different EMBREE_MAX_ISA,
    // changes outputs with every other field here unchanged. Measured
    // 2026-08-15 on an SSE2 -> multi-ISA swap of identical Embree 4.3.3
    // sources: Europa and Bennu byte-identical, a lunar polar DEM patch moved
    // 75 of 260 output files, worst face-frame |dT| 0.0091 K. These two keys
    // are what lets a future results diff be attributed instead of guessed at.
    kv("embree_library", PTM_BUILD_EMBREE_LIB);
    kv("embree_max_isa", PTM_BUILD_EMBREE_MAX_ISA);
    // Thread count belongs in the provenance for the same reason Embree does:
    // it changes the bytes. Several OpenMP reductions in the solver and the
    // statistics accumulators sum in thread-partition order, so the same
    // binary on the same config at 8 and at 48 threads writes different
    // statistics files. A baseline manifest that does not record this compares
    // two runs that were never comparable.
    {
        const char* env_nt = std::getenv("OMP_NUM_THREADS");
        kv("omp_num_threads_env", env_nt && *env_nt ? env_nt : "(unset)");
        if (p.omp_max_threads > 0) {
            out << "omp_max_threads=" << p.omp_max_threads << '\n';
        } else {
            kv("omp_max_threads", "unknown (caller did not supply it)");
        }
    }
    kv("command_line", joinArgv(p.argv));

    kv("body", p.body.empty() ? "unknown" : p.body);
    kv("generic_body", p.generic_body ? "true" : "false");
    kv("mesh_source", p.mesh_source.empty() ? "unknown" : p.mesh_source);
    out << "mesh_faces=" << p.mesh_faces << '\n';
    kv("mesh_length_unit", p.mesh_length_unit.empty() ? "unknown" : p.mesh_length_unit);
    out << std::setprecision(12);
    out << "mesh_median_radius=" << p.mesh_median_radius << '\n';

    kv("config_file", config_abs.empty() ? "none" : config_abs);
    kv("config_copy", config_copy.empty() ? "none" : config_copy);
    kv("config_md5", config_md5.empty() ? "none" : config_md5);

    std::string out_abs = std::filesystem::weakly_canonical(p.output_dir, ec).string();
    if (ec) { out_abs = p.output_dir; ec.clear(); }
    kv("output_directory", out_abs);

    out.close();
    if (!out) {
        std::cerr << "[provenance] WARNING: " << path << " failed to close cleanly."
                  << std::endl;
        return false;
    }
    std::cout << "[provenance] " << path << "  engine " << PTM_BUILD_VERSION
              << " @ " << PTM_BUILD_GIT_SHA << ", md5 "
              << (exe_md5.empty() ? "unknown" : exe_md5) << std::endl;
    return true;
}

}  // namespace ptm
