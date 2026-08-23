#!/usr/bin/env python3
"""Every listed header must compile ALONE, and every listed file must include
what it uses.

Why this exists: libstdc++ and libc++ disagree about what a standard header
pulls in transitively. This repo is developed on macOS/libc++ and built in CI on
Linux/libstdc++, so a header that quietly relies on `<vector>` dragging in
`<cstdint>` compiles here and fails there. That is a CI round-trip per missing
include, and this branch spent one on exactly `uint64_t`.

Two checks, because neither alone is sufficient:

1. STANDALONE COMPILE. A one-line translation unit that includes only the
   header. This catches a header that depends on its includer's includes --
   the failure mode that hits every consumer, not just the strict compiler.
   Necessary but NOT sufficient: libc++ is lenient, so a header can pass here
   and still fail on libstdc++.

2. SYMBOL AUDIT. Each file's std:: and C-library usages are matched against the
   headers the STANDARD requires for them, not against what the local library
   happens to provide. This is the part that catches the libstdc++-only
   failures without having libstdc++.

The GPU translation units (2026-08-22). Everything under src/cpp/gpu is
compiled ONLY with PTM_ENABLE_GPU=ON, and until the gpu-portable-isa CI job
existed that combination had never been built on Linux at all -- so this whole
subtree carried its libc++-transitive includes unchallenged for years.
src/cpp/gpu/thermal_gpu_pool.cpp used std::this_thread::sleep_for with no
<thread>, and its first Linux compile failed on exactly that. Those files are
now in FILES.

GPU_REGION_FILES is a third mode for the shared translation units --
radiation_exchange.cpp, ptm_thermal_sim.cpp, the GPU tests -- that compile in
BOTH configurations. Only the text inside their `#ifdef PTM_GPU_OPENCL` blocks
is audited. That is deliberate: what those blocks use is invisible to the
CPU-only Linux CI leg, whereas the rest of each file is already proven against
libstdc++ every time that leg goes green. Auditing them whole would drag in
pre-existing include-what-you-use debt that no compiler has ever objected to,
and a gate that reports things nobody will fix is a gate people switch off.

Usage:
    python3 scripts/check_header_self_sufficiency.py            # default file set
    python3 scripts/check_header_self_sufficiency.py FILE...    # explicit
Exit code is the number of files with findings, so it is usable as a gate.
"""

import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Headers that must compile standalone.
HEADERS = [
    "src/cpp/fluxcubes/flux_cube_format.h",
    "src/cpp/fluxcubes/flux_cube_pass.h",
    "src/cpp/radiation/nearfield_quadrature.h",
    "src/cpp/radiation/radiosity_reconstruction.h",
]

# Everything under src/cpp/gpu: compiled only with PTM_ENABLE_GPU=ON, so the
# CPU-only CI legs never see it. Listed explicitly rather than globbed, so that
# a new GPU source has to be added here consciously.
GPU_FILES = [
    "src/cpp/gpu/bvh_gpu_pool.cpp",
    "src/cpp/gpu/bvh_gpu_pool.h",
    "src/cpp/gpu/bvh_opencl.cpp",
    "src/cpp/gpu/bvh_opencl.h",
    "src/cpp/gpu/cwbvh_builder.cpp",
    "src/cpp/gpu/cwbvh_builder.h",
    "src/cpp/gpu/gpu_config.h",
    "src/cpp/gpu/gpu_guardrails.h",
    "src/cpp/gpu/gpu_memory.h",
    "src/cpp/gpu/gpu_spmv_escalation.cpp",
    "src/cpp/gpu/gpu_utils.h",
    "src/cpp/gpu/machine_profile.h",
    "src/cpp/gpu/opencl_config.h",
    "src/cpp/gpu/radiosity_gpu_pool.cpp",
    "src/cpp/gpu/radiosity_gpu_pool.h",
    "src/cpp/gpu/radiosity_opencl.cpp",
    "src/cpp/gpu/radiosity_opencl.h",
    "src/cpp/gpu/thermal_gpu.cpp",
    "src/cpp/gpu/thermal_gpu.h",
    "src/cpp/gpu/thermal_gpu_pool.cpp",
    "src/cpp/gpu/thermal_gpu_pool.h",
    "src/cpp/gpu/thermal_kernels.cpp",
    "src/cpp/gpu/thermal_kernels.h",
    "src/cpp/gpu/thermal_opencl.cpp",
    "src/cpp/gpu/thermal_opencl.h",
]

# Shared translation units: audited ONLY inside their #ifdef PTM_GPU_OPENCL
# blocks (see the module docstring for why the rest is left alone).
GPU_REGION_FILES = [
    "src/cpp/illumination/illumination_model.cpp",
    "src/cpp/illumination/illumination_model.h",
    "src/cpp/main/ptm_thermal_sim.cpp",
    "src/cpp/radiation/benchmark_gpu_visibility.cpp",
    "src/cpp/radiation/radiation_exchange.cpp",
    "src/cpp/radiation/radiation_exchange.h",
    "src/cpp/radiation/test_bowl_crater.cpp",
    "src/cpp/radiation/test_two_band_rownorm.cpp",
]

# Files whose includes must cover their usages.
FILES = HEADERS + [
    "src/cpp/fluxcubes/flux_cube_pass.cpp",
    "src/cpp/fluxcubes/flux_cube_geom.cpp",
    "src/cpp/fluxcubes/test_flux_cube_pass.cpp",
    "src/cpp/radiation/test_flux_cube_identity.cpp",
    "src/cpp/radiation/test_vf_production.cpp",
    "src/cpp/radiation/nearfield_quadrature.cpp",
] + GPU_FILES + GPU_REGION_FILES

# symbol pattern -> the header(s) the STANDARD requires. A tuple means any one
# of them satisfies the usage (std::exception is declared in <exception>, and
# <stdexcept> is required to include it; std::abs has integer and
# floating-point overloads in <cstdlib> and <cmath> respectively).
NEED = {
    r'\bstd::string\b|\bstd::to_string\b|\bstd::sto[dilf]\w*\b|\bstd::getline\b'
    r'|\bstd::string_view\b': 'string',
    r'\bstd::vector\b': 'vector',
    r'\bstd::array\b': 'array',
    r'\bstd::deque\b': 'deque',
    r'\bstd::map\b|\bstd::multimap\b': 'map',
    r'\bstd::set\b|\bstd::multiset\b': 'set',
    r'\bstd::unordered_map\b': 'unordered_map',
    r'\bstd::unordered_set\b': 'unordered_set',
    r'\bstd::sort\b|\bstd::stable_sort\b|\bstd::max\b|\bstd::min\b|\bstd::clamp\b'
    r'|\bstd::nth_element\b|\bstd::unique\b|\bstd::find\b|\bstd::find_if\b|\bstd::copy\b'
    r'|\bstd::fill\b|\bstd::fill_n\b|\bstd::min_element\b|\bstd::max_element\b'
    r'|\bstd::minmax_element\b|\bstd::count\b|\bstd::count_if\b|\bstd::lower_bound\b'
    r'|\bstd::upper_bound\b|\bstd::reverse\b|\bstd::transform\b|\bstd::any_of\b'
    r'|\bstd::all_of\b|\bstd::none_of\b|\bstd::for_each\b|\bstd::remove_if\b'
    r'|\bstd::partial_sort\b': 'algorithm',
    r'\bstd::accumulate\b|\bstd::iota\b|\bstd::inner_product\b|\bstd::partial_sum\b': 'numeric',
    r'\bstd::sqrt\b|\bstd::fabs\b|\bstd::cos\b|\bstd::sin\b|\bstd::atan2?\b|\bstd::acos\b'
    r'|\bstd::asin\b|\bstd::tan\b|\bstd::pow\b|\bstd::floor\b|\bstd::ceil\b|\bstd::exp\b'
    r'|\bstd::log\b|\bstd::log10\b|\bstd::round\b|\bstd::isnan\b|\bstd::isfinite\b'
    r'|\bstd::isinf\b|\bstd::hypot\b|\bM_PI\b': 'cmath',
    r'\bstd::abs\b': ('cmath', 'cstdlib'),
    r'\bstd::shared_ptr\b|\bstd::unique_ptr\b|\bstd::make_shared\b|\bstd::make_unique\b'
    r'|\bstd::weak_ptr\b': 'memory',
    r'\bstd::runtime_error\b|\bstd::invalid_argument\b|\bstd::out_of_range\b'
    r'|\bstd::logic_error\b|\bstd::length_error\b': 'stdexcept',
    r'\bstd::exception\b': ('stdexcept', 'exception'),
    r'\bstd::ostringstream\b|\bstd::istringstream\b|\bstd::stringstream\b': 'sstream',
    r'\bstd::ifstream\b|\bstd::ofstream\b|\bstd::fstream\b': 'fstream',
    r'\bstd::cout\b|\bstd::cerr\b|\bstd::endl\b|\bstd::cin\b|\bstd::flush\b': 'iostream',
    r'\bstd::memcpy\b|\bstd::memset\b|\bstd::memmove\b|\bstd::strncpy\b|\bstd::strlen\b'
    r'|\bstd::strcmp\b|\bstd::strncmp\b|\bstd::strstr\b': 'cstring',
    r'\bstd::printf\b|\bstd::snprintf\b|\bstd::fprintf\b|\bstd::sprintf\b|\bstd::fopen\b'
    r'|\bstd::fclose\b|\bstd::fwrite\b|\bstd::fread\b|\bstd::FILE\b|\bstd::fflush\b'
    r'|\bstd::fputs\b|\bstd::puts\b': 'cstdio',
    r'\bstd::getenv\b|\bstd::atof\b|\bstd::atoi\b|\bstd::atoll\b|\bstd::exit\b'
    r'|\bstd::strtol\w*\b|\bstd::strtod\b|\bstd::strtoul\w*\b|\bstd::abort\b'
    r'|\bstd::qsort\b': 'cstdlib',
    r'\bu?int(8|16|32|64)_t\b|\buintptr_t\b|\bintptr_t\b': 'cstdint',
    r'\bsize_t\b|\bptrdiff_t\b': ('cstddef', 'cstdint'),
    r'\bstd::function\b|\bstd::bind\b|\bstd::ref\b|\bstd::cref\b|\bstd::hash\b'
    r'|\bstd::greater\b|\bstd::less\b': 'functional',
    r'\bstd::pair\b|\bstd::make_pair\b|\bstd::move\b|\bstd::forward\b|\bstd::swap\b': 'utility',
    r'\bstd::tuple\b|\bstd::make_tuple\b|\bstd::tie\b': 'tuple',
    r'\bstd::thread\b|\bstd::this_thread\b': 'thread',
    r'\bstd::mutex\b|\bstd::recursive_mutex\b|\bstd::lock_guard\b|\bstd::unique_lock\b'
    r'|\bstd::scoped_lock\b|\bstd::call_once\b|\bstd::once_flag\b': 'mutex',
    r'\bstd::condition_variable\b': 'condition_variable',
    r'\bstd::future\b|\bstd::async\b|\bstd::promise\b|\bstd::packaged_task\b'
    r'|\bstd::launch\b|\bstd::shared_future\b': 'future',
    r'\bstd::atomic\b|\bstd::memory_order\w*\b': 'atomic',
    r'\bstd::chrono\b': 'chrono',
    r'\bstd::optional\b|\bstd::nullopt\b': 'optional',
    r'\bstd::is_same\b|\bstd::enable_if\b|\bstd::decay\b|\bstd::remove_reference\b'
    r'|\bstd::is_integral\b|\bstd::is_floating_point\b|\bstd::conditional\b': 'type_traits',
    r'\bstd::localtime\b|\bstd::gmtime\b|\bstd::mktime\b|\bstd::strftime\b'
    r'|\bstd::time_t\b': 'ctime',
    r'\bstd::isspace\b|\bstd::tolower\b|\bstd::toupper\b|\bstd::isdigit\b'
    r'|\bstd::isalpha\b': 'cctype',
    r'\bstd::mt19937\w*\b|\bstd::uniform_real_distribution\b'
    r'|\bstd::uniform_int_distribution\b|\bstd::random_device\b'
    r'|\bstd::normal_distribution\b': 'random',
    r'\bstd::numeric_limits\b': 'limits',
    r'\bstd::setprecision\b|\bstd::setw\b|\bstd::setfill\b|\bstd::setbase\b': 'iomanip',
    r'\bstd::filesystem\b': 'filesystem',
}


def strip_noncode(src):
    """Remove comments and string literals -- a word in prose is not a usage."""
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'"(\\.|[^"\\])*"', '""', src)
    return src


def standalone(header):
    with tempfile.TemporaryDirectory() as d:
        tu = os.path.join(d, "t.cpp")
        rel = header.split("src/cpp/", 1)[1]
        with open(tu, "w") as f:
            f.write('#include "%s"\nint main(){return 0;}\n' % rel)
        r = subprocess.run(
            ["clang++", "-std=c++17", "-fsyntax-only", "-Wno-nonportable-include-path",
             "-I", os.path.join(REPO, "src/cpp"),
             "-I", os.path.join(REPO, "src/cpp/fluxcubes"), tu],
            capture_output=True, text=True)
        return r.returncode == 0, r.stderr


def gpu_regions(body):
    """Just the text inside `#ifdef PTM_GPU_OPENCL` ... matching `#endif`.

    Nested #if/#endif are tracked so a guard inside the block does not end it
    early. Two exclusions keep the result honest:
      * `#ifndef PTM_GPU_OPENCL` is the CPU-only fallback, not GPU code.
      * an `#else`/`#elif` at the block's own depth ends the GPU branch; the
        text after it is the CPU fallback and the CPU CI leg already compiles
        it. Collecting it would let a symbol used only in the fallback be
        reported as a GPU-only include gap -- a false positive.
    """
    lines = body.split('\n')
    out, i, n = [], 0, len(lines)
    while i < n:
        opens = re.match(r'\s*#\s*if(def)?\s+.*PTM_GPU_OPENCL', lines[i])
        if opens and not re.match(r'\s*#\s*ifndef\s+PTM_GPU_OPENCL\b', lines[i]):
            depth, j, collecting = 1, i + 1, True
            while j < n and depth:
                if re.match(r'\s*#\s*if', lines[j]):
                    depth += 1
                elif re.match(r'\s*#\s*endif', lines[j]):
                    depth -= 1
                elif depth == 1 and re.match(r'\s*#\s*el(se|if)\b', lines[j]):
                    collecting = False
                elif collecting:
                    out.append(lines[j])
                j += 1
            i = j
        else:
            i += 1
    return '\n'.join(out)


def audit(path, gpu_regions_only=False):
    src = open(os.path.join(REPO, path)).read()
    incs = set(re.findall(r'#include\s*<([^>]+)>', src))
    body = strip_noncode(src)
    # An #include line is not a usage: <cstring> must not satisfy itself.
    body = re.sub(r'#\s*include[^\n]*', '', body)
    if gpu_regions_only:
        body = gpu_regions(body)
    missing = []
    for pat, hdr in NEED.items():
        hdrs = (hdr,) if isinstance(hdr, str) else hdr
        m = re.search(pat, body)
        if m and not (set(hdrs) & incs):
            missing.append((hdrs[0], m.group(0)))
    return sorted(set(missing))


def main():
    files = sys.argv[1:] or FILES
    bad = 0
    print("--- standalone header compile ---")
    for h in HEADERS:
        if h not in files:
            continue
        ok, err = standalone(h)
        print("  %s  %s" % ("PASS" if ok else "FAIL", h))
        if not ok:
            bad += 1
            print("        " + err.strip().splitlines()[0] if err.strip() else "")
    print("--- include-what-you-use ---")
    for p in files:
        regions_only = p in GPU_REGION_FILES
        miss = audit(p, gpu_regions_only=regions_only)
        print("  %s  %-46s%s"
              % ("OK  " if not miss else "MISS", os.path.basename(p),
                 "  [PTM_GPU_OPENCL regions only]" if regions_only else ""))
        for h, ex in miss:
            print("          needs <%s>   (uses %s)" % (h, ex))
        if miss:
            bad += 1
    print("\n%s (%d file(s) with findings)"
          % ("check_header_self_sufficiency: clean" if not bad else "FINDINGS", bad))
    return bad


if __name__ == "__main__":
    sys.exit(main())
