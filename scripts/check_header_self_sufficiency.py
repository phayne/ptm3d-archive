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
    "src/cpp/radiation/radiosity_reconstruction.h",
]

# Files whose includes must cover their usages.
FILES = HEADERS + [
    "src/cpp/fluxcubes/flux_cube_pass.cpp",
    "src/cpp/fluxcubes/flux_cube_geom.cpp",
    "src/cpp/fluxcubes/test_flux_cube_pass.cpp",
    "src/cpp/radiation/test_flux_cube_identity.cpp",
]

# symbol pattern -> the header the STANDARD requires.
NEED = {
    r'\bstd::string\b|\bstd::to_string\b|\bstd::stod\b|\bstd::getline\b': 'string',
    r'\bstd::vector\b': 'vector',
    r'\bstd::array\b': 'array',
    r'\bstd::map\b': 'map',
    r'\bstd::unordered_map\b': 'unordered_map',
    r'\bstd::sort\b|\bstd::max\b|\bstd::min\b|\bstd::clamp\b|\bstd::nth_element\b'
    r'|\bstd::unique\b|\bstd::find\b|\bstd::copy\b|\bstd::fill\b|\bstd::greater\b': 'algorithm',
    r'\bstd::sqrt\b|\bstd::fabs\b|\bstd::cos\b|\bstd::sin\b|\bstd::atan2\b|\bstd::acos\b'
    r'|\bstd::asin\b|\bstd::tan\b|\bstd::pow\b|\bstd::floor\b|\bM_PI\b': 'cmath',
    r'\bstd::shared_ptr\b|\bstd::unique_ptr\b|\bstd::make_shared\b': 'memory',
    r'\bstd::runtime_error\b|\bstd::invalid_argument\b|\bstd::exception\b': 'stdexcept',
    r'\bstd::ostringstream\b|\bstd::istringstream\b|\bstd::stringstream\b': 'sstream',
    r'\bstd::ifstream\b|\bstd::ofstream\b|\bstd::fstream\b': 'fstream',
    r'\bstd::cout\b|\bstd::cerr\b|\bstd::endl\b': 'iostream',
    r'\bstd::memcpy\b|\bstd::memset\b|\bstd::strncpy\b|\bstd::strlen\b|\bstd::strcmp\b': 'cstring',
    r'\bstd::printf\b|\bstd::snprintf\b|\bstd::fprintf\b': 'cstdio',
    r'\bstd::getenv\b|\bstd::atof\b|\bstd::atoi\b|\bstd::atoll\b|\bstd::exit\b': 'cstdlib',
    r'\bu?int(8|16|32|64)_t\b': 'cstdint',
    r'\bsize_t\b': 'cstddef',
    r'\bstd::function\b': 'functional',
    r'\bstd::pair\b|\bstd::make_pair\b|\bstd::move\b': 'utility',
    r'\bstd::mt19937_64\b|\bstd::uniform_real_distribution\b': 'random',
    r'\bstd::numeric_limits\b': 'limits',
    r'\bstd::setprecision\b|\bstd::setw\b': 'iomanip',
    r'\bstd::filesystem\b': 'filesystem',
    r'\bstd::accumulate\b': 'numeric',
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


def audit(path):
    src = open(os.path.join(REPO, path)).read()
    incs = set(re.findall(r'#include\s*<([^>]+)>', src))
    body = strip_noncode(src)
    missing = []
    for pat, hdr in NEED.items():
        m = re.search(pat, body)
        if m and hdr not in incs:
            missing.append((hdr, m.group(0)))
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
        miss = audit(p)
        print("  %s  %-46s" % ("OK  " if not miss else "MISS", os.path.basename(p)))
        for h, ex in miss:
            print("          needs <%s>   (uses %s)" % (h, ex))
        if miss:
            bad += 1
    print("\n%s (%d file(s) with findings)"
          % ("check_header_self_sufficiency: clean" if not bad else "FINDINGS", bad))
    return bad


if __name__ == "__main__":
    sys.exit(main())
