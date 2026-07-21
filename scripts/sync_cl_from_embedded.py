#!/usr/bin/env python3
"""
Single-source the OpenCL thermal kernel (Phase R / Fermi F5).

The authoritative kernel source is the embedded raw-string literal
`getEmbeddedKernelSource()` in src/cpp/gpu/thermal_opencl.cpp (it is what the
solver compiles by default and what the byte-identity gates validate). This
tool regenerates src/cpp/gpu/thermal_kernels.cl to be a VERBATIM copy of that
string, so the on-disk .cl can never silently diverge from the compiled kernel
(the F5 hazard: buildKernels() can load an external .cl via kernel_path, and a
stale .cl would swap in wrong physics). The runtime guard in buildKernels()
compares any file-loaded source against the embedded string and warns loudly on
any difference; this tool keeps the in-tree .cl passing that guard.

Usage:
    python3 scripts/sync_cl_from_embedded.py            # regenerate + report
    python3 scripts/sync_cl_from_embedded.py --check     # exit 1 if out of sync
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "src/cpp/gpu/thermal_opencl.cpp"
CL = REPO / "src/cpp/gpu/thermal_kernels.cl"
OPEN = 'R"CL_SOURCE('
CLOSE = ')CL_SOURCE"'


def extract_embedded(cpp_text: str) -> str:
    i = cpp_text.find(OPEN)
    if i < 0:
        sys.exit(f"error: {OPEN} not found in {CPP}")
    if cpp_text.find(OPEN, i + 1) >= 0:
        sys.exit(f"error: multiple {OPEN} literals found; extractor assumes one")
    start = i + len(OPEN)
    j = cpp_text.find(CLOSE, start)
    if j < 0:
        sys.exit(f"error: closing {CLOSE} not found in {CPP}")
    return cpp_text[start:j]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="verify the .cl matches the embedded string; exit 1 if not")
    args = p.parse_args(argv)

    embedded = extract_embedded(CPP.read_text())
    current = CL.read_text() if CL.exists() else None

    if args.check:
        if current == embedded:
            print("in sync: thermal_kernels.cl == embedded getEmbeddedKernelSource()")
            return 0
        print("OUT OF SYNC: thermal_kernels.cl != embedded source. Run "
              "scripts/sync_cl_from_embedded.py to regenerate.", file=sys.stderr)
        return 1

    CL.write_text(embedded)
    print(f"wrote {CL} ({len(embedded)} bytes) verbatim from "
          f"getEmbeddedKernelSource() in {CPP.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
