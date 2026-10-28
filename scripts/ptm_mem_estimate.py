#!/usr/bin/env python3
"""
PTM3D memory estimator (portability Phase 1 §5).

Computes the dominant resident-memory footprints BEFORE a run and warns/refuses
when they exceed available RAM -- the laptop-portability guardrail. The two
dominant terms (per the Phase 0 audit, the July 2026 portability audit (docs/PORTABILITY-AUDIT.md in git history) §4):

  1. Absorbed-illumination series held resident before the thermal solve:
        n_faces x n_steps x n_components x sizeof(float64)
     (components: direct + primary-reflected + primary-IR ~= 3).
  2. Sparse view-factor matrix:
        nnz x bytes_per_nnz   (~12 B: int32 col + float64 val, or CSR-ish).

Usable as a library (runners call check_memory before launch) or a CLI:
    python3 scripts/ptm_mem_estimate.py --faces 100000 --steps 360
    python3 scripts/ptm_mem_estimate.py --faces 500000 --steps 1440 --vf-nnz 8e8

Dependency-free; unit-tested in scripts/test_ptm_run_guard.py's sibling checks
(see __main__ self-test at the bottom).
"""
from __future__ import annotations

import argparse
import os
import sys

GiB = 1024.0 ** 3


def estimate_illum_series_bytes(n_faces: int, n_steps: int,
                                n_components: int = 3,
                                dtype_bytes: int = 8) -> int:
    """Resident bytes for the full absorbed-illumination series."""
    return int(n_faces) * int(n_steps) * int(n_components) * int(dtype_bytes)


def estimate_vf_bytes(nnz: int, bytes_per_nnz: int = 12) -> int:
    """Resident bytes for the sparse view-factor matrix (0 if nnz falsy)."""
    return int(nnz) * int(bytes_per_nnz) if nnz else 0


def physical_ram_bytes() -> int:
    """Best-effort total physical RAM in bytes (0 if undeterminable)."""
    try:  # POSIX (Linux, macOS)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS fallback via sysctl
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        return int(out.strip())
    except Exception:
        return 0


def check_memory(n_faces: int, n_steps: int, *, n_components: int = 3,
                 vf_nnz: int = 0, ram_bytes: int | None = None,
                 headroom_frac: float = 0.75) -> dict:
    """Estimate footprint vs available RAM. Returns a dict with the bytes and a
    verdict: 'ok' | 'warn' | 'over'. 'warn' when the estimate exceeds
    headroom_frac of RAM; 'over' when it exceeds RAM outright. When RAM is
    undeterminable (ram_bytes==0) the verdict is 'unknown' (never a silent OK)."""
    illum = estimate_illum_series_bytes(n_faces, n_steps, n_components)
    vf = estimate_vf_bytes(vf_nnz)
    total = illum + vf
    ram = physical_ram_bytes() if ram_bytes is None else ram_bytes
    if not ram:
        verdict = "unknown"
    elif total > ram:
        verdict = "over"
    elif total > headroom_frac * ram:
        verdict = "warn"
    else:
        verdict = "ok"
    return dict(illum_bytes=illum, vf_bytes=vf, total_bytes=total,
                ram_bytes=ram, verdict=verdict,
                headroom_frac=headroom_frac)


def format_report(r: dict) -> str:
    def g(b):
        return f"{b / GiB:.2f} GiB"
    lines = [
        f"illumination series : {g(r['illum_bytes'])}",
        f"view-factor matrix  : {g(r['vf_bytes'])}",
        f"TOTAL resident est. : {g(r['total_bytes'])}",
        f"physical RAM        : {g(r['ram_bytes']) if r['ram_bytes'] else 'unknown'}",
        f"verdict             : {r['verdict'].upper()}",
    ]
    if r["verdict"] == "over":
        lines.append("  -> OVER RAM: reduce faces/steps, use sparse VF, or stream "
                     "(set PTM_RAM_ILLUM_MAX_GB below the series size).")
    elif r["verdict"] == "warn":
        lines.append(f"  -> above {int(r['headroom_frac']*100)}% of RAM: risky on this "
                     "host; consider chunking the output window.")
    elif r["verdict"] == "unknown":
        lines.append("  -> RAM undeterminable; cannot certify this fits. Verify manually.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--faces", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--vf-nnz", type=float, default=0, help="view-factor non-zeros")
    p.add_argument("--ram-gib", type=float, default=None,
                   help="override detected RAM (GiB) for what-if estimates")
    p.add_argument("--strict", action="store_true",
                   help="exit nonzero on 'over' (or 'warn'/'unknown')")
    args = p.parse_args(argv)
    ram = int(args.ram_gib * GiB) if args.ram_gib is not None else None
    r = check_memory(args.faces, args.steps, n_components=args.components,
                     vf_nnz=int(args.vf_nnz), ram_bytes=ram)
    print(format_report(r))
    if args.strict and r["verdict"] in ("over", "warn", "unknown"):
        return 1
    if r["verdict"] == "over":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
