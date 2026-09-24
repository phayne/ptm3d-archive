#!/usr/bin/env python3
"""Stitch per-step frame files from N sequential wedge-safe-runner chunk dirs
into a single continuously-numbered directory.

CONTEXT: `run_yearlong_wedge_safe.py --keep-time-series` writes real per-step
binary frames (e.g. <chunk>/temperature/temperature_NNNN.bin) into EACH chunk
dir separately, each chunk restarting its own local numbering at 0000.
`ptm_merge_chunks` only merges statistics/_raw accumulators (max/min/sum/
count) into a single statistics/ dir -- it does NOT stitch per-step frames
across chunks, so a consumer expecting `<run_dir>/<field>/<field>_NNNN.bin`
(e.g. `diviner_compare.py --max-mode rigorous`) finds nothing at the
top-level merged output dir.

This script concatenates the frame sequence from multiple chunk dirs (in the
order given -- normally chunk_000, chunk_001, ...) into
`<out-dir>/<field>/<field>_NNNN.bin` with globally-increasing indices
(hardlinked, not copied, for speed/disk -- falls back to a copy across
filesystems). Numbering preserves temporal order across chunks for anyone
who cares about it; note that a rigorous max-over-time comparison does NOT
depend on order (np.maximum is commutative), only on including every frame
exactly once.

Usage:
    python3 scripts/stitch_frame_dirs.py --field temperature \\
        --chunk-dirs .../coupled/chunk_000 .../coupled/chunk_001 \\
        --out-dir .../coupled
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True,
                    help="frame subdirectory name, e.g. temperature/illumination/"
                         "direct_solar/primary_reflected/primary_ir")
    ap.add_argument("--chunk-dirs", nargs="+", required=True, type=Path,
                    help="chunk directories IN TEMPORAL ORDER (chunk_000 chunk_001 ...)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="run-level output dir; frames written to <out-dir>/<field>/")
    ap.add_argument("--width", type=int, default=0,
                    help="zero-pad width of output index; 0 (default) = the engine's rule, "
                         "max(6, digits(total frames - 1)), so a lexicographic sort of the "
                         "output is its numeric order (BUGS.md PTM-664)")
    a = ap.parse_args()

    dst_dir = a.out_dir / a.field
    dst_dir.mkdir(parents=True, exist_ok=True)

    if a.width <= 0:
        n_all = sum(1 for cd in a.chunk_dirs if (cd / a.field).is_dir()
                    for p in (cd / a.field).glob(f"{a.field}_*.bin")
                    if re.fullmatch(re.escape(a.field) + r"_\d+\.bin", p.name))
        a.width = max(6, len(str(max(n_all - 1, 0))))

    global_idx = 0
    n_total = 0
    for cd in a.chunk_dirs:
        src_field_dir = cd / a.field
        if not src_field_dir.is_dir():
            print(f"[stitch] WARNING: {src_field_dir} missing -- skipping (this chunk "
                  "may not have had frames for this field, or was run before "
                  "--keep-time-series).")
            continue
        files = sorted(src_field_dir.glob(f"{a.field}_*.bin"),
                       key=lambda p: int(p.name.rsplit("_", 1)[1][:-4]))
        for f in files:
            dst = dst_dir / f"{a.field}_{global_idx:0{a.width}d}.bin"
            if dst.exists():
                dst.unlink()
            try:
                import os
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
            global_idx += 1
        print(f"[stitch] {src_field_dir} -> {len(files)} frames (global idx "
              f"{global_idx - len(files)}..{global_idx - 1})")
        n_total += len(files)

    print(f"[stitch] TOTAL {n_total} frames written to {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
