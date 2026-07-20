#!/usr/bin/env python3
"""
F11 — H-matrix vs sparse top-K view-factor head-to-head benchmark.

Parses the ``bin/benchmark_hmatrix`` output (typically
``tmp/benchmark_hmatrix.log``) and plots:

  (a) per-row storage: sparse top-K vs H-matrix vs dense (extrapolated);
  (b) matrix-vector-product wall-clock per call;
  (c) max relative MVP error of H-matrix relative to sparse-top-K
      reference.

Companion to §6.2 of the PTM3D methods paper.  Anchored on
test_bowl_crater meshes (456, 1044, 1872, 2940 faces).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, COLORS, COLUMN_W, DOUBLE_W, panel_label


_RE_BENCH = re.compile(r"^--- Benchmark: (\d+) rings x (\d+) azimuth ---")
_RE_FACES = re.compile(r"^\s*Faces:\s*(\d+)")
_RE_SPARSE_MEM = re.compile(r"^\s*Memory:\s*([\d.]+)\s*MB\s*$")
_RE_HMAT_MEM = re.compile(r"^H-matrix statistics:.*$")
_RE_HMAT_USAGE = re.compile(r"^\s*Memory usage:\s*([\d.]+)\s*MB\s*$")
_RE_SPARSE_MVP = re.compile(
    r"Benchmarking sparse MVP \(\d+ iterations\)\.\.\.\s*([\d.]+)\s*ms/MVP"
)
_RE_HMAT_MVP = re.compile(
    r"Benchmarking H-matrix MVP \(\d+ iterations\)\.\.\.\s*([\d.]+)\s*ms/MVP"
)
_RE_MVP_ERR = re.compile(
    r"MVP max relative error:\s*([\d.]+)%"
)


def parse_log(path: Path) -> list[dict]:
    """Extract per-benchmark statistics."""
    benches = []
    cur = None
    state_hmat = False
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if m := _RE_BENCH.match(line):
            if cur is not None and "n_faces" in cur:
                benches.append(cur)
            cur = {"rings": int(m.group(1)), "az": int(m.group(2))}
            state_hmat = False
            continue
        if cur is None:
            continue
        if m := _RE_FACES.match(line):
            cur["n_faces"] = int(m.group(1))
        elif m := _RE_SPARSE_MEM.match(line):
            # The first ``Memory: X MB`` line after Benchmark is the sparse memory;
            # subsequent ones are H-matrix block memory.  Use state_hmat to gate.
            if not state_hmat and "sparse_mem_mb" not in cur:
                cur["sparse_mem_mb"] = float(m.group(1))
        elif _RE_HMAT_MEM.match(line):
            state_hmat = True
        elif state_hmat and (m := _RE_HMAT_USAGE.match(line)):
            cur["hmat_mem_mb"] = float(m.group(1))
        elif m := _RE_SPARSE_MVP.search(line):
            cur["sparse_mvp_ms"] = float(m.group(1))
        elif m := _RE_HMAT_MVP.search(line):
            cur["hmat_mvp_ms"] = float(m.group(1))
        elif m := _RE_MVP_ERR.search(line):
            cur["mvp_rel_err_pct"] = float(m.group(1))
    if cur is not None and "n_faces" in cur:
        benches.append(cur)
    # Keep only the first benchmark of each face count (the main loop; ignore
    # SVD-method-comparison repetitions later in the log)
    seen = set()
    uniq = []
    for b in benches:
        if b["n_faces"] in seen:
            continue
        seen.add(b["n_faces"])
        uniq.append(b)
    return uniq


def make_figure(log_path: Path, out_path: Path) -> None:
    apply_style()
    benches = parse_log(log_path)
    if not benches:
        raise SystemExit(f"No benchmark records parsed from {log_path}")
    N = np.array([b["n_faces"] for b in benches])
    sm = np.array([b.get("sparse_mem_mb", np.nan) for b in benches])
    hm = np.array([b.get("hmat_mem_mb", np.nan) for b in benches])
    sm_mvp = np.array([b.get("sparse_mvp_ms", np.nan) for b in benches])
    hm_mvp = np.array([b.get("hmat_mvp_ms", np.nan) for b in benches])
    err = np.array([b.get("mvp_rel_err_pct", np.nan) for b in benches])

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W * 1.18, 0.46 * DOUBLE_W))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86,
                        bottom=0.18, wspace=0.34)

    # ----- (a) Per-row storage ----------------------------------------------
    ax = axes[0]
    # Dense storage extrapolation: N^2 * 8 bytes
    dense_mb = (N.astype(np.float64) ** 2 * 8.0) / (1024 ** 2)
    ax.plot(N, dense_mb, "-", color="#cc6600", lw=1.2,
            label=r"dense ($N^2 \cdot$8 B)")
    ax.plot(N, sm, "o-", color="#777777", lw=2.0, mfc="white", mew=1.8, ms=7,
            label="sparse top-K")
    ax.plot(N, hm, "s-", color=COLORS["ptm"], lw=2.0, mfc="white", mew=1.8, ms=7,
            label="H-matrix")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mesh faces $N$")
    ax.set_ylabel("Storage (MB)")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.0)
    panel_label(ax, "(a)")

    # ----- (b) MVP wall-clock -----------------------------------------------
    ax = axes[1]
    ax.plot(N, sm_mvp, "o-", color="#777777", lw=2.0,
            mfc="white", mew=1.8, ms=7, label="sparse top-K")
    ax.plot(N, hm_mvp, "s-", color=COLORS["ptm"], lw=2.0,
            mfc="white", mew=1.8, ms=7, label="H-matrix")
    ax.set_xscale("log")
    ax.set_xlabel("Mesh faces $N$")
    ax.set_ylabel("MVP time (ms/call)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.0,
              bbox_to_anchor=(1.0, 0.88))
    panel_label(ax, "(b)")

    # ----- (c) MVP error vs sparse reference --------------------------------
    ax = axes[2]
    ax.plot(N, err, "s-", color=COLORS["ptm"], lw=2.0,
            mfc="white", mew=1.8, ms=7,
            label="H-matrix vs sparse")
    ax.axhline(5.0, color=COLORS["neutral"], lw=1.0, ls=":")
    ax.text(N.max() * 0.9, 4.8, r"5 % reference", color=COLORS["neutral"],
            fontsize=9, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Mesh faces $N$")
    ax.set_ylabel("max rel MVP error (%)")
    ax.set_ylim(0, max(10.0, np.nanmax(err) * 1.4))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.0)
    panel_label(ax, "(c)")

    fig.suptitle(
        "H-matrix vs sparse top-K view factors "
        "(bowl-crater benchmark, ACA tol = 0.01)",
        y=0.96, fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

    print(f"\nParsed {len(benches)} benchmark records from {log_path.name}:")
    print(f"  {'N':>6}  {'sparse MB':>10}  {'H-mat MB':>10}  "
          f"{'sparse MVP':>10}  {'H-mat MVP':>10}  {'rel err':>8}")
    for b in benches:
        print(f"  {b['n_faces']:>6}  "
              f"{b.get('sparse_mem_mb', float('nan')):>10.2f}  "
              f"{b.get('hmat_mem_mb', float('nan')):>10.2f}  "
              f"{b.get('sparse_mvp_ms', float('nan')):>8.2f} ms  "
              f"{b.get('hmat_mvp_ms', float('nan')):>8.2f} ms  "
              f"{b.get('mvp_rel_err_pct', float('nan')):>6.2f}%")
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=Path("tmp/benchmark_hmatrix.log"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/v2/hmatrix_vs_mc.png"))
    args = p.parse_args()
    make_figure(args.log, args.out)


if __name__ == "__main__":
    main()
