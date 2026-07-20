#!/usr/bin/env python3
"""Compare the H1 beaming diagnostic runs (A baseline / B albedo-fix / C full beaming).

For each run: central-floor summer-window max temperature (statistics/temperature_max.bin)
and delivered peak indirect flux (PTM_DUMP_QIND_SERIES dump). The central floor = faces
within R_FLOOR of the core-mesh planar center that are cold in the baseline (PSR). This
isolates whether the beaming/albedo corrections add flux to the floor and whether the
floor temperature responds (the 'flux added but under-responds' test).

Usage: python3 h1_diag_compare.py
"""
from __future__ import annotations
import sys, struct
from pathlib import Path
import numpy as np

REPO = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm")
OUT = Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output")
MESH = REPO / "data/dem/validation/meshes/haworth_radcore_r25.ply"
R_FLOOR_KM = 2.0           # central-floor radius from the core planar center
PSR_TMAX_K = 120.0         # baseline Tmax below this = shadowed floor candidate

# (label, output-dir tag, qind dump path). First entry is the baseline.
RUNS = [
    ("base (rms0)",  "hmfix_diag_e95_base", "/tmp/qind_e95_base.bin"),
    ("rms30 beam",   "hmfix_diag_e95_r30",  "/tmp/qind_e95_r30.bin"),
    ("rms40 beam",   "hmfix_diag_e95_r40",  "/tmp/qind_e95_r40.bin"),
]



def read_qidt(path: Path):
    """PTM_DUMP_QIND_SERIES: magic 'QIDT', u32 nf, u32 ns, nf*ns float32 (face-major)."""
    with open(path, "rb") as f:
        magic, nf, ns = struct.unpack("<III", f.read(12))
        assert magic == 0x51494454, f"bad magic {magic:#x}"
        a = np.frombuffer(f.read(nf * ns * 4), dtype="<f4").reshape(nf, ns)
    return a                      # (n_faces, n_steps) indirect flux W/m^2


def load_tmax(run_dir: Path):
    p = run_dir / "statistics/temperature_max.bin"
    return np.fromfile(p, dtype=np.float32) if p.exists() else None


FLOOR_TMAX_K = 55.0        # deep-PSR central floor = baseline Tmax below this
                           # (walls/rim are >100 K; avoids needing the mesh geometry)


def main():
    tmax = {lbl: load_tmax(OUT / tag) for (lbl, tag, _q) in RUNS}
    base_lbl = RUNS[0][0]
    if tmax[base_lbl] is None:
        print("baseline statistics not present yet:", OUT / RUNS[0][1]); return
    nf = tmax[base_lbl].size
    floor = tmax[base_lbl] < FLOOR_TMAX_K     # deep shadowed floor (coldest faces)
    print(f"core faces={nf}  deep-PSR floor (baseline Tmax<{FLOOR_TMAX_K:.0f}K)={floor.sum()}")
    print()

    print("Central-floor SUMMER-MAX temperature (K):")
    print(f"  {'run':<12}{'median':>8}{'p90':>8}{'max':>8}   vs base")
    base = None
    for lbl, tag, _q in RUNS:
        t = tmax[lbl]
        if t is None:
            print(f"  {lbl:<12}  (not present yet)"); continue
        fv = t[floor]
        med, p90, mx = np.median(fv), np.percentile(fv, 90), fv.max()
        if base is None:
            base = med
        d = f"  d_med={med-base:+.2f}"
        print(f"  {lbl:<12}{med:8.2f}{p90:8.2f}{mx:8.2f}{d}")
    print()

    print("Central-floor delivered PEAK indirect flux (W/m^2, max over summer window):")
    qbase = None
    for lbl, tag, qpath in RUNS:
        qp = Path(qpath)
        if not qp.exists():
            print(f"  {lbl:<12} qind dump not present"); continue
        peak = read_qidt(qp).max(axis=1)
        fl = floor[:peak.size]
        med, mx = np.median(peak[fl]), peak[fl].max()
        if qbase is None:
            qbase = med
        print(f"  {lbl:<12} median={med:.3f} ({100*(med/qbase-1):+.0f}%)  max={mx:.3f}")
    print()
    print("INTERPRETATION: how much does higher RMS slope (more hot Sun-facing micro-")
    print("facets) raise the delivered floor flux and the floor Tmax?")


if __name__ == "__main__":
    main()
