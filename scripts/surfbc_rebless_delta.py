#!/usr/bin/env python3
"""Attribute a surface-BC re-bless to its two independent causes.

This change is NOT byte-identical, and two separate things moved: the surface
energy balance (seed / relaxation / exit criterion / clamp removal) and the
Stefan-Boltzmann constant. They must be reported as two numbers, not one, so
run three arms of the SAME config and difference them:

    A  baseline                       (old surface BC, five sigma values)
    B  baseline + sigma migration     (old surface BC, one sigma)
    C  the branch                     (new surface BC, one sigma)

    sigma contribution      = B - A
    surface-BC contribution = C - B

Usage:
    python3 scripts/surfbc_rebless_delta.py <root>   # root holds A_baseline/,
                                                     # B_sigma_only/, C_full/
Each of those holds <body>/statistics/*.bin in the engine's little-endian
float32-with-4-byte-header format.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIELDS = ("temperature_max", "temperature_avg", "temperature_min")


def load(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw[4:], dtype="<f4").astype(np.float64)


def stats(d: np.ndarray) -> str:
    return (f"mean {d.mean():+.3e}  median {np.median(d):+.3e}  "
            f"max|.| {np.abs(d).max():.3e}  min {d.min():+.3e}  max {d.max():+.3e}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    arms = {"A": root / "A_baseline", "B": root / "B_sigma_only", "C": root / "C_full"}
    for name, p in arms.items():
        if not p.is_dir():
            print(f"error: missing arm {name} at {p}", file=sys.stderr)
            return 1

    bodies = sorted({p.name for p in arms["A"].iterdir() if p.is_dir()})
    for body in bodies:
        print(f"\n================ {body} ================")
        for field in FIELDS:
            try:
                a = load(arms["A"] / body / "statistics" / f"{field}.bin")
                b = load(arms["B"] / body / "statistics" / f"{field}.bin")
                c = load(arms["C"] / body / "statistics" / f"{field}.bin")
            except FileNotFoundError as exc:
                print(f"  {field}: MISSING ({exc.filename}) -- not compared")
                continue
            if not (a.size == b.size == c.size):
                print(f"  {field}: SIZE MISMATCH {a.size}/{b.size}/{c.size} -- "
                      "not compared")
                continue
            print(f"  {field}  ({a.size} faces, baseline range "
                  f"{a.min():.3f} .. {a.max():.3f} K)")
            print(f"    sigma      (B-A): {stats(b - a)}")
            print(f"    surface BC (C-B): {stats(c - b)}")
            print(f"    total      (C-A): {stats(c - a)}")
            print(f"    byte-identical?  B==A {not (b - a).any()}   "
                  f"C==B {not (c - b).any()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
