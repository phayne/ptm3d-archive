#!/usr/bin/env python3
"""Standalone regression test for scripts/ema_hapke.py (no pytest on this machine).

Run:  python3 scripts/test_ema_hapke.py
Exits 0 if every assertion passes, 1 (with a printed FAIL summary) otherwise.

The numeric expectations here are NOT invented -- the 4-dp alb_point values ARE
the actual `ti<TI>_alb<p>` sweep directory names on disk, and the six-anchor `p`
values are the literature geometric albedos the campaign calibrated to. This test
is what pins ema_hapke.py to the outputs that already exist (see BUGS.md EMA-01).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ema_hapke as H


_FAILURES = []


def check(cond, msg):
    """Record a boolean check; never abort early so the summary lists every failure."""
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f" FAIL  {msg}")
        _FAILURES.append(msg)


def main():
    # --- 1. alb_point 4-dp values == the real on-disk sweep directory names ----------
    # These are the actual `ti<TI>_alb<p>` directory p-strings on disk.
    alb_expect = {
        0.143: [0.0185, 0.0371, 0.0741],
        0.193: [0.0255, 0.0509, 0.1019],
        0.465: [0.0694, 0.1388, 0.2777],
    }
    for w, exp in alb_expect.items():
        got = [round(H.alb_point(w, s)[0], 4) for s in (0.5, 1.0, 2.0)]
        check(got == exp, f"alb_point(w={w}, s in 0.5/1.0/2.0)[0] 4dp == {exp}  (got {got})")

    # --- 2. exact disk_p_from_w(0.143) ------------------------------------------------
    p143 = H.disk_p_from_w(0.143)
    check(abs(p143 - 0.037061) < 5e-6,
          f"disk_p_from_w(0.143) == 0.037061 +/- 5e-6  (got {p143:.6f})")

    # --- 3. six literature anchors: round(p, 3) == published geometric albedo ---------
    anchors = {0.797: 0.313, 0.465: 0.139, 0.416: 0.121,
               0.204: 0.054, 0.193: 0.051, 0.143: 0.037}
    for w, p_lit in anchors.items():
        got = round(H.disk_p_from_w(w), 3)
        check(got == p_lit, f"round(disk_p_from_w({w}), 3) == {p_lit}  (got {got})")

    # --- 4. identity in w at scale == 1.0 ---------------------------------------------
    for w in (0.143, 0.193, 0.204, 0.416, 0.465, 0.797):
        w_consistent = H.alb_point(w, 1.0)[1]
        check(abs(w_consistent - w) < 1e-9,
              f"alb_point({w}, 1.0)[1] == {w} (identity at scale=1)  (got {w_consistent:.12f})")

    # --- 5. round-trip through the inversion ------------------------------------------
    for w in (0.1, 0.3, 0.5, 0.7, 0.9):
        rt = H.w_from_disk_p(H.disk_p_from_w(w))
        check(abs(rt - w) < 1e-9,
              f"w_from_disk_p(disk_p_from_w({w})) == {w} +/- 1e-9  (got {rt:.12f})")

    # --- 6. strict monotonicity of disk_p_from_w on (0, 1) ----------------------------
    grid = [i / 2000.0 for i in range(1, 2000)]  # 0.0005 .. 0.9995
    ps = [H.disk_p_from_w(w) for w in grid]
    mono = all(ps[i + 1] > ps[i] for i in range(len(ps) - 1))
    check(mono, "disk_p_from_w is strictly increasing over a fine grid on (0, 1)")

    # --- summary ----------------------------------------------------------------------
    print()
    if _FAILURES:
        print(f"RESULT: FAIL -- {len(_FAILURES)} assertion(s) failed:")
        for m in _FAILURES:
            print(f"  - {m}")
        return 1
    print("RESULT: PASS -- all ema_hapke assertions hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
