"""Fermi's decisive measurement (BUGS.md EMA-07).

Question: does the w->p opposition convention move the recovered thermal inertia, or does it
only move the reflected-light normalization?

Method: hold the sweep grid FIXED (its albedo axis comes from sweep_point.json on disk and does
not depend on the formula). Vary only `p_nom`, which is the single place `disk_p_from_w` enters
the real fit (ema_ti_fit.py:379). It sets BOTH the injected truth albedo and the Gaussian prior
centre. Run the identical injection-recovery for three conventions:

  verbatim   p = w/8 + 0.49 r0 + 0.196 r0^2          bracket (1+B0)P(0) = 2   [what shipped]
  excluded   p = (r0/2)(1 + r0/3)                    bracket = 1              [true opposition-excluded]
  camera     p = (w/8)[(1+B0)P(0) - 1] + (r0/2)(1+r0/3)  with the body's OWN B0,g  [what the renderer assumes]

Same seeds, same noise realisations across conventions, so differences are the convention alone.
No new simulations: cached sweeps only.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np

PTM = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"
sys.path.insert(0, PTM + "/scripts")
import ema_ti_fit as F                     # noqa: E402
import ema_hapke as H                      # noqa: E402
from ptm_progress import Heartbeat         # noqa: E402

# ---- exact memoisation -------------------------------------------------------------
# load_sweep_grid calls ca_timestep(), parse_ply_facets() and read_sun_vectors() ONCE PER
# CELL. All three depend only on the body's geometry/illumination, which is identical
# across sweep cells (only the thermophysics differs). VERIFIED: direct_solar_0000.bin and
# direct_solar_0149.bin are byte-identical (md5) across ti100_alb{0.0185,0.0371,0.0741};
# same for mesh.ply and body_positions.csv. So caching per body is exact, not an
# approximation -- and it cuts ~4,400 cold OneDrive reads per body down to ~310.
# (Measured: 3 s of CPU in 22 min of wall clock. Pure I/O.)
def _body_of(path):
    return path.split("/thermal/")[1].split("/")[0]


def _memo_by_body(fn):
    cache = {}
    def wrapped(thermal_dir, *a, **k):
        b = _body_of(thermal_dir)
        if b not in cache:
            cache[b] = fn(thermal_dir, *a, **k)
        return cache[b]
    return wrapped


F.ca_timestep = _memo_by_body(F.ca_timestep)
F.read_sun_vectors = _memo_by_body(F.read_sun_vectors)
_orig_ply = F.parse_ply_facets
_ply_cache = {}
def _ply_memo(mesh_ply):
    b = _body_of(mesh_ply)
    if b not in _ply_cache:
        _ply_cache[b] = _orig_ply(mesh_ply)
    return _ply_cache[b]
F.parse_ply_facets = _ply_memo

BODIES = ["623chimaera", "13294rockox", "23871ousha",
          "10253westerwald", "59980moza", "88055ghaf"]
TRUTHS = [56.0, 178.0]                     # the two interior grid nodes the campaign fits
NSEED = int(os.environ.get("NSEED", "12"))


def p0_hg(g):
    """One-term Henyey-Greenstein phase function at zero phase."""
    return (1.0 - g) / (1.0 + g) ** 2


def conventions(w, B0, g):
    r0 = H._r0(w)
    excl = 0.5 * r0 * (1.0 + r0 / 3.0)
    return {
        "verbatim": H.disk_p_from_w(w),
        "excluded": excl,
        "camera": (w / 8.0) * ((1.0 + B0) * p0_hg(g) - 1.0) + excl,
    }


def run_body(body):
    num = F._body_num_from_sweep_root(f"{TROOT}/{body}/sweep")
    hp = F._drv.HAPKE_DEFAULTS[num]
    t0 = time.time()
    grid = F.load_sweep_grid(f"{TROOT}/{body}/sweep")
    load_s = time.time() - t0
    lo, hi = float(grid.albs[0]), float(grid.albs[-1])

    convs = conventions(hp["w"], hp["B0"], hp["g"])
    out = {"body": body, "w": hp["w"], "B0": hp["B0"], "g": hp["g"],
           "grid_alb_range": [lo, hi], "grid_tis": [float(x) for x in grid.tis],
           "load_s": round(load_s, 1), "conventions": {}}

    for cname, p_nom in convs.items():
        inside = lo <= p_nom <= hi
        rec = {"p_nom": p_nom, "inside_grid": bool(inside),
               "clipped_to": None if inside else (lo if p_nom < lo else hi),
               "truths": {}}
        for ti_true in TRUTHS:
            fits = []
            for seed in range(1, NSEED + 1):
                rng = np.random.default_rng(seed)          # SAME seed across conventions
                clean = grid.predict(ti_true, p_nom)       # NB: predict() CLIPS p to the grid
                sigT = np.clip([float(F._noise.ir_sigma_T(T)) for T in clean], 0.3, 40.0)
                obs = clean + rng.normal(0, 1, clean.size) * sigT
                sigma_p = 0.03 * p_nom
                p_vis = p_nom + rng.normal(0, sigma_p)
                r = F.fit_ti(obs, sigT, grid, p_vis, sigma_p)
                fits.append((r["ti"], r["ti_lo"], r["ti_hi"], r["alb"], r["chi2"] / max(r["ndof"], 1)))
            a = np.array(fits)
            rec["truths"][str(int(ti_true))] = {
                "ti_med": float(np.median(a[:, 0])),
                "ti_p16": float(np.percentile(a[:, 0], 16)),
                "ti_p84": float(np.percentile(a[:, 0], 84)),
                "ci_lo_med": float(np.median(a[:, 1])),
                "ci_hi_med": float(np.median(a[:, 2])),
                "alb_med": float(np.median(a[:, 3])),
                "chi2_red_med": float(np.median(a[:, 4])),
            }
        out["conventions"][cname] = rec
    return out


def main():
    only = sys.argv[1:] or BODIES
    results = []
    hb = Heartbeat("/tmp/opposition_ti_diff", unit_total=len(only), label="starting")
    hb.__enter__()
    for i, b in enumerate(only, 1):
        hb.beat(i - 1, label=f"loading grid: {b}")
        print(f"\n=== {b} ===", flush=True)
        try:
            r = run_body(b)
        except SystemExit as e:
            print(f"  SKIP: {e}", flush=True)
            hb.beat(i, label=f"skipped {b}")
            continue
        hb.beat(i, label=f"done {b}")
        results.append(r)
        lo, hi = r["grid_alb_range"]
        print(f"  w={r['w']}  B0={r['B0']}  g={r['g']}   grid albedo axis [{lo:.4f}, {hi:.4f}]"
              f"   (grid load {r['load_s']}s)", flush=True)
        for cname, rec in r["conventions"].items():
            tag = "in-grid" if rec["inside_grid"] else f"OUTSIDE -> clipped to {rec['clipped_to']:.4f}"
            print(f"  {cname:9s} p_nom={rec['p_nom']:.4f}  [{tag}]", flush=True)
            for t, d in rec["truths"].items():
                print(f"      truth TI={t:>3}  ->  TI_fit={d['ti_med']:6.1f} "
                      f"[{d['ti_p16']:.1f},{d['ti_p84']:.1f}]  "
                      f"1sig CI [{d['ci_lo_med']:.0f},{d['ci_hi_med']:.0f}]  "
                      f"alb={d['alb_med']:.4f}  chi2/ndof={d['chi2_red_med']:.2f}", flush=True)

    dst = "/tmp/opposition_ti_diff.json"
    json.dump(results, open(dst, "w"), indent=2)

    print("\n" + "=" * 92)
    print("VERDICT TABLE — recovered TI (median of %d seeds) by opposition convention" % NSEED)
    print("=" * 92)
    print(f"{'body':16s} {'truth':>6s} {'verbatim':>10s} {'excluded':>10s} {'camera':>10s}"
          f" {'Δ(excl-verb)':>13s} {'Δ(cam-verb)':>12s}")
    for r in results:
        for t in ("56", "178"):
            v = r["conventions"]["verbatim"]["truths"][t]["ti_med"]
            e = r["conventions"]["excluded"]["truths"][t]["ti_med"]
            c = r["conventions"]["camera"]["truths"][t]["ti_med"]
            print(f"{r['body']:16s} {t:>6s} {v:10.1f} {e:10.1f} {c:10.1f}"
                  f" {e-v:+13.1f} {c-v:+12.1f}")
    n_out = sum(1 for r in results if not r["conventions"]["camera"]["inside_grid"])
    print(f"\ncamera-consistent p_nom falls OUTSIDE the swept albedo range for {n_out}/{len(results)} bodies")
    print(f"wrote {dst}")
    hb.__exit__(None, None, None)


if __name__ == "__main__":
    main()
