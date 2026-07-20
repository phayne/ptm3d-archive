#!/usr/bin/env python
"""Round-6 FINAL coverage-accumulation renders for (269) Justitia proxops
(Ada, 2026-07-05). Regenerates the 4 per-phase coverage-accum animations +
the full-period highest-resolution-wins animation with the round-6
sub-spacecraft-tracking camera + green footprint box (default-on), via the
PARALLEL two-/three-stage renderer in ``ema_pretty_figs`` (byte-identical
frame content to the serial path — verified), at 1600x900 with a size-target
CRF fallback so every MP4 lands < ~95 MB. Predecessors are archived
non-destructively by the underlying renderers.

CRITICAL (macOS multiprocessing 'spawn'): the parallel renderer creates
spawn subprocesses, so the entry point MUST be under ``if __name__ ==
'__main__'`` — otherwise each spawn child re-imports an unguarded __main__ and
recursively re-launches the whole render (the freeze_support recursion). This
script is guarded, AND runs each render in its OWN fresh subprocess: that
guarantees a single Cocoa/NSOpenGL context per process (concurrent GL contexts
crash on this Mac — verified) and isolates each render.

Usage:
  python ema_render_justitia_coverage_round6.py                 # run all 5
  python ema_render_justitia_coverage_round6.py <name>          # one render
      <name> in {orbit_high, orbit_mid, orbit_low, descent, fullperiod}
"""
import sys
import os
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RENDERS = ["orbit_high", "orbit_mid", "orbit_low", "descent", "fullperiod"]


def _run_one(name):
    """Run a single render IN THIS (guarded) process and print a JSON result."""
    import ema_pretty_figs as E

    if name == "fullperiod":
        # Non-destructively archive the prior full-period raster .npz (the
        # renderer overwrites it in place; round 6 is a view-only change so the
        # content is identical, but keep the predecessor anyway).
        out_dir = E.camera_dir("269justitia") / "proxops" / "fullperiod"
        raster = out_dir / "fullperiod_coverage_raster_truegsd.npz"
        if raster.exists():
            from datetime import datetime, timezone
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raster.rename(raster.with_name(
                f"{raster.stem}_preRound6ParallelReRender_{stamp}{raster.suffix}"))
        res = E.make_justitia_proxops_fullperiod_animation(parallel_workers=8)
        cov = res
    else:
        res = E.make_justitia_proxops_phase_animations(name, parallel_workers=8)
        cov = res["coverage_accum_detail"]

    out = {
        "name": name,
        "out_mp4": str(cov["out_mp4"]),
        "archived": str(cov.get("archived_legacy_mp4")),
        "n_frames": cov.get("n_frames"),
        "texture_width": cov["texture_width"],
        "resolution": f"{cov['target_width_px']}x{cov['target_height_px']}",
        "final_crf": cov["final_crf"],
        "final_size_mb": round(cov["final_size_mb"], 2),
        "area_pct": round(cov["area_pct"], 4),
        "facet_pct_legacy": round(cov["facet_pct_legacy"], 4),
        "n_texels_touched": int(cov["n_texels_touched"]),
        "wall_extract_s": round(cov["wall_extract_s"], 1),
        "wall_render_s": round(cov["wall_render_s"], 1),
        "wall_savefig_s": round(cov["wall_savefig_s"], 1),
        "wall_encode_s": round(cov["wall_encode_s"], 1),
        "wall_total_s": round(cov["wall_total_s"], 1),
        "thermal_ext": cov["thermal_ext"],
    }
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return out


def _orchestrate(names):
    """Run each render as a fresh subprocess (one GL context per process)."""
    results = []
    for name in names:
        print(f"\n===== rendering {name} =====", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), name],
            capture_output=True, text=True)
        wall = time.time() - t0
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("RESULT_JSON ")), None)
        if proc.returncode != 0 or line is None:
            print(f"  FAILED {name} rc={proc.returncode} wall={wall:.0f}s", flush=True)
            print("  --- stderr tail ---", flush=True)
            print("\n".join(proc.stderr.splitlines()[-15:]), flush=True)
            results.append({"name": name, "FAILED": True})
            continue
        r = json.loads(line[len("RESULT_JSON "):])
        r["subprocess_wall_s"] = round(wall, 1)
        results.append(r)
        print(f"  OK {name}: {r['n_frames']} frames, {r['resolution']}, "
              f"tex={r['texture_width']}, {r['final_size_mb']} MB (crf {r['final_crf']}), "
              f"area_pct={r['area_pct']}, wall={wall:.0f}s "
              f"(extract {r['wall_extract_s']} / GL {r['wall_render_s']} / "
              f"savefig {r['wall_savefig_s']} / encode {r['wall_encode_s']})", flush=True)

    print("\n===== SUMMARY =====", flush=True)
    all_under = True
    for r in results:
        if r.get("FAILED"):
            print(f"  {r['name']:12s}  FAILED"); all_under = False; continue
        under = r["final_size_mb"] < 100.0
        all_under &= under
        print(f"  {r['name']:12s}  {r['n_frames']:>4} f  {r['resolution']}  "
              f"tex={r['texture_width']:>5}  {r['final_size_mb']:>6.2f} MB "
              f"{'<100 OK' if under else '>=100 FAIL'}  wall={r['subprocess_wall_s']:.0f}s  "
              f"area_pct={r['area_pct']}")
    print(f"\n  ALL < 100 MB: {'YES' if all_under else 'NO'}", flush=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _run_one(sys.argv[1])
    else:
        _orchestrate(RENDERS)
