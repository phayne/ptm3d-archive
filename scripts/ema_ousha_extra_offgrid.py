#!/usr/bin/env python3
"""Rocky, 2026-07-06: TASK B -- fill the 23871 Ousha TI-recovery grid with several
more injected-TI forward cases (32-316 SI range), then build a combined recovery
dataset (5 grid-node self-checks + 2 existing off-grid [75,200] + N NEW off-grid
truths) for a fuller, uncertainty-quantified fit_recovery.png.

CRITICAL setup (per task brief + memory ema-ti-fit-hardcoded-albedo-bug /
ema-ousha-terminatorfix-ramextract-validation):
  - binary: a FRESH worktree build pinned at git commit 79f091e (NOT the live
    $PTM/build/bin/ptm_thermal_sim, which has ~1500 lines of uncommitted
    unrelated diffs from other agents' in-progress work as of 2026-07-06).
    VALIDATED byte-identical against the archived offgrid/ti075 case (0/21
    sampled per-timestep temperature files differ, temperature_avg.bin
    max|diff|=0) after fixing a build-time SPICE_KERNELS_DIR bake-in (see
    session note / memory for detail).
  - PTM_RAM_ILLUM_MAX_GB=40 (RAM-extract override, required for this 81,920-face
    ico6 mesh; set in THIS process's environ before any subprocess call).
  - Ousha-correct albedo/config: reuses /tmp/ema_sweep_driver.py's own
    resolve_target()/build_thermal_config()/HAPKE_DEFAULTS['23871'] and the
    Ousha-parametrized /tmp/ema_ti_fit.py fit machinery (already patched for the
    hardcoded-'623' bug) -- NOT ema_offgrid_fu2.py's Chimaera-hardcoded original.
  - CPU only (use_gpu=False, baked into build_thermal_config already), chi=2.7,
    radiation off, Qb=0 -- identical to the existing sweep + off-grid cases.
  - Does NOT rebuild/touch the existing 15-cell sweep grid (thermal/23871ousha/
    sweep/) -- reused as-is as the fit reference/interpolant.

Subcommands:
  sims   -- run the NEW injected-TI forward thermal sims only (no camera QC, not
            needed for the recovery fit -- see docstring note below)
  fit    -- combined ensemble-noise fit (200 seeds) of: the 5 sweep grid nodes'
            OWN nominal-albedo curves (self-consistency anchor), the 2 EXISTING
            off-grid cases (75,200; reused, not re-run), and the NEW off-grid
            cases -- all against the UNCHANGED 15-point sweep grid.

NOTE on camera QC: the original off-grid script's camera IR render + noise pass
is a VISUAL QA product only -- ensemble_fit() (both there and here) derives its
noisy "obs" directly from the analytic IR NEDT sigma model applied to the REAL
thermal-model clean curve, not from rendered/noised camera pixels. Skipped here
for the 8 new cases to save wall time; does not affect the recovery numbers.
"""
import os, sys, glob, json, importlib.util
import numpy as np

os.environ["PTM_RAM_ILLUM_MAX_GB"] = "40"

def _load(mod, path):
    s = importlib.util.spec_from_file_location(mod, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m

_drv = _load("ema_driver", "/tmp/ema_sweep_driver.py")
_fit = _load("ema_fit", "/tmp/ema_ti_fit.py")          # Ousha-albedo-bug-patched scratch copy
_noise = _load("ema_noise", "/tmp/ema_camera_noise.py")

import yaml, spiceypy as sp

PINNED_TBIN = "/tmp/ptm_79f091e_pinned/build/bin/ptm_thermal_sim"
NUM, NAME = "23871", "Ousha"
BODYDIR = f"{NUM}{NAME.lower()}"
TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"
GRID_TI = (32.0, 56.0, 100.0, 178.0, 316.0)
EXISTING_OFFGRID_TI = (75.0, 200.0)
NEW_TI = [45.0, 65.0, 90.0, 130.0, 150.0, 225.0, 250.0, 280.0]

EXTRA_ROOT = f"{TROOT}/{BODYDIR}/offgrid_extra"
FIT_OUT = f"{TROOT}/{BODYDIR}/ti_fit/offgrid_extra"


def build(ti):
    R = _drv.resolve_target(NUM, NAME)
    ks = _drv.ti_to_ks(ti)
    p, w = _drv.alb_point(R['hp']['w'], 1.0)          # nominal albedo (scale=1.0), same as existing off-grid
    outdir = f"{EXTRA_ROOT}/ti{ti:03.0f}"
    os.makedirs(outdir, exist_ok=True)
    win_h = 1.5 * R['period']; start_et = R['etm'] - 0.75 * R['period'] * 3600.0
    start_time = sp.et2utc(start_et, 'ISOC', 0).replace('Z', '')
    tconf = _drv.build_thermal_config(
        body_name=R['gb'].get('name', NAME), bcfg=R['bcfg'], bdir=R['bdir'], naif=R['naif'],
        ra=R['ra'], dec=R['dec'], radius=R['radius'], period=R['period'], use_mesh=R['use_mesh'],
        shape_mesh_file=R['shape_mesh_file'], shape_mesh_scale=R['shape_mesh_scale'],
        gb_orbit=R['gb']['orbit'], start_time=start_time, duration_days=win_h / 24.0,
        outdir=outdir, ks=ks, geom_albedo=p, hapke_w=w)
    cfgp = outdir + "/config.yaml"
    yaml.safe_dump(tconf, open(cfgp, 'w'), sort_keys=False)
    json.dump(dict(target=f"{NUM}{NAME}", naif=R['naif'], ti=ti, ks=ks, geom_albedo=p, hapke_w=w,
                    rhos=_drv.TI_RHOS, cp0=_drv.TI_CP0, binary="79f091e (pinned worktree rebuild, "
                    "byte-identical-verified against archived offgrid/ti075)", use_gpu=False, chi=2.7,
                    config=cfgp, offgrid=True, ram_illum_max_gb=40,
                    truth_note="off-grid independent forward truth (NOT a sweep-grid node); "
                               "Task-B fill-in case"),
              open(outdir + "/offgrid_point.json", 'w'), indent=2)
    return R, outdir, cfgp, p, w


def run_sims():
    os.makedirs(EXTRA_ROOT, exist_ok=True)
    logdir = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/logs"
    for ti in NEW_TI:
        R, outdir, cfgp, p, w = build(ti)
        tag = f"ti{ti:03.0f}"
        print(f"[{tag}] NAIF {R['naif']} CA {R['ca_utc']} ks={_drv.ti_to_ks(ti):.6g} "
              f"p={p:.4f} w={w:.4f} -> {outdir}", flush=True)
        if _drv.thermal_done(outdir):
            print(f"[{tag}] thermal already complete -> skip", flush=True)
            continue
        logf = f"{logdir}/ema_{NUM}_offgrid_extra_{tag}.log"
        print(f"[{tag}] thermal run starting (PINNED binary {PINNED_TBIN}) ... (log {logf})", flush=True)
        rc = _drv.run([PINNED_TBIN, "--config", cfgp], logf)
        ok = os.path.exists(outdir + "/statistics/temperature_avg.bin")
        print(f"[{tag}] thermal rc={rc} done={ok}", flush=True)
        if not ok:
            print(f"[{tag}] THERMAL FAILED -- see {logf}", flush=True)


# ---------------- combined ensemble fit ----------------
def _ensemble_one(tag, outdir, ti, grid, p_nom, sigma_p, n_seeds, seed0=1000):
    cur = _fit.extract_local_time_curve(outdir)
    clean = cur['T']
    valid0 = np.isfinite(clean)
    tis_rec = []
    recs = []
    for s in range(n_seeds):
        seed = seed0 + int(round(ti)) * 100000 + s
        rng = np.random.default_rng(seed)
        sigT = np.clip([float(_noise.ir_sigma_T(T)) if np.isfinite(T) else np.nan for T in clean],
                        0.3, 40.0)
        obs = clean.copy()
        obs[valid0] = clean[valid0] + rng.normal(0, 1, valid0.sum()) * sigT[valid0]
        p_vis = p_nom + rng.normal(0, sigma_p)
        r = _fit.fit_ti(obs, sigT, grid, p_vis, sigma_p)
        tis_rec.append(r['ti'])
        if s < 5:
            recs.append(dict(seed=seed, obs=obs.tolist(), sigT=sigT.tolist(), ti_fit=r['ti']))
    tis_rec = np.array(tis_rec)
    med = float(np.median(tis_rec)); lo16 = float(np.percentile(tis_rec, 16))
    hi84 = float(np.percentile(tis_rec, 84))
    in_ci = lo16 <= ti <= hi84
    print(f"[{tag}] truth TI={ti:.0f} -> median={med:.1f} [16%={lo16:.1f}, 84%={hi84:.1f}] "
          f"truth-in-CI={in_ci} (n_seeds={n_seeds})", flush=True)
    return dict(tag=tag, truth_ti=ti, median=med, p16=lo16, p84=hi84, truth_in_ci=bool(in_ci),
                n_seeds=n_seeds, all_fits=tis_rec.tolist(), clean_curve=clean.tolist(),
                lst=cur['lst'].tolist(), sample_realizations=recs)


def ensemble_fit_combined(n_seeds=200, seed0=1000, sigma_p_frac=0.03):
    os.makedirs(FIT_OUT, exist_ok=True)
    sweep_root = f"{TROOT}/{BODYDIR}/sweep"
    grid = _fit.load_sweep_grid(sweep_root)
    p_nom = _fit._drv.disk_p_from_w(_fit._drv.HAPKE_DEFAULTS[NUM]['w'])
    sigma_p = sigma_p_frac * p_nom
    results = {}

    # (1) 5 grid-node self-consistency anchors (nominal-albedo sweep dirs, REUSED, no new sim)
    for ti in GRID_TI:
        pnom_str = f"{p_nom:.4f}"
        outdir = f"{sweep_root}/ti{ti:g}_alb{pnom_str}"
        if not os.path.isdir(outdir):
            # fall back to a glob in case of float-format mismatch
            cand = sorted(glob.glob(f"{sweep_root}/ti{ti:g}_alb*"))
            outdir = cand[len(cand)//2] if cand else None  # pick the nominal (middle) albedo point
        if outdir is None or not _drv.thermal_done(outdir):
            print(f"SKIP grid node TI={ti}: {outdir} not complete"); continue
        tag = f"grid_ti{ti:03.0f}"
        r = _ensemble_one(tag, outdir, ti, grid, p_nom, sigma_p, n_seeds, seed0)
        r['kind'] = 'grid_node'
        results[tag] = r

    # (2) 2 existing off-grid cases (REUSED, no new sim)
    for ti in EXISTING_OFFGRID_TI:
        outdir = f"{TROOT}/{BODYDIR}/offgrid/ti{ti:03.0f}"
        if not _drv.thermal_done(outdir):
            print(f"SKIP existing off-grid TI={ti}: {outdir} not complete"); continue
        tag = f"offgrid_ti{ti:03.0f}"
        r = _ensemble_one(tag, outdir, ti, grid, p_nom, sigma_p, n_seeds, seed0)
        r['kind'] = 'offgrid_existing'
        results[tag] = r

    # (3) NEW off-grid cases (Task B fill-in, this script's run_sims())
    for ti in NEW_TI:
        outdir = f"{EXTRA_ROOT}/ti{ti:03.0f}"
        if not _drv.thermal_done(outdir):
            print(f"SKIP new off-grid TI={ti}: {outdir} not complete"); continue
        tag = f"new_ti{ti:03.0f}"
        r = _ensemble_one(tag, outdir, ti, grid, p_nom, sigma_p, n_seeds, seed0)
        r['kind'] = 'offgrid_new'
        results[tag] = r

    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ('all_fits',)}
               for k, v in results.items()}, open(FIT_OUT + "/fit_results_summary.json", 'w'), indent=2)
    json.dump(results, open(FIT_OUT + "/fit_results_full.json", 'w'), indent=2)
    print("wrote", FIT_OUT + "/fit_results_summary.json,", FIT_OUT + "/fit_results_full.json")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sims", "fit"])
    ap.add_argument("--n-seeds", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "sims":
        run_sims()
    else:
        ensemble_fit_combined(n_seeds=a.n_seeds)
