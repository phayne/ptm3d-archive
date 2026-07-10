#!/usr/bin/env python3
"""Rocky, 2026-07-06: TASK B (final step) -- build the fuller, uncertainty-
quantified fit_recovery.png/pdf for 23871 Ousha in pretty-plots style, from the
combined ensemble-fit summary written by /tmp/ema_ousha_extra_offgrid.py's
`fit` subcommand: 15 true-TI cases total spanning the 32-316 SI range (5 reuse
existing sweep simulations directly at those TI values; the other 10 are
independent forward thermal simulations at additional TI values, 2 run
previously on 2026-07-04, 8 new this pass).

Per Paul (2026-07-06 relabel pass): plot ALL 15 points UNIFORMLY -- one marker,
one color, one series -- no distinction/legend/wording by provenance. Clean-
figure rule: axis labels+units, legend (1:1 line + the point series), short
title only on the figure. Method/caveat detail goes in fit_recovery.caption.md
(also with no provenance-category wording, per the same instruction).
"""
import json, sys
import numpy as np

PTM_SCRIPTS = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/scripts"
sys.path.insert(0, PTM_SCRIPTS)
import ema_pretty_figs as epf
import matplotlib.pyplot as plt

TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"
NUM, NAME = "23871", "Ousha"
BODYDIR = f"{NUM}{NAME.lower()}"
OUT = f"{TROOT}/{BODYDIR}/ti_fit"
COMBINED = f"{OUT}/offgrid_extra/fit_results_summary.json"

summary = json.load(open(COMBINED))
items = sorted(summary.items(), key=lambda kv: kv[1]['truth_ti'])

epf.setup_style()
fig, ax = plt.subplots(figsize=(7.0, 7.0))

all_ti = [v['truth_ti'] for v in summary.values()]
lo = min(all_ti) * 0.6
hi = max(all_ti) * 1.6

ax.plot([lo, hi], [lo, hi], 'k:', lw=1.5, label='1:1', zorder=1)

first = True
for tag, v in items:
    yerr = np.array([[v['median'] - v['p16']], [v['p84'] - v['median']]])
    ax.errorbar(v['truth_ti'], v['median'], yerr=yerr, fmt='o', ms=9,
                color='#3778bf', capsize=3.5, elinewidth=1.5, alpha=0.9,
                label='Recovered TI (68% CI)' if first else None, zorder=3)
    first = False

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("True thermal inertia (SI)")
ax.set_ylabel("Recovered thermal inertia (SI, median + 68% CI)")
ax.set_title(f"({NUM}) {NAME} — TI recovery under IR NEDT noise")
ax.grid(alpha=0.2, which='both', linewidth=0.5)
ax.legend(loc='upper left', frameon=True, fontsize=11.5)

plt.tight_layout()
written = epf.save_pub(fig, OUT + "/fit_recovery")
plt.close(fig)
print("wrote", [str(p) for p in written])

# ---- caption (no provenance-category wording, per Paul's instruction) ----
n_total = len(summary)
n_ci = sum(1 for v in summary.values() if v.get('truth_in_ci'))

cap = [
    f"# ({NUM}) {NAME} thermal-inertia recovery",
    "",
    f"200-seed noise-ensemble fit ({n_total} true-TI cases total, {n_ci}/{n_total} recover "
    "the truth within the 68% [16,84] percentile CI) against the real 15-point (TI x albedo) "
    "sweep grid used as the fitting reference. Five cases reuse existing sweep simulations "
    "directly at TI = 32, 56, 100, 178, 316 SI; the remaining ten are independent forward "
    "thermal simulations at additional TI values (two run previously on 2026-07-04, eight new "
    "this pass).",
    "",
]
for tag, v in items:
    cap.append(f"- TI={v['truth_ti']:.0f} -> median {v['median']:.1f} "
               f"[{v['p16']:.1f},{v['p84']:.1f}] SI, truth-in-CI={v['truth_in_ci']}")
cap += [
    "",
    "Method: same as the 2026-07-04 validation (memory "
    "ema-ousha-terminatorfix-ramextract-validation) -- IR NEDT sigma(T) noise model, geometric "
    "albedo free with a 3%-of-nominal Gaussian prior from the (synthetic) CNAC constraint, "
    "Delta chi^2 profile for the TI 1-sigma interval, 200 noise realizations per case.",
    "",
    "The eight new cases (this pass) were run on a git-commit-79f091e-PINNED worktree rebuild "
    "of ptm_thermal_sim (the live $PTM build has substantial uncommitted diffs from unrelated "
    "in-progress work as of 2026-07-06) -- validated byte-identical against the archived "
    "TI=75 case before use (0/21 sampled per-timestep temperature files differed). CPU, "
    "PTM_RAM_ILLUM_MAX_GB=40 (RAM-extract), chi=2.7, radiation off, Qb=0 -- identical settings "
    "to the existing sweep simulations. Camera/IR rendering was skipped for the eight new "
    "cases (QA-only in the existing pipeline; the recovery fit itself uses the real "
    "thermal-model curve + the analytic NEDT model directly, not rendered pixels).",
    "",
    "Caveat (carried from the original 2026-07-04 validation): the sweep varies SURFACE TI via "
    "ks holding kd fixed (deep TI ~775 SI); at the highest true TI values the diurnal skin "
    "depth approaches the ks->kd transition, so recovered TI at the high end (~250-316 SI) is "
    "an effective surface TI, not an unbounded bulk value.",
]
open(OUT + "/fit_recovery.caption.md", "w").write("\n".join(cap) + "\n")
print("wrote", OUT + "/fit_recovery.caption.md")
