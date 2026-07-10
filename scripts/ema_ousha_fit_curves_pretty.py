#!/usr/bin/env python3
"""Rocky, 2026-07-06: TASK A -- restyle 23871 Ousha's ti_fit/fit_curves.png with
pretty-plots (setup_style() + pretty_plots.planetary), SAME content/data as the
existing figure (Paul likes this one): per-truth panels showing the clean grid
curve, a noisy synthetic IR observation (NEDT-derived error bars), and the best
fit curve, for truth-TI = 56 and 178 SI (the two INTERIOR grid nodes used by the
existing ti_fit/fit_results.json).

Reproducibility: re-derives the SAME arrays (grid, clean, noisy obs, best fit)
using the identical Ousha-parametrized fit tool (/tmp/ema_ti_fit.py, the
albedo-bug-patched scratch copy -- NOT the tracked scripts/ema_ti_fit.py, which
still hardcodes '623'), the SAME sweep root, SAME default seed=1, and the SAME
--truth-ti 56 178 the existing fit_results.json was produced with. Verifies its
own regenerated fit_results.json is numerically identical to the on-disk one
before touching any figure, as a do-no-harm check.

Clean-figure rule (2026-07-06, Paul): axis labels+units, legend, short title
only on the figure. Caveats (grid-interpolation caveat, synthetic-truth
methodology, recovered TI+CI, chi2/ndof) go in a separate fit_curves.caption.md.
"""
import os, sys, json, importlib.util
import numpy as np

def _load(mod, path):
    s = importlib.util.spec_from_file_location(mod, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m

_fit = _load("ema_fit", "/tmp/ema_ti_fit.py")

PTM_SCRIPTS = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/scripts"
sys.path.insert(0, PTM_SCRIPTS)
import ema_pretty_figs as epf

TROOT = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/EMA/science/thermal"
NUM, NAME = "23871", "Ousha"
BODYDIR = f"{NUM}{NAME.lower()}"
SWEEP_ROOT = f"{TROOT}/{BODYDIR}/sweep"
OUT = f"{TROOT}/{BODYDIR}/ti_fit"
TRUTH_TI = [56.0, 178.0]
SEED = 1
VIS_SIGMA_FRAC = 0.03

grid = _fit.load_sweep_grid(SWEEP_ROOT)
rng = np.random.default_rng(SEED)
body_num = NUM
body_key = body_num if body_num in _fit._drv.HAPKE_DEFAULTS else '623'
p_nom = _fit._drv.disk_p_from_w(_fit._drv.HAPKE_DEFAULTS[body_key]['w'])

results = []
for ti_true in TRUTH_TI:
    clean = grid.predict(ti_true, p_nom)
    sigT = np.clip([float(_fit._noise.ir_sigma_T(T)) for T in clean], 0.3, 40.0)
    obs = clean + rng.normal(0, 1, clean.size) * sigT
    sigma_p = VIS_SIGMA_FRAC * p_nom
    p_vis = p_nom + rng.normal(0, sigma_p)
    r = _fit.fit_ti(obs, sigT, grid, p_vis, sigma_p)
    r.update(ti_true=ti_true, obs=obs, clean=clean, sigT=sigT)
    results.append(r)
    print(f"truth TI={ti_true:.0f} -> TI_fit={r['ti']:.0f} [{r['ti_lo']:.0f},{r['ti_hi']:.0f}] "
          f"alb={r['alb']:.4f}+-{r['alb_sig']:.4f} chi2/ndof={r['chi2']:.1f}/{r['ndof']}")

# ---- do-no-harm check: regenerated fit_results.json must match the on-disk one ----
new_summary = [{k: r[k] for k in ('ti_true', 'ti', 'ti_lo', 'ti_hi', 'alb', 'alb_sig', 'chi2', 'ndof')}
               for r in results]
old_summary = json.load(open(OUT + "/fit_results.json"))
match = all(
    all(abs(new_summary[i][k] - old_summary[i][k]) < 1e-6 for k in new_summary[i])
    for i in range(len(new_summary))
)
print(f"do-no-harm check: regenerated fit_results.json matches on-disk exactly: {match}")
if not match:
    print("!!! MISMATCH -- printing both for inspection:")
    print("new:", json.dumps(new_summary, indent=2))
    print("old:", json.dumps(old_summary, indent=2))
    sys.exit(1)

# ---- pretty-plots figure ----
epf.setup_style()
import matplotlib.pyplot as plt
from pretty_plots import planetary

fig, axs = plt.subplots(1, len(results), figsize=(6.6 * len(results), 5.0))
axs = np.atleast_1d(axs)
body_label = f"({NUM}) {NAME}"
for ax, r in zip(axs, results):
    ax.plot(grid.lst, r['clean'], '-', lw=2.2, color='0.15', label="True", zorder=2)
    ax.errorbar(grid.lst, r['obs'], yerr=r['sigT'], fmt='o', ms=4.5, color='#c1272d',
                ecolor='#e8a2a5', elinewidth=1.4, capsize=2.5, alpha=0.9,
                label="Synthetic IR obs (NEDT)", zorder=3)
    ax.plot(grid.lst, grid.predict(r['ti'], r['alb']), '--', color='#3778bf', lw=2.2,
            label="Best fit", zorder=4)
    ax.set_xlabel("Local solar time (h)")
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title(f"{body_label} — true TI = {r['ti_true']:.0f} SI")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="best", frameon=True, fontsize=12)

plt.tight_layout()
written = epf.save_pub(fig, OUT + "/fit_curves")
plt.close(fig)
print("wrote", [str(p) for p in written])

# ---- caption file (clean-figure rule: caveats/method notes off-figure) ----
cap_lines = [
    f"# {body_label} thermal-inertia fit — synthetic-truth diurnal curves",
    "",
    "Injection-recovery test of the TI x albedo fitting pipeline on the real 15-point "
    f"({NUM}) {NAME} sweep grid (TI = [32, 56, 100, 178, 316] SI x 3 albedo points). For "
    "each panel, a synthetic 'truth' diurnal curve is drawn from the grid at the labeled "
    "true TI and nominal albedo, a single noisy realization is generated using the IR "
    "NEDT sigma(T) model, and the pipeline is fit back against the grid.",
    "",
    "Recovered TI (this fit, seed=1, single realization -- NOT the 200-seed ensemble used "
    "for the fit_recovery figure):",
]
for r in results:
    cap_lines.append(
        f"- Truth TI={r['ti_true']:.0f} SI -> fit {r['ti']:.1f} "
        f"[{r['ti_lo']:.1f}, {r['ti_hi']:.1f}] SI (Delta chi^2=1); "
        f"albedo {r['alb']:.4f} +/- {r['alb_sig']:.4f}; chi2/ndof = {r['chi2']:.1f}/{r['ndof']}"
    )
cap_lines += [
    "",
    "Caveat: both true TI values (56, 178 SI) are themselves EXACT values already present in "
    "the fitting grid, so this is primarily a fit-machinery self-consistency check "
    "(interpolation is trivial there), not an independent forward-model recovery test -- see "
    "fit_recovery.png/fit_recovery.caption.md for the genuine recovery test using independent "
    "forward thermal simulations at additional TI values.",
    "",
    "Data: sweep grid `.../thermal/23871ousha/sweep/`; fit tool "
    "`/tmp/ema_ti_fit.py` (Ousha-parametrized, hardcoded-albedo bug fixed -- see memory "
    "ema-ti-fit-hardcoded-albedo-bug); binary 79f091e, CPU, PTM_RAM_ILLUM_MAX_GB=40, chi=2.7.",
]
open(OUT + "/fit_curves.caption.md", "w").write("\n".join(cap_lines) + "\n")
print("wrote", OUT + "/fit_curves.caption.md")
