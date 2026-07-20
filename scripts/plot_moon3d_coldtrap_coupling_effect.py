#!/usr/bin/env python3
"""Moon3D cold-trap area fraction — effect of radiative coupling (2-panel).

Reads the precomputed data from
``docs/figures/moon3d_500k/psr_coldtrap_vs_latitude.json`` and makes a 2-panel
figure that isolates the effect of inter-face radiative coupling on the
cold-trap area fraction:

  (a) cold-trap area fraction vs latitude, with and without coupling, the gap
      between them shaded;
  (b) the coupling-induced reduction — absolute (percentage points, left axis)
      and relative (% of the no-coupling area, right axis).

Cold trap: surface T_max < 110 K. Coupling warms cold faces via inter-face IR,
so it always *reduces* the cold-trap area.

Styled with the `pretty_plots` planetary family for consistency with the other
publication figures.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pretty_plots as pp
pp.use("planetary")
import matplotlib.pyplot as plt

REPO = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"
_ap = argparse.ArgumentParser(description="Moon3D cold-trap coupling-effect 2-panel figure")
_ap.add_argument("--data", default=REPO + "/docs/figures/moon3d_500k/psr_coldtrap_vs_latitude.json",
                 help="psr_coldtrap JSON (lat, coldtrap_coupled, coldtrap_radoff)")
_ap.add_argument("--out", default=REPO + "/docs/figures/moon3d_500k/coldtrap_coupling_effect",
                 help="output base path (.png/.pdf appended)")
_a = _ap.parse_args()
DATA = _a.data
OUT = _a.out
COLD_THRESH = 110.0  # K
TRUE_LAT = -84.781   # Moon3D true latitude

C_COUPLED = "#b2182b"   # rad-coupled
C_RADOFF = "#777777"    # no coupling
C_ABS = "#2166ac"       # absolute reduction
C_REL = "#1b7837"       # relative reduction

d = json.load(open(DATA))
lat = np.array(d["lat"])
ctc = np.array(d["coldtrap_coupled"])
cto = np.array(d["coldtrap_radoff"])

order = np.argsort(lat)
lat, ctc, cto = lat[order], ctc[order], cto[order]

reduction_pp = 100 * (cto - ctc)                         # percentage points
# relative reduction (% of no-coupling area); guard against divide-by-zero
rel = np.where(cto > 0, 100 * (cto - ctc) / cto, np.nan)

fig, ax = plt.subplots(1, 2, figsize=(16.5, 7.0))

# --- Panel (a): absolute fractions, gap shaded ----------------------------
ax[0].fill_between(lat, 100 * ctc, 100 * cto, color=C_COUPLED, alpha=0.15,
                   zorder=1, label="area lost to coupling")
ax[0].plot(lat, 100 * cto, "--s", color=C_RADOFF, lw=2.4, ms=9,
           label="no coupling", zorder=3)
ax[0].plot(lat, 100 * ctc, "-o", color=C_COUPLED, lw=2.6, ms=9,
           label="rad-coupled", zorder=4)
ax[0].set_xlabel("latitude [deg]")
ax[0].set_ylabel("cold-trap area fraction [%]")
ax[0].set_title("(a) cold-trap area  ($T_{max}$ < %g K)" % COLD_THRESH)
ax[0].grid(alpha=0.3)
# legend on the LEFT (Paul's review): the cold-trap curve peaks at the left
# (-85) and descends to the right, so the lower-left quadrant is clear of all
# series; keep the legend out of the data band.
ax[0].legend(fontsize=15, framealpha=0.92, loc="lower left")

# --- Panel (b): the coupling effect (difference) --------------------------
ax[1].bar(lat, reduction_pp, width=3.2, color=C_ABS, alpha=0.85,
          zorder=2, label="absolute (left)")
ax[1].set_xlabel("latitude [deg]")
ax[1].set_ylabel("reduction [percentage points]", color=C_ABS)
ax[1].tick_params(axis="y", labelcolor=C_ABS)
# extra top headroom so the upper-left legend floats clear of the tallest bars
# (mid-latitude absolute reduction peaks near ~6 pp)
ax[1].set_ylim(0, max(reduction_pp) * 1.45)
ax[1].set_title("(b) reduction from radiative coupling")
ax[1].grid(alpha=0.3)

axr = ax[1].twinx()
axr.plot(lat, rel, "-D", color=C_REL, lw=2.6, ms=9, zorder=4,
         label="relative (right)")
axr.set_ylabel("relative reduction [%]", color=C_REL)
axr.tick_params(axis="y", labelcolor=C_REL)
axr.set_ylim(0, 100)

# merged legend for panel (b), on the LEFT (Paul's review): both the bars and
# the relative-reduction line start low at the left (-85), so the upper-left
# quadrant is clear -- this moves the legend off the rising relative curve that
# ends high at the right.
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=15, framealpha=0.92, loc="upper left")

fig.tight_layout()

fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
print("saved", OUT + ".pdf / .png")
for la, a, r in zip(lat, reduction_pp, rel):
    print("  lat %7.2f : -%.2f pp  (-%.1f%% relative)" % (la, a, r))
