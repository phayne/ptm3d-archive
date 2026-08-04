#!/usr/bin/env python3
"""Martinez & Siegler (2021) vs Hayne et al. (2017) regolith conductivity.

Reproduces the two laws PTM3D can now select between, evaluated with PTM3D's own
lunar thermophysical defaults, and writes two clean figures plus a machine-readable
table of the ratios.

Provenance
----------
Model coefficients: Martinez, A., & Siegler, M. A. (2021), "A Global Thermal
Conductivity Model for Lunar Regolith at Low Temperatures", JGR Planets 126,
e2021JE006829, Eqs. 9 and 10.
Hayne defaults: src/cpp/thermal/planetary_properties.h::getMoonProperties()
(ks 7.4e-4, kd 3.4e-3 W/m/K; rhos 1100, rhod 1800 kg/m^3; chi 2.7; H 0.06 m).

The curves here are computed in Python directly from the published coefficients,
i.e. INDEPENDENTLY of the C++ implementation -- which is the point: the C++
anchor test (build/bin/test_conductivity_model) checks the code against these
same published numbers, so agreement between the two is evidence, not tautology.

Usage:  python3 scripts/plot_conductivity_models.py --outdir docs/figures/conductivity
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Martinez & Siegler (2021) Eq. 9 -----------------------------------------
AC, BC, CC, DC, EC, FC, GC, HC, IC = (
    -0.203297, -11.472, 22.5793, -14.3084, 3.41742,
    0.01101, -2.8049e-5, 3.35837e-8, -1.40021e-11,
)
# --- Martinez & Siegler (2021) Eq. 10 ----------------------------------------
A1, A2, B1, B2 = 5.0821e-6, 5.1e-3, 2.0022e-13, 1.953e-10
T_MIN = 2.7  # PTM3D's ABSOLUTE_ZERO_K; Eq. 9 diverges below ~1 K

# --- PTM3D lunar defaults (Hayne et al. 2017) --------------------------------
KS, KD = 7.4e-4, 3.4e-3
RHOS, RHOD = 1100.0, 1800.0
CHI, H = 2.7, 0.06


def k_am(T):
    t = np.maximum(np.asarray(T, float), T_MIN)
    return (AC + BC * t**-4 + CC * t**-3 + DC * t**-2 + EC * t**-1
            + FC * t + GC * t**2 + HC * t**3 + IC * t**4)


def k_martinez(T, rho):
    t = np.maximum(np.asarray(T, float), T_MIN)
    return (A1 * rho - A2) * k_am(t) + (B1 * rho - B2) * t**3


def kc_hayne(rho):
    """Hayne's kc(z) re-expressed in the local density, since both kc(z) and
    rho(z) are the same exponential blend."""
    df = (RHOD - np.asarray(rho, float)) / (RHOD - RHOS)
    return KD + (KS - KD) * df


def k_hayne(T, rho):
    return kc_hayne(rho) * (1.0 + CHI * (np.asarray(T, float) / 350.0) ** 3)


def git_sha(repo):
    try:
        return subprocess.run(
            ["/Library/Developer/CommandLineTools/usr/bin/git", "-C", str(repo),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures/conductivity")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]

    T = np.linspace(T_MIN, 400.0, 1200)

    # ---- Figure 1: k(T) at the surface, mid and deep density nodes ----------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colors = {1100: "#1b6ca8", 1300: "#0f8b8d", 1800: "#a63d40"}
    for rho in (1100, 1300, 1800):
        ax.plot(T, k_hayne(T, rho) * 1e3, color=colors[rho], ls="--", lw=1.8,
                label=f"Hayne 2017, {rho} kg m$^{{-3}}$")
        ax.plot(T, k_martinez(T, rho) * 1e3, color=colors[rho], ls="-", lw=2.0,
                label=f"Martinez 2021, {rho} kg m$^{{-3}}$")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal conductivity ($10^{-3}$ W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Regolith conductivity: Martinez & Siegler (2021) vs Hayne et al. (2017)")
    ax.set_xlim(0, 400)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_conductivity_models.{ext}", dpi=200)
    plt.close(fig)

    # ---- Figure 2: the ratio, which is what actually drives the run --------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for rho in (1100, 1300, 1800):
        ax.plot(T, k_martinez(T, rho) / k_hayne(T, rho), color=colors[rho], lw=2.0,
                label=f"{rho} kg m$^{{-3}}$")
    ax.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$k_\mathrm{Martinez}\,/\,k_\mathrm{Hayne}$")
    ax.set_title("Conductivity ratio at PTM3D's lunar densities")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=9, frameon=False, title="bulk density")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_conductivity_ratio.{ext}", dpi=200)
    plt.close(fig)

    # ---- Machine-readable table + provenance sidecar -----------------------
    rows = []
    for rho in (1100, 1300, 1800):
        for t in (30, 60, 100, 150, 200, 250, 300, 350, 400):
            kh, km = float(k_hayne(t, rho)), float(k_martinez(t, rho))
            rows.append({"T_K": t, "rho_kg_m3": rho,
                         "k_hayne_W_m_K": kh, "k_martinez_W_m_K": km,
                         "ratio_martinez_over_hayne": km / kh})
    meta = {
        "generator": str(Path(__file__).resolve()),
        "git_sha": git_sha(repo),
        "model_martinez": {"reference": "Martinez & Siegler (2021) JGR Planets 126, "
                                        "e2021JE006829, Eqs. 9-10",
                           "A1": A1, "A2": A2, "B1": B1, "B2": B2,
                           "kam_coeffs": [AC, BC, CC, DC, EC, FC, GC, HC, IC],
                           "T_floor_K": T_MIN},
        "model_hayne": {"reference": "Hayne et al. (2017) JGR Planets 122, 2371",
                        "ks": KS, "kd": KD, "rhos": RHOS, "rhod": RHOD,
                        "chi": CHI, "H_m": H},
        "table": rows,
    }
    (out / "conductivity_models.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out}/fig_conductivity_models.[png,pdf]")
    print(f"wrote {out}/fig_conductivity_ratio.[png,pdf]")
    print(f"wrote {out}/conductivity_models.json  (git {meta['git_sha']})")


if __name__ == "__main__":
    main()
