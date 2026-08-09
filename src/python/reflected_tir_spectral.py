#!/usr/bin/env python3
"""Spectral quantification of reflected thermal-IR from PSR floors, using the
actual Apollo highland-soil reflectance (62231) that P. Lucey used.

Lucey (email, Nov 2025; "Error in PSR temperature estimates 8-14 microns"):
for an instrument that measures PSR temperature in the 8-14 um band, reflected
thermal-IR from the sunlit ~300 K walls is NOT negligible -- it overestimates the
brightness T by ~+18 K at 70 K down to ~+2.5 K at 120 K. He computed this with a
blackbody radiance->T calibration, the highland-soil spectrum + Kirchhoff (eps=1-R).

This script (a) reproduces that 8-14 um overestimate from his spectrum as a
validation, and (b) shows that the SAME reflected term, integrated into the
Diviner BOLOMETRIC (channels 3-9, weighted to 50-400 um at PSR temps), contributes
only ~+0.2-0.3 K -- because the cold floor's own emission dominates the long-wave
channels while the reflected wall-IR is confined to 8-14 um. So the effect is
important for narrowband 8-14 um TIR (Lucey's prism proposal) but negligible for
the Diviner Tbol used in our model comparison.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import openpyxl

H, C, KB, SIG = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 5.670374419e-8
XLSX = (Path(__file__).resolve().parents[2]
        / "docs/lucey_tir_reflection/LVO_ReferenceSpectra15041_62231.xlsx")
# Lucey's digitized overestimate curve (8-14 um), from his attached IDL figure
LUCEY_8_14 = {70: 18.5, 80: 13.0, 90: 8.5, 100: 5.0, 110: 3.5, 120: 2.5}
# Diviner thermal channels (um)
DIVINER_CH = [(8.10, 8.40), (8.38, 8.68), (8.55, 8.85),
              (13.0, 23.0), (25.0, 41.0), (50.0, 100.0), (100.0, 400.0)]
TWALL, F_IR = 300.0, 0.6           # wall T; incident IR irradiance = F_IR * sigma Tfloor^4


def planck(lam_um, T):
    lam = np.asarray(lam_um, float) * 1e-6
    with np.errstate(over="ignore"):
        return (2*H*C**2 / lam**5) / (np.exp(H*C/(lam*KB*T)) - 1.0) * 1e-6


def load_reflectance(lam):
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    rows = [r for r in wb.active.iter_rows(values_only=True)][1:]
    wl = np.array([r[0] for r in rows], float)
    R62 = np.array([r[2] for r in rows], float)        # 62231 highland soil
    # real data to ~10 um; held at the 10-um edge value beyond (Christiansen-low R)
    R = np.clip(np.interp(lam, wl, R62), 0.0, 0.6)
    return R


def band_bt(lam, L, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    tgt = np.trapz(L[m], lam[m])
    Ts = np.linspace(40, 320, 2801)
    integ = np.array([np.trapz(planck(lam[m], T), lam[m]) for T in Ts])
    return np.interp(tgt, integ, Ts)


def bol_bt(lam, L):
    def cs(Lx):
        return sum(np.trapz(Lx[(lam >= lo) & (lam <= hi)], lam[(lam >= lo) & (lam <= hi)])
                   for lo, hi in DIVINER_CH)
    tgt = cs(L)
    Ts = np.linspace(40, 320, 2801)
    integ = np.array([cs(planck(lam, T)) for T in Ts])
    return np.interp(tgt, integ, Ts)


def main():
    lam = np.linspace(5.0, 400.0, 8000)
    R = load_reflectance(lam)
    eps = 1.0 - R
    Lwall = eps * planck(lam, TWALL)
    wall_int = np.trapz(Lwall, lam)

    Tfs = np.array([70, 80, 90, 100, 110, 120])
    d814, dbol, b14e = [], [], []
    print("Reflected-TIR overestimate using Lucey's highland-soil spectrum (62231):")
    print(f"  {'Tf':>4} | {'8-14um Δ':>9} {'(Lucey)':>8} | {'bolometric Δ':>12}")
    for Tf in Tfs:
        I_inc = (Lwall / wall_int) * (F_IR * SIG * Tf**4)
        Le = eps * planck(lam, Tf)
        Lt = Le + R * I_inc / np.pi
        de = band_bt(lam, Lt, 8, 14) - band_bt(lam, Le, 8, 14)
        do = bol_bt(lam, Lt) - bol_bt(lam, Le)
        d814.append(de); dbol.append(do); b14e.append(band_bt(lam, Le, 8, 14))
        print(f"  {Tf:4d} | {de:9.1f} {LUCEY_8_14[Tf]:8.1f} | {do:12.2f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from ptm.pubstyle import apply_style
        apply_style()
    except Exception:
        pass
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    # left: spectral radiance of an 80 K floor
    Tf = 80.0
    I_inc = (Lwall / wall_int) * (F_IR * SIG * Tf**4)
    Le = eps * planck(lam, Tf)
    Lr = R * I_inc / np.pi
    axL.fill_between([8, 14], 1e-8, 1e3, color="C1", alpha=0.13, label="8–14 µm band")
    for lo, hi in DIVINER_CH:
        axL.axvspan(lo, hi, color="0.5", alpha=0.10)
    axL.plot(lam, Le, "C0-", lw=1.8, label=f"floor emission ({Tf:.0f} K)")
    axL.plot(lam, Lr, "C3-", lw=1.8, label="reflected wall-IR (300 K)")
    axL.plot(lam, Le + Lr, "k--", lw=1.1, label="total")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlim(5, 400); axL.set_ylim(1e-7, 2e-1)
    axL.set_xlabel("wavelength (µm)")
    axL.set_ylabel("spectral radiance (W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)")
    axL.set_title("PSR floor radiance (Apollo 62231 ε)", fontsize=10)
    axL.legend(fontsize=7.5, loc="lower center")

    # right: overestimate vs Tfloor, 8-14 um vs Diviner bolometric, with Lucey points
    axR.plot(Tfs, d814, "C3o-", lw=1.8, label="this model, 8–14 µm")
    axR.plot(list(LUCEY_8_14), list(LUCEY_8_14.values()), "k^", ms=7,
             label="Lucey (8–14 µm)")
    axR.plot(Tfs, dbol, "C0s-", lw=1.8, label="this model, Diviner $T_{bol}$")
    axR.set_xlabel("PSR floor temperature (K)")
    axR.set_ylabel("brightness-T overestimate (K)")
    axR.set_title("Reflected-TIR overestimate: band vs bolometric", fontsize=10)
    axR.legend(fontsize=8)
    axR.set_ylim(-1, 20)
    fig.suptitle("Reflected thermal-IR: significant for 8–14 µm (matches Lucey), "
                 "negligible for Diviner $T_{bol}$ (50–400 µm-weighted)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("docs/figures/diviner_validation/reflected_tir_spectral")
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(out.with_suffix(fmt), dpi=dpi)
    print(f"\nwrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
