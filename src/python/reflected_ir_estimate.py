#!/usr/bin/env python3
"""Estimate the reflected-IR contribution to Diviner's PSR bolometric T.

Lucey & Hayne (2021): the sunlit crater walls bathe the PSR floor in thermal IR
(>3 um); the floor reflects its IR albedo (1-eps ~ 0.05) of that incident flux,
and the reflected wall-IR lands in Diviner's thermal channels (hence the
bolometric T) but is ABSENT from our emission-only model comparison.

Per face, Diviner sees an apparent bolometric:
    sigma T_app^4 = eps*sigma*T^4  +  (1-eps) * I_inc        (Lambertian floor)
with I_inc ~ Q_indirect / eps  (the model radiosity absorbs eps*I_inc).
The model currently compares the EMISSION-only footprint
    T_bol_emit = eps^(1/4) * <A T^4 / A>^(1/4).
This script forms the apparent footprint with the reflected term and reports the
boost ΔT (= contribution to Diviner not in our comparison), over the Haworth
window, using each face's peak T and the radiosity indirect flux at that instant
(PTM_DUMP_REFLIR dump: per face float64 Tpeak, q_at_peak).
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diviner_compare import read_ply, face_geometry
from diviner_data import latlon_to_diviner_xy, KM_PER_DEG

SIGMA = 5.670374419e-8
EPS = 0.95
LAT, LON = -87.45, -2.1
WINDOW_KM, CELL_M, CORE_KM = 15.0, 250.0, 15.0
RUN = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output/sweep_haworth_dc_pass0"
REFLIR = "/tmp/haworth_reflir.bin"


def main():
    with open(REFLIR, "rb") as fh:
        magic, nf = struct.unpack("II", fh.read(8))
        assert magic == 0x52464952, f"bad magic 0x{magic:08x}"
        pk = np.frombuffer(fh.read(), np.float64).reshape(nf, 2)
    Tpeak = pk[:, 0]
    q_peak = pk[:, 1]                       # radiosity indirect absorbed at peak (W/m^2)
    I_inc = q_peak / EPS                    # incident IR irradiance on the floor
    print(f"faces={nf}  Tpeak: med={np.median(Tpeak):.1f} max={Tpeak.max():.1f}")
    print(f"q@peak (indirect absorbed): med={np.median(q_peak):.2f} max={q_peak.max():.2f} W/m^2")

    v, f, flat, flon = read_ply(str(Path(RUN) / "mesh.ply"))
    area, flat, flon = face_geometry(v, f, flat, flon)
    ds = np.fromfile(Path(RUN) / "statistics" / "direct_solar_max.bin", np.float32)[1:]
    psr_face = ds < 5.0

    # grid into 250 m cells in the Diviner deg frame
    xd0, yd0 = latlon_to_diviner_xy(LAT, LON)
    cd = (CELL_M / 1000.0) / KM_PER_DEG
    W = WINDOW_KM / KM_PER_DEG
    ex = np.arange(xd0 - W, xd0 + W + cd, cd)
    ey = np.arange(yd0 - W, yd0 + W + cd, cd)
    nx, ny = len(ex) - 1, len(ey) - 1
    fxd, fyd = latlon_to_diviner_xy(flat, flon)
    rkm = np.hypot(fxd - xd0, fyd - yd0) * KM_PER_DEG
    ix = np.floor((fxd - ex[0]) / cd).astype(int)
    iy = np.floor((fyd - ey[0]) / cd).astype(int)
    keep = (rkm <= CORE_KM) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = iy[keep] * nx + ix[keep]
    A = area[keep]
    Asum = np.bincount(lin, weights=A, minlength=nx * ny)

    # area-weighted footprint quantities
    emit_flux = EPS * SIGMA * Tpeak ** 4                       # emission only
    refl_flux = (1.0 - EPS) * I_inc                            # reflected wall-IR
    app_flux = emit_flux + refl_flux                          # apparent (Diviner)
    sumT4 = np.bincount(lin, weights=A * (Tpeak[keep] ** 4), minlength=nx * ny)
    sum_emit = np.bincount(lin, weights=A * emit_flux[keep], minlength=nx * ny)
    sum_app = np.bincount(lin, weights=A * app_flux[keep], minlength=nx * ny)
    ok = Asum > 0
    with np.errstate(invalid="ignore"):
        Tbol_emit = np.where(ok, EPS ** 0.25 * (sumT4 / Asum) ** 0.25, np.nan)
        Tbol_app = np.where(ok, (sum_app / Asum / SIGMA) ** 0.25, np.nan)
    dT = Tbol_app - Tbol_emit

    # PSR cells (emission Tbol < 200 K)
    psr = ok & (Tbol_emit < 200.0)
    print(f"\npopulated cells: {int(ok.sum())}  PSR cells (Tbol_emit<200K): {int(psr.sum())}")
    print("=== reflected-IR boost to apparent footprint bolometric T (PSR cells) ===")
    print(f"  ΔT median = {np.nanmedian(dT[psr]):+.2f} K")
    print(f"  ΔT mean   = {np.nanmean(dT[psr]):+.2f} K")
    print(f"  ΔT p90    = {np.nanpercentile(dT[psr], 90):+.2f} K   p99 = {np.nanpercentile(dT[psr], 99):+.2f} K")
    print(f"  ΔT max    = {np.nanmax(dT[psr]):+.2f} K")
    # boost vs cell warmth
    for lo, hi in [(0, 90), (90, 110), (110, 140), (140, 250)]:
        m = psr & (Tbol_emit >= lo) & (Tbol_emit < hi)
        if m.sum() > 5:
            print(f"  Tbol_emit {lo:3d}-{hi:3d} K: n={int(m.sum()):5d}  ΔT med={np.nanmedian(dT[m]):+.2f} K")
    print("\n=== implied effect on the Haworth PSR bias (currently -8.7 K) ===")
    print(f"  adding reflected IR warms the model footprint by ~{np.nanmedian(dT[psr]):+.1f} K (median)")
    print(f"  -> new PSR bias ~ {-8.7 + np.nanmedian(dT[psr]):+.1f} K (median); "
          f"{-8.7 + np.nanpercentile(dT[psr],90):+.1f} K at p90 cells")


if __name__ == "__main__":
    main()
