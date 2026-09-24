#!/usr/bin/env python3
"""Surface-temperature time series for a few Moon3D faces with contrasting
thermal environments, over one lunation.

Faces are auto-selected from the per-face statistics + geometry to span the
regimes: the sunniest peak, a permanently/largely shadowed facet, the facet most
warmed by inter-face IR (radiative coupling), and a near-level reference facet.
Each face's instantaneous surface T (radiation ON, solid) is plotted vs time;
the radiation-OFF curve (dashed) shows the coupling contribution.

    python3 moon3d_temp_timeseries.py \
        --coupled  output/moon3d_500k_clean \
        --baseline output/moon3d_500k_clean_radoff \
        --out docs/figures/moon3d_500k/temp_timeseries.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from moon3d_temp_3d import read_ply_vertices_double, tri_faces
from ptm_pubstyle import apply_style, DOUBLE_W, load_face_mask


def load_stat(d: Path, name: str) -> np.ndarray:
    return np.fromfile(d / "statistics" / f"temperature_{name}.bin", dtype=np.float32)[1:]


def load_step(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ncycles", type=int, default=2,
                    help="number of synodic cycles to tile (shows curves connect)")
    ap.add_argument("--out", default="docs/figures/moon3d_500k/temp_timeseries.png")
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="exclude faces whose coupled max-T exceeds this (K) from "
                         "face selection (view-factor over-count artifacts); 0 = keep all")
    ap.add_argument("--reclassify-mask", default="",
                    help="boolean face-mask (make_psr_mask.py): also exclude spurious-lit "
                         "deep-PSR faces from selection")
    a = ap.parse_args()
    cdir, bdir = Path(a.coupled), Path(a.baseline)

    # geometry: face normals (outward, figure-up after z-flip) + centroids
    full = pv.read(str(cdir / "mesh.ply"))
    f = tri_faces(full)
    v = read_ply_vertices_double(str(cdir / "mesh.ply"))
    v = v - v.mean(0); v[:, 2] *= -1.0
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    nrm[nrm[:, 2] < 0] *= -1.0           # outward (figure +z up)
    tilt = np.degrees(np.arccos(np.clip(nrm[:, 2], -1, 1)))   # slope from level

    cmax, cavg, cmin = (load_stat(cdir, k) for k in ("max", "avg", "min"))
    bavg = load_stat(bdir, "avg")
    warm = cavg - bavg                   # radiative warming

    # mask view-factor-overcount artifacts (coupled max-T above the physical limit)
    keep = (cmax <= a.mask_tmax) if a.mask_tmax > 0 else np.ones(len(cmax), bool)
    if a.reclassify_mask:
        keep = keep & ~load_face_mask(a.reclassify_mask)   # also drop spurious-lit deep-PSR faces
    if a.mask_tmax > 0 or a.reclassify_mask:
        print(f"  excluding {int(np.count_nonzero(~keep))} hot faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) from selection")

    # ---- select faces by thermal regime ----
    # Pick REPRESENTATIVE faces, not statistical extremes: the global argmax(cmax)
    # is a rare near-vertical facet (>500 K, top-5 of 509k) and argmax(warm) lands
    # on a face whose rad-off baseline is spuriously cold (+300 K "warming" is not
    # physical for inter-face IR). Use robust, regime-typical selectors instead.
    flat = np.where((tilt < 5) & keep)[0]

    # warm sunlit slope: warmest typical KEPT slope (~the physical subsolar max when
    # masking, else the legacy 415 K), not the handful of over-count spikes.
    tgt = float(np.percentile(cmax[keep], 99.5)) if a.mask_tmax > 0 else 415.0
    sun_c = np.where((cmax >= tgt - 8) & (cmax <= tgt + 8) & (tilt < 60) & (cmin > 40) & keep)[0]
    warm_sunlit = int(sun_c[np.argmin(np.abs(cmax[sun_c] - tgt))]) if sun_c.size \
        else int(np.argmax(np.where(keep, cmax, -np.inf)))

    # IR-warmed shadowed floor: a cold, never-strongly-lit face (cmax < 160 K)
    # lifted by inter-face IR -- restrict to a believable lift (<= 70 K) so we
    # don't grab a baseline artifact.
    flr_c = np.where((cmax < 160) & (cmin > 35) & (warm > 10) & (warm < 70) & keep)[0]
    ir_floor = int(flr_c[np.argmax(warm[flr_c])]) if flr_c.size else int(np.argmax(warm))

    # cold shadowed facet: stays cold all cycle, negligible IR gain (cold trap).
    shadowed = int(np.argmin(cmax))

    # near-level reference: typical level surface near the median peak T.
    ref = int(flat[np.argmin(np.abs(cmax[flat] - np.median(cmax)))]) if flat.size \
        else int(np.argmin(np.abs(tilt - np.median(tilt))))

    sel = {
        "warm sunlit slope": warm_sunlit,
        "near-level reference": ref,
        "IR-warmed shadowed floor": ir_floor,
        "cold shadowed facet": shadowed,
    }
    colors = dict(zip(sel, ["tab:red", "tab:gray", "tab:green", "tab:blue"]))

    # ---- time axis (days) from body_positions ----
    t = np.genfromtxt(cdir / "body_positions.csv", delimiter=",",
                      names=True, dtype=None, encoding=None)["time"]
    tt = (np.array(t, dtype="datetime64[s]") - np.array(t[0], dtype="datetime64[s]")) / np.timedelta64(1, "D")

    # ---- read the time series for only the selected faces ----
    csteps = sorted((cdir / "temperature").glob("temperature_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    bsteps = sorted((bdir / "temperature").glob("temperature_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    idx = np.array(list(sel.values()))
    Tc = np.array([load_step(p)[idx] for p in csteps])   # (nt, nface)
    Tb = np.array([load_step(p)[idx] for p in bsteps])
    nt = min(len(tt), len(Tc))
    tt, Tc, Tb = tt[:nt], Tc[:nt], Tb[:nt]

    # ---- tile over a couple of synodic cycles ----
    # The run spans one synodic month (~360 deg of subsolar longitude); step 0 and
    # step N-1 are phase-matched to <1 deg. Repeating the series shows the curves
    # connect across the period seam: the apparent "jump" at t=0 is just the dawn
    # cliff (sharp grazing-Sun terminator), not an equilibration transient.
    period = tt[-1] + (tt[1] - tt[0])          # one full synodic month (days)
    ncyc = a.ncycles
    ttN = np.concatenate([tt + c * period for c in range(ncyc)])
    TcN = np.tile(Tc, (ncyc, 1))
    TbN = np.tile(Tb, (ncyc, 1))

    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 0.5 * DOUBLE_W))
    for c in range(1, ncyc):                   # subtle period boundaries
        ax.axvline(c * period, color="0.6", ls=":", lw=0.8, zorder=0)
    for j, (name, fi) in enumerate(sel.items()):
        ax.plot(ttN, TcN[:, j], color=colors[name], lw=1.8,
                label=f"{name}  (slope {tilt[fi]:.0f}°, $T_{{max}}$ = {cmax[fi]:.0f} K)")
        ax.plot(ttN, TbN[:, j], color=colors[name], lw=1.0, ls="--", alpha=0.55)
    ax.set_xlabel("time (Earth days)")
    ax.set_ylabel("surface temperature (K)")
    ax.set_xlim(ttN[0], ttN[-1])
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)   # headroom for the legend
    ax.grid(alpha=0.3)
    ax.text(period, ax.get_ylim()[0], " 1 synodic month ", color="0.5",
            fontsize=7, va="bottom", ha="center")
    ax.legend(title="solid = rad ON, dashed = rad OFF", fontsize=7.3,
              title_fontsize=7.0, ncol=2, loc="upper center", framealpha=0.92)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}" + (" (+pdf)" if out.suffix == ".png" else ""))
    for name, fi in sel.items():
        print(f"  {name:20} face {fi:7d} slope {tilt[fi]:5.1f}deg  "
              f"Tmax {cmax[fi]:6.1f} Tmin {cmin[fi]:6.1f} warm {warm[fi]:+5.1f}K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
