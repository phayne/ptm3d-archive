#!/usr/bin/env python3
"""Gray -> two-band radiosity spatial delta on the Haworth SE-rim summer-max
footprint T_bol map.

Both runs share the same DEM/mesh, site registration, and 365-step full-year
schedule; the ONLY difference is the radiosity band treatment (gray lumping vs
band-separated visible+thermal-IR). Differencing their per-cell summer-max maps
isolates where the band separation changes the surface temperature -- expected
to concentrate in the shadowed / permanently-shadowed (PSR) cells that see the
sunlit rim only in scattered visible + re-emitted IR.

Reuses diviner_seasonal_compare's pipeline (binning, registration, season split)
so the two maps are on the identical grid. Reads the tracked run outputs only;
writes a 1x3 figure (gray | two-band | two-band-minus-gray) to docs/figures/v2/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import spiceypy as sp

sys.path.insert(0, str(Path(__file__).parent))
import diviner_seasonal_compare as C
from ptm_pubstyle import apply_style

# Haworth SE-rim site (the "sunlit" key in diviner_seasonal_compare.SITES).
SITE_LAT, SITE_LON = -88.181, 6.041


def summer_max_map(run: Path):
    """Per-cell summer MAX footprint T_bol on the standard comparison grid."""
    smask, _wmask, _lats = C.season_masks(run)
    g = C.model_seasonal(run, SITE_LAT, SITE_LON, smask)
    return g


def classify_delta(gray, delta, core):
    """Mean delta stratified by the gray summer-max regime (deep shadow / PSR,
    partial, sunlit) so we can say WHERE the band-separation warming lands."""
    v = core & np.isfinite(gray) & np.isfinite(delta)
    out = {}
    bins = [("deep shadow / PSR (<100 K)", gray < 100.0),
            ("partial (100-200 K)", (gray >= 100.0) & (gray < 200.0)),
            ("sunlit (>=200 K)", gray >= 200.0)]
    for label, sel in bins:
        m = v & sel
        d = delta[m]
        out[label] = dict(n=int(d.size),
                          mean=float(d.mean()) if d.size else float("nan"),
                          p95=float(np.percentile(d, 95)) if d.size else float("nan"),
                          maxv=float(d.max()) if d.size else float("nan"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gray", default="/Users/paha3326/dev/heat3d/e2_twoband/e3_gray",
                    help="Gray (band_separated=false) run directory.")
    ap.add_argument("--twoband", default="/Users/paha3326/dev/heat3d/e2_twoband/e3_twoband",
                    help="Two-band (band_separated=true) run directory.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: docs/figures/v2).")
    ap.add_argument("--figname", default="fig_graydelta_summermax",
                    help="Output figure basename (no extension).")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir) if args.outdir else C.REPO / "docs/figures/v2"
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style()
    sp.furnsh(C.METAKERNEL)
    g_gray = summer_max_map(Path(args.gray))
    g_tb = summer_max_map(Path(args.twoband))
    sp.kclear()

    core = g_tb["core"]
    gray = np.where(core, g_gray["max"], np.nan)
    tb = np.where(core, g_tb["max"], np.nan)
    delta = tb - gray

    # Report the spatial distribution of the change.
    valid = core & np.isfinite(delta)
    dv = delta[valid]
    print(f"[graydelta] summer-max two-band minus gray, N_core={dv.size}")
    print(f"   overall  mean={dv.mean():+.3f}  median={np.median(dv):+.3f}  "
          f"p95={np.percentile(dv,95):+.3f}  max={dv.max():+.3f}  min={dv.min():+.3f} K")
    for label, s in classify_delta(gray, delta, core).items():
        print(f"   {label:28s} N={s['n']:6d} mean={s['mean']:+.3f} "
              f"p95={s['p95']:+.3f} max={s['maxv']:+.3f} K")

    # Symmetric diverging scale from the 98th percentile of |delta|, >=0.5 K.
    dcap = max(0.5, float(np.ceil(np.percentile(np.abs(dv), 98) * 2) / 2))
    ext = [(g_tb["ex"][0]-g_tb["x0"])/1e3, (g_tb["ex"][-1]-g_tb["x0"])/1e3,
           (g_tb["ey"][0]-g_tb["y0"])/1e3, (g_tb["ey"][-1]-g_tb["y0"])/1e3]
    vlo, vhi = 30, 300     # matches SCALES_BY_SITE["sunlit"][("summer","max")]

    gray_disp = np.where(core, C.fill_internal_gaps(gray), np.nan)
    tb_disp = np.where(core, C.fill_internal_gaps(tb), np.nan)
    delta_disp = np.where(core, C.fill_internal_gaps(delta), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    panels = [(gray_disp, "inferno", Normalize(vlo, vhi), "gray radiosity", "K"),
              (tb_disp, "inferno", Normalize(vlo, vhi), "two-band radiosity", "K"),
              (delta_disp, "RdBu_r", TwoSlopeNorm(0.0, -dcap, dcap),
               "two-band $-$ gray", "$\\Delta$K")]
    for ax, (arr, cmap, norm, title, cbl) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbl)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("Haworth SE rim summer-max $T_{bol}$: two-band vs gray radiosity",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for fmt, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(outdir / (args.figname + fmt), dpi=dpi)
    plt.close(fig)
    print(f"[graydelta] wrote {outdir/(args.figname + '.png')}")


if __name__ == "__main__":
    main()
