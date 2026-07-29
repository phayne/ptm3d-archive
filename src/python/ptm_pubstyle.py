"""
Shared matplotlib style for PTM3D publication figures (docs/figures/v2/).

Built on top of `pretty_plots` (https://github.com/phayne/pretty-plots) to get
IDL-style defaults — bold axis lines, full box-frame, inward ticks, IDL Hershey
serif, IDL colour tables — with a few PTM3D-specific colour constants and
helpers on top.

Use:

    import matplotlib
    matplotlib.use('Agg')
    from ptm_pubstyle import apply_style, COLORS, COLUMN_W, DOUBLE_W

    apply_style()                     # IDL family
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 0.75 * DOUBLE_W))
    ...
    fig.savefig(out_path)
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pretty_plots import idl as _pp_idl
    HAS_PRETTY = True
except Exception:
    _pp_idl = None
    HAS_PRETTY = False

# AGU / Icarus single- and double-column widths (inches)
COLUMN_W = 3.5
DOUBLE_W = 7.2

# PTM3D colour palette — kept compatible with the IDL `colortables` so curves
# read well against IDL's white background and full-box frame
COLORS = {
    "ptm":      "#1f77b4",   # IDL default blue
    "ref":      "#d62728",   # IDL default red
    "neutral":  "#444444",
    "lat0":     "#1f77b4",
    "lat30":    "#2ca02c",
    "lat45":    "#ff7f0e",
    "lat60":    "#9467bd",
    "lat85":    "#d62728",
    "shadow":   "#cccccc",
    "envelope": "#cfd8e8",
    "good":     "#2ca02c",
    "bad":      "#d62728",
}

LATITUDE_COLORS = {
    0:  COLORS["lat0"],
    30: COLORS["lat30"],
    45: COLORS["lat45"],
    60: COLORS["lat60"],
    85: COLORS["lat85"],
}


def apply_style(variant: str | None = None) -> None:
    """Apply the IDL family of pretty_plots, then a few PTM3D refinements.

    `variant` may be one of pretty_plots' IDL variants: aas / aa / nature /
    icarus / hershey / latex. Defaults to the bare IDL family.
    """
    if HAS_PRETTY:
        _pp_idl.use(variant=variant)
    else:
        # Minimal IDL-ish fallback if pretty_plots isn't installed.
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "lines.linewidth": 1.8,
            "axes.linewidth": 1.4,
            "axes.spines.top":   True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top":   True,
            "ytick.right": True,
            "savefig.dpi": 300,
        })

    # PTM3D refinements on top of IDL — large fonts for clean publication plots
    mpl.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "lines.linewidth": 2.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.dpi": 110,
    })


def panel_label(ax, label, x=0.02, y=0.97, **kw):
    """Place a panel label like '(a)' in the top-left corner."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        ha="left", va="top",
        **kw,
    )


def savefig_pinned(fig, out_stem, formats=(".png", ".pdf"), canvas_px=None, dpi=300):
    """Save `fig` to `out_stem` + each of `formats`.

    If `canvas_px=(w, h)` is given, pin the saved canvas to that EXACT pixel
    size at `dpi` (matching a reference/annotated sibling 1:1) instead of the
    default `savefig.bbox="tight"` rcParam set in `apply_style()`. That rcParam
    crops to rendered-CONTENT extent, so removing a title/stat-box SHRINKS the
    crop even for an otherwise pixel-identical figure -- `bbox_inches=None`
    does NOT disable it; an explicit full-canvas `Bbox` is required. This is
    the pattern validated on `diviner_seasonal_compare.py` /
    `fig8_hybrid_ringfix_scatter.py`'s `--pubclean-canvas-px`
    (manuscript-v4 review, 2026-07-22): pin every publication-clean variant to
    its annotated sibling's own shipped pixel size so the manuscript swap is
    exact-dims, not just visually close.

    Returns the list of Paths written.
    """
    from pathlib import Path
    out_stem = Path(out_stem)
    written = []
    if canvas_px is None:
        for suf in formats:
            p = out_stem.with_suffix(suf)
            fig.savefig(p, dpi=dpi)
            written.append(p)
        return written
    from matplotlib.transforms import Bbox
    w_px, h_px = canvas_px
    w_in, h_in = w_px / dpi, h_px / dpi
    # Center a fixed w_in x h_in window on the figure's ACTUAL rendered content
    # extent (fig.get_tightbbox(), in inches) rather than anchoring at (0,0) of
    # a resized canvas -- a plain (0,0)-origin Bbox silently clips content that
    # sits left/below the figure's nominal origin once tight_layout has been
    # called with a different (e.g. title-reserving) rect than the pinned
    # target aspect implies (found 2026-07-22 on the indirect-bands map figure:
    # the left panel title got clipped). This is the same fix already applied
    # to diviner_seasonal_compare.py's --overview mode for a similar gotcha.
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    cx, cy = (tb.x0 + tb.x1) / 2.0, (tb.y0 + tb.y1) / 2.0
    bbox = Bbox.from_bounds(cx - w_in / 2.0, cy - h_in / 2.0, w_in, h_in)
    for suf in formats:
        p = out_stem.with_suffix(suf)
        fig.savefig(p, dpi=(dpi if suf == ".png" else None), bbox_inches=bbox)
        written.append(p)
    return written


def load_face_mask(path):
    """Load a boolean per-face mask (uint32 n, then n uint8) written by
    make_psr_mask.py — True = spurious-lit face to reclassify as PSR."""
    import numpy as np
    with open(path, "rb") as fh:
        n = int(np.fromfile(fh, np.uint32, 1)[0])
        return np.fromfile(fh, np.uint8, n).astype(bool)
