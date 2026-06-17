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
