#!/usr/bin/env python3
"""Step 2 — 3D oblique perspective of the Moon3D 500k thermal model.

Left panel:  radiatively-coupled mean temperature (colored surface).
Right panel: difference from the no-coupling baseline (coupled - baseline),
             i.e. the warming produced by inter-face IR / reflected coupling.

Both panels share the same oblique camera. The display geometry is decimated
for matplotlib tractability; per-face temperatures are mapped from the full
509k-face result by nearest-centroid lookup, so the science values are intact.

    python3 moon3d_temp_3d.py \
        --ply       output/moon3d_500k_clean/mesh.ply \
        --coupled   output/moon3d_500k_clean/statistics \
        --baseline  output/moon3d_500k_clean_radoff/statistics \
        --field avg \
        --out docs/figures/moon3d_500k/temp_3d.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, DOUBLE_W, panel_label, savefig_pinned


def load_stat(stat_dir: Path, name: str) -> np.ndarray:
    raw = np.fromfile(stat_dir / f"temperature_{name}.bin", dtype=np.float32)
    return raw[1:]


def tri_faces(mesh: pv.PolyData) -> np.ndarray:
    return mesh.faces.reshape(-1, 4)[:, 1:]


# VTK/pyvista downcast PLY point coordinates to float32. At selenocentric
# magnitudes (~1.7e6 m) the float32 ulp (~0.12 m) is comparable to the entire
# patch relief (~0.3 m), so the DEM collapses to a few discrete elevations.
# Read the vertex block in native double precision instead.
_PLY_TYPE = {b"double": "<f8", b"float": "<f4", b"int": "<i4",
             b"uint": "<u4", b"uchar": "<u1", b"char": "<i1",
             b"short": "<i2", b"ushort": "<u2"}


def read_ply_vertices_double(path: str) -> np.ndarray:
    """Return the (N,3) x,y,z vertex coordinates of a binary_little_endian PLY
    in full double precision (parsing the vertex element directly)."""
    with open(path, "rb") as fh:
        assert fh.readline().strip() == b"ply"
        fmt = fh.readline().strip()
        assert b"binary_little_endian" in fmt, f"unsupported PLY format: {fmt}"
        n_vert, props, cur = None, [], None
        while True:
            tok = fh.readline().strip().split()
            if not tok:
                continue
            if tok[0] == b"element":
                cur = tok[1]
                if cur == b"vertex":
                    n_vert = int(tok[2])
            elif tok[0] == b"property" and cur == b"vertex":
                props.append(tok[1])          # scalar property type
            elif tok[0] == b"end_header":
                break
        dt = np.dtype([(f"p{i}", _PLY_TYPE[t]) for i, t in enumerate(props)])
        rec = np.frombuffer(fh.read(n_vert * dt.itemsize), dtype=dt, count=n_vert)
    return np.column_stack([rec["p0"], rec["p1"], rec["p2"]]).astype(np.float64)


def centroids(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0


def render_panel(ax, v, f, fc_vals, cmap, norm, elev, azim, title, cbar_label,
                 zexag=3.0, mask=None, mask_color=(0.7, 0.7, 0.7, 1.0),
                 show_title=True):
    """Flat-shaded Poly3DCollection colored by per-face fc_vals.

    Vertical relief is exaggerated by `zexag` for legibility (the real patch
    is a shallow ~0.2 depth/diameter crater) and the box aspect follows the
    true (exaggerated) data extents rather than a forced cube. Faces flagged
    in `mask` (view-factor over-count artifacts) are drawn in a neutral gray
    so they neither color-code as false-hot nor drive the color scale.

    `show_title` (default True, backward compatible for all other callers --
    moon3d_indirect_bands.py, moon3d_temp_3d_angles.py, the latitude montage,
    etc.): set False to suppress the per-panel header (manuscript v4, Fig. 12
    "temp_3d_native_twoband_v3" review round, 2026-07-27 -- Paul's ask to move
    panel identification into the caption, matching the suptitle already
    suppressed under --pubclean).
    """
    v = v.copy()
    v[:, 2] *= zexag
    polys = v[f]                                   # (nf, 3, 3)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(fc_vals)
    if mask is not None:
        colors[mask] = mask_color
    pc = Poly3DCollection(polys, facecolors=colors, edgecolors="none",
                          linewidths=0.0, antialiased=False, rasterized=True)
    ax.add_collection3d(pc)

    lo, hi = v.min(0), v.max(0)
    d = hi - lo
    for setlim, a0, a1 in ((ax.set_xlim, lo[0], hi[0]),
                           (ax.set_ylim, lo[1], hi[1]),
                           (ax.set_zlim, lo[2], hi[2])):
        setlim(a0, a1)
    try:
        ax.set_box_aspect((d[0], d[1], d[2]))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if show_title:
        ax.set_title(title, pad=-2)
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.5, pad=-0.02, aspect=16)
    cb.set_label(cbar_label)
    return pc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--field", default="avg", choices=["min", "avg", "max"])
    ap.add_argument("--out", default="docs/figures/moon3d_500k/temp_3d.png")
    ap.add_argument("--display-faces", type=int, default=140000,
                    help="decimate to ~this many faces for matplotlib display")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-62.0)
    ap.add_argument("--mask-tmax", type=float, default=0.0,
                    help="draw faces whose coupled MAX-T exceeds this (K) in neutral "
                         "gray and exclude them from the color scale (view-factor "
                         "over-count artifacts); 0 = keep all")
    ap.add_argument("--pubclean", action="store_true",
                    help="Publication-clean variant (Paul's no-on-figure-notes rule): "
                         "suppress the upper suptitle AND the two per-panel headers "
                         "(content identification moves to the caption instead, "
                         "manuscript v4 review round 2, 2026-07-27). Colorbars/axis "
                         "labels unaffected. Additive, default OFF. NOTE: the first "
                         "pubclean pass (2026-07-22, temp_3d_native_twoband_v2) "
                         "suppressed the suptitle only -- this flag now ALSO drops "
                         "panel titles, matching the sibling scripts' --pubclean "
                         "convention (diviner_seasonal_compare.py etc.); does not "
                         "affect the already-shipped _v2 files.")
    ap.add_argument("--pubclean-canvas-px", default=None, metavar="WxH",
                    help="With --pubclean, pin the saved canvas to this exact pixel "
                         "size at 300 dpi so it matches the annotated (titled) "
                         "sibling's own shipped size 1:1 (the tight-bbox crop "
                         "otherwise shrinks once the suptitle is removed).")
    a = ap.parse_args()

    full = pv.read(a.ply)               # connectivity only (points are float32)
    ff = tri_faces(full)
    vf = read_ply_vertices_double(a.ply)   # full-precision DEM vertices
    assert len(vf) == full.n_points, "PLY vertex order mismatch"
    vf = vf - vf.mean(0)                 # recenter in double
    # South-polar patch: selenocentric +z points toward the north pole, so the
    # outward (topographic-up) direction is -z. Flip z so the surface renders
    # right-side-up with +z up in the figure.
    vf[:, 2] *= -1.0

    coupled = load_stat(Path(a.coupled), a.field)
    baseline = load_stat(Path(a.baseline), a.field)
    assert len(coupled) == len(ff) == len(baseline), \
        f"face/stat mismatch: {len(ff)} faces, {len(coupled)} coupled, {len(baseline)} baseline"
    diff = coupled - baseline

    # view-factor over-count artifact mask (on coupled MAX-T, not the rendered field)
    if a.mask_tmax > 0:
        hot = load_stat(Path(a.coupled), "max") > a.mask_tmax
        print(f"masking {int(hot.sum())} over-count faces "
              f"(coupled max-T > {a.mask_tmax:.0f} K) as neutral gray")
    else:
        hot = np.zeros(len(coupled), bool)
    good = ~hot

    # Render the full clean geometry directly: decimate_pro shatters this
    # open lunar patch (boundary + near-coplanar facets) into spikes, and the
    # mesh barely reduces anyway, so the full 509k faces are drawn (already
    # recentered above). matplotlib handles it with rasterized polygons.
    vd, fd = vf, ff
    coupled_d, diff_d = coupled, diff
    print(f"display mesh: {len(fd)} faces (full resolution)")

    apply_style()
    field_name = {"min": "minimum", "avg": "mean", "max": "maximum"}[a.field]
    fig = plt.figure(figsize=(DOUBLE_W, 0.52 * DOUBLE_W))
    axL = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    axR = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

    # Left-panel colorbar label: mathtext subscript matching the selected
    # field (avg/min/max), e.g. "$T_\mathrm{avg}$ (K)" -- Paul's ask, manuscript
    # v4 review round 2 (2026-07-27), replacing the generic "T (K)" so the
    # colorbar itself carries the field identity now that the panel header
    # (which used to say "Radiatively-coupled mean T") is suppressed under
    # --pubclean.
    cbar_label_L = rf"$T_\mathrm{{{a.field}}}$ (K)"
    nL = Normalize(np.percentile(coupled_d[good], 1), np.percentile(coupled_d[good], 99))
    render_panel(axL, vd, fd, coupled_d, "inferno", nL, a.elev, a.azim,
                 f"Radiatively-coupled {field_name} T", cbar_label_L, mask=hot,
                 show_title=not a.pubclean)
    axL.text2D(0.03, 0.95, "(a)", transform=axL.transAxes,
               fontsize=10, fontweight="bold", va="top")

    dmax = np.percentile(np.abs(diff_d[good]), 99.5)
    dmax = max(dmax, 1.0)
    if diff_d[good].min() < -0.5:
        nR = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax); cmapR = "RdBu_r"
    else:
        nR = Normalize(0.0, dmax); cmapR = "hot"
    render_panel(axR, vd, fd, diff_d, cmapR, nR, a.elev, a.azim,
                 f"Coupling effect (Δ{field_name} T)", "ΔT (K)", mask=hot,
                 show_title=not a.pubclean)
    axR.text2D(0.03, 0.95, "(b)", transform=axR.transAxes,
               fontsize=10, fontweight="bold", va="top")

    if not a.pubclean:
        fig.suptitle("Moon3D 509k — radiative coupling on the thermal field",
                     fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 1) if a.pubclean else (0, 0, 1, 0.95))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.pubclean_canvas_px:
        w, h = (int(v) for v in a.pubclean_canvas_px.lower().split("x"))
        written = savefig_pinned(fig, out.with_suffix(""), canvas_px=(w, h))
        print(f"wrote {written[0]} (+pdf)  [pinned canvas {w}x{h}px]")
    else:
        fig.savefig(out, dpi=300)
        if out.suffix == ".png":
            fig.savefig(out.with_suffix(".pdf"))
        print(f"wrote {out}" + (" (+pdf)" if out.suffix == ".png" else ""))
    dg = diff[good]
    print(f"  Δ{field_name}T (non-over-count faces): min={dg.min():+.1f} "
          f"max={dg.max():+.1f} mean={dg.mean():+.2f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
