#!/usr/bin/env python3
"""Four-panel 3D-globe view of the surface ice phase regime.

Renders the same field shown in `timescales_map.png` (log10 of
τ_cryst / τ_amorph; RdBu_r, ±6 dynamic range) on a Europa-shaped globe,
viewed from each of the four cardinal hemispheres:

  - Sub-Jovian apex     (0°E)
  - Trailing apex       (90°E)   ← peak amorphization
  - Anti-Jovian apex   (180°E)
  - Leading apex       (270°E)   ← Tara/Powys side

Light-gray lat/lon graticule lines (every 30°) overlaid on each globe.
A single shared horizontal colorbar at the bottom of the 2×2 grid.

Output: `docs/figures/jwst/regime_globes.{png,pdf}` (or override via --out).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mass_balance as mb_mod  # noqa: E402
import crystallization as cr  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "python"))
import extract_timeseries as _ets  # noqa: E402

SECONDS_PER_YEAR = 365.25 * 86400.0

HEMI_VIEWS = [
    ("Sub-Jovian (0°E)",   0.0,   0.0),
    ("Trailing (90°E)",    0.0,  90.0),
    ("Anti-Jovian (180°E)", 0.0, 180.0),
    ("Leading (270°E)",    0.0, 270.0),
]

# Tara, Powys — same labels as the equirect map. Stored as (name, lat, lon_E).
LANDMARKS = [
    ("Tara",  -10.0, 285.0),
    ("Powys",   0.0, 215.0),
]


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def _ll_to_xyz(lat_deg, lon_deg, r=1.0):
    """Spherical (lat, lon[°E]) → unit-sphere Cartesian. Hand-rolled to ensure
    the convention matches PTM3D's mesh.ply (east-positive longitude, with
    sub-Jovian at 0°E)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def load_pv_mesh(run_dir: Path):
    """Load the PTM3D mesh.ply, returning (pv.PolyData, face_lat_deg,
    face_lon_deg). Vertex positions are normalized to a unit sphere so the
    camera setup below is body-radius-independent."""
    verts, faces, _, lat, lon = _ets.read_binary_mesh_flexible(
        str(run_dir / "mesh.ply"))
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / np.where(norms > 0, norms, 1.0)

    n_faces = faces.shape[0]
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64), faces
    ]).ravel()
    mesh = pv.PolyData(verts_unit, pv_faces)
    return mesh, lat, lon


# ---------------------------------------------------------------------------
# Graticules
# ---------------------------------------------------------------------------

def _visible_segments(pts, view_dir, min_dot=0.05):
    """Split a polyline into segments containing only points whose unit
    normal is on the visible hemisphere (pts·view_dir > min_dot)."""
    pts = np.asarray(pts)
    norms = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    visible = norms @ view_dir > min_dot
    segments = []
    cur = []
    for p, v in zip(pts, visible):
        if v:
            cur.append(p)
        else:
            if len(cur) >= 2:
                segments.append(np.asarray(cur))
            cur = []
    if len(cur) >= 2:
        segments.append(np.asarray(cur))
    return segments


def graticule_polydatas(view_lat, view_lon,
                        lat_step=30.0, lon_step=30.0,
                        n_points=721, r=1.005):
    """Return PolyData line segments for the lat/lon graticules, already
    clipped to the visible hemisphere of the (view_lat, view_lon) apex."""
    view_dir = _ll_to_xyz(view_lat, view_lon, r=1.0)
    view_dir = view_dir / np.linalg.norm(view_dir)

    polys = []
    # Latitude circles (full 360°, clipped to visible half)
    for lat_deg in np.arange(-90 + lat_step, 90, lat_step):
        lons = np.linspace(0, 360, n_points)
        pts = _ll_to_xyz(np.full(n_points, lat_deg), lons, r=r)
        for seg in _visible_segments(pts, view_dir):
            polys.append(pv.lines_from_points(seg))
    # Longitude meridians (full pole-to-pole arcs)
    for lon_deg in np.arange(0, 360, lon_step):
        lats = np.linspace(-89.5, 89.5, n_points)
        pts = _ll_to_xyz(lats, np.full(n_points, lon_deg), r=r)
        for seg in _visible_segments(pts, view_dir):
            polys.append(pv.lines_from_points(seg))
    return polys


# ---------------------------------------------------------------------------
# Single-globe render
# ---------------------------------------------------------------------------

def render_globe(mesh, scalars, vmin, vmax, cmap,
                 view_lat, view_lon, out_path,
                 graticule_polys, landmarks_pv):
    """Render one globe view to `out_path` (PNG with transparent background)."""
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1400])
    plotter.set_background("white")
    plotter.enable_depth_peeling(8)

    mesh["scalar"] = scalars
    plotter.add_mesh(
        mesh, scalars="scalar", cmap=cmap, clim=[vmin, vmax],
        show_edges=False, lighting=False, smooth_shading=False,
        show_scalar_bar=False,
    )

    # Graticules — light gray, slightly outside the mesh sphere so they
    # render cleanly on top.
    for poly in graticule_polys:
        plotter.add_mesh(poly, color="#9a9a9a", line_width=2.0,
                         lighting=False)

    # Landmark markers — only render the ones on the visible hemisphere.
    view_dir = _ll_to_xyz(view_lat, view_lon, r=1.0)
    view_dir = view_dir / np.linalg.norm(view_dir)
    for name, lat, lon in landmarks_pv:
        pt = _ll_to_xyz(lat, lon, r=1.0)
        if pt @ view_dir < 0.05:
            continue
        pt_top = _ll_to_xyz(lat, lon, r=1.025)
        sphere = pv.Sphere(radius=0.025, center=pt_top)
        plotter.add_mesh(sphere, color="white", lighting=False)
        ring = pv.Sphere(radius=0.030, center=pt_top)
        plotter.add_mesh(ring, color="black", style="wireframe",
                         line_width=1.6, lighting=False)

    # Use parallel (orthographic) projection so the rendered hemisphere
    # fills the panel cleanly, with no perspective foreshortening.
    plotter.enable_parallel_projection()
    apex = _ll_to_xyz(view_lat, view_lon, r=1.0)
    cam_pos = apex / np.linalg.norm(apex) * 4.0
    view_up = np.array([0.0, 0.0, 1.0])
    plotter.camera_position = [tuple(cam_pos), (0.0, 0.0, 0.0), tuple(view_up)]
    plotter.camera.parallel_scale = 1.05

    plotter.screenshot(out_path, transparent_background=True)
    plotter.close()


def _autocrop(path, pad=10):
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img.split()[3])
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(r0 - pad, 0); r1 = min(r1 + pad, img.height - 1)
    c0 = max(c0 - pad, 0); c1 = min(c1 + pad, img.width - 1)
    img.crop((c0, r0, c1 + 1, r1 + 1)).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/jwst/regime_globes.png"))
    args = p.parse_args()

    print(f"Loading mass balance from {args.run}…")
    mb = mb_mod.load_mass_balance(args.run)

    # log10(τ_cryst / τ_amorph). Same color mapping as `timescales_map.png`.
    porosity = cr.porosity_from_latitude(mb["lat"])
    k_cr = cr.k_cryst(mb["T_max"], porosity)
    k_am = cr.k_amorph(mb["dose_rate"])
    big = 1e15
    tau_cryst = np.where(k_cr > 1e-30, 1.0 / (k_cr * SECONDS_PER_YEAR), big)
    tau_amorph = np.where(k_am > 1e-30, 1.0 / (k_am * SECONDS_PER_YEAR), big)
    log_ratio = np.log10(np.maximum(tau_cryst, 1e-30)
                         / np.maximum(tau_amorph, 1e-30))
    log_ratio = np.clip(log_ratio, -6.0, 6.0)

    print(f"Loading mesh from {args.run}/mesh.ply…")
    mesh, _, _ = load_pv_mesh(args.run)
    print(f"  {mesh.n_points} verts, {mesh.n_cells} faces")

    # Render each hemisphere to a temp PNG then composite via matplotlib.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        panel_imgs = []
        for label, vlat, vlon in HEMI_VIEWS:
            tmp = td / f"hemi_{int(vlon):03d}.png"
            print(f"Rendering {label}…")
            grats = graticule_polydatas(vlat, vlon)
            render_globe(mesh, log_ratio, vmin=-6.0, vmax=6.0, cmap="RdBu_r",
                         view_lat=vlat, view_lon=vlon, out_path=str(tmp),
                         graticule_polys=grats, landmarks_pv=LANDMARKS)
            _autocrop(str(tmp), pad=20)
            panel_imgs.append((label, tmp))

        # 2x2 matplotlib mosaic
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 11.5),
                                gridspec_kw=dict(wspace=0.02, hspace=0.06))
        for ax, (label, png) in zip(axs.flat, panel_imgs):
            from PIL import Image
            ax.imshow(np.asarray(Image.open(png)))
            ax.set_title(label, fontsize=18, pad=6)
            ax.set_axis_off()

        # Shared colorbar (horizontal, below the grid)
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=-6, vmax=6))
        sm.set_array([])
        cax = fig.add_axes([0.18, 0.05, 0.65, 0.022])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                          ticks=[-6, -3, 0, 3, 6])
        cb.ax.set_xticklabels([r"$10^{-6}$" + "\nCrystalline",
                                r"$10^{-3}$",
                                "1\n(boundary)",
                                r"$10^{3}$",
                                r"$10^{6}$" + "\nAmorphous"], fontsize=11)
        cb.set_label(r"$\tau_{\mathrm{cryst}}\,/\,\tau_{\mathrm{amorph}}$",
                     fontsize=14)

        fig.suptitle("Surface phase regime — global view", fontsize=20, y=0.97)
        fig.subplots_adjust(top=0.93, bottom=0.11, left=0.04, right=0.96)

        out_png = args.out
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {out_png}")
        print(f"Wrote {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    raise SystemExit(main())
