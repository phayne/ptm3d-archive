"""Nearest-neighbor equirectangular projection of per-face values to a lat/lon grid."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# Selected Europa landmarks (IAU east-positive longitude per the run convention)
# Cross-check: trailing apex at 90°E, leading apex at 270°E.
EUROPA_LANDMARKS = [
    # name,                     lat,   lon (E),  marker
    # Coordinates per Cartwright 2025 Fig 1 caption (Tara 10°S, 75°W; Powys 0°, 145°W).
    ("Tara Regio",               -10.0, 285.0, "chaos"),
    ("Powys Regio",                0.0, 215.0, "chaos"),
    ("Pwyll",                    -25.2,  89.0, "crater"),
    ("Cilix",                      1.5, 182.0, "crater"),
    ("Trailing apex",              0.0,  90.0, "ref"),
    ("Leading apex",               0.0, 270.0, "ref"),
    ("Sub-Jovian",                 0.0,   0.0, "ref"),
    ("Anti-Jovian",                0.0, 180.0, "ref"),
]


def project_to_grid(lat_face: np.ndarray,
                    lon_face: np.ndarray,
                    values: np.ndarray,
                    grid_dpd: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbor rasterization onto an equirectangular grid (lat/lon, deg).

    Returns (lon_grid_1d, lat_grid_1d, V_2d). V_2d shape is (n_lat, n_lon).
    Uses xyz-on-sphere kd-tree to avoid lon-wrap pathologies.
    """
    rad_lat = np.deg2rad(lat_face)
    rad_lon = np.deg2rad(lon_face)
    xyz_face = np.stack([
        np.cos(rad_lat) * np.cos(rad_lon),
        np.cos(rad_lat) * np.sin(rad_lon),
        np.sin(rad_lat),
    ], axis=-1)
    tree = cKDTree(xyz_face)

    n_lon = 360 * grid_dpd + 1
    n_lat = 180 * grid_dpd + 1
    lon_grid = np.linspace(0.0, 360.0, n_lon)
    lat_grid = np.linspace(-90.0, 90.0, n_lat)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    rad_LAT = np.deg2rad(LAT)
    rad_LON = np.deg2rad(LON)
    xyz_grid = np.stack([
        np.cos(rad_LAT) * np.cos(rad_LON),
        np.cos(rad_LAT) * np.sin(rad_LON),
        np.sin(rad_LAT),
    ], axis=-1).reshape(-1, 3)

    _, idx = tree.query(xyz_grid, k=1)
    V = values[idx].reshape(LAT.shape)
    return lon_grid, lat_grid, V


def annotate_landmarks(ax, kinds=("chaos", "ref"), color="white",
                       text_color=None, fontsize=8):
    """Overlay landmark markers and labels on an equirectangular axis."""
    if text_color is None:
        text_color = color
    for name, plat, plon, kind in EUROPA_LANDMARKS:
        if kinds is not None and kind not in kinds:
            continue
        marker = "o" if kind == "ref" else "s" if kind == "chaos" else "x"
        ax.scatter([plon], [plat], s=70 if kind != "ref" else 30,
                   facecolors="none", edgecolors=color, linewidths=1.4,
                   marker=marker, zorder=10)
        ax.annotate(name, (plon, plat), xytext=(6, 5), textcoords="offset points",
                    color=text_color, fontsize=fontsize, fontweight="bold",
                    zorder=11)
