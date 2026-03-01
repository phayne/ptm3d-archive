from __future__ import annotations

"""
Convert DAMIT shape models to Wavefront OBJ format for PTM3D mesh import.

DAMIT vertices are typically unitless (normalized to unit volume). This module
scales them by the physical radius (from HORIZONS) and writes standard OBJ.
"""

import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)


class ShapeConverter:
    """Convert DAMIT shape data to OBJ files suitable for PTM3D."""

    @staticmethod
    def damit_to_obj(
        vertices: list[list[float]],
        faces: list[list[int]],
        output_path: str,
        radius_km: float = 0.0,
        body_name: str = "",
    ) -> str:
        """
        Convert DAMIT shape model to Wavefront OBJ format.

        Parameters
        ----------
        vertices : list of [x, y, z]
            DAMIT vertex coordinates (unitless or in km).
        faces : list of [v1, v2, v3]
            Triangle face indices (0-based).
        output_path : str
            Path to write the OBJ file.
        radius_km : float
            Physical mean radius in km. If > 0, vertices are scaled from
            unitless to km (assuming DAMIT normalization to unit volume).
            If 0, vertices are written as-is.
        body_name : str
            Body name for the OBJ comment header.

        Returns
        -------
        str
            Path to the written OBJ file.
        """
        if not vertices or not faces:
            raise ValueError("Empty vertices or faces list")

        # Compute current mean radius of the shape model
        current_mean_r = _compute_mean_radius(vertices)

        # Determine scale factor
        if radius_km > 0 and current_mean_r > 0:
            scale = radius_km / current_mean_r
            logger.info(
                "Scaling DAMIT shape: current mean radius %.4f -> %.2f km (scale=%.4f)",
                current_mean_r, radius_km, scale
            )
        else:
            scale = 1.0
            logger.info("Writing DAMIT shape without scaling (mean radius=%.4f)", current_mean_r)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"# Wavefront OBJ - PTM3D shape model\n")
            if body_name:
                f.write(f"# Body: {body_name}\n")
            if radius_km > 0:
                f.write(f"# Scaled to mean radius: {radius_km} km\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
            f.write(f"#\n")

            # Write vertices (scaled to km)
            for v in vertices:
                f.write(f"v {v[0] * scale:.8f} {v[1] * scale:.8f} {v[2] * scale:.8f}\n")

            # Write faces (OBJ uses 1-based indices)
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        logger.info("Wrote OBJ file: %s (%d vertices, %d faces)",
                     output_path, len(vertices), len(faces))
        return output_path

    @staticmethod
    def damit_to_obj_ellipsoidal(
        vertices: list[list[float]],
        faces: list[list[int]],
        output_path: str,
        semi_axes_km: tuple[float, float, float],
        body_name: str = "",
    ) -> str:
        """
        Convert DAMIT shape model to OBJ with ellipsoidal scaling.

        Instead of uniform radius scaling, applies direction-dependent
        scaling to match tri-axial ellipsoid dimensions.

        Parameters
        ----------
        vertices : list of [x, y, z]
            DAMIT vertex coordinates (unitless).
        faces : list of [v1, v2, v3]
            Triangle face indices (0-based).
        output_path : str
            Path to write the OBJ file.
        semi_axes_km : tuple of (a, b, c)
            Semi-axes along x, y, z in km.
        body_name : str
            Body name for the header.
        """
        if not vertices or not faces:
            raise ValueError("Empty vertices or faces list")

        # Find the extent of the shape model along each axis
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]

        extent_x = (max(xs) - min(xs)) / 2.0
        extent_y = (max(ys) - min(ys)) / 2.0
        extent_z = (max(zs) - min(zs)) / 2.0

        a_km, b_km, c_km = semi_axes_km

        # Scale factors per axis
        sx = a_km / extent_x if extent_x > 0 else 1.0
        sy = b_km / extent_y if extent_y > 0 else 1.0
        sz = c_km / extent_z if extent_z > 0 else 1.0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"# Wavefront OBJ - PTM3D shape model (ellipsoidal scaling)\n")
            if body_name:
                f.write(f"# Body: {body_name}\n")
            f.write(f"# Semi-axes: ({a_km}, {b_km}, {c_km}) km\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
            f.write(f"#\n")

            for v in vertices:
                f.write(f"v {v[0] * sx:.8f} {v[1] * sy:.8f} {v[2] * sz:.8f}\n")

            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        logger.info("Wrote ellipsoidal OBJ: %s", output_path)
        return output_path


def _compute_mean_radius(vertices: list[list[float]]) -> float:
    """Compute the mean distance of vertices from the origin."""
    if not vertices:
        return 0.0
    total = 0.0
    for v in vertices:
        total += math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return total / len(vertices)
