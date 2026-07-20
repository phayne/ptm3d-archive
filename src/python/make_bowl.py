#!/usr/bin/env python3
"""Generate a spherical-cap bowl-crater mesh (+ flat apron, + optional fractal
roughness) as a binary PLY with outward vertex normals, for the controlled
view-factor estimator harness. Known analytic floor terrain VF (Ingersoll):
the floor center sees its rim at elevation beta_c = atan(2 d / D), so
F_terrain(center) = sin^2(beta_c)  (= 1 - F_sky).

Usage:
  make_bowl.py --dd 0.05 --diameter 20000 --n-rings 72 --n-az 192 \
               --rough-deg 0 --out /tmp/bowl.ply
"""
from __future__ import annotations
import argparse
import struct
import numpy as np


def build(dd, D, n_rings, n_az, n_flat_rings, flat_extent, rough_deg, seed):
    r_rim = D / 2.0
    d = dd * D
    R = (r_rim**2 + d**2) / (2.0 * d)
    z_c = R - d
    th_max = np.arcsin(r_rim / R)
    V = [np.array([0.0, 0.0, z_c - R])]          # floor center (z = -d)
    for k in range(1, n_rings + 1):
        th = th_max * k / n_rings
        rho, z = R * np.sin(th), z_c - R * np.cos(th)
        for j in range(n_az):
            phi = 2 * np.pi * j / n_az
            V.append(np.array([rho * np.cos(phi), rho * np.sin(phi), z]))
    rim0 = 1 + (n_rings - 1) * n_az
    H = flat_extent * r_rim
    for k in range(1, n_flat_rings + 1):
        t = k / n_flat_rings
        for j in range(n_az):
            phi = 2 * np.pi * j / n_az
            sc = r_rim + t * (H - r_rim)
            V.append(np.array([sc * np.cos(phi), sc * np.sin(phi), 0.0]))
    V = np.array(V)

    # faces
    Fc = []
    for j in range(n_az):
        Fc.append((0, 1 + j, 1 + (j + 1) % n_az))
    for k in range(1, n_rings):
        bi, bo = 1 + (k - 1) * n_az, 1 + k * n_az
        for j in range(n_az):
            j1 = (j + 1) % n_az
            Fc.append((bi + j, bo + j, bo + j1))
            Fc.append((bi + j, bo + j1, bi + j1))
    ring_starts = [rim0 - n_az] + [rim0 + (k - 1) * n_az for k in range(1, n_flat_rings + 1)]
    for ri in range(len(ring_starts) - 1):
        bi, bo = ring_starts[ri], ring_starts[ri + 1]
        for j in range(n_az):
            j1 = (j + 1) % n_az
            Fc.append((bi + j, bo + j, bo + j1))
            Fc.append((bi + j, bo + j1, bi + j1))
    Fc = np.array(Fc, np.int64)

    # Facet-scale roughness: per-vertex Gaussian z-noise with std proportional to
    # the LOCAL mesh spacing, so adjacent facets tilt by ~rough_deg RMS regardless
    # of resolution (finer mesh -> smaller bumps, same slope). Reproduces the
    # constant-RMS-slope, resolution-dependent grazing-occlusion regime of Haworth.
    if rough_deg > 0:
        rng = np.random.default_rng(seed)
        rho = np.hypot(V[:, 0], V[:, 1])
        spacing = r_rim / n_rings                         # radial vertex spacing
        std = 0.5 * spacing * np.tan(np.deg2rad(rough_deg))
        inside = rho <= r_rim
        V[inside, 2] += rng.normal(0.0, std, inside.sum())

    # face normals (orient outward = +z up out of the cavity) -> vertex normals
    p0, p1, p2 = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30)
    fn[fn[:, 2] < 0] *= -1                                    # outward = up
    vn = np.zeros_like(V)
    for t in range(3):
        np.add.at(vn, Fc[:, t], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30)
    return V, Fc, vn, dict(beta_c=np.arctan2(2 * d, D), F_terrain=np.sin(np.arctan2(2 * d, D))**2)


def write_ply(path, V, Fc, vn):
    with open(path, "wb") as f:
        hdr = ("ply\nformat binary_little_endian 1.0\n"
               f"element vertex {len(V)}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property float nx\nproperty float ny\nproperty float nz\n"
               f"element face {len(Fc)}\n"
               "property list uchar int vertex_indices\nend_header\n")
        f.write(hdr.encode())
        buf = np.empty((len(V), 6), np.float32)
        buf[:, :3] = V; buf[:, 3:] = vn
        f.write(buf.tobytes())
        for tri in Fc:
            f.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd", type=float, default=0.05)
    ap.add_argument("--diameter", type=float, default=20000.0)
    ap.add_argument("--n-rings", type=int, default=72)
    ap.add_argument("--n-az", type=int, default=192)
    ap.add_argument("--n-flat-rings", type=int, default=14)
    ap.add_argument("--flat-extent", type=float, default=1.7)
    ap.add_argument("--rough-deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V, Fc, vn, info = build(a.dd, a.diameter, a.n_rings, a.n_az, a.n_flat_rings,
                            a.flat_extent, a.rough_deg, a.seed)
    write_ply(a.out, V, Fc, vn)
    print(f"wrote {a.out}: {len(V)} verts, {len(Fc)} faces  "
          f"beta_c={np.degrees(info['beta_c']):.2f} deg  "
          f"F_terrain(center) analytic = {info['F_terrain']:.5f}  rough={a.rough_deg} deg")


if __name__ == "__main__":
    main()
