#!/usr/bin/env python3
"""Prototype of HORIZON-ELEVATION visibility for the crater-floor view factor,
to validate the fix for the fine-mesh floor->rim coupling collapse before
implementing it in the C++ VF module.

Idea (matches the DEM horizon scan, robust to grazing tangency): from face i,
in each azimuth build the running-max terrain elevation angle vs distance over a
heightfield rasterized from the mesh. A target face j is visible iff its
elevation angle θ_j (from i) is >= the running-max horizon angle BEFORE its
distance d_j (i.e. no nearer terrain rises above the sightline). This replaces
the single 3D centroid ray, which over-occludes grazing floor->rim paths.

Compares, for the floor, F_terrain / F_warm under (a) the M1 3D-ray proxy and
(b) horizon visibility, at 100k vs 400k, vs the DEM ground truth (0.0055).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_floor_vf import face_centroids  # noqa: E402

MD = Path("data/dem/validation/meshes")
SIG = 5.670374419e-8  # SI-2019 (exact); canonical: src/cpp/constants/stefan_boltzmann.h


def load(mesh):
    from plyfile import PlyData
    pd = PlyData.read(str(MD / mesh))
    v = np.stack([np.asarray(pd['vertex']['x'], float),
                  np.asarray(pd['vertex']['y'], float),
                  np.asarray(pd['vertex']['z'], float)], 1)
    vn = np.stack([np.asarray(pd['vertex']['nx'], float),
                   np.asarray(pd['vertex']['ny'], float),
                   np.asarray(pd['vertex']['nz'], float)], 1)
    fd = pd['face'].data
    fld = next(n for n in ('vertex_indices', 'vertex_index') if n in fd.dtype.names)
    F = np.stack([np.asarray(r, np.int64) for r in fd[fld]])
    p0, p1, p2 = v[F[:, 0]], v[F[:, 1]], v[F[:, 2]]
    c = (p0 + p1 + p2) / 3.0
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    # face normal from geometry, oriented to agree with the PLY vertex normals
    ng = np.cross(p1 - p0, p2 - p0); ng /= (2 * A[:, None] + 1e-30)
    vavg = (vn[F[:, 0]] + vn[F[:, 1]] + vn[F[:, 2]])
    flip = np.sum(ng * vavg, axis=1) < 0
    ng[flip] *= -1
    return c, ng, A


def enu_frame(c):
    """Local ENU at the crater center; up = radial (selenocentric)."""
    C = c.mean(0)
    up = C / np.linalg.norm(C)
    ref = np.array([0, 0, 1.0])
    east = np.cross(up, ref); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    e = (c - C) @ east
    n = (c - C) @ north
    u = (c - C) @ up
    # face normal vertical component (for backface): n_face . up
    return np.stack([e, n, u], 1), np.array([east, north, up])


def main():
    LO, HI = 250.0, 1e9   # warm threshold (K)
    N_AZ = 180
    CELL = 40.0           # heightfield cell (m)
    DSTEP = 40.0          # march step (m)
    for tag, mesh, tmaxf in [
        ("100k", "haworth_radcore_r25.ply", "hmfix_haworth_r25_fixed"),
        ("400k", "haworth_radcore_r25_100m.ply", "hmfix_haworth_r25_100m_fixed"),
    ]:
        c, nrm, A = load(mesh)
        nf = len(A)
        P, B = enu_frame(c)          # P[:,0]=e P[:,1]=n P[:,2]=u
        e, n, u = P[:, 0], P[:, 1], P[:, 2]
        tmax = np.fromfile(Path("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/output")
                           / tmaxf / "statistics/temperature_max.bin", np.float32)[:nf].astype(float)

        # heightfield (max u per cell)
        emin, nmin = e.min(), n.min()
        ne = int((e.max() - emin) / CELL) + 2
        nn = int((n.max() - nmin) / CELL) + 2
        H = np.full((ne, nn), -1e9)
        ie = ((e - emin) / CELL).astype(int)
        iN = ((n - nmin) / CELL).astype(int)
        for k in range(nf):
            if u[k] > H[ie[k], iN[k]]:
                H[ie[k], iN[k]] = u[k]
        # floor probe faces: within 2500 m of center, lowest 25% u
        r = np.hypot(e, n)
        near = r < 2500.0
        thr = np.quantile(u[near], 0.25)
        floor = np.where(near & (u <= thr))[0]
        step = max(1, len(floor) // 40)
        probes = floor[::step]

        azs = np.linspace(0, 2 * np.pi, N_AZ, endpoint=False)
        sgrid = np.arange(DSTEP, 30000.0, DSTEP)

        F_ray = []  # straight-line "M1-ray" proxy on the heightfield (running-max ON exact line)
        F_hzn = []  # horizon visibility
        Fw_hzn = []
        for pf in probes:
            ei, ni, ui = e[pf], n[pf], u[pf]
            # per-azimuth running-max horizon angle vs distance
            # store horizon(az, dist) running max
            hor = np.full((N_AZ, len(sgrid)), -np.pi / 2)
            for ai, az in enumerate(azs):
                ce, cn = np.cos(az), np.sin(az)
                xs = ei + sgrid * ce
                ys = ni + sgrid * cn
                ce_i = ((xs - emin) / CELL).astype(int)
                cn_i = ((ys - nmin) / CELL).astype(int)
                ok = (ce_i >= 0) & (ce_i < ne) & (cn_i >= 0) & (cn_i < nn)
                ang = np.full(len(sgrid), -np.pi / 2)
                hz = H[ce_i[ok], cn_i[ok]]
                valid = hz > -1e8
                dd = sgrid[ok]
                a = np.full(ok.sum(), -np.pi / 2)
                a[valid] = np.arctan2(hz[valid] - ui, dd[valid])
                ang[ok] = a
                hor[ai] = np.maximum.accumulate(ang)
            # F_terrain total via horizon (sky-complement)
            hmax = hor[:, -1]
            F_hzn.append(np.mean(np.sin(np.clip(hmax, 0, None))**2))

            # per-target coupling for WARM faces (and all, for F_terrain check)
            de = e - ei; dn = n - ni; du = u - ui
            d = np.sqrt(de*de + dn*dn + du*du)
            sel = (d > 1.0)
            azt = (np.arctan2(dn, de)) % (2*np.pi)
            tht = np.arctan2(du, np.sqrt(de*de+dn*dn))
            # M1 form factor with backface + grazing cull (as production)
            r_hat = np.stack([de, dn, du], 1) / (d[:, None] + 1e-30)
            # project r_hat to ENU then dot with face normals (also in selenocentric);
            # use the actual probe-face normal for cos_i (not global up).
            rhat3 = np.stack([de, dn, du], 1) / (d[:, None] + 1e-30)
            rhat_sel = rhat3 @ B                    # ENU->selenocentric components via B rows
            cos_i = rhat_sel @ nrm[pf]              # probe face normal . r_hat
            cos_j = -(rhat_sel * nrm).sum(1)        # target normal . (-r_hat)
            cull = sel & (cos_i > 0.01) & (cos_j > 0.01)
            estF = np.zeros(nf)
            estF[cull] = (A[cull] / np.pi) * (cos_i[cull] * cos_j[cull]) / (d[cull]**2)
            warm = tmax > LO
            # horizon visibility per target: running-max horizon angle just BEFORE d_j
            ai_t = (azt / (2*np.pi) * N_AZ).astype(int) % N_AZ
            di_t = np.clip((d / DSTEP).astype(int) - 1, 0, len(sgrid) - 1)
            hor_at = hor[ai_t, di_t]
            vis_h = tht >= (hor_at - np.deg2rad(0.5))   # visible iff at/above running horizon
            Fw_hzn.append(estF[cull & warm & vis_h].sum())
            if pf == probes[0]:
                w = warm & sel
                ci_ok = w & (cos_i > 0.01); cj_pos = w & (cos_j > 0.0); cj_ok = w & (cos_j > 0.01)
                both = w & (cos_i > 0.01) & (cos_j > 0.01)
                print(f"   [{tag} probe0] warm={w.sum()}  cos_i>0.01:{ci_ok.sum()}  "
                      f"cos_j>0(faces floor):{cj_pos.sum()}  cos_j>0.01:{cj_ok.sum()}  both:{both.sum()}  "
                      f"| cos_i med={np.median(cos_i[w]):.3f}  cos_j med(of cos_j>0)={np.median(cos_j[cj_pos]) if cj_pos.sum() else 0:.4f}")
        print(f"[{tag}] floor probes={len(probes)}  "
              f"F_terrain(horizon)={np.mean(F_hzn):.4f} (DEM truth 0.0055)  "
              f"F_warm(horizon-vis)={np.mean(Fw_hzn):.5f}  (ray-trace prod: 100k 0.0040 / 400k ~0)")


if __name__ == "__main__":
    main()
