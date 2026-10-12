#!/usr/bin/env python3
"""Mesh ladders for the Moon3D R1 benchmark (arms A-D). Writes binary PLYs and a
manifest (face count, md5, generator parameters, source md5) to <out>/manifest.json.

  A  Ingersoll bowl      exported by bowl_crater_ptm itself (PTM_BOWL_WRITE_PLY),
                         so the PLY is byte-for-byte the mesh the accuracy runs use.
  B  fBm rough surface   Gaussian random field, fixed seed, two variants:
                         band-limited  -- the SAME Fourier modes at every level,
                                          truncated at the coarsest level's Nyquist,
                                          so refinement adds sampling only;
                         extended      -- each level keeps every mode below its own
                                          Nyquist (the shared modes have identical
                                          coefficients), so refinement adds roughness.
                         Placed in a local tangent plane at (lat, lon) on the Moon.
  C  Moon3D decimations  VTK quadric decimation of the 1.7M-face original, native
                         selenocentric frame (the frame the v4 509k CSR was built in).
  D  Haworth             existing public meshes, verified and recorded, not modified.

EXCLUDED by ruling: every AGT/Artemis site. This script refuses any path that
names one (see AGT_PATTERNS).

Usage (europa_viz python: numpy, vtk):
  make_meshes.py --out ~/projects/moon3d_R1/bench/meshes --arms A,B,C,D [--dry-run]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, log, md5  # noqa: E402

R_MOON = 1737400.0
AGT_PATTERNS = re.compile(r"(A0\dM[PM]\d|A3GT|agt[_/]|artemis|/a3gt/|A03[A-Z]{2}\d)", re.I)

# Ladders (targets; actual counts are recorded in the manifest).
BOWL_N = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7]
FBM_N = [1e3, 1e4, 1e5, 1e6, 1e7]
MOON3D_N = [5e3, 2e4, 5e4, 1e5, 2.5e5, 5.09337e5]
MOON3D_SRC = Path.home() / "Library/CloudStorage/OneDrive-UCB-O365/moon/moon3d/moon3d_c_cartesian_NEW.vtk"
MOON3D_509K = Path.home() / "projects/moon3d/meshes/moon3d_c_cartesian_NEW_500k_clean.vtk"
HAWORTH_DIR = Path.home() / "Projects/ptm_data_staging/src/dem_validation/meshes"
HAWORTH = ["haworth_radcore_r25", "haworth_radcore_r25_100m", "haworth_radcore_r32c", "haworth_r15"]
HAWORTH_EXTRA = [Path.home() / "projects/moon3d/meshes/fig8_hybrid/sunlit_radcore_r50.ply"]


def refuse_agt(p):
    if AGT_PATTERNS.search(str(p)):
        raise SystemExit(f"FATAL: {p} looks like an AGT/Artemis site; excluded by ruling.")


def write_ply(path, V, F):
    """Binary little-endian PLY, double xyz, uchar+int32 faces (the layout
    mc_ref_pp and assemble_vf_haworth.py prove by file size)."""
    V = np.ascontiguousarray(V, dtype="<f8")
    F = np.ascontiguousarray(F, dtype="<i4")
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(V)}\nproperty double x\nproperty double y\nproperty double z\n"
           f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    rec = np.empty(len(F), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    rec["n"] = 3
    rec["v"] = F
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(V.tobytes())
        fh.write(rec.tobytes())


def tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


# ---------------------------------------------------------------- arm A: bowl
def bowl_params(n_target):
    """bowl_crater_ptm's mesh has nf = n_az (2 n_rings - 1 + 2 n_flat). Scale the
    gate's (72, 192, 14) uniformly so the aspect (and the floor fraction) is
    preserved down the ladder."""
    s = math.sqrt(n_target / (192 * (2 * 72 - 1 + 2 * 14)))
    nr, na, nfl = max(4, round(72 * s)), max(8, round(192 * s)), max(2, round(14 * s))
    return nr, na, nfl, na * (2 * nr - 1 + 2 * nfl)


def arm_a(out, bowl_bin, dds, dry):
    rows = []
    for n in BOWL_N:
        nr, na, nfl, nf = bowl_params(n)
        for dd in dds:
            prefix = out / f"bowl_n{nf}"
            ply = Path(f"{prefix}_dd{dd:.4g}.ply")
            row = {"arm": "A", "family": "bowl", "dd": dd, "n_target": n, "n_faces": nf,
                   "n_rings": nr, "n_az": na, "n_flat": nfl, "path": str(ply),
                   "mask": str(ply.with_name(ply.stem + "_mask.npy"))}
            if not dry and not ply.exists():
                env = dict(os.environ, PTM_BOWL_NRINGS=str(nr), PTM_BOWL_NAZ=str(na),
                           PTM_BOWL_NFLAT=str(nfl), PTM_BOWL_WRITE_PLY=str(prefix))
                r = subprocess.run([str(bowl_bin), str(out), str(dd)], env=env,
                                   capture_output=True, text=True)
                if r.returncode != 0 or not ply.exists():
                    raise SystemExit(f"FATAL: bowl export failed:\n{r.stdout}\n{r.stderr}")
            if ply.exists():
                row["md5"] = md5(ply)
                m = np.load(row["mask"])
                row["n_floor"] = int((m == 1).sum())
            rows.append(row)
    return rows


# ---------------------------------------------------------------- arm B: fBm
def enu(lat_deg, lon_deg):
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    E = np.array([-math.sin(lo), math.cos(lo), 0.0])
    N = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    U = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    return E, N, U


def fbm_levels(n_targets, L, hurst, rms_slope_deg, seed, lat, lon):
    """Yield (variant, n_target, V, F, info). Grid n x n -> 2 (n-1)^2 faces.
    h(x) = Re sum_k c_k exp(2 pi i k.x / L), |c_k|^2 ~ |k|^-(2H+2); the c_k are
    drawn ONCE on the finest grid (fixed seed) and cropped per level, so shared
    modes are identical across levels and variants."""
    ns = [int(round(math.sqrt(n / 2.0))) + 1 for n in n_targets]
    nmax, n0 = max(ns), min(ns)
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(nmax, d=1.0 / nmax)          # integer wavenumbers
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K = np.hypot(KX, KY)
    amp = np.zeros_like(K)
    amp[K > 0] = K[K > 0] ** (-(hurst + 1.0))
    C = amp * (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape))
    del amp, rng

    def field(n, kcut):
        """Evaluate h on an n x n grid using modes |kx|,|ky| < kcut (strict, so no
        mode sits on a Nyquist line where +k and -k alias)."""
        sel = (np.abs(KX) < kcut) & (np.abs(KY) < kcut)
        Cn = np.zeros((n, n), dtype=complex)
        ix = (KX[sel].astype(int)) % n
        iy = (KY[sel].astype(int)) % n
        Cn[ix, iy] = C[sel]
        h = np.real(np.fft.ifft2(Cn)) * n * n
        # spectral RMS slope of the evaluated field: sum |c|^2 (2 pi k / L)^2 / 2
        s2 = 0.5 * np.sum(np.abs(C[sel]) ** 2 * (2 * np.pi * K[sel] / L) ** 2)
        return h, math.sqrt(s2)

    # Amplitude: the band-limited field has the requested RMS slope.
    _, s_bl = field(n0, n0 / 2.0)
    scale = math.tan(math.radians(rms_slope_deg)) / s_bl
    E, N, U = enu(lat, lon)
    for variant in ("bandlimited", "extended"):
        for n, nt in zip(ns, n_targets):
            kcut = n0 / 2.0 if variant == "bandlimited" else n / 2.0
            h, s = field(n, kcut)
            h *= scale
            x = np.linspace(0.0, L, n, endpoint=False) - L / 2
            X, Y = np.meshgrid(x, x, indexing="ij")
            P = (R_MOON * U[None, :] + X.reshape(-1, 1) * E + Y.reshape(-1, 1) * N
                 + h.reshape(-1, 1) * U)
            idx = np.arange(n * n).reshape(n, n)
            a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
            c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
            # (x east, y north): a->b->c is counter-clockwise seen from +U = normal up
            F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
            info = {"grid_n": n, "kcut": kcut, "rms_slope_deg": math.degrees(math.atan(s * scale)),
                    "h_rms_m": float(h.std()), "domain_m": L, "hurst": hurst, "seed": seed,
                    "lat": lat, "lon": lon, "radius_m": R_MOON}
            yield variant, nt, P, F, info


def arm_b(out, dry, L, hurst, slope, seed, lat, lon):
    rows = []
    if dry:
        for variant in ("bandlimited", "extended"):
            for nt in FBM_N:
                n = int(round(math.sqrt(nt / 2.0))) + 1
                rows.append({"arm": "B", "family": f"fbm_{variant}", "n_target": nt,
                             "n_faces": 2 * (n - 1) ** 2, "grid_n": n})
        return rows
    for variant, nt, V, F, info in fbm_levels(FBM_N, L, hurst, slope, seed, lat, lon):
        ply = out / f"fbm_{variant}_n{len(F)}.ply"
        if not ply.exists():
            write_ply(ply, V, F)
        nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        up = (nrm @ enu(lat, lon)[2] > 0).mean()
        if up < 1.0:
            log(f"WARNING: {ply.name}: {100 * (1 - up):.3f}% of faces face down (overhang)")
        rows.append({"arm": "B", "family": f"fbm_{variant}", "n_target": nt,
                     "n_faces": int(len(F)), "path": str(ply), "md5": md5(ply), **info})
        log(f"B {variant} n={len(F)} slope={info['rms_slope_deg']:.1f} deg")
    return rows


# --------------------------------------------------------- arm C: Moon3D
def arm_c(out, dry):
    refuse_agt(MOON3D_SRC)
    rows = [{"arm": "C", "family": "moon3d_original", "n_target": 1.7e6, "path": str(MOON3D_SRC),
             "note": "1.7M original, native frame; nnz counted from the sweep only (no assembly)"},
            {"arm": "C", "family": "moon3d_509k_v4_existing", "n_target": 509337,
             "path": str(MOON3D_509K),
             "csr": str(Path.home() / "projects/moon3d/vf/resweep_v4_2026-08-24/moon3d509k/"
                        "moon3d_c_cartesian_NEW_500k_clean_v4.csr"),
             "note": "paper-lineage mesh and v4 CSR; NOT re-swept (PTM-19); cross-check point"}]
    if dry:
        for nt in MOON3D_N:
            rows.append({"arm": "C", "family": "moon3d_dec", "n_target": nt})
        return rows
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    rd = vtk.vtkPolyDataReader() if MOON3D_SRC.suffix == ".vtk" else vtk.vtkPLYReader()
    rd.SetFileName(str(MOON3D_SRC))
    rd.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(rd.GetOutputPort())
    tri.Update()
    src = tri.GetOutput()
    if src.GetPoints().GetData().GetDataTypeAsString() != "double":
        # vtkPLYReader demotes to float32: 0.125 m resolution at the lunar radius,
        # coarser than a 2.5 mm Moon3D facet. Never decimate from that.
        raise SystemExit(f"FATAL: {MOON3D_SRC} read as float32; need double precision points")
    n0 = src.GetNumberOfCells()
    rows[0].update(n_faces=int(n0), md5=md5(MOON3D_SRC))
    rows[1].update(md5=md5(MOON3D_509K))
    # vtkQuadricDecimation works and OUTPUTS in float32 and, at selenocentric
    # coordinates (|P| ~ 1.74e6 m), barely decimates at all (measured 2026-10-12:
    # 99% requested -> 0.1% achieved on the 509k mesh). Decimate in a
    # centroid-local frame (|P| ~ 1 m, float32 ~ 1e-7 m) and restore the offset
    # in double.
    P0 = vtk_to_numpy(src.GetPoints().GetData()).astype(np.float64)
    # mc_ref_pp reads only binary PLY: export the original too (zero-area faces dropped,
    # counted; a zero-area launch face has no normal and would cast undefined rays).
    F0 = vtk_to_numpy(src.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int32)
    a0 = tri_areas(P0, F0)
    bad0 = a0 <= 1e-14 * a0.max()
    orig_ply = out / f"moon3d_original_n{int((~bad0).sum())}.ply"
    if not orig_ply.exists():
        write_ply(orig_ply, P0, F0[~bad0])
    rows[0].update(path=str(orig_ply), n_faces=int((~bad0).sum()), md5=md5(orig_ply),
                   source=str(MOON3D_SRC), source_md5=rows[0]["md5"],
                   degenerate_dropped=int(bad0.sum()))
    ctr = P0.mean(axis=0)
    loc = vtk.vtkPolyData()
    loc.DeepCopy(src)
    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    pts.SetData(numpy_to_vtk(P0 - ctr, deep=True))
    loc.SetPoints(pts)
    for nt in MOON3D_N:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(loc)
        dec.SetTargetReduction(1.0 - nt / n0)
        dec.VolumePreservationOn()
        dec.Update()
        pd = dec.GetOutput()
        V = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64) + ctr
        F = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int32)
        a = tri_areas(V, F)
        bad = a <= 1e-14 * max(a.max(), 1e-300)
        if bad.any():
            log(f"WARNING: decimation to {nt:.0f}: dropping {int(bad.sum())} degenerate faces")
            F = F[~bad]
        ply = out / f"moon3d_dec_n{len(F)}.ply"
        write_ply(ply, V, F)
        rows.append({"arm": "C", "family": "moon3d_dec", "n_target": nt, "n_faces": int(len(F)),
                     "path": str(ply), "md5": md5(ply), "source": str(MOON3D_SRC),
                     "source_md5": rows[0]["md5"], "degenerate_dropped": int(bad.sum()),
                     "decimation": "vtkQuadricDecimation in a centroid-local frame, VolumePreservationOn, VTK "
                                   + vtk.vtkVersion.GetVTKVersion()})
        log(f"C decimated to {len(F)} faces")
    return rows


# --------------------------------------------------------- arm D: Haworth
def arm_d(dry):
    rows = []
    for p in [HAWORTH_DIR / f"{n}.ply" for n in HAWORTH] + HAWORTH_EXTRA:
        refuse_agt(p)
        row = {"arm": "D", "family": p.stem, "path": str(p),
               "note": "DEM-limited: refinement past the source DEM adds no topography "
                       "(source beats sampling); discretization-only"}
        if not dry:
            if not p.exists():
                raise SystemExit(f"FATAL: Haworth mesh missing: {p}")
            with open(p, "rb") as fh:
                head = fh.read(4096).decode("latin-1")
            m = re.search(r"element face (\d+)", head)
            row.update(n_faces=int(m.group(1)) if m else None, md5=md5(p))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--bowl-bin", default=str(REPO / "build_cpu/bin/bowl_crater_ptm"))
    ap.add_argument("--bowl-dd", default="0.2,0.1", help="d/D values (0.2 primary, 0.1 secondary)")
    ap.add_argument("--fbm-domain-m", type=float, default=10.0)
    ap.add_argument("--fbm-hurst", type=float, default=0.8)
    ap.add_argument("--fbm-rms-slope-deg", type=float, default=25.0)
    ap.add_argument("--fbm-seed", type=int, default=20261012)
    ap.add_argument("--lat", type=float, default=-85.0)
    ap.add_argument("--lon", type=float, default=0.0)
    ap.add_argument("--max-n", type=float, default=None,
                    help="drop ladder rungs above this face count (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="list the ladders; write nothing")
    a = ap.parse_args()
    if a.max_n:
        global BOWL_N, FBM_N, MOON3D_N
        BOWL_N = [n for n in BOWL_N if n <= a.max_n]
        FBM_N = [n for n in FBM_N if n <= a.max_n]
        MOON3D_N = [n for n in MOON3D_N if n <= a.max_n]
    out = Path(a.out).expanduser()
    refuse_agt(out)
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    arms = set(a.arms.split(","))
    rows = []
    if "A" in arms:
        rows += arm_a(out, a.bowl_bin, [float(x) for x in a.bowl_dd.split(",")], a.dry_run)
    if "B" in arms:
        rows += arm_b(out, a.dry_run, a.fbm_domain_m, a.fbm_hurst, a.fbm_rms_slope_deg,
                      a.fbm_seed, a.lat, a.lon)
    if "C" in arms:
        rows += arm_c(out, a.dry_run)
    if "D" in arms:
        rows += arm_d(a.dry_run)
    if a.dry_run:
        for r in rows:
            print(json.dumps(r))
        return
    man = out / "manifest.json"
    old = json.loads(man.read_text()) if man.exists() else []
    keep = [r for r in old if r.get("arm") not in arms]
    man.write_text(json.dumps(keep + rows, indent=1))
    log(f"wrote {man} ({len(keep + rows)} meshes)")


if __name__ == "__main__":
    main()
