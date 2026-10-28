#!/usr/bin/env python3
"""Shared surface-side rule regression test: mc_ref_pp traces the ENGINE's surface side, never the raw winding alone.

Synthetic, no external data: a 1.2 m bowl (depth 0.25 m, ~2,400 faces) placed on the lunar
surface at 85 deg S, written twice -- wound OUTWARD (sky side) and wound INWARD (the inward-wound Moon3D mesh
Moon3D 509k situation). The current referee (ensure_mc_ref_pp.resolve_current) must:
  O1  outward mesh: 0 faces reversed under both rules, and partner records BYTE-IDENTICAL to
      PTM_PP_ORIENT=raw (the pre-fix behavior) -> outward-wound meshes are unaffected.
  O2  inward mesh, no acknowledgment: exit 5 before tracing, no partner .bin written.
  O3  inward mesh + PTM_ILLUM_TRUE_NORMALS=1 + PTM_PP_ORIENT_EXPECT_REVERSAL=1: every face
      reversed, and kept partners / SumF equal to the outward mesh to round-off (the two
      differ only in vertex order: centroid summation order and Embree's triangle).
  O4  inward mesh + PTM_PP_ORIENT=raw: the UNDERSIDE -- far fewer partners (this is the bug).
  O5  the ORIENTATION line and the STATS orientation keys are present and agree.
Exit 0 on PASS. Run: python3 scripts/ptm12_diag/test_mc_ref_pp_orientation.py
"""
import json, os, re, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ensure_mc_ref_pp

R_MOON = 1737400.0
NRAYS = 20000


def bowl_mesh(n=36):
    """Local grid over [-0.6, 0.6]^2 m with a spherical-cap bowl, then placed tangent to the Moon
    at 85 S, 30 E. Returns body-centered vertices and OUTWARD-wound faces."""
    xs = np.linspace(-0.6, 0.6, n)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    Z = np.where(r < 0.5, -0.25 * (1 - (r / 0.5) ** 2), 0.0)
    lat, lon = np.radians(-85.0), np.radians(30.0)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    e1 = np.cross([0, 0, 1.0], up); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    P = (R_MOON * up)[None, :] + X.reshape(-1, 1) * e1 + Y.reshape(-1, 1) * e2 + Z.reshape(-1, 1) * up
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            F += [(a, b, d), (a, d, c)]
    F = np.array(F)
    # make the winding outward (normal . up > 0) on every face of this height field
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = nrm @ up < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, F


def write_ply(path, P, F):
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty double x\n"
           f"property double y\nproperty double z\nelement face {len(F)}\n"
           "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(np.asarray(P, "<f8").tobytes())
        for f in F:
            fh.write(struct.pack("<B3i", 3, *map(int, f)))


def run(binary, work, ply, tag, env):
    nf = int(open(ply, "rb").read(400).split(b"element face ")[1].split(b"\n")[0])
    tmax = work / "tmax.bin"
    tmax.write_bytes(struct.pack("<i", nf) + np.zeros(nf, "<f4").tobytes())
    fl = work / "faces.txt"
    fl.write_text("\n".join(str(i) for i in range(0, nf, 7)) + "\n")   # every 7th face
    outb = work / f"{tag}.bin"
    if outb.exists():
        outb.unlink()
    e = {k: v for k, v in os.environ.items() if not k.startswith("PTM_")}
    e.update(env)
    r = subprocess.run([str(binary), str(ply), str(tmax), str(fl), str(NRAYS), "1e-4", "0.2", "1", str(outb)],
                       capture_output=True, text=True, env=e, timeout=600)
    toks = [ln.split() for ln in r.stdout.splitlines() if ln and not ln.startswith("#")]
    if any(len(t) < 6 for t in toks):
        raise SystemExit("FAIL: this referee prints no nkept column (it predates f1d8995 and the shared surface-side rule)")
    rows = {int(t[0]): (float(t[1]), int(t[5])) for t in toks}
    orient = next((ln for ln in r.stderr.splitlines() if ln.startswith("# ORIENTATION")), None)
    stats = next((json.loads(ln[8:]) for ln in r.stderr.splitlines() if ln.startswith("# STATS ")), None)
    return r.returncode, rows, (outb.read_bytes() if outb.exists() else None), orient, stats


def main():
    binary = ensure_mc_ref_pp.resolve_current()
    fails = []
    with tempfile.TemporaryDirectory(prefix="mc_ref_pp_orient_") as w:
        work = Path(w)
        P, F = bowl_mesh()
        out_ply, in_ply = work / "bowl_out.ply", work / "bowl_in.ply"
        write_ply(out_ply, P, F)
        write_ply(in_ply, P, F[:, [0, 2, 1]])
        # O1
        rc_raw, rows_raw, bin_raw, _, _ = run(binary, work, out_ply, "raw", {"PTM_PP_ORIENT": "raw"})
        for rule, env in (("radial_per_face", {}), ("winding_majority", {"PTM_ILLUM_TRUE_NORMALS": "1"})):
            rc, rows, b, orient, stats = run(binary, work, out_ply, "o1", env)
            if rc != 0 or b != bin_raw or not orient or f"rule={rule}" not in orient or " reversed=0/" not in orient:
                fails.append(f"O1 {rule}: rc={rc} identical_to_raw={b == bin_raw} orient={orient}")
            # O5
            if not stats or stats.get("orientation_rule") != rule or stats.get("orientation_reversed") != 0 \
                    or stats.get("orientation_digest") not in (orient or ""):
                fails.append(f"O5 {rule}: STATS orientation keys {stats and {k: v for k, v in stats.items() if 'orient' in k}}")
        # O2
        rc, _, b, orient, _ = run(binary, work, in_ply, "o2", {"PTM_ILLUM_TRUE_NORMALS": "1"})
        if rc != 5 or b is not None:
            fails.append(f"O2: expected exit 5 and no .bin, got rc={rc} bin={'yes' if b else 'no'}")
        # O3
        rc, rows_in, b, orient, _ = run(binary, work, in_ply, "o3",
                                        {"PTM_ILLUM_TRUE_NORMALS": "1", "PTM_PP_ORIENT_EXPECT_REVERSAL": "1"})
        nf = len(F)
        k_out = np.array([rows_raw[f][1] for f in sorted(rows_raw)])
        s_out = np.array([rows_raw[f][0] for f in sorted(rows_raw)])
        k_in = np.array([rows_in[f][1] for f in sorted(rows_raw)]) if rc == 0 else None
        s_in = np.array([rows_in[f][0] for f in sorted(rows_raw)]) if rc == 0 else None
        if rc != 0 or f"reversed={nf}/{nf}" not in (orient or "") or \
                abs(k_in.mean() / k_out.mean() - 1) > 1e-3 or np.abs(s_in - s_out).max() > 2e-4:
            fails.append(f"O3: rc={rc} orient={orient} kept {None if k_in is None else k_in.mean():} vs {k_out.mean():.2f}")
        # O4 (documents the bug: the underside of a bowl sees far less)
        rc, rows_u, _, _, _ = run(binary, work, in_ply, "o4", {"PTM_PP_ORIENT": "raw"})
        k_u = np.array([rows_u[f][1] for f in sorted(rows_raw)])
        if rc != 0 or not k_u.mean() < 0.5 * k_out.mean():
            fails.append(f"O4: underside kept {k_u.mean():.1f} vs sky side {k_out.mean():.1f} (expected < 50%)")
        print(f"sky side kept/face {k_out.mean():.1f}  SumF {s_out.mean():.4f} | inward+engine rule kept "
              f"{'n/a' if k_in is None else f'{k_in.mean():.1f}'} | underside (raw) kept {k_u.mean():.1f}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails)); return 1
    print("PASS: mc_ref_pp orientation O1-O5"); return 0


if __name__ == "__main__":
    sys.exit(main())
