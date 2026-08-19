#!/usr/bin/env python3
"""Finding 1, decisive measurement: the ENGINE's own view-factor row for a level
facet, entry by entry, against the flux cube's zenith row at parity height.

This is a PURELY GEOMETRIC comparison. No temperature, no radiosity, no solver
enters it, so it cannot be confounded by any thermal difference -- if the two
rows agree the flux gap is not a view-factor gap, and if they disagree the gap
is exactly the disagreement.

Three-way outcome:
  A. Row sums agree (within MC error) -> the cube's VF row is right.
     RESOLVED 2026-08-19: this is the outcome against the CORRECTED matrix. The
     original 0.51x was outcome B, and the defect was in the offline assembler
     (divide by hit count rather than by 2), not in the cube.
  B. The ENGINE's row sum is systematically LARGER than the ray-traced one, by
     about the observed flux ratio -> the engine over-counts near-field view
     factor on level terrain. Documented failure class; the cube is the
     instrument that found it.
  C. The cube's row sum is larger, or the two disagree in a way that does not
     track the flux ratio -> the cube geometry is wrong and Finding 1 is mine.

The engine row is read directly out of the corrected CSR by seeking, so nothing
is re-derived and no intermediate tool can distort it. Format (documented at
radiation_exchange.cpp:265): int32 n, float64 geomhash, int64 indptr[n+1],
int64 nnz, int32 cols[nnz], float data[nnz].
"""
import argparse, struct, sys
import numpy as np

HDR = "<4sIIIQdQQddddddIIQdddd12ddII"
CUBE = "<iidd3dif4fd"
PATCH = "<QII6f"
FACE = ["zenith", "nadir", "north", "south", "east", "west"]


def load_fcg(path):
    b = open(path, "rb").read()
    hs = struct.calcsize(HDR); v = struct.unpack_from(HDR, b, 0)
    nc, nf, nnz, jl, hn = v[2], v[3], v[4], v[-2], v[-1]
    cs, ps = struct.calcsize(CUBE), struct.calcsize(PATCH)
    cubes = [struct.unpack_from(CUBE, b, hs + i * cs) for i in range(nc)]
    off = hs + nc * cs
    pats = [struct.unpack_from(PATCH, b, off + k * ps) for k in range(nc * nf)]
    off += nc * nf * ps
    cols = np.frombuffer(b, dtype="<i4", count=nnz, offset=off); off += 4 * nnz
    vals = np.frombuffer(b, dtype="<f4", count=nnz, offset=off); off += 4 * nnz
    names = []
    if hn:
        noff = off + jl
        names = [b[noff + k * 32: noff + k * 32 + 32].split(b"\0")[0].decode() for k in range(nc)]
    return v, cubes, pats, cols, vals, names


class CSR:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.n = struct.unpack("<i", self.f.read(4))[0]
        self.gh = struct.unpack("<d", self.f.read(8))[0]
        self.indptr_off = 12
        self.f.seek(12 + 8 * (self.n + 1))
        self.nnz = struct.unpack("<q", self.f.read(8))[0]
        self.cols_off = 12 + 8 * (self.n + 1) + 8
        self.data_off = self.cols_off + 4 * self.nnz

    def row(self, j):
        self.f.seek(self.indptr_off + 8 * j)
        a, b = struct.unpack("<qq", self.f.read(16))
        k = b - a
        if k <= 0:
            return np.empty(0, "<i4"), np.empty(0, "<f4")
        self.f.seek(self.cols_off + 4 * a)
        c = np.frombuffer(self.f.read(4 * k), dtype="<i4")
        self.f.seek(self.data_off + 4 * a)
        d = np.frombuffer(self.f.read(4 * k), dtype="<f4")
        return c, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcg", required=True)
    ap.add_argument("--csr", required=True)
    ap.add_argument("--face", default="zenith")
    ap.add_argument("--max-cubes", type=int, default=40)
    a = ap.parse_args()
    fi = FACE.index(a.face)

    v, cubes, pats, cols, vals, names = load_fcg(a.fcg)
    csr = CSR(a.csr)
    print("cache: %d cubes, %d rays/patch, height %.4f m, geomhash %.10g"
          % (v[2], v[14], v[12], v[5]))
    print("CSR  : n=%d nnz=%d geomhash=%.10g" % (csr.n, csr.nnz, csr.gh))
    if abs(csr.gh - v[5]) > 1e-3 * abs(v[5]):
        print("FATAL: geomhash mismatch -- different meshes, comparison meaningless")
        return 2
    print("mesh identity confirmed (same geomhash)\n")

    print("Comparing the cube's %s row against the ENGINE's own VF row for the SAME facet.\n"
          % a.face)
    print("%-6s %9s %10s %10s %10s %8s %8s %9s" %
          ("name", "face_id", "cube_sumF", "eng_sumF", "ratio", "cube_nnz", "eng_nnz", "overlapF"))
    rc, re_, rr = [], [], []
    for k in range(min(a.max_cubes, v[2])):
        c = cubes[k]
        j = c[7]
        p = pats[k * 6 + fi]
        o, ln = p[0], p[1]
        ccol = cols[o:o + ln].astype(np.int64)
        cval = vals[o:o + ln].astype(np.float64)
        ecol, eval_ = csr.row(j)
        ecol = ecol.astype(np.int64); eval_ = eval_.astype(np.float64)
        cs_, es = cval.sum(), eval_.sum()
        # weight the engine puts on faces the cube also sees
        inter = np.isin(ecol, ccol)
        ov = eval_[inter].sum() if ecol.size else 0.0
        rc.append(cs_); re_.append(es); rr.append(cs_ / es if es > 0 else np.nan)
        print("%-6s %9d %10.6f %10.6f %10.4f %8d %8d %9.6f"
              % (names[k] if names else str(k), j, cs_, es,
                 cs_ / es if es > 0 else float("nan"), ln, ecol.size, ov))
    rc = np.array(rc); re_ = np.array(re_); rr = np.array(rr)
    print("\n--- summary over %d level facets ---" % len(rc))
    print("cube  sum_j F : min %.6f  median %.6f  max %.6f" % (rc.min(), np.median(rc), rc.max()))
    print("engine sum_j F: min %.6f  median %.6f  max %.6f" % (re_.min(), np.median(re_), re_.max()))
    print("ratio cube/engine: min %.4f  median %.4f  max %.4f" % (np.nanmin(rr), np.nanmedian(rr), np.nanmax(rr)))
    print("\nINTERPRETATION")
    m = np.nanmedian(rr)
    if 0.9 <= m <= 1.1:
        print("  Outcome A: the two view-factor rows AGREE.")
        print("  NOTE (2026-08-19): against the ORIGINAL MP026 matrix this ratio was 0.51 and")
        print("  outcome B held. The cause was found in the OFFLINE ASSEMBLER -- entries were")
        print("  divided by the hit count instead of by 2, inflating far-field view factors by")
        print("  up to exactly 2x (fixed on fix/vf-assembler-divisor). Against the re-assembled")
        print("  matrix, with the cube's rows UNCHANGED, the ratio is ~0.98-0.99. If you are")
        print("  seeing outcome A, you are pointed at a corrected matrix; if you see 0.51, you")
        print("  are pointed at a pre-fix one.")
    elif m < 0.9:
        print("  Outcome B: the ENGINE's row carries %.2fx the geometric weight the ray-traced" % (1.0 / m))
        print("  hemisphere finds, on LEVEL terrain where the true answer is bounded by the")
        print("  sky. If that tracks the flux ratio, the engine over-counts near-field view")
        print("  factor and the cube is the instrument that found it.")
    else:
        print("  Outcome C: the CUBE's row is larger. Finding 1 is then mine, not the engine's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
