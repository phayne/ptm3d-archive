#!/usr/bin/env python3
"""Register an occluder ring's inner hole to the core mesh's real boundary.

Background
----------
A3GT/CP-22 sites use the core+ring architecture (`docs/extended-occluder-
architecture.md`): a high-resolution thermally solved core PLY plus a coarse
LOLA occluder ring that is added to the shadow-ray BVH only. The two meshes
are built independently, and both ring builders used to punch the ring's
inner hole with a **nominal circle** test on the **triangle centroid**:

    keep face  <=>  geodesic(site_center, centroid(face)) >= R_core

That rule is wrong in a way that is invisible to every mesh-quality check:

  * it is a statement about the *centroid*, so a kept triangle's inward-most
    vertex still protrudes inside R_core by up to its centroid-to-vertex
    distance -- which at LOLA ring resolution is 100-130 m; and
  * R_core is a *nominal* radius, whereas the core mesh's actual outer
    boundary is its own clipped polygon and does not lie on that circle.

The result is ring surface lying planimetrically on top of core terrain. At
lunar polar sun elevations (0-5 deg) that overlapping lip casts a hard,
near-permanent spurious shadow on the core's edge faces. Measured at
lunar polar site A: 2,730 core faces covered by ring geometry, P(severe cold) 0.165
under the lip vs 0.031 clear, and no severely cold edge face anywhere more
than 200 m from the lip. Same-order overlap at lunar polar site B and lunar polar site C.

This module replaces the nominal circle with the core mesh's **actual outer
boundary polygon**, and adds a registration gate that refuses to emit a ring
that overlaps the core.

The ring is shadow-only geometry: it never enters a view-factor matrix, so
re-clipping a ring never requires a VF rebuild (the never-rebuild rule is untouched).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BODY_R_M = 1737400.0          # Moon (2015) sphere, matches the DTM tools
EDGE_SAMPLES = 64             # per triangle edge, for the exact overlap test


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def site_polar(lat_deg, lon_deg, site_lat_deg, site_lon_deg,
               body_r_m: float = BODY_R_M):
    """Great-circle distance (m) and azimuth (deg, 0=N, 90=E) from the site."""
    slat = np.radians(site_lat_deg)
    slon = np.radians(site_lon_deg)
    la = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lo = np.radians(np.asarray(lon_deg, dtype=np.float64))
    dlat = la - slat
    dlon = lo - slon
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(slat) * np.cos(la) * np.sin(dlon / 2.0) ** 2)
    dist = body_r_m * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    y = np.cos(la) * np.sin(dlon)
    x = np.cos(slat) * np.sin(la) - np.sin(slat) * np.cos(la) * np.cos(dlon)
    az = np.degrees(np.arctan2(y, x)) % 360.0
    return dist, az


def boundary_edges(tris: np.ndarray) -> np.ndarray:
    """Edges adjacent to exactly one face."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]],
                       axis=0)
    es = np.sort(e, axis=1)
    view = np.ascontiguousarray(es).view(
        np.dtype((np.void, es.dtype.itemsize * es.shape[1])))
    _, idx, counts = np.unique(view, return_index=True, return_counts=True)
    return es[idx[counts == 1]]


class CoreBoundary:
    """The core mesh's outer boundary as a radial profile r(azimuth).

    Validated on construction: exactly one closed boundary loop, every
    boundary vertex of degree 2, and single valued in azimuth about the site
    center (i.e. star shaped). Anything else raises -- a core mesh with
    interior holes or a re-entrant outer rim must not be silently registered
    against, because r(az) would then under-report the overlap.
    """

    def __init__(self, az: np.ndarray, r: np.ndarray, n_loops: int):
        order = np.argsort(az)
        self.az = az[order]
        self.r = r[order]
        self.n_loops = n_loops
        # wrapped pads so interpolation is continuous across 0/360
        self._az_p = np.concatenate([self.az - 360.0, self.az, self.az + 360.0])
        self._r_p = np.concatenate([self.r, self.r, self.r])

    def radius_at(self, az_deg):
        return np.interp(np.asarray(az_deg, dtype=np.float64) % 360.0,
                         self._az_p, self._r_p)

    @property
    def n_vertices(self):
        return self.az.size


def core_boundary_profile(core_ply: Path, site_lat: float, site_lon: float,
                          *, body_r_m: float = BODY_R_M) -> CoreBoundary:
    """Extract and validate the core mesh's outer boundary polygon."""
    lat, lon, _, tris = read_ply(core_ply)
    dist, az = site_polar(lat, lon, site_lat, site_lon, body_r_m)
    be = boundary_edges(tris)
    if be.size == 0:
        raise ValueError(
            f"{core_ply}: mesh has NO boundary edges (closed surface?) -- "
            "cannot register a ring against it")
    bv, deg = np.unique(be.ravel(), return_counts=True)
    if not np.all(deg == 2):
        raise ValueError(
            f"{core_ply}: {int((deg != 2).sum())} boundary vertices are not of "
            "degree 2 -- the boundary is not a set of simple closed loops "
            "(non-manifold or T-junction). Refusing to register against an "
            "ill-defined outer polygon; fix the core mesh first.")
    n_loops = count_loops(be, bv)
    if n_loops != 1:
        raise ValueError(
            f"{core_ply}: boundary has {n_loops} loops, expected exactly 1 "
            "(an outer rim with no interior holes). A core mesh with holes "
            "would let r(azimuth) under-report ring/core overlap. Refusing.")
    az_b, r_b = az[bv], dist[bv]
    # Star-shapedness about the site center. r(azimuth) is only a faithful
    # description of the boundary when it is single valued; on a re-entrant rim
    # it interpolates ACROSS the notch and under-reports overlap there, so a
    # ring lying in the notch would be neither clipped nor flagged. Refuse
    # rather than certify something the test cannot see. The wrap-around pair
    # (last vertex -> first) is included; a notch straddling azimuth 0 is
    # otherwise never examined.
    o = np.argsort(az_b)
    azs, rs = az_b[o], r_b[o]
    daz = np.diff(np.concatenate([azs, azs[:1] + 360.0]))
    dr = np.abs(np.diff(np.concatenate([rs, rs[:1]])))
    daz_m = daz * np.pi / 180.0 * np.maximum(
        np.concatenate([rs[:-1], rs[-1:]]), 1.0)
    steep = dr > np.maximum(daz_m, 1.0) * 4.0
    n_steep = int(steep.sum())
    if n_steep > 0.02 * azs.size:
        raise ValueError(
            f"{core_ply}: outer boundary is not single valued in azimuth "
            f"({n_steep} of {azs.size} neighboring vertex pairs jump radially "
            f"more than 4x their azimuthal separation; boundary r spans "
            f"{r_b.min():.0f}-{r_b.max():.0f} m). This is a property of the "
            "CORE MESH, not of any ring: its boundary is deeply re-entrant, so "
            "r(azimuth) cannot describe it and would silently under-report "
            "ring/core overlap. Inspect or rebuild the core mesh; do not "
            "register a ring against it.")
    if n_steep:
        print(f"[ring-registration] NOTE: {Path(core_ply).name}'s outer "
              f"boundary has {n_steep} of {azs.size} mildly re-entrant vertex "
              "pairs (under the 2% refusal threshold). Overlap may be "
              "under-reported at those azimuths.", file=sys.stderr)
    return CoreBoundary(az_b, r_b, n_loops)


def count_loops(be: np.ndarray, bv: np.ndarray) -> int:
    """Number of connected components of the boundary edge graph."""
    index = {int(v): i for i, v in enumerate(bv)}
    adj: list[list[int]] = [[] for _ in bv]
    for a, b in be:
        ia, ib = index[int(a)], index[int(b)]
        adj[ia].append(ib)
        adj[ib].append(ia)
    seen = np.zeros(bv.size, dtype=bool)
    loops = 0
    for s in range(bv.size):
        if seen[s]:
            continue
        loops += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
    return loops


# --------------------------------------------------------------------------
# the clip
# --------------------------------------------------------------------------
def _edge_sample_penetration(dist, az, tris, core: CoreBoundary,
                             cand: np.ndarray, margin_m: float,
                             chunk: int = 50_000):
    """Max penetration inside r_core(az)+margin along each candidate triangle's
    three edges.

    Chunked: an unbatched version allocates five (n_cand, EDGE_SAMPLES) float64
    temporaries, which reaches ~6 GB on a 2.8 M-face ring and can take the
    caller out with a MemoryError instead of a skipped site.

    Edge sampling alone does not prove a triangle is clear -- a core-boundary
    bulge entirely contained inside one ring triangle would touch no edge. The
    callers pair this with `_core_vertices_in_triangles`, which closes exactly
    that case."""
    if cand.size == 0:
        return np.zeros(0)
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)[None, :]
    worst = np.full(cand.size, -np.inf)
    a = np.radians(az)
    ex, ey = dist * np.sin(a), dist * np.cos(a)
    for c0 in range(0, cand.size, chunk):
        blk = cand[c0:c0 + chunk]
        sl = slice(c0, c0 + blk.size)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p0x = ex[tris[blk, i]][:, None]
            p0y = ey[tris[blk, i]][:, None]
            p1x = ex[tris[blk, j]][:, None]
            p1y = ey[tris[blk, j]][:, None]
            sx = p0x + (p1x - p0x) * t
            sy = p0y + (p1y - p0y) * t
            sr = np.hypot(sx, sy)
            saz = np.degrees(np.arctan2(sx, sy)) % 360.0
            pen = (core.radius_at(saz) + margin_m) - sr
            worst[sl] = np.maximum(worst[sl], pen.max(axis=1))
    return worst


def _core_vertices_in_triangles(dist, az, tris, core: CoreBoundary,
                                cand: np.ndarray) -> np.ndarray:
    """Candidate triangles that planimetrically CONTAIN a core boundary vertex.

    Catches the one case edge sampling cannot: a bulge of the core rim sitting
    entirely inside a single ring triangle, touching none of its three edges.
    Without this the gate's verdict would be stronger than its test."""
    if cand.size == 0:
        return np.zeros(0, dtype=bool)
    ca = np.radians(core.az)
    px, py = core.r * np.sin(ca), core.r * np.cos(ca)
    a = np.radians(az)
    ex, ey = dist * np.sin(a), dist * np.cos(a)
    ax, ay = ex[tris[cand, 0]], ey[tris[cand, 0]]
    bx, by = ex[tris[cand, 1]], ey[tris[cand, 1]]
    gx, gy = ex[tris[cand, 2]], ey[tris[cand, 2]]
    hit = np.zeros(cand.size, dtype=bool)
    for k in range(px.size):
        qx, qy = px[k], py[k]
        m = ((qx >= np.minimum(np.minimum(ax, bx), gx))
             & (qx <= np.maximum(np.maximum(ax, bx), gx))
             & (qy >= np.minimum(np.minimum(ay, by), gy))
             & (qy <= np.maximum(np.maximum(ay, by), gy)) & ~hit)
        if not m.any():
            continue
        d1 = (qx - bx[m]) * (ay[m] - by[m]) - (ax[m] - bx[m]) * (qy - by[m])
        d2 = (qx - gx[m]) * (by[m] - gy[m]) - (bx[m] - gx[m]) * (qy - gy[m])
        d3 = (qx - ax[m]) * (gy[m] - ay[m]) - (gx[m] - ax[m]) * (qy - ay[m])
        inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0))
                   & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
        idx = np.where(m)[0]
        hit[idx[inside]] = True
    return hit


def _candidate_band(dist, az, tris, core: CoreBoundary, margin_m: float):
    """Faces that could possibly reach the core rim.

    The band is derived from the ring's OWN largest near-hole facet, not a
    hardcoded constant: a ring built with a small point budget over a large
    R_ext can carry kilometer-scale facets, and a fixed band would let one span
    the hole unseen -- the original defect, certified clean."""
    vmin = dist[tris].min(axis=1)
    vmax = dist[tris].max(axis=1)
    r_rim = core.r.max() + max(margin_m, 0.0)
    near = vmin < r_rim + 2000.0
    span = float((vmax[near] - vmin[near]).max()) if near.any() else 0.0
    band = r_rim + max(2.0 * span, 1000.0)
    return np.where(vmin < band)[0]


def clip_ring_to_core(lat, lon, tris, core: CoreBoundary,
                      site_lat: float, site_lon: float, *,
                      margin_m: float = 0.0,
                      body_r_m: float = BODY_R_M):
    """Keep only ring faces that lie entirely outside the core boundary.

    Returns (keep_mask, dist, az)."""
    dist, az = site_polar(lat, lon, site_lat, site_lon, body_r_m)
    cand = _candidate_band(dist, az, tris, core, margin_m)
    keep = np.ones(tris.shape[0], dtype=bool)
    # cheap pass: any vertex strictly inside r_core(az)+margin
    v_pen = (core.radius_at(az[tris[cand]]) + margin_m) - dist[tris[cand]]
    keep[cand[(v_pen > 0.0).any(axis=1)]] = False
    # exact pass: an edge of a surviving candidate still crossing inside
    still = cand[keep[cand]]
    pen = _edge_sample_penetration(dist, az, tris, core, still, margin_m)
    keep[still[pen > 0.0]] = False
    # containment pass: a surviving candidate that swallows a core rim vertex
    left = still[pen <= 0.0]
    keep[left[_core_vertices_in_triangles(dist, az, tris, core, left)]] = False
    return keep, dist, az


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def registration_report(lat, lon, tris, core: CoreBoundary,
                        site_lat: float, site_lon: float, *,
                        az_bin_deg: float = 5.0,
                        body_r_m: float = BODY_R_M) -> dict:
    """Measure ring-vs-core overlap and the per-azimuth seam gap."""
    dist, az = site_polar(lat, lon, site_lat, site_lon, body_r_m)
    used = np.unique(tris)                 # ignore orphan vertices
    v_inside = dist[used] < core.radius_at(az[used])
    cand = _candidate_band(dist, az, tris, core, 0.0)
    pen = _edge_sample_penetration(dist, az, tris, core, cand, 0.0)
    contained = _core_vertices_in_triangles(dist, az, tris, core, cand)
    bad = cand[(pen > 0.0) | contained]

    # The ring's INNER hole boundary: boundary vertices near the core rim.
    # Selecting by "r < 2 * R_core" instead misreads a ring whose outer radius
    # is itself below 2 * R_core (the outer rim then reports as the seam) and
    # silently reports nothing at all for a ring with no hole.
    be = boundary_edges(tris)
    bv = np.unique(be.ravel())
    span = _near_hole_span(dist, tris, core)
    inner = bv[np.abs(dist[bv] - core.radius_at(az[bv]))
               < max(4.0 * span, 500.0)]
    gap = dist[inner] - core.radius_at(az[inner])

    nb = max(int(round(360.0 / az_bin_deg)), 1)
    prof_gap = np.full(nb, np.nan)
    if inner.size:
        b = (az[inner] / az_bin_deg).astype(int) % nb
        for k in range(nb):
            m = b == k
            if m.any():
                prof_gap[k] = gap[m].min()
    return {
        "n_ring_faces": int(tris.shape[0]),
        "n_overlapping_faces": int(bad.size),
        "n_vertices_inside_core": int(v_inside.sum()),
        "max_overlap_m": float(max(pen.max(), 0.0)) if pen.size else 0.0,
        "min_clearance_m": float(-pen.max()) if pen.size else float("nan"),
        "n_inner_boundary_vertices": int(inner.size),
        "gap_min_m": float(gap.min()) if gap.size else float("nan"),
        "gap_median_m": float(np.median(gap)) if gap.size else float("nan"),
        "gap_max_m": float(gap.max()) if gap.size else float("nan"),
        "az_bin_deg": az_bin_deg,
        "gap_profile_m": prof_gap,
        "core_boundary_r_min_m": float(core.r.min()),
        "core_boundary_r_max_m": float(core.r.max()),
        "core_boundary_vertices": int(core.n_vertices),
    }


def _near_hole_span(dist, tris, core: CoreBoundary) -> float:
    """Radial span of the ring's largest facet near the core rim (m)."""
    vmin = dist[tris].min(axis=1)
    vmax = dist[tris].max(axis=1)
    near = vmin < core.r.max() + 2000.0
    return float((vmax[near] - vmin[near]).max()) if near.any() else 0.0


def print_report(rep: dict, *, name: str = "", show_profile: bool = True,
                 stream=sys.stdout) -> None:
    p = lambda *a: print(*a, file=stream)          # noqa: E731
    p(f"[ring-registration] {name}")
    p(f"  core outer boundary : {rep['core_boundary_vertices']} vertices, "
      f"r = {rep['core_boundary_r_min_m']:.1f} .. "
      f"{rep['core_boundary_r_max_m']:.1f} m")
    p(f"  ring faces          : {rep['n_ring_faces']}")
    p(f"  ring vertices inside the core boundary : "
      f"{rep['n_vertices_inside_core']}")
    p(f"  ring faces overlapping the core        : "
      f"{rep['n_overlapping_faces']}")
    p(f"  max planimetric overlap                : {rep['max_overlap_m']:.1f} m"
      f"   (closest approach: {rep.get('min_clearance_m', float('nan')):.1f} m "
      "of clearance)")
    if rep["n_inner_boundary_vertices"] == 0:
        p("  seam gap: NOT MEASURED -- this ring has no boundary vertices near "
          "the core rim (no inner hole?). The gap profile below is empty.")
    p(f"  seam gap (ring inner boundary - core edge): "
      f"min {rep['gap_min_m']:.1f} m, median {rep['gap_median_m']:.1f} m, "
      f"max {rep['gap_max_m']:.1f} m")
    if show_profile:
        prof = rep["gap_profile_m"]
        step = rep["az_bin_deg"]
        p(f"  per-azimuth minimum seam gap (m), {step:g} deg bins "
          "(negative = OVERLAP):")
        for row0 in range(0, prof.size, 12):
            chunk = prof[row0:row0 + 12]
            head = f"    az {row0 * step:5.0f}-{(row0 + chunk.size) * step:5.0f}:"
            p(head + " " + " ".join(
                ("  n/a" if not np.isfinite(v) else f"{v:6.1f}") for v in chunk))


def gate(rep: dict, *, name: str = "", stream=sys.stdout) -> bool:
    """True if the ring is registered (no overlap). Prints the verdict."""
    ok = rep["n_overlapping_faces"] == 0 and rep["max_overlap_m"] <= 0.0
    if ok:
        print(f"[ring-registration] GATE PASS: no ring geometry lies inside "
              f"the core outer boundary.", file=stream)
    else:
        print(f"[ring-registration] GATE FAIL: {rep['n_overlapping_faces']} "
              f"ring face(s) overlap the core mesh planimetrically, by up to "
              f"{rep['max_overlap_m']:.1f} m.\n"
              f"  An overlapping ring lip casts a spurious hard shadow on core "
              f"edge faces at grazing sun and produces the >50 K discontinuous "
              f"cold rim artifact. Refusing to certify this ring.",
              file=stream)
    return ok


# --------------------------------------------------------------------------
# PLY reader (binary_little_endian, the DTM tool's schema)
# --------------------------------------------------------------------------
_PLY_T = {"double": "<f8", "float": "<f4", "int": "<i4", "uint": "<u4",
          "short": "<i2", "ushort": "<u2", "char": "|i1", "uchar": "|u1"}


def read_ply(path: Path):
    """Return (latitude, longitude, elevation, tris) from a binary PLY.

    Uses the per-vertex latitude/longitude the DTM tools write. If they are
    absent the values are derived from the Cartesian coordinates, and that
    substitution is announced -- it must never happen silently, because a
    body-radius or frame mismatch there would silently mis-register the ring.
    """
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY: {path}")
        n_v = n_f = 0
        vprops: list[tuple[str, str]] = []
        fextra: list[str] = []
        cur = None
        binary = False
        while True:
            ln = f.readline().decode("ascii", "replace").strip()
            if not ln:
                raise ValueError(f"{path}: truncated header")
            if ln.startswith("format"):
                binary = "binary_little_endian" in ln
            elif ln.startswith("element vertex"):
                n_v = int(ln.split()[-1]); cur = "v"
            elif ln.startswith("element face"):
                n_f = int(ln.split()[-1]); cur = "f"
            elif ln.startswith("property list"):
                pass
            elif ln.startswith("property") and cur == "v":
                t = ln.split()
                vprops.append((t[2], _PLY_T.get(t[1], "<f4")))
            elif ln.startswith("property") and cur == "f":
                t = ln.split()
                fextra.append(_PLY_T.get(t[1], "<f4"))
            elif ln == "end_header":
                break
        if not binary:
            raise ValueError(f"{path}: only binary_little_endian PLY supported")
        vdt = np.dtype(vprops)
        vert = np.frombuffer(f.read(n_v * vdt.itemsize), dtype=vdt).copy()
        blob = f.read()
    extra = sum(np.dtype(t).itemsize for t in fextra)
    stride = 1 + 12 + extra
    if len(blob) != n_f * stride:
        raise ValueError(f"{path}: face block {len(blob)} != {n_f}*{stride}")
    fview = np.frombuffer(blob, dtype=np.uint8).reshape(n_f, stride)
    if not np.all(fview[:, 0] == 3):
        raise ValueError(f"{path}: non-triangular face found")
    tris = fview[:, 1:13].copy().view("<i4").reshape(n_f, 3).astype(np.int64)
    names = {p[0] for p in vprops}
    if {"latitude", "longitude"} <= names:
        lat = vert["latitude"].astype(np.float64)
        lon = vert["longitude"].astype(np.float64)
    else:
        print(f"[ring-registration] NOTE: {path.name} carries no per-vertex "
              "latitude/longitude; deriving them from x,y,z on a "
              f"{BODY_R_M:.0f} m sphere.", file=sys.stderr)
        x = vert[vprops[0][0]].astype(np.float64)
        y = vert[vprops[1][0]].astype(np.float64)
        z = vert[vprops[2][0]].astype(np.float64)
        r = np.sqrt(x * x + y * y + z * z)
        lat = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-30), -1, 1)))
        lon = np.degrees(np.arctan2(y, x))
    elev = (vert["elevation"].astype(np.float64) if "elevation" in names
            else np.zeros(n_v))
    return lat, lon, elev, tris


# --------------------------------------------------------------------------
# CLI: check an existing ring
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure (and gate on) an occluder ring's planimetric "
                    "registration to its core mesh's outer boundary.")
    ap.add_argument("--core", required=True, type=Path)
    ap.add_argument("--ring", required=True, type=Path)
    ap.add_argument("--site-lat", type=float, required=True)
    ap.add_argument("--site-lon", type=float, required=True)
    ap.add_argument("--az-bin-deg", type=float, default=5.0)
    ap.add_argument("--body-radius-m", type=float, default=BODY_R_M)
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()

    try:
        core = core_boundary_profile(a.core, a.site_lat, a.site_lon,
                                     body_r_m=a.body_radius_m)
        lat, lon, _, tris = read_ply(a.ring)
    except (ValueError, OSError) as e:
        # Documented contract is a clear non-zero exit, not a stack trace at an
        # operator running this as a routine pre-flight check.
        print(f"[ring-registration] CANNOT CHECK: {e}", file=sys.stderr)
        return 2
    rep = registration_report(lat, lon, tris, core, a.site_lat, a.site_lon,
                              az_bin_deg=a.az_bin_deg,
                              body_r_m=a.body_radius_m)
    print_report(rep, name=a.ring.name, show_profile=not a.no_profile)
    return 0 if gate(rep, name=a.ring.name) else 1


if __name__ == "__main__":
    sys.exit(main())
