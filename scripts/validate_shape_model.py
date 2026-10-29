#!/usr/bin/env python3
"""Closed-form validation of a triangulated shape model, before it ships.

A shape model that LOADS is not a shape model that is right. The failures this
catches are the quiet ones: a mesh in meters where the run expects kilometers,
a surface with holes that the radiation solver will happily integrate over, an
inward-facing facet winding that flips every normal, a body whose center of
mass is not at the origin the engine assumes it is.

Everything here is a closed-form integral over the triangles -- no sampling, no
tolerance to tune -- so the numbers are comparable against a published value:

  volume, surface area      divergence-theorem sums over the facets
  volume-equivalent radius  R_vol = (3V / 4pi)^(1/3)
  centroid                  the VOLUME centroid, which is the center of mass
                            of a uniform-density body (what a shape model's
                            "origin at the center of mass" claim means)
  inertia tensor            exact uniform-density polyhedron integral, and its
                            eigenvectors -- the principal axes
  topology                  boundary edges counted EXACTLY as the engine counts
                            them, by vertex INDEX

That last one is the point of this tool rather than a general mesh library.
``IlluminationModel::initializeScene`` decides open-vs-closed by hashing
undirected edges as pairs of vertex INDICES
(``src/cpp/illumination/illumination_model.cpp:454``), and that verdict gates
the analytic body-occlusion pre-check. A mesh that is closed as a SURFACE but
carries duplicate vertices at a seam reports two half-edges under different
indices, so the engine sees boundary edges and calls it an open patch. Checking
closure geometrically would agree with the archive and disagree with the run.
So this reproduces the index rule, and reports the geometric picture separately.

Unreferenced vertices are reported rather than treated as an error. A primary
archive product is shipped BYTE-FOR-BYTE so its checksum still means something,
and the OSIRIS-REx OLA global models carry a few hundred unreferenced duplicate
vertices by construction -- inert for the engine (they are in no facet) but they
do make V - E + F differ from 2, which looks alarming until it is named.

Usage::

    python3 scripts/validate_shape_model.py <mesh.obj> [...]
    python3 scripts/validate_shape_model.py new.obj --compare old.obj
    python3 scripts/validate_shape_model.py new.obj \\
        --expect-radius-km 0.24503 --expect-closed --tolerance-percent 1.0

Exit status is 1 if any ``--expect-*`` check fails, so this can gate a commit.
With no expectations it only reports, and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys

import numpy as np

# The engine's kilometers-vs-meters discriminator on the median |face center|
# (ptm_thermal_sim.cpp, the mesh length-unit guard). Repeated here ONLY to predict what
# the run will print; the engine remains the authority, and this tool says
# "predicts" rather than "is" for exactly that reason.
UNIT_DISCRIMINATOR = 10000.0
AMBIGUOUS_FACTOR = 50.0


def load_obj(path):
    """Vertices and 0-based triangle indices from a Wavefront OBJ.

    Deliberately as permissive as the engine's own parser
    (``src/cpp/mesh/mesh_utils.cpp:1265``): ``v``/``f`` only, ``f`` entries may
    carry ``v/vt/vn`` decorations, everything else is a comment. Reading it the
    same way is what makes a disagreement here a real disagreement.
    """
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]),
                                 float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:4]]
                # OBJ is 1-based; negative indices are relative to the end.
                faces.append([i - 1 if i > 0 else len(vertices) + i
                              for i in idx])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def obj_header(path, limit=200):
    """The leading ``#`` comment block, which is where these models keep their
    own provenance (product version, data source, and the producer's OWN
    closure and volume numbers -- the thing worth disagreeing with)."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                out.append(line.rstrip("\n"))
                if len(out) >= limit:
                    break
            elif line.strip():
                break
    return out


def file_digests(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha.hexdigest(), size


def topology(vertices, faces):
    """Edge bookkeeping, by vertex index, exactly as the engine does it."""
    a = faces[:, [0, 1, 2]].ravel()
    b = faces[:, [1, 2, 0]].ravel()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo.astype(np.uint64) << np.uint64(32) | hi.astype(np.uint64)
    _, counts = np.unique(key, return_counts=True)

    referenced = np.unique(faces)
    # Duplicate POSITIONS among referenced vertices are what would split a
    # closed surface into two index-distinct half-edges.
    used = vertices[referenced]
    dup_referenced = used.shape[0] - np.unique(used, axis=0).shape[0]

    return {
        "unique_edges": int(counts.size),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "referenced_vertices": int(referenced.size),
        "unreferenced_vertices": int(vertices.shape[0] - referenced.size),
        "duplicate_referenced_vertices": int(dup_referenced),
        "euler_all": int(vertices.shape[0] - counts.size + faces.shape[0]),
        "euler_referenced": int(referenced.size - counts.size + faces.shape[0]),
    }


def geometry(vertices, faces):
    """Volume, area, centroid and inertia by exact integration.

    Each facet plus the origin forms a tetrahedron; summing signed tetrahedron
    contributions gives the enclosed volume for ANY closed surface, whether or
    not the origin is inside it. The inertia integral uses the same
    decomposition with the canonical tetrahedron covariance, so it is exact for
    a uniform-density polyhedron rather than a facet-center approximation.
    """
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    # 6 * signed tetrahedron volume, positive when the winding is outward.
    six_v = np.einsum("ij,ij->i", p0, np.cross(p1, p2))
    volume = six_v.sum() / 6.0

    # Volume centroid: each tetrahedron's centroid is the mean of its four
    # vertices, and the origin contributes nothing.
    centroid = (np.einsum("i,ij->j", six_v, (p0 + p1 + p2)) / 4.0) / (
        6.0 * volume) if volume != 0 else np.zeros(3)

    # Uniform-density covariance, canonical tetrahedron form.
    canonical = np.array([[2.0, 1.0, 1.0],
                          [1.0, 2.0, 1.0],
                          [1.0, 1.0, 2.0]]) / 120.0
    tri = np.stack([p0, p1, p2], axis=1)              # (F, 3 vertices, 3 xyz)
    cov = np.einsum("f,fji,jk,fkl->il", six_v, tri, canonical, tri)
    inertia = np.trace(cov) * np.eye(3) - cov

    eigenvalues, eigenvectors = np.linalg.eigh(inertia)

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    radius = np.linalg.norm(vertices, axis=1)

    # Extent along the PRINCIPAL axes, not the model axes. Published "overall
    # dimensions" for a small body are quoted this way, and for a near-oblate
    # spinning top the two in-plane principal axes are nearly degenerate, so
    # they can sit tens of degrees from the model's x and y while the model
    # frame is still perfectly correct. Measuring both is what keeps a 7 deg
    # in-plane rotation from being misread as a frame error.
    referenced = np.unique(faces)
    projected = (vertices[referenced] - centroid) @ eigenvectors
    principal_extent = projected.max(axis=0) - projected.min(axis=0)

    return {
        "volume": float(volume),
        "surface_area": float(areas.sum()),
        "r_vol": float((3.0 * abs(volume) / (4.0 * math.pi)) ** (1.0 / 3.0)),
        "centroid": centroid,
        "extent_lo": lo,
        "extent_hi": hi,
        "extent": hi - lo,
        # Ordered longest-first, which is the order "565 x 535 x 508" is in.
        "principal_extent": np.sort(principal_extent)[::-1],
        "inward_facets": int(np.count_nonzero(six_v < 0.0)),
        "degenerate_facets": int(np.count_nonzero(areas <= 0.0)),
        "facet_area_mean": float(areas.mean()),
        "facet_area_min": float(areas.min()),
        "facet_area_max": float(areas.max()),
        "edge_length_mean": float(np.linalg.norm(
            np.concatenate([p1 - p0, p2 - p1, p0 - p2]), axis=1).mean()),
        "vertex_radius_median": float(np.median(radius)),
        "vertex_radius_min": float(radius.min()),
        "vertex_radius_max": float(radius.max()),
        "face_center_radius_median": float(np.median(
            np.linalg.norm((p0 + p1 + p2) / 3.0, axis=1))),
        "inertia_eigenvalues": eigenvalues,
        "principal_axes": eigenvectors.T,
    }


def predict_units(face_center_median, body_radius_km):
    """What the engine's mesh length-unit guard will print for this mesh, and why.

    Reported as a prediction rather than a fact: the guard is in the engine and
    a second copy of a rule is a rule that drifts. Run the engine to confirm.
    """
    meters = face_center_median > UNIT_DISCRIMINATOR
    verdict = "METERS" if meters else "KILOMETERS"
    ambiguous = (UNIT_DISCRIMINATOR / AMBIGUOUS_FACTOR < face_center_median
                 < UNIT_DISCRIMINATOR * AMBIGUOUS_FACTOR)
    note = ""
    if body_radius_km and body_radius_km > 0.0:
        ratio = face_center_median / body_radius_km
        if 0.5 <= ratio <= 2.0:
            note = (f"confirmed by R_body: |face_center| / R_body = "
                    f"{ratio:.4f} assuming KILOMETERS")
        elif 0.5 <= ratio / 1000.0 <= 2.0:
            note = (f"confirmed by R_body assuming METERS "
                    f"(ratio {ratio / 1000.0:.4f})")
        else:
            note = (f"NOT confirmed by R_body = {body_radius_km} km "
                    f"(ratio {ratio:.4g} as km)")
    if ambiguous and "confirmed by" not in note:
        note += ("  FATAL RISK: median sits in the ambiguous band "
                 f"({UNIT_DISCRIMINATOR / AMBIGUOUS_FACTOR:g} .. "
                 f"{UNIT_DISCRIMINATOR * AMBIGUOUS_FACTOR:g}) with no radius "
                 "confirmation; the engine refuses this case.")
    return verdict, note


def describe(path, body_radius_km=0.0, show_header=False):
    vertices, faces = load_obj(path)
    if vertices.size == 0 or faces.size == 0:
        sys.exit(f"ERROR: {path} yielded {len(vertices)} vertices and "
                 f"{len(faces)} facets; this is not a triangulated OBJ.")
    md5, sha256, size = file_digests(path)
    topo = topology(vertices, faces)
    geom = geometry(vertices, faces)
    verdict, note = predict_units(geom["face_center_radius_median"],
                                  body_radius_km)

    print(f"\n=== {path}")
    print(f"  bytes {size}  md5 {md5}")
    print(f"  sha256 {sha256}")
    if show_header:
        for line in obj_header(path):
            print(f"  | {line}")

    print("  -- counts")
    print(f"    vertices                  {vertices.shape[0]}")
    print(f"    facets                    {faces.shape[0]}")
    print(f"    referenced vertices       {topo['referenced_vertices']}")
    print(f"    unreferenced vertices     {topo['unreferenced_vertices']}")
    print(f"    duplicate referenced      "
          f"{topo['duplicate_referenced_vertices']}")

    print("  -- topology (engine rule: boundary edges by vertex INDEX)")
    print(f"    unique edges              {topo['unique_edges']}")
    print(f"    boundary edges            {topo['boundary_edges']}")
    print(f"    non-manifold edges (>2)   {topo['non_manifold_edges']}")
    print(f"    Euler V-E+F (all verts)   {topo['euler_all']}")
    print(f"    Euler V-E+F (referenced)  {topo['euler_referenced']}")
    closed = topo["boundary_edges"] == 0
    print(f"    engine will report        "
          f"{'CLOSED (watertight)' if closed else 'OPEN patch'}")

    print("  -- geometry (mesh's own length unit)")
    print(f"    volume                    {geom['volume']:.12g}")
    print(f"    surface area              {geom['surface_area']:.12g}")
    print(f"    R_vol = (3V/4pi)^(1/3)    {geom['r_vol']:.9g}")
    print(f"    centroid                  "
          f"[{geom['centroid'][0]:.6e}, {geom['centroid'][1]:.6e}, "
          f"{geom['centroid'][2]:.6e}]")
    print(f"    |centroid| / R_vol        "
          f"{np.linalg.norm(geom['centroid']) / geom['r_vol']:.3e}")
    print(f"    extent (x, y, z)          "
          f"[{geom['extent'][0]:.6f}, {geom['extent'][1]:.6f}, "
          f"{geom['extent'][2]:.6f}]")
    print(f"    extent on principal axes  "
          f"[{geom['principal_extent'][0]:.6f}, "
          f"{geom['principal_extent'][1]:.6f}, "
          f"{geom['principal_extent'][2]:.6f}]")
    print(f"    facet area mean/min/max   {geom['facet_area_mean']:.6g} / "
          f"{geom['facet_area_min']:.6g} / {geom['facet_area_max']:.6g}")
    print(f"    edge length mean          {geom['edge_length_mean']:.6g}")
    print(f"    inward-wound facets       {geom['inward_facets']}")
    print(f"    degenerate facets         {geom['degenerate_facets']}")

    print("  -- principal axes (uniform density, eigenvectors of I)")
    for i, (value, axis) in enumerate(zip(geom["inertia_eigenvalues"],
                                          geom["principal_axes"])):
        # Sign of an eigenvector is arbitrary; report the axis it lies nearest
        # and how far off, which is the question the pole convention asks.
        nearest = int(np.argmax(np.abs(axis)))
        tilt = math.degrees(math.acos(min(1.0, abs(axis[nearest]))))
        print(f"    I{i} = {value:.9g}  axis "
              f"[{axis[0]:+.6f}, {axis[1]:+.6f}, {axis[2]:+.6f}]"
              f"  ({tilt:.3f} deg off {'xyz'[nearest]})")

    print("  -- units")
    print(f"    median |face center|      "
          f"{geom['face_center_radius_median']:.9g}")
    print(f"    engine will infer         {verdict}")
    if note:
        print(f"    {note}")

    return {"path": path, "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]), "md5": md5, "sha256": sha256,
            "size": size, "closed": closed, **topo, **geom}


def compare(new, old):
    print(f"\n=== comparison: {old['path']}  ->  {new['path']}")
    rows = [
        ("facets", old["faces"], new["faces"], "{:d}"),
        ("vertices", old["vertices"], new["vertices"], "{:d}"),
        ("volume", old["volume"], new["volume"], "{:.9g}"),
        ("R_vol", old["r_vol"], new["r_vol"], "{:.9g}"),
        ("surface area", old["surface_area"], new["surface_area"], "{:.9g}"),
        ("edge length mean", old["edge_length_mean"],
         new["edge_length_mean"], "{:.6g}"),
        ("extent x", old["extent"][0], new["extent"][0], "{:.6f}"),
        ("extent y", old["extent"][1], new["extent"][1], "{:.6f}"),
        ("extent z", old["extent"][2], new["extent"][2], "{:.6f}"),
        ("boundary edges", old["boundary_edges"],
         new["boundary_edges"], "{:d}"),
    ]
    print(f"    {'quantity':20s} {'old':>18s} {'new':>18s} {'change':>12s}")
    for label, before, after, fmt in rows:
        if isinstance(before, (int, np.integer)) and fmt == "{:d}":
            change = f"{after - before:+d}"
        else:
            change = (f"{100.0 * (after - before) / before:+.3f}%"
                      if before else "n/a")
        print(f"    {label:20s} {fmt.format(before):>18s} "
              f"{fmt.format(after):>18s} {change:>12s}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Closed-form validation of a triangulated shape model.")
    parser.add_argument("mesh", nargs="+", help="OBJ file(s) to validate")
    parser.add_argument("--compare", metavar="OBJ", default=None,
                        help="an earlier model to tabulate the change against")
    parser.add_argument("--body-radius-km", type=float, default=0.0,
                        help="the run's generic_body radius, so the engine's "
                             "unit inference can be predicted the way the "
                             "engine makes it")
    parser.add_argument("--header", action="store_true",
                        help="echo the OBJ's own leading comment block")
    parser.add_argument("--expect-radius-km", type=float, default=None,
                        help="published volume-equivalent radius to check "
                             "R_vol against (mesh assumed to be in km)")
    parser.add_argument("--expect-closed", action="store_true",
                        help="fail unless the engine's rule reports CLOSED")
    parser.add_argument("--expect-facets", type=int, default=None,
                        help="fail unless the facet count matches exactly")
    parser.add_argument("--expect-md5", default=None,
                        help="fail unless the file md5 matches")
    parser.add_argument("--tolerance-percent", type=float, default=1.0,
                        help="tolerance for --expect-radius-km "
                             "(default: 1.0 percent)")
    args = parser.parse_args(argv)

    results = [describe(path, args.body_radius_km, args.header)
               for path in args.mesh]

    if args.compare:
        baseline = describe(args.compare, args.body_radius_km, args.header)
        compare(results[0], baseline)

    failures = []
    primary = results[0]
    if args.expect_closed and not primary["closed"]:
        failures.append(
            f"{primary['path']}: the engine's rule reports OPEN "
            f"({primary['boundary_edges']} boundary edges); --expect-closed "
            f"was given.")
    if args.expect_facets is not None and primary["faces"] != args.expect_facets:
        failures.append(
            f"{primary['path']}: {primary['faces']} facets, expected "
            f"{args.expect_facets}.")
    if args.expect_md5 and primary["md5"] != args.expect_md5:
        failures.append(
            f"{primary['path']}: md5 {primary['md5']}, expected "
            f"{args.expect_md5}.")
    if args.expect_radius_km is not None:
        error = 100.0 * (primary["r_vol"] - args.expect_radius_km) / \
            args.expect_radius_km
        verdict = "ok" if abs(error) <= args.tolerance_percent else "FAIL"
        print(f"\n  R_vol {primary['r_vol']:.6f} km vs published "
              f"{args.expect_radius_km:.6f} km: {error:+.3f}% "
              f"(tolerance {args.tolerance_percent:g}%) -> {verdict}")
        if verdict == "FAIL":
            failures.append(
                f"{primary['path']}: R_vol is {error:+.3f}% from the published "
                f"{args.expect_radius_km} km, outside "
                f"{args.tolerance_percent:g}%.")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nvalidate_shape_model: all requested checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
