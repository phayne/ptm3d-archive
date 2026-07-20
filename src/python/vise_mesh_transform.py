"""Shared mesh I/O + relocate/scale transform for the Lunar-VISE camera
DEM-resolution error study.

The source SfS DTM (`cp22_1m_SFS_2M.ply`) is a body-fixed lunar mesh in
*meters* sitting at lat ~ -84.5 deg (a south-polar crater). For the VISE
study we need it (a) scaled so the smallest facets are ~1 cm and (b)
relocated to the planned landing site so SPICE computes the correct Sun
geometry there.

Transform (rigid + uniform scale, topography preserved):
  1. Recenter on the source patch centroid C.
  2. Express each vertex in the source local ENU frame (E0, N0, U0=C^).
  3. Multiply the local (e, n, u) coordinates by `scale`.
  4. Re-embed in the target local ENU frame (E1, N1, U1) at the landing
     site on a sphere of radius `body_radius_m`, so local up -> radial and
     local north -> geographic north. Mean u ~ 0, so the patch sits at the
     reference radius.

Source normals are all zero in the input and the C++ loader recomputes
face normals from geometry, so we preserve zero normals (fast, faithful).
Only latitude/longitude/elevation are recomputed for file consistency.
"""

import sys
import numpy as np


def latlon_to_unit(lat_deg, lon_deg):
    """Body-fixed unit vector for a geographic lat/lon (deg)."""
    la = np.radians(lat_deg)
    lo = np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], dtype=np.float64)


def enu_frame(up):
    """Right-handed ENU basis (east, north, up) for a given radial `up`.

    east = z_hat x up, north = up x east. Falls back to a stable basis if
    `up` is (anti)parallel to z_hat (not the case for either the -84.5 deg
    source patch or the +36.5 deg target, but kept for safety).
    """
    up = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, up)
    n = np.linalg.norm(east)
    if n < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = east / n
    north = np.cross(up, east)
    north = north / np.linalg.norm(north)
    return east, north, up


def relocate_scale(V, target_lat, target_lon, scale, body_radius_m):
    """Apply the recenter -> scale -> re-embed transform to vertices V.

    V : (Nv, 3) float64 body-fixed meters.
    Returns (Vnew, lat_deg, lon_deg, elev_m).
    """
    C = V.mean(axis=0)
    E0, N0, U0 = enu_frame(C)

    d = V - C
    e = d @ E0
    n = d @ N0
    u = d @ U0

    e *= scale
    n *= scale
    u *= scale

    U1 = latlon_to_unit(target_lat, target_lon)
    E1, N1, U1 = enu_frame(U1)
    origin = body_radius_m * U1

    Vnew = (origin
            + np.outer(e, E1)
            + np.outer(n, N1)
            + np.outer(u, U1))

    r = np.linalg.norm(Vnew, axis=1)
    lat = np.degrees(np.arcsin(np.clip(Vnew[:, 2] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(Vnew[:, 1], Vnew[:, 0]))
    elev = r - body_radius_m
    return Vnew, lat, lon, elev


def source_local_coords(V):
    """Return (e, n, u, C, E0, N0, U0): source vertices expressed in the
    source patch local ENU frame (meters, no scaling). Same frame used by
    `relocate_scale`, so a regridded surface re-embeds consistently."""
    C = V.mean(axis=0)
    E0, N0, U0 = enu_frame(C)
    d = V - C
    return d @ E0, d @ N0, d @ U0, C, E0, N0, U0


def embed_local(e, n, u, target_lat, target_lon, scale, body_radius_m):
    """Embed source-local (e, n, u) meters into the target site frame with
    uniform `scale` (mirrors the second half of `relocate_scale`)."""
    U1 = latlon_to_unit(target_lat, target_lon)
    E1, N1, U1 = enu_frame(U1)
    origin = body_radius_m * U1
    es, ns, us = e * scale, n * scale, u * scale
    Vnew = (origin
            + np.outer(es, E1)
            + np.outer(ns, N1)
            + np.outer(us, U1))
    r = np.linalg.norm(Vnew, axis=1)
    lat = np.degrees(np.arcsin(np.clip(Vnew[:, 2] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(Vnew[:, 1], Vnew[:, 0]))
    elev = r - body_radius_m
    return Vnew, lat, lon, elev


def regrid_to_posting(e, n, u, posting_m):
    """Resample a scattered (e, n, u) point cloud onto a regular lattice of
    spacing `posting_m` (emulates a coarser orbital DEM). Each lattice node
    elevation is the mean of source points snapped to it.

    Returns (Vloc (Nn,3) e/n/u of valid nodes, faces (Nf,3) int32). Only
    quads whose four corner nodes all received >=1 source point are
    triangulated, so the footprint stays inside real coverage.
    """
    e0, n0 = e.min(), n.min()
    ie = np.round((e - e0) / posting_m).astype(np.int64)
    in_ = np.round((n - n0) / posting_m).astype(np.int64)
    nx = int(ie.max()) + 1
    ny = int(in_.max()) + 1

    flat = in_ * nx + ie
    usum = np.zeros(nx * ny, dtype=np.float64)
    cnt = np.zeros(nx * ny, dtype=np.int64)
    np.add.at(usum, flat, u)
    np.add.at(cnt, flat, 1)

    valid = cnt > 0
    umean = np.zeros(nx * ny, dtype=np.float64)
    umean[valid] = usum[valid] / cnt[valid]

    gx = (np.arange(nx) * posting_m + e0)
    gy = (np.arange(ny) * posting_m + n0)
    GX, GY = np.meshgrid(gx, gy)              # (ny, nx)
    Vloc = np.column_stack([GX.ravel(), GY.ravel(),
                            umean]).astype(np.float64)

    # Quad -> 2 triangles where all 4 corners are valid.
    vg = valid.reshape(ny, nx)
    j, i = np.where(vg[:-1, :-1] & vg[:-1, 1:] &
                    vg[1:, :-1] & vg[1:, 1:])
    bl = j * nx + i
    br = bl + 1
    tl = bl + nx
    tr = tl + 1
    faces = np.empty((bl.size * 2, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([bl, br, tr])
    faces[1::2] = np.column_stack([bl, tr, tl])

    # Drop unreferenced nodes and reindex.
    used = np.unique(faces)
    remap = np.full(Vloc.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    Vloc = Vloc[used]
    faces = remap[faces].astype(np.int32)
    return Vloc, faces


def vertex_cluster_decimate(e, n, u, faces, cell_m):
    """Grid vertex-clustering decimation in the source local ENU frame.

    Snap every vertex to a `cell_m` lattice; each occupied cell becomes one
    output vertex at the mean (e, n, u) of its members. Original faces are
    re-indexed onto cluster IDs; faces that collapse (non-distinct corners)
    are dropped, and duplicate faces removed. This preserves the full
    footprint and a valid subset of the original connectivity while growing
    the feature size to ~`cell_m` (a faithful coarser-DEM model).

    Returns (re, rn, ru, new_faces int32).
    """
    e0, n0 = e.min(), n.min()
    ce = np.floor((e - e0) / cell_m).astype(np.int64)
    cn = np.floor((n - n0) / cell_m).astype(np.int64)
    key = cn * (ce.max() + 1) + ce
    uniq, inv = np.unique(key, return_inverse=True)
    nc = uniq.size

    se = np.zeros(nc); sn = np.zeros(nc); su = np.zeros(nc)
    cnt = np.zeros(nc, dtype=np.int64)
    np.add.at(se, inv, e)
    np.add.at(sn, inv, n)
    np.add.at(su, inv, u)
    np.add.at(cnt, inv, 1)
    re, rn, ru = se / cnt, sn / cnt, su / cnt

    cf = inv[faces]                                  # faces -> cluster IDs
    ok = ((cf[:, 0] != cf[:, 1]) &
          (cf[:, 1] != cf[:, 2]) &
          (cf[:, 0] != cf[:, 2]))
    cf = cf[ok]

    # Deduplicate faces (sort each row so rotations/flips collapse).
    s = np.sort(cf, axis=1)
    _, keep = np.unique(s, axis=0, return_index=True)
    cf = cf[np.sort(keep)]

    used = np.unique(cf)
    remap = np.full(nc, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    new_faces = remap[cf].astype(np.int32)
    return re[used], rn[used], ru[used], new_faces


def read_ascii_ply(path):
    """Fast bulk read of the cp22-style ASCII PLY.

    Header is the project's fixed layout: 9 vertex props
    (x y z nx ny nz latitude longitude elevation) then triangle faces
    (`3 i j k`). Returns (V float64 (Nv,3), faces int32 (Nf,3)).
    """
    with open(path, "r") as f:
        nv = nf = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            tok = line.split()
            if len(tok) >= 3 and tok[0] == "element" and tok[1] == "vertex":
                nv = int(tok[2])
            elif len(tok) >= 3 and tok[0] == "element" and tok[1] == "face":
                nf = int(tok[2])
            elif tok and tok[0] == "end_header":
                break
        if nv is None or nf is None:
            raise ValueError("Missing vertex/face counts in PLY header")
        body = f.read()

    arr = np.fromstring(body, sep=" ", dtype=np.float64)
    nvert_tok = nv * 9
    nface_tok = nf * 4
    if arr.size < nvert_tok + nface_tok:
        raise ValueError(
            f"PLY body short: got {arr.size} numbers, "
            f"need {nvert_tok + nface_tok} ({nv} verts, {nf} faces)")

    verts = arr[:nvert_tok].reshape(nv, 9)
    V = np.ascontiguousarray(verts[:, 0:3])
    faces = arr[nvert_tok:nvert_tok + nface_tok].reshape(nf, 4)
    faces = np.ascontiguousarray(faces[:, 1:4].astype(np.int32))
    return V, faces


def write_binary_ply(path, V, lat, lon, elev, faces):
    """Write a binary_little_endian PLY in the project's vertex layout
    (x y z double, nx ny nz float [zeros], latitude longitude elevation
    double; faces = uchar list of int). The C++ generic loader keys on
    property names, so this round-trips and loads fast."""
    nv = V.shape[0]
    nf = faces.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Lunar-VISE relocated+scaled mesh\n"
        f"element vertex {nv}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property double latitude\n"
        "property double longitude\n"
        "property double elevation\n"
        f"element face {nf}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    vbuf = np.empty(nv, dtype=[
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("latitude", "<f8"), ("longitude", "<f8"), ("elevation", "<f8"),
    ])
    vbuf["x"] = V[:, 0]
    vbuf["y"] = V[:, 1]
    vbuf["z"] = V[:, 2]
    vbuf["nx"] = 0.0
    vbuf["ny"] = 0.0
    vbuf["nz"] = 0.0
    vbuf["latitude"] = lat
    vbuf["longitude"] = lon
    vbuf["elevation"] = elev

    fbuf = np.empty(nf, dtype=[("c", "u1"), ("v", "<i4", 3)])
    fbuf["c"] = 3
    fbuf["v"] = faces.astype("<i4")

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vbuf.tobytes())
        f.write(fbuf.tobytes())


def min_edge_sample(V, faces, n=400000):
    """Approx minimum triangle edge over a face sample (sanity check)."""
    step = max(1, faces.shape[0] // n)
    fs = faces[::step]
    p0 = V[fs[:, 0]]
    p1 = V[fs[:, 1]]
    p2 = V[fs[:, 2]]
    e = np.stack([
        np.linalg.norm(p0 - p1, axis=1),
        np.linalg.norm(p1 - p2, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
    ], axis=1).min(axis=1)
    return float(e.min()), float(np.median(e)), float(e.max())


def bbox_report(V):
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    return mn, mx, mx - mn


if __name__ == "__main__":
    print("This is a shared module; run vise_relocate_scale_mesh.py instead.",
          file=sys.stderr)
    sys.exit(1)
