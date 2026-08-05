#!/usr/bin/env python3
"""Build a LOLA occluder-ring PLY WITHOUT dtm_to_ply_delaunay (which is gone
from the evicted dtm/ tree). Replicates a3gt_v2_ring_mesh.py's output:
a coarse LOLA annulus [R_core, R_ext] around the site, Delaunay-triangulated
in the south-polar-stereo plane, written in the exact dtm PLY schema
(binary_le; dbl x,y,z / flt nx,ny,nz / dbl lat,lon,elev; faces uchar+int*3),
in the selenocentric frame XYZ = (R+elev)*spherical(lat,lon) — verified to
0.000 m against MP026's dtm-built ring.

  python3 build_lola_ring.py --lola <LOLA.tif> --site-lat -84.223 \
      --site-lon 33.502 --r-core-km 3 --r-ext-km 81.6712 \
      --max-points 80000 --out ring.ply
"""
from __future__ import annotations
import argparse
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

R = 1737400.0  # Moon 2015 sphere radius (m), matches the LOLA base CRS

def geodesic_m(slat, slon, lat, lon):
    """Great-circle distance (m) on the R sphere; all args radians."""
    ca = np.sin(slat)*np.sin(lat) + np.cos(slat)*np.cos(lat)*np.cos(lon-slon)
    return R * np.arccos(np.clip(ca, -1.0, 1.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lola", required=True)
    ap.add_argument("--site-lat", type=float, required=True)  # deg, south negative
    ap.add_argument("--site-lon", type=float, required=True)  # deg east
    ap.add_argument("--r-core-km", type=float, default=3.0)
    ap.add_argument("--r-ext-km", type=float, required=True)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--elev-scale", type=float, default=None,
                    help="Meters per stored DEM unit. Default: read band.GetScale(). "
                         "Raw LOLA polar .IMG store scaled integers (SCALING_FACTOR=0.5) that "
                         "GDAL returns UNSCALED -- pass 0.5 for those. "
                         "See docs/ptm21-ring-scale-fix.md (PTM-21).")
    ap.add_argument("--elev-offset", type=float, default=None,
                    help="Additive meters. Default: read band.GetOffset() (or 0).")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    slat, slon = np.radians(a.site_lat), np.radians(a.site_lon)
    # site projected coords (south-polar stereo, lat_0=-90 lon_0=0):
    #   rho = 2R tan(colat/2), x = rho sin(lon), y = rho cos(lon)
    colat = np.radians(90.0 - abs(a.site_lat))
    rho = 2.0*R*np.tan(colat/2.0)
    x0, y0 = rho*np.sin(slon), rho*np.cos(slon)

    ds = gdal.Open(a.lola)
    gt = ds.GetGeoTransform()      # (ulx, dx, 0, uly, 0, dy<0)
    ulx, dx, _, uly, _, dyg = gt
    dy = abs(dyg)
    band = ds.GetRasterBand(1)

    # read a window of half-size R_ext*1.05 around the site
    half = int(a.r_ext_km*1.05*1000.0/dx) + 2
    col0 = max(0, int((x0-ulx)/dx) - half)
    row0 = max(0, int((uly-y0)/dy) - half)
    n = 2*half
    n = min(n, ds.RasterXSize-col0, ds.RasterYSize-row0)
    arr = band.ReadAsArray(col0, row0, n, n).astype(np.float64)

    # Apply the DEM's elevation scale/offset. LOLA polar PDS .IMG store elevation
    # as scaled integers (SCALING_FACTOR, e.g. 0.5 m/DN); GDAL's ReadAsArray
    # returns raw DN UNSCALED. Using raw DN as meters silently DOUBLED a ring
    # once (PTM-21, sunlit_ring_50_200). The projected Float32 .tif stores
    # physical meters directly (no scale metadata) and stays byte-identical.
    _int_types = (gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_UInt16,
                  gdal.GDT_Int32, gdal.GDT_UInt32)
    scale = a.elev_scale if a.elev_scale is not None else band.GetScale()
    offset = a.elev_offset if a.elev_offset is not None else band.GetOffset()
    if scale is None:
        if band.DataType in _int_types:
            raise SystemExit(
                f"[build_lola_ring] REFUSING to proceed: {a.lola} is an integer raster "
                f"({gdal.GetDataTypeName(band.DataType)}) with NO elevation scale metadata "
                f"(band.GetScale() is None) and no --elev-scale given. Raw integer DN are "
                f"almost certainly NOT meters (LOLA polar .IMG use 0.5) -- refusing to assume "
                f"1.0 and silently mis-scale the ring. Pass --elev-scale explicitly.")
        scale = 1.0  # float raster, no scale metadata => physical meters (the .tif path)
    if offset is None:
        offset = 0.0
    if scale != 1.0 or offset != 0.0:
        print(f"[build_lola_ring] applying DEM elevation scale={scale} offset={offset} "
              f"(source {a.lola})")
    arr = arr * scale + offset

    cols = np.arange(col0, col0+n)
    rows = np.arange(row0, row0+n)
    PX = ulx + (cols+0.5)*dx
    PY = uly - (rows+0.5)*dy
    PXg, PYg = np.meshgrid(PX, PY)
    px = PXg.ravel(); py = PYg.ravel(); el = arr.ravel()
    good = el > -1e30
    px, py, el = px[good], py[good], el[good]

    # pixel -> lat/lon (inverse polar stereo)
    rho_p = np.sqrt(px*px + py*py)
    colat_p = 2.0*np.arctan2(rho_p, 2.0*R)
    lat = -(90.0 - np.degrees(colat_p))            # south negative, deg
    lon = np.degrees(np.arctan2(px, py))           # east, deg
    d = geodesic_m(slat, slon, np.radians(lat), np.radians(lon))

    disk = d <= a.r_ext_km*1000.0                  # mesh the full disk, drop core later
    px, py, el, lat, lon = px[disk], py[disk], el[disk], lat[disk], lon[disk]
    d = d[disk]

    # subsample to the point budget. Weight prob ~ 1/dist so the near-field
    # (which sets the shadowing horizon) stays dense, ~matching dtm's
    # density-exp=1.0 linear falloff. (Uniform-area sampling under-resolves
    # the near horizon and lets grazing sun leak -> validated-failing.)
    if px.size > a.max_points:
        w = 1.0/np.maximum(d, 1000.0); w /= w.sum()
        sel = rng.choice(px.size, a.max_points, replace=False, p=w)
        px, py, el, lat, lon, d = px[sel], py[sel], el[sel], lat[sel], lon[sel], d[sel]

    # selenocentric XYZ = (R+elev)*spherical(lat,lon)  [verified vs dtm: 0.000 m]
    la, lo = np.radians(lat), np.radians(lon)
    rr = R + el
    X = rr*np.cos(la)*np.cos(lo); Y = rr*np.cos(la)*np.sin(lo); Z = rr*np.sin(la)

    # Delaunay in the projection plane (same as dtm tool)
    from scipy.spatial import Delaunay
    tri = Delaunay(np.column_stack([px, py]))
    faces = tri.simplices.astype(np.int64)

    # drop triangles whose centroid is inside R_core (geodesic) -> annulus
    clat = la[faces].mean(axis=1); clon = lo[faces].mean(axis=1)
    cd = geodesic_m(slat, slon, clat, clon)
    faces = faces[cd >= a.r_core_km*1000.0]

    # renumber used vertices
    used = np.unique(faces)
    remap = np.full(px.size, -1, dtype=np.int64); remap[used] = np.arange(used.size)
    faces = remap[faces]
    X, Y, Z, lat, lon, el = X[used], Y[used], Z[used], lat[used], lon[used], el[used]

    # outward radial normals (irrelevant to shadow BVH, but fill the schema)
    nrm = np.column_stack([X, Y, Z]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    nv, nf = X.size, faces.shape[0]
    vdt = np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('nx','<f4'),('ny','<f4'),
                    ('nz','<f4'),('lat','<f8'),('lon','<f8'),('el','<f8')])
    varr = np.empty(nv, dtype=vdt)
    varr['x'],varr['y'],varr['z'] = X,Y,Z
    varr['nx'],varr['ny'],varr['nz'] = nrm[:,0],nrm[:,1],nrm[:,2]
    varr['lat'],varr['lon'],varr['el'] = lat,lon,el
    frows = np.zeros((nf,13), dtype=np.uint8)
    frows[:,0] = 3
    frows[:,1:13] = faces.astype('<i4').view(np.uint8).reshape(nf,12)

    hdr = (b"ply\nformat binary_little_endian 1.0\n"
           b"comment DTM to PLY with Delaunay triangulation for PTM3D\n"
           b"element vertex %d\n"
           b"property double x\nproperty double y\nproperty double z\n"
           b"property float nx\nproperty float ny\nproperty float nz\n"
           b"property double latitude\nproperty double longitude\nproperty double elevation\n"
           b"element face %d\nproperty list uchar int vertex_indices\nend_header\n"
           % (nv, nf))
    with open(a.out, "wb") as f:
        f.write(hdr); varr.tofile(f); frows.tofile(f)
    print(f"wrote {a.out}: {nv} verts, {nf} faces, "
          f"elev[{el.min():.0f},{el.max():.0f}] dist_km[{(geodesic_m(slat,slon,la[used] if False else np.radians(lat),np.radians(lon)).min())/1e3:.2f},"
          f"{geodesic_m(slat,slon,np.radians(lat),np.radians(lon)).max()/1e3:.2f}]")

if __name__ == "__main__":
    main()
