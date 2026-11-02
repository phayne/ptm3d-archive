#!/usr/bin/env python3
"""Regression test for scripts/build_lola_ring.py DEM-window clipping.

The ring builder reads a DEM window of half-size ~R_ext around the site. It used
to read a SQUARE window whose side was clipped by min(x-room, y-room), so a site
near ONE raster edge silently lost ring coverage on the OTHER axis too (found on
Amundsen, 84.5 S, with LDEM_80S_20M: the DEM edge 137 km away along x also cut
the tangential y side to 137 km of the requested 200).

This test runs the real main() on a synthetic flat polar-stereographic DEM (an
in-memory stand-in for osgeo.gdal, so no GDAL install is needed) with the site
50 km from the +x raster edge and R_ext = 100 km. Along x the ring must stop at
the raster edge; along +/-y (fully inside the raster) it must reach ~R_ext.

Pure-python, no GPU, no sim binary -- runnable anytime:
    python3 scripts/test_build_lola_ring_window.py
"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = 1737400.0

# Synthetic DEM: 1200 x 1200 px at 500 m, centered on the pole (+/-300 km), Float32
# meters, elevation 0 everywhere.
DX = 500.0
NPX = 1200
ULX, ULY = -NPX * DX / 2, NPX * DX / 2
SITE_X, SITE_Y = 250000.0, 0.0          # 50 km from the +x edge, centered in y
R_EXT_KM = 100.0


def _fake_gdal():
    g = types.ModuleType("osgeo.gdal")
    g.GDT_Byte, g.GDT_UInt16, g.GDT_Int16, g.GDT_UInt32, g.GDT_Int32, g.GDT_Float32 = 1, 2, 3, 4, 5, 6
    g.UseExceptions = lambda: None
    g.GetDataTypeName = lambda t: str(t)

    class Band:
        DataType = g.GDT_Float32
        def GetScale(self): return None
        def GetOffset(self): return None
        def ReadAsArray(self, xoff, yoff, xsize, ysize):
            assert xoff >= 0 and yoff >= 0 and xoff + xsize <= NPX and yoff + ysize <= NPX, \
                f"window out of raster: {xoff},{yoff},{xsize},{ysize}"
            return np.zeros((ysize, xsize), np.float32)

    class DS:
        RasterXSize = NPX
        RasterYSize = NPX
        def GetGeoTransform(self): return (ULX, DX, 0.0, ULY, 0.0, -DX)
        def GetRasterBand(self, i): return Band()

    g.Open = lambda path: DS()
    pkg = types.ModuleType("osgeo")
    pkg.gdal = g
    return pkg, g


def _read_ply_latlon(path):
    b = Path(path).read_bytes()
    h = b.index(b"end_header\n") + 11
    nv = int(b[:h].decode().split("element vertex ")[1].split()[0])
    vd = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f4'), ('ny', '<f4'),
                   ('nz', '<f4'), ('lat', '<f8'), ('lon', '<f8'), ('el', '<f8')])
    v = np.frombuffer(b, vd, nv, h)
    return v['lat'], v['lon']


def main():
    pkg, g = _fake_gdal()
    sys.modules["osgeo"] = pkg
    sys.modules["osgeo.gdal"] = g
    sys.path.insert(0, str(HERE))
    import build_lola_ring as blr

    rho = np.hypot(SITE_X, SITE_Y)
    site_lat = -(90.0 - np.degrees(2.0 * np.arctan(rho / (2.0 * R))))
    site_lon = np.degrees(np.arctan2(SITE_X, SITE_Y))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ring.ply"
        sys.argv = ["build_lola_ring.py", "--lola", "synthetic.tif",
                    "--site-lat", repr(site_lat), "--site-lon", repr(site_lon),
                    "--r-core-km", "10", "--r-ext-km", str(R_EXT_KM),
                    "--max-points", "30000", "--allow-unregistered", "--out", str(out)]
        blr.main()
        lat, lon = _read_ply_latlon(out)

    # back to projected coordinates relative to the site
    colat = np.radians(90.0 + lat)
    rho_v = 2.0 * R * np.tan(colat / 2.0)
    px = rho_v * np.sin(np.radians(lon)) - SITE_X
    py = rho_v * np.cos(np.radians(lon)) - SITE_Y
    band = np.abs(px) < 10e3                    # a 20 km strip along the y axis through the site
    reach_py = py[band & (py > 0)].max() / 1e3
    reach_my = -py[band & (py < 0)].min() / 1e3
    reach_px = px.max() / 1e3
    print(f"ring reach from site: +y {reach_py:.1f} km, -y {reach_my:.1f} km, +x {reach_px:.1f} km "
          f"(R_ext {R_EXT_KM:.0f} km; +x raster edge at {(ULX + NPX*DX - SITE_X)/1e3:.0f} km)")
    fails = 0
    for name, val in (("+y", reach_py), ("-y", reach_my)):
        if val >= 0.95 * R_EXT_KM:
            print(f"ok:   {name} coverage {val:.1f} km >= {0.95*R_EXT_KM:.0f} km")
        else:
            print(f"FAIL: {name} coverage {val:.1f} km < {0.95*R_EXT_KM:.0f} km "
                  f"(the other axis's raster edge clipped this one)")
            fails += 1
    if reach_px <= 50.0 + 1.0:
        print(f"ok:   +x coverage {reach_px:.1f} km stops at the raster edge")
    else:
        print(f"FAIL: +x coverage {reach_px:.1f} km extends past the raster edge"); fails += 1
    print("PASS" if fails == 0 else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
