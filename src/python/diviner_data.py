#!/usr/bin/env python3
"""Loader/helper for the Diviner south-polar maximum bolometric brightness
temperature grid (Tbols_SP_*.mat) used for PTM3D validation.

The .mat stores T_max_bol, T_avg_bol, x, y each as a flattened 7680x7680
(58,982,400,) column. PROJECTION (per the data provider): a simple polar
azimuthal-equidistant grid in DEGREES of colatitude --

    r   = 90 - |lat|         (colatitude, deg; 0 at the pole, 30 at 60 S)
    x   = r * sin(lon)        (lon = east longitude, deg)
    y   = r * cos(lon)

so x,y are in DEGREES (range +/-30 = the whole 60-90 S cap), NOT km. The grid
is 60 deg across / 7680 = 0.0078 deg/px ~= 237 m/px (~ Diviner footprint).
Reshape order 'F' yields T[row=y_index, col=x_index]. ~21% NaN (off-disk).

NOTE: x = r sin(lon), y = r cos(lon) measures lon from the +y axis, so
lon = atan2(x, y) (NOT atan2(y, x)).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

MOON_R = 1737400.0
KM_PER_DEG = np.pi * MOON_R / 180.0 / 1000.0     # 30.32 km per degree of arc


def latlon_to_diviner_xy(lat, lon):
    """(lat[deg, negative south], lon[deg east]) -> Diviner (x, y) in DEGREES."""
    r = 90.0 + np.asarray(lat, float)            # = 90 - |lat|
    lo = np.radians(np.asarray(lon, float))
    return r * np.sin(lo), r * np.cos(lo)


def diviner_xy_to_latlon(x, y):
    """Diviner (x, y) [deg] -> (lat[deg, negative], lon[deg east])."""
    r = np.hypot(np.asarray(x, float), np.asarray(y, float))
    lat = r - 90.0
    lon = np.degrees(np.arctan2(np.asarray(x, float), np.asarray(y, float)))
    return lat, lon


@dataclass
class DivinerGrid:
    t_max: np.ndarray      # (ny, nx) K, NaN off-disk
    t_avg: np.ndarray      # (ny, nx) K
    x_deg: np.ndarray      # (nx,) column coordinate, degrees colatitude
    y_deg: np.ndarray      # (ny,) row coordinate, degrees
    res_deg: float         # pixel size, degrees

    @property
    def res_m(self):
        return self.res_deg * KM_PER_DEG * 1000.0

    def extent_km(self):
        return (self.x_deg.min() * KM_PER_DEG, self.x_deg.max() * KM_PER_DEG,
                self.y_deg.min() * KM_PER_DEG, self.y_deg.max() * KM_PER_DEG)

    def sample_latlon(self, lat, lon):
        """Nearest-pixel Tmax at (lat, lon) arrays. NaN if off-grid."""
        xd, yd = latlon_to_diviner_xy(lat, lon)
        dx = self.x_deg[1] - self.x_deg[0]
        dy = self.y_deg[1] - self.y_deg[0]
        ix = np.round((xd - self.x_deg[0]) / dx).astype(int)
        iy = np.round((yd - self.y_deg[0]) / dy).astype(int)
        ok = (ix >= 0) & (ix < len(self.x_deg)) & (iy >= 0) & (iy < len(self.y_deg))
        out = np.full(np.shape(xd), np.nan, float)
        out[ok] = self.t_max[iy[ok], ix[ok]]
        return out


def load_diviner(path: str | Path) -> DivinerGrid:
    m = sio.loadmat(str(path))
    x = np.asarray(m["x"]).ravel()
    y = np.asarray(m["y"]).ravel()
    n = int(round(len(x) ** 0.5))
    tmax = np.asarray(m["T_max_bol"]).reshape(n, n, order="F").astype(np.float32)
    tavg = np.asarray(m["T_avg_bol"]).reshape(n, n, order="F").astype(np.float32)
    X = x.reshape(n, n, order="F")
    Y = y.reshape(n, n, order="F")
    x_deg = X[0, :].astype(np.float64)
    y_deg = Y[:, 0].astype(np.float64)
    res_deg = float(np.median(np.diff(x_deg)))
    return DivinerGrid(t_max=tmax, t_avg=tavg, x_deg=x_deg, y_deg=y_deg, res_deg=res_deg)


if __name__ == "__main__":
    import sys
    g = load_diviner(sys.argv[1] if len(sys.argv) > 1 else
                     os.environ.get("PTM_DIVINER_MAT", ""))
    print(f"grid {g.t_max.shape}  res {g.res_deg:.5f} deg ({g.res_m:.1f} m)")
    print(f"x_deg {g.x_deg.min():.2f}..{g.x_deg.max():.2f}  -> lat at edge "
          f"{diviner_xy_to_latlon(30,0)[0]:.0f}")
    # quick check: Tmax at the pole and at a few known PSRs
    for nm, la, lo in [("pole", -89.99, 0), ("Shackleton ctr", -89.66, 129.2),
                       ("Haworth", -87.45, -2.1), ("Faustini", -87.3, 84.0)]:
        print(f"  Tmax at {nm} ({la},{lo}) = {g.sample_latlon(la, lo):.0f} K")
