#!/usr/bin/env python3
"""Step 3a: solve the UTC epoch for local solar time ~10:00 at the
Lunar-VISE site, in March 2028 (representative mid-morning).

Lunar local solar time:  LST = 12 + wrap180(site_lon - subsolar_lon)/15
(hours), where subsolar_lon is the Sun's sub-point longitude in IAU_MOON.
We scan March 2028, find the LST=10 crossing nearest mid-month, refine by
bisection, and report Sun elevation/azimuth at the site.

Writes tmp/vise/epoch.json.
"""

import argparse
import json

import numpy as np
import spiceypy as sp

MOON_TM = ("/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/"
           "europa/heat3d/ptm/data/kernels/moon.tm")


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def subsolar_lon_lat(et):
    pos, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    r, lon, lat = sp.reclat(pos)
    return np.degrees(lon), np.degrees(lat)


def lst_hours(et, site_lon):
    slon, _ = subsolar_lon_lat(et)
    return 12.0 + wrap180(site_lon - slon) / 15.0


def sun_elev_az(et, site_lat, site_lon):
    pos, _ = sp.spkpos("SUN", et, "IAU_MOON", "LT+S", "MOON")
    s = pos / np.linalg.norm(pos)
    la, lo = np.radians(site_lat), np.radians(site_lon)
    up = np.array([np.cos(la) * np.cos(lo),
                   np.cos(la) * np.sin(lo),
                   np.sin(la)])
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    elev = np.degrees(np.arcsin(np.clip(np.dot(s, up), -1, 1)))
    az = np.degrees(np.arctan2(np.dot(s, east), np.dot(s, north))) % 360.0
    return elev, az


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site-lat", type=float, default=36.5)
    ap.add_argument("--site-lon", type=float, default=-40.7)
    ap.add_argument("--target-lst", type=float, default=10.0)
    ap.add_argument("--month-start", default="2028-03-01T00:00:00")
    ap.add_argument("--month-end", default="2028-04-01T00:00:00")
    ap.add_argument("--anchor", default="2028-03-15T12:00:00",
                    help="Prefer the LST crossing nearest this date.")
    ap.add_argument("--out", default="tmp/vise/epoch.json")
    args = ap.parse_args()

    sp.furnsh(MOON_TM)
    et0 = sp.str2et(args.month_start)
    et1 = sp.str2et(args.month_end)
    eta = sp.str2et(args.anchor)

    # Coarse scan (2 h) for sign changes of g = LST - target (unwrapped via
    # nearest-to-target representative so a single mid-month root is found).
    ets = np.arange(et0, et1, 7200.0)
    g = np.array([lst_hours(e, args.site_lon) - args.target_lst
                  for e in ets])
    roots = []
    for i in range(len(ets) - 1):
        if g[i] == 0.0 or g[i] * g[i + 1] < 0.0:
            a, b = ets[i], ets[i + 1]
            ga = g[i]
            for _ in range(60):
                m = 0.5 * (a + b)
                gm = lst_hours(m, args.site_lon) - args.target_lst
                if ga * gm <= 0.0:
                    b = m
                else:
                    a, ga = m, gm
            roots.append(0.5 * (a + b))
    if not roots:
        raise SystemExit("No LST crossing found in window")

    et = min(roots, key=lambda e: abs(e - eta))
    utc = sp.et2utc(et, "ISOC", 3)
    lst = lst_hours(et, args.site_lon)
    elev, az = sun_elev_az(et, args.site_lat, args.site_lon)
    slon, slat = subsolar_lon_lat(et)

    rec = dict(utc=utc, et=float(et), local_solar_time_h=float(lst),
               sun_elevation_deg=float(elev), sun_azimuth_deg=float(az),
               subsolar_lon_deg=float(slon), subsolar_lat_deg=float(slat),
               site_lat=args.site_lat, site_lon=args.site_lon)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {args.out}")
    sp.kclear()


if __name__ == "__main__":
    main()
