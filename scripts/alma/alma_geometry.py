#!/usr/bin/env python3
"""
Project ALMA image pixels onto Europa's surface (lat/lon coordinates).

Given a brightness temperature FITS image and the observation geometry
(sub-observer point, Europa angular diameter), computes the body-fixed
latitude and longitude for each pixel.

Usage:
  python alma_geometry.py tb_image.fits --sub-lon 45 --sub-lat -1.5 --ang-diam 0.77
  python alma_geometry.py tb_image.fits --spice --epoch "2015-11-17T14:43:00"
"""

import argparse
import numpy as np
from pathlib import Path

# Europa parameters
EUROPA_RADIUS_KM = 1560.8  # Mean radius [km]


def compute_disk_geometry(nx, ny, pixel_scale_arcsec, ang_diam_arcsec,
                          sub_obs_lon_deg, sub_obs_lat_deg,
                          center_x=None, center_y=None):
    """Compute Europa lat/lon for each pixel in an ALMA image.

    Uses orthographic projection from the observer's viewpoint.

    Parameters
    ----------
    nx, ny : int
        Image dimensions (pixels).
    pixel_scale_arcsec : float
        Pixel scale in arcseconds/pixel.
    ang_diam_arcsec : float
        Europa's angular diameter in arcseconds.
    sub_obs_lon_deg : float
        Sub-observer longitude in degrees West.
    sub_obs_lat_deg : float
        Sub-observer latitude in degrees.
    center_x, center_y : float, optional
        Pixel coordinates of Europa's disk center. Default: image center.

    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude arrays (degrees). NaN outside the disk.
    emission_angle : ndarray
        Emission angle at each pixel (degrees). NaN outside disk.
    on_disk : ndarray
        Boolean mask, True for pixels on Europa's disk.
    """
    if center_x is None:
        center_x = nx / 2.0
    if center_y is None:
        center_y = ny / 2.0

    radius_arcsec = ang_diam_arcsec / 2.0
    radius_pix = radius_arcsec / pixel_scale_arcsec

    # Pixel grid (x = RA direction, y = Dec direction)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = (xx - center_x)  # offset from center in pixels
    dy = (yy - center_y)

    # Normalized coordinates on the unit disk
    xn = dx / radius_pix  # ranges [-1, 1] across the disk
    yn = dy / radius_pix

    r2 = xn**2 + yn**2
    on_disk = r2 <= 1.0

    # Initialize output
    lat = np.full((ny, nx), np.nan)
    lon = np.full((ny, nx), np.nan)
    emission_angle = np.full((ny, nx), np.nan)

    # For points on the disk, compute body-fixed coordinates
    # In the orthographic projection, (xn, yn) correspond to the
    # apparent position on the disk as seen from the observer.
    # z_n = sqrt(1 - xn² - yn²) is the line-of-sight component.

    mask = on_disk
    xd = xn[mask]
    yd = yn[mask]
    zd = np.sqrt(1.0 - xd**2 - yd**2)

    # Emission angle = angle between surface normal and line of sight
    emission_angle[mask] = np.degrees(np.arccos(zd))

    # Convert from observer-frame (xd, yd, zd) to body-fixed (lat, lon)
    # Observer frame: z = toward observer, y = north, x = west on sky
    # (Note: RA increases east, so x in pixel coords may be flipped —
    #  we handle this via the sign convention for longitude)

    # Rotate from observer frame to body-fixed:
    # 1. Rotate by sub-observer latitude about x-axis
    # 2. Rotate by sub-observer longitude about z-axis

    sub_lat_rad = np.radians(sub_obs_lat_deg)
    sub_lon_rad = np.radians(sub_obs_lon_deg)

    cos_lat0 = np.cos(sub_lat_rad)
    sin_lat0 = np.sin(sub_lat_rad)

    # Rotation from observer frame to planetocentric:
    # First rotate by -sub_lat about x (tilt the pole)
    x1 = xd
    y1 = yd * cos_lat0 - zd * sin_lat0
    z1 = yd * sin_lat0 + zd * cos_lat0

    # Now (x1, y1, z1) are in a frame where z = planet rotation axis
    # and the sub-observer point is at lon = sub_obs_lon
    body_lat = np.degrees(np.arcsin(np.clip(z1, -1, 1)))

    # Longitude: atan2(x1, y1) gives angle from the sub-observer meridian
    # x is positive to the west (Europa convention: west longitude)
    delta_lon = np.degrees(np.arctan2(x1, y1))
    body_lon = (sub_obs_lon_deg + delta_lon) % 360.0

    lat[mask] = body_lat
    lon[mask] = body_lon

    return lat, lon, emission_angle, on_disk


def extract_tb_on_disk(tb_data, lat, lon, emission_angle, on_disk,
                       max_emission_angle=80.0):
    """Extract brightness temperature values on Europa's disk.

    Parameters
    ----------
    tb_data : ndarray
        Brightness temperature image (K).
    lat, lon, emission_angle : ndarray
        Geometry arrays from compute_disk_geometry.
    on_disk : ndarray
        Boolean mask for on-disk pixels.
    max_emission_angle : float
        Maximum emission angle to include (degrees). Limb pixels are noisy.

    Returns
    -------
    dict
        Dictionary with arrays: lat, lon, tb, emission_angle, pixel_x, pixel_y
    """
    valid = on_disk & np.isfinite(tb_data) & (emission_angle < max_emission_angle)

    yy, xx = np.mgrid[0:tb_data.shape[0], 0:tb_data.shape[1]]

    return {
        "lat": lat[valid],
        "lon": lon[valid],
        "tb": tb_data[valid],
        "emission_angle": emission_angle[valid],
        "pixel_x": xx[valid],
        "pixel_y": yy[valid],
        "n_pixels": int(np.sum(valid)),
    }


def get_geometry_from_spice(epoch_str, observer="EARTH"):
    """Compute Europa observation geometry using SPICE.

    Parameters
    ----------
    epoch_str : str
        UTC epoch string, e.g., "2015-11-17T14:43:00"
    observer : str
        Observer body name (default: EARTH for ground-based).

    Returns
    -------
    dict
        sub_obs_lon, sub_obs_lat, ang_diam_arcsec, distance_km
    """
    import spiceypy as spice

    # Load kernels (user must have SPICE kernels configured)
    # Typical kernels needed: naif0012.tls, pck00011.tpc, jup365.bsp,
    # de440.bsp or similar
    et = spice.str2et(epoch_str)

    # Sub-observer point on Europa
    spoint, trgepc, srfvec = spice.subpnt(
        "NEAR POINT/ELLIPSOID", "EUROPA", et, "IAU_EUROPA", "LT+S", observer
    )

    # Convert to lat/lon
    r, lon_rad, lat_rad = spice.reclat(spoint)
    sub_obs_lon = np.degrees(lon_rad) % 360.0  # Convert to 0-360 West
    sub_obs_lat = np.degrees(lat_rad)

    # Angular diameter
    dist = spice.vnorm(srfvec)
    ang_radius_rad = np.arctan(EUROPA_RADIUS_KM / dist)
    ang_diam_arcsec = 2.0 * np.degrees(ang_radius_rad) * 3600.0

    return {
        "sub_obs_lon": sub_obs_lon,
        "sub_obs_lat": sub_obs_lat,
        "ang_diam_arcsec": ang_diam_arcsec,
        "distance_km": dist,
        "epoch": epoch_str,
    }


def process_fits(fits_path, sub_lon, sub_lat, ang_diam, output_prefix=None,
                 max_emission_angle=80.0):
    """Process a single FITS file: project to lat/lon and extract Tb.

    Parameters
    ----------
    fits_path : str or Path
        Path to FITS file (should be in Tb or Jy/beam).
    sub_lon, sub_lat : float
        Sub-observer longitude (°W) and latitude (°).
    ang_diam : float
        Europa angular diameter in arcseconds.
    output_prefix : str, optional
        Prefix for output files.
    max_emission_angle : float
        Maximum emission angle for extraction.

    Returns
    -------
    dict
        Extracted data and geometry information.
    """
    from astropy.io import fits as pyfits

    fits_path = Path(fits_path)
    if output_prefix is None:
        output_prefix = fits_path.stem

    with pyfits.open(str(fits_path)) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.copy()

        # Squeeze to 2D
        while data.ndim > 2:
            data = data[0]

        ny, nx = data.shape

        # Pixel scale from header
        cdelt1 = abs(hdr.get("CDELT1", 0))  # degrees/pixel
        cdelt2 = abs(hdr.get("CDELT2", 0))
        pixel_scale = (cdelt1 + cdelt2) / 2.0 * 3600.0  # arcsec/pixel

        print(f"\nProcessing: {fits_path.name}")
        print(f"  Image: {nx} x {ny} pixels, {pixel_scale:.4f} arcsec/pixel")
        print(f"  Sub-observer: lon={sub_lon:.1f}°W, lat={sub_lat:.1f}°")
        print(f"  Angular diameter: {ang_diam:.3f} arcsec")

    # Compute geometry
    lat, lon, emang, on_disk = compute_disk_geometry(
        nx, ny, pixel_scale, ang_diam, sub_lon, sub_lat
    )

    n_disk = np.sum(on_disk)
    print(f"  Pixels on disk: {n_disk} / {nx*ny}")

    # Extract Tb on disk
    extracted = extract_tb_on_disk(data, lat, lon, emang, on_disk,
                                  max_emission_angle)

    print(f"  Valid pixels (emission < {max_emission_angle}°): {extracted['n_pixels']}")
    if extracted["n_pixels"] > 0:
        print(f"  Tb range: {np.nanmin(extracted['tb']):.1f} - "
              f"{np.nanmax(extracted['tb']):.1f} K")
        print(f"  Tb mean: {np.nanmean(extracted['tb']):.1f} K")

    # Save geometry arrays as NPZ
    output_dir = fits_path.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{output_prefix}_geometry.npz"
    np.savez_compressed(
        str(npz_path),
        lat=lat, lon=lon, emission_angle=emang, on_disk=on_disk,
        tb=data,
        sub_obs_lon=sub_lon, sub_obs_lat=sub_lat,
        ang_diam=ang_diam, pixel_scale=pixel_scale,
        **{k: v for k, v in extracted.items() if isinstance(v, np.ndarray)},
    )
    print(f"  Saved geometry: {npz_path}")

    return {
        "geometry": {"lat": lat, "lon": lon, "emission_angle": emang, "on_disk": on_disk},
        "extracted": extracted,
        "header_info": {
            "pixel_scale": pixel_scale,
            "image_size": (nx, ny),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Project ALMA image pixels onto Europa lat/lon"
    )
    parser.add_argument("input", help="Input FITS file (brightness temperature)")
    parser.add_argument("--sub-lon", type=float, required=True,
                        help="Sub-observer longitude (degrees West)")
    parser.add_argument("--sub-lat", type=float, default=-1.5,
                        help="Sub-observer latitude (degrees, default: -1.5)")
    parser.add_argument("--ang-diam", type=float, default=0.77,
                        help="Europa angular diameter (arcsec, default: 0.77)")
    parser.add_argument("--max-emission", type=float, default=80.0,
                        help="Max emission angle for extraction (degrees)")
    parser.add_argument("-o", "--output-prefix", help="Output file prefix")
    args = parser.parse_args()

    process_fits(
        args.input,
        sub_lon=args.sub_lon,
        sub_lat=args.sub_lat,
        ang_diam=args.ang_diam,
        output_prefix=args.output_prefix,
        max_emission_angle=args.max_emission,
    )


if __name__ == "__main__":
    main()
