#!/usr/bin/env python3
"""
Depth-weighted emission: a subsurface temperature profile → brightness temperature.

Turns a PTM3D subsurface temperature profile T(z) into the brightness
temperature a radiometer would measure, by integrating the plane-parallel
radiative transfer equation through the profile. Because the absorption
coefficient of the regolith is finite, the measurement is a weighted average
over depth rather than a reading of the surface temperature — which is what
makes it sensitive to subsurface structure, and therefore to thermal inertia.

The radiative transfer integral:
    Tb(ν) = ε(ν) ∫₀^∞ T(z) κ_a(ν, T(z)) exp(-∫₀^z κ_a(ν, T(z')) dz') dz

where κ_a is the power absorption coefficient of the porous ice regolith and
ε is the surface emissivity. Both come from ``dielectric_mixing.py``.

This module is instrument-independent: the observing frequency is an argument
in GHz, and no observatory, receiver band, or observing program is assumed.

Usage:
  python depth_weighted_emission.py profiles.npz --freq 233 --porosity 0.5
  python depth_weighted_emission.py --demo
"""

import argparse
import numpy as np
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dielectric_mixing import (
    absorption_coefficient, penetration_depth, porous_ice_permittivity,
    fresnel_emissivity, REFERENCE_FREQUENCIES_GHZ
)

# numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and REMOVED the old name, so
# neither spelling works on both sides of that release. Bind whichever this
# numpy has, and the module runs on everything from the declared floor
# (numpy>=1.20) upward. The two are the same routine: on identical inputs they
# return the same bits on numpy 1.24.4, 1.26.4, 2.2.6 and 2.4.3, for both the
# 1-D and the ``axis=`` call below.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:                  # numpy < 2.0
    _trapezoid = np.trapz


def radiative_transfer_1d(z, T_z, freq_ghz, porosity, emissivity="fresnel",
                          mixing="maxwell_garnett"):
    """Compute brightness temperature from a single T(z) profile.

    Evaluates the plane-parallel radiative transfer integral numerically:
        Tb = ε ∫ T(z) κ(z) exp(-τ(z)) dz

    where τ(z) = ∫₀^z κ(z') dz' is the optical depth.

    Parameters
    ----------
    z : ndarray
        Depth array in meters (monotonically increasing, starting near 0).
    T_z : ndarray
        Temperature at each depth point (K).
    freq_ghz : float
        Observing frequency in GHz.
    porosity : float or ndarray
        Porosity at each depth (scalar for uniform, array for depth-varying).
    emissivity : float or "fresnel"
        Surface emissivity.  "fresnel" (default) computes the nadir
        Fresnel emissivity from the dielectric model.
    mixing : str
        Dielectric mixing model.

    Returns
    -------
    float
        Brightness temperature Tb in K.
    """
    z = np.asarray(z, dtype=float)
    T_z = np.asarray(T_z, dtype=float)
    n = len(z)

    if np.isscalar(porosity):
        porosity = np.full(n, porosity)
    else:
        porosity = np.asarray(porosity, dtype=float)

    # Resolve emissivity
    if emissivity == "fresnel":
        T_surf = float(T_z[0]) if len(T_z) > 0 else 100.0
        eps = fresnel_emissivity(freq_ghz, T_surf, float(porosity[0]),
                                 0.0, mixing)
    else:
        eps = float(emissivity)

    # Compute absorption coefficient at each depth
    kappa = np.array([
        absorption_coefficient(freq_ghz, T_z[i], porosity[i], mixing)
        for i in range(n)
    ])

    # Compute optical depth τ(z) by cumulative trapezoidal integration
    tau = np.zeros(n)
    for i in range(1, n):
        dz = z[i] - z[i - 1]
        tau[i] = tau[i - 1] + 0.5 * (kappa[i - 1] + kappa[i]) * dz

    # Radiative transfer integral: Tb = ε ∫ T(z) κ(z) exp(-τ(z)) dz
    integrand = T_z * kappa * np.exp(-tau)
    Tb = eps * _trapezoid(integrand, z)

    # Add contribution from the bottom boundary if τ_max < ~5
    # (profile doesn't extend deep enough to be optically thick)
    if tau[-1] < 5.0:
        Tb += eps * T_z[-1] * np.exp(-tau[-1])

    return float(Tb)


def radiative_transfer_batch(z, T_profiles, freq_ghz, porosity,
                             emissivity=1.0, mixing="maxwell_garnett"):
    """Compute Tb for multiple T(z) profiles (e.g., all mesh faces).

    Parameters
    ----------
    z : ndarray, shape (n_depth,)
        Common depth grid in meters.
    T_profiles : ndarray, shape (n_faces, n_depth)
        Temperature profiles for each face.
    freq_ghz : float
        Frequency in GHz.
    porosity : float or ndarray
        Porosity (scalar or shape (n_depth,) for depth-varying).
    emissivity : float
        Surface emissivity.
    mixing : str
        Mixing model.

    Returns
    -------
    ndarray, shape (n_faces,)
        Brightness temperature for each face.
    """
    n_faces = T_profiles.shape[0]
    Tb = np.zeros(n_faces)

    for i in range(n_faces):
        Tb[i] = radiative_transfer_1d(
            z, T_profiles[i], freq_ghz, porosity, emissivity, mixing
        )

    return Tb


def radiative_transfer_emission_angle(z, T_z, freq_ghz, porosity,
                                      emission_angle_deg, emissivity="fresnel",
                                      mixing="maxwell_garnett"):
    """Compute Tb accounting for emission angle (limb effects).

    At emission angle θ, the effective path length through each layer
    increases by sec(θ), increasing the optical depth:
        τ_eff(z) = τ(z) / cos(θ)

    Parameters
    ----------
    z, T_z, freq_ghz, porosity, mixing :
        Same as radiative_transfer_1d.
    emission_angle_deg : float
        Emission angle in degrees (0 = nadir, 90 = limb).
    emissivity : float or "fresnel"
        Surface emissivity.  "fresnel" (default) computes the
        angle-dependent Fresnel emissivity from the dielectric model,
        which is the physically correct choice for a full RT integral.
        A scalar value overrides it (e.g. 0.75, to reproduce a published
        surface-only fit).

    Returns
    -------
    float
        Brightness temperature in K.
    """
    z = np.asarray(z, dtype=float)
    T_z = np.asarray(T_z, dtype=float)
    n = len(z)

    if np.isscalar(porosity):
        porosity_arr = np.full(n, porosity)
    else:
        porosity_arr = np.asarray(porosity, dtype=float)

    cos_theta = np.cos(np.radians(emission_angle_deg))

    # Resolve emissivity
    if emissivity == "fresnel":
        # Use representative surface temperature for dielectric calc
        T_surf = float(T_z[0]) if len(T_z) > 0 else 100.0
        p_surf = float(porosity_arr[0])
        eps = fresnel_emissivity(freq_ghz, T_surf, p_surf,
                                 emission_angle_deg, mixing)
    else:
        eps = float(emissivity)

    # Compute absorption coefficient at each depth
    kappa = np.array([
        absorption_coefficient(freq_ghz, T_z[i], porosity_arr[i], mixing)
        for i in range(n)
    ])

    # Optical depth scaled by 1/cos(θ)
    kappa_eff = kappa / cos_theta

    tau = np.zeros(n)
    for i in range(1, n):
        dz = z[i] - z[i - 1]
        tau[i] = tau[i - 1] + 0.5 * (kappa_eff[i - 1] + kappa_eff[i]) * dz

    integrand = T_z * kappa_eff * np.exp(-tau)
    Tb = eps * _trapezoid(integrand, z)

    if tau[-1] < 5.0:
        Tb += eps * T_z[-1] * np.exp(-tau[-1])

    return float(Tb)


def beam_convolve(Tb_map, lat_map, lon_map, beam_fwhm_arcsec,
                  ang_diam_arcsec, sub_obs_lon, sub_obs_lat):
    """Convolve a model Tb map with a 2D Gaussian instrument beam.

    Projects the model map to the observer's plane-of-sky, convolves
    with the beam, then maps back to lat/lon. Use this when comparing a
    model against a spatially resolved observation whose angular
    resolution is coarser than the model grid.

    Parameters
    ----------
    Tb_map : ndarray, shape (n_lat, n_lon)
        Model brightness temperature on a regular lat/lon grid.
    lat_map, lon_map : ndarray
        1D arrays of latitude and longitude values (degrees).
    beam_fwhm_arcsec : float or tuple
        Beam FWHM in arcseconds (scalar for circular, tuple (major, minor)).
    ang_diam_arcsec : float
        Angular diameter of the body in arcseconds.
    sub_obs_lon, sub_obs_lat : float
        Sub-observer point in degrees.

    Returns
    -------
    ndarray
        Beam-convolved Tb map (same shape as input).
    """
    from scipy.ndimage import gaussian_filter

    n_lat = len(lat_map)
    n_lon = len(lon_map)

    # Angular size of Europa in the sky
    radius_arcsec = ang_diam_arcsec / 2.0

    # Pixel scale on the sky: approximate using the angular extent
    # covered by the lat/lon grid
    dlat = np.mean(np.diff(lat_map))
    dlon = np.mean(np.diff(lon_map))

    # Convert beam FWHM to sigma (FWHM = 2.355 × σ)
    if np.isscalar(beam_fwhm_arcsec):
        beam_maj = beam_min = beam_fwhm_arcsec
    else:
        beam_maj, beam_min = beam_fwhm_arcsec

    # Approximate: 1 degree of latitude ≈ radius_arcsec * π/180 arcsec
    arcsec_per_deg_lat = radius_arcsec * np.pi / 180.0 * 2.0 / np.pi
    # This is approximate; for proper convolution we'd project to plane-of-sky

    # For a simple approach: compute sigma in grid units
    sigma_lat = (beam_maj / 2.355) / (abs(dlat) * ang_diam_arcsec / 180.0 * np.pi)
    sigma_lon = (beam_min / 2.355) / (abs(dlon) * ang_diam_arcsec / 180.0 * np.pi)

    # Clamp to reasonable values
    sigma_lat = max(sigma_lat, 0.5)
    sigma_lon = max(sigma_lon, 0.5)

    # Apply Gaussian smoothing
    Tb_smooth = gaussian_filter(Tb_map, sigma=[sigma_lat, sigma_lon],
                                mode='wrap')

    return Tb_smooth


def project_model_to_image(face_tb, face_lat, face_lon,
                           nx, ny, pixel_scale_arcsec, ang_diam_arcsec,
                           sub_obs_lon, sub_obs_lat):
    """Project face-resolved Tb onto an image plane matching an observation.

    Parameters
    ----------
    face_tb : ndarray, shape (n_faces,)
        Brightness temperature for each mesh face.
    face_lat, face_lon : ndarray, shape (n_faces,)
        Face center latitude and longitude (degrees).
    nx, ny : int
        Output image dimensions in pixels.
    pixel_scale_arcsec : float
        Image pixel scale (arcsec/pixel).
    ang_diam_arcsec : float
        Angular diameter of the body (arcsec).
    sub_obs_lon, sub_obs_lat : float
        Sub-observer point (degrees).

    Returns
    -------
    ndarray, shape (ny, nx)
        Tb image. NaN for off-disk pixels.
    """
    radius_arcsec = ang_diam_arcsec / 2.0
    radius_pix = radius_arcsec / pixel_scale_arcsec

    center_x = nx / 2.0
    center_y = ny / 2.0

    # Convert face lat/lon to observer-frame coordinates
    lat_rad = np.radians(face_lat)
    delta_lon_rad = np.radians(face_lon - sub_obs_lon)
    sub_lat_rad = np.radians(sub_obs_lat)

    # Check visibility: face must be on the near side
    cos_c = (np.sin(sub_lat_rad) * np.sin(lat_rad) +
             np.cos(sub_lat_rad) * np.cos(lat_rad) * np.cos(delta_lon_rad))
    visible = cos_c > 0

    # Orthographic projection to observer plane
    x_obs = np.cos(lat_rad) * np.sin(delta_lon_rad)
    y_obs = (np.cos(sub_lat_rad) * np.sin(lat_rad) -
             np.sin(sub_lat_rad) * np.cos(lat_rad) * np.cos(delta_lon_rad))

    # Convert to pixel coordinates
    px = center_x + x_obs * radius_pix
    py = center_y + y_obs * radius_pix

    # Build output image by nearest-neighbor assignment
    image = np.full((ny, nx), np.nan)
    weight = np.zeros((ny, nx))

    for i in range(len(face_tb)):
        if not visible[i]:
            continue
        ix = int(round(px[i]))
        iy = int(round(py[i]))
        if 0 <= ix < nx and 0 <= iy < ny:
            if np.isnan(image[iy, ix]) or weight[iy, ix] == 0:
                image[iy, ix] = face_tb[i]
                weight[iy, ix] = 1.0
            else:
                # Average if multiple faces map to same pixel
                w = weight[iy, ix]
                image[iy, ix] = (image[iy, ix] * w + face_tb[i]) / (w + 1)
                weight[iy, ix] = w + 1

    return image


def process_ptm3d_output(npz_path, freq_ghz=233.0, porosity=0.5,
                         emissivity=0.75, mixing="maxwell_garnett",
                         output_dir=None):
    """Process a PTM3D output file to produce predicted Tb.

    Parameters
    ----------
    npz_path : str or Path
        Path to PTM3D output .npz file containing T(z) profiles.
    freq_ghz : float
        Observing frequency in GHz.
    porosity : float or ndarray
        Porosity (uniform or depth-varying).
    emissivity : float
        Surface emissivity.
    mixing : str
        Dielectric mixing model.
    output_dir : str or Path, optional
        Directory for output files.

    Returns
    -------
    dict
        Predicted Tb values and metadata.
    """
    npz_path = Path(npz_path)
    data = np.load(str(npz_path))

    # Expected arrays from PTM3D output:
    # - depth_grid: shape (n_depth,) in meters
    # - temperatures: shape (n_faces, n_depth)
    # - face_lat, face_lon: shape (n_faces,)
    # - (optional) face_areas: shape (n_faces,)

    if "depth_grid" not in data:
        print("  WARNING: No depth_grid found in PTM3D output.")
        print("  Available keys:", list(data.keys()))
        return None

    z = data["depth_grid"]
    T_profiles = data["temperatures"]
    face_lat = data.get("face_lat", None)
    face_lon = data.get("face_lon", None)

    n_faces, n_depth = T_profiles.shape
    print(f"\n  Processing PTM3D output: {npz_path.name}")
    print(f"  Faces: {n_faces}, Depth layers: {n_depth}")
    print(f"  Depth range: {z[0]*100:.2f} - {z[-1]*100:.2f} cm")
    print(f"  Frequency: {freq_ghz:.1f} GHz, Porosity: {porosity*100:.0f}%")
    print(f"  Emissivity: {emissivity:.2f}")

    # Check depth coverage vs. penetration depth
    L = penetration_depth(freq_ghz, np.mean(T_profiles[:, 0]), porosity, mixing)
    print(f"  Penetration depth: {L*100:.2f} cm "
          f"({'OK' if z[-1] > 3*L else 'WARNING: profile too shallow'})")

    # Compute Tb for each face
    Tb = radiative_transfer_batch(z, T_profiles, freq_ghz, porosity,
                                  emissivity, mixing)

    print(f"  Predicted Tb range: {np.nanmin(Tb):.1f} - {np.nanmax(Tb):.1f} K")
    print(f"  Predicted Tb mean:  {np.nanmean(Tb):.1f} K")

    result = {
        "Tb": Tb,
        "freq_ghz": freq_ghz,
        "porosity": porosity,
        "emissivity": emissivity,
        "mixing": mixing,
        "penetration_depth_m": L,
    }

    if face_lat is not None:
        result["face_lat"] = face_lat
        result["face_lon"] = face_lon

    # Save output
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{npz_path.stem}_Tb_{freq_ghz:.0f}GHz.npz"
        np.savez_compressed(str(out_path), **result)
        print(f"  Saved: {out_path}")

    return result


def compute_disk_averaged_tb(z, T_profiles, face_lat, freq_ghz, porosity,
                             emissivity=0.75, mixing="maxwell_garnett",
                             sub_obs_lat=0.0):
    """Compute disk-averaged brightness temperature.

    Weights each face by its projected area as seen from the observer:
        <Tb> = Σ Tb_i × cos(emission_i) / Σ cos(emission_i)

    This is useful for comparison with spatially-unresolved or
    disk-integrated measurements.

    Parameters
    ----------
    z : ndarray
        Depth grid (m).
    T_profiles : ndarray, shape (n_faces, n_depth)
        Temperature profiles.
    face_lat : ndarray
        Face latitudes (degrees).
    freq_ghz : float
        Frequency (GHz).
    porosity, emissivity, mixing : as above.
    sub_obs_lat : float
        Sub-observer latitude (degrees).

    Returns
    -------
    float
        Disk-averaged Tb in K.
    """
    Tb = radiative_transfer_batch(z, T_profiles, freq_ghz, porosity,
                                  emissivity, mixing)

    # Weight by cosine of latitude (proxy for projected area)
    cos_lat = np.cos(np.radians(face_lat - sub_obs_lat))
    cos_lat = np.maximum(cos_lat, 0)  # Only visible hemisphere

    Tb_avg = np.average(Tb, weights=cos_lat)
    return float(Tb_avg)


def main():
    parser = argparse.ArgumentParser(
        description="Subsurface T(z) → depth-weighted brightness temperature"
    )
    parser.add_argument("input", nargs="?",
                        help="PTM3D output .npz file")
    parser.add_argument("--freq", type=float, default=233.0,
                        help="Observing frequency in GHz (default: 233)")
    parser.add_argument("--porosity", type=float, default=0.5,
                        help="Porosity (default: 0.5)")
    parser.add_argument("--emissivity", type=float, default=0.75,
                        help="Surface emissivity (default: 0.75)")
    parser.add_argument("--mixing", choices=["maxwell_garnett", "bruggeman"],
                        default="maxwell_garnett")
    parser.add_argument("-o", "--output-dir",
                        help="Output directory")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic T(z) profile for testing")
    args = parser.parse_args()

    freq_ghz = args.freq

    if args.demo:
        # Generate a synthetic T(z) profile for testing
        print("\n  Running demo with a synthetic icy-surface T(z) profile...")
        z = np.linspace(0, 0.5, 100)  # 0 to 50 cm

        # Typical icy-surface diurnal profile: warm surface, cold subsurface
        T_surface = 110.0  # K (subsolar point)
        T_deep = 95.0      # K (diurnal mean)
        skin_depth = 0.02   # 2 cm thermal skin depth

        T_z = T_deep + (T_surface - T_deep) * np.exp(-z / skin_depth)

        print(f"  T(0) = {T_z[0]:.1f} K, T(50cm) = {T_z[-1]:.1f} K")
        print(f"  Thermal skin depth: {skin_depth*100:.1f} cm")

        for freq in REFERENCE_FREQUENCIES_GHZ:
            Tb = radiative_transfer_1d(z, T_z, freq, args.porosity,
                                       args.emissivity, args.mixing)
            L = penetration_depth(freq, T_surface, args.porosity, args.mixing)
            print(f"  {freq:6.1f} GHz: Tb = {Tb:.2f} K, "
                  f"L_e = {L*100:.2f} cm")

        # Test emission angle dependence
        print(f"\n  Emission angle dependence (233 GHz):")
        for angle in [0, 20, 40, 60, 75]:
            Tb = radiative_transfer_emission_angle(
                z, T_z, 233.0, args.porosity, angle,
                args.emissivity, args.mixing
            )
            print(f"    θ = {angle:>2d}°: Tb = {Tb:.2f} K")
        return

    if args.input is None:
        parser.print_help()
        return

    process_ptm3d_output(
        args.input, freq_ghz, args.porosity,
        args.emissivity, args.mixing, args.output_dir
    )


if __name__ == "__main__":
    main()
