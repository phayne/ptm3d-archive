#!/usr/bin/env python3
"""
Convert ALMA FITS images from Jy/beam to brightness temperature (K).

Supports both Rayleigh-Jeans approximation and full Planck inversion.
At 233 GHz and ~100 K, Rayleigh-Jeans is accurate to ~1%.

Usage:
  python alma_to_tb.py input.fits                    # R-J approximation
  python alma_to_tb.py input.fits --planck           # Full Planck inversion
  python alma_to_tb.py input.fits -o output_tb.fits  # Specify output
"""

import argparse
import numpy as np
from pathlib import Path

# Physical constants (SI)
C_LIGHT = 2.99792458e8       # Speed of light [m/s]
K_BOLTZ = 1.380649e-23       # Boltzmann constant [J/K]
H_PLANCK = 6.62607015e-34    # Planck constant [J·s]


def beam_solid_angle(bmaj_deg, bmin_deg):
    """Compute beam solid angle from FWHM axes in degrees.

    Ω_beam = π * θ_maj * θ_min / (4 ln 2)

    Parameters
    ----------
    bmaj_deg, bmin_deg : float
        Beam FWHM major and minor axes in degrees.

    Returns
    -------
    float
        Beam solid angle in steradians.
    """
    bmaj_rad = np.radians(bmaj_deg)
    bmin_rad = np.radians(bmin_deg)
    return np.pi * bmaj_rad * bmin_rad / (4.0 * np.log(2.0))


def jy_per_beam_to_tb_rj(flux_jy, freq_hz, omega_beam):
    """Convert flux density (Jy/beam) to Rayleigh-Jeans brightness temperature.

    T_b = c² S_ν / (2 k_B ν² Ω_beam)

    Parameters
    ----------
    flux_jy : array_like
        Flux density in Jy/beam (1 Jy = 1e-26 W/m²/Hz).
    freq_hz : float
        Observing frequency in Hz.
    omega_beam : float
        Beam solid angle in steradians.

    Returns
    -------
    array_like
        Brightness temperature in K.
    """
    flux_si = flux_jy * 1e-26  # Convert Jy to W/m²/Hz
    specific_intensity = flux_si / omega_beam  # W/m²/Hz/sr
    tb = C_LIGHT**2 * specific_intensity / (2.0 * K_BOLTZ * freq_hz**2)
    return tb


def jy_per_beam_to_tb_planck(flux_jy, freq_hz, omega_beam):
    """Convert flux density (Jy/beam) to brightness temperature via Planck inversion.

    More accurate than Rayleigh-Jeans at high frequencies or low temperatures.

    Parameters
    ----------
    flux_jy : array_like
        Flux density in Jy/beam.
    freq_hz : float
        Observing frequency in Hz.
    omega_beam : float
        Beam solid angle in steradians.

    Returns
    -------
    array_like
        Brightness temperature in K.
    """
    flux_si = flux_jy * 1e-26
    specific_intensity = flux_si / omega_beam  # W/m²/Hz/sr

    # Planck inversion: T = hν/k / ln(1 + 2hν³/(c²·Iν))
    x = H_PLANCK * freq_hz / K_BOLTZ
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 1.0 + 2.0 * H_PLANCK * freq_hz**3 / (C_LIGHT**2 * specific_intensity)
        tb = np.where(specific_intensity > 0, x / np.log(arg), 0.0)

    return tb


def convert_fits(input_path, output_path=None, use_planck=False, freq_override=None):
    """Convert an ALMA FITS image from Jy/beam to brightness temperature.

    Parameters
    ----------
    input_path : str or Path
        Path to input FITS file.
    output_path : str or Path, optional
        Path for output FITS file. Default: input_tb.fits
    use_planck : bool
        Use full Planck inversion instead of Rayleigh-Jeans.
    freq_override : float, optional
        Override frequency in GHz (if FITS header is missing it).

    Returns
    -------
    dict
        Metadata about the conversion.
    """
    from astropy.io import fits as pyfits

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_tb")

    print(f"\nConverting: {input_path.name}")
    print(f"Method: {'Planck inversion' if use_planck else 'Rayleigh-Jeans'}")

    with pyfits.open(str(input_path)) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.copy()

        # Handle 4D cubes (freq, stokes, y, x) — squeeze to 2D
        while data.ndim > 2:
            data = data[0]

        # Get frequency — prefer CRVAL3 (actual observing frequency) over
        # RESTFRQ (rest frequency reference, often different for continuum)
        if freq_override is not None:
            freq_hz = freq_override * 1e9
        elif "CRVAL3" in hdr and hdr.get("CTYPE3", "").startswith("FREQ"):
            freq_hz = float(hdr["CRVAL3"])
        elif "RESTFRQ" in hdr:
            freq_hz = float(hdr["RESTFRQ"])
        elif "CRVAL3" in hdr:
            freq_hz = float(hdr["CRVAL3"])
        else:
            raise ValueError("Cannot determine frequency from FITS header. "
                             "Use --freq to specify.")

        freq_ghz = freq_hz / 1e9

        # Get beam parameters
        bmaj = hdr.get("BMAJ")
        bmin = hdr.get("BMIN")
        if bmaj is None or bmin is None:
            raise ValueError("Cannot find beam parameters (BMAJ/BMIN) in header.")

        omega = beam_solid_angle(bmaj, bmin)
        bmaj_mas = bmaj * 3600 * 1000
        bmin_mas = bmin * 3600 * 1000

        # Get flux units
        bunit = hdr.get("BUNIT", "").strip()
        if "Jy/beam" not in bunit and "JY/BEAM" not in bunit.upper():
            print(f"  WARNING: BUNIT = '{bunit}', expected 'Jy/beam'")
            if "K" in bunit.upper():
                print("  Image appears to already be in temperature units!")
                return {"already_tb": True}

        print(f"  Frequency: {freq_ghz:.3f} GHz ({C_LIGHT/freq_hz*1e3:.3f} mm)")
        print(f"  Beam: {bmaj_mas:.1f} x {bmin_mas:.1f} mas")
        print(f"  Beam solid angle: {omega:.4e} sr")
        print(f"  Image size: {data.shape}")

        # Convert
        if use_planck:
            tb = jy_per_beam_to_tb_planck(data, freq_hz, omega)
        else:
            tb = jy_per_beam_to_tb_rj(data, freq_hz, omega)

        # Mask invalid pixels (NaN, zero, negative)
        valid = np.isfinite(data) & (data > 0)
        tb[~valid] = np.nan

        # Statistics on valid pixels
        tb_valid = tb[valid]
        if len(tb_valid) > 0:
            print(f"  Tb range: {np.nanmin(tb_valid):.1f} - {np.nanmax(tb_valid):.1f} K")
            print(f"  Tb mean:  {np.nanmean(tb_valid):.1f} K")
            print(f"  Valid pixels: {np.sum(valid)} / {data.size}")

            # R-J correction factor at this frequency and temperature
            x = H_PLANCK * freq_hz / (K_BOLTZ * np.nanmean(tb_valid))
            rj_correction = x / (np.exp(x) - 1)
            print(f"  R-J correction factor: {rj_correction:.4f} "
                  f"(error ~ {(1-rj_correction)*100:.1f}%)")

        # Update header and write
        hdr["BUNIT"] = "K"
        hdr["BTYPE"] = "Brightness Temperature"
        hdr.add_history(f"Converted from Jy/beam to Tb using "
                        f"{'Planck' if use_planck else 'Rayleigh-Jeans'}")
        hdr.add_history(f"Frequency: {freq_ghz:.3f} GHz, "
                        f"Beam: {bmaj_mas:.1f}x{bmin_mas:.1f} mas")

        # Write output — reshape to match original dimensions
        out_data = tb
        if hdul[0].data.ndim > 2:
            out_data = tb.reshape(hdul[0].data.shape[-2:])

        pyfits.writeto(str(output_path), out_data, hdr, overwrite=True)
        print(f"  Written: {output_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "freq_ghz": freq_ghz,
        "beam_mas": (bmaj_mas, bmin_mas),
        "tb_range": (float(np.nanmin(tb_valid)), float(np.nanmax(tb_valid)))
        if len(tb_valid) > 0 else None,
        "method": "planck" if use_planck else "rayleigh-jeans",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ALMA FITS images from Jy/beam to brightness temperature"
    )
    parser.add_argument("input", nargs="+", help="Input FITS file(s)")
    parser.add_argument("-o", "--output", help="Output FITS file (single input only)")
    parser.add_argument("--planck", action="store_true",
                        help="Use full Planck inversion (default: Rayleigh-Jeans)")
    parser.add_argument("--freq", type=float,
                        help="Override frequency in GHz")
    args = parser.parse_args()

    for fpath in args.input:
        output = args.output if len(args.input) == 1 else None
        convert_fits(fpath, output, use_planck=args.planck, freq_override=args.freq)


if __name__ == "__main__":
    main()
