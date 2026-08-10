#!/usr/bin/env python3
"""
Dielectric properties of porous water ice at millimeter wavelengths.

Computes the complex permittivity, absorption coefficient, and penetration
depth of a porous water-ice regolith. The parameterizations are validated
over roughly 80–270 K and 30–400 GHz, which covers the icy-satellite surface
regime (80–130 K) at millimeter wavelengths.

This module is instrument-independent: it takes a frequency in GHz and a
temperature in K and returns bulk electromagnetic properties. Pair it with
``depth_weighted_emission.py`` to turn a subsurface temperature profile into
a brightness temperature.

Models:
  - Ice permittivity: Mätzler (2006), based on Hufford (1991)
  - Porosity mixing: Maxwell-Garnett effective medium theory
  - Absorption: standard electromagnetic loss formulation

References:
  Hufford (1991) — Int. J. Infrared Millimeter Waves, 12, 677
  Mätzler (2006) — Thermal Microwave Radiation, IET, Ch. 5.3
  Mishima et al. (1983) — J. Chem. Phys., 78, 6399
  Warren & Brandt (2008) — J. Geophys. Res., 113, D14220

Usage:
  python dielectric_mixing.py --freq 233 --temp 100 --porosity 0.5
  python dielectric_mixing.py --plot   # Generate diagnostic plots
"""

import argparse
import numpy as np

# Physical constants
C_LIGHT = 2.99792458e8       # Speed of light [m/s]
K_BOLTZ = 1.380649e-23       # Boltzmann constant [J/K]
H_PLANCK = 6.62607015e-34    # Planck constant [J·s]

# Frequencies used by the diagnostics and the command-line summary. They are
# spread across the millimeter window (about 3 mm, 1.3 mm and 0.9 mm) purely so
# the printed tables and plots span the useful range; nothing in the physics
# below is tied to these values, and any frequency in GHz may be passed instead.
REFERENCE_FREQUENCIES_GHZ = (97.5, 233.0, 343.5)


# ─────────────────────────────────────────────────────────────────────
# 1. Pure ice permittivity
# ─────────────────────────────────────────────────────────────────────

def epsilon_real_ice(T):
    """Real part of pure ice permittivity (nearly constant).

    From Mätzler (2006), Eq. 5.35:
        ε'(T) = 3.1884 + 9.1e-4 * (T - 273.15)

    Parameters
    ----------
    T : float or ndarray
        Temperature in Kelvin.

    Returns
    -------
    float or ndarray
        Real permittivity ε'.
    """
    return 3.1884 + 9.1e-4 * (np.asarray(T, dtype=float) - 273.15)


def epsilon_imag_ice(freq_ghz, T):
    """Imaginary part of pure ice permittivity.

    Implements the Hufford (1991) / Mätzler (2006) parameterization
    combining Debye relaxation tail and infrared resonance wing.

    At icy-satellite surface temperatures (80–130 K), the Debye relaxation
    frequency is extremely low (< 1 Hz), so the Debye tail contribution at
    mm-wave frequencies is negligible. The dominant contribution comes
    from the infrared lattice absorption wing.

    Parameters
    ----------
    freq_ghz : float or ndarray
        Frequency in GHz.
    T : float or ndarray
        Temperature in Kelvin.

    Returns
    -------
    float or ndarray
        Imaginary permittivity ε''.
    """
    freq_ghz = np.asarray(freq_ghz, dtype=float)
    T = np.asarray(T, dtype=float)

    theta = 300.0 / T - 1.0

    # --- Debye relaxation contribution ---
    # Relaxation time: Arrhenius form
    # At T < 243 K, use the low-temperature parameterization
    # (Mätzler 2006, Eq. 5.36–5.37)
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    eps_debye = alpha / freq_ghz

    # --- Infrared resonance wing contribution ---
    # Two-term fit from Mätzler (2006):
    #   β = β₁ + β₂
    # where β₁ is the high-frequency Debye tail and β₂ is the IR wing
    b1_num = 0.502 - 0.131 * theta / (1.0 + theta)
    beta1 = b1_num * 1e-4
    beta1 = np.maximum(beta1, 0.0)  # Ensure non-negative

    beta2 = 0.542e-6 * ((1.0 + theta) / (theta + 0.0073)) ** 2

    eps_ir = (beta1 + beta2) * freq_ghz

    return eps_debye + eps_ir


def loss_tangent_ice(freq_ghz, T):
    """Loss tangent of pure ice: tan δ = ε''/ε'.

    Parameters
    ----------
    freq_ghz : float or ndarray
        Frequency in GHz.
    T : float or ndarray
        Temperature in Kelvin.

    Returns
    -------
    float or ndarray
        Loss tangent.
    """
    return epsilon_imag_ice(freq_ghz, T) / epsilon_real_ice(T)


# ─────────────────────────────────────────────────────────────────────
# 2. Porosity mixing models
# ─────────────────────────────────────────────────────────────────────

def maxwell_garnett(eps_matrix, eps_inclusion, f_inclusion):
    """Maxwell-Garnett effective medium approximation.

    Treats inclusions as spheres embedded in a matrix material.

    Parameters
    ----------
    eps_matrix : complex
        Permittivity of the matrix (host medium).
    eps_inclusion : complex
        Permittivity of the inclusions.
    f_inclusion : float
        Volume fraction of inclusions (0 to 1).

    Returns
    -------
    complex
        Effective permittivity.
    """
    em = np.asarray(eps_matrix, dtype=complex)
    ei = np.asarray(eps_inclusion, dtype=complex)
    f = np.asarray(f_inclusion, dtype=float)

    numer = ei + 2.0 * em + 2.0 * f * (ei - em)
    denom = ei + 2.0 * em - f * (ei - em)

    return em * numer / denom


def bruggeman(eps1, eps2, f1):
    """Bruggeman (symmetric) effective medium approximation.

    Treats both phases symmetrically — no distinguished host.
    Solved iteratively from the self-consistency equation:
        f₁ (ε₁ - εeff)/(ε₁ + 2εeff) + f₂ (ε₂ - εeff)/(ε₂ + 2εeff) = 0

    Parameters
    ----------
    eps1, eps2 : complex
        Permittivities of phases 1 and 2.
    f1 : float
        Volume fraction of phase 1.

    Returns
    -------
    complex
        Effective permittivity.
    """
    e1 = np.asarray(eps1, dtype=complex)
    e2 = np.asarray(eps2, dtype=complex)
    f = np.asarray(f1, dtype=float)
    f2 = 1.0 - f

    # Iterative solution (Newton's method)
    eps_eff = f * e1 + f2 * e2  # Initial guess: linear mixing
    for _ in range(50):
        g1 = f * (e1 - eps_eff) / (e1 + 2.0 * eps_eff)
        g2 = f2 * (e2 - eps_eff) / (e2 + 2.0 * eps_eff)
        residual = g1 + g2

        # Derivative of residual w.r.t. eps_eff
        dg1 = -f * (3.0 * e1) / (e1 + 2.0 * eps_eff) ** 2
        dg2 = -f2 * (3.0 * e2) / (e2 + 2.0 * eps_eff) ** 2
        d_residual = dg1 + dg2

        delta = residual / d_residual
        eps_eff = eps_eff - delta
        if np.all(np.abs(delta) < 1e-12 * np.abs(eps_eff)):
            break

    return eps_eff


def porous_ice_permittivity(freq_ghz, T, porosity, mixing="maxwell_garnett"):
    """Effective complex permittivity of porous ice regolith.

    For high porosity (>50%), ice grains are inclusions in vacuum.
    For low porosity (<50%), vacuum pores are inclusions in ice.
    The Maxwell-Garnett model switches topology at 50% porosity
    (Hashin-Shtrikman bounds). Bruggeman is symmetric.

    Parameters
    ----------
    freq_ghz : float or ndarray
        Frequency in GHz.
    T : float or ndarray
        Temperature in Kelvin.
    porosity : float
        Porosity (void fraction), 0.0 = solid ice, 1.0 = vacuum.
    mixing : str
        Mixing model: "maxwell_garnett" or "bruggeman".

    Returns
    -------
    complex or ndarray
        Effective complex permittivity.
    """
    eps_ice = epsilon_real_ice(T) + 1j * epsilon_imag_ice(freq_ghz, T)
    eps_vac = 1.0 + 0j

    f_ice = 1.0 - porosity  # Ice volume fraction

    if mixing == "bruggeman":
        return bruggeman(eps_ice, eps_vac, f_ice)
    elif mixing == "maxwell_garnett":
        # Switch topology at 50% porosity
        if porosity > 0.5:
            # Ice inclusions in vacuum matrix
            return maxwell_garnett(eps_vac, eps_ice, f_ice)
        else:
            # Vacuum inclusions in ice matrix
            return maxwell_garnett(eps_ice, eps_vac, porosity)
    else:
        raise ValueError(f"Unknown mixing model: {mixing}")


# ─────────────────────────────────────────────────────────────────────
# 3. Absorption coefficient and penetration depth
# ─────────────────────────────────────────────────────────────────────

def absorption_coefficient(freq_ghz, T, porosity=0.0, mixing="maxwell_garnett"):
    """Compute the power absorption coefficient κ_a.

    κ_a = 2π ν / c × Im(√ε_eff)

    or equivalently for low-loss materials (ε'' << ε'):
    κ_a ≈ 2π ν √ε' × tan δ / c

    Parameters
    ----------
    freq_ghz : float or ndarray
        Frequency in GHz.
    T : float or ndarray
        Temperature in Kelvin.
    porosity : float
        Porosity (void fraction).
    mixing : str
        Mixing model.

    Returns
    -------
    float or ndarray
        Absorption coefficient in 1/m.
    """
    freq_hz = freq_ghz * 1e9
    eps_eff = porous_ice_permittivity(freq_ghz, T, porosity, mixing)

    # Exact formula: κ = 2πν/c × Im(√ε)
    n_complex = np.sqrt(eps_eff)
    kappa = 2.0 * np.pi * freq_hz / C_LIGHT * np.imag(n_complex)

    return np.real(kappa)


def fresnel_emissivity(freq_ghz, T, porosity, emission_angle_deg=0.0,
                       mixing="maxwell_garnett"):
    """Compute mm-wave surface emissivity from Fresnel equations.

    For a full radiative transfer integral through T(z), the correct
    emissivity is (1 - R) where R is the Fresnel power reflectivity at
    the vacuum–regolith interface.  This is the physically consistent
    alternative to a fitted effective emissivity (values near 0.75 are
    common in the literature): those are calibrated against a surface-only
    model (Tb = ε T_surf) and would double-count the subsurface probing
    effect if reused inside an RT integral.

    Parameters
    ----------
    freq_ghz : float
        Frequency in GHz.
    T : float
        Representative temperature in K (for dielectric calculation).
    porosity : float
        Porosity (void fraction).
    emission_angle_deg : float
        Emission angle in degrees (0 = nadir).
    mixing : str
        Dielectric mixing model.

    Returns
    -------
    float
        Surface emissivity ε = 1 − R(θ), averaged over polarizations.
    """
    eps_eff = porous_ice_permittivity(freq_ghz, T, porosity, mixing)
    n_complex = np.sqrt(eps_eff)
    n = np.real(n_complex)  # real refractive index (loss is negligible for R)

    theta = np.radians(np.asarray(emission_angle_deg, dtype=float))
    cos_i = np.cos(theta)
    sin_i = np.sin(theta)

    # Fresnel reflection coefficients (vacuum → dielectric)
    cos_t = np.sqrt(np.maximum(1.0 - (sin_i / n)**2, 0.0))
    Rs = np.abs((cos_i - n * cos_t) / (cos_i + n * cos_t))**2
    Rp = np.abs((n * cos_i - cos_t) / (n * cos_i + cos_t))**2
    R = 0.5 * (Rs + Rp)

    return float(1.0 - R)


def penetration_depth(freq_ghz, T, porosity=0.0, mixing="maxwell_garnett"):
    """Electrical (1/e) penetration depth in meters.

    L = 1 / κ_a

    Parameters
    ----------
    freq_ghz : float or ndarray
        Frequency in GHz.
    T : float or ndarray
        Temperature in Kelvin.
    porosity : float
        Porosity.
    mixing : str
        Mixing model.

    Returns
    -------
    float or ndarray
        Penetration depth in meters.
    """
    kappa = absorption_coefficient(freq_ghz, T, porosity, mixing)
    return np.where(kappa > 0, 1.0 / kappa, np.inf)


# ─────────────────────────────────────────────────────────────────────
# 4. Convenience / diagnostic functions
# ─────────────────────────────────────────────────────────────────────

def print_dielectric_summary(freq_ghz, T, porosity=0.0, mixing="maxwell_garnett"):
    """Print a summary of dielectric properties."""
    eps_eff = porous_ice_permittivity(freq_ghz, T, porosity, mixing)
    eps_r = np.real(eps_eff)
    eps_i = np.imag(eps_eff)
    tand = eps_i / eps_r
    kappa = absorption_coefficient(freq_ghz, T, porosity, mixing)
    L = penetration_depth(freq_ghz, T, porosity, mixing)
    wavelength_mm = C_LIGHT / (freq_ghz * 1e9) * 1e3

    print(f"\n  Ice Dielectric Properties")
    print(f"  {'─'*50}")
    print(f"  Frequency:      {freq_ghz:.1f} GHz ({wavelength_mm:.2f} mm)")
    print(f"  Temperature:    {T:.1f} K")
    print(f"  Porosity:       {porosity*100:.0f}%")
    print(f"  Mixing model:   {mixing}")
    print(f"  {'─'*50}")
    print(f"  ε' (real):      {eps_r:.4f}")
    print(f"  ε'' (imag):     {eps_i:.4e}")
    print(f"  tan δ:          {tand:.4e}")
    print(f"  κ_a:            {kappa:.4f} m⁻¹")
    print(f"  Penetration:    {L:.4f} m ({L*100:.2f} cm)")
    print(f"  {'─'*50}")

    # Pure ice for reference
    if porosity > 0:
        eps_pure = epsilon_real_ice(T) + 1j * epsilon_imag_ice(freq_ghz, T)
        kappa_pure = absorption_coefficient(freq_ghz, T, 0.0)
        L_pure = penetration_depth(freq_ghz, T, 0.0)
        print(f"  (Pure ice:      ε'={np.real(eps_pure):.4f}, "
              f"κ={kappa_pure:.4f} m⁻¹, L={L_pure*100:.2f} cm)")


def make_diagnostic_plots(output_dir=None):
    """Generate diagnostic plots of ice dielectric properties.

    Creates:
    1. Loss tangent vs. temperature at three representative frequencies
    2. Penetration depth vs. porosity at 100 K
    3. Absorption coefficient vs. frequency for several temperatures
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        from pathlib import Path
        output_dir = Path(__file__).parent.parent.parent / "data" / "emission" / "diagnostics"
    else:
        from pathlib import Path
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Representative millimeter frequencies spanning the useful range.
    bands = {f"{f:g} GHz": f for f in REFERENCE_FREQUENCIES_GHZ}

    # ── Plot 1: Loss tangent vs temperature ──
    fig, ax = plt.subplots(figsize=(8, 5))
    T_arr = np.linspace(50, 270, 200)

    for label, freq in bands.items():
        tand = loss_tangent_ice(freq, T_arr)
        ax.semilogy(T_arr, tand, label=label, linewidth=2)

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Loss tangent (tan δ)", fontsize=12)
    ax.set_title("Ice Loss Tangent vs. Temperature (Pure Ice)", fontsize=13)
    ax.legend(fontsize=10)
    ax.axvspan(80, 130, alpha=0.15, color="blue", label="Europa range")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 270)
    fig.tight_layout()
    fig.savefig(str(output_dir / "loss_tangent_vs_temperature.png"), dpi=150)
    print(f"  Saved: {output_dir / 'loss_tangent_vs_temperature.png'}")
    plt.close(fig)

    # ── Plot 2: Penetration depth vs porosity ──
    fig, ax = plt.subplots(figsize=(8, 5))
    porosity_arr = np.linspace(0.0, 0.8, 100)

    for label, freq in bands.items():
        L_arr = [penetration_depth(freq, 100.0, p) for p in porosity_arr]
        ax.plot(porosity_arr * 100, np.array(L_arr) * 100, label=label, linewidth=2)

    ax.set_xlabel("Porosity (%)", fontsize=12)
    ax.set_ylabel("Penetration Depth (cm)", fontsize=12)
    ax.set_title("Penetration Depth vs. Porosity at T = 100 K", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Porosity ranges typically inferred for icy-satellite regolith.
    ax.axvspan(40, 50, alpha=0.15, color="red", label="40-50%")
    ax.axvspan(50, 75, alpha=0.15, color="green", label="50-75%")
    fig.tight_layout()
    fig.savefig(str(output_dir / "penetration_depth_vs_porosity.png"), dpi=150)
    print(f"  Saved: {output_dir / 'penetration_depth_vs_porosity.png'}")
    plt.close(fig)

    # ── Plot 3: Absorption coefficient vs frequency ──
    fig, ax = plt.subplots(figsize=(8, 5))
    freq_arr = np.linspace(50, 400, 200)
    temps = [80, 100, 120, 150, 200]

    for T in temps:
        kappa = absorption_coefficient(freq_arr, T, porosity=0.5)
        ax.semilogy(freq_arr, kappa, label=f"T = {T} K", linewidth=2)

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("Absorption Coefficient (m⁻¹)", fontsize=12)
    ax.set_title("Absorption Coefficient (50% porosity, Maxwell-Garnett)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Mark the reference frequencies
    for label, freq in bands.items():
        ax.axvline(freq, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(output_dir / "absorption_vs_frequency.png"), dpi=150)
    print(f"  Saved: {output_dir / 'absorption_vs_frequency.png'}")
    plt.close(fig)

    # ── Plot 4: Maxwell-Garnett vs Bruggeman comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    porosity_arr = np.linspace(0.01, 0.95, 100)
    freq = 233.0
    T = 100.0

    for model, ls in [("maxwell_garnett", "-"), ("bruggeman", "--")]:
        eps_arr = [porous_ice_permittivity(freq, T, p, model) for p in porosity_arr]
        eps_arr = np.array(eps_arr)
        axes[0].plot(porosity_arr * 100, np.real(eps_arr), ls, label=model, linewidth=2)
        axes[1].semilogy(porosity_arr * 100, np.imag(eps_arr), ls, label=model, linewidth=2)

    axes[0].set_xlabel("Porosity (%)")
    axes[0].set_ylabel("ε' (real)")
    axes[0].set_title("Real Permittivity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Porosity (%)")
    axes[1].set_ylabel("ε'' (imaginary)")
    axes[1].set_title("Imaginary Permittivity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Mixing Models at {freq:.0f} GHz, {T:.0f} K", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_dir / "mixing_model_comparison.png"), dpi=150)
    print(f"  Saved: {output_dir / 'mixing_model_comparison.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Ice dielectric properties at millimeter wavelengths"
    )
    parser.add_argument("--freq", type=float, default=233.0,
                        help="Frequency in GHz (default: 233)")
    parser.add_argument("--temp", type=float, default=100.0,
                        help="Temperature in K (default: 100)")
    parser.add_argument("--porosity", type=float, default=0.0,
                        help="Porosity, 0-1 (default: 0 = solid ice)")
    parser.add_argument("--mixing", choices=["maxwell_garnett", "bruggeman"],
                        default="maxwell_garnett",
                        help="Porosity mixing model")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--scan-porosity", action="store_true",
                        help="Print table of properties vs. porosity")
    args = parser.parse_args()

    if args.plot:
        make_diagnostic_plots()
        return

    if args.scan_porosity:
        print(f"\n  Dielectric properties vs. porosity at "
              f"{args.freq:.1f} GHz, {args.temp:.0f} K ({args.mixing})")
        print(f"  {'Porosity':>10} {'ε_real':>10} {'ε_imag':>12} {'tan δ':>12} "
              f"{'κ (m⁻¹)':>10} {'L (cm)':>10}")
        print(f"  {'─'*70}")
        for p in np.arange(0, 0.85, 0.05):
            eps = porous_ice_permittivity(args.freq, args.temp, p, args.mixing)
            kappa = absorption_coefficient(args.freq, args.temp, p, args.mixing)
            L = penetration_depth(args.freq, args.temp, p, args.mixing)
            tand = np.imag(eps) / np.real(eps)
            print(f"  {p*100:>9.0f}% {np.real(eps):>10.4f} {np.imag(eps):>12.4e} "
                  f"{tand:>12.4e} {kappa:>10.4f} {L*100:>10.2f}")
        return

    print_dielectric_summary(args.freq, args.temp, args.porosity, args.mixing)

    # Also show the three reference frequencies
    print("\n  Reference frequencies at current settings:")
    for freq in REFERENCE_FREQUENCIES_GHZ:
        kappa = absorption_coefficient(freq, args.temp, args.porosity, args.mixing)
        L = penetration_depth(freq, args.temp, args.porosity, args.mixing)
        print(f"    {freq:>5.1f} GHz: κ = {kappa:.4f} m⁻¹, "
              f"L = {L*100:.2f} cm")


if __name__ == "__main__":
    main()
