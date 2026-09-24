"""Helpers for loading per-face mass-balance data from a PTM3D exosphere run.

The ptm_exosphere_main run produces per-epoch binary files (one float32 per
face) for every tracked variable. For mass-balance analysis we want the
*final-epoch* cumulative totals divided by the elapsed simulation time —
this gives the average rate per face over the simulated geological window.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import loaders  # noqa: E402

SECONDS_PER_YEAR = 365.25 * 86400.0


def find_last_epoch(run_dir: Path | str, subdir: str) -> int:
    """Highest epoch index among <subdir>/<subdir>_NNNN.bin files."""
    files = sorted((Path(run_dir) / subdir).glob(f"{subdir}_*.bin"), key=lambda p: int(str(p).rsplit("_", 1)[1][:-4]))
    if not files:
        raise FileNotFoundError(f"No files in {Path(run_dir)/subdir}/")
    m = re.search(r"_(\d+)\.bin$", files[-1].name)
    return int(m.group(1))


def load_evolution_total_years(run_dir: Path | str) -> float:
    """Parse evolution_summary.txt for total simulation time in years."""
    path = Path(run_dir) / "evolution_summary.txt"
    if not path.exists():
        return float("nan")
    with open(path) as f:
        for line in f:
            if "Total time:" in line or "Elapsed time:" in line:
                m = re.search(r"([\d.eE+-]+)\s*years", line)
                if m:
                    return float(m.group(1))
    return float("nan")


def load_mass_balance(run_dir: Path | str, epoch: int = -1):
    """Load the headline per-face mass-balance quantities at a given epoch.

    Returns a dict with:
      lat, lon       — face center coordinates (deg)
      total_years    — simulation duration up to this epoch
      cumulative_sublimated, cumulative_deposited, cumulative_sputtered  [kg/m²]
      net_loss      = sublimated + sputtered − deposited [kg/m²]
      sublimation_rate, deposition_rate, sputtering_rate, net_loss_rate  [kg/m²/yr]
      bond_albedo, T_max, T_mean, T_min, dose_rate_eV_mol_s, phi_C, nonice
    """
    run = Path(run_dir)
    lat, lon = loaders.load_mesh(run)
    nf = lat.size

    if epoch < 0:
        epoch = find_last_epoch(run, "cumulative_sublimated_kg_m2")
    e = f"{epoch:04d}"

    sub  = loaders.load_per_face_field(run, f"cumulative_sublimated_kg_m2/cumulative_sublimated_kg_m2_{e}.bin", nf)
    dep  = loaders.load_per_face_field(run, f"cumulative_deposited_kg_m2/cumulative_deposited_kg_m2_{e}.bin", nf)
    sput = loaders.load_per_face_field(run, f"cumulative_sputtered_kg_m2/cumulative_sputtered_kg_m2_{e}.bin", nf)
    Tmax = loaders.load_per_face_field(run, f"temperature_max_K/temperature_max_K_{e}.bin", nf)
    Tmean= loaders.load_per_face_field(run, f"temperature_mean_K/temperature_mean_K_{e}.bin", nf)
    Tmin = loaders.load_per_face_field(run, f"temperature_min_K/temperature_min_K_{e}.bin", nf)
    # bond_albedo lives in the thermal+exosphere parent run (the cryst-only
    # runs were launched without --albedo-coupling so don't emit it).
    # Bond albedo is geographically fixed (from the FITS map), so we always
    # source it from the canonical albedo-map run.
    import struct
    candidates = [
        run / f"bond_albedo/bond_albedo_{e}.bin",
        run.parent / run.name.replace("_cryst", "")
            / "exosphere/self-consistent/bond_albedo/bond_albedo_0000.bin",
        Path("output/europa_ico6_albedo-map/exosphere/self-consistent/"
             "bond_albedo/bond_albedo_0000.bin"),
    ]
    alb = None
    for c in candidates:
        if c.exists():
            with open(c, "rb") as fh:
                n_in = struct.unpack("<I", fh.read(4))[0]
                alb = np.frombuffer(fh.read(n_in * 4), dtype="<f4").astype(np.float64).copy()
            break
    if alb is None:
        alb = np.full(nf, np.nan)
    dose = loaders.load_per_face_field(run, f"radiation_dose_rate_eV_mol_s/radiation_dose_rate_eV_mol_s_{e}.bin", nf)
    phi  = loaders.load_per_face_field(run, f"crystalline_fraction/crystalline_fraction_{e}.bin", nf)
    nonice = loaders.load_per_face_field(run, f"nonice_concentration/nonice_concentration_{e}.bin", nf)

    total_yr = load_evolution_total_years(run)
    if not np.isfinite(total_yr) or total_yr <= 0:
        # Fallback: 100 cycles/epoch × 3.5512 days × (epoch+1) cycles
        total_yr = (epoch + 1) * 100 * 3.5512 / 365.25

    net_loss = sub + sput - dep  # kg/m^2 (positive = net ablation)
    return dict(
        lat=lat, lon=lon, total_years=float(total_yr), epoch=epoch,
        cumulative_sublimated=sub, cumulative_deposited=dep,
        cumulative_sputtered=sput,
        sublimation_rate=sub / total_yr,
        deposition_rate=dep / total_yr,
        sputtering_rate=sput / total_yr,
        net_loss=net_loss,
        net_loss_rate=net_loss / total_yr,
        bond_albedo=alb,
        T_max=Tmax, T_mean=Tmean, T_min=Tmin,
        dose_rate=dose,
        phi_C=phi,
        nonice=nonice,
    )


def zonal_mean(lat: np.ndarray, values: np.ndarray,
               lat_bins: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` in latitude bins. Returns (bin_centers_deg, means)."""
    if lat_bins is None:
        lat_bins = np.linspace(-87.5, 87.5, 36)  # 5-deg bins
    centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    means = np.full(centers.size, np.nan)
    for i, (lo, hi) in enumerate(zip(lat_bins[:-1], lat_bins[1:])):
        m = (lat >= lo) & (lat < hi)
        if m.any():
            # Weight by cos(lat) for zonal area-mean
            w = np.cos(np.deg2rad(lat[m]))
            means[i] = float(np.average(values[m], weights=w))
    return centers, means


def longitudinal_mean(lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
                      lat_band: tuple[float, float] = (-30.0, 30.0),
                      lon_bins: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` in longitude bins, restricted to a latitude band."""
    if lon_bins is None:
        lon_bins = np.linspace(0.0, 360.0, 37)
    centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    band = (lat >= lat_band[0]) & (lat <= lat_band[1])
    means = np.full(centers.size, np.nan)
    for i, (lo, hi) in enumerate(zip(lon_bins[:-1], lon_bins[1:])):
        m = band & (lon >= lo) & (lon < hi)
        if m.any():
            w = np.cos(np.deg2rad(lat[m]))
            means[i] = float(np.average(values[m], weights=w))
    return centers, means


def amorphization_rate_total(mb: dict, e_amorph_eV: float = 3.0) -> np.ndarray:
    """Combined amorphization rate per face [s^-1]:
       k_amorph_radiation = D_dot / E_amorph    [Eq. D29]
       Plus the *equivalent* depositional dilution rate ≈ deposition_flux /
       (rho_ice * skin_depth) when the deposit equals one skin-depth ice
       reservoir per characteristic time. The C++ exosphere uses a
       mass-weighted dilution (Eq D39); here we report the equivalent
       linear rate so the two pathways can be compared on the same axis."""
    rho_ice = 917.0  # kg/m^3
    skin_depth = 0.05  # m (config default)
    reservoir = rho_ice * skin_depth  # kg/m^2 of ice in the active layer

    k_rad = mb["dose_rate"] / e_amorph_eV
    # Use the deposition_rate (kg/m^2/yr) -> per second
    dep_per_s = mb["deposition_rate"] / SECONDS_PER_YEAR
    k_dep_dilution = dep_per_s / reservoir
    return k_rad + k_dep_dilution


def crystallization_rate_eff(T_K: np.ndarray, porosity: float | np.ndarray = 0.0):
    """Mitchell 2017 Avrami effective rate (matches crystallization.k_cryst)."""
    from crystallization import k_cryst
    return k_cryst(T_K, porosity)
