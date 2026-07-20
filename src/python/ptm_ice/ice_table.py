"""Ice table solver utilities

This module provides a lightweight, dependency-free port of key functions
from `diviner_ice-table_solver-v2.ipynb` into a reusable API.

Public API:
- IceTable: dataclass to hold configuration parameters.
- compute_ice_table(surface_temp_timeseries, params) -> dict with results

Notes:
- This is a self-contained, numpy-free implementation for portability. If you
  prefer vectorized operations use the numpy-based variant.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, List
import math

@dataclass
class IceTable:
    """Configuration for the ice table solver.

    Attributes:
    - thermal_inertia: float, in J m^-2 K^-1 s^-1/2 (optional naming preserved)
    - albedo: float, unitless
    - emissivity: float, unitless
    - diffusivity: float, m^2/s (if needed)
    - density: float, kg/m^3
    - specific_heat: float, J/(kg K)
    - latent_heat: float, J/kg
    - time_step: float, seconds for internal solver (default: 3600s)
    """
    thermal_inertia: float = 50.0
    albedo: float = 0.2
    emissivity: float = 0.95
    diffusivity: float = 1e-6
    density: float = 1000.0
    specific_heat: float = 1000.0
    latent_heat: float = 2.5e6
    time_step: float = 3600.0


def _blackbody_flux(temp_k: float, emissivity: float = 1.0) -> float:
    sigma = 5.670374419e-8
    return emissivity * sigma * (temp_k ** 4)


def _kelvin(t_c: float) -> float:
    return t_c + 273.15


def compute_ice_table(surface_temps_c: Sequence[float], params: IceTable) -> Dict[str, object]:
    """Compute a simple ice table/time-integrated mass balance from surface temperatures.

    This is a simplified port of the notebook's higher-fidelity solver. It
    approximates net sublimation/condensation by comparing emitted thermal
    flux to a reference and integrating a naive mass change using latent heat.

    Args:
      surface_temps_c: sequence of surface temperatures in Celsius sampled
          at regular intervals (we assume uniform spacing in time).
      params: IceTable dataclass with physical parameters.

    Returns:
      A dict with keys:
        - 'times_s': list of times in seconds (relative, starting at 0)
        - 'surface_temps_c': input temperatures as list
        - 'mass_change_kg_m2': cumulative mass change (kg/m^2)
        - 'rates_kg_m2_s': list of instantaneous mass change rates

    Notes:
      - This routine is intentionally simple. It's useful for unit tests and
        as a building block for a more advanced solver.
    """
    n = len(surface_temps_c)
    if n == 0:
        return {'times_s': [], 'surface_temps_c': [], 'mass_change_kg_m2': 0.0, 'rates_kg_m2_s': []}

    dt = params.time_step
    times = [i * dt for i in range(n)]
    rates: List[float] = []
    mass_change = 0.0

    for t_c in surface_temps_c:
        t_k = _kelvin(t_c)
        emitted = _blackbody_flux(t_k, params.emissivity)
        # approximate incoming radiative flux from a reference: use Stefan-Boltzmann at 200 K
        ref_flux = _blackbody_flux(200.0, params.emissivity)
        net_flux = ref_flux - emitted  # positive -> net gain (condensation), negative -> loss (sublimation)

        # Convert flux to mass rate via latent heat. flux (W/m2) -> kg/m2/s
        rate = net_flux / params.latent_heat
        rates.append(rate)
        mass_change += rate * dt

    return {
        'times_s': times,
        'surface_temps_c': list(surface_temps_c),
        'mass_change_kg_m2': mass_change,
        'rates_kg_m2_s': rates,
    }


class SimpleSolver:
    """A small stateful solver wrapper around `compute_ice_table`.

    Provides incremental update ability similar to a class-based solver in the
    original notebook.
    """
    def __init__(self, params: IceTable):
        self.params = params
        self.times: List[float] = []
        self.temps: List[float] = []
        self.mass_change: float = 0.0

    def step(self, temp_c: float) -> Tuple[float, float]:
        """Advance solver by one internal timestep with given surface temp.

        Returns (rate, cumulative_mass_change)
        """
        res = compute_ice_table([temp_c], self.params)
        # single step results
        rate = res['rates_kg_m2_s'][0]
        self.times.append(0.0 if not self.times else self.times[-1] + self.params.time_step)
        self.temps.append(temp_c)
        self.mass_change += rate * self.params.time_step
        return rate, self.mass_change


def _example_usage():
    params = IceTable()
    temps = [ -150.0 + 10.0 * math.sin(2 * math.pi * i / 24.0) for i in range(24) ]
    res = compute_ice_table(temps, params)
    print('mass change (kg/m2):', res['mass_change_kg_m2'])


if __name__ == '__main__':
    _example_usage()
