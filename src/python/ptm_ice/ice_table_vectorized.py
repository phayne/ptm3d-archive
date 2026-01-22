"""Vectorized ice table depth solver (NumPy prototype)

Implements a high-throughput Newton iteration to compute ice stability depth
for large arrays of (Tavg, Tmax) values. Provides chunked processing to scale
up to ~1e7 faces without exhausting memory.

Physics recap (simplified from notebook):
  - Temperature profile vs depth: T(z) = Tavg + dT * exp(-z/zs) + (Q/k) * z
  - Saturation vapor pressure: pv = pt * exp(-L*mu/kb * (1/T - 1/Tt))
  - f(z) root: f = 2*z*J0/(mu*l) - A*pv*T**-0.5, solve f=0 for z
  - Newton step: z_{n+1} = z_n - f/f'

We vectorize f and f' (via F1/F2) and iterate under convergence mask.

Primary entry point:
  compute_ice_depths_vectorized(tavg, tmax, J0, delta=1e-4, z0=0.0, l=100e-6,
                                zs=0.1, Q=0.02, k=1e-3, max_iter=200,
                                chunk_size=None, return_iterations=False,
                                dtype=np.float64)

Returns an array of depths (same shape as inputs), and optionally iteration counts.
Values where Tavg > Tmax are set to NaN.

Design notes:
    - Provides both pure NumPy and optional Numba-parallel backends.
    - Minimizes temporaries by reusing arrays; careful with dtype.
  - chunk_size controls streaming over large arrays; if None processes all.
  - For 1e7 elements with float64 (8 bytes) per array, working set kept small by
    allocating a few arrays per chunk (~ < 5 * chunk_size * itemsize).
"""
from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Optional, Literal

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

# Physical constants (module-level for reuse)
_mu = 2.99e-26          # kg (water molecule mass)
_kb = 1.38e-23          # J/K Boltzmann
_Tt = 273.16            # K triple-point temp
_pt = 611.657           # Pa triple-point pressure
_L = 2.834e6            # J/kg enthalpy of sublimation
_A = 6.21e23            # Pre-factor (2*pi*kb*mu)**-0.5

__all__ = [
    'compute_ice_depths_vectorized',
    'water_psat_vectorized',
    'compute_ice_depths_numba',
    'compute_ice_depths',
]

def water_psat_vectorized(T: np.ndarray) -> np.ndarray:
    """Vectorized saturation vapor pressure [Pa]."""
    # pv = pt * exp(-L*mu/kb * (1/T - 1/Tt))
    return _pt * np.exp(-_L * _mu / _kb * (1.0 / T - 1.0 / _Tt))

def _temperature_profile(Tavg: np.ndarray, dT: np.ndarray, z: np.ndarray, zs: float, Q: float, k: float) -> np.ndarray:
    # T(z) = Tavg + dT * exp(-z/zs) + (Q/k)*z
    return Tavg + dT * np.exp(-z / zs) + (Q / k) * z

def _F1(J0: float, T: np.ndarray, z: np.ndarray, l: float, pv: np.ndarray) -> np.ndarray:
    # f = 2*z*J0/(mu*l) - A*pv*T**-0.5
    return 2.0 * z * J0 / (_mu * l) - _A * pv * T**-0.5

def _F4(z: np.ndarray, dT: np.ndarray, zs: float, Q: float, k: float) -> np.ndarray:
    # dTdz = -dT/zs * exp(-z/zs) + Q/k
    return -dT / zs * np.exp(-z / zs) + Q / k

def _F3(T: np.ndarray, z: np.ndarray, dT: np.ndarray, pv: np.ndarray, zs: float, Q: float, k: float) -> np.ndarray:
    # dpdz = L*mu/kb * T**-2 * pv * F4(z)
    return _L * _mu / _kb * T**-2 * pv * _F4(z, dT, zs, Q, k)

def _F2(J0: float, T: np.ndarray, z: np.ndarray, l: float, dT: np.ndarray, pv: np.ndarray, zs: float, Q: float, k: float) -> np.ndarray:
    # fp = 2*J0/(mu*l) - A*T**-0.5*F3(T,z,dT,pv,...) - 0.5*pv*A*T**-1.5*F4(z,...)
    return 2.0 * J0 / (_mu * l) - _A * T**-0.5 * _F3(T, z, dT, pv, zs, Q, k) - 0.5 * pv * _A * T**-1.5 * _F4(z, dT, zs, Q, k)

def compute_ice_depths_vectorized(
    tavg: np.ndarray,
    tmax: np.ndarray,
    J0: float,
    delta: float = 1e-4,
    z0: float = 0.0,
    l: float = 100e-6,
    zs: float = 0.1,
    Q: float = 0.02,
    k: float = 1e-3,
    max_iter: int = 200,
    chunk_size: Optional[int] = None,
    return_iterations: bool = False,
    dtype = np.float64,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute ice stability depth for arrays of mean & max temperature.

    Parameters
    ----------
    tavg : np.ndarray
        Mean surface temperature [K].
    tmax : np.ndarray
        Maximum surface temperature [K]. Same shape as tavg.
    J0 : float
        Vapor flux limit [kg m^-2 s^-1].
    delta : float
        Convergence tolerance on |dz| [m].
    z0 : float
        Initial depth guess [m].
    l : float
        Molecular mean free path [m].
    zs : float
        Thermal skin depth [m].
    Q : float
        Geothermal heat flow [W m^-2].
    k : float
        Thermal conductivity [W m^-1 K^-1].
    max_iter : int
        Maximum Newton iterations.
    chunk_size : int or None
        If set, process in chunks of this size (flattened order) to limit memory.
    return_iterations : bool
        If True, also return iteration counts array.
    dtype : numpy dtype
        Working dtype (np.float64 or np.float32).

    Returns
    -------
    depths : np.ndarray
        Ice stability depths (NaN where no solution).
    iters : np.ndarray, optional
        Iteration counts if return_iterations is True.
    """
    if tavg.shape != tmax.shape:
        raise ValueError("tavg and tmax must share shape")

    flat_tavg = np.asarray(tavg, dtype=dtype).ravel()
    flat_tmax = np.asarray(tmax, dtype=dtype).ravel()
    N = flat_tavg.size

    depths = np.empty(N, dtype=dtype)
    iters = np.zeros(N, dtype=np.int32) if return_iterations else None

    # Determine chunking
    if chunk_size is None or chunk_size <= 0 or chunk_size >= N:
        chunk_size = N

    # Simple physical guardrails
    z_min = 0.0
    z_max = 5.0  # meters; adjustable upper bound for stability

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Tavg_chunk = flat_tavg[start:end]
        Tmax_chunk = flat_tmax[start:end]

        # Initialize z
        z = np.full(Tavg_chunk.shape, z0, dtype=dtype)
        dT = Tmax_chunk - Tavg_chunk

        # Mask where solution possible (Tavg <= Tmax)
        active = Tavg_chunk <= Tmax_chunk
        # Initialize arrays for iteration
        converged = ~active  # Already converged if inactive (will set NaN)

        # Newton iteration
        for it in range(max_iter):
            if not np.any(active & ~converged):
                break
            idx = active & ~converged
            # Compute temperature at current depth
            Tz = _temperature_profile(Tavg_chunk[idx], dT[idx], z[idx], zs, Q, k)
            pv = water_psat_vectorized(Tz)
            f = _F1(J0, Tz, z[idx], l, pv)
            fp = _F2(J0, Tz, z[idx], l, dT[idx], pv, zs, Q, k)
            # Newton step
            # Protect against division by extremely small fp
            fp_safe = np.where(np.abs(fp) < 1e-30, np.sign(fp) * 1e-30, fp)
            dz = -f / fp_safe
            # Step damping to avoid wild excursions
            dz = np.clip(dz, -0.5, 0.5)
            z[idx] += dz
            # Enforce bounds
            np.clip(z[idx], z_min, z_max, out=z[idx])
            # Convergence check
            newly_converged = np.abs(dz) <= delta
            converged_idx = np.where(idx)[0][newly_converged]
            converged[converged_idx] = True
            if return_iterations:
                iters[start:end][idx] = it + 1
        # Mark any non-converged active cells as NaN (did not reach tolerance)
        not_converged_active = active & ~converged
        if np.any(not_converged_active):
            z[not_converged_active] = np.nan
        # Assign results
        z[~active] = np.nan
        depths[start:end] = z

    return (depths.reshape(tavg.shape), iters.reshape(tavg.shape) if return_iterations else None)

# ===================== Numba Backend =====================
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _compute_chunk_numba(Tavg, Tmax, J0, delta, z0, l, zs, Q, k, max_iter):
        N = Tavg.size
        depths = np.empty(N, dtype=np.float64)
        iters = np.zeros(N, dtype=np.int32)
        z_min = 0.0
        z_max = 5.0
        for i in prange(N):
            Tavg_i = Tavg[i]
            Tmax_i = Tmax[i]
            if Tavg_i > Tmax_i:
                depths[i] = np.nan
                continue
            z = z0
            dT = Tmax_i - Tavg_i
            converged = False
            for it in range(max_iter):
                Tz = Tavg_i + dT * math.exp(-z / zs) + (Q / k) * z
                pv = _pt * math.exp(-_L * _mu / _kb * (1.0 / Tz - 1.0 / _Tt))
                f = 2.0 * z * J0 / (_mu * l) - _A * pv * Tz ** -0.5
                dTdz = -dT / zs * math.exp(-z / zs) + Q / k
                dpdz = _L * _mu / _kb * Tz ** -2 * pv * dTdz
                fp = 2.0 * J0 / (_mu * l) - _A * Tz ** -0.5 * dpdz - 0.5 * pv * _A * Tz ** -1.5 * dTdz
                if abs(fp) < 1e-30:
                    fp = math.copysign(1e-30, fp)
                dz = -f / fp
                if dz > 0.5: dz = 0.5
                if dz < -0.5: dz = -0.5
                z += dz
                if z < z_min: z = z_min
                if z > z_max: z = z_max
                if abs(dz) <= delta:
                    depths[i] = z
                    iters[i] = it + 1
                    converged = True
                    break
            if not converged:
                depths[i] = np.nan
        return depths, iters

def compute_ice_depths_numba(
    tavg: np.ndarray,
    tmax: np.ndarray,
    J0: float,
    delta: float = 1e-4,
    z0: float = 0.0,
    l: float = 100e-6,
    zs: float = 0.1,
    Q: float = 0.02,
    k: float = 1e-3,
    max_iter: int = 200,
    chunk_size: Optional[int] = None,
    return_iterations: bool = False,
    dtype = np.float64,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Numba-parallel backend for ice depth computation.

    Falls back to vectorized NumPy implementation if numba unavailable.
    Interface mirrors compute_ice_depths_vectorized.
    """
    if not NUMBA_AVAILABLE:
        return compute_ice_depths_vectorized(
            tavg, tmax, J0, delta=delta, z0=z0, l=l, zs=zs, Q=Q, k=k,
            max_iter=max_iter, chunk_size=chunk_size,
            return_iterations=return_iterations, dtype=dtype)

    if tavg.shape != tmax.shape:
        raise ValueError("tavg and tmax must share shape")
    flat_tavg = np.asarray(tavg, dtype=np.float64).ravel()
    flat_tmax = np.asarray(tmax, dtype=np.float64).ravel()
    N = flat_tavg.size
    if chunk_size is None or chunk_size <= 0 or chunk_size >= N:
        chunk_size = N
    depths = np.empty(N, dtype=np.float64)
    iters_full = np.zeros(N, dtype=np.int32) if return_iterations else None
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        d_chunk, it_chunk = _compute_chunk_numba(
            flat_tavg[start:end], flat_tmax[start:end], J0, delta, z0, l, zs, Q, k, max_iter)
        depths[start:end] = d_chunk
        if return_iterations:
            iters_full[start:end] = it_chunk
    depths = depths.reshape(tavg.shape).astype(dtype, copy=False)
    if return_iterations:
        return depths, iters_full.reshape(tavg.shape)
    return depths, None

def compute_ice_depths(
    tavg: np.ndarray,
    tmax: np.ndarray,
    J0: float,
    backend: Literal['numpy','numba'] = 'numpy',
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatcher to chosen backend.

    backend='numpy'  -> compute_ice_depths_vectorized
    backend='numba'  -> compute_ice_depths_numba (falls back if unavailable)
    """
    if backend == 'numpy':
        return compute_ice_depths_vectorized(tavg, tmax, J0, **kwargs)
    elif backend == 'numba':
        return compute_ice_depths_numba(tavg, tmax, J0, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

# Simple self-test when run directly
if __name__ == '__main__':
    # Small synthetic test
    Tavg = np.array([90.0, 100.0, 120.0])
    Tmax = np.array([120.0, 140.0, 110.0])  # last should produce NaN
    depths, iters = compute_ice_depths_vectorized(Tavg, Tmax, J0=3e-14, return_iterations=True)
    print('Depths:', depths)
    print('Iterations:', iters)
