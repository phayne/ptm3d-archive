import numpy as np
from ptm_ice.ice_table_vectorized import compute_ice_depths_vectorized

# Reference scalar implementation (minimal) for validation
# Direct port of notebook logic for one (Tavg, Tmax) pair
from math import exp, sqrt, nan

def ice_table_depth_scalar(J0, Tavg, Tmax, delta, z0=0.0, l=100e-6, zs=0.1, Q=0.02, k=1e-3, max_iter=200):
    mu = 2.99e-26
    kb = 1.38e-23
    Tt = 273.16
    pt = 611.657
    L = 2.834e6
    A = 6.21e23
    if Tavg > Tmax:
        return nan
    z = z0
    dT = Tmax - Tavg
    for _ in range(max_iter):
        Tz = Tavg + dT * exp(-z / zs) + (Q / k) * z
        pv = pt * exp(-L * mu / kb * (1.0 / Tz - 1.0 / Tt))
        f = 2.0 * z * J0 / (mu * l) - A * pv * Tz ** -0.5
        # Derivatives
        dTdz = -dT / zs * exp(-z / zs) + Q / k
        dpdz = L * mu / kb * Tz ** -2 * pv * dTdz
        fp = 2.0 * J0 / (mu * l) - A * Tz ** -0.5 * dpdz - 0.5 * pv * A * Tz ** -1.5 * dTdz
        dz = -f / fp
        z += dz
        if abs(dz) <= delta:
            break
    return z

def test_vectorized_matches_scalar_small():
    J0 = 3e-14
    delta = 1e-4
    Tavg = np.array([90.0, 100.0, 120.0])
    Tmax = np.array([120.0, 140.0, 110.0])
    depths_vec, _ = compute_ice_depths_vectorized(Tavg, Tmax, J0, delta=delta, return_iterations=True)
    depths_ref = np.array([ice_table_depth_scalar(J0, Tavg[i], Tmax[i], delta) for i in range(Tavg.size)])
    # Allow small tolerance differences
    for dv, dr in zip(depths_vec, depths_ref):
        if np.isnan(dr):
            assert np.isnan(dv)
        else:
            assert np.isclose(dv, dr, rtol=1e-3, atol=5e-4)

def test_chunking_identity():
    rng = np.random.default_rng(42)
    N = 5000
    Tavg = rng.uniform(80, 140, size=N)
    Tmax = Tavg + rng.uniform(5, 40, size=N)  # ensure Tmax > Tavg
    J0 = 3e-14
    d_full, _ = compute_ice_depths_vectorized(Tavg, Tmax, J0, chunk_size=None)
    d_chunk, _ = compute_ice_depths_vectorized(Tavg, Tmax, J0, chunk_size=1000)
    assert np.allclose(d_full, d_chunk, rtol=1e-6, atol=1e-6)

def test_nan_where_no_solution():
    Tavg = np.array([150.0])
    Tmax = np.array([140.0])  # Mean > Max -> invalid
    J0 = 3e-14
    d, _ = compute_ice_depths_vectorized(Tavg, Tmax, J0)
    assert np.isnan(d[0])
