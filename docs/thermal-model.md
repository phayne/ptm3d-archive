# Thermal Model Documentation

This document describes the physics and algorithms used in PTM3D's thermal model.

## Overview

The thermal model solves the 1D heat conduction equation with temperature-dependent material properties and radiative boundary conditions.

## Heat Equation

The governing equation is:

```
ρ cp(T) ∂T/∂t = ∂/∂z [k(T) ∂T/∂z]
```

Where:
- `ρ` is density (kg/m³)
- `cp(T)` is temperature-dependent heat capacity (J/kg/K)
- `k(T)` is temperature-dependent thermal conductivity (W/m/K)
- `T` is temperature (K)
- `z` is depth (m)
- `t` is time (s)

## Boundary Conditions

### Surface (z = 0)
Radiative equilibrium:
```
k ∂T/∂z|₀ = ε σ T⁴ - (1-A) Q_solar - Q_indirect
```

### Bottom (z = z_max)
Fixed heat flux:
```
k ∂T/∂z|_bottom = Q_geo
```

## Numerical Method

### Crank-Nicolson Scheme
- Second-order accurate in time and space
- Unconditionally stable
- Uses Thomas algorithm (tridiagonal matrix solver)

### Adaptive Depth Grid
- Logarithmic spacing near surface
- Coarser grid at depth
- Configurable parameters: m (layers per skin depth), n (growth factor), b (skin depths to bottom)

## Temperature-Dependent Properties

### Thermal Conductivity
```
k(T) = k_c + k_r · T³
```
Where `k_c` is contact conductivity and `k_r` is radiative conductivity coefficient.

### Heat Capacity
Polynomial fit:
```
cp(T) = c₀ + c₁T + c₂T² + c₃T³
```
Evaluated using Horner's method for efficiency.

## Spectral Equilibration

FFT-based analytical solver for rapid equilibration:
1. Decompose surface flux into Fourier harmonics
2. Solve heat equation analytically for each harmonic
3. Iterate to update linearized radiative BC
4. Handoff converged profile to transient solver

## References

- Hayne et al. (2017) - Grid optimization methodology
- Spencer et al. (1989) - Thermal inertia formulation
