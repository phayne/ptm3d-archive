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
The total thermal conductivity includes both solid conduction and radiative transfer:
```
k(T) = k_c · (1 + χ · (T/350)³)
```
Where:
- `k_c` is the contact (solid) conductivity at the reference temperature
- `χ` is the radiative conductivity parameter (default: 2.7 from Mitchell & de Pater 1994)
- The T³ term accounts for radiative heat transfer through pore spaces

### Heat Capacity
Polynomial fit:
```
cp(T) = c₀ + c₁T + c₂T² + c₃T³
```
Evaluated using Horner's method for efficiency.

## Thermal Equilibration

PTM3D uses diurnal time-stepping to achieve thermal equilibration:
1. Run multiple complete diurnal cycles (configurable via `--equilibration-orbits`)
2. Track temperature changes between cycles
3. Continue until temperatures converge to periodic steady state
4. Use converged profile as initial condition for science simulation

The number of equilibration cycles depends on the thermal inertia: low inertia surfaces (regolith) equilibrate quickly (~5-10 cycles), while high inertia surfaces may require more cycles.

## Depth-Dependent Properties

Density varies with depth following an exponential profile:
```
ρ(z) = ρ_s + (ρ_d - ρ_s) · exp(-z/H)
```
Where:
- `ρ_s` is surface density
- `ρ_d` is deep density
- `H` is the e-folding depth scale

Thermal conductivity also varies with depth through the density dependence:
```
k_c(z) = k_c0 · (ρ(z) / ρ_s)
```

## References

- Hayne et al. (2017) JGR Planets - Grid optimization methodology and lunar thermophysical properties
