# Thermal Model Documentation

This document describes the physics and algorithms used in PTM3D's thermal model.

## Overview

The thermal model solves the 1D heat conduction equation with temperature-dependent material properties and radiative boundary conditions.

## Heat Equation

The governing equation is:

$$\rho \cdot c_p(T) \frac{\partial T}{\partial t} = \frac{\partial}{\partial z} \left[ k(T) \frac{\partial T}{\partial z} \right]$$

Where:
- $\rho$ is density (kg/m³)
- $c_p(T)$ is temperature-dependent heat capacity (J/kg/K)
- $k(T)$ is temperature-dependent thermal conductivity (W/m/K)
- $T$ is temperature (K)
- $z$ is depth (m)
- $t$ is time (s)

## Boundary Conditions

### Surface ($z = 0$)
Radiative equilibrium:

$$k \left. \frac{\partial T}{\partial z} \right|_0 = \varepsilon \sigma T^4 - (1-A) Q_{\text{solar}} - Q_{\text{indirect}}$$

### Bottom ($z = z_{\max}$)
Fixed heat flux:

$$k \left. \frac{\partial T}{\partial z} \right|_{\text{bottom}} = Q_{\text{geo}}$$

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

$$k(T) = k_c \cdot \left(1 + \chi \cdot \left(\frac{T}{350}\right)^3\right)$$

Where:
- $k_c$ is the contact (solid) conductivity at the reference temperature
- $\chi$ is the radiative conductivity parameter (default: 2.7 from Mitchell & de Pater 1994)
- The $T^3$ term accounts for radiative heat transfer through pore spaces

### Heat Capacity
Polynomial fit:

$$c_p(T) = c_0 + c_1 T + c_2 T^2 + c_3 T^3$$

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

$$\rho(z) = \rho_s + (\rho_d - \rho_s) \cdot \exp(-z/H)$$

Where:
- $\rho_s$ is surface density
- $\rho_d$ is deep density
- $H$ is the e-folding depth scale

Thermal conductivity also varies with depth through the density dependence:

$$k_c(z) = k_{c0} \cdot \frac{\rho(z)}{\rho_s}$$

## References

Hayne, P. O., Bandfield, J. L., Siegler, M. A., Vasavada, A. R., Ghent, R. R., Williams, J.-P., Greenhagen, B. T., Aharonson, O., Elder, C. M., Lucey, P. G., & Paige, D. A. (2017). Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 122(12), 2371–2400. https://doi.org/10.1002/2017JE005387

Mitchell, D. L., & De Pater, I. (1994). Microwave imaging of Mercury's thermal emission at wavelengths from 0.3 to 20.5 cm. *Icarus*, 110(1), 2–32. https://doi.org/10.1006/icar.1994.1104

Vasavada, A. R., Bandfield, J. L., Greenhagen, B. T., Hayne, P. O., Siegler, M. A., Williams, J.-P., & Paige, D. A. (2012). Lunar equatorial surface temperatures and regolith properties from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 117(E12). https://doi.org/10.1029/2011JE003987
