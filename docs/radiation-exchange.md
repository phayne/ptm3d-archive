# Radiation Exchange Documentation

This document describes the radiation exchange module for modeling inter-face thermal radiation.

## Overview

The radiation exchange module computes view factors between mesh faces and solves the radiosity equation for coupled thermal-radiation problems.

## Radiosity Equation

For diffuse gray surfaces:
```
B_i = ε_i σ T_i⁴ + (1-ε_i) Σⱼ F_ij B_j + (1-ε_i) Q_solar,i
```

Where:
- `B_i` is the radiosity of face i (W/m²)
- `ε_i` is emissivity
- `F_ij` is the view factor from face i to face j
- `Q_solar,i` is direct solar flux on face i

## View Factor Computation

View factors are computed using ray casting with Intel Embree:
1. Sample rays from face i toward face j
2. Check for occlusion
3. Apply reciprocity: `A_i F_ij = A_j F_ji`
4. Store in sparse CSR matrix format

### H-Matrix Compression

For large meshes (>50K faces), view factors are compressed using hierarchical matrices:
- Quadtree spatial partitioning
- Low-rank SVD for far-field blocks
- Sparse storage for near-field
- O(N log N) storage vs O(N²) for dense

## Thermal-Radiation Coupling

Iterative solver for coupled problems:
1. Compute temperatures with current indirect flux
2. Update radiosity with new temperatures
3. Recompute indirect flux
4. Check convergence criteria
5. Repeat until converged

### Convergence Criteria
- Temperature change < threshold (default: 0.5 K)
- Flux change < threshold (default: 0.1 W/m²)
- Fraction of faces still changing < threshold (default: 1%)

## Command-Line Options

```
--radiation-exchange      Enable radiation exchange
--rad-hmatrix            Use H-matrix compression
--rad-view-factor-file   Save/load view factors
--rad-coupling-iter      Max coupling iterations
--rad-coupling-temp-tol  Temperature convergence threshold
```

## References

- Potter et al. (2023) - H-matrix implementation
- Modest (2013) - Radiative Heat Transfer
