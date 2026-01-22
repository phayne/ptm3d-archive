# Radiation Exchange Documentation

This document describes the radiation exchange module for modeling inter-face thermal radiation.

## Overview

The radiation exchange module computes view factors between mesh faces and solves the radiosity equation for coupled thermal-radiation problems.

## Radiosity Equation

For diffuse gray surfaces, the radiosity equation is:

$$B_i = \varepsilon_i \sigma T_i^4 + (1-\varepsilon_i) \sum_j F_{ij} B_j + (1-\varepsilon_i) Q_{\text{solar},i}$$

Where:
- $B_i$ is the radiosity of face $i$ (W/m²)
- $\varepsilon_i$ is emissivity
- $\sigma$ is the Stefan-Boltzmann constant ($5.67 \times 10^{-8}$ W/m²/K⁴)
- $F_{ij}$ is the view factor from face $i$ to face $j$
- $Q_{\text{solar},i}$ is direct solar flux on face $i$

The indirect flux absorbed by face $i$ is:

$$Q_{\text{indirect},i} = \varepsilon_i \sum_j F_{ij} B_j$$

## View Factor Computation

The view factor $F_{ij}$ represents the fraction of radiation leaving surface $i$ that reaches surface $j$:

$$F_{ij} = \frac{1}{A_i} \int_{A_i} \int_{A_j} \frac{\cos\theta_i \cos\theta_j}{\pi r^2} V_{ij} \, dA_j \, dA_i$$

Where:
- $\theta_i$ and $\theta_j$ are angles from the surface normals
- $r$ is the distance between surface elements
- $V_{ij}$ is the visibility function (0 or 1)

View factors are computed using ray casting with Intel Embree:
1. Sample rays from face $i$ toward face $j$
2. Check for occlusion (visibility)
3. Apply reciprocity: $A_i F_{ij} = A_j F_{ji}$
4. Store in sparse CSR matrix format

### H-Matrix Compression

For large meshes (>50K faces), view factors are compressed using hierarchical matrices:
- **3D octree spatial partitioning** (better than quadtree for crater geometry where Z variation matters)
- Low-rank SVD with randomized approximation for far-field blocks
- Sparse storage for near-field interactions
- $O(N \log N)$ storage vs $O(N^2)$ for dense
- Ray-packet optimization with AVX-512/AVX2 SIMD

The low-rank approximation exploits the fact that view factors between distant clusters decay smoothly:

$$F_{\text{far}} \approx U \Sigma V^T$$

where $U$, $V$ are orthogonal matrices and $\Sigma$ contains singular values truncated at tolerance $\tau$.

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

## GPU Acceleration

The radiosity solver can be accelerated on GPU using OpenCL:
- Sparse matrix-vector multiply (SpMV) for view factor operations
- CSR format for efficient GPU memory access
- Jacobi iteration with parallel convergence tracking

Enable with `--rad-use-gpu` (requires OpenCL build).

## File Formats

### Sparse View Factor Format (.vfm)
- Magic number: `0x5646464D` ("VFFM")
- CSR sparse matrix storage
- Auto-detected on load

### H-Matrix Format (.hmat)
- Magic number: `0x484D4154` ("HMAT")
- Hierarchical block structure with low-rank and sparse blocks
- Binary serialization for save/load

## References

Potter, S. F., Bertone, S., Schörghofer, N., & Mazarico, E. (2023). Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces. *Journal of Computational Physics: X*, 17, 100130. https://doi.org/10.1016/j.jcpx.2023.100130
