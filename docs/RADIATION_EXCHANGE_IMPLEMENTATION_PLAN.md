# Radiation Exchange Module Implementation Plan for PTM3D

## Executive Summary

This document outlines a plan for implementing view factor-based radiation exchange in the PTM3D thermal simulation framework. The module will enable accurate modeling of:
1. **Scattered solar radiation** - sunlight reflected from neighboring terrain into shadowed regions
2. **Thermal re-radiation** - IR emission from warm surfaces heating cooler neighboring surfaces

These effects are critical for accurate temperature predictions in permanently shadowed regions (PSRs), craters, and rough terrain where indirect radiation can dominate the energy budget.

---

## 1. Background and Physics

### 1.1 The Radiosity Equation

For a system of N diffusely-emitting Lambertian surfaces, the radiosity (total outgoing radiative flux) at surface i is:

```
B_i = ε_i σ T_i^4 + ρ_i Σ_j F_ij B_j + ρ_i Q_solar,i
```

Where:
- `B_i` = radiosity (outgoing flux) from face i [W/m²]
- `ε_i` = emissivity of face i
- `ρ_i = 1 - ε_i` = reflectivity (assuming opaque surfaces)
- `σ` = Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²/K⁴)
- `T_i` = temperature of face i [K]
- `F_ij` = view factor from face i to face j
- `Q_solar,i` = direct solar flux absorbed at face i

### 1.2 View Factor Definition

The view factor F_ij represents the fraction of radiation leaving surface i that arrives at surface j:

```
F_ij = (1/A_i) ∫∫_{A_i,A_j} [cos(θ_i) cos(θ_j) / (π r²)] V_ij dA_j dA_i
```

Where:
- `θ_i`, `θ_j` = angles between surface normals and the connecting line
- `r` = distance between surface elements
- `V_ij` = visibility function (1 if unoccluded, 0 otherwise)

### 1.3 Key Properties

1. **Reciprocity**: `A_i F_ij = A_j F_ji`
2. **Summation**: `Σ_j F_ij + F_i,sky = 1` (for closed system including sky)
3. **Self-view**: `F_ii = 0` for convex surfaces

---

## 2. Literature Review Summary

### 2.1 Potter et al. (2023) - Hierarchical Low-Rank View Factors

**Reference**: Potter, S.F., Bertone, S., Schörghofer, N., & Mazarico, E. (2023). "Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces." *J. Comp. Phys. X*, 17, 100130.

**Key Contributions**:
- Developed H-matrix compression for view factor matrices
- Achieves O(N) storage and O(N) matrix-vector multiply (vs O(N²) for dense)
- Quadtree/octree spatial partitioning for block identification
- Sparse SVD for low-rank block compression
- Orders of magnitude speedup for large meshes (>10k faces)

**Applicability to PTM3D**:
- Ideal for meshes with 10k-100k+ faces
- Requires significant implementation effort
- Consider as Phase 3 optimization

### 2.2 Davidsson & Rickman (2014) - View Factor Methods Comparison

**Reference**: Davidsson, B.J.R., & Rickman, H. (2014). "Accuracy of view factor calculations for digital terrain models of comets and asteroids." *Astron. Astrophys.*, 570, A12.

**Methods Compared**:
| Method | Accuracy | Speed | Best For |
|--------|----------|-------|----------|
| M1 (Centroid) | ~1% for Ns>100 | Fastest | Distant faces |
| M3-Linear (Subdivision) | ~1% all Ns | Medium | Close faces |
| M4 (Line integral) | Reference | Slowest | Adjacent faces |

**Recommendation**: Combined approach selecting method based on face separation.

### 2.3 Schörghofer Planetary-Code-Collection

**Reference**: https://github.com/nschorgh/Planetary-Code-Collection (Topo3D module)

**Key Features**:
- Fortran/C implementation of 3D terrain thermal models
- View factor computation with horizon calculations
- Used for lunar PSR studies
- Integrates with SPICE for ephemeris

---

## 3. Implementation Strategy

### 3.1 Phased Approach

Given the computational complexity, we propose a three-phase implementation:

#### Phase 1: Foundation (Core Infrastructure)
- View factor data structures
- Basic view factor computation (M1 centroid method)
- Integration points in thermal solver
- Selective face identification

#### Phase 2: Optimization (Performance)
- Adaptive method selection (M1/M3/M4)
- Visibility acceleration using Embree
- Sparse matrix storage
- OpenMP parallelization

#### Phase 3: Advanced (Scalability)
- H-matrix compression (Potter et al.)
- GPU acceleration option
- Out-of-core processing for huge meshes

### 3.2 Selective Computation Strategy

**Key Insight**: Not all faces require radiation exchange. The computational burden can be dramatically reduced by:

1. **Pre-screening faces** that receive significant indirect radiation:
   - Faces in topographic depressions (craters, valleys)
   - Faces that are shadowed for significant time fractions
   - Faces with steep local slopes (>30°)

2. **Two-pass simulation**:
   - **Pass 1**: Run standard illumination + thermal model without radiation exchange
   - **Pass 2**: Identify "cold" faces that would benefit from indirect heating
   - **Pass 3**: Re-run thermal model for selected faces with radiation exchange

3. **Locality exploitation**:
   - Most radiation exchange is local (nearby faces)
   - Faces beyond ~5 crater diameters contribute negligibly
   - Use spatial indexing to limit view factor computation

---

## 4. Detailed Design

### 4.1 New Module Structure

```
src/cpp/radiation/
├── CMakeLists.txt
├── view_factor.h              # View factor computation
├── view_factor.cpp
├── radiation_exchange.h       # Radiosity solver
├── radiation_exchange.cpp
├── face_selection.h           # Selective face identification
├── face_selection.cpp
└── h_matrix.h                 # (Phase 3) Hierarchical compression
```

### 4.2 Core Data Structures

```cpp
namespace radiation {

// View factor matrix storage (sparse CSR format)
struct ViewFactorMatrix {
    std::vector<double> values;     // Non-zero F_ij values
    std::vector<int> col_indices;   // Column indices
    std::vector<int> row_ptrs;      // Row pointers
    size_t n_faces;                 // Matrix dimension

    // Methods
    double get(int i, int j) const;
    void set(int i, int j, double value);
    void matvec(const std::vector<double>& x, std::vector<double>& y) const;
};

// Face classification for selective computation
struct FaceClassification {
    std::vector<bool> needs_exchange;      // True if face needs radiation exchange
    std::vector<int> exchange_face_ids;    // Indices of faces needing exchange
    std::vector<double> shadow_fractions;  // Time fraction in shadow
    std::vector<double> max_view_angles;   // Maximum terrain view angle
};

// Configuration for radiation exchange
struct RadiationConfig {
    bool enabled = false;
    double min_shadow_fraction = 0.5;       // Minimum shadow fraction to consider
    double min_view_angle = 10.0;           // Minimum terrain view angle (degrees)
    double visibility_threshold = 0.001;    // Minimum F_ij to store
    int max_neighbors = 1000;               // Maximum neighbors per face
    double search_radius_factor = 5.0;      // Search radius in units of face diameter

    // Method selection
    enum ViewFactorMethod { CENTROID, SUBDIVISION, LINE_INTEGRAL, ADAPTIVE };
    ViewFactorMethod method = ADAPTIVE;
    int subdivision_level = 4;              // For M3 method
    double adaptive_threshold = 50.0;       // Ns threshold for method switching
};

} // namespace radiation
```

### 4.3 View Factor Computation

```cpp
class ViewFactorCalculator {
public:
    ViewFactorCalculator(const ptm::Mesh& mesh,
                         RTCScene embree_scene,
                         const RadiationConfig& config);

    // Compute view factor matrix for selected faces
    ViewFactorMatrix compute(const std::vector<int>& face_ids);

    // Single view factor computation (for testing/debugging)
    double computePair(int face_i, int face_j);

private:
    // M1: Centroid approximation (fast, for distant faces)
    double computeCentroid(int face_i, int face_j);

    // M3: Subdivision method (accurate, for close faces)
    double computeSubdivision(int face_i, int face_j, int level);

    // M4: Line integral (most accurate, for adjacent faces)
    double computeLineIntegral(int face_i, int face_j);

    // Visibility check using Embree
    bool isVisible(const std::array<double,3>& p1,
                   const std::array<double,3>& p2);

    // Batch visibility check
    std::vector<bool> checkVisibilityBatch(
        const std::vector<std::array<double,3>>& origins,
        const std::vector<std::array<double,3>>& targets);

    // Spatial search for potential neighbors
    std::vector<int> findPotentialNeighbors(int face_id, double radius);
};
```

### 4.4 Radiosity Solver

```cpp
class RadiositySolver {
public:
    RadiositySolver(const ViewFactorMatrix& F,
                    const std::vector<double>& emissivity,
                    const std::vector<double>& areas);

    // Solve radiosity equation: B = E + ρ*F*B + ρ*Q_solar
    // Returns: absorbed flux at each face (for use in thermal solver)
    std::vector<double> solve(
        const std::vector<double>& temperatures,  // Current T for emission
        const std::vector<double>& solar_flux,    // Direct solar input
        int max_iterations = 50,
        double tolerance = 0.01);

    // Iterative Jacobi solver (simple, robust)
    std::vector<double> solveJacobi(
        const std::vector<double>& temperatures,
        const std::vector<double>& solar_flux,
        int max_iterations,
        double tolerance);

    // Direct solver for small systems
    std::vector<double> solveDirect(
        const std::vector<double>& temperatures,
        const std::vector<double>& solar_flux);

private:
    const ViewFactorMatrix& F_;
    std::vector<double> emissivity_;
    std::vector<double> reflectivity_;  // 1 - emissivity
    std::vector<double> areas_;
};
```

### 4.5 Integration with Thermal Model

The key integration point is in the surface boundary condition. Currently in `thermal_model.cpp`:

```cpp
void surfTemp(Profile& p, double Qs) {
    // Current: Qs is direct + primary solar/IR flux
    // Solve: ε σ T^4 = (1-A) Qs + k ∂T/∂z
}
```

**Modified version with radiation exchange**:

```cpp
void surfTempWithRadiation(Profile& p, double Qs, double Q_indirect) {
    // Q_indirect = scattered solar + thermal emission from neighbors
    // Solve: ε σ T^4 = (1-A) Qs + Q_indirect + k ∂T/∂z
}
```

The `Q_indirect` term comes from the radiosity solver and includes:
1. Scattered solar radiation from illuminated neighbors
2. Thermal IR emission from all visible neighbors

### 4.6 Main Simulation Flow

```cpp
// In ptm_thermal_sim.cpp

void runWithRadiationExchange(
    const Mesh& mesh,
    const IlluminationModel& illum,
    const thermal::Config& config,
    const radiation::RadiationConfig& rad_config)
{
    // === Phase 1: Initial thermal simulation without exchange ===
    std::vector<Profile> profiles = initializeProfiles(mesh, config);
    std::vector<std::vector<double>> temperatures_history;

    runThermalSimulation(profiles, illum, temperatures_history);

    // === Phase 2: Identify faces needing radiation exchange ===
    FaceClassification classification = classifyFaces(
        mesh, illum, temperatures_history, rad_config);

    std::cout << "Faces requiring radiation exchange: "
              << classification.exchange_face_ids.size()
              << " / " << mesh.faces.size() << std::endl;

    if (classification.exchange_face_ids.empty()) {
        std::cout << "No faces need radiation exchange. Done." << std::endl;
        return;
    }

    // === Phase 3: Compute view factors for selected faces ===
    ViewFactorCalculator vf_calc(mesh, embree_scene, rad_config);
    ViewFactorMatrix F = vf_calc.compute(classification.exchange_face_ids);

    std::cout << "View factor matrix: " << F.n_faces << " faces, "
              << F.values.size() << " non-zero entries" << std::endl;

    // === Phase 4: Re-run thermal simulation with radiation exchange ===
    RadiositySolver radiosity(F, emissivities, areas);

    for (int t = 0; t < n_timesteps; t++) {
        // Get current temperatures for exchange faces
        std::vector<double> T_exchange = extractTemperatures(
            profiles, classification.exchange_face_ids);

        // Get solar fluxes for this timestep
        std::vector<double> Q_solar = extractSolarFlux(
            illum, t, classification.exchange_face_ids);

        // Solve radiosity for indirect flux
        std::vector<double> Q_indirect = radiosity.solve(T_exchange, Q_solar);

        // Advance thermal profiles with indirect flux
        for (size_t i = 0; i < classification.exchange_face_ids.size(); i++) {
            int face_id = classification.exchange_face_ids[i];
            double Qs = illum.getIlluminationValue(face_id, t);
            profiles[face_id].update_T(dt, Qs + Q_indirect[i], Qb);
        }
    }
}
```

---

## 5. Computational Complexity Analysis

### 5.1 View Factor Computation

| Phase | Operation | Complexity | Notes |
|-------|-----------|------------|-------|
| Neighbor search | K-d tree query | O(N log N) | One-time setup |
| Visibility tests | Embree ray trace | O(N × k × log N) | k = avg neighbors |
| View factor calc | Per-pair integral | O(N × k) | Using M1 centroid |
| Total (Phase 1-2) | | O(N × k × log N) | k typically 50-500 |

### 5.2 Radiosity Solve (per timestep)

| Method | Complexity | Memory | Best For |
|--------|------------|--------|----------|
| Jacobi iteration | O(N × k × iter) | O(N × k) | Sparse systems |
| Direct (LU) | O(N³) | O(N²) | Small N (<1000) |
| H-matrix | O(N × log²N) | O(N × log N) | Large N (>10k) |

### 5.3 Memory Estimates

For a mesh with N faces and average k non-zero view factors per face:

| Component | Memory | Example (N=50k, k=200) |
|-----------|--------|------------------------|
| View factor matrix (sparse) | 8×N×k bytes | 80 MB |
| Radiosity vectors | 8×N×2 bytes | 0.8 MB |
| Face classification | N bytes | 0.05 MB |
| Total | | ~80 MB |

For dense storage (not recommended for large meshes):
- Dense matrix: 8×N² bytes = 20 GB for N=50k

---

## 6. Implementation Timeline

### Phase 1: Foundation (Weeks 1-3)

1. **Week 1**: Core data structures
   - [ ] Create `src/cpp/radiation/` directory structure
   - [ ] Implement `ViewFactorMatrix` sparse storage
   - [ ] Implement `FaceClassification` structure
   - [ ] Add `RadiationConfig` to configuration system

2. **Week 2**: View factor computation
   - [ ] Implement M1 centroid view factor
   - [ ] Integrate with Embree for visibility
   - [ ] Implement spatial neighbor search
   - [ ] Unit tests for view factor accuracy

3. **Week 3**: Thermal integration
   - [ ] Modify `surfTemp()` for indirect flux
   - [ ] Implement simple Jacobi radiosity solver
   - [ ] End-to-end test with simple crater geometry

### Phase 2: Optimization (Weeks 4-6)

4. **Week 4**: Accuracy improvements
   - [ ] Implement M3 subdivision method
   - [ ] Implement adaptive method selection
   - [ ] Validate against analytical solutions

5. **Week 5**: Performance optimization
   - [ ] OpenMP parallelization of view factor computation
   - [ ] Batch visibility testing with SIMD
   - [ ] Memory-efficient sparse matrix operations

6. **Week 6**: Integration and testing
   - [ ] Full integration with main simulation driver
   - [ ] Command-line options for radiation exchange
   - [ ] Performance benchmarking
   - [ ] Documentation

### Phase 3: Advanced Features (Future)

- [ ] H-matrix compression for large meshes
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Multi-scattering (beyond single bounce)
- [ ] Wavelength-dependent albedo

---

## 7. Validation Strategy

### 7.1 Analytical Test Cases

1. **Parallel plates**: Known analytical F_12 = 1
2. **Perpendicular plates with common edge**: F_12 = 0.2 (analytical)
3. **Spherical crater**: F_wall,floor from geometry

### 7.2 Benchmark Against Literature

1. **Ingersoll crater model**: Compare PSR temperatures
2. **Schörghofer Topo3D**: Cross-validate view factors
3. **Potter et al. test cases**: Synthetic craters

### 7.3 Physical Validation

1. **Energy conservation**: Σ_j F_ij + F_sky = 1
2. **Reciprocity**: A_i F_ij = A_j F_ji
3. **Temperature bounds**: T > T_equilibrium without exchange

---

## 8. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Numerical instability in radiosity solver | High | Medium | Use robust iterative methods, damping |
| Memory exhaustion for large meshes | High | Medium | Sparse storage, chunked processing |
| Poor performance | Medium | Medium | Profiling, selective computation |
| Accuracy issues | Medium | Low | Extensive validation, method comparison |
| Integration complexity | Medium | Medium | Clean interfaces, incremental integration |

---

## 9. References

1. Potter, S.F., Bertone, S., Schörghofer, N., & Mazarico, E. (2023). "Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces." *J. Comp. Phys. X*, 17, 100130.

2. Davidsson, B.J.R., & Rickman, H. (2014). "Accuracy of view factor calculations for digital terrain models." *Astron. Astrophys.*, 570, A12.

3. Ingersoll, A.P., Svitek, T., & Murray, B.C. (1992). "Stability of polar frosts in spherical bowl-shaped craters on the Moon, Mercury, and Mars." *Icarus*, 100, 40-47.

4. Schörghofer, N. (2020). "Planetary-Code-Collection." https://github.com/nschorgh/Planetary-Code-Collection

5. Howell, J.R., Mengüç, M.P., & Siegel, R. (2015). *Thermal Radiation Heat Transfer*. CRC Press, 6th edition.

---

## 10. Appendix: View Factor Formulas

### A.1 Centroid Approximation (M1)

For two triangular faces i and j with centroids c_i, c_j and normals n_i, n_j:

```
r = c_j - c_i
d = |r|
r_hat = r / d

cos_i = max(0, n_i · r_hat)
cos_j = max(0, -n_j · r_hat)

F_ij ≈ (A_j / π) × (cos_i × cos_j) / d²
```

### A.2 Subdivision Method (M3)

Subdivide face j into M sub-triangles, compute centroid approximation for each:

```
F_ij = Σ_{m=1}^{M} (A_jm / A_j) × F_i,jm
```

### A.3 Visibility Integration

For accurate view factors with partial occlusion, sample multiple points on each face:

```
F_ij = (1 / N_samples) Σ_{k=1}^{N_samples} V(p_ik, p_jk) × F_ij^(centroid)
```

Where V(p1, p2) is the visibility function (0 or 1) computed via ray tracing.
