# Radiation Exchange Integration Plan for PTM3D

## Overview

This plan describes the integration of the radiation exchange module (`src/cpp/radiation/`) into the main simulation file (`ptm_thermal_sim.cpp`). The goal is to enable terrain-to-terrain thermal radiation exchange, which moderates temperatures in shadowed regions (craters, highlands).

## Key Requirements from User

1. **Face selection based on local horizon and view factors**: Faces with significant view factors are identified based on terrain geometry (horizon angles) and the computed view factor matrix. These faces are updated during radiation-thermal coupling iterations.

2. **Proper equilibration integration**: The equilibration cycles must be properly integrated with the main simulation thermal run. The Profile object maintains all thermal state; we use warm-start from equilibration.

3. **No separate depth profile handling**: The thermal inertia and depth profiles are contained within the Profile/Model objects. No need to pass separate depth profile vectors.

## Architecture

### Existing Radiation Module Components

| File | Class | Purpose |
|------|-------|---------|
| `radiation_config.h` | `RadiationConfig`, `FaceClassification` | Configuration and face selection data |
| `radiation_exchange.h/cpp` | `RadiationExchangeManager` | High-level API for view factors and radiosity |
| `face_selection.h/cpp` | `FaceSelector`, helper functions | Face classification based on geometry |
| `view_factor.h/cpp` | `ViewFactorMatrix`, `ViewFactorCalculator` | Sparse view factor computation |
| `hmatrix_view_factor.h/cpp` | `HMatrixViewFactor` | H-matrix compressed view factors |

### Integration Points in ptm_thermal_sim.cpp

1. **Include headers** (near top of file)
2. **Command-line argument parsing** (in help text and parsing loop)
3. **RadiationConfig setup** (after parsing, before simulation)
4. **Initial thermal run** (unchanged - runs equilibration + main simulation)
5. **Post-thermal radiation iteration loop** (NEW - after initial thermal run)
6. **Selective thermal re-run function** (NEW - for radiation iterations)
7. **Output of indirect flux** (add to illumination arrays for export)

## Detailed Implementation Steps

### Step 1: Add Include Directives

Add at the top of `ptm_thermal_sim.cpp`:
```cpp
#include "../radiation/radiation_config.h"
#include "../radiation/radiation_exchange.h"
```

### Step 2: Add Radiation Command-Line Arguments

Add to the help text and argument parsing:

**Command-line flags to support:**
- `--radiation-exchange` - Enable radiation exchange
- `--rad-min-shadow FRAC` - Min shadow fraction (default: 0.3)
- `--rad-min-angle DEG` - Min terrain view angle (default: 5.0)
- `--rad-max-neighbors N` - Max neighbors per face (default: 2000)
- `--rad-search-radius FACTOR` - Search radius factor (default: 10.0)
- `--rad-max-iter N` - Max radiosity iterations (default: 100)
- `--rad-tolerance TOL` - Convergence tolerance (default: 0.01)
- `--rad-verbose` - Enable verbose output
- `--rad-use-gpu` - Use GPU radiosity solver
- `--rad-view-factor-file FILE` - Save/load view factors
- `--rad-coupling-iter N` - Max coupling iterations (default: 10)
- `--rad-coupling-temp-tol K` - Temperature convergence threshold (default: 0.5)
- `--rad-coupling-flux-tol W` - Flux change threshold (default: 0.1)
- `--rad-coupling-face-frac F` - Face fraction for stopping (default: 0.01)
- `--rad-hmatrix` - Enable H-matrix compression
- `--rad-hmatrix-tol TOL` - H-matrix tolerance (default: 0.01)
- `--rad-hmatrix-depth N` - H-matrix max depth (default: 10)
- `--rad-hmatrix-block-size N` - H-matrix block size (default: 64)
- `--rad-hmatrix-rank N` - H-matrix max rank (default: 50)

### Step 3: Create RadiationConfig from SimulationConfig

After argument parsing, populate a `radiation::RadiationConfig`:
```cpp
radiation::RadiationConfig rad_config;
rad_config.enabled = config.radiation.enabled;
rad_config.min_shadow_fraction = config.radiation.min_shadow_fraction;
rad_config.min_terrain_view_angle = config.radiation.min_terrain_view_angle;
rad_config.max_neighbors = config.radiation.max_neighbors;
// ... etc
```

### Step 4: Face Classification Strategy

The face classification determines which faces need radiation exchange.

**Primary Physics Criterion: Terrain View Angle**

The **terrain view angle** is the key criterion. It measures the maximum horizon elevation angle - i.e., whether the face has neighboring terrain ABOVE its local tangent plane that could radiate to it.

- **Flat surface on a sphere**: All neighbors curve BELOW the tangent plane, so `max_terrain_angle = 0` → NO exchange needed
- **Crater floor**: Walls rise ABOVE the tangent plane, so `max_terrain_angle > 0` → Exchange needed
- **Crater wall**: Can see the opposite wall above its horizon, so `max_terrain_angle > 0` → Exchange needed
- **Surface next to a cliff**: The cliff rises above the horizon, so `max_terrain_angle > 0` → Exchange needed

**Criteria (meets ANY → included in exchange):**
1. **Terrain view angle > min_terrain_view_angle** (default 5°): Face can "see" terrain above its local horizon - the **primary physical criterion**
2. **Shadow fraction > min_shadow_fraction** (default 0.3): Face spends significant time in shadow - a **secondary indicator** that may catch PSRs not caught by terrain angle
3. **Sky view factor < max_sky_view_factor** (default 0.95): Face has partially obscured sky - **redundant safety net**

**Note on Shadow Fraction**: You correctly identified that the shadow fraction criterion alone is insufficient - a fully sunlit crater wall can still receive significant indirect IR from hot neighboring faces. The terrain view angle criterion captures this case. The shadow fraction is kept as a secondary indicator because:
- It catches PSR faces that may have unusual geometry
- It's computed from actual illumination rather than geometric approximation
- The OR logic is conservative (includes more faces rather than fewer)

**Implementation:**
1. After illumination is computed, extract per-face illumination time series
2. Estimate terrain view angles using `estimateTerrainViewAngles()` - this is the key step
3. Compute shadow fractions from illumination (flux < threshold → shadowed)
4. Use FaceSelector::classify() with OR logic to identify exchange faces

### Step 5: View Factor Computation/Loading

**Logic:**
1. Check if `--rad-view-factor-file` exists → attempt to load with auto-detection
2. Auto-detect format based on magic number in file header:
   - `.vfm` format: sparse matrix (magic = 0x5646464D "VFFM")
   - `.hmat` format: H-matrix compressed (magic = 0x484D4154 "HMAT")
3. If loaded successfully, skip computation
4. If not loaded, compute view factors using `ViewFactorCalculator` or `HMatrixViewFactor` depending on `--rad-hmatrix` flag
5. Save to file if filename provided (format matches computation method)

### Step 6: Radiation-Thermal Coupling Iteration Loop

**Key insight**: The initial thermal run establishes the baseline temperature field. Radiation iterations refine temperatures for faces that receive significant indirect flux.

**Algorithm:**
```
1. Run initial thermal model (equilibration + full simulation) → surface_temps[face][time]
2. If radiation enabled:
   a. Classify faces for radiation exchange
   b. Compute/load view factors
   c. Initialize RadiationExchangeManager
   d. For iter = 1 to max_coupling_iterations:
      i.   Compute indirect flux for all timesteps
           - For each timestep t:
             * temperatures = surface_temps[:, t]
             * solar_flux = direct_solar[:, t] + primary_reflected[:, t]
             * indirect[:, t] = manager.computeIndirectFlux(temperatures, solar_flux)
      ii.  Determine which faces need thermal re-run:
           - If iter == 1: faces where indirect > 1% of total absorbed
           - If iter > 1: faces where |Δindirect| > coupling_flux_threshold
      iii. Re-run thermal model for selected faces with updated flux
           - absorbed_flux[face][t] += indirect[face][t]
           - Use warm-start from previous iteration temperatures
      iv.  Check convergence:
           - mean(|ΔT|) < coupling_temp_threshold
           - fraction of faces still changing < coupling_face_fraction
           - If converged, break
   e. Update illumination arrays with indirect flux for export
```

### Step 7: Selective Thermal Re-run Function

**New function signature:**
```cpp
bool runThermalModelsForSelectedFaces_RadiationIteration(
    const std::vector<int>& face_indices,           // Faces to update
    const std::vector<std::vector<double>>& absorbed_flux_all,  // [n_faces][n_times]
    const std::vector<double>& times,
    const thermal::Config& config,
    std::vector<std::vector<double>>& surface_temps_inout,  // Updated in place
    int thermal_threads);
```

**Key behavior:**
- Only processes faces in `face_indices`
- Uses `absorbed_flux_all` which includes indirect radiation (added to absorbed solar/IR)
- Warm-starts from existing `surface_temps_inout` (no equilibration - incremental update only)
- Updates temperatures in place
- Does NOT re-run equilibration - these are incremental corrections to the baseline solution

**Rationale for no equilibration:**
The initial thermal run (without radiation exchange) already establishes the equilibrated temperature field. The radiation iterations are incremental corrections - they update temperatures to account for the indirect flux that wasn't included in the initial run. Re-running equilibration would be wasteful and could introduce discontinuities.

**Flux handling:**
The indirect flux from radiation exchange is added to the absorbed flux:
```cpp
total_absorbed[face][t] = original_absorbed[face][t] + indirect_flux[face][t]
```
where:
- `original_absorbed` = (1-α) × direct_solar + (1-α) × primary_reflected + ε × primary_IR
- `indirect_flux` = ε × Σⱼ Fᵢⱼ × Bⱼ (absorbed indirect radiation from radiosity solver)

Both terms have units of W/m² and represent absorbed power per unit area.

### Step 8: Progress Reporting

For each radiation iteration:
```
--- Radiation iteration N (adaptive, max M) ---
  Computing indirect flux [========================================] 100%
  Indirect flux: mean = X.X W/m^2, max = X.X W/m^2
  Selective thermal re-run: X of Y exchange faces need update
  Rad-Iter Thermal [========================================] 100%
  Temperature change: mean = X.X K, max = X.X K
  [Converged | Not converged] after N iterations
```

### Step 9: Output Integration

After radiation iterations complete:
1. Add indirect flux to illumination arrays (for export)
2. Update any statistics that reference illumination
3. Export radiation exchange statistics if verbose

## Memory Considerations

For large meshes (>1M faces) with many timesteps:
- Illumination series: n_faces × n_times × 8 bytes (double)
- Temperature series: n_faces × n_times × 8 bytes
- View factors: O(n_faces × max_neighbors) sparse entries
- For 1.3M faces × 288 timesteps: ~3 GB per array

**Mitigation:**
- Use chunked processing (already in place for thermal model)
- H-matrix compression reduces view factor memory for very large meshes
- Streaming mode disabled when radiation enabled (need full arrays)

## File Changes Summary

| File | Changes |
|------|---------|
| `ptm_thermal_sim.cpp` | Major additions: headers, CLI args, radiation iteration loop, selective thermal function |
| `CMakeLists.txt` (main) | Link to radiation library |
| `config_parser.cpp` | Parse radiation YAML section and CLI flags (already partially done) |

## Testing Plan

1. **Unit test**: Run bowl crater test (`test_bowl_crater`) to verify view factors
2. **Integration test**: Small mesh (res 4-5) with `--radiation-exchange`
3. **Validation**: Compare exchange face temperatures with/without radiation
   - Expected: shadowed faces warmer with radiation exchange
4. **Performance**: Benchmark view factor computation for large meshes (res 8)

## Success Criteria

1. Shadowed crater floors show warmer nighttime temperatures (~60-100K instead of ~40-50K)
2. Temperature gradients at crater walls are physically reasonable
3. Convergence within 3-5 iterations for most cases
4. View factors can be cached and reloaded
5. Wall-clock time increase acceptable (view factors cached between runs)

## Phase 2 Improvements (Future)

1. Multi-bounce scattering (currently single-bounce)
2. GPU-accelerated view factor computation
3. Time-varying emissivity based on temperature
4. Subsurface lateral heat conduction
