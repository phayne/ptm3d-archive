# Radiation Exchange Module

This document describes the radiation exchange module used in PTM3D for modeling inter-face thermal radiation in complex topography (craters, rough terrain, permanently shadowed regions). The module computes view factors between mesh faces, solves the radiosity equation for coupled thermal-radiation problems, and iteratively couples the resulting indirect flux back into the 1D thermal conduction model at each surface element.

## 1. Physical Context

On airless bodies, surfaces in complex topography exchange thermal infrared radiation and reflect incident sunlight toward neighboring faces. In permanently shadowed regions (PSRs) such as polar craters, this indirect radiation can be the dominant heat source, controlling cold-trap temperatures and volatile stability. The radiation exchange module captures these effects by solving for multi-bounce radiative transfer among all mutually visible mesh faces.

The module sits between the illumination model, which provides direct solar flux $Q_{\text{solar},i}(t)$, primary-body reflected flux $Q_{\text{refl},i}(t)$, and primary-body thermal IR flux $Q_{\text{IR},i}(t)$, and the 1D thermal conduction solver, which computes subsurface temperature profiles and surface temperatures at each face. Indirect flux from the radiation exchange is added to the absorbed flux before entering the thermal solver's surface boundary condition.

## 2. Surface Energy Balance

At the surface ($z = 0$) of each mesh face $i$, the thermal model solves

$$Q_{s,i} + k_0 \frac{T_1 - T_0}{\Delta z_0} = \varepsilon_i \sigma T_0^4$$

where $T_0$ is the surface temperature (the unknown), $T_1$ is the first subsurface node temperature, $k_0$ is the thermal conductivity at the surface, $\Delta z_0$ is the first layer thickness, $\varepsilon_i$ is the broadband IR emissivity, and $\sigma = 5.67051 \times 10^{-8}$ W m$^{-2}$ K$^{-4}$ is the Stefan-Boltzmann constant. The surface temperature is found via Newton-Raphson iteration on

$$f(T_0) = Q_{s,i} + k_0 \frac{T_1 - T_0}{\Delta z_0} - \varepsilon_i \sigma T_0^4 = 0$$

with derivative $f'(T_0) = -k_0/\Delta z_0 - 4\varepsilon_i \sigma T_0^3$.

The total absorbed surface flux $Q_{s,i}(t)$ combines four components:

$$Q_{s,i}(t) = \bigl[Q_{\text{solar},i}(t) + Q_{\text{refl},i}(t)\bigr](1 - \alpha_i) + Q_{\text{IR},i}(t)\,\varepsilon_i + Q_{\text{indirect},i}(t)$$

where:

- $Q_{\text{solar},i}(t)$: direct solar flux (W/m$^2$), computed from SPICE ephemeris with finite solar disk integration, shadow ray tracing (Embree BVH), and eclipse geometry.
- $Q_{\text{refl},i}(t)$: reflected solar from the primary body (e.g., Earthshine for lunar cases), based on phase angle and Lambertian sphere model.
- $Q_{\text{IR},i}(t)$: thermal IR from the primary body, scaled by the solid-angle view factor $(R_p / d)^2$ and the primary's effective temperature.
- $\alpha_i$: visible-wavelength Bond albedo, optionally incidence-angle-dependent via the parametrization $\alpha(\theta) = \alpha_0 + a\,(\theta/(\pi/4))^3 + b\,(\theta/(\pi/2))^8$ (Hayne et al. 2017).
- $\varepsilon_i$: broadband IR emissivity (absorptivity by Kirchhoff's law). Default 0.96 for icy surfaces.
- $Q_{\text{indirect},i}(t)$: absorbed indirect flux from the radiosity solver (this module), already weighted by $\varepsilon_i$ internally.

## 3. Radiosity Equation

For an enclosure of $N$ diffuse gray surfaces, the radiosity (total outgoing radiant flux) of face $i$ satisfies

$$B_i = \varepsilon_i \sigma T_i^4 + \rho_i \sum_{j=1}^{N} F_{ij}\, B_j + \rho_i\, Q_{\text{solar},i}$$

where $B_i$ is the radiosity (W/m$^2$), $\rho_i = 1 - \varepsilon_i$ is the reflectivity, and $F_{ij}$ is the view factor from face $i$ to face $j$. The three terms represent thermal self-emission, reflected incoming radiation from other faces, and reflected incident solar flux.

In matrix form, the system is

$$(I - \rho\, F)\, \mathbf{B} = \mathbf{E} + \rho\, \mathbf{Q}_{\text{solar}}$$

where $\mathbf{E}_i = \varepsilon_i \sigma T_i^4$ and $\rho$ denotes the diagonal matrix of reflectivities.

After solving for $\mathbf{B}$, the absorbed indirect flux at each face is

$$Q_{\text{indirect},i} = \varepsilon_i \sum_{j=1}^{N} F_{ij}\, B_j$$

This is computed as a matrix-vector product $\mathbf{q} = F\,\mathbf{B}$ followed by elementwise multiplication by $\varepsilon_i$. By Kirchhoff's law ($\alpha_i = \varepsilon_i$ in the thermal IR), this correctly accounts for the fraction of incoming radiation absorbed at each face.

## 4. View Factor Computation

### 4.1 Definition

The view factor $F_{ij}$ is the fraction of radiation leaving surface $i$ that is intercepted by surface $j$:

$$F_{ij} = \frac{1}{A_i} \int_{A_i} \int_{A_j} \frac{\cos\theta_i \cos\theta_j}{\pi\, r^2}\, V_{ij}\, dA_j\, dA_i$$

where $\theta_i$ and $\theta_j$ are the angles between the respective surface normals and the line connecting the two area elements, $r$ is their separation distance, and $V_{ij} \in \{0, 1\}$ is the binary visibility function. View factors satisfy the reciprocity relation $A_i F_{ij} = A_j F_{ji}$ and the sum rule $\sum_j F_{ij} + F_{\text{sky},i} = 1$, where $F_{\text{sky},i}$ is the sky view factor (fraction of the hemisphere visible to space).

### 4.2 Computation Methods

PTM3D implements three methods, selected adaptively based on the dimensionless separation number $N_s = r / \sqrt{A_{\text{eff}}}$, where $A_{\text{eff}} = \sqrt{A_i A_j}$:

**M1 — Centroid approximation** (far field, $N_s > 5$). Evaluates the integrand at the face centroids only:

$$F_{ij} \approx \frac{A_j}{\pi} \frac{\cos\theta_i \cos\theta_j}{r^2}$$

where $\cos\theta_i = \hat{n}_i \cdot \hat{r}_{ij}$ and $\cos\theta_j = -\hat{n}_j \cdot \hat{r}_{ij}$, with $\hat{r}_{ij}$ the unit vector from centroid $i$ toward centroid $j$. A single Embree shadow ray verifies visibility. This method follows the M1 formulation of Rezac & Zhao (2020).

**M3 — Subdivision** (near field, $N_s \leq 5$). Subdivides the receiving face $j$ into sample points (default: $\sim$10 points at subdivision level 3) and averages the centroid-based estimate over them:

$$F_{ij} \approx \frac{A_j}{\pi\, n_{\text{pts}}} \sum_{k=1}^{n_{\text{pts}}} \frac{\cos\theta_{i,k} \cos\theta_{j,k}}{r_k^2}$$

This captures the spatial variation of the integrand over the receiving face, improving accuracy for nearby faces where the centroid approximation breaks down.

**Adaptive** (default). Selects M1 or M3 based on the threshold `adaptive_ns_threshold` (default 5.0).

### 4.3 Visibility Testing

All view factor computations require visibility checks between face pairs. These are performed using Intel Embree ray tracing:

- Single shadow rays for centroid (M1) calculations.
- Batched ray queries for M3 subdivision points.
- **SIMD ray packets**: AVX-512 16-wide (`rtcIntersect16`) or AVX2 8-wide (`rtcIntersect8`) packets, auto-detected at runtime, for batched visibility testing during H-matrix construction.

Faces that are geometrically back-to-back ($\hat{n}_i \cdot \hat{r}_{ij} \leq 0$ or $\hat{n}_j \cdot \hat{r}_{ij} \geq 0$) or at grazing angles ($\cos\theta < 0.01$, i.e., $\theta > 89.4°$) are culled before any ray trace.

### 4.4 Neighbor Selection

Computing all $N^2$ view factor pairs is prohibitive for large meshes. Instead, each face selects a subset of neighbors:

**Adaptive selection** (default). For each face $i$:
1. Find candidate faces within a search radius of $R_{\text{search}} = f_R \sqrt{A_i}$ (default $f_R = 10$) using a spatial grid.
2. Estimate the radiative potential of each candidate: $\hat{F}_{ij} \propto A_j |\cos\theta_i||\cos\theta_j| / r^2$.
3. Rank candidates by potential and greedily select neighbors in descending order until 95% of the total radiative potential is captured (configurable via `captured_energy_target`).
4. Enforce bounds: at least `min_neighbors` (default 20) and at most `max_neighbors` (default 500) per face.
5. Track global memory usage via atomic counters to stay within the memory budget.

**Fixed selection** (alternative). Simply selects the top `max_neighbors` candidates by potential.

### 4.5 Sparse Storage

The resulting view factor matrix is stored in Compressed Sparse Row (CSR) format:
- `values_`: non-zero $F_{ij}$ entries
- `col_indices_`: column indices (face $j$)
- `row_ptrs_`: row pointers (size $N+1$)

Operations: `matvec` (matrix–vector multiplication, $\mathbf{y} = F\mathbf{x}$), `matvecTranspose` ($\mathbf{y} = F^T\mathbf{x}$), `get(i,j)`, `getRow(i)`, and `rowSums` (for verifying the sum rule).

## 5. H-Matrix Compression

For large meshes ($\gtrsim$50K faces), storing the full sparse view factor matrix becomes expensive. Following Potter et al. (2023), the module uses hierarchical matrix (H-matrix) compression to achieve $O(N \log N)$ storage and matrix-vector product cost.

### 5.1 Spatial Partitioning

Mesh faces are recursively partitioned using a **3D octree** (preferred over a 2D quadtree because crater geometry has significant Z-axis variation). Each octree node stores the face indices, a 3D bounding box, and a diameter (the bounding box diagonal). Subdivision continues until reaching `max_tree_depth` (default 10) or `min_block_size` (default 64 faces).

### 5.2 Admissibility Condition

For each pair of octree leaf clusters $(X, Y)$, the standard admissibility condition determines whether the view factor subblock can be approximated as low-rank:

$$\max\bigl(\text{diam}(X),\, \text{diam}(Y)\bigr) \leq \eta \cdot \text{dist}(X, Y)$$

with $\eta = 0.8$. Here $\text{dist}(X, Y)$ is the Euclidean distance between cluster centers in 3D. If admissible (clusters are well-separated), the block is compressed; otherwise it is recursively subdivided into child blocks, up to `max_hierarchical_levels` (default 2) additional levels.

An additional **cluster-level visibility culling** step uses the average normal vectors of each cluster to conservatively skip block pairs that are geometrically non-visible (both clusters face away from each other).

### 5.3 Block Types

Each block in the H-matrix is stored as one of:
- **ZeroBlock**: clusters cannot see each other (no view factors).
- **LowRankBlock**: compressed as $F_{\text{far}} \approx U \Sigma V^T$, where $U$, $V$ are dense matrices of size $m \times k$ and $n \times k$, with $k \ll \min(m,n)$ the numerical rank.
- **SparseBlock**: full CSR storage for small or near-field blocks.
- **DenseBlock**: full dense storage.
- **HierarchicalBlock**: recursively subdivided into child blocks.
- **DiskBlock**: for out-of-core processing on very large meshes ($>$2M faces).

### 5.4 Low-Rank Compression

Two compression methods are available:

**Adaptive Cross Approximation (ACA)** — default, 50–100$\times$ faster than SVD. Builds the low-rank factorization directly by greedily sampling rows and columns:

1. Start with pivot row $i_0 = 0$.
2. For rank $k = 0, 1, \ldots$:
   - Compute row $i$ of the view factor subblock.
   - Subtract the current rank-$k$ approximation.
   - Find the pivot column $j$ with maximum residual.
   - If the residual $|r_{ij}| < \tau_{\text{ACA}} \|F\|$ or $|r_{ij}| < 10^{-15}$: stop (converged).
   - Compute column $j$, subtract current approximation, add rank-1 update.
3. Result: rank-$k$ factorization with $O(k(m+n))$ storage.

Configuration: `aca_tolerance` (default $10^{-3}$), `aca_max_rank` (default 50), `aca_min_block_size` (default 100; smaller blocks use SVD).

**SVD** — fallback for small blocks or when ACA fails. Computes the full subblock, then truncates via `Eigen::JacobiSVD` (or randomized SVD with oversampling parameter $p = 10$ and $q = 2$ power iterations) at tolerance `hmatrix_tolerance` (default $10^{-2}$), retaining at most `hmatrix_max_rank` (default 50) singular values.

### 5.5 Serialization

H-matrices are serialized to disk in a custom binary format:
- **`.hmat`** files, magic number `0x484D4154`.
- Stores the hierarchical block structure, low-rank factors, and sparse blocks.
- Loaded by `RadiationExchangeManager::loadViewFactors()`, which auto-detects the format.

Sparse view factor matrices use a separate format:
- **`.vfm`** files, magic number `0x5646464D` ("VFFM").
- CSR data arrays stored directly.

Both formats are auto-detected on load.

### 5.6 Memory-Mapped Loading

For very large pre-computed H-matrices (meshes $>$500K faces), the module supports memory-mapped (mmap) loading. Blocks are loaded on-demand during `matvec` operations rather than reading the entire file into RAM.

## 6. Radiosity Solver

The radiosity equation $(I - \rho F)\mathbf{B} = \mathbf{E} + \rho \mathbf{Q}_{\text{solar}}$ is solved iteratively.

### 6.1 Jacobi Iteration

Used when the view factor operator is an H-matrix (no efficient row access) or when Gauss-Seidel acceleration is disabled:

$$\mathbf{B}^{(k+1)} = \mathbf{E} + \rho\, \mathbf{Q}_{\text{solar}} + \rho\, F\, \mathbf{B}^{(k)}$$

initialized with $\mathbf{B}^{(0)} = \mathbf{E} + \rho\,\mathbf{Q}_{\text{solar}}$ (no inter-face contribution). The core operation each iteration is a single matrix-vector product $F \mathbf{B}^{(k)}$, which is $O(N \log N)$ for the H-matrix or $O(\text{nnz})$ for the sparse matrix (nnz = the number of stored nonzero entries).

### 6.2 Gauss-Seidel Iteration

Used when the view factor matrix is sparse CSR (row access is efficient) and `use_acceleration` is enabled (default):

$$B_i^{(k+1)} = E_i + \rho_i Q_{\text{solar},i} + \rho_i \sum_j F_{ij}\, B_j^{(*)}$$

where $B_j^{(*)}$ uses the most recently updated value of $B_j$ (updated values from earlier in the same sweep for $j < i$, values from the previous iteration for $j > i$). This typically converges faster than Jacobi because updated values propagate within a single iteration.

### 6.3 Convergence

The solver stops when either:

$$\frac{\max_i |B_i^{(k+1)} - B_i^{(k)}|}{\max_i |B_i^{(k+1)}|} < \tau_{\text{solver}}$$

(default $\tau_{\text{solver}} = 0.01$), or the maximum iteration count is reached (default 100).

Convergence diagnostics are available via `getLastIterations()`, `getLastResidual()`, and `didConverge()`.

### 6.4 Energy Balance Verification

After solving, the module can verify global energy conservation:

$$\text{imbalance} = \left| \sum_i \varepsilon_i \sigma T_i^4 A_i - \sum_i (\varepsilon_i Q_{\text{solar},i} + Q_{\text{indirect},i}) A_i \right|$$

This check is available via `RadiositySolver::checkEnergyBalance()`.

### 6.5 GPU Acceleration (OpenCL)

The radiosity solver can be offloaded to GPU via OpenCL (`--rad-use-gpu`). The GPU solver uses:

- **CSR sparse matrix-vector multiply (SpMV)** for the $F\mathbf{B}$ product.
- **Parallel Jacobi iteration** with convergence checks every 10 iterations.
- **Warm start**: reuses the previous solution as the initial guess.

Kernels: `spmv_kernel`, `emission_kernel` ($\varepsilon \sigma T^4$), `solar_source_kernel` ($\rho Q_{\text{solar}}$), `jacobi_kernel`, `indirect_flux_kernel`. Validated to $< 0.001\%$ relative error vs. the CPU solver.

## 7. Thermal-Radiation Coupling

The radiation exchange and thermal conduction are coupled through an outer iteration loop. Because surface temperatures depend on indirect flux and indirect flux depends on surface temperatures (via the radiosity equation), the system must be solved iteratively.

### 7.1 Coupling Algorithm

Starting from the surface temperatures $T_i(t)$ computed by the initial thermal solve (without indirect flux):

**For each coupling iteration $k = 1, 2, \ldots, k_{\max}$:**

1. **Compute time-varying indirect flux.** For each timestep $t = 1, \ldots, N_t$:
   - Extract current surface temperatures $\{T_i(t)\}$ and direct solar flux $\{Q_{\text{solar},i}(t)\}$.
   - Solve the radiosity equation to obtain $Q_{\text{indirect},i}(t)$.
   - Store in the array `indirect_flux_series[i][t]`.

2. **Identify faces needing thermal re-run.** Two modes (configured via `--rad-coupling-select-mode`):

   - **`physics`** (default): linearized ΔT predictor. For each face $i$:
     $$\widehat{\Delta T}_i = \frac{(1-\alpha_i)\,\Delta F_i}{4\,\varepsilon\,\sigma\,T_{\text{eff},i}^3},\qquad
       T_{\text{eff},i} = \max\bigl(\max_t T_i(t),\;40\ \text{K}\bigr)$$
     where $\Delta F_i = \max_t Q_{\text{indirect},i}^{(k)}(t) - \max_t Q_{\text{indirect},i}^{(k-1)}(t)$ on iter $k>0$, or $\max_t Q_{\text{indirect},i}^{(0)}(t)$ on iter 0. Include face $i$ if $\widehat{\Delta T}_i \geq \tau_T \cdot s$, where $s$ is the safety factor (default 0.5). Also include face $i$ if its observed $\max_t |T_i^{(k)}(t)-T_i^{(k-1)}(t)| \geq \tau_T \cdot s$ (per-face freeze: if the face is still moving from the previous re-thermal, recompute it; once it has stopped *and* its flux is stable, drop it).
   - **`legacy`**: include face $i$ if $\max_t Q_{\text{indirect},i}(t) > \tau_Q$ (default 0.1 W/m$^2$). Reproduces pre-2026-05-09 behavior byte-for-byte.

3. **Check convergence** (for $k > 1$). Compute:
   - $\Delta T_{\max} = \max_{i,t} |T_i^{(k)}(t) - T_i^{(k-1)}(t)|$
   - In `physics` mode, $f_{\text{changed}} = \text{count}\bigl(\max_t|\Delta T_i| > \tau_T\bigr)\,/\,N$ (true face-fraction).
   - In `legacy` mode, $f_{\text{changed}} = \text{count}(|\Delta T| > \tau_T)\,/\,(N\cdot N_t)$ (sample-fraction, unchanged from pre-fix).

   Converged when **either** $\Delta T_{\max} < \tau_T$ **or** $f_{\text{changed}} < f_{\text{threshold}}$ (changed from AND to OR in commit 807d374; the AND-conjunction blocked exit when a handful of pathological PSR-floor edges oscillated by hundreds of K even after 99.99% of the mesh had settled).

4. **Re-run thermal model** for selected faces with the combined time-varying flux:

   $$Q_{s,i}(t) = \bigl[Q_{\text{solar},i}(t) + Q_{\text{refl},i}(t)\bigr](1 - \alpha_i) + Q_{\text{IR},i}(t)\,\varepsilon_i + Q_{\text{indirect},i}(t)$$

   Each face's 1D thermal profile is re-solved over all timesteps using this updated flux. The previous iteration's temperature profile is used as a warm start.

5. **Update** $T_i(t)$ and return to step 1.

### 7.2 Convergence Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| $k_{\max}$ | `--rad-coupling-iter` | 10 | Maximum coupling iterations |
| $\tau_T$ | `--rad-coupling-temp-tol` | 0.5 K | Temperature change threshold |
| $\tau_Q$ | `--rad-coupling-flux-tol` | 0.1 W/m$^2$ | Indirect flux threshold (legacy mode only) |
| $f_{\text{threshold}}$ | `--rad-coupling-face-frac` | 0.01 (1%) | Convergence threshold on faces (physics) or samples (legacy) still changing |
| select mode | `--rad-coupling-select-mode` | `physics` | Face-selection criterion: `physics` (default) or `legacy` |
| $s$ | `--rad-coupling-select-safety` | 0.5 | Safety factor on the predicted-ΔT threshold — **`coupling_mode: legacy` only** |
| coupling mode | `--rad-coupling-mode` | `bounded` | `bounded` \| `fixed` \| `legacy` — see §7.2.1 |
| exchange selection | `--rad-exchange-selection` | `all` | `all` \| `shadow` — see §7.2.2 |
| $\tau_{\text{sel}}$ | `--rad-coupling-select-dt` | 0 K | Absolute predicted-ΔT selection threshold, `bounded`/`fixed` only |

In `bounded` (default) the loop stops when the max per-face temperature change falls
below $\tau_T$, or at $k_{\max}$. In `legacy` it also stops when the *fraction* of
faces still changing falls below $f_{\text{threshold}}$ (either/or, changed from AND
in May 2026). In `fixed` it always runs $k_{\max}$ iterations.

### 7.2.1 Chunk-decomposition independence (PTM-89)

Every input to the shipped (pre-2026-08-05) selection and exit decisions is a maximum
**over the output window**: $\max_t Q_{\text{indirect},i}$, $\max_t T_i$,
$\max_{i,t}|T^k - T^{k-1}|$, and the count of faces whose window-max change exceeds
$\tau_T$. A half-length window therefore scores every face lower than the full window,
selects fewer faces, and can stop an iteration earlier — so **the converged field
depended on where the chunk boundaries fell**. Two mechanisms, both real:

1. **Selection.** With $\tau_T\cdot s = 1$ K, a face whose window-max indirect flux
   sits near the threshold is re-thermal'd in a 60-step window and frozen in the
   30-step half. Measured on the 12 800-face `lunar_crater_80N` gate (60 frames,
   29.53 d, seam at frame 30): the single-shot selected 2578/2578/2547 faces and the
   first chunk 2559/2559/2380, for a peak single-shot-vs-2-chunk difference of
   **41.4 K**, 28 faces above 1 K.
2. **Exit.** The face-fraction criterion has **no error bound** — it stops while up to
   $f_{\text{threshold}}$ of the mesh is still moving by an arbitrary amount — and its
   count is also a window maximum, so a short window reaches it first.

`bounded` removes both: the selection threshold becomes the absolute
$\tau_{\text{sel}}$ (default 0, i.e. re-thermal every face carrying any indirect flux)
and the face-fraction exit is disabled. On the same gate, the first chunk then becomes
**bit-identical** to the single-shot over all 30 × 12 800 samples, and the whole-run
residual falls to **0.014 K** — below the **0.029 K** radiation-free chunking floor set
by the `float32` 30-layer checkpoint. Cost on that gate: +2.4 % face-re-thermals
(7703 → 7889) and no wall-time change outside run-to-run noise.

`fixed` additionally disables the early exit on $\tau_T$, because *any* data-driven
stopping rule is a function of the window. It is the only mode whose iteration count
cannot depend on the decomposition by construction; on the gate it costs +70 %
face-re-thermals (7703 → 13 141, 3 iterations → 5) for the same 0.014 K residual, so
`bounded` is the default and `fixed` is for when the iteration counts must be provably
equal.

`legacy` reproduces pre-PTM-89 behavior bit-for-bit (verified: byte-identical output
trees against the `8fbedd4` build on the same gate, coupled and radoff, single-shot and
2-chunk). Use it to reproduce a published run, and only for that: it re-introduces the
defect. It can be selected without editing a config via `PTM_COUPLING_MODE=legacy`.

Setting $\tau_{\text{sel}} > 0$ trades accuracy for speed on very large meshes and
re-introduces the dependence at up to that magnitude per face; the run warns when it
is set, and warns that `coupling_face_fraction` is inert, rather than ignoring either
in silence.

**Not covered by the measurement above:** the Moon `lunar_crater_80N` gate only. The
ratio (faces carrying any indirect flux)/(faces above the 1 K predicted-ΔT threshold)
is a property of the mesh — 1.02–1.05 here — and has **not** been measured on the
140 k-face `fig8_hybrid` production mesh, where the shipped threshold selected 48 % of
faces. The worst case is a full-mesh re-thermal per iteration, i.e. ≤ 2.1× the
re-thermal component of the coupling stage on that mesh.

### 7.2.2 The radiation-exchange face set (PTM-98)

`RadiositySolver::solve()` returns the absorbed indirect flux $Q_{\text{indirect},i}$
for **every** face — the classification never enters the solve. The *exchange set* is
applied afterwards, as a copy mask deciding which of those computed values survive
into the thermal step. Everything it excludes is silently reset to exactly
0 W/m$^2$.

| knob | flag | default | meaning |
|---|---|---|---|
| exchange selection | `--rad-exchange-selection` | `all` | `all` \| `shadow` |
| $f_{\text{shadow}}$ | `--rad-min-shadow-fraction` | 0.3 | `shadow` mode only |

The pre-2026-08-05 gate (`shadow`) kept a face only if it was in shadow for at least
$f_{\text{shadow}}$ of the run's **own illumination series**, whose length is

$$N = n_{\text{equilibration steps}} + n_{\text{output steps}}$$

and a **warm-started chunk has no equilibration steps at all**. So the criterion is a
function of the chunk plan. Chunk 0 is accidentally protected — its ~120 equilibration
samples span several diurnal cycles, so nearly every face clears 0.3 — while every
resumed chunk is scored on its output window alone. A chunk that happens to fall in
local daytime therefore reclassifies faces *out of radiation exchange entirely*.

Measured on the 12 800-face `lunar_crater_80N` gate (60 frames, 29.53 d), per-face
$|T_{\text{chunked}} - T_{\text{single}}|$ over all frames:

| decomposition | `shadow` (pre-fix) | `all` (default) | `all` + `coupling_mode: fixed` | radoff null |
|---|---|---|---|---|
| 3 × 20 frames | **82.383 K**, 6367 faces > 1 K | 0.081 K, 0 > 1 K | **0.011 K** | 0.040 K |
| 6 × 10 frames | **82.383 K**, 26 063 faces > 1 K | 0.081 K, 0 > 1 K | **0.015 K** | 0.040 K |
| 2 × 30 frames | clean | clean | clean | 0.029 K |

In the 3-chunk case the last chunk kept only 2265 of 12 800 faces, and **361 faces
received exactly 0 W/m$^2$ in both bands** where the single-shot gives 17.5 W/m$^2$
mean / 63.0 W/m$^2$ max of thermal IR. Note the 2-chunk row: a 30-frame window still
spans enough night for every face to clear the gate, so **a 2-chunk test cannot detect
this** — decomposition independence must be certified at ≥ 3 chunks.

`all` (default) keeps every face's computed flux. This is the only setting with a
physical justification: the radiosity solution is a self-consistent global linear
solve, so discarding part of its output is data loss, not an approximation. It is also
free — the exchange set never gated the solve, only the copy. Measured on the affected
chunk: radiosity sweeps unchanged (IR = 2, visible = 2), re-thermal 2265 faces/51 ms →
2626 faces/52 ms, wall time 1.776 s → 1.791 s (+0.8 %, inside the run-to-run spread)
for a 5.65× larger exchange set. The exact upper bound is the equivalent single-shot's
cost, because the set is now window-independent and identical to it.

Correcting the mask **also changes single-shot results** for any window that was being
truncated (a short warm-started window, or `equilibration_orbits: 0`): the previously
zeroed faces warm, by construction. On a 10-frame example, 284 faces recovered
12.3 W/m$^2$ mean IR and warmed by +2.84 K mean / +60.2 K max, and 0 of 2840 masked
face-frames got colder — restoring a positive absorbed flux can only warm.

The residual 0.081 K under `all` is **not** this defect: it is the PTM-89 exit
criterion (§7.2.1), which is still a window maximum in `bounded` — the 20-frame chunk
exits after 2 Picard sweeps where the single-shot takes 3. `coupling_mode: fixed`
removes it, giving the 0.011 K figures above.

`shadow` reproduces the pre-fix behavior bit-for-bit (verified byte-identical output
trees, coupled and radoff, single-shot and 3-chunk) and is provided only to reproduce
published runs; pair it with `coupling_mode: legacy`. It warns, naming the number of
faces it is about to zero. `min_shadow_fraction` was parsed but never bridged into
`RadiationConfig` before this change (PTM-99), so the hard-coded 0.3 always won
silently; it is now plumbed and warns when set in `all` mode, where it is inert.
Selectable without editing a config via `PTM_EXCHANGE_SELECTION=shadow`.

**Not covered:** the Moon `lunar_crater_80N` gate only, GPU re-thermal path only, no
non-Moon body, no CPU coupled path, and no production-mesh (140 k/509 k-face) cost
measurement. The Smith beamed-emission receiver and emitter sets (`rms_slope_deg > 0`)
are still selected from window-scoped shadow fractions and remain
decomposition-dependent; beaming is off by default and the run now warns when it is on.

### 7.3 Optimizations

- **Physics-based selective thermal re-run** (default): only faces whose linearized radiative response predicts $\widehat{\Delta T}\geq \tau_T\cdot s$ are re-solved (or whose observed $\Delta T$ from the previous iter still exceeds that threshold). This naturally handles cold PSR floors (where the response is amplified by $T^{-3}$) and warm slopes asymmetrically, and freezes faces that have already converged.
- **Legacy absolute-flux selection** (`--rad-coupling-select-mode legacy`): include faces where $\max_t Q_{\text{indirect},i}(t) > \tau_Q$. Reachable for byte-equivalent reproductions of pre-fix runs.
- **Warm-start thermal solver**: each coupling iteration initializes from the previous iteration's temperature profile, reducing the number of time steps needed to reach the new equilibrium.
- **Per-face albedo matching**: the coupling loop uses the same per-face albedo maps as the initial thermal solve, loaded from binary files, ensuring consistency.

## 7.5 Validation: Bowl Crater (Ingersoll 1992)

The radiation exchange module is validated against the analytic two-zone bowl-crater model of Ingersoll, Svitek & Murray (1992) using the standalone test program `bin/test_bowl_crater`. The crater is a spherical cap with depth-to-diameter ratio $d/D \in \{0.1, 0.2, 0.3\}$, discretized into 1404 triangular faces (20 radial rings × 36 azimuthal divisions). The test exercises the full view-factor and radiosity-solver pipeline at four crater latitudes ($0°$, $70°$, $80°$, $85°$).

Figure C.1 summarizes the validation. Panel (a) shows the geometry and the rim-shadow critical solar elevation $\beta_c = \arctan(2 d/D)$. Panel (b) shows the analytic Ingersoll equilibrium floor temperature $T_f(\beta)$ for the three depth ratios using Moon defaults $A = 0.12$, $\varepsilon = 0.95$, $T_w = 200$ K, recovering the characteristic shoulder at $\beta = \beta_c$ above which the floor sees direct sun. Panel (c) compares PTM3D's per-face floor sky view factor $F_\text{sky}$ against the analytic spherical-cap formula $\sin^2\alpha$: the four latitudes agree with the analytic mean (0.8621) to within 0.5 %, with a per-mesh-face range of $\pm0.4 \%$ around the mean. Panel (d) reports the radiosity solver diagnostics: view-factor density 99.86 %, reciprocity test passed for all 1.97 M significant pairs, CPU radiosity solver converging in 2 iterations to residual $\sim 5 \times 10^{-5}$, and GPU vs CPU agreement of 0.0001 % relative error.

![Figure C.1: Bowl-crater radiation exchange validation](figures/v2/bowl_crater_ingersoll.png)

## 8. Face Classification

Not all mesh faces require radiation exchange. The `FaceSelector` classifies faces based on:

- **Shadow fraction**: faces that spend $> 30\%$ of the diurnal cycle in shadow (configurable via `min_shadow_fraction`) are candidates, as they likely reside in topographic concavities where inter-face exchange is significant.
- **Geometric classification**: based on face normals and local topography.

The classification stores per-face shadow fractions, minimum temperatures, and a boolean `needs_exchange` flag. Only faces flagged for exchange participate in view factor computation and radiosity solving.

## 9. Module Architecture

### 9.1 Key Classes

| Class | Header | Role |
|-------|--------|------|
| `RadiationExchangeManager` | `radiation_exchange.h` | Top-level coordinator: face classification, view factor computation, radiosity solving |
| `RadiositySolver` | `radiation_exchange.h` | Solves the radiosity equation; supports sparse, H-matrix, and GPU backends |
| `ViewFactorCalculator` | `view_factor.h` | Computes view factors using Embree; adaptive M1/M3 methods |
| `ViewFactorMatrix` | `view_factor.h` | CSR sparse storage with `matvec`, row access, and binary I/O |
| `HMatrixViewFactor` | `hmatrix_view_factor.h` | Hierarchical matrix with octree partitioning, ACA/SVD compression |
| `IViewFactorOperator` | `radiation_exchange.h` | Abstract interface for matrix-vector product (bridges sparse and H-matrix) |
| `FaceSelector` | `radiation_exchange.h` | Classifies faces by shadow fraction and geometry |
| `RadiationConfig` | `radiation_config.h` | All configuration parameters |
| `FaceClassification` | `radiation_config.h` | Per-face classification results |
| `RadiosityOpenCLSolver` | `gpu/radiosity_opencl.h` | GPU-accelerated Jacobi solver (OpenCL) |

### 9.2 Data Flow

```
IlluminationModel
  ├── Q_solar(i,t)     ──┐
  ├── Q_refl(i,t)       │
  └── Q_IR(i,t)         │
                         ▼
              calculateAbsorbedFlux()
                         │
                         ▼
              Initial thermal solve ──► T_i(t)
                         │
          ┌──────────────┘
          ▼
  Radiation-thermal coupling loop:
          │
          ├─► RadiositySolver::solve(T, Q_solar)
          │       │
          │       ├── E_i = ε_i σ T_i⁴
          │       ├── (I - ρF) B = E + ρ Q_solar
          │       └── Q_indirect_i = ε_i Σ_j F_ij B_j
          │                │
          ├─► Select faces ◄┘
          │
          ├─► Re-solve thermal with Q_s + Q_indirect
          │
          └─► Convergence check ──► done / next iteration
```

## 10. Command-Line Options

```
--radiation-exchange           Enable radiation exchange
--rad-hmatrix                  Use H-matrix compression for view factors
--rad-use-gpu                  Use GPU radiosity solver (requires OpenCL build)
--rad-verbose                  Print detailed progress information
--rad-view-factor-file FILE    Save/load pre-computed view factors
--rad-coupling-iter N          Max coupling iterations (default: 10)
--rad-coupling-temp-tol K      Temperature convergence threshold in K (default: 0.5)
--rad-coupling-flux-tol W      Flux change threshold in W/m² (legacy mode only; default: 0.1)
--rad-coupling-face-frac F     Face fraction still changing to stop (default: 0.01)
--rad-coupling-select-mode M   Selection mode: 'physics' (default) or 'legacy'
--rad-coupling-select-safety F Safety factor on predicted-ΔT threshold (default: 0.5)
--rad-aca-tolerance T          ACA relative convergence tolerance (default: 1e-3)
--rad-aca-max-rank R           Maximum ACA rank before fallback (default: 50)
--rad-hmatrix-threads N        Limit H-matrix build threads (0 = auto)
```

## 10.9 Radiance-weighted view-factor pruning (adjustable, default OFF)

The corrected per-partner sparse matrix (CSR format, §4) is large because a face
couples to many partners, most contributing a tiny view factor. A *geometry*
cutoff (drop small `F_ij`, or
drop by distance) is physically wrong — a large, directly-lit distant wall can
matter more than a small nearby slope. The principled cutoff is on **radiative
contribution**, weighting each view factor by the radiance the partner emits:

> keep partner `j` for receiving face `i` iff `F_ij · R_j ≥ ε` (W/m²), where
> `R_j = σ·T_max,j⁴ + A₀·direct_solar_max,j` — the max IR + reflected-solar
> radiance `j` presents over the cycle.

Cold faint partners (PSR walls, `σT⁴ ≈ 1.4 W/m²` at 70 K) fall away; warm faint
partners (sunlit rims, `≈ 850–1200 W/m²`) survive — the all-scales physics, with
the illumination pattern in the weight. This is **provably optimal** for a given
number of retained matrix entries: it drops exactly the lowest-contribution
ones. True `F` values are kept (no renormalization); the discarded contribution
is bounded and **reported**.

**Where it lives.** Assembly-time, in `scripts/ptm12_diag/assemble_vf_haworth.py`
(the sim is unchanged — it just loads whatever CSR it is given):

- `--radiance-prune-eps ε` — threshold in W/m². **`0.0` (default) = no pruning;
  the output is numerically identical to the last digit.** Also settable via the
  environment variable `PTM_RADIANCE_PRUNE_EPS`, and documented in the radiation
  config as `radiation.radiance_prune_eps` (the runners propagate it to the
  assembler; the sim parses and **logs** it for provenance but does not prune —
  see "Config key and precedence" below).
- `--radiance-weight PATH` — the per-face `R` source, **two documented paths**:
  1. a **prior-run stats directory** (`temperature_max.bin` +
     `direct_solar_max.bin`) — exact radiance from a completed coupled/radoff run;
  2. a **`.npy`/`.bin` of per-face `R`** — a cheap *illumination-proxy* built
     without a full thermal run (illumination-only pass → `direct_solar_max`,
     `T_eq = ((1−A)·direct_solar_max/εσ)^¼`, `R ≈ σT_eq⁴ + A₀·direct_solar_max`).
- `--radiance-albedo A₀` — visible-band albedo (default 0.12).

**Config key and precedence.** The same threshold is exposed as
`radiation.radiance_prune_eps` (float, default `0.0` = off) in the sim config
(parsed in `config_parser.cpp`, stored on the radiation config, distinct from the
build-time `visibility_threshold` storage floor). The sim does **not** prune — it
logs the effective value at radiation init (`radiance-prune epsilon: X W/m²
(assembly-time; 0=off)`) for provenance. The runners (e.g.
`scripts/run_yearlong_wedge_safe.py`) read the config key and export it as
`PTM_RADIANCE_PRUNE_EPS` for the VF-assembly step they invoke. Precedence is
**env overrides config**: an already-set `PTM_RADIANCE_PRUNE_EPS` is left untouched
(it wins), and the runner fills the env from the config only when it is unset; a
config value of `0.0` exports nothing, so the default path is byte-identical. When
a `<csr>.prune_receipt.json` sidecar accompanies a `PTM_HMATRIX_VF_REPLACE` CSR,
the sim additionally logs a one-line summary of that receipt (eps + max discarded
contribution).

**Mandatory receipt.** Every pruned assembly prints and writes
`<out>.csr.prune_receipt.json`: matrix entries kept/dropped, size, predicted speedup, the
per-receiving-face **discarded contribution** (W/m², max/p99/mean) and its
**ΔT bound** `ΔT ≈ dQ/(4σT³)` at the face's own temperature — *so the user sees
the trade before running.* The ΔT bound is a **loose upper bound** (it uses the
partner's *max* radiance, timing-decoupled), so the measured ΔT is smaller — but
"smaller than a large loose bound" is not "safe," which is why a real window test
governs adoption.

**Measured counter-example — AGT / MP026 (the cautionary receipt).** On the
diffuse South-Pole terrain, `ε = 1e-3` gives **3.2× fewer stored matrix entries** (and ~2.0×
radiosity wall-time), but the Y00_summer window **FAILED** the < 1 K bar against
the full-fidelity baseline: per-face `T_max` off by **mean 3.3 K / max 22.6 K on
PSR floors** (n≈101 k faces < 120 K), while warm rims passed (< 0.6 K). The floors
are radiation-dominated and heated by the *aggregate* of many faint partners, so
dropping them cools the coldest, most science-critical faces. (Evidence lives in
the internal A3GT campaign archive; the numbers above are from its
`verdict.json`. Any pruned assembly reproduces the same accounting in its own
`<out>.csr.prune_receipt.json` — the receipt format described earlier in this
section.) **Guidance:** pruning does **not** pay
on diffuse polar terrain; it pays on **isolated-crater or warm-body** terrain
where a face's coupling is dominated by a few high-radiance partners. Default OFF
for all sims; enable only with a per-terrain window test and the receipt in hand.

## 11. References

- Potter, S. F., Bertone, S., Schörghofer, N., & Mazarico, E. (2023). Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces. *Journal of Computational Physics: X*, 17, 100130. https://doi.org/10.1016/j.jcpx.2023.100130
- Rezac, L. & Zhao, Y. (2020). [View factor computation methods for thermal modeling of planetary surfaces.]
- Ingersoll, A. P., Svitek, T., & Murray, B. C. (1992). Stability of polar frosts in spherical bowl-shaped craters on the Moon, Mercury, and Mars. *Icarus*, 100, 40–47. [Validation test case for the module: `test_bowl_crater`.]
- Hayne, P. O., et al. (2017). Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 122, 2371–2400. [Incidence-angle-dependent albedo parametrization.]
