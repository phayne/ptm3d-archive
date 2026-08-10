# Thermal Conduction Model

## 1. Overview

The PTM3D thermal model solves the one-dimensional vertical heat conduction equation independently at every triangular mesh face, using temperature- and depth-dependent material properties and a nonlinear radiative surface boundary condition. The solver is fully decoupled from the [illumination model](illumination-model.md) and the [radiation exchange module](radiation-exchange.md) through a single prescribed time series of absorbed surface flux $Q_s(t)$ per face: the illumination module supplies direct solar, primary-body reflected, and primary-body thermal IR fluxes; the radiation exchange module adds the indirect (multi-bounce) flux when enabled. Lateral conduction between faces, phase change, and advective transport are not represented — face-to-face energy exchange happens entirely through the radiation exchange module, not through the thermal solver.

The model was originally developed for the lunar regolith (Hayne et al., 2017) and extended to icy bodies. Numerical implementation choices are guided by the unconditional stability of the Crank–Nicolson scheme on the implicit pentadiagonal system that arises from non-uniform vertical grids.

## 2. Governing Equation

At each face, the temperature $T(z, t)$ in the subsurface satisfies

$$\rho(z)\thinspace c_p(T)\thinspace \frac{\partial T}{\partial t}  =  \frac{\partial}{\partial z}\negthinspace\left[\thinspace k(T, z)\thinspace \frac{\partial T}{\partial z} \thinspace\right] \tag{B1}$$

where $\rho$ is bulk density (kg m⁻³), $c_p$ is specific heat capacity (J kg⁻¹ K⁻¹), $k$ is bulk thermal conductivity (W m⁻¹ K⁻¹), $z$ is depth below the surface (m, positive downward), and $t$ is time (s). The PDE is parabolic and quasi-linear: nonlinearity enters only through the temperature dependence of $c_p$ and $k$. There is no internal volumetric source term; all heat input is supplied through the boundary conditions (§5).

The 1D-vertical assumption is justified because the diurnal thermal skin depth (§4.1) is small compared with the lateral scale of mesh faces in every application reported here ($\delta_1 \lesssim 10$ cm on the Moon and $\lesssim 5$ cm on Europa, vs. face edges $\gtrsim 1$ km on global meshes and $\gtrsim 1$ m on regional topographic meshes), so horizontal gradients of $T$ within a single face are negligible. Sublimation- and sputtering-driven mass flux is treated separately by the exosphere model and feeds back into the thermal model only through albedo and thermal-property updates between epochs.

## 3. Thermophysical Properties

### 3.1 Thermal Conductivity

We adopt the Hayne et al. (2017) formulation, which augments a depth-stratified lattice (contact) conductivity with a $T^3$ radiative term that approximates radiative heat transfer through pore spaces:

$$k(T, z)  =  k_c(z)\thinspace\bigl[\thinspace1 + \chi\thinspace(T / 350\thickspace\text{K})^3\thinspace\bigr] \tag{B2}$$

where $k_c(z)$ is the contact conductivity (§3.3) and $\chi$ is a dimensionless radiative parameter. We use $\chi = 2.7$ following Mitchell & de Pater (1994). The reference temperature 350 K is the value at which the formula was calibrated against laboratory measurements; above ~250 K the radiative contribution exceeds the lattice term, while at lunar nighttime temperatures (~100 K) it is negligible. Equation (B2) is evaluated as $k = k_c(1 + R_{350} T^3)$ with the precomputed constant $R_{350} = \chi/350^3 \approx 6.3 \times 10^{-8}$ K⁻³.

### 3.2 Heat Capacity

Heat capacity is evaluated as a polynomial in temperature,

$$c_p(T)  =  c_0  +  \sum_{j=1}^{N_c} c_j\thinspace T^j \tag{B3}$$

stored as a constant term $c_0$ plus an ordered coefficient list $\lbracec_j\rbrace$ in the body's `ThermophysicalProperties` record (`planetary_properties.h`). The polynomial is evaluated by Horner's method to reduce $2N_c$ multiplications to $N_c$ per call; in the inner thermal loop this is vectorized with AVX-512 or AVX2 fused multiply-add instructions where available, with a scalar fallback (`thermal_model.cpp:74–170`). For the Moon we use the four-term polynomial of Hayne et al. (2017), Table A1; for water-ice surfaces we use the linear approximation $c_p(T) \approx 7.49\thinspace T$ (J kg⁻¹ K⁻¹) derived from the heat-capacity data of Giauque & Stout (1936), valid below $\sim$270 K.

### 3.3 Density and Contact-Conductivity Stratification

Both the bulk density and the contact conductivity decay exponentially from surface to deep values with a common e-folding scale height $H$:

$$\rho(z)  =  \rho_d  +  (\rho_s - \rho_d)\thinspace \exp(-z / H), \tag{B4}$$

$$k_c(z)  =  k_d  +  (k_s - k_d)\thinspace \exp(-z / H). \tag{B5}$$

At $z = 0$, $\rho = \rho_s$ and $k_c = k_s$ (the porous, low-conductivity surface values); as $z \to \infty$, $\rho \to \rho_d$ and $k_c \to k_d$ (the compacted, higher-conductivity deep values). This two-parameter form follows Hayne et al. (2017) for the Moon and is retained for icy bodies with body-specific $\lbrace\rho_s, \rho_d, k_s, k_d, H\rbrace$.

### 3.4 Per-Face Property Maps

Body-uniform defaults can be overridden per face through a `ThermalPropertyMap` (`thermal_property_map.h`), which supports spatially varying thermal inertia, scale height $H$, and Bond albedo, sampled from FITS-format observational maps (e.g., the Diviner-derived $H$-parameter map of Hayne et al., 2017). When a map is supplied, each face's property is interpolated bilinearly from its centroid latitude/longitude.

### 3.5 Body-Specific Defaults

The defaults adopted in this work are listed below. Moon values reproduce Hayne et al. (2017), Table A1; the Europa values are appropriate for porous water-ice regolith and have been used throughout this paper.

| Symbol | Quantity | Moon | Europa | Units | Source |
|---|---|---|---|---|---|
| $\rho_s$ | Surface density | 1100 | 500 | kg m⁻³ | Hayne et al. (2017); model default |
| $\rho_d$ | Deep density | 1800 | 700 | kg m⁻³ | Hayne et al. (2017); model default |
| $H$ | Density/conductivity scale height | 0.06 | 0.07 | m | Hayne et al. (2017); model default |
| $k_s$ | Surface contact conductivity | $7.4 \times 10^{-4}$ | $9.6 \times 10^{-3}$ | W m⁻¹ K⁻¹ | Hayne et al. (2017); $TI \approx 60$ tiu fit |
| $k_d$ | Deep contact conductivity | $3.4 \times 10^{-3}$ | $5.0 \times 10^{-2}$ | W m⁻¹ K⁻¹ | Hayne et al. (2017); model default |
| $c_p$ coefficients | $\lbracec_0; c_1, c_2, c_3, c_4\rbrace$ | $\lbrace-3.6125;\thinspace 2.7431,\thinspace 2.36 \times 10^{-3},\thinspace -1.234 \times 10^{-5},\thinspace 8.91 \times 10^{-9}\rbrace$ | $\lbrace0;\thinspace 7.49\rbrace$ | (J kg⁻¹ K⁻¹ per power) | Ledlow et al. (1992) via Hayne et al. (2017); Giauque & Stout (1936) |
| $\chi$ | Radiative conductivity parameter | 2.7 | 2.7 | – | Mitchell & de Pater (1994) |
| $\varepsilon$ | Broadband IR emissivity | 0.95 | 0.90 | – | Hayne et al. (2017); model default |
| $A$ | Bond albedo | 0.12 | 0.50 | – | Hayne et al. (2017); model default |
| $Q_b$ | Internal heat flux | 0.018 | 0.030 | W m⁻² | Langseth et al. (1976); model default |

The Moon entry additionally carries an incidence-angle-dependent albedo correction $A(\theta) = A_0 + a(\theta/(\pi/4))^3 + b(\theta/(\pi/2))^8$ with $\lbracea, b\rbrace = \lbrace0.06, 0.25\rbrace$ (Hayne et al., 2017). For Europa we use a constant Bond albedo, optionally overridden per face by a high-resolution albedo map (see [illumination-model.md](illumination-model.md)).

## 4. Spatial Discretization

### 4.1 Skin Depth and Grid Scaling

The diurnal thermal skin depth is

$$\delta_1  =  \sqrt{\thinspace\kappa\thinspace P / \pi\thinspace} \tag{B6}$$

where $\kappa = k / (\rho c_p)$ is the thermal diffusivity and $P$ is the diurnal period. $\delta_1$ is the depth at which the diurnal temperature oscillation has decayed by a factor of $e$. In practice $\delta_1$ is evaluated at the surface using $k_s$, $\rho_s$, and a representative $c_p(200~\text{K})$ to set grid scales (`thermal_model.cpp:647–658`).

### 4.2 Geometric Depth Grid

The default depth grid is geometrically stretched: the uppermost layer thickness is set by a target resolution of the skin depth, and successive layers grow by a fixed multiplicative factor:

$$\Delta z_0 = \delta_1 / m, \qquad \Delta z_{i+1} = \Delta z_i\thinspace(1 + 1/n), \qquad z_{\max} = b\thinspace \delta_1 \tag{B7}$$

The defaults $m = 4$, $n = 10$, $b = 5$ were selected by sensitivity analysis against Diviner observations (Hayne et al., 2017), giving an RMS surface-temperature error below 4 K with a systematic offset of $+2.65$ K relative to a much finer reference grid.

### 4.3 Adaptive Depth Grid

When `ADAPTIVE_GRID = true` (default), the grid is reconstructed per face from the local diffusivity profile $\kappa(z)$. The number of layers per skin depth decays with depth following an e-folding scale of `RES_DECAY_SKINS = 3.0` (in skin depths), with a floor of `MIN_LAYERS_DEEP_PER_SKIN = 1` and a per-step growth-factor cap `R_MAX = 2.0` to prevent aliasing of strongly stretched layers (`thermal_model.cpp:1611–1745`). The total depth always reaches $b\thinspace\delta_1$.

### 4.4 Non-Uniform-Grid Finite-Difference Coefficients

On a non-uniform vertical grid, the second-order central approximation of $\partial_z(k\thinspace \partial_z T)$ at interior node $i$ is

$$\frac{\partial}{\partial z}\negthinspace\left(k\thinspace \frac{\partial T}{\partial z}\right)_i \thickspace\approx\thickspace g_{1,i}\thinspace k_{i-1}(T_{i-1} - T_i)  +  g_{2,i}\thinspace k_i\thinspace(T_{i+1} - T_i) \tag{B8}$$

with grid coupling coefficients

$$g_{1,i}  =  \frac{2\thinspace\Delta z_{i+1}}{D_i}, \qquad g_{2,i}  =  \frac{2\thinspace\Delta z_i}{D_i}, \qquad D_i  =  \Delta z_i\thinspace \Delta z_{i+1}\thinspace (\Delta z_i + \Delta z_{i+1}). \tag{B9}$$

For uniform spacing this reduces to the familiar $1/\Delta z^2$ stencil; for stretched grids the asymmetric weighting in (B9) preserves second-order accuracy. The coefficients $g_1$, $g_2$, and $D \equiv d_{3z}$ are pre-computed once during `Profile` construction (`thermal_model.cpp:687–689`) and reused on every timestep. Cell-centered conductivities $k_{i-1}$ (lower interface) and $k_i$ (upper interface) are used directly without arithmetic or harmonic averaging, consistent with the cell-centered finite-difference convention.

## 5. Boundary Conditions

### 5.1 Surface Energy Balance

At $z = 0$, the surface temperature $T_0$ satisfies the energy balance

$$Q_s  +  k_0\thinspace \frac{T_1 - T_0}{\Delta z_0}  =  \varepsilon\thinspace \sigma\thinspace T_0^{\thinspace4} \tag{B10}$$

where $\sigma = 5.670\times 10^{-8}$ W m⁻² K⁻⁴ is the Stefan–Boltzmann constant, $\varepsilon$ is the broadband IR emissivity, $k_0$ and $\Delta z_0$ are the surface-layer conductivity and thickness, and $T_1$ is the temperature of the first subsurface node. The absorbed surface flux $Q_s(t)$ is composed by the upstream modules,

$$Q_s  =  (1 - A)\thinspace\bigl[\thinspaceQ_{\text{solar}} + Q_{\text{refl}}\thinspace\bigr]  +  \varepsilon\thinspace Q_{\text{IR}}  +  Q_{\text{indirect}}, \tag{B11}$$

with $Q_{\text{solar}}$, $Q_{\text{refl}}$, $Q_{\text{IR}}$ from [illumination-model.md](illumination-model.md) §3–§5 and $Q_{\text{indirect}}$ from [radiation-exchange.md](radiation-exchange.md) §3 (already weighted by $\varepsilon$ via Kirchhoff's law inside that module).

### 5.2 Newton–Raphson Surface Solver

Equation (B10) is nonlinear in $T_0$ through the $T_0^4$ term and is solved at each timestep by damped Newton–Raphson iteration (`thermal_model.cpp:1250–1296`). Defining the residual

$$f(T_0)  =  Q_s + k_0\thinspace(T_1 - T_0)/\Delta z_0  -  \varepsilon\thinspace \sigma\thinspace T_0^{\thinspace4}, \tag{B12}$$

with derivative $f'(T_0) = -k_0/\Delta z_0 - 4\thinspace\varepsilon\thinspace\sigma\thinspace T_0^{\thinspace3}$, the iterate is updated by $T_0^{(k+1)} = T_0^{(k)} - \omega\thinspace f / f'$ with relaxation factor $\omega = 0.8$. The initial guess is the radiative-equilibrium estimate $T_0^{(0)} = (Q_s/\varepsilon\sigma)^{1/4}$ when $Q_s > 0$ and $T_0^{(0)} = T_1$ otherwise; iteration stops when $|\Delta T_0| < 0.001$ K, with a hard cap of 10 iterations (typical convergence is reached in 2–4). The solution is clamped to the physical range $[3, 500]$ K.

### 5.3 Bottom Boundary

A prescribed conductive heat flux $Q_b$ is imposed at the deepest node, giving the linear update

$$T_n  =  T_{n-1}  +  Q_b\thinspace \frac{\Delta z_{n-1}}{k_n} \tag{B13}$$

(`thermal_model.cpp:1298–1305`). Body-specific values are listed in §3.5.

### 5.4 Numerical Safety Bounds

After the implicit interior solve (§6), each $T_i$ is clamped to $[2.7, 500]$ K, with non-finite values rolled back to the previous timestep's value (`thermal_model.cpp:1241–1247`). These guards do not activate in any of the production runs reported here; they exist to keep transient violations during eclipse ingress or strong shadowing from corrupting subsequent timesteps.

## 6. Temporal Discretization

### 6.1 Crank–Nicolson on the Non-Uniform Grid

Time integration uses the implicit, second-order Crank–Nicolson scheme. Substituting (B8) into (B1) and centering the spatial operator at the midpoint $t^{n+1/2}$ yields, for each interior node $i$ and timestep $\Delta t$,

$$T_i^{n+1}  -  T_i^n  =  \frac{\Delta t}{2\thinspace\rho_i\thinspace c_{p,i}} \Bigl[\thinspace \mathcal{L}\thinspace T^{n+1}  +  \mathcal{L}\thinspace T^n \thinspace\Bigr], \tag{B14}$$

with $\mathcal{L}\thinspaceT \equiv g_{1,i} k_{i-1}(T_{i-1} - T_i) + g_{2,i} k_i (T_{i+1} - T_i)$. Defining $\alpha = g_{1,i} k_{i-1}$, $\beta = g_{2,i} k_i$, and $\nu = \Delta t / (2\thinspace\rho_i c_{p,i})$, the resulting tridiagonal system $a_i T_{i-1}^{n+1} + b_i T_i^{n+1} + c_i T_{i+1}^{n+1} = d_i$ has

$$a_i = -\nu\thinspace\alpha, \quad b_i = 1 + \nu\thinspace(\alpha + \beta), \quad c_i = -\nu\thinspace\beta, \tag{B15}$$

$$d_i = T_i^n + \nu\bigl[\thinspace\alpha\thinspace T_{i-1}^n - (\alpha + \beta)\thinspace T_i^n + \beta\thinspace T_{i+1}^n\thinspace\bigr]. \tag{B16}$$

The surface and bottom boundary values $T_0^{n+1}$ and $T_n^{n+1}$ (computed via §5.2 and §5.3) are folded into the right-hand side at the first and last interior rows, after which the corresponding off-diagonal entries are zeroed. Material properties $k(T)$, $c_p(T)$, and $\rho c_p$ are updated using $T^n$ before assembling (B15)–(B16), giving an effectively semi-implicit treatment of the nonlinearity (Picard linearization).

### 6.2 Thomas Algorithm

The tridiagonal system is inverted in $O(N_z)$ operations with the Thomas algorithm (`thermal_model.cpp:267–290`). The CN matrix is strictly diagonally dominant for $\nu > 0$, since $|b_i| = 1 + \nu(\alpha + \beta) > \nu(\alpha + \beta) = |a_i| + |c_i|$, so the algorithm is stable without pivoting. Solver buffers are held in thread-local memory pools to avoid allocation in the inner loop, and the implicit path is taken for all profiles by default (`IMPLICIT_ALL = true`, `THOMAS_LIMIT = 4096`). On modern CPUs the assembly stage of (B15)–(B16) is vectorized using AVX-512 or AVX2 fused multiply-add instructions for the temperature-dependent $k$ and $c_p$ updates; per-face profiles are dispatched in parallel across faces with OpenMP, and the entire integrator can optionally be offloaded to OpenCL on AMD or Apple GPUs (`gpu/thermal_opencl.cpp`), validated against the CPU path to within numerical precision.

### 6.3 Adaptive Time Step

The base timestep is set to $\Delta t = P / N_{\text{day}}$ with $N_{\text{day}} = $ `MAIN_LOOP_STEPS_PER_DAY` (default 768; ~11 s on the Moon, ~6 s on Europa). When `MULTIRATE_ADAPTIVE_STEP = true` (default), $\Delta t$ is scaled by a multiplier $s \in [0.1,\thinspace 8]$ that grows by a factor `DT_GROW = 1.0` (no growth) or shrinks by `DT_SHRINK = 0.1` based on the magnitude of the surface-temperature change in the previous step relative to `SURF_DMAX = 0.1` K (`thermal_model.cpp:1907–1925`). The unconditional stability of CN means that aggressive shrinking is preferred to growth: errors from oversized steps appear as low-order oscillations in the surface temperature, and the multirate controller reins them in within one or two steps.

### 6.4 Event-Aware Refinement

Sharp flux discontinuities — most importantly Jupiter eclipses on Europa, which can drop the absorbed flux by an order of magnitude in $\sim$1 minute — are handled by the event-aware controller (`thermal_model.cpp:1515–1546`). When `EVENT_AWARE_DT = true`, the timestep is additionally limited so that the linearly extrapolated change in $Q_s$ across the step satisfies $|\Delta Q_s| \le$ `FLUX_DQ_TARGET` ($= 0.1$ W m⁻²), with a floor of `MIN_EVENT_DT` ($= 0.25$ s). Refinement is triggered when the local flux derivative exceeds `FLUX_DERIV_THRESH` ($= 5\times 10^{-4}$ W m⁻² s⁻¹). For known flux knots (e.g., eclipse ingress and egress times tagged in the precomputed flux series) the controller additionally aligns one timestep boundary to each knot, so the discontinuity is sampled exactly rather than smeared across a CN step.

## 7. Equilibration

### 7.1 Strategy

Before the science integration window, the thermal model is spun up to a quasi-periodic diurnal steady state from a depth-dependent initial profile. The initial condition uses an exponential blend toward radiative equilibrium,

$$T(z,\thinspace t = 0)  =  T_{\text{deep}} + (T_{\text{avg}} - T_{\text{deep}})\thinspace \exp(-z / \delta_1), \tag{B17}$$

with $T_{\text{max}} = ((1-A)\thinspaceS/(\varepsilon\sigma))^{1/4}$, $T_{\text{avg}} = T_{\text{max}}/\sqrt{2}$, and $T_{\text{deep}} = T_{\text{avg}} + \min(Q_b\thinspace z/k,\thinspace 50\thickspace\text{K})$ (`thermal_model.cpp:718–778`). The model is then advanced through `NYEARSEQ` orbits (default 1, equivalent to 29.5 lunar diurnal cycles) of clear-sky diurnal forcing using the `fastEquilibrate` routine.

### 7.2 Decoupled Equilibration Cadence

Because Crank–Nicolson is unconditionally stable, the equilibration cadence can be much coarser than the science-window cadence without compromising the spin-up. PTM3D supports three equilibration cadence policies, in priority order (`ptm_thermal_sim.cpp:7796–7830`):

1. **Explicit override.** `EQUILIBRATION_STEPS_PER_DAY` (CLI `--equilibration-steps-per-day`, YAML `thermal.equilibration_steps_per_day`) directly sets $\Delta t_{\text{eq}} = P / N_{\text{eq}}$.
2. **Accuracy target.** `EQUILIBRATION_SAMPLES_PER_DIURNAL_CYCLE` (CLI `--equilibration-samples-per-diurnal`) requests a body-independent number of samples per diurnal cycle. Calibration on Moon ico3 vs. a Fourier-stable reference gives maximum surface-temperature errors of ~7 K at 30 samples cycle⁻¹, ~4 K at 120 samples cycle⁻¹ (the GUI default), and ~3 K at 240 samples cycle⁻¹.
3. **Legacy fallback.** Match the science-window output cadence.

The default `EQUILIBRATION_STEPS_PER_DAY = 192` was chosen as a balance between cost and accuracy: a sensitivity study on the Moon (`tools/test_equilibration_dt.cpp`) reports a worst-case surface-T error below 1 K relative to a Fourier-stable reference (~5000 steps day⁻¹) after 24 lunar diurnal cycles, while running ~26× faster. Decoupling the equilibration cadence from the (typically much finer) output cadence is essential for camera-simulation runs: on a 4.8 M-face lunar mesh with 1-minute output, setting `equilibration_steps_per_day: 24` reduces the equilibration phase from 85,048 to 1,417 steps and the peak memory footprint from 227 GB to 13 GB, with <2 K accuracy impact at the first output timestep.

## 8. Validation

### 8.1 Lunar Diurnal Cycle vs. Hayne et al. (2017)

We validate the implementation against the Diviner-derived lunar regolith model of Hayne et al. (2017) by running PTM3D on a global icosphere mesh (resolution 7) for a single synodic month with the Moon defaults of §3.5. Figure B.1 shows (a) the equatorial diurnal surface-temperature cycle, (b) the post-sunset cooling curve at three latitudes, (c) a tabular comparison of the first four diurnal moments, and (d) the input parameters.

![Figure B.1: Lunar surface-temperature validation](figures/v2/lunar_validation.png)

PTM3D reproduces the Hayne et al. (2017) lunar surface temperatures to within 1.4 K (peak), 1.7 K (midnight), 1.1 K (minimum), and **1.27 K RMS** over fifteen Hayne nighttime reference points spanning three latitudes. The figure uses six equilibration orbits on the ico7 topographic mesh with the Hayne (2017) Table A1 thermophysical defaults: $A_0 = 0.12$ with the angular-dependent Bond-albedo correction $A(\theta) = A_0 + a(\theta/\pi/4)^3 + b(\theta/\pi/2)^8$ (with $a = 0.06$, $b = 0.25$) composed on top of a per-face $A_0$ from the LRO-derived Bond-albedo map. All thermophysical parameters ($\rho_s$, $\rho_d$, $H$, $K_s$, $K_d$, $\chi$, $\varepsilon$, $c_p$ polynomial coefficients, $Q_b$) match Hayne (2017) Table A1 exactly. The vertical depth grid uses $(m, n, b) = (10, 15, 20)$, placing the bottom Neumann boundary at $\sim 12$ effective skin depths so the diurnal wave is fully damped before it.

### 8.2 Subsurface Diurnal Wave

The shape and depth dependence of the diurnal thermal wave provide a second internal-consistency check. Figure B.2 shows the modeled subsurface temperature evolution at the lunar equator: panel (a) displays $T(z)$ at six local times spanning one synodic day, panel (b) shows the corresponding depth–time heatmap with the diurnal skin depth $\delta_1 = 3.4$ cm (eq. B6) marked, and panel (c) compares the diurnal amplitude to the analytic semi-infinite slab decay $\exp(-z/\delta_1)$. The amplitude decays approximately exponentially within the upper $\sim$5 skin depths and the wave acquires a phase lag with depth, both as expected for a linear semi-infinite slab with constant properties; departures from the analytical solution at depth reflect the depth-dependent $\rho(z)$ and $k(z)$ of (B4)–(B5).

![Figure B.2: Lunar subsurface diurnal thermal wave](figures/v2/thermal_wave_profiles.png)

### 8.3 Equilibration Convergence

The equilibration cadence accuracy is documented by the `tools/test_equilibration_dt` test program, which runs `fastEquilibrate` over 24 lunar diurnal cycles at four latitudes (0°, 30°, 60°, 85°) and reports the maximum and mean per-layer temperature differences against a Fourier-stable reference solution. Figure B.3 visualizes the resulting convergence: at the production cadence `EQUILIBRATION_STEPS_PER_DAY = 192`, the worst-case error across all latitudes and depth nodes is 0.99 K (latitude 85°), just below the 1 K production target. Coarsening to 96 steps day⁻¹ raises the worst-case error to 2.2 K, and to 24 steps day⁻¹ to 11.9 K; refining to 768 steps day⁻¹ reduces the error to 0.21 K. The error scales as $\Delta t^2$, consistent with the second-order accuracy of Crank–Nicolson.

![Figure B.3: Equilibration cadence convergence](figures/v2/equilibration_convergence.png)

### 8.4 Surface Energy Conservation

Figure B.4 quantifies the closure of the surface energy balance over a fully integrated synodic month on the ico7 topographic mesh (327,680 faces) after six equilibration orbits. For an equilibrated diurnal cycle the per-face energy budget should satisfy $\langle Q_{\mathrm{abs}}\rangle_t = \langle\varepsilon\sigma T^4\rangle_t - Q_b$. Both the simulation and the post-hoc accounting in this figure apply the Hayne et al. (2017) angular albedo correction $A(\theta)$ on top of the per-face base albedo $A_0$, so any residual reflects an actual closure error rather than a parameter mismatch. The bulk-mean fractional residual is 3.7 %, with a global imbalance of 2.0 % between absorbed and emitted radiation integrated over the synodic month (panel c). The remaining residual reflects a combination of (i) finite-time-of-flight thermal lag between absorbed solar input and re-radiated thermal emission and (ii) the difference between $\langle T \rangle^4$ and $\langle T^4 \rangle$ that Jensen's inequality introduces when applying $\sigma T^4$ to time-averaged temperatures.

![Figure B.4: Surface energy-balance closure](figures/v2/energy_conservation.png)

### 8.5 Computational Performance

Figure B.5 reports measured wall-clock times for the illumination and thermal modules vs lunar icosphere mesh resolution (ico4..ico6, 5,120 to 81,920 faces) on a 2-die AMD W6800X workstation. Each measurement is a one–lunar-day, 144–timestep run with no radiation exchange and no equilibration. Three thermal back-ends are compared: CPU OpenMP (24 cores), single-die OpenCL GPU, and the four-die OpenCL pool (`--gpu-devices 4`). The CPU illumination benchmark uses Intel Embree.

The post-illumination phase (thermal solver plus statistics export and binary output) scales roughly linearly with mesh size, with the GPU back-ends pulling ahead at ico6 (81,920 faces) by a factor of 1.5×. The actual GPU thermal kernel is much faster than this end-to-end metric (e.g. 252 ms for the ico6 single-GPU run vs 6.9 s post-illumination on CPU); the figure intentionally reports the user-visible total because output and statistics dominate at small mesh size. The dashed lines are the model extrapolation in `plots/performance/runtime_estimates.csv`, anchored on a 4 M-face Mons-Mouton run; measurements track the model within ±20 %.

![Figure B.5: Computational performance scaling](figures/v2/performance_scaling.png)

### 8.6 Europa Validation

Validation of PTM3D against Europa-specific observations (Galileo PPR diurnal curves and ALMA depth-resolved brightness temperatures) is presented in the main text Section 3 and is not repeated here. The 1D thermal solver is identical for the Moon and Europa runs; only the input parameters of §3.5 and the orbital/illumination geometry change.

## 9. Configuration Reference

The key user-facing parameters of the thermal solver are summarized below; a complete list lives in the `Config` class (`thermal_model.h:179–294`).

| Parameter | Default | Description |
|---|---|---|
| `chi` | 2.7 | Radiative conductivity parameter $\chi$ (eq. B2) |
| `m` | 4 | Layers in the upper skin depth (eq. B7) |
| `n` | 10 | Layer growth factor (eq. B7) |
| `b` | 5 | Skin depths to the bottom layer (eq. B7) |
| `ADAPTIVE_GRID` | true | Use per-profile adaptive grid (§4.3) |
| `RES_DECAY_SKINS` | 3.0 | e-folding scale for resolution decay with depth |
| `MIN_LAYERS_DEEP_PER_SKIN` | 1 | Floor on layers per skin depth in the deep |
| `R_MAX` | 2.0 | Per-step layer-thickness growth cap |
| `NYEARSEQ` | 1 | Equilibration duration (orbits) |
| `EQUILIBRATION_STEPS_PER_DAY` | 192 | Equilibration cadence (§7.2) |
| `MAIN_LOOP_STEPS_PER_DAY` | 768 | Science-window cadence (§6.3) |
| `IMPLICIT_ALL` | true | Force CN+Thomas for all profiles |
| `THOMAS_LIMIT` | 4096 | Profile-size limit for the Thomas path |
| `MULTIRATE_ADAPTIVE_STEP` | true | Enable surface-driven adaptive $\Delta t$ (§6.3) |
| `SURF_DMAX` | 0.1 | Target maximum $|\Delta T_{\text{surf}}|$ per step (K) |
| `DT_GROW`, `DT_SHRINK` | 1.0, 0.1 | Adaptive-step multipliers |
| `MAX_DT_SCALE` | 8.0 | Cap on the adaptive-step multiplier |
| `EVENT_AWARE_DT` | true | Enable flux-event $\Delta t$ refinement (§6.4) |
| `FLUX_DERIV_THRESH` | $5 \times 10^{-4}$ | $|dQ_s/dt|$ trigger threshold (W m⁻² s⁻¹) |
| `FLUX_DQ_TARGET` | 0.1 | Target maximum $|\Delta Q_s|$ per step (W m⁻²) |
| `MIN_EVENT_DT` | 0.25 | Minimum $\Delta t$ during flux events (s) |

## 10. Symbols and Units

| Symbol | Meaning | Units |
|---|---|---|
| $T(z, t)$ | Subsurface temperature | K |
| $T_0, T_1, \ldots, T_n$ | Discrete temperatures at depth nodes $0, 1, \ldots, n$ | K |
| $z, \Delta z_i$ | Depth, layer thickness | m |
| $t, \Delta t$ | Time, timestep | s |
| $P$ | Diurnal period | s |
| $\rho(z)$ | Bulk density | kg m⁻³ |
| $\rho_s, \rho_d, H$ | Surface, deep density, e-folding scale (eq. B4) | kg m⁻³, m |
| $c_p(T)$ | Specific heat capacity | J kg⁻¹ K⁻¹ |
| $k(T, z)$ | Bulk thermal conductivity (eq. B2) | W m⁻¹ K⁻¹ |
| $k_c(z)$ | Contact (lattice) conductivity (eq. B5) | W m⁻¹ K⁻¹ |
| $k_s, k_d$ | Surface, deep contact conductivity | W m⁻¹ K⁻¹ |
| $\chi, R_{350}$ | Radiative parameter, $R_{350} = \chi/350^3$ | –, K⁻³ |
| $\kappa$ | Thermal diffusivity, $\kappa = k/(\rho c_p)$ | m² s⁻¹ |
| $\delta_1$ | Diurnal thermal skin depth (eq. B6) | m |
| $g_{1,i}, g_{2,i}, D_i$ | Non-uniform-grid coupling coefficients (eq. B9) | m⁻², m⁻³ |
| $\varepsilon$ | Broadband IR emissivity | – |
| $A, A(\theta)$ | Bond albedo, incidence-angle-dependent albedo | – |
| $\sigma$ | Stefan–Boltzmann constant ($5.670 \times 10^{-8}$) | W m⁻² K⁻⁴ |
| $Q_s$ | Total absorbed surface flux (eq. B11) | W m⁻² |
| $Q_{\text{solar}}, Q_{\text{refl}}, Q_{\text{IR}}, Q_{\text{indirect}}$ | Component fluxes | W m⁻² |
| $Q_b$ | Internal (geothermal) heat flux | W m⁻² |
| $a_i, b_i, c_i, d_i$ | Tridiagonal system entries (eq. B15–B16) | varied |
| $\alpha, \beta, \nu$ | Tridiagonal-coefficient shorthand | varied |

## 11. References

Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. *Mathematical Proceedings of the Cambridge Philosophical Society*, 43(1), 50–67. https://doi.org/10.1017/S0305004100023197

Giauque, W. F., & Stout, J. W. (1936). The entropy of water and the third law of thermodynamics: The heat capacity of ice from 15 to 273 K. *Journal of the American Chemical Society*, 58(7), 1144–1150. https://doi.org/10.1021/ja01298a023

Hayne, P. O., Bandfield, J. L., Siegler, M. A., Vasavada, A. R., Ghent, R. R., Williams, J.-P., Greenhagen, B. T., Aharonson, O., Elder, C. M., Lucey, P. G., & Paige, D. A. (2017). Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 122(12), 2371–2400. https://doi.org/10.1002/2017JE005387

Langseth, M. G., Keihm, S. J., & Peters, K. (1976). Revised lunar heat-flow values. *Proceedings of the Lunar Science Conference*, 7, 3143–3171.

Ledlow, M. J., Zeilik, M., Burns, J. O., Gisler, G. R., Zhao, J.-H., & Baker, D. N. (1992). Subsurface emissions from Mercury — VLA radio observations at 2 and 6 centimeters. *Astrophysical Journal*, 384, 640–655. https://doi.org/10.1086/170906

Mitchell, D. L., & de Pater, I. (1994). Microwave imaging of Mercury's thermal emission at wavelengths from 0.3 to 20.5 cm. *Icarus*, 110(1), 2–32. https://doi.org/10.1006/icar.1994.1104

Thomas, L. H. (1949). *Elliptic problems in linear differential equations over a network*. Watson Scientific Computing Laboratory Report. Columbia University.

Vasavada, A. R., Bandfield, J. L., Greenhagen, B. T., Hayne, P. O., Siegler, M. A., Williams, J.-P., & Paige, D. A. (2012). Lunar equatorial surface temperatures and regolith properties from the Diviner Lunar Radiometer Experiment. *Journal of Geophysical Research: Planets*, 117(E12). https://doi.org/10.1029/2011JE003987
