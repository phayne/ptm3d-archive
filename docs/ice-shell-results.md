# Europa Equilibrium Ice Shell Thickness: O&S Model Results

Computed using the Ojakangas & Stevenson (1989) equilibrium ice shell model
with both the O&S analytical surface temperature and PTM3D numerical surface
temperatures (N-S symmetrized, ICO7 mesh, 327,680 faces, 144 timesteps).

Model parameters: $C_R = 0.85$, $e = 0.01$, $a = 1.569 \times 10^6$ m,
$\rho = 3040$ kg/m$^3$, $g = 1.31$ m/s$^2$.

---

## 1. Average Shell Thickness

| Case | O&S (km) | PTM3D zonal (km) | PTM3D 2D (km) | Paper (km) |
|------|----------|-------------------|----------------|------------|
| Maxwell, $H$=0 | 19.7 | 19.6 | 19.6 | 23.0 |
| Maxwell, $H$=10 | 16.0 | 15.9 | 15.9 | 21.0 |
| Flow law $n_G$=3 | 25.4 | 25.1 | 25.1 | 25.0 |
| Flow law $n_G$=5.2 | 13.5 | 13.4 | 13.4 | 13.0 |

The flow law cases match the paper within 2-4%.  Maxwell cases are 14-24%
thinner, likely due to higher sensitivity to the $C_R$ correction factor.
Using the full 2D PTM3D temperature field versus its zonal average changes
the mean thickness by < 0.01 km (negligible).

---

## 2. Spherical Harmonic Decomposition

All coefficients in km.  $b_{\ell m}^c$ denotes the $\cos(m\lambda)$
component; $b_{\ell m}^s$ denotes $\sin(m\lambda)$.

### Maxwell, $H$ = 0

| Harmonic | O&S | PTM3D zonal | PTM3D 2D |
|----------|-----|-------------|----------|
| $b_{00}$ (mean) | 19.748 | 19.561 | 19.567 |
| $b_{10}$ | 0.000 | 0.000 | 0.000 |
| $b_{11}^c$ | -0.083 | -0.083 | **+0.182** |
| $b_{11}^s$ | 0.000 | 0.000 | +0.001 |
| $b_{20}$ | -2.911 | -3.113 | -3.117 |
| $b_{21}^c$ | 0.000 | 0.000 | 0.000 |
| $b_{21}^s$ | 0.000 | 0.000 | 0.000 |
| $b_{22}^c$ | +1.414 | +1.405 | **+1.439** |
| $b_{22}^s$ | 0.000 | 0.000 | -0.001 |

### Maxwell, $H$ = 10

| Harmonic | O&S | PTM3D zonal | PTM3D 2D |
|----------|-----|-------------|----------|
| $b_{00}$ (mean) | 16.032 | 15.878 | 15.882 |
| $b_{10}$ | 0.000 | 0.000 | 0.000 |
| $b_{11}^c$ | -0.063 | -0.063 | **+0.146** |
| $b_{11}^s$ | 0.000 | 0.000 | +0.001 |
| $b_{20}$ | -1.288 | -1.468 | -1.469 |
| $b_{21}^c$ | 0.000 | 0.000 | 0.000 |
| $b_{21}^s$ | 0.000 | 0.000 | 0.000 |
| $b_{22}^c$ | +0.870 | +0.864 | **+0.892** |
| $b_{22}^s$ | 0.000 | 0.000 | -0.001 |

### Flow law $n_G$ = 3

| Harmonic | O&S | PTM3D zonal | PTM3D 2D |
|----------|-----|-------------|----------|
| $b_{00}$ (mean) | 25.369 | 25.123 | 25.128 |
| $b_{10}$ | 0.000 | 0.000 | 0.000 |
| $b_{11}^c$ | -0.098 | -0.097 | **+0.229** |
| $b_{11}^s$ | 0.000 | 0.000 | +0.001 |
| $b_{20}$ | -1.113 | -1.414 | -1.415 |
| $b_{21}^c$ | 0.000 | 0.000 | 0.000 |
| $b_{21}^s$ | 0.000 | 0.000 | 0.000 |
| $b_{22}^c$ | +1.141 | +1.133 | **+1.176** |
| $b_{22}^s$ | 0.000 | 0.000 | -0.001 |

### Flow law $n_G$ = 5.2

| Harmonic | O&S | PTM3D zonal | PTM3D 2D |
|----------|-----|-------------|----------|
| $b_{00}$ (mean) | 13.514 | 13.382 | 13.385 |
| $b_{10}$ | 0.000 | 0.000 | 0.000 |
| $b_{11}^c$ | -0.051 | -0.051 | **+0.122** |
| $b_{11}^s$ | 0.000 | 0.000 | +0.001 |
| $b_{20}$ | -0.275 | -0.440 | -0.441 |
| $b_{21}^c$ | 0.000 | 0.000 | 0.000 |
| $b_{21}^s$ | 0.000 | 0.000 | 0.000 |
| $b_{22}^c$ | +0.536 | +0.533 | **+0.555** |
| $b_{22}^s$ | 0.000 | 0.000 | 0.000 |

---

## 3. Table I Comparison: $4t_{22}$ and $t_{20} + 2t_{22}$

| Case | $4t_{22}$ (km) | | | | $t_{20} + 2t_{22}$ (km) | | | |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | **O&S** | **PTM3D zon.** | **PTM3D 2D** | **Paper** | **O&S** | **PTM3D zon.** | **PTM3D 2D** | **Paper** |
| Maxwell, $H$=0 | 5.66 | 5.62 | 5.76 | 6.6 | -0.08 | -0.30 | -0.24 | -0.1 |
| Maxwell, $H$=10 | 3.48 | 3.46 | 3.57 | 4.6 | +0.45 | +0.26 | +0.31 | +0.4 |
| Flow law $n_G$=3 | 4.56 | 4.53 | 4.70 | 4.4 | +1.17 | +0.85 | +0.94 | +1.1 |
| Flow law $n_G$=5.2 | 2.15 | 2.13 | 2.22 | 2.0 | +0.80 | +0.62 | +0.67 | +0.8 |

---

## 4. Moment of Inertia Contributions

From Eqs 5.9-5.10 of O&S (1989):

$$\Delta\!\left(\frac{B-A}{C}\right) \approx 10^{-7} \times \frac{4\,t_{22}}{1\;\text{km}}$$

$$\Delta\!\left(\frac{B-C}{A}\right) \approx 10^{-7} \times \frac{t_{20} + 2\,t_{22}}{1\;\text{km}}$$

| Case | $\Delta(B\!-\!A)/C$ | | | $\Delta(B\!-\!C)/A$ | | |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| | **O&S** | **PTM3D zon.** | **PTM3D 2D** | **O&S** | **PTM3D zon.** | **PTM3D 2D** |
| Maxwell, $H$=0 | $5.66 \times 10^{-7}$ | $5.62 \times 10^{-7}$ | $5.76 \times 10^{-7}$ | $-8.2 \times 10^{-9}$ | $-3.0 \times 10^{-8}$ | $-2.4 \times 10^{-8}$ |
| Maxwell, $H$=10 | $3.48 \times 10^{-7}$ | $3.46 \times 10^{-7}$ | $3.57 \times 10^{-7}$ | $+4.5 \times 10^{-8}$ | $+2.6 \times 10^{-8}$ | $+3.1 \times 10^{-8}$ |
| Flow law $n_G$=3 | $4.56 \times 10^{-7}$ | $4.53 \times 10^{-7}$ | $4.70 \times 10^{-7}$ | $+1.2 \times 10^{-7}$ | $+8.5 \times 10^{-8}$ | $+9.4 \times 10^{-8}$ |
| Flow law $n_G$=5.2 | $2.15 \times 10^{-7}$ | $2.13 \times 10^{-7}$ | $2.22 \times 10^{-7}$ | $+8.0 \times 10^{-8}$ | $+6.2 \times 10^{-8}$ | $+6.7 \times 10^{-8}$ |

**Polar wander stability**: $\Delta(B\!-\!C)/A > 0$ implies the shell's
mass distribution is incompatible with synchronous rotation and could drive
polar wander.  Only the Maxwell $H$=0 case has $\Delta(B\!-\!C)/A < 0$
(stable); all other cases are marginally unstable at the $10^{-8}$ level.

---

## 5. Effect of PTM3D Longitude-Dependent Temperatures

The PTM3D simulation produces a ~2 K sub-Jovian/anti-Jovian temperature
asymmetry at the equator (sub-Jovian ~108 K, anti-Jovian ~110 K after
N-S symmetrization).  Compared to zonal averaging:

| Quantity | Change (PTM3D 2D vs zonal) | Relative |
|----------|---------------------------|----------|
| Mean thickness | < 0.01 km | < 0.03% |
| $b_{11}^c$ (sub-Jovian/anti-Jovian) | +0.17 to +0.33 km | **dominant new harmonic** |
| $4t_{22}$ | +0.09 to +0.17 km | 2-4% |
| $t_{20} + 2t_{22}$ | +0.05 to +0.09 km | 5-10% |
| $\Delta(B\!-\!A)/C$ | $+1 \times 10^{-8}$ to $+2 \times 10^{-8}$ | 3-4% |
| $\Delta(B\!-\!C)/A$ | $+1 \times 10^{-8}$ | small |

The longitude-dependent temperature introduces a degree-1 harmonic
($b_{11}^c \sim 0.2$ km) that makes the sub-Jovian hemisphere ~0.2-0.3 km
thicker than the anti-Jovian hemisphere.  This projects primarily onto
$\ell = 1$, not $\ell = 2$, so the effect on $\Delta(B\!-\!A)/C$ and
$\Delta(B\!-\!C)/A$ is small (a few percent).

---

## 6. Notes

- All PTM3D results use N-S symmetrized temperatures (poles averaged with
  their mirror counterparts using a KDTree on the sphere).  Before
  symmetrization the north pole was ~72 K and the south pole ~49 K; after,
  both are ~60 K.

- The O&S analytical polar temperature (~52 K) is lower than the
  symmetrized PTM3D value (~60 K), producing thicker polar shells in the
  O&S model.  This is the main driver of the $b_{20}$ difference between
  O&S and PTM3D.

- The $b_{11}^c$ component in the O&S model is small and negative (-0.05 to
  -0.10 km) due to tidal strains alone.  PTM3D's longitude-dependent
  temperature flips this to positive (+0.12 to +0.23 km), with the cooler
  sub-Jovian hemisphere getting a thicker shell.

- The $C_R = 0.85$ calibration reproduces the paper's flow law average
  thicknesses within 2-4%, but Maxwell cases remain 14-24% low.  No single
  $C_R$ reproduces all Table I entries simultaneously.

- The spatial harmonics ($4t_{22}$, $t_{20} + 2t_{22}$) are systematically
  ~2$\times$ the paper's values for $4t_{22}$ but agree better for
  $t_{20} + 2t_{22}$.  This appears to be a structural discrepancy that
  cannot be resolved by $C_R$ tuning.

---

## 7. Albedo Map Simulation Results

Sections 1-6 used a constant-albedo PTM3D simulation (144 timesteps).
The following results compare two new 288-timestep ICO7 simulations:

- **Constant albedo** ($A = 0.12$, `use_albedo_map: false`)
- **Albedo map** (spatially variable $A(\theta, \lambda)$ from Europa
  Bond albedo FITS map, `use_albedo_map: true`)

Both use N-S symmetrized temperatures.

### 7.1 Surface Temperature

| Model | Equator (K) | Pole (K) | Mean (K) |
|-------|:-----------:|:--------:|:--------:|
| O&S analytical | 108.8 | 52.1 | 101.5 |
| Constant albedo (2D) | 105.1 | 46.1 | 99.9 |
| Albedo map (2D) | 109.6 | 46.4 | 97.0 |

Sub-Jovian ($\lambda = 0$) vs anti-Jovian ($\lambda = 180°$)
equatorial temperature:

| Simulation | Sub-Jovian (K) | Anti-Jovian (K) | $\Delta T$ (K) |
|-----------|:-:|:-:|:-:|
| Constant albedo | 105.1 | 107.4 | $-2.3$ |
| Albedo map | 109.6 | 105.8 | $+3.8$ |

The albedo map raises the sub-Jovian equatorial temperature (+4.5 K
relative to constant albedo) but lowers the anti-Jovian temperature
($-1.6$ K), reflecting the spatial variation in Europa's surface albedo.
The global mean temperature is ~3 K lower with the albedo map, indicating
the mean albedo from the map is somewhat higher than the constant
$A = 0.12$.

### 7.2 Average Shell Thickness

| Case | O&S (km) | Const-alb zon. | Const-alb 2D | Alb-map zon. | Alb-map 2D | Paper (km) |
|------|:--------:|:-:|:-:|:-:|:-:|:--------:|
| Maxwell, $H$=0 | 19.7 | 20.1 | 20.1 | 20.6 | 20.7 | 23 |
| Maxwell, $H$=10 | 16.0 | 16.3 | 16.3 | 16.8 | 16.8 | 21 |
| Flow law $n_G$=3 | 25.4 | 25.8 | 25.8 | 26.5 | 26.6 | 25 |
| Flow law $n_G$=5.2 | 13.5 | 13.7 | 13.7 | 14.1 | 14.1 | 13 |

The albedo map simulation produces shells ~0.5-0.8 km thicker than
constant albedo, consistent with the ~3 K lower mean surface
temperature. Both PTM3D simulations give shells 0.3-0.5 km thicker
than O&S.

### 7.3 Spherical Harmonic Decomposition (Albedo Map, 2D)

All coefficients in km. **Bold** values indicate new harmonics introduced
by the albedo map that are absent or negligible in the constant-albedo
case.

#### Maxwell, $H$ = 0

| Harmonic | O&S | Const-alb 2D | Alb-map 2D |
|----------|:---:|:---:|:---:|
| $b_{00}$ (mean) | 19.748 | 20.075 | 20.673 |
| $b_{10}$ | 0.000 | 0.000 | 0.000 |
| $b_{11}^c$ | $-0.083$ | $+0.216$ | $+0.364$ |
| $b_{11}^s$ | 0.000 | $-0.001$ | **$-1.286$** |
| $b_{20}$ | $-2.911$ | $-3.252$ | $-2.897$ |
| $b_{22}^c$ | $+1.414$ | $+1.482$ | $+1.497$ |
| $b_{22}^s$ | 0.000 | 0.000 | **$-0.100$** |

#### Maxwell, $H$ = 10

| Harmonic | O&S | Const-alb 2D | Alb-map 2D |
|----------|:---:|:---:|:---:|
| $b_{00}$ (mean) | 16.032 | 16.293 | 16.785 |
| $b_{11}^c$ | $-0.063$ | $+0.174$ | $+0.295$ |
| $b_{11}^s$ | 0.000 | $-0.001$ | **$-1.048$** |
| $b_{20}$ | $-1.288$ | $-1.553$ | $-1.244$ |
| $b_{22}^c$ | $+0.870$ | $+0.919$ | $+0.928$ |
| $b_{22}^s$ | 0.000 | 0.000 | **$-0.081$** |

#### Flow law $n_G$ = 3

| Harmonic | O&S | Const-alb 2D | Alb-map 2D |
|----------|:---:|:---:|:---:|
| $b_{00}$ (mean) | 25.369 | 25.779 | 26.558 |
| $b_{11}^c$ | $-0.098$ | $+0.272$ | $+0.462$ |
| $b_{11}^s$ | 0.000 | $-0.001$ | **$-1.662$** |
| $b_{20}$ | $-1.113$ | $-1.524$ | $-1.016$ |
| $b_{22}^c$ | $+1.141$ | $+1.213$ | $+1.222$ |
| $b_{22}^s$ | 0.000 | 0.000 | **$-0.128$** |

#### Flow law $n_G$ = 5.2

| Harmonic | O&S | Const-alb 2D | Alb-map 2D |
|----------|:---:|:---:|:---:|
| $b_{00}$ (mean) | 13.514 | 13.731 | 14.148 |
| $b_{11}^c$ | $-0.051$ | $+0.144$ | $+0.245$ |
| $b_{11}^s$ | 0.000 | $-0.001$ | **$-0.886$** |
| $b_{20}$ | $-0.275$ | $-0.491$ | $-0.214$ |
| $b_{22}^c$ | $+0.536$ | $+0.573$ | $+0.577$ |
| $b_{22}^s$ | 0.000 | 0.000 | **$-0.068$** |

### 7.4 Table I Comparison: $4t_{22}$ and $t_{20} + 2t_{22}$

| Case | $4t_{22}$ (km) | | | | $t_{20} + 2t_{22}$ (km) | | | |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| | **O&S** | **Const-alb** | **Alb-map** | **Paper** | **O&S** | **Const-alb** | **Alb-map** | **Paper** |
| Maxwell, $H$=0 | 5.66 | 5.93 | 5.99 | 6.6 | $-0.08$ | $-0.29$ | $+0.10$ | $-0.1$ |
| Maxwell, $H$=10 | 3.48 | 3.68 | 3.71 | 4.6 | $+0.45$ | $+0.29$ | $+0.61$ | $+0.4$ |
| Flow law $n_G$=3 | 4.56 | 4.85 | 4.89 | 4.4 | $+1.17$ | $+0.90$ | $+1.43$ | $+1.1$ |
| Flow law $n_G$=5.2 | 2.15 | 2.29 | 2.31 | 2.0 | $+0.80$ | $+0.65$ | $+0.94$ | $+0.8$ |

### 7.5 Moment of Inertia Contributions

| Case | $\Delta(B\!-\!A)/C$ | | $\Delta(B\!-\!C)/A$ | |
|------|:-:|:-:|:-:|:-:|
| | **Const-alb** | **Alb-map** | **Const-alb** | **Alb-map** |
| Maxwell, $H$=0 | $5.93 \times 10^{-7}$ | $5.99 \times 10^{-7}$ | $-2.9 \times 10^{-8}$ | $+1.0 \times 10^{-8}$ |
| Maxwell, $H$=10 | $3.68 \times 10^{-7}$ | $3.71 \times 10^{-7}$ | $+2.9 \times 10^{-8}$ | $+6.1 \times 10^{-8}$ |
| Flow law $n_G$=3 | $4.85 \times 10^{-7}$ | $4.89 \times 10^{-7}$ | $+9.0 \times 10^{-8}$ | $+1.4 \times 10^{-7}$ |
| Flow law $n_G$=5.2 | $2.29 \times 10^{-7}$ | $2.31 \times 10^{-7}$ | $+6.6 \times 10^{-8}$ | $+9.4 \times 10^{-8}$ |

---

## 8. Effect of Spatially Variable Albedo

### 8.1 New Harmonics

The albedo map introduces two harmonics that are absent in the
constant-albedo and O&S models:

**$b_{11}^s$ (leading/trailing asymmetry):** The dominant new signal.
Values range from $-0.9$ to $-1.7$ km, making the **leading hemisphere**
($\lambda = 90°$) **thicker** than the trailing hemisphere
($\lambda = 270°$).  This reflects Europa's leading-hemisphere albedo
enhancement: the leading hemisphere has higher albedo (lower
temperature, thicker shell).  This $\ell = 1$, $m = 1$ harmonic is
comparable in magnitude to the tidal $b_{22}^c$ but projects onto
different moment-of-inertia components.

**$b_{22}^s$ (tidal pattern rotation):** A smaller signal ($-0.07$ to
$-0.13$ km) that breaks the $\cos(2\lambda)$ symmetry of the
tidally-driven thickness pattern.  The albedo map introduces a
$\sin(2\lambda)$ component that effectively rotates the tidal thickness
pattern by a few degrees in longitude.

### 8.2 Changes to Existing Harmonics

| Quantity | Change (alb-map vs const-alb, 2D) | Notes |
|----------|:-:|-------|
| $b_{00}$ (mean) | $+0.4$ to $+0.8$ km | Higher mean albedo $\Rightarrow$ thicker shell |
| $b_{11}^c$ (sub/anti-Jovian) | $+0.10$ to $+0.19$ km | Amplified by albedo variations |
| $b_{11}^s$ (leading/trailing) | $-0.9$ to $-1.7$ km | **Entirely new; dominant albedo signal** |
| $b_{20}$ (pole-equator) | $+0.28$ to $+0.51$ km | Albedo map reduces pole-equator contrast |
| $b_{22}^c$ ($= t_{22}$) | $+0.004$ to $+0.015$ km | Negligible change ($<1$%) |
| $4t_{22}$ | $+0.02$ to $+0.06$ km | $<1$% change |
| $t_{20} + 2t_{22}$ | $+0.29$ to $+0.53$ km | 30-60% change |
| $\Delta(B\!-\!A)/C$ | $+1.5 \times 10^{-9}$ to $+5.9 \times 10^{-9}$ | $<1$% change |
| $\Delta(B\!-\!C)/A$ | $+2.8 \times 10^{-8}$ to $+5.3 \times 10^{-8}$ | **30-60% change** |

### 8.3 Implications

1. **$\Delta(B\!-\!A)/C$ is robust to albedo variations.** The tidal
   dissipation pattern overwhelmingly controls $4t_{22}$, and the albedo
   map changes it by $< 1$%.  For all cases,
   $\Delta(B\!-\!A)/C \sim 2$-$6 \times 10^{-7}$.

2. **$\Delta(B\!-\!C)/A$ is sensitive to albedo.** The albedo map
   increases $t_{20} + 2t_{22}$ by 30-60%, primarily through the
   $b_{20}$ term (pole-equator contrast).  For Maxwell $H = 0$, this
   flips $\Delta(B\!-\!C)/A$ from negative (stable) to positive
   (unstable), meaning all four cases are now marginally unstable.

3. **The leading/trailing asymmetry ($b_{11}^s \sim 1$-$1.7$ km) is
   the largest non-tidal signal.** This degree-1 harmonic does not
   affect the second-degree moments of inertia but would produce a
   measurable center-of-figure to center-of-mass offset of order
   $\delta r / a \sim b_{11}^s / a \sim 10^{-3}$.

4. **Albedo map effects are primarily $\ell = 1$, not $\ell = 2$.**
   The spatially variable albedo redistributes shell mass at degree-1
   scales (hemispheric asymmetries), which affects the center of
   figure but not the dynamical flattening or equatorial ellipticity.

---

## 9. Notes on Updated Analysis

- All results in Sections 7-8 use 288-timestep PTM3D simulations
  (one orbital period, $\Delta t \approx 1066$ s) with N-S
  symmetrization.  Sections 1-6 used an older 144-timestep simulation.

- The O&S analytical results are identical between sections since they
  depend only on the analytical surface temperature model.

- The constant-albedo simulation uses $A = 0.12$ uniformly. The
  albedo-map simulation reads spatially variable Bond albedo from a
  0.5-degree FITS map projected onto the ICO7 mesh. Both simulations
  use otherwise identical thermal parameters ($\varepsilon = 0.95$,
  $k_s = 7.4 \times 10^{-4}$ W m$^{-1}$ K$^{-1}$, etc.).

- The $b_{22}^c$ normalization uses
  $b_{22}^c = \frac{5}{48\pi} \int b \, P_2^2(\cos\theta) \cos(2\lambda)
  \sin\theta \, d\theta \, d\lambda$, consistent with the
  unnormalized associated Legendre function $P_2^2(\cos\theta) =
  3\sin^2\theta$ and the orthogonality integral
  $\int |P_2^2 \cos(2\lambda)|^2 \sin\theta \, d\theta \, d\lambda =
  48\pi/5$.

---

## 10. SH-Smoothed Shell Thickness (Albedo Map)

### 10.1 Motivation

The albedo-map surface temperature field contains spatial structure at all
scales resolved by the ICO7 mesh (~0.5 deg).  However, lateral thermal
diffusion within the ice shell smooths small-scale temperature variations
before they can affect shell thickness.  Rather than assuming a specific
diffusion timescale, we parameterize the smoothing by truncating the surface
temperature's spherical harmonic (SH) expansion at degree $\ell_{\max}$.
This asks: "what shell thickness pattern results if only harmonics up to
degree $\ell_{\max}$ are retained in the surface temperature?"

The SH smoothing is performed by:
1. FFT along longitude to extract Fourier components at each colatitude
2. Legendre projection of each Fourier component onto $P_\ell^m(\cos\theta)$
3. Reconstruction from harmonics up to $\ell_{\max}$

### 10.2 Convergence of Geophysical Quantities

All quantities are reported for $\ell_{\max}$ = 2, 4, 8, 16, 32, and full
resolution (no truncation).

#### Maxwell, $H$ = 0

| $\ell_{\max}$ | $\langle b \rangle$ (km) | $b_{11}^c$ (km) | $b_{11}^s$ (km) | $b_{20}$ (km) | $b_{22}^c$ (km) | $4t_{22}$ (km) | $t_{20}\!+\!2t_{22}$ (km) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 20.66 | +0.383 | $-1.269$ | $-2.922$ | +1.506 | 6.02 | +0.09 |
| 4 | 20.66 | +0.368 | $-1.283$ | $-2.922$ | +1.495 | 5.98 | +0.07 |
| 8 | 20.67 | +0.366 | $-1.283$ | $-2.903$ | +1.497 | 5.99 | +0.09 |
| 16 | 20.67 | +0.367 | $-1.283$ | $-2.900$ | +1.496 | 5.99 | +0.09 |
| 32 | 20.67 | +0.366 | $-1.283$ | $-2.897$ | +1.496 | 5.99 | +0.10 |
| Full | 20.67 | +0.364 | $-1.286$ | $-2.897$ | +1.497 | 5.99 | +0.10 |

#### Maxwell, $H$ = 10

| $\ell_{\max}$ | $\langle b \rangle$ (km) | $b_{11}^c$ (km) | $b_{11}^s$ (km) | $b_{20}$ (km) | $b_{22}^c$ (km) | $4t_{22}$ (km) | $t_{20}\!+\!2t_{22}$ (km) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 16.77 | +0.304 | $-1.038$ | $-1.271$ | +0.932 | 3.73 | +0.59 |
| 4 | 16.77 | +0.297 | $-1.046$ | $-1.265$ | +0.926 | 3.71 | +0.59 |
| 8 | 16.78 | +0.296 | $-1.046$ | $-1.250$ | +0.928 | 3.71 | +0.61 |
| 16 | 16.78 | +0.297 | $-1.046$ | $-1.246$ | +0.927 | 3.71 | +0.61 |
| 32 | 16.78 | +0.297 | $-1.046$ | $-1.244$ | +0.927 | 3.71 | +0.61 |
| Full | 16.78 | +0.295 | $-1.048$ | $-1.244$ | +0.928 | 3.71 | +0.61 |

#### Flow law $n_G$ = 3

| $\ell_{\max}$ | $\langle b \rangle$ (km) | $b_{11}^c$ (km) | $b_{11}^s$ (km) | $b_{20}$ (km) | $b_{22}^c$ (km) | $4t_{22}$ (km) | $t_{20}\!+\!2t_{22}$ (km) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 26.54 | +0.474 | $-1.650$ | $-1.072$ | +1.228 | 4.91 | +1.38 |
| 4 | 26.54 | +0.467 | $-1.658$ | $-1.054$ | +1.220 | 4.88 | +1.39 |
| 8 | 26.55 | +0.464 | $-1.659$ | $-1.026$ | +1.222 | 4.89 | +1.42 |
| 16 | 26.55 | +0.465 | $-1.659$ | $-1.020$ | +1.221 | 4.89 | +1.42 |
| 32 | 26.55 | +0.465 | $-1.659$ | $-1.017$ | +1.221 | 4.89 | +1.43 |
| Full | 26.56 | +0.462 | $-1.662$ | $-1.016$ | +1.222 | 4.89 | +1.43 |

#### Flow law $n_G$ = 5.2

| $\ell_{\max}$ | $\langle b \rangle$ (km) | $b_{11}^c$ (km) | $b_{11}^s$ (km) | $b_{20}$ (km) | $b_{22}^c$ (km) | $4t_{22}$ (km) | $t_{20}\!+\!2t_{22}$ (km) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 14.14 | +0.250 | $-0.880$ | $-0.247$ | +0.580 | 2.32 | +0.91 |
| 4 | 14.14 | +0.248 | $-0.884$ | $-0.235$ | +0.576 | 2.30 | +0.92 |
| 8 | 14.14 | +0.247 | $-0.884$ | $-0.220$ | +0.577 | 2.31 | +0.93 |
| 16 | 14.14 | +0.247 | $-0.884$ | $-0.217$ | +0.577 | 2.31 | +0.94 |
| 32 | 14.14 | +0.247 | $-0.884$ | $-0.215$ | +0.577 | 2.31 | +0.94 |
| Full | 14.15 | +0.245 | $-0.886$ | $-0.214$ | +0.577 | 2.31 | +0.94 |

### 10.3 Moment of Inertia vs $\ell_{\max}$

| Case | $\Delta(B\!-\!A)/C$ | | $\Delta(B\!-\!C)/A$ | |
|------|:-:|:-:|:-:|:-:|
| | **$\ell_{\max}$=2** | **Full** | **$\ell_{\max}$=2** | **Full** |
| Maxwell, $H$=0 | $6.02 \times 10^{-7}$ | $5.99 \times 10^{-7}$ | $+8.9 \times 10^{-9}$ | $+9.6 \times 10^{-9}$ |
| Maxwell, $H$=10 | $3.73 \times 10^{-7}$ | $3.71 \times 10^{-7}$ | $+5.9 \times 10^{-8}$ | $+6.1 \times 10^{-8}$ |
| Flow law $n_G$=3 | $4.91 \times 10^{-7}$ | $4.89 \times 10^{-7}$ | $+1.4 \times 10^{-7}$ | $+1.4 \times 10^{-7}$ |
| Flow law $n_G$=5.2 | $2.32 \times 10^{-7}$ | $2.31 \times 10^{-7}$ | $+9.1 \times 10^{-8}$ | $+9.4 \times 10^{-8}$ |

### 10.4 Interpretation

1. **All geophysical quantities converge by $\ell_{\max} \approx 4$.**
   The maximum change between $\ell_{\max}=2$ and full resolution is
   $<1$% for most quantities.  The largest relative change is in
   $b_{20}$ (up to 13% for flow law $n_G = 5.2$), reflecting the
   modest contribution of degrees 3-8 to the pole-equator temperature
   contrast.

2. **The albedo map signal is dominated by $\ell \le 2$ harmonics.**
   Europa's leading-hemisphere albedo enhancement is a degree-1 feature,
   and the global mean albedo variation is degree-0.  These are fully
   captured at $\ell_{\max} = 2$.  The tidal thickness pattern
   ($b_{22}^c$) is also degree-2 by construction, so it too is
   captured at $\ell_{\max} = 2$.

3. **SH truncation does not change any qualitative conclusions.**
   The leading/trailing asymmetry ($b_{11}^s \approx -1.3$ km for
   Maxwell $H$=0) and all moment-of-inertia quantities are robust to
   the choice of $\ell_{\max}$.  In particular, $\Delta(B\!-\!A)/C$
   varies by $<0.5$% and $\Delta(B\!-\!C)/A$ by $<8$% across the
   full range of $\ell_{\max}$ tested.

4. **Physical implication:** Because lateral thermal diffusion in the
   ice shell filters out structure at scales below
   $\sqrt{\kappa\,t}$ (where $\kappa$ is the shell's thermal diffusivity
   and $t$ the relevant timescale), and because the shell thickness
   depends almost entirely on $\ell \le 4$ temperature structure,
   the geophysical results are insensitive to the assumed diffusion
   timescale.  Even for very short timescales (high $\ell_{\max}$),
   the results are unchanged.


## 11. Cyclical Polar Wander from Trailing-Hemisphere Darkening

### 11.1 Motivation

Sections 7–10 established that Europa's leading/trailing albedo
asymmetry significantly alters the shell's moment-of-inertia signature.
For the Maxwell $H$=0 case, spatially varying albedo flips
$\Delta(B\!-\!C)/A$ from $-2.9 \times 10^{-8}$ (stable) to
$+1.0 \times 10^{-8}$ (unstable), while all other rheology cases see
30–60% increases in $\Delta(B\!-\!C)/A$.

The leading/trailing albedo contrast is maintained by magnetospheric
plasma bombardment, which preferentially darkens Europa's trailing
hemisphere on timescales of $\sim 10^3$–$10^5$ yr (Paranicas et al.
2001).  If the resulting MoI instability drives true polar wander
(TPW), the reorientation would expose fresh bright ice on the new
trailing hemisphere, resetting the albedo.  This suggests a
self-resetting cycle:

1. Bombardment darkens trailing hemisphere → lowers albedo → raises
   $T_s$ → thins shell
2. Shell thickness redistribution drives $\Delta(B\!-\!C)/A > 0$
   (unstable)
3. Shell reorients (TPW), exposing fresh ice to the trailing side
4. Cycle restarts

However, 3 of 4 rheology cases predict $\Delta(B\!-\!C)/A > 0$ even
at $f=0$ (no darkening), implying permanent instability rather than
cyclical behavior.  Since Europa IS dynamically stable, a stabilizing
mechanism is needed: the **fossil bulge**.

### 11.2 Fossil Bulge

The fossil bulge is a permanent elastic deformation of the cold outer
shell that exactly cancels the $l=2$ shell thickness harmonics when
the surface albedo is uniform.  Physically, the cold outer shell
retains a permanent tidal/rotational deformation that stabilizes
Europa's orientation (O&S 1989, Section 6).

**Model.**  We track the fossil bulge as two $l=2$ harmonic
coefficients $(t_{20}^{\rm fossil}, t_{22}^{\rm fossil})$, initialized
to cancel the shell's equilibrium harmonics at $f=0$:

$$t_{20}^{\rm fossil} = -t_{20}^{\rm shell}(f\!=\!0), \quad
  t_{22}^{\rm fossil} = -t_{22}^{\rm shell}(f\!=\!0)$$

The stability criterion uses the **total** (shell + fossil) harmonics:

$$\Delta(B\!-\!C)/A_{\rm total} = 10^{-7} \times
  \bigl[(t_{20}^{\rm shell} + t_{20}^{\rm fossil})
  + 2(t_{22}^{\rm shell} + t_{22}^{\rm fossil})\bigr] / {\rm km}$$

This gives $\Delta(B\!-\!C)/A_{\rm total} = 0$ at $f=0$ for ALL
rheologies.

### 11.3 Rotation Matrix for Reorientation

At reorientation, the total mass pattern (shell + fossil) rotates by
90° about the sub-Jovian ($x$) axis.  The shell thickness is expanded
using unnormalized associated Legendre polynomials:

$$b(\theta, \lambda) = \langle b \rangle + t_{20}\,P_2^0(\cos\theta)
  + t_{22}\,P_2^2(\cos\theta)\cos 2\lambda$$

where $P_2^0 = (3\cos^2\!\theta - 1)/2$ and $P_2^2 = 3\sin^2\!\theta$.

Direct coordinate substitution under $R_x(\pi/2)$ (active rotation:
$\cos\theta' = \sin\theta\sin\lambda$ for the inverse) gives:

$$P_2^0(\cos\theta') = -\tfrac{1}{2}\,P_2^0(\cos\theta)
  - \tfrac{1}{4}\,P_2^2(\cos\theta)\cos 2\lambda$$

$$P_2^2(\cos\theta')\cos 2\lambda' = -3\,P_2^0(\cos\theta)
  + \tfrac{1}{2}\,P_2^2(\cos\theta)\cos 2\lambda$$

Therefore the 2×2 rotation matrix for the unnormalized $(t_{20},
t_{22})$ coefficients is:

$$M = \begin{pmatrix} -1/2 & -3 \\ -1/4 & 1/2 \end{pmatrix}$$

This matrix has $\det(M) = -1$ (reflection) and $M^2 = I$ (involution),
verified numerically.  The asymmetry arises because $P_2^0$ and
$P_2^2$ have different norms ($\int P_{20}^2 \sin\theta\,d\theta\,d\lambda
= 4\pi/5$ vs $\int (P_{22}\cos 2\lambda)^2 \sin\theta\,d\theta\,d\lambda
= 48\pi/5$).  The quadratic invariant $t_{20}^2 + 12\,t_{22}^2$ is
preserved by $M$.

**Important:** For rotation angles $\beta \neq 90°$, the $(t_{20},
t_{22})$ subspace is not closed — the rotation mixes in $Y_{21}$
components.  The 2D matrix is valid only for $\beta = \pi/2$.

### 11.4 Reorientation Logic

At each reorientation event:

1. Compute total harmonics: $(t_{20}, t_{22})_{\rm total} =
   (t_{20}, t_{22})_{\rm shell} + (t_{20}, t_{22})_{\rm fossil}$
2. Rotate total: $(t_{20}, t_{22})_{\rm rot} = M \cdot
   (t_{20}, t_{22})_{\rm total}$
3. Reset shell: $f \to 0$, $b \to b_{\rm eq}(f\!=\!0)$
4. New fossil: $(t_{20}, t_{22})_{\rm fossil}^{\rm new} =
   (t_{20}, t_{22})_{\rm rot} - (t_{20}, t_{22})_{\rm shell}(f\!=\!0)$

The rotated total equals the physical mass distribution in the new
body frame.  Subtracting the shell's $f=0$ equilibrium gives the
residual permanent deformation (the new fossil bulge).

### 11.5 Model Details

Implemented in `scripts/polar_wander_cycle.py`.

**Albedo darkening.**  The surface albedo varies as:

$$A(\theta, \lambda, f) = A_{\rm bright} + (A_{\rm dark} - A_{\rm bright})\,f\,w(\lambda)$$

where $f \in [0, 1]$ is the darkening fraction,
$w(\lambda) = \max(0, \sin\lambda)$ is the trailing-hemisphere
weighting (peak at $\lambda = +\pi/2 = 90°\,\mathrm{E}$), $A_{\rm bright} = 0.55$
(fresh ice), and $A_{\rm dark} = 0.15$ (irradiated surface).

**Surface temperature.**  From O&S radiative equilibrium with
spatially varying albedo:

$$T_s(\theta, \lambda) = \left[\frac{(1-A)\,F_A(\theta)}{\sigma}\right]^{1/4}$$

**Shell thickness and MoI.**  The equilibrium thickness $b(\theta,
\lambda)$ is computed from O&S Eq 2.8 with the spatially varying
$T_s$.  The tidal dissipation integral $D(\theta, \lambda)$ is
pre-computed once per rheology case.

**Time-dependent evolution.**  Operator-splitting with timestep
$\Delta t = 10^3$ yr:

1. Exponential darkening: $f(t+\Delta t) = 1 - (1-f(t))\,e^{-\Delta t/\tau_{\rm dark}}$
2. Surface temperature from current albedo (instantaneous)
3. Equilibrium thickness from current $T_s$
4. Thickness relaxation: $b(t+\Delta t) = b_{\rm eq} + (b(t) - b_{\rm eq})\,e^{-\Delta t/\tau_{\rm shell}}$
5. Reorientation check: if $\Delta(B\!-\!C)/A_{\rm total} >
   \Delta_{\rm threshold}$, apply the rotation and reset

**Instability threshold.**  With exact fossil cancellation at $f=0$,
any infinitesimal darkening drives $\Delta(B\!-\!C)/A_{\rm total} > 0$.
A finite threshold ($\Delta_{\rm threshold} = 5 \times 10^{-9}$) is
needed to represent the viscous resistance to TPW.

### 11.6 Quasi-Static Results

Without the fossil bulge (shell-only harmonics):

| Rheology | $\Delta(B\!-\!C)/A$ at $f$=0 | $\Delta(B\!-\!C)/A$ at $f$=1 | $f_{\rm crit}$ |
|----------|----------------------------:|----------------------------:|--------:|
| Maxwell $H$=0 | $-1.26 \times 10^{-8}$ | $+6.19 \times 10^{-8}$ | **0.124** |
| Maxwell $H$=10 | $+4.37 \times 10^{-8}$ | $+9.94 \times 10^{-8}$ | — |
| Flow $n_G$=3 | $+1.15 \times 10^{-7}$ | $+2.01 \times 10^{-7}$ | — |
| Flow $n_G$=5.2 | $+7.91 \times 10^{-8}$ | $+1.27 \times 10^{-7}$ | — |

With the fossil bulge, ALL rheologies start at $\Delta(B\!-\!C)/A = 0$
at $f=0$ and increase monotonically with darkening.  The sensitivity
$d[\Delta(B\!-\!C)/A]/df$ differs between rheologies, being largest
for the flow laws.

### 11.7 Time-Dependent Cycle Period

For the Maxwell $H$=0 case with $\tau_{\rm dark} = 10^4$ yr,
$\tau_{\rm shell} = 10^7$ yr, and fossil bulge enabled:

- **Cycle period: 1.5 Myr** (5 events over 6.6 Myr)
- Darkening saturates in $\sim 5\tau_{\rm dark} = 0.05$ Myr
- The bottleneck is thermal relaxation of the shell thickness
- Mean thickness oscillates between 20.15 km (dark) and 20.31 km (bright)

The fossil bulge evolves gradually across cycles.  Starting from
$(t_{20}^{\rm fossil}, t_{22}^{\rm fossil}) = (3.095, -1.485)$ km, the
90° rotation at each reorientation shifts the harmonics by
$\sim 0.05$ km per cycle.  The quadratic invariant
$t_{20}^2 + 12\,t_{22}^2$ is conserved by the rotation.

### 11.8 Parameter Survey

Sweeping $\tau_{\rm dark} \in [10^3, 10^5]$ yr and $\tau_{\rm shell}
\in [10^6, 10^8]$ yr:

| $\tau_{\rm shell}$ | Cycle period |
|-------------------:|-------------:|
| $10^6$ yr | $\sim 0.2$ Myr |
| $10^7$ yr | $\sim 1.5$ Myr |
| $10^8$ yr | $\sim 15$ Myr |

The cycle period scales linearly with $\tau_{\rm shell}$ and is
insensitive to $\tau_{\rm dark}$ (confirmed across two decades).

### 11.9 Discussion

**Physical viability.** The fossil bulge model makes all four rheology
cases viable for cyclical polar wander.  The cycle period of
$\sim 1$–$2$ Myr (for $\tau_{\rm shell} = 10^7$ yr) is fast enough to
be geologically significant but slow enough for the magnetospheric
darkening to accumulate.

**Self-consistency.** The key test is whether the system remains
stable after each reorientation.  With the rotation matrix $M$ and
small mass anomaly at threshold, the rotated total is close to zero,
producing a new fossil bulge that again nearly cancels the shell
harmonics.  The fossil evolves gradually over many cycles rather than
diverging.

**Threshold dependence.** The cycle period depends on the instability
threshold $\Delta_{\rm threshold}$, which represents the minimum
$\Delta(B\!-\!C)/A$ needed to overcome viscous/elastic resistance to
TPW.  For $\Delta_{\rm threshold} = 5 \times 10^{-9}$, the period is
$\sim 1.5$ Myr; for $\Delta_{\rm threshold} = 2 \times 10^{-8}$, it
increases to $\sim 8$ Myr.

**Limitations.**
1. The model assumes instantaneous 90° reorientation.  In reality,
   reorientation is gradual and the rotation angle depends on the
   viscous/elastic structure.
2. The instability threshold is a free parameter; a viscous-relaxation
   model would provide a physical constraint.
3. The fossil bulge is treated as a point in $(t_{20}, t_{22})$ space;
   the full spatial pattern is not tracked.
4. The shell thermal response timescale $\tau_{\rm shell}$ is poorly
   constrained ($10^6$–$10^8$ yr).

**Observational implications.** Multiple reorientation events over
Europa's $\sim 100$ Myr surface age would produce global-scale
tectonic features.  The gradually evolving fossil bulge means
successive reorientations are not identical, potentially producing
systematic trends in fracture orientations.

### 11.10 Full Inertia Tensor Model

The $l=2$ model (Sections 11.2–11.9) uses only the $(t_{20}, t_{22})$
harmonic coefficients and assumes fixed 90° rotations about the
sub-Jovian axis.  This has three limitations: (1) the 2D
$(t_{20}, t_{22})$ subspace cannot represent the full inertia tensor;
(2) rotations by angles other than 90° mix in $Y_{21}$ components
not captured by the 2D space; (3) the instability threshold is an
ad hoc parameter rather than arising from a physical model.

The **full inertia tensor** model addresses these by computing the
3×3 inertia tensor directly from the shell thickness field
$b(\theta, \lambda)$ on the full $(181 \times 361)$ grid:

$$I_{ij} = \rho R^4 \iint b(\theta,\lambda)\,
  (\delta_{ij} - \hat{r}_i \hat{r}_j)\,
  \sin\theta\,d\theta\,d\lambda$$

where $\hat{r} = (\sin\theta\cos\lambda,\,\sin\theta\sin\lambda,\,
\cos\theta)$.  The integration uses trapezoidal weights at the
longitude endpoints ($\lambda = \pm\pi$ are identical points) to
eliminate quadrature anisotropy ($I_{xx} \neq I_{yy}$ for a uniform
shell without the half-weighting correction).

**Eigenvalue decomposition.** The total inertia tensor
$I_{\rm total} = I_{\rm shell} + I_{\rm fossil}$ is diagonalized via
`np.linalg.eigh`.  The eigenvector of the maximum eigenvalue
($\hat{v}_{\rm max}$) gives the lowest-energy spin axis.  The
**misalignment angle** $\delta = \arccos(|\hat{v}_{\rm max} \cdot
\hat{s}|)$ quantifies the spin pole's offset from equilibrium.

**Tidal coordinate transformation.** The O&S tidal strain formulas
(Appendix B) use $\theta$ = colatitude from the **spin axis** and
$\lambda$ = longitude from the sub-Jovian meridian, both in the
**spin frame**.  When the spin pole $\hat{s}$ is not aligned with
the body-frame $\hat{z}$, these tidal coordinates differ from the
body-frame grid.  For each body-frame grid point $\hat{r}$:

$$\theta_{\rm tidal} = \arccos(\hat{r} \cdot \hat{s}), \qquad
  \lambda_{\rm tidal} = \operatorname{atan2}(
  \hat{r}_\perp \cdot \hat{e}_{\rm trail},\,
  \hat{r}_\perp \cdot \hat{e}_{\rm sub})$$

where $\hat{e}_{\rm sub}$ is the sub-Jovian direction projected
onto the equatorial plane of $\hat{s}$, and $\hat{e}_{\rm trail}
= \hat{s} \times \hat{e}_{\rm sub}$ is the trailing direction.
Tidal heating $D$, darkening weight $w$, and solar flux all depend
on $\hat{s}$ and are recomputed as the spin pole drifts.  The
inertia tensor and fossil bulge are computed in body-frame
coordinates (they represent physical mass distributions).

To avoid recomputing $D$ at every timestep (expensive for flow-law
rheologies), it is cached and only updated when the spin pole has
moved by more than 1° since the last computation.

**Fossil bulge as a 2D field.** The fossil is represented as a full
2D field $b_{\rm fossil}(\theta,\lambda)$, initialized as:

$$b_{\rm fossil} = (\alpha - 1) \times
  \bigl[b_{\rm eq}(f\!=\!0) - \langle b_{\rm eq} \rangle\bigr]$$

where $\alpha > 1$ is the **fossil bulge factor** (default 1.5).
This REINFORCES the shell's natural stable pattern ($C > B > A$),
providing a genuine stability margin: with $\alpha = 1.5$, the fossil
adds 50% extra stabilization beyond the shell's own MoI structure.

**Fossil relaxation.** The fossil relaxes toward the current
equilibrium anomaly on timescale $\tau_{\rm visc}$:

$$b_{\rm fossil}(t + \Delta t) = b_{\rm target}
  + (b_{\rm fossil}(t) - b_{\rm target})\,
  e^{-\Delta t / \tau_{\rm visc}}$$

where $b_{\rm target} = (\alpha - 1) \times
[b_{\rm eq}(t) - \langle b_{\rm eq}(t) \rangle]$.  This models
continuous lithospheric freezing: during stable periods the fossil
adapts to reinforce the current shell; after perturbation the fossil
is "out of date" and evolves toward the new target on
$\tau_{\rm visc} \sim 3 \times 10^7$ yr.

### 11.11 Continuous Polar Wander

Rather than discrete 90° jumps, the full tensor model implements
**continuous polar wander** where the spin pole drifts gradually
toward the maximum-MoI eigenvector $\hat{v}_{\rm max}$:

$$\hat{s}(t + \Delta t) = \hat{s}\cos\Delta\alpha
  + \hat{v}_\perp \sin\Delta\alpha, \qquad
  \Delta\alpha = \delta\,(1 - e^{-\Delta t / \tau_{\rm TPW}})$$

where $\delta$ is the current misalignment angle,
$\hat{v}_\perp$ is the component of $\hat{v}_{\rm max}$
perpendicular to the current spin pole, and $\tau_{\rm TPW} \sim
10^5$ yr is the TPW relaxation timescale (controlled by viscous
coupling between the shell and interior).

**Typical behavior.** With default parameters ($\tau_{\rm dark} =
10^4$ yr, $\tau_{\rm shell} = 10^7$ yr, $\tau_{\rm TPW} = 10^5$ yr,
$\alpha = 1.5$, $\tau_{\rm visc} = 3 \times 10^7$ yr):

- The pole remains stable for $\sim 5$ Myr while the fossil bulge
  decays, despite the surface having fully darkened
- Once the fossil weakens sufficiently, the pole enters a regime
  of **continuous rotation** along the trailing–leading great circle
  ($\phi = \pm 90°$)
- Each full revolution ($0° \to 180° \to 0°$) takes $\sim 0.8$ Myr
- The misalignment angle $\delta$ spikes to $\sim 80°$ during the
  initial breakout, then settles to $\sim 40°$–$50°$ oscillation
  during steady-state rotation
- Mean shell thickness decreases in steps as the pole passes through
  $\theta \approx 90°$ where solar heating is most redistributed

**Why continuous rotation instead of oscillation.** Because tidal
heating and darkening both rotate with the spin pole (computed in
tidal/spin-frame coordinates), the shell equilibrium tracks the
current pole direction rather than being anchored to the body frame.
Trailing-hemisphere darkening always thins the shell on the current
trailing side, shifting $\hat{v}_{\rm max}$ toward the current
leading direction.  As the pole follows, the trailing direction
rotates with it, maintaining the driving torque.  There is no
equilibrium: the pole continuously chases a target that moves with
it.  The only body-frame anchor is the fossil bulge, which decays on
$\tau_{\rm visc}$; once it has weakened, the rotation is self-sustaining.

The trajectory is strictly confined to the trailing–leading meridian
($\phi = \pm 90°$) because the darkening asymmetry lies purely in
this direction (the sub-Jovian/anti-Jovian axis sees no darkening
contrast).

**Timescale hierarchy.** The dynamics are controlled by four
timescales:

| Timescale | Symbol | Default | Role |
|-----------|--------|---------|------|
| Darkening | $\tau_{\rm dark}$ | $10^4$ yr | Drives instability |
| TPW relaxation | $\tau_{\rm TPW}$ | $10^5$ yr | Pole drift rate |
| Shell thermal | $\tau_{\rm shell}$ | $10^7$ yr | Shell adjustment (damping) |
| Fossil viscous | $\tau_{\rm visc}$ | $3 \times 10^7$ yr | Fossil decay (stabilization) |

The onset time of TPW is controlled by $\tau_{\rm visc}$ (fossil
decay); the rotation period is set by $\tau_{\rm TPW}$ and the
magnitude of the darkening-driven MoI misalignment; $\tau_{\rm dark}$
determines how quickly the driving force saturates.

**CLI flags:**

```
--fossil-factor F       Fossil bulge factor α (default: 1.5)
--tau-visc T            Fossil relaxation timescale [yr] (default: 3e7)
--tau-tpw T             TPW relaxation timescale [yr] (default: 1e5)
--misalignment-threshold D  Milestone logging threshold [deg] (default: 5.0)
--no-full-inertia       Use l=2 harmonics instead of full tensor
--dt DT                 Time step [yr] (default: tau_tpw/10)
--t-max T               Total simulation time [yr] (default: 1e8)
```

### 11.12 Physically-Derived TPW Dynamics

Sections 11.10–11.11 used an exponential relaxation timescale
$\tau_{\rm TPW} = 10^5$ yr as a **free parameter**.  However, the
physical TPW timescale on Europa is set by the balance between driving
torques (from inertia tensor misalignment) and dissipation (viscous
deformation of the ice shell).  Ojakangas & Stevenson (1989, hereafter
O&S) and Burnett & Hayne (2023, hereafter B&H) provide analytical
models that constrain this timescale.

The free $\tau_{\rm TPW}$ has two problems: (1) it is unconstrained
(6–100× faster than physically-derived estimates); (2) it is
independent of the shell's rheological state, so there is no feedback
between shell evolution and reorientation dynamics.

We replace the free parameter with three selectable dynamics modes:

1. **`exponential`** — original behavior (backward compatible)
2. **`ospw`** — O&S overdamped viscous dissipation with iceberg effect
3. **`coupled`** — OSPW dissipation + B&H gravity-gradient
   shell–interior coupling

### 11.13 OSPW Overdamped Model

The O&S (1989) TPW model treats reorientation as an overdamped
pendulum: inertial terms are negligible and the angular velocity is
set by the torque balance between tidal driving and viscous friction.

**Driving torque.** A shell triaxiality $\varepsilon = \Delta I / I_0$
misaligned from the tidal axis by angle $\psi$ produces:

$$\tau_{\rm drive} = \frac{1}{2}\,\varepsilon\,n^2\,I_0\,\sin 2\psi$$

where $n$ is Europa's mean orbital motion and $I_0$ the shell moment
of inertia.

**Viscous friction.** O&S compute a friction coefficient $C_f$ from
the rate of viscous dissipation in the shell during steady rotation:

$$C_f = \frac{16\pi}{15}\,\rho_{\rm ice}\,R^4\,b_c\,\frac{n^2}{l(l+1)}\,
  \int_0^{b_c} \frac{r^{2l+2}}{\eta(z)}\,dz$$

where the viscosity profile $\eta(z)$ follows an Arrhenius law with
activation parameter $l = 15$ and melting viscosity
$\eta_0 = 3 \times 10^{14}$ Pa s.  The viscosity increases
exponentially from the base toward the surface: $\eta(z) = \eta_0
\exp[l\,(T_m - T(z))/(T_m - T_s)]$.

**Overdamped angular velocity.** In the overdamped limit ($\ddot\psi
\approx 0$):

$$\omega = \frac{2\varepsilon^2 n^2 I_0}{C_f}\,\sin 2\psi$$

**OSPW epsilon correction.** The geometric triaxiality
$\varepsilon_{\rm geom} = \Delta I_{\rm shell} / I_0$ overestimates
the effective driving by $\sim 20\times$ because it does not account
for the ocean's isostatic response to shell thickness variations.
O&S show that the gravitational potential harmonics are dominated by
the bottom-loading (ice–ocean density contrast), not the surface
topography.  The correction factor is:

$$\varepsilon_{\rm eff} = \varepsilon_{\rm geom} \times
  \frac{\Delta\rho}{\rho_{\rm ice}} \times
  \left[-(1 - C_w) + 4\,C_w\,\frac{b_c}{R}\,
  \frac{\rho_{\rm ice}}{\Delta\rho}\right]$$

where $C_w \approx 0.98$ is the compensation factor and
$\Delta\rho = \rho_{\rm ice} - \rho_{\rm ocean} \approx -83$ kg/m³.
For Europa defaults, this gives
$\varepsilon_{\rm eff} / \varepsilon_{\rm geom} \approx 0.048$.

**Iceberg amplification.** The O&S "iceberg effect" amplifies gravity
harmonics by a factor of $\sim 8.4\times$ through the bottom-loading
bracket in the equation above.  This is applied to the shell's
inertia tensor anomaly (non-isotropic part) before computing the
triaxiality parameter.

**OSPW timescale.** The characteristic reorientation timescale is:

$$\tau_f = \sqrt{\frac{C_f\,\tau_{\rm th}}{2\varepsilon^2 n^2 I_0}}
  \sim 1\;\text{Myr}$$

at $T_f = 130$ K, compared with $\tau_{\rm TPW} = 0.1$ Myr used in
the exponential model.

**Implementation.** The `_ospw_step()` method computes the angular
displacement per timestep from the instantaneous torque balance.  The
driving torque uses the misalignment angle from the full 3×3 inertia
tensor, and the fossil bulge decays explicitly via $b_{\rm fossil}$,
so the O&S growth factor $(t/\tau_{\rm th})$ is captured automatically
without special treatment.

### 11.14 Burnett & Hayne Interior Coupling

The coupled model extends OSPW by adding gravity-gradient coupling
between the ice shell and the rocky interior, following B&H (2023).

**Shell–interior coupling.** The interior has its own spin axis,
initially aligned with the body $z$-axis (tidal lock).  The coupling
torque is:

$$\tau_{\rm couple} = K_G\,\sin 2\delta$$

where $\delta$ is the angular separation between shell and interior
spin axes, and $K_G = \frac{3}{2}(B_i - A_i)\,n^2$ depends on the
interior's triaxiality $(B_i - A_i)/C_i = \varepsilon_i$.

**Two-body dynamics.** Two torques act on the shell in generally
different directions:

1. Driving torque → pushes the shell toward $\hat{v}_{\rm max}$ (the
   lowest-energy orientation)
2. Coupling torque → pulls the shell toward the interior pole

The interior feels the opposite coupling torque and drifts toward the
shell pole on a timescale modulated by $\gamma = C_i/C_s \approx 28$.

**Coupling regimes.** The behavior depends critically on the interior
triaxiality $\varepsilon_i$:

- **Rigid interior** ($\varepsilon_i \sim 10^{-5}$): $K_G$ is small
  ($\sim 2.6 \times 10^{19}$ N m), coupling is negligible, shell
  rotates freely as in the OSPW-only case.
- **Hydrostatic interior** ($\varepsilon_i \sim 5 \times 10^{-4}$):
  $K_G$ is large ($\sim 1.3 \times 10^{21}$ N m), coupling resists
  shell reorientation and can delay or suppress TPW.

**Implementation.** The `_coupled_step()` method computes a 2D
displacement vector on the unit sphere:

$$\delta\hat{s} = \omega_{\rm drive}\,\Delta t\,\hat{v}_\perp
  + \omega_{\rm couple}\,\Delta t\,\hat{v}_{\rm couple}$$

where $\hat{v}_\perp$ points from $\hat{s}$ toward
$\hat{v}_{\rm max}$ and $\hat{v}_{\rm couple}$ points from $\hat{s}$
toward the interior pole.  The step size is limited to prevent
numerical overshooting ($|\delta\hat{s}| \leq \psi$).  The interior
pole is updated separately.

### 11.15 Simulation Results: Model Comparison

All simulations use the Maxwell $H$=0 rheology with default
parameters ($\tau_{\rm dark} = 10^4$ yr, $\tau_{\rm shell} = 10^7$ yr,
$\alpha = 1.5$, $\tau_{\rm visc} = 3 \times 10^7$ yr, $t_{\rm max} =
10^8$ yr).

**Exponential TPW** ($\tau_{\rm TPW} = 10^5$ yr).  Onset at
$\sim 5$ Myr, then continuous rotation at $\sim 450$ deg/Myr along
the trailing–leading meridian.  The free $\tau_{\rm TPW}$ is
unconstrained.  See `docs/figures/polar_wander/`.

**OSPW overdamped**.  Onset at $\sim 5$ Myr.  The physically-derived
timescale ($\tau_f \sim 1$ Myr) governs the angular velocity, which
is of order $\sim 2500$ deg/Myr during active rotation.  The dynamics
are qualitatively similar to the exponential case — continuous
rotation after the fossil decays — but the rotation rate is determined
by the shell's viscous structure rather than a free parameter.
See `docs/figures/polar_wander_ospw/`.

**Coupled — weak** ($\varepsilon_i = 10^{-5}$, rigid interior).  With
a rigid interior, the gravity-gradient coupling torque is $\sim 50\times$
weaker than needed to resist OSPW driving.  The shell rotates
essentially freely, reproducing the OSPW-only behavior.  The interior
pole remains nearly fixed.  See `docs/figures/polar_wander_coupled/`.

**Coupled — strong** ($\varepsilon_i = 5 \times 10^{-4}$,
hydrostatic interior).  The coupling torque delays the onset of TPW
to $\sim 16$ Myr (vs $\sim 5$ Myr without coupling).  The initial
breakout is a discrete $\sim 90°$ jump, followed by continuous
rotation.  The shell–interior separation reaches large values
($\sim 180°$), indicating that the OSPW driving torque eventually
overcomes the coupling.  See
`docs/figures/polar_wander_coupled_strong/`.

| Mode | Onset [Myr] | Rotation character | Interior behavior |
|------|:-----------:|-------------------|-------------------|
| Exponential | $\sim 5$ | Continuous, $\sim 450$ deg/Myr | N/A |
| OSPW | $\sim 5$ | Continuous, $\sim 2500$ deg/Myr | N/A |
| Coupled (rigid) | $\sim 5$ | Continuous, $\sim 2700$ deg/Myr | Fixed |
| Coupled (hydrostatic) | $\sim 16$ | Episodic → continuous | Dragged slowly |

### 11.16 Discussion of TPW Dynamics

**Timescale comparison.** The exponential model's free $\tau_{\rm TPW}
= 10^5$ yr is $\sim 10\times$ shorter than the OSPW-derived $\tau_f
\sim 10^6$ yr.  This means the exponential model produces artificially
fast TPW.  However, because the continuous rotation regime is
self-sustaining (the driving torque rotates with the pole), the
rotation rate is not simply $1/\tau$.  In both models, the rotation
persists indefinitely once the fossil bulge has decayed.

**OSPW epsilon correction.** The correction from geometric to
effective epsilon ($\varepsilon_{\rm eff} \approx 0.048 \times
\varepsilon_{\rm geom}$) accounts for the ocean's isostatic response.
Without this correction, the OSPW model would overestimate the driving
torque by $\sim 20\times$.

**Interior coupling as a clock.** The most significant effect of
interior coupling is **delaying** the onset of TPW.  With a
hydrostatic interior, the $\sim 16$ Myr delay is set by the time
required for (a) the fossil to decay below the coupling resistance
threshold and (b) the driving torque to overcome the restoring
gravity gradient.  This suggests that interior coupling could
determine the recurrence interval of TPW on Europa, if the interior
is close to hydrostatic equilibrium.

**Regime sensitivity.** The coupled model's behavior is highly
sensitive to the interior triaxiality $\varepsilon_i$, which is poorly
constrained for Europa.  A rigid interior ($\varepsilon_i \sim
10^{-5}$) produces negligible coupling; a hydrostatic interior
($\varepsilon_i \sim 5 \times 10^{-4}$) produces strong coupling with
qualitative changes in behavior (delayed onset, episodic jumps).
Measuring Europa's libration amplitude or gravitational field at
degree-2 could constrain $\varepsilon_i$.

**Limitations.**
1. The OSPW epsilon correction assumes full isostatic compensation
   ($C_w \approx 0.98$); partial compensation would increase the
   effective driving.
2. The interior is treated as a rigid body with fixed triaxiality;
   in reality, viscoelastic relaxation would evolve $\varepsilon_i$
   over the TPW timescale.
3. The 2D vector-sum approach in `_coupled_step()` is a first-order
   (forward Euler) approximation; for large timesteps or strong
   coupling, higher-order integration may be needed.
4. The model does not include tidal dissipation within the interior,
   which could provide additional damping of the shell–interior
   differential rotation.

**CLI flags** (new, in addition to those listed in Section 11.11):

```
--tpw-model MODEL      TPW dynamics: exponential, ospw, or coupled
--ospw-Tf T            Fracture base temperature [K] (default: 130)
--ospw-eta0 ETA        Melting viscosity [Pa s] (default: 3e14)
--ospw-l L             Arrhenius activation parameter (default: 15)
--ospw-bc BC           Shell thickness [km] (default: 20)
--ospw-d-lith D        Lithosphere thickness [km] (default: 2)
--no-iceberg           Disable iceberg amplification
--iceberg-amp F        Override iceberg amplification factor (0=auto)
--interior-gamma G     C_interior/C_shell ratio (default: 28)
--interior-triaxiality E  Interior (B-A)/C (default: 1e-5)
--interior-KG-factor F K_G/K_G* coupling strength (default: 0.5)
```

**References.**
- Ojakangas, G. W., & Stevenson, D. J. (1989). Polar wander of an
  ice shell on Europa. *Icarus*, 81, 242–270.
- Burnett, P. J., & Hayne, P. O. (2023). Coupled spin dynamics of
  Europa's ice shell and interior. *Planetary Science Journal*,
  submitted.
