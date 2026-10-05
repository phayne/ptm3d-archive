# Europa's Ice Shell: Thickness Structure & Polar-Wander Dynamics

> **Status (2026-10-04): the numbers in this document are superseded.** They were
> computed before the 2026-10 corrections to the shell and polar-wander code: the
> thickness law is now O&S Eq. 2.8 with $C_R = 0.5$ (the earlier solver dropped a
> factor of 2 and was tuned to $C_R = 0.85$); every harmonic projection and inertia sum
> now uses a periodic grid (the earlier grids counted the λ = ±180° meridian twice); the
> polar-wander dynamics follow OSPW Eqs. 2.11, 2.18, 3.8–3.10 and 5.39; and the
> pole is seeded with a physical offset $\psi_0$. The inputs are also March-2026 PTM3D
> products. Regenerate before quoting any value.

**Coupling a 3-D thermal model (PTM3D) to the Ojakangas & Stevenson (1989) equilibrium ice-shell model**

**Paul O. Hayne — CU Boulder / LASP**

*Presentation brief. Each section below is sized to become one slide; the Backup section holds supporting detail.*

---

## 1. Motivation & Approach

**Question:** What is the 3-D thickness structure of Europa's ice shell, and is the resulting mass distribution dynamically stable against (i) **true polar wander (TPW)** and (ii) **non-synchronous rotation (NSR)** — the latter driven by tidal torques acting on shell-thickness/mass asymmetries?

**Method (pipeline):** PTM3D thermal model → surface-temperature field $T_s(\theta,\lambda)$ → O&S (1989) equilibrium shell model → thickness $b(\theta,\lambda)$ → spherical harmonics → moments of inertia / stability. *(PTM3D: ICO7 mesh, 327,680 faces, 288 timesteps/orbit, N–S symmetrized.)*

**Two new ingredients vs. classic 1-D O&S:**
- **Longitude-dependent surface temperature** — arising from insolation differences due to **eclipses by Jupiter** and **Jupiter-shine** (sub-Jovian / anti-Jovian asymmetry).
- **Spatially variable albedo map** (leading / trailing asymmetry).

**Rheology cases:** $H$ = basal core heat flux (**0 vs 10 mW m⁻²**); $n_G$ = Glen's-law creep exponent (**3 vs 5.2**).

**Symbols:** $C_R$ = tidal-strain correction factor (Ross & Schubert 1987); $e$ = orbital eccentricity (0.01); $a$ = Europa radius (1569 km); $\rho$ = mean density (3040 kg m⁻³); $g$ = surface gravity (1.31 m s⁻²).

---

# Part 1 — Equilibrium Shell Thickness & Structure

## 2. Surface temperature: 3-D model vs. analytical

![Surface temperature: O&S analytical, PTM3D, and difference](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/albedo_map/fig5_surface_temperature.png)

- **O&S analytical (a):** smooth; equator 109 K, pole 52 K.
- **PTM3D (b):** equator 105–110 K, pole ~46 K, with real spatial structure.
- **Difference (c):** reveals the hemispheric temperature asymmetries — sub-/anti-Jovian (~2–4 K, from eclipses + Jupiter-shine) and leading/trailing (from albedo) — that 1-D models cannot capture.

---

## 3. Equilibrium shell thickness — four rheologies

![Equilibrium ice-shell thickness, albedo-map surface temperatures](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/albedo_map/fig1_thickness_mollweide.png)

Mean thickness ranges from ~14 km (stiff) to ~27 km (soft). The bright equatorial band is **tidal-dissipation thickening** ($t_{22}$ pattern).

| Case | O&S (km) | PTM3D, albedo-map (km) | Paper (km) |
|------|:-------:|:----------------------:|:----------:|
| Maxwell, $H$=0   | 19.7 | 20.7 | 23 |
| Maxwell, $H$=10  | 16.0 | 16.8 | 21 |
| Flow law $n_G$=3 | 25.4 | 26.6 | 25 |
| Flow law $n_G$=5.2 | 13.5 | 14.1 | 13 |

Flow-law cases match the original paper within **2–4%**; Maxwell cases run 14–24% thinner (structural offset, not resolvable by $C_R$ tuning).

---

## 4. Hemispheric asymmetries from 3-D temperature & albedo

The 3-D temperature field adds **degree-1 (hemispheric) harmonics** absent from the 1-D model:

| Harmonic | Physical meaning | Maxwell $H$=0 (km) | Range across rheologies (km) |
|----------|------------------|:------------------:|:----------------------------:|
| $b_{11}^c$ | sub-Jovian / anti-Jovian | $+0.36$ | $+0.25$ to $+0.46$ |
| $b_{11}^s$ | **leading / trailing** | $-1.29$ | $-0.89$ to $-1.66$ |
| $b_{22}^c$ ($=t_{22}$) | tidal pattern | $+1.50$ | $+0.58$ to $+1.22$ |

- **Longitude-dependent $T_s$** → cooler sub-Jovian hemisphere ~0.2–0.3 km thicker ($b_{11}^c$).
- **Albedo map** → dominant non-tidal signal: brighter, colder **leading hemisphere ~1–1.7 km thicker** ($b_{11}^s$) — comparable to the tidal $b_{22}$ term.
- A small $b_{22}^s$ ($-0.07$ to $-0.13$ km) rotates the tidal pattern a few degrees in longitude.

> **Superscripts:** $b_{\ell m}^c$ and $b_{\ell m}^s$ are the $\cos(m\lambda)$ and $\sin(m\lambda)$ components of the spherical harmonic of degree $\ell$, order $m$ (here $\ell$ = 1, 2).

---

## 5. Moments of inertia & stability

$$\Delta\!\left(\tfrac{B-A}{C}\right) \approx 10^{-7}\,(4t_{22}), \qquad
  \Delta\!\left(\tfrac{B-C}{A}\right) \approx 10^{-7}\,(t_{20}+2t_{22})$$

| Case | $\Delta(B\!-\!A)/C$ (albedo-map) | $\Delta(B\!-\!C)/A$ const-albedo | $\Delta(B\!-\!C)/A$ albedo-map |
|------|:-------------------------------:|:-------------------------------:|:-----------------------------:|
| Maxwell, $H$=0   | $5.99\times10^{-7}$ | $-2.9\times10^{-8}$ (stable) | $+1.0\times10^{-8}$ (**unstable**) |
| Maxwell, $H$=10  | $3.71\times10^{-7}$ | $+2.9\times10^{-8}$ | $+6.1\times10^{-8}$ |
| Flow law $n_G$=3 | $4.89\times10^{-7}$ | $+9.0\times10^{-8}$ | $+1.4\times10^{-7}$ |
| Flow law $n_G$=5.2 | $2.31\times10^{-7}$ | $+6.6\times10^{-8}$ | $+9.4\times10^{-8}$ |

- **$\Delta(B\!-\!A)/C$ is robust** (<1% albedo sensitivity) — set by the tidal dissipation pattern.
- **$\Delta(B\!-\!C)/A$ is albedo-controlled** (30–60% change). For Maxwell $H$=0, variable albedo **flips it from stable to marginally unstable** → motivates polar wander.

---

# Part 2 — Polar-Wander Dynamics

## 6. Polar Wander

Magnetospheric plasma preferentially **darkens the trailing hemisphere** on $10^3$–$10^5$ yr timescales (Paranicas et al. 2001; **Burnett & Hayne 2021**). This suggests a **self-resetting cycle**:

1. Darkening → lower albedo → higher $T_s$ → **thins** trailing-hemisphere shell
2. Mass redistribution drives $\Delta(B\!-\!C)/A>0$ → **instability**
3. Shell reorients (TPW) → exposes fresh bright ice on the new trailing side
4. Albedo resets → cycle restarts

A stabilizer is required (3 of 4 rheologies are unstable even without darkening): the **fossil bulge** — a permanent elastic $\ell=2$ deformation of the cold outer shell.

---

## 7. Darkening drives instability; the fossil bulge sets the baseline

![Instability vs. darkening fraction, with and without fossil bulge](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/polar_wander/fig5_all_rheologies.png)

- **Without fossil bulge (left):** 3 of 4 rheologies have $\Delta(B\!-\!C)/A>0$ even at $f=0$ → *permanent* instability (unphysical — Europa is stable today).
- **With fossil bulge (right):** all rheologies start at $\Delta(B\!-\!C)/A=0$ and grow with darkening fraction $f$.
- For Maxwell $H$=0, instability sets in at critical darkening **$f_{\rm crit}\approx0.12$**.

---

## 8. Shell thickness through a darkening cycle *(no fossil bulge)*

![Shell thickness at progressive darkening, Maxwell H=0](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/polar_wander/fig4_mollweide.png)

**Quasi-static, shell-only field (NO fossil bulge)** — the bare destabilizing trend. Maxwell $H$=0, trailing hemisphere darkens ($f$: 0 → 1):

- Mean thickness drops 20.3 km (bright) → 19.3 km (dark).
- $\Delta(B\!-\!C)/A$: $-1.3\times10^{-8}$ (stable) → $+6.2\times10^{-8}$ (unstable).
- The trailing hemisphere visibly thins while the tidal equatorial band persists.

---

## 9. Cyclical & continuous polar wander

![Polar-wander cycle time series, Maxwell H=0, with fossil bulge](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/polar_wander/fig2_time_series.png)

Full 3×3 inertia-tensor model **with the fossil bulge as stabilizer** ($\alpha=1.5$):

- **Cycle period ~1.5 Myr** (Maxwell $H$=0, $\tau_{\rm shell}=10^7$ yr); bottleneck is shell thermal relaxation.
- Period **scales linearly with $\tau_{\rm shell}$** (~0.2–15 Myr over $10^6$–$10^8$ yr); insensitive to $\tau_{\rm dark}$.
- Discrete 90° flips become **continuous rotation** along the trailing–leading meridian once the fossil decays.

---

## 10. Physical timescales & interior coupling

![OSPW vs. full-tensor trajectories](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/docs/figures/ice_shell_dynamics/comparison/compare_trajectories.png)

- Replace the free TPW timescale with the **O&S overdamped (OSPW)** model (viscous shell dissipation + iceberg/isostatic corrections): $\tau_f\sim0.64$ Myr; the free $\tau_{\rm TPW}=10^5$ yr was **~100× too fast**.
- **Burnett & Hayne (2023)** gravity-gradient shell–interior coupling: a near-**hydrostatic interior delays TPW onset ~5 → ~16 Myr**; a rigid interior → free rotation.
- **Interior triaxiality sets the recurrence interval** — constrainable via libration / degree-2 gravity.

*Figure: OSPW overdamped trajectories (top) vs. our continuous full-tensor rotation (bottom).*

---

## 11. Key takeaways

1. **3-D thermal structure matters.** Spatially variable albedo adds the largest *non-tidal*, degree-1 thickness signal — a leading/trailing asymmetry $b_{11}^s\sim1$–1.7 km — that 1-D models miss entirely.
2. **Stability splits by moment.** $\Delta(B\!-\!A)/C$ is tidally robust; $\Delta(B\!-\!C)/A$ is albedo-controlled and can be **destabilizing** (flips Maxwell $H$=0 stable → unstable).
3. **Recurrent polar wander.** Trailing-hemisphere darkening plausibly drives TPW on **~1–2 Myr** (shell-thermal) timescales; interior coupling can **delay onset to ~16 Myr**.
4. **Robust to smoothing.** All geophysical quantities converge by spherical-harmonic degree $\ell\approx4$ (see Backup) — insensitive to the unknown lateral-diffusion scale.

---

## 12. Caveats & references

- Maxwell-rheology thicknesses run 14–24% below the original paper; the $4t_{22}$ spatial amplitude is ~2× the paper's — both unresolved by $C_R$ tuning (structural).
- The fossil-bulge factor ($\alpha=1.5$) and the instability threshold are partly **ad hoc**.
- The strong/weak interior-coupling outcome hinges on Europa's poorly constrained interior triaxiality $\varepsilon_i$.

**References**
- Ojakangas, G.W. & Stevenson, D.J. (1989a). Thermal state of an ice shell on Europa. *Icarus* 81, 220–241.
- Ojakangas, G.W. & Stevenson, D.J. (1989b). Polar wander of an ice shell on Europa. *Icarus* 81, 242–270.
- Ross, M.N. & Schubert, G. (1987). Tidal heating in an internal ocean model of Europa. *Nature* 325, 133–134.
- Paranicas, C. et al. (2001). Electron bombardment of Europa. *Geophys. Res. Lett.* 28, 673–676.
- Burnett, E.R. & Hayne, P.O. (2021). Europa's hemispheric color dichotomy as a constraint on non-synchronous rotation. *Icarus* 364, 114438. doi:[10.1016/j.icarus.2021.114438](https://doi.org/10.1016/j.icarus.2021.114438).
- Burnett, E.R. & Hayne, P.O. (2023). Coupled spin dynamics of Europa's ice shell and interior. *Planet. Sci. J.*

---

# Backup — Supporting Detail

## B1. Robustness to thermal smoothing

![Geophysical quantities vs. spherical-harmonic truncation degree](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/smoothed/fig7_sh_convergence.png)

Truncate the surface-temperature spherical-harmonic expansion at degree $\ell_{\max}$ (proxy for lateral diffusion):
- All geophysical quantities converge by $\ell_{\max}\approx4$ (mostly <1% of full resolution).
- Albedo signal is degree-1, tidal signal degree-2 — both captured at low degree.
- Results insensitive to the poorly-known lateral-diffusion timescale.

---

## B2. Cycle-period scaling

![Cycle period vs. shell and darkening timescales](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/plots/ice-shell-dynamics/polar_wander/fig3_cycle_period.png)

- Cycle period scales linearly with the shell thermal timescale $\tau_{\rm shell}$: ~0.2 Myr ($10^6$ yr) → ~15 Myr ($10^8$ yr).
- Insensitive to the darkening timescale $\tau_{\rm dark}$ (over two decades).
- Nominal $\tau_{\rm shell}=10^7$ yr → recurrence ~1–2 Myr.

---

## B3. TPW timescale comparison

![TPW timescale comparison](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/docs/figures/ice_shell_dynamics/comparison/compare_timescales.png)

OSPW overdamped timescale $\tau_f$ (viscous shell dissipation) vs. the free exponential $\tau_{\rm TPW}=10^5$ yr:
- $\tau_f\sim0.64$ Myr at $T_f$=130 K — the free $\tau_{\rm TPW}$ is ~100× too fast and unconstrained.
- The OSPW timescale derives from the shell's viscous structure (Arrhenius viscosity profile).

---

## B4. Model-framework comparison: ours vs. OSPW vs. B&H

![TPW model comparison](/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm/docs/figures/ice_shell_dynamics/comparison/compare_summary.png)

Reorientation type, timescale, driving mechanism, shell dissipation, interior coupling, geometry, fossil bulge, and instability criterion across the three frameworks.

---

## B5. Interior coupling regimes

Adding Burnett & Hayne (2023) gravity-gradient shell–interior coupling:

| Mode | Onset | Rotation character | Interior |
|------|:-----:|--------------------|----------|
| Exponential | ~5 Myr | continuous, ~450 deg/Myr | N/A |
| OSPW (overdamped) | ~5 Myr | continuous, ~2500 deg/Myr | N/A |
| Coupled — **rigid** ($\varepsilon_i\sim10^{-5}$) | ~5 Myr | continuous, ~2700 deg/Myr | fixed |
| Coupled — **hydrostatic** ($\varepsilon_i\sim5\times10^{-4}$) | **~16 Myr** | episodic → continuous | dragged slowly |

- A near-hydrostatic interior delays TPW onset from ~5 to ~16 Myr; a rigid interior → free rotation.
- Interior triaxiality $\varepsilon_i$ sets the TPW recurrence interval (constrainable by libration / degree-2 gravity).

---

*Full numerical tables and model derivations: `docs/ice-shell-results.md`.*
