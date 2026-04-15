# Paper Plan: Europa Surface Temperatures and Volatile Stability with PTM3D

*Created: 2026-02-28*

## Context

Europa is a prime target for astrobiology and planetary science, with the Europa Clipper mission now en route. The E-THEMIS instrument paper (Christensen et al. 2024) explicitly notes that **no Europa thermal model exists with 3D topographic capability** needed to remove shadowing and scattering effects from thermal observations. PTM3D fills this gap as the first model combining 3D mesh-resolved thermophysics (illumination, conduction, radiation exchange) with an exosphere model (sublimation, ballistic transport, sputtering, ice layer evolution) for Europa. This paper validates PTM3D against existing observations (Galileo PPR, ALMA), presents global and regional thermal results, characterizes volatile stability and ice migration, and makes testable predictions for Europa Clipper. Crystallization/amorphization modeling and comparison with JWST crystallinity observations are deferred to a companion paper.

**Target journal**: JGR: Planets
**Figures**: 13-16
**Model description**: Appendices in this paper

---

## 1. Paper Outline and Sections

### Title (working)
"Surface Temperatures, Volatile Stability, and Ice Migration on Europa: A 3D Thermophysical and Exosphere Model"

### Abstract
- PTM3D: first 3D thermophysical + exosphere model for EuropaCygnusx1buc*

- Validated against Galileo PPR (86-132 K), ALMA subsurface
- Key results: global T maps, eclipse effects, thermal segregation, ice migration patterns
- PSR volatile stability (H2O, CO2, SO2) at poles
- Predictions for E-THEMIS

### 1. Introduction (~2 pages)
- Europa as a high-priority target; Europa Clipper mission context
- Surface temperature controls volatile stability and surface evolution
- Existing thermal observations: Galileo PPR (Spencer et al. 1999), nighttime anomalies, thermal inertia 30-140 SI
- ALMA depth-resolved thermophysics (Thelen et al. 2024): porosity gradients, hemispheric asymmetry
- JWST: crystalline ice distribution (Trumbo et al. 2025), endogenous CO2 (Villanueva et al. 2023)
- Gap: no 3D model exists (Christensen et al. 2024 E-THEMIS paper states this explicitly)
- This work: PTM3D model, validation, global/regional results, exosphere, PSR stability, E-THEMIS predictions
- Paper organization paragraph

### 2. Model Overview (~3-4 pages, brief; details in appendices)
- 2.1 Mesh and geometry: icosahedral subdivision, Europa shape model, regional topographic meshes
- 2.2 Illumination: SPICE ephemeris, Embree ray tracing, finite solar disk, eclipses, Jupiter IR + reflected light
- 2.3 Thermal conduction: 1D Crank-Nicolson, T-dependent properties (k, cp, rho), Europa ice regolith parameters
- 2.4 Radiation exchange: view factors, radiosity solver (for topographic models)
- 2.5 Exosphere model (brief): sublimation (Marti & Mauersberger 1993), ballistic transport, sputtering (Cassidy et al. 2013), ice layer evolution, bidirectional albedo-thermal coupling
- 2.6 Europa-specific parameters: Table of adopted thermal/physical properties

### 3. Model Validation (~3 pages)
- 3.1 Comparison with Galileo PPR: diurnal temperature curves at low latitudes (86-132 K range)
- 3.2 Comparison with ALMA: subsurface temperature profiles, leading/trailing asymmetry
- 3.3 Comparison with 1D models: show where 3D effects matter (topographic shadowing, radiation exchange)

### 4. Global Thermal Results (~4-5 pages)
- 4.1 Global temperature maps: max, min, mean surface temperatures
- 4.2 Diurnal temperature curves: equatorial, mid-latitude, polar; leading vs. trailing
- 4.3 Eclipse effects: temperature drops during Jupiter shadow transits, spatial pattern, recovery timescales
- 4.4 Albedo effects: impact of spatially varying Bond albedo on temperature distribution
- 4.5 Leading/trailing hemisphere asymmetries: temperature contrasts driven by albedo and thermal inertia differences

### 5. Regional Topographic Studies (~3 pages)
- 5.1 Pwyll crater: temperature distribution, shadowing effects, comparison with PPR anomaly
- 5.2 Ridge features: temperature modulation by topography, radiation exchange between ridge flanks
- 5.3 Other regions (Cilix, Agenor, Rhadamanthys): topographic thermal signatures
- 5.4 E-THEMIS predictions: what E-THEMIS will observe at these features at different resolutions

### 6. Exosphere and Volatile Transport (~4-5 pages)
- 6.1 Sublimation patterns: latitude and longitude dependence, diurnal variation
- 6.2 Sputtering vs. sublimation: where each dominates; transition at non-ice layer thickness threshold
- 6.3 Ice migration: net polar accumulation, equatorial depletion, characteristic transport distances
- 6.4 Thermal segregation: self-consistent albedo-composition feedback (Spencer 1987 mechanism), timescales (<1000 yr)
- 6.5 Lag deposit formation: non-ice concentration evolution, dust armor effects

### 7. Permanently Shadowed Regions and Volatile Stability (~3 pages)
- 7.1 Identification of PSRs: polar topography, shadow mapping
- 7.2 PSR temperatures: minimum achievable temperatures, dependence on latitude and topography
- 7.3 H2O stability: cold-trapping efficiency, accumulation rates
- 7.4 CO2 stability: sublimation temperature (~81 K), stability zones relative to PSR temperatures
- 7.5 SO2 and other volatiles: stability thresholds, implications for surface composition
- 7.6 Implications for Europa Clipper observations of polar regions

### 8. Discussion (~2-3 pages)
- 8.1 Synthesis: how thermal patterns drive surface evolution
- 8.2 Implications for surface composition and age: ice migration timescales vs. geological resurfacing
- 8.3 Connection to JWST observations: CO2 stability in Tara Regio context
- 8.4 Predictions for Europa Clipper: specific testable predictions for E-THEMIS, MISE, and other instruments
- 8.5 Model limitations and future work: depth-dependent thermal inertia, porosity evolution, thermal conductivity feedback, coupling with geologic models

### 9. Conclusions (~1 page)
- Bullet-point summary of key findings
- Emphasis on PTM3D as enabling tool for Europa Clipper science

### Appendices (Model Details)
See Section 2 below.

### Data Availability Statement
- Model outputs archived at [repository]
- PTM3D code availability statement

---

## 2. Appendix Structure (Model Details)

The appendices provide the full technical description of PTM3D so the main text can focus on results. Divide into four logical appendices:

### Appendix A: Illumination Model
- SPICE kernel setup for Jupiter system
- Ray-traced shadow calculation with Embree
- Finite solar disk sampling (not point source)
- Eclipse geometry: Jupiter shadow cylinder/cone intersection
- Jupiter reflected light and thermal IR contributions
- Albedo maps: Bond albedo from FITS data, incidence-angle dependence
- Equations: direct solar flux, reflected/IR flux from Jupiter

### Appendix B: Thermal Conduction Model
- 1D heat equation with depth-dependent properties
- Crank-Nicolson discretization, Thomas algorithm
- Grid structure: m layers in upper skin depth, growth factor, total depth
- Temperature-dependent thermal conductivity: k(T, z) with depth profile
- Heat capacity: cp(T) polynomial (water ice)
- Radiative conductivity term
- Surface boundary condition: radiative equilibrium with optional internal heat flux
- Spectral equilibration for spin-up (FFT-based, Appendix B.1)
- Validation: comparison with lunar Diviner data or analytic solutions

### Appendix C: Radiation Exchange Model
- View factor computation: ray-traced, sparse storage
- H-matrix compression for large meshes
- Radiosity equation: B = eps*sigma*T^4 + rho*F*B + rho*Q_solar
- Jacobi iteration solver
- Thermal-radiation coupling: adaptive iteration with convergence criteria

### Appendix D: Exosphere Model
- D.1 Introduction and overview
- D.2 Sublimation: Marti & Mauersberger vapor pressure, Hertz-Knudsen equation
- D.3 Ballistic molecular transport: collisionless exosphere, modified Bessel function kernel, hop distances
- D.4 Jeans escape: Chamberlain approximation (negligible for H2O)
- D.5 Sputtering: Cassidy et al. rates, spatial distribution (trailing enhancement, polar enhancement), temperature-dependent yield
- D.6 Ice layer model: non-ice concentration tracking, lag deposit formation, albedo-composition coupling (thermal segregation feedback)
- D.7 Multi-timescale method: operator splitting for century-scale evolution, with albedo update step
- D.8 Model parameters

---

## 3. Key Figures (13-16 total)

### Validation (Figures 1-2)
1. **Diurnal temperature curves vs. Galileo PPR** — Model vs. observed temperatures at specific locations (equatorial, Pwyll region). Demonstrates model fidelity.
2. **Global temperature map vs. ALMA** — Side-by-side comparison of model brightness temperatures with ALMA observations at 0.88/1.25/3.05 mm wavelengths.

### Global Thermal Results (Figures 3-6)
3. **Global max/min/mean temperature maps** — Orthographic or Mollweide projections showing spatial distribution of T_max, T_min, T_mean over one orbit. Core result figure.
4. **Diurnal temperature curves by latitude** — Family of curves showing T(t) at equator, 30, 60, 80 deg latitude for leading and trailing hemispheres.
5. **Eclipse temperature effects** — Map or time series showing temperature drop during eclipse passage through Jupiter's shadow. Include spatial pattern of eclipse cooling.
6. **Albedo-temperature correlation** — Scatter plot or map showing how spatially varying albedo drives temperature contrasts; quantify thermal segregation potential.

### Regional Studies (Figures 7-9)
7. **Pwyll crater temperature map** — High-resolution temperature distribution showing topographic shadowing, radiation exchange between crater walls. Compare with PPR anomaly.
8. **Ridge thermal profiles** — Cross-section temperature profiles across ridge features showing how ridge geometry modulates temperature.
9. **E-THEMIS predictions composite** — Simulated E-THEMIS observations at different resolutions (8 km, 1 km, 100 m) for key regional targets, showing what topographic thermal signatures will be detectable.

### Exosphere and Volatile Transport (Figures 10-13)
10. **Sublimation flux map** — Global map of net sublimation/deposition rate, showing equatorial loss and polar accumulation.
11. **Sputtering vs. sublimation regime map** — Map showing where each process dominates, with transition controlled by non-ice layer thickness/albedo.
12. **Ice migration and thermal segregation** — Time evolution showing progressive ice removal from dark/warm regions and accumulation in bright/cold regions. Show runaway albedo feedback.
13. **Lag deposit / non-ice concentration evolution** — Time evolution of non-ice concentration showing lag deposit formation and self-limiting armor effect.

### PSR and Volatile Stability (Figures 14-16)
14. **Polar PSR identification** — Map of permanently shadowed regions near Europa's poles with temperature color scale.
15. **PSR temperature histogram** — Distribution of minimum temperatures in PSRs; overlay stability thresholds for H2O, CO2, SO2.
16. **Volatile stability zones** — Polar projection maps showing stability regions for each volatile species, with implications for cold-trapping.

### Optional figure
17. **Model schematic** — Diagram showing PTM3D workflow (mesh → illumination → thermal → radiation → exosphere), could go in Section 2 or as a graphical abstract.

---

## 4. Important Results to Emphasize

### Primary Results (abstract-worthy)
1. **First 3D thermophysical model for Europa** — Fills the gap identified by E-THEMIS team; enables topographic correction of thermal observations
2. **Validated against two independent datasets** — Galileo PPR, ALMA; demonstrates model reliability across wavelengths and depths
3. **Rapid thermal segregation timescale (<1000 yr)** — Ice removal and migration occurs much faster than geological resurfacing timescales (~50-100 Myr), implying surface composition is in dynamic equilibrium with current thermal environment
4. **Sublimation dominates sputtering when ice is exposed** (leading hemisphere), but sputtering takes over once lag deposit forms — this regime transition is a novel quantitative result
5. **Net polar ice accumulation** — Equator-to-pole transport; polar regions are net sinks for water ice
6. **PSR volatile stability** — First quantitative assessment of what volatiles (H2O, CO2, SO2) are stable in Europa's polar PSRs

### Secondary Results (discussion-worthy)
7. **Eclipse cooling signatures** — Quantify temperature drop and recovery; testable by E-THEMIS if observations span eclipse events
8. **Topographic thermal signatures at specific features** — Pwyll, ridges, chaos terrain; direct predictions for E-THEMIS at each resolution tier
9. **Lag deposit armor effect** — Non-ice concentration buildup eventually quenches sublimation, creating self-limiting evolution

### Framing for Impact
- **Europa Clipper relevance**: Every result section should connect to what Europa Clipper will observe. E-THEMIS predictions are a key selling point.
- **JWST synergy**: CO2 stability results connect directly to ongoing JWST observations; crystallinity comparison deferred to companion paper.
- **Thermal segregation mechanism**: Spencer (1987) proposed this for Galilean satellites; PTM3D is the first model to self-consistently simulate the full feedback loop (albedo → temperature → sublimation → composition → albedo) on Europa, with bidirectional coupling between the thermal and exosphere modules.
- **No endogenic anomalies detected to date**: The model provides the baseline "passive" thermal field that must be subtracted to detect endogenic heat flow — frame PTM3D as an essential tool for this.

---

## 5. Remaining Work Before Writing

### Simulations to Run
- [ ] Permanently shadowed region models using real Europa polar topography
- [ ] Volatile stability calculations for CO2 and SO2 in PSRs
- [ ] Verify ice transport timescale (<1000 yr claim)
- [ ] Quantify sublimation vs. sputtering regime transition (non-ice layer thickness threshold)

### Analysis to Complete
- [ ] Direct comparison plots with Galileo PPR data points
- [ ] Simulated ALMA brightness temperatures from model output
- [ ] Simulated E-THEMIS observations at 8 km, 1 km, 100 m resolution

*(Crystallinity map comparison with JWST Fresnel peak locations deferred to companion paper)*

### Figures to Generate
- [ ] All 13-16 figures listed above
- [ ] Determine best projection (Mollweide, orthographic, polar) for each map figure

---

## 6. Verification Plan

Before submission:
1. Cross-check all model-observation comparisons for consistency
2. Verify mass conservation in exosphere model (sublimated + sputtered ≈ deposited + escaped)
3. Confirm PSR temperatures are physically reasonable (compare with lunar PSR temperatures scaled for Europa's environment)
4. Have co-authors review E-THEMIS prediction figures for instrument capability realism
5. Check all parameter values against published references

---

## Key References

- Christensen et al. (2024) — E-THEMIS instrument paper, Space Science Reviews
- Spencer et al. (1999) — Galileo PPR Europa temperatures, Science
- Spencer (1987) — Thermal segregation on Galilean satellites, Icarus
- Thelen et al. (2024) — ALMA subsurface thermophysics, PSJ
- Trumbo et al. (2025) — JWST crystalline ice on Europa
- Villanueva et al. (2023) — JWST CO2 on Europa, Science
- Cassidy et al. (2013) — Magnetospheric sputtering rates
- Marti & Mauersberger (1993) — H2O vapor pressure at low temperatures
- Rathbun et al. (2010) — Galileo PPR hotspot detection limits, Icarus
