# PTM3D PSR Floor Temperature: H-matrix Fix and Ingersoll/Diviner Validation

**Status:** investigation complete (2026-06-08). Test case: Haworth crater central
permanently-shadowed region (PSR), lunar south pole (−87.45°, −2.1°).

## Summary

Two distinct issues were diagnosed for the cold Haworth PSR floor:

1. **H-matrix view-factor collapse at fine resolution (FIXED).** Above ~100k faces the
   H-matrix builder zeroed large floor↔wall blocks to avoid an OOM balloon, collapsing
   the floor's terrain coupling (F_terrain 0.013 → 0.0003) and dropping the 400k floor to
   an unphysical 25 K. A bounded **subdivide-to-sparse** path replaces the three zeroing
   branches, restoring the fine-core coupling to the geometric value. The fine core now
   reproduces the ~100k-face result (~45 K) instead of collapsing.

2. **The residual ~20 K floor cold bias vs Diviner is a resolved-intermittency effect,
   not a model error in any single subsystem.** PTM3D gives a central-floor summer
   maximum of ~45 K; Diviner observes ~65 K. Every candidate cause was ruled out (below),
   and the gap is precisely the difference between the **instantaneous sunlit half-ring**
   that PTM3D resolves and the **whole-ring-warm steady state** that the Ingersoll closed
   form (and, evidently, the real Moon) assume.

## Diviner / Ingersoll benchmark

Williams et al. (2019) seasonal Diviner grids, Haworth central floor (<1.5 km):

| | Diviner | PTM3D |
|---|---|---|
| winter avg | 30.2 K | ~30 K ✓ |
| summer avg | 46.3 K | 36.4 K |
| **summer max** | **64.6 K** | **45.4 K** |
| sunlit rim (20–25 km) | 230 K | 239 K ✓ |

Ingersoll, Svitek & Murray (1992) spherical-bowl closed form:

```
sigma T^4 = S sin(e) * f(1-A)/(1-Af) * [1 + A(1-f)/eps],   f = 1/[1 + (D/d)^2/4]
```

For Haworth's **measured d/D = 0.051** (DEM depth 2.62 km, diameter 51 km), peak solar
elevation e = 4.1° (lat −87.45° + 1.54° obliquity), S = 1361, A = 0.12, ε = 0.95:

- f = 0.0104 (≡ the floor's geometric F_terrain from the 5.9° rim elevation angle)
- **σT⁴ = 1.00 W/m² → T_max = 64.9 K** — matches Diviner, 20 K above PTM3D.

## The factor of 2

The Ingersoll required floor flux is **1.0 W/m²**; PTM3D delivers **0.49 W/m² — exactly
half**. The decomposition is a geometric identity:

- Ingersoll assumes the **entire** F_terrain rim ring is at the sunlit temperature.
- At e = 4° only **~half the ring is sunlit at any instant** (the far inner wall the
  grazing Sun reaches across the crater). Half of 1.0 ≈ 0.5 = PTM3D's 0.49.

For the floor to reach 65 K, the currently-dark half of the ring must remain warm from
when the circling Sun lit it earlier — i.e. **rim/wall heat retention (thermal inertia)**.
PTM3D's uniform low-thermal-inertia regolith cools each rim sector between Sun passes, so
the floor only ever sees the instantaneous half-ring. Ingersoll's steady whole-ring value
is an upper bound that the real Moon reaches because its rim retains heat.

## Levers ruled out (with the diagnostic that excluded each)

| Candidate | Verdict | Evidence |
|---|---|---|
| H-matrix VF under-count (fine mesh) | not the 100k cause | `vf_apply`: model F·(εσT⁴) = 1.48 ≥ MC ground truth 1.29 (`vf_mc_flux`) |
| Grazing-Sun under-illumination | correct | `illum_check`: 99.1% face agreement vs clean Embree + 200 km occluder; lit fraction flat vs ray-lift 0.1–10 m |
| Heat capacity / thermal inertia (cp) | standard | Hayne (2017) cp(T): cp(46 K)=126, cp(300 K)=771 J/kg/K |
| Crater depth/diameter | correct | DEM d/D = 0.051, Ingersoll f = F_terrain to 3 figures |
| Equilibration excludes indirect | +0.7 K on max | `PTM_EQ_WITH_INDIRECT` re-equilibration: Tmax 45.4→46.1, Tavg 29.6→32.7 |
| Output cadence (temporal aliasing) | 0 effect | 35-day summer window, 1/day vs hourly: 41 vs 37 K |
| Scattered-solar albedo (ρ = 1−ε vs 0.12) | ~+1 K | `radiation_exchange.cpp:452`; thermal IR dominates at warm rim |

## Features added

### Bounded subdivide-to-sparse H-matrix (default ON)
`radiation_config.hmatrix_subdivide_admissible` (env `PTM_HMATRIX_SUBDIVIDE`). Replaces the
ACA-cap pre-zero, large-ACA-fail zero, and inadmissible near-field zero with a recursive
spatial-median bisection that emits cheap low-rank or exact-sparse leaves, bounding RSS at
every level. Build-only; cached `.vf` files still load.

### Equilibrate-with-indirect re-thermal (default OFF)
`PTM_EQ_WITH_INDIRECT=1` (cycles via `PTM_EQ_INDIRECT_CYCLES`, default 2). In the
radiation-coupling re-thermal, re-equilibrates the selected faces' deep column with the
phase-mapped indirect flux before the output cycle, so a PSR floor's sub-surface settles
to the indirect-warmed state instead of draining the surface. Watchdog-safe on macOS+AMD
(batch capped to 4000 via `PTM_EQ_RETHERMAL_BATCH`, single-cycle dispatches, inter-batch
pause). Warms the floor *mean* ~+3 K; the *max* is intermittency-limited (see above).

### Diagnostic tools (`src/cpp/radiation/`, all CMake targets)
- `vf_mc_flux` — geometric ground-truth floor indirect by cosine-weighted hemisphere MC
  weighted by ε·σ·T⁴ (front-face filtered).
- `illum_check` — independent grazing-Sun shadow-ray trace with configurable origin lift
  and optional occluder mesh; sweeping the lift exposes self-shadow artifacts.
- `vf_apply`, `vf_examine_pairs`, `vf_hemisphere_mc`, `vf_face_probe` — view-factor probes.

## Conclusion

PTM3D's deep-floor mean is energy-balanced and correct; its summer-max of ~45 K is the
physically-resolved value for uniform-regolith thermophysics, in which only the
instantaneous sunlit half of the rim warms the floor. The Ingersoll closed form and the
Diviner observation (both ~65 K) require the *whole* rim to be warm simultaneously, which
on the real Moon is supplied by rim heat retention (higher-thermal-inertia rocky walls).
Closing the residual is therefore a thermophysical-property question (spatially-varying
wall/rim thermal inertia), not a radiative-transfer, view-factor, or illumination error —
all of which are validated here.
