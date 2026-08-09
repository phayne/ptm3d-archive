# PTM3D vs Diviner seasonal polar temperatures

Comparison of PTM3D against the Diviner **level-4 seasonal polar temperature
product** (Williams et al. 2019, *JGR Planets* 124, 2505; data:
`https://luna1.diviner.ucla.edu/~jpierre/diviner/level4_polar/`).

## Diviner product
- Bolometric brightness temperature, 240 m/pix, poleward of 80° S, compiled from
  nadir RDR 2009-07-05 → 2019-02-17.
- Split into **summer** and **winter** seasons, defined by the subsolar latitude
  being **below** (summer, S pole more illuminated) or **above** (winter) the
  equator — the ±1.54° obliquity.
- `additional_maps/polar_south_80_{summer,winter}_{max,avg,min}.txt`: sparse ASCII
  `x, y, clon, clat, tbol`. We use **max** and **avg** (= mean) for both seasons.
  Registration is by the `clon/clat` columns (no projection ambiguity, unlike the
  earlier `.mat` product). Files pre-filtered to lat < −86.5° → `*_polar.npy`.

## PTM3D side
- Year-long radiation-coupled runs are split by season: subsolar latitude per
  output timestep is computed via SPICE (`data/kernels/moon.tm`, `subslr` in
  IAU_MOON); summer = sublat<0 (188 steps), winter = sublat≥0 (177 steps). The
  ±1.54° subsolar cycle recurs yearly, so a single simulated year matches the
  Diviner multi-year climatology.
- **All four sites use deep-column-equilibrated runs** (≥25 equilibration orbits +
  ⟨indirect⟩-in-equilibration + coarse eq cadence + 200 km occluder, H-matrix VF).
  The per-site run directories (see `RUN_OVERRIDE` in `diviner_seasonal_compare.py`)
  are `divval_haworth_r32c`, `divval_faustini_fix`, `divval_shackleton_r32`, and
  `divval_sunlit_fix` (Haworth SE rim). The radiation-core extent differs per site
  to cover each crater (Haworth 32 km clipped; Faustini/SE-rim 25 km; Shackleton
  32 km); the extent moves the PSR-floor max by ~5 K. The 2-orbit production runs
  (`output/divval_{site}_cr`), whose **winter warm bias** is the under-equilibration
  artifact, are kept only as the reference "OLD" table below.
- Per grid cell (250 m, south-polar stereographic R=1737.4 km), the footprint
  bolometric `Tbol = ε^¼ (Σ A_i T_i⁴ / Σ A_i)^¼` is formed each timestep; the
  **seasonal max** = max over the season's timesteps, **seasonal mean** = mean.

## Results (bias = model − Diviner, K; see `summary.json`, `seasonal_summary.png`)

**CORRECTED (all sites: eq=25 + eq_indirect + 200 km occluder, H-matrix VF).**
Values below are taken verbatim from `summary.json` / `seasonal_summary.png`:

| Site | summer max | summer mean | winter max | winter mean |
|------|-----------:|------------:|-----------:|------------:|
| Haworth (r32c)       | −18.3 / 19 | −10.5 / 11 | −8.1 / 9  | **−0.3 / 1** |
| Faustini (r25)       | −17.2 / 32 | −12.9 / 13 | −10.4 / 26 | −5.1 / 6 |
| Shackleton (r32)     | −8.7 / 32  | −8.8 / 19  | −1.2 / 36 | −1.4 / 16 |
| Haworth SE rim (r25) | −16.5 / 37 | −15.2 / 18 | −2.9 / 44 | −4.7 / 7 |

**Production (OLD: 2-orbit, 40 km occluder) — for reference, the biases were warm:**

| Site | summer max | winter max | winter mean |
|------|-----------:|-----------:|------------:|
| Haworth        | +1.6  | +22.0 | +29.2 |
| Faustini       | −5.3  | +10.6 | +19.6 |
| Shackleton     | −11.4 | −8.0  | +7.2  |
| Haworth SE rim | +21.7 | **+70.1** | +34.3 |

(bias / RMSE, K)

## Key findings
1. **The winter warm bias was an under-equilibration artifact, now resolved.** The
   2-orbit production runs leave a *warm transient deep column*: the regolith
   reservoir has a multi-year time constant (τ_b ∼ (b²/π)·P_diur ≈ 70 lunations at
   b=15), so 2 orbits (≈2 months) is far from steady state. Equilibrating Haworth
   to the converged deep column (eq=25/49/75 are identical to <0.1 K) drops
   **winter-mean from +29.2 to +0.6 K (RMSE 1.6), winter-max from +22.0 to −6.6,
   summer-mean from +11.5 to −9.7**. Both the walls' off-peak temperatures (Tmin
   59→41 K) and the floor's own reservoir cool; the walls' *peak* (237 K) is
   unchanged, being set by direct sun. The recommended configuration is **≥25
   equilibration orbits, ⟨indirect⟩-in-equilibration, and a coarse equilibration
   cadence**.
2. **The production summer-max match was coincidental, and the residual summer-max
   cold bias is now understood.** The 2-orbit run's good summer-max (+1.6) was
   propped up by the same warm reservoir; with a correctly cold deep column the
   summer-max falls to ~−18 K, exposing a *separate* deficit — the peak wall→floor
   indirect flux is ~half what is needed. This was run to ground in June 2026
   (`docs/psr-floor-temperature-validation.md`): for Haworth's measured d/D = 0.051
   the Ingersoll (1992) bowl-crater closed form gives T_max = 64.9 K = Diviner, and
   PTM3D delivers **exactly half** the required floor flux (0.49 vs 1.0 W/m²). The
   factor of 2 is a geometric identity — at the e ≈ 4° grazing Sun only ~half the
   rim ring is sunlit at any instant, whereas Ingersoll/Diviner assume the *whole*
   ring is warm; closing it needs **rim heat retention (higher wall thermal
   inertia)**, not a radiative-transfer, view-factor, or illumination fix (all
   validated correct there). The Smith roughness-beaming term is real but ~30× too
   weak to cover it.
3. **All four sites are now deep-column-equilibrated** (eq ≥ 25), so the winter
   warm bias is gone everywhere (winter-mean bias |≤ 5 K|; Haworth −0.3 K). The
   remaining cross-site RMSE (Faustini/Shackleton/SE-rim summer-max 32–37 K) is
   driven by the summer-max floor deficit of finding 2 plus near-pole registration
   on the bimodal sunlit-rim sites — keep validation sites ≥ ~2° off the pole.
4. **Product dependence:** the same max-bolometric quantity differs by ~10 K
   between the Landis `.mat` product and Williams 2019 — the Diviner
   product/processing choice matters at the ~10 K level for PSR maxima.

## Files
- `seasonal_{site}.png/pdf` — per-site 4×3 (rows: summer/winter × max/mean;
  cols: Diviner | PTM3D | model−Diviner), per-panel robust color scale.
- `seasonal_summary.png/pdf`, `summary.json` — cross-site bias/RMSE.
- Code: `src/python/diviner_seasonal_compare.py` (+ `diviner_compare.py` helpers,
  spiceypy season split). Data: `data/diviner_seasonal/`.
