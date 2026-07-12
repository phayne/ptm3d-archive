# bowl_crater_ingersoll_twoband.png

Validation of the PTM3D radiation module with the **NEW two-band (visible +
thermal-IR) radiosity** against the Ingersoll, Svitek & Murray (1992) spherical
bowl-crater model. **Supersedes the gray-radiosity figure**
`bowl_crater_ingersoll_production.png`: the earlier figure kept its shadow
temperature panel analytic-only because the gray radiosity was albedo-dependently
biased; the two-band solve removes that bias, so the mesh now sits on the analytic
shadow curve and is overlaid.

Terminology: **shadow / PSR** (the whole spherical-cap interior is one surface;
there is no flat "floor"). All numbers are read from tracked harness CSVs by
`src/python/bowl_crater_twoband_figure.py`; nothing is read from `/tmp`.

**Caption.** (a) Shadow-equilibrium temperature of the permanently shadowed
spherical-cap interior versus solar elevation `beta`: two-band mesh (open
markers) over the Ingersoll (1992) Eq 8 analytic shadow curve (solid lines), for
`d/D` in {0.1, 0.2, 0.3}, Moon `A = 0.12`, `eps = 0.95`. Only fully-shadowed
points are shown (harness guard `floor_lit_max == 0`, i.e. `beta` below the
interior-illumination limit ~0.82-0.85 `beta_crit`). The mesh reproduces the
analytic shadow to `|dT| < 0.27 %` across the shadowed range. (b) **The two-band
fix.** Shadow-flux error versus Eq 8 as a function of Bond albedo `A`, at
`d/D = 0.2`, `beta = 0.6 beta_crit`: the **gray** single-band radiosity carries an
albedo-dependent bias that grows from `-1.0 %` to `-19.3 %` across
`A = 0.06..0.30`; a naive **source-only patch** (`PTM_SOLAR_ALBEDO`, fixing only
the first-bounce source, not the over-absorption) overshoots from `-0.0 %` to
`+7.6 %`; the **two-band** radiosity is a flat, albedo-independent `~-0.056 %`
residual (pure view-factor faceting). (c) Shadow sky view factor `F_sky` from the
production view-factor calculator at ~33k faces, as percent deviation from the
analytic spherical-cap value `sin^2 alpha = 1 - f`; markers = mesh area-weighted
mean, whiskers = per-face min/max; the analytic value is the zero line. `F_sky` is
purely geometric and is unchanged by the two-band solve. (d) `F_sky` convergence
for `d/D = 0.2`: per-face peak deviation, per-face std, and mean `|bias|` versus
facet count `N`, log-log, with `N^(-1/2)` and `N^(-1)` guide slopes — the error
converges (it does not plateau); the spread `~ N^(-1/2)` is flat-triangle faceting
of the curved cap.

## Numbers shown

Panel (a) — two-band mesh shadow `T` vs Ingersoll Eq 8, Moon `A = 0.12`
(`output/two_band_prod/E1_beta_sweep.csv`, fully-shadowed points only, nf = 6000):

| d/D | beta range (deg) | beta_crit (deg) | # points | max \|dT\| vs Eq 8 |
|-----|------------------|-----------------|----------|--------------------|
| 0.1 | 1.70 .. 9.05  | 11.31 | 6 | 0.143 % |
| 0.2 | 3.27 .. 17.44 | 21.80 | 6 | 0.238 % |
| 0.3 | 4.64 .. 26.01 | 30.96 | 7 | 0.271 % |

(Largest deviations are at the most grazing `beta`, where the shadow flux is
smallest and the relative faceting/anti-aliasing error is largest.)

Panel (b) — shadow-flux error vs Eq 8 (%), `d/D = 0.2`, `beta = 0.6 beta_crit`:

| A | gray (single band) | source-only patch | two-band |
|-----|--------|--------|----------|
| 0.06 | -1.0012 | -0.0012 | -0.05612 |
| 0.12 | -6.3261 | +0.7245 | -0.05629 |
| 0.20 | -12.6009 | +2.8222 | -0.05659 |
| 0.30 | -19.3136 | +7.5810 | -0.05741 |

Two-band spread over the 4x albedo range: **+1.28e-03 %** (flat / albedo-
independent). Source: two-band = `output/two_band_prod/E1_bowl_eq8.csv`
(Jacobi; Gauss-Seidel identical to ~1e-6); gray + source-only patch =
`output/bowl_conv/radiosity_eq8_twoband.csv`.

Panel (c) — production `F_sky`, mesh mean and per-face peak deviation, nf = 32832
(`output/bowl_conv/results.csv`):

| d/D | analytic sin^2 alpha | mesh mean (area-wt) | mean dev | per-face min/max | peak dev |
|-----|----------------------|---------------------|----------|------------------|----------|
| 0.1 | 0.961538 | 0.961533 | -0.0006 % | 0.961504 / 0.961562 | 0.0036 % |
| 0.2 | 0.862069 | 0.862060 | -0.0010 % | 0.861962 / 0.862147 | 0.0124 % |
| 0.3 | 0.735294 | 0.735269 | -0.0034 % | 0.735127 / 0.735404 | 0.0227 % |

Panel (d) — convergence (`d/D = 0.2`): peak dev 0.0918 -> 0.0108 %, per-face std
0.0351 -> 0.0058 %, mean `|bias|` 0.0459 -> 0.0008 % for N = 1400 -> 40446.
Spread `~ N^(-1/2)`, mean bias `~ N^(-1)`.

## Provenance

- HEAD commit `f5c455e` (branch `feat/two-band-radiosity`); the committed
  two-band code.
- `build/bin/bowl_crater_ptm` md5 `31bf19e57b8f365b0305949fafecef31`;
  `libradiation.a` md5 `4e07d0e65dd2dffb033eee07e2cc760b` (static lib; no dylib).
  `test_bowl_crater` (radiation gate) md5 `0aa7023271981be2f62dc87eefb5b64e`, PASS.
- The working-tree superset over `f5c455e` (an aniso-beaming comment-only hunk in
  `radiation_exchange.cpp` + three default-OFF `radiation_config.h` fields) does
  not touch the radiosity solar-scatter path exercised here; buried-ice and
  `ptm_thermal_sim` are not linked into `bowl_crater_ptm`.
- Panel (b) two-band values are byte-identical to the pre-commit Phase-3 CSV, so
  the committed binary reproduces the earlier result exactly.

## Reproduce

    PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python3
    # data (output/ is gitignored; regenerate if empty):
    bash output/two_band_prod/run_e1_beta_sweep.sh   # panel (a): two-band T vs beta
    bash output/two_band_prod/run_e1_bowl_eq8.sh      # panel (b): two-band A-sweep
    bash output/bowl_conv/run_stage1.sh               # panel (c): F_sky at 33k, 3 d/D
    bash output/bowl_conv/run_stage2.sh               # panel (d): convergence, d/D=0.2
    # gray + source-only patch for panel (b) live in
    #   output/bowl_conv/radiosity_eq8_twoband.csv (from output/bowl_conv/run_twoband.sh)
    # figure:
    $PY src/python/bowl_crater_twoband_figure.py
