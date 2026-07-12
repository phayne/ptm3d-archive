# bowl_crater_ingersoll_fig7.png — Moon3D manuscript Fig. 7 (two-band rebuild)

Composite rebuild of the manuscript's Fig. 7, matching the original layout (3D
crater renders on top, two line panels below) but computed with the corrected
PTM3D **two-band (visible + thermal-IR) radiosity** and Paul's four revisions:
d/D ∈ {0.05, 0.10, 0.20}; two-band (not gray) with the Moon Bond albedo A = 0.12;
render rows expressed as β/β_c; the lower-right panel now shows the shadow-flux
error vs the Ingersoll analytical solution (replacing Q_ind); "Numerical"/
"Analytical" legends and "shadow / PSR" terminology throughout (never "floor").

Terminology: the spherical-cap crater interior is a single curved surface with no
flat floor; all quantities are the **shadow / PSR** (permanently shadowed
interior) values. All numbers are read from tracked harness artifacts under
`output/two_band_prod/fig7/`; nothing is read from `/tmp`.

## Draft caption (manuscript style)

> **Figure 7.** Simulation of radiative-equilibrium temperatures in a bowl-shaped
> (spherical-cap) crater with the 3D thermophysical model, and comparison to the
> analytical solution of Ingersoll, Svitek & Murray (1992). Each crater is a
> spherical cap of depth-to-diameter ratio d/D set in a square flat apron at the
> rim level; radiative equilibrium is solved with the two-band (visible +
> thermal-IR) radiosity for the Moon (Bond albedo A = 0.12, emissivity
> ε = 0.95). **(a)** Tilted 3D renders of the equilibrium temperature (shared
> logarithmic color scale, left) for d/D = 0.05, 0.10, and 0.20 (columns) at two
> solar elevations, β/β_c = 0.30 (upper) and β/β_c = 0.85 (lower), where
> β_c = arctan(2 d/D) is the elevation at which direct sun first reaches the
> crater's lowest point. Because the rows are fixed in β/β_c, the absolute solar
> elevation — and hence the sunlit apron temperature — differs between columns
> (β = 1.7°/4.9°, 3.4°/9.6°, and 6.5°/18.5° for the three d/D). **(b)** Mean
> temperature of the permanently shadowed interior (PSR) versus β/β_c; solid
> lines are the Ingersoll et al. (1992) Eq. 8 analytic shadow solution
> (Analytical), open circles the two-band mesh result (Numerical). Over the
> shadowed range the mesh reproduces the analytic shadow temperature to within
> ≤ 0.1 K (|ΔT| ≤ 0.26%, 0.10%, and 0.07% for d/D = 0.05, 0.10, 0.20). **(c)**
> Fractional error of the numerical shadow flux relative to the analytical
> solution, 100·(numerical − analytical)/analytical, versus β/β_c (linear axis).
> The error is below ~1.2% across the shadowed range (peak 1.04%, 0.40%, and
> 0.30% for d/D = 0.05, 0.10, 0.20, at the coldest, smallest-signal elevations).
> The x-axis is limited to β/β_c ≤ 0.8 to exclude the near-critical edge-lighting
> spike, where the finite crater's sunward interior edge begins to catch direct
> sun before the deepest point does (β/β_c ≈ 0.83, 0.83, 0.84 for the three d/D)
> and the PSR shadow comparison no longer applies.

## Numbers shown (two-band, A = 0.12, ε = 0.95, nf = 15,840)

Data: `output/two_band_prod/fig7/fig7_sweep.csv` + `fig7_dd{0,1,2}_*.npy`.

Render rows: β/β_c = 0.30 and 0.85. Absolute β per column [deg]:

| d/D | β_c | β (row 0.30) | β (row 0.85) |
|-----|------|--------------|--------------|
| 0.05 | 5.71 | 1.71 | 4.85 |
| 0.10 | 11.31 | 3.39 | 9.61 |
| 0.20 | 21.80 | 6.54 | 18.53 |

Panel (b) — shadow (PSR) mean T, Numerical vs Analytical Eq 8, over the shadowed
range (β/β_c ≤ 0.82–0.84):

| d/D | max \|ΔT\| (%) | max \|ΔT\| (K) |
|-----|----------------|----------------|
| 0.05 | 0.258 | 0.10 |
| 0.10 | 0.098 | 0.06 |
| 0.20 | 0.074 | 0.09 |

Panel (c) — shadow-flux error, numerical vs the analytical solution, over the
plotted range β/β_c ≤ 0.8:

| d/D | max \|error\| over β/β_c ≤ 0.8 (%) | at β/β_c |
|-----|-----------------------------------|----------|
| 0.05 | 1.039 | 0.10 |
| 0.10 | 0.395 | 0.10 |
| 0.20 | 0.298 | 0.15 |

The panels are plotted to β/β_c = 0.8, excluding the near-critical edge-lighting
spike (the sunward interior edge first catches direct sun at β/β_c ≈ 0.83, 0.83,
0.84 for the three d/D, off-figure); beyond this the PSR is substantially sunlit
and the shadow comparison no longer applies.

## Provenance

- Two-band data: `bowl_crater_ptm` md5 `5e0e19175e0bf5180c44bfef3766d73b`
  (branch `feat/two-band-radiosity`, HEAD `f5c455e` + the uncommitted
  `PTM_BOWL_FIG7` harness block in `bowl_crater_ptm.cpp`). The production
  radiosity library is UNCHANGED: `libradiation.a` md5
  `4e07d0e65dd2dffb033eee07e2cc760b` == the f5c455e committed two-band lib, so
  the solver driving this figure is the same code that passed Phase E1. The
  harness edit is confined to the standalone `bowl_crater_ptm` main (a
  measurement fixture NOT linked into `ptm_thermal_sim`).
- Radiation gate `test_bowl_crater` md5 `0aa7023271981be2f62dc87eefb5b64e`:
  PASSED (reciprocity max residual 7.0e-14%, GPU-vs-CPU < 1%, F_sky 0.861389 vs
  analytic 0.862069).
- Two-band ON (`PTM_RAD_TWO_BAND=1`, `PTM_SOLAR_ALBEDO=0.12`), Jacobi. For the
  uniform-albedo bowl the uniform env scalar A = 0.12 is identical to a per-face
  Bond albedo (every face A = 0.12), so this is the incidence-consistent case.
- Mesh: 40 rings × 160 az × 10 flat = 15,840 faces; production
  `ViewFactorCalculator` + Embree ray-traced occluded view factors.

## Reproduce

    export DEVELOPER_DIR=/Library/Developer/CommandLineTools
    PTM=/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm
    PY=/Users/paha3326/anaconda3/envs/europa_viz/bin/python3
    # data (output/ is gitignored; regenerate if empty):
    cd "$PTM"
    export PTM_BOWL_NRINGS=40 PTM_BOWL_NAZ=160 PTM_BOWL_NFLAT=10
    export PTM_BOWL_FIG7=1 PTM_BOWL_ALBEDO=0.12 PTM_SOLAR_ALBEDO=0.12 PTM_RAD_TWO_BAND=1
    export PTM_BOWL_FIG7_NB=15 PTM_BOWL_FIG7_FMIN=0.10 PTM_BOWL_FIG7_FTAIL=0.80 \
           PTM_BOWL_FIG7_NBTAIL=9 PTM_BOWL_FIG7_FMAX=0.88 PTM_BOWL_FIG7_RB=0.30,0.85
    export PTM_BOWL_FIG7_SWEEP=output/two_band_prod/fig7/fig7_sweep.csv
    ./build/bin/bowl_crater_ptm output/two_band_prod/fig7 0.05 0.10 0.20
    # figure:
    $PY src/python/bowl_crater_ingersoll_fig7.py

## Relationship to the other bowl figures

This OVERWRITES the superseded 2-panel `bowl_crater_ingersoll_fig7.png` with the
full composite. The 4-panel two-band diagnostic
`bowl_crater_ingersoll_twoband.png` (`bowl_crater_twoband_figure.py`) and the
gray-radiosity `bowl_crater_ingersoll_production.png` are left untouched.
