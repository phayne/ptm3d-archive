# Ice Transport Figure Captions

Figures generated from season-averaged PTM3D exosphere simulations of Europa.
All longitude values use the east longitude convention (0–360°E), with the
sub-Jovian point at 0°E, trailing hemisphere apex at 90°E, anti-Jovian point at
180°E, and leading hemisphere apex at 270°E.

---

## Data Source

Thermal baseline: PTM3D thermophysical model on an icosphere level-6 mesh
(81,920 triangular faces). Two orbital seasons computed with diurnal stride 2
(200 timesteps per diurnal cycle), phase-aligned and averaged to produce
season-mean, season-min, and season-max surface temperatures.

Exosphere transport model: Multi-timescale operator-splitting evolution over
100,000 epochs × 100 diurnal cycles/epoch ≈ 97,230 years to steady state.
Initial non-ice (dust/lag) concentration: 5% uniform. Bond albedo fixed at 0.5
(no albedo–composition feedback). Ice density: 917 kg/m³.

### Case 1: Sublimation only
- Transport driven by sublimation and ballistic redeposition only
- Net flux = deposition − sublimation − escape (sublimation escape ≈ 5%)
- Final non-ice concentration range: 5.0–9.6%, mean 5.7%
- Total sublimated: 2.05 × 10⁸ kg; total deposited: 1.95 × 10⁸ kg

### Case 2: Sublimation + sputtering
- Adds magnetospheric plasma sputtering (Cassidy et al. 2013; 2 × 10²⁷ H₂O
  molecules/s global rate)
- Trailing hemisphere (90°E) enhanced 3:1 over leading (270°E) via cosine
  ion flux distribution
- Temperature-dependent yield: Y(T) = Y₀[1 + q₀ exp(−Q/kT)] normalized at
  100 K (Teolis et al. 2017)
- Sputtered H₂O: 85% of total; O₂: 10%; H₂: 5%
- Escape fractions: H₂O 10%, O₂ 90%, H₂ 95%, giving a net ~22% of sputtered mass
  lost to space per timestep
- 90% of sputtered H₂O redeposits locally (same face; not transported
  ballistically)
- Final non-ice concentration range: 7.5–14.9%, mean 9.2%
- Total sublimated: 2.00 × 10⁸ kg; total deposited: 1.91 × 10⁸ kg;
  total sputtered: 4.98 × 10⁷ kg; total escaped: 5.94 × 10⁷ kg

---

## Globe Figures

### `globe_sublimation_only.png` / `.pdf`

**Caption**: Net ice flux for the sublimation-only case at steady state
(epoch 99,999), shown in four orthographic views: sub-Jovian (0°E), trailing
(90°E), anti-Jovian (180°E), and leading (270°E). Blue indicates net ice
deposition; red indicates net ice removal. The color scale uses a
`TwoSlopeNorm` centered at zero with per-case limits (−20.5 to +4.0 μm/kyr),
allowing the asymmetric positive and negative ranges to be resolved
independently. Graticule lines at 30° intervals. Elevation angle 15°.

Data range: −20.5 to +4.0 μm/kyr; 58% of faces show net deposition.

### `globe_sublimation___sputtering.png` / `.pdf`

**Caption**: Net ice flux for the sublimation + sputtering case at steady
state (epoch 99,999), same four views as above. Sputtering escape provides a
net mass sink at all latitudes and longitudes, so the entire surface shows net
removal (100% negative). The trailing hemisphere (90°E) shows the strongest
removal due to enhanced magnetospheric ion bombardment. Per-case color scale:
−43.6 to −6.1 μm/kyr (color scale still centered at 0 for comparison with
the sublimation-only case, though no positive values exist).

Data range: −43.6 to −6.1 μm/kyr; 0% positive.

### `globe_comparison.png` / `.pdf`

**Caption**: Side-by-side comparison of net ice flux at steady state for the
sublimation-only (left column) and sublimation + sputtering (right column)
cases, using a common `TwoSlopeNorm` color scale spanning both datasets
(−43.6 to +4.0 μm/kyr). Four views per column: sub-Jovian (0°E), trailing
(90°E), anti-Jovian (180°E), leading (270°E). View labels centered between
columns. Shared RdBu colorbar with explicitly computed tick locations for both
the compressed positive and extended negative ranges. This figure highlights
how sputtering converts the sublimation-only pattern (net deposition at poles,
net removal at equator) into global net removal dominated by the trailing
hemisphere.

---

## Mollweide Figures

### `mollweide_sublimation_only.png` / `.pdf`

**Caption**: Mollweide equal-area projection of net ice flux for the
sublimation-only case at steady state. Centered at 0°E (sub-Jovian point).
Longitude labels in east longitude convention (0°E, 90°E, 180°E, 270°E);
latitude labels at 30° intervals. The zonal banding shows equatorial
sublimation (red, peaking at sub-solar latitudes ±15°) and polar deposition
(blue, poleward of ±50°). Per-case `TwoSlopeNorm` color scale.

Data: Regridded from 81,920 icosphere faces to 181 × 361 regular lat–lon
grid via linear interpolation (nearest-neighbor gap fill at boundaries).

### `mollweide_sublimation___sputtering.png` / `.pdf`

**Caption**: Mollweide equal-area projection of net ice flux for the
sublimation + sputtering case at steady state. Same projection and labeling as
above. Sputtering produces net removal everywhere; the trailing hemisphere
(centered on 90°E) shows the strongest ice loss due to the 3:1 ion flux
enhancement. The equatorial band near the trailing apex shows the deepest
removal (−44 μm/kyr) from combined peak sublimation and peak sputtering.

### `mollweide_comparison.png` / `.pdf`

**Caption**: Stacked Mollweide projections comparing (a) sublimation-only and
(b) sublimation + sputtering net ice flux at steady state. Both panels share a
common `TwoSlopeNorm` color scale (−43.6 to +4.0 μm/kyr) and a single
horizontal colorbar. Panel labels follow (a)/(b) convention. This projection
shows most clearly the longitudinal asymmetry introduced by sputtering: while
sublimation-only transport is nearly symmetric in longitude (slight asymmetry
from illumination geometry), sputtering imposes a strong trailing/leading
contrast.

---

## Profile Figure

### `transport_profiles_combined.png` / `.pdf`

**Caption**: Equatorial and latitudinal profiles of ice transport quantities,
comparing sublimation-only (blue) and sublimation + sputtering (orange) cases.
Six panels arranged in a 3×2 grid:

**(a)** Surface temperature vs. east longitude along the equatorial band
(|latitude| < 5°). Single dark gray curve (identical for both cases since
albedo feedback is disabled): solid line is the annual (season) average; shaded
band spans the diurnal min–max range. Peak temperature ~132 K at the
sub-solar point.

**(b)** Zonal-mean surface temperature vs. latitude. Annual average and
min–max range. Case colors shown for reference (both overlay exactly).

**(c)** Normalized vapor pressure P_v / P̄ vs. east longitude along the
equatorial band (log scale). P̄ is the area-weighted global mean vapor
pressure. Values > 1 indicate faces that preferentially sublime; values < 1
indicate faces where deposition is favored. Both cases show P_v/P̄ peak near
the sub-solar point and minimum at the terminator.

**(d)** Zonal-mean P_v / P̄ vs. latitude (log scale). Strong equator-to-pole
gradient spanning ~8 orders of magnitude. Both cases overlap because
temperatures are identical.

**(e)** Net ice transfer rate (μm/kyr) vs. east longitude along the equatorial
band. Negative = net removal, positive = net deposition. Sublimation-only
(blue) shows removal at the sub-solar point and near-zero elsewhere. Sputtering
(orange) shifts the entire curve downward by the local sputtering rate,
producing stronger removal at all longitudes with trailing hemisphere
enhancement.

**(f)** Zonal-mean net ice rate vs. latitude. Sublimation-only case shows the
classic equatorial sublimation / polar deposition pattern. Sputtering case
shows net removal at all latitudes, with the most negative values at the
equator.

Equatorial binning: 72 bins of 5° width in the |lat| < 5° band.
Latitudinal binning: 90 bins of 2° width, zonal average over all
longitudes.

---

## Reproduction Instructions

### Globe and Mollweide figures

```bash
python3 scripts/plot_ice_transport.py \
  --data-dir output/europa_ico6_uniform-ice_averaged/exosphere/sublimation_no-albedo \
             output/europa_ico6_uniform-ice_averaged/exosphere/sputtering_no-albedo \
  --labels "Sublimation only" "Sublimation + sputtering" \
  --save-dir docs/figures/transport_averaged
```

### Profile figure

```bash
python3 scripts/plot_transport_profiles.py \
  --data-dir output/europa_ico6_uniform-ice_averaged/exosphere/sublimation_no-albedo \
             output/europa_ico6_uniform-ice_averaged/exosphere/sputtering_no-albedo \
  --labels "Sublimation only" "Sublimation + sputtering" \
  --save-dir docs/figures/transport_averaged
```

### Dependencies

- Python 3, NumPy, SciPy, Matplotlib (≥3.4 for `TwoSlopeNorm`)
- Input data: PTM3D exosphere output with binary per-face fields (`net_flux_kg_m2_s/`,
  `temperature_mean_K/`, `vapor_pressure_Pa/`, etc.) and `mesh.ply` (binary PLY
  with per-face latitude/longitude attributes)

### Script parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Colormap | `RdBu` | Diverging blue (deposition) / red (removal) |
| Norm | `TwoSlopeNorm(vcenter=0)` | Asymmetric diverging scale centered at zero |
| Globe elevation | 15° | Slight tilt for polar visibility |
| Graticule | 30° spacing, color `'0.55'`, linewidth 0.2, alpha 0.35 | Subtle grid overlay |
| Regrid resolution | 181 × 361 (0.5° lat × 1° lon) | For Mollweide projection |
| Ice density | 917 kg/m³ | For kg/m²/s to μm/kyr conversion |
| Unit conversion | 1 kg/m²/s = (10⁶/917) × (1000 × 365.25 × 24 × 3600) μm/kyr | ≈ 3.44 × 10¹³ μm/kyr per kg/m²/s |

---

## Diurnal Surface Pressure Figure

### `diurnal_surface_pressure.png` / `.pdf`

**Caption**: Global area-weighted mean surface pressure vs. orbital phase for
the season-averaged diurnal cycle (288 timesteps over one Europa orbit = 3.55
days). Two panels:

**(a)** Surface pressure contributions from H₂O sublimation (solid blue),
H₂O sputtering (dashed orange), O₂ sputtering (dash-dot green), H₂ sputtering
(dotted red), and total sputtering (thin gray). The sublimation pressure varies
by ~5 orders of magnitude (3 × 10⁻¹⁴ to 2 × 10⁻⁹ Pa) over the diurnal cycle,
with a sharp minimum near 75° orbital phase when the nightside hemisphere faces
the observer. Sputtering-derived pressures are nearly constant (~10⁻⁹ Pa for
H₂O, ~10⁻¹¹ Pa for O₂ and H₂), because the sputtering source rate depends
weakly on temperature through the Teolis et al. (2017) yield enhancement.

**(b)** Ratio of total sputtering pressure to H₂O sublimation pressure. The
ratio exceeds unity (dotted horizontal line) during the nightside portion of
the orbit (orbital phase ~60–200°), indicating that sputtering dominates the
exosphere surface pressure when sublimation is suppressed by low temperatures.

Physics: Sublimation pressure computed from the Marti & Mauersberger (1993)
vapor pressure formula: log₁₀(P [Pa]) = −2663.5/T + 12.537. Sputtering
equivalent pressure computed by equating the non-escaping sputtered flux to
Maxwellian effusion (P = nkT where n = 4Φ/v_th). Sputtering model uses
Cassidy et al. (2013) global rate of 2 × 10²⁷ H₂O molecules/s with 3:1
trailing/leading hemisphere contrast and Teolis et al. (2017) temperature-
dependent yield. Species fractions: H₂O 85%, O₂ 10%, H₂ 5%; escape fractions:
H₂O 10%, O₂ 90%, H₂ 95%.

Data source: 288 per-timestep temperature files from the season-averaged
PTM3D simulation (81,920 icosphere faces).

---

## Tangent Column Density Figure

### `tangent_column_density.png` / `.pdf`

**Caption**: Tangent (limb) column density of the combined H₂O + O₂ + H₂
exosphere for four sub-observer views: sub-Jovian (0°E), trailing (90°E),
anti-Jovian (180°E), and leading (270°E). Each panel shows Europa's disk
(filled gray circle, radius = 1 R_Europa) with the exosphere column density
rendered as a 2D color map on a log₁₀ scale (cm⁻²) for lines of sight tangent
to the surface at each apparent position.

Color scale: log₁₀(N_tang) from 6 to 17 cm⁻², using the `inferno` colormap.
The bright ring immediately outside the disk represents the dense near-surface
exosphere (column densities ~10¹⁴–10¹⁷ cm⁻² at the equatorial limb); column
density falls off exponentially with altitude on the scale height H = kT/(mg).
The thin equator and pole markers (gray crosshairs) indicate the disk geometry.

Physics: For an exponential atmosphere with surface number density n₀ and
scale height H, the tangent column density at altitude h above the surface is:

N_tang(h) = n₀ √(2πRH) exp(−h/H)

where R = 1560.8 km is Europa's radius. The surface density n₀ includes both
sublimation (n_sub = P_v/kT) and sputtering contributions (n_sput = 4Φ/v_th)
for all three species. Each species has its own scale height: H_H₂O = kT/(m_H₂O g),
H_O₂ = kT/(m_O₂ g), H_H₂ = kT/(m_H₂ g). At 100 K: H_H₂O ≈ 35 km,
H_O₂ ≈ 20 km, H_H₂ ≈ 313 km.

Limb profile construction: For each sub-observer longitude, limb faces are
selected within ±30° of the limb circle (at observer longitude ± 90°). Surface
density n₀ and scale height H are binned by latitude (90 bins of 2° width),
area-weighted. For each image pixel outside the disk, the tangent altitude
h = (r − 1)R_Europa and the apparent latitude are computed, and the column
density is evaluated by interpolating the latitude-binned profiles.

Data source: Diurnal-average temperature (mean of 288 timesteps) per face
from the season-averaged PTM3D simulation.

---

## Scale Height Figure

### `scale_height_mollweide.png` / `.pdf`

**Caption**: Mollweide equal-area projection of the H₂O scale height H = kT/(mg)
at the diurnal-average surface temperature, centered at 0°E (sub-Jovian point).
Scale height ranges from ~20 km at the poles (T ≈ 51 K) to ~38 km at the
sub-solar equatorial region (T ≈ 109 K). The `viridis` colormap spans the
1st–99th percentile range.

This quantity determines the vertical extent of the sublimation-driven
exosphere: the tangent column density at altitude h falls as exp(−h/H).
The equator-to-pole gradient in H reflects the corresponding temperature
gradient and controls the latitudinal structure visible in the limb profile
figures.

Data source: Same diurnal-average temperature as the tangent column density
figure.

---

## Reproduction Instructions (Diurnal Pressure & Column Density Figures)

### Diurnal pressure, limb profiles, and scale height

```bash
python3 scripts/plot_diurnal_pressure.py \
  --ptm-dir output/europa_ico6_uniform-ice_averaged \
  --save-dir docs/figures/transport_averaged
```

### Optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hemisphere-contrast` | 3.0 | Sputtering trailing/leading ratio |
| `--sputtering-scale` | 1.0 | Scale factor for global sputtering rate |

### Physical constants (matching C++ code)

| Constant | Value | Source |
|----------|-------|--------|
| H₂O molecular mass | 2.99151 × 10⁻²⁶ kg | exosphere_physics.h |
| Europa radius | 1560.8 km | exosphere_physics.h |
| Europa surface gravity | 1.314 m/s² | exosphere_physics.h |
| Global H₂O sputtering rate | 2 × 10²⁷ molecules/s | Cassidy et al. 2013 |
| Trailing/leading contrast | 3:1 | sputtering_model.h |
| Yield activation energy | 0.06 eV | Teolis et al. 2017 |
| O₂ molecular mass | 5.3135 × 10⁻²⁶ kg | sputtering_model.h |
| H₂ molecular mass | 3.3474 × 10⁻²⁷ kg | sputtering_model.h |
