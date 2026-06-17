# Cold PSR Floor vs Diviner: Three Physical Hypotheses

**Status:** investigation in progress (started 2026-06-08). Test case: Haworth
central permanently-shadowed floor (−87.45°, −2.1°), summer (peak-illumination)
season. Companion to `docs/psr-floor-temperature-validation.md` (which ruled out
view-factor, illumination, cp, d/D, equilibration, and cadence errors).

## 1. The deficit, in flux terms

Williams et al. (2019) seasonal Diviner, Haworth central floor (<1.5 km), vs PTM3D:

| | Diviner | PTM3D | implied floor flux (εσT⁴, ε=0.95) |
|---|---|---|---|
| summer **max** | 64.6 K | 45.4 K | 0.94 vs **0.23 W/m²** |
| summer **mean** | 46.3 K | 36.4 K | 0.25 vs **0.09 W/m²** |
| winter mean | 30.2 K | ~30 K ✓ | 0.045 |

Both the summer **max and mean** are deficient (the mean by ~10 K), so this is at
least partly a **genuine flux deficit**, not only a measurement-statistic effect.
A pure measurement bias (H3) cannot move the mean.

The Ingersoll–Svitek–Murray (1992) bowl-crater closed form with Haworth's measured
d/D = 0.051 and peak Sun elevation e = 4.1° gives the floor's radiative-equilibrium
max **T = 64.9 K, requiring ~1.0 W/m²** of absorbed floor flux when the **whole**
warm rim ring is seen. PTM3D delivers **0.49 W/m² at the peak instant — exactly
half** — because at e ≈ 4° only ~half the rim ring is sunlit at any instant. The
floor's *delivered* peak (0.49) is itself larger than the εσT⁴ implied by its
45.4 K max (0.23), i.e. the floor does **not** fully equilibrate to the brief peak —
**duration/duty-cycle matters**, not just peak magnitude.

## 2. Current-model audit (what is and isn't in the code)

From a read-only audit of `src/cpp/radiation/`:

- **Reflected solar is included** in the radiosity as a separate source
  `B = εσT⁴ + ρ·F·B + ρ·Q_solar` (`radiation_exchange.cpp:452,494,558`). A shadowed
  floor face *does* receive reflected sunlight from sunlit walls.
- **…but with `ρ = 1 − ε ≈ 0.05`** (thermal reflectivity), set at
  `radiation_exchange.cpp:36`, **not the solar Bond albedo 0.12**. The wall absorbs
  88% of insolation (Bond albedo used for *direct* absorption) yet is modeled as
  re-scattering only ~5%. **The reflected-solar channel into the floor is under-fed
  by ~2.4×** — a genuine bug.
- **Exchange is purely Lambertian** (`F_ij = A_j cos_i cos_j/(π r²)`,
  `view_factor.cpp:1158`); no directional/phase-function weighting except the
  optional Smith term.
- **Smith (1967) beaming** (`smith_beaming.h`, applied in `addBeamedFlux`,
  `radiation_exchange.cpp:1514`, gated by `rms_slope_deg`, default 0 = OFF) is
  **thermal-IR only — never applied to reflected solar** — and energy-normalized
  (pure angular redistribution of *isothermal* emission). It beams toward the Sun
  (`chi` peaks at `cos_psi = d·ŝ → 1`, `smith_beaming.h:21`); the geometry in the
  code is **correct** — `dir_ji` is emitter→receiver = wall→floor
  (`radiation_exchange.cpp:1489`), and the floor sits at ~10° phase
  (§3-H1), so `cos_psi ≈ +0.985` and `chi > 1` → it correctly beams **toward** the
  floor (the prior study's "+0.5 K, right sign"). The smallness is **not** a
  geometry error; it is the **isothermal-facet assumption** (see §3-H1).
- **Extension hook** for an alternative phase function: `addBeamedFlux()` +
  `buildBeamPairs()` carry per-pair geometry (emitter→receiver direction, emission
  angle, Sun direction, VF weight, temperatures) — enough to evaluate any phase
  function, including an anisothermal directional-emissivity model and a
  backscattering **solar** lobe (both currently absent).

## 3. Back-of-envelope per hypothesis

### H3 — anisothermality / measurement bias: **largely ruled out for the deep floor**

Claim under test: unresolved roughness gives a floor facet T-distribution; the
T⁴-weighted bolometric brightness temperature `T_bol = (⟨T⁴⟩)^¼` is pulled toward
warm facets, reading ~65 K where the mean physical T is ~40 K.

The kernel of truth — `T_bol = ⟨T⁴⟩^¼ ≥ ⟨T⟩` (Jensen) — is real, **but the model's
footprint comparison already computes `T_bol = ε^¼(Σ A_i T_i⁴/Σ A_i)^¼` over the
mesh facets in each 250 m cell.** So the question is only whether *sub-mesh*
refinement raises `⟨T⁴⟩`. It does not, for the deep floor:

1. **Energy conservation.** In (quasi-)radiative equilibrium each facet obeys
   `εσT_i⁴ = absorbed_i`, so `⟨T⁴⟩ = ⟨absorbed⟩/(εσ)`. The total indirect
   intercepted by a rough patch equals that of the flat patch — the projected area
   toward the rim is conserved, `Σ_i F_i A_i = F_flat A_flat` — or is *reduced* by
   self-shadowing. Hence `⟨T⁴⟩ ≤ T_flat⁴` and **`T_bol(rough) ≤ T_bol(flat)`**:
   refining resolution cannot raise T_bol; if anything roughness cools it.
2. **The warm tail is energy-limited.** To keep `⟨T⁴⟩ = 45⁴` while putting facets at
   80 K (T⁴ = 4.1e7 = 16× the mean) requires < 6% of the area at 80 K with the rest
   near 0 K — and even then `⟨T⁴⟩^¼` is unchanged at 45 K *by construction*. You
   cannot have a mean of 40 K **and** a substantial 70–80 K population without
   **extra energy**.
3. **No sub-grid direct Sun on the deep floor.** The only way to add that extra
   energy is sub-resolution facets catching direct sunlight. At Haworth the floor's
   rim horizon is ~6° and the peak Sun is 4.1° — **the Sun is below the rim from
   every point on the floor**, so no facet, however tilted, sees it.

**Verdict:** for the deep central floor, H3 cannot bias T_bol upward and cannot
touch the mean deficit. It *does* operate where there is genuine sub-resolution
warm terrain — shallower PSRs, near-rim cells, pole-facing-slope bases that catch
the grazing Sun (the known "sub-km craters → +63 K" effect). → De-prioritize for the
deep floor; a quick numerical check of (1)–(2) on the model's own facet distribution
is cheap and will confirm.

**Important synthesis:** the anisothermality intuition behind H3 is correct — it is
just **mislocated**. It is energy-conservation-protected (toothless) on the shadowed
floor, but **dominant on the sunlit wall** (§3-H1a), where sub-grid direct Sun
creates a real hot/cold micro-facet contrast and the floor, viewing the wall at ~10°
phase, preferentially sees the hot facets. H3-on-the-floor ≈ 0; H3-on-the-wall (=H1a)
is potentially the leading term.

### H2 — high-thermal-inertia walls/rim: **most physical; addresses mean *and* max**

The floor is duty-cycle-limited: at the peak instant only ~28% of the rim is warm.
The Ingersoll "whole ring warm" value (64.9 K) is the ceiling H2 can reach by
**sustaining** previously-lit rim sectors as the Sun circles in azimuth
(~12°/day at Haworth). Two competing effects of raising wall TI:

- **(−)** lower each wall facet's *peak* T (less spiky) → lower instantaneous flux
  when a sector is directly lit;
- **(+)** raise each wall facet's *off-peak* T (cools slower) → more sectors warm
  simultaneously, for longer → more **sustained** floor flux.

Because the floor integrates over time (and εσT⁴ for the floor is set by the
*sustained*, not the instantaneous-spike, flux), the **(+)** duration effect should
dominate. Magnitude depends on the wall cooling time vs the azimuthal sweep: 10×
regolith TI (Γ ~ 50 → ~500) extends warm-lag from ~hours to ~a day, widening the
warm arc modestly (BOE **+5 to +15 K**, not obviously the full +20). Headroom exists
(up to 64.9 K). → **Highest-value test.** Start with a *simple* wall-cooling +
ring-integral model (no full PTM3D), then a 2-component run if promising.

### H1 — wall beaming (corrected geometry): **promoted to a co-leading lever**

**The geometry (corrected 2026-06-08).** Put the Sun in +x at e = 4° elevation. The
rays graze the near rim and light the **far inner wall** (its inner normal points
back toward +x/up, toward the Sun). The floor sits at bottom-center. The wall→floor
direction is nearly horizontal (+x) and ~5.9° **down** (2.62 km over 25.5 km), while
wall→Sun is +x at 4° **up**. The **phase angle** between them is therefore only
**α ≈ 4° + 5.9° ≈ 10°** (vector check: 9.85°). *The floor sits deep in the
backscatter / opposition lobe of the sunlit wall* — it views the wall almost back
along the incoming sunbeam, just below it. (The floor never sees the Sun-side rim's
sunlit outer face; all the sunlit terrain it sees is far wall at small phase.) This
is the key correction: emission/scattering beamed **toward the Sun is beamed toward
the floor**, not away from it.

Three consequences, all currently mis- or un-modeled:

- **(a) Anisothermal wall IR — the big one.** The wall's mean facet temperature is
  ~290 K, but sub-resolution rough micro-facets that face the grazing Sun run far
  hotter: a facet tilted ~25° toward the 4° Sun absorbs `S·cos(31°)·(1−A) ≈ 1030
  W/m²` → radiative-equilibrium **T ≈ 370–390 K**, while shadowed micro-facets are
  cold. εσT⁴ scales as (370/290)⁴ ≈ **2.6×**. Because the floor views the wall at
  ~10° phase (backscatter), it **preferentially sees exactly these hot Sun-facing
  micro-facets** → the wall's *effective brightness temperature toward the floor*
  exceeds its mean. The Lambertian radiosity uses the facet **mean** T and misses
  this entirely; the Smith term has the right *direction* but assumes **isothermal**
  facets, so it captures only the weak angular redistribution (≈+0.5 K) and none of
  the (370/290)⁴ temperature contrast — explaining the "~30× too small." A
  visibility-weighted hot-facet ensemble plausibly raises the floor IR by **~1.5–2.5×**
  (0.49 → ~0.7–1.2 W/m²) — potentially gap-closing. This is the **wall-side analog of
  H3**: anisothermality has teeth *here* (sub-grid direct Sun creates a real
  temperature contrast), whereas on the shadowed floor energy conservation kills it.
  This is the lunar **directional/anisothermal thermal emission** effect
  (Bandfield 2015; Rubanenko & Hayne).
- **(b) Reflected-solar backscatter (opposition).** Lunar regolith is strongly
  retroreflective; near α ≈ 10° the bidirectional reflectance exceeds the Lambert
  value by ~2–3× (opposition surge + intrinsic backscatter). So reflected sunlight is
  **beamed toward the floor**, not isotropic. The model has it **Lambertian** (no
  surge) and the channel is entirely un-beamed.
- **(c) Reflected-solar albedo fix (ρ = 0.05 → 0.12).** Genuine bug (thermal
  reflectivity used for solar scattering). At the warm wall reflected sunlight is
  ~20% of the IR (A·Q_w ≈ 0.12·680 ≈ 82 vs εσ290⁴ ≈ 381 W/m²); combining the 2.4×
  albedo fix with the ~2–3× backscatter surge makes the floor-directed solar channel
  ~**6×** its current value — from ~7% to ~40% of the floor IR.

**Net BOE:** with the corrected backscatter geometry, H1 is **not** the modest lever I
first estimated. Anisothermal wall IR (a) alone could ~double the floor flux, with the
solar surge (b)+(c) adding tens of percent more. Both are physically well-motivated
(known lunar effects) and **absent or isothermal-only** in the current model.

**Simple-model result (`src/python/h1a_anisothermal_beaming.py`, 2026-06-08).** A
Gaussian micro-slope ensemble on the sunlit wall (each facet at its own
radiative-equilibrium T(incidence) with Smith G1 masking; directional brightness toward
the floor at the real ~10° phase, ratioed to the mean-T Lambertian value) gives an
enhancement

> **E ≈ 2.0** (1.6–2.3 over RMS micro-slope 20–30°, background fraction β = 0.2, wall
> slope 8–11°; hot-spot-correlated bracket 2.2).

with the right physical sensitivities: E **grows with RMS slope** (more hot Sun-facing
facets), **shrinks with a warmer background** (β dilutes the contrast), and **grows with
a more grazing floor→wall view** (deeper crater: E 1.93→2.17 over d/D 0.03→0.12). E ≈ 2
raises the **delivered** floor flux from 0.49 to ~0.98 W/m² — almost exactly the
Ingersoll-required 0.94. The **realized** floor-temperature gain is bracketed:

| | conservative (45·E^¼, response unchanged) | equilibrium (Ingersoll) |
|---|---|---|
| E = 2.0 | **53.5 K (+8)** | **65.3 K (+20)** |

i.e. H1a *supplies* the missing flux; whether the floor reaches 64.6 K also needs it to
*equilibrate* to that flux (the model floor currently under-responds — 45 K for 0.49
W/m² — because the brief seasonal peak is inertia-smoothed and drains to the cold deep
column). **That under-response is exactly H2's domain** (a more sustained warm ring lets
the floor equilibrate). So H1a and H2 are **complementary**, and together plausibly close
the ~20 K gap.

**Solar opposition refinement (combined budget).** Adding the reflected-solar channel
to the same ensemble — albedo fix (ρ 0.05 → Bond 0.12, ×2.4), shadow-hiding (reflected
∝ μ_s like absorbed, so it gets the same ~2× backscatter selectivity), and an intrinsic
particle/coherent backscatter surge B_opp ≈ 1.3 — gives:

> floor flux **0.49 → 1.08 W/m²** (×2.2): IR anisothermal **0.93** + reflected-solar
> opposition **0.15** (14% of total). Floor T: **66.9 K** (equilibrium) / **54.8 K**
> (conservative).

So the **reflected-solar opposition is a real but secondary add-on (~+2 K equilibrium)**;
the **anisothermal IR is the dominant lever**. The albedo fix (ρ→Bond) is the free,
unambiguous part.

**Correction + azimuthal integration (decisive, 2026-06-08).** Two refinements walk the
optimistic single-pair `E ≈ 2` back to the production-relevant number:

1. **Right baseline.** The production code multiplies the wall's flat `εσT_j⁴` by a
   factor, so the physically-correct enhancement is *rough-anisothermal / flat-Lambertian*
   — the **energy-normalized Smith χ**, not my `E` (which used rough-*isothermal* as the
   denominator). Crucially, the existing Smith χ **already weights emission by `m·ŝ`**
   (hot Sun-facing facets) — it is *not* isothermal-only as I'd claimed. At the far-wall
   backscatter geometry (cos_sun 0.26, cos_e 0.09, cos_psi 0.98) the production 24×24×33 χ
   table gives **χ = 1.91**, matching a direct quadrature (1.92).
2. **Azimuthal integration.** The floor sees the *whole* sunlit arc, not just the far
   wall. Integrating the per-azimuth χ weighted by each wall's T⁴ (∝ μ_s) gives a floor
   enhancement of **≈ 1.64** (w = 20°) — it does **not** dilute, because the hottest wall
   (φ = 180°, μ_s = 0.26) sits exactly at the χ = 1.93 backscatter peak; the cold side
   walls (χ < 1) contribute little.

**So the physics predicts the Smith beaming should give a ~1.6× floor-flux enhancement —
yet the production run gave only +0.5 K** ("~30× too small"). The χ is correct, so the
loss is **downstream of χ**: the beamed flux is *real but brief and intermittent* (the
far-wall backscatter peak is narrow in time/season), and the floor's seasonal-**max**
barely responds because its inertia + conduction to the cold deep column smooth the brief
peak — **the same under-response that is H2's domain.** Net: **H1a delivers the flux
magnitude; H2 delivers the duration so the floor can equilibrate to it. Neither alone
closes the gap** (H1a alone ≈ the known +0.5 K; H2 alone leaves the un-beamed peak too
low). → The PTM3D step is therefore a **diagnostic** (instrument the beamed-flux magnitude
vs the floor-T response, then test beaming + sustained-wall-TI together), not a from-scratch
re-implementation — the existing Smith term is the right vehicle and its χ is correct.

## 4. Recommended sequence (BOE → simple → complex)

Reprioritized after the H1 geometry correction — **H1 (anisothermal wall beaming)
now co-leads with H2**:

1. **H1a anisothermal-wall emission — simple model first** (potentially the largest,
   newly promoted): a Gaussian-slope micro-facet ensemble on the sunlit wall, each
   facet at its own radiative-equilibrium T(incidence), integrate the
   backscatter-direction (≈10° phase, wall→floor) brightness; ratio to the mean-T
   Lambertian value = the floor-IR enhancement. Add a solar backscatter (opposition)
   lobe + the ρ→Bond-albedo fix for the reflected channel. Target: does it reach the
   ~2× floor-flux enhancement the BOE suggests?
2. **H2 wall-TI — simple model in parallel** (most physical, addresses mean+max): a
   lightweight wall-facet thermal model over the synodic cycle at 1× vs 10× TI →
   warm-arc width vs azimuth → ring-integrated floor flux → floor T.
3. **H3 confirmation (cheap):** a short numerical check that sub-facet refinement of
   the model's own floor T-distribution leaves `⟨T⁴⟩^¼` unchanged (confirms §3-H3
   for the deep floor).
4. **Full PTM3D** only for whichever of H1/H2 survives the simple models — H1a would
   become a directional/anisothermal-emissivity term in `addBeamedFlux`; H2 a small
   slope/elevation property patch.

(Numbers: σ = 5.670×10⁻⁸ W m⁻² K⁻⁴, ε = 0.95, S = 1361 W m⁻², A_Bond = 0.12,
Haworth d/D = 0.051, e_peak = 4.1°, rim-horizon ≈ 6°, floor F_terrain ≈ 0.0055.)

## 5. PTM3D diagnostic result (2026-06-08) — beaming is real but small

Implemented (env-gated, default byte-identical): **albedo fix** (`PTM_SOLAR_ALBEDO`,
reflected-solar source uses the Bond albedo) and **solar opposition beaming**
(`PTM_BEAM_SOLAR`, `addBeamedFlux` beams the reflected-solar source with the same χ, not
just thermal IR — `radiation_exchange.cpp`). Diagnostic: Haworth 35-day summer-peak
window, eq25, **CPU radiosity** (so the albedo fix is in the loop), 14.9 M wall→floor beam
pairs at rms = 20°. Run A = baseline; Run C = IR + solar beaming + Bond albedo
(`examples/hmfix_diag_{a,c}.yaml`, `src/python/h1_diag_compare.py`). Central floor =
4961 PSR faces within 2 km of center:

| | Run A | Run C | Δ |
|---|---|---|---|
| central-floor summer-max (median) | 41.1 K | 43.0 K | **+1.9 K** |
| delivered peak indirect flux (median) | 0.436 W/m² | 0.501 W/m² | **+15%** |

**Interpretation:**
1. **Beaming (IR + solar opposition + albedo) is real but small: +15% flux, +1.9 K.**
   Larger than the IR-only Smith +0.5 K (the solar opposition roughly triples it), but
   nowhere near the ~20 K gap. The integrated +15% is far below the idealized single-pair
   χ ≈ 1.6–1.9 because the floor's flux is **diluted** — much of it comes from cold near
   walls, off-peak sectors, and multi-bounce that get no backscatter enhancement; only the
   sunlit-far-wall-at-backscatter fraction is boosted. My idealized ring integration
   over-stated it by assuming the floor flux is dominated by that one wall.
2. **The dominant deficit is the floor's UNDER-RESPONSE, not the flux magnitude.** The
   floor sits at 41 K while the radiative-equilibrium value for its *delivered* 0.44 W/m²
   is ≈ 54 K (ε≈0.9 in the radiosity). That ~13 K gap is the **conductive drain to the
   cold (~30 K) deep column + inertial smoothing of the brief seasonal peak** — i.e. **H2's
   domain**, and a far bigger lever (~+13 K available) than beaming (+1.9 K). The floor
   *does* respond proportionally to a flux *change* (+15% → +1.9 K ≈ flux^¼ on the 41 K
   base), so it is not insensitive — it is held below equilibrium by the cold reservoir.

**Roughness sweep (rms 20→30→40°, Bandfield et al. 2015; ε=0.95 throughout).** Re-ran
the diagnostic at higher RMS micro-slope to test whether more hot Sun-facing facets help.
The deep-PSR floor (68 k faces, baseline Tmax < 55 K) median summer-max:

| RMS slope | floor Tmax | Δ vs base | delivered flux |
|---|---|---|---|
| 0 (base) | 42.4 K | — | 0.503 W/m² |
| 20° | — | +1.9 K | +15% |
| 30° | 44.6 K | **+2.17 K** | +17% |
| 40° | 44.7 K | **+2.23 K** | +17% |

**The beaming SATURATES**: 30° → 40° adds only +0.06 K (flux flat at +17%). The Smith χ
at the floor's grazing-backscatter geometry stops growing (energy-normalized + clamped),
and the flux dilution caps the integrated effect regardless of roughness. So **even at
Bandfield's high roughness, beaming is a ~+2 K effect** — it cannot close the ~20 K gap.
(The earlier "ε≈0.9" was a display-rounding artifact: `emissivity_override: 0.95` logged
`0.95 → 0.95`, confirming the model was already at ε=0.95.)

**Conclusion (H1):** beaming is a correct, now-implemented, but **second-order** effect
(~+2 K, saturating with roughness). Keep the knobs (`PTM_BEAM_SOLAR` + `PTM_SOLAR_ALBEDO`).
*(See §6 — H2 then turned out NOT to be a lever either, which reframes this.)*

## 6. H2 simple model (2026-06-08) — wall thermal inertia is NOT a lever

`/tmp/h2_wall_ti_model.py` (→ `src/python/`): a 1D-conduction wall-ring + floor model.
N=24 azimuthal wall columns + 1 floor column, implicit conduction with cp(T) (Hayne-like),
proper polar insolation (subsolar lat ±1.54°/yr; Sun circles in azimuth once per synodic
month, rises/sets at −87.45°), floor driven by `F_terrain·⟨εσT_wall⁴⟩`. Floor TI fixed
(low); wall TI scaled 1×/3×/10× (k ∝ TI²):

| wall TI | floor summer-max | floor annual-mean | wall summer-max | wall off-peak Tmin |
|---|---|---|---|---|
| 1× | 56.6 K | 51.5 K | 275 K | 69 K |
| 3× | 56.4 K (−0.2) | 51.4 K | 273 K | 90 K |
| 10× | **56.1 K (−0.5)** | 51.5 K | 269 K | 118 K |

**Higher wall TI does NOT warm the floor — it slightly cools it.** The result is robust
(it survives cp(T) and rests on energy conservation, not calibration):
- **Energy conservation fixes the floor's annual-mean flux.** Over a cycle the wall emits
  exactly what it absorbs (∫σT_wall⁴ dt conserved) regardless of TI, so the floor's
  time-integrated indirect flux — and hence its **deep-column temperature and the
  conductive drain** — is *unchanged* by wall TI. My earlier "H2 warms the deep column →
  floor equilibrates" reasoning was **wrong**: TI redistributes wall emission in time, it
  does not add any.
- **The floor's summer flux is T⁴-dominated by the instantaneously-sunlit peak walls**
  (~270 K), which high TI slightly *lowers* (Jensen). The off-peak trailing walls, even at
  118 K with 10× TI, are far too cool (118⁴/270⁴ ≈ 3.6%) to matter in the T⁴-weighted sum.
  Realistic TI cannot keep a wall hot over the ~2-week dark half of the synodic cycle.

**So none of H1 (+2 K), H2 (~0), or H3 (~0, §3) closes the ~20 K gap.** All three are
either second-order or energy-conservation-forbidden. This points away from a
floor-thermophysics fix and toward: (a) the Diviner ~65 K representing warmer terrain than
the deep central floor (basin / wall-toe / pole-facing-slope base — the earlier red-team
reading), and/or (b) the floor's **annual-mean** indirect flux (deep-column temperature)
being under-delivered in PTM3D — note this idealized ring gives a 51 K deep column vs
PTM3D's ~30 K, a 20 K difference that is the actual size of the gap and is set by the
*annual-mean* warm-wall coupling (VF / duty cycle), not by TI or beaming.
