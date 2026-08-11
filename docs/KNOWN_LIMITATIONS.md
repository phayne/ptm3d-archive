# PTM3D — Known Limitations and Gotchas

What is known to be wrong or misleading right now, with measured magnitudes and the populations
affected; modeling limitations; configuration behavior worth knowing before a production run; and
workflow caveats. Reading the run log resolves several of them — always confirm what the run
actually loaded.

Every figure below is a measurement unless it is labeled as an extrapolation or as arithmetic on a
configuration. Where a limitation has an open internal tracking identifier, it is given in the
reference table that closes the numbered sections; the entries stand on their own without it.

---

## Is PTM3D right for my problem?

Most people who pick up PTM3D are doing something it does well. A page about deep columns should
not put you off; it should tell you which of the two columns below your question sits in.

**Well established, and what the model is validated against**

| Use | Basis |
|---|---|
| Surface temperatures and diurnal cycles | Lunar Diviner bolometric temperatures, RMS < 4 K; the lunar equatorial cycle reproduces noon ~388 K / night ~99 K |
| Peak and mean temperatures (`temperature_max`, `temperature_avg`) | Insensitive to the spin-up limitation below: lengthening lunar spin-up from 50 to 100 lunations moves lit-face `temperature_max` by a mean of 0.016 K |
| Illumination, shadowing, terminator geometry on real terrain | Embree ray tracing with a finite solar disk; crater-floor illumination validated to ~99% against an independent clean-Embree reference |
| Inter-face radiation exchange and self-heating | Analytic bowl-crater view factor to 0.49% (Ingersoll et al. 1992); view-factor reciprocity to machine precision |
| Thermal inertia on lit faces; asteroid radiometry-style forward modeling | The diurnal amplitude and phase lag are surface-layer quantities and equilibrate quickly |
| Parameter sweeps and sensitivity studies over albedo, thermal inertia, conductivity, emissivity, roughness | The overrides that reach the solver are listed under *Configuration gotchas*; the ones that do not are named there too |

**Currently under repair — check here before you report a number**

| Use | Why |
|---|---|
| Absolute subsurface temperatures at depth on faces that receive no sunlight | Spin-up in every shipped configuration is far shorter than the deep column's relaxation time (section 1), and on the GPU path the dark-face deep column stops relaxing altogether about 0.36 K short (section 3) |
| Any statistic below about 24 K on the Moon or 28 K on Europa | The solver enforces a hard lower bound at the geothermal equilibrium temperature (section 2) |
| Sensitivity of results to the basal heat flow on a built-in body | `thermal.geothermal_heat_flow` is announced and then discarded on Moon, Europa and Mercury (section 4) |
| Very low conductivity generic bodies (porous or cometary regolith, insulating lags) | A `generic_body.surface` value below 5e-7 is silently replaced by Europa's ice properties (section 5) |
| Ice-table depth and cold-trap volatile budgets on lunar polar meshes | Affected by sections 1, 2 and 3 together; the 0.36 K deep offset alone is about 6 cm of table depth at the measured gradient |

Nothing in sections 1–6 has been shown to affect surface temperature, peak temperature,
illumination or the radiation-exchange gates. Section 7 is the exception to that reassurance, and
applies to every kind of run: a silently ignored key or flag means the model you ran is not the
model you wrote.

---

## 1. Spin-up length is unchecked, and every shipped configuration is short

**What it is.** `thermal.equilibration_orbits` sets how long the solver runs before the output
window opens. Nothing in the engine measures whether that was long enough, nothing warns, and the
result of a short spin-up is not an obviously broken field — it is a plausible one carrying
structure inherited from the initial condition. The surface layer equilibrates in a few cycles;
the column below it relaxes on a much longer timescale, and that is the timescale the setting has
to be judged against.

**Who is affected.** Everyone, on every body and both backends, but by very different amounts:

| Population | Share of a lunar polar mesh | Deep-column error at the shipped 50 lunations | More cycles fix it? |
|---|---|---|---|
| Faces that receive flux during spin-up | 29.7% | **7.2 K median**, tens of kelvin on the worst faces | **Yes** |
| Faces that receive none | 70.3% | 0.36 K, frozen | No — see section 3 |

**Measured.** On a 509,337-face lunar polar terrain mesh, three states of the same run differing
only in `thermal.equilibration_cycles` (50, 100 and 200 integrated lunations — on the pre-2026-08-11
engine, where that meant 2, 4 and 8 passes over a 25-cycle block), comparing the mean of the bottom
five depth nodes (0.85–1.00 m):

- 50 → 100 lunations moves the deep column on forced faces by a median of **6.57 K**, maximum
  **21.03 K**.
- 100 → 200 lunations moves it a further **0.61 K** median.
- Fitting a single relaxation mode to those two gaps gives a time constant of **20.4 lunations**
  and a remaining error of **7.2 K median at 50 lunations, 0.62 K at 100, and about 0.005 K at
  200**. The fitted time constant agrees with an independently measured slowest eigenvalue of
  18–23 lunations for this grid.
- The **surface** node moves by only 1.12 K median over the same 50 → 100 step, and lit-face
  `temperature_max` by a mean of 0.016 K. This is why the model validates against Diviner while
  carrying a several-kelvin deep error.

**The methods trap: an aggregate statistic reports success.** Over the whole mesh the median deep
drift from 50 to 100 lunations is **exactly 0.000000 K**, because the frozen dark 70% dominates the
median while the lit third moves six degrees. The all-face mean is −2.18 K. Judging spin-up by a
single mesh-wide number — or by "the field stopped changing" — will report convergence at exactly
the point where two different failures are present.

**What to do instead.**

1. **Read the integrated cycle count off the banner.** `thermal.equilibration_orbits` is now the
   integrated number of diurnal cycles of real forcing on every solver path, and every run prints
   `Equilibration (configured)` and `Equilibration (integrated)` before it starts. Note what that
   number still excludes: the CPU scalar and SIMD paths run a fixed analytic clear-sky
   pre-conditioner of `NYEARSEQ × round(year/day)` cycles first — 24 on the Moon, 2 on Europa —
   which no key configures and the GPU path does not have. The banner prints its length too.

   *Fixed in the 2026-08-11 equilibration-semantics change (BUGS.md PTM-146).* Before that date
   `equilibration_orbits` was the *block* length and the GPU pass count defaulted to 2, so a GPU run
   integrated `equilibration_orbits × equilibration_cycles` — twice the stated number — while a CPU
   run integrated it once. `equilibration_orbits: 25` bought 50 lunations on the GPU and 25 on the
   CPU. **Every GPU result produced before 2026-08-11 was spun up for twice its config value**, so
   read an older run's provenance that way; the "shipped 50 lunations" measurements in this section
   came from `equilibration_orbits: 25` on that pre-change engine. The shipped GPU configurations
   were rewritten alongside the fix to state the integrated number they had always run (10 became
   20, with `equilibration_cycles: 2`), so their results did not change.

2. **Count e-foldings.** Divide integrated cycles by the deep relaxation time. Measured anchors:
   roughly **15–23 lunations** on the Moon and about **9.5 orbits** on Europa, both for the engine's
   default depth grid; a deeper grid (`thermal.thermal_b`) is slower. Three e-foldings leaves 5% of
   the initial deep error, seven leaves 0.1%. In this repository, of the 77 configurations that set
   `equilibration_orbits`, the 15 lunar thermal ones buy **0.01 to 0.13 e-foldings** and the 55
   Europa ones buy **0.5 to 2.1** — none reaches three. (The four generic-body asteroid
   configurations at 40–50 rotations are a different regime with a much shallower column and have
   not been scored; two further lunar configurations set 0 and are illumination-only, where that is
   correct.) The Europa figure was quoted as **1.1** before 2026-08-11 because it was read off the
   config value of a GPU configuration that was silently integrating twice it — a worked example of
   why this section says to read the banner and not the key.

3. **Run the differential yourself.** Run the same configuration at `N` and `2N` and compare the
   product you intend to report — per population, not as one mesh-wide statistic. If it moves, `N`
   was too short. This is the check tutorial 02 performs for Europa.

4. **Use the conservation identity, which the frozen case cannot fool.** At a converged steady
   state on an unforced face, the conductive flux through *every* interface equals the basal flux:
   `k_j (T_j − T_{j+1}) / dz_j = Q_b`. On the shipped lunar polar state that ratio is about 0.91
   over the top fifteen interfaces, rising to 1.00 at the bottom — a residual that no
   pass-to-pass or cycle-to-cycle comparison detects. The exactly converged unforced profile is
   closed form and costs nothing to evaluate: `T_0 = (Q_b/εσ)^(1/4)`, then
   `T_(j+1) = T_j + Q_b·dz_j/k_j`.

**Cost of running longer.** Extra passes cost extra thermal integration only — illumination is
traced once and the flux array is built once. On the 509,337-face mesh above, four passes (100
integrated lunations) took 3.1 hours of wall time for one chunk. Reaching the ~0.005 K level on
lit faces needs about 200 integrated lunations, roughly four times the thermal cost of the shipped
50. The defaults have deliberately not been changed here; the choice belongs with the run.

**A related modeling choice worth knowing.** Equilibration integrates with **no inter-face
radiation**: indirect flux is added during spin-up only if `thermal.eq_indirect_flux_file` supplies
it from an earlier pass, and it is empty by default. Shadowed faces in a coupled crater run
therefore enter the output window from a deep state equilibrated against direct and primary flux
alone. This is also why 70% of a polar mesh registers as unforced above, which is more faces than
are truly in permanent shadow.

**Status.** Open. There is no convergence diagnostic in the run path.

---

## 2. The model cannot report a temperature below the geothermal floor

**What it is.** At the end of every substep the solver raises **every depth node** of every face to
at least `T_geo = (Q_b/(εσ))^(1/4)`, the temperature at which a surface radiates the basal heat
flux in steady darkness. The bound is correct as a statement about a *converged surface* node; it
is applied instantaneously, to interior nodes where it does not hold, and it is silent — no
counter, no log line.

For the built-in bodies, with their compiled default `Q_b` and emissivity:

| Body | `Q_b` (W/m²) | ε | Floor |
|---|---|---|---|
| Moon | 0.018 | 0.95 | **24.0426 K** |
| Mercury | 0.020 | 0.93 | 24.8160 K |
| Europa | 0.030 | 0.90 | 27.6894 K |

**Who is affected.** Anyone reporting a minimum temperature, a histogram of cold temperatures, or
an area or volume statistic with a threshold near or below the floor — lunar ultracold traps most
directly. All three solver paths enforce it (CPU scalar, CPU SIMD batch, GPU persistent), at five
sites, so switching backends does not avoid it. Europa is unaffected in practice: its floor of
27.7 K sits far below a 60–132 K surface, so the bound never binds.

**Measured.** On a banked 509,337-face, 50-layer lunar state — 25,466,850 nodes — the minimum
temperature anywhere is **24.042612 K**, the float32 value of the analytic floor, and **not one
node lies below it**. Of the 358,294 faces that received no flux during spin-up, **41.3%** have
their surface node sitting exactly on the floor value.

**Practical consequence.** A map value, histogram edge or reported `T_min` sitting at exactly
24.04 K is the bound, not physics. An "ultracold trap" area computed with a threshold below
24.04 K is identically zero by construction, and one just above it is set by where the bound
happens to bind. The bound also erases the conductive step between the surface and the first
interior node, which is part of the deep offset in section 3.

**What to do instead.** For sub-floor statistics there is no configuration workaround today; the
bound is unconditional whenever `Q_b > 0`. Report the floor value alongside any cold statistic so a
reader can see whether it is binding, and treat the interval below `(Q_b/(εσ))^(1/4)` as
unresolved rather than empty.

**Status.** Open. Removing it requires re-blessing every lunar product with columns approaching
24 K, so it has not been changed.

---

## 3. On the GPU path the deep column of an unforced face stops relaxing early

**What it is.** The GPU persistent kernel carries its temperature state in **single precision**
(`float`). At the production substep the per-step change at depth on an unforced column falls
below one unit in the last place of float32 and rounds to zero, so the column stops relaxing while
it is still short of steady state — and then stays there for any number of further cycles.

**This is a GPU-path defect, and the CPU is immune.** The CPU solvers hold the profile in double
precision (`aligned_vector<double>`), where the same per-step change is many orders of magnitude
above the rounding limit. A double-precision transcription of the same kernel, same substep, from
the same banked state keeps converging where the single-precision arm freezes. For a mesh small
enough to run on CPU, `use_gpu: false` avoids this entirely.

**Who is affected.** GPU runs, on the unforced 70.3% population of section 1 — which is where PSR
floors, cold traps and ice-table depths live. Forced faces are dominated by section 1 instead,
which is 20 times larger and *is* curable.

**Measured.** On the banked 200-lunation lunar state, comparing the unforced columns against the
closed-form converged profile for the same grid and basal flux:

- Analytic surface-to-bottom rise **5.8483 K**; measured **5.4896 K** on the 147,873 faces pinned
  at the floor — a **0.359 K** deficit. Across all 358,294 unforced faces the median deficit is
  0.468 K (8.0%).
- It is frozen, not slow: over the 100-lunation gap from 100 to 200 integrated lunations the mean
  deep change on unforced faces is **0.0000 K** and **100.00%** of them are bit-identical over the
  bottom five nodes.
- About a fifth of the deficit is the floor of section 2 erasing the surface-to-first-node step; the
  remainder is the rounding arrest.

**What this is worth in practice.** At the measured gradient of 5.85 K/m, 0.36 K is about **6 cm**
of ice-table depth. It is not a large absolute temperature error. Note also that on this mesh
*every* unforced face has a deep node below 40 K, where absolute water sublimation rates are
negligible, so the large *fractional* sensitivity of sublimation to temperature at these
temperatures multiplies a vanishing flux; do not translate 0.36 K into a large volatile-loss error
without checking the absolute temperature of the faces concerned.

**What to do instead.** Run on the CPU where the mesh size allows. Where it does not, treat
absolute deep temperatures on unforced GPU faces as carrying a floor-side bias of a few tenths of
a kelvin, and prefer differences between runs on the same backend to absolute values.

**Status.** Open, root cause established.

---

## 4. `thermal.geothermal_heat_flow` is discarded on built-in bodies

**What it is.** On a built-in body (Moon, Europa, Mercury) the value is parsed, applied to the
main-scope planet object, and announced —
`Overriding planet geothermal heat flow: …` — and then thrown away, because every mesh solver
rebuilds its own planet from the body name. This is true on **both** backends and at all four
solver entry points, not only on the GPU. The Moon therefore always integrates its compiled
default `Q_b = 0.018 W/m²` (Langseth et al. 1976) whatever the config asks for and whatever the
banner says.

**Who is affected.** Anyone varying the basal heat flow on a built-in body — geothermal
sensitivity studies, deep-column and ice-table work, and anything that depends on the floor of section 2,
whose value is set by `Q_b`. Generic bodies are not affected: `generic_body.surface.Qb` is
forwarded and does reach the solver.

**Measured.** On a global ico3 diurnal run and on a 12,800-face 80°N crater with radiation
coupling, raising `geothermal_heat_flow` from the default to **5.0 W/m²** — 278× — leaves every
`statistics/*.bin` **byte-identical**. A live channel could not have been missed: 5.0 W/m² at
ε = 0.95 implies a floor of 98.2 K.

**What to do instead.** Use `thermal.heat_flow_mask_file`, which seeds the per-face `Q_b` array and
does reach the solver; the same 5.0 W/m² applied that way moves the coldest face by 34 K. On a
generic body, use `generic_body.surface.Qb`. `examples/moon_ico3_qb_falsifier.yaml` is the standing
fixture: it records the number to expect now and the number that must appear once the override is
forwarded.

**Note on the related fix.** The property-override class was repaired for `ks_override`,
`kd_override`, `emissivity_override`, `albedo_override` and `constant_thermal_inertia`, which now
apply on all bodies and abort the run if they cannot be applied. `geothermal_heat_flow` was left
out of that repair. It is the one member of the class still open.

**Status.** Open.

---

## 5. A very small `generic_body.surface` value becomes zero, and the solver substitutes Europa

**What it is.** Generic-body properties are forwarded to the solver through a converter that
formats each value with six decimal places. A value below **5e-7** in its native units therefore
arrives as the string `0.000000`. The receiving code discards anything that is not greater than
zero and keeps what the solver already had — and for a generic body, whose name matches no built-in
entry, that is the **Europa ice placeholder** (`ks = 0.0096`, `kd = 0.05` W/m/K). Nothing warns.
The engine prints *"not in built-in list; using placeholder defaults (generic body config will
override)"* and then does not override.

**Who is affected.** Anyone modeling a very low conductivity surface on a generic body: porous or
underdense cometary and trans-Neptunian regolith, insulating lags, low-thermal-inertia end members
of a sweep. Bodies at ordinary asteroid values are untouched — `ks = 5e-3`, `kd = 0.2` round-trip
exactly at six decimals — which is why this went unnoticed through an entire asteroid campaign.

**Measured.** Three runs of `examples/quickstart_asteroid_diurnal.yaml` differing only in
`generic_body.surface.{ks,kd}`: `ks = kd = 4e-7` and `ks = 0.0096, kd = 0.05` produce
**byte-identical** statistics, while the shipped control at `8.333e-3` produces different output,
so the comparison discriminates. The requested run is a **24,000× conductivity error** and a
**155× thermal-inertia error** (0.69 SI requested, 107 SI delivered): median diurnal amplitude
198.3 K requested against 49.4 K delivered.

**Two further consequences of the same converter.** Every forwarded generic-body value is rounded
to six decimals, so `ks = 0.008333333` is delivered as `0.008333` (4e-5 relative). And
`generic_body.surface.cp_coefficients` is formatted element by element, so a Moon-like coefficient
with an `e-9` term becomes `0.000000` in the list. Where zero is a legal value — `Qb`, `cp0`,
`albedo` — the run proceeds with zero rather than substituting a placeholder.

**What to do instead.** Keep every `generic_body.surface` value at or above 5e-7, and check the
delivered thermal inertia against what you asked for. Where the property has an `*_override` form
(`ks_override`, `kd_override`, `emissivity_override`, `albedo_override`) that channel uses full
precision and is not affected.

**Status.** Open.

---

## 6. CPU and GPU are not interchangeable at the level of a few tenths of a kelvin

Beyond section 3, two differences between the backends are worth knowing before a CPU/GPU
comparison is read as physics. Both are open.

- **The CPU SIMD batch solves the surface boundary condition with a weaker Newton iteration.**
  The batched CPU path runs a fixed five iterations with a relaxation factor and **no convergence
  test**; the CPU scalar path and the GPU kernel use ten iterations with a bracketed seed and a
  step limiter. All three converge to the same root on well-behaved faces; they need not agree to
  the last tenth of a kelvin on cold or sharply forced ones.

- **The CPU floors every layer thickness at 1 mm; the GPU does not.** The CPU profile builder
  applies `dz = max(0.001, z[i+1] − z[i])` and derives its whole conduction stencil from the
  floored values, while the GPU kernel differences the depth grid directly. On lunar and Europa
  grids the floor never binds. It **does** bind on fast-rotating, low-conductivity bodies: for the
  shipped `examples/quickstart_asteroid_diurnal.yaml` (5 h period, thermal inertia 100 SI) the
  diurnal skin depth is 6.31 mm and the target top-layer thickness is 0.631 mm, so the CPU
  integrates a top cell **1.59× thicker** than its own depth grid says; tutorial 03's exercise at
  32 SI reaches 0.202 mm and a factor of 4.9. The two backends are then integrating different
  geometry near the surface, which is where the thermal-inertia signature is set. The size of the
  effect on results has not been measured — this is stated from the code and from arithmetic on
  the shipped configuration.

---

## 7. Unknown configuration keys and unknown command-line flags are accepted in silence

**What it is.** There is no configuration schema. An empty YAML file is a valid config, there are
no required keys, and a misspelled key is not an error — the engine never sees it, and the run
proceeds on the default. The one runtime warner that exists is scoped to the `thermophysical:`
block; the other eight sections have nothing. On the command line the argument loop has no final
`else`, so an unrecognized token is dropped and the run exits 0 with no diagnostic.

**Who is affected.** Anyone driving PTM3D from a script or a sweep, where a typo is invisible and
the run looks successful. `thermal.equilibration_orbit` (singular) runs at the default of 1.0.
`--output-dir` is silently ignored — the real flag is `--output` — and the run writes to the
config's directory instead, which can overwrite an earlier result.

**Measured.** `ptm_thermal_sim --config <cfg> --this-flag-does-not-exist 42` returns rc=0 with zero
warnings in the log. Separately, **18 flags** exist only in a command-line override function that
is compiled and never called, so they parse nowhere and do nothing: `--chi`, `--body-radius`,
`--ephemeris-source`, `--h-parameter-map`, `--property-maps-directory`,
`--use-default-property-map`, `--no-albedo-map`, `--occluder-file`, `--spin-period`,
`--spin-pole-ra`, `--spin-pole-dec`, and the whole `--orbit-{sma,ecc,inc,lan,argp,ma,epoch}`
family. Three were falsified directly: `--orbit-ecc 0.30`, `--spin-period 12.0` and
`--body-radius 5000` each return rc=0 and leave the output byte-identical. None of the 18 appears
in `--help`, so only source-reading or guessing finds them. Note that the set is almost exactly the
generic-body orbit and spin parameters — the knobs an asteroid run reaches for first. Set them in
the `generic_body:` block, which does work.

Across the shipped configurations one validator measured **19 distinct dead keys**, including the
entire `visualization:` block (read by nothing) and a configuration where `use_default_rhos: false`
with `rhos: 900.0` ran at 1100.

**What to do instead.** Check a config before running it:

```python
import yaml
from ptm.engine_config import unknown_keys
print(unknown_keys(yaml.safe_load(open("my_config.yaml"))))
```

This lists every dotted key the engine's parser never reads, derived from the parser source rather
than from documentation. For the command line, prefer the config file, and confirm from the run log
that the value you set is the value that was echoed.

**Status.** Open.

---

### Reference

| Section | Internal identifier |
|---|---|
| Section 1 — spin-up adequacy | PTM-157, PTM-146 |
| Section 2 — geothermal floor | PTM-159 |
| Section 3 — deep-column arrest | PTM-160 |
| Section 4 — geothermal heat flow discarded | PTM-161, PTM-123 |
| Section 5 — generic-body truncation | PTM-201, PTM-60 |
| Section 6 — CPU SIMD surface Newton | PTM-216 |
| Section 7 — unknown keys and flags | PTM-193, PTM-203, PTM-215, PTM-32 |

---

## Configuration gotchas

- **Most of the `thermophysical:` block is not read, and the run now says so.** On a *built-in*
  body (Moon/Europa/Mercury) the parser consumes only `use_default_albedo`,
  `constant_thermal_inertia`, `albedo_override`, `emissivity_override`, `ks_override`,
  `kd_override`, `incidence_angle_albedo`, `albedo_model`, `albedo_hapke_w` and
  `conductivity_model`. Plain-named properties (`ks`, `kd`, `H`, `rhos`, `rhod`, `cp0`, `Qb`,
  `chi`, `albedo`, `emissivity`) and every `use_default_*` flag other than `use_default_albedo`
  are **decorative there** — they are honored only under `generic_body.surface:`. Editing them on
  a named body changes nothing. As of v1.0.3 the engine prints one `[config] WARNING:` line naming
  every such key it found, instead of dropping them in silence; **it does not start honoring
  them**, because doing so would change results for the 130+ shipped configs that carry them.
  There is still no config route to a non-default `H` on a named body except `h_parameter_map`.

- **A malformed config file is now fatal.** Previously a YAML parse failure — a bad scalar, a
  malformed document, a rejected enum value — was downgraded to a `Warning:` and the entire
  simulation ran on built-in defaults (default body, mesh, timesteps, properties **and output
  directory**) exiting 0, so no orchestrator could detect it. `--config` with an unloadable file
  now prints `FATAL:` and returns 1. If you relied on the old behavior, omit `--config`
  deliberately rather than passing a broken one.

- **`constant_thermal_inertia > 0` silently disables the T-dependent conductivity term (`chi`).**
  Selecting a constant thermal inertia forces `chi = 0` (and collapses `kd→ks`, `rhod→rhos`) by
  design, so a `chi` you also set is dropped without warning. To use a real `chi > 0`, set
  `constant_thermal_inertia: 0` and specify `ks/kd/rhos/rhod/cp0` directly.

- **`albedo_override` is nearly inert when a per-face albedo map is loaded.** With
  `use_albedo_map: true` (or any `albedo_map:`), the map's per-face albedo is what the absorbed-solar
  flux uses; `albedo_override` then only sets the equilibration seed and the fallback for faces the
  map does not cover. Measured effect on a global Moon ico5 run: ≤0.51 K. To sweep albedo, disable
  the map (`use_albedo_map: false`) or scale the map itself. `ks_override`, `kd_override` and
  `emissivity_override` are unaffected by this and apply in full.

- **A requested property override that no solver applies now aborts the run.** `ks_override`,
  `kd_override`, `emissivity_override`, `albedo_override` and `constant_thermal_inertia` were
  previously reported in the log and then ignored on built-in bodies (Moon/Europa/Mercury) for every
  mesh run — the sweep measured nothing and said it had worked. They now apply on all bodies, and a
  run that cannot apply a configured override exits non-zero before writing any statistics. Confirm
  from the log line `[Thermo override] applied at solver: …`, which reports the value the **solver**
  used; the earlier `Property override (requested): …` line reports only what was asked for.
  Conductivity/emissivity/constant-TI results produced on built-in bodies before this fix are not
  what their configs claim. `thermal.geothermal_heat_flow` is the one member of this class that was
  **not** repaired — see section 4 above.

- **`thermal.thermal_n` and its `--thermal-n` flag do nothing.** The key is parsed and forwarded,
  and is consumed only by a legacy grid generator that cannot be reached: the adaptive grid is
  selected by a compiled-in constant with no configuration route. The live grid controls are
  `thermal_m` (layers within the upper skin depth) and `thermal_b` (skin depths to the bottom
  layer).

- **The albedo the log reports is the albedo the flux used — and a disagreement is now warned.**
  The GPU thermal path used to rebuild its own `Planet` from the body name, which reinstated the
  body default (Moon 0.1) while the main-scope banner still announced the configured value (e.g.
  0.12). The *physics* always used the per-face albedo array, so the banner — not the result — was
  wrong. If the two scopes ever disagree again, the run now prints
  `[albedo] WARNING: … DISAGREES with the operative …` naming both numbers. Treat that line as a
  configuration error to resolve, not as cosmetic.

## Mesh geometry and units

- **PTM3D meshes are not all in the same length unit, and the mesh file carries no unit.**
  Global icosphere meshes are generated in **kilometers**; DEM/site meshes are imported in
  **meters**. The engine infers the unit from the face-center magnitudes, states its conclusion at
  startup, and — as of v1.0.3 — **refuses to run** on the four cases where the inference is
  ambiguous rather than guessing (a wrong guess is a 1000× error in every SPICE-derived length,
  which is how a planetary-shadow radius once silently disabled every eclipse). If you hit that
  FATAL, rescale the mesh *file* so its face centers are unambiguously meters or kilometers, or
  supply the body radius through the generic-body shape block.

- **`mesh.scale_factor` is not fixed — it is only no longer silent.** It scales the vertices and
  leaves `face_centers`, `face_areas` and `face_normals` at the loader's values, so the mesh ends
  up internally inconsistent (face centers are no longer the averages of their own vertices) while
  the log says "Applied scale factor". The mesh-units guard above refuses such a mesh and names it,
  but the scaling itself is still wrong. No shipped config sets it; rescale the mesh file instead.

## Illumination

- **Near-terminator direct illumination can be non-monotonic.** At grazing solar incidence (~90°)
  the finite-solar-disk model can yield small, non-monotonic direct-flux values on near-terminator
  faces (order ≤1 W/m²). Benign for energy balance, visible only in per-face direct flux at the
  terminator.

- **Residual "crater-center flash" on non-watertight meshes.** Grazing sunlight can thread small
  interior mesh holes and briefly light floor faces. The interior-hole seal
  (`PTM_ILLUM_SEAL_HOLES=1`) closes watertight interior hole loops, but **open and non-manifold
  hole components are left unsealed**. Ensure meshes are watertight for micro-relief floor studies;
  the seal is a mitigation, not a guarantee.

- **The shadow ray starts a fixed ~1 cm off the surface, which can hop a millimeter-scale
  occluder.** On meshes whose relief is finer than that offset — polar micro-relief, sub-meter
  crater floors — the ray can step over a rim that genuinely blocks the Sun and light a face that
  should be in permanent shadow. The default is unchanged (so results stay comparable), and the
  mitigations are **opt-in** environment knobs: `PTM_ILLUM_RISE_EPS=1` with `PTM_ILLUM_RISE_HABS=0`
  replaces the metric offset with an out-of-plane-rise self-relief test, and
  `PTM_ILLUM_TRUE_NORMALS=1` stops the radial-up normal flip fabricating sunlit overhangs. See
  `docs/ENVIRONMENT.md` for the full list and the recommended-ON tags. Relevant to cold-trap floor
  temperatures on fine meshes.

  On a smooth global icosphere `PTM_ILLUM_TRUE_NORMALS` now has no effect at all, and the engine
  says so. Such a mesh carries analytic radial normals, deliberately substituted for the geometric
  winding normals to remove a ~0.5 K pentagon-vertex bias, and it has no overhang faces for the
  knob to correct. Recomputing winding normals there used to undo the pentagon fix silently: on a
  1,280-face ico3 run it changed the average direct solar flux on 1,208 faces (94.4%, up to
  0.58 W/m²) and the peak temperature on 1,216 faces (95.0%, up to 1.86 K), while reporting "0
  overhang faces". The illumination path now declines the override on any mesh whose normals were
  analytically substituted, prints that it declined, and leaves imported and topographic meshes —
  where the winding normal *is* the surface normal — bit-for-bit unchanged. Earlier releases did
  not: a global-icosphere run made with `PTM_ILLUM_TRUE_NORMALS=1` before this change carries the
  pentagon bias and should be repeated.

- **An unrecognized primary body is refused, not defaulted.** The primary-body reflection/IR paths
  only know EARTH, JUPITER, SATURN and MARS. A name outside that set used to take one of three
  different silent fallbacks depending on which code path ran. The run now warns once, naming the
  body and the consequence, instead of quietly emitting a made-up primary IR flux.

## View factors / radiation

- **Deep permanently-shadowed floors remain slightly under-coupled to warm walls.** Even with the
  corrected view-factor operator, radiative coupling into the deepest cold-trap floors is a small
  underestimate (a radiative-equilibrium effect of order ~1–2 K for most floors, less for the
  deepest). Relevant when comparing modeled cold-trap floor temperatures to observations.

- **Do not rebuild the `.vf` from a mesh that changed after the view factors were built.** The
  view-factor builder uses the mesh's raw vertex ordering ("winding"); if the mesh face count or
  winding differs from the one the view factors were built against, radiation exchange is silently
  wrong (or silently does nothing). Reuse the paired mesh + view-factor file together, and confirm
  the load log shows matching dimensions **and Non-zeros > 0** (a nonzero count of stored matrix
  entries).

- **Low-rank (ACA) far-field view factors can carry tiny negative entries.** A physical view factor
  is ≥ 0, but the low-rank far-field approximation does not preserve non-negativity, so near-zero
  true entries reconstruct with small ± sign noise. The net row-sum effect is negligible on smooth
  production meshes (~−3e-5), but can be larger on steep/rough geometry. Clamp reconstructed `F ≥ 0`
  or tighten the ACA tolerance if it matters for your case.

- **The view-factor occlusion test uses meter-scale thresholds — coarse for sub-meter meshes.** The
  visibility test starts rays a fixed ~0.5 m offset and auto-passes pairs closer than ~12.5 cm. On
  sub-meter meshes (face edges ≪ 1 m) most of the near field auto-passes, so genuinely-occluded near
  pairs can still exchange radiation. Affects fine (sub-meter) meshes; the corrected-VF operator
  mitigates the net magnitude.

- **GPU two-band radiosity is limited to meshes with Σ_j F_ij ≤ 1.** The GPU Jacobi kernel does not
  apply the row-sum renormalization, so on meshes whose centroid view-factor row sums exceed 1 the
  solver warns and falls back to the (always-protected) CPU two-band solver. The CPU path is
  unrestricted.

## Output format

- **`statistics/*.bin` files begin with a 4-byte header.** The layout is
  `[int32 n_faces][float32 × n_faces]`. A custom reader must consume the leading `int32` (the
  shipped `scripts/ptm_qa.py` reader does); reading the file as raw `float32` shifts every face by
  one index and mislabels the first value.

- **A `thermal_state_*.bin` checkpoint carries no depth grid.** It stores a magic number, the face
  and layer counts, and the temperatures — so a checkpoint is not self-describing in meters, and
  the depths must come from the configuration that produced it.

## Data files and maps

- **The shipped lunar Bond-albedo FITS was registered 180° off in longitude, and older copies
  still are.** `data/albedo_maps/Moon-BondAlbedo.fits` declared `CRVAL1=0` (column 0 = IAU_MOON
  longitude 0) while its raster was actually laid out −180…180-centered, so the dark nearside maria
  sampled onto the bright farside highlands. This is a **data** defect, not a reader defect — the
  Europa FITS uses the same loader and the same convention and registers correctly, so patching the
  loader would break Europa. The distributed archive now carries the corrected map; a copy fetched
  before that revision must be re-fetched, or re-registered in place with
  `scripts/fix_moon_albedo_lon_registration.py` (requires `astropy`; idempotent — it refuses to roll
  a map that already passes the maria-dark test). Lunar topography was always correct.

- **Per-face albedo `.bin` maps must be regenerated from a corrected FITS.** The
  `Moon_Albedo_res*.bin` files were sampled from the old FITS and additionally sampled −latitude in
  the generator, so they carry both errors; the camera path reads them directly. Regenerate with
  `build/bin/albedo_map_generator <fits> data/albedo_maps 3 9`. The *thermal* loader collapses the
  `.bin` to a uniform mean, so thermal runs were not affected by the `.bin` error; camera renders
  were. Europa's bins were already correct and are unchanged.

## Physical constants

- **Saturn-shine and Mars-shine changed in v1.0.3; this is a correction, not a regression.** The
  four primary-body `(TEMPERATURE, IR_FLUX)` constants are consumed as *emitter-surface* fluxes and
  must therefore satisfy `IR_FLUX = σT⁴`. Three of the four did not. `SATURN_IR_FLUX` was 15.0
  W/m² — almost exactly the *solar* constant at Saturn (1361/9.58² = 14.83), i.e. a solar value
  entered for thermal emission — and is now σ(95 K)⁴ = **4.62 W/m² (÷3.25)**. `MARS_IR_FLUX` was
  70.0 and is now σ(210 K)⁴ = **110.3 W/m² (×1.58)**. Any run with Saturn or Mars as the primary
  will produce different primary-IR heating than a pre-v1.0.3 build. **No published PTM3D result
  is affected**: every campaign to date used Jupiter (flux unchanged) or the Sun alone, and no
  Saturn- or Mars-primary baseline exists to regress against — so this is validated by arithmetic,
  not by comparison. Earth was already consistent (0.1%); Jupiter's *temperature* constant moved
  124.4 → 124.90 K purely to make its pair self-consistent, and is read by nothing. A load-time
  check now warns if any pair drifts more than 5% apart.

- **The Stefan-Boltzmann constant is now a single value (resolved 2026-08-11).** Five values used
  to coexist: `5.67051196e-8` (thermal solver, radiosity, every GPU kernel), `5.670374419e-8`
  (`constants.cpp`, camera, beaming), `5.670374e-8` (the Ingersoll bowl-crater validation gate and
  `vf_mc_flux` — a value no solver used), `5.67e-8` (`thermal_config.sigma`, i.e. what every mesh
  run's surface energy balance actually used) and `5.6704e-8` (the Ojakangas-Stevenson module).
  The spread was 2.4e-5 relative, far below any physical uncertainty, but it meant the surface
  energy balance and the radiosity that feeds it did not conserve energy with each other by
  construction, and an analytic cross-check reproduced the engine to about five significant figures
  rather than exactly. Everything now uses the SI-2019 value `5.670374419e-8`, defined once in
  `src/cpp/constants/stefan_boltzmann.h`; the two OpenCL kernel texts carry their own `#define`
  (OpenCL cannot include a C++ header) and are held to it by a `static_assert` plus a runtime
  check at kernel build. The change shifts temperatures by `dT/T = -6.06e-6`: 0.15 mK at 24 K,
  0.61 mK at 100 K, 2.37 mK at 390 K.

## Workflow / tooling

- **The chunked runner can pick an out-of-date executable.** `run_chunked_simulation.py` may prefer
  a build in `~/dev/heat3d/ptm-build` if that directory exists, even if it is older than your
  in-tree build.
  Pass `--ptm-bin` / `--merge-bin` explicitly to pin the binary you built.

- **The multi-site batch tools depend on an external mesh helper.** `a3gt_v2_orchestrator.py`
  and `run_a3gt_batch.py` expect a `dtm_to_ply_delaunay` binary on a fixed path; provide it, or
  adjust the path, before running a batch.

- **Occluder rings built from raw LOLA PDS `.IMG` need an explicit elevation scale.** Raw polar
  `.IMG` products store scaled integers (`SCALING_FACTOR = 0.5 m/DN`) that GDAL returns *unscaled*,
  so a ring built without `--elev-scale 0.5` comes out exactly 2× too tall and over-shadows grazing
  sun. `scripts/build_lola_ring.py` now aborts rather than assuming 1.0 when the raster is an
  integer type and carries no scale metadata; projected Float32 `.tif` sources are unaffected
  (scale 1.0, byte-identical output).

## GPU backends

- **The HIP and CUDA backends are UNTESTED. Only the OpenCL backend is exercised.** Every PTM3D
  result to date was produced on macOS with `GPU_BACKEND=OPENCL`, whose kernel is the string in
  `thermal_opencl.cpp::getEmbeddedKernelSource()`. `src/cpp/gpu/thermal_kernels.cpp` is compiled
  **only** by the HIP/CUDA paths, so nothing in it has ever executed here and there is no baseline
  to regress it against. In v1.0.3 its `computeK()` was corrected from `k = kc + χ(T/350)³` to the
  Hayne multiplicative form `k = kc·(1 + χ(T/350)³)` that the CPU, `.cl` and embedded-OpenCL paths
  all use — the additive form was ~700× too large at lunar noon and would have made any HIP/CUDA
  run silently and grossly wrong. **This correction is textual and could not be runtime-tested on
  the development machine (no HIP/CUDA hardware).** Anyone bringing up a HIP or CUDA host should
  validate against the CPU solver before trusting results, and should treat the whole file as
  unverified, not just this function.

- **`src/cpp/gpu/thermal_kernels.cl` is a generated mirror, not the kernel that runs.** The OpenCL
  solver compiles the embedded string in `thermal_opencl.cpp::getEmbeddedKernelSource()`; editing
  the `.cl` file by hand changes nothing at runtime and will be overwritten. The `.cl` is
  regenerated verbatim from that string by `scripts/sync_cl_from_embedded.py`, and CI enforces
  `--check`, so it does track the live kernel (including `conductivity_model: martinez2021`) — but
  read it as a mirror, and make kernel edits in `thermal_opencl.cpp`.

- **`src/cpp/gpu/radiosity_kernels.cl` has no such mirror check and has drifted.** The
  radiosity solver likewise compiles an embedded `R"CL_SOURCE(...)"` string
  (`radiosity_opencl.cpp::buildKernels()`), but `scripts/sync_cl_from_embedded.py` and the CI check
  cover only the thermal pair, and the on-disk `radiosity_kernels.cl` is now ~2 kB divergent from
  the string that actually compiles. Unlike the thermal case there is **no** `kernel_path` /
  external-load path in the radiosity solver, so a stale file cannot swap in wrong physics at
  runtime — but the file is copied to the build tree and installed to `share/ptm/kernels`, where a
  reader would reasonably take it for the kernel. Treat it as out of date until the sync tool is
  generalized to both pairs.
