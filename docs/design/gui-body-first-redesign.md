# PTM3D GUI — a body-first redesign

Status: **design investigation, awaiting Paul.** No GUI code was changed. Written 2026-08-26
against `main` at `0e6f6ff` (PR #133, the plain-language merge, on top of Stage 1 `#131` and
Stage 2 `#132`). Every number below was measured on that tree; §1 says with what, and the
measurement script is committed as `scripts/gui_option_usage_audit.py` so the numbers can be
re-derived rather than believed.

The brief: the GUI offers far too many options, most of which a GUI user should not touch. Replace
the option-first surface with a **body-first** one — large buttons for Moon, Europa, Mercury and
Asteroids, and a per-body option set that is appropriate to that body.

This document is the evidence for that change, a per-body surface design, an architecture that
keeps the existing panel reachable, text mockups, a staged estimate, and §10, the decisions that
need Paul before anything is built.

---

## 0. The three findings that shaped the design

1. **115 of the engine's 184 configuration keys never take a second value anywhere in the 78
   shipped run configurations.** Two thirds of the surface is, in practice, a constant. §2.
2. **The body dropdown offers seven built-in bodies and the engine implements three.** Selecting
   Mars, Ceres, Titan or Enceladus produces a run that exits 0, writes all sixteen statistics
   binaries, and is entirely fictitious: Europa's radius, Europa's albedo, Europa's metakernel,
   and **0.000000000 W/m² of solar flux on every face at every timestep**. Measured, four arms.
   §3. A body-first landing screen removes this class by construction — a body without engine
   backing cannot have a button.
3. **Mercury runs today, and its day is wrong by a factor of three.** The engine sets
   `props.day = 58.646` days (`planetary_properties.h:185`) and the run banner prints it as
   *"Synodic day (solar day): 58.646 Earth days"*. Mercury's solar day is 175.94 days — the 3:2
   spin-orbit resonance means three sidereal rotations per two orbits. `thermal.equilibration_orbits`
   counts cycles of that wrong length. §5.3.

---

## 1. Method

### 1.1 What was measured, and how to rerun it

| Question | Instrument | Result |
|---|---|---|
| Which engine keys does a body actually vary? | `scripts/gui_option_usage_audit.py`, over `configs/` + `examples/`, every config completed from `ptm.engine_defaults.ENGINE_DEFAULTS` first | §2 tables |
| How many controls does the panel show today? | Live `ConfigurationPanel` offscreen, walking `tab.widgets` and testing each widget's ancestry against `tab.advanced_sections` | **145 controls**, 71 Basic / 74 Advanced, 11 sections |
| What does the engine do with an unbacked body name? | Four real runs, `simulation.body` = Mars / Ceres / Titan / Enceladus, ico2, 4 steps, CPU, no `generic_body:` | rc 0, zero flux, Europa properties. §3 |
| Is Mercury runnable, or nominal? | Real run on the shipped `data/kernels/mercury.tm` | rc 0, max flux 6298.97 W/m², T_max 577.05 K. §5.3 |
| Which mesh gets a per-face albedo map? | `ptm_thermal_sim.cpp:2280` (`is_custom_mesh` branch) and `:2408`/`:2471` (per-face binary, size must equal the face count) | §5.1.2 |
| Where does the beaming rule live? | `gui/gui_core/facet_scale.py`, `docs/radiation-exchange.md` §8.4 | §5.1.3 |
| Does anything check SPK coverage? | `grep -rn 'spkcov\|spkobj'` over `src/cpp/`, `src/python/`, `gui/`, `scripts/` | **zero hits anywhere** |

Reproduce the audit:

```bash
python3 scripts/gui_option_usage_audit.py .          # the tables in section 2
python3 scripts/gui_option_usage_audit.py . --keys   # with the key lists
```

The four unbacked-body runs and the Mercury run are four invocations of
`build/bin/ptm_thermal_sim --config <probe>.yaml` on ico2 meshes with four timesteps; each takes
under ten seconds and needs only the ~44 MB core SPICE set.

### 1.2 Two rules that make the counts mean what they say

**Every config is completed from the engine's own default table before comparison.** Omitting a
key and writing its default are the same run, so counting "this file does not mention the key" as
variation inflates every cluster. Measured: it inflates Moon-local from 28 varying science keys to
**85**.

**Bookkeeping keys are counted separately.** `output.directory` differs between any two runs by
construction, `radiation.view_factor_file` is a cache derived from the mesh, and `thermal.threads`
is a property of the machine. None is a scientific choice, and every surface must expose them
whatever the body. The audit script's `BOOKKEEPING` set names all fifteen.

`examples/validation/` holds five configurations whose entire purpose is to differ in
`radiation.use_hmatrix`, `thermal.use_gpu` and friends. They are reported separately; sizing a
user-facing surface against them would be sizing it against a regression harness.

### 1.3 Effort scale

**S** ≈ under a day. **M** ≈ one to three days. **L** ≈ a week or more, or needs a decision first.

---

## 2. The empirical case

### 2.1 The headline

Of the **184 keys** the engine's parser reads, across the **78 runnable configurations** shipped in
`configs/` and `examples/` (excluding the five validation arms):

| | keys |
|---|---|
| Take a second value somewhere in the corpus | **69** |
| Never take a second value anywhere | **115** |

Two thirds of the configuration surface is a constant in every run this project has ever shipped.
That is the measurement the redesign rests on.

### 2.2 Per body cluster

A cluster is `simulation.body` crossed with whether the mesh is imported. The `local` split is the
engine's own, and it is the mesh **file** alone:

```cpp
thermal_config.custom_properties["is_custom_mesh"] = (mesh_file.empty()) ? "0" : "1";
                                                        // ptm_thermal_sim.cpp:17036
```

`mesh.use_topographic` deliberately does not count. A topographic run is still the procedural
icosphere at `mesh.resolution` (`:15115`) with LOLA relief applied to its vertices, so it keeps the
`20 * 4^level` face count and the per-face albedo binary still matches. Splitting on it instead put
`examples/moon_global_topo_res5.yaml` — whose own header calls it *"the GUI's default global-Moon
setup"* — in the local cluster, one key away from its documented flat twin, and inflated the
Moon-local headline. Caught in review; the numbers below are the corrected ones.

| cluster | configs | keys that vary | of those, science | of those, bookkeeping | constant |
|---|---|---|---|---|---|
| Moon-global (icosphere) | 12 | 18 | **17** | 1 | 166 |
| Moon-local (imported mesh) | 7 | 34 | **28** | 6 | 150 |
| Europa-global (icosphere) | 10 | 17 | **15** | 2 | 167 |
| Europa-local (Sutton DEM) | 45 | 4 | **2** | 2 | 180 |
| Mercury | **0** | — | — | — | — |
| Asteroid / generic body | 4 | 40 | **36** | 4 | 144 |

Read the Europa-local row again. **Forty-five shipped configurations — nine latitudes crossed with
five Sutton DEMs — differ in exactly two scientific keys**: `mesh.import_file` and
`mesh.latitude_override`. One hundred and eighty of the 184 keys are identical in all forty-five
files. `examples/python/sweep_replaces_45_configs.py` already proves this key-by-key against the
shipped files, so it is not an inference.

### 2.3 What the bodies share, and what they do not

Unioning each body's global and local clusters:

| body | configs | science keys that vary |
|---|---|---|
| Moon | 19 | 32 |
| Europa | 55 | 18 |
| Mercury | 0 | — |
| Asteroid | 4 | 36 |

* Moon and Europa share **16** varying keys (epoch, duration, timesteps, mesh resolution, albedo
  source, equilibration, GPU, radiation on/off and its coupling threshold, VTK mode).
* **16** vary on the Moon and never on Europa — the property map, the topographic directory, the
  H-matrix switch, the geothermal heat flow, `thermal_b`, the shadow-fraction floor.
* **24** vary only on asteroids, and all twenty-four are `generic_body.*`: the orbit, the spin, the
  shape and the surface properties. Defining the body *is* the asteroid workflow.
* **Zero keys vary in every cluster.** There is no universal core. That is the structural argument
  for per-body surfaces rather than one surface with per-body defaults.

### 2.4 Against what the panel shows today

Measured offscreen on `0e6f6ff`:

| section | controls | Basic | Advanced |
|---|---|---|---|
| Start (task cards) | 0 | 0 | 0 |
| Simulation | 4 | 4 | 0 |
| Mesh | 8 | 5 | 3 |
| Properties | 22 | 18 | 4 |
| Thermal | 39 | 7 | 32 |
| Output | 7 | 7 | 0 |
| Body | 21 | 21 | 0 |
| Illumination | 10 | 3 | 7 |
| Radiation | 28 | 6 | 22 |
| GPU | 6 | 0 | 6 |
| Engine Flags | 0 | 0 | 0 |
| **total** | **145** | **71** | **74** |

The Basic tier is 71 controls across nine sections. The corpus says a whole-Moon run is a
17-key decision and a Europa DEM site is a 2-key decision. **The gap between 71 and 17 is the
problem this redesign exists to close**, and Stage 2's field-level tiering — which was the right
move — cannot close it, because a tier is a property of a field and the right answer is a property
of the *body*.

---

## 3. The motivating defect: four buttons the engine cannot honor

`gui/gui_panels/config_panel.py:701`:

```python
BUILTIN_BODIES = ["Moon", "Europa", "Mercury", "Enceladus", "Titan", "Mars", "Ceres"]
```

`planetary_properties.h:79-92` implements three of them and resolves everything else to Europa,
with one `std::cerr` note and no warning, no counter and no nonzero exit.

Four runs were made, identical but for `simulation.body`, ico2 (320 faces), four timesteps, CPU,
radiation off, no `generic_body:` block. All four behaved identically:

| observable | measured, all four of Mars / Ceres / Titan / Enceladus |
|---|---|
| exit code | **0** |
| metakernel loaded | `data/kernels/europa.tm` |
| `R_body` | **1560.8 km** (Europa; Mars is 3389.5 km) |
| banner "Synodic day (solar day)" | **3.55 Earth days** (Europa) |
| default albedo | **0.5** (Europa) |
| average and maximum total flux | **0.000000000 W/m²** |
| `temperature_{min,avg,max}.bin` | **27.6896 K on all 320 faces**, identical |
| `effective_albedo.bin` | all NaN (no face was ever lit) |
| statistics binaries written | 16 |
| diagnostic emitted | one line: `Note: Body 'Mars' not in built-in list; using placeholder defaults (generic body config will override)` |

The `[solar-const] WARNING: only 0/4 timesteps have a valid sun distance` line does appear, but it
is phrased as a solar-constant caveat, not as "this run has no illumination."

Meanwhile the GUI, on selecting Mars, shows the Properties tab populated with **Mars** values
(`config_panel.py:3060-3067`: albedo 0.25, ks 0.02, kd 2.0, rhos 1500) and a banner reading
*"Built-in body (Mars) uses SPICE ephemeris. Fields below are for reference only,"* then disables
the generic-body fields (`:4171-4180`) — so the one route that would make Mars real is closed from
that page.

`docs/configuration.md:674-677` documents the fallback correctly. That is not the issue. The issue
is that the GUI presents it as a first-class choice with a body-specific property panel, and the
run that follows is silently empty.

**A body-first landing screen closes this class by construction.** A body gets a button when the
engine can run it; there is no dropdown row for a body it cannot. §6.4 proposes the generated
capabilities table that makes the two lists provably the same list.

---

## 4. Design principles

1. **The button is the decision.** Choosing a body picks the properties, the metakernel, the
   datasets, the sensible epoch and duration, and which physics switches are even offered. The user
   confirms and adjusts; they do not assemble.
2. **Never more than ~15 visible controls before Advanced.** The corpus supports it for three of
   the four bodies outright; §5.4 says plainly where it does not.
3. **A default that came from the body is labeled as such.** "Thermal inertia: 55 SI (Moon
   default)" beats a bare number, and it is what lets a user know when they have left the beaten
   path.
4. **Nothing is hidden that is set to a non-default value.** If an opened file sets a key the card
   does not show, the card says so and offers the full panel. Silently hiding a live setting is the
   DP-1 class of defect that Stage 1 spent fourteen items removing.
5. **The full panel is never removed.** It is one click away from every card and edits the same
   `ConfigManager`.
6. **Every automatic choice is announced and reversible.** Auto-selecting an albedo map is an edit
   to the user's configuration; Stage 2 already learned that the hard way with the
   `use_albedo_map` fallback on open (`GUI-ALBEDO-FALLBACK-EDIT`).

---

## 5. The four bodies

### 5.1 Moon

The best-supported body: 19 shipped configurations, both global and local, the albedo map, the
per-face property map, the topographic catalog, the beaming calibration, the Diviner
cross-validation, and every tutorial.

#### 5.1.1 The card

A segmented control at the top picks the mode, because the corpus is bimodal and the two modes have
almost disjoint option sets.

**Mode A — the whole Moon** (11 visible controls; the cluster varies 17 science keys, folded into
eleven controls by the smart defaults below):

| # | control | writes | default and its source |
|---|---|---|---|
| 1 | Start date | `simulation.start_time` | today, rounded to midnight UTC |
| 2 | How long | `simulation.duration_days` | 1 synodic period = 29.530588 d, from `planetary_properties.h:145` |
| 3 | Output samples | `simulation.timesteps` | 144 (~5 h cadence) |
| 4 | Detail | `mesh.resolution` | 5 (20480 faces), labeled with face count and a measured runtime estimate |
| 5 | Real topography | `mesh.use_topographic`, `mesh.topo_directory` | on when `data/topo/moon/topo_moon_ico<N>.bin` is on disk; the engine hard-errors rather than substituting a smooth sphere (`ptm_thermal_sim.cpp:15128`), so the card must check first |
| 6 | Surface albedo | `illumination.use_albedo_map`, `illumination.albedo_maps_directory` | **auto**, see §5.1.2 |
| 7 | Angular albedo | `thermophysical.incidence_angle_albedo`, `thermophysical.albedo_model` | Hayne (2017); Hapke is opt-in, per Paul's ruling recorded at `config_panel.py:3050-3053` |
| 8 | Per-face properties | `thermal.use_default_property_map` | **auto**: on when a bundled map matches the mesh |
| 9 | Equilibration | `thermal.equilibration_orbits` | 50 cycles (Paul's standing ruling) |
| 10 | Run on GPU | `thermal.use_gpu` | off |
| 11 | Output folder | `output.directory` | — |

Controls 5 to 8 are why this card is eleven and not seventeen: the topography flag, the two albedo
keys, the property-map flag and the angular-albedo pair are five of the cluster's seventeen varying
keys, and each pair collapses into one decision with a checked precondition.

**Mode B — a place on the Moon** (15 visible, exactly at the target; the cluster varies 28 science
keys, the rest becoming defaults or living behind Advanced). Drops controls 4 and 5, adds a mesh
group and a radiation group:

| # | control | writes |
|---|---|---|
| 4a | Terrain mesh | `mesh.import_file` |
| 4b | Occluder ring | `mesh.occluder_file` |
| 4c | Where it sits | `mesh.latitude_override`, `mesh.longitude_override` |
| 12 | Self-heating between facets | `radiation.enabled` (+ `radiation.coupling_mode: bounded`) |
| 13 | Sub-facet roughness | `radiation.beaming`, `radiation.rms_slope_deg` — **gated**, see §5.1.3 |
| 14 | Reuse view factors | `radiation.view_factor_file` |

Control 5 (real topography) drops out in this mode: it is a property of the procedural icosphere
path and none of the seven imported-mesh configurations set it.

Everything else the two clusters vary — `thermal.thermal_b`, `radiation.use_hmatrix`,
`radiation.min_shadow_fraction`, `radiation.max_coupling_iterations`,
`thermal.equilibration_samples_per_diurnal_cycle`, `thermal.gpu_devices`, `output.format`,
`output.include_primary_ir` — stays exactly where Stage 2 put it, behind the Advanced disclosure
on the full panel. They vary in the corpus because performance work and validation arms varied
them, not because a science question required it.

#### 5.1.2 The auto-albedo rule, stated exactly

The rule is not a GUI invention. The engine already branches on it and the card must mirror the
engine rather than guess:

```
per-face albedo binary applies  <=>  mesh.import_file is empty

    engine:  thermal_config.custom_properties["is_custom_mesh"]
                 = (mesh_file.empty()) ? "0" : "1";       ptm_thermal_sim.cpp:17036
             read back at :2245 and :7528, and it selects the albedo branch at
             :2280 (custom -> FITS) versus the standard per-face binary path.
```

**`mesh.use_topographic` is not part of the rule.** A topographic run keeps the procedural
icosphere's `20 * 4^level` face count, so the per-face binary still matches; the engine agrees,
because `is_custom_mesh` never sees that key. Getting this wrong is easy — this document's own
first draft did, and §2.2 records the correction.

* **Global icosphere (imported mesh empty).** The engine loads the per-face binary
  `<Body>_Albedo_res<mesh.resolution>.bin` (`ptm_thermal_sim.cpp:2408`) and **requires
  `loaded_albedos.size() == num_faces`** (`:2471`). The card offers *Built-in map* as the default,
  shows the resolved path, and shows on-disk status through the existing
  `gui_core.datasets.albedo_status()`.
* **Imported mesh.** The per-face binary cannot apply: the face count will not match
  the icosphere it was built for, and the engine prints one **stdout** line
  (`Binary albedo file size mismatch: expected N faces, got M values`) before falling back to a
  uniform albedo. So the card offers, in this order:
  1. a latitude/longitude FITS map `Moon-0.5deg-BondAlbedo.fits`, which the engine *does*
     interpolate onto an arbitrary mesh (`:2329`, the `is_custom_mesh` branch), when the file is on
     disk; otherwise
  2. **a single constant albedo**, which is the default, with the value shown and its source named.

  It must **not** offer "Built-in albedo map". Today it does: `datasets.albedo_status(body,
  resolution, root)` (`gui/gui_core/datasets.py:169`) keys on the body and the resolution alone and
  has no notion of `mesh.import_file`, so a lunar crater configuration is told a per-face map is
  available when the engine will discard it. That is a promise the engine cannot keep, and closing
  it is item M-2 in §8.

The same rule, with `Europa` substituted, drives the Europa card. The bodies that ship albedo maps
are exactly `{moon, europa}` (`datasets.py:47`); Mercury is not one of them.

#### 5.1.3 The beaming rule

Paul's standing ruling (2026-08-25): infrared beaming parameterizes sub-facet roughness at the
thermal-isolation scale of roughly a millimeter to a centimeter, so it belongs **on** when facets
are far larger than that and **off** when the mesh already resolves it. The criterion is in
`docs/radiation-exchange.md` §8 and implemented GUI-side in `gui/gui_core/facet_scale.py`:

| fifth-percentile facet scale | policy |
|---|---|
| below 0.05 m | **refuse** — the mesh resolves the roughness; beaming would double-count it |
| 0.05 m to 1 m | allow, with the measured number acknowledged |
| 1 m and above | allow |
| body other than the Moon | refuse — `w = 32.5 deg` is a lunar regolith calibration |

The gate is the fifth percentile, not the mean, because A3GT v2 meshes are genuinely
multi-resolution and the mean fails open: measured, the A3GT core reads 3.48 m as a mean and
0.708 m at p05.

**There is still no engine-side guard** (`ENG-BEAM-SCALE-1`). The card is the only thing between a
CLI-adjacent user and double-counted roughness, which is an argument for §6.4's capabilities table
carrying the rule where both the GUI and the engine can read it.

#### 5.1.4 Standard lunar thermophysics, and the one honest caveat

On a built-in body the engine reads only `use_default_albedo`, `constant_thermal_inertia`,
`incidence_angle_albedo`, `albedo_model`, `albedo_hapke_w`, `conductivity_model` and the five
`*_override` keys. The plainly-named properties in `thermophysical:` are read by nothing and the
engine says so once, per run (PTM-67, `config_parser.cpp:155-204`).

So the Moon card shows the standard property set as **labeled read-only values** — albedo 0.12,
emissivity 0.95, `ks` 7.4e-4, `kd` 3.4e-3, `rhos` 1100, `rhod` 1800, `H` 0.06 m, `Qb` 0.018 W/m²
— with a single "Customize" disclosure that reveals **exactly the five keys that have an override
path** (albedo, emissivity, `ks`, `kd`, `H`). It must not offer `rhos`, `rhod`, `cp0`, `Qb` or
`chi` as editable on the Moon; Stage 1 already disabled those rows, and the card should not
reintroduce them.

### 5.2 Europa

Fifty-five shipped configurations, and the tightest option set in the corpus.

#### 5.2.1 The card

**Mode A — the whole Europa** (9 visible; cluster varies 15 science keys):
start date, duration (default one synodic period = 3.551181 d, synchronous rotation), output
samples, detail (`mesh.resolution`, default 6), albedo source (auto, §5.1.2, using
`Europa_Albedo_res<N>.bin`), equilibration cycles (default 20, the value both shipped arms use),
GPU, integration window (`thermal.integration_days`), output folder.

**Mode B — a Sutton DEM site** (5 visible; cluster varies **2** science keys):

| # | control | writes | note |
|---|---|---|---|
| 1 | Terrain | `mesh.import_file` | a registry combo, §5.2.2 |
| 2 | Latitude | `mesh.latitude_override` | the nine shipped values: 0, 15, 30, 45, 60, 75, 80, 85, and the pole |
| 3 | View factors | `radiation.view_factor_file` | derived from the mesh path, shown, editable |
| 4 | Self-heating | `radiation.enabled` | — |
| 5 | Output folder | `output.directory` | — |

Because the cluster is a clean Cartesian product, this card should offer **"run the grid"**: a
sweep over the terrain list crossed with the latitude list, producing the same runs as the
forty-five files. The machinery exists (`ptm.Sweep`, `src/python/ptm/sweep.py:517`) and
`examples/python/sweep_replaces_45_configs.py` already checks the equivalence point by point. This
is the strongest single argument that a body card can be *more* capable than the panel it replaces,
not less.

#### 5.2.2 The Sutton DEM registry

The five DEMs are named only inside the forty-five configurations:

| terrain | file | posting |
|---|---|---|
| Agenor | `data/external/sutton_DEMs/Agenor_DEM_750m_SOCET.ply` | 750 m |
| Cilix | `data/external/sutton_DEMs/Cilix_DEM_350m_SOCET.ply` | 350 m |
| Pwyll | `data/external/sutton_DEMs/Pwyll_DEM_750m_SOCET.ply` | 750 m |
| Rhadamanthys | `data/external/sutton_DEMs/Rhadamanthys_DEM_450m_SOCET.ply` | 450 m |
| Ridges | `data/external/sutton_DEMs/Ridges_DEM_60m_SOCET.ply` | 60 m |

Two rules for the registry, both learned from Stage 2:

* **Derive it from the shipped configurations, not from a hand-typed table.** A table someone
  retypes drifts from the files; the audit script's own `collect()` shows how cheap the derivation
  is.
* **Show on-disk status, never assume presence.** These DEMs are not distributed with the
  repository. The registry entry must carry the same three states `gui_core.datasets` already uses:
  present, missing, and *dataless* (a OneDrive cloud placeholder that exists but has no bytes) —
  the third state is the one that produces the most confusing failures on this machine.

#### 5.2.3 Ice properties

Europa's built-in set is porous ice: albedo 0.5, emissivity 0.9, `ks` 0.0096, `kd` 0.05,
`rhos` 500, `rhod` 700, `H` 0.07 m, `Qb` 0.03 W/m², with a linear heat capacity
(`cpCoeff = {7.49}`), giving a thermal inertia near 60 SI at 100 K. Same presentation rule as the
Moon: labeled read-only, with a Customize disclosure exposing only the five overridable keys.

The card should show the derived **thermal inertia** alongside the conductivity, because that is
the number the literature and the observations speak in.

#### 5.2.4 The caveat the card must carry

Europa has a permanent primary-shine floor: Jupiter-shine means its night side never sits at
exactly zero absorbed flux. That is precisely what made every Europa GPU radiation-off run before
`233e06c` exposed to the surface-Newton under-convergence collapse (PTM-115): measured max
`dT = +12.36 K`, per-face `T_min` off by up to 10.90 K, at latitudes 10-36 degrees, with plausible
65-90 K minima that no range screen catches. The Moon was safe from the same defect only because
its radiation-off nights are exactly zero.

Practical consequence for the card: **print the engine build and its provenance beside the results**
and note that archived Europa products predating that fix are in re-run scope. The `R < 5`
predicate from the ledger is a triage screen for archived output — it can prove a face *affected*
and never proves one *clean* — so it belongs in the results browser, not as a configuration option.

Beaming is refused on Europa regardless of facet scale (§5.1.3).

### 5.3 Mercury — runnable, undefended, and wrong about its own day

This section is the audit the brief asked for, and the answer is that Mercury is thin in a specific
and fixable way rather than absent.

**What exists.**

* A full built-in property set: `planetary_properties.h:181-212` — albedo 0.07, emissivity 0.93,
  `Qb` 0.02, `ks` 0.002, `kd` 2.0, `rhos` 1800, `rhod` 2200, `H` 0.1 m, `cp0` 800,
  `cpCoeff {0.25, 0}`, `S` 9126.6 W/m², radius 2439.7 km.
* A shipped metakernel, `data/kernels/mercury.tm`, loading `lsk/naif0012.tls`, `pck/pck00011.tpc`
  and `spk/de440s.bsp` — **the same ~44 MB core set the Moon needs**, no extra download.
* `PlanetaryBody::MERCURY` wired through the solver (`ptm_thermal_sim.cpp:15569`,
  `thermal_model.cpp:430`).
* A GUI property preset, including Paul's ruling that Mercury defaults to Hapke while the Moon
  defaults to Hayne (`config_panel.py:3040-3046`).
* DEM and albedo-map *generators* accept `Mercury` (`dem_mesh_generator.cpp:637`,
  `albedo_map_generator.cpp:686`).

**It runs.** Measured, ico2 (320 faces), four timesteps, CPU, radiation off:

```
Loaded SPICE meta-kernel: data/kernels/mercury.tm
[mesh-units] ... R_body = 2439.7 km
Default albedo: 0.07     Using incidence-angle-dependent albedo (Hayne et al. 2017 formula)
Average total flux: 3182.988552459 W/m^2    Maximum: 6298.965531552 W/m^2
temperature_max: min 24.816 K, max 577.054 K      rc = 0
```

**What does not exist.**

* **Zero shipped configurations.** No quickstart, no tutorial, no validation arm, no notebook.
  Mercury is the only built-in body with no example to copy.
* **No datasets.** `ALBEDO_BODIES = {"moon", "europa"}` and `TOPO_BODIES = {"moon"}`
  (`gui/gui_core/datasets.py:46-47`). No albedo map, no per-face property map, no DEM catalog
  entry. Every Mercury run is uniform-albedo on an icosphere.
* **No beaming calibration** — `w = 32.5 deg` is lunar.
* **No validation of any kind** against MESSENGER or radar data in this repository.

**Two defects the card would have to disclose.**

1. **The day is wrong by a factor of three.** `props.day = 58.646 * 24 * 3600`
   (`planetary_properties.h:185`) is the *sidereal* rotation period, and the run banner prints it as
   `Synodic day (solar day): 58.646 Earth days`. Mercury's solar day is 175.94 days — the 3:2
   spin-orbit resonance gives three rotations per two orbits. `thermal.equilibration_orbits` counts
   "day/night cycles" of that wrong length, so a request for N cycles delivers N/3 real diurnal
   cycles. The GUI's own table has the right number and a comment claiming it matches the engine:
   `"Mercury": 175.942, # Solar day (2/3 orbital period)` under the header *"These are
   sunrise-to-sunrise periods matching the C++ thermal model (thermo.day)"*
   (`gui/gui_tooltips.py:1755-1758`). For the Moon (29.530588) and Europa (3.55) the claim holds;
   Mercury is the sole outlier.
2. **The temperature ceiling is set for icy bodies.**
   `const double MAX_PHYSICAL_TEMP_K = 500.0; // Maximum physically reasonable temperature for icy
   bodies (K)` (`thermal_model.h:167`). The engine itself warns on a Mercury run:
   *"68 of 320 faces have a cycle-mean radiative-equilibrium seed above 500.0 K ... the CLAMP will
   truncate the solution. Results on those faces are not trustworthy."* Mercury's subsolar
   temperature is roughly 700 K, so this is not an edge case; it is the sub-solar point.
   Complicating it: the run reported `temperature_max = 577.054 K`, above the clamp the warning
   names, on the `CPU SIMD batch (runThermalModelsForAllFacesBatchSoA)` path. Either the warning
   over-claims for that path or the clamp is applied inconsistently across solver flavors. **Root
   cause not established here** — it is filed as an owed row, not diagnosed.

**Recommendation.** Mercury should be a button, and the button should be honest. Concretely: ship a
`examples/quickstart_mercury_diurnal.yaml` (it costs nothing — the kernels are already there), fix
`props.day` to 175.94 days, and either raise the ceiling for rocky bodies or make the card refuse
to start a Mercury run whose seed exceeds it. Until the ceiling is settled, the card carries a
prominent, non-dismissible banner naming the sub-solar limitation. Whether that is enough for
first-class status is **DP-3**.

### 5.4 Asteroids

The asteroid workflow is different in kind: the other three bodies come with an identity, and here
the user supplies one. Twenty-four of the thirty-six varying keys are `generic_body.*`.

This is also the body where the ≤15 target does not hold honestly, and pretending otherwise would
be the wrong kind of simplification. The card is therefore **two panes**: *define the body* (a form
whose size is set by physics, roughly 17 fields) and *set up the run* (about 10 controls, the same
shape as the other cards). The body definition is done once and cached; the run pane is what the
user returns to.

#### 5.4.1 Define the body

| group | fields | notes |
|---|---|---|
| Identity | `generic_body.name`, `generic_body.naif_id` | a HORIZONS search fills both |
| Ephemeris | `generic_body.ephemeris_source` (`spice` / `keplerian`), `generic_body.spice_metakernel` | **§5.4.2** |
| Orbit | `semi_major_axis_au`, `eccentricity`, `inclination_deg`, `lon_ascending_node_deg`, `arg_perihelion_deg`, `mean_anomaly_deg`, `epoch` | shown only for `keplerian` |
| Spin | `period_hours`, `pole_ra_deg`, `pole_dec_deg`, `prime_meridian_deg` | with the derived obliquity, which the panel already computes |
| Shape | `shape.type` (`spherical` / `ellipsoid` / mesh), `radius_km` or `semi_axes_km`, `shape.mesh_file`, `mesh_scale` | **§5.4.3** |
| Surface | `albedo`, `emissivity`, `ks`, `kd`, `rhos`, `rhod`, `H`, `cp0`, `Qb`, `chi` | behind a taxonomy preset, **§5.4.4** |

#### 5.4.2 The SPK coverage check — the PTM-300/301 lesson, in the interface

**There is no coverage check anywhere in this repository.** `grep -rn 'spkcov\|spkobj'` over
`src/cpp/`, `src/python/`, `gui/` and `scripts/` returns nothing.

What that cost: the committed Justitia SPK covers **2034-10-12 to 2034-11-11**, exactly 30.0 days.
Every SPICE-mode Justitia configuration targets **2035-08-24 to 2036-02-13**, the earliest start
about 318 days past the end of coverage. So **every "SPICE" Justitia run ever made was 100 percent
Keplerian** — and nothing said so, because `getSunPosition` catches the `spkpos_c` failure, calls
`reset_c()` to clear the SPICE error, and returns a Keplerian position per epoch with zero logging
(`illumination_model.cpp:2793-2818`). Fourteen of fifteen configurations completed rc 0 with
non-degenerate temperatures and `ephemeris_source: spice` in their provenance. The only related
output is a startup banner reading *"Keplerian fallback: enabled"*, which states availability and
never engagement.

The body card must make this impossible to miss:

```
  Ephemeris   ( ) Keplerian elements  (•) SPICE kernel

  Kernel      data/horizons_cache/bodies/269_Justitia_(A887_SA)/ephemeris.spk
  Coverage    2034-10-12 23:58 → 2034-11-11 23:58 UTC   (30.0 days)

  ⚠  YOUR RUN WINDOW IS OUTSIDE THIS SPK.
     Requested:  2035-08-24 00:00 → 2036-02-13 00:00 UTC
     Equilibration reaches back to 2035-06-11, also outside.
     Every epoch of this run would be served by the Keplerian fallback.

     [ Fetch a wider SPK ]   [ Switch to Keplerian deliberately ]   [ Run anyway ]
```

Three requirements behind that mockup:

1. **The check covers the equilibration window, not just the output window.** Equilibration runs
   before `simulation.start_time` and is where a partially covered SPK silently degrades.
2. **Keplerian is an explicit choice with its own radio button**, never a fallback the user
   discovers afterward. Picking it is a decision recorded in the configuration.
3. **"Run anyway" is allowed but disclosed**, and the choice should be written into the run's
   provenance so the output can be triaged later. The engine-side half of this — count the fallback
   engagements and print one summary line — remains PTM-301's remedy and is not something the GUI
   can substitute for.

`spiceypy` is already a dependency of the tooling that measured the Justitia coverage, so
`spkcov` plus `spkobj` is the implementation; parsing the SPK comment area is the fallback when the
binding is unavailable.

#### 5.4.3 DAMIT shape import, and the gitignore trap

`src/python/damit/` already provides `damit_client.py` and `shape_converter.py`, and the HORIZONS
panel can fetch a shape model.

The trap: **`.gitignore:14` is a bare `*.obj`**, written for compiled object files (the section
header at line 12 says so). Every DAMIT shape model in this project is therefore untracked by
default. The consequence is on the record as PTM-303: the only Justitia shape model in existence
here is a 2026-07-04 *reconstruction* from a preserved PLY, 574 vertices / 1144 faces, md5
`9626bf7eefe8af89ab9ae5cad8bdc021`, single copy, untracked, absent from the OneDrive tree — while
the tracked `body_config.yaml` references it as `mesh_file` and `index.json` reports
`"has_damit_shape": true` with a `shape_path` pointing at a file that does not exist. Chimaera got
the right treatment (an explicit `!` exception in `873938d`); Justitia never did.

So on import the card must:

1. Copy the `.obj` into the body's cache directory with a provenance header naming the DAMIT model
   identifier, the download date and the source URL.
2. **Say that the file is gitignored**, and offer a one-click "track this shape" that appends the
   `!` exception the way `873938d` did for Chimaera.
3. Report vertices, faces, the derived mean radius, and the units it inferred. Asteroid `.obj` are
   conventionally in kilometers while lunar and Europa DEM are in meters; the engine prints its
   inference (`[mesh-units] inferred KILOMETERS ... confirmed by R_body`) and the card should show
   the same line before the run rather than after.

#### 5.4.4 Surface properties, thermal inertia, and one live landmine

Thermal inertia is the asteroid observable, so the card should present `TI = sqrt(k rho cp)` as the
primary control with a taxonomy preset behind it (the existing `S-type`, `C-type`, `Bare Rock`,
`Icy Body` entries at `config_panel.py:3074-3103`), and derive `ks`, `kd`, `rhos`, `rhod`, `cp0` to
match — showing all five, so the user can see what the preset did.

**The landmine.** A `generic_body.surface.*` value below about 5e-7 is forwarded through
`Config::setProperty(key, double)`, which is `std::to_string` — six decimal places — so it arrives
as the string `"0.000000"`, fails the receiving `> 0.0` guard, and the solver silently keeps the
**Europa placeholder property**, because a generic body's name resolves to Europa by design
(PTM-201; measured: `ks = kd = 4e-7` and `ks = 0.0096, kd = 0.05` produce byte-identical
statistics). A 24000-fold conductivity error, silently. `ks < 5e-7 W/m/K` is not exotic — it is the
porous end of the cometary and trans-Neptunian regolith literature, which is exactly where someone
modeling an insulating lag would go.

Until PTM-201 is fixed engine-side, the card must **refuse** a `generic_body.surface` conductivity
or density below the round-trip threshold, and say why. This is the same shape as the beaming rule:
the GUI holding a line the engine does not yet hold.

### 5.5 A fifth button?

`Custom body` — a generic body that is not an asteroid — is what a user reaches for to model
Enceladus, Titan, Ceres or an exoplanet analog. It is the honest destination for the four dropdown
entries §3 shows to be broken, and it is mechanically the asteroid card with the taxonomy presets
swapped for an icy or rocky set. Whether it is a fifth button or a mode inside the asteroid card is
**DP-4**.

---

## 6. Architecture

### 6.1 Body-first and the existing panel

The panel is not replaced and not rewritten. The body cards are a new view over the same
`ConfigManager`; the eleven-section nav rail becomes the **Advanced** escape hatch, reachable from
every card by one control, and every edit made there is visible in the card when the user returns.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  landing screen: which body?                                     │
  └───────────────┬──────────────────────────────────────────────────┘
                  │  choose Moon / Europa / Mercury / Asteroid
  ┌───────────────▼──────────────────────────────────────────────────┐
  │  body card  (≤15 controls, body defaults applied and labeled)    │
  │                                       [ All settings…  ] ────┐   │
  └───────────────┬──────────────────────────────────────────────┼───┘
                  │  Run                                         │
  ┌───────────────▼──────────────────────┐   ┌───────────────────▼───┐
  │  console / results                    │   │  the existing 11-tab  │
  └───────────────────────────────────────┘   │  panel, unchanged     │
                                              └───────────────────────┘
                            same ConfigManager throughout
```

Two invariants, both of which Stage 1 and Stage 2 paid for:

* **The card writes only what it shows, plus the body defaults it announces.** No card writes a key
  the user cannot see somewhere and cannot get back to.
* **Round-tripping is preserved.** Stage 2's ruamel writer means an opened file survives a no-edit
  save with zero diff lines on all 77 corpus files. A card must not regress that by rewriting the
  whole document to express a body default.

### 6.2 What happens to the five Stage-2 task cards

Four of the five are already body cards wearing different names:

| Stage-2 preset | disposition |
|---|---|
| `lunar_global_diurnal` | becomes **Moon → whole Moon** |
| `lunar_polar_crater` | becomes **Moon → a place on the Moon** |
| `europa_diurnal` | becomes **Europa → whole Europa** |
| `generic_body` | becomes **Asteroid** |
| `reproduce_baseline` | **not a body.** It is an entry *mode* |

`reproduce_baseline` reads a finished run's own `config_used.yaml` and `provenance.txt` rather than
guessing values, and that behavior is right and should be kept. It belongs on the landing screen as
a third way in, beside "start from a body" and "open a configuration file" — not as a fifth body
button. That is **DP-5**.

The `Preset.apply()` contract — a preset reports the exact dotted keys it wrote, and refuses
beaming past the facet-scale rule rather than turning it on and hoping — should be lifted verbatim
into the body cards. It is the mechanism that makes an automatic choice auditable.

### 6.3 File → Open of an arbitrary configuration

Today `load_project` calls `load_from_yaml` then `load_from_config` and lands on whatever section
was last shown (`gui/ptm3d_gui.py:537-549`). Proposed:

1. **Detect the body** with the same rule the audit script uses: `simulation.body`, plus
   `generic_body`, plus whether the mesh is imported. It is three lines and it is the engine's own
   distinction.
2. **Land on that body's card** with the values from the file.
3. **Diff the file against the card's exposed set.** For every key that is (a) outside what the card
   shows and (b) different from the body default, list it, above the fold:

   ```
   This file sets 6 options this card does not show:
       radiation.use_hmatrix            = true      (default false)
       radiation.min_shadow_fraction    = 0.15      (default 0.30)
       thermal.thermal_b                = 24        (default 20)
       thermal.gpu_devices              = 4         (default 0)
       thermal.equilibration_samples_per_diurnal_cycle = 48   (default -1)
       output.include_primary_ir        = true      (default false)
   They are unchanged and will be written back as they are.
   [ Show all settings ]
   ```

4. **Fall back to the classic panel** when the body cannot be determined, or when the file sets so
   many out-of-card keys that the card would be misleading. The threshold is a judgment call; the
   audit says a Moon-local file legitimately carries about six such keys, so something near ten is
   the natural line.

The rule that must not bend: **a value the card does not show is never silently dropped and never
silently applied.** That is the class Stage 1 spent fourteen items on.

### 6.4 What the engine could contribute: a generated body-capabilities table

`src/python/ptm/engine_defaults.py` is generated from `config_parser.{h,cpp}` and re-derived on
every run of `test_config_parity.py`, so an engine-side change to a default becomes a test failure
and a diff rather than a Python layer quietly believing the old value. **The same pattern applied to
bodies would close §3's defect permanently.**

Proposed `src/python/ptm/body_capabilities.py`, generated from `planetary_properties.h`,
`data/kernels/*.tm`, `gui_core/datasets.py`'s dataset sets and `scripts/fetch_kernels.py`:

```python
BODY_CAPABILITIES = {
    "Moon": dict(
        builtin_properties=True,           # planetary_properties.h getMoonProperties
        metakernel="moon.tm",
        kernel_set="core",                 # scripts/fetch_kernels.py --bodies moon
        albedo_map=True, property_map=True, topo_catalog=True,
        beaming_calibrated=True,           # w = 32.5 deg, Jhoti et al. 2024 Table 1
        solar_day_days=29.530588,
        temperature_ceiling_ok=True,       # T_max well under MAX_PHYSICAL_TEMP_K
    ),
    ...
    # Mars, Ceres, Titan, Enceladus are ABSENT. A body with no entry gets no
    # button, and selecting it in the classic panel warns.
}
```

Four things it buys:

1. A body cannot appear on the landing screen unless the engine backs it.
2. The Mercury day discrepancy becomes a **test failure** the moment `solar_day_days` disagrees
   with `props.day`, instead of a comment in `gui_tooltips.py` that says the two match.
3. The beaming body list stops being a tuple in `facet_scale.py` that the engine cannot see, which
   is half of `ENG-BEAM-SCALE-1`.
4. The `MAX_PHYSICAL_TEMP_K` ceiling becomes a per-body property a card can check *before* a
   multi-hour run rather than a warning in the middle of one.

Cost: **M**, plus the parity test. It is the single highest-leverage engine-side item here, and it
is the only one that turns a class of GUI defect into a build failure.

---

## 7. Mockups

These are text mockups of a proposal, not screenshots. Where a value appears elsewhere in this
document it is the measured one; values that exist only to make a mockup readable — a runtime
estimate, a spin pole — are marked *illustrative*.

### 7.1 The landing screen

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PTM3D                                                                       │
│                                                                              │
│  What would you like to model?                                               │
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│   │              │  │              │  │              │  │              │   │
│   │     MOON     │  │    EUROPA    │  │   MERCURY    │  │  ASTEROID    │   │
│   │              │  │              │  │              │  │              │   │
│   │  global or   │  │  global or   │  │    global    │  │  DAMIT shape │   │
│   │  polar site  │  │ a Sutton DEM │  │              │  │  or a sphere │   │
│   │              │  │              │  │  ⚠ limited   │  │              │   │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Open a configuration file…      Reproduce a published run…      All settings…│
└──────────────────────────────────────────────────────────────────────────────┘
```

The Mercury tile carries its warning badge on the landing screen, not two clicks in. Whether it is
there at all is DP-3.

### 7.2 The Moon card, whole-Moon mode

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ← bodies          MOON                                     [ All settings… ]│
│                                                                              │
│  ( • ) The whole Moon        (   ) A place on the Moon                       │
│                                                                              │
│  WHEN                                                                        │
│    Start          2026-04-05 00:00 UTC                                       │
│    For            1.0   [ synodic periods ▾ ]      = 29.53 days              │
│    Samples        144                              ≈ 4.9 h apart             │
│                                                                              │
│  WHERE                                                                       │
│    Detail         [ ico5 — 20,480 faces ▾ ]        est. 4 min, CPU (illustrative)│
│                                                                              │
│  SURFACE                                                                     │
│    Albedo         ( • ) Built-in map    (   ) One value: 0.12                │
│                   data/albedo_maps/Moon_Albedo_res5.bin  ✓ on disk           │
│    Angle response ( • ) Hayne (2017)    (   ) Hapke                          │
│    Properties     Moon standard — A 0.12, ε 0.95, ks 7.4e-4, H 0.06 m        │
│                   ▸ Customize (albedo, emissivity, ks, kd, H)                │
│    Property map   ☑ per-face H map for ico5        ✓ on disk                 │
│                                                                              │
│  HOW LONG TO SETTLE                                                          │
│    Equilibration  50 cycles                        (project standard)        │
│                                                                              │
│  WHERE TO PUT IT                                                             │
│    Output         output/moon_global_2026-04-05                              │
│    Run on         ( • ) CPU (48 threads)   (   ) GPU                         │
│                                                                              │
│                                              [ Check ]        [ Run ]        │
└──────────────────────────────────────────────────────────────────────────────┘
```

Ten controls. Every number that came from the body says so.

### 7.3 The Moon card's auto-albedo behavior

Switching the mode control to *A place on the Moon* and picking a mesh changes the SURFACE group in
place, and narrates the change rather than performing it silently:

```
  SURFACE
    Albedo         (   ) Built-in map    ( • ) One value: 0.12
                   ⓘ  Per-face albedo maps are built for a specific icosphere
                      and are matched by face count. This run uses an imported
                      mesh (haworth_radcore_r32c.ply, 99,665 faces), so the
                      per-face map cannot apply — the engine would load it,
                      find a size mismatch, and fall back to a single value.
                      A latitude/longitude map (Moon-0.5deg-BondAlbedo.fits)
                      would apply, but is not on disk.       [ Fetch ]
```

And the beaming control, on `sunlit_radcore_r50` (the mean and fifth-percentile facet scales are
the measured ones from `facet_scale.py`):

```
    Roughness      ☑ Sub-facet roughness (Smith 1967), rms slope 32.5°
                   facet scale 236.6 m (finest 5%: 169.4 m) — far above the
                   millimeter-to-centimeter roughness scale, so beaming adds
                   what this mesh does not resolve.
```

versus, on `moon3d_...NEW_500k_clean` (509,337 faces, mean facet scale 1.96 mm; no fifth-percentile
figure has been measured for this mesh, so the card would compute and show one):

```
    Roughness      ☐ Sub-facet roughness                          [disabled]
                   facet scale 1.96 mm — this mesh already resolves the
                   roughness. Turning beaming on would count it twice.
                   See docs/radiation-exchange.md §8.4.
```

### 7.4 The Europa card, DEM-site mode

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ← bodies          EUROPA                                   [ All settings… ]│
│                                                                              │
│  (   ) The whole Europa        ( • ) A Sutton DEM site                       │
│                                                                              │
│    Terrain        [ Ridges — 60 m posting ▾ ]                                │
│                   data/external/sutton_DEMs/Ridges_DEM_60m_SOCET.ply         │
│                   ⚠ not on disk — this DEM is not distributed with PTM3D     │
│    Latitude       [ 80° ▾ ]        one of 0, 15, 30, 45, 60, 75, 80, 85, pole│
│    View factors   Ridges_DEM_60m_SOCET.vf         will be built on first run │
│    Self-heating   ☑ radiation exchange between facets                        │
│    Output         output/europa_lat80_ridges                                 │
│                                                                              │
│    ☐ Run the whole grid instead — 5 terrains × 9 latitudes = 45 runs         │
│                                                                              │
│  Ice properties   Europa standard — A 0.5, ε 0.9, TI ≈ 60 SI, H 0.07 m       │
│                   ▸ Customize (albedo, emissivity, ks, kd, H)                │
│  ⓘ Sub-facet roughness is not offered on Europa: the Smith slope is a lunar  │
│    regolith calibration and has no measured value for ice.                   │
│                                                                              │
│                                              [ Check ]        [ Run ]        │
└──────────────────────────────────────────────────────────────────────────────┘
```

Five controls reproduce a forty-five configuration campaign, which is what the audit says the
campaign actually is.

### 7.5 The asteroid card, body pane

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ← bodies          ASTEROID                                 [ All settings… ]│
│                                                                              │
│  ( • ) Define the body        (   ) Set up the run                           │
│                                                                              │
│    Body           269 Justitia (A887 SA)              [ Search HORIZONS… ]   │
│    NAIF ID        20000269                                                   │
│                                                                              │
│    Ephemeris      (   ) Keplerian elements     ( • ) SPICE kernel            │
│      Kernel       …/269_Justitia_(A887_SA)/ephemeris.spk                     │
│      Coverage     2034-10-12 → 2034-11-11 UTC  (30.0 days)                   │
│      ⚠  YOUR RUN WINDOW IS OUTSIDE THIS SPK                                  │
│         run:      2035-08-24 → 2036-02-13   (starts 318 d past coverage)     │
│         with equilibration, back to 2035-06-11 — also outside                │
│         Every epoch would fall back to Keplerian, silently, in the engine.    │
│         [ Fetch a wider SPK ]  [ Use Keplerian deliberately ]  [ Run anyway ] │
│                                                                              │
│    Spin           period, pole RA/Dec, obliquity      (values illustrative)  │
│    Shape          ( • ) DAMIT mesh   (   ) sphere   (   ) ellipsoid          │
│                   shape.obj — 574 vertices, 1144 faces, mean radius 25.36 km │
│                   units inferred: KILOMETERS (confirmed against radius_km)   │
│                   ⚠ .obj is gitignored here (.gitignore:14, a bare *.obj)    │
│                     so this shape is NOT tracked and exists in one copy.     │
│                     [ Track this shape ]                                     │
│    Surface        [ C-type ▾ ]  TI 100 SI → ks 0.008333, ρs 1500, cp0 800    │
│                   (the quickstart's own worked example, examples/            │
│                    quickstart_asteroid_diurnal.yaml)                         │
│                   ▸ Show all ten surface properties                          │
│                                                                              │
│                                                     [ Save this body ]       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Staged plan

### Stage A — the landing screen and two cards (the value is here)

| item | what | effort |
|---|---|---|
| A-1 | Landing screen with body buttons; routes to a card or to the classic panel | **M** |
| A-2 | `BodyCard` base: applies labeled body defaults, reports every key it writes (lift `Preset.apply()`'s contract), one control back to the full panel | **M** |
| A-3 | Moon card, both modes | **M** |
| A-4 | Europa card, both modes | **M** |
| A-5 | Body detection on File → Open, and the out-of-card key disclosure of §6.3 | **M** |
| A-6 | Retire the four unbacked dropdown bodies, or gate them behind a refusal that names §3's measurement | **S** |
| A-7 | Fix `datasets.albedo_status()` to take the mesh into account, so it stops promising a per-face map on an imported mesh | **S** |

### Stage B — asteroids

| item | what | effort |
|---|---|---|
| B-1 | Asteroid card, both panes, reusing the HORIZONS panel and `BodyDefinitionTab` fields | **L** |
| B-2 | SPK coverage check with the window-versus-coverage verdict, including the equilibration window | **M** |
| B-3 | Keplerian as an explicit radio choice, written into the configuration and the provenance | **S** |
| B-4 | DAMIT import with provenance header, the gitignore warning and the one-click `!` exception | **M** |
| B-5 | Refuse `generic_body.surface` values below the `std::to_string` round-trip threshold (PTM-201) | **S** |
| B-6 | Thermal inertia as the primary control, with the five derived properties shown | **S** |

### Stage C — Mercury, and the engine-side foundation

| item | what | effort |
|---|---|---|
| C-1 | `body_capabilities.py`, generated, with a parity test | **M** |
| C-2 | Fix `props.day` for Mercury (58.646 → 175.94 d) and re-derive the GUI's synodic table from the engine | **S** engine, **M** with the re-validation it implies |
| C-3 | Settle `MAX_PHYSICAL_TEMP_K` for rocky bodies, and resolve the 577 K measurement | **M**, needs a decision |
| C-4 | `examples/quickstart_mercury_diurnal.yaml` plus a Mercury notebook section | **S** |
| C-5 | Mercury card | **S** once C-1 to C-4 land |

### Stage D — the payoff the cards enable

| item | what | effort |
|---|---|---|
| D-1 | "Run the grid" on the Europa DEM card, backed by `ptm.Sweep` | **M** |
| D-2 | The same on the Moon site card, over a site manifest | **M** |
| D-3 | A `Custom body` card, if DP-4 says yes | **M** |

---

## 9. What this investigation did not cover

* **No GUI code was written or run in a window.** `VisualizationPanel()` still crashes headless
  (SIGSEGV, exit 139) and `PTM3DMainWindow` constructs one, so nothing here is a claim about
  look-and-feel, modal behavior or the result views. The control counts in §2.4 come from a real
  `ConfigurationPanel` instantiated offscreen; everything else in §5 and §7 is a proposal, not a
  measurement of an implementation.
* **The four unbacked-body runs and the Mercury run are ico2, four timesteps, CPU, radiation off.**
  They establish what the body-name path does. They are not statements about any production
  configuration, any GPU path, or any longer run.
* **The Mercury 577 K measurement is reported, not diagnosed.** It says the run exceeded the clamp
  the engine's own warning names, on the CPU SIMD batch path. It does not establish why.
* **No mesh was loaded to compute a facet scale here**; §5.1.3's numbers are quoted from
  `facet_scale.py` and the beaming session that measured them, not re-measured.
* **DEM module, camera panel, results plotting, dashboard and the exosphere binary are untouched**,
  as is every non-macOS platform.
* **The corpus is the shipped corpus.** The EMA asteroid campaign configurations live outside
  `main` (and PTM-300 records that about fourteen of them are untracked), so the asteroid cluster
  here is four files. Its 36 varying keys are almost certainly an *under*-count of what a real
  asteroid campaign touches.

---

## 10. Decisions for Paul

**DP-1 — does body-first replace the landing screen, or live beside it?**
Ada recommends **replace**: the body buttons become the first thing the application shows, with
"Open a configuration file" and "Reproduce a published run" as secondary entries on the same
screen. The alternative — a body strip above the existing tabs — keeps the 145-control surface as
the first impression, which is the thing the brief asks to fix.

**DP-2 — do the cards replace the Basic tier, or sit above it?**
Stage 2 built field-level Basic/Advanced (71/74). If the cards land, "Basic" has two meanings.
Ada recommends collapsing to two levels: **card** and **all settings**, retiring the Basic/Advanced
toggle. That discards recent work, so it is Paul's call, not Ada's.

**DP-3 — is Mercury a first-class button given §5.3?**
Options: (a) ship the button with a permanent limitations banner; (b) hold the button until
`props.day` and the 500 K ceiling are fixed; (c) no Mercury button, and Mercury reachable only
through the classic panel. Ada recommends **(b)** — the day error is a factor of three on the
quantity that defines every equilibration count, and a button is an endorsement.

**DP-4 — a fifth "Custom body" button?**
It is the honest destination for Enceladus, Titan, Mars and Ceres once §3's dropdown entries are
retired. Ada recommends **yes, but as a mode inside the asteroid card** first, promoted to its own
button only if usage justifies it.

**DP-5 — what happens to the five Stage-2 task cards?**
Ada recommends subsuming four into body cards and keeping `reproduce_baseline` as a landing-screen
entry mode (§6.2). It reads a finished run's own files rather than guessing, which is worth
preserving whatever else changes.

**DP-6 — is the ≤15 target a rule or an aspiration?**
It holds comfortably for Moon-global (11 controls over 17 varying keys), Europa-global (9) and
Europa-local (5). Moon-local lands at exactly 15. It does not hold for the asteroid
body-definition form, which is about 17 fields set by physics. Ada recommends **≤15 for the run
pane, no cap on a body-definition form**, and saying so plainly rather than splitting the asteroid
form to hit a number.

**DP-7 — how strict is the File → Open fallback?**
How many out-of-card, non-default keys before the card is misleading and the classic panel is the
honest landing place? Ada suggests **ten**, from the measurement that a legitimate Moon-local
configuration carries about six.

**DP-8 — should `body_capabilities.py` be built, and generated or hand-maintained?**
Ada recommends **generated, with a parity test**, on the `engine_defaults.py` precedent. It is the
only item here that converts a class of GUI defect into a build failure, and it is what stops §3
recurring.

**DP-9 — does the beaming rule move into the engine?**
`ENG-BEAM-SCALE-1` is still open: a CLI user has nothing between them and double-counted roughness.
If DP-8 lands, the rule can live in the shared table and both sides can read it. Ada recommends
**yes**, after the table.

**DP-10 — do the body cards get to launch sweeps?**
The Europa DEM card can reproduce a forty-five configuration campaign from five controls, and the
machinery exists. It is also the point where a GUI card can start a multi-day job. Ada recommends
**yes, behind an explicit confirmation that names the run count and the estimated wall time**.

**DP-11 — Mercury's temperature ceiling.**
`MAX_PHYSICAL_TEMP_K = 500.0` is commented "Maximum physically reasonable temperature for icy
bodies" and Mercury's sub-solar point is roughly 700 K. Raise it per body, refuse the run, or leave
it and warn? This is a physics decision, not an interface one, and it gates DP-3.

---

## 11. Owed ledger rows

Not written to `BUGS.md`, per the instruction for this investigation.

| id | finding |
|---|---|
| `ENG-BODY-UNBACKED-RUN` | `simulation.body` naming a body with no built-in properties and no `generic_body:` block completes with exit 0, Europa's radius, metakernel and albedo, and **zero solar flux on every face** — a full, silently fictitious output directory. Measured on Mars, Ceres, Titan and Enceladus. The one `Note:` line is not proportionate to the outcome. |
| `GUI-BODY-DROPDOWN-UNBACKED` | `BUILTIN_BODIES` (`config_panel.py:701`, `:3821`) offers those same four bodies as built-in, shows body-specific thermophysical values for them, and disables the generic-body fields that would make them real. |
| `ENG-MERCURY-SOLAR-DAY` | `props.day = 58.646` days (`planetary_properties.h:185`) is Mercury's **sidereal** period; the run banner labels it "Synodic day (solar day)". The solar day is 175.94 days. `thermal.equilibration_orbits` counts cycles of the wrong length. `gui_tooltips.py:1755-1758` carries the correct 175.942 under a comment asserting the two agree. |
| `ENG-MERCURY-TEMP-CEILING` | `MAX_PHYSICAL_TEMP_K = 500.0`, commented "for icy bodies", is the ceiling on a body whose sub-solar temperature is roughly 700 K; the engine warns that results above it "are not trustworthy". |
| `ENG-CLAMP-PATH-INCONSISTENT` | Measured `temperature_max = 577.054 K` on the `CPU SIMD batch` path in the same run whose warning says the 500 K clamp "will truncate the solution". Either the warning over-claims for that path or the clamp is not applied uniformly across solver flavors. **Root cause not established.** |
| `GUI-ALBEDO-STATUS-IGNORES-MESH` | `datasets.albedo_status(body, resolution, root)` (`gui/gui_core/datasets.py:169`) has no notion of `mesh.import_file`, so an imported-mesh configuration is told a per-face albedo map is available when the engine will reject it on a size mismatch (`ptm_thermal_sim.cpp:2471`) and fall back to a uniform value, announcing it only on stdout. |
| `CFG-VIZ-SECTION-DEAD` | Two shipped Europa configurations carry a top-level `visualization:` block (10 keys) that the engine parser does not read; `visualization` is not among the eleven `ENGINE_SECTIONS`. Unlike the `thermophysical:` dead keys, nothing warns about it. |

Cross-references to rows already filed: **PTM-67** (dead `thermophysical:` keys), **PTM-115**
(Europa's primary-shine exposure), **PTM-201** (`generic_body.surface` truncation to the Europa
placeholder), **PTM-300 / PTM-301** (Justitia SPK coverage and the silent per-epoch Keplerian
fallback), **PTM-303** (the untracked DAMIT shape), **ENG-BEAM-SCALE-1** (no engine-side facet-scale
guard).

---

## 12. Reproducing every number here

```bash
# section 2 — the option-usage audit
python3 scripts/gui_option_usage_audit.py . --keys

# section 2.4 — the control count, offscreen, QSettings redirected first
#   (see the probe in the session notes; PTM3DMainWindow must NOT be built)

# sections 3 and 5.3 — the body probes.
# Run FROM THE REPOSITORY ROOT: the shipped metakernels carry a cwd-relative
# PATH_VALUES = ( 'data/kernels' ).  Needs only the ~44 MB core kernel set
# (python3 scripts/fetch_kernels.py --bodies moon).
for B in Mercury Mars Ceres Titan Enceladus; do
  cat > /tmp/probe_$B.yaml <<YAML
simulation: {start_time: '2026-01-01T00:00:00', duration_days: 0.5,
             timesteps: 4, body: $B}
mesh:       {resolution: 2, import_file: '', scale_factor: 1.0,
             use_topographic: false}
thermal:    {run_thermal_mesh: true, threads: 4, equilibration_orbits: 0.0,
             use_gpu: false, compute_statistics: true}
thermophysical: {use_default_albedo: true}
radiation:  {enabled: false}
illumination: {include_primary_reflection: false, use_batch_processing: true,
               use_albedo_map: false}
output:     {directory: /tmp/out_$B, format: binary, vtk_mode: none}
advanced:   {verbose: false}
YAML
  ./build/bin/ptm_thermal_sim --config /tmp/probe_$B.yaml \
    | grep -E "meta-kernel|R_body =|Synodic day|Default albedo|Average total flux|not in built-in|seed above"
done
# Mercury prints a nonzero flux; the other four print 0.000000000 W/m^2 and
# Europa's R_body = 1560.8 km.

# the grep that shows nothing checks SPK coverage
grep -rn 'spkcov\|spkobj' src/cpp/ src/python/ gui/ scripts/
```
