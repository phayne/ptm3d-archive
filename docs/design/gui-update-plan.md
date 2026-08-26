# PTM3D GUI — update plan (functionality and usability)

Status: **plan only. Nothing here is implemented.** Written 2026-08-25 against `main` at
`ddaeb3b` (PR #128, the Smith-beaming merge). Every number below was measured on that tree,
not estimated; §1.1 lists the probes and how to rerun them.

The GUI was last changed substantively on **2026-08-09** (`b4a4d73`, an American-English
sweep). The only later commit touching `gui/` is `2a0ac1f` (2026-08-21), which changed
`scripts/run_chunked_simulation.py` and mentions the GUI only in its message. In the sixteen
days since, the engine grew the beaming switch and its roughness knob, the `gpu:` guardrails
section, the near-field quadrature order, the spin-up seed selector, and the flux-cube output
product. None of them reached the GUI.

This document is the audit, the defect list, and a three-stage proposal. §7 is the list of
decisions that need Paul before anything is built.

---

## 1. Method

### 1.1 What was measured, and how to rerun it

| Question | Instrument | Result |
|---|---|---|
| Which engine keys can the GUI write? | `ConfigManager._config_for_yaml()` over three arms (built-in defaults; built-in with every `use_default_*` off; generic body fully populated), unioned and compared against `ptm.engine_defaults.ENGINE_DEFAULTS` | 101 of 184 reachable, **83 unreachable** |
| Which keys have a control? | Live `ConfigurationPanel` instantiated under `QT_QPA_PLATFORM=offscreen`, walking `tab.widgets` | 92 dict-bound fields over 9 tabs, plus 6 handler-wired keys |
| Where does the GUI's default disagree with the engine's? | Same dump, keyed diff against `ENGINE_DEFAULTS` | 8 keys |
| What does open-then-save do to a config? | `load_from_yaml` then `_config_for_yaml` on two shipped examples | comments 10 → 0 and 19 → 0; 33 and 42 keys added; one silent value change |
| Are the injected engine environment variables live? | Two-arm engine run, `examples`-class Moon ico3, CPU, radiation off, grep for the engine's own banner | `PTM_ILLUM_RISE_HABS=0` **inert** as the GUI sets it |
| Does the full window build headless? | Direct `VisualizationPanel()` construction offscreen | SIGSEGV (exit 139) before its first print |

Probes live in `~/projects/ada_guiplan_2026-08-25/`. The engine arms used
`~/projects/ptm-postmerge-2026-08-24/build/bin/ptm_thermal_sim`, md5
`9d657590009ade0c0d8f2d07d2faa7a2`, commit `53a6a65`; `git diff 53a6a65..ddaeb3b --
src/cpp/illumination/ gui/` is **empty**, so that binary is a valid instrument for both the
illumination and the GUI questions asked of it.

### 1.2 Effort scale used below

**S** = under half a day. **M** = one to three days including its own test. **L** = more than
three days, or needs a design decision before it can be scoped.

---

## 2. Current-state audit

### 2.1 Coverage headline

```
engine config surface          184 keys / 11 sections
GUI can write                  101 keys   (55%)
GUI cannot write                83 keys   (45%)
GUI has a control for           98 keys   (92 bound + 6 handler-wired)
written with no control         11 keys
written and read by nothing     23 keys   (4 engine_env + 19 generic-body round-trip)
```

The 23 dead keys are **not** a defect: `src/python/tests/test_config_parity.py` pins them
deliberately (`GUI_GENERIC_BODY_DEAD_KEYS`), because the GUI reads them back to repopulate its
own form. The `engine_env.*` four are consumed by `gui_core.preflight.build_engine_env`, not by
the parser. Both are documented in that test. Leave them alone.

### 2.2 The 83 unreachable keys, with a disposition

Disposition key: **MUST** = changes the answer, a colleague cannot get it right without the
control. **SHOULD** = operations, performance or reproducibility; a colleague will eventually
need it. **HIDE** = expert or diagnostic; keep it in the YAML and out of the UI.

#### radiation (19 unreachable)

| Key | Engine default | Disposition | Note |
|---|---|---|---|
| `beaming` | `false` | **MUST** | The switch. Paul's 2026-08-25 ruling makes it standard for lunar polar site configs. |
| `rms_slope_deg` | `0.0` | **MUST** | The knob. `beaming: true` with this at 0 is **fatal at startup** (`ptm_thermal_sim.cpp:13592`), so the two must be one control. |
| `coupling_mode` | `bounded` | **MUST** | PTM-102, open since 2026-08-07. The Radiation tab shows a field labeled "Face Fraction (legacy mode only)" and offers no way to select the mode; the engine prints `WARNING [PTM-89] ... IGNORED in coupling_mode='bounded'`. |
| `nearfield_quadrature_order` | `3` | **MUST** | Order 1 is the M1 centroid estimator and reads +51.7% on the touching-pair closed form; the engine warns it is "a diagnostic setting, not a production one". |
| `use_monte_carlo_vf`, `mc_rays_per_face` | `false`, `100000` | **MUST** | Selects a different view-factor builder. Builder choice is the subject of `docs/design/vf-unification.md`. |
| `band_separated` | `true` | SHOULD | Two-band accounting; turning it off changes what `indirect_solar` and `indirect_ir` mean. |
| `exchange_selection`, `coupling_select_mode`, `coupling_select_dt`, `coupling_select_safety` | `all`, `physics`, `0.0`, `0.5` | SHOULD | The physics-based selective re-thermal path. |
| `gpu_devices`, `gpu_device_ids`, `gpu_batch_size`, `gpu_inter_batch_pause_ms` | `1`, unset, `0`, `0` | SHOULD | First-class wedge mitigations. See §2.4. |
| `radiance_prune_eps`, `use_cluster_cull`, `nearfield_window_factor`, `beam_vf_threshold` | `0.0`, `true`, `4.0`, `1e-7` | HIDE | Tuning; the defaults are the validated ones. |

#### thermal (35 unreachable)

| Key | Engine default | Disposition | Note |
|---|---|---|---|
| `spinup_seed` | `physical` | **MUST** | Selects the initial condition. `legacy` exists to reproduce a pre-fix baseline digit-for-digit — and it is exactly the setting the GUI's injected environment silently overrides on GPU (defect **D2**). |
| `compute_ice_stability` + `ice_stability.*` (5) | `false` + 5 | **MUST** | A whole science product (`ice_depth.bin`) with zero GUI surface. |
| `equilibration_target_K`, `equilibration_stop_on_certificate`, `equilibration_cycles` | `0.1`, `false`, `-1` | **MUST** | The convergence criterion the equilibration is scored against. The GUI exposes only the cycle *count* and a cadence tier. |
| `eq_indirect_passes`, `eq_indirect_source`, `eq_indirect_flux_file` | `1`, `auto`, `''` | **MUST** | The indirect-flux equilibration path. Its default moved this summer (PR #78). |
| `analytic_seed_cycles`, `seed_anchor`, `seed_anchor_refuse_two_timescale` | `-1.0`, `lumped`, `false` | SHOULD | Seed internals; couple them to `spinup_seed` in one section. |
| `initial_state_file`, `final_state_file` | `''` | **MUST** | Warm start and chunk handoff. Without these the GUI cannot resume a run, which is the whole long-run workflow. |
| `checkpoint_every_output_steps`, `checkpoint_lookahead_steps` | `0`, `0` | SHOULD | Crash resilience for a multi-hour run. |
| `heat_flow_mask_file`, `heat_flow_mask_value` | `''`, `1.0` | SHOULD | Spatially varying `Q_b`. |
| `gpu_devices`, `gpu_device_ids`, `gpu_batch_size`, `gpu_inter_batch_pause_ms`, `auto_calibrate_gpu` | `1`, unset, unset, `0`, `true` | SHOULD | See §2.4. |
| `thermal_b`, `thermal_m`, `thermal_n` | `15`, `10`, `15` | SHOULD, **read-only** | The depth grid. Standing instruction: do not retune `thermal_b`. Display it, do not offer to change it. |
| `eq_extrapolation.*` (4) | `off` + 3 | HIDE | Experimental (spectral equilibration). |
| `indirect_mean_out_file` | `''` | HIDE | Diagnostic dump. |

#### thermophysical (4 unreachable)

| Key | Engine default | Disposition | Note |
|---|---|---|---|
| `H_override` | `-1.0` | **MUST** | The regolith density scale height — arguably *the* lunar thermal parameter. `ConfigManager.ENGINE_OVERRIDE_KEYS` is `("albedo", "emissivity", "ks", "kd")`, so `H_override` has no write path at all, and `ThermophysicalTab` additionally disables the H field on built-in bodies. **A colleague cannot change H on the Moon from the GUI.** Verified live: unchecking every `use_default_*` yields `albedo_override, emissivity_override, ks_override, kd_override` and nothing else. |
| `conductivity_model` | `hayne2017` | **MUST** | Selects Hayne (2017) against the Martinez model. |
| `albedo_model`, `albedo_hapke_w` | `''`, `-1.0` | SHOULD | The GUI's Hayne/Hapke selector writes `incidence_angle_albedo` only; `albedo_model` is the newer, more explicit channel. |

#### mesh (4), output (3), illumination (5), simulation (1)

| Key | Disposition | Note |
|---|---|---|
| `mesh.occluder_file` | **MUST** | The core-plus-ring architecture (`docs/extended-occluder-architecture.md`) is unreachable from the GUI. Every A3GT-class site run needs it. |
| `mesh.latitude_override`, `longitude_override` | **MUST** | Places an imported mesh on the body. Without them an imported DEM sits wherever its own frame put it. |
| `mesh.export_basic` | HIDE | |
| `output.write_raw_stats` | SHOULD | `statistics/_raw` is what the chunk merger consumes. |
| `output.skip_mesh`, `skip_time_series` | SHOULD | Output-size control on a long run. |
| `illumination.use_gpu` | SHOULD, with a caveat | Deliberately absent today, and the reasoning is sound (CPU illumination is both correct and faster here). Expose it as an advanced control *with the measurement in the tooltip*, rather than leaving colleagues to wonder why the checkbox exists for thermal and radiation only. |
| `illumination.gpu_devices`, `gpu_device_ids`, `gpu_inter_batch_pause_ms` | SHOULD | Follows `use_gpu`. |
| `illumination.bvh_cache_file` | SHOULD | Reusing a BVH is a large saving on a repeat run. |
| `simulation.start_time_offset_seconds` | SHOULD | |

#### gpu (5) and flux_cubes (7)

Whole sections with no GUI surface at all. Treated separately in §2.4 and §2.5.

### 2.3 The 11 keys written with no control

Written into every saved YAML, always at the engine default, with no widget and — for ten of
the eleven — no mention anywhere in `config_panel.py`:

```
advanced.default_visible_albedo   output.complete_vtk_file    thermal.enable_shadow_flux
advanced.verbose                  output.location_data        thermal.geothermal_heat_flow
thermal.crater_dd_ratio           output.thermal_vtk_file     thermal.integration_offset_days
thermophysical.constant_thermal_inertia (handler-wired)       thermal.integration_offset_steps
```

They are inert — every one carries the engine's own default — but they add eleven lines to
every saved config that look like settings and are not. Two of them are worth a control rather
than deletion: `thermal.geothermal_heat_flow` (per-run `Q_b`) and `thermal.enable_shadow_flux`.

### 2.4 The `gpu:` guardrails section — and why it is not a simple "expose it"

`config_parser.h:798-830` defines five keys. Four are ordinary. The fifth is not:

```cpp
// NOT a YAML key: true when min_vram_margin_gib was written in the
// config or the environment, which makes the floor enforcing on every
// platform rather than only on the wedge-prone one.
bool min_vram_margin_explicit = false;
```

`gpu.min_vram_margin_gib` is the **only key in the whole 184-key surface whose semantics depend
on whether it was written at all** (grep for `_explicit` in `config_parser.cpp`: three hits, two
of them the albedo pair, one this). The GUI's serialization model is "write every key I know
about". Applied here, that model would silently promote a macOS-plus-AMD guardrail into a
universal one for every config the GUI ever touches, on every machine.

This is the sharpest instance of a general problem: **a GUI that always writes everything cannot
represent "unset".** See decision **DP-1**.

### 2.5 `flux_cubes:` — a product with no surface

Seven keys, an output format, `docs/flux-cubes.md`, and no mention anywhere in `gui/`. Also
absent from `docs/configuration.md`. Lowest priority of the missing sections, but it should be
either exposed or explicitly declared CLI-only.

### 2.6 Where the GUI's default disagrees with the engine's

Eight keys. Six are deliberate and documented in `config_manager.py`. Two are not:

| Key | Engine | GUI | Verdict |
|---|---|---|---|
| `simulation.body` | `Europa` | `Moon` | Deliberate (Paul's D1, 2026-08-02). |
| `simulation.duration_days` | `3.55` | `29.530588` | Deliberate — one lunar synodic period. |
| `simulation.timesteps` | `72` | `144` | Deliberate. |
| `thermal.run_thermal_mesh` | `false` | `true` | Deliberate — the GUI is a thermal application. |
| `thermal.equilibration_samples_per_diurnal_cycle` | `-1.0` | `120.0` | Deliberate — the "Balanced" accuracy tier. |
| `mesh.topo_directory` | `''` | `data/topo` | Deliberate — the built-in dataset contract. |
| **`radiation.max_coupling_iterations`** | **`16`** | **`10`** | **Stale.** The engine's default was raised when the relative convergence criterion landed. The GUI stamps the old cap into every config it saves. Under `coupling_mode: bounded` the loop usually exits on its own criterion well before either cap, so this is normally inert — but it is a *cap*, and on a high-spectral-radius mesh it binds. Fix to 16, or make it follow the engine. |
| **`radiation.use_gpu`** | **`false`** | **`true`** | **Questionable.** Injected into every saved config, including configs whose author never mentioned radiation. Inert while `radiation.enabled: false`; live the moment anyone ticks the radiation box. |

---

## 3. Defects

Ordered by how much damage they can do to a result.

### D1 — the GUI overwrites the config file it was given, silently, on Run

**Trigger.** `ptm3d_gui.py:692`. If a project file is open, `run_simulation` calls
`save_project()` before launching — no prompt, no diff, no backup. `save_project_to`
(`:580`) calls `ConfigManager.save_to_yaml`, which is `yaml.dump` of the merged dictionary.
Save and Save As do the same thing, but those the user asked for.

**Measured, on two shipped example configs:**

| Config | lines | comments | keys | value changed |
|---|---|---|---|---|
| `examples/tutorial04_crater80N_coupled.yaml` | 56 → 79 | 10 → 0 | 37 → 70 (+33) | none |
| `examples/quickstart_moon_diurnal.yaml` | 56 → 79 | 19 → 0 | 28 → 70 (+42) | `illumination.albedo_maps_directory: 'data/albedo_maps'` → `''` |

Among the injected keys: `radiation.use_gpu: true` and `radiation.max_coupling_iterations: 10`
into a config that mentioned neither.

The 2026-08-14 handoff recorded a worse instance on
`~/dev/heat3d/a3gt_tutorial_ms3/ms3_tutorial_demo.yaml`: 274 lines and 185 comments reduced to
92 lines and 0 comments, **and `thermal.use_gpu` flipped false to true** before a live demo.
The comment loss reproduces exactly — that file today is 92 lines with 0 comments, and a fresh
round trip through it now changes nothing, which is what an already-flattened file should do.
**The `use_gpu` flip did not reproduce.** At `ConfigManager` level the round trip reports 0
changed keys, and no code in `gui/` writes `thermal.use_gpu` other than the checkbox
(`config_panel.py:1587`, which defaults to `False`). Candidate explanations not excluded: the
`gpu_accelerated` preset, or a click. **Treat the value flip as unreproduced, not as fixed.**

**Severity.** The comment loss is annoying. The silent write of a *different* file than the one
the author wrote is the serious half: a config is a scientific record, and a run that quietly
rewrites its own inputs cannot be audited afterward.

### D2 — PTM-276: the injected `PTM_EQ_PERFACE_FLUX=1` overrides an explicit `spinup_seed: legacy` on GPU

`preflight.py:61` puts `PTM_EQ_PERFACE_FLUX=1` into every GUI run's environment. Traced through
`ddaeb3b`:

| Path | `spinup_seed` | Effect of the injected variable |
|---|---|---|
| CPU SIMD batch | either | Inert, and the engine says so: `[eq-seed] WARNING: PTM_EQ_PERFACE_FLUX is set but is superseded by thermal.spinup_seed ... NO effect on this run` (`ptm_thermal_sim.cpp:5064`, inside `runThermalModelsForAllFacesBatchSoA`) |
| GPU persistent | `physical` (default) | Inert — the branch at `:7921` takes the shared-seed path and never calls `eqPerfaceFluxMode()` |
| GPU persistent | **`legacy`** | **Live.** `:7966` and `:8658` call `eqPerfaceFluxMode()`, which reads the variable and switches the seed from the first-10-step instantaneous flux with a 100 K nightside floor to the per-face time mean |

`spinup_seed: legacy` exists for exactly one purpose — reproducing a pre-fix baseline
digit-for-digit. The GUI is the one caller that can defeat that purpose without the user having
set anything. Disclosed in the log (`[eq-seed] LEGACY GPU equilibration seed: PER-FACE ...`) but
never asked for.

Two consequences worth separating: the GUI should not inject it (this plan), and the engine's
CPU path claims "NO effect on this run" in a build where the GPU path honors it (an engine-side
asymmetry — §9, owed row).

### D3 — the injected `PTM_ILLUM_RISE_HABS=0` is inert; the GUI's own comment says it is applied

**This is new, and it is the sharper of the two environment findings.**

`preflight.py:59` sets `PTM_ILLUM_RISE_HABS=0` and documents it as
"grazing rise-floor leak fix: h_abs = 0 kills 100% of leaks with zero shadow acne". But
`rise_habs_` is read **only** inside `traceOccludedRiseEps` (`illumination_model.cpp:1419`),
which is called only when `use_rise_eps_` is true (`:3266`, `:3354`, `:4089`, `:4176`), which
requires `PTM_ILLUM_RISE_EPS=1` (`:616`). **The GUI never sets it.**

`docs/ENVIRONMENT.md:14-17` states the validated recommended-ON set as six variables:

```
PTM_ILLUM_TRUE_NORMALS=1  PTM_ILLUM_RISE_EPS=1  PTM_ILLUM_RISE_KEPS=0.2
PTM_ILLUM_RISE_HABS=0     PTM_ILLUM_SEAL_HOLES=1  PTM_EQ_PERFACE_FLUX=1
```

`RECOMMENDED_ENGINE_ENV` carries **four**. The two missing ones are the gate and its parameter.

**Discriminating test.** Two engine runs, identical config (Moon ico3, 48 output steps, CPU,
radiation off), differing only in the environment; grep for the engine's own banner:

| Arm | environment | `rise-epsilon ... ENABLED` | `ABSOLUTE self-relief floor OVERRIDE` |
|---|---|---|---|
| A | the GUI's exact four | 0 | 0 |
| B | the documented six | 1 | 1 |

Both `rc=0`. Arm B prints
`PTM-13 Class-2 fix: ABSOLUTE self-relief floor OVERRIDE h_eps = 0.000000 mm (env PTM_ILLUM_RISE_HABS)`;
arm A prints nothing of the kind. **Every GUI run since 2026-07-29 has been missing the
grazing-incidence leak fix while its own source comment said otherwise.**

`docs/tutorials/07_gui_walkthrough.md:240-242` repeats the four-variable claim, so the tutorial
is wrong in the same way.

### D4 — the four injected variables are also merged into a config that deliberately omits them

`ConfigManager.load_from_yaml` merges the loaded file **into** `DEFAULT_CONFIG`, whose
`engine_env` is `dict(RECOMMENDED_ENGINE_ENV)` (`config_manager.py:242`). A config carrying `engine_env: {}` — a
deliberate, documented way of saying "no fix flags" — comes back carrying all four. A config
that sets one of them to `'0'` keeps that one and silently gains the other three. There is no
"this config asked for nothing" state.

### D5 — the presets are dead code, and the README advertises them

`ConfigManager.PRESETS` defines five (`quick`, `highres`, `moon_thermal`, `crater_radex`,
`gpu_accelerated`). `grep -rn "load_preset" gui/` returns **only the definition**. No menu, no
button, no dialog. `gui/README.md:32` lists "**Presets**: Quick-start configurations for common
scenarios" under Features.

### D6 — `radiation.max_coupling_iterations` is pinned to a stale value

See §2.6. GUI 10, engine 16.

### D7 — four stale `gui/_private_tools/` references

`gui/README.md:235`, `gui/dashboard.py:5`, `gui/dashboard_core/module_interface.py:4`,
`gui/dashboard_core/chaining_manager.py:20`. The directory was removed on 2026-08-03
(`2b5df5b`); these were flagged as "informational, not fixed" then and are still here. Comments
only, break nothing.

### D8 — `docs/tutorials/07_gui_walkthrough.md` carries 7 unfilled screenshot placeholders

`[SCREENSHOT-PAUL #1..#5]` plus their summary table. Owed since 2026-08-07; blocked on a live
display, because the window cannot be built headless (§6.1).

### D9 — `docs/configuration.md` does not document keys the GUI is about to expose

Zero occurrences of `nearfield_quadrature_order`, `spinup_seed`, or `flux_cubes` in 678 lines.
A "Read the docs for this key" link from a new control would point at nothing. This is a doc
gap, not a GUI defect, but it gates Stage 2's inline help.

---

## 4. Usability review

### 4.1 The workflow today

`config → validate → auto-save → preflight → single engine process → console → results browser`.
Each link works. Three things are missing from it.

**No run is resumable.** `thermal.initial_state_file` and `final_state_file` have no controls,
and the GUI launches `ptm_thermal_sim` directly (`simulation_runner.py:68`) rather than through
`scripts/run_chunked_simulation.py`. So a GUI run gets none of the chunked runner's GPU cool-down,
stall abort, per-attempt log rotation, or resume. To the GUI's credit this is **disclosed, not
silent**: `ptm_run.long_run_advisory` emits a warning saying exactly that. But disclosure is not
a workflow, and the wedge on this machine can power the box off.

**Nothing between "the run finished" and "the run is right."** `scripts/ptm_qa.py thermal
<run_dir>` is the project's own machine-readable correctness gate and the GUI never calls it.

**No cost estimate outside view factors.** The Phase-3 estimator (`preflight.py:395`) prices the
view-factor build well. It prices nothing else — not the thermal solve, not equilibration, and
now not beaming, whose measured overhead is +10.5% after the `perf/beaming-overhead` work
(+22.6% before it).

### 4.2 What a colleague stumbles on

Ranked by how likely it is to produce a wrong answer rather than an annoyance.

1. **H is not settable on the Moon** (§2.2). The single most-tuned lunar parameter, and the
   field is disabled with no override path behind it.
2. **"Face Fraction (legacy mode only)"** is visible with no way to select the mode it belongs
   to, and the engine warns at runtime that it is ignored. A user reasonably concludes the run
   used it.
3. **Radiation is behind an Advanced toggle that is off by default.** Correct for a first run;
   wrong the moment anyone works on a crater, which is most of what this model is for.
4. **The Basic tier has no radiation and no illumination, so the Basic tier cannot express a
   crater run at all.** The tier boundary is drawn by tab, not by task.
5. **Opening a config and pressing Run rewrites it** (D1).
6. **Every default run prints a yellow `[eq-seed] WARNING` about an environment variable the
   user never set** (D2). Warning fatigue, from the tool's own injection.
7. **No occluder-file control**, so the documented site architecture is CLI-only.
8. **Presets advertised, absent** (D5).

### 4.3 Grouping the larger key surface

Adding the MUST set roughly doubles the control count. Tab-by-tab is already near its limit
(the Radiation tab holds 18 fields).

The proposal is **task-first, section-second**: keep the nine section tabs as the complete
reference view, and put a small number of *task cards* in front of them. A task card is a short
form that sets a coherent group of keys and states what it changed. "Polar crater with
self-heating" is a task; `radiation.captured_energy_target` is not.

And **move the tier boundary from tabs to fields.** Every tab gets Basic fields inline and an
"Advanced" disclosure below them, so Basic can reach `radiation.enabled` and `beaming` without
also revealing `aca_max_rank`. The nav rail stops hiding whole tabs.

### 4.4 Presets, and the beaming scale rule they must encode

`docs/radiation-exchange.md` §8.4 gives a quantitative criterion. Facet scale is the square root
of the mean face area, and the comparison is against the thermal-isolation scale of roughly a
millimeter to a centimeter:

| Mesh | faces | facet scale | beaming |
|---|---|---|---|
| `sunlit_radcore_r50` | 140,091 | 237 m | ON |
| `haworth_radcore_r32c` | 99,665 | 181 m | ON |
| A3GT `A03MP026` 3 km core | 2,397,890 | 3.46 m | ON |
| `moon3d_c...NEW_500k_clean` | 509,337 | **1.96 mm** | **OFF** — the mesh *is* at the roughness scale |

**There is no engine-side facet-scale guard.** The engine enforces only that `beaming: true`
carries a positive slope; grep for a facet-scale check in `src/cpp/` returns nothing. The
tripwire at `radiation_exchange.cpp:2969` catches the opposite mistake (knob without switch).
So if the GUI offers the switch, **the GUI is the only thing standing between a user and
double-counted roughness.**

Proposed encoding, entirely computable before launch:

```
facet_scale = sqrt(total_face_area / n_faces)

built-in icosphere:  total area = 4*pi*R_body^2, n_faces = 20 * 4^resolution   (exact, free)
imported mesh:       sum triangle areas from the PLY; sample 5,000 evenly
                     spaced faces above 5,000,000 faces and say so in the label

facet_scale < 0.05 m   ->  beaming control DISABLED, with the reason and the number
0.05 m .. 1.0 m        ->  enabled, WARN and require an explicit acknowledgment
>= 1.0 m               ->  enabled, no warning
body is not the Moon   ->  disabled regardless of scale; w = 32.5 deg is calibrated on
                           lunar regolith and docs/radiation-exchange.md declines to carry
                           it to icy or small bodies
```

The 0.05 m boundary is one decade above the top of the "millimeters to centimeters" band in
§8.4, chosen so the moon3d 509k mesh at 1.96 mm is refused with margin and the A3GT 3 km core
at 3.46 m passes cleanly. It is a proposal, not a measurement — **DP-4**.

Proposed preset list, each stating which keys it sets:

| Preset | Sets | Beaming |
|---|---|---|
| Global lunar diurnal | body Moon, icosphere res 5, one synodic period, radiation off | n/a (off) |
| Lunar polar crater, self-heating | imported mesh + occluder, `radiation.enabled`, `coupling_mode: bounded` | **on**, `rms_slope_deg: 32.5`, if the scale rule passes |
| Europa diurnal | body Europa, albedo map on | off — and the control disabled with the reason |
| Asteroid or custom body | generic body, HORIZONS-populated | off, disabled |
| Reproduce a published baseline | `spinup_seed`, `analytic_seed_cycles`, `engine_env` as the baseline recorded | off |

The last one is the useful pair to `spinup_seed: legacy` and would have made D2 impossible to
hit by accident.

### 4.5 Run monitoring

The console classifier (`ptm.run.classify_severity`) is content-based and sound: `_ERROR_RE`
matches `FATAL`/`ERROR` with an optional tag, crash signatures and "simulation failed";
`_WARN_RE` matches `warn`, "conservative defaults", "falling back", SPICE diagnostics.

Against the banners a colleague actually needs to see:

| Banner | Reaches the console? | Surfaced as? |
|---|---|---|
| `[Radiosity] ACTIVE RUNG: ...` | yes | plain info, scrolls past |
| `[gpu-guard]` placement and margin decisions | yes | info, unless the text says "refuse"/"fall back" |
| `[config] radiation.beaming = false: rms_slope_deg ... INERT` | yes | info |
| `WARNING [PTM-101]` beaming chunk dependence | yes | warning (correct) |
| `[eq-seed]` seed announcement | yes | info |
| `[eq-seed] WARNING` about the injected variable | yes | warning — **which the GUI itself caused** |
| `[preflight] VF-COST` | yes | info |

Nothing is lost. But the decisive lines — which solver rung is active, which seed, whether
beaming engaged, which environment the physics saw — are three lines in a scrolling console.
They belong in a persistent **run header**: a small pinned panel that fills in as the engine
announces itself, so the answer to "what did this run actually do" is visible without scrolling.

---

## 5. Staged plan

Nothing here is implemented. Every stage ends with `python3 -m pytest
src/python/tests/test_config_parity.py` green (its GUI arm is the parity gate) and a
byte-comparison of a saved config before and after, which is the invariant the 2026-08-02 work
established and should not be dropped.

### Stage 1 — correctness

Ships the defects and the science keys. No layout change, so it can land while the UI design is
still being argued about.

| # | Item | Effort | Description |
|---|---|---|---|
| 1.1 | Stop rewriting the user's file on Run (**D1**) | **M** | `run_simulation` writes the engine-facing YAML *beside the output directory* in every case, never over `current_project_file` — the code path already exists (`save_run_yaml`, used today only when no project file is open). The project file changes only when the user presses Save. Depends on **DP-1**. |
| 1.2 | Fix `RECOMMENDED_ENGINE_ENV` (**D2**, **D3**) | **S** | Add `PTM_ILLUM_RISE_EPS: '1'` and `PTM_ILLUM_RISE_KEPS: '0.2'`; drop `PTM_EQ_PERFACE_FLUX`. Add a test that pins the table against the six-variable set in `docs/ENVIRONMENT.md`, so the next divergence fails a test instead of running for a month. Depends on **DP-2**. |
| 1.3 | Honor a config that declines the flags (**D4**) | **S** | Distinguish "no `engine_env` key" (apply the recommended set) from "an `engine_env` block that is present" (apply exactly what it says). One flag in `load_from_yaml`. |
| 1.4 | `radiation.beaming` + `rms_slope_deg` with the scale rule | **M** | One control pair on the Radiation tab; the slope disabled until the switch is on; the switch disabled by the facet-scale and body rules of §4.4 with the computed number in the tooltip; on load, warn when the incoming config used the pre-2026-08-25 implicit switch (`config_parser.cpp:631`). Depends on **DP-4**. |
| 1.5 | `thermophysical.H_override` and `conductivity_model` | **S** | Extend `ENGINE_OVERRIDE_KEYS` to include `H`, and stop disabling the H field on built-in bodies. Small change, removes the worst science gap. |
| 1.6 | `radiation.coupling_mode` (**PTM-102**) | **S** | A three-way selector; gate the "Face Fraction (legacy mode only)" field on it so the label stops lying. |
| 1.7 | `thermal.spinup_seed` + `analytic_seed_cycles` + `seed_anchor` | **S** | One "Initial condition" group. Pairs with 1.2 — once the GUI stops injecting the variable, `legacy` means what it says. |
| 1.8 | `mesh.occluder_file`, `latitude_override`, `longitude_override` | **S** | Three fields on the Mesh tab; extend `preflight._FILE_FIELDS` so the occluder gets the same existence, dataless and cloud checks as the core mesh. |
| 1.9 | `radiation.nearfield_quadrature_order`, `use_monte_carlo_vf`, `mc_rays_per_face` | **S** | Advanced group on the Radiation tab; warn on order 1 with the engine's own wording. |
| 1.10 | `thermal.initial_state_file` / `final_state_file` | **S** | Two file fields; `initial_state_file` joins `_FILE_FIELDS`. |
| 1.11 | `radiation.max_coupling_iterations` default 10 → 16 (**D6**) | **S** | Or drive every GUI default from `ptm.engine_defaults` — see **DP-3**. |
| 1.12 | Retire the 11 orphan keys (§2.3) | **S** | Delete nine from `DEFAULT_CONFIG`; give `geothermal_heat_flow` and `enable_shadow_flux` real controls. |
| 1.13 | Fix the four stale `_private_tools` comments (**D7**) | **S** | |
| 1.14 | Correct `docs/tutorials/07_gui_walkthrough.md` and `gui/README.md` | **S** | The environment set (D3), the presets claim (D5). |

**Stage 1 total: 2 M, 12 S.**

### Stage 2 — usability

| # | Item | Effort | Description |
|---|---|---|---|
| 2.1 | Field-level Basic/Advanced instead of tab-level | **M** | Each tab renders Basic fields, then a collapsed "Advanced" section. Radiation and Illumination leave the hidden-tab set; the nav rail always shows all nine. Pure visibility, no change to what is written. |
| 2.2 | Task cards | **L** | Three or four cards ahead of the tabs, each a short form over a coherent key group, each printing what it set. Needs **DP-5**. |
| 2.3 | Presets, reachable, with the beaming rule encoded (**D5**) | **M** | Replace `ConfigManager.PRESETS` with the §4.4 list; a File menu entry and a first-run card. A preset that would violate the scale rule refuses and says why. |
| 2.4 | Validation on edit, using the engine's own words | **M** | `ConfigManager.validate()` today checks nine things. The engine has far more, with better messages. Add a "Check configuration" action that writes the YAML to a temporary file and runs the engine's own validation entry point, reporting its text verbatim. Falls back to today's checks if the binary is missing. Needs a confirmed non-executing validation mode in the engine — **DP-6**. |
| 2.5 | `gpu:` guardrails | **M** | Depends entirely on **DP-1** — a section that must be able to say "unset". |
| 2.6 | `thermal.compute_ice_stability` + `ice_stability.*` | **M** | A collapsed group under Thermal, plus an `ice_depth.bin` view in the results browser. |
| 2.7 | Equilibration convergence controls | **S** | `equilibration_target_K`, `equilibration_stop_on_certificate`, `eq_indirect_passes`, `eq_indirect_source` next to the existing cadence tier. |
| 2.8 | GPU device and batch controls | **S** | `gpu_devices`, `gpu_device_ids`, `gpu_batch_size`, `gpu_inter_batch_pause_ms` for thermal, radiation and illumination, populated from the OpenCL probe `ConfigManager` already runs. |
| 2.9 | Depth grid, read-only | **S** | Show `thermal_b`, `thermal_m`, `thermal_n` and the derived skin-depth count. Display only; the standing instruction not to retune `thermal_b` stays. |
| 2.10 | `docs/configuration.md` rows for the newly exposed keys (**D9**) | **S** | |

**Stage 2 total: 1 L, 5 M, 4 S.**

### Stage 3 — monitoring and integration

| # | Item | Effort | Description |
|---|---|---|---|
| 3.1 | Run header panel | **M** | A pinned four-to-six-line summary that fills in from the engine's own banners: binary md5, active solver rung, seed mode, beaming state and slope, environment applied, GPU placement decision. Pure parsing of existing output. |
| 3.2 | Offer the chunked runner for long runs | **L** | Turn `long_run_advisory` from a warning into an offer: launch through `scripts/run_chunked_simulation.py` with cool-down, stall abort and resume, then `ptm_merge_chunks`. Needs the merged-versus-per-step output trade-off stated in the dialog. |
| 3.3 | Post-run QA | **M** | Run `scripts/ptm_qa.py thermal <run_dir>` on completion and show the JSON verdict beside the results. |
| 3.4 | Cost estimate beyond view factors | **M** | Extend `VFCostEstimate` with a thermal-solve and equilibration estimate, and a beaming line (measured +10.5% after `perf/beaming-overhead`, +22.6% before). Label approximate, as the existing estimator does. |
| 3.5 | Resume a stopped run | **S** | Given 1.10, offer "resume from `final_state_file`" when one is present in the output directory. |
| 3.6 | `flux_cubes:` | **M** | Depends on **DP-7**. |
| 3.7 | The 5 live screenshots (**D8**) | **S**, Paul-blocked | Cannot be produced by an agent (§6.1). |

**Stage 3 total: 1 L, 4 M, 2 S.**

---

## 6. What this audit did not cover

### 6.1 The GUI was never launched

`ConfigurationPanel`, `ConsolePanel` and `ResultsPanel` instantiate and can be inspected under
`QT_QPA_PLATFORM=offscreen`; the inventory in §2.1 comes from a real `ConfigurationPanel`
object. **`VisualizationPanel()` still segfaults headless** — measured today on `ddaeb3b`, exit
139, before its first print, so the crash is at import of the pyvistaqt/VTK stack. `PTM3DMainWindow`
constructs one, so the whole window, the 3D and 2D views, the modal dialogs in situ, and every
claim about *look and feel* are outside what was checked. Launching on Paul's display was
declined: this is a plan, and starting the GUI risks a click that starts a simulation.

Everything in §4 is therefore read from code and from the offscreen object model. Claims about
tab contents, field counts, tier assignment and what is written are measured. Claims about
crowding, readability and click paths are inference and are labeled as such.

### 6.2 Not assessed

* **The DEM/mesh module** (`gui/dem_*`, 1,435 lines) and `run_dashboard.py`. Out of scope; they
  have their own workflow and the README already calls the dashboard less-tested.
* **The camera panel** (1,643 lines). Its own subsystem.
* **The results and visualization side** beyond the key surface — no assessment of whether the
  plots are right, only that `ptm_qa.py` is not wired in.
* **Windows and Linux.** Every measurement is macOS, Python 3.8.18, PyQt6 6.7.1 in
  `~/anaconda3/envs/europa_viz`.
* **Whether any of the 83 unreachable keys is unreachable *deliberately*.** Dispositions in §2.2
  are proposals from the engine's own documentation, not rulings.

### 6.3 Explicitly refuted from the brief

* **`thermal.equilibration_orbits` is not missing.** It is written by default and bound to a
  widget (`config_panel.py:1148`). Verified in both probes.
* **The `thermal.use_gpu` false-to-true flip did not reproduce** (D1). Do not carry it forward
  as a known behavior.

---

## 7. Decisions for Paul

**DP-1 — the config-rewrite model.** Three options, and the choice governs items 1.1, 2.5 and
the whole `gpu:` question:

  **(a) Read-only open.** The GUI never writes the file it opened; Run always writes a separate
  engine-facing YAML beside the output. Simplest, safest, smallest change. Loses in-place
  editing of an existing project.

  **(b) Comment-preserving round trip.** Move to `ruamel.yaml` round-trip mode and write back
  only keys the user actually touched. Preserves comments and "unset", and is the only option
  that makes `gpu.min_vram_margin_gib` safe (§2.4). Costs a new dependency and a rewrite of
  `_config_for_yaml`.

  **(c) Status quo plus a warning.** Keep rewriting, but show a diff and ask first.

  *Ada's recommendation: (a) for Stage 1 — it fixes D1 in a day. (b) as a Stage 2 goal, because
  without it the GUI can never represent "this key is unset."*

**DP-2 — the injected environment.** Confirm: add `PTM_ILLUM_RISE_EPS=1` and
`PTM_ILLUM_RISE_KEPS=0.2` (D3), and drop `PTM_EQ_PERFACE_FLUX` (D2). Note that adding the two
**changes the physics of every future GUI run** relative to every past one — that is the point,
but it should be a decision, not a side effect. A second question: should the GUI inject anything
at all, or write the block into the config and let the engine apply it (`engine_env:` has done
that since 2026-08-04)?

**DP-3 — should the GUI's defaults be generated?** `ptm.engine_defaults` is a generated,
test-pinned table of all 184 keys. `ConfigManager.DEFAULT_CONFIG` is hand-maintained, and has
drifted twice (D6, and `radiation.use_gpu`). Deriving GUI defaults from that table — with an
explicit, listed set of deliberate overrides — would make the next drift a test failure. Costs
one refactor of `DEFAULT_CONFIG`.

**DP-4 — the beaming scale rule.** Approve the numeric boundaries in §4.4: disable below 0.05 m,
warn between 0.05 m and 1 m, allow above, and lunar bodies only. And: should the same rule
become an engine-side warning? There is none today, so a CLI user has nothing between them and
double-counted roughness.

**DP-5 — the preset list.** Approve, cut or extend the five in §4.4. In particular, is
"Reproduce a published baseline" worth building, and which baselines should it carry?

**DP-6 — validate against the engine?** Item 2.4 wants to hand a config to `ptm_thermal_sim` and
report its validation text verbatim, so the GUI stops maintaining a second, weaker copy of the
rules. Is there a supported parse-and-exit mode, and if not, is adding one acceptable?

**DP-7 — `flux_cubes:` in the GUI, or CLI-only?** Seven keys and an output product with no GUI
surface and no `docs/configuration.md` entry.

**DP-8 — expose the `gpu:` guardrails, auto-set them, or leave them alone?** Given §2.4, "expose
them" is only safe under DP-1(b). The alternative is a single "GPU safety" selector that writes
`placement_policy` and `thin_margin_policy` and deliberately never writes
`min_vram_margin_gib`.

**DP-9 — the tier boundary.** Approve moving Basic/Advanced from whole tabs to fields (item 2.1).
The consequence is that Radiation and Illumination stop being hidden, which is a visible change
to the first-run experience.

---

## 8. Sequencing

Stage 1 items 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14 are independent of
every decision except DP-2 and can start as soon as that one is answered. Item 1.1 waits on DP-1;
item 1.4 waits on DP-4. Stage 2 waits on Stage 1 and on DP-1, DP-5, DP-6, DP-8, DP-9. Stage 3
waits on Stage 1 item 1.10 and on DP-7.

The single highest-value item is **1.2**: two lines, and it restores a validated illumination fix
that every GUI run has been missing since 2026-07-29.
