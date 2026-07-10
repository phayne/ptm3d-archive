# Project map archive — pre-slim narrative (2026-07-09)

This is the verbatim `Project map` section of `~/.claude/ptm-project-knowledge.md` as it
stood on 2026-07-09, before it was reduced to a registry. It had grown to 705 lines of
append-only campaign narrative (Justitia coverage rounds 1-6, the roughness study, the
buried-ice study, per-body render history) that every agent was instructed to read before
doing any work.

It is retained because the round-by-round reasoning is genuinely useful history. It is no
longer read-first. Mojibake (27 sequences) was repaired before archiving.

The live registry now lives in `~/.claude/ptm-project-knowledge.md`; durable facts in
project memory; bugs in `$PTM/BUGS.md`.

---

## Project map — where artifacts live (VERIFY before use)
Docs live in `$PTM/docs/`; **runs, meshes, view-factors (VF), and baselines live OUTSIDE the repo**,
in several near-identically-named dirs. Do **not** grope with `find` or guess — resolve from here, and
for any regression/comparison run follow the "resolve & confirm baseline" rule in the Workflow above.

**Output / data roots**
- **Repo** `$PTM` = `.../europa/heat3d/ptm` → `docs/` (+ `docs/figures/`), `examples/`
  (`a3gt_v2_template*.yaml`, `cameras/`), `scripts/` (`a3gt_v2_orchestrator.py`,
  `run_chunked_simulation.py`, `moon3d_*`, `ema_*`), `build/bin/` (`ptm_thermal_sim`, `ptm_camera_sim`,
  `ptm_merge_chunks`), `data/dem/a3gt_v2/` (per-site core `*_R3km_linear.ply` + ring
  `*_ring_Rcore3_Rext*km.ply`), `SESSION-NOTES-*.md`, `TICKET-*.md`.
- **A3GT** outputs: `~/Research/moon/artemis/a3gt/` (many near-name dirs — see registry). Campaign meta
  (manifests, DEM catalogs): `~/dev/heat3d/a3gt_v2_meta/`. Logs: `~/dev/heat3d/logs/`.
- **Moon3D** mesh/VF: `.../OneDrive.../moon/moon3d/` (`moon3d_c_cartesian_NEW_500k_clean.vtk` = 509k
  mesh; the paired `.vf`, ~24 GB, is only needed for *coupled* radiation runs — illumination-only runs
  need no VF). Baseline **runs**: `~/dev/heat3d/moon3d_verify/` (`radoff/` = illumination / no-rad
  do-no-harm baseline; `coupled/` = radiation-coupled). Baseline **figures**: `$PTM/docs/figures/moon3d_500k/`.
- **EMA conference property tables** (2026-07-04): `.../OneDrive.../EMA/science/tables/asteroid_properties/`
  — regenerable MEASURED/ASSUMED/UNVERIFIED property tables (source `asteroid_properties.yaml` +
  `gen_tables.py` -> MD/TeX/PDF) for all 6 flyby targets. PDF compile is blocked by a machine-wide
  MiKTeX admin-gate (memory `miktex-admin-update-check-gate`), not a content bug. See memory
  `ema-conference-tables-and-spin-provenance` for the full per-body provenance breakdown.
- **EMA asteroids**: `.../OneDrive.../EMA/science/{thermal,camera}/<target>/` — `flyby/` is canonical;
  `flyby_preHapke/`, `flyby_buggyGPU/`, `flyby_topoonly/` are regression/variant dirs (269justitia uses
  orbit/shape-variant subdirs instead of `flyby/`). **Noisy camera MP4 convention (fixed 2026-07-03)**:
  canonical full-length (49f/12fps, matching the clean flyby MP4s) noisy movies live at
  `camera/<t>/flyby/noisy/noisy_ir.mp4` (IR3S thermal), `camera/<t>/flyby/noisy/noisy_visible.mp4`
  (IR3S's OWN reflected channel — NOT CNAC), and `camera/<t>/flyby/visible/noisy/noisy_visible.mp4`
  (CNAC visible — the **science keeper**). `ema_camera_noise.py`'s MP4 assembly still buries the
  movie in whatever dir it just noised (unfixed tool bug, flagged for Ada — see memory
  `ema-noisy-mp4-shortframe-bug-and-fix`), so any FUTURE noise pass must explicitly re-point/copy
  the MP4 to these canonical paths, not just run the tool and trust its default `--output-dir`.

**Baseline registry** — resolve the comparison baseline from here (confirm the ⚠️ ones with the user):
- **A3GT A03MP026** → canonical baseline `~/Research/moon/artemis/a3gt/sites_v2_3km_corering_postfix/A03MP026`.
  ⚠️ **2026-07-03: `view_factors.hmat` reuse is BROKEN TWO WAYS — do NOT assume it's usable.**
  (1) The **current** `data/dem/a3gt_v2/A03MP026_..._R3km_linear.ply` was regenerated Jun 12 (after the VF)
  and now has 2,397,890 faces vs. the VF's 1,796,406 — use the baseline's own saved
  `sites_v2_3km_corering_postfix/A03MP026/mesh.ply` (1,796,406 faces) as `import_file` instead to match.
  (2) **Even with matching dimensions, the VF file itself is EMPTY** — confirmed by parsing it directly:
  32-byte header (rows=cols=1,796,406, **nnz=0**) + an all-zero row-pointer array, total bytes match exactly,
  so it's a well-formed but genuinely empty sparse matrix (radiation exchange silently no-ops if reused).
  Root cause of both is a likely-degenerate June 10 VF build. CONFIRM the VF's `Non-zeros` in the load log
  is `> 0` (not just dimension-matched) before trusting any A03MP026 radiation-coupled run, including
  whether the June baseline's own results ever had real radiation exchange. See memory
  `ptm-a3gt-vf-reuse-and-baseline-id` for the full finding + options. Ring **81.67 km**; **1440**
  timesteps/window (hourly, 30-day); Hayne albedo (`incidence_angle_albedo: false` on the current binary).
  ⚠️ **Window schedule is ambiguous — 8 windows (Y00/05/09/14) vs 16 (adds Y02/07/12/16) — CONFIRM with the
  user before running.**
- **Moon3D illumination do-no-harm** → baseline `~/dev/heat3d/moon3d_verify/radoff/`; mesh
  `moon3d_c_cartesian_NEW_500k_clean.vtk`; compare figures vs `$PTM/docs/figures/moon3d_500k/`.
- **EMA Chimaera** → `.../EMA/science/thermal/623chimaera/flyby/` is CURRENT as of 2026-07-04 ~14:51
  (terminator-fix binary **`79f091e`**, CPU, chi=2.7 — see manifest
  `.../EMA/science/logs/ema_chimaera_only_terminatorfix_manifest.json`). This SUPERSEDES the
  conduction-fix `5bfbdbe` generation this doc previously listed as current (79f091e reverted a
  terminator-cold regression that had crept into 5bfbdbe, while KEEPING the day/S/year/Qb-forwarding +
  equilibration-duration-cap fixes) — found/confirmed 2026-07-04 during the pretty-plots figure revamp,
  see memory `ema-chimaera-pretty-plots-figures-2026-07-04`. Prior generations archived non-destructively,
  do not confuse with current: `flyby_preHapke/` (pre-Hapke regression baseline), `flyby_topoonly/`
  (topology-fix validation render), `flyby_preSmoothIC_20260703_225821/` (seam-fixed-only,
  pre-conduction-fix), `flyby_preConductionFix_20260704_121627/` (seam-fixed c70862b generation), and
  `flyby_preTerminatorFix_20260704_142231/` (the conduction-fix 5bfbdbe generation, immediately prior to
  the current one). 5-node TI sweep grid `.../thermal/623chimaera/sweep/` (TI=[32,56,100,178,316] SI,
  CURRENT = 79f091e/CPU/chi=2.7, 15/15 cells clean); off-grid TI-recovery validation (TI=75,200,
  independent forward sims, NOT grid nodes) `.../thermal/623chimaera/offgrid/{ti075,ti200}/` + fit
  `.../thermal/623chimaera/ti_fit_offgrid/` (both truths recovered within 68% CI on the current gen:
  TI=75→77 [47,118], TI=200→194 [156,255]) — see memory `ema-chimaera-offgrid-ti-recovery-validation`
  (methodology; superseded numbers are in `ema-conductionfix-cpu-campaign-2026-07-04`). Canonical
  figures dir: `.../thermal/623chimaera/figures/` (original/low-res) and NEW publication-styled
  `.../thermal/623chimaera/figures_pub/` + `camera/623chimaera/**/*_pub.mp4` (pretty_plots-styled,
  300dpi+vector PDF, 2026-07-04 conference revamp — see memory
  `ema-chimaera-pretty-plots-figures-2026-07-04`, reusable module `$PTM/scripts/ema_pretty_figs.py`).
  **Round 2 (same day, later)**: added `camera/623chimaera/diurnal/chimaera_diurnal_pub.mp4`
  (body-fixed-camera diurnal-cycle render, regenerated from the CURRENT 79f091e data — supersedes
  `chimaera_diurnal_OLD-BUGGY_preTerminatorFix_20260702.mp4`, renamed in place, NOT deleted, from
  the pre-terminator-fix binary); IR3S instrument label renamed "ECAM-IR3S (VNFOV)" → **"EMACS IR
  Cam"** (`IR_CAM_LABEL` constant in `ema_pretty_figs.py`; CNAC untouched; diurnal render carries no
  camera label at all — it's a direct thermal-model render, not instrument-simulated); per-frame UTC
  now a clean `YYYY-MM-DD HH:MM:SS UTC` on every animation frame; found+fixed a bug where
  `savefig.bbox='tight'` was silently cropping video frames to odd non-1920 dimensions (all
  `*_pub.mp4` now regenerated at a clean 1920×N). Round-1's static `figures_pub/snapshot_stills_
  {ir,reflected}.{png,pdf}` were NOT regenerated in round 2 (still show the old IR3S label; out of
  scope — rename was animations-only). See memory
  `ema-chimaera-diurnal-rename-utc-round2-2026-07-04`.
  ⚠️ **2026-07-06: this `*_pub.mp4` flyby set is now LABEL-STALE** — Paul's
  standing convention (memory `crew-instrument-name-emacs-ir-cam`) renamed the
  instrument label again, "EMACS IR Cam" (no slash, above) → **"EMACS/IR Cam"**
  (with slash); `IR_CAM_LABEL` in `ema_pretty_figs.py` was updated to the new
  value but Chimaera's own `flyby/*_pub.mp4` files were never regenerated
  against it (found while restyling Ousha's flyby animations to match, see the
  Ousha entry below — Ousha's new files correctly carry the current label, one
  step ahead of Chimaera's own on this axis). Cheap fix: re-run
  `make_chimaera_animations()`, no data change needed; not yet done.
  **The other 4 EMA targets (Rockox/Westerwald/Moza/Ghaf) are NOT on this terminator-fix
  generation** — see the CPU streaming-cliff landmine above; their `flyby`/`sweep`/`ti_fit` were archived
  pre-conduction-fix and not yet regenerated on either 5bfbdbe or 79f091e. **Ousha IS regenerated** —
  see the dedicated entry below.
- ✅ **EMA Ousha (23871)** → `.../EMA/science/thermal/23871ousha/{flyby,sweep,ti_fit,offgrid,
  ti_fit_offgrid,figures}` is CURRENT as of 2026-07-04 ~16:15 (terminator-fix binary **`79f091e`**,
  CPU, chi=2.7, **`PTM_RAM_ILLUM_MAX_GB=40`** — required + confirmed WORKING: this 81,920-face ico6
  mesh's ~14.8 GB illumination RAM-EXTRACTED, did NOT hit the streaming cliff below, ~70 min
  end-to-end). Terminator spotcheck PASS (0 never-lit faces); 15/15-cell sweep clean (incl. the
  TI=32/316 rows that crash on GPU); off-grid recovery (TI=75→77.7 [69.3,84.7], TI=200→205.7
  [168.3,263.0]) both truths in-CI. Nominal (TI=109.5 SI fixed) surface-T: day-max 192.87 K /
  night-min 52.74 K (carries the ~15-16 K deferred cold-bias caveat, see the coldbias-conclusion
  memory). Stale pre-fix generations retained (`flyby_buggyGPU/`, `flyby_preGenericBodyThermoFix/`,
  `flyby_preHapke/`, `flyby_preSmoothIC_20260703_225821/`, `flyby_preConductionFix_20260704_121627/`,
  `sweep_preTerminatorFix_20260704_150224/`, `ti_fit_preTerminatorFix_20260704_150224/`) — do not
  confuse with current. Found a SECOND Chimaera-hardcoded script (`/tmp/ema_offgrid_fu2.py`, distinct
  from the known `ema_ti_fit.py` bug) — worked around with `/tmp/ema_offgrid_fu2_ousha.py`, not yet
  generalized. See memory `ema-ousha-terminatorfix-ramextract-validation`.
  **Rockox/Westerwald/Moza/Ghaf were explicitly NOT run this pass** — scoped single-target
  validation of the icosphere/RAM-extract fix before Justitia; still on the pre-fix generation.
  ✅ **2026-07-06 pretty-plots restyle + densified TI recovery (Rocky)** (memory
  `ema-ousha-tifit-prettyplots-recovery-densify-2026-07-06`, session note
  `SESSION-NOTES-2026-07-06-ousha-tifit-pretty-plots-recovery-densify.md`).
  `ti_fit/fit_curves.{png,pdf}` restyled with `ema_pretty_figs.setup_style()`
  (pretty_plots.planetary) — same content/data (do-no-harm verified: regenerated
  `fit_results.json` numerically identical to on-disk). `ti_fit/fit_recovery.{png,pdf}`
  DENSIFIED from 2 points to a **15-point, 200-seed-ensemble, 68%-CI ison** recovery
  curve spanning the full 32-316 SI grid range: 5 grid-node self-checks (reused sweep
  cells) + the 2 existing off-grid cases (75,200 — reproduced exactly) + **8 NEW
  independent forward sims** (TI=45,65,90,130,150,225,250,280 SI) at
  `.../offgrid_extra/{ti045,...,ti280}/`; combined fit data at
  `.../ti_fit/offgrid_extra/fit_results_{summary,full}.json`. **14/15 truth-in-CI**
  (the 1 exception, TI=32, is a 1-ulp floating-point boundary artifact at the grid's
  own lower edge, not a real recovery failure). Both new figures + `.caption.md` files
  per the clean-figure-style rule; originals archived non-destructively
  (`fit_curves_prePrettyPlots_20260706_142821.png`,
  `fit_recovery_preTaskB_20260706_142821.png`).
  ⚠️ **Landmine found: the live `$PTM` build is NO LONGER the 79f091e binary that
  generated this sweep** — as of 2026-07-06 the working tree has ~1500 uncommitted
  lines of OTHER agents' in-progress diffs (ice-depth, directional-emission/roughness)
  on top of 79f091e; `build/bin/ptm_thermal_sim` md5 is now `8498d91b...`, not the
  campaign's recorded `bd9b785b...`. For the 8 new forward sims, Rocky instead built a
  PINNED, isolated binary via `git worktree add --detach /tmp/ptm_79f091e_pinned
  79f091e` (md5 `4bc4b27b55d75e1c7d9cdb8197c1b2ba`) and validated it BYTE-IDENTICAL
  against the archived `offgrid/ti075` case before trusting it. **Any future
  reproducible re-run of a specific historical PTM3D commit should use this same
  git-worktree-pin pattern**, not the live (possibly-since-modified) `$PTM/build`.
  ⚠️ **Second landmine (general PTM3D bug, not Ousha-specific): `SPICE_KERNELS_DIR`
  is a CMake-configure-time-baked-in ABSOLUTE path** (`CMAKE_CURRENT_SOURCE_DIR}/data/
  kernels`, see `CMakeLists.txt:348` / `config.h.in`), naive-string-concatenated (not
  `std::filesystem::path` joined) in `illumination_model.cpp:1791-1849`'s default
  SPICE fallback init — building `ptm_thermal_sim` from ANY tree other than `$PTM`
  itself (e.g. a worktree, a CI checkout) silently breaks SPICE init unless
  `-DSPICE_KERNELS_DIR="$PTM/data/kernels"` is passed explicitly at configure time
  (the real kernel files there are untracked/local-only, `git ls-files data/kernels`
  shows only `README.md`). Flagged for Ada/Fermi, not fixed in source.
  ✅ **2026-07-06 flyby `_pub` animations, render/encode only (Rocky)** (memory
  `ema-ousha-flyby-pub-animations-2026-07-06`, session note
  `SESSION-NOTES-2026-07-06-ousha-flyby-pub-animations.md`): 6 new
  `camera/23871ousha/flyby/**/*_pub.mp4` files (the 5 Iris asked for +
  1 bonus CNAC-noisy for full parity), restyled to match Chimaera's own
  `_pub` set — same `ema_pretty_figs.render_mp4_from_fits` calls/params via
  a new parallel `make_ousha_animations()` driver, reusing Ousha's existing
  canonical camera FITS frames (no thermal/VF/GPU re-run). All 6 confirmed
  1920×1648/12fps/49f/4.083s h264, exact spec match to Chimaera's 6
  `_pub.mp4` files. Colorbar range: Chimaera's own code already auto-scales
  per-body (`vmin=None`→0.0 floor, `vmax=None`→ that run's own 99.5th
  percentile) — applying the same defaults to Ousha reproduces the identical
  approach, no fixed-vs-auto tradeoff needed. **Ousha's new files correctly
  carry the CURRENT "EMACS/IR Cam" label — one step AHEAD of Chimaera's own
  `_pub.mp4` set, which is now label-stale (see the Chimaera entry above)**.
  Non-destructive, no commit.
  **Addendum (same day, later)**: the 2 noisy SNAPSHOT MONTAGES
  (`flyby/noisy/{noisy_ir,noisy_visible}_snapshots.png`, pre-existing raw
  3-panel CA±2min montages, previously untouched by any pretty-plots pass)
  restyled to match — new `fig_noisy_snapshot_montage()` +
  `make_ousha_noisy_snapshots()`, sourced from the same `flyby/anim/noisy/`
  frames the sibling `_pub.mp4` used, `vmin=0/vmax`=99.5th-pct computed over
  all 49 frames (verified numerically to match the movies' own scale
  exactly: TEMP_K 192.67K / RADIANCE 7.733 — ⚠️ **this specific TEMP_K-based
  vmax is SUPERSEDED, see Addendum 2**), `ca_utc` empirically measured
  (frame_0024, 2032-12-25T16:47:20, range 520.5 km). Kept Chimaera's panel
  LAYOUT/titles, used the current (non-stale) "EMACS/IR Cam" label +
  per-panel UTC stamps. Originals archived non-destructively
  (`*_prePub_20260706_201612.png`). Found+fixed a real bug: forcing
  `savefig.bbox=None` (copied reflexively from the video-frame renderers)
  clipped the suptitle on this single-static-image save — removed the
  override, restored tight-bbox (the same pattern `fig_snapshot_stills`
  already relies on).
  ⚠️➡️✅ **Addendum 2 (2026-07-06/07, Paul review): the IR montage's "noise"
  was a NO-OP — ROOT-CAUSED + FIXED, plus an 80%-fill crop added.**
  `ema_camera_noise.py::add_ir_noise` perturbs the `THERMAL` HDU only, never
  `TEMP_K` — confirmed `TEMP_K` clean-vs-noisy max abs diff = 0.0 on every
  checked pixel. **This means `noisy_ir_pub.mp4` (both Ousha's AND
  Chimaera's — same `render_mp4_from_fits(ext="TEMP_K", ...)` call in both
  `make_ousha_animations`/`make_chimaera_animations`) has the IDENTICAL
  defect and is NOT genuinely noisy either**, unlike this doc's earlier
  entry above implied — flagged for a Paul re-run decision, NOT yet fixed
  in the movies. `RADIANCE` (visible/reflected channel, both the montage and
  `noisy_visible_pub.mp4`/CNAC) was cross-checked and confirmed genuinely
  noisy already (never had this defect). Fix: new `_noisy_ir_temp_field()`
  reconstructs true noisy brightness-T via the NONLINEAR Planck inversion of
  noised THERMAL + a per-pixel clean-frame calibration correction (exact at
  zero noise); a first linearized attempt (`ir_sigma_T`'s own model
  literally applied) diverged to 75,295 K + negatives at cold/low-signal
  pixels, so the nonlinear form is required for numerical sanity
  ([20.5,203.3] K bounded, vs. the linear attempt's [-45,75295] K). Also
  added `_bbox_crop_window()` (true integer-pixel crop, no resampling,
  `interpolation="nearest"`, per-panel body fills ~80%, real per-panel GSD
  annotated: 88×88px@96.9m, 98×98px@86.0m, 88×88px@95.9m for
  CA−2min/CA/CA+2min). See memory `ema-ousha-flyby-pub-animations-2026-07-06`
  ADDENDUM 2 + `SESSION-NOTES-2026-07-06-ousha-flyby-pub-animations.md` for
  the full per-panel noise numbers (incl. a real cold-vs-warm asymmetry
  finding flagged for Fermi). Non-destructive (3 generations archived), no
  commit.
  ✅ **Addendum 3 (2026-07-06/07): Ada's source fix landed in
  `ema_camera_noise.py` (module-level `invplanck()`+`recompute_temp_k()`,
  the exact kernel `_noisy_ir_temp_field` duplicated, cross-validated to
  7.6e-6 K) — campaign-wide noisy-IR re-render EXECUTED.** Chimaera's and
  Ousha's `noisy_ir_pub.mp4` regenerated (now reads corrected `TEMP_K`
  directly, no workaround), visually confirmed genuinely noisy. Ousha's 2
  snapshot montages regenerated (`fig_noisy_snapshot_montage` simplified to
  read `TEMP_K`/`RADIANCE` directly, `_noisy_ir_temp_field` kept unused as
  historical reference), confirmed consistent with the prior
  workaround-based version to 7.6e-6 K. **Correction to the prior
  inventory**: the "raw" (non-pub) `noisy_ir.mp4`/`noisy_ir_snapshots.png`
  across ALL 6 bodies were NEVER actually broken — `ema_campaign_driver.py`'s
  native `make_mp4`/`snapshot_png` (auto-invoked by
  `ema_camera_noise.py::process_path`) compute brightness-T on the fly from
  `THERMAL`, never reading the stale `TEMP_K` HDU; only the 2 `_pub` movies
  + Ousha's montages were ever affected. All 18 movies' underlying FITS
  regenerated in place anyway (`--seed 42`, verified to reproduce existing
  noise bit-for-bit; per-dir `--mode` read from each file's own `NOISEMOD`
  header: flyby=`both`, offgrid=`ir`); Rockox/Westerwald/Moza/Ghaf's
  `flyby/visible/noisy/` (stuck at a stale 3-frame short movie) fixed to 49
  frames + real noise in the same pass. 20 pre-existing deliverables
  archived (`*_preNoiseFix_20260706_214042.*`) before overwrite. Cross-body
  clean-vs-noisy `TEMP_K` σ: Chimaera 19.35 K, Ousha 22.96 K (both
  legitimate) — **but see the flagged data-gap finding immediately below
  before trusting the other 4 bodies' numbers.** Caption/honesty note (cold-
  limb non-Gaussian floor pileup, un-debiased single-look retrieval,
  spec-assumed NEDT) at
  `.../EMA/science/camera/NOISY_IR_NOISE_MODEL_CAVEATS.md` + a short
  `noisy_ir.caption.md` in every directory holding a noisy IR product (off-
  frame, per the clean-figure rule). See memory
  `ema-noisy-ir-campaign-refit-2026-07-06`,
  `SESSION-NOTES-2026-07-06-ema-noisy-ir-campaign-refit.md`. No commit.
  ⚠️ **MAJOR FINDING (pre-existing, NOT caused by this fix or any prior
  session): Rockox/Westerwald/Moza/Ghaf have NO real thermal-IR camera data
  at all.** Found while computing the σ report above: `THERMAL`/`TEMP_K`
  are all-zero on every on-body pixel of every clean frame for all 4 bodies
  (confirmed per-frame for Rockox, spot-checked for the other 3).
  Root-caused via `render.log`: these 4 bodies' only camera render used
  `examples/cameras/ema_cnac.yaml` (**visible-only CNAC**) — there is no
  `flyby/anim/` directory for any of them, unlike Chimaera/Ousha (each has a
  SEPARATE real `flyby/anim/` rendered with
  `examples/cameras/ema_ecam_ir3s_rockox.yaml`, the actual IR3S camera,
  confirmed non-zero `TEMP_K` 80-260 K). **Their `noisy_ir.mp4`/
  `noisy_ir_snapshots.png` have ALWAYS been (before AND after every fix to
  date) a pure noise-floor artifact** — a small Gaussian added to a zero
  signal, clipped at 0, with the per-movie auto-`vmax` (dominated by that
  tiny noise ceiling) making it LOOK like a normal thermal image. Needs an
  actual new IR3S-camera `ptm_camera_sim` render for these 4 bodies (a
  compute job, not a post-processing fix) to produce real data — flagged
  for Paul, not built. Do not present these 4 bodies' "noisy IR" products
  as genuine science without that follow-up render.
  ✅ **2026-07-06 QUARANTINED (Paul's option B, Rocky)** (memory
  `ema-noisy-ir-campaign-refit-2026-07-06` addendum, session note
  `SESSION-NOTES-2026-07-06-ema-null-ir-quarantine.md`). Re-confirmed all 4
  bodies independently (astropy read of all 49 clean frames each — `TEMP_K`/
  `THERMAL` all-zero, global max 0.0). Moved (not deleted) all 24 live IR
  deliverables (2 locations × 3 files × 4 bodies: `noisy_ir.mp4`,
  `noisy_ir_snapshots.png`, `noisy_ir.caption.md` under both
  `flyby/visible/anim/noisy/` and `flyby/visible/noisy/`) into
  `camera/<body>/flyby/_QUARANTINE_null_IR_20260706_221642/` (relative
  structure preserved), each with a `README.md` explaining the null-signal
  finding + the deferred real-IR-render path (option A, not run). Live
  `noisy_ir.caption.md` replaced with a one-line pointer stub to the
  quarantine README. Campaign-level
  `.../EMA/science/camera/NOISY_IR_NOISE_MODEL_CAVEATS.md` now opens with a
  notice that only Chimaera + Ousha have real IR. Confirmed untouched:
  Chimaera/Ousha (zero files touched), the 4 bodies' CNAC visible products
  and source FITS (`RADIANCE` real, only IR HDUs null), and the already-
  archived `*_preNoiseFix_20260706_214042.mp4` (excluded from the move,
  left in place). Non-destructive, no commit.
- ✅ **EMA Justitia (269) proxops sim** → `.../EMA/science/{thermal,camera}/269justitia/proxops/` is the
  first (and only) proximity-ops product, built 2026-07-04 (binary `79f091e`, CPU, chi=2.7, DAMIT shape
  574v/1144f, TI=100 SI nominal at T=0, 66-frame IR3S sequence over one ~18.7h spacecraft-orbit window).
  Config `$PTM/examples/justitia_proxops.yaml`. Obs-window surface T 101.76-243.93 K (carries the ~15-16 K
  deferred cold-bias caveat, see `ema-asteroid-coldbias-investigation-conclusion`). **The
  `data/horizons_cache/bodies/269_Justitia_(A887_SA)/` directory was reconstructed from scratch** (shape.obj
  recovered from an archived `mesh.ply`, `body_config.yaml`/spin-PCK/metakernels built fresh, `ephemeris.spk`
  = a copy of the user-provided `Justitia_20240612.bsp`) — it never actually existed before despite being
  referenced by the canonical `justitia_damit_real_real.yaml`; correct SPICE naif_id is **`20000269`**
  (8-digit — the config's `2000269` does NOT match either provided SPK). ⚠️ **Landmine:**
  `thermophysical.constant_thermal_inertia > 0` unconditionally zeroes `chi` regardless of config (by
  design, undocumented) — any config wanting BOTH a nominal-TI number AND chi>0 must set
  `constant_thermal_inertia: 0` and specify ks/kd/rhos/rhod/cp0 directly instead. Two new animations at
  `.../camera/269justitia/proxops/justitia_proxops_{camera_view,coverage_accum}_pub.mp4`; the latter uses
  NEW reusable `render_coverage_accumulation_animation()` in `$PTM/scripts/ema_pretty_figs.py`. Existing
  `269justitia/{circular-no-obliq,eccentric-obliq,cicular-noobliq-shape,ecc-obliq-shape}` variants
  untouched. See memory `ema-justitia-proxops-sim-build-2026-07-04` for full detail (P_orbit derivation,
  the multi-phase-SPK MIDGM-plateau judgment call, the IR3S frame-filling-FOV finding).
  ℹ️ **"MIDGM" filename note:** "midgm" in `ema_reftraj_khamis_proxops_midgm_v2.bsp` names the
  assumed **G·M (mass)** of Justitia used to COMPUTE the trajectory (Justitia's real GM is `n.a.` in
  Horizons — genuinely unmeasured); low-GM/high-GM SIBLING TRAJECTORY FILES do not exist anywhere
  under `EMA/trajectory/` (inventoried — see memory `ema-justitia-proxops-trajectory-inventory-
  2026-07-04`; the tree's only other `.bsp`s are an unrelated EMTG interplanetary-cruise family).
  **Resolved (2026-07-04, Hayne):** that missing-sibling-files axis is NOT being used — the imaging
  campaign is organized around the 3 altitude tiers **within the one available file** ("High/Mid/Low
  orbit," no "GM" in the names to avoid the collision), which is what Rocky's analysis found from the
  start. Do not re-relitigate this; the phases figure `.../camera/269justitia/proxops/phases/
  altitude_profile.png` reflects the final naming.
  ✅ **2026-07-04 smooth-thermal 4-target campaign COMPLETE** (memory
  `ema-justitia-proxops-smooth-thermal-campaign-2026-07-04`): ROOT-CAUSED the camera thermal-realism
  gap (`ptm_camera_sim`'s `--smooth-shading` only interpolated the REFLECTED-channel normal+albedo,
  never thermal), Ada landed the fix (`ptm_camera_sim` md5 `7004db7169839bc76557967a0ee569c4` —
  additive `TEMP_K_SMOOTH` FITS extension, Gouraud-interpolated per-vertex T, sharp terminator via
  `direct_solar`), Rocky executed the full re-render: 3 NEW thermal solves (High 254.5km/79.2h, Low
  62.26km/9.60h, Descent/close-approach 33.5km-min) + reused Mid's existing thermal (97.10km/18.70h)
  + 4 camera-render sequences (66 frames each, 100% frame-filling at every phase) + 8 new animations
  under `.../camera/269justitia/proxops/{orbit_high,orbit_mid,orbit_low,descent}/` with an
  auto-detecting `TEMP_K_SMOOTH`-vs-`TEMP_K` picker + an honesty caption on every frame
  (`ema_pretty_figs.py`, `HONESTY_CAPTION`/`_pick_thermal_ext`/`make_justitia_proxops_phase_
  animations`). **Coverage/GSD conference story confirmed:** High orbit 1080/1144 faces (94.4%) at
  42 m/px vs. Low orbit 176/1144 (15.4%) at 10.3 m/px and the descent 208/1144 (18.2%) down to
  5.5 m/px at the 33.5 km closest point — wide-orbit/coarse vs. narrow-orbit/fine tradeoff, visually
  confirmed (contiguous imaged band vs. scattered triangular patches) in the coverage-accum renders.
  Original flat `proxops/` MIDGM product (thermal + camera + its 2 original MP4s) is UNTOUCHED.
  ✅ **2026-07-05 coverage-accum SWATH FIX** (memory `ema-justitia-coverage-swath-fix-2026-07-05`,
  `SESSION-NOTES-2026-07-05-justitia-coverage-swath-fix.md`): the coverage-accum animations above
  were per-FACE_IDX (whole facet "pops" covered on first pixel hit) — reworked to genuine sub-facet
  swath accumulation (adaptive mesh subdivision by facet size + exact ray-plane barycentric from
  per-pixel LAT/LON + per-facet-per-frame AABB-overlap fill, NOT nearest-point matching, which was
  prototyped and rejected for looking speckled not contiguous). Found+fixed a real bug mid-task (an
  earlier centroid-inside-box test silently dropped 84% of genuine imaging events on coarse facets;
  switched to AABB-overlap, verified exact match to the ground-truth facet-count fractions above).
  All 4 phase coverage-accum MP4s regenerated in swath mode; **89.5-100% of ever-touched facets are
  marked only PARTIALLY covered** (not whole-triangle fill) across all 4 phases — direct evidence of
  genuine sub-facet resolution. New area-weighted coverage % (24.3%/4.2%/2.1%/1.5% for
  high/mid/low/descent) is the more honest metric and intentionally much lower than the old
  facet-count % above (a single pixel no longer "fills" a multi-km facet). `render_coverage_
  accumulation_animation()` reworked in place; old body preserved as `render_coverage_accumulation_
  animation_facet_legacy()`. Facet-based predecessor MP4s archived non-destructively (2 generations
  per phase: the true original + an intermediate buggy first-fix attempt). NOT committed (Paul
  holding camera work uncommitted).
  ✅ **2026-07-05 coverage-accum TEXTURE-MAPPED RENDER (round 2)** (memory
  `ema-justitia-coverage-texture-render-round2-2026-07-05`,
  `SESSION-NOTES-2026-07-05-justitia-coverage-texture-render-round2.md`): Paul reviewed the round-1
  swath fix above and found it STILL looked like "scattered sliver-like triangles," correctly
  diagnosing the root cause as the `matplotlib`/`mplot3d` `Poly3DCollection` polygon renderer itself
  (approximate depth-sort, worse with more/thinner sliver polygons), not the accumulation math.
  Reworked to a texture-mapped PyVista/VTK render (true z-buffer): round-1's sub-facet accumulation
  is REUSED UNCHANGED (area_pct/facet_pct_legacy reproduce round-1 numbers exactly, confirming only
  the renderer changed) and baked into a single equirectangular raster texture (antimeridian seam
  fixed via per-face longitude unwrap + unwelded mesh + texture wrap=REPEAT, validated with a
  synthetic gridline test) mapped onto the whole body in one draw call per frame. Found a real
  PyVista/VTK quirk (off-screen `screenshot()` doesn't auto-render after camera/texture/point
  updates — needs an explicit `.render()` first) and a real DATA finding that CORRECTS round 1's
  memory: per-pixel LATITUDE/LONGITUDE is NOT finely continuous within a single frame (~92-98% of
  facet-frame pairs have <=8 distinct values regardless of pixel count) — round 1's "continuous
  variation" claim was very likely measured across frames (parallax), not within one. Renamed
  round-1's polygon function to `render_coverage_accumulation_animation_subfacet_polygon_legacy()`;
  new `render_coverage_accumulation_animation_texture()` is now the live `render_coverage_
  accumulation_animation` (module alias, no call-site changes needed). All 4 phase coverage-accum
  MP4s regenerated (1920x1080/8fps/66f/8.25s, same spec); before/after stills at both a low- and
  high-coverage phase confirm the sliver look is gone (smooth solid silhouette, no wireframe
  artifacts, imaged region reads as one connected blocky/staircase-edged swath, not confetti). NOT
  committed (Paul holding camera work uncommitted).
  ✅ **2026-07-05 coverage-accum TRUE-PER-PIXEL GSD (round 3, the ROOT FIX)** (memory
  `ema-justitia-coverage-truegsd-round3-2026-07-05`,
  `SESSION-NOTES-2026-07-05-justitia-coverage-truegsd-round3.md`): new `ptm_camera_sim` binary
  (md5 `25866a03673bbf743f635be1b51fca13`) writes exact per-pixel `SURF_X_KM/SURF_Y_KM/SURF_Z_KM`
  body-fixed hit points (confirmed: a 54,201-pixel facet now yields 54,201 DISTINCT points, vs.
  the SAME facet-frame's LATITUDE/LONGITUDE fields, which STILL reduce to only 6 distinct values
  — round 2's finding stands, sidestepped not fixed). All 4 phases RE-RENDERED on the new binary
  (CLI reconstructed from the 2026-07-04 campaign memory verbatim, since no script/manifest
  recorded it — byte-verified reproduction: CAM_X/Y/Z/FOV/UTC exact match, FACE_IDX exact,
  TEMP_K exact, old pre-SURF frames archived to `frames_preSURF_20260705_111207/` not deleted).
  Round 1/2's sub-triangle-AABB accumulation REMOVED entirely; new
  `render_coverage_accumulation_animation_texture_truegsd()` bins the real hit point directly
  into texels via the task's own `r/lat/lon` formula (round 2's texture renderer preserved as
  `..._texture_subfacet_legacy`). Texture resolution now tuned PER PHASE from MEASURED true GSD
  (not assumed): High 4096 (38.9 m/texel vs measured ~40-43m), Mid 12800 (12.45 m vs ~12-13m),
  Low/Descent 16384 (9.73 m — a deliberate ceiling, coarser than Low's ~6-7m and Descent's
  confirmed 2.08 m closest-approach minimum, honestly reported not hidden). Before/after +
  zoom stills confirm the fix reads as genuinely continuous with smooth curved swath boundaries
  in all 4 phases (dramatic improvement over both round 1 and round 2, especially visible in
  Orbit Mid). **Found+investigated a real, quantified residual**: a fine texel-scale speckle at
  swath EDGES (measured 10.6% local hole fraction near a boundary vs 0.02% in swath interiors,
  Orbit High) — confirmed genuine via native-resolution array inspection (not a PyVista
  render-setting artifact; tested interpolate/mipmap/MSAA/SSAA/4x-supersampling, none removed
  it), root-caused to marginal pixel-density-vs-texel-density at single-pass/oblique boundaries.
  A candidate morphological-closing fix was tested and REJECTED (it fabricated +11.8 percentage
  points of never-imaged coverage) — NOT applied; delivered data is the raw unprocessed
  per-pixel accumulation. **This SUPERSEDES round 2 as current** — round 1/2 MP4s archived
  non-destructively (`*_facetbased_preSwathFix_<stamp>.mp4`, up to 5 generations per phase now).
  NOT committed (Paul holding camera work uncommitted).
  ✅ **2026-07-05 round 4 (Paul-approved refinements on the round-3 pipeline)** (memory
  `ema-justitia-coverage-round4-refinements-2026-07-05`,
  `SESSION-NOTES-2026-07-05-justitia-coverage-round4-refinements.md`): (1) trimmed all 4 phase
  coverage_accum MP4s' on-animation text to ONLY UTC + % coverage (dropped `HONESTY_CAPTION` +
  verbose wording; `area_pct`/`facet_pct_legacy` numbers unchanged — text-only edit, verified);
  (2)+(3) NEW `render_coverage_accumulation_animation_fullperiod_truegsd()` builds ONE
  full-mission coverage animation (264 frames, all 4 phases in confirmed-chronological
  High→Mid→Low→Descent order, HIGHEST-RESOLUTION-WINS per texel — a later finer-GSD hit
  overwrites, a later coarser hit never regresses, unit-tested before trusting on real data) at
  `.../camera/269justitia/proxops/fullperiod/justitia_fullperiod_coverage_accum_pub.mp4`
  (16384-wide/9.73 m texel, same practical ceiling as round 3's Low/Descent). GSD =
  `|SURF_pixel-CAM|*IFOV` (IFOV=17µm/102.8mm=165.37 µrad, cross-checked vs FOV_X/640=165.22
  µrad) independently reproduces round 3's measured per-phase GSDs incl. Descent's 2.077 m
  closest-approach value; found `DIST_KM` was ALREADY byte-identical to the computed
  `|SURF-CAM|` range (validated cross-check, not substituted). (4) NEW
  `fig_coverage_resolution_histogram()` — % of surface vs. best achieved GSD, 4 visually
  distinct tiers matching the 4 discrete altitude regimes flown, at
  `.../fullperiod/coverage_resolution_histogram.{png,pdf}`.
  ⚠️ **Real finding, flagged not silently patched**: the full-period product's
  `area_pct=37.64%` is much lower than High orbit's OWN per-phase `area_pct=81.89%`, even
  though `facet_pct_legacy=94.41%` matches exactly (same facets). Root-caused (not assumed) to
  the 16384-wide/9.73 m combined grid being ~4x FINER than High/Mid orbit's own native pixel
  spacing (~40 m/~12 m) — a coarse pixel only paints ONE fine texel of the ~16 its true
  footprint should cover, so High's genuine, smoothly-filling-at-its-own-resolution coverage
  reads as sparse/speckled at this finer shared grid (visually confirmed: salt-and-pepper holes
  in the t=30s full-period still). Quantified: the histogram's coarsest bin (27-43 m) credits
  only ~11.2% of area vs. High's own 81.89% — an ~7.3x reduction consistent with the ~4x
  grid-mismatch hypothesis. NOT fixed in code (no single grid size avoids bias in both
  directions — a coarser shared grid would instead OVER-claim fine-GSD area, the same
  fabrication direction round 3 already rejected); kept the conservative, texel-exact,
  never-fabricates accounting. **Read the histogram's "62.4% never imaged" with this caveat** —
  it does not mean 62% of the body was never seen by the camera, only that it isn't confirmed
  at this specific fine binning; `facet_pct_legacy` remains the texel-grid-independent
  "ever imaged" statistic. Flagged as a pending decision for Paul (keep as-is vs. an alternative
  per-phase-native-weighted accounting). NOT committed; all 4 trimmed animations + the
  full-period animation + histogram are additive/non-destructive outputs.
  ✅ **2026-07-05 round 5 (Paul-approved: footprint-fill + cadence root-cause fix)** (memory
  `ema-justitia-coverage-round5-footprint-fill-cadence-fix-2026-07-05`, `SESSION-NOTES-2026-07-05-
  justitia-coverage-round5-footprint-fill-cadence-fix.md`): **Fix 1 (footprint-fill)** — each
  imaged pixel now fills its true emission-elongated ~GSD footprint (new `_footprint_fill_radii`/
  `_footprint_splat_indices` in `ema_pretty_figs.py`, using the already-present `EMI_ANG` FITS
  field, no new camera binary) instead of a single texel/point; verified via 189 boundary-crop
  measurements at High orbit (mean fill 50.9%->83.2%). **Fix 2 (cadence root-cause)** — the
  original `N=66=360/(FOV_H*0.9)` cadence design conflated the camera's own FOV angle with the
  altitude-dependent body-center angular footprint width (only valid when altitude~R; Justitia's
  real altitude/R ratios span ~11 down to ~1.3), confirmed via direct per-pixel texel-overlap
  MEASUREMENT (not just theory) at the original cadence: High/Mid healthy (0 zero-overlap pairs),
  Low marginal (mean 23.6%/min 5.6%, no literal gaps), **Descent severe (60/65=92.3% of
  consecutive frame pairs shared ZERO texels** — a non-orbital close-approach trajectory, worse
  than orbit_low despite the task naming orbit_low). Re-rendered on the UNCHANGED SURF-writing
  binary (md5 `25866a03673bbf743f635be1b51fca13`): **orbit_low N=66->132** (uniform, same window;
  AFTER overlap mean 48.67%/min 36.25%, 0/131 zero-overlap) and **descent N=66->1072**
  (motion/footprint-ADAPTIVE, non-uniform — more frames where footprint is small/motion is fast;
  an intermediate N=380 only partially fixed it, 35.4% zero-overlap remaining; AFTER overlap mean
  52.26%/min 19.88%, **0/1071 zero-overlap — fully closed**, visually confirmed contiguous
  criss-crossing swath bands replacing the prior scattered-confetti look). Methodological finding:
  a calibrated exact-spherical-geometry footprint model over-predicts real achieved overlap by
  ~2-3x — real cadence design on this pipeline needs a render-then-measure empirical loop, not a
  geometric estimate alone. Regenerated all 4 per-phase animations + the full-period
  highest-resolution-wins animation (1336 frames now, was 264) + the histogram:
  **full-period `area_pct` 37.64%->89.67%**, `facet_pct_legacy` 94.41%->96.94%, never-imaged
  62.4%->10.33% — DIRECTLY closes round 4's own flagged grid-mismatch under-credit finding (was
  explained/flagged there, is now actually fixed). Fix 1 found to be a no-op at Low/Descent's own
  native texel scale (their GSD is already finer than 1 texel there) — their coverage gains are
  Fix 2's doing, reported honestly not conflated. Old 66-frame orbit_low/descent camera frames +
  the round-4 fullperiod raster/histogram archived non-destructively. NOT committed
  (`ema_pretty_figs.py` still untracked). Flagged for Ada: the PyVista coverage-accumulation
  renderer takes a real, multi-tens-of-minutes wall-time cost at N>1000 frames (2801 s for
  descent's 1072-frame animation, 3506 s for the 1336-frame full-period one) — not pathological,
  just a genuine per-frame-texture-churn cost worth knowing before requesting even finer cadence.
  ✅ **2026-07-05 round 6 (sub-spacecraft-tracking camera + green footprint box, code+cheap-
  verify only, full renders NOT run — handed to Ada)** (memory
  `ema-justitia-coverage-round6-tracking-footprintbox-2026-07-05`, `SESSION-NOTES-2026-07-05-
  justitia-coverage-round6-tracking-footprint-box.md`): **Improvement 1** — new
  `_tracked_camera_pose()` re-points the render camera every frame so the real spacecraft's
  sub-spacecraft/nadir point stays centered (fixes "imaged region rotates behind the limb").
  Quantified the round-5 bug via pure geometry over ALL 1336 real frames (no rendering): the OLD
  fixed camera had the target >90 deg from its view axis (literally behind the limb) for
  37.7-50.8% of frames in EVERY phase; the NEW tracked camera is exactly 0 deg by construction.
  **Improvement 2** — new `_frame_footprint_hull_points()` draws a green convex-hull outline of
  the CURRENT frame's real imaged footprint (image-pixel-space hull -> real SURF_X/Y/Z lookup),
  replaced every frame. Found every sampled real Justitia proxops frame (all 4 phases) is
  100%-valid (whole sensor on-body), so the hull always degenerates to the disk's silhouette —
  correct given the real data, not a bug. Both applied to BOTH renderers
  (`render_coverage_accumulation_animation_texture_truegsd`/alias `render_coverage_
  accumulation_animation`, and `render_coverage_accumulation_animation_fullperiod_truegsd`),
  default-on (`track_subspacecraft=True, draw_footprint_box=True`), **zero call-site changes
  needed** at any existing entry point. Verified CHEAPLY per task instruction (unit tests of 3
  new helpers, a 10-frame low-res real-data render + a flags-off regression twin reproducing
  round 5 exactly, a 6-frame synthetic 2-phase fullperiod smoke test, and the full-phase-length
  pure-geometry angle check above) — **no full-length animation rendered this round**; flagged
  for Ada a modest additional per-frame cost (one `ConvexHull` call ~35 ms + one VTK line-actor
  add/remove) on top of round 5's already-flagged texture-churn cost at N>1000 frames. NOT
  committed (`ema_pretty_figs.py` still untracked).
  ✅ **Justitia roughness study (Smith 1967 directional emission, `TICKET-2026-07-04-justitia-
  roughness-smith-emission.md`) — COMPLETE & Fermi-VETTED through Phase 3 OSSE (2026-07-05); pending
  only Paul's T0.1b scope decision. Overview memory: `ema-justitia-roughness-study-complete-2026-07-05`.**
  Phase 2 output root `.../EMA/science/thermal/269justitia/proxops/roughness_phase2_forward/{thermal,
  analysis,figures,fisher_forecast,osse_phase3}/` (+ camera QC frames under the
  `camera/.../roughness_phase2_forward/camera_qc/` sibling) on `ptm_camera_sim` md5
  `1726c8fb7f4c61d989f2f165321fd5ce` (Ada's directional-emission build: `--emission-rms-slope-deg`,
  `BT_K`/`BT_K_OBS` HDUs). 9-cell Gamma(56/100/178 SI) x Albedo(0.015/0.03/0.06) thermal grid x
  theta_bar(0/10/20/35 deg) camera grid, rendered over the existing High-orbit proxops epochs only
  (Phase 1's leverage-carrying geometry). Guard-mask fraction 0.000% everywhere (no STOP on that
  gate). Phase 2's original per-facet-SNR detectability calc was WRONG figure of merit (Fermi);
  Rocky's Phase-2.5 pooled Fisher forecast (2026-07-05, `fisher_forecast/{fisher_forecast_results.json,
  fisher_population.csv}`, figs 4-6) found sigma(theta_bar)=3.6-4.0deg/5.1-9.6sigma detection — but
  Fermi then found THAT forecast's noise model itself wrong (single-look NEDT applied to a facet-
  AVERAGED observable, no n_pix division) and ordered Phase 3 as a confirmatory OSSE with the
  corrected noise model.
  **Phase 3 (2026-07-05, Rocky, CPU/Python only, no new renders):** self-contained MCMC (no emcee
  anywhere on this machine) injection-recovery at theta_bar=20/35deg through the real 1690-obs
  population + corrected noise (`sigma_i=sqrt((NEDT/sqrt(n_pix/psf_corr))^2+sigma_model^2)`,
  sigma_model swept {1,2,3,5}K). **All 8/8 configs bracket the injected truth at 68% CI**;
  rho(theta_bar,Gamma)=-0.25 to -0.31 (confirms separation). **BUT the exact recompute (3
  cross-validated methods) gives detection significance 15-118 sigma / sigma(theta_bar)=0.35-1.4deg —
  an ORDER OF MAGNITUDE tighter than Fermi's own stated "~5-8.5 sigma/2-6deg" correction headline** —
  flagged for Fermi's re-vet (same pattern as the earlier Rank-1 vs. Fermi-preliminary gap), NOT
  silently substituted either way. **Separately, the misspecification arm (isothermal-chi fit vs. an
  h1a-adapted anisothermal truth) found a SEVERE, lower-bound-only theta_bar-HIGH bias: +5.0deg
  (truth=35deg) to +19.5-19.9deg (truth=20deg) — recovered theta_bar PINS at the model's usable
  ceiling in ALL 4 tested configs, because the true anisothermal BT excess can exceed the isothermal
  model's ENTIRE achievable dynamic range (Wien-regime band-radiance effect at these ~137K
  temperatures) — isothermal-chi may be structurally INADEQUATE at this amplitude.** Flagged clearly
  for Paul (the T0.1b anisothermal-camera-build decision), NOT built. New outputs:
  `roughness_phase2_forward/osse_phase3/{osse_main_recovery_results.json,
  osse_main_posterior_samples.npz, osse_misspec_aniso_bias_results.json}`,
  `fisher_forecast/fisher_forecast_corrected_noise.json`; fig4/fig5 CAPTIONS honesty-fixed (numbers
  unchanged, pre-fix archived as `..._preHonestyFix_20260705.*`), new fig7 (OSSE recovery +
  misspecification bias). See memory `ema-justitia-roughness-phase3-osse-2026-07-05` (Phase 3, this
  result), `ema-justitia-roughness-fisher-pooled-rank1-2026-07-05` (Phase 2.5 forecast being
  confirmed/revised), `ema-justitia-roughness-phase2-forward-gating-2026-07-05` (the earlier
  overturned STOP), and `SESSION-NOTES-2026-07-05-justitia-roughness-phase3-osse.md` /
  `SESSION-NOTES-2026-07-05-justitia-roughness-fisher-pooled-rank1.md` for full detail. **Status:
  Fermi FINAL vet COMPLETE (2026-07-05, "sound-with-caveats") — every load-bearing number independently
  reproduced. RECONCILED honest framing: roughness DETECTABLE (>=15 sigma formal, noise-model-dependent
  — NOT the physical limiter) + cleanly SEPARABLE from thermal inertia (|rho|~0.3, not degenerate);
  precision is MODEL-FIDELITY-limited (the 15-118 sigma / 0.35-1.4deg is the correct pixel-averaging
  consequence, not a headline). The misspecification theta_bar-HIGH bias (+5-20deg, structurally
  inadequate at theta_bar=35deg w/ 10x margin) is CONFIRMED REAL. HARD RULE: do NOT publish a
  quantitative slope from the isothermal pipeline. PENDING PAUL: the T0.1b anisothermal-build decision
  (needed only for a quantitative slope; a feasibility/separation talk is complete without it). See
  overview memory `ema-justitia-roughness-study-complete-2026-07-05`.**
  ✅ **2026-07-06 conference summary figure (clean 2-panel replacement for fig1-7)** — Paul asked for
  ONE simple figure per the new figure-style rule (below): `.../roughness_phase2_forward/figures/
  roughness_summary_2panel.{png,pdf}` + `roughness_summary_2panel.caption.md` (same dir). Panel (a)
  Delta T_b (rough-smooth) vs. solar incidence, theta_bar=10/20/35deg (reuses
  `binned_deltabt_vs_incidence.csv` unchanged); panel (b) a clean redraw of fig4's theta_bar-Gamma
  Fisher covariance ellipse (rho=-0.21 to -0.26, matches Paul's |rho|~0.3). `$T_\mathrm{b}$` notation
  throughout (not "BT"); all caveats (isothermal-lower-bound bias, forecast-not-posterior) moved
  off-figure into the caption file. fig1-7 verified UNCHANGED (non-destructive); no commit. See memory
  `ema-justitia-roughness-2panel-summary-figure-2026-07-06`.
  ✅ **2026-07-06 depth-profile thermal study — SCOPED+BUILT+RUN (thermal only), STOPPED before
  camera/plots pending Fermi vet** (memory `ema-justitia-depth-profile-thermal-scope-build-run-
  2026-07-06`, `SESSION-NOTES-2026-07-06-justitia-depth-profile-thermal-scoping-build-run.md`).
  ⚠️ **PTM3D has NO native discrete buried-ice-layer capability** — confirmed by source read: the
  only depth-dependent property mechanism is the smooth two-layer exponential
  `k(z)=kd+(ks-kd)e^{-z/H}` / `rho(z)` same form (no step, no tabulated profile anywhere); `cp0` is
  a single GLOBAL scalar (not depth-dependent — best achievable "ice" TI via this model is ~1238 SI
  using the regolith's cp0, not the physical ~2000+ SI real ice would have); `chi`'s T³ radiative
  term is a single global multiplier on `kc(z)` at ALL depths (inflates deep "ice" k by ~20-50%
  unphysically). FLAGGED for Paul/Fermi, not silently patched. **Separately fixable without a code
  change**: the DEFAULT grid (`thermal_m=10`) catastrophically under-resolves a shallow ice
  boundary once `kd` is set ice-like (only 3 layers span 0→2cm) because the deep diffusivity jump
  (~400x) blows up the adaptive-grid resolution target and the hardcoded `Rmax=2.0` growth cap
  overshoots in 2-3 steps — raising the YAML key `thermal_m` (not `m`; docs mislabel it) to 150
  fixes it (17 layers to 2cm / 58 to 10cm), verified against the real run log. New configs
  `$PTM/examples/justitia_proxops_ice2cm.yaml` / `_ice10cm.yaml` (H chosen == requested depth d, a
  convention not a sharp step) ran clean on CPU, exit 0, ~8-9s each, into NEW dirs
  `.../thermal/269justitia/{ice_2cm,ice_10cm}/` (non-destructive; existing `proxops/` = case 1,
  untouched). Sanity check: clean monotonic diurnal-amplitude separation at mid-lat face 65
  (homog 76.9K > 10cm-ice 41.1K > 2cm-ice 24.1K) and whole-mesh day-max/night-min, case1's day-max
  reproduces the previously recorded 243.93K almost exactly. Camera sim / GSD-encoded plots NOT
  yet run — next step is Fermi's vet of the approximation.
  ✅➡️⚠️ **2026-07-06 native discrete-ice IMPLEMENT + the DEFINITIVE two-layer VERIFY** (memories
  `ptm-buried-ice-layer-implemented-2026-07-06` then **`ptm-buried-ice-layer-verify-refutes-
  undercoupling-2026-07-06`**; `SESSION-NOTES-2026-07-06-buried-ice-layer-{IMPLEMENT,VERIFY}.md`).
  Ada built a real discrete two-material step (`ice_depth>0`, per-profile CPU-only, k_ice=488/T+0.468,
  cp 7.49T, ρ920) → effective ice TI ~2000 SI confirmed. Then the controlled harness
  `$PTM/src/cpp/tools/test_ice_twolayer.cpp` (target `test_ice_twolayer`, drives the real
  `Profile::update_T` vs Fermi's FV reference, no SPICE/mesh) **REFUTED the suspected 4x
  under-coupling**: the converged 2 cm night-min warming is **+9.7-10.7 K** (homog amp 78.7 == Fermi
  78.5), STRONG not weak. Ada's earlier `+1.4 K` (IMPLEMENT item-4, and the sanity numbers above)
  were ARTIFACTS of (1) a min-dz floor bug corrupting the thermal_m=150 grid geometry [FIXED], (2)
  operator-split CN dt-nonconvergence on high-κ ice (needs ~6144 steps/day; production uses 192/768),
  (3) the Europa-ico5 standalone fallback. Interface-k scheme & domain immaterial. **Current ice
  binary `ptm_thermal_sim` md5 `8498d91b` (was `a40e35d`), do-no-harm byte-identical, NOT committed.
  Rocky must DROP thermal_m 150→~20-30 + force fine ice dt before the {0.5,1,2,5,10 cm} depth set** —
  the thermal_m=150 recommendation above is SUPERSEDED (it makes 0.13 mm layers the CN can't
  converge). Also flagged a pre-existing surfTemp terminator cold-spike (not fixed).
  ✅ **2026-07-06 ice-interface energy leak FIXED** (memory `ptm-ice-interface-energy-leak-FIXED-2026-07-06`,
  `SESSION-NOTES-2026-07-06-ice-interface-energy-leak-FIX.md`). Fermi's ~8% surface energy-conservation
  violation ROOT-CAUSED (per-step energy budget added to `test_ice_twolayer`) to a **frozen fixed-T
  BOTTOM boundary**, NOT the operator split / interface-k: `Profile::update_T` only calls `botTemp`
  when `Qb!=0`, so with Qb=0 the base node stays frozen at its init value (a Dirichlet reservoir);
  buried high-κ ice conducts the diurnal mean into the shallow domain (zmax 0.29 m < ice skin 0.42 m)
  so the base pumps ~3.5 W/m² UP that the surface radiates. Fix: gated `else if (has_ice_layer_)
  botTemp(*this,0.0)` (⇒ `T[n]=T[n-1]`, insulated) — do-no-harm byte-identical (0/495 .bin, both CPU
  paths; bowl-crater PASS). Post-fix conserves (F_bottom=0 all depths, `<Ts⁴>^.25`→166.0 K), **2 cm
  dMIN +10.7→+5.5** (Fermi FV +5.96), day-max pinned, 5/10 cm null — the old +9.7-vs-5.6 gap is
  resolved. **Deliverable is `libthermal.dylib` md5 `c647a16e`** (baseline `a9a490a1`); the
  `ptm_thermal_sim` EXE md5 stays `8498d91b` because it links libthermal DYNAMICALLY and embeds no
  thermal code — for any thermal-only .cpp fix compare/report the DYLIB md5, not the exe. Rocky's
  corrected set: {0.5,1,2,5,10 cm} thermal_m=20 spd=6144 forced-node ON, then camera + plots. NOT committed.
  ✅ **2026-07-06 FINAL DELIVERABLE — ice-depth-study camera + plots (Rocky)** (memory
  `ptm-justitia-ice-depth-study-camera-plots-complete-2026-07-06`, session note
  `SESSION-NOTES-2026-07-06-justitia-ice-depth-study-camera-plots.md`). Corrected thermal re-run
  (config lock `thermal_m=20`/`equilibration_steps_per_day=6144`/main-loop 6144 steps-day via
  `timesteps=23040`/`duration_days=3.75`, `equilibration_orbits=40` unchanged, `Qb=0`) on the
  energy-leak-fixed `libthermal.dylib` `c647a16e...` (rpath-verified in place): new configs
  `$PTM/examples/justitia_ice_depth_study_{d0p5cm,d1cm,d2cm,d5cm,d10cm,homog_ctrl}.yaml` → new dirs
  `.../thermal/269justitia/ice_depth_study/{d0p5cm,d1cm,d2cm,d5cm,d10cm,homog_ctrl}/` (all rc=0,
  23040/23040 timesteps; existing smooth-approx `ice_2cm`/`ice_10cm` untouched/superseded).
  `homog_ctrl` = a **consistent-insulated baseline** (`ice_depth=0.03` with ice material properties
  set identical to the regolith's own k(T)/cp(T)/ρ, so `has_ice_layer_=true` engages the energy-leak
  fix's gated insulated-BC while being thermally == plain regolith) — avoids the ~+0.57 K frozen-base
  offset a fresh no-ice control would carry in the ice-minus-homog differential.
  Camera: reused the real 66-epoch/~78h High-orbit proxops sequence (full 0-24h LST coverage at every
  latitude incl. 80-90°, pre-validated); extracted per-facet-per-frame `BT_K_OBS` ($T_\mathrm{b}$),
  lat/lon (from `SURF_X/Y/Z_KM`), local solar time, GSD →
  `.../camera/269justitia/ice_depth_study/analysis/{case}_facet_per_frame.csv` (no animations/FITS
  kept). **Found+fixed a real 12h local-solar-time phase bug** (naive hour-angle formula combining
  camera `SURF_X/Y/Z_KM` lon with a fresh SPICE-derived sun_lon put the peak at local midnight;
  sun_lon itself verified via `subslr()`; fix = +12h shift) — see memory
  `ptm-camera-surf-frame-localtime-180deg-mismatch-2026-07-06` (flagged for Ada/Fermi; the existing
  `ema_pretty_figs.py::lst_bin_diurnal` path tested unaffected/self-consistent). Two figures +
  separate caption.md files at `.../camera/269justitia/ice_depth_study/figures/{tb_vs_localtime_by_
  latitude,predawn_deltatb_vs_depth}.{png,pdf,caption.md}` (clean-figure rule, `$T_\mathrm{b}$`
  notation, dataviz-skill palette). **Measured dMIN vs Fermi-locked**: 0.5cm +27.5/+24.5, 1cm
  +18.5/+16.5, 2cm +7.1/+6.0, 5cm −0.4/NULL, 10cm +0.1/NULL — within ~1-3K, 5/10cm cleanly null. Not
  committed. **Addendum (same day, later, no re-run)**: added a 2nd Plot-A version
  `tb_vs_localtime_by_latitude_binned.{png,pdf,caption.md}` (all raw points + 1h-bin mean±1σ
  lines/bands). **Final revision (same day, later still)**: renamed **IR3S → "EMACS/IR Cam"**
  everywhere (both Plot A titles, `IR_CAM_LABEL` const in `ema_pretty_figs.py`, all 3 captions —
  per standing memory `crew-instrument-name-emacs-ir-cam`); removed the skin-depth marker from
  Plot B; NEDT now `NEDT` not `NEΔT`; and Plot B's NEDT reference changed from flat 180K-spec
  lines to **shaded bands evaluated at the ACTUAL modeled pre-dawn scene T_b (129.31-157.20K,
  the real night-min population dMIN comes from)**: EMACS/IR Cam **11.5-51.1K** (vs 5.0K@180K
  spec, `scripts/ema_camera_noise.py`'s existing model) and **EMBIRS 0.44-0.68K** (vs 0.35K@180K
  spec; EMIRS flight spare, single-scan best-thermal-channel NEDT from EMIRS's NESR, Edwards+2021
  SSR 217:77 Table 4 — Fermi gave 3 reference points, Rocky fitted an effective channel
  wavelength ≈24.74µm into the same Planck-derivative form to evaluate at the exact pre-dawn T,
  reproducing the 150K consistency point to ~4%). Detectability read: EMACS/IR Cam single-look
  NEDT sits ABOVE every plotted signal (needs binning); EMBIRS sits well below the 1-2cm signals
  (>10× margin) → detects to ≈2-5cm, limited by resolution/pointing/calibration not noise.
  ✅ **2026-07-06 optimal-detection-LST + 2nd Plot-B curve (Rocky, no re-render/re-run)** (memory
  `ptm-justitia-ice-depth-study-camera-plots-complete-2026-07-06` ADDENDUM 4, session note
  ADDENDUM 4). SNR(LST)=|Δ$T_\mathrm{b}$|/NEDT(scene T_b(LST)) scanned over 24h for the
  representative ~2cm case: **global peak at LST[10.0,10.5)h≈10:15, scene T_b=216.9K,
  Δ$T_\mathrm{b}$=+3.94K, NEDT=1.98K, SNR≈1.99** — ~9x better than pre-dawn's own SNR≈0.22 for the
  same depth (NEDT collapses ~16x faster than the signal decays through mid-morning). **NOT
  globally consistent across depths** (a real finding, not an artifact): 0.5cm/1cm's OWN peak SNR
  is later (~12.5-13h) and higher (≈6.8/4.2) — at the single 2cm-optimized LST plotted, 0.5cm has
  already crossed its own zero-crossing (-6.64K) and 1cm sits almost exactly on its own
  zero-crossing (+0.81K, SNR≈0.38, worse than its own pre-dawn value); 5/10cm null at every LST.
  Found+fixed a real pandas bug along the way: `IntervalIndex` scalar `.loc`/`.reindex` is a
  CONTAINMENT lookup not a label lookup (right-inclusive bin `(a,b]`'s edge `b` silently retrieves
  the PRECEDING bin) — map `Interval.left` first. Added 2nd Plot-B curve ("Optimal LST ≈ 10:15",
  violet, dataviz-skill categorical slot 5) + 2nd EMACS/IR Cam NEDT band; kept EMBIRS band as-is.
  `predawn_deltatb_vs_depth.{png,pdf,caption.md}` revised in place; new
  `analysis/optimal_lst_delta_tb_vs_depth.csv`. Non-destructive, no commit.
- **EMA Rockox** → `data/horizons_cache/bodies/13294_Rockox_(1998_QO105)/` SPICE cache was corrupted
  2026-07-03 (wrong SPK window + malformed metakernel) then repaired; see memory
  `ema-rockox-spice-cache-corruption-fix` before trusting/reusing that body's cache without
  re-checking `spkcov`.
- ⚠️➡️✅ **EMA 5-asteroid sweep — the 2026-07-03 stand-down's root cause is FIXED; a full regen is
  RUNNING (launched ~23:02 that same day).** The seam artifact was root-caused to a GPU per-face
  initial-condition discontinuity (memory `ema-spherical-seam-rootcause-ic-discontinuity`), fixed
  on branch `feat/gpu-smooth-ic-seed` commit `c70862b` (binary md5 `a6141315acaf24cf1fef8fdace03808b`).
  A detached overnight campaign (`/tmp/ema_regen_campaign.py`, master PID 27559, watchdog PID 27567)
  archived ALL stale bad-IC outputs (suffix `_preSmoothIC_20260703_225821`) then re-ran everything:
  Moza nominal FIRST as a go/no-go seam-proof gate, then the remaining 5 nominal, IR noise/MP4
  rebuild, Chimaera sweep+fit+off-grid, and the 5-asteroid sweep+fits. Check
  `.../EMA/science/logs/ema_regen_manifest.json` for current phase status before assuming any of
  this is done or trusting older `*_preSmoothIC_*`-suffixed dirs as current. See memory
  `ema-seamfix-overnight-regen-launch-2026-07-03` (supersedes `ema-5asteroid-sweep-standdown-seam-artifact`).
- ⚠️ **`ema_ti_fit.py fit` hardcoded its nominal-albedo lookup to Chimaera** (`HAPKE_DEFAULTS['623']`)
  regardless of `--sweep-root` — would silently corrupt any OTHER target's TI fit (wrong/clipped
  albedo). Fixed in the `/tmp/ema_ti_fit.py` scratch copy only (do-no-harm verified byte-identical
  on Chimaera); tracked `scripts/ema_ti_fit.py` still has the bug — port the fix before using the
  tracked copy for a non-Chimaera target. See memory `ema-ti-fit-hardcoded-albedo-bug`.
- ⚠️❌ **EMA 5-asteroid sweep (phase 9) is NOT actually complete for 4 of 5 targets.** Only Rockox
  has a clean 15/15-cell sweep. Westerwald (11/15), Ghaf (12/15), Ousha (12/15), and Moza (9/15)
  each have missing cells — ALL concentrated at the sweep grid's TI extremes (TI=32: 12/15 attempts
  crashed = 80%; TI=316: 4/15 = 27%; TI=56/100/178: 0/45), same GPU-hang/SIGABRT signature every
  time (`ThermalOpenCLSolver::solvePersistentMultipleCycles`, ~2s into first GPU dispatch, before
  any output). This is a systematic edge-of-stability issue in the GPU solver at very low/high `ks`,
  NOT a transient write hiccup. Ghaf's and Ousha's `ti_fit` runs exited rc=0 despite running on an
  incomplete 12/15 grid (missing the whole TI=32 row) — their "success" does not mean "ran on the
  full intended grid." Do NOT trust any of these 4 targets' `ti_fit/fit_results.json` as computed
  on a complete grid, and do NOT assume phase 9/10 finished cleanly just because the manifest shows
  phase 9 DONE. See memory `ema-5asteroid-sweep-ti-edge-gpu-crash` for the full cell-by-cell
  breakdown and options. **SUPERSEDED for the launch approach** (GPU abandoned in favor of CPU on
  2026-07-04, see next item) but the underlying 4-target incomplete-grid data itself is untouched.
- ⚠️❌ **2026-07-04 conduction-fix (5bfbdbe) CPU/chi=2.7 regen: Chimaera DONE, the other 5 targets
  are NOT regenerated and their pre-fix output is now ARCHIVED (not live).** Chimaera's full
  pipeline (nominal, 15/15-cell sweep, fit, off-grid, figures) completed cleanly on CPU in ~27.5
  min — see memory `ema-conductionfix-cpu-campaign-2026-07-04`. But Rockox (first of the remaining
  5, 81,920-face icosphere) hit a severe **streamed-illumination I/O performance cliff on CPU**:
  its thermal solve alone projected ETA ~2h for ONE nominal cell (~120x slower than Chimaera's
  ~60s, wildly disproportionate to the ~4.5x face-count ratio) — Chimaera's smaller mesh (~1.8GB
  illumination estimate) fully EXTRACTS into RAM before solving, while Rockox's (~14.8GB estimate)
  falls into a "stream from disk during the solve" branch that is catastrophically slow on this
  OneDrive-synced machine. Campaign was manually HALTED rather than let it run for days. **Do NOT
  treat `thermal/{13294rockox,10253westerwald,23871ousha,59980moza,88055ghaf}/{flyby,sweep,ti_fit}`
  as current** — those were archived pre-flight (suffix `_preConductionFix_20260704_121627`) and
  never regenerated; only Chimaera's are live/current. Route to Ada to diagnose the RAM-extract-
  vs-stream threshold/implementation before relaunching the remaining 5.
- ⚠️ **Trap dirs (near-name — never use blindly):** `sites_v2_3km_rad_corering/A03MP026` has **no
  `view_factors.hmat`** → a run there silently proceeds *without* radiation VF = silently-wrong results;
  `sites_v2_3km_rad_corering_regression_35b052e/A03MP026` holds the **6.1 GB rebuilt VF** that
  hard-wedged the GPU (2026-07-03). Symlinked VFs are fragile — deleting `sites_v2_ms3/...` silently
  breaks the canonical baseline. See memory `ptm-a3gt-vf-reuse-and-baseline-id`.

