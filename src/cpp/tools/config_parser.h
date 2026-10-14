#ifndef PTM_CONFIG_PARSER_H
#define PTM_CONFIG_PARSER_H

#include <string>
#include <map>
#include <vector>
#include <optional>
#include <stdexcept>

namespace ptm {

/**
 * @brief Structure to hold all PTM simulation configuration parameters
 * 
 * This structure organizes parameters into logical groups matching the
 * YAML configuration file structure and command-line arguments.
 */
struct SimulationConfig {
    // Time and Simulation
    struct TimeParams {
        std::string start_time = "2025-01-01T00:00:00";
        // Ephemeris-time offset (seconds) added to start_time AFTER it is
        // resolved to ET. 0 (default) = start exactly at start_time.
        //
        // The epoch grid's cadence is a number of ET seconds, but start_time is
        // a UTC string. Shifting a window by adding to the UTC string is
        // therefore not the same shift: TDB - UTC carries a periodic
        // relativistic term (~1.7 ms amplitude, annual period) and steps by a
        // whole second at a leap second. Offsetting a 29.5-day chunk boundary
        // in UTC misplaced it by 2.2e-4 s; across a leap second it would be 1 s.
        // scripts/run_chunked_simulation.py sets this so chunk k begins exactly
        // k*N*dt ET-seconds after the base start.
        double start_time_offset_seconds = 0.0;
        double duration_days = 3.55;
        int timesteps = 72;
        std::string body = "Europa";
        // BUGS.md PTM-301. When the loaded SPK does not span the run window
        // (equilibration spin-up included), the engine falls back to Keplerian
        // two-body propagation per epoch. false (default) = warn, naming both
        // spans, and run anyway; true = refuse. Default false because it is a
        // real, deliberately available capability -- a run may legitimately
        // want the fallback -- but it is now an explicit choice rather than an
        // invisible substitution. Also settable as PTM_SPICE_COVERAGE_STRICT=1.
        bool spice_coverage_strict = false;
    } time;
    
    // Mesh Configuration
    struct MeshParams {
        int resolution = 5;
        std::string import_file = "";
        // Optional occluder-only mesh (coarse LOLA ring/annulus around the
        // SfS core). Added to the illumination shadow-ray BVH ONLY — never
        // thermally solved, radiosity-coupled, or output. Empty = none.
        // See docs/extended-occluder-architecture.md.
        std::string occluder_file = "";
        double scale_factor = 1.0;
        bool export_basic = false;
        bool use_topographic = false;
        std::string topo_directory = "";
        double latitude_override = -999.0;
        double longitude_override = -999.0;
    } mesh;
    
    // Thermal Model
    struct ThermalParams {
        bool run_thermal_mesh = false;
        int threads = 0;  // 0 = auto
        // INTEGRATED number of diurnal cycles of real, ephemeris-driven
        // spin-up before the output window opens. Identical on the CPU
        // scalar, CPU SIMD and GPU persistent paths. equilibration_cycles
        // below changes only HOW the total is reached, never the total.
        // ⚠️ Before 2026-08-11 this was the BLOCK length and the GPU
        // integrated 2x it (BUGS.md PTM-146); shipped GPU configs were
        // rewritten to state the integrated number they always ran.
        // ⚠️ The CPU paths additionally run a fixed analytic clear-sky
        // pre-conditioner of NYEARSEQ * round(year/day) cycles (Moon 24,
        // Europa 2) that this key does not count and the GPU lacks; the
        // pre-flight banner prints it.
        double equilibration_orbits = 1.0;
        // Separate (coarser) cadence for the equilibration period.
        // Highest-priority explicit override: if > 0, used directly (in steps per Earth-day).
        // -1.0 (default) = unset; behavior determined by equilibration_samples_per_diurnal_cycle.
        double equilibration_steps_per_day = -1.0;

        // Accuracy-based equilibration cadence: target number of illumination/flux samples
        // per diurnal (day/night) cycle for the target body. The actual steps/Earth-day is
        // computed at runtime as (samples_per_diurnal_cycle / synodic_day_earth).
        // Physics guidance: 24 samples/cycle yields <2 K accuracy for the Moon; 48 gives
        // <1 K; 12 is marginal; <6 is not recommended.
        //  > 0 = auto-compute equilibration cadence with this many samples per diurnal cycle
        //  <= 0 = fall back to legacy behavior (match output cadence)
        // Only consulted when equilibration_steps_per_day <= 0.
        double equilibration_samples_per_diurnal_cycle = -1.0;

        // Number of REPLAYS of the equilibration forcing block (GPU persistent
        // solver only). A pure memory/cost dial: the assembled block is
        // equilibration_orbits / equilibration_cycles diurnal cycles long and
        // each pass replays the same uploaded flux array with the temperature
        // state persisting, so the INTEGRATED spin-up is equilibration_orbits
        // at every pass count while the flux RAM and the illumination trace
        // shrink by this factor. That is the only affordable route past the RAM
        // wall: a 25-cycle eq array is ~6.1 GB on the 509k-face lunar mesh.
        //   -1 (default) = unset → 1 pass (block == equilibration_orbits).
        //   >= 1         = explicit replay count.
        // Trade-off, not a free lunch: N replays of a short block are not the
        // same forcing as one long block of real ephemeris, so prefer a block
        // that is a whole number of diurnal cycles (the engine warns otherwise).
        // On CPU runs the pass count is downgraded to 1 with a printed notice —
        // the integrated length is preserved by assembling the full block.
        int equilibration_cycles = -1;

        // Accuracy the spin-up estimator is asked to reach, in kelvin. The
        // estimator is REPORT-ONLY: it prints how many diurnal cycles THIS
        // problem needs to settle to this accuracy, from tau1 (the slowest
        // eigenvalue of the actual discrete column) and the seed's distance
        // from its own conductive steady profile. It changes no state and
        // never overrides equilibration_orbits / equilibration_cycles.
        //
        // Default 0.1 K. Justification: it is the threshold the equilibration
        // work already uses for the Kelvin conversions of the residual, and it
        // sits an order of magnitude below the ~6 K by which the lit faces
        // miss at the shipped 50-lunation default -- so a run that is actually
        // wrong says so, and a run that is fine does not cry wolf.
        double equilibration_target_K = 0.1;

        // Drive the LENGTH of the analytic clear-sky pre-conditioner from the
        // retrospective certificate instead of from a fixed cycle count: stop
        // when the cycle-mean interface flux residual, converted to kelvin,
        // reaches equilibration_target_K.
        //
        // DEFAULT FALSE, so nothing changes for any existing run until it is
        // asked for. With it false the pre-conditioner loop is byte-identical
        // to what it was; with it true the loop can end EARLY (never late --
        // the configured cycle count is still the ceiling), and the achieved
        // E is reported either way.
        //
        // Implemented on the CPU SIMD batch path only. The other three solver
        // entry points REFUSE to start with it set rather than ignoring it;
        // silently running a fixed count under a key that asks for a criterion
        // is exactly the failure this key exists to end.
        bool equilibration_stop_on_certificate = false;

        // ---- thermal.eq_extrapolation -------------------------------------
        // Pass-stride RRE extrapolation of the equilibration state. DEFAULT
        // OFF: with method == "off" the solver runs the pre-existing pass loop
        // with no added work and byte-identical output.
        //
        // There is deliberately no `stride` knob. The pass map is autonomous by
        // construction (the same forcing block is replayed byte-identically),
        // so pass stride is correct for every body and every config without a
        // predicate. A one-cycle stride is available by setting
        // equilibration_orbits: 1 -- no new knob needed.
        struct EqExtrapolation {
            // "off" | "rre"
            std::string method = "off";
            // Polynomial order. Needs equilibration_cycles >= order_p + 1.
            // 3 and 4 are the validated values; < 3 is REFUSED (p=2 is worse
            // than no extrapolation at all: 54.82 K vs plain-2's 31.88 K).
            int    order_p = 4;
            // Certificate reporting threshold. Faces with ehat > tol_K are
            // counted `cert_refused`; the extrapolated state still ships.
            double tol_K   = 0.5;
            // Level-2 hold threshold, K per pass: a layer whose LAST-pass
            // increment |S_{N-1}-S_{N-2}| is below this is treated as converged
            // and held at S_last instead of extrapolated.
            double hold_K  = 1e-3;
        } eq_extrapolation;

        bool compute_statistics = true;
        bool use_gpu = false;  // Use GPU thermal solver (OpenCL)

        // Multi-GPU controls (OpenCL only). Defaults preserve legacy behavior:
        //   gpu_devices == 1 (and gpu_device_ids empty) → first available GPU,
        //   gpu_batch_size == 0 → existing 6 GB / 500 K-face cap,
        //   gpu_inter_batch_pause_ms == 0 → no pause.
        // Set gpu_devices = 0 (or "auto" via the CLI flag) to use every
        // enumerated GPU. gpu_device_ids overrides gpu_devices when non-empty;
        // indices come from --list-gpu-devices.
        int gpu_devices = 1;
        std::vector<int> gpu_device_ids;
        size_t gpu_batch_size = 0;
        int    gpu_inter_batch_pause_ms = 0;

        // Phase R (auto-calibrate): default ON. When true and no matching host
        // machine profile exists, the GPU solver probes THIS run's grid inline
        // (once) to inject a representative dispatch self-chunk ceiling before the
        // main solve. false (or env PTM_NO_AUTO_CALIBRATE, or an explicit
        // PTM_MAX_TIMESTEPS_PER_DISPATCH) restores the profile/conservative path.
        // Numerically inert: affects dispatch batching only, never physics.
        bool auto_calibrate_gpu = true;

        // Per-face thermal property map: a per-face binary float32 array (one
        // value per mesh face, in face-index order) overriding the body-default
        // H-parameter (or, equivalently, the depth-weighted thermal inertia).
        // Only one of {h_parameter_map, thermal_inertia_map} may be set.
        // If use_default_property_map = true and both paths are empty, the
        // bundled icosphere map for the current body is auto-discovered under
        // <property_maps_directory>/<body>/.
        std::string thermal_inertia_map = "";
        std::string h_parameter_map = "";
        bool        use_default_property_map = false;
        std::string property_maps_directory  = "data/property_maps";
        bool enable_shadow_flux = false;
        double crater_dd_ratio = 8.0;
        double integration_days = -1.0;  // -1 means use output window
        int integration_steps = -1;
        double integration_offset_days = 0.0;
        int integration_offset_steps = 0;
        bool debug = false;               // Enable thermal debug output for one equatorial face
        double debug_lat_threshold = 10.0; // Maximum latitude (deg) for debug face selection
        // Vertical depth-grid parameters. m = layers per upper skin depth;
        // n = layer growth factor (in units used by the adaptive grid); b =
        // skin depths to bottom layer. The defaults below give a bottom at
        // ~b·δ_diurnal so the diurnal wave is fully damped before the
        // bottom Neumann BC clips it. With the radiative-conductivity term
        // χ(T/350)³ ~doubling K at deep temperatures, the effective skin
        // depth at depth is ~1.5–2× the surface lattice value — so b needs
        // to be ≥ ~10 for the wave to attenuate by exp(-5) below the BC.
        // The previous defaults (m=4, n=10, b=5) gave only ~3 actual
        // e-folds at depth, leaving a ~tens-of-K diurnal residual at the
        // bottom that contaminated extrapolated location-export depths.
        int grid_m = 10;                  // Layers in upper skin depth
        int grid_n = 15;                  // Layer growth factor (only used by legacy non-adaptive path)
        int grid_b = 15;                  // Skin depths to bottom (≈9 e-folds of damping at deep T; bottom amp <0.05 K)
        // Geothermal heat flow
        double geothermal_heat_flow = -1.0; // W/m², -1 = use planet default
        // Per-face heat flow mask (for localized endogenic heat)
        std::string heat_flow_mask_file = "";  // Path to face index file
        double heat_flow_mask_value = 1.0;     // Qb for masked faces (W/m²)

        // Per-face thermal-state checkpointing (used by the chunked-run
        // orchestrator scripts/run_chunked_simulation.py).
        // initial_state_file: if non-empty, load per-face T(z) profile
        //   from this binary file at the start of the run, skip
        //   equilibration entirely (forces equilibration_steps=0).
        // final_state_file: if non-empty, dump per-face T(z) profile
        //   to this binary file at the end of the run (after the GPU
        //   solver finishes).
        // Format: uint32 magic 0x50544D54 ('PTMT'), uint32 n_faces,
        //   uint32 n_layers, then n_layers*n_faces float32 values
        //   (layer-major to match the GPU SoA layout).
        std::string initial_state_file = "";
        std::string final_state_file   = "";

        // Per-field opt-out for the PTMT v2 identity guards
        // (thermal/thermal_state_io.h checkRestore). Each entry NAMES one
        // clause and downgrades it from a refusal to a loud warning:
        //   depth_grid  mesh  config  body  grid_m  grid_n  grid_b
        //   refgrid_H   epoch
        // There is deliberately no blanket "force": switching every guard off
        // at once is the warn-instead-of-refuse policy that made PTM-76
        // possible, wearing a different name. The batch-completion-partial
        // refusal cannot be reached from here at all.
        //
        // `epoch` is the one most likely to be reached for, and the one to
        // reach for most carefully: a cross-season warm start, measured on one
        // lunar polar site, moved temperature_min by 5.75 K rms / 49.9 K max
        // and shifted the ice-stability depth product by 19.3 percentage
        // points, and no gate in the QA battery saw it.
        std::vector<std::string> state_restore_allow_mismatch;

        // Checkpoint handoff look-ahead, in OUTPUT steps. 0 (default) = the
        // state written to final_state_file is the state at the LAST exported
        // frame. 1 = the run integrates one extra grid point PAST the exported
        // output window and checkpoints there, so the checkpoint is the state
        // at the FIRST epoch of the next window.
        //
        // Why this exists. The output grid is half-open: a window of N steps
        // over D days covers epochs t0 + i*(D/N) for i in [0, N), and the next
        // window starts at t0 + N*(D/N). The solver integrates
        // total_time_steps - 1 intervals, so without a look-ahead its final
        // state sits at t0 + (N-1)*(D/N) -- one cadence short of where the next
        // window begins. A chunked run must therefore either overlap by one
        // epoch (duplicating a sample) or hand off a state that is one step
        // stale. With checkpoint_lookahead_steps = 1 the chunk integrates N
        // intervals over N+1 grid points, exports and reduces only the first N,
        // and checkpoints at the extra point -- both properties hold at once.
        // scripts/run_chunked_simulation.py sets this for every chunk.
        //
        // The extra grid point is NOT exported (no frame, no time_steps.txt
        // row) and NOT reduced into statistics/*.bin; it only costs one extra
        // illumination sample and one extra integration step.
        int checkpoint_lookahead_steps = 0;

        // Phase R (R4): mid-run thermal-state checkpoint cadence, in OUTPUT
        // steps. 0 (default) = off = checkpoint only at chunk boundaries (the
        // status quo). NOTE (architectural, documented in
        // docs/PORTABILITY-AUDIT.md): true per-output-step full-mesh
        // checkpointing is INCOMPATIBLE with the current GPU persistent
        // solver, which is batch-outer/time-inner (each face-batch runs ALL
        // timesteps before the next batch starts), so no single wall-clock
        // moment holds all faces at the same output step. A partial full-mesh
        // dump at a batch boundary would carry un-solved faces at 0 K and is a
        // resume FOOTGUN (the zero-column runaway guarded in the GPU path). The
        // sim therefore WARNS loudly if this is set > 0 rather than silently
        // no-op'ing; blast-radius bounding today is the runner's chunk split
        // (run_yearlong_wedge_safe.py --stage-b-nchunks). The natural home for
        // per-step checkpointing is the CPU-first solver (Phase 1).
        int checkpoint_every_output_steps = 0;

        // ------------------------------------------------------------------
        // BATCH-COMPLETION CHECKPOINTING (plan phase 1D'). This is the answer
        // to the paragraph directly above, and it is the INVERSE of it.
        //
        // The solver is batch-outer, so there is no moment when all faces
        // share an output step -- but at a batch boundary a well-defined set
        // of faces is FINISHED FOR THE WHOLE WINDOW. Each completed batch's
        // products are banked to a self-describing partial
        // (thermal/batch_checkpoint.h, "PTMB"); a restart validates them,
        // skips those batches and integrates only the rest.
        //
        // OFF BY DEFAULT, and byte-identity when off is the acceptance test.
        //
        // ⛔ WHAT THIS DOES NOT BOUND. A run that dies inside a batch loses
        // that batch's whole time integration -- that is the granularity, by
        // construction. Finer bounding is still the runner's process-level
        // chunk split (scripts/run_chunked_simulation.py), which is already
        // built and already mathematically equivalent to a single shot.
        //
        // ⛔ COST. A partial carries the batch's SURFACE TIME SERIES, not just
        // its column, because every statistic and every per-step product is a
        // reduction of that series. At the lunar polar production shape
        // (F = 2,397,890, T = 1440) the partial set is ~26 GiB per window, not
        // the ~210 MiB a column-only dump would cost. The engine prints the
        // exact figure at startup and REFUSES above batch_checkpoint_max_gib
        // rather than filling a disk quietly.
        bool batch_checkpoint = false;
        // Where the partials live. Empty (default) =
        // <output.directory>/_thermal_state/partial.
        std::string batch_checkpoint_dir = "";
        // Ceiling on the projected partial-set size, in GiB. Exceeding it is a
        // REFUSAL naming the projected size, not a warning: a 26 GiB surprise
        // mid-window is a disk event, not a checkpoint.
        double batch_checkpoint_max_gib = 64.0;
        // How much completed work to accumulate before writing a partial, in
        // MiB. 0 (default) DERIVES it as one sixteenth of the projected set,
        // clamped to [1, 64] MiB -- a fixed threshold makes the feature
        // silently useless on any run smaller than it. The engine prints the
        // resolved value and the implied worst-case loss at startup.
        double batch_checkpoint_flush_mib = 0.0;

        // Deep radiation-coupling, two-pass scheme (fixes the cold-PSR-floor
        // bias from equilibrating the deep column WITHOUT the radiosity
        // indirect flux). Orchestrated as two binary runs:
        //  - indirect_mean_out_file: if non-empty, after the radiation-coupling
        //    Picard loop, write the per-face OUTPUT-window time-mean indirect
        //    flux <q_ind>[f] (W/m^2) to this binary file.
        //  - eq_indirect_flux_file: if non-empty, load per-face <q_ind>[f] from
        //    this file and ADD it to the EQUILIBRATION-phase absorbed flux only
        //    (the output window already gets the time-varying indirect via the
        //    coupling), so the deep column equilibrates consistent with the
        //    wall-heating forcing.
        // Format: uint32 magic 0x51494E44 ('QIND'), uint32 n_faces, then
        //   n_faces float64 values (per global face id).
        std::string indirect_mean_out_file = "";
        std::string eq_indirect_flux_file  = "";

        // Where the equilibration-phase indirect flux comes from when
        // radiation coupling is enabled (docs/radiation-exchange.md 7.7):
        //   "auto" (default) -- the engine estimates it internally. One extra
        //     radiosity sweep over the window plus one extra whole-mesh thermal
        //     solve: the first solve gives surface temperatures, the sweep gives
        //     Q_indirect(i,t), its output-window time mean becomes <q_ind>[i],
        //     and the whole-mesh solve is repeated with <q_ind> in the
        //     equilibration forcing. This is the two-pass file workflow below,
        //     internalized; no user action.
        //   "converged" -- same machinery, but <q_ind> is read AFTER the
        //     coupling loop converges instead of at its iteration 0. Costs one
        //     extra coupling loop per pass (~2.0x wall on the 80 N crater) and
        //     is the only setting that removes the iteration-0 low bias: "auto"
        //     reads a sweep whose emitters still carry no output-window
        //     indirect, so its <q_ind> stays ~13% low at ANY pass count.
        //   "off" -- legacy: the equilibration phase sees no indirect term, so
        //     a face whose only heat source is neighboring-face infrared
        //     equilibrates to the bare geothermal floor.
        // A non-empty eq_indirect_flux_file overrides all three (explicit wins).
        // Overridable without editing a configuration through
        // PTM_EQ_INDIRECT_SOURCE, for reproducing a run published under "off".
        std::string eq_indirect_source = "auto";

        // How many times the internal estimate is refreshed and the whole-mesh
        // solve repeated, on whichever sweep eq_indirect_source selects. Pass 1
        // estimates <q_ind> from temperatures that equilibrated with NO indirect
        // term at all, so its emitters are too cold and its estimate is a lower
        // bound -- measured 11.7% low on the 80 N crater. Each further pass
        // re-reduces <q_ind> from the temperatures the previous pass produced,
        // which is a fixed-point iteration on the equilibration forcing.
        //   1 -- default, and the knee for eq_indirect_source: converged (pass 2
        //        moved that crater's deep column a further 0.056 K).
        //   2 -- the knee for eq_indirect_source: auto, but a DOMINATED one. On
        //        the same crater it bought 0.284 K of deep column for +0.26x
        //        wall, where switching to eq_indirect_source: converged at ONE
        //        pass bought 3.90 K for +0.73x. The 3.61 K it cannot reach is a
        //        property of the iteration-0 estimator, not of the pass count.
        //        Spend on the estimator first; see docs/radiation-exchange.md
        //        7.7 for the table both figures come from.
        // COST is linear: each pass is one radiosity sweep plus one whole-mesh
        // thermal solve over the equilibration+output span (under "converged",
        // one whole extra coupling loop plus that solve). The solve scales with
        // the equilibration span, so a second pass gets steadily worse value as
        // equilibration_orbits grows. Must be >= 1; 0 is not an alias for "off"
        // and is rejected. Overridable through PTM_EQ_INDIRECT_PASSES.
        int eq_indirect_passes = 1;

        // Which INITIAL CONDITION every solver path starts from
        // (src/cpp/thermal/eq_seed.h). Overridable through PTM_SPINUP_SEED.
        //   "physical" (default) -- one shared per-face seed: the cycle-mean
        //     conductive steady profile T(z) = T_0 + Qb*integral dz/k anchored
        //     at T_0 = ((<F_abs> + Qb)/(eps*sigma))^0.25, with <F_abs> that
        //     face's own time-mean absorbed flux. A permanently shadowed column
        //     starts on the geothermal floor instead of 42-47 K above it, and
        //     the CPU analytic clear-sky pre-conditioner is OFF.
        //   "legacy" -- each path's pre-2026-08-11 seed AND the analytic
        //     pre-conditioner, i.e. exactly what produced every result in this
        //     repository before that date. Use it to reproduce an old baseline.
        std::string spinup_seed = "physical";

        // Which SURFACE ANCHOR the shared seed hangs its conductive profile
        // from (src/cpp/thermal/eq_seed.h). Overridable through
        // PTM_SEED_ANCHOR. Only reachable under spinup_seed: physical.
        //   "lumped" (default) -- the derived lumped-substrate anchor: solve
        //     eps*sigma*T^4 + G*T = F(t) + Qb + G*T_d for T_d = <T(t)>, with G
        //     fixed by a describing-function variance match against this
        //     face's own forcing spectrum. Anchors the column at the CYCLE
        //     MEAN, which is what the conductive profile describes.
        //   "warm" -- T_0 = ((<F_abs>+Qb)/(eps*sigma))^0.25 only. That is the
        //     exact FOURTH moment, and by Jensen it is >= <T>: +45 K mean on
        //     the lit population of the shipped lunar 80N crater. Byte-
        //     identical to the pre-2026-08-12 seed.
        std::string seed_anchor = "lumped";

        // INDIRECT-AWARE spin-up seed for radiation-coupled runs (BUGS.md
        // PTM-727). The seed is built from the face's absorbed DIRECT forcing
        // only, so a never-lit column (indirect is its only heat) starts on
        // the geothermal floor (25.28 K on Moon3D) and two equilibration
        // lunations leave it ~50 K too cold at depth. Acts ONLY in the
        // eq-indirect re-solve (thermal.eq_indirect_source != off), the first
        // point at which an indirect estimate exists.
        //   "off" (default) -- byte-identical to every earlier run.
        //   "series" -- seed each face from its absorbed flux PLUS the
        //     iteration-0 (or converged-loop) time-varying indirect flux over
        //     the OUTPUT window (uniform cadence, the span the sweep solved),
        //     with the lumped anchor allowed to run on that composite series.
        //     Recommended: anchors the column near the cycle-mean <T>.
        //   "mean" -- add the per-face time-mean <q_ind> to every forcing
        //     sample; the anchor still stands down (radiation-coupled), so the
        //     surface is the WARM bracket T_rad(<F>+<q_ind>) -- by Jensen above
        //     <T> on faces whose indirect swings (24.8 K median on Moon3D
        //     never-lit faces). Kept for A/B attribution, not recommended.
        std::string seed_indirect = "off";

        // What a TWO-TIMESCALE face does. The anchor's closure picks ONE
        // conductance G from a variance match, and with diurnal AND seasonal
        // power in the same series that single G is being asked to cover two
        // regimes at once. Detection is always on and always reported
        // (status lumped-two-timescale / fallback-two-timescale, with the
        // measured sub-diurnal variance fraction).
        //
        // The ACTION defaults to FALSE -- run the anchor anyway -- because
        // refusing means substituting T_w, and T_w is the estimator with the
        // +45 K Jensen bias. Measured on tutorial04_crater80N_radoff (a
        // 2-lunation span, so the sub-solar declination drift IS a second
        // timescale): the gate fires on 158 of 12,800 faces, 111 of them dark
        // and flat where T_w is already right, but 47 of them with a 576 W/m^2
        // diurnal swing where |T_w - reference| is 70 K median / 84 K max.
        // Set true to restore the strict refusal.
        bool seed_anchor_refuse_two_timescale = false;

        // Length, in diurnal cycles, of the CPU analytic clear-sky
        // pre-conditioner: `equil_days` cycles of cos(lat)*cos(h) insolation
        // with NO topography, NO shadowing, NO IR and NO indirect flux,
        // integrated before the real forcing. It is a FORCING, not a seed, it
        // is not counted by equilibration_orbits, and the GPU has no analog.
        //   -1 (default) = follow spinup_seed (physical -> 0, legacy -> the
        //     historic NYEARSEQ*round(year/day): Moon 24, Europa 2, Mercury 4)
        //    0 = off;  > 0 = exactly this many, and the run says so loudly.
        // Overridable through PTM_ANALYTIC_SEED_CYCLES.
        double analytic_seed_cycles = -1.0;

        // Schorghofer & Taylor (2007) ice-stability output. When enabled,
        // the thermal solver accumulates time-averaged sublimation rate
        // ⟨E(T(z,t))⟩ at each native depth node during the integration
        // window, and post-solve solves J(z) = μℓ⟨E⟩(z)/(2z) = J₀ for
        // the stability depth per face. Disabled by default — zero
        // overhead when off. Output: statistics/ice_depth.bin (one
        // float32 per face).
        bool compute_ice_stability = false;
        struct IceStabilityParams {
            double flux_threshold_kgm2s     = 3.0e-14;  // J₀, ~1 mm/Ga
            double grain_hop_length_um      = 75.0;     // ℓ
            double geothermal_gradient_K_per_m = 0.0;   // extra g on top of solver T(z); 0 = trust solver
            double sublimation_enthalpy_kJmol  = 51.058; // Q (Schorghofer Eq. 4)
            int    n_depth_subsamples       = -1;       // -1 = use every native depth node
        } ice_stability;
    } thermal;
    
    // Illumination
    struct IlluminationParams {
        bool include_primary_reflection = true;
        bool use_batch_processing = true;
        bool use_albedo_map = true;           // Auto-discover/load spatially-varying albedo maps
        std::string albedo_map = "";
        std::string albedo_maps_directory = "data/albedo_maps";
        std::string positions_file = "";

        // GPU-accelerated shadow ray tracing (off by default; CPU Embree is
        // the historical default for large lunar runs). When use_gpu is true,
        // the BVH ray tracer is initialized and routed through BVHGPUPool.
        bool use_gpu = false;

        // Multi-GPU dispatch knobs for the BVH ray tracer. Defaults preserve
        // single-GPU behavior (gpu_devices == 1, gpu_device_ids empty).
        // gpu_devices == 0 / "auto" → use every enumerated GPU.
        // gpu_device_ids overrides gpu_devices when non-empty.
        int gpu_devices = 1;
        std::vector<int> gpu_device_ids;
        int  gpu_inter_batch_pause_ms = 0;

        // Path to a serialized BVH cache (writes after first build, reads
        // on subsequent runs). Empty = auto-derive under the output directory
        // (<output>/bvh_cache.bvh1). Set to "off" to disable.
        std::string bvh_cache_file = "";
    } illumination;
    
    // Output Configuration
    struct OutputParams {
        std::string directory = "output";
        std::string format = "binary";  // "binary" or "vtk"
        std::string vtk_mode = "both";  // "thermal", "complete", "both"
        bool include_primary_ir = false;
        std::string thermal_vtk_file = "";
        std::string complete_vtk_file = "";
        std::string locations_file = "";
        std::string series_output = "";
        std::string location_data = "";
        std::string depths_file = "";
        bool skip_time_series = false;  // skip per-step binary writes (statistics still computed)
        // Write merge-friendly raw statistics (per-face sums, sample counts,
        // and the Schorghofer ice-depth ⟨E(z)⟩ accumulator) to
        // statistics/_raw/ alongside the usual {datatype}_{max,avg,min}.bin.
        // Used by the chunked-run orchestrator + ptm_merge_chunks tool.
        bool write_raw_stats = false;

        // ---- Subsurface temperature map product -----------------------------
        // Mesh-wide temperature statistics AT DEPTH, mirroring the surface
        // products (statistics/temperature_{avg,max,min}.bin) one file set per
        // requested depth, named `temperature_z<meters>m_<stat>.bin`.
        //
        // EMPTY IS OFF and is byte-neutral to every other product; that is the
        // acceptance test for this feature, not an incidental property.
        //
        // The resolved weights, the snapped depths, the residuals and the
        // PREDICTED interpolation bias per depth are written to
        // statistics/depth_map_z.json. The refusal surface (a depth below the
        // deepest solved node, duplicates, an unknown statistic, a file-name
        // collision after snapping, an oversized product) lives in
        // src/cpp/thermal/depth_map_plan.h, which is where it is unit-tested.
        std::vector<double> map_depths_m;

        // Snap each requested depth to the nearest grid node. DEFAULT ON,
        // because the INSTANTANEOUS linear-interpolation error at an off-node
        // depth is 1.217e-3 x A -- 0.195 K at a lunar-equatorial amplitude
        // A = 160 K, over the project's 0.15 K floor by itself and independent
        // of how the statistic is accumulated (measured by
        // test_thermal_wave_depth, 2026-09-04). Only a depth landing ON a node
        // avoids it. Turning this off is legitimate, but the run then prints
        // the per-depth bias it is spending.
        bool map_depth_snap_to_node = true;

        // Which statistics to write per depth: any subset of avg, max, min.
        // `avg` commutes with interpolation exactly; `max`/`min` do not, which
        // is why they are accumulated IN the solver at the requested depth
        // rather than interpolated from per-layer extrema afterwards (that
        // route costs 1.229e-3 x A and is worst at layer 0).
        std::vector<std::string> map_depth_stats = {"avg", "max", "min"};

        // Skip writing mesh.ply to the output directory. Useful for the
        // chunked-run workflow: chunk 0 emits mesh.ply once for the
        // orchestrator to propagate, chunks 1+ set this true and skip
        // the redundant ~hundreds-of-MB write per chunk.
        bool skip_mesh = false;
    } output;

    // Flux cubes -- a grid of PASSIVE virtual measurement fixtures on the
    // terrain. Reports incident irradiance on six oriented faces, component
    // resolved, at every output step. Cubes never occlude, emit, scatter, or
    // enter the radiosity solve, so `enabled: true` must be byte-neutral to
    // every mesh-wide field.
    //
    // TOP LEVEL, deliberately: nested under `output:` it would be reachable by
    // the body-defaults machinery that makes `thermophysical:` decorative on a
    // built-in body (PTM-67). See docs/flux-cubes.md.
    struct FluxCubeParams {
        bool enabled = false;
        // Geometry cache from ptm_flux_cube_geom. REQUIRED when enabled; a
        // missing or key-mismatched cache is a hard error, never a downgrade to
        // "run without cubes" -- a run asked for a product must not succeed
        // quietly having produced none.
        std::string cache_file = "";
        // MULTI-HEIGHT. A cache carries ONE height: FcgHeader.height_m is a
        // scalar and it is part of the cache key, so "cubes at 0.35 m, 1.0 m and
        // 2.0 m" is three caches, not one. They are consumed in ONE run -- one
        // thermal solve, three flux-cube passes, three .fcb outputs -- because
        // re-running the solve per height would be three times the cost for a
        // field that does not depend on the height at all.
        //
        // ⛔ THIS IS A LIST AND IT IS GUARDED. yaml-cpp iterates a SCALAR as an
        // EMPTY range, so `cache_files: layout.fcg` would clear the vector and
        // never fill it: a run with no cubes, no message and exit 0. Guarded in
        // config_parser.cpp with requireYamlSequence, same as the six keys
        // PTM-CONFIG-SEQUENCE-SCALAR-SILENT covered on 2026-09-06.
        //
        // Empty (the default) => the single-cache `cache_file` path, whose
        // outputs keep their historical names. Setting BOTH is refused rather
        // than merged: which one wins is not a question a config should leave
        // open.
        std::vector<std::string> cache_files;
        bool write_binary = true;      // flux_cubes/flux_cubes.fcb
        bool write_csv = true;         // flux_cubes/csv/cube_XXXXX.csv
        // 0 = every cube. On a 720-step production window the full fan-out is
        // 1,257 files per window; cap it here and let the binary carry them all.
        int  write_csv_max_cubes = 0;
        // Solar shadow probe: 1 = center ray only, 5 = center + the four
        // horizontal face centers (spread = edge_m), which also sets the
        // per-epoch `penumbral` flag when the five disagree.
        int  shadow_probe_rays = 5;
        // Named lat/lon locations, the user-facing analog of the engine's
        // `locations:` mechanism. These do NOT place anything by themselves --
        // placement is baked into the cache by ptm_flux_cube_geom, which is the
        // only thing that may touch geometry. What the engine does with this
        // list is VERIFY that the cache actually contains every name asked for,
        // and refuse otherwise. That closes the "edited the YAML, forgot to
        // rebuild the cache" trap, which would otherwise hand back a product
        // silently describing the previous set of locations.
        struct NamedLocation {
            std::string name;
            double lat_deg = 0.0;
            double lon_deg = 0.0;
            double height_m = -1.0;   // <0 => cache/site default
            double edge_m = -1.0;     // <0 => cache/site default
        };
        std::vector<NamedLocation> locations;
    } flux_cubes;

    // Shadow depth -- the maximum height above the local surface at which the
    // Sun is not even partially visible, per face and per output epoch.
    //
    // TOP LEVEL for the same reason flux_cubes is: nested under `output:` it
    // would be reachable by the body-defaults machinery that makes
    // `thermophysical:` decorative on a built-in body (PTM-67).
    //
    // `enabled: false` (the default) must be byte-neutral to every other
    // output, and `true` must be too -- the pass reads arrays and writes its
    // own files and touches no mesh field.
    struct ShadowDepthParams {
        bool enabled = false;
        // The search ceiling. 5 m is Paul's number, from crew mobility and
        // thermal constraints; it is configurable because that is a mission
        // parameter and not a physical constant.
        double h_max_m = 5.0;
        // March rungs, then bisections inside the bracket the march found.
        // Pure bisection is NOT an option here: it assumes the
        // blocked-below/clear-above transition is single-valued and an overhang
        // breaks that. Final bracket = h_max/(march_steps * 2^bisect_rounds) =
        // 0.0195 m at the defaults.
        int march_steps = 16;
        int bisect_rounds = 4;
        // Solar-disk escalation count when the 5-ray probe splits. The disk is
        // the engine's own finite disk; the umbra is DEFINED by it.
        int solar_disk_samples = 128;
        // Per-epoch frames in shadow_depth/. The window statistics and the
        // manifest are written regardless.
        bool write_frames = true;
        // Sampled monotonicity audit: rungs ABOVE the first clear one, which the
        // march never looks at. 0 disables it, and the manifest then records a
        // zero sample rather than a zero failure count.
        int monotonicity_faces = 10000;
        int monotonicity_epochs = 10;
    } shadow_depth;

    // Radiation Exchange
    struct RadiationParams {
        bool enabled = false;
        // PTM-98 — which faces keep the indirect flux the radiosity solve already
        // computed for them. The solve always covers ALL faces; this set is only
        // a post-solve copy mask.
        //
        //   "all" (default): every face keeps its flux. Window-INDEPENDENT.
        //   "shadow": legacy gate, keep only faces shadowed at least
        //       min_shadow_fraction of the time. The shadow fraction is measured
        //       over the run's OWN illumination series, whose length is
        //       n_equilibration_steps + n_output_steps -- and a warm-started chunk
        //       has NO equilibration samples. So a chunked run reclassifies faces
        //       out of radiation exchange and zeroes their indirect flux: measured
        //       82.38 K / 6367 faces > 1 K on a 3-chunk lunation of the 12,800-face
        //       lunar_crater_80N gate, with 361 faces at exactly 0 W/m^2 where the
        //       single-shot gives 18.1 mean / 63.0 max. Provided ONLY to reproduce
        //       published runs; combine with coupling_mode: legacy.
        //
        // Override without editing a config: PTM_EXCHANGE_SELECTION=all|shadow
        // or --rad-exchange-selection <mode>.
        std::string exchange_selection = "all";
        // Consulted ONLY when exchange_selection == "shadow". Before PTM-99 this
        // key parsed but was never bridged into radiation::RadiationConfig, so the
        // hard-coded 0.3 always won silently.
        double min_shadow_fraction = 0.3;
        bool verbose = false;
        bool use_gpu = false;
        std::string view_factor_file = "";
        // Picard iteration cap for the radiation-thermal coupling loop.
        // RAISED 10 -> 16 on 2026-08-15. The outer Picard rate is a property of
        // the VF SPECTRUM, r <= eps*rho(F)/(1-(1-eps)*rho(F)), and on the 509k
        // lunar flagship (rho(F) = 0.7766, r measured 0.745) the sweep-to-sweep
        // increment first falls below 1 K at sweep 16 -- so 10 truncated it, and
        // the habitual 5 truncated it badly (worst face +26.2 K, monotone
        // warming, i.e. a capped run is COLD). Cheap now that the default-on
        // equilibration-radiosity skip removes ~46.5% of a chunk's wall clock.
        // The engine prints rho(F) and the implied sweep count before the loop
        // and WARNS when this value is below it -- an advisory, not a clamp.
        int max_coupling_iterations = 16;
        double coupling_temp_threshold = 0.5;
        double coupling_flux_threshold = 0.1;
        double coupling_face_fraction = 0.01;
        // Face-selection mode for the rad-thermal coupling re-thermal phase.
        // "physics" (default): linearized ΔT predictor, ΔT ≈ (1−α)·ΔF / (4εσT³).
        //                      Re-runs faces whose predicted T change exceeds
        //                      coupling_temp_threshold·coupling_select_safety.
        //                      Also implicitly freezes faces whose observed
        //                      ΔT in the previous iter was below threshold.
        // "legacy": absolute flux threshold (max indirect flux > coupling_flux_threshold).
        //           Reproduces pre-2026-05-09 behavior for byte-identical re-runs.
        std::string coupling_select_mode = "physics";
        // Safety factor applied to the predicted-ΔT threshold. The predictor
        // is a linearization that breaks down for ΔF ≳ T·few; multiplying the
        // threshold by this factor (default 0.5) errs on the side of inclusion.
        // NOTE: only consulted when coupling_mode == "legacy" (see below).
        double coupling_select_safety = 0.5;
        // PTM-89 — decomposition (chunking) independence of the Picard coupling.
        //
        // Every input to the shipped face-selection and exit decisions is a
        // max OVER THE WINDOW, so a 30-step window and a 60-step window select
        // different faces and stop after a different number of iterations: the
        // converged field then depends on where the chunk boundaries fell.
        // Measured on the 12,800-face lunar_crater_80N gate (60 frames, seam at
        // 30): single-shot vs 2-chunk differed by up to 41.4 K, 28 faces > 1 K.
        //
        //   "bounded" (default): the selection threshold is the ABSOLUTE
        //       coupling_select_dt (default 0 K = re-thermal every face that
        //       carries any indirect flux), and the unbounded
        //       coupling_face_fraction OR-exit is disabled. Same gate: chunk A
        //       becomes BIT-IDENTICAL to the single-shot and the whole-run
        //       residual falls to 0.014 K, below the 0.029 K radiation-free
        //       chunking floor set by the float32 checkpoint.
        //   "fixed":   as "bounded", and the early exit on
        //       coupling_temp_threshold is disabled too, so every window runs
        //       exactly max_coupling_iterations sweeps. This is the only
        //       stopping rule that cannot depend on the window by construction.
        //   "legacy":  pre-PTM-89 behavior, bit-for-bit. Threshold is
        //       coupling_temp_threshold·coupling_select_safety and the
        //       face-fraction OR-exit is live. Provided to REPRODUCE published
        //       runs; it re-introduces the defect.
        //
        // Override without editing a config: PTM_COUPLING_MODE=bounded|fixed|legacy
        // or --rad-coupling-mode <mode>.
        std::string coupling_mode = "bounded";
        // Absolute predicted-ΔT selection threshold [K], "bounded"/"fixed" only.
        // 0 = select every face with any indirect flux (decomposition-independent).
        // >0 trades accuracy for speed and RE-INTRODUCES PTM-89 at that magnitude;
        // the run warns when it is set.
        double coupling_select_dt = 0.0;
        bool use_hmatrix = false;
        bool use_aca = true;
        double aca_tolerance = 1e-3;
        int aca_max_rank = 50;
        int hmatrix_threads = 0;  // 0 = auto
        // Adaptive neighbor selection (sparse view factors)
        bool adaptive_neighbors = true;           // Enable per-face adaptive selection
        double captured_energy_target = 0.95;     // Target fraction of radiative potential (95%)
        int max_neighbors = 500;                  // Hard ceiling per face (reduced from 2000 for better performance)
        int min_neighbors = 20;                   // Minimum neighbors per face
        bool use_cluster_cull = true;             // Conservative octree cluster-pair cull before ranking
        // Near-field (M2) view-factor quadrature order. Order m samples BOTH
        // faces of a near pair at the centroids of m x m equal-area
        // sub-triangles (m^2 points/face, m^4 kernel evaluations/pair), which
        // makes A_i F_ij == A_j F_ji an algebraic identity. 3 = production
        // default; 2 is ~2.8x cheaper on the near-field kernel at essentially
        // the same mass-weighted accuracy. Valid 1..8; out of range is FATAL.
        // Only the in-engine SPARSE view-factor builder reads this -- the
        // H-matrix and Monte-Carlo builders have their own kernels, and a run
        // that LOADS a cached radiation.view_factor_file never evaluates it.
        int nearfield_quadrature_order = 3;
        // Near/far switch for BOTH in-engine builders:
        //     near <=> r < nearfield_window_factor * (sqrt(A_i) + sqrt(A_j)).
        // 0 = never refine (pure M1); very large = always refine (pure M2).
        // Replaced the internal adaptive_ns_threshold (a threshold on
        // Ns = r / (A_i A_j)^(1/4), which is blind to an area contrast) --
        // NOT a rename: the two are different quantities.
        double nearfield_window_factor = 4.0;
        // Monte Carlo ray-gather view-factor builder (mc_view_factor.cpp).
        bool use_monte_carlo_vf = false;
        int mc_rays_per_face = 100000;
        // Anisotropic (roughness-beamed) thermal emission, Smith 1967.
        // THE SWITCH. false (default) = pure Lambertian emission, byte-identical
        // to every pre-2026-08-25 run. true REQUIRES rms_slope_deg > 0 (the
        // knob); the engine refuses to start on beaming: true with no slope.
        // Standard for lunar polar SITE and REGIONAL configs (facets >> the ~mm-cm
        // thermal-isolation scale). OFF on sub-cm meshes, which already resolve the
        // roughness this parameterizes -- Paul, 2026-08-25. See
        // docs/radiation-exchange.md section 8.
        bool beaming = false;
        // THE KNOB. RMS micro-slope w [deg] of the sub-facet slope distribution.
        // Literature range 20-37 deg; 32.5 is the Jhoti et al. Table 1 lunar
        // highlands value used for the validated polar result. Inert unless
        // beaming: true. 0 (default) = Lambertian.
        double rms_slope_deg = 0.0;
        double beam_vf_threshold = 1e-7;
        // Radiance-weighted view-factor pruning threshold [W/m^2], applied at VF
        // ASSEMBLY time (scripts/ptm12_diag/assemble_vf_haworth.py) -- NOT in the
        // sim. 0.0 (default) = no pruning, byte-identical. DISTINCT from
        // visibility_threshold (a build-time storage floor). The runners propagate
        // this to the assembler via PTM_RADIANCE_PRUNE_EPS; env overrides config.
        // See docs/radiation-exchange.md 10.9.
        double radiance_prune_eps = 0.0;
        // Band-separated (visible + thermal-IR) radiosity (D1). Production DEFAULT
        // ON (D2, 2026-07-12); gray recovered exactly with band_separated:false.
        // See docs/two-band-radiosity-derivation.tex.
        bool band_separated = true;
        // Multi-GPU radiosity dispatch (mirrors thermal pool config).
        // Per-timestep solves are independent so the timestep-batch can be
        // split across N GPUs. Default == 1 preserves legacy single-GPU
        // behavior byte-identically. Only consulted when use_gpu = true.
        int gpu_devices = 1;
        std::vector<int> gpu_device_ids;
        // 0 = auto = whole sweep per dispatch (memory-capped, floored at
        // 4 * gpu_devices); radiation::resolvePoolBatchSize().
        int gpu_batch_size = 0;
        int gpu_inter_batch_pause_ms = 0;    // Watchdog mitigation (mac+AMD)
    } radiation;

    // Generic Body Configuration (for user-defined bodies)
    struct GenericBodyParams {
        std::string name = "";            // Body name from config YAML
        std::string config_file = "";     // Path to body config YAML file
        std::string ephemeris_source = "auto"; // "auto", "keplerian", or "spice"
        std::string spice_metakernel = "";    // Path to SPICE metakernel file
        std::string naif_id = "";             // NAIF/SPICE body ID (e.g. "2000269")

        // Shape parameters (-1 = use default or omit)
        std::string shape_type = "";      // "spherical", "ellipsoidal", "custom_mesh"
        double radius_km = -1.0;
        double semi_axis_a_km = -1.0;
        double semi_axis_b_km = -1.0;
        double semi_axis_c_km = -1.0;
        std::string mesh_file = "";       // Path to custom mesh file (OBJ/PLY)
        double mesh_scale = 1.0;          // Scale factor for custom mesh

        // Orbital elements (-1 = not specified)
        double semi_major_axis_au = -1.0;
        double eccentricity = -1.0;
        double inclination_deg = -1.0;
        double lon_ascending_node_deg = -1.0;
        double arg_perihelion_deg = -1.0;
        double mean_anomaly_deg = -1.0;
        // Epoch of the osculating elements above (TDB; orbital/epoch_parse.h).
        // Wired BUGS.md PTM-663: mean_anomaly_deg is M0 AT this epoch. Before
        // that it was parsed and dropped, and M0 was applied at J2000.
        std::string orbital_epoch = "";

        // Spin state (-1 = not specified)
        double spin_period_hours = -1.0;
        double pole_ra_deg = -1.0;
        double pole_dec_deg = -1.0;
        // Prime meridian angle W0 (degrees) at the spin epoch below, measured
        // about the pole from the body's reference axis (orbital/body_fixed_frame.cpp):
        // J2000 ecliptic +x (the vernal equinox) projected onto the body's
        // equatorial plane, or ecliptic +y if the pole is within 25.8 deg of +x.
        // W(t) =
        // W0 + 360 deg * (t - epoch) / period. Wired BUGS.md PTM-663; before
        // that it was parsed and dropped.
        double prime_meridian_deg = 0.0;
        bool prime_meridian_set = false;  // the key was present in this file
        // Epoch of prime_meridian_deg, on the TDB scale (orbital/epoch_parse.h).
        // "" = J2000 (JD 2451545.0 TDB), which is what every run used before.
        std::string spin_epoch = "";

        // Thermophysical overrides (-1 = use body default)
        double albedo = -1.0;
        double emissivity = -1.0;
        double ks = -1.0;
        double kd = -1.0;
        double rhos = -1.0;
        double rhod = -1.0;
        double H_scale = -1.0;
        double cp0 = -1.0;
        double Qb = -1.0;
        double chi = -1.0;  // Radiative conductivity parameter (-1 = use default 2.7)

        // Is this section populated?
        bool hasOrbitalElements() const {
            return semi_major_axis_au > 0.0;
        }
        bool hasSpinState() const {
            return spin_period_hours > 0.0;
        }
        bool isConfigured() const {
            return !config_file.empty() || !name.empty()
                || hasOrbitalElements() || hasSpinState();
        }
    } generic_body;

    // Thermophysical overrides (from YAML thermophysical section)
    // These flags indicate the user explicitly set a property value
    // rather than relying on the body's default.
    struct ThermophysicalParams {
        bool albedo_explicit = false;
        double constant_thermal_inertia = -1.0;  // >0: constant TI mode (chi=0, cp=const)
        // Optional scalar overrides applied to the body Planet after construction
        // (Moon/SPICE path included). Sentinel <0 (or <0 for albedo) = leave the
        // body default untouched. Used for property sensitivity sweeps.
        double albedo_override = -1.0;      // >=0: override Bond albedo A0
        double emissivity_override = -1.0;  // >0:  override emissivity
        double ks_override = -1.0;          // >0:  override surface conductivity ks
        double kd_override = -1.0;          // >0:  override deep conductivity kd
        // Regolith density/conductivity e-folding depth H (m), the Hayne et al.
        // (2017) Table-A1 scale over which rho(z) and kc(z) relax from their
        // surface to their deep values:
        //     rho(z) = rhod + (rhos - rhod) * exp(-z/H)
        //     kc(z)  = kd   + (ks   - kd  ) * exp(-z/H)
        // Only the per-face `thermal.h_parameter_map` / `thermal.thermal_inertia_map`
        // channel could reach H before this key existed, so a scalar H sweep on a
        // built-in body was not expressible at all (a plain `thermophysical.H`
        // landed in the dead-key warning). NOTE H is inert when ks == kd AND
        // rhos == rhod -- which `constant_thermal_inertia` deliberately forces --
        // and a per-face H map out-ranks this scalar on every face it covers;
        // both cases warn at startup rather than flatlining a sweep in silence.
        double H_override = -1.0;           // >0:  override e-folding depth H (m)

        // Directional-hemispheric (Hapke) albedo model, OFF by default. When
        // enabled, the absorbed-solar term uses the analytic closed form
        //   A_h(mu0) = 1 - gamma * H(mu0, w),  gamma = sqrt(1-w),
        //   H(mu0,w) = (1 + 2*mu0) / (1 + 2*mu0*gamma),  mu0 = cos(incidence),
        // replacing the constant (1 - albedo). Gated so baseline (Moon3D / A3GT /
        // Europa) runs stay byte-identical. Opt in via `incidence_angle_albedo:
        // true` or `albedo_model: hapke` in the thermophysical section.
        bool   albedo_hapke = false;        // gate: use A_h(i) closed form
        double albedo_hapke_w = -1.0;       // single-scattering albedo w; <0 =
                                            // derive from the Bond albedo A0.

        // Regolith thermal-conductivity law.
        //   "hayne2017"    (default) k = kc(z) * (1 + chi*(T/350)^3)
        //   "martinez2021"           k = (A1*rho-A2)*k_am(T) + (B1*rho-B2)*T^3
        //                            Martinez & Siegler (2021), JGR Planets 126,
        //                            e2021JE006829. Temperature-dependent SOLID
        //                            conduction; intended for ultracold regolith
        //                            (PSRs) where the Hayne form is extrapolated
        //                            below its ~150 K laboratory floor.
        // Stored as the RAW string. The parser reports an unrecognized value on
        // stderr but does not throw; ptm_thermal_sim.cpp validates this string
        // right after config load and exits 1, so an unknown law can never be
        // silently replaced by Hayne. (The original reason for not throwing was
        // that a throw was swallowed -- BUGS.md PTM-66. That is fixed in this
        // release; the two-step is kept for the better message. See
        // config_parser.cpp.)
        // NOTE selecting martinez2021 makes ks/kd INERT for conduction (Eq. 10
        // depends on rho(z) only) -- the engine says so at startup.
        std::string conductivity_model = "hayne2017";
    } thermophysical;

    // Legacy accessor (for compatibility)
    bool thermophysical_albedo_explicit = false;

    // Advanced Options
    struct AdvancedParams {
        bool verbose = false;
        double default_visible_albedo = -1.0;  // -1 = use planet default
    } advanced;

    // GPU safety policy (cross-cutting, not per-solver tuning).
    //
    // These keys decide whether a GPU rung is ALLOWED and which dies each pool
    // may use; the per-solver gpu_devices / gpu_device_ids / gpu_batch_size
    // keys decide how a rung that is allowed then runs. They came out of the
    // 2026-08-20/21 wedge forensics: see src/cpp/gpu/gpu_guardrails.h for the
    // mechanism and docs/gpu-wedge-guardrails.md for the platform-gating table
    // and the cost of each guardrail.
    struct GpuParams {
        // Floor on the per-device VRAM headroom a resident radiosity CSR
        // replica may leave behind [GiB]. A rung with less is refused on
        // macOS+AMD (or wherever this is set explicitly) and flagged
        // elsewhere. 0 = the pre-2026-08-22 behavior (fit check only).
        // Consequence of the 2.0 default, accepted deliberately: a site whose
        // replica fits by less than 2 GiB now degrades -- loudly -- to CPU
        // radiosity instead of running on a margin thinner than the thermal
        // pool's own per-device ask.
        double min_vram_margin_gib = 2.0;
        // auto | shared | disjoint. Whether the radiosity replica and the GPU
        // thermal pool may occupy the same dies. See decidePlacement().
        std::string placement_policy = "auto";
        // Margin below which `auto` separates the two pools [GiB].
        double placement_margin_gib = 4.0;
        // What a margin below min_vram_margin_gib DOES:
        //   separate = keep the GPU rung, but only behind disjoint placement,
        //              and refuse only below hard_min_vram_margin_gib. DEFAULT
        //              (Paul's ruling, 2026-08-22).
        //   refuse   = give up the GPU rung (radiosity on the CPU).
        // Both remove the co-residency the 2026-08 wedges died in; they differ
        // by a lot in price on a large-matrix site. Measured on A3GT lunar polar site B,
        // the CPU radiosity rung cost 108.9 min/window against 31.4 min on the
        // GPU pool (+77.5 measured), while separation costs a derived
        // +7-10 min. See docs/gpu-wedge-guardrails.md.
        std::string thin_margin_policy = "separate";
        // The floor below which even `separate` refuses [GiB].
        double hard_min_vram_margin_gib = 0.25;
        // NOT a YAML key: true when min_vram_margin_gib was written in the
        // config or the environment, which makes the floor enforcing on every
        // platform rather than only on the wedge-prone one.
        bool min_vram_margin_explicit = false;
    } gpu;
};

/**
 * @brief High-performance YAML configuration parser for PTM
 * 
 * Optimized for fast parsing with minimal memory allocations.
 * Falls back to reasonable defaults if YAML support is not available.
 */
class ConfigParser {
public:
    ConfigParser() = default;
    ~ConfigParser() = default;
    
    // Disable copy, enable move for performance
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;
    ConfigParser(ConfigParser&&) = default;
    ConfigParser& operator=(ConfigParser&&) = default;
    
    /**
     * @brief Load configuration from YAML file
     * @param filename Path to YAML configuration file
     * @return SimulationConfig structure with loaded parameters
     * @throws std::runtime_error if file cannot be parsed
     */
    SimulationConfig loadFromYAML(const std::string& filename);

    /**
     * @brief Apply a config's `engine_env:` block to the process environment (PTM-42).
     *
     * PTM3D configs may carry an `engine_env:` mapping recording the environment
     * variables a run needs (`PTM_ILLUM_TRUE_NORMALS`, `PTM_EQ_PERFACE_FLUX`, ...).
     * Until 2026-08-04 nothing in the engine read that block: it was documentation
     * applied only if a human remembered to export it, and a regression pair could
     * silently end up with two different illumination configurations while looking
     * identical on paper (BUGS.md PTM-42, which absorbs PTM-52).
     *
     * Contract:
     *  - The PROCESS environment always WINS. A variable already set to a different
     *    value is left alone and the conflict is logged, so a deliberate
     *    command-line override (or `scripts/ptm_run_with_engine_env.py`) still works
     *    and nothing changes under a caller's feet.
     *  - Every variable is logged with its disposition (applied / already set /
     *    conflict), so the run log records the environment the physics actually saw.
     *  - Call this BEFORE loadFromYAML and before any solver construction: several
     *    gated features latch their variable on first call.
     *
     * @param filename Path to the YAML configuration file
     * @return number of variables newly applied to the environment
     */
    static int applyEngineEnv(const std::string& filename);

    /**
     * @brief Save configuration to YAML file
     * @param filename Path to output YAML file
     * @param config Configuration to save
     * @return true if successful, false otherwise
     */
    bool saveToYAML(const std::string& filename, const SimulationConfig& config);
    
    /**
     * @brief Override config parameters from command line arguments
     * @param config Configuration structure to modify
     * @param argc Argument count
     * @param argv Argument vector
     * 
     * Command line arguments take precedence over config file values.
     * This allows users to override specific parameters without editing files.
     */
    void overrideFromCommandLine(SimulationConfig& config, int argc, char** argv);
    
    /**
     * @brief Generate a template YAML configuration file
     * @param filename Output filename for template
     * @return true if successful
     * 
     * Creates a well-documented template with all available parameters
     * and their default values.
     */
    static bool generateTemplate(const std::string& filename);
    
    /**
     * @brief Validate configuration for consistency
     * @param config Configuration to validate
     * @return true if valid, false with error messages printed
     */
    static bool validate(const SimulationConfig& config);
    
    /**
     * @brief Get the last error message
     */
    const std::string& getLastError() const { return last_error_; }

private:
    std::string last_error_;
    
    // Helper methods for parsing sections
    void parseTimeSection(SimulationConfig::TimeParams& params, const void* node);
    void parseMeshSection(SimulationConfig::MeshParams& params, const void* node);
    void parseThermalSection(SimulationConfig::ThermalParams& params, const void* node);
    void parseIlluminationSection(SimulationConfig::IlluminationParams& params, const void* node);
    void parseOutputSection(SimulationConfig::OutputParams& params, const void* node);
    void parseRadiationSection(SimulationConfig::RadiationParams& params, const void* node);
    void parseAdvancedSection(SimulationConfig::AdvancedParams& params, const void* node);
};

} // namespace ptm

#endif // PTM_CONFIG_PARSER_H