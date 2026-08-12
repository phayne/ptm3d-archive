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
        double duration_days = 3.55;
        int timesteps = 72;
        std::string body = "Europa";
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
        //   "off" -- legacy: the equilibration phase sees no indirect term, so
        //     a face whose only heat source is neighboring-face infrared
        //     equilibrates to the bare geothermal floor.
        // A non-empty eq_indirect_flux_file overrides both (explicit wins).
        // Overridable without editing a configuration through
        // PTM_EQ_INDIRECT_SOURCE, for reproducing a run published under "off".
        std::string eq_indirect_source = "auto";

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
        // Skip writing mesh.ply to the output directory. Useful for the
        // chunked-run workflow: chunk 0 emits mesh.ply once for the
        // orchestrator to propagate, chunks 1+ set this true and skip
        // the redundant ~hundreds-of-MB write per chunk.
        bool skip_mesh = false;
    } output;
    
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
        int max_coupling_iterations = 10;
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
        // Monte Carlo ray-gather view-factor builder (mc_view_factor.cpp).
        bool use_monte_carlo_vf = false;
        int mc_rays_per_face = 100000;
        // Anisotropic (roughness-beamed) thermal emission, Smith 1967.
        // 0 (default) = Lambertian, byte-identical. RMS micro-slope [deg].
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
        int gpu_batch_size = 0;              // 0 = auto = 4 * gpu_devices
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
        std::string orbital_epoch = "";

        // Spin state (-1 = not specified)
        double spin_period_hours = -1.0;
        double pole_ra_deg = -1.0;
        double pole_dec_deg = -1.0;
        double prime_meridian_deg = 0.0;

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