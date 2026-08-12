// solver_disclosure.h -- which thermal solver actually ran, and why.
//
// The defect this closes
// ----------------------
// PTM3D ships four reachable whole-mesh thermal solvers, and one of them is
// selected by a MEMORY-SIZE ESTIMATE rather than by configuration:
//
//   ptm_thermal_sim.cpp:  use_streamed_thermal =
//                             using_chunked_storage
//                          && (memory_required_gb_est > STREAM_THRESHOLD_GB)
//
// Above that threshold a CPU run silently swaps its solver from the default
// SIMD batch solver to the scalar solver. The only thing the log said about it
// was "Will stream illumination to the thermal solver" -- a statement about
// ILLUMINATION storage, which never mentions that the SOLVER changed. A
// 509,337-face x 1440-step lunar run estimates 16.4 GB by that predicate's own
// arithmetic (faces x steps x 8 B x 3 arrays, divided by 1024^3) and so crosses
// the 10 GB default; production runs took the swap and no log recorded it.
//
// What the swap costs is deliberately NOT characterized here. The streamed
// path is byte-identical to the --no-simd-batch run the same config would
// otherwise get (measured on 0bb5218, 80N tutorial crater: only provenance.txt
// differs, 0.000 K on every temperature statistic, on four config variants), so
// the substitution that remains is CPU scalar for CPU SIMD batch. How far those
// two integrators diverge is an open question being settled elsewhere, and a
// number pinned here would go stale the moment it is answered -- which is the
// exact failure this file was written to prevent. Report the substitution; let
// whoever settles the divergence describe it.
//
// Three caveats this file used to print are gone, and must not come back. The
// streamed path applying a constant albedo instead of the configured angular
// model, producing no effective_albedo statistic, and dropping the
// equilibration indirect-flux term were all closed by
// fix/streamed-flux-composition (66c6dd7) and fix/ir-albedo-kirchhoff
// (2aa8f67). Each was re-measured on 0bb5218 on 2026-08-12 with a falsifier
// that fires the right way: toggling incidence_angle_albedo moves streamed
// temperature_avg by 17.0 K mean / 24.5 K max over 11,400 faces, a supplied
// eq_indirect_flux_file (uniform 2 W/m^2) moves it by 3.00 K mean / 26.0 K max
// over all 12,800, and effective_albedo.bin is written on every streamed run
// and md5-equal to the scalar run's. Printing them today would stamp a defect
// that no longer exists into every run's provenance.
//
// The contract
// ------------
// Every run states, exactly once and in one place, which thermal solver
// executed and why it was selected -- to the log and to provenance.txt, so a
// finished run directory answers the question without its log.
//
// The value reported MUST come from the same variable the dispatch branches
// on. A banner re-derived from a second copy of the predicate can drift out of
// sync with the branch, and a banner that can disagree with reality is worse
// than no banner. Callers therefore compute the path ONCE, disclose it, and
// switch on the same value -- see the `switch` in solveWholeMeshThermal and
// the streamed branch beside it.

#ifndef PTM_SOLVER_DISCLOSURE_H
#define PTM_SOLVER_DISCLOSURE_H

#include <string>

namespace ptm {

// The whole-mesh thermal solvers this executable can dispatch to.
enum class ThermalSolverPath {
    Unset,          // nothing disclosed yet
    CpuScalar,      // runThermalModelsForAllFaces
    CpuSimdBatch,   // runThermalModelsForAllFacesBatchSoA        (default)
    CpuStreamed,    // runThermalModelsForAllFacesStreamed        (scalar core)
    GpuPersistent   // runThermalModelsForAllFaces_GPU_Persistent
};

// Why that path was taken. `MemoryThreshold` is the invisible one this file
// exists for.
enum class ThermalSolverReason {
    Unset,
    ConfigFlag,       // an explicit request (thermal.use_gpu, --no-simd-batch)
    Default,          // nothing asked; the shipped default applied
    MemoryThreshold   // the illumination footprint estimate crossed a threshold
};

// The radiation-coupling selective re-thermal, which re-solves a SUBSET of
// faces after each radiosity pass and therefore owns the final temperatures of
// those faces. It has its own dispatch and its own GPU->CPU fallback.
enum class RethermalPath {
    NotRun,           // no radiation coupling, or no face selected
    GpuSelectedFaces, // runThermalForSelectedFaces_GPU
    CpuScalarFaces    // runThermalModelWithMatchedTimes, per selected face
};

// Announce the whole-mesh thermal solver. Prints the banner on the first call.
//
// Idempotent by design: solveWholeMeshThermal is invoked more than once on the
// equilibration-indirect path, and re-announcing would be noise. A repeat with
// the SAME path is silent. A repeat with a DIFFERENT path means the run
// changed solver mid-flight, which no caller intends -- that warns loudly and
// the record becomes "mixed", so provenance.txt cannot claim a single path the
// run did not have.
//
//   path                 the value the dispatch itself switches on
//   reason               how it was chosen
//   detail               one line naming the deciding quantity (estimate and
//                        threshold, flag name, ...). Goes verbatim into
//                        provenance.txt, so keep it to one line.
//   simd_batch_requested the effective use_simd_batch setting, so an override
//                        of it by the streamed path can be reported as such.
void discloseThermalSolverPath(ThermalSolverPath path,
                               ThermalSolverReason reason,
                               const std::string& detail,
                               bool simd_batch_requested);

// Announce the radiation-coupling selective re-thermal. Same idempotency rule;
// a GPU->CPU fallback between coupling iterations is a genuine change and is
// reported as "mixed" rather than hidden behind whichever ran last.
void discloseRethermalPath(RethermalPath path, const std::string& detail);

// Stable machine-readable ids, for provenance.txt and for tests.
const char* thermalSolverPathId(ThermalSolverPath p);   // "cpu_simd_batch", ...
const char* thermalSolverEntryPoint(ThermalSolverPath p);
// The time integrator a path actually runs. CpuStreamed reports "cpu_scalar"
// because that is the integrator it loops -- and on current main a streamed run
// is byte-identical to a --no-simd-batch run of the same config, so for that
// path the core id is the whole answer to "did my physics change?".
//
// Read it together with `thermal_solver_simd_batch_requested`. A run that asked
// for the SIMD batch solver and got `cpu_scalar` did change integrator. That
// substitution, and only that, is what `thermal_solver_caveats` now reports --
// the size of the resulting difference is not asserted here.
const char* thermalSolverCoreId(ThermalSolverPath p);
const char* thermalSolverReasonId(ThermalSolverReason r);
const char* rethermalPathId(RethermalPath p);

// The caveat token for a (path, simd_batch_requested) pair, or "none".
//
// ONE definition, called by both the banner and the provenance writer, for the
// same reason the path itself is computed once: a log line and a provenance key
// that are derived separately can disagree, and then neither can be trusted.
const char* thermalSolverCaveats(ThermalSolverPath p, bool simd_batch_requested);

// Accessors, for the provenance writer and for callers that need to assert.
ThermalSolverPath   recordedThermalSolverPath();
ThermalSolverReason recordedThermalSolverReason();
std::string         recordedThermalSolverDetail();
RethermalPath       recordedRethermalPath();

// Append the thermal-solver record to <output_dir>/provenance.txt.
//
// Appended rather than written with the rest of the provenance because the
// rest is written before the expensive work starts (so a run that dies in the
// solver still leaves a record of what it was), while the solver path is only
// known once the solver has been dispatched. Returns false and warns on any
// failure, including "nothing was ever disclosed" -- that case means a solver
// entry point was added without a disclosure, which is the exact regression
// this file exists to prevent.
bool appendThermalSolverProvenance(const std::string& output_dir);

// Test hook: forget the recorded path. Only for unit tests that exercise more
// than one path in a single process.
void resetThermalSolverDisclosureForTest();

}  // namespace ptm

#endif  // PTM_SOLVER_DISCLOSURE_H
