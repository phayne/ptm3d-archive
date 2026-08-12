// solver_disclosure.cpp -- see solver_disclosure.h for the contract.

#include "solver_disclosure.h"

#include <fstream>
#include <iostream>

namespace ptm {
namespace {

struct Record {
    ThermalSolverPath   path   = ThermalSolverPath::Unset;
    ThermalSolverReason reason = ThermalSolverReason::Unset;
    std::string         detail;
    bool                simd_batch_requested = false;
    bool                mixed = false;   // more than one path ran

    RethermalPath rethermal        = RethermalPath::NotRun;
    std::string   rethermal_detail;
    bool          rethermal_mixed  = false;
};

// Process-global, because the dispatch and the provenance writer sit ~1,700
// lines apart in main() with no object between them to hang this on. Single
// threaded: every call is from main()'s dispatch, outside any parallel region.
Record& record() {
    static Record r;
    return r;
}

const char* kRule =
    "=======================================================================";

}  // namespace

const char* thermalSolverPathId(ThermalSolverPath p) {
    switch (p) {
        case ThermalSolverPath::CpuScalar:     return "cpu_scalar";
        case ThermalSolverPath::CpuSimdBatch:  return "cpu_simd_batch";
        case ThermalSolverPath::CpuStreamed:   return "cpu_streamed";
        case ThermalSolverPath::GpuPersistent: return "gpu_persistent";
        case ThermalSolverPath::Unset:         break;
    }
    return "none";
}

const char* thermalSolverEntryPoint(ThermalSolverPath p) {
    switch (p) {
        case ThermalSolverPath::CpuScalar:     return "runThermalModelsForAllFaces";
        case ThermalSolverPath::CpuSimdBatch:  return "runThermalModelsForAllFacesBatchSoA";
        case ThermalSolverPath::CpuStreamed:   return "runThermalModelsForAllFacesStreamed";
        case ThermalSolverPath::GpuPersistent: return "runThermalModelsForAllFaces_GPU_Persistent";
        case ThermalSolverPath::Unset:         break;
    }
    return "none";
}

const char* thermalSolverCoreId(ThermalSolverPath p) {
    switch (p) {
        // The streamed path is the scalar solver with an on-demand flux getter
        // in front of it: it loops the same runThermalModelWithMatchedTimes.
        // Reporting its core as "cpu_scalar" is the whole point of the field.
        case ThermalSolverPath::CpuStreamed:
        case ThermalSolverPath::CpuScalar:     return "cpu_scalar";
        case ThermalSolverPath::CpuSimdBatch:  return "cpu_simd_batch";
        case ThermalSolverPath::GpuPersistent: return "gpu_opencl_float32";
        case ThermalSolverPath::Unset:         break;
    }
    return "none";
}

const char* thermalSolverReasonId(ThermalSolverReason r) {
    switch (r) {
        case ThermalSolverReason::ConfigFlag:      return "config_flag";
        case ThermalSolverReason::Default:         return "default";
        case ThermalSolverReason::MemoryThreshold: return "memory_threshold";
        case ThermalSolverReason::Unset:           break;
    }
    return "none";
}

const char* rethermalPathId(RethermalPath p) {
    switch (p) {
        case RethermalPath::GpuSelectedFaces: return "gpu_selected_faces";
        case RethermalPath::CpuScalarFaces:   return "cpu_scalar_selected_faces";
        case RethermalPath::NotRun:           break;
    }
    return "not_run";
}

// Long-form label plus, where it matters, what the reader would otherwise get
// wrong. Only the streamed path carries a caveat, because only the streamed
// path is a solver substitution the caller did not ask for.
static const char* pathLabel(ThermalSolverPath p) {
    switch (p) {
        case ThermalSolverPath::CpuScalar:     return "CPU scalar";
        case ThermalSolverPath::CpuSimdBatch:  return "CPU SIMD batch";
        case ThermalSolverPath::CpuStreamed:   return "CPU streamed";
        case ThermalSolverPath::GpuPersistent: return "GPU persistent";
        case ThermalSolverPath::Unset:         break;
    }
    return "none";
}

void discloseThermalSolverPath(ThermalSolverPath path,
                               ThermalSolverReason reason,
                               const std::string& detail,
                               bool simd_batch_requested) {
    Record& r = record();

    if (r.path != ThermalSolverPath::Unset) {
        if (r.path == path) return;   // the eq-indirect second solve; expected
        // Not expected. Say so rather than let the last writer win.
        std::cerr << "\n[thermal-solver] WARNING: this run dispatched MORE THAN ONE "
                     "whole-mesh thermal solver:\n"
                  << "[thermal-solver]   first: " << pathLabel(r.path)
                  << " (" << thermalSolverEntryPoint(r.path) << ")\n"
                  << "[thermal-solver]   then:  " << pathLabel(path)
                  << " (" << thermalSolverEntryPoint(path) << ")\n"
                  << "[thermal-solver]   The two do not share a surface boundary "
                     "condition, a spin-up or a clamp set, so the output field is a "
                     "mixture.\n"
                  << "[thermal-solver]   provenance.txt will record this as 'mixed'."
                  << std::endl;
        r.mixed = true;
        r.detail += "; then " + std::string(thermalSolverPathId(path)) + ": " + detail;
        return;
    }

    r.path   = path;
    r.reason = reason;
    r.detail = detail;
    r.simd_batch_requested = simd_batch_requested;

    std::cout << "\n" << kRule << "\n"
              << "[thermal-solver] PATH:   " << pathLabel(path)
              << "  (" << thermalSolverEntryPoint(path) << ")\n"
              << "[thermal-solver] REASON: " << detail << "\n";

    if (path == ThermalSolverPath::CpuStreamed) {
        // The substitution, stated. This is the text whose absence made every
        // production-mesh CPU comparison unattributable.
        //
        // Be precise about WHAT it substitutes. The streamed path reuses the
        // scalar TIME INTEGRATOR (runThermalModelWithMatchedTimes), but it
        // builds its own forcing inline instead of routing through
        // calculateAbsorbedFlux(..., planet, ...). Calling it "the scalar
        // solver" without saying that would invite the reader to equate this
        // run with a --no-simd-batch run, which it is NOT: measured 4.82 K max
        // / 3.97 K mean apart on 11,664 of 12,800 faces of the 80N tutorial
        // crater, same config, same illumination, 4,200 output steps.
        std::cout
            << "[thermal-solver] NOTE:   the streamed path reuses the CPU SCALAR time\n"
            << "[thermal-solver]         integrator (runThermalModelWithMatchedTimes), but it\n"
            << "[thermal-solver]         is NOT equivalent to a --no-simd-batch run. It builds\n"
            << "[thermal-solver]         its absorbed flux INLINE, so:\n"
            << "[thermal-solver]           1. the albedo is a per-face CONSTANT. The configured\n"
            << "[thermal-solver]              angular-albedo model (Hayne, or Hapke via\n"
            << "[thermal-solver]              incidence_angle_albedo) is NOT applied -- the\n"
            << "[thermal-solver]              other CPU paths route through\n"
            << "[thermal-solver]              calculateAbsorbedFlux(..., planet, ...).\n"
            << "[thermal-solver]           2. no effective_albedo statistic is produced, because\n"
            << "[thermal-solver]              A_eff accumulates inside that same function.\n"
            << "[thermal-solver]           3. the equilibration indirect-flux term is not\n"
            << "[thermal-solver]              carried; this path takes no eq_indirect argument.\n";
        if (simd_batch_requested) {
            std::cout
                << "[thermal-solver] WARNING: use_simd_batch is true for this run, but the\n"
                << "[thermal-solver]         streamed path does not honor it. The SIMD batch\n"
                << "[thermal-solver]         solver was NOT used. Raise the streaming\n"
                << "[thermal-solver]         threshold (PTM_RAM_ILLUM_MAX_GB) above the\n"
                << "[thermal-solver]         estimate above to keep the SIMD batch solver.\n";
        }
    }
    std::cout << kRule << "\n" << std::endl;
}

void discloseRethermalPath(RethermalPath path, const std::string& detail) {
    Record& r = record();
    if (r.rethermal != RethermalPath::NotRun) {
        if (r.rethermal == path) return;
        std::cerr << "[thermal-solver] WARNING: the radiation-coupling re-thermal changed "
                     "solver mid-run: " << rethermalPathId(r.rethermal) << " -> "
                  << rethermalPathId(path) << " (" << detail << ").\n"
                  << "[thermal-solver]   provenance.txt will record this as 'mixed'."
                  << std::endl;
        r.rethermal_mixed = true;
        r.rethermal_detail += "; then " + std::string(rethermalPathId(path)) + ": " + detail;
        return;
    }
    r.rethermal        = path;
    r.rethermal_detail = detail;
    std::cout << "[thermal-solver] RE-THERMAL (radiation coupling): "
              << rethermalPathId(path) << " -- " << detail << std::endl;
}

ThermalSolverPath   recordedThermalSolverPath()   { return record().path; }
ThermalSolverReason recordedThermalSolverReason() { return record().reason; }
std::string         recordedThermalSolverDetail() { return record().detail; }
RethermalPath       recordedRethermalPath()       { return record().rethermal; }

bool appendThermalSolverProvenance(const std::string& output_dir) {
    const Record& r = record();

    if (r.path == ThermalSolverPath::Unset) {
        // A whole-mesh solve completed without disclosing itself. That is a
        // missing call at a NEW solver entry point, not a user error -- but it
        // is exactly the silence this module removes, so it must not pass
        // unremarked.
        std::cerr << "[thermal-solver] WARNING: no thermal solver disclosed its path; "
                     "provenance.txt will say 'none'.\n"
                     "  A solver entry point was probably added without a "
                     "discloseThermalSolverPath() call."
                  << std::endl;
    }

    const std::string path = output_dir + "/provenance.txt";
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "[thermal-solver] WARNING: cannot append the solver record to "
                  << path << "; this run's directory will not say which solver ran."
                  << std::endl;
        return false;
    }

    auto kv = [&out](const char* k, const std::string& v) { out << k << '=' << v << '\n'; };

    kv("thermal_solver_path",
       r.mixed ? "mixed" : thermalSolverPathId(r.path));
    kv("thermal_solver_entry_point",
       r.mixed ? "mixed" : thermalSolverEntryPoint(r.path));
    kv("thermal_solver_core",
       r.mixed ? "mixed" : thermalSolverCoreId(r.path));
    kv("thermal_solver_selected_by",
       r.mixed ? "mixed" : thermalSolverReasonId(r.reason));
    kv("thermal_solver_reason", r.detail.empty() ? "none" : r.detail);
    kv("thermal_solver_simd_batch_requested", r.simd_batch_requested ? "true" : "false");
    // `thermal_solver_core` names the time integrator, which for the streamed
    // path is the scalar one. That alone would read as "same as a
    // --no-simd-batch run", and it is not: the streamed path also drops the
    // equilibration phase, the angular-albedo model and the eq-indirect term.
    // A reader who has only the provenance file must be told.
    kv("thermal_solver_caveats",
       (!r.mixed && r.path == ThermalSolverPath::CpuStreamed)
           ? "constant_albedo_angular_model_ignored;no_effective_albedo_statistic;"
             "no_eq_indirect_flux"
           : "none");
    kv("thermal_rethermal_path",
       r.rethermal_mixed ? "mixed" : rethermalPathId(r.rethermal));
    kv("thermal_rethermal_reason",
       r.rethermal_detail.empty() ? "none" : r.rethermal_detail);

    out.close();
    if (!out) {
        std::cerr << "[thermal-solver] WARNING: " << path
                  << " failed to close cleanly after the solver record." << std::endl;
        return false;
    }
    return true;
}

void resetThermalSolverDisclosureForTest() { record() = Record(); }

}  // namespace ptm
