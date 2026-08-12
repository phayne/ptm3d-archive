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

const char* thermalSolverCaveats(ThermalSolverPath p, bool simd_batch_requested) {
    // Exactly one case left: a streamed run whose config asked for the SIMD
    // batch solver got the CPU scalar integrator instead. That is a
    // SUBSTITUTION the caller did not request, which is reportable on its own
    // terms; how far the two integrators actually diverge is a separate
    // question and deliberately not asserted here. Everything else --
    // including a streamed run that asked for the scalar solver anyway, which
    // is byte-identical to what it asked for -- has nothing to disclose beyond
    // its path.
    if (p == ThermalSolverPath::CpuStreamed && simd_batch_requested) {
        return "cpu_scalar_integrator_substituted_for_requested_simd_batch";
    }
    return "none";
}

// Long-form label plus, where it matters, what the reader would otherwise get
// wrong. Only one situation still carries a caveat: a streamed run that had
// asked for the SIMD batch solver, because that -- and only that -- is a
// substitution of one integrator for another that the caller did not request.
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
                  << "[thermal-solver]   Two solvers contributed to the output field, so "
                     "no single path id describes it and any comparison against a "
                     "single-path run is unattributable.\n"
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
        // The equivalence, stated -- because it IS the answer the reader needs,
        // and because saying nothing here is what left every production-mesh
        // CPU comparison unattributable in the first place.
        //
        // This text used to name three defects: a constant albedo in place of
        // the configured angular model, no effective_albedo statistic, and no
        // equilibration indirect-flux term. All three were real, all three were
        // fixed (66c6dd7, 2aa8f67), and all three were re-measured as absent
        // on 0bb5218 on 2026-08-12. See solver_disclosure.h for the
        // falsifiers. Do not reinstate them without re-measuring: a caveat
        // that outlives its defect is a wrong label with an engine's
        // authority behind it.
        std::cout
            << "[thermal-solver] NOTE:   the streamed path is the CPU SCALAR path with an\n"
            << "[thermal-solver]         on-demand flux getter in front of it. It composes\n"
            << "[thermal-solver]         absorbed flux through the same AbsorbedFluxComposer and\n"
            << "[thermal-solver]         loops the same integrator (runThermalModelWithMatchedTimes),\n"
            << "[thermal-solver]         and its output is BYTE-IDENTICAL to a --no-simd-batch run\n"
            << "[thermal-solver]         of the same config.\n";
        if (simd_batch_requested) {
            // Left exactly as it was. It is true, it names the remedy, and it
            // carries no measurement that another branch could invalidate.
            // Characterizing HOW the two integrators differ belongs with
            // whoever settles it -- see the note on thermalSolverCaveats().
            std::cout
                << "[thermal-solver] WARNING: use_simd_batch is true for this run, but the\n"
                << "[thermal-solver]         streamed path does not honor it. The SIMD batch\n"
                << "[thermal-solver]         solver was NOT used. Raise the streaming\n"
                << "[thermal-solver]         threshold (PTM_RAM_ILLUM_MAX_GB) above the\n"
                << "[thermal-solver]         estimate above to keep the SIMD batch solver.\n";
        }
    }
    // Printed for every path, from the same function the provenance key calls,
    // so the log and provenance.txt cannot disagree about the caveat.
    std::cout << "[thermal-solver] CAVEATS: "
              << thermalSolverCaveats(path, simd_batch_requested) << "\n";
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
    // path is the scalar one -- and a streamed run IS byte-identical to a
    // --no-simd-batch run, so for a reader who asked for the scalar solver
    // there is nothing further to say. The one thing that still needs saying is
    // the integrator SUBSTITUTION, when the config asked for the SIMD batch
    // solver and the memory threshold overrode it. Same function as the banner,
    // so the two cannot drift apart.
    kv("thermal_solver_caveats",
       r.mixed ? "mixed"
               : thermalSolverCaveats(r.path, r.simd_batch_requested));
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
