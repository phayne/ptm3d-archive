// run_provenance.h -- what a finished run records about how it was produced.
//
// `metadata.txt` was, until this file existed, the ONLY thing the engine wrote
// about a run: face count, step counts, which components were exported. It
// records no engine version, no commit, no binary checksum, no body, no mesh
// unit and no copy of the configuration, so run provenance was not
// reconstructable from the outputs -- it lived in build logs, shell history
// and hand-kept md5 notes.
//
// This writes two additional files into the run directory and changes no
// existing format:
//
//   provenance.txt    key=value, one per line, same shape as metadata.txt
//   config_used.yaml  a verbatim byte copy of the configuration file that was
//                     loaded (absent when the run took no --config)
//
// `provenance.txt` records the command line as well, because the CLI can
// override the file: the YAML copy alone would misreport a run launched with
// --thermal-inertia or --albedo.
//
// Honest limits, stated in the file itself rather than left to be discovered:
//   * the checksum is of the EXECUTABLE. If the project is ever built with
//     shared libraries, a change confined to (say) libthermal would move the
//     library and not the executable, and this checksum would not notice.
//     `engine_shared_libraries` records whether that risk is live.
//   * `config_used.yaml` is the file as read, not a dump of the resolved
//     configuration after CLI and engine_env are applied. `command_line`
//     carries the rest.

#ifndef PTM_RUN_PROVENANCE_H
#define PTM_RUN_PROVENANCE_H

#include <string>
#include <vector>

namespace ptm {

// MD5 of a file's contents, lowercase hex. Empty string if the file cannot be
// read. Streaming, 64 KiB at a time, so hashing a 30 MB executable costs a few
// tens of milliseconds and no meaningful memory.
std::string md5File(const std::string& path);

// Absolute path of the running executable, or "" if the platform hook is
// unavailable. argv0 is the fallback, resolved against the current directory.
std::string selfExecutablePath(const char* argv0);

struct RunProvenance {
    std::string output_dir;        // where to write
    std::string config_file;       // as given on the command line ("" = none)
    std::vector<std::string> argv; // the full command line, argv[0] included
    std::string body;              // resolved body name
    bool        generic_body = false;
    std::string mesh_source;       // "icosphere resolution 5" or a file path
    std::size_t mesh_faces = 0;
    std::string mesh_length_unit;  // "meters" | "kilometers" | "unknown"
    double      mesh_median_radius = 0.0;  // in the unit above
};

// Write provenance.txt (and config_used.yaml, when a config was loaded).
// Returns false and prints a WARNING on any failure -- provenance is not worth
// aborting a run over, but a silent failure would defeat the purpose.
bool writeRunProvenance(const RunProvenance& p);

}  // namespace ptm

#endif  // PTM_RUN_PROVENANCE_H
