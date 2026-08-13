// thermal_state_io.h -- the thermal-state checkpoint ("PTMT") format.
//
// ONE implementation, shared by every solver path. Before this file the reader
// and the writer lived inline in the GPU persistent solver and were duplicated
// a third time by the PTM-57 post-coupling rewrite; the CPU paths had no copy
// at all, which is BUGS.md PTM-76.
//
// ---------------------------------------------------------------------------
// FORMAT (v1 -- unchanged; every banked state on disk is this layout)
// ---------------------------------------------------------------------------
//     uint32   magic     = 0x50544D54   ('PTMT')
//     uint32   n_faces
//     uint32   n_layers
//     float32  T[n_layers * n_faces]    LAYER-MAJOR:  T[l * n_faces + f]
//
// The temperature block is float32 on every path, including the float64 CPU
// solvers. That is the format's precision, not a solver's: a CPU column written
// to a checkpoint and restored is quantized to float32 exactly once, at the
// write. See the note on the round trip below.
//
// ---------------------------------------------------------------------------
// WHAT THE FORMAT DOES NOT CARRY -- read this before trusting a restore
// ---------------------------------------------------------------------------
// The header records the SHAPE (n_faces x n_layers) and nothing else. It does
// NOT record the depth grid the column was written on: not z[], not z_max, not
// the (m, n, b) grid parameters, not the body, not the thermophysical
// properties that set the skin depth. Two runs whose grids differ but whose
// LAYER COUNT happens to agree will load each other's checkpoints without
// complaint, and the restored column will be silently reinterpreted onto the
// wrong depths.
//
// That gap is deliberate here and is NOT closed by this file: extending the
// header would invalidate every banked state on disk, which is Paul's call and
// not a refactor's. What this file does instead is make the gap AUDIBLE --
// gridFingerprint() prints the live grid at both write and restore so the two
// can be compared in the logs, and the message says outright which quantities
// were not verified.
//
// ---------------------------------------------------------------------------
// ROUND TRIP
// ---------------------------------------------------------------------------
// write(x) then read() returns bit-identical float32 for every value: no
// byte-swapping, no text formatting, no lossy transform. A float64 CPU column
// is rounded to float32 by the CAST at the write boundary, once. Restoring it
// gives back exactly the float32 that was written, widened to float64 exactly
// (every float32 is representable in float64). So a CPU->file->CPU round trip
// reproduces the written FILE bit-for-bit, and reproduces the in-memory float64
// column to float32 precision. State that boundary; do not hide it in a
// tolerance.

#ifndef PTM_THERMAL_STATE_IO_H
#define PTM_THERMAL_STATE_IO_H

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace thermal {
namespace state_io {

// 'PTMT'
constexpr std::uint32_t kMagic = 0x50544D54u;

// A one-line description of the depth grid a checkpoint is being written on or
// restored onto. The file cannot carry this (see the header note), so it is
// logged at both ends and compared by eye or by diff.
inline std::string gridFingerprint(const double* z, std::size_t n) {
    std::ostringstream os;
    if (!z || n == 0) { os << "n_layers=0"; return os.str(); }
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "n_layers=%zu z[0]=%.9g dz[0]=%.9g z_max=%.9g",
                  n, z[0], (n > 1 ? z[1] - z[0] : 0.0), z[n - 1]);
    os << buf;
    return os.str();
}

// Read a checkpoint and verify it against the live shape.
// Returns true and fills `out` (size n_layers*n_faces, layer-major) on success.
// Returns false and fills `err` with a complete, actionable message otherwise.
inline bool read(const std::string& path,
                 std::size_t n_faces,
                 std::size_t n_layers,
                 std::vector<float>& out,
                 std::string& err) {
    err.clear();
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        err = "cannot open '" + path + "' for reading";
        return false;
    }
    std::uint32_t magic = 0, file_faces = 0, file_layers = 0;
    is.read(reinterpret_cast<char*>(&magic),       sizeof(std::uint32_t));
    is.read(reinterpret_cast<char*>(&file_faces),  sizeof(std::uint32_t));
    is.read(reinterpret_cast<char*>(&file_layers), sizeof(std::uint32_t));
    if (!is) {
        err = "'" + path + "' is shorter than the 12-byte PTMT header";
        return false;
    }
    if (magic != kMagic) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "bad magic 0x%08X (expected 0x%08X, 'PTMT')",
                      magic, kMagic);
        err = std::string(buf) + " in '" + path + "'";
        return false;
    }
    if (file_faces != static_cast<std::uint32_t>(n_faces) ||
        file_layers != static_cast<std::uint32_t>(n_layers)) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "shape mismatch: file is %u faces x %u layers, this run "
                      "is %zu x %zu",
                      file_faces, file_layers, n_faces, n_layers);
        err = std::string(buf) + " ('" + path + "')";
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(file_faces) * file_layers;
    out.resize(count);
    is.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    if (!is) {
        err = "short read on '" + path + "' (header promised " +
              std::to_string(count) + " float32 values)";
        out.clear();
        return false;
    }
    return true;
}

// Write a checkpoint. `data` must be n_layers*n_faces, layer-major.
inline bool write(const std::string& path,
                  std::size_t n_faces,
                  std::size_t n_layers,
                  const std::vector<float>& data,
                  std::string& err) {
    err.clear();
    if (data.size() != n_faces * n_layers) {
        err = "refusing to write '" + path + "': buffer holds " +
              std::to_string(data.size()) + " values, header would claim " +
              std::to_string(n_faces * n_layers);
        return false;
    }
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        err = "cannot open '" + path + "' for writing";
        return false;
    }
    const std::uint32_t magic  = kMagic;
    const std::uint32_t faces  = static_cast<std::uint32_t>(n_faces);
    const std::uint32_t layers = static_cast<std::uint32_t>(n_layers);
    os.write(reinterpret_cast<const char*>(&magic),  sizeof(std::uint32_t));
    os.write(reinterpret_cast<const char*>(&faces),  sizeof(std::uint32_t));
    os.write(reinterpret_cast<const char*>(&layers), sizeof(std::uint32_t));
    os.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(float)));
    os.flush();
    if (!os.good()) {
        err = "short write on '" + path + "'";
        return false;
    }
    return true;
}

}  // namespace state_io
}  // namespace thermal

#endif  // PTM_THERMAL_STATE_IO_H
