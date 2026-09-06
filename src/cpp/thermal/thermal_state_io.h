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
//
// ---------------------------------------------------------------------------
// THE WRITE IS ATOMIC (BUGS.md PTM-STATE-ATOMIC)
// ---------------------------------------------------------------------------
// write() never truncates the destination. It creates `<path>.tmp-<pid>` in the
// SAME DIRECTORY (therefore the same filesystem), writes the whole file there,
// fsync()s it, and only then rename()s it onto `<path>`. POSIX rename within a
// filesystem is atomic, so at every instant `<path>` is either the previous
// complete checkpoint or the new complete one -- never a prefix of either.
//
// Before this, write() opened the destination with ios::trunc: the destination
// was destroyed the moment the write began, and a kill, a full disk or a crash
// mid-stream left a truncated file. read() refuses such a file (good) -- but
// the checkpoint it refuses is the one that was already there. The PTM-57
// post-coupling re-archive (ptm_thermal_sim.cpp) is the sharp case: it is a
// read-modify-write IN PLACE, so it destroyed a VALID checkpoint in order to
// write a better one. It reads the whole file into memory before calling
// write(), so the temp-then-rename here makes that path a write-aside-and-swap
// with no change at the call site.
//
// WHAT IS AND IS NOT GUARANTEED. The FILE is fsync'd before the rename, so the
// new name can never resolve to unwritten blocks. The DIRECTORY is deliberately
// NOT fsync'd: without it, a machine crash inside the writeback window resolves
// `<path>` to either the old name or the new one -- both complete files -- which
// is exactly the property this function exists to provide, at no cost. Adding
// the directory fsync would only decide WHICH of two valid states survives.
// On macOS, fsync() does not flush the drive's own write cache (F_FULLFSYNC
// does, far more slowly); that distinction matters only for power loss, and
// under power loss the old checkpoint still survives.
//
// KNOWN, DELIBERATE CONSEQUENCE: a process killed between the open and the
// rename leaves `<path>.tmp-<pid>` behind. Nothing reads or globs that name
// (the orchestrators glob `thermal_state_after_chunk_*.bin`, which the suffix
// does not match), and the alternative -- deleting files this function did not
// create -- is worse. write() cleans up its own temp on every failure it can
// still observe.

#ifndef PTM_THERMAL_STATE_IO_H
#define PTM_THERMAL_STATE_IO_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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

namespace detail {

// Write the whole buffer or say why not. ::write() is allowed to return a short
// count on a regular file (and does, at an RLIMIT_FSIZE or a full disk), so a
// single unchecked call is a silent-truncation path.
inline bool writeAll(int fd, const void* buf, std::size_t n, std::string& err) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            err = std::string("write failed: ") + std::strerror(errno);
            return false;
        }
        if (w == 0) {
            err = "write made no progress with " + std::to_string(n) +
                  " bytes still to go";
            return false;
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

}  // namespace detail

// Write a checkpoint. `data` must be n_layers*n_faces, layer-major.
//
// ATOMIC: temp file in the same directory, fsync, rename. See the block at the
// top of this file for the guarantee and its limits. The BYTES are unchanged
// from the pre-atomic writer -- 12-byte header then the float32 body, nothing
// added -- so a completed write is byte-identical to what this function has
// always produced.
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

    // Same directory as the destination, so rename(2) stays within one
    // filesystem and is atomic. O_EXCL so a temp left behind by an earlier
    // killed process is never silently reused or clobbered.
    std::string tmp;
    int fd = -1;
    const long pid = static_cast<long>(::getpid());
    for (int attempt = 0; attempt < 1024; ++attempt) {
        tmp = path + ".tmp-" + std::to_string(pid);
        if (attempt > 0) tmp += "-" + std::to_string(attempt);
        fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd >= 0) break;
        if (errno != EEXIST) {
            err = "cannot create the temporary file '" + tmp + "' for '" +
                  path + "': " + std::strerror(errno);
            return false;
        }
    }
    if (fd < 0) {
        err = "cannot create a temporary file beside '" + path +
              "': 1024 candidate names all exist (leftovers from killed "
              "writers?)";
        return false;
    }

    // Any failure from here on must leave the DESTINATION untouched and take
    // the temp with it.
    auto fail = [&](const std::string& what) -> bool {
        err = what + " (writing '" + path + "' via '" + tmp + "')";
        ::close(fd);
        ::unlink(tmp.c_str());
        return false;
    };

    const std::uint32_t hdr[3] = {
        kMagic,
        static_cast<std::uint32_t>(n_faces),
        static_cast<std::uint32_t>(n_layers)
    };
    std::string w_err;
    if (!detail::writeAll(fd, hdr, sizeof(hdr), w_err))
        return fail("header " + w_err);
    if (!data.empty() &&
        !detail::writeAll(fd, data.data(), data.size() * sizeof(float), w_err))
        return fail("body " + w_err);

    // fsync BEFORE the rename: the new name must never resolve to blocks that
    // were never written. ENOSPC and EIO commonly surface here, not at write().
    while (::fsync(fd) != 0) {
        if (errno == EINTR) continue;
        return fail(std::string("fsync failed: ") + std::strerror(errno));
    }
    // close() can also report a deferred write error; it is not a formality.
    if (::close(fd) != 0) {
        const std::string why = std::strerror(errno);
        fd = -1;
        err = "close failed: " + why + " (writing '" + path + "' via '" +
              tmp + "')";
        ::unlink(tmp.c_str());
        return false;
    }
    fd = -1;

    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const std::string why = std::strerror(errno);
        ::unlink(tmp.c_str());
        err = "cannot rename '" + tmp + "' onto '" + path + "': " + why +
              " -- the previous checkpoint at that path is UNCHANGED";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// PROBE -- "what is this file", without being told the shape first
// ---------------------------------------------------------------------------
// read() verifies a file against a shape the caller already knows. An inspector
// (tools/ptm_state_info.cpp) knows nothing, so it needs the header read and the
// length rule to live HERE rather than as a second implementation of the
// format somewhere else -- which is the duplication this file was created to
// end.
struct FileInfo {
    std::uint32_t magic    = 0;
    std::uint32_t n_faces  = 0;
    std::uint32_t n_layers = 0;
    std::uint64_t values   = 0;   // n_faces * n_layers
    std::uint64_t bytes_expected = 0;   // 12 + values*4, for the v1 format
    std::uint64_t bytes_actual   = 0;
    std::uint64_t bytes_extra    = 0;   // actual - expected, when the file is longer
};

// True if the file carries a well-formed PTMT header AND holds at least the
// bytes that header promises. A LONGER file is accepted and reported through
// `bytes_extra` -- it is not silently ignored, because a future v2 trailer is
// exactly what a longer file would be.
inline bool probe(const std::string& path, FileInfo& out, std::string& err) {
    err.clear();
    out = FileInfo();

    std::ifstream is(path, std::ios::binary);
    if (!is) {
        err = "cannot open '" + path + "' for reading";
        return false;
    }
    is.read(reinterpret_cast<char*>(&out.magic),    sizeof(std::uint32_t));
    is.read(reinterpret_cast<char*>(&out.n_faces),  sizeof(std::uint32_t));
    is.read(reinterpret_cast<char*>(&out.n_layers), sizeof(std::uint32_t));
    if (!is) {
        err = "'" + path + "' is shorter than the 12-byte PTMT header";
        return false;
    }
    if (out.magic != kMagic) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "bad magic 0x%08X (expected 0x%08X, 'PTMT')",
                      out.magic, kMagic);
        err = std::string(buf) + " in '" + path + "'";
        return false;
    }
    if (out.n_faces == 0 || out.n_layers == 0) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "degenerate shape %u faces x %u layers: a checkpoint "
                      "with no faces or no layers carries no state",
                      out.n_faces, out.n_layers);
        err = std::string(buf) + " in '" + path + "'";
        return false;
    }

    out.values = static_cast<std::uint64_t>(out.n_faces) * out.n_layers;
    out.bytes_expected = 12ull + out.values * sizeof(float);

    is.seekg(0, std::ios::end);
    if (!is) {
        err = "cannot measure the length of '" + path + "'";
        return false;
    }
    out.bytes_actual = static_cast<std::uint64_t>(is.tellg());

    if (out.bytes_actual < out.bytes_expected) {
        err = "'" + path + "' is TRUNCATED: the header promises " +
              std::to_string(out.bytes_expected) + " bytes (" +
              std::to_string(out.n_faces) + " faces x " +
              std::to_string(out.n_layers) + " layers) and the file holds " +
              std::to_string(out.bytes_actual);
        return false;
    }
    out.bytes_extra = out.bytes_actual - out.bytes_expected;
    return true;
}

}  // namespace state_io
}  // namespace thermal

#endif  // PTM_THERMAL_STATE_IO_H
