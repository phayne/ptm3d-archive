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
// FORMAT (v2 -- an OPTIONAL TRAILER appended after the v1 body)
// ---------------------------------------------------------------------------
//     [0 .. 12)                v1 header, byte-for-byte UNCHANGED
//     [12 .. 12 + F*L*4)       v1 body,   byte-for-byte UNCHANGED
//     [12 + F*L*4 .. EOF)      v2 trailer, present iff the file is LONGER than
//                              the v1 length the header implies
//
//     uint32   trailer_magic = 0x50544D58 ('PTMX')
//     uint32   version       = 2
//     uint32   trailer_bytes             total, including magic and crc32
//     uint32   n_layers                  must equal the v1 header's n_layers
//     float64  z[n_layers]               THE DEPTH GRID (closes PTM-STATE-GRID-BLIND)
//     float64  et_seconds                SPICE ET this column is valid at
//                                        (closes PTM-STATE-NO-EPOCH)
//     float64  output_cadence_s          the writing run's output cadence
//     float64  refgrid_H
//     int32    grid_m, grid_n, grid_b, reserved0
//     float64  mesh_geomhash
//     float64  w0_mean, w0_p05, w0_p95   see the W(0) note below
//     uint64   config_identity_hash      thermal/state_identity.h
//     char[32] body_name                 NUL-padded
//     char[48] engine_git_sha            NUL-padded
//     uint32   flags                     see `flag::` below
//     uint32   crc32                     over EVERY byte of the file before it
//
// Every float64 field may be NaN and every integer field may be its sentinel
// (-1 for the grid ints, 0 for the hash, "" for the strings). That is the
// format's way of saying NOT RECORDED, and it is deliberately not zero for the
// floats: a NaN cannot be mistaken for a measurement, and every comparison
// against it fails loudly instead of matching by accident.
//
// BACKWARD COMPATIBILITY IS THE POINT. Every state banked before this existed
// is a v1 file, loads unchanged, and is reported as "v1 -- these quantities
// were NOT verified", naming each one. Forward: `trailer_bytes` and `version`
// let a later revision append fields; a reader that finds a LONGER trailer than
// it knows reads the prefix it understands and says so.
//
// ---------------------------------------------------------------------------
// WHAT THE v1 FORMAT DOES NOT CARRY -- and what v2 now does
// ---------------------------------------------------------------------------
// The v1 header records the SHAPE (n_faces x n_layers) and nothing else. It
// does NOT record the depth grid the column was written on: not z[], not z_max,
// not the (m, n, b) grid parameters, not the body, not the thermophysical
// properties that set the skin depth, and not the EPOCH. Two runs whose grids
// differ but whose LAYER COUNT happens to agree will load each other's
// checkpoints without complaint, and the restored column is silently
// reinterpreted onto the wrong depths. Those are BUGS.md PTM-STATE-GRID-BLIND
// and PTM-STATE-NO-EPOCH.
//
// The v2 trailer closes both FOR FILES THAT CARRY IT. For a v1 file the gap is
// unchanged and is made AUDIBLE instead: gridFingerprint() prints the live grid
// at both write and restore, and checkRestore() emits one `unverified` line per
// quantity the file could not answer.
//
// ---------------------------------------------------------------------------
// W(0) IS RESERVED, NOT POPULATED, AND SAYS SO
// ---------------------------------------------------------------------------
// w0_* is the surface swing potential <T^4> - <T>^4, an input to the warm-start
// admissibility rule. This engine version writes NaN into all three, always.
// The reason is worth stating rather than hiding: computing it needs a
// reduction over the output-window surface time series, which exists on some
// solver paths at the moment of the write and not on others -- so populating it
// where it is available would make a PRESENT field mean different things
// depending on which solver ran. A field that is sometimes a measurement and
// sometimes a coincidence is the silent-substitution class. NaN everywhere is
// honest, and the epoch clause below already refuses every case W(0) would
// have.
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
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
// 'PTMX' -- the v2 trailer. A DIFFERENT magic from the file's, on purpose: it
// is what tells a reader that the extra bytes probe() reports are a trailer and
// not somebody's appended notes.
constexpr std::uint32_t kTrailerMagic   = 0x50544D58u;
constexpr std::uint32_t kTrailerVersion = 2u;

// Bytes of trailer that do NOT depend on n_layers. Serialized field by field
// below -- never as a struct memcpy, so no compiler's padding can change the
// on-disk layout.
constexpr std::uint32_t kTrailerFixedBytes =
    4 + 4 + 4 + 4        // magic, version, trailer_bytes, n_layers
  + 8 + 8 + 8            // et_seconds, output_cadence_s, refgrid_H
  + 4 + 4 + 4 + 4        // grid_m, grid_n, grid_b, reserved0
  + 8                    // mesh_geomhash
  + 8 + 8 + 8            // w0_mean, w0_p05, w0_p95
  + 8                    // config_identity_hash
  + 32 + 48              // body_name, engine_git_sha
  + 4 + 4;               // flags, crc32

constexpr std::size_t kBodyNameBytes = 32;
constexpr std::size_t kGitShaBytes   = 48;

namespace flag {
// bit0 -- the state is the POST-radiation-coupling column (BUGS.md PTM-57).
//         Clear means it is iter-0's un-coupled column, which the next chunk
//         would resume radiation-fed faces from.
constexpr std::uint32_t kPostCouplingComplete = 1u << 0;
// bit1 -- EVERY face carries the coupled column, not just the re-thermal'd
//         selection. Clear with bit0 set means a partial patch (which is the
//         correct answer for untouched faces, but the reader should know).
constexpr std::uint32_t kAllFacesPatched = 1u << 1;
// bit2 -- the writing run had thermal coupling_select_dt > 0, so the
//         re-thermal selector could freeze a face that had nonzero indirect
//         flux (Fermi F4). A warm start must refuse such a state.
constexpr std::uint32_t kCouplingSelectDtPositive = 1u << 2;
// bit3 -- RESERVED for batch-completion partials (plan phase 1D'). Nothing
//         writes it today; a reader REFUSES it, so the bit can never be
//         introduced later and silently load as a whole-mesh state.
constexpr std::uint32_t kBatchCompletionPartial = 1u << 3;
}  // namespace flag

constexpr double kNotRecorded = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) -- the same one zlib and PNG
// use, so a value here can be checked by any other tool.
//
// It covers the WHOLE file, not just the trailer. The failure it exists for is
// OneDrive's lazy-fetch placeholder: a file of exactly the right LENGTH whose
// contents are zeros. Length checks pass on that; magic passes on that if the
// first block is real; only a checksum over the body catches it.
// ---------------------------------------------------------------------------
// Built exactly once. A function-local static with its own `built` flag would
// race if two threads ever checksummed at the same time; C++11 guarantees the
// initialization of a function-local static is thread-safe, so the table is
// built inside the initializer rather than beside it.
inline const std::uint32_t* crc32Table() {
    static const std::vector<std::uint32_t> t = [] {
        std::vector<std::uint32_t> v(256);
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            v[i] = c;
        }
        return v;
    }();
    return t.data();
}

// Running CRC. Seed with crc32Init(), feed any number of chunks, finish with
// crc32Final(). Chunking must not change the answer, and does not.
constexpr std::uint32_t crc32Init() { return 0xFFFFFFFFu; }

inline std::uint32_t crc32Update(std::uint32_t crc, const void* buf,
                                 std::size_t n) {
    const std::uint32_t* t = crc32Table();
    const unsigned char* p = static_cast<const unsigned char*>(buf);
    for (std::size_t i = 0; i < n; ++i)
        crc = t[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

inline std::uint32_t crc32Final(std::uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

// ---------------------------------------------------------------------------
// The v2 trailer, in memory.
// ---------------------------------------------------------------------------
struct Trailer {
    std::vector<double> z;                    // the depth grid, n_layers long
    double et_seconds       = kNotRecorded;   // epoch this column is valid at
    double output_cadence_s = kNotRecorded;   // the writing run's cadence
    double refgrid_H        = kNotRecorded;
    int    grid_m = -1, grid_n = -1, grid_b = -1;
    double mesh_geomhash = kNotRecorded;
    double w0_mean = kNotRecorded, w0_p05 = kNotRecorded, w0_p95 = kNotRecorded;
    std::uint64_t config_identity_hash = 0;
    std::string body_name;
    std::string engine_git_sha;
    std::uint32_t flags = 0;

    // Read-back only. Set by readTrailer(); ignored by write(), which always
    // computes the checksum from the bytes it is about to lay down.
    std::uint32_t crc32_stored = 0;
    std::uint32_t version_read = 0;
    std::uint32_t bytes_read   = 0;

    std::uint32_t sizeBytes() const {
        return kTrailerFixedBytes +
               static_cast<std::uint32_t>(z.size() * sizeof(double));
    }
};

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

// --- trailer serialization ------------------------------------------------
// Field by field, little-endian on every platform this project builds for. A
// struct memcpy would let a compiler's padding rules into the on-disk format.
inline void putU32(std::vector<unsigned char>& b, std::uint32_t v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    b.insert(b.end(), p, p + 4);
}
inline void putI32(std::vector<unsigned char>& b, std::int32_t v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    b.insert(b.end(), p, p + 4);
}
inline void putU64(std::vector<unsigned char>& b, std::uint64_t v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    b.insert(b.end(), p, p + 8);
}
inline void putF64(std::vector<unsigned char>& b, double v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    b.insert(b.end(), p, p + 8);
}
inline void putStr(std::vector<unsigned char>& b, const std::string& s,
                   std::size_t width) {
    // Truncate rather than overflow, and never write the terminator past the
    // field: a longer name is a labeling problem, not a corruption one.
    const std::size_t n = (s.size() < width) ? s.size() : (width - 1);
    b.insert(b.end(), s.begin(), s.begin() + static_cast<long>(n));
    b.insert(b.end(), width - n, static_cast<unsigned char>(0));
}

// Reader side. `off` advances; `ok` latches false on any overrun so a
// malformed trailer produces one refusal, not a cascade of garbage fields.
struct Cursor {
    const unsigned char* p = nullptr;
    std::size_t n = 0, off = 0;
    bool ok = true;
    bool take(void* dst, std::size_t k) {
        if (!ok || off + k > n) { ok = false; return false; }
        std::memcpy(dst, p + off, k);
        off += k;
        return true;
    }
    std::uint32_t u32() { std::uint32_t v = 0; take(&v, 4); return v; }
    std::int32_t  i32() { std::int32_t  v = 0; take(&v, 4); return v; }
    std::uint64_t u64() { std::uint64_t v = 0; take(&v, 8); return v; }
    double        f64() { double v = 0;        take(&v, 8); return v; }
    std::string   str(std::size_t width) {
        std::vector<char> tmp(width, '\0');
        if (!take(tmp.data(), width)) return std::string();
        // NUL-terminated within the field; a field that is all non-NUL is
        // still readable, just full.
        std::size_t len = 0;
        while (len < width && tmp[len] != '\0') ++len;
        return std::string(tmp.data(), len);
    }
};

// Serialize everything EXCEPT the trailing crc32, which the caller appends
// once it has folded this buffer into the running checksum.
inline std::vector<unsigned char> serializeTrailerPrefix(const Trailer& t) {
    std::vector<unsigned char> b;
    b.reserve(t.sizeBytes());
    putU32(b, kTrailerMagic);
    putU32(b, kTrailerVersion);
    putU32(b, t.sizeBytes());
    putU32(b, static_cast<std::uint32_t>(t.z.size()));
    for (double z : t.z) putF64(b, z);
    putF64(b, t.et_seconds);
    putF64(b, t.output_cadence_s);
    putF64(b, t.refgrid_H);
    putI32(b, t.grid_m);
    putI32(b, t.grid_n);
    putI32(b, t.grid_b);
    putI32(b, 0);                       // reserved0
    putF64(b, t.mesh_geomhash);
    putF64(b, t.w0_mean);
    putF64(b, t.w0_p05);
    putF64(b, t.w0_p95);
    putU64(b, t.config_identity_hash);
    putStr(b, t.body_name,      kBodyNameBytes);
    putStr(b, t.engine_git_sha, kGitShaBytes);
    putU32(b, t.flags);
    return b;                            // crc32 not yet appended
}

}  // namespace detail

// Write a checkpoint. `data` must be n_layers*n_faces, layer-major.
//
// ATOMIC: temp file in the same directory, fsync, rename. See the block at the
// top of this file for the guarantee and its limits.
//
// `trailer` == nullptr writes a v1 file whose bytes are IDENTICAL to what this
// function produced before v2 existed -- 12-byte header then the float32 body,
// nothing added. The argument is mandatory rather than defaulted on purpose:
// every call site has to decide whether the state it is banking can describe
// itself, and a site that quietly downgraded a v2 file to v1 (the PTM-57
// re-archive is a read-modify-WRITE and would have) is exactly the silent loss
// this signature makes impossible to write by accident.
inline bool write(const std::string& path,
                  std::size_t n_faces,
                  std::size_t n_layers,
                  const std::vector<float>& data,
                  const Trailer* trailer,
                  std::string& err) {
    err.clear();
    if (data.size() != n_faces * n_layers) {
        err = "refusing to write '" + path + "': buffer holds " +
              std::to_string(data.size()) + " values, header would claim " +
              std::to_string(n_faces * n_layers);
        return false;
    }
    if (trailer && trailer->z.size() != n_layers) {
        err = "refusing to write '" + path + "': the v2 trailer carries " +
              std::to_string(trailer->z.size()) +
              " depth nodes but the header would claim " +
              std::to_string(n_layers) +
              " layers. A trailer that disagrees with its own header is worse "
              "than no trailer.";
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
    // The checksum covers everything before it, in write order, so it is
    // accumulated as the bytes go out rather than by re-reading the file.
    std::uint32_t crc = crc32Init();
    if (!detail::writeAll(fd, hdr, sizeof(hdr), w_err))
        return fail("header " + w_err);
    crc = crc32Update(crc, hdr, sizeof(hdr));
    if (!data.empty()) {
        if (!detail::writeAll(fd, data.data(), data.size() * sizeof(float),
                              w_err))
            return fail("body " + w_err);
        crc = crc32Update(crc, data.data(), data.size() * sizeof(float));
    }
    if (trailer) {
        std::vector<unsigned char> tb = detail::serializeTrailerPrefix(*trailer);
        crc = crc32Update(crc, tb.data(), tb.size());
        const std::uint32_t crc_final = crc32Final(crc);
        detail::putU32(tb, crc_final);
        if (tb.size() != trailer->sizeBytes())
            return fail("internal: trailer serialized to " +
                        std::to_string(tb.size()) + " bytes, sizeBytes() says " +
                        std::to_string(trailer->sizeBytes()));
        if (!detail::writeAll(fd, tb.data(), tb.size(), w_err))
            return fail("trailer " + w_err);
    }

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
// `bytes_extra` -- it is not silently ignored, because a v2 identity trailer is
// exactly what those extra bytes are. probe() deliberately does NOT interpret
// them: it answers "what shape is this", and readTrailer() answers "and what
// does it say about itself".
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

// ---------------------------------------------------------------------------
// READ THE v2 TRAILER
// ---------------------------------------------------------------------------
// Three outcomes, and they are NOT collapsed into a bool, because "this file
// has no trailer" and "this file has a trailer that is wrong" must lead to
// different behavior: the first is a v1 file and loads with warnings, the
// second is corruption and refuses.
enum class TrailerStatus {
    Absent,      // a v1 file: exactly the length its header implies
    Ok,          // a well-formed v2 trailer whose CRC32 verifies
    Malformed    // extra bytes that are not a trailer, or a trailer that fails
                 // its own magic / size / layer count / checksum
};

// Streams the WHOLE file to verify the checksum, which is the point: a
// length-correct, zero-filled OneDrive placeholder passes probe() and fails
// here. Cost is one sequential read of the file.
inline TrailerStatus readTrailer(const std::string& path, Trailer& out,
                                 std::string& err) {
    err.clear();
    out = Trailer();

    FileInfo info;
    if (!probe(path, info, err)) return TrailerStatus::Malformed;
    if (info.bytes_extra == 0) return TrailerStatus::Absent;

    if (info.bytes_extra < kTrailerFixedBytes) {
        err = "'" + path + "' has " + std::to_string(info.bytes_extra) +
              " bytes past the v1 body, which is less than the " +
              std::to_string(kTrailerFixedBytes) +
              "-byte fixed part of a PTMT v2 trailer. Those bytes are not a "
              "trailer this reader can interpret.";
        return TrailerStatus::Malformed;
    }

    std::ifstream is(path, std::ios::binary);
    if (!is) { err = "cannot re-open '" + path + "'"; return TrailerStatus::Malformed; }

    // One pass: checksum the v1 part while streaming, then hold the trailer.
    std::uint32_t crc = crc32Init();
    std::vector<char> buf(1u << 20);
    std::uint64_t left = info.bytes_expected;
    while (left > 0) {
        const std::size_t want =
            static_cast<std::size_t>(left < buf.size() ? left : buf.size());
        is.read(buf.data(), static_cast<std::streamsize>(want));
        if (static_cast<std::size_t>(is.gcount()) != want) {
            err = "short read while checksumming '" + path + "'";
            return TrailerStatus::Malformed;
        }
        crc = crc32Update(crc, buf.data(), want);
        left -= want;
    }

    std::vector<unsigned char> tb(static_cast<std::size_t>(info.bytes_extra));
    is.read(reinterpret_cast<char*>(tb.data()),
            static_cast<std::streamsize>(tb.size()));
    if (static_cast<std::size_t>(is.gcount()) != tb.size()) {
        err = "short read on the trailer of '" + path + "'";
        return TrailerStatus::Malformed;
    }

    detail::Cursor c{tb.data(), tb.size(), 0, true};
    const std::uint32_t tmagic = c.u32();
    if (tmagic != kTrailerMagic) {
        char b2[160];
        std::snprintf(b2, sizeof(b2),
                      "the %llu bytes past the v1 body start with 0x%08X, not "
                      "the PTMT v2 trailer magic 0x%08X ('PTMX')",
                      (unsigned long long)info.bytes_extra, tmagic,
                      kTrailerMagic);
        err = std::string(b2) + " in '" + path + "'";
        return TrailerStatus::Malformed;
    }
    out.version_read = c.u32();
    out.bytes_read   = c.u32();
    const std::uint32_t t_layers = c.u32();

    if (out.version_read < kTrailerVersion) {
        err = "'" + path + "' carries a PTMT trailer version " +
              std::to_string(out.version_read) + "; this build reads version " +
              std::to_string(kTrailerVersion) + " and will not guess at an "
              "older layout.";
        return TrailerStatus::Malformed;
    }
    if (out.bytes_read != info.bytes_extra) {
        err = "'" + path + "' trailer claims " + std::to_string(out.bytes_read) +
              " bytes but " + std::to_string(info.bytes_extra) +
              " bytes follow the v1 body";
        return TrailerStatus::Malformed;
    }
    if (t_layers != info.n_layers) {
        err = "'" + path + "' trailer describes " + std::to_string(t_layers) +
              " depth nodes but the v1 header claims " +
              std::to_string(info.n_layers) + " layers";
        return TrailerStatus::Malformed;
    }
    // A NEWER writer may append fields; say so plainly instead of failing on a
    // format that is deliberately extensible. The prefix this build understands
    // is still read and checked.
    //
    // This disclosure is INDEPENDENT of the size check on purpose. A future v3
    // that keeps the byte layout but redefines a field -- populating w0_*, or
    // changing the mesh-hash formula -- has the same length, and nesting the
    // version notice inside the size test would let exactly that file be read
    // silently as if it were understood.
    if (out.version_read > kTrailerVersion) {
        err = "'" + path + "' was written by a NEWER PTMT trailer version " +
              std::to_string(out.version_read) + " (" +
              std::to_string(out.bytes_read) +
              " bytes); reading the version-" +
              std::to_string(kTrailerVersion) + " prefix only. Fields this "
              "build knows may have been redefined.";
        // not fatal -- but the caller must print this
    } else if (out.bytes_read !=
               kTrailerFixedBytes +
                   t_layers * static_cast<std::uint32_t>(sizeof(double))) {
        err = "'" + path + "' trailer is " + std::to_string(out.bytes_read) +
              " bytes; version " + std::to_string(kTrailerVersion) +
              " with " + std::to_string(t_layers) + " layers must be " +
              std::to_string(kTrailerFixedBytes +
                             t_layers * sizeof(double)) + ".";
        return TrailerStatus::Malformed;
    }

    out.z.resize(t_layers);
    for (std::uint32_t i = 0; i < t_layers; ++i) out.z[i] = c.f64();
    out.et_seconds       = c.f64();
    out.output_cadence_s = c.f64();
    out.refgrid_H        = c.f64();
    out.grid_m           = c.i32();
    out.grid_n           = c.i32();
    out.grid_b           = c.i32();
    (void)c.i32();                        // reserved0
    out.mesh_geomhash    = c.f64();
    out.w0_mean          = c.f64();
    out.w0_p05           = c.f64();
    out.w0_p95           = c.f64();
    out.config_identity_hash = c.u64();
    out.body_name        = c.str(kBodyNameBytes);
    out.engine_git_sha   = c.str(kGitShaBytes);
    out.flags            = c.u32();
    if (!c.ok) {
        err = "'" + path + "' trailer ended mid-field";
        return TrailerStatus::Malformed;
    }

    // The checksum is the LAST four bytes of the trailer, not the four after
    // the fields this build happens to know. Taking it from `c.off` instead
    // would make a NEWER writer's file -- which appends fields between the
    // known prefix and the checksum -- fail with a CRC error rather than the
    // honest "written by a newer version" message above. FORMAT CONTRACT: any
    // later revision keeps the v2 field order as a prefix, appends after it,
    // and leaves crc32 last.
    std::memcpy(&out.crc32_stored, tb.data() + (tb.size() - 4), 4);
    crc = crc32Update(crc, tb.data(), tb.size() - 4);
    const std::uint32_t crc_computed = crc32Final(crc);
    if (crc_computed != out.crc32_stored) {
        char b2[224];
        std::snprintf(b2, sizeof(b2),
                      "CRC32 MISMATCH: the file checksums to 0x%08X, its own "
                      "trailer records 0x%08X. The bytes on disk are not the "
                      "bytes that were written (truncation, a partial "
                      "cloud-sync fetch, or edit-in-place)",
                      crc_computed, out.crc32_stored);
        err = std::string(b2) + " -- '" + path + "'";
        return TrailerStatus::Malformed;
    }
    return TrailerStatus::Ok;
}

// ---------------------------------------------------------------------------
// RESTORE ADMISSIBILITY
// ---------------------------------------------------------------------------
// What the RUN has, to be compared against what the FILE says. Fields left at
// their sentinels are simply not compared, and every skipped comparison is
// reported as `unverified` rather than passing quietly.
struct Identity {
    std::vector<double> z;                    // the live depth grid
    double et_first         = kNotRecorded;   // epoch of this run's FIRST
                                              //   integrated step
    double output_cadence_s = kNotRecorded;   // this run's output cadence
    double refgrid_H        = kNotRecorded;
    int    grid_m = -1, grid_n = -1, grid_b = -1;
    double mesh_geomhash = kNotRecorded;
    std::uint64_t config_identity_hash = 0;
    std::string body_name;
    // True when this run couples radiation, i.e. when a PRE-coupling
    // checkpoint (flag bit0 clear) would cold-start its radiation-fed faces.
    bool coupled_run = false;
};

struct RestoreVerdict {
    std::vector<std::string> refusals;    // any entry => the run must stop
    std::vector<std::string> warnings;    // loud, but the run continues
    std::vector<std::string> unverified;  // the file could not answer these
    bool ok() const { return refusals.empty(); }
};

namespace detail {

inline bool relEqual(double a, double b, double rel) {
    if (!(std::isfinite(a) && std::isfinite(b))) return false;
    const double scale = (std::fabs(a) > std::fabs(b) ? std::fabs(a)
                                                      : std::fabs(b));
    return std::fabs(a - b) <= rel * (scale > 0.0 ? scale : 1.0);
}

inline bool named(const std::vector<std::string>& allow, const char* key) {
    for (const std::string& s : allow) if (s == key) return true;
    return false;
}

}  // namespace detail

// Compare a checkpoint's trailer against the live run.
//
// `allow_mismatch` downgrades a NAMED field from a refusal to a warning. It is
// a list of field keys -- "depth_grid", "mesh", "config", "body", "grid_m",
// "grid_n", "grid_b", "refgrid_H", "epoch" -- and deliberately NOT a blanket
// force: a `--force` that turns off every guard at once is the "warn instead of
// refuse" policy wearing a hat, and warning instead of refusing is what made
// PTM-76 possible.
//
// `flag::kBatchCompletionPartial` is the one refusal `allow_mismatch` cannot
// reach. A partial state is not a whole-mesh state that happens to disagree
// about something; it is a different object.
inline void checkRestore(const Trailer* t, TrailerStatus status,
                         const Identity& run,
                         const std::vector<std::string>& allow_mismatch,
                         RestoreVerdict& v) {
    v = RestoreVerdict();

    auto fail = [&](const char* key, const std::string& msg) {
        if (detail::named(allow_mismatch, key))
            v.warnings.push_back(std::string(msg) +
                                 "  [downgraded to a warning by "
                                 "thermal.state_restore_allow_mismatch: [" +
                                 key + "]]");
        else
            v.refusals.push_back(std::string(msg) +
                                 "  [to override this ONE field: "
                                 "thermal.state_restore_allow_mismatch: [" +
                                 key + "]]");
    };

    if (status == TrailerStatus::Absent || t == nullptr) {
        v.unverified.push_back(
            "depth grid z[]      -- a restore onto a DIFFERENT grid with the "
            "same layer count is accepted in silence (PTM-STATE-GRID-BLIND)");
        v.unverified.push_back(
            "epoch               -- the epoch this column belongs to is not in "
            "the file, so a cross-window or cross-season restore cannot be "
            "detected (PTM-STATE-NO-EPOCH)");
        v.unverified.push_back(
            "body + properties   -- the physics the column was integrated with");
        v.unverified.push_back(
            "mesh identity       -- whether these faces are this mesh's faces");
        v.unverified.push_back(
            "radiation coupling  -- whether the column is pre- or "
            "post-coupling (PTM-57)");
        return;
    }

    // ---- structural: a partial is not a whole-mesh state ------------------
    if (t->flags & flag::kBatchCompletionPartial) {
        v.refusals.push_back(
            "this checkpoint is a BATCH-COMPLETION PARTIAL (trailer flag bit3): "
            "it covers only the faces one device had finished, and the rest of "
            "the column block is not a solved state. It can never be promoted "
            "to a whole-mesh checkpoint. This refusal is NOT overridable.");
    }

    // ---- the silent-corruption pair ---------------------------------------
    if (!t->z.empty() && !run.z.empty()) {
        if (t->z.size() != run.z.size()) {
            fail("depth_grid",
                 "depth grid: the checkpoint carries " +
                 std::to_string(t->z.size()) + " nodes, this run builds " +
                 std::to_string(run.z.size()));
        } else {
            std::size_t bad = t->z.size();
            for (std::size_t i = 0; i < t->z.size(); ++i) {
                if (!detail::relEqual(t->z[i], run.z[i], 1e-12)) { bad = i; break; }
            }
            if (bad < t->z.size()) {
                char b2[224];
                std::snprintf(b2, sizeof(b2),
                              "depth grid: node %zu is %.17g m in the "
                              "checkpoint and %.17g m in this run (first of "
                              "possibly several). The column would be "
                              "reinterpreted onto the wrong depths",
                              bad, t->z[bad], run.z[bad]);
                fail("depth_grid", b2);
            }
        }
    } else if (t->z.empty()) {
        v.unverified.push_back("depth grid z[]      -- absent from the trailer");
    }

    if (std::isfinite(t->mesh_geomhash) && std::isfinite(run.mesh_geomhash)) {
        if (!detail::relEqual(t->mesh_geomhash, run.mesh_geomhash, 1e-12)) {
            char b2[224];
            std::snprintf(b2, sizeof(b2),
                          "mesh: geometry hash %.17g in the checkpoint, %.17g "
                          "in this run. These faces are not those faces",
                          t->mesh_geomhash, run.mesh_geomhash);
            fail("mesh", b2);
        }
    } else {
        v.unverified.push_back("mesh identity       -- not recorded");
    }

    // ---- physics identity --------------------------------------------------
    if (!t->body_name.empty() && !run.body_name.empty() &&
        t->body_name != run.body_name) {
        fail("body", "body: the checkpoint was integrated on '" + t->body_name +
                     "', this run is '" + run.body_name + "'");
    }
    if (t->config_identity_hash != 0 && run.config_identity_hash != 0) {
        if (t->config_identity_hash != run.config_identity_hash) {
            char b2[224];
            std::snprintf(b2, sizeof(b2),
                          "physics: config identity hash 0x%016llX in the "
                          "checkpoint, 0x%016llX in this run. See "
                          "thermal/state_identity.h for exactly which "
                          "quantities that covers -- and which it does not",
                          (unsigned long long)t->config_identity_hash,
                          (unsigned long long)run.config_identity_hash);
            fail("config", b2);
        }
    } else {
        v.unverified.push_back(
            "thermophysics       -- config identity hash not recorded");
    }

    // Named individually even though z[] already implies them: a report that
    // says "grid_b was 15, this run is 32" is actionable and "node 7 differs"
    // is not.
    if (t->grid_m >= 0 && run.grid_m >= 0 && t->grid_m != run.grid_m)
        fail("grid_m", "thermal.grid_m: " + std::to_string(t->grid_m) +
                       " in the checkpoint, " + std::to_string(run.grid_m) +
                       " in this run");
    if (t->grid_n >= 0 && run.grid_n >= 0 && t->grid_n != run.grid_n)
        fail("grid_n", "thermal.grid_n: " + std::to_string(t->grid_n) +
                       " in the checkpoint, " + std::to_string(run.grid_n) +
                       " in this run");
    if (t->grid_b >= 0 && run.grid_b >= 0 && t->grid_b != run.grid_b)
        fail("grid_b", "thermal.grid_b: " + std::to_string(t->grid_b) +
                       " in the checkpoint, " + std::to_string(run.grid_b) +
                       " in this run");
    if (std::isfinite(t->refgrid_H) && std::isfinite(run.refgrid_H) &&
        !detail::relEqual(t->refgrid_H, run.refgrid_H, 1e-12)) {
        char b2[192];
        std::snprintf(b2, sizeof(b2),
                      "refgrid_H: %.17g m in the checkpoint, %.17g m in this "
                      "run (the median H every solver path builds its shared "
                      "grid on)", t->refgrid_H, run.refgrid_H);
        fail("refgrid_H", b2);
    }

    // ---- the epoch clause --------------------------------------------------
    // This is the clause that separates a chunk HAND-OFF from a warm start.
    // The gap is SIGNED, and the sign changes what happened:
    //
    //  * within a tolerance             -- the states TILE; this is the chunked
    //                                      workflow and it passes silently.
    //  * BEHIND by <= one output cadence -- a one-step-stale hand-off. That is
    //                                      what a chunk chain looks like when
    //                                      thermal.checkpoint_lookahead_steps
    //                                      is 0. WARN with the number.
    //  * BEHIND by more                 -- the state never existed at this
    //                                      run's epoch. REFUSE.
    //  * AHEAD                          -- an OVERLAP, not staleness. REFUSE,
    //                                      with its own message: the advice for
    //                                      a stale hand-off makes this worse.
    //
    // The measured cost of getting this wrong is not hypothetical. A
    // cross-season warm start, measured on one lunar polar site, moved
    // `temperature_min` by 5.75 K rms and 49.9 K at worst, and shifted the
    // ice-stability depth product by 19.3 percentage points -- and NOTHING in
    // the QA battery detected it.
    if (std::isfinite(t->et_seconds) && std::isfinite(run.et_first)) {
        const double d = run.et_first - t->et_seconds;
        const double ad = std::fabs(d);
        const double cadence = std::isfinite(run.output_cadence_s)
                                   ? std::fabs(run.output_cadence_s)
                                   : 0.0;
        // Well below any cadence this engine runs, well above the ~1e-7 s
        // resolution of an ephemeris time near J2000 + 1e9 s.
        const double tol = 1e-3;
        if (ad > tol) {
            char b2[512];
            if (d < 0.0) {
                // The checkpoint is AHEAD of this run's first epoch. That is
                // not a stale hand-off, it is an OVERLAP: the run would
                // re-integrate time the banked column has already been through,
                // double-counting that span in the merged statistics. The sign
                // matters, so it gets its own message -- telling this operator
                // to "set checkpoint_lookahead_steps" would make it worse.
                std::snprintf(b2, sizeof(b2),
                              "epoch: the checkpoint is at ET %.6f, which is "
                              "%.6f s (%.4f days) AFTER this run's first epoch "
                              "ET %.6f. The run would re-integrate a span the "
                              "banked column has already passed, so the two "
                              "windows OVERLAP rather than tile. This is a "
                              "chunk index or a start_time_offset_seconds that "
                              "was not advanced, not a look-ahead problem.",
                              t->et_seconds, -d, -d / 86400.0, run.et_first);
                fail("epoch", b2);
            } else if (cadence > 0.0 && ad <= cadence * (1.0 + 1e-9)) {
                std::snprintf(b2, sizeof(b2),
                              "epoch: the checkpoint is at ET %.6f and this run "
                              "starts at ET %.6f -- %.6f s apart, which is one "
                              "output cadence (%.6f s). That is a hand-off "
                              "written WITHOUT thermal.checkpoint_lookahead_"
                              "steps: 1, so the restored column is one step "
                              "stale. The run continues; set the look-ahead in "
                              "the writing chunk to remove the seam.",
                              t->et_seconds, run.et_first, d, cadence);
                v.warnings.push_back(b2);
            } else {
                std::snprintf(b2, sizeof(b2),
                              "epoch: the checkpoint is at ET %.6f and this run "
                              "starts at ET %.6f -- %.6f s apart (%.4f days). "
                              "The banked column is a state that never existed "
                              "at this run's epoch, so restoring it reports an "
                              "unintegrated gap as physics. Chaining is "
                              "admissible only where the states TILE the "
                              "timeline (scripts/run_chunked_simulation.py); "
                              "across windows or seasons it is not.",
                              t->et_seconds, run.et_first, d, d / 86400.0);
                fail("epoch", b2);
            }
        }
    } else {
        v.unverified.push_back(
            "epoch               -- not recorded by one side; a cross-window "
            "restore cannot be detected");
    }

    // ---- coupling provenance ----------------------------------------------
    if (t->flags & flag::kCouplingSelectDtPositive) {
        v.warnings.push_back(
            "the writing run had a positive radiation coupling_select_dt, so "
            "its re-thermal selector could leave a face with nonzero indirect "
            "flux frozen at its iter-0 column (BUGS.md PTM-89). Treat this "
            "checkpoint as provisional.");
    }
    if (run.coupled_run && !(t->flags & flag::kPostCouplingComplete)) {
        v.warnings.push_back(
            "this checkpoint is the PRE-coupling (iter-0) column and this run "
            "couples radiation: the faces radiation feeds resume from an "
            "un-coupled state. That is BUGS.md PTM-57.");
    } else if ((t->flags & flag::kPostCouplingComplete) &&
               !(t->flags & flag::kAllFacesPatched)) {
        v.warnings.push_back(
            "this checkpoint is post-coupling, but only the re-thermal'd "
            "selection carries the coupled column; the rest keep iter-0's. "
            "That is the intended behavior at default settings -- recorded "
            "here so it is never inferred.");
    }
}

}  // namespace state_io
}  // namespace thermal

#endif  // PTM_THERMAL_STATE_IO_H
