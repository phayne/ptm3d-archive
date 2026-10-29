#ifndef PTM_FILEIO_FRAME_NAMING_H
#define PTM_FILEIO_FRAME_NAMING_H

// Per-step frame file names: <dir>/<prefix>_<index>.bin.
//
// The index used to be zero-padded to FOUR digits. std::setw is a MINIMUM
// width, so frame 10000 was written "10000" and an alphabetical sort of a run
// with more than 10,000 frames came out 0000, 0001, ..., 1000, 10000, 10001,
// ..., 1001 -- which scrambled a Justitia TWG Case 4 reduction (161,693
// frames) until the sun direction fitted from the absorbed flux exposed it.
//
// The WRITER now pads to frameIndexWidth(n_frames): at least six digits, and
// more if the run has more frames than six digits can hold, so every frame of
// one run has the same width and a lexicographic sort IS the numeric sort.
//
// READERS must still sort NUMERICALLY (parse the integer after the last '_'),
// never lexicographically: runs written before this change have four-digit
// names that roll over to five at 10,000, and a directory that mixes widths
// would defeat any width-based rule. resolveFrameFile() below is the C++
// reader-side helper for code that looks up frame k by index.

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace ptm {

// Minimum index width for newly written frames.
constexpr int kFrameIndexMinWidth = 6;
// Width every run before the frame-name fix wrote (a minimum, not a maximum).
constexpr int kFrameIndexLegacyWidth = 4;

// Number of digits the writer pads to for a run exporting `n_frames` frames
// (indices 0 .. n_frames-1): max(kFrameIndexMinWidth, digits(n_frames-1)).
inline int frameIndexWidth(long long n_frames) {
    long long last = (n_frames > 0) ? n_frames - 1 : 0;
    int digits = 1;
    while (last >= 10) { last /= 10; ++digits; }
    return digits > kFrameIndexMinWidth ? digits : kFrameIndexMinWidth;
}

// Zero-padded index string for frame `idx` of a run exporting `n_frames`.
inline std::string frameIndexString(long long idx, long long n_frames) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%0*lld", frameIndexWidth(n_frames), idx);
    return std::string(buf);
}

// Existing file for frame `idx` in `dir` named `<prefix>_<index>.bin`, trying
// every width a PTM3D writer has used: the current rule first (six digits and
// up), then the legacy four-digit names. Returns "" when none exists.
inline std::string resolveFrameFile(const std::string& dir,
                                    const std::string& prefix,
                                    long long idx) {
    auto exists = [](const std::string& p) {
        struct stat st;
        return ::stat(p.c_str(), &st) == 0;
    };
    char buf[32];
    // Widths 6..12 cover every n_frames a 64-bit index can reach in practice;
    // once the index has at least `w` digits all remaining widths give the
    // same string, so the loop stops there.
    for (int w = kFrameIndexMinWidth; w <= 12; ++w) {
        std::snprintf(buf, sizeof buf, "%0*lld", w, idx);
        const std::string p = dir + "/" + prefix + "_" + buf + ".bin";
        if (exists(p)) return p;
        if (static_cast<int>(std::string(buf).size()) > w) break;
    }
    std::snprintf(buf, sizeof buf, "%0*lld", kFrameIndexLegacyWidth, idx);
    const std::string legacy = dir + "/" + prefix + "_" + buf + ".bin";
    if (exists(legacy)) return legacy;
    return "";
}

// Remove every `<prefix>_<digits>.bin` in `dir` whose index is NOT `width`
// digits wide -- i.e. frames a PREVIOUS run (written under the other naming)
// left in a directory this run is about to fill. Returns the number removed
// (0 when the directory does not exist yet).
//
// Why: before the frame-name fix a re-run into the same output directory wrote
// the same names and overwrote the old frames. With the new width it would
// leave the old 4-digit set beside the new one, and every reader that lists
// frames numerically would then load both (duplicate indices) or refuse the
// directory. Only OTHER-width files are touched: frames of this width are the
// ones this run writes, so a second export pass in the same run is untouched.
// The caller reports the count; a nonzero count must never be silent.
inline long long removeOtherWidthFrames(const std::string& dir,
                                        const std::string& prefix, int width) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return 0;
    long long removed = 0;
    const std::string head = prefix + "_";
    while (dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() <= head.size() + 4 || name.compare(0, head.size(), head) != 0 ||
            name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        const std::string digits = name.substr(head.size(), name.size() - head.size() - 4);
        bool all = !digits.empty();
        for (char c : digits) if (c < '0' || c > '9') { all = false; break; }
        if (!all || static_cast<int>(digits.size()) == width) continue;
        if (::unlink((dir + "/" + name).c_str()) == 0) ++removed;
    }
    ::closedir(d);
    return removed;
}

}  // namespace ptm

#endif  // PTM_FILEIO_FRAME_NAMING_H
