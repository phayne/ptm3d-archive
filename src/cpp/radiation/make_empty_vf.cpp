/**
 * @file make_empty_vf.cpp
 * @brief Emit a correctly-dimensioned, current-format (HMAT/v2) EMPTY view-factor
 *        file (nnz=0) for a mesh, via HMatrixViewFactor::writeBinary.
 *
 * Purpose (AGT rolling-VF campaign): ptm_thermal_sim rebuilds the (multi-GB)
 * H-matrix ONLY when the configured view_factors.hmat is MISSING or fails to
 * load; if it exists AND loads, it is used and then overridden at run time by
 * PTM_HMATRIX_VF_REPLACE (the corrected per-partner CSR). A stale-format base
 * (e.g. the old MFFV/v1 lunar polar site D file) FAILS to load and silently triggers the
 * full recompute -- the exact wedge we avoid (the stale-format fallback). This tool
 * writes a base that the CURRENT binary loads clean with nnz=0, using the
 * production serializer so the format can never drift from the loader.
 *
 * It reads only the face count from the PLY header (the same count the sim's
 * mesh loader reports for a triangulated PLY) and round-trips the result
 * through readBinary as a fail-loud self-check.
 *
 * Meshes the sim loads but this tool cannot parse -- legacy VTK 5.1 BINARY, for
 * one, which is what the 509,337-face moon3d flagship ships as -- can supply the
 * count directly with --faces N. The count is then the CALLER's claim, so it is
 * echoed and the loader's own face-count/geomhash gate remains the thing that
 * catches a wrong one (a base .vf whose dimensions disagree with the mesh fails
 * to load, which is the recompute wedge this tool exists to prevent). Prefer the
 * PLY form whenever the mesh is a PLY.
 *
 * Usage: make_empty_vf <mesh.ply> <out.vf>
 *        make_empty_vf --faces <N> <out.vf>
 */
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "hmatrix_view_factor.h"

// Read the 'element face N' count from a binary/ascii PLY header. Fail loud on
// anything it can't prove -- never guess a face count for a base .vf.
static size_t ply_face_count(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "FATAL: cannot open mesh PLY '" << path << "'\n"; std::exit(3); }
    std::string line, magic;
    std::getline(f, magic);
    if (magic.rfind("ply", 0) != 0) { std::cerr << "FATAL: not a PLY file: '" << path << "'\n"; std::exit(3); }
    size_t nf = 0; bool have = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream is(line);
        std::string t; is >> t;
        if (t == "element") {
            std::string w; size_t n; is >> w >> n;
            if (w == "face") { nf = n; have = true; }
        } else if (t == "end_header") {
            break;
        }
    }
    if (!have || nf == 0) {
        std::cerr << "FATAL: no 'element face N>0' in PLY header of '" << path << "'\n";
        std::exit(3);
    }
    return nf;
}

int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::string(argv[1]) == "--faces")) {
        std::cerr << "usage: make_empty_vf <mesh.ply> <out.vf>\n"
                     "       make_empty_vf --faces <N> <out.vf>   (for meshes this "
                     "tool cannot parse, e.g. legacy VTK 5.1 BINARY)\n";
        return 2;
    }

    size_t nf = 0;
    std::string out;
    if (argc == 4) {
        const std::string nstr = argv[2];
        out = argv[3];
        char* end = nullptr;
        const unsigned long long v = std::strtoull(nstr.c_str(), &end, 10);
        if (!end || *end != '\0' || v == 0) {
            std::cerr << "FATAL: --faces needs a positive integer, got '" << nstr << "'\n";
            return 2;
        }
        nf = static_cast<size_t>(v);
        std::cout << "faces " << nf << " (supplied on the command line, NOT read from a "
                     "mesh -- the engine's own face-count gate is what will catch it if "
                     "this disagrees with the mesh)\n";
    } else {
        const std::string mesh = argv[1];
        out = argv[2];
        nf = ply_face_count(mesh);
        std::cout << "mesh " << mesh << ": " << nf << " faces\n";
    }

    radiation::HMatrixViewFactor H;
    H.setEmptyDimensions(nf);
    if (!H.writeBinary(out)) {
        std::cerr << "FATAL: writeBinary failed for '" << out << "'\n";
        return 1;
    }

    // Fail-loud self-check: the production reader must load it back with the
    // right dimensions and zero non-zeros.
    radiation::HMatrixViewFactor H2;
    if (!H2.readBinary(out)) {
        std::cerr << "FATAL: round-trip readBinary FAILED for '" << out
                  << "' -- the written base would trigger a silent VF recompute.\n";
        return 1;
    }
    if (H2.numRows() != nf || H2.numCols() != nf) {
        std::cerr << "FATAL: round-trip dimension mismatch (" << H2.numRows() << "x"
                  << H2.numCols() << " != " << nf << ")\n";
        return 1;
    }
    std::cout << "OK: wrote empty HMAT/v2 base '" << out << "' (" << nf
              << " faces, nnz=0); round-trip readBinary verified.\n";
    return 0;
}
