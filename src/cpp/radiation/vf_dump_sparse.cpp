/**
 * @file vf_dump_sparse.cpp
 * @brief Dump the PRODUCTION in-engine sparse view-factor matrix to a CSR file.
 *
 * WHY THIS EXISTS
 * ---------------
 * PTM3D computes view factors three ways -- the in-engine sparse builder
 * (view_factor.cpp), the in-engine H-matrix builder, and an OFFLINE Monte-Carlo
 * referee pipeline (scripts/ptm12_diag/mc_ref_pp.cpp + assemble_vf_haworth.py)
 * -- and until now nothing compared their answers. Each was gated against
 * closed forms in isolation, which is necessary and not sufficient: FC-08 was a
 * factor-of-two inflation in the offline assembler that every in-isolation
 * invariant passed, and the 2026-08-23 near-field bias was a -21.7 percent
 * error on touching pairs in the in-engine builder that the offline pipeline
 * would have contradicted immediately, had anyone asked it.
 *
 * Asking it needs both matrices in one format. The offline pipeline already
 * writes CSR; this writes the same CSR from the in-engine sparse builder, so
 * scripts/ptm12_diag/test_vf_cross_agreement.py can difference them entry by
 * entry on one synthetic mesh.
 *
 * THE FILE FORMAT is assemble_vf_haworth.py's, exactly, so one reader serves
 * both sides:
 *     int32    N
 *     float64  geomhash = sum(A) + sum(A_i * (i+1)) * 1e-9
 *     int64    indptr[N + 1]
 *     int64    nnz
 *     int32    cols[nnz]
 *     float32  data[nnz]           <- F_ij, the view factor, NOT A_i F_ij
 * Row i holds the partners face i RECEIVES from, matching both the assembler
 * and ViewFactorMatrix.
 *
 * THE MESH READER is deliberately local and deliberately strict: binary
 * little-endian PLY, float64 vertices, uchar + 3 x int32 faces -- the exact
 * layout the gate writes and mc_ref_pp reads. It does NOT use MeshUtils,
 * because MeshUtils::importTriangulatedMesh auto-orients normals outward from
 * the mesh centroid, and the gate's geometry (a small element under a disk that
 * faces DOWN at it) is not star-shaped. Silently flipping a normal would make
 * the two sides disagree for a reason that has nothing to do with the kernel.
 * Normals come from the winding, cross(v1 - v0, v2 - v0), which is what the
 * assembler and mc_ref_pp both use.
 *
 * NEIGHBOR TRUNCATION IS OFF BY DEFAULT here, and that is a deliberate
 * difference from a production run: the shipped adaptive selector targets
 * captured_energy_target = 0.95 and so reads ~0.949 of a closed form BY DESIGN
 * (gate arm [D2]). Comparing a 95-percent-captured matrix against a full
 * offline one would measure the selector, not the estimator. Pass
 * --adaptive-neighbors to score the production selector instead.
 *
 * Usage:
 *   vf_dump_sparse <mesh.ply> <out.csr> [--quad-order M]
 *                  [--visibility-threshold X] [--window-factor X]
 *                  [--max-neighbors N] [--adaptive-neighbors] [--quiet]
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <embree4/rtcore.h>

#include "../mesh/mesh_utils.h"
#include "radiation_config.h"
#include "view_factor.h"

namespace {

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "FATAL: " << msg << "\n";
    std::exit(1);
}

// Strict binary-little-endian PLY reader; see the header comment for why it is
// local. Anything it does not understand is fatal, never guessed at.
std::shared_ptr<ptm::Mesh> readBinaryPly(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);

    std::string line;
    if (!std::getline(f, line) || line.substr(0, 3) != "ply")
        fail(path + " is not a PLY file");

    size_t nv = 0, nf = 0;
    bool little = false, have_format = false;
    int vprops = 0;
    bool vprops_double = true;
    std::string elem;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line == "end_header") break;
        if (line.rfind("format", 0) == 0) {
            have_format = true;
            little = line.find("binary_little_endian") != std::string::npos;
            if (!little)
                fail(path + ": only binary_little_endian is supported; convert it first");
        } else if (line.rfind("element vertex", 0) == 0) {
            nv = std::strtoul(line.c_str() + 14, nullptr, 10);
            elem = "v";
        } else if (line.rfind("element face", 0) == 0) {
            nf = std::strtoul(line.c_str() + 12, nullptr, 10);
            elem = "f";
        } else if (line.rfind("property", 0) == 0 && elem == "v") {
            ++vprops;
            if (line.find("double") == std::string::npos) vprops_double = false;
        }
    }
    if (!have_format) fail(path + ": no format line");
    if (nv == 0 || nf == 0) fail(path + ": zero vertices or faces");
    if (vprops != 3 || !vprops_double)
        fail(path + ": expected exactly 3 double vertex properties (x, y, z), got " +
             std::to_string(vprops) + (vprops_double ? " double" : " with a non-double"));

    auto mesh = std::make_shared<ptm::Mesh>();
    mesh->vertices.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
        double xyz[3];
        f.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
        if (!f) fail(path + ": truncated vertex block");
        mesh->vertices[i] = {{xyz[0], xyz[1], xyz[2]}};
    }
    mesh->faces.resize(nf);
    for (size_t i = 0; i < nf; ++i) {
        unsigned char c = 0;
        int32_t idx[3];
        f.read(reinterpret_cast<char*>(&c), 1);
        f.read(reinterpret_cast<char*>(idx), sizeof(idx));
        if (!f) fail(path + ": truncated face block");
        if (c != 3) fail(path + ": face " + std::to_string(i) + " has " +
                         std::to_string(int(c)) + " vertices; must be triangulated");
        mesh->faces[i] = {{idx[0], idx[1], idx[2]}};
    }
    return mesh;
}

// Winding normals and centroids. NOT auto-oriented -- see the header comment.
std::vector<double> computeFaceData(ptm::Mesh& m) {
    const size_t nf = m.faces.size();
    m.face_centers.resize(nf);
    m.face_normals.resize(nf);
    std::vector<double> areas(nf);
    for (size_t i = 0; i < nf; ++i) {
        const auto& v0 = m.vertices[m.faces[i][0]];
        const auto& v1 = m.vertices[m.faces[i][1]];
        const auto& v2 = m.vertices[m.faces[i][2]];
        m.face_centers[i] = {{(v0[0] + v1[0] + v2[0]) / 3.0,
                              (v0[1] + v1[1] + v2[1]) / 3.0,
                              (v0[2] + v1[2] + v2[2]) / 3.0}};
        const double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        const double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        double n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                       e1[2] * e2[0] - e1[0] * e2[2],
                       e1[0] * e2[1] - e1[1] * e2[0]};
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(len > 0.0)) fail("degenerate face " + std::to_string(i) + " (zero area)");
        areas[i] = 0.5 * len;
        m.face_normals[i] = {{n[0] / len, n[1] / len, n[2] / len}};
    }
    return areas;
}

RTCScene makeScene(RTCDevice dev, const ptm::Mesh& m) {
    RTCScene scene = rtcNewScene(dev);
    RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    const size_t nv = m.vertices.size(), nf = m.faces.size();
    float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), nv));
    for (size_t i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k) vb[3 * i + k] = static_cast<float>(m.vertices[i][k]);
    unsigned* ib = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), nf));
    for (size_t i = 0; i < nf; ++i)
        for (int k = 0; k < 3; ++k) ib[3 * i + k] = static_cast<unsigned>(m.faces[i][k]);
    rtcCommitGeometry(g);
    rtcAttachGeometry(scene, g);
    rtcReleaseGeometry(g);
    rtcCommitScene(scene);
    return scene;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mesh_path, out_path;
    int quad_order = -1;
    int max_neighbors = -1;
    bool adaptive = false;
    bool quiet = false;
    double vis_threshold = -1.0, window_factor = -1.0;

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) fail(std::string(what) + " expects a value");
            return argv[++i];
        };
        if (a == "--quad-order") quad_order = std::stoi(need("--quad-order"));
        else if (a == "--visibility-threshold") vis_threshold = std::stod(need("--visibility-threshold"));
        else if (a == "--window-factor") window_factor = std::stod(need("--window-factor"));
        // --ns-threshold selected the estimator through a DIFFERENT quantity
        // (Ns = r / (A_i A_j)^(1/4)). Silently accepting the old number under
        // the new rule would change the physics without saying so.
        else if (a == "--ns-threshold")
            fail("--ns-threshold was removed with adaptive_ns_threshold (2026-08-23). "
                 "The near/far switch is now r < w (sqrt(A_i) + sqrt(A_j)); use "
                 "--window-factor w. They are NOT the same number: the old default 5 "
                 "corresponds to w = 2.5 on an equal-area pair only.");
        else if (a == "--max-neighbors") max_neighbors = std::stoi(need("--max-neighbors"));
        else if (a == "--adaptive-neighbors") adaptive = true;
        else if (a == "--quiet") quiet = true;
        else if (a.rfind("--", 0) == 0) fail("unknown option " + a);
        else pos.push_back(a);
    }
    if (pos.size() != 2) {
        std::cerr << "usage: vf_dump_sparse <mesh.ply> <out.csr> [--quad-order M]\n"
                     "       [--visibility-threshold X] [--window-factor X]\n"
                     "       [--max-neighbors N] [--adaptive-neighbors] [--quiet]\n";
        return 2;
    }
    mesh_path = pos[0];
    out_path = pos[1];

    auto mesh = readBinaryPly(mesh_path);
    const std::vector<double> areas = computeFaceData(*mesh);
    const size_t n = mesh->faces.size();

    radiation::RadiationConfig cfg;                 // production defaults, then:
    cfg.adaptive_neighbors = adaptive;
    cfg.max_neighbors = (max_neighbors > 0) ? max_neighbors : static_cast<int>(n);
    cfg.search_radius_factor = 1000.0;              // the extent-based radius already
                                                    // covers the mesh; belt and braces
    cfg.verbose = false;
    if (quad_order > 0) {
        if (quad_order > radiation::kMaxConfigNearFieldQuadOrder)
            fail("--quad-order must be 1.." +
                 std::to_string(radiation::kMaxConfigNearFieldQuadOrder));
        cfg.nearfield_quadrature_order = quad_order;
    }
    if (vis_threshold >= 0.0) cfg.visibility_threshold = vis_threshold;
    if (window_factor >= 0.0) cfg.nearfield_window_factor = window_factor;

    if (!quiet) {
        std::cout << "[vf_dump_sparse] " << mesh_path << ": " << n << " faces, "
                  << mesh->vertices.size() << " vertices\n"
                  << "[vf_dump_sparse] nearfield_quadrature_order="
                  << cfg.nearfield_quadrature_order
                  << "  nearfield_window_factor=" << cfg.nearfield_window_factor
                  << "  visibility_threshold=" << cfg.visibility_threshold
                  << "  adaptive_neighbors=" << (cfg.adaptive_neighbors ? "true" : "false")
                  << "  max_neighbors=" << cfg.max_neighbors << "\n";
    }

    RTCDevice dev = rtcNewDevice(nullptr);
    RTCScene scene = makeScene(dev, *mesh);
    radiation::ViewFactorMatrix F;
    {
        radiation::ViewFactorCalculator calc(mesh, dev, scene, cfg);
        F = calc.computeAll();
        calc.reportNearFieldBoundClamps();
    }
    rtcReleaseScene(scene);
    rtcReleaseDevice(dev);

    if (F.numRows() != n)
        fail("calculator returned " + std::to_string(F.numRows()) + " rows for " +
             std::to_string(n) + " faces");

    // Flatten to the assembler's CSR layout.
    std::vector<int64_t> indptr(n + 1, 0);
    std::vector<int32_t> out_cols;
    std::vector<float> out_data;
    std::vector<int> cols;
    std::vector<double> vals;
    double max_rowsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        F.getRow(static_cast<int>(i), cols, vals);
        // Ascending column order, so the file is directly comparable to the
        // assembler's (which sorts by row then leaves columns as they fall) and
        // so a bisecting reader can be used on either.
        std::vector<size_t> order(cols.size());
        for (size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return cols[a] < cols[b]; });
        double rs = 0.0;
        for (size_t k : order) {
            out_cols.push_back(cols[k]);
            out_data.push_back(static_cast<float>(vals[k]));
            rs += vals[k];
        }
        max_rowsum = std::max(max_rowsum, rs);
        indptr[i + 1] = static_cast<int64_t>(out_cols.size());
    }

    // Same geomhash formula as assemble_vf_haworth.py, so a mismatch between the
    // two sides is visible as a mesh-identity failure rather than as a silent
    // comparison of two different geometries.
    double area_sum = 0.0, weighted = 0.0;
    for (size_t i = 0; i < n; ++i) {
        area_sum += areas[i];
        weighted += areas[i] * static_cast<double>(i + 1);
    }
    const double geomhash = area_sum + weighted * 1e-9;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) fail("cannot write " + out_path);
    const int32_t n32 = static_cast<int32_t>(n);
    const int64_t nnz = static_cast<int64_t>(out_cols.size());
    out.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
    out.write(reinterpret_cast<const char*>(&geomhash), sizeof(geomhash));
    out.write(reinterpret_cast<const char*>(indptr.data()),
              static_cast<std::streamsize>(indptr.size() * sizeof(int64_t)));
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    out.write(reinterpret_cast<const char*>(out_cols.data()),
              static_cast<std::streamsize>(out_cols.size() * sizeof(int32_t)));
    out.write(reinterpret_cast<const char*>(out_data.data()),
              static_cast<std::streamsize>(out_data.size() * sizeof(float)));
    out.close();
    if (!out) fail("write failed on " + out_path);

    if (!quiet) {
        std::cout << "[vf_dump_sparse] wrote " << out_path << "  nnz=" << nnz
                  << "  max row sum=" << max_rowsum << "  geomhash=" << geomhash << "\n";
    }
    return 0;
}
