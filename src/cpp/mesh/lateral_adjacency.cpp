// lateral_adjacency.cpp -- see lateral_adjacency.h.
#include "lateral_adjacency.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ptm {
namespace lateral {

namespace {
struct V3 { double x, y, z; };
inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline double norm(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline uint64_t edgeKey(int64_t a, int64_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}
}  // namespace

std::string Adjacency::summary() const {
    std::ostringstream s;
    s << "[lateral] adjacency: " << n_faces << " faces, " << n_interior_edges
      << " coupled edges; adiabatic: " << n_boundary_edges << " open-boundary, "
      << n_excluded_edges << " to excluded faces, " << n_nonmanifold_edges
      << " non-manifold; " << n_degenerate_faces << " degenerate face(s) never coupled";
    return s.str();
}

Adjacency build(const std::vector<double>& vertices,
                const std::vector<int64_t>& faces,
                const std::vector<uint8_t>& included) {
    if (faces.size() % 3 != 0 || vertices.size() % 3 != 0)
        throw std::invalid_argument("lateral::build: vertex/face arrays are not multiples of 3");
    const int64_t nf = static_cast<int64_t>(faces.size() / 3);
    const int64_t nv = static_cast<int64_t>(vertices.size() / 3);
    if (!included.empty() && static_cast<int64_t>(included.size()) != nf)
        throw std::invalid_argument("lateral::build: 'included' has the wrong length");
    if (nv >= (int64_t(1) << 32))
        throw std::invalid_argument("lateral::build: more than 2^32 vertices");
    auto inc = [&](int64_t f) { return included.empty() || included[f] != 0; };
    auto vtx = [&](int64_t v) { return V3{vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]}; };

    Adjacency A;
    A.n_faces = nf;
    A.nbr.assign(3 * nf, -1);
    A.nbr_slot.assign(3 * nf, -1);
    A.edge_len.assign(3 * nf, 0.0);
    A.d_half.assign(3 * nf, 0.0);
    A.area.assign(nf, 0.0);
    A.mean_edge.assign(nf, 0.0);

    std::vector<uint8_t> degenerate(nf, 0);
    for (int64_t f = 0; f < nf; ++f) {
        const V3 p0 = vtx(faces[3 * f]), p1 = vtx(faces[3 * f + 1]), p2 = vtx(faces[3 * f + 2]);
        const V3 c{(p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0, (p0.z + p1.z + p2.z) / 3.0};
        const V3 p[3] = {p0, p1, p2};
        A.area[f] = 0.5 * norm(cross(sub(p1, p0), sub(p2, p0)));
        double esum = 0.0;
        bool zero_edge = false;
        for (int e = 0; e < 3; ++e) {
            const V3 a = p[e], b = p[(e + 1) % 3];
            const double L = norm(sub(b, a));
            const V3 m{0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)};
            A.edge_len[3 * f + e] = L;
            A.d_half[3 * f + e] = norm(sub(m, c));
            esum += L;
            zero_edge = zero_edge || !(L > 0.0);
        }
        A.mean_edge[f] = esum / 3.0;
        if (!(A.area[f] > 0.0) || zero_edge) degenerate[f] = 1;
    }

    // edge -> list of (face, slot); collect over ALL faces so excluded neighbors are seen.
    std::unordered_map<uint64_t, std::vector<std::pair<int64_t, int>>> emap;
    emap.reserve(static_cast<size_t>(nf) * 3 / 2 + 16);
    for (int64_t f = 0; f < nf; ++f)
        for (int e = 0; e < 3; ++e)
            emap[edgeKey(faces[3 * f + e], faces[3 * f + (e + 1) % 3])].push_back({f, e});

    for (const auto& kv : emap) {
        const auto& lst = kv.second;
        std::vector<std::pair<int64_t, int>> in;
        bool has_excluded = false;
        for (const auto& fs : lst) {
            if (inc(fs.first)) in.push_back(fs);
            else has_excluded = true;
        }
        if (in.empty()) continue;
        if (in.size() > 2) { ++A.n_nonmanifold_edges; continue; }
        if (in.size() == 1) {
            if (has_excluded) ++A.n_excluded_edges; else ++A.n_boundary_edges;
            continue;
        }
        const int64_t i = in[0].first, j = in[1].first;
        const int si = in[0].second, sj = in[1].second;
        if (i == j || degenerate[i] || degenerate[j]) {  // a face meeting itself, or a degenerate face
            ++A.n_boundary_edges;
            continue;
        }
        A.nbr[3 * i + si] = static_cast<int32_t>(j);
        A.nbr[3 * j + sj] = static_cast<int32_t>(i);
        A.nbr_slot[3 * i + si] = static_cast<int8_t>(sj);
        A.nbr_slot[3 * j + sj] = static_cast<int8_t>(si);
        // the shared edge has one length; take it from the lower face index on both sides so
        // the pair's inputs are bit-identical (they are equal up to vertex order anyway)
        const double L = (i < j) ? A.edge_len[3 * i + si] : A.edge_len[3 * j + sj];
        A.edge_len[3 * i + si] = L;
        A.edge_len[3 * j + sj] = L;
        ++A.n_interior_edges;
    }
    for (int64_t f = 0; f < nf; ++f) A.n_degenerate_faces += degenerate[f];
    return A;
}

}  // namespace lateral
}  // namespace ptm
