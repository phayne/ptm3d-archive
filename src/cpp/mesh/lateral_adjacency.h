// lateral_adjacency.h -- edge adjacency and two-point-flux geometry for lateral heat
// conduction between facet columns (C1c, Moon3D R1; formulation signed off by Fermi
// 2026-10-18, LATERAL_SIGNOFF_FERMI.md).
//
// Built once per mesh from raw vertex/face arrays, so it has no dependency on the Mesh
// class and is unit-testable on its own. For each face i and each of its 3 edges e
// (edge e joins local vertices e and (e+1)%3):
//
//   nbr[3i+e]      the face sharing edge e, or -1 (adiabatic: open boundary, excluded
//                  neighbor, or non-manifold edge)
//   nbr_slot[3i+e] which edge slot of nbr[3i+e] is the same edge (so the pair is stored
//                  symmetrically and G_ij is built from identical inputs on both sides)
//   edge_len[3i+e] l_ij, the shared-edge length
//   d_half[3i+e]   d_i = |c_i - m_e|, centroid of face i to the edge midpoint (Fermi R1:
//                  the TPFA path runs through the midpoint, d_ij = d_i + d_j, and the
//                  conductance is the series form l h / (d_i/k_i + d_j/k_j))
//
// Make-silence-impossible rules: open-boundary, non-manifold, excluded-neighbor and
// degenerate edges are COUNTED, and the counts are part of the result for the run log.
// A non-manifold edge (shared by more than two included faces) couples nobody: every
// face on it sees -1. Choosing two of the faces arbitrarily, as the illumination self-skip
// table does (illumination_model.cpp:1263), would make the network depend on face order.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ptm {
namespace lateral {

struct Adjacency {
    int64_t n_faces = 0;
    std::vector<int32_t> nbr;       // [3*n_faces]
    std::vector<int8_t>  nbr_slot;  // [3*n_faces], -1 where nbr == -1
    std::vector<double>  edge_len;  // [3*n_faces]
    std::vector<double>  d_half;    // [3*n_faces]
    std::vector<double>  area;      // [n_faces]
    std::vector<double>  mean_edge; // [n_faces], facet size l_i for the D* activation rule
    // Counts, by edge (not by face slot).
    int64_t n_interior_edges = 0;      // coupled pairs
    int64_t n_boundary_edges = 0;      // one included face, no partner: adiabatic
    int64_t n_excluded_edges = 0;      // partner face exists but is excluded (ring/fill): adiabatic
    int64_t n_nonmanifold_edges = 0;   // > 2 included faces: adiabatic for all of them
    int64_t n_degenerate_faces = 0;    // zero-area or zero-length-edge faces (included, never coupled)

    std::string summary() const;       // one line for the run log
};

// vertices: [3*n_vertices] xyz (m). faces: [3*n_faces] vertex indices.
// included: per face, 1 = a thermal column that takes part in lateral conduction,
// 0 = excluded (ring/occluder faces, hole-seal fill triangles). Empty = all included.
// Excluded faces get all -1 slots and are never coupled.
Adjacency build(const std::vector<double>& vertices,
                const std::vector<int64_t>& faces,
                const std::vector<uint8_t>& included);

}  // namespace lateral
}  // namespace ptm
