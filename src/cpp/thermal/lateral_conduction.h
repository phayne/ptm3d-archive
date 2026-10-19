// lateral_conduction.h -- the lateral (in-layer) conduction operator between facet
// columns (C1c, Moon3D R1). Formulation: LATERAL_SIGNOFF_FERMI.md (2026-10-18).
//
// For one depth node k of the column grid, on every included face i:
//   C_i  = (rho c_p)_ik * A_i * h_k        h_k = column control-volume thickness of node k
//   G_ij = l_ij * h_k / (d_i/k_i + d_j/k_j)  series two-point flux through the edge midpoint (R1)
// and one BACKWARD-EULER step of length dt solves
//   (C + dt L) x = C T_old,    L the graph Laplacian of G   (SPD M-matrix)
// by Jacobi-preconditioned CG. The state is then advanced in FLUX FORM from the converged
// iterate:
//   Q_i = -dt * sum_j G_ij (x_i - x_j),   T_i += Q_i / C_i
// G_ij == G_ji bit-for-bit and (x_i - x_j) == -(x_j - x_i) exactly, so sum_i Q_i == 0 to
// round-off whatever the CG tolerance: energy conservation does not depend on convergence.
// A capped solve (iteration limit reached before rel_tol) is NOT applied: T is left untouched,
// StepStats.capped is set, and the caller must stop the run. (An unconverged iterate applied in
// flux form conserves energy but is unstable -- see backwardEulerStep.)
//
// The surface node (massless, algebraic) and the bottom node (algebraic geothermal-flux BC)
// are NOT lateral nodes: the caller passes interior nodes 1..n-2 only, and re-applies the
// bottom BC after a lateral step (Fermi R2).
#pragma once
#include <cstdint>
#include <vector>

#include "../mesh/lateral_adjacency.h"

namespace ptm {
namespace lateral {

enum class Scheme { BackwardEuler, TRBDF2 };

struct SolverOptions {
    Scheme scheme = Scheme::BackwardEuler;   // Fermi: BE; TR-BDF2 if G10 fails at production dt
    double rel_tol = 1e-8;   // ||b - A x||_2 / ||b||_2, per linear solve
    int    max_iter = 2000;
};

struct StepStats {
    int    iterations = 0;
    double rel_residual = 0.0;
    bool   capped = false;          // hit max_iter before rel_tol
    double sum_Q = 0.0;             // sum_i C_i dT_i  (J/m of node... energy units); ~0
    double sum_abs_Q = 0.0;         // sum_i |C_i dT_i|
    double max_abs_dT = 0.0;
};

struct Workspace {                  // reused across calls (no per-step allocation)
    std::vector<double> G, C, diag, b, x, r, z, p, Ap;
};

// Per-slot conductance for node thickness h: G[3i+e] = active ? l h / (d_i/k_i + d_j/k_j) : 0.
// edge_active may be null (all coupled edges active). k: [n_faces] conductivity at this node.
void assembleConductance(const Adjacency& adj, const uint8_t* edge_active,
                         const double* k, double h, std::vector<double>& G);

// One lateral step of length dt on one depth node (BE, or TR-BDF2 per opt.scheme; both in
// conservative flux form). T, k, rhocp: [n_faces].
// Faces with no active edge or zero capacity are left untouched.
StepStats backwardEulerStep(const Adjacency& adj, const uint8_t* edge_active,
                            double* T, const double* k, const double* rhocp,
                            double h, double dt, const SolverOptions& opt, Workspace& ws);

// D* = 8 k / (pi h_eff), h_eff = 4 eps sigma T^3 + sqrt(k rho c_p) sqrt(omega)
// (Hayne et al. 2021, Methods E). Units: k W/m/K, rho kg/m^3, c_p J/kg/K, T K, omega rad/s.
double conductionLength(double k, double rho, double cp, double emissivity, double T, double omega);

// Option-4 activation (Fermi F3 ruling): edge (i,j) active iff
//   min(l_i, l_j) < c * max(D*_i, D*_j)
// with l the facet size (mean edge) and D* evaluated at each face's T_ref (the seed
// cycle-mean). Returns [3*n_faces] 0/1, symmetric by construction. n_active_edges counts pairs.
std::vector<uint8_t> edgeActivation(const Adjacency& adj, const std::vector<double>& Dstar,
                                    double c, int64_t* n_active_edges);

}  // namespace lateral
}  // namespace ptm
