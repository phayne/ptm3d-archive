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
    Scheme scheme = Scheme::TRBDF2;   // Fermi addendum 2026-10-18: TR-BDF2 at production dt; BE = documented fallback
    double rel_tol = 1e-8;   // ||b - A x||_2 / ||b||_2, per linear solve
    int    max_iter = 2000;   // set per run with iterationCap(Fo_max)
};

// Fermi addendum: cap >= max(200, 50 sqrt(1 + 4 Fo_max)), Fo_max = max_i dt sum_j G_ij / C_i.
int iterationCap(double Fo_max);
// Fo_max of one node layer for step dt (what the cap and the log need).
double maxFourier(const Adjacency& adj, const std::vector<double>& G, const std::vector<double>& C, double dt);

struct StepStats {
    int    iterations = 0;
    double rel_residual = 0.0;
    bool   capped = false;          // hit max_iter before rel_tol
    double sum_Q = 0.0;             // sum_i C_i dT_i  (J/m of node... energy units); ~0
    double sum_abs_Q = 0.0;         // sum_i |C_i dT_i|
    double max_abs_dT = 0.0;
    double Fo_max = 0.0;            // max_i dt sum_j G_ij / C_i for this step
    int64_t worst_face = -1;        // face with the largest |residual| of the last solve (for the FATAL)
};

struct Workspace {                  // reused across calls (no per-step allocation)
    std::vector<double> G, C, diag, b, x, r, z, p, Ap;
};

// Per-slot conductance for node thickness h: G[3i+e] = active ? l h / (d_i/k_i + d_j/k_j) : 0.
// edge_active may be null (all coupled edges active). k: [n_faces] conductivity at this node.
void assembleConductance(const Adjacency& adj, const uint8_t* edge_active,
                         const double* k, double h, std::vector<double>& G);

// One lateral step of length dt on one depth node (TR-BDF2 default, or BE, per opt.scheme;
// both in conservative flux form). On st.capped the caller must FATAL, naming step, layer,
// st.iterations, st.rel_residual and st.worst_face. T, k, rhocp: [n_faces].
// Faces with no active edge or zero capacity are left untouched.
StepStats backwardEulerStep(const Adjacency& adj, const uint8_t* edge_active,
                            double* T, const double* k, const double* rhocp,
                            double h, double dt, const SolverOptions& opt, Workspace& ws);

// Balanced-split explicit stage (Fermi ruling 2026-10-19): the lateral increment over tau at the
// substep-start state, inc_i = tau * sum_j G_ij (T_j - T_i) / C_i with C_i = rc_i A_i h, added as a
// source to the vertical CN step and removed again before the implicit BE correction. Faces with
// zero capacity get 0. Returns sum_i C_i inc_i (should be round-off) in *sum_q and sum |C_i inc_i|
// in *sum_abs_q; both deterministic for any thread count.
void explicitIncrement(const Adjacency& adj, const std::vector<double>& G, const double* T,
                       const double* rhocp, double h, double tau, double* inc,
                       double* sum_q, double* sum_abs_q);

// D* = 8 k / (pi h_eff), h_eff = 4 eps sigma T^3 + sqrt(k rho c_p) sqrt(omega)
// (Hayne et al. 2021, Methods E). Units: k W/m/K, rho kg/m^3, c_p J/kg/K, T K, omega rad/s.
double conductionLength(double k, double rho, double cp, double emissivity, double T, double omega);

// Option-4 activation (Fermi F3 ruling): edge (i,j) active iff
//   min(l_i, l_j) < c * max(D*_i, D*_j)
// with l the facet size (mean edge) and D* evaluated at each face's T_ref (the seed
// cycle-mean). Returns [3*n_faces] 0/1, symmetric by construction. n_active_edges counts pairs.
std::vector<uint8_t> edgeActivation(const Adjacency& adj, const std::vector<double>& Dstar,
                                    double c, int64_t* n_active_edges);

// -----------------------------------------------------------------------------------------------
// Network: the balanced-split lateral stages on ALL capacitive node layers at once (C1c perf
// fusion, 2026-10-21). The engine and the G10' harness both call this, so G10' measures the code
// the engine runs.
//
// Arrays are layer-major, [layer * n_faces + face]. Every kernel is ONE parallel region over
// (layer, 8192-face chunk) blocks instead of one region per layer per kernel (the per-layer form
// cost 585 s wall / 6373 s sys on the n968 smoke, almost all OpenMP region overhead).
//
// ARITHMETIC CONTRACT: per layer, every value is computed by the same expression tree as the
// single-layer functions above (assembleConductance, explicitIncrement, backwardEulerStep with
// Scheme::BackwardEuler, iterationCap, maxFourier), and every reduction that feeds the state (the
// PCG dot products) uses the same fixed 8192-element chunks, each summed in index order, chunk
// partials summed serially in chunk order.
//   - Thread count: results are BIT-IDENTICAL for any thread count (the per-block code does not
//     depend on the thread; test_lateral_network N1).
//   - Versus the single-layer path: equal to round-off, not bit-for-bit by construction, because
//     libthermal is compiled with -ffast-math and the compiler may contract/reassociate the two
//     loop shapes differently. Measured (test_lateral_network N2, 28,800 faces x 6 layers, 20
//     steps): implicit-stage T bit-identical, explicit increment within 4.4e-16 K, identical PCG
//     iteration counts; the n968 engine smoke's float32 outputs are byte-identical.
//   - The closure DIAGNOSTIC (sum Q / sum |Q|) is chunked, so its last digits in the log differ
//     from the serial face-order sum; it never feeds the state.
class Network {
public:
    struct Stats {
        int     iterations = 0;          // PCG iterations summed over layers (the engine's log unit)
        double  fo_max = 0.0;            // max over layers of max_i dt sum_j G_ij / C_i
        double  closure_worst = 0.0;     // max over layers of |sum_i Q_i| / sum_i |Q_i|
        bool    capped = false;          // some layer hit its iteration cap (or CG broke down)
        int     capped_layer = -1;       // first such layer (0-based node-layer index)
        int     cap = 0;                 // that layer's iteration cap
        int     capped_iterations = 0;
        double  capped_rel_residual = 0.0;
        int64_t worst_face = -1;         // that layer's largest-|residual| face
        double  capped_fo = 0.0;
    };

    // adj and edge_active (may be null = every coupled edge) must outlive the network.
    // h: [n_layers] control-volume thickness per node layer. n_threads <= 0 = OpenMP default.
    void init(const Adjacency* adj, const uint8_t* edge_active, std::vector<double> h, int n_threads);
    int     layers() const { return n_layers_; }
    int64_t faces() const { return nf_; }

    // Conductances G (all layers) from K, and capacities C = RC * area * h. K, RC: [n_layers * nf]
    // at the substep-start state. Both stages below use these.
    void assemble(const double* K, const double* RC);
    // Explicit stage: inc = tau * sum_j G_ij (T_j - T_i) / C_i (0 where C_i <= 0). Returns the
    // worst per-layer closure |sum q| / sum |q| of the increment's energy (round-off).
    double explicitIncrement(const double* T, double tau, double* inc);
    // Implicit stage: per layer, (C + tau L) x = C T by Jacobi-PCG (x0 = T, rel_tol, per-layer cap
    // iterationCap(Fo_layer)), then T += Q / C with Q the flux form of x. On Stats.capped NOTHING
    // is applied to any layer (T untouched): the caller must stop the run.
    Stats implicitBE(double* T, double tau, double rel_tol);

    // MONOLITHIC substep (C1c, 2026-10-21, after G15): the production vertical Crank-Nicolson rows and a
    // BACKWARD-EULER lateral term solved as ONE coupled system -- no splitting:
    //     a x[l-1] + b x[l] + c x[l+1] + (tau / C_l) sum_j G_lj (x_l - x_j) = d        (per face, layer l)
    // A, B, Cc, D: [n_layers * nf] the row-normalized CN coefficients of the interior nodes, exactly as
    // thermal::buildTridiagonalSystemBatch leaves them (surface and bottom values folded into d; A of the
    // top row and Cc of the bottom row are 0). Rows scaled by C/tau make the operator symmetric positive
    // definite (h g1 = 1/dz_lo, h g2 = 1/dz_hi on PTM's node grid), solved by CG with an exact per-column
    // tridiagonal (Thomas) preconditioner. Then, for EXACT lateral energy conservation, the lateral flux of
    // the converged iterate (antisymmetric: sums to 0 to round-off) is applied as a source and every column
    // is solved exactly: T_out = tridiag^-1 (D + src). Steady states are exact; the BE lateral term is
    // L-stable (no lateral overshoot at any Fo). assemble(K, RC) at the substep-start state must precede.
    // On Stats.capped nothing is written to T_out.
    Stats implicitCoupled(const double* A, const double* B, const double* Cc, const double* D,
                          double tau, double rel_tol, double* T_out);

private:
    const Adjacency* adj_ = nullptr;
    const uint8_t* act_ = nullptr;
    std::vector<double> h_;
    int n_layers_ = 0, nt_ = 0;
    int64_t nf_ = 0, nc_ = 0;                       // faces, chunks per layer
    std::vector<double> G_, C_;                     // [L * 3nf], [L * nf]
    std::vector<double> diag_, x_, r_, z_, p_, Ap_, rhs_;   // [L * nf]
    std::vector<double> part_;                      // [L * nc] chunk partials
};

}  // namespace lateral
}  // namespace ptm
