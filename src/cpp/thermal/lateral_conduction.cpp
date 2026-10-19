// lateral_conduction.cpp -- see lateral_conduction.h.
#include "lateral_conduction.h"

#include <algorithm>
#include <cmath>

namespace ptm {
namespace lateral {

void assembleConductance(const Adjacency& adj, const uint8_t* edge_active,
                         const double* k, double h, std::vector<double>& G) {
    const int64_t nf = adj.n_faces;
    G.assign(3 * nf, 0.0);
    for (int64_t i = 0; i < nf; ++i) {
        for (int e = 0; e < 3; ++e) {
            const int64_t s = 3 * i + e;
            const int32_t j = adj.nbr[s];
            if (j < 0 || (edge_active && !edge_active[s])) continue;
            const int64_t sj = 3 * static_cast<int64_t>(j) + adj.nbr_slot[s];
            // Written so both sides evaluate the SAME expression tree on swapped operands of a
            // commutative +: G[s] == G[sj] bit-for-bit.
            const double ri = adj.d_half[s] / k[i];
            const double rj = adj.d_half[sj] / k[j];
            const double rsum = (i < j) ? (ri + rj) : (rj + ri);
            G[s] = adj.edge_len[s] * h / rsum;
        }
    }
}

namespace {
// y = A x with A = C + dt L (rows of faces with no active edge reduce to C_i x_i)
inline void applyA(const Adjacency& adj, const std::vector<double>& G, const std::vector<double>& diag,
                   double dt, const std::vector<double>& x, std::vector<double>& y) {
    const int64_t nf = adj.n_faces;
    for (int64_t i = 0; i < nf; ++i) {
        double acc = diag[i] * x[i];
        for (int e = 0; e < 3; ++e) {
            const int64_t s = 3 * i + e;
            if (G[s] != 0.0) acc -= dt * G[s] * x[adj.nbr[s]];
        }
        y[i] = acc;
    }
}
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

namespace {
// Solve (C + theta L) x = rhs by Jacobi-PCG, x0 = x (in/out). Returns iterations, residual, capped.
void pcgSolve(const Adjacency& adj, const std::vector<double>& G, const std::vector<double>& C,
              double theta, const std::vector<double>& rhs, const SolverOptions& opt, Workspace& ws,
              std::vector<double>& x, StepStats& st) {
    const int64_t nf = adj.n_faces;
    for (int64_t i = 0; i < nf; ++i) {
        double gs = 0.0;
        for (int e = 0; e < 3; ++e) gs += G[3 * i + e];
        ws.diag[i] = C[i] + theta * gs;
    }
    applyA(adj, G, ws.diag, theta, x, ws.Ap);
    for (int64_t i = 0; i < nf; ++i) ws.r[i] = rhs[i] - ws.Ap[i];
    const double bnorm = std::sqrt(dot(rhs, rhs));
    const double denom = (bnorm > 0.0) ? bnorm : 1.0;
    double rnorm = std::sqrt(dot(ws.r, ws.r));
    int it = 0;
    if (rnorm / denom > opt.rel_tol) {
        for (int64_t i = 0; i < nf; ++i) ws.z[i] = (ws.diag[i] > 0.0) ? ws.r[i] / ws.diag[i] : 0.0;
        ws.p = ws.z;
        double rz = dot(ws.r, ws.z);
        for (it = 1; it <= opt.max_iter; ++it) {
            applyA(adj, G, ws.diag, theta, ws.p, ws.Ap);
            const double pAp = dot(ws.p, ws.Ap);
            if (!(pAp > 0.0)) break;
            const double alpha = rz / pAp;
            for (int64_t i = 0; i < nf; ++i) { x[i] += alpha * ws.p[i]; ws.r[i] -= alpha * ws.Ap[i]; }
            rnorm = std::sqrt(dot(ws.r, ws.r));
            if (rnorm / denom <= opt.rel_tol) break;
            for (int64_t i = 0; i < nf; ++i) ws.z[i] = (ws.diag[i] > 0.0) ? ws.r[i] / ws.diag[i] : 0.0;
            const double rz_new = dot(ws.r, ws.z);
            const double beta = rz_new / rz;
            rz = rz_new;
            for (int64_t i = 0; i < nf; ++i) ws.p[i] = ws.z[i] + beta * ws.p[i];
        }
        if (it > opt.max_iter) it = opt.max_iter;
    }
    st.iterations += it;
    st.rel_residual = std::max(st.rel_residual, rnorm / denom);
    if (!(rnorm / denom <= opt.rel_tol)) st.capped = true;   // cap OR CG breakdown
}

// Q_i = -w * sum_j G_ij (y_i - y_j): exactly antisymmetric pair fluxes (sum_i Q_i == 0 to round-off).
void fluxForm(const Adjacency& adj, const std::vector<double>& G, double w, const std::vector<double>& y,
              std::vector<double>& Q) {
    const int64_t nf = adj.n_faces;
    for (int64_t i = 0; i < nf; ++i) {
        double q = 0.0;
        for (int e = 0; e < 3; ++e) {
            const int64_t s = 3 * i + e;
            if (G[s] != 0.0) q -= w * G[s] * (y[i] - y[adj.nbr[s]]);
        }
        Q[i] = q;
    }
}
}  // namespace

StepStats backwardEulerStep(const Adjacency& adj, const uint8_t* edge_active,
                            double* T, const double* k, const double* rhocp,
                            double h, double dt, const SolverOptions& opt, Workspace& ws) {
    const int64_t nf = adj.n_faces;
    StepStats st;
    assembleConductance(adj, edge_active, k, h, ws.G);
    ws.C.resize(nf); ws.diag.resize(nf); ws.b.resize(nf); ws.x.resize(nf);
    ws.r.resize(nf); ws.z.resize(nf); ws.p.resize(nf); ws.Ap.resize(nf);
    std::vector<double>& Q = ws.b;   // b is free once a solve has consumed it
    for (int64_t i = 0; i < nf; ++i) ws.C[i] = rhocp[i] * adj.area[i] * h;

    // final[i] = base[i] + Qsum[i] / C[i], all updates in conservative (flux) form
    std::vector<double> base(T, T + nf), Qsum(nf, 0.0);
    // A capped (unconverged) iterate must NOT be applied. The flux-form update from it would
    // still conserve energy, but it is T_old - (dt/C) L x: an explicit-like step on the error,
    // amplified by ~dt*lambda_max (Fo_lat * degree, up to ~50 on Moon3D). Measured on the skewed
    // 24x24 unit lattice with a 2-iteration cap: 1e24 K after 20 steps. T is left untouched and
    // the caller must refuse the run (it is reported, never silently accepted).
    if (opt.scheme == Scheme::BackwardEuler) {
        // (C + dt L) x = C T ;  T_new = T - dt C^-1 L x
        for (int64_t i = 0; i < nf; ++i) { ws.b[i] = ws.C[i] * T[i]; ws.x[i] = T[i]; }
        pcgSolve(adj, ws.G, ws.C, dt, ws.b, opt, ws, ws.x, st);
        if (st.capped) return st;
        fluxForm(adj, ws.G, dt, ws.x, Q);
        for (int64_t i = 0; i < nf; ++i) Qsum[i] = Q[i];
    } else {
        // TR-BDF2 (gamma = 2 - sqrt 2), L-stable, second order. Both stages in flux form:
        //   TR  : (C + g dt/2 L) x1 = C T - g dt/2 L T ;  T1 = T - g dt/2 C^-1 L (T + x1)
        //   BDF2: (C + w dt L) x2 = C (a T1 + b T)     ;  T_new = a T1 + b T - w dt C^-1 L x2
        //   a = 1/(g(2-g)), b = -(1-g)^2/(g(2-g)), a + b = 1 (so a T1 + b T conserves), w = (1-g)/(2-g)
        const double g = 2.0 - std::sqrt(2.0);
        const double a = 1.0 / (g * (2.0 - g)), bb = -(1.0 - g) * (1.0 - g) / (g * (2.0 - g));
        const double w = (1.0 - g) / (2.0 - g);
        std::vector<double> T0(T, T + nf), LT(nf), T1(nf), y(nf);
        fluxForm(adj, ws.G, 1.0, T0, LT);   // LT = -L T
        for (int64_t i = 0; i < nf; ++i) { ws.b[i] = ws.C[i] * T0[i] + 0.5 * g * dt * LT[i]; ws.x[i] = T0[i]; }
        pcgSolve(adj, ws.G, ws.C, 0.5 * g * dt, ws.b, opt, ws, ws.x, st);
        if (st.capped) return st;
        for (int64_t i = 0; i < nf; ++i) y[i] = T0[i] + ws.x[i];
        fluxForm(adj, ws.G, 0.5 * g * dt, y, Q);
        std::vector<double> Q1(Q.begin(), Q.end());
        for (int64_t i = 0; i < nf; ++i) T1[i] = (ws.C[i] > 0.0) ? T0[i] + Q1[i] / ws.C[i] : T0[i];
        for (int64_t i = 0; i < nf; ++i) {
            base[i] = a * T1[i] + bb * T0[i];
            ws.b[i] = ws.C[i] * base[i];
            ws.x[i] = T1[i];
        }
        pcgSolve(adj, ws.G, ws.C, w * dt, ws.b, opt, ws, ws.x, st);
        if (st.capped) return st;
        fluxForm(adj, ws.G, w * dt, ws.x, Q);
        // Energy: sum C base = a sum C T1 + b sum C T0 = sum C T0 + a sum Q1; carry a*Q1 in Qsum so
        // the accounting below sees the whole step's exchange.
        for (int64_t i = 0; i < nf; ++i) {
            Qsum[i] = Q[i];
            st.sum_abs_Q += std::fabs(a * Q1[i]);
            st.sum_Q += a * Q1[i];
        }
    }
    for (int64_t i = 0; i < nf; ++i) {
        const double q = Qsum[i];
        st.sum_Q += q;
        st.sum_abs_Q += std::fabs(q);
        if (ws.C[i] > 0.0) {
            const double Tn = base[i] + q / ws.C[i];
            st.max_abs_dT = std::max(st.max_abs_dT, std::fabs(Tn - T[i]));
            T[i] = Tn;
        }
    }
    return st;
}

double conductionLength(double k, double rho, double cp, double emissivity, double T, double omega) {
    const double sigma = 5.670374419e-8;
    const double h_eff = 4.0 * emissivity * sigma * T * T * T + std::sqrt(k * rho * cp) * std::sqrt(omega);
    return 8.0 * k / (M_PI * h_eff);
}

std::vector<uint8_t> edgeActivation(const Adjacency& adj, const std::vector<double>& Dstar,
                                    double c, int64_t* n_active_edges) {
    const int64_t nf = adj.n_faces;
    std::vector<uint8_t> act(3 * nf, 0);
    int64_t n = 0;
    for (int64_t i = 0; i < nf; ++i)
        for (int e = 0; e < 3; ++e) {
            const int32_t j = adj.nbr[3 * i + e];
            if (j < 0) continue;
            const double lmin = std::min(adj.mean_edge[i], adj.mean_edge[j]);
            const double dmax = std::max(Dstar[i], Dstar[j]);
            if (lmin < c * dmax) {
                act[3 * i + e] = 1;
                if (i < j) ++n;
            }
        }
    if (n_active_edges) *n_active_edges = n;
    return act;
}

}  // namespace lateral
}  // namespace ptm
