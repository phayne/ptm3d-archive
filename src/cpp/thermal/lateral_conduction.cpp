// lateral_conduction.cpp -- see lateral_conduction.h.
#include "lateral_conduction.h"

#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ptm {
namespace lateral {

void assembleConductance(const Adjacency& adj, const uint8_t* edge_active,
                         const double* k, double h, std::vector<double>& G) {
    const int64_t nf = adj.n_faces;
    G.assign(3 * nf, 0.0);
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nf; ++i) {
        double acc = diag[i] * x[i];
        for (int e = 0; e < 3; ++e) {
            const int64_t s = 3 * i + e;
            if (G[s] != 0.0) acc -= dt * G[s] * x[adj.nbr[s]];
        }
        y[i] = acc;
    }
}
// Deterministic for ANY thread count: fixed 8192-element chunks, partials summed serially.
constexpr int64_t kChunk = 8192;
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    const int64_t n = static_cast<int64_t>(a.size());
    const int64_t nc = (n + kChunk - 1) / kChunk;
    std::vector<double> part(static_cast<size_t>(nc), 0.0);
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nc; ++c) {
        double s = 0.0;
        const int64_t hi = std::min(n, (c + 1) * kChunk);
        for (int64_t i = c * kChunk; i < hi; ++i) s += a[i] * b[i];
        part[static_cast<size_t>(c)] = s;
    }
    double s = 0.0;
    for (double v : part) s += v;
    return s;
}
}  // namespace

namespace {
// Solve (C + theta L) x = rhs by Jacobi-PCG, x0 = x (in/out). Returns iterations, residual, capped.
void pcgSolve(const Adjacency& adj, const std::vector<double>& G, const std::vector<double>& C,
              double theta, const std::vector<double>& rhs, const SolverOptions& opt, Workspace& ws,
              std::vector<double>& x, StepStats& st) {
    const int64_t nf = adj.n_faces;
    #pragma omp parallel for schedule(static)
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
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < nf; ++i) { x[i] += alpha * ws.p[i]; ws.r[i] -= alpha * ws.Ap[i]; }
            rnorm = std::sqrt(dot(ws.r, ws.r));
            if (rnorm / denom <= opt.rel_tol) break;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < nf; ++i) ws.z[i] = (ws.diag[i] > 0.0) ? ws.r[i] / ws.diag[i] : 0.0;
            const double rz_new = dot(ws.r, ws.z);
            const double beta = rz_new / rz;
            rz = rz_new;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < nf; ++i) ws.p[i] = ws.z[i] + beta * ws.p[i];
        }
        if (it > opt.max_iter) it = opt.max_iter;
    }
    st.iterations += it;
    st.rel_residual = std::max(st.rel_residual, rnorm / denom);
    if (!(rnorm / denom <= opt.rel_tol)) {   // cap OR CG breakdown
        st.capped = true;
        double worst = -1.0;
        for (int64_t i = 0; i < nf; ++i)
            if (std::fabs(ws.r[i]) > worst) { worst = std::fabs(ws.r[i]); st.worst_face = i; }
    }
}

// Q_i = -w * sum_j G_ij (y_i - y_j): exactly antisymmetric pair fluxes (sum_i Q_i == 0 to round-off).
void fluxForm(const Adjacency& adj, const std::vector<double>& G, double w, const std::vector<double>& y,
              std::vector<double>& Q) {
    const int64_t nf = adj.n_faces;
    #pragma omp parallel for schedule(static)
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
    st.Fo_max = maxFourier(adj, ws.G, ws.C, dt);

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

void explicitIncrement(const Adjacency& adj, const std::vector<double>& G, const double* T,
                       const double* rhocp, double h, double tau, double* inc,
                       double* sum_q, double* sum_abs_q) {
    const int64_t nf = adj.n_faces;
    const int64_t nc = (nf + kChunk - 1) / kChunk;
    std::vector<double> pq(static_cast<size_t>(nc), 0.0), pa(static_cast<size_t>(nc), 0.0);
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nc; ++c) {
        double sq = 0.0, sa = 0.0;
        const int64_t hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = c * kChunk; i < hi; ++i) {
            double q = 0.0;
            for (int e = 0; e < 3; ++e) {
                const int64_t s = 3 * i + e;
                if (G[s] != 0.0) q += G[s] * (T[adj.nbr[s]] - T[i]);
            }
            q *= tau;
            const double C = rhocp[i] * adj.area[i] * h;
            inc[i] = (C > 0.0) ? q / C : 0.0;
            sq += q; sa += std::fabs(q);
        }
        pq[static_cast<size_t>(c)] = sq; pa[static_cast<size_t>(c)] = sa;
    }
    double a = 0.0, b = 0.0;
    for (int64_t c = 0; c < nc; ++c) { a += pq[static_cast<size_t>(c)]; b += pa[static_cast<size_t>(c)]; }
    if (sum_q) *sum_q = a;
    if (sum_abs_q) *sum_abs_q = b;
}

int iterationCap(double Fo_max) {
    return std::max(200, static_cast<int>(std::ceil(50.0 * std::sqrt(1.0 + 4.0 * std::max(0.0, Fo_max)))));
}

double maxFourier(const Adjacency& adj, const std::vector<double>& G, const std::vector<double>& C, double dt) {
    double fo = 0.0;
    for (int64_t i = 0; i < adj.n_faces; ++i) {
        if (!(C[i] > 0.0)) continue;
        const double gs = G[3 * i] + G[3 * i + 1] + G[3 * i + 2];
        fo = std::max(fo, dt * gs / C[i]);
    }
    return fo;
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

// ---------------------------------------------------------------------------------------------
// Network (see the header for the arithmetic contract).
// ---------------------------------------------------------------------------------------------
void Network::init(const Adjacency* adj, const uint8_t* edge_active, std::vector<double> h, int n_threads) {
    adj_ = adj;
    act_ = edge_active;
    h_ = std::move(h);
    n_layers_ = static_cast<int>(h_.size());
    nf_ = adj->n_faces;
    nc_ = (nf_ + kChunk - 1) / kChunk;
#ifdef _OPENMP
    nt_ = (n_threads > 0) ? n_threads : omp_get_max_threads();
#else
    nt_ = 1;
    (void)n_threads;
#endif
    const size_t L = static_cast<size_t>(n_layers_);
    const size_t N = L * static_cast<size_t>(nf_);
    G_.assign(3 * N, 0.0);
    C_.assign(N, 0.0);
    diag_.assign(N, 0.0); x_.assign(N, 0.0); r_.assign(N, 0.0); z_.assign(N, 0.0);
    p_.assign(N, 0.0); Ap_.assign(N, 0.0); rhs_.assign(N, 0.0);
    part_.assign(L * static_cast<size_t>(nc_), 0.0);
}

void Network::assemble(const double* K, const double* RC) {
    const Adjacency& adj = *adj_;
    const int64_t nf = nf_, nc = nc_, nb = static_cast<int64_t>(n_layers_) * nc;
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t l = b / nc, c = b % nc;
        const double h = h_[static_cast<size_t>(l)];
        const double* k = K + l * nf;
        const double* rc = RC + l * nf;
        double* G = G_.data() + 3 * l * nf;
        double* C = C_.data() + l * nf;
        const int64_t hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = c * kChunk; i < hi; ++i) {
            for (int e = 0; e < 3; ++e) {   // == assembleConductance
                const int64_t s = 3 * i + e;
                G[s] = 0.0;
                const int32_t j = adj.nbr[s];
                if (j < 0 || (act_ && !act_[s])) continue;
                const int64_t sj = 3 * static_cast<int64_t>(j) + adj.nbr_slot[s];
                const double ri = adj.d_half[s] / k[i];
                const double rj = adj.d_half[sj] / k[j];
                const double rsum = (i < j) ? (ri + rj) : (rj + ri);
                G[s] = adj.edge_len[s] * h / rsum;
            }
            C[i] = rc[i] * adj.area[i] * h;   // == backwardEulerStep's ws.C
        }
    }
}

double Network::explicitIncrement(const double* T, double tau, double* inc) {
    const Adjacency& adj = *adj_;
    const int64_t nf = nf_, nc = nc_, nb = static_cast<int64_t>(n_layers_) * nc;
    std::vector<double> pq(static_cast<size_t>(nb), 0.0), pa(static_cast<size_t>(nb), 0.0);
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t l = b / nc, c = b % nc;
        const double* G = G_.data() + 3 * l * nf;
        const double* C = C_.data() + l * nf;
        const double* Tl = T + l * nf;
        double* il = inc + l * nf;
        double sq = 0.0, sa = 0.0;
        const int64_t hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = c * kChunk; i < hi; ++i) {   // == explicitIncrement
            double q = 0.0;
            for (int e = 0; e < 3; ++e) {
                const int64_t s = 3 * i + e;
                if (G[s] != 0.0) q += G[s] * (Tl[adj.nbr[s]] - Tl[i]);
            }
            q *= tau;
            il[i] = (C[i] > 0.0) ? q / C[i] : 0.0;
            sq += q; sa += std::fabs(q);
        }
        pq[static_cast<size_t>(b)] = sq; pa[static_cast<size_t>(b)] = sa;
    }
    double worst = 0.0;
    for (int64_t l = 0; l < n_layers_; ++l) {
        double a = 0.0, s = 0.0;
        for (int64_t c = 0; c < nc; ++c) {
            a += pq[static_cast<size_t>(l * nc + c)];
            s += pa[static_cast<size_t>(l * nc + c)];
        }
        if (s > 0.0) worst = std::max(worst, std::fabs(a) / s);
    }
    return worst;
}

Network::Stats Network::implicitBE(double* T, double tau, double rel_tol) {
    const Adjacency& adj = *adj_;
    const int L = n_layers_;
    const int64_t nf = nf_, nc = nc_, nb = static_cast<int64_t>(L) * nc;
    auto U = [](int64_t v) { return static_cast<size_t>(v); };
    Stats st;
    // Per-layer cap from Fo (== maxFourier + iterationCap, evaluated per layer).
    std::vector<int> cap(U(L));
    std::vector<double> fo(U(L), 0.0);
    for (int l = 0; l < L; ++l) {
        const double* G = G_.data() + 3 * static_cast<int64_t>(l) * nf;
        const double* C = C_.data() + static_cast<int64_t>(l) * nf;
        double f = 0.0;
        for (int64_t i = 0; i < nf; ++i) {
            if (!(C[i] > 0.0)) continue;
            const double gs = G[3 * i] + G[3 * i + 1] + G[3 * i + 2];
            f = std::max(f, tau * gs / C[i]);
        }
        fo[U(l)] = f;
        cap[U(l)] = iterationCap(f);
        st.fo_max = std::max(st.fo_max, f);
    }
    // Setup: rhs = C T, x0 = T, diag = C + tau sum G.
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t l = b / nc, c = b % nc, o = l * nf;
        const double* G = G_.data() + 3 * o;
        const double* C = C_.data() + o;
        const double* Tl = T + o;
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = lo; i < hi; ++i) {
            rhs_[U(o + i)] = C[i] * Tl[i];
            x_[U(o + i)] = Tl[i];
            double gs = 0.0;
            for (int e = 0; e < 3; ++e) gs += G[3 * i + e];
            diag_[U(o + i)] = C[i] + tau * gs;
        }
    }
    // Ap = A x0, r = rhs - Ap; chunk partials of dot(rhs, rhs) and dot(r, r).
    std::vector<double> pb(U(nb), 0.0);
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t l = b / nc, c = b % nc, o = l * nf;
        const double* G = G_.data() + 3 * o;
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = lo; i < hi; ++i) {   // == applyA, then r = rhs - Ap
            double acc = diag_[U(o + i)] * x_[U(o + i)];
            for (int e = 0; e < 3; ++e) {
                const int64_t s = 3 * i + e;
                if (G[s] != 0.0) acc -= tau * G[s] * x_[U(o + adj.nbr[s])];
            }
            Ap_[U(o + i)] = acc;
            r_[U(o + i)] = rhs_[U(o + i)] - acc;
        }
        double sb = 0.0, sr = 0.0;
        for (int64_t i = lo; i < hi; ++i) sb += rhs_[U(o + i)] * rhs_[U(o + i)];
        for (int64_t i = lo; i < hi; ++i) sr += r_[U(o + i)] * r_[U(o + i)];
        pb[U(b)] = sb;
        part_[U(b)] = sr;
    }
    auto layerSum = [&](const std::vector<double>& p, int l) {
        double s = 0.0;
        for (int64_t c = 0; c < nc; ++c) s += p[U(l * nc + c)];
        return s;
    };
    std::vector<double> denom(U(L)), rnorm(U(L)), rz(U(L), 0.0), alpha(U(L), 0.0), beta(U(L), 0.0);
    std::vector<int> it(U(L), 0);
    std::vector<uint8_t> run(U(L), 0);   // layer still iterating
    int n_run = 0;
    for (int l = 0; l < L; ++l) {
        const double bnorm = std::sqrt(layerSum(pb, l));
        denom[U(l)] = (bnorm > 0.0) ? bnorm : 1.0;
        rnorm[U(l)] = std::sqrt(layerSum(part_, l));
        if (rnorm[U(l)] / denom[U(l)] > rel_tol) { run[U(l)] = 1; it[U(l)] = 1; ++n_run; }
    }
    // z = r / diag (and p = z on the first pass), chunk partials of dot(r, z); running layers only.
    auto zStep = [&](bool set_p) {
        #pragma omp parallel for num_threads(nt_) schedule(static)
        for (int64_t b = 0; b < nb; ++b) {
            const int64_t l = b / nc, c = b % nc, o = l * nf;
            if (!run[U(l)]) continue;
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int64_t i = lo; i < hi; ++i) {
                z_[U(o + i)] = (diag_[U(o + i)] > 0.0) ? r_[U(o + i)] / diag_[U(o + i)] : 0.0;
                if (set_p) p_[U(o + i)] = z_[U(o + i)];
            }
            double s = 0.0;
            for (int64_t i = lo; i < hi; ++i) s += r_[U(o + i)] * z_[U(o + i)];
            part_[U(b)] = s;
        }
    };
    if (n_run) {
        zStep(true);
        for (int l = 0; l < L; ++l) if (run[U(l)]) rz[U(l)] = layerSum(part_, l);
    }
    while (n_run > 0) {
        // Ap = A p and chunk partials of dot(p, Ap)
        #pragma omp parallel for num_threads(nt_) schedule(static)
        for (int64_t b = 0; b < nb; ++b) {
            const int64_t l = b / nc, c = b % nc, o = l * nf;
            if (!run[U(l)]) continue;
            const double* G = G_.data() + 3 * o;
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int64_t i = lo; i < hi; ++i) {
                double acc = diag_[U(o + i)] * p_[U(o + i)];
                for (int e = 0; e < 3; ++e) {
                    const int64_t s = 3 * i + e;
                    if (G[s] != 0.0) acc -= tau * G[s] * p_[U(o + adj.nbr[s])];
                }
                Ap_[U(o + i)] = acc;
            }
            double s = 0.0;
            for (int64_t i = lo; i < hi; ++i) s += p_[U(o + i)] * Ap_[U(o + i)];
            part_[U(b)] = s;
        }
        for (int l = 0; l < L; ++l) {
            if (!run[U(l)]) continue;
            const double pAp = layerSum(part_, l);
            if (!(pAp > 0.0)) { run[U(l)] = 0; --n_run; continue; }   // CG breakdown (reported below)
            alpha[U(l)] = rz[U(l)] / pAp;
        }
        if (n_run == 0) break;
        // x += alpha p, r -= alpha Ap, chunk partials of dot(r, r)
        #pragma omp parallel for num_threads(nt_) schedule(static)
        for (int64_t b = 0; b < nb; ++b) {
            const int64_t l = b / nc, c = b % nc, o = l * nf;
            if (!run[U(l)]) continue;
            const double a = alpha[U(l)];
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int64_t i = lo; i < hi; ++i) { x_[U(o + i)] += a * p_[U(o + i)]; r_[U(o + i)] -= a * Ap_[U(o + i)]; }
            double s = 0.0;
            for (int64_t i = lo; i < hi; ++i) s += r_[U(o + i)] * r_[U(o + i)];
            part_[U(b)] = s;
        }
        for (int l = 0; l < L; ++l) {
            if (!run[U(l)]) continue;
            rnorm[U(l)] = std::sqrt(layerSum(part_, l));
            if (rnorm[U(l)] / denom[U(l)] <= rel_tol) { run[U(l)] = 0; --n_run; }
        }
        if (n_run == 0) break;
        zStep(false);
        for (int l = 0; l < L; ++l) {
            if (!run[U(l)]) continue;
            const double rz_new = layerSum(part_, l);
            beta[U(l)] = rz_new / rz[U(l)];
            rz[U(l)] = rz_new;
        }
        #pragma omp parallel for num_threads(nt_) schedule(static)
        for (int64_t b = 0; b < nb; ++b) {
            const int64_t l = b / nc, c = b % nc, o = l * nf;
            if (!run[U(l)]) continue;
            const double bt = beta[U(l)];
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int64_t i = lo; i < hi; ++i) p_[U(o + i)] = z_[U(o + i)] + bt * p_[U(o + i)];
        }
        for (int l = 0; l < L; ++l) {   // == the for (it = 1; it <= max_iter; ++it) loop + clamp
            if (!run[U(l)]) continue;
            if (++it[U(l)] > cap[U(l)]) { it[U(l)] = cap[U(l)]; run[U(l)] = 0; --n_run; }
        }
    }
    for (int l = 0; l < L; ++l) {
        st.iterations += it[U(l)];
        const double rr = rnorm[U(l)] / denom[U(l)];
        if (!(rr <= rel_tol) && !st.capped) {   // cap OR CG breakdown: report the first such layer
            st.capped = true;
            st.capped_layer = l;
            st.cap = cap[U(l)];
            st.capped_iterations = it[U(l)];
            st.capped_rel_residual = rr;
            st.capped_fo = fo[U(l)];
            double worst = -1.0;
            const int64_t o = static_cast<int64_t>(l) * nf;
            for (int64_t i = 0; i < nf; ++i)
                if (std::fabs(r_[U(o + i)]) > worst) { worst = std::fabs(r_[U(o + i)]); st.worst_face = i; }
        }
    }
    if (st.capped) return st;   // nothing applied to any layer (see backwardEulerStep)
    // T += Q / C with Q = -tau sum_j G_ij (x_i - x_j)  (== fluxForm + the BE update)
    std::vector<double> pq(U(nb), 0.0), pa(U(nb), 0.0);
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t l = b / nc, c = b % nc, o = l * nf;
        const double* G = G_.data() + 3 * o;
        const double* C = C_.data() + o;
        double* Tl = T + o;
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        double sq = 0.0, sa = 0.0;
        for (int64_t i = lo; i < hi; ++i) {
            double q = 0.0;
            for (int e = 0; e < 3; ++e) {
                const int64_t s = 3 * i + e;
                if (G[s] != 0.0) q -= tau * G[s] * (x_[U(o + i)] - x_[U(o + adj.nbr[s])]);
            }
            sq += q; sa += std::fabs(q);
            if (C[i] > 0.0) Tl[i] = Tl[i] + q / C[i];
        }
        pq[U(b)] = sq; pa[U(b)] = sa;
    }
    for (int l = 0; l < L; ++l) {
        double a = 0.0, s = 0.0;
        for (int64_t c = 0; c < nc; ++c) { a += pq[U(l * nc + c)]; s += pa[U(l * nc + c)]; }
        if (s > 0.0) st.closure_worst = std::max(st.closure_worst, std::fabs(a) / s);
    }
    return st;
}

Network::Stats Network::implicitCoupled(const double* A, const double* B, const double* Cc, const double* D,
                                        double tau, double rel_tol, double* T_out) {
    const Adjacency& adj = *adj_;
    const int L = n_layers_;
    const int64_t nf = nf_, nc = nc_;
    auto U = [](int64_t v) { return static_cast<size_t>(v); };
    Stats st;
    // Fo (report + cap): max over layers and faces of tau sum G / C
    for (int l = 0; l < L; ++l) {
        const double* G = G_.data() + 3 * static_cast<int64_t>(l) * nf;
        const double* C = C_.data() + static_cast<int64_t>(l) * nf;
        for (int64_t i = 0; i < nf; ++i) {
            if (!(C[i] > 0.0)) continue;
            st.fo_max = std::max(st.fo_max, tau * (G[3 * i] + G[3 * i + 1] + G[3 * i + 2]) / C[i]);
        }
    }
    const int cap = iterationCap(st.fo_max);
    // scaled coefficients: s = C / tau; Ms x = s (a x[l-1] + b x[l] + c x[l+1]) + sum G (x_l - x_j)
    // stored in the PCG work arrays: diag_ = s b + sum G ; rhs_ = s d ; Ap_ (reused) holds s a, z_ holds s c
    // for the duration of the setup; the column preconditioner is built from them.
    std::vector<double> sa(U(L) * U(nf)), sc(U(L) * U(nf));
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t c = 0; c < nc; ++c) {
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int l = 0; l < L; ++l) {
            const int64_t o = static_cast<int64_t>(l) * nf;
            const double* G = G_.data() + 3 * o;
            for (int64_t i = lo; i < hi; ++i) {
                const double s = C_[U(o + i)] / tau;
                const double gs = G[3 * i] + G[3 * i + 1] + G[3 * i + 2];
                sa[U(o + i)] = s * A[o + i];
                sc[U(o + i)] = s * Cc[o + i];
                diag_[U(o + i)] = s * B[o + i] + gs;
                rhs_[U(o + i)] = s * D[o + i];
                x_[U(o + i)] = D[o + i] / B[o + i];   // x0: the column's own diagonal guess (refined below)
            }
        }
    }
    // x0 = the uncoupled column solve (vertical CN alone): Thomas per face on (A, B, Cc, D)
    auto thomasColumn = [&](const double* a, const double* b, const double* cc, const double* r, double* x,
                            int64_t i, std::vector<double>& cp, std::vector<double>& dp) {
        // a, b, cc, r, x: layer-major arrays; solves column i
        double m = b[i];
        cp[0] = cc[i] / m;
        dp[0] = r[i] / m;
        for (int l = 1; l < L; ++l) {
            const int64_t o = static_cast<int64_t>(l) * nf + i;
            m = b[o] - a[o] * cp[U(l - 1)];
            cp[U(l)] = cc[o] / m;
            dp[U(l)] = (r[o] - a[o] * dp[U(l - 1)]) / m;
        }
        x[static_cast<int64_t>(L - 1) * nf + i] = dp[U(L - 1)];
        for (int l = L - 2; l >= 0; --l) {
            const int64_t o = static_cast<int64_t>(l) * nf + i;
            x[o] = dp[U(l)] - cp[U(l)] * x[o + nf];
        }
    };
    #pragma omp parallel num_threads(nt_)
    {
        std::vector<double> cp(U(L)), dp(U(L));
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int64_t i = lo; i < hi; ++i) thomasColumn(A, B, Cc, D, x_.data(), i, cp, dp);
        }
    }
    // y = Ms v  (vertical scaled tridiagonal + lateral graph), per face chunk, all layers
    auto applyM = [&](const std::vector<double>& v, std::vector<double>& y, int64_t c) {
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int l = 0; l < L; ++l) {
            const int64_t o = static_cast<int64_t>(l) * nf;
            const double* G = G_.data() + 3 * o;
            for (int64_t i = lo; i < hi; ++i) {
                double acc = diag_[U(o + i)] * v[U(o + i)];
                if (l > 0) acc += sa[U(o + i)] * v[U(o + i - nf)];
                if (l < L - 1) acc += sc[U(o + i)] * v[U(o + i + nf)];
                for (int e = 0; e < 3; ++e) {
                    const int64_t s = 3 * i + e;
                    if (G[s] != 0.0) acc -= G[s] * v[U(o + adj.nbr[s])];
                }
                y[U(o + i)] = acc;
            }
        }
    };
    // preconditioner: per column, the scaled tridiagonal with diag_ (includes sum G): z = P^-1 r
    auto precond = [&](std::vector<double>& cp, std::vector<double>& dp, int64_t c) {
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int64_t i = lo; i < hi; ++i) thomasColumn(sa.data(), diag_.data(), sc.data(), r_.data(), z_.data(), i, cp, dp);
    };
    auto dotPart = [&](const std::vector<double>& u, const std::vector<double>& v, int64_t c) {
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        double s = 0.0;
        for (int64_t i = lo; i < hi; ++i)
            for (int l = 0; l < L; ++l) { const size_t k = U(static_cast<int64_t>(l) * nf + i); s += u[k] * v[k]; }
        return s;
    };
    auto sumParts = [&]() { double s = 0.0; for (int64_t c = 0; c < nc; ++c) s += part_[U(c)]; return s; };
    std::vector<double> pb(U(nc), 0.0);
    // r0 = b - M x0 ; ||b||
    #pragma omp parallel for num_threads(nt_) schedule(static)
    for (int64_t c = 0; c < nc; ++c) {
        applyM(x_, Ap_, c);
        const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
        for (int l = 0; l < L; ++l)
            for (int64_t i = lo; i < hi; ++i) { const size_t k = U(static_cast<int64_t>(l) * nf + i); r_[k] = rhs_[k] - Ap_[k]; }
        pb[U(c)] = dotPart(rhs_, rhs_, c);
        part_[U(c)] = dotPart(r_, r_, c);
    }
    double bn = 0.0; for (double v : pb) bn += v;
    const double denom = (bn > 0.0) ? std::sqrt(bn) : 1.0;
    double rnorm = std::sqrt(sumParts());
    int it = 0;
    if (rnorm / denom > rel_tol) {
        double rz = 0.0;
        #pragma omp parallel num_threads(nt_)
        {
            std::vector<double> cp(U(L)), dp(U(L));
            #pragma omp for schedule(static)
            for (int64_t c = 0; c < nc; ++c) {
                precond(cp, dp, c);
                const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
                for (int l = 0; l < L; ++l)
                    for (int64_t i = lo; i < hi; ++i) { const size_t k = U(static_cast<int64_t>(l) * nf + i); p_[k] = z_[k]; }
                part_[U(c)] = dotPart(r_, z_, c);
            }
        }
        rz = sumParts();
        for (it = 1; it <= cap; ++it) {
            #pragma omp parallel for num_threads(nt_) schedule(static)
            for (int64_t c = 0; c < nc; ++c) { applyM(p_, Ap_, c); part_[U(c)] = dotPart(p_, Ap_, c); }
            const double pAp = sumParts();
            if (!(pAp > 0.0)) break;
            const double alpha = rz / pAp;
            #pragma omp parallel for num_threads(nt_) schedule(static)
            for (int64_t c = 0; c < nc; ++c) {
                const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
                for (int l = 0; l < L; ++l)
                    for (int64_t i = lo; i < hi; ++i) {
                        const size_t k = U(static_cast<int64_t>(l) * nf + i);
                        x_[k] += alpha * p_[k];
                        r_[k] -= alpha * Ap_[k];
                    }
                part_[U(c)] = dotPart(r_, r_, c);
            }
            rnorm = std::sqrt(sumParts());
            if (rnorm / denom <= rel_tol) break;
            #pragma omp parallel num_threads(nt_)
            {
                std::vector<double> cp(U(L)), dp(U(L));
                #pragma omp for schedule(static)
                for (int64_t c = 0; c < nc; ++c) { precond(cp, dp, c); part_[U(c)] = dotPart(r_, z_, c); }
            }
            const double rz_new = sumParts();
            const double beta = rz_new / rz;
            rz = rz_new;
            #pragma omp parallel for num_threads(nt_) schedule(static)
            for (int64_t c = 0; c < nc; ++c) {
                const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
                for (int l = 0; l < L; ++l)
                    for (int64_t i = lo; i < hi; ++i) { const size_t k = U(static_cast<int64_t>(l) * nf + i); p_[k] = z_[k] + beta * p_[k]; }
            }
        }
        if (it > cap) it = cap;
    }
    st.iterations = it;
    if (!(rnorm / denom <= rel_tol)) {
        st.capped = true; st.capped_layer = -1; st.cap = cap; st.capped_iterations = it;
        st.capped_rel_residual = rnorm / denom; st.capped_fo = st.fo_max;
        double worst = -1.0;
        for (int64_t k = 0; k < static_cast<int64_t>(r_.size()); ++k)
            if (std::fabs(r_[U(k)]) > worst) { worst = std::fabs(r_[U(k)]); st.worst_face = k % nf; st.capped_layer = static_cast<int>(k / nf); }
        return st;
    }
    // exact-conservation finish: src = -(tau / C) sum_j G (x_i - x_j) (antisymmetric pair fluxes), then the
    // exact column solve tridiag(T_out) = D + src. Closure = |sum C src| / sum |C src| per layer.
    std::vector<double> pq(U(L) * U(nc), 0.0), pa(U(L) * U(nc), 0.0);
    #pragma omp parallel num_threads(nt_)
    {
        std::vector<double> cp(U(L)), dp(U(L));
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t lo = c * kChunk, hi = std::min(nf, (c + 1) * kChunk);
            for (int l = 0; l < L; ++l) {
                const int64_t o = static_cast<int64_t>(l) * nf;
                const double* G = G_.data() + 3 * o;
                double sq = 0.0, sab = 0.0;
                for (int64_t i = lo; i < hi; ++i) {
                    double q = 0.0;
                    for (int e = 0; e < 3; ++e) {
                        const int64_t s = 3 * i + e;
                        if (G[s] != 0.0) q -= tau * G[s] * (x_[U(o + i)] - x_[U(o + adj.nbr[s])]);
                    }
                    sq += q; sab += std::fabs(q);
                    r_[U(o + i)] = D[o + i] + ((C_[U(o + i)] > 0.0) ? q / C_[U(o + i)] : 0.0);   // r_ reused as the RHS
                }
                pq[U(l) * U(nc) + U(c)] = sq; pa[U(l) * U(nc) + U(c)] = sab;
            }
            for (int64_t i = lo; i < hi; ++i) thomasColumn(A, B, Cc, r_.data(), T_out, i, cp, dp);
        }
    }
    for (int l = 0; l < L; ++l) {
        double a = 0.0, sab = 0.0;
        for (int64_t c = 0; c < nc; ++c) { a += pq[U(l) * U(nc) + U(c)]; sab += pa[U(l) * U(nc) + U(c)]; }
        if (sab > 0.0) st.closure_worst = std::max(st.closure_worst, std::fabs(a) / sab);
    }
    return st;
}

}  // namespace lateral
}  // namespace ptm
