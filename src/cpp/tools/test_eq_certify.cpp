// Gate for the RETROSPECTIVE equilibration certificate
// (thermal::eqCertifyColumn, src/cpp/thermal/eq_certificate.{h,cpp}).
//
// The certificate answers "did this column actually settle?", from the
// cycle-mean interface flux residual r_c = qbar_c - Qb converted to kelvin.
// The gates below are built as REDUCTIONS -- a demonstrably settled column and
// a demonstrably unsettled one, both constructed here from closed form rather
// than asserted -- plus the property the whole quantity exists for.
//
// GATES
//   1. REDUCTION, settled. The zero-forcing conductive steady profile is
//      closed form: eps*sigma*T0^4 = Qb, then T[j+1] = T[j] + Qb*dz[j]/k[j].
//      It is the only exactly-known state in the system. Its interface flux
//      must equal Qb at every interface, and the certificate must PASS it with
//      E far below the target.
//   2. REDUCTION, unsettled. The same column started isothermal has developed
//      no interior gradient at all, so q_c = 0 and r_c = -Qb exactly. The
//      certificate must FAIL it, and the kelvin figure must be the physically
//      right one, Qb / (4 eps sigma T^3).
//   3. ARREST IMMUNITY -- the reason this quantity and not another. A column
//      that is FROZEN (bit-identical from one pass to the next) but NOT
//      settled passes every increment-based convergence test by construction,
//      because the increment is exactly zero. The certificate must still FAIL
//      it. This is checked as a DISCRIMINATION: the increment test and the
//      certificate are both run on the same state and must disagree.
//   4. SIGN. C_below * dTbar/dt = Qb - qbar_c = -r_c. A column colder than
//      steady below the cut has q_c < Qb and must be reported WARMING. The
//      design note wrote this with the opposite sign, which relabels every
//      faithfully relaxing column as arrested; this gate is its falsifier.
//   5. CUT RULE. z_c = min(5*delta_ref, z_max/2); the clamp is reported when
//      it binds; the cut is never the bottom-BC identity interface (n-2),
//      which carries q == Qb by construction and would certify anything.
//   6. Qb IS READ OUT OF THE STATE, not the config. eqQbFromState() must
//      recover the Qb a column was built with, through the engine's own bottom
//      BC (k at the DEEPER node there, unlike the interior stencil).
//   7. HARMONIC vs ARITHMETIC mean of h. On a duty cycle the arithmetic mean
//      is optimistic by a large factor; the certificate must use the harmonic
//      mean when given it, and must BRACKET rather than substitute when not.
//   8. PROVENANCE. A snapshot of a forced column does not contain a cycle
//      mean, and must come back INDETERMINATE -- not PASS, not FAIL.
//
// CPU only. No GPU, no mesh, no simulation: safe to run while a production job
// owns the devices.
//
// Exit status: 0 = PASS. CI greps "[test_eq_certify] PASS".

#include "../thermal/eq_certificate.h"
#include "../thermal/thermal_model.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {

const double kSigma = ptm::kStefanBoltzmann;

// The shipped production lunar column: Moon, m=10 n=15 b=32 -- the grid the
// 50->100 lunation drift and the float32 arrest were both measured on. Built
// through thermal::Profile so this test exercises the ENGINE's grid, not a
// transcription of it (a transcription wrong by a few percent is
// indistinguishable from an energy-conservation defect; that ambiguity is why
// tools/dump_thermal_grid exists).
struct Column {
    thermal::Planet planet;
    thermal::Config config;
    std::vector<double> z, T, k, rho, cp;
    int n = 0;
    double Qb = 0.0, eps = 0.0;
};

Column makeProductionLunarColumn() {
    Column c;
    c.planet = thermal::Planet("Moon");
    c.config.m = 10;
    c.config.n = 15;
    c.config.b = 32;
    c.Qb  = c.planet.Qb;
    c.eps = c.planet.emissivity;

    thermal::Profile prof(c.planet, 0.0, c.config, /*verbose=*/false);
    c.n = prof.nlayers;
    c.z.assign(prof.z().begin(), prof.z().begin() + c.n);
    c.rho.assign(prof.rho().begin(), prof.rho().begin() + c.n);
    c.T.assign(c.n, 0.0);
    c.k.assign(c.n, 0.0);
    c.cp.assign(c.n, 0.0);
    return c;
}

// Fill k and cp for a given T, through the engine's own property model.
void refreshProperties(Column& c) {
    thermal::Profile prof(c.planet, 0.0, c.config, /*verbose=*/false);
    for (int i = 0; i < c.n; ++i) prof.T()[i] = c.T[i];
    prof.update_cp();
    prof.update_k();
    for (int i = 0; i < c.n; ++i) { c.k[i] = prof.k()[i]; c.cp[i] = prof.cp()[i]; }
}

// The bottom node is set by the engine's bottom BC, which uses k at the DEEPER
// node -- and k there depends on the node's own temperature, so the BC and the
// property model are mutually dependent. The solver reaches that fixed point
// by iterating; three passes here take the self-consistency residual to
// roundoff, which is what a converged state on disk actually looks like.
void settleBottomNode(Column& c);

// THE ONLY EXACTLY-KNOWN STATE IN THE SYSTEM. Zero forcing, so the surface
// balance is eps*sigma*T0^4 = Qb and the interior is a pure conductive ramp
// carrying exactly Qb. Explicit, no iteration: k at each step is evaluated at
// the SHALLOWER node, matching both the CN stencil and eqInterfaceFlux().
void fillZeroForcingSteady(Column& c) {
    const double R350 = c.config.getR350();
    thermal::Profile prof(c.planet, 0.0, c.config, /*verbose=*/false);
    const double T0 = std::pow(c.Qb / (c.eps * kSigma), 0.25);
    c.T[0] = T0;
    for (int i = 1; i < c.n; ++i) {
        const double kj = prof.kc()[i - 1] *
                          (1.0 + R350 * c.T[i - 1] * c.T[i - 1] * c.T[i - 1]);
        c.T[i] = c.T[i - 1] + c.Qb * (c.z[i] - c.z[i - 1]) / kj;
    }
    refreshProperties(c);
    settleBottomNode(c);
}

void settleBottomNode(Column& c) {
    for (int it = 0; it < 3; ++it) {
        c.T[c.n - 1] = c.T[c.n - 2] +
                       c.Qb * (c.z[c.n - 1] - c.z[c.n - 2]) / c.k[c.n - 1];
        refreshProperties(c);
    }
}

// delta_ref for a period P, on the grid's own terms: delta_0 = z_max/b, scaled
// by sqrt(P/P_diurnal). This is the caller's choice to make, and the test makes
// it the same way the retrospective tool does.
double deltaRefFor(const Column& c, double P_diurnal, double P_ref) {
    const double d0 = thermal::eqGridSkinDepth(c.z.data(), c.n, c.config.b);
    return d0 * std::sqrt(P_ref / P_diurnal);
}

thermal::EqCertifyInput baseInput(const Column& c, double P_diurnal, double P_ref) {
    thermal::EqCertifyInput in;
    in.z = c.z.data(); in.T = c.T.data(); in.k = c.k.data();
    in.rho = c.rho.data(); in.cp = c.cp.data();
    in.n = c.n;
    in.Qb = -1.0;                       // measured out of the state
    in.eps = c.eps;
    in.sigma = kSigma;
    in.P_ref = P_ref;
    in.delta_ref = deltaRefFor(c, P_diurnal, P_ref);
    in.target_K = 0.1;
    in.T_max = c.T[0];
    in.h_harmonic = 4.0 * c.eps * kSigma * c.T[0] * c.T[0] * c.T[0];
    in.no_periodic_forcing = true;      // zero forcing: the snapshot IS the mean
    return in;
}

bool near(double a, double b, double rel) {
    const double s = std::max(std::abs(a), std::abs(b));
    return s == 0.0 ? true : std::abs(a - b) / s <= rel;
}

}  // namespace

int main() {
    bool ok = true;
    const double P_lun = 29.530588 * 86400.0;      // synodic month [s]
    const double P_ann = 365.25 * 86400.0;

    std::printf("=== test_eq_certify: retrospective equilibration certificate ===\n\n");

    // -----------------------------------------------------------------------
    // Gate 6 first: it underpins every other gate's Qb.
    // -----------------------------------------------------------------------
    Column settled = makeProductionLunarColumn();
    fillZeroForcingSteady(settled);
    {
        const double qb = thermal::eqQbFromState(settled.z.data(), settled.T.data(),
                                                 settled.k.data(), settled.n);
        std::printf("Gate 6  Qb read out of the state's own bottom BC\n");
        std::printf("  built with Qb = %.10g W/m^2   recovered %.10g   rel %.2e\n",
                    settled.Qb, qb, std::abs(qb / settled.Qb - 1.0));
        if (!near(qb, settled.Qb, 1e-12)) { ok = false; std::printf("  -> FAIL\n"); }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 1: REDUCTION, settled column certifies.
    // -----------------------------------------------------------------------
    {
        std::vector<double> q(settled.n - 1);
        const bool okf = thermal::eqInterfaceFlux(settled.z.data(), settled.T.data(),
                                                  settled.k.data(), settled.n, q.data());
        double worst = 0.0;
        for (int j = 0; j < settled.n - 2; ++j)     // n-2 is the bottom-BC identity
            worst = std::max(worst, std::abs(q[j] / settled.Qb - 1.0));

        thermal::EqCertifyResult r =
            thermal::eqCertifyColumn(baseInput(settled, P_lun, P_ann));

        std::printf("Gate 1  REDUCTION -- a demonstrably SETTLED column\n");
        std::printf("  zero-forcing conductive steady profile, T0 = %.6f K\n",
                    settled.T[0]);
        std::printf("  interface flux vs Qb, worst of %d interfaces: rel %.2e\n",
                    settled.n - 2, worst);
        std::printf("  cut j=%d  z=%.4f m (%.2f skin depths%s)\n",
                    r.cut.j, r.cut.z, r.cut.n_delta, r.cut.clamped ? ", CLAMPED" : "");
        std::printf("  r_c = %.3e W/m^2   E_max = %.3e K   verdict %s\n",
                    r.r_cut, r.E_max_hi, thermal::eqVerdictName(r.v_max));
        if (!okf || !(worst < 1e-12)) { ok = false; std::printf("  -> FAIL: not a steady state\n"); }
        if (!r.valid || r.v_max != thermal::EqVerdict::Pass ||
            r.v_mean != thermal::EqVerdict::Pass ||
            !(r.E_max_hi < 1e-6)) {
            ok = false; std::printf("  -> FAIL: a settled column must certify\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 2: REDUCTION, unsettled column does not certify -- and Gate 4, the
    // sign, on the same state (it is colder than steady below the cut).
    // -----------------------------------------------------------------------
    Column unsettled = makeProductionLunarColumn();
    {
        for (int i = 0; i < unsettled.n; ++i) unsettled.T[i] = settled.T[0];
        refreshProperties(unsettled);
        // Bottom node still obeys the engine's bottom BC, so Qb is recoverable.
        settleBottomNode(unsettled);

        thermal::EqCertifyResult r =
            thermal::eqCertifyColumn(baseInput(unsettled, P_lun, P_ann));
        const double h = 4.0 * unsettled.eps * kSigma *
                         unsettled.T[0] * unsettled.T[0] * unsettled.T[0];
        // Scored against the Qb the STATE carries, which is the whole point of
        // eqQbFromState -- so the closed form uses that Qb, not the config's.
        const double E_expected = r.Qb_used / h;

        std::printf("Gate 2  REDUCTION -- a demonstrably UNSETTLED column\n");
        std::printf("  isothermal at %.4f K: no interior gradient has developed\n",
                    unsettled.T[0]);
        std::printf("  Qb in the state %.10g W/m^2 (built with %.10g)\n",
                    r.Qb_used, unsettled.Qb);
        std::printf("  r_c = %.6e W/m^2  (exactly -Qb)\n", r.r_cut);
        std::printf("  E_max = %.4f K   closed form Qb/(4 eps sigma T^3) = %.4f K\n",
                    r.E_max_hi, E_expected);
        std::printf("  verdict %s\n", thermal::eqVerdictName(r.v_max));
        if (!r.valid || r.v_max != thermal::EqVerdict::Fail) {
            ok = false; std::printf("  -> FAIL: an unsettled column must NOT certify\n");
        }
        if (!near(r.r_cut, -r.Qb_used, 1e-14) || !near(r.E_max_hi, E_expected, 1e-12)) {
            ok = false; std::printf("  -> FAIL: kelvin conversion is not the closed form\n");
        }
        if (!near(r.Qb_used, unsettled.Qb, 1e-12)) {
            ok = false; std::printf("  -> FAIL: the state does not carry the Qb it was built with\n");
        }

        std::printf("Gate 4  SIGN -- C dTbar/dt = Qb - qbar = -r\n");
        std::printf("  q_c < Qb, so the sub-column is starved and must be WARMING\n");
        std::printf("  dTdt_below = %+.3e K/s  (%+.3f K per lunation)\n",
                    r.dTdt_below, r.dTdt_below * P_lun);
        if (!(r.dTdt_below > 0.0)) {
            ok = false;
            std::printf("  -> FAIL: sign inverted (this is the NOTE 11.4 defect)\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 3: ARREST IMMUNITY, as a discrimination.
    // -----------------------------------------------------------------------
    {
        // "Two consecutive passes" of an arrested column: the SAME state twice.
        // That is not an artificial construction -- it is literally what the
        // banked 8-pass moon3d state does, bit-identical across 100 further
        // lunations on 100.0000% of the dark faces.
        double max_increment = 0.0;
        for (int i = 0; i < unsettled.n; ++i) {
            const float a = static_cast<float>(unsettled.T[i]);
            const float b = static_cast<float>(unsettled.T[i]);   // frozen
            max_increment = std::max(max_increment,
                                     std::abs(static_cast<double>(b - a)));
        }
        const bool increment_test_says_converged = (max_increment == 0.0);

        thermal::EqCertifyResult r =
            thermal::eqCertifyColumn(baseInput(unsettled, P_lun, P_ann));
        const bool certificate_says_converged = (r.v_max == thermal::EqVerdict::Pass);

        std::printf("Gate 3  ARREST IMMUNITY -- the reason for this quantity\n");
        std::printf("  a FROZEN but UNSETTLED column, scored two ways:\n");
        std::printf("    max pass-to-pass increment : %.3e K  -> increment test says %s\n",
                    max_increment,
                    increment_test_says_converged ? "CONVERGED" : "not converged");
        std::printf("    certificate                : E = %.4f K -> says %s\n",
                    r.E_max_hi, thermal::eqVerdictName(r.v_max));
        if (!increment_test_says_converged) {
            ok = false;
            std::printf("  -> FAIL: the discrimination is vacuous unless the "
                        "increment test passes\n");
        }
        if (certificate_says_converged) {
            ok = false;
            std::printf("  -> FAIL: the certificate was fooled by the arrest\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 5: the cut rule.
    // -----------------------------------------------------------------------
    {
        const double z_max = settled.z[settled.n - 1];
        // The grid's OWN delta_0 = z_max/b, scaled by sqrt(P/P_diurnal).
        const double d_diu  = deltaRefFor(settled, P_lun, P_lun);
        const double d_ann  = deltaRefFor(settled, P_lun, P_ann);
        // The state's surface kappa gives a different answer -- printed so the
        // choice is visible rather than assumed. cp(24 K) is ~9x below the
        // cp(200 K) the grid was built from, so this runs ~3x deeper.
        const double kappa0 = settled.k[0] / (settled.rho[0] * settled.cp[0]);
        std::printf("Gate 5  (for the record) grid delta_diurnal %.6f m vs "
                    "state-surface %.6f m\n", d_diu,
                    std::sqrt(kappa0 * P_lun / M_PI));

        thermal::EqCut ca = thermal::eqChooseCut(settled.z.data(), settled.n, d_ann);
        thermal::EqCut cd = thermal::eqChooseCut(settled.z.data(), settled.n, d_diu);

        std::printf("Gate 5  CUT RULE  z_c = min(5*delta_ref, z_max/2)\n");
        std::printf("  z_max = %.6f m   delta_diurnal = %.6f m   delta_annual = %.6f m\n",
                    z_max, d_diu, d_ann);
        std::printf("  annual : target %.4f m -> j=%d z=%.4f m (%.2f delta)%s\n",
                    ca.z_target, ca.j, ca.z, ca.n_delta, ca.clamped ? "  CLAMPED" : "");
        std::printf("  diurnal: target %.4f m -> j=%d z=%.4f m (%.2f delta)%s\n",
                    cd.z_target, cd.j, cd.z, cd.n_delta, cd.clamped ? "  CLAMPED" : "");
        // On this grid 5*delta_annual = 0.552 m exceeds z_max/2 = 0.502 m, so
        // the clamp MUST bind -- and it must say so. The diurnal cut is deep
        // inside the column and must NOT report a clamp.
        if (!(ca.clamped && !cd.clamped)) {
            ok = false; std::printf("  -> FAIL: clamp reporting is wrong\n");
        }
        if (!(ca.z_target <= 0.5 * z_max + 1e-12 && ca.z_target <= 5.0 * d_ann + 1e-12)) {
            ok = false; std::printf("  -> FAIL: target is not min(5 delta, z_max/2)\n");
        }
        if (ca.j >= settled.n - 2 || cd.j >= settled.n - 2) {
            ok = false;
            std::printf("  -> FAIL: cut landed on the bottom-BC identity interface\n");
        }
        // A pathologically deep reference period must clamp and stay legal.
        thermal::EqCut cx = thermal::eqChooseCut(settled.z.data(), settled.n, 1e6);
        if (!(cx.j >= 1 && cx.j < settled.n - 2 && cx.clamped)) {
            ok = false; std::printf("  -> FAIL: cut is not bounded for a huge delta_ref\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 7: harmonic vs arithmetic mean of h.
    // -----------------------------------------------------------------------
    {
        // A duty cycle: hot for a quarter of the cycle, cold for three
        // quarters. This is the lunar situation in miniature -- the surface is
        // radiatively stiff when it is hot and slack when it is cold, and a
        // flux residual displaces it most while it is slack.
        const double eps = settled.eps;
        const double T_hot = 350.0, T_cold = 100.0, f_hot = 0.25;
        const double h_hot  = 4.0 * eps * kSigma * T_hot  * T_hot  * T_hot;
        const double h_cold = 4.0 * eps * kSigma * T_cold * T_cold * T_cold;
        const double h_arith = f_hot * h_hot + (1.0 - f_hot) * h_cold;
        const double h_harm  = 1.0 / (f_hot / h_hot + (1.0 - f_hot) / h_cold);

        thermal::EqCertifyInput in = baseInput(unsettled, P_lun, P_ann);
        in.T_max = T_hot;
        in.h_harmonic = h_harm;
        thermal::EqCertifyResult rh = thermal::eqCertifyColumn(in);

        in.h_harmonic = h_arith;                 // the wrong choice, for scale
        thermal::EqCertifyResult ra = thermal::eqCertifyColumn(in);

        in.h_harmonic = 0.0;                     // unknown: bracket it
        in.h_lower = h_cold;                     // min over the cycle
        in.h_upper = h_hot;                      // max over the cycle
        thermal::EqCertifyResult rb = thermal::eqCertifyColumn(in);

        std::printf("Gate 7  HARMONIC vs ARITHMETIC cycle mean of h\n");
        std::printf("  duty cycle %.0f%% at %.0f K, %.0f%% at %.0f K\n",
                    100 * f_hot, T_hot, 100 * (1 - f_hot), T_cold);
        std::printf("  E_mean(harmonic)   = %.4f K\n", rh.E_mean_hi);
        std::printf("  E_mean(arithmetic) = %.4f K   -> optimistic by %.2fx\n",
                    ra.E_mean_hi, rh.E_mean_hi / ra.E_mean_hi);
        std::printf("  bracket when h is unknown: [%.4f, %.4f] K, verdict %s\n",
                    rb.E_mean_lo, rb.E_mean_hi, thermal::eqVerdictName(rb.v_mean));
        if (!(rh.E_mean_hi > ra.E_mean_hi * 1.5)) {
            ok = false;
            std::printf("  -> FAIL: the two means are not being distinguished\n");
        }
        if (!(rb.E_mean_lo <= rh.E_mean_hi && rh.E_mean_hi <= rb.E_mean_hi)) {
            ok = false;
            std::printf("  -> FAIL: the bracket does not contain the harmonic answer\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Gate 8: provenance of qbar.
    // -----------------------------------------------------------------------
    {
        thermal::EqCertifyInput in = baseInput(settled, P_lun, P_ann);
        in.no_periodic_forcing = false;      // a forced face...
        in.q_is_cycle_mean     = false;      // ...scored from ONE snapshot
        thermal::EqCertifyResult snap = thermal::eqCertifyColumn(in);

        in.q_phase_bound = 1e-6;             // a measured bound, W/m^2
        thermal::EqCertifyResult bounded = thermal::eqCertifyColumn(in);

        in.q_phase_bound = 0.0;
        in.q_is_cycle_mean = true;           // a true cycle mean
        thermal::EqCertifyResult mean = thermal::eqCertifyColumn(in);

        std::printf("Gate 8  PROVENANCE of the cycle mean\n");
        std::printf("  forced face, one snapshot       -> %s\n",
                    thermal::eqVerdictName(snap.v_max));
        std::printf("  forced face, measured bound     -> %s\n",
                    thermal::eqVerdictName(bounded.v_max));
        std::printf("  true cycle mean                 -> %s\n",
                    thermal::eqVerdictName(mean.v_max));
        if (snap.v_max != thermal::EqVerdict::Indeterminate) {
            ok = false;
            std::printf("  -> FAIL: a snapshot of a forced column is not a cycle mean\n");
        }
        if (bounded.v_max != thermal::EqVerdict::Pass ||
            mean.v_max != thermal::EqVerdict::Pass) {
            ok = false;
            std::printf("  -> FAIL: a settled column must certify once the mean is known\n");
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // The banner a user actually sees.
    // -----------------------------------------------------------------------
    {
        thermal::EqCertifyCensus census(0.1, P_ann);
        thermal::EqCertifyResult s = thermal::eqCertifyColumn(baseInput(settled, P_lun, P_ann));
        thermal::EqCertifyResult u = thermal::eqCertifyColumn(baseInput(unsettled, P_lun, P_ann));
        for (int i = 0; i < 700; ++i) census.add(s, /*zero_forcing=*/true);
        for (int i = 0; i < 300; ++i) census.add(u, /*zero_forcing=*/false);
        census.report(std::cout, /*cycles_used=*/50.0);
    }

    if (!ok) { std::printf("\n[test_eq_certify] FAIL\n"); return 1; }
    std::printf("\n[test_eq_certify] PASS\n");
    return 0;
}
