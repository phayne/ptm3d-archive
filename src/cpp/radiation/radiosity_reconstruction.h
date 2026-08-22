// radiosity_reconstruction.h
// ---------------------------------------------------------------------------
// Host-side reconstruction of the terrain's OUTGOING radiosity B from the
// quantities the engine already publishes.
//
// WHY THIS FILE EXISTS AT ALL
// ---------------------------
// The converged radiosity vectors B_ir / B_vis are function-local inside
// RadiositySolver::solveTwoBand (radiation_exchange.cpp:1691) and are discarded
// when it returns; on the multi-GPU pool route B never leaves the device. Only
// the ABSORBED band fluxes escape, via lastIndirect{IR,Visible}() and their
// batch siblings.
//
// A passive probe (a flux cube) needs B, not the absorbed flux. Rather than add
// a device->host readback to a kernel path that is currently bit-identical to
// the CPU, the host inverts the solver's own two closing identities. The
// inversion is EXACT, not approximate: solveTwoBand computes
//
//     Q_ind_ir [i] = eps_i     * (B_ir [i] - emission  [i]) / (1 - eps_i)
//     Q_ind_vis[i] = (1 - A_i) * (B_vis[i] - vis_source[i]) /      A_i
//
// (radiation_exchange.cpp:1726-1742, with emission = eps*sigma*T^4,
// vis_source = A_src*Q_dir, reflectivity_ = 1 - emissivity_), so
//
//     B_ir [i] = eps_i*sigma*T_i^4 + (1 - eps_i) * ( Q_ind_ir [i] / eps_i    )
//     B_vis[i] = A_src_i * Q_dir_i +      A_i    * ( Q_ind_vis[i] / (1 - A_i) )
//
// which is what this header spells, ONCE. Both the flux-cube post-solve pass
// and test_flux_cube_identity call these functions; neither writes the algebra
// out again. That is deliberate -- PTM-106c is the defect class where one
// formula acquires two spellings that agree today and drift later.
//
// GUARDS
// ------
// The inversion divides by eps_i and by (1 - A_i). Where either denominator
// collapses the identity is ill-posed and the SOLVER ITSELF silently switches
// to a fallback (last_jacobi_incoming_, guarded by REFL_GUARD = 1e-9) that the
// host cannot observe or reproduce. So this header REFUSES rather than
// approximating: checkReconstructible() returns a reason string and the caller
// is expected to make that a hard error. Never a quiet substitution.
//
// UNITS: everything is W/m^2 except temperature (K). B is OUTGOING radiosity
// (emitted + reflected), i.e. exactly what a passive patch's view-factor row
// should be dotted against.
// ---------------------------------------------------------------------------
#ifndef PTM_RADIOSITY_RECONSTRUCTION_H
#define PTM_RADIOSITY_RECONSTRUCTION_H

#include <cstddef>
#include <string>
#include <vector>

namespace radiation {

// Stefan-Boltzmann, the same constant the solver's computeThermalEmission uses.
constexpr double kReconSigmaSB = 5.670374419e-8;

// Denominator floors. A face outside these cannot be reconstructed, because the
// solver's own REFL_GUARD fallback becomes active and is invisible from here.
constexpr double kReconMinEmissivity = 1e-3;   // eps_i must exceed this
constexpr double kReconMaxAlbedo     = 1.0 - 1e-3;  // A_i must stay below this

// Thermal-IR outgoing radiosity of one face.
//   eps        emissivity of the face (operative value; PTM_RAD_EMISSIVITY aware)
//   T          surface temperature [K]
//   q_ind_ir   absorbed indirect IR flux [W/m^2] (lastIndirectIR()[i])
inline double reconstructBIR(double eps, double T, double q_ind_ir) {
    const double T2 = T * T;
    const double emission = eps * kReconSigmaSB * T2 * T2;
    return emission + (1.0 - eps) * (q_ind_ir / eps);
}

// Visible (solar) outgoing radiosity of one face.
//   a_src      PTM-106 incidence-dependent SOURCE albedo actually used this step
//   q_dir      direct-beam solar flux on the face [W/m^2]
//   a_bond     hemispheric/Bond albedo = the diffuse OPERATOR reflectance
//   q_ind_vis  absorbed indirect visible flux [W/m^2] (lastIndirectVisible()[i])
// NOTE the two albedos are different physical quantities and are NOT
// interchangeable (radiation_exchange.h:320-340). Passing a_bond for a_src is
// the PTM-106 defect, not a simplification.
inline double reconstructBVis(double a_src, double q_dir,
                              double a_bond, double q_ind_vis) {
    return a_src * q_dir + a_bond * (q_ind_vis / (1.0 - a_bond));
}

// Whether every face can be reconstructed. Returns true when it can; otherwise
// returns false and fills `reason` with the offending face and value. The
// caller must treat false as a hard error -- reconstructing anyway would divide
// by a collapsing denominator, and falling back silently is the failure mode
// this whole file is arranged to avoid.
inline bool checkReconstructible(const std::vector<double>& emissivity,
                                 const std::vector<double>& bond_albedo,
                                 std::string& reason) {
    for (size_t i = 0; i < emissivity.size(); ++i) {
        if (!(emissivity[i] > kReconMinEmissivity)) {
            reason = "face " + std::to_string(i) + " has emissivity " +
                     std::to_string(emissivity[i]) + " <= " +
                     std::to_string(kReconMinEmissivity) +
                     "; the IR radiosity inversion divides by emissivity and the "
                     "solver's own REFL_GUARD fallback is not observable from the host";
            return false;
        }
    }
    for (size_t i = 0; i < bond_albedo.size(); ++i) {
        if (!(bond_albedo[i] < kReconMaxAlbedo)) {
            reason = "face " + std::to_string(i) + " has Bond albedo " +
                     std::to_string(bond_albedo[i]) + " >= " +
                     std::to_string(kReconMaxAlbedo) +
                     "; the visible radiosity inversion divides by (1 - albedo)";
            return false;
        }
    }
    reason.clear();
    return true;
}

}  // namespace radiation

#endif  // PTM_RADIOSITY_RECONSTRUCTION_H
