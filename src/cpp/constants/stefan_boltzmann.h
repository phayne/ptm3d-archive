#ifndef PTM_STEFAN_BOLTZMANN_H
#define PTM_STEFAN_BOLTZMANN_H

// ---------------------------------------------------------------------------
// THE Stefan-Boltzmann constant. One definition for the whole engine.
//
//     sigma = 2 pi^5 k_B^4 / (15 h^3 c^2)
//
// Since the 2019 SI redefinition k_B, h and c are exact by definition, so
// sigma is exact too: 5.670374419...e-8 W m^-2 K^-4. It is not a measured
// quantity with an uncertainty, and there is no defensible reason to carry a
// second value.
//
// Before 2026-08-11 FIVE values shipped simultaneously:
//   5.67051196e-8   (CODATA 1986) -- the thermal solver, radiosity, every GPU
//                   kernel; 24 sites
//   5.670374419e-8  -- constants.cpp, the camera, anisothermal beaming
//   5.670374e-8     -- bowl_crater_ptm.cpp (the Ingersoll VALIDATION gate, a
//                   value the engine itself never used) and vf_mc_flux
//   5.67e-8         -- thermal_config.sigma for every mesh run, i.e. the value
//                   the surface energy balance actually ran with
//   5.6704e-8       -- a since-removed Python research module
//
// Two of those were live inconsistencies inside one routine:
// Profile::update_T computed T_geo with a 5.670374419e-8 fallback while the
// surface Newton in the same call used Config::sigma, and the mesh path set
// Config::sigma to 5.67e-8 -- so the surface balance and the radiosity that
// feeds it did not conserve energy with each other by construction, at
// 2.4256e-5 relative. In temperature that is dT/T = -6.064e-6:
// 0.15 mK at 24 K, 0.24 mK at 40 K, 0.61 mK at 100 K, 2.37 mK at 390 K.
// Bookkeeping in size, but free to remove and impossible to defend.
//
// OpenCL kernels cannot include this header. Their kernel text carries its own
// `#define SIGMA`, and the host static_asserts the two agree -- see
// gpu/thermal_opencl.cpp and gpu/radiosity_opencl.cpp. Do NOT copy the literal
// anywhere else.
// ---------------------------------------------------------------------------
namespace ptm {

inline constexpr double kStefanBoltzmann = 5.670374419e-8;  // W m^-2 K^-4

}  // namespace ptm

#endif  // PTM_STEFAN_BOLTZMANN_H
