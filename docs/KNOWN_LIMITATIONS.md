# Known limitations

What the model does not do, or does only approximately, in this release. Read this before a
production run. Fixed problems are listed in [CHANGELOG.md](../CHANGELOG.md), not here.

## Physics and numerics

1. **No lateral heat conduction.** Each face is an independent 1D column. Where facets are
   smaller than a few centimeters (the Moon3D testbed mesh has ~3 mm facets), small cold traps
   are warmed by their neighbors in reality but not in the model; for a pit with
   depth-to-diameter 0.1 at 85°S, the 110 K cold trap disappears below about 5 cm diameter.

2. **Spin-up length is your choice.** `thermal.equilibration_orbits` (default 1) sets how many
   diurnal cycles are integrated before the output window. Surface temperatures settle in a few
   cycles; the deep column relaxes over roughly 15 to 23 lunations on the Moon and about 10
   orbits on Europa. The run prints the number of cycles it estimates you need and warns when
   the setting is shorter (the estimate is conservative); for any quantity you report, run N and
   2N cycles and compare.

3. **Radiation exchange during spin-up is approximate.** In a coupled run the spin-up is forced
   with an estimate of the indirect (scattered and re-emitted) flux taken from the first
   radiation solution, folded over whole diurnal cycles. On the shipped 80°N crater a one-cycle spin-up
   then lies within about 0.4 K (median) of a 40-cycle reference, for lit and never-lit faces
   alike. Where this seed cannot act (a warm-started chunk, the CPU scalar solver, an output
   window shorter than one diurnal cycle) the run says so and uses the previous seed, and
   absolute temperatures are uncertain by about 1 to 2 K. The difference between coupled and
   uncoupled runs is more robust than either absolute value.

4. **The column solver is first order in time.** The surface temperature is solved from the
   previous substep's subsurface state, so the error falls linearly with the substep. At the
   default settings the bias in time-mean temperature is at most about 0.1 K; it is larger for
   peak temperatures at coarse output cadence.

5. **CPU and GPU solvers are not equivalent.** The GPU holds each output interval's absorbed
   flux constant across its substeps, which raises lit-face peak temperatures by about 0.4 K
   (median) and a few kelvin in the tail at daily cadence, and more at coarser cadence; it also
   caps the substep count at 200. Use the CPU solver for accuracy-critical work, and never mix
   backends within one comparison.

6. **No temperature below the geothermal floor.** Every layer is bounded below by
   T = (Q_b/εσ)^(1/4): 24.04 K on the Moon at the default Q_b = 0.018 W/m² (25.28 K at
   0.022 W/m²). A value at the floor is the bound, not a solution.

7. **View factors follow the engine's face-orientation rule.** By default each face normal is
   flipped radially outward; with `PTM_ILLUM_TRUE_NORMALS=1` the triangle winding is kept, with
   one sign per connected surface. A wrong side is not caught by the usual checks: a view-factor
   matrix traced on the underside still passes reciprocity, row-sum and nonzero tests. The
   offline referee applies the same rule and refuses a mesh it would reverse on more than 1% of
   faces. A precomputed matrix loaded with `PTM_HMATRIX_VF_REPLACE` is refused if its recorded
   orientation differs from the run's; a matrix with no orientation record is treated as raw
   winding. Reuse a view-factor file only with the exact mesh it was built from. On closed,
   non-convex bodies the radial rule can choose the wrong side of true overhangs.

8. **Fine-mesh illumination corrections are opt-in.** By default the shadow ray starts about
   1 cm above the face and normals are flipped radially outward, which can light
   millimeter-scale shadowed floors and create false sunlit overhangs. On fine polar meshes,
   enable the full validated set (`PTM_ILLUM_TRUE_NORMALS=1`, `PTM_ILLUM_RISE_EPS=1`,
   `PTM_ILLUM_RISE_KEPS=0.2`, `PTM_ILLUM_RISE_HABS=0`, `PTM_ILLUM_SEAL_HOLES=1`), never
   `PTM_ILLUM_RISE_EPS` alone; see [ENVIRONMENT.md](ENVIRONMENT.md).

9. **Sub-facet roughness (beaming) is a scale choice.** On meter-scale and coarser polar meshes
   beaming is needed to match Diviner (at Haworth it moves the cold-trap summer-maximum bias from
   −10.3 K to −0.8 K median); on meshes whose facets already resolve the roughness (below 5 cm)
   it would count the same physics twice, and the engine refuses it.

## Configuration

- **Very small generic-body properties are lost.** A `generic_body.surface` value below 5e-7 is
  rounded to zero and replaced by a placeholder without warning. Keep such values at or above
  5e-7, or set them with the `*_override` keys, which keep full precision.
- **Plain property keys are ignored on the Moon, Mercury and Europa** (the run warns); use the
  `thermophysical.*_override` keys. A per-face albedo map overrides `albedo_override` for
  absorbed sunlight.
- **Unknown keys and command-line flags are reported but do not stop the run.** Check the
  warnings, or run `ptm_thermal_sim --config FILE --validate-config` first.
- **`mesh.scale_factor` does not work**; the engine refuses such a mesh. Rescale the mesh file.

## Platforms

- **GPU:** only OpenCL on macOS with AMD GPUs has been exercised; the HIP and CUDA backends are
  untested. Long GPU runs on macOS with AMD hardware can hang the machine; see
  [dev/gpu-wedge-guardrails.md](dev/gpu-wedge-guardrails.md).
- **Windows:** use WSL2 and the Linux instructions.
