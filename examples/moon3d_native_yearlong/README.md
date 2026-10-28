# Moon3D year-long simulation (configurations from the Moon3D revision)

The two configurations here are the settings of the Moon3D paper's year-long runs (JGR Planets,
revised version, Section 4.6): one full year (2030-09-01 to 2031-09-01, 1,484 output steps) of
the 509,337-face Moon3D testbed crater scan placed at the IM-2 landing site (84.79°S, 29.2°E).

| File | Radiation exchange | Purpose |
|---|---|---|
| `moon3d_native_yearlong_coupled.yaml` | on (two-band radiosity) | the science run |
| `moon3d_native_yearlong_radoff.yaml` | off | the comparison run; otherwise identical |

Settings worth knowing: an 8-cycle spin-up with the default indirect-aware seed, the CPU solver
(`thermal.use_gpu: false`), `thermal_b: 32` (a deeper grid for the annual wave), a basal heat
flow of 0.022 W/m² (the engine's lunar default is 0.018 W/m²), the Hapke incidence-dependent
albedo, no sub-facet roughness (this mesh's ~3 mm facets already resolve it), and the validated
fine-mesh illumination settings in an `engine_env:` block.

## Inputs you must supply

Neither input is in the repository or in the Zenodo data archive.

1. **The mesh**, `data/meshes/moon3d_c_cartesian_NEW_500k_clean_native.ply` (509,337 faces,
   about 23 MB). It is resolved against `$PTM_DATA_ROOT/meshes/`, then `<repo>/data/meshes/`.
   The fetchable dataset `moon3d-mesh` is the undecimated 1.7 M-face scan, not this file.
2. **The view-factor matrix.** The revision's runs replaced the engine's view factors with a
   precomputed sky-side matrix from the offline Monte Carlo referee, loaded through the
   environment variable `PTM_HMATRIX_VF_REPLACE=/path/to/matrix.csr` (see
   [docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md)). The engine refuses a matrix that does not
   match the mesh or that was traced on the other side of the surface. The config's
   `view_factor_file` (resolved against this directory) names the placeholder the engine loads
   before the replacement. **If neither file is present the engine builds the view factors
   itself**, which on this mesh takes hours and a very large amount of memory; confirm that the
   log says `Loaded view factors from:` before leaving the run.

## Running

On a machine with enough memory (the coupled run peaked at about 142 GB):

```bash
export PTM_HMATRIX_VF_REPLACE=/path/to/moon3d_509k_sky_pooled.csr
./build/bin/ptm_thermal_sim --config examples/moon3d_native_yearlong/moon3d_native_yearlong_coupled.yaml
```

The revision's coupled run took about 15 hours on 48 cores. Outputs go to
`output/moon3d_native_yearlong/{coupled,radoff}`.
