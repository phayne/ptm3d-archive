# Moon3D native year-long campaign — base configs

The two YAML files here are the **base configs for the Moon3D year-long (annual) lunar
campaign**: one full 365.25-day simulation of the 509k-face `moon3d` polar mesh with
two-band radiosity coupling, and its illumination-only companion.

| File | Radiation | Purpose |
|---|---|---|
| `moon3d_native_yearlong_coupled.yaml` | `radiation.enabled: true` (band-separated radiosity, compiled default); `radiation.beaming: false` **explicitly, and it must stay off — see below** | The science run. Produces the annual T_max map, cold-trap area, and volatile-stability figures. |
| `moon3d_native_yearlong_radoff.yaml` | `radiation.enabled: false` | The no-coupling comparison pair. Identical in every other value. |

Both are derived from `examples/moon3d_500k_clean.yaml` (2026-07-12): only the mesh,
the output directory and the annual `duration_days`/`timesteps`/`thermal_b` settings
differ. `thermal_b: 32` carries its own derivation in a comment in the file — read it
before changing it.

## Why these files are here

Until 2026-08-15 they existed **only in one working tree on one machine**, untracked,
with absolute paths baked in. That is exactly how the `moon3d_lat_final` dataset was
lost and had to be re-run. Tracking them is the fix.

Sanitizing them changed **only path-valued keys** — every physical, numerical and
scheduling value is byte-for-byte what the campaign ran.

## ⛔ 2026-08-25: beaming stays OFF on this mesh, and it is a physics rule

`moon3d_native_yearlong_coupled.yaml` carries an explicit `radiation.beaming: false`.
That is not a leftover default — it is the answer to a question this directory will keep
being asked, now that Smith (1967) beamed emission is standard on lunar polar **site**
configs.

Beaming is a **sub-facet** model. `chi` stands in for roughness the mesh does not
resolve, at the scale where surface elements become thermally isolated from one another —
millimeters to centimeters on lunar regolith. **This mesh's facets are ~2 mm**: 509,337
faces over 1.96 m² of total area, a simulated crater about 1.6 m across. It already
resolves the structure `chi` is a stand-in for.

> **Paul's ruling, 2026-08-25.** "The Moon3D mesh resolution is comparable to the
> roughness scale (millimeters to centimeters) that the beaming is intended to capture,
> so the beaming should be OFF for small local meshes like this."

Turning it on here would count the same physics twice. Beaming belongs on **site and
regional** meshes — meters and up (`sunlit_radcore_r50` at 237 m, `haworth_radcore_r32c`
at 181 m, an A3GT 3 km core at 3.46 m). The criterion, the measured facet scales, and
the validated result are in
[`docs/radiation-exchange.md` §8.4](../../docs/radiation-exchange.md).

Because beaming is off, **every physical, numerical and scheduling value here still
matches the campaign that produced the shipped `moon3d_native_yearlong` dataset** and its
annual T_max map, cold-trap area and volatile-stability figures.

`moon3d_native_yearlong_radoff.yaml` needs no such key: it has `radiation.enabled: false`,
so beaming has nothing to act on there.

## The three site-local inputs you must supply

None of these is in the repository, and none can be fetched with `scripts/fetch_data.py`.

### 1. The mesh — `mesh.import_file: data/meshes/moon3d_c_cartesian_NEW_500k_clean.vtk`

A 509,337-face legacy-VTK mesh (~32 MB). Resolution follows the `PTM_DATA_ROOT`
contract in `docs/ENVIRONMENT.md`: an absolute path wins, then `$PTM_DATA_ROOT/meshes/`,
then `<repo>/data/meshes/`. **Symlink** it rather than copying — `data/meshes/` is only
partly covered by `.gitignore`, and a 32 MB blob committed by accident is a problem:

```sh
ln -s /path/to/moon3d/meshes/moon3d_c_cartesian_NEW_500k_clean.vtk data/meshes/
```

> ⚠️ If the file is missing the engine stops with a clear error naming
> `mesh.import_file` — but its suggested remedy,
> `scripts/fetch_data.py --dataset moon3d-mesh`, is **wrong for this file**. That
> dataset delivers `moon3d_cartesian_double.ply`, a different mesh. The hint comes from
> a prefix match on `data/meshes/moon3d` in `config_parser.cpp`; it is not a claim that
> this particular VTK is downloadable. It is not.

> Despite the name, this mesh is **not a global Moon**. It is a simulated polar crater
> roughly 1.8 m across at 84.8°S with ~2.5 mm facets — an *open* patch, not a closed body.

### 2. The view-factor matrix — `radiation.view_factor_file: view_factors/…vf`

Coupled run only (`radiation.enabled: false` makes it inert in the radoff config).

**A relative `view_factor_file` resolves against this YAML's own directory**, not the
repository root and not the working directory — `radiation.view_factor_file` is the one
path key that does *not* go through the `PTM_DATA_ROOT` contract
(`src/cpp/tools/config_parser.cpp`, and again in `ptm_thermal_sim.cpp`). So the path
above means `examples/moon3d_native_yearlong/view_factors/`.

```sh
mkdir -p examples/moon3d_native_yearlong/view_factors
ln -s /path/to/moon3d/vf/moon3d_c_cartesian_NEW_500k_clean.vf \
      examples/moon3d_native_yearlong/view_factors/
```

> ⛔ **If that file is absent the engine does not stop — it BUILDS the matrix.** For this
> mesh that is the documented GPU-wedge trigger (`BUGS.md` PTM-19): never rebuild a
> `moon3d` `.vf`. The stored matrix is ~25 GB and takes hours; a rebuilt one is also not
> the matrix the campaign used. Confirm the run's log says `Loaded view factors from:`
> and never `Computing view factors for` / `Building H-matrix view factor for`.
> `scripts/run_yearlong_wedge_safe.py` enforces exactly that check for you.

The `view_factors/` directory is gitignored here so a stray copy cannot be committed.

### 3. The corrected view-factor matrix — `PTM_HMATRIX_VF_REPLACE` (environment, not config)

The campaign ran with the **corrected per-partner-replacement CSR**, supplied through the
environment because no config key exists for it:

```sh
export PTM_HMATRIX_VF_REPLACE=/path/to/moon3d/vf/moon3d_c_cartesian_NEW_500k_clean_corrected_pp.csr
```

Documented in `docs/ENVIRONMENT.md`. Since 2026-08-16 the operator the replacement lands in
follows `radiation.use_gpu`, so this one variable is enough; the campaign as originally run
also exported `PTM_HMATRIX_VF_REPLACE_GPU=1`, which is now redundant but still honored as an
explicit override. The correct-engagement banner is
`[PTM_HMATRIX_VF_REPLACE] ROUTE = GPU (radiation.use_gpu in the config) … mesh-hash OK`
(it read `REPLACED H-matrix with symmetric per-partner referee VF … mesh-hash OK` on the
CPU route before that date, and still does on the CPU route); `run_yearlong_wedge_safe.py`
checks for it.

**This path is per machine and is deliberately not recorded in the YAML.** There is no
in-repository copy and no download.

## Output

`output.directory` points at `output/moon3d_native_yearlong/{coupled,radoff}`, relative to
the working directory and covered by the repository's `.gitignore`. On a real run
`scripts/run_yearlong_wedge_safe.py --output-dir …` overrides it per chunk; the value in
the file only matters for a direct `ptm_thermal_sim --config …` invocation.

## How the campaign was actually launched

Not directly. A single invocation would submit all
`13 orbits × 120 samples = 1560` equilibration timesteps as one unbroken GPU dispatch and
hit the macOS + AMD command-buffer watchdog. Use the tracked two-stage driver, which
splits equilibration from the output window and reuses the `TMTP` checkpoint:

```sh
python3 scripts/run_yearlong_wedge_safe.py --mode calibrate \
    --config examples/moon3d_native_yearlong/moon3d_native_yearlong_coupled.yaml \
    --candidate-batch-size 8000 --output-dir <scratch>/calibrate

python3 scripts/run_yearlong_wedge_safe.py --mode run \
    --config examples/moon3d_native_yearlong/moon3d_native_yearlong_coupled.yaml \
    --output-dir <output-parent>/coupled \
    --stage-a-batch-size <calibrated> --stage-a-output-steps 20 \
    --stage-b-batch-size 200000 --ptm-bin <bin> --merge-bin <bin>
```

Read that script's module docstring first — it documents the wedge root cause and the
continuity guarantee for the two-stage handoff.

## Two things these configs do NOT do

- **The `thermophysical:` block is decorative on the Moon.** Loading either file prints
  `[config] WARNING: 19 key(s) under 'thermophysical:' are read by nothing`
  (`albedo`, `emissivity`, `ks`, `kd`, `H`, `rhos`, `rhod`, `cp0`, `Qb`, `chi` and their
  `use_default_*` flags). That is pre-existing (`docs/KNOWN_LIMITATIONS.md`, PTM-67) and
  was equally true of the campaign runs — the block records intent, and the run used the
  built-in Moon defaults. Use the `*_override` keys if you need a non-default value.
  The whole `visualization:` block and `output.no_maps` are inert too: 30 dead keys per
  file by `src/python/tests/test_config_parity.py`'s own reckoning. They are kept
  because the campaign ran with them and removing them would make these files something
  other than what produced the published figures. Note that the CI dead-key gate globs
  `examples/*.yaml` only, so no config in an `examples/` **subdirectory** — these two
  included — is actually checked by it.
- **The illumination-fix environment flags are not in the YAML.** The campaign ran with
  `PTM_ILLUM_TRUE_NORMALS=1 PTM_ILLUM_RISE_EPS=1 PTM_ILLUM_RISE_KEPS=0.2
  PTM_ILLUM_RISE_HABS=0 PTM_ILLUM_SEAL_HOLES=1`; see `docs/ENVIRONMENT.md`.
  `run_yearlong_wedge_safe.py` gates on each of their startup banners.
