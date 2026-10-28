# Changelog

## Unreleased (release/r1-engine): the Moon3D revision engine

Results change for radiation-coupled runs; see "Behavior changes".

### Behavior changes

- **Spin-up seed for coupled runs.** `thermal.seed_indirect` defaults to `series`: in a
  radiation-coupled run the spin-up starts each column from its absorbed flux plus the
  time-varying indirect flux of the first radiation solution, folded over whole diurnal cycles.
  Every face of a coupled run starts closer to its converged state (measured on the shipped
  80°N crater with a 1-cycle spin-up: lit-face mean temperature within 0.36 K of a 40-cycle
  reference instead of 1.79 K). Where it cannot act (radiation off, a warm-started chunk, the
  CPU scalar solver, an output window shorter than one diurnal cycle) the run says so and uses
  the previous seed. `provenance.txt` records `seed_indirect`, where it came from, and whether it
  acted. Previous behavior: `thermal.seed_indirect: off` or `--seed-indirect off`.
- **CPU coupling passes re-seed with the same series seed** (`PTM_CPU_RETHERMAL=seed-series` is
  now the default). On the same crater, never-lit mean temperature moves from 1.75 K to 0.38 K
  from the 40-cycle reference. Previous behavior: `PTM_CPU_RETHERMAL=reseed`.
- **Chunked CPU coupled runs** write the coupled end-of-window column to the checkpoint, so a
  chunked series matches the single-shot series at the seams.
- **View-factor orientation check.** A precomputed matrix loaded with `PTM_HMATRIX_VF_REPLACE`
  is refused (exit 3) when it was built for the other side of the surface than the run
  illuminates (`PTM_VF_ORIENT_OVERRIDE=1` proceeds with a warning). The offline referee
  `mc_ref_pp` orients faces by the same rule and refuses a mesh it would have to reverse on more
  than 1% of faces.
- **Unknown configuration keys and command-line arguments** are reported as warnings instead
  of being ignored silently; `--validate-config` summarizes them ("ACCEPTED WITH WARNINGS").
- **`time_steps.txt`** labels column 1 correctly: ephemeris seconds past J2000 (the values
  never changed).
- Engine messages no longer carry internal tracking numbers.

### Additions

- `--seed-indirect off|series|mean`; `PTM_TIMING_JSON` (per-stage timing report); view-factor
  statistics sidecars; `assemble_vf_haworth.py --symmetrize-divisor pooled`; an out-of-core view
  factor assembler.
- Apple Silicon (arm64) builds.

### Removed from the repository

- Draft-paper documents, Europa ice-shell research code and documents, internal campaign scripts
  that pointed at private paths, and dated design and audit records.

### Examples and documentation

- Quickstart spin-up raised to 80 cycles; tutorial 2 uses the fine-mesh illumination settings;
  the Moon3D year-long example carries the revision's settings, including Q_b = 0.022 W/m².
- Documentation rewritten for this release; `KNOWN_LIMITATIONS.md` lists only open limitations.

## Fixed in earlier development (after v1.0.3)

Listed so that results from older builds can be read correctly.

- `thermal.geothermal_heat_flow` was ignored on the Moon, Europa and Mercury (all solvers): runs
  used the built-in Q_b whatever the configuration said.
- The CPU SIMD solver used a weaker surface-temperature iteration that could leave a face up to
  tens of kelvin warm at a shadow edge; it now shares the scalar solver's method.
- The CPU solver floored layer thickness at 1 mm, so on fast-rotating, low-inertia bodies its
  top cell was thicker than its own grid.
- The default spin-up seed became the per-face conductive steady profile (`spinup_seed:
  physical`); the 24-cycle clear-sky pre-conditioner of earlier CPU runs is off unless requested.
- Per-step frames are numbered with at least six digits (four before); readers sort by the parsed
  index.
- `equilibration_orbits` means integrated cycles on every solver (earlier GPU runs integrated
  twice the configured value).
- Mercury used its sidereal rotation as the solar day (a factor of three).
- An unrecognized body name silently ran with Europa's properties; it is now refused.
- A 500 K temperature cap applied on one CPU path only; it was removed on all paths.
- The lunar Bond-albedo map was registered 180° off in longitude (the distributed archive is
  corrected); Saturn and Mars primary infrared fluxes were inconsistent with their temperatures.
- The near-field view-factor occlusion test used fixed meter-scale offsets; it is now relative to
  facet size. The GPU radiosity solver now renormalizes view-factor rows above 1 in-kernel.
- Thermal-state checkpoints carry an identity record (grid, mesh, configuration, body) and refuse
  a mismatched restore.
- A `PTM_ILLUM_TRUE_NORMALS` run on a smooth icosphere undid the analytic radial normals; it is
  now declined there.
