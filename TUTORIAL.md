# PTM3D laptop tutorial

A 10-minute, 16 GB-friendly walkthrough of PTM3D's three core capabilities.
Everything here stays under ~13k faces / ~240 steps (see the memory table in
[README.md](README.md) and `scripts/ptm_mem_estimate.py`). Build first with
[INSTALL.md](INSTALL.md) (the CPU core is fine for all of this), fetch SPICE
kernels once with `python3 scripts/fetch_kernels.py`, and run every command
**from the repository root**.

Numbered deep-dive tutorials live in [docs/tutorials/](docs/tutorials/).

## 0. Estimate before you run

```bash
python3 scripts/ptm_mem_estimate.py --faces 40000 --steps 360
# -> TOTAL resident est. : 0.32 GiB ... verdict : OK
```

## Step 1 — Radiation, validated: the Ingersoll bowl crater

Comparison to the analytic solution that validates the numerical model (Ingersoll, Svitek & Murray 1992). No mesh or SPICE needed.

```bash
cmake --build build --target test_bowl_crater -j
./build/bin/test_bowl_crater
```

Expected (verified 2026-07-29, CPU-only build):

```
  F_wall (analytical)  = 0.137931
  F_terrain (computed) = 0.138611
  Error: 0.492713%
--- Reciprocity Test ---
  Max relative residual (full scan): ~7e-14%
  Result: PASSED
--- Radiosity Solver Test ---
  Iterations: 2
  Converged: YES
```

The view-factor error vs. the analytic bowl-crater solution is **<0.5%** and
reciprocity holds to machine precision (`Result: PASSED` appears once per tested
crater configuration). This is the gate every radiation change has to pass; the
repository's CI workflow (`.github/workflows/ci.yml`) builds and runs it on
Linux and macOS.

## Step 2 — A global Moon diurnal run (thermal, CPU)

The quickstart config: one lunation on a 1280-face icosphere.

```bash
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

Expected: seconds-to-minutes of wall time; `output/quickstart_moon_diurnal/`
with `statistics/*.bin` per-face extremes — equatorial noon Tmax ≈ 385 K,
median night Tmin ≈ 93 K (verified values and how to read the binaries:
[docs/tutorials/02_lunar_diurnal_icosphere.md](docs/tutorials/02_lunar_diurnal_icosphere.md)
and [docs/tutorials/06_interpreting_outputs.md](docs/tutorials/06_interpreting_outputs.md)).

## Step 3 — A polar crater with radiosity (shipped mesh)

Uses the shipped 12,800-face bowl-crater mesh at 80°N
(`data/meshes/lunar_crater_80N.ply`) — small enough for a laptop, shadowed
enough that inter-face radiation matters. Run the radiation-off control and
the coupled run, then compare crater-floor temperatures:

```bash
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_radoff.yaml
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_coupled.yaml
```

Expected: the coupled run computes (and saves for reuse) the face-to-face view
factors, the radiation–temperature iteration converges, and shadowed floor
faces come out warmer than the control — walkthrough and numbers in
[docs/tutorials/04_crater_radiosity.md](docs/tutorials/04_crater_radiosity.md).

> **On v1.0.3 the second command does not complete** (`FATAL: ... no
> post-equilibration T(z) hand-off`, exit 1). The control run is unaffected, and
> so is everything in steps 1 and 2. Read the status note in
> [04_crater_radiosity.md](docs/tutorials/04_crater_radiosity.md) before you run
> it, and use step 1 + step 2 to validate your install.

## What you exercised

- Analytic radiation correctness (bowl crater).
- The threaded CPU thermal solver on a real global body with SPICE ephemerides.
- Inter-face radiosity self-heating on a polar crater mesh — all within a
  laptop's RAM.

## Next

- Interpreting every output file: [docs/tutorials/06_interpreting_outputs.md](docs/tutorials/06_interpreting_outputs.md).
- Scale up faces/steps only after `ptm_mem_estimate.py` says `OK` for your RAM.
- Environment variables (validated illumination fixes, radiosity band
  separation): [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- GPU acceleration (optional): build with `-DPTM_ENABLE_GPU=ON` and run
  `ptm_thermal_sim --calibrate-gpu` once per host.
