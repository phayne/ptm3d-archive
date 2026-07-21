# PTM3D laptop tutorial (SKELETON)

> **Status:** skeleton (portability Phase 1). The commands and structure are laid out; the
> numeric expected-outputs marked `[verify]` still need a confirmed run on a 16 GB laptop-class
> machine (they were not runnable during the overnight build, which shared the GPU with a
> production job). Treat `[verify]` values as placeholders until confirmed.

A 10-minute, 16 GB-friendly walkthrough of PTM3D's three core capabilities. Everything here stays
under ~100k faces / ~360 steps (see `docs/PORTABILITY-AUDIT.md` §4 and `scripts/ptm_mem_estimate.py`).
Build first with `INSTALL.md` (CPU core is fine for all of this).

## 0. Estimate before you run
```bash
python3 scripts/ptm_mem_estimate.py --faces 40000 --steps 360   # expect: OK on 16 GB
```

## Step 1 — Radiation, validated: the Ingersoll bowl crater
The analytic gate. No mesh or SPICE needed.
```bash
cmake --build build --target test_bowl_crater -j
./build/bin/test_bowl_crater
```
Expected: the printed max-temperature ratio matches Ingersoll & Svitek (1992) within tolerance,
and the test prints `PASS`. This is the same gate CI runs. `[verify: exact PASS line]`

## Step 2 — A 1k-face asteroid diurnal run (thermal, CPU)
Generate a small icosphere "asteroid" and run one diurnal cycle. The icosphere mesh generator
ships with the repo (`ptm-mesh` tooling / the icosphere path).
```bash
# 2a. Generate a low-res sphere (~1k faces). [verify: exact generator invocation]
#     e.g. an ico3/ico4 icosphere written to /tmp/asteroid_1k.ply
# 2b. Run a single diurnal cycle with a modest thermal inertia:
./build/bin/ptm_thermal_sim --config examples/<asteroid_1k_diurnal>.yaml
```
Expected wall time: minutes on a laptop CPU. Output: `statistics/` with per-face Tmax/Tavg.
Inspect with the analysis tooling:
```bash
python3 -c "import sys; sys.path.insert(0,'scripts')"   # or use the ptm-analyze skill
```
`[verify: a relativized examples/asteroid_1k_diurnal.yaml is added (Phase 1 §4)]`

## Step 3 — A small crater with radiosity (the decimated Moon mesh)
Uses the spared small tutorial mesh `moon3d_c_cartesian_NEW_dec` (~57 MB VF — laptop-sized).
```bash
# 3a. Confirm/point at the decimated mesh + its compact view-factor matrix.
# 3b. Run with inter-face radiation (radiosity) ON:
./build/bin/ptm_thermal_sim --config examples/<moon3d_dec_radiosity>.yaml
```
Expected: warmer crater floors than a no-radiosity run (self-heating), and VF row-sums ≤ 1 in the
load banner. `[verify: relativized dec config + the compact VF path]`

## What you exercised
- Analytic radiation correctness (bowl crater).
- The threaded CPU thermal solver on a real (if tiny) body.
- Inter-face radiosity self-heating on a decimated crater mesh — all within a laptop's RAM.

## Next
- Scale up faces/steps only after `ptm_mem_estimate.py` says `OK` for your RAM.
- For production multi-site campaigns, see the chunked runner + SLURM array pattern
  (portability plan §"Cluster story").
- GPU acceleration (optional): build with `-DPTM_ENABLE_GPU=ON` and run
  `ptm_thermal_sim --calibrate-gpu` once per host.
