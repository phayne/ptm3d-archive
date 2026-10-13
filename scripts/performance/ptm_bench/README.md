# ptm_bench: Moon3D R1 benchmark and accuracy harness (C2/C3)

Reviewer 1 asked for two things before any scaling claim means anything: that
accuracy improves with N, and that the cost exponents are measured rather than
assumed. This harness produces both. Scope is what R1 needs, not a general
framework. Plan of record: `~/projects/moon3d_R1/PTM3D_changes_for_R1.md`, section 4.

Python with numpy and PyYAML (VTK for arm C meshes), e.g.
`~/anaconda3/envs/europa_viz/bin/python3`.

| file | role |
|---|---|
| `make_meshes.py` | mesh ladders: A bowl (exported by `bowl_crater_ptm`), B fBm (band-limited / extended), C Moon3D decimations, D Haworth (verified, not modified) + `manifest.json` |
| `plan.py` | builds the batch, predicts wall time and peak memory per run against the hard caps, prints the GPU-safety block; `--write plan.json` |
| `run_bench.py` | executes `plan.json` resumably (JSONL ledger); refuses a busy machine; records provenance (SHA, exe and every dylib md5, Embree) |
| `accuracy.py` | bowl metrics: floor F_sky bias and peak (kept separate), T_floor vs Ingersoll Eq. 8, per-face Q_ind vs a same-N reference |
| `collect.py` | ledger -> one CSV for the exponent fits; flags shared-machine rows |
| `inventory.py` | timings ALREADY measured (Stage 1/3, keps sweep, v4 sweeps): per-component splits recovered from logs, overlap/contamination flags -> `bench/inventory.json` |
| `templates/moon_lunation.yaml` | engine template: one lunation, 360 steps, 2 spin-up orbits |

Engine and tool hooks this relies on (all inert when unset):

- `PTM_TIMING_JSON=<path>`: `ptm_thermal_sim` writes per-phase wall/CPU/peak RSS, per-coupling-iteration
  records and the radiosity traversal counters (`src/cpp/main/run_timing.h`). Outputs are byte-identical
  with it set or unset.
- `[VF-STATS]` banner on every `PTM_HMATRIX_VF_REPLACE` load; `<csr>.vf_stats.json` from
  `assemble_vf_haworth.py` (`scripts/ptm12_diag/vf_stats.py`, C4); `# STATS` line from `mc_ref_pp`.
- `bowl_crater_ptm`: `PTM_BOWL_WRITE_PLY`, `PTM_BOWL_VF_CSR`, `PTM_BOWL_ESTIMATOR=sparse_production`,
  `PTM_BOWL_WRITE_CSR`, `PTM_BOWL_DUMP`.

## Order of operations

```
PY=~/anaconda3/envs/europa_viz/bin/python3
B=~/projects/moon3d_R1/bench
$PY make_meshes.py --out $B/meshes --arms A,B,C,D
$PY plan.py --manifest $B/meshes/manifest.json --root $B/runs --write $B/plan.json   # dry run + plan
$PY run_bench.py --plan $B/plan.json --bin-dir <build>/bin --dry-run                  # what would run
# only on a quiet machine, after Stage 1 and Stage 3; GPU rows additionally need --gpu-ok
nohup caffeinate -i $PY run_bench.py --plan $B/plan.json --bin-dir <build>/bin --gpu-ok > $B/batch.log 2>&1 &
$PY collect.py $B/runs/ledger.jsonl --csv $B/runs/collected.csv
```

Rules the code enforces: no timing on a shared machine (`--allow-shared` is for smoke tests and marks
every record); no coupled row without a CSR (a silent in-engine VF build on a large lunar mesh is the
wedge trigger); coupled rows must log `REPLACED H-matrix`; GPU rows get batch 1280, pause 1500 ms,
`PTM_RAD_RERUN_MAX_GPUS=1`; no AGT/Artemis path is accepted by `make_meshes.py`.

Disk (PTM-722). The batch writes far more than it keeps, so the runner cleans up as it goes and
records what it deletes in the ledger:
- M1: a sweep that is never assembled writes no partner bins (`/dev/null`); its nnz is reported as a
  BRACKET [D, 2D] from the `mc_ref_pp` STATS lines (D = directed partners kept), never as a count.
- M2: partner bins are deleted after a successful assembly (file count and bytes recorded).
- M3: a CSR this plan produced is deleted once EVERY consumer run (repeats included) is `done`;
  md5, bytes, N, nnz and the vf_stats sidecar go into the ledger first, and the sidecar, provenance
  and rowsum stay on disk. A failed or pending consumer keeps it. External CSRs are never touched.
- M4: per-step frame directories are deleted after each engine row; `statistics/`, `timing.json`
  and the small top-level files stay.
- Guard: before each row, free >= 2 x the row's `predicted_disk_peak_bytes` + `--disk-reserve-gib`
  (default 15). Otherwise the batch stops between rows (ledger `abort_disk`, exit 3); resume later.
`plan.py` prints each row's disk peak and final footprint and the simulated batch peak.

Trimmed batch (Paul 2026-10-13, < 24 h predicted). Rates are MEASURED on the R1 binary family
(10 clean Stage-3 Moon3D 509k lunations; `inventory.py`), not assumed. Coupled timing rows are
SHORT windows (30 output steps at the lunation cadence + 0.25 spin-up orbit, `coupling_mode: fixed`,
2 iterations: iteration 1 = transient re-thermal with the equilibration replay, iteration 2 = steady).
Rad-off rows keep full lunations because illumination cost is not uniform over a lunation. Sweeps above
~1e6 faces are seeded-random 1-in-10 ROW SAMPLES (`row_sample_1_in`): per-face sweep cost and an nnz
bracket from the per-face `nkept` column of `mc_ref_pp` (validated against full sweeps within 2 SE).
The full Haworth r25_100m refinement sweep is opt-in (`plan.py --with-haworth-refinement`).

Precision trap: VTK's PLY reader (and so pyvista) demotes double vertices to float32, i.e. 0.125 m at
the lunar radius, coarser than a 2.5 mm Moon3D facet. Read these PLYs with numpy, or VTK legacy files.
