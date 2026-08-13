# Chunked-run equivalence and the output epoch grid

A chunked run splits one long simulation into consecutive windows and merges the
per-window statistics. It is only useful if the merged result is the result the
single-shot run would have produced. This note states the construction that makes
that true, and records what it is measured to be worth.

## The two requirements

**R1, tiling.** The output epochs of the chunks must partition the single-shot
output epochs: every epoch sampled once, none skipped, none sampled twice. A
duplicated epoch weights one instant twice in every reduction (`temperature_avg`,
`illumination_avg`, the Schorghofer ice accumulator); a skipped one drops an
instant.

**R2, handoff.** The state chunk *k* hands to chunk *k+1* must be the state at
chunk *k+1*'s **first** output epoch, not at chunk *k*'s last one.

## The construction

The grid is half-open. With cadence `dt = duration_days*86400/timesteps`, the
single-shot run samples `t0 + i*dt` for `i` in `[0, N_total)`; chunk *k* is
handed the sub-range `[c_k, c_k + N_k)` of the same grid. R1 then holds by
construction.

R2 needs one more grid point than the export does, because the solver integrates
`total_time_steps − 1` intervals and so lands its final state on its last grid
point. Chunk *k* therefore

- integrates `N_k` intervals over `N_k + 1` grid points, `[t_k, t_k + N_k*dt]`;
- exports and reduces only the first `N_k`;
- writes `final_state_file` at the extra point, which is `t_{k+1}`.

That is `thermal.checkpoint_lookahead_steps: 1`. The extra point is not written
as a frame, does not appear in `time_steps.txt`, is not reduced into
`statistics/`, and is excluded from the equilibration seed's forcing mean, so it
changes nothing except where the checkpoint sits.

Chunk offsets are carried as `simulation.start_time_offset_seconds`, an
**ephemeris**-seconds offset, not as a shifted UTC `start_time` string. TDB − UTC
has a periodic relativistic term (~1.7 ms amplitude, annual period) and steps by
a whole second at a leap second, so offsetting in UTC does not move a boundary by
an exact number of cadence seconds.

## What was wrong before 2026-08-12

`IlluminationModel::calculateIlluminationFromMesh` built one uniform grid as

```cpp
et_step = (et_end - et_start) / (time_steps_ - 1);
```

with `time_steps_ = equilibration_steps + output_steps` and `et_end - et_start`
the combined equilibration+output span. The **output** cadence was therefore set
by dividing the **combined** span, by one fewer interval than there are samples.
Three consequences, all measured on an 80 N lunar crater (12,800 faces, 236
output steps over 59.06 days, one equilibration lunation, CPU, radiation off):

1. The cadence was long by `N_tot/(N_tot − 1)`: 21683.362 s instead of
   21621.966 s, 0.28 %.
2. Output frame 0 was displaced from the nominal start by
   `equilibration_days*86400/(N_tot − 1)`: `2026-04-05T01:59:53.722` for a run
   whose `start_time` was `2026-04-05T00:00:00`.
3. `N_tot` depends on the length of the run, so chunks of the same simulation
   got *different* cadences. The chunk grids did not tile: max epoch
   misalignment 14,438 s at 2 chunks, 18,061 s at 4, 18,798 s at an unequal
   5-way split, with one near-duplicate epoch per seam.

Point 2 applies to every run with an equilibration prefix and point 1 to every
run at all, single-shot included. The epoch **labels** on products generated
before this date are wrong by those amounts; the physics at those (shifted)
epochs was solved correctly.

A separate 0.5 s of quantization came from anchoring the grid on
`calculateAdjustedStartTime()`, which round-trips through
`et2utc_c(..., "ISOC", 0, ...)` — whole seconds. The grid is now anchored on
`start_time` directly.

## Measured

80 N lunar crater, 12,800 faces, 236 output steps over 59.06 days, CPU SIMD,
radiation off, `origin/main` at `53b86cc` on both sides.

Epoch grid, chunked versus single-shot:

| chunks | before: max offset | before: near-dup epochs | after: max offset | after: near-dup |
|---|---|---|---|---|
| 1 | 0.000 s | 0 | 0.000 s | 0 |
| 2 | 14 438.444 s | 1 | 0.000 s | 0 |
| 4 | 18 060.805 s | 3 | 0.000 s | 0 |
| 5 (48,47,47,47,47) | 18 798.149 s | 4 | 1.1e-6 s | 0 |

`illumination_avg`, `illumination_max`, `direct_solar_avg`, `direct_solar_max`
are **byte-identical** between the chunked and single-shot runs at every chunk
count. They are solver-independent, so they measure the epoch grid alone; before
the fix they differed by 0.55 W m<sup>-2</sup> mean / 5.7 max at 2 chunks.

Per-frame surface temperature, all chunks warm-started from a shared checkpoint
(so the cold-start seed is out of the comparison), 4 chunks:

| | before | after |
|---|---|---|
| chunk 0, all 59 frames | — | **byte-identical** |
| chunks 1-3, interior frames | — | 3.05e-5 K max (the float32 checkpoint quantum) |
| chunks 1-3, frame 0 (the seam) | — | up to 37.8 K |
| merged `temperature_avg` | 0.2585 mean / 9.798 max K | 0.00081 mean / 0.160 max K |

The merged mean improves by 320x and the max by 61x.

## What is still not equivalent

Two mechanisms survive this change. Neither is an epoch-grid defect.

**The seam frame is the restored initial condition, verbatim.** Frame 0 of every
chunk after the first reports the checkpoint column as restored, rather than the
surface re-solved at that frame's own epoch — the massless surface node makes
row 0 a diagnostic of the epoch it was last evaluated at. Worth up to 37.8 K in
that one frame, 8e-4 K once averaged over 236 samples. It is the deliberately
unfixed row-0 convention; changing it is a convention decision, not a bug fix.

**A cold chunk 0's equilibration seed depends on the length of the chunk.** The
`spinup_seed: physical` seed anchors each column on `<F_abs>`, the time mean of
that face's absorbed flux over the *whole* forcing table. That table is
`equilibration_steps + output_steps` long, so chunk 0 of a 4-way split averages
over 177 steps where the single-shot run averages over 354, and the two runs
start from different initial conditions. On the crater above this leaves the
production (cold chunk 0) configuration at 0.717 mean / 13.05 max K on
`temperature_avg` at 4 chunks, against 1.141 / 13.50 before — an improvement, but
not equivalence. The same dependence means a single-shot run's initial condition
changes if you ask it for more output days at the same equilibration.
