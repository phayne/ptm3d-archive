# Tutorial 04 — Crater radiosity: radiation exchange on vs. off

**Configs:**
[`examples/tutorial04_crater80N_radoff.yaml`](../../examples/tutorial04_crater80N_radoff.yaml)
(control) and
[`examples/tutorial04_crater80N_coupled.yaml`](../../examples/tutorial04_crater80N_coupled.yaml)
**Mesh:** `data/meshes/lunar_crater_80N.ply` — a 12,800-face bowl crater at
80°N, shipped with the repo.
**Cost:** ~8.5 s (control, measured) + coupled (incl. the view-factor build) on
a many-core workstation; a few minutes on a laptop. <1 GB RAM; the saved
view-factor file is 36,278,242 bytes (~35 MB), measured.

> **Status on v1.0.3 (checked 2026-08-05): the coupled run does not complete.**
> The control run (`..._radoff.yaml`) is fine. The coupled run builds the view
> factors, runs the thermal solve, enters the coupling stage and then stops
> with:
>
> ```
> FATAL: radiation coupling is enabled but the thermal solver returned no
> post-equilibration T(z) hand-off. ... Refusing to write results.
> ```
>
> exit 1, no output written. This is a guard added in v1.0.3 that fires on the
> CPU solver, and it reproduces on an unrelated mesh as well, so it is not
> specific to this crater. **Everything below the "Run both" section — the
> comparison table and the indirect-flux numbers quoted in
> [06_interpreting_outputs.md](06_interpreting_outputs.md) — was measured on
> 2026-07-29 and could not be reproduced on v1.0.3.** Treat those numbers as
> the physics you should expect, not as a check of your install; use the
> quickstart (tutorial 02) and `test_bowl_crater` for that.

## The physics question

A polar crater floor that never sees the Sun directly is not at 0 flux: it is
heated by sunlight **scattered** off the sunlit walls and by **thermal IR
emitted** by the warm rim. PTM3D models this with a radiosity solve over
face-to-face view factors, coupled to the thermal model. This tutorial
measures exactly how much that matters, by differencing two otherwise
identical runs.

## Run both

```bash
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_radoff.yaml
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_coupled.yaml
```

In the coupled run, watch for (elided). The first four lines are what v1.0.3
prints today; the iteration lines are from the 2026-07-29 run and are **not**
reached on v1.0.3 (see the status note above):

```
Radiation exchange enabled from config file
Building H-matrix view factor for 12800 faces...
H-matrix saved to .../output/tutorial04_crater80N.vf (36278242 bytes)
[Radiosity] View-factor row sums Σ_j F_ij: max=0.140, mean=0.027 (0/12800 rows > 1 ...)
--- Radiation Iteration 1 ---
--- Radiation Iteration 3 ---
  Converged on face-fraction criterion (2.34e-02% < 1.00e+00%)
```

(The "H-matrix" in the log is a hierarchically compressed storage scheme that
keeps the face-to-face view-factor matrix affordable; see
[../radiation-exchange.md](../radiation-exchange.md).) Row sums well below 1
(an open crater sees mostly sky) and convergence in ~3 coupling iterations are
the expected health signs. A second coupled run skips the view-factor
computation and reports `Loaded view factors from: .../tutorial04_crater80N.vf`.

Note: two-band (visible/IR-separated) radiosity is the default
(`radiation.band_separated`), which is why the coupled run also writes
`indirect_solar` and `indirect_ir` outputs — see
[06_interpreting_outputs.md](06_interpreting_outputs.md).

## Compare

```bash
python3 - <<'EOF'
import struct, array, statistics as st
def load(p):
    b = open(p, 'rb').read(); n = struct.unpack('<i', b[:4])[0]
    a = array.array('f'); a.frombytes(b[4:4+4*n]); return list(a)
base = 'output/tutorial04_crater80N_%s/statistics/%s.bin'
shad = [i for i, v in enumerate(load(base % ('radoff', 'direct_solar_max'))) if v < 50.0]
for name in ('temperature_min', 'temperature_avg'):
    off = load(base % ('radoff', name)); on = load(base % ('coupled', name))
    print(f"{name}: shadowed-face mean {st.mean(off[i] for i in shad):.1f} K "
          f"-> {st.mean(on[i] for i in shad):.1f} K with radiation")
EOF
```

Result measured 2026-07-29 on a CPU-only build (1671/12800 faces with
`direct_solar_max` < 50 W/m²). **Not reproducible on v1.0.3** — the coupled run
needed for the right-hand column aborts, per the status note above:

| Quantity (shadowed faces) | radiation OFF | radiation ON | effect |
|---|---:|---:|---:|
| mean T_avg | 67.6 K | 102.1 K | **+34.5 K** |
| mean T_min | 65.9 K | 74.4 K | +8.5 K |
| largest single-face ΔT_avg | | | +37.1 K |
| largest single-face ΔT_max | | | +94.0 K |

The **median face on the mesh changes by 0.00 K** — flat sunlit terrain
outside the crater doesn't care — while the shadowed floor and walls warm by
tens of kelvin. That asymmetry is the signature of correctly-scoped radiation
exchange (and why cold-trap studies cannot skip it).

## Things to try

- `PTM_DUMP_VF_ROWSUMS=/tmp/rowsums.bin` to dump per-face ΣF for QA
  (see [../ENVIRONMENT.md](../ENVIRONMENT.md)).
- `data/meshes/lunar_crater_85N.ply` (same generator, 85°N) for a colder,
  more shadowed version of the same experiment.
- Raise `max_coupling_iterations` and tighten `coupling_temp_threshold` to
  see the convergence criteria trade off runtime vs. accuracy.
