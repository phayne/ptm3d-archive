# Tutorial 04 — Crater radiosity: radiation exchange on vs. off

**Configs:**
[`examples/tutorial04_crater80N_radoff.yaml`](../../examples/tutorial04_crater80N_radoff.yaml)
(control) and
[`examples/tutorial04_crater80N_coupled.yaml`](../../examples/tutorial04_crater80N_coupled.yaml)
**Mesh:** `data/meshes/lunar_crater_80N.ply` — a 12,800-face bowl crater at
80°N, shipped with the repo.
**Cost (measured 2026-08-05, 48-core workstation, CPU-only build):** 7.8 s
(control) + 14.1 s (coupled, including the view-factor build); a few minutes on
a laptop. <1 GB RAM; the saved view-factor file is 36,278,242 bytes (~35 MB).

> **On the `v1.0.3` tag exactly, the coupled run aborts** with
> `FATAL: ... no post-equilibration T(z) hand-off` and exit 1. This is fixed
> on current `main`, where the coupled run completes; the numbers below are
> from the fixed build.

## The physics question

A polar crater floor that never sees the Sun directly is not at 0 flux: it is
heated by sunlight **scattered** off the sunlit walls and by **thermal IR
emitted** by the warm rim. PTM3D models this with a radiosity solve over
face-to-face view factors, coupled to the thermal model. This tutorial
quantifies the effect by differencing two otherwise identical runs.

## Run both

```bash
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_radoff.yaml
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_coupled.yaml
```

In the coupled run, watch for (elided, measured 2026-08-05):

```
Radiation exchange enabled from config file
Building H-matrix view factor for 12800 faces...
H-matrix saved to .../output/tutorial04_crater80N.vf (36278242 bytes)
[Radiosity] View-factor row sums Σ_j F_ij: max=0.140, mean=0.027 (0/12800 rows > 1 ...)
Faces requiring radiation exchange: 12800 / 12800
--- Radiation Iteration 1 ---
  Re-running thermal model for 2578 faces...
--- Radiation Iteration 4 ---
  Re-running thermal model for 1942 faces...
  Converged on max-temperature criterion (0.20 K < 2.00 K)
Simulation completed in 14.061 seconds
```

On the CPU solver the run also prints a `[Re-thermal] WARNING:` noting that
each coupling iteration re-equilibrates the selected faces from scratch rather
than resuming a saved post-equilibration column, so CPU and GPU coupled fields
can differ. This is expected on the CPU path.

(The "H-matrix" in the log is a hierarchically compressed storage scheme that
keeps the face-to-face view-factor matrix affordable; see
[../radiation-exchange.md](../radiation-exchange.md).) Row sums well below 1
(an open crater sees mostly sky) and convergence in a handful of coupling
iterations (4 here) are the expected health signs. A second coupled run skips
the view-factor
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
# "Shadowed set" = faces whose peak direct solar flux in the CONTROL run is
# < 50 W/m2. The population is printed because it is part of the result.
dsm  = load(base % ('radoff', 'direct_solar_max'))
shad = [i for i, v in enumerate(dsm) if v < 50.0]
print(f"shadowed set: {len(shad)}/{len(dsm)} faces (direct_solar_max < 50 W/m2, control run)")
for name in ('temperature_min', 'temperature_avg', 'temperature_max'):
    off = load(base % ('radoff', name)); on = load(base % ('coupled', name))
    d = [on[i] - off[i] for i in range(len(off))]
    print(f"{name}: shadowed-face mean {st.mean(off[i] for i in shad):.1f} K "
          f"-> {st.mean(on[i] for i in shad):.1f} K with radiation "
          f"| largest single-face change {max(d):+.1f} K "
          f"| median over ALL faces {st.median(d):+.2f} K")
EOF
```

Verified result (2026-08-05, CPU-only build). **Shadowed set: 1,422 of 12,800
faces**, by the criterion in the snippet above (`direct_solar_max < 50 W/m²` in
the control run):

| Quantity (1,422 shadowed faces) | radiation OFF | radiation ON | effect |
|---|---:|---:|---:|
| mean T_avg | 66.3 K | 108.3 K | **+42.0 K** |
| mean T_min | 64.6 K | 74.4 K | +9.7 K |
| mean T_max | 69.2 K | 154.9 K | +85.8 K |
| largest single-face ΔT_avg | | | +43.1 K |
| largest single-face ΔT_max | | | +88.0 K |

The **median face on the mesh changes by 0.00 K** — flat sunlit terrain
outside the crater is unaffected — while the shadowed floor and walls warm by
tens of kelvin. That asymmetry is the expected signature of radiation
exchange, and why cold-trap studies cannot skip it.

Quote the **population and the criterion** together with these numbers: both
are build-dependent, because the shadowed set is selected by the illumination
model. The same criterion selected 1,671 faces on 2026-07-29 and 1,422 on
2026-08-05.

## Things to try

- `PTM_DUMP_VF_ROWSUMS=/tmp/rowsums.bin` to dump per-face ΣF for QA
  (see [../ENVIRONMENT.md](../ENVIRONMENT.md)).
- `data/meshes/lunar_crater_85N.ply` (same generator, 85°N) for a colder,
  more shadowed version of the same experiment.
- Raise `max_coupling_iterations` and tighten `coupling_temp_threshold` to
  see the convergence criteria trade off runtime vs. accuracy.
