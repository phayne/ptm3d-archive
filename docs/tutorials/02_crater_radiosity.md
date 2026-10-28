# Tutorial 2: crater radiation exchange, on versus off

**Configs:**
[`examples/tutorial04_crater80N_radoff.yaml`](../../examples/tutorial04_crater80N_radoff.yaml)
(control) and
[`examples/tutorial04_crater80N_coupled.yaml`](../../examples/tutorial04_crater80N_coupled.yaml)
**Mesh:** `data/meshes/lunar_crater_80N.ply`, a 12,800-face bowl crater at 80°N (about 29 m
facets), shipped with the repository.
**Cost:** about 35 s (control) and 110 s (coupled, including the view-factor build) on four cores;
under 1 GB of memory. The view-factor file is 36,341,754 bytes.

## The question

A crater floor that never sees the Sun is not at zero flux: it receives sunlight **scattered** by
the sunlit walls and **thermal infrared** emitted by the warm rim. PTM3D computes this with a
radiosity solve over face-to-face view factors, coupled to the thermal model. This tutorial
measures the effect by differencing two otherwise identical runs.

## The illumination settings

Both configs start with an `engine_env:` block that turns on the validated fine-mesh
illumination corrections:

```yaml
engine_env:
  PTM_ILLUM_TRUE_NORMALS: '1'
  PTM_ILLUM_RISE_EPS: '1'
  PTM_ILLUM_RISE_KEPS: '0.2'
  PTM_ILLUM_RISE_HABS: '0'
  PTM_ILLUM_SEAL_HOLES: '1'
```

They are off by default so that earlier results stay reproducible. They matter on meshes with
sub-meter relief (polar micro-topography, the Moon3D testbed scan), where the default 1 cm
shadow-ray offset can step over a small rim and the default outward normal flip can create
false sunlit overhangs; on this 29 m-facet crater they change few faces. Always use them as a
set: `PTM_ILLUM_RISE_EPS=1` alone makes illumination worse. The run log echoes each one as
`(applied)`. See [ENVIRONMENT.md](../ENVIRONMENT.md).

## Run both

```bash
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_radoff.yaml
./build/bin/ptm_thermal_sim --config examples/tutorial04_crater80N_coupled.yaml
```

In the coupled run, look for:

```
Building H-matrix view factor for 12800 faces...
H-matrix saved to .../output/tutorial04_crater80N.vf (36341754 bytes)
[Radiosity] View-factor row sums Σ_j F_ij: max=0.1, mean=0.0 (0/12800 rows > 1 renormalized for energy conservation)
[eq-seed] DEFAULT CHANGED: this coupled run seeds the spin-up WITH indirect flux ...
--- Radiation Iteration 1 ---
  Re-running thermal model for 2627 faces...
--- Radiation Iteration 3 ---
  Converged on max-temperature criterion (0.77 K < 2.00 K)
```

Row sums well below 1 (an open crater sees mostly sky) and convergence within a few coupling
iterations are the expected signs. A second coupled run loads the saved file
(`Loaded view factors from: ...`) instead of building it. The `DEFAULT CHANGED` line says that
the spin-up of coupled runs now includes the indirect flux; `thermal.seed_indirect: off`
restores the earlier behavior.

**Spin-up.** These configs integrate 8 diurnal cycles, and the run prints a spin-up WARNING
(estimated requirement about 62 cycles for 0.1 K; see [Tutorial 1](01_quickstart.md)). On this
crater an 8-cycle spin-up is within a few tenths of a kelvin (median) of a 40-cycle reference,
which takes about 12 minutes on four cores. Raise `equilibration_orbits` for production work.

Two-band (visible and infrared) radiosity is the default (`radiation.band_separated`), so the
coupled run also writes `indirect_solar` and `indirect_ir` outputs.

## Compare

```bash
python3 - <<'EOF'
import struct, array, statistics as st
def load(p):
    b = open(p, 'rb').read(); n = struct.unpack('<i', b[:4])[0]
    a = array.array('f'); a.frombytes(b[4:4+4*n]); return list(a)
base = 'output/tutorial04_crater80N_%s/statistics/%s.bin'
# "Shadowed set": faces whose peak direct solar flux in the CONTROL run is < 50 W/m2.
dsm  = load(base % ('radoff', 'direct_solar_max'))
shad = [i for i, v in enumerate(dsm) if v < 50.0]
print(f"shadowed set: {len(shad)}/{len(dsm)} faces")
for name in ('temperature_min', 'temperature_avg', 'temperature_max'):
    off = load(base % ('radoff', name)); on = load(base % ('coupled', name))
    d = [on[i] - off[i] for i in range(len(off))]
    print(f"{name}: shadowed mean {st.mean(off[i] for i in shad):.1f} K "
          f"-> {st.mean(on[i] for i in shad):.1f} K | largest change {max(d):+.1f} K "
          f"| median over all faces {st.median(d):+.2f} K")
EOF
```

Expected (CPU build of this release). **Shadowed set: 1,423 of 12,800 faces.**

| Quantity (1,423 shadowed faces) | radiation off | radiation on | change |
|---|---:|---:|---:|
| mean T_avg | 24.6 K | 103.4 K | **+78.8 K** |
| mean T_min | 24.4 K | 64.1 K | +39.6 K |
| mean T_max | 26.1 K | 155.4 K | +129.3 K |
| largest single-face change in T_avg | | | +80.6 K |
| median change over all 12,800 faces, T_avg | | | +0.19 K |

Without radiation exchange the shadowed faces sit near the geothermal floor (24.04 K); with it
they warm by tens of kelvin, while the median face on the mesh, most of it sunlit terrain, moves
by about 0.2 K. Quote the population and the criterion along with such numbers: the shadowed set
is chosen by the illumination model and changes with it.

## Things to try

- Read the two indirect bands (Tutorial 3): over the shadowed set, the median peak indirect flux
  is 4.5 W/m² of scattered sunlight but 28.7 W/m² of thermal infrared. The warm rim's emission,
  not scattered sunlight, heats this crater's floor.
- `data/meshes/lunar_crater_85N.ply`: the same crater at 85°N, colder and more shadowed.
- Raise `radiation.max_coupling_iterations` and lower `coupling_temp_threshold` to trade runtime
  for convergence.
