# Tutorial 1: quickstart, a global Moon over one lunation

**Config:** [`examples/quickstart_moon_diurnal.yaml`](../../examples/quickstart_moon_diurnal.yaml)
**Cost:** about 30 s on four cores; under 1 GB of memory.

## What the run does

One lunar day (29.53 Earth days) of surface and subsurface temperatures over the whole Moon on a
level-3 icosphere (1,280 faces, about 480 km across each):

1. Spin-up. Each face's column is integrated for 80 diurnal cycles of the real forcing,
   sampled 48 times per cycle, before the output window opens.
2. Illumination. For each of the 240 output steps (about 3 h apart), shadow rays are traced
   from every face toward the Sun's SPICE position; Earth's reflected light and thermal
   emission are added.
3. Thermal solve. A 1D conduction column under each face, with lunar regolith properties
   (density and conductivity increasing with depth, temperature-dependent conductivity and heat
   capacity).
4. Statistics. Per-face minimum, mean and maximum over the output window, for temperature and
   for each flux component.

## Check the build first (optional, seconds)

```bash
cmake --build build --target test_bowl_crater -j && ./build/bin/test_bowl_crater
python3 scripts/ptm_mem_estimate.py --faces 1280 --steps 4080     # memory estimate (240 output + 3840 spin-up steps)
```

`test_bowl_crater` compares analytic view factors of a bowl-shaped crater with the closed form
(`Error: 0.492713%`) and checks reciprocity; it ends with `[test_bowl_crater] PASS: 12 of 12
verdicts passed`. It is a quick check of the radiosity code, not of the production view-factor
builder (see [VALIDATION.md](../VALIDATION.md), Fig. 7).

## Run it

From the repository root:

```bash
python3 scripts/fetch_kernels.py --bodies moon   # once, about 44 MB from NAIF
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

`fetch_kernels.py` without `--bodies` also fetches the 1.1 GB Jupiter-system ephemeris that the
Europa examples need.

Lines to look for in the log:

```
Mesh created with 642 vertices and 1280 faces
Initializing SPICE with meta-kernel
Equilibration (integrated): 80.00 day/night cycles = 1 pass(es) x 80.00-cycle forcing block  [default]
    lit   (Qs>0)     n=     1280   tau1 = 9.56 cycles   E0 = 99.088 K   cycles needed: median 66.2, p95 68.8
[Statistics] Exported to output/quickstart_moon_diurnal/statistics
Simulation completed in 29.951 seconds
```

### The spin-up check

The "cycles needed" line is the engine's estimate of how many diurnal cycles of spin-up this
mesh needs for the deep column to settle to within `thermal.equilibration_target_K` (0.1 K by
default). It fits a single relaxation time (`tau1`, here about 9.6 lunations) to each face, and
it is conservative (it tends to ask for more than a measured convergence study needs). When the
configured `equilibration_orbits` is below the 95th-percentile requirement, the run prints a
WARNING with the predicted residual and continues. This quickstart integrates 80 cycles, so it
does not warn. Surface temperatures settle much faster than the deep column. Measured on this
run, a 1-cycle spin-up gives peak temperatures within 0.25 K of the 80-cycle run, but nighttime
minima up to 4.9 K off (median 1.1 K) and mean temperatures up to 2.6 K off.

## Check the result

```bash
python3 - <<'EOF'
import struct, array
def load(p):
    b = open(p, 'rb').read()
    n = struct.unpack('<i', b[:4])[0]          # 4-byte face-count header
    a = array.array('f'); a.frombytes(b[4:4+4*n])
    return sorted(a)
for name in ('temperature_max', 'temperature_min', 'temperature_avg'):
    v = load(f'output/quickstart_moon_diurnal/statistics/{name}.bin')
    print(f"{name}: min={v[0]:.1f}  median={v[len(v)//2]:.1f}  max={v[-1]:.1f} K")
EOF
```

Expected (CPU build of this release):

```
temperature_max: min=182.0  median=369.8  max=385.0 K
temperature_min: min=66.8  median=92.6  max=95.0 K
temperature_avg: min=107.8  median=208.2  max=215.7 K
```

These values have simple physical explanations. The equatorial noon maximum of about 385 K is
the radiative balance near 1 AU with a normal-incidence Bond albedo of 0.12. The nighttime value
of about 93 K is the lunar nighttime plateau set by the regolith's thermal inertia. The polar
peak of about 182 K reflects grazing sunlight at the poles.

Results land in `output/quickstart_moon_diurnal/` (about 10 MB). [Tutorial 3](03_outputs.md)
explains every file.

## Things to try

- `mesh.resolution: 4` (5,120 faces) for a sharper map.
- `simulation.timesteps: 480` for a 1.5 h output cadence.
- `thermal.equilibration_orbits: 1` to see the spin-up warning and compare the statistics.
- Open `output/quickstart_moon_diurnal/` in the GUI ([Tutorial 4](04_gui.md)), or convert it for
  ParaView: `python src/python/vtk_converter.py output/quickstart_moon_diurnal output/qs_paraview`.
