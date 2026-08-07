# Tutorial 02 — A global lunar diurnal run on an icosphere

**Config:** [`examples/quickstart_moon_diurnal.yaml`](../../examples/quickstart_moon_diurnal.yaml)
**Cost:** ~6 s on a many-core workstation, minutes on a 2-core laptop; <1 GB RAM.

## What this does

Simulates one full lunar day (29.53 Earth days) of surface temperatures over
the entire Moon on a level-3 icosphere (1280 faces, ~480 km face pitch):

1. **Illumination**: for each of 480 timesteps (240 equilibration + 240
   output), shadow rays are traced (using the Embree ray-tracing library) from
   every face toward the Sun's SPICE-computed position, plus Earth reflected
   light and Earth thermal IR.
2. **Thermal**: a 1D Crank-Nicolson conduction column under every face with
   lunar regolith properties (density/conductivity increasing with depth,
   T-dependent k and cp), spun up over one equilibration lunation.
3. **Statistics**: per-face min/mean/max over the output lunation for
   temperature and each flux component.

## Run it

```bash
python3 scripts/fetch_kernels.py --bodies moon   # once, ~44 MB from NAIF
# (the bare command also pulls the 1.1 GB Galilean ephemeris for europa.tm)
./build/bin/ptm_thermal_sim --config examples/quickstart_moon_diurnal.yaml
```

Watch for these lines: `Mesh created with 642 vertices and 1280
faces`, `Initializing SPICE with meta-kernel`, the illumination progress bar,
and `[Statistics] Exported to output/quickstart_moon_diurnal/statistics`.
The program should finish without errors (exit code 0).

## Check the result

```bash
python3 - <<'EOF'
import struct, array
def load(p):
    b = open(p, 'rb').read()
    n = struct.unpack('<i', b[:4])[0]          # 4-byte face-count header
    a = array.array('f'); a.frombytes(b[4:4+4*n])
    return list(a)
for name in ('temperature_max', 'temperature_min', 'temperature_avg'):
    v = sorted(load(f'output/quickstart_moon_diurnal/statistics/{name}.bin'))
    print(f"{name}: min={v[0]:.1f}  median={v[len(v)//2]:.1f}  max={v[-1]:.1f} K")
EOF
```

Verified output (2026-08-05, v1.0.3, CPU-only build):

```
temperature_max: min=182.0  median=369.8  max=385.0 K
temperature_min: min=66.3   median=92.6   max=95.9 K
temperature_avg: min=107.2  median=208.1  max=216.4 K
```

Physical sanity checks:

- **Equatorial noon ~385 K**: radiative balance at ~0.98 AU with Bond albedo
  0.12 gives $T \approx (0.88 \cdot S/\sigma)^{1/4} \approx 387$ K. ✓
- **Night ~93 K**: the classic Diviner-era lunar nighttime plateau (thermal
  inertia of the upper regolith). ✓
- **Polar Tmax ~182 K**: grazing incidence at the poles. ✓

These values are build-dependent: the same table on a pre-v1.0.3 binary read
`min=149.8 median=372.5 max=388.5 K` for `temperature_max`. Check your version
before comparing against the numbers above.

## Things to try

- `resolution: 4` (5120 faces) for a sharper map — still trivially
  laptop-sized (`scripts/ptm_mem_estimate.py --faces 5120 --steps 480`).
- `timesteps: 480` for ~1.5 h output cadence.
- Convert to ParaView:
  `python src/python/vtk_converter.py output/quickstart_moon_diurnal output/qs_paraview`
- Read every output file: [06_interpreting_outputs.md](06_interpreting_outputs.md).
