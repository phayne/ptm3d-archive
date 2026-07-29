# Tutorial 06 — Interpreting the outputs

What every file in a PTM3D output directory means and how to read it,
using the runs from tutorials [02](02_lunar_diurnal_icosphere.md) and
[04](04_crater_radiosity.md).

## Directory layout

From the tutorial-04 coupled run:

```
output/tutorial04_crater80N_coupled/
├── mesh.ply                  # the simulated mesh (face order matches all .bin files)
├── metadata.txt              # faces, time_steps, exported components
├── time_steps.txt            # per-step elapsed seconds (+ UTC label column)
├── body_positions.csv        # Sun / primary-body vectors per output step
├── heartbeat.txt             # liveness file for long runs (monitoring)
├── statistics/               # per-face min/avg/max over the output window
├── temperature/              # per-timestep snapshots: temperature_00001.bin ...
├── illumination/             #   total absorbed flux per step
├── direct_solar/             #   direct solar component
├── primary_reflected/        #   reflected light from the primary body
├── primary_ir/               #   thermal IR from the primary body
├── indirect_solar/           #   scattered-sunlight indirect flux  (radiation runs)
└── indirect_ir/              #   thermal-IR indirect flux          (radiation runs)
```

## Binary format: 4-byte header + float32 per face

Every `.bin` (statistics and per-timestep alike) is:

| bytes | content |
|---|---|
| 0–3 | `int32` (a 4-byte integer, little-endian byte order): number of faces `n` |
| 4–(4+4n) | `n × float32` (single-precision, 4 bytes each), one value per mesh face, in mesh face order |

Reader (Python standard library only — no extra packages):

```python
import struct, array

def load_bin(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<i', b[:4])[0]
    a = array.array('f')
    a.frombytes(b[4:4 + 4*n])
    assert len(a) == n, f"truncated file: {len(a)} of {n} values"
    return a

t = load_bin('output/quickstart_moon_diurnal/statistics/temperature_max.bin')
print(len(t), 'faces; global max', max(t), 'K')
```

Skipping the 4-byte header (e.g. a bare `np.fromfile(..., dtype='f4')`) shifts
every face by one and injects a bogus first value — the classic mistake.

Units: temperatures in **K**; all flux files in **W/m²** (absorbed, i.e. after
albedo).

## statistics/: Tmax vs Tavg vs Tmin

For each quantity `X`, `statistics/X_{min,avg,max}.bin` are the per-face
extremes/mean **over the output window only** (equilibration steps are
excluded). Interpretation differs by moment:

- **`temperature_max`** — the local-noon (or peak-flux) temperature. Sensitive
  to instantaneous illumination: a face that catches the Sun through a notch
  for one timestep records it here. When hunting artifacts, always ask whether
  an anomaly is max-only.
- **`temperature_avg`** — the time-mean, which controls ice stability and
  subsurface heat content. Robust to single-step glitches; the quantity to
  trust first.
- **`temperature_min`** — the pre-dawn (or eternal-shadow) floor. In PSRs with
  radiation off this approaches the geothermal floor; with radiation on it is
  set by indirect heating (see tutorial 04).

Statistics exist for `temperature`, `illumination` (total), `direct_solar`,
`primary_reflected`, `primary_ir`, and — on radiation runs — `indirect_solar`
and `indirect_ir`. `ice_depth.bin` (ice-stability runs) holds the ice table
depth in meters.

## Band-separated indirect flux (radiation runs)

With two-band radiosity (`radiation.band_separated`, the default on coupled
runs), the indirect flux a face **absorbs** is split by where the energy came
from:

- **`indirect_solar`** — sunlight scattered off other terrain, absorbed with
  the solar absorptivity (1 − Bond albedo);
- **`indirect_ir`** — thermal emission from other terrain, absorbed with the
  emissivity ε.

The two sum to the total indirect flux used by the thermal solve. Their ratio
is diagnostic: scattered sunlight matters near sunlit walls at noon, thermal
IR dominates in deep shadow and at night. In the tutorial-04 run the shadowed
faces peak at a median 2.8 W/m² indirect solar but 23.9 W/m² indirect IR —
the warm rim's thermal emission, not scattered sunlight, is what heats this
crater's floor.

## Per-timestep series

`temperature/temperature_00001.bin` … `temperature_00118.bin` are snapshots at
each output step (same header + float32 format). `time_steps.txt` column 1
gives each step's elapsed seconds from simulation start; `metadata.txt` lists
which components were exported. Note the UTC label column in
`time_steps.txt`/`body_positions.csv` is measured from the J2000 epoch, not
the configured start date — use the elapsed-seconds column for timing.

## Converting for visualization

```bash
# ParaView time series (writes statistics.pvd + per-step .vtu):
python src/python/vtk_converter.py output/tutorial04_crater80N_coupled output/t04_paraview

# NetCDF / HDF5 / GeoTIFF:
python src/python/netcdf_converter.py --input output/t04_paraview/statistics.pvd \
       --output output/t04.nc --format pvd
```

See [../USER_GUIDE.md](../USER_GUIDE.md) for the full tool list.
