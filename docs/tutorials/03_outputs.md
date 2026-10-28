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
├── provenance.txt            # which engine produced this, and how — see below
├── config_used.yaml          # verbatim copy of the configuration that was loaded
├── time_steps.txt            # per-step time stamp (+ UTC label column) — see below
├── body_positions.csv        # Sun / primary-body vectors per output step
├── heartbeat.txt             # liveness file for monitoring — see note below
├── statistics/               # per-face min/avg/max over the output window
├── temperature/              # per-timestep snapshots: temperature_0000.bin ...
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
every face by one and injects a bogus first value — a common mistake.

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
  set by indirect heating (see tutorial 04). The solver enforces the geothermal
  floor `(Q_b/εσ)^(1/4)` as a hard lower bound on every layer — 24.04 K on the
  Moon, 27.69 K on Europa — so a value sitting exactly there is the bound rather
  than a converged temperature, and nothing below it can be reported at all. See
  [`KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md) section 2 before using a
  statistic near it.

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
IR dominates in deep shadow and at night. In the tutorial-04 coupled run the
shadowed faces peak at a median **3.3 W/m² indirect solar** but **28.6 W/m²
indirect IR** (measured 2026-08-05 over the 1,422-face shadowed set defined in
[04_crater_radiosity.md](04_crater_radiosity.md), from
`statistics/indirect_{solar,ir}_max.bin`) — the warm rim's thermal emission,
not scattered sunlight, is what heats this crater's floor.

## Per-timestep series

`temperature/temperature_0000.bin` … `temperature_0117.bin` are snapshots at
each output step (same header + float32 format). The index is **zero-based and
zero-padded to four digits**, so a 118-step run ends at `_0117`, not `_00118`;
`metadata.txt` lists the face count, the step count and which components were
exported.

**`provenance.txt` and `config_used.yaml`.** `provenance.txt` is `key=value`,
one per line, like `metadata.txt`, and it answers "which binary produced this
directory, running what?" — engine version, git SHA, build type, whether the
GPU core was compiled in, the compiler, the **md5 of the executable itself**,
the full command line, the body, the mesh source, and the mesh length unit the
engine actually inferred and used. `config_used.yaml` beside it is a byte copy
of the configuration file that was loaded.

The pair is deliberately redundant: the YAML alone would misreport a run
launched with a CLI override such as `--thermal-inertia`, so `command_line`
carries the rest. `config_used.yaml` is the file **as read**, not a dump of the
resolved configuration.

```
schema=ptm3d-provenance-1
engine_version=1.1
engine_git_sha=2436cad9f5
engine_binary_md5=2884834efa55f531771f80b9d354ade6
engine_shared_libraries=false
command_line=./build/bin/ptm_thermal_sim --config moon.yaml --albedo 0.24
body=Moon
mesh_faces=20480
mesh_length_unit=kilometers
```

`engine_shared_libraries` is the honest caveat on the checksum: when it is
`false` — the default build — the executable is the whole engine and its md5
is a complete identity. Under `-DBUILD_SHARED_LIBS=ON` a change confined to one
project library would move that library and leave the executable's md5
unchanged, and the field says so.

`ptm info <run>` prints all of this, and `ptm.Results.provenance()` returns it;
on a run made before the engine wrote the file, the same fields appear under
`not_recorded` rather than being guessed at.

**Timing columns.** In `time_steps.txt`, column 1 is **seconds past the J2000
epoch** (SPICE ephemeris time), *not* elapsed time from the start of the run —
note that the file's own header comment mislabels it. Column 2 is a UTC label
anchored to the configured `simulation.start_time`. In the tutorial-04 control
run (`start_time: 2026-04-05T00:00:00`) the first row reads
`828630100.682384  2026-04-05T03:00:31.497`, and consecutive rows differ by
21714.19 s = the configured ~6 h cadence. For elapsed time, subtract the first
column-1 value; for a calendar axis, use column 2 (or `body_positions.csv`,
whose `time` column carries the same UTC).

**`heartbeat.txt`** is written by the radiation-coupling and re-thermal loops,
plus once at startup. The startup write happens before the output directory is
created, so on a directory's *first* run the file only appears if the run
reaches one of those loops. Its absence is not a failure.

## Converting for visualization

```bash
# ParaView time series (writes statistics.pvd + per-step .vtu):
python src/python/vtk_converter.py output/tutorial04_crater80N_coupled output/t04_paraview

# NetCDF / HDF5 / GeoTIFF:
python src/python/netcdf_converter.py --input output/t04_paraview/statistics.pvd \
       --output output/t04.nc --format pvd
```

Both converters need Python packages the model itself does not: `vtk` for the
first, `vtk` + `netCDF4` for the second. Neither fails silently — each checks
its imports up front and prints the `pip install` line (`vtk_converter.py` also
names the interpreter it is running under).

See [../USER_GUIDE.md](../USER_GUIDE.md) for the full tool list.
