# PTM3D tutorials

Numbered, runnable walkthroughs. Start with the top-level
[TUTORIAL.md](../../TUTORIAL.md) (10-minute overview), then go deeper here.
All commands are run **from the repository root** after building
([INSTALL.md](../../INSTALL.md)) and fetching SPICE kernels
(`python3 scripts/fetch_kernels.py`).

| # | Tutorial | What it covers | Runtime | Beyond a built binary, needs | Status |
|---|---|---|---|---|---|
| 01 | Bowl-crater radiation gate (`test_bowl_crater`) — in [TUTORIAL.md](../../TUTORIAL.md) step 1 | The radiosity solver checked against the Ingersoll & Svitek analytic bowl-crater solution: the correctness gate to run before trusting any radiation result | seconds | nothing (no SPICE, no data) | in TUTORIAL.md |
| 02 | [Lunar diurnal run on an icosphere](02_lunar_diurnal_icosphere.md) | One lunation over the whole Moon on a 1,280-face icosphere: config anatomy, the run log, per-face statistics, and the numbers to expect | ~6 s engine (48-core workstation); minutes on a 2-core laptop | Moon SPICE kernels, ~44 MB (`fetch_kernels.py --bodies moon`) | verified on v1.0.3 (2026-08-05) |
| 03 | Custom body on a Keplerian orbit — see notebook [03](notebooks/03_asteroid_thermal.ipynb) | The generic-body path driven by orbital elements and a spin state, with a thermal-inertia comparison. Written as a notebook rather than a markdown tutorial; there is no separate `03_*.md` | see the notebook table below | — | covered by notebook 03 |
| 04 | [Crater radiosity: radiation on vs. off](04_crater_radiosity.md) | Two otherwise identical runs on the shipped 12,800-face polar crater, differenced to isolate scattered sunlight and rim thermal IR on a shadowed floor | 7.8 s (control) + 14.1 s (coupled, including the view-factor build) | shipped mesh `data/meshes/lunar_crater_80N.ply`; writes a ~35 MB view-factor file | verified 2026-08-05 (both arms; needs a build newer than the v1.0.3 tag) |
| 05 | GPU acceleration — see [INSTALL.md](../../INSTALL.md#gpu-accelerated-build-opencl-capable-hosts) | Building with `-DPTM_ENABLE_GPU=ON`, the one-time `--calibrate-gpu` host profile, and `thermal.use_gpu`. Covered by INSTALL.md and the README; no separate tutorial, because every tutorial here is deliberately CPU-only and portable | — | an OpenCL device | documented in INSTALL.md, no dedicated tutorial |
| 06 | [Interpreting the outputs](06_interpreting_outputs.md) | Every file in an output directory, the binary formats, the `time_steps.txt` column convention, and which files appear only with radiation enabled | reading only — uses the outputs from 02 and 04 | the outputs from tutorials 02 and 04 | verified 2026-08-05 |
| 07 | [The PTM3D GUI, end to end](07_gui_walkthrough.md) | The same ground as the command-line tutorials through the graphical front end: config panels, preflight, view-factor cost estimate, console, results browser | ~4 s engine; a few minutes of clicking | `PyQt6`; `pyvista` + `pyvistaqt` and `matplotlib` strongly recommended | verified 2026-08-07 (5 live-display screenshots pending) |

## Jupyter notebooks

A parallel, notebook-native track under [`notebooks/`](notebooks/) for
colleagues who work in Jupyter rather than the CLI. Each one checks its
prerequisites, runs the engine, and reaches its first figure inside ten lines of
code; the config walkthrough, physical interpretation and exercises follow.
They are committed **with their executed outputs**, so the plots render directly
on GitHub; re-run with `Kernel > Restart & Run All` to regenerate them.

Start with 01. It introduces the run reader, the on-disk format and the
color-scale convention that the other three use without re-deriving them.

Every notebook is CPU-only: nothing here needs a GPU, and nothing here competes
for one.

They need a Python environment and the repository's own `ptm` package, which
carries the run reader (`ptm.Results`), the engine launcher (`ptm.run`) and the
plotting (`ptm.plots`) the notebooks use throughout:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt   # numpy, matplotlib, pyyaml, jupyterlab, spiceypy
python3 -m pip install -e .                  # the ptm package
jupyter lab docs/tutorials/notebooks/
```

`pip install -e .` brings numpy, matplotlib and PyYAML with it, so on a bare
environment it is the only line that is strictly required; `requirements.txt`
adds JupyterLab and SpiceyPy. Of those, SpiceyPy is used by notebook 04 alone.

Each notebook's first cell checks for the package, the binary and the data it
needs, and names the command that fixes whatever is missing.

| # | Notebook | What it covers | Runtime | Beyond a built binary, needs | Status |
|---|---|---|---|---|---|
| 01 | [`01_moon_diurnal.ipynb`](notebooks/01_moon_diurnal.ipynb) | Moon, `examples/quickstart_moon_diurnal.yaml` (ico3, 1,280 faces). Peak temperature on globes, config anatomy, the on-disk format, the color scale a planetary temperature field needs, the diurnal curve and its nighttime plateau, day/night contrast against latitude, and — section 9 — the same run on real LOLA topography (res-5) against a smooth-sphere control. The notebook counterpart of tutorial 02 above | 44 s end to end (three engine runs: 5.1 s + 13.5 s + 13.4 s) | the `ptm` package (`pip install -e .`); Moon SPICE kernels, ~44 MB (`fetch_kernels.py --bodies moon`); for section 9, `fetch_data.py --core` (topography, property maps, albedo maps) | verified 2026-08-09 |
| 02 | [`02_europa_thermal.ipynb`](notebooks/02_europa_thermal.ipynb) | Europa, `examples/quickstart_europa_diurnal.yaml` (ico4, 5,120 faces). A Jupiter eclipse inside the output window and its cooling signature, why Europa's night stays warm relative to the Moon's, a spin-up-length check that separates a numerical artifact from the real sub-Jupiter eclipse notch, and — section 8 — the observed global Bond-albedo map against uniform ice, tested against the $(1-A)^{1/4}$ law | 39 s end to end (three engine runs: 11.2 s + 5.3 s + 10.6 s) | the `ptm` package (`pip install -e .`); Europa SPICE kernels, ~1.1 GB including `jup365.bsp` (`fetch_kernels.py --bodies europa`); for section 8, `fetch_data.py --dataset albedo-maps` | verified 2026-08-09 |
| 03 | [`03_asteroid_thermal.ipynb`](notebooks/03_asteroid_thermal.ipynb) | The generic-body path with no ephemeris file: a synthetic asteroid on a Keplerian orbit, `examples/quickstart_asteroid_diurnal.yaml`, run at two thermal inertias to show amplitude damping and phase lag, plus how to confirm a thermophysical edit reached the solver when no log line reports it | 47 s end to end (two engine runs, 19.9 s each) | the `ptm` package (`pip install -e .`) and only the SPICE leapseconds kernel (`fetch_kernels.py`, a few hundred kB) — no body ephemeris, no DEM, no maps | verified 2026-08-09 |
| 04 | [`04_asteroid_spice.ipynb`](notebooks/04_asteroid_spice.ipynb) | The generic-body path on a real SPICE ephemeris and a real facet shape: (101955) Bennu at perihelion and aphelion, `examples/quickstart_asteroid_bennu.yaml`, with the shape/spin/ephemeris provenance written down and the Keplerian approximation of tutorial 03 as a third arm | 122 s end to end (three engine runs, ~31 s each) | the `ptm` package (`pip install -e .`); the radar shape model and archived SPK ship in the repository under `data/horizons_cache/`; generic SPICE kernels via `fetch_kernels.py` | verified 2026-08-09 |

The runtimes above were measured on a 48-core workstation against a default
CPU-only build; the portable `-DPTM_ARCH_NATIVE=OFF` build in INSTALL.md's TL;DR
has AVX disabled and runs the engine roughly 2-3x slower.

To record which binary and commit produced a given set of numbers, ask the
package rather than the notebook:

```bash
python3 -c "from ptm import run; print(run.binary_provenance())"
```
