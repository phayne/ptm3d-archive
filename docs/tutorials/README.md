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
colleagues who work in Jupyter rather than the CLI. Each notebook is
self-contained (environment check, config walkthrough, run, output reader,
plots, interpretation, exercises) and is committed **with its executed
outputs** so the plots render directly on GitHub; re-run with
`Kernel > Restart & Run All` to regenerate them.

Every notebook is CPU-only: nothing here needs a GPU, and nothing here competes
for one.

They need a Python environment, which the repository root supplies in one step:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt   # numpy, matplotlib, pyyaml, jupyterlab, spiceypy
jupyter lab docs/tutorials/notebooks/
```

Of those, SpiceyPy is used by notebook 04 alone; the other three are used by all
four.

| # | Notebook | What it covers | Runtime | Beyond a built binary, needs | Status |
|---|---|---|---|---|---|
| 01 | [`01_moon_diurnal.ipynb`](notebooks/01_moon_diurnal.ipynb) | Moon, `examples/quickstart_moon_diurnal.yaml` (ico3, 1,280 faces). Config anatomy, the binary output format, temperature map, diurnal curve, day/night contrast against latitude, orthographic globes, and — section 6 — the same run on real LOLA global topography (res-5) against a smooth-sphere control. The notebook counterpart of tutorial 02 above | 36 s end to end (three engine runs: 2.3 s + 12.7 s + 12.4 s) | Moon SPICE kernels, ~44 MB (`fetch_kernels.py --bodies moon`); for section 6, `fetch_data.py --core` (topography, property maps, albedo maps) | verified 2026-08-09 |
| 02 | [`02_europa_thermal.ipynb`](notebooks/02_europa_thermal.ipynb) | Europa, `examples/quickstart_europa_diurnal.yaml` (ico4, 5,120 faces). A Jupiter eclipse inside the output window and its cooling signature, day/night contrast against the Moon, orthographic globes, a spin-up-length check that separates a numerical artifact from the real sub-Jupiter eclipse notch, and — section 6 — the observed global Bond-albedo map against uniform ice | 34 s end to end (three engine runs: 10.0 s + 3.5 s + 9.7 s) | Europa SPICE kernels, ~1.1 GB including `jup365.bsp` (`fetch_kernels.py --bodies europa`); for section 6, `fetch_data.py --dataset albedo-maps` | verified 2026-08-09 |
| 03 | [`03_asteroid_thermal.ipynb`](notebooks/03_asteroid_thermal.ipynb) | The generic-body path with no ephemeris file: a synthetic asteroid on a Keplerian orbit, `examples/quickstart_asteroid_diurnal.yaml`, run at two thermal inertias to show amplitude damping and phase lag | 21 s end to end (two engine runs, 6.9 s + 7.3 s) | only the SPICE leapseconds kernel (`fetch_kernels.py`, a few hundred kB) — no body ephemeris, no DEM, no maps | verified 2026-08-09 |
| 04 | [`04_asteroid_spice.ipynb`](notebooks/04_asteroid_spice.ipynb) | The generic-body path on a real SPICE ephemeris and a real facet shape: (101955) Bennu at perihelion and aphelion, `examples/quickstart_asteroid_bennu.yaml`, with the Keplerian approximation of tutorial 03 as a third arm | 53 s end to end (three engine runs, ~9.5 s each) | the radar shape model and archived SPK ship in the repository under `data/horizons_cache/`; generic SPICE kernels via `fetch_kernels.py` | verified 2026-08-09 |

Each notebook records the binary md5 and commit that produced its numbers in
its own Version / Provenance cell.
