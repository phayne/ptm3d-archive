# PTM3D tutorials

Numbered, runnable-verified walkthroughs. Start with the top-level
[TUTORIAL.md](../../TUTORIAL.md) (10-minute overview), then go deeper here.
All commands are run **from the repository root** after building
([INSTALL.md](../../INSTALL.md)) and fetching SPICE kernels
(`python3 scripts/fetch_kernels.py`).

| # | Tutorial | Status |
|---|----------|--------|
| 01 | Bowl-crater radiation gate (`test_bowl_crater`) — covered in [TUTORIAL.md](../../TUTORIAL.md) step 1 | in TUTORIAL.md |
| 02 | [Lunar diurnal run on an icosphere](02_lunar_diurnal_icosphere.md) | verified on v1.0.3 (2026-08-05) |
| 03 | Custom body via Keplerian elements / Horizons | planned |
| 04 | [Crater radiosity: radiation on vs. off](04_crater_radiosity.md) | verified 2026-08-05 (both arms; needs the post-v1.0.3 PTM-94 fix) |
| 05 | GPU acceleration and `--calibrate-gpu` | planned |
| 06 | [Interpreting the outputs](06_interpreting_outputs.md) | verified 2026-08-05 |
| 07 | [The PTM3D GUI, end to end](07_gui_walkthrough.md) | verified 2026-08-07 (panels rendered offscreen from `origin/main`; 5 live-display screenshots pending) |

## Jupyter notebooks

A parallel, notebook-native onboarding track under [`notebooks/`](notebooks/)
for colleagues who work in Jupyter rather than the CLI + doc pair above. Each
notebook is self-contained (environment check, config walkthrough, run,
output reader, plots, interpretation, exercises) and is committed **with its
executed outputs** so the plots render directly on GitHub — re-run it
yourself (`Kernel > Restart & Run All`) rather than trusting a stale render.

| # | Notebook | Body / example | Status |
|---|---|---|---|
| 01 | [`01_moon_diurnal.ipynb`](notebooks/01_moon_diurnal.ipynb) | Moon, `examples/quickstart_moon_diurnal.yaml` (ico3, CPU-only, ~6-8 s) — notebook counterpart of tutorial 02 above | verified 2026-08-07 |
| 02 | [`02_europa_thermal.ipynb`](notebooks/02_europa_thermal.ipynb) | Europa diurnal run, `examples/quickstart_europa_diurnal.yaml` (ico4, 5120 faces, CPU-only, ~6.6 s) — includes a Jupiter eclipse captured in the diurnal window and its thermal-cooling signature; day/night contrast vs. latitude cross-referenced against tutorial 02's Moon numbers | executed end-to-end 2026-08-07; see the notebook's Provenance cell for the exact binary/commit |
| 03 | [`03_asteroid_thermal.ipynb`](notebooks/03_asteroid_thermal.ipynb) | Asteroid thermal model: a generic body on a Keplerian orbit, `examples/quickstart_asteroid_diurnal.yaml` | verified on v1.0.3 (2026-08-07) |
| 04 | [`04_asteroid_spice.ipynb`](notebooks/04_asteroid_spice.ipynb) | Asteroid thermal model on a real SPICE ephemeris: (101955) Bennu, `examples/quickstart_asteroid_bennu.yaml` (real radar shape model + OSIRIS-REx-archived ephemeris, CPU-only, ~30 s/run) — notebook counterpart of tutorial 03 above | executed end-to-end 2026-08-07; see the notebook's Provenance cell for the exact binary/commit |
