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

## Jupyter notebooks

A parallel, notebook-native onboarding track under [`notebooks/`](notebooks/)
for colleagues who work in Jupyter rather than the CLI + doc pair above. Each
notebook is self-contained (environment check, config walkthrough, run,
output reader, plots, interpretation, exercises) and is committed **with its
executed outputs** so the plots render directly on GitHub — re-run it
yourself (`Kernel > Restart & Run All`) rather than trusting a stale render.

| Notebook | Body / example | Status |
|---|---|---|
| [`01_moon_diurnal.ipynb`](notebooks/01_moon_diurnal.ipynb) | Moon, `examples/quickstart_moon_diurnal.yaml` (ico3, CPU-only, ~6-8 s) | verified 2026-08-07 — notebook counterpart of tutorial 02 above |
