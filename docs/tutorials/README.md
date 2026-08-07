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

## Notebooks

Interactive Jupyter counterparts to the walkthroughs above — same configs,
same reader code, executed end-to-end with outputs committed so the
numbers/figures are visible without re-running.

| Notebook | Covers | Status |
|---|---|---|
| [`notebooks/02_europa_thermal.ipynb`](notebooks/02_europa_thermal.ipynb) | Europa diurnal run (`examples/quickstart_europa_diurnal.yaml`, ico4, 5120 faces): config walkthrough, output formats, a surface temperature map, and a Jupiter eclipse captured in the diurnal window (direct solar flux → exactly 0 body-wide for ~2.7 h) with its thermal-cooling signature on a single face; day/night contrast vs. latitude, cross-referenced against tutorial 02's Moon numbers. | executed end-to-end 2026-08-07 (CPU-only, ~6.6 s run); see notebook's own Provenance cell for the exact binary/commit |
