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
