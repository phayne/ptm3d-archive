# (101955) Bennu — what is in this directory, and what runs use

## `bennu_radar_shape.obj` — live

Nolan et al. (2013) radar + lightcurve shape model, PDS Small Bodies Node,
`urn:nasa:pds:ast-bennu.radar.shape-model::1.1`. 1348 vertices, 2692 facets,
already in km. `examples/quickstart_asteroid_bennu.yaml` loads it directly
(`generic_body.shape.mesh_file`). Unchanged.

## `ephemeris.spk` — PROVENANCE ONLY since 2026-08-27, not loaded by any run

This is JPL small-body solution **76** for Bennu, dated 2013-02-01:

| | |
|---|---|
| identity | byte-identical to NAIF's `bennu_refdrmc_v1.bsp` **and** to `sb-101955-76.bsp` — md5 `1c1b36831546de7434af83602756b962` for all three, 25,600 bytes |
| contents | body 2101955 relative to the Sun (10) |
| coverage | 2015-01-01 to 2023-05-31 (TDB) |
| sources | `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/bennu_refdrmc_v1.bsp` and `https://naif.jpl.nasa.gov/pub/naif/ORX/kernels/spk/sb-101955-76.bsp` |

It is retained, under its original name, for three reasons: it is the ephemeris
behind every Bennu number published from this repository before 2026-08-27; it
is the ephemeris the `asteroid_bennu_spice_cpu` validation arm was banked on
until that date; and `docs/tutorials/notebooks/04_asteroid_spice.ipynb`
furnishes this exact path in its own cells, so deleting or renaming it would
break a shipped, executed notebook.

**It is not on any engine path.** `data/kernels/bennu.tm` no longer names it.

## What replaced it

`data/kernels/spk/sb-101955-118_long.bsp` — JPL solution **118**, the final
OSIRIS-REx reconstruction (Farnocchia et al. 2021,
doi:10.1016/j.icarus.2021.114594), covering **1900-01-01 to 2135-10-01**.
80.7 MB, so it is fetched rather than committed:

```bash
python3 scripts/fetch_kernels.py --bodies bennu
```

`data/kernels/bennu.tm` carries the URL, the SHA-256, the file's own
description, and the reason NAIF's 2015–2023 copy of the same solution was not
used.

## How much the two differ

Measured on this machine, 2026-08-27, engine `51bf1688b0`, everything except
the kernel held fixed.

**Inside the overlap** (`spkdiff`, 1000 samples over 2015-01-01 → 2023-05-30):
maximum position difference 31.97 km, mean 20.91 km — relative 1.7e-7. Over the
tutorial's own window on 2019-01-10: 13.93 km, of which 6.36 km is heliocentric
distance, so insolation moves by 9.5e-8 and the solar direction by 5.4e-6 deg.
On the `asteroid_bennu_spice_cpu` arm that is 7 of 16 output files changed, at
most **2.14e-4 K** in temperature and 2.44e-4 W/m² in direct solar flux — one to
a few float32 ULP. Physically nothing; byte-identity, everything.

**Outside the overlap** is the reason for the change. At 2026-01-01 the old file
has no data, and `IlluminationModel::getSunPosition`
(`src/cpp/illumination/illumination_model.cpp:2818`) falls back to the config's
Keplerian elements **without printing anything** — the banner still says
`Ephemeris: SPICE` and the run still exits 0. Measured against the real
ephemeris at that epoch, that fallback was wrong by:

| quantity | error |
|---|---|
| heliocentric distance | 0.0919 AU (13.75 million km), 1.1370 AU vs 1.0451 AU |
| solar direction in the body frame | 23.5 deg |
| insolation | +18.3% |
| direct solar flux | up to 196 W/m² (mean 144) |
| surface temperature | up to **23.9 K** (mean 16.3 K on `temperature_max`) |

A *missing* kernel, by contrast, is loud: `SPICE(NOSUCHFILE)`, `FATAL: SPICE
initialization failed`, rc=1 — also measured. It is only the out-of-coverage
case that is silent.
