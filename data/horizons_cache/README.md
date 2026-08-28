# `data/horizons_cache/` — JPL Horizons small-body cache

This directory is the cache the `generic_body` (small-body) path writes when it
resolves an asteroid or comet through JPL Horizons: one subdirectory per body
holding the fetched `ephemeris.spk`, an auto-generated `metakernel.tm`, and the
`body_config.yaml` derived from the Horizons orbital elements, plus a master
`index.json`.

**It is written by the tool, not by hand.** `src/python/data_manager.py` creates
and updates it; a body that is not in the cache is fetched on first use.

## What is checked into git, and what it is for

Four bodies are committed — 269 Justitia, 623 Chimaera, 13294 Rockox, and
101955 Bennu. They are here as **provenance for published small-body runs**, so
the orbit and spin elements those runs used can be inspected. They are **not
runnable inputs on a fresh clone**, for two independent reasons:

1. **The shape meshes are not distributed.** Each `body_config.yaml` points at a
   `shape*.obj` (for example `shape_refined.obj`) that `.gitignore` excludes.
2. **`index.json` and every `metakernel.tm` carry absolute paths from the
   machine that generated them.** `PATH_VALUES` in the metakernels and the
   `config_path` / `spk_path` / `metakernel_path` / `shape_path` entries in
   `index.json` are machine-specific strings; they do not resolve anywhere else,
   and the kernel subdirectories they name (`$KERNELS/lsk`, `$KERNELS/pck`,
   `$KERNELS/spk`) are not part of the cache at all.

The failure mode is benign: `DataManager.is_cached()` tests those paths with
`os.path.exists()`, finds nothing, and treats the body as uncached, so the next
run re-queries Horizons and rewrites the entry with paths valid for *your*
machine. Nothing silently uses a wrong ephemeris. But do not read the committed
absolute paths as something you can follow, and do not hand a committed
`metakernel.tm` to SPICE directly — regenerate it.

Bennu is the exception to both. It is not a Horizons-fetched entry at all — it
has no `body_config.yaml`, no `metakernel.tm`, and no `index.json` record,
because Horizons refuses to generate an SPK for it. Its directory holds three
files, only one of which a run loads:

| file | status |
|---|---|
| `g_06370mm_alt_obj_0000n00000_v021.obj` | **live** — the OSIRIS-REx OLA v21 global shape model, 6.37 m sampling, 49,152 facets, km |
| `bennu_radar_shape.obj` | archived — the pre-encounter Nolan et al. (2013) radar model, superseded 2026-08-28 |
| `ephemeris.spk` | archived — JPL solution 76, superseded 2026-08-27 |

**Neither archived file is on any engine path**, and both are kept deliberately:
they are what earlier Bennu numbers from this repository were computed on, and
`docs/tutorials/notebooks/04_asteroid_spice.ipynb` names both paths in its own
cells. As of 2026-08-27 `data/kernels/bennu.tm` loads
`spk/sb-101955-118_long.bsp`, which `scripts/fetch_kernels.py` downloads; as of
2026-08-28 `examples/quickstart_asteroid_bennu.yaml` loads the OLA shape model.
See `bodies/101955_Bennu_(1999_RQ36)/PROVENANCE.md` for the checksums, the
citations, the closed-form mesh validation and the measured differences.

Note that the shape models are tracked despite `.gitignore`'s `*.obj` rule (a
C/C++ object-file pattern that also matches every Wavefront shape model in the
tree). They are excepted by name, not force-added — see that rule's comment.

## Regenerating a body locally

Fetch the body through the normal `generic_body` path (the GUI's Horizons panel,
or `DataManager` directly). That rewrites `bodies/<body>/` and the matching
`index.json` entry with paths rooted at your own checkout. `data/horizons_cache/`
is auto-detected relative to the directory containing `CMakeLists.txt`, so no
configuration is needed; set the `cache_dir` argument if you want it elsewhere.

The `ema_*` campaign scripts do **not** read `index.json`; they build their paths
from the `PTM` environment variable, so regenerating the cache does not affect
them. They **do** read `body_config.yaml` — see the albedo note below.

## `surface.albedo` is a BOND albedo, and the file says so

Every `body_config.yaml` written before 2026-08-27 carried SBDB's **geometric**
albedo p_V in `generic_body.surface.albedo`, which the engine reads as a **Bond**
albedo: `thermal_model.cpp:2906` computes `1 - albedo` and multiplies the solar
constant by it. The writer warned about it in a log and wrote the number anyway,
and the warning did not hold — a generated (101955) Bennu configuration reached a
run with `surface.albedo: 0.044`, Bennu's p_V, against a Bond albedo of 0.0118.

Since then, every cached body carries the conversion and says which convention it
is in:

```yaml
  surface:
    albedo: 0.014526                    # the BOND albedo the engine consumes
    albedo_type: bond                   # so no reader has to guess
    albedo_geometric_pv: 0.037          # what SBDB published
    albedo_slope_parameter_g: 0.15      # the G used, and whether it was assumed
    albedo_provenance: "Bond albedo A_B = p_V x q, q = 0.290 + 0.684 G ..."
```

`A_B = p_V q` with `q = 0.290 + 0.684 G` (Bowell et al. 1989, *Asteroids II*;
Harris 1998, *Icarus* 131, 291). `scripts/fix_cached_body_albedo.py` performs the
rewrite, is idempotent, and leaves a file that already says `albedo_type: bond`
alone. A file with **no** `albedo_type` predates the change and holds p_V; the
GUI converts it on read and says it did.

⚠️ **One consumer wants the other convention.** `scripts/ema_campaign_driver.py`
reads this same field (`albedo=float(gb['surface']['albedo'])`) and uses it as the
**geometric** albedo p — deliberately: its Hapke single-scattering albedos `w` are
calibrated to reproduce those p values, and the absorbed-flux albedo it sets is
`thermophysical.albedo_hapke_w`. Rewriting a cache that campaign reads would
silently change the number it is built on. That is why
`fix_cached_body_albedo.py` defaults to **this repository's** cache and requires
`--cache-dir` for any other, and why the EMA campaign's own cache (under the
OneDrive `PTM` tree) has deliberately **not** been rewritten.
