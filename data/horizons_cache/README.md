# `data/horizons_cache/` — JPL Horizons small-body cache

This directory is the cache the `generic_body` (small-body) path writes when it
resolves an asteroid or comet through JPL Horizons: one subdirectory per body
holding the fetched `ephemeris.spk`, an auto-generated `metakernel.tm`, and the
`body_config.yaml` derived from the Horizons orbital elements, plus a master
`index.json`.

**It is written by the tool, not by hand.** `src/python/data_manager.py` creates
and updates it; a body that is not in the cache is fetched on first use.

## What is checked into git, and what it is for

Three bodies are committed — 269 Justitia, 623 Chimaera, 13294 Rockox. They are
here as **provenance for published small-body runs**, so the orbit and spin
elements those runs used can be inspected. They are **not runnable inputs on a
fresh clone**, for two independent reasons:

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

## Regenerating a body locally

Fetch the body through the normal `generic_body` path (the GUI's Horizons panel,
or `DataManager` directly). That rewrites `bodies/<body>/` and the matching
`index.json` entry with paths rooted at your own checkout. `data/horizons_cache/`
is auto-detected relative to the directory containing `CMakeLists.txt`, so no
configuration is needed; set the `cache_dir` argument if you want it elsewhere.

The `ema_*` campaign scripts do **not** read `index.json`; they build their paths
from the `PTM` environment variable, so regenerating the cache does not affect
them.
