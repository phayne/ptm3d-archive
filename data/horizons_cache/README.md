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
the orbit and spin elements those runs used can be inspected.

**269 Justitia, 623 Chimaera and 101955 Bennu are runnable from a fresh clone.**
Their shape models are tracked (see the policy below), and their `metakernel.tm`
and `index.json` entries carry **repository-root-relative** paths, not absolute
ones.

Justitia's committed kernel was verified against fourteen real Justitia
configurations on 2026-08-31 (the `examples/justitia_*.yaml` set, which is local
to one machine and **not yet tracked** — see PTM-300/PTM-446): all fourteen
resolve every metakernel member, and the committed SPK
(**2033-12-14T23:58:51 → 2036-05-31T23:58:51 UTC, 899.0 days for body
20000269**, one interval) contains every run window *including its
equilibration reach-back*. All fourteen ran to `rc=0` with
`[ephemeris] Keplerian fallback: 0 of N Sun-position epochs`.

Two things a fresh clone still has to supply:

1. **The standard kernels.** `$KERNELS/lsk/naif0012.tls`, `$KERNELS/pck/pck00011.tpc`
   and `$KERNELS/spk/de440s.bsp` are not part of the cache. Fetch them with
   `python3 scripts/fetch_kernels.py`.
2. **The working directory.** A metakernel's `PATH_VALUES` are resolved by
   SPICE's `furnsh_c` against the **current working directory**, not by the
   engine's own path contract — so **launch these runs from the repository
   root.** (The engine resolves `generic_body.spice_metakernel` itself, from any
   cwd; it is the paths *inside* the metakernel that are cwd-relative.)

**13294 Rockox is the remaining exception**: its `index.json` entry still carries
absolute paths from the machine that generated it, and it has no tracked shape.
Treat it as provenance only.

⛔ **Do not regenerate a committed `metakernel.tm`.** `SPKGenerator.create_metakernel()`
emits absolute paths by design (the deep-OneDrive-path and SPICE 78-character
truncation workaround), so regenerating one bakes in a single machine's layout
and the file stops resolving anywhere else. The committed metakernels are
hand-written relative on purpose; each says so in its own header. This is not
hypothetical — it is how the fourteen Justitia configurations came to name a
directory that does not exist (PTM-446), and a stray regeneration was found
sitting in the working tree on 2026-08-31.

## Shape-model policy

**A shape model that a committed configuration loads is a production input, so
it is tracked.** Concretely, every such file has all four of:

1. **The file itself, tracked in git.**
2. **A named `!` exception in `.gitignore`**, by exact filename. `.gitignore`'s
   `*.obj` is a C/C++ object-file rule that also swallows every Wavefront shape
   model in the tree (PTM-303). The exception is written out rather than the file
   force-added, because a force-add is invisible to the next person and silently
   stops applying the day someone re-adds the file without `-f`.
3. **A machine-readable provenance sidecar** next to it, `<stem>_provenance.json`
   — source, archive, version/model id, URL, md5, sha256, byte count, vertex and
   face counts, units, and the date retrieved. `gui_core.body_fetch.shape_provenance()`
   reads exactly this name.
4. **A `PROVENANCE.md` in the body directory** explaining what the file is, what
   it supersedes, what was measured and what was not.

The tracked set, as of 2026-08-31:

| body | file | what it is |
|---|---|---|
| (269) Justitia | `shape.obj` | DAMIT model 4966 (asteroid id 2858), the original |
| (269) Justitia | `shape_reconstructed_20260704.obj` | the superseded 2026-07-04 hand reconstruction (PTM-303), geometrically identical to the original — max vertex deviation 0.000000000 km |
| (623) Chimaera | `shape.obj` | DAMIT model 9110, the original (byte-identical to a 2026-08-31 re-download) |
| (623) Chimaera | `shape_refined.obj` | 2-pass Loop subdivision of the above; **this** is what `body_config.yaml` loads, and its generator is not in this repository |
| (101955) Bennu | `g_06370mm_alt_obj_0000n00000_v021.obj` | OSIRIS-REx OLA v21 global model |
| (101955) Bennu | `bennu_radar_shape.obj` | the superseded Nolan et al. (2013) radar model |

Superseded models are kept, not deleted, for the same reason in every case: they
are the geometry behind results already published from this repository, and an
artifact that has been deleted cannot be re-checked.

**Adding a body's shape is a four-file act**, and skipping any one of them is the
failure this policy exists to prevent: a tracked mesh with no provenance is a
number no one can re-derive, and an untracked mesh is one copy on one machine.

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

Bennu's shape models follow the same policy as the rest — tracked, excepted by
name in `.gitignore`, sidecar plus `PROVENANCE.md`.

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
