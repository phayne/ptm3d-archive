# PTM3D bug ledger

Replaces "flagged for Ada," which lost bugs for months and let one bug class
(a per-body constant hardcoded to Chimaera) bite two different scripts.

**Rule: a fix lands in tracked source, or it becomes a row here. Never a `/tmp` scratch patch.**
A scratch copy in `/tmp` is not a fix — `/tmp` is cleared on reboot. Two shipped tools were
dead on import for days because their only working dependency lived there.

Status: `OPEN` · `BLOCKED` (waiting on another row) · `FIXED` · `BY-DESIGN` (documented, not a bug)

| id | symptom | root cause | status | fixed in |
|---|---|---|---|---|
| EMA-01 | `ema_ti_fit.py` `fit` path cannot run | `disk_p_from_w` and `alb_point` existed **only** in `/tmp/ema_sweep_driver.py`, destroyed with `/tmp`. Not in any tracked file or git branch (searched 2026-07-09). Hapke disk-integrated photometry — **must be re-derived deliberately, never approximated**: a wrong `w → p` mapping silently biases every thermal-inertia fit. Fermi's formula survived in the `ema_campaign_driver.py:45-48` comment and is now transcribed **verbatim** (coefficients unchanged) into the pure tracked module `scripts/ema_hapke.py`; `ema_ti_fit.py` imports it. Pinned by `scripts/test_ema_hapke.py` (22 assertions, incl. the actual on-disk sweep-dir `p` strings). The "opposition-EXCLUDED" label in that comment is **wrong** (the `+w/8` term hardwires bracket = 2); documented in `ema_hapke.py`, not silently changed. | FIXED | 2026-07-09, `scripts/ema_hapke.py` + `test_ema_hapke.py` |
| EMA-02 | Every non-Chimaera TI fit silently uses Chimaera's albedo | `ema_ti_fit.py` hardcoded `HAPKE_DEFAULTS['623']` in the `fit` path, plus a bare `0.143` (= Chimaera's `w`) in `selftest` — three sites, not one. The `fit` path now derives the body from `--sweep-root` via `_body_num_from_sweep_root` (mirrors `ema_ousha_extra_offgrid.py:148`) and **fails loudly** if it can't resolve a `HAPKE_DEFAULTS` entry instead of defaulting to Chimaera. The two `selftest` sites are a **synthetic** machinery test with no `sweep_root`/body in scope, so `0.143` is a fixed Chimaera anchor by construction (documented in-code), not the silent-default bug. | FIXED | 2026-07-09, `ema_ti_fit.py` |
| EMA-03 | Importing `ema_campaign_driver.py` **ran the entire EMA campaign** off the importing process's `argv` | No `if __name__ == "__main__":` guard; lines 267–398 were unguarded top-level code. This is *why* two scripts imported a `/tmp` scratch copy instead. | FIXED | 2026-07-09, `ast`-verified |
| EMA-04 | `ema_camera_noise.py` and `ema_ti_fit.py` raised `FileNotFoundError` on import | Both `importlib`-loaded `/tmp/ema_sweep_driver.py`. `ema_camera_noise.py` is **tracked and was committed** (`ce566d3`) in this broken state. | FIXED | 2026-07-09 — now import the guarded tracked driver |
| EMA-05 | `ema_ousha_extra_offgrid.py` imports three deleted `/tmp` modules | Written as scratch; never tracked. Rescued byte-identical from `/tmp` before the next reboot destroyed it. | OPEN — repoint its imports once EMA-01 is resolved | — |
| EMA-06 | Rockox / Westerwald / Moza / Ghaf "noisy IR" products are noise on a null signal | Their only camera render used the visible-only CNAC config; `TEMP_K`/`THERMAL` are all-zero on-body. Products quarantined 2026-07-06. | DEFERRED (Paul: quarantine, don't re-render) | — |
| PTM-01 | A campaign can silently run on a stale binary | `run_chunked_simulation.py:71-77` `_resolve_build_dir()` prefers `~/dev/heat3d/ptm-build` over the in-tree `$PTM/build`. Documented as a landmine since 2026-07-03 — and it bit anyway. | OPEN — emit a loud warn naming the chosen build dir + its md5 | — |
| PTM-02 | Building from any tree other than `$PTM` silently breaks SPICE init | `SPICE_KERNELS_DIR` is baked in at CMake-configure time (`CMakeLists.txt:348`) and string-concatenated in `illumination_model.cpp`. Breaks `git worktree` builds. | OPEN (unverified this pass) | — |
| PTM-03 | `thermophysical.constant_thermal_inertia > 0` forces `chi = 0` | Intentional: "constant TI in T and depth" cannot hold with χ's T³ term. Emits a notice at `ptm_thermal_sim.cpp:8800`. | BY-DESIGN — this notice is the **precedent** PTM-05 should generalize | — |
| PTM-04 | `a3gt_v2_orchestrator.py:53`, `run_a3gt_batch.py:82` depend on a binary at `/tmp/dtm_to_ply_delaunay` | Same ephemeral-dependency class as EMA-04: a `/tmp` artifact treated as durable. | OPEN | — |
| PTM-06 | **Every Python analysis of `statistics/*.bin` has been off by one face.** The file format is `[int32 n_faces][float32 × n_faces]`, but `ptm-analyze`/`ptm-sanity-check` documented `np.fromfile(p, dtype="<f4")` with *"assert `len == n_faces`"* and *"`size == 4·n_faces`"*. That recipe reads the 4-byte header as a temperature (a denormal ~1e-41, indistinguishable from 0 K) and shifts every face by one index. | Undocumented header. Confirmed on 5/5 runs across **both** the generic-body and planetary paths (Chimaera, Justitia, Ousha, A3GT, Moon3D): `header_int32 == mesh_faces == len(body)` exactly. The documented assertion would have failed on every run ever produced — so it was never run. Found 2026-07-09 while building `scripts/ptm_qa.py`, by the gate flagging a face-count mismatch. | FIXED in docs + tooling (`ptm_qa.py::stat`, `ptm-analyze`, `ptm-sanity-check`). **OPEN: audit existing per-face figures.** Any map built with the old recipe mislabels every face and carries a spurious first value. | 2026-07-09 |
| PTM-05 | Config parameters reach the solver by side channels nobody watches | `applyGenericBodyThermo` forwards `day`/`S`/`year`/`Qb` (`ptm_thermal_sim.cpp:2238-2241`, fixed 2026-07-04) — but `chi` and constant-TI never pass through it at all (`:8650`, `:8780+`). A parameter arriving by an unwatched path is how an 85-hour asteroid day survived a whole campaign. | OPEN — **the fix is one place where every effective parameter resolves and echoes with its provenance** (`configured` vs `body-default-substituted`), not another forwarding line | — |

## The archetype

Nine of this session's failures share one habit: **when something cannot be done as asked, do
something else quietly.** Silent default substitution, silent χ zeroing, a silent stale binary,
a silent wrong albedo, a sweep returning `rc=0` with cells missing, an empty view-factor matrix
no-opping radiation exchange.

**Every silent override, fallback, substitution, or skip must warn or fail. Never proceed quietly.**
`PTM-03`'s notice is what the rest should look like.
