# RUNBOOK — two-band radiosity validation & paper re-runs

**Run these in your own terminal**, not the dashboard chat agent (that agent is hard-blocked from
launching `ptm_thermal_sim` by the GPU-wedge→power-off guard in `~/.claude/dashboard/server.py:83`;
your terminal is unaffected). Prepared by Iris for Paul, 2026-07-12.

## Provenance (verified)
- Branch **`feat/two-band-radiosity`**, HEAD **`f5c455e`** (two-band committed). Parent **`e50770b`** = pre-change.
- NEW binary already built: `build/bin/ptm_thermal_sim` md5 `ef4c343f` (clean f5c455e).
- ⚠️ Working tree has **held edits restored** (aniso-beaming, buried-ice) — a plain rebuild in the
  main tree would fold them in. To build the OLD binary cleanly, use a **git worktree** (below) so the
  main tree is never disturbed.
- Build recipe: `DEVELOPER_DIR=/Library/Developer/CommandLineTools /usr/local/bin/cmake --build build --target ptm_thermal_sim -j8`.
- `git` here = `/Library/Developer/CommandLineTools/usr/bin/git` (the `/usr/bin/git` shim is broken).

---

## E2 — Moon3D do-no-harm (small, CPU, SAFE — no wedge risk)
**Goal:** prove default-OFF (`band_separated:false`) on the production radiosity path is byte-identical
to pre-change. Fermi says this holds *by construction* on H-matrix/Jacobi/Σ≤1 paths; this is the
empirical confirmation.

Config (Ada staged): `~/dev/heat3d/e2_twoband/e2_crater.yaml` — radiation ON, **CPU** (`use_gpu:false`
throughout), 48 steps, `incidence_angle_albedo:false`. Needs `crater.ply` + a small VF `crater.vf`.

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
PTM=/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm
E2=/Users/paha3326/dev/heat3d/e2_twoband
NEW=$PTM/build/bin/ptm_thermal_sim            # ef4c343f, clean f5c455e (reuse as-is)

# 0. mesh: crater.ply is ABSENT (verified 2026-07-12 — Ada). e2_crater.yaml's import_file points at it,
#    so it MUST be generated first. Build a small synthetic bowl (Moon radius; ~12.5k faces => quick CPU
#    sparse VF). crater_mesh_generator is a mesh tool, NOT a sim — safe to run.
ls -la $E2/crater.ply    # expect: "No such file" -> build it on the next line
$PTM/build/bin/crater_mesh_generator \
    --diameter 1000 --depth-ratio 0.2 --latitude 85 --resolution 40 \
    --body-radius 1737400 --surrounding 2.0 --output $E2/crater.ply
#    crater.vf is then built INLINE on the first ($OLD) run below — e2_crater.yaml has use_hmatrix:false
#    + use_aca:false, so it's a small SPARSE VF over ~12.5k faces: fast, CPU, zero wedge risk.
#    (CLI resolved from `crater_mesh_generator --help`; defaults are Moon: body-radius 1737400 m.)

# 1. OLD binary from the parent commit, in an isolated worktree (main tree untouched):
$GIT -C $PTM worktree add /tmp/ptm-old e50770b
( cd /tmp/ptm-old && /usr/local/bin/cmake -B build -DCMAKE_BUILD_TYPE=Release . \
    && /usr/local/bin/cmake --build build --target ptm_thermal_sim -j8 )
OLD=/tmp/ptm-old/build/bin/ptm_thermal_sim

# 2. build crater.vf once (small mesh → quick). Run OLD, then NEW (default-OFF), to separate dirs:
$OLD --config $E2/e2_crater.yaml 2>&1 | tee $E2/run_old.log      # writes .../out_old (edit output.directory)
$NEW --config $E2/e2_crater.yaml 2>&1 | tee $E2/run_new.log      # band_separated defaults OFF

# 3. byte-identity + Fermi's log check:
diff -r $E2/out_old/statistics $E2/out_new/statistics && echo "E2 PASS: byte-identical"
grep -i "rows *> *1 renormalized\|View-factor row sums" $E2/run_new.log   # expect "0/N" on this small sparse mesh
```
**Expected:** byte-identical statistics (PASS). If `rows>1` is **nonzero**, that gray output *legitimately*
differs — it's the PTM-08 GS row-norm fix, not a two-band leak (note it, don't chase it).
**Optional toggle check:** rerun NEW with `radiation.band_separated: true` (+ `PTM_SOLAR_ALBEDO=0.12`) →
the shadow should warm slightly vs OFF, confirming the switch is live.
Cleanup: `$GIT -C $PTM worktree remove /tmp/ptm-old`.

---

## E3 — Diviner Haworth SE-rim (HEAVY — VF rebuild + 2 seasonal runs; wedge-aware)
**Goal:** does two-band do-no-harm + modestly warm the PSR vs Diviner? **Frame it as a few-K warming,
NOT closing the 10–20 K cold bias** — that bias is IR/equilibration-duty-cycle-dominated, which two-band
does not touch (Fermi; memories `psr_cold_bias_equilibration`, `vf_psr_underradiation_todo`).

Resolved artifacts (all present on disk):
- config `examples/divval_sunlit_fix.yaml` (ring 25→200, eq=25 seasonal).
- mesh `data/dem/validation/meshes/sunlit_radcore_r25.ply` (99,511 f) + `sunlit_ring_25_200.ply` (954,404 f) ≈ 1.05 M.
- Diviner `data/diviner_seasonal/polar_south_80_{summer,winter}_{max,avg}_polar.npy` (Williams 2019).

⚠️ **Landmines:**
1. **VF must be REBUILT** (`sunlit_radcore_r25.vf` absent — verified 2026-07-12). It is built **CORE-ONLY**:
   the 99,511-face `sunlit_radcore_r25.ply` core is the radiosity/VF mesh; the 954,404-face
   `sunlit_ring_25_200.ply` is a **shadow-only BVH occluder** (`mesh.occluder_file`), NOT part of the VF —
   the sim "runs entirely on `mesh` (the core)" (`src/cpp/main/ptm_thermal_sim.cpp:9507`). So the VF matrix
   is **99,511 × 99,511**. Build it **DETACHED + monitored** with H-matrix + ACA forced (already in the YAML)
   **plus `PTM_HMATRIX_ACA_CAP=60000`** — see step (a).
   ⚠️ **Do NOT gate the confirm on "compact = tens of MB." That criterion is a trap here.** The sibling
   *same-r25-class* `haworth_radcore_r25.vf` (a 99k-face core) is **5.9 GB** on disk and `haworth_radcore_r32c.vf`
   is **6.0 GB** (both PRESENT, not evicted — and for the **wrong site**, Haworth PSR not the sunlit SE rim, so
   don't reuse them). And the A3GT "7 MB compact baseline" was later proven **EMPTY (nnz=0 → silent
   no-radiation)** while its real VF was 4.74 GB. So a *correct, populated* sunlit VF may well be **multi-GB**,
   and a tens-of-MB result is more likely an **empty/degenerate** VF (silent no-coupling) than a success.
   **Confirm CORRECTNESS, not smallness** (step a): matrix dim == 99,511, non-empty coupling, and **no**
   `Empty view factor matrix uploaded` / `does not match mesh face count` warning on first load.
   `# CONFIRM:` the *expected* size for the shallower sunlit core is unknown until built — Haworth's 5.9 GB is
   a deeper bowl (denser mutual visibility); record the actual `ls -la` size, don't assume it.
2. **Albedo gate:** the YAML's `incidence_angle_albedo:true` flipped meaning on 2026-07-02 (`e9c6946`).
   **Set `incidence_angle_albedo: false`** to reproduce the original Hayne(2017) angular-albedo treatment.
3. Two-band here runs **CPU** (GPU kernel won't engage: init-order + H-matrix→CPU) — budget wall-clock accordingly.
4. Per-timestep incidence-albedo update is **deferred** (static base albedo, warns) → an expected ~0.3–1 K
   monotone-cold approximation. Optional bracket: rerun two-band at a flux-weighted-grazing flat albedo
   (~0.20) — the floor-T delta bounds the approximation (<1 K predicted).

```bash
# ---- E3 artifacts (self-contained; re-declare so this block runs standalone) ----
PTM=/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm
E2=/Users/paha3326/dev/heat3d/e2_twoband            # reuse the E2 dir as a scratch/log location
MESHDIR=$PTM/data/dem/validation/meshes
VF=$MESHDIR/sunlit_radcore_r25.vf
CFG=$PTM/examples/divval_sunlit_fix.yaml            # band_separated defaults OFF => this is ALSO the Run-1/gray config
# NB: the VF is pure GEOMETRY (albedo-independent), so ONE build serves both the gray and two-band runs.
#     If this launch will ALSO serve as Run 1, first apply landmine 2 (set incidence_angle_albedo:false).

# a. Build the COMPACT view factors. There is NO build-VF-and-exit flag: ptm_thermal_sim computes + saves the
#    VF in an EARLY phase (BEFORE illumination), then continues into the run. The compact H-matrix/ACA path is
#    forced by the YAML (use_hmatrix:true, use_aca:true, aca_tolerance:0.001, aca_max_rank:50) PLUS the env
#    PTM_HMATRIX_ACA_CAP=60000 — WITHOUT that env the ~100k-core floor<->wall blocks ZERO OUT instead of
#    ACA-compressing (=> silent no-coupling). Provenance: this is exactly how the production Diviner-validation
#    driver builds these *_radcore_r25.vf files — scripts/divval_orchestrate.py:16 (RUN_ENV) + :85-103 (radiation YAML).
rm -f "$VF"                                          # force a clean build; a stale/partial .vf is LOADED, not rebuilt
nohup env PTM_HMATRIX_ACA_CAP=60000 \
     "$PTM/build/bin/ptm_thermal_sim" --config "$CFG" \
     > "$E2/vf_build.log" 2>&1 &
VFPID=$!; echo "VF build PID $VFPID"

#    Monitor the "Early View Factor Computation" block:
tail -f "$E2/vf_build.log"
#    EXPECT (compact H-matrix/ACA path):
#      "Using H-matrix compression for view factors..."
#      "  Using ACA compression (tolerance=0.001, max_rank=50)"
#      "H-matrix view factors computed:"  +  "Memory usage: <N> MB"  +  "Compression ratio: <N>x"
#      "Saved view factors to: .../sunlit_radcore_r25.vf"
#    WEDGE RED FLAGS -> kill immediately (`kill $VFPID`):
#      "Using Monte Carlo ray-gather for view factors..."   (dense fallback = the documented wedge path)
#      build RSS climbing into the tens of GB, or the .vf growing toward multiple GB with no compression line.
#    Once "Saved view factors to:" prints, the file is fully flushed. Then EITHER keep step (a) a pure VF-only
#    build (`kill "$VFPID"`), OR — if the YAML already has incidence_angle_albedo:false — let it run on as Run 1 (b).

# a-confirm. Verify CORRECTNESS, not smallness, BEFORE any seasonal run reuses this VF (see landmine 1):
ls -la "$VF"                                         # RECORD the actual size — do NOT gate on "tens of MB"
#    Dimensions + non-empty are surfaced on the FIRST load. To make them print, set radiation.verbose:true in
#    the config (the load detail is verbose-gated); then in the run log EXPECT:
#      "Size: 99511 x 99511"     (== core face count; a "does not match mesh face count" warning => wrong mesh/VF)
#    and it must NOT print "Empty view factor matrix uploaded" (nnz=0 silent-no-radiation — the A3GT trap).
#    # CONFIRM: if the built .vf is SPARSE-format (magic "MFFV"), you can also verify run-free with
#    #   src/python/vf_reciprocity.py --vfm $VF --mesh $MESHDIR/sunlit_radcore_r25.ply  (prints N / nnz / value
#    #   stats). vf_reciprocity.py reads ONLY the sparse MFFV format — it errors "Bad magic" on an H-matrix
#    #   file, in which case rely on the load-log "Size:" / "Empty view factor matrix" checks above.
# b. Run 1 — gray baseline (band_separated:false), incidence_angle_albedo:false, DETACHED + monitored.
# c. Run 2 — two-band (band_separated:true + per-face albedo), same config, DETACHED + monitored.
# d. Compare both to Diviner seasonal via the compare scripts:
#    src/python/diviner_seasonal_compare.py  (site key `sunlit`; RUN_OVERRIDE → divval_sunlit_fix)
```
**Report:** gray-vs-two-band PSR temperature maps + RMS/bias vs Diviner (summer/winter, max/mean).

---

## After E2/E3 — paper re-runs, default-flip, figures
- If validation is clean: decide the **default-flip** (`radiation.band_separated` default) — a *separate*,
  Fermi-reviewed commit. Keep it OFF until then.
- Re-generate any paper numbers/figures affected by two-band. Fig. 7 is already rebuilt
  (`docs/figures/v2/bowl_crater_ingersoll_fig7.{png,pdf}`).
- Remaining production-completion items (TICKET Phase B/C): GPU/multi-GPU two-band mirror, per-band
  stats-file output, per-timestep incidence-albedo update.

## Commit / push (only on your explicit go)
Stage **only** the two-band paths — never `git add -A` (the tree carries unrelated held edits). The
current two-band commit is `f5c455e`; push with:
`$GIT -C $PTM push -u origin feat/two-band-radiosity` (your call).
