# Flux cubes

A **flux cube** is a virtual measurement fixture placed on a site's terrain. It reports the
incident irradiance on six oriented faces, component-resolved, at every output step, plus the
temperature of the mesh face beneath it. It is a *sensor*, not a body:

- it does not occlude, emit, or scatter;
- it is not a row or a column of the view-factor matrix and never enters the radiosity solve;
- cubes do not see each other; cube-to-cube coupling is identically zero;
- everything reported is **incident irradiance in W/m^2** — no albedo, no emissivity, no cube
  material is applied anywhere. Consumers bolt their own optical model on top.

Turning cubes on must not move a single byte of the mesh temperature or flux fields.

---

## 1. Geometry: the cube is sized to the mesh, not to a fixed number of metres

The cube center height and edge are **site-relative by default**:

```
  local_median_edge = median over CORE faces whose centroid lies inside the grid disk
                      of the equivalent edge  e = sqrt(4A/sqrt(3))
                      (the edge of an equilateral triangle of the same area A)

  height_m = height_factor * local_median_edge     (default height_factor = 0.1)
  edge_m   = edge_factor   * local_median_edge     (default edge_factor   = 0.1)
```

`height_m` is a **plumb-line** height along the site vertical `U` — what a leveled lander leg
gives — not a height along the local face normal. The two differ by `h(1 - cos slope)`, which is
sub-millimetre at the measured 4 degree median inner slope, but they must never be mixed and
hardware people assume the plumb line.

The definitions are not assumed: the tool measures `local_median_edge` on the actual mesh, prints
it, and writes it into the file header, so any number in a delivered product can be re-derived.

**Absolute overrides win.** `height_m` / `edge_m` given explicitly (CLI `--height-m` / `--edge-m`)
override the factors, and the file header records `height_factor = 0` to say so.

> **Coarse-mesh sites need the override.** The factor is a fraction of the *local facet size*, so
> it only means "just above the ground" where the mesh is fine. A03MP026 has a dedicated
> high-resolution SfS core (median equivalent edge 1.083 m in the grid region) and resolves to
> **h = 0.1083 m**. A03SP010 has only the generic 80 m/px LOLA base and an 8,746-face core; there
> a 0.1x height is **metres above the ground**, which is a different instrument entirely. Set
> `height_m` explicitly at such sites.

Measured at A03MP026 (2,397,890-face core, 724,593 faces inside the 500 m grid disk):

| quantity | value |
|---|---|
| local median equivalent edge | 1.082969 m |
| resolved height | 0.108297 m |
| resolved edge | 0.108297 m |
| cubes placed (25 m spacing, 500 m radius) | 1,257 |
| frame max tilt across the grid | 0.016438 deg |

---

## 2. `f_max` is an expected-mode disclosure, not a warning

`f_max` is the largest single view-factor entry in a patch's row — the share of that face's
entire incident terrain flux that comes from one mesh facet. `n_faces_50pct` is how many partners
it takes to reach half the terrain weight.

At a **site-relative height of 0.1 x the local facet edge, a large `f_max` on the nadir face is
the intended behaviour, not a defect.** The whole point of putting the cube ~11 cm above a ~1.1 m
facet is that the nadir face becomes a near-unity-view-factor sensor of the facet beneath it.

Measured at A03MP026, `h = 0.1083 m`, 65,536 rays per patch:

| face | min | p05 | p50 | p95 | max |
|---|---|---|---|---|---|
| zenith | 0.0000 | 0.0000 | 0.0001 | 0.0021 | 0.0350 |
| **nadir** | 0.2107 | 0.5013 | **0.8521** | 0.9765 | 0.9949 |
| north | 0.0713 | 0.1841 | 0.3207 | 0.4350 | 0.4895 |
| south | 0.0885 | 0.1980 | 0.3486 | 0.4673 | 0.5600 |
| east | 0.0814 | 0.2019 | 0.3474 | 0.4893 | 0.6465 |
| west | 0.0837 | 0.1785 | 0.3146 | 0.4326 | 0.5046 |

The nadir `n_faces_50pct` is **1** at both the median and the minimum. A threshold warning at
`f_max > 0.25` would fire on **99.6% of nadir patches and 69.4% of all patches** — it would be
noise, and noise is how a real warning gets ignored. So `f_max` and `n_faces_50pct` are
**reported per cube, always**, and the height-factor path emits no warning. An absolute
`height_m` override that lands unexpectedly close to the ground is the case worth flagging, and
that is where the note belongs.

### Near-field error bound, re-derived at h = 0.1083 m

The design's original bound was derived at `h = 1 m`, where the nadir face averaged ~6 facets.
At the site-relative height it must be restated, because the two heights make *different
instruments* and the error moves in opposite directions:

**The numerical error falls.** The ray-traced row integrates the true facet geometry exactly, so
there is no error from the facet *count*; the only error is within-facet radiosity
non-uniformity. At the measured median `f_max = 0.852`, the neighbouring facets carry the
remaining 0.148, and the measured facet-to-facet relative `B_ir` spread inside 1 m is 2.7%. So
the reported nadir irradiance deviates from the underlying facet's own radiosity by

```
  0.148 x 2.7%  ~  0.40%
```

which is better-posed than the h = 1 m case, not worse.

**The model sensitivity rises, and this is the part to disclose.** Because the nadir face is
~85% one facet, it inherits that single facet's temperature error *whole*, instead of averaging
it down across ~6 facets the way a 1 m height did. The measured facet-to-facet temperature spread
inside 1 m is 1.51 K. So the nadir reading is a **sharper but less smoothed** instrument: it
measures one facet nearly exactly, and it does not average away facet-scale scatter. That is the
correct answer for a sensor 11 cm above the ground, and it is what the product should say.

**Do not add a sub-facet near-field model.** That would mean inventing a temperature field the
engine does not have.

---

## 3. Sampling and the ray budget

Rays are drawn **cosine-weighted** about each patch normal using a stratified Shirley-Chiu
concentric mapping with a **per-patch Cranley-Patterson rotation**. Every ray then carries equal
weight and

```
  F[c][p][j] = (rays hitting core face j) / N
  f_sky      = (rays escaping)            / N
  f_ring     = (rays hitting the ring)    / N
```

so `sum_j F + f_sky + f_ring = 1` exactly, and for Lambertian emitters of radiosity `B_j` the
irradiance on the patch is exactly `E = sum_j F[j] B_j`.

The per-patch rotation is not decoration. The engine's own sampler calls `hammersley(k, N, 0, 0)`
with zero offsets, so every face reuses the identical point grain, whose worst-case error sits on
the 45/135 degree diagonal. With 7,542 patches on a regular 25 m lattice, a shared grain would
produce a **coherent stripe across the flux map** rather than white noise. The rotation is free.

### Accuracy metric

**`closure_residual` cannot measure sampling accuracy and must never be quoted as if it did.**
This estimator sums to 1 by construction, so the residual is identically zero (measured: 0.000e+00
on every patch at every budget). It is retained as a **bookkeeping** check — it catches a dropped
or double-counted ray channel — and it is labelled as such in the file.

The metric that does measure accuracy is the **split-half total variation** between two
independently rotated half-rows:

```
  splithalf_tv = 0.5 * sum_j |F_a[j] - F_b[j]|
```

which bounds the flux disagreement directly: `|E_a - E_b| <= splithalf_tv * max_j B_j`.

### Measured budget sweep (A03MP026, production grid, 8 threads, nice +10)

Split-half TV, p50 per face orientation:

| face | 4,096 | 16,384 | 65,536 |
|---|---|---|---|
| zenith | 1.95e-3 | 1.65e-3 | 1.30e-3 |
| nadir | 8.30e-3 | 4.70e-3 | 2.78e-3 |
| north | 3.39e-2 | 2.47e-2 | 1.77e-2 |
| south | 2.93e-2 | 1.94e-2 | 1.27e-2 |
| east | 3.69e-2 | 2.61e-2 | 1.86e-2 |
| west | 2.81e-2 | 1.89e-2 | 1.21e-2 |

Cost — peak RSS was 0.62 GB in **all three** arms:

| n_rays | rays | wall | cache | nnz |
|---|---|---|---|---|
| 4,096 | 30.9 M | 6 s | 6.0 MB | 737,488 |
| 16,384 | 123.6 M | 14 s | 16.5 MB | 2,113,257 |
| 65,536 | 494.5 M | 46 s | 47.2 MB | 6,131,014 |

**Default: `n_rays = 65536`.** The measured throughput is **11.9-20 M rays/s on eight threads**,
about 15x the planning figure, because these rays are short and spatially coherent — all
originating inside one 1 km patch — so the BVH stays cache-resident. That is *not* the regime the
whole-body illumination throughput was measured in, and the illumination derate should not be
reused for local hemisphere sweeps. With the whole sweep costing under a minute, the ~1.4x
accuracy gain on the binding side-face rows is free.

### Truncation

`truncate_rel` defaults to **0 (keep every partner face)**. Truncation buys a flux error bound
that is embarrassing to write down: dropping weight `w_res` costs `|dE| <= w_res * max_j B_j`, and
`B_ir` reaches `sigma*(363.5 K)^4 ~ 990 W/m^2` at this site, so even `w_res = 0.002` is
**2.0 W/m^2** — larger than the entire Earthshine budget the product is asked to resolve. When
set above zero, the tool prints the dropped weight and the resulting bound per face orientation.
Never silent.

---

## 4. The occluder ring carries no radiosity — and it is NOT invisible at this height

Rays that hit the occluder ring are counted in their **own** `f_ring` channel. They are never
folded into `f_sky`: the ring has no thermal solution, so it has no radiosity to contribute, and
calling it "sky" would silently delete a real IR source from the budget.

**The design expected `f_ring ~ 0` at a 500 m grid inside a 3 km core. Measured, it is not.**
At A03MP026 with the 27.3 km ring:

| face | median f_ring | max over all patches |
|---|---|---|
| zenith | 0.0002 | |
| nadir | 0.0000 | |
| north / south / east / west | 0.0008 - 0.0064 | **0.0397** |

The cause is the low cube height itself: a side patch's cosine lobe points at the horizon, so
rays that clear the local terrain run ~2.5 km to the core rim and land on the ring beyond it. Up
to ~4% of a side face's terrain-IR budget is therefore **missing** from the product. At ~100 K
ring terrain that is order 0.1-0.2 W/m^2, which is **larger than the entire primary-body IR term**
(measured 0.010-0.047 W/m^2 at this site) that the product exists to resolve.

This is measured and printed by the geometry tool on every build, not assumed. It needs a scope
decision: either accept and document the deficit, extend the thermal core, or give the ring a
radiosity surrogate.

---

## 5. Tooling

```
  ptm_flux_cube_geom     one-time, CPU-only geometry build  -> <site>.fcg
      no GPU lock, no OpenCL context; safe alongside production GPU windows.
      DO NOT overlap a view-factor assemble step (those peak above 300 GB RSS).
```

The engine **loads** a `.fcg` and refuses on a key mismatch (core geometry hash, core face count,
ring hash, grid parameters, ray count, sampler, seed). It never rebuilds inline: a 7,542-row
hemisphere sweep starting unannounced inside a production window is the kind of surprise that
ends campaigns.

Refusals are hard, never skips:

- a lattice point with no terrain below it aborts the build — silently dropping cubes would change
  the cube count between runs of the same configuration;
- a lattice point landing on the occluder ring aborts — there is no `T_face` and no radiosity there;
- a site within 0.1 degrees of the spin pole aborts — local north is undefined and the face naming
  would be arbitrary;
- `--ring` given but unreadable aborts — the ring changes every sky fraction in the file.

### Validation

`scripts/flux_cube_closed_form_check.py` builds its own analytic meshes and scores the geometry
against closed forms using the estimator's a-priori binomial error bar:

- **infinite flat plane**, `h = 0.1 m`: `f_sky` must be 1 (zenith), 0 (nadir), 1/2 (each side).
  Measured 1.00000 / 0.00000 / 0.50013-0.50015 against a 3-sigma bar of 0.01172.
- **finite square plane**, `L = 10 m`, `h = 5 m`: the cosine-weighted nadir row sum must fall
  between the inscribed and circumscribed disk values, `0.8000 <= sum_j F <= 0.8889`. Measured
  0.83130. A uniform-hemisphere sampler would read 0.5528, so this arm discriminates the cosine
  weighting rather than merely accommodating it.

---

## 6. The engine pass

Flux cubes are evaluated in a **post-solve pass**, once, after the radiation coupling loop has
converged and immediately before the location export. Not inside the loop, for three reasons: the
sweep is the hot loop and only the converged answer is a product; inside the loop the band series
are Picard-blended and `B_ir` carries an emission term that moves with `T` between iterations, so
a blended `B` is not the `B` of any state; and the loop has two dispatch sites (single-timestep and
multi-GPU pool), so evaluating there would mean two insertion points that must agree forever. After
the loop both routes have written the same series, and that confluence is what makes the pass
**route-blind**.

### Reconstruction, not readback

The converged radiosity vectors are function-local in `solveTwoBand` and on the GPU pool never
leave the device. Rather than add a device-to-host transfer to a kernel path that is currently
bit-identical to the CPU, the host inverts the solver's own two closing identities:

```
  B_ir [j] = eps_j*sigma*T_j^4 + (1 - eps_j) * ( Q_ind_ir [j] / eps_j    )
  B_vis[j] = A_src_j * Q_dir_j +      A_j    * ( Q_ind_vis[j] / (1 - A_j) )
```

This is exact, not approximate, and it is spelled **once**, in
`src/cpp/radiation/radiosity_reconstruction.h`, called by both the pass and its test.
Measured against the solver's own retained `B`: **7.7e-17 relative** (bar 1e-12).

`A_src(t)` is not cached. `ptm106_fill` is just `applyIncidenceAlbedo()` of quantities that all
survive to post-solve, so the pass calls that same function rather than storing
`n_faces x n_steps` values or writing the formula a second time.

### Refusals

Every one of these is a hard error with a reason, and each has a negative-control test:

| condition | why it is refused |
|---|---|
| cache built for a different mesh | never apply a foreign cache; the exact regeneration command is printed, and it is never rebuilt inline (PTM-19) |
| `equilibration_steps + n_steps != total_time_steps` | cube rows would stop lining up with `time_steps.txt` (PTM-250/251) |
| gray path (no band-separated series) | one mixed radiosity cannot be split; emitting zeros or a merged column would be worse than failing |
| no ray-occlusion function | assuming every cube is unshadowed fabricates direct solar flux at every shadowed cube in the grid |
| `eps` or `(1-A)` collapsing | the inversion is ill-posed and the solver's own `REFL_GUARD` fallback is invisible from the host |

A pass failure aborts the run. A run that was asked for a product and made none must fail, not
succeed quietly.

### Units

⛔ `sun_positions` and `primary_positions` are in **kilometres**; the mesh is in metres for lunar
and Europa DEMs and kilometres for asteroid shape models. `PassInputs::mesh_to_km` carries the
conversion, set at the call site by the same magnitude test the illumination model uses. Getting
this wrong does not raise an error: the direction to the primary points into the ground, every
disk sample reads as occluded, and the primary-body flux comes out **exactly zero**. That happened
during development and was caught only by comparing against the engine's own `primary_ir`
statistics. The pass now warns whenever the primary is in the ephemeris and no cube sees it at any
epoch.

## 7. Configuration

```yaml
flux_cubes:
  enabled: false            # default OFF; ON must be byte-neutral to every mesh field
  cache_file: ''            # REQUIRED when enabled; built by ptm_flux_cube_geom
  write_binary: true        # flux_cubes/flux_cubes.fcb
  write_csv: true           # flux_cubes/csv/cube_XXXXX.csv
  write_csv_max_cubes: 0    # 0 = every cube; cap it on long production windows
  shadow_probe_rays: 5      # 1 = centre only; 5 = centre + four rim points
```

The block is **top level**, not nested under `output:`, so it cannot be reached by the
body-defaults machinery that makes `thermophysical:` decorative on a built-in body (PTM-67). A
`warnUnreadFluxCubeKeys()` names any key the parser does not consume. The geometry keys
(`spacing_m`, `height_factor`, `n_rays`, ...) are accepted but reported as **inert**: they live in
the cache header, and honoring a YAML copy would create two sources of truth for one number.

## 8. Outputs

| file | producer |
|---|---|
| `flux_cubes/flux_cubes.fcb` | engine, float64 flux `[cube][face][component][step]` |
| `flux_cubes/csv/cube_NNNNN.csv` | engine, one row per output step, 36 flux columns |
| `<site>_flux_cubes.h5` | `scripts/ptm_flux_cubes_to_hdf5.py` (h5py, `europa_viz`) |

HDF5 is written from Python, not C++: the repo declares HDF5 off by default in-tree to avoid
macOS/Conda link issues, and adding an HDF5 link would change the production binary's provenance
for every remaining site run in order to serve a file format. The requirement -- CSV **and** HDF5
-- is met; only the producer moves.

The packager's `--qa` mode can fail, and is meant to: `total == sum(components)`, no NaN or
negative irradiance, closure per patch, and `nadir terrain_ir >= eps*sigma*T_face^4 * f_max`, which
is the check that catches a mis-indexed cube-to-face mapping that every purely structural check
would pass.

### Compatibility with `skip_time_series`

The chunked-run orchestrator forces `output.skip_time_series = True` unconditionally
(`scripts/run_chunked_simulation.py`). Flux cubes read the **in-memory** band series, not the
per-step frames on disk, so the two are compatible -- but that was a claim, and it has now been
measured rather than assumed: on the crater80N config the `.fcb` is **byte-identical**
(md5 `0133a0e073a3162ee21ce2dc38066b59`) with `skip_time_series` true and false, while the
per-step frame count goes 118 -> 0.

⛔ **The pass is NOT chunk-aware.** Under the chunked runner each chunk would run its own
post-solve pass over its own sub-window and emit its own `flux_cubes.fcb`, and `ptm_merge_chunks`
has no rule for merging them. Run a window that needs cubes as a **single direct engine run**, or
treat the per-chunk files as separate products. Making the cube axis mergeable is a v2 item.

## 9. Validation status

| gate | result |
|---|---|
| radiosity reconstruction vs the solver's own `B` | 7.7e-17 / 9.4e-17 relative (bar 1e-12) |
| closed forms, infinite plane | `f_sky` 1.00000 / 0.00000 / 0.50014, 3-sigma bar 0.00586 |
| cosine weighting, analytic bracket | `sum_j F` = 0.83080 in [0.8000, 0.8889]; uniform would give 0.5528 |
| flush limit, h = 1.0 / 0.1 / 0.01 m | `f_max` 0.1307 -> 0.9095 -> 0.9990, monotone |
| pass unit test (3 arms, 5 negative controls) | 0 failures; nadir `terrain_ir` = 904.784902 W/m^2 at 360 K |
| Ingersoll bowl crater | PASSED, 0.492713% |
| mesh-wide passivity, cubes ON and OFF | byte-identical to the pristine baseline engine |

**Passivity is the load-bearing one.** On the 12,800-face crater80N config, 862 physics output
files, the aggregate md5 is `cab1be996c71c52543c006822e6a9deb` for all of: the pristine engine, this
engine with no `flux_cubes` block, this engine with `enabled: false`, and this engine with
**`enabled: true`**. The probe does not perturb the simulation.

⛔ **What that does NOT cover.** Moon, built-in body, CPU `SparseVFOperator` route, two-band
radiosity, single-shot (non-chunked) run. It says nothing about Europa, generic bodies, the gray
path, the GPU pool route, the H-matrix route, or the chunked/warm-start path.

---

## 10. Shadow strata: "window-dark" is not "PSR"

A **permanently shadowed region (PSR)** is dark in **all seasons**. A cube that receives no direct
solar flux during one evaluated window is **window-dark**, which is a different and weaker claim:
if the cube were truly permanently shadowed it would also have been dark in the summer window.

The distinction is not pedantic, because the two answer different questions:

- **window-dark** exercises the **indirect-only flux path** — direct solar identically zero,
  terrain IR down in the tens of mW/m², the primary body's IR no longer negligible. That is the
  physics regime a PSR cube lives in, and it is what validates the dark-flux code path.
- **PSR** is a statement about the *site*, decided by the site's `psr_mask` across all seasons, not
  by any single window.

`scripts/flux_cube_strata.py` reports the `window_dark` stratum and refuses to call it PSR.

### Whether a genuine PSR is even reachable at A03MP026

Measured directly from the campaign's own `cogs/A03MP026/psr_mask.tif` (1 m polar-stereographic;
the mask is 0/1 at 99.87% of valid pixels, the remaining 0.13% being the COG's linear-interpolation
boundary, so both a strict and an inclusive count are given):

| quantity | value |
|---|---|
| PSR pixels within 500 m of the site | **441** inclusive (`>= 0.5`) / 346 strict (`== 1`) |
| nearest PSR pixel to the site | **118.7 m** |
| distinct PSR patches inside the grid | 7 |
| largest patch | **195 m²**, only **15 × 15 m** across, centre at site ENU E = −109.9 m, N = +62.8 m (lat −84.220932, lon 33.466089) |
| PSR areal fraction of the grid | 441 m² / 0.785 km² ≈ **0.056%** |
| 25 m production lattice points landing on PSR | **0 of 1257** (≈0.7 expected by area) |

⚠️ The site-local ENU frame is rotated from the projection's axes by the site longitude (33.5°) —
in a south-polar stereographic projection with `lon_0 = 0`, local north is the *radial outward*
direction, not projected +Y. Skipping that rotation moves the patch centroid by tens of metres and
changes which lattice points appear to be on PSR.

**Consequence: the production 25 m lattice cannot sample a genuine PSR at this site.** The patches
are far smaller than the lattice spacing. Reaching one requires a *targeted dense array*, not a
finer global grid — see §11.

## 11. Targeted PSR array (designed, not yet run)

To obtain a true-PSR stratum, place a second, small, dense array on the largest in-grid PSR patch.

Recommended: **spacing 1.0 m, radius 10 m, centred on lat −84.220932, lon 33.466089**
→ 317 cubes, of which **196 (62%) fall on PSR pixels**. Sizing alternatives, all measured against
the mask:

| spacing | radius | cubes | on PSR |
|---|---|---|---|
| 5.0 m | 20 m | 49 | 9 (18%) |
| 2.0 m | 12 m | 113 | 51 (45%) |
| 2.0 m | 20 m | 317 | 51 (16%) |
| **1.0 m** | **10 m** | **317** | **196 (62%)** |

⚠️ At 1 m spacing the site-relative height rule gives h = 0.1083 m from a 1.08 m median facet, so
the cubes are ~10 spacings apart vertically — fine, but the array is dense relative to the mesh and
`f_max` should be inspected for the vertex degeneracy of §2/FC-04.

**Tooling status.** `ptm_flux_cube_geom` already accepts an arbitrary `--center-lat/--center-lon`
with its own `--spacing/--radius`, so building the targeted cache needs **no code change**. What
the engine does *not* yet support is **two arrays in one run**: `flux_cubes.cache_file` is a single
path, and the `.fcg` has no array identifier, so cubes from two lattices would share one `(i, j)`
index space ambiguously. Two options:

1. **Second run** (no code change): one extra ~40 min winter-window run against a second cache.
2. **Combined cache** (format v2): add an `array_id` to `CubeRecord` and let the geometry tool
   append a second lattice. One run, one product, at the cost of a format revision and a reader
   update in the packager.

Option 1 is recommended first — it needs nothing new and keeps the two products independently
auditable. Option 2 is worth doing only if targeted arrays become routine across sites.
