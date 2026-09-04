# View-factor unification — design

Status: **stages 1 and 2 implemented** (`feat/vf-unification-stage1`, `feat/vf-unification-stage2`).
Paul ruled on D1–D6 on 2026-08-23; §5 records each ruling and what stage 2 measured or built
for it. Written 2026-08-23.

Paul's standing top science priority is *one physics across all paths*. For the thermal solver
that priority produced the divergence map (`SESSION-NOTES-2026-08-11`); this document is the
same exercise for the **view factor**, which is the other half of the radiation budget and,
unlike the solver, is written to disk and reused for months.

The scope here is *how an entry of `F` is produced*, not how the radiosity system is solved.

---

## 1. Why this matters more than a solver difference

A view-factor matrix is a **cached artifact**. A `.vf` / `.hmat` / `.csr` built in July is
loaded unchanged in August, and PTM-19 forbids rebuilding a production one casually. So a
divergence between builders is not a transient: it is baked into every product downstream of
whichever builder happened to run, and the choice of builder has never been purely a
configuration decision —

* `radiation_exchange.cpp:2594` promotes a run to the H-matrix builder above
  **100,000 faces**, silently, whatever the config said;
* `radiation_exchange.cpp:2785` diverts to the Monte-Carlo builder when
  `use_monte_carlo_vf` is set, and `:2781` notes the >1M-face regime;
* several production campaigns bypass the engine entirely and inject an offline-assembled CSR
  through `PTM_HMATRIX_VF_REPLACE` (`radiation_exchange.cpp:280`).

Three of those four are reached by size or by an environment variable, not by a physics
choice.

---

## 2. Inventory — every view-factor construction path

`file:line` references are at `feat/vf-unification-stage1` tip. **P** marks a path a production
run can take; **D** marks a diagnostic.

### 2.1 Sparse in-engine builder — `view_factor.cpp` — **P**

Entry `ViewFactorCalculator::computeAll()` (`view_factor.cpp:810`). Default builder below
100,000 faces. Writes the sparse `.vf`.

| aspect | value |
|---|---|
| far kernel | M1 centroid, `computeCentroid()` `:1180`, $F_{ij} = A_j \cos_i \cos_j / (\pi r^2)$ |
| near kernel | shared `radiation::nearFieldPairIntegral()` (`nearfield_quadrature.h:200`) at fixed order `nearfield_quadrature_order` (3), via `computeSubdivision()` `:1259` |
| near/far window | **shared** `radiation::isNearFieldPair()` (`nearfield_quadrature.h:66`), $r < w(\sqrt{A_i}+\sqrt{A_j})$, $w$ = `nearfield_window_factor` = 4 — *stage 1; was $N_s \le 5$* |
| grazing cut | **shared** `radiation::kGrazingCosCut` = 1e-6 (`radiation_config.h:47`) — *stage 1; already 1e-6 here* |
| occlusion | one Embree `rtcIntersect1` per pair (`isVisible()` `:1468`) or an 8/16-wide packet in the batched far path (`prepareRay()` `:306`); offset $\max(0.001r,\thinspace 0.5\ \mathrm{m})$ capped at $0.1r$, same at the far end, and **any segment shorter than 0.1 m after both margins is assumed VISIBLE** |
| occlusion ordering | **canonical**, lower face index first, at every site — *stage 1; was canonical only in the near field* |
| symmetrization | none. Reciprocity is an identity of both kernels, not a post-hoc symmetrization |
| storage floor | `visibility_threshold` = 1e-8, applied to the **direction-dependent** $F_{ij}$ (`:993`, `:1449`, `:1460`) |
| pair set | `selectNeighborsAdaptive()` `:1742` to `captured_energy_target` = 0.95, capped at `max_neighbors` = 500, within `search_radius_factor` = 10 |
| gated by | `test_vf_production` arms [I] [R] [S] [D] [T] [W] |

### 2.2 H-matrix builder — `hmatrix_view_factor.cpp` — **P**

Selected by `use_hmatrix`, or automatically above 100,000 faces. Writes `.hmat`.

**Four** matrix-entry generators, not one:

| generator | file:line | used for |
|---|---|---|
| `computeViewFactor()` | `:3918` | single entries |
| `computeViewFactorBlock()` | `:3117` | dense / small blocks, batched rays |
| `computeViewFactorRow()` | `:3495` | ACA cross sampling |
| `computeViewFactorColumn()` | `:3599` | ACA cross sampling |

| aspect | value |
|---|---|
| far kernel | M1 centroid, identical algebra to §2.1 |
| near kernel | shared `nearFieldPairIntegral()` via `computeViewFactorRefined()` `:4080`, at an order `refinedQuadratureOrder()` `:4043` = $\mathrm{clamp}(\max(m_{\text{cfg}}, \lceil w d/r \rceil), 1, 16)$ — the config key is a **floor** here, not the order |
| near/far window | **shared**, `isNearFieldPair()` `:4032` → `radiation::isNearFieldPair()`. Applied in **all four** generators — *stage 1; the two ACA cross generators previously did not refine at all* |
| grazing cut | **shared** 1e-6 — *stage 1; was `MIN_COS_ANGLE = 0.01`, four copies* |
| pre-cull | drops a candidate whose **M1** estimate is below `visibility_threshold * 0.1` **before** refinement. The sparse builder has no such pre-cull (§5, decision 4) |
| occlusion | `setCanonicalVisibilityRay()` `:4000`, same offsets as §2.1, optional normal lift `PTM_HMATRIX_RAY_LIFT` (default 0), optional heightfield horizon `use_horizon_vis_` |
| occlusion ordering | **canonical** — *stage 1* |
| compression | ACA on admissible blocks (`tryCompressToLowRankACA()` `:3703`, tol 1e-3, max rank 50); randomized SVD on small blocks (`tryCompressToLowRank()` `:3411`, tol 1e-2, `max_svd_rank` 50) |
| admissibility | $\max(\mathrm{diam}) \le 0.8 \thinspace \mathrm{dist}(\text{centers})$ on the bounding-box **diagonal**, `isAdmissible()` `:3898` |
| gated by | `test_vf_production` arms [H] and [U] |

### 2.3 Monte-Carlo ray-gather builder — `mc_view_factor.cpp` — **P**, **UNGATED**

`computeMonteCarlo()` (`mc_view_factor.cpp:66`). Selected by `use_monte_carlo_vf`; exists
because H-matrix build memory is infeasible above ~4M faces.

| aspect | value |
|---|---|
| estimator | cosine-weighted hemisphere sampling from the face **CENTROID**, `mc_rays_per_face` = 100,000. $F_{ij}$ = hit fraction |
| near/far | **none**. There is no near-field treatment at all; a touching pair is estimated by the same centroid-launched sampler as a distant one, so it carries M1's one-sided near-field bias |
| grazing cut | none explicitly — the cosine weighting handles it |
| occlusion | exact by construction, `rtcIntersect16`, origin lifted $10^{-4}\sqrt{\bar A}$ along the normal, `tnear` = 1e-4 |
| symmetrization | **none**. The matrix is not reciprocal, at any budget |
| truncation | top-`max_neighbors` per face; the discarded tail is added to the **sky** channel so $\sum_j F_{ij} + F_{sky} = 1$ exactly |
| gated by | **nothing** |

### 2.4 Offline referee + assembler — `scripts/ptm12_diag/` — **P**

`mc_ref_pp.cpp` sweeps rays; `assemble_vf_haworth.py` (and `assemble_vf.py`,
`assemble_vf2.py`) assembles a `.csr`, injected at run time by `PTM_HMATRIX_VF_REPLACE`
(`radiation_exchange.cpp:280`). This built the A3GT and moon3d production matrices.

| aspect | value |
|---|---|
| estimator | hemisphere ray sweep from the face **CENTROID**, `rise-march` self-relief filter, `keps` = 0.2 |
| near/far | **none**; same centroid-launch bias as §2.3 |
| occlusion | exact, 0.1 mm ray offset |
| symmetrization | explicit, `assemble_vf_haworth.py:318`, divisor **2** since the FC-08 fix — a missing ray direction is a measured zero, never a missing datum |
| keep rule | `MIN_HITS` |
| gated by | `test_vf_assembler_gate.py` (CI job `vf-assembler-gate`) and `test_vf_cross_agreement.py` |

⛔ **The referee cannot referee the near field.** It launches from the centroid, so it shares
M1's one-sided bias and on near pairs *prefers the wrong estimator* — mean relative $|\Delta|$
0.0349 against engine order 1 versus 0.0474 against order 3, while order 1 reads $+51.7$
percent on the touching closed form. This is stated in the cross gate's own banner.

### 2.5 Flux-cube geometry — `src/cpp/fluxcubes/flux_cube_geom.cpp` — **P**

A fifth construction, easy to miss because its rows are not indexed by mesh face on the
receiving side. `ptm_flux_cube_geom` places virtual cubes on the terrain and ray-traces a
cosine-weighted view-factor row from each of the six oriented patches over the core mesh, plus
separate sky and occluder-ring channels, $\sum_j F_j + f_{sky} + f_{ring} = 1$ exactly.
16,384 rays per patch, `tnear` = 0. Same estimator family as §2.3/§2.4 (centroid-launched
hemisphere sampling), and it inherits the same absence of any near-field treatment. The engine
**loads** the cache and refuses on a key mismatch; it never rebuilds it.

### 2.6 Harness and diagnostic replicas — **D**

| path | file:line | note |
|---|---|---|
| `bowl_crater_ptm.cpp` KPROBE kernel | `:289` (`tri_subcentroids`) | a THIRD, deliberate replica of the near-field kernel, plus the retired one-sided rule, so `PTM_BOWL_KPROBE` stays an independent reading. Must NOT be pointed at the shared header |
| `test_bowl_crater.cpp` | `:669` | the Ingersoll gate builds `F` from its own inline analytic centroid formula with **no ray tracing**. "Ingersoll passes" is not evidence about any builder |
| `vf_hemisphere_mc.cpp`, `vf_mc_flux.cpp` | — | independent hemisphere-MC ground truths, used to arbitrate the H-matrix floor coupling |
| `vf_dump_sparse.cpp` | `:240` | drives §2.1 directly; the cross-agreement gate's engine side |
| `make_empty_vf.cpp` | — | writes a valid `nnz = 0` HMAT/v2 via the production serializer. Not an estimator |
| `benchmark_gpu_visibility.cpp` | — | GPU BVH vs Embree **rays/second only**. The production VF build has no GPU path; every builder above uses Embree on the CPU |

### 2.7 What consumes `F` and can still change it

Not construction, but on the same audit line: `vf_row_norm_` (`radiation_exchange.cpp:532`)
rescales any row with $\sum_j F_{ij} > 1$, and `smith_beaming.h` renormalizes view factors for
anisotropic emission at `beam_vf_threshold`. A near-field entry that is silently too large is
absorbed by the row norm, which is why the shared kernel's physical-bound clamp is **counted
and printed** rather than silent.

---

## 3. What stage 1 changed, and what it measured

Three commits, each with its own gate and its own attribution.

### 3.1 The ACA cross generators now refine the near field

`computeViewFactorRow`/`Column` were pure M1 with no near-field branch, so whether a close pair
got the M2 quadrature depended on which **block** the octree put it in. Now all four generators
call one `isNearFieldPair()`.

**Measured exposure: zero, in every production-reachable configuration.** ACA runs only on
admissible blocks, and the admissibility condition — on the bounding-box *diagonal*, at
$\eta = 0.8$ — keeps the closest pair those generators ever see outside the near-field window:

| mesh | faces/octree node | ACA cross pairs | near pairs | closest approach $r/4d$ |
|---|---|---|---|---|
| Ingersoll cap, 1,404 faces | 29 | 0 | 0 | ACA never runs |
| Ingersoll cap, 4,544 faces | 94 | 0 | 0 | ACA never runs |
| Ingersoll cap, 6,960 faces | 145 | 3,939,688 | 0 | **1.609** |
| `lunar_crater_80N`, 12,800 faces | 261 | 92,191 | 0 | **2.040** |

ACA does not run below roughly 7,000 faces at all: the octree targets $\sim n/49$ faces per
node for a medium mesh and a block only reaches ACA above `aca_min_block_size` = 100.

So the fix is **inert on these meshes** and its value is that the invariant is now structural
rather than an accident of two constants. **1.6× is a margin nobody declared**, so the builder
now prints the closest approach on every build (not behind `verbose`) and exposes it to gates
through `crossGeneratorNearPairs()` / `crossGeneratorMinROver4d()`.

The defect **is** reachable. At `min_block_size` = `aca_min_block_size` = 16 on the 1,404-face
cap, 4,264 near pairs go through the cross generators and the assembled matrix departs from the
exact one by **4.646e-02** on the worst entry, with that entry equal to its own M1 centroid
value to every digit. After the fix: **7.4e-15**. That configuration is what gate arm [U] runs.

### 3.2 One near/far window

$N_s = r/(A_iA_j)^{1/4} \le 5$ (sparse) and $r < 4(\sqrt{A_i}+\sqrt{A_j})$ (H-matrix) became
one rule with a config key. The sum is the physically motivated length scale because the
centroid error is set by the **larger** of the two solid angles; the geometric mean cannot see
an area contrast:

| $A_j/A_i$ | M1 error at the old cut ($N_s = 5$) | at the new cut ($w = 4$) |
|---|---|---|
| 1 | $+3.49$ percent | $+1.38$ percent |
| 100 | $+16.7$ percent | $+2.3$ percent |
| 10000 | $+150$ percent | about $+2$ percent |

A 100:1 *linear* contrast is a core-plus-ring mesh. Cost and effect on `lunar_crater_80N`:

| $w$ | nnz | mean $\sum_j F_{ij}$ | total mass | vs $w = 8$ | VF build |
|---|---|---|---|---|---|
| 0 (pure M1) | 904,256 | 0.0047792 | 61.174 | $+1.877$ percent | 0.3 s |
| 2.5 (old rule, equal areas) | 904,506 | 0.0046940 | 60.083 | $+0.0138$ percent | 0.3 s |
| **4.0 (default)** | 904,385 | 0.0046931 | 60.071 | $+0.0068$ percent | 0.3 s |
| 8.0 | 904,357 | 0.0046926 | 60.065 | — | 0.3 s |

End to end through `baseline_crater80N_vfbuild_sparse`: 2,562 of 12,800 faces move on
`temperature_max`, median $-1.5 \times 10^{-4}$ K, worst $0.019$ K — cooling, which is the sign
the one-signed M1 bias requires. The H-matrix arm is unchanged.

### 3.3 One grazing cut, one occlusion-ray convention

`MIN_COS_ANGLE` 0.01 (H-matrix, four copies) against 1e-6 (sparse) meant the two builders
disagreed about which pairs **exist**. Unified at 1e-6. Sparse byte-identical; H-matrix on
crater80N moves 2,623 of 12,800 faces, median $+1.5 \times 10^{-3}$ K, worst $+0.025$ K —
warming, because it now keeps grazing terrain-IR partners it used to delete. Compression ratio
36.2 → 36.1 and build 0.86 → 0.82 s: the cull was not paying for itself. On the gate's cap it
removes 1,584 sparse-only entries and improves the two builders' area-weighted mass agreement
2.5×, 0.1557 → 0.0627 percent.

The occlusion query is now issued in canonical (lower face index first) order everywhere, so a
float32 traversal difference between the two ends of one segment can no longer decide that
$F_{ij}$ exists while $F_{ji}$ does not. Sparse byte-identical; H-matrix moves exactly **one**
face of 12,800, by $3.05 \times 10^{-5}$ K.

---

## 4. Target architecture

Two artifacts, consumed by every path that can consume them.

**A. One kernel library.** `nearfield_quadrature.h` is the seed and already holds the
estimator, the near/far predicate and the physical-bound tripwire. It should grow to hold, and
be the only definition of:

* `nearFieldPairIntegral()` — the symmetric double-area quadrature (already shared);
* `isNearFieldPair()` — the near/far predicate (already shared);
* `kGrazingCosCut` (already shared, in `radiation_config.h`);
* `centroidPairIntegral()` — the M1 kernel, currently written out **seven** times with
  identical algebra: `view_factor.cpp:1252`, `:1367`, `:1448` and
  `hmatrix_view_factor.cpp:3212`, `:3542`, `:3646`, `:3962`;
* `visibilityRay()` — the offset/margin arithmetic and the canonical ordering, currently in
  three places (`view_factor.cpp:306`, `:1468`, `hmatrix_view_factor.cpp:4000`), down from
  four before stage 1.

**B. One policy struct.** A `ViewFactorPolicy` carrying every number that decides *what* is
computed rather than *how*: `nearfield_window_factor`, `nearfield_quadrature_order` (and each
builder's order rule as an explicit enum rather than an implicit convention),
`kGrazingCosCut`, `visibility_threshold` and whether it applies to the directed or the
undirected entry, the ray offset constants, and the pre-cull factor. Every builder takes one by
value and **prints it** at build time, so a `.vf`'s provenance is recoverable from its log.

Everything else — how pairs are enumerated, how blocks are decomposed, whether rays are batched
— stays per-builder, because those are genuine engineering differences that do not change the
physics of a pair.

---

## 4b. What stage 2 changed, and what it measured

Five commits. Every number below is on `lunar_crater_80N` (12,800 faces, 17.7 m facets) unless
stated otherwise, at `OMP_NUM_THREADS=48` for the validation arms and 8 for the builder probes.

### 4b.1 The H-matrix matvec is deterministic

`HMatrixViewFactor::matvec()` ran `parallel for schedule(dynamic)` over every root block and
accumulated into `y` through `#pragma omp atomic`, so the order threads reached a given `y[i]`
was not fixed and 338 of 12,800 faces moved by one ULP between runs. Root blocks are now grouped
so no two groups write the same row, disjointness **verified** rather than assumed, each group
summed by one thread in fixed block order. Four consecutive runs byte-identical at 8 and at 48
threads; negative control `PTM_HMATRIX_MATVEC_ATOMIC=1` red 3 of 3.

The build was never the problem: a new `contentHash()`, printed on every build with the build
thread count, read `0xbe50eb778db92c5a` across four runs and across 4, 8 and 48 threads while the
temperatures moved. That measurement is what separated the two failure modes.

### 4b.2 ⛔ The sparse builder was doing NO occlusion at all

`initEmbree` recenters the scene vertices on the mesh bounding box before the float32 conversion
(`adc2aa5`, 2026-07-20). `ViewFactorCalculator`'s `face_centers_` stay in the mesh frame, and
nobody told it. Every occlusion ray started 1.74e6 m from the geometry:

| | pairs rejected as blocked |
|---|---|
| engine, before | **0** of 1,307,376 |
| `vf_dump_sparse` (scene and rays both in the mesh frame) | 1,226,778 of 7,536,236 |
| engine, after | 224,412 of 1,521,968 |

Effect on `baseline_crater80N_vfbuild_sparse`: 2,626 of 12,800 faces, `temperature_max` median
−0.0064 K, **worst −6.52 K**, every face cooling. Affects any run that BUILDS a sparse `.vf`
in-engine on a mesh whose bounding-box center is far from the body center — every DEM patch under
the 100,000-face H-matrix promotion threshold. `test_vf_production` builds its own scene at mesh
coordinates and is blind to it by construction.

### 4b.3 One scale-relative occlusion ray (D5)

Same crater at three facet sizes, geometry and absolute coordinates identical:

| facet | rule | rejected as blocked | never traced | nnz |
|---|---|---|---|---|
| 17.7 m | retired | 224,412 | 0 | 1,297,556 |
| 17.7 m | new | 224,412 | 0 | 1,297,556 |
| 0.177 m | retired | 212,364 | 108 | 1,302,540 |
| 0.177 m | new | 232,718 | 0 | 1,297,556 |
| 1.77 mm | retired | **0** | 1,468,252 | 1,307,376 |
| 1.77 mm | new | 102,635 | 0 | 1,297,556 |

The new rule returns the same nnz at all three scales; the absolute rule cannot. On the 1.77 mm
copy the temperature effect is 2,626 faces, worst **−7.84 K**. On the 17.7 m gate mesh the two
rules make the same decision on every pair, because $0.02\sqrt{2A} = 0.500$ m there — the retired
constant, exactly.

### 4b.4 The Monte-Carlo builder (D2)

Reciprocity by combining the two directions with divisor 2; the near field from the shared
quadrature over a geometry-enumerated pair set; the sky channel derived rather than accumulated.
Gate arm **[M]**: closure +1.86 percent against the closed form (the sparse builder reads +2.04
on the same mesh), reciprocity 3.9e-16, worst near-field entry against the sparse builder
6.0e-16, 39,180 near pairs present. Negative control `PTM_MC_LEGACY=1`: **M2 and M3 red at 0.90
and 0.89 while M1 still passes** — the closure test cannot see either defect.

Two further scene-coupling defects surfaced and were fixed: the builder recomputed the
bounding-box shift itself, and it ignored `mesh->face_normals` in favor of an "away from the
origin" rule. Given a mesh-frame scene the first produced 33 partners per face instead of 425;
the second produced an entirely empty matrix. Neither errored.

### 4b.5 The storage floor and the pre-cull (D4)

Both now direction-independent. The floor moves 11 entries and 0.000005 percent of the mass on
this mesh; 20 near pairs bypass the M1 pre-cull and the arm is byte-identical either way.

---

## 5. Decision points for Paul

Each of these is a real choice with a cost, not an oversight. Stage 1 deliberately stopped
short of all of them.

**D1. Does the offline assembler adopt the engine kernel, or stay an independent referee?**

*For adopting:* it is a production path (§2.4) — A3GT and moon3d products came from it — and it
has no near-field treatment at all, so every touching pair in those matrices carries M1's
one-sided bias, the same bias the engine's M2 fix removed on 2026-08-23. "One physics across
all paths" says a production estimator should not be exempt because it happens to be written in
another language.

*For staying independent:* it is the only estimator in the system that does not share code with
the engine, and that independence has already paid for itself — the cross-agreement gate exists
because the two disagree, and FC-08 was found by arbitrating one against a closed form. Making
the referee call the engine's kernel would make agreement a tautology and delete the check.

*Honest middle:* keep it independent **and document its limits in the gate itself**, which is
already half-done — the near-field blindness is stated in the cross gate's NOT COVERED banner.
The remaining gap is that nobody has measured what the missing near-field treatment costs the
shipped A3GT/moon3d matrices. That measurement is cheap on the surviving raw sweeps and does
not require changing the assembler. **Recommend: measure first, decide after.**

**RULED (Paul, 2026-08-23): measure first. MEASURED** — `~/projects/ada_vfstage2_2026-08-23/d1_table.txt`,
from the surviving raw sweeps, no re-sweep and no rebuild:

| product | faces | facet | near-field mass share | RAW sweep vs engine M2 | estimator bias alone | worst face |
|---|---|---|---|---|---|---|
| moon3d fig8-hybrid core | 140,091 | 172 m | 20.34% | −22.50% | **+0.62%** | +0.033 K |
| A3GT lunar polar site A core | 2,397,890 | 2.00 m | 6.33% | −23.95% | **+0.73%** | +0.020 K |
| A3GT lunar polar site B core | 1,798,135 | 2.04 m | 8.67% | −35.78% | **+1.42%** | +0.057 K |
| A3GT lunar polar site C core | 8,746 | 56.8 m | 6.63% | −22.92% | **+1.14%** | +0.014 K |

The raw column and the bias column are different questions. The raw difference conflates the
estimator's near-field bias (positive) with the sweep's occlusion and its `keps` = 0.2 self-relief
filter (both negative), and is 30 times the size of either. Separating them by the sweep's ratio to
its **own** analytic estimator — a point emitter at the centroid against a subdivided receiver,
which is what a centroid-launched gather estimates with no occlusion — leaves the estimator bias
alone, and it is under 1.5 percent mass-weighted on every product. By $\sigma T^4$ linearization at
250 K that is **at most 0.06 K on the worst sampled face and under 0.01 K at the 95th percentile**.

⛔ **The 509k moon3d product's raw sweep does NOT survive.** What survives under
`~/projects/moon3d/vf/fig8_hybrid_sweep/` is the fig8-hybrid 140k core at 172 m facets. The 509k
patch is 2.48 mm-faceted, a regime this table does not reach, and §4b.3 shows the occlusion rule
alone is worth −7.84 K there.

**D2. Should the Monte-Carlo builder get a near-field treatment, or be documented as
far-field-only?**

It is ungated, it is a production path above ~4M faces, it is not reciprocal at any budget, and
it has no near-field treatment. Options: (a) gate it and leave it as-is, documenting that it is
a far-field estimator; (b) give it a near-field correction (replace the top-`k` nearest
partners' MC estimates with the shared quadrature); (c) retire it in favor of the H-matrix once
the out-of-core path is trusted. (b) is the only one that makes it "one physics", and it is the
most work. **No recommendation without a measurement of how much near-field mass a 4M-face mesh
actually carries.**

**RULED (Paul, 2026-08-23): (b), give it the near field. DONE**, see §4b.4 and gate arm [M].

**D3. Is `w = 4` the right window, and should the H-matrix keep its adaptive order?**

Stage 1 chose $w = 4$ because it left the H-matrix untouched and widening is the safe direction.
$w = 8$ would halve the residual again (table in §3.2) at a cost that was unmeasurable on
crater80N but is not obviously unmeasurable on a 500k-face mesh. Separately, the H-matrix's
adaptive order means the two builders still differ on a close pair by a *declared* amount —
0.455 on the worst entry at the default order. Pinning both to a fixed order was **measured and
rejected** (4× mass-weighted regression on the H-matrix); pinning both to the *adaptive* rule
has not been measured on the sparse builder, and would cost it up to 191 evaluations per near
pair instead of 81.

**D4. `visibility_threshold` and the H-matrix pre-cull.**

Two asymmetries stage 1 did not touch. (a) `visibility_threshold` is applied to the
**direction-dependent** $F_{ij}$, so an area-asymmetric pair can be stored one-sided —
reciprocity then reads a defect that is really a storage artifact. (b) The H-matrix drops a
candidate whose **M1** estimate is below `visibility_threshold * 0.1` *before* refinement, so a
near pair whose refined value would clear the floor can be deleted on the strength of an
estimate the builder itself considers wrong there. Both are one-line changes with unmeasured
cost.

**RULED (Paul, 2026-08-23): implement both. DONE**, see §4b.5. Both can only ADD entries; both
are measured inert on every mesh available here, and the pre-cull's reachable population (20
pairs on the gate mesh) is now a printed number rather than an argument.

**D5. The 0.5 m absolute ray offset.**

`max(0.001 r, 0.5 m)` capped at `0.1 r`, plus "a segment shorter than 0.1 m after both margins
is assumed VISIBLE", is shared by both in-engine builders — consistently, and consistently
wrong on a sub-meter mesh. On the 2.48 mm-facet moon3d patch every near pair skips the
occlusion test entirely. This is [[ptm-shadow-ray-offset-scale-bug]] and it is a *physics* bug,
not a unification one; it is listed here because any fix must land in the shared
`visibilityRay()` of §4 rather than in one builder.

**RULED (Paul, 2026-08-23): yes, make it scale-aware. DONE**, see §4b.3. `visibilityRay()` now
exists, in `nearfield_quadrature.h`, and is the only copy.

**D6. Which production artifacts change, and what re-runs that implies.**

| artifact | affected by | re-run implied |
|---|---|---|
| Europa sparse `.vf` (icosphere, uniform facets) | §3.2 window only. Uniform areas ⇒ the change is the $N_s$ 5 to 8 widening; expect a sub-0.02 K shift of the same character as crater80N | **rebuild only if a fresh `.vf` is wanted.** Cached files are unaffected; the 53 shipped configs that name a non-empty `view_factor_file:` are inert to all of stage 1 |
| A3GT ring/core `.csr` (offline assembler) | **nothing in stage 1** — that path is §2.4 and stage 1 touched only the in-engine builders | none |
| A3GT / moon3d `.hmat` bases | §3.3 grazing cut, if rebuilt | PTM-19: **do not rebuild**. A rebuild is a campaign decision, not a side effect |
| moon3d 509k `.vf` | PTM-19 forbids rebuilding it | none |
| flux-cube `.fcg` caches | nothing | none |

Nothing in stage 1 forces a re-run. Everything in stage 1 changes what a **new** build produces.

**RULED (Paul, 2026-08-23): no re-runs of anything now.** Stage 2 does not change that for cached
artifacts, but it does change the reading of §4b.2 for any sparse `.vf` BUILT in-engine on a DEM
patch since 2026-07-20: such a matrix has no occlusion in it. Whether any shipped product came
from that path is a provenance question, not a code question, and this document does not answer
it.

---

## 6. Staged migration plan

| stage | content | gate | estimated cost |
|---|---|---|---|
| **1 — done** | ACA cross-generator refinement; one near/far window (`nearfield_window_factor`); one grazing cut; canonical occlusion ordering; arm [U]; exposure instrumentation | `test_vf_production` 29 verdicts incl. [U0]–[U2]; pre-fix control red on [U1] alone; both fresh-VF validation arms re-banked with attribution | delivered |
| **2** | Extract `centroidPairIntegral()` and `visibilityRay()` into the shared header; introduce `ViewFactorPolicy` and print it at build time | byte-identity on both fresh-VF arms — but see the `-ffast-math` caveat below | ~1 day |
| **3** | Gate the Monte-Carlo builder: run it through `test_vf_production`'s closed forms at a high ray budget, with an explicit statement of what it does not do (no near field, not reciprocal) | new arm [M], plus a ray-budget-invariance check (the FC-08 detector) | ~1 day |
| **4** | Decisions D1/D2/D4 implemented, whichever way Paul rules | per decision | 1–4 days |

⛔ **Byte-identity is not available as a stage-2 acceptance criterion in this repo.**
`-ffast-math` (`CMakeLists.txt:259`, which overrides the earlier `-ffp-contract=off`) makes an
arithmetically exact extraction move the last digit: a pure code motion moved
`test_vf_production` R1c from 6.601e-16 to 6.757e-16, and building both trees with
`USE_AGGRESSIVE_OPTIMIZATIONS=OFF` made every verdict identical. Stage 2 must be accepted on
*verdict* identity plus a stated bound, not on md5s.

---

## 7. Things a future reader should not have to rediscover

* **Reciprocity cannot detect an estimator that is wrong the same way in both directions.** M1
  satisfies $A_iF_{ij} = A_jF_{ji}$ as an algebraic identity, so an all-M1 block is perfectly
  reciprocal and perfectly wrong; [H2] read 4e-14 throughout the ACA gap. Closure cannot see a
  scale error either (FC-08 was a factor of two with $\sum_j F_{ij} \le 0.18$).
* **`min_block_size >= n_faces` is not a dense reference.** That route still passes the single
  block through `tryCompressToLowRank()`, which caps the rank at `max_svd_rank` = 50 and returns
  success *even when the requested tolerance was not met* — on crater80N it produced NEGATIVE
  view factors and a fabricated 17 percent mass difference. Use
  `HMatrixViewFactor::computeDenseReference()` (`:4222`).
* ~~**The `moon_crater80N_vfbuild_hmatrix` arm is not reproducible run to run.**~~ **FIXED in
  stage 2.** The cause was not the build: the assembled matrix's content digest was identical
  across four runs while the temperatures moved. It was `HMatrixViewFactor::matvec()`, which ran
  `parallel for schedule(dynamic)` over every root block and accumulated through
  `#pragma omp atomic`, so the order threads reached a given `y[i]` was not fixed. Root blocks are
  now grouped by written row — verified disjoint, not assumed — and each group is summed by one
  thread in fixed block order. Four consecutive runs are byte-identical at 8 and at 48 threads.
* **`ptm_thermal_sim`'s md5 depends on the worktree path** — it embeds the absolute
  `data/kernels` directory — so the same commit built in two worktrees gives two md5s.
  `test_vf_production` and `test_bowl_crater` do not and reproduce exactly.
* **Statistics are thread-count dependent.** The same binary at `OMP_NUM_THREADS=8` and at 48
  gives different `temperature_sum.bin`, and nothing else. Since stage 2 the engine writes
  `omp_max_threads` into `provenance.txt`, `ptm_validate_baseline.py bank` records it, and
  `verify` reports a thread-count change before the byte verdict. Manifests banked before that
  carry no thread count and `verify` says so rather than comparing silently.
