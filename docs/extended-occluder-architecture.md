# Core-thermal + occluder-ring architecture

## Context

A3GT v2 needs distant topography (massifs, crater rims out to the physics
horizon ~20 km) to shadow the science ROI. The first implementation built a
single **extended mesh** (core + ring) and ran the whole sim on it, then
cropped to the core. Real-site testing (2026‑05‑16/17) showed this is the
wrong shape:

- The extended SfS mesh is ~2.8 M faces. The GPU thermal solver
  (`solvePersistentMultipleCycles`) **deterministically wedges** the
  macOS+AMD driver at that size (was intermittent at the 1.8 M production
  size). Not OOM (384 GB), not a panic, not a code bug — a face-count
  scaling wall.
- Radiation VF on 2.8 M faces = ~60 GB / 605 M CSR entries.
- H‑matrix VF is *not* a mitigation: the template's own history shows it
  OOM‑grows on >1 M-face meshes; MC sparse is the lean path.
- The crop step and the extended-mesh density calibration exist only to
  service this wrong shape.

The ring was always meant to occlude sunlight, not to be thermally solved.
This doc develops that original intent.

## Design

**Two meshes, no merge, no crop, no subsetting.**

| Mesh | Source | Role | Face count |
|---|---|---|---|
| `core.ply` | SfS DEM, **standard R3km build** (today's production mesh) | thermal + radiation + output, **unchanged** | ~1.8 M (proven-stable) |
| `ring.ply` | coarse LOLA annulus R_core→R_ext | **occluder triangles in the illumination ray-trace BVH only** | ~10⁵ (coarse) |

The simulation runs on `core.ply` *exactly as production does today* — every
`N_faces` assumption in thermal, radiation, statistics, VTK/binary output,
the GPU persistent kernel, and the GUI is untouched. The **only** new
behavior: the illumination model adds the ring triangles to its shadow-ray
acceleration structure, so direct-solar occlusion on core faces accounts for
distant terrain.

### Why this is the right shape

- **GPU stability**: thermal/radiation stay at the proven ~1.8 M production
  scale. The 2.8 M-face wedge problem disappears by construction.
- **No crop**: thermal output is natively the core ROI. `a3gt_v2_crop_core`
  and the extended-mesh density calibration (`a3gt_v2_extended_mesh`
  budget/p sweeps) become unnecessary for this path.
- **Minimal, isolated C++**: one subsystem (illumination BVH), not the
  fragile thermal/radiation/output code.
- **Ring is cheap**: occluders only need to block rays. Coarse LOLA-native
  facets (~80–160 m) suffice; ~10⁵ triangles, trivial BVH cost.

### Shared-face-set assumptions (from code map)

Illumination (Embree `illumination_model.cpp:237‑441` and GPU BVH
`:344‑431`), radiation (`radiation_exchange.cpp:780‑916`, F is
N×N), thermal (`ptm_thermal_sim.cpp:641‑692`,
`runThermalModelsForAllFaces_GPU_Persistent`, arrays `[face*tsteps]`), and
output all assume `N_faces == mesh->faces`. **This design keeps that true**:
`mesh->faces` = core only. The ring is *not* in `mesh`; it is a separate
triangle set handed only to the illumination BVH.

## C++ touch points (minimal, illumination-only)

1. **`config_parser.{h,cpp}`** (`MeshParams`, ~`config_parser.h:28‑37`): add
   `std::string occluder_file = "";`; YAML `mesh.occluder_file`; CLI
   `--occluder-file` (parse near `ptm_thermal_sim.cpp:7816+`).
2. **`IlluminationModel`** (`illumination_model.{h,cpp}`):
   - `initializeScene(vertices, faces)` (`:237‑441`) → add optional
     `occluder_vertices`, `occluder_faces`. Append occluder triangles to the
     Embree scene (second `rtcAttachGeometry`, or concatenated buffers) and
     to the GPU BVH input (`:344‑431`, `bvh_opencl` build arrays).
   - Occluders are appended **after** core faces; their hit IDs are
     ≥ `N_core` and are never indexed into flux/thermal arrays. The shadow
     test only needs *any-hit* ("is the sun blocked"): a ray from a core
     face that hits an occluder = shadowed — already the correct semantics.
   - Audit `intersectsFirst` consumers (`:443‑450`) for self-intersection
     epsilon keyed on face id; ensure `id ≥ N_core` is treated as
     "occluder → shadowed", never as a lit/solved face.
3. **`ptm_thermal_sim.cpp` main()** (~`:8319‑8436`, `:8698‑8770`): after
   loading the core mesh, if `mesh.occluder_file` set, load it with the
   existing mesh loader into a throwaway `Mesh`, and pass its
   vertices/faces to `illumination_model` scene init as occluders. Radiation
   manager, thermal solver, statistics, output: **no change** (core mesh).

No change to: thermal solver, radiation VF, statistics, crop (removed for
this path), GUI, binary/VTK output.

## Orchestration (Python)

1. **Core mesh** = the existing standard R3km SfS build (the proven
   production mesh; no extended-density calibration needed). Existing
   `a3gt_v2_orchestrator.build_mesh` legacy path already produces this.
2. **Ring mesh** = coarse LOLA annulus. Build a LOLA-only (or SfS+LOLA)
   coarse disk to R_ext via the existing VRT path at a coarse
   `--roi-max-points`, then **post-filter out faces within ~0.95·R_core**
   (reuse `a3gt_v2_crop_core.geodesic_km`/mask, inverted) → `ring.ply`.
   No `dtm_to_ply_delaunay` change (no `--roi-inner-radius` needed).
3. **R_ext** sizing: keep the physics-horizon `extended_mesh_params`
   (relief / tan(sun_elev_min), 20 km cap) — it now sizes the *ring*, not a
   thermal mesh. The 1.4 M budget / p calibration is **not** needed (the
   ring is coarse-by-design; the core is the standard production mesh).
4. Orchestrator: per site emit `mesh.import_file=core.ply`,
   `mesh.occluder_file=ring.ply`. Retire `--extended` crop wiring and the
   merge-side crop call for this path.

### Kept vs retired from prior work

- **Kept & useful**: `dem_catalog.py`, `dem_vrt.py` (+`-te` clip), the
  staged canonical LOLA base, physics-horizon R_ext, the catalog/association
  manifest. These now feed the ring build.
- **Retired for this path**: `a3gt_v2_crop_core.py` (no crop), the
  extended-mesh density calibration / budget sweeps (core = standard mesh).
  Leave the modules in the tree (provenance) but unused by the production
  path.

## Physics simplification (needs sign-off)

The ring is occluder-only → **distant (R_core→R_ext) terrain IR
re-radiation and reflected-solar onto the core are dropped.** Justification:
- Near crater walls/rim (within the 3 km core) are *in the core mesh*, so
  their reflected-solar + IR onto PSR floors is **fully retained** (core
  radiation unchanged) — this is the dominant warming term for cold traps.
- Distant 3–20 km terrain contributes to the core mainly via **direct-sun
  occlusion**, which this design **retains** (that was the whole point).
- Distant-terrain IR onto the core is geometrically small (∝ 1/r², low view
  factor) and second-order vs the near-field already modeled.

If a site needs distant-terrain IR, a follow-up could add a cheap one-way
"ring as grey IR source at an analytic equilibrium T" term without putting
the ring in the thermal solve — out of scope here.

## Validation plan

1. **Occlusion parity**: a core run with `occluder_file` = a ring containing
   a known massif shows shadowing on core faces that a no-ring run does not;
   spot-check shadow geometry vs sun vector.
2. **Zero-regression**: core run with no `occluder_file` is byte-identical
   to today's standard production run (occluder path is purely additive).
3. **GPU stability**: A03MM022 + A03SP008 full 8-window rad-coupled at
   production core scale — no wedge (same as current production).
4. **End-to-end**: orchestrator builds core+ring, runs, output is the core
   ROI directly (no crop), GUI/COG read it unchanged.

## Open decisions for sign-off

1. **Physics**: accept dropping distant (3–20 km) IR/reflected onto the core
   (near-field retained)? [recommended: yes]
2. **Core mesh**: use the standard R3km production mesh as the core (proven,
   no recalibration) vs a custom-density core? [recommended: standard]
3. **Ring build**: Python post-filter of a coarse LOLA disk (no C++
   mesh-tool change) vs add `--roi-inner-radius` to `dtm_to_ply_delaunay`?
   [recommended: Python post-filter]
