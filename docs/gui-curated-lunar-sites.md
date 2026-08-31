# Curated lunar sites, and finding a view-factor matrix that is already here

Companion to [`gui-curated-body-sources.md`](gui-curated-body-sources.md). That
document covers the tier for bodies whose **ephemeris and shape** a live service
correctly refuses to serve. This one covers the other half of the same problem,
one level down: a **site** on a body, whose terrain mesh, occluder ring and
view-factor matrix are only meaningful together.

Two mechanisms, in `gui/gui_core/`:

| module | what it does |
|---|---|
| `curated_sites.py` | a table of site sets that have been RUN, with everything each one wires and everything it refuses |
| `vf_discovery.py` | for any mesh, ranks the view-factor files already sitting beside it |

Neither imports Qt, so both are measurable without a window.

## The problem

A local-mesh run on the Moon needs three files:

```yaml
mesh:
  import_file:   <site>_R3km_linear.ply          # the terrain
  occluder_file: <site>_ring_Rcore3_Rext<M>km.ply # shadow-only surroundings
radiation:
  view_factor_file: <a matrix built for THAT terrain>
```

Every way of getting the combination wrong produces a run that **finishes**.
From this project's own A3GT campaign, all measured:

* a `view_factors.hmat` with **zero stored nonzeros** — dimensions match, the
  loader is content, and facet-to-facet radiation silently computes nothing;
* a matrix for a mesh that had since been **regenerated**: 1,796,406 rows
  against a 2,397,890-face terrain. Caught, by the count;
* three **radiance-pruned** experimental matrices in the same directory as the
  good one, whose own receipts record *"MEASURED FAIL: PSR-floor Tmax mean
  3.3 K / max 22.6 K cold … Diffuse polar terrain does NOT prune safely"*;
* **three occluder rings for one site**, of which the production runs use the
  27.3108 km one and the offline matrix sweep used the 81.6712 km one — two
  different jobs, and the bigger ring is not the better answer to the first;
* one accidental **rebuild** of a multi-gigabyte matrix, which hard-wedged the
  graphics cards and needed a reboot. That is what PTM-19 exists to prevent, and
  a wrong or missing cache path is how it happens by accident.

None of that is visible in a file picker.

## The curated site tier

`Moon → Local mesh` carries a **Curated site** list. Choosing an entry writes
the whole set at once and prints where each part came from:

| key | value |
|---|---|
| `simulation.body` | the entry's body |
| `mesh.import_file` | the core terrain |
| `mesh.occluder_file` | the ring the **production runs** used |
| `mesh.latitude_override` | `-999.0` — see below |
| `mesh.longitude_override` | `-999.0` |
| `radiation.enabled` | `true` |
| `radiation.view_factor_file` | the **empty base stub** |
| `radiation.use_gpu` | the route the proven campaign ran |
| `engine_env.PTM_HMATRIX_VF_REPLACE` | the replacement CSR |

Two of those rows are not obvious.

**The base stub is not cosmetic.** The engine has no reader for the
replacement-CSR layout in its ordinary cache path; that matrix arrives through
`PTM_HMATRIX_VF_REPLACE` (`radiation_exchange.cpp:280`) while the normal loader
still runs. With `view_factor_file` empty, the loader finds no cache and
**rebuilds** — multi-gigabyte, on a full core+ring lunar mesh. The stub is a
correctly-dimensioned, current-format (`HMAT`/v2), `nnz=0` file that loads in
milliseconds so the rebuild never starts. It must also be an empty stub rather
than a real matrix: a real one leaves two matrices live at once, and the
Gauss-Seidel and GPU-decline paths read the uncorrected one while the banner
names the corrected one (`radiation_exchange.cpp:370`).

**The placement keys are set to the not-placed sentinel on purpose.**
`mesh.latitude_override > -900.0` makes the engine rotate the loaded mesh so its
vertex mean lands on the given point (`ptm_thermal_sim.cpp:15634` →
`rotateMeshToLatLon`). These DEMs are built into lunar body-frame coordinates
and are *already* at their own latitude and longitude — measured, the A03MP026
mesh's vertex mean sits at −84.2234, 33.5023 against a name that says
84.223S/33.502E. Writing the rounded name back would rotate by ~0.0004°, which
is past the engine's 1e-10 radian no-op guard, moving the core about 12 m — and
the occluder ring is **not** rotated with it. A rotation does not change face
areas, so the matrix's geometry-hash gate stays green throughout. Every
production configuration for these sites leaves both keys unset; so does the
registry, and the card shows the coordinates as text instead.

### Local-only entries

The A3GT artifacts are hundreds of gigabytes and are **not distributed** with
PTM3D. Every entry is marked `local_only` with the working tree it needs, and
its status is computed from the disk in front of the user. On another machine
the entry is still listed, says the files are not there, and **writes nothing** —
it does not emit paths and leave the launch check to discover them. A set that
is only partly present is refused for the same reason: a terrain with no matrix
is a run that starts and is wrong.

### Which matrix generation an entry wires

Where several matrices for one site exist, the entry names the ones it does not
use and why — the same discipline as the Bennu tier's superseded-kernel note.
For A03MP026 that includes the July matrix the project map names first
(2,484,716,524 nonzeros, no provenance sidecar, built before the FC-08
symmetrization-divisor fix). The three curated sites are all on the **same**
matrix generation, which is what the GEN-2 campaign ran; mixing generations
across a table of sites would be three runs on two physics with nothing saying
so.

### Adding a site

1. Run the set to completion and check the output. An entry is a claim that this
   happened, not that the files exist.
2. Add a `CuratedSite` to `CURATED_SITES` with every number **read off the
   files**: face counts from the PLY headers, rows/nonzeros/geometry hash from
   the CSR, and the stub's format and emptiness from its 80 bytes.
3. List what the entry does **not** use, and why, in `superseded`.
4. Run `python3 src/python/tests/test_gui_curated_sites.py`. It re-reads every
   number, re-measures the mesh's position from its own vertices, checks the
   provenance sidecar against the matrix beside it, and emits a configuration
   through the engine's `--validate-config`. Checks whose files are absent
   **skip by name** rather than passing.

## Finding a matrix that is already on disk

Selecting any local mesh, on any body, enables **Look for matching view
factors** beside the View-factor cache field. It scans the mesh's own directory
and the directory the cache field already points at — not recursively — and
ranks what it finds.

Nothing is chosen automatically. A view-factor file does not record the surface
it was built for, and two sites at the same posting can produce meshes with the
same face count.

| rank | tier | what was actually checked |
|---|---|---|
| 0 | provenance-verified | a `.csr` whose `.provenance.json` names **this** mesh, whose recorded geometry hash equals the matrix header's, and whose face count equals the mesh's |
| 1 | current-generation sparse | `VFFM` version 2, rows == faces |
| 2 | H-matrix | `HMAT` version 2, rows == faces |
| 3 | replacement CSR | rows == faces, **no** provenance sidecar |
| 4 | superseded near-field generation | `VFFM` version 1, rows == faces |
| 5 | empty base stub | rows == faces and `nnz == 0` |

Refused outright, each with its reason and both numbers where there are two:
a row count that disagrees with the mesh; a cloud placeholder (refused on the
`stat`, before anything reads it and blocks); a truncated or unidentifiable
file; a radiance-pruned matrix whose receipt records a failure; and the five
A3GT matrices PTM-312 records as defective, which are loadable, correctly
dimensioned, and known to be wrong.

Three things the ranking deliberately does **not** claim:

* **The mesh's own geometry hash is never recomputed.** It is
  `sum(A_i) + sum((i+1)·A_i)·1e-9` over every face area, which means reading the
  whole mesh and re-deriving its triangle areas in the engine's face order. The
  engine computes it at load and **aborts** on a mismatch
  (`radiation_exchange.cpp:298`), so a foreign matrix is caught loudly, before
  any physics. Rank 0 says the sidecar and the matrix agree with each other and
  name this mesh.
* **HMAT carries no physics-generation stamp.** The sparse header does
  (`view_factor.cpp:517-525`: identical bytes, version 1 versus 2, different
  near-field estimator, and the engine announces a v1 load). The H-matrix header
  has magic, a format version that has been 2 for the life of the format,
  dimensions and block counts. So an H-matrix candidate is reported as
  *format-compatible, physics generation unverifiable*, every time.
* **The extension is not the format.** In this project the two disagree
  routinely: `~/projects/a3gt/campaign/base_vf/*.hmat` are sparse version-1
  files with zero nonzeros, and the current empty stubs are H-matrix version 2
  named `.vf`. Everything is identified by its magic number. (`MFFV`, which
  PTM-24 calls "the old format", is not a third magic: it is `VFFM`'s four bytes
  as they sit on disk, little-endian.)

A `.csr` chosen this way is wired to `engine_env.PTM_HMATRIX_VF_REPLACE` and the
card says the run **also** needs a base file in the cache field, because a
replacement matrix with no base rebuilds the matrix it was just handed. The
discovery path does not invent one; a curated site brings its own.
