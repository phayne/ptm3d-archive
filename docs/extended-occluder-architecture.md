# Core mesh plus occluder ring

## What it is

High-resolution site studies need distant topography — massifs and crater rims
out to the physics horizon, typically 20 km — to shadow the region of interest,
but that terrain does not need to be thermally solved. Meshing it at the core's
resolution multiplies the face count several-fold and drives the thermal solver,
the view-factor matrix and the output volume past what the machine can carry: a
2.8 M-face core plus ring costs roughly 60 GB and 605 M stored view-factor
entries, against about 1.8 M faces for the core alone.

The occluder-ring architecture separates the two roles. The simulation runs on
the core mesh exactly as it would without a ring; a second, coarse mesh is handed
to the illumination ray tracer as shadow-casting geometry only.

| Mesh | Source | Role | Typical size |
|---|---|---|---|
| core | site DEM at production resolution | thermal, radiation, statistics, output | ~10<sup>6</sup> faces |
| ring | coarse polar DEM annulus, inner radius to physics horizon | occluder triangles in the shadow-ray BVH only | ~10<sup>5</sup> faces |

Ring triangles are appended to the ray-tracing acceleration structure after the
core faces, so their primitive IDs are at or above the core face count and are
never indexed into a flux, temperature or output array. The shadow test needs
only an any-hit query — a ray from a core face that hits a ring triangle is
shadowed — which is already the correct semantics.

Everything downstream of illumination is unchanged: `n_faces` still equals the
core mesh's face count in the thermal solver, in the N×N view-factor operator, in
the statistics and in the VTK and binary output.

## Using it

```yaml
mesh:
  import_file:   core.ply     # solved
  occluder_file: ring.ply     # shadow-casting only
```

An empty or absent `occluder_file` skips the whole path, and the run is
byte-identical to a core-only run. When the file is set but cannot be loaded the
run warns and continues **without** distant-terrain shadowing rather than failing
silently; the log line names the file.

A successful load reports the geometry it registered:

```
Loading occluder mesh (shadow-only): ring.ply
Occluder mesh registered: 61490 verts, 121872 tris (shadow BVH only)
```

`mesh.scale_factor` is applied to the occluder as well as the core, so both
meshes must be in the same length unit.

This is a CPU (Embree) and GPU ray-tracing feature; it is not conditional on the
GPU build.

## Building the ring

`scripts/build_lola_ring.py` builds a coarse annulus from a polar DEM. Raw polar
PDS `.IMG` products store scaled integers, so pass `--elev-scale` for those; the
script aborts rather than assuming a scale of 1.0 when the raster is an integer
type and carries no scale metadata. See
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

Coarse facets are adequate: occluders only have to block rays, so DEM-native
resolution of 80-160 m over the annulus is enough, and the added BVH cost is
small.

`scripts/a3gt_v2_ring_mesh.py` is the A3GT campaign path (a coarse full disk
from the site's clipped VRT, then the inner hole).

### The inner hole must be registered to the core mesh

The ring's inner hole is **not** a nominal circle of radius `R_core`. It is
clipped against the core mesh's **actual outer boundary polygon**, which is
what `--core-ply` supplies; `scripts/ring_registration.py` extracts that
polygon, performs the clip, and gates the result. Both builders refuse to
emit an unregistered ring unless `--allow-unregistered` is passed.

This is not cosmetic. Both builders used to keep a ring triangle whenever its
**centroid** lay outside a nominal `R_core` circle. A kept triangle's
inward-most vertex then protrudes inside `R_core` by up to its
centroid-to-vertex distance -- 100-130 m at LOLA ring resolution -- so coarse
ring surface ends up lying planimetrically on top of the core's own terrain.
At lunar polar sun elevations of 0-5 deg, that overlapping lip casts a hard,
near-permanent spurious shadow on core edge faces. Measured at A03MP026:
2,730 core faces covered by ring geometry, a severe-cold rate of 0.165 under
the lip against 0.031 on clear edge faces, and no severely cold edge face
anywhere more than 200 m from the lip. The same defect was present at every
A3GT ring measured (A03N1016 97 m, A03SP010 88 m, and MP026's wide sweep ring
78 m).

Registration trades the overlap for a small **gap**: the clip drops whole
triangles, so the ring's inner boundary sits a fraction of one ring facet
outside the core rim (at A03MP026, 0.4 m minimum / 46 m median / 167 m
maximum). That band carries no occluder geometry, which can slightly
*under*-shadow a core edge face at grazing sun. The trade is heavily
favorable -- an overlap produces a hard permanent shadow worth 50-300 K,
while the gap omits ~100 m of near-flat LOLA terrain -- but it is a trade, and
the gate prints the per-azimuth gap profile so it can be inspected.

Re-clipping a ring never requires a view-factor rebuild: the ring is
shadow-only geometry and never enters a VF matrix.

## What the ring does not do

The ring is occluder-only, so **thermal IR re-radiation and reflected sunlight
from the distant annulus onto the core are not modeled**. This is a deliberate
approximation:

- Near walls and rims inside the core mesh are thermally solved, so their
  reflected sunlight and IR onto shadowed floors is fully retained. That is the
  dominant warming term for cold traps.
- Distant terrain contributes to the core mainly through direct-solar occlusion,
  which the ring does retain.
- Distant-terrain IR onto the core falls as $1/r^2$ and carries a small view
  factor, so it is second order against the near field already modeled.

If a study needs distant-terrain IR, the ring would have to be added to the
radiation exchange as a one-way source; it is not today.

## Verifying a ring

1. **Occlusion is active.** A core run with a ring containing a known massif
   shadows core faces that a ring-free run leaves lit. Check the shadow geometry
   against the Sun vector at a known epoch.
2. **No regression without a ring.** A core run with no `occluder_file` is
   byte-identical to the same run before the ring existed; the path is purely
   additive.
3. **Ring elevations are right.** `scripts/audit_ring_horizon.py` compares the
   ring's horizon profile against the source DEM. A ring built with the wrong
   elevation scale reads as a systematically over-tall horizon and
   over-shadows grazing sun.
4. **The inner hole is registered.** Run the gate on any ring before using it:

   ```
   python3 scripts/ring_registration.py --core core.ply --ring ring.ply \
       --site-lat <lat> --site-lon <lon>
   ```

   It exits non-zero if any ring geometry lies inside the core's outer
   boundary, and prints the per-azimuth seam-gap profile either way. A ring
   that fails this test produces a discontinuous cold rim on the core's edge
   faces that no mesh-quality, coverage or face-count check will catch.
