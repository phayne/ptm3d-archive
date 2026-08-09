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
