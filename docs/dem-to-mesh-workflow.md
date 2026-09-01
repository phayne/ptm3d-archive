# DEM to mesh: turning a GeoTIFF DTM into a PTM3D mesh

This page covers the whole path from a digital terrain model (DTM) raster to a
PLY mesh `ptm_thermal_sim` can run on — from the command line and from the GUI
dashboard's **DEM module**.

Read the [Status and known gaps](#status-and-known-gaps) section before you
plan work around this toolchain. One of the three advertised capabilities
(adaptive mesh refinement) is **not implemented**, and this page says exactly
what to use instead.

---

## 1. What the tools are

Two binaries, both built from this repository into `build/bin/`:

| Binary | Source | Does |
|---|---|---|
| `dtm_to_ply_delaunay` | `src/cpp/tools/dtm_to_ply_delaunay.cpp` | GeoTIFF DTM → PLY. Delaunay triangulation of the projected point cloud, region-of-interest extraction, radial density grading, edge filtering, mesh quality control. **This is the working tool.** |
| `dtm_advanced_processor` | `src/cpp/tools/dtm_advanced_processor.cpp` | Intended: ROI extraction + adaptive radial mesh refinement. **Its mesh pipeline is unfinished; it refuses to write a mesh and exits 2.** |

A third, `dem_mesh_generator` (`src/cpp/tools/dem_mesh_generator.cpp`), is a
separate, older DEM→mesh path referenced by `data/external/README.md` and
`data/dem/copernicus/README.md`. It is **not** what the GUI's DEM module runs,
and it is not covered here.

All three need **GDAL** at build time. Without GDAL, CMake prints

```
CMake Warning: GDAL not properly configured. SKIPPING these targets:
dem_mesh_generator, dtm_to_ply_delaunay, dtm_advanced_processor. ...
```

and configures successfully without them. Install GDAL and **re-run
`cmake -S . -B build`** — the skip is a configure-time decision.

### Build

```bash
# Prerequisite (macOS/Linux, recommended):
conda install -c conda-forge gdal

cd <repo>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dtm_to_ply_delaunay dtm_advanced_processor -j 8
```

`dtm_to_ply_delaunay` also uses [CDT](https://github.com/artem-ogre/CDT) 1.4.4,
a header-only Delaunay library vendored at `third_party/CDT/` (MPL-2.0). There
is nothing to install for it. That target is deliberately compiled with
`-fno-fast-math -ffp-contract=off`, overriding the project-wide `-ffast-math`:
CDT's exact-arithmetic geometric predicates require IEEE-754 semantics, and
reassociation or FMA contraction can silently corrupt the triangulation on
near-degenerate triangles — precisely the ones that ROI thinning produces.

---

## 2. The quickest real example

Convert a whole DTM at half resolution:

```bash
./build/bin/dtm_to_ply_delaunay input_dtm.tif out_mesh.ply \
    --downsample 2 \
    --body-radius 1737400
```

Extract a 2 km region of interest around a site, with a point budget and a
full-resolution core:

```bash
./build/bin/dtm_to_ply_delaunay input_dtm.tif site_mesh.ply \
    --body-radius 1737400 \
    --roi-center -89.95 0.0 \
    --roi-radius 2.0 \
    --roi-max-points 8000 \
    --roi-core-frac 0.3
```

The second command is the one that matters: it is how you get a mesh that is
fine where the physics is and coarse where it is not.

---

## 3. `dtm_to_ply_delaunay` — every flag

```
Usage: dtm_to_ply_delaunay input.tif output.ply [options]
```

### Sampling

| Flag | Default | Meaning |
|---|---|---|
| `--downsample N` | 1 | Take every Nth pixel in each direction before anything else. Applied to the whole raster, ROI or not. |
| `--body-radius METERS` | auto | Body radius used to place vertices in Cartesian coordinates and to measure geodesic distances. Auto-detected from the GeoTIFF's CRS when omitted; pass it explicitly if the CRS is missing or wrong. Moon = 1737400. |

### Region of interest and density grading

| Flag | Default | Meaning |
|---|---|---|
| `--roi-center LAT LON` | (off) | Enables ROI mode and sets its center, in degrees. |
| `--roi-radius KM` | 0 | Hard cutoff: points beyond this geodesic distance from the center are dropped entirely. |
| `--roi-max-points N` | 0 (no budget) | Target vertex budget. The tool binary-searches a falloff strength `beta` so the expected kept-point count lands at or below N. |
| `--roi-core-frac F` | 0.15 | Fraction of `--roi-radius` kept at **full** resolution. `0.3` with a 2 km radius means the inner 0.6 km is untouched and grading starts outside it. Clamped to [0, 1]. If the core alone already exceeds the budget, the tool says so and thins the entire ROI instead. |
| `--density-exp P` | 2.0 | Falloff exponent. Keep probability outside the core is `1 / (1 + beta·d^P)²` where `d` is the normalized distance from the core edge to the rim. 1.0 = linear, 2.0 = quadratic, 0.5 = square root. Lower = gentler falloff. Floored at 0.1. |

Thinning is **probabilistic, not a stride**, which is what makes the density
transition smooth rather than banded. The RNG is seeded with a fixed constant
(42), so the same command on the same input reproduces the same mesh exactly.

### Triangulation

| Flag | Default | Meaning |
|---|---|---|
| `--max-edge METERS` | auto | Drop triangles with any projected edge longer than this. `0` disables filtering entirely (no holes, but long thin triangles across ROI gaps survive). Omit for auto: `pixel_size × downsample × (1 + beta) × 3`. |
| `--swap-axes` | off | Swap X/Y when inverting a polar stereographic projection. Use when a polar DTM comes out mirrored or rotated 90°. |

### Mesh quality control

Runs after triangulation, in this order: sliver removal → normal orientation
(BFS flood fill) → small-hole filling.

| Flag | Default | Meaning |
|---|---|---|
| `--no-qc` | (QC on) | Skip the entire quality-control sweep. |
| `--qc-sliver-angle DEG` | 5.0 | Remove **boundary** triangles whose minimum angle is below this. Interior slivers are kept — removing them would open holes. |
| `--qc-max-hole N` | 20 | Fill boundary loops of at most N edges (triangle fill for 3, centroid fan otherwise). `0` disables hole filling. |

### Not implemented

| Flag | Status |
|---|---|
| `--south-pole-coords` | **Accepted but a no-op.** It is parsed and stored, and then never read. It has never applied any coordinate correction. Meshes built with and without it are byte-identical. The tool now prints a warning when you pass it. For a genuine polar-stereographic axis swap use `--swap-axes`. |

Unrecognized or incomplete arguments are no longer dropped silently — the tool
warns on stderr and continues.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | A mesh with at least one face was written. |
| 1 | The input could not be opened, or the PLY could not be written. |
| 2 | **The mesh came out empty, and no file was written.** |

Exit 2 is worth knowing about. Zero faces always means a request that could not
be honored — an `--roi-center` off the raster (a wrong latitude sign on a polar
DEM is the usual way in), an `--roi-radius` below the pixel size, a `--max-edge`
under the point spacing so every triangle is filtered, or an all-NoData region.
The tool used to write a ~360-byte PLY with `element vertex 0` / `element face 0`,
print `Success!` and exit 0; the GUI keys on the exit code, loaded that empty
mesh and passed it downstream as a converted result. It now refuses, names the
likely causes, and points at its own "Skipped … points" / "Filtered … triangles"
counts.

### What it writes

Binary little-endian PLY:

- **Vertices**: `double x y z`, `float nx ny nz`, `double latitude longitude elevation`.
  The double-precision lat/lon/elevation ride along with the mesh, so downstream
  tools do not have to re-derive them.
- **Faces**: `uchar` count + three `int` vertex indices.

Cartesian coordinates are body-fixed: +X through (0°N, 0°E), +Y through
(0°N, 90°E), +Z through the north pole; radius is `body_radius + elevation`.

---

## 4. `dtm_advanced_processor` — status

This tool is documented here so nobody spends an afternoon on it.

It **runs**, prints a plausible banner, and then refuses:

```
$ ./build/bin/dtm_advanced_processor synth.tif out.ply \
      --refine-center -89.95 0.0 --refine-inner 1 --refine-outer 4 \
      --refine-max-faces 50000 --initial-downsample 4 --radius 1737400

STATUS: adaptive refinement in this tool is NOT IMPLEMENTED.
...
WARNING: --initial-downsample is parsed but NEVER USED (value 4 has no effect).
...
ERROR: adaptive refinement needs an initial mesh, and the
       'create initial mesh from DTM' step is NOT IMPLEMENTED.
       Refusing to write an empty mesh to: out.ply
$ echo $?
2
```

Three pieces were never written by the original author:

1. **ROI → initial mesh.** The ROI extractor runs and produces a real gridded
   elevation patch, and the result is then discarded.
2. **Initial mesh from DTM.** With no input mesh, none is built, so the refiner
   is handed empty vertex and triangle vectors.
3. **The elevation callback.** `getElevation(lat, lon)` returns `0.0f` for every
   query, so subdivision midpoints would sit on a flat datum regardless of the
   terrain.

The `ROIExtractor` and `AdaptiveMeshRefiner` classes themselves are complete and
ported unchanged; only `main()`'s wiring is missing.

**Before this port, the failure was silent.** The original wrote a 432-byte PLY
containing `element vertex 0` / `element face 0`, printed
`Processing completed successfully!`, reported `Average edge length: nan m`, and
**exited 0**. The GUI saw a zero exit code and reported success. That is why the
symptom looked like "refinement produces a broken mesh" rather than "refinement
does not exist". It now exits 2 and writes nothing.

### Use this instead

Everything adaptive refinement was meant to deliver — fine facets at the site,
coarse facets at the rim, under a face budget — is what `dtm_to_ply_delaunay`'s
ROI grading already does:

| Refinement concept | `dtm_to_ply_delaunay` equivalent |
|---|---|
| `--refine-center LAT LON` | `--roi-center LAT LON` |
| `--refine-inner KM` (full resolution radius) | `--roi-core-frac` × `--roi-radius` |
| `--refine-outer KM` (minimum resolution radius) | `--roi-radius KM` |
| `--refine-max-faces N` | `--roi-max-points N` (vertex budget; faces ≈ 2×) |
| falloff shape | `--density-exp P` |
| `--refine-min-edge` / `--refine-max-edge` | `--max-edge METERS` (upper bound only) |

Measured on a synthetic 400×400 south-polar lunar DTM at 20 m/pixel, with
`--roi-center -89.95 0.0 --roi-radius 2.0 --roi-max-points 8000
--roi-core-frac 0.3`:

| Ring from ROI center | Vertices | Vertex density |
|---|---|---|
| 0.0 – 0.6 km (the protected core) | 2822 | 2495 verts/km² |
| 0.6 – 1.2 km | 4280 | 1262 verts/km² |
| 1.2 – 1.8 km | 788 | 139 verts/km² |
| 1.8 – 2.4 km | 95 | 12 verts/km² |

7985 vertices / 15916 faces total (budget 8000 honored), and the farthest vertex
sits 1.9996 km from the center against the requested 2.0 km cutoff. That is a
208× core-to-rim density ratio from one command.

---

## 5. Through the GUI dashboard

```bash
cd gui/
python run_dashboard.py
```

The dashboard is the optional multi-module launcher (`run_gui.py` is the
supported entry point for the thermal application alone; the DEM module lives
only in the dashboard). Pick the **DEM** module in the sidebar.

**Preview page** — `dem_panels/dem_preview_panel.py`

1. **Browse…** and select a GeoTIFF DTM. The panel reads it with the GDAL Python
   bindings (`osgeo`) and draws an elevation map.
   *Without `osgeo` the panel still accepts the file and passes it downstream; it
   just shows an install notice instead of the map. Conversion still works —
   the binaries link GDAL themselves.*
2. Tick **Enable ROI extraction** and set center latitude/longitude and radius,
   or click on the preview to place the center.

**Configuration page** — `dem_panels/dem_config_panel.py`

- *Grid & Body*: downsample factor, body (sets the radius), body radius.
- *ROI*: core fraction, density exponent. The center/radius come from the
  preview page.
- *Triangulation*: max-edge mode (auto / disabled / explicit) and its value.
- *Coordinate corrections*: swap axes; **south pole coords — this checkbox is a
  no-op**, see §3.
- *Quality control*: enable, sliver angle, max hole edges.
- *Output*: the destination `.ply`.

Press **Convert**. The panel streams the tool's own output live, including its
progress percentages, into the status line.

**Mesh page** — `dem_panels/dem_mesh_panel.py`

The refinement controls (center, inner/outer radius, max faces, min/max edge,
initial downsample) map onto `dtm_advanced_processor`. **This action cannot
succeed** — see §4. The module now announces that before launching the tool
rather than letting you discover it from the exit code.

### Where the GUI looks for the binaries

`gui/dem_core/dem_runner.py` searches, in order:

1. `<repo>/build/bin/<name>` — what `cmake --build build` produces. **Preferred.**
2. `<repo>/../dtm/<name>` — the legacy standalone `dtm/` working directory, kept
   only so existing local setups keep working. Binaries there predate the port
   and may be built for a different CPU architecture.
3. `PATH`.

If none matches, the module prints the exact `cmake` commands to build it and
names GDAL as the prerequisite.

### Platform support

The DEM module's conversion and refinement actions stream the child process's
stdout/stderr using `select.select()` on pipes. **That is POSIX-only** — on
Windows, `select` handles sockets only. macOS and Linux work; on Windows the
module reports that it is unsupported and points at the command line, rather
than failing somewhere deep in the read loop. There is no fallback
implementation.

---

## 6. Feeding the mesh to a simulation

Point `mesh.import_file` at the PLY:

```yaml
mesh:
  import_file: site_mesh.ply
```

Two things to check before trusting a DEM-derived mesh, both of which have
bitten this project before:

- **Coverage is not topology.** A mesh can pass every face-count, radial-extent
  and elevation check and still be non-manifold with interior holes. Check the
  boundary-edge count (all boundary edges should be on the outer rim) and the
  non-manifold-edge count (must be 0). Regenerating from the raster at a coarser
  point budget is safer than decimating an existing mesh — re-triangulation is
  clean by construction.
- **Units.** DEM-derived meshes are in **meters**. Asteroid `.obj` shape models
  are in **kilometers**. A frame slip mis-scales ray offsets by ~1000×.

---

## Status and known gaps

| Item | Status |
|---|---|
| DTM → PLY conversion | **Works.** |
| ROI extraction + radial density grading | **Works** (`dtm_to_ply_delaunay --roi-*`). |
| Mesh quality control (slivers, normals, holes) | **Works.** |
| Empty-mesh handling | **Fixed.** Both tools now refuse and exit 2 instead of writing a 0-face PLY and reporting success. |
| Adaptive mesh refinement | **Not implemented** (`dtm_advanced_processor`). Fails loudly, exit 2. Use ROI grading instead (§4). |
| `--south-pole-coords` | **No-op.** Parsed, stored, never read. Warns when used. |
| `--initial-downsample` (refiner) | **Dead.** Parsed, never used. Warns when used. |
| Windows support for the GUI DEM actions | **Not supported** (POSIX pipe I/O). Command line works anywhere the binaries build. |

## Port record

Both sources were brought into this repository from a standalone `heat3d/dtm`
working directory that was never in version control and never in CMake. The port
preserves the algorithms; the only changes are the guards and disclosures listed
above, plus two missing standard-library includes.

Verification: on a synthetic south-polar-stereographic lunar GeoTIFF, the
in-repo `dtm_to_ply_delaunay` produces output **byte-identical** (md5-equal) to
the pre-port standalone binary, for both a full-raster conversion (40000
vertices / 79202 faces, md5 `cc403f9615934558fd6f9188fba9b825`) and an ROI
conversion (7985 / 15916, md5 `cb9082ba8eeae8f204871a9085440f2b`). Both runs log
`Detected POLAR STEREOGRAPHIC projection`, so the parity covers the projected
code path (`polarStereoToLatLon`, `latLonToPolarStereo`), not the unprojected
fallback.

The in-repo build must keep `-fno-fast-math -ffp-contract=off` for that to hold:
with the project-wide `-ffast-math` the float32 vertex normals drift by 1–2 ULP
(≤ 2.98e-7) while every double-precision field and the entire face-index array
stay bit-identical.

The regression test is `src/python/tests/test_dem_tools.py`. It generates the
synthetic DTM itself, so it needs no fixture data. Two things it deliberately
does not assume:

- The CRS is written from a **literal WKT string**, not built with
  `ImportFromProj4`. That call returns 5 with an empty WKT on any host whose
  `proj.db` does not match its PROJ library, and `SetProjection('')` succeeds
  silently — which would leave the fixture projection-less and every geometric
  check quietly measuring the unprojected fallback. The smoke arms additionally
  assert `Detected POLAR STEREOGRAPHIC projection` appears in the tool's output.
- When a binary is absent the affected arms **skip with a message naming the
  missing binary and the `cmake --build` command**, and the runner prints that
  they are not passes.
