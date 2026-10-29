# (101955) Bennu — what is in this directory, and what runs use

## `g_06370mm_alt_obj_0000n00000_v021.obj` — live since 2026-08-28

The OSIRIS-REx global shape model of Bennu, OLA version 21, at 6.37 m ground
sample distance. **49,152 facets, 25,350 vertices, already in kilometers.**
`examples/quickstart_asteroid_bennu.yaml` loads it directly
(`generic_body.shape.mesh_file`), and the GUI's curated Bennu card offers it.

| | |
|---|---|
| archive | PDS Small Bodies Node, `urn:nasa:pds:orex.altimetry:data_derived_altimetry_lidar_global_models:g_06370mm_alt_obj_0000n00000_v021.obj`, version 1.0 |
| source URL | `https://sbnarchive.psi.edu/pds4/orex/orex.altimetry/data_derived_altimetry_lidar_global_models/global_digital_terrain_models/OLAv21/g_06370mm_alt_obj_0000n00000_v021.obj` |
| PDS4 label | the same URL with `.xml` appended |
| citation | Barnouin, O.; Daly, M.; Espiritu, R.; OSIRIS-REx Altimetry Working Group (2026), *Bennu Shape Model, g_06370mm_alt_obj_0000n00000_v021.obj* — the archive's own `Citation_Information` |
| shape references | Barnouin et al. (2019), *Nature Geoscience* **12**, 247–252, doi:10.1038/s41561-019-0330-x; Daly et al. (2020), *Science Advances* **6**, doi:10.1126/sciadv.abd3649 |
| size | 3,741,261 bytes |
| md5 | `bd1b8b152c20ac46a8509640b0ce6ae7` |
| sha256 | `4a998173386899b3c6aeced38450b656e146a4e4f7ecfb9ad7b524d97f840e73` |
| retrieved | 2026-08-28 |

**Shipped byte for byte as the archive serves it.** Nothing was decimated,
recentered, rewound or cleaned. That is the point of the checksums above: they
are the archive's bytes, so anyone can re-download and compare. Every number
below is a measurement of this file, not of something derived from it.

### Reading the file name

`g_06370mm_alt_obj_0000n00000_v021.obj` is the mission's own product name and
it is kept rather than renamed, because the name IS the provenance:

| field | value | meaning |
|---|---|---|
| `g_` | global | as opposed to `l_`, a local or tile product |
| `06370mm` | 6.37 m | ground sample distance |
| `alt` | altimetry | the OLA (laser altimeter) pipeline |
| `obj` | shape model | the base triangulated model, as opposed to `ptm` |
| `0000n00000` | 0 N, 0 E | tile center; global products carry the null tile |
| `v021` | OLA v21 | model version |

### Why this product, and not the alternatives

**Why the PDS archive and not the NASA SVS page** (`https://svs.gsfc.nasa.gov/5069`).
SVS publishes the same lineage as a visualization export: OLA v21 PTM at 17.9
million polygons (816 MB OBJ) and OLA v20 PTM at 3.4 million (195 MB). Both are
two to three orders of magnitude past what a thermal run needs, both are the
Poisson variant rather than the base model, and neither carries a PDS4 label. The
SBN archive serves the primary product, in a published resolution ladder, natively
in OBJ and kilometers, with a label that states the frame and the facet counts.
SVS was still useful: it is what identified OLA v21 as the current model.

**Why `alt_obj` and not `alt_ptm`.** The OSIRIS-REx Altimetry SIS (UA-SIS-9.4.4-307
Rev 2.2, §5.2.9) is explicit: *"To begin using the global (`g_`) files, you first
look for the OBJ file. This is the starting point to any shape model built by the
ALTWG."* The `ptm` products are a Poisson reconstruction whose purpose is to
represent boulder overhangs; they are published only at 3.25 m and finer (17 MB
and up), carry irregular facet counts, and resolve a feature class that does not
exist at 6.37 m sampling.

**Why v21 and not v20 — measured, not assumed.** The v20 file at this same
resolution declares, in its own header, `DATASRC = SPC` and
`PRODNAME = g_06370mm_spc_obj_0000n00000_v020.fits`: that rung of the v20 ladder
came out of the stereophotoclinometry pipeline. The v21 file declares
`DATASRC = OLA`, `DATASRCV = ola-v21`. Only v21 is the laser-altimetry model at
this resolution, and SIS §5.2.9 directs users to the latest version. The two
differ by 0.06% in volume and 0.02% in volume-equivalent radius, so this choice is
about provenance rather than about the numbers.

**Why 6.37 m, and no decimation.** The producer publishes the ladder, so the right
move is to take a rung rather than decimate a finer one — a decimator has to be
trusted, and grid vertex-clustering is known in this project to produce meshes
that pass every coverage and face-count check while being topologically broken.
The full v21 `alt_obj` ladder, with the facet counts read from the PDS4 labels:

| GSD | vertices | facets | archive size | disposition |
|---|---|---|---|---|
| 12.60 m | 6,534 | 12,288 | 929 KB | too coarse |
| **6.37 m** | **25,350** | **49,152** | **3.6 MB** | **shipped, tracked** |
| 3.25 m | 99,846 | 196,608 | 14 MB | optional fetch, `bennu-shape-hires` |
| 1.68 m | 396,294 | 786,432 | 56 MB | not registered |
| 0.88 m | 1,579,014 | 3,145,728 | 225 MB | not registered |

6.37 m sits in the same facet class as this project's other production asteroid
meshes (Ousha runs an 81,920-face ico6) and is small enough to track in git. The
3.25 m rung is registered as an optional download in `data/data_manifest.json`
(`python3 scripts/fetch_data.py --dataset bennu-shape-hires`) for the reason
measured below: **volume converges at 6.37 m but surface area does not.**

### Validation of the mesh

`python3 scripts/validate_shape_model.py <this file> --body-radius-km 0.2447
--expect-closed`. Every quantity is a closed-form integral over the facets.

| quantity | Nolan radar (old) | OLA v21 6.37 m (new) | change |
|---|---|---|---|
| facets | 2,692 | 49,152 | +46,460 |
| vertices | 1,348 | 25,350 | +24,002 |
| mean edge length | 25.60 m | 6.373 m | −75.1% |
| volume (km³) | 0.062265648668 | 0.061297244689 | −1.555% |
| **R_vol** | **245.878 m** | **244.597 m** | **−0.521%** |
| surface area (km²) | 0.78546767958 | 0.80532433977 | +2.528% |
| extent, model axes (m) | 566.5 × 535.9 × 508.8 | 565.4 × 543.5 × 496.0 | |
| extent, principal axes (m) | 566.4 × 535.9 × 508.8 | 561.3 × 540.4 × 495.8 | |
| boundary edges | 0 | 0 | closed both |
| non-manifold edges | 0 | 0 | |
| inward-wound facets | 0 | 0 | |
| \|centroid\| / R_vol | 1.79e−04 | 5.54e−03 | |
| max-inertia axis off +z | 0.002° | 0.684° | |

**The tool agrees with the producer to twelve significant figures.** The OBJ's own
comment block carries the volume, surface area, centroid, principal axes and
extents that the ALTWG computed, and `validate_shape_model.py` reproduces all of
them (for example volume `0.0612972446892` against the header's
`0.06129724468916739`). That agreement is what makes the rest of this table worth
reading: two independent computations of the same integrals.

**Closed, and the engine will say so.** 0 boundary edges, 0 non-manifold edges;
the header's own `Surface Closed? = Yes` agrees. The engine decides open versus
closed by hashing undirected edges as pairs of vertex *indices*
(`src/cpp/illumination/illumination_model.cpp:454`), and that verdict gates the
analytic body-occlusion pre-check, so it is checked here by the same index rule
rather than geometrically.

**Units.** Median |face center| = 0.2398, far below the engine's 10,000
kilometers-versus-meters discriminator and far below its ambiguous band, and
`R_body = 0.2447 km` confirms it independently (ratio 0.9799). The run prints
`[mesh-units] inferred KILOMETERS ... confirmed by R_body`. No `mesh_scale` is
needed and none is set.

**Frame — nothing is rotated.** The PDS4 label states it directly: *"The origin is
the center of mass, and the axes are the principal axes of the shape model... The
z-axis is the spin axis or positive pole. The units are kilometers."* This is the
same convention the radar model shipped in ("center of mass origin, principal-axis
frame, z = spin (long) axis"), so replacing one with the other changes no frame
and the configured spin pole is unchanged. PTM3D consumes the mesh in the
body-fixed frame and applies the pole from `generic_body.spin`; there is no
rotation step for either model.

Two frame details worth stating rather than discovering later:

- **The in-plane principal axes are 7° off the model x and y, and that is
  correct.** Bennu is a near-oblate spinning top: the two in-plane moments differ
  by 3.4% (I₀ = 0.0013981, I₁ = 0.0014462), so they are nearly degenerate and their
  orientation is not physically determined. The label says as much — *"the
  principal axes are not uniquely determined due to the symmetry of the shape"*.
  The axis that matters is well determined: the maximum-inertia axis lies **0.684°
  from +z**. The producer's own header lists the same 7°.
- **The shape model's +z is the angular-momentum pole, which is not IAU north.**
  Bennu is a retrograde rotator (obliquity ≈ 178°). SIS §5.2.6 warns that SPICE
  PCK files use the IAU pole, *"opposite to the +z direction in the Bennu
  coordinate system"*. This repository's configured pole (RA 86.639°, Dec −65.109°,
  from ecliptic λ, β = 45°, −88°) is the angular-momentum pole, so it agrees with
  the mesh's +z. Anyone importing a Bennu PCK pole instead would be 180° out.

**772 unreferenced vertices, and why they stay.** The file carries 772 duplicate
vertices that appear in no facet, so V − E + F = 774 rather than 2; the producer's
header flags this itself. They are inert for the engine — 0 duplicates among the
24,578 *referenced* vertices, so no seam is split, and V − E + F = 2 on the
referenced subgraph. Stripping them would make the file tidier and would break the
checksums above, which is the wrong trade for an archive product. The engine's OBJ
parser keeps unreferenced vertices in the array and never indexes them
(`src/cpp/mesh/mesh_utils.cpp:1265`).

**One number does not match, and it is not a resolution artifact.** This project
carried 245.03 m as Bennu's expected volume-equivalent radius. The v21 model gives
**244.597 m**, low by 0.43 m (0.18%). The obvious explanation — that a 6.37 m
tessellation chords across convexities and loses volume — was tested and
**refuted**: the 3.25 m rung, with four times the facets, gives 244.615 m, moving
the deficit by only +0.018 m (0.007%) when the truncation hypothesis predicts
roughly a fourfold reduction. The v20 model gives 244.645 m. So ~244.6 m is the
OLA model's own volume-equivalent radius, and the 245.03 m figure comes from a
different model or a different definition. It is recorded here as an open
provenance question rather than reconciled by assertion; it is 0.18%, which is
below the level at which it changes any thermal result.

**Surface area is not resolution-converged, and volume is.** Between the 6.37 m
and 3.25 m rungs, R_vol moves 0.007% while surface area moves +4.1% (0.8053 →
0.8383 km²). Facet-scale roughness keeps accumulating area as the sampling
refines, exactly as it should. Bulk quantities are safe at 6.37 m; anything that
integrates over facet-scale slope — self-heating, beaming, roughness — is not, and
that is why the 3.25 m rung is registered as an optional fetch rather than merely
mentioned.

### One behavior change a run will actually print

On the quickstart configuration the new mesh reports **10 faces of 49,152 that
receive no sun over the output window**, and 8 of those seed at the 2.7 K CMB
floor; the radar model reported 2,692 of 2,692 ever lit and no such warning. This
is resolution, not a defect, and it was checked rather than assumed: 7 of the 10
sit at latitude −85.8° to −87.5° with local slopes of 0.9° to 5.7°, so they are
polar facets in grazing illumination, not steep pockets in shadow. Bennu's
obliquity is ≈178°, which keeps the sub-solar latitude within a couple of degrees
of the equator, so the polar facets are permanently near their own horizon; a
25 m facet spans enough terrain to catch some grazing sun, a 6.4 m facet need
not. Read it as "unlit over this run's 192-step, roughly four-rotation output
window", which is what was measured — not as a permanent shadow, which a
four-rotation window cannot establish.

### Not covered by the checks above

The mesh is validated; the *thermal consequences* of changing it are not. No
old-versus-new temperature comparison was run, and none should be inferred: the
new model has 18× the facets and 1/4 the edge length, so a diurnal field on it is
not comparable to one on the radar model face by face. The acceptance run
confirmed the curated card runs to completion on this mesh, not that any
previously published Bennu number is reproduced. There are no published PTM3D
Bennu results to reproduce.

## `bennu_radar_shape.obj` — ARCHIVED since 2026-08-28, not loaded by any run

Nolan et al. (2013) radar + lightcurve shape model, PDS Small Bodies Node,
`urn:nasa:pds:ast-bennu.radar.shape-model::1.1`, doi:10.1016/j.icarus.2013.05.028.
1,348 vertices, 2,692 facets, km, md5 `df9cbba2f8d59d5e50fa73d85636767e`.

It is a pre-encounter model: radar plus lightcurve inversion from 2013, before
OSIRIS-REx arrived. It was the best available shape for Bennu until the mission
returned one, and it is now superseded by the OLA model above.

It is retained, under its original name, because it is the shape behind every
Bennu number published from this repository before 2026-08-28, and because
`docs/tutorials/notebooks/04_asteroid_spice.ipynb` reads this exact path in its
own cells — the notebook deliberately prints this model's header while discussing
where an asteroid shape comes from, and it names the file directly. Deleting or
renaming it would break a shipped, executed notebook.

**It is not on any engine path.** `examples/quickstart_asteroid_bennu.yaml` and the
GUI's curated registry both name the OLA model.

## `ephemeris.spk` — PROVENANCE ONLY since 2026-08-27, not loaded by any run

This is JPL small-body solution **76** for Bennu, dated 2013-02-01:

| | |
|---|---|
| identity | byte-identical to NAIF's `bennu_refdrmc_v1.bsp` **and** to `sb-101955-76.bsp` — md5 `1c1b36831546de7434af83602756b962` for all three, 25,600 bytes |
| contents | body 2101955 relative to the Sun (10) |
| coverage | 2015-01-01 to 2023-05-31 (TDB) |
| sources | `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/bennu_refdrmc_v1.bsp` and `https://naif.jpl.nasa.gov/pub/naif/ORX/kernels/spk/sb-101955-76.bsp` |

It is retained, under its original name, for three reasons: it is the ephemeris
behind every Bennu number published from this repository before 2026-08-27; it
is the ephemeris the `asteroid_bennu_spice_cpu` validation case was banked on
until that date; and `docs/tutorials/notebooks/04_asteroid_spice.ipynb`
furnishes this exact path in its own cells, so deleting or renaming it would
break a shipped, executed notebook.

**It is not on any engine path.** `data/kernels/bennu.tm` no longer names it.

## What replaced it

`data/kernels/spk/sb-101955-118_long.bsp` — JPL solution **118**, the final
OSIRIS-REx reconstruction (Farnocchia et al. 2021,
doi:10.1016/j.icarus.2021.114594), covering **1900-01-01 to 2135-10-01**.
80.7 MB, so it is fetched rather than committed:

```bash
python3 scripts/fetch_kernels.py --bodies bennu
```

`data/kernels/bennu.tm` carries the URL, the SHA-256, the file's own
description, and the reason NAIF's 2015–2023 copy of the same solution was not
used.

## How much the two ephemerides differ

Measured on this machine, 2026-08-27, engine `51bf1688b0`, everything except
the kernel held fixed.

**Inside the overlap** (`spkdiff`, 1000 samples over 2015-01-01 → 2023-05-30):
maximum position difference 31.97 km, mean 20.91 km — relative 1.7e-7. Over the
tutorial's own window on 2019-01-10: 13.93 km, of which 6.36 km is heliocentric
distance, so insolation moves by 9.5e-8 and the solar direction by 5.4e-6 deg.
On the `asteroid_bennu_spice_cpu` validation case that is 7 of 16 output files changed, at
most **2.14e-4 K** in temperature and 2.44e-4 W/m² in direct solar flux — one to
a few float32 ULP. Physically nothing; byte-identity, everything.

**Outside the overlap** is the reason for the change. At 2026-01-01 the old file
has no data, and `IlluminationModel::getSunPosition`
(`src/cpp/illumination/illumination_model.cpp:2818`) falls back to the config's
Keplerian elements **without printing anything** — the banner still says
`Ephemeris: SPICE` and the run still exits 0. Measured against the real
ephemeris at that epoch, that fallback was wrong by:

| quantity | error |
|---|---|
| heliocentric distance | 0.0919 AU (13.75 million km), 1.1370 AU vs 1.0451 AU |
| solar direction in the body frame | 23.5 deg |
| insolation | +18.3% |
| direct solar flux | up to 196 W/m² (mean 144) |
| surface temperature | up to **23.9 K** (mean 16.3 K on `temperature_max`) |

A *missing* kernel, by contrast, is loud: `SPICE(NOSUCHFILE)`, `FATAL: SPICE
initialization failed`, rc=1 — also measured. It is only the out-of-coverage
case that is silent.
