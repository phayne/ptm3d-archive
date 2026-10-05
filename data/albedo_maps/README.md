# Albedo maps (`albedo_maps/`)

Bond-albedo maps for the Moon and Europa: per-face binaries for each icosphere
level (`Moon_Albedo_res3..9.bin`, `Europa_Albedo_res3..9.bin`) plus the source
FITS maps they were sampled from. Used when `illumination.use_albedo_map: true`
(directory set by `albedo_maps_directory`, default `data/albedo_maps`).

**Not in git.** Fetch with:

```bash
python3 scripts/fetch_data.py --dataset albedo-maps   # ~150 MB
```

## Source

- **Moon:** normal-albedo mapping from LRO instruments (Diviner/LROC/LOLA
  heritage), converted to Bond albedo with the project's incidence-angle
  parametrization (Hayne et al. 2017, App.).
- **Europa:** `Europa-0.5deg-BondAlbedo.fits` and `Europa-1deg-BondAlbedo.fits` are
  the Bond albedo map of Mergny et al. (2025), derived from Voyager/Galileo imaging:

  > Mergny, C., Schmidt, F., Andrieu, F., & Belgacem, I. (2025). A Bond albedo map of
  > Europa. *Astronomy & Astrophysics*. https://doi.org/10.1051/0004-6361/202453058

  The title, authors, journal and year were checked against the DOI's Crossref record
  on 2026-10-05. That record carries no volume or article number, so neither is given
  here. `scripts/jwst/plot_per_latitude_histograms.py` samples the same map ("Mergny et
  al. Bond-albedo distribution"). The shipped `Europa_Albedo_res*.bin` are sampled from
  the 0.5° file.
- Original mission data: NASA PDS (public domain).

## Processing

Source maps are regridded to each icosphere's face centers by
`albedo_map_generator`, which samples the source FITS at each face's physical
**+latitude** (north positive) and IAU longitude — the same convention as every
runtime sampler (`ptm_thermal_sim.cpp`, `thermal_model.cpp`, `albedo_manager.cpp`,
the DEM sampler) and the photo-validated Moon camera bin. Each per-face binary is
an 8-byte little-endian face count followed by one float64 (Bond albedo) per face.

## Europa map gaps and how they are filled (PTM-695)

The Europa FITS maps have gaps. Their pixels hold **−1.0**, but the header declares
`MISSING = -9999`, a value that appears nowhere in the data. In the 0.5° map:

| Pixels | Count (of 259,200) | Where | Treated as |
|---|---|---|---|
| −1.0 | 12,190 (4.70%) | every pixel south of −84.5°, a thinning annulus out to −77°, and the two seam columns at 0° and 359.5° E (all latitudes) | fill |
| other negative values (−0.90 to −0.13) | 24 | 12 on the rim of the southern gap, 8 north of 88°, 4 between −80° and −60° | fill (outside the Bond-albedo range) |
| exactly 0.0 | 128 | 114 of them north of 85° | data |
| exactly 1.0 | 321 | all latitudes; 143 of them north of 85° | data |

The 1° map has the same structure (3,304 at −1.0, 9 other negatives, 29 at 0.0, 101 at 1.0).

**How fill is decided.** A pixel is fill if it equals a value the header declares (FITS
`BLANK`, `MISSING` or `NODATA`, each read independently), equals a sentinel configured
for the map (`albedo_map_generator --fill-value V`), lies outside the physical
Bond-albedo range [0, 1], or is not finite. Every engine path that loads an albedo FITS
applies the [0, 1] range, so the −1.0 gaps are fill even without `--fill-value`. The
shipped Europa bins are generated with `--fill-value -1` anyway, so the log attributes
them to the sentinel. Each load prints one line with the count in each category.

**The 0.0 and 1.0 pixels are data.** Both values are pile-ups at the ends of a clipped
range, not sentinels. 321 pixels sit at exactly 1.0, compared with 8 to 60 in each
narrow bin just below it. The orphaned high-resolution Mergny FITS
(19631 × 9816, quantized in steps of 0.0051, and also −1.0 where there are no data)
piles up at 0 and 1 in the same way. Block-averaged onto the 0.5° grid, it does not
show the 1.0 pixels as gaps. None of their blocks is fill (median fill fraction 0), the
high-resolution mean there is ordinary terrain (median 0.646), and that mean is the same
as for the 501 genuine pixels between 0.95 and 1.0 (median 0.644). Their 3 × 3
neighborhoods are ordinary too (median 0.616). They behave as the noisy upper tail of
the data, clipped at 1.0. The same holds when the 143 pixels north of 85° and the 178
south of it are compared separately with their own 0.95-1.0 controls. The −1.0 pixels
are the opposite case: 10,779 of the 12,190 fall in blocks that are entirely fill in the high-resolution map, which also confirms
that the two maps are registered (correlation 0.815 with no shift; 0.73 at one pixel of
shift and 0.35 with the rows flipped).

**North of about 85° the 0.5° map is degraded, though not filled.** Its rows there
repeat a few values (79% of the 89.5° row is a single value), and 832 pixels read below
0.05 where the high-resolution map averages about 0.47. This is where most of the 0.0
and 1.0 pixels are, and 8 of the negative ones. The rule above cannot catch it,
because these are valid albedos in the wrong place (PTM-700).

**The pipeline**, in `SpatialMap` (`src/cpp/maps/spatial_map.cpp`) and
`albedo_map_generator`:

1. **Load and classify.** The FITS is read and its rows flipped (row 0 = north). Every
   fill pixel is rewritten to the map's single missing value.
2. **Heal the longitude seam.** In each row where the two edge columns are fill and the
   nearest interior columns are valid, the edge columns are filled by linear
   interpolation across the seam (`heal_cylindrical_seam`). Before PTM-695 this step
   never fired on Europa. The edge-column detector compares an edge column with its
   neighbor, the neighbor contained −1.0 as well, and so −1.0 stayed in the map as data.
3. **Sample each face at its center** by bilinear interpolation. If any weighted corner
   is fill, the weights are renormalized over the valid corners. A fill pixel is never
   blended in. `getAverageValueInTriangle` (a 7-point barycentric average) follows the
   same rule but is not used by the generator.
4. **Nearest-neighbor fill** applies only to faces with no valid corner in their
   bilinear support. Each such face copies the nearest sampled face, by
   latitude-longitude distance. The generator reports how many faces were renormalized
   and how many were filled.
5. **Post-processing that predates PTM-695.** Any face outside [0.01, 0.99] is replaced
   by an inverse-distance mean of up to 12 neighbors within 30°, and values are then
   floored at 0.01 (`fixOutOfRangeAlbedos`). This also smooths genuine faces next to the
   1.0 pixels (PTM-699).

Regenerate the Europa bins with:

```bash
build/bin/albedo_map_generator data/albedo_maps/Europa-0.5deg-BondAlbedo.fits \
    data/albedo_maps 3 9 --fill-value -1
```

`scripts/jwst/plot_per_latitude_histograms.py` reads the 1° map directly and uses its
own rule, keeping values strictly between 0.02 and 0.98. That rule also drops the 0.0
and 1.0 pixels, along with the gaps.

**The archive is out of date.** The v1 `albedo_maps.tar.gz` deposit carries the bins
made before PTM-695. Until a corrected archive is published, regenerate the Europa bins
locally with the command above. Moon bins are unaffected: they come out byte-identical.

## Lunar map registration (PTM-34 longitude, PTM-35 latitude — corrected)

Two registration defects in the earlier lunar maps are corrected in this
revision:

- **PTM-34 — source FITS longitude (+180°).** The original `Moon-BondAlbedo.fits`
  was mislabeled by 180° in longitude: its WCS declared `CRVAL1=0` (column 0 =
  IAU_MOON lon 0, east, 0..360) but the raster was a −180..180-centered map
  (IAU lon 0 at the center column), so the dark nearside maria landed on the
  bright farside highlands. The FITS is now rolled +180° to match its header and
  IAU_MOON: nearside maria ≈ 0.07–0.09; farside highlands and the Tycho region
  ≈ 0.17. The topographic meshes (`data/topo/`) were always correct and the
  Europa FITS is correct, so the roll is applied to the lunar file only — the
  shared FITS loader is **not** patched. Producer:
  `scripts/fix_moon_albedo_lon_registration.py`; verify with
  `scripts/verify_moon_albedo_registration.py`. **Both scripts need `numpy` and
  `astropy`** (`python3 -m pip install -r scripts/requirements.txt`); neither is
  a build or runtime dependency of the engine.

- **PTM-35 — per-face generator latitude.** `albedo_map_generator` previously
  sampled each face at −latitude, flipping the baked `Moon_Albedo_res*.bin`
  hemispheres relative to the FITS-direct runtime path. It now samples +latitude
  (see Processing). Moon bins regenerated from the corrected FITS match the
  photo-validated `Moon_Albedo_res9_camera.bin` to RMS ≈ 0.005 (the −latitude
  bins differed by ≈ 0.031).

Regenerate the per-face bins from a corrected FITS with the fixed generator:

```bash
build/bin/albedo_map_generator data/albedo_maps/Moon-BondAlbedo.fits \
    data/albedo_maps 3 9
```

**Europa is unaffected by PTM-35.** Its runtime path samples the FITS live at +latitude
and its shipped `Europa_Albedo_res*.bin` were already +latitude. (Regenerating them with
the PTM-35 generator matched them everywhere except about 365 seam faces at res6, which
an older seam treatment had produced. PTM-695 changes the Europa bins for a different
reason, the gap fill; see above.)

The distributed `albedo_maps.tar.gz` in this revision carries the corrected Moon
FITS + regenerated Moon bins (Europa untouched); `data_manifest.json`'s sha256 and
size are updated to match. If you have a local copy fetched before this revision,
re-fetch. See `docs/KNOWN_LIMITATIONS.md` (PTM-34 / PTM-35).

### Load-time registration guard

Both defects are invisible to every scalar summary: a rolled or flipped map keeps
its albedo VALUE distribution, so the map's range, its mean, and the reported
effective albedo `A_eff` all look completely normal while the albedo sits on the
wrong terrain. (Measured on `moon_global_res7`: the 180-degree roll moved the
global mean `A_eff` by 0.0001 and the global mean `T_max` by 0.002 K, but moved
individual faces by up to 15 K.) A stale local copy therefore produces plausible,
wrong results indefinitely.

The engine checks registration spatially at **every** site that loads a per-face
albedo map — the CPU absorbed-flux path, the GPU persistent thermal path, and the
radiation-coupling load that feeds the two-band radiosity source albedo — and
reports:

```
[albedo] Lunar map registration check: maria 0.0807 < highlands 0.1597 (contrast 0.0790, ...) -- OK
```

If the ordering is inverted it prints a loud warning naming the file and
**continues** — it never silently corrects the map, because the fix belongs in the
data and a blanket correction here would break Europa, whose FITS registers
correctly through the same loader. The check only applies to bodies with a
published dark/bright signature (currently the Moon) and only when the mesh
actually resolves the reference sites, so regional DEM meshes and non-lunar bodies
are declined rather than flagged. The two statistics it reports cannot separate a
longitude roll from a latitude flip; for that, run
`scripts/verify_moon_albedo_registration.py`, which scores the map against the
source FITS under all four candidate transforms and names the axis. Regression
gate: `build/bin/test_albedo_registration_check`.

Each of the three sites checks independently rather than trusting that an earlier
one ran: a config can reach the radiation-coupling load on a path the
absorbed-flux check never sees. The per-location depth-profile export — often the
only product a study actually reads — is covered without a fourth copy, because it
no longer resolves an albedo of its own: it composes absorbed flux through the
shared `AbsorbedFluxComposer`, and the check runs inside `build()`.

### Checking a local copy for staleness

The engine's guard fires only once a run reaches a map. To audit a checkout up
front:

```bash
python3 scripts/fetch_data.py --dataset albedo-maps --verify   # exit 1 if stale
python3 scripts/verify_moon_albedo_registration.py \
    --bins-dir data/albedo_maps --topo-dir data/topo/moon --geographic
```

`--verify` re-hashes the downloaded archive **and every file it was unpacked to**.
That second half matters: the `.unpacked` marker records only that an unpack once
happened, so before this check a tree whose marker matched the manifest could
still hold the old 180°-rolled bins and pass verification cleanly. That is
precisely how PTM-34 survived a corrected archive — the archive was right, the
loader was right, and nothing compared them to the bytes on disk.

The `--geographic` mode of the verify script needs no FITS and is the cheapest
independent detector: the nearside is mare-dominated and must read **darker** than
the farside. Correct maps give nearside ≈ 0.1165 vs farside ≈ 0.1511 (res6);
180°-rolled maps give exactly the inverse.

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
