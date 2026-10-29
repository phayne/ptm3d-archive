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
- **Europa:** Voyager/Galileo-derived Bond albedo maps.
- Original mission data: NASA PDS (public domain).

## Processing

Source maps are regridded to each icosphere's face centers by
`albedo_map_generator`, which samples the source FITS at each face's physical
**+latitude** (north positive) and IAU longitude — the same convention as every
runtime sampler (`ptm_thermal_sim.cpp`, `thermal_model.cpp`, `albedo_manager.cpp`,
the DEM sampler) and the photo-validated Moon camera bin. Each per-face binary is
an 8-byte little-endian face count followed by one float64 (Bond albedo) per face.

## Lunar map registration (longitude, latitude — corrected)

Two registration defects in the earlier lunar maps are corrected in this
revision:

- **Source FITS longitude (+180°).** The original `Moon-BondAlbedo.fits`
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

- **Per-face generator latitude.** `albedo_map_generator` previously
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

**Europa is unaffected.** Its runtime path samples the FITS live at +latitude and
its shipped `Europa_Albedo_res*.bin` were already +latitude (regenerating with the
fixed generator reproduces them); the Europa maps are unchanged.

The distributed `albedo_maps.tar.gz` in this revision carries the corrected Moon
FITS + regenerated Moon bins (Europa untouched); `data_manifest.json`'s sha256 and
size are updated to match. If you have a local copy fetched before this revision,
re-fetch. See `docs/KNOWN_LIMITATIONS.md`.

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
precisely how survived a corrected archive — the archive was right, the
loader was right, and nothing compared them to the bytes on disk.

The `--geographic` mode of the verify script needs no FITS and is the cheapest
independent detector: the nearside is mare-dominated and must read **darker** than
the farside. Correct maps give nearside ≈ 0.1165 vs farside ≈ 0.1511 (res6);
180°-rolled maps give exactly the inverse.

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
