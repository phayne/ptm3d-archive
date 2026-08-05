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

**Europa is unaffected.** Its runtime path samples the FITS live at +latitude and
its shipped `Europa_Albedo_res*.bin` were already +latitude (regenerating with the
fixed generator reproduces them); the Europa maps are unchanged.

The distributed `albedo_maps.tar.gz` in this revision carries the corrected Moon
FITS + regenerated Moon bins (Europa untouched); `data_manifest.json`'s sha256 and
size are updated to match. If you have a local copy fetched before this revision,
re-fetch. See `docs/KNOWN_LIMITATIONS.md` (PTM-34 / PTM-35).

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
