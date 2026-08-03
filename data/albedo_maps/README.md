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

Source maps regridded to each icosphere's face centers; per-face binaries use
the 4-byte-header single-precision layout
(`docs/tutorials/06_interpreting_outputs.md`).

## Known issue: lunar map longitude registration (PTM-30)

The distributed `Moon-BondAlbedo.fits` is **mislabeled by 180° in longitude**: its
WCS header declares `CRVAL1=0` (column 0 = IAU_MOON lon 0, east, 0..360) but the
raster is actually a −180..180-centered map (IAU lon 0 at the center column). As
shipped, the dark nearside maria land on the bright farside highlands and vice-versa.
The topographic meshes (`data/topo/`) are correct; the Europa FITS is correct, so
this is specific to the lunar file — do **not** patch the shared FITS loader.

Re-register the local copy (heals the wrap seam, rolls the data by +180° so it
matches its header and IAU_MOON; original kept as `*.orig-mislabeled-180`):

```bash
python3 scripts/fix_moon_albedo_lon_registration.py
```

Re-running `fetch_data.py --dataset albedo-maps` restores the mislabeled file, so
re-run the fix afterward. Per-face camera `.bin` maps must be regenerated from the
corrected FITS before any Moon camera render. See `BUGS.md` PTM-30.

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
