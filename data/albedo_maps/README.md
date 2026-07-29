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

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
