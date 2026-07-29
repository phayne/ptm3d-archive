# Diviner-validation crater meshes (`dem/validation/`)

Terrain meshes, occluder rings, and precomputed view-factor files for the
Diviner cross-validation sites (Haworth, Faustini, Shackleton, and a sunlit
control site), plus the LOLA 20 m polar-stereographic source rasters. These are
referenced by the `divval_*` configs in `examples/campaigns/`.

**Not in git.** Fetch with:

```bash
python3 scripts/fetch_data.py --dataset validation-meshes   # ~740 MB
```

## Source

- **Terrain:** LRO LOLA 20 m/px south-polar stereographic DEM (Smith et al.
  2010; NASA PDS, public domain).
- **Validation reference:** Diviner Lunar Radiometer bolometric temperatures
  (Paige et al. 2010; Williams et al. 2017).

## Processing

DEM crops triangulated to radiation-core meshes (`*_radcore_*.ply`) with coarse
occluder rings (`*_ring_*.ply`) for far-field shadowing; view factors (`*.vf`)
precomputed with the in-repo H-matrix builder. **Keep each mesh + view-factor
pair together** — view factors are only valid for the exact mesh they were
built from (see `docs/KNOWN_LIMITATIONS.md`).

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
