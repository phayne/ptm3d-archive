# Lunar global topography grids (`topo/moon/`)

Per-face elevation grids for the global Moon icosphere meshes, used when
`mesh.use_topographic: true` (files `topo_moon_ico3.bin` … `topo_moon_ico9.bin`;
one binary per icosphere subdivision level).

**Not in git.** Fetch with:

```bash
python3 scripts/fetch_data.py --dataset moon-topo          # ico3-ico9, ~613 MB
```

(`ico10`/`ico11` grids exist for very-high-resolution global runs — 1.8/7.4 GB —
and are distributed separately as an optional tier.)

## Source

- **Product:** LRO LOLA global lunar DEM (LDEM), resampled onto the PTM3D
  icosphere face centers.
- **Citation:** Smith, D. E., et al. (2010). The Lunar Orbiter Laser Altimeter
  investigation on the Lunar Reconnaissance Orbiter mission. *Space Sci. Rev.*,
  150, 209–241. https://doi.org/10.1007/s11214-009-9512-y
- **Original data:** NASA PDS Geosciences Node (public domain).

## Processing

LDEM elevations sampled at each icosphere face center; stored as one
single-precision value per face in mesh face order (same 4-byte-header binary
layout as the output statistics files — see
`docs/tutorials/03_outputs.md`).

## License

Derived from U.S. government (NASA PDS) public-domain data; the derived grids
are distributed under CC-BY-4.0. Please cite Smith et al. (2010) when using
lunar topography results.
