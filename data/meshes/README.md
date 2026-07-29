# Meshes (`meshes/`)

- `lunar_crater_80N.ply`, `lunar_crater_85N.ply` — synthetic bowl-crater meshes
  (12,800 faces) generated with the in-repo `crater_mesh_generator`; shipped
  with the repository and used by the tutorials
  (`docs/tutorials/04_crater_radiosity.md`).
- `moon3d_cartesian_double.ply` — the 500k-face global Moon3D terrain mesh
  (LOLA-derived). **Not in git** (32 MB); fetch with:

```bash
python3 scripts/fetch_data.py --dataset moon3d-mesh
```

## Source

Synthetic craters: generated (spherical-bowl geometry, Ingersoll et al. 1992
configuration). Moon3D mesh: derived from the LRO LOLA global DEM (Smith et
al. 2010; NASA PDS, public domain).

## License

CC-BY-4.0 for the LOLA-derived mesh; the synthetic meshes are part of the
PTM3D distribution (MIT).
