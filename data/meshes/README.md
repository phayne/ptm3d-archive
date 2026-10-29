# Meshes (`meshes/`)

- `lunar_crater_80N.ply`, `lunar_crater_85N.ply`: synthetic bowl-crater meshes (12,800 faces)
  generated with the in-repository `crater_mesh_generator`; shipped with the repository and used
  by the tutorials ([`docs/tutorials/02_crater_radiosity.md`](../../docs/tutorials/02_crater_radiosity.md)).
- `moon3d_cartesian_double.ply`: the Moon3D testbed crater scan, a 1,698,304-face Delaunay
  triangulation of an 849,283-point Moon3D instrument point cloud. The Moon3D paper used a
  509,337-face decimated version of it, which is not distributed. **Not in git** (32 MB); fetch
  with:

```bash
python3 scripts/fetch_data.py --dataset moon3d-mesh
```

## Source

Synthetic craters: spherical-bowl geometry (Ingersoll et al. 1992 configuration). Moon3D scan:
Moon3D project testbed measurements (Haddad et al. 2026).

## License

CC-BY-4.0 for the Moon3D scan (as archived on Zenodo); the synthetic meshes are part of the PTM3D
distribution (MIT).
