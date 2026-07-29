# PTM3D data directory

Small data files (locations, cached small-body ephemerides, tutorial crater
meshes, mission SPK files) ship with the repository. **Large science datasets
do not live in git** — they are distributed as versioned archives and fetched
with the bundled script:

```bash
python3 scripts/fetch_data.py --list          # show available datasets
python3 scripts/fetch_data.py --core          # everything needed by the shipped examples (~1.7 GB)
python3 scripts/fetch_data.py --dataset moon-topo
```

The catalog is `data/data_manifest.json`: for every dataset it records the
files, download URLs, sizes, and SHA-256 checksums (downloads are resumable
and verified, exactly like `scripts/fetch_kernels.py` for SPICE kernels).

| Subdirectory | Contents | How it gets here |
|---|---|---|
| `kernels/` | SPICE kernels | `scripts/fetch_kernels.py` |
| `topo/moon/` | global lunar topography grids (ico3–ico9) | `fetch_data.py --dataset moon-topo` |
| `property_maps/moon/` | Diviner-derived thermophysical property maps | `fetch_data.py --dataset moon-property-maps` |
| `albedo_maps/` | Moon + Europa albedo maps | `fetch_data.py --dataset albedo-maps` |
| `dem/validation/` | Diviner-validation crater meshes + view factors | `fetch_data.py --dataset validation-meshes` |
| `dem/copernicus/` | Copernicus crater DEM crop (SLDEM2015) | `fetch_data.py --dataset copernicus-dem` |
| `meshes/` | tutorial crater meshes (shipped) + the Moon3D global mesh (fetched) | shipped / `fetch_data.py --dataset moon3d-mesh` |
| `spice/` | mission trajectory SPKs (Artemis-2, Clipper EGA) | shipped (small) |
| `horizons_cache/` | small-body ephemerides from JPL Horizons | shipped; self-populates |
| `locations/`, `diviner_seasonal/`, `a3gt_sites.geojson` | small analysis inputs | shipped |
| `external/` | placeholder paths used by internal study configs | not distributed (see its README) |

**Data location override:** set `PTM_DATA_ROOT=/path/to/data` to keep the
datasets outside the repository; every relative `data/...` reference in configs
resolves against it (see `docs/ENVIRONMENT.md`). Each dataset directory has its
own README with source, citation, processing, and license details.
