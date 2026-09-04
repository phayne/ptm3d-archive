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

The datasets are archived on Zenodo as *Supporting data for Planetary
Thermophysical Model 3D (PTM3D)* — concept DOI
[10.5281/zenodo.21693403](https://doi.org/10.5281/zenodo.21693403)
(v1: [10.5281/zenodo.21693404](https://doi.org/10.5281/zenodo.21693404)),
license CC-BY-4.0. Please cite the data DOI (and the per-dataset source
citations in each subdirectory README) alongside the software.

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
| `horizons_cache/` | small-body ephemerides from JPL Horizons | self-populates on first use; the 3 committed bodies are provenance only (see its README) |
| `locations/`, `diviner_seasonal/` | small analysis inputs | shipped |
| `external/` | placeholder paths used by internal study configs | not distributed (see its README) |

**Data location override:** set `PTM_DATA_ROOT=/path/to/data` to keep the
datasets outside the repository; every relative `data/...` reference in configs
resolves against it (see `docs/ENVIRONMENT.md`). Each dataset directory has its
own README with source, citation, processing, and license details.
