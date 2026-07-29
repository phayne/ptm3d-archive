# Mission trajectory kernels (`spice/`)

Small SPICE SPK (trajectory) files used by the camera/flyby examples. Shipped
with the repository (unlike the large generic kernels in `data/kernels/`, which
are fetched with `scripts/fetch_kernels.py`).

| File | Contents |
|---|---|
| `artemis2_moon.bsp`, `artemis2_moon_earth.bsp` | Artemis-2 trajectory relative to Moon/Earth |
| `artemis2_moon_vectors.txt` | the state vectors the SPKs were built from |
| `clipper_ega_moon.bsp` | Europa Clipper Earth-gravity-assist lunar flyby |

## Source and processing

Trajectories exported from publicly released mission design products and
converted to SPK with NAIF utilities (`mkspk`). For authoritative, current
mission kernels use the NAIF archives at
https://naif.jpl.nasa.gov/naif/data.html.

## License

Derived from publicly released NASA trajectory information; provided for
reproducing the shipped examples.
