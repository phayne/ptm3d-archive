# Lunar thermophysical property maps (`property_maps/moon/`)

Per-face maps used by the default lunar property model: the Diviner-derived
H-parameter (regolith density-scale depth) `hpar_moon_icoN.bin`, maria/highlands
masks `mask_moon_icoN.bin`, and thermal-inertia-at-273-K maps
`ti273k_moon_icoN.bin`, for icosphere levels 3–9.

**Not in git.** Fetch with:

```bash
python3 scripts/fetch_data.py --dataset moon-property-maps   # ~60 MB
```

## Source

- **Heritage:** H-parameter and thermal-inertia mapping from the Diviner Lunar
  Radiometer Experiment.
- **Citations:** Hayne, P. O., et al. (2017). Global regolith thermophysical
  properties of the Moon from the Diviner Lunar Radiometer Experiment.
  *J. Geophys. Res. Planets*, 122, 2371–2400.
  https://doi.org/10.1002/2017JE005387 ·
  Paige, D. A., et al. (2010). *Science*, 330, 479–482.
- **Original data:** NASA PDS (public domain).

## Processing

Diviner-derived global maps sampled at each icosphere face center; one
single-precision value per face in mesh face order (4-byte-header binary
layout, see `docs/tutorials/06_interpreting_outputs.md`).

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
Please cite Hayne et al. (2017) when these maps inform your results.
