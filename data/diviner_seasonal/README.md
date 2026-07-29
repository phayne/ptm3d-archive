# Diviner seasonal polar temperature maps (`diviner_seasonal/`)

Season-binned Diviner bolometric temperature maps of the lunar polar regions,
stored as NumPy arrays (`*.npy`, shipped with the repository) and used by the
Diviner cross-validation analysis scripts. The much larger raw text exports
(~1.7 GB) are not in git and are distributed as an optional dataset tier.

## Source

- **Instrument:** Diviner Lunar Radiometer Experiment aboard LRO.
- **Citations:** Paige, D. A., et al. (2010). Diviner Lunar Radiometer
  observations of cold traps in the Moon's south polar region. *Science*, 330,
  479–482. https://doi.org/10.1126/science.1187726 ·
  Williams, J.-P., et al. (2017). The global surface temperatures of the Moon
  as measured by the Diviner Lunar Radiometer Experiment. *Icarus*, 283,
  300–325.
- **Original data:** NASA PDS (public domain).

## Processing

Diviner bolometric brightness temperatures binned by season and map-projected
to polar stereographic grids.

## License

Derived from NASA PDS public-domain data; distributed under CC-BY-4.0.
