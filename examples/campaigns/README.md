# Internal campaign configs — NOT runnable from a clone

The YAML files in this directory are **internal production-campaign
configurations** (A3GT Artemis-3 sites, Diviner validation, Haworth/Amundsen
craters, Moon3D latitude sweeps, asteroid campaigns, ...). They are kept for
provenance and as worked examples of advanced options, but they embed
**machine-specific absolute paths** (meshes, DEMs, view-factor matrices, and
output locations on the development workstation) and reference multi-GB data
products that do not ship with the repository.

**They will not run from a fresh clone.** For runnable examples use the
configs in [`examples/`](../) — start with
[`quickstart_moon_diurnal.yaml`](../quickstart_moon_diurnal.yaml) and the
tutorials in `docs/tutorials/`.
