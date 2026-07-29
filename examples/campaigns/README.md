# Internal campaign configs — NOT runnable from a clone

The YAML files in this directory are **internal production-campaign
configurations** (A3GT Artemis-3 sites, Diviner validation, Haworth/Amundsen
craters, Moon3D latitude sweeps, asteroid campaigns, ...). They are kept for
provenance and as worked examples of advanced options. Their paths are now
repository-relative, but they reference multi-GB data products that are either
distributed separately (`scripts/fetch_data.py` — see `data/README.md`) or not
distributed at all (`data/external/...` — see the README there).

**Most will not run from a fresh clone without fetching or supplying those
datasets.** For runnable examples use the
configs in [`examples/`](../) — start with
[`quickstart_moon_diurnal.yaml`](../quickstart_moon_diurnal.yaml) and the
tutorials in `docs/tutorials/`.
