# External-data placeholders (`external/`)

Some **study configurations** (in `configs/`)
reference datasets that PTM3D does not distribute — collaborator DEMs (e.g. the
Europa SOCET stereo DEMs under `sutton_DEMs/`), large intermediate terrain
products (`dtm/`, `moon3d/`), and private analysis inputs (`research/`).

Those configs used to carry absolute paths from the development machine; they
now point under `data/external/...` instead. **Nothing is fetched into this
directory automatically.** If you need to run one of those configs, obtain the
dataset from its source (or contact the maintainers) and place it at the
referenced path — or simply edit the config to point at your own copy.

Everything needed by the runnable examples in `examples/` is covered by
`scripts/fetch_kernels.py` and `scripts/fetch_data.py --core` — no file in
`data/external/` is required for them.
