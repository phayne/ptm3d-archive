# External-data placeholders (`external/`)

Some **study configurations** (in `configs/`)
reference datasets that PTM3D does not distribute — third-party DEMs (e.g. the
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

## Europa SOCET DEMs (`sutton_DEMs/`)

The five Europa terrains the `configs/europa_*` studies use — Agenor Linea,
Cilix crater, Pwyll crater, the pits and chaos near Rhadamanthys Linea, and the
ridged terrain near Yelland Linea — are SOCET SET stereo digital terrain models
built from Galileo Solid State Imager images. They were produced at the U.S.
Geological Survey Astrogeology Science Center under NASA PDART award
NNH16AC37I, and are public under CC BY 4.0. Cite the dataset:

> Bland, Michael (2022). *Europa DEMs and Orthoimages in Support of Bland et
> al. 2021 (Remote Sensing)*. figshare. Dataset.
> <https://doi.org/10.6084/m9.figshare.16904632>

and the paper it supports:

> Bland, M. T., Kirk, R. L., Galuszka, D. M., Mayer, D. P., Beyer, R. A., and
> Fergason, R. L. (2021). "How Well Do We Know Europa's Topography? An
> Evaluation of the Variability in Digital Terrain Models of Europa."
> *Remote Sensing* **13**(24), 5097. <https://doi.org/10.3390/rs13245097>

The deposit holds both SOCET SET and Ames Stereo Pipeline versions. PTM3D uses
the SOCET SET ones: the five `.ply` meshes the configurations name are exactly
the five `*_DEM_*_SOCET.tif` products of the deposit, stem for stem —
`Agenor_DEM_750m_SOCET`, `Cilix_DEM_350m_SOCET`, `Pwyll_DEM_750m_SOCET`,
`Rhadamanthys_DEM_450m_SOCET`, `Ridges_DEM_60m_SOCET`. Download those and
convert them with `dem_mesh_generator`.

**Read the deposit's own usage note before you publish on these.** It states
that the products "should only be used to assess and or verify the conclusions
described in Bland et al. 2021 ... as the DTMs themselves have been superseded
by updated versions that are registered to Europa's cartographic coordinate
system." A PTM3D thermal run is not that stated use. What PTM3D consumes is the
relief, which is the part the evaluation was about; the absolute cartographic
placement is the part that has been superseded. So treat a result from these
meshes as a study of a terrain of that character, not of a mapped location, and
prefer the registered versions if you need the placement to be current.

The directory name `sutton_DEMs/` is historical and is **not** an attribution —
it records the colleague who shared the files, not who made them. It is kept
only because 45 shipped configurations reference that path.
