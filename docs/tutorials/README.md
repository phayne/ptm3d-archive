# PTM3D tutorials

One track, in order. Run every command from the repository root after building
([INSTALL.md](../../INSTALL.md)) and fetching the Moon SPICE kernels
(`python3 scripts/fetch_kernels.py --bodies moon`). Everything here runs on the CPU; times are
for four cores of a workstation, so allow longer on a laptop.

| # | Tutorial | What it covers | Time | Needs |
|---|---|---|---|---|
| 1 | [Quickstart: a global Moon](01_quickstart.md) | One lunation on a 1,280-face icosphere: the run log, the spin-up check, the per-face statistics and the numbers to expect | 30 s | Moon kernels (44 MB) |
| 2 | [Crater radiation exchange](02_crater_radiosity.md) | Radiation on vs. off on the shipped 12,800-face polar crater; the fine-mesh illumination settings | 35 s + 110 s | shipped mesh |
| 3 | [The output files](03_outputs.md) | Every file in an output directory, the binary format, `time_steps.txt`, provenance | reading only | outputs of 1 and 2 |
| 4 | [The GUI](04_gui.md) | The same ground through the desktop application | a few minutes | `pip install -r gui/requirements.txt` |

## Jupyter notebooks

Executed notebooks under [`notebooks/`](notebooks/): each checks its prerequisites, runs the
engine and makes its figures, and the committed copies show the outputs. Start with 01; it
introduces the run reader and the binary format the others use.

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt   # numpy, matplotlib, pyyaml, jupyterlab, spiceypy
python3 -m pip install -e .                  # the ptm package the notebooks import
jupyter lab docs/tutorials/notebooks/
```

| # | Notebook | What it covers | Time (4 cores) | Needs |
|---|---|---|---|---|
| 01 | [`01_moon_diurnal`](notebooks/01_moon_diurnal.ipynb) | The quickstart run as maps and curves, the same run on LOLA topography, and how to change a thermophysical property | about 8 min | Moon kernels; `fetch_data.py --core` for section 9 |
| 02 | [`02_europa_thermal`](notebooks/02_europa_thermal.ipynb) | Europa: a Jupiter eclipse and its cooling, spin-up length, the observed albedo map | about 2 min | `fetch_kernels.py --bodies europa` (1.1 GB); `fetch_data.py --dataset albedo-maps` |
| 03 | [`03_asteroid_thermal`](notebooks/03_asteroid_thermal.ipynb) | A synthetic asteroid on a Keplerian orbit at two thermal inertias; which keys a generic body reads | about 1.5 min | the leapseconds kernel only |
| 04 | [`04_asteroid_spice`](notebooks/04_asteroid_spice.ipynb) | (101955) Bennu on its SPICE ephemeris with the OSIRIS-REx shape model, at perihelion and aphelion | about 30 min | `fetch_kernels.py --bodies bennu` (81 MB) |

Data is looked up under `$PTM_DATA_ROOT` if it is set, then under the repository's `data/`
directory (see [data/README.md](../../data/README.md)).

To record which binary produced a set of numbers:

```bash
python3 -c "from ptm import run; print(run.binary_provenance())"
```
