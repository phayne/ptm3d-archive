# SPICE kernels

PTM3D loads SPICE data from this directory (compiled in as `SPICE_KERNELS_DIR`,
a CMake cache variable defaulting to `<repo>/data/kernels`).

The repo ships only the small **metakernel templates** (`moon.tm`, `europa.tm`,
`mercury.tm`, `bennu.tm`). The binary kernels they reference (~44 MB for the
Moon set) are **not** in git — fetch them with:

```bash
python3 scripts/fetch_kernels.py                  # default (auto): ~1.23 GB
python3 scripts/fetch_kernels.py --bodies moon    # Moon + Mercury core, ~44 MB
python3 scripts/fetch_kernels.py --bodies europa  # core + jup365.bsp (1.1 GB)
python3 scripts/fetch_kernels.py --bodies bennu   # core + Bennu SPK (80.7 MB)
```

The bare command defaults to `--bodies auto`, which also fetches whatever the
metakernels in this directory require. `europa.tm` and `bennu.tm` both ship
here, so the default pulls `jup365.bsp` (1.1 GB) and `sb-101955-118_long.bsp`
(80.7 MB) as well — ~1.23 GB total. Pass `--bodies moon` if you only need the
lunar/Mercury set.

Expected layout after fetching:

```
data/kernels/
├── moon.tm  europa.tm  mercury.tm  bennu.tm   # metakernels (shipped)
├── lsk/naif0012.tls                 # leapseconds (required, all modes)
├── pck/pck00011.tpc                 # body orientation/shape constants
├── pck/moon_pa_de440_200625.bpc     # lunar principal-axes orientation
├── fk/moon_de440_250416.tf          # MOON_PA / MOON_ME frame definitions
└── spk/de440s.bsp                   # planetary ephemeris
    (+ spk/jup365.bsp for Europa, spk/sb-101955-118_long.bsp for Bennu)
```

All of these come from NASA NAIF except Bennu's, which comes from JPL Solar
System Dynamics — see `bennu.tm`'s own header for the URL, the coverage
(1900-01-01 to 2135-10-01), the solution it carries, and why NAIF's copy of the
same solution is not the one used.

**Run from the repo root.** The metakernels use `PATH_VALUES = ('data/kernels')`,
which SPICE resolves against the process working directory. If you must run
from elsewhere, edit `PATH_VALUES` to the absolute path of this directory.

See `INSTALL.md`, section "SPICE kernel data".
