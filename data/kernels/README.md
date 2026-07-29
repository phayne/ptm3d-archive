# SPICE kernels

PTM3D loads SPICE data from this directory (compiled in as `SPICE_KERNELS_DIR`,
a CMake cache variable defaulting to `<repo>/data/kernels`).

The repo ships only the small **metakernel templates** (`moon.tm`, `europa.tm`,
`mercury.tm`). The binary kernels they reference (~45 MB for the Moon set) are
**not** in git — fetch them from NASA NAIF with:

```bash
python3 scripts/fetch_kernels.py                  # Moon + Mercury set (~45 MB)
python3 scripts/fetch_kernels.py --bodies europa  # also jup365.bsp for Europa
```

Expected layout after fetching:

```
data/kernels/
├── moon.tm  europa.tm  mercury.tm   # metakernels (shipped)
├── lsk/naif0012.tls                 # leapseconds (required, all modes)
├── pck/pck00011.tpc                 # body orientation/shape constants
├── pck/moon_pa_de440_200625.bpc     # lunar principal-axes orientation
├── fk/moon_de440_250416.tf          # MOON_PA / MOON_ME frame definitions
└── spk/de440s.bsp                   # planetary ephemeris (+ spk/jup365.bsp for Europa)
```

**Run from the repo root.** The metakernels use `PATH_VALUES = ('data/kernels')`,
which SPICE resolves against the process working directory. If you must run
from elsewhere, edit `PATH_VALUES` to the absolute path of this directory.

See `INSTALL.md`, section "SPICE kernel data".
