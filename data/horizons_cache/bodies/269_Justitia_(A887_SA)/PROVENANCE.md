# (269) Justitia — what is in this directory, and what runs use

## `shape.obj` — the DAMIT original, tracked since 2026-08-31

DAMIT **model 4966** (Marciniak et al. 2025) for asteroid (269) Justitia
(DAMIT **asteroid** id 2858 -- a different number space; DAMIT *model* 2858
is an unrelated body), converted to Wavefront OBJ by
this project's own converter (`src/python/damit/shape_converter.py`) and scaled
to a mean vertex radius of **25.364 km**, which is `generic_body.shape.radius_km`
in `body_config.yaml`. **574 vertices, 1144 faces, kilometers.**

| | |
|---|---|
| archive | DAMIT, Astronomical Institute, Charles University, Prague |
| model page | `https://damit.cuni.cz/projects/damit/asteroid_models/view/4966` |
| asteroid page | `https://damit.cuni.cz/projects/damit/asteroids/view/2858` (DAMIT asteroid id 2858) |
| model files | `.../generated_files/open/AsteroidModel/4966/{shape.obj,spin.txt}` |
| archive reference | Ďurech, J., Sidorin, V., Kaasalainen, M. (2010), *DAMIT: a database of asteroid models*, A&A **513**, A46, doi:10.1051/0004-6361/200912693 |
| size | 37,311 bytes |
| md5 | `4e7c109b8a2e17bc3c84bbd8a123a365` |
| sha256 | `d32c75e1562333f83a64492e6067ff1baa6551df660575ce446299f45f599a0f` |
| retrieved | 2026-08-31 |
| machine-readable | `shape_provenance.json` |

A DAMIT model is **dimensionless** — lightcurve inversion fixes the shape, not
the size — so the converter scales it on the way in. Change `radius_km` and the
model has to be re-fetched, or the mesh scale set by hand.

**The pole travels with the shape.** DAMIT publishes it as ecliptic
(λ, β) = (73°, −81°), which converts to equatorial RA/Dec = (85.069°, −57.852°),
and the rotation period is 33.1296 h. Those are the values `body_config.yaml`
and `justitia_spin.tpc` carry (the `.tpc` states the period as a prime-meridian
rate, 260.8065684617242 °/day = 33.1280 h, which is the JPL HORIZONS/LCDB
synodic period, NOT DAMIT's sidereal 33.1296 h: they differ by 5.76 s, not by
rounding. A run that needs DAMIT's own rotational phase must use 33.1296 h). A shape
model is defined in the body-fixed frame its own pole and period establish, so
this mesh with a different pole is Justitia pointing the wrong way, and nothing
downstream re-checks it.

**DAMIT has exactly one model for this asteroid**, so there was no spin solution
to choose between: `search_asteroid("269")` returns a single exact match,
`damit_id` 2858, and that is the id `data/horizons_cache/index.json` already
recorded. That field is the DAMIT **asteroid** id; the one model under it is
**model 4966** (spin.txt: `73 -81 33.1296` / `2445918 0`, i.e. JD0 = 2445918,
phi0 = 0). Verified 2026-09-23: DAMIT's `AsteroidModel/4966/shape.obj` has the
same 1144 face index lists as `shape.obj`, and every vertex equals it after one
uniform scale factor of 0.7936780 (max residual 5e-9 km).

## `shape_reconstructed_20260704.obj` — superseded, not on any engine path

The 2026-07-04 hand reconstruction described in **PTM-303**: rebuilt from
`mesh.ply` vertex/face data preserved in prior thermal-sim output, because the
`shape.obj` the configurations referenced was missing from disk. Its own header
was, until 2026-08-31, the only provenance record it had. 41,016 bytes,
md5 `9626bf7eefe8af89ab9ae5cad8bdc021`.

**It is faithful, and that is now measured rather than assumed.** Compared with
the DAMIT original re-downloaded on 2026-08-31:

| | reconstruction | DAMIT model 4966 |
|---|---|---|
| vertices | 574 | 574 |
| faces | 1144 | 1144 |
| face index lists | identical | identical |
| mean vertex radius | 25.364000 km | 25.364000 km |
| max in-order vertex deviation | **0.000000000 km** | — |
| md5 | `9626bf7e…` | `4e7c109b…` |

The two files differ **only** in header text and in decimal formatting — the
reconstruction writes ten decimal places with trailing zeros
(`-1.4482393400`), the converter writes eight significant figures
(`-1.44823934`). That is the whole of the md5 difference. Every Justitia result
produced from the reconstruction stands: the geometry it ran on and the geometry
in `shape.obj` today are the same numbers.

It is retained, and tracked, for the same reason Bennu's superseded radar shape
is: it is the file behind every Justitia product made between 2026-07-04 and
2026-08-31, and a superseded artifact that has been deleted cannot be
re-checked. **No configuration in this repository names it** — the fourteen
`examples/justitia_*.yaml` and `body_config.yaml` all name `shape.obj`. A
pre-tracking backup, under the original file name, is at
`~/projects/shape_backup_2026-08-31/shape.obj` (same md5).

Reproduce the comparison with
`~/projects/ada_justitia_verify_2026-08-31/compare_shapes.py`.

## `ephemeris.spk` and `metakernel.tm`

The committed SPK is a HORIZONS `SMB_SPK_FILE` product for body **20000269**
covering **2033-12-14T23:58:51 → 2036-05-31T23:58:51 UTC (899.0 days)**, which
spans every `examples/justitia_*.yaml` window including its equilibration
reach-back. See `../../README.md` for the coverage table and for why the eight-
digit NAIF id matters.

`metakernel.tm` is **hand-written with repository-root-relative `PATH_VALUES`**,
deliberately, so a fresh clone resolves it. `SPKGenerator.create_metakernel()`
emits absolute paths by design (the deep-OneDrive-path and SPICE 78-character
truncation workaround); regenerating this file bakes in one machine's layout and
is how the fourteen Justitia configurations came to name a path that does not
exist. **Do not regenerate it.** Its `PATH_VALUES` are resolved by `furnsh_c`
against the **current working directory**, so runs that use it must be launched
from the repository root.
