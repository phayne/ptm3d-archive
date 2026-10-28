# (623) Chimaera — what is in this directory, and what runs use

## `shape.obj` — the DAMIT original, tracked since 2026-08-31

DAMIT model **9110** for asteroid (623) Chimaera, converted by this project's own
converter (`src/python/damit/shape_converter.py`) and scaled to a mean vertex
radius of **22.045 km**, which is `generic_body.shape.radius_km` in
`body_config.yaml`. **574 vertices, 1144 faces, kilometers.**

| | |
|---|---|
| archive | DAMIT, Astronomical Institute, Charles University, Prague |
| model page | `https://astro.troja.mff.cuni.cz/projects/damit/asteroids/view/9110` |
| archive reference | Ďurech, J., Sidorin, V., Kaasalainen, M. (2010), *DAMIT: a database of asteroid models*, A&A **513**, A46, doi:10.1051/0004-6361/200912693 |
| size | 37,163 bytes |
| md5 | `f3fbeb01a9ebbe4f9f71535c46848aa8` |
| sha256 | `1e3789b94f627e6cf049b380094afee3a4f4a5eef7080b9c3524e6685d44eea1` |
| created | 2026-06-30 |
| machine-readable | `shape_provenance.json` |

**Re-downloaded and checked on 2026-08-31**: fetching DAMIT 9110 again at the
same `radius_km = 22.045` produced a **byte-identical** file, same md5. This is
the archive model as the converter writes it, unmodified — no reconstruction
step ever stood between DAMIT and this file (unlike Justitia's).

DAMIT lists exactly one model for this asteroid, so there was no spin solution to
choose. The pole travels with it: ecliptic (λ, β) = (139°, 5°) → equatorial
RA/Dec = (143.077°, 19.873°), period 14.6241 h.

## `shape_refined.obj` — the mesh the body config actually loads

Loop subdivision of `shape.obj`, **2 passes** (1144 → 4576 → 18304 faces), then
renormalized to the same mean vertex radius. **9,154 vertices, 18,304 faces,
kilometers**, mean vertex radius measured at 22.044999999956 km.

| | |
|---|---|
| derived from | `shape.obj`, md5 `f3fbeb01a9ebbe4f9f71535c46848aa8` |
| size | 650,498 bytes |
| md5 | `a17fe60812a836bf63a519794eed4e36` |
| sha256 | `f8d719b51c10b7168c8972d12b3cd3d481c879c2dc186f2302b8de2ea9af62f3` |
| created | 2026-07-01 |
| machine-readable | `shape_refined_provenance.json` |

`body_config.yaml` names **this** file, not the 1144-face parent
(`generic_body.shape.mesh_file`). The published EMA Chimaera products are on the
refined mesh.

**Its generator is not in this repository.** No Loop-subdivision tool exists
under `scripts/`, `src/python/` or `gui/` (grepped 2026-08-31), so this file
cannot be regenerated from tracked source. That is precisely why it is tracked
rather than treated as a rebuildable intermediate. Anyone reproducing it has to
write the subdivider first and should not expect bit-for-bit agreement.

**Not covered.** Face count, mean radius and the parent's identity were checked
on 2026-08-31. Mesh topology — boundary edges, non-manifold edges, winding — was
not re-measured, and no thermal comparison between the parent and the refined
mesh was run.

## `ephemeris.spk` and `metakernel.tm`

The SPK is a HORIZONS `SMB_SPK_FILE` product for body **20000623**. `metakernel.tm`
is hand-written with repository-root-relative `PATH_VALUES` so a fresh clone
resolves it; its paths are resolved by `furnsh_c` against the **current working
directory**, so runs must be launched from the repository root. **Do not
regenerate it** — `SPKGenerator.create_metakernel()` emits absolute paths by
design, and baking those in makes the file resolve on one machine only.
