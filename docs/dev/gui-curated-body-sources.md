# Curated body sources: bodies the live services correctly refuse

> There is a second tier of this registry, one level down: curated *sites* on
> a body, whose terrain mesh, occluder ring and view-factor matrix are only
> meaningful together, plus the search that finds a view-factor matrix already
> sitting beside a mesh. See
> [`gui-curated-lunar-sites.md`](gui-curated-lunar-sites.md).

## Motivation

The Asteroid card fetches an ephemeris from JPL HORIZONS and a shape model from
DAMIT. For most small bodies that works. For (101955) Bennu, both refuse, and
both are right:

```
HORIZONS was reached and REFUSED to build a kernel for 101955 Bennu (1999 RQ36).
... SPK creation is not available for pre-computed objects in the major body index
```

Bennu's HORIZONS record is `*SPECIAL*`, served from a pre-computed `ORX_merged`
kernel. No designation form changes that; we tested four
of them and the POST file endpoint on 2026-08-27. DAMIT has no Bennu either, because
DAMIT publishes lightcurve inversions and Bennu's shape is a radar model
(Nolan et al. 2013) later refined by OLA.

The result is two correct refusals and a user with nowhere to go, while an 80.7 MB kernel
and an 89 kB shape model for exactly that body already ship with, or are
fetchable by, this repository.

## The resolution order

```
curated registry (itself cache-first)  ->  body cache  ->  live service
```

For a body in the registry neither of the last two is reached. That is the
purpose of the tier, not an optimization: HORIZONS' refusal for Bennu is correct and
permanent, so asking is a slower route to a message the user can do nothing
with.

The registry comes before the body cache, and that ordering is
essential. Bennu's
`data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)/ephemeris.spk` is JPL
solution 76, covering 2015–2023, superseded on 2026-08-27 and kept only for
provenance. Inside that span it would pass an ordinary coverage check, so a
plain cache-first rule would wire it silently, and a 2026 run would then
fall off the end of its coverage into the engine's unlogged Keplerian fallback,
measured at 0.092 AU and up to 23.9 K wrong.

The tier's own first step is still a cache check (is the pinned kernel
already fetched into `data/kernels`?), and if it is, the whole flow completes
with no network call of any kind, not even a HEAD request. This matters
because `scripts/fetch_kernels.py`'s `download()` asks the server for a
`Content-Length` *before* it looks at the local file, so "call the downloader to
find out whether the file is here" would put a round trip in front of a body
that needs none.

"Already fetched" means every kernel the metakernel names, not just the
body's own SPK. `bennu.tm` loads four (leap seconds, planetary constants,
`de440s.bsp` and `sb-101955-118_long.bsp`), and only the last is in
`BODY_KERNELS["bennu"]`; the rest are `KERNELS_CORE`. `furnsh` stops at the
first member it cannot open, so a status built from the body's row alone
reported *"bennu.tm and its 1 pinned kernel are on this disk"* on a tree where
three of the four were missing. `curated_kernel_files` therefore starts from
the metakernel's own `KERNELS_TO_LOAD` and looks each member up in the pinned
table by filename; a missing member with no pinned source is reported, not
offered as a download.

## What the user sees

With the kernel present (the ordinary case after
`python3 scripts/fetch_kernels.py`):

* The Ephemeris section carries a provenance line before anything is pressed,
  *"Ephemeris source: JPL solution 118, OSIRIS-REx reconstruction, 1900-2135.
  Farnocchia et al. (2021), doi:10.1016/j.icarus.2021.114594. bennu.tm →
  already on this disk — no download and no network needed"*, and says that
  HORIZONS is not asked and DAMIT is not searched.
* The ephemeris button reads **Use JPL solution 118**, not *Fetch kernel from
  HORIZONS*.
* The shape button reads **Mission shape model (radar)…**, not *DAMIT shape
  model…*.
* Pressing either wires the card and reports the coverage read back out of the
  kernel, not taken from the table.

With the kernel absent, the button produces an offer, not a download:
the size (80.7 MB), the source, and the equivalent command
(`python3 scripts/fetch_kernels.py --bodies bennu`). Nothing is transferred
until the user confirms. The transfer itself is `fetch_kernels.download()`
(its resume, its bounded retry and its pinned-sha256 check), reused rather than
reimplemented.

## Where the pin lives

In exactly one place: `BODY_KERNELS` in `scripts/fetch_kernels.py`, which
holds `(subdir, filename, url, sha256)` per `--bodies` selector. The GUI
registry *imports that table by path* (`curated_sources.fetch_kernels_module`)
and never restates a URL or a hash.

`src/python/tests/test_gui_curated_sources.py::
test_the_pinned_hash_is_not_copied_into_the_gui` greps every `gui/**/*.py` for
each pinned URL and hash and fails if one appears. A pin with two copies is a
pin that goes stale in one of them, silently, which is the failure a pin
exists to prevent.

## Adding a body

Do not add a body on the strength of "a mission went there". Each field of
an entry must resolve to a file or a citation that is actually in this
repository; the test suite checks that, including against the shipped example
configuration for the same body.

1. Pin the kernel. Add its `(subdir, filename, url, sha256)` to a new list
   in `scripts/fetch_kernels.py` and register that list in `BODY_KERNELS`
   under a new `--bodies` selector. Add the selector to the `--bodies`
   `choices` and to `select_kernels`. Record the sha256 you measured, with the
   date and the URL you measured it from.
2. Ship the metakernel: `data/kernels/<body>.tm`, in the same shape as
   `bennu.tm`: `PATH_VALUES = ( 'data/kernels' )`, a `$KERNELS/...` entry per
   member, and a prose header giving the solution, its coverage, its citation,
   and what it supersedes. Paths longer than about 99 characters need SPICE's
   `+` continuation.
3. Ship the shape, with a `PROVENANCE.md` beside it naming the archive
   (a PDS LID, a DAMIT model id), the vertex and facet counts, the units, and
   what it supersedes. `.gitignore` line 14 is `*.obj` (a rule written for C
   object files that matches every asteroid shape model here), so a
   shape that is meant to ship must be added with `git add -f`, and the
   registry's `tracked` field must say so. The test compares that field
   against `git ls-files`.
4. Add the `CuratedBody` to `CURATED_BODIES` in
   `gui/gui_core/curated_sources.py`:

   | field | what it is |
   |---|---|
   | `key`, `name`, `number` | the body's identity; `name` must match the shipped example's `generic_body.name` |
   | `aliases` | every form a user types. Matching is case- and punctuation-folded, so `Bennu`, `101955`, `1999 RQ36` and `(101955) Bennu` all resolve |
   | `ephemeris.fetch_bodies` | the `--bodies` selector from step 1: the key into `BODY_KERNELS` and the command shown to the user |
   | `ephemeris.metakernel` | the bare `.tm` name written into `generic_body.spice_metakernel` |
   | `ephemeris.naif_id` | the id inside the kernel, read back from it, not computed from the asteroid number. Small-body SPK ids exist in a `2000000+N` and a `20000000+N` convention and only the kernel's own works |
   | `ephemeris.coverage_start/_end` | the published window. The test reads the kernel and compares |
   | `ephemeris.description` / `short` | the note line, and the button label (a button is 180 px at its narrowest) |
   | `ephemeris.reference` | the citation |
   | `ephemeris.approx_bytes` | for disclosure only. The authority on the bytes is the pinned sha256 |
   | `shape.*` | repo-relative path, description, archive id, mean radius, mesh scale, vertex/facet counts, units, and `tracked` |
   | `why_curated` | why the live services cannot serve this body. The user reads this *instead of* the service's refusal; without it the tier is unexplained |
   | `notes` | anything on disk that this entry supersedes, and by how much |

5. Run the suite: `python3 src/python/tests/test_gui_curated_sources.py`
   checks every path, the `--bodies` selector, the coverage against the kernel
   itself, the `tracked` claim against git, parity with the shipped example
   configuration, and that no pinned URL or hash reached the GUI.

## What being curated does not mean

It does not mean being quiet. A curated body still:

* has its coverage read back out of the kernel and compared with the run
  window *including the equilibration reach-back*, and is refused if it does
  not span it, the same discipline as the HORIZONS path, for the same reason
  (the engine substitutes Keplerian elements per epoch and logs
  nothing);
* says which source answered, and that the live service was skipped;
* names anything on disk that it supersedes. Bennu's body cache holds
  `ephemeris.spk`, JPL solution 76, covering 2015-2023 only, kept for
  provenance. Inside 2015-2023 that file would pass an ordinary coverage
  check, so the curated entry takes precedence and the card reports the
  older file as `NOT USED` rather than leaving two ephemerides on disk with
  nothing said about which ran.

## Reading a metakernel's coverage

`bennu.tm` is a KPL metakernel, not a binary SPK. Handing it to `spkobj`
answers `SPICE(INVALIDARCHTYPE)`, which used to be shown to the user as the
reason their kernel was no good; the kernel was fine, and the reader was wrong
about what it had. `gui_core.body_cards.spk_coverage` now parses a metakernel,
resolves its members, and reports coverage for the NAIF id asked about; see
that module for why the union across a metakernel's bodies would report three
centuries of coverage for every asteroid (`de440s.bsp` spans 1849–2150).
