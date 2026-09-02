# CDT (vendored)

This directory contains a vendored copy of [artem-ogre/CDT](https://github.com/artem-ogre/CDT),
a header-only C++ library for constrained Delaunay triangulation. PTM3D uses the
unconstrained 2D Delaunay path to triangulate the projected DTM point cloud in
`src/cpp/tools/dtm_to_ply_delaunay.cpp`.

## Provenance

- Upstream: <https://github.com/artem-ogre/CDT>
- Vendored release: **1.4.4** (tag `1.4.4`).
- Files present: the 11 headers of `CDT/include/`, **byte-identical to upstream
  1.4.4** (verified by per-file md5 against
  `raw.githubusercontent.com/artem-ogre/CDT/1.4.4/CDT/include/`), plus the
  upstream `LICENSE` (Mozilla Public License 2.0).
- Header-only: no `.cpp` is compiled. `CDT_INLINE_IF_HEADER_ONLY` resolves to
  `inline` because we never define `CDT_USE_AS_COMPILED_LIBRARY`.

Per-file md5 as vendored:

| File | md5 |
|---|---|
| `CDT.h` | `f8b901af0dd1700c8bff75219d81385e` |
| `CDT.hpp` | `8bc71b440130d4445d629385ff495ca7` |
| `CDTUtils.h` | `a4351934889679410502ef7886938f6a` |
| `CDTUtils.hpp` | `3277496a5cf8778717c7216d096ade9f` |
| `KDTree.h` | `e1991dd67ad371486654d746dac078b4` |
| `LocatorKDTree.h` | `d046f94d6c34a1403d8c4bf010043ae0` |
| `portable_nth_element.hpp` | `4c4f72135e4dcf61090cac2cab60947e` |
| `predicates.h` | `1f7302508e3b12157f4c93685f6629af` |
| `remove_at.hpp` | `7ccad5edd2b77954825d6f1e52129f1b` |
| `Triangulation.h` | `9eacfcabf8ee77374f26da170fcc1210` |
| `Triangulation.hpp` | `5ae519ed4a36b84151fdaf4275490691` |

## Why CDT

`dtm_to_ply_delaunay` triangulates a *variable-density* point cloud (the ROI
density falloff thins points radially). A grid-following triangulation leaves
disconnected patches at the density transitions; a true Delaunay triangulation
of the projected 2D coordinates does not. CDT is header-only, has no
dependencies beyond the standard library, and ships robust adaptive
`orient2d`/`incircle` predicates (Shewchuk), which matter for the near-degenerate
triangles that thinning produces.

## Licensing note

CDT is MPL-2.0, which is a **file-level** copyleft. The headers here are
unmodified; PTM3D links them into `dtm_to_ply_delaunay` without deriving from
them. If you ever need to patch a CDT header, send the patch upstream and
re-vendor rather than editing in place, so this directory stays a clean
unmodified copy.

## How to refresh

```sh
cd third_party/CDT
TAG=1.4.4   # or a newer upstream tag
for f in CDT.h CDT.hpp CDTUtils.h CDTUtils.hpp KDTree.h LocatorKDTree.h \
         portable_nth_element.hpp predicates.h remove_at.hpp \
         Triangulation.h Triangulation.hpp; do
  curl -L -o "include/$f" \
    "https://raw.githubusercontent.com/artem-ogre/CDT/${TAG}/CDT/include/$f"
done
curl -L -o LICENSE "https://raw.githubusercontent.com/artem-ogre/CDT/${TAG}/LICENSE"
```

Then update the tag and the md5 table above, rebuild `dtm_to_ply_delaunay`, and
re-run the DEM-tools smoke test
(`src/python/tests/test_dem_tools.py`) — the triangulation is what that test
measures, so a CDT upgrade that changes triangle counts will show up there.

Do not edit the headers directly.
