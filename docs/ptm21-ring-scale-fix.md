# PTM-21 — occluder-ring 2× elevation-scale fix + corrected fig8_hybrid ring

2026-07-22 (Ada). Fix + rebuild for the PTM-21 defect: `build_lola_ring.py` silently
produced the fig8_hybrid ring `sunlit_ring_50_200.ply` with elevations **exactly 2×**
LOLA truth (over-tall horizon, over-shadows grazing sun). See `BUGS.md` PTM-21 and
`scripts/audit_ring_horizon.py`.

## Root cause
`build_lola_ring.py` read `band.ReadAsArray()` (raw DN) and treated it as meters,
never applying `GetScale()`/`GetOffset()`. LOLA polar PDS `.IMG` carry
`SCALING_FACTOR = 0.5 m/DN`, so raw DN = 2× physical elevation. Silent because the
projected Float32 `.tif` the tool was validated against has scale=1.

## Builder fix (`scripts/build_lola_ring.py`)
- Applies `scale = --elev-scale or band.GetScale()`, `offset = --elev-offset or GetOffset()`.
- **Fail-loud guard:** if scale is `None` **and** the raster is an integer type (the raw-PDS
  hallmark) and no `--elev-scale` given → abort with an explanatory error (never assume 1.0).
- Float raster + no scale metadata → scale 1.0 (the `.tif` path, unchanged).
- Byte-identical for scale=1 `.tif` sources — verified: a small `.tif` ring built with the
  original vs fixed script is md5-identical (`ba0f4a18…`), no scale banner printed.
- Guard paths unit-tested on synthetic Int16 rasters (abort / `--elev-scale` / auto-scale).
- The out-of-repo copy `~/a3gt_campaign/build_lola_ring.py` was replaced with this fixed
  version (md5 `a6368474…`) so the buggy copy can't be silently re-invoked.

## Corrected ring rebuild
The 20 m source `LDEM_80S_20M.IMG` is **OneDrive-dehydrated and will not hydrate**
(`/moon/lola/dem` share, persistent ETIMEDOUT — same failure Luna hit on the Collaborators
share), so GDAL cannot open it headlessly. Rebuilt instead from the accessible, properly-scaled
80 m LOLA base DEM `data/dem/a3gt_v2/_lola_base_spstereo.tif` (same polar-stereo variant A,
k0=1, R=1737400, Float32 scale=1; agrees with the 20 m truth TIF to 2.6 m):

```
build_lola_ring.py --lola data/dem/a3gt_v2/_lola_base_spstereo.tif \
  --site-lat -88.1 --site-lon 6.3 --r-core-km 48 --r-ext-km 200 --max-points 80000 \
  --out sunlit_ring_50_200.ply
```

## Ring artifacts (`~/projects/moon3d/meshes/fig8_hybrid/`)
| file | md5 | note |
|---|---|---|
| `sunlit_ring_50_200.ply` | `73d839dd…` | **CORRECTED production** (fixed reader, 80 m base, 61490 v) |
| `sunlit_ring_50_200_DEFECT2X.ply` | `e8177c17…` | defective 2× ring, kept until re-run validates, then deletable |
| `sunlit_ring_50_200_ALT_el2_controlled.ply` | `e76b851a…` | el/2 of the defect (identical geometry, only z÷2); the strict-controlled-swap alternative — provably == the fixed reader run on the 20 m IMG (sampling/triangulation are elevation-independent) |

## Verification (`scripts/verify_corrected_ring.py`, figure `figures_ring_audit/ring_corrected_verification.png`)
- Elevation fit vs LOLA truth: DEFECT slope **2.0000** → CORRECTED **1.0002** (base rebuild),
  **1.0000** (el/2). OLD reference rings 0.9995–0.9996.
- Range-matched horizon collapse (ring band, beyond 46 km): corrected vs OLD ring
  **mean|Δ| 0.080°**, vs TIF truth **0.081°** (residual = coarse-binning noise); the DEFECT
  sat **+1.20° median / +4.50° max** above truth.
- Base-80m rebuild vs el/2 agree to **mean|Δ| 0.062°** → the 80 m resampling confound is
  negligible for this coarse occluder.

## Not done (awaits Paul/Sagan)
- Production coupled+radoff re-run with the corrected ring — held for Luna.
- True 20 m-IMG rebuild — needs the `/moon/lola/dem` OneDrive share re-authed. The 80 m ring
  and the el/2 alternative both pass verification; either is production-ready.
- If a strict only-scale-changed controlled swap is wanted for the re-run, use
  `sunlit_ring_50_200_ALT_el2_controlled.ply` instead of the 80 m rebuild.
