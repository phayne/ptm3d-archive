# Copernicus Crater DEM for PTM3D

Cropped lunar DEM around Copernicus crater (9.62 N, 339.92 E; ~93 km diameter, Copernican-era). Prepared for a colleague to review before running PTM3D thermal models.

## Files

| File | Description |
|---|---|
| `copernicus_sldem2015_512ppd.tif` | Working DEM. Float32, geographic (lon/lat deg), 3430 x 3379 px, ~28 MB compressed |
| `copernicus_sldem2015_512ppd_km.tif` | Intermediate (km units, before scaling) - safe to delete |
| `copernicus_sldem2015_512ppd_hillshade.png` | Grayscale hillshade for quick visual check |
| `copernicus_sldem2015_512ppd_color.png` | Color-coded elevation (deep purple to white) |
| `copernicus_sldem2015_512ppd_colorshade.png` | Color elevation modulated by hillshade (combined view) |
| `copernicus_color_ramp.txt` | GDAL color-relief ramp used for the color PNGs |

## Source

- **Product:** SLDEM2015 (Barker et al. 2016), 512 ppd (~59 m/pixel at the equator)
- **Tile:** `sldem2015_512_00n_30n_315_360_float.img` (covers 0-30 N, 315-360 E)
- **URL:** https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/
- **Retrieved:** 2026-05-26 (via GDAL /vsicurl/ ranged reads against the PDS host - no full tile download)
- **Vertical accuracy:** ~3-4 m RMS (per Barker et al.)
- **Coregistration:** absolute LOLA frame (GRGM900B gravity)

## Coordinate reference

- **CRS:** geographic, body sphere R = 1737400 m
- **PROJ:** `+proj=longlat +R=1737400 +no_defs`
- **Prime meridian:** 0 deg, positive east
- **Origin (upper-left pixel):** lon 336.570 E, lat 12.920 N
- **Pixel size:** 0.001953125 deg (= 1/512 deg); ~59.2 m at lat 9.62 N
- **Dimensions:** 3430 cols x 3379 rows
- **Center:** 339.920 E, 9.620 N (matches IAU Copernicus center)
- **NoData:** -3.4028235e+38 (none present in this ROI, full coverage)

## Elevation

- **Units:** meters above 1737.4 km mean sphere (project convention)
- **Source units were km in the IMG;** values multiplied by 1000 in step B below
- **Range in this ROI:**
  - min = -3715.9 m (Copernicus floor)
  - max =  +853.0 m (rim / surrounding mare highs)
  - mean = -1001.7 m (dominated by crater interior)

These values are physically consistent: the surrounding plains sit ~1 km below the lunar mean radius (Procellarum is depressed), the rim rises ~700-900 m above the plains, the floor lies ~2.5-3 km below the rim.

## ROI choice

200 km box (~2R from center; per user choice over the 300 km and 500 km alternatives). Bounds:

- lon: 336.57 to 343.27 E
- lat:   6.32 to  12.92 N

This contains the full crater rim and continuous (proximal) ejecta blanket. **Distal rays and the secondary crater field extend further** (visible Stadius secondary chains reach ~1.5-2 crater radii) - if the colleague needs those, regenerate with a 300-500 km box.

## Reproduction recipe

Conda env (one-time):
```bash
conda create -n gdal-moon -c conda-forge gdal curl -y
conda activate gdal-moon
```

Crop (remote read, no full-tile download):
```bash
gdalwarp \
  -t_srs '+proj=longlat +a=1737400 +b=1737400 +no_defs' \
  -te 336.57 6.32 343.27 12.92 \
  -tr 0.001953125 0.001953125 \
  -r near \
  -of GTiff -co TILED=YES -co COMPRESS=DEFLATE -co PREDICTOR=3 \
  --config GDAL_PAM_ENABLED NO \
  /vsicurl/https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_512_00n_30n_315_360_float.lbl \
  copernicus_sldem2015_512ppd_km.tif
```

Scale km -> m:
```bash
gdal_calc.py -A copernicus_sldem2015_512ppd_km.tif --A_band=1 \
  --calc="A*1000.0" \
  --outfile=copernicus_sldem2015_512ppd.tif \
  --type=Float32 --NoDataValue=-3.4028235e+38 \
  --co=TILED=YES --co=COMPRESS=DEFLATE --co=PREDICTOR=3 --overwrite
gdal_edit.py -a_srs '+proj=longlat +R=1737400 +no_defs' copernicus_sldem2015_512ppd.tif
```

Previews (hillshade z-scale needs `-s 30320` because horizontal is degrees and vertical is meters at lat 9.62 N):
```bash
gdaldem hillshade -z 2 -s 30320 -compute_edges -of PNG \
  copernicus_sldem2015_512ppd.tif copernicus_sldem2015_512ppd_hillshade.png

gdaldem color-relief -of PNG -alpha \
  copernicus_sldem2015_512ppd.tif copernicus_color_ramp.txt \
  copernicus_sldem2015_512ppd_color.png
```

## Considered and rejected for now

- **LROC NAC stereo DTMs (~5 m/px):** exist for individual features near Copernicus (`NAC_DTM_COPERNSEC01` NE secondaries, `NAC_DTM_COPRNEJCT` S melt flow, `NAC_DTM_COPERNGRAB` graben) but no single NAC product covers the full 93 km crater plus ejecta. Available via https://data.lroc.im-ldi.com/lroc/rdr_product_select if the colleague wants sub-50 m detail in specific spots later (would be VRT-mosaicked over this SLDEM2015 base via the existing `ptm/scripts/dem_vrt.py` pattern).
- **Kaguya TC DTMs (~10 m/px):** stereo strips already incorporated into SLDEM2015.
- **PGDA dedicated mid-latitude products:** none currently listed; PGDA's high-res lunar work is presently south-pole-focused.
- **LOLA GDR tile (the user's first instinct, ~120 m/px):** SLDEM2015 is the same LOLA data merged with Kaguya TC stereo for ~2x higher resolution and better detail in the topographic shape, so it strictly improves on the LOLA-only GDR for this latitude band.

## PTM3D ingest

PTM3D's `dem_mesh_generator` / `dtm_to_ply_converter` auto-detects geographic vs projected GeoTIFFs (`ptm/src/cpp/tools/dem_mesh_generator.cpp`) and converts to body-centered Cartesian using the mean lunar radius. Elevations in meters above the 1737.4 km sphere match the convention used by every other DEM in `ptm/data/dem/`.
