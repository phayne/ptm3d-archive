// Standalone gate for FITS fill handling in SpatialMap (Europa Bond-albedo
// fill-sentinel bug).
//
// The Europa Bond-albedo FITS maps fill their gaps with -1.0 but declare
// MISSING = -9999 in the header. The loader only ever excluded the declared
// value, so bilinear sampling blended -1.0 into every face whose support
// touched a gap. Two further defects made it worse:
//   - the header lookup chained BLANK -> MISSING -> NODATA on one CFITSIO status,
//     so a file without BLANK never had MISSING or NODATA read at all, and
//   - the partial-support fallback summed the valid corner terms WITHOUT
//     dividing by their weight, pulling any sample next to a gap toward zero.
//
// Each case below fails on the code before the fix. Synthetic maps and FITS
// files written to the temp directory only, so it runs anywhere.
//
// Build: cmake --build build --target test_fits_fill_sentinel
// Run:   build/bin/test_fits_fill_sentinel

#include "spatial_map.h"

#include <fitsio.h>

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using ptm::MapFillPolicy;
using ptm::MapSampleInfo;
using ptm::SpatialMap;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) ++g_failures;
}

// The rasters are float, so compare against double literals at float precision.
bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// A 4x4 regional (non-cylindrical) raster, lon 10..13, lat 0..3, so the seam
// logic stays out of the way. Rows are stored top (lat_max) first, matching
// SpatialMap's internal layout.
std::vector<float> uniformRaster(float v) { return std::vector<float>(16, v); }

// Write a 2D float FITS image with optional header keywords. Row 0 of `data`
// is written first (FITS row 1).
std::string writeFits(const std::string& name, int w, int h, const std::vector<float>& data,
                      const std::vector<std::pair<std::string, double>>& keys) {
    const auto path = (std::filesystem::temp_directory_path() / name).string();
    std::filesystem::remove(path);
    fitsfile* f = nullptr;
    int status = 0;
    long naxes[2] = {w, h};
    fits_create_file(&f, path.c_str(), &status);
    fits_create_img(f, FLOAT_IMG, 2, naxes, &status);
    double crval1 = 0.0, crval2 = -90.0, cdelt1 = 360.0 / w, cdelt2 = 180.0 / h, crpix = 1.0;
    fits_update_key(f, TDOUBLE, "CRVAL1", &crval1, nullptr, &status);
    fits_update_key(f, TDOUBLE, "CRVAL2", &crval2, nullptr, &status);
    fits_update_key(f, TDOUBLE, "CDELT1", &cdelt1, nullptr, &status);
    fits_update_key(f, TDOUBLE, "CDELT2", &cdelt2, nullptr, &status);
    fits_update_key(f, TDOUBLE, "CRPIX1", &crpix, nullptr, &status);
    fits_update_key(f, TDOUBLE, "CRPIX2", &crpix, nullptr, &status);
    for (const auto& kv : keys) {
        double v = kv.second;
        fits_update_key(f, TDOUBLE, kv.first.c_str(), &v, nullptr, &status);
    }
    long fpixel[2] = {1, 1};
    fits_write_pix(f, TFLOAT, fpixel, static_cast<long>(data.size()),
                   const_cast<float*>(data.data()), &status);
    fits_close_file(f, &status);
    if (status) {
        std::printf("  cannot write test FITS %s (CFITSIO status %d)\n", path.c_str(), status);
        ++g_failures;
    }
    return path;
}

// A global 8x4 raster (45 x 45 deg pixels) of 0.6 with one gap pixel at column 3
// of FITS row 2, i.e. in the middle of the map, away from the seam columns.
std::vector<float> globalRasterWithGap(float gap) {
    std::vector<float> d(32, 0.6f);
    d[2 * 8 + 3] = gap;
    return d;
}

// Sample a point whose bilinear support has weight 0.25 on each of four pixels,
// one of which is the gap pixel: halfway between columns 3 and 4 (lon 157.5) and
// halfway between the two middle internal rows (lat 0).
constexpr double kLonMid = 157.5;
constexpr double kLatMid = 0.0;

}  // namespace

int main() {
    std::printf("test_fits_fill_sentinel\n");

    // 1. A -1.0 neighbor no longer contaminates a sample. Grid point (1,1) is
    //    -1; the sample at its corner cell has bilinear weight 0.25 on it.
    std::printf("1. undeclared -1.0 sentinel\n");
    {
        std::vector<float> d = uniformRaster(0.6f);
        d[1 * 4 + 1] = -1.0f;
        SpatialMap bare;
        bare.initializeFromData(d.data(), 4, 4, 10.0, 13.0, 0.0, 3.0);
        // Pixel (x=1,y=1) sits at lon 11, lat 2; (1.5, 1.5) in pixel space is lon 11.5, lat 1.5.
        const double contaminated = bare.getValue(11.5, 1.5);
        check(near(contaminated, 0.75 * 0.6 + 0.25 * -1.0, 1e-7),
              "control: with no fill policy the -1.0 pixel is blended in (" +
                  std::to_string(contaminated) + ")");

        SpatialMap fixed;
        fixed.initializeFromData(d.data(), 4, 4, 10.0, 13.0, 0.0, 3.0,
                                 ptm::MapProjection::EQUIRECTANGULAR, -9999.0,
                                 MapFillPolicy::bondAlbedo());
        MapSampleInfo info;
        const double v = fixed.getValue(11.5, 1.5, &info);
        check(near(v, 0.6, 1e-7), "Bond-albedo policy: sample next to -1.0 is exactly 0.6 (" +
                                std::to_string(v) + ")");
        check(info.fill_corners == 1 && near(info.fill_weight, 0.25, 1e-12),
              "sample info reports 1 fill corner of weight 0.25");
        check(fixed.getFillCounts().out_of_range == 1, "fill count: 1 pixel outside [0,1]");

        MapFillPolicy sentinel;
        sentinel.fill_values.push_back(-1.0);
        SpatialMap configured;
        configured.initializeFromData(d.data(), 4, 4, 10.0, 13.0, 0.0, 3.0,
                                      ptm::MapProjection::EQUIRECTANGULAR, -9999.0, sentinel);
        check(near(configured.getValue(11.5, 1.5), 0.6, 1e-7) &&
                  configured.getFillCounts().configured == 1,
              "configured --fill-value -1 sentinel: excluded and counted as configured");
    }

    // 2. Renormalization, not a partial sum. A declared (-9999) missing pixel next
    //    to a non-uniform field: the result must be the weighted mean of the
    //    valid corners. The old fallback returned the bare partial sum.
    std::printf("2. renormalized weights over the valid corners\n");
    {
        std::vector<float> d = uniformRaster(0.5f);
        d[1 * 4 + 1] = -9999.0f;  // weight 0.25 at the sample below
        d[1 * 4 + 2] = 0.2f;      // weight 0.25
        d[2 * 4 + 1] = 0.8f;      // weight 0.25
        d[2 * 4 + 2] = 0.4f;      // weight 0.25
        SpatialMap m;
        m.initializeFromData(d.data(), 4, 4, 10.0, 13.0, 0.0, 3.0);
        const double v = m.getValue(11.5, 1.5);
        const double expect = (0.2 + 0.8 + 0.4) / 3.0;
        check(near(v, expect, 1e-7), "value = mean of the 3 valid corners " + std::to_string(expect) +
                                         " (got " + std::to_string(v) + "; the old partial sum was " +
                                         std::to_string(0.25 * (0.2 + 0.8 + 0.4)) + ")");

        // Every weighted corner is fill: the sample is missing, not a number.
        std::vector<float> g = uniformRaster(-9999.0f);
        g[0] = 0.5f;  // a valid pixel outside the sample's support
        SpatialMap all_fill;
        all_fill.initializeFromData(g.data(), 4, 4, 10.0, 13.0, 0.0, 3.0);
        MapSampleInfo info;
        const double w = all_fill.getValue(11.5, 1.5, &info);
        check(w == all_fill.getMissingValue() && info.no_valid_support,
              "no valid corner in the support -> missing value, no_valid_support set");

        // A zero-weight fill corner does not change an on-pixel sample.
        std::vector<float> e = uniformRaster(0.5f);
        e[1 * 4 + 2] = -9999.0f;
        SpatialMap on_pixel;
        on_pixel.initializeFromData(e.data(), 4, 4, 10.0, 13.0, 0.0, 3.0);
        check(near(on_pixel.getValue(11.0, 2.0), 0.5, 1e-7),
              "sample exactly on a valid pixel ignores a zero-weight fill neighbor");
    }

    // 3. FITS header keywords are honored, each one independently.
    std::printf("3. FITS BLANK / MISSING / NODATA keywords\n");
    {
        const std::string blank =
            writeFits("ptm_test_fill_blank.fits", 8, 4, globalRasterWithGap(-1.0f), {{"BLANK", -1.0}});
        SpatialMap m;
        check(m.loadFromFITS(blank), "load FITS with BLANK = -1");
        check(m.getMissingValue() == -1.0, "missing value read from BLANK (" +
                                               std::to_string(m.getMissingValue()) + ")");
        check(m.getFillCounts().declared == 1, "1 pixel counted as declared fill");
        check(near(m.getValue(kLonMid, kLatMid), 0.6, 1e-7),
              "BLANK pixel excluded from the sample (" + std::to_string(m.getValue(kLonMid, kLatMid)) +
                  ")");

        // MISSING without BLANK: the old one-status chain never read it.
        const std::string missing = writeFits("ptm_test_fill_missing.fits", 8, 4,
                                              globalRasterWithGap(-5.0f), {{"MISSING", -5.0}});
        SpatialMap mm;
        check(mm.loadFromFITS(missing) && mm.getMissingValue() == -5.0,
              "MISSING honored when BLANK is absent (" + std::to_string(mm.getMissingValue()) + ")");
        check(near(mm.getValue(kLonMid, kLatMid), 0.6, 1e-7), "MISSING pixel excluded from the sample");

        const std::string nodata = writeFits("ptm_test_fill_nodata.fits", 8, 4,
                                             globalRasterWithGap(-7.0f), {{"NODATA", -7.0}});
        SpatialMap nd;
        check(nd.loadFromFITS(nodata) && nd.getMissingValue() == -7.0 &&
                  near(nd.getValue(kLonMid, kLatMid), 0.6, 1e-7),
              "NODATA honored when BLANK and MISSING are absent");

        // BLANK and MISSING both present and different: both are fill.
        std::vector<float> two = globalRasterWithGap(-1.0f);
        two[2 * 8 + 4] = -9999.0f;
        const std::string both = writeFits("ptm_test_fill_both.fits", 8, 4, two,
                                           {{"BLANK", -1.0}, {"MISSING", -9999.0}});
        SpatialMap b;
        check(b.loadFromFITS(both) && b.getFillCounts().declared == 2 &&
                  near(b.getValue(kLonMid, kLatMid), 0.6, 1e-7),
              "BLANK and MISSING both declared: both values are fill");
    }

    // 4. The Europa case: header says MISSING = -9999, data use -1.0.
    std::printf("4. header misdeclares the sentinel (the Europa Bond-albedo files)\n");
    {
        const std::string path = writeFits("ptm_test_fill_europa.fits", 8, 4,
                                           globalRasterWithGap(-1.0f), {{"MISSING", -9999.0}});
        SpatialMap bare;
        bare.loadFromFITS(path);
        check(near(bare.getValue(kLonMid, kLatMid), 0.75 * 0.6 + 0.25 * -1.0, 1e-7),
              "control: without the albedo policy the -1.0 pixel is still blended in");
        SpatialMap alb;
        alb.loadFromFITS(path, MapFillPolicy::bondAlbedo());
        check(near(alb.getValue(kLonMid, kLatMid), 0.6, 1e-7) &&
                  alb.getFillCounts().out_of_range == 1,
              "Bond-albedo policy excludes it and counts it as outside [0,1]");
    }

    // 5. Seam healing never interpolates from an undeclared fill pixel.
    std::printf("5. cylindrical seam healing\n");
    {
        // 8x4 global raster; both edge columns are entirely -1 (as in the Europa
        // files); column 1 is also -1 in internal row 2.
        std::vector<float> d(32, 0.6f);
        for (int y = 0; y < 4; ++y) {
            d[y * 8 + 0] = -1.0f;
            d[y * 8 + 7] = -1.0f;
        }
        d[2 * 8 + 1] = -1.0f;
        SpatialMap old_style;
        old_style.initializeFromData(d.data(), 8, 4, 0.0, 360.0, -90.0, 90.0);
        const double healed_old = old_style.getValue(0, 2);
        SpatialMap alb;
        alb.initializeFromData(d.data(), 8, 4, 0.0, 360.0, -90.0, 90.0,
                               ptm::MapProjection::EQUIRECTANGULAR, -9999.0,
                               MapFillPolicy::bondAlbedo());
        // Without the policy the edge-column detector never fires: column 1 holds
        // a -1.0 too, so the -1.0 edge column is not "far outside" its neighbor's
        // range, and the raw -1.0 stays in the map as if it were data.
        check(healed_old == -1.0 && old_style.getValue(0, 1) == -1.0,
              "control: without the policy the -1.0 seam column is left in as data (" +
                  std::to_string(healed_old) + ")");
        check(alb.getValue(0, 2) == alb.getMissingValue(),
              "with the policy that seam pixel stays fill instead of a -1.0 blend");
        check(near(alb.getValue(0, 1), 0.6, 1e-7) && near(alb.getValue(7, 1), 0.6, 1e-7),
              "seam pixels with valid interior neighbors are still healed (0.6)");
    }

    // 6. A map with no fill is sampled bit-for-bit the same with or without the
    //    policy (the Moon-map path must not move).
    std::printf("6. all-valid maps are unchanged\n");
    {
        std::vector<float> d(32 * 16);
        for (size_t i = 0; i < d.size(); ++i) {
            d[i] = 0.07f + 0.17f * static_cast<float>((i * 7919) % 101) / 100.0f;
        }
        SpatialMap a, b;
        a.initializeFromData(d.data(), 32, 16, 0.0, 360.0, -90.0, 90.0);
        b.initializeFromData(d.data(), 32, 16, 0.0, 360.0, -90.0, 90.0,
                             ptm::MapProjection::EQUIRECTANGULAR, -9999.0,
                             MapFillPolicy::bondAlbedo());
        bool same = true;
        for (double la = -89.5; la <= 89.5; la += 0.731) {
            for (double lo = 0.0; lo < 360.0; lo += 0.917) {
                if (a.getValue(lo, la) != b.getValue(lo, la)) same = false;
            }
        }
        check(same && b.getFillCounts().total() == 0,
              "every sample identical with and without the policy; 0 fill pixels");
    }

    // 7. A float image whose declared sentinel is not representable as a float:
    //    the pixels hold float(-9999.99), the header the double -9999.99. Matched
    //    exactly, they never compare equal and the sentinel is sampled as data.
    std::printf("7. single-precision sentinel matching\n");
    {
        const std::string path = writeFits("ptm_test_fill_float.fits", 8, 4,
                                           globalRasterWithGap(-9999.99f), {{"MISSING", -9999.99}});
        SpatialMap m;
        m.loadFromFITS(path);
        check(m.getFillCounts().declared == 1 && near(m.getValue(kLonMid, kLatMid), 0.6, 1e-7),
              "MISSING = -9999.99 on a float image is matched at float precision and excluded (" +
                  std::to_string(m.getValue(kLonMid, kLatMid)) + ")");
    }

    std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
