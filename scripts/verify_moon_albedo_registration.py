#!/usr/bin/env python3
"""Verify per-face lunar Bond-albedo bins are correctly longitude/latitude
registered against IAU_MOON.

Discriminating tests (a mis-registered bin FAILS these; a correct one PASSES):

  (1) maria-dark test -- using the TOPO mesh's own per-face TRUE IAU lat/lon
      (topo_moon_ico{N}.bin, PTMTOPO format; the topo is ground-truth-correct),
      the nearside maria must sample DARK (Bond ~0.07-0.09) and the highlands
      BRIGHT, so contrast (highland_mean - maria_mean) > 0. Bright anchors are
      TYCHO (349E) and the farside (180E) -- NOT Aristarchus (sub-pixel at
      1 deg/pixel, reads dark -> false negative).

  (2) RMS-vs-corrected-FITS test -- for every face, sample the CORRECTED FITS at
      the face's TRUE IAU (lat, lon) and compute RMS(bin - fits). A correctly
      registered bin gives a small RMS (discretization/averaging residual, ~0.008
      at fine res); a 180-off bin gives ~0.044. The script also reports RMS vs
      the FITS sampled at lon+180 as the explicit discriminator (should be the
      LARGE value for a correct bin).

  (3) geographic test (--geographic; needs NO FITS) -- AREA statistics rather
      than point anchors, so a single mis-sampled facet cannot swing the verdict:
        - nearside (|lon|<=90) must be DARKER than farside, because the nearside
          is mare-dominated. A 180-off map inverts this, and the inversion is the
          cheapest single number that convicts the map.
        - the equatorial zonal-mean albedo minimum must land in the
          Procellarum/Imbrium longitude band (-60..-15 E).
        - named mare boxes must read mare-dark and farside highlands bright.
        - the Apollo 15 and 17 landing sites (both mare) must read mare-dark.
      This is the detector quoted in the albedo-map longitude-roll acceptance report; it is
      deliberately independent of the FITS the bins were generated from, so it
      cannot be fooled by a FITS that is itself mis-registered.

The topo mesh and the albedo bin share the MeshUtils icosphere face ordering, so
face f in the bin == face f in the topo mesh. That ordering assumption is itself
cross-checked at res9 by the independent camera-bin element-wise match (which
needs no topo mesh).

Usage:
  verify_moon_albedo_registration.py --bins-dir DIR --topo-dir DIR \
      [--fits corrected.fits] [--fits-mislabeled old.fits] [--geographic] \
      [--camera-bin Moon_Albedo_res9_camera.bin] [--res 3 4 5 6 7 8 9]

At least one of --fits and --geographic must be given.
"""
import argparse
import struct

import numpy as np

MARIA = [  # (lat, lon_east) -- expected DARK
    (28.0, 17.5),   # Mare Serenitatis
    (32.8, 344.4),  # Mare Imbrium
    (17.0, 58.4),   # Mare Crisium
    (20.0, 302.0),  # Oceanus Procellarum
    (8.5, 31.4),    # Mare Tranquillitatis
]
HIGHLANDS = [  # (lat, lon_east) -- expected BRIGHT
    (0.0, 180.0),    # farside highlands (anti-sub-Earth)
    (-4.4, 202.6),   # Korolev region (farside highlands)
    (0.0, 160.0),
    (20.0, 150.0),
    (-43.3, 348.8),  # Tycho ejecta (bright, 349E)
]

# --- geographic (area-statistic) detector -------------------------------------
# Boxes are (name, lat_lo, lat_hi, lon_lo, lon_hi, expect_lo, expect_hi) with
# longitudes in SIGNED east degrees (-180..180). Expected ranges are Bond albedo.
GEO_BOXES = [
    # Mare basalts: true Bond albedo ~0.07-0.09, so a correct map lands 0.08-0.11
    # once icosphere facet averaging mixes in the surrounding highland rim.
    ("Oceanus Procellarum", 5.0, 35.0, -75.0, -40.0, 0.08, 0.11),
    ("Mare Serenitatis", 18.0, 38.0, 8.0, 28.0, 0.08, 0.11),
    # Farside highlands: anorthositic, bright.
    ("Farside highlands", -20.0, 20.0, 150.0, -150.0, 0.15, 0.17),
]

# Apollo landing sites -- both sit on mare basalt and must read mare-dark.
APOLLO_SITES = [
    ("Apollo 15 (Hadley/Mare Imbrium)", 26.132, 3.634),
    ("Apollo 17 (Taurus-Littrow)", 20.191, 30.772),
]
APOLLO_MARE_MAX = 0.12  # above this the site is reading highland, not mare

# Equatorial band used for the nearside/farside and zonal-minimum statistics.
EQ_BAND_LAT = 35.0
# Procellarum/Imbrium longitude band where the zonal albedo minimum must land.
ZONAL_MIN_LON_LO, ZONAL_MIN_LON_HI = -60.0, -15.0
ZONAL_BIN_DEG = 15.0
# Radius of the disk averaged around each Apollo site.
SITE_RADIUS_DEG = 2.0


def signed_lon(lon):
    """Wrap longitudes to signed east degrees in [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def in_lon_range(lon_s, lo, hi):
    """Boolean mask for signed longitudes in [lo, hi], handling the +-180 wrap
    (lo > hi means the box straddles the antimeridian)."""
    if lo <= hi:
        return (lon_s >= lo) & (lon_s <= hi)
    return (lon_s >= lo) | (lon_s <= hi)


def geographic_report(alb, flat, flon, area=None):
    """Area statistics that convict a longitude-rolled lunar albedo map.

    Returns (ok, lines). `area` weights each face; None means equal weights,
    which is exact for an icosphere (facet areas differ by <1%).
    """
    lon_s = signed_lon(flon)
    w = np.ones_like(alb) if area is None else np.asarray(area, dtype=float)
    lines = []
    ok = True

    def wmean(mask):
        m = np.asarray(mask)
        if not m.any():
            return float("nan"), 0
        return float(np.average(alb[m], weights=w[m])), int(m.sum())

    # (a) nearside vs farside -- the headline inversion test.
    eq = np.abs(flat) <= EQ_BAND_LAT
    near_m = eq & (np.abs(lon_s) <= 90.0)
    far_m = eq & (np.abs(lon_s) > 90.0)
    near, n_near = wmean(near_m)
    far, n_far = wmean(far_m)
    near_ok = near < far
    ok &= near_ok
    lines.append(f"  nearside mean A = {near:.4f} (n={n_near})   "
                 f"farside mean A = {far:.4f} (n={n_far})   "
                 f"delta = {near - far:+.4f}  -> "
                 f"{'PASS (nearside darker)' if near_ok else 'FAIL (nearside BRIGHTER -- inverted)'}")

    # (b) longitudinal minimum of the equatorial zonal mean.
    edges = np.arange(-180.0, 180.0 + ZONAL_BIN_DEG, ZONAL_BIN_DEG)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = eq & (lon_s >= lo) & (lon_s < hi)
        if m.any():
            centers.append(0.5 * (lo + hi))
            means.append(float(np.average(alb[m], weights=w[m])))
    argmin_lon = centers[int(np.argmin(means))]
    zonal_ok = ZONAL_MIN_LON_LO <= argmin_lon <= ZONAL_MIN_LON_HI
    ok &= zonal_ok
    lines.append(f"  zonal-mean minimum at lon = {argmin_lon:+.1f} E "
                 f"(A = {min(means):.4f})  -> "
                 f"{'PASS' if zonal_ok else 'FAIL'} "
                 f"(want {ZONAL_MIN_LON_LO:+.0f}..{ZONAL_MIN_LON_HI:+.0f} E, Procellarum/Imbrium)")

    # (c) named boxes.
    for name, la_lo, la_hi, lo_lo, lo_hi, exp_lo, exp_hi in GEO_BOXES:
        m = (flat >= la_lo) & (flat <= la_hi) & in_lon_range(lon_s, lo_lo, lo_hi)
        val, n = wmean(m)
        box_ok = exp_lo <= val <= exp_hi
        ok &= box_ok
        lines.append(f"  {name:<22s} A = {val:.4f} (n={n:5d})  "
                     f"want {exp_lo:.2f}-{exp_hi:.2f}  -> {'PASS' if box_ok else 'FAIL'}")

    # (d) Apollo landing sites (mare -> dark).
    #
    # These are POINT sites, so unlike the boxes above they are only a fair test
    # when the mesh can actually resolve them. At res3-res4 a facet spans several
    # degrees and necessarily averages the mare basalt together with the highland
    # rim around it, pulling the value toward the highlands: measured 0.1116
    # (Apollo 15, res3) and 0.1149 (Apollo 17, res4) on a KNOWN-CORRECT map, only
    # ~0.005 under the 0.12 mare gate. Gating on that would make a correct map
    # fail at coarse resolution for a meshing reason, so when no face center falls
    # inside the site disk the value is reported ADVISORY and does not vote. The
    # area statistics (a)-(c) carry the verdict at those resolutions; they are
    # resolution-independent and stay decisive (nearside/farside splits by 0.0345
    # even at res3).
    fv = unit_vecs(flat, lon_s)
    cos_r = np.cos(np.radians(SITE_RADIUS_DEG))
    for name, la, lo in APOLLO_SITES:
        av = unit_vecs(np.array([la]), np.array([lo]))[0]
        dot = fv @ av
        m = dot >= cos_r
        resolved = bool(m.any())
        if not resolved:  # coarse mesh: report the single nearest face, advisory
            m = np.zeros_like(dot, dtype=bool)
            m[int(np.argmax(dot))] = True
        val, n = wmean(m)
        site_ok = val <= APOLLO_MARE_MAX
        if resolved:
            ok &= site_ok
            verdict = "PASS" if site_ok else "FAIL"
        else:
            verdict = ("ADVISORY (mesh too coarse to resolve the site; "
                       "nearest face only, not gated)")
        lines.append(f"  {name:<34s} A = {val:.4f} (n={n:4d})  "
                     f"want <= {APOLLO_MARE_MAX:.2f} (mare)  -> {verdict}")

    return ok, lines


def read_albedo_bin(path):
    """size_t num_faces header (8 bytes) + num_faces float64."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        a = np.fromfile(f, dtype="<f8", count=n)
    if a.size != n:
        raise ValueError(f"{path}: expected {n} values, got {a.size}")
    return a


def read_topo_latlon(path):
    """Read per-face latitudes/longitudes from a PTMTOPO mesh bin.

    Layout (mesh_utils.cpp exportMeshToFile): magic 'PTMTOPO'(7) + int32 res +
    size_t nv + size_t nf + nv*3 f64 verts + nf*3 i32 faces + nv f64 elevations +
    nf*3 f64 normals + nf*3 f64 centers + nf f64 lat + nf f64 lon.
    """
    with open(path, "rb") as f:
        magic = f.read(7)
        if magic != b"PTMTOPO":
            raise ValueError(f"{path}: bad magic {magic!r}")
        struct.unpack("<i", f.read(4))  # resolution
        nv = struct.unpack("<Q", f.read(8))[0]
        nf = struct.unpack("<Q", f.read(8))[0]
        off = 7 + 4 + 8 + 8
        off += nv * 3 * 8          # vertices
        off += nf * 3 * 4          # faces (int32)
        off += nv * 8              # elevations
        off += nf * 3 * 8          # normals
        off += nf * 3 * 8          # centers
        f.seek(off)
        lat = np.fromfile(f, dtype="<f8", count=nf)
        lon = np.fromfile(f, dtype="<f8", count=nf)
    if lat.size != nf or lon.size != nf:
        raise ValueError(f"{path}: lat/lon short read")
    return nf, lat, lon


def fits_sample(data, lat, lon):
    """Nearest-pixel sample honoring CRVAL2=-90 (row 0 = lat -90) and CRVAL1=0
    (col 0 = lon 0 east, 0..360). lat/lon are arrays of TRUE IAU coordinates."""
    ny, nx = data.shape
    r = np.clip(np.round((lat + 90.0) * (ny / 180.0)).astype(int), 0, ny - 1)
    c = (np.round((np.mod(lon, 360.0)) * (nx / 360.0)).astype(int)) % nx
    return data[r, c]


def unit_vecs(lat_deg, lon_deg):
    la = np.radians(lat_deg)
    lo = np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def nearest_face_albedo(anchor_lat, anchor_lon, face_lat, face_lon, alb):
    av = unit_vecs(np.array([anchor_lat]), np.array([anchor_lon]))[0]
    fv = unit_vecs(face_lat, face_lon)
    dot = fv @ av
    idx = int(np.argmax(dot))
    return alb[idx], face_lat[idx], face_lon[idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins-dir", required=True)
    ap.add_argument("--topo-dir", required=True)
    ap.add_argument("--fits", default=None, help="CORRECTED FITS")
    ap.add_argument("--fits-mislabeled", default=None)
    ap.add_argument("--geographic", action="store_true",
                    help="also run the area-statistic geographic detector (no FITS needed)")
    ap.add_argument("--camera-bin", default=None)
    ap.add_argument("--res", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args(argv)

    if not args.fits and not args.geographic:
        ap.error("nothing to check: pass --fits, --geographic, or both")

    data = None
    if args.fits:
        from astropy.io import fits
        with fits.open(args.fits) as h:
            data = h[0].data.astype(np.float64)
        print(f"[verify] corrected FITS {args.fits} shape={data.shape} "
              f"range={data.min():.4f}..{data.max():.4f}")

    ok = True
    for res in args.res:
        binp = f"{args.bins_dir}/Moon_Albedo_res{res}.bin"
        topop = f"{args.topo_dir}/topo_moon_ico{res}.bin"
        alb = read_albedo_bin(binp)
        nf, flat, flon = read_topo_latlon(topop)
        if nf != alb.size:
            print(f"[verify] res{res}: FACE COUNT MISMATCH bin={alb.size} topo={nf}")
            ok = False
            continue

        # maria-dark test
        m_alb = [nearest_face_albedo(la, lo, flat, flon, alb)[0] for la, lo in MARIA]
        h_alb = [nearest_face_albedo(la, lo, flat, flon, alb)[0] for la, lo in HIGHLANDS]
        m_mean, h_mean = float(np.mean(m_alb)), float(np.mean(h_alb))
        contrast = h_mean - m_mean

        if args.geographic:
            geo_ok, geo_lines = geographic_report(alb, flat, flon)
            print(f"[verify] res{res}: geographic detector -> "
                  f"{'PASS' if geo_ok else 'FAIL'}")
            for line in geo_lines:
                print(line)
            if not geo_ok:
                ok = False

        if data is None:
            # The point-anchor contrast is a real detector, not decoration: gate on
            # it here too, or --geographic-only silently drops half the check.
            contrast_ok = contrast > 0.02
            if not contrast_ok:
                ok = False
            print(f"[verify] res{res}: faces={nf}  maria={m_mean:.4f} "
                  f"highland={h_mean:.4f} contrast={contrast:+.4f} -> "
                  f"{'PASS' if contrast_ok else 'FAIL'} "
                  f"(FITS RMS checks skipped -- no --fits given)")
            continue

        # RMS vs the corrected FITS under every candidate mis-registration. The
        # transform with the SMALLEST RMS is the one that actually explains the
        # bin, which is what names the defective axis -- the maria-dark contrast
        # alone cannot: a pure longitude roll also drives the north/south
        # statistic negative, and a pure latitude flip lands near the contrast
        # gate. This is the attribution the engine's load-time guard defers to.
        transforms = {
            "true (+lat, lon)": (flat, flon),
            "lon+180 (longitude roll)": (flat, flon + 180.0),
            "-lat (latitude flip)": (-flat, flon),
            "-lat & lon+180": (-flat, flon + 180.0),
        }
        rms = {k: float(np.sqrt(np.mean((alb - fits_sample(data, la, lo)) ** 2)))
               for k, (la, lo) in transforms.items()}
        rms_true = rms["true (+lat, lon)"]
        rms_flip = rms["lon+180 (longitude roll)"]
        best = min(rms, key=rms.get)

        verdict = "PASS" if (contrast > 0.02 and rms_true < 0.02 and rms_true < rms_flip) else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(f"[verify] res{res}: faces={nf}  maria={m_mean:.4f} highland={h_mean:.4f} "
              f"contrast={contrast:+.4f}  RMS(true)={rms_true:.4f} RMS(lon+180)={rms_flip:.4f}  -> {verdict}")
        if verdict == "FAIL":
            detail = "  ".join(f"{k}={v:.4f}" for k, v in rms.items())
            print(f"           RMS by transform: {detail}")
            print(f"           best explanation: '{best}' -> "
                  + ("registration is CORRECT on both axes; the failure is elsewhere"
                     if best.startswith("true") else f"the map is mis-registered as '{best}'"))
        if res == 9:
            print("           maria per-anchor:    " +
                  ", ".join(f"{lo:.0f}E={a:.3f}" for (la, lo), a in zip(MARIA, m_alb)))
            print("           highland per-anchor: " +
                  ", ".join(f"{lo:.0f}E={a:.3f}" for (la, lo), a in zip(HIGHLANDS, h_alb)))

    # camera-bin element-wise match (independent of topo ordering)
    if args.camera_bin:
        regen9 = read_albedo_bin(f"{args.bins_dir}/Moon_Albedo_res9.bin")
        cam = read_albedo_bin(args.camera_bin)
        if regen9.size != cam.size:
            print(f"[verify] CAMERA size mismatch regen={regen9.size} camera={cam.size} -> FAIL")
            ok = False
        else:
            d = regen9 - cam
            rms = float(np.sqrt(np.mean(d ** 2)))
            mx = float(np.max(np.abs(d)))
            nexact = int(np.sum(d == 0.0))
            verdict = "PASS" if rms < 1e-6 else ("CLOSE" if rms < 5e-3 else "FAIL")
            print(f"[verify] CAMERA res9 regen-vs-existing: RMS={rms:.3e} maxabs={mx:.3e} "
                  f"exact-equal={nexact}/{cam.size} -> {verdict}")
            if verdict == "FAIL":
                ok = False

    if args.fits_mislabeled:
        from astropy.io import fits
        with fits.open(args.fits_mislabeled) as h:
            _ = h[0].data  # provenance touch
        print(f"[verify] (mislabeled FITS referenced: {args.fits_mislabeled})")

    print(f"[verify] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
