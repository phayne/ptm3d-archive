#!/usr/bin/env python3
"""Verify per-face lunar Bond-albedo bins are correctly longitude/latitude
registered against IAU_MOON (PTM-34, formerly mislabeled PTM-30).

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

The topo mesh and the albedo bin share the MeshUtils icosphere face ordering, so
face f in the bin == face f in the topo mesh. That ordering assumption is itself
cross-checked at res9 by the independent camera-bin element-wise match (which
needs no topo mesh).

Usage:
  verify_moon_albedo_registration.py --bins-dir DIR --topo-dir DIR \
      --fits corrected.fits [--fits-mislabeled old.fits] \
      [--camera-bin Moon_Albedo_res9_camera.bin] [--res 3 4 5 6 7 8 9]
"""
import argparse
import struct
import sys

import numpy as np
from astropy.io import fits

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
    ap.add_argument("--fits", required=True, help="CORRECTED FITS")
    ap.add_argument("--fits-mislabeled", default=None)
    ap.add_argument("--camera-bin", default=None)
    ap.add_argument("--res", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args(argv)

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

        # RMS vs the corrected FITS under every candidate mis-registration. The
        # transform with the SMALLEST RMS is the one that actually explains the
        # bin, which is what names the defective axis -- the maria-dark contrast
        # alone cannot: a pure longitude roll also drives the north/south
        # statistic negative, and a pure latitude flip lands near the contrast
        # gate. This is the attribution the engine's load-time guard defers to.
        transforms = {
            "true (+lat, lon)": (flat, flon),
            "lon+180 (PTM-34)": (flat, flon + 180.0),
            "-lat (PTM-35)": (-flat, flon),
            "-lat & lon+180": (-flat, flon + 180.0),
        }
        rms = {k: float(np.sqrt(np.mean((alb - fits_sample(data, la, lo)) ** 2)))
               for k, (la, lo) in transforms.items()}
        rms_true = rms["true (+lat, lon)"]
        rms_flip = rms["lon+180 (PTM-34)"]
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
        with fits.open(args.fits_mislabeled) as h:
            _ = h[0].data  # provenance touch
        print(f"[verify] (mislabeled FITS referenced: {args.fits_mislabeled})")

    print(f"[verify] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
