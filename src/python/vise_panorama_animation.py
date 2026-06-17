#!/usr/bin/env python3
"""Render the Lunar-VISE panoramas to PNG frames + MP4 animations.

For each mesh (fine, coarse): one PNG per azimuth frame, a stitched
horizontal panorama strip, and an MP4 sweeping the frames in azimuth
order. Also a side-by-side fine|coarse MP4.

Canonical stretch (matches the GUI / Clipper-EGA contract): gamma 1/2.2
on radiance / p99.5, shared across BOTH meshes so they are directly
comparable; sky pixels (FACE_IDX < 0) forced to black. MP4 via a direct
ffmpeg rawvideo pipe (imageio_ffmpeg is not installed here).
"""

import argparse
import json
import os
import subprocess

import numpy as np
from astropy.io import fits
import imageio.v2 as imageio

FFMPEG = "/usr/local/bin/ffmpeg"


def load_frame(path):
    with fits.open(path) as h:
        rad = np.asarray(h[0].data, dtype=np.float64)
        face = np.asarray(h["FACE_IDX"].data)
    sky = face < 0
    rad[sky] = 0.0
    return rad, sky


def to_rgb(rad, sky, lo, hi, gamma):
    # Hard stretch: clip to the surface-radiance percentile window [lo, hi]
    # (the lunar surface spans a narrow band, so 0..max washes it out), then
    # apply gamma. Sky forced black.
    g = np.clip((rad - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma
    img = (g * 255).astype(np.uint8)
    img[sky] = 0
    return np.dstack([img, img, img])


def mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pano-roots", nargs="+", required=True,
                    help="label=dir (dir holds frame_XX/frame_0000.fits)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--lo-pct", type=float, default=2.0,
                    help="Low clip percentile of SURFACE radiance.")
    ap.add_argument("--hi-pct", type=float, default=98.0,
                    help="High clip percentile of SURFACE radiance.")
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="Gamma after clip (1=linear; <1 brightens mids).")
    ap.add_argument("--sbs-name", default="pano_sidebyside.mp4",
                    help="Side-by-side MP4 filename (override to avoid "
                         "clobbering another comparison).")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    order = [fr["index"] for fr in
             sorted(man["frames"], key=lambda f: f["azimuth_deg"])]
    az = {fr["index"]: fr["azimuth_deg"] for fr in man["frames"]}
    roots = dict(p.split("=", 1) for p in args.pano_roots)

    # Shared hard stretch across all frames and meshes (so fine vs coarse
    # are directly comparable). Window from SURFACE pixels only.
    raw = {}
    allv = []
    for label, root in roots.items():
        for k in order:
            r, s = load_frame(os.path.join(root, f"frame_{k:02d}",
                                           "frame_0000.fits"))
            raw[(label, k)] = (r, s)
            surf = r[~s]
            if surf.size:
                allv.append(surf[surf > 0])
    surf_all = np.concatenate(allv)
    lo = np.percentile(surf_all, args.lo_pct)
    hi = np.percentile(surf_all, args.hi_pct)
    print(f"hard stretch: surface p{args.lo_pct:g}={lo:.3f} .. "
          f"p{args.hi_pct:g}={hi:.3f} W m^-2 sr^-1, gamma={args.gamma}")

    fdir = os.path.join(args.out_dir, "frames")
    adir = os.path.join(args.out_dir, "anim")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    rgb = {}
    for label in roots:
        seq = []
        for k in order:
            r, s = raw[(label, k)]
            im = to_rgb(r, s, lo, hi, args.gamma)
            rgb[(label, k)] = im
            png = os.path.join(fdir, f"{label}_az{az[k]:+06.1f}.png")
            imageio.imwrite(png, im)
            seq.append(im)
        # Stitched strip (azimuth increases left->right).
        strip = np.concatenate(seq, axis=1)
        imageio.imwrite(os.path.join(fdir, f"{label}_panorama_strip.png"),
                        strip)
        mp4(seq, os.path.join(adir, f"pano_{label}.mp4"), args.fps)
        print(f"[{label}] {len(seq)} PNGs + strip + pano_{label}.mp4")

    if len(roots) >= 2:
        a, b = list(roots)[0], list(roots)[1]
        h = rgb[(a, order[0])].shape[0]
        bar = np.full((h, 4, 3), 60, np.uint8)
        comp = [np.concatenate([rgb[(a, k)], bar, rgb[(b, k)]], axis=1)
                for k in order]
        mp4(comp, os.path.join(adir, args.sbs_name), args.fps)
        print(f"side-by-side {a}|{b} -> {args.sbs_name}")

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
