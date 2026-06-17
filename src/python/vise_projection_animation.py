#!/usr/bin/env python3
"""Animate the fine-scale simulated panorama back-projected onto the
coarse mesh, building up frame-by-frame over the azimuth sweep.

For every panorama frame the FINE-render radiance (the truth image the
camera sees) is laid onto the surface using the per-pixel ray hit of
(a) the COARSE mesh and (b) the FINE mesh, accumulated into a common
top-down orthographic grid in the site ENU frame. As the camera sweeps,
the ortho mosaic fills in; the coarse-mesh version shows the geometric
ortho-rectification error a coarse DEM introduces.

Outputs (per run dir):
  anim/pano_proj_coarse.mp4      cumulative fine image on the coarse DEM
  anim/pano_proj_sidebyside.mp4  fine-on-fine (truth) | fine-on-coarse
  frames/proj_*_final.png        final cumulative stills
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vise_backproject import cam_basis, pixel_dirs, read_hdus  # noqa: E402

FFMPEG = "/usr/local/bin/ffmpeg"


def gray_rgb(img, lo, hi, gamma):
    g = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma
    out = (np.where(np.isfinite(img), g, 0.0) * 255).astype(np.uint8)
    return np.dstack([out, out, out])[::-1]            # row 0 = north (up)


def mp4(frames, path, fps):
    # libx264 + yuv420p needs even dimensions; pad with black if odd.
    h, w = frames[0].shape[:2]
    H2, W2 = h + (h & 1), w + (w & 1)
    p = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W2}x{H2}", "-r", str(fps), "-i", "-", "-an",
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)
    for fr in frames:
        if (H2, W2) != fr.shape[:2]:
            fr = np.pad(fr, ((0, H2 - h), (0, W2 - w), (0, 0)))
        p.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    p.stdin.close()
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    if p.returncode != 0 or not os.path.exists(path) \
            or os.path.getsize(path) < 1024:
        raise RuntimeError(f"ffmpeg failed for {path}:\n{err[-800:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scene", default="tmp/vise/scene_params.json")
    ap.add_argument("--fine-root", required=True)
    ap.add_argument("--coarse-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res-m", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--hold", type=int, default=1,
                    help="Repeat each cumulative state N times (slower).")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    scene = json.load(open(args.scene))
    cam = np.array(man["cam_pos_m"])
    up_world = np.array(man["up"])
    nx, ny = man["pixels"]
    fx = man["focal_mm"] / man["pixel_pitch_mm"][0]
    fy = man["focal_mm"] / man["pixel_pitch_mm"][1]
    g = np.array(scene["ground_point_bf_m"])
    E = np.array(scene["site_E"]); N = np.array(scene["site_N"])

    order = sorted(man["frames"], key=lambda f: f["azimuth_deg"])

    # Pass 1: per-frame (fine_pts, coarse_pts, rad); collect extent.
    per = []
    allf = []
    for fr in order:
        k = fr["index"]
        fwd, right, up = cam_basis(cam, np.array(fr["look_at_m"]), up_world)
        d = pixel_dirs(nx, ny, fx, fy, fwd, right, up)
        rad_f, ext_f = read_hdus(os.path.join(
            args.fine_root, f"frame_{k:02d}", "frame_0000.fits"))
        _, ext_c = read_hdus(os.path.join(
            args.coarse_root, f"frame_{k:02d}", "frame_0000.fits"))
        df = np.asarray(ext_f["DIST_KM"], float) * 1000.0
        dc = np.asarray(ext_c["DIST_KM"], float) * 1000.0
        faf = np.asarray(ext_f["FACE_IDX"])
        fac = np.asarray(ext_c["FACE_IDX"])
        v = ((faf >= 0) & (fac >= 0) & (df > 0) & (dc > 0)).ravel()
        if not v.any():
            continue
        hf = cam[None, None, :] + df[..., None] * d
        hc = cam[None, None, :] + dc[..., None] * d
        rf = hf - g
        rc = hc - g
        fpt = np.column_stack([(rf @ E).ravel()[v], (rf @ N).ravel()[v]])
        cpt = np.column_stack([(rc @ E).ravel()[v], (rc @ N).ravel()[v]])
        rr = rad_f.ravel()[v]
        per.append((fpt, cpt, rr))
        allf.append(fpt)

    allf = np.concatenate(allf)
    lo_xy = np.percentile(allf, 0.5, axis=0)
    hi_xy = np.percentile(allf, 99.5, axis=0)
    span = hi_xy - lo_xy
    lo_xy -= 0.05 * span
    hi_xy += 0.05 * span
    res = args.res_m
    W = int(np.ceil((hi_xy[0] - lo_xy[0]) / res))
    H = int(np.ceil((hi_xy[1] - lo_xy[1]) / res))

    fs = np.zeros(W * H); fc = np.zeros(W * H)
    cs = np.zeros(W * H); cc = np.zeros(W * H)

    def add(sumv, cntv, pts, vals):
        gx = np.floor((pts[:, 0] - lo_xy[0]) / res).astype(np.int64)
        gy = np.floor((pts[:, 1] - lo_xy[1]) / res).astype(np.int64)
        m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        idx = gy[m] * W + gx[m]
        sumv += np.bincount(idx, vals[m], minlength=W * H)
        cntv += np.bincount(idx, minlength=W * H)

    def img(sumv, cntv):
        out = np.full(W * H, np.nan)
        nz = cntv > 0
        out[nz] = sumv[nz] / cntv[nz]
        return out.reshape(H, W)

    fine_seq, coarse_seq = [], []
    for fpt, cpt, rr in per:
        add(fs, fc, fpt, rr)
        add(cs, cc, cpt, rr)
        fine_seq.append(img(fs, fc))
        coarse_seq.append(img(cs, cc))

    fin_final = fine_seq[-1]
    surf = fin_final[np.isfinite(fin_final)]
    lo, hi = np.percentile(surf, [2, 98])

    bar = None
    fine_rgb = [gray_rgb(a, lo, hi, args.gamma) for a in fine_seq]
    coarse_rgb = [gray_rgb(a, lo, hi, args.gamma) for a in coarse_seq]

    os.makedirs(os.path.join(args.out_dir, "anim"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "frames"), exist_ok=True)
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(args.out_dir, "frames",
                                 "proj_coarse_final.png"), coarse_rgb[-1])
    imageio.imwrite(os.path.join(args.out_dir, "frames",
                                 "proj_fine_final.png"), fine_rgb[-1])

    def expand(seq):
        out = []
        for fr in seq:
            out += [fr] * args.hold
        return out

    mp4(expand(coarse_rgb),
        os.path.join(args.out_dir, "anim", "pano_proj_coarse.mp4"),
        args.fps)

    h = fine_rgb[0].shape[0]
    bar = np.full((h, 4, 3), 60, np.uint8)
    sbs = [np.concatenate([f, bar, c], axis=1)
           for f, c in zip(fine_rgb, coarse_rgb)]
    mp4(expand(sbs),
        os.path.join(args.out_dir, "anim", "pano_proj_sidebyside.mp4"),
        args.fps)

    print(f"grid {W}x{H} @ {res} m, {len(per)} frames, stretch "
          f"[{lo:.2f},{hi:.2f}]")
    print(f"wrote {args.out_dir}/anim/pano_proj_coarse.mp4 + "
          f"pano_proj_sidebyside.mp4 (+ final PNGs)")


if __name__ == "__main__":
    main()
