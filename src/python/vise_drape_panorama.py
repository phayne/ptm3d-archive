#!/usr/bin/env python3
"""Camera-perspective panorama of the FINE high-res simulated image
DRAPED onto the COARSE mesh.

Per frame the COARSE-mesh render supplies the 3D geometry/silhouette
(which pixels see surface, and the ground position of each). That ground
position is used to sample a FINE ortho texture (the high-res simulated
image as a function of ground E,N, accumulated from the fine panorama).
So you see the fine imagery painted on the coarse terrain: where the
coarse DEM mislocates the surface, the fine detail is dragged to the
wrong place and the panorama is visibly distorted vs the true fine view.

Outputs (per run dir):
  anim/pano_drape_coarse.mp4     draped camera panorama (fine img on coarse mesh)
  anim/pano_drape_vs_fine.mp4    draped | true fine panorama
  frames/drape_az*.png, drape_panorama_strip.png
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


def hits_en(fits_path, cam, d, g, E, N):
    """Return (en (ny,nx,2), surface mask) for a render FITS."""
    rad, ext = read_hdus(fits_path)
    dist = np.asarray(ext["DIST_KM"], float) * 1000.0
    face = np.asarray(ext["FACE_IDX"])
    surf = (face >= 0) & (dist > 0)
    hit = cam[None, None, :] + dist[..., None] * d
    rel = hit - g
    en = np.stack([rel @ E, rel @ N], axis=2)
    return rad, en, surf


def mp4(frames, path, fps):
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
        p.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
    p.stdin.close()
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    if p.returncode or not os.path.exists(path) \
            or os.path.getsize(path) < 1024:
        raise RuntimeError(f"ffmpeg failed {path}\n{err[-600:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scene", default="tmp/vise/scene_params.json")
    ap.add_argument("--fine-root", required=True)
    ap.add_argument("--coarse-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tex-res-m", type=float, default=1.5,
                    help="Fine ground-texture resolution (m).")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--gamma", type=float, default=0.9)
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

    # ---- Pass 1: fine ground texture T(E,N) from the fine panorama ----
    f_en, f_rad, f_surf, dirs = [], [], [], {}
    for fr in order:
        k = fr["index"]
        fwd, r, u = cam_basis(cam, np.array(fr["look_at_m"]), up_world)
        d = pixel_dirs(nx, ny, fx, fy, fwd, r, u)
        dirs[k] = d
        rad, en, s = hits_en(os.path.join(
            args.fine_root, f"frame_{k:02d}", "frame_0000.fits"),
            cam, d, g, E, N)
        f_en.append(en[s]); f_rad.append(rad[s])
    FEN = np.concatenate(f_en); FRAD = np.concatenate(f_rad)
    lo = np.percentile(FEN, 0.5, axis=0)
    hi = np.percentile(FEN, 99.5, axis=0)
    span = hi - lo; lo -= 0.05 * span; hi += 0.05 * span
    res = args.tex_res_m
    TW = int(np.ceil((hi[0] - lo[0]) / res))
    TH = int(np.ceil((hi[1] - lo[1]) / res))
    gx = np.floor((FEN[:, 0] - lo[0]) / res).astype(np.int64)
    gy = np.floor((FEN[:, 1] - lo[1]) / res).astype(np.int64)
    m = (gx >= 0) & (gx < TW) & (gy >= 0) & (gy < TH)
    idx = gy[m] * TW + gx[m]
    s_ = np.bincount(idx, FRAD[m], minlength=TW * TH)
    c_ = np.bincount(idx, minlength=TW * TH).astype(float)
    T = np.full(TW * TH, np.nan)
    T[c_ > 0] = s_[c_ > 0] / c_[c_ > 0]
    T = T.reshape(TH, TW)
    surf_vals = T[np.isfinite(T)]
    s_lo, s_hi = np.percentile(surf_vals, [2, 98])
    print(f"texture {TW}x{TH} @ {res} m, stretch [{s_lo:.2f},{s_hi:.2f}]")

    def stretch(img):
        gimg = np.clip((img - s_lo) / max(s_hi - s_lo, 1e-9), 0, 1) \
            ** args.gamma
        out = (np.where(np.isfinite(img), gimg, 0.0) * 255).astype(np.uint8)
        return np.dstack([out, out, out])

    def sample_T(en, surf):
        gx = np.floor((en[..., 0] - lo[0]) / res).astype(np.int64)
        gy = np.floor((en[..., 1] - lo[1]) / res).astype(np.int64)
        ok = surf & (gx >= 0) & (gx < TW) & (gy >= 0) & (gy < TH)
        out = np.full(en.shape[:2], np.nan)
        out[ok] = T[gy[ok], gx[ok]]
        return out

    # ---- Pass 2: per frame, drape onto coarse geometry ----
    os.makedirs(os.path.join(args.out_dir, "anim"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "frames"), exist_ok=True)
    import imageio.v2 as imageio
    drape_seq, fine_seq = [], []
    az = {fr["index"]: fr["azimuth_deg"] for fr in order}
    for fr in order:
        k = fr["index"]
        d = dirs[k]
        _, cen, csurf = hits_en(os.path.join(
            args.coarse_root, f"frame_{k:02d}", "frame_0000.fits"),
            cam, d, g, E, N)
        draped = stretch(sample_T(cen, csurf))
        frad, _ = read_hdus(os.path.join(
            args.fine_root, f"frame_{k:02d}", "frame_0000.fits"))
        # true fine view uses the same texture stretch for comparability
        fimg = stretch(np.where(frad > 0, frad, np.nan))
        imageio.imwrite(os.path.join(args.out_dir, "frames",
                        f"drape_az{az[k]:+06.1f}.png"), draped)
        drape_seq.append(draped); fine_seq.append(fimg)

    imageio.imwrite(os.path.join(args.out_dir, "frames",
                    "drape_panorama_strip.png"),
                    np.concatenate(drape_seq, axis=1))
    mp4(drape_seq, os.path.join(args.out_dir, "anim",
        "pano_drape_coarse.mp4"), args.fps)
    bar = np.full((drape_seq[0].shape[0], 4, 3), 60, np.uint8)
    sbs = [np.concatenate([f, bar, dp], axis=1)
           for f, dp in zip(fine_seq, drape_seq)]
    mp4(sbs, os.path.join(args.out_dir, "anim",
        "pano_drape_vs_fine.mp4"), args.fps)
    print(f"wrote {args.out_dir}/anim/pano_drape_coarse.mp4 + "
          f"pano_drape_vs_fine.mp4 (fine | draped)")


if __name__ == "__main__":
    main()
