#!/usr/bin/env python3
"""Step 5: back-project the panorama pixels onto the fine and coarse meshes.

The fine render is the truth image (what the camera sees). Each pixel ray
is reconstructed exactly as ptm_camera_sim does it (pinhole, anisotropic
pitch, image-top = +up), then its surface hit is taken from the per-pixel
DIST_KM of (a) the FINE render and (b) the COARSE render. The FINE radiance
is deposited at both hit locations into a common top-down orthographic
grid in the site local ENU frame. The fine vs coarse difference is the
pure DEM-resolution geolocation (ortho-rectification) error.

Outputs output/visecam_backproj/backproj.npz.
"""

import argparse
import glob
import json
import os

import numpy as np
from astropy.io import fits


def cam_basis(cam, look, up_world):
    fwd = look - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def pixel_dirs(nx, ny, fx, fy, fwd, right, up):
    px = np.arange(nx) + 0.5
    py = np.arange(ny) + 0.5
    X, Y = np.meshgrid(px, py)                       # (ny, nx)
    x_d = (X - 0.5 * nx) / fx
    y_d = -((Y - 0.5 * ny) / fy)                     # image top = +up
    d = (fwd[None, None, :]
         + x_d[..., None] * right[None, None, :]
         + y_d[..., None] * up[None, None, :])
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return d                                          # (ny, nx, 3)


def read_hdus(path):
    with fits.open(path) as h:
        rad = np.asarray(h[0].data, dtype=np.float64)
        ext = {hd.name: hd.data for hd in h[1:] if hd.name}
    return rad, ext


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="output/vise_pano_manifest.json")
    ap.add_argument("--scene", default="tmp/vise/scene_params.json")
    ap.add_argument("--fine-root", default="output/visecam_pano_fine")
    ap.add_argument("--coarse-root", default="output/visecam_pano_coarse")
    ap.add_argument("--out", default="output/visecam_backproj/backproj.npz")
    ap.add_argument("--res-m", type=float, default=5.0,
                    help="Ortho grid resolution (m) in scene units.")
    ap.add_argument("--emissivity", type=float, default=0.95,
                    help="Thermal emissivity for the EPF radiance.")
    ap.add_argument("--pad-frac", type=float, default=0.05)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    scene = json.load(open(args.scene))
    cam = np.array(man["cam_pos_m"])
    up_world = np.array(man["up"])
    nx, ny = man["pixels"]
    fx = man["focal_mm"] / man["pixel_pitch_mm"][0]
    fy = man["focal_mm"] / man["pixel_pitch_mm"][1]

    g = np.array(scene["ground_point_bf_m"])
    E = np.array(scene["site_E"])
    N = np.array(scene["site_N"])
    U = np.array(scene["site_U"])

    SIGMA = 5.670374419e-8
    eps = args.emissivity
    # Energy-conserving EPF: L(theta_e) = K * (cos theta_e)^0.5 with
    # K = 1.25 * eps*sigma*T^4/pi so the hemispherical exitance stays
    # eps*sigma*T^4 (int_0^1 u^1.5 du = 0.4 -> 2*pi*K*0.4 = eps sig T4).
    def epf_radiance(T, emi_deg):
        cos_e = np.clip(np.cos(np.radians(emi_deg)), 0.0, 1.0)
        return 1.25 * eps * SIGMA * np.power(np.clip(T, 0, None), 4) \
            / np.pi * np.sqrt(cos_e)

    nfr = len(man["frames"])
    fine_pts, coarse_pts, rads, disps = [], [], [], []
    pha_fs, pha_cs = [], []
    inc_fs, inc_cs = [], []
    tmp_fs, tmp_cs, thm_fs, thm_cs = [], [], [], []

    for fr in man["frames"]:
        k = fr["index"]
        look = np.array(fr["look_at_m"])
        fwd, right, up = cam_basis(cam, look, up_world)
        d = pixel_dirs(nx, ny, fx, fy, fwd, right, up)   # (ny,nx,3)

        ff = os.path.join(args.fine_root, f"frame_{k:02d}", "frame_0000.fits")
        cf = os.path.join(args.coarse_root, f"frame_{k:02d}", "frame_0000.fits")
        rad_f, ext_f = read_hdus(ff)
        rad_c, ext_c = read_hdus(cf)

        dist_f = np.asarray(ext_f["DIST_KM"], dtype=np.float64) * 1000.0
        dist_c = np.asarray(ext_c["DIST_KM"], dtype=np.float64) * 1000.0
        face_f = np.asarray(ext_f["FACE_IDX"])
        face_c = np.asarray(ext_c["FACE_IDX"])
        pha_f = np.asarray(ext_f["PHA_ANG"], dtype=np.float64)
        pha_c = np.asarray(ext_c["PHA_ANG"], dtype=np.float64)
        inc_f = np.asarray(ext_f["INC_ANG"], dtype=np.float64)
        inc_c = np.asarray(ext_c["INC_ANG"], dtype=np.float64)
        tmp_f = np.asarray(ext_f["TEMP_K"], dtype=np.float64)
        tmp_c = np.asarray(ext_c["TEMP_K"], dtype=np.float64)
        emi_f = np.asarray(ext_f["EMI_ANG"], dtype=np.float64)
        emi_c = np.asarray(ext_c["EMI_ANG"], dtype=np.float64)
        thm_f = epf_radiance(tmp_f, emi_f)
        thm_c = epf_radiance(tmp_c, emi_c)

        valid = (face_f >= 0) & (face_c >= 0) & (dist_f > 0) & (dist_c > 0)
        if not valid.any():
            continue

        hit_f = cam[None, None, :] + dist_f[..., None] * d
        hit_c = cam[None, None, :] + dist_c[..., None] * d
        rel_f = hit_f - g
        rel_c = hit_c - g
        ef = rel_f @ E; nf = rel_f @ N
        ec = rel_c @ E; nc = rel_c @ N

        v = valid.ravel()
        fine_pts.append(np.column_stack([ef.ravel()[v], nf.ravel()[v]]))
        coarse_pts.append(np.column_stack([ec.ravel()[v], nc.ravel()[v]]))
        rads.append(rad_f.ravel()[v])
        disps.append(np.hypot((ec - ef).ravel()[v], (nc - nf).ravel()[v]))
        pha_fs.append(pha_f.ravel()[v])
        pha_cs.append(pha_c.ravel()[v])
        inc_fs.append(inc_f.ravel()[v])
        inc_cs.append(inc_c.ravel()[v])
        tmp_fs.append(tmp_f.ravel()[v])
        tmp_cs.append(tmp_c.ravel()[v])
        thm_fs.append(thm_f.ravel()[v])
        thm_cs.append(thm_c.ravel()[v])

    fine_pts = np.concatenate(fine_pts)
    coarse_pts = np.concatenate(coarse_pts)
    rads = np.concatenate(rads)
    disps = np.concatenate(disps)
    pha_fs = np.concatenate(pha_fs)
    pha_cs = np.concatenate(pha_cs)
    pha_err = np.abs(pha_cs - pha_fs)
    inc_fs = np.concatenate(inc_fs)
    inc_cs = np.concatenate(inc_cs)
    inc_err = np.abs(inc_cs - inc_fs)
    tmp_fs = np.concatenate(tmp_fs)
    tmp_cs = np.concatenate(tmp_cs)
    tmp_err = np.abs(tmp_cs - tmp_fs)
    thm_fs = np.concatenate(thm_fs)
    thm_cs = np.concatenate(thm_cs)
    thm_err = np.abs(thm_cs - thm_fs)

    # Common ortho extent from the truth (fine) hits, robust to outliers.
    lo = np.percentile(fine_pts, 0.5, axis=0)
    hi = np.percentile(fine_pts, 99.5, axis=0)
    span = hi - lo
    lo -= args.pad_frac * span
    hi += args.pad_frac * span
    res = args.res_m
    W = int(np.ceil((hi[0] - lo[0]) / res))
    H = int(np.ceil((hi[1] - lo[1]) / res))

    def accumulate(pts, vals):
        gx = np.floor((pts[:, 0] - lo[0]) / res).astype(np.int64)
        gy = np.floor((pts[:, 1] - lo[1]) / res).astype(np.int64)
        m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        idx = gy[m] * W + gx[m]
        s = np.bincount(idx, weights=vals[m], minlength=W * H)
        c = np.bincount(idx, minlength=W * H).astype(np.float64)
        out = np.full(W * H, np.nan)
        nz = c > 0
        out[nz] = s[nz] / c[nz]
        return out.reshape(H, W), c.reshape(H, W)

    fine_img, fine_cnt = accumulate(fine_pts, rads)
    coarse_img, _ = accumulate(coarse_pts, rads)
    disp_img, _ = accumulate(fine_pts, disps)   # disp mapped at truth location
    # Phase angle: truth phase placed at the fine hit; coarse-DEM phase at the
    # coarse hit; the error is mapped at the truth location.
    fine_pha_img, _ = accumulate(fine_pts, pha_fs)
    coarse_pha_img, _ = accumulate(coarse_pts, pha_cs)
    pha_err_img, _ = accumulate(fine_pts, pha_err)
    fine_inc_img, _ = accumulate(fine_pts, inc_fs)
    coarse_inc_img, _ = accumulate(coarse_pts, inc_cs)
    inc_err_img, _ = accumulate(fine_pts, inc_err)
    fine_temp_img, _ = accumulate(fine_pts, tmp_fs)
    coarse_temp_img, _ = accumulate(coarse_pts, tmp_cs)
    temp_err_img, _ = accumulate(fine_pts, tmp_err)
    fine_therm_img, _ = accumulate(fine_pts, thm_fs)
    coarse_therm_img, _ = accumulate(coarse_pts, thm_cs)
    therm_err_img, _ = accumulate(fine_pts, thm_err)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        fine_img=fine_img, coarse_img=coarse_img, disp_img=disp_img,
        fine_pha_img=fine_pha_img, coarse_pha_img=coarse_pha_img,
        pha_err_img=pha_err_img,
        fine_inc_img=fine_inc_img, coarse_inc_img=coarse_inc_img,
        inc_err_img=inc_err_img,
        fine_temp_img=fine_temp_img, coarse_temp_img=coarse_temp_img,
        temp_err_img=temp_err_img,
        fine_therm_img=fine_therm_img, coarse_therm_img=coarse_therm_img,
        therm_err_img=therm_err_img,
        fine_cnt=fine_cnt,
        extent=np.array([lo[0], hi[0], lo[1], hi[1]]),
        res_m=res, n_frames=nfr,
        disp_stats=np.array([np.nanmean(disps), np.percentile(disps, 95),
                             np.nanmax(disps)]),
        pha_stats=np.array([np.nanmean(pha_err), np.percentile(pha_err, 95),
                            np.nanmax(pha_err)]),
        inc_stats=np.array([np.nanmean(inc_err), np.percentile(inc_err, 95),
                            np.nanmax(inc_err)]),
        temp_stats=np.array([np.nanmean(tmp_err),
                             np.percentile(tmp_err, 95),
                             np.nanmax(tmp_err)]),
        therm_stats=np.array([np.nanmean(thm_err),
                              np.percentile(thm_err, 95),
                              np.nanmax(thm_err)]))
    print(f"frames={nfr} px={rads.size:,} grid={W}x{H} res={res} m")
    print(f"displacement (m): mean={disps.mean():.2f} "
          f"p95={np.percentile(disps,95):.2f} max={disps.max():.2f}")
    print(f"phase err (deg): mean={pha_err.mean():.3f} "
          f"p95={np.percentile(pha_err,95):.3f} max={pha_err.max():.3f}")
    print(f"incidence err (deg): mean={inc_err.mean():.3f} "
          f"p95={np.percentile(inc_err,95):.3f} max={inc_err.max():.3f}")
    print(f"temp err (K): mean={tmp_err.mean():.2f} "
          f"p95={np.percentile(tmp_err,95):.2f} max={tmp_err.max():.2f}")
    print(f"EPF thermal err (W/m2/sr): mean={thm_err.mean():.3f} "
          f"p95={np.percentile(thm_err,95):.3f} max={thm_err.max():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
