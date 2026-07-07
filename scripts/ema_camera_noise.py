#!/usr/bin/env python3
"""EMA camera noise post-processor (DEV PHASE 2, tool #1).

Adds instrument noise to CLEAN rendered camera FITS (a single file or a flyby dir
of frame_*.fits) and, for a dir, rebuilds a noisy PNG snapshot + MP4 using the
EXISTING driver assembler helpers (make_mp4 / make_mp4_reflected / snapshot_png*)
-- reused, not reinvented. STANDALONE: it imports ema_campaign_driver READ-ONLY
(the module's __main__ guard means importing runs no campaign code, so it does not
race Rocky's running driver) purely to reuse those helpers + the Planck constants.

NOISE MODELS
------------
IR (ECAM IR3S, THERMAL HDU = spectral radiance eps*B_lambda(T), W/m^2/sr/um @10.5um):
  Spec: NEDT = 5 K at 180 K. Modeled as a microbolometer-like CONSTANT radiance
  noise anchored at the reference temperature:
      sigma_L = NEDT * eps * dB_lambda/dT |_{T=180K}
  added (Gaussian) to every on-body pixel's THERMAL radiance. Because sigma_L is
  constant while dB/dT rises with T, the EFFECTIVE per-pixel temperature noise
      sigma_T(T) = sigma_L / (eps * dB_lambda/dT|_T)      (= 5 K exactly at 180 K)
  is smaller for hotter pixels and larger for colder ones -- the physically
  consistent behavior of a detector/background-noise-limited thermal imager.
  ASSUMPTION for Fermi: constant radiance NEP (no explicit read-floor/photon term,
  no per-band integration). Defensible first-order model; a fuller model would set
  sigma_L from the actual detector NEP at 10.5 um.

Visible (CNAC, RADIANCE HDU = reflected radiance, W/m^2/sr):
  Spec: SNR = 10 at L_ref = 1 W/m^2/sr. Modeled as shot-noise-like, signal-dependent:
      sigma(L) = sqrt(L * L_ref) / SNR      (= L/SNR at L_ref, and sigma ~ sqrt(L))
  the sqrt(L) scaling is the Poisson/photon-shot signature. ASSUMPTION for Fermi:
  pure shot noise anchored at the spec point; no additive read-noise floor and no
  radiance->electrons conversion via the real CNAC QE/gain/integration time.

Deterministic + idempotent: per-frame RNG is seeded by (--seed, frame index) so a
given (seed, frame) always yields the same realization; existing noisy outputs are
skipped unless --overwrite.
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
from astropy.io import fits

PTM = "/Users/paha3326/Library/CloudStorage/OneDrive-UCB-O365/europa/heat3d/ptm"

# Reuse the driver's assembler/snapshot helpers + Planck constants, READ-ONLY.
_spec = importlib.util.spec_from_file_location("ema_driver", "/tmp/ema_sweep_driver.py")
# Rocky (2026-07-03): the TRACKED scripts/ema_campaign_driver.py on
# feat/generic-body-thermo-fix has NO `if __name__=='__main__':` guard -- its
# top-level code runs unconditionally on import and reads sys.argv of the
# IMPORTING process to select targets (`if len(sys.argv)>1: sel=set(sys.argv[1:])`).
# Importing it from here would risk silently kicking off a REAL campaign run
# using THIS script's argv. The /tmp scratch driver IS properly guarded
# (`if __name__=='__main__': main(...)`), so import it here instead -- inert.
_d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_d)   # no main() -> guarded
EPS, LAM, HP, CC, KB = _d.EPS, _d.LAM, _d.HP, _d.CC, _d.KB

# ---- noise-model spec constants ----
NEDT      = 5.0     # K   IR noise-equivalent differential temperature ...
NEDT_TREF = 180.0   # K   ... specified at this scene temperature
VIS_SNR   = 10.0    #     visible signal-to-noise ...
VIS_LREF  = 1.0     # W/m^2/sr  ... at this reference radiance

def planck_lambda(T):
    """B_lambda(T) [W/m^2/sr/um] at LAM, in the SAME convention as the driver's
    invplanck (so invplanck(planck_lambda(T)) == T)."""
    a = 2*HP*CC**2/LAM**5
    x = HP*CC/(LAM*KB*np.asarray(T, float))
    return a/np.expm1(x)/1e6

def dplanck_dT(T, dT=1e-3):
    """d B_lambda/dT via a centered difference (units of THERMAL HDU per K)."""
    return (planck_lambda(np.asarray(T,float)+dT) - planck_lambda(np.asarray(T,float)-dT))/(2*dT)

def invplanck(L):
    """Brightness temperature [K] from spectral radiance L [THERMAL-HDU units,
    W/m^2/sr/um @ LAM] -- the exact inverse of planck_lambda
    (``invplanck(planck_lambda(T)) == T``). This is the SHARED reconstruction
    kernel reused by ema_pretty_figs._noisy_ir_temp_field so the two tools cannot
    drift. NONLINEAR/monotonic and physically bounded: a floored (>=0) radiance
    maps to a finite very-cold temperature, so -- unlike the local LINEARIZED
    inversion -- it does not blow up on cold/low-signal pixels where
    dPlanck/dT -> 0."""
    L = np.clip(np.asarray(L, float), 1e-12, None)
    a = 2*HP*CC**2/LAM**5
    return HP*CC/(LAM*KB*np.log(a/(L*1e6) + 1.0))

def ir_sigma_L():
    """Constant IR radiance-noise sigma [THERMAL-HDU units] from NEDT@Tref."""
    return NEDT * EPS * dplanck_dT(NEDT_TREF)

def ir_sigma_T(T):
    """Effective per-pixel temperature noise at scene temperature T (= NEDT at Tref)."""
    T = np.asarray(T, float)
    return ir_sigma_L() / (EPS * dplanck_dT(np.clip(T, 1.0, None)))

def vis_sigma_L(L):
    """Shot-noise-like visible sigma at radiance L: sqrt(L*Lref)/SNR."""
    return np.sqrt(np.clip(np.asarray(L, float), 0.0, None) * VIS_LREF) / VIS_SNR

def _mask_on_body(hdul, ref_name):
    """On-body pixel mask: FACE_IDX>=0 if present, else the reference HDU>0."""
    names = [h.name for h in hdul]
    if 'FACE_IDX' in names:
        return np.asarray(hdul['FACE_IDX'].data) >= 0
    return np.asarray(hdul[ref_name].data, float) > 0

def add_ir_noise(thermal, mask, rng):
    out = np.asarray(thermal, float).copy()
    out[mask] += rng.normal(0.0, ir_sigma_L(), size=int(mask.sum()))
    out[out < 0] = 0.0
    return out

def recompute_temp_k(tempk_clean, thermal_clean, thermal_noisy, mask):
    """Recompute a TEMP_K field consistent with an ALREADY-noised (and already
    radiance-clipped) THERMAL field, so the two HDUs stay ONE measurement sharing
    ONE noise realization (do NOT draw a second, independent noise realization for
    TEMP_K). Uses the exact-at-zero-noise additive-calibration reconstruction --
    the SAME kernel ema_pretty_figs._noisy_ir_temp_field uses:

        recon_clean(px) = invplanck(THERMAL_clean(px) / EPS)
        correction(px)  = TEMP_K_clean(px) - recon_clean(px)   # deterministic
        TEMP_K_noisy(px) = invplanck(THERMAL_noisy(px) / EPS) + correction(px)

    The per-pixel additive ``correction`` reconciles this approximate
    monochromatic-Planck inversion with whatever exact bandpass/emissivity
    convention ptm_camera_sim used to write TEMP_K, so the reconstruction is EXACT
    in the zero-noise limit (correction is a fixed per-pixel constant). Only
    ``mask`` (on-body) pixels are recomputed; off-body pixels are returned
    unchanged from ``tempk_clean``."""
    out = np.asarray(tempk_clean, float).copy()
    recon_clean = invplanck(np.asarray(thermal_clean, float)[mask] / EPS)
    correction  = np.asarray(tempk_clean, float)[mask] - recon_clean
    out[mask]   = invplanck(np.asarray(thermal_noisy, float)[mask] / EPS) + correction
    return out

def add_vis_noise(radiance, mask, rng):
    out = np.asarray(radiance, float).copy()
    out[mask] += rng.standard_normal(int(mask.sum())) * vis_sigma_L(out[mask])
    out[out < 0] = 0.0
    return out

def process_fits(inpath, outpath, mode, rng):
    """Write a noisy copy of one FITS. IR noises THERMAL; vis noises RADIANCE; both
    noises whichever are present. Other HDUs/headers are preserved verbatim."""
    hdul = fits.open(inpath)
    names = [h.name for h in hdul]
    if mode in ('ir', 'both') and 'THERMAL' in names:
        m = _mask_on_body(hdul, 'THERMAL')
        thermal_clean = np.asarray(hdul['THERMAL'].data, float).copy()
        thermal_noisy = add_ir_noise(hdul['THERMAL'].data, m, rng).astype(np.float32)
        hdul['THERMAL'].data = thermal_noisy
        # TEMP_K and THERMAL are the SAME measurement: recompute TEMP_K from the
        # noised + already-clipped THERMAL now in hand (one shared noise
        # realization, NOT a second independent draw), so the cold floor stays
        # physical and the two HDUs stay consistent. On-body pixels only --
        # off-body TEMP_K bytes are left untouched. Reuses the shared
        # ema_camera_noise kernel that ema_pretty_figs._noisy_ir_temp_field also
        # calls, so the noiser and the figure/movie path cannot drift.
        if 'TEMP_K' in names:
            tempk_orig = hdul['TEMP_K'].data
            new_tempk  = np.array(tempk_orig, copy=True)
            recomputed = recompute_temp_k(tempk_orig, thermal_clean, thermal_noisy, m)
            new_tempk[m] = recomputed[m].astype(tempk_orig.dtype)
            hdul['TEMP_K'].data = new_tempk
    if mode in ('vis', 'both') and 'RADIANCE' in names:
        m = _mask_on_body(hdul, 'RADIANCE')
        hdul['RADIANCE'].data = add_vis_noise(hdul['RADIANCE'].data, m, rng).astype(np.float32)
    hdul[0].header['NOISED']  = (True, 'instrument noise added by ema_camera_noise')
    hdul[0].header['NOISEMOD'] = (mode, 'noise channels: ir|vis|both')
    hdul.writeto(outpath, overwrite=True)
    hdul.close()

def process_path(inpath, outdir, mode, seed, overwrite, do_png, do_mp4, title):
    """Process a single FITS or a dir of frame_*.fits."""
    if os.path.isfile(inpath):
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, os.path.splitext(os.path.basename(inpath))[0] + "_noisy.fits")
        if os.path.exists(outp) and not overwrite:
            print("skip (exists):", outp); return [outp]
        process_fits(inpath, outp, mode, np.random.default_rng([seed, 0]))
        print("wrote:", outp)
        return [outp]
    frames = sorted(glob.glob(os.path.join(inpath, "frame_*.fits")))
    if not frames:
        raise SystemExit(f"no frame_*.fits in {inpath}")
    os.makedirs(outdir, exist_ok=True)
    out = []
    for i, fp in enumerate(frames):
        outp = os.path.join(outdir, os.path.basename(fp))
        if os.path.exists(outp) and not overwrite:
            out.append(outp); continue
        process_fits(fp, outp, mode, np.random.default_rng([seed, i]))
        out.append(outp)
    print(f"wrote {len(out)} noisy frames -> {outdir}")
    n = len(out)
    if do_png and n >= 3:
        if mode in ('ir', 'both'):
            _d.snapshot_png(outdir, os.path.join(outdir, "noisy_ir_snapshots.png"), title+" [NOISY IR]")
        if mode in ('vis', 'both'):
            _d.snapshot_png_reflected(outdir, os.path.join(outdir, "noisy_visible_snapshots.png"), title+" [NOISY vis]")
    if do_mp4 and n >= 2:
        if mode in ('ir', 'both'):
            _d.make_mp4(outdir, os.path.join(outdir, "noisy_ir.mp4"), title+" [NOISY IR]")
        if mode in ('vis', 'both'):
            _d.make_mp4_reflected(outdir, os.path.join(outdir, "noisy_visible.mp4"), title+" [NOISY vis]")
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description="EMA camera noise post-processor (IR NEDT / vis SNR)")
    ap.add_argument("--input", required=True, help="clean camera FITS file OR a dir of frame_*.fits")
    ap.add_argument("--mode", choices=["ir", "vis", "both"], required=True)
    ap.add_argument("--output-dir", default=None, help="default: <dir>/noisy for a dir, or alongside for a file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--title", default="EMA noisy render")
    a = ap.parse_args(argv)
    if a.output_dir is None:
        a.output_dir = (os.path.join(a.input, "noisy") if os.path.isdir(a.input)
                        else os.path.dirname(os.path.abspath(a.input)) or ".")
    print(f"IR: sigma_L={ir_sigma_L():.6g} (THERMAL units); sigma_T(180K)={float(ir_sigma_T(180.0)):.3f}K "
          f"sigma_T(250K)={float(ir_sigma_T(250.0)):.3f}K | vis: sigma(1)={float(vis_sigma_L(1.0)):.4f} "
          f"sigma(4)={float(vis_sigma_L(4.0)):.4f} W/m^2/sr")
    process_path(a.input, a.output_dir, a.mode, a.seed, a.overwrite,
                 not a.no_png, not a.no_mp4, a.title)

if __name__ == "__main__":
    main()
