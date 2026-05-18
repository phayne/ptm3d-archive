#!/usr/bin/env python3
"""
F1 — Lunar surface-temperature validation (two-panel).

  (a) Equatorial diurnal surface-temperature cycle, PTM3D vs Hayne (2017)
      reference points (peak / midnight / minimum)
  (b) Post-sunset cooling at 0°, 30°, 60° latitude with Hayne (2017) Fig. A2
      points overlaid

Run after `bin/ptm_thermal_sim --config examples/lunar_validation.yaml`.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ptm_pubstyle import apply_style, COLORS, DOUBLE_W, LATITUDE_COLORS, panel_label


LUNAR_DAY_S = 29.530588 * 86400.0


# Hayne et al. (2017) reference temperatures, equator (Table A1, Fig A2)
HAYNE_REF = {
    "peak":     (12.0, 385.0),
    "midnight": (0.0, 101.0),
    "minimum":  (5.0, 95.0),
}

# Hayne et al. (2017) Fig. A2 nighttime cooling, three latitudes
HAYNE_NIGHT_HOURS = np.array([20, 22, 24, 26, 28])  # 24+ for plotting continuity
HAYNE_NIGHT = {
    0:  np.array([109.0, 104.0, 100.0,  97.0, 94.5]),
    30: np.array([106.5, 101.0,  97.0,  94.0, 91.0]),
    60: np.array([ 95.0,  90.0,  87.0,  84.0, 82.0]),
}


def read_binary_data(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data[:n] if data.size > n else data


def read_mesh_lats_lons(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        is_binary = any("binary" in h for h in header)
        n_v = int(next(h for h in header if h.startswith("element vertex")).split()[-1])
        n_f = int(next(h for h in header if h.startswith("element face")).split()[-1])
        if not is_binary:
            raise NotImplementedError("ASCII PLY not supported in this loader")
        vbytes = n_v * 6 * 8
        _ = np.frombuffer(f.read(vbytes), dtype="<f8").reshape(n_v, 6)[:, :3]
        lats = np.empty(n_f, dtype=np.float32)
        lons = np.empty(n_f, dtype=np.float32)
        for i in range(n_f):
            count = struct.unpack("<B", f.read(1))[0]
            f.read(4 * count)
            lat, lon = struct.unpack("<ff", f.read(8))
            lats[i] = lat
            lons[i] = lon
    return lats, lons


def load_run(run_dir: Path) -> dict:
    meta = {}
    with open(run_dir / "metadata.txt") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k.strip()] = v.strip()
    n_faces = int(meta.get("faces", 0))

    times = []
    with open(run_dir / "time_steps.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            times.append(float(line.split()[0]))
    times = np.array(times)
    n_steps = len(times)

    lats, lons = read_mesh_lats_lons(run_dir / "mesh.ply")
    if len(lats) != n_faces:
        n_faces = len(lats)

    print(f"  loading {n_steps} temperature snapshots × {n_faces} faces...")
    T = np.empty((n_steps, n_faces), dtype=np.float32)
    temp_dir = run_dir / "temperature"
    for t in range(n_steps):
        candidate = temp_dir / f"temperature_{t:04d}.bin"
        if not candidate.exists():
            candidate = temp_dir / f"{t:06d}.bin"
        T[t] = read_binary_data(candidate)
        if (t + 1) % 60 == 0:
            print(f"    {t + 1}/{n_steps}")
    return {"times": times, "T": T, "lats": lats, "lons": lons}


def local_hour_for_face(times: np.ndarray, T_face: np.ndarray) -> np.ndarray:
    peak = times[np.argmax(T_face)]
    return ((times - peak) / LUNAR_DAY_S * 24.0 + 12.0) % 24.0


def diurnal_cycle(times, T, face_idxs, n_bins=48):
    edges = np.linspace(0, 24, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    samples = [[] for _ in range(n_bins)]
    for f in face_idxs:
        h = local_hour_for_face(times, T[:, f])
        b = np.minimum((h / 24.0 * n_bins).astype(int), n_bins - 1)
        for bi, ti in zip(b, T[:, f]):
            samples[bi].append(ti)
    mean = np.array([np.mean(s) if s else np.nan for s in samples])
    return centers, mean


def select_band(lats, center_deg, half_width=5.0):
    return np.where(np.abs(lats - center_deg) < half_width)[0]


def make_figure(data: dict, out_path: Path) -> None:
    apply_style()

    eq_idx  = select_band(data["lats"],  0.0, 5.0)
    l30_idx = select_band(data["lats"], 30.0, 5.0)
    l60_idx = select_band(data["lats"], 60.0, 5.0)

    h_eq, T_eq = diurnal_cycle(data["times"], data["T"], eq_idx)
    h_30, T_30 = diurnal_cycle(data["times"], data["T"], l30_idx)
    h_60, T_60 = diurnal_cycle(data["times"], data["T"], l60_idx)

    # Nighttime view: rotate 0..5h to 24..29h for continuity. Cut off at 05:00
    # to avoid the sunrise transition at 06:00 dragging the y-axis up.
    def night_view(h, T):
        h2 = np.where(h < 12, h + 24, h)
        keep = (h2 >= 18) & (h2 <= 29)
        h2 = h2[keep]
        T = T[keep]
        order = np.argsort(h2)
        return h2[order], T[order]

    h_eq_n,  T_eq_n  = night_view(h_eq, T_eq)
    h_30_n,  T_30_n  = night_view(h_30, T_30)
    h_60_n,  T_60_n  = night_view(h_60, T_60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W * 1.4, 0.55 * DOUBLE_W * 1.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.14, wspace=0.22)

    # ---- (a) equatorial diurnal cycle
    ax1.plot(h_eq, T_eq, color=COLORS["ptm"], lw=2.2, label="PTM3D")
    for key, (lh, T) in HAYNE_REF.items():
        ax1.plot(lh, T, "*", mfc="none", mec=COLORS["ref"], mew=1.6, ms=18, zorder=5)
    ax1.plot([], [], "*", mfc="none", mec=COLORS["ref"], mew=1.6, ms=14,
             label="Hayne 2017")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 6))
    ax1.set_ylim(70, 410)
    ax1.set_xlabel("Local solar time (h)")
    ax1.set_ylabel(r"Surface temperature $T_0$ (K)")
    ax1.legend(loc="upper right")
    panel_label(ax1, "(a)", x=0.04, y=0.95)

    # ---- (b) nighttime cooling at three latitudes
    for lat, h, T, marker in [
        (0,  h_eq_n,  T_eq_n,  "o"),
        (30, h_30_n,  T_30_n,  "s"),
        (60, h_60_n,  T_60_n,  "^"),
    ]:
        c = LATITUDE_COLORS[lat]
        ax2.plot(h, T, color=c, lw=2.2, label=fr"PTM3D {lat}$^\circ$")
        ax2.plot(HAYNE_NIGHT_HOURS, HAYNE_NIGHT[lat], marker,
                 mfc="none", mec=c, mew=1.6, ms=10,
                 label=fr"Hayne {lat}$^\circ$")
    ax2.set_xlim(18, 29)
    ax2.set_xticks([18, 20, 22, 24, 26, 28])
    ax2.set_xticklabels(["18", "20", "22", "00", "02", "04"])
    ax2.set_ylim(75, 130)
    ax2.set_xlabel("Local solar time (h)")
    ax2.set_ylabel(r"Surface temperature $T_0$ (K)")
    ax2.legend(loc="upper right", ncol=2)
    panel_label(ax2, "(b)", x=0.04, y=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")

    # Print residuals for the docstring
    def interp_at(hours, T, target_hour):
        h2 = np.where(hours < 12, hours + 24, hours)
        order = np.argsort(h2)
        return float(np.interp(target_hour, h2[order], T[order]))

    rms = []
    for lat, h, T in [(0, h_eq, T_eq), (30, h_30, T_30), (60, h_60, T_60)]:
        for h_ref, t_h in zip(HAYNE_NIGHT_HOURS, HAYNE_NIGHT[lat]):
            rms.append(interp_at(h, T, h_ref) - t_h)
    rms = float(np.sqrt(np.mean(np.array(rms) ** 2)))
    print(f"  Hayne 15-pt nighttime RMS: {rms:.2f} K")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default="output/lunar_validation")
    p.add_argument("--out", default="docs/figures/v2/lunar_validation.png")
    args = p.parse_args()
    print(f"Reading {args.run_dir}")
    data = load_run(Path(args.run_dir))
    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
