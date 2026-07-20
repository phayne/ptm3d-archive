"""Thin wrappers around PTM3D binary I/O for the JWST comparison toolkit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse the canonical mesh/timestep loaders from src/python.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "python"))
import extract_timeseries as _ets  # noqa: E402


def load_mesh(run_dir: Path | str):
    """Return (face_lat, face_lon) arrays in degrees from <run_dir>/mesh.ply."""
    run_dir = Path(run_dir)
    _, _, _, lat, lon = _ets.read_binary_mesh_flexible(str(run_dir / "mesh.ply"))
    return lat, lon


def load_per_face_field(run_dir: Path | str, rel_path: str, num_faces: int) -> np.ndarray:
    """Load a single per-face binary file (e.g. statistics/temperature_max.bin)."""
    run_dir = Path(run_dir)
    arr = _ets.read_binary_timestep(str(run_dir / rel_path), num_faces)
    if arr is None:
        raise FileNotFoundError(f"Could not read {run_dir / rel_path}")
    return arr


def load_temperature_timeseries(run_dir: Path | str,
                                num_faces: int,
                                subdir: str = "temperature") -> tuple[np.ndarray, np.ndarray]:
    """Load all per-step temperature files into (Nt, Nfaces) array.

    Returns (times_s, temperatures_t) where times_s comes from time_steps.txt.
    """
    run_dir = Path(run_dir)
    files = sorted((run_dir / subdir).glob("temperature_*.bin"))
    if not files:
        raise FileNotFoundError(f"No temperature_*.bin files in {run_dir/subdir}")

    Nt = len(files)
    Tt = np.zeros((Nt, num_faces), dtype=np.float32)
    for i, f in enumerate(files):
        arr = _ets.read_binary_timestep(str(f), num_faces)
        if arr is None:
            raise IOError(f"Failed to read {f}")
        Tt[i] = arr.astype(np.float32)

    times_s = _read_time_steps(run_dir / "time_steps.txt", Nt)
    return times_s, Tt


def _read_time_steps(path: Path, expected_n: int) -> np.ndarray:
    """time_steps.txt: comment lines start with '#'; data is whitespace-separated
    with the elapsed-time-in-seconds in column 0 (ET seconds; absolute reference).
    We rebase to start at zero."""
    if not path.exists():
        return np.linspace(0.0, 3.5512 * 86400.0, expected_n, endpoint=False)
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(float(line.split()[0]))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return np.linspace(0.0, 3.5512 * 86400.0, expected_n, endpoint=False)
    return arr - arr[0]  # rebase to t=0 at first step
