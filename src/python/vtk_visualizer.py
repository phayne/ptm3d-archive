#!/usr/bin/env python3
"""
Render a PTM simulation time series (simulation.pvd) to a movie using PyVista.

- Reads the PVD to discover per-timestep VTU files
- Renders a chosen cell scalar (e.g., illumination, temperature)
- Programmable camera and lighting
- Annotates each frame with UTC time (from VTU FieldData "UTC" or from epoch in PVD)
- Writes an .mp4 (or .avi) movie with a scalar colorbar

Usage:
  python render_pvd_movie.py --pvd /path/to/simulation.pvd --output movie.mp4 --field temperature \
    --camera-position 1.0,2.0,3.0 --focal-point 0,0,0 --view-up 0,0,1 --light headlight --framerate 24
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from xml.etree import ElementTree as ET
from datetime import datetime, timezone
import re
import io

import pyvista as pv
import numpy as np
from packaging import version

def parse_vec3(s: Optional[str], default=(None, None, None)) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not s:
        return default
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {s}")
    return float(parts[0]), float(parts[1]), float(parts[2])

def parse_clim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 2:
        raise ValueError(f"--clim expects 'min,max', got: {s}")
    vmin, vmax = float(parts[0]), float(parts[1])
    if vmax <= vmin:
        raise ValueError("clim max must be > min")
    return (vmin, vmax)

def utc_from_epoch(epoch: float, fmt: str) -> str:
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime(fmt)
    except Exception:
        return ""

def read_time_steps_file(ts_path: Path) -> Tuple[List[float], List[str]]:
    """Read PTM3D time_steps.txt -> (elapsed_seconds, utc_strings)."""
    times: List[float] = []
    tstrs: List[str] = []
    try:
        with open(ts_path, 'r') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                    except Exception:
                        times.append(0.0)
                    tstrs.append(parts[1])
    except Exception:
        pass
    return times, tstrs

def _has_trailing_after_vtkfile(path: Path) -> bool:
    """Return True if there is any non-whitespace data after </VTKFile>."""
    try:
        size = os.path.getsize(path)
        if size <= 0:
            return False
        # Read last chunk only
        tail_len = min(65536, size)
        with open(path, 'rb') as f:
            f.seek(size - tail_len)
            buf = f.read(tail_len)
        end_tag = b'</VTKFile>'
        pos = buf.rfind(end_tag)
        if pos == -1:
            # No end tag in tail; might still be valid earlier, but treat as suspicious
            return True
        after = buf[pos + len(end_tag):]
        # Allow only whitespace/newlines
        return any(ch not in b' \t\r\n' for ch in after)
    except Exception:
        return False

def validate_vtu_header(path: Path) -> bool:
    """Quick check for header and no obvious trailing junk."""
    try:
        with open(path, 'rb') as f:
            head = f.read(64)
        if not head.startswith(b'<?xml'):
            return False
        # If trailing garbage exists, mark invalid
        if _has_trailing_after_vtkfile(path):
            return False
        return os.path.getsize(path) > 256
    except Exception:
        return False

def repair_vtu_trailing_garbage(path: Path) -> bool:
    """
    Truncate any trailing garbage after the final </VTKFile>.
    Returns True if file looks valid after fix, False otherwise.
    """
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if not data:
            return False
        end_tag = b'</VTKFile>'
        pos = data.rfind(end_tag)
        if pos == -1:
            return False
        new_len = pos + len(end_tag)
        if new_len < len(data):
            tmp = path.with_suffix(path.suffix + ".fix")
            with open(tmp, 'wb') as out:
                out.write(data[:new_len])
            os.replace(tmp, path)
        return validate_vtu_header(path)
    except Exception:
        return False

def _grid_looks_valid(g: pv.UnstructuredGrid) -> bool:
    """Heuristic to detect usable grids."""
    try:
        if g is None:
            return False
        if g.n_cells <= 0 or g.n_points <= 0:
            return False
        # Some corrupted reads return empty data
        if len(g.cell_data.keys()) == 0 and len(g.point_data.keys()) == 0:
            return False
        return True
    except Exception:
        return False

def try_read_grid(fp: Path) -> Optional[pv.UnstructuredGrid]:
    """
    Try to read a VTU with PyVista. If parse fails or grid is empty,
    attempt to repair trailing garbage and re-read.
    """
    # First attempt
    try:
        g = pv.read(str(fp))
        if _grid_looks_valid(g):
            return g
    except Exception:
        g = None

    # If we get here, try to repair then re-read
    if repair_vtu_trailing_garbage(fp):
        try:
            g2 = pv.read(str(fp))
            if _grid_looks_valid(g2):
                return g2
        except Exception:
            return None
    return None

def resolve_field_or_fallback(grid: pv.UnstructuredGrid, requested: str) -> str:
    """
    If requested field is missing, try 'illumination'; else first available cell array; else raise.
    """
    names = list(grid.cell_data.keys())
    if requested in grid.cell_data:
        return requested
    if "illumination" in grid.cell_data:
        print(f"Warning: Field '{requested}' not found. Using 'illumination' instead.")
        return "illumination"
    if names:
        print(f"Warning: Field '{requested}' not found. Using '{names[0]}' instead.")
        return names[0]
    raise KeyError(f"Field '{requested}' not found and no cell_data available in the VTU.")

def compute_global_range(files: List[Path], field: str) -> Tuple[float, float]:
    vmin, vmax = np.inf, -np.inf
    for fp in files:
        g = try_read_grid(fp)
        if g is None or field not in g.cell_data:
            if g is not None:
                g.clear_data()
            continue
        arr = np.asarray(g.cell_data[field])
        if arr.size == 0:
            g.clear_data()
            continue
        vmin = min(vmin, float(np.nanmin(arr)))
        vmax = max(vmax, float(np.nanmax(arr)))
        g.clear_data()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise RuntimeError("Failed to compute global range; use --clim or omit --global-range.")
    return (vmin, vmax)

def parse_pvd_with_index(pvd_path: Path) -> List[Tuple[Path, float, Optional[int]]]:
    """
    Parse PVD and return list of (vtu_path, timestep_value, time_index) in file order.
    time_index is inferred from filename pattern '*_XXXX.vtu' if present.
    """
    tree = ET.parse(str(pvd_path))
    root = tree.getroot()
    coll = root.find('Collection') or root.find('{*}Collection')
    datasets = []
    if coll is None:
        return datasets
    for ds in coll.findall('DataSet') + coll.findall('{*}DataSet'):
        file_attr = ds.attrib.get('file') or ds.attrib.get('source')
        t_attr = ds.attrib.get('timestep')
        if not file_attr:
            continue
        vtu_path = (pvd_path.parent / file_attr).resolve()
        try:
            tval = float(t_attr) if t_attr is not None else 0.0
        except Exception:
            tval = 0.0
        # infer time index from filename
        m = re.search(r'_(\d{4})\.vtu$', Path(file_attr).name)
        tidx = int(m.group(1)) if m else None
        datasets.append((vtu_path, tval, tidx))
    # Keep all entries; we’ll repair on demand
    return datasets

def get_utc_from_grid_or_map(grid: pv.UnstructuredGrid,
                             fallback_epoch: Optional[float],
                             fmt: str,
                             idx_to_utc: Optional[Dict[int, str]] = None,
                             time_index: Optional[int] = None) -> str:
    """
    Priority:
      1) FieldData 'UTC' (string) in VTU
      2) FieldData 'TimeValue' (epoch seconds) in VTU
      3) time_steps.txt mapping (by time_index)
      4) fallback_epoch (from PVD timestep)
    """
    # 1) VTU FieldData string
    try:
        arr = grid.field_data.GetAbstractArray("UTC")
        if arr is not None and arr.GetNumberOfValues() > 0:
            return arr.GetValue(0)
    except Exception:
        pass
    # 2) VTU FieldData numeric epoch
    try:
        arr_num = grid.field_data.get("TimeValue", None)
        if arr_num is not None:
            if np.size(arr_num) > 0:
                return utc_from_epoch(float(np.ravel(arr_num)[0]), fmt)
    except Exception:
        pass
    # 3) time_steps.txt mapping
    if idx_to_utc is not None and time_index is not None:
        s = idx_to_utc.get(int(time_index))
        if isinstance(s, str) and s:
            return s
    # 4) Fallback to PVD epoch/elapsed
    if fallback_epoch is not None:
        return utc_from_epoch(fallback_epoch, fmt)
    return ""

def parse_pvd(pvd_path: Path) -> List[Tuple[Path, float]]:
    """
    Return list of (vtu_path, timestep_value) in order.
    """
    tree = ET.parse(str(pvd_path))
    root = tree.getroot()
    coll = root.find('Collection')
    if coll is None:
        coll = root.find('{*}Collection')
    datasets = []
    for ds in coll.findall('DataSet') + coll.findall('{*}DataSet'):
        file_attr = ds.attrib.get('file') or ds.attrib.get('source')  # some writers use 'source'
        t_attr = ds.attrib.get('timestep')
        if file_attr is None or t_attr is None:
            continue
        vtu_path = (pvd_path.parent / file_attr).resolve()
        try:
            tval = float(t_attr)
        except Exception:
            tval = 0.0
        datasets.append((vtu_path, tval))
    return datasets

def ensure_field(grid: pv.UnstructuredGrid, field: str) -> None:
    if field not in grid.cell_data:
        names = list(grid.cell_data.keys())
        raise KeyError(f"Field '{field}' not found in cell_data. Available: {names}")

def add_configured_light(plotter: pv.Plotter, kind: str,
                         light_pos: Optional[Tuple[float,float,float]],
                         focal_point: Optional[Tuple[float,float,float]],
                         intensity: float) -> None:
    # Remove defaults to avoid double lighting unless user wants both
    try:
        plotter.remove_all_lights()
    except Exception:
        plotter.lights = []

    kind = (kind or "headlight").lower()
    if kind == "headlight":
        try:
            light = pv.Light(light_type="headlight")  # PyVista >= 0.41
        except Exception:
            light = pv.Light()
            light.set_headlight()
        light.intensity = float(intensity)
        plotter.add_light(light)
    else:
        # Positional light
        lp = light_pos if (light_pos and None not in light_pos) else (10.0, 10.0, 10.0)
        fp = focal_point if (focal_point and None not in focal_point) else (0.0, 0.0, 0.0)
        light = pv.Light(position=lp, focal_point=fp, color='white', intensity=float(intensity))
        light.positional = True
        light.cone_angle = 60.0
        plotter.add_light(light)

def main():
    parser = argparse.ArgumentParser(description="Render a simulation.pvd time series to a movie using PyVista")
    parser.add_argument("--pvd", required=True, help="Path to simulation.pvd")
    parser.add_argument("--output", required=True, help="Output movie path (e.g., movie.mp4)")
    parser.add_argument("--field", required=True, help="Cell scalar name to render (e.g., illumination, temperature)")
    parser.add_argument("--clim", type=str, help="Color limits 'min,max' (optional)")
    parser.add_argument("--global-range", action="store_true", help="Compute color limits across all timesteps")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--framerate", type=int, default=24, help="Movie frame rate")
    parser.add_argument("--window-size", type=str, default="1920,1080", help="Render window size 'W,H'")
    parser.add_argument("--camera-position", type=str, help="Camera position 'x,y,z'")
    parser.add_argument("--focal-point", type=str, help="Camera focal point 'x,y,z'")
    parser.add_argument("--view-up", type=str, help="Camera view up 'x,y,z'")
    parser.add_argument("--light", type=str, default="headlight", help="Lighting: 'headlight' or 'positional'")
    parser.add_argument("--light-position", type=str, help="Positional light position 'x,y,z'")
    parser.add_argument("--light-intensity", type=float, default=1.0, help="Light intensity")
    parser.add_argument("--time-format", type=str, default="%Y-%m-%d %H:%M:%S UTC", help="UTC annotation strftime format")
    parser.add_argument("--font-size", type=int, default=16, help="UTC text font size")
    parser.add_argument("--transparent", action="store_true", help="Transparent background")
    parser.add_argument("--show-edges", action="store_true", help="Render triangle edges")
    parser.add_argument("--time-file", type=str, help="Optional path to time_steps.txt")
    args = parser.parse_args()

    pvd_path = Path(args.pvd).resolve()
    if not pvd_path.exists():
        print(f"ERROR: PVD not found: {pvd_path}")
        return 1

    datasets = parse_pvd_with_index(pvd_path)
    if not datasets:
        print(f"ERROR: No DataSet entries found in {pvd_path}")
        return 1

    files = [fp for fp, _, _ in datasets]
    times = [t for _, t, _ in datasets]
    indices = [idx for _, _, idx in datasets]

    # Optional time_steps.txt mapping (auto-detect next to PVD if not provided)
    ts_path = Path(args.time_file).resolve() if args.time_file else (pvd_path.parent / "time_steps.txt")
    idx_to_utc: Dict[int, str] = {}
    if ts_path.exists():
        ts_times, ts_strings = read_time_steps_file(ts_path)
        for i, s in enumerate(ts_strings):
            idx_to_utc[i] = s

    # Load first valid VTU as base (repair if needed)
    base = None
    base_idx = 0
    for i, fp in enumerate(files):
        g = try_read_grid(fp)
        if g is None:
            print(f"Error: Failed to read VTU '{fp.name}'. Skipping.")
            continue
        base = g
        base_idx = i
        break
    if base is None:
        print("ERROR: Could not read any VTU from PVD (all invalid or unreadable).")
        return 1

    # Resolve the field (with fallback) before continuing
    try:
        field_name = resolve_field_or_fallback(base, args.field)
    except KeyError as e:
        print(f"ERROR: {e}")
        return 1

    # Determine color limits
    clim = parse_clim(args.clim)
    if args.global_range and clim is None:
        print("Computing global color limits across all timesteps...")
        try:
            clim = compute_global_range(files, field_name)
            print(f"  Global range: [{clim[0]:.6g}, {clim[1]:.6g}]")
        except Exception as e:
            print(f"  Warning: failed to compute global range: {e}. Proceeding without fixed clim.")
            clim = None

    # Parse rendering options
    win_w, win_h = [int(v) for v in (args.window_size.split(',') if args.window_size else ["1920","1080"])]
    cam_pos = parse_vec3(args.camera_position, default=(None,None,None))
    foc_pt  = parse_vec3(args.focal_point, default=(None,None,None))
    view_up = parse_vec3(args.view_up, default=(None,None,None))
    light_pos = parse_vec3(args.light_position, default=(None,None,None))

    # Setup plotter
    pv.global_theme.colorbar_vertical = True
    plotter = pv.Plotter(off_screen=True, window_size=(win_w, win_h))
    plotter.set_background("transparent" if args.transparent else "black")
    add_configured_light(plotter, args.light, light_pos, foc_pt, args.light_intensity)

    # Add mesh and scalar bar using the resolved field
    actor = plotter.add_mesh(
        base,
        scalars=field_name,
        cmap=args.cmap,
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args={"title": field_name, "vertical": True},
        show_edges=args.show_edges,
        copy_mesh=True
    )

    # Camera
    if None not in cam_pos and None not in foc_pt and None not in view_up:
        plotter.camera_position = [tuple(cam_pos), tuple(foc_pt), tuple(view_up)]
    else:
        plotter.camera.Zoom(1.0)

    # Time annotation for first frame (use matching dataset index)
    utc0 = get_utc_from_grid_or_map(
        base,
        times[base_idx] if base_idx < len(times) else None,
        args.time_format,
        idx_to_utc=idx_to_utc,
        time_index=indices[base_idx] if indices and indices[base_idx] is not None else base_idx
    )
    text_actor = plotter.add_text(f"{utc0}", position="upper_left", font_size=args.font_size, color="white", shadow=True)

    # Movie writer
    plotter.open_movie(str(Path(args.output).resolve()), framerate=int(args.framerate))

    # Render first frame
    plotter.write_frame()

    # Iterate timesteps starting after base_idx
    for idx_iter in range(base_idx + 1, len(files)):
        fp = files[idx_iter]
        tval = times[idx_iter] if idx_iter < len(times) else None
        tidx = indices[idx_iter] if idx_iter < len(indices) else idx_iter

        g = try_read_grid(fp)
        if g is None:
            print(f"Warning: Failed to read VTU '{Path(fp).name}', skipping frame {idx_iter}")
            continue

        if field_name not in g.cell_data:
            # Try to resolve a new fallback field for this timestep
            try:
                new_field = resolve_field_or_fallback(g, field_name)
                if new_field != field_name:
                    field_name = new_field
                    # Update actor lookup table title
                    try:
                        actor.mapper.array_name = field_name
                    except Exception:
                        pass
                    try:
                        plotter.scalar_bar.SetTitle(field_name)
                    except Exception:
                        pass
            except KeyError:
                print(f"Warning: No cell data found in {Path(fp).name}, skipping frame {idx_iter}")
                g.clear_data()
                continue

        # Update scalars on the base geometry (assumes static mesh topology)
        base.cell_data[field_name] = np.asarray(g.cell_data[field_name])

        # Update color limits if not fixed
        if clim is None:
            arr = base.cell_data[field_name]
            finite = np.isfinite(arr)
            if np.any(finite):
                plotter.update_scalar_bar_range([float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))])

        # Update UTC annotation
        utc_txt = get_utc_from_grid_or_map(
            g,
            tval,
            args.time_format,
            idx_to_utc=idx_to_utc,
            time_index=(tidx if tidx is not None else idx_iter)
        )
        try:
            text_actor.SetInput(str(utc_txt))
        except Exception:
            plotter.remove_actor(text_actor)
            text_actor = plotter.add_text(f"{utc_txt}", position="upper_left", font_size=args.font_size, color="white", shadow=True)

        g.clear_data()
        plotter.write_frame()

    plotter.close()
    print(f"Movie written to: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())