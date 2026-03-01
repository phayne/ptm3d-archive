"""
Simulation Runner for Viewshed GUI.
Runs viewshed computation in a background QThread, calling pegasus_sim
functions directly (Python-native, no subprocess).
"""

import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from . import viewshed_data_loader as loader

SYNODIC_MONTH_DAYS = 29.53059


def _parse_utc(utc_str):
    """Parse PTM3D UTC string like '2030-09-01T01:28:08.225'."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(utc_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse UTC: {utc_str}")


def _ptm_time_range(ptm_timesteps):
    """Return (first_dt, last_dt) from the PTM timestep list."""
    if not ptm_timesteps:
        return None, None
    first_dt = _parse_utc(ptm_timesteps[0][1])
    last_dt = _parse_utc(ptm_timesteps[-1][1])
    return first_dt, last_dt


def _check_window_overlap(window_start, window_end, ptm_first, ptm_last):
    """Check if a time window overlaps with the PTM data range.

    Returns True if there is any overlap.
    """
    if ptm_first is None or ptm_last is None:
        return False
    return window_start <= ptm_last and window_end >= ptm_first


def _compute_time_shift(mission_start, mission_end, ptm_first, ptm_last):
    """Determine the appropriate time shift for mapping mission to PTM data.

    Returns:
        shift_days: Number of days to shift mission times (0 if direct overlap,
                    positive/negative multiples of synodic month otherwise).
        shift_applied: Boolean indicating whether a shift was applied.
    """
    if ptm_first is None or ptm_last is None:
        return 0.0, False

    # Check if mission window directly overlaps with PTM data
    if _check_window_overlap(mission_start, mission_end, ptm_first, ptm_last):
        return 0.0, False

    # No direct overlap - try shifting by synodic months
    # Find the shift that minimizes the gap
    best_shift = 0.0
    best_overlap = False

    for n in range(-12, 13):  # Check +/- 1 year of synodic months
        if n == 0:
            continue
        shift = n * SYNODIC_MONTH_DAYS
        shifted_start = mission_start - timedelta(days=shift)
        shifted_end = mission_end - timedelta(days=shift)

        if _check_window_overlap(shifted_start, shifted_end, ptm_first, ptm_last):
            # Found an overlapping shift - prefer smaller shifts
            if not best_overlap or abs(n) < abs(best_shift / SYNODIC_MONTH_DAYS):
                best_shift = shift
                best_overlap = True

    if best_overlap:
        return best_shift, True

    # No synodic shift gives overlap - use single synodic month as fallback
    return SYNODIC_MONTH_DAYS, True


def _map_mission_to_ptm_timestep(mission_time, ptm_timesteps, shift_days=0.0):
    """Map a mission datetime to the nearest PTM3D timestep index.

    Args:
        mission_time: The mission datetime to map.
        ptm_timesteps: List of (index, utc_string) tuples.
        shift_days: Number of days to shift the mission time (0 for direct mapping).

    Returns:
        (best_idx, best_diff_seconds) -- the index and the temporal offset
        in seconds between the (possibly shifted) mission time and the nearest PTM step.
    """
    shifted = mission_time - timedelta(days=shift_days)
    best_idx = 0
    best_diff = float("inf")
    for idx, (_, utc_str) in enumerate(ptm_timesteps):
        ptm_dt = _parse_utc(utc_str)
        diff = abs((shifted - ptm_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx, best_diff


class ViewshedSimulationThread(QThread):
    """Thread for running viewshed simulations without blocking GUI."""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished_signal = pyqtSignal(int)  # exit code

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._stop_requested = False

        # Results stored after run for access by visualization panel
        self.mesh = None
        self.waypoints = None
        self.waypoint_results = []
        self.best_ice_est = None
        self.best_snr = None
        self.ice_frac_truth = None
        self.earth_visibility_fraction = None
        self.summary_rows = []

    def stop(self):
        """Request the simulation to stop."""
        self._stop_requested = True

    def _emit(self, msg):
        self.output_ready.emit(msg)

    def _emit_err(self, msg):
        self.error_ready.emit(msg)

    def run(self):
        """Execute the viewshed simulation pipeline."""
        try:
            self._run_pipeline()
        except Exception as e:
            self._emit_err(f"Simulation error: {e}")
            self._emit_err(traceback.format_exc())
            self.finished_signal.emit(-1)

    def _run_pipeline(self):
        cfg = self.config_manager.get_config()
        inp = cfg["input"]
        mission = cfg["mission"]
        vs_cfg = cfg["viewshed"]
        ice_cfg = cfg["ice"]
        out_cfg = cfg["output"]
        mask_defs = cfg.get("masks", [])

        ptm_dir = Path(inp["ptm_output_dir"])
        mesh_path = Path(inp["mesh_file"]) if inp["mesh_file"] else ptm_dir / "mesh.ply"
        waypoints_path = Path(inp["waypoints_csv"])
        output_dir = Path(out_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure pegasus_sim is importable
        pegasus_root = Path(__file__).resolve().parent.parent.parent.parent / "pegasus-sim"
        if str(pegasus_root) not in sys.path:
            sys.path.insert(0, str(pegasus_root))

        # ----- 1. Read mesh -----
        self._emit(f"Reading mesh: {mesh_path}")
        self.mesh = loader.load_mesh(mesh_path)
        n_faces = self.mesh["n_faces"]
        self._emit(f"  {self.mesh['n_vertices']:,} vertices, {n_faces:,} faces")

        if self._stop_requested:
            self.finished_signal.emit(-2)
            return

        # ----- 2. Read temperature statistics -----
        tmax_path = ptm_dir / "statistics" / "temperature_max.bin"
        self._emit(f"Reading T_max: {tmax_path}")
        T_max = loader.load_ptm_binary(tmax_path)
        if len(T_max) != n_faces:
            self._emit_err(f"T_max count {len(T_max)} != mesh faces {n_faces}")
            self.finished_signal.emit(-1)
            return

        # ----- 3. Read timesteps -----
        ts_path = ptm_dir / "time_steps.txt"
        self._emit(f"Reading timesteps: {ts_path}")
        ptm_timesteps = loader.load_timesteps(ptm_dir)
        self._emit(f"  {len(ptm_timesteps)} timesteps")

        # ----- 4. Read waypoints -----
        self._emit(f"Reading waypoints: {waypoints_path}")
        self.waypoints = loader.load_waypoints(waypoints_path)
        n_waypoints = len(self.waypoints)
        self._emit(f"  {n_waypoints} waypoints")

        if self._stop_requested:
            self.finished_signal.emit(-2)
            return

        # ----- 5. Map mission dates to PTM timesteps -----
        mission_start = datetime.strptime(mission["start_utc"][:10], "%Y-%m-%d")
        mission_end = datetime.strptime(mission["end_utc"][:10], "%Y-%m-%d")
        mission_duration = (mission_end - mission_start).total_seconds()

        self._emit(f"Mission: {mission['start_utc']} to {mission['end_utc']}")

        # Check PTM data range and determine if time shifting is needed
        ptm_first, ptm_last = _ptm_time_range(ptm_timesteps)
        if ptm_first and ptm_last:
            self._emit(
                f"PTM data range: {ptm_first.strftime('%Y-%m-%d %H:%M')} to "
                f"{ptm_last.strftime('%Y-%m-%d %H:%M')}"
            )

        # Compute optimal time shift (0 if mission overlaps with PTM data)
        shift_days, shift_applied = _compute_time_shift(
            mission_start, mission_end, ptm_first, ptm_last
        )

        # Store shift for use in aggregation functions
        self._time_shift_days = shift_days

        # Compute effective (possibly shifted) mission window
        effective_start = mission_start - timedelta(days=shift_days)
        effective_end = mission_end - timedelta(days=shift_days)

        if shift_applied:
            self._emit(
                f"Shifted mission window: {effective_start.strftime('%Y-%m-%d %H:%M')} to "
                f"{effective_end.strftime('%Y-%m-%d %H:%M')} "
                f"(shift: {shift_days:.1f} days = {shift_days/SYNODIC_MONTH_DAYS:.1f} synodic months)"
            )
            # Validate shifted window overlap
            if not _check_window_overlap(effective_start, effective_end, ptm_first, ptm_last):
                self._emit(
                    "WARNING: Shifted mission window does not overlap with PTM data range! "
                    "Timestep mapping will be unreliable."
                )
            elif effective_start < ptm_first or effective_end > ptm_last:
                self._emit(
                    "WARNING: Shifted mission window only partially overlaps PTM data range."
                )
        else:
            self._emit("Mission window directly overlaps PTM data range (no shifting needed)")

        # Compute mean PTM timestep interval for match quality assessment
        ptm_dt_seconds = None
        if len(ptm_timesteps) >= 2 and ptm_first and ptm_last:
            ptm_dt_seconds = (ptm_last - ptm_first).total_seconds() / (len(ptm_timesteps) - 1)

        wp_times = []
        wp_ptm_indices = []
        for i in range(n_waypoints):
            frac = i / max(n_waypoints - 1, 1)
            mission_time = mission_start + timedelta(seconds=frac * mission_duration)
            ptm_idx, offset_sec = _map_mission_to_ptm_timestep(
                mission_time, ptm_timesteps, shift_days
            )
            wp_times.append(mission_time)
            wp_ptm_indices.append(ptm_idx)
            offset_hours = offset_sec / 3600.0
            quality = ""
            if ptm_dt_seconds and offset_sec > ptm_dt_seconds:
                quality = f"  (WARNING: offset {offset_hours:.1f} h > 1 PTM step)"
            self._emit(
                f"  {self.waypoints[i]['label']:>8s}: "
                f"{mission_time.strftime('%Y-%m-%d %H:%M')} -> PTM step {ptm_idx} "
                f"(offset {offset_hours:.1f} h){quality}"
            )

        # ----- 6. Generate truth ice map -----
        threshold = ice_cfg["temperature_threshold_K"]
        ice_frac_val = ice_cfg["ice_fraction"]
        gradient_K = ice_cfg["gradient_scale_K"]

        if gradient_K > 0:
            # Smooth sigmoid transition
            x = (threshold - T_max) / gradient_K
            ice_frac_truth = (ice_frac_val / (1.0 + np.exp(-x))).astype(np.float32)
        else:
            ice_frac_truth = np.where(
                T_max < threshold, ice_frac_val, 0.0
            ).astype(np.float32)

        self.ice_frac_truth = ice_frac_truth
        n_icy = int(np.sum(ice_frac_truth > 0))
        self._emit(
            f"\nTruth ice map: {n_icy:,} / {n_faces:,} faces "
            f"({100.0 * n_icy / n_faces:.1f}%) have ice "
            f"(T_max < {threshold} K, fraction = {ice_frac_val})"
        )

        if out_cfg["write_truth_map"]:
            loader.write_ptm_binary(output_dir / "truth_ice_fraction.bin", ice_frac_truth)

        if self._stop_requested:
            self.finished_signal.emit(-2)
            return

        # ----- 7. Import viewshed computation -----
        try:
            from pegasus_sim.viewshed import compute_viewshed
        except ImportError:
            self._emit_err(
                "Could not import pegasus_sim.viewshed. "
                "Ensure pegasus-sim is installed or accessible."
            )
            self.finished_signal.emit(-1)
            return

        # ----- 8. Compute viewshed at each waypoint -----
        body_radius = mission["body_radius_m"]
        max_emission = vs_cfg["max_emission_angle_deg"]

        self._emit(f"\n{'='*60}")
        self._emit(f"Computing viewsheds at {n_waypoints} waypoints")
        self._emit(f"{'='*60}")

        best_ice_est = np.zeros(n_faces, dtype=np.float64)
        best_snr = np.zeros(n_faces, dtype=np.float64)

        self.waypoint_results = []
        self.summary_rows = []

        for wp_i in range(n_waypoints):
            if self._stop_requested:
                self._emit("Simulation stopped by user.")
                self.finished_signal.emit(-2)
                return

            wp = self.waypoints[wp_i]
            label = wp["label"]
            ptm_idx = wp_ptm_indices[wp_i]

            self._emit(f"\n--- Waypoint {label} (step {ptm_idx}) ---")

            # Read temperature at this timestep
            temp_path = ptm_dir / "temperature" / f"temperature_{ptm_idx:04d}.bin"
            temperature = loader.load_ptm_binary(temp_path)

            # Read solar illumination at this timestep (if available)
            # PTM3D outputs to direct_solar/direct_solar_XXXX.bin
            solar_flux = None
            solar_path = ptm_dir / "direct_solar" / f"direct_solar_{ptm_idx:04d}.bin"
            if solar_path.exists():
                solar_flux = loader.load_ptm_binary(solar_path)
            else:
                # Try alternative naming conventions
                for alt_dir, alt_prefix in [
                    ("direct_solar_flux", "direct_solar_flux"),
                    ("solar_flux", "solar_flux"),
                ]:
                    alt_path = ptm_dir / alt_dir / f"{alt_prefix}_{ptm_idx:04d}.bin"
                    if alt_path.exists():
                        solar_flux = loader.load_ptm_binary(alt_path)
                        break

            # Compute geometric viewshed
            viewshed = compute_viewshed(
                observer_lat=wp["latitude"],
                observer_lon=wp["longitude"],
                observer_height_m=wp["height"],
                face_centers=self.mesh["face_centers"],
                face_normals=self.mesh["face_normals"],
                face_areas=self.mesh["face_areas"],
                body_radius_m=body_radius,
                max_emission_angle_deg=max_emission,
            )

            n_visible = int(np.sum(viewshed["visibility"] > 0))
            self._emit(
                f"  Viewshed: {n_visible:,} visible faces "
                f"({100.0 * n_visible / n_faces:.1f}%)"
            )

            # Store per-waypoint result (include mission time for display)
            wp_result = {
                "label": label,
                "ptm_idx": ptm_idx,
                "mission_time": wp_times[wp_i],  # datetime for UTC display
                "viewshed": viewshed,
                "temperature": temperature,
                "solar_flux": solar_flux,
                "n_visible": n_visible,
            }
            self.waypoint_results.append(wp_result)

            # Write per-waypoint outputs
            # Use NaN for non-visible faces (not 0.0) so they display as masked
            if out_cfg["write_per_waypoint"]:
                vis = viewshed["visibility"]
                mask = vis > 0

                loader.write_ptm_binary(
                    output_dir / f"visibility_{label}.bin",
                    vis.astype(np.float32),
                )
                loader.write_ptm_binary(
                    output_dir / f"distance_km_{label}.bin",
                    np.where(mask, viewshed["distance_km"], np.nan).astype(np.float32),
                )
                loader.write_ptm_binary(
                    output_dir / f"emission_angle_{label}.bin",
                    np.where(mask, viewshed["emission_angle_deg"], np.nan).astype(np.float32),
                )
                loader.write_ptm_binary(
                    output_dir / f"elevation_angle_{label}.bin",
                    np.where(mask, viewshed["elevation_angle_deg"], np.nan).astype(np.float32),
                )
                loader.write_ptm_binary(
                    output_dir / f"temperature_{label}.bin",
                    np.where(mask, temperature, np.nan).astype(np.float32),
                )
                # Write solar illumination if available
                if solar_flux is not None:
                    loader.write_ptm_binary(
                        output_dir / f"direct_solar_{label}.bin",
                        np.where(mask, solar_flux, np.nan).astype(np.float32),
                    )

            self.summary_rows.append({
                "label": label,
                "ptm_step": ptm_idx,
                "n_visible": n_visible,
            })

            # Progress
            progress = 100.0 * (wp_i + 1) / n_waypoints
            self.progress_update.emit(progress)

        self.best_ice_est = best_ice_est
        self.best_snr = best_snr

        # ----- 9. Compute aggregation fields -----
        self._emit(f"\n{'='*60}")
        self._emit("Computing aggregation fields")
        self._emit(f"{'='*60}")

        # Visibility count: how many waypoints each face is visible from
        vis_count = np.zeros(n_faces, dtype=np.int32)
        for wr in self.waypoint_results:
            vis_count += (wr["viewshed"]["visibility"] > 0).astype(np.int32)

        loader.write_ptm_binary(
            output_dir / "visibility_count.bin",
            vis_count.astype(np.float32),
        )
        n_any = int(np.sum(vis_count > 0))
        self._emit(
            f"  Visibility count: {n_any:,} faces visible from >= 1 waypoint"
        )

        # Illumination hours (using temperature proxy if illumination data unavailable)
        self._compute_illumination_hours(
            ptm_dir, ptm_timesteps, mission_start, mission_end,
            n_faces, output_dir
        )

        # Mean temperature across traverse window
        self._compute_mean_temperature(
            ptm_dir, ptm_timesteps, mission_start, mission_end,
            n_faces, output_dir
        )

        # Earth visibility fraction (from Primary IR data)
        self._compute_earth_visibility(
            ptm_dir, ptm_timesteps, mission_start, mission_end,
            n_faces, output_dir
        )

        # ----- 10. Compute custom masks -----
        if mask_defs:
            self._emit(f"\nComputing {len(mask_defs)} custom mask(s)")
            self._compute_masks(mask_defs, n_faces, output_dir)

        # ----- 11. VTK / HDF5 export -----
        if out_cfg.get("write_vtk", False):
            self._export_vtk(output_dir, mesh_path, n_faces)
        if out_cfg.get("write_hdf5", False):
            self._export_hdf5(output_dir, n_faces)

        # ----- 12. Summary -----
        self._emit(f"\n{'='*60}")
        self._emit(
            f"{'Waypoint':>10s} {'Step':>5s} {'Visible':>9s}"
        )
        self._emit(f"{'-'*10:>10s} {'-'*5:>5s} {'-'*9:>9s}")
        for row in self.summary_rows:
            self._emit(
                f"{row['label']:>10s} {row['ptm_step']:>5d} "
                f"{row['n_visible']:>9,d}"
            )

        # ----- 13. Write metadata (for visualization panel) -----
        import json

        # Compute face index for each waypoint (closest face to lat/lon position)
        face_lats = self.mesh.get("face_latitudes")
        face_lons = self.mesh.get("face_longitudes")

        def find_closest_face(wp_lat, wp_lon):
            """Find the mesh face closest to the given lat/lon."""
            if face_lats is None or face_lons is None:
                return None
            # Simple angular distance (works for small regions)
            dlat = face_lats - wp_lat
            dlon = face_lons - wp_lon
            # Wrap longitude difference to [-180, 180]
            dlon = np.where(dlon > 180, dlon - 360, dlon)
            dlon = np.where(dlon < -180, dlon + 360, dlon)
            dist_sq = dlat**2 + dlon**2
            return int(np.argmin(dist_sq))

        # Build waypoint metadata with face indices
        waypoint_meta = []
        for i, wr in enumerate(self.waypoint_results):
            wp = self.waypoints[i]
            face_idx = find_closest_face(wp["latitude"], wp["longitude"])
            waypoint_meta.append({
                "label": wr["label"],
                "ptm_idx": wr["ptm_idx"],
                "mission_time_utc": wr["mission_time"].strftime("%Y-%m-%dT%H:%M:%S"),
                "face_idx": face_idx,
                "latitude": wp["latitude"],
                "longitude": wp["longitude"],
                "height": wp["height"],
            })

        metadata = {
            "waypoints": waypoint_meta,
            "mission_start_utc": mission_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "mission_end_utc": mission_end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        metadata_path = output_dir / "viewshed_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._emit(f"\nOutput directory: {output_dir}")
        self._emit("Done.")
        self.finished_signal.emit(0)

    def _compute_illumination_hours(self, ptm_dir, ptm_timesteps,
                                     mission_start, mission_end,
                                     n_faces, output_dir):
        """Compute per-face direct solar illumination hours during the traverse window.

        Uses direct_solar data (not total illumination which includes reflected/IR).
        """
        # Find timesteps in the window (use computed time shift)
        ts_in_window = []
        shift_days = getattr(self, '_time_shift_days', 0.0)
        shifted_start = mission_start - timedelta(days=shift_days)
        shifted_end = mission_end - timedelta(days=shift_days)

        for idx, utc_str in ptm_timesteps:
            try:
                ts_dt = _parse_utc(utc_str)
                if shifted_start <= ts_dt <= shifted_end:
                    ts_in_window.append((idx, ts_dt))
            except ValueError:
                continue

        if len(ts_in_window) < 2:
            ptm_first, ptm_last = _ptm_time_range(ptm_timesteps)
            window_type = "effective window" if shift_days != 0 else "mission window"
            self._emit(
                f"  Direct solar hours: {len(ts_in_window)} timesteps in {window_type} "
                f"[{shifted_start.strftime('%Y-%m-%d')} .. {shifted_end.strftime('%Y-%m-%d')}], "
                f"need >= 2. PTM range: "
                f"[{ptm_first.strftime('%Y-%m-%d') if ptm_first else '?'} .. "
                f"{ptm_last.strftime('%Y-%m-%d') if ptm_last else '?'}]. Skipping."
            )
            return

        # Compute time interval between timesteps (hours)
        dt_hours = []
        for i in range(len(ts_in_window)):
            if i == 0:
                dt = (ts_in_window[1][1] - ts_in_window[0][1]).total_seconds() / 3600
            elif i == len(ts_in_window) - 1:
                dt = (ts_in_window[-1][1] - ts_in_window[-2][1]).total_seconds() / 3600
            else:
                dt = (ts_in_window[i + 1][1] - ts_in_window[i - 1][1]).total_seconds() / 7200
            dt_hours.append(abs(dt))

        illum_hours = np.zeros(n_faces, dtype=np.float64)
        n_processed = 0
        data_source = None

        for (ts_idx, _), dt_h in zip(ts_in_window, dt_hours):
            flux = None

            # Priority 1: Try direct_solar data (direct solar flux only)
            direct_solar_path = ptm_dir / "direct_solar" / f"direct_solar_{ts_idx:04d}.bin"
            if direct_solar_path.exists():
                flux = loader.load_ptm_binary(direct_solar_path)
                if data_source is None:
                    data_source = "direct_solar"

            # Priority 2: Fallback to illumination (total illumination)
            if flux is None or len(flux) != n_faces:
                flux = loader.load_illumination_timestep(ptm_dir, ts_idx)
                if flux is not None and len(flux) == n_faces:
                    if data_source is None:
                        data_source = "illumination (fallback)"

            # Priority 3: Use temperature proxy as last resort
            if flux is None or len(flux) != n_faces:
                try:
                    temp = loader.load_temperature_timestep(ptm_dir, ts_idx)
                    if len(temp) == n_faces:
                        flux = (temp > 100.0).astype(np.float32)  # rough day/night threshold
                        if data_source is None:
                            data_source = "temperature proxy (fallback)"
                    else:
                        continue
                except FileNotFoundError:
                    continue

            if flux is not None and len(flux) == n_faces:
                illuminated = flux > 0
                illum_hours += illuminated.astype(np.float64) * dt_h
                n_processed += 1

        if n_processed > 0:
            loader.write_ptm_binary(
                output_dir / "illumination_hours.bin",
                illum_hours.astype(np.float32),
            )
            n_lit = int(np.sum(illum_hours > 0))
            self._emit(
                f"  Direct solar hours: {n_lit:,} faces with > 0 hours "
                f"(from {n_processed} timesteps, using {data_source})"
            )
        else:
            self._emit(
                f"  Direct solar hours: {len(ts_in_window)} timesteps matched window but "
                f"no data loaded successfully (check file naming/face count). Skipping."
            )

    def _compute_mean_temperature(self, ptm_dir, ptm_timesteps,
                                   mission_start, mission_end,
                                   n_faces, output_dir):
        """Compute mean temperature across the traverse window."""
        shift_days = getattr(self, '_time_shift_days', 0.0)
        shifted_start = mission_start - timedelta(days=shift_days)
        shifted_end = mission_end - timedelta(days=shift_days)

        temp_sum = np.zeros(n_faces, dtype=np.float64)
        n_ts = 0

        for idx, utc_str in ptm_timesteps:
            try:
                ts_dt = _parse_utc(utc_str)
            except ValueError:
                continue
            if not (shifted_start <= ts_dt <= shifted_end):
                continue
            try:
                temp = loader.load_temperature_timestep(ptm_dir, idx)
                if len(temp) == n_faces:
                    temp_sum += temp.astype(np.float64)
                    n_ts += 1
            except FileNotFoundError:
                continue

        if n_ts > 0:
            mean_temp = (temp_sum / n_ts).astype(np.float32)
            loader.write_ptm_binary(
                output_dir / "temperature_mean_traverse.bin",
                mean_temp,
            )
            self._emit(
                f"  Mean temperature: computed from {n_ts} timesteps, "
                f"range [{mean_temp.min():.1f}, {mean_temp.max():.1f}] K"
            )
        else:
            ptm_first, ptm_last = _ptm_time_range(ptm_timesteps)
            window_type = "effective window" if shift_days != 0 else "mission window"
            self._emit(
                f"  Mean temperature: no timesteps matched {window_type} "
                f"[{shifted_start.strftime('%Y-%m-%d')} .. {shifted_end.strftime('%Y-%m-%d')}] "
                f"or data files could not be loaded. PTM range: "
                f"[{ptm_first.strftime('%Y-%m-%d') if ptm_first else '?'} .. "
                f"{ptm_last.strftime('%Y-%m-%d') if ptm_last else '?'}]. Skipping."
            )

    def _compute_earth_visibility(self, ptm_dir, ptm_timesteps,
                                    mission_start, mission_end,
                                    n_faces, output_dir):
        """Compute per-face Earth visibility fraction during the traverse window.

        Uses PTM3D primary_ir data: a face has Earth visible when primary_ir > 0.
        Writes earth_visibility_fraction.bin (aggregated) and per-timestep
        earth_visible_XXXX.bin files.
        """
        # Find timesteps in the traverse window (use computed time shift)
        ts_in_window = []
        shift_days = getattr(self, '_time_shift_days', 0.0)
        shifted_start = mission_start - timedelta(days=shift_days)
        shifted_end = mission_end - timedelta(days=shift_days)

        for idx, utc_str in ptm_timesteps:
            try:
                ts_dt = _parse_utc(utc_str)
                if shifted_start <= ts_dt <= shifted_end:
                    ts_in_window.append((idx, ts_dt))
            except ValueError:
                continue

        if len(ts_in_window) < 2:
            ptm_first, ptm_last = _ptm_time_range(ptm_timesteps)
            window_type = "effective window" if shift_days != 0 else "mission window"
            self._emit(
                f"  Earth visibility: {len(ts_in_window)} timesteps in {window_type} "
                f"[{shifted_start.strftime('%Y-%m-%d')} .. {shifted_end.strftime('%Y-%m-%d')}], "
                f"need >= 2. PTM range: "
                f"[{ptm_first.strftime('%Y-%m-%d') if ptm_first else '?'} .. "
                f"{ptm_last.strftime('%Y-%m-%d') if ptm_last else '?'}]. Skipping."
            )
            return

        # Count per-face visible timesteps
        vis_count = np.zeros(n_faces, dtype=np.float64)
        n_loaded = 0

        for idx, ts_dt in ts_in_window:
            ir_data = loader.load_primary_ir_timestep(ptm_dir, idx)
            if ir_data is not None and len(ir_data) == n_faces:
                earth_vis = (ir_data > 0).astype(np.float32)
                vis_count += earth_vis.astype(np.float64)
                n_loaded += 1
                # Write per-timestep binary (0/1)
                loader.write_ptm_binary(
                    output_dir / f"earth_visible_{idx:04d}.bin", earth_vis
                )

        if n_loaded == 0:
            self._emit(
                f"  Earth visibility: {len(ts_in_window)} timesteps matched window but "
                f"no primary_ir data loaded (check primary_ir/ directory). Skipping."
            )
            return

        # Compute fraction and store
        earth_vis_frac = (vis_count / n_loaded).astype(np.float32)
        self.earth_visibility_fraction = earth_vis_frac

        loader.write_ptm_binary(
            output_dir / "earth_visibility_fraction.bin", earth_vis_frac
        )

        n_ever = int(np.sum(earth_vis_frac > 0))
        n_always = int(np.sum(earth_vis_frac >= 1.0))
        self._emit(
            f"  Earth visibility: {n_loaded} timesteps processed, "
            f"{n_ever:,} faces visible at least once, "
            f"{n_always:,} always visible"
        )

    def _export_vtk(self, output_dir, mesh_path, n_faces):
        """Write per-waypoint VTU files and a PVD collection for ParaView.

        Each waypoint produces a VTU file containing:
          - Per-waypoint fields: visibility, distance_km, emission_angle,
            elevation_angle, temperature, direct_solar
          - Aggregation fields (constant across timesteps): visibility_count,
            illumination_hours, earth_visibility_fraction, mean_temperature

        A PVD (ParaView Data) collection file references all VTU files with
        mission UTC times converted to hours-since-start for the timestep axis.

        Uses raw VTK API (not PyVista) to ensure correct VTK_TRIANGLE cell
        types in the VTU output, which is required for ParaView rendering.
        """
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            self._emit("  VTK export: vtk not available, skipping.")
            return

        if self.mesh is None:
            self._emit("  VTK export: no mesh data available, skipping.")
            return
        if not self.waypoint_results:
            self._emit("  VTK export: no waypoint results available, skipping.")
            return

        try:
            output_dir = Path(output_dir)
            vtk_dir = output_dir / "vtk"
            vtk_dir.mkdir(parents=True, exist_ok=True)

            # Build VTK unstructured grid from mesh geometry
            vertices = np.asarray(self.mesh["vertices"], dtype=np.float64)
            # pegasus_sim uses "face_indices"; fallback reader uses "faces"
            face_key = (
                "face_indices" if "face_indices" in self.mesh else "faces"
            )
            faces = np.asarray(self.mesh[face_key], dtype=np.int64)

            ugrid = vtk.vtkUnstructuredGrid()
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
            ugrid.SetPoints(vtk_points)

            # Connectivity: [3, v0, v1, v2, 3, v0, v1, v2, ...]
            cell_array = np.empty((len(faces), 4), dtype=np.int64)
            cell_array[:, 0] = 3
            cell_array[:, 1:4] = faces
            cells = vtk.vtkCellArray()
            cells.SetCells(
                len(faces),
                numpy_support.numpy_to_vtk(
                    cell_array.ravel(), deep=True,
                    array_type=vtk.VTK_ID_TYPE,
                ),
            )
            ugrid.SetCells(vtk.VTK_TRIANGLE, cells)

            # Add lat/lon as base cell data
            for key, name in [
                ("face_latitudes", "latitude"),
                ("face_longitudes", "longitude"),
            ]:
                arr = self.mesh.get(key)
                if arr is not None and len(arr) == n_faces:
                    vtk_arr = numpy_support.numpy_to_vtk(
                        np.asarray(arr, dtype=np.float64), deep=True,
                    )
                    vtk_arr.SetName(name)
                    ugrid.GetCellData().AddArray(vtk_arr)

            # Load aggregation fields (included in every timestep)
            agg_names = [
                "visibility_count", "illumination_hours",
                "earth_visibility_fraction", "mean_temperature",
            ]
            agg_fields = {}
            for name in agg_names:
                p = output_dir / f"{name}.bin"
                if p.exists():
                    data = loader.load_ptm_binary(p)
                    if len(data) == n_faces:
                        agg_fields[name] = data

            # Per-waypoint field names (files: <field>_<label>.bin)
            wp_field_names = [
                "visibility", "distance_km", "emission_angle",
                "elevation_angle", "temperature", "direct_solar",
            ]

            # Compute time axis: hours since first waypoint
            first_time = self.waypoint_results[0]["mission_time"]
            pvd_entries = []

            def _add_cell_array(grid, name, data):
                """Add a cleaned float32 array as cell data."""
                clean = np.nan_to_num(
                    np.asarray(data, dtype=np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                vtk_arr = numpy_support.numpy_to_vtk(clean, deep=True)
                vtk_arr.SetName(name)
                grid.GetCellData().AddArray(vtk_arr)

            for wr in self.waypoint_results:
                label = wr["label"]
                dt_hours = (
                    (wr["mission_time"] - first_time).total_seconds() / 3600.0
                )

                # Deep-copy base grid for this timestep
                grid = vtk.vtkUnstructuredGrid()
                grid.DeepCopy(ugrid)

                # Add per-waypoint fields
                for fname in wp_field_names:
                    p = output_dir / f"{fname}_{label}.bin"
                    if p.exists():
                        data = loader.load_ptm_binary(p)
                        if len(data) == n_faces:
                            _add_cell_array(grid, fname, data)

                # Add aggregation fields
                for aname, adata in agg_fields.items():
                    _add_cell_array(grid, aname, adata)

                # Write VTU with compression
                vtu_name = f"viewshed_{label}.vtu"
                vtu_path = vtk_dir / vtu_name
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetInputData(grid)
                writer.SetFileName(str(vtu_path))
                writer.SetDataModeToAppended()
                writer.SetCompressorTypeToZLib()
                writer.SetCompressionLevel(1)
                writer.SetEncodeAppendedData(0)
                writer.Write()

                pvd_entries.append((dt_hours, vtu_name))

            # Write PVD collection file
            pvd_path = vtk_dir / "viewshed.pvd"
            with open(pvd_path, "w") as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1">\n')
                f.write('  <Collection>\n')
                for t_val, vtu_file in pvd_entries:
                    f.write(
                        f'    <DataSet timestep="{t_val:.6f}" '
                        f'file="{vtu_file}"/>\n'
                    )
                f.write('  </Collection>\n')
                f.write('</VTKFile>\n')

            n_wp = len(pvd_entries)
            n_agg = len(agg_fields)
            self._emit(
                f"  VTK export: {n_wp} waypoint VTU files in {vtk_dir}/, "
                f"PVD collection: {pvd_path} "
                f"({n_wp} timesteps, {n_agg} aggregation fields per file)"
            )
        except Exception as e:
            self._emit_err(f"  VTK export failed: {e}")

    def _export_hdf5(self, output_dir, n_faces):
        """Write a single HDF5 file with all output fields."""
        try:
            import h5py
        except ImportError:
            self._emit("  HDF5 export: h5py not available, skipping.")
            return

        try:
            h5_path = output_dir / "viewshed_results.h5"
            with h5py.File(str(h5_path), "w") as f:
                f.attrs["software"] = "PEGASUS Viewshed GUI"
                f.attrs["n_faces"] = n_faces

                # Mesh group
                if self.mesh is not None:
                    mg = f.create_group("mesh")
                    for key in ("face_centers", "face_normals", "face_areas",
                                "face_latitudes", "face_longitudes"):
                        if key in self.mesh:
                            mg.create_dataset(key, data=self.mesh[key],
                                              compression="gzip")

                # Data group: all .bin fields
                dg = f.create_group("data")
                count = 0
                for p in sorted(Path(output_dir).glob("*.bin")):
                    data = loader.load_ptm_binary(p)
                    if len(data) == n_faces:
                        dg.create_dataset(p.stem, data=data,
                                          compression="gzip")
                        count += 1

            self._emit(f"  HDF5 export: {h5_path} ({count} fields)")
        except Exception as e:
            self._emit_err(f"  HDF5 export failed: {e}")

    def _compute_masks(self, mask_defs, n_faces, output_dir):
        """Evaluate and write custom boolean masks."""
        # Build a namespace of available fields for evaluation
        field_data = self._build_field_namespace(n_faces)

        for mask_def in mask_defs:
            name = mask_def.get("name", "unnamed_mask")
            conditions = mask_def.get("conditions", [])
            if not conditions:
                continue

            try:
                result_mask = self._evaluate_mask(conditions, field_data, n_faces)
                n_true = int(np.sum(result_mask))
                loader.write_ptm_binary(
                    output_dir / f"mask_{name}.bin",
                    result_mask.astype(np.float32),
                )
                self._emit(
                    f"  Mask '{name}': {n_true:,} / {n_faces:,} faces "
                    f"({100.0 * n_true / n_faces:.1f}%)"
                )
            except Exception as e:
                self._emit_err(f"  Mask '{name}' failed: {e}")

    def _build_field_namespace(self, n_faces):
        """Build a dict of field_name -> numpy array for mask evaluation."""
        fields = {}

        # Static fields from mesh
        if self.mesh is not None:
            fields["latitude"] = self.mesh.get("face_latitudes", np.zeros(n_faces))
            fields["longitude"] = self.mesh.get("face_longitudes", np.zeros(n_faces))
            fields["face_area"] = self.mesh.get("face_areas", np.zeros(n_faces))
            fields["elevation"] = self.mesh.get("face_elevations", np.zeros(n_faces))

        # Temperature statistics from PTM dir
        ptm_dir = self.config_manager.get_value("input.ptm_output_dir")
        if ptm_dir:
            stats = loader.load_temperature_statistics(ptm_dir)
            for key in ("T_max", "T_min", "T_mean"):
                if key in stats:
                    fields[key] = stats[key]

        # Ice truth
        if self.ice_frac_truth is not None:
            fields["ice_fraction_truth"] = self.ice_frac_truth

        # Per-waypoint fields (indexed by waypoint label)
        for wr in self.waypoint_results:
            label = wr["label"]
            vs = wr["viewshed"]
            fields[f"visibility_{label}"] = vs["visibility"]
            fields[f"distance_km_{label}"] = vs["distance_km"]
            fields[f"emission_angle_{label}"] = vs["emission_angle_deg"]
            fields[f"elevation_angle_{label}"] = vs["elevation_angle_deg"]
            fields[f"temperature_{label}"] = wr["temperature"]

        # Aggregation: visibility count
        vis_count = np.zeros(n_faces, dtype=np.int32)
        for wr in self.waypoint_results:
            vis_count += (wr["viewshed"]["visibility"] > 0).astype(np.int32)
        fields["visibility_count"] = vis_count.astype(np.float32)

        # Aggregation: earth visibility fraction
        if self.earth_visibility_fraction is not None:
            fields["earth_visibility_fraction"] = self.earth_visibility_fraction

        return fields

    def _evaluate_mask(self, conditions, field_data, n_faces):
        """Evaluate a list of conditions with AND/OR connectors.

        Returns a boolean numpy array.
        """
        OPS = {
            "<": np.less,
            "<=": np.less_equal,
            "==": np.equal,
            "=": np.equal,
            ">=": np.greater_equal,
            ">": np.greater,
            "!=": np.not_equal,
        }

        result = None

        for cond in conditions:
            field_name = cond["field"]
            op_str = cond["op"]
            value = float(cond["value"])
            connector = cond.get("connector", "AND").upper()

            # Resolve waypoint-specific field
            waypoint = cond.get("waypoint", "")
            if waypoint and not field_name.endswith(f"_{waypoint}"):
                field_name = f"{field_name}_{waypoint}"

            if field_name not in field_data:
                raise ValueError(f"Unknown field: {field_name}")

            op_func = OPS.get(op_str)
            if op_func is None:
                raise ValueError(f"Unknown operator: {op_str}")

            cond_result = op_func(field_data[field_name], value)

            if result is None:
                result = cond_result
            elif connector == "OR":
                result = result | cond_result
            else:
                result = result & cond_result

        return result if result is not None else np.ones(n_faces, dtype=bool)
