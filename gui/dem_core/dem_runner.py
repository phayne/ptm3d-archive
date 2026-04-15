"""
DEM processing threads: wraps the dtm_to_ply_delaunay and
dtm_advanced_processor executables.
"""

import math
import os
import re
import subprocess
import select
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


def _find_executable(name):
    """Search for a DEM tool executable.

    Searches relative to gui/ directory:
      ../../dtm/<name>     (the dtm directory)
      ../build/bin/<name>  (PTM build directory)
    Then falls back to PATH.
    """
    gui_dir = Path(__file__).resolve().parent.parent
    search_paths = [
        gui_dir / ".." / ".." / "dtm" / name,
        gui_dir / ".." / "build" / "bin" / name,
    ]

    for p in search_paths:
        resolved = p.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)

    # Try PATH
    import shutil
    found = shutil.which(name)
    return found


def _run_subprocess(process_holder, cmd, output_signal, error_signal,
                    progress_signal, finished_signal, stop_flag):
    """Execute subprocess with non-blocking I/O. Shared by both threads."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
            bufsize=0,
        )
        process_holder[0] = process

        stdout_buf = bytearray()
        stderr_buf = bytearray()

        def drain(buf, is_stderr):
            parts = re.split(b'[\r\n]', buf)
            complete = parts[:-1]
            remainder = parts[-1] if parts else b""
            for seg in complete:
                if not seg:
                    continue
                decoded = seg.decode('utf-8', errors='replace').strip()
                if not decoded:
                    continue
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', decoded)
                if match:
                    progress_signal.emit(float(match.group(1)))
                signal = error_signal if is_stderr else output_signal
                signal.emit(decoded)
            buf.clear()
            buf.extend(remainder)

        while process.poll() is None and not stop_flag[0]:
            reads = []
            if process.stdout:
                reads.append(process.stdout)
            if process.stderr:
                reads.append(process.stderr)
            if not reads:
                break
            ready, _, _ = select.select(reads, [], [], 0.1)
            for fd in ready:
                data = fd.read1(4096) if hasattr(fd, 'read1') else fd.read(4096)
                if not data:
                    continue
                if fd == process.stdout:
                    stdout_buf.extend(data)
                    drain(stdout_buf, False)
                else:
                    stderr_buf.extend(data)
                    drain(stderr_buf, True)

        # Drain remaining
        for fd, buf, is_err in [
            (process.stdout, stdout_buf, False),
            (process.stderr, stderr_buf, True),
        ]:
            if fd:
                remaining = fd.read()
                if remaining:
                    buf.extend(remaining)
                    drain(buf, is_err)

        exit_code = process.wait()
        finished_signal.emit(exit_code)

    except Exception as e:
        error_signal.emit(f"Execution error: {e}")
        finished_signal.emit(-1)


class DEMConvertThread(QThread):
    """Wraps the dtm_to_ply_delaunay executable (Delaunay triangulation)."""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished = pyqtSignal(int)

    def __init__(self, config_manager):
        super().__init__()
        self._cm = config_manager
        self._process = [None]
        self._stop = [False]

    def run(self):
        executable = _find_executable("dtm_to_ply_delaunay")
        if not executable:
            self.error_ready.emit(
                "Could not find dtm_to_ply_delaunay executable.\n"
                "Build it: cd dtm && make"
            )
            self.finished.emit(-1)
            return

        config = self._cm.get_config()
        dem_file = config["input"]["dem_file"]
        output_file = config["conversion"]["output_file"]

        if not dem_file or not output_file:
            self.error_ready.emit("DEM file and output file must be specified")
            self.finished.emit(-1)
            return

        # Build command
        cmd = [executable, dem_file, output_file]

        # Downsample
        downsample = config["conversion"].get("downsample", 1)
        if downsample > 1:
            cmd.extend(["--downsample", str(downsample)])

        # Body radius
        body_radius = config["input"].get("body_radius_m", 0)
        if body_radius > 0:
            cmd.extend(["--body-radius", str(body_radius)])

        # ROI parameters
        if config["roi"].get("enabled", False):
            lat = config["roi"]["center_lat"]
            lon = config["roi"]["center_lon"]
            cmd.extend(["--roi-center", str(lat), str(lon)])

            radius_km = config["roi"].get("radius_km", 10.0)
            cmd.extend(["--roi-radius", str(radius_km)])

            max_pts = config["roi"].get("max_points", 0)
            if max_pts > 0:
                cmd.extend(["--roi-max-points", str(max_pts)])

            core_frac = config["conversion"].get("roi_core_frac", 0.15)
            cmd.extend(["--roi-core-frac", str(core_frac)])

        # Density falloff exponent
        density_exp = config["conversion"].get("density_exp", 2.0)
        if density_exp != 2.0:
            cmd.extend(["--density-exp", str(density_exp)])

        # Max edge length
        max_edge = config["conversion"].get("max_edge_m", -1.0)
        if max_edge >= 0:
            # 0 = disable filtering, >0 = threshold; -1 = auto (omit flag)
            cmd.extend(["--max-edge", str(max_edge)])

        # Coordinate corrections
        if config["conversion"].get("south_pole_coords", False):
            cmd.append("--south-pole-coords")
        if config["conversion"].get("swap_axes", False):
            cmd.append("--swap-axes")

        # Quality control
        qc = config.get("quality_control", {})
        if not qc.get("enabled", True):
            cmd.append("--no-qc")
        else:
            sliver = qc.get("sliver_angle_deg", 5.0)
            if sliver != 5.0:
                cmd.extend(["--qc-sliver-angle", str(sliver)])
            max_hole = qc.get("max_hole_edges", 20)
            if max_hole != 20:
                cmd.extend(["--qc-max-hole", str(max_hole)])

        self.output_ready.emit(f"Running: {' '.join(cmd)}")
        _run_subprocess(
            self._process, cmd,
            self.output_ready, self.error_ready,
            self.progress_update, self.finished,
            self._stop,
        )

    def stop(self):
        self._stop[0] = True
        if self._process[0]:
            self._process[0].terminate()


class DEMRefineThread(QThread):
    """Wraps the dtm_advanced_processor executable."""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished = pyqtSignal(int)

    def __init__(self, config_manager):
        super().__init__()
        self._cm = config_manager
        self._process = [None]
        self._stop = [False]

    def run(self):
        executable = _find_executable("dtm_advanced_processor")
        if not executable:
            self.error_ready.emit(
                "Could not find dtm_advanced_processor executable.\n"
                "Build it: cd dtm && make"
            )
            self.finished.emit(-1)
            return

        config = self._cm.get_config()
        dem_file = config["input"]["dem_file"]
        output_file = config["refinement"]["output_file"]

        if not dem_file or not output_file:
            self.error_ready.emit("DEM file and refinement output file must be specified")
            self.finished.emit(-1)
            return

        cmd = [executable, dem_file, output_file]

        ref = config["refinement"]
        cmd.extend(["--refine-center", str(ref["center_lat"]), str(ref["center_lon"])])
        cmd.extend(["--refine-inner", str(ref["inner_radius_km"])])
        cmd.extend(["--refine-outer", str(ref["outer_radius_km"])])
        cmd.extend(["--refine-max-faces", str(ref["max_faces"])])

        if ref.get("min_edge_m", 0) > 0:
            cmd.extend(["--refine-min-edge", str(ref["min_edge_m"])])
        if ref.get("max_edge_m", 0) > 0:
            cmd.extend(["--refine-max-edge", str(ref["max_edge_m"])])

        initial_ds = ref.get("initial_downsample", 10)
        if initial_ds > 1:
            cmd.extend(["--initial-downsample", str(initial_ds)])

        body_radius = config["input"].get("body_radius_m", 0)
        if body_radius > 0:
            cmd.extend(["--radius", str(body_radius)])

        self.output_ready.emit(f"Running: {' '.join(cmd)}")
        _run_subprocess(
            self._process, cmd,
            self.output_ready, self.error_ready,
            self.progress_update, self.finished,
            self._stop,
        )

    def stop(self):
        self._stop[0] = True
        if self._process[0]:
            self._process[0].terminate()
