"""
DEM processing thread: wraps the dtm_to_ply_delaunay executable.

That binary is built from this repository -- ``src/cpp/tools/`` -- and lands in
``build/bin/``. It requires GDAL at build time; see
``docs/dem-to-mesh-workflow.md`` and INSTALL.md ("DEM / DTM tools").

PLATFORM NOTE: ``_run_subprocess`` below streams the child's output with
``select.select`` on pipe file descriptors. That is a POSIX-only construct --
on Windows ``select`` accepts sockets only, so DEM conversion does not work
there. macOS and Linux are supported. There is no fallback: rather than promise
something we do not do, the code fails visibly with the platform check in
``_require_posix_pipes``.
"""

import math
import os
import re
import subprocess
import select
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


# Build target name -> the CMake/`cmake --build` target that produces it.
# Used to give an accurate "how do I get this binary" message.
_BUILD_TARGETS = {
    "dtm_to_ply_delaunay": "dtm_to_ply_delaunay",
}


def _repo_root():
    """Repository root, resolved from this file (gui/dem_core/ -> repo)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_executable(name):
    """Search for a DEM tool executable.

    Search order, most authoritative first:
      1. <repo>/build/bin/<name>   -- what `cmake --build build` produces
      2. <repo>/../dtm/<name>      -- legacy standalone dtm/ working directory
      3. PATH

    The in-repo build directory is searched FIRST so that a rebuild of this
    checkout is what actually runs. The legacy ``dtm/`` location is kept only
    so existing local setups do not break; those binaries predate the port and
    may be built for a different architecture.
    """
    repo = _repo_root()
    search_paths = [
        repo / "build" / "bin" / name,
        repo / ".." / "dtm" / name,
    ]

    for p in search_paths:
        resolved = p.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)

    # Try PATH
    import shutil
    found = shutil.which(name)
    return found


def _missing_executable_message(name):
    """Accurate build instructions for a DEM tool that was not found."""
    repo = _repo_root()
    target = _BUILD_TARGETS.get(name, name)
    return (
        f"Could not find the '{name}' executable.\n"
        f"Searched: {repo / 'build' / 'bin' / name}, "
        f"{(repo / '..' / 'dtm' / name)}, and PATH.\n"
        "\n"
        "Build it from this repository:\n"
        f"    cd {repo}\n"
        "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        f"    cmake --build build --target {target} -j 8\n"
        "\n"
        "PREREQUISITE: GDAL (C++ headers and library) must be installed and\n"
        "visible to CMake, or this target is SKIPPED at configure time -- look\n"
        "for 'GDAL not properly configured ... SKIPPING' in the CMake output.\n"
        "On macOS: conda install -c conda-forge gdal   (or: brew install gdal)\n"
        "See INSTALL.md, section 'DEM / DTM tools'."
    )


def _require_posix_pipes():
    """Return an error string if pipe-based streaming is unsupported here.

    ``select.select`` on pipes is POSIX-only; on Windows it handles sockets
    only. Report that plainly instead of failing deep inside the read loop.
    """
    if sys.platform.startswith("win"):
        return (
            "The DEM conversion action is not supported on Windows.\n"
            "It streams the child process's output with select.select() on "
            "pipes, which is POSIX-only.\n"
            "Run dtm_to_ply_delaunay from the command line instead (see "
            "docs/dem-to-mesh-workflow.md), or use macOS/Linux."
        )
    return None


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
        unsupported = _require_posix_pipes()
        if unsupported:
            self.error_ready.emit(unsupported)
            self.finished.emit(-1)
            return

        executable = _find_executable("dtm_to_ply_delaunay")
        if not executable:
            self.error_ready.emit(_missing_executable_message("dtm_to_ply_delaunay"))
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
