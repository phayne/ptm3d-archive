"""
Simulation Runner for PTM3D GUI
Handles simulation execution in background thread

The process driving -- executable discovery, the output severity classifier,
the CPU-clock stall watchdog and process-group teardown -- lives in ``ptm.run``
and is shared with the command-line and notebook paths. What remains here is
the Qt layer: a QThread that forwards ``ptm.run``'s callbacks to signals.
"""

import subprocess
import sys
import os
import re
import select
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from ._ptm_bootstrap import ensure_ptm_importable

ensure_ptm_importable()

from ptm import run as ptm_run                        # noqa: E402
from ptm.run import EngineProcess, build_command      # noqa: E402

from gui_core.preflight import build_engine_env, find_project_root  # noqa: E402


class SimulationThread(QThread):
    """Thread for running PTM simulations without blocking GUI"""

    output_ready = pyqtSignal(str)    # info  (white)
    warning_ready = pyqtSignal(str)   # warn  (yellow)
    error_ready = pyqtSignal(str)     # error (red) -- genuine failures only
    progress_update = pyqtSignal(float)
    # Stall watchdog: (message, stalled). Emitted every WATCHDOG_INTERVAL_S
    # while the child's CPU clock is not advancing (blocked on file I/O --
    # e.g. a dehydrated cloud file -- or a wedged GPU), and once when
    # progress resumes. Lets the GUI distinguish "computing" from "hung".
    heartbeat = pyqtSignal(str, bool)
    finished = pyqtSignal(int)  # Exit code

    # Watchdog cadence and the stall threshold live with the driver.
    WATCHDOG_INTERVAL_S = EngineProcess.WATCHDOG_INTERVAL_S
    STALL_CPU_DELTA_S = EngineProcess.STALL_CPU_DELTA_S

    def __init__(self, config_file, working_dir=None, config_dict=None):
        super().__init__()
        self.config_file = config_file
        self.working_dir = working_dir
        self.config_dict = config_dict  # Full config for engine env + CLI options
        self.process = None             # the child Popen, once started
        self._engine = None             # ptm.run.EngineProcess
        self._stop_requested = False

    def run(self):
        """Execute the simulation and emit output"""
        try:
            executable = SimulationRunner.find_ptm_executable()
            if not executable:
                self.error_ready.emit("Could not find ptm_thermal_sim executable")
                self.finished.emit(-1)
                return

            # Radiation options are deliberately NOT duplicated onto the CLI:
            # config_parser.cpp reads the whole radiation section from the YAML
            # (see ptm.run.build_command).
            command = build_command(executable, self.config_file, self.config_dict)

            # Pin the subprocess cwd to the project root (derived from the
            # executable location) so relative config paths (data/...,
            # output/...) resolve deterministically instead of against
            # wherever the GUI happened to be launched from.
            if self.working_dir is None:
                self.working_dir = str(find_project_root(executable))

            # Engine environment: inherited env + the recommended-ON physics
            # fix flags + the user's Engine Flags table (config['engine_env']).
            # build_engine_env also sets PYTHON_EXECUTABLE so C++ can invoke
            # the correct Python (conda/venv).
            env, applied = build_engine_env(self.config_dict or {})
            self.output_ready.emit(f"[run] cwd: {self.working_dir}")
            if applied:
                self.output_ready.emit("[run] engine env: " + " ".join(applied))

            self._engine = EngineProcess(
                command,
                working_dir=self.working_dir,
                env=env,
                on_output=self.output_ready.emit,
                on_warning=self.warning_ready.emit,
                on_error=self.error_ready.emit,
                on_progress=self.progress_update.emit,
                on_heartbeat=lambda msg, stalled: self.heartbeat.emit(msg, stalled),
            )
            if self._stop_requested:      # stop() arrived before the child started
                self.finished.emit(-1)
                return

            returncode = self._engine.run()
            self.process = self._engine.process
            self.finished.emit(returncode)

        except Exception as e:
            self.error_ready.emit(f"Execution error: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        """Stop the running simulation.

        The child runs in its own session, so this signals the whole process
        group: TERM first, KILL after 5 s, both waits bounded so a process
        stuck in an uninterruptible cloud-file read never hangs the GUI.
        """
        self._stop_requested = True
        if self._engine is not None:
            self._engine.stop()
            self.process = self._engine.process

    # --- Severity classification -------------------------------------------
    # The engine writes informational and warning banners to stderr during
    # successful runs, so routing by stream mislabels a good run as failed;
    # ptm.run.classify_severity decides on content instead. Kept as methods
    # because the GUI and the tutorial screenshot script call them on an
    # instance.
    def classify_severity(self, text):
        """Classify one output line as 'error' | 'warning' | 'info'."""
        return ptm_run.classify_severity(text)

    # Back-compat alias (kept private-style for internal callers/tests).
    _classify_severity = classify_severity

    @staticmethod
    def _sample_cputime(pid):
        """Child CPU time in seconds via ps, or None."""
        return ptm_run.sample_cputime(pid)

    def _extract_progress(self, text):
        """Extract a progress percentage from text, or None."""
        return ptm_run.extract_progress(text)

    def _is_progress_line(self, text):
        """Check if line is a progress/info update (not an error).

        Retained for reference; routing now goes through classify_severity."""
        tl = text.lower()

        # Progress bars with percentages
        if '[' in text and ']' in text and '%' in text:
            return True

        # Known progress patterns
        if 'calculating' in tl and '%' in text:
            return True
        if 'thermal simulation' in tl:
            return True
        if 'progress' in tl:
            return True

        # Informational patterns (not errors)
        if 'processing chunk' in tl:
            return True
        if 'computing' in tl and 'of' in tl:
            return True
        if 'loading' in tl:
            return True
        if 'initializing' in tl:
            return True
        if 'writing' in tl and 'output' in tl:
            return True
        if 'saving' in tl:
            return True
        if 'reading' in tl:
            return True
        if 'starting' in tl:
            return True
        if 'finished' in tl:
            return True
        if 'complete' in tl and 'error' not in tl:
            return True
        if 'iteration' in tl:
            return True
        if 'convergence' in tl:
            return True

        return False


#: The checkout this GUI lives in: gui/gui_core/simulation_runner.py -> <root>.
_GUI_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Probed after <root>/build/bin, relative to the current directory. Kept for
#: launches from inside build/ or from a directory beside it.
_RELATIVE_BIN_CANDIDATES = ("../build/bin", "../../build/bin", "./build/bin",
                            "../bin", "./bin")


def _find_binary(name):
    return ptm_run.find_engine_executable(
        name,
        search_roots=(_GUI_REPO_ROOT,),
        extra_candidates=[f"{d}/{name}" for d in _RELATIVE_BIN_CANDIDATES],
    )


class SimulationRunner:
    """Helper class for locating and running PTM simulations"""

    @staticmethod
    def find_ptm_executable():
        """Locate the ptm_thermal_sim executable"""
        return _find_binary("ptm_thermal_sim")

    @staticmethod
    def find_converter_script():
        """Locate the VTK converter script.

        ``vtk_converter_optimized.py`` is probed first and is not in the
        repository -- the name is left in place for trees that carry it. The
        shipped converter is ``src/python/vtk_converter.py``, whose CLI takes
        exactly the four flags ConverterThread passes (--input-dir,
        --output-dir, --format, --no-statistics). Before this fallback existed,
        this returned None in every clean checkout and the GUI's post-run
        conversion could not run at all.
        """
        names = ("vtk_converter_optimized.py", "vtk_converter.py")
        roots = (_GUI_REPO_ROOT, Path(".."), Path("../.."), Path("."))
        for name in names:
            for root in roots:
                candidate = (root / "src" / "python" / name).resolve()
                if candidate.exists():
                    return candidate
        return None


class CameraSimThread(QThread):
    """Thread for running ptm_camera_sim without blocking GUI."""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    frame_rendered = pyqtSignal(str)   # emits FITS path when a frame is written
    finished = pyqtSignal(int)

    def __init__(self, command, working_dir=None):
        super().__init__()
        self.command = command          # full CLI arg list
        self.working_dir = working_dir
        self.process = None
        self._stop_requested = False

    def run(self):
        try:
            env = os.environ.copy()
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                bufsize=0,
                cwd=self.working_dir,
                env=env,
            )

            stdout_buf = bytearray()
            stderr_buf = bytearray()

            def drain(buf, is_stderr):
                parts = re.split(b'[\r\n]', buf)
                complete = parts[:-1]
                remainder = parts[-1] if parts else b""
                for seg in complete:
                    if not seg:
                        continue
                    line = seg.decode('utf-8', errors='replace').strip()
                    if not line:
                        continue
                    # Parse frame progress
                    m = re.match(r'=== Frame (\d+)/(\d+) ===', line)
                    if m:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        self.progress_update.emit(100.0 * cur / max(1, tot))
                    # Detect written FITS
                    if 'Wrote FITS:' in line:
                        fits_path = line.split('Wrote FITS:')[-1].strip()
                        self.frame_rendered.emit(fits_path)
                    signal = self.error_ready if is_stderr else self.output_ready
                    signal.emit(line)
                buf.clear()
                buf.extend(remainder)

            while not self._stop_requested:
                if self.process.poll() is not None:
                    tail = self.process.stdout.read() or b""
                    if tail:
                        stdout_buf.extend(tail)
                    tail = self.process.stderr.read() or b""
                    if tail:
                        stderr_buf.extend(tail)
                    break
                readable, _, _ = select.select(
                    [self.process.stdout, self.process.stderr], [], [], 0.1)
                for stream in readable:
                    data = stream.read(4096)
                    if not data:
                        continue
                    if stream is self.process.stdout:
                        stdout_buf.extend(data)
                        drain(stdout_buf, False)
                    else:
                        stderr_buf.extend(data)
                        drain(stderr_buf, True)

            for buf, is_err in [(stdout_buf, False), (stderr_buf, True)]:
                if buf:
                    try:
                        t = buf.decode('utf-8', errors='replace').strip()
                        if t:
                            self.output_ready.emit(t)
                    except Exception:
                        pass

            self.process.wait()
            self.finished.emit(self.process.returncode)

        except Exception as e:
            self.error_ready.emit(f"Camera sim error: {e}")
            self.finished.emit(-1)

    def stop(self):
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    @staticmethod
    def find_camera_executable():
        """Locate ptm_camera_sim executable."""
        return _find_binary("ptm_camera_sim")


class ConverterThread(QThread):
    """Thread for running VTK conversion"""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished = pyqtSignal(int)

    def __init__(self, input_dir, output_dir, output_format='vtk', no_statistics=False):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_format = output_format
        self.no_statistics = no_statistics
        self.process = None
        self._stop_requested = False

    def run(self):
        """Run the conversion"""
        try:
            script = SimulationRunner.find_converter_script()
            if not script:
                self.error_ready.emit("Could not find vtk_converter_optimized.py")
                self.finished.emit(-1)
                return

            python_exe = sys.executable
            cmd = [
                python_exe,
                str(script),
                "--input-dir", str(self.input_dir),
                "--output-dir", str(self.output_dir),
                "--format", self.output_format
            ]

            if self.no_statistics:
                cmd.append("--no-statistics")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            while not self._stop_requested:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    self.output_ready.emit(line.strip())

                    # Extract progress
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                    if match:
                        self.progress_update.emit(float(match.group(1)))

            # Read any remaining stderr
            stderr_output = self.process.stderr.read()
            if stderr_output:
                for line in stderr_output.strip().split('\n'):
                    if line:
                        self.error_ready.emit(line)

            self.process.wait()
            self.finished.emit(self.process.returncode)

        except Exception as e:
            self.error_ready.emit(f"Conversion error: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        """Stop the conversion"""
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
