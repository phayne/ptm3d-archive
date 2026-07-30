"""
Simulation Runner for PTM3D GUI
Handles simulation execution in background thread
"""

import subprocess
import signal
import sys
import os
import re
import select
import time
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from gui_core.preflight import build_engine_env, find_project_root


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

    # --- Severity classification -------------------------------------------
    # The engine writes many INFORMATIONAL and WARNING banners to stderr
    # ([eq-seed] equilibration seed..., [machine-profile] ...CONSERVATIVE
    # defaults, [GPU Thermal] dispatch self-chunk ceiling..., progress). A
    # successful run therefore emits stderr lines that are NOT errors, so
    # routing purely by "came from stderr" mislabels a good run as failed.
    # Classify by CONTENT/severity instead, on both streams.
    #
    # Genuine-failure signatures (red) are kept narrow so normal banners are
    # never shown as errors; a real failure also carries a non-zero exit
    # code, which the GUI surfaces red independently.
    _ERROR_RE = re.compile(
        r'(?<![A-Za-z])(?:FATAL|ERROR)\s*:'          # "ERROR:" / "FATAL:"
        r'|traceback \(most recent call last\)'
        r'|segmentation fault|abort trap|bus error|core dumped'
        r'|terminate called|what\(\):|std::bad_alloc|std::runtime_error'
        r'|uncaught exception|assertion failed'
        r'|simulation failed',
        re.IGNORECASE)
    _WARN_RE = re.compile(
        r'(?<![A-Za-z])warn(?:ing)?\b'               # "WARNING" / "warn:"
        r'|conservative defaults|no usable gpu profile'
        r'|falling back|fall back to|fallback|deprecat'
        r'|spice error|spice\(',                     # engine SPICE diagnostics
        re.IGNORECASE)

    WATCHDOG_INTERVAL_S = 30.0   # child cputime sampling cadence
    STALL_CPU_DELTA_S = 0.5      # < this much CPU advance per interval = stalled

    def __init__(self, config_file, working_dir=None, config_dict=None):
        super().__init__()
        self.config_file = config_file
        self.working_dir = working_dir
        self.config_dict = config_dict  # Full config for engine env + CLI options
        self.process = None
        self._stop_requested = False

    # NOTE: radiation options are deliberately NOT duplicated as command-line
    # flags. config_parser.cpp reads the complete radiation section from the
    # YAML itself (enabled, use_hmatrix, use_gpu, view_factor_file, coupling
    # and ACA parameters, band_separated, gpu_devices, ...). The old
    # _build_radiation_args copied a subset onto the CLI "because they were
    # not in the YAML parser" -- that has been false for a long time, and the
    # CLI copy risked silently overriding the YAML whenever GUI and engine
    # defaults diverged. YAML is the single source of truth now.

    def _build_gpu_thermal_args(self):
        """Build command-line arguments for GPU thermal options (not in YAML parser)"""
        args = []
        if not self.config_dict:
            return args

        thermal = self.config_dict.get("thermal", {})
        if thermal.get("use_gpu", False):
            args.append("--use-gpu-thermal")

        return args

    def _build_generic_body_args(self):
        """Build command-line arguments for generic body configuration."""
        args = []
        if not self.config_dict:
            return args

        generic = self.config_dict.get("generic_body", {})
        config_file = generic.get("config_file", "")
        if config_file:
            args.extend(["--generic-body-config", str(config_file)])

        return args

    def run(self):
        """Execute the simulation and emit output"""
        try:
            executable = SimulationRunner.find_ptm_executable()
            if not executable:
                self.error_ready.emit("Could not find ptm_thermal_sim executable")
                self.finished.emit(-1)
                return

            # Build command with config file and additional CLI args for GPU/
            # generic-body options (radiation comes from the YAML -- see note
            # above _build_gpu_thermal_args).
            command = [str(executable), "--config", str(self.config_file)]
            command.extend(self._build_gpu_thermal_args())
            command.extend(self._build_generic_body_args())

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

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                bufsize=0,
                cwd=self.working_dir,
                env=env,
                start_new_session=True  # own process group: clean group kill
            )

            stdout_buffer = bytearray()
            stderr_buffer = bytearray()

            def drain_buffer(buf, is_stderr):
                parts = re.split(b'[\r\n]', buf)
                complete = parts[:-1]
                remainder = parts[-1] if parts else b""

                for seg in complete:
                    if not seg:
                        continue
                    decoded = seg.decode('utf-8', errors='replace').strip()
                    if not decoded:
                        continue

                    # Extract progress if available
                    progress = self._extract_progress(decoded)
                    if progress is not None:
                        self.progress_update.emit(progress)

                    # Route by severity/content, NOT by stream (see class
                    # docstring): the engine emits info/warning banners on
                    # stderr during successful runs.
                    self._emit_classified(decoded)

                buf.clear()
                buf.extend(remainder)

            # Stall watchdog state: sample the child's CPU clock every
            # WATCHDOG_INTERVAL_S. A process blocked in a file read (e.g. a
            # dehydrated cloud-synced file) accumulates no CPU time, which is
            # otherwise indistinguishable from long computation.
            next_watchdog = time.monotonic() + self.WATCHDOG_INTERVAL_S
            last_cputime = None
            stall_since = None

            # Pump both pipes
            while not self._stop_requested:
                if self.process.poll() is not None:
                    tail_out = self.process.stdout.read() or b""
                    tail_err = self.process.stderr.read() or b""
                    if tail_out:
                        stdout_buffer.extend(tail_out)
                    if tail_err:
                        stderr_buffer.extend(tail_err)
                    break

                now = time.monotonic()
                if now >= next_watchdog:
                    next_watchdog = now + self.WATCHDOG_INTERVAL_S
                    cpu = self._sample_cputime(self.process.pid)
                    if cpu is not None:
                        if (last_cputime is not None
                                and (cpu - last_cputime) < self.STALL_CPU_DELTA_S):
                            if stall_since is None:
                                stall_since = now - self.WATCHDOG_INTERVAL_S
                            mins = (now - stall_since) / 60.0
                            self.heartbeat.emit(
                                f"No progress for {mins:.1f} min (CPU idle) -- "
                                "likely blocked on file I/O (cloud-synced or "
                                "dataless file?)", True)
                        else:
                            if stall_since is not None:
                                self.heartbeat.emit(
                                    "Progress resumed (CPU active)", False)
                            stall_since = None
                        last_cputime = cpu

                readable, _, _ = select.select(
                    [self.process.stdout, self.process.stderr], [], [], 0.1
                )
                for stream in readable:
                    data = stream.read(4096)
                    if not data:
                        continue
                    if stream is self.process.stdout:
                        stdout_buffer.extend(data)
                        drain_buffer(stdout_buffer, is_stderr=False)
                    else:
                        stderr_buffer.extend(data)
                        drain_buffer(stderr_buffer, is_stderr=True)

            # Emit remaining content (classified, so a genuine error in the
            # final chunk still surfaces red).
            for buf, is_stderr in [(stdout_buffer, False), (stderr_buffer, True)]:
                if buf:
                    try:
                        tail = buf.decode('utf-8', errors='replace').strip()
                        if tail:
                            self._emit_classified(tail)
                    except Exception:
                        pass

            self.process.wait()
            self.finished.emit(self.process.returncode)

        except Exception as e:
            self.error_ready.emit(f"Execution error: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        """Stop the running simulation.

        The child is launched with start_new_session=True, so signal the
        whole process group: TERM first, KILL after 5 s. A process stuck in
        an uninterruptible cloud-file read may not die instantly even on
        KILL; the final wait is bounded so this never hangs the caller.
        """
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass  # bounded: do not hang the GUI on a stuck process

    def _signal_group(self, sig):
        """Send `sig` to the child's process group (fallback: child only)."""
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                if sig == signal.SIGKILL:
                    self.process.kill()
                else:
                    self.process.terminate()
            except Exception:
                pass

    @staticmethod
    def _sample_cputime(pid):
        """Child CPU time in seconds via ps (portable on macOS/Linux).
        Returns None if unavailable."""
        try:
            out = subprocess.run(
                ["ps", "-o", "cputime=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:
            return None
        if not out:
            return None
        # Format: [[dd-]hh:]mm:ss[.ff]
        try:
            days = 0
            if "-" in out:
                d, out = out.split("-", 1)
                days = int(d)
            parts = [float(p) for p in out.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, s = parts[-3], parts[-2], parts[-1]
            return days * 86400.0 + h * 3600.0 + m * 60.0 + s
        except (ValueError, IndexError):
            return None

    def classify_severity(self, text):
        """Classify a single output line as 'error' | 'warning' | 'info'
        by its CONTENT (works for stdout and stderr alike). Genuine failure
        signatures -> error (red); explicit warnings and known
        conservative-fallback banners -> warning (yellow); everything else,
        including the engine's [eq-seed]/[machine-profile]/[GPU Thermal]
        banners and progress lines -> info (white)."""
        if self._ERROR_RE.search(text):
            return "error"
        if self._WARN_RE.search(text):
            return "warning"
        return "info"

    # Back-compat alias (kept private-style for internal callers/tests).
    _classify_severity = classify_severity

    def _emit_classified(self, text):
        sev = self.classify_severity(text)
        if sev == "error":
            self.error_ready.emit(text)
        elif sev == "warning":
            self.warning_ready.emit(text)
        else:
            self.output_ready.emit(text)

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

    def _extract_progress(self, text):
        """Extract progress percentage from text"""
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            return float(match.group(1))
        return None


class SimulationRunner:
    """Helper class for locating and running PTM simulations"""

    @staticmethod
    def find_ptm_executable():
        """Locate the ptm_thermal_sim executable"""
        possible_paths = [
            Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_thermal_sim",
            Path("../build/bin/ptm_thermal_sim"),
            Path("../../build/bin/ptm_thermal_sim"),
            Path("./build/bin/ptm_thermal_sim"),
            Path("../bin/ptm_thermal_sim"),
            Path("./bin/ptm_thermal_sim"),
        ]

        for path in possible_paths:
            resolved = path.resolve()
            if resolved.exists():
                return resolved

        return None

    @staticmethod
    def find_converter_script():
        """Locate the VTK converter script"""
        possible_paths = [
            Path(__file__).parent.parent.parent / "src" / "python" / "vtk_converter_optimized.py",
            Path("../src/python/vtk_converter_optimized.py"),
            Path("../../src/python/vtk_converter_optimized.py"),
            Path("./src/python/vtk_converter_optimized.py"),
        ]

        for path in possible_paths:
            resolved = path.resolve()
            if resolved.exists():
                return resolved

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
        candidates = [
            Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_camera_sim",
            Path("../build/bin/ptm_camera_sim"),
            Path("../../build/bin/ptm_camera_sim"),
            Path("./build/bin/ptm_camera_sim"),
        ]
        for p in candidates:
            resolved = p.resolve()
            if resolved.exists():
                return resolved
        return None


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
