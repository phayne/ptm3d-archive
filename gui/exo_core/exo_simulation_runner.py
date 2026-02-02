"""
Simulation Runner for Exosphere GUI.
Executes ptm_exosphere_main in a background thread.
"""

import os
import re
import select
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class ExoSimulationThread(QThread):
    """Thread for running exosphere simulations without blocking GUI."""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished = pyqtSignal(int)  # exit code

    def __init__(self, config_manager, working_dir=None):
        super().__init__()
        self.config_manager = config_manager
        self.working_dir = working_dir
        self.process = None
        self._stop_requested = False

    def run(self):
        """Execute the exosphere simulation and emit output."""
        try:
            executable = ExoSimulationRunner.find_exosphere_executable()
            if not executable:
                self.error_ready.emit(
                    "Could not find ptm_exosphere_main executable. "
                    "Check that it was built (cmake --build . --target ptm_exosphere)."
                )
                self.finished.emit(-1)
                return

            command = [str(executable)] + self.config_manager.build_command_args()

            self.output_ready.emit(f"Running: {' '.join(command)}")

            env = os.environ.copy()
            env["PYTHON_EXECUTABLE"] = sys.executable

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                bufsize=0,
                cwd=self.working_dir,
                env=env,
            )

            stdout_buffer = bytearray()
            stderr_buffer = bytearray()

            def drain_buffer(buf, is_stderr):
                parts = re.split(b"[\r\n]", buf)
                complete = parts[:-1]
                remainder = parts[-1] if parts else b""

                for seg in complete:
                    if not seg:
                        continue
                    decoded = seg.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue

                    progress = self._extract_progress(decoded)
                    if progress is not None:
                        self.progress_update.emit(progress)

                    if is_stderr and self._is_progress_line(decoded):
                        self.output_ready.emit(decoded)
                    else:
                        signal = self.error_ready if is_stderr else self.output_ready
                        signal.emit(decoded)

                buf.clear()
                buf.extend(remainder)

            while not self._stop_requested:
                if self.process.poll() is not None:
                    tail_out = self.process.stdout.read() or b""
                    tail_err = self.process.stderr.read() or b""
                    if tail_out:
                        stdout_buffer.extend(tail_out)
                    if tail_err:
                        stderr_buffer.extend(tail_err)
                    break

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

            # Emit remaining content
            for buf, is_stderr in [(stdout_buffer, False), (stderr_buffer, True)]:
                if buf:
                    try:
                        tail = buf.decode("utf-8", errors="replace").strip()
                        if tail:
                            self.output_ready.emit(tail)
                    except Exception:
                        pass

            self.process.wait()
            self.finished.emit(self.process.returncode)

        except Exception as e:
            self.error_ready.emit(f"Execution error: {e}")
            self.finished.emit(-1)

    def stop(self):
        """Stop the running simulation."""
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def _is_progress_line(self, text):
        """Check if a stderr line is informational rather than an error."""
        tl = text.lower()
        if "[" in text and "]" in text and "%" in text:
            return True
        progress_keywords = [
            "calculating", "progress", "processing", "computing",
            "loading", "initializing", "writing", "saving", "reading",
            "starting", "finished", "complete", "iteration", "convergence",
            "epoch", "timestep", "exporting",
        ]
        return any(kw in tl for kw in progress_keywords) and "error" not in tl

    def _extract_progress(self, text):
        """Extract progress percentage from text."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if match:
            return float(match.group(1))
        # Also try epoch-based progress: "Epoch 500/1000"
        match = re.search(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                return 100.0 * current / total
        return None


class ExoSimulationRunner:
    """Helper class for locating the exosphere executable."""

    @staticmethod
    def find_exosphere_executable():
        """Locate the ptm_exosphere_main executable."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "build" / "bin" / "ptm_exosphere_main",
            Path("../build/bin/ptm_exosphere_main"),
            Path("../../build/bin/ptm_exosphere_main"),
            Path("./build/bin/ptm_exosphere_main"),
            Path("../bin/ptm_exosphere_main"),
            Path("./bin/ptm_exosphere_main"),
        ]

        for path in possible_paths:
            resolved = path.resolve()
            if resolved.exists():
                return resolved

        # Try PATH
        import shutil
        found = shutil.which("ptm_exosphere_main")
        if found:
            return Path(found)

        return None
