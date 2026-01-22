"""
Simulation Runner for PTM3D GUI
Handles simulation execution in background thread
"""

import subprocess
import sys
import re
import select
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal


class SimulationThread(QThread):
    """Thread for running PTM simulations without blocking GUI"""

    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    progress_update = pyqtSignal(float)
    finished = pyqtSignal(int)  # Exit code

    def __init__(self, config_file, working_dir=None, config_dict=None):
        super().__init__()
        self.config_file = config_file
        self.working_dir = working_dir
        self.config_dict = config_dict  # Full config for extracting radiation options
        self.process = None
        self._stop_requested = False

    def _build_radiation_args(self):
        """Build command-line arguments for radiation options (not in YAML parser)"""
        args = []
        if not self.config_dict:
            return args

        rad = self.config_dict.get("radiation", {})
        if not rad.get("enabled", False):
            return args

        # Enable radiation exchange
        args.append("--radiation-exchange")

        # H-matrix compression
        if rad.get("use_hmatrix", False):
            args.append("--rad-hmatrix")

        # GPU radiosity
        if rad.get("use_gpu", False):
            args.append("--rad-use-gpu")

        # Verbose output
        if rad.get("verbose", False):
            args.append("--rad-verbose")

        # View factor file
        vf_file = rad.get("view_factor_file", "")
        if vf_file:
            args.extend(["--rad-view-factor-file", str(vf_file)])

        # Coupling parameters
        max_iter = rad.get("max_coupling_iterations", 10)
        if max_iter != 10:
            args.extend(["--rad-coupling-iter", str(max_iter)])

        temp_tol = rad.get("coupling_temp_threshold", 0.5)
        if temp_tol != 0.5:
            args.extend(["--rad-coupling-temp-tol", str(temp_tol)])

        flux_tol = rad.get("coupling_flux_threshold", 0.1)
        if flux_tol != 0.1:
            args.extend(["--rad-coupling-flux-tol", str(flux_tol)])

        face_frac = rad.get("coupling_face_fraction", 0.01)
        if face_frac != 0.01:
            args.extend(["--rad-coupling-face-frac", str(face_frac)])

        # ACA parameters (if not using ACA, use SVD)
        if not rad.get("use_aca", True):
            args.append("--rad-use-svd")
        else:
            aca_tol = rad.get("aca_tolerance", 0.001)
            if aca_tol != 0.001:
                args.extend(["--rad-aca-tolerance", str(aca_tol)])
            aca_rank = rad.get("aca_max_rank", 50)
            if aca_rank != 50:
                args.extend(["--rad-aca-max-rank", str(aca_rank)])

        return args

    def _build_gpu_thermal_args(self):
        """Build command-line arguments for GPU thermal options (not in YAML parser)"""
        args = []
        if not self.config_dict:
            return args

        thermal = self.config_dict.get("thermal", {})
        if thermal.get("use_gpu", False):
            args.append("--use-gpu-thermal")

        return args

    def run(self):
        """Execute the simulation and emit output"""
        try:
            executable = SimulationRunner.find_ptm_executable()
            if not executable:
                self.error_ready.emit("Could not find ptm_thermal_sim executable")
                self.finished.emit(-1)
                return

            # Build command with config file and additional CLI args for radiation/GPU
            command = [str(executable), "--config", str(self.config_file)]
            command.extend(self._build_radiation_args())
            command.extend(self._build_gpu_thermal_args())

            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                bufsize=0,
                cwd=self.working_dir
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

                    # Route output
                    if is_stderr and self._is_progress_line(decoded):
                        self.output_ready.emit(decoded)
                    else:
                        signal = self.error_ready if is_stderr else self.output_ready
                        signal.emit(decoded)

                buf.clear()
                buf.extend(remainder)

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
                        tail = buf.decode('utf-8', errors='replace').strip()
                        if tail:
                            self.output_ready.emit(tail)
                    except Exception:
                        pass

            self.process.wait()
            self.finished.emit(self.process.returncode)

        except Exception as e:
            self.error_ready.emit(f"Execution error: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        """Stop the running simulation"""
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def _is_progress_line(self, text):
        """Check if line is a progress update"""
        tl = text.lower()
        return (('[' in text and ']' in text and '%' in text) or
                ('calculating' in tl and '%' in text) or
                ('thermal simulation' in tl and '%' in text) or
                ('progress' in tl and '%' in text))

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
