"""
Execution module for running PTM simulations and VTK conversion
Enhanced to support multiple output formats
"""

import subprocess
import os
import sys
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import select


class ExecutionThread(QThread):
    """Thread for running external processes without blocking GUI"""
    
    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    finished = pyqtSignal(int)  # Exit code
    
    def __init__(self, command, working_dir=None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        
    def run(self):
        """Execute the command and emit output"""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,  # Read bytes
                bufsize=0,
                cwd=self.working_dir,
                shell=isinstance(self.command, str)
            )

            # Use bytearray so we can mutate in-place
            stdout_buffer = bytearray()
            stderr_buffer = bytearray()

            def is_progress_line(s: str) -> bool:
                s_low = s.lower()
                return (('[' in s and ']' in s and '%' in s) or
                        ('calculating' in s_low and '%' in s) or
                        ('thermal simulation' in s_low and '%' in s) or
                        ('progress' in s_low and '%' in s))

            def drain_buffer(buf: bytearray, is_stderr: bool):
                # Split on either \n or \r (progress updates come via \r)
                import re
                parts = re.split(b'[\r\n]', buf)
                complete = parts[:-1]
                remainder = parts[-1] if parts else b""
                for seg in complete:
                    if not seg:
                        continue
                    decoded = seg.decode('utf-8', errors='replace').strip()
                    if not decoded:
                        continue
                    # Progress printed to stderr should be treated as normal output
                    if is_stderr and is_progress_line(decoded):
                        self.output_ready.emit(decoded)
                    else:
                        (self.error_ready.emit if is_stderr else self.output_ready.emit)(decoded)
                # keep only the trailing remainder
                buf.clear()
                buf.extend(remainder)

            # Pump both pipes
            while True:
                if self.process.poll() is not None:
                    # Drain remaining data
                    tail_out = self.process.stdout.read() or b""
                    tail_err = self.process.stderr.read() or b""
                    if tail_out:
                        stdout_buffer.extend(tail_out)
                    if tail_err:
                        stderr_buffer.extend(tail_err)
                    break

                readable, _, _ = select.select([self.process.stdout, self.process.stderr], [], [], 0.1)
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

            # Emit any remaining partial line
            if stderr_buffer:
                try:
                    tail = stderr_buffer.decode('utf-8', errors='replace').strip()
                    if tail:
                        self.output_ready.emit(tail if is_progress_line(tail) else tail)
                except Exception:
                    pass
            if stdout_buffer:
                try:
                    tail = stdout_buffer.decode('utf-8', errors='replace').strip()
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
        """Stop the running process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


class SimulationRunner:
    """Helper class for running PTM simulations"""
    
    @staticmethod
    def find_ptm_executable():
        """Locate the ptm_thermal_sim executable"""
        # Common locations to check
        possible_paths = [
            Path("../build/bin/ptm_thermal_sim"),
            Path("../../build/bin/ptm_thermal_sim"),
            Path("./build/bin/ptm_thermal_sim"),
            Path("../bin/ptm_thermal_sim"),
            Path("./bin/ptm_thermal_sim"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.resolve()
        
        return None
    
    @staticmethod
    def build_simulation_command(config_file):
        """Build the command to run the simulation"""
        executable = SimulationRunner.find_ptm_executable()
        
        if not executable:
            raise FileNotFoundError(
                "Could not find ptm_thermal_sim executable.\n"
                "Please ensure the project is built:\n"
                "  cd ../build\n"
                "  make"
            )
        
        config_path = Path(config_file).resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        return [str(executable), "--config", str(config_path)]


class VTKConverter:
    """Helper class for running VTK conversion with multiple format support"""
    
    @staticmethod
    def find_converter_script():
        """Locate the vtk_converter_optimized.py script"""
        possible_paths = [
            Path("../src/python/vtk_converter_optimized.py"),
            Path("../../src/python/vtk_converter_optimized.py"),
            Path("./src/python/vtk_converter_optimized.py"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.resolve()
        
        return None
    
    @staticmethod
    def build_converter_command(input_dir, output_dir, output_format='vtk',
                                timesteps=None, parallel=0, no_statistics=False):
        """
        Build the command to run VTK conversion with format selection.
        
        Args:
            input_dir: Input directory containing binary data
            output_dir: Output directory for converted files
            output_format: Output format - 'vtk', 'numpy', 'hdf5', or 'all'
            timesteps: Optional timestep specification (e.g., "0-10,20")
            parallel: Number of parallel processes (0=auto, only for VTK)
            no_statistics: Skip statistics conversion
        
        Returns:
            List of command arguments
        """
        script = VTKConverter.find_converter_script()
        
        if not script:
            raise FileNotFoundError(
                "Could not find vtk_converter_optimized.py script.\n"
                "Expected location: ../src/python/vtk_converter_optimized.py"
            )
        
        # Get Python executable (prefer conda environment)
        python_exe = sys.executable
        
        # Build command
        cmd = [
            python_exe,
            str(script),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--format", output_format
        ]
        
        if timesteps:
            cmd.extend(["--timesteps", timesteps])
        
        if parallel > 0:
            cmd.extend(["--parallel", str(parallel)])
        
        if no_statistics:
            cmd.append("--no-statistics")
        
        return cmd


class NetCDFConverter:
    """Helper class to run NetCDF conversion from VTU/PVD or HDF5"""

    @staticmethod
    def find_converter_script():
        possible_paths = [
            Path("../src/python/netcdf_converter.py"),
            Path("../../src/python/netcdf_converter.py"),
            Path("./src/python/netcdf_converter.py"),
        ]
        for path in possible_paths:
            if path.exists():
                return path.resolve()
        return None

    @staticmethod
    def build_converter_command(input_path, output_nc, format_hint='auto'):
        script = NetCDFConverter.find_converter_script()
        if not script:
            raise FileNotFoundError(
                "Could not find netcdf_converter.py script.\n"
                "Expected location: ../src/python/netcdf_converter.py"
            )
        python_exe = sys.executable
        cmd = [
            python_exe,
            str(script),
            "--input", str(input_path),
            "--output", str(output_nc),
            "--format", format_hint,
        ]
        return cmd