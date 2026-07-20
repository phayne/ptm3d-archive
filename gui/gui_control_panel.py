"""
Control Panel for PTM3D Configuration Builder
Dramatic top-mounted execution controls with retro-futuristic styling
Enhanced with multi-format export options
"""

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                             QLabel, QMessageBox, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

from gui_execution import ExecutionThread, SimulationRunner, VTKConverter


class ControlPanel(QWidget):
    """Top-mounted control panel with big red RUN button"""
    
    def __init__(self, config_manager, parent_window):
        super().__init__()
        self.config_manager = config_manager
        self.parent_window = parent_window
        self.simulation_thread = None
        self.converter_thread = None
        self.setObjectName("controlPanel")
        self.init_ui()
        
    def init_ui(self):
        """Initialize the control panel UI"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # === LEFT SIDE: BIG RED RUN BUTTON ===
        self.run_button = QPushButton("▶ RUN")
        self.run_button.setObjectName("bigRunButton")
        self.run_button.setMinimumSize(200, 80)
        self.run_button.setFont(QFont("Courier New", 24, QFont.Bold))
        self.run_button.clicked.connect(self.run_simulation)
        main_layout.addWidget(self.run_button)
        
        # === MIDDLE: STATUS AND OPTIONS ===
        middle_layout = QVBoxLayout()
        middle_layout.setSpacing(5)
        
        # Status line
        status_layout = QHBoxLayout()
        status_label_text = QLabel("STATUS:")
        status_label_text.setStyleSheet("color: #00ffff; font-weight: bold; font-size: 15px;")
        status_layout.addWidget(status_label_text)
        
        self.status_label = QLabel("READY")
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 18px;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        middle_layout.addLayout(status_layout)
        
        # Executable status
        exe_layout = QHBoxLayout()
        exe_label = QLabel("EXECUTABLE:")
        exe_label.setStyleSheet("color: #00ffff; font-size: 15px;")
        exe_layout.addWidget(exe_label)
        
        self.exe_status = QLabel("CHECKING...")
        self.exe_status.setStyleSheet("color: #ffff00; font-size: 14px;")
        exe_layout.addWidget(self.exe_status)
        
        exe_layout.addStretch()
        middle_layout.addLayout(exe_layout)
        
        # Auto-convert checkbox
        self.auto_convert_check = QCheckBox("Auto-convert after simulation")
        self.auto_convert_check.setChecked(True)
        self.auto_convert_check.setStyleSheet("color: #e0e0e0; font-size: 15px;")
        middle_layout.addWidget(self.auto_convert_check)
        
        middle_layout.addStretch()
        main_layout.addLayout(middle_layout, 1)
        
        # === CENTER: CONVERSION FORMAT OPTIONS ===
        format_group = QGroupBox("Conversion Output Formats")
        format_group.setObjectName("formatGroupBox")
        format_group.setStyleSheet("""
            QGroupBox#formatGroupBox {
                border: 2px solid #00ffff;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #00ffff;
                font-weight: bold;
            }
            QGroupBox#formatGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #00ffff;
            }
        """)
        
        format_layout = QVBoxLayout()
        format_layout.setSpacing(5)
        
        # VTK/VTU checkbox
        self.format_vtk_check = QCheckBox("VTK/VTU (ParaView/PyVista)")
        self.format_vtk_check.setChecked(True)
        self.format_vtk_check.setToolTip("VTK unstructured grid files for visualization in ParaView or PyVista")
        self.format_vtk_check.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        format_layout.addWidget(self.format_vtk_check)
        
        # PVD checkbox (always with VTK)
        self.format_pvd_check = QCheckBox("PVD Collection (time series)")
        self.format_pvd_check.setChecked(True)
        self.format_pvd_check.setToolTip("ParaView Data file linking all timesteps (automatically included with VTK)")
        self.format_pvd_check.setStyleSheet("color: #e0e0e0; font-size: 14px; padding-left: 20px;")
        self.format_pvd_check.setEnabled(False)  # Always on with VTK
        format_layout.addWidget(self.format_pvd_check)
        
        # NumPy checkbox
        self.format_numpy_check = QCheckBox("NumPy Arrays (.npz)")
        self.format_numpy_check.setChecked(False)
        self.format_numpy_check.setToolTip("NumPy compressed arrays for Python analysis (SciPy, Pandas, Matplotlib)")
        self.format_numpy_check.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        format_layout.addWidget(self.format_numpy_check)
        
        # HDF5 checkbox
        self.format_hdf5_check = QCheckBox("HDF5 (.h5)")
        self.format_hdf5_check.setChecked(False)
        self.format_hdf5_check.setToolTip("Hierarchical Data Format for large-scale analysis (requires h5py)")
        self.format_hdf5_check.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        format_layout.addWidget(self.format_hdf5_check)

        # NetCDF checkbox
        self.format_netcdf_check = QCheckBox("NetCDF (.nc)")
        self.format_netcdf_check.setChecked(False)
        self.format_netcdf_check.setToolTip("NetCDF export for analysis (requires netCDF4)")
        self.format_netcdf_check.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        format_layout.addWidget(self.format_netcdf_check)
        
        # Statistics checkbox
        self.include_statistics_check = QCheckBox("Include Statistics (min/max/avg)")
        self.include_statistics_check.setChecked(True)
        self.include_statistics_check.setToolTip("Export statistical summaries of data")
        self.include_statistics_check.setStyleSheet("color: #00ff00; font-size: 14px;")
        format_layout.addWidget(self.include_statistics_check)
        
        format_group.setLayout(format_layout)
        main_layout.addWidget(format_group)
        
        # Connect VTK checkbox to update PVD availability
        self.format_vtk_check.stateChanged.connect(self.update_format_dependencies)
        
        # === RIGHT SIDE: SECONDARY CONTROLS ===
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        
        # Stop button
        self.stop_button = QPushButton("■ STOP")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(35)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        right_layout.addWidget(self.stop_button)
        
        # Convert button
        self.convert_button = QPushButton("🔄 CONVERT")
        self.convert_button.setObjectName("convertButton")
        self.convert_button.setMinimumHeight(35)
        self.convert_button.clicked.connect(self.run_vtk_conversion)
        right_layout.addWidget(self.convert_button)
        
        main_layout.addLayout(right_layout)
        
        # Check executables on startup
        self.check_executables()
        
    def update_format_dependencies(self):
        """Update dependent format options when VTK is toggled"""
        vtk_enabled = self.format_vtk_check.isChecked()
        self.format_pvd_check.setEnabled(vtk_enabled)
        if vtk_enabled:
            self.format_pvd_check.setChecked(True)
        
    def check_executables(self):
        """Check if required executables exist"""
        try:
            exe_path = SimulationRunner.find_ptm_executable()
            if exe_path:
                self.exe_status.setText(f"✓ {exe_path.name}")
                self.exe_status.setStyleSheet("color: #00ff00; font-size: 14px;")
                self.exe_status.setToolTip(str(exe_path))
            else:
                self.exe_status.setText("✗ NOT FOUND - BUILD REQUIRED")
                self.exe_status.setStyleSheet("color: #ff0000; font-size: 14px;")
                self.run_button.setEnabled(False)
            
            # Check converter
            script_path = VTKConverter.find_converter_script()
            if not script_path:
                self.convert_button.setEnabled(False)
                
        except Exception as e:
            self.exe_status.setText(f"✗ ERROR: {str(e)}")
            self.exe_status.setStyleSheet("color: #ff0000; font-size: 14px;")
            self.run_button.setEnabled(False)
    
    def get_console_widget(self):
        """Return reference to console output widget (to be set by parent)"""
        return getattr(self.parent_window, 'console_output', None)

    def append_console(self, text, error=False):
        """Append text to console output if available"""
        # Route through ConsoleDock so whitespace and progress are handled correctly
        dock = getattr(self.parent_window, 'console_dock', None)
        if dock:
            dock.append_line(text, error=error)
    
    def get_selected_formats(self):
        """Get list of selected output formats"""
        formats = []
        if self.format_vtk_check.isChecked():
            formats.append('vtk')
        if self.format_numpy_check.isChecked():
            formats.append('numpy')
        if self.format_hdf5_check.isChecked():
            formats.append('hdf5')
        
        # If multiple formats selected, use 'all' for efficiency
        if len(formats) > 1:
            return 'all'
        elif len(formats) == 1:
            return formats[0]
        else:
            # Default to VTK if nothing selected
            return 'vtk'
    
    def run_simulation(self):
        """Run the PTM simulation"""
        # Check if config has unsaved changes
        if self.parent_window.unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Configuration has unsaved changes. Save before running?",
                QMessageBox.Save | QMessageBox.Ignore | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.parent_window.save_config()
                if self.parent_window.unsaved_changes:  # Save was cancelled
                    return
            elif reply == QMessageBox.Cancel:
                return
        
        # Check if we have a saved config file
        if not self.parent_window.current_file:
            QMessageBox.warning(
                self,
                "No Configuration File",
                "Please save the configuration before running the simulation."
            )
            return
        
        config_file = self.parent_window.current_file
        
        try:
            # Build command
            command = SimulationRunner.build_simulation_command(config_file)
            
            # Clear console and show command
            console = self.get_console_widget()
            if console:
                console.clear()
                # Reset progress tracking
                parent_console = self.parent_window.console_dock
                parent_console.last_line_was_progress = False
            
            self.append_console("=" * 70)
            self.append_console("INITIATING SIMULATION SEQUENCE")
            self.append_console("=" * 70)
            self.append_console(f"COMMAND: {' '.join(command)}")
            self.append_console(f"CONFIG:  {config_file}")
            self.append_console("=" * 70)
            self.append_console("")
            
            # Update UI state
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("RUNNING...")
            self.status_label.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 18px;")
            
            # Pulse animation for run button
            self.run_button.setText("⚡ RUNNING")
            
            # Create and start execution thread
            self.simulation_thread = ExecutionThread(command)
            self.simulation_thread.output_ready.connect(self.on_simulation_output)
            self.simulation_thread.error_ready.connect(
                lambda msg: self.append_console(msg, error=True)
            )
            self.simulation_thread.finished.connect(self.on_simulation_finished)
            self.simulation_thread.start()
            
        except Exception as e:
            self.append_console(f"SIMULATION LAUNCH FAILED: {e}", error=True)
            self.reset_ui_state()
            QMessageBox.critical(self, "Launch Error", f"Failed to start simulation:\n{str(e)}")
    
    @pyqtSlot(str)
    def on_simulation_output(self, line):
        """Handle simulation output"""
        # Plain text; ConsoleDock colors and preserves spaces
        self.append_console(line)

    @pyqtSlot(int)
    def on_simulation_finished(self, exit_code):
        """Handle simulation completion"""
        self.append_console("")
        self.append_console("=" * 70)
        
        if exit_code == 0:
            self.append_console("✓ SIMULATION COMPLETED SUCCESSFULLY")
            self.status_label.setText("COMPLETE")
            self.status_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 18px;")
            
            # Flash the run button green
            self.run_button.setStyleSheet(
                "background-color: #00ff00; color: #0a0e27; border: 3px solid #00ff00;"
            )
            
            # Auto-convert if enabled
            if self.auto_convert_check.isChecked():
                self.append_console("INITIATING AUTO-CONVERSION...")
                self.append_console("=" * 70)
                # Small delay for dramatic effect
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(1000, self.run_vtk_conversion)
            else:
                self.append_console("AUTO-CONVERSION DISABLED")
                self.append_console("=" * 70)
        else:
            self.append_console(f"✗ SIMULATION FAILED (EXIT CODE: {exit_code})", error=True)
            self.status_label.setText("FAILED")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 18px;")
            self.append_console("=" * 70)
            
            # Flash the run button red
            self.run_button.setStyleSheet(
                "background-color: #ff0000; color: #ffffff; border: 3px solid #ff0000;"
            )
        
        self.reset_ui_state()
    
    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation_thread:
            self.append_console("⚠ EMERGENCY STOP INITIATED", error=True)
            self.simulation_thread.stop()
            self.status_label.setText("STOPPED")
            self.status_label.setStyleSheet("color: #ff8800; font-weight: bold; font-size: 18px;")
        
        self.reset_ui_state()
    
    def run_vtk_conversion(self):
        """Run data conversion with selected formats"""
        # Get output directory from config
        config = self.config_manager.get_config()
        output_dir = Path(config["output"]["directory"])
        
        if not output_dir.exists():
            QMessageBox.warning(
                self,
                "Output Not Found",
                f"Output directory does not exist:\n{output_dir}\n\n"
                "Please run the simulation first."
            )
            return
        
        # Check for required data files
        mesh_file = output_dir / "mesh.ply"
        if not mesh_file.exists():
            QMessageBox.warning(
                self,
                "Data Not Found",
                f"Simulation output not found in:\n{output_dir}\n\n"
                "Please run the simulation first."
            )
            return
        
        # Validate format selection
        if not any([self.format_vtk_check.isChecked(), 
                    self.format_numpy_check.isChecked(),
                    self.format_hdf5_check.isChecked(),
                    self.format_netcdf_check.isChecked()]):
            QMessageBox.warning(
                self,
                "No Format Selected",
                "Please select at least one output format."
            )
            return
        
        # Determine output directory based on formats
        selected_format = self.get_selected_formats()
        if selected_format == 'all':
            conversion_output_dir = output_dir  # Use subdirectories
        else:
            conversion_output_dir = output_dir / selected_format
        
        try:
            # Build command with selected formats
            command = VTKConverter.build_converter_command(
                input_dir=output_dir,
                output_dir=conversion_output_dir,
                output_format=selected_format,
                parallel=0,  # Auto-detect
                no_statistics=not self.include_statistics_check.isChecked()
            )
            
            # Show command
            self.append_console("")
            self.append_console("=" * 70)
            self.append_console("INITIATING DATA CONVERSION")
            self.append_console("=" * 70)
            
            # Display selected formats
            formats = []
            if self.format_vtk_check.isChecked():
                formats.append("VTK/VTU + PVD")
            if self.format_numpy_check.isChecked():
                formats.append("NumPy (.npz)")
            if self.format_hdf5_check.isChecked():
                formats.append("HDF5 (.h5)")
            if self.format_netcdf_check.isChecked():
                formats.append("NetCDF (.nc)")
            
            self.append_console(f"OUTPUT FORMATS: {', '.join(formats)}")
            self.append_console(f"STATISTICS: {'ENABLED' if self.include_statistics_check.isChecked() else 'DISABLED'}")
            self.append_console(f"COMMAND: {' '.join(command)}")
            self.append_console(f"INPUT:   {output_dir}")
            self.append_console(f"OUTPUT:  {conversion_output_dir}")
            self.append_console("=" * 70)
            self.append_console("")
            
            # Update UI
            self.convert_button.setEnabled(False)
            self.convert_button.setText("⚡ CONVERTING")
            
            # Create and start conversion thread
            self.converter_thread = ExecutionThread(command)
            self.converter_thread.output_ready.connect(self.on_converter_output)
            self.converter_thread.error_ready.connect(
                lambda msg: self.append_console(msg, error=True)
            )
            self.converter_thread.finished.connect(self.on_converter_finished)
            self.converter_thread.start()
            
        except Exception as e:
            self.append_console(f"CONVERSION LAUNCH FAILED: {e}", error=True)
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 CONVERT")
            QMessageBox.critical(self, "Conversion Error", f"Failed to start conversion:\n{str(e)}")
    
    @pyqtSlot(str)
    def on_converter_output(self, line):
        self.append_console(line)
    
    @pyqtSlot(int)
    def on_converter_finished(self, exit_code):
        """Handle converter completion"""
        self.append_console("")
        self.append_console("=" * 70)
        
        if exit_code == 0:
            self.append_console("✓ DATA CONVERSION COMPLETED SUCCESSFULLY")
            config = self.config_manager.get_config()
            output_dir = Path(config["output"]["directory"])
            
            # Show output locations for each format
            if self.format_vtk_check.isChecked():
                vtk_dir = output_dir / "vtk" if self.get_selected_formats() == 'all' else output_dir
                self.append_console(f"VTK FILES: {vtk_dir}")
            
            if self.format_numpy_check.isChecked():
                numpy_dir = output_dir / "numpy" if self.get_selected_formats() == 'all' else output_dir
                self.append_console(f"NUMPY FILES: {numpy_dir}")
            
            if self.format_hdf5_check.isChecked():
                hdf5_dir = output_dir / "hdf5" if self.get_selected_formats() == 'all' else output_dir
                self.append_console(f"HDF5 FILES: {hdf5_dir}")

            # If NetCDF selected, run NetCDF conversion using PVD or HDF5
            if self.format_netcdf_check.isChecked():
                # Determine PVD path from VTK output
                fmt = self.get_selected_formats()
                vtk_dir = output_dir / "vtk" if fmt == 'all' else output_dir
                pvd_path = vtk_dir / "statistics.pvd"
                netcdf_dir = output_dir / ("netcdf" if fmt == 'all' else "netcdf")
                netcdf_dir.mkdir(exist_ok=True)
                netcdf_file = netcdf_dir / "statistics.nc"
                try:
                    from gui_execution import NetCDFConverter
                    cmd = NetCDFConverter.build_converter_command(pvd_path, netcdf_file, format_hint='pvd')
                    self.append_console(f"NETCDF CMD: {' '.join(cmd)}")
                    self.append_console(f"NETCDF INPUT: {pvd_path}")
                    self.append_console(f"NETCDF OUTPUT: {netcdf_file}")
                    # Run NetCDF conversion synchronously in a new thread
                    self.converter_thread = ExecutionThread(cmd)
                    self.converter_thread.output_ready.connect(self.on_converter_output)
                    self.converter_thread.error_ready.connect(
                        lambda msg: self.append_console(msg, error=True)
                    )
                    self.converter_thread.finished.connect(lambda code: self.append_console(
                        f"{'✓' if code == 0 else '✗'} NETCDF CONVERSION {'COMPLETED' if code == 0 else 'FAILED'}"
                    ))
                    self.converter_thread.start()
                except Exception as e:
                    self.append_console(f"NETCDF CONVERSION LAUNCH FAILED: {e}", error=True)
        else:
            self.append_console(f"✗ DATA CONVERSION FAILED (EXIT CODE: {exit_code})", error=True)
        
        self.append_console("=" * 70)
        self.convert_button.setEnabled(True)
        self.convert_button.setText("🔄 CONVERT")
    
    def reset_ui_state(self):
        """Reset UI to ready state"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_button.setText("▶ RUN")
        # Reset button style
        self.run_button.setStyleSheet("")
        if self.status_label.text() in ["RUNNING...", "STOPPED"]:
            self.status_label.setText("READY")
            self.status_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 18px;")