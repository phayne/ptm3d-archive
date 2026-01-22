"""
Visualization Panel for PTM3D GUI
3D mesh visualization using PyVista and 2D plots using Matplotlib
Enhanced with face selection via click or lat/lon input, and data export
"""

import numpy as np
from pathlib import Path
import struct
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFrame, QLineEdit, QFileDialog,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D visualization disabled.")

# Matplotlib for 2D plots
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. 2D plotting disabled.")


class PyVistaWidget(QWidget):
    """3D visualization widget using PyVista"""

    face_selected = pyqtSignal(int, dict)  # face_id, data dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.current_data = None
        self.current_field = "temperature"
        self.current_timestep = 0
        self.results_dir = None
        self.timesteps = []
        self.mesh_data = None

        # Hover highlighting state
        self.hover_face_id = None
        self.hover_highlight_actor = None
        self.selected_face_id = None

        # Color scale state
        self.fixed_color_scale = True
        self.color_range_min = None
        self.color_range_max = None
        self.global_data_range = None  # Will be computed from all timesteps

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            # Create PyVista Qt interactor
            self.plotter = QtInteractor(self)
            self.plotter.set_background('#1e1e1e')
            self.plotter.add_axes()
            layout.addWidget(self.plotter.interactor, 1)

            # Face picking state
            self.picking_enabled = False
            self.picking_active = False

            # Add control buttons overlay at top
            controls_layout = QHBoxLayout()
            controls_layout.setContentsMargins(5, 5, 5, 0)

            # Select Face toggle button
            self.select_face_btn = QPushButton("Select Face Mode")
            self.select_face_btn.setCheckable(True)
            self.select_face_btn.setMaximumWidth(140)
            self.select_face_btn.setStyleSheet("""
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #606060;
                    border-radius: 4px;
                    padding: 5px 10px;
                }
                QPushButton:checked {
                    background-color: #ff9800;
                    color: black;
                    border: 1px solid #ffb74d;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:checked:hover {
                    background-color: #ffa726;
                }
            """)
            self.select_face_btn.clicked.connect(self.toggle_face_selection)
            controls_layout.addWidget(self.select_face_btn)

            # Status label
            self.pick_status_label = QLabel("")
            self.pick_status_label.setStyleSheet("color: #888888; font-size: 11px;")
            controls_layout.addWidget(self.pick_status_label)

            controls_layout.addStretch()

            # Color scale controls
            self.fixed_scale_check = QCheckBox("Fixed Scale")
            self.fixed_scale_check.setChecked(True)
            self.fixed_scale_check.setStyleSheet("color: #e0e0e0; font-size: 11px;")
            self.fixed_scale_check.stateChanged.connect(self.on_fixed_scale_changed)
            controls_layout.addWidget(self.fixed_scale_check)

            # Min value input
            self.scale_min_label = QLabel("Min:")
            self.scale_min_label.setStyleSheet("color: #888888; font-size: 11px;")
            self.scale_min_label.setVisible(False)
            controls_layout.addWidget(self.scale_min_label)

            self.scale_min_input = QDoubleSpinBox()
            self.scale_min_input.setRange(-1000, 10000)
            self.scale_min_input.setDecimals(1)
            self.scale_min_input.setValue(0)
            self.scale_min_input.setMaximumWidth(70)
            self.scale_min_input.setVisible(False)
            self.scale_min_input.valueChanged.connect(self.on_color_range_changed)
            controls_layout.addWidget(self.scale_min_input)

            # Max value input
            self.scale_max_label = QLabel("Max:")
            self.scale_max_label.setStyleSheet("color: #888888; font-size: 11px;")
            self.scale_max_label.setVisible(False)
            controls_layout.addWidget(self.scale_max_label)

            self.scale_max_input = QDoubleSpinBox()
            self.scale_max_input.setRange(-1000, 10000)
            self.scale_max_input.setDecimals(1)
            self.scale_max_input.setValue(400)
            self.scale_max_input.setMaximumWidth(70)
            self.scale_max_input.setVisible(False)
            self.scale_max_input.valueChanged.connect(self.on_color_range_changed)
            controls_layout.addWidget(self.scale_max_input)

            layout.insertLayout(0, controls_layout)

        else:
            # Fallback message
            label = QLabel("PyVista not available.\nInstall with: pip install pyvista pyvistaqt")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #ff9800; font-size: 14px;")
            layout.addWidget(label)

    def load_results(self, results_dir):
        """Load simulation results from directory"""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            return False

        # Reset color range when loading new results
        self.global_data_range = None

        # Load mesh
        mesh_file = self.results_dir / "mesh.ply"
        if mesh_file.exists():
            self.load_mesh(mesh_file)

        # Load timesteps
        time_file = self.results_dir / "time_steps.txt"
        if time_file.exists():
            self.load_timesteps(time_file)

        # Load initial data
        if len(self.timesteps) > 0:
            self.load_timestep_data(0)

        return True

    def load_mesh(self, mesh_file):
        """Load mesh from PLY file"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            # Read PLY mesh data
            vertices, faces, normals, latitudes, longitudes = self.read_binary_mesh(mesh_file)

            # Create PyVista mesh
            n_faces = len(faces)
            cells = np.column_stack([
                np.full(n_faces, 3, dtype=np.int64),
                faces
            ]).ravel()

            self.mesh = pv.PolyData(vertices, cells)

            # Store face metadata
            self.mesh['latitude'] = latitudes
            self.mesh['longitude'] = longitudes

            # Store for later use
            self.mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'latitudes': latitudes,
                'longitudes': longitudes
            }

            self.update_display()

        except Exception as e:
            print(f"Error loading mesh: {e}")

    def read_binary_mesh(self, mesh_file):
        """Read binary PLY mesh file"""
        with open(mesh_file, 'rb') as f:
            # Parse header
            header = self.parse_ply_header(f)

            is_little_endian = (header['format'] == 'binary_little_endian')
            endian = '<' if is_little_endian else '>'

            # Read vertices
            vertex_dtype = []
            for prop_type, prop_name in header['vertex_props']:
                if prop_type == 'float':
                    vertex_dtype.append((prop_name, f'{endian}f4'))
                elif prop_type == 'double':
                    vertex_dtype.append((prop_name, f'{endian}f8'))

            vertex_dtype = np.dtype(vertex_dtype)
            vertex_buffer = f.read(header['num_vertices'] * vertex_dtype.itemsize)
            vertex_array = np.frombuffer(vertex_buffer, dtype=vertex_dtype)

            vertices = np.empty((header['num_vertices'], 3), dtype=np.float32)
            vertices[:, 0] = vertex_array['x']
            vertices[:, 1] = vertex_array['y']
            vertices[:, 2] = vertex_array['z']

            normals = None
            if 'nx' in vertex_array.dtype.names:
                normals = np.empty((header['num_vertices'], 3), dtype=np.float32)
                normals[:, 0] = vertex_array['nx']
                normals[:, 1] = vertex_array['ny']
                normals[:, 2] = vertex_array['nz']

            # Read faces
            faces = np.empty((header['num_faces'], 3), dtype=np.int32)
            latitudes = np.empty(header['num_faces'], dtype=np.float32)
            longitudes = np.empty(header['num_faces'], dtype=np.float32)

            for i in range(header['num_faces']):
                count_byte = f.read(1)
                idx_data = f.read(12)
                faces[i] = struct.unpack(f'{endian}III', idx_data)
                lat_data = f.read(4)
                latitudes[i] = struct.unpack(f'{endian}f', lat_data)[0]
                lon_data = f.read(4)
                longitudes[i] = struct.unpack(f'{endian}f', lon_data)[0]

        return vertices, faces, normals, latitudes, longitudes

    def parse_ply_header(self, f):
        """Parse PLY file header"""
        header = {
            'format': 'ascii',
            'num_vertices': 0,
            'num_faces': 0,
            'vertex_props': [],
            'face_props': []
        }
        current_element = None

        while True:
            line = f.readline()
            if isinstance(line, bytes):
                line = line.decode('ascii')
            line = line.strip()

            if line == 'end_header':
                break

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'format':
                header['format'] = parts[1]
            elif parts[0] == 'element':
                current_element = parts[1]
                if current_element == 'vertex':
                    header['num_vertices'] = int(parts[2])
                elif current_element == 'face':
                    header['num_faces'] = int(parts[2])
            elif parts[0] == 'property':
                if current_element == 'vertex' and len(parts) >= 3:
                    header['vertex_props'].append((parts[1], parts[2]))

        return header

    def load_timesteps(self, time_file):
        """Load timestep metadata"""
        self.timesteps = []
        with open(time_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.timesteps.append({
                        'time': float(parts[0]),
                        'label': parts[1]
                    })

    def load_timestep_data(self, timestep_index):
        """Load data for a specific timestep"""
        if not self.results_dir or not self.mesh:
            return

        self.current_timestep = timestep_index
        n_faces = self.mesh.n_cells

        # Load requested field
        data = self.load_field_data(self.current_field, timestep_index, n_faces)
        if data is not None:
            self.current_data = data
            self.mesh[self.current_field] = data
            self.update_display()

    def load_field_data(self, field_name, timestep_index, expected_count):
        """Load a specific data field for a timestep"""
        field_dirs = {
            'temperature': 'temperature',
            'direct_solar': 'direct_solar',
            'illumination': 'illumination',
            'primary_reflected': 'primary_reflected',
            'primary_ir': 'primary_ir'
        }

        dir_name = field_dirs.get(field_name, field_name)
        data_dir = self.results_dir / dir_name

        if not data_dir.exists():
            return None

        file_path = data_dir / f"{dir_name}_{timestep_index:04d}.bin"
        if not file_path.exists():
            return None

        return self.read_binary_data(file_path, expected_count)

    def read_binary_data(self, file_path, expected_count):
        """Read binary data file"""
        try:
            with open(file_path, 'rb') as f:
                # Try header format first
                header = f.read(4)
                if len(header) == 4:
                    count = struct.unpack('<I', header)[0]
                    expected_size = count * 4

                    # Validate
                    f.seek(0, 2)
                    file_size = f.tell()

                    if file_size == 4 + expected_size and count == expected_count:
                        # Header format
                        f.seek(4)
                        data = np.frombuffer(f.read(), dtype='<f4')
                        return data
                    else:
                        # Try headerless format
                        f.seek(0)
                        data = np.frombuffer(f.read(), dtype='<f4')
                        if len(data) == expected_count:
                            return data

            return np.zeros(expected_count, dtype=np.float32)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return np.zeros(expected_count, dtype=np.float32)

    def update_display(self):
        """Update the 3D display"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            return

        self.plotter.clear()

        # Determine scalar field and colormap
        scalar_field = self.current_field
        if scalar_field in self.mesh.array_names:
            scalars = scalar_field
        else:
            scalars = None

        # Choose colormap based on field
        cmap = 'plasma' if 'temperature' in self.current_field.lower() else 'viridis'

        # Determine color range (clim)
        clim = None
        if self.fixed_color_scale:
            # Use global range across all timesteps
            if self.global_data_range is None:
                self.global_data_range = self.compute_global_data_range()
            if self.global_data_range is not None:
                clim = self.global_data_range
        else:
            # Use manual range
            if self.color_range_min is not None and self.color_range_max is not None:
                clim = (self.color_range_min, self.color_range_max)

        # Add mesh to plotter with horizontal colorbar at bottom
        mesh_kwargs = {
            'scalars': scalars,
            'cmap': cmap,
            'show_edges': False,
            'lighting': True,
            'smooth_shading': True,
            'pickable': True,
            'name': 'main_mesh',
            'scalar_bar_args': {
                'title': self.get_field_label(),
                'color': 'white',
                'title_font_size': 12,
                'label_font_size': 10,
                'vertical': False,
                'position_x': 0.2,
                'position_y': 0.02,
                'width': 0.6,
                'height': 0.06,
                'n_labels': 5,
                'fmt': '%.1f'
            }
        }
        if clim is not None:
            mesh_kwargs['clim'] = clim

        self.plotter.add_mesh(self.mesh, **mesh_kwargs)

        # Re-add selection highlight if there was one
        if self.selected_face_id is not None:
            self.update_selection_highlight(self.selected_face_id)

        # Add timestamp label in upper-right corner (smaller font)
        if self.timesteps and 0 <= self.current_timestep < len(self.timesteps):
            ts_info = self.timesteps[self.current_timestep]
            ts_label = ts_info.get('label', f"Timestep {self.current_timestep}")
            self.plotter.add_text(
                ts_label,
                position='upper_right',
                font_size=9,
                color='white',
                name='timestamp_label'
            )

        # Add light source indicator (optional)
        if hasattr(self, 'sun_position'):
            self.plotter.add_point_labels(
                [self.sun_position], ['Sun'],
                point_size=20, font_size=12, text_color='yellow'
            )

    def toggle_face_selection(self, checked):
        """Toggle face selection mode on/off"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            self.select_face_btn.setChecked(False)
            self.pick_status_label.setText("Load results first")
            return

        if checked:
            self.enable_hover_picking()
            if self.picking_active:
                self.pick_status_label.setText("Hover to highlight, click to select")
            else:
                self.pick_status_label.setText("Selection mode failed")
                self.select_face_btn.setChecked(False)
        else:
            self.disable_hover_picking()
            self.pick_status_label.setText("")
            # Remove any hover highlight
            self.clear_hover_highlight()

    def enable_hover_picking(self):
        """Enable hover-based face picking with highlighting"""
        if not PYVISTA_AVAILABLE or not self.mesh:
            return

        try:
            # Get the VTK interactor
            iren = self.plotter.iren.interactor

            # Connect mouse move event for hover highlighting
            iren.AddObserver('MouseMoveEvent', self.on_mouse_move)
            # Connect left button press for selection
            iren.AddObserver('LeftButtonPressEvent', self.on_left_click)

            self.picking_active = True
            self.picking_enabled = True

        except Exception as e:
            print(f"Hover picking setup failed: {e}")
            self.picking_active = False

    def disable_hover_picking(self):
        """Disable hover picking mode"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            # Note: VTK observers are tricky to remove without IDs
            # We'll just set picking_active to False to ignore events
            self.picking_active = False
            self.clear_hover_highlight()
        except Exception as e:
            print(f"Error disabling hover picking: {e}")

    def on_mouse_move(self, obj, event):
        """Handle mouse move for hover highlighting"""
        if not self.picking_active or not self.mesh:
            return

        try:
            # Get mouse position
            iren = self.plotter.iren.interactor
            x, y = iren.GetEventPosition()

            # Use VTK cell picker
            picker = self.plotter.iren.get_picker()
            if picker is None:
                import vtk
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.005)

            # Perform pick
            renderer = self.plotter.renderer
            picker.Pick(x, y, 0, renderer)

            cell_id = picker.GetCellId()

            if cell_id >= 0 and cell_id != self.hover_face_id:
                # New face under cursor - update highlight
                self.hover_face_id = cell_id
                self.update_hover_highlight(cell_id)

                # Update status label with face info
                lat = self.mesh['latitude'][cell_id] if 'latitude' in self.mesh.array_names else 0
                lon = self.mesh['longitude'][cell_id] if 'longitude' in self.mesh.array_names else 0
                self.pick_status_label.setText(f"Face {cell_id} ({lat:.2f}°, {lon:.2f}°) - Click to select")

            elif cell_id < 0 and self.hover_face_id is not None:
                # No face under cursor - clear highlight
                self.clear_hover_highlight()
                self.hover_face_id = None
                self.pick_status_label.setText("Hover to highlight, click to select")

        except Exception as e:
            # Silently ignore errors during mouse move to avoid spam
            pass

    def on_left_click(self, obj, event):
        """Handle left click for face selection"""
        if not self.picking_active or not self.mesh:
            return

        try:
            # Get mouse position
            iren = self.plotter.iren.interactor
            x, y = iren.GetEventPosition()

            # Use VTK cell picker
            import vtk
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)

            renderer = self.plotter.renderer
            picker.Pick(x, y, 0, renderer)

            cell_id = picker.GetCellId()

            if cell_id >= 0:
                self.selected_face_id = cell_id
                self._emit_face_selection(cell_id)
                # Keep hover highlight but add selection indicator
                self.update_selection_highlight(cell_id)

        except Exception as e:
            print(f"Click selection error: {e}")

    def update_hover_highlight(self, face_id):
        """Update the hover highlight to show a specific face"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        try:
            # Remove old hover highlight if exists
            if self.hover_highlight_actor is not None:
                self.plotter.remove_actor(self.hover_highlight_actor, render=False)
                self.hover_highlight_actor = None

            # Extract the single cell and create a highlight mesh
            cell = self.mesh.extract_cells([face_id])
            if cell.n_cells > 0:
                # Add as wireframe with bright color
                self.hover_highlight_actor = self.plotter.add_mesh(
                    cell,
                    color='cyan',
                    style='wireframe',
                    line_width=3,
                    render=False,
                    name='hover_highlight',
                    pickable=False
                )
                self.plotter.render()

        except Exception as e:
            print(f"Hover highlight error: {e}")

    def clear_hover_highlight(self):
        """Clear the hover highlight"""
        if not PYVISTA_AVAILABLE:
            return

        try:
            if self.hover_highlight_actor is not None:
                self.plotter.remove_actor(self.hover_highlight_actor, render=False)
                self.hover_highlight_actor = None
                self.plotter.render()
        except Exception:
            pass

    def update_selection_highlight(self, face_id):
        """Update the permanent selection highlight"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        try:
            # Remove old selection marker
            try:
                self.plotter.remove_actor('selected_face_marker', render=False)
            except Exception:
                pass

            # Extract the cell and highlight it
            cell = self.mesh.extract_cells([face_id])
            if cell.n_cells > 0:
                # Add solid highlight with different color from hover
                self.plotter.add_mesh(
                    cell,
                    color='yellow',
                    style='surface',
                    opacity=0.7,
                    render=False,
                    name='selected_face_marker',
                    pickable=False
                )

                # Also add face center marker
                if self.mesh_data is not None:
                    faces = self.mesh_data['faces']
                    vertices = self.mesh_data['vertices']
                    if face_id < len(faces):
                        face_verts = vertices[faces[face_id]]
                        center = face_verts.mean(axis=0)
                        self.plotter.add_points(
                            np.array([center]),
                            color='yellow',
                            point_size=12,
                            render=False,
                            name='selected_center_marker'
                        )

                self.plotter.render()

        except Exception as e:
            print(f"Selection highlight error: {e}")

    def enable_face_picking(self):
        """Enable face picking on the mesh via left-click (legacy method)"""
        # Redirect to hover-based picking
        self.enable_hover_picking()

    def disable_face_picking(self):
        """Disable face picking mode (legacy method)"""
        self.disable_hover_picking()

    def on_fixed_scale_changed(self, state):
        """Handle fixed scale checkbox change"""
        self.fixed_color_scale = (state == Qt.CheckState.Checked.value)

        # Show/hide manual range inputs
        show_inputs = not self.fixed_color_scale
        self.scale_min_label.setVisible(show_inputs)
        self.scale_min_input.setVisible(show_inputs)
        self.scale_max_label.setVisible(show_inputs)
        self.scale_max_input.setVisible(show_inputs)

        # If switching to manual, populate with current range
        if show_inputs and self.global_data_range is not None:
            self.scale_min_input.blockSignals(True)
            self.scale_max_input.blockSignals(True)
            self.scale_min_input.setValue(self.global_data_range[0])
            self.scale_max_input.setValue(self.global_data_range[1])
            self.scale_min_input.blockSignals(False)
            self.scale_max_input.blockSignals(False)

        # Update display
        self.update_display()

    def on_color_range_changed(self):
        """Handle manual color range change"""
        if not self.fixed_color_scale:
            self.color_range_min = self.scale_min_input.value()
            self.color_range_max = self.scale_max_input.value()
            self.update_display()

    def compute_global_data_range(self):
        """Compute the global min/max across all timesteps for the current field"""
        if not self.results_dir:
            return None

        field_dirs = {
            'temperature': 'temperature',
            'direct_solar': 'direct_solar',
            'illumination': 'illumination',
            'primary_reflected': 'primary_reflected',
            'primary_ir': 'primary_ir'
        }

        dir_name = field_dirs.get(self.current_field, self.current_field)
        data_dir = self.results_dir / dir_name

        if not data_dir.exists():
            return None

        global_min = float('inf')
        global_max = float('-inf')

        # Sample a subset of files to compute range (for performance)
        files = sorted(data_dir.glob(f"{dir_name}_*.bin"))
        # Sample every Nth file if there are many
        step = max(1, len(files) // 10)
        sampled_files = files[::step]

        for file_path in sampled_files:
            data = self.read_binary_data(file_path, self.mesh.n_cells if self.mesh else 0)
            if data is not None and len(data) > 0:
                global_min = min(global_min, np.min(data))
                global_max = max(global_max, np.max(data))

        if global_min < float('inf') and global_max > float('-inf'):
            return (global_min, global_max)
        return None

    def get_field_label(self):
        """Get display label for current field"""
        labels = {
            'temperature': 'Temperature (K)',
            'direct_solar': 'Direct Solar (W/m²)',
            'illumination': 'Total Illumination (W/m²)',
            'primary_reflected': 'Primary Reflected (W/m²)',
            'primary_ir': 'Primary IR (W/m²)'
        }
        return labels.get(self.current_field, self.current_field)

    def set_timestep(self, index):
        """Set current timestep"""
        if 0 <= index < len(self.timesteps):
            self.load_timestep_data(index)

    def set_field(self, field_name):
        """Set current data field"""
        self.current_field = field_name
        # Reset global range when field changes
        self.global_data_range = None
        if self.mesh and self.results_dir:
            self.load_timestep_data(self.current_timestep)

    def reset_camera(self):
        """Reset camera to default view"""
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.reset_camera()

    def export_image(self, filename):
        """Export current view as image"""
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.screenshot(filename)

    def find_face_by_latlon(self, lat, lon):
        """Find the nearest face to given lat/lon coordinates"""
        if self.mesh_data is None:
            return None

        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']

        # Calculate angular distance (simple approximation)
        dlat = lats - lat
        dlon = lons - lon
        # Handle longitude wraparound
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)

        dist = np.sqrt(dlat**2 + dlon**2)
        face_id = int(np.argmin(dist))

        return face_id

    def highlight_face(self, face_id):
        """Highlight a specific face on the mesh (called from external sources like 2D plots)"""
        if not PYVISTA_AVAILABLE or not self.mesh or face_id is None:
            return

        # Use the selection highlight method
        self.selected_face_id = face_id
        self.update_selection_highlight(face_id)

    def on_surface_picked(self, point):
        """Handle surface point picking - find the cell containing the picked point"""
        if point is None or self.mesh is None:
            return

        try:
            # Find the cell that contains this point
            cell_id = self.mesh.find_closest_cell(point)
            if cell_id is not None and cell_id >= 0:
                self._emit_face_selection(cell_id)
        except Exception as e:
            print(f"Surface picking error: {e}")

    def on_cell_picked(self, cell):
        """Handle cell (face) picking (rectangle selection mode)"""
        if cell is None:
            return

        # Handle different return types from PyVista versions
        if hasattr(cell, 'GetCellId'):
            face_id = cell.GetCellId()
        elif isinstance(cell, (int, np.integer)):
            face_id = int(cell)
        elif hasattr(cell, 'n_cells') and cell.n_cells > 0:
            # It's a mesh subset - find the original cell ID
            # PyVista returns the picked cell as a new mesh
            face_id = self.find_cell_in_mesh(cell)
        else:
            print(f"Unknown cell type: {type(cell)}")
            return

        if face_id is None or face_id < 0:
            return

        self._emit_face_selection(face_id)

    def on_click_position(self, point):
        """Handle click position fallback for face selection"""
        if point is None or self.mesh is None:
            return

        # Find closest cell to click point
        try:
            cell_id = self.mesh.find_closest_cell(point)
            if cell_id >= 0:
                self._emit_face_selection(cell_id)
        except Exception as e:
            print(f"Click position handling: {e}")

    def find_cell_in_mesh(self, picked_cell):
        """Find cell ID of picked cell in original mesh"""
        if picked_cell is None or self.mesh is None:
            return None

        try:
            # Get center of picked cell
            center = picked_cell.cell_centers().points[0]
            # Find closest cell in original mesh
            return self.mesh.find_closest_cell(center)
        except Exception:
            return None

    def _emit_face_selection(self, face_id):
        """Emit face selection signal with data"""
        if face_id < 0 or face_id >= self.mesh.n_cells:
            return

        # Gather data for this face
        data = {
            'face_id': face_id,
            'latitude': float(self.mesh['latitude'][face_id]) if 'latitude' in self.mesh.array_names else None,
            'longitude': float(self.mesh['longitude'][face_id]) if 'longitude' in self.mesh.array_names else None
        }

        # Add current field value
        if self.current_field in self.mesh.array_names:
            data[self.current_field] = float(self.mesh[self.current_field][face_id])

        # Highlight the selected face
        self.highlight_face(face_id)

        self.face_selected.emit(face_id, data)

    def refresh(self):
        """Refresh the display"""
        self.update_display()

    def cleanup(self):
        """Clean up resources"""
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter'):
            self.plotter.close()


class PlotWidget(QWidget):
    """2D plotting widget using Matplotlib with face selection and export"""

    face_selection_requested = pyqtSignal(int)  # Request to highlight face in 3D

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_dir = None
        self.timesteps = []
        self.timestep_labels = []
        self.current_face = None
        self.current_face_lat = None
        self.current_face_lon = None
        self.mesh_data = None
        self.current_plot_data = {}  # Store data for export
        self.current_timestep = 0  # Track current timestep for distribution plots

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        if not MATPLOTLIB_AVAILABLE:
            label = QLabel("Matplotlib not available.\nInstall with: pip install matplotlib")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #ff9800; font-size: 14px;")
            layout.addWidget(label)
            return

        # === Face Selection Controls ===
        selection_group = QGroupBox("Location Selection")
        selection_layout = QGridLayout()

        # Lat/Lon input
        selection_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90.0, 90.0)
        self.lat_input.setDecimals(4)
        self.lat_input.setValue(0.0)
        self.lat_input.setSuffix("°")
        selection_layout.addWidget(self.lat_input, 0, 1)

        selection_layout.addWidget(QLabel("Longitude:"), 0, 2)
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180.0, 360.0)
        self.lon_input.setDecimals(4)
        self.lon_input.setValue(0.0)
        self.lon_input.setSuffix("°")
        selection_layout.addWidget(self.lon_input, 0, 3)

        self.find_btn = QPushButton("Find Location")
        self.find_btn.clicked.connect(self.on_find_location)
        selection_layout.addWidget(self.find_btn, 0, 4)

        # Face ID display
        selection_layout.addWidget(QLabel("Face ID:"), 1, 0)
        self.face_id_label = QLabel("--")
        self.face_id_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        selection_layout.addWidget(self.face_id_label, 1, 1)

        # Actual lat/lon of selected face
        selection_layout.addWidget(QLabel("Actual:"), 1, 2)
        self.actual_latlon_label = QLabel("--")
        self.actual_latlon_label.setStyleSheet("color: #4caf50;")
        selection_layout.addWidget(self.actual_latlon_label, 1, 3, 1, 2)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        # === Plot Type Selection ===
        plot_control_layout = QHBoxLayout()

        plot_control_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Temperature Time Series",
            "All Fields Time Series",
            "Temperature Distribution",
            "Temperature vs Latitude",
            "Global Statistics"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        plot_control_layout.addWidget(self.plot_type_combo)

        plot_control_layout.addStretch()

        # Export buttons
        self.export_data_btn = QPushButton("Export Data (ASCII)")
        self.export_data_btn.clicked.connect(self.export_data_ascii)
        plot_control_layout.addWidget(self.export_data_btn)

        self.export_plot_btn = QPushButton("Export Plot (PNG)")
        self.export_plot_btn.clicked.connect(self.export_plot_image)
        plot_control_layout.addWidget(self.export_plot_btn)

        layout.addLayout(plot_control_layout)

        # === Matplotlib Figure ===
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #252526; color: white;")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # Initialize with default plot
        self.init_empty_plot()

    def init_empty_plot(self):
        """Show empty plot with instructions"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)
        ax.text(0.5, 0.5, "Click on 3D mesh or enter Lat/Lon to select a location",
                ha='center', va='center', color='#888888', fontsize=14,
                transform=ax.transAxes)
        ax.set_title("Temperature Time Series")
        self.canvas.draw()

    def style_axis(self, ax):
        """Apply dark theme styling to axis"""
        ax.set_facecolor('#252526')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#404040')

    def load_results(self, results_dir, mesh_data=None):
        """Load results for plotting"""
        self.results_dir = Path(results_dir)
        self.mesh_data = mesh_data

        # Load timesteps with labels
        time_file = self.results_dir / "time_steps.txt"
        if time_file.exists():
            self.timesteps = []
            self.timestep_labels = []
            with open(time_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.timesteps.append(float(parts[0]))
                        self.timestep_labels.append(parts[1])

        # Reset selection
        self.current_face = None
        self.init_empty_plot()

    def on_find_location(self):
        """Find face at specified lat/lon"""
        if self.mesh_data is None:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        lat = self.lat_input.value()
        lon = self.lon_input.value()

        # Find nearest face
        face_id = self.find_nearest_face(lat, lon)
        if face_id is not None:
            self.select_face(face_id)
            self.face_selection_requested.emit(face_id)

    def find_nearest_face(self, lat, lon):
        """Find the nearest face to given lat/lon"""
        if self.mesh_data is None:
            return None

        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']

        dlat = lats - lat
        dlon = lons - lon
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)

        dist = np.sqrt(dlat**2 + dlon**2)
        return int(np.argmin(dist))

    def select_face(self, face_id, face_data=None):
        """Select a face and update display"""
        self.current_face = face_id

        # Get face lat/lon
        if face_data and 'latitude' in face_data:
            self.current_face_lat = face_data['latitude']
            self.current_face_lon = face_data['longitude']
        elif self.mesh_data is not None and face_id < len(self.mesh_data['latitudes']):
            self.current_face_lat = self.mesh_data['latitudes'][face_id]
            self.current_face_lon = self.mesh_data['longitudes'][face_id]

        # Update UI
        self.face_id_label.setText(str(face_id))
        if self.current_face_lat is not None:
            self.actual_latlon_label.setText(
                f"({self.current_face_lat:.4f}°, {self.current_face_lon:.4f}°)"
            )
            # Update input fields to match
            self.lat_input.blockSignals(True)
            self.lon_input.blockSignals(True)
            self.lat_input.setValue(self.current_face_lat)
            self.lon_input.setValue(self.current_face_lon)
            self.lat_input.blockSignals(False)
            self.lon_input.blockSignals(False)

        # Update plot
        self.update_plot()

    def show_face_timeseries(self, face_id, face_data=None):
        """Show time series for a specific face (called from 3D view)"""
        self.select_face(face_id, face_data)

    def on_plot_type_changed(self, plot_type):
        """Handle plot type change"""
        self.update_plot()

    def set_timestep(self, timestep):
        """Set current timestep and update relevant plots"""
        self.current_timestep = timestep
        # Update plots that depend on timestep (distribution and vs latitude)
        plot_type = self.plot_type_combo.currentText() if hasattr(self, 'plot_type_combo') else ""
        if plot_type in ["Temperature Distribution", "Temperature vs Latitude"]:
            self.update_plot()

    def update_plot(self):
        """Update the current plot"""
        if not MATPLOTLIB_AVAILABLE or not self.results_dir:
            return

        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Temperature Time Series":
            self.plot_temperature_timeseries()
        elif plot_type == "All Fields Time Series":
            self.plot_all_fields_timeseries()
        elif plot_type == "Temperature Distribution":
            self.plot_temperature_distribution()
        elif plot_type == "Temperature vs Latitude":
            self.plot_temperature_vs_latitude()
        elif plot_type == "Global Statistics":
            self.plot_global_statistics()

    def plot_temperature_timeseries(self):
        """Plot temperature time series for selected face"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        if self.current_face is None:
            ax.text(0.5, 0.5, "Select a location on the 3D mesh\nor enter Lat/Lon coordinates",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            ax.set_title("Temperature Time Series")
            self.canvas.draw()
            return

        temp_dir = self.results_dir / "temperature"
        if not temp_dir.exists():
            ax.text(0.5, 0.5, "Temperature data not found",
                    ha='center', va='center', color='#ff5722', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        # Load temperature data for this face
        files = sorted(temp_dir.glob("temperature_*.bin"))
        temps = []
        times = []

        for i, f in enumerate(files):
            data = self.read_binary_data(f)
            if data is not None and self.current_face < len(data):
                temps.append(data[self.current_face])
                times.append(i)

        if temps:
            ax.plot(times, temps, color='#ff5722', linewidth=2, marker='o',
                    markersize=3, label='Temperature')
            ax.fill_between(times, min(temps), temps, alpha=0.3, color='#ff5722')

            # Store for export
            self.current_plot_data = {
                'title': f'Temperature Time Series - Face {self.current_face}',
                'columns': ['Timestep', 'Time_Label', 'Temperature_K'],
                'data': []
            }
            for i, t in enumerate(temps):
                label = self.timestep_labels[i] if i < len(self.timestep_labels) else ""
                self.current_plot_data['data'].append([i, label, t])

            # Add statistics
            ax.axhline(np.mean(temps), color='#4caf50', linestyle='--',
                       linewidth=1, label=f'Mean: {np.mean(temps):.1f} K')
            ax.axhline(np.max(temps), color='#f44336', linestyle=':',
                       linewidth=1, label=f'Max: {np.max(temps):.1f} K')
            ax.axhline(np.min(temps), color='#2196f3', linestyle=':',
                       linewidth=1, label=f'Min: {np.min(temps):.1f} K')

            ax.legend(loc='upper right', facecolor='#252526', edgecolor='#404040',
                      labelcolor='white')

        title = f"Temperature Time Series - Face {self.current_face}"
        if self.current_face_lat is not None:
            title += f" ({self.current_face_lat:.2f}°, {self.current_face_lon:.2f}°)"
        ax.set_title(title)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_all_fields_timeseries(self):
        """Plot all available fields for selected face"""
        self.figure.clear()

        if self.current_face is None:
            ax = self.figure.add_subplot(111)
            self.style_axis(ax)
            ax.text(0.5, 0.5, "Select a location first",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        fields = [
            ('temperature', 'Temperature (K)', '#ff5722'),
            ('illumination', 'Illumination (W/m²)', '#4caf50'),
            ('direct_solar', 'Direct Solar (W/m²)', '#ffeb3b'),
            ('primary_reflected', 'Reflected (W/m²)', '#2196f3'),
        ]

        axes = self.figure.subplots(2, 2)
        self.current_plot_data = {
            'title': f'All Fields Time Series - Face {self.current_face}',
            'columns': ['Timestep'],
            'data': []
        }

        for ax_idx, (field, label, color) in enumerate(fields):
            ax = axes[ax_idx // 2, ax_idx % 2]
            self.style_axis(ax)

            field_dir = self.results_dir / field
            if field_dir.exists():
                files = sorted(field_dir.glob(f"{field}_*.bin"))
                values = []
                for f in files:
                    data = self.read_binary_data(f)
                    if data is not None and self.current_face < len(data):
                        values.append(data[self.current_face])

                if values:
                    ax.plot(range(len(values)), values, color=color, linewidth=1.5)
                    ax.fill_between(range(len(values)), 0, values, alpha=0.3, color=color)

            ax.set_title(label, fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.grid(True, alpha=0.3, color='#404040')

        self.figure.suptitle(f'Face {self.current_face} ({self.current_face_lat:.2f}°, {self.current_face_lon:.2f}°)',
                             color='white', fontsize=12)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_temperature_distribution(self):
        """Plot temperature distribution histogram for current timestep"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            if files:
                # Use current timestep, clamped to valid range
                timestep_idx = min(self.current_timestep, len(files) - 1)
                timestep_idx = max(0, timestep_idx)
                data_file = files[timestep_idx]
                data = self.read_binary_data(data_file)
                if data is not None and len(data) > 0:
                    n, bins, patches = ax.hist(data, bins=50, color='#ff5722',
                                               alpha=0.8, edgecolor='none')

                    # Store for export
                    self.current_plot_data = {
                        'title': f'Temperature Distribution (Timestep {timestep_idx})',
                        'columns': ['Bin_Center_K', 'Count'],
                        'data': []
                    }
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    for bc, count in zip(bin_centers, n):
                        self.current_plot_data['data'].append([bc, count])

                    # Mark selected face if any
                    if self.current_face is not None and self.current_face < len(data):
                        face_temp = data[self.current_face]
                        ax.axvline(face_temp, color='yellow', linewidth=2,
                                   label=f'Selected: {face_temp:.1f} K')
                        ax.legend(facecolor='#252526', edgecolor='#404040', labelcolor='white')

                    # Show timestep label if available
                    ts_label = ""
                    if timestep_idx < len(self.timestep_labels):
                        ts_label = f" - {self.timestep_labels[timestep_idx]}"
                    ax.set_title(f'Temperature Distribution (Timestep {timestep_idx}{ts_label})')

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Number of Faces')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_temperature_vs_latitude(self):
        """Plot temperature vs latitude scatter for current timestep"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        if self.mesh_data is None:
            ax.text(0.5, 0.5, "Mesh data not loaded",
                    ha='center', va='center', color='#888888', fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw()
            return

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            if files:
                # Use current timestep, clamped to valid range
                timestep_idx = min(self.current_timestep, len(files) - 1)
                timestep_idx = max(0, timestep_idx)
                data_file = files[timestep_idx]
                temps = self.read_binary_data(data_file)
                lats = self.mesh_data['latitudes']

                if temps is not None and len(temps) == len(lats):
                    ax.scatter(lats, temps, s=1, alpha=0.3, c='#2196f3')

                    # Store for export (sampled to reduce size)
                    self.current_plot_data = {
                        'title': 'Temperature vs Latitude',
                        'columns': ['Latitude_deg', 'Temperature_K'],
                        'data': []
                    }
                    # Sample every Nth point to keep export reasonable
                    step = max(1, len(lats) // 10000)
                    for i in range(0, len(lats), step):
                        self.current_plot_data['data'].append([lats[i], temps[i]])

                    # Mark selected face
                    if self.current_face is not None and self.current_face < len(temps):
                        ax.scatter([lats[self.current_face]], [temps[self.current_face]],
                                   s=100, c='yellow', marker='*', zorder=10,
                                   label=f'Selected Face {self.current_face}')
                        ax.legend(facecolor='#252526', edgecolor='#404040', labelcolor='white')

                    # Title with timestep info
                    ts_label = ""
                    if timestep_idx < len(self.timestep_labels):
                        ts_label = f" - {self.timestep_labels[timestep_idx]}"
                    ax.set_title(f'Temperature vs Latitude (Timestep {timestep_idx}{ts_label})')

        if not ax.get_title():
            ax.set_title('Temperature vs Latitude')
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_global_statistics(self):
        """Plot global min/max/mean temperature over time"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.style_axis(ax)

        temp_dir = self.results_dir / "temperature"
        if temp_dir.exists():
            files = sorted(temp_dir.glob("temperature_*.bin"))
            mins, maxs, means = [], [], []

            for f in files:
                data = self.read_binary_data(f)
                if data is not None and len(data) > 0:
                    mins.append(np.min(data))
                    maxs.append(np.max(data))
                    means.append(np.mean(data))

            if mins:
                timesteps = range(len(mins))
                ax.fill_between(timesteps, mins, maxs, alpha=0.3, color='#ff5722',
                                label='Min-Max Range')
                ax.plot(timesteps, means, color='#ff5722', linewidth=2, label='Mean')
                ax.plot(timesteps, mins, color='#2196f3', linewidth=1, linestyle='--', label='Min')
                ax.plot(timesteps, maxs, color='#f44336', linewidth=1, linestyle='--', label='Max')

                # Store for export
                self.current_plot_data = {
                    'title': 'Global Temperature Statistics',
                    'columns': ['Timestep', 'Min_K', 'Max_K', 'Mean_K'],
                    'data': []
                }
                for i in range(len(mins)):
                    self.current_plot_data['data'].append([i, mins[i], maxs[i], means[i]])

                ax.legend(loc='upper right', facecolor='#252526', edgecolor='#404040',
                          labelcolor='white')

        ax.set_title('Global Temperature Statistics')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def read_binary_data(self, file_path):
        """Read binary data file"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if len(header) == 4:
                    count = struct.unpack('<I', header)[0]
                    f.seek(0, 2)
                    file_size = f.tell()

                    if file_size == 4 + count * 4:
                        f.seek(4)
                        return np.frombuffer(f.read(), dtype='<f4')
                    else:
                        f.seek(0)
                        return np.frombuffer(f.read(), dtype='<f4')

            return None
        except Exception:
            return None

    def export_data_ascii(self):
        """Export current plot data as ASCII file"""
        if not self.current_plot_data or not self.current_plot_data.get('data'):
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            str(Path.home() / "ptm3d_plot_data.dat"),
            "Data Files (*.dat *.txt *.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w') as f:
                # Write header
                f.write(f"# {self.current_plot_data['title']}\n")
                f.write(f"# Generated by PTM3D GUI\n")
                if self.current_face is not None:
                    f.write(f"# Face ID: {self.current_face}\n")
                    if self.current_face_lat is not None:
                        f.write(f"# Location: ({self.current_face_lat:.6f}, {self.current_face_lon:.6f})\n")
                f.write("#\n")

                # Write column headers
                f.write("# " + "\t".join(self.current_plot_data['columns']) + "\n")

                # Write data
                for row in self.current_plot_data['data']:
                    line = "\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row)
                    f.write(line + "\n")

            QMessageBox.information(self, "Export Complete",
                                    f"Data exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def export_plot_image(self):
        """Export current plot as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            str(Path.home() / "ptm3d_plot.png"),
            "PNG Images (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=150, facecolor='#1e1e1e',
                                edgecolor='none', bbox_inches='tight')
            QMessageBox.information(self, "Export Complete",
                                    f"Plot exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plot:\n{str(e)}")

    def update_plots(self):
        """Update plots (called when results are loaded)"""
        self.update_plot()


class VisualizationPanel(QWidget):
    """Combined visualization panel with 3D view and 2D plots"""

    face_selected = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_dir = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget for 3D/2D views
        self.tab_widget = QTabWidget()

        # 3D View Tab
        self.view_3d = PyVistaWidget()
        self.view_3d.face_selected.connect(self.on_face_selected)
        self.tab_widget.addTab(self.view_3d, "3D Visualization")

        # 2D Plots Tab
        self.plots_2d = PlotWidget()
        self.plots_2d.face_selection_requested.connect(self.on_face_selection_from_plots)
        self.tab_widget.addTab(self.plots_2d, "2D Plots")

        layout.addWidget(self.tab_widget)

        # Controls panel at bottom
        self.create_controls()
        layout.addWidget(self.controls_widget)

    def create_controls(self):
        """Create visualization control panel"""
        self.controls_widget = QFrame()
        self.controls_widget.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self.controls_widget)
        layout.setContentsMargins(10, 5, 10, 5)

        # Timestep slider
        layout.addWidget(QLabel("Timestep:"))
        self.timestep_slider = QSlider(Qt.Orientation.Horizontal)
        self.timestep_slider.setMinimum(0)
        self.timestep_slider.setMaximum(0)
        self.timestep_slider.valueChanged.connect(self.on_timestep_slider_changed)
        layout.addWidget(self.timestep_slider, 1)

        self.timestep_label = QLabel("0/0")
        layout.addWidget(self.timestep_label)

        layout.addWidget(QLabel(" "))  # Spacer

        # Playback controls
        self.play_btn = QPushButton("Play")
        self.play_btn.setMaximumWidth(60)
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.advance_timestep)
        self.is_playing = False

        layout.addWidget(QLabel(" | "))

        # View options
        self.show_edges_check = QCheckBox("Show Edges")
        self.show_edges_check.stateChanged.connect(self.on_show_edges_changed)
        layout.addWidget(self.show_edges_check)

        reset_btn = QPushButton("Reset View")
        reset_btn.setMaximumWidth(80)
        reset_btn.clicked.connect(self.reset_camera)
        layout.addWidget(reset_btn)

    def load_results(self, results_dir):
        """Load simulation results"""
        self.results_dir = results_dir

        # Load into 3D view
        self.view_3d.load_results(results_dir)

        # Load into 2D plots
        self.plots_2d.load_results(results_dir, self.view_3d.mesh_data)

        # Update slider range
        n_timesteps = len(self.view_3d.timesteps)
        self.timestep_slider.setMaximum(max(0, n_timesteps - 1))
        self.timestep_label.setText(f"0/{n_timesteps}")

    def set_timestep(self, index):
        """Set current timestep"""
        self.timestep_slider.setValue(index)

    def set_field(self, field_name):
        """Set data field to visualize"""
        self.view_3d.set_field(field_name)

    def on_timestep_slider_changed(self, value):
        """Handle timestep slider change"""
        self.view_3d.set_timestep(value)
        self.plots_2d.set_timestep(value)  # Update 2D plots as well
        n_total = len(self.view_3d.timesteps)
        self.timestep_label.setText(f"{value}/{n_total}")

    def toggle_playback(self):
        """Toggle animation playback"""
        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            self.animation_timer.start(200)  # 5 fps
            self.play_btn.setText("Pause")
            self.is_playing = True

    def advance_timestep(self):
        """Advance to next timestep in animation"""
        current = self.timestep_slider.value()
        max_val = self.timestep_slider.maximum()
        if current < max_val:
            self.timestep_slider.setValue(current + 1)
        else:
            self.timestep_slider.setValue(0)  # Loop

    def on_show_edges_changed(self, state):
        """Handle show edges checkbox"""
        pass

    def on_face_selected(self, face_id, data):
        """Handle face selection from 3D view"""
        self.face_selected.emit(face_id, data)
        self.plots_2d.show_face_timeseries(face_id, data)
        self.tab_widget.setCurrentWidget(self.plots_2d)

    def on_face_selection_from_plots(self, face_id):
        """Handle face selection request from 2D plots"""
        # Highlight in 3D view
        self.view_3d.highlight_face(face_id)
        # Switch to 3D tab briefly to show selection
        # self.tab_widget.setCurrentWidget(self.view_3d)

    def reset_camera(self):
        """Reset camera to default view"""
        self.view_3d.reset_camera()

    def export_image(self, filename):
        """Export current visualization"""
        self.view_3d.export_image(filename)

    def refresh(self):
        """Refresh visualization"""
        self.view_3d.refresh()
        self.plots_2d.update_plots()

    def cleanup(self):
        """Clean up resources"""
        self.animation_timer.stop()
        self.view_3d.cleanup()
