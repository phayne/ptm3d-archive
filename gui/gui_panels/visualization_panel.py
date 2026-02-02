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
    QMessageBox, QDialog, QDialogButtonBox, QProgressDialog,
    QRadioButton, QButtonGroup, QApplication
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
        self.detected_hemisphere = 'global'  # 'north', 'south', or 'global'

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

        # Reset camera tracking so first load gets default view centered on mesh
        self._has_loaded_data = False

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

            # Detect hemisphere for camera orientation
            self.detected_hemisphere = self.detect_hemisphere(latitudes)

            self.update_display()

        except Exception as e:
            print(f"Error loading mesh: {e}")

    def detect_hemisphere(self, latitudes):
        """Detect which hemisphere the mesh is predominantly in.

        Args:
            latitudes: Array of face latitudes in degrees

        Returns:
            str: 'north', 'south', or 'global' (spans both hemispheres significantly)
        """
        if latitudes is None or len(latitudes) == 0:
            return 'global'

        mean_lat = np.mean(latitudes)
        min_lat = np.min(latitudes)
        max_lat = np.max(latitudes)
        lat_range = max_lat - min_lat

        # If mesh spans more than 60 degrees of latitude, consider it global
        if lat_range > 60:
            # But still bias toward the hemisphere with more coverage
            if mean_lat < -15:
                return 'south'
            elif mean_lat > 15:
                return 'north'
            return 'global'

        # For regional meshes, use mean latitude
        if mean_lat < -5:
            return 'south'
        elif mean_lat > 5:
            return 'north'
        else:
            return 'global'

    def set_hemisphere_camera(self):
        """Set camera orientation based on detected hemisphere.

        For southern hemisphere meshes, orients the view so that south pole
        is "up" (looking from below the mesh with inverted z-axis).
        """
        if not PYVISTA_AVAILABLE or not self.plotter or not self.mesh:
            return

        hemisphere = getattr(self, 'detected_hemisphere', 'global')

        # Get mesh center and bounds for camera positioning
        bounds = self.mesh.bounds
        center = self.mesh.center
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_dim = max(x_range, y_range, z_range)

        if hemisphere == 'south':
            # View from below: camera at negative z, looking up, with -z as "up"
            camera_pos = (center[0], center[1], center[2] - max_dim * 2)
            focal_point = center
            view_up = (0, 1, 0)  # Y-up when looking from below
            self.plotter.camera_position = [camera_pos, focal_point, view_up]
        elif hemisphere == 'north':
            # Standard view from above: camera at positive z, looking down
            camera_pos = (center[0], center[1], center[2] + max_dim * 2)
            focal_point = center
            view_up = (0, 1, 0)  # Y-up
            self.plotter.camera_position = [camera_pos, focal_point, view_up]
        else:
            # Global mesh - use default isometric view
            self.plotter.reset_camera()
            self.plotter.view_isometric()

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

        # Save camera position before clearing to preserve user's view
        camera_position = self.plotter.camera_position

        self.plotter.clear()

        # Flag to check if we should restore camera (not on first load)
        # Use getattr to check the actual value, not just existence
        restore_camera = camera_position is not None and getattr(self, '_has_loaded_data', False)

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

        # Restore camera position if this isn't the first load
        if restore_camera:
            self.plotter.camera_position = camera_position
        else:
            # First load - set camera based on detected hemisphere
            self.set_hemisphere_camera()
            # Mark that we've loaded data, so future updates preserve camera
            self._has_loaded_data = True

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

    def export_animation(self, filename, format_type="gif", fps=5, progress_callback=None):
        """Export animation of all timesteps.

        Args:
            filename: Output file path
            format_type: "gif" or "mp4"
            fps: Frames per second
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            True if successful, False otherwise
        """
        if not PYVISTA_AVAILABLE or not self.mesh or not self.results_dir:
            return False

        n_timesteps = len(self.timesteps)
        if n_timesteps == 0:
            return False

        try:
            # Store current timestep to restore later
            original_timestep = self.current_timestep

            # Store camera position to maintain consistent view
            camera_position = self.plotter.camera_position

            if format_type.lower() == "gif":
                # Use PyVista's GIF export
                self.plotter.open_gif(filename)
            else:
                # Use PyVista's movie export (requires ffmpeg)
                self.plotter.open_movie(filename, framerate=fps)

            # Iterate through all timesteps
            for i in range(n_timesteps):
                # Load and display timestep
                self.load_timestep_data(i)

                # Restore camera position to ensure consistent framing
                self.plotter.camera_position = camera_position

                # Ensure complete render before capturing frame
                self.plotter.render()

                # Write frame
                self.plotter.write_frame()

                # Report progress
                if progress_callback:
                    progress_callback(i + 1, n_timesteps)

            # Mute the plotter before closing to prevent any additional frame capture
            # by ensuring render state is stable
            self.plotter.render()

            # Close the animation file
            self.plotter.mwriter.close()  # Close writer directly to avoid extra frame

            # Reinitialize plotter after close
            self._reinit_plotter()

            # Restore original timestep
            self.load_timestep_data(original_timestep)

            return True

        except Exception as e:
            print(f"Animation export error: {e}")
            # Try to recover
            try:
                # Try to close writer if it exists
                if hasattr(self.plotter, 'mwriter') and self.plotter.mwriter:
                    try:
                        self.plotter.mwriter.close()
                    except Exception:
                        pass
                self._reinit_plotter()
                self.load_timestep_data(original_timestep if 'original_timestep' in dir() else 0)
            except Exception:
                pass
            return False

    def _reinit_plotter(self):
        """Reinitialize the plotter after closing for animation export."""
        if not PYVISTA_AVAILABLE:
            return

        # Get the layout
        layout = self.layout()

        # Remove old plotter widget
        if hasattr(self, 'plotter') and self.plotter:
            try:
                layout.removeWidget(self.plotter.interactor)
            except Exception:
                pass

        # Create new plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1e1e1e')
        self.plotter.add_axes()

        # Add back to layout (index 1, after controls)
        layout.insertWidget(1, self.plotter.interactor, 1)

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

        # Time axis settings
        self.use_time_axis = False  # False = timestep index, True = actual time
        self.time_unit = "s"  # Current time unit: s, min, hr, d
        self.time_unit_label = "Time (s)"  # Label for x-axis
        self.time_scale = 1.0  # Multiplier to convert timestep values to display unit

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

        # Time axis toggle
        plot_control_layout.addWidget(QLabel("  X-Axis:"))
        self.time_axis_btn = QPushButton("Timestep")
        self.time_axis_btn.setCheckable(True)
        self.time_axis_btn.setMaximumWidth(100)
        self.time_axis_btn.setToolTip("Toggle between timestep index and actual time units")
        self.time_axis_btn.clicked.connect(self.toggle_time_axis)
        plot_control_layout.addWidget(self.time_axis_btn)

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

        # Calculate appropriate time unit for this dataset
        self.calculate_time_unit()

        # Reset time axis mode and update button
        self.use_time_axis = False
        if hasattr(self, 'time_axis_btn'):
            self.time_axis_btn.setText("Timestep")
            self.time_axis_btn.setChecked(False)

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

    def toggle_time_axis(self):
        """Toggle between timestep index and actual time units"""
        self.use_time_axis = not self.use_time_axis

        if self.use_time_axis:
            # Calculate and set appropriate time unit
            self.calculate_time_unit()
            self.time_axis_btn.setText(self.time_unit_label.replace("Time ", "").strip("()"))
            self.time_axis_btn.setChecked(True)
        else:
            self.time_axis_btn.setText("Timestep")
            self.time_axis_btn.setChecked(False)

        # Refresh the current plot
        self.update_plot()

    def calculate_time_unit(self):
        """Calculate the best time unit based on timestep interval.

        Automatically selects seconds, minutes, hours, or days based on
        the typical interval between timesteps.
        """
        if len(self.timesteps) < 2:
            # Default to seconds if we can't calculate
            self.time_unit = "s"
            self.time_unit_label = "Time (s)"
            self.time_scale = 1.0
            return

        # Calculate average timestep interval (timesteps are in seconds)
        intervals = []
        for i in range(1, len(self.timesteps)):
            intervals.append(self.timesteps[i] - self.timesteps[i - 1])
        avg_interval = sum(intervals) / len(intervals)

        # Total duration
        total_duration = self.timesteps[-1] - self.timesteps[0]

        # Choose unit based on total duration and interval
        # Goal: have x-axis values in a reasonable range (not too many digits)
        if total_duration < 120:  # Less than 2 minutes
            self.time_unit = "s"
            self.time_unit_label = "Time (s)"
            self.time_scale = 1.0
        elif total_duration < 7200:  # Less than 2 hours
            self.time_unit = "min"
            self.time_unit_label = "Time (min)"
            self.time_scale = 1.0 / 60.0
        elif total_duration < 172800:  # Less than 2 days
            self.time_unit = "hr"
            self.time_unit_label = "Time (hr)"
            self.time_scale = 1.0 / 3600.0
        else:  # 2 days or more
            self.time_unit = "d"
            self.time_unit_label = "Time (days)"
            self.time_scale = 1.0 / 86400.0

    def get_time_axis_values(self, n_points=None):
        """Get x-axis values based on current time axis mode.

        Args:
            n_points: Number of data points (defaults to len(self.timesteps))

        Returns:
            tuple: (x_values, x_label) where x_values is a list/array and x_label is the axis label
        """
        if n_points is None:
            n_points = len(self.timesteps)

        if self.use_time_axis and len(self.timesteps) >= n_points:
            # Use actual time values, scaled to current unit
            # Subtract first timestep to start from 0
            t0 = self.timesteps[0] if self.timesteps else 0
            x_values = [(self.timesteps[i] - t0) * self.time_scale for i in range(n_points)]
            x_label = self.time_unit_label
        else:
            # Use timestep index
            x_values = list(range(n_points))
            x_label = "Timestep"

        return x_values, x_label

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

        for i, f in enumerate(files):
            data = self.read_binary_data(f)
            if data is not None and self.current_face < len(data):
                temps.append(data[self.current_face])

        # Get x-axis label (even if no data, for consistent labeling)
        _, x_label = self.get_time_axis_values(len(files) if files else 1)

        if temps:
            # Get x-axis values based on time axis mode
            x_values, x_label = self.get_time_axis_values(len(temps))

            ax.plot(x_values, temps, color='#ff5722', linewidth=2, marker='o',
                    markersize=3, label='Temperature')
            ax.fill_between(x_values, min(temps), temps, alpha=0.3, color='#ff5722')

            # Store for export (always include both timestep and time info)
            self.current_plot_data = {
                'title': f'Temperature Time Series - Face {self.current_face}',
                'columns': ['Timestep', 'Time_Label', f'Time_{self.time_unit}', 'Temperature_K'],
                'data': []
            }
            t0 = self.timesteps[0] if self.timesteps else 0
            for i, temp in enumerate(temps):
                label = self.timestep_labels[i] if i < len(self.timestep_labels) else ""
                time_val = (self.timesteps[i] - t0) * self.time_scale if i < len(self.timesteps) else i
                self.current_plot_data['data'].append([i, label, time_val, temp])

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
        ax.set_xlabel(x_label)
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

        # Get x-axis label once for all subplots
        _, x_label = self.get_time_axis_values()

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
                    # Get x-axis values based on time axis mode
                    x_values, _ = self.get_time_axis_values(len(values))
                    ax.plot(x_values, values, color=color, linewidth=1.5)
                    ax.fill_between(x_values, 0, values, alpha=0.3, color=color)

            ax.set_title(label, fontsize=10)
            ax.set_xlabel(x_label, fontsize=9)
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

        x_label = "Timestep"  # Default

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
                # Get x-axis values based on time axis mode
                x_values, x_label = self.get_time_axis_values(len(mins))

                ax.fill_between(x_values, mins, maxs, alpha=0.3, color='#ff5722',
                                label='Min-Max Range')
                ax.plot(x_values, means, color='#ff5722', linewidth=2, label='Mean')
                ax.plot(x_values, mins, color='#2196f3', linewidth=1, linestyle='--', label='Min')
                ax.plot(x_values, maxs, color='#f44336', linewidth=1, linestyle='--', label='Max')

                # Store for export (include both timestep and time)
                self.current_plot_data = {
                    'title': 'Global Temperature Statistics',
                    'columns': ['Timestep', f'Time_{self.time_unit}', 'Min_K', 'Max_K', 'Mean_K'],
                    'data': []
                }
                t0 = self.timesteps[0] if self.timesteps else 0
                for i in range(len(mins)):
                    time_val = (self.timesteps[i] - t0) * self.time_scale if i < len(self.timesteps) else i
                    self.current_plot_data['data'].append([i, time_val, mins[i], maxs[i], means[i]])

                ax.legend(loc='upper right', facecolor='#252526', edgecolor='#404040',
                          labelcolor='white')

        ax.set_title('Global Temperature Statistics')
        ax.set_xlabel(x_label)
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


class ExportDataDialog(QDialog):
    """Dialog for configuring and executing data export in GeoTIFF/HDF5 formats."""

    def __init__(self, results_dir, mesh_data, timesteps, parent=None):
        super().__init__(parent)
        self.results_dir = Path(results_dir)
        self.mesh_data = mesh_data
        self.timesteps = timesteps
        self.export_worker = None

        # Import availability and detection functions
        from gui_core.data_exporter import (
            SCIPY_AVAILABLE, RASTERIO_AVAILABLE, H5PY_AVAILABLE,
            GridResolutionEstimator, FIELD_INFO, BODY_RADII,
            detect_body_from_mesh, choose_projection, POLAR_LAT_THRESHOLD
        )
        self.scipy_available = SCIPY_AVAILABLE
        self.rasterio_available = RASTERIO_AVAILABLE
        self.h5py_available = H5PY_AVAILABLE
        self.body_radii = BODY_RADII

        # Auto-detect body from mesh vertices
        self.detected_body = detect_body_from_mesh(mesh_data) if mesh_data else None
        if self.detected_body is None:
            self.detected_body = 'Moon'  # fallback
        self.body_radius = BODY_RADII.get(self.detected_body, 1737400.0)

        # Auto-detect projection
        self.projection = 'eqc'  # fallback
        if mesh_data and 'latitudes' in mesh_data:
            self.projection = choose_projection(mesh_data['latitudes'])
            lats = mesh_data['latitudes']
            if self.projection == 'stere':
                min_lat = float(np.nanmin(lats))
                max_lat = float(np.nanmax(lats))
                self.pole_lat = 90.0 if min_lat > POLAR_LAT_THRESHOLD else -90.0

        # Auto-detect resolution
        self.auto_resolution = 0.05  # fallback
        if SCIPY_AVAILABLE and mesh_data:
            self.auto_resolution = GridResolutionEstimator.estimate_resolution(
                mesh_data['latitudes'], mesh_data['longitudes']
            )

        # Detect available fields
        self.available_fields = []
        for field_name in FIELD_INFO:
            field_dir = self.results_dir / field_name
            if field_dir.exists() and any(field_dir.glob("*.bin")):
                self.available_fields.append(field_name)

        self.setWindowTitle("Export Simulation Data")
        self.setMinimumWidth(480)
        self.init_ui()

    def init_ui(self):
        from gui_core.data_exporter import FIELD_INFO

        layout = QVBoxLayout(self)

        # Format selection
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()

        self.format_btn_group = QButtonGroup(self)
        format_row = QHBoxLayout()

        self.geotiff_radio = QRadioButton("GeoTIFF")
        self.hdf5_radio = QRadioButton("HDF5")
        self.both_radio = QRadioButton("Both")
        self.both_radio.setChecked(True)

        self.format_btn_group.addButton(self.geotiff_radio, 0)
        self.format_btn_group.addButton(self.hdf5_radio, 1)
        self.format_btn_group.addButton(self.both_radio, 2)

        format_row.addWidget(self.geotiff_radio)
        format_row.addWidget(self.hdf5_radio)
        format_row.addWidget(self.both_radio)
        format_row.addStretch()
        format_layout.addLayout(format_row)

        # Warnings for missing dependencies
        warnings = []
        if not self.rasterio_available:
            warnings.append("rasterio not available (pip install rasterio)")
            self.geotiff_radio.setEnabled(False)
            self.both_radio.setEnabled(False)
            self.hdf5_radio.setChecked(True)
        if not self.h5py_available:
            warnings.append("h5py not available (pip install h5py)")
            self.hdf5_radio.setEnabled(False)
            if not self.rasterio_available:
                self.both_radio.setEnabled(False)
            elif self.both_radio.isChecked():
                self.geotiff_radio.setChecked(True)
        if not self.scipy_available:
            warnings.append("scipy not available - regridding disabled (pip install scipy)")

        if warnings:
            warn_label = QLabel("\n".join(warnings))
            warn_label.setStyleSheet("color: #ff9800; font-size: 11px; font-style: italic;")
            format_layout.addWidget(warn_label)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Fields selection
        fields_group = QGroupBox("Fields")
        fields_layout = QVBoxLayout()

        self.field_checks = {}
        default_checked = {'temperature', 'illumination', 'direct_solar'}
        for field_name in self.available_fields:
            info = FIELD_INFO.get(field_name, {})
            label = info.get('label', field_name)
            units = info.get('units', '')
            cb = QCheckBox(f"{label} ({units})" if units else label)
            cb.setChecked(field_name in default_checked)
            cb.stateChanged.connect(self._update_estimate)
            fields_layout.addWidget(cb)
            self.field_checks[field_name] = cb

        if not self.available_fields:
            no_fields_label = QLabel("No data fields found in results directory.")
            no_fields_label.setStyleSheet("color: #ff5252; font-style: italic;")
            fields_layout.addWidget(no_fields_label)

        fields_group.setLayout(fields_layout)
        layout.addWidget(fields_group)

        # Grid & Projection
        grid_group = QGroupBox("Grid & Projection")
        grid_layout = QGridLayout()

        # Auto-detected body
        radius_km = self.body_radius / 1000.0
        body_label = QLabel(f"Body: {self.detected_body} (R = {radius_km:.1f} km)")
        body_label.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(body_label, 0, 0, 1, 2)

        # Projection type
        if self.projection == 'stere':
            pole_str = "North" if getattr(self, 'pole_lat', 90) > 0 else "South"
            proj_text = f"Polar stereographic ({pole_str} pole, +proj=stere)"
        else:
            proj_text = "Equirectangular (+proj=eqc)"
        proj_label = QLabel(f"CRS: {proj_text}")
        proj_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(proj_label, 0, 2)

        grid_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.001, 10.0)
        self.resolution_spin.setDecimals(4)
        self.resolution_spin.setSingleStep(0.01)
        self.resolution_spin.setValue(self.auto_resolution)
        self.resolution_spin.setSuffix(" degrees")
        self.resolution_spin.valueChanged.connect(self._update_grid_info)
        grid_layout.addWidget(self.resolution_spin, 1, 1)

        auto_label = QLabel(f"(auto-detected: {self.auto_resolution}\u00b0)")
        auto_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(auto_label, 1, 2)

        self.grid_size_label = QLabel("")
        self.grid_size_label.setStyleSheet("color: #888888; font-size: 11px;")
        grid_layout.addWidget(self.grid_size_label, 2, 0, 1, 3)

        self.per_timestep_check = QCheckBox("One GeoTIFF per timestep (smaller files)")
        self.per_timestep_check.setChecked(True)
        self.per_timestep_check.setToolTip(
            "Write individual GeoTIFF files per timestep instead of one "
            "large multi-band file. Recommended for large meshes."
        )
        grid_layout.addWidget(self.per_timestep_check, 3, 0, 1, 3)

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()

        output_layout.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        default_out = str(self.results_dir / "exports")
        self.output_dir_edit.setText(default_out)
        self.output_dir_edit.textChanged.connect(self._update_estimate)
        output_layout.addWidget(self.output_dir_edit, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(browse_btn, 0, 2)

        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet("color: #888888; font-size: 11px;")
        output_layout.addWidget(self.estimate_label, 1, 0, 1, 3)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Buttons
        self.button_box = QDialogButtonBox()
        self.export_btn = self.button_box.addButton(
            "Export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.cancel_btn = self.button_box.addButton(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.export_btn.clicked.connect(self._start_export)
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initialize displays
        self._update_grid_info()
        self._update_estimate()

    def _get_body_radius(self):
        """Get detected body radius in meters."""
        return self.body_radius

    def _update_grid_info(self):
        """Update grid size display when resolution changes."""
        if not self.mesh_data:
            return
        res = self.resolution_spin.value()
        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']
        body_radius = self._get_body_radius()

        if self.projection == 'stere':
            # For stereographic, grid is in projected meters
            from gui_core.data_exporter import _forward_polar_stereo
            pole_lat = getattr(self, 'pole_lat', 90.0)
            center_lon = float((np.nanmin(lons) + np.nanmax(lons)) / 2)
            x, y = _forward_polar_stereo(lats, lons, pole_lat, center_lon, body_radius)
            # Pixel size in meters from degree-resolution mapped to projected space
            pixel_m = body_radius * np.radians(res)
            x_range = float(np.nanmax(x) - np.nanmin(x))
            y_range = float(np.nanmax(y) - np.nanmin(y))
            n_x = int(np.ceil(x_range / pixel_m)) + 1
            n_y = int(np.ceil(y_range / pixel_m)) + 1
            self.grid_size_label.setText(
                f"Grid size: {n_y} \u00d7 {n_x} pixels "
                f"(~{pixel_m:.0f} m/px, stereographic)"
            )
        else:
            lat_range = np.ptp(lats)
            lon_range = np.ptp(lons)
            n_lat = int(np.ceil(lat_range / res)) + 1
            n_lon = int(np.ceil(lon_range / res)) + 1
            # Pixel size in meters at center latitude
            center_lat = (np.nanmin(lats) + np.nanmax(lats)) / 2
            pixel_m = body_radius * np.radians(res)
            pixel_m_lon = pixel_m * np.cos(np.radians(center_lat))
            self.grid_size_label.setText(
                f"Grid size: {n_lat} \u00d7 {n_lon} pixels "
                f"(~{pixel_m:.0f} \u00d7 {pixel_m_lon:.0f} m/px at center)"
            )
        self._update_estimate()

    def _update_estimate(self):
        """Update estimated output size."""
        selected_fields = [f for f, cb in self.field_checks.items() if cb.isChecked()]
        n_fields = len(selected_fields)
        n_timesteps = len(self.timesteps)

        if n_fields == 0 or n_timesteps == 0:
            self.estimate_label.setText("Estimated size: --")
            self.export_btn.setEnabled(False)
            return

        self.export_btn.setEnabled(True)

        res = self.resolution_spin.value()
        lats = self.mesh_data['latitudes']
        lons = self.mesh_data['longitudes']
        n_faces = len(lats)

        if self.projection == 'stere':
            from gui_core.data_exporter import _forward_polar_stereo
            pole_lat = getattr(self, 'pole_lat', 90.0)
            center_lon = float((np.nanmin(lons) + np.nanmax(lons)) / 2)
            body_radius = self._get_body_radius()
            x, y = _forward_polar_stereo(lats, lons, pole_lat, center_lon, body_radius)
            pixel_m = body_radius * np.radians(res)
            n_lat = int(np.ceil((np.nanmax(y) - np.nanmin(y)) / pixel_m)) + 1
            n_lon = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / pixel_m)) + 1
        else:
            n_lat = int(np.ceil(np.ptp(lats) / res)) + 1
            n_lon = int(np.ceil(np.ptp(lons) / res)) + 1

        total_bytes = 0
        fmt_id = self.format_btn_group.checkedId()

        if fmt_id == 0 or fmt_id == 2:  # GeoTIFF or Both
            # Multi-band GeoTIFF: ~4 bytes per pixel per band, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_lat * n_lon * 4 * 0.5

        if fmt_id == 1 or fmt_id == 2:  # HDF5 or Both
            # Unstructured: n_fields * n_timesteps * n_faces * 4 bytes, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_faces * 4 * 0.5
            # Regridded: n_fields * n_timesteps * n_lat * n_lon * 4 bytes, compressed ~50%
            total_bytes += n_fields * n_timesteps * n_lat * n_lon * 4 * 0.5

        # Format size
        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes / 1024:.1f} KB"
        elif total_bytes < 1024**3:
            size_str = f"{total_bytes / 1024**2:.1f} MB"
        else:
            size_str = f"{total_bytes / 1024**3:.2f} GB"

        self.estimate_label.setText(f"Estimated size: ~{size_str}")

    def _browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.output_dir_edit.text(),
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _start_export(self):
        """Start the export process."""
        from gui_core.data_exporter import DataExportWorker

        selected_fields = [f for f, cb in self.field_checks.items() if cb.isChecked()]
        if not selected_fields:
            QMessageBox.warning(self, "No Fields", "Please select at least one field to export.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please specify an output directory.")
            return

        # Determine formats
        fmt_id = self.format_btn_group.checkedId()
        formats = []
        if fmt_id == 0:
            formats = ['geotiff']
        elif fmt_id == 1:
            formats = ['hdf5']
        else:
            formats = ['geotiff', 'hdf5']

        config = {
            'results_dir': str(self.results_dir),
            'mesh_data': self.mesh_data,
            'timesteps': self.timesteps,
            'fields': selected_fields,
            'formats': formats,
            'resolution': self.resolution_spin.value(),
            'output_dir': output_dir,
            'geotiff_per_timestep': self.per_timestep_check.isChecked(),
            'body_radius': self._get_body_radius(),
        }

        # Disable controls during export
        self.export_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel Export")

        # Create progress dialog
        n_timesteps = len(self.timesteps)
        n_fields = len(selected_fields)
        total_steps = 1 + (n_fields * n_timesteps) + n_fields

        self.progress = QProgressDialog("Starting export...", "Cancel", 0, total_steps, self)
        self.progress.setWindowTitle("Exporting Data")
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.canceled.connect(self._cancel_export)
        self.progress.show()

        # Start worker thread
        self.export_worker = DataExportWorker(config)
        self.export_worker.progress_updated.connect(self._on_progress)
        self.export_worker.status_updated.connect(self._on_status)
        self.export_worker.export_finished.connect(self._on_finished)
        self.export_worker.start()

    def _cancel_export(self):
        """Cancel the running export."""
        if self.export_worker:
            self.export_worker.cancel()

    def _on_progress(self, current, total):
        """Handle progress update."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.setMaximum(total)
            self.progress.setValue(current)

    def _on_status(self, message):
        """Handle status update."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.setLabelText(message)

    def _on_finished(self, success, message):
        """Handle export completion."""
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()

        if success:
            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        else:
            QMessageBox.warning(self, "Export", message)
            self.export_btn.setEnabled(True)
            self.cancel_btn.setText("Cancel")


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

        layout.addWidget(QLabel(" | "))

        # Export Image button
        export_img_btn = QPushButton("Export Image")
        export_img_btn.setMaximumWidth(100)
        export_img_btn.clicked.connect(self.export_current_image)
        layout.addWidget(export_img_btn)

        # Export Animation button
        export_anim_btn = QPushButton("Export Animation")
        export_anim_btn.setMaximumWidth(120)
        export_anim_btn.clicked.connect(self.show_export_animation_dialog)
        layout.addWidget(export_anim_btn)

        # Export Data button
        export_data_btn = QPushButton("Export Data")
        export_data_btn.setMaximumWidth(100)
        export_data_btn.clicked.connect(self.show_export_data_dialog)
        layout.addWidget(export_data_btn)

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

    def export_current_image(self):
        """Export current 3D view as image"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        # Get current field and timestep for default filename
        field = self.view_3d.current_field
        timestep = self.view_3d.current_timestep
        default_name = f"ptm3d_{field}_t{timestep:04d}.png"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image",
            str(Path.cwd() / default_name),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;TIFF Images (*.tiff *.tif);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.view_3d.export_image(filename)
            QMessageBox.information(self, "Export Complete",
                                   f"Image exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                               f"Failed to export image:\n{str(e)}")

    def show_export_animation_dialog(self):
        """Show dialog to configure and export animation"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        n_timesteps = len(self.view_3d.timesteps)
        if n_timesteps == 0:
            QMessageBox.warning(self, "No Timesteps", "No timesteps available for animation.")
            return

        # Create export dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Animation")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Format selection
        format_group = QGroupBox("Animation Format")
        format_layout = QGridLayout()

        format_layout.addWidget(QLabel("Format:"), 0, 0)
        format_combo = QComboBox()
        format_combo.addItems(["GIF (.gif)", "MP4 Video (.mp4)"])
        format_layout.addWidget(format_combo, 0, 1)

        format_layout.addWidget(QLabel("FPS:"), 1, 0)
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 60)
        fps_spin.setValue(5)
        format_layout.addWidget(fps_spin, 1, 1)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Info label
        info_label = QLabel(f"Animation will include {n_timesteps} frames\n"
                           f"Current field: {self.view_3d.current_field}")
        info_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(info_label)

        # Note about MP4
        note_label = QLabel("Note: MP4 export requires ffmpeg to be installed.")
        note_label.setStyleSheet("color: #ff9800; font-size: 11px;")
        layout.addWidget(note_label)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get selected format
        format_idx = format_combo.currentIndex()
        format_type = "gif" if format_idx == 0 else "mp4"
        fps = fps_spin.value()

        # Get output filename
        default_name = f"ptm3d_animation_{self.view_3d.current_field}"
        if format_type == "gif":
            file_filter = "GIF Images (*.gif);;All Files (*)"
            default_name += ".gif"
        else:
            file_filter = "MP4 Videos (*.mp4);;All Files (*)"
            default_name += ".mp4"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Animation",
            str(Path.cwd() / default_name),
            file_filter
        )

        if not filename:
            return

        # Stop playback if running
        if self.is_playing:
            self.toggle_playback()

        # Show progress dialog
        progress = QProgressDialog("Exporting animation...", "Cancel", 0, n_timesteps, self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        was_cancelled = False

        def update_progress(current, total):
            nonlocal was_cancelled
            progress.setValue(current)
            progress.setLabelText(f"Rendering frame {current}/{total}...")
            QApplication.processEvents()
            if progress.wasCanceled():
                was_cancelled = True

        # Export animation
        success = self.view_3d.export_animation(
            filename,
            format_type=format_type,
            fps=fps,
            progress_callback=update_progress
        )

        progress.close()

        if was_cancelled:
            QMessageBox.information(self, "Cancelled", "Animation export was cancelled.")
            # Try to delete partial file
            try:
                Path(filename).unlink(missing_ok=True)
            except Exception:
                pass
        elif success:
            QMessageBox.information(self, "Export Complete",
                                   f"Animation exported to:\n{filename}")
        else:
            QMessageBox.critical(self, "Export Failed",
                               "Failed to export animation.\n"
                               "For MP4 format, ensure ffmpeg is installed.")

    def show_export_data_dialog(self):
        """Show dialog for exporting data in GeoTIFF/HDF5 formats"""
        if not self.view_3d.mesh or not self.view_3d.results_dir:
            QMessageBox.warning(self, "No Data", "Please load simulation results first.")
            return

        if not self.view_3d.timesteps:
            QMessageBox.warning(self, "No Timesteps", "No timestep data available to export.")
            return

        dialog = ExportDataDialog(
            results_dir=str(self.view_3d.results_dir),
            mesh_data=self.view_3d.mesh_data,
            timesteps=self.view_3d.timesteps,
            parent=self
        )
        dialog.exec()

    def cleanup(self):
        """Clean up resources"""
        self.animation_timer.stop()
        self.view_3d.cleanup()
