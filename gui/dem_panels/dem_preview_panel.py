"""
DEM Preview Panel: GeoTIFF elevation preview with ROI selection.
"""

import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout,
    QCheckBox, QLineEdit, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class DEMPreviewPanel(QWidget):
    """GeoTIFF preview with interactive ROI selection."""

    roi_selected = pyqtSignal(float, float, float)  # lat, lon, radius_km
    dem_loaded = pyqtSignal(str)  # path to loaded DEM

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self._cm = config_manager
        self._dem_data = None
        self._dem_extent = None  # (lon_min, lon_max, lat_min, lat_max)
        self._dem_path = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # File selection
        file_group = QGroupBox("DEM Input")
        file_layout = QHBoxLayout(file_group)

        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Select a GeoTIFF DEM file...")
        self._file_edit.setReadOnly(True)
        file_layout.addWidget(self._file_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dem)
        file_layout.addWidget(browse_btn)

        layout.addWidget(file_group)

        # DEM info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Matplotlib canvas for elevation preview
        if HAS_MATPLOTLIB:
            self._figure = Figure(figsize=(6, 5), dpi=100)
            self._figure.set_facecolor('#1e1e1e')
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._ax = self._figure.add_subplot(111)
            self._ax.set_facecolor('#2d2d2d')
            self._ax.tick_params(colors='#a0a0a0')
            self._ax.set_title("DEM Preview", color='#e0e0e0', fontsize=11)
            layout.addWidget(self._canvas, 1)
        else:
            placeholder = QLabel("matplotlib not available for DEM preview")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder, 1)

        # ROI selection controls
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QFormLayout(roi_group)

        self._roi_enabled = QCheckBox("Enable ROI extraction")
        self._roi_enabled.toggled.connect(self._on_roi_toggled)
        roi_layout.addRow(self._roi_enabled)

        self._roi_lat = QDoubleSpinBox()
        self._roi_lat.setRange(-90.0, 90.0)
        self._roi_lat.setDecimals(4)
        self._roi_lat.setSuffix(" deg")
        roi_layout.addRow("Center Latitude:", self._roi_lat)

        self._roi_lon = QDoubleSpinBox()
        self._roi_lon.setRange(-360.0, 360.0)
        self._roi_lon.setDecimals(4)
        self._roi_lon.setSuffix(" deg")
        roi_layout.addRow("Center Longitude:", self._roi_lon)

        self._roi_radius = QDoubleSpinBox()
        self._roi_radius.setRange(0.1, 1000.0)
        self._roi_radius.setValue(10.0)
        self._roi_radius.setDecimals(1)
        self._roi_radius.setSuffix(" km")
        roi_layout.addRow("Radius:", self._roi_radius)

        self._roi_max_points = QSpinBox()
        self._roi_max_points.setRange(1000, 10000000)
        self._roi_max_points.setValue(500000)
        self._roi_max_points.setSingleStep(100000)
        roi_layout.addRow("Max Points:", self._roi_max_points)

        apply_roi_btn = QPushButton("Apply ROI")
        apply_roi_btn.clicked.connect(self._apply_roi)
        roi_layout.addRow(apply_roi_btn)

        layout.addWidget(roi_group)

        self._on_roi_toggled(False)

    def _browse_dem(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select DEM File",
            str(Path.home()),
            "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if filepath:
            self._load_dem(filepath)

    def _load_dem(self, filepath):
        """Load and display a GeoTIFF DEM."""
        if not HAS_GDAL:
            self._info_label.setText(
                "GDAL Python bindings (osgeo) not available; DEM preview is "
                "disabled.\n"
                "Install with: conda install -c conda-forge gdal\n"
                "(pip install GDAL needs the GDAL C library already present "
                "and matching -- see INSTALL.md, 'DEM / DTM tools'.)"
            )
            self._file_edit.setText(filepath)
            self._cm.set_value("input.dem_file", filepath)
            self.dem_loaded.emit(filepath)
            return

        try:
            ds = gdal.Open(filepath, gdal.GA_ReadOnly)
            if ds is None:
                QMessageBox.warning(self, "Error", f"Could not open: {filepath}")
                return

            # Read metadata
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            gt = ds.GetGeoTransform()
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()

            # Read elevation data (downsampled for preview if large)
            preview_max = 2000
            x_step = max(1, cols // preview_max)
            y_step = max(1, rows // preview_max)
            data = band.ReadAsArray(
                buf_xsize=cols // x_step,
                buf_ysize=rows // y_step
            ).astype(np.float64)

            # Mask nodata
            if nodata is not None:
                data[data == nodata] = np.nan

            # Compute geographic extent
            lon_min = gt[0]
            lon_max = gt[0] + cols * gt[1]
            lat_max = gt[3]
            lat_min = gt[3] + rows * gt[5]

            self._dem_data = data
            self._dem_extent = (lon_min, lon_max, lat_min, lat_max)
            self._dem_path = filepath

            ds = None  # Close dataset

            # Update UI
            self._file_edit.setText(filepath)
            self._cm.set_value("input.dem_file", filepath)

            valid_data = data[np.isfinite(data)]
            self._info_label.setText(
                f"Size: {cols} x {rows} pixels | "
                f"Elevation: {valid_data.min():.0f} to {valid_data.max():.0f} m | "
                f"Lat: [{lat_min:.3f}, {lat_max:.3f}] | "
                f"Lon: [{lon_min:.3f}, {lon_max:.3f}]"
            )

            # Set default ROI center to DEM center
            self._roi_lat.setValue((lat_min + lat_max) / 2)
            self._roi_lon.setValue((lon_min + lon_max) / 2)

            self._plot_dem()
            self.dem_loaded.emit(filepath)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load DEM:\n{e}")

    def _plot_dem(self):
        """Render the DEM elevation on the matplotlib canvas."""
        if not HAS_MATPLOTLIB or self._dem_data is None:
            return

        self._ax.clear()
        self._ax.set_facecolor('#2d2d2d')

        im = self._ax.imshow(
            self._dem_data,
            extent=self._dem_extent,
            cmap='terrain',
            aspect='auto',
            origin='upper'
        )

        self._ax.set_xlabel("Longitude (deg)", color='#a0a0a0', fontsize=10)
        self._ax.set_ylabel("Latitude (deg)", color='#a0a0a0', fontsize=10)
        self._ax.set_title("DEM Elevation", color='#e0e0e0', fontsize=11)
        self._ax.tick_params(colors='#a0a0a0', labelsize=9)

        # Draw ROI circle if enabled
        if self._roi_enabled.isChecked():
            self._draw_roi_circle()

        # Colorbar
        if hasattr(self, '_colorbar') and self._colorbar:
            self._colorbar.remove()
        self._colorbar = self._figure.colorbar(im, ax=self._ax, label='Elevation (m)')
        self._colorbar.ax.yaxis.label.set_color('#a0a0a0')
        self._colorbar.ax.tick_params(colors='#a0a0a0')

        self._figure.tight_layout()
        self._canvas.draw()

    def _draw_roi_circle(self):
        """Draw the ROI circle on the DEM plot."""
        if self._dem_extent is None:
            return

        lat = self._roi_lat.value()
        lon = self._roi_lon.value()
        radius_km = self._roi_radius.value()

        # Approximate radius in degrees
        # 1 degree latitude ~ 111 km (varies slightly with body)
        body_radius_m = self._cm.get_value("input.body_radius_m", 1737400.0)
        km_per_deg = (body_radius_m / 1000.0) * np.pi / 180.0
        radius_deg = radius_km / km_per_deg

        circle = plt.Circle(
            (lon, lat), radius_deg,
            fill=False, color='#FF4444', linewidth=2, linestyle='--'
        )
        self._ax.add_patch(circle)

        self._ax.plot(lon, lat, '+', color='#FF4444', markersize=12, markeredgewidth=2)

    def _on_roi_toggled(self, enabled):
        self._roi_lat.setEnabled(enabled)
        self._roi_lon.setEnabled(enabled)
        self._roi_radius.setEnabled(enabled)
        self._roi_max_points.setEnabled(enabled)
        self._cm.set_value("roi.enabled", enabled)

        if self._dem_data is not None:
            self._plot_dem()

    def _apply_roi(self):
        """Apply ROI settings to config and emit signal."""
        lat = self._roi_lat.value()
        lon = self._roi_lon.value()
        radius = self._roi_radius.value()
        max_pts = self._roi_max_points.value()

        self._cm.set_value("roi.enabled", self._roi_enabled.isChecked())
        self._cm.set_value("roi.center_lat", lat)
        self._cm.set_value("roi.center_lon", lon)
        self._cm.set_value("roi.radius_km", radius)
        self._cm.set_value("roi.max_points", max_pts)

        if self._dem_data is not None:
            self._plot_dem()

        self.roi_selected.emit(lat, lon, radius)

    def load_from_config(self):
        """Refresh UI from config manager."""
        dem_file = self._cm.get_value("input.dem_file", "")
        if dem_file and dem_file != self._dem_path:
            self._load_dem(dem_file)

        self._roi_enabled.setChecked(self._cm.get_value("roi.enabled", False))
        self._roi_lat.setValue(self._cm.get_value("roi.center_lat", 0.0))
        self._roi_lon.setValue(self._cm.get_value("roi.center_lon", 0.0))
        self._roi_radius.setValue(self._cm.get_value("roi.radius_km", 10.0))
        self._roi_max_points.setValue(self._cm.get_value("roi.max_points", 500000))
