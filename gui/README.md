# PTM3D GUI

Modern PyQt6-based graphical user interface for the Planetary Thermophysical Model simulation suite.

## Supported entry point

**`python run_gui.py` is the ONE supported way to launch the PTM3D GUI** (the
thermal simulation application, `ptm3d_gui.py`). Run it from the `gui/`
directory:

```bash
cd gui/
python run_gui.py
```

An optional multi-module launcher (`run_dashboard.py`) bundles the DEM/mesh
tools alongside the thermal app; it is a convenience wrapper and gets less
testing than `run_gui.py`.

Before a simulation launches, the GUI now runs a pre-flight check
(`gui_core/preflight.py`): paths must exist and be readable, cloud-synced
paths (`/Library/CloudStorage/...`) require an explicit override, and
"dataless" cloud placeholder files (evicted from local disk) block the launch
— reading one hangs the engine indefinitely. Keep simulation input data on a
local disk, not in a cloud-synced folder.

## Features

### 1. Simulation Configuration
- **Tabbed Interface**: Organized parameter settings for Simulation, Mesh, Thermal, Radiation, Illumination, and Output
- **YAML Preview**: Live preview of configuration as you edit
- **Presets**: Quick-start configurations for common scenarios
- **Validation**: Real-time configuration validation before running

### 2. 3D Visualization (PyVista)
- Interactive 3D mesh rendering with zoom, pan, and rotate
- Multiple data fields: Temperature, Illumination, Solar flux
- Face picking with detailed information
- Timestep animation with playback controls
- Export high-resolution images

### 3. 2D Analysis (Matplotlib)
- Temperature distribution histograms
- Temperature vs. latitude scatter plots
- Time series analysis
- Min/max/mean statistics over time
- Individual face time series on selection

### 4. Results Browser
- Browse simulation output directories
- View available data fields and timesteps
- Load and compare multiple results
- Export to various formats (VTK, NumPy, HDF5, NetCDF)

### 5. Console Output
- Real-time simulation progress
- Color-coded messages (info, success, warning, error)
- Auto-scroll with toggle
- Exportable log files

## Installation

### Prerequisites
- Python 3.9 or later
- PTM3D C++ simulation built and available

### Install Dependencies

```bash
cd gui/
pip install -r requirements.txt
```

### Required packages:
- `PyQt6` - GUI framework
- `pyvista` + `pyvistaqt` - 3D visualization
- `matplotlib` - 2D plotting
- `numpy` - Numerical operations
- `PyYAML` - Configuration file handling

### Optional packages (for enhanced features):
- `pyopencl` - GPU detection
- `h5py` - HDF5 export
- `netCDF4` - NetCDF export

## Usage

### Launch the GUI

```bash
# From the gui/ directory
python run_gui.py

# Or with a project file
python run_gui.py --project /path/to/config.ptm3d

# Or with results to visualize
python run_gui.py --results /path/to/output/
```

### Quick Start

1. **Configure Simulation**
   - Select the "Configuration" tab
   - Set simulation parameters (body, duration, timesteps)
   - Configure mesh resolution
   - Set thermal and illumination options
   - Choose output directory

2. **Save and Run**
   - Save project (Ctrl+S)
   - Click "Run" button or press F5
   - Monitor progress in Console

3. **Visualize Results**
   - Results load automatically after simulation
   - Switch between 3D view and 2D plots
   - Use timestep slider or Play button for animation
   - Click faces in 3D view for details

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New project |
| Ctrl+O | Open project |
| Ctrl+S | Save project |
| Ctrl+L | Load results |
| F5 | Run simulation |
| Shift+F5 | Stop simulation |
| F6 | Refresh visualization |
| Home | Reset camera |
| Ctrl+` | Toggle console |
| Ctrl+E | Export image |

## Project Structure

```
gui/
├── ptm3d_gui.py           # Main application
├── run_gui.py             # Launcher script
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── gui_core/              # Core modules
│   ├── __init__.py
│   ├── config_manager.py  # Configuration handling
│   ├── styles.py          # Modern dark theme
│   └── simulation_runner.py # Background execution
│
└── gui_panels/            # UI panels
    ├── __init__.py
    ├── config_panel.py    # Configuration tabs
    ├── visualization_panel.py  # 3D/2D views
    ├── results_panel.py   # Results browser
    └── console_panel.py   # Console output
```

## Configuration File Format

Projects are saved as YAML files (`.ptm3d` or `.yaml`):

```yaml
simulation:
  start_time: "2030-04-10T00:00:00"
  duration_days: 3.55
  timesteps: 144
  body: Moon

mesh:
  resolution: 5
  import_file: ""
  scale_factor: 1.0

thermal:
  run_thermal_mesh: true
  threads: 0  # auto
  equilibration_orbits: 2
  use_gpu: false

radiation:
  enabled: false
  use_hmatrix: false

output:
  directory: output/my_simulation
  format: binary
  vtk_mode: both
```

## Troubleshooting

### PyVista not showing 3D view
- Ensure VTK is properly installed: `pip install vtk`
- Try: `pip install --upgrade pyvista pyvistaqt`
- On macOS, you may need: `pip install pyobjc`

### High DPI scaling issues
- Set environment variable: `QT_ENABLE_HIGHDPI_SCALING=1`
- Or: `QT_SCALE_FACTOR=1.5`

### Simulation executable not found
- Ensure PTM3D is built: `cd build && make`
- Check the build path in simulation_runner.py

### Memory issues with large meshes
- Reduce mesh resolution
- Disable statistics computation
- Use H-matrix compression for radiation

## Development

### Adding New Configuration Options

1. Add default value in `gui_core/config_manager.py`
2. Add widget in appropriate tab in `gui_panels/config_panel.py`
3. Map to C++ command-line flag in `gui_core/simulation_runner.py`

### Adding New Visualization Features

1. Extend `PyVistaWidget` in `gui_panels/visualization_panel.py`
2. Add controls to the visualization panel
3. Connect signals to main window

## Building a distribution package

Build a public/colleague tarball with `git archive`, which honors the
repository `.gitattributes` `export-ignore` rules:

```bash
git archive --format=tar.gz -o ptm3d.tar.gz HEAD
```

This excludes per-developer settings (`.claude/`, `gui/.claude/`) and the
Paul-only tools staged under `gui/_private_tools/` (batch runs, exosphere,
viewshed). A raw `tar`/`cp -r` of the working tree does **not** honor those
rules and would leak the excluded paths — use `git archive`.

Some input datasets (SPICE kernels, DEMs, view factors) are large and are not
bundled; see the top-level data/distribution docs. If the GUI is run against a
cloud-synced folder whose files have been evicted to the cloud ("dataless"
placeholders), the pre-flight check will block the launch rather than let the
engine hang reading them — keep input data on a local disk.

## License

Part of the PTM3D simulation package.
