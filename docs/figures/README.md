# Figure Placeholders

This directory contains placeholder files for figures referenced in the README.

## Figures to Generate

### 1. ptm3d_workflow_placeholder.png
**Description:** PTM3D simulation workflow diagram
**Content:** Flowchart showing:
- Input: Mesh + SPICE Kernels + Config
- Illumination Model (ray tracing, solar flux)
- Thermal Model (1D heat conduction)
- Radiation Exchange (view factors, radiosity) [optional loop]
- Output: Temperature maps, VTK files, statistics

### 2. architecture_placeholder.png
**Description:** Module architecture and dependencies
**Content:** Block diagram showing:
- Core modules: thermal, illumination, radiation, mesh, ice
- Optional: gpu (OpenCL/HIP/CUDA)
- Research: exosphere, viewshed
- External: Embree, SPICE, CFITSIO, Eigen

### 3. temperature_map_placeholder.png
**Description:** Example surface temperature visualization
**Content:** 3D mesh colored by temperature showing:
- Color scale from cold (blue) to hot (red)
- Day/night terminator visible
- Mesh resolution apparent

### 4. thermal_profile_placeholder.png
**Description:** Subsurface temperature profiles
**Content:** Plot showing:
- X-axis: Temperature (K)
- Y-axis: Depth (m, log scale)
- Multiple curves for different local times (0h, 6h, 12h, 18h)
- Thermal skin depth indicated

### 5. validation_placeholder.png
**Description:** Validation against Diviner observations
**Content:** Comparison plot showing:
- PTM3D predictions vs Diviner measurements
- Scatter plot or time series
- RMS error annotation

## Generating Figures

These figures can be generated using:
- ParaView for 3D visualizations
- Matplotlib for plots
- Draw.io or similar for diagrams

Example ParaView workflow:
```bash
# Convert simulation output
python src/python/vtk_converter.py output/simulation output/paraview

# Open in ParaView and screenshot
```

Example Matplotlib for thermal profiles:
```python
import matplotlib.pyplot as plt
import numpy as np

# Load thermal debug data
data = np.loadtxt('output/thermal_debug.txt')
# Plot depth profiles...
```
