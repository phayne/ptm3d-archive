# ParaView Viewshed Animation Guide

This guide explains how to visualize your viewshed data in ParaView with the viewpoint prominently marked, and animate through visibility timesteps.

## Overview

Your viewshed PVD files contain:
- **visibility**: 0 (not visible from viewpoint) to 1 (visible)
- **distance_km**: Distance from viewpoint to each face
- **elevation_deg**: Elevation angle from viewpoint
- **azimuth_deg**: Azimuth angle from viewpoint
- **viewpoint_face**: A binary mask (1.0 for the viewpoint face, 0.0 elsewhere)

The `viewpoint_face` array marks which face the observer/rover is standing on, enabling you to highlight it visually.

## Quick Start

### Option 1: Automatic Setup (Recommended)

1. **In ParaView**, open your viewshed PVD file:
   ```
   File > Open Data > traverse_viewsheds.pvd
   ```

2. **Run the setup script** (inside ParaView):
   ```
   Macros > Edit Macros
   ```
   Paste the contents of `paraview_viewshed_setup.py` into the macro editor, then click **Execute** (or press Ctrl+Enter).

3. **Open the Animation View** (ParaView 5.13.2):
   - Look for the **Animation** panel at the bottom of the interface, or
   - Go to **View > Panels > Animation** (if available), or
   - Try **Windows > Dockable Windows > Animation** (some versions)

4. **Play the animation**:
   - Click the Play button in the Animation panel
   - The visibility data will animate through timesteps
   - The viewpoint face is highlighted in **bright red with yellow edges**

### Option 2: Manual Setup

If the script approach doesn't work, you can set this up manually in ParaView:

1. **Load the PVD file**:
   ```
   File > Open Data > traverse_viewsheds.pvd
   ```

2. **Color by visibility**:
   - Select the object in the pipeline
   - In the Properties panel, go to **Coloring**
   - Select **visibility** from the dropdown
   - Apply a colormap (e.g., "Blue to Red")
   - Set Opacity to ~0.8

3. **Highlight the viewpoint**:
   - In the pipeline, right-click on the loaded PVD
   - Select **Filters > Threshold...**
   - Set scalars to **viewpoint_face**
   - Set threshold range to **[0.5, 1.5]** (this selects only faces where viewpoint_face > 0.5)
   - Click **Apply**
   - In Properties, set color to **red** and increase **Line Width** to 2–3

3. **Set up animation** (ParaView 5.13.2):
   - Look for the **Animation** panel (usually at the bottom or right side of the interface)
   - If not visible, try: **Windows > Dockable Windows > Animation** or **View > Panels > Animation**
   - The animation should automatically detect timesteps from your PVD
   - Click **Play** to animate through timesteps
   - The visibility data will smoothly transition between frames
   - The viewpoint face stays highlighted in red throughout

4. **Configure animation mode** (optional):
   - In the Animation panel, look for options to:
     - Set **Loop** for continuous playback
     - Adjust **Frame Rate** (30 fps is typical)
     - Control playback speed

## Customization

### Change Viewpoint Marker Color

In the Threshold representation properties:
- **Ambient Color**: Base color
- **Diffuse Color**: Main color (set to RGB where R=1, G=0, B=0 for red)
- **Specular Color**: Highlight reflection (white for gloss)

Example: For **bright cyan**:
- Diffuse: [0, 1, 1]
- Ambient: [0, 0.5, 0.5]

### Change Visibility Colormap

With the main object selected:
1. Click the colored bar in the scalar bar legend
2. Choose a different preset (e.g., "Viridis", "Cool to Warm", "Rainbow")

### Scale the Mesh

If the mesh appears too small or large:
- In Properties, find **Geometry** section
- Use **Scale** to adjust

### Enhance Visibility

For better distinction:
- **Main visibility surface**: Set Representation to "Surface with Edges"
  - This adds black edges to show mesh cell boundaries
- **Viewpoint marker**: Keep as "Surface" with bright color and high opacity

## Animation Export (Advanced)

To export the animation as a movie file (ParaView 5.13.2):

1. In ParaView, go to **File > Export Animation** (or similar export option)
2. Choose a format (MP4, AVI, PNG sequence)
3. Configure resolution and frame rate
4. Click **Export**

ParaView will render each frame and save to the output file.

**Note:** If you don't see "Export Animation" under File, try:
- **File > Export** and select the format you want
- Look for animation/video export in the menus
- Alternatively, use Python scripting (see below)

## Troubleshooting

**"viewpoint_face array not found"**
- Your viewshed VTU may not include this array
- Regenerate viewsheds with recent code that marks viewpoint

**Animation not smooth**
- Increase number of frames in Animation settings
- Check **Windows > Dockable Windows** for the Animation panel (ParaView 5.13.2)
- Look for playback rate/speed controls in the Animation panel

**Red face not visible**
- Increase opacity of the threshold object to 1.0
- Increase **Edge Visibility** and **Line Width** in Properties
- Check that the threshold scalar range is correct ([0.5, 1.5])

**Can't find the Animation controls**
- Go to **Windows > Dockable Windows** and look for "Animation"
- Alternative menu: try **View > Panels > Animation** if available
- The Animation controls may be in a tab at the bottom of the main viewport
- Check the right sidebar for animation timeline/controls

## Python Scripting

For more advanced automation, you can write Python scripts directly in ParaView's Python console (`View > Python Console`):

```python
from paraview.simple import *

# Load PVD
reader = OpenDataFile('traverse_viewsheds.pvd')
view = GetActiveViewOrCreate('RenderView')

# Display with visibility coloring
Show(reader, view)
ColorBy(Show(reader, view), ('CELLS', 'visibility'))

# Threshold for viewpoint
threshold = Threshold(Input=reader, Scalars=['CELLS', 'viewpoint_face'], 
                      ThresholdMethod='Between', LowerThreshold=0.5, UpperThreshold=1.5)
threshold_display = Show(threshold, view)
threshold_display.DiffuseColor = [1.0, 0.0, 0.0]  # Red
threshold_display.LineWidth = 2.0

Render()
```

## Related Tools

- **vtk_converter_optimized.py**: Converts PTM3D outputs to VTU/PVD
- **netcdf_converter.py**: Converts VTU/PVD to NetCDF for other tools (e.g., Panoply)
- **dark_horse_viewshed**: C++ tool that generates the PVD files

## Questions?

See the PTM3D documentation for details on viewshed computation and thermal integration.
