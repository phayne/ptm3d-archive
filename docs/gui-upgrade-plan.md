# PTM3D Configuration GUI Upgrade Plan

## Executive Summary

This document outlines a plan to upgrade the PTM3D Configuration Builder GUI to expose all current simulation capabilities while retaining the existing "retro-futurism/art deco console" aesthetic. The current GUI covers approximately 60% of available options; this upgrade will achieve 100% coverage with improved organization and usability.

---

## 1. Gap Analysis: Current GUI vs PTM3D Capabilities

### 1.1 Options Currently in GUI (Covered)

| Category | Options | Status |
|----------|---------|--------|
| **Simulation** | start_time, duration_days, timesteps, body | ✓ Complete |
| **Mesh** | resolution, use_topographic, topo_directory, import_file, scale_factor, export_basic | ✓ Complete |
| **Thermal (Basic)** | run_thermal_mesh, threads, equilibration_orbits, compute_statistics, thermal_inertia_map | ✓ Complete |
| **Thermal (Integration)** | integration_days, integration_steps, integration_offset_days, integration_offset_steps | ✓ Complete |
| **Thermal (Shadow)** | enable_shadow_flux, crater_dd_ratio | ✓ Complete |
| **Illumination** | include_primary_reflection, use_batch_processing, albedo_map, albedo_maps_directory, positions_file | ✓ Complete |
| **Output** | directory, format, vtk_mode, include_primary_ir, locations_file, series_output, location_data, no_maps, depths_file | ✓ Complete |
| **Visualization** | create, width, height, fps, camera_distance, camera_latitude, camera_longitude, emission_angle, azimuth, save_frames | ✓ Complete |
| **Advanced** | verbose, default_visible_albedo | ✓ Complete |

### 1.2 Options Missing from GUI (Gaps)

#### **HIGH PRIORITY - Core Features**

| Category | Missing Option | CLI Flag | Description |
|----------|---------------|----------|-------------|
| **Thermal** | spectral_equilibration | `--spectral-equilibration` | FFT-based analytical equilibration (faster) |
| **Thermal** | use_gpu | `--use-gpu` | GPU acceleration for thermal solver |
| **Thermal** | use_gpu_illumination | `--use-gpu-illumination` | GPU ray tracing for illumination |
| **Thermal** | grid_quality | `--grid-quality` | Depth grid quality: coarse/standard/fine/auto |
| **Radiation** | enabled | `--radiation-exchange` | Enable radiation exchange |
| **Radiation** | min_shadow_fraction | `--rad-min-shadow` | Face selection threshold |
| **Radiation** | min_terrain_view_angle | `--rad-min-angle` | Face selection angle |
| **Radiation** | max_neighbors | `--rad-max-neighbors` | Max neighbors per face |
| **Radiation** | search_radius_factor | `--rad-search-radius` | Search radius factor |
| **Radiation** | max_iterations | `--rad-max-iter` | Radiosity solver iterations |
| **Radiation** | convergence_tolerance | `--rad-tolerance` | Solver convergence threshold |
| **Radiation** | verbose | `--rad-verbose` | Detailed progress output |
| **Radiation** | view_factor_file | `--rad-view-factor-file` | View factor cache file |
| **Radiation** | use_gpu | `--rad-use-gpu` | GPU radiosity solver |

#### **MEDIUM PRIORITY - Advanced Radiation Features**

| Category | Missing Option | CLI Flag | Description |
|----------|---------------|----------|-------------|
| **Radiation Coupling** | max_coupling_iterations | `--rad-coupling-iter` | Radiation-thermal coupling iterations |
| **Radiation Coupling** | coupling_temp_threshold | `--rad-coupling-temp-tol` | Temperature convergence [K] |
| **Radiation Coupling** | coupling_flux_threshold | `--rad-coupling-flux-tol` | Flux change threshold [W/m²] |
| **Radiation Coupling** | coupling_face_fraction | `--rad-coupling-face-frac` | Fraction of faces for stop |
| **H-Matrix** | use_hmatrix | `--rad-hmatrix` | Enable H-matrix compression |
| **H-Matrix** | hmatrix_tolerance | `--rad-hmatrix-tol` | SVD truncation tolerance |
| **H-Matrix** | hmatrix_max_depth | `--rad-hmatrix-depth` | Maximum quadtree depth |
| **H-Matrix** | hmatrix_min_block_size | `--rad-hmatrix-block-size` | Minimum block size |
| **H-Matrix** | hmatrix_max_rank | `--rad-hmatrix-rank` | Maximum rank for low-rank approx |

#### **LOW PRIORITY - Output Customization**

| Category | Missing Option | CLI Flag | Description |
|----------|---------------|----------|-------------|
| **Output** | thermal_vtk_file | `--thermal-vtk-file` | Custom thermal VTK filename |
| **Output** | complete_vtk_file | `--complete-vtk-file` | Custom complete VTK filename |

### 1.3 Summary Statistics

- **Total Options in PTM3D**: ~65
- **Options in Current GUI**: ~40 (62%)
- **Options Missing**: ~25 (38%)
- **After Upgrade**: 65 (100%)

---

## 2. Proposed Tab Reorganization

### 2.1 Current Tab Structure (7 tabs)
```
⚙ SIMULATION | ◬ MESH | ♨ THERMAL | ☀ ILLUMINATION | 💾 OUTPUT | 📊 VISUALIZATION | ⚡ ADVANCED
```

### 2.2 Proposed Tab Structure (8 tabs)
```
⚙ SIMULATION | ◬ MESH | ♨ THERMAL | ☢ RADIATION | ☀ ILLUMINATION | 💾 OUTPUT | 📊 VISUALIZATION | ⚡ ADVANCED
```

**Rationale**: Radiation exchange is a major feature with 20+ parameters. It deserves its own dedicated tab rather than being buried in "Advanced" or split across tabs.

---

## 3. Detailed Tab Specifications

### 3.1 ⚙ SIMULATION Tab (Unchanged)
```
┌─────────────────────────────────────────────────────────────┐
│  TIME PARAMETERS                                            │
├─────────────────────────────────────────────────────────────┤
│  Start Time:        [2025-01-01T00:00:00_____]             │
│  Duration (days):   [3.55_______]                          │
│  Time Steps:        [72_________]                          │
├─────────────────────────────────────────────────────────────┤
│  PLANETARY BODY                                             │
├─────────────────────────────────────────────────────────────┤
│  Target Body:       [Europa ▾]                             │
│                                                             │
│  ℹ Configure fundamental simulation parameters.             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ◬ MESH Tab (Unchanged)
No changes needed - already complete.

### 3.3 ♨ THERMAL Tab (Major Updates)

**NEW: GPU Acceleration Section**
**NEW: Equilibration Method Section**
**NEW: Grid Quality Section**

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL CONTROL                                              │
├─────────────────────────────────────────────────────────────┤
│  [✓] Run Thermal Model                                     │
│  CPU Threads:       [0_________] (0 = auto)                │
│  [✓] Compute Statistics                                    │
├─────────────────────────────────────────────────────────────┤
│  GPU ACCELERATION                                     [NEW] │
├─────────────────────────────────────────────────────────────┤
│  [ ] Use GPU for Thermal Solver (OpenCL)                   │
│  [ ] Use GPU for Illumination Ray Tracing                  │
│                                                             │
│  ⚡ GPU Status: AMD Radeon Pro W6800X (32 GB) ✓            │
├─────────────────────────────────────────────────────────────┤
│  EQUILIBRATION                                              │
├─────────────────────────────────────────────────────────────┤
│  Method:            [Spectral (FFT) ▾]  ← NEW              │
│                     • Time-Stepping (legacy)                │
│                     • Spectral (FFT) - faster               │
│  Equilibration Orbits: [10_______]                         │
├─────────────────────────────────────────────────────────────┤
│  DEPTH GRID                                           [NEW] │
├─────────────────────────────────────────────────────────────┤
│  Grid Quality:      [auto ▾]                               │
│                     • coarse (~40 layers, fast)            │
│                     • standard                              │
│                     • fine (~100 layers, accurate)         │
│                     • auto (adapts to material/GPU)        │
├─────────────────────────────────────────────────────────────┤
│  INTEGRATION WINDOW                                         │
├─────────────────────────────────────────────────────────────┤
│  Integration Days:  [-1.0______] (-1 = use output window)  │
│  Integration Steps: [-1________]                           │
│  Offset (days):     [0.0_______]                           │
│  Offset (steps):    [0_________]                           │
├─────────────────────────────────────────────────────────────┤
│  SHADOW/CRATER FLUX                                         │
├─────────────────────────────────────────────────────────────┤
│  [ ] Enable Shadow Flux                                    │
│  Crater D/h Ratio:  [8.0_______]                           │
├─────────────────────────────────────────────────────────────┤
│  THERMAL PROPERTY MAPS                                      │
├─────────────────────────────────────────────────────────────┤
│  Thermal Inertia Map: [____________________] [Browse...]   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 ☢ RADIATION Tab (NEW)

This is an entirely new tab consolidating all radiation exchange parameters.

```
┌─────────────────────────────────────────────────────────────┐
│  ☢ RADIATION EXCHANGE                                       │
├─────────────────────────────────────────────────────────────┤
│  ENABLE/DISABLE                                             │
├─────────────────────────────────────────────────────────────┤
│  [✓] Enable Radiation Exchange Between Faces               │
│  [ ] Verbose Progress Output                               │
│  [ ] Use GPU for Radiosity Solver (OpenCL)                 │
├─────────────────────────────────────────────────────────────┤
│  FACE SELECTION CRITERIA                                    │
├─────────────────────────────────────────────────────────────┤
│  Min Shadow Fraction:    [0.3_______] (0-1)                │
│  Min Terrain View Angle: [5.0_______] degrees              │
│                                                             │
│  ℹ Faces with shadow fraction < threshold are excluded     │
│    from radiation exchange to improve performance.         │
├─────────────────────────────────────────────────────────────┤
│  NEIGHBOR SEARCH                                            │
├─────────────────────────────────────────────────────────────┤
│  Max Neighbors per Face: [2000______]                      │
│  Search Radius Factor:   [10.0______] × √(face area)       │
├─────────────────────────────────────────────────────────────┤
│  RADIOSITY SOLVER                                           │
├─────────────────────────────────────────────────────────────┤
│  Max Iterations:         [100_______]                      │
│  Convergence Tolerance:  [0.01______]                      │
├─────────────────────────────────────────────────────────────┤
│  THERMAL-RADIATION COUPLING                                 │
├─────────────────────────────────────────────────────────────┤
│  Max Coupling Iterations:[10________]                      │
│  Temperature Tolerance:  [0.5_______] K                    │
│  Flux Change Tolerance:  [0.1_______] W/m²                 │
│  Face Fraction to Stop:  [0.01______] (1%)                 │
│                                                             │
│  ℹ Coupling iterations stop when < X% of faces still show  │
│    temperature changes above threshold.                     │
├─────────────────────────────────────────────────────────────┤
│  VIEW FACTOR CACHING                                        │
├─────────────────────────────────────────────────────────────┤
│  View Factor File: [output/view_factors.bin__] [Browse...] │
│                                                             │
│  ℹ Save/load view factors to skip recomputation.           │
│    Use .hmat extension for H-matrix format.                │
├─────────────────────────────────────────────────────────────┤
│  H-MATRIX COMPRESSION (Large Meshes)                        │
├─────────────────────────────────────────────────────────────┤
│  [ ] Enable H-Matrix Compression                           │
│                                                             │
│  When enabled (reduces memory 2-4× for >100K faces):       │
│  SVD Tolerance:          [0.01______] (1% error)           │
│  Max Quadtree Depth:     [10________]                      │
│  Min Block Size:         [64________] faces                │
│  Max Low-Rank:           [50________]                      │
│                                                             │
│  ⚠ H-matrix requires recomputation of view factors.        │
│    Delete existing view factor file to use new format.     │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 ☀ ILLUMINATION Tab (Unchanged)
No changes needed - already complete.

### 3.6 💾 OUTPUT Tab (Minor Updates)

Add custom VTK filename fields:

```
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT DESTINATION                                         │
├─────────────────────────────────────────────────────────────┤
│  Output Directory:  [output______________] [Browse...]     │
│  Format:            [binary ▾]                             │
│  VTK Mode:          [both ▾]                               │
│  [ ] Include Primary IR (VTK)                              │
├─────────────────────────────────────────────────────────────┤
│  CUSTOM VTK FILENAMES (Optional)                      [NEW] │
├─────────────────────────────────────────────────────────────┤
│  Thermal VTK File:  [________________________]             │
│  Complete VTK File: [________________________]             │
├─────────────────────────────────────────────────────────────┤
│  LOCATION EXPORTS                                           │
├─────────────────────────────────────────────────────────────┤
│  ... (existing fields)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.7 📊 VISUALIZATION Tab (Unchanged)
No changes needed - already complete.

### 3.8 ⚡ ADVANCED Tab (Unchanged)
No changes needed - already complete.

---

## 4. UI/UX Enhancements

### 4.1 Collapsible Sections

For the new RADIATION tab with many parameters, implement collapsible sections:

```python
class CollapsibleSection(QWidget):
    """Art deco styled collapsible section"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(f"▼ {title}")
        self.content_area = QWidget()
        self.is_collapsed = False

    def toggle(self):
        self.is_collapsed = not self.is_collapsed
        self.content_area.setVisible(not self.is_collapsed)
        symbol = "▶" if self.is_collapsed else "▼"
        self.toggle_button.setText(f"{symbol} {self.title}")
```

**Styling:**
```css
QPushButton#collapsibleHeader {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a2347, stop:0.5 #141b3d, stop:1 #1a2347);
    border: 1px solid #00ffff;
    border-radius: 3px;
    color: #00ffff;
    font-weight: bold;
    text-align: left;
    padding: 8px 15px;
}

QPushButton#collapsibleHeader:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a2347, stop:0.5 #1e2854, stop:1 #1a2347);
    border-color: #00ff00;
}
```

### 4.2 GPU Status Indicator

Add a live GPU status indicator to the THERMAL tab:

```python
class GPUStatusIndicator(QWidget):
    """Displays GPU availability and info"""

    def __init__(self):
        super().__init__()
        self.status_label = QLabel("⚡ GPU Status:")
        self.info_label = QLabel("Checking...")
        self.check_gpu()

    def check_gpu(self):
        # Try to detect OpenCL devices
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    dev = devices[0]
                    mem_gb = dev.global_mem_size / (1024**3)
                    self.info_label.setText(f"✓ {dev.name} ({mem_gb:.0f} GB)")
                    self.info_label.setStyleSheet("color: #00ff00;")
                    return
            self.info_label.setText("✗ No GPU found")
            self.info_label.setStyleSheet("color: #ff8800;")
        except ImportError:
            self.info_label.setText("⚠ PyOpenCL not installed")
            self.info_label.setStyleSheet("color: #ffff00;")
```

### 4.3 Preset Updates

Add new presets for radiation-enabled simulations:

```python
PRESETS = {
    # ... existing presets ...

    "crater_radex": {
        "simulation": {
            "body": "Moon",
            "duration_days": 29.5,
            "timesteps": 96
        },
        "mesh": {
            "import_file": ""  # User must specify
        },
        "thermal": {
            "run_thermal_mesh": True,
            "spectral_equilibration": True,
            "equilibration_orbits": 0  # Not needed with spectral
        },
        "radiation": {
            "enabled": True,
            "verbose": True,
            "min_shadow_fraction": 0.3,
            "max_coupling_iterations": 10
        }
    },

    "large_mesh_hmatrix": {
        "radiation": {
            "enabled": True,
            "use_hmatrix": True,
            "hmatrix_tolerance": 0.01,
            "view_factor_file": "view_factors.hmat"
        }
    },

    "gpu_accelerated": {
        "thermal": {
            "use_gpu": True,
            "spectral_equilibration": True
        },
        "radiation": {
            "enabled": True,
            "use_gpu": True
        }
    }
}
```

### 4.4 Validation Enhancements

Add validation rules for new parameters:

```python
def validate(self):
    errors = []

    # ... existing validations ...

    # Radiation exchange validations
    rad = self.config.get("radiation", {})
    if rad.get("enabled"):
        if rad.get("min_shadow_fraction", 0) < 0 or rad.get("min_shadow_fraction", 0) > 1:
            errors.append("Radiation: min_shadow_fraction must be between 0 and 1")
        if rad.get("max_neighbors", 0) < 10:
            errors.append("Radiation: max_neighbors should be at least 10")
        if rad.get("convergence_tolerance", 0) <= 0:
            errors.append("Radiation: convergence_tolerance must be positive")

    # H-matrix validations
    if rad.get("use_hmatrix"):
        if rad.get("hmatrix_tolerance", 0) <= 0 or rad.get("hmatrix_tolerance", 0) > 1:
            errors.append("H-matrix: tolerance must be between 0 and 1")
        if rad.get("hmatrix_max_rank", 0) < 1:
            errors.append("H-matrix: max_rank must be positive")

    # GPU validations
    thermal = self.config.get("thermal", {})
    if thermal.get("use_gpu") or thermal.get("use_gpu_illumination") or rad.get("use_gpu"):
        # Warn if GPU not available (non-fatal)
        if not self._check_gpu_available():
            errors.append("Warning: GPU requested but OpenCL not detected")

    return (len([e for e in errors if not e.startswith("Warning")]) == 0, errors)
```

### 4.5 Tooltips with Defaults

Enhance tooltips to show default values:

```python
def add_double_field(self, layout, row, label_text, key, min_val, max_val,
                     decimals, tooltip="", default=None):
    """Add a double input field with default value shown in tooltip"""
    if default is not None:
        tooltip = f"{tooltip}\n\nDefault: {default}"
    # ... rest of implementation
```

---

## 5. Config Manager Updates

### 5.1 DEFAULT_CONFIG Updates

```python
DEFAULT_CONFIG = {
    # ... existing sections ...

    "thermal": {
        "run_thermal_mesh": True,
        "threads": 0,
        "equilibration_orbits": 10,
        "spectral_equilibration": False,  # NEW
        "use_gpu": False,                  # NEW
        "use_gpu_illumination": False,     # NEW
        "grid_quality": "auto",            # NEW
        "compute_statistics": True,
        "thermal_inertia_map": "",
        "enable_shadow_flux": False,
        "crater_dd_ratio": 8.0,
        "integration_days": -1.0,
        "integration_steps": -1,
        "integration_offset_days": 0.0,
        "integration_offset_steps": 0
    },

    "radiation": {                         # NEW SECTION
        "enabled": False,
        "min_shadow_fraction": 0.3,
        "min_terrain_view_angle": 5.0,
        "max_neighbors": 2000,
        "search_radius_factor": 10.0,
        "max_iterations": 100,
        "convergence_tolerance": 0.01,
        "verbose": False,
        "use_gpu": False,
        "view_factor_file": "",
        "max_coupling_iterations": 10,
        "coupling_temp_threshold": 0.5,
        "coupling_flux_threshold": 0.1,
        "coupling_face_fraction": 0.01,
        "use_hmatrix": False,
        "hmatrix_tolerance": 0.01,
        "hmatrix_max_depth": 10,
        "hmatrix_min_block_size": 64,
        "hmatrix_max_rank": 50
    },

    "output": {
        # ... existing fields ...
        "thermal_vtk_file": "",           # NEW
        "complete_vtk_file": ""           # NEW
    }
}
```

---

## 6. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Update `config_manager.py` with new DEFAULT_CONFIG structure
2. Add radiation section to YAML generation/parsing
3. Create `RadiationTab` class in `gui_tabs.py`
4. Update main window to include new tab

### Phase 2: Thermal Tab Updates (Week 1)
1. Add GPU acceleration section
2. Add equilibration method dropdown
3. Add grid quality dropdown
4. Add GPU status indicator

### Phase 3: Radiation Tab Implementation (Week 2)
1. Implement all radiation exchange parameters
2. Add thermal-radiation coupling section
3. Add H-matrix compression section
4. Add collapsible sections for organization

### Phase 4: UI Enhancements (Week 2)
1. Implement `CollapsibleSection` widget
2. Add `GPUStatusIndicator` widget
3. Update presets
4. Enhance validation

### Phase 5: Testing & Polish (Week 3)
1. Test all parameter combinations
2. Verify YAML round-trip
3. Test with actual simulations
4. Update help text and tooltips

---

## 7. File Changes Summary

| File | Changes |
|------|---------|
| `config_manager.py` | Add radiation section, new thermal fields, output fields |
| `gui_tabs.py` | Add RadiationTab, update ThermalTab, minor OutputTab changes |
| `ptm_config_gui.py` | Add radiation tab to tab widget |
| `gui_styles.py` | Add collapsible section styles, GPU indicator styles |
| `gui_control_panel.py` | No changes needed |

### Estimated Lines of Code
- **New code**: ~600 lines (RadiationTab, CollapsibleSection, GPUIndicator)
- **Modified code**: ~150 lines (config_manager, thermal tab, output tab)
- **Total**: ~750 lines

---

## 8. Questions for User

1. **Tab Order**: Should the RADIATION tab come before or after THERMAL?
   - Option A: `THERMAL | RADIATION` (workflow order - thermal first, then radiation)
   - Option B: `RADIATION | THERMAL` (radiation modifies thermal, so show first)

2. **Collapsible Sections**: Should advanced radiation parameters (H-matrix, coupling) be collapsed by default?
   - Option A: All expanded (full visibility)
   - Option B: Advanced sections collapsed by default (cleaner initial view)

3. **GPU Status Check**: Should the GUI actively probe for OpenCL on startup?
   - This adds a small delay but gives immediate feedback
   - Alternative: Check only when GPU checkbox is clicked

4. **Presets**: Should existing presets be updated to use new features (spectral equilibration, GPU)?
   - This would change default behavior
   - Alternative: Add new presets, keep existing ones unchanged
