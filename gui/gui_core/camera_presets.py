"""
Camera preset definitions for the PTM3D Camera Simulation GUI.
Each preset is a dict of camera intrinsic parameters that auto-fill the GUI fields.
Native resolution (native_nx/ny) is the full sensor pixel count; the GUI applies
a resolution scale factor to derive the actual rendering resolution.
"""

CAMERA_PRESETS = {
    "Nikon D5 @ 200mm": {
        "focal_length_mm": 200.0,
        "sensor_width_mm": 35.9,
        "sensor_height_mm": 23.9,
        "native_nx": 5568,
        "native_ny": 3712,
        "k1": -0.005,
        "k2": 0.0002,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "description": "Nikon D5 with AF-S NIKKOR 70-200mm f/2.8E at 200mm",
    },
    "Nikon D5 @ 100mm": {
        "focal_length_mm": 100.0,
        "sensor_width_mm": 35.9,
        "sensor_height_mm": 23.9,
        "native_nx": 5568,
        "native_ny": 3712,
        "k1": -0.008,
        "k2": 0.0003,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "description": "Nikon D5 with AF-S NIKKOR 70-200mm f/2.8E at 100mm",
    },
    "Nikon D5 @ 50mm": {
        "focal_length_mm": 50.0,
        "sensor_width_mm": 35.9,
        "sensor_height_mm": 23.9,
        "native_nx": 5568,
        "native_ny": 3712,
        "k1": -0.003,
        "k2": 0.0001,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "description": "Nikon D5 with AF-S NIKKOR 50mm f/1.4G",
    },
    "Generic 35mm": {
        "focal_length_mm": 50.0,
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "native_nx": 1024,
        "native_ny": 1024,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "description": "Generic full-frame 35mm pinhole camera",
    },
    "Custom": {
        "focal_length_mm": 50.0,
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "native_nx": 1024,
        "native_ny": 1024,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "description": "User-defined camera parameters",
    },
}

# Ordered list for the dropdown (Custom last)
PRESET_NAMES = [
    "Nikon D5 @ 200mm",
    "Nikon D5 @ 100mm",
    "Nikon D5 @ 50mm",
    "Generic 35mm",
    "Custom",
]

# Resolution scale options: label → fraction of native resolution
RESOLUTION_SCALES = [
    ("1/8 (preview)", 0.125),
    ("1/4 (fast)", 0.25),
    ("1/2 (balanced)", 0.5),
    ("Full", 1.0),
    ("Custom", None),
]
