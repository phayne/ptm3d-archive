#!/usr/bin/env python3
"""Launch the PTM Exosphere GUI application."""

import os
import sys
from pathlib import Path

# Fix Qt plugin discovery on macOS (conda environments often lack QT_PLUGIN_PATH)
if "QT_PLUGIN_PATH" not in os.environ:
    try:
        import PyQt6
        plugin_path = str(Path(PyQt6.__file__).parent / "Qt6" / "plugins")
        if Path(plugin_path).is_dir():
            os.environ["QT_PLUGIN_PATH"] = plugin_path
    except ImportError:
        pass

# Ensure gui/ directory is in sys.path
gui_dir = str(Path(__file__).parent)
if gui_dir not in sys.path:
    sys.path.insert(0, gui_dir)

from exosphere_gui import main

if __name__ == "__main__":
    main()
