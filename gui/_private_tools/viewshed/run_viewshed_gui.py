#!/usr/bin/env python3
"""
Launcher for the PEGASUS Viewshed GUI.

Usage:
    python gui/run_viewshed_gui.py
"""

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

# This tool lives under gui/_private_tools/viewshed/ (not part of the public
# build). It needs BOTH its own package dir (viewshed_gui, viewshed_core,
# viewshed_panels) and the parent gui/ dir (gui_core, gui_panels,
# dashboard_core) on sys.path.
tool_dir = Path(__file__).resolve().parent   # _private_tools/viewshed
gui_dir = tool_dir.parents[1]                 # gui/
for _p in (str(tool_dir), str(gui_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viewshed_gui import main

if __name__ == "__main__":
    main()
