#!/usr/bin/env python3
"""
PTM3D Unified Dashboard Launcher
Run this script to start the PTM3D Dashboard application.

Usage:
    python run_dashboard.py [OPTIONS]

Options:
    --project PATH    Load a project file on startup
    --results PATH    Load results directory on startup
    --help            Show this help message
"""

import sys
import os
import argparse

# Add this directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def setup_qt_plugin_path():
    """Set up Qt plugin path to fix 'cocoa' plugin not found error on macOS."""
    try:
        import PyQt6
        qt6_path = os.path.dirname(PyQt6.__file__)
        plugin_path = os.path.join(qt6_path, 'Qt6', 'plugins')
        if os.path.exists(plugin_path):
            os.environ['QT_PLUGIN_PATH'] = plugin_path
            platforms_path = os.path.join(plugin_path, 'platforms')
            if os.path.exists(platforms_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platforms_path
    except ImportError:
        pass


setup_qt_plugin_path()


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    for pkg, name in [
        ("PyQt6", "PyQt6"),
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)

    # Optional but recommended
    optional_missing = []
    for pkg, name in [
        ("pyvista", "pyvista"),
        ("pyvistaqt", "pyvistaqt"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            optional_missing.append(name)

    if missing:
        print("ERROR: Missing required dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall with: pip install " + " ".join(missing))
        sys.exit(1)

    if optional_missing:
        print("WARNING: Missing optional dependencies (some features disabled):")
        for m in optional_missing:
            print(f"  - {m}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PTM3D Unified Dashboard"
    )
    parser.add_argument("--project", help="Load a project file on startup")
    parser.add_argument("--results", help="Load results directory on startup")
    args = parser.parse_args()

    check_dependencies()

    # Enable high DPI scaling
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("PTM3D Dashboard")
    app.setOrganizationName("PTM3D")

    # Apply dark theme
    from gui_core.styles import ModernStyle
    app.setStyleSheet(ModernStyle.get_stylesheet())

    # Create main window
    from dashboard import UnifiedDashboard
    window = UnifiedDashboard()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
