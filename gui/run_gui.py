#!/usr/bin/env python3
"""
PTM3D GUI Launcher Script
Run this script to start the PTM3D GUI application.

Usage:
    python run_gui.py [OPTIONS]

Options:
    --project PATH    Load a project file on startup
    --results PATH    Load results directory on startup
    --help            Show this help message
"""

import sys
import os
import argparse

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Fix Qt plugin path for PyQt6 on macOS
def setup_qt_plugin_path():
    """Set up Qt plugin path to fix 'cocoa' plugin not found error"""
    try:
        import PyQt6
        qt6_path = os.path.dirname(PyQt6.__file__)
        plugin_path = os.path.join(qt6_path, 'Qt6', 'plugins')
        if os.path.exists(plugin_path):
            os.environ['QT_PLUGIN_PATH'] = plugin_path
            # Also set QT_QPA_PLATFORM_PLUGIN_PATH for platform plugins specifically
            platforms_path = os.path.join(plugin_path, 'platforms')
            if os.path.exists(platforms_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platforms_path
    except ImportError:
        pass

setup_qt_plugin_path()


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []

    # Core dependencies
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import yaml
    except ImportError:
        missing.append("PyYAML")

    # Optional but recommended
    optional_missing = []

    try:
        import pyvista
        import pyvistaqt
    except ImportError:
        optional_missing.append("pyvista pyvistaqt (3D visualization)")

    try:
        import matplotlib
    except ImportError:
        optional_missing.append("matplotlib (2D plotting)")

    if missing:
        print("ERROR: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    if optional_missing:
        print("WARNING: Missing optional dependencies (some features disabled):")
        for dep in optional_missing:
            print(f"  - {dep}")
        print()

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PTM3D - Planetary Thermophysical Model GUI"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Path to project file to load on startup"
    )
    parser.add_argument(
        "--results", "-r",
        type=str,
        help="Path to results directory to load on startup"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip dependency check"
    )

    args = parser.parse_args()

    # Check dependencies
    if not args.no_check:
        if not check_dependencies():
            sys.exit(1)

    # Import and run GUI
    try:
        from ptm3d_gui import main as run_app, PTM3DMainWindow
        from PyQt6.QtWidgets import QApplication

        # High DPI support
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

        app = QApplication(sys.argv)
        app.setApplicationName("PTM3D")
        app.setOrganizationName("PTM3D")
        app.setApplicationVersion("2.0")

        # Create main window
        window = PTM3DMainWindow()

        # Load project if specified
        if args.project:
            if os.path.exists(args.project):
                window.load_project(args.project)
            else:
                print(f"Warning: Project file not found: {args.project}")

        # Load results if specified
        if args.results:
            if os.path.isdir(args.results):
                window.results_panel.load_results(args.results)
                window.left_tabs.setCurrentWidget(window.results_panel)
            else:
                print(f"Warning: Results directory not found: {args.results}")

        window.show()
        sys.exit(app.exec())

    except ImportError as e:
        print(f"ERROR: Failed to import GUI modules: {e}")
        print("\nMake sure you're running from the gui/ directory or the path is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
