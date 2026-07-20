"""
Abstract base class for all dashboard modules.
Each module (DEM, Thermal, Exosphere, Viewshed) implements this interface.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget


class AbstractModule(QObject):
    """Base class for dashboard modules.

    Modules are composed into the dashboard via a QStackedWidget.
    Each module owns its config manager, panels, and simulation thread.
    """

    # Emitted when a module produces output that other modules can consume.
    # Args: (output_type: str, metadata: dict)
    #   output_type: e.g. "mesh_file", "output_dir"
    #   metadata: {"path": "/abs/path", ...}
    output_available = pyqtSignal(str, dict)

    # Emitted when the module's run status changes.
    # Args: (status: str)  — "idle", "running", "completed", "failed"
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = "idle"

    # -- Identity ----------------------------------------------------------

    def name(self) -> str:
        """Short machine-readable name, e.g. 'dem', 'thermal'."""
        raise NotImplementedError

    def display_name(self) -> str:
        """Human-readable label for sidebar button."""
        raise NotImplementedError

    def icon_text(self) -> str:
        """Short text shown on the sidebar button (2-4 chars)."""
        raise NotImplementedError

    # -- Widget ------------------------------------------------------------

    def widget(self) -> QWidget:
        """Return the module's main content widget for the QStackedWidget."""
        raise NotImplementedError

    # -- Configuration -----------------------------------------------------

    def config_manager(self):
        """Return the module's config manager instance (or None)."""
        return None

    # -- Chaining ----------------------------------------------------------

    def set_input(self, input_type: str, value: str):
        """Receive a chained input from another module.

        Called by ChainingManager when an upstream module produces output.
        Subclasses should update their config manager and refresh UI.
        """
        pass

    def get_outputs(self) -> dict:
        """Return current output paths/metadata as {type: path}."""
        return {}

    # -- Simulation --------------------------------------------------------

    def run_simulation(self):
        """Start the module's primary action (simulation, conversion, etc.)."""
        pass

    def stop_simulation(self):
        """Stop a running simulation."""
        pass

    def is_running(self) -> bool:
        """Return True if the module is currently running."""
        return self._status == "running"

    def status(self) -> str:
        return self._status

    def _set_status(self, status: str):
        self._status = status
        self.status_changed.emit(status)

    # -- Lifecycle ---------------------------------------------------------

    def on_activated(self):
        """Called when this module's page becomes visible in the dashboard."""
        pass

    def on_deactivated(self):
        """Called when this module's page is hidden."""
        pass
