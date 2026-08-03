"""
Module chaining: auto-links outputs from one module as inputs to another.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from .module_registry import ModuleRegistry


class ChainingManager(QObject):
    """Manages auto-linking of outputs between modules.

    When a module emits output_available(type, metadata), the chaining
    manager looks up registered chains and calls set_input() on targets.
    """

    chain_updated = pyqtSignal(str, str, str)  # source_module, output_key, value

    # Chain definitions: (source_module, output_key) -> [(target_module, input_key)]
    # The Exosphere and Viewshed modules were moved out of the public build
    # (gui/_private_tools/); their chain targets were removed with them.
    CHAINS = {
        ("dem", "mesh_file"): [
            ("thermal", "mesh_file"),
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry = None
        self._console = None

    def connect_modules(self, registry: ModuleRegistry, console=None):
        """Connect output_available signals from all registered modules."""
        self._registry = registry
        self._console = console

        for module in registry.modules():
            module.output_available.connect(self._on_output)

    def _on_output(self, output_type: str, metadata: dict):
        """Handle output_available signal from a module."""
        sender = self.sender()
        if sender is None:
            return

        source_name = sender.name()
        key = (source_name, output_type)
        value = metadata.get("path", "")

        if not value:
            return

        if key in self.CHAINS:
            for target_name, input_key in self.CHAINS[key]:
                if self._registry is None:
                    continue
                target = self._registry.get(target_name)
                if target:
                    target.set_input(input_key, value)
                    self.chain_updated.emit(source_name, output_type, value)

                    if self._console:
                        short_path = value
                        if len(short_path) > 60:
                            short_path = "..." + short_path[-57:]
                        self._console.append_info(
                            f"[CHAIN] {source_name} {output_type} "
                            f"-> {target_name}: {short_path}"
                        )
