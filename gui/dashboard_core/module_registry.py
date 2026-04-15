"""
Module registry: stores and retrieves dashboard modules by name.
"""

from typing import Dict, List, Optional
from .module_interface import AbstractModule


class ModuleRegistry:
    """Registry for dashboard modules."""

    def __init__(self):
        self._modules: Dict[str, AbstractModule] = {}
        self._order: List[str] = []

    def register(self, module: AbstractModule):
        """Register a module. Order is preserved for sidebar layout."""
        name = module.name()
        if name in self._modules:
            raise ValueError(f"Module '{name}' already registered")
        self._modules[name] = module
        self._order.append(name)

    def get(self, name: str) -> Optional[AbstractModule]:
        """Get a module by name."""
        return self._modules.get(name)

    def modules(self) -> List[AbstractModule]:
        """Return all modules in registration order."""
        return [self._modules[n] for n in self._order]

    def names(self) -> List[str]:
        """Return module names in registration order."""
        return list(self._order)

    def __len__(self):
        return len(self._modules)

    def __contains__(self, name: str):
        return name in self._modules
