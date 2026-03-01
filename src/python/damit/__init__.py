"""
DAMIT (Database of Asteroid Models from Inversion Techniques) integration for PTM3D.

Provides asteroid shape model download and OBJ conversion.
"""

from .damit_client import DAMITClient
from .shape_converter import ShapeConverter

__all__ = ["DAMITClient", "ShapeConverter"]
