"""ptm_ice package

Provides thermal/ice table utilities and pipeline for processing PTM3D outputs.

Primary modules:
- ice_table: Simple scalar mass-balance approximation
- ice_table_vectorized: Vectorized Newton solver (NumPy/Numba backends)
- ice_depth_pipeline: End-to-end pipeline for PTM3D temperature outputs
"""

from .ice_table import IceTable, compute_ice_table
from .ice_table_vectorized import (
	compute_ice_depths_vectorized,
	compute_ice_depths_numba,
	compute_ice_depths,
)
from .ice_depth_pipeline import (
	run_pipeline,
	stream_binary_mean_max,
	stream_vtk_mean_max,
	compute_ice_depth_from_stats,
)

__all__ = [
	# Scalar solver
	"IceTable",
	"compute_ice_table",
	# Vectorized solver
	"compute_ice_depths_vectorized",
	"compute_ice_depths_numba",
	"compute_ice_depths",
	# Pipeline
	"run_pipeline",
	"stream_binary_mean_max",
	"stream_vtk_mean_max",
	"compute_ice_depth_from_stats",
]
