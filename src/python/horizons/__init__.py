"""
JPL HORIZONS integration for PTM3D.

Provides search, orbital element retrieval, SPK download, and
body configuration YAML generation from the HORIZONS system.
"""

from .albedo import bond_albedo_from_pv, phase_integral
from .horizons_client import (
    HorizonsClient,
    HorizonsError,
    HorizonsHTTPError,
    HorizonsNetworkError,
    HorizonsNotFoundError,
    HorizonsParseError,
    HorizonsRefusalError,
    small_body_spk_command,
)
from .spk_generator import SPKGenerator
from .body_config_writer import BodyConfigWriter

__all__ = [
    "BodyConfigWriter",
    "HorizonsClient",
    # The five failures, exported so a caller can tell "the network is down"
    # from "HORIZONS refused this object" without matching on message text.
    "HorizonsError",
    "HorizonsHTTPError",
    "HorizonsNetworkError",
    "HorizonsNotFoundError",
    "HorizonsParseError",
    "HorizonsRefusalError",
    "SPKGenerator",
    "bond_albedo_from_pv",
    "phase_integral",
    "small_body_spk_command",
]
