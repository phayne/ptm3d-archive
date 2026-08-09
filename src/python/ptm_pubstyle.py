"""Compatibility shim -- the implementation moved to ``ptm.pubstyle``.

The 41 figure scripts under ``src/python`` that reach this module by putting
their own directory on ``sys.path`` keep working unchanged. New code should
import from the package::

    from ptm.pubstyle import apply_style, COLORS, DOUBLE_W

Every name below is the one object defined in ``ptm/pubstyle.py``.
"""

from ptm.pubstyle import (  # noqa: F401
    COLORS,
    COLUMN_W,
    DOUBLE_W,
    HAS_PRETTY,
    LATITUDE_COLORS,
    apply_style,
    load_face_mask,
    panel_label,
    savefig_pinned,
)

__all__ = [
    "COLORS",
    "COLUMN_W",
    "DOUBLE_W",
    "HAS_PRETTY",
    "LATITUDE_COLORS",
    "apply_style",
    "load_face_mask",
    "panel_label",
    "savefig_pinned",
]
