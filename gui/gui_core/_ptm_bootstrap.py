"""Make the ``ptm`` package importable from a plain checkout.

``gui_core`` imports the shared PTM3D Python layer (``ptm.results``,
``ptm.run``). That package lives at ``src/python/ptm`` and is normally on the
path because the repository was installed with ``pip install -e .``. The GUI is
also launched straight out of a clone that was never pip-installed, so this
adds ``src/python`` to ``sys.path`` in that case and only in that case.

This is the one place in the GUI that touches ``sys.path``; import it first
from ``gui_core/__init__.py``. It is a bridge for the not-installed case, not a
mechanism to build on -- add a dependency to ``pyproject.toml`` instead.
"""

import sys
from pathlib import Path

_SRC_PYTHON = Path(__file__).resolve().parents[2] / "src" / "python"


def ensure_ptm_importable():
    """Return the path added to ``sys.path``, or ``None`` if nothing was needed."""
    try:
        import ptm  # noqa: F401
        return None
    except ImportError:
        pass
    if not (_SRC_PYTHON / "ptm" / "__init__.py").is_file():
        raise ImportError(
            "the PTM3D Python package was not found. Expected either an "
            f"installed 'ptm' (pip install -e . from the repository root) or "
            f"{_SRC_PYTHON / 'ptm'} in a checkout."
        )
    sys.path.insert(0, str(_SRC_PYTHON))
    return _SRC_PYTHON
