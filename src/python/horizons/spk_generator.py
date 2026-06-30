from __future__ import annotations

"""
SPICE meta-kernel generator for HORIZONS-downloaded SPK files.

Creates .tm meta-kernel files that the PTM3D C++ code can load via CSPICE.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Standard generic kernels (shipped with CSPICE or widely available)
DEFAULT_GENERIC_KERNELS = [
    "lsk/naif0012.tls",       # Leap seconds
    "pck/pck00011.tpc",       # Planetary constants
    "spk/de440s.bsp",         # Planetary ephemeris (compact)
]

# SPICE truncates any single quoted kernel-pool string token (a path in
# KERNELS_TO_LOAD, or a directory in PATH_VALUES) once it grows long enough.
# The empirical hard limit for a single value string is ~99 characters; a
# truncated path then fails furnsh with NOSUCHFILE/FILEREADFAILED.  SPICE
# concatenates consecutive quoted strings when the earlier one ends with the
# continuation marker '+', so long paths must be split across continuation
# strings.  Keep each emitted string comfortably below the limit.
_SPICE_STR_CONTINUATION = "+"
_SPICE_MAX_STR_LEN = 78  # safe per-string budget, well under SPICE's ~99 cap


def _continued_string_lines(value: str, indent: str) -> str:
    """Emit a kernel-pool string value, splitting long paths with '+'.

    Returns one or more indented, single-quoted strings.  When ``value`` is
    too long for a single SPICE string token it is broken into chunks joined
    by the SPICE continuation marker ('+') so furnsh reassembles the full
    path.  This is robust to arbitrary path lengths (deep OneDrive paths,
    spaces, parentheses).
    """
    # Budget for path characters per chunk: the chunk plus a trailing '+'
    # continuation marker must stay under the per-string limit.
    chunk_len = _SPICE_MAX_STR_LEN - len(_SPICE_STR_CONTINUATION)
    if len(value) <= _SPICE_MAX_STR_LEN:
        return f"{indent}'{value}'"

    chunks = [value[i : i + chunk_len] for i in range(0, len(value), chunk_len)]
    lines = []
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1:
            lines.append(f"{indent}'{chunk}{_SPICE_STR_CONTINUATION}',")
        else:
            lines.append(f"{indent}'{chunk}'")
    return "\n".join(lines)


class SPKGenerator:
    """Generate SPICE meta-kernels for HORIZONS-downloaded SPK files."""

    def __init__(self, kernels_base_dir: str = ""):
        """
        Parameters
        ----------
        kernels_base_dir : str
            Base directory where standard SPICE kernels are stored.
            If empty, uses a relative path placeholder.
        """
        self.kernels_base_dir = kernels_base_dir

    def create_metakernel(
        self,
        output_path: str,
        body_spk_path: str,
        body_name: str = "",
        additional_kernels: Optional[List[str]] = None,
    ) -> str:
        """
        Write a SPICE meta-kernel (.tm) file.

        Parameters
        ----------
        output_path : str
            Path to write the .tm file.
        body_spk_path : str
            Path to the body-specific SPK file (from HORIZONS).
        body_name : str
            Body name for the comment header.
        additional_kernels : list[str], optional
            Additional kernel paths to include.

        Returns
        -------
        str
            Path to the written meta-kernel.
        """
        kernels = list(DEFAULT_GENERIC_KERNELS)

        if additional_kernels:
            kernels.extend(additional_kernels)

        # Determine the KERNELS PATH_VALUE from the kernels base dir.
        if self.kernels_base_dir:
            kernels_dir = os.path.abspath(self.kernels_base_dir)
        else:
            # Use the directory containing the meta-kernel as the base.
            kernels_dir = os.path.dirname(os.path.abspath(output_path))

        # The body SPK is always last (highest priority).  It lives at an
        # absolute path whose length frequently exceeds the per-string limit
        # SPICE imposes on a single KERNELS_TO_LOAD token, which would
        # truncate the path and make furnsh fail with NOSUCHFILE.  Reference
        # it through a dedicated BODY PATH_SYMBOL so the token written into
        # KERNELS_TO_LOAD stays short ('$BODY/<basename>').  The directory
        # itself (the BODY path value) may still be long, so it is emitted
        # with '+' continuation below.
        body_spk_abs = os.path.abspath(body_spk_path)
        body_dir = os.path.dirname(body_spk_abs)
        body_spk_name = os.path.basename(body_spk_abs)

        # Build the generic-kernel list string ($KERNELS-relative entries).
        kernel_lines = []
        for k in kernels:
            if os.path.isabs(k):
                kernel_lines.append(_continued_string_lines(k, "            "))
            else:
                kernel_lines.append(
                    _continued_string_lines(f"$KERNELS/{k}", "            ")
                )

        # Body SPK referenced via the BODY symbol so the token stays short.
        kernel_lines.append(
            _continued_string_lines(f"$BODY/{body_spk_name}", "            ")
        )

        kernels_str = "\n".join(kernel_lines)

        # PATH_VALUES entries are split with '+' continuation when long so the
        # deep OneDrive directory paths are not truncated by SPICE.
        kernels_dir_str = _continued_string_lines(kernels_dir, "                   ")
        body_dir_str = _continued_string_lines(body_dir, "                   ")

        content = f"""\
KPL/MK

\\begintext

  Meta-kernel for PTM3D generic body simulation.
  Body: {body_name or 'Unknown'}
  Auto-generated by PTM3D HORIZONS integration.

  Standard kernels: leapseconds, planetary constants, planetary ephemeris.
  Body SPK: {os.path.basename(body_spk_path)}

\\begindata

  PATH_VALUES  = (
{kernels_dir_str},
{body_dir_str}
  )
  PATH_SYMBOLS = ( 'KERNELS',
                   'BODY' )

  KERNELS_TO_LOAD = (
{kernels_str}
  )

\\begintext
"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

        logger.info("Wrote meta-kernel: %s", output_path)
        return output_path

    def create_minimal_metakernel(
        self,
        output_path: str,
        body_name: str = "",
    ) -> str:
        """
        Write a minimal meta-kernel with only leapseconds (for Keplerian mode).

        This is sufficient for str2et_c() time conversion without needing
        full ephemeris data.
        """
        if self.kernels_base_dir:
            path_values = os.path.abspath(self.kernels_base_dir)
        else:
            path_values = os.path.dirname(os.path.abspath(output_path))

        content = f"""\
KPL/MK

\\begintext

  Minimal meta-kernel for PTM3D Keplerian-mode simulation.
  Body: {body_name or 'Unknown'}
  Contains only leapseconds kernel for time conversion.

\\begindata

  PATH_VALUES  = ( '{path_values}' )
  PATH_SYMBOLS = ( 'KERNELS' )

  KERNELS_TO_LOAD = (
            '$KERNELS/lsk/naif0012.tls'
  )

\\begintext
"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

        logger.info("Wrote minimal meta-kernel: %s", output_path)
        return output_path
