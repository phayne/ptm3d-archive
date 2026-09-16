#!/usr/bin/env python3
"""Resolved-facet population helpers for threshold fractions and counts.

Paul's ruling, 2026-09-15, verbatim:

    "If a facet has no value, then it should leave the denominator and the
     numerator"

A facet whose value is not finite -- NaN, +inf or -inf -- is UNRESOLVED. It is
never counted as qualifying and never counted as not-qualifying: it leaves the
population entirely. Every fraction built with these helpers therefore reads
"of the facets we could actually evaluate, what fraction qualifies", not "of
all facets".

Two traps these helpers exist to close:

  * A bare comparison is NOT a finiteness test. ``NaN < 110.0`` is False, so a
    facet with no value is silently reported as "not a cold trap"; ``-inf <
    110.0`` is True, so a corrupt facet is silently reported AS a cold trap;
    and ``+inf > 400.0`` is True, so a corrupt facet joins a hot-face count.
    Use ``resolved_mask()``, never ``np.isnan()``.
  * A fraction over a shrinking denominator is not interpretable on its own.
    Report ``population_note()`` (or the ``n_resolved`` this module returns)
    next to every fraction that is printed, written to a file, or drawn on a
    figure.

Arithmetic note: these helpers do not cast. ``area_fraction`` sums the caller's
array in the caller's dtype and, when every facet is resolved, the boolean
index produces a contiguous copy of the same values in the same order -- so the
pairwise summation is bit-for-bit the one the unguarded expression performed.
That is what makes the guard provably inert on clean data.
"""
from __future__ import annotations

import sys

import numpy as np

__all__ = [
    "resolved_mask",
    "resolved_values",
    "warn_if_unresolved",
    "count_fraction",
    "area_fraction",
    "population_note",
]


def resolved_mask(*fields) -> np.ndarray:
    """Boolean mask, True where EVERY field has a value (is finite).

    Pass every field the fraction depends on -- including the area weights for
    an area-weighted fraction. A facet is resolved only if all of them are.
    """
    if not fields:
        raise ValueError("resolved_mask() needs at least one field")
    first = np.asarray(fields[0])
    mask = np.isfinite(first)
    for other in fields[1:]:
        arr = np.asarray(other)
        if arr.shape != first.shape:
            raise ValueError(
                f"resolved_mask(): shape mismatch {arr.shape} vs {first.shape}")
        mask = mask & np.isfinite(arr)
    return mask


def resolved_values(field, resolved) -> np.ndarray:
    """The field restricted to the facets that have a value.

    For the order statistics -- median, mean, min, max -- that sit next to
    these fractions. Leaving them over the full population lets one NaN facet
    turn a reported number into NaN and one +inf facet turn a maximum into
    Infinity, which `json.dump` then writes as a bare `NaN`/`Infinity` literal
    that is not valid JSON.
    """
    field = np.asarray(field)
    if field.shape != np.asarray(resolved).shape:
        raise ValueError(
            f"resolved_values(): shape mismatch {field.shape} vs "
            f"{np.asarray(resolved).shape}")
    out = field[resolved]
    if out.size == 0:
        raise ValueError(
            "resolved_values(): no facet has a value -- an order statistic "
            "over an empty population is undefined, not zero")
    return out


def warn_if_unresolved(label: str, resolved: np.ndarray, stream=None) -> int:
    """Warn loudly when any facet leaves the population; return how many did.

    Silence is not an option here: a fraction whose denominator quietly shrank
    would otherwise be indistinguishable from one computed over every facet.
    """
    stream = sys.stderr if stream is None else stream
    n_total = int(np.asarray(resolved).size)
    n_res = int(np.count_nonzero(resolved))
    n_unresolved = n_total - n_res
    if n_unresolved:
        print(f"  WARNING: {label}: {n_unresolved} of {n_total} facets have no "
              f"value (non-finite) and leave BOTH the numerator and the "
              f"denominator; fractions are over the {n_res} resolved facets.",
              file=stream)
    return n_unresolved


def _population_size(resolved: np.ndarray, label: str) -> int:
    n_res = int(np.count_nonzero(resolved))
    if n_res == 0:
        raise ValueError(
            f"{label}: no facet has a value -- a fraction over an empty "
            "population is undefined, not zero")
    return n_res


def _as_mask(qualifying, resolved) -> np.ndarray:
    """Coerce `qualifying` to a boolean mask and AND it with `resolved`.

    The cast is not cosmetic: a caller passing an integer array would get a
    silent bitwise AND (wrong answer, no error) and one passing floats a
    TypeError, and this is the shared entry point for every call site.
    """
    qualifying = np.asarray(qualifying)
    if qualifying.dtype != bool:
        qualifying = qualifying.astype(bool)
    return qualifying & resolved


def count_fraction(qualifying, resolved, label: str = "count fraction") -> float:
    """COUNT fraction of qualifying facets among the RESOLVED facets."""
    n_res = _population_size(resolved, label)
    return float(np.count_nonzero(_as_mask(qualifying, resolved)) / n_res)


def area_fraction(qualifying, resolved, area, label: str = "area fraction") -> float:
    """AREA-weighted fraction of qualifying area among the RESOLVED area.

    The unresolved facets' area leaves the total as well as the numerator.
    """
    _population_size(resolved, label)
    area = np.asarray(area)
    denom = area[resolved].sum()
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"{label}: resolved area is {denom}, not a positive total")
    return float(area[_as_mask(qualifying, resolved)].sum() / denom)


def population_note(resolved: np.ndarray, area=None) -> str:
    """One-line disclosure of the population a fraction was computed over."""
    n_total = int(np.asarray(resolved).size)
    n_res = int(np.count_nonzero(resolved))
    note = f"N={n_res} resolved of {n_total}"
    if n_res != n_total:
        note += f" ({n_total - n_res} unresolved, excluded)"
    if area is not None:
        area = np.asarray(area)
        note += (f"; area {float(area[resolved].sum()):.6g} of "
                 f"{float(area.sum()):.6g}")
    return note
