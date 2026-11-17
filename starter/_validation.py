"""Input validation shared by the learner and by counterfactual queries.

The public entry points are deliberately strict: malformed graphs, parameters,
row data, index maps and non-finite numbers raise :class:`InputError` (a
``ValueError``) instead of being silently coerced or reported as a numerical
optimisation failure.
"""
from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping

import numpy as np


class InputError(ValueError):
    """Invalid problem input (as opposed to a failed numerical optimisation)."""


# Bounds from the task statement.
LEARNED_BOUNDS = {
    "B": (-1.5, 1.5),
    "bias": (-2.0, 2.0),
    "variance": (0.1, 3.0),
}
EXTERNAL_BOUNDS = {
    "B": (-3.0, 3.0),
    "bias": (-5.0, 5.0),
    "variance": (0.05, 5.0),
}
TRAINING_VALUE_RANGE = (-20.0, 20.0)
TRAINING_DO_RANGE = (-10.0, 10.0)
QUERY_VALUE_RANGE = (-20.0, 20.0)


def _as_list(value, name):
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise InputError(f"{name} must be a sequence, got {type(value).__name__}")
    try:
        return list(value)
    except TypeError as exc:  # pragma: no cover - message is the useful part
        raise InputError(f"{name} must be a sequence") from exc


def as_finite_float(value, name):
    """Return ``float(value)``, rejecting bools, non-numbers and non-finite."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InputError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{name} must be finite, got {value!r}")
    return result


def in_range(value, lo, hi, name):
    value = as_finite_float(value, name)
    if not (lo <= value <= hi):
        raise InputError(f"{name}={value!r} is outside the legal range [{lo}, {hi}]")
    return value


_INTEGER_KEY = re.compile(r"[+-]?\d+\Z")


def as_index(key, name):
    """Parse an observation index expressed as an int or an integer string."""
    if isinstance(key, (bool, np.bool_)):
        raise InputError(f"{name} index must be an integer, got {key!r}")
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, str):
        text = key.strip()
        if _INTEGER_KEY.fullmatch(text):
            return int(text)
    raise InputError(f"{name} index must be an integer or integer string, got {key!r}")


def validate_parents(parents):
    """Validate and canonicalise ``parents``.

    Returns ``(d, parents_tuple)`` where every parent list is sorted.  Parent
    lists must be distinct and contain only indices strictly below the child.
    """
    raw = _as_list(parents, "parents")
    d = len(raw)
    if not (2 <= d <= 5):
        raise InputError(f"the graph must have between 2 and 5 variables, got {d}")
    canonical = []
    for j, ps in enumerate(raw):
        values = _as_list(ps, f"parents[{j}]")
        seen = set()
        clean = []
        for k in values:
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
                raise InputError(f"parents[{j}] must contain integer indices, got {k!r}")
            k = int(k)
            if not (0 <= k < j):
                raise InputError(
                    f"parents[{j}] contains {k}, which is not a distinct index < {j}"
                )
            if k in seen:
                raise InputError(f"parents[{j}] contains duplicate index {k}")
            seen.add(k)
            clean.append(k)
        canonical.append(tuple(sorted(clean)))
    return d, tuple(canonical)


def parse_index_map(mapping, d, name, lo, hi):
    """Parse a ``{index: value}`` map, rejecting duplicate index spellings."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise InputError(f"{name} must be a mapping from index to number")
    result = {}
    for key, value in mapping.items():
        idx = as_index(key, name)
        if not (0 <= idx < d):
            raise InputError(f"{name} contains invalid variable index {idx} for d={d}")
        if idx in result:
            raise InputError(
                f"{name} contains duplicate index {idx} (e.g. both 0 and '0')"
            )
        result[idx] = in_range(value, lo, hi, f"{name}[{idx}]")
    return result


def parse_values(values, d, name, lo, hi):
    """Parse a length-``d`` row of ``None``/finite in-range values."""
    raw = _as_list(values, name)
    if len(raw) != d:
        raise InputError(f"{name} must have length {d}, got {len(raw)}")
    result = []
    for j, value in enumerate(raw):
        if value is None:
            result.append(None)
        else:
            result.append(in_range(value, lo, hi, f"{name}[{j}]"))
    return result


def validate_ridge(ridge):
    return in_range(ridge, 0.01, 1.0, "ridge")


def validate_max_iter(max_iter):
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise InputError("max_iter must be a positive integer")
    max_iter = int(max_iter)
    if max_iter <= 0:
        raise InputError("max_iter must be a positive integer")
    return max_iter


def _array(value, shape, name):
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric array") from exc
    if arr.shape != shape:
        raise InputError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains a non-finite number")
    return arr


def model_arrays(model, bounds, name="model"):
    """Validate a model dictionary and return ``(B, bias, variance)`` arrays.

    The model is expected to use the existing project key names
    (``B``/``bias``/``variance``); the shorter aliases ``b``/``v`` are also
    accepted.  A legal model is lower triangular in the supplied topological
    order, so every entry on or above the diagonal must be exactly zero.
    """
    if not isinstance(model, Mapping):
        raise InputError(f"{name} must be a mapping")
    if "B" not in model:
        raise InputError(f"{name} is missing the 'B' matrix")
    try:
        B = np.asarray(model["B"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name}['B'] must be a numeric matrix") from exc
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise InputError(f"{name}['B'] must be a square matrix, got {B.shape}")
    d = B.shape[0]
    if not (2 <= d <= 5):
        raise InputError(f"{name}['B'] must be between 2x2 and 5x5, got {d}x{d}")
    if not np.all(np.isfinite(B)):
        raise InputError(f"{name}['B'] contains a non-finite number")

    for j in range(d):
        if np.any(B[j, j:] != 0.0):
            raise InputError(
                f"{name}['B'] is not lower triangular in the supplied order"
            )

    bias_key = "bias" if "bias" in model else ("b" if "b" in model else None)
    variance_key = (
        "variance" if "variance" in model else ("v" if "v" in model else None)
    )
    if bias_key is None:
        raise InputError(f"{name} is missing the 'bias' vector")
    if variance_key is None:
        raise InputError(f"{name} is missing the 'variance' vector")

    bias = _array(model[bias_key], (d,), f"{name}['{bias_key}']")
    variance = _array(model[variance_key], (d,), f"{name}['{variance_key}']")

    blo, bhi = bounds["B"]
    if np.any((B < blo) | (B > bhi)):
        raise InputError(f"{name}['B'] entries must lie in [{blo}, {bhi}]")
    clo, chi = bounds["bias"]
    if np.any((bias < clo) | (bias > chi)):
        raise InputError(f"{name}['bias'] entries must lie in [{clo}, {chi}]")
    vlo, vhi = bounds["variance"]
    if np.any((variance < vlo) | (variance > vhi)):
        raise InputError(f"{name}['variance'] entries must lie in [{vlo}, {vhi}]")

    if "parents" in model:
        pd, pcanon = validate_parents(model["parents"])
        if pd != d:
            raise InputError(f"{name}['parents'] length does not match B")
        for j in range(d):
            allowed = set(pcanon[j])
            for k in range(d):
                if k not in allowed and B[j, k] != 0.0:
                    raise InputError(
                        f"{name}['B'][{j},{k}] is nonzero but {k} is not a parent"
                    )
    return B, bias, variance


def learned_model_arrays(model, parents, bounds=LEARNED_BOUNDS, name="model"):
    """Validate a model whose sparsity pattern is fixed by ``parents``."""
    d, pcanon = validate_parents(parents)
    B, bias, variance = model_arrays(model, bounds, name=name)
    if B.shape != (d, d):
        raise InputError(f"{name}['B'] has the wrong size for parents")
    for j in range(d):
        allowed = set(pcanon[j])
        for k in range(d):
            if k not in allowed and B[j, k] != 0.0:
                raise InputError(
                    f"{name}['B'][{j},{k}] must be zero because {k} is not a parent of {j}"
                )
    return B, bias, variance


def prepare_rows(rows, d):
    """Validate training rows and precompute the indexing used by the likelihood."""
    raw_rows = _as_list(rows, "rows")
    if not (1 <= len(raw_rows) <= 100):
        raise InputError(f"rows must contain between 1 and 100 records, got {len(raw_rows)}")
    prepared = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise InputError(f"rows[{i}] must be a mapping")
        if "values" not in row:
            raise InputError(f"rows[{i}] is missing 'values'")
        values = parse_values(
            row["values"], d, f"rows[{i}].values", *TRAINING_VALUE_RANGE
        )
        do = parse_index_map(
            row.get("do", {}), d, f"rows[{i}].do", *TRAINING_DO_RANGE
        )
        for j, fixed in do.items():
            if values[j] is None:
                raise InputError(
                    f"rows[{i}] has do={{{j}: {fixed}}} but values[{j}] is missing"
                )
            if float(values[j]) != fixed:
                raise InputError(
                    f"rows[{i}] contradicts its do: values[{j}]={values[j]!r} "
                    f"but do[{j}]={fixed!r}"
                )

        value_array = np.full(d, np.nan, dtype=float)
        for j, value in enumerate(values):
            if value is not None:
                value_array[j] = value

        do_idx = np.array(sorted(do), dtype=int)
        do_val = np.array([do[j] for j in do_idx], dtype=float)
        do_val_full = np.full(d, np.nan, dtype=float)
        do_val_full[do_idx] = do_val
        do_set = set(do)

        R = np.array([j for j in range(d) if j not in do_set], dtype=int)
        R_pos = np.full(d, -1, dtype=int)
        R_pos[R] = np.arange(len(R), dtype=int)
        observed_full = [j for j in R if values[j] is not None]
        obs_pos = np.array([R_pos[j] for j in observed_full], dtype=int)
        y_obs = np.array([values[j] for j in observed_full], dtype=float)

        prepared.append(
            {
                "values": value_array,
                "do": do,
                "do_idx": do_idx,
                "do_val": do_val,
                "do_val_full": do_val_full,
                "R": R,
                "R_pos": R_pos,
                "obs_pos": obs_pos,
                "obs_full": np.array(observed_full, dtype=int),
                "y_obs": y_obs,
                "has_density": bool(len(observed_full) > 0),
            }
        )
    return prepared
