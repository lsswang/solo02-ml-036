"""Bounded linear-Gaussian structural models and the mixed-experiment objective.

A model is ``X = B X + bias + eps`` with independent
``eps_j ~ N(0, variance_j)``.  ``B`` is strictly lower triangular in the
caller's topological order, so ``A = I - B`` is unit lower triangular and
always invertible.

The objective is the sum of the negative log *marginal* densities of each
record's observed, non-intervened variables, plus a ridge penalty on the free
``B`` and ``bias`` entries.  For a record with intervention set ``D`` the
structural equations of variables in ``D`` are replaced by their fixed values;
the remaining equations are retained.  Writing ``R`` for the non-intervened
variables,

    x_R = A_RR^{-1} (bias_R + B_RD d + eps_R),   A_RR = I - B_RR.

If the observed subset is empty the record contributes zero log density.

Parameters are flattened in the order required by the task: the free ``B``
entries grouped by child, then ``bias[0..d-1]``, then ``variance[0..d-1]``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._validation import (
    LEARNED_BOUNDS,
    InputError,
    prepare_rows,
    validate_parents,
)


def flat_blocks(parents):
    """Ordered ``(j, k)`` pairs for every free ``B`` entry."""
    return [(j, int(k)) for j in range(len(parents)) for k in parents[j]]


def n_free(parents):
    return sum(len(ps) for ps in parents)


def lower_bounds(parents):
    d = len(parents)
    return np.concatenate(
        [
            np.full(n_free(parents), LEARNED_BOUNDS["B"][0]),
            np.full(d, LEARNED_BOUNDS["bias"][0]),
            np.full(d, LEARNED_BOUNDS["variance"][0]),
        ]
    )


def upper_bounds(parents):
    d = len(parents)
    return np.concatenate(
        [
            np.full(n_free(parents), LEARNED_BOUNDS["B"][1]),
            np.full(d, LEARNED_BOUNDS["bias"][1]),
            np.full(d, LEARNED_BOUNDS["variance"][1]),
        ]
    )


def pack_parameters(parents, B, bias, variance):
    """Flatten a model into the task's parameter ordering."""
    parts = [np.asarray(B, dtype=float)[j, list(ps)] for j, ps in enumerate(parents)]
    parts.append(np.asarray(bias, dtype=float).reshape(-1))
    parts.append(np.asarray(variance, dtype=float).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0)


def unpack_parameters(theta, parents):
    """Rebuild ``(B, bias, variance)`` from the flat parameter vector."""
    d = len(parents)
    theta = np.asarray(theta, dtype=float)
    B = np.zeros((d, d), dtype=float)
    cursor = 0
    for j, ps in enumerate(parents):
        count = len(ps)
        if count:
            B[j, list(ps)] = theta[cursor : cursor + count]
        cursor += count
    bias = theta[cursor : cursor + d].copy()
    cursor += d
    variance = theta[cursor : cursor + 2 * d].copy()
    cursor += d
    if cursor != len(theta):
        raise InputError(
            f"parameter vector has length {len(theta)}, expected {cursor}"
        )
    return B, bias, variance


def clip_parameters(theta, lower, upper):
    return np.minimum(np.maximum(np.asarray(theta, dtype=float), lower), upper)


def projection_residual(theta, grad, lower, upper):
    """``max |theta - clip(theta - grad, lower, upper)|`` in parameter units.

    At an interior optimum this equals ``max |grad|``.  At an active bound it
    is zero when the gradient points out of the feasible box and positive when
    the gradient still points into the box, which is exactly the first-order
    optimality condition for a box constraint.
    """
    theta = np.asarray(theta, dtype=float)
    projected = clip_parameters(theta - np.asarray(grad, dtype=float), lower, upper)
    return float(np.max(np.abs(theta - projected)))


def _row_terms(row, parents, B, bias, variance):
    """Return likelihood pieces and gradient pieces for one record.

    Returns ``None`` when the record has no observed non-intervened variable
    (zero log density, zero gradient).  Otherwise returns
    ``(log_density, coeff_B, coeff_bias, coeff_variance)`` where the
    coefficient arrays are indexed by full variable indices.
    """
    d = len(parents)
    do_map = row["do"]
    do_set = set(do_map)
    R = [j for j in range(d) if j not in do_set]
    p = len(R)
    if p == 0 or len(row["obs_pos"]) == 0:
        return None

    rpos = {j: p_i for p_i, j in enumerate(R)}
    A = np.eye(p)
    c = np.asarray(bias, dtype=float)[R].copy()
    for r, j in enumerate(R):
        for k in parents[j]:
            if k in do_set:
                c[r] += B[j, k] * do_map[k]
            else:
                A[r, rpos[k]] -= B[j, k]

    # Unit lower triangular; solve is exact and cheap.
    L = np.linalg.solve(A, np.eye(p))
    mu = L @ c

    obs = row["obs_pos"]
    m = len(obs)
    LO = L[obs, :]
    vR = np.asarray(variance, dtype=float)[R]
    Sigma = (LO * vR) @ LO.T
    # Sigma is SPD: L is invertible and the rows selected by obs are
    # independent, so L_O diag(v_R) L_O^T has full rank.
    cf = cho_factor(Sigma, lower=True, check_finite=False)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    r = np.asarray(row["y_obs"], dtype=float) - mu[obs]
    U = cho_solve(cf, r, check_finite=False)
    log_density = (
        -0.5 * float(r @ U)
        - 0.5 * logdet
        - 0.5 * m * np.log(2.0 * np.pi)
    )

    W = cho_solve(cf, np.eye(m), check_finite=False)
    G = np.outer(U, U) - W  # W (r r^T - Sigma) W
    a = LO.T @ U  # p
    M = (G @ LO) * vR  # m x p
    T = LO.T @ M @ L.T  # p x p
    S = LO.T @ G @ LO  # p x p, only diagonal needed

    coeff_B = np.zeros((d, d))
    coeff_bias = np.zeros(d)
    coeff_variance = np.zeros(d)
    for r_i, j in enumerate(R):
        aj = float(a[r_i])
        for k in parents[j]:
            if k in do_set:
                coeff_B[j, k] += aj * do_map[k]
            else:
                q = rpos[k]
                coeff_B[j, k] += aj * mu[q] + T[r_i, q]
        coeff_bias[j] += aj
        coeff_variance[j] += 0.5 * S[r_i, r_i]
    return log_density, coeff_B, coeff_bias, coeff_variance


def _ensure_prepared(rows, parents):
    """Accept either already-prepared rows or raw task rows."""
    if isinstance(rows, (list, tuple)) and rows:
        first = rows[0]
        if isinstance(first, dict) and "obs_pos" in first:
            return rows
    d, _ = validate_parents(parents)
    return prepare_rows(rows, d)


def row_log_density(row, parents, B, bias, variance):
    """Log density of one row's observed, non-intervened variables."""
    if "obs_pos" not in row:
        d, _ = validate_parents(parents)
        row = prepare_rows([row], d)[0]
    terms = _row_terms(
        row, parents, np.asarray(B, float), np.asarray(bias, float),
        np.asarray(variance, float),
    )
    return 0.0 if terms is None else float(terms[0])


def objective_gradient(theta, parents, rows, ridge):
    """Penalised negative log-likelihood and its exact gradient.

    ``rows`` may be raw task rows or the pre-validated representation from
    :func:`starter._validation.prepare_rows`; ``theta`` is the flat parameter
    vector described in the module docstring.
    """
    rows = _ensure_prepared(rows, parents)
    theta = np.asarray(theta, dtype=float)
    B, bias, variance = unpack_parameters(theta, parents)
    pairs = flat_blocks(parents)

    total_ll = 0.0
    grad_B = np.zeros((len(parents), len(parents)))
    grad_bias = np.zeros(len(parents))
    grad_variance = np.zeros(len(parents))
    for row in rows:
        terms = _row_terms(row, parents, B, bias, variance)
        if terms is None:
            continue
        ll, cB, cb, cv = terms
        total_ll += ll
        grad_B += cB
        grad_bias += cb
        grad_variance += cv

    penalty = 0.0
    grad = np.zeros(len(theta))
    cursor = 0
    for j, k in pairs:
        value = float(B[j, k])
        penalty += value * value
        grad[cursor] = -grad_B[j, k] + 2.0 * ridge * value
        cursor += 1
    penalty += float(bias @ bias)
    for j in range(len(parents)):
        grad[cursor] = -grad_bias[j] + 2.0 * ridge * float(bias[j])
        cursor += 1
    for j in range(len(parents)):
        grad[cursor] = -grad_variance[j]
        cursor += 1

    objective = -total_ll + ridge * penalty
    return float(objective), grad


def objective(theta, parents, rows, ridge):
    """Penalised negative log-likelihood without the gradient."""
    return objective_gradient(theta, parents, rows, ridge)[0]


def log_likelihood(theta, parents, rows):
    """Unpenalised marginal log-likelihood of the rows."""
    rows = _ensure_prepared(rows, parents)
    B, bias, variance = unpack_parameters(theta, parents)
    total = 0.0
    for row in rows:
        total += row_log_density(row, parents, B, bias, variance)
    return float(total)


def prepared(rows, parents):
    """Validate ``parents`` and ``rows`` once for repeated objective calls."""
    d, pcanon = validate_parents(parents)
    return pcanon, prepare_rows(rows, d)


__all__ = [
    "flat_blocks",
    "n_free",
    "lower_bounds",
    "upper_bounds",
    "pack_parameters",
    "unpack_parameters",
    "clip_parameters",
    "projection_residual",
    "row_log_density",
    "objective_gradient",
    "objective",
    "log_likelihood",
    "prepared",
]
