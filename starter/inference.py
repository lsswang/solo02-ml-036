"""Observational conditioning and individual-level counterfactual queries.

The legacy :func:`distribution` and :func:`condition` implement the ordinary
(non-causal) Gaussian conditioning used by the original demo.  The extension is
:func:`counterfactual`, which implements Pearl's abduction / action /
prediction contract for a *single individual*.

Derivation
----------
Write ``A = I - B`` (unit lower triangular) and ``G = A^{-1}``, so the factual
world is ``X = G (bias + eps)`` with ``eps ~ N(0, V)``, ``V = diag(variance)``.

**Abduction.**  Evidence ``X_O = y`` are linear constraints on the noise:
``G_O eps = y - G_O bias``.  Hence, with ``C = G_O`` and
``S = C V C^T = Cov(X_O)`` (positive definite because ``G`` is invertible and
``V`` is positive definite),

    eps | evidence ~ N( V C^T S^{-1} (y - C bias),
                        V - V C^T S^{-1} C V ).

The posterior noise may be singular and correlated; it is never resampled and
the inverse of the posterior covariance is never formed.

**Action.**  Each do variable's structural equation is replaced by a constant,
i.e. its noise is no longer used.  With ``P`` the projector onto non-do
coordinates and ``A_cf`` the modified unit-triangular matrix,

    X_cf = A_cf^{-1} (bias_cf + P eps).

**Prediction.**  Reusing the *same* posterior noise gives

    E[X_cf | evidence]  = A_cf^{-1} (bias_cf + P mu_eps),
    Cov(X_cf | evidence) = K Sigma_eps K^T,          K = A_cf^{-1} P,
    Cov(X, X_cf | ev.)   = G Sigma_eps K^T.

Do variables are deterministic in the counterfactual, so their variance and
cross terms are exactly zero.  Empty evidence reduces to the population
intervention law; empty ``do`` makes the counterfactual the factual random
vector.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._validation import (
    EXTERNAL_BOUNDS,
    QUERY_VALUE_RANGE,
    model_arrays,
    parse_index_map,
)


def distribution(model):
    """Population mean and covariance of an un-intervened linear-Gaussian SCM."""
    B, bias, variance = model_arrays(model, EXTERNAL_BOUNDS, "model")
    d = len(bias)
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))
    return G @ bias, (G * variance) @ G.T


def condition(model, evidence):
    """Ordinary observational conditioning ``P(X | X_evidence = y)``.

    Retained for the original demo; this is *not* a counterfactual query.
    """
    B, bias, variance = model_arrays(model, EXTERNAL_BOUNDS, "model")
    d = len(bias)
    mean, cov = distribution({"B": B, "bias": bias, "variance": variance})
    obs = parse_index_map(evidence, d, "evidence", *QUERY_VALUE_RANGE)
    if not obs:
        return mean, cov
    idx = np.array(sorted(obs), dtype=int)
    y = np.array([obs[j] for j in idx], dtype=float)
    cross = cov[:, idx]
    gain = np.linalg.solve(cov[np.ix_(idx, idx)], cross.T).T
    mean_c = mean + gain @ (y - mean[idx])
    cov_c = cov - gain @ cross.T
    return mean_c, 0.5 * (cov_c + cov_c.T)


def _counterfactual_from_arrays(B, bias, variance, evidence, do):
    d = len(bias)
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))

    # --- Abduction: posterior of eps given the individual's evidence --------
    obs_idx = np.array(sorted(evidence), dtype=int)
    V = np.asarray(variance, dtype=float)
    if len(obs_idx):
        C = G[obs_idx, :]  # m x d
        y = np.array([evidence[j] for j in obs_idx], dtype=float)
        offset = C @ bias
        S = (C * V) @ C.T
        S = 0.5 * (S + S.T)
        cf = cho_factor(S, lower=True, check_finite=False)
        solved = cho_solve(cf, (y - offset), check_finite=False)
        mu_eps = (V * (C.T @ solved))
        Sigma_eps = (V[:, None] * np.eye(d)) - (V[:, None] * (C.T @ cho_solve(cf, C, check_finite=False))) * V[None, :]
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)
    else:
        mu_eps = np.zeros(d)
        Sigma_eps = np.diag(V)

    # --- Action: replace do equations by constants -------------------------
    do_set = sorted(do)
    A_cf = A.copy()
    bias_cf = np.asarray(bias, dtype=float).copy()
    free_mask = np.ones(d)
    for j in do_set:
        A_cf[j, :] = 0.0
        A_cf[j, j] = 1.0
        bias_cf[j] = do[j]
        free_mask[j] = 0.0
    P = np.diag(free_mask)

    # --- Prediction: reuse the same posterior noise ------------------------
    K = np.linalg.solve(A_cf, P)
    mean_cf = np.linalg.solve(A_cf, bias_cf + P @ mu_eps)
    cov_cf = K @ Sigma_eps @ K.T
    cov_cf = 0.5 * (cov_cf + cov_cf.T)

    mean_fact = G @ (bias + mu_eps)
    cov_fact = G @ Sigma_eps @ G.T
    cov_fact = 0.5 * (cov_fact + cov_fact.T)
    cross = G @ Sigma_eps @ K.T

    # Numerical hygiene for deterministic coordinates: enforce exact zeros.
    if do_set:
        ids = np.array(do_set, dtype=int)
        cov_cf[ids, :] = 0.0
        cov_cf[:, ids] = 0.0
        cross[:, ids] = 0.0
    return mean_cf, cov_cf, cross, mean_fact, cov_fact


def counterfactual(model, evidence, do):
    """Individual counterfactual query on an external legal model.

    Parameters
    ----------
    model
        ``{'B', 'bias', 'variance'}`` (``b``/``v`` and optional ``parents``
        are accepted).  Model values must satisfy the external-query bounds.
    evidence
        ``{index: value}`` actual observations of the same individual in the
        un-intervened world.  May be empty.
    do
        ``{index: value}`` values fixed in the alternative world.  May be
        empty, in which case the counterfactual is the factual random vector.
        A variable may appear in both maps with different values: the evidence
        describes the factual world, ``do`` the alternative one.

    Returns
    -------
    dict with ``mean`` and ``covariance`` of ``X_cf | evidence`` in the
    original variable order, plus the factual ``factual_mean`` /
    ``factual_covariance`` and ``cross_covariance = Cov(X, X_cf | evidence)``.
    """
    B, bias, variance = model_arrays(model, EXTERNAL_BOUNDS, "model")
    d = len(bias)
    ev = parse_index_map(evidence, d, "evidence", *QUERY_VALUE_RANGE)
    do_map = parse_index_map(do, d, "do", *QUERY_VALUE_RANGE)
    mean_cf, cov_cf, cross, mean_fact, cov_fact = _counterfactual_from_arrays(
        B, bias, variance, ev, do_map
    )
    return {
        "mean": mean_cf.tolist(),
        "covariance": cov_cf.tolist(),
        "factual_mean": mean_fact.tolist(),
        "factual_covariance": cov_fact.tolist(),
        "cross_covariance": cross.tolist(),
    }


__all__ = ["distribution", "condition", "counterfactual"]
