"""Linear-Gaussian estimation.

The legacy :func:`fit_observational` (complete cases, observational rows only)
is kept so the original demo still runs.  The extension is :func:`fit`, a
bounded maximum-marginal-likelihood fit for mixed experiments: observational
rows, rows with fixed randomised interventions and rows with missing entries.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._structural import (
    clip_parameters,
    lower_bounds,
    objective_gradient,
    pack_parameters,
    projection_residual,
    unpack_parameters,
    upper_bounds,
)
from ._validation import (
    LEARNED_BOUNDS,
    InputError,
    learned_model_arrays,
    prepare_rows,
    validate_max_iter,
    validate_parents,
    validate_ridge,
)

CONVERGENCE_TOL = 1e-5


def fit_observational(rows, parents):
    """Existing complete, observational linear-Gaussian regression."""
    complete = [
        r["values"]
        for r in rows
        if not r.get("do") and all(x is not None for x in r["values"])
    ]
    if not complete:
        raise ValueError("no complete observational rows")
    x = np.asarray(complete, dtype=float)
    d = x.shape[1]
    B = np.zeros((d, d))
    bias = np.zeros(d)
    variance = np.zeros(d)
    for j, ps in enumerate(parents):
        design = np.column_stack([np.ones(len(x)), x[:, ps]])
        beta = np.linalg.lstsq(design, x[:, j], rcond=None)[0]
        bias[j] = beta[0]
        B[j, ps] = beta[1:]
        variance[j] = max(float(np.mean((x[:, j] - design @ beta) ** 2)), 1e-6)
    return {"B": B.tolist(), "bias": bias.tolist(), "variance": variance.tolist()}


def _model_dict(B, bias, variance):
    return {
        "B": np.asarray(B, dtype=float).tolist(),
        "bias": np.asarray(bias, dtype=float).tolist(),
        "variance": np.asarray(variance, dtype=float).tolist(),
    }


def _projected_lbfgs(theta0, lower, upper, fun, max_iter):
    """Minimise ``fun`` on a box, returning ``(x, scipy_success, result)``."""

    def wrapped(x):
        x = clip_parameters(x, lower, upper)
        value, grad = fun(x)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return 1e300, np.zeros_like(x)
        return value, grad

    cache = {}

    def tracked(x):
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key not in cache:
            cache[key] = wrapped(x)
        return cache[key]

    options = {"maxiter": int(max_iter), "ftol": 1e-15, "gtol": 0.0}
    try:
        result = minimize(
            tracked,
            np.array(theta0, dtype=float),
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower.tolist(), upper.tolist())),
            options=options,
        )
        return np.asarray(result.x, dtype=float), bool(result.success), result
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.array(theta0, dtype=float), False, None


def _newton_polish(theta, lower, upper, fun, iterations=40):
    """Damped Newton refinement at (or near) the identified optimum.

    The gradient is differentiated numerically to get a small Hessian (at most
    15 free parameters), rows/columns touching active bounds are frozen, and a
    ridge-regularised Newton step is backtracked.  Every accepted step strictly
    lowers the objective, so the best legal model is never lost.  This is what
    drives the projected-gradient residual below the ``1e-5`` threshold once
    L-BFGS has located the basin; it costs a few dozen extra gradient calls.
    """
    theta = np.array(theta, dtype=float)
    value, grad = fun(theta)
    if not (np.isfinite(value) and np.all(np.isfinite(grad))):
        return theta, value, 0
    tol = 1e-7
    used = 0
    for _ in range(iterations):
        # Identify frozen coordinates: on a bound with the gradient pointing
        # out of the box, the optimum stays on that bound.
        active = np.zeros(len(theta), dtype=bool)
        at_lower = theta <= lower + tol
        at_upper = theta >= upper - tol
        active |= at_lower & (grad > 0)
        active |= at_upper & (grad < 0)
        free = ~active
        if not np.any(free):
            break
        # Projected gradient residual in the free subspace.
        residual = np.abs(grad[free])
        if residual.size == 0 or np.max(residual) <= 1e-10:
            break

        # Numerical Hessian of the objective over the free coordinates.
        idx = np.flatnonzero(free)
        n = len(idx)
        H = np.empty((n, n))
        h = 1e-5
        for a, i in enumerate(idx):
            step = np.zeros_like(theta)
            step[i] = h
            gp = fun(clip_parameters(theta + step, lower, upper))[1]
            gm = fun(clip_parameters(theta - step, lower, upper))[1]
            H[a, :] = (gp[idx] - gm[idx]) / (2.0 * h)
        H = 0.5 * (H + H.T)
        # Levenberg damping keeps the step defined for indefinite curvature.
        scale = max(1.0, float(np.max(np.abs(np.diag(H))))) if n else 1.0
        damping = 1e-8 * scale
        try:
            direction = np.linalg.solve(
                H + damping * np.eye(n), -grad[idx]
            )
        except np.linalg.LinAlgError:
            direction = -grad[idx]
        if not np.all(np.isfinite(direction)):
            break
        slope = float(grad[idx] @ direction)
        if slope >= 0.0:
            # Not a descent direction; fall back to scaled steepest descent.
            direction = -grad[idx] / max(1.0, float(np.max(np.abs(grad[idx]))))
            slope = float(grad[idx] @ direction)
            if slope >= 0.0:
                break
        accepted = False
        alpha = 1.0
        current_residual = np.max(np.abs(grad[idx]))
        for _ in range(30):
            trial = theta.copy()
            trial[idx] = theta[idx] + alpha * direction
            trial = clip_parameters(trial, lower, upper)
            trial_value, trial_grad = fun(trial)
            if not np.isfinite(trial_value):
                alpha *= 0.5
                continue
            # Near the optimum the objective change can fall below double
            # precision while the gradient is not yet small.  Accept a step
            # that does not worsen the objective beyond floating-point noise
            # if it reduces the projected residual; otherwise require a real
            # descent step.
            tolerance = 1e-10 * max(1.0, abs(value))
            trial_residual = np.max(np.abs(trial_grad[idx]))
            if (
                trial_value <= value + tolerance
                and trial_residual < current_residual
            ) or trial_value < value - 1e-15:
                if trial_value <= value + tolerance:
                    theta, value, grad = trial, trial_value, trial_grad
                    accepted = True
                    used += 1
                    break
            alpha *= 0.5
        if not accepted:
            break
        # Stop once the projected residual is comfortably below tolerance.
        if np.max(np.abs(grad[free])) <= 1e-9:
            break
    return theta, value, used


def fit(rows, parents, initial, ridge=0.1, max_iter=500):
    """Fit the bounded mixed-experiment model from a legal initial guess.

    Parameters
    ----------
    rows, parents, ridge
        Training records, graph and penalty as described by the task.  All are
        validated strictly.
    initial
        ``{'B': ..., 'bias': ..., 'variance': ...}`` (``b``/``v`` aliases
        accepted) inside the learning bounds and matching the ``parents``
        sparsity pattern.
    max_iter
        Positive cap on optimiser iterations.

    Returns
    -------
    dict
        ``model`` is the actual returned model; ``objective`` is the
        penalised negative log-likelihood there (never above
        ``initial_objective``); ``theta``/``gradient`` are in the task's flat
        ordering, and ``gradient_residual`` is
        ``max|theta - clip(theta - gradient, lower, upper)|``.  ``converged``
        is true only when that residual is at most ``1e-5``.  ``status`` is
        one of ``converged``, ``max_iter``, ``failure`` or
        ``stationarity_not_reached``.
    """
    d, pcanon = validate_parents(parents)
    ridge = validate_ridge(ridge)
    max_iter = validate_max_iter(max_iter)
    prepared_rows = prepare_rows(rows, d)

    B0, bias0, variance0 = learned_model_arrays(
        initial, pcanon, LEARNED_BOUNDS, "initial"
    )
    theta0 = pack_parameters(pcanon, B0, bias0, variance0)
    lower = lower_bounds(pcanon)
    upper = upper_bounds(pcanon)
    if np.any(theta0 < lower) or np.any(theta0 > upper):
        raise InputError("initial model is outside the learning bounds")

    def fun(theta):
        return objective_gradient(theta, pcanon, prepared_rows, ridge)

    f0 = float(fun(theta0)[0])
    if not np.isfinite(f0):
        raise InputError("initial model gives a non-finite objective")

    best_theta = theta0.copy()
    best_f = f0
    iterations = 0
    status = "max_iter"

    theta_opt, success, result = _projected_lbfgs(theta0, lower, upper, fun, max_iter)
    if result is not None:
        iterations = int(getattr(result, "nit", 0))
        candidate = np.asarray(result.x, dtype=float)
        f_candidate, _ = fun(candidate)
        if np.isfinite(f_candidate) and f_candidate <= best_f:
            best_theta, best_f = candidate, f_candidate
    else:
        iterations = max_iter  # no usable optimiser result: treat as exhausted

    cap_hit = iterations >= max_iter
    if result is None:
        status = "failure"
    elif cap_hit:
        # Respect the caller's cap literally: no extra polishing, and the
        # result is reported as not converged even if the cap happened to
        # land on a stationary point.
        status = "max_iter"
    else:
        budget = max(0, max_iter - iterations)
        polished_theta, polished_f, extra = _newton_polish(
            best_theta, lower, upper, fun, iterations=min(40, budget)
        )
        iterations += extra
        if np.isfinite(polished_f) and polished_f <= best_f + 1e-12:
            best_theta, best_f = polished_theta, min(best_f, polished_f)

    value, grad = fun(best_theta)
    residual = projection_residual(best_theta, grad, lower, upper)
    converged = bool(
        not cap_hit and np.isfinite(value) and residual <= CONVERGENCE_TOL
    )
    if converged:
        status = "converged"
    elif not cap_hit and status != "failure":
        status = "stationarity_not_reached"

    B, bias, variance = unpack_parameters(best_theta, pcanon)
    return {
        "model": _model_dict(B, bias, variance),
        "parameters": {
            "B": B.tolist(),
            "bias": bias.tolist(),
            "variance": variance.tolist(),
        },
        "objective": float(value),
        "initial_objective": float(f0),
        "ridge": ridge,
        "parents": [list(ps) for ps in pcanon],
        "theta": best_theta.tolist(),
        "gradient": grad.tolist(),
        "gradient_residual": float(residual),
        "converged": converged,
        "status": status,
        "iterations": int(iterations),
        "max_iter": int(max_iter),
    }


fit_mixed = fit

__all__ = ["fit", "fit_mixed", "fit_observational", "CONVERGENCE_TOL"]
