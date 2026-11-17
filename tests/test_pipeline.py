"""Regression checks for the mixed-experiment learner and counterfactual query.

Every check is written from an independent construction (Monte Carlo, direct
Gaussian algebra, finite differences) rather than by reusing the production
code paths it is testing.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from starter._structural import objective_gradient, row_log_density
from starter._validation import InputError, prepare_rows
from starter.estimator import fit
from starter.inference import condition, counterfactual

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gaussian_logpdf(mean, cov, y):
    d = len(y)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    delta = np.asarray(y) - np.asarray(mean)
    return -0.5 * (delta @ np.linalg.solve(cov, delta) + logdet + d * math.log(2 * math.pi))


def _independent_row_density(d, parents, B, bias, variance, row):
    """Independent construction: build the truncated model and marginalise."""
    do = row["do"]
    keep = [j for j in range(d) if j not in do]
    if not keep:
        return 0.0
    pos = {j: i for i, j in enumerate(keep)}
    Bk = np.zeros((len(keep), len(keep)))
    base = np.zeros(len(keep))
    for a, j in enumerate(keep):
        base[a] = bias[j]
        for k in parents[j]:
            if k in do:
                base[a] += B[j, k] * do[k]
            else:
                Bk[a, pos[k]] = B[j, k]
    G = np.linalg.solve(np.eye(len(keep)) - Bk, np.eye(len(keep)))
    mu = G @ base
    cov = G @ np.diag(variance[keep]) @ G.T
    observed = [pos[j] for j in keep if row["values"][j] is not None]
    if not observed:
        return 0.0
    y = [row["values"][keep[i]] for i in observed]
    return _gaussian_logpdf(mu[observed], cov[np.ix_(observed, observed)], y)


def _posterior_noise(B, bias, variance, evidence):
    """Independent abduction through the joint Gaussian of (X, eps)."""
    d = len(bias)
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))
    V = np.diag(variance)
    if not evidence:
        return np.zeros(d), V
    idx = sorted(evidence)
    C = G[idx, :]
    S = C @ V @ C.T
    mu = V @ C.T @ np.linalg.solve(S, np.array([evidence[j] for j in idx]) - C @ bias)
    Sigma = V - V @ C.T @ np.linalg.solve(S, C @ V)
    return mu, 0.5 * (Sigma + Sigma.T)


def _counterfactual_moments(B, bias, variance, evidence, do):
    """Independent counterfactual moments via the (X, eps) joint law."""
    d = len(bias)
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))
    mu_eps, Sigma_eps = _posterior_noise(B, bias, variance, evidence)

    Acf = A.copy()
    bcf = np.asarray(bias, dtype=float).copy()
    P = np.eye(d)
    for j in do:
        Acf[j, :] = 0.0
        Acf[j, j] = 1.0
        bcf[j] = do[j]
        P[j, j] = 0.0
    K = np.linalg.solve(Acf, P)
    mean_cf = np.linalg.solve(Acf, bcf + P @ mu_eps)
    cov_cf = K @ Sigma_eps @ K.T
    mean_f = G @ (bias + mu_eps)
    cov_f = G @ Sigma_eps @ G.T
    cross = G @ Sigma_eps @ K.T
    return mean_cf, cov_cf, cross, mean_f, cov_f, mu_eps, Sigma_eps


# ---------------------------------------------------------------------------
# Two-variable hand calculation: conditioning vs intervention
# ---------------------------------------------------------------------------
def test_two_variable_hand_conditioning_vs_do():
    # X0 = eps0, X1 = 2 X0 + eps1, eps ~ N(0, I).
    model = {"B": [[0, 0], [2, 0]], "bias": [0, 0], "variance": [1, 1]}
    # Observational P(X0 | X1 = 2): X0 ~ N(0,1), X1 = 2 X0 + eps1, so
    # E[X0|X1=2] = Cov(X0,X1)/Var(X1) * 2 = (2/5)*2 = 4/5 and
    # Var = 1 - 4/5 = 1/5.
    mean, cov = condition(model, {"1": 2})
    np.testing.assert_allclose(mean, [0.8, 2.0], atol=1e-12)
    np.testing.assert_allclose(cov, [[0.2, 0.0], [0.0, 0.0]], atol=1e-12)

    # Counterfactual for the same individual, do(X0 = 0), keeping eps.
    # X1 = 2 eps0 + eps1 with eps ~ N(0,I); conditioning on X1 = 2 gives
    # posterior mean eps = (2,1)*2/5 = (4/5, 2/5), so X1_cf = eps1 has mean
    # 2/5.  Posterior cov eps = I - C^T C / 5, so Var(eps1|ev) = 4/5 and
    # Cov(eps0, eps1|ev) = -2/5.
    out = counterfactual(model, {"1": 2}, {"0": 0})
    np.testing.assert_allclose(out["mean"], [0.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(out["covariance"], [[0, 0], [0, 0.8]], atol=1e-12)
    np.testing.assert_allclose(out["factual_mean"], [0.8, 2.0], atol=1e-12)
    # Cov(X0, X1_cf) = Cov(eps0, eps1|ev) = -2/5; Cov(X1, X1_cf) = 0 because
    # X1 is exactly known given the evidence.
    np.testing.assert_allclose(
        out["cross_covariance"], [[0, -0.4], [0, 0.0]], atol=1e-12
    )
    assert out["covariance"][0][0] == 0.0
    assert out["covariance"][0][1] == 0.0


def test_counterfactual_empty_maps_and_population_intervention():
    B = np.array([[0, 0, 0], [0.5, 0, 0], [-0.4, 0.8, 0]])
    bias = np.array([0.3, -0.6, 0.9])
    v = np.array([0.5, 1.2, 0.7])
    model = {"B": B.tolist(), "bias": bias.tolist(), "variance": v.tolist()}

    # Empty do: counterfactual is the factual random vector.
    same = counterfactual(model, {0: 1.1, 2: -0.4}, {})
    np.testing.assert_allclose(same["mean"], same["factual_mean"], atol=1e-12)
    np.testing.assert_allclose(same["covariance"], same["factual_covariance"], atol=1e-12)
    np.testing.assert_allclose(same["cross_covariance"], same["factual_covariance"], atol=1e-12)

    # Empty evidence: population intervention law from the truncated model.
    out = counterfactual(model, {}, {1: 0.7})
    Acf = np.eye(3) - B
    bcf = bias.copy()
    P = np.eye(3)
    Acf[1, :] = 0.0
    Acf[1, 1] = 1.0
    bcf[1] = 0.7
    P[1, 1] = 0.0
    Gcf = np.linalg.solve(Acf, np.eye(3))
    np.testing.assert_allclose(out["mean"], Gcf @ bcf, atol=1e-12)
    np.testing.assert_allclose(
        out["covariance"], Gcf @ P @ np.diag(v) @ P.T @ Gcf.T, atol=1e-12
    )
    assert out["covariance"][1][1] == 0.0
    assert out["covariance"][1][0] == 0.0


def test_same_variable_observed_and_intervened_differently():
    model = {"B": [[0, 0], [2, 0]], "bias": [0, 0], "variance": [1, 1]}
    out = counterfactual(model, {0: 1.0}, {0: 0.0})
    # X0 = eps0 pins eps0 = 1; eps1 is independent of eps0 and unobserved.
    # The alternative world has X0 = 0 and X1 = eps1, so X1_cf has mean 0
    # and variance 1, while factually X1 = 2 + eps1.
    np.testing.assert_allclose(out["factual_mean"], [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(out["mean"], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out["covariance"], [[0, 0], [0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(out["factual_covariance"], [[0, 0], [0, 1.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# Monte Carlo against the abduction/action/prediction construction
# ---------------------------------------------------------------------------
def test_counterfactual_matches_monte_carlo_reused_noise():
    rng = np.random.default_rng(3)
    d = 3
    B = np.array([[0, 0, 0], [0.5, 0, 0], [-0.4, 0.8, 0]])
    bias = np.array([0.3, -0.6, 0.9])
    v = np.array([0.5, 1.2, 0.7])
    evidence = {0: 1.1, 2: -0.4}
    do = {1: 0.7}
    out = counterfactual(
        {"B": B.tolist(), "bias": bias.tolist(), "variance": v.tolist()}, evidence, do
    )
    mean_cf, cov_cf, cross, mean_f, cov_f, mu, Sigma = _counterfactual_moments(
        B, bias, v, evidence, do
    )

    # Draw the posterior noise once and reuse it for both worlds.
    w, U = np.linalg.eigh(Sigma)
    R = U @ np.diag(np.sqrt(np.clip(w, 0, None)))
    N = 300_000
    eps = mu + rng.standard_normal((N, d)) @ R.T
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))
    X = (G @ (bias[:, None] + eps.T)).T
    Acf = A.copy()
    bcf = bias.copy()
    P = np.eye(d)
    Acf[1, :] = 0.0
    Acf[1, 1] = 1.0
    bcf[1] = 0.7
    P[1, 1] = 0.0
    Gcf = np.linalg.solve(Acf, np.eye(d))
    Xcf = (Gcf @ (bcf[:, None] + P @ eps.T)).T

    assert np.max(np.abs(Xcf.mean(0) - out["mean"])) < 5e-3
    assert np.max(np.abs(np.cov(Xcf.T) - out["covariance"])) < 8e-3
    assert np.max(np.abs(np.cov(X.T) - out["factual_covariance"])) < 8e-3
    cross_mc = (X - X.mean(0)).T @ (Xcf - Xcf.mean(0)) / N
    assert np.max(np.abs(cross_mc - out["cross_covariance"])) < 8e-3


def test_counterfactual_covariance_symmetric_psd_and_deterministic_dos():
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = int(rng.integers(2, 6))
        parents = [[k for k in range(j) if rng.random() < 0.5] for j in range(d)]
        B = np.zeros((d, d))
        for j, ps in enumerate(parents):
            for k in ps:
                B[j, k] = rng.uniform(-2.5, 2.5)
        model = {
            "B": B.tolist(),
            "bias": rng.uniform(-4, 4, d).tolist(),
            "variance": rng.uniform(0.05, 5, d).tolist(),
        }
        ev = {int(j): float(rng.uniform(-15, 15)) for j in rng.choice(d, size=d // 2, replace=False)}
        do = {int(j): float(rng.uniform(-15, 15)) for j in rng.choice(d, size=max(1, d // 3), replace=False)}
        out = counterfactual(model, ev, do)
        C = np.asarray(out["covariance"])
        assert np.max(np.abs(C - C.T)) < 1e-12
        assert np.min(np.linalg.eigvalsh(C)) > -1e-8
        for j in do:
            assert C[j, j] == 0.0
            assert np.all(C[j, :] == 0.0)
            assert np.all(C[:, j] == 0.0)
            assert np.all(np.asarray(out["cross_covariance"])[:, j] == 0.0)


# ---------------------------------------------------------------------------
# Mixed missing-data likelihood
# ---------------------------------------------------------------------------
def test_row_density_matches_independent_truncated_gaussian():
    d = 3
    parents = [[], [0], [0, 1]]
    B = np.array([[0, 0, 0], [0.4, 0, 0], [-0.3, 0.7, 0]])
    bias = np.array([0.2, -0.5, 0.8])
    v = np.array([0.6, 1.1, 0.4])
    rows = [
        {"values": [0.5, None, 1.2], "do": {}},
        {"values": [None, 0.7, None], "do": {}},
        {"values": [0.1, 0.2, 0.3], "do": {0: 0.1}},
        {"values": [None, None, None], "do": {}},
        {"values": [None, 1.0, 0.9], "do": {2: 0.9}},
    ]
    prepared = prepare_rows(rows, d)
    for row, prep in zip(rows, prepared):
        expected = _independent_row_density(d, parents, B, bias, v, row)
        got = row_log_density(prep, parents, B, bias, v)
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_full_observational_rows_reduce_to_plain_gaussian():
    d = 2
    parents = [[], [0]]
    B = np.array([[0, 0], [0.7, 0]])
    bias = np.array([0.3, -0.2])
    v = np.array([0.5, 1.3])
    row = {"values": [1.0, 2.0], "do": {}}
    prep = prepare_rows([row], d)
    A = np.eye(d) - B
    G = np.linalg.solve(A, np.eye(d))
    mu = G @ bias
    cov = G @ np.diag(v) @ G.T
    expected = multivariate_normal(mu, cov).logpdf(row["values"])
    np.testing.assert_allclose(
        row_log_density(prep[0], parents, B, bias, v), expected, atol=1e-12
    )


def test_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(11)
    for _ in range(10):
        d = int(rng.integers(2, 6))
        parents = [[k for k in range(j) if rng.random() < 0.6] for j in range(d)]
        rows = []
        for _ in range(int(rng.integers(1, 30))):
            vals = [
                None if rng.random() < 0.2 else float(rng.uniform(-4, 4))
                for _ in range(d)
            ]
            do = {
                j: vals[j]
                for j in range(d)
                if rng.random() < 0.25 and vals[j] is not None
            }
            rows.append({"values": vals, "do": do})
        prep = prepare_rows(rows, d)
        nf = sum(len(ps) for ps in parents)
        theta = np.concatenate(
            [
                rng.uniform(-1.4, 1.4, nf),
                rng.uniform(-1.9, 1.9, d),
                rng.uniform(0.1, 3.0, d),
            ]
        )
        ridge = float(rng.uniform(0.01, 1))
        _, grad = objective_gradient(theta, parents, prep, ridge)
        numeric = np.empty_like(theta)
        h = 1e-6
        for i in range(len(theta)):
            step = np.zeros_like(theta)
            step[i] = h
            numeric[i] = (
                objective_gradient(theta + step, parents, prep, ridge)[0]
                - objective_gradient(theta - step, parents, prep, ridge)[0]
            ) / (2 * h)
        np.testing.assert_allclose(grad, numeric, atol=2e-6, rtol=2e-5)


def test_parameter_ordering_groups_b_then_bias_then_variance():
    from starter._structural import pack_parameters

    parents = [[], [0], [0, 2], [2]]
    B = np.zeros((4, 4))
    for j, ps in enumerate(parents):
        for i, k in enumerate(ps):
            B[j, k] = 10 * j + i
    bias = np.arange(4) + 100.0
    var = np.arange(4) + 200.0
    theta = pack_parameters(parents, B, bias, var)
    # B free entries: row1 [B10=10], row2 [B20=20, B22=21], row3 [B32=30].
    expected = np.concatenate([[10], [20, 21], [30], bias, var])
    np.testing.assert_array_equal(theta, expected)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _fixture():
    return json.loads((ROOT / "fixtures" / "train.json").read_text())


def test_fit_converges_beats_initial_and_is_legal():
    fixture = _fixture()
    result = fit(
        fixture["rows"], fixture["parents"], fixture["initial"], fixture["ridge"]
    )
    assert result["converged"]
    assert result["gradient_residual"] <= 1e-5
    assert result["objective"] <= result["initial_objective"] + 1e-12
    model = result["model"]
    B = np.asarray(model["B"])
    assert np.max(np.abs(B)) <= 1.5
    assert np.all(np.abs(model["bias"]) <= 2.0)
    assert np.all((np.asarray(model["variance"]) >= 0.1) & (np.asarray(model["variance"]) <= 3.0))
    # Sparsity pattern is untouched.
    for j, ps in enumerate(fixture["parents"]):
        for k in range(len(fixture["parents"])):
            if k not in ps:
                assert B[j, k] == 0.0


def test_fit_residual_and_gradient_are_recomputable():
    fixture = _fixture()
    result = fit(fixture["rows"], fixture["parents"], fixture["initial"], fixture["ridge"])
    theta = np.asarray(result["theta"])
    grad = np.asarray(result["gradient"])
    parents = [tuple(ps) for ps in result["parents"]]
    prep = prepare_rows(fixture["rows"], len(parents))
    value, g = objective_gradient(theta, parents, prep, result["ridge"])
    np.testing.assert_allclose(value, result["objective"], atol=1e-12)
    np.testing.assert_allclose(g, grad, atol=1e-10)
    from starter._structural import lower_bounds, upper_bounds, projection_residual

    residual = projection_residual(theta, g, lower_bounds(parents), upper_bounds(parents))
    np.testing.assert_allclose(residual, result["gradient_residual"], atol=1e-12)


def test_fit_honours_iteration_cap_and_reports_not_converged():
    fixture = _fixture()
    result = fit(
        fixture["rows"], fixture["parents"], fixture["initial"], fixture["ridge"], max_iter=1
    )
    assert result["iterations"] <= 1
    assert result["converged"] is False
    assert result["objective"] <= result["initial_objective"] + 1e-12


def test_fit_from_boundary_initial_stays_convergent_and_legal():
    rng = np.random.default_rng(21)
    d = 4
    parents = [[], [0], [0, 1], [1, 2]]
    rows = []
    for _ in range(80):
        vals = [None if rng.random() < 0.2 else float(rng.uniform(-8, 8)) for _ in range(d)]
        do = {
            j: vals[j]
            for j in range(d)
            if rng.random() < 0.2 and vals[j] is not None and abs(vals[j]) <= 10
        }
        rows.append({"values": vals, "do": do})
    B = np.zeros((d, d))
    for j, ps in enumerate(parents):
        for k in ps:
            B[j, k] = rng.choice([-1.5, 1.5])
    initial = {
        "B": B.tolist(),
        "bias": rng.choice([-2.0, 2.0], d).tolist(),
        "variance": rng.choice([0.1, 3.0], d).tolist(),
    }
    result = fit(rows, parents, initial, 0.4)
    assert result["converged"], result["status"]
    assert result["gradient_residual"] <= 1e-5
    assert result["objective"] <= result["initial_objective"] + 1e-9
    assert np.max(np.abs(result["model"]["B"])) <= 1.5


def test_gradient_with_internal_intervention_and_deep_parents():
    rng = np.random.default_rng(4)
    parents = [[], [0], [0, 1], [1, 2], [0, 2, 3]]
    rows = [
        {"values": [0.5, 0.2, -0.3, 1.2, 0.4], "do": {1: 0.2}},
        {"values": [0.1, 0.3, 0.6, 0.7, None], "do": {2: 0.6}},
        {"values": [None, 0.2, 0.1, None, 0.9], "do": {}},
        {"values": [0.0, 0.0, 0.0, 0.0, 0.0], "do": {0: 0.0, 3: 0.0}},
    ]
    prep = prepare_rows(rows, 5)
    nf = sum(len(ps) for ps in parents)
    theta = np.concatenate(
        [rng.uniform(-1, 1, nf), rng.uniform(-1, 1, 5), rng.uniform(0.2, 2.5, 5)]
    )
    _, grad = objective_gradient(theta, parents, prep, 0.3)
    numeric = np.empty_like(theta)
    h = 1e-6
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = h
        numeric[i] = (
            objective_gradient(theta + step, parents, prep, 0.3)[0]
            - objective_gradient(theta - step, parents, prep, 0.3)[0]
        ) / (2 * h)
    np.testing.assert_allclose(grad, numeric, atol=2e-6, rtol=2e-5)


def test_fit_random_problems_converge_and_never_worsen():
    rng = np.random.default_rng(7)
    for _ in range(15):
        d = int(rng.integers(2, 6))
        parents = [[k for k in range(j) if rng.random() < 0.5] for j in range(d)]
        rows = []
        for _ in range(int(rng.integers(1, 100))):
            vals = [
                None if rng.random() < 0.15 else float(rng.uniform(-5, 5))
                for _ in range(d)
            ]
            do = {
                j: vals[j]
                for j in range(d)
                if rng.random() < 0.2 and vals[j] is not None
            }
            rows.append({"values": vals, "do": do})
        B = np.zeros((d, d))
        for j, ps in enumerate(parents):
            for k in ps:
                B[j, k] = rng.uniform(-1, 1)
        initial = {
            "B": B.tolist(),
            "bias": rng.uniform(-1, 1, d).tolist(),
            "variance": rng.uniform(0.3, 2, d).tolist(),
        }
        result = fit(rows, parents, initial, float(rng.uniform(0.01, 1)))
        assert result["objective"] <= result["initial_objective"] + 1e-9
        assert result["converged"], result["status"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_inputs_raise():
    good = {"B": [[0, 0], [0.4, 0]], "bias": [0.0, 0.0], "variance": [1.0, 1.0]}
    rows = [{"values": [1.0, 2.0], "do": {}}]

    with pytest.raises(InputError):
        fit(rows, [[0], [0]], good, 0.1)  # parent not below child
    with pytest.raises(InputError):
        fit(rows, [[], [0, 0]], good, 0.1)  # duplicate parent
    with pytest.raises(InputError):
        fit(rows, [[], [0]], {"B": [[0, 0], [2.0, 0]], "bias": [0, 0], "variance": [1, 1]}, 0.1)
    with pytest.raises(InputError):
        fit(rows, [[], [0]], {"B": [[0, 0], [0.4, 0]], "bias": [0, 0], "variance": [1, 1]}, 2.0)
    with pytest.raises(InputError):
        fit([{"values": [1.0, 2.0], "do": {}}], [[], [0]], good, 0.1, max_iter=0)
    with pytest.raises(InputError):
        fit([{"values": [1.0, float("nan")], "do": {}}], [[], [0]], good, 0.1)
    with pytest.raises(InputError):
        fit([{"values": [1.0], "do": {}}], [[], [0]], good, 0.1)
    with pytest.raises(InputError):
        fit([{"values": [1.0, 2.0], "do": {0: 2.0}}], [[], [0]], good, 0.1)  # contradiction
    with pytest.raises(InputError):
        fit([{"values": [1.0, None], "do": {1: 2.0}}], [[], [0]], good, 0.1)  # do value missing

    with pytest.raises(InputError):
        counterfactual(good, {"0": 1.0, 0: 2.0}, {})  # duplicate key
    with pytest.raises(InputError):
        counterfactual(good, {"2": 1.0}, {})
    with pytest.raises(InputError):
        counterfactual(good, {}, {"0": 21.0})
    with pytest.raises(InputError):
        counterfactual({"B": [[0, 0], [3.5, 0]], "bias": [0, 0], "variance": [1, 1]}, {}, {})
    with pytest.raises(InputError):
        counterfactual({"B": [[0, 1], [0, 0]], "bias": [0, 0], "variance": [1, 1]}, {}, {})


# ---------------------------------------------------------------------------
# Fixture query from the repository
# ---------------------------------------------------------------------------
def test_query_fixture_documentation_example():
    query = json.loads((ROOT / "fixtures" / "query.json").read_text())
    out = counterfactual(query["model"], query["evidence"], query["do"])
    # X1 = 2 X0 + eps1, evidence X1 = 2 gives posterior means eps0 = 4/5 and
    # eps1 = 2/5; do(X0 = 0) implies X1_cf = eps1 with mean 2/5 and variance
    # 4/5 (the observational answer would be mean 2 and variance 0.2).
    np.testing.assert_allclose(out["mean"], [0.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(out["covariance"], [[0, 0], [0, 0.8]], atol=1e-12)
