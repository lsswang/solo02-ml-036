"""End-to-end demonstration.

Runs four demonstrations that each exercise a different contract:

1. the legacy complete-case observational regression and ordinary Gaussian
   conditioning (the original demo, unchanged in spirit);
2. mixed-experiment learning with missing entries, randomised interventions
   and an empty-observation record, followed by a counterfactual query that
   consumes *this fit's* ``B``, ``bias`` and ``variance``;
3. the same individual evidence and do-request against different legal
   training data, showing that the learned model changes the counterfactual;
4. an independent external model queried directly, plus the two-variable
   hand calculation that separates ``P(X | evidence)`` from a counterfactual.

All parameters actually used are printed so every number is reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from starter.estimator import fit, fit_observational
from starter.inference import condition, counterfactual

ROOT = Path(__file__).parent


def _counterfactual_with_params(result, evidence, do):
    """Query a fitted model, echoing the exact parameters used."""
    out = counterfactual(result["model"], evidence, do)
    out["model_used"] = result["model"]
    out["objective"] = result["objective"]
    return out


def main():
    training = json.loads((ROOT / "fixtures" / "train.json").read_text())
    query = json.loads((ROOT / "fixtures" / "query.json").read_text())

    # 1. Legacy observational demo.
    legacy_model = fit_observational(training["rows"], training["parents"])
    legacy_mean, legacy_cov = condition(query["model"], query["evidence"])

    # 2. Mixed-experiment fit and an individual counterfactual on that fit.
    fit_result = fit(
        training["rows"],
        training["parents"],
        training["initial"],
        training["ridge"],
    )
    # Evidence only constrains the noise through the child variable; the
    # intervention sets the parent, so the counterfactual mean and variance
    # genuinely depend on the learned B, bias and variance.
    individual_evidence = {"1": 2.0}
    individual_do = {"0": 0.5}
    fitted_query = _counterfactual_with_params(
        fit_result, individual_evidence, individual_do
    )
    # Confirm the returned parameters reproduce the reported objective.
    from starter._structural import objective_gradient
    from starter._validation import prepare_rows

    theta = np.asarray(fit_result["theta"])
    parents = [tuple(ps) for ps in fit_result["parents"]]
    prepared = prepare_rows(training["rows"], len(parents))
    recomputed, _ = objective_gradient(theta, parents, prepared, training["ridge"])

    # 3. Same individual request, different legal training evidence.  The
    #    second collection raises the intercept of the first variable.
    shifted_rows = []
    for row in training["rows"]:
        values = list(row["values"])
        for j in range(len(values)):
            if values[j] is not None:
                values[j] = values[j] + (1.0 if j == 0 else 0.0)
        # Keys in the fixture may be integer strings; normalise before shifting.
        do = {}
        for key, value in row.get("do", {}).items():
            idx = int(key)
            do[idx] = value + (1.0 if idx == 0 else 0.0)
        shifted_rows.append({"values": values, "do": do})
    shifted = {
        "rows": shifted_rows,
        "parents": training["parents"],
        "initial": training["initial"],
        "ridge": training["ridge"],
    }
    shifted_fit = fit(
        shifted["rows"], shifted["parents"], shifted["initial"], shifted["ridge"]
    )
    shifted_query = _counterfactual_with_params(
        shifted_fit, individual_evidence, individual_do
    )

    #    A second legal change: drop the randomised-intervention rows and keep
    #    only complete observational rows.  This changes the learned variances,
    #    hence the counterfactual covariance.
    observational_only = [
        {"values": list(row["values"]), "do": {}}
        for row in training["rows"]
        if not row.get("do") and all(x is not None for x in row["values"])
    ]
    observational_fit = fit(
        observational_only,
        training["parents"],
        training["initial"],
        training["ridge"],
    )
    observational_query = _counterfactual_with_params(
        observational_fit, individual_evidence, individual_do
    )

    # 4a. Independent external-model query (no fitting).
    external = counterfactual(query["model"], query["evidence"], query["do"])
    external_no_do = counterfactual(query["model"], query["evidence"], {})

    # 4b. Two-variable hand check: X0 = eps0, X1 = 2 X0 + eps1, eps ~ N(0, I).
    hand_model = {"B": [[0, 0], [2, 0]], "bias": [0, 0], "variance": [1, 1]}
    hand_conditional = condition(hand_model, {"1": 2})
    hand_counterfactual = counterfactual(hand_model, {"1": 2}, {"0": 0})

    report = {
        "1_legacy_observational": {
            "complete_case_model": legacy_model,
            "observational_condition_mean": legacy_mean.tolist(),
            "observational_condition_covariance": legacy_cov.tolist(),
        },
        "2_mixed_fit_and_counterfactual": {
            "parents": fit_result["parents"],
            "objective": fit_result["objective"],
            "initial_objective": fit_result["initial_objective"],
            "converged": fit_result["converged"],
            "status": fit_result["status"],
            "gradient_residual": fit_result["gradient_residual"],
            "iterations": fit_result["iterations"],
            "model_used": fit_result["model"],
            "objective_recomputed_from_returned_parameters": recomputed,
            "evidence": individual_evidence,
            "do": individual_do,
            "counterfactual_mean": fitted_query["mean"],
            "counterfactual_covariance": fitted_query["covariance"],
            "factual_mean": fitted_query["factual_mean"],
            "cross_covariance": fitted_query["cross_covariance"],
        },
        "3_same_query_other_training_evidence": {
            "model_used": shifted_fit["model"],
            "objective": shifted_fit["objective"],
            "converged": shifted_fit["converged"],
            "evidence": individual_evidence,
            "do": individual_do,
            "counterfactual_mean": shifted_query["mean"],
            "counterfactual_covariance": shifted_query["covariance"],
            "mean_change_vs_fit_1": (
                np.asarray(shifted_query["mean"]) - np.asarray(fitted_query["mean"])
            ).tolist(),
        },
        "3b_same_query_observational_rows_only": {
            "model_used": observational_fit["model"],
            "objective": observational_fit["objective"],
            "converged": observational_fit["converged"],
            "evidence": individual_evidence,
            "do": individual_do,
            "counterfactual_mean": observational_query["mean"],
            "counterfactual_covariance": observational_query["covariance"],
            "mean_change_vs_fit_1": (
                np.asarray(observational_query["mean"])
                - np.asarray(fitted_query["mean"])
            ).tolist(),
            "covariance_change_vs_fit_1": (
                np.asarray(observational_query["covariance"])
                - np.asarray(fitted_query["covariance"])
            ).tolist(),
        },
        "4a_external_model_query": {
            "model_used": query["model"],
            "evidence": query["evidence"],
            "do": query["do"],
            "counterfactual_mean": external["mean"],
            "counterfactual_covariance": external["covariance"],
            "factual_mean": external["factual_mean"],
            "cross_covariance": external["cross_covariance"],
            "empty_do_counterfactual_mean": external_no_do["mean"],
            "empty_do_counterfactual_covariance": external_no_do["covariance"],
        },
        "4b_two_variable_hand_check": {
            "model": hand_model,
            "evidence": {"1": 2},
            "do": {"0": 0},
            "observational_mean": hand_conditional[0].tolist(),
            "observational_covariance": hand_conditional[1].tolist(),
            "counterfactual_mean": hand_counterfactual["mean"],
            "counterfactual_covariance": hand_counterfactual["covariance"],
            "cross_covariance": hand_counterfactual["cross_covariance"],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
