# Causal model learning and individual queries

`python3 demo.py` exercises four contracts end to end:

1. the legacy complete-case observational regression and ordinary Gaussian
   conditioning (the original demo);
2. mixed-experiment learning (observational rows, fixed randomised
   interventions, missing entries) followed by an individual counterfactual
   that consumes **this fit's** `B`, `bias` and `variance`;
3. the *same* individual evidence and `do` request against a different legal
   training collection, showing the learned model and the counterfactual move;
4. an independent external model queried directly, plus a two-variable hand
   calculation that separates `P(X | evidence)` from `do(X)`.

Every model actually used is printed so all numbers are reproducible. Python
3.11, NumPy and SciPy are required. All data are synthetic. `train.json` holds
observations, fixed-value randomised interventions and missing entries;
intervened values appear in both `do` and `values` and must agree.

No graph structure discovery, real intervention, real patient data or online
service is implemented.

## Model and objective (`starter/estimator.py`, `starter/_structural.py`)

`X = B X + bias + eps`, with `eps_j ~ N(0, variance_j)` independent and `B`
strictly lower triangular in the supplied order (`A = I - B` is unit lower
triangular and always invertible). Free parameters are bounded:

* `B` free entries in `[-1.5, 1.5]`,
* `bias` in `[-2, 2]`,
* `variance` in `[0.1, 3]`;

non-parent entries are fixed at zero by the graph.

For each row the `do` variables' structural equations are replaced by their
constants and the remaining equations are kept. Writing `R` for the
non-intervened variables and `D` for the intervened ones,

```
x_R = A_RR^{-1} (bias_R + B_RD d + eps_R),    A_RR = I - B_RR,
```

so the row's contribution is the *marginal* Gaussian log density of the
observed subset of `x_R`. Unobserved non-intervened variables are integrated
out; they are never imputed and incomplete rows are never dropped. If no
non-intervened variable is observed the row contributes zero. The density of
an intervened variable under the original model is not multiplied in, and
observed coordinates are not treated as independent of each other.

The objective is

```
sum_rows  -log p(x_obs,row | do_row)  +  ridge * (sum_free B^2 + sum bias^2)
```

with all Gaussian normalisation constants and variance terms included. The
ridge penalty never touches `variance`.

Parameter ordering (also the order of `theta` and `gradient` in the fit
result): free `B` entries grouped by child with parents in increasing index
order, then `bias[0..d-1]`, then `variance[0..d-1]`. `gradient_residual` is
`max |theta - clip(theta - gradient, lower, upper)|`; the fit reports
`converged` only when it is at most `1e-5`.

### Likelihood and gradient

Per row, `A_RR` is solved once (it is at most 5x5 and triangular), giving the
conditional mean and covariance of the observed coordinates. The gradient is
computed analytically with reverse-mode-style adjoints of the closed-form
Gaussian log density:

```
r = y - mu,   U = Sigma^{-1} r,
d logp / d mu        = U,
d logp / d Sigma     = (U U^T - Sigma^{-1}) / 2,
```

and these are chained through `mu = L (bias + B d)`, `Sigma = L_O V L_O^T`
with `L = A_RR^{-1}`. Because `A_RR` is unit lower triangular, `dL/dB` is given
by `L dB L`, so the chain is a short sequence of small matrix products. The
implementation is checked against central finite differences in
`tests/test_pipeline.py` (tolerance `atol=2e-6`, `rtol=2e-5`).

## Fitting

`fit(rows, parents, initial, ridge, max_iter=500)` validates everything,
starts from the caller's legal `initial` and minimises the objective on the
box with L-BFGS-B (analytic gradient, every trial point clipped into the box),
followed by a damped Newton polish with a numerically differentiated Hessian on
the free coordinates (at most 20 free parameters for 5 variables, so this is
cheap) and bound-frozen coordinates. The polish only accepts steps that do not increase the objective,
so the best legal model seen is kept and the returned `objective` is never
above `initial_objective`. The problem is non-convex: no global optimality,
unique parameters or recovery of hidden truth is claimed. If the iteration cap
is hit, or the optimiser fails numerically, the best feasible model is
returned with `converged: False` and a `status` of `max_iter`, `failure` or
`stationarity_not_reached`.

The returned dictionary contains the actual `model` (`B`/`bias`/`variance`),
`objective`, `initial_objective`, `theta`, `gradient`, `gradient_residual`,
`converged`, `status`, `iterations` and `max_iter`.

## Individual counterfactual (`starter/inference.py`)

`counterfactual(model, evidence, do)` accepts any external legal model (bounds
`B` in `[-3, 3]`, `bias` in `[-5, 5]`, `variance` in `[0.05, 5]`, values in
`[-20, 20]`); it never refits or rebuilds the parameters. With
`G = A^{-1}` and `V = diag(variance)`:

1. **Abduction.** `X_O = y` are linear constraints on the noise:
   `eps | evidence ~ N(V C^T S^{-1}(y - C bias), V - V C^T S^{-1} C V)` with
   `C = G_O`, `S = C V C^T`. The posterior noise may be singular or
   correlated; no independent noise is resampled and no posterior precision
   is formed.
2. **Action.** For each `do` variable the structural equation is replaced by
   its constant; its noise is dropped. With `P` the projector onto
   non-intervened coordinates and `A_cf` the modified matrix:
   `X_cf = A_cf^{-1} (bias_cf + P eps)`.
3. **Prediction.** Reusing the *same* posterior noise,

   ```
   E[X_cf | evidence]   = A_cf^{-1} (bias_cf + P mu_eps)
   Cov(X_cf | evidence) = K Sigma_eps K^T,        K = A_cf^{-1} P
   Cov(X, X_cf | ev.)   = G Sigma_eps K^T
   ```

Do variables are deterministic, so their rows/columns of the covariance and
cross-covariance are exactly zero. Empty evidence gives the population
intervention law; empty `do` makes the counterfactual the factual random
vector. The same variable may appear in `evidence` and `do` with different
values: evidence describes the factual world and `do` the alternative one.

Results are returned in the original variable order as `mean`, `covariance`,
`factual_mean`, `factual_covariance` and `cross_covariance`. All covariances
are symmetrised; no inverse of a potentially singular covariance is taken
(The observed covariance `S` is used, which is positive definite because `G`
is invertible; the posterior noise matrix is only ever multiplied, never
inverted.)

## Verification performed

* **Two-variable hand check.** `X0 = eps0`, `X1 = 2 X0 + eps1`, `eps ~ N(0,I)`.
  Observational `P(X0 | X1 = 2)` has mean `4/5` and variance `1/5`.
  Counterfactual `do(X0 = 0)` for the same individual reuses the posterior
  noise: `E[X1_cf] = 2/5`, `Var = 4/5`, `Cov(X0, X1_cf) = -2/5`, while
  `Cov(X1, X1_cf) = 0`. These are exact and asserted in the tests and printed
  by the demo; they differ from the observational answer, which is the point.
* **Truncated density.** Each mixed row's log density is compared against an
  independently constructed truncated Gaussian (explicit `A_RR`, marginal
  Gaussian via `slogdet`/`solve`) including rows with empty observation sets,
  rows with `do`, and fully observed rows matched against
  `scipy.stats.multivariate_normal`.
* **Gradient.** Central finite differences for random graphs (2–5 variables,
  up to 100 rows, missing entries and interventions) agree to about `1e-8`,
  comfortably inside the stated tolerances.
* **Counterfactual moments.** 300k Monte-Carlo draws of the posterior noise,
  reused in both worlds, reproduce the analytic mean, covariance,
  factual covariance and cross-covariance to sampling error.
* **Fit.** Random mixed problems (2–5 variables, 1–100 rows, missing entries,
  interventions, boundary-valued initial models) converge with projected
  residual below `1e-5`, always with `objective <= initial_objective`; the
  fixture fits with residual around `1e-15`.

Run `python3 -m pytest tests -q` for the full suite (17 tests).

## Cost

Let `d <= 5`, `n <= 100` rows, `p <= d(d-1)/2 + 2d` free parameters. Each
objective/gradient evaluation costs `O(n d^3)` (one `d x d` solve and one
`m x m` Cholesky per row, `m <= d`). L-BFGS converges in a few dozen
iterations on the supplied problem sizes, so a fit is `O(n d^3 * iterations)`
— milliseconds to a fraction of a second here. A counterfactual query is a
single `O(d^3)` solve plus a few `d x d` products, independent of training
data size.

## Input errors

`starter/_validation.py` raises `InputError` (a `ValueError`) for: invalid
graphs (parent index not below the child, duplicates, wrong variable count),
parameters outside the applicable bounds, a non-lower-triangular `B`, a
non-parent nonzero entry, invalid/non-integer/out-of-range observation
indices, duplicate index spellings (`0` and `"0"`), contradictory training
`do` readings (or a missing value for an intervened variable), values outside
the stated ranges, non-finite numbers, non-positive or non-integer iteration
caps, and ridge outside `[0.01, 1]`. These are never reported as
non-convergence or as "no counterfactual".
