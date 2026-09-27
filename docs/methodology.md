# Methodology and interpretation boundaries

## Hierarchical estimation

Regional intercepts partially pool around a shared mean. Log media coefficients partially pool around channel-specific means, with estimated between-region scales. Normalized geometric adstock uses lags 0–12. Saturation is `a / (a + K)` with slope fixed at one. Full specification and priors are in [model_specification.md](model_specification.md).

Training-only transformations are reused throughout. Prior scale elicitation is data-informed; it is not a fully data-independent prior predictive exercise. Noncentered parameterization improves sampling without changing the priors. Initial sampler attempts were refined by changing target acceptance and the mass matrix, not by choosing priors to remove inconvenient results.

## Validation

The first frozen temporal holdout compares against regional historical means and seasonal naive predictions. Its protocol is [here](../HOLDOUT_PROTOCOL.md). Coverage and region-level results are reported alongside aggregate error. Aggregate predictive accuracy does not establish intervention validity.

## Incremental effect definition

For a reporting-window channel effect, remove only that channel's in-window exposures, preserve earlier exposure history and hold covariates and other channels fixed. Sum the expected-outcome differences over the reporting window. Convert to revenue using each region-week's conversion value. Effects occurring after the window are excluded.

ROAS divides this revenue effect by in-window channel spend. Marginal ROAS differentiates the response when spend and exposure scale proportionally at fixed cell-level impression prices. Revenue shares use observed revenue as the denominator; channel shares need not total 100%.

## Optimization

The objective is posterior expected revenue under a fixed total budget and ±20% channel bounds. Each channel keeps historical regional/weekly flighting. A common allocation is chosen and evaluated across draws, preserving parameter dependence. Exact gradients and optimality conditions are checked after interpolation-based optimization. This is a historical scenario, not a completed future deployment plan.

## Calibration

The hypothetical experiment targets Channel2 +20% for eight weeks with twelve followup weeks. For each candidate theta, replay both full exposure schedules and compare expected revenue. Add a Normal experimental-summary likelihood. The posterior changes jointly; parameters are not overwritten with an experimental point estimate.

The experimental summary is assumed independent of historical outcomes. It is used once. It is not the earlier weak experiment's result. The historical channel table and experiment have different counterfactual/time scopes. A tighter posterior under this stipulated evidence is not validation of functional form or all causal assumptions.

For comparing decisions, both the old and new allocations are evaluated under the same updated posterior. Comparing predicted gains across different posterior beliefs does not isolate the value of changing the allocation.

## Limits

The data are synthetic. Paid contributions are constrained positive, stable and additive; interactions and spillovers are absent. Control validity is assumed, and timing of sentiment measurement is undocumented. Gaussian noise permits negative outcome draws. Uncertainty intervals omit model misspecification, price changes, future covariate uncertainty and unknown causal confounding. The optimization does not incorporate profit margins, inventory or implementation costs.
