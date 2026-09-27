# First temporal holdout evaluation — protocol fixed before scoring

## Frozen model

Use `reports/mcmc/posterior_samples.npz` from the final four-chain run. Do not refit parameters, change priors, select channels, or revise preprocessing after seeing these results. Record SHA-256 fingerprints of the model, preprocessing code, source data and posterior file in the evaluation manifest before computing scores.

## Target and information set

Evaluate the last 26 weeks, 2023-07-24 through 2024-01-15, for the same 40 observed regions (1,040 region-week outcomes). This is **conditional prediction**, using actual holdout impressions, control variables and population. It is not an advance forecast with unknown future inputs, a new-region test, or a causal intervention test. Contemporaneous controls may not have been available at the start of each forecast week.

Training-only outcome and exposure scales are unchanged. Carry the full training exposure history into normalized 12-week adstock. Future exposure must never affect predictions for earlier weeks. The existing time basis extrapolates using its original training scale. No holdout outcomes enter posterior prediction or baseline construction.

## Preselected comparisons and metrics

- MMM point prediction: mean of the conditional means from all 4,000 posterior samples.
- Regional mean baseline: each region's mean training scaled per-capita KPI over the same 118 effective fitting weeks, multiplied by holdout population and the training outcome scale.
- Seasonal naive baseline: the scaled per-capita KPI from 52 weeks earlier in that region, multiplied by holdout population and the training outcome scale. All reference outcomes fall in training.
- Primary metric: region-week RMSE on scaled per-capita outcome (equal cell weighting).
- Secondary metrics: cell MAE, original-unit RMSE, pooled WAPE (sum absolute errors / sum observed outcomes), signed bias, and weekly aggregate RMSE/WAPE. WAPE is not the average of regional percentage errors.
- MMM uncertainty: central 50%, 80%, 90% and 95% posterior predictive intervals using all 4,000 draws with observation noise. Report empirical coverage and width. Report the 90% interval score, which penalizes both excessive width and misses. Baselines provide point predictions only; no invented baseline probability distributions.
- Separately report 90% coverage and width for aggregate weekly outcomes, computed by summing within each joint predictive draw before taking quantiles.
- Show errors by region and over all 26 weeks. Do not select only favorable regions. Report extrapolation checks for media/control inputs, negative predictions and expected means.

## Interpretation

Coverage fractions are descriptive; region-week outcomes are dependent. Do not attach naive binomial confidence intervals or claim statistical significance from cell counts. Predictive uncertainty conditions on the provided inputs and does not include future-input uncertainty, model-selection uncertainty, or causal-assumption uncertainty. In-sample and holdout results must remain distinct. Good conditional prediction does not validate channel attribution.

This is the first evaluation of this holdout. Any model revision motivated by these results makes this period a validation set; it must not subsequently be described as an untouched final test.
