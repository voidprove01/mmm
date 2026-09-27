# Module 02 — Prior predictive checks

This document specifies the custom hierarchical Bayesian MMM used in this project. It differs from Meridian's default specification. All channels are anonymous and all source observations are simulated.

## Data and estimand

- Outcome: total conversions, transformed to conversions per capita and divided by its pooled training mean. Population is treated as a known exposure/size adjustment, not as proof of causal control.
- 130 initial weeks for training; final 26 weeks reserved. No holdout outcomes are used in figures, prior checks, or scaling.
- 12-week maximum lag; first 12 training weeks are media warmup only. This leaves 118 weeks per region for future fitting. Retain training media history when later predicting the holdout.
- Each channel's impressions per capita are divided by that channel's positive training median, pooled across regions. Regional comparisons are now on a per-capita exposure scale.
- Controls: competitor sales, sentiment, Promo, organic impressions per capita. Each is standardized using pooled training mean and SD. Organic media is a linear nuisance covariate in this simplified first model, not a paid ROI channel. This simplification needs later sensitivity analysis.
- Time basis: centered linear trend (one unit = 130 weeks) and annual sine/cosine with 52-week period. No additional intercept in the time basis.

## Likelihood and transformations

For scaled outcome y*: y*[g,t] ~ Normal(mu[g,t], sigma).

mu[g,t] = alpha[g] + B[t] delta + Z[g,t] gamma + sum_m beta[g,m] h(a[g,t,m]).

Adstock: a[g,t,m] = sum(l=0..12) lambda[m]^l x[g,t-l,m] / sum(l=0..12) lambda[m]^l.

Hill: h(a)=a/(a+k[m]); slope is fixed to 1. Unobserved prehistory is zero, but warmup rows are excluded from summaries and future likelihood. Because the lag is finite, excluded warmup prevents unknown prehistory from contaminating included rows.

## Initial priors (all Normal second arguments below are SDs)

| Parameter | Prior | Interpretation |
|---|---|---|
| alpha0 | Normal(0.6, 0.2) | Overall baseline on scaled per-capita outcome scale |
| tau_alpha | HalfNormal(0.2) | Between-region baseline SD |
| alpha[g] | Normal(alpha0, tau_alpha) | Regional baseline, partial pooling |
| b[m] | Normal(log(0.15), 0.5) | Population center of log channel response ceiling |
| tau[m] | HalfNormal(0.3) | Regional SD of log channel effects |
| log beta[g,m] | Normal(b[m], tau[m]) | Positive region-specific channel ceiling |
| lambda[m] | Beta(2,2) | Shared regional decay; mean 0.5 |
| k[m] | LogNormal(0,0.6) | Shared half-saturation; median 1 on scaled media axis |
| gamma[j] | Normal(0,0.1) | Per-SD control effects |
| delta trend | Normal(0,0.2) | Linear time effect |
| delta sine/cosine | Normal(0,0.1), independently | Seasonal effects |
| sigma | HalfNormal(0.15) | Common observation-noise SD |

Hyperpriors are mutually independent. Regional draws are conditionally independent given hyperparameters. Baseline and media regional deviations are independent in this baseline.

These are initial working priors, not externally elicited business truths. The outcome scaling uses training Y, so the check is **data-informed scale elicitation**, not a fully data-independent prior predictive assessment. No tuning to holdout is allowed. Avoid narrowing priors merely until the observed line is covered.

## Causal assumptions and limitations

Sentiment and other adjustment variables are treated as not caused by paid media for this baseline only. Survey timing is undocumented: this assumption is not established by the source. Including/excluding sentiment will be a sensitivity analysis, not a test proving exogeneity. Organic exposure might also mediate paid effects, and controlling it can change the estimand.

Media contributions are positive, additive, stable over time, and have no cross-channel interactions or cross-region spillovers. Gaussian errors are conditionally independent with common SD on the scaled per-capita scale. Normal likelihood can predict negative outcomes; do not clip them in diagnostic checks. Population scaling induces different noise scales in original counts.

Prior plots describe mathematical implications, not convergence, identification, causal validity, or achieved performance. Aggregated bands may hide problems at the region-week level, so we report the negative prediction fraction across all region-week draws as well.

## Run and inspect

From the project root: `python src/prior_predictive.py`.

Or open `notebooks/01_model_and_validation.ipynb`, which walks through preprocessing, generation, checks and interpretation. See `requirements.txt` for the versions used. No MCMC or GPU is required.

## Completed extra diagnostics

The region-week distribution pools 1,000 draws x 40 regions x 118 training weeks. Negative predictions: 1.197%; negative conditional means: 0.805%. Of negative predictions, 55.7% have negative means and 44.3% have nonnegative means. These are descriptive Monte Carlo fractions, not independent-binomial estimates.

Component charts show equal-cell-weighted prior distributions and distributions conditional on negative means. Conditioning changes their relationships and does not identify causal blame. Component sums were checked against mu and Y draw by draw. No priors were changed. Reducing sigma alone cannot remove negative conditional means.
