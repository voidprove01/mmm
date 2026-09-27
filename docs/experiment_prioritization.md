# Decision-focused experiment prioritization

## Decision and candidate set

This extension asks which experiment could improve a fixed-budget allocation decision **before observing its result**. It starts from the historical-data MCMC posterior, excluding the hypothetical Channel2 evidence used in the calibration example. The source observations remain simulated.

The decision replays the existing 118-week training-period spending template, totaling 160.88 million currency units, with channel multipliers between 0.8 and 1.2. Utility is expected in-window revenue; controls, conversion values, media prices and within-channel regional/weekly patterns remain fixed. This is a comparable historical scenario, not an estimate of an actual company's next-quarter cash flow. Value is horizon-specific and should not simply be rescaled to a different horizon.

Each candidate increases **one** channel's spending and impressions by 20% for eight weeks, with twelve followup weeks. Other channels remain unchanged. Input schedules use the same training-only template as the earlier calibration scenario. The estimand is the incremental revenue over those twenty weeks divided by extra intervention spend, aggregated over all 40 regions. This defines an estimand and an observation model, not a completed randomization or power plan.

| Dimension | Choices |
|---|---|
| Channel | Channel0, Channel1, Channel2, Channel3, Channel4 |
| Assumed precision | Low: return SE 0.60; medium: 0.30; high: 0.15 |
| Experiment summary likelihood | Normal(exact model intervention return, assumed SE) |
| Decision alternative | No new experiment; retain the current posterior-optimal allocation |

These are **15 screening scenarios**, not fifteen executed experiments. SE 0.15 retains the earlier calibration scenario's precision; 0.30 and 0.60 are twofold and fourfold larger standard errors. None is claimed feasible without further design work. A common return SE does not imply equal sample sizes, expense or revenue-lift precision across channels. The candidate table reports each scenario's implied lift SE and additional media spend.

Single-channel increases require extra test budget. If the firm cannot provide it, the candidate set must instead contain budget transfers and the corresponding joint intervention likelihoods. Decreases, multiple treatment intensities, joint-channel tests and alternative windows are outside this first candidate set. The best of these fifteen scenarios is not necessarily the globally best experiment.

## Value calculation

Let D be historical data, theta the MMM parameters, a a feasible allocation, U(a, theta) its expected revenue, and z a prospective experimental summary for design e. The current action is a0 = argmax_a E[U(a, theta) | D]. The expected value of sample information is:

EVSI(e) = E_z[max_a E[U(a, theta) | D, z, e] - E[U(a0, theta) | D, z, e]].

The outer expectation uses the current posterior predictive distribution of z. Including the same current action in every updated feasible set makes EVSI nonnegative. This averages over all prospective experimental results; it does not extrapolate from the one hypothetical result in the calibration notebook.

The implementation uses all 4,000 original posterior draws as a finite approximation. For each prospective summary, Normal likelihood weights update **whole parameter draws**, preserving their joint dependence. The updated expected response curves are optimized under the unchanged constraints. One-dimensional numerical quadrature averages conditional decision improvements over the predictive summary distribution. This is an efficient alternative to repeatedly generating individual-level experimental datasets or fitting thousands of MCMC models.

The expected value of perfect information (EVPI) is calculated as the average draw-specific optimal revenue minus the optimal average revenue. It supplies an upper bound on EVSI under this same finite-posterior decision model. Revenue components unchanged by allocation cancel from both differences.

## Numerical checks and interpretation

- Reproduce the existing uncalibrated optimal allocation and the original Channel2 intervention predictions before valuing new evidence.
- Compare 256- and 512-node predictive quadrature, verify predictive probability integrates to one, and verify fixed-budget feasibility.
- Check that EVSI lies between zero and EVPI, and that it increases as measurement noise decreases, within numerical tolerance.
- Report likelihood-weight concentration (weight ESS), including the predictive probability of ESS below 100. This is not autocorrelation-adjusted MCMC ESS. Rare posterior tails with few effective draws may be unreliable.
- Recompute using the first two and last two MCMC chains separately. Differences are sensitivity diagnostics, not confidence intervals; a close ranking should not be treated as definitive.

All estimates condition on the historical posterior, response specification, independent unbiased experimental summaries, assumed precision and stable transport to the budget scenario. Importance reweighting cannot discover parameter regions missing from the saved draws. The computation checks integration accuracy, not real-world validity. Stronger evidence or rankings sensitive to posterior support would require additional posterior sampling or explicit refits.

EVSI here is **gross expected revenue decision value**, before experiment costs. Extra experimental media spend is reported separately; it is not automatically the net economic cost because the test itself may generate revenue and has operational and opportunity costs. Actual cash-value ranking requires margins, experimental cash flows, execution costs and delay costs on a consistent horizon. This analysis does not claim a positive net ROI or a demonstrated field precision.

## Reproduction

The full reproduction pipeline includes this extension. To rerun it separately after the original posterior and source data exist in a run directory, execute:

```bash
python src/prioritize_experiments.py --model-root runs/my-reproduction --cache runs/experiment_prioritization/cache.npz
```

The extension writes small public reports; intermediate response curves remain in the ignored cache. Delete or change the cache path after changing the underlying data, posterior or response computation. The saved report records data and posterior hashes. Notebook 03 reads the resulting snapshots alongside the calibration results.

## Research connection

The question is motivated by [Abadie et al., Estimating the Value of Evidence-Based Decision Making](https://arxiv.org/abs/2306.13681): the economic value of evidence depends on how it informs decisions. This extension implements model-conditional Bayesian EVSI for a constrained MMM allocation problem. It is not a replication of the paper's empirical Bayes estimation procedure.
