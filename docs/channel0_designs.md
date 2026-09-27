# Channel0: expanding intervention intensity

## Fixed decision and experiment windows

Compare Channel0 spending changes of **−20%, +10%, and +20%** against an unchanged baseline. Each intervention lasts eight weeks, followed by twelve observation weeks, using the same training-only input template and 40-region target population. Other channels are held fixed. The downstream decision, historical-data posterior and 118-week revenue objective remain those of the [experiment prioritization analysis](experiment_prioritization.md). The hypothetical calibration evidence is excluded.

The interventions are relative to the historical template, not sequential changes on top of the optimized allocation. Although +10% is near the current +11.3% recommended channel multiplier, their durations and schedules differ. The experiment does not directly validate the whole 118-week allocation.

## Comparable measurement uncertainty

Hold the **standard error of the total incremental revenue estimate** constant across interventions within each precision tier. Anchor the three tiers to the earlier Channel0 +20% scenario: revenue SE equals its additional spend multiplied by 0.60, 0.30 or 0.15. This preserves the original +20% results while making smaller interventions less precise in return units.

At the middle tier, all designs have revenue SE about **118,904 CU**. Additional spending is +396,346 CU for +20% and +198,173 CU for +10%; their implied return SEs are 0.30 and 0.60, respectively. The −20% intervention saves 396,346 CU. Its revenue effect is a signed loss; no negative standard error or misleading positive revenue lift is introduced.

The likelihood is Normal(signed treatment-minus-control revenue, common revenue SE). For numerical conditioning, both revenue and its SE are divided by the same positive reference spend for every design. Whole posterior draws are likelihood-reweighted, retaining their dependence. The observation noise remains an assumption: equal revenue SE does not prove equal field feasibility, randomization design, cost or power.

## What the values mean

EVSI measures the expected improvement in the **subsequent allocation decision**, averaged over possible experimental summaries. It excludes the experiment's own revenue gains or losses, spending changes, execution costs, profit margins and delay costs. The design table shows experiment spending and expected revenue changes separately; they are not added to EVSI or presented as net cash value. Lost revenue under the spending cut must be considered alongside saved spend before choosing a real test.

Under the fixed Hill slope-one response, reducing exposure can produce a larger absolute revenue contrast than increasing it by the same percentage. At equal revenue noise, this may make a cut more informative. This is a consequence of the assumed response curvature, not a general rule that spending cuts are always the best experiments. Structural misspecification, changing prices or operational restrictions could change the ranking.

## Checks and limitations

The +20% contrast and EVSI reproduce the earlier Channel0 scenario. Draw-level checks verify the signs and the expected concavity relationships. Numerical integration is compared at 256 and 512 nodes; estimates are also recomputed using separate two-chain groups. These are sensitivity diagnostics, not confidence intervals. Importance weights can be concentrated in rare outcomes, and the finite posterior approximation cannot introduce parameter regions absent from the saved draws. No achieved precision or experimental power is claimed.

Read the [results](../reports/channel0_designs/results.md) before choosing the next design. The next practical gate is whether the relative information values survive feasible uncertainty levels and the full experimental economic tradeoff. Region selection and randomization follow that gate.

## Reproduce

The full pipeline includes this comparison after the initial candidate screen. To rerun it independently:

```bash
python src/expand_channel0_designs.py --model-root runs/my-reproduction --cache runs/experiment_prioritization/cache.npz
```

The matching first-stage report must exist in this project's reports directory. Cache inputs are checked against source data and posterior hashes.
