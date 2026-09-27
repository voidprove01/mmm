# Decision memo: budget planning and the next channel experiment

**Audience:** Marketing, Finance and Analytics leadership  
**Decision requested:** Prioritize Channel0 experiment feasibility work; retain −20% and +20% interventions for comparison. Treat the modeled allocation as a planning candidate, pending business-specific validation.  
**Context:** Simulated-data portfolio case. Experimental precision is assumed; the calibration example uses hypothetical evidence. No experiment execution or realized revenue improvement is claimed.

## 1. What should we do, and why?

Prioritize **Channel0** for the next stage of experimental design. Across the five channels and three assumed precision levels examined, its information has the greatest expected value for the fixed-budget allocation decision. Channel2 ranks last despite having the most uncertain intervention return: across most prospective results, its allocation remains at the +20% spending cap. Measurement should target uncertainty that can change a consequential decision.

For Channel0, retain **−20% and +20% spending interventions**, each lasting eight weeks with twelve followup weeks. The −20% design has a modest information-value advantage under equal revenue-measurement noise, but its expected revenue loss during the test means it is not automatically the preferred business choice. The +10% alternative requires less additional spend but provides less information at the same absolute measurement noise.

Continue to use the allocation as a planning scenario. At a fixed total budget of **160.88m CU**, the hypothetical-calibration example recommends Channel0 +10.2%, Channel1 −20.0%, Channel2 +20.0%, Channel3 +6.2%, and Channel4 −11.6% relative to historical spending. See the [allocation table](../reports/calibrated_budget/allocation.md). This maximizes modeled expected revenue within the stated constraints, not profit.

**Keep the evidence states separate:** experiment prioritization starts from the historical-data posterior, before hypothetical calibration. The calibrated allocation illustrates how a stipulated observation could update a decision; it is not evidence already available to a company and is not used to rank prospective experiments.

## 2. What is the expected value, and how uncertain is it?

At assumed experimental return SE **0.30**, gross expected information value is approximately **1.088m CU for Channel0**, versus **0.024m CU for Channel2**, over the 118-week historical budget scenario. These values average over possible experiment results and compare choosing an allocation after observing evidence with retaining the current optimal allocation. See the [channel comparison](../reports/experiment_prioritization/results.md).

Within Channel0, hold the revenue-summary SE at **118,904 CU** across interventions:

| Intervention | Test spending change | Expected test revenue change | Expected downstream information value |
|---|---:|---:|---:|
| −20% | −396,346 CU | −516,863 CU | 1,135,956 CU |
| +10% | +198,173 CU | +235,316 CU | 684,830 CU |
| +20% | +396,346 CU | +457,369 CU | 1,088,305 CU |

Negative spending is a saving; negative revenue is a loss. Test effects cover eight intervention weeks and twelve followup weeks. Information value concerns the separate 118-week allocation scenario and **excludes the test's own economics**, implementation and delay costs. It is not net cash value. Equal assumed revenue SE also does not establish equal experimental cost or feasibility. See the [design comparison](../reports/channel0_designs/results.md).

The cut's medium-precision information advantage over +20% is about **47,651 CU (4.4%)**. Both posterior chain-group checks preserve that ranking, but these checks are not confidence intervals or evidence of robustness to different response models. Avoid choosing between the two designs on this small advantage alone.

Separately, the hypothetical-calibrated allocation has expected revenue gain **1.43m CU**, with a **90% credible interval of [−2.52m, 5.13m]**, versus original historical spending. The interval allows losses; it is not an uncertainty interval for experimental information value.

## 3. What assumptions and limitations could change the decision?

- **Model and causal assumptions:** Responses are positive, additive and stable, with fixed Hill slope, geometric carryover and no modeled spillovers. Control validity is assumed; sentiment timing remains undocumented. The concave response helps explain why a cut can generate a stronger signal than an equal percentage increase. That advantage may change under another response specification.
- **Decision scope:** Total budget and ±20% channel bounds are fixed; historical regional and weekly patterns, impression prices and conversion values persist. The 118-week scenario is not a forecast of the next planning period. Changing constraints or the decision horizon could change experiment priorities.
- **Predictive evidence:** The historical-data posterior improves holdout RMSE by 7.9%, but beats regional means in only 27/40 regions. Nominal 90% intervals cover 88.2% of outcomes. These results support further model evaluation, not causal validation; they do not evaluate the hypothetical-calibrated posterior.
- **Experimental feasibility:** All tested precision levels are assumed. The earlier power analysis did not establish a feasible design meeting its targets. Neither hypothetical calibration nor information-value screening resolves that gap. Updating assumes an independent, unbiased experimental summary that transports to the budget decision.
- **Economics and computation:** Margins, execution costs, delays and operational restrictions are missing. Information values also depend on finite posterior draws; integration and chain-group checks address numerical sensitivity, not all model uncertainty. The tested candidate set is limited and does not establish a globally optimal experiment.

## 4. What must happen before execution, and when do we review?

Proceed with a bounded feasibility comparison of the two shortlisted Channel0 designs:

| Owner | Required input | Decision it enables |
|---|---|---|
| Analytics | Credible region assignment and estimation strategy; achievable revenue SE and its sensitivity to dependence and spillovers | Determine whether the assumed precision can be delivered |
| Finance | Contribution margins, test revenue/spending effects, implementation costs and delay costs on an agreed horizon | Compare net expected economic value, including a no-experiment option |
| Marketing | Funding for an increase, tolerance for lost conversions under a cut, operational constraints and the next planning window | Establish which interventions are executable and relevant |
| Analytics and Marketing | Sensitivity of priority to spending bounds and response assumptions | Determine whether the shortlist survives plausible changes |

These are proposed responsibilities, not completed checks or assigned commitments. Review the shortlist when these inputs are available. Advance to execution only if a feasible design has an acceptable downside and a defensible net benefit over acting without new evidence. Otherwise revise the design or defer experimentation.

Before launch, agree on the estimand, analysis, monitoring and stopping rules. After the experiment, incorporate its evidence once, inspect compatibility with historical predictions, and reassess allocation under current business inputs. Keep untouched evaluation data for model changes informed by the existing holdout.

**Current recommendation:** support Channel0 feasibility work and budget planning. The available analysis does not yet justify selecting the spending cut, launching an experiment, or deploying the full allocation change.
