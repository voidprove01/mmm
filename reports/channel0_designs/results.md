# Channel0 intervention-intensity results

All values are model-conditional and use simulated source data. Revenue SE is fixed across interventions within a tier; no precision or power is claimed achievable.

## Middle precision tier

| Intervention | Test spend change (CU) | Expected test revenue change (CU) | Revenue SE (CU) | Downstream EVSI (CU) | Expected total budget moved |
|---|---:|---:|---:|---:|---:|
| -20% | -396,346 | -516,863 | 118,904 | 1,135,956 | 2.87% |
| +10% | +198,173 | +235,316 | 118,904 | 684,830 | 2.47% |
| +20% | +396,346 | +457,369 | 118,904 | 1,088,305 | 2.84% |

Test revenue includes eight intervention weeks and twelve followup weeks. EVSI covers the separate 118-week budget-decision template. Test revenue and spending are not included in EVSI; the two horizons should not be conflated.

## Precision sensitivity

| Intervention | Low precision EVSI | Medium precision EVSI | High precision EVSI |
|---|---:|---:|---:|
| -20% | 736,921 | 1,135,956 | 1,324,469 |
| +10% | 294,052 | 684,830 | 1,100,839 |
| +20% | 668,284 | 1,088,305 | 1,308,428 |

The shared revenue SEs are 237,808, 118,904 and 59,452 CU. For +10%, the corresponding return SEs are 1.20, 0.60 and 0.30; for either 20% intervention, they are 0.60, 0.30 and 0.15 when scaled by absolute spend change.

## Interpretation

The −20% intervention ranks slightly above +20% on information value in all three precision scenarios and both chain-group checks. At medium precision, the advantage is about **47,651 CU (4.4%)**. Both two-chain groups preserve the comparison, but this does not establish robustness to response-model or measurement-noise assumptions. At high precision, the difference falls to about 16,041 CU.

The concave Hill response predicts a larger absolute revenue loss from −20% than the gain from +20%: approximately 516,863 versus 457,369 CU. Equal observation noise can therefore make the cut more informative. This is an explanation conditional on the model, not a general preference for cuts. The +10% signal is smaller, and its information value is lower when absolute revenue noise is held fixed.

**Decision implication:** keep −20% and +20% on the shortlist. Do not select the cut from EVSI alone: its expected lost revenue, saved spend, margins, operational constraints and achievable precision must be considered together. +10% remains a smaller-spend alternative; it is not equivalent in information under equal revenue noise.

## Numerical checks

The +20% effects and EVSI reproduce the initial candidate screen. Maximum 256-to-512-node EVSI change is 9.52 CU. At most 1.64% of predictive probability has weight ESS below 100. Weight ESS is not autocorrelation-adjusted MCMC ESS.

At medium precision, the two chain groups yield −20% EVSI of 1.121m and 1.147m CU, compared with +20% EVSI of 1.073m and 1.100m CU. These splits are numerical sensitivity checks, not uncertainty intervals for information value.

[Full results and diagnostics](design_comparison.csv) · [Design definitions](../../docs/channel0_designs.md)
