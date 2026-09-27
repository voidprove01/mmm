# Candidate experiment results

Gross expected revenue decision value, in source currency units, over the 118-week historical allocation scenario. These are assumed-precision scenarios, not measured field results.

| Channel | Low precision (SE 0.60) | Medium precision (SE 0.30) | High precision (SE 0.15) | Extra test spend |
|---|---:|---:|---:|---:|
| Channel0 | 668,284 | 1,088,305 | 1,308,428 | 396,346 |
| Channel1 | 154,183 | 342,416 | 456,750 | 246,195 |
| Channel2 | 15,532 | 23,624 | 31,580 | 50,375 |
| Channel3 | 48,757 | 156,003 | 349,313 | 634,919 |
| Channel4 | 45,427 | 138,606 | 293,438 | 323,545 |

No experiment: zero incremental information value. Perfect information upper bound: **2.755 million CU** under the same posterior and decision model. Extra test spend is not the full net experiment cost.

Channel0 ranks first and Channel2 last at all three precision levels, including separate first-two-chain and last-two-chain calculations. This is a gross-value ranking at equal assumed return SE, not equal cost or equal feasibility.

Channel2 has the largest current experimental-return SD (0.685), but the lowest EVSI. At return SE 0.30 it remains at the +20% budget bound with approximately 99.98% posterior-predictive probability. This relates uncertainty to its decision consequences; it does not imply no possible experiment on Channel2 could be useful.

Quadrature refinements change EVSI by at most 8 CU. Finite posterior sensitivity is larger: Channel2 at SE 0.30 yields about 19,621 CU and 33,041 CU in the two chain groups, compared with 23,624 CU using all draws. These splits are diagnostics, not confidence intervals. At SE 0.15, 3.56% of Channel2 predictive probability has likelihood-weight ESS below 100. Tail estimates and small differences deserve caution.

[Full candidate results](candidate_results.csv) · [Protocol and limitations](../../docs/experiment_prioritization.md)
