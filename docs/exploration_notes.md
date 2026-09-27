# Optional exploration: why the release is focused

The development version included detailed region-error audits and several notebooks on simulated geo experiments and power. These are omitted from the public walkthrough to keep the main question visible.

The important negative finding is retained: a 40-geo paired experiment with Channel2 +20% had very low power under the chosen Gaussian noise model. Duration, spend intensity and baseline-adjustment variants did not reach an 80% target for the tested decision scenarios. Those simulations were conditional on a particular noise model, not evidence that all geo experiments are infeasible. Treating MMM residuals as independent future shocks is itself a substantive assumption.

The calibration notebook does not disguise that failed design as a successful experiment. It starts with a separately stipulated high-information summary, clearly labeled hypothetical, so readers can see the evidence-to-posterior-to-decision workflow. Neither the experimental summary nor resulting posterior precision is claimed as a real-world measurement achievement.

Full exploration records remain in the author's local development workspace. The compact release preserves conclusions and limitations without publishing every intermediate notebook or failed sampler output.
