# Good current predictions can conceal incorrect relaxation

## Question and setup

Can a compact neural surrogate learn the nonlinear current response of an electrolyte, including its small algebraic relaxation toward a stationary value?

The input is an electric-field history. Targets are the electrostatic correction and history-dependent hydrodynamic coefficient from a Gaussian covariance model for a symmetric binary electrolyte. Known bare-current and hydrodynamic field factors are supplied exactly. This avoids giving a model credit for fitting a dominant known baseline.

The dataset contains 1,088 complete histories spanning **100 Debye times**, sampled every 0.05 times. Steps, pulses, reversals, and random switching change early, then hold the final field through time 100. Whole histories define training (768), validation (128), and tests (192, including unseen waveforms and larger amplitudes).

A 32-unit GRU is compared with causal ridge regression using lagged polynomial field features. Both respect field parity and equilibrium; inputs are shifted one sample to preserve covariance causality. The GRU has 120 warm-up epochs on the first 12 times, then 20 full-history epochs with the same optimizer state and a lower learning rate. Full 100-time validation selects the supplied checkpoint. Regression is fitted on those same complete histories.

## What the reader should see

![Different field histories and their separate current responses](../results/field_responses.png)

These additional step, pulse, reversal, and oscillating protocols distribute changes throughout the 100-time window. They illustrate the field-to-current problem and probe transfer to later changes; they are not training examples. The plotted coefficients include the full hydrodynamic term, while the error metric removes its known instantaneous baseline.

On the 128 standard test histories, GRU electrostatic and hydrodynamic-residual NRMSE are **0.56% and 0.62%**, versus **6.38% and 18.10%** for regression. Early-prefix GRU errors are 1.11% and 1.28%; the normalizations differ, so the lower full-history scores alone do not establish better transient learning.

## The failure average error hides

A sustained f=3 step continues through 1,000 times. All models have the same true stationary value subtracted. The reference uses independently refined wavevector quadrature.

| Departure from stationary gamma at time 1,000 | Electrostatic | Hydrodynamic |
|---|---:|---:|
| Full reference | -2.72e-9 | +5.06e-7 |
| GRU trained through time 100 | -2.66e-6 | +3.91e-5 |
| Physical reduction | -2.79e-9 | +5.14e-7 |

![Unscaled stationary departures on logarithmic axes](../results/relaxation.png)

The reference's late log–log slopes are approximately -1.5. The GRU departures flatten around a biased stationary value, with a visible mismatch before the training boundary. The graph establishes incorrect relaxation toward the physical limit; it does not determine the network's decay law about its own biased limit. Longer training histories therefore did not resolve this failure for the supplied fit.

## Physical comparison and scope

The reduced model retains 288 Fourier modes with known field-dependent covariance equations. Learning adjusts shared positive integration weights. Its worst held-out tail relative RMS error is **1.12%**, versus **3.90%** for ordinary weights on the same grid, over times 10–1,000 and five held-out amplitudes.

This is supervised physical model reduction: it receives the dynamics and longer on/off fitting curves. Its accuracy cannot be attributed to architecture alone. Switch-off also matters: electrostatic relaxation includes an exponential factor, and the hydrodynamic current vanishes at zero field. A universal algebraic output envelope would impose incorrect physics.

**A small average current error is insufficient evidence that a surrogate has learned slow relaxation.** The portfolio contribution is a reproducible baseline, careful current targets, held-out histories, and a diagnostic that exposes failure on physically meaningful small signals.

All labels are synthetic theory predictions. These single-seed fits do not establish that GRUs cannot learn algebraic memory; the finite physical grid is also only validated over a finite window. Stationary bias, tiny-signal weighting, and float32 precision remain distinct issues. S4, pLSTM, and exploratory filter banks are omitted to keep this project focused.

References: [2023 temporal-response paper](https://doi.org/10.1088/1742-5468/acdced), [2025 covariance paper](https://doi.org/10.1088/1742-5468/adb4ce). Exact scores are in [metrics.json](../results/metrics.json).
