# Controlled loss experiment

**Hypothesis:** emphasizing small late-time departures in the training loss can improve relaxation predictions without changing the recurrent architecture.

## Fixed protocol

- Three training seeds: 13, 29, and 47, controlling initialization and minibatch order.
- The same 768 training and 128 validation histories, sampled through 100 Debye times. No added histories enter fitting.
- A shared 120-epoch MSE warm-up on the early prefix for each seed. Both continuation arms receive identical model weights, Adam moments, output scales, and minibatch order. Batch-order hashes are recorded and checked.
- Twenty continuation epochs for each arm, learning rate 0.0003, batches of 64, and gradient clipping at 1. Both use the predeclared final epoch; no test score, long-step diagnostic, or validation checkpoint search selects the winner.
- Validation MSE and tail loss are recorded for both arms after each epoch. They monitor the experiment rather than change its budget.

The control minimizes RMS-scaled MSE. The other arm minimizes MSE plus a late-time penalty:

```text
tail = mean((asinh((prediction - stationary)/epsilon)
           - asinh((target     - stationary)/epsilon))²)
loss = MSE + weight * tail
```

The penalty applies at least eight Debye times after the latest field change. Epsilon is 10% of the median departure eight times after switch-on in the **training step histories**, with a float32 arithmetic floor. A fixed calibration rule chooses the weight so that its initial gradient norm is 25% of the MSE gradient norm on the first 64 training records. The weight is fixed thereafter; the rule is the same for each seed. Neither calibration uses validation or test labels.

Known stationary coefficients center the loss only. They are not neural inputs or inference corrections, and no decay exponent is imposed. This physical information is an explicit part of the loss intervention. The calibration resolves departures around the selected waiting time; it does not guarantee relative accuracy on arbitrarily smaller signals.

## Evaluation

Report all six endpoints on the standard tests, unseen waveform/amplitude sets, and 32 evaluation-only late-change histories. Ordinary current errors are separated into recent changes (elapsed time below 1), intermediate relaxation (1–8), and long holds (at least 8). Field-off hydrodynamic NRMSE is undefined because the reference current is identically zero; its absolute error is still reported.

The long-step suite uses amplitudes 0.4, 1.2, 2.2, 3, and 4.8. Tail errors are reported separately over elapsed times 10–99.5 and 100–1,000, subtracting the same physical stationary value. Relative denominators have a floor of 1e-12, with the fraction of reference samples above that floor recorded. The f=3 curve and these amplitude diagnostics follow earlier exploration; they are not blind benchmarks.

Per-seed scores and paired differences accompany means and sample standard deviations. Three seeds characterize training variability on this fixed dataset, not general statistical confidence. The tail figure shows mean absolute departure and the range across seeds.

Finite-endpoint prediction error, stationary departure, and prediction variation over times 500–1,000 are recorded separately. A biased endpoint is not evidence for a particular decay law around the network's own infinite-time limit. Output spacing in float32 provides a numerical scale, not a decomposition of training and rounding errors.

## Reproduce

```bash
python -m electrolyte_surrogate.ablate
python -m electrolyte_surrogate.evaluate --ablation runs/loss-ablation
```

Training writes to `runs/loss-ablation/`. Warm-up states and completed arms can be reused on an identical rerun; changed settings require a new output directory. Interrupted arms restart from the shared warm-up. A short workflow check is:

```bash
python -m electrolyte_surrogate.ablate --seeds 13 --warmup-epochs 2 --epochs 1 --output runs/loss-smoke
```

The supplied six endpoints, training records, configuration, and source hashes are in `artifacts/loss_ablation/`. Evaluate them using `--ablation artifacts/loss_ablation`. Full results are in `results/loss_ablation.json`; signed tail predictions are saved separately. Earlier exploratory runs are not substituted into this matched experiment.
