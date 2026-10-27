# Learning electrolyte memory

**Can a model predict currents accurately while learning the wrong long-time relaxation?**

A small GRU predicts nonlinear electrolyte currents from unseen electric-field histories with about 0.6% normalized error over 100 Debye times. Yet its small departure from stationarity levels off while the physical reference continues to decay algebraically.

This scientific-ML study makes that failure measurable, then compares it with a reduced model that retains the physical memory and learns positive integration weights.

## From a field history to separate current terms

![Field histories and resulting current terms over 100 Debye times](results/field_responses.png)

Each column shows a different driving protocol: a step, pulse, reversal, or oscillating field. The rows show the field (also the dimensionless bare-current coefficient), electrostatic current coefficient, and full hydrodynamic current coefficient. These additional examples place changes throughout the 100-time window; they are separate from the training histories. The physical reduction and reference nearly overlap.

## Average accuracy and slow relaxation

| Model | Electrostatic test NRMSE | Hydrodynamic residual test NRMSE |
|---|---:|---:|
| Causal regression baseline | 6.38% | 18.10% |
| GRU trained through 100 Debye times | 0.56% | 0.62% |
| Physical reduction, learned weights | 0.00057% | 0.00095% |

These scores cover all samples of the 128 standard test histories. The full evaluation also reports 64 waveform/amplitude challenge histories and separate early-time scores, because the long stationary holds can make average errors look small.

![Small departures reveal incorrect long-time relaxation](results/relaxation.png)

The GRU's departure from the **true stationary current** becomes biased even within the 100-time training window. The step diagnostic continues through 1,000 times to expose the discrepancy. The selected 288-mode physical reduction reaches **1.12% worst held-out tail relative RMS error through 1,000 times**, compared with **3.90%** for ordinary integration weights on the same grid.

These are different learning problems: the GRU learns dynamics from sampled histories through time 100; the physical reduction receives known equations and is fitted to on/off curves through time 1,000. Its success demonstrates supervised physical model reduction, not architectural superiority or neural discovery of the decay law.

## Install and run

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m electrolyte_surrogate.demo
python -m electrolyte_surrogate.evaluate
python -m unittest discover -s tests -v
```

The demo writes the four-case figure and prediction CSVs to `runs/demo/`. Evaluation checks all 192 test histories and writes tail figures, signed predictions, and warmed inference timings to `results/reproduced/`. Saved results and checkpoints are included; training is optional. No GPU is required.

To fit each model separately:

```bash
python -m electrolyte_surrogate.train --model baseline
python -m electrolyte_surrogate.train --model gru
python -m electrolyte_surrogate.train --model physical
```

New fits go to `runs/training/`. GRU training uses 120 warm-up epochs on the early prefix, followed by 20 epochs on complete 100-time histories, selecting the checkpoint on full-history validation. See [the reproduction guide](docs/REPRODUCING.md) for settings, custom fields, and data generation.

## Where things live

```text
src/electrolyte_surrogate/
    model.py          GRU definition and prediction
    baseline.py       Regression fitting and prediction
    reduction.py      Physical modes and weight fitting
    physics.py        Full covariance reference calculation
    data.py           Loading, splits, and current reconstruction
    generate_data.py  Rebuild the 100-time synthetic histories
    train.py          Training commands
    evaluate.py       Metrics, tail diagnostics, and timing
    demo.py           Four field-to-current examples
artifacts/            Fitted models and training history
data/                Dataset and an example field CSV
results/              Measured results, figures, and predictions
tests/               Physics and ML correctness checks
docs/                Summary, conventions, and reproduction guide
pyproject.toml        Dependencies and package installation
```

Regression coefficients are stored in `artifacts/baseline.npz`, GRU weights and normalization in `artifacts/gru.pt`, and physical integration weights in `artifacts/covariance_weights.npz`. **Code defines the model; an artifact stores what was fitted.**

## Experimental scope

- 1,088 whole histories: 768 training, 128 validation, and 192 tests, all spanning 0–100 Debye times at intervals of 0.05.
- Training protocols change during the early part of the history, then hold the final field through time 100. The overview adds late-change timing probes.
- Validation selects settings and checkpoints. The repeatedly inspected f=3 long step is a diagnostic, not a blind test.
- Current errors exclude known baselines. Tail tests subtract the same true stationary value; switch-off also checks signs.

The data are synthetic predictions of a symmetric binary Gaussian point-ion theory, not experimental validation. A finite physical mode set approximates the continuum tail over a finite tested window. These single-seed fits do not establish a general limitation of GRUs.

Read [the project summary](docs/SUMMARY.md) and [physical conventions](docs/PHYSICS.md). Development used Codex assistance, with numerical verification against the original calculations.

Code, supplied synthetic data, and trained weights are released under the [MIT license](LICENSE). Papers are cited rather than redistributed.

[Upload instructions](docs/GITHUB.md) · [Verification record](docs/VERIFICATION.json)
