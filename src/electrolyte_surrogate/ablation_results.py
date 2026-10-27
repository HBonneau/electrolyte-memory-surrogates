"""Evaluate the paired loss study without using test scores to select models."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DT
from .metrics import ordinary_tests, timing_tests
from .model import load_gru, predict
from .physics import reference_coefficients, stationary_coefficients
from .reduction import TEST_FIELDS


def step_suite():
    """Independent fine quadrature for five predeclared amplitude diagnostics."""
    sample = np.unique(np.r_[0, np.rint(np.geomspace(1, 20000, 300)).astype(int), 20000])
    elapsed = sample * DT
    observed = np.r_[0., .5 + elapsed]
    truth, stationary, refinement = [], [], []
    for field in TEST_FIELDS:
        values = np.where(observed >= .5, field, 0.)
        fine = reference_coefficients(observed, values, n_s=128, n_u=96)[1:]
        coarse = reference_coefficients(observed, values, n_s=96, n_u=64)[1:]
        steady = stationary_coefficients(field)
        late = elapsed >= 100
        uncertainty = np.max(np.abs(fine[late]-coarse[late]) / np.maximum(np.abs(fine[late]-steady), 1e-12), axis=0)
        if np.max(uncertainty) > .01:
            raise RuntimeError("Step-suite quadrature uncertainty exceeds 1% of the resolved tail.")
        truth.append(fine)
        stationary.append(steady)
        refinement.append(uncertainty.tolist())
    return sample, elapsed, np.stack(truth), np.stack(stationary), refinement


def tail_metrics(truth, estimate, stationary, elapsed):
    """Separate error versus the physical limit from variation about an endpoint."""
    departure = truth - stationary[:, None, :]
    error = estimate - truth
    report = {}
    for label, lower, upper in [("within_training", 10., 99.5), ("beyond_training", 100., 1000.)]:
        selected = (elapsed >= lower) & (elapsed <= upper)
        denominator = np.maximum(np.abs(departure[:, selected]), 1e-12)
        report[label] = dict(
            relative_rmse=np.sqrt(np.mean((error[:, selected]/denominator)**2, axis=(0, 1))).tolist(),
            reference_fraction_above_floor=np.mean(np.abs(departure[:, selected]) >= 1e-12, axis=(0, 1)).tolist(),
            floor=1e-12)
    # A finite endpoint is not an independently established infinite-time limit.
    late = elapsed >= 500
    endpoint = estimate[:, -1]
    variation = np.ptp(estimate[:, late], axis=1)
    ulp = np.abs(np.spacing(endpoint.astype(np.float32))).astype(float)
    report["endpoint_diagnostic"] = dict(
        fields=TEST_FIELDS, elapsed_interval=[500., 1000.],
        endpoint_stationary_departure=(endpoint-stationary).tolist(),
        reference_endpoint_departure=(truth[:, -1]-stationary).tolist(),
        endpoint_prediction_error=(endpoint-truth[:, -1]).tolist(),
        late_prediction_range=variation.tolist(),
        late_reference_range=np.ptp(truth[:, late], axis=1).tolist(),
        float32_output_ulp=ulp.tolist(),
        endpoint_error_in_output_ulps=(np.abs(endpoint-truth[:, -1])/np.maximum(ulp, 1e-30)).tolist(),
        interpretation="Finite-window endpoint bias and numerical variation; not a measured decay exponent about the model's own limit.")
    return report


def aggregate(rows, arm, value):
    values = np.array([value(row) for row in rows if row["arm"] == arm])
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                values=values.tolist())


def evaluate_study(study, data, probes, output):
    """Score all fixed-budget endpoints; preserve per-seed and signed tail data."""
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((study / "configuration.json").read_text())
    sample, elapsed, truth, stationary, refinement = step_suite()
    uniform = np.arange(20011) * DT
    selected = sample + 10
    rows, curves, csv_rows, hashes = [], {"mse": [], "mixed": []}, [], {}
    for seed in configuration["seeds"]:
        for arm in ["mse", "mixed"]:
            checkpoint_path = study / f"seed-{seed}" / f"{arm}.pt"
            hashes[str(checkpoint_path.relative_to(study))] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            model, scale, metadata = load_gru(checkpoint_path)
            if (metadata["seed"] != seed or metadata["continuation_epochs"] != configuration["epochs"]
                    or metadata["loss"] != arm):
                raise ValueError("Ablation checkpoint metadata do not match the experiment.")

            def run(tau, field):
                return predict(field, model, scale)[0]

            ordinary, prefix, regimes = ordinary_tests(data, {arm: run})
            timing = timing_tests(probes, {arm: run})[arm]
            estimates = np.stack([run(uniform, np.where(uniform >= .5, f, 0.))[selected]
                                  for f in TEST_FIELDS])
            tails = tail_metrics(truth, estimates, stationary, elapsed)
            row = dict(seed=seed, arm=arm,
                       ordinary={name: result[arm] for name, result in ordinary.items()},
                       prefix_0_12={name: result[arm] for name, result in prefix.items()},
                       regimes={name: result[arm] for name, result in regimes.items()},
                       timing_probes=timing, tails=tails)
            rows.append(row)
            curves[arm].append(np.abs((estimates[TEST_FIELDS.index(3.)]-stationary[TEST_FIELDS.index(3.)])/[3., 1.]))
            for i, field in enumerate(TEST_FIELDS):
                for t, reference, estimate in zip(elapsed, truth[i], estimates[i]):
                    csv_rows.append([seed, 0 if arm == "mse" else 1, field, t, *reference, *estimate])
            print(f"Evaluated loss study: seed={seed}, loss={arm}", flush=True)
    summary = {}
    measures = dict(
        Q_el_nrmse=lambda row: row["ordinary"]["test"]["Q_el_nrmse"],
        hydrodynamic_residual_nrmse=lambda row: row["ordinary"]["test"]["hydrodynamic_residual_nrmse"],
        recent_Q_el_nrmse=lambda row: row["regimes"]["test"]["recent_change"]["Q_el_nrmse"],
        timing_Q_el_nrmse=lambda row: row["timing_probes"]["overall"]["Q_el_nrmse"],
        tail_relative_rmse_el=lambda row: row["tails"]["beyond_training"]["relative_rmse"][0],
        tail_relative_rmse_hyd=lambda row: row["tails"]["beyond_training"]["relative_rmse"][1])
    for arm in ["mse", "mixed"]:
        summary[arm] = {name: aggregate(rows, arm, measure) for name, measure in measures.items()}
    paired = {}
    for name, measure in measures.items():
        a = [measure(row) for row in rows if row["arm"] == "mse"]
        b = [measure(row) for row in rows if row["arm"] == "mixed"]
        difference = np.asarray(b)-np.asarray(a)
        paired[name] = dict(mixed_minus_mse=difference.tolist(),
                            mean_difference=float(np.mean(difference)),
                            mixed_lower_count=int(np.sum(difference < 0)))
    report = dict(configuration=configuration, checkpoint_hashes=hashes,
                  reference=dict(fields=TEST_FIELDS, fine_quadrature=[128, 96], coarse_quadrature=[96, 64],
                                 late_interval=[100., 1000.], relative_quadrature_difference=refinement),
                  summary=summary, paired_differences=paired, rows=rows,
                  uncertainty="Sample standard deviations across training seeds; not confidence intervals or dataset uncertainty.")
    (output / "loss_ablation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savetxt(output / "loss_ablation_tail_predictions.csv", np.asarray(csv_rows), delimiter=",",
               header="seed,arm_0_mse_1_mixed,field,elapsed,reference_Q_el,reference_Y_hyd,predicted_Q_el,predicted_Y_hyd", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    positive = elapsed > 0
    reference = np.abs((truth[TEST_FIELDS.index(3.)]-stationary[TEST_FIELDS.index(3.)])/[3., 1.])
    for j, ax in enumerate(axes):
        for arm, color, label in [("mse", "#d56b24", "MSE"), ("mixed", "#176b9c", "MSE + late asinh")]:
            values = np.stack(curves[arm])[:, positive, j]
            ax.fill_between(elapsed[positive], np.maximum(values.min(axis=0), 1e-16),
                            np.maximum(values.max(axis=0), 1e-16), color=color, alpha=.15)
            ax.loglog(elapsed[positive], values.mean(axis=0), color=color, label=label)
        ax.loglog(elapsed[positive], reference[positive, j], "k--", label="Full reference")
        ax.axvline(99.5, color=".5", ls=":", label="100-time training horizon")
        ax.set(xlabel=r"Time since step / $t_D$", ylabel=r"$|\gamma-\gamma_\infty|$",
               title=["Electrostatic", "Hydrodynamic"][j])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Paired loss experiment: mean absolute departure; shading spans {len(configuration['seeds'])} seeds")
    fig.savefig(output / "loss_ablation.png", dpi=180)
    plt.close(fig)
    return summary
