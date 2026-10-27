"""Measure ordinary prediction accuracy separately from tiny relaxation tails."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baseline import load_baseline, predict_baseline
from .data import DT, load_dataset
from .model import load_gru, predict
from .physics import reference_coefficients, stationary_coefficients
from .reduction import CovarianceModes, FIT_TIMES, TEST_FIELDS
from .metrics import metrics, ordinary_tests, timing_tests

COLORS = {"gru": "#d56b24", "physical": "#8543a1", "baseline": "#687986"}
LABELS = {"gru": "GRU trained through 100 Debye times",
          "physical": "Physical reduction", "baseline": "Regression baseline"}


def load_predictors(artifacts):
    """Load saved models once; expose the same (time, field) inference interface."""
    gru, scale, metadata = load_gru(artifacts / "gru.pt")
    if metadata["training_tmax"] != 100:
        raise ValueError("Use a GRU trained on complete 100-time histories.")
    if not np.isclose(metadata["dt"], DT):
        raise ValueError("The GRU checkpoint must use dt=0.05.")
    baseline = load_baseline(artifacts / "baseline.npz")
    physical = CovarianceModes.load(artifacts / "covariance_weights.npz")

    def predict_gru(tau, field):
        return predict(field, gru, scale)[0]

    def predict_regression(tau, field):
        return predict_baseline(field, baseline)

    def predict_physical(tau, field):
        if field.ndim == 1:
            return physical.predict(tau, field)[0]
        return np.stack([physical.predict(tau, row)[0] for row in field])

    return dict(gru=predict_gru, baseline=predict_regression, physical=predict_physical)


def plot_accuracy(report, output):
    """Keep the data-matched comparison separate from the physical reduction."""
    conditions = [
        ("Standard whole history", lambda name: report["ordinary"]["test"][name]),
        (r"After change (<1 $t_D$)", lambda name: report["regimes"]["test"][name]["recent_change"]),
        (r"Long hold ($\geq8\,t_D$)", lambda name: report["regimes"]["test"][name]["long_hold"]),
        ("Unseen waveforms", lambda name: report["ordinary"]["waveform_ood"][name]),
        ("Larger amplitudes", lambda name: report["ordinary"]["amplitude_ood"][name]),
        ("Late field changes", lambda name: report["timing_probes"][name]["overall"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    positions = np.arange(len(conditions))
    for ax, key, title in zip(axes, ["Q_el_nrmse", "hydrodynamic_residual_nrmse"],
                              ["Electrostatic current", "Hydrodynamic residual"]):
        values = {name: np.array([100 * getter(name)[key] for _, getter in conditions])
                  for name in ["baseline", "gru"]}
        ax.hlines(positions, np.minimum(values["baseline"], values["gru"]),
                  np.maximum(values["baseline"], values["gru"]), color=".8", zorder=1)
        for name in ["baseline", "gru"]:
            ax.scatter(values[name], positions, color=COLORS[name], label=LABELS[name], s=35, zorder=2)
        ax.set(xscale="log", xlabel="NRMSE (%) — logarithmic scale", title=title)
        ax.set_yticks(positions, [label for label, _ in conditions])
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("Regression and GRU: same training data, different evaluation conditions")
    fig.savefig(output / "accuracy_by_regime.png", dpi=180)
    plt.close(fig)


def step_test(predictors, output):
    """Use full uniform neural histories, sparse observations, and a refined solver."""
    field_value, onset, horizon = 3., .5, 1000.
    tau = np.arange(int(round((onset + horizon) / DT)) + 1) * DT
    field = np.where(tau >= onset, field_value, 0.)
    sample = np.unique(np.r_[0, np.rint(np.geomspace(1, 20000, 300)).astype(int), 20000])
    elapsed = sample * DT
    observed = np.r_[0., onset + elapsed]
    sampled_field = np.where(observed >= onset, field_value, 0.)
    truth = reference_coefficients(observed, sampled_field, n_s=128, n_u=96)[1:]
    coarse = reference_coefficients(observed, sampled_field, n_s=96, n_u=64)[1:]
    steady = stationary_coefficients(field_value)
    delta = truth - steady
    late = elapsed >= 100
    refinement = np.max(np.abs(coarse[late] - truth[late]) / np.abs(delta[late]), axis=0)
    if np.max(refinement) > .01:
        raise RuntimeError("Reference quadrature uncertainty exceeds 1% of the tail.")
    selected = int(round(onset / DT)) + sample
    estimates = {name: run(tau, field)[selected] for name, run in predictors.items()
                 if name != "baseline"}
    report = dict(field=field_value, elapsed_horizon=horizon, stationary_latent=steady.tolist(),
                  late_interval=[100., 1000.], quadrature_relative_difference=refinement.tolist(),
                  reference_final_gamma_departure=(delta[-1] / [field_value, 1]).tolist(),
                  reference_slopes=[float(np.polyfit(np.log(elapsed[late]),
                      np.log(np.abs(delta[late, j])), 1)[0]) for j in range(2)], models={})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    rows = [elapsed, *truth.T, *delta.T]
    headers = ["elapsed", "reference_Q_el", "reference_Y_hyd", "reference_delta_Q_el", "reference_delta_Y_hyd"]
    for name, estimate in estimates.items():
        departure = estimate - steady
        relative = (departure[late] - delta[late]) / np.maximum(np.abs(delta[late]), 1e-12)
        report["models"][name] = dict(
            final_gamma_departure=(departure[-1] / [field_value, 1]).tolist(),
            tail_relative_rmse=np.sqrt(np.mean(relative**2, axis=0)).tolist(),
            slopes=[float(np.polyfit(np.log(elapsed[late]),
                        np.log(np.maximum(np.abs(departure[late, j]), 1e-300)), 1)[0]) for j in range(2)])
        rows.extend(estimate.T)
        headers.extend([name + "_Q_el", name + "_Y_hyd"])
        for j in range(2):
            mask = elapsed > 0
            axes[j].loglog(elapsed[mask], np.abs(departure[mask, j]) / [field_value, 1][j],
                           color=COLORS[name], label=LABELS[name])
    for j in range(2):
        mask = elapsed > 0
        axes[j].loglog(elapsed[mask], np.abs(delta[mask, j]) / [field_value, 1][j],
                       "k--", label="Full covariance reference", zorder=5)
        axes[j].axvline(99.5, color=".5", ls=":", label="100-time training horizon")
        axes[j].set_xlabel(r"Time since step / $t_D$")
        axes[j].set_ylabel(r"$|\gamma-\gamma_\infty|$")
        axes[j].set_title(["Electrostatic", "Hydrodynamic"][j] + " departure")
        axes[j].grid(alpha=.2)
        axes[j].legend(fontsize=7)
    fig.suptitle("A small current error can conceal the wrong long-time relaxation")
    fig.savefig(output / "relaxation.png", dpi=180)
    plt.close(fig)
    np.savetxt(output / "step_predictions.csv", np.column_stack(rows), delimiter=",",
               header=",".join(headers), comments="")
    return report


def switch_off_test(predictors, output):
    """Compare the same finite prehistory; magnitude plots also need sign checks."""
    tau = np.arange(2401) * DT
    field = np.where(tau < 100, 3., 0.)
    elapsed = np.arange(401) * DT
    observed = np.r_[0., 100 + elapsed]
    truth = reference_coefficients(observed, np.where(observed < 100, 3., 0.), n_s=96, n_u=64)[1:, 0]
    significant = np.abs(truth) >= 1e-12
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    report, columns, headers = {}, [elapsed, truth], ["elapsed_after_off", "reference_Q_el"]
    for name, run in predictors.items():
        if name == "baseline":
            continue
        full = run(tau, field)
        current = full[2000:, 0]
        relative = (current[significant] - truth[significant]) / np.abs(truth[significant])
        wrong = current * truth < 0
        report[name] = dict(relative_rmse_above_floor=float(np.sqrt(np.mean(relative**2))),
                            wrong_sign_samples=int(np.sum(wrong & significant)),
                            reference_floor=1e-12, scored_samples=int(significant.sum()))
        ax.semilogy(elapsed, np.maximum(np.abs(current), 1e-16), color=COLORS[name], label=LABELS[name])
        ax.scatter(elapsed[wrong & significant][::8], np.abs(current[wrong & significant][::8]),
                   color=COLORS[name], marker="x", s=18)
        columns.append(current)
        headers.append(name + "_Q_el")
    ax.semilogy(elapsed, np.abs(truth), "k--", label="Full covariance reference")
    ax.set(xlabel=r"Time after switch-off / $t_D$", ylabel=r"$|Q_{el}|$",
           title="Hold f=3 for 100 Debye times, then switch off; crosses mark wrong signs")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(output / "switch_off.png", dpi=180)
    plt.close(fig)
    np.savetxt(output / "switch_off_predictions.csv", np.column_stack(columns), delimiter=",",
               header=",".join(headers), comments="")
    return report


def physical_tail_tests(artifacts):
    """Recheck held-out amplitudes using the original physical-mode score."""
    teacher = CovarianceModes.gauss(96, 48)
    learned = CovarianceModes.load(artifacts / "covariance_weights.npz")
    report = {}
    late = FIT_TIMES >= 10
    for name, model in [("learned", learned), ("ordinary_weights", CovarianceModes.gauss())]:
        rows = []
        for field in TEST_FIELDS:
            truth, estimate = teacher.step(field, FIT_TIMES), model.step(field, FIT_TIMES)
            relative = (estimate["on"][late] - truth["on"][late]) / np.maximum(np.abs(truth["departure"][late]), 1e-12)
            off_relative = (estimate["off_scaled"][late] - truth["off_scaled"][late]) / np.maximum(np.abs(truth["off_scaled"][late]), 1e-12)
            rows.append(dict(field=field, on_tail_relative_rmse=np.sqrt(np.mean(relative**2, axis=0)).tolist(),
                             off_scaled_relative_rmse=float(np.sqrt(np.mean(off_relative**2)))))
        report[name] = dict(rows=rows, worst_tail_relative_rmse=max(
            max(max(row["on_tail_relative_rmse"]), row["off_scaled_relative_rmse"]) for row in rows))
    return report


def benchmark(data, predictors):
    """Same warmed single-history task; exclude model loading and training."""
    index = np.flatnonzero((data["split"] == "test") & (data["family"] == "random"))[0]
    tau, field = data["tau"], data["field"][index]
    runners = {**predictors, "reference": reference_coefficients}
    durations = {}
    for name, run in runners.items():
        run(tau, field)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            run(tau, field)
            samples.append(time.perf_counter() - started)
        durations[name] = float(np.median(samples))
    return dict(history_index=int(index), median_seconds=durations,
                speedup_vs_reference={name: durations["reference"] / value for name, value in durations.items() if name != "reference"},
                scope=f"Five warmed CPU calls per model; one identical {len(tau)}-sample history; includes input preparation; excludes loading, training, and data generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.npz"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--timing-data", type=Path, default=Path("data/timing_probe.npz"))
    parser.add_argument("--ablation", type=Path, help="Optional directory produced by electrolyte_surrogate.ablate")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    data, predictors = load_dataset(args.data), load_predictors(args.artifacts)
    ordinary, prefix, regimes = ordinary_tests(data, predictors)
    with np.load(args.timing_data, allow_pickle=False) as archive:
        probes = {key: archive[key] for key in archive.files}
    report = dict(data_horizon=100, ordinary=ordinary, prefix_0_12=prefix, regimes=regimes,
                  timing_probes=timing_tests(probes, predictors))
    print("Ordinary test histories evaluated.", flush=True)
    report["step"] = step_test(predictors, args.output)
    report["switch_off"] = switch_off_test(predictors, args.output)
    report["physical_tails"] = physical_tail_tests(args.artifacts)
    report["timing"] = benchmark(data, predictors)
    plot_accuracy(report, args.output)
    if args.ablation:
        from .ablation_results import evaluate_study
        report["loss_ablation"] = evaluate_study(args.ablation, data, probes, args.output)
    report["artifact_hashes"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [args.data, args.timing_data, *args.artifacts.glob("*.pt"), *args.artifacts.glob("*.npz")]}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"standard_test": report["ordinary"]["test"],
                      "reference_slopes": report["step"]["reference_slopes"],
                      "saved_to": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
