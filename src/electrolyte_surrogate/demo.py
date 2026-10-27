"""Show field changes and both current terms across a 100-Debye-time window."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import DT, current_coefficients, load_field
from .evaluate import COLORS, LABELS, load_predictors
from .metrics import metrics
from .physics import reference_coefficients


def overview_fields(tau):
    """Independent timing-transfer probes, not extra training examples."""
    step = np.where(tau >= 10, 3., 0.)
    pulse = np.where((tau >= 20) & (tau < 40), 3., 0.)
    reversal = np.zeros_like(tau)
    reversal[(tau >= 10) & (tau < 45)] = 3
    reversal[(tau >= 45) & (tau < 75)] = -3
    oscillatory = np.zeros_like(tau)
    active = (tau >= 10) & (tau < 90)
    oscillatory[active] = 3 * np.sin(2 * np.pi * (tau[active] - 10) / 20)
    return dict(step=step, pulse=pulse, reversal=reversal, oscillatory=oscillatory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional tau,f CSV instead of the four overview cases")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.input:
        tau, field = load_field(args.input, uniform=True)
        cases = {"custom": field}
    else:
        tau = np.arange(2001) * DT
        cases = overview_fields(tau)
    predictors = load_predictors(args.artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(cases), figsize=(max(7, 3.6 * len(cases)), 7),
                             sharex=True, squeeze=False, layout="constrained")
    report = {}
    for col, (case, field) in enumerate(cases.items()):
        print("Calculating overview case:", case, flush=True)
        truth = reference_coefficients(tau, field)
        estimates = {name: run(tau, field) for name, run in predictors.items()}
        axes[0, col].step(tau, field, where="post", color="#526575")
        axes[0, col].set_title(case.capitalize())
        columns = [tau, field, *current_coefficients(field, truth).T]
        headers = ["tau", "f", "reference_Q_el", "reference_Q_hyd"]
        for name, latent in estimates.items():
            displayed = current_coefficients(field, latent)
            for j in range(2):
                axes[j + 1, col].plot(tau, displayed[:, j], color=COLORS[name], label=LABELS[name])
            columns.extend(displayed.T)
            headers.extend([name + "_Q_el", name + "_Q_hyd"])
        for j in range(2):
            axes[j + 1, col].plot(tau, current_coefficients(field, truth)[:, j],
                                  "k--", label="Full reference", zorder=5)
        axes[2, col].set_xlabel(r"Time / $t_D$")
        np.savetxt(args.output / f"overview_{case}_predictions.csv", np.column_stack(columns), delimiter=",",
                   header=",".join(headers), comments="")
        report[case] = {name: metrics(truth, estimate, field) for name, estimate in estimates.items()}
    axes[0, 0].set_ylabel("Field f and bare current")
    axes[1, 0].set_ylabel(r"Electrostatic $Q_{el}$")
    axes[2, 0].set_ylabel(r"Hydrodynamic $Q_{hyd}$")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Electric fields and induced current coefficients over 100 Debye times"
                 if args.input is None else "Electric field and induced current coefficients")
    fig.savefig(args.output / "field_responses.png", dpi=180)
    plt.close(fig)
    (args.output / "overview_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Outputs:", args.output)


if __name__ == "__main__":
    main()
