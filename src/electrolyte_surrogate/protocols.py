"""A fixed evaluation-only set with field changes throughout 100 Debye times."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import DT
from .losses import elapsed_since_change
from .physics import reference_coefficients

PROBE_SEED = 20261027


def timing_histories(seed=PROBE_SEED):
    """32 new timing probes: eight steps, pulses, reversals, and switching runs."""
    rng = np.random.default_rng(seed)
    tau = np.arange(2001) * DT
    for family in ["late_step", "late_pulse", "late_reversal", "late_switching"]:
        for _ in range(8):
            field = np.zeros_like(tau)
            amplitude = float(rng.uniform(.3, 5) * rng.choice([-1, 1]))
            if family == "late_step":
                field[tau >= rng.uniform(25, 70)] = amplitude
            elif family == "late_pulse":
                start = rng.uniform(20, 65)
                field[(tau >= start) & (tau < start + rng.uniform(2, 15))] = amplitude
            elif family == "late_reversal":
                start, middle, stop = rng.uniform(5, 20), rng.uniform(35, 55), rng.uniform(70, 90)
                field[(tau >= start) & (tau < middle)] = amplitude
                field[(tau >= middle) & (tau < stop)] = -amplitude
            else:
                edges = np.r_[rng.uniform(2, 8), np.sort(rng.uniform(15, 85, 6)), 90.]
                for left, right in zip(edges[:-1], edges[1:]):
                    value = 0. if rng.random() < .2 else rng.uniform(.3, 5) * rng.choice([-1, 1])
                    field[(tau >= left) & (tau < right)] = value
            yield family, field.astype(np.float32)


def regime_masks(field):
    """Disjoint elapsed-time bins plus the field-off portion of long holds."""
    elapsed, changed = elapsed_since_change(field)
    return dict(recent_change=changed & (elapsed < 1 - 1e-8),
                relaxing=changed & (elapsed >= 1 - 1e-8) & (elapsed < 8 - 1e-8),
                long_hold=changed & (elapsed >= 8 - 1e-8),
                long_hold_field_off=changed & (elapsed >= 8 - 1e-8) & (np.atleast_2d(field) == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/generated_data/timing_probe.npz"))
    args = parser.parse_args()
    tau = np.arange(2001) * DT
    families, fields, targets, refinement = [], [], [], []
    for index, (family, field) in enumerate(timing_histories(), start=1):
        fine = reference_coefficients(tau, field, n_s=96, n_u=64)
        coarse = reference_coefficients(tau, field, n_s=48, n_u=48)
        refinement.append(np.max(np.abs(fine-coarse), axis=0).tolist())
        families.append(family)
        fields.append(field)
        targets.append(fine)  # Preserve float64 for small-signal evaluation.
        print(f"Timing probe {index}/32: {family}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tau=tau, field=np.stack(fields), target=np.stack(targets),
                        family=np.array(families))
    configuration = dict(seed=PROBE_SEED, role="Evaluation only; never used for fitting or checkpoint selection",
                         histories=32, dt=DT, horizon=100., fine_quadrature=[96, 64],
                         coarse_quadrature=[48, 48], max_abs_quadrature_difference=np.max(refinement, axis=0).tolist(),
                         dataset_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(configuration, indent=2) + "\n")


if __name__ == "__main__":
    main()
