"""Rebuild the synthetic, whole-history dataset used in the GRU experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import DT
from .physics import reference_coefficients

TAU = np.arange(2001) * DT
FAMILIES = ["step", "pulse", "paired", "reversal", "random"]
COUNTS = dict(train=768, validation=128, test=128, waveform_ood=32, amplitude_ood=32)


def field_history(rng, family, bounds=(.3, 5)):
    """Generate one held-sample input, preserving the recorded random recipe."""
    field = np.zeros(len(TAU))
    cutoff = min(len(TAU) - 1, int(10 / DT))

    def amplitude():
        return float(rng.uniform(*bounds) * rng.choice([-1, 1]))

    start = int(rng.integers(4, 25))
    if family == "step":
        field[start:] = amplitude()
    elif family == "pulse":
        end = min(cutoff, start + int(rng.integers(3, 100)))
        field[start:end] = amplitude()
    elif family == "paired":
        width1 = int(rng.integers(3, 45))
        gap = int(rng.integers(2, 45))
        width2 = int(rng.integers(3, 45))
        field[start:start + width1] = amplitude()
        next_start = start + width1 + gap
        field[next_start:min(cutoff, next_start + width2)] = amplitude()
    elif family == "reversal":
        middle = int(rng.integers(start + 5, 100))
        end = int(rng.integers(middle + 5, cutoff))
        first = amplitude()
        field[start:middle] = first
        field[middle:end] = -np.sign(first) * abs(amplitude())
    elif family == "random":
        positions = np.unique(np.r_[start, rng.integers(start + 1, cutoff, size=8), cutoff])
        for left, right in zip(positions[:-1], positions[1:]):
            field[left:right] = 0 if rng.random() < .2 else amplitude()
    elif family in ["sine", "triangle"]:
        phase = (TAU - TAU[start]) / float(rng.uniform(.6, 3))
        active = (TAU >= TAU[start]) & (TAU < 9)
        wave = np.sin(2 * np.pi * phase)
        if family == "triangle":
            wave = 2 / np.pi * np.arcsin(wave)
        field[active] = amplitude() * wave[active]
    else:
        raise ValueError(f"Unknown field family: {family}")
    return field


def histories(seed=20261013):
    """Yield distinct inputs with deterministic, complete-history splits."""
    rng = np.random.default_rng(seed)
    seen = set()
    for split, count in COUNTS.items():
        for index in range(count):
            family = (["sine", "triangle"][index % 2] if split == "waveform_ood"
                      else FAMILIES[index % len(FAMILIES)])
            bounds = (6, 8) if split == "amplitude_ood" else (.3, 5)
            while True:
                field = field_history(rng, family, bounds)
                signature = hashlib.sha256(field.astype("<f4").tobytes()).hexdigest()
                if signature not in seen:
                    seen.add(signature)
                    break
            yield split, family, field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/generated_data/dataset.npz"))
    args = parser.parse_args()
    fields, targets, splits, families = [], [], [], []
    for index, (split, family, field) in enumerate(histories(), start=1):
        target = reference_coefficients(TAU, field)
        fields.append(field.astype(np.float32))
        targets.append(target.astype(np.float32))
        splits.append(split)
        families.append(family)
        if index % 64 == 0:
            print(f"Generated {index}/1088 histories", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tau=TAU, field=np.stack(fields), target=np.stack(targets),
                        split=np.array(splits), family=np.array(families))
    configuration = dict(seed=20261013, dt=DT, horizon=100, counts=COUNTS,
                         quadrature=[48, 48], targets=["Q_el", "gamma_hyd_plus_one"],
                         initial_state="zero-field equilibrium", interpolation="sample hold")
    args.output.with_suffix(".json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
