"""Fit the GRU, regression baseline, or physical integration weights."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .baseline import fit_baseline
from .data import DT, load_dataset, split_arrays
from .model import ElectrolyteGRU
from .reduction import fit_weights


def _train_epoch(model, optimizer, rng, fields, targets):
    model.train()
    total_loss = 0.
    order = rng.permutation(len(fields))
    for start in range(0, len(order), 64):
        indices = order[start:start + 64]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(fields[indices]) - targets[indices])**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += float(loss.detach()) * len(indices)
    return total_loss / len(fields)


def _validation(model, fields, targets):
    model.eval()
    with torch.inference_mode():
        error = (model(fields) - targets)**2
        return float(error.mean()), float(error[:, :241].mean())


def fine_tune_gru(model, optimizer, rng, fields, targets, val_fields, val_targets, epochs):
    """Continue through the complete 100-time histories, preserving Adam state."""
    for group in optimizer.param_groups:
        group["lr"] = .0003
    initial, prefix = _validation(model, val_fields, val_targets)
    best_loss, best_epoch = initial, 0
    best_weights = copy.deepcopy(model.state_dict())
    history = [dict(phase="long", epoch=0, training_mse=None,
                    validation_mse=initial, prefix_validation_mse=prefix)]
    for epoch in range(1, epochs + 1):
        training = _train_epoch(model, optimizer, rng, fields, targets)
        validation, prefix = _validation(model, val_fields, val_targets)
        history.append(dict(phase="long", epoch=epoch, training_mse=training,
                            validation_mse=validation, prefix_validation_mse=prefix))
        if validation < best_loss * (1 - 1e-4):
            best_loss, best_epoch = validation, epoch
            best_weights = copy.deepcopy(model.state_dict())
        print(f"100-time epoch {epoch}/{epochs}: validation MSE={validation:.6g}", flush=True)
    return best_weights, best_epoch, history


def warmup_gru(data, seed=13, epochs=120):
    """Fit the common early-time initialization used by both loss experiments."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    field, target = split_arrays(data, "train")
    val_field, val_target = split_arrays(data, "validation")
    scale = np.sqrt(np.mean(target[:, :241]**2, axis=(0, 1))).astype(np.float32)
    fields, targets = torch.from_numpy(field), torch.from_numpy(target / scale)
    val_fields = torch.from_numpy(val_field)
    val_targets = torch.from_numpy(val_target / scale)
    model = ElectrolyteGRU(hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=7)
    history = []
    for epoch in range(1, epochs + 1):
        training = _train_epoch(model, optimizer, rng, fields[:, :241], targets[:, :241])
        validation, _ = _validation(model, val_fields[:, :241], val_targets[:, :241])
        scheduler.step(validation)
        history.append(dict(phase="warmup", epoch=epoch,
                            training_mse=training, validation_mse=validation))
        if epoch == 1 or epoch % 10 == 0:
            print(f"Warm-up {epoch}/{epochs}: validation MSE={validation:.6g}", flush=True)
    return model, optimizer, rng, scale, history


def train_gru(data, epochs=20, seed=13, warmup_epochs=120):
    """Warm up on the prefix, then select by full 100-time validation MSE."""
    model, optimizer, rng, scale, history = warmup_gru(data, seed, warmup_epochs)
    field, target = split_arrays(data, "train")
    val_field, val_target = split_arrays(data, "validation")
    fields, targets = torch.from_numpy(field), torch.from_numpy(target / scale)
    val_fields, val_targets = torch.from_numpy(val_field), torch.from_numpy(val_target / scale)
    best_weights, best_epoch, long_history = fine_tune_gru(
        model, optimizer, rng, fields, targets, val_fields, val_targets, epochs
    )
    checkpoint = dict(state_dict=best_weights, hidden=32, target_scale=scale.tolist(),
                      dt=DT, initial_state="equilibrium", training_tmax=float(data["tau"][-1] - data["tau"][0]),
                      best_epoch=warmup_epochs + best_epoch, selected_additional_epoch=best_epoch,
                      warmup_epochs=warmup_epochs, continuation_epochs=epochs, seed=seed)
    return checkpoint, history + long_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gru", "baseline", "physical"], default="gru")
    parser.add_argument("--data", type=Path, default=Path("data/dataset.npz"))
    parser.add_argument("--epochs", type=int, default=20, help="GRU epochs on complete 100-time histories")
    parser.add_argument("--warmup-epochs", type=int, default=120, help="Initial GRU epochs on the prefix")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("runs/training"))
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup_epochs < 1:
        parser.error("Epochs must be positive.")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.model == "physical":
        model, report = fit_weights()
        model.save(args.output / "covariance_weights.npz")
    else:
        data = load_dataset(args.data)
        if args.model == "gru":
            checkpoint, history = train_gru(data, epochs=args.epochs, seed=args.seed,
                                            warmup_epochs=args.warmup_epochs)
            torch.save(checkpoint, args.output / "gru.pt")
            (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
            selected = next(row for row in history if row["phase"] == "long"
                            and row["epoch"] == checkpoint["selected_additional_epoch"])
            report = dict(seed=args.seed, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                          selected_additional_epoch=checkpoint["selected_additional_epoch"],
                          validation_scaled_mse=selected["validation_mse"])
        else:
            field, target = split_arrays(data, "train")
            val_field, val_target = split_arrays(data, "validation")
            scale = np.sqrt(np.mean(target[:, :241]**2, axis=(0, 1))).astype(np.float32)
            fitted = fit_baseline(field, target, val_field, val_target, scale)
            np.savez(args.output / "baseline.npz", **{
                key: value for key, value in fitted.items() if key != "validation_mse"
            })
            report = {key: fitted[key] for key in ["lags", "ridge", "validation_mse"]}
    report.update(model=args.model, training_seconds=time.perf_counter() - started)
    (args.output / f"{args.model}_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
