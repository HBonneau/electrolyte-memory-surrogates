"""Compare MSE and MSE + late-asinh with paired seeds and fixed epoch budgets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import DT, load_dataset, split_arrays
from .losses import TailLoss, calibrate_epsilon, gradient_weight
from .model import ElectrolyteGRU
from .train import warmup_gru


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_pair(data, seed, warmup_epochs, epochs, output):
    """Share warm-up weights, Adam moments and minibatch order between arms."""
    output.mkdir(parents=True, exist_ok=True)
    warmup_path = output / "warmup_state.pt"
    if warmup_path.exists():
        state = torch.load(warmup_path, map_location="cpu", weights_only=False)
        assert state["seed"] == seed and state["warmup_epochs"] == warmup_epochs
    else:
        started = time.perf_counter()
        model, optimizer, rng, scale, history = warmup_gru(data, seed, warmup_epochs)
        state = dict(model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                     numpy_rng=rng.bit_generator.state, scale=scale, seed=seed,
                     warmup_epochs=warmup_epochs, warmup_seconds=time.perf_counter()-started,
                     history=history)
        torch.save(state, warmup_path)
    field, target = split_arrays(data, "train")
    vf, vy = split_arrays(data, "validation")
    scale = state["scale"]
    fields, targets = torch.from_numpy(field), torch.from_numpy(target / scale)
    val_fields, val_targets = torch.from_numpy(vf), torch.from_numpy(vy / scale)
    epsilon, calibration = calibrate_epsilon(data, scale)
    tail, val_tail = TailLoss(field, target, scale, epsilon), TailLoss(vf, vy, scale, epsilon)
    calibrator = ElectrolyteGRU(32)
    calibrator.load_state_dict(state["model"])
    weight, calibration_gradient = gradient_weight(calibrator, fields, targets, tail)
    report = dict(seed=seed, warmup_epochs=warmup_epochs, continuation_epochs=epochs,
                  warmup_seconds=state["warmup_seconds"], epsilon_calibration=calibration,
                  weight_calibration=calibration_gradient, arms={})
    for arm in ["mse", "mixed"]:
        checkpoint_path = output / f"{arm}.pt"
        history_path = output / f"{arm}_history.json"
        if checkpoint_path.exists() and history_path.exists():
            existing = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert existing["continuation_epochs"] == epochs and existing["seed"] == seed
            report["arms"][arm] = json.loads(history_path.read_text())
            continue
        model = ElectrolyteGRU(32)
        model.load_state_dict(state["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = .0003
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        history = []
        batch_order = hashlib.sha256()
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.
            order = rng.permutation(len(fields))
            batch_order.update(order.astype("<i8").tobytes())
            for start in range(0, len(order), 64):
                indices = order[start:start + 64]
                optimizer.zero_grad(set_to_none=True)
                estimate = model(fields[indices])
                mse = torch.mean((estimate - targets[indices])**2)
                loss = mse if arm == "mse" else mse + weight * tail(estimate, indices)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            model.eval()
            with torch.inference_mode():
                estimate = model(val_fields)
                validation = float(torch.mean((estimate - val_targets)**2))
                tail_validation = float(val_tail(estimate))
            history.append(dict(epoch=epoch, training_objective=total/len(fields),
                                validation_mse=validation, validation_tail_asinh=tail_validation))
            write_json(output / f"{arm}_progress.json", history)
            print(f"seed={seed} {arm} {epoch}/{epochs}: val MSE={validation:.7g}; "
                  f"tail={tail_validation:.7g}", flush=True)
        result = dict(history=history, continuation_seconds=time.perf_counter()-started,
                      batch_order_sha256=batch_order.hexdigest(),
                      selection="Predeclared final epoch; no test or diagnostic selection")
        checkpoint = dict(state_dict=model.state_dict(), hidden=32, target_scale=scale.tolist(),
                          dt=DT, initial_state="equilibrium", training_tmax=100., seed=seed,
                          warmup_epochs=warmup_epochs, continuation_epochs=epochs,
                          loss=arm, tail_weight=weight if arm == "mixed" else 0.,
                          epsilon=epsilon.tolist(), selection=result["selection"])
        torch.save(checkpoint, checkpoint_path)
        write_json(history_path, result)
        report["arms"][arm] = result
    assert report["arms"]["mse"]["batch_order_sha256"] == report["arms"]["mixed"]["batch_order_sha256"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/loss-ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--warmup-epochs", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup_epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive epoch budgets and distinct seeds.")
    torch.set_num_threads(2)
    data = load_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                         seeds=args.seeds, warmup_epochs=args.warmup_epochs, epochs=args.epochs,
                         learning_rate=.0003, batch_size=64, waiting_time=8.,
                         epsilon_fraction=.1, gradient_fraction=.25,
                         selection="Fixed final continuation epoch for both arms",
                         platform=platform.platform(), machine=platform.machine(),
                         python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                        for name in ["ablate.py", "train.py", "losses.py", "model.py"]})
    config_path = args.output / "configuration.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == configuration, "Use a new output directory for a changed experiment."
    write_json(config_path, configuration)
    reports = [train_pair(data, seed, args.warmup_epochs, args.epochs, args.output / f"seed-{seed}")
               for seed in args.seeds]
    write_json(args.output / "training.json", dict(configuration=configuration, seeds=reports))


if __name__ == "__main__":
    main()
