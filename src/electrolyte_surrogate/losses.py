"""Training-only calibration for a stationary-centered, late-time asinh loss."""
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

from .data import DT


def stationary_targets(field, order=96):
    """Integrate stationary targets in batches, including zero and weak fields."""
    unique, inverse = np.unique(np.asarray(field), return_inverse=True)
    x, wx = leggauss(order)
    x = (x + 1) / 2
    s, ws = x / (1 - x), wx / (2 * (1 - x)**2)
    u, wu = leggauss(order)
    s2, u2 = s[None, :, None]**2, u[None, None, :]**2
    A, B = 1 + s2, 1 + 2 * s2
    weights = ws[None, :, None] * wu[None, None, :]
    targets = np.empty((len(unique), 2))
    for start in range(0, len(unique), 64):
        f = unique[start:start + 64].astype(float)[:, None, None]
        v = f*f*u2
        C = A + v
        targets[start:start + len(f), 0] = -f[:, 0, 0] * np.sum(
            weights * s2*u2 / (B*C), axis=(1, 2)) / (8*np.pi**2)
        targets[start:start + len(f), 1] = 3/(2*np.pi) * np.sum(
            weights * (1-u2)*v*s2 / (A*B*C), axis=(1, 2))
    return targets[inverse].reshape((*np.shape(field), 2))


def elapsed_since_change(field):
    """Return elapsed times and a mask excluding initial equilibrium prehistory."""
    field = np.atleast_2d(field)
    change = np.zeros_like(field, dtype=bool)
    change[:, 1:] = np.diff(field, axis=1) != 0
    indices = np.arange(field.shape[1])[None, :]
    last = np.maximum.accumulate(np.where(change, indices, -1), axis=1)
    return (indices - last) * DT, last >= 0


def calibrate_epsilon(data, scale, waiting_time=8.):
    """Set epsilon to 10% of median training-step departures at the given time."""
    selected = (data["split"] == "train") & (data["family"] == "step")
    departures = []
    for field, target in zip(data["field"][selected], data["target"][selected]):
        changes = np.flatnonzero(np.diff(field) != 0) + 1
        index = changes[0] + int(round(waiting_time / DT))
        departures.append(np.abs(target[index].astype(float) - stationary_targets(field[index])))
    typical = np.median(departures, axis=0)
    epsilon = np.maximum(.1 * typical, 10 * np.finfo(np.float32).eps * scale)
    return epsilon, dict(waiting_time=waiting_time, fraction=.1,
                         calibration_split="train", calibration_family="step",
                         histories=int(selected.sum()), typical_departure=typical.tolist(),
                         epsilon=epsilon.tolist())


class TailLoss:
    """Squared asinh differences, centered on known stationary coefficients.

    Stationary values enter the training loss only. They are never supplied
    as neural inputs or used to correct predictions at inference.
    """

    def __init__(self, field, target, scale, epsilon, waiting_time=8.):
        self.center = torch.from_numpy((stationary_targets(field) / scale).astype(np.float32))
        self.epsilon = torch.from_numpy((epsilon / scale).astype(np.float32))
        self.transformed = torch.asinh((torch.from_numpy(target / scale) - self.center) / self.epsilon)
        elapsed, changed = elapsed_since_change(field)
        self.mask = torch.from_numpy(changed & (elapsed >= waiting_time - 1e-8))

    def __call__(self, estimate, indices=None):
        center = self.center if indices is None else self.center[indices]
        transformed = self.transformed if indices is None else self.transformed[indices]
        mask = self.mask if indices is None else self.mask[indices]
        difference = torch.asinh((estimate - center) / self.epsilon) - transformed
        if not torch.any(mask):
            return estimate.sum() * 0
        return torch.mean(difference[mask]**2)


def gradient_weight(model, fields, targets, tail, fraction=.25):
    """Fix the tail weight from gradient norms of the first 64 training records."""
    indices = np.arange(min(64, len(fields)))
    estimate = model(fields[indices])
    mse = torch.mean((estimate - targets[indices])**2)
    parameters = tuple(model.parameters())
    gm = torch.autograd.grad(mse, parameters, retain_graph=True)
    gt = torch.autograd.grad(tail(estimate, indices), parameters)
    norm = lambda gradients: float(torch.sqrt(sum(torch.sum(g*g) for g in gradients)))
    mse_norm, tail_norm = norm(gm), norm(gt)
    weight = fraction * mse_norm / max(tail_norm, 1e-20)
    return weight, dict(gradient_fraction=fraction, calibration_histories=len(indices),
                        mse_gradient_norm=mse_norm, tail_gradient_norm=tail_norm,
                        tail_weight=weight)
