"""Causal polynomial ridge regression with a validation-selected lag window."""
import numpy as np

from .data import FIELD_SCALE


def lag_features(field, lags, powers):
    """Use strictly previous fields; prepend zero-field equilibrium prehistory."""
    field = np.asarray(field) / FIELD_SCALE
    shifted = np.stack([
        np.pad(field, ((0, 0), (lag, 0)))[:, :field.shape[1]]
        for lag in range(1, lags + 1)
    ], axis=-1)
    return np.concatenate([shifted ** power for power in powers], axis=-1)


def predict_baseline(field, baseline):
    """Return the two latent targets, using the same parity as the GRU."""
    field = np.asarray(field, dtype=np.float32)
    single = field.ndim == 1
    field = np.atleast_2d(field)
    lags = int(baseline["lags"])
    odd = lag_features(field, lags, [1, 3]) @ baseline["odd"]
    even = lag_features(field, lags, [2]) @ baseline["even"]
    latent = np.stack([odd, even], axis=-1) * baseline["target_scale"]
    return latent[0] if single else latent


def load_baseline(path):
    """Load fitted coefficients, lag count, ridge factor, and output scales."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def fit_baseline(field, target, val_field, val_target, scale):
    """Select the lag window and ridge factor on validation histories only."""
    best = None
    for lags in [20, 60, 120]:
        systems = []
        for channel, powers in enumerate([[1, 3], [2]]):
            size = lags * len(powers)
            gram = np.zeros((size, size))
            cross = np.zeros(size)
            for start in range(0, len(field), 24):
                design = lag_features(field[start:start + 24], lags, powers)
                design = design.reshape(-1, size).astype(np.float64)
                response = (target[start:start + 24, :, channel] / scale[channel]).ravel()
                gram += design.T @ design
                cross += design.T @ response
            systems.append((gram, cross))
        for ridge in [1e-5, 1e-3, 1e-1]:
            weights = [
                np.linalg.solve(
                    gram + ridge * np.trace(gram) / len(gram) * np.eye(len(gram)), cross
                )
                for gram, cross in systems
            ]
            candidate = dict(lags=lags, ridge=ridge, odd=weights[0], even=weights[1],
                             target_scale=scale)
            score = float(np.mean(((predict_baseline(val_field, candidate) - val_target) / scale) ** 2))
            if best is None or score < best["validation_mse"]:
                best = {**candidate, "validation_mse": score}
    return best
