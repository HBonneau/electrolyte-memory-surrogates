"""Current errors, elapsed-time regimes, and whole-history evaluation."""
import numpy as np

from .data import split_arrays
from .protocols import regime_masks


def metrics(truth, estimate, field):
    """NRMSE of interactions, excluding the known hydrodynamic baseline -f."""
    truth, estimate, field = [np.asarray(x, dtype=np.float64) for x in (truth, estimate, field)]
    actual = np.stack([truth[..., 0], field * truth[..., 1]], axis=-1)
    predicted = np.stack([estimate[..., 0], field * estimate[..., 1]], axis=-1)
    axes = tuple(range(actual.ndim - 1))
    rmse = np.sqrt(np.mean((predicted - actual)**2, axis=axes))
    rms = np.sqrt(np.mean(actual**2, axis=axes))
    return dict(Q_el_nrmse=float(rmse[0] / rms[0]) if rms[0] > 0 else None,
                hydrodynamic_residual_nrmse=float(rmse[1] / rms[1]) if rms[1] > 0 else None,
                Q_el_rmse=float(rmse[0]), hydrodynamic_residual_rmse=float(rmse[1]))


def regime_metrics(truth, estimate, field):
    """Score time since the latest change; retain RMSE when reference RMS is zero."""
    report = {}
    for name, selected in regime_masks(field).items():
        if np.ndim(field) == 1:
            selected = selected[0]
        report[name] = dict(samples=int(selected.sum()))
        if selected.any():
            report[name].update(metrics(truth[selected], estimate[selected], field[selected]))
    return report


def ordinary_tests(data, predictors):
    results, prefix, regimes = {}, {}, {}
    for name in ["test", "waveform_ood", "amplitude_ood"]:
        field, truth = split_arrays(data, name)
        results[name], prefix[name], regimes[name] = {}, {}, {}
        for key, run in predictors.items():
            estimate = run(data["tau"], field)
            results[name][key] = metrics(truth, estimate, field)
            prefix[name][key] = metrics(truth[:, :241], estimate[:, :241], field[:, :241])
            regimes[name][key] = regime_metrics(truth, estimate, field)
    return results, prefix, regimes


def timing_tests(data, predictors):
    """Fixed held-out protocols with late changes, grouped by protocol family."""
    field, truth = data["field"], data["target"]
    report = {}
    for name, run in predictors.items():
        estimate = run(data["tau"], field)
        families = {}
        for family in np.unique(data["family"]):
            selected = data["family"] == family
            families[str(family)] = metrics(truth[selected], estimate[selected], field[selected])
        report[name] = dict(overall=metrics(truth, estimate, field),
                            regimes=regime_metrics(truth, estimate, field), families=families)
    return report
