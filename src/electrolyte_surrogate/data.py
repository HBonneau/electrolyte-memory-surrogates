"""Load whole-history splits and reconstruct the two current coefficients."""
from pathlib import Path

import numpy as np

DT = 0.05
FIELD_SCALE = 5.0  # Largest field magnitude in the training domain.
SPLITS = ("train", "validation", "test", "waveform_ood", "amplitude_ood")


def load_dataset(path):
    """Return named arrays; one row is a complete field/current history."""
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data["target"].shape != (*data["field"].shape, 2):
        raise ValueError("Targets must have shape (histories, times, 2).")
    if len(data["tau"]) != data["field"].shape[1]:
        raise ValueError("The time grid and history lengths must agree.")
    if not np.isfinite(data["tau"]).all() or not np.allclose(np.diff(data["tau"]), DT):
        raise ValueError("The learning models require uniform dt=0.05.")
    if len(data["tau"]) != 2001 or not np.isclose(data["tau"][0], 0) or not np.isclose(data["tau"][-1], 100):
        raise ValueError("This experiment uses complete histories from 0 to 100 Debye times.")
    if not np.isfinite(data["field"]).all() or not np.isfinite(data["target"]).all():
        raise ValueError("Fields and targets must be finite.")
    return data


def split_arrays(data, name):
    """Select histories without mixing time samples across splits."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split: {name}")
    selected = data["split"] == name
    return data["field"][selected], data["target"][selected]


def current_coefficients(field, latent):
    """Map [Q_el, gamma_hyd+1] to [Q_el, Q_hyd] without dividing by f."""
    return np.stack([latent[..., 0], field * (-1 + latent[..., 1])], axis=-1)


def load_field(path, *, uniform=False):
    """Read a sample-held tau,f CSV; neural inference requires dt=0.05."""
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Provide at least two finite CSV rows with columns tau,f.")
    tau, field = values.T
    if np.any(np.diff(tau) <= 0):
        raise ValueError("Times must be strictly increasing.")
    if uniform and not np.allclose(np.diff(tau), DT, rtol=1e-6, atol=1e-8):
        raise ValueError("The GRU and regression models require dt=0.05.")
    return tau, field
