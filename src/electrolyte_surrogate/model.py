"""A causal GRU with field parity and equilibrium imposed by construction."""
import numpy as np
import torch
from torch import nn

from .data import FIELD_SCALE, current_coefficients


class ElectrolyteGRU(nn.Module):
    """Predict Q_el and gamma_hyd+1 from previous fields, using shared weights."""

    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        with torch.no_grad():
            self.gru.bias_ih_l0.zero_()
            self.gru.bias_hh_l0.zero_()
            # Start the update gate biased toward retaining the previous state.
            self.gru.bias_ih_l0[hidden:2 * hidden] = 2.5

    def forward(self, field):
        # The covariance is continuous at a field jump: the latent output at
        # time n depends on f through n-1. Current f_n enters reconstruction.
        previous = torch.cat([torch.zeros_like(field[:, :1]), field[:, :-1]], dim=1) / FIELD_SCALE

        def features(values):
            return torch.stack([values, values * values], dim=-1)

        batch = len(field)
        inputs = torch.cat([
            features(previous),
            features(-previous),
            features(torch.zeros_like(previous)),
        ], dim=0)
        state, _ = self.gru(inputs)
        raw = self.head(state)
        positive, negative, zero = raw[:batch], raw[batch:2 * batch], raw[2 * batch:]
        electrostatic = (positive[..., 0] - negative[..., 0]) / 2
        hydrodynamic = (positive[..., 1] + negative[..., 1]) / 2 - zero[..., 1]
        return torch.stack([electrostatic, hydrodynamic], dim=-1)


def load_gru(path):
    """Load tensors, architecture, output scales and sampling metadata together."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ElectrolyteGRU(checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Float32 matches the original training-time evaluation and epoch ablation.
    scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    return model, scale, checkpoint


def predict(field, model, scale):
    """Return latent targets and currents for one history or a history batch."""
    field = np.asarray(field, dtype=np.float32)
    single = field.ndim == 1
    field = np.atleast_2d(field)
    if field.ndim != 2 or field.shape[1] < 1 or not np.isfinite(field).all():
        raise ValueError("Fields must be finite arrays of shape (time,) or (batch,time).")
    with torch.inference_mode():
        latent = model(torch.from_numpy(field)).numpy() * scale
    current = current_coefficients(field, latent)
    return (latent[0], current[0]) if single else (latent, current)
