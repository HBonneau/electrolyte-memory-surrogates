"""Physical covariance modes with learned positive integration weights.

The dynamics are supplied by the electrolyte theory. Only shared quadrature
weights are fitted; no stationary correction is added during prediction.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import svd
from scipy.optimize import Bounds, LinearConstraint, minimize

from .data import current_coefficients

TRAIN_FIELDS = [.3, .5, .8, 1., 1.4, 1.8, 2.4, 3.2, 4., 4.6, 5.]
VALIDATION_FIELDS = [.65, 1.6, 2.8, 4.3]
TEST_FIELDS = [.4, 1.2, 2.2, 3., 4.8]
FIT_TIMES = np.r_[0., np.geomspace(.001, 1000., 121)]


def mode_grid(radial=24, angular=12):
    """Map radial nodes to infinity; exploit positive-u angular symmetry."""
    x, radial_weights = leggauss(radial)
    x = (x + 1) / 2
    s = x / (1 - x)
    radial_weights = radial_weights / (2 * (1 - x) ** 2)
    x, angular_weights = leggauss(angular)
    u = (x + 1) / 2
    # The half-interval Jacobian and the factor two from symmetry cancel.
    s, u = np.meshgrid(s, u, indexing="ij")
    weights = radial_weights[:, None] * angular_weights
    return s.ravel(), u.ravel(), weights.ravel()


class CovarianceModes:
    """Evolve three physical covariance variables per retained wavevector."""

    def __init__(self, s, u, weights):
        self.s, self.u, self.weights = [np.asarray(x, dtype=np.float64) for x in (s, u, weights)]
        if not (self.s.shape == self.u.shape == self.weights.shape and self.s.ndim == 1):
            raise ValueError("Mode coordinates and weights must be matching one-dimensional arrays.")
        if not all(np.isfinite(x).all() for x in (self.s, self.u, self.weights)):
            raise ValueError("Mode coordinates and weights must be finite.")
        if np.any(self.weights < 0):
            raise ValueError("Integration weights must be nonnegative.")
        self.s2 = self.s ** 2
        self.su = self.s * self.u
        self.B = 1 + 2 * self.s2

    @classmethod
    def gauss(cls, radial=24, angular=12):
        return cls(*mode_grid(radial, angular))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive["s"], archive["u"], archive["weights"])

    def save(self, path):
        np.savez_compressed(path, s=self.s, u=self.u, weights=self.weights)

    def stationary(self, field):
        A = 1 + self.s2
        v = field * field * self.u ** 2
        C = A + v
        return np.stack([-v / (2 * A * self.B * C), -v / (2 * A * C),
                         -field * self.su / (self.B * C)])

    def readout(self, state):
        """Return the two unweighted current features for each mode."""
        p, a, b = state
        s_u = self.su if b.ndim == 1 else self.su[:, None]
        angular = 1 - self.u ** 2
        if b.ndim > 1:
            angular = angular[:, None]
        return np.stack([s_u * b / (8 * np.pi ** 2),
                         3 / (2 * np.pi) * angular * (p - a)], axis=-1)

    def propagate_deviation(self, deviation, field, elapsed):
        """Exact held-field covariance propagator, including very slow modes."""
        dt = np.asarray(elapsed)[None, :]
        dp, da, db = np.asarray(deviation)[:, :, None]
        if field == 0:
            slow = (dp + da) / 2 * np.exp(-2 * self.s2[:, None] * dt)
            fast = (dp - da) / 2 * np.exp(-(self.B + 1)[:, None] * dt)
            return np.stack([slow + fast, slow - fast, db * np.exp(-self.B[:, None] * dt)])

        g = field * self.su
        base = self.s2 + .5
        delta = .25 - g * g
        positive, negative = delta > 0, delta < 0
        zero = ~(positive | negative)
        h = np.empty((len(g), dt.shape[1]))
        v = np.empty_like(h)
        # Combine decay factors before evaluation to avoid cosh overflow.
        r = np.sqrt(delta[positive])[:, None]
        ep = np.exp(-(base[positive, None] - r) * dt)
        em = np.exp(-(base[positive, None] + r) * dt)
        h[positive], v[positive] = (ep + em) / 2, (ep - em) / (2 * r)
        r = np.sqrt(-delta[negative])[:, None]
        decay = np.exp(-base[negative, None] * dt)
        h[negative], v[negative] = decay * np.cos(r * dt), decay * np.sin(r * dt) / r
        decay = np.exp(-base[zero, None] * dt)
        h[zero], v[zero] = decay, decay * dt

        G, L = g[:, None] * v, v / 2
        determinant = np.exp(-self.B[:, None] * dt)
        next_p = (h*h + G*G + L*L)*dp + 2*L*(h*da + G*db)
        next_a = 2*L*h*dp + (h*h - G*G + L*L)*da + 2*h*G*db
        next_b = -2*L*G*dp - 2*h*G*da + (determinant - 2*G*G)*db
        return np.stack([next_p, next_a, next_b])

    def step_features(self, field, elapsed):
        """Stationary, on-transient, and exponentially scaled off features."""
        steady = self.stationary(field)
        delta = self.propagate_deviation(-steady, field, elapsed)
        b_scaled = steady[2, :, None] * np.exp(-2 * self.s2[:, None] * np.asarray(elapsed)[None, :])
        off_scaled = self.su[:, None] * b_scaled / (8 * np.pi ** 2)
        return self.readout(steady), self.readout(delta), off_scaled

    def step(self, field, elapsed):
        stat, delta, off = self.step_features(field, elapsed)
        stationary = self.weights @ stat
        departure = np.einsum("k,ktc->tc", self.weights, delta)
        return dict(stationary=stationary, departure=departure,
                    on=stationary + departure, off_scaled=self.weights @ off,
                    off=np.exp(-np.asarray(elapsed)) * (self.weights @ off))

    def predict(self, tau, field, initial_field=0):
        """Retain covariance history across sampled field changes."""
        tau, field = np.asarray(tau, dtype=float), np.asarray(field, dtype=float)
        if (tau.shape != field.shape or tau.ndim != 1 or len(tau) < 2
                or not np.isfinite(tau).all() or not np.isfinite(field).all()
                or not np.isfinite(initial_field) or np.any(np.diff(tau) <= 0)):
            raise ValueError("Provide finite field/time arrays with increasing times.")
        state = self.stationary(initial_field)
        latent = np.empty((len(tau), 2))
        changes = np.flatnonzero(np.diff(field) != 0) + 1
        edges = np.r_[0, changes, len(tau) - 1]
        for left, right in zip(edges[:-1], edges[1:]):
            value = float(field[left])
            steady = self.stationary(value)
            states = steady[:, :, None] + self.propagate_deviation(
                state - steady, value, tau[left:right + 1] - tau[left]
            )
            states[:, :, 0] = state
            latent[left:right + 1] = np.einsum("k,ktc->tc", self.weights, self.readout(states))
            state = states[:, :, -1]
        return latent, current_coefficients(field, latent)


def fit_weights(radial=24, angular=12, ridge=1e-5):
    """Fit positive weights to training fields; return separate validation scores.

    Fine-grid targets cover 0.001 to 1000 Debye times. These targets and the
    supplied dynamics differ from the short-history GRU training problem.
    """
    model = CovarianceModes.gauss(radial, angular)
    teacher = CovarianceModes.gauss(96, 48)
    matrices, targets, steady_matrices, steady_targets = [], [], [], []

    def scale(values):
        return np.maximum(np.abs(values), max(1e-12, 1e-8 * np.max(np.abs(values))))

    for field in TRAIN_FIELDS:
        stat, delta, off = model.step_features(field, FIT_TIMES)
        truth = teacher.step(field, FIT_TIMES)
        for channel in range(2):
            denominator = scale(truth["departure"][:, channel])
            matrices.append(delta[:, :, channel].T * model.weights / denominator[:, None])
            targets.append(truth["departure"][:, channel] / denominator)
            normal = abs(truth["stationary"][channel])
            steady_matrices.append(stat[:, channel] * model.weights / normal)
            steady_targets.append(truth["stationary"][channel] / normal)
        denominator = scale(truth["off_scaled"])
        matrices.append(off.T * model.weights / denominator[:, None])
        targets.append(truth["off_scaled"] / denominator)

    design, target = np.vstack(matrices), np.concatenate(targets)
    A, b = np.asarray(steady_matrices), np.asarray(steady_targets)
    # Rank-revealed stationary equalities avoid an arbitrary large penalty.
    U, singular, V = svd(A, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    constraints = V[:rank]
    goal = (U[:, :rank].T @ b) / singular[:rank]
    start = np.ones(len(model.s))
    start += constraints.T @ (goal - constraints @ start)

    def objective(weights):
        residual = design @ weights - target
        return .5 * np.mean(residual**2) + .5 * ridge * np.mean((weights - 1)**2)

    def gradient(weights):
        return (design.T @ (design @ weights - target) / len(target)
                + ridge * (weights - 1) / len(weights))

    fitted = minimize(
        objective, start, jac=gradient, method="SLSQP", bounds=Bounds(0, np.inf),
        constraints=[LinearConstraint(constraints, goal, goal)],
        options=dict(maxiter=1000, ftol=1e-13),
    )
    if not fitted.success:
        raise RuntimeError(f"Weight fitting failed: {fitted.message}")
    model.weights *= fitted.x
    validation = {}
    late = FIT_TIMES >= 10
    for field in VALIDATION_FIELDS:
        truth, estimate = teacher.step(field, FIT_TIMES), model.step(field, FIT_TIMES)
        error = estimate["on"] - truth["on"]
        relative = error[late] / np.maximum(np.abs(truth["departure"][late]), 1e-12)
        validation[str(field)] = np.sqrt(np.mean(relative**2, axis=0)).tolist()
    return model, dict(radial=radial, angular=angular, ridge=ridge,
                       iterations=int(fitted.nit), objective=float(fitted.fun),
                       validation_on_tail_relative_rmse=validation,
                       stationary_constraint_residual=float(np.max(np.abs(constraints @ fitted.x - goal))))
