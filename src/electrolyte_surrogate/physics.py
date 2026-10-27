"""Binary symmetric point-ion electrolyte, initially in a specified steady state.

Reference: Bonneau et al., J. Stat. Mech. (2025) 033201, Eqs. (10)-(12),
(36)-(38); equivalent to Eqs. (23) and (17) of the 2023 paper.
Requires NumPy and SciPy. Physical inputs and outputs use SI units.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import Boltzmann, elementary_charge
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags


@dataclass(frozen=True)
class Parameters:
    temperature: float       # K
    density: float           # number density PER SPECIES, m^-3
    permittivity: float      # absolute permittivity, F/m (not relative)
    mobility: float          # mechanical mobility kappa, m/(N s); D = kappa k_B T
    viscosity: float         # Pa s
    charge: float = elementary_charge  # magnitude of each ion's charge, C

    def scales(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("All physical parameters must be finite and positive.")
        thermal = Boltzmann * self.temperature
        length = np.sqrt(thermal * self.permittivity
                         / (2 * self.charge**2 * self.density))
        time = length**2 / (self.mobility * thermal)
        field = thermal / (self.charge * length)
        sigma0 = 2 * self.charge**2 * self.mobility * self.density
        radius = 1 / (6 * np.pi * self.viscosity * self.mobility)
        return dict(lambda_D=length, t_D=time, E_D=field,
                    sigma0=sigma0, r_s=radius, J_D=sigma0 * field)


def solve_current(times, E, parameters, *, switches=(), initial_field=0.0,
                  piecewise_constant=False, return_covariance=True,
                  n_s=32, n_u=16, rtol=1e-7, atol=1e-9):
    """Solve the covariance equation and return the longitudinal current density.

    times: strictly increasing observation times in seconds, at least two.
    E(t): scalar, right-continuous electric field along x in V/m, called at
          arbitrary physical times, not just the observation times.
    parameters: Parameters instance; equal mobilities and charges +/-q.
    switches: ALL field discontinuities strictly inside the time interval,
              in seconds. The ODE integration restarts at these times.
    initial_field: prehistory field in V/m. The initial covariance is its
                   stationary state (zero gives equilibrium, the default).
    piecewise_constant: declare that E is constant between switches. Uses
                        the exact constant-coefficient covariance propagator.
    return_covariance: set False to omit the returned 2x2 covariance array.
    n_s, n_u: Gauss-Legendre orders for s=k*lambda_D and u=k_x/k.
              The radial map s=x/(1-x) covers [0, infinity); no UV cutoff.
    rtol, atol: BDF tolerances; atol is scaled per mode for the large-s tail.

    The initial state at times[0] is the steady state BEFORE any field jump.
    Returns c=C_tilde/density, shape (time, s, u, 2, 2), including self
    correlations; the species order is (+,-). Currents have units A/m^2.
    Also returns field=f, el_coefficient=Q_el, and gamma_hyd, so dimensionless
    surrogate targets remain defined at E=0 without dividing by the field.
    Quadrature orders and tolerances must be checked for each driving history.
    """
    t = np.asarray(times, dtype=float)
    jumps = np.asarray(switches, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("times must be finite and strictly increasing.")
    if (jumps.ndim != 1 or not np.all(np.isfinite(jumps))
            or np.any(jumps <= t[0]) or np.any(jumps >= t[-1])):
        raise ValueError("switches must lie strictly inside the time interval.")
    if (not isinstance(n_s, int) or not isinstance(n_u, int)
            or n_s < 2 or n_u < 2 or rtol <= 0 or atol <= 0):
        raise ValueError("Use quadrature orders >=2 and positive tolerances.")
    scales = parameters.scales()
    tau = (t - t[0]) / scales["t_D"]
    xs, ws = leggauss(n_s)
    x, ws = (xs + 1) / 2, ws / 2
    s, ws = x / (1 - x), ws / (1 - x)**2
    u, wu = leggauss(n_u)
    S, U = np.meshgrid(s, u, indexing="ij")
    weights = ws[:, None] * wu[None, :]
    s2, su = S.ravel()**2, (S * U).ravel()
    B = 2 * s2 + 1
    a_eq = 1 / (2 * (1 + s2))
    count = len(s2)

    # c++=c--=1-a_eq+p; c+-=a_eq+a+i*b. Evolve deviations
    # (p,a,b) from equilibrium, avoiding cancellation of large-s self terms.
    # Eq. (12): c' = 2*s^2*I - omega*c - c*omega.conj().
    # omega = [[s^2+1/2+i*f*s*u, -1/2],
    #          [-1/2, s^2+1/2-i*f*s*u]], f(tau)=E(t)/E_D.
    def field(physical_time):
        value = float(E(float(physical_time)))
        if not np.isfinite(value):
            raise ValueError("E(t) must return a finite scalar.")
        return value / scales["E_D"]

    if not np.isfinite(initial_field):
        raise ValueError("initial_field must be finite.")

    def stationary(f):
        A = 1 + s2
        fu2 = f*f * U.ravel()**2
        C = A + fu2
        return np.concatenate((-fu2/(2*A*B*C), -fu2/(2*A*C),
                               -f*su/(B*C)))

    state = stationary(float(initial_field) / scales["E_D"])
    history = np.empty((3 * count, len(t)))
    # Pair deviations decay as s^-4; Im(c+-) decays as s^-3.
    tol = atol * np.concatenate(((1 + S.ravel())**-4,
                                (1 + S.ravel())**-4,
                                (1 + S.ravel())**-3))
    edges = np.r_[t[0], np.unique(jumps), t[-1]]
    minus_B, identity = diags(-B), diags(np.ones(count))
    for left, right in zip(edges[:-1], edges[1:]):
        start, stop = (np.array([left, right]) - t[0]) / scales["t_D"]

        selected = np.flatnonzero((t >= left) & (t <= right))
        evaluation = np.unique(np.r_[tau[selected], stop])
        if piecewise_constant:
            f = field(left)
            dt = evaluation - start
            p0, a0, b0 = state.reshape(3, count)
            if f == 0:
                # Explicitly preserve the exponentially small imaginary part.
                slow = (p0+a0)[:, None]/2 * np.exp(-2*s2[:, None]*dt)
                fast = (p0-a0)[:, None]/2 * np.exp(-(2*s2+2)[:, None]*dt)
                propagated = np.concatenate((slow+fast, slow-fast,
                                  b0[:, None]*np.exp(-B[:, None]*dt)), axis=0)
            else:
                ps, ast, bs = stationary(f).reshape(3, count)
                Dd, Da, Db = (p0-ps)[:, None], (a0-ast)[:, None], (b0-bs)[:, None]
                g = f*su
                base = s2+.5
                delta = .25-g*g
                h, v = np.empty((count, len(dt))), np.empty((count, len(dt)))
                positive, negative = delta > 0, delta < 0
                zero = ~(positive | negative)
                # Combine exponential factors before evaluation, avoiding
                # cosh overflow for times much longer than the Debye time.
                r = np.sqrt(delta[positive])[:, None]
                ep = np.exp(-(base[positive, None]-r)*dt)
                em = np.exp(-(base[positive, None]+r)*dt)
                h[positive], v[positive] = (ep+em)/2, (ep-em)/(2*r)
                r = np.sqrt(-delta[negative])[:, None]
                decay = np.exp(-base[negative, None]*dt)
                h[negative], v[negative] = decay*np.cos(r*dt), decay*np.sin(r*dt)/r
                decay = np.exp(-base[zero, None]*dt)
                h[zero], v[zero] = decay, decay*dt
                G, L = g[:, None]*v, v/2
                # det(exp(-omega*dt))=exp(-B*dt), evaluated directly to
                # avoid subtracting almost equal h^2 and L^2.
                determinant = np.exp(-B[:, None]*dt)
                propagated = np.concatenate((
                    ps[:, None]+(h*h+G*G+L*L)*Dd+2*L*(h*Da+G*Db),
                    ast[:, None]+2*L*h*Dd+(h*h-G*G+L*L)*Da+2*h*G*Db,
                    bs[:, None]-2*L*G*Dd+(determinant-2*G*G)*Db-2*h*G*Da), axis=0)
                # Keep the initial covariance exactly continuous at a jump.
                if np.any(dt == 0):
                    propagated[:, dt == 0] = state[:, None]
            history[:, selected] = propagated[:, np.searchsorted(evaluation, tau[selected])]
            state = propagated[:, -1]
            continue

        def coupling(time):
            # At the endpoint use the left limit; the next segment uses E(right).
            physical = t[0] + time * scales["t_D"]
            physical = np.clip(physical, left, np.nextafter(right, left))
            return 2 * field(physical) * su

        def rhs(time, y):
            p, a, b = y.reshape(3, count)
            g = coupling(time)
            return np.concatenate((-B*p + a, p - B*a + g*b,
                                   -B*b - g*(a_eq + a)))

        def jac(time, y):
            g = diags(coupling(time))
            return bmat([[minus_B, identity, None],
                         [identity, minus_B, g],
                         [None, -g, minus_B]], format="csc")

        solution = solve_ivp(rhs, (start, stop), state, method="BDF",
                             jac=jac, t_eval=evaluation, rtol=rtol, atol=tol)
        if not solution.success:
            raise RuntimeError(solution.message)
        history[:, selected] = solution.y[:, np.searchsorted(evaluation, tau[selected])]
        state = solution.y[:, -1]

    p, a, b = history.reshape(3, n_s, n_u, len(t))
    integrate = lambda values: np.einsum("su,sut->t", weights, values)
    # Azimuthal integration contributes 2*pi. No division by E(t), even at E=0.
    el = integrate(S[:, :, None] * U[:, :, None] * b) / (8 * np.pi**2)
    # The equilibrium hydrodynamic integral is exactly -1. Subtract it
    # analytically; p-a is the remaining pair-correlation correction.
    hyd = -1 + 3/(2*np.pi) * integrate((1 - U[:, :, None]**2) * (p - a))
    f = np.array([field(time) for time in t])
    J_bare = scales["J_D"] * f
    J_el = scales["J_D"] / (parameters.density * scales["lambda_D"]**3) * el
    J_hyd = scales["J_D"] * scales["r_s"] / scales["lambda_D"] * f * hyd
    covariance = None
    if return_covariance:
        covariance = np.empty((len(t), n_s, n_u, 2, 2), dtype=complex)
        diagonal = (1 - a_eq.reshape(n_s, n_u)[:, :, None] + p).transpose(2, 0, 1)
        off_diagonal = (a_eq.reshape(n_s, n_u)[:, :, None] + a + 1j*b).transpose(2, 0, 1)
        covariance[..., 0, 0] = covariance[..., 1, 1] = diagonal
        covariance[..., 0, 1] = off_diagonal
        covariance[..., 1, 0] = off_diagonal.conj()
    return dict(t=t, tau=tau, s=s, u=u, c=covariance, scales=scales,
                field=f, el_coefficient=el, gamma_hyd=hyd,
                J_bare=J_bare, J_el=J_el, J_hyd=J_hyd,
                J_third=np.zeros_like(t), J_total=J_bare + J_el + J_hyd)


def unit_parameters():
    """Synthetic parameters giving unit Debye scales, not material estimates."""
    return Parameters(1 / Boltzmann, 1, 2, 1, 1 / (6 * np.pi), charge=1)


def reference_coefficients(tau, field, *, n_s=48, n_u=48, initial_field=0):
    """Return [Q_el, gamma_hyd+1] for a sampled, piecewise-constant field."""
    tau, field = np.asarray(tau, dtype=float), np.asarray(field, dtype=float)
    if tau.shape != field.shape or tau.ndim != 1 or not np.isfinite(field).all():
        raise ValueError("Supply matching one-dimensional time and finite field arrays.")
    changes = np.flatnonzero(np.diff(field) != 0) + 1
    switches = tau[changes[changes < len(tau) - 1]]

    def applied(time):
        index = np.searchsorted(tau, time, side="right") - 1
        return float(field[np.clip(index, 0, len(field) - 1)])

    result = solve_current(
        tau, applied, unit_parameters(), switches=switches,
        initial_field=initial_field, piecewise_constant=True,
        return_covariance=False, n_s=n_s, n_u=n_u,
    )
    return np.column_stack([result["el_coefficient"], result["gamma_hyd"] + 1])


def stationary_coefficients(field, n_s=128, n_u=96):
    """Integrate the exact stationary covariance without small-field divisions."""
    nodes, radial_weights = leggauss(n_s)
    x = (nodes + 1) / 2
    s = x / (1 - x)
    radial_weights = radial_weights / (2 * (1 - x) ** 2)
    u, angular_weights = leggauss(n_u)
    s, u = np.meshgrid(s, u, indexing="ij")
    weights = radial_weights[:, None] * angular_weights
    A, B = 1 + s * s, 1 + 2 * s * s
    v = field * field * u * u
    C = A + v
    b = -field * s * u / (B * C)
    p_minus_a = v * s * s / (A * B * C)
    return np.array([
        np.sum(weights * s * u * b) / (8 * np.pi ** 2),
        3 / (2 * np.pi) * np.sum(weights * (1 - u * u) * p_minus_a),
    ])
