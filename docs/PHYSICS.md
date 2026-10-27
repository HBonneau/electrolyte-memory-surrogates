# Physical conventions

The reference is a three-dimensional symmetric binary point-ion electrolyte with charges +/-q, equal density per species, equal mechanical mobility, and a Gaussian covariance closure. It follows the signs and scales in the [2025 paper](https://doi.org/10.1088/1742-5468/adb4ce) and the current corresponding to equation 17 of the [2023 paper](https://doi.org/10.1088/1742-5468/acdced).

## Units and targets

With theta = k_B T, density rho per species, absolute permittivity epsilon, mobility kappa, and viscosity eta:

```text
lambda_D = sqrt(theta epsilon / (2 q² rho))
t_D      = lambda_D² / (kappa theta)
E_D      = theta / (q lambda_D)
sigma_0  = 2 q² kappa rho
J_D      = sigma_0 E_D
r_s      = 1 / (6 pi eta kappa)
```

The sampled inputs use tau = t/t_D and f = E/E_D. Targets are `Q_el` and `Y_hyd = gamma_hyd+1`. Reconstruct

$$
Q_{hyd}=f(-1+Y_{hyd}),\qquad
\frac{J}{J_D}=f+\frac{Q_{el}}{\rho\lambda_D^3}+\frac{r_s}{\lambda_D}Q_{hyd}.
$$

The Gaussian three-point current is zero. The hydrodynamic self-correlation is removed. No current calculation divides by the instantaneous field. Electrostatic current can persist at f=0; hydrodynamic current is then exactly zero through its known field factor.

## Memory states

At each wavevector define s = k lambda_D, u = k_x/k, B = 1+2s², and a_eq = 1/[2(1+s²)]. Three real covariance deviations obey

```text
p' = -B p + a
a' = p - B a + 2 f s u b
b' = -B b - 2 f s u (a_eq + a)
```

Their equilibrium initial values are zero. Current coefficients are weighted wavevector integrals of `s*u*b/(8*pi²)` and `3*(1-u²)*(p-a)/(2*pi)`. Smooth prescribed fields use BDF integration. Declared held fields use exact propagation between switches; finite quadrature is still an approximation to the wavevector integral.

The reduced model uses the same equations at fewer nodes and learns nonnegative integration weights shared by both currents and all fields. The stationary constraints apply only at its training amplitudes; held-out inference does not add an analytic stationary correction.

## API

`physics.solve_current(times, E, parameters, ...)` accepts seconds, a callable field in V/m, and explicit SI material parameters; it returns separate dimensional currents. `physics.reference_coefficients(tau, field)` returns dimensionless latent targets for a sampled held field. Its synthetic unit parameters specify no real material.

The GRU and regression start at zero-field equilibrium and require dt=0.05. Their learning histories span 100 Debye times; the physical reduction uses on/off fitting curves through time 1,000. The physical API supports nonuniform times and a stationary nonzero prehistory through `initial_field`. All tested models retain history across field changes.

The physical reduced grid eventually decays exponentially because it has finitely many modes. Its demonstrated approximation to the continuum algebraic behavior ends at 1,000 Debye times.

## Loss intervention

The paired neural experiment centers an asinh penalty on the known stationary coefficients and applies it after eight times without a field change. Stationary values enter this training objective only. Predictions retain the original recurrent state and current reconstruction; no stationary correction or universal algebraic envelope is applied at inference. The switch-off response therefore remains an independent diagnostic with its distinct exponential factor.
