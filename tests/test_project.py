"""Independent physical checks and guards against temporal/data leakage."""
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm
import torch

from electrolyte_surrogate.baseline import load_baseline, predict_baseline
from electrolyte_surrogate.data import current_coefficients, load_dataset
from electrolyte_surrogate.generate_data import histories
from electrolyte_surrogate.model import load_gru, predict
from electrolyte_surrogate.losses import TailLoss, calibrate_epsilon, elapsed_since_change, stationary_targets
from electrolyte_surrogate.physics import solve_current, unit_parameters, reference_coefficients
from electrolyte_surrogate.physics import stationary_coefficients
from electrolyte_surrogate.reduction import CovarianceModes
from electrolyte_surrogate.evaluate import metrics
from electrolyte_surrogate.protocols import timing_histories, regime_masks
from electrolyte_surrogate.ablation_results import tail_metrics

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_generator_matches_stored_histories_and_splits(self):
        data = load_dataset(ROOT / "data/dataset.npz")
        self.assertEqual(data["field"].shape, (1088, 2001))
        self.assertEqual(data["tau"][-1], 100.)
        signatures = set()
        for index, (split, family, field) in enumerate(histories()):
            assert_array_equal(field.astype(np.float32), data["field"][index])
            self.assertEqual(split, data["split"][index])
            self.assertEqual(family, data["family"][index])
            signatures.add(data["field"][index].tobytes())
        self.assertEqual(len(signatures), 1088)
        self.assertEqual(sum(data["split"] == "train"), 768)
        self.assertEqual(sum(data["split"] == "validation"), 128)

    def test_late_timing_probes_are_separate_from_the_learning_dataset(self):
        data = load_dataset(ROOT / "data/dataset.npz")
        with np.load(ROOT / "data/timing_probe.npz", allow_pickle=False) as probe:
            expected = list(timing_histories())
            self.assertEqual(len(expected), 32)
            known = {field.tobytes() for field in data["field"]}
            for i, (family, field) in enumerate(expected):
                assert_array_equal(field, probe["field"][i])
                self.assertEqual(family, probe["family"][i])
                self.assertNotIn(field.tobytes(), known)
                known.add(field.tobytes())
                last = np.flatnonzero(np.diff(field) != 0)[-1] + 1
                self.assertGreater(probe["tau"][last], 10.)
            self.assertEqual(probe["tau"][-1], 100.)
            self.assertTrue(np.isfinite(probe["target"]).all())


class PhysicsTests(unittest.TestCase):
    def test_covariance_matches_independent_matrix_exponential(self):
        times = np.array([0., .03, .7])
        result = solve_current(times, lambda t: 1.7, unit_parameters(),
                               n_s=12, n_u=8, piecewise_constant=True)
        for i, j in [(2, 1), (5, 4), (9, 6)]:
            s, u = result["s"][i], result["u"][j]
            A, B, C = 1 + s*s, 1 + 2*s*s, 1 + s*s + 1.7**2*u*u
            stationary = np.array([
                [(B*B + 4*s*s*1.7**2*u*u)/(2*B*C), (1/(2*C) - 1j*1.7*s*u/(B*C))],
                [(1/(2*C) + 1j*1.7*s*u/(B*C)), (B*B + 4*s*s*1.7**2*u*u)/(2*B*C)],
            ])
            equilibrium = np.array([[1 - 1/(2*A), 1/(2*A)], [1/(2*A), 1 - 1/(2*A)]])
            omega = np.array([[s*s + .5 + 1j*1.7*s*u, -.5], [-.5, s*s + .5 - 1j*1.7*s*u]])
            for n, time in enumerate(times):
                propagator = expm(-omega * time)
                expected = stationary + propagator @ (equilibrium - stationary) @ propagator.conj().T
                assert_allclose(result["c"][n, i, j], expected, rtol=1e-12, atol=1e-13)

    def test_stationary_current_agrees_with_paper_and_stays_fixed(self):
        result = solve_current([0., 1., 1000.], lambda t: 3., unit_parameters(), initial_field=3.,
                               n_s=64, n_u=48, piecewise_constant=True)
        assert_allclose(result["el_coefficient"] / 3, -.0021458069059455023, rtol=1e-10)
        assert_allclose(result["gamma_hyd"], -.9357611930641627, rtol=1e-10)
        self.assertTrue(np.all(result["J_third"] == 0))

    def test_switch_off_exponential_and_zero_hydrodynamic_current(self):
        times = np.array([0., .1, 1.])
        result = solve_current(times, lambda t: 0., unit_parameters(), initial_field=3.,
                               n_s=12, n_u=8, piecewise_constant=True)
        s = result["s"][:, None]
        expected = result["c"][0, ..., 0, 1].imag[None, ...] * np.exp(
            -(1 + 2*s*s)[None, ...] * times[:, None, None])
        assert_allclose(result["c"][..., 0, 1].imag, expected, atol=1e-14)
        assert_array_equal(result["J_hyd"], 0.)
        self.assertTrue(np.all(result["J_el"] < 0))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = load_dataset(ROOT / "data/dataset.npz")
        cls.gru, cls.scale, metadata = load_gru(ROOT / "artifacts/gru.pt")
        assert metadata["training_tmax"] == 100
        cls.baseline = load_baseline(ROOT / "artifacts/baseline.npz")
        cls.physical = CovarianceModes.load(ROOT / "artifacts/covariance_weights.npz")

    def runners(self, field):
        return [predict(field, self.gru, self.scale)[0],
                predict_baseline(field, self.baseline),
                self.physical.predict(self.data["tau"], field)[0]]

    def test_parity_equilibrium_and_future_field_independence(self):
        field = self.data["field"][900]
        estimates = self.runners(field)
        reversed_estimates = self.runners(-field)
        zero_estimates = self.runners(np.zeros_like(field))
        changed = field.copy()
        changed[120:] = 7
        altered_estimates = self.runners(changed)
        for actual, reversed_, zero, altered in zip(estimates, reversed_estimates, zero_estimates, altered_estimates):
            assert_allclose(reversed_[:, 0], -actual[:, 0], atol=1e-10)
            assert_allclose(reversed_[:, 1], actual[:, 1], atol=1e-10)
            assert_allclose(zero, 0, atol=1e-10)
            assert_allclose(altered[:121], actual[:121], atol=1e-10)
            current = current_coefficients(field, actual)
            assert_array_equal(current[field == 0, 1], 0.)

    def test_reduced_model_matches_full_solver_for_nonuniform_switches(self):
        tau = np.array([0., .01, .1, .7, 1.1, 2., 5., 20.])
        field = np.array([3., 3., -1.2, -1.2, 0., 2.2, 0., 0.])
        expected = reference_coefficients(tau, field, n_s=96, n_u=64, initial_field=1.6)
        actual, _ = self.physical.predict(tau, field, initial_field=1.6)
        assert_allclose(actual[:, 0], expected[:, 0], atol=1e-6, rtol=0)
        assert_allclose(actual[:, 1], expected[:, 1], atol=1e-5, rtol=0)

    def test_saved_gru_retains_reference_accuracy(self):
        indices = np.flatnonzero(self.data["split"] == "test")[:8]
        field, truth = self.data["field"][indices], self.data["target"][indices]
        estimate = predict(field, self.gru, self.scale)[0]
        score = metrics(truth, estimate, field)
        self.assertLess(score["Q_el_nrmse"], .05)
        self.assertLess(score["hydrodynamic_residual_nrmse"], .05)


class LossTests(unittest.TestCase):
    def test_stationary_centers_match_independent_reference(self):
        fields = np.array([0., .001, -.3, 1.2, 3., -5.])
        expected = np.stack([stationary_coefficients(f) for f in fields])
        assert_allclose(stationary_targets(fields), expected, rtol=1e-10, atol=1e-12)

    def test_tail_loss_masks_prehistory_and_calibration_uses_only_training(self):
        data = load_dataset(ROOT / "data/dataset.npz")
        train = data["split"] == "train"
        scale = np.sqrt(np.mean(data["target"][train, :241]**2, axis=(0, 1))).astype(np.float32)
        epsilon, _ = calibrate_epsilon(data, scale)
        altered = {**data, "target": data["target"].copy()}
        altered["target"][~train] = 1e3
        alternate, _ = calibrate_epsilon(altered, scale)
        assert_array_equal(epsilon, alternate)
        indices = np.flatnonzero(train & (data["family"] == "step"))[:2]
        field, target = data["field"][indices], data["target"][indices]
        loss = TailLoss(field, target, scale, epsilon)
        estimate = torch.from_numpy(target / scale).clone().requires_grad_()
        self.assertEqual(float(loss(estimate)), 0.)
        loss(estimate + .01).backward()
        elapsed, changed = elapsed_since_change(field)
        mask = changed & (elapsed >= 8 - 1e-8)
        self.assertTrue(torch.isfinite(estimate.grad).all())
        assert_array_equal(estimate.grad.numpy()[~mask], 0.)
        self.assertGreater(float(torch.sum(torch.abs(estimate.grad))), 0.)


class EvaluationTests(unittest.TestCase):
    def test_tail_error_does_not_mistake_a_finite_physical_tail_for_bias(self):
        elapsed = np.array([10., 50., 100., 500., 1000.])
        steady = np.tile([-.01, .1], (5, 1))
        truth = steady[:, None, :] + np.ones((5, len(elapsed), 2)) / elapsed[None, :, None]**1.5
        report = tail_metrics(truth, truth.copy(), steady, elapsed)
        assert_array_equal(report["beyond_training"]["relative_rmse"], 0.)
        diagnostic = report["endpoint_diagnostic"]
        assert_array_equal(diagnostic["endpoint_prediction_error"], 0.)
        self.assertTrue(np.all(np.array(diagnostic["reference_endpoint_departure"]) > 0))
        assert_array_equal(diagnostic["late_prediction_range"], diagnostic["late_reference_range"])

    def test_regimes_restart_at_each_change_and_do_not_overlap(self):
        field = np.zeros((1, 2001), dtype=np.float32)
        field[:, 200:1000] = 3
        masks = regime_masks(field)
        self.assertFalse(any(mask[0, :200].any() for mask in masks.values()))
        self.assertTrue(masks["recent_change"][0, 200])
        self.assertTrue(masks["long_hold"][0, 400])
        self.assertTrue(masks["recent_change"][0, 1000])
        self.assertTrue(masks["long_hold_field_off"][0, 1200])
        disjoint = sum(mask.astype(int) for name, mask in masks.items() if name != "long_hold_field_off")
        assert_array_equal(disjoint[0, 200:], 1)

    def test_zero_signal_metrics_remain_valid_json(self):
        zero = np.zeros((10, 2))
        result = metrics(zero, zero, np.zeros(10))
        self.assertIsNone(result["Q_el_nrmse"])
        self.assertIsNone(result["hydrodynamic_residual_nrmse"])
        self.assertEqual(result["Q_el_rmse"], 0.)


if __name__ == "__main__":
    unittest.main()
