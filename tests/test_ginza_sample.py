"""传播路径、beamformed PDP 和链路预算的纯 CPU 数值测试。"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mMIMO_sleep.data.ginza_sample import (
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT_M_PER_S,
    NoValidPathsError,
    compute_beamformed_path_powers,
    compute_delay_statistics,
    compute_link_budget,
    extract_path_arrays,
    infer_los_from_delay,
)


class GinzaSampleMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coefficients = np.zeros((1, 2, 1, 2, 3), dtype=np.complex128)
        self.coefficients[0, 0, 0, 0, 0] = 1.0 + 0.0j
        self.coefficients[0, 0, 0, 0, 1] = 2.0 + 0.0j
        self.coefficients[0, 1, 0, 0, 2] = 3.0 + 0.0j
        self.delays = np.array([1e-6, 2e-6, 3e-6])
        self.paths = SimpleNamespace(
            a=(self.coefficients,),
            tau=self.delays.reshape(1, 1, 3),
            valid=np.array([True, True, True]).reshape(1, 1, 3),
        )

    def test_extract_preserves_single_path_axis(self) -> None:
        paths = SimpleNamespace(
            a=(self.coefficients[..., :1],),
            tau=np.array([[[1e-6]]]),
            valid=np.array([[[True]]]),
        )
        coefficients, delays, valid = extract_path_arrays(
            paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        self.assertEqual(coefficients.shape, (2, 2, 1))
        self.assertEqual(delays.shape, (1,))
        self.assertEqual(valid.shape, (1,))

    def test_beamformed_path_powers(self) -> None:
        coefficients, _, _ = extract_path_arrays(
            self.paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        weights = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        np.testing.assert_allclose(
            compute_beamformed_path_powers(coefficients, weights),
            [1.0, 4.0, 9.0],
        )

    def test_power_weighted_delay_and_rms(self) -> None:
        powers = np.array([1.0, 4.0, 9.0])
        statistics = compute_delay_statistics(
            self.delays,
            powers,
            np.array([True, True, True]),
        )
        expected_mean = float(np.average(self.delays, weights=powers))
        expected_rms = float(
            np.sqrt(np.average((self.delays - expected_mean) ** 2, weights=powers))
        )
        self.assertAlmostEqual(statistics.power_weighted_delay_s, expected_mean)
        self.assertAlmostEqual(statistics.rms_delay_spread_s, expected_rms)
        self.assertEqual(statistics.strongest_path_delay_s, 3e-6)
        self.assertEqual(statistics.valid_path_count, 3)

    def test_invalid_path_does_not_contribute_to_pdp(self) -> None:
        statistics = compute_delay_statistics(
            self.delays,
            np.array([1.0, 4.0, 1e12]),
            np.array([True, True, False]),
        )
        self.assertAlmostEqual(statistics.power_weighted_delay_s, 1.8e-6)
        self.assertEqual(statistics.usable_path_count, 2)

    def test_zero_total_path_power_fails(self) -> None:
        with self.assertRaises(NoValidPathsError):
            compute_delay_statistics(
                self.delays,
                np.zeros(3),
                np.array([True, True, True]),
            )

    def test_los_is_explicit_geometric_delay_heuristic(self) -> None:
        distance = 300.0
        direct_delay = distance / SPEED_OF_LIGHT_M_PER_S
        self.assertTrue(
            infer_los_from_delay(
                np.array([direct_delay, direct_delay + 1e-6]),
                distance,
                tolerance_s=1e-9,
            )
        )
        self.assertFalse(
            infer_los_from_delay(
                np.array([direct_delay + 1e-6]),
                distance,
                tolerance_s=1e-9,
            )
        )

    def test_link_budget_preserves_sleep_power_without_double_counting(self) -> None:
        normal = np.array([1e-12, 4e-12])
        sleep = normal * 0.5
        result = compute_link_budget(
            normal,
            sleep,
            tx_power_dbm=30.0,
            temperature_k=290.0,
            noise_figure_db=5.0,
            subcarrier_spacing_hz=30_000.0,
            sleep_weight_norm_squared=0.5,
        )
        self.assertAlmostEqual(result.normal_tx_power_w, 1.0)
        self.assertAlmostEqual(result.sleep_tx_power_w, 0.5)
        self.assertAlmostEqual(result.per_subcarrier_tx_power_w, 0.5)
        self.assertAlmostEqual(result.rx_power_loss_db, 10.0 * math.log10(2.0))
        expected_noise = BOLTZMANN_CONSTANT * 290.0 * 30_000.0 * 10**0.5
        self.assertAlmostEqual(result.per_subcarrier_noise_power_w, expected_noise)

    def test_linear_average_is_distinct_from_mean_db(self) -> None:
        result = compute_link_budget(
            np.array([1e-12, 4e-12]),
            np.array([0.5e-12, 2e-12]),
            tx_power_dbm=30.0,
            temperature_k=290.0,
            noise_figure_db=5.0,
            subcarrier_spacing_hz=30_000.0,
            sleep_weight_norm_squared=0.5,
        )
        self.assertGreater(
            result.normal_rx_power_dbm,
            result.normal_mean_subcarrier_rx_power_dbm,
        )

    def test_real_weights_are_rejected(self) -> None:
        coefficients, _, _ = extract_path_arrays(
            self.paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        with self.assertRaises(TypeError):
            compute_beamformed_path_powers(coefficients, np.array([1.0, 0.0]))

    def test_extracts_real_and_imag_paths_to_complex_coefficients(self) -> None:
        real = np.zeros((1, 2, 1, 2, 3), dtype=np.float32)
        imag = np.zeros_like(real)
        real[0, 0, 0, 0, 0] = 1.0
        imag[0, 1, 0, 1, 2] = 1.0
        paths = SimpleNamespace(
            a=(real, imag),
            tau=np.ones((1, 1, 3), dtype=np.float64) * 1e-6,
            valid=np.full((1, 1, 3), True),
        )
        coefficients, delays, valid = extract_path_arrays(
            paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        self.assertTrue(np.iscomplexobj(coefficients))
        self.assertEqual(coefficients.shape, (2, 2, 3))
        expected = real[0, :, 0, :, :] + 1j * imag[0, :, 0, :, :]
        np.testing.assert_allclose(coefficients, expected)
        self.assertEqual(delays.shape, (3,))
        self.assertEqual(valid.shape, (3,))

    def test_extract_raises_on_real_imag_shape_mismatch(self) -> None:
        real = np.zeros((1, 2, 1, 2, 3))
        imag = np.zeros((1, 2, 1, 2, 4))
        paths = SimpleNamespace(
            a=(real, imag),
            tau=np.zeros((1, 1, 3)),
            valid=np.full((1, 1, 3), True),
        )
        with self.assertRaises(ValueError):
            extract_path_arrays(paths, num_rx_ports=2, num_tx_ports=2)

    def test_extracted_coefficients_are_complex(self) -> None:
        real = np.zeros((1, 2, 1, 2, 3))
        imag = np.zeros_like(real)
        paths = SimpleNamespace(
            a=(real, imag),
            tau=np.zeros((1, 1, 3)),
            valid=np.full((1, 1, 3), True),
        )
        coefficients, _, _ = extract_path_arrays(
            paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        self.assertTrue(np.iscomplexobj(coefficients))

    def test_extract_preserves_single_path_axis_for_tuple(self) -> None:
        real = np.zeros((1, 2, 1, 2, 1))
        imag = np.zeros_like(real)
        real[0, 0, 0, 0, 0] = 1.0
        imag[0, 1, 0, 1, 0] = 1.0
        paths = SimpleNamespace(
            a=(real, imag),
            tau=np.array([[[1e-6]]]),
            valid=np.array([[[True]]]),
        )
        coefficients, delays, valid = extract_path_arrays(
            paths,
            num_rx_ports=2,
            num_tx_ports=2,
        )
        self.assertEqual(coefficients.shape, (2, 2, 1))
        self.assertEqual(delays.shape, (1,))
        self.assertEqual(valid.shape, (1,))
        self.assertTrue(np.iscomplexobj(coefficients))

    def test_imaginary_part_changes_path_power_and_delay(self) -> None:
        real = np.zeros((2, 2, 2), dtype=np.float64)
        imag = np.zeros_like(real)
        # Path 0 carries real-only energy on TX port 0.
        real[:, 0, 0] = 1.0
        # Path 1 carries imaginary-only energy on TX port 0.
        imag[:, 0, 1] = 1.0
        weights = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        complex_coeff = real + 1j * imag

        power_complex = compute_beamformed_path_powers(complex_coeff, weights)
        effective_real = np.einsum("rtp,t->rp", real, weights)
        power_real = np.sum(np.abs(effective_real) ** 2, axis=0, dtype=np.float64)

        self.assertFalse(np.allclose(power_complex, power_real))
        # Path 0 is purely real (power 2 from two RX ports).
        self.assertEqual(power_complex[0], 2.0)
        self.assertEqual(power_real[0], 2.0)
        # Path 1 is purely imaginary (real-only gives 0, complex gives 2).
        self.assertEqual(power_complex[1], 2.0)
        self.assertEqual(power_real[1], 0.0)

        delays = np.array([1e-6, 2e-6])
        usable = np.array([True, True])
        stats_real = compute_delay_statistics(delays, power_real, usable)
        stats_complex = compute_delay_statistics(delays, power_complex, usable)
        self.assertNotAlmostEqual(
            stats_real.power_weighted_delay_s,
            stats_complex.power_weighted_delay_s,
        )
        self.assertNotAlmostEqual(
            stats_real.rms_delay_spread_s,
            stats_complex.rms_delay_spread_s,
        )


class CfrConsistencyToleranceTest(unittest.TestCase):
    """Regression for the _compute_channels wideband/center CFR consistency check."""

    def _make_trial_tensors(self, relative_error: float):
        base = torch.tensor(
            [[1.0 + 1.0j, 0.5 + 0.2j],
             [0.3 - 0.1j, 0.2 + 0.4j]],
            dtype=torch.complex64,
        )
        perturbed = base * (1.0 + relative_error)
        return perturbed, base

    def test_4e5_relative_difference_passes(self) -> None:
        # Representative of the complex64 GPU materialization difference
        # observed for UE 6 / UE 19 before the tolerance was relaxed.
        perturbed, base = self._make_trial_tensors(relative_error=4e-5)
        torch.testing.assert_close(perturbed, base, rtol=1e-4, atol=1e-12)

    def test_1e2_relative_difference_fails(self) -> None:
        # A clearly erroneous CFR mismatch must still fail.
        perturbed, base = self._make_trial_tensors(relative_error=1e-2)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(perturbed, base, rtol=1e-4, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
