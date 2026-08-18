"""Executable specification tests for pmi_mask.py."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

import mMIMO_sleep.codebook.pmi_mask as pmi_mask
from mMIMO_sleep.array_config import ArrayConfig
from mMIMO_sleep.codebook.dft import generate_dft_codebook
from mMIMO_sleep.codebook.pmi import PMI, pmi_to_beam_index

# ---------------------------------------------------------------------------
# Deterministic helper tests
# ---------------------------------------------------------------------------


def test_compute_loss_at_normal_peak_db_uses_normal_argmax() -> None:
    """Loss must use sleep power gathered at the normal-power argmax."""
    normal = torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.0, 3.0]])
    sleep = torch.tensor([[10.0, 20.0, 5.0], [1.0, 2.0, 30.0]])
    # normal argmax per row: [1, 2]; sleep at same indices: [20, 30]
    expected = torch.tensor([-10.0, -10.0])
    result = pmi_mask._compute_loss_at_normal_peak_db(normal, sleep)
    torch.testing.assert_close(result, expected)


def test_compute_loss_at_normal_peak_db_differs_from_separate_peak() -> None:
    """Normal-peak-direction loss must differ from a buggy separate-peak ratio."""
    normal = torch.tensor([[1.0, 4.0]])  # argmax at index 1
    sleep = torch.tensor([[8.0, 2.0]])  # argmax at index 0
    correct = pmi_mask._compute_loss_at_normal_peak_db(normal, sleep)
    # correct: 4 / 2 -> 10*log10(2) dB
    torch.testing.assert_close(
        correct,
        torch.tensor(
            [10.0 * math.log10(2.0)],
            dtype=correct.dtype,
            device=correct.device,
        ),
        atol=1e-3,
        rtol=0.0,
    )
    # buggy separate-peak ratio: 4 / 8 -> ~-3.01 dB
    wrong = 10.0 * torch.log10(
        normal.max(dim=-1).values / sleep.max(dim=-1).values
    )
    assert not torch.allclose(correct, wrong)


def test_zero_sleep_power_returns_positive_inf() -> None:
    """When sleep power is zero at the normal peak direction, loss is +inf."""
    normal = torch.tensor([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    sleep = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.5],
    ])
    loss = pmi_mask._compute_loss_at_normal_peak_db(normal, sleep)
    assert torch.isposinf(loss[0])
    torch.testing.assert_close(
        loss[1],
        torch.tensor(10.0 * math.log10(4.0)),
    )


def test_compute_loss_at_normal_peak_db_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        pmi_mask._compute_loss_at_normal_peak_db(
            torch.zeros(2, 3), torch.zeros(2, 4)
        )


def test_compute_loss_at_normal_peak_db_rejects_empty_direction_dim() -> None:
    with pytest.raises(ValueError):
        pmi_mask._compute_loss_at_normal_peak_db(
            torch.zeros(2, 0), torch.zeros(2, 0)
        )


def test_total_power_is_noncoherent_sum() -> None:
    """Total power is the incoherent sum over polarizations."""
    e0 = torch.tensor([1.0 + 1.0j, 1.0 - 1.0j])
    e1 = torch.tensor([1.0 + 1.0j, -1.0 + 1.0j])
    p = pmi_mask._total_power_from_polarized_fields(e0, e1)
    expected = e0.abs().square() + e1.abs().square()
    torch.testing.assert_close(p, expected)


def test_total_power_is_not_coherent_sum_squared() -> None:
    """Guard against accidentally summing fields before squaring."""
    e0 = torch.tensor([1.0 + 0.0j])
    e1 = torch.tensor([1.0 + 0.0j])
    p = pmi_mask._total_power_from_polarized_fields(e0, e1)
    # noncoherent: |1|^2 + |1|^2 = 2
    # coherent: |1 + 1|^2 = 4
    assert p.item() == 2.0
    assert p.item() != 4.0


def test_mask_from_loss_boundary() -> None:
    """Boundary uses inclusive <= tolerance."""
    target = pmi_mask.IDEAL_HALF_APERTURE_LOSS_DB
    tol = 1.0
    loss = torch.tensor([
        target - 1.0001,
        target - 1.0,
        target,
        target + 1.0,
        target + 1.0001,
    ])
    mask = pmi_mask._mask_from_loss(loss, target, tol)
    expected = torch.tensor([False, True, True, True, False])
    assert torch.equal(mask, expected)


def test_mask_from_loss_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        pmi_mask._mask_from_loss(torch.zeros(3), 0.0, -0.1)


def test_mask_from_loss_nan_is_invalid() -> None:
    loss = torch.tensor([0.0, float("nan"), 1.0])
    mask = pmi_mask._mask_from_loss(loss, 0.0, 0.5)
    assert mask.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# Visible direction-cosine grid
# ---------------------------------------------------------------------------


def test_visible_uv_grid_properties() -> None:
    u_h, u_v, visible = pmi_mask._make_visible_uv_grid(
        121, dtype=torch.float32, device="cpu"
    )
    assert u_h.shape == (121, 121)
    assert u_v.shape == (121, 121)
    assert visible.shape == (121, 121)
    assert u_h.dtype == torch.float32
    assert u_v.dtype == torch.float32
    assert visible.dtype == torch.bool
    # unit-circle corners are invisible
    assert not visible[0, 0].item()
    assert not visible[0, 120].item()
    assert not visible[120, 0].item()
    assert not visible[120, 120].item()
    # center is visible
    assert visible[60, 60].item()
    # all marked visible satisfy the constraint
    assert torch.all(
        u_h[visible].square() + u_v[visible].square() <= 1.0 + 1e-6
    )


def test_visible_uv_grid_dtype_float64() -> None:
    u_h, u_v, visible = pmi_mask._make_visible_uv_grid(
        11, dtype=torch.float64, device="cpu"
    )
    assert u_h.dtype == torch.float64
    assert u_v.dtype == torch.float64


# ---------------------------------------------------------------------------
# Array-factor helper
# ---------------------------------------------------------------------------


def _direct_array_factor(
    weights_2d: torch.Tensor,
    u_h: torch.Tensor,
    u_v: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    """Brute-force reference with the same phase convention as the helper."""
    num_v, num_h = weights_2d.shape
    result = torch.zeros(len(u_h), dtype=torch.complex128)
    for idx in range(len(u_h)):
        for r in range(num_v):
            for c in range(num_h):
                angle = (
                    2.0
                    * math.pi
                    * spacing
                    * (u_h[idx].item() * c + u_v[idx].item() * r)
                )
                phase = complex(math.cos(angle), -math.sin(angle))
                result[idx] += weights_2d[r, c].item() * phase
    return result


def test_array_factor_matches_direct_sum_2d() -> None:
    weights = torch.tensor(
        [
            [1.0 + 0.0j, 0.5 + 0.5j, -1.0 + 0.0j],
            [0.0 + 1.0j, 1.0 - 1.0j, 0.0 + 0.5j],
        ],
        dtype=torch.complex64,
    )
    u_h = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    u_v = torch.tensor([0.3, 0.4, -0.1], dtype=torch.float32)
    af = pmi_mask._array_factor_from_weights(
        weights, u_h, u_v, spacing_wavelengths=0.5
    )
    expected = _direct_array_factor(weights, u_h, u_v, 0.5).to(torch.complex64)
    torch.testing.assert_close(af, expected, atol=1e-5, rtol=1e-5)


def test_array_factor_matches_direct_sum_batched() -> None:
    weights = torch.randn(3, 2, 3, dtype=torch.complex64)
    u_h = torch.tensor([0.1, -0.2], dtype=torch.float32)
    u_v = torch.tensor([0.3, 0.4], dtype=torch.float32)
    af = pmi_mask._array_factor_from_weights(
        weights, u_h, u_v, spacing_wavelengths=0.5
    )
    expected = torch.stack(
        [
            _direct_array_factor(w, u_h, u_v, 0.5).to(torch.complex64)
            for w in weights
        ]
    )
    torch.testing.assert_close(af, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("phase_sign", [-1, 1])
def test_array_factor_peak_follows_dft_phase_sign(phase_sign: int) -> None:
    """For a 1x4 array, i11=1, d=0.5, the peak is at u_h = phase_sign * 0.5."""
    config = ArrayConfig(
        num_subarray_rows=1,
        num_horizontal=4,
        elements_per_subarray=1,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=1,
        num_horizontal_beams=4,
        num_i2=1,
        phase_sign=phase_sign,
        device="cpu",
    )
    # beam index 1 -> (i12=0, i11=1)
    w = codebook[1, : config.num_physical_elements].reshape(1, 4)
    grid_points = 41
    u = torch.linspace(-1.0, 1.0, grid_points)
    u_h, u_v = torch.meshgrid(u, u, indexing="ij")
    visible = u_h.square() + u_v.square() <= 1.0
    af = pmi_mask._array_factor_from_weights(
        w, u_h[visible], u_v[visible], spacing_wavelengths=0.5
    )
    power = af.abs().square()
    peak_idx = power.argmax()
    peak_u_h = u_h[visible][peak_idx].item()
    expected_u_h = phase_sign * 0.5
    assert abs(peak_u_h - expected_u_h) < 0.03  # grid step is 0.05


# ---------------------------------------------------------------------------
# Public API shape, dtype and device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_compute_total_peak_loss_db_shape_dtype(
    elements_per_subarray: int,
) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    loss = pmi_mask.compute_total_peak_loss_db(
        config,
        num_vertical_beams=4,
        num_horizontal_beams=8,
        num_i2=4,
        grid_points=61,
        beam_batch_size=8,
        device="cpu",
    )
    assert loss.shape == (4, 8)
    assert loss.dtype == torch.float32


def test_compute_total_peak_loss_db_complex128_yields_float64() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    loss = pmi_mask.compute_total_peak_loss_db(
        config, 4, 8, dtype=torch.complex128, device="cpu"
    )
    assert loss.dtype == torch.float64


def test_compute_total_peak_loss_db_rejects_non_complex_dtype() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(TypeError):
        pmi_mask.compute_total_peak_loss_db(
            config, 4, 8, dtype=torch.float32, device="cpu"
        )


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_create_total_loss_pmi_mask_shape_dtype(
    elements_per_subarray: int,
) -> None:
    config = ArrayConfig(4, 8, elements_per_subarray, 2)
    mask = pmi_mask.create_total_loss_pmi_mask(
        config,
        num_vertical_beams=4,
        num_horizontal_beams=8,
        num_i2=4,
        grid_points=61,
        device="cpu",
    )
    assert mask.shape == (4, 8)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# PMI index helpers
# ---------------------------------------------------------------------------


def test_pmi_indices_from_mask_axis_order() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    indices = pmi_mask.pmi_indices_from_mask(mask)
    assert indices == [(0, 0), (1, 1)]


def test_beam_indices_from_mask_exact_expansion() -> None:
    mask = torch.tensor([[False, True]])
    num_i2 = 3
    indices = pmi_mask.beam_indices_from_mask(mask, num_i2=num_i2)
    expected = [
        pmi_to_beam_index(
            PMI(i11=1, i12=0, i2=i2),
            num_horizontal_beams=2,
            num_vertical_beams=1,
            num_i2=num_i2,
        )
        for i2 in range(num_i2)
    ]
    assert indices == expected


def test_beam_indices_from_mask_multi_spatial() -> None:
    mask = torch.tensor([[False, True], [True, False]])
    num_i2 = 3
    indices = pmi_mask.beam_indices_from_mask(mask, num_i2=num_i2)
    expected: list[int] = []
    for i12, i11 in [(0, 1), (1, 0)]:
        for i2 in range(num_i2):
            expected.append(
                pmi_to_beam_index(
                    PMI(i11=i11, i12=i12, i2=i2),
                    num_horizontal_beams=2,
                    num_vertical_beams=2,
                    num_i2=num_i2,
                )
            )
    assert indices == expected


def test_beam_indices_from_mask_rejects_non_bool() -> None:
    with pytest.raises(TypeError):
        pmi_mask.beam_indices_from_mask(torch.zeros(2, 2), num_i2=4)


def test_beam_indices_from_mask_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        pmi_mask.beam_indices_from_mask(torch.tensor([True, False]), num_i2=4)


def test_beam_indices_from_mask_rejects_non_positive_num_i2() -> None:
    with pytest.raises(ValueError):
        pmi_mask.beam_indices_from_mask(torch.tensor([[True]]), num_i2=0)


# ---------------------------------------------------------------------------
# i2 invariance of total-power pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_i2_total_power_pattern_invariance(elements_per_subarray: int) -> None:
    """For a fixed spatial PMI, all i2 produce the same total-power pattern."""
    config = ArrayConfig(2, 4, elements_per_subarray, 2)
    codebook = generate_dft_codebook(
        config, 2, 4, num_i2=4, device="cpu"
    )
    grid_points = 41
    u = torch.linspace(-1.0, 1.0, grid_points)
    u_h, u_v = torch.meshgrid(u, u, indexing="ij")
    visible = u_h.square() + u_v.square() <= 1.0
    u_h_f = u_h[visible]
    u_v_f = u_v[visible]

    spatial_idx = 3
    patterns: list[torch.Tensor] = []
    for i2 in range(4):
        beam_idx = spatial_idx * 4 + i2
        w = codebook[beam_idx]
        pol0 = w[: config.num_physical_elements].reshape(
            config.num_physical_rows, config.num_horizontal
        )
        pol1 = w[config.num_physical_elements :].reshape(
            config.num_physical_rows, config.num_horizontal
        )
        e0 = pmi_mask._array_factor_from_weights(pol0, u_h_f, u_v_f, 0.5)
        e1 = pmi_mask._array_factor_from_weights(pol1, u_h_f, u_v_f, 0.5)
        patterns.append(
            pmi_mask._total_power_from_polarized_fields(e0, e1)
        )
    for i in range(1, 4):
        torch.testing.assert_close(
            patterns[0], patterns[i], atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_compute_total_peak_loss_db_i2_invariance(
    elements_per_subarray: int,
) -> None:
    config = ArrayConfig(4, 8, elements_per_subarray, 2)
    kwargs: dict = dict(
        num_vertical_beams=4,
        num_horizontal_beams=8,
        grid_points=61,
        beam_batch_size=8,
        device="cpu",
    )
    loss_i2_1 = pmi_mask.compute_total_peak_loss_db(
        config, num_i2=1, **kwargs
    )
    loss_i2_4 = pmi_mask.compute_total_peak_loss_db(
        config, num_i2=4, **kwargs
    )
    torch.testing.assert_close(loss_i2_1, loss_i2_4, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Batching and grid convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beam_batch_size", [1, 4, 1000])
def test_batch_size_invariance(beam_batch_size: int) -> None:
    config = ArrayConfig(4, 8, 2, 2)
    kwargs: dict = dict(
        num_vertical_beams=4,
        num_horizontal_beams=8,
        num_i2=4,
        grid_points=61,
        device="cpu",
    )
    loss_1 = pmi_mask.compute_total_peak_loss_db(
        config, beam_batch_size=1, **kwargs
    )
    loss_b = pmi_mask.compute_total_peak_loss_db(
        config, beam_batch_size=beam_batch_size, **kwargs
    )
    loss_default = pmi_mask.compute_total_peak_loss_db(config, **kwargs)
    torch.testing.assert_close(loss_1, loss_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(loss_1, loss_default, atol=1e-5, rtol=1e-5)


def test_loss_converges_with_grid_resolution() -> None:
    """Compare a stable broadside-ish beam under two grid densities."""
    config = ArrayConfig(2, 4, 1, 2)
    kwargs: dict = dict(
        num_vertical_beams=2, num_horizontal_beams=4, num_i2=4, device="cpu"
    )
    loss_41 = pmi_mask.compute_total_peak_loss_db(
        config, grid_points=41, **kwargs
    )
    loss_81 = pmi_mask.compute_total_peak_loss_db(
        config, grid_points=81, **kwargs
    )
    # i12=0, i11=2 is broadside-ish for this small array
    torch.testing.assert_close(loss_41[0, 2], loss_81[0, 2], atol=0.2, rtol=0.0)


# ---------------------------------------------------------------------------
# Helper reuse: pmi_mask must delegate to dft and muting
# ---------------------------------------------------------------------------


def test_compute_total_peak_loss_db_uses_codebook_helpers() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    num_vertical_beams = 2
    num_horizontal_beams = 4
    num_i2 = 4
    num_ports = config.num_physical_ports
    total_beams = num_vertical_beams * num_horizontal_beams * num_i2
    fake_codebook = torch.randn(
        total_beams, num_ports, dtype=torch.complex64
    )
    fake_mask = torch.ones(num_ports, dtype=torch.float32)
    fake_sleep = fake_codebook * 0.5

    with (
        patch("mMIMO_sleep.codebook.pmi_mask.generate_dft_codebook") as mock_dft,
        patch(
            "mMIMO_sleep.codebook.pmi_mask.create_right_half_mask"
        ) as mock_mask,
        patch(
            "mMIMO_sleep.codebook.pmi_mask.apply_muting_mask"
        ) as mock_apply,
    ):
        mock_dft.return_value = fake_codebook
        mock_mask.return_value = fake_mask
        mock_apply.return_value = fake_sleep
        pmi_mask.compute_total_peak_loss_db(
            config,
            num_vertical_beams=num_vertical_beams,
            num_horizontal_beams=num_horizontal_beams,
            num_i2=num_i2,
            grid_points=21,
            device="cpu",
        )
        mock_dft.assert_called_once()
        mock_mask.assert_called_once()
        mock_apply.assert_called_once()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_vertical_beams", 0),
        ("num_vertical_beams", -1),
        ("num_horizontal_beams", 0),
        ("num_i2", 0),
    ],
)
def test_compute_total_peak_loss_db_rejects_non_positive_counts(
    name: str, value: int
) -> None:
    config = ArrayConfig(4, 8, 2, 2)
    kwargs: dict = dict(
        num_vertical_beams=4, num_horizontal_beams=8, num_i2=4, device="cpu"
    )
    kwargs[name] = value
    with pytest.raises(ValueError):
        pmi_mask.compute_total_peak_loss_db(config, **kwargs)


def test_compute_total_peak_loss_db_rejects_invalid_phase_sign() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(ValueError):
        pmi_mask.compute_total_peak_loss_db(config, 4, 8, phase_sign=2)


@pytest.mark.parametrize(
    "spacing", [0.0, -0.5, float("nan"), float("inf")]
)
def test_compute_total_peak_loss_db_rejects_invalid_spacing(
    spacing: float,
) -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(ValueError):
        pmi_mask.compute_total_peak_loss_db(
            config, 4, 8, spacing_wavelengths=spacing
        )


def test_compute_total_peak_loss_db_rejects_small_grid() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(ValueError):
        pmi_mask.compute_total_peak_loss_db(config, 4, 8, grid_points=1)


@pytest.mark.parametrize("beam_batch_size", [0, -1])
def test_compute_total_peak_loss_db_rejects_invalid_batch_size(
    beam_batch_size: int,
) -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(ValueError):
        pmi_mask.compute_total_peak_loss_db(
            config, 4, 8, beam_batch_size=beam_batch_size
        )


def test_create_total_loss_pmi_mask_rejects_negative_tolerance() -> None:
    config = ArrayConfig(4, 8, 2, 2)
    with pytest.raises(ValueError):
        pmi_mask.create_total_loss_pmi_mask(
            config, 4, 8, tolerance_db=-1.0
        )


@pytest.mark.parametrize("field", ["target_loss_db", "tolerance_db"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_create_total_loss_pmi_mask_rejects_non_finite_target_tolerance(
    field: str, bad_value: float
) -> None:
    config = ArrayConfig(4, 8, 2, 2)
    kwargs: dict = {field: bad_value}
    with pytest.raises(ValueError):
        pmi_mask.create_total_loss_pmi_mask(config, 4, 8, **kwargs)
