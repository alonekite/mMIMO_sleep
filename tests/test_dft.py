"""Tests for dft.py."""

from __future__ import annotations

import pytest
import torch

from mMIMO_sleep.array_config import ArrayConfig
from mMIMO_sleep.codebook.dft import dft_vector, generate_dft_codebook


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_codebook_shape(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=4,
        device="cpu",
    )
    assert codebook.shape == (
        8 * 32 * 4,
        config.num_physical_ports,
    )


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_codeword_unit_norm(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=4,
        device="cpu",
    )
    norms = codebook.abs().square().sum(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_subarray_elements_share_phase(elements_per_subarray: int) -> None:
    """Within one subarray, all physical elements share the same complex weight."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=4,
        device="cpu",
    )
    # Use beam index 0 (i12=0, i11=0, i2=0) and polarization 0.
    w = codebook[0, : config.num_physical_elements].reshape(
        config.num_physical_rows, config.num_horizontal
    )
    for sub_row in range(config.num_subarray_rows):
        start = sub_row * config.elements_per_subarray
        end = start + config.elements_per_subarray
        subarray_weights = w[start:end, :]
        # All rows in the subarray should be identical up to amplitude scaling.
        first = subarray_weights[0:1, :]
        torch.testing.assert_close(
            subarray_weights,
            first.expand_as(subarray_weights),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_subarray_expansion_normalization(elements_per_subarray: int) -> None:
    """Subarray expansion preserves per-polarization spatial power."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=4,
        device="cpu",
    )
    w = codebook[0, : config.num_physical_elements]
    # Total per-polarization power should be 1/num_polarizations.
    per_pol_power = w.abs().square().sum()
    torch.testing.assert_close(
        per_pol_power,
        torch.tensor(0.5, dtype=per_pol_power.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_elements_per_subarray_one_degeneration() -> None:
    """elements_per_subarray=1 should give a one-to-one logical-to-physical map."""
    config = ArrayConfig(
        num_subarray_rows=8,
        num_horizontal=8,
        elements_per_subarray=1,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=1,
        device="cpu",
    )
    assert codebook.shape == (8 * 32, 8 * 8 * 2)


@pytest.mark.parametrize("num_i2", [1, 2, 4])
def test_i2_co_phasing_values(num_i2: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=num_i2,
        device="cpu",
    )
    num_elements = config.num_physical_elements
    # Pick a fixed spatial beam.
    spatial_index = 5
    pol0 = codebook[spatial_index * num_i2, :num_elements]
    for i2 in range(num_i2):
        pol1 = codebook[spatial_index * num_i2 + i2, num_elements:]
        expected_phase = torch.exp(
            torch.tensor(1j * 2.0 * torch.pi * i2 / num_i2)
        )
        # Compare phase difference, ignoring overall common phase.
        ratio = pol1 / (pol0 + 1e-12)
        # The ratio should be close to expected_phase for all elements.
        angles = torch.angle(ratio)
        expected_angle = torch.angle(expected_phase)
        angle_diff = (angles - expected_angle + torch.pi) % (2 * torch.pi) - torch.pi
        assert torch.all(angle_diff.abs() < 1e-4)


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_port_ordering_pol_major(elements_per_subarray: int) -> None:
    """Polarization-major ordering: pol 0 ports precede pol 1 ports."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_i2=4,
        device="cpu",
    )
    num_elements = config.num_physical_elements
    # pol0 and pol1 weights should be identical for i2=0 (co-phase = 1).
    pol0 = codebook[0, :num_elements]
    pol1 = codebook[0, num_elements:]
    torch.testing.assert_close(pol0, pol1, atol=1e-5, rtol=1e-5)


def test_dft_vector_unit_norm() -> None:
    v = dft_vector(8, 3, 8, device="cpu")
    torch.testing.assert_close(v.abs().square().sum(), torch.tensor(1.0))


def test_unsupported_num_polarizations() -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=3,
    )
    with pytest.raises(NotImplementedError):
        generate_dft_codebook(
            config,
            num_vertical_beams=8,
            num_horizontal_beams=32,
            num_i2=4,
            device="cpu",
        )
