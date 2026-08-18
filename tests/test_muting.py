"""Tests for muting.py."""

from __future__ import annotations

import pytest
import torch

from mMIMO_sleep.array_config import ArrayConfig
from mMIMO_sleep.codebook.dft import generate_dft_codebook
from mMIMO_sleep.codebook.muting import (
    active_power_fraction,
    apply_muting_mask,
    create_right_half_mask,
)


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_right_half_mask_length(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    mask = create_right_half_mask(config, device="cpu")
    assert mask.numel() == config.num_physical_ports


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_both_polarizations_muted_together(
    elements_per_subarray: int,
) -> None:
    """Same physical element position must be muted in both polarizations."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    mask = create_right_half_mask(config, device="cpu")
    num_elements = config.num_physical_elements
    pol0 = mask[:num_elements]
    pol1 = mask[num_elements:]
    torch.testing.assert_close(pol0, pol1)


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_active_and_muted_port_counts(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    mask = create_right_half_mask(config, device="cpu")
    active = int(mask.sum().item())
    muted = mask.numel() - active
    assert active == config.num_physical_ports // 2
    assert muted == config.num_physical_ports // 2


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_apply_muting_no_renormalization(
    elements_per_subarray: int,
) -> None:
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
    mask = create_right_half_mask(config, device="cpu")
    sleep = apply_muting_mask(codebook, mask)
    # Sleep weights are elementwise products; no scaling is applied.
    torch.testing.assert_close(sleep, codebook * mask.to(codebook.dtype))


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_active_power_fraction(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    mask = create_right_half_mask(config, device="cpu")
    assert active_power_fraction(mask) == pytest.approx(0.5)


def test_odd_num_horizontal_raises() -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=7,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    with pytest.raises(ValueError):
        create_right_half_mask(config, device="cpu")


def test_apply_muting_incompatible_dimensions() -> None:
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
        num_i2=4,
        device="cpu",
    )
    bad_mask = torch.ones(codebook.shape[-1] + 1)
    with pytest.raises(ValueError):
        apply_muting_mask(codebook, bad_mask)
