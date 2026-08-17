"""Tests for pmi_mask.py."""

from __future__ import annotations

import pytest
import torch

from mMIMO_sleep.codebook.pmi_mask import (
    compute_total_peak_loss_db,
    create_total_loss_pmi_mask,
    pmi_indices_from_mask,
)


def test_compute_total_peak_loss_db_shape() -> None:
    loss = compute_total_peak_loss_db(
        num_vertical=8,
        num_horizontal=8,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        grid_points=61,
        device="cpu",
    )
    assert loss.shape == (8, 32)


def test_pmi_mask_shape_and_content() -> None:
    mask = create_total_loss_pmi_mask(
        num_vertical=8,
        num_horizontal=8,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        target_loss_db=6.0,
        tolerance_db=1.0,
        grid_points=61,
        device="cpu",
    )
    assert mask.shape == (8, 32)
    assert mask.dtype == torch.bool
    # Most beams should be close to 6 dB; at least one should be selected.
    assert mask.any()
    # With the default 8x8 right-half muting, some edge beams deviate, so
    # the mask should not be all True.
    assert not mask.all()


def test_pmi_mask_tolerance_extremes() -> None:
    # Very tight tolerance around an impossible value should yield no matches.
    mask_tight = create_total_loss_pmi_mask(
        num_vertical=8,
        num_horizontal=8,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        target_loss_db=100.0,
        tolerance_db=0.1,
        grid_points=61,
        device="cpu",
    )
    assert not mask_tight.any()

    # Very loose tolerance should select everything.
    mask_loose = create_total_loss_pmi_mask(
        num_vertical=8,
        num_horizontal=8,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        target_loss_db=6.0,
        tolerance_db=100.0,
        grid_points=61,
        device="cpu",
    )
    assert mask_loose.all()


def test_pmi_indices_from_mask() -> None:
    mask = torch.tensor(
        [
            [True, False],
            [False, True],
        ]
    )
    indices = pmi_indices_from_mask(mask)
    assert indices == [(0, 0), (1, 1)]
