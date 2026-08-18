"""Tests for pmi.py."""

from __future__ import annotations

import pytest

from mMIMO_sleep.codebook.pmi import PMI, beam_index_to_pmi, pmi_to_beam_index


def test_pmi_round_trip() -> None:
    pmi = PMI(i11=5, i12=3, i2=2)
    beam_index = pmi_to_beam_index(
        pmi,
        num_horizontal_beams=32,
        num_vertical_beams=8,
        num_i2=4,
    )
    recovered = beam_index_to_pmi(
        beam_index,
        num_horizontal_beams=32,
        num_vertical_beams=8,
        num_i2=4,
    )
    assert recovered == pmi


@pytest.mark.parametrize("num_i2", [1, 2, 4])
def test_pmi_ordering_i2_fastest(num_i2: int) -> None:
    """Consecutive beam indices change i2 fastest, then i11, then i12."""
    indices = [
        pmi_to_beam_index(
            PMI(i11=i11, i12=i12, i2=i2),
            num_horizontal_beams=4,
            num_vertical_beams=2,
            num_i2=num_i2,
        )
        for i12 in range(2)
        for i11 in range(4)
        for i2 in range(num_i2)
    ]
    assert indices == list(range(2 * 4 * num_i2))


def test_pmi_independent_of_subarray_size() -> None:
    """PMI conversion does not depend on subarray geometry."""
    pmi = PMI(i11=3, i12=1, i2=0)
    for num_v in [4, 8]:
        beam_index = pmi_to_beam_index(
            pmi,
            num_horizontal_beams=8,
            num_vertical_beams=num_v,
            num_i2=4,
        )
        recovered = beam_index_to_pmi(
            beam_index,
            num_horizontal_beams=8,
            num_vertical_beams=num_v,
            num_i2=4,
        )
        assert recovered == pmi


@pytest.mark.parametrize(
    "i11,i12,i2",
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (8, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
    ],
)
def test_pmi_to_beam_index_invalid(i11: int, i12: int, i2: int) -> None:
    with pytest.raises(ValueError):
        pmi_to_beam_index(
            PMI(i11=i11, i12=i12, i2=i2),
            num_horizontal_beams=8,
            num_vertical_beams=4,
            num_i2=4,
        )


def test_beam_index_to_pmi_invalid() -> None:
    with pytest.raises(ValueError):
        beam_index_to_pmi(
            8 * 4 * 4,  # one past the end
            num_horizontal_beams=8,
            num_vertical_beams=4,
            num_i2=4,
        )
