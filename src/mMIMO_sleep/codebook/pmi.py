"""PMI conversion utilities for the simplified two-dimensional DFT codebook."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PMI:
    """Rank-one PMI for the simplified DFT codebook."""

    i11: int
    i12: int


def _validate_dimensions(
    num_horizontal_beams: int,
    num_vertical_beams: int,
) -> None:
    if num_horizontal_beams <= 0:
        raise ValueError("num_horizontal_beams must be positive.")
    if num_vertical_beams <= 0:
        raise ValueError("num_vertical_beams must be positive.")


def pmi_to_beam_index(
    pmi: PMI,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
) -> int:
    """Convert ``PMI(i11, i12)`` to the row index of the DFT codebook."""
    _validate_dimensions(num_horizontal_beams, num_vertical_beams)
    if not 0 <= pmi.i11 < num_horizontal_beams:
        raise ValueError(
            f"i11 must be in [0, {num_horizontal_beams}), got {pmi.i11}."
        )
    if not 0 <= pmi.i12 < num_vertical_beams:
        raise ValueError(
            f"i12 must be in [0, {num_vertical_beams}), got {pmi.i12}."
        )
    return pmi.i12 * num_horizontal_beams + pmi.i11


def beam_index_to_pmi(
    beam_index: int,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
) -> PMI:
    """Convert a DFT codebook row index to ``PMI(i11, i12)``."""
    _validate_dimensions(num_horizontal_beams, num_vertical_beams)
    num_codewords = num_horizontal_beams * num_vertical_beams
    if not 0 <= beam_index < num_codewords:
        raise ValueError(
            f"beam_index must be in [0, {num_codewords}), got {beam_index}."
        )
    i12, i11 = divmod(beam_index, num_horizontal_beams)
    return PMI(i11=i11, i12=i12)
