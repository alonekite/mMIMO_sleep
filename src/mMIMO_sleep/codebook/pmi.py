"""PMI conversion utilities for the rank-one DFT codebook."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PMI:
    """Rank-one PMI for the simplified DFT codebook.

    The three indices correspond to the horizontal beam (``i11``), vertical
    beam (``i12``), and cross-polarization co-phasing (``i2``).  Their natural
    ordering in a flat codebook is ``(i12, i11, i2)`` with ``i2`` varying
    fastest.
    """

    i11: int
    i12: int
    i2: int


def _validate_dimensions(
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> None:
    if not all(
        isinstance(x, int) for x in (num_horizontal_beams, num_vertical_beams, num_i2)
    ):
        raise TypeError("Beam counts must be integers.")
    if num_horizontal_beams <= 0:
        raise ValueError("num_horizontal_beams must be positive.")
    if num_vertical_beams <= 0:
        raise ValueError("num_vertical_beams must be positive.")
    if num_i2 <= 0:
        raise ValueError("num_i2 must be positive.")


def pmi_to_beam_index(
    pmi: PMI,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> int:
    """Convert ``PMI(i11, i12, i2)`` to the flat codebook row index.

    The index ordering is ``(i12, i11, i2)`` with ``i2`` as the fastest
    varying dimension.
    """
    _validate_dimensions(num_horizontal_beams, num_vertical_beams, num_i2)
    if not 0 <= pmi.i11 < num_horizontal_beams:
        raise ValueError(
            f"i11 must be in [0, {num_horizontal_beams}), got {pmi.i11}."
        )
    if not 0 <= pmi.i12 < num_vertical_beams:
        raise ValueError(
            f"i12 must be in [0, {num_vertical_beams}), got {pmi.i12}."
        )
    if not 0 <= pmi.i2 < num_i2:
        raise ValueError(f"i2 must be in [0, {num_i2}), got {pmi.i2}.")
    return (pmi.i12 * num_horizontal_beams + pmi.i11) * num_i2 + pmi.i2


def beam_index_to_pmi(
    beam_index: int,
    *,
    num_horizontal_beams: int,
    num_vertical_beams: int,
    num_i2: int,
) -> PMI:
    """Convert a flat codebook row index to ``PMI(i11, i12, i2)``.

    The inverse of :func:`pmi_to_beam_index`, using the same ordering.
    """
    _validate_dimensions(num_horizontal_beams, num_vertical_beams, num_i2)
    total = num_horizontal_beams * num_vertical_beams * num_i2
    if not 0 <= beam_index < total:
        raise ValueError(
            f"beam_index must be in [0, {total}), got {beam_index}."
        )
    tmp, i2 = divmod(beam_index, num_i2)
    i12, i11 = divmod(tmp, num_horizontal_beams)
    return PMI(i11=i11, i12=i12, i2=i2)
