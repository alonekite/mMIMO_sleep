"""DFT codebook generation for a uniform planar array (UPA).

The codebook ordering is ``beam_index = i12 * num_horizontal_beams + i11``.
Array elements are flattened in row-major order, with the horizontal element
index varying fastest. This ordering is shared with ``muting.py``.
"""

from __future__ import annotations

from typing import Final

import torch


_COMPLEX_DTYPES: Final = (torch.complex64, torch.complex128)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def dft_vector(
    num_antennas: int,
    beam_index: int,
    num_beams: int,
    *,
    phase_sign: int = 1,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create one unit-norm oversampled DFT steering vector.

    Args:
        num_antennas: Number of physical antenna elements on the axis.
        beam_index: DFT beam index in ``[0, num_beams)``.
        num_beams: Number of sampled beams on the axis. This may exceed
            ``num_antennas`` to create an oversampled codebook.
        phase_sign: ``+1`` or ``-1``. Choose the sign that matches the array
            and angle convention used by the simulator.
        dtype: Complex output dtype.
        device: PyTorch device for the returned tensor.

    Returns:
        Complex tensor with shape ``[num_antennas]`` and unit squared norm.
    """
    _validate_positive("num_antennas", num_antennas)
    _validate_positive("num_beams", num_beams)
    if not 0 <= beam_index < num_beams:
        raise ValueError(
            f"beam_index must be in [0, {num_beams}), got {beam_index}."
        )
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    if dtype not in _COMPLEX_DTYPES:
        raise TypeError(f"dtype must be complex64 or complex128, got {dtype}.")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    element_index = torch.arange(
        num_antennas,
        dtype=real_dtype,
        device=device,
    )
    phase = (
        phase_sign
        * 2.0
        * torch.pi
        * element_index
        * beam_index
        / num_beams
    )
    vector = torch.exp(1j * phase).to(dtype=dtype)
    return vector / torch.sqrt(
        torch.tensor(num_antennas, dtype=real_dtype, device=device)
    )


def generate_dft_codebook(
    num_vertical: int,
    num_horizontal: int,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    *,
    phase_sign: int = 1,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a separable two-dimensional DFT codebook for a UPA.

    Returns:
        Tensor with shape
        ``[num_vertical_beams * num_horizontal_beams,
        num_vertical * num_horizontal]``. Every row is a unit-norm codeword.
    """
    for name, value in (
        ("num_vertical", num_vertical),
        ("num_horizontal", num_horizontal),
        ("num_vertical_beams", num_vertical_beams),
        ("num_horizontal_beams", num_horizontal_beams),
    ):
        _validate_positive(name, value)

    vertical_vectors = torch.stack(
        [
            dft_vector(
                num_vertical,
                i12,
                num_vertical_beams,
                phase_sign=phase_sign,
                dtype=dtype,
                device=device,
            )
            for i12 in range(num_vertical_beams)
        ]
    )
    horizontal_vectors = torch.stack(
        [
            dft_vector(
                num_horizontal,
                i11,
                num_horizontal_beams,
                phase_sign=phase_sign,
                dtype=dtype,
                device=device,
            )
            for i11 in range(num_horizontal_beams)
        ]
    )

    # [Kv, Kh, Nv, Nh] -> [Kv * Kh, Nv * Nh]. The horizontal antenna
    # and horizontal beam indices vary fastest.
    codebook = torch.einsum(
        "iv,jh->ijvh",
        vertical_vectors,
        horizontal_vectors,
    )
    return codebook.reshape(
        num_vertical_beams * num_horizontal_beams,
        num_vertical * num_horizontal,
    )
