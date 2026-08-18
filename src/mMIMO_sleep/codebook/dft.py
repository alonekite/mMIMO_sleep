"""DFT codebook generation for a uniform planar array (UPA).

The codebook is ordered as ``beam_index = ((i12 * num_horizontal_beams) +
i11) * num_i2 + i2``, where ``i12`` is the vertical beam index, ``i11`` is
the horizontal beam index, and ``i2`` is the co-phasing index.  ``i2``
varies fastest.

Physical ports are stored in polarization-major order: all polarization-0
ports precede all polarization-1 ports.  Within each polarization, elements
are stored in row-major order with the horizontal index varying fastest.
"""

from __future__ import annotations

from typing import Final

import torch

from mMIMO_sleep.array_config import ArrayConfig

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
    config: ArrayConfig,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    num_i2: int = 4,
    *,
    phase_sign: int = 1,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a dual-polarized subarray DFT codebook.

    The codebook contains separable two-dimensional DFT weights defined on a
    logical subarray grid.  Each logical subarray weight is expanded vertically
    to ``config.elements_per_subarray`` physical elements and duplicated for
    both polarizations, with polarization 1 applying an ``i2`` co-phasing
    rotation relative to polarization 0.

    Args:
        config: Array configuration.  Only ``num_polarizations=2`` is
            currently supported.
        num_vertical_beams: Number of vertical DFT beams.
        num_horizontal_beams: Number of horizontal DFT beams.
        num_i2: Number of co-phasing values between the two polarizations.
        phase_sign: Sign convention for the DFT phase progression.
        dtype: Complex output dtype.
        device: PyTorch device for the returned tensor.

    Returns:
        Complex tensor with shape
        ``(num_vertical_beams * num_horizontal_beams * num_i2,
        config.num_physical_ports)``.  Beam ordering is ``(i12, i11, i2)``
        with ``i2`` varying fastest.  Ports are ordered in polarization-major
        row-major order.
    """
    if not isinstance(config, ArrayConfig):
        raise TypeError(f"config must be ArrayConfig, got {type(config)}.")
    if config.num_polarizations != 2:
        raise NotImplementedError(
            "Only num_polarizations=2 is currently supported, "
            f"got {config.num_polarizations}."
        )
    if config.port_order != "polarization-major":
        raise NotImplementedError(
            "Only 'polarization-major' port_order is currently supported, "
            f"got {config.port_order!r}."
        )
    for name, value in (
        ("num_vertical_beams", num_vertical_beams),
        ("num_horizontal_beams", num_horizontal_beams),
        ("num_i2", num_i2),
    ):
        _validate_positive(name, value)
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    if dtype not in _COMPLEX_DTYPES:
        raise TypeError(f"dtype must be complex64 or complex128, got {dtype}.")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    eps = config.elements_per_subarray

    # One-dimensional DFT steering vectors on the logical subarray grid.
    vertical_vectors = torch.stack(
        [
            dft_vector(
                config.num_subarray_rows,
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
                config.num_horizontal,
                i11,
                num_horizontal_beams,
                phase_sign=phase_sign,
                dtype=dtype,
                device=device,
            )
            for i11 in range(num_horizontal_beams)
        ]
    )

    # Logical 2D weights: [num_vertical_beams, num_horizontal_beams,
    # num_subarray_rows, num_horizontal].
    logical_weights = torch.einsum(
        "iv,jh->ijvh",
        vertical_vectors,
        horizontal_vectors,
    )

    # Expand each logical subarray row to ``eps`` physical rows.
    physical_weights = logical_weights.repeat_interleave(eps, dim=2)

    # Flatten physical elements in row-major order (horizontal fastest) and
    # normalize so that each polarization carries ``1/num_polarizations``
    # of the total power.
    scale = 1.0 / torch.sqrt(
        torch.tensor(
            eps * config.num_polarizations,
            dtype=real_dtype,
            device=device,
        )
    )
    physical_weights = physical_weights.reshape(
        num_vertical_beams,
        num_horizontal_beams,
        config.num_physical_elements,
    )
    physical_weights = physical_weights * scale

    # i2 co-phasing values applied to polarization 1.
    i2_index = torch.arange(
        num_i2,
        dtype=real_dtype,
        device=device,
    )
    i2_phase = torch.exp(1j * 2.0 * torch.pi * i2_index / num_i2).to(dtype=dtype)

    # [num_vertical_beams, num_horizontal_beams, num_i2, num_physical_elements]
    pol0 = physical_weights.unsqueeze(2).expand(
        num_vertical_beams,
        num_horizontal_beams,
        num_i2,
        config.num_physical_elements,
    )
    pol1 = pol0 * i2_phase.view(num_i2, 1)

    # Polarization-major ordering.
    codebook = torch.cat([pol0, pol1], dim=-1)

    # Beam ordering (i12, i11, i2), i2 fastest.
    return codebook.reshape(
        num_vertical_beams * num_horizontal_beams * num_i2,
        config.num_physical_ports,
    )
