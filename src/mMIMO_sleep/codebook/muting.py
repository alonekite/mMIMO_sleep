"""Half-panel muting utilities for row-major UPA beam weights."""

from __future__ import annotations

import torch


def create_right_half_mask(
    num_vertical: int,
    num_horizontal: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a mask that disables the right half of a rectangular panel.

    The returned mask follows the same row-major element order as ``dft.py``:
    horizontal element index varies fastest.

    Returns:
        Tensor with shape ``[num_vertical * num_horizontal]`` containing
        ones for active elements and zeros for muted elements.
    """
    if num_vertical <= 0:
        raise ValueError("num_vertical must be positive.")
    if num_horizontal <= 0:
        raise ValueError("num_horizontal must be positive.")
    if num_horizontal % 2 != 0:
        raise ValueError(
            "num_horizontal must be even to mute exactly half of the panel."
        )

    mask = torch.ones(
        (num_vertical, num_horizontal),
        dtype=dtype,
        device=device,
    )
    mask[:, num_horizontal // 2 :] = 0
    return mask.flatten()


def apply_muting_mask(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply an antenna mask without compensating power renormalization.

    Args:
        weights: Complex beam weights with shape ``[..., num_tx]``.
        mask: Real or complex mask with shape ``[num_tx]``.

    Returns:
        Muted weights with the same shape, dtype, and device as ``weights``.
    """
    if weights.ndim < 1:
        raise ValueError("weights must have at least one dimension.")
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {mask.shape}.")
    if weights.shape[-1] != mask.shape[0]:
        raise ValueError(
            "weights and mask have incompatible antenna dimensions: "
            f"{weights.shape[-1]} and {mask.shape[0]}."
        )
    return weights * mask.to(device=weights.device, dtype=weights.dtype)


def active_power_fraction(mask: torch.Tensor) -> float:
    """Return the fraction of active per-chain transmit power."""
    if mask.ndim != 1 or mask.numel() == 0:
        raise ValueError("mask must be a non-empty one-dimensional tensor.")
    return float(torch.mean(torch.abs(mask).square()).item())
