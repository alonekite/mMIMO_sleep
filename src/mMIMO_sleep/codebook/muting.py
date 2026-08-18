"""Physical-port muting utilities."""

from __future__ import annotations

import torch

from mMIMO_sleep.array_config import ArrayConfig


def create_right_half_mask(
    config: ArrayConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a right-half muting mask for physical ports.

    The left half of horizontal columns (``[0, num_horizontal / 2)``) is kept
    active; the right half (``[num_horizontal / 2, num_horizontal)``) is muted.
    The mask is built using ``config.physical_port_index`` so it follows the
    configured ``port_order`` without duplicating the indexing formula.

    Args:
        config: Array configuration.
        dtype: Output dtype for the mask.
        device: PyTorch device for the returned tensor.

    Returns:
        Real tensor with shape ``(config.num_physical_ports,)``.  Active
        ports contain ``1``, muted ports contain ``0``.
    """
    if not isinstance(config, ArrayConfig):
        raise TypeError(f"config must be ArrayConfig, got {type(config)}.")
    if config.port_order != "polarization-major":
        raise NotImplementedError(
            "Only 'polarization-major' port_order is currently supported, "
            f"got {config.port_order!r}."
        )
    if config.num_horizontal % 2 != 0:
        raise ValueError(
            "num_horizontal must be even to mute exactly half of the panel, "
            f"got {config.num_horizontal}."
        )

    half = config.num_horizontal // 2
    mask = torch.zeros(
        config.num_physical_ports,
        dtype=dtype,
        device=device,
    )
    for pol in range(config.num_polarizations):
        for row in range(config.num_physical_rows):
            for col in range(half):
                idx = config.physical_port_index(pol, row, col)
                mask[idx] = 1
    return mask


def apply_muting_mask(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply an antenna mask without compensating power renormalization.

    Args:
        weights: Complex beam weights with shape ``(..., num_ports)``.
        mask: Real mask with shape ``(num_ports,)``.

    Returns:
        Elementwise product ``weights * mask`` with the same shape, dtype,
        and device as ``weights``.
    """
    if weights.ndim < 1:
        raise ValueError("weights must have at least one dimension.")
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {mask.shape}.")
    if weights.shape[-1] != mask.shape[0]:
        raise ValueError(
            "weights and mask have incompatible last dimensions: "
            f"{weights.shape[-1]} and {mask.shape[0]}."
        )
    return weights * mask.to(device=weights.device, dtype=weights.dtype)


def active_power_fraction(mask: torch.Tensor) -> float:
    """Return the fraction of active per-chain transmit power."""
    if mask.ndim != 1 or mask.numel() == 0:
        raise ValueError("mask must be a non-empty one-dimensional tensor.")
    return float(torch.mean(torch.abs(mask).square()).item())
