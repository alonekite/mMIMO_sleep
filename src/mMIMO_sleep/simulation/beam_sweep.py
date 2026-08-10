"""Beam-sweep operations using a precomputed MIMO channel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BeamSelection:
    """Result of selecting one or more best codebook beams."""

    beam_indices: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


def compute_effective_channels(
    channel: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Apply every codebook beam to a channel tensor.

    Args:
        channel: Complex tensor with shape ``[..., num_rx, num_tx]``.
        codebook: Complex tensor with shape ``[num_beams, num_tx]``.

    Returns:
        Effective channels with shape ``[..., num_rx, num_beams]``.
    """
    if channel.ndim < 2:
        raise ValueError(
            "channel must have shape [..., num_rx, num_tx]."
        )
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [num_beams, num_tx].")
    if channel.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            "channel and codebook have incompatible transmit dimensions: "
            f"{channel.shape[-1]} and {codebook.shape[-1]}."
        )
    if not torch.is_complex(channel) or not torch.is_complex(codebook):
        raise TypeError("channel and codebook must both be complex tensors.")

    codebook = codebook.to(device=channel.device, dtype=channel.dtype)
    return torch.einsum("...rt,bt->...rb", channel, codebook)


def compute_beam_power(
    channel: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Compute ``|H w_b|^2`` for every beam.

    The output shape is ``[..., num_rx, num_beams]``.
    """
    return compute_effective_channels(channel, codebook).abs().square()


def select_best_beam(
    channel: torch.Tensor,
    codebook: torch.Tensor,
    *,
    reduce_dims: Sequence[int] | None = None,
    reduction: str = "mean",
) -> BeamSelection:
    """Select the strongest wideband rank-one codebook beam.

    ``reduce_dims`` identifies dimensions of the beam-power tensor to combine
    before taking ``argmax`` over its last (beam) dimension. For example:

    * ``H[Nrx, Ntx]``: use ``reduce_dims=(0,)``.
    * ``H[K, Nrx, Ntx]``: use ``reduce_dims=(0, 1)``.
    * ``H[U, K, Nrx, Ntx]``: use ``reduce_dims=(1, 2)`` to obtain one beam per UE.

    When ``reduce_dims`` is omitted, only the receive-antenna dimension is
    reduced. This is convenient for narrowband channels with optional batches.
    """
    power = compute_beam_power(channel, codebook)
    if reduce_dims is None:
        reduce_dims = (power.ndim - 2,)

    normalized_dims = tuple(
        dim if dim >= 0 else power.ndim + dim for dim in reduce_dims
    )
    if not normalized_dims:
        raise ValueError("reduce_dims must contain at least one dimension.")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError("reduce_dims contains duplicate dimensions.")
    if any(dim < 0 or dim >= power.ndim - 1 for dim in normalized_dims):
        raise ValueError(
            "reduce_dims must refer to non-beam dimensions of the power tensor."
        )

    if reduction == "mean":
        scores = power.mean(dim=normalized_dims)
    elif reduction == "sum":
        scores = power.sum(dim=normalized_dims)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    beam_indices = scores.argmax(dim=-1)
    selected_weights = codebook.to(
        device=channel.device,
        dtype=channel.dtype,
    )[beam_indices]
    return BeamSelection(
        beam_indices=beam_indices,
        weights=selected_weights,
        scores=scores,
    )
