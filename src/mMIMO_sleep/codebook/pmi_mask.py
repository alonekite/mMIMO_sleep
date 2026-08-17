"""PMI mask utilities based on total peak effective-power loss.

A "PMI mask" is a boolean tensor indexed by ``(i12, i11)`` that selects the
beam indices whose total peak effective-power loss is close to a target
value (default 6 dB for right-half panel muting).
"""

from __future__ import annotations

from typing import Final

import torch

from .dft import generate_dft_codebook
from .muting import apply_muting_mask, create_right_half_mask

_TWO_PI: Final = 2.0 * torch.pi


def _make_uv_grid(
    num_points: int,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(u_h, u_v, visible)`` direction-cosine grids."""
    u = torch.linspace(-1.0, 1.0, num_points, device=device)
    u_v, u_h = torch.meshgrid(u, u, indexing="ij")
    visible = u_h.square() + u_v.square() <= 1.0
    return u_h, u_v, visible


def compute_total_peak_loss_db(
    num_vertical: int,
    num_horizontal: int,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    *,
    phase_sign: int = 1,
    spacing_wavelengths: float = 0.5,
    grid_points: int = 121,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute total peak effective-power loss in dB for every PMI.

    ``total_peak_loss_db[i12, i11]`` is
    ``10 * log10(peak_power_normal / peak_power_sleep)`` for the beam
    ``PMI(i11, i12)``.  The peak is taken over the visible direction-cosine
    disk.

    Args:
        num_vertical: Number of vertical antenna elements.
        num_horizontal: Number of horizontal antenna elements.
        num_vertical_beams: Number of vertical DFT beams.
        num_horizontal_beams: Number of horizontal DFT beams.
        phase_sign: DFT phase sign convention (+1 or -1).
        spacing_wavelengths: Element spacing in wavelengths.
        grid_points: Number of points along each direction-cosine axis.
        device: Torch device for the computation.

    Returns:
        Tensor with shape ``(num_vertical_beams, num_horizontal_beams)``.
    """
    codebook = generate_dft_codebook(
        num_vertical,
        num_horizontal,
        num_vertical_beams,
        num_horizontal_beams,
        phase_sign=phase_sign,
        device=device,
    )
    right_half_mask = create_right_half_mask(
        num_vertical, num_horizontal, device=device
    )
    sleep_codebook = apply_muting_mask(codebook, right_half_mask)

    u_h, u_v, visible = _make_uv_grid(grid_points, device)

    weights_2d = codebook.reshape(-1, num_vertical, num_horizontal)
    sleep_weights_2d = sleep_codebook.reshape(-1, num_vertical, num_horizontal)

    real_dtype = weights_2d.real.dtype
    n_v = torch.arange(num_vertical, device=weights_2d.device, dtype=real_dtype)
    n_h = torch.arange(num_horizontal, device=weights_2d.device, dtype=real_dtype)
    phase_scale = -_TWO_PI * spacing_wavelengths
    steering_v = torch.exp(1j * phase_scale * u_v[..., None] * n_v)
    steering_h = torch.exp(1j * phase_scale * u_h[..., None] * n_h)

    field_normal = torch.einsum(
        "bvh,ijv,ijh->bij", weights_2d, steering_v, steering_h
    )
    field_sleep = torch.einsum(
        "bvh,ijv,ijh->bij", sleep_weights_2d, steering_v, steering_h
    )

    power_normal = field_normal.abs().square()
    power_sleep = field_sleep.abs().square()

    num_beams = num_vertical_beams * num_horizontal_beams
    visible_expanded = visible.unsqueeze(0).expand(num_beams, -1, -1)
    sentinel = torch.tensor(-1.0, device=power_normal.device, dtype=power_normal.dtype)
    peak_normal = torch.where(visible_expanded, power_normal, sentinel).amax(dim=(1, 2))
    peak_sleep = torch.where(visible_expanded, power_sleep, sentinel).amax(dim=(1, 2))

    ratio = peak_normal / peak_sleep.clamp_min(1e-12)
    loss_db = 10.0 * torch.log10(ratio.clamp_min(1e-12))
    return loss_db.reshape(num_vertical_beams, num_horizontal_beams)


def create_total_loss_pmi_mask(
    num_vertical: int,
    num_horizontal: int,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    *,
    target_loss_db: float = 6.0,
    tolerance_db: float = 1.0,
    phase_sign: int = 1,
    spacing_wavelengths: float = 0.5,
    grid_points: int = 121,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a boolean PMI mask for near-target total peak loss.

    A PMI ``(i11, i12)`` is marked ``True`` when
    ``|total_peak_loss_db[i12, i11] - target_loss_db| <= tolerance_db``.

    Args:
        target_loss_db: Desired total peak loss in dB.
        tolerance_db: Acceptable deviation from ``target_loss_db`` in dB.
        Other arguments: see :func:`compute_total_peak_loss_db`.

    Returns:
        Boolean tensor with shape ``(num_vertical_beams, num_horizontal_beams)``.
    """
    loss_db = compute_total_peak_loss_db(
        num_vertical=num_vertical,
        num_horizontal=num_horizontal,
        num_vertical_beams=num_vertical_beams,
        num_horizontal_beams=num_horizontal_beams,
        phase_sign=phase_sign,
        spacing_wavelengths=spacing_wavelengths,
        grid_points=grid_points,
        device=device,
    )
    return torch.abs(loss_db - target_loss_db) <= tolerance_db


def pmi_indices_from_mask(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Convert a PMI mask to a list of ``(i11, i12)`` tuples.

    The returned tuples are sorted first by ``i12``, then by ``i11``.
    """
    rows, cols = torch.where(mask)
    return [(int(c), int(r)) for r, c in zip(rows.tolist(), cols.tolist())]
