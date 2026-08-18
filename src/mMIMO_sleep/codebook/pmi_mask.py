"""PMI mask utilities based on total peak array-factor power loss.

A "PMI mask" is a boolean tensor indexed by ``(i12, i11)`` that selects the
spatial beam indices whose total peak array-factor power loss is close to a
target value (default ``10*log10(4)`` dB for right-half panel muting).

Only array-factor physics is modelled here; element patterns are deliberately
excluded and handled by downstream Sionna-RT integration.
"""

from __future__ import annotations

import math
from typing import Final

import torch

from mMIMO_sleep.array_config import ArrayConfig

from .dft import generate_dft_codebook
from .muting import apply_muting_mask, create_right_half_mask
from .pmi import PMI, pmi_to_beam_index

IDEAL_HALF_APERTURE_LOSS_DB: Final = 10.0 * math.log10(4.0)

_ALLOWED_COMPLEX_DTYPES: Final = (torch.complex64, torch.complex128)


def _strict_positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_common_args(
    config: ArrayConfig,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    num_i2: int,
    phase_sign: int,
    spacing_wavelengths: float,
    grid_points: int,
    beam_batch_size: int | None,
    dtype: torch.dtype,
) -> None:
    if not isinstance(config, ArrayConfig):
        raise TypeError(f"config must be ArrayConfig, got {type(config)}.")
    _strict_positive_int("num_vertical_beams", num_vertical_beams)
    _strict_positive_int("num_horizontal_beams", num_horizontal_beams)
    _strict_positive_int("num_i2", num_i2)
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    if not isinstance(spacing_wavelengths, (int, float)):
        raise TypeError(
            f"spacing_wavelengths must be a real number, got {type(spacing_wavelengths)}."
        )
    if not math.isfinite(spacing_wavelengths) or spacing_wavelengths <= 0:
        raise ValueError(
            f"spacing_wavelengths must be a positive finite number, "
            f"got {spacing_wavelengths}."
        )
    _strict_positive_int("grid_points", grid_points)
    if grid_points < 2:
        raise ValueError(f"grid_points must be at least 2, got {grid_points}.")
    if beam_batch_size is not None:
        _strict_positive_int("beam_batch_size", beam_batch_size)
    if dtype not in _ALLOWED_COMPLEX_DTYPES:
        raise TypeError(
            f"dtype must be torch.complex64 or torch.complex128, got {dtype}."
        )


def _make_visible_uv_grid(
    grid_points: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(u_h, u_v, visible)`` direction-cosine grids.

    ``visible`` is True exactly where ``u_h**2 + u_v**2 <= 1``.
    """
    u = torch.linspace(-1.0, 1.0, grid_points, dtype=dtype, device=device)
    u_h, u_v = torch.meshgrid(u, u, indexing="ij")
    visible = u_h.square() + u_v.square() <= 1.0
    return u_h, u_v, visible


def _array_factor_from_weights(
    weights_2d: torch.Tensor,
    u_h: torch.Tensor,
    u_v: torch.Tensor,
    spacing_wavelengths: float,
) -> torch.Tensor:
    """Compute array factor for arbitrary leading batch dimensions.

    ``weights_2d`` has shape ``(..., num_physical_rows, num_horizontal)``.
    ``u_h`` and ``u_v`` are one-dimensional tensors of visible direction cosines.

    Returns a tensor of shape ``(..., num_visible_directions)`` using the
    convention

        AF(u_h, u_v) = Σ_v Σ_h w[v,h] *
                       exp(-j*2π*d*(u_h*h + u_v*v))
    """
    if weights_2d.ndim < 2:
        raise ValueError("weights_2d must have at least two dimensions.")
    if u_h.ndim != 1 or u_v.ndim != 1 or u_h.shape != u_v.shape:
        raise ValueError("u_h and u_v must be one-dimensional with the same shape.")
    real_dtype = weights_2d.real.dtype
    num_physical_rows = weights_2d.shape[-2]
    num_horizontal = weights_2d.shape[-1]
    n_v = torch.arange(
        num_physical_rows, dtype=real_dtype, device=weights_2d.device
    )
    n_h = torch.arange(
        num_horizontal, dtype=real_dtype, device=weights_2d.device
    )
    u_h = u_h.to(device=weights_2d.device, dtype=real_dtype)
    u_v = u_v.to(device=weights_2d.device, dtype=real_dtype)
    phase_scale = -2.0 * math.pi * spacing_wavelengths
    steering_v = torch.exp(phase_scale * 1j * u_v.unsqueeze(-1) * n_v)
    steering_h = torch.exp(phase_scale * 1j * u_h.unsqueeze(-1) * n_h)
    return torch.einsum("...vh,nv,nh->...n", weights_2d, steering_v, steering_h)


def _total_power_from_polarized_fields(
    field_pol0: torch.Tensor,
    field_pol1: torch.Tensor,
) -> torch.Tensor:
    """Return the incoherent total power of two polarized array-factor fields."""
    if field_pol0.shape != field_pol1.shape:
        raise ValueError(
            "field_pol0 and field_pol1 must have the same shape, "
            f"got {field_pol0.shape} and {field_pol1.shape}."
        )
    return field_pol0.abs().square() + field_pol1.abs().square()


def _compute_loss_at_normal_peak_db(
    normal_power: torch.Tensor,
    sleep_power: torch.Tensor,
) -> torch.Tensor:
    """Compute peak loss using sleep power at the normal-power argmax.

    ``normal_power`` and ``sleep_power`` have shape ``(..., num_directions)``.
    Returns ``10*log10(normal_peak / sleep_at_peak)`` with the same leading
    shape ``(...)``.

    If ``sleep_at_peak`` is zero while ``normal_peak`` is positive, the
    corresponding loss is ``+inf``.
    """
    if normal_power.shape != sleep_power.shape:
        raise ValueError(
            "normal_power and sleep_power must have the same shape, "
            f"got {normal_power.shape} and {sleep_power.shape}."
        )
    if normal_power.ndim < 1 or normal_power.shape[-1] == 0:
        raise ValueError(
            "normal_power/sleep_power must have a non-empty last dimension."
        )
    argmax_idx = normal_power.argmax(dim=-1)
    normal_peak = normal_power.gather(-1, argmax_idx.unsqueeze(-1)).squeeze(-1)
    sleep_at_peak = sleep_power.gather(-1, argmax_idx.unsqueeze(-1)).squeeze(-1)
    pos_inf = torch.tensor(
        float("inf"), dtype=normal_peak.dtype, device=normal_peak.device
    )
    nan = torch.tensor(
        float("nan"), dtype=normal_peak.dtype, device=normal_peak.device
    )
    ratio = torch.where(
        sleep_at_peak > 0,
        normal_peak / sleep_at_peak,
        torch.where(normal_peak > 0, pos_inf, nan),
    )
    return 10.0 * torch.log10(ratio)


def _mask_from_loss(
    loss_db: torch.Tensor,
    target_loss_db: float,
    tolerance_db: float,
) -> torch.Tensor:
    """Return True where ``|loss_db - target_loss_db| <= tolerance_db``.

    Non-finite ``loss_db`` entries are always marked False.
    """
    if not isinstance(target_loss_db, (int, float)):
        raise TypeError(
            f"target_loss_db must be a real number, got {type(target_loss_db)}."
        )
    if not isinstance(tolerance_db, (int, float)):
        raise TypeError(
            f"tolerance_db must be a real number, got {type(tolerance_db)}."
        )
    if not math.isfinite(target_loss_db):
        raise ValueError(
            f"target_loss_db must be finite, got {target_loss_db}."
        )
    if not math.isfinite(tolerance_db) or tolerance_db < 0:
        raise ValueError(
            f"tolerance_db must be finite and non-negative, got {tolerance_db}."
        )
    return torch.isfinite(loss_db) & (
        torch.abs(loss_db - target_loss_db) <= tolerance_db
    )


def compute_total_peak_loss_db(
    config: ArrayConfig,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    num_i2: int = 4,
    *,
    phase_sign: int = 1,
    spacing_wavelengths: float = 0.5,
    grid_points: int = 121,
    beam_batch_size: int | None = None,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute total peak array-factor power loss in dB for every spatial PMI.

    ``total_peak_loss_db[i12, i11]`` is
    ``10 * log10(P_normal(peak_dir) / P_sleep(peak_dir))`` where ``peak_dir``
    is the direction that maximizes the normal-state total power.  Both
    polarizations are included in the total power.

    Args:
        config: Array configuration; single source of truth for topology and
            port ordering.
        num_vertical_beams: Number of vertical DFT beams.
        num_horizontal_beams: Number of horizontal DFT beams.
        num_i2: Number of cross-polarization co-phasing values.
        phase_sign: DFT phase sign convention (+1 or -1).
        spacing_wavelengths: Element spacing in wavelengths.
        grid_points: Number of points along each direction-cosine axis.
        beam_batch_size: Number of spatial beams to process per batch;
            ``None`` selects a small default.
        dtype: Complex dtype for the codebook and array-factor computation.
        device: Torch device for the computation.

    Returns:
        Real tensor with shape ``(num_vertical_beams, num_horizontal_beams)``.
    """
    _validate_common_args(
        config,
        num_vertical_beams,
        num_horizontal_beams,
        num_i2,
        phase_sign,
        spacing_wavelengths,
        grid_points,
        beam_batch_size,
        dtype,
    )

    if device is not None:
        device = torch.device(device)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    codebook = generate_dft_codebook(
        config,
        num_vertical_beams,
        num_horizontal_beams,
        num_i2,
        phase_sign=phase_sign,
        dtype=dtype,
        device=device,
    )
    muting_mask = create_right_half_mask(
        config, dtype=real_dtype, device=device
    )
    sleep_codebook = apply_muting_mask(codebook, muting_mask)

    # One representative i2 block is sufficient because the total-power pattern
    # is independent of i2 for ideal dual-polarized elements.
    spatial_codebook = codebook[0::num_i2]
    sleep_spatial_codebook = sleep_codebook[0::num_i2]

    u_h, u_v, visible = _make_visible_uv_grid(
        grid_points, dtype=real_dtype, device=device
    )
    u_h_visible = u_h[visible]
    u_v_visible = u_v[visible]

    num_elements = config.num_physical_elements
    num_physical_rows = config.num_physical_rows
    num_horizontal = config.num_horizontal

    total_beams = num_vertical_beams * num_horizontal_beams
    if beam_batch_size is None:
        beam_batch_size = min(32, total_beams)

    loss_flat = torch.empty(total_beams, dtype=real_dtype, device=device)

    for start in range(0, total_beams, beam_batch_size):
        end = min(start + beam_batch_size, total_beams)

        chunk_normal = spatial_codebook[start:end]
        chunk_sleep = sleep_spatial_codebook[start:end]

        pol0_n = chunk_normal[:, :num_elements]
        pol1_n = chunk_normal[:, num_elements:]
        pol0_s = chunk_sleep[:, :num_elements]
        pol1_s = chunk_sleep[:, num_elements:]

        pol0_n_2d = pol0_n.reshape(-1, num_physical_rows, num_horizontal)
        pol1_n_2d = pol1_n.reshape(-1, num_physical_rows, num_horizontal)
        pol0_s_2d = pol0_s.reshape(-1, num_physical_rows, num_horizontal)
        pol1_s_2d = pol1_s.reshape(-1, num_physical_rows, num_horizontal)

        e0_n = _array_factor_from_weights(
            pol0_n_2d, u_h_visible, u_v_visible, spacing_wavelengths
        )
        e1_n = _array_factor_from_weights(
            pol1_n_2d, u_h_visible, u_v_visible, spacing_wavelengths
        )
        e0_s = _array_factor_from_weights(
            pol0_s_2d, u_h_visible, u_v_visible, spacing_wavelengths
        )
        e1_s = _array_factor_from_weights(
            pol1_s_2d, u_h_visible, u_v_visible, spacing_wavelengths
        )

        p_n = _total_power_from_polarized_fields(e0_n, e1_n)
        p_s = _total_power_from_polarized_fields(e0_s, e1_s)

        loss_flat[start:end] = _compute_loss_at_normal_peak_db(p_n, p_s)

    return loss_flat.reshape(num_vertical_beams, num_horizontal_beams)


def create_total_loss_pmi_mask(
    config: ArrayConfig,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    num_i2: int = 4,
    *,
    target_loss_db: float = IDEAL_HALF_APERTURE_LOSS_DB,
    tolerance_db: float = 1.0,
    phase_sign: int = 1,
    spacing_wavelengths: float = 0.5,
    grid_points: int = 121,
    beam_batch_size: int | None = None,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a boolean PMI mask for near-target total peak loss.

    A spatial PMI ``(i11, i12)`` is marked ``True`` when
    ``|total_peak_loss_db[i12, i11] - target_loss_db| <= tolerance_db``.

    Args:
        target_loss_db: Desired total peak loss in dB.
        tolerance_db: Acceptable deviation from ``target_loss_db`` in dB.
        Other arguments: see :func:`compute_total_peak_loss_db`.

    Returns:
        Boolean tensor with shape ``(num_vertical_beams, num_horizontal_beams)``.
    """
    if not isinstance(target_loss_db, (int, float)):
        raise TypeError(
            f"target_loss_db must be a real number, got {type(target_loss_db)}."
        )
    if not isinstance(tolerance_db, (int, float)):
        raise TypeError(
            f"tolerance_db must be a real number, got {type(tolerance_db)}."
        )
    if not math.isfinite(target_loss_db):
        raise ValueError(f"target_loss_db must be finite, got {target_loss_db}.")
    if not math.isfinite(tolerance_db) or tolerance_db < 0:
        raise ValueError(
            f"tolerance_db must be finite and non-negative, got {tolerance_db}."
        )

    loss_db = compute_total_peak_loss_db(
        config,
        num_vertical_beams,
        num_horizontal_beams,
        num_i2,
        phase_sign=phase_sign,
        spacing_wavelengths=spacing_wavelengths,
        grid_points=grid_points,
        beam_batch_size=beam_batch_size,
        dtype=dtype,
        device=device,
    )
    return _mask_from_loss(loss_db, target_loss_db, tolerance_db)


def pmi_indices_from_mask(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Convert a PMI mask to a list of ``(i11, i12)`` tuples.

    The returned tuples are sorted first by ``i12`` (row), then by ``i11``
    (column).
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got shape {mask.shape}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}.")
    rows, cols = torch.where(mask)
    return [(int(c), int(r)) for r, c in zip(rows.tolist(), cols.tolist())]


def beam_indices_from_mask(
    mask: torch.Tensor,
    *,
    num_i2: int,
) -> list[int]:
    """Expand every selected spatial PMI to all i2 values.

    Returns flat codebook indices ordered as ``(i12, i11, i2)`` with ``i2``
    varying fastest.  The conversion uses :func:`pmi_to_beam_index` so the
    flat-index formula is not duplicated here.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got shape {mask.shape}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}.")
    if type(num_i2) is not int:
        raise TypeError(f"num_i2 must be an int, got {type(num_i2)}.")
    if num_i2 <= 0:
        raise ValueError(f"num_i2 must be positive, got {num_i2}.")

    num_vertical_beams = mask.shape[0]
    num_horizontal_beams = mask.shape[1]
    indices: list[int] = []
    for i11, i12 in pmi_indices_from_mask(mask):
        for i2 in range(num_i2):
            indices.append(
                pmi_to_beam_index(
                    PMI(i11=i11, i12=i12, i2=i2),
                    num_horizontal_beams=num_horizontal_beams,
                    num_vertical_beams=num_vertical_beams,
                    num_i2=num_i2,
                )
            )
    return indices
