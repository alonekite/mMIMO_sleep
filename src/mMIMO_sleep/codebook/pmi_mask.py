"""PMI validity mask for the dual-polarized subarray panel.

The project panel has ``num_subarray_rows`` logical rows. Each logical
subarray contains ``elements_per_subarray`` vertically adjacent physical
elements. The two elements in one subarray use the same spatial beamforming
weight for a given polarization. Every physical position has two ideal
orthogonal polarizations, whose relative phase is selected by ``i2``.

For every spatial PMI ``(i11, i12)``, this module computes
``10*log10(max(EIRP_normal) / max(EIRP_sleep))``. The right half of the panel
is muted for both polarizations and the remaining weights are not
renormalized. The two polarization powers are added before the peak is taken.
Consequently the result is independent of ``i2`` under the ideal orthogonal
polarization assumption, so the mask is indexed only by ``[i12, i11]``.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import torch

from .dft import (
    generate_dual_polarized_subarray_codebook,
    unflatten_dual_polarized_weights,
)
from .muting import apply_muting_mask, create_right_half_mask


_TWO_PI: Final = 2.0 * torch.pi
_TR38901_THETA_3DB_DEG: Final = 65.0
_TR38901_PHI_3DB_DEG: Final = 65.0
_TR38901_SIDE_LOBE_ATTENUATION_DB: Final = 30.0
_TR38901_FRONT_BACK_ATTENUATION_DB: Final = 30.0

ElementPattern = Literal["tr38901", "isotropic"]


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _make_uv_grid(
    num_points: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return horizontal/vertical direction cosines and the visible disk."""
    _validate_positive_int("grid_points", num_points)
    if num_points < 3:
        raise ValueError("grid_points must be at least 3.")

    u = torch.linspace(-1.0, 1.0, num_points, dtype=dtype, device=device)
    u_v, u_h = torch.meshgrid(u, u, indexing="ij")
    visible = u_h.square() + u_v.square() <= 1.0
    return u_h, u_v, visible


def _tr38901_element_power_gain(
    u_h: torch.Tensor,
    u_v: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Return normalized TR 38.901 element power gain on the front hemisphere.

    The panel lies in the local y-z plane and points along local +x. Thus
    ``u_h`` is the local-y direction cosine and ``u_v`` is the local-z
    direction cosine. The maximum element gain is normalized to one because a
    common absolute gain cancels in the normal/sleep loss ratio.
    """
    real_dtype = u_h.dtype
    eps = torch.finfo(real_dtype).eps

    u_forward = torch.sqrt(
        (1.0 - u_h.square() - u_v.square()).clamp_min(0.0)
    )
    theta_deg = torch.rad2deg(torch.acos(u_v.clamp(-1.0, 1.0)))
    phi_deg = torch.rad2deg(torch.atan2(u_h, u_forward.clamp_min(eps)))

    vertical_attenuation_db = torch.minimum(
        12.0 * ((theta_deg - 90.0) / _TR38901_THETA_3DB_DEG).square(),
        torch.tensor(
            _TR38901_SIDE_LOBE_ATTENUATION_DB,
            dtype=real_dtype,
            device=u_h.device,
        ),
    )
    horizontal_attenuation_db = torch.minimum(
        12.0 * (phi_deg / _TR38901_PHI_3DB_DEG).square(),
        torch.tensor(
            _TR38901_FRONT_BACK_ATTENUATION_DB,
            dtype=real_dtype,
            device=u_h.device,
        ),
    )
    total_attenuation_db = torch.minimum(
        vertical_attenuation_db + horizontal_attenuation_db,
        torch.tensor(
            _TR38901_FRONT_BACK_ATTENUATION_DB,
            dtype=real_dtype,
            device=u_h.device,
        ),
    )
    gain = torch.pow(
        torch.tensor(10.0, dtype=real_dtype, device=u_h.device),
        -total_attenuation_db / 10.0,
    )
    return torch.where(visible, gain, torch.zeros_like(gain))


def _element_power_gain(
    pattern: ElementPattern,
    u_h: torch.Tensor,
    u_v: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    if pattern == "tr38901":
        return _tr38901_element_power_gain(u_h, u_v, visible)
    if pattern == "isotropic":
        return torch.where(
            visible,
            torch.ones_like(u_h),
            torch.zeros_like(u_h),
        )
    raise ValueError(
        "element_pattern must be 'tr38901' or 'isotropic', "
        f"got {pattern!r}."
    )


def compute_total_peak_loss_db(
    num_subarray_rows: int,
    num_horizontal: int,
    elements_per_subarray: int,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    *,
    num_polarization_phases: int = 4,
    phase_sign: int = 1,
    vertical_spacing_wavelengths: float = 0.5,
    horizontal_spacing_wavelengths: float = 0.5,
    element_pattern: ElementPattern = "tr38901",
    port_order: str = "polarization-major",
    grid_points: int = 121,
    beam_batch_size: int = 32,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute normal-to-sleep peak EIRP loss for every spatial PMI.

    ``loss_db[i12, i11]`` contains the loss for ``PMI(i11, i12)``. A complete
    dual-polarized physical-port codebook is generated so that the same
    subarray sharing, normalization and port ordering used by the link
    simulation are also applied here.

    Only the ``i2=0`` codeword is evaluated for each spatial PMI. This is exact
    for transmitted total power: both ideal orthogonal-polarization powers are
    added, and their relative phase cannot change the sum.
    """
    for name, value in (
        ("num_subarray_rows", num_subarray_rows),
        ("num_horizontal", num_horizontal),
        ("elements_per_subarray", elements_per_subarray),
        ("num_vertical_beams", num_vertical_beams),
        ("num_horizontal_beams", num_horizontal_beams),
        ("num_polarization_phases", num_polarization_phases),
        ("grid_points", grid_points),
        ("beam_batch_size", beam_batch_size),
    ):
        _validate_positive_int(name, value)
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3.")
    if phase_sign not in (-1, 1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign}.")
    _validate_positive_float(
        "vertical_spacing_wavelengths", vertical_spacing_wavelengths
    )
    _validate_positive_float(
        "horizontal_spacing_wavelengths", horizontal_spacing_wavelengths
    )
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"dtype must be complex64 or complex128, got {dtype}.")

    physical_rows = num_subarray_rows * elements_per_subarray
    num_spatial_beams = num_vertical_beams * num_horizontal_beams

    full_codebook = generate_dual_polarized_subarray_codebook(
        num_subarray_rows=num_subarray_rows,
        num_horizontal=num_horizontal,
        elements_per_subarray=elements_per_subarray,
        num_vertical_beams=num_vertical_beams,
        num_horizontal_beams=num_horizontal_beams,
        num_polarization_phases=num_polarization_phases,
        phase_sign=phase_sign,
        port_order=port_order,
        dtype=dtype,
        device=device,
    )
    expected_shape = (
        num_spatial_beams * num_polarization_phases,
        physical_rows * num_horizontal * 2,
    )
    if full_codebook.shape != expected_shape:
        raise RuntimeError(
            "Unexpected dual-polarized codebook shape: "
            f"expected {expected_shape}, got {tuple(full_codebook.shape)}."
        )

    # Ordering is (i12, i11, i2), with i2 varying fastest.
    normal_flat = full_codebook.reshape(
        num_spatial_beams,
        num_polarization_phases,
        -1,
    )[:, 0, :]

    right_half_mask = create_right_half_mask(
        physical_rows,
        num_horizontal,
        num_polarizations=2,
        port_order=port_order,
        dtype=normal_flat.real.dtype,
        device=device,
    )
    sleep_flat = apply_muting_mask(normal_flat, right_half_mask)

    normal = unflatten_dual_polarized_weights(
        normal_flat,
        num_rows=physical_rows,
        num_cols=num_horizontal,
        port_order=port_order,
    )
    sleep = unflatten_dual_polarized_weights(
        sleep_flat,
        num_rows=physical_rows,
        num_cols=num_horizontal,
        port_order=port_order,
    )

    real_dtype = normal.real.dtype
    u_h, u_v, visible = _make_uv_grid(
        grid_points,
        dtype=real_dtype,
        device=normal.device,
    )
    element_gain = _element_power_gain(
        element_pattern,
        u_h,
        u_v,
        visible,
    )

    row_index = torch.arange(
        physical_rows, dtype=real_dtype, device=normal.device
    )
    column_index = torch.arange(
        num_horizontal, dtype=real_dtype, device=normal.device
    )
    steering_v = torch.exp(
        -1j
        * _TWO_PI
        * vertical_spacing_wavelengths
        * u_v[..., None]
        * row_index
    )
    steering_h = torch.exp(
        -1j
        * _TWO_PI
        * horizontal_spacing_wavelengths
        * u_h[..., None]
        * column_index
    )

    peak_normal_parts: list[torch.Tensor] = []
    peak_sleep_parts: list[torch.Tensor] = []
    negative_infinity = torch.tensor(
        -torch.inf, dtype=real_dtype, device=normal.device
    )

    for start in range(0, num_spatial_beams, beam_batch_size):
        stop = min(start + beam_batch_size, num_spatial_beams)

        # Weights: [beam, row, column, polarization].
        # Fields:  [beam, polarization, u_v, u_h].
        field_normal = torch.einsum(
            "brcp,ijr,ijc->bpij",
            normal[start:stop],
            steering_v,
            steering_h,
        )
        field_sleep = torch.einsum(
            "brcp,ijr,ijc->bpij",
            sleep[start:stop],
            steering_v,
            steering_h,
        )

        # Add ideal orthogonal-polarization powers, never their fields.
        eirp_normal = field_normal.abs().square().sum(dim=1) * element_gain
        eirp_sleep = field_sleep.abs().square().sum(dim=1) * element_gain

        eirp_normal = torch.where(
            visible.unsqueeze(0), eirp_normal, negative_infinity
        )
        eirp_sleep = torch.where(
            visible.unsqueeze(0), eirp_sleep, negative_infinity
        )
        peak_normal_parts.append(eirp_normal.amax(dim=(1, 2)))
        peak_sleep_parts.append(eirp_sleep.amax(dim=(1, 2)))

    peak_normal = torch.cat(peak_normal_parts)
    peak_sleep = torch.cat(peak_sleep_parts)
    tiny = torch.finfo(real_dtype).tiny
    loss_db = 10.0 * torch.log10(
        peak_normal.clamp_min(tiny) / peak_sleep.clamp_min(tiny)
    )
    return loss_db.reshape(num_vertical_beams, num_horizontal_beams)


def create_total_loss_pmi_mask(
    num_subarray_rows: int,
    num_horizontal: int,
    elements_per_subarray: int,
    num_vertical_beams: int,
    num_horizontal_beams: int,
    *,
    target_loss_db: float = 6.020599913279624,
    tolerance_db: float = 1.0,
    **compute_kwargs: Any,
) -> torch.Tensor:
    """Return ``mask[i12, i11]`` for near-target peak EIRP loss.

    The default target is the exact half-aperture, no-renormalization value
    ``10*log10(4) = 6.0206 dB``. Remaining keyword arguments are forwarded to
    :func:`compute_total_peak_loss_db`.
    """
    if tolerance_db < 0.0:
        raise ValueError(
            f"tolerance_db must be non-negative, got {tolerance_db}."
        )
    loss_db = compute_total_peak_loss_db(
        num_subarray_rows=num_subarray_rows,
        num_horizontal=num_horizontal,
        elements_per_subarray=elements_per_subarray,
        num_vertical_beams=num_vertical_beams,
        num_horizontal_beams=num_horizontal_beams,
        **compute_kwargs,
    )
    return torch.abs(loss_db - target_loss_db) <= tolerance_db


def pmi_indices_from_mask(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Convert ``mask[i12, i11]`` to sorted ``(i11, i12)`` tuples."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {mask.shape}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}.")

    i12_values, i11_values = torch.where(mask)
    return [
        (int(i11), int(i12))
        for i12, i11 in zip(i12_values.tolist(), i11_values.tolist())
    ]


def project_default_pmi_mask(
    *,
    tolerance_db: float = 1.0,
    grid_points: int = 121,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create the mask for the project's 4x8 subarray / 8x8 physical panel."""
    return create_total_loss_pmi_mask(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_vertical_beams=8,
        num_horizontal_beams=32,
        num_polarization_phases=4,
        vertical_spacing_wavelengths=0.5,
        horizontal_spacing_wavelengths=0.5,
        element_pattern="tr38901",
        port_order="polarization-major",
        tolerance_db=tolerance_db,
        grid_points=grid_points,
        device=device,
    )
