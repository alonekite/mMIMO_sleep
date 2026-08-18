"""Sionna RT array interface: explicit conversion between project conventions
and Sionna RT ``PlanarArray`` / precoding vector ordering.

The conversion functions in this module form the **single boundary** where
project-specific ``ArrayConfig`` / codebook weights are mapped to Sionna RT.
If Sionna RT's internal ordering ever changes, or if a different port ordering
is chosen in ``ArrayConfig``, the conversion logic is isolated here.
"""

from __future__ import annotations

import torch

from mMIMO_sleep.array_config import ArrayConfig


# Sionna RT numbers physical elements column-first (vertical/row varies fastest).
# The project codebook uses row-major (horizontal/col varies fastest) within
# each polarization.  This helper maps a project element index to a Sionna
# element index, isolated in this module so the conversion is explicit.
def _project_to_sionna_element_index(config: ArrayConfig, project_index: int) -> int:
    row = project_index // config.num_horizontal
    col = project_index % config.num_horizontal
    return col * config.num_physical_rows + row


def _sionna_to_project_element_index(config: ArrayConfig, sionna_index: int) -> int:
    col = sionna_index // config.num_physical_rows
    row = sionna_index % config.num_physical_rows
    return row * config.num_horizontal + col


def _require_sionna_rt():
    """Lazy import helper to avoid hard dependency on Sionna RT."""
    try:
        import sionna.rt as rt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Sionna RT is required. Install with: pip install -e '.[dev,rt]'"
        ) from exc
    return rt


def array_config_to_planar_array(
    config: ArrayConfig,
    *,
    pattern: str = "iso",
    polarization: str = "VH",
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
):
    """Create a Sionna RT ``PlanarArray`` from ``ArrayConfig``.

    Sionna RT places the array in the y-z plane and numbers elements
    column-first (row/vertical varies fastest) from top-left to bottom-right.
    The project codebook uses row-major (column/horizontal varies fastest)
    within each polarization.  The conversion is performed in
    ``weights_to_sionna_precoding``.

    ``polarization="VH"`` creates two linear patterns: pattern 0 = V,
    pattern 1 = H.  Sionna RT stores the precoding vector in pattern-major
    order, which again matches the project polarization-major ordering.
    """
    rt = _require_sionna_rt()
    # Sionna RT PlanarArray models physical elements and uses column-first
    # (vertical/row varies fastest) element ordering.  Project codebook weights
    # use row-major (horizontal/col varies fastest).  The conversion is applied
    # in ``weights_to_sionna_precoding``.
    return rt.PlanarArray(
        num_rows=config.num_physical_rows,
        num_cols=config.num_horizontal,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        pattern=pattern,
        polarization=polarization,
    )


def weights_to_sionna_precoding(
    weights: torch.Tensor,
    config: ArrayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert project codebook weights to Sionna RT ``precoding_vec`` format.

    Project ordering per polarization: row-major (col varies fastest).
    Sionna RT element ordering: column-first (row varies fastest).  This
    function reorders within each polarization block so the weight index
    corresponds to the correct Sionna RT antenna port.

    Sionna RT expects ``(real, imag)`` components.  The returned PyTorch
    tensors must be converted to ``mi.TensorXf`` by the caller.
    """
    if weights.shape[-1] != config.num_physical_ports:
        raise ValueError(
            f"weights last dimension {weights.shape[-1]} does not match "
            f"ArrayConfig physical ports {config.num_physical_ports}"
        )

    num_elements = config.num_physical_elements
    reordered = []
    # Build the inverse mapping: for each Sionna element index, which project
    # element index supplies its weight.
    inverse_mapping = torch.tensor(
        [
            _sionna_to_project_element_index(config, i)
            for i in range(num_elements)
        ],
        dtype=torch.long,
        device=weights.device,
    )
    for pol in range(config.num_polarizations):
        start = pol * num_elements
        end = start + num_elements
        pol_weights = weights[..., start:end]
        # reordered[..., sionna_index] = pol_weights[..., project_index]
        reordered.append(pol_weights[..., inverse_mapping])
    reordered = torch.cat(reordered, dim=-1)
    return reordered.real, reordered.imag
