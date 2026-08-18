"""Single source of truth for antenna array configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArrayConfig:
    """Antenna array configuration shared across codebook and Sionna RT.

    The array is modelled as a grid of logical subarrays.  Each subarray is a
    vertical stack of ``elements_per_subarray`` physical elements sharing the
    same spatial beam weight.  The horizontal dimension of a subarray is one
    physical column.

    Attributes:
        num_subarray_rows: Number of logical subarray rows.
        num_horizontal: Number of physical columns (and logical subarray
            columns).
        elements_per_subarray: Number of vertical physical elements per
            logical subarray.  Allowed values are 1, 2, 3, 4.
        num_polarizations: Number of polarizations per physical element.
        port_order: String identifying the physical-port ordering.  Only
            ``"polarization-major"`` is currently supported; it is kept as an
            explicit configurable parameter so that Sionna RT integration
            can introduce a mapping at its boundary if needed.
    """

    num_subarray_rows: int
    num_horizontal: int
    elements_per_subarray: int
    num_polarizations: int = 2
    port_order: str = "polarization-major"

    def __post_init__(self) -> None:
        if self.num_subarray_rows <= 0:
            raise ValueError(
                f"num_subarray_rows must be positive, got {self.num_subarray_rows}."
            )
        if self.num_horizontal <= 0:
            raise ValueError(
                f"num_horizontal must be positive, got {self.num_horizontal}."
            )
        if self.elements_per_subarray not in (1, 2, 3, 4):
            raise ValueError(
                "elements_per_subarray must be one of 1, 2, 3, or 4, "
                f"got {self.elements_per_subarray}."
            )
        if self.num_polarizations <= 0:
            raise ValueError(
                f"num_polarizations must be positive, got {self.num_polarizations}."
            )
        if self.port_order != "polarization-major":
            raise ValueError(
                "only 'polarization-major' port_order is currently supported, "
                f"got {self.port_order!r}."
            )

    @property
    def num_physical_rows(self) -> int:
        """Total number of physical element rows."""
        return self.num_subarray_rows * self.elements_per_subarray

    @property
    def num_logical_subarrays(self) -> int:
        """Total number of logical subarrays."""
        return self.num_subarray_rows * self.num_horizontal

    @property
    def num_logical_polarized_ports(self) -> int:
        """Total number of logical subarray-polarization ports."""
        return self.num_logical_subarrays * self.num_polarizations

    @property
    def num_physical_elements(self) -> int:
        """Total number of physical antenna elements."""
        return self.num_physical_rows * self.num_horizontal

    @property
    def num_physical_ports(self) -> int:
        """Total number of physical element-polarization ports."""
        return self.num_physical_elements * self.num_polarizations

    def physical_port_index(
        self,
        polarization: int,
        physical_row: int,
        physical_column: int,
    ) -> int:
        """Return the physical port index for polarization-major ordering.

        Ports are ordered as ``[pol0 elements, pol1 elements, ...]``.  Within
        each polarization, elements are stored in row-major order with the
        horizontal index varying fastest:

        .. code-block:: text

            port = pol * num_physical_elements + row * num_horizontal + col
        """
        if self.port_order != "polarization-major":
            raise ValueError(
                f"physical_port_index is not implemented for {self.port_order!r}."
            )
        if not 0 <= polarization < self.num_polarizations:
            raise ValueError(
                f"polarization must be in [0, {self.num_polarizations}), "
                f"got {polarization}."
            )
        if not 0 <= physical_row < self.num_physical_rows:
            raise ValueError(
                f"physical_row must be in [0, {self.num_physical_rows}), "
                f"got {physical_row}."
            )
        if not 0 <= physical_column < self.num_horizontal:
            raise ValueError(
                f"physical_column must be in [0, {self.num_horizontal}), "
                f"got {physical_column}."
            )
        return (
            polarization * self.num_physical_elements
            + physical_row * self.num_horizontal
            + physical_column
        )
