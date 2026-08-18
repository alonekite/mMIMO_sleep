"""Tests for array_config.py."""

from __future__ import annotations

import pytest

from mMIMO_sleep.array_config import ArrayConfig


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
def test_physical_dimensions(elements_per_subarray: int) -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    assert config.num_physical_rows == 4 * elements_per_subarray
    assert config.num_logical_subarrays == 4 * 8
    assert config.num_logical_polarized_ports == 4 * 8 * 2
    assert config.num_physical_elements == config.num_physical_rows * 8
    assert config.num_physical_ports == config.num_physical_elements * 2


def test_invalid_elements_per_subarray() -> None:
    with pytest.raises(ValueError):
        ArrayConfig(
            num_subarray_rows=4,
            num_horizontal=8,
            elements_per_subarray=5,
        )


def test_invalid_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        ArrayConfig(num_subarray_rows=0, num_horizontal=8, elements_per_subarray=2)
    with pytest.raises(ValueError):
        ArrayConfig(num_subarray_rows=4, num_horizontal=0, elements_per_subarray=2)
    with pytest.raises(ValueError):
        ArrayConfig(
            num_subarray_rows=4,
            num_horizontal=8,
            elements_per_subarray=2,
            num_polarizations=0,
        )


def test_unsupported_port_order() -> None:
    with pytest.raises(ValueError):
        ArrayConfig(
            num_subarray_rows=4,
            num_horizontal=8,
            elements_per_subarray=2,
            port_order="interleaved",
        )


def test_physical_port_index() -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    # Polarization-major ordering: pol 0 first, then pol 1.
    num_elements = config.num_physical_elements
    assert config.physical_port_index(0, 0, 0) == 0
    assert config.physical_port_index(0, 0, 7) == 7
    assert config.physical_port_index(0, 1, 0) == 8
    assert config.physical_port_index(1, 0, 0) == num_elements
    assert config.physical_port_index(1, 0, 7) == num_elements + 7


def test_physical_port_index_invalid() -> None:
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    with pytest.raises(ValueError):
        config.physical_port_index(2, 0, 0)
    with pytest.raises(ValueError):
        config.physical_port_index(0, 8, 0)
    with pytest.raises(ValueError):
        config.physical_port_index(0, 0, 8)
