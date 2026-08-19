"""Sionna RT integration tests for ArrayConfig/codebook port ordering.

These tests verify that project beam weights map to the correct Sionna RT
antenna ports.  They are skipped on environments without Sionna RT.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
import torch

from mMIMO_sleep.array_config import ArrayConfig
from mMIMO_sleep.simulation.sionna_array import (
    _project_to_sionna_element_index,
    array_config_to_planar_array,
    weights_to_sionna_precoding,
)

_HAS_SIONNA = importlib.util.find_spec("sionna.rt") is not None

pytestmark = pytest.mark.skipif(not _HAS_SIONNA, reason="Sionna RT not installed")


def _set_mitsuba_variant() -> None:
    import mitsuba as mi

    if mi.variant() != "cuda_ad_mono_polarized":
        mi.set_variant("cuda_ad_mono_polarized")


@pytest.fixture(autouse=True)
def _ensure_variant() -> None:
    _set_mitsuba_variant()


@pytest.mark.parametrize("elements_per_subarray", [1, 2])
def test_weights_to_sionna_precoding_reorders_elements(
    elements_per_subarray: int,
) -> None:
    """``weights_to_sionna_precoding`` reorders from row-major to column-first."""
    config = ArrayConfig(
        num_subarray_rows=2,
        num_horizontal=2,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    num_elements = config.num_physical_elements
    # Make each element value unique so reordering is detectable.
    pol0 = torch.arange(num_elements, dtype=torch.float32)
    pol1 = torch.arange(num_elements, dtype=torch.float32) + 100.0
    weights = torch.complex(
        torch.cat([pol0, pol1]),
        torch.cat([pol0 + 0.5, pol1 + 0.5]),
    )
    real, imag = weights_to_sionna_precoding(weights, config)

    # Verify each polarization block is reordered independently.
    for pol in range(config.num_polarizations):
        start = pol * num_elements
        for project_e in range(num_elements):
            sionna_e = _project_to_sionna_element_index(config, project_e)
            expected_real = (pol * 100.0) + project_e
            expected_imag = expected_real + 0.5
            assert real[start + sionna_e].item() == pytest.approx(expected_real)
            assert imag[start + sionna_e].item() == pytest.approx(expected_imag)


@pytest.mark.parametrize("elements_per_subarray", [1, 2, 3, 4])
@pytest.mark.parametrize("polarization", ["VH", "cross"])
def test_array_geometry_matches_array_config(
    elements_per_subarray: int,
    polarization: str,
) -> None:
    """Sionna RT element positions and port count match ``ArrayConfig``."""
    config = ArrayConfig(
        num_subarray_rows=2,
        num_horizontal=2,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    array = array_config_to_planar_array(
        config, pattern="iso", polarization=polarization
    )

    assert array.array_size == config.num_physical_elements
    assert array.num_ant == config.num_physical_ports

    positions = np.asarray(array.normalized_positions)
    # Sionna RT places the array in the y-z plane.  Columns vary fastest
    # (vertical/row index varies fastest).
    y = positions[1]
    z = positions[2]

    # PlanarArray does not expose num_rows/num_cols, so infer them from the
    # unique coordinates.  Each column shares a y-coordinate, each row a
    # z-coordinate.
    assert len(np.unique(np.round(y, decimals=6))) == config.num_horizontal
    assert len(np.unique(np.round(z, decimals=6))) == config.num_physical_rows

    # Project element positions: row-major, col fastest.
    # Map each project element to its Sionna element index and compare.
    for row in range(config.num_physical_rows):
        for col in range(config.num_horizontal):
            project_element = row * config.num_horizontal + col
            sionna_element = _project_to_sionna_element_index(config, project_element)
            expected_y = (col - (config.num_horizontal - 1) / 2) * 0.5
            expected_z = ((config.num_physical_rows - 1) / 2 - row) * 0.5
            assert y[sionna_element] == pytest.approx(expected_y)
            assert z[sionna_element] == pytest.approx(expected_z)


def test_cross_array_128_ports() -> None:
    """128-port cross-polarized TX array can be built from project config."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    array = array_config_to_planar_array(
        config, pattern="tr38901", polarization="cross"
    )

    assert array.array_size == 64
    assert array.num_ant == 128

    positions = np.asarray(array.normalized_positions)
    y = positions[1]
    z = positions[2]
    assert len(np.unique(np.round(y, decimals=6))) == 8
    assert len(np.unique(np.round(z, decimals=6))) == 8


@pytest.mark.parametrize("elements_per_subarray", [1, 2])
@pytest.mark.parametrize("tx_polarization", ["V", "H"])
def test_one_hot_port_mapping_via_channel_phase(
    elements_per_subarray: int,
    tx_polarization: str,
) -> None:
    """One-hot TX weights illuminate the expected physical element/polarization.

    For each project element we look at the LOS channel coefficient of the
    corresponding Sionna RT port (after the explicit row-major -> column-first
    mapping).  The phase progression across elements must match the geometric
    phase implied by ``ArrayConfig`` positions, and the orthogonal polarization
    must be essentially silent.
    """
    import sionna.rt as rt

    config = ArrayConfig(
        num_subarray_rows=2,
        num_horizontal=2,
        elements_per_subarray=elements_per_subarray,
        num_polarizations=2,
    )
    tx_array = array_config_to_planar_array(config, pattern="iso", polarization="VH")
    rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization=tx_polarization,
    )

    scene = rt.load_scene()
    scene.tx_array = tx_array
    scene.rx_array = rx_array

    # Place RX far away at an oblique angle so phases vary across the array.
    wavelength = float(scene.wavelength[0])
    R = 50.0 * wavelength
    y_offset = 10.0 * wavelength
    z_offset = 5.0 * wavelength
    direction_norm = math.sqrt(R * R + y_offset * y_offset + z_offset * z_offset)
    u_y = y_offset / direction_norm
    u_z = z_offset / direction_norm

    scene.add(rt.Transmitter(name="tx", position=[0.0, 0.0, 0.0]))
    scene.add(rt.Receiver(name="rx", position=[R, y_offset, z_offset]))

    paths = rt.PathSolver()(
        scene,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert np.asarray(paths.valid).any(), "Expected at least one LOS path"

    a0 = np.asarray(paths.a[0])
    a1 = np.asarray(paths.a[1])
    # Internal Sionna shape: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
    h = a0[0, 0, 0, :, 0] + 1j * a1[0, 0, 0, :, 0]
    assert h.shape == (config.num_physical_ports,)

    # Project positions.
    project_y = np.empty(config.num_physical_elements, dtype=np.float64)
    project_z = np.empty(config.num_physical_elements, dtype=np.float64)
    for row in range(config.num_physical_rows):
        for col in range(config.num_horizontal):
            e = row * config.num_horizontal + col
            project_y[e] = (col - (config.num_horizontal - 1) / 2) * 0.5
            project_z[e] = ((config.num_physical_rows - 1) / 2 - row) * 0.5

    expected_active_pol = {"V": 0, "H": 1}[tx_polarization]
    active_start = expected_active_pol * config.num_physical_elements
    active_end = active_start + config.num_physical_elements
    inactive_slice = slice(
        (1 - expected_active_pol) * config.num_physical_elements,
        (2 - expected_active_pol) * config.num_physical_elements,
    )

    # Orthogonal polarization must be essentially silent.
    inactive_amplitudes = np.abs(h[inactive_slice])
    active_amplitudes = np.abs(h[active_start:active_end])
    assert inactive_amplitudes.max() < 1e-9 * active_amplitudes.max()

    # Map project element order to Sionna element order and compare relative
    # phases.  A common global phase is removed by subtracting the first
    # element's phase.
    measured_phases = np.empty(config.num_physical_elements, dtype=np.float64)
    expected_phases = np.empty(config.num_physical_elements, dtype=np.float64)
    for e in range(config.num_physical_elements):
        sionna_e = _project_to_sionna_element_index(config, e)
        measured_phases[e] = np.angle(h[active_start + sionna_e])
        expected_phases[e] = 2.0 * math.pi * (u_y * project_y[e] + u_z * project_z[e])

    measured_rel = measured_phases - measured_phases[0]
    expected_rel = expected_phases - expected_phases[0]
    phase_diff = (measured_rel - expected_rel + math.pi) % (2.0 * math.pi) - math.pi
    np.testing.assert_allclose(phase_diff, np.zeros_like(phase_diff), atol=0.05)


@pytest.mark.parametrize("polarization", ["VH", "cross"])
def test_codebook_weights_run_in_radio_map_solver(polarization: str) -> None:
    """A codebook beam weight vector can be supplied to RadioMapSolver."""
    import mitsuba as mi
    import sionna.rt as rt

    config = ArrayConfig(
        num_subarray_rows=2,
        num_horizontal=2,
        elements_per_subarray=1,
        num_polarizations=2,
    )
    tx_array = array_config_to_planar_array(
        config,
        pattern="tr38901" if polarization == "cross" else "iso",
        polarization=polarization,
    )
    rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    scene = rt.load_scene()
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    scene.add(rt.Transmitter(name="tx", position=[0.0, 0.0, 2.0]))
    scene.add(rt.Receiver(name="rx", position=[10.0, 0.0, 2.0]))

    # Generate a single beam weight vector from the project codebook.
    from mMIMO_sleep.codebook.dft import generate_dft_codebook

    codebook = generate_dft_codebook(
        config,
        num_vertical_beams=2,
        num_horizontal_beams=2,
        num_i2=4,
        device="cpu",
    )
    weights = codebook[0]
    real, imag = weights_to_sionna_precoding(weights, config)

    # Convert to Mitsuba tensors on the active backend.
    precoding_vec = (
        mi.TensorXf(real.numpy()),
        mi.TensorXf(imag.numpy()),
    )

    radio_map = rt.RadioMapSolver()(
        scene,
        center=mi.Point3f(10.0, 0.0, 2.0),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(2.0, 2.0),
        cell_size=mi.Point2f(0.5, 0.5),
        precoding_vec=precoding_vec,
        samples_per_tx=10000,
        max_depth=1,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert radio_map is not None
    pg = np.asarray(radio_map.path_gain)
    assert pg.ndim >= 2
    assert np.isfinite(pg).all()


@pytest.mark.parametrize("polarization", ["VH", "cross"])
def test_one_hot_project_to_sionna_port_mapping(
    polarization: str,
) -> None:
    """Each project port illuminates exactly one expected Sionna RT port.

    Sionna RT stores the precoding vector in pattern-major order:
    ``[pattern 0 over all elements, pattern 1 over all elements]``.  Within each
    pattern, elements are numbered column-first (row varies fastest).  The
    project uses polarization-major order with row-major elements, so the
    mapping must place project pol0/pol1 into Sionna pattern 0/1 and reorder
    elements within each block.
    """
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=4,
        elements_per_subarray=1,
        num_polarizations=2,
    )
    num_elements = config.num_physical_elements

    test_ports = [
        0,  # pol0, first element
        num_elements - 1,  # pol0, last element
        num_elements,  # pol1, first element
        2 * num_elements - 1,  # pol1, last element
        # A middle row/column element: row 1, col 2.
        1 * config.num_horizontal + 2,
        # The same physical element on pol1.
        num_elements + 1 * config.num_horizontal + 2,
    ]

    for project_port in test_ports:
        weights = torch.zeros(config.num_physical_ports, dtype=torch.complex64)
        weights[project_port] = 1.0 + 0.0j

        real, imag = weights_to_sionna_precoding(weights, config)
        sionna_weights = torch.complex(real, imag)

        # Exactly one Sionna port is active.
        non_zero = torch.nonzero(sionna_weights.abs() > 1e-9).flatten()
        assert non_zero.numel() == 1, (
            f"Expected one active Sionna port for project port {project_port}, "
            f"got {non_zero.numel()}"
        )
        active_sionna_port = int(non_zero.item())

        project_pol = project_port // num_elements
        project_element = project_port % num_elements
        expected_sionna_element = _project_to_sionna_element_index(
            config, project_element
        )
        expected_sionna_port = project_pol * num_elements + expected_sionna_element

        assert active_sionna_port == expected_sionna_port, (
            f"Project port {project_port} (pol {project_pol}, element "
            f"{project_element}) mapped to Sionna port {active_sionna_port}, "
            f"expected {expected_sionna_port}"
        )


@pytest.mark.parametrize("polarization", ["VH", "cross"])
def test_weights_to_sionna_precoding_power_conservation(
    polarization: str,
) -> None:
    """Mapping is a pure permutation + real/imag split; total power is conserved."""
    config = ArrayConfig(
        num_subarray_rows=4,
        num_horizontal=8,
        elements_per_subarray=2,
        num_polarizations=2,
    )
    rng = torch.Generator().manual_seed(42)
    weights = torch.complex(
        torch.randn(config.num_physical_ports, generator=rng, dtype=torch.float32),
        torch.randn(config.num_physical_ports, generator=rng, dtype=torch.float32),
    )
    input_power = float(torch.abs(weights).square().sum())

    real, imag = weights_to_sionna_precoding(weights, config)
    output_power = float((real.square() + imag.square()).sum())

    assert input_power == pytest.approx(output_power, rel=1e-6)


@pytest.mark.parametrize("polarization", ["VH", "cross"])
def test_one_hot_port_maps_to_expected_sionna_channel_port(
    polarization: str,
) -> None:
    """Integration test: project one-hot weights target the right Sionna port.

    A one-hot project port selects exactly one Sionna RT transmit port.  We
    obtain the narrowband channel ``H`` from Sionna RT and verify that the
    effective channel ``H @ w_sionna`` equals the Sionna channel coefficient at
    the expected port.
    """
    import mitsuba as mi
    import sionna.rt as rt

    config = ArrayConfig(
        num_subarray_rows=2,
        num_horizontal=2,
        elements_per_subarray=1,
        num_polarizations=2,
    )
    tx_array = array_config_to_planar_array(
        config, pattern="tr38901", polarization=polarization
    )
    # RX polarization is arbitrary for this test; V is sufficient to verify
    # port mapping and finiteness.
    rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    scene = rt.load_scene()
    scene.tx_array = tx_array
    scene.rx_array = rx_array

    wavelength = float(scene.wavelength[0])
    R = 50.0 * wavelength
    scene.add(rt.Transmitter(name="tx", position=[0.0, 0.0, 0.0]))
    scene.add(rt.Receiver(name="rx", position=[R, 10.0 * wavelength, 5.0 * wavelength]))

    paths = rt.PathSolver()(
        scene,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    assert np.asarray(paths.valid).any(), "Expected at least one LOS path"
    assert paths.num_tx == 1
    assert paths.num_rx == 1
    assert paths.a[0].shape[3] == config.num_physical_ports

    h = np.asarray(
        paths.cfr(
            frequencies=mi.Float([0.0]),
            out_type="numpy",
            normalize=False,
        )
    ).squeeze()
    assert h.shape == (config.num_physical_ports,)
    assert np.isfinite(h).all()

    num_elements = config.num_physical_elements
    test_ports = [0, num_elements - 1, num_elements, 2 * num_elements - 1]
    for project_port in test_ports:
        weights = torch.zeros(config.num_physical_ports, dtype=torch.complex64)
        weights[project_port] = 1.0
        real, imag = weights_to_sionna_precoding(weights, config)
        w_sionna = np.asarray(torch.complex(real, imag))

        expected_pol = project_port // num_elements
        expected_element = _project_to_sionna_element_index(
            config, project_port % num_elements
        )
        expected_sionna_port = expected_pol * num_elements + expected_element

        effective = np.dot(h, w_sionna)
        assert np.isfinite(effective)
        assert np.allclose(effective, h[expected_sionna_port], atol=1e-5), (
            f"Project port {project_port} effective channel {effective} does not "
            f"match expected Sionna port {expected_sionna_port} "
            f"{h[expected_sionna_port]}."
        )
