"""RunPod-only smoke tests for the Sionna RT / Mitsuba environment.

These tests are skipped automatically on the local Mac because Sionna RT is
intentionally not installed there.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch


def _has_sionna_rt() -> bool:
    try:
        return importlib.util.find_spec("sionna.rt") is not None
    except Exception:  # pragma: no cover - defensive
        return False


HAS_SIONNA_RT = _has_sionna_rt()

pytestmark = pytest.mark.skipif(
    not HAS_SIONNA_RT,
    reason="Sionna RT is only installed in the RunPod/GPU environment",
)


def test_environment_check() -> None:
    """Verify PyTorch, Mitsuba and Sionna RT are present and consistent."""
    import mitsuba as mi
    import sionna.rt

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Sionna RT:", getattr(sionna.rt, "__version__", "unknown"))
    print("Mitsuba:", getattr(mi, "__version__", "unknown"))
    print("Variant:", mi.variant())

    assert mi.variant() is not None


def test_minimal_path_solver() -> None:
    """Minimal Sionna RT PathSolver smoke test using the Sionna RT 1.2.2 API.

    This test exercises the full PyTorch + Mitsuba + Dr.Jit + Sionna RT GPU
    pipeline with a tiny synthetic scene:

    1. Load an empty Sionna RT scene.
    2. Add one transmitter and one receiver.
    3. Configure the simplest planar arrays (2x2 isotropic, single-pol).
    4. Run ``PathSolver`` once.
    5. Assert that the computation succeeds and returns at least one valid path.
    6. Assert that path coefficients and delays contain no NaN or Inf.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA not available; Sionna RT PathSolver smoke test requires a GPU"
        )

    import mitsuba as mi
    import sionna.rt as rt

    # Ensure Mitsuba uses the CUDA backend expected by Sionna RT.
    mi.set_variant("cuda_ad_mono_polarized")

    # 1. Empty synthetic scene is sufficient for a LOS/free-space path.
    scene = rt.load_scene()

    # 2. Simplest planar arrays: 2x2 isotropic elements, single (V) polarization.
    array = rt.PlanarArray(
        num_rows=2, num_cols=2, pattern="iso", polarization="V"
    )
    scene.tx_array = array
    scene.rx_array = array

    # 3. One TX and one RX separated in free space.
    scene.add(rt.Transmitter(name="tx", position=[0.0, 0.0, 2.0]))
    scene.add(rt.Receiver(name="rx", position=[10.0, 0.0, 2.0]))

    # 4. Run the path solver once.
    paths = rt.PathSolver()(scene)

    # 5. At least one valid path must exist.
    assert paths.valid is not None
    valid = np.asarray(paths.valid)
    assert valid.any(), "Expected at least one valid propagation path"

    # 6. Path coefficients and delays must be finite and non-negative.
    assert paths.a is not None
    assert paths.tau is not None

    a0 = np.asarray(paths.a[0])
    a1 = np.asarray(paths.a[1])
    tau = np.asarray(paths.tau)

    assert np.isfinite(a0).all(), "Path coefficients contain NaN or Inf"
    assert np.isfinite(a1).all(), "Path coefficients contain NaN or Inf"
    assert np.isfinite(tau).all(), "Path delays contain NaN or Inf"
    assert (tau >= 0).all(), "Path delays must be non-negative"

    # Basic shape sanity: (tx_devices, tx_ports, rx_devices, rx_ports, paths)
    assert a0.ndim == 5
    assert a0.shape[1] == 4, "TX array should expose 4 ports (2x2)"
    assert a0.shape[3] == 4, "RX array should expose 4 ports (2x2)"
