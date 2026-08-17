"""RunPod-only smoke tests for the Sionna RT / Mitsuba environment.

These tests are skipped automatically on the local Mac because Sionna RT is
intentionally not installed there.
"""

from __future__ import annotations

import importlib.util

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


def test_minimal_path_solver_placeholder() -> None:
    """Placeholder for a minimal Sionna RT scene integration.

    When implemented on RunPod this should:

    1. Load one simple scene.
    2. Add one transmitter and one receiver.
    3. Configure an 8x8 physical array (128 TX ports with dual polarization).
    4. Run PathSolver once.
    5. Obtain a CIR/CFR tensor H and confirm the TX-port dimension is 128.
    6. Compute ``H @ w_normal`` and ``H @ w_sleep`` for a representative beam.
    """
    # TODO: implement on RunPod once the scene asset and array conventions are
    # finalized.  The test exists now so that CI/pytest discovery records the
    # intended integration point.
    pytest.skip("Minimal PathSolver integration not yet implemented")
