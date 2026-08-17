"""Compatibility helpers for optional Sionna RT / Mitsuba / Dr.Jit imports.

Sionna RT and its native dependencies are intentionally *not* installed on the
local Mac development environment.  They are only installed in the RunPod/GPU
environment through the ``[rt]`` optional dependency group.

Modules that need Sionna RT should import it lazily inside functions, or use
:func:`require_sionna_rt` / :func:`require_mitsuba` to get a clear error
message when the optional dependencies are missing.
"""

from __future__ import annotations

import importlib.util


def _package_available(name: str) -> bool:
    """Return ``True`` if ``name`` can be imported without side effects."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - defensive
        return False


def require_sionna_rt():
    """Return the ``sionna.rt`` module, or raise a clear RuntimeError."""
    if not _package_available("sionna.rt"):
        raise RuntimeError(
            "Sionna RT is required for ray-tracing simulation. "
            "Install the project with: pip install -e '.[dev,rt]'"
        )
    import sionna.rt

    return sionna.rt


def require_mitsuba():
    """Return the ``mitsuba`` module, or raise a clear RuntimeError."""
    if not _package_available("mitsuba"):
        raise RuntimeError(
            "Mitsuba is required for ray-tracing simulation. "
            "Install the project with: pip install -e '.[dev,rt]'"
        )
    import mitsuba as mi

    return mi
