#!/usr/bin/env python3
"""Unified smoke test for the mMIMO_sleep RunPod image, with two modes:

  --mode cpu (default)
      Import-level / CPU-only checks. Does NOT require a GPU. Use this on
      any plain Docker build machine (including a Mac, or a CI runner with
      no GPU) right after `docker build` to catch packaging/import errors
      early. `torch.cuda.is_available()` is reported but NOT required to
      be True in this mode.

  --mode gpu
      Full acceptance check for a real RunPod GPU pod. Everything in
      --mode cpu, PLUS `torch.cuda.is_available()` must be True, a GPU
      name must be readable, and Mitsuba must expose a working CUDA
      variant. This mode is meant to be the final RunPod acceptance test,
      not something you run on a build machine without a GPU.

Run with the image's interpreter, e.g.:

    /opt/venv/bin/python /opt/smoke_test.py --mode cpu
    /opt/venv/bin/python /opt/smoke_test.py --mode gpu

Exits with status 0 if every required check passes, non-zero otherwise.
Each check is independent and prints a PASS/FAIL/INFO line so a single
failure does not hide the results of the others.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, fn):
    """Run fn(); print PASS/FAIL; record failures. Returns fn()'s result or None."""
    try:
        result = fn()
        print(f"[PASS] {name}: {result}")
        return result
    except Exception as exc:  # noqa: BLE001 - smoke test wants to keep going
        print(f"[FAIL] {name}: {exc}")
        FAILURES.append(name)
        return None


def info(name: str, fn):
    """Run fn() and print its result/exception WITHOUT counting it as a failure.

    Used for GPU-related facts in --mode cpu, where e.g.
    torch.cuda.is_available() is expected to be False and should not fail
    the build-machine smoke test.
    """
    try:
        result = fn()
        print(f"[INFO] {name}: {result}")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] {name}: <unavailable> ({exc})")
        return None


def _torch_version():
    import torch

    return torch.__version__


def _torch_cuda_version():
    import torch

    return torch.version.cuda


def _torch_cuda_is_available():
    import torch

    return torch.cuda.is_available()


def _gpu_name():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    return torch.cuda.get_device_name(0)


def _sionna_version():
    return importlib.metadata.version("sionna-rt")


def _mitsuba_version_and_any_variant():
    import mitsuba as mi

    version = importlib.metadata.version("mitsuba")
    variants = list(mi.variants())
    if not variants:
        raise RuntimeError("mitsuba reports no available variants at all")
    # Prefer a CPU-friendly variant (scalar/llvm) so this works without a GPU.
    preferred = next(
        (v for v in variants if "cuda" not in v), variants[0]
    )
    mi.set_variant(preferred)
    return f"{version} (variant={mi.variant()})"


def _mitsuba_version_and_cuda_variant():
    import mitsuba as mi

    version = importlib.metadata.version("mitsuba")
    cuda_variants = [v for v in mi.variants() if "cuda" in v]
    if not cuda_variants:
        raise RuntimeError("no CUDA mitsuba variant available")
    mi.set_variant(cuda_variants[0])
    return f"{version} (variant={mi.variant()})"


def _drjit_version():
    return importlib.metadata.version("drjit")


def _planar_array():
    import sionna.rt as rt

    array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    return type(array).__name__


def _project_import():
    import mMIMO_sleep  # noqa: F401

    return "import ok (namespace package)"


def _pip_check():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output or "pip check failed")
    return output or "No broken requirements found."


def _kernelspec_argv0():
    candidates = [
        Path("/usr/local/share/jupyter/kernels/mmimo-sleep/kernel.json"),
        Path.home() / ".local/share/jupyter/kernels/mmimo-sleep/kernel.json",
    ]
    for path in candidates:
        if path.exists():
            spec = json.loads(path.read_text())
            argv0 = spec["argv"][0]
            if argv0 != "/opt/venv/bin/python":
                raise RuntimeError(
                    f"{path}: argv[0]={argv0!r}, expected /opt/venv/bin/python"
                )
            return f"{path}: argv[0]={argv0}"
    raise RuntimeError("mmimo-sleep kernelspec not found")


def run_common_checks() -> None:
    """Checks that apply in both --mode cpu and --mode gpu."""
    check("sys.executable", lambda: sys.executable)
    check("sys.version", lambda: sys.version.replace("\n", " "))
    check("torch.__version__", _torch_version)
    check("torch.version.cuda", _torch_cuda_version)
    check("sionna-rt version", _sionna_version)
    check("drjit version", _drjit_version)
    check("PlanarArray(1x1) creation", _planar_array)
    check("project package import (mMIMO_sleep)", _project_import)
    check("pip check", _pip_check)
    check("jupyter kernelspec argv[0]", _kernelspec_argv0)


def run_cpu_mode() -> None:
    print("Mode: CPU/import (no GPU required)")
    run_common_checks()
    # GPU facts are informational only in CPU mode: a build machine without
    # a GPU is expected to report False/unavailable here, and that must not
    # fail the smoke test.
    info("torch.cuda.is_available()", _torch_cuda_is_available)
    info("GPU name", _gpu_name)
    check("mitsuba version + variant (any available variant)", _mitsuba_version_and_any_variant)


def run_gpu_mode() -> None:
    print("Mode: GPU acceptance (CUDA required)")
    run_common_checks()

    def _cuda_must_be_available():
        available = _torch_cuda_is_available()
        if not available:
            raise RuntimeError(
                "torch.cuda.is_available() returned False. See docker/README.md "
                "'CUDA Error 804 diagnosis logic' for how to investigate."
            )
        return available

    check("torch.cuda.is_available()", _cuda_must_be_available)
    check("GPU name", _gpu_name)
    check("mitsuba version + CUDA variant", _mitsuba_version_and_cuda_variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu: import-only checks, no GPU required (default). "
        "gpu: full RunPod acceptance check, requires a working CUDA device.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("mMIMO_sleep RunPod environment smoke test")
    print("=" * 70)

    if args.mode == "cpu":
        run_cpu_mode()
    else:
        run_gpu_mode()

    print("=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print(f"RESULT: all checks passed ({args.mode} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
