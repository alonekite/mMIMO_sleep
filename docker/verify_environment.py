#!/usr/bin/env python3
"""Build-time dependency verification for the mMIMO_sleep RunPod image.

Run with the image's /opt/venv interpreter, immediately after installing
``torch==2.13.0+cu126`` and ``docker/requirements-runpod.txt``. Exits
non-zero (failing the `docker build`) if:

  - ``pip check`` reports any broken/incompatible requirement;
  - ``torch.__version__`` is not exactly "2.13.0+cu126";
  - ``torch.version.cuda`` is not exactly "12.6";
  - any installed distribution's name unambiguously identifies it as an
    NVIDIA CUDA 13 runtime package, i.e. the name ends with "-cu13"
    (case-insensitive), matching NVIDIA's own split-package naming
    convention (e.g. ``nvidia-cublas-cu13``, ``nvidia-cuda-runtime-cu13``).

Deliberately does NOT reject a package merely because its name contains
the substring "cuda" (e.g. a hypothetical "cuda-utils" or "pycuda"
package) -- only the specific "-cuNN" suffix convention is inspected, so
this check stays precise instead of blindly matching on "cuda".

Always writes the full resolved ``pip freeze`` output to
/opt/build/pip-freeze.txt (for auditability), before running any
assertion, so the freeze file is available even if a later assertion
fails the build.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

FREEZE_PATH = Path("/opt/build/pip-freeze.txt")
EXPECTED_TORCH_VERSION = "2.13.0+cu126"
EXPECTED_TORCH_CUDA = "12.6"
_CU13_SUFFIX_RE = re.compile(r"-cu13$", re.IGNORECASE)


def _fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def _write_pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(result.stdout)
    print(
        f"[INFO] wrote resolved environment "
        f"({len(result.stdout.splitlines())} distributions) to {FREEZE_PATH}"
    )
    print(result.stdout)
    return result.stdout


def _run_pip_check() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        _fail(f"pip check reported broken requirements:\n{output}")
    print(f"[PASS] pip check: {output or 'No broken requirements found.'}")


def _check_torch_versions() -> None:
    import torch

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        _fail(
            f"torch.__version__ == {torch.__version__!r}, "
            f"expected {EXPECTED_TORCH_VERSION!r}"
        )
    print(f"[PASS] torch.__version__ == {torch.__version__!r}")

    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        _fail(
            f"torch.version.cuda == {torch.version.cuda!r}, "
            f"expected {EXPECTED_TORCH_CUDA!r}"
        )
    print(f"[PASS] torch.version.cuda == {torch.version.cuda!r}")


def _check_no_cuda13_packages() -> None:
    offenders = sorted(
        dist.metadata["Name"]
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name") and _CU13_SUFFIX_RE.search(dist.metadata["Name"])
    )
    if offenders:
        _fail(
            "the following installed distribution name(s) unambiguously "
            f"indicate a CUDA 13 NVIDIA runtime package (name ends with "
            f"'-cu13'): {offenders}"
        )
    print("[PASS] no installed distribution name ends with '-cu13'")


def main() -> int:
    _write_pip_freeze()
    _run_pip_check()
    _check_torch_versions()
    _check_no_cuda13_packages()
    print("[PASS] all dependency assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
