#!/usr/bin/env python3
"""Dr.Jit / Mitsuba import diagnostic — runs as a Dockerfile.runpod `RUN`
layer immediately BEFORE the CPU smoke test.

Attempts `import drjit` then `import mitsuba`. If both succeed, prints one
`[INFO] import <module>: OK` line per module and exits 0 — successful
builds get no extra noise. If either import fails, prints an extensive,
failure-only dynamic-linking diagnostic dump, then re-raises so this build
step (and therefore the Docker build) still fails here, with the
diagnostic already captured in the build log, one step before the CPU
smoke test would fail again on the same import.

This script performs NO installs, NO version changes, and NO speculative
fixes. It only inspects and reports the current state:

  - `python --version`, `sys.version`, `sys.executable`,
    `sysconfig.get_config_var("SOABI")`
  - `pip show drjit mitsuba sionna-rt`
  - every `*.so*` file under the installed `drjit` package directory
  - `ldd` on any `_drjit_ext*.so` extension module found there
  - `readelf -d` RPATH/RUNPATH/NEEDED entries for the same file(s)
  - the real, resolved location of `libdrjit-extra.so` and
    `libdrjit-core.so` anywhere under /opt/venv
  - the matching entries (if any) from `ldconfig -p`

Requires `readelf` (from the `binutils` package) and `ldd` (from
`libc-bin`, already present in the base image) to be on PATH; the
Dockerfile installs `binutils` for this purpose only.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import traceback


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (result.stdout + result.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostic must never crash itself
        return f"<failed to run {cmd!r}: {exc!r}>"


def _section(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _drjit_package_dirs() -> list[str]:
    try:
        spec = importlib.util.find_spec("drjit")
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] could not resolve import spec for 'drjit': {exc!r}")
        return []
    if spec is None:
        print("[diag] importlib.util.find_spec('drjit') returned None "
              "(package not found on sys.path at all).")
        return []
    dirs = list(spec.submodule_search_locations or [])
    if not dirs and spec.origin:
        dirs = [os.path.dirname(spec.origin)]
    return dirs


def dump_diagnostics(exc: BaseException, module_name: str) -> None:
    _section(f"Dr.Jit/Mitsuba import diagnostic (failed importing '{module_name}')")

    print("---- exception detail ----")
    print(f"type(exception): {type(exc)!r}")
    print(f"repr(exception): {exc!r}")
    print(f"exception.__cause__: {exc.__cause__!r}")
    print(f"exception.__context__: {exc.__context__!r}")
    print("---- full traceback ----")
    traceback.print_exc()

    print("---- python identity ----")
    print("python --version:", _run([sys.executable, "--version"]))
    print(f"sys.version: {sys.version}")
    print(f"sys.executable: {sys.executable}")
    print(f'sysconfig.get_config_var("SOABI"): {sysconfig.get_config_var("SOABI")!r}')

    print("---- pip show drjit mitsuba sionna-rt ----")
    print(_run([sys.executable, "-m", "pip", "show", "drjit", "mitsuba", "sionna-rt"]))

    print("---- drjit package directory .so* files ----")
    pkg_dirs = _drjit_package_dirs()
    print(f"drjit package dir(s): {pkg_dirs}")
    so_files: list[str] = []
    for d in pkg_dirs:
        so_files.extend(sorted(glob.glob(os.path.join(d, "**", "*.so*"), recursive=True)))
    if not so_files:
        print("<no .so* files found under the resolved drjit package dir(s)>")
    for f in so_files:
        print(f)

    print("---- ldd / readelf -d on _drjit_ext*.so ----")
    ext_files = [f for f in so_files if "_drjit_ext" in os.path.basename(f)]
    if not ext_files:
        for d in pkg_dirs:
            ext_files.extend(
                sorted(glob.glob(os.path.join(d, "**", "*drjit_ext*"), recursive=True))
            )
    if not ext_files:
        print("<no _drjit_ext*.so found under drjit package dir(s); "
              "listing every .so* found above instead>")
        ext_files = so_files
    for f in ext_files:
        print(f"# ldd {f}")
        print(_run(["ldd", f]))
        print(f"# readelf -d {f} (RPATH/RUNPATH/NEEDED only)")
        readelf_out = _run(["readelf", "-d", f])
        relevant = [
            line for line in readelf_out.splitlines()
            if any(key in line for key in ("RPATH", "RUNPATH", "NEEDED"))
        ]
        if relevant:
            print("\n".join(relevant))
        else:
            print(f"<no RPATH/RUNPATH/NEEDED lines in readelf output; raw output:>\n{readelf_out}")

    print("---- real location of libdrjit-extra.so / libdrjit-core.so ----")
    print("# find /opt/venv -iname 'libdrjit-extra.so*'")
    print(_run(["find", "/opt/venv", "-iname", "libdrjit-extra.so*"]) or "<not found under /opt/venv>")
    print("# find /opt/venv -iname 'libdrjit-core.so*'")
    print(_run(["find", "/opt/venv", "-iname", "libdrjit-core.so*"]) or "<not found under /opt/venv>")

    print("---- ldconfig -p | grep -i drjit ----")
    ldconfig_out = _run(["bash", "-c", "ldconfig -p | grep -i drjit || true"])
    print(ldconfig_out or "<no drjit entries registered in the ldconfig cache>")

    _section(f"END Dr.Jit/Mitsuba import diagnostic ('{module_name}')")


def main() -> int:
    for module_name in ("drjit", "mitsuba"):
        try:
            importlib.import_module(module_name)
            print(f"[INFO] import {module_name}: OK")
        except Exception as exc:  # noqa: BLE001 - must catch to dump diagnostics, then re-raise
            dump_diagnostics(exc, module_name)
            print(
                f"[FAIL] import {module_name} failed; diagnostics printed above. "
                "Re-raising so this build step fails here (before the CPU "
                "smoke test layer)."
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
