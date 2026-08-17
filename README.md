# mMIMO_sleep

PMI-aware link-budget prediction for half-panel Massive MIMO sleep.

## Environment setup

The project is split into two environments:

- **Local Mac / CPU-only environment**: pure Python/PyTorch development
  (codebook, PMI mapping, muting, ML models, non-RT unit tests and notebooks).
  Sionna RT, Mitsuba, Dr.Jit and LLVM are **not** installed.
- **RunPod / GPU environment**: Sionna RT ray-tracing, scene loading,
  PathSolver, CIR/CFR generation, radio maps and batch UE data generation.

### Local Mac

```bash
python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e ".[dev]"
```

This installs PyTorch, NumPy, Matplotlib, pytest, JupyterLab and the local
package, but **not** Sionna RT.

### RunPod / GPU

```bash
python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip
python -m pip install -e ".[dev,rt]"
```

The ``[rt]`` extra installs ``sionna-rt==1.2.2`` (and its native dependencies
such as Mitsuba/Dr.Jit).  Do **not** upgrade to Sionna RT 2.x.

## Running tests

### Local (no Sionna RT)

```bash
./.venv/bin/python -m pytest -q
```

Only pure-PyTorch tests run.  Sionna RT tests are skipped automatically because
the optional ``[rt]`` dependencies are not installed.

### RunPod smoke test

After installing with ``[dev,rt]``:

```bash
./.venv/bin/python -m pytest -q tests/test_rt_smoke.py -v
```

This checks that PyTorch, CUDA, Mitsuba and Sionna RT are available.  The
second test is a placeholder for a minimal scene/transmitter/receiver/
PathSolver integration that will be implemented on RunPod.

## Optional dependency groups

| Group | Installed on Mac | Installed on RunPod | Contents |
|---|---|---|---|
| default (core) | yes | yes | PyTorch, NumPy, Matplotlib |
| ``dev`` | yes | yes | pytest, JupyterLab, ipykernel, ruff |
| ``rt`` | **no** | yes | ``sionna-rt==1.2.2`` |

## Notes on Sionna RT imports

Code that genuinely requires Sionna RT should import it lazily inside the
function or method that uses it.  A helper is provided in
``src/mMIMO_sleep/simulation/rt_compat.py``:

```python
from mMIMO_sleep.simulation.rt_compat import require_sionna_rt


def generate_paths(scene, **solver_kwargs):
    rt = require_sionna_rt()
    solver = rt.PathSolver()
    return solver(scene, **solver_kwargs)
```

Pure-PyTorch modules (``codebook``, ``simulation.beam_sweep``, etc.) remain
importable when Sionna RT is absent.
