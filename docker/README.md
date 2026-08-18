# mMIMO_sleep RunPod Docker Image

This directory contains the build artifacts for a reproducible RunPod
container image for the `mMIMO_sleep` project. The goal is to eliminate the
two recurring failures documented in `../handoff20260817.md` and
`../handoff20260818.md`:

1. `.venv/bin/python` becoming a dangling symlink after a pod
   migration/restart, because the system `python3.11` it pointed to did not
   exist in the new container.
2. `torch.cuda.is_available()` returning `False` with `Error 804: forward
   compatibility was attempted on non supported HW`, because a CUDA
   forward-compatibility `libcuda.so` was loaded ahead of the real host
   driver library.

## 1. Design summary

| Aspect | Decision |
|---|---|
| Base image | `nvidia/cuda:12.6.3-base-ubuntu22.04` (fixed tag) |
| Why compatible with cu126 wheels | It's an Ubuntu 22.04 image tagged for CUDA 12.6. The `-base` variant does ship real CUDA 12.6 runtime packaging (the CUDA base package, NVIDIA Container Toolkit hook metadata, and core shared libraries such as `libcudart`) — it is not only metadata/env vars — but it excludes the full development toolkit (`nvcc`, headers) and, necessarily, the GPU driver itself, which can never be baked into any container image and is always injected by the RunPod host via the NVIDIA Container Toolkit at container start. The bulk of the CUDA *runtime* libraries actually used by `torch==2.13.0+cu126` (cuBLAS, cuDNN, NCCL, ...) come from the `nvidia-*-cu12` pip wheels bundled with PyTorch itself (in `site-packages/nvidia/*`), independent of the base image. Host driver compatibility with CUDA 12.6 remains essential regardless. |
| Python 3.11 | Installed from the **deadsnakes PPA** as the unversioned `python3.11` / `python3.11-venv` packages (major.minor pin only — see "Python 3.11 pinning" below for why an exact patch pin was deliberately dropped). |
| Where the venv lives | `/opt/venv`, **inside the image**, not under `/workspace`. This is the core fix for failure (1): the interpreter is now part of the immutable, versioned image and can never again dangle after a pod migration. |
| What goes in the image | OS packages (including `llvm`, needed for Dr.Jit/Mitsuba's LLVM JIT backend), Python 3.11, `/opt/venv` with all pinned dependencies (`pip`/`setuptools`/`wheel` bootstrap, PyTorch cu126, Sionna RT, Mitsuba, Dr.Jit, NumPy, Matplotlib, Jupyter/ipykernel, pytest, ruff), the Jupyter kernelspec, `docker/smoke_test.py`, `docker/verify_environment.py`, `docker/runpod-start.sh`. |
| What stays on `/workspace` (network volume) | The project source (`src/`, `tests/`, `notebook/`, `pyproject.toml`, handoff docs, `.git`). The image never bakes in project code; `runpod-start.sh` runs `pip install -e /workspace/mMIMO_sleep --no-deps` on every start so code changes on the volume take effect immediately without a rebuild. |
| Avoiding the old `.venv` symlink problem | `Dockerfile.runpod` and `runpod-start.sh` never reference `/workspace/mMIMO_sleep/.venv` at all. The old `.venv` directory is left untouched on the volume (for manual/local-debug use only) but is not part of the new workflow. The Jupyter kernelspec and `PATH` always point at `/opt/venv/bin/python`. |

### Python 3.11 pinning: ABI vs. exact patch version

**Correction from an earlier draft of this document**: it previously claimed
Python 3.11 needed an *exact* patch version (`3.11.15`) to match wheel ABI.
That claim was wrong and has been removed. The facts:

- The `cp311` wheel ABI tag, the stable ABI, and extension module (`.so`)
  naming are all determined by the **major.minor** version (3.11) only.
  Any `3.11.x` patch release is ABI-compatible with any other `3.11.x`
  patch release. `torch`, `mitsuba`, `drjit`, etc. wheels tagged `cp311`
  work identically on 3.11.0 through 3.11.latest.
- An exact apt pin (`python3.11=3.11.15-1+jammy1`) was tried for maximum
  reproducibility with the historical RunPod `.venv`, but **deadsnakes
  does not guarantee indefinite retention of superseded patch builds**.
  When a newer 3.11.x is uploaded to the PPA, older `.deb` files are
  routinely pruned from the Launchpad archive. An exact pin would make a
  future `docker build` fail outright ("has no installation candidate")
  once that specific patch is no longer published — even though nothing
  in this repository changed.
- **Decision**: pin only `python3.11` / `python3.11-venv` (major.minor).
  This guarantees the build keeps working indefinitely and the ABI stays
  correct, at the cost of not guaranteeing the *exact same patch build*
  across rebuilds performed months apart.
- **Reproducibility trade-off / mitigation**: if you need the exact same
  interpreter byte-for-byte, don't rely on rebuilding from this
  Dockerfile at an arbitrary future date — keep the already-built,
  versioned image (`:v1`, `:v2`, ...) as your artifact of record, and only
  rebuild deliberately. Each build also records the resolved version to
  `/opt/build/python-version.txt` inside the image and prints it to the
  build log, so you can always audit exactly which patch was used for a
  given tag.

### Preventing `pip install -e ".[dev,rt]"` from swapping PyTorch

Two layers of defense (see `Dockerfile.runpod` comments for details):

1. **Install order**: `torch==2.13.0+cu126` is installed first, directly
   from `https://download.pytorch.org/whl/cu126`, before any other Python
   package.
2. **`docker/constraints.txt`**, referenced via `PIP_CONSTRAINT` for the
   entire image build, pins `torch==2.13.0+cu126` so any `pip install`
   command run during the build (or manually inside a running container)
   will error out instead of silently resolving a different build.
3. At container start, the project is installed with
   `pip install -e /workspace/mMIMO_sleep --no-deps --no-build-isolation`,
   which never re-resolves `torch>=2.2,<3` from `pyproject.toml` at all, and
   never needs to download build-time tooling either: `setuptools` and
   `wheel` (the tools `pyproject.toml`'s `[build-system]` and an editable
   install need) are already pinned and installed in `/opt/venv` at image
   build time (see `Dockerfile.runpod`'s venv bootstrap step).

### CUDA Error 804 diagnosis logic

**Build time** (`Dockerfile.runpod`): intentionally does **nothing** to
CUDA compat libraries. There is no GPU/driver present during `docker
build` at all, so there is no way to verify whether a compat `libcuda.so`
would shadow anything real — unconditionally deleting files under those
conditions would be exactly the kind of blind system mutation we want to
avoid. The base image's compat package (if any) is simply left in place;
it is inert until a container actually runs on a GPU host.

**Run time** (`docker/runpod-start.sh`, function
`diagnose_and_fix_cuda_compat`): runs once at every container start, before
anything imports `torch`, and follows this evidence-gated procedure:

1. **Print diagnostics first, unconditionally**:
   - `nvidia-smi --query-gpu=driver_version,name` (driver version + GPU
     name, if `nvidia-smi` is present at all).
   - `LD_LIBRARY_PATH` (as inherited from the container environment).
   - Every `libcuda.so.1` provider known to `ldconfig -p`, in the exact
     priority order the dynamic linker would resolve them.
2. **Only act if shadowing is confirmed**: the first (highest-priority)
   entry from step 1 is inspected.
   - If it is **not** under a `*/compat/*` directory: nothing is
     changed — the real driver is already winning, so this is not
     Error 804's root cause.
   - If it **is** under `*/compat/*`, the remaining entries are searched
     for a non-compat provider (i.e. a real driver library). If none is
     found, nothing is changed — disabling the compat entry would leave
     the container with *no* `libcuda.so.1` at all, which is strictly
     worse.
   - Only when a real driver library is confirmed to exist **and** is
     currently being shadowed by a compat entry does the script rename
     the offending `/etc/ld.so.conf.d/*compat*.conf` file(s) to `.bak` and
     re-run `ldconfig`.
3. **Idempotent by construction**: once a `*compat*.conf` file has been
   renamed to `*.conf.bak`, it no longer matches the glob on a subsequent
   run, so the diagnostic is simply re-evaluated from scratch and (with
   the compat entry gone) falls into the "nothing to do" branch.
4. **Fully audited**: every read (`nvidia-smi`, `ldconfig -p` output) and
   every decision/mutation is written to stdout with a `[runpod-start]`
   prefix, and mutations are additionally prefixed `AUDIT:` so they are
   easy to `grep` out of container logs.

This is why **Error 804 specifically** is the trigger: it is NVIDIA's own
error code for "a forward-compatibility CUDA driver library was loaded,
but this GPU does not support forward compatibility" — which is exactly
the consumer-GPU (e.g. RTX 4090) + shadowed-compat-library combination
this function detects and, only when confirmed, corrects.

### Dr.Jit / `libatomic1`: a misleading error message, and the real root cause

The **first real GitHub Actions build** of this image failed with:

```
Could not import the Dr.Jit binary extension. It is likely that the Python
version for which Dr.Jit was compiled (3.11.14) is incompatible with the
current interpreter (3.11.15).
```

**This message is misleading.** It suggests a Python patch-version ABI
mismatch, but that is not possible here: `drjit==1.3.1` ships a `cp311`
wheel, and the `cp311` ABI tag, stable ABI, and extension module (`.so`)
naming are all determined by Python's **major.minor** version only — any
`3.11.x` patch release is ABI-compatible with any other `3.11.x` patch
release (see "Python 3.11 pinning" above). Dr.Jit's own import wrapper
prints this specific message for *any* failure to load its binary
extension, regardless of the actual cause, which made it look like a
patch-version problem when it was not.

**The real root cause**, found by inspecting `exception.__cause__` (not
just `str(exception)`) and running `ldd` on the extension module itself:

```
exception.__cause__:
ImportError: libatomic.so.1: cannot open shared object file: No such file or directory

ldd _drjit_ext*.so:
libatomic.so.1 => not found
```

Dr.Jit's compiled binary extension (`_drjit_ext*.so`) links against
`libatomic.so.1` (part of GCC's `libatomic` runtime, used for atomic
memory operations). The base image
(`nvidia/cuda:12.6.3-base-ubuntu22.04`) does not install this library by
default. The fix (in `Dockerfile.runpod`) is a single added apt package:

```
apt-get install -y --no-install-recommends libatomic1 ...
```

(Ubuntu 22.04's package name is `libatomic1`; the shared library it ships
is `libatomic.so.1`.) A build-time assertion immediately follows the
`apt-get install` step and greps `ldconfig -p` for `libatomic.so.1`,
failing the build immediately if it isn't registered — the same
fail-fast pattern already used for the `llvm`/`libLLVM` check above.

**Lesson**: when a compiled extension's own error message names a
specific hypothesis (here, a Python patch-version mismatch), verify it
against `exception.__cause__`/`exception.__context__` and `ldd` on the
actual `.so` before changing any pinned version — the wrapping library's
message can be generic and wrong, while the underlying `ImportError` and
`ldd`'s "not found" line are ground truth. No Python, Dr.Jit, Mitsuba,
Sionna RT, or PyTorch version was changed to fix this; it was purely a
missing system shared library.

## 2. Files in this directory / repo root

| File | Purpose |
|---|---|
| `../Dockerfile.runpod` | Image build definition. |
| `../.dockerignore` | Keeps `.git`, `.venv`, notebooks, docs, and secrets out of the build context/image. |
| `constraints.txt` | Pins `torch==2.13.0+cu126` for the whole build (via `PIP_CONSTRAINT`). |
| `requirements-runpod.txt` | Pinned versions for Sionna RT, Mitsuba, Dr.Jit, NumPy, Matplotlib, Jupyter/ipykernel, pytest, ruff. |
| `smoke_test.py` | Unified verification script (see Section 4 below); also run as a `RUN` layer inside `Dockerfile.runpod` itself in `--mode cpu`. |
| `verify_environment.py` | Build-time-only dependency assertions run as a `RUN` layer right after `requirements-runpod.txt` is installed: `pip check`, a `pip freeze` snapshot written to `/opt/build/pip-freeze.txt`, exact `torch.__version__`/`torch.version.cuda` checks, and a guard against any resolved CUDA 13 NVIDIA package (name ending `-cu13`). |
| `runpod-start.sh` | Idempotent container entrypoint (`CMD` in `Dockerfile.runpod`). |
| `README.md` | This file. |

## 3. Build

**RunPod GPU hosts are x86_64 (`linux/amd64`)**. If you build on Apple
Silicon (M-series Mac, `arm64`), you MUST force the target platform or the
resulting image will not run on RunPod. Use `docker buildx` (bundled with
modern Docker Desktop) for this, run from the project root
(`/workspace/mMIMO_sleep`):

```bash
# Build for linux/amd64 and load into the local Docker daemon (Mac M-series or any host):
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.runpod \
  -t mmimo-sleep-runpod:v1 \
  --load \
  .
```

On a native x86_64 Linux build machine, a plain `docker build` also works
since the host architecture already matches, but using `buildx --platform
linux/amd64` unconditionally (as above) is recommended everywhere so the
command is portable and unambiguous regardless of the build host's own
architecture.

Notes:

- The build context is the whole project root, but `.dockerignore` excludes
  `.git/`, `.venv/`, `notebook/`, `*.md`, and any dataset/output/cache
  directories, so none of those are sent to the Docker daemon or baked into
  layers. All files actually referenced by `COPY` in `Dockerfile.runpod`
  (`docker/constraints.txt`, `docker/requirements-runpod.txt`,
  `docker/smoke_test.py`, `docker/verify_environment.py`,
  `docker/runpod-start.sh`) have plain `.txt`/`.py`/`.sh` extensions and are
  not matched by any `.dockerignore` pattern — verified by inspection (see
  Section 8 static checks below).
- This build downloads PyTorch (~3-4 GB with CUDA libs) and the Sionna
  RT/Mitsuba/Dr.Jit wheels; expect the first build to take a while and use
  several GB of disk space for layer cache. Building `linux/amd64` on an
  `arm64` Mac uses QEMU emulation and will be noticeably slower than a
  native x86_64 build.
- **The build itself now fails fast on regressions**: after installing
  `requirements-runpod.txt`, a `RUN` layer executes
  `docker/verify_environment.py` (`pip check`, `pip freeze` snapshot, exact
  torch/CUDA version assertions, CUDA-13-package guard); after the Jupyter
  kernelspec and `smoke_test.py` are copied in, a further `RUN` layer
  executes `smoke_test.py --mode cpu`. Either failing stops `docker build`
  before an image is produced — no separate post-build smoke-test step is
  needed.
- **This environment could not run `docker build` directly** (no Docker
  daemon available in this session) — the files above were written and
  reviewed but the actual build was not executed. Please run the build
  command yourself and share the output if anything fails.

## 4. Smoke test — CPU/import mode vs. GPU acceptance mode

`docker/smoke_test.py` has two modes (see its own docstring for full
detail):

| Mode | Requires GPU? | Intended use |
|---|---|---|
| `--mode cpu` (default) | No | Run on **any** build machine right after `docker build` (including the Mac used to build, or a CI runner) to catch packaging/import errors early. `torch.cuda.is_available()` and GPU name are printed as `[INFO]` only — a `False`/unavailable result here does NOT fail the test. |
| `--mode gpu` | Yes | Final RunPod acceptance test on an actual GPU pod. `torch.cuda.is_available()` must be `True`, GPU name must resolve, and Mitsuba must expose a working CUDA variant — any of these failing fails the test. |

Both modes also check: `sys.executable`/`sys.version`, `torch.__version__`,
`torch.version.cuda`, Sionna RT version, Dr.Jit version,
`sionna.rt.PlanarArray(num_rows=1, num_cols=1, ...)` creation, `import
mMIMO_sleep`, `pip check`, and the Jupyter kernelspec `argv[0] ==
/opt/venv/bin/python`.

### CPU/import mode — any build machine, no GPU needed

```bash
docker run --rm mmimo-sleep-runpod:v1 \
  /opt/venv/bin/python /opt/smoke_test.py --mode cpu
```

Note: the `import mMIMO_sleep` check reports a `[PASS]` with a "skipped"
message (not a `[FAIL]`) when `/workspace/mMIMO_sleep` is not mounted —
expected both at `docker build` time (this is exactly the `RUN
/opt/venv/bin/python /opt/smoke_test.py --mode cpu` layer in
`Dockerfile.runpod`) and on any other build-verification machine with no
project volume attached. Once the volume is mounted and
`pip install -e ... --no-deps --no-build-isolation` has run (see Section 3
of `runpod-start.sh`), this check reports a real import instead.

### GPU acceptance mode — final RunPod validation

On a GPU host (e.g. an actual RunPod GPU pod, or any machine with
`nvidia-container-toolkit` and a GPU):

```bash
docker run --rm --gpus all \
  -v /workspace/mMIMO_sleep:/workspace/mMIMO_sleep \
  mmimo-sleep-runpod:v1 \
  bash -lc "/opt/venv/bin/pip install -e /workspace/mMIMO_sleep --no-deps && /opt/venv/bin/python /opt/smoke_test.py --mode gpu"
```

You can also run the project's own pytest suite the same way once the
volume is mounted and the project is editable-installed:

```bash
docker run --rm --gpus all \
  -v /workspace/mMIMO_sleep:/workspace/mMIMO_sleep \
  mmimo-sleep-runpod:v1 \
  bash -lc "/opt/venv/bin/pip install -e /workspace/mMIMO_sleep --no-deps && /opt/venv/bin/python -m pytest -q /workspace/mMIMO_sleep/tests"
```

## 5. Push to a registry

Versioned tags only — **never overwrite `latest`**. Since RunPod requires
`linux/amd64`, push directly from `buildx` with `--platform linux/amd64
--push` (this builds and pushes in one step, without needing a separate
local `--load`). Example for Docker Hub:

```bash
docker login
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.runpod \
  -t <your-dockerhub-username>/mmimo-sleep-runpod:v1 \
  --push \
  .
```

For a subsequent update, bump the tag (`v2`, `v3`, ...) rather than
reusing `v1` or `latest`, so running pods can pin an exact, known-good
image and upgrades are explicit:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.runpod \
  -t <your-dockerhub-username>/mmimo-sleep-runpod:v2 \
  --push \
  .
```

If you prefer to build and inspect locally before pushing, use `--load`
(Section 3) first, run the smoke tests (Section 4), then re-run with
`--push` once satisfied — `buildx` will reuse the build cache so the
second invocation is fast.

**I have not pushed anything anywhere** — no registry credentials are
configured in this session, and none should be entered here. See Section 8
for what I still need from you if you want this automated.

## 6. RunPod Custom Template configuration

When creating/editing the RunPod Template:

| Field | Value |
|---|---|
| Container Image | `<your-dockerhub-username>/mmimo-sleep-runpod:v1` (or your registry path) |
| Container Start Command | leave default (image `CMD` is `/opt/runpod-start.sh`), or explicitly set it to `/opt/runpod-start.sh` |
| Volume Mount Path | **`/workspace`** (strictly this path, not `/workspace/mMIMO_sleep`; the image expects the project at the `mMIMO_sleep` subdirectory of the volume mount, i.e. `PROJECT_DIR=/workspace/mMIMO_sleep`, matching how the existing RunPod Network Volume for this project is already organized) |
| Expose HTTP Ports | `8888` (JupyterLab, if `START_JUPYTER=1`) |
| Expose TCP Ports | `22` (SSH) |
| Environment variable `PUBLIC_KEY` | your SSH public key (RunPod's standard convention; injected into `/root/.ssh/authorized_keys` by `runpod-start.sh`). **Do not bake any key into the image.** |
| Environment variable `RUNPOD_JUPYTER_TOKEN` (optional) | a token string if you want JupyterLab to require authentication; leave unset only on trusted networks |
| Environment variable `START_JUPYTER` (optional) | set to `0` to disable the built-in JupyterLab launch if you prefer to start it manually |

## 7. VS Code interpreter and Notebook kernel

Once connected to the pod (Remote-SSH):

- **`Python: Select Interpreter`**: choose `/opt/venv/bin/python`.
- **Notebook `Select Kernel`**: choose **`Python (mMIMO_sleep)`** (kernelspec
  name `mmimo-sleep`, registered at
  `/usr/local/share/jupyter/kernels/mmimo-sleep/kernel.json`, `argv[0] =
  /opt/venv/bin/python`).

Do **not** select anything under `/workspace/mMIMO_sleep/.venv` any more —
that directory is no longer part of the supported workflow (kept only for
manual/legacy debugging on the network volume; see the design summary
above).

Also confirm (same as before, unrelated to this image) that the
`ms-python.python` and `ms-toolsai.jupyter` extensions are installed **for
the remote host**, not just locally — check the Extensions panel while
connected via Remote-SSH; each remote extension shows an "Install in SSH:
\<host\>" button if it is only installed locally.

## 8. Static checks performed in this review

Performed without building or running the image (no Docker daemon in this
session):

| Check | Result |
|---|---|
| `bash -n docker/runpod-start.sh` | Passed — no syntax errors. |
| `python3 -m py_compile docker/smoke_test.py` | Passed — no syntax errors. |
| Dockerfile linter (`hadolint`) | Not available in this environment (not installed, no network install performed since that would be an unrequested system change). If you have `hadolint` locally, run `hadolint Dockerfile.runpod` before building; the file already follows its main recommendations (pinned base image tag, `--no-install-recommends`, combined `apt-get update && install` in one `RUN`, apt list cleanup). |
| `.dockerignore` vs. `COPY` targets | Manually cross-checked every `COPY` source in `Dockerfile.runpod` (`docker/constraints.txt`, `docker/requirements-runpod.txt`, `docker/smoke_test.py`, `docker/runpod-start.sh`) against every pattern in `.dockerignore`. None match: the ignore file's `*.md` / `*.pem` / `*.key` / directory patterns do not touch `.txt`, `.py`, or `.sh` files under `docker/`. |
| Cross-file path/version consistency | Manually cross-checked: `PROJECT_DIR=/workspace/mMIMO_sleep` (Dockerfile ENV, `runpod-start.sh` default) matches the RunPod Volume Mount Path `/workspace` + project subdirectory; `/opt/venv/bin/python` (Dockerfile `PATH`/`VENV_PATH`, kernelspec `argv[0]` in both the Dockerfile's `ipykernel install` call and `runpod-start.sh`'s re-assertion, and `smoke_test.py`'s `_kernelspec_argv0` check) is identical everywhere; `torch==2.13.0+cu126` appears identically in `Dockerfile.runpod`'s install step, `docker/constraints.txt`, and `docker/README.md`; kernel name `mmimo-sleep` / display name `Python (mMIMO_sleep)` match across the Dockerfile, `runpod-start.sh`, and `smoke_test.py`. |

If you have Docker available, two more checks are recommended before the
real build:

```bash
docker run --rm -i hadolint/hadolint < Dockerfile.runpod
docker buildx build --platform linux/amd64 -f Dockerfile.runpod --check .
```

## 9. GitHub Actions build

`.github/workflows/build-runpod-image.yml` implements a **manually
triggered only** (`workflow_dispatch`) build. It never runs on push, pull
request, or a schedule — you must click "Run workflow" every time.

### One-time setup (you must do this; not done by this workflow)

1. **Create the Docker Hub repository** `DOCKERHUB_USERNAME/mmimo-sleep-runpod`
   (replace `DOCKERHUB_USERNAME` with your actual Docker Hub username) if it
   does not already exist. The workflow pushes to it but does not create it.
2. **GitHub Repository Variable** (Settings → Secrets and variables →
   Actions → Variables): `DOCKERHUB_USERNAME` = your Docker Hub username.
   This is a **Variable**, not a Secret, since a username is not sensitive
   and repository variables are visible in workflow logs/config, unlike
   secrets.
3. **GitHub Repository Secret** (Settings → Secrets and variables → Actions
   → Secrets): `DOCKERHUB_TOKEN` = a Docker Hub **access token** (Docker Hub
   → Account Settings → Security → New Access Token; do not use your
   account password). Grant it read/write scope on the target repository
   only.

### Running it

**Actions → Build RunPod image → Run workflow**, then fill in `image_tag`
(default `v1`). Use `v1` for the first image. **Once a tag has been
published, treat it as immutable: do not re-run the workflow with that
same tag again** (the workflow itself does not technically block this —
Docker Hub allows overwriting any tag — this is a process rule you must
follow, not a mechanical restriction). For every subsequent build, bump
the tag (`v2`, `v3`, ...) so running RunPod pods can keep pinning a known,
unchanged image.

### What the workflow does, in order

1. Checks out the repo.
2. Verifies `Dockerfile.runpod`, `.dockerignore`, `docker/runpod-start.sh`,
   `docker/smoke_test.py`, `docker/verify_environment.py`,
   `docker/requirements-runpod.txt`, `docker/constraints.txt` all exist.
3. Validates `image_tag`: rejects empty, `latest`, and anything that isn't a
   legal Docker tag.
4. Prints `df -h` and `docker version` for diagnostics (no secrets).
5. Logs in to Docker Hub (`docker/login-action@v4`, credentials from
   `vars.DOCKERHUB_USERNAME` / `secrets.DOCKERHUB_TOKEN`, never printed).
6. Sets up Buildx (`docker/setup-buildx-action@v4`).
7. Builds **and pushes** `Dockerfile.runpod` for `linux/amd64` in a single
   `docker/build-push-action@v7` step (`push: true`, no `load: true`),
   using the GitHub Actions cache (`cache-from: type=gha`,
   `cache-to: type=gha,mode=max`). The CPU/import smoke test
   (`smoke_test.py --mode cpu`) and the dependency assertions
   (`verify_environment.py`) both run as `RUN` layers **inside**
   `Dockerfile.runpod` (see Sections 4 and 2 above) — if either fails,
   `buildx` fails before producing a final image, so `push: true` never
   gets a chance to push a broken image. This also avoids `load: true`
   pulling the multi-GB image into the runner's local Docker store just to
   re-run a check that already ran during the build.
8. Verifies the pushed manifest reports `linux/amd64`
   (`docker buildx imagetools inspect`).
9. Writes the final image name and digest (captured directly from the
   `build-push-action` step's `digest` output) to the GitHub Actions Job
   Summary.

### Important: GitHub Actions can only run the CPU smoke test

GitHub-hosted runners have **no GPU**. This workflow validates imports,
package versions, and packaging correctness only
(`--mode cpu`, see Section 4 above). **GPU acceptance
(`--mode gpu` — `torch.cuda.is_available()`, GPU name, Mitsuba CUDA
variant) must still be run manually on an actual RunPod GPU pod** before
trusting a newly pushed tag for production workloads. The workflow's job
summary explicitly reminds you of this every run.

### Not automated by design

- No trigger other than manual `workflow_dispatch` — this avoids
  accidentally rebuilding/pushing a multi-GB image on every commit.
- No cleanup of the GitHub runner's disk in this first version. If the
  runner runs out of space, the job fails with `df -h` output already
  printed for diagnosis, rather than silently deleting unrelated runner
  state via a third-party cleanup action.
- No registry credentials are stored in the workflow file itself — only
  `${{ vars.DOCKERHUB_USERNAME }}` and `${{ secrets.DOCKERHUB_TOKEN }}`
  references.

## 10. Known limitation of this session

No Docker daemon was available in this environment, so `docker build` /
`docker run` were not executed here. All files were written and manually
reviewed against the current, validated RunPod environment (package
versions cross-checked against `.venv/bin/pip freeze` and
`handoff20260817.md` / `handoff20260818.md`), but the actual image build
and the GPU-dependent smoke test still need to be run by you (or in CI) on
a machine with Docker and, for the CUDA checks, a GPU.
