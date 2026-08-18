#!/usr/bin/env bash
# RunPod container start command for the mMIMO_sleep image.
#
# Idempotent: safe to run multiple times (e.g. container restarts) without
# corrupting the environment. Does NOT run apt-get or reinstall
# PyTorch/Sionna RT — all of that is baked into the image at build time.
set -uo pipefail

VENV_PYTHON="/opt/venv/bin/python"
PROJECT_DIR="${PROJECT_DIR:-/workspace/mMIMO_sleep}"
KERNEL_NAME="mmimo-sleep"
KERNEL_DISPLAY_NAME="Python (mMIMO_sleep)"

log() {
    echo "[runpod-start] $*"
}

# ---------------------------------------------------------------------------
# 1. Sanity-check the baked-in Python environment.
# ---------------------------------------------------------------------------
if [ ! -x "${VENV_PYTHON}" ]; then
    log "FATAL: ${VENV_PYTHON} not found or not executable. The image is broken."
    exit 1
fi
log "Using interpreter: ${VENV_PYTHON} ($(${VENV_PYTHON} --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Diagnose, and only if confirmed necessary, fix a CUDA
#    forward-compatibility library shadowing the real host driver.
#
#    Background: consumer GPUs (e.g. RTX 4090) do not support NVIDIA
#    driver forward-compatibility. If a compat `libcuda.so.1` (usually
#    under /usr/local/cuda-*/compat/) is registered in the dynamic linker
#    cache ahead of the real, host-mounted driver library, PyTorch fails
#    with:
#        Error 804: forward compatibility was attempted on non supported HW
#    This has been observed to be injected by the RunPod host into a
#    freshly, randomly named /etc/ld.so.conf.d/*compat*.conf file on some
#    (but not all) pod instances — see handoff20260817.md Section 3 and
#    handoff20260818.md Section 3.2 for the historical incidents.
#
#    This function NEVER blindly deletes system files or host-injected
#    configs. It only acts after confirming, from ldconfig's own resolution
#    order, that (a) a compat library is currently winning priority over
#    (b) a real, non-compat driver library that is actually present. If
#    only a compat library exists (no real driver found at all), or if the
#    real driver is already winning, nothing is modified. Every decision
#    and every mutation is logged with an "AUDIT:" prefix. The mutation
#    itself (`mv foo.conf foo.conf.bak`) is idempotent: on a second run,
#    the renamed file no longer matches the `*compat*.conf` glob, so
#    ldconfig's resolution is re-diagnosed from scratch and no further
#    action is taken.
# ---------------------------------------------------------------------------
diagnose_and_fix_cuda_compat() {
    log "---- CUDA driver / library diagnostics ----"

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1 \
            | while IFS= read -r line; do log "nvidia-smi: ${line}"; done
    else
        log "nvidia-smi not found in PATH (driver tools not installed, or no GPU visible)."
    fi

    log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

    # All registered libcuda.so.1 providers, in ldconfig's priority order
    # (the first line is the one that will actually be loaded).
    mapfile -t all_libcuda < <(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF}')

    if [ "${#all_libcuda[@]}" -eq 0 ]; then
        log "ldconfig reports no libcuda.so.1 provider at all."
        log "Not modifying anything. CUDA will only work if the NVIDIA Container" \
            "Toolkit injects a driver library via a bind mount or LD_LIBRARY_PATH" \
            "that ldconfig does not yet know about."
        return
    fi

    log "ldconfig-registered libcuda.so.1 providers, in resolution order:"
    for p in "${all_libcuda[@]}"; do
        log "    ${p}"
    done

    local resolved="${all_libcuda[0]}"
    log "Currently winning (highest-priority) libcuda.so.1: ${resolved}"

    case "${resolved}" in
        */compat/*)
            log "The winning libcuda.so.1 is under a 'compat' (forward-compatibility)" \
                "directory. Checking whether a real, non-compat driver library is also" \
                "present but shadowed..."
            local real=""
            for p in "${all_libcuda[@]}"; do
                case "${p}" in
                    */compat/*) continue ;;
                    *) real="${p}"; break ;;
                esac
            done
            if [ -z "${real}" ]; then
                log "No non-compat libcuda.so.1 provider is present anywhere in ldconfig's" \
                    "cache. AUDIT: NOT modifying /etc/ld.so.conf.d/ — disabling the compat" \
                    "entry would leave the container with no libcuda.so.1 at all, which is" \
                    "worse than the current state."
                return
            fi
            log "CONFIRMED shadowing: real driver library exists at '${real}' but is" \
                "currently overridden by the compat library at '${resolved}'. This is the" \
                "exact condition that produces Error 804 on consumer GPUs. Disabling the" \
                "compat ld.so.conf.d entry so the real driver takes priority."
            local changed=0
            for f in /etc/ld.so.conf.d/*compat*.conf; do
                [ -e "$f" ] || continue
                mv "$f" "$f.bak"
                log "AUDIT: disabled ${f} -> ${f}.bak"
                changed=1
            done
            if [ "${changed}" = "1" ]; then
                ldconfig
                local new_resolved
                new_resolved="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')"
                log "AUDIT: re-ran ldconfig. libcuda.so.1 now resolves to: ${new_resolved:-<none>}"
            else
                log "WARNING: the winning provider looked like a compat path, but no" \
                    "matching /etc/ld.so.conf.d/*compat*.conf file was found to disable" \
                    "(it may be registered via /etc/ld.so.conf directly, or via a path" \
                    "baked into the base image rather than a conf.d fragment)." \
                    "No automatic action taken; investigate manually with 'ldconfig -p |" \
                    "grep libcuda' and 'grep -rl compat /etc/ld.so.conf*'."
            fi
            ;;
        *)
            log "The winning libcuda.so.1 (${resolved}) is NOT under a compat directory." \
                "No forward-compatibility shadowing detected; leaving ld.so.conf.d untouched."
            ;;
    esac
}
diagnose_and_fix_cuda_compat

# ---------------------------------------------------------------------------
# 3. Editable-install the project from the network volume, without
#    touching any pinned dependency (torch, sionna-rt, mitsuba, drjit, ...).
# ---------------------------------------------------------------------------
if [ -d "${PROJECT_DIR}" ] && [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
    log "Project directory found at ${PROJECT_DIR}; installing editable (--no-deps)..."
    if "${VENV_PYTHON}" -m pip install -e "${PROJECT_DIR}" --no-deps --no-build-isolation; then
        log "Editable install of mMIMO_sleep refreshed successfully."
    else
        log "WARNING: editable install of ${PROJECT_DIR} failed. Continuing startup" \
            "so SSH/Jupyter remain reachable for debugging."
    fi
else
    log "WARNING: ${PROJECT_DIR}/pyproject.toml not found."
    log "WARNING: is the RunPod Network Volume mounted at /workspace? Project code" \
        "will be unavailable until it is."
fi

# ---------------------------------------------------------------------------
# 4. Register/confirm the Jupyter kernelspec. The kernelspec is baked into
#    the image at /usr/local/share/jupyter/kernels/${KERNEL_NAME}, but we
#    re-assert it here in case a volume mount or prior manual change
#    shadowed it. This call is idempotent (it just overwrites kernel.json).
# ---------------------------------------------------------------------------
SYSTEM_KERNEL_JSON="/usr/local/share/jupyter/kernels/${KERNEL_NAME}/kernel.json"
if [ -f "${SYSTEM_KERNEL_JSON}" ]; then
    log "Jupyter kernelspec already present: ${SYSTEM_KERNEL_JSON}"
else
    log "Jupyter kernelspec missing, (re-)installing..."
    "${VENV_PYTHON}" -m ipykernel install \
        --prefix=/usr/local \
        --name "${KERNEL_NAME}" \
        --display-name "${KERNEL_DISPLAY_NAME}"
fi

# ---------------------------------------------------------------------------
# 5. Start SSHD (RunPod's standard connection method).
# ---------------------------------------------------------------------------
mkdir -p /var/run/sshd
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
fi

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    # Overwrite (not append) so re-runs stay idempotent instead of
    # accumulating duplicate keys.
    echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    log "Installed SSH public key from \$PUBLIC_KEY into /root/.ssh/authorized_keys."
else
    log "WARNING: \$PUBLIC_KEY is not set; no SSH key was installed. Set it in the" \
        "RunPod template's environment variables to enable SSH access."
fi

sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh start || log "WARNING: 'service ssh start' reported an error (may already be running)."

# ---------------------------------------------------------------------------
# 6. Optionally start JupyterLab if requested.
# ---------------------------------------------------------------------------
if [ "${START_JUPYTER:-1}" = "1" ]; then
    log "Starting JupyterLab on port 8888 (token/password disabled: RUNPOD_JUPYTER_TOKEN=${RUNPOD_JUPYTER_TOKEN:-<none>})"
    nohup "${VENV_PYTHON}" -m jupyter lab \
        --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --ServerApp.token="${RUNPOD_JUPYTER_TOKEN:-}" \
        --notebook-dir="${PROJECT_DIR}" \
        > /var/log/jupyterlab.log 2>&1 &
    log "JupyterLab logs: /var/log/jupyterlab.log"
fi

log "Startup complete. Container will keep running."
exec sleep infinity
