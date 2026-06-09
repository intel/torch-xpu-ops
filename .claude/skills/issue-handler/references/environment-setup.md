---
name: environment-setup
description: oneAPI/python activation, build commands, and the torch-xpu-ops xpu.txt pin workflow.
---

# Environment Setup

This doc contains the env settings and build commands for PyTorch Development.
All paths and flags below are **environment-dependent placeholders** — adapt them to the local machine (oneAPI install location, target GPU arch, the PyTorch checkout's Python
environment). `$PYTORCH_DIR` is the local PyTorch checkout (e.g. `~/pytorch`).

## Activate the environment (MANDATORY)

Run these before ANY test or import of `torch`. Without them
`torch.xpu.is_available()` returns False and XPU tests collect 0 items:
```bash
# Source oneAPI if the pytorch is locally built.
# If it is installed via pip, this step is not needed.
source ~/intel/oneapi/setvars.sh --force 2>/dev/null
# Activate the Python environment (adjust the path to your local setup)
# For example, if it is the conda env, source the conda env instead.
source $PYTORCH_DIR/.venv/bin/activate
```
All test/build commands run from `$PYTORCH_DIR/`.

## Build (only if you changed C++/CUDA/SYCL code)

Python-only changes need no rebuild.
After editing C++/SYCL, rebuild before verifying:

- **pytorch repo:**
  ```bash
  source ~/intel/oneapi/setvars.sh --force 2>/dev/null \
    && git submodule sync && git submodule update --init --recursive \
    && TORCH_XPU_ARCH_LIST=pvc USE_XPU=1 pip install -e . -v --no-build-isolation 2>&1 | tail -20
  ```
  (`TORCH_XPU_ARCH_LIST=pvc` is the target GPU arch — change to match yours.)
- **torch-xpu-ops repo:** no separate build step (it builds as part of pytorch).

If after build `torch.xpu.is_available()` returns False, it is highly likely the oneAPI was not sourced correctly; a clean rebuild is needed.

## torch-xpu-ops submodule pin (xpu.txt)

pytorch pins torch-xpu-ops at a specific commit via `third_party/xpu.txt`.
During the CMake build pytorch reads this SHA, fetches torch-xpu-ops, and checks
out that exact commit into `third_party/torch-xpu-ops/`.

**To test a torch-xpu-ops fix inside pytorch:**
1. Copy only the changed files into `$PYTORCH_DIR/third_party/torch-xpu-ops/`.
2. Run `ninja -C $PYTORCH_DIR/build` for an incremental rebuild (recompiles only
   changed files).
3. Run tests from the pytorch root directory.
4. After testing, restore:
   `cd $PYTORCH_DIR/third_party/torch-xpu-ops && git checkout .`

**Do NOT** do a full `git checkout <branch>` in `third_party/torch-xpu-ops/` —
it changes mtimes on all files and triggers a massive ninja rebuild. Copy only
the changed files to keep incremental builds fast.
**Do NOT** modify `third_party/xpu.txt` — changing the pin triggers CMake
reconfiguration and a full rebuild from scratch (~hours).

## Rebuild pitfalls

- **Always rebuild after rebase or branch switch.** After `git rebase`,
  `git checkout`, or any operation that changes the commit base, rebuild before
  running tests — otherwise C++ extensions are stale and results are unreliable.
- Editable installs resolve Python from source but C++ headers from the
  installed location (`torch/include/`). After editing a C++ header, **manually
  copy** it to the installed include path.
- Delete the PCH cache (`/tmp/torchinductor_<user>/precompiled_headers/`) after
  modifying any header under `torch/csrc/inductor/cpp_wrapper/` — stale
  precompiled headers mask the fix.
