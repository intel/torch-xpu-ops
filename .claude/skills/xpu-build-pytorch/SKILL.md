---
name: xpu-build-pytorch
description: Build PyTorch from source with Intel XPU (GPU) support. Use when the user asks to build PyTorch, set up the XPU build environment, rebuild after code changes, install PyTorch for XPU, or configure oneAPI. Handles prerequisites check, build_pytorch.env setup, build verification, and torch-xpu-ops development pin override.
---

# Build PyTorch with XPU Support

This project builds only as part of PyTorch. No standalone build exists.
The only build command is `pip install -e . -v --no-build-isolation`. Never use any other command.

## Quick start

With `build_pytorch.env` already configured (see Instructions):

```bash
source build_pytorch.env
pip install -e . -v --no-build-isolation
python -c "import torch; print('XPU available:', torch.xpu.is_available())"
```

## Instructions

Always check local memory for build configuration (env vars, paths, incremental-build shortcuts)
before running the build. Apply what you find; if nothing applicable is in memory, ask the user.

### 1. Verify prerequisites

```bash
# Confirm in PyTorch root (xpu.txt is unique to this repo)
test -f third_party/xpu.txt || echo "ERROR: Not in PyTorch root"

# Verify oneAPI installation (icpx not in PATH before sourcing vars — check directory instead)
test -d /opt/intel/oneapi/compiler || echo "WARNING: oneAPI not found at /opt/intel/oneapi/compiler"
```

### 2. Configure `build_pytorch.env`

Create in PyTorch root. Adjust paths for your system.

```bash
# Target hardware: pvc (Data Center GPU Max), dg2 (Arc GPU), etc.
# Omitting targets all supported architectures — safe default.
# export TORCH_XPU_ARCH_LIST=pvc

export USE_XPU=1
export USE_CUDA=0

# Adjust to your oneAPI installation path
source /opt/intel/oneapi/compiler/latest/env/vars.sh
# Alternative: source /opt/intel/oneapi/setvars.sh
# Optional PTI support: source /opt/intel/oneapi/pti/latest/env/vars.sh
```

### 3. Build

```bash
source build_pytorch.env
pip install -e . -v --no-build-isolation
```

### 4. Verify

```bash
source build_pytorch.env
python -c "import torch; print('XPU available:', torch.xpu.is_available())"
```

Expected: `XPU available: True`

## Examples

**Debug / faster iteration build** (shrinks translation unit scope):

```bash
source build_pytorch.env
BUILD_SEPARATE_OPS=1 pip install -e . -v --no-build-isolation
```

**Verify device count after build:**

```bash
source build_pytorch.env
python -c "import torch; print(torch.xpu.device_count(), 'XPU device(s) found')"
```

## Best practices

- **Never skip sourcing `build_pytorch.env`** before running tests — XPU ops are unavailable without the oneAPI runtime.
- **Always rebuild after `git rebase` or `git checkout`** — stale C++ extensions produce unreliable or silently wrong test results.
- **After editing a C++ header**: manually copy to `torch/include/` — editable installs serve C++ headers from the installed path, not source.
- **After modifying inductor headers**: delete the PCH cache before rebuilding:
  ```bash
  rm -rf /tmp/torchinductor_$USER/precompiled_headers/
  ```
- **Leave `TORCH_XPU_ARCH_LIST` commented out** unless you know your target hardware — setting `pvc` on an Arc GPU produces a silently incorrect build.

## Requirements

- Intel oneAPI Base Toolkit installed (provides `icpx`, `libsycl`, DPC++ runtime)
- PyTorch cloned from source with `third_party/torch-xpu-ops` populated
- Python environment with build dependencies (`ninja`, `cmake`, `pyyaml`, `typing_extensions`)

## Advanced usage

For developing `torch-xpu-ops` locally (commit pin override so CMake does not overwrite
your changes on every build), see [reference.md](reference.md).
