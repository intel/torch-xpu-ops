# Environment Setup — Shared Reference

Used by `fix/implement` and `fix/verify`. Covers activating the XPU
environment, building PyTorch from source, and the torch-xpu-ops submodule
pin workflow.

`$PYTORCH_DIR` is the local PyTorch checkout. Set it before running any
commands.

## Activate the environment (MANDATORY)

Run before ANY test or import of `torch`. Without this, `torch.xpu.is_available()`
returns False and XPU tests collect 0 items.

1. **Source oneAPI** — use the `source-oneapi` skill. Skip if PyTorch was
   installed via pip (not built from source).
2. **Activate the Python environment:**
   ```bash
   source $PYTORCH_DIR/.venv/bin/activate  # adjust to your setup
   ```

## Build (only when C++/SYCL code changed)

Python-only changes need no rebuild.

```bash
# Source oneAPI first (use source-oneapi skill), then:
git submodule sync && git submodule update --init --recursive \
  && TORCH_XPU_ARCH_LIST=pvc USE_XPU=1 pip install -e . -v --no-build-isolation 2>&1 | tail -20
```

`TORCH_XPU_ARCH_LIST=pvc` targets Data Center GPU Max — change to match your
hardware (`dg2` for Arc GPU). If `torch.xpu.is_available()` returns False after
build, oneAPI was not sourced correctly; clean rebuild needed.

## torch-xpu-ops submodule pin (xpu.txt)

PyTorch pins torch-xpu-ops at a specific commit via `third_party/xpu.txt`.
CMake reads this SHA and checks out that commit into `third_party/torch-xpu-ops/`.

**To test a torch-xpu-ops fix inside PyTorch without a full rebuild:**
1. Copy only the changed files into `$PYTORCH_DIR/third_party/torch-xpu-ops/`.
2. Run `ninja -C $PYTORCH_DIR/build` for an incremental rebuild.
3. Run tests from `$PYTORCH_DIR/`.
4. After testing, restore: `cd $PYTORCH_DIR/third_party/torch-xpu-ops && git checkout .`

**Do NOT** `git checkout <branch>` inside `third_party/torch-xpu-ops/` — changes
mtimes on all files and triggers a massive rebuild. Copy only changed files.
**Do NOT** modify `third_party/xpu.txt` — triggers CMake reconfiguration and a
full rebuild from scratch.

## Rebuild pitfalls

- **Always rebuild after rebase or branch switch.** Stale C++ extensions produce
  unreliable or silently wrong test results.
- After editing a C++ header, manually copy it to `torch/include/` — editable
  installs serve C++ headers from the installed path, not source.
- Delete the PCH cache after modifying inductor headers:
  ```bash
  rm -rf /tmp/torchinductor_$USER/precompiled_headers/
  ```
