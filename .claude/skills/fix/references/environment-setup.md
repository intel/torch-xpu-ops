# Environment Setup — Shared Reference

Used by `fix/implement` and `fix/verify`. Covers activating the environment
and when to rebuild. Specifics (build command, env vars, arch flags) are in
the domain skill loaded by the orchestrator.

`$PYTORCH_DIR` is the local PyTorch checkout. Set it before running any
commands.

## Activate the environment (MANDATORY)

Run before ANY test or import. Without this, backend device tests may collect
0 items or silently produce wrong results.

1. **Activate the Python environment:**
   ```bash
   source $PYTORCH_DIR/.venv/bin/activate  # adjust to your setup
   ```
2. **Source any required runtime** — see domain skill for backend-specific
   steps (e.g. `xpu-build-pytorch` for XPU).

## Build (only when C++/SYCL code changed)

Python-only changes need no rebuild. When C++ or SYCL files are changed,
rebuild using the command from the domain skill before running tests.

## Rebuild pitfalls

- **Always rebuild after rebase or branch switch.** Stale C++ extensions
  produce unreliable or silently wrong test results.
- After editing a C++ header, manually copy it to `torch/include/` — editable
  installs serve C++ headers from the installed path, not source.
- Delete the PCH cache after modifying inductor headers:
  ```bash
  rm -rf /tmp/torchinductor_$USER/precompiled_headers/
  ```
