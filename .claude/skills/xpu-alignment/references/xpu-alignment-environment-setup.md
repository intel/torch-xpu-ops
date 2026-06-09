---
name: xpu-alignment-environment-setup
description: How to set up the run environment before scanning. Covers finding or creating the workspace XPU Python interpreter, installing or refreshing the XPU nightly, checking GitHub access, and fixing common preflight failures.
---

# Environment & Preflight

This skill is self-contained and relies on no helper scripts. Everything below
is performed inline with the workspace XPU interpreter and the GitHub MCP server
(or `gh` CLI fallback).

## Workspace-local XPU interpreter

- Look for `.venv/bin/python` or `.conda*/bin/python` inside the workspace.
  Never use an interpreter outside the workspace.
- If no workspace venv exists, create one in-workspace: `python -m venv .venv`.
- Ensure the latest XPU nightly is installed by running the upgrade install
  below. `pip` resolves to the newest nightly and no-ops if the workspace is
  already current, so there is no separate "is it stale?" judgment to make:

  ```bash
  python -m pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/xpu
  ```

  Run this inline with the workspace interpreter.

## GitHub access

Use the GitHub MCP server, or the `gh` CLI as a fallback. Never hardcode tokens;
rely on the ambient `gh auth` / MCP credentials.

## Step 0 preflight checklist

1. Verify XPU torch import and `torch.xpu.is_available()`.
2. Run the upgrade install above so the workspace holds the latest nightly.
3. Verify GitHub access.
4. Create output directories: `artifacts/details`, `reports`, `scripts`.
5. Save `collect_env` output to `artifacts/collect_env.txt`.

## Preflight failure remedies

- `torch.xpu.is_available()` returns `False` -> ensure Intel oneAPI runtime is
  loaded (`source /opt/intel/oneapi/setvars.sh`) and the XPU driver is installed.
- `import torch` fails -> reinstall the XPU nightly wheel into the workspace venv.
- GitHub access fails -> verify auth (`gh auth status`) or MCP credentials.
- Output directory creation fails -> check filesystem permissions.
