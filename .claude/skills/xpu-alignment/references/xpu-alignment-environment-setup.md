# Environment and Preflight

## Workspace-local interpreter

Use `.venv/bin/python` or `.conda*/bin/python` inside the workspace. If none
exists, create an in-workspace virtual environment. Never use an unrelated
interpreter.

Use a current XPU nightly unless the caller explicitly selects another build:

```bash
python -m pip install --pre --upgrade torch \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

Record the interpreter path, imported torch path, `torch.__version__`,
`torch.version.git_version`, build source/date, and XPU device in the scan report.
When a linked fix is newer than the tested build, refresh the nightly before
deciding whether the fix works.

## Controls

Run an eager control that imports torch, allocates and operates on an XPU tensor,
checks its device, and synchronizes. If selected candidates use `torch.compile`,
also run a minimal XPU compile control.

A failed eager control blocks all candidate execution. A failed compile control
blocks compiler candidates. Record one shared environment blocker and repair it
before rerunning affected candidates; do not report each candidate as an
independent XPU issue.

## GitHub access

Use the GitHub MCP server or ambient `gh` authentication. Never hardcode a token
or pass credentials to a repro process.

## Preflight checklist

1. Verify the workspace interpreter and tested-build provenance.
2. Pass the eager XPU control and any required compile control.
3. Verify read-only GitHub access.
4. Create `artifacts/details`, `reports`, and `scripts`.
5. Save:

   ```bash
   python -m torch.utils.collect_env > artifacts/collect_env.txt
   ```

6. Configure a fresh temporary and Inductor/Triton cache namespace for each repro
   attempt. Do not reuse a process or cache after a device assert, crash, or
   timeout.

## Common failures

- XPU unavailable: load the oneAPI runtime and verify the driver.
- Torch import failure: reinstall the XPU nightly in the workspace environment.
- GitHub access failure: verify ambient MCP or `gh auth` credentials.
- Shared Dynamo, loader, dependency, topology, or cache failure: isolate it as one
  environment blocker, repair it, and rerun every affected candidate.
