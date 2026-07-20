# Environment Setup

## Interpreter and access

- Use `.venv/bin/python` or `.conda*/bin/python` inside the workspace. Create an
  in-workspace environment when none exists.
- Refresh the XPU nightly when requested:

  ```bash
  python -m pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/xpu
  ```

- Use ambient GitHub MCP or `gh` credentials. Never store a token in an artifact.
- Save `python -m torch.utils.collect_env` to `artifacts/collect_env.txt`.

## Controls

Run an eager control that imports torch, confirms XPU availability, allocates and
operates on an XPU tensor, checks its device, and synchronizes. When any candidate
uses compiler paths, also run a minimal XPU `torch.compile` control.

Write `artifacts/environment.json` before candidate execution:

```json
{
  "schema_version": 3,
  "interpreter": "<workspace-local path>",
  "torch_version": "<version>",
  "xpu_available": true,
  "collect_env": "artifacts/collect_env.txt",
  "controls": [{"name": "eager-xpu", "result": "pass", "log": "<path>"}],
  "cache_policy": "per-attempt",
  "incidents": []
}
```

Record each control's command, result, and log. A failed eager control blocks all
execution. A failed compiler control blocks compiler cases until repaired.

## Attempt isolation

Give every control and repro attempt a fresh namespace:

```text
artifacts/cache/<case-or-control>/<attempt-id>/
  inductor/
  triton/
  xdg/
  tmp/
```

Point `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, and
temporary-directory variables at that namespace. Run each attempt in a fresh
process. Never use an earlier run, case, or attempt cache as validation evidence.
After a device assert, segfault, timeout, or suspected cache corruption, discard
the namespace and retry cleanly.

Issue-eligible behavior must repeat in at least two independent process/cache
namespaces with the same stage and normalized signature. A signature that changes
under clean isolation is `verification-gap`, not a confirmed issue.

## Environment incidents

An incident is one normalized dependency, topology, loader, registry, cache, or
control failure affecting multiple cases. Record one id, signature, root cause,
affected case keys, attempted remedy, and clean-retry result. Attach every
affected case to it.

Do not turn one incident into multiple candidate bugs. Repair or isolate it and
rerun all affected cases. A persistent clean-environment failure may be reported
as one incident; its candidate cases remain blocked.
