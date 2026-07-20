# Environment Setup

## Interpreter and tested build

Use `.venv/bin/python` or `.conda*/bin/python` inside the workspace. Record:

- interpreter and imported torch paths
- `torch.__version__`, `torch.version.git_version`, build date/source, and XPU
  device identity
- whether the build contains each known linked fix commit
- `python -m torch.utils.collect_env` at `artifacts/collect_env.txt`

Use a current XPU nightly unless the caller explicitly selects another build.
When a relevant upstream fix is newer than the tested build, refresh the nightly
or classify the case as a verification gap; never infer an independent XPU bug
from a build that cannot contain the shared fix.

Use ambient GitHub MCP or `gh` credentials only in collection/review processes.
Never store a token in an artifact or pass credentials to a repro process.

## Controls

Run an eager control that imports torch, allocates and operates on an XPU tensor,
checks its device, and synchronizes. If any case uses compiler paths, also run a
minimal XPU `torch.compile` control.

Write immutable `artifacts/environment.json` before candidate execution:

```json
{
  "interpreter": "<workspace-local path>",
  "torch": {
    "version": "<version>",
    "git_version": "<commit>",
    "path": "<imported package path>",
    "build_source": "<nightly or caller-selected build>",
    "build_date": "<date>",
    "linked_fix_containment": [
      {"commit": "<sha>", "contained": true}
    ]
  },
  "xpu": {"available": true, "device": "<device>"},
  "collect_env": "artifacts/collect_env.txt",
  "controls": [
    {"name": "eager-xpu", "command": "<command>", "result": "pass", "log": "<path>"}
  ],
  "repro_sandbox": {"result": "pass", "policy": "<policy summary>"}
}
```

A failed eager control blocks all execution. A failed compiler control blocks
compiler cases. An unresolved control incident sets workflow status to `blocked`,
not `completed`.

Never modify `environment.json` after the first attempt starts. Record later
incidents and retries in mutable `artifacts/environment_incidents.json`; evidence
hashes the immutable environment snapshot.

## Repro sandbox and attempt isolation

Fetched GitHub text and copied repro code are untrusted. Review each repro before
execution and run it with:

- only the current attempt namespace writable; keep ledgers, prior evidence, and
  the rest of the run directory read-only
- network access and secret-bearing home/config paths denied
- GitHub tokens, proxy credentials, SSH agents, credential helpers, and unrelated
  environment variables removed
- an isolated unprivileged UID/user namespace and PID namespace or equivalent;
  deny signaling, tracing, or inspecting unrelated processes
- CPU, memory, process-count, file-size, and wall-time limits; terminate the
  complete process group/cgroup on timeout
- a restricted syscall profile where the platform supports it
- a fresh process, temporary directory, and cache namespace

If the available runtime cannot enforce these boundaries while exposing XPU,
record a `security-sandbox` incident and do not execute the repro.

Use one namespace per control/case attempt:

```text
artifacts/cache/<case-or-control>/<attempt-id>/{inductor,triton,xdg,tmp}/
```

Point Inductor, Triton, XDG, and temporary-directory variables there. Never reuse
an earlier attempt as evidence. Discard the namespace after a device assert,
segfault, timeout, or suspected corruption.

## Environment incidents

One normalized dependency, topology, loader, cache, control, or sandbox failure
affecting multiple cases is one incident. Record its id, signature, root cause,
affected case keys, remedy, and clean-retry result in
`environment_incidents.json`.
Repair or isolate it and rerun every affected case. An unresolved incident keeps
those cases blocked and prevents workflow completion.

Each incident has `status: active|retrying|resolved`, creation/resolution times,
and clean-retry evidence. After a remedy, set `retrying` and rerun affected
cases. Set `resolved` only after successful clean retries; otherwise return it
to `active` and block those cases. Historical resolved incidents do not block
completion.
