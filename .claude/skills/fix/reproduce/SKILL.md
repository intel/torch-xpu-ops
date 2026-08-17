---
name: fix/reproduce
description: >
  Verify whether a bug still exists by running tests against nightly wheel,
  source build, or CI environment — in that order. Used by both issue-handler
  and nightly-ci-fix orchestrators before starting a fix.
---

# Reproduce — Verify the Bug Exists

Runs a test and determines whether the bug reproduces. Uses a three-stage
approach: nightly wheel first (fast), source build at CI commit second (precise),
CI environment alignment third (last resort).

The orchestrator decides what to do with the output — this skill only reports
the result.

## Inputs

- `reproducer_command` — pytest/python command or test name. If absent, return
  `NO_REPRODUCER` immediately. Also return `NO_REPRODUCER` if the command runs
  but pytest reports `collected 0 items` (the referenced test does not exist
  in the tree at this base) — this is the same "no runnable reproducer"
  condition as an absent command, just discovered a step later.

  Providers (set by the orchestrator, not this skill):
  - **Issue body** — extracted by `issue-triage` from the reproducer section
    (via `issue-handler`).
  - **CI failure log** — the failing pytest node id from the nightly CI report
    (via `xpu-nightly-ci-fix`).
  - **Per-entry loop** — one test entry from a skip-list issue (via
    `fix/skip-list`).

  `fix/root-cause` is **not** a provider. It runs after this skill and only
  analyzes; if it finds the reproducer was wrong (e.g. wrong assertion), it
  asks the orchestrator to re-invoke this skill with a corrected command.
- `ci_commit` — upstream commit hash from the CI report. Only used as a
  fallback base when `origin/main` fails to build (optional).
- `pytorch_dir` — path to a local PyTorch checkout (optional). If absent and
  stage 2 is needed, clone to `agent_space_xpu/pytorch/`.

## Stage 1: Nightly Wheel (fast path)

Most failures reproduce here. Start here before doing anything heavier.

### Install

Always reproduce against the **latest** available XPU nightly. Do not reuse a
stale wheel from a previous session — a bug may already be fixed in a newer
nightly, and re-verifying an old wheel produces misleading `REPRODUCED`
results.

```bash
# Query the latest nightly, then install/upgrade to it explicitly.
pip3 index versions torch --pre \
  --index-url https://download.pytorch.org/whl/nightly/xpu | head -1
pip3 install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

If `torchvision`/`torchaudio` pin `torch` back to an older build, reinstall
`torch` alone with `--no-deps` to keep the newest, e.g.
`pip3 install --pre --upgrade --no-deps torch==<latest> --index-url ...`.

Record the exact wheel version used (`python -c "import torch;
print(torch.__version__)"`) in the reproduce output and in any issue comment,
so downstream stages and re-verifications know which nightly was tested.


### Working directory

Do NOT run the nightly-wheel reproducer with `cwd` inside any pytorch
source checkout. Python resolves `import torch` against the local `torch/`
package before site-packages, so it will load the in-tree `torch/_C.so`
built at whatever revision that tree happens to be — typically stale
relative to the installed wheel — and fail with
`ImportError: undefined symbol: ...`. Either `cd $(mktemp -d)` (or any
non-pytorch dir) before running, or invoke the reproducer with an
absolute path from outside the tree.

### torch-xpu-ops test invocation

Tests under `torch-xpu-ops/test/xpu/` use
`sys.path.append("../../../../test/functorch")` and pull `common_utils`,
`common_methods_invocations`, etc. from `pytorch/test/`. That relative
path only resolves when `cwd` is
`<pytorch_dir>/third_party/torch-xpu-ops/test/xpu/`. Standard setup:

- pytorch checkout at `agent_space_xpu/pytorch/`
- `third_party/torch-xpu-ops/` inside it symlinked (or cloned) to the
  working torch-xpu-ops tree
- run `pytest` from `<pytorch_dir>/third_party/torch-xpu-ops/test/xpu/`

The pytorch tree does NOT need to be built for the nightly-wheel path —
the wheel provides the runtime, the source tree only supplies the test
support modules.

### Use the test's own assertion

When writing a standalone reproducer for a `TestCase.assertEqual`
failure, use the test's own assertion. **Do NOT substitute**
`torch.allclose`, `torch.equal`, or bare `==` — they have different
(usually stricter) default tolerances and will manufacture false
positives.

If the failure log says `AssertionError: Tensor-likes are not close`,
the assertion is `torch.testing._comparison.assert_close`, which has
dtype-specific defaults (bf16: `rtol=0.016, atol=1e-5`). Reproduce
through `assert_close` or via `TestCase.assertEqual`:

```python
import sys; sys.path.insert(0, "<pytorch>/test")
from torch._dynamo.test_case import TestCase   # or the base class the failing test uses

class T(TestCase):
    def test_x(self, device):
        ...
        self.assertEqual(out_ref, out)

T().test_x(device='xpu')
```

### Run test

Run the test. Result interpretation: if `all skipped` by `@skipIfXpu`, load
`fix/skip-management` and follow its "Temporarily remove for reproduction"
procedure before concluding — only return `CANNOT_VERIFY` if the skip is
environmental (not an XPU marker). `xfailed` → `FAILED`.

### Decision

| Result | Condition | Action |
|--------|-----------|--------|
| `CANNOT_VERIFY` | env problem (wheel install failed, runtime missing) | Report to orchestrator, stop |
| `REPRODUCED` | FAILED | Return `REPRODUCED(stage=nightly, refined_command=...)` |
| → stage 2 | PASSED (any nightly age) | Proceed to source build at `origin/main` to confirm |

## Stage 2: Source Build at origin/main

Nightly passing is not conclusive — it may lag behind CI. Build from
`origin/main` to verify. Even when the failure came from a specific CI
commit, we only consider fixes on top of `origin/main` — downstream
stages branch off it.

### Prepare pytorch checkout

If `pytorch_dir` is provided: `git -C $pytorch_dir fetch origin`

If not provided, clone:
```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git \
  agent_space_xpu/pytorch
```

Then:
```bash
git -C $pytorch_dir checkout origin/main
git -C $pytorch_dir submodule update --init --recursive
```

Leave HEAD on `origin/main` at exit.

### Build and run

Load the `xpu-build-pytorch` skill and follow it for the build. Do not
hand-roll the build here.

If the `origin/main` build fails for a reason unrelated to the bug
(broken trunk, upstream infra issue, etc.) and `ci_commit` is
available, fall back once:

```bash
git -C $pytorch_dir checkout $ci_commit
git -C $pytorch_dir submodule update --init --recursive
```

Rebuild via `xpu-build-pytorch`. If this succeeds, proceed with the
test on `ci_commit` and record `base=<ci_commit_sha>` in the output so
the orchestrator branches its fix off the same base. If the fallback
build also fails, escalate as
`CANNOT_VERIFY(blocker=trunk and ci_commit both fail to build)`
rather than silently reproducing on some other base.

Then run the test. Result interpretation: same `all skipped` handling as
Stage 1 (load `fix/skip-management` and try its "Temporarily remove for
reproduction" procedure first); `xfailed` → `FAILED`.

### Decision

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=source_build, base=origin/main\|<ci_commit_sha>, refined_command=...)` |
| `PASSED` | Proceed to stage 3 |

## Stage 3: CI Environment Alignment

Only reached when nightly wheel and source build at `origin/main` both pass.
The failure may be specific to the CI environment (docker image, artifacts, env vars).

### Set up the CI environment

Use the setup script to automatically find the latest run with successful
linux builds, download the matching wheels, pull the CI docker image, and
generate a ready-to-run container command:

```bash
bash .claude/skills/fix/reproduce/scripts/ci_env_setup.sh \
  $pytorch_dir \
  --py 3.10 \
  --outdir agent_space_xpu/ci_env
```

The script prints the exact `docker run` command and saves it to
`agent_space_xpu/ci_env/run_container.sh`. Inside the container:

```bash
pip install /workspace/wheels/*.whl --pre
# then run the failing test
```

### What to check if the failure still doesn't reproduce

From the CI job log, extract and align any remaining differences:
- Full test command with all flags (`--timeout`, `-x`, specific env vars)
- Any environment variables set in the CI job

### Decision

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=ci_env, refined_command=..., env_diff=...)` |
| `PASSED` | Return `NOT_REPRODUCED` — issue no longer exists; orchestrator reports to user or triage collects reason |

## Output

Return one of these to the orchestrator:

```
REPRODUCED
  stage: nightly | source_build | ci_env
  base: origin/main | <ci_commit_sha>    # base for downstream build (default origin/main; ci_commit_sha only when source_build fell back)
  refined_command: <exact command that reproduced the failure>
  env_diff: <environment differences found, if stage=ci_env>

NOT_REPRODUCED
  reason: <which stage passed and what was checked>

NO_REPRODUCER
  reason: no_command | collected_zero
  (returned when either no reproducer_command was provided, or pytest
  reports `collected 0 items` for the provided command)

CANNOT_VERIFY
  stage: nightly | source_build | ci_env
  blocker: <what went wrong>
```

The orchestrator decides the next step based on this output.
