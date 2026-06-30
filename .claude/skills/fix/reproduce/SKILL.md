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
  `NO_REPRODUCER` immediately.
- `ci_commit` — upstream commit hash from the CI report (optional but
  recommended).
- `pytorch_dir` — path to a local PyTorch checkout (optional). If absent and
  stage 2 is needed, clone to `agent_space_xpu/pytorch/`.

## Stage 1: Nightly Wheel (fast path)

Most failures reproduce here. Start here before doing anything heavier.

### Install

```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

### Run test

Read [../references/run-test.md](../references/run-test.md) now for path
resolution, command format, and result interpretation.

### Decision

| Result | Condition | Action |
|--------|-----------|--------|
| `CANNOT_VERIFY` | env problem (wheel install failed, runtime missing) | Report to orchestrator, stop |
| `REPRODUCED` | FAILED | Return `REPRODUCED(stage=nightly, refined_command=...)` |
| → stage 2 | PASSED (any nightly age) | Proceed to source build at `origin/main` to confirm |

## Stage 2: Source Build at origin/main

Nightly passing is not conclusive — it may lag behind CI or not reflect the
exact environment. Build from `origin/main` to verify.

### Prepare pytorch checkout

If `pytorch_dir` is provided: `git -C $pytorch_dir fetch origin && git -C $pytorch_dir checkout origin/main`

If not provided, clone:
```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git \
  agent_space_xpu/pytorch
git -C agent_space_xpu/pytorch checkout origin/main
git -C agent_space_xpu/pytorch submodule update --init --recursive
```

### Build and run

Activate environment and build (see `/xpu-build-pytorch` skill and
domain skill for build command), then run the test (see
[../references/run-test.md](../references/run-test.md)).

### Decision

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=source_build, refined_command=...)` |
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
  refined_command: <exact command that reproduced the failure>
  env_diff: <environment differences found, if stage=ci_env>

NOT_REPRODUCED
  reason: <which stage passed and what was checked>

NO_REPRODUCER
  (no reproducer_command was provided)

CANNOT_VERIFY
  stage: nightly | source_build | ci_env
  blocker: <what went wrong>
```

The orchestrator decides the next step based on this output.
