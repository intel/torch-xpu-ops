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
  stage 2 is needed, clone to `agent_space/pytorch/`.

## Stage 1: Nightly Wheel (fast path)

Most failures reproduce here. Start here before doing anything heavier.

### Install

Use the nightly wheel install command from the domain skill loaded by the
orchestrator (e.g. `fix/domains/xpu-kernel`).

### Check commit alignment (if ci_commit provided)

```bash
python -c "import torch; print(torch.version.git_version)"
```

Compare `nightly_commit` vs `ci_commit` using:
```bash
# From any pytorch checkout — determines which commit came first
git -C <any_pytorch_dir> merge-base --is-ancestor <ci_commit> <nightly_commit> \
  && echo "nightly is newer" || echo "nightly is older"
```

### Run test

Read [../references/run-test.md](../references/run-test.md) now for path
resolution, command format, and result interpretation.

### Decision

| Result | Condition | Action |
|--------|-----------|--------|
| `CANNOT_VERIFY` | env problem (wheel install failed, runtime missing) | Report to orchestrator, stop |
| `REPRODUCED` | FAILED | Return `REPRODUCED(stage=nightly, refined_command=...)` |
| → stage 2 | PASSED and nightly_commit is older than ci_commit | nightly is stale, proceed to stage 2 |
| → stage 3 | PASSED and nightly_commit is same or newer than ci_commit | inconclusive, proceed to stage 3 |

## Stage 2: Source Build at CI Commit

Only reached when nightly is too old to be conclusive.

### Prepare pytorch checkout

If `pytorch_dir` is provided: `git -C $pytorch_dir checkout <ci_commit>`

If not provided, clone:
```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git \
  agent_space/pytorch
git -C agent_space/pytorch checkout <ci_commit>
git -C agent_space/pytorch submodule update --init --recursive
```

### Build and run

Activate environment and build (see
[../references/environment-setup.md](../references/environment-setup.md) and
domain skill for build command), then run the test (see
[../references/run-test.md](../references/run-test.md)).

### Decision

| Result | Action |
|--------|--------|
| `CANNOT_VERIFY` | Report to orchestrator, stop |
| `REPRODUCED` | Return `REPRODUCED(stage=source_build, refined_command=...)` |
| `PASSED` | Proceed to stage 3 |

## Stage 3: CI Environment Alignment

Only reached when both nightly and source build pass locally.
The failure may be CI-environment-specific.

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
