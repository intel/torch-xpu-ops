---
name: ut-refactor-review
description: Review PyTorch upstream unit-test (UT) PRs that enable Intel GPU (XPU) on existing tests. Use when reviewing PRs under test/ that port device-generic tests to XPU, add allow_xpu=True, generalize CUDA-hardcoded tests, or add XPU skips/xfails/tolerance overrides in OpInfo.
---

# XPU UT Refactor Review Skill

Review PyTorch (`pytorch/pytorch`) pull requests that **enable XPU on existing
upstream unit tests**. These PRs almost never add new operator logic; they make
existing tests device-agnostic and opt XPU into them. The review must focus on
what CI cannot check: whether the generalization preserves the original test
intent, whether device gating is precise, and whether every skip/xfail is
justified and traceable.

## Scope: when this skill applies

Use this skill (instead of the generic [`pr-review`](https://github.com/pytorch/pytorch/blob/main/.claude/skills/pr-review/SKILL.md)
skill in `pytorch/pytorch`) when the diff is predominantly:
- `test/**` changes that swap CUDA-hardcoded constructs for device-generic ones
- `instantiate_device_type_tests(..., allow_xpu=True)` additions
- `onlyAccelerator` / `onlyNativeDeviceTypesAnd([...])` decorator migrations
- XPU entries in OpInfo (`common_methods_invocations.py`, `opinfo/definitions/*`):
  `DecorateInfo(... device_type='xpu' ...)`, `toleranceOverride`, skips, xfails
- New `TestXxxDevice` classes split out from a device-agnostic `TestXxx`

If the PR also changes operator kernels or `native_functions.yaml`, hand those
files to the [`pr-review`](https://github.com/pytorch/pytorch/blob/main/.claude/skills/pr-review/SKILL.md)
skill and apply this skill only to the test files.

## Usage Modes

### No Argument

If invoked with no arguments, **do not review**. Ask:

> What would you like me to review?
> - A PR number or URL (e.g., `159118` or the full PR URL)
> - A local branch

### PR Mode

```
/ut-refactor-review 159118
/ut-refactor-review https://github.com/pytorch/pytorch/pull/159118
/ut-refactor-review 159118 detailed
```

Obtain the PR title, description, diff, changed-file list, and existing review
comments before reviewing. If the command does not name a repo, default to
fetching the PR from `pytorch/pytorch`.

Suggested fetch commands (CLI environments with `gh`):

    gh pr view <PR_NUMBER> --repo pytorch/pytorch --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
    gh pr diff <PR_NUMBER> --repo pytorch/pytorch
    gh pr view <PR_NUMBER> --repo pytorch/pytorch --json comments,reviews

### Local Branch Mode

```
/ut-refactor-review branch
/ut-refactor-review branch detailed
```

Review the current branch's changes relative to `main` (diff, commit log, and
changed-file list). Use the branch name in the review header instead of a PR
number.

## Review Philosophy

Go through **every changed line** against
[references/xpu-ut-review-checklist.md](references/xpu-ut-review-checklist.md).
For anything this skill does not address, defer to the
[`pr-review`](https://github.com/pytorch/pytorch/blob/main/.claude/skills/pr-review/SKILL.md)
skill in `pytorch/pytorch`.

## Files to Reference

- [references/xpu-ut-review-checklist.md](references/xpu-ut-review-checklist.md) — the line-by-line checklist
- `torch/testing/_internal/common_device_type.py` — `instantiate_device_type_tests`, `onlyAccelerator`, `allow_xpu`, `only_for`
- `torch/testing/_internal/common_utils.py` — `TEST_XPU`, `TEST_CUDA`, `TEST_HPU`, `xfailIf`, `HardwareClassification`
- `torch/testing/_internal/common_methods_invocations.py`, `torch/testing/_internal/opinfo/definitions/*` — OpInfo `DecorateInfo`, `toleranceOverride`
