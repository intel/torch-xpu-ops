---
name: upstream-ut-review
description: Review PyTorch upstream unit-test (UT) PRs that enable Intel GPU (XPU) on existing tests. Use when reviewing PRs under test/ that port device-generic tests to XPU, add allow_xpu=True, generalize CUDA-hardcoded tests, or add XPU skips/xfails/tolerance overrides in OpInfo.
---

# XPU Upstream UT Review Skill

Review PyTorch (`pytorch/pytorch`) pull requests that **enable XPU on existing
upstream unit tests**. These PRs almost never add new operator logic; they make
existing tests device-agnostic and opt XPU into them. The review must focus on
what CI cannot check: whether the generalization preserves the original test
intent, whether device gating is precise, and whether every skip/xfail is
justified and traceable.

## Scope: when this skill applies

Use this skill (instead of the generic `pr-review`) when the diff is
predominantly:
- `test/**` changes that swap CUDA-hardcoded constructs for device-generic ones
- `instantiate_device_type_tests(..., allow_xpu=True)` additions
- `onlyAccelerator` / `onlyNativeDeviceTypesAnd([...])` decorator migrations
- XPU entries in OpInfo (`common_methods_invocations.py`, `opinfo/definitions/*`):
  `DecorateInfo(... device_type='xpu' ...)`, `toleranceOverride`, skips, xfails
- New `TestXxxDevice` classes split out from a device-agnostic `TestXxx`

If the PR also changes operator kernels or `native_functions.yaml`, hand those
files to the `pr-review` skill and apply this skill only to the test files.

## Usage Modes

### No Argument

If invoked with no arguments, **do not review**. Ask:

> What would you like me to review?
> - A PR number or URL (e.g., `159118` or the full PR URL)
> - A local branch

### PR Mode

```
/upstream-ut-review 159118
/upstream-ut-review https://github.com/pytorch/pytorch/pull/159118
/upstream-ut-review 159118 detailed
```

Obtain the PR title, description, diff, changed-file list, and existing review
comments before reviewing. If the command does not name a repo, default to
fetching the PR from `pytorch/pytorch`

### Local Branch Mode

```
/upstream-ut-review branch
/upstream-ut-review branch detailed
```

Review the current branch's changes relative to `main` (diff, commit log, and
changed-file list). Use the branch name in the review header instead of a PR
number.

## Review Philosophy

1. **Only report problems.** Output contains issues and actionable requests
   only. Omit empty sections. Do not praise correct choices.
2. **Preserve original test intent.** The prime directive of these PRs is "keep
   the original code styles" and coverage. A refactor that silently drops a test
   case, weakens an assertion, or changes what a test exercises is a defect even
   if CI is green.
3. **Device gating must be precise, not blanket.** A capability check must name
   the device that lacks the capability. `if device_type == "xpu": skip` is
   almost always wrong; the correct form gates on the actual condition (e.g.
   `device_type == "cuda" and not sm_is_or_higher_than(...)`).
4. **Every skip/xfail needs a reason and a tracking issue.** No bare
   `@skipXPU` / `xfailIf(TEST_XPU)` without an adjacent issue link. Prefer a
   `pytorch/pytorch` issue when the failure is in-tree.
5. **skip vs xfail vs tolerance carry different meaning.** Unsupported
   capability -> `skipIf`. Known-failing-but-should-pass -> `xfailIf`. Numeric
   drift -> `toleranceOverride`, never a skip. Getting this wrong hides
   regressions.
6. **Investigate, don't guess.** When unsure whether a generalization is
   behavior-preserving, spawn a sub-agent to read the pre-change test and the
   surrounding harness (`common_device_type.py`, `common_utils.py`).
7. **Be specific and actionable.** Reference `file:line`; name the exact
   decorator/util the author should use.

## Review Workflow

### Step 1: Understand the port

1. Read the PR title/description. These PRs follow a template ("use
   `torch.accelerator.current_accelerator()`", "enabled XPU for some test path",
   "skip cases XPU does not support"). Note the claimed method list.
2. Identify each file's role: pure device-generalization, OpInfo skip/xfail, or
   test-class split.

### Step 2: Deep Review

Go through **every changed line** against
[references/xpu-ut-review-checklist.md](references/xpu-ut-review-checklist.md).
For each generalized test, confirm the XPU path actually runs (not silently
skipped by an inherited decorator) and that the assertion is unchanged. When
unsure whether a change preserves intent, read the pre-change test and how
sibling tests in the same file already handle device gating (spawn a sub-agent
for this).

### Step 3: Verify skips and issue links

For every added skip/xfail, confirm the linked issue exists and describes the
same failure. Flag bare skips and skips that should be `xfail` or
`toleranceOverride`.

### Step 4: Cross-device blast radius

Device-generic edits and OpInfo `DecorateInfo` changes can affect CUDA, MPS,
HPU, and CPU. Confirm an `allow_xpu=True` addition or a changed `only_for` /
`device_type` predicate does not accidentally disable or re-enable another
backend. This was a real reviewer concern in the sample (an XPU-gated
`DecorateInfo` change risked affecting MPS).

### Step 5: Consolidate & fact-check

Deduplicate findings (same root cause / same fix / same `file:line` -> one
finding). Then spawn one verification sub-agent per surviving finding to
re-read the code and return valid / invalid / needs-rewording. Drop invalids.

## Output Format

**Omit sections with no problems.** Every sentence identifies a problem or
requests a change.

```markdown
## PR Review: #<number>
<!-- Or: ## Branch Review: <branch-name> (vs main) -->

### Summary
What the PR does (1 sentence), then the verdict.

### Test Intent & Coverage
[Problems only — dropped/weakened coverage, class-split regressions, missing enablement prerequisites, tests not running on the device they claim to cover]

### Device Gating
[Problems only — blanket/imprecise gating, hardcoded CUDA constructs left behind, missing allow_xpu, wrong only_for]

### Refactor Purity & Classification
[Problems only — behavior change or enablement inside a "refactor", split-class name/classification/instantiation mismatch]

### Skips / Xfails / Tolerances
[Problems only — bare or wrong-mechanism skips, missing/wrong issue links, over-broad scope]

### Cross-Device Impact
[Problems only — other backends affected by the change; unordered @parametrize; global side effects]

### Backward Compatibility
[Problems only — shared harness/util changes affecting other tests or out-of-tree backends]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification. IMPORTANT: do NOT use `#N` (e.g. #1, #2) to reference findings — GitHub auto-links these. Use descriptive references or inline the file path.]
```

Assign each finding to exactly one section, first match wins: Backward
Compatibility -> Cross-Device Impact -> Refactor Purity & Classification ->
Device Gating -> Skips/Xfails/Tolerances -> Test Intent & Coverage. State the
full consequence once.

### Specific Comments (Detailed Review Only)

Only when the user requests a "detailed" review. Group by severity:

- **🔴 Must Fix** — dropped coverage, wrong gating that hides failures, bare/incorrect skip, behavior-changing refactor
- **🟡 Should Fix** — imprecise but non-hiding gating, missing issue link, naming/convention mismatch
- **🟢 Suggestion** — style nits

For each, quote the offending line and give a concrete fix.

## Files to Reference

- [references/xpu-ut-review-checklist.md](references/xpu-ut-review-checklist.md) — the line-by-line checklist
- `torch/testing/_internal/common_device_type.py` — `instantiate_device_type_tests`, `onlyAccelerator`, `allow_xpu`, `only_for`
- `torch/testing/_internal/common_utils.py` — `TEST_XPU`, `TEST_CUDA`, `TEST_HPU`, `xfailIf`, `HardwareClassification`
- `torch/testing/_internal/common_methods_invocations.py`, `torch/testing/_internal/opinfo/definitions/*` — OpInfo `DecorateInfo`, `toleranceOverride`
