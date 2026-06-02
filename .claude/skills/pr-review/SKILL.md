---
name: pr-review
description: Review pull requests for XPU operator or backend code. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# XPU PR Review Skill

Review torch-xpu-ops pull requests focusing on what CI cannot check: correctness against CPU/CUDA semantics, XPU-specific risks (synchronization, indexing, precision), test adequacy, and backward compatibility.

Detailed reference:
- [torch-xpu-ops-review-notes.md](references/torch-xpu-ops-review-notes.md)
- [review-checklist.md](references/review-checklist.md)
- [bc-guidelines.md](references/bc-guidelines.md)

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, **do not perform a review**. Instead, ask:

> What would you like me to review?
> - A PR number or URL (e.g., `/pr-review 12345`)
> - A local branch (e.g., `/pr-review branch`)

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/intel/torch-xpu-ops/pull/12345
```

For a detailed review with line-by-line specific comments:

```
/pr-review 12345 detailed
```

Use `gh` CLI to fetch PR data:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits

# Get the diff
gh pr diff <PR_NUMBER>

# Get PR comments
gh pr view <PR_NUMBER> --json comments,reviews
```

### Local Branch Mode

Review changes in the current branch that are not in `main`:

```
/pr-review branch
/pr-review branch detailed
```

Use git commands to get branch changes:

```bash
git branch --show-current
git diff --name-only main...HEAD
git diff main...HEAD
git log main..HEAD --oneline
git diff --stat main...HEAD
```

### GitHub Actions Mode

When invoked via `@copilot /pr-review` or `@claude /pr-review` on a GitHub PR, detect this mode by the presence of `<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in the prompt.

Use git commands to get the diff and commit history (do NOT use `gh` CLI in this mode):

```bash
git diff origin/<baseBranch>...HEAD
git diff --stat origin/<baseBranch>...HEAD
git log origin/<baseBranch>..HEAD --oneline
```

## Review Philosophy

1. **Only report problems** — The review output must contain only issues, concerns, and actionable suggestions. Do NOT mention things that are done correctly. If a section has no problems, omit it entirely.
2. **Investigate, don't guess** — When uncertain whether a checklist item applies, spawn a sub-agent to read the relevant code. A reviewer who guesses wrong provides negative value.
3. **Review the design, not just the implementation** — Question whether the change belongs in this repository and layer at all (vs upstream PyTorch XPU, oneDNN XPU, or oneMKL).
4. **Focus on what CI cannot check** — Don't comment on formatting, linting, or CI failures. Focus on semantic correctness, XPU-specific risks, BC implications, and test adequacy.
5. **Everything is a must-fix** — There are no "nits." If it's worth mentioning, it's worth fixing.
6. **Be specific and actionable** — Reference file paths and line numbers. Name the function/class/file the author should look at.
7. **Match the immediate context** — Read how similar features are already implemented in the same file/module.
8. **Verify CPU/CUDA parity from source** — Do not infer behavior from memory. Inspect the actual upstream implementation from `pytorch/pytorch`.

### Using sub-agents

The review checklist is large. **Spawn sub-agents** to investigate whether checklist items apply: read surrounding code, check upstream PyTorch implementation for parity, or verify tests exist. Spawn them in parallel for independent areas.

## Review Workflow

### Step 1: Understand Context

Before reviewing, build understanding:
1. What problem does this PR solve?
2. Which files/modules are changed?
3. Is this a functional fix, performance optimization, refactor, or test-only change?
4. Is the change in the right layer (kernel, dispatch, test, build, or fallback)?
5. Spawn sub-agents to read the unchanged code surrounding each significantly changed file

### Step 2: Deep Review

Go through **every changed line** in the diff and evaluate against the review checklist in [review-checklist.md](references/review-checklist.md).

Pay special attention to XPU-specific risks:
- **Synchronization**: hidden host sync, unnecessary synchronize, stream misuse
- **Indexing**: 32-bit vs 64-bit indexing, large tensor overflow risk
- **Layout**: contiguous vs non-contiguous, channels_last handling
- **Precision**: FP32 / BF16 / FP16 behavior, accumulation dtype, AMP impact
- **Kernel efficiency**: branch divergence, work-group choice, unnecessary copies
- **Fallback/dispatch**: wrong registration, silent fallback, inconsistent path coverage

### Step 3: Check Backward Compatibility

Evaluate BC implications per [bc-guidelines.md](references/bc-guidelines.md). For non-trivial BC questions, spawn a sub-agent to search for existing callers of the modified API.

### Step 4: Formulate Review

Structure your review with actionable feedback organized by category. Every finding should be traceable to a specific line in the diff.

### Step 5: Fact-Check

After drafting the review, spawn a sub-agent per reported issue (in parallel) to independently verify the claim by re-reading the relevant code. Drop invalid issues, reword uncertain ones with a note about confidence level.

## Output Format

**Omit sections where you have no problems to report.** Every sentence must identify a problem or request a change.

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
What the PR does (1 sentence), then the overall verdict.

### Correctness
[Problems only — semantic parity, edge cases, behavioral issues]

### XPU-Specific Risks
[Problems only — synchronization, indexing, precision, kernel issues]

### Dispatch & Registration
[Problems only — yaml wiring, fallback, backend path]

### Testing
[Problems only — missing tests, wrong patterns, inadequate coverage]

### Backward Compatibility
[Problems only]

### Performance
[Problems only]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Missing tests (new functionality without tests, bug fixes without regression tests) always means **Request Changes**.

[Brief justification — focus on what blocks approval]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

```markdown
### Specific Comments
- `src/ATen/native/xpu/sycl/MyKernel.cpp:42` - Index calculation looks 32-bit; large tensor overflow should be checked.
- `test/xpu/test_ops.py:100-105` - Missing test for non-contiguous input case.
```

## Review Checklist (Quick Reference)

- [ ] Change belongs in `torch-xpu-ops` (not upstream PyTorch XPU, oneDNN XPU, or oneMKL)
- [ ] Semantics match CPU/CUDA behavior (verified from source)
- [ ] No hidden synchronization or race-risk
- [ ] 64-bit indexing considered where needed
- [ ] Layout handling is correct (non-contiguous, channels_last)
- [ ] BF16/FP16/AMP path is safe
- [ ] Dispatch / fallback logic is correct
- [ ] Tests are sufficient (dtype, layout, shape, API variants)
- [ ] Performance claim is supported with evidence
- [ ] No backward compatibility issues
- [ ] PR scope is reviewable; if >350 changed lines, call out the size risk

## Files to Reference

When reviewing, consult these for context:
- `yaml/xpu_functions.yaml` — XPU operator declarations
- `yaml/native/native_functions.yaml` — Native function schemas
- `src/ATen/native/xpu/` — XPU operator implementations
- `src/ATen/native/xpu/sycl/` — SYCL kernel implementations
- `src/ATen/native/xpu/XPUFallback.template` — Fallback logic
- `test/xpu/` — XPU-specific tests
- `test/test_ops_xpu.py` — OpInfo-based XPU operator tests

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: pr-review.

Otherwise, keep the reply in the requested review format and do not force an extra trailing sentence.
