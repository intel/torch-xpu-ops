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
- [pr-submission-guidelines.md](references/pr-submission-guidelines.md) — for PR authors (process & etiquette)

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
# Get current branch name
git branch --show-current

# Get list of changed files compared to main
git diff --name-only main...HEAD

# Get full diff compared to main
git diff main...HEAD

# Get commit log for the branch
git log main..HEAD --oneline

# Get diff stats (files changed, insertions, deletions)
git diff --stat main...HEAD
```

For local branch reviews:
- The "Summary" should describe what the branch changes accomplish based on commit messages and diff
- Use the current branch name in the review header instead of a PR number
- All other review criteria apply the same as PR reviews

### GitHub Actions Mode

When invoked via `@copilot /pr-review` or `@claude /pr-review` on a GitHub PR, detect this mode by the presence of `<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in the prompt.

The prompt already contains:
- PR metadata (title, author, branch names, additions/deletions, file count)
- PR body/description
- All comments and review comments (with file/line references)
- List of changed files with paths and change types

Use git commands to get the diff and commit history. The base branch name is in the
prompt context (look for `PR Branch: <head> -> <base>` or the `baseBranch` field).

```bash
# Get the full diff against the base branch
git diff origin/<baseBranch>...HEAD

# Get diff stats
git diff --stat origin/<baseBranch>...HEAD

# Get commit history for this PR
git log origin/<baseBranch>..HEAD --oneline

# If the base branch ref is not available, fetch it first
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI commands in this mode -- only git commands are available.
All PR metadata, comments, and reviews are already in the prompt context;
only the diff and commit log need to be fetched via git.

## Review Philosophy

1. **Only report problems** — The review output must contain only issues, concerns, and actionable suggestions. Do NOT mention things that are done correctly, do NOT praise good decisions, do NOT explain why something is fine. If a section has no problems, omit it entirely. The reader's time is precious — every sentence must point to something that needs fixing or further discussion.
2. **Investigate, don't guess** — When uncertain whether a checklist item applies, spawn a sub-agent to read the relevant code. A reviewer who guesses wrong provides negative value.
3. **Review the design, not just the implementation** — A PR can have perfectly correct implementation of a bad design. Question side-channel communication, on/off private flags, and demand concrete interface documentation for new contracts between components.
4. **Focus on what CI cannot check** — Don't comment on formatting, linting, type errors, or CI failures. Focus on design quality, interface correctness, thread safety, BC implications, test adequacy, and pattern adherence.
5. **Everything is a must-fix** — There are no "nits." If it's worth mentioning, it's worth fixing. Every inconsistency degrades the codebase over time.
6. **Be specific and actionable** — Reference file paths and line numbers. Name the function/class/file the author should use.
7. **Match the immediate context** — Read how similar features are already implemented in the same file. Pattern mismatches within a file are always wrong.
8. **Assume competence** — The author knows PyTorch; explain only non-obvious context.
9. **No repetition** — Each observation appears in exactly one section of the review output.
10. **Verify CPU/CUDA parity from source** — Do not infer behavior from memory. Inspect the actual upstream implementation from `pytorch/pytorch`.

### Using sub-agents

The review checklist is large. **Spawn sub-agents** to investigate whether checklist items apply: read surrounding code, check upstream PyTorch implementation for parity, or verify tests exist. Spawn them in parallel for independent areas.

## Review Workflow

### Step 1: Understand Context

Before reviewing, build understanding of what the PR touches and why:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)
4. Spawn sub-agents to read the unchanged code surrounding each significantly changed file to understand existing patterns and infrastructure

### Step 2: Verify Upstream Semantics

For every changed kernel or operator file, fetch and read the corresponding upstream PyTorch implementation BEFORE evaluating correctness:

- `src/ATen/native/xpu/<Op>.cpp` → read `aten/src/ATen/native/<Op>.cpp`
- `src/ATen/native/xpu/sycl/<Op>Kernels.cpp` → read `aten/src/ATen/native/cuda/<Op>.cu`
- Shared math utilities (e.g., `MathExtensions.h`) → read `aten/src/ATen/native/Math.h`

Use `gh api` or spawn a sub-agent to fetch the upstream file content. Do NOT proceed to the deep review until upstream code has been read. Quote or summarize relevant upstream patterns in your working notes before continuing.

### Step 3: Deep Review

Go through **every changed line** in the diff and evaluate against the review checklist in [review-checklist.md](references/review-checklist.md).

If the diff adds or modifies any agent instruction files (`SKILL.md`, `AGENTS.md`, `claude.md`, `copilot-instructions.md`, or files under `.claude/` / `.github/`), load the skill-writer skill and evaluate those changes against its guidelines. Do NOT skip this — treat it as a blocking gate for those files.

Pay special attention to XPU-specific risks:
- **Synchronization**: hidden host sync, unnecessary synchronize, stream misuse
- **Indexing**: 32-bit vs 64-bit indexing, large tensor overflow risk
- **Layout**: contiguous vs non-contiguous, channels_last handling
- **Precision**: FP32 / BF16 / FP16 behavior, accumulation dtype, AMP impact
- **Kernel efficiency**: branch divergence, work-group choice, unnecessary copies
- **Fallback/dispatch**: wrong registration, silent fallback, inconsistent path coverage

### Step 4: Check Backward Compatibility

Evaluate BC implications per [bc-guidelines.md](references/bc-guidelines.md). For non-trivial BC questions, spawn a sub-agent to search for existing callers of the modified API.

### Step 5: Formulate Review

Structure your review with actionable feedback organized by category. Every finding should be traceable to a specific line in the diff.

### Step 6: Fact-Check

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

[Brief justification — focus on what blocks approval. IMPORTANT: Do NOT use `#N` (e.g., #1, #2, #3) to reference findings — GitHub auto-links these to real issues/PRs. Instead use descriptive references like "the step numbering issue", "the stale path in auto-labeling", or inline the file path.]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

When performing a detailed review, group findings by severity:
- **🔴 Must Fix** — Incorrect terminology, bugs, magic numbers in logic, correctness issues
- **🟡 Should Fix** — Naming inconsistency, missing comments on non-obvious logic
- **🟢 Suggestion** — Style nits, minor improvements

For each finding, quote the offending line and provide a concrete fix.

```markdown
### 🔴 Must Fix (N issues)

**[Category] file.cpp:42** — <description>
  <quoted code>
  → <suggested fix>

### 🟡 Should Fix (N issues)
...

### 🟢 Suggestions (N issues)
...

### ✅ What looks good
<briefly note well-written parts — good reviews are balanced>
```

If there are zero issues in a severity level, omit that section. Always include the "What looks good" section in detailed reviews.

## Intel GPU Terminology

**Principle: For SYCL programming, always use SYCL programming model terms. Hardware architecture terms should only appear in comments explaining the motivation for a particular optimization.**

This is a **🔴 Must Fix** category.

### SYCL Programming Model Terms

Use these terms in all code, variable names, and function names:

| ❌ Deprecated Term | ✅ Current Term | Notes |
|---|---|---|
| SIMD width / SIMD length / simd_width | **subgroup size** | SYCL/oneAPI standard term |
| SIMD lane | **work-item (within subgroup)** | Aligns with SYCL spec |
| SIMD-8 / SIMD-16 / SIMD-32 | **subgroup size 8 / 16 / 32** | Use numeric subgroup size |
| thread block / threadblock | **work-group** | SYCL/oneAPI standard term |

### Hardware Architecture Terms (comments/documentation only)

Use these **only in comments** to explain why a particular optimization choice was made (e.g., occupancy, register pressure, memory alignment). They must NOT appear in variable names, function names, or general code.

| ❌ Deprecated Term | ✅ Current Intel Term | Generic Term | Abbreviation |
|---|---|---|---|
| Execution Unit (EU) | **Xe Vector Engine** | Vector Engine | **XVE** |
| Systolic / "DPAS part of EU" | **Xe Matrix eXtension** | Matrix Engine | **XMX** |
| Subslice (SS) / Dual Subslice (DSS) | **Xe-core** | — | **XC** |
| HW thread | **XVE thread** | — | Each XVE thread executes a subgroup (subgroup size 16 or 32) |
| SIMD-16 / SIMD-32 (hardware context) | **XVE thread width** | — | The number of data elements processed per thread; maps to subgroup size |
| GRF file / GRF count | **register file / register count** | — | Use architecture-neutral where possible |
| SLM (ambiguous) | **SLM (Shared Local Memory)** | — | Spell out on first use |

### Where to check:
- Variable names: `subslice_count` → `xc_count` or `xecore_count`
- Function names: `getSimdWidth()` → `getSubgroupSize()`
- Comments and docstrings
- Log/error messages
- Test names and descriptions

### Examples

```
// 🔴 Bad
int num_subslices = device.get_info<ext::info::device::gpu_subslices_per_slice>();
int simd_len = 16;
// Each EU has 8 HW threads

// ✅ Good
int num_xecores = device.get_info<ext::info::device::gpu_subslices_per_slice>();
// Note: API name still uses "subslices" — wrap in a helper if possible
int subgroup_size = 16;
// Each XVE supports 8 concurrent threads
```

**Edge case — API boundaries:** When the underlying API (Level Zero, OpenCL, SYCL extensions) still uses old terms in function/enum names, it's acceptable to use them **at the call site only**. Wrap in a helper with modern naming, and add a comment:

```
// ✅ OK — API uses legacy name, but our abstraction uses modern term
// Level Zero API still exposes "subslice" in its info query
int xecore_count = zeDeviceGetSubsliceCount(device);  // legacy API name
```

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
