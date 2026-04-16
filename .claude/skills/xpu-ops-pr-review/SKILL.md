---
name: xpu-ops-pr-review
description: "Review intel/torch-xpu-ops pull requests and draft maintainer-style PR replies. Use when the user asks to review a PR, branch, commit, or diff in torch-xpu-ops, check whether a review concern is valid, reply to a reviewer, write file-scoped findings, or prepare a repo-ready AI-assisted review comment."
---

# XPU Ops PR Review

This Skill reviews `torch-xpu-ops` and closely related PyTorch XPU backend changes. It is designed for repository use with GitHub Copilot and should produce concise, evidence-based review output rather than generic commentary.

This Skill is for analysis and draft generation only. It does not monitor GitHub, post comments automatically, manage state files, or use credentials.

## When to Use This Skill

Use this Skill when:

- Reviewing a `torch-xpu-ops` pull request, branch, commit, or diff
- Checking whether a reviewer, Copilot, or bot concern is valid
- Drafting a reply to a PR review comment or inline thread
- Writing file-scoped findings with concrete evidence
- Preparing a repo-ready AI-assisted review comment in English

Do not use this Skill when:

- The task is issue filing or issue-template formatting
- The task is general XPU debugging without a concrete review context
- The task is automatic GitHub posting or workflow automation

## Quick Start

Typical prompts that should trigger this Skill:

- "Review this torch-xpu-ops PR"
- "Check whether this review comment is valid"
- "Draft a reply to this inline reviewer concern"
- "Review this branch against main"
- "Summarize the blocking findings in this PR"

## Load First

- The PR description, commit summary, or triggering review request
- The changed diff hunk, repo-relative file path, and changed-side line number when available
- The original review comment or thread body when replying to a specific concern
- Nearby implementation context or full file content when the claim depends on surrounding logic
- Matching CUDA, CPU, or upstream PyTorch code when parity or expected semantics is discussed
- references/torch-xpu-ops-review-notes.md
- references/pytorch-pr-review-skill.md
- references/review-checklist.md
- references/bc-guidelines.md

## If Context Is Missing

If the user asks for a review but does not provide enough context:

- Ask what should be reviewed: PR, branch, commit, diff, or review thread
- Ask for the exact file and line when the task is a reply to one comment
- Ask for the relevant hunk if the concern depends on changed lines
- Ask for the upstream counterpart when the question is about CUDA or CPU parity

If evidence is still incomplete after inspection, say `not verified` rather than making a definitive public claim.

## Review Modes

- Default mode is detailed PR review mode.
- Detailed PR review mode means the final answer is file-scoped, evidence-based, and anchored to the smallest verified line, hunk, function, registration block, or test block.
- Thread reply mode is for answering one existing PR comment or inline review thread with a concise maintainer-style reply.
- Quick triage mode is for identifying whether a concern looks valid before drafting a full reply.
- Only collapse to a brief summary mode when the human explicitly asks for a quick, concise, or high-level review.

## Review Philosophy

- Investigate, do not guess. If a checklist item is uncertain, inspect the nearby code, declarations, registrations, tests, or callers before making the claim.
- Focus on what CI may not catch: semantic regressions, dispatch and registration mistakes, missing guards, missing tests for logic changes, BC-sensitive behavior, and XPU-specific parity gaps.
- Review the design, not just the local edit. A change can be mechanically correct but still introduce a bad contract or inconsistent backend behavior.
- Match the immediate local context. Pattern mismatches inside the same subsystem are suspicious until proven otherwise.
- Be specific and actionable. Every finding or reply should point to a concrete location and state what to verify, restore, or update.
- No repetition. Each observation should appear once in the most appropriate section.
- Treat user-visible behavior changes as BC-sensitive until verified otherwise.
- Prefer the strongest reviewer-actionable findings over generic prose.

## Review Inputs To Inspect

For full PR reviews, inspect as many of these as are relevant:

- PR title, body, and commit summary
- Changed files and diff hunks
- Existing review comments and unresolved threads
- Registration files such as `yaml/xpu_functions.yaml` and `yaml/native/native_functions.yaml`
- Kernel sources under `src/ATen/native/xpu/` and `src/ATen/native/xpu/sycl/`
- Relevant tests under `test/xpu/` and operator-level coverage such as `test/test_ops_xpu.py`
- CI or build files when the change touches infrastructure

## Review Workflow

1. Identify whether the request is a full PR review or a narrow thread reply.
2. Understand the scope of the change from the diff, title, PR body, and review thread.
3. Inspect every materially changed hunk relevant to the request.
4. Check nearby unchanged code, yaml, registrations, tests, or callers when needed to validate the claim.
5. If parity is discussed, inspect the exact CUDA, CPU, or upstream counterpart before accepting or rejecting the concern.
6. Choose the correct severity or verdict based on verified evidence.
7. Produce either a structured PR review or a concise thread reply.

## XPU Review Focus Areas

- Operator wiring: `yaml/xpu_functions.yaml`, `yaml/native/native_functions.yaml`, generated registrations, and dispatch key coverage move together.
- Test quality: logic changes should extend existing XPU or operator coverage instead of adding brittle one-off checks.
- Semantic parity: output shape, dtype promotion, empty and zero-dim behavior, non-contiguous inputs, inf and nan handling, and backward behavior should match the intended PyTorch contract.
- Kernel safety: watch for raw pointer misuse, writable access where read-only is expected, address-space mistakes, missing synchronization assumptions, or silent fallback behavior.
- BC impact: exceptions, defaults, return behavior, and public semantics should be treated as potentially breaking until verified.
- CI-blind risk: generated-code paths, Windows differences, compile-time wiring, and untested build config changes deserve explicit scrutiny.

## Evidence Policy

- Every finding must include direct evidence from the diff or closely related inspected code.
- Evidence should be checkable: repo-relative path plus exact line, hunk, registration block, test block, or concrete code fact.
- Separate evidence from inference. If something is only a hypothesis, label it as `not verified`.
- Do not claim CUDA or CPU parity mismatches unless you inspected the exact counterpart.
- Do not claim a missing registration, build break, or semantic regression unless the supporting code was actually checked.
- If a point cannot be anchored to concrete evidence, keep it out of the main finding list and move it to residual risk or omit it.

## Review Priorities

Prioritize findings in this order:

1. Correctness, safety, and semantic regressions
2. Dispatch, registration, backward, or generated-code mistakes
3. Missing or weak tests for logic changes
4. BC-sensitive user-visible behavior changes
5. Residual risks in CI, build, or platform coverage

Do not pad the review with style-only comments, lint-like remarks, or repeated observations.

## Thread Reply Verdict Rules

| Verdict | Use when |
|---|---|
| **Bug** or **Issue** | The code has a real correctness, safety, BC, or infrastructure problem that should be fixed |
| **Not valid** | The concern is wrong once checked against the actual code or upstream reference |
| **Technically correct but not practically significant** | The concern is true in theory but does not change observable behavior or reviewer action |
| **done** | The PR already contains the fix |
| **Agreed** | You independently verified that a human reviewer's concern is correct |

If the evidence is not strong enough for one of these verdicts, do not force a public verdict. Say it is not verified and ask for the missing context.

Use verdict labels only for thread replies. For full PR reviews, use the structured review format with `Status:` and file-scoped findings.

## Bot And Human Comment Rules

Treat these as automated comments and do not agree with or duplicate them:

- Any self-authored automated comment ending with `*[AI-assisted reply]*`
- Any self-authored automated comment containing `Requested in [this mention]`
- `copilot[bot]`
- `github-actions[bot]`

`Agreed` is only valid for a human reviewer comment after independent verification.

If the same file and line already have an equivalent self-authored or bot-authored automated reply, skip the duplicate unless you have materially new evidence.

## Repository-Specific Heuristics

These are common torch-xpu-ops patterns. Verify them against the actual code before using them in a review or reply.

| Concern family | Typical outcome once verified |
|---|---|
| CUDA and XPU parity complaint, but CUDA upstream behaves the same way | **Not valid** |
| Dtype dispatch described as too narrow, but CUDA upstream uses the same `AT_DISPATCH_*` macro | **Not valid** |
| Batched sparse CSR or dim > 2 complaint, but neither CUDA nor CPU supports that path | **Not valid** |
| Explicit `.wait()` concern on an in-order SYCL queue with no behavior change | **Technically correct but not practically significant** |
| Typo, unused variable, or small cleanup already fixed in the PR | **done** |

## Review Style

- English only for public draft output
- Write like a concise human reviewer, not a checklist narrator
- Prefer 2-5 strong findings over long, repetitive coverage
- Present findings first when doing a full review
- Start thread replies with a bold verdict
- Cite exact repo-relative paths and upstream locations when possible
- No thanks, apologies, motivational filler, or generic praise
- Explain the reason, not just the conclusion
- If there are no blocking findings, say so explicitly and mention any residual risk or testing gap that was not verified
- If the user wants repo-ready AI-assisted text, end thread replies with a blank line plus `*[AI-assisted reply]*`

## Output Modes

### 1. Detailed PR Review

Use this when the user asks to review a PR, branch, or diff. The output should be file-scoped and evidence-based.

Default format:

```markdown
**Status: [Approve | Request Changes | Comment]**

### Summary
[1-2 sentences on the overall XPU review conclusion]

### File Comments

#### repo/relative/path.ext

**High: [Short title]**
- **Line / Scope**: L123
- **Problem**: [what is wrong]
- **Evidence**: [repo-relative path/reference + concrete code fact]
- **Why it matters**: [crash / regression / linker error / undefined behavior / missing coverage]
- **Suggestion**: [specific action]

### Cross-Cutting / Residual Risk
[test gaps, blast radius, CI dimensions not verified, or follow-up work]
```

Rules:

- Start directly with `**Status:`.
- Findings come before residual-risk discussion.
- Use repo-relative paths only.
- Do not invent line numbers.
- Keep the strongest 2-5 findings unless the user explicitly asks for exhaustive coverage.
- If a point cannot be anchored to verified evidence, move it to residual risk or omit it.

### 2. Thread Reply Draft

Use this when the user wants a single reply to a PR comment or inline review thread. Return only the comment text unless the user explicitly asks for extra notes.

Default public reply shape:

```markdown
**<Verdict>** - <One sentence with the core conclusion and the key evidence.>

<One or two short follow-up sentences with the exact reason, parity citation, or concrete fix.>

*[AI-assisted reply]*
```

If the user does not want AI-assisted footer text, omit only the last line.

### 3. Reply Draft Plus Internal Rationale

Use this when the user asks whether a concern is valid and also wants a short explanation. Keep the public reply and private rationale clearly separated.

### 4. Multi-Reply Review Pass

Use this when the user wants help answering several review comments in one PR. Group drafts by file or thread, but keep each individual reply concise.

## Common Pitfalls

- Treating parity as a fact without checking the CUDA or CPU counterpart
- Claiming a registration problem without inspecting yaml or generated wiring
- Writing a full review when the user only asked for a narrow thread reply
- Repeating the same finding in multiple sections
- Using local absolute paths in public review text
- Agreeing with self-authored automated comments

## Do Not Do These

- Do not include tokens, monitoring instructions, cron details, or posting scripts.
- Do not invent line numbers, upstream parity claims, or test evidence.
- Do not agree with your own bot output.
- Do not write a full PR review when the user asked for a narrow thread reply.
- Do not use local paths like `/home/...` in public draft output.

## Reference Files

- references/torch-xpu-ops-review-notes.md
- references/pytorch-pr-review-skill.md
- references/review-checklist.md
- references/bc-guidelines.md