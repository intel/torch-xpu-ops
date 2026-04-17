---
name: xpu-ops-pr-review
description: "Review intel/torch-xpu-ops pull requests, diffs, commits, and review threads with XPU-specific checks for backend ownership, SYCL and library boundaries, async stream semantics, layout handling, BF16/FP16 numerics, 64-bit indexing, dispatch wiring, and test gaps. Use when the user asks to review a torch-xpu-ops PR, validate a review concern, or draft a maintainer-style reply."
---

# XPU Ops PR Review

This Skill reviews `intel/torch-xpu-ops` and closely related PyTorch XPU backend changes with a repository-specific lens.

Do not review these changes like generic operator patches. Many operators in this repository are implemented as SYCL kernels, some linear algebra paths route through oneMKL, and some conv or gemm critical paths belong in upstream PyTorch oneDNN XPU code instead of `torch-xpu-ops`. The first review question is whether the change lives in the correct repository, layer, and backend path.

This Skill is for analysis and draft generation only. It does not post to GitHub, manage credentials, or automate workflows.

## When to Use This Skill

Use this Skill when:

- Reviewing a `torch-xpu-ops` pull request, branch, commit, or diff
- Checking whether a reviewer, Copilot, or bot concern is valid
- Drafting a reply to an inline review thread or PR comment
- Writing file-scoped findings with concrete evidence
- Evaluating XPU-specific correctness, performance, or regression risk

Do not use this Skill when:

- The task is issue filing or issue-template formatting
- The task is general XPU debugging without a concrete review target
- The task is automatic GitHub posting or workflow automation

## Quick Start

Typical prompts that should trigger this Skill:

- "Review this torch-xpu-ops PR"
- "Check whether this review comment is valid"
- "Draft a reply to this inline reviewer concern"
- "Review this branch against main"
- "Summarize the blocking findings in this diff"

## Load First

- The PR description, commit summary, or triggering review request
- The changed diff hunk, repo-relative file path, and changed-side line number when available
- The original review comment or thread body when replying to a specific concern
- Nearby implementation context or full file content when the claim depends on surrounding logic
- The matching CUDA, CPU, or upstream PyTorch code when parity or expected semantics is discussed
- `references/torch-xpu-ops-review-notes.md`
- `references/review-checklist.md`
- `references/bc-guidelines.md`

## If Context Is Missing

If the user asks for a review but does not provide enough context:

- Ask what should be reviewed: PR, branch, commit, diff, or review thread
- Ask for the exact file and line when the task is a reply to one comment
- Ask for the relevant hunk if the concern depends on changed lines
- Ask for the upstream counterpart when the question is about CUDA or CPU parity

If evidence is still incomplete after inspection, say `not verified` rather than making a definitive public claim.

## Review Modes

- Default mode is detailed PR review mode
- Detailed PR review mode means the final answer is file-scoped, evidence-based, and anchored to the smallest verified line, hunk, function, registration block, or test block
- Thread reply mode is for answering one existing PR comment or inline review thread with a concise maintainer-style reply
- Quick triage mode is for identifying whether a concern looks valid before drafting a full reply
- Only collapse to a brief summary mode when the human explicitly asks for a quick, concise, or high-level review

## Review Philosophy

- Investigate, do not guess. If a checklist item is uncertain, inspect the nearby code, declarations, registrations, tests, or callers before making the claim.
- Check repository and backend ownership first. A mechanically correct implementation in the wrong layer is still a review problem.
- Review the design, not just the local edit. A change can be locally correct but still create a backend contract mismatch, silent fallback, or parity drift.
- Focus on what CI may not catch: semantic regressions, stream hazards, hidden synchronization, layout regressions, dtype or accumulation mistakes, indexing overflow, and incomplete test coverage.
- Treat performance claims as unverified until the diff or PR includes benchmark evidence or a strong design rationale.
- Be specific and actionable. Every finding or reply should point to a concrete location and state what to verify, restore, or extend.
- No repetition. Each observation should appear once in the most appropriate section.
- Treat user-visible behavior changes as backward-compatibility-sensitive until verified otherwise.

## Eight Critical XPU Review Areas

### 1. Backend Ownership And Repository Placement

- Verify whether the change belongs in `torch-xpu-ops`, upstream PyTorch XPU code, oneDNN XPU, or oneMKL
- Check whether the PR is re-implementing an existing backend or library path that should be reused instead
- Ask whether the operator wiring, kernel location, and backend dispatch all point to the same design intent

### 2. Semantic Parity With CPU Or CUDA

- Check input validation, error behavior, dtype promotion, broadcasting, stride handling, and output semantics
- Verify `functional`, `out=`, `inplace`, view, backward, and deterministic behavior when relevant
- Inspect empty tensors, zero-size dimensions, non-contiguous inputs, scalar tensors, and channels-last behavior when the operator can see them

### 3. Async Execution And Synchronization

- Look for hidden host synchronization, blocking APIs, `.item()`, debug reads, or host-driven data dependence
- Check whether cross-stream ordering or event usage is correct when results are consumed on another stream
- Treat newly added broad `synchronize()` calls as suspicious until they are shown to be required rather than hiding a race

### 4. Layout, Memory Format, And Allocation Behavior

- Verify whether non-contiguous and channels-last inputs are truly supported or silently converted through `.contiguous()`
- Check whether the change introduces extra copies, format conversions, or oversized temporary buffers
- Confirm that output allocation and temporary tensors preserve intended memory format when appropriate

### 5. Dtype, Precision, And Numerical Stability

- Check input dtype, compute dtype, and accumulation dtype separately
- Pay extra attention to BF16, FP16, autocast, reductions, norms, softmax-style kernels, and atomic accumulation patterns
- Do not accept a performance optimization that silently reduces accumulation precision without explicit justification and tests

### 6. 64-Bit Indexing And Large Tensor Safety

- Review index, offset, stride, and `numel` math for 32-bit overflow risk
- Check pointer arithmetic, flattening logic, and address calculations in loops or helpers
- Large-tensor correctness is a hard review gate even if small tests pass

### 7. SYCL Kernel Mapping And Intel XPU Performance Model

- Check work-group sizing, subgroup assumptions, branch structure, register pressure risks, and memory-access patterns
- Flag repeated queue, context, descriptor, or expensive host-side setup in hot paths
- Be cautious of generic code paths that make the main performance path slower just to handle rare edge cases

### 8. Dispatch, Fallback, Registration, And Tests

- Verify yaml wiring, native implementation, backward path, fallback behavior, and generated-code expectations together
- Check that unsupported cases fail explicitly or fall back intentionally rather than silently taking the wrong path
- Require tests designed around XPU risk dimensions, not just one happy-path assertion

## Review Workflow

1. Identify whether the request is a full PR review, a branch review, a diff review, or a narrow thread reply.
2. Determine the change scope and backend boundary before judging the local implementation.
3. Inspect every materially changed hunk relevant to the request.
4. Check nearby unchanged code, yaml, registrations, tests, or callers when needed to validate the claim.
5. If parity is discussed, inspect the exact CUDA, CPU, or upstream counterpart before accepting or rejecting the concern.
6. Apply the XPU-specific checks in `references/torch-xpu-ops-review-notes.md` and the concise checklist in `references/review-checklist.md`.
7. Evaluate backward-compatibility sensitivity using `references/bc-guidelines.md` whenever user-visible behavior may change.
8. Produce either a structured PR review or a concise thread reply.

## Evidence Policy

- Every finding must include direct evidence from the diff or closely related inspected code
- Evidence should be checkable: repo-relative path plus exact line, hunk, registration block, test block, or concrete code fact
- Separate evidence from inference. If something is only a hypothesis, label it as `not verified`.
- Do not claim CUDA or CPU parity mismatches unless you inspected the exact counterpart.
- Do not claim a missing registration, build break, or semantic regression unless the supporting code was actually checked.
- If a point cannot be anchored to concrete evidence, keep it out of the main finding list and move it to residual risk or omit it.

## Review Priorities

Prioritize findings in this order:

1. Correctness, safety, and semantic regressions
2. Async stream hazards, hidden synchronization, and large-tensor overflow risk
3. Dispatch, registration, backward, or generated-code mistakes
4. Missing or weak tests for logic changes
5. Backward-compatibility-sensitive user-visible behavior changes
6. Performance regressions or unproven optimization claims

Do not pad the review with style-only comments, lint-like remarks, or repeated observations.

## Time-Limited Triage

If time is limited, check these three areas first:

1. Hidden synchronization or stream-dependency bugs
2. 64-bit indexing and large-tensor overflow risk
3. Channels-last plus BF16 or FP16 correctness and performance path integrity

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
- **Why it matters**: [crash / regression / undefined behavior / missing coverage / hidden sync / parity drift]
- **Suggestion**: [specific action]

### Cross-Cutting / Residual Risk
[test gaps, blast radius, CI dimensions not verified, or follow-up work]
```

Rules:

- Start directly with `**Status:`
- Findings come before residual-risk discussion
- Use repo-relative paths only
- Do not invent line numbers
- Keep the strongest 2-5 findings unless the user explicitly asks for exhaustive coverage
- If a point cannot be anchored to verified evidence, move it to residual risk or omit it

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
- Missing the repository-boundary question and reviewing only the local kernel body
- Ignoring hidden synchronization, channels-last fallbacks, or 64-bit indexing risks because the small tests pass
- Using local absolute paths in public review text
- Agreeing with self-authored automated comments

## Do Not Do These

- Do not include tokens, monitoring instructions, cron details, or posting scripts
- Do not invent line numbers, upstream parity claims, or test evidence
- Do not agree with your own bot output
- Do not write a full PR review when the user asked for a narrow thread reply
- Do not use local paths like `/home/...` in public draft output

## Reference Files

- `references/torch-xpu-ops-review-notes.md`
- `references/review-checklist.md`
- `references/bc-guidelines.md`
