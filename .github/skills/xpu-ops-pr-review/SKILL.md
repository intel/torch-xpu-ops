---
name: xpu-ops-pr-review
description: Review pull requests for XPU operator or backend code. Use when reviewing PRs in xpu ops, torch-xpu-ops, SYCL kernels, backend dispatch, performance optimization, or tests for Intel GPU / XPU related changes.
---

# XPU PR Reviewer

This Skill helps review PRs for XPU-related code with focus on correctness, performance, maintainability, and XPU-specific risks.

Detailed reference:
- [torch-xpu-ops-review-notes.md](references/torch-xpu-ops-review-notes.md)
- [review-checklist.md](references/review-checklist.md)
- [bc-guidelines.md](references/bc-guidelines.md)

## When to use this Skill

Use this Skill when:
- Reviewing PRs in xpu ops / torch-xpu-ops
- Reviewing SYCL kernel changes
- Checking backend dispatch / registration / fallback logic
- Reviewing XPU performance optimizations
- Reviewing XPU operator tests

## Instructions

### 1. Understand the PR
First identify:
- What problem does this PR solve?
- Which files/modules are changed?
- Is this a functional fix, performance optimization, refactor, or test-only change?
- Is the change in the right layer (kernel, dispatch, test, build, or fallback)?

Before reviewing, load and use these references:
- [torch-xpu-ops-review-notes.md](references/torch-xpu-ops-review-notes.md)
- [review-checklist.md](references/review-checklist.md)
- [bc-guidelines.md](references/bc-guidelines.md)

### 2. Review correctness
Check whether:
- Behavior matches CPU/CUDA semantics
- If CPU/CUDA parity matters, inspect the actual upstream implementation from a local `pytorch/pytorch` checkout; if it is not available locally, fetch or clone it before concluding
- Do not write a CPU/CUDA parity conclusion from model memory alone; use checked source from `pytorch/pytorch`
- Edge cases are covered: empty tensor, non-contiguous, broadcast, scalar, large shape
- out / inplace / backward behavior is correct
- Error handling and unsupported cases are explicit

### 3. Review XPU-specific risks
Pay special attention to:
- **Synchronization**: hidden host sync, unnecessary synchronize, stream misuse
- **Indexing**: 32-bit vs 64-bit indexing, large tensor overflow risk
- **Layout**: contiguous vs non-contiguous, channels_last handling
- **Precision**: FP32 / BF16 / FP16 behavior, accumulation dtype, AMP impact
- **Kernel efficiency**: branch divergence, work-group choice, unnecessary copies, temp buffers
- **Fallback/dispatch**: wrong registration, silent fallback, inconsistent path coverage

### 4. Review tests
Check whether tests cover:
- Correctness for normal and edge cases
- Multiple dtypes if relevant
- Layout variations if relevant
- Large tensor / indexing cases if relevant
- Performance evidence if PR claims optimization

### 5. Give review output
Structure feedback as:
1. **Summary**: what the PR changes
2. **Strengths**: what looks good
3. **Risks / Issues**: correctness, XPU-specific, test gaps
4. **Required changes**: must-fix items
5. **Optional suggestions**: nice-to-have improvements

## Review checklist

- [ ] Semantics match expected PyTorch behavior
- [ ] No hidden synchronization or race-risk
- [ ] 64-bit indexing considered where needed
- [ ] Layout handling is correct
- [ ] BF16/FP16/AMP path is safe
- [ ] Dispatch / fallback logic is correct
- [ ] Tests are sufficient
- [ ] Performance claim is supported
- [ ] PR scope is reviewable; if the PR exceeds 350 changed lines, call out the size as a review risk unless it is clearly justified

## Output style

Be concise, specific, and actionable.
Prefer comments like:
- “This may introduce hidden host synchronization.”
- “Please confirm this path supports non-contiguous input.”
- “Index calculation looks 32-bit; large tensor overflow should be checked.”
- “PR claims speedup but lacks benchmark evidence.”
- “Please add BF16/channels_last coverage for this path.”

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: xpu-ops-pr-review.

Otherwise, keep the reply in the requested review format and do not force an extra trailing sentence.

## Best practices

- Focus on high-risk issues first
- Separate must-fix issues from suggestions
- Do not approve performance PRs without evidence
- Do not rely only on happy-path tests
- Flag fixes that use synchronization to hide correctness problems
- Flag PRs with more than 350 changed lines as a scope problem unless the size is clearly justified