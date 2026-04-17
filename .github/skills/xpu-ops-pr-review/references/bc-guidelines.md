# Backward Compatibility Guidelines For Torch XPU Ops Reviews

Treat any user-visible behavior change in `torch-xpu-ops` as backward-compatibility-sensitive until verified otherwise.

## What Counts As User-Visible Behavior

This includes more than public Python signatures. Flag changes in:

- Output values, shapes, strides, or layout behavior
- Error type, error timing, or error messages when tests or callers depend on them
- Dtype promotion, accumulation precision, or autocast behavior
- Fallback behavior versus explicit failure
- Determinism, synchronization side effects, or default execution path
- `functional`, `out=`, `inplace`, view, or backward semantics

## Review Rules

- A change can be BC-sensitive even when it looks like a bug fix
- If behavior changes, ask whether existing callers may rely on the old behavior
- Do not dismiss a semantic difference just because it only affects XPU users
- Treat silent fallback changes as BC-sensitive when they alter observed behavior or performance contracts

## Common BC Questions For XPU Reviews

1. Does the PR change which backend path is used for a public operator?
2. Does it change dtype promotion, accumulation precision, or mixed-precision behavior?
3. Does it alter output layout, contiguity expectations, or channels-last handling?
4. Does it raise a new error or reject inputs that previously ran?
5. Does it change `out=`, `inplace`, backward, or view behavior?
6. Does it make execution more synchronous in a way users can observe?

## Acceptable Outcomes

- No user-visible behavior change after inspection
- A real bug fix with a clear correctness justification and tests
- An intentional behavior change with explicit rationale and acknowledgement of the compatibility impact

## Reviewer Guidance

- If you cannot verify the blast radius, mark the point as BC-sensitive or `not verified` rather than asserting it is safe
- Ask for release-note or migration discussion only when the change really alters user-visible behavior
- Keep BC findings concrete: say what changed, who could observe it, and what evidence you checked