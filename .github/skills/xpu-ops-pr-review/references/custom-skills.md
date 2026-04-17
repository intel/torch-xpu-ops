# Custom Skills For XPU Ops PR Review

Use this file when the user wants to extend `xpu-ops-pr-review` with repository-specific or team-specific guidance.

## Core Rule

Write custom skill content in English unless the user explicitly requests another language.

## What Belongs In A Custom Skill Extension

Add only stable, reusable guidance such as:

- Repository-specific review policy
- Team-specific escalation rules
- Required reviewer reply format
- Label, workflow, or ownership conventions
- Extra files or subsystems that must be inspected for certain change types
- Organization-specific performance, compatibility, or testing gates

## What Does Not Belong

Do not put these into a reusable custom skill:

- One-off PR details
- Temporary reviewer preferences
- Unverified assumptions about backend behavior
- Ad hoc conclusions from a single review thread
- Anything that weakens the core evidence policy or XPU safety checks

## Non-Negotiable Review Invariants

Any custom extension must preserve these baseline rules:

- Findings stay evidence-based
- CPU or CUDA parity claims must be verified against the actual counterpart
- Hidden synchronization and stream ordering remain high-priority checks
- Layout, memory format, BF16 or FP16 numerics, and 64-bit indexing stay mandatory review areas
- Dispatch, fallback, and test coverage must still be checked together

## Recommended Structure

When adding a custom extension, use this shape:

```markdown
# <Custom Skill Name>

## Purpose
[One short paragraph describing the repo- or team-specific extension]

## When To Use
- [Trigger phrase or scenario]
- [Trigger phrase or scenario]

## Extra Checks
- [Stable custom review rule]
- [Stable custom review rule]

## Output Rules
- [Reply style, escalation rule, or formatting expectation]

## References
- [Relevant file or subsystem]
```

## Example Custom Extension

```markdown
# Intel Internal Review Addendum

## Purpose
Apply Intel-specific review expectations on top of the standard `xpu-ops-pr-review` workflow.

## When To Use
- The PR touches release gating, CI labels, or merge policy
- The reviewer asks for Intel-specific escalation or ownership guidance

## Extra Checks
- Confirm whether CI-disabling labels are justified by the actual file scope
- Check whether the affected subsystem has an expected maintainer or owner to notify
- Call out changes that alter internal review routing or workflow assumptions

## Output Rules
- Keep public review text concise and evidence-based
- Separate repo policy concerns from correctness findings
```

## Editing Guidance

- Prefer adding a new section or a new reference file instead of bloating the main `SKILL.md`
- Keep custom extensions shorter than the core skill unless the repo truly needs a large policy overlay
- If a custom extension changes trigger behavior, update the main skill description so the skill remains discoverable