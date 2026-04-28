# Triage Patterns

Decision heuristics for separating real bugs from false positives.

## Quick Rule

If the finding does not identify a user-visible contract difference or a concrete missing runtime path, downgrade first.

## False Positives — Recognize and Dismiss

| Pattern | Why it's usually not a bug | Verify instead |
|---------|---------------------------|----------------|
| No dedicated XPU kernel file | May use structured/shared/generic path | Backend YAML, delegate, composite, decomp |
| XPU kernel shorter than CUDA | Conciseness ≠ defect | Helper definitions, validation sites |
| oneDNN vs cuDNN | Vendor library choice | User-visible behavior |
| Helper call sites differ | Helpers may be equivalent | Actual helper definitions |
| Structured wrapper lacks local reg | Inherits delegate coverage | `structured_delegate` target |
| No backend-local backward symbol | Generic autograd may cover | `derivatives.yaml`, shared formulas |
| Skip/xfail is main evidence | Test metadata only | Runtime coverage and source |
| CUDA entry in YAML | CUDA presence ≠ XPU absence | Fallback, composite, delegate, decomp |
| Fallback = missing impl | Fallback IS runtime coverage | Whether CPU fallback is a defect |

## Real Bugs — When to Escalate

Escalate when any of these are established:
- Valid input space changes between CUDA and XPU
- A parameter is ignored or handled differently
- Result values, warnings, or error paths diverge
- No callable XPU runtime path exists after all exclusions
- Concrete XPU-side validation, alias, numeric, or dispatch defect visible in code
- Silent CPU fallback on an op that should run on GPU

## Review Habits

- Read helper definitions before comparing call sites.
- Compare exact schemas, not just base op names.
- Record negative evidence alongside bug-looking evidence.
- Treat test metadata as supporting evidence only.
- Prefer current source-backed facts over stale historical text.
- Judge family-level truth first, then row-level labeling.

## Governance: Skill vs Automation

When reviewing findings for process improvement:
- **Keep in skill text**: reading order, recurring patterns, cautions needing judgment
- **Promote to automation only when**: condition is a stable boolean, applies to a family (not one-off), current facts clearly overturn stale text, and the rule cannot suppress future real bugs
