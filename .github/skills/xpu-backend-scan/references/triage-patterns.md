# Triage Patterns

Context for prioritizing findings. All findings are reported — nothing is dismissed.

## Patterns That Need Extra Context

When you see these patterns, report the finding but include the additional context to help humans prioritize:

| Pattern | Context to add | What to check |
|---------|---------------|---------------|
| No dedicated XPU kernel file | May use structured/shared/generic path | Backend YAML, delegate, composite, decomp |
| XPU kernel shorter than CUDA | Conciseness ≠ defect | Helper definitions, validation sites |
| oneDNN vs cuDNN | Vendor library choice, not a bug by itself | User-visible behavior differences |
| Helper call sites differ | Helpers may be equivalent | Actual helper definitions |
| Structured wrapper lacks local reg | May inherit delegate coverage | `structured_delegate` target |
| No backend-local backward symbol | Generic autograd may cover | `derivatives.yaml`, shared formulas |
| Skip/xfail in test metadata | Report it — may indicate a known gap | Runtime coverage and source-backed evidence |
| CUDA entry in YAML | CUDA presence does not prove XPU absence | Check all coverage signals |
| Only fallback coverage | Report as "fallback only" (low priority) — XPU still lacks native GPU impl | Whether the op should have native XPU support |

## High-Priority Signals — Escalate

Escalate when any of these are established:
- Valid input space changes between CUDA and XPU
- A parameter is ignored or handled differently
- Result values, warnings, or error paths diverge
- No callable XPU runtime path exists after all checks
- Concrete XPU-side validation, alias, numeric, or dispatch defect visible in code
- Silent CPU fallback on an op that should run on GPU

## Review Habits

- Read helper definitions before comparing call sites.
- Compare exact schemas, not just base op names.
- Record all evidence (positive and negative) for each finding.
- Test skip/xfail is a signal to report, not to dismiss.
- Prefer current source-backed facts over stale historical text.
- Judge family-level truth first, then row-level labeling.
