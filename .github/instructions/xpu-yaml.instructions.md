---
applyTo: "yaml/**"
---

When reviewing operator-definition or config changes in this repository:

## For `yaml/**` changes
- Check whether the definition or config change is intentional.
- Check whether implementation and tests are updated accordingly.
- Flag schema or config changes that are not reflected in `src/` or `test/`.
- Check whether dispatch, fallback, or backend coverage implied by the yaml change still matches the intended XPU behavior.

## Cross-file consistency
If both `yaml/` and `src/` change in the same PR:
- Verify consistency across definition, implementation, and tests.
- Call out mismatches between declared behavior and tested behavior.

## Review output requirements
- Distinguish must-fix issues from follow-up questions.
- Prefer precise file-level comments.
- Prioritize correctness and regression risk over formatting/style.