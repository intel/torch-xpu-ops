---
applyTo: "src/**,yaml/**"
---

When reviewing implementation or operator-definition changes in this repository:

## For `src/**` changes
- Check correctness of kernel and operator behavior.
- Check indexing assumptions and boundary handling.
- Check dtype-specific behavior and conversion semantics.
- Check shape, stride, layout, and memory-sensitive behavior.
- Check whether the code path could change synchronization, fallback behavior, or backend-specific semantics.

## For `yaml/**` changes
- Check whether the definition or config change is intentional.
- Check whether implementation and tests are updated accordingly.
- Flag schema or config changes that are not reflected in `src/` or `test/`.

## Cross-file consistency
If both `src/` and `yaml/` change in the same PR:
- Verify consistency across definition, implementation, and tests.
- Call out mismatches between declared behavior and tested behavior.

## Review output requirements
- Distinguish must-fix issues from follow-up questions.
- Prefer precise file-level comments.
- Prioritize correctness and regression risk over formatting/style.