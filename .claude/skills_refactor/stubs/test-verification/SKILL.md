---
name: test-verification
description: >
  Moved to fix/reproduce and fix/verify. Use those skills instead.
---

# Moved to fix/reproduce and fix/verify

This skill has been split into two:

- **`fix/reproduce`** — verifies whether a bug exists, using a three-stage
  approach (nightly wheel → source build at CI commit → CI environment
  alignment). Run before starting a fix.

- **`fix/verify`** — verifies whether a fix works, using source build only.
  Optionally runs before/after diff and lint. Run after `fix/implement`.

The split reflects that the two stages have different environment requirements:
reproduce uses nightly wheel as the fast path; verify always requires source
build because local code changes must be tested.
