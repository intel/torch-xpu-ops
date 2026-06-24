---
name: issue-fix
description: >
  Moved to fix/implement. Use that skill instead.
---

# Moved to fix/implement

This skill has been refactored into `fix/implement`.

Use `fix/implement` directly. It contains the same fix implementation logic —
with the `allow_skip` parameter controlling whether skip decorators are
permitted (false for issue-handler, true for nightly-ci-fix), and pipeline
mode issue body writes removed (those now belong to the orchestrator).
