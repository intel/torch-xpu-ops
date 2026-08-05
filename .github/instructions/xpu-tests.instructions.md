---
applyTo: "test/**"
---

When reviewing tests in this repository:

- Prefer focused regression coverage tied to the changed operator or behavior.
- If production code changes without corresponding targeted tests, call it out.
- Check whether the test validates the changed behavior directly instead of only exercising the happy path.
- Check whether boundary cases, dtype-sensitive behavior, and failure-mode coverage are missing when a bug fix is claimed.
- Avoid rewarding broad but low-signal test additions that do not clearly validate the intended change.

When suggesting improvements:
- Prefer the smallest useful test extension that increases confidence.
- Ask for explicit regression coverage when a previously failing or incorrect behavior is being fixed.