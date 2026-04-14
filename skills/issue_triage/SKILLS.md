## Workflow

### Phase 1: Classify Issue

Apply rules in priority order:

**L1 — No Action Needed**:
- All dynamically skipped cases are crossed out (passed_count == total, failed_count == 0)
- Issue labeled as `duplicate` (kept for historical reasons)
- Fix has been implemented and merged (verified via git log or PR status); only engineer review needed
- All cases verified PASSED by running on server (see Server Verification below)

**L2 — Additional Information Required**:
- Partial cases passed, remaining may need doc/internal reports
- Labeled as `random` or exhibits flaky behavior

**L3 — Engineer Intervention Required**:
- All cases still failing
- Issue labeled `wontfix` or `not_target` with failing cases (needs skip PR)
- Requires modifications to PyTorch repo (signals below)
- Feature requests, external dependency issues, bugs needing investigation

### Phase 2: Reply in Github
Please comment on the level and reasons for categorizing the issue in the designated GitHub issue.