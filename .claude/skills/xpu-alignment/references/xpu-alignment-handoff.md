# Issue-Handler Handoff

`issue-handler` exclusively owns implementation, rebuild verification, and PR
creation. Alignment prepares handoff but never starts it without separate
authorization.

## Eligibility

Include a case only when:

- it belongs to a full-window scan with `workflow_status: completed`
- scan audit and independent review are `pass`
- evidence and assessment are `valid`
- `final_verdict` is `confirmed-xpu-issue`
- `resolution_scope` is `xpu-fix-required`
- review unit `verdict` is `needs-xpu-fix`
- review `fix_state` is `unfixed`
- a canonical XPU tracker exists and filing disposition is
  `use-existing-xpu-tracker` or `filed-xpu-tracker`
- implementation repository, code owner, and concrete change are known
- no active/retrying incident, design decision, merged fix, or active competing
  XPU fix remains; resolved incidents may remain in history

Never hand off a direct-review-only unit, `track-upstream`, `non-issue`, or
`verification-gap`.

## Report

Write `reports/handler_handoff.md`, or state that no case is eligible:

```md
## <case_key> - <title>

- **XPU tracking issue:** <URL>
- **Behavior canonical:** <URL or none>
- **Implementation repository:** <pytorch/pytorch | intel/torch-xpu-ops>
- **Runtime evidence / assessment:** <paths and valid statuses>
- **Reproducer:** `<command>` - <attempt logs>
- **Required build level:** <Python wheel | incremental C++/SYCL/ATen rebuild>
- **Fix scope proof:** <why an XPU-specific change is required>
- **Competing fixes:** <none>
- **Verification gaps:** <none>
- **Recommended issue-handler mode:** interactive
- **Next action / risk:** <concise implementation action>
```

Default to interactive mode. Pipeline mode, issue state changes, comments, and
PR creation each require explicit authorization for named cases.
