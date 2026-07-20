# Issue-Handler Handoff

`issue-handler` exclusively owns implementation, rebuild verification, and PR
creation. Alignment prepares but does not start it without separate authorization.

## Eligibility

Include a case only when:

- the run-level audit is `PASS`
- `final_verdict` is `confirmed-xpu-issue`
- the XPU tracking issue exists and is canonical
- runtime evidence and assessment are valid
- `resolution_scope` is `xpu-fix-required`
- implementation repository, code owner, and concrete change are known
- no environment incident, design decision, merged fix, or active competing
  XPU-specific fix PR remains

Never hand off `track-upstream`, `non-issue`, or `verification-gap`. A fix in
shared PyTorch code is eligible only when the assessment proves XPU still needs
an independent change in that repository.

## Report

Write `reports/handler_handoff.md`, or one line saying no case is eligible:

```md
## <case_key> — <title>

- **XPU tracking issue:** <URL>
- **Behavior canonical:** <URL or none>
- **Implementation repository:** <pytorch/pytorch | intel/torch-xpu-ops>
- **Runtime evidence / assessment:** <paths and PASS statuses>
- **Reproducer:** `<command>` — <attempt logs>
- **Required build level:** <Python wheel | incremental C++/SYCL/ATen rebuild>
- **Fix scope proof:** <why an XPU-specific change is required>
- **Competing fixes:** <none>
- **Verification gaps:** <none>
- **Recommended issue-handler mode:** interactive
- **Next action / risk:** <concise implementation action>
```

Do not emit a batch handoff for a partial or blocked run. Default to interactive
handler mode. Pipeline mode, issue status changes, comments, and PR creation each
require explicit authorization for the named cases.
