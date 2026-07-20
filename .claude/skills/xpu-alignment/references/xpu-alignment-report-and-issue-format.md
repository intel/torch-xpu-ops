# Report, Deduplication, and Filing

## Scan report

Write `reports/full_scan.md` with one numbered section per terminal case. Include:

- case key, source ids, and upstream URLs
- repro, attempt logs, runtime evidence, and assessment paths/statuses
- `reproduction_status`, `issue_validity`, `execution_path`,
  `resolution_scope`, `final_verdict`, and `filing_disposition`
- environment incident when present
- behavior canonical, XPU tracking canonical, fix PRs, tracking repository, and
  implementation repository

Do not summarize `matched-upstream` as a confirmed issue. The dashboard reports
separate counts for runtime status, issue validity, execution path, resolution
scope, final verdict, and filing disposition.

Only a run-level audit `PASS` may produce a final full-window dashboard. Otherwise
write a progress checkpoint with `STATUS=partial` or `STATUS=blocked`, pending
case keys, and any individually audited case outcomes.

## Issue drafts

Write a draft only when the assessment is `PASS`, `final_verdict` is
`confirmed-xpu-issue`, and `filing_disposition` is `file-xpu-tracker`:

````md
## Issue <n>

**Suggested title:** [xpu-alignment] <XPU-specific problem>
**Suggested labels:** xpu-alignment, confirmed
**Tracking repository:** <intel/torch-xpu-ops or experimental override>
**Implementation repository:** <pytorch/pytorch or intel/torch-xpu-ops>
**Upstream behavior:** <behavior canonical and source URLs>
**Case key:** <case_key>
**Runtime evidence:** <evidence path> — audit `valid`
**Issue assessment:** <assessment path> — proof ladder `PASS`

### Describe the bug
<reference behavior, observed XPU behavior, and why XPU needs an independent fix>

```python
<minimal deterministic XPU reproducer>
```

```text
<two clean attempts with matching decisive output>
```

### Ownership and canonical state
<XPU code path, proposed implementation repository, existing fix PRs, and
why no XPU tracking duplicate exists>

### Versions
<contents of artifacts/collect_env.txt>
````

An upstream behavior issue may be linked without suppressing the draft. An
existing XPU tracking issue changes the disposition to `duplicate` and suppresses
the draft. If no case is eligible, write one line saying so.

## Deduplication

Search the tracking repository independently using:

- upstream/source URLs
- operation, dtype, shape, and normalized error terms
- XPU code-path and root-cause terms
- linked PR and commit ids

Inspect likely matches; title similarity alone is insufficient. Record the live
behavior issue, XPU tracking issue, and all fix PRs in their separate assessment
fields. Multiple source rows sharing a case can produce at most one tracking
issue.

Refresh live issue and PR state immediately before filing. A shared fix that
covers XPU changes the case to `track-upstream`; an XPU-specific fix PR remains
linked but does not replace the required tracking issue.

## Authorization

A case-level assessment `PASS` may support a draft or authorized filing while the
run remains partial. State that the full-window scan is incomplete; never present
the case as a complete-window conclusion.

Summarize each proposed filing and ask for explicit authorization naming the
draft and target repository. Create only the approved issues. Issue creation,
comments, reviews, labels, and closures are separate GitHub write actions and
require their own authorization. Record created or matched URLs in the assessment,
case ledger, and report.
