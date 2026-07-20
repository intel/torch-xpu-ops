# Report, Deduplication, and Filing

## Scan report

Write `reports/full_scan.md` with the current
`artifacts/coverage.json.workflow_status` and one numbered section per selected
case. Include:

- case key, source ids, and upstream URLs
- repro, attempt logs, evidence, and assessment paths/statuses
- reproduction status, issue validity, execution path, resolution scope, final
  verdict, and filing disposition
- environment incident or construction blocker
- behavior canonical, XPU tracker, fix PRs, and both repositories

Keep runtime, validity, execution, scope, verdict, and filing counts separate.
Never summarize `matched-upstream` as a confirmed issue. A partial or blocked
report lists all pending/blocking case keys. `full_scan.md` is the scan record;
`review_dashboard.md` is the final decision dashboard.

## Issue drafts

Write a draft only when evidence and assessment are `valid`, every proof gate is
`pass`, `final_verdict` is `confirmed-xpu-issue`, and
`filing_disposition` is `file-xpu-tracker`:

````md
## Issue <n>

**Suggested title:** [xpu-alignment] <XPU-specific problem>
**Suggested labels:** <existing repository labels only>
**Tracking repository:** <intel/torch-xpu-ops or explicit override>
**Implementation repository:** <pytorch/pytorch or intel/torch-xpu-ops>
**Upstream behavior:** <behavior canonical and source URLs>
**Case key:** <case_key>
**Runtime evidence / assessment:** <paths>

### Describe the bug
<reference behavior, observed XPU behavior, and independent XPU fix proof>

```python
<minimal deterministic XPU reproducer>
```

```text
<two clean attempts with the same decisive stage and signature>
```

### Ownership and canonical state
<XPU code path, target implementation repository, linked fixes, and dedup result>

### Versions
- PyTorch: <version and git commit>
- XPU device: <device>
- Build: <source/date>
- Environment: <sanitized relevant details>
````

Do not paste raw credentials, usernames, hostnames, home paths, or the full
`collect_env` output into a public issue. If no case is eligible, record that no
draft was generated.

## Deduplication and filing

Search the tracking repository using source URLs, operation/dtype/shape/error
terms, XPU code-path/root-cause terms, and linked PR/commit ids. Inspect likely
matches; title similarity alone is insufficient.

Refresh live state immediately before filing. A shared fix changes the case to
`track-upstream`. A preexisting XPU tracker changes filing disposition to
`use-existing-xpu-tracker`, not the confirmed verdict.

File only after global `review_state.review_status` is `pass`, the corresponding
unit has `verdict: needs-xpu-fix`, and live dedup still finds no tracker. Then
summarize each proposed filing and ask for explicit authorization naming the
draft and target repository. Create only approved issues. After creation:

- move a previously completed workflow back to `partial`
- set `xpu_tracking_canonical_url` to the created URL
- set `tracker_origin: created-this-run`
- set `filing_disposition: filed-xpu-tracker`
- preserve `final_verdict: confirmed-xpu-issue`
- update the assessment, case ledger, coverage, and scan report
- mark the local draft as filed and record the created URL
- rebuild the review manifest and repeat independent review

Issue creation, comments, reviews, labels, closures, and dashboard publication
are separate GitHub write actions with separate authorization.
