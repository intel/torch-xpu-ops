# Reports, Drafts, and Filing

## Full scan report

Write `reports/full_scan.md` directly. Include every row with
`local_status == done` and exclude deep rejects.

Each numbered entry includes:

- candidate id, title, kind, and source URL
- repro and output-log paths
- exact ``Local XPU result: `<bucket>` `` line
- decisive observed behavior and upstream oracle
- provisional route for `confirmed` and `related-failure`
- blocker reason for blocked buckets

Keep `confirmed` and `related-failure` visibly labeled **provisional pending
independent review**. The final summary reports filter and provisional bucket
counts but makes no issue-filing claim.

## Provisional issue drafts

Write `reports/issue_drafts.md` for provisional `confirmed` and
`related-failure` rows:

````md
## Draft <n> - provisional

**Suggested title:** [xpu-alignment] <behavior>
**Tracking repository:** <suggested repository>
**Upstream source:** <URL>
**Local XPU result:** <confirmed | related-failure>
**Review verdict:** pending

### Observed behavior
<upstream oracle, faithful XPU adaptation, actual result, and target-stage proof>

```python
<minimal deterministic reproducer>
```

```text
<decisive output and parent termination record when applicable>
```

### Versions
- PyTorch: <version and git commit>
- XPU device: <device>
- Build: <source/date>
````

Sanitize credentials, usernames, hostnames, home paths, and unrelated environment
details. If no provisional issue candidate exists, record that in one line.

## Apply review

After independent review:

- keep and update a draft only for `needs-xpu-fix`
- mark `duplicate` with its canonical XPU tracker and remove it from filing
- mark `track-upstream`, `fixed`, `non-issue`, and `verification-gap` as not
  eligible, with the review-conclusion link
- do not silently delete rejected drafts; retain their disposition for audit

The review conclusions, not the provisional bucket, control filing.

## Filing gate

For each `needs-xpu-fix` draft:

1. Refresh the upstream behavior issue, all linked/superseded fix PRs, and the
   target tracking repository.
2. Search by source URL, operation/error signature, root cause, and fix commit.
3. If an XPU tracker exists, use it and do not create another.
4. Summarize the reviewed draft, target repository, labels, and dedup result.
5. Ask for explicit authorization naming that draft and repository.
6. Create only approved issues and record their URLs in the draft and review
   reports.

Issue creation does not authorize comments, labels, closure, handler execution,
or PR publication.

## Implementation handoff

Only a reviewed `needs-xpu-fix` case with a canonical XPU issue may enter
`issue-handler`. Pass the issue URL, repro command, logs, ownership conclusion,
required build level, and known verification gaps.

`issue-handler` owns implementation, rebuild, and fix verification.
`xpu-ops-pr-creation` prepares an Intel repository PR after verification but does
not authorize publication. A `pytorch/pytorch` implementation follows that
repository's PR process. Every PR publication is separately authorized.
