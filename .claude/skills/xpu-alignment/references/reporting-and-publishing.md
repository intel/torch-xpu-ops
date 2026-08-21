# Reporting and Publishing

Keep human reports concise and evidence-led. Machine manifests carry workflow
state; reports explain judgments and link to the raw evidence.

## Scan report

`reports/scan_report.md` includes:

- window, event semantics, environment, and collection completeness;
- collected/rejected/validated counts and result counts;
- one evidence section for each `confirmed`, `related-failure`, blocker, or other
  candidate needing human attention;
- pending work and collection/execution blockers;
- negative results summarized by category, with ledger links rather than a
  mandatory prose section for every ordinary rejection.

Each actionable evidence section links its source, reproducer, execution record,
raw log, target-path XPU proof, upstream oracle comparison, and provisional route.

## Review report

`review/review_report.md` states review status, exact mandatory scope, negative
samples, promotions, and exclusions. For each formal unit include:

- source and current live state;
- repro fidelity and XPU execution evidence;
- real-bug and ownership reasoning;
- canonical tracker and relevant fix PRs;
- implementation repository;
- verdict and one concrete next action.

## Issue payload

Automation creates `review/final_issue_<id>.json` only for a reviewed
`needs-xpu-fix` unit that has no existing canonical tracker requiring reuse. Its
title starts with `[xpu-alignment]`, its labels are exactly `["ai_generated"]`,
and its body includes:

- upstream source and scan window;
- observed XPU behavior and independent-review verdict;
- target-path evidence and a copy-pasteable reproducer;
- relevant raw output and environment/build identity;
- canonical tracking and implementation repositories.

The payload is immutable input to an external publisher; downstream automation
must not ask an agent to rewrite it after approval.

## Publishing boundary

Interactive mode presents reviewed payloads, checks live duplicate state again,
and asks the user immediately before each kind of GitHub write. Automation agents
never publish. A workflow may publish under a predeclared external policy only
after validating the versioned scan/review contracts. An incomplete collection,
scan, or review must disable unattended publishing while preserving completed
work for human triage.
