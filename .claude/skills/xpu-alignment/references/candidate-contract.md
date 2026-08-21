# Candidate and Ledger Contract

Read this reference for a time-window scan, candidate filtering, local result
classification, or ledger validation.

## Event set

Interpret a requested date range as a half-open UTC interval `[start, end)`.
Completely enumerate:

- issues created in the interval;
- PRs created or merged in the interval;
- commits reachable from the `pytorch/pytorch` default branch whose committer
  timestamp is in the interval. This is the reproducible API proxy for entering
  the branch; a separately enumerated merged PR retains its merge timestamp.

An old object's comment, label, or other ordinary update does not create a new
candidate. Fetch old linked objects as context. Deduplicate repeated query results
by stable GitHub identity while retaining all applicable event types.

Use any reliable read-only GitHub interface. Record each query or endpoint,
pagination count, result count, and whether an API/search cap was reached. Split
queries when necessary. A source is complete only when pagination ended normally
without truncation. Failure or uncertainty makes collection `incomplete` or
`blocked`, never silently complete.

Save the deduplicated metadata set to `artifacts/raw_candidates.json`. Its ids
and per-kind counts must match the ledger and collection manifest; this proves
that filtering did not silently drop a collected object before ledger creation.

## Triage

Create every ledger row before filtering. Titles and labels are inexpensive
signals; inspect bodies, linked objects, changed files, tests, or diffs whenever
they could change the decision. The question is whether the object contains a
plausible behavior/fix signal that could reach XPU, not whether it matches a fixed
keyword list.

Typical high-confidence rejections include documentation or infrastructure-only
work, nonfunctional refactors, backend code with no XPU/shared analogue, invalid
feature requests, and test toggles with no underlying behavior change. These are
examples, not automatic rules.

When an issue, PR, and commit clearly describe one behavior/root cause, choose the
object with the best reproducer and context as primary. Reject the others with
`rejection_category: duplicate-chain` and `duplicate_of: <primary id>`. An
uncertain relationship is not a duplicate.

Use one rejection category:

`docs-ci-release`, `platform-exclusive`, `test-toggle`, `nonfunctional`,
`duplicate-chain`, `insufficient-repro-context`, `nonbug`,
`no-shared-bug-signal`, or `other`.

## Local result vocabulary

Assign a result from observed evidence, not from a title or expected answer:

| Result | Meaning |
|---|---|
| `confirmed` | The target path ran on XPU and exhibited the same upstream oracle/signature. |
| `related-failure` | The target path ran on XPU and exhibited a different, independently stated actionable defect. |
| `not-reproduced` | The faithful target ran on XPU and the upstream oracle did not fail. |
| `blocked-env` | The validation could not start because its required runtime/dependency/topology was unavailable. |
| `blocked-platform` | XPU has no corresponding path on which to test the claim. |
| `blocked-fetch` | Required public source material could not be retrieved. |
| `blocked-script-error` | The script/setup failed before a defensible result, or target-path XPU proof is absent. |
| `needs-performance-harness` | The claim is performance-only and no valid comparative harness is available. |

For numerical behavior, preserve the upstream oracle and its dtype/operator
tolerances. Do not apply one global tolerance. A crash, timeout, or signal may be
classified only when parent-observed execution data and target-stage evidence
make the signature defensible.

## Version 1 ledger

`artifacts/candidate_ledger.jsonl` contains one JSON object per collected GitHub
object. Every row has:

```json
{
  "schema_version": 1,
  "id": "issue-123",
  "kind": "issue",
  "title": "...",
  "url": "https://github.com/pytorch/pytorch/issues/123",
  "events": [{"type": "created", "at": "2026-08-20T12:34:56Z"}],
  "triage_status": "pending",
  "triage_reason": "",
  "rejection_category": null,
  "duplicate_of": null,
  "validation_status": "not-needed",
  "local_result": null,
  "repro_path": null,
  "log_path": null
}
```

Allowed states and invariants:

- `triage_status`: `pending`, `reject`, or `validate`.
- A rejected row uses `validation_status: not-needed`, has a concrete reason and
  rejection category, and has no local result.
- A selected row uses `triage_status: validate` and `validation_status: pending`
  or `done`.
- `validation_status: done` requires exactly one local result. `confirmed`,
  `related-failure`, and `not-reproduced` also require a repro path, immutable
  execution result, raw log, and target-path XPU evidence.
- `duplicate_of`, when present, names another ledger id and is used only with
  `rejection_category: duplicate-chain`.

The ledger is a resume record, not proof that enumeration completed. That proof
belongs in the scan manifest. A completed scan has no `pending` triage row and no
selected row whose validation is pending.
