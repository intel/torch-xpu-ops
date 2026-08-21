# Automation Contract (v1)

Automation separates agent judgment from reproducible execution and publishing.
The orchestrator supplies a run directory and one role: `scan-prepare`,
`scan-finalize`, or `review`. Agents never modify GitHub objects.

## Layout

Paths are relative to the run directory:

```text
scan_manifest.json
artifacts/
  raw_candidates.json
  candidate_ledger.jsonl
  collect_env.txt
  execution_plan.json
  execution_results.json
  details/<id>.json
  output_<id>.log
scripts/repro_<id>.py
reports/scan_report.md
review/
  review_manifest.json
  review_report.md
  final_issue_<id>.json
```

The scan upload is immutable input to review and the gate. Review outputs live
under `review/`; the reviewer must not rewrite scan-owned files. A workflow may
transport a copy of the scan beside review output, but the gate derives scan state
from the original scan upload.

## `scan-prepare`

Verify the environment, completely enumerate the requested event set when
possible, write every ledger row, triage candidates, prepare faithful reproducers,
and perform semantic precheck. Do not execute a reproducer. Write an execution
plan containing only approved scripts.

Write `scan_manifest.json`:

```json
{
  "schema_version": 1,
  "mode": "automation",
  "phase": "prepared",
  "status": "incomplete",
  "scan_window": {
    "start": "2026-08-20T00:00:00Z",
    "end": "2026-08-21T00:00:00Z",
    "timezone": "UTC"
  },
  "environment": {
    "python": "/usr/bin/python3",
    "torch": "...",
    "xpu_available": true,
    "device": "..."
  },
  "collection": {
    "status": "complete",
    "sources": {
      "issues": {"event_types": ["created"], "queries": [{"request": "...", "pages": 1, "count": 0, "truncated": false}], "pages": 1, "count": 0, "truncated": false},
      "prs": {"event_types": ["created", "merged"], "queries": [{"request": "...", "pages": 1, "count": 0, "truncated": false}], "pages": 1, "count": 0, "truncated": false},
      "commits": {"event_types": ["default-branch"], "queries": [{"request": "...", "pages": 1, "count": 0, "truncated": false}], "pages": 1, "count": 0, "truncated": false}
    },
    "errors": []
  },
  "raw_candidates": "artifacts/raw_candidates.json",
  "ledger": "artifacts/candidate_ledger.jsonl",
  "execution_plan": "artifacts/execution_plan.json",
  "execution_results": "artifacts/execution_results.json",
  "blockers": ["execution-pending"]
}
```

Collection status is `complete`, `incomplete`, or `blocked`. Preserve partial
results and describe errors. The top-level prepared status remains `incomplete`
until external execution and finalization finish.

Each source `count` is its deduplicated object count after unioning that source's
event queries. Each `queries` entry records a non-secret request description and
its pages, raw result count, and truncation state; source `pages` is their sum.
`artifacts/raw_candidates.json` is a JSON array containing at least the ledger's
source fields for every collected object. Its ids and per-kind counts must exactly
match the ledger and the three source counts.

`artifacts/execution_plan.json` has `schema_version: 1` and a `scripts` array.
Each entry contains `id`, run-directory-relative `path` and `log_path`,
`timeout_seconds`, `sha256`, `precheck_status: approved`, `upstream_oracle`,
`target_xpu_path`, and `xpu_proof`. Unit ids match
`[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Paths must remain inside the run directory.

## Credential-free execution

The orchestrator runs the approved plan with a fixed Python executable outside
an agent step, under a non-root identity that cannot write scan-owned files, and
writes `artifacts/execution_results.json`. Its v1 `results` entries contain `id`,
script/log paths, verified SHA-256, runner status, timeout state, return code or
signal, duration, and timestamps. Raw stdout/stderr goes to the planned log. The
runner records observations; it does not decide XPU buckets.

## `scan-finalize`

Read the immutable plan, execution results, logs, details, and ledger. Reconcile
each result with its planned oracle and XPU proof; update selected ledger rows to
terminal local results and write `reports/scan_report.md`.

Set the manifest to `phase: final`. Set `status: complete` only when collection is
complete, all ledger triage decisions are terminal, and every selected validation
is done with valid evidence. Otherwise set `incomplete` or `blocked`, retain
pending rows, and list blockers. Never reject a row merely to make the run appear
complete.

## `review`

Act as the independent reviewer in
[review-contract.md](review-contract.md). Write only under `review/`.

`review/review_manifest.json`:

```json
{
  "schema_version": 1,
  "review_status": "complete",
  "sample_policy": {"per_category": 3, "order": "id-lexical"},
  "mandatory_units": ["issue-123"],
  "negative_samples": [{"id": "pr-456", "category": "not-reproduced", "outcome": "accepted"}],
  "promoted_units": [],
  "units": [{
    "id": "issue-123",
    "verdict": "needs-xpu-fix",
    "implementation_repository": "intel/torch-xpu-ops",
    "canonical_tracker": null,
    "payload": "review/final_issue_issue-123.json"
  }],
  "blockers": []
}
```

The mandatory set must exactly match actionable rows derived from the immutable
scan ledger. `units` covers every mandatory and promoted id exactly once. Only a
`needs-xpu-fix` unit without a reusable canonical tracker may name a payload.
`review_status: blocked` lists blockers and contains no publishable payloads.

## Gate requirements

The external gate validates schema version, path/id safety, collection and scan
status, terminal ledger state, exact actionable review coverage, deterministic
negative sampling, verdict vocabulary, payload ownership, and payload shape.

Unattended publishing requires complete collection, a complete scan, a complete
review, no unresolved blocked validation, clean producing jobs, and the caller's
explicit automation policy. A negative-sample promotion or `verification-gap`
also requires human triage. A partial run may route completed reviewed payloads
to human triage but cannot use the unattended path.
