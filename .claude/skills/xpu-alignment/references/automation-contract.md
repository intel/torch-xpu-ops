# Automation Contract (v1)

The orchestrator supplies a run directory and exactly one role: `scan` or
`review`. Agents write artifacts but never modify GitHub objects.

## Layout and ownership

```text
scan.json                 # scan-owned canonical state
scripts/repro_<id>.py     # exact reproducer bytes
logs/<id>.log             # raw execution output
scan_report.md            # optional human summary
review/review.json        # reviewer-owned canonical state
review/review_report.md   # optional human explanation
```

The scan upload is immutable input to review and the gate. The reviewer writes
only under `review/`. The gate validates the original scan upload, not a copy
that passed through the reviewer workspace.

Unit ids match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Every artifact path is relative
to and remains inside the run directory.

## Scan role

The scan agent enumerates, triages, writes, and directly executes faithful XPU
reproducers. Run each script with a bounded timeout in a fresh child process whose
environment omits GitHub, model-provider, cloud, and publishing credentials.

Write `scan.json`:

```json
{
  "schema_version": 1,
  "status": "complete",
  "scan_window": {
    "start": "2026-08-20T00:00:00Z",
    "end": "2026-08-21T00:00:00Z"
  },
  "collection": {
    "complete": true,
    "sources": ["issues-created", "prs-created", "prs-merged", "default-branch-commits"],
    "errors": []
  },
  "environment": {
    "python": "/usr/bin/python3",
    "torch": "...",
    "xpu_available": true,
    "device": "..."
  },
  "candidates": [{
    "id": "issue-123",
    "kind": "issue",
    "title": "...",
    "url": "https://github.com/pytorch/pytorch/issues/123",
    "triage": "validate",
    "reason": "...",
    "local_result": "confirmed",
    "reproducer": "scripts/repro_issue-123.py",
    "log": "logs/issue-123.log",
    "target_path_verified": true,
    "oracle": "..."
  }],
  "blockers": []
}
```

`status` is `complete`, `incomplete`, or `blocked`. `complete` requires complete
collection, no blockers, and a terminal triage/result for every candidate.

`triage` is `reject` or `validate`. A rejected candidate has a concrete reason and
`local_result: null`. A validated candidate uses one result from `evidence.md`.
`confirmed`, `related-failure`, and `not-reproduced` require existing reproducer
and log files plus `target_path_verified: true` and a nonempty oracle. Blocked
results preserve the available evidence and make the scan incomplete.

## Review role

The reviewer receives the immutable scan upload, covers every `confirmed` and
`related-failure` candidate, and writes only `review/review.json` plus an optional
human report:

```json
{
  "schema_version": 1,
  "status": "complete",
  "scan_sha256": "...",
  "units": [{
    "id": "issue-123",
    "verdict": "needs-xpu-fix",
    "implementation_repository": "intel/torch-xpu-ops",
    "canonical_tracker": null,
    "payload": {
      "title": "[xpu-alignment] ...",
      "body": "...",
      "labels": ["ai_generated"]
    }
  }],
  "blockers": []
}
```

`scan_sha256` is the SHA-256 of the exact `scan.json` reviewed. `units` covers the
provisional actionable set exactly once. Only `needs-xpu-fix` without a canonical
tracker has a payload. `status: blocked` lists blockers and contains no payloads.

## Gate requirements

The external gate validates schema version, ids and paths, scan window,
collection/scan completion, referenced evidence files, scan digest, exact review
coverage, verdict vocabulary, payload ownership, and payload shape.

Unattended filing additionally requires clean scan and review jobs and exactly
one review-approved `needs-xpu-fix` payload. Two or more payloads go to human
triage. A blocked or incomplete run never files unattended.
