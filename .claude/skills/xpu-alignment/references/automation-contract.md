# Automation Contract (v1)

The orchestrator supplies a run directory and exactly one agent role:
`scan-prepare`, `scan-finalize`, or `review`. Agents write only their owned
artifacts and never modify GitHub objects. A deterministic runner sits between
the two scan roles.

## Layout and ownership

```text
prepare.json                 # scan-prepare-owned inventory and execution plan
scripts/repro_<id>.py        # scan-prepare-owned exact reproducer bytes
runner/results.json          # runner-owned execution metadata
runner/logs/<id>.log         # runner-owned raw stdout/stderr
scan.json                    # scan-finalize-owned canonical scan state
scan_report.md               # optional scan-finalize explanation
review/review.json           # reviewer-owned canonical review state
review/review_report.md      # optional reviewer explanation
```

Every downstream artifact names the SHA-256 of the exact upstream JSON bytes it
consumed. The gate validates the original upload from each owner, not a copy that
passed through a later agent workspace. Unit ids match
`[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Every artifact path is relative to and remains
inside the run directory.

## `scan-prepare` role

Use read-only GitHub access to exhaust the requested half-open UTC window. Write
`prepare.json` and `scripts/` only; do not execute a reproducer or write results.
This role does not require an XPU runtime.

```json
{
  "schema_version": 1,
  "status": "complete",
  "scan_window": {
    "start": "2026-08-20T00:00:00Z",
    "end": "2026-08-21T00:00:00Z"
  },
  "collection": {
    "observed_count": 31,
    "unique_count": 29,
    "queries": [{
      "source": "issues-created",
      "request": "gh api ...",
      "pages": 2,
      "count": 31,
      "truncated": false,
      "errors": []
    }]
  },
  "inventory": [{
    "id": "issue-123",
    "kind": "issue",
    "title": "...",
    "url": "https://github.com/pytorch/pytorch/issues/123",
    "events": [{"type": "created", "at": "2026-08-20T03:00:00Z"}],
    "triage": "validate",
    "reason": "shared operator path"
  }],
  "executions": [{
    "id": "issue-123",
    "script": "scripts/repro_issue-123.py",
    "script_sha256": "...",
    "timeout_seconds": 120,
    "oracle": "...",
    "target_path": "..."
  }],
  "blockers": []
}
```

The required query sources are `issues-created`, `prs-created`, `prs-merged`,
and `default-branch-commits`. There may be multiple queries per source. Every
query records its exact request, page count, result count, truncation flag, and
errors. `observed_count` is the sum of the query counts before deduplication;
`unique_count` is the number of inventory entries after deduplication.
`status: complete` requires every required source, no query error or truncation,
and both counts to match their evidence.

The inventory contains every distinct object returned by the queries. Every item
has exactly one `triage` value: `reject` or `validate`, plus a concrete reason.
Every validated item has exactly one execution entry; rejected items have none.
An execution identifies immutable script bytes, uses the default 120-second
timeout unless evidence justifies a smaller value, and states the upstream oracle
and expected XPU target path. Any missing coverage makes preparation incomplete.

This evidence makes the agent's claimed enumeration auditable. It does not prove
that the agent chose every possible query or made every semantic rejection
correctly; automation deliberately uses neither an external collector nor
negative-sample review.

## Deterministic runner

The runner is not an agent. It validates `prepare.json`, re-hashes every script,
and executes validated entries serially. Each child receives only the allowlisted
runtime variables needed for Python, locale, HOME, and XPU. It never receives
GitHub, model-provider, cloud, or publishing credentials.

The runner continues after a timeout, nonzero exit, signal, or launch error and
writes one result for every execution-plan entry:

```json
{
  "schema_version": 1,
  "prepare_sha256": "...",
  "status": "complete",
  "results": [{
    "id": "issue-123",
    "script_sha256": "...",
    "command": ["/usr/bin/python3", "scripts/repro_issue-123.py"],
    "log": "runner/logs/issue-123.log",
    "log_sha256": "...",
    "returncode": 0,
    "timed_out": false,
    "duration_seconds": 1.25,
    "error": null
  }]
}
```

`status: complete` means the runner produced a structurally valid result for
every planned execution, not that every reproducer succeeded. A digest mismatch
or missing result blocks finalization.

## `scan-finalize` role

Read immutable `prepare.json`, scripts, `runner/results.json`, and raw logs. Write
only `scan.json` and optional `scan_report.md`:

```json
{
  "schema_version": 1,
  "status": "complete",
  "prepare_sha256": "...",
  "runner_sha256": "...",
  "environment": {
    "python": "/usr/bin/python3",
    "torch": "...",
    "xpu_available": true,
    "device": "..."
  },
  "candidates": [{
    "id": "issue-123",
    "local_result": "confirmed",
    "target_path_verified": true,
    "evidence": "runner/logs/issue-123.log"
  }],
  "blockers": []
}
```

`status` is `complete`, `incomplete`, or `blocked`. Candidates cover the entire
validated set exactly once and use a result from `evidence.md`. `confirmed`,
`related-failure`, and `not-reproduced` require a successful runner record,
matching script and log digests, target-path proof, and a defensible oracle.
Timeouts, launch errors, environment failures, or inconclusive evidence use a
`blocked-*` result and make the scan incomplete. Rejected inventory items remain
in `prepare.json` and are not copied into `scan.json`.

This role interprets the runner's recorded XPU environment and does not require
an XPU device or GitHub access of its own.

## `review` role

The reviewer receives the immutable prepare, runner, and scan artifacts. It does
not execute code or sample rejected inventory. It covers every `confirmed` and
`related-failure` candidate and writes only `review/`:

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

`units` covers the provisional actionable set exactly once. Only
`needs-xpu-fix` without a canonical tracker has a payload. `status: blocked`
lists blockers and contains no payloads.

This role requires read-only GitHub access to refresh source and tracker state,
but it does not require an XPU runtime.

## Gate requirements

The external gate validates each owner's original artifact: schema version,
scan window, query evidence, inventory/triage/execution coverage, paths and
digests, runner coverage, terminal scan results, exact review coverage, verdict
vocabulary, payload ownership, and payload shape.

Unattended filing additionally requires clean producer jobs and exactly one
review-approved payload. Two or more payloads go to human triage. A blocked or
incomplete run publishes no candidate verdict and never files unattended.
