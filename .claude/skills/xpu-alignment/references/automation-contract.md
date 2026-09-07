# Automation Contract

The orchestrator supplies a run directory and exactly one agent role:
`scan-prepare`, `scan-finalize`, or `review`. Agents write only their owned
artifacts and never modify GitHub objects. A deterministic collector runs before
the agents, and a deterministic runner sits between the two scan roles.

## Layout and ownership

```text
collection/collection.json   # collector-owned manifest and inventory
collection/pages/<source>/   # collector-owned raw GraphQL responses
prepare.json                 # scan-prepare-owned decisions and execution plan
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

## Deterministic collector

The collector uses read-only GitHub credentials and no model, XPU, cloud, or
publishing credentials. It resolves the repository's default branch and freezes
its head SHA at collection start. It then uses cursor pagination, not GitHub
Search, to enumerate the requested half-open UTC window once:

- issues ordered by creation time;
- pull requests ordered by creation time;
- merged pull requests ordered by update time and filtered by merge time;
- commits reachable from the frozen default-branch head.

Every successful page is stored as immutable JSON before requesting the next
page. Each page entry records its path, SHA-256, input cursor, output cursor, and
item count. The source state continuously records its aggregate progress,
remaining quota, and reset time; these rate fields need not be repeated in every
page entry. A source stops only after reaching the lower time boundary or
exhausting its connection. The collector refreshes a partial manifest after every
page, including `in-progress` and `not-started` source progress, and uses an
internal soft deadline before the workflow's thirty-minute hard timeout so
interruption evidence is not held only in memory.

```json
{
  "schema_version": 1,
  "status": "partial",
  "repository": "pytorch/pytorch",
  "scan_window": {
    "start": "2026-08-20T00:00:00Z",
    "end": "2026-08-21T00:00:00Z"
  },
  "snapshot": {
    "collected_at": "2026-08-21T02:00:03Z",
    "default_branch": "main",
    "default_branch_head": "..."
  },
  "sources": [{
    "source": "issues-created",
    "status": "partial",
    "pages_completed": 1,
    "items_fetched": 25,
    "last_cursor": "...",
    "boundary_reached": false,
    "rate_remaining": 0,
    "rate_reset_at": "2026-08-21T03:00:00Z",
    "error": {"kind": "rate-limit", "message": "wait budget exhausted"},
    "pages": [{
      "path": "collection/pages/issues-created/page_0001.json",
      "sha256": "...",
      "cursor": null,
      "next_cursor": "...",
      "count": 25
    }]
  }],
  "observed_count": 25,
  "unique_count": 25,
  "inventory": [{
    "id": "issue-123",
    "kind": "issue",
    "title": "...",
    "url": "https://github.com/pytorch/pytorch/issues/123",
    "events": [{"type": "created", "at": "2026-08-20T03:00:00Z"}]
  }],
  "blockers": ["issues-created:rate-limit"]
}
```

The sample abbreviates the other three required source entries and the rest of
the inventory; a real manifest always includes all four sources and exact counts.

The required sources are `issues-created`, `prs-created`, `prs-merged`, and
`default-branch-commits`. `observed_count` counts source events before object
deduplication; `unique_count` counts inventory objects after stable-identity
deduplication. A PR created and merged in the window is one inventory object
with both events.

`status: complete` requires every source to be complete, every page and cursor
link to validate, exact count agreement, and no blocker. `status: partial`
requires the same structural integrity for all pages that were returned, names
the failed source and progress, and has at least one blocker. Network and server
errors receive bounded retries. Rate-limit responses honor their advertised
retry/reset time for at most ten minutes total; the collector job itself is
bounded to thirty minutes. A malformed response, repeated cursor, missing raw
page, or digest mismatch is not a valid partial collection.

## `scan-prepare` role

Read the immutable collection artifact and use read-only GitHub access only for
the source details needed to judge each observed object. Write `prepare.json`
and `scripts/` only; do not execute a reproducer or write results. This role does
not require an XPU runtime.

```json
{
  "schema_version": 1,
  "status": "complete",
  "scan_window": {
    "start": "2026-08-20T00:00:00Z",
    "end": "2026-08-21T00:00:00Z"
  },
  "collection_sha256": "...",
  "collection_status": "partial",
  "decisions": [{
    "id": "issue-123",
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

`collection_sha256` names the exact collector manifest bytes, and
`collection_status` must match the original manifest. `decisions` covers every
observed inventory id exactly once with `reject` or `validate` and a concrete
reason. Every validated item has exactly one execution entry; rejected items
have none. An object whose body, reproducer, tests, or diff shows independent XPU
work already tracked or implemented upstream is rejected with
`already-xpu-scoped` in its reason; a title, label, or XPU mention alone cannot
establish that scope. Generic, shared, or multi-backend work originating in CPU,
CUDA, ROCm, MPS, or another backend remains eligible when XPU parity is unknown.
For an explicitly linked issue/PR/commit chain, validate one canonical object at
most and reject the rest with `duplicate-chain` plus the canonical inventory id
in each free-text reason.
An execution identifies immutable script bytes, uses the default 120-second
timeout unless evidence justifies a smaller value, and states the upstream oracle
and expected XPU target path. Any missing detail or coverage makes preparation
incomplete. A structurally valid partial collection may still have a complete
preparation relative to its observed inventory; that does not make the collection
complete. The deterministic inventory does not prove that each semantic rejection
is correct, and automation deliberately uses no negative-sample review.

## Deterministic runner

The runner is not an agent. It validates the original collection and
`prepare.json`, re-hashes every script, and executes validated entries serially.
Each child receives only the allowlisted
runtime variables needed for Python, locale, HOME, and XPU. It never receives
GitHub, model-provider, cloud, or publishing credentials and has no outbound
network access. It runs each reproducer in a separate process group and removes
the entire group on completion or timeout. The runner acts as a child subreaper
and also terminates detached descendants so they cannot escape the bound.

Before running a reproducer, the runner probes the execution environment as the
same unprivileged user with the same credential-free environment used by the
children. Failure to import PyTorch or an unavailable XPU is a global environment
failure. Optional metadata failures retain a null value and append a concrete
warning. Each reproducer receives a separate writable scratch directory for
`HOME`, `TMPDIR`, and `TORCH_COMPILE_DEBUG_DIR`; immutable inputs remain
read-only.

The runner continues after a timeout, nonzero exit, signal, or launch error and
writes one result for every execution-plan entry:

```json
{
  "schema_version": 1,
  "collection_sha256": "...",
  "prepare_sha256": "...",
  "status": "complete",
  "environment": {
    "python_executable": "/usr/bin/python3",
    "python_version": "3.13.7",
    "torch_version": "2.9.0.dev20260820+xpu",
    "torch_path": "/opt/conda/lib/python3.13/site-packages/torch/__init__.py",
    "xpu_available": true,
    "xpu_device_name": "Intel(R) Data Center GPU Max 1550",
    "environment_warnings": []
  },
  "results": [{
    "id": "issue-123",
    "script_sha256": "...",
    "command": ["/usr/bin/python3", "-I", "-u", "scripts/repro_issue-123.py"],
    "log": "runner/logs/issue-123.log",
    "log_sha256": "...",
    "returncode": 0,
    "timed_out": false,
    "duration_seconds": 1.25,
    "error": null
  }]
}
```

`command` records the invoked Python options and artifact-relative script path;
the scratch working directory, credential-free environment, and process-group
isolation remain runner-owned execution context. Python runs unbuffered so
diagnostic output written before a timeout or signal is retained in the log.

`status: complete` means the runner produced a structurally valid result for
every planned execution, not that every reproducer succeeded. The collection
digest must match the prepare artifact and original collector manifest. A digest
mismatch or missing result blocks finalization. A valid partial collection does
not prevent execution or publication of fully covered, independently reviewed
units.

## `scan-finalize` role

Read immutable collection, `prepare.json`, scripts, `runner/results.json`, and raw
logs. Write only `scan.json` and optional `scan_report.md`:

```json
{
  "schema_version": 1,
  "status": "complete",
  "collection_sha256": "...",
  "collection_status": "partial",
  "prepare_sha256": "...",
  "runner_sha256": "...",
  "environment": {
    "python_executable": "/usr/bin/python3",
    "python_version": "3.13.7",
    "torch_version": "2.9.0.dev20260820+xpu",
    "torch_path": "/opt/conda/lib/python3.13/site-packages/torch/__init__.py",
    "xpu_available": true,
    "xpu_device_name": "Intel(R) Data Center GPU Max 1550",
    "environment_warnings": []
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

`status` is `complete` or `incomplete`. Candidates cover the entire
validated set exactly once and use a result from `evidence.md`. `confirmed`,
`related-failure`, and `not-reproduced` require a successful runner record,
matching script and log digests, target-path proof, and a defensible oracle.
Timeouts, launch errors, environment failures, or inconclusive evidence use a
`blocked-*` result and make the scan incomplete. Rejected inventory items remain
in the collection and prepare artifacts and are not copied into `scan.json`.
`status: complete` is relative to the observed inventory; it does not clear a
partial collection scope.

The environment object must exactly match the runner artifact. This role
interprets the runner's recorded XPU environment and does not require an XPU
device or GitHub access of its own.

## `review` role

The reviewer receives the immutable collection, prepare, runner, and scan
artifacts. It does not execute code or sample rejected inventory. It covers every
`confirmed` and `related-failure` candidate and writes only `review/`:

```json
{
  "schema_version": 1,
  "status": "complete",
  "collection_sha256": "...",
  "collection_status": "partial",
  "scan_sha256": "...",
  "units": [{
    "id": "issue-123",
    "verdict": "needs-xpu-fix",
    "implementation_repository": "pytorch/pytorch",
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
`needs-xpu-fix` without a canonical tracker has a payload. Its
`implementation_repository` is the GitHub `owner/repo` where the code change
belongs; the payload still targets `intel/torch-xpu-ops`. For `track-upstream`,
the field names the repository that already owns the implementation, or
`intel/torch-xpu-ops` for observation-only parity work that depends on an
upstream change landing. Other verdicts do not use this field. `status: blocked`
lists blockers and contains no payloads.
When an existing
`intel/torch-xpu-ops` issue covers the same work, record its URL as
`canonical_tracker`; do not create a payload or comment on that tracker.

This role requires read-only GitHub access to refresh source and tracker state,
but it does not require an XPU runtime.

## Gate requirements

The external gate validates each owner's original artifact: schema version,
scan window, raw page digests and cursor evidence, collection progress, inventory
metadata and counts, triage/execution coverage, paths and digests, runner
coverage, terminal scan results, exact review coverage, verdict vocabulary,
payload ownership, and payload shape.

Clean producer jobs and complete artifact coverage are required for publication.
An individual runner-backed unit blocker excludes only that unit; it does not
invalidate other fully covered, independently reviewed payloads. A scheduled run
automatically files all review-approved payloads when there are one to three.
With four or more payloads, it publishes every candidate as a draft for manual
handling and files none automatically. The same policy applies to a structurally
valid partial collection, but the workflow also publishes the source progress
and errors, notifies maintainers for a scheduled run, and finishes red. Source
progress is also shown when a partial collection has an unrelated global
blocker. Dry runs publish drafts only and never notify. A malformed collection,
incomplete coverage, environment core failure, or producer job failure publishes
only a blocker summary.
