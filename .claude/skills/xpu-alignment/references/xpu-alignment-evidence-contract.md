# Runtime Evidence Contract

Runtime evidence proves what executed. It does not decide whether the behavior is
a real issue or needs an XPU-specific fix. Every `repro-ready` case creates
`artifacts/evidence/<case_key>.json`; semantic conclusions live in the case
assessment.

**Contents:** evidence record, runtime statuses, proof rules, audit.

## Evidence record

Use schema version 3. Paths are relative to the run directory.

```json
{
  "schema_version": 3,
  "case_key": "issue-123",
  "repro_path": "scripts/repro_issue-123.py",
  "repro_sha256": "<sha256>",
  "environment_fingerprint": "<sha256>",
  "attempts": [
    {
      "attempt_id": "attempt-1",
      "cache_namespace": "artifacts/cache/issue-123/attempt-1",
      "log_path": "artifacts/output_issue-123_attempt-1.log",
      "log_sha256": "<sha256>",
      "runtime_result": {
        "case_key": "issue-123",
        "xpu_proof": {
          "devices": ["xpu:0"],
          "key_ops": ["aten::<op>"],
          "evidence": "input=xpu:0; output=xpu:0"
        },
        "path_observations": {
          "fallback_detected": false,
          "frontend_only": false,
          "compiler_backend": "inductor"
        },
        "execution": {
          "mode": "compile",
          "eager_baseline": {"status": "pass"},
          "target": {
            "expected_stage": "inductor",
            "observed_stage": "inductor",
            "status": "expected-failure"
          }
        },
        "upstream_oracle": {
          "source": "<URL or test path>",
          "description": "<expected behavior>",
          "signature": "<normalized signature>"
        },
        "observed_oracle": {
          "description": "<observed behavior>",
          "signature": "<normalized signature>"
        },
        "exit_status": 0,
        "reproduction_status": "matched-upstream",
        "fidelity": "matched"
      }
    }
  ],
  "final_runtime": {
    "reproduction_status": "matched-upstream",
    "stable_signature": "<normalized signature>",
    "qualifying_attempts": ["attempt-1", "attempt-2"]
  },
  "evidence_contract_version": 3
}
```

Each attempt's final nonempty log line is
`XPU_ALIGNMENT_RESULT=<canonical-one-line-JSON>`. Copy that exact object into
`runtime_result`. Before it, emit device, observed-stage, observed-signature,
fallback, and key-op markers. Record the log hash, cache namespace, and exit
status. Fingerprint the stable interpreter, torch version, controls, and
isolation policy from `environment.json`.

## Runtime statuses

Use exactly one:

| Status | Runtime meaning |
|---|---|
| `matched-upstream` | The intended stage produced the same normalized behavior as the current upstream oracle. |
| `different-failure` | The intended XPU path reached a distinct behavior with its own observed signature. |
| `not-reproduced` | The intended stage ran and the expected failure was absent. |
| `oracle-not-reached` | A precondition or earlier stage prevented the target oracle. |
| `blocked-env` | A dependency, topology, control, or shared incident blocked execution. |
| `blocked-platform` | No corresponding XPU path exists. |
| `blocked-fetch` | Required source material could not be retrieved. |
| `blocked-script-error` | The repro itself failed before a meaningful oracle. |
| `repro-construction` | A faithful runnable repro could not yet be built. |
| `verification-gap` | Runtime evidence is insufficient or inconsistent. |
| `needs-performance-harness` | Only a missing benchmark can answer the runtime question. |

`matched-upstream` is not synonymous with a bug. `different-failure` is not
issue evidence until the assessment establishes an independent reference oracle.

## Runtime proof rules

- Prove a key input, intermediate, output, or compiler target is XPU. Printing
  `torch.xpu.is_available()` is never proof.
- Record fallback observations. An XPU output after CPU fallback does not prove
  XPU kernel execution.
- Compiler cases require a passing eager baseline and the intended target stage.
  An eager failure before a compile oracle is `oracle-not-reached`.
- Derive signatures from actual logs and current upstream sources. Do not use a
  static case-to-status table.
- Preserve unedited attempt evidence. A crash, device assert, or timeout gets a
  fresh process and cache namespace for retry.
- Require two qualifying attempts when a provisional assessment says
  `xpu-fix-required`, or when `different-failure` might support a new XPU issue.
  One audited attempt is sufficient for a stable shared-fix/non-issue outcome;
  any observed instability still requires a clean retry.

## Audit

Write `artifacts/audit.json`; never repair source artifacts during audit. Report
all errors and warnings. Runtime evidence is valid only when:

1. Repro, log, and environment hashes recompute exactly.
2. Every cache namespace is unique to its case and attempt.
3. The final log line byte-matches `runtime_result`.
4. Device, stage, signature, fallback, and key-op markers match the record.
5. The selected runtime status follows the table and reaches required stages.
6. A provisional XPU-fix or independently actionable different-failure case has
   two stable qualifying attempts.
7. Shared failure signatures reference one environment incident and clean retry.

The run audit additionally checks exact raw/source-ledger equality, unique
case-ledger keys, valid references between ledgers/evidence/assessments, complete
ready cases, and coverage/report count equality. It records `PASS` only when the
entire window is closed. A case assessment may independently record `PASS` for
authorized case-level filing while the run remains partial.
