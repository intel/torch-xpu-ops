# Runtime Evidence Contract

Runtime evidence proves what executed. It does not decide whether the behavior is
a real issue or needs an XPU-specific fix. Every `repro-ready` case that executes
creates `artifacts/evidence/<case_key>.json`; pre-execution blockers belong in
the case ledger and incident record. Semantic conclusions belong to assessment.

## Evidence record

Paths are relative to the run directory. Hash exact file bytes with SHA-256.

```json
{
  "case_key": "pytorch-pytorch-issue-999",
  "repro_path": "scripts/repro_pytorch-pytorch-issue-999.py",
  "repro_sha256": "<sha256>",
  "environment_path": "artifacts/environment.json",
  "environment_sha256": "<sha256>",
  "attempts": [
    {
      "attempt_id": "attempt-1",
      "command": ["<python>", "scripts/repro_pytorch-pytorch-issue-999.py"],
      "cache_namespace": "artifacts/cache/pytorch-pytorch-issue-999/attempt-1",
      "log_path": "artifacts/output_pytorch-pytorch-issue-999_attempt-1.log",
      "log_sha256": "<sha256>",
      "runtime_result": {
        "case_key": "pytorch-pytorch-issue-999",
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
            "status": "pass"
          }
        },
        "upstream_oracle": {
          "source": "<URL or test path>",
          "description": "<expected behavior>",
          "raw_signature": "<exact decisive text>",
          "normalized_signature": "<normalized signature>",
          "normalization_rule": "<case-specific deterministic rule>"
        },
        "observed_oracle": {
          "description": "<observed behavior>",
          "raw_signature": "<exact decisive text>",
          "normalized_signature": "<normalized signature>"
        },
        "exit_status": 0,
        "reproduction_status": "not-reproduced",
        "fidelity": "matched"
      }
    }
  ],
  "final_runtime": {
    "reproduction_status": "not-reproduced",
    "stable_signature": "<absence-of-failure signature>",
    "qualifying_attempts": ["attempt-1"]
  },
  "evidence_status": "valid"
}
```

`qualifying_attempts` must reference existing attempt ids. One attempt is enough
only for a stable non-issue or unresolved result. Before `actionable-bug` or
`input-validation-defect`, add a second independent attempt with the same
decisive stage and normalized signature.

End each attempt log with
`XPU_ALIGNMENT_RESULT=<one-line-JSON>`. Audit by removing that exact prefix,
parsing the remaining JSON, and comparing the parsed value to `runtime_result`;
do not compare serialized object bytes. Hash the unmodified log bytes.

Use `fidelity: matched`, `equivalent-adaptation`, or `invalid`. An invalid
adaptation cannot produce `matched-upstream`. Derive normalized signatures from
raw decisive text with the recorded deterministic rule; remove only unstable
values such as addresses or temporary paths, never semantic error content.

## Runtime statuses

Use exactly one:

| Status | Runtime meaning |
|---|---|
| `matched-upstream` | The intended stage produced the current upstream oracle. |
| `different-failure` | The intended XPU path reached a distinct signature. |
| `not-reproduced` | The intended stage ran and the expected failure was absent. |
| `oracle-not-reached` | An earlier condition prevented the target oracle. |
| `blocked-script-error` | The repro failed before a meaningful oracle. |
| `verification-gap` | Runtime evidence is insufficient or inconsistent. |

`matched-upstream` is not synonymous with a bug. `different-failure` is not
issue evidence until assessment establishes its own reference oracle.

## Runtime proof and audit

- Prove a key input, intermediate, output, or compiler target is XPU. Printing
  availability is not proof.
- Record fallback. An XPU output after CPU fallback does not prove XPU execution.
- Compiler cases require a passing eager baseline and the intended target stage.
  An eager failure before compile is `oracle-not-reached`.
- Preserve every attempt. Use a fresh process and cache after crash or timeout.
- Require two clean qualifying attempts for every potentially actionable defect.

Write `artifacts/audit.json` with `scan_audit_status: pass|fail` and every error;
coverage uses `pending` before that audit exists.
Set `evidence_status=valid` only when file hashes match, cache namespaces are
unique, terminal JSON parses and equals `runtime_result`, device/stage/signature
markers required by the selected status agree, qualifying ids exist, and attempt
count is sufficient.

The scan audit additionally checks collection shard completion, exact raw/source
ledger equality, valid selected-source/case references, legal state transitions,
all unblocked selected cases terminally evidenced and assessed, and
coverage/report count equality. It cannot pass with any blocked case, unresolved
incident, invalid evidence, or pending case.
