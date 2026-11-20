# evidence.json

Written by `e2e_accuracy_collect_evidence.py collect`; `e2e_accuracy_triage.py
confirm` adds `confirm.<sku>`; `e2e_accuracy_collect_evidence.py merge` adds
`groups[].confirm_result`.

```jsonc
{
  "run": {
    "run_id": 123, "test_type": "build-nightly", "driver": "rolling",
    "history": [122, 121],          // earlier scheduled build runs compared with, newest first
    "runners": {"timm_models": "<runner name>"},
    "env": {"versions": {"torch": "<sha>", "torch_xpu_ops": "<sha>", "torchbench": "<sha>"},
            "packages": {"transformers": "4.x", "timm": "1.x", "numpy": "1.26.4"},
            "collect_env": "<collect_env.py output>"},
    "integrity": ["torchbench: no accuracy CSV"]   // suites the gate could not judge
  },
  "digest": "<16 hex>",                            // copy into drafts.json
  "groups": [{
    "id": "g1",
    "signature": "fail_accuracy|accuracy failed for key name stages.N...",
    "category": "accuracy",  // accuracy crash eager_fail eager_nondeterministic oom timeout env machine new_pass
    "skus": ["pvc-1100"],    // cards the rows failed on tonight; "unknown" if not logged
    "reps": ["<key>"],       // rows the confirm stage reran
    "confirm_skip": "",      // non-empty: why the group was not confirmed
    "confirm_result": {      // only for confirmed groups
      "conclusion": "sku_specific",
      "reps": {"<key>": {"conclusion": "sku_specific",
                         "runs": {"pvc-1100": ["fail_accuracy", "fail_accuracy", "fail_accuracy"], "pvc-1550": ["pass", "pass"]},
                         "public_wheel": "fail_accuracy", "public_wheel_reproduces": true, "public_versions": {"torch": "..."}}}
    },
    "rows": [{
      "key": "timm_models,bfloat16,training,convnextv2_nano.fcmae_ft_in22k_in1k",  // suite,dtype,mode,model
      "suite": "...", "dtype": "...", "mode": "...", "model": "...",
      "value": "fail_accuracy",       // tonight's result; the value an issue mutes
      "reference": "pass",            // ci_expected_accuracy/<driver>
      "gate": "real_failed",          // real_failed, timeout or new_pass
      "message": "Accuracy failed for key name ...",  // last error line of the log
      "log_excerpt": ["..."],         // from the first traceback, or the last 40 lines
      "log": "<suite>/.../logs-<model>-worker1-card1.log",
      "cmd": "python benchmarks/dynamo/timm_models.py --accuracy ...",
      "card": "1", "device": "Intel(R) Data Center GPU Max 1100", "sku": "pvc-1100",
      "batch_size": "8", "elapsed": "42.3",
      "other_results": {"training float32": "pass"},  // same model, other mode/dtype tonight
      "issues": [5395],               // open `test: e2e` issues listing this key
      // Over the earlier runs that ran this row (run.history, newest first; a nightly skips p2):
      // first_seen = oldest run of the unbroken streak with this value, this run if none;
      // last_good = newest run where it passed, null if none in run.history
      "first_seen": 123, "last_good": 122
    }]
  }],
  "confirm_plan": [{"group": "g1", "reps": ["<key>"]}],
  "confirm": {"pvc-1100": {"<key>": {"runs": [{"value": "...", "card": "0", "device": "...", "sku": "...",
                                               "elapsed": "...", "tail": ["..."]}],
                                     "public_wheel": {"value": "...", "versions": {"torch": "..."}}}}},
  "issues": [{"issue_number": 5395, "issue_title": "...", "labels": ["..."], "table_rows": [["..."]], "auto": false}]
}
```

Confirm conclusions, with A = the card that failed tonight and B = the other:

| conclusion | meaning |
|---|---|
| `sku_independent` | A and B fail with tonight's value every time |
| `sku_specific` | A fails every time, B passes every time |
| `sku_different_value` | both fail every time, with different values |
| `random` | A or B both passes and fails across runs |
| `value_varies` | every run fails, but with more than one value (e.g. `out of memory` and `core_dump`) |
| `not_reproduced` | every rerun passed |
| `single_sku` | A fails every time, B has no result |
| `unknown_sku` / `unconfirmed` | tonight's card is unknown / no rerun produced a result |
| `mixed` | the group's representatives disagree |
