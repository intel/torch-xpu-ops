# Evidence directory

Everything a nightly UT run is known to have done. All of it was read off
artifacts by a deterministic collector; none of it is a judgement.

## `run.json`

```jsonc
{
  "run_id": 12345678,
  "created_at": "2026-08-30",

  // Everything below is keyed by test leg - `basic` or `op_ut` - because a
  // bisect range is per leg. The baseline commit and tonight's commit have to
  // come from the same leg or the compare link spans the wrong commits.
  "job_urls":      { "basic": "https://github.com/.../job/123" },
  "torch":         { "basic": "abc1234..." },
  "torch_xpu_ops": { "basic": "def5678..." },
  "runners":       { "basic": "bmg-test-04" },   // which machine ran the leg
  "collect_env":   { "basic": "PyTorch version: ..." },

  // Which leg each category belongs to.
  "category_leg": { "op_extended": "basic", "op_ut": "op_ut" },

  "gates": {
    "build_failed": false,   // nothing downstream can be trusted
    "abort":        false,   // more failures than the run-level abort threshold
    "oversized":    false    // too many failures to group; file nothing
  },

  "legs": {
    "basic": {
      "runner_name": "bmg-test-04",
      "new_failures": 312,
      // Failures whose message matches the infra denylist. A share, not a
      // verdict: the filing step decides what a share means.
      "infra_pattern_cases": ["op_extended,test_ops_xpu.TestFooXPU,test_a"],
      "infra_pattern_ratio": 0.03
    }
  },

  // Category health and anything the collector skipped, carried through so the
  // final report is complete. Useful context, nothing to decide.
  // Thresholds the filing rules refer to, stated here so there is one source
  // of truth for them.
  "limits": { "max_issues_per_run": 15, "max_cases_per_issue": 400,
              "safe_body_chars": 60000, "hard_body_chars": 65536,
              "infra_max_test_files": 5, "infra_leg_share": 0.3,
              "infra_leg_min_cases": 10 },

  // Resolved label lists, keyed `<cls>|<leg>`, the runner being per leg. Every
  // case in cases.json also carries its own; both are copied, never derived.
  "labels": { "regression|op_ut": ["skipped", "skipped_bmg", "regression"],
              "persistent|op_ut": ["skipped", "skipped_bmg"],
              "unknown|basic":    ["skipped", "skipped_bmg"] },

  "marker_template": "<!-- ut-auto-issue:{version}:run={run_id}:part={part}/{parts} -->",
  "marker_version": "v1",

  "report": {
    "categories": [{"category": "op_ut", "state": "complete",
                    "actual": 178102, "expected": 178548}],
    "skipped_legs": [],
    "vanished_modules": [],
    "baseline_walk": []
  }
}
```

A category with `state` other than `complete` produced no filable failures;
its cases are already absent from `cases.json`.

## `cases.json`

Every new failure in the run. This is the set to group, and the set your
grouping must cover exactly.

```jsonc
{
  "count": 312,
  "cases": [
    {
      "line": "op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32",
      "category": "op_extended",
      "leg": "basic",
      "class_name": "test_ops_xpu.TestFooXPU",
      "test_name": "test_bar_xpu_float32",
      "test_file": "test_ops_xpu.py",
      "module": "test_ops_xpu",
      "is_collection_error": false,
      "message": "RuntimeError: index 5 is out of bounds for dimension 0 ...",
      "cls": "regression",
      "labels": ["skipped", "skipped_bmg", "regression"],
      "runner_name": "bmg-test-04",
      "has_traceback": true
    }
  ],
  "reproduce": {
    "op_extended": { "file_path": "cd pytorch/third_party/torch-xpu-ops/test/xpu/extended",
                     "command_template": "pytest -sv failed_case" }
  }
}
```

`line` is the exact string that ends up in an issue's `Cases:` block and is
what mutes the test. Copy it; never edit it.

`is_collection_error` true means the row is a test *file* that would not
import, not a test that failed. `class_name` is empty for these, and
`test_name` holds the module path. One such row stands for every case in that
file, all of which stopped running rather than failing.

`cls` is the comparison against the previous healthy nightly for that
category, already done:

| Value | Meaning |
|---|---|
| `regression` | passed in the baseline, fails now |
| `new_case_failure` | absent from the baseline, or present but skipped |
| `persistent` | already failing in the baseline; the onset predates it |
| `unknown` | no usable baseline for that category |

`labels` is the issue's label list, already resolved from `cls` and
`runner_name`. Copy it. `skipped` is on every case; `skipped_bmg` only on a
BMG runner; `persistent` and `unknown` deliberately carry no classification
label.

`reproduce` gives the directory and command shape for each category, recorded
by the UT job itself. Use it to build `reproduce_command`.

## `classifications.json`

The same classifications with the working behind them.

```jsonc
{
  "by_case": { "op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32": "regression" },
  "counts":  { "regression": 180, "new_case_failure": 96, "persistent": 30, "unknown": 6 },

  // For a new_case_failure: `absent` means upstream added the case, `skipped`
  // means it existed and was being skipped and now runs. Different stories.
  "new_case_reason": { "op_extended,test_ops_xpu.TestFooXPU,test_baz": "absent" },

  // One entry per whole-module row: what that file used to run.
  "collection_context": [
    { "line": "op_ut,,test_sparse_xpu", "category": "op_ut",
      "module": "test_sparse_xpu",
      "state": "was passing",        // was passing | known, none passing
                                     // | new test file | no baseline
      "baseline_passed": 412, "baseline_run": 12345000 }
  ],

  // The nightly each category was compared against.
  "baselines": {
    "op_extended": { "run_id": 12345000, "created_at": "2026-08-29",
                     "age_in_runs": 1, "leg": "basic", "job_url": "...",
                     "torch": "abc1234", "torch_xpu_ops": "def5678" }
  }
}
```

`age_in_runs` above 1 means intervening nightlies did not complete that
category, so the commit range is wider than one night.

## `tracebacks.json`

```jsonc
{ "by_case": { "op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32":
               ["Traceback (most recent call last):", "  File ...", "..."] } }
```

A sample, not the whole run: one case per distinct (test file, exact message),
capped. `has_traceback` on a case record says whether it is in here. Paste
these lines into the ErrorLog block as they are; never write a traceback of
your own, and never quote one for a case that is not in the issue without
saying which case it came from.

## `blocks.json`

Markdown already rendered from the facts, to be pasted rather than rebuilt.
The bisect range is the reason this file exists: the baseline commit and
tonight's commit have to come from the same test leg, nothing in the rendered
link says which leg it came from, and a range spanning the wrong commits sends
a reader hunting through the wrong history.

```jsonc
{
  "baseline_table_header": ["| Category | | Run | Date | torch | torch-xpu-ops |",
                            "|---|---|---|---|---|---|"],
  "baseline_table_rows": { "op_extended": ["| op_extended | Last good | ...",
                                           "| op_extended | First seen bad | ..."] },
  "compare_links":     { "op_extended": "Changes in range (op_extended): [pytorch](...)" },
  "baseline_staleness":{ "op_extended": "Note: the last healthy ... nightly was 3 runs back ..." },
  "collection_error":  { "op_ut,,test_sparse_xpu": {
                           "table_row": "| `test_sparse_xpu` | op_ut | was passing | 412 |",
                           "verdict": "Classified as a **regression**: ...",
                           "baseline_passed": 412 } }
}
```

`baseline_table_rows` and `compare_links` only make sense in an issue whose
classification is `regression`; for the other three the baseline run is not a
last-known-good for those cases.

## `digest.json`

```jsonc
{ "all_cases": "<sha256 of the sorted case lines>", "count": 312 }
```

A checksum of the case set, for the audit that runs after filing.
