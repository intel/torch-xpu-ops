# evidence.json

One file, read off the run's artifacts by a deterministic collector. None of it
is a judgement.

```jsonc
{
  "run": {
    "run_id": 12345678,
    "created_at": "2026-08-30",
    // Of the case set. Copy it into drafts.json: the filing step refuses
    // drafts written against a different night's failures.
    "digest": "9f2a...",

    // Keyed by UT job - `basic` or `op_ut` - because a bisect range is per
    // job: the baseline commit and tonight's must come from the same one.
    "job_urls":      { "basic": "https://github.com/.../job/123" },
    "torch":         { "basic": "abc1234..." },
    "torch_xpu_ops": { "basic": "def5678..." },
    "runners":       { "basic": "bmg-test-04" },   // which machine ran it
    "collect_env":   { "basic": "PyTorch version: ..." },
    "category_ut_job": { "op_extended": "basic", "op_ut": "op_ut" },

    // Either one means the run is not worth filing from, and the filing step
    // refuses anyway.
    "gates": { "build_failed": false, "too_many": false },
    "ut_jobs": { "basic": { "runner_name": "bmg-test-04", "new_failures": 312 } },

    // What each category was compared against, and how old that comparison is.
    "baselines": {
      "op_extended": { "run_id": 12345000, "created_at": "2026-08-29",
                       "age_in_runs": 1, "ut_job": "basic", "job_url": "...",
                       "torch": "abc1234", "torch_xpu_ops": "def5678" }
    },

    "report": {
      // Anything but `complete` means the category was truncated and its
      // failures were dropped during collection: a statement about the machine
      // rather than about the code.
      "categories": [{ "category": "op_ut", "state": "complete",
                       "actual": 178102, "expected": 178548 }],
      "skipped_ut_jobs": [],

      // Per module, what the baseline ran and this run does not have at all.
      // These did not fail; they did not run.
      //   module_gone  the file produced no cases at all - it stopped
      //                importing, or a skip pattern emptied it.
      //   removed      names went and none arrived. Nothing here is a rename.
      //   moved        names went and names arrived. Only this one makes
      //                "absent from the baseline" mean something else.
      "vanished_cases": [{
        "category": "op_ut", "module": "test_ops_xpu", "kind": "moved",
        "cases": 3,              // baseline names absent tonight
        "baseline_passed": 412,  // of those, how many the baseline passed
        "baseline_run": 12345000,
        "lost_names":   ["test_foo_xpu_float32"],       // up to 20 of each
        "gained_names": ["test_foo_new_xpu_float32"]
      }],

      // Every nightly the collector looked at per category, and why it was or
      // was not usable. Context for a category with no baseline; nothing to
      // act on.
      "baseline_walk": []
    }
  },

  // One record per new failure. This is the set your grouping must cover
  // exactly.
  "count": 312,
  "counts_by_cls": { "regression": 40, "new_case_failure": 272 },
  "cases": [{
    // The muting line. Copy this string; never build one.
    "line": "op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32",
    "category": "op_extended",
    "ut_job": "basic",
    "class_name": "test_ops_xpu.TestFooXPU",
    "test_name": "test_bar_xpu_float32",
    "test_file": "test_ops_xpu.py",
    "module": "test_ops_xpu",
    // A test *file* that would not import. Such a row stands in for every case
    // in the file and never shares a group with a real case.
    "is_collection_error": false,
    "message": "RuntimeError: ...",   // the last line of the traceback
    // Exact set membership against the baseline. Not yours to question,
    // override, or restate as your own finding.
    "cls": "regression",
    "cls_reason": "passed in the baseline",
    "runner_name": "bmg-test-04"
  }],

  // One per whole-module row: what that file used to run.
  "collection_context": [
    { "line": "op_ut,,test_foo_xpu", "category": "op_ut",
      "module": "test_foo_xpu", "state": "was passing",
      "baseline_passed": 412, "baseline_run": 12345000 }
  ],

  // Per category, for the filing step's Reproduce section.
  "reproduce": {
    "op_extended": { "file_path": "cd pytorch/third_party/torch-xpu-ops/test/xpu/extended",
                     "command_template": "pytest -sv failed_case" }
  },

  // Full <failure> text from the JUnit XML, keyed by case line, one case per
  // distinct (test file, exact message). Read these to understand what a group
  // is; to choose which one an issue shows, name the case in the draft's
  // `error_case` rather than copying the text. Only a case with an entry here
  // may be named. Long ones are already cut to their two ends, which is where
  // a failure says what went wrong - so what is here is what to quote, and
  // there is nothing further to trim.
  "tracebacks": {
    "op_ut,test_foo_xpu.TestFooXPU,test_a": ["Traceback ...", "..."]
  }
}
```

`cls` is one of:

| `cls` | Means | Label the issue gets |
|---|---|---|
| `regression` | passed in the baseline, fails now | `regression` |
| `new_case_failure` | absent from the baseline, or present but skipped there | `new_case_failure` |
| `persistent` | already failing in the baseline; onset predates it | none |
| `unknown` | no usable baseline, or the module's names moved upstream | none |

A failing case in a `moved` module is `unknown` rather than `new_case_failure`.
Whether a gained name is a lost one renamed is a judgement about two strings,
so the collector does not make it - see [SKILL.md](../SKILL.md).
