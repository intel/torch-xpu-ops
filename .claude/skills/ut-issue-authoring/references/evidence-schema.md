# Evidence directory

Everything a nightly UT run is known to have done. All of it was read off
artifacts by a deterministic collector; none of it is a judgement.

## `run.json`

```jsonc
{
  "run_id": 12345678,
  "created_at": "2026-08-30",
  // Of the case set. Copy it into drafts.json: the filing step refuses drafts
  // written against a different night's failures.
  "digest": "9f2a...",

  // Everything below is keyed by UT job - `basic` or `op_ut` - because a
  // bisect range is per UT job. The baseline commit and tonight's commit have
  // to come from the same one or the compare link spans the wrong commits.
  "job_urls":      { "basic": "https://github.com/.../job/123" },
  "torch":         { "basic": "abc1234..." },
  "torch_xpu_ops": { "basic": "def5678..." },
  "runners":       { "basic": "bmg-test-04" },   // which machine ran it
  "collect_env":   { "basic": "PyTorch version: ..." },

  // Which UT job each category belongs to.
  "category_ut_job": { "op_extended": "basic", "op_ut": "op_ut" },

  // Either of these means the run is not worth filing from, and the filing
  // step will refuse anyway.
  "gates": {
    "build_failed": false,   // nothing downstream can be trusted
    "too_many":     false    // more failures than a night's worth of grouping
  },

  "ut_jobs": {
    "basic": { "runner_name": "bmg-test-04", "new_failures": 312 }
  },

  // The nightly each category was compared against. Rendered into the issue by
  // the filing step; here so you can see how old the comparison is.
  "baselines": {
    "op_extended": { "run_id": 12345000, "created_at": "2026-08-29",
                     "age_in_runs": 1, "ut_job": "basic", "job_url": "...",
                     "torch": "abc1234", "torch_xpu_ops": "def5678" }
  },

  "report": {
    "categories": [{"category": "op_ut", "state": "complete",
                    "actual": 178102, "expected": 178548}],
    "skipped_ut_jobs": [],
    "vanished_cases": [],
    "baseline_walk": []
  }
}
```

A category with `state` other than `complete` produced no filable failures: its
cases were dropped during collection, because a truncated run is a statement
about the machine rather than about the code.

`report.vanished_cases` is what the baseline ran and this run does not have at
all, per module - a module that stopped importing, a skip pattern wide enough
to empty a file, or a test removed or renamed in stock pytorch. These did not
fail; they did not run.

```jsonc
{
  "category": "op_ut", "module": "test_ops_xpu",
  "cases": 3,              // baseline names absent tonight
  "baseline_passed": 412,  // of those, how many the baseline passed
  "module_gone": false,    // true when the module produced nothing at all
  "baseline_run": 12345000,
  "lost_names":   ["test_foo_xpu_float32"],        // up to 20 of each
  "gained_names": ["test_foo_new_xpu_float32"]
}
```

A failing case in such a module is classified `unknown` rather than
`new_case_failure`, because "absent from the baseline" stops meaning "new
test" once the module's names have moved. Whether a gained name is a lost one
renamed is a judgement about two strings, so the collector does not make it -
see [SKILL.md](../SKILL.md).

`report.baseline_walk` records every nightly the collector looked at per
category and why it was or was not usable. Context for a category that ended up
with no baseline; nothing to decide.

## `cases.json`

One record per new failure. This is the set your grouping must cover exactly.

```jsonc
{
  "count": 312,
  "counts_by_cls": { "regression": 40, "new_case_failure": 272 },
  "cases": [
    {
      // The muting line. Copy this string; never build one.
      "line": "op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32",
      "category": "op_extended",
      "ut_job": "basic",
      "class_name": "test_ops_xpu.TestFooXPU",
      "test_name": "test_bar_xpu_float32",
      "test_file": "test_ops_xpu.py",
      "module": "test_ops_xpu",
      // True for a test *file* that would not import. Such a row stands in for
      // every case in the file and never shares a group with a real case.
      "is_collection_error": false,
      "message": "RuntimeError: ...",
      // Exact set membership against the baseline. Not yours to question,
      // override, or restate as your own finding.
      "cls": "regression",
      "cls_reason": "passed in the baseline",
      "runner_name": "bmg-test-04",
      "has_traceback": true
    }
  ],
  // One per whole-module row: what that file used to run.
  "collection_context": [
    {"line": "op_ut,,test_foo_xpu", "category": "op_ut", "module": "test_foo_xpu",
     "state": "was passing", "baseline_passed": 412, "baseline_run": 12345000}
  ],
  // Per category, for the filing step's Reproduce section.
  "reproduce": {
    "op_extended": {"file_path": "cd pytorch/third_party/torch-xpu-ops/test/xpu/extended",
                    "command_template": "pytest -sv failed_case"}
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

## `tracebacks.json`

```jsonc
{ "by_case": { "op_ut,test_foo_xpu.TestFooXPU,test_a": ["Traceback ...", "..."] } }
```

Full `<failure>` text from the JUnit XML, split into lines, for one case per
distinct (test file, exact message). The `message` field in `cases.json` is
only the last exception line. The filing step picks the traceback for each
issue; read these to understand what a group is, not to copy them anywhere.

## `drafts.json`

What you write. The schema is in [SKILL.md](../SKILL.md). The filing step
rejects a draft whose `cases` contain a line absent from `cases.json`, whose
cases do not all share one `cls`, or which mixes whole-module rows with
ordinary cases - so those are checks, not suggestions.
