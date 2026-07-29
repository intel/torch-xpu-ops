---
name: analyze-ut-results
description: Run XPU unit tests, collect JUnit XML results, group failures by error pattern, and cross-reference known pytorch/pytorch and intel/torch-xpu-ops issues. Returns categorized failures with root-cause hypotheses and a fixable-vs-infrastructure verdict for downstream fixing or issue submission, plus a top-level passed verdict when all cases pass.
---

# `analyze-ut-results`

## Purpose

Given one or more XPU test files (or existing JUnit XML), run them, aggregate
results, group failures by error pattern, and classify each group as a
**fixable test-code bug** or an **infrastructure/backend bug**. This is the
analysis stage that feeds `fix-ut-test-code` and `create-xpu-issue`.

## Input

Accepts any one of these forms:

1. **Test targets** (preferred for targeted analysis) — a set of
   `{test_file, test_class, test_name}` entries, supplied as EITHER a JSON list
   OR a CSV file.

   JSON list:
   ```json
   [
     {"test_file": "test/xpu/dynamo/test_streams_xpu.py",
      "test_class": "TestStreamsXPU",
      "test_name": "test_record_stream_xpu"},
     {"test_file": "test/xpu/test_ops_xpu.py",
      "test_class": "TestCommonXPU",
      "test_name": "test_out_warning_xpu_float32"}
   ]
   ```

   CSV file — a header row naming the columns, then one row per test. Column
   order does not matter; the three columns `test_file`, `test_class`,
   `test_name` are required (extra columns are ignored):
   ```csv
   test_file,test_class,test_name
   test/xpu/dynamo/test_streams_xpu.py,TestStreamsXPU,test_record_stream_xpu
   test/xpu/test_ops_xpu.py,TestCommonXPU,test_out_warning_xpu_float32
   ```
   Parse the CSV into the same list of `{test_file, test_class, test_name}`
   entries, then proceed identically to the JSON form. Read it with
   `python3 -c "import csv,sys; ..."` (do not hand-split lines — fields may be
   quoted). Run only these specific tests (see Workflow step 1). Group by
   `test_file` so each file is one pytest invocation.
2. **Test files** — one or more test files (run the whole file), including:
   - `test/xpu/**/*_xpu.py` (torch-xpu-ops wrapped XPU tests)
   - files under `pytorch/test/**` (upstream PyTorch tests)
3. **Existing XML results** — a directory (or list) of already-produced
   `test_*.xml` JUnit files. In this case **do NOT run any test locally**: the
   XML already contains the outcomes and tracebacks. Skip Workflow step 1
   entirely and start at step 2 (grouping). The XPU env check is also
   unnecessary in this mode.

Optional `conda_env` and `pytorch_root` are provided by the calling agent
(defaults `pytorch_opencode_env` and `$HOME/daisy_pytorch`). Use them as given.

When targets are given as `{test_file, test_class, test_name}`, carry those
identifiers through to the output `tests` list so callers can key results back
to the exact rows.

## Environment

The calling agent (ut-follow-up / classify-ut) has already established the
conda env and pytorch checkout. **Do NOT run `setup_env.sh` or create/activate
any environment here.** Just use the `conda_env` and `pytorch_root` passed in.

When the input runs tests locally (forms 1 and 2), sanity-check the provided
env before running (do not attempt to fix or bootstrap it):

```bash
conda run -n "${conda_env}" python3 -c "import torch; assert torch.xpu.is_available(), 'XPU unavailable'"
```

If XPU is unavailable, stop and report the broken env to the caller; do not
proceed. For existing XML input (form 3) skip this check entirely.

## Workflow

### 1. Run tests and collect results

**Skip this step entirely for form 3 (existing XML results)** — go straight to
step 2 and read the outcomes/tracebacks from the provided `test_*.xml` files.

For forms 1 and 2, run pytest in the provided env/checkout to produce the XML:

```bash
cd "${pytorch_root}/third_party/torch-xpu-ops/test/xpu"
```

Before each pytest invocation, resolve the run root from `test_file`:

- If `test_file` ends with `_xpu.py` -> run from
  `${pytorch_root}/third_party/torch-xpu-ops/test/xpu`.
- If `test_file` does **not** end with `_xpu.py` -> run from `${pytorch_root}`
  (this covers files under `pytorch/test/**`).

Path handling rule:
- For `_xpu.py` files, use the path relative to
  `${pytorch_root}/third_party/torch-xpu-ops/test/xpu`.
- For non-`_xpu.py` files, use the path relative to `${pytorch_root}`
  (typically `test/...`).

**Targeted mode** (input is a list of `{test_file, test_class, test_name}`):
group the targets by `test_file`, then run one pytest invocation per file that
selects only the listed tests. Prefer exact node ids; fall back to `-k` when a
class/name is ambiguous.

```bash
# One _xpu file, selected tests as node ids (run root: third_party/torch-xpu-ops/test/xpu):
pytest -v --timeout=120 --junit-xml=test_streams_xpu.xml \
  "dynamo/test_streams_xpu.py::TestStreamsXPU::test_record_stream_xpu" \
  "dynamo/test_streams_xpu.py::TestStreamsXPU::test_wait_stream_xpu"

# One upstream pytorch/test file (no _xpu postfix; run root: ${pytorch_root}):
pytest -v --timeout=120 --junit-xml=test_view_ops.xml \
  "test/test_view_ops.py::TestViewOps::test_view_copy_xpu"

# Or select by name/class within a file when node ids are awkward:
pytest -v --timeout=120 --junit-xml=test_ops_xpu.xml test_ops_xpu.py \
  -k "TestCommonXPU and test_out_warning_xpu_float32"
```

**Whole-file mode** (input is one or more test files):

```bash
pytest -v --timeout=120 --junit-xml=test_<name>_xpu.xml dynamo/test_<name>_xpu.py
```

For non-`_xpu.py` files, run from `${pytorch_root}` with the `test/...` path,
for example:

```bash
pytest -v --timeout=120 --junit-xml=test_<name>.xml test/<subdir>/test_<name>.py
```

Run each test file in a separate pytest invocation for isolation (avoid state
leakage). Then aggregate the produced XML.

### 2. Group failures by error pattern

Aggregate the XML results — the ones you just produced (forms 1/2) or the ones
supplied as input (form 3) — pointing `-i` at the relevant `*.xml`:

```bash
python ../../.github/scripts/check-ut.py -i *.xml
```

Result categories:
- **Passed** — executed and passed.
- **Failed** — executed but assertion/exception occurred.
- **Skipped** — skipped by decorator or `skipIf`.
- **xFailed** — expected failure that occurred (acceptable; not a defect).

For each failure, extract the terminal exception message and stack-trace
origin. Cluster failures whose messages share the same root signature into one
group. For each group record:

| Field | Meaning |
|---|---|
| `signature` | The distinguishing substring of the error (verbatim) |
| `test_count` | Number of failing tests in the group |
| `tests` | List of `{test_file, test_class, test_name}` for the failing tests |
| `category` | `test-code`, `infrastructure`, `backend`, or `pytorch-codebase` |
| `root_cause` | One-to-three sentence hypothesis |

When targets were supplied as `{test_file, test_class, test_name}`, reuse those
exact identifiers in `tests` so results map back 1:1 to the caller's rows.

Do NOT hardcode known error catalogs — derive categories from the actual
observed messages each run.

### 3. Cross-reference known issues

For each group, search both repos before concluding it is a new bug:

```bash
# pytorch/pytorch open issues for the error keyword
curl -s "https://api.github.com/search/issues?q=repo:pytorch/pytorch+<keyword>+is:issue+state:open&per_page=10" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Found:', d['total_count']); [print(f\"#{i['number']}|{i['title'][:80]}\") for i in d['items'][:5]]"

# pytorch/pytorch DISABLED tests matching the test name
curl -s "https://api.github.com/search/issues?q=repo:pytorch/pytorch+DISABLED+<test_name>+is:issue&per_page=10"

# intel/torch-xpu-ops existing issues
gh search issues --repo intel/torch-xpu-ops "<keyword>" is:issue --limit 10
```

Run these searches in parallel where possible. For each match, verify with
`gh issue view <n> --repo <repo> --json title,state,labels,url` and record the
verified URL, state, and labels.

### 4. Verdict per group

Apply this decision in order:

```
Is the failure reproducible from TEST CODE only (import path, CUDA->XPU API,
missing skip guard, syntax)?
- YES -> category = test-code      -> downstream: fix-ut-test-code
- NO
  - Matches a known infra/backend issue -> attach issue URL
  - No match -> category = backend|infrastructure|pytorch-codebase (new)
              -> downstream: create-xpu-issue
```

## Output

Return JSON:

```json
{
  "verdict": "passed|has-failures",
  "groups": [
    {
      "signature": "...",
      "test_count": 2,
      "tests": [
        {"test_file": "test/xpu/dynamo/test_streams_xpu.py",
         "test_class": "TestStreamsXPU",
         "test_name": "test_record_stream_xpu"}
      ],
      "category": "test-code|infrastructure|backend|pytorch-codebase",
      "root_cause": "...",
      "known_issues": [{"url": "...", "state": "OPEN|CLOSED", "labels": ["..."]}],
      "verdict": "fix-ut-test-code|create-xpu-issue"
    }
  ]
}
```

Top-level verdict rules:
- `"passed"`: all executed cases are passed/skipped/xfailed and there are no
  failure groups to fix/submit.
- `"has-failures"`: at least one failure group exists in `groups`.

## Constraints

- **Tools**: `bash` (`pytest`, `curl`, `gh`, python), `read`, `grep`. No web tools.
- Parse CSV input with Python's `csv` module (`import csv`), not manual string
  splitting — fields may be quoted or contain commas.
- **Existing XML input runs NO local tests**: for form 3, skip Workflow step 1
  and the env activation; read outcomes/tracebacks straight from the XML.
- **Run-root selection is mandatory**: choose pytest working directory from
  `test_file` postfix — `_xpu.py` runs under
  `${pytorch_root}/third_party/torch-xpu-ops/test/xpu`; non-`_xpu.py` runs
  under `${pytorch_root}`.
- `gh search issues` MUST include `is:issue` to exclude PRs.
- Timeout each test at 120s. Max 2-3 concurrent pytest/agent runs.
- Never invent issue numbers; only cite issues verified via `gh issue view`.
- Focus on XPU-specific failures; always check pytorch/pytorch DISABLED first.
