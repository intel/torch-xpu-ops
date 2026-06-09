# check_known_issue

Search both `intel/torch-xpu-ops` and `pytorch/pytorch` for known issues related to a test case. Returns matched issues with issue state, labels, and match evidence for downstream classification.

Derived from the **Known Issue** decision axis in `classify_ut/RULES.md` (Confidence Rubric & Need-Human-Check Rule, Deep Analysis Requirements, Dynamic-Skip Rule).

## Inputs

| Parameter | Description | Default |
|-----------|-------------|---------|
| `test_file` | Test file path relative to `$PYTORCH_SRC` (e.g. `test/dynamo/test_streams.py`) | **Required** |
| `class_name` | Test class name (e.g. `TestStreamsCUDA`) | **Required** |
| `test_name` | Test method name (e.g. `test_local_stream_enter_exit`) | **Required** |
| `error_message` | Error message from test failure. Used to extract operator/API names for issue search. | `None` |
| `test_code_block` | Source code of the test method body. Used to extract API calls and operator references. | `None` |
| `traceback` | Full traceback from test failure. Stack frames can identify the failing operator or CUDA-specific path. | `None` |
| `PYTORCH_SRC` | PyTorch source checkout path | `$HOME/upstream/pytorch` |

## Output

```python
{
    "has_known_issue": bool,              # True if at least one matching issue found
    "matches": [
        {
            "issue_url": str,             # Full issue URL
            "repo": str,                  # "intel/torch-xpu-ops" or "pytorch/pytorch"
            "issue_number": int,          # Issue number
            "title": str,                 # Issue title
            "state": str,                 # "OPEN" or "CLOSED"
            "labels": [str],              # Issue labels (e.g. ["skipped", "bug"])
            "match_type": str,            # How this issue matched (see table below)
            "match_evidence": str,        # What exactly matched (e.g. "test_foo in issue body")
            "relevance": str,             # "HIGH", "MEDIUM", or "LOW"
        },
    ],
    "search_summary": {
        "intel/torch-xpu-ops": {
            "searches_run": int,          # Number of searches performed
            "total_results": int,         # Total unique results across all searches
            "top_keywords": [str],        # Keywords that produced the most relevant results
        },
        "pytorch/pytorch": {
            "searches_run": int,
            "total_results": int,
            "top_keywords": [str],
        },
    },
    "discovery_steps": [
        {
            "step": int,
            "action": str,
            "keyword": str,
            "repo": str,
            "results": int,
            "key_match": str or None,
        },
    ],
}
```

### Match types

| `match_type` | Meaning | Relevance |
|---|---|---|
| `exact_test_name` | Issue title or body names the exact test method | HIGH |
| `class_name` | Issue references the test class | HIGH |
| `operator_api_match` | Issue tracks an operator/API found in test code, error, or traceback | HIGH |
| `error_message_match` | Error message keywords match issue description | MEDIUM |
| `file_path_match` | Issue references the test file | MEDIUM |
| `keyword_match` | Generic keyword match (test name substring, related term) | LOW |
| `label_inherited` | Issue was found via a labeled issue's related issue references | LOW |

## Execution

This skill is executed by an `explore` subagent. The orchestrator delegates the entire search as one task:

```python
task(
    subagent_type="explore",
    load_skills=["check_known_issue"],
    description="Search known issues for test case",
    prompt="Search known issues for <test_name> in <class_name> (<device>). Test file: <test_file>. Error message (if any): <error_message>. Test code block: <test_code_block>. Traceback: <traceback>. Return matched issues with state, labels, and match evidence."
)
```

The `explore` subagent is well-suited for running parallel `gh search issues` queries, extracting keywords from test code, and verifying issue state via `gh issue view`. It follows the Search Flow below, using its own tools (`bash` with `gh` CLI, `grep` for extracting keywords) and running independent searches in parallel.

## Search Flow

To minimize execution turns and lower latency, you MUST run independent searches concurrently using parallel `gh search issues` calls. Both repos are searched independently.

### Step 1: Extract search keywords

Extract keywords from all available inputs. Independent extractions run in parallel:

```
From test_name:      <test_name> stripped of device suffix (_cuda, _xpu, _cpu)
From class_name:     <class_name> stripped of device suffix (CUDA, XPU)
From test_file:      basename of test file (e.g. test_streams.py)
From error_message:  operator names (aten::*), API names (torch.*), key error terms
From test_code_block: API calls (torch.*, aten::*), decorator names, special function names
From traceback:      operator names in stack frames (aten::*), file paths (torch/cuda/...)
```

**Priority order for keyword generation:**

| Priority | Keyword Source | Example |
|----------|---------------|---------|
| 1 | Operator from error/traceback | `aten::_cudnn_rnn` |
| 2 | API from error/test code | `torch.cuda.jiterator` |
| 3 | Exact test name | `test_local_stream_enter_exit` |
| 4 | Test name without suffix | `test_local_stream_enter_exit_xpu` and `test_local_stream_enter_exit` |
| 5 | Class name | `TestStreams` |
| 6 | File name | `test_streams` |
| 7 | Error message keywords | `CUDA error`, `invalid argument` |

### Step 2: Search `intel/torch-xpu-ops` (parallel)

Run searches in parallel with keyword variations. The `gh search issues` command targets issues only:

```bash
# Run ALL of these in parallel:
gh search issues "<test_name> xpu" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &
gh search issues "<test_name>" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &
gh search issues "<class_name>" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &
gh search issues "<operator_or_api>" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &
gh search issues "<error_keyword>" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &
wait
```

Collect all unique results. Deduplicate by issue number.

### Step 3: Search `pytorch/pytorch` (parallel)

Same pattern for the upstream repo. These searches are independent of Step 2 and run in parallel with them:

```bash
# Run ALL of these in parallel:
gh search issues "<test_name> xpu" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url &
gh search issues "<test_name>" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url &
gh search issues "<class_name>" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url &
gh search issues "<operator_or_api>" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url &
gh search issues "<error_keyword>" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url &
wait
```

Collect all unique results. Deduplicate by issue number.

### Step 4: Verify issue state and labels (parallel)

For every unique issue matched, run `gh issue view` to fetch authoritative state, labels, and body:

```bash
# Run for ALL unique issues in parallel:
gh issue view <number> --repo=<repo> --json=state,labels,title,body,closedAt,url &
```

This is critical because `gh search issues` results can be stale. The `gh issue view` call confirms:
- Current state (OPEN vs CLOSED)
- Full label set (especially `skipped`, `not_target`, `wontfix`)
- When it was closed (if CLOSED)
- Body text for deeper evidence matching

### Step 5: Classify each match

For each matched issue, determine its `match_type` and `relevance`:

| Condition | match_type | relevance |
|-----------|------------|-----------|
| Issue title contains exact test name | `exact_test_name` | HIGH |
| Issue body mentions exact test name | `exact_test_name` | HIGH |
| Issue references class name | `class_name` | HIGH |
| Issue tracks an operator/API that appears in test code, error, or traceback | `operator_api_match` | HIGH |
| Error message keywords appear in issue title/body | `error_message_match` | MEDIUM |
| Issue references test file path | `file_path_match` | MEDIUM |
| Keyword matched but no direct test/API reference | `keyword_match` | LOW |

Assign the highest relevance among all match reasons for each issue.

### Step 6: Synthesize results

Combine all matches into the output structure. For `has_known_issue`:

- `True` if at least one match with `relevance` of `HIGH` or `MEDIUM` is found in either repo
- `False` if only `LOW` relevance matches or no matches at all

Populate `search_summary` with statistics from each repo.

## Label Interpretation

Use these to interpret what an issue's labels mean for downstream classification.

| Label | Meaning for Classification |
|-------|---------------------------|
| `not_target` | Issue is explicitly a non-target feature → `Not applicable` |
| `wontfix` | Issue will not be fixed → `Not applicable` |
| `skipped` | Test is dynamically skipped on XPU → Follow **Dynamic-Skip Rule** (local verify required) |
| `bug` | XPU implementation bug → `Failures (xpu broken)` |
| `feature` | Missing XPU feature → `Feature gap` |
| `enhancement` | Missing XPU feature / improvement → `Feature gap` or `To be enabled` |
| `question` | Not a tracked issue → ignore for classification |
| `triaged` | Issue has been triaged → respect its other labels |
| No relevant labels | Depends on issue state: CLOSED → `To be enabled`; OPEN → needs further analysis |

### Issue state interpretation

| State + Label | Verdict Signal |
|--------------|---------------|
| CLOSED + `not_target` / `wontfix` | HIGH confidence `Not applicable` |
| CLOSED + `skipped` | Stale skip decorator → `To be enabled` (skip should be removed) |
| CLOSED + no relevant label | Likely fixed → `To be enabled` |
| OPEN + `skipped` | Active XPU skip → follow **Dynamic-Skip Rule** (local verify) |
| OPEN + `bug` | Active XPU bug → `Failures (xpu broken)` |
| OPEN + `feature` / `enhancement` | Missing XPU feature → `Feature gap` |
| CLOSED (any label, issue body says "fixed by") | Check if the fix resolved the XPU issue → may be `To be enabled` |

## Example Output

```python
{
    "has_known_issue": True,
    "matches": [
        {
            "issue_url": "https://github.com/intel/torch-xpu-ops/issues/3346",
            "repo": "intel/torch-xpu-ops",
            "issue_number": 3346,
            "title": "[PVC+FP16] accuracy issue in test_Conv2d_naive_groups",
            "state": "OPEN",
            "labels": ["bug", "triaged", "P0"],
            "match_type": "exact_test_name",
            "match_evidence": "Issue title contains 'test_Conv2d_naive_groups'",
            "relevance": "HIGH",
        },
        {
            "issue_url": "https://github.com/intel/torch-xpu-ops/issues/2918",
            "repo": "intel/torch-xpu-ops",
            "issue_number": 2918,
            "title": "CUDA jiterator is not supported on XPU",
            "state": "CLOSED",
            "labels": ["not_target", "wontfix"],
            "match_type": "operator_api_match",
            "match_evidence": "Error message contains 'torch.cuda.jiterator'; test code block uses jiterator API",
            "relevance": "HIGH",
        },
        {
            "issue_url": "https://github.com/pytorch/pytorch/issues/180324",
            "repo": "pytorch/pytorch",
            "issue_number": 180324,
            "title": "[XPU] test_stream_pointer fails on Intel GPU",
            "state": "OPEN",
            "labels": ["module: xpu", "triaged"],
            "match_type": "keyword_match",
            "match_evidence": "Test name 'test_stream_pointer' and 'xpu' appear in issue title",
            "relevance": "MEDIUM",
        },
    ],
    "search_summary": {
        "intel/torch-xpu-ops": {
            "searches_run": 5,
            "total_results": 3,
            "top_keywords": ["test_Conv2d_naive_groups", "jiterator"],
        },
        "pytorch/pytorch": {
            "searches_run": 5,
            "total_results": 1,
            "top_keywords": ["test_stream_pointer xpu"],
        },
    },
    "discovery_steps": [
        {"step": 1, "action": "search by test name", "keyword": "test_Conv2d_naive_groups", "repo": "intel/torch-xpu-ops", "results": 1, "key_match": "Issue #3346"},
        {"step": 1, "action": "search by operator", "keyword": "torch.cuda.jiterator", "repo": "intel/torch-xpu-ops", "results": 1, "key_match": "Issue #2918"},
        {"step": 2, "action": "search by test name", "keyword": "test_stream_pointer", "repo": "pytorch/pytorch", "results": 1, "key_match": "Issue #180324"},
    ],
}
```

## Reference: Known Issue Search Patterns from classify_ut

The following search patterns are sourced from `classify_ut/RULES.md` and the status-specific subskills. Use them as templates when constructing search keywords.

### Direct test name search

```bash
gh search issues "test_foo" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url
gh search issues "test_foo xpu" --repo=pytorch/pytorch --limit=10 --json=number,title,state,labels,url
```

### Class name search

```bash
gh search issues "TestFoo" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url
```

### Operator/API search (from error message or test code)

```bash
gh search issues "aten::_cudnn_rnn" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url
gh search issues "torch.cuda.jiterator" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url
```

### File path search

```bash
gh search issues "test_streams.py" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url
```

### `not_target` / `wontfix` label filter

```bash
gh search issues "test_foo" --repo=intel/torch-xpu-ops --limit=10 --label=not_target --json=number,title,state,labels,url
gh search issues "test_foo" --repo=intel/torch-xpu-ops --limit=10 --label=wontfix --json=number,title,state,labels,url
```

### Label + keyword combination

```bash
gh search issues "test_foo xpu" --repo=intel/torch-xpu-ops --limit=10 --label=skipped --json=number,title,state,labels,url
```

### Verify issue state

```bash
gh issue view 3346 --repo=intel/torch-xpu-ops --json=state,labels,title,body,closedAt,url
gh issue view 180324 --repo=pytorch/pytorch --json=state,labels,title,body,closedAt,url
```

## Hard Constraints

- Search **both** `intel/torch-xpu-ops` and `pytorch/pytorch` repos. An issue in only one repo is insufficient to conclude "no known issue."
- Always call `gh issue view` to verify issue state. `gh search issues` results can be stale.
- Never treat a CLOSED issue as active evidence of a bug — check what the closure resolved.
- If no results come from keyword-based searches, try reducing keywords to the base test name (strip `_cuda`/`_xpu`/`_cpu` suffix).
- If still no results, report `has_known_issue = False` with empty matches. Do NOT fabricate or infer issues.
- For `not_target`/`wontfix` labeled issues, report them at HIGH relevance regardless of match_type — these are authoritative classification signals.
