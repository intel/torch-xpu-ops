---
name: check-community-change
description: Deep analysis to determine if an upstream community change (removal, rename, refactor) broke a test case. Uses pytest --collect-only or source inspection.
---

# `check-community-change`

## Objective
Determine if a test failure is due to an upstream **community change** in `pytorch/pytorch` (e.g., base function deleted, renamed, or refactored) vs a local device gap.

## Inputs
- `test_file`, `class_name`, `test_name`, `device` (default `"cuda"`), `PYTORCH_SRC`, `conda_env`

`PYTORCH_SRC` is the `pytorch_folder` the calling agent already prepared; use it
as given. **Do NOT set up or activate any environment** (no `setup_env.sh`).
`conda_env` is the environment the caller established; the Step 3 Path A
(`import torch` device check and `pytest --collect-only`) MUST run through it via
`conda run -n "${conda_env}" ...`. If `conda_env` is not provided, skip Path A
and use Path B (source inspection) directly.

## Output Format
Return this JSON object:
```python
{
    "community_change": bool,
    "evidence": {
        "base_function": { "found": bool, "actual_method_name": str, "method_line": int },
        "device_case": { "verification_method": "file_existence_check|collect_only|source_inspection", "generated": bool },
        "git_history": { "commits_removing_method": [{"hash": str, "diff_summary": str}] }
    },
    "classification": {
        "change_type": "base_function_removed|base_function_renamed|device_case_not_generated|device_case_renamed|not_a_community_change|file_deleted",
        "change_scope": "base_test_removed|device_removed|refactored|renamed|none",
        "detail_reason": str  # e.g., "Base test removed: test_foo in test_ops.py (commit <hash>)"
    },
    "verdict": "Community Change" | "Not a community change"
}
```

## Deep Analysis Workflow

### 0. Export `PYTORCH_SRC`

Before running any command below, export the caller-provided path so every
`$PYTORCH_SRC` reference (and every `cd "$PYTORCH_SRC"`) resolves correctly. Do
NOT set up or activate any environment:

```bash
export PYTORCH_SRC="<pytorch_folder the caller provided>"
```

If `PYTORCH_SRC` is unset when a command runs, `$PYTORCH_SRC/...` expands to an
absolute path from filesystem root and silently matches nothing, and
`cd "$PYTORCH_SRC"` would `cd` to `/`.

### 1. Mandatory Input Scrubbing
- **Ignore** any pre-existing `Reason` or `DetailReason` from the input task. Do not carry them forward. Base your verdict strictly on local source and git history.
- **Never read the Excel file directly.**

### 1.5 Fast Path: Test File Existence (MANDATORY)

Before any git log or source search, check whether the test file still exists in `PYTORCH_SRC`:

```bash
if [ -f "$PYTORCH_SRC/$test_file" ]; then
    echo "EXISTS"
else
    echo "MISSING"
fi
```

- **If `MISSING`**: The test file was deleted upstream. Verdict is **Community Change** (`file_deleted`). Do NOT proceed to Steps 2 or 3 — the file deletion alone is sufficient evidence.
  - Run git log to identify the removal commit:
    ```bash
    cd "$PYTORCH_SRC" && git log --oneline -5 --diff-filter=D -- "$test_file"
    ```
  - Verify each commit is merged into HEAD:
    ```bash
    git merge-base --is-ancestor <commit_hash> HEAD
    ```
  - Output:
    ```python
    {
        "community_change": True,
        "evidence": {
            "base_function": {"found": False, "actual_method_name": "", "method_line": 0},
            "device_case": {"verification_method": "file_existence_check", "generated": False},
            "git_history": {"commits_removing_method": [{"hash": str, "diff_summary": str}]}
        },
        "classification": {
            "change_type": "file_deleted",
            "change_scope": "base_test_removed",
            "detail_reason": "Test file {test_file} was deleted from pytorch/pytorch (commit <hash>)."
        },
        "verdict": "Community Change"
    }
    ```

- **If `EXISTS`**: Proceed to Step 2.

### 2. Base Function Existence
Strip `test_name` suffixes (`_<device>`, `_<dtype>`, OpInfo suffixes) to determine the `base_method`.

Delegate the search to the `explore` agent:
```python
task(
    subagent_type="explore",
    run_in_background=True,
    description="Find base function definition",
    prompt="Find the definition of base method '<base_method>' in <test_file>. PYTORCH_SRC=<path>. Look for 'def <base_method>' or decorators like '@ops'. Return the actual method name and line number if found."
)
```
- **If Found:** Proceed to Step 3.
- **If NOT Found:**
  Search local git history for renaming/removal.
  ```bash
  cd "$PYTORCH_SRC" && git log --oneline -20 -- "$test_file"
  ```
  **MANDATORY GIT GATE:** Any cited commit MUST be merged into HEAD. Check via:
  ```bash
  git merge-base --is-ancestor <commit_hash> HEAD
  ```
  If merged, inspect `git show <commit_hash>`. Record the rename/removal. Verdict: **Community Change**.

### 3. Device Case Generation
If the base function exists, does it generate the target device case? Check device availability (through the caller's env):
```bash
conda run -n "${conda_env}" python3 -c "import torch; print(torch.<device>.is_available())"
```

If `conda_env` is unavailable or `import torch` fails, skip Path A and go
straight to Path B (source inspection).

#### Path A: Device Available (`pytest --collect-only`)
Authoritative ground truth. Run:
```bash
cd "$PYTORCH_SRC" && conda run -n "${conda_env}" python3 -m pytest "$test_file" --collect-only -q 2>/dev/null | grep "::${class_name}::"
```
- **Exact match found:** Not a community change.
- **Similar name found:** Renamed -> **Community Change**.
- **Not found:** Device excluded by design/gap -> **Not a community change** (it's a dtype/device gap).

#### Path B: Device Unavailable (Source Inspection)
Delegate to the subskill:
```python
task(
    subagent_type="explore",
    load_skills=["check-community-change-source-inspection"],
    prompt="Run source inspection for test_name='<test_name>', base_method='<base_method>', class_name='<class_name>', target_device='<device>'. Test file: <test_file_path>. PYTORCH_SRC=<path>."
)
```
Map its boolean findings: `device_case_renamed` -> **Community Change**. `device_case_not_generated` -> **Not a community change**.

## Strict Constraints (ZERO TOLERANCE)

1. **Tool Whitelist**: `read`, `bash` (local `git`, `grep`, `pytest`), `grep`, `glob`.
2. **Forbidden Tools**: NO `gh` CLI. NO `websearch`. NO `webfetch`. **No external network calls.**
3. **No Open PRs**: Evidence must be from local merged `git` history. A PR URL or unmerged commit is strictly invalid.
4. **Git Evidence Audit**: Every `community_change = True` verdict due to removal/rename MUST cite at least one verified ancestor commit hash.
5. **No Blind Copies**: Output `detail_reason` MUST NOT simply repeat the input's `Reason` or `DetailReason`.
6. **File existence check is MANDATORY and runs before Steps 2–3**: Step 1.5 (file existence check) MUST be the first analysis step after input scrubbing. Do NOT search for base method definitions, run git log, or check device generation before verifying the test file exists. File deletion is the strongest form of community change — no deeper analysis needed.