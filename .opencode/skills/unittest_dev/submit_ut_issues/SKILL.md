# Submit XPU UT Issues to Intel/torch-xpu-ops

## Overview

This skill documents the workflow for analyzing XPU unit test failures, identifying root causes, classifying issues, applying fixes when test code is the problem, and submitting well-documented issues to intel/torch-xpu-ops.

When the failure is observed during a porting or enablement PR on `intel/torch-xpu-ops`, every submitted issue MUST include a **Context** section that cross-links the PR (see Issue Template).

---

## Tools

### Core Tools

| Tool | Purpose | Key Commands |
|------|---------|-------------|
| **Bash** | Execute shell commands for pytest, git, curl, jq | `pytest`, `curl`, `git`, `gh` |
| **Read** | Read test files, tracebacks, XML results | File traversal, content inspection |
| **Edit** | Modify test files to apply fixes | `edit` with `oldString`/`newString` |
| **Write** | Create issue markdown files | Document findings, workflows |
| **Glob** | Find test files and XML results | Pattern matching for files |
| **Grep** | Search for error patterns | Regex search in codebase |
| **Question** | Confirm user intent | Clarify decisions, gather requirements |
| **Task** | Launch sub-agents for parallel analysis | Complex investigation |
| **WebFetch** | Retrieve GitHub issue details | API calls, status checks |

### Bash Tool - Command Examples

```bash
# Run tests with XML output
pytest -v --junit-xml=test_<name>_xpu.xml dynamo/test_<name>_xpu.py

# Check environment
python3 -c "import torch; print(torch.xpu.is_available())"

# GitHub API verification
curl -s "https://api.github.com/repos/intel/torch-xpu-ops" -H "Authorization: Bearer $GITHUB_TOKEN"

# Generate test summary
python ../../.github/scripts/check-ut.py -i *.xml
```

### Grep Tool - Search Patterns

```bash
# Search for error patterns
pattern="ocloc failed|stream_index|record_stream|CPU backend"

# Search PyTorch DISABLED tests
pattern="DISABLED.*test_name"

# Search for related Intel issues
pattern="#2979|#1762|issue.*stream"
```

### Task Tool - Usage

```bash
# Parallel investigation of multiple error patterns
(task parallel for ocloc/IGC failures, stream API failures, record_stream CPU)
```

---

## Preconditions

### 1. Environment Setup

```bash
# Conda environment must be accessible
source ~/miniforge3/bin/activate pytorch_opencode_env

# Verify PyTorch with XPU
python3 -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"

# Verify git remotes configured
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops
git remote -v
# Expected: daisyden (fork) and intel (upstream) remotes
```

### 2. GitHub Authentication

```bash
# Token must have 'repo' scope for creating/updating issues
# Verify token works:
curl -s -X GET "https://api.github.com/user" \
  -H "Authorization: Bearer $GITHUB_TOKEN" | grep login

# Token stored securely (not hardcoded in scripts)
export GITHUB_TOKEN="ghp_XXXXXXXXXXXXXXX"
```

### 3. Test Execution Environment

```bash
# Working directory: pytorch repo root
cd $HOME/daisy_pytorch

# OR: torch-xpu-ops test directory
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu

# XML output directory must be writable
ls -la | grep test_*.xml  # Check for existing XML files
```

### 4. Required Packages

```bash
# Core dependencies
pip install pytest junitparser

# GitHub CLI (alternative to curl API)
which gh || pip install gh
gh auth status
```

### 5. Network Access

```bash
# Must be able to reach GitHub API
curl -s --max-time 10 "https://api.github.com" | grep -q "cloud" && echo "Connected"

# Must be able to reach PyPI
pip index versions torch 2>/dev/null && echo "PyPI accessible"
```

### 6. Source Code Accessibility

```bash
# pytorch source must be accessible for import path additions
ls -la $HOME/daisy_pytorch/test/dynamo/utils.py

# torch-xpu-ops test directory structure
ls -la $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu/dynamo/
```

### 7. Prerequisites Checklist

| Item | Verification Command | Expected Result |
|------|---------------------|-----------------|
| Conda env | `conda info --envs` | `pytorch_opencode_env` listed |
| XPU available | `python -c "import torch; print(torch.xpu.is_available())"` | `True` |
| Git remotes | `cd torch-xpu-ops && git remote -v` | 2 remotes configured |
| GitHub token | `curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user` | User JSON |
| pytest | `pytest --version` | Version >= 7.0 |
| Network | `curl -s --max-time 5 https://api.github.com` | Connected |

---

## Constraints

### 1. Authentication Constraints

| Rule | Description | Enforcement |
|------|-------------|-------------|
| **No Hardcoded Tokens** | Never embed tokens in scripts or files | Code review |
| **Environment Variables** | Pass tokens via `$GITHUB_TOKEN` | Required |
| **Scope Requirement** | Token must have `repo` scope | Verify before use |
| **Token Expiry** | Expired tokens return 401 | Check token validity |

```bash
# CORRECT: Use environment variable
export GITHUB_TOKEN="ghp_XXX"
curl -H "Authorization: Bearer $GITHUB_TOKEN" ...

# WRONG: Never hardcode
curl -H "Authorization: Bearer ghp_XXX" ...  # PROHIBITED
```

### 2. API Rate Limits

| API Tier | Limit | Window |
|----------|-------|--------|
| Authenticated | 5,000 requests | per hour |
| Unauthenticated | 60 requests | per hour |

```bash
# Check remaining rate limit
curl -I -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/rate_limit | jq '.rate'

# Batch strategy: Pre-collect data before API calls
# Collect all data locally first, then make minimal API calls
```

### 3. Test Execution Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Timeout | 120s per test | Prevent hangs |
| XPU Required | Yes | Tests need hardware |
| Isolation | Required | Avoid state leakage |
| Parallelism | Limited | 2-3 concurrent agents max |

```bash
# Timeout enforcement example
pytest --timeout=120 dynamo/test_*.py

# Isolation: Run each test file independently
pytest -v dynamo/test_a_xpu.py
pytest -v dynamo/test_b_xpu.py  # Separate invocation
```

### 4. Issue Submission Constraints

| Constraint | Rule | Example |
|------------|------|---------|
| One Issue per Category | Group by error pattern | All ocloc errors → 1 issue |
| Required Labels | Must include `skipped` | `["skipped", "module: ut"]` |
| Label Limit | Max 100 per issue | Prune labels |
| No Duplicates | Check existing first | Search before create |
| **PR Context (when filed during a porting PR)** | **MUST include a Context section naming the PR by number, linking its URL, stating current PR state for the test, and what becomes possible once resolved** | See Issue Template |

### 5. Code Modification Constraints

| Fix Type | Allowed | Not Allowed |
|----------|---------|-------------|
| Import path additions | Yes | ... |
| API generalization (CUDA→XPU) | Yes | ... |
| Skip logic additions | Yes | ... |
| Test assertion changes | No | ... |
| Backend/Infrastructure fixes | No | ... |

**Allowed Modifications**:
- Add `sys.path` to pytorch test/dynamo
- Generalize `torch.cuda.X` → device-aware flow
- Add attribute checks with skip
- Fix f-string/syntax errors

**Prohibited Modifications**:
- Change test expectations/assertions
- Modify backend/infra code
- Remove test coverage
- Weaken skip conditions

### 6. Documentation Constraints

| Required Field | Format | Example |
|----------------|--------|---------|
| File:Line Reference | `path/file.py:123` | `test_misc_xpu.py:456` |
| pytest Command | Full command | `pytest -v dynamo/test_misc_xpu.py -k test_foo` |
| Related Issues | Verified links | Intel #3386, PyTorch #178155 |
| Versions | Package list | Intel IGC 1.0.224, PyTorch 2.5.0 |

### 7. Decision Hierarchy

```
Before creating issue, verify in order:

1. Can it be fixed in TEST CODE?
   ├── Import path missing → Add sys.path → RERUN → Verify
   ├── API not generalized → Use xpu-aware flow → RERUN → Verify
   └── Skip condition needed → Add skip logic → RERUN → Verify
   
2. Is it a KNOWN PATTERN?
   ├── Cross-ref PyTorch DISABLED
   ├── Check intel/torch-xpu-ops existing
   └── Search error keyword in issues
   
3. Is it a NEW BUG?
   ├── Document error pattern
   ├── Group related tests
   ├── Create well-structured issue
   └── Apply appropriate labels
   
4. FINAL: Submit only if truly unfixable AND valid infrastructure bug
```

### 8. Scope Constraints

| Scope Item | Limit | Notes |
|------------|-------|-------|
| Test Files per Session | 3-5 files | Comprehensive coverage |
| Agents Parallelism | 2-3 max | Resource limits |
| Issues per Category | 1 | Group by pattern |
| Labels per Issue | ≤5 | Focus on key labels |
| Summary Files | 1 | Aggregated view |

---

## Prerequisites

- GitHub token with `repo` scope for intel/torch-xpu-ops
- PyTorch development environment with XPU support
- Conda environment: `pytorch_opencode_env`

## Workflow

### Phase 1: Run Tests and Collect Results

#### Step 1.1: Activate Environment and Run Tests

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu

# Run tests with XML output
pytest -v --junit-xml=test_<name>_xpu.xml dynamo/test_<name>_xpu.py
```

#### Step 1.2: Collect All Results

```bash
# Generate aggregated test summary
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu
python ../../.github/scripts/check-ut.py -i *.xml
```

#### Step 1.3: Understand Test Results

- **Passed**: Test executed and passed
- **Failed**: Test executed but assertion/failure occurred
- **Skipped**: Test skipped by decorator or skipIf condition
- **xFailed**: Expected failure that occurred (acceptable)

---

### Phase 2: Group Failures by Error Pattern

#### Step 2.1: Analyze Error Messages

Collect failure patterns and categorize:

**ocloc/IGC-related** (4 patterns):
- "ocloc failed with error code"
- "Loading of IGC library has failed"
- "IGC initialization failure"
- Error code 250

**Stream API-related** (3 patterns):
- "streams::wait_stream() Expected 'int' ... found NoneType"
- "streams::record_event() Expected 'int' ... found NoneType"  
- "Could not run 'aten::record_stream' with arguments from the 'CPU' backend"

**Autocast dtype-related**:
- "Object comparison failed: torch.float32 != torch.float64"
- "XPU Autocast does not support fp32 dtypes"

**Import/Module-related** (2 patterns):
- "ModuleNotFoundError: No module named 'utils'"
- "failed to find name in frame builtins" (Dynamo graph break)

**Test Code Bug (Fixable)**:
- f-string format errors
- Missing sys.path configurations
- CUDA/XPU API not generalized

#### Step 2.2: Cross-Reference Similar Issues

```bash
# Search for related PyTorch issues
curl -s "https://api.github.com/search/issues?q=repo:pytorch/pytorch+<error_keyword>+is:issue+state:open&per_page=10" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Found: {d['total_count']}\"); [print(f\"#{i['number']}|{i['title'][:80]}\") for i in d['items'][:5]]"

# Search for DISABLED tests related to the issue
curl -s "https://api.github.com/search/issues?q=repo:pytorch/pytorch+DISABLED+<related_keyword>+is:issue&per_page=10"
```

#### Step 2.3: Group Issues Table

| Error Pattern | Test Count | Category | Root Cause |
|--------------|------------|----------|------------|
| ocloc/IGC failure | 4 | Infrastructure | oneAPI/IGC config |
| stream_index None | 3 | Dynamo Generalization | torch.compile trace |
| autocast dtype | 2 | Backend Feature | autocast precision |
| record_stream CPU | 4 | Backend/API | CPU not supported |
| Import/module | 2 | Test Infrastructure | path/config |

---

### Phase 3: Fix Test Code Issues (Deep Analysis)

#### Step 3.1: Identify Fixable Issues

Issues are fixable if:

1. **Import Path Issues**: Add sys.path to pytorch test/dynamo
```python
from pathlib import Path
PYTORCH_DYNAMO_PATH = str(Path(__file__).resolve().parents[5] / "test" / "dynamo")
if os.path.exists(PYTORCH_DYNAMO_PATH) and PYTORCH_DYNAMO_PATH not in sys.path:
    sys.path.insert(0, PYTORCH_DYNAMO_PATH)
```

2. **CUDA→XPU Generalization**: Replace CUDA-specific APIs
```python
# Before (fails on XPU-only)
s = torch.cuda.Stream(device=GPU_TYPE)

# After (generalized)
s = torch.xpu.Stream(device=GPU_TYPE) if GPU_TYPE == "xpu" else torch.cuda.Stream(device=GPU_TYPE)
```

3. **Missing Attribute Checks**: Add skip logic
```python
cs = torch.xpu.current_stream()
if not hasattr(cs, 'cuda_stream'):
    self.skipTest("cuda_stream attribute not available on XPU stream")
```

4. **Test Bugs**: Fix syntax/import errors in test code

#### Step 3.2: Apply Fix and Rerun

```bash
# Apply fix to test file
# Then rerun specific test
pytest -v dynamo/test_<name>_xpu.py -k "test_pattern"

# Check if passed
# If still failing, root cause is NOT test bug - document as issue
```

#### Step 3.3: Classification Decision Tree

```
Is the test failure due to test code bug?
├── YES: Import path missing → Fix sys.path, rerun
├── YES: CUDA->XPU not generalized → Generalize API, rerun  
├── YES: Missing attribute check → Add skip logic, rerun
├── NO:  Backend/Infrastructure issue → Submit as issue
└── NO:  PyTorch codebase issue → Submit as issue
```

---

### Phase 4: Create Well-Documented Issues

#### Step 4.1: Issue Structure

Each issue must include:

1. **Title**: `[Bug Skip] Category - Brief Description`

2. **Cases Section** (op_ut format):
```markdown
Cases:
op_ut,<full_python_module_path>,<TestClass.test_name>
```

3. **Error Message**: Full error text如有trace细节

4. **Test Code Snippet**: Actual test code that triggers failure
```python
# file:line reference
def test_example(self):
    # problematic code
    foo = torch.xpu.operation()  # FAILS here
```

5. **Traceback**: pytest_command and full traceback
```markdown
pytest_command:
cd ~/daisy_pytorch/third_party/torch-xpu-ops/test/xpu && pytest -v ...

Traceback (most recent call last):
  ...
```

6. **Root Cause Analysis**: Detailed explanation of why it fails

7. **Related Issues**: Links to Intel and PyTorch issues
```markdown
## Related Intel Issues
- #XXXX - Title (same error pattern)

## Related PyTorch Issues
- pytorch#XXXXX - Title (DISABLED if applicable)
- pytorch#XXXXX - Title (known limitation)
```

8. **Versions**: Package versions
```markdown
Intel: intel-igc-cm 1.0.224, intel-ocloc 25.18.33578
PyTorch: pytorch-triton-xpu 3.6.0+git225cdbde
```

#### Step 4.2: Labels

Apply appropriate labels:
- `skipped` - Required for dynamic skip template
- `module: ut` - Unit test issue
- `module: dynamo` / `module: inductor` / `module: nn` - PyTorch component
- `dtype: amp_bf16` / `dependency component: oneAPI` - Specific feature labels

#### Step 4.3: Submit via GitHub API

```bash
export GITHUB_TOKEN="<token>"

curl -s -X POST "https://api.github.com/repos/intel/torch-xpu-ops/issues" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d '{
    "title": "[Bug Skip] <Title>",
    "body": "<Full issue body with all sections>",
    "labels": ["skipped", "module: ut"]
  }'
```

---

### Phase 5: Update Issues with Missing Information

If initial submit lacks details:

```bash
# Update issue body
curl -s -X PATCH "https://api.github.com/repos/intel/torch-xpu-ops/issues/<issue_number>" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d '{"body": "<Updated body with traceback and code snippets>"}'
```

---

## Deep Analysis Guidelines

### Pattern 1: ocloc/IGC Failures

**Detection**: Search for "ocloc", "IGC", "libigc.so", "error code 250"

**Root Cause Analysis**:
- Check if oneAPI toolkit properly installed
- Verify LD_LIBRARY_PATH includes compiler runtime
- Check libigc.so version compatibility

**Related Intel Issues**: #2979, #1762

### Pattern 2: Stream API Failures

**Detection**: Search for "streams::wait_stream", "streams::record_event", "stream_index"

**Root Cause Analysis (Deep)**:
- Test uses torch.cuda/XPU Stream but dynamo lowers to primitive ops
- Stream context manager not propagated correctly in dynamo trace
- None handling missing for optional stream_index parameter

**Cross-Check**: PyTorch #180179 - CUDA Stream trace issues

### Pattern 3: Autocast Dtype Failures

**Detection**: Search for "torch.float32 != torch.float64", "autocast dtype"

**Root Cause Analysis (Deep)**:
- XPU autocast may not support all CUDA dtype promotions
- Graph export shows different dtype than runtime execution
- IGC float64 precision may require specific compiler flags

**Related Intel Issues**: #2680, #3359, #3356

### Pattern 4: record_stream CPU Backend

**Detection**: Search for "aten::record_stream", "CPU backend"

**Root Cause Analysis (Deep)**:
- Tensors created on CPU but stream APIs are XPU-specific
- record_stream operator not implemented for CPU backend
- XPU inductor lowers differently than CUDA

**Cross-Check PyTorch DISABLED**: #178155, #134746, #138557 (already disabled for instability)

### Pattern 5: Import/Module Failures

**Detection**: Search for "ModuleNotFoundError", "builtins" graph break

**Root Cause Analysis (Deep)**:
- PyTorch test utilities (utils.py, test_functions.py) not copied
- Dynamo traces function without parent's module globals
- Sibling module imports fail

**Fix Logic**:
```python
# Check if module can be imported from pytorch source
PYTORCH_DYNAMO_PATH = str(Path(__file__).resolve().parents[5] / "test" / "dynamo")
if os.path.exists(PYTORCH_DYNAMO_PATH) and PYTORCH_DYNAMO_PATH not in sys.path:
    sys.path.insert(0, PYTORCH_DYNAMO_PATH)
```

---

## Issue Template

```markdown
## Bug Description

<Brief description of the issue>

## Affected Tests

Cases:
op_ut,<module_path>,<TestClass.test_name>

## Error Message

```
<Full error message verbatim>
```

## Test Code Snippet

```python
# <file>:<line_start>-<line_end>
<Relevant test code showing the failing line>
```

## Traceback

```
pytest_command:
<Command to reproduce>

Traceback:
<Full stack trace>
```

## Root Cause Analysis

<Detailed explanation of WHY this failure occurs>

## Context

<REQUIRED whenever the failure is observed during a porting / enablement PR.
This section makes the issue self-contained for reviewers landing on it from
the PR, and lets future maintainers find the PR from the issue once the
underlying gap is fixed. Include all three points below.>

This issue tracks a gap identified during PR #<NNNN>, which <one-line
description of the PR scope, e.g. "ports a batch of CUDA tests to XPU under
`third_party/torch-xpu-ops/test/xpu/`">.

In that PR, `<test_name(s)>` is currently <skipped|failing|enabled-but-failing>
on XPU. Once <the underlying gap is resolved>, the <skip|failing assertion>
in PR #<NNNN> (`<file path>`) can be <removed|will pass without further changes>.

PR: https://github.com/intel/torch-xpu-ops/pull/<NNNN>

## Related PyTorch Issues

- pytorch#XXXXX - <Title> (DISABLED if applicable)
- pytorch#XXXXX - <Title> (related known issue)

## Related Intel/torch-xpu-ops Issues

- #XXXX - <Title>
- #XXXX - <Title>

## Versions

Intel: <relevant packages>
PyTorch: <version>
```

### Context Section: When Required

The **Context** section is **MANDATORY** whenever the failure was observed during a porting or enablement effort that has (or will produce) a PR on `intel/torch-xpu-ops`. Concretely:

- The failing test was newly added or newly enabled in a PR on `intel/torch-xpu-ops`.
- The PR currently masks the failure with a skip / xfail / weakened assertion that should be reverted once the issue is fixed.
- The PR enables the test "loud-fail" so this issue tracks the actual underlying gap.

If the failure is observed outside any PR context (e.g. routine CI on `main`), the Context section may be omitted; otherwise it MUST contain:

1. **PR cross-link** — issue body must name the PR by number (`PR #NNNN`) AND include a full URL to the PR.
2. **Current PR state for this test** — `skipped`, `failing`, or `enabled-but-failing`, with the file path that holds the workaround.
3. **What becomes possible once resolved** — explicitly state whether the skip can be removed, the assertion will pass without further code changes, etc.

### After Filing

When the Context section is used:

- Add the issue URL to the relevant commit message in the porting PR.
- Add an inline code comment next to the skip / weakened check pointing at the issue (`# Tracked in intel/torch-xpu-ops#NNNN`).
- If the PR description has a "Tracking issues" section, append the new issue URL there. If no PR exists yet, queue the URL for inclusion in the eventual PR body.
- Optionally, post a comment on the PR linking to the new issue so reviewers see it without scrolling the description.

---

## Deep Analysis with Subagents

### Overview

When encountering complex failure patterns, use the **Task tool with explore/general subagents** to perform parallel deep analysis. This approach accelerates root cause identification by distributing investigation across multiple agents.

### Parallel Analysis Strategy

| Phase | Action | Subagent Type | Output |
|-------|--------|---------------|--------|
| 1 | Analyze failure logs | explore | Error patterns, stack traces |
| 2 | Search codebase for similar issues | explore | Related code locations |
| 3 | Cross-reference known Intel/PyTorch issues | explore | Issue links and patterns |
| 4 | Synthesize findings | general | Root cause, fix recommendations |

### Launching Deep Analysis Subagents

#### Pattern 1: Error Log Analysis

```python
# Launch explore agent to analyze test failure logs
(task_id: "log_analysis") ->
  prompt: """
Analyze the following test failure logs for pattern identification:

1. Search for error keywords: "ocloc", "stream_index", "record_stream", "CPU backend", "dtype"
2. Identify stack trace patterns and their source locations
3. Categorize errors by infrastructure vs test code issues
4. Return: categorized errors, occurrence count, related PyTorch DISABLED tests

Working directory: $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu
XML Results location: ~/test_results/
"""
```

#### Pattern 2: Code Source Investigation

```python
# Launch explore agent to trace failure to source
(task_id: "code_trace") ->
  prompt: """
Investigate the following code paths for failure root cause:

1. For ocloc/IGC errors:
   - Search torch-xpu-ops for ocloc usage patterns
   - Check LD_LIBRARY_PATH configuration
   - Trace IGC library loading path

2. For stream API errors:
   - Search aten/stream implementations
   - Check ingestor lowering code
   - Find stream_index parameter handling

3. For autocast dtype errors:
   - Search autocast implementation
   - Check IGC precision settings
   - Find dtype conversion paths

Return: code locations, relevant function signatures, fix locations
"""
```

#### Pattern 3: Known Issue Cross-Reference

```python
# Launch general agent to research related issues
(task_id: "issue_research") ->
  prompt: """
Research known issues for the following error patterns:

1. ocloc/IGC failures:
   - Fetch intel/torch-xpu-ops issues #2979, #1762
   - Search for related oneAPI/IGC issues
   - Check PyTorch DISABLED tests

2. Stream API failures:
   - Fetch PyTorch issues #180179, #178155, #134746
   - Check Intel stream API issue status
   - Find workaround patterns

3. record_stream CPU backend:
   - Fetch PyTorch DISABLED tests #138557
   - Search for CPU/XPU tensor mismatch issues
   - Check XPU record_stream implementation status

Return: Issue URLs, status, posted workarounds, labels
"""
```

### Aggregating Subagent Findings

After parallel analysis, aggregate results using this decision tree:

```
                    ┌─────────────────┐
                    │ Subagent Report │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Test Code Bug│   │Known Infra Bug│   │ New Bug      │
   │ (FIXABLE)    │   │(DOCUMENTED)   │   │(NEW ISSUE)   │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │                  │                  │
          ▼                  ▼                  ▼
   Apply sys.path fix   Cross-ref existing  Create new issue
   Rerun test          Check similar          with full docs
   Verify pass         DISABLED tests        Group by pattern
```

### Subagent Prompt Template

Use this template for consistent subagent guidance:

```markdown
# Subagent Task Template

## Task: Deep <Error_Pattern> Analysis

### Input Data
- XML results: <path/to/test-results.xml>
- Test file: <path/to/test_file.py>
- Error pattern: "<error_keyword>"

### Investigation Scope

#### 1. Code Analysis
- Search patterns: ["<pattern1>", "<pattern2>"]
- Files to examine: [<file1>, <file2>]
- Functions: [<func1>, <func2>]

#### 2. Log Analysis  
- Error keywords: <keyword1>, <keyword2>
- Trace patterns: <typical stack trace format>
- Frequency: <number of occurrences>

#### 3. Known Issues
- Intel issues: #[issues separated by comma]
- PyTorch issues: #[disabilities separated by comma]
- Labels to check: [label1, label2]

### Deliverables

1. **Root Cause**: 2-3 sentence description
2. **Code Locations**: file:line references
3. **Known Issue Links**: Verified URLs
4. **Fix Recommendation**: If test code bug
5. **Issue Submission**: If infrastructure bug

### Output Format

Return JSON:
```json
{
  "root_cause": "...",
  "code_locations": ["file:line"],
  "known_issues": [{"url": "...", "title": "..."}],
  "fix_type": "test|infrastructure|new_bug",
  "recommendation": "..."
}
```

### Constraints

- Focus on XPU-specific failures only
- Check PyTorch DISABLED tests first
- Prioritize fixable test code issues
- Document all findings for issue submission
```

### Multi-Subagent Workflow Example

```python
# Example: Investigate 4 error patterns in parallel

# Subagent 1: ocloc/IGC failures
(task type:"explore", description:"Analyze ocloc IGC failures") ->
  Log pattern: "ocloc failed error 250"
  Tests: test_ctx_manager_xpu (4 tests)

# Subagent 2: Stream API failures  
(task type:"explore", description:"Trace stream_index None errors") ->
  Log pattern: "stream_index None"
  Tests: test_streams_xpu (3 tests)

# Subagent 3: Autocast dtype precision
(task type:"explore", description:"Investigate autocast dtype mismatch") ->
  Log pattern: "float32 != float64"
  Tests: test_ctx_manager_xpu (2 tests)

# Subagent 4: record_stream CPU backend
(task type:"explore", description:"Check record_stream CPU errors") ->
  Log pattern: "CPU backend not supported"
  Tests: test_streams_xpu (4 tests)

# After all complete: Aggregate findings
(general agent prompt) ->
  Combine 4 subagent reports into:
    - Fixable test code issues (apply fixes)
    - Documented infrastructure issues (submit 4 issues)
    - Known PyTorch DISABLED patterns (mark as wontfix)
```

### Verification Checklist After Analysis

| Step | Check | Pass Criteria |
|------|-------|---------------|
| 1 | Subagent findings logged | All agents returned results |
| 2 | Root cause identified | One sentence summary exists |
| 3 | Code locations verified | file:line refs valid |
| 4 | Related issues cross-checked | URLs verified accessible |
| 5 | Fix recommendation clear | Action path defined |
| 6 | Decision made | Fix/Document/Submit selected |

---

## Quick Reference Commands

```bash
# Activate environment
source ~/miniforge3/bin/activate pytorch_opencode_env

# Run single test
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu
pytest -v dynamo/test_<name>_xpu.py -k "test_pattern"

# Run all and collect
pytest -v --junit-xml=test_*.xml dynamo/

# Generate summary
python ../../.github/scripts/check-ut.py -i *.xml

# Search PyTorch issues
curl -s "https://api.github.com/search/issues?q=repo:pytorch/pytorch+<keyword>+is:issue+state:open"

# Submit issue
curl -s -X POST "https://api.github.com/repos/intel/torch-xpu-ops/issues" \
  -H "Authorization: Bearer $GITHUB_TOKEN" -H "Content-Type: application/json" \
  -d '{"title":"...","body":"...","labels":["skipped","module: ut"]}'
```

---

## Common Labels

| Label | Usage |
|-------|-------|
| `skipped` | Required for dynamic skip issues |
| `module: ut` | Unit test issues |
| `module: dynamo` | Dynamo/compiler issues |
| `module: inductor` | Inductor/triton issues |
| `module: nn` | Neural network layer issues |
| `dtype: amp_bf16` | AMP/autocast related |
| `dependency component: oneAPI` | oneAPI toolkit issues |
| `dependency component: IGC` | IGC compiler issues |