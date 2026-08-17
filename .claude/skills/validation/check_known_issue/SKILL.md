---
name: check-known-issue
description: Search both intel/torch-xpu-ops and pytorch/pytorch for known issues related to a test case. Returns matched issues with issue state, labels, and match evidence for downstream classification.
---

# `check-known-issue`

## Objective
Determine if a failing test case is already tracked as a known issue in upstream (`pytorch/pytorch`) or downstream (`intel/torch-xpu-ops`) repositories to avoid duplicate reporting.

## Inputs
- `test_file`, `class_name`, `test_name`, `error_message`, `device`

## Output Format
Return this JSON object:
```python
{
    "has_known_issue": bool,
    "matches": [
        {
            "issue_url": str,
            "repo": str,
            "issue_number": int,
            "title": str,
            "state": "OPEN|CLOSED",
            "labels": [str],
            "match_evidence": str,  # Explain WHY this issue matches the test
            "relevance": "HIGH|MEDIUM|LOW"
        }
    ],
    "classification": {
        "Reason": str,
        "DetailReason": str
    }
}
```

## Deep Analysis Workflow

### 1. Mandatory Input Scrubbing
- **Ignore** any pre-existing `Reason` or `DetailReason` from the task input. Do not carry them forward.
- Base your search strictly on the provided test metadata and error message.

### 1.5. MANDATORY Deterministic Pass (run FIRST, before any heuristic search)

Empirically, GitHub's `gh search issues` does NOT tokenize compound identifiers:
searching an operator substring like `adaptive_max_pool1d` returns nothing, while
searching the FULL test name or the class name reliably finds the issue. Skip-tracking
issues (`label:skipped`) list their covered cases verbatim in the issue BODY in the form
`op_ut,<dotted.module.path>.<ClassName>,<full_test_name>`. This pass matches that body
text deterministically so detection does not depend on query phrasing or ranking.

Run ALL of the following unconditionally (do NOT skip based on earlier hits):

```bash
# A) Full-test-name search — the most reliable gh query (matches body verbatim).
gh search issues "<full_test_name> is:issue" --repo=intel/torch-xpu-ops --limit=20 --json=number,title,state,labels,url
gh search issues "<full_test_name> is:issue" --repo=pytorch/pytorch --limit=20 --json=number,title,state,labels,url

# B) Class-name search — reliable when the exact test name is body-only.
gh search issues "<class_name> is:issue" --repo=intel/torch-xpu-ops --limit=20 --json=number,title,state,labels,url

# C) Unconditional label:skipped body scan on intel/torch-xpu-ops.
#    Dump ALL open skipped issues WITH BODIES, then string-match locally for the
#    full test name AND the class name. This is the deterministic anchor.
gh issue list --repo=intel/torch-xpu-ops --label skipped --state open --limit 200 \
    --json number,title,state,labels,url,body
```

**Deterministic match rule (no judgment):** For every issue body returned by (A), (B),
or (C), test whether it contains the literal `<full_test_name>` string (case-insensitive),
OR a line matching `,<class_name>,<full_test_name>` (the `op_ut,...` skip-list format).
If EITHER matches:
- Set `has_known_issue = True`, `relevance = "HIGH"`.
- `match_evidence = "Deterministic body match: issue body contains literal '<full_test_name>'"`.
- Proceed directly to **Step 5 (Classification Synthesis)** using that issue's state/labels.
- Do NOT run the heuristic fallback — the deterministic match wins.

If NONE of (A), (B), (C) yields a literal match, fall through to the heuristic search
(Steps 2-4) below, which uses agent judgment for fuzzier matches (error snippets,
related operators, DISABLED-issue resolution).

### 2. Fallback Path: Parallel Issue Search via `gh search issues` (heuristic + agent judgment)
Extract precise keywords from the inputs (e.g., test name, operator name, class name, error snippet).
Launch parallel `gh search issues` commands across **both** repos.

**CRITICAL**: 
- You MUST append `is:issue` to exclude PRs.
- `gh search issues` queries are case-insensitive by default.
- GitHub's search API indexes issue TITLES better than BODIES. If the test name is specific (e.g. `adaptive_max_pool1d`), also search the operator name extracted from the test name — it may appear in either title or body.
- **ALWAYS check the error message for inline issue URLs FIRST** — before any search. Test skip messages often directly embed GitHub issue URLs like `https://github.com/pytorch/pytorch/issues/158115`. Extract them with regex and verify with `gh issue view`.

```bash
# Run ALL of these in parallel via the bash tool:

# STEP 0: Extract inline issue/PR URLs from the error message (fastest signal)
# Use grep or python to find URLs matching github.com/*/issues/NNNNN in the error_message input
# Then verify each with: gh issue view <NUMBER> --repo=<REPO> --json title,state,labels,url

# Search by the full test name (may match title or body)
gh search issues "<test_name_no_suffix> is:issue" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &

# Search by operator name extracted from test name (e.g., "adaptive_max_pool1d" from "test_comprehensive_nn_functional_adaptive_max_pool1d")
# Extract the operator name by removing test prefixes like "test_comprehensive_nn_functional_", "test_dtypes_", "test_", etc.
gh search issues "<operator_name> is:issue" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &

# Search by class name
gh search issues "<class_name> is:issue" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &

# Search by error snippet (short, unique part of the error message)
gh search issues "<error_snippet> is:issue" --repo=intel/torch-xpu-ops --limit=10 --json=number,title,state,labels,url &

# Also search pytorch/pytorch
gh search issues "<test_name_no_suffix> xpu is:issue" --repo=pytorch/pytorch --limit=5 --json=number,title,state,labels,url &
wait
```

**MANDATORY: Validate search results.** After `wait`, check EACH result:
- Did the command exit with code 0? (check `$?` or whether output is valid JSON)
- Is the output an empty array `[]` (no results found)?

If ANY of the `gh search issues` commands returned empty (zero results) or HTTP error, do NOT conclude "no issue exists". The GH search API has poor body-text indexing and may miss issues where the test name only appears in the body. Proceed to the **Web-Fetch Fallback** below for deeper search.

### 3. Web-Fetch Fallback: Direct Issue Listing

If the primary search returned empty results (or HTTP error), you MUST use `webfetch` as a fallback. GitHub's issue search is index-based; direct URL queries can still find issues that the search API missed.

**IMPORTANT: GitHub's web issue listing defaults to sorting by "Most recently updated" and paginates at ~12 results per page.** If the issue is older or hasn't been updated recently, it may not appear in the first page. You MUST fetch multiple pages (at least pages 1-3) for broad searches.

**Strategy A — Search by test name/operator on intel/torch-xpu-ops:**

```bash
# Test name search (exact test method name — may match body text):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<test_name_no_suffix>" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<test_name_no_suffix>" format=text

# Operator name extracted from test name (e.g., "adaptive_max_pool1d"):
# This is CRITICAL because the operator name is more likely to appear in issue bodies
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<operator_name>" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<operator_name>" format=text

# Class name search (catches issues filed against a whole test class):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<class_name>" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<class_name>+xpu" format=text

# File name search (test file basename like "test_decomp_xpu"):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<test_file_basename>" format=text
```

**Strategy B — Search by label + operator/pattern on intel/torch-xpu-ops:**
Issues tracking known UT failures are commonly tagged with the `skipped` label:

```bash
# Browse open issues with the `skipped` label. 
# Since there may be 50+ issues, search MULTIPLE pages:
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+label%3Askipped" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?page=2&q=is%3Aissue+is%3Aopen+label%3Askipped" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?page=3&q=is%3Aissue+is%3Aopen+label%3Askipped" format=text

# Also search closed skipped issues (if no open match found):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+label%3Askipped+<operator_name>" format=text
```

**Strategy C — Search pytorch/pytorch:**

```bash
webfetch "https://github.com/pytorch/pytorch/issues?q=is%3Aissue+<test_name_no_suffix>+xpu" format=text
webfetch "https://github.com/pytorch/pytorch/issues?q=is%3Aissue+<operator_name>+xpu" format=text
```

**How to evaluate `webfetch` results:**
The fetched page will contain issue title/body text. Look for:
- Exact test method name match (`test_comprehensive_nn_functional_adaptive_max_pool1d_xpu_bfloat16`)
- Exact class name match (`TestDecompXPU`)
- File path references (`test_decomp_xpu.py`)
- Error message snippets that match the input `error_message`
- Operator name match in issue title or body (`adaptive_max_pool1d`)

For each page fetched, carefully scan the issue titles listed. If none match, move to the next page. If you find a title that references the test class or file, fetch the issue body to check for the exact test name.

**IMPORTANT**: `webfetch` returns rendered HTML as text. Search the returned text for the test name, class name, and key error terms using Python string matching or regex. If the page lists issue entries (which it will when results are found), each entry typically contains the issue title, labels, and a snippet.

### 4. Deep Context Matching
For the most promising search results (from the deterministic pass or either heuristic path), fetch the full issue body to verify context:

Primary path:
```bash
gh issue view <number> --repo=<repo> --json title,body,state,labels,url
```

Fallback path (if `gh issue view` also fails):
```bash
webfetch "https://github.com/intel/torch-xpu-ops/issues/<number>" format=text
webfetch "https://github.com/pytorch/pytorch/issues/<number>" format=text
```

Use agent reasoning to determine if the issue actually describes this test's failure:
- **HIGH relevance**: Exact test name, exact stack trace, or exact operator in the same context.
- **MEDIUM relevance**: Similar error message, same class, or related operator.
- **LOW relevance**: Mention of the file or generic error.

### 5. Classification Synthesis
If `has_known_issue == True`, select the highest relevance match and apply this EXACT mapping:

**For issues in `intel/torch-xpu-ops`:**

| Issue State | Labels | `Reason` | `DetailReason` |
|-------------|--------|----------|----------------|
| OPEN | `bug` or no label | `Failures (xpu broken)` | `"Known bug: <Title> - <URL>"` |
| OPEN | `feature` / `enhancement` | `Feature gap` | `"Missing feature: <Title> - <URL>"` |
| OPEN | `skipped` | `Failures (xpu broken)` | `"Known issue <repo>#<number> (OPEN, skipped) — <Title>. <URL>"` |
| CLOSED | `not_target` / `wontfix` | `Not Applicable` | `"Not applicable per closed issue: <Title> - <URL>"` |
| CLOSED | other or no relevant label | `To be enabled` | `"Issue closed, awaiting enablement: <Title> - <URL>"` |

**For issues in `pytorch/pytorch`:**

| Issue State | Labels | `Reason` | `DetailReason` |
|-------------|--------|----------|----------------|
| OPEN | has `module: xpu` label | `Failures (xpu broken)` | `"Known XPU bug in pytorch/pytorch#<number> (OPEN) — <Title>. <URL>"` |
| OPEN | no `module: xpu` label | `Failures (stock broken)` | `"Known upstream bug (no XPU label) in pytorch/pytorch#<number> (OPEN) — <Title>. <URL>"` |
| OPEN | `feature` / `enhancement`, no `module: xpu` | `Feature gap` | `"Missing upstream feature: <Title> - <URL>"` |
| CLOSED | `not_target` / `wontfix` | `Not Applicable` | `"Not applicable per closed issue: <Title> - <URL>"` |
| CLOSED | title starts with `DISABLED` | *(see DISABLED issue resolution below)* | *(see DISABLED issue resolution below)* |
| CLOSED | other or no relevant label | `To be enabled` | `"Issue closed, awaiting enablement: <Title> - <URL>"` |

> **Key rule**: A `pytorch/pytorch` issue is only `Failures (xpu broken)` when it carries the `module: xpu` label. Without that label it is a stock/upstream failure that happens to affect XPU — use `Failures (stock broken)` so the two failure modes are clearly distinguished.

#### DISABLED Issue Resolution (Closed `pytorch/pytorch` Issues)

`pytorch/pytorch` uses a bot-managed system where flaky or broken tests are temporarily disabled by filing issues with the title format `DISABLED test_name (ClassName)`. When such an issue is **closed**, it means the bot re-enabled the test — but the test source code may still have an explicit skip decorator referencing a **different, still-active** issue.

When you encounter a **closed** `pytorch/pytorch` issue whose title starts with `DISABLED`, apply this resolution procedure **before** returning the result:

**Step 1 — Identify the base test name and class.**

The DISABLED issue title format is: `DISABLED <test_name> (<module>.<ClassName>)`.

- Extract `test_name` and `ClassName` from the title.
- The `test_name` in the title is the **base name** (device/dtype suffixes are stripped). For example, a DISABLED issue for `test_bitwise_ops` covers all parametrized variants: `test_bitwise_ops_xpu`, `test_bitwise_ops_xpu_float32`, etc.
- Use the input `test_name` to identify the matching method name in the test file. Strip device/dtype suffixes (e.g., `_xpu`, `_xpu_bfloat16`, `_xpu_float32`) to recover the base method name used in source.

**Step 2 — Search the test source file for skip decorators.**

Locate the relevant test file using the input `test_file` (e.g., `test/test_ops.py`, `test/test_decomp.py`). Then search that file for the base test method and any skip decorators applied to it:

```bash
# Find the test method definition and surrounding decorators
grep -n "def <base_test_name>\|unittest\.skip\|skipIf\|skipUnless\|skip_but_pass_in_sandcastle\|skipIfXpu\|skipIfRocm\|skipIfTorchDynamo\|expectedFailure" \
    /path/to/<test_file>.py | head -60

# Also search for issue URLs near the test method
grep -n "github\.com/[^/]*/[^/]*/issues/[0-9]" /path/to/<test_file>.py | head -40
```

Because `instantiate_device_type_tests` generates device-specific subclasses (e.g., `TestCommonXPU` from `TestCommon`), the skip decorator for an XPU test may be on the **base class method** in the upstream test file, not in the XPU-specific wrapper. Check both locations:
1. The XPU test wrapper file referenced by `test_file` (e.g., `test/xpu/test_ops_xpu.py`)
2. The upstream test file that the XPU wrapper imports (e.g., `test/test_ops.py`)

**Step 3 — Evaluate the skip decorator.**

After finding the test method's skip decorators, check whether any embed a GitHub issue URL:

- **Skip decorator found with an issue URL**: Extract the issue URL (regex: `https://github\.com/[^/]+/[^/]+/issues/\d+`). Look up that issue with `gh issue view <number> --repo=<repo> --json title,state,labels,url`. Then classify using the **state and labels of that extracted issue**, not the original closed DISABLED issue.
  - Return `issue_url` = the extracted issue's URL.
  - Apply the standard `pytorch/pytorch` or `intel/torch-xpu-ops` classification table rows to the extracted issue.

- **Skip decorator found with no issue URL** (e.g., `@unittest.skip("TODO: broken")`): The test is still explicitly skipped but there is no active tracking issue. Treat as `has_known_issue = False` — set `Reason = "Submit Issue"` and note in `DetailReason` that the test has a skip decorator without a linked issue.

- **No skip decorator found on the test method**: The DISABLED issue is closed and the test code has no override skip — the fix landed and the test is active. Treat as `has_known_issue = False`, `Reason = "Submit Issue"`, `DetailReason = "DISABLED issue pytorch/pytorch#<number> is closed and no skip decorator found in source — test is active."`.

**Summary table for closed DISABLED issues:**

| Skip decorator in source? | Issue URL in decorator? | Extracted issue state | `has_known_issue` | `issue_url` | `Reason` / `DetailReason` |
|---|---|---|---|---|---|
| Yes | Yes | OPEN | `true` | Extracted issue URL | Classify by extracted issue's labels (standard table) |
| Yes | Yes | CLOSED | `false` | — | `"Submit Issue"` / `"DISABLED issue closed; skip decorator references closed issue <URL> — test is blocked with no active tracker."` |
| Yes | No | — | `false` | — | `"Submit Issue"` / `"Skip decorator present but no issue URL found; submit new issue."` |
| No | — | — | `false` | — | `"Submit Issue"` / `"DISABLED issue pytorch/pytorch#<number> is closed and no skip decorator found — test is active."` |

If `has_known_issue == False`:
- `classification.Reason` = `"Submit Issue"`
- `classification.DetailReason` = `"No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`

**IMPORTANT**: After exhausting all search strategies (deterministic pass 1.5 + heuristic fallback A/B/C), if you genuinely found no matching issue, set `has_known_issue = False`. Be thorough — the goal is to avoid duplicate issue reporting.

## Strict Constraints (ZERO TOLERANCE)

1. **NO Pull Requests**: You MUST append `is:issue` to every `gh search issues` command. PRs are strictly invalid evidence. For `webfetch` fallback, the GitHub issue listing URL naturally filters to issues.
2. **Explicit Evidence**: The `DetailReason` MUST contain the literal Issue URL as shown in the Synthesis table.
3. **Tool Preference**: Use `bash` (with `gh` CLI) as your primary tool. Prefer `webfetch` over `websearch` for fallback because `webfetch` retrieves a specific known URL with deterministic results, whereas `websearch` uses opaque search indexing that may miss body-level content.
4. **No Blind Copies**: Output `DetailReason` MUST NOT simply repeat the input's `DetailReason`.
5. **No Excel Reads**: Never read the input Excel file directly. Rely only on the passed task parameters.
6. **Search Must Actually Succeed Before Concluding 'No Results'**: If both the deterministic pass and heuristic fallback fail or produce no parseable output, the skill MUST report an error rather than returning `has_known_issue = False` with certainty. Use `has_known_issue = False` with `confidence = "Low"` and note in `DetailReason` which search methods failed.
7. **Extract Inline Issue URLs from Error Messages**: Before running any search, scan the `error_message` input for GitHub issue URLs matching `github.com/([^/]+)/([^/]+)/issues/(\d+)`. If found, look them up with `gh issue view` and verify relevance. This is the FASTEST and most reliable signal — the error message itself may contain the exact issue URL. Do NOT skip this step even if search results are empty.
8. **Deterministic Pass Is Mandatory**: You MUST execute Step 1.5 (full-test-name search, class-name search, and the unconditional `label:skipped` body dump) on EVERY invocation. Never emit `has_known_issue = False` for a `failed`/`skipped` test without having run the Step 1.5 `gh issue list --label skipped --state open ... --json ...,body` dump and locally string-matched the full test name and class name against every returned body. Returning "Submit Issue / no known issue" without this deterministic pass is a hard violation.