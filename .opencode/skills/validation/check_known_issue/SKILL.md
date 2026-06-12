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

### 2. Primary Path: Parallel Issue Search via `gh search issues`
Extract precise keywords from the inputs (e.g., test name without suffix, class name, operator name, unique error snippet).
Launch parallel `gh search issues` commands across **both** repos.

**CRITICAL**: You MUST append `is:issue` to exclude PRs.

```bash
# Run ALL of these in parallel via the bash tool:
gh search issues "<test_name_no_suffix> is:issue" --repo=intel/torch-xpu-ops --limit=5 --json=number,title,state,labels,url &
gh search issues "<test_name_no_suffix> xpu is:issue" --repo=pytorch/pytorch --limit=5 --json=number,title,state,labels,url &
gh search issues "<error_snippet_or_operator> is:issue" --repo=intel/torch-xpu-ops --limit=5 --json=number,title,state,labels,url &
wait
```

**MANDATORY: Validate search results.** After `wait`, check EACH result:
- Did the command exit with code 0? (check `$?` or whether output is valid JSON)
- Is the output an empty array `[]` (no results found)?

If ANY of the `gh search issues` commands returned HTTP error (non-zero exit / empty stdout / non-JSON output like HTML error page), you MUST proceed to the **Fallback Path** below. Do NOT assume "no results" — the search API may be unavailable.

### 3. Fallback Path: Web-Fetch Direct Issue Listing (when `gh search issues` fails)

If the primary search failed (API error), you MUST use `webfetch` as a fallback. GitHub's issue search is index-based; direct URL queries can still work when the search API returns 502.

**Strategy A — Search by test name on intel/torch-xpu-ops:**

```bash
# Test name search (exact test method name is the most reliable signal):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<test_name_no_suffix>" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<test_name_no_suffix>" format=text

# Class name search (catches issues filed against a whole test class):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<class_name>" format=text
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<class_name>+xpu" format=text

# File name search:
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+<test_file_basename>" format=text
```

**Strategy B — Search by label + operator/pattern on intel/torch-xpu-ops:**
Issues tracking known UT failures are commonly tagged with the `skipped` label:

```bash
# Browse all open issues with the `skipped` label (most relevant for UT failures):
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+label%3Askipped" format=text

# If the error message mentions a specific aten operator:
webfetch "https://github.com/intel/torch-xpu-ops/issues?q=is%3Aissue+is%3Aopen+<operator_name>" format=text
```

**Strategy C — Search pytorch/pytorch:**

```bash
webfetch "https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+<test_name_no_suffix>" format=text
webfetch "https://github.com/pytorch/pytorch/issues?q=is%3Aissue+<test_name_no_suffix>+xpu" format=text
```

**How to evaluate `webfetch` results:**
The fetched page will contain issue title/body text. Look for:
- Exact test method name match (`test_record_stream_problem_basic`)
- Exact class name match (`TestStreams`)
- File path references (`test_streams_xpu.py`)
- Error message snippets that match the input `error_message`

**IMPORTANT**: `webfetch` returns rendered HTML as text. Search the returned text for the test name, class name, and key error terms using Python string matching or regex. If the page lists issue entries (which it will when results are found), each entry typically contains the issue title, labels, and a snippet.

### 4. Deep Context Matching
For the most promising search results (from either primary or fallback path), fetch the full issue body to verify context:

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

| Issue State | Labels | `Reason` | `DetailReason` |
|-------------|--------|----------|----------------|
| OPEN | `bug` or no label | `Failures (xpu broken)` | `"Known bug: <Title> - <URL>"` |
| OPEN | `feature` / `enhancement` | `Feature gap` | `"Missing feature: <Title> - <URL>"` |
| OPEN | `skipped` | `Failures (xpu broken)` | `"Known issue <repo>#<number> (OPEN, skipped) — <Title>. <URL>"` |
| CLOSED | `not_target` / `wontfix` | `Not Applicable` | `"Not applicable per closed issue: <Title> - <URL>"` |
| CLOSED | other or no relevant label | `To be enabled` | `"Issue closed, awaiting enablement: <Title> - <URL>"` |

If `has_known_issue == False`:
- `classification.Reason` = `"Submit Issue"`
- `classification.DetailReason` = `"No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`

**IMPORTANT**: After exhausting all search strategies (primary + fallback A/B/C), if you genuinely found no matching issue, set `has_known_issue = False`. Be thorough — the goal is to avoid duplicate issue reporting.

## Strict Constraints (ZERO TOLERANCE)

1. **NO Pull Requests**: You MUST append `is:issue` to every `gh search issues` command. PRs are strictly invalid evidence. For `webfetch` fallback, the GitHub issue listing URL naturally filters to issues.
2. **Explicit Evidence**: The `DetailReason` MUST contain the literal Issue URL as shown in the Synthesis table.
3. **Tool Preference**: Use `bash` (with `gh` CLI) as your primary tool. Prefer `webfetch` over `websearch` for fallback because `webfetch` retrieves a specific known URL with deterministic results, whereas `websearch` uses opaque search indexing that may miss body-level content.
4. **No Blind Copies**: Output `DetailReason` MUST NOT simply repeat the input's `DetailReason`.
5. **No Excel Reads**: Never read the input Excel file directly. Rely only on the passed task parameters.
6. **Search Must Actually Succeed Before Concluding 'No Results'**: If both primary and fallback paths fail or produce no parseable output, the skill MUST report an error rather than returning `has_known_issue = False` with certainty. Use `has_known_issue = False` with `confidence = "Low"` and note in `DetailReason` which search methods failed.