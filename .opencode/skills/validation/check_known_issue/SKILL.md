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

### 2. Parallel Issue Search
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

### 3. Deep Context Matching
For the most promising search results, fetch the issue body to verify context:
```bash
gh issue view <number> --repo=<repo> --json title,body,state,labels,url
```
Use agent reasoning to determine if the issue actually describes this test's failure:
- **HIGH relevance**: Exact test name, exact stack trace, or exact operator in the same context.
- **MEDIUM relevance**: Similar error message, same class, or related operator.
- **LOW relevance**: Mention of the file or generic error.

### 4. Classification Synthesis
If `has_known_issue == True`, select the highest relevance match and apply this EXACT mapping:

| Issue State | Labels | `Reason` | `DetailReason` |
|-------------|--------|----------|----------------|
| OPEN | `bug` or no label | `Failures (xpu broken)` | `"Known bug: <Title> - <URL>"` |
| OPEN | `feature` / `enhancement` | `Feature gap` | `"Missing feature: <Title> - <URL>"` |
| OPEN | `skipped` | `Failures (xpu broken)` | `"Test skipped due to failure: <Title> - <URL>"` |
| CLOSED | `not_target` / `wontfix` | `Not Applicable` | `"Not applicable per closed issue: <Title> - <URL>"` |
| CLOSED | other or no relevant label | `To be enabled` | `"Issue closed, awaiting enablement: <Title> - <URL>"` |

If `has_known_issue == False`:
- `classification.Reason` = `"Submit Issue"`
- `classification.DetailReason` = `"No known issue found in pytorch/pytorch or intel/torch-xpu-ops for this test. Submit a new issue with the error details."`

## Strict Constraints (ZERO TOLERANCE)

1. **NO Pull Requests**: You MUST append `is:issue` to every `gh search issues` command. PRs are strictly invalid evidence.
2. **Explicit Evidence**: The `DetailReason` MUST contain the literal Issue URL as shown in the Synthesis table.
3. **Tool Restriction**: Use `bash` (with `gh` CLI) as your primary tool. Do not use `webfetch` or `websearch`.
4. **No Blind Copies**: Output `DetailReason` MUST NOT simply repeat the input's `DetailReason`.
5. **No Excel Reads**: Never read the input Excel file directly. Rely only on the passed task parameters.