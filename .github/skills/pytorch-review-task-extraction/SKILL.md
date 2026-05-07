---
name: pytorch-review-task-extraction
description: >
  Extract actionable tasks from code review comments.
  Used by private_review agent to parse reviewer feedback into a task list.
---

# Review Task Extraction

## When to use
Called by `private_review.py` to convert code review comments into a numbered task list.

## Instructions
Extract concise, actionable tasks from the following code review.

Return ONLY a numbered list (1. 2. 3. ...), one task per line.
Each task should be a concrete action the developer must take.

Do NOT include section headers, verdicts, analysis, or explanations —
only actionable items.

## Example Output
```
1. Replace hardcoded timeout with config constant MAX_CI_TIMEOUT
2. Add error handling for the case where PR is already merged
3. Fix typo in function name: `get_isue_detail` → `get_issue_detail`
```
