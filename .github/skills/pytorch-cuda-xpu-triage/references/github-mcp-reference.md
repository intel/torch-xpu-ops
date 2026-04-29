# GitHub Search Reference

## Capabilities assumed
The agent environment provides GitHub search, issue/PR read, commit inspection, and file content retrieval. If unavailable, stop and report what needs manual lookup.

## Example query
```
repo:pytorch/pytorch is:issue "incorrect result" CUDA
```

Adapt keywords, filters (`is:pr`, `is:merged`, `is:open`), and backend terms (`cuda`, `rocm`, `xpu`, `device-specific`) to the specific bug family. Include both open and closed/merged items.

## Narrowing workflow
1. Find a candidate (issue, PR, or commit) with a clear bug pattern or regression test.
2. Read the summary/description and any test names involved.
3. Read linked issues or PRs if the bug scenario is unclear.
4. Inspect only the touched operator/test files in the relevant diff.
