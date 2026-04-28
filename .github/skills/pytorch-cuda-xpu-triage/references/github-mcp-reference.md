# GitHub Search Reference

## Capabilities assumed
The agent environment provides GitHub search, issue/PR read, commit inspection, and file content retrieval. If unavailable, stop and report what needs manual lookup.

## Example queries
```
repo:pytorch/pytorch is:issue "incorrect result" CUDA
repo:pytorch/pytorch is:issue is:open "crash" cuda
repo:pytorch/pytorch is:pr "fix cuda"
repo:pytorch/pytorch is:pr is:open "non-contiguous" cuda
repo:pytorch/pytorch is:pr is:merged "empty tensor"
```

Include both open and closed/merged items — an open issue or unmerged PR with a clear reproducer is equally valid for XPU validation.

Adapt keywords to the specific bug family being investigated.

## Narrowing workflow
1. Find a merged PR with a regression test.
2. Read PR summary and changed test names.
3. Read linked issue if the bug scenario is unclear.
4. Inspect only the touched operator/test files in the landing commit diff.
5. Synthesize a minimal XPU repro from the test pattern.

## Pitfalls
- Not every CUDA fix is an XPU candidate.
- Skip large compiler/infra PRs.
- A test name alone often reveals the bug shape — avoid reading entire file diffs.
