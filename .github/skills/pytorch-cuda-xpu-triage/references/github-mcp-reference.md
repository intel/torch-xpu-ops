# GitHub Search Reference

Read-only upstream search workflow. Issue creation is out of scope.

## Capabilities assumed
The agent environment provides GitHub search, issue/PR read, commit inspection, and file content retrieval. If unavailable, stop and report what needs manual lookup.

## Example queries
```
repo:pytorch/pytorch is:issue is:closed "incorrect result" CUDA
repo:pytorch/pytorch is:pr is:merged "fix cuda"
repo:pytorch/pytorch is:pr is:merged "non-contiguous" cuda
repo:pytorch/pytorch is:pr is:merged "empty tensor"
```

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
