# GitHub tool reference for this skill

This skill assumes GitHub Copilot can search issues, pull requests, commits, and changed files. Keep this workflow read-only. Issue creation belongs to the separate `xpu-ops-issue-creation` skill.

## Useful GitHub capabilities
- `search_issues` — Find closed issues and PRs using GitHub search syntax.
- `issue_read` — Fetch issue body, comments, and labels.
- `search_pull_requests` — Search merged PRs directly.
- `pull_request_read` — Read PR summary, files, diff, comments, and reviews selectively.
- `list_commits` and `get_commit` — Walk from PR to the landing commit and inspect the exact fix.
- `get_file_contents` — Read only the regression test or source file touched by the PR.

If the current Copilot environment does not expose these GitHub tools, stop and tell the user what upstream search still needs to be done manually.

## Query patterns that work well
### Issues
- `repo:pytorch/pytorch is:issue is:closed CUDA crash label:module: cuda`
- `repo:pytorch/pytorch is:issue is:closed CUDA "non-contiguous"`
- `repo:pytorch/pytorch is:issue is:closed CUDA "empty tensor"`
- `repo:pytorch/pytorch is:issue is:closed CUDA "incorrect result"`

### Pull requests
- `repo:pytorch/pytorch is:pr is:merged "fix cuda"`
- `repo:pytorch/pytorch is:pr is:merged "add cuda test"`
- `repo:pytorch/pytorch is:pr is:merged "incorrect on cuda"`
- `repo:pytorch/pytorch is:pr is:merged "non-contiguous" "cuda"`

### Commit-focused follow-up
After identifying a PR, inspect:
1. changed tests first
2. then the narrowest operator implementation file
3. then only enough diff to infer the bug family

## Practical narrowing heuristic
1. Find a merged PR with a regression test.
2. Read the PR summary and changed test names.
3. Read the linked issue if the bug scenario is not obvious.
4. Read the landing commit diff only for the touched operator/test files.
5. Synthesize one minimal XPU repro.

## Avoid these traps
- Do not treat every CUDA fix as an XPU candidate.
- Do not mine huge compiler or infra PRs first.
- Do not read whole-file diffs when a single test name gives the bug shape.
- Do not mix issue filing into this skill.
