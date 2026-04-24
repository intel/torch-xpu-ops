# GitHub MCP reference for this workflow

## Server setup
This skill uses GitHub MCP for all GitHub metadata access.
The hosted GitHub MCP endpoint authenticates via `GITHUB_PAT`.

Required token scope: `repo`
Optional: `read:org` for org metadata

Do not commit a token into any repository.
Use one of these local-only options instead:
- `export GITHUB_PAT` in your shell before starting the agent
- store it in a local `.env` that stays gitignored and source it in your shell

Remote GitHub MCP is preferred here because it avoids a local Docker dependency,
exposes issue, PR, commit, and code search tools in one endpoint, and can be
scoped per agent in your OpenCode configuration.

## Most useful GitHub MCP capabilities for CUDA-fix mining
- `search_issues` — Find closed issues and PRs using GitHub search syntax.
  - `repo:pytorch/pytorch is:issue is:closed CUDA incorrect result`
  - `repo:pytorch/pytorch is:pr is:merged fix cuda non-contiguous`
  - `repo:pytorch/pytorch is:pr is:merged "incorrect on cuda"`
- `issue_read` — Fetch issue body, comments, sub-issues, and labels.
- `search_pull_requests` — Search merged PRs directly with GitHub syntax.
- `pull_request_read` — Use get, get_files, get_diff, get_comments, and get_reviews selectively.
- `list_commits` and `get_commit` — Walk from PR to the landing commit and inspect the exact fix.
- `get_file_contents` — Read only the specific regression test or source file touched by the PR.
- `issue_write` — Create the final XPU issue once local validation is complete.

## Duplicate search for intel/torch-xpu-ops
Before creating an issue, search the target repo directly.

Use strong anchors first:
- exact upstream PyTorch issue URL
- exact upstream PR URL
- exact upstream commit SHA or commit URL
- plain bug statement without the `[ai_generated]` prefix

Then fall back to semantic search:
- `repo:intel/torch-xpu-ops is:issue is:open xpu non-contiguous`
- `repo:intel/torch-xpu-ops is:issue is:open xpu empty tensor`
- `repo:intel/torch-xpu-ops is:issue is:open xpu dtype promotion`
- `repo:intel/torch-xpu-ops is:issue is:open xpu reduction mismatch`
- `repo:intel/torch-xpu-ops is:issue is:open xpu masked scatter`

When filing, prefer `issue_read` on the strongest duplicate candidate before
deciding whether to create a new report. Also check recently closed issues when
the anchor search already matches the same upstream issue, PR, or commit.

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
- Do not create issues before local validation.
