---
name: xpu-bug-issue-filer
description: Turn a locally reproduced XPU backend bug into a clean GitHub issue for the XPU ops repository. Normalizes repro results, cross-links upstream PyTorch issue/PR/commit, drafts a structured report with ai_generated labeling and folded collect_env output, and optionally creates the issue via GitHub MCP. Use only after a local XPU nightly run confirms a real bug or semantic mismatch.
license: Apache-2.0
compatibility: Requires OpenCode with GitHub MCP configured (GITHUB_PAT set) for issue creation. A local XPU nightly run confirming the bug is required before this skill may be invoked.
metadata:
  workflow: validated-bug-to-issue
  audience: backend-triage
  author: laifenxiawucha
  version: "1.0"
---
## What I do
- Normalize local repro results into a concise issue report.
- Cross-link the upstream PyTorch issue, PR, and commit that motivated the test.
- Draft the title and body for the XPU ops repository.
- Optionally use GitHub MCP to create the issue after the content is confirmed.

## When to use me
Use this only after a local XPU nightly run has demonstrated a real bug, semantic mismatch, or missing supported behavior.

## Preconditions
Default filing target: intel/torch-xpu-ops. Search for duplicates there before drafting unless the user explicitly names a different repository.

Do not proceed to issue creation without all of the following:
- a minimal reproducer script
- a local XPU run result
- version and environment details
- at least one upstream PyTorch reference from issue, PR, or commit

## Duplicate detection policy
Use a two-stage duplicate check before drafting or creating an issue:
1. Strong-anchor search: upstream PyTorch issue URL, PR URL, commit SHA, or the plain bug statement without the `[ai_generated]` prefix.
2. Semantic search: operator name, validation/error string, short bug statement.

Treat an existing issue as a duplicate when it already tracks the same upstream reference or the same validated XPU mismatch, even if the wording differs.
Check open issues first, then recently closed issues.

## Issue structure
Use [assets/xpu_issue_template.md](assets/xpu_issue_template.md) as the structural reference.

**Title:** `[ai_generated] <plain bug statement>` — e.g.:
- XPU validation mismatch for invalid torch.native_channel_shuffle inputs
- XPU crash on empty-tensor reduction validation path
- XPU wrong result for non-contiguous masked op input

**Body sections required:**
- `### 🐛 Describe the bug`: two short sentences — what fails on XPU and which build, plus the concrete mismatch.
- `Reproducer:` — minimal runnable Python reproducer.
- `Observed output:` — actual traceback, assertion, crash symptom, or wrong-value output.
- `Additional context`:
  - Date, Build, upstream issue URL, upstream PR URL, upstream commit SHA
  - `Assisted-by: opencode: <actual-model> [GitHub-API] [collect_env]`
    (use the active model string; repository default is `github-copilot/gpt-5.4` unless overridden)
- `### Versions`: collect_env output wrapped in
  `<details><summary>Collected with torch/utils/collect_env.py</summary>` + fenced `text` block + `</details>`.

**Environment to capture:**
- torch.__version__, torch.version.git_version, platform, Python version, XPU availability
- oneAPI or driver/runtime details if already present in the local environment output
- collect_env output via `scripts/run_collect_env.sh`

See [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) for bug confirmation criteria.

## Issue creation workflow
1. Run the duplicate detection policy and stop early if the issue already exists.
2. Draft the issue body in markdown.
3. Ask for confirmation only if the workflow requires a final human review.
4. Create the issue with the `ai_generated` label.
5. If the create tool cannot set labels directly, add the `ai_generated` label immediately after creation.
6. After creation, return the issue URL and a one-line rationale.

## Guardrails
- Do not file duplicates if a substantially identical open issue already exists; search first.
- Do not refile a bug that already has an open or recently closed issue with the same upstream issue, PR, or commit anchor unless the user explicitly asks for a follow-up.
- Do not overfit the root cause. State observations and likely operator family.
- If the behavior is unsupported by design, do not file as a bug. Instead, note why it is unsupported.
- If the repro is flaky, mark it clearly and avoid issue creation until stabilized.
