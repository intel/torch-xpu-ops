---
name: xpu-ops-issue-creation
description: >
  How to create an intel/torch-xpu-ops issue from a validated XPU bug
  or a CI UT failure triggered by an upstream PyTorch change.
  Use when an agent already has either a confirmed local repro or
  concrete CI failure evidence and needs to check duplicates, draft
  the issue body, and optionally create the issue.
license: Apache-2.0
compatibility: Designed for agent skills under .github/skills. Works best when GitHub issue search and create tools are available; if shell access is unavailable for a manual-validation case, ask the user to run collect_env locally and paste the output.
metadata:
  workflow: validated-bug-to-issue
  audience: backend-triage
  version: "1.2"
---

# Issue Creation - torch-xpu-ops

This skill is intended for agent runtimes that load skills from `.github/skills`. Keep it text-first and avoid depending on repo-local helper scripts.

## What I do
- Normalize local repro results or CI UT failures into a concise issue report.
- Cross-link the upstream PyTorch issue, PR, and commit that motivated the test.
- Draft the title and body for the `intel/torch-xpu-ops` repository.
- Optionally use the GitHub issue tools available in the current agent environment to create the issue after the content is confirmed.

## When to use me
Use this only after either:
- a local XPU nightly run has demonstrated a real bug, semantic mismatch, or missing supported behavior, or
- a CI UT failure has been tied to an upstream PyTorch change and includes enough test and log context to act as the reproducer.

## Required tools and context
- Use the GitHub issue search, read, and create tools available in the current agent environment.
- If issue creation tools are unavailable, stop after drafting the issue body and explain the blocker.
- If shell access is unavailable for a manual-validation case, ask the user to run the environment collection command locally and paste the output.

## Preconditions
Default filing target: `intel/torch-xpu-ops`. Search for duplicates there before drafting unless the user explicitly names a different repository.

Do not proceed to issue creation without all of the following common inputs:
- a short bug statement and the affected op, module, or test area
- at least one upstream PyTorch reference from issue, PR, or commit
- failure output or a mismatch summary that is specific enough to debug

Then require one of these evidence sets:

**Manual validation path**
- a minimal reproducer script
- a local XPU run result
- version and environment details

**CI UT failure path**
- a CI job link
- the failing test identifier
- the exact or reconstructed rerun command
- the relevant traceback or log excerpt

## Duplicate detection policy
Use a two-stage duplicate check before drafting or creating an issue:
1. Strong-anchor search: upstream PyTorch issue URL, PR URL, commit SHA, or the plain bug statement without the `[ai_generated]` prefix.
2. Semantic search: operator name, validation or error string, short bug statement.

Treat an existing issue as a duplicate when it already tracks the same upstream reference or the same validated XPU mismatch, even if the wording differs.
Check open issues first, then recently closed issues.

## Issue structure
Use [assets/xpu_issue_template.md](assets/xpu_issue_template.md) as the body template.

**Title:** `[ai_generated] <plain bug statement>` — for example:
- [ai_generated] XPU validation mismatch for invalid `torch.native_channel_shuffle` inputs
- [ai_generated] XPU crash on empty-tensor reduction validation path
- [ai_generated] XPU wrong result for non-contiguous masked op input

**Body sections required:**
- `### 🐛 Describe the bug`: two short sentences — what fails on XPU and which build, plus the concrete mismatch.
- `Failure source`: choose `CI UT failure` or `Manual validation`.
- `Affected op/module`: name the likely op family, module, or failing test area.
- `### Upstream reference`:
  - upstream commit, upstream PR, upstream issue when available
  - one-line summary of what the upstream change did
- `### Failure details`:
  - `Failure type`: `new test added`, `existing test broken`, `test expectation changed`, `build break`, or another short classifier
  - Fill only the CI or manual subsection that matches `Failure source`
  - For CI UT failures: CI job link, failing test, rerun command, and the relevant log excerpt
  - For manual validation: minimal runnable Python reproducer and observed output
- `### Versions`:
  - Manual validation: collect_env output wrapped in `<details><summary>Collected with python -W ignore::RuntimeWarning -m torch.utils.collect_env</summary>` plus a fenced `text` block
  - CI UT failure: job link is sufficient by default; add collect_env only when you have local confirmation and it adds value
- `Assisted-by: opencode: [actual-model] [GitHub-API] [collect_env or CI-link]`
  - use the active model string

**Environment to capture:**
- Manual validation: `torch.__version__`, `torch.version.git_version`, platform, Python version, XPU availability
- Manual validation: oneAPI or driver/runtime details if already present in the local environment output
- Manual validation: collect_env output via `python -W ignore::RuntimeWarning -m torch.utils.collect_env`
- CI UT failure: CI job link, build identifier if visible, failing test path, rerun command, and the traceback or assertion excerpt

See [references/validation-criteria-reference.md](references/validation-criteria-reference.md) for confirmation criteria and capture requirements.

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