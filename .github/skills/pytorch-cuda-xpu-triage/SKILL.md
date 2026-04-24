---
name: pytorch-cuda-xpu-triage
description: Find recent upstream backend fixes that may expose XPU gaps and draft local repro steps. Use when GitHub Copilot is asked to scan pytorch/pytorch for CUDA, XPU, or backend-divergence issues and prepare candidates for local validation.
license: Apache-2.0
compatibility: Designed for GitHub Copilot skills under .github/skills. Works best when GitHub issue, PR, commit, and code search tools are available; local validation can be done either by Copilot with shell access or by a human using the returned commands.
metadata:
  workflow: github-copilot-to-local-xpu
  audience: backend-triage
  author: laifenxiawucha
  version: "1.1"
---

# PyTorch CUDA/XPU Triage

This skill is written for GitHub Copilot. Keep it text-first: store durable guidance in `SKILL.md` and `references/`, and do not depend on repo-local helper scripts.

## What I do
- Search `pytorch/pytorch` issues, pull requests, and commits for backend-divergence bug-fix signals.
- Pivot from issue to PR to commit and extract the minimal bug pattern.
- Rank candidates by likelihood of backend divergence affecting XPU.
- Produce concise repro plans and minimal Python reproducers for every qualified candidate, not just the top one.
- Stop before filing anything unless local XPU execution confirms a bug.

## When to use me
Use this skill when GitHub Copilot needs to mine upstream backend fixes for likely XPU parity gaps and hand back repro guidance for local validation.

Interpret user focus words as a soft hint about the desired bug family, edge condition, or semantic shape unless the user explicitly says the match must be exact.

## Required tools and context
- Use the GitHub search, issue, PR, commit, and file-reading tools available in the current Copilot environment.
- If those GitHub tools are unavailable, stop and report the blocker.
- Do not assume a custom local integration, machine-specific setup, or helper scripts.
- Only move to local validation after candidates are selected.
- If shell access is unavailable, return exact commands for the user to run locally and say what evidence to paste back.

See [references/github-mcp-reference.md](references/github-mcp-reference.md) for query patterns.
See [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) for local run commands and confirmation criteria.

## Search strategy
Start with read-only GitHub scanning in `pytorch/pytorch`. Default to the most recent 1 day. If that yields no strong candidate, widen once to the most recent 7 days. Do not widen further unless the user explicitly asks.

Look for closed issues and merged PRs that include terms such as:
- incorrect result on cuda
- incorrect result on xpu
- incorrect result on accelerator
- device divergence
- non-contiguous
- dtype promotion
- empty tensor
- scalar tensor
- reduction mismatch
- advanced indexing
- masked scatter
- NaN or inf mismatch
- autograd incorrect

Also search for merged PRs containing phrases like:
- fix cuda
- add cuda test
- add xpu test
- add accelerator test
- device-specific bug
- non contiguous
- zero size
- empty tensor
- incorrect on cuda
- incorrect on xpu
- crash on cuda
- crash on xpu

## Candidate ranking rubric
Prefer candidates that satisfy more of the following:
- The PR added or modified regression tests or benchmark coverage.
- The change touched `ATen/native` kernels, device checks, dispatch logic, or test expectations.
- The bug involves edge-case semantics or a backend-specific performance regression with a narrow trigger and clear signal.
- The issue or PR shows CPU correctness but CUDA divergence.
- The operator is implemented on multiple accelerators where semantic drift is plausible.
- The candidate preserves the user's edge condition or failure shape even if the exact operator name is different.
- The fix lives in shared frontend validation, composite fallback, common `ATen/native` code, or other code paths likely inherited by XPU.

Lower priority when dominated by:
- NVIDIA library integration specifics without semantic implications.
- CUDA graph, Triton, or compiler-only infrastructure.
- Build-system, packaging, or CI-only failures.
- Broad performance-only tuning with no clear regression signal, reproducer shape, or backend divergence evidence.

The candidate does not need to be CUDA-only. A strong fix can be motivated by CUDA, XPU, or another backend, as long as it exposes a backend semantic mismatch, validation gap, or backend-specific regression that XPU might share.

## Qualification gate
Before writing any reproducer, explicitly classify each candidate as either:
- qualified for repro generation
- rejected without repro

Do not require exact operator-name equality with the user's focus unless the user explicitly requested exact matching. A candidate can still qualify when it shares the same edge condition, semantic failure mode, or backend-divergence family.

Reject without repro when the available issue, PR, test, or diff information already shows one of the following:
- the change is obviously infra-only or compiler-only
- the change is packaging, CI, build, or tooling only
- the change is performance-only and the report provides no backend-specific regression signal, reproducer shape, or realistic validation path
- the fix is so old and generalized that the current backend stack very likely already inherited it
- the test or patch does not expose a plausible XPU parity gap

If a candidate is rejected, say why and move to the next candidate inside the active time window. If every candidate in the active time window is rejected, stop and report that outcome instead of forcing a reproducer.

## Output format
For each qualified candidate, provide:
1. Candidate summary
- operator or API surface
- bug family
- why XPU might share the defect
- why it matches the user's requested bug family or edge condition, even if the operator differs
2. GitHub evidence
- issue link and number
- PR link and number
- commit SHA
- impacted test or source file paths
3. Reproducer hypothesis
- minimal tensor shapes
- dtype and layout constraints
- device placement
- expected vs suspected actual behavior
4. Reproducer plan
- standalone Python script targeting `torch.xpu` when available
- CPU fallback comparison when useful
- any setup command the user must run first
5. Validation plan
- exact command to run locally
- what outcome counts as reproduced
- what logs or exception text to capture

Preserve the full ordered candidate list for the active window. Do not collapse the run to a single winner when multiple candidates remain qualified.

For a rejected candidate, provide instead:
1. Candidate summary
2. GitHub evidence
3. Explicit rejection reason
4. Why no reproducer should be attempted

## Local validation handoff
Keep reproducers minimal and deterministic.
- Prefer a single operator family per script.
- Set seeds when randomness exists.
- Compare CPU and XPU outputs when possible.
- Exercise the exact edge condition inferred from the upstream fix.
- Print environment info at the top: `torch.__version__`, `torch.version.git_version` if present, and `torch.xpu.is_available()` if present.
- If you cannot run locally, return copy-paste commands and say what evidence the user should paste back.

## Guardrails
- Do not file an XPU issue from this skill.
- Do not claim a bug is present on XPU until a local run shows a mismatch, crash, or unsupported behavior that should be supported.
- If the upstream fix is ambiguous, say so and produce a weaker candidate note instead of overstating confidence.
- Do not generate a reproducer for a candidate you can already reject from the available evidence.
- Do not overfit to literal focus words when a nearby bug-family match is clearly stronger for XPU parity triage.
