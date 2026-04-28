---
name: pytorch-cuda-xpu-triage
description: Mine recent upstream backend fixes in pytorch/pytorch that may expose XPU parity gaps, qualify candidates, and draft minimal local reproducers. Use when scanning for CUDA/XPU/backend-divergence issues to prepare for local validation.
---

# PyTorch Backend-Fix to XPU Triage

Mine recent `pytorch/pytorch` backend-fix signals, qualify XPU-relevant candidates, and produce minimal reproducers.

Detailed reference:
- [references/github-mcp-reference.md](references/github-mcp-reference.md) — search patterns and narrowing workflow
- [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) — environment, run commands, confirmation criteria

## Workflow

### Step 1: Search upstream fixes
- Search closed issues and merged PRs in `pytorch/pytorch` for backend-divergence signals (incorrect results, device-specific crashes, dtype/stride/empty-tensor edge cases, etc.).
- Default time window: most recent 1 day; widen to 7 days only if no strong candidate found.
- Pivot from issue → PR → commit to extract the minimal bug pattern.

### Step 2: Qualify candidates
For each candidate, classify as **qualified** or **rejected**.

Prefer candidates where:
- The PR adds or modifies regression tests
- The change touches `ATen/native` kernels, dispatch logic, or shared validation code
- The bug involves edge-case semantics with a narrow trigger and clear signal
- The fix lives in code paths likely inherited by XPU

Reject when:
- The change is infra-only, compiler-only, build/packaging/CI-only
- Performance-only tuning with no regression signal or reproducer shape
- The fix does not expose a plausible XPU parity gap

For rejected candidates, state the reason briefly and move on. If all candidates are rejected, report that outcome.

### Step 3: Draft reproducers
For each qualified candidate, produce:
1. **Summary** — operator, bug family, why XPU might share the defect
2. **Evidence** — issue/PR links, commit SHA, impacted files
3. **Reproducer** — standalone Python script targeting `torch.xpu`, comparing against CPU
4. **Validation plan** — exact run command, expected outcome, what to capture

### Step 4: Local validation handoff
- Keep scripts minimal and deterministic (one op family, seeded randomness, CPU vs XPU comparison).
- Print `torch.__version__` and `torch.xpu.is_available()` at the top.
- If shell access is unavailable, return copy-paste commands and specify what evidence to paste back.

## Guardrails
- Do not file issues from this skill.
- Do not claim a bug exists on XPU until a local run confirms it.
- Do not force a reproducer when evidence already shows rejection is appropriate.
