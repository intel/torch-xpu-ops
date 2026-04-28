---
name: pytorch-cuda-xpu-triage
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, generate reproducers, and validate on local XPU nightly. Use when triaging upstream backend fixes for XPU parity.
---

# PyTorch Backend-Fix to XPU Triage

Scan `pytorch/pytorch` for issues, PRs, and commits across any backend (CUDA, ROCm, CPU, etc.), generate minimal reproducers, and run them on the latest local XPU torch nightly to determine whether XPU shares the same bug.

Detailed reference:
- [references/github-mcp-reference.md](references/github-mcp-reference.md) — search patterns and narrowing workflow
- [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) — environment, run commands, confirmation criteria

## Workflow

### Step 1: Search upstream signals
- Search issues, PRs, and commits in `pytorch/pytorch` for backend bug-fix signals — both open and closed/merged. Sources may be CUDA-specific, ROCm-specific, or cross-backend — the key question is whether the bug pattern could also manifest on XPU.
- Default time window: most recent 1 day; widen to 7 days if needed. Collect **all** matching candidates in the window, not just the first few.
- Follow links between issues, PRs, and commits to narrow down the exact bug trigger (operator, shape, dtype, edge case).

### Step 2: Qualify candidates
For each candidate, classify as **qualified** or **rejected**.

Qualify when:
- The change adds or modifies regression tests
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
3. **Reproducer** — standalone Python script that runs on `torch.xpu` and compares against CPU reference
4. **Validation plan** — exact run command, expected outcome, what to capture

### Step 4: Run on local XPU nightly
- Ensure the latest XPU torch nightly is installed (`pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/xpu`).
- Run each reproducer script locally on XPU hardware.
- Keep scripts minimal and deterministic (one op family, seeded randomness, CPU vs XPU comparison).
- Print `torch.__version__` and `torch.xpu.is_available()` at the top.
- If shell access is unavailable, return copy-paste commands and specify what evidence to paste back.

## Guardrails
- Do not file issues from this skill.
- Do not claim a bug exists on XPU until a local run confirms it.
- Do not force a reproducer when evidence already shows rejection is appropriate.
