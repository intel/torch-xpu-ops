---
name: pytorch-cuda-xpu-triage
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, adapt upstream reproducers for XPU, and validate on local XPU nightly. Use when triaging upstream backend fixes for XPU parity.
---

# PyTorch Backend-Fix to XPU Triage

Scan `pytorch/pytorch` for issues, PRs, and commits across any backend (CUDA, ROCm, CPU, etc.), adapt upstream reproducers for XPU, and run them on the latest local XPU torch nightly to determine whether XPU shares the same bug.

Detailed reference:
- [references/github-mcp-reference.md](references/github-mcp-reference.md) — search patterns and narrowing workflow
- [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) — environment, run commands, confirmation criteria

## Workflow

### Step 1: Search upstream signals
- Use **GitHub MCP** tools (search, issue/PR read, commit inspect) to search `pytorch/pytorch` for backend bug-fix signals — both open and closed/merged. See [references/github-mcp-reference.md](references/github-mcp-reference.md) for query patterns. If MCP is unavailable, fall back to `gh` CLI or manual lookup and report the gap.
- Use the time window specified by the caller (e.g., last 1 day, last 7 days, or all time). Paginate through **all** search results in the window — do not stop after the first page.
- Sources may be CUDA-specific, ROCm-specific, or cross-backend — the key question is whether the bug pattern could also manifest on XPU.
- Follow links between issues, PRs, and commits to narrow down the exact bug trigger (operator, shape, dtype, edge case).
- Save the raw scan results to a local file (e.g., `triage_scan_<date>.md`) listing every candidate with its URL, title, and status. This makes the scan auditable and re-runnable without re-querying GitHub.

### Step 2: Filter candidates
For each candidate, decide: **reject** or **pass through**.

Reject when:
- The change is infra-only, compiler-only, build/packaging/CI-only
- The change is documentation-only or typo-only with no code or test impact

Everything not rejected passes to the next step. State the reject reason briefly and move on. If all candidates are rejected, report that outcome.

Append each decision (URL, reject/pass, reason) to the local scan file so the full filtering rationale is preserved.

At the end of filtering, append a summary to the scan file: total candidates, number rejected (broken down by reject reason), and number passed through. This distribution helps refine the reject rules over time.

### Step 3: Draft reproducers
For each passing candidate, produce:
1. **Summary** — operator, bug family, why XPU might share the defect
2. **Evidence** — issue/PR links, commit SHA, impacted files
3. **Reproducer** — prefer extracting the regression test or reproducer from the upstream issue/PR/commit and adapting it to run on `torch.xpu`; only write a new script if no existing repro is available
4. **Validation plan** — exact run command, expected outcome, what to capture

Append each reproducer script (as a fenced code block) and its validation plan to the local scan file under the corresponding candidate entry.

### Step 4: Run on local XPU nightly
- Ensure the latest XPU torch nightly is installed (`pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/xpu`).
- Run each reproducer script locally on XPU hardware.
- Print `torch.__version__` and `torch.xpu.is_available()` at the top.
- Verify the operator actually ran on XPU (not CPU fallback) — e.g., check output tensor `.device` is `xpu`, or run with `TORCH_SHOW_DISPATCH_TRACE=1` and confirm XPU dispatch. If the op fell back to CPU, note it and do not count the result as XPU-validated.
- If shell access is unavailable, return copy-paste commands and specify what evidence to paste back.
- After all runs complete, update the local scan file with the final status of each candidate (confirmed / not-reproduced / unverified), the full run output (stdout/stderr), and a one-line result summary.
- Append a validation summary: total tested, number confirmed, number not-reproduced, number unverified, and number that fell back to CPU.

## Guardrails
- Do not file issues from this skill.
- Mark a candidate as **confirmed** only after a local run reproduces the issue. If local run is not possible, mark as **unverified**.
