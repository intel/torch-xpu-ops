---
name: pytorch-cuda-fix-xpu-alignment
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, adapt upstream reproducers for XPU, and validate on local XPU nightly. Use when aligning upstream backend fixes for XPU parity.
---

# PyTorch CUDA-Fix XPU Alignment

Scan `pytorch/pytorch` for issues, PRs, and commits across any backend (CUDA, ROCm, CPU, etc.), adapt upstream reproducers for XPU, and run them on the latest local XPU torch nightly to determine whether XPU shares the same bug.

Detailed reference:
- [references/local-xpu-validation-reference.md](references/local-xpu-validation-reference.md) — environment, device mapping, run commands, confirmation criteria

## Workflow

### Step 1: Collect candidate list
- Use **GitHub MCP** tools (search, issue/PR read, commit inspect) to search `pytorch/pytorch` for backend bug-fix signals — both open and closed/merged. Example query: `repo:pytorch/pytorch is:issue "incorrect result" CUDA`. Adapt keywords, filters (`is:pr`, `is:merged`, `is:open`), and backend terms to the specific bug family. If MCP is unavailable, fall back to `gh` CLI or manual lookup and report the gap.
- Use the time window specified by the caller (e.g., last 1 day, last 7 days, or all time). Paginate through **all** search results in the window — do not stop after the first page.
- Sources may be CUDA-specific, ROCm-specific, or cross-backend — the key question is whether the bug pattern could also manifest on XPU.
- Save the raw scan results to a local file (e.g., `triage_scan_<date>.md`) listing every candidate with its URL, title, and date. Only collect lightweight metadata at this stage — do not read full issue/PR bodies yet.

### Step 2: Process candidates one at a time
For each candidate in the list, perform the following sequence before moving to the next candidate. This keeps context usage minimal and allows resuming if interrupted.

#### 2a. Read & Filter
- Read the candidate's details (description, linked PRs/commits, test names).
- Decide: **reject** or **pass through**.
- Reject when:
  - The change is infra-only, compiler-only, build/packaging/CI-only
  - The change is documentation-only or typo-only with no code or test impact
- Append the decision (URL, reject/pass, reason) to the local scan file.
- If rejected, move to the next candidate immediately.

#### 2b. Draft reproducer
- Follow links to narrow down the exact bug trigger (operator, shape, dtype, edge case).
- Produce:
  1. **Summary** — operator, bug family, why XPU might share the defect
  2. **Evidence** — issue/PR links, commit SHA, impacted files
  3. **Reproducer** — prefer extracting the regression test or reproducer from the upstream issue/PR/commit and adapting it to run on `torch.xpu`; only write a new script if no existing repro is available
  4. **Validation plan** — exact run command, expected outcome, what to capture
- Append the reproducer (as a fenced code block) and validation plan to the scan file under this candidate's entry.

#### 2c. Run on local XPU nightly
- Ensure the latest XPU torch nightly is installed (`pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/xpu`).
- Run the reproducer script locally on XPU hardware.
- Print `torch.__version__` and `torch.xpu.is_available()` at the top.
- Verify the operator actually ran on XPU (not CPU fallback) — e.g., check output tensor `.device` is `xpu`, or run with `TORCH_SHOW_DISPATCH_TRACE=1` and confirm XPU dispatch. If the op fell back to CPU, note it and do not count the result as XPU-validated.
- If shell access is unavailable, return copy-paste commands and specify what evidence to paste back.
- Update the scan file with: final status (confirmed / not-reproduced / unverified), full run output (stdout/stderr), and a one-line result summary.

#### 2d. Route (confirmed only)
For each confirmed bug, determine the target repository for the fix:
- Search `pytorch/pytorch` under `aten/src/ATen/native/xpu/` for the affected op's XPU kernel.
- If found → `target: pytorch/pytorch` (the XPU implementation lives upstream).
- If not found → `target: intel/torch-xpu-ops` (the XPU kernel lives in this repo, or needs to be added here).
- If the bug is in shared dispatch/core infra (not XPU-specific code) → `target: pytorch/pytorch`.
- Record the target repo and reasoning in the scan file.

### Step 3: Summarize
After all candidates have been processed, append a final summary to the scan file:
- **Filter statistics** — total candidates, number rejected (broken down by reject reason), number passed
- **Validation statistics** — total tested, number confirmed, number not-reproduced, number unverified, number that fell back to CPU
- **Routing statistics** — of confirmed bugs: number targeting `pytorch/pytorch`, number targeting `intel/torch-xpu-ops`

## Guardrails
- Do not file issues from this skill.
- Mark a candidate as **confirmed** only after a local run reproduces the issue. If local run is not possible, mark as **unverified**.
