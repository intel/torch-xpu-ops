---
name: pytorch-cuda-fix-xpu-alignment
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, adapt upstream reproducers for XPU, and validate on local XPU nightly. Use when aligning upstream backend fixes for XPU parity.
---

# PyTorch CUDA-Fix XPU Alignment

Scan `pytorch/pytorch` for issues, PRs, and bug-fix commits across any backend (CUDA, ROCm, CPU, MPS) that may also affect XPU due to shared code paths. Adapt upstream reproducers for XPU, validate locally, route confirmed bugs, produce an auditable report.

## Rules

1. GitHub MCP first, `gh` CLI as fallback. Never hardcode tokens.
2. Workspace-local XPU interpreter: `./.conda-xpu-triage/bin/python`. If missing, create it in-workspace; never use interpreters outside this workspace.
3. Preflight checks nightly version; reinstall if stale or missing to ensure the latest nightly.
4. Zero pending actionable rows = done. Otherwise write `## Progress checkpoint`.
5. **Batched pipeline**: batch deep-filter → batch write repros → serial execution → batch route → batch report. Ledger enables resume from any interruption.

## Key Tables

### CUDA-to-XPU Mapping

| CUDA | XPU |
|------|-----|
| `torch.cuda.*` | `torch.xpu.*` |
| `"cuda"` / `"cuda:0"` | `"xpu"` / `"xpu:0"` |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` |
| `torch.backends.cudnn.*` | `torch.backends.mkldnn.*` |
| cuDNN / NCCL | oneDNN / oneCCL |
| `CUDA_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` |

### Ledger Schema (`artifacts/candidate_ledger.jsonl`)

| Field | Values |
|-------|--------|
| `id` | issue/PR number or short SHA |
| `kind` | `issue` / `pr` / `commit` |
| `title_status` | `pass` / `reject` |
| `deep_status` | `pending` / `pass-to-repro` / `reject` |
| `local_status` | `pending` / `done` |
| `local_bucket` | one of the buckets below |
| `target` | `pytorch/pytorch` / `intel/torch-xpu-ops` / `null` |

Actionable = `title_status == "pass" && deep_status != "reject" && local_status == "pending"`

### Bucket Vocabulary

| Bucket | Meaning |
|--------|---------|
| `confirmed` | same bug reproduces on XPU |
| `related-failure` | XPU fails differently on same scenario |
| `not-reproduced` | upstream failure does not reproduce |
| `blocked-env` | missing dependency or distributed setup |
| `blocked-platform` | XPU lacks required path |
| `blocked-fetch` | cannot fetch issue details |
| `blocked-commit-context` | commit lacks enough context for repro |
| `blocked-script-error` | repro failed before reaching oracle |
| `needs-performance-harness` | perf-only, needs benchmark |
| `not-applicable` | rejected before validation |

### Confirmation Criteria

Sufficient: crash, segfault, assertion failure, hang, wrong numerical result, wrong shape/stride/dtype, off-by-one beyond atol=1e-4.

Not sufficient: tiny float noise within tolerance, documented unsupported behavior, invalid repro setup.

## Workflow

### Step 0: Preflight

Verify XPU torch import and `torch.xpu.is_available()`, verify GitHub API access, create output directories (`artifacts/details`, `reports`, `scripts`), and save `collect_env` output.

### Step 1: Collect candidates

Search `pytorch/pytorch` using GitHub MCP (fall back to `gh` CLI). Use caller-specified time window. Paginate with `per_page=100`; split date ranges if hitting the 1000-result cap.

**Source types:**

1. **Issues** — `repo:pytorch/pytorch is:issue` + bug-signal keywords (`crash`, `incorrect`, `wrong`, `regression`, `segfault`, `fail`)
2. **PRs** — `repo:pytorch/pytorch is:pr is:merged` + fix-signal keywords (`fix`, `bug`, `correct`, `resolve`)
3. **Commits** — search merged commits fixing backend bugs via commit search API or from merged bug-fix PRs. Look for commits touching core paths (`aten/`, `torch/`, `c10/`) with fix-related messages.

Save to `artifacts/raw_candidates.json` (deduplicated by id, metadata only — no bodies/diffs yet). Each entry has `kind: "issue"|"pr"|"commit"`.

### Step 1.5: Title triage

Initialize ledger from raw candidates. Reject by title/message alone when it clearly indicates non-bug or platform-exclusive scope:
- `DISABLED test_` (CI noise)
- Platform prefixes: `[Windows]`, `[MPS]`, `[Build]`, `[Dependabot]`, `[RFC]`
- Docs/CI/infra/release-only keywords
- Obvious duplicates of already-processed candidates
- For commits: pure refactor/style/typo/doc commits (no functional change)

**Principle**: reject only when you're confident the title rules out XPU relevance. When in doubt, pass.

### Step 2: Batched pipeline

#### 2a. Batch deep-filter

Fetch all passed candidates' details → save to `artifacts/details/<id>.json`:
- **Issues/PRs**: fetch body, linked PRs/commits, test names
- **Commits**: fetch commit message + diff (`gh api repos/pytorch/pytorch/commits/<sha>` or `git show`). Save the diff summary and affected files.

For each, decide reject or pass-to-repro.

**Rejection principle**: reject only when the content confirms the bug is in platform-exclusive code with no XPU equivalent. Examples:
- Metal/MPS shaders, HIP driver-level, Windows linker
- CUDA allocator internals (CUDACachingAllocator), hardware-specific (device capability, RTX model)
- vLLM/distributed infrastructure with no standalone PyTorch repro available
- Commits that only touch CUDA-specific codegen templates (`torch/inductor/codegen/cuda/`) with no shared path

**Pass principle**: if the bug touches shared code (Inductor, Dynamo, autograd, dispatcher, ATen, Triton, runtime) or you're unsure, pass it through. Attempt a reproducer — that's cheaper than a false negative.

**Commit-specific**: if the diff is too small or lacks context to construct a meaningful reproducer (e.g., one-line typo fix in a comment), set `deep_status: "reject"` with reason "insufficient commit context" and move on.

#### 2b. Batch write reproducers

For all `pass-to-repro` candidates, write `scripts/repro_<id>.py` in one pass. Each repro must:
1. Print `torch.__version__` and `torch.xpu.is_available()`
2. Verify the op ran on XPU (not CPU fallback)
3. Print `RESULT: <bucket>` as the final meaningful line

**Repro source by kind:**
- **Issues**: extract the reproducer from the issue body
- **PRs**: extract from the PR description, or from the test added/modified in the PR's commits
- **Commits**: extract the regression test added in the commit (look for new `test_*` functions in `test/` files in the diff). If no test was added, construct a minimal repro from the fix diff — the "before" state is the bug.

Prefer extracting existing upstream code and adapting it (CUDA→XPU mapping above). Only write from scratch when no upstream repro exists.

#### 2c. Serial execution

Run each repro script sequentially (for crash/timeout isolation) with a timeout (~120s). Capture stdout/stderr to `artifacts/output_<id>.log`. Parse each `RESULT:` line → update ledger (`local_status: "done"`, `local_bucket`).

If tensor `.device` is `cpu`, mark `blocked-script-error` (CPU fallback, not valid XPU test).

#### 2d. Batch route

For confirmed/related-failure bugs:
- Shared code (Inductor, autograd, dispatcher, ATen, Triton, runtime) → `pytorch/pytorch`
- XPU kernel in `aten/src/ATen/native/xpu/` → `pytorch/pytorch`
- XPU kernel not upstream → `intel/torch-xpu-ops`
- Bug reveals XPU backend gap (different error, missing feature) → `intel/torch-xpu-ops`
- CPU-only crashes that affect all backends → `pytorch/pytorch`
- When in doubt → `pytorch/pytorch`

#### 2e. Batch write report

Generate `reports/full_scan.md` with all tested candidates:
```
1. `#<id>` — Title (kind: issue/pr/commit)
Summary: ...
Evidence: <url>
Local XPU result: `<bucket>`
```

### Step 3: Audit

```bash
bash scripts/audit_scan_report.sh reports/full_scan.md artifacts/candidate_ledger.jsonl
```

Write `## Final Summary` only when audit passes. Include filter stats, validation stats, routing stats.

## Guardrails

- Do not file issues from this skill.
- `confirmed` requires a local run reproducing the issue.
- Never hardcode GitHub tokens.