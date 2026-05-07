---
name: pytorch-cuda-fix-xpu-alignment
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, adapt upstream reproducers for XPU, and validate on local XPU nightly. Use when aligning upstream backend fixes for XPU parity.
---

# PyTorch CUDA-Fix XPU Alignment

Scan `pytorch/pytorch` for upstream backend bugs that may also affect XPU. Adapt upstream reproducers, validate locally on XPU, route confirmed bugs, and keep the scan auditable and resumable.

## Rules

1. GitHub MCP first, `gh` CLI as fallback. Never hardcode tokens.
2. Use the caller-provided or currently selected XPU-capable Python environment; keep the XPU nightly fresh and upgrade it if stale.
3. Persist scan state locally. `artifacts/candidate_ledger.jsonl` is the source of truth; raw candidates, fetched details, repro outputs, and final reports must be written to disk. On resume, skip ledger rows already marked done.
4. Paginate exhaustively. If the result set is too large, split the query window instead of silently truncating.
5. Zero pending actionable rows = done. Otherwise write `## Progress checkpoint`.

## Workflow

### Step 0: Preflight

- Verify the selected interpreter can import `torch` and reports `torch.xpu.is_available()`.
- Check the nightly freshness from `torch.__version__`; if it is stale, upgrade in that same environment.
- Verify GitHub access through MCP or `gh`.
- Create the local run directories and save `collect_env` output under `artifacts/`.

### Step 1: Collect candidates

- Search `pytorch/pytorch` issues, merged PRs, and fix-related commits for the caller-specified time window.
- Use broad bug and fix signals; adjust keywords as needed for coverage, but do not rely on a single narrow query.
- Paginate through the full result set. If you approach GitHub's search caps, split the date range and continue.
- Save raw metadata locally to `artifacts/raw_candidates.json`.
- Deduplicate overlapping hits, preferring the richer source when a PR and one of its commits describe the same fix.

### Step 2: Triage

- Initialize `artifacts/candidate_ledger.jsonl` from the raw candidates.
- Reject only when the title or fetched details clearly show the case is irrelevant, non-bug, or platform-exclusive.
- When in doubt, keep the candidate for deeper review instead of over-filtering early.

### Step 3: Reproduce

- Fetch details for non-rejected candidates and save them under `artifacts/details/`.
- Prefer upstream reproducers or regression tests; adapt them to XPU instead of inventing a new repro unless necessary.
- Each repro script must print a machine-parseable result line (e.g., `RESULT: <bucket>`) so outcomes can be collected programmatically.
- Write local repro scripts and capture local outputs when a repro is feasible.
- If a meaningful repro cannot be constructed, record a blocked result in the ledger and move on.

### Step 4: Execute and classify

- Run reproducers one at a time with a timeout to isolate crashes and hangs.
- Confirm the operation actually ran on XPU rather than silently falling back to CPU.
- Update the ledger with a concise outcome such as `confirmed`, `related-failure`, `not-reproduced`, or a `blocked-*` result.

### Step 5: Route and report

- Route shared-code bugs to `pytorch/pytorch`.
- Route XPU backend gaps or non-upstream XPU kernels to `intel/torch-xpu-ops`.
- Generate `reports/full_scan.md` from locally validated candidates.

### Step 6: Audit

- Audit the local ledger and report for completeness.
- `actionable = title_status == "pass" && deep_status != "reject" && local_status == "pending"`
- If actionable rows remain, write `## Progress checkpoint` with what is left.
- If actionable rows are zero, write `## Final Summary` with filter, validation, and routing stats.

## Reference

### CUDA-to-XPU Mapping

| CUDA | XPU |
|------|-----|
| `torch.cuda.*` | `torch.xpu.*` |
| `"cuda"` / `"cuda:0"` | `"xpu"` / `"xpu:0"` |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` |
| `torch.backends.cudnn.*` | `torch.backends.mkldnn.*` |
| cuDNN / NCCL | oneDNN / oneCCL |
| `CUDA_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` |

### Minimum Ledger Fields

| Field | Purpose |
|-------|---------|
| `id` | issue/PR number or short SHA |
| `kind` | `issue` / `pr` / `commit` |
| `title_status` | early reject or pass |
| `deep_status` | pending / pass-to-repro / reject |
| `local_status` | pending / done |
| `local_bucket` | local outcome summary |
| `target` | `pytorch/pytorch` / `intel/torch-xpu-ops` / `null` |

### Example Outcome Buckets

- `confirmed`
- `related-failure`
- `not-reproduced`
- `blocked-env`
- `blocked-platform`
- `blocked-script-error`
- `blocked-fetch`

### Confirmation Criteria

Sufficient: crash, segfault, assertion failure, hang, wrong numerical result, wrong shape/stride/dtype, off-by-one beyond atol=1e-4.

Not sufficient: tiny float noise within tolerance, documented unsupported behavior, invalid repro setup.

## Guardrails

- Do not file issues from this skill.
- `confirmed` requires a local run reproducing the issue.
- Never hardcode GitHub tokens.
- Never treat missing pagination as success.
- Do not reject candidates merely because they mention cuDNN or NCCL — these map to oneDNN/oneCCL on XPU and bugs in those paths are often relevant.
- Keep raw candidates, the ledger, fetched details, repro outputs, and the final report on disk for auditability.
