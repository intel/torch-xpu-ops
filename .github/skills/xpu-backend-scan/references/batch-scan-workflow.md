# Batch Scan Workflow

Procedures for full scan and daily scan modes. Read this when running `scan all`, `scan daily`, or `scan since <date>`.

## Environment Setup

Before any batch scan, prepare local clones of both repositories.

### Clone or update repos

```bash
# intel/torch-xpu-ops — use treeless clone (keeps full commit history, downloads blobs on demand)
if [[ -d torch-xpu-ops/.git ]]; then
  git -C torch-xpu-ops pull --ff-only
else
  git clone --filter=blob:none https://github.com/intel/torch-xpu-ops.git torch-xpu-ops
fi

# pytorch/pytorch — same approach
if [[ -d pytorch/.git ]]; then
  git -C pytorch pull --ff-only
else
  git clone --filter=blob:none https://github.com/pytorch/pytorch.git pytorch
fi
```

`--filter=blob:none` keeps the full commit history (required for `git log --since`) but downloads file content on demand, making the initial clone much faster than a full clone.

If clone or pull fails, log the error and proceed with whichever repo is available. Note the gap in the output.

### Enumerate all XPU operators

Collect operators from these five sources across both repos:

| # | Source | Repo | What to extract |
|---|--------|------|----------------|
| 1 | `yaml/xpu_functions.yaml` | torch-xpu-ops | All op names in `supported:` list |
| 2 | `src/ATen/native/xpu/` + `src/ATen/native/xpu/sycl/` | torch-xpu-ops | Op names from `TORCH_IMPL_FUNC` and filenames |
| 3 | `src/ATen/native/xpu/XPUFallback.template` | torch-xpu-ops | All ops in `fallback_list` |
| 4 | `aten/src/ATen/native/xpu/` | pytorch | Op names from `TORCH_IMPL_FUNC` and filenames |
| 5 | `aten/src/ATen/native/native_functions.yaml` | pytorch | Ops with XPU dispatch key entries |

**Deduplication**: Merge all sources into a single list deduplicated by exact schema name (e.g., `aten::add.Tensor`). Record which sources each op came from — this metadata goes into the per-op `signals` field. Dedup is within a single scan run only; across scan runs, no dedup — each run produces independent files.

## Full Scan Workflow

Triggered by: `scan all`

### F1: Setup

Run Environment Setup above. Record repo paths in the output JSON.

### F2: Create output files

Create `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` and the matching `.md` in the working directory. The timestamp prevents same-day collisions.

Initialize JSON:
```json
{
  "scan_date": "2026-04-29T143052",
  "scan_mode": "full",
  "scan_status": "in_progress",
  "next_op_index": 0,
  "repos": {
    "torch_xpu_ops": "<path>",
    "pytorch": "<path>"
  },
  "operators": [],
  "summary": null
}
```

### F3: Process operators one by one

For each operator in the deduplicated list, execute the **Per-Operator Analysis** (Step 1 → Step 4 in SKILL.md) at full depth:
- Read source code, compare CUDA/XPU implementations, check all coverage signals
- After completing each op, **immediately** append its result to both the JSON `operators` array and the Markdown file
- Update `next_op_index` in JSON after each op

Per-operator JSON entry:
```json
{
  "schema": "aten::addmm",
  "status": "completed",
  "verdict": "Fallback only",
  "priority": "low",
  "signals": ["xpu_fallback_template"],
  "sources": ["xpu_functions_yaml", "fallback_template"],
  "evidence": {
    "xpu_files": ["src/ATen/native/xpu/XPUFallback.template:L42"],
    "cuda_files": ["aten/src/ATen/native/cuda/Blas.cpp"],
    "details": "..."
  },
  "next_action": "informational only"
}
```

**Error handling**: If analysis of a single op fails (e.g., file not found, ambiguous schema), record `"status": "error"` with a `"error_detail"` field and continue to the next op. Do not abort the entire scan.

### F4: Auto-resume on interruption

If the context window is nearly exhausted:
1. Set `"scan_status": "interrupted"` in JSON
2. Write all pending data to disk
3. The `operators` array is the source of truth — `next_op_index` is a hint for fast seeking

To resume: the user says "resume scan" in a new conversation, pointing to the JSON file. The agent reads the file, collects all schemas with `"status": "completed"`, and continues with the next unprocessed op. No other human intervention is needed.

### F5: Summary

After all operators are processed, append summary to both JSON and Markdown:
```json
{
  "summary": {
    "total": 800,
    "by_verdict": {
      "xpu_defect": 12,
      "parity_gap": 8,
      "missing_native_impl": 45,
      "fallback_only": 230,
      "needs_review": 15,
      "error": 2
    },
    "by_priority": {"high": 65, "medium": 15, "low": 230, "error": 2},
    "coverage_distribution": {
      "native_xpu": 450,
      "composite_or_decomp": 80,
      "structured_delegate": 38,
      "fallback_only": 230,
      "no_coverage": 2
    }
  }
}
```

Set `"scan_status": "completed"`.

`coverage_distribution` categorizes by the highest-priority coverage signal found for each op. Each op appears in exactly one category, so values sum to `total`.

## Daily Scan Workflow

Triggered by: `scan daily` or `scan since <date>`

### D1: Setup

Run Environment Setup (clone or pull). Then detect changed files:

```bash
# In torch-xpu-ops:
git -C torch-xpu-ops log --since='<window>' --name-only --pretty=format: -- \
  yaml/xpu_functions.yaml yaml/native/ src/ATen/native/xpu/

# In pytorch:
git -C pytorch log --since='<window>' --name-only --pretty=format: -- \
  aten/src/ATen/native/xpu/ aten/src/ATen/native/native_functions.yaml test/xpu/
```

Default `<window>` is `1 day ago` for `scan daily`. For `scan since <date>`, use the caller-specified date.

### D2: Map changed files to operators

Use concrete commands to extract affected op names from each file type:

**YAML changes** (`xpu_functions.yaml`, `native_functions.yaml`):
```bash
git -C <repo> diff HEAD~1 -- <yaml_file> | grep '^[+-]' | grep -v '^[+-][+-]' | sed 's/^[+-]//' | tr -d ' '
```
Parse the added/removed lines to get op schemas.

**Source file changes** (`src/ATen/native/xpu/`, `sycl/`):
```bash
grep -h 'TORCH_IMPL_FUNC\|REGISTER_XPU_DISPATCH' <changed_files> | sed 's/.*(\(.*\)).*/\1/'
```
Extract op names from registration macros in the changed files.

**Fallback template changes**:
```bash
git -C torch-xpu-ops diff HEAD~1 -- src/ATen/native/xpu/XPUFallback.template | grep '^[+-].*"aten::' | sed 's/.*"\(aten::[^"]*\)".*/\1/'
```

**Test file changes** (`test/xpu/`): These do not directly map to ops. Flag the test file as context — the agent inspects which ops the test exercises.

### D3: Create output files

Create `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` and `.md`. Same JSON schema as full scan, with `"scan_mode": "daily"`.

### D4: Process operators one by one

Same as Full Scan F3 — deep analysis per op, immediate write, auto-resume capable.

### D5: Summary

Same as Full Scan F5.

## Output Files

### File naming

- Full scan: `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`
- Daily scan: `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`

**Never overwrite previous scan files.** Each scan run produces its own timestamped files. All history is preserved — no dedup across runs.

### Markdown report format

```markdown
# XPU Backend Scan — Full — 2026-04-29

## Summary
| Metric | Count |
|--------|-------|
| Total operators | 800 |
| XPU defect (high) | 12 |
| Parity gap (high) | 8 |
| Missing native impl (high) | 45 |
| Fallback only (low) | 230 |
| Needs review (medium) | 15 |
| Error | 2 |

## Findings

### aten::addmm (Fallback only — low)
- **Signals**: xpu_fallback_template
- **Sources**: xpu_functions_yaml, fallback_template
- **Evidence**: ...
- **Next action**: informational only

### aten::_scaled_dot_product_flash_attention (Missing native impl — high)
...
```
