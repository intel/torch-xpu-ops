---
name: xpu-backend-scan
description: Analyze whether a PyTorch operator has correct XPU support. Inspects dispatch coverage, CUDA-XPU parity, fallback status, and missing implementation evidence. Reports all findings with priority — does not dismiss or waive anything.
---

# XPU Backend Scan

Analyze whether a PyTorch operator has correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

Scope: all XPU-related code in both `pytorch/pytorch` (upstream XPU backend, dispatch, codegen, tests) and `intel/torch-xpu-ops` (XPU kernels, SYCL implementations, fallback, backend YAML).

References:
- [references/dispatch-coverage.md](references/dispatch-coverage.md) — how to determine XPU coverage and where to look
- [references/triage-patterns.md](references/triage-patterns.md) — pattern context for prioritization

## Scan Modes

This skill supports three invocation modes:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Single op** | Operator name/schema | Analyze one or more specific operators (default) |
| **Full scan** | `scan all` | Deep-analyze ALL XPU operators across both repos |
| **Daily scan** | `scan daily` or `scan since <date>` | Deep-analyze ops affected by recent changes |

## Input / Output

**Mode A — Single op analysis**

- **Input**: One or more operator names, schemas, or overloads to analyze.
- **Output**: Per-operator finding with verdict, priority, evidence, coverage signals.

**Mode B — Full scan**

- **Input**: `scan all` (optionally specify output directory)
- **Output**: `xpu_scan_full_<date>.json` + `xpu_scan_full_<date>.md` containing deep analysis of every XPU operator found across both repos.

**Mode C — Daily scan**

- **Input**: `scan daily` or `scan since <date>` (optionally specify output directory)
- **Output**: `xpu_scan_daily_<date>.json` + `xpu_scan_daily_<date>.md` containing deep analysis of operators affected by recent changes.

**Important**: Report everything. Do not dismiss or waive any finding. Assign priority instead.

## Workflow

### Step 1: Identify the operator surface

Pin the exact schema/overload from `native_functions.yaml` (torch-xpu-ops `yaml/native/` or upstream `aten/src/ATen/native/`). Do not compare by base op name or filename alone.

### Step 2: Collect ALL XPU coverage signals

Check every source below and record what you find. Do not stop at the first positive signal — collect all of them:
1. Backend YAML with explicit XPU dispatch keys (native XPU path)
2. Source-backed registration (`TORCH_IMPL_FUNC`, landed implementation in `src/ATen/native/xpu/sycl/`)
3. Structured delegate or codegen path resolving to XPU
4. Composite (`CompositeImplicitAutograd`/`CompositeExplicitAutograd`) or decomposition
5. `XPUFallback.template` — explicit per-op fallback (CPU fallback, callable but not native XPU)

See [references/dispatch-coverage.md](references/dispatch-coverage.md) for where to look and how to interpret each signal.

### Step 3: Classify the finding

- **XPU defect** (high) — intrinsic problem in XPU code: broken dispatch, silent CPU fallback, missing validation, backward gap, race condition
- **Parity gap** (high) — both CUDA and XPU have coverage, but user-visible contract differs: input space, parameter semantics, dtype support, backward behavior, error paths
- **Missing native implementation** (high) — CUDA has usable support, XPU has no native path (no backend YAML, no source-backed kernel, no composite/decomp)
- **Fallback only** (low) — op is callable only via CPU fallback in `XPUFallback.template`; XPU lacks a native GPU implementation
- **Needs review** (medium) — mixed evidence, cannot conclude without runtime validation

### Step 4: Validate before concluding

- Require user-visible contract difference, not just implementation shape difference.
- Read helper definitions before comparing call sites.
- Distinguish family-level truth from row-level truth.
- Do not call a finding runtime-confirmed from static review alone.

## Hard Rules

- If CUDA has a feature that XPU lacks, report it.
- If XPU silently falls back to CPU, report as XPU defect.
- If XPU only has CPU fallback coverage, report as "fallback only" (low priority).
- SYCL vs CUDA style differences are not bugs.
- Vendor library choice (oneDNN vs cuDNN) is not itself a bug.
- Missing local XPU kernel file is not evidence of missing support — check delegates, composites, shared paths first.
- Test skip/xfail metadata is a signal worth reporting, not dismissing.

## Output

Return findings grounded in inspected code:
- Exact schema/overload under review
- Verdict with priority (high / medium / low)
- All coverage signals found (list each one)
- XPU-side evidence (files, code paths)
- Peer evidence (CUDA, upstream, shared paths)
- Next action (hand off for repro, needs runtime check, or informational only)

---

## Environment Setup (Full Scan / Daily Scan)

Before running a batch scan, prepare the environment:

1. **Clone intel/torch-xpu-ops** (if not already present):
   ```bash
   git clone --depth=1 https://github.com/intel/torch-xpu-ops.git torch-xpu-ops
   ```

2. **Clone pytorch/pytorch** (if not already present):
   ```bash
   git clone --depth=1 https://github.com/pytorch/pytorch.git pytorch
   ```

3. **Enumerate all XPU operators** from these sources (union, no dedup at enumeration):
   - `torch-xpu-ops/yaml/xpu_functions.yaml` — supported op list (~741 ops)
   - `torch-xpu-ops/src/ATen/native/xpu/` — all kernel source files (extract op names)
   - `torch-xpu-ops/src/ATen/native/xpu/XPUFallback.template` — fallback op list
   - `pytorch/aten/src/ATen/native/xpu/` — upstream XPU native code
   - `pytorch/aten/src/ATen/native/native_functions.yaml` — ops with XPU dispatch keys

4. Merge into a single ordered op list. This is the scan target.

---

## Full Scan Workflow

Triggered by: `scan all`

### F1: Setup
Run Environment Setup above. Record repo paths in the output JSON.

### F2: Create output files
Create `xpu_scan_full_<date>.json` and `xpu_scan_full_<date>.md` in the working directory.

Initialize JSON:
```json
{
  "scan_date": "<date>",
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
For each operator in the merged list, execute the **Single Op Workflow** (Step 1 → Step 4) at full depth:
- Read source code, compare CUDA/XPU implementations, check all coverage signals
- After completing each op, **immediately** append its result to both JSON (`operators` array) and Markdown
- Update `next_op_index` in JSON after each op

Per-operator JSON entry:
```json
{
  "schema": "aten::addmm",
  "status": "completed",
  "verdict": "Fallback only",
  "priority": "low",
  "signals": ["xpu_fallback_template"],
  "evidence": {
    "xpu_files": ["src/ATen/native/xpu/XPUFallback.template:L42"],
    "cuda_files": ["aten/src/ATen/native/cuda/Blas.cpp"],
    "details": "..."
  },
  "next_action": "informational only"
}
```

### F4: Auto-resume on interruption
If context window is nearly exhausted:
1. Set `"scan_status": "interrupted"` and `"next_op_index": N` in JSON
2. Write all pending data to disk
3. In the **next conversation**, read the JSON file, find `next_op_index`, skip all ops with `"status": "completed"`, and continue from where it left off

No human intervention required. The JSON file is the checkpoint.

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
      "needs_review": 15
    },
    "by_priority": {"high": 65, "medium": 15, "low": 230},
    "by_source": {
      "xpu_functions_yaml": 741,
      "kernel_source_files": 52,
      "fallback_template": 180,
      "upstream_xpu": 30
    }
  }
}
```
Set `"scan_status": "completed"`.

---

## Daily Scan Workflow

Triggered by: `scan daily` or `scan since <date>`

### D1: Setup
Run Environment Setup. Then detect changes:
```bash
# In torch-xpu-ops:
git log --since='<window>' --name-only --pretty=format: -- yaml/ src/ATen/native/xpu/

# In pytorch:
git log --since='<window>' --name-only --pretty=format: -- aten/src/ATen/native/xpu/ aten/src/ATen/native/native_functions.yaml
```

### D2: Map files to operators
From changed files, extract affected operator names:
- YAML changes → parse added/removed/modified op entries
- Source file changes → extract op names from filenames and `TORCH_IMPL_FUNC` registrations
- Fallback template changes → parse diff for added/removed ops

### D3: Create output files
Create `xpu_scan_daily_<date>.json` and `xpu_scan_daily_<date>.md`.
Same JSON schema as full scan, with `"scan_mode": "daily"`.

### D4: Process operators one by one
Same as Full Scan F3 — deep analysis per op, immediate write, auto-resume capable.

### D5: Summary
Same as Full Scan F5.

---

## Output Files

### File naming
- Full scan: `xpu_scan_full_<YYYY-MM-DD>.json` / `.md`
- Daily scan: `xpu_scan_daily_<YYYY-MM-DD>.json` / `.md`

**Never overwrite previous scan files.** Each scan produces its own dated files. All history is preserved.

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

## Findings

### aten::addmm (Fallback only — low)
- **Signals**: xpu_fallback_template
- **Evidence**: ...
- **Next action**: informational only

### aten::_scaled_dot_product_flash_attention (Missing native impl — high)
...
```

---

## Auto-Resume

The JSON file is the checkpoint. To resume an interrupted scan:

1. Read the JSON file for the scan in progress
2. Check `scan_status` — if `"interrupted"`, resume is needed
3. Read `next_op_index` to know where to continue
4. Skip all entries in `operators` array that have `"status": "completed"`
5. Continue processing from `next_op_index` onward
6. When complete, set `"scan_status": "completed"` and write summary

This mechanism is fully automatic — no human input required between interruption and resumption.
