# Batch Scan Workflow

Procedures for full scan and daily scan modes.

## Environment Setup

Before any batch scan, ensure both repositories are accessible:
- `intel/torch-xpu-ops`
- `pytorch/pytorch`

Use existing local checkouts, GitHub MCP tools, or clone as needed. If a repo is unreachable, proceed with whichever is available and note the gap.

### Enumerate all XPU operators

Collect operators from these sources across both repos:
- `torch-xpu-ops/yaml/xpu_functions.yaml` — supported op list
- `torch-xpu-ops/src/ATen/native/xpu/` and `torch-xpu-ops/src/ATen/native/xpu/sycl/` — kernel implementations (`TORCH_IMPL_FUNC`)
- `torch-xpu-ops/src/ATen/native/xpu/XPUFallback.template` — fallback op list
- `pytorch/aten/src/ATen/native/xpu/` — upstream XPU native code
- `pytorch/aten/src/ATen/native/native_functions.yaml` — ops with XPU dispatch keys

Merge into a single list deduplicated by exact schema name. Record which sources each op came from.

## Full Scan Workflow

Triggered by: `scan all`

1. Check the workspace root for any existing `xpu_scan_full_*.json` with `"scan_status": "interrupted"`. If found, resume from where it left off — read the file, skip already-completed ops, and continue. Do not re-create the file.
2. If no interrupted scan exists, run Environment Setup, enumerate all ops, and create `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` and matching `.md` in the workspace root.
3. For each operator, execute the Per-Operator Analysis (SKILL.md Step 1→3) at full depth. After each op, **immediately** append results to both JSON and Markdown. Update progress in JSON.
4. After all ops are processed, append summary statistics and set `"scan_status": "completed"`.

## Summary Statistics

Every scan report (full or daily) must end with concrete counts:

- **Total**: N ops scanned
- **By verdict**: XPU defect, Parity gap, Missing native implementation, Fallback only, Needs review, No issue — each with count
- **By priority**: high, medium, low — each with count
- **By coverage type**: native kernel, composite/decomp, fallback only, no coverage — each with count
- **By target repo**: `pytorch/pytorch` (count), `intel/torch-xpu-ops` (count) — where the fix should land
- **Top reasons** (descending by count): list the most frequent finding reasons with their counts, e.g. "missing backward: 15, silent CPU fallback: 8, dtype gap: 5"

These numbers are the basis for any future triage decisions. Do not summarize with prose alone — always provide the raw counts.

## Daily Scan Workflow

Triggered by: `scan daily` or `scan since <date>`

1. Run Environment Setup (clone or pull).
2. Use `git log --since='<window>' --name-only` in both repos to find changed files under XPU-related paths (`yaml/`, `src/ATen/native/xpu/`, `aten/src/ATen/native/xpu/`, `test/xpu/`).
3. Map changed files to affected operators — extract op names from diffs, registration macros, and YAML entries.
4. Create `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` and `.md`.
5. Process affected ops same as full scan step 3-4.

## Output Files

- Full scan: `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`
- Daily scan: `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`

**Never overwrite previous scan files.** Each run produces its own timestamped files. All history is preserved.

JSON contains: scan metadata, per-operator results (schema, verdict, priority, signals, evidence, next action), and summary statistics. Markdown mirrors this as a human-readable report.
