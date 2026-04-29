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
- `torch-xpu-ops/src/ATen/native/xpu/` and `sycl/` — kernel implementations (`TORCH_IMPL_FUNC`)
- `torch-xpu-ops/src/ATen/native/xpu/XPUFallback.template` — fallback op list
- `pytorch/aten/src/ATen/native/xpu/` — upstream XPU native code
- `pytorch/aten/src/ATen/native/native_functions.yaml` — ops with XPU dispatch keys

Merge into a single list deduplicated by exact schema name. Record which sources each op came from.

## Full Scan Workflow

Triggered by: `scan all`

1. Run Environment Setup. Enumerate all ops.
2. Create `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` and matching `.md` in the workspace root. The timestamp prevents same-day collisions.
3. For each operator, execute the Per-Operator Analysis (SKILL.md Step 1→3) at full depth. After each op, **immediately** append results to both JSON and Markdown. Update progress in JSON.
4. If context window is nearly exhausted, write `"scan_status": "interrupted"` to JSON and stop. To resume, the user says "resume scan" and provides the path to the interrupted JSON file. The agent reads it, skips completed ops, and continues.
5. After all ops are processed, append summary statistics (by verdict, by priority, by coverage type) and set `"scan_status": "completed"`.

## Daily Scan Workflow

Triggered by: `scan daily` or `scan since <date>`

1. Run Environment Setup (clone or pull).
2. Use `git log --since='<window>' --name-only` in both repos to find changed files under XPU-related paths (`yaml/`, `src/ATen/native/xpu/`, `aten/src/ATen/native/xpu/`, `test/xpu/`).
3. Map changed files to affected operators — extract op names from diffs, registration macros, and YAML entries.
4. Create `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` and `.md`.
5. Process affected ops same as full scan step 3-5.

## Output Files

- Full scan: `xpu_scan_full_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`
- Daily scan: `xpu_scan_daily_<YYYY-MM-DD>_<HHMMSS>.json` / `.md`

**Never overwrite previous scan files.** Each run produces its own timestamped files. All history is preserved.

JSON contains: scan metadata, per-operator results (schema, verdict, priority, signals, evidence, next action), and summary statistics. Markdown mirrors this as a human-readable report.
