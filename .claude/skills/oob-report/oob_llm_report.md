<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# HuggingFace OOB LLM — Report Generation

Guide for generating per-model T1/T2/R analysis reports from LLM profiling data
collected via `skills/oob_llm_profile.md`.

**Prerequisites**: Profiling data already collected for target models on at least one platform
(B70/4080S). Each model needs: `baseline/*.json` (T2), `calcflops/*_calcflops.txt` (T1),
and `profiler/*/trace.json` (profiler trace).

**Working directory**: All commands must be run from the repo root
(`$OOB_REPO_ROOT`). The script imports shared parsing functions from
`scripts/oob300/`.

---

## Platform naming

| User-facing name | Internal platform ID | CLI flag | Notes |
|-----------------|---------------------|----------|-------|
| B70 | G31 | `--b70-dir` | XPU, profiler trace only (no unitrace) |
| 4080S | 4080 | `--4080s-dir` | CUDA, profiler trace only |

Both platforms use torch profiler traces for actual GPU times. The script identifies
XPU vs CUDA by platform key (G31=XPU, 4080=CUDA), not by actual_source like OOB 300.

---

## Per-Model Report

**Script**: `scripts/oob_llm/generate_all_llm_reports.py`

### Data layout (per platform)

```
{result_dir}/
  baseline/{model_safe}/
    {model_safe}_benchmark_{timestamp}.json     # T2 from measurements.e2e_latency
  calcflops/
    {model_safe}_bs{N}_calcflops.txt            # 24-column pipe-delimited format
  profiler/{model_safe}/
    trace.json                                  # torch profiler Chrome trace
```

Where `model_safe` = model_id with `/` replaced by `_` (e.g., `google_gemma-2b`).

### T2 extraction

T2 = median of `measurements.e2e_latency` array from the baseline JSON, converted
from seconds to milliseconds. The script handles two JSON formats:
- **4080S (CUDA)**: `{hash_key: {metadata, measurements, config}}`
- **B70 (XPU)**: `{metadata, measurements, config}` (flat, no hash wrapper)

### Invocation

```bash
# Cross-platform reports (models on both platforms):
python scripts/oob_llm/generate_all_llm_reports.py \
  --4080s-dir $OOB_RESULTS_LLM_4080S \
  --b70-dir $OOB_RESULTS_LLM_B70 \
  --output-dir reports/oob_llm/per_model \
  --config config/hardware_specs.yaml

# Single model:
python scripts/oob_llm/generate_all_llm_reports.py \
  --4080s-dir ... --b70-dir ... \
  --models google_gemma-2b

# Single platform (B70 only):
python scripts/oob_llm/generate_all_llm_reports.py \
  --b70-dir $OOB_RESULTS_LLM_B70 \
  --output-dir reports/oob_llm/per_model \
  --config config/hardware_specs.yaml

# Skip already generated reports:
python scripts/oob_llm/generate_all_llm_reports.py \
  --4080s-dir ... --b70-dir ... --skip-existing
```

### Output

Reports are written to `{output_dir}/{model_safe}_bfloat16_generate.md`.

### Report structure

Uses the same 5-section format as OOB 300 reports (see `skills/oob_report_eager.md`):

1. **Summary** — T2/T1/R metrics, HW specs, cross-platform R ratio, action items, assessment
2. **Projection Quality** — Ops where R_op deviates (>1.05 or <0.80), T2 coverage by T1
3. **XPU vs CUDA Consistency** — Graph consistency (calcflops), trace comparison, shape diffs
4. **XPU vs 4080S Per-Op Efficiency** — Per-op R comparison, XPU wins/behind verdict
5. **Optimization Targets** — Ranked by T2 saving if matching 4080S efficiency

Additional LLM-specific metadata in the header:
- **Sequence length** and **Tokens generated** (from baseline config)
- **T2 metric**: median e2e_latency (forward + generate)

### Key differences from OOB 300 reports

| Aspect | OOB 300 | HF OOB LLM |
|--------|---------|-------------|
| Actual source | unitrace (XPU) / profiler (CUDA) | profiler (both) |
| XPU/CUDA identification | By actual_source field | By platform key (G31=XPU, 4080=CUDA) |
| T2 unit in source | milliseconds (in baseline.txt) | seconds (in baseline JSON, converted to ms) |
| Report filename | `{model}_bs{N}_{precision}_{test}.md` | `{model_safe}_bfloat16_generate.md` |
| Precision/mode | fp16/eval or bf16/train | bfloat16/generate |

### Platform identification (monkey-patch)

The script overrides `generate_report._find_cuda_platform` and
`generate_report._find_xpu_platforms` to identify platforms by key instead of
actual_source. This is necessary because both platforms use profiler traces,
but the OOB 300 code assumes XPU = unitrace and CUDA = profiler.

```python
_CUDA_PLATFORMS = {"4080"}
_XPU_PLATFORMS = {"B580", "G31"}
```

---

## Graph Consistency Report

**Script**: `scripts/oob300/compare_graphs.py` (shared with OOB 300)

Standalone report comparing calcflops computational graphs between CUDA and XPU.
Detects dispatch path divergence (SDPA fusion, XPU-specific overrideable ops, etc.).
This is separate from the fleet summary's Section 7, which embeds a subset of the same analysis.

### Invocation

```bash
python scripts/oob300/compare_graphs.py \
    --dir-a $OOB_RESULTS_LLM_4080S/calcflops \
    --dir-b $OOB_RESULTS_LLM_B70/calcflops \
    --label-a CUDA --label-b XPU \
    --precision bfloat16 --test generate \
    -o reports/oob_llm/graph_consistency_generate.md
```

Note: `--dir-a` and `--dir-b` point to the `calcflops/` subdirectories (not the
platform root), because `compare_graphs.py` does `os.listdir(data_dir)` for
`*_calcflops.txt` files directly in the given directory. The `--precision` and
`--test` flags only affect the report header text (not file discovery).

### Output

Report written to `reports/oob_llm/graph_consistency_generate.md`.

### Report structure

- **Fleet Summary** — # models compared, # identical, # different
- **Difference Categories** — SDPA-only, Significant (>1% diff), Minor
- **Model Scorecard** — per-model FLOPs/Mem diff %, CUDA-only/XPU-only ops
- **Significant Differences — Per-Op Detail** — for each model with >1% diff,
  shows XPU-only ops, CUDA-only ops, and ops with FLOPs/memory mismatches
- **Op Differences Across Fleet** — aggregated: which ops differ most frequently

---

## Fleet Summary Report

**Script**: `scripts/oob_llm/generate_llm_fleet_summary.py`

Generates a 7-section fleet-level summary by reusing `generate_fleet_summary.generate_fleet_report()`
from OOB 300, with LLM-specific adaptations.

### Key differences from OOB 300 fleet summary

| Aspect | OOB 300 | HF OOB LLM |
|--------|---------|-------------|
| Suite classification | YAML model lists (torchbench/timm/huggingface) | Task field from `models.yaml` (causal-lm, etc.) |
| Platform IDs | B580, G31, 4080 | G31, 4080 (no B580 for LLM) |
| Precision/mode | fp16/eval or bf16/train | bfloat16/generate |
| XPU/CUDA identification | By actual_source | By platform key (monkey-patched) |

### Invocation

```bash
python scripts/oob_llm/generate_llm_fleet_summary.py \
    --4080s-dir $OOB_RESULTS_LLM_4080S \
    --b70-dir $OOB_RESULTS_LLM_B70 \
    --config config/hardware_specs.yaml \
    --models-yaml benchmark/OOB_llm/models.yaml \
    -o reports/oob_llm/summary_generate.md
```

### Output

Report written to `reports/oob_llm/summary_generate.md`.

### Report structure (7 sections)

1. **Overall** — Geomean R per platform, model count, T2 ratios
2. **Per-Suite Geomean** — Breakdown by task type (causal-lm, etc.)
3. **Model Scorecard** — Sorted by R_G31/R_4080 ascending (worst first)
4. **Worst 10 Models** — With top gap op and diagnosis
5. **Op Priority Ranking** — Which op optimization improves fleet geomean R the most
6. **Projection Accuracy** — Overcounting, Undercounting, Uncovered Ops (per platform)
7. **Graph Consistency** — Calcflops graph comparison: CUDA vs XPU

---

## Data locations

Configure data directories in `tools/agentic_xpu/.env` (see `.env.example` for variable names).
The `--4080s-dir` and `--b70-dir` CLI flags correspond to
`$OOB_RESULTS_LLM_4080S` and `$OOB_RESULTS_LLM_B70`.

---

## Future work

- **Unitrace**: Not available for LLM benchmark currently. If added, would improve
  XPU per-op timing accuracy (no profiler overhead).
- **More models**: 41 total in models.yaml; need to collect complete 3-pass data
  for more models on both platforms.
