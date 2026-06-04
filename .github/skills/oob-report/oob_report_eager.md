<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# OOB 300 Models — Report Generation

Guide for generating per-model and fleet-level T1/T2/R analysis reports from profiling data
collected via `skills/oob_profile_eager.md`.

**Prerequisites**: Profiling data already collected for target models on at least one platform
(B580/4080S/B70). Each model needs: `baseline.txt` (T2), `calcflops.txt` (T1),
`trace.json` (profiler), and optionally `unitrace.json` (XPU only).

**Working directory**: All commands must be run from the repo root
(`$OOB_REPO_ROOT`). Scripts use relative imports from `scripts/oob300/`.

### Platform naming

| User-facing name | Internal platform ID | CLI flag prefix | Notes |
|-----------------|---------------------|-----------------|-------|
| B580 | B580 | `--b580-*` | XPU, has unitrace |
| B70 | G31 | `--b70-*` | XPU, has unitrace. G31 is B70's internal platform codename |
| 4080S | 4080 | `--4080s-*` | CUDA, NO unitrace (uses profiler trace for actual times) |

Script output and reports will display the **internal ID** (G31, B580, 4080).
CLI flags use the **user-facing name** (--b70-*, --b580-*, --4080s-*).

---

## Per-Model Report

**Script**: `scripts/oob300/generate_report.py`

### Input files per platform

| File | Source step | Contains |
|------|------------|----------|
| `<model>_bs<N>_baseline.txt` | Step 1 (T2) | "CPU Wall Time per batch: X.XXX milliseconds" |
| `<model>_bs<N>_calcflops.txt` | Step 4 (T1) | Per-op cumulative FLOPs/memory, last iteration only |
| `<model>_bs<N>_trace.json` | Step 2 (profiler) | Chrome trace: host ops + device kernels linked by External id |
| `<model>_bs<N>_unitrace.json` | Step 3 (XPU only) | Unitrace: per-kernel durations, no profiler overhead |

**Note**: Result directories may also contain `<model>_bs<N>_report.txt` files —
these are legacy profiling logs and are NOT used by the report generator.

### Extracting T2 from baseline.txt

```bash
grep "CPU Wall Time per batch:" <model>_bs<N>_baseline.txt
# Output: "CPU Wall Time per batch: 245.678 milliseconds"
# Extract the float value: 245.678
```

The value is in milliseconds and is passed directly to `--<platform>-t2`.

### Invocation

```bash
python scripts/oob300/generate_report.py \
  --model <model> --bs <N> --precision fp16 --test eval \
  --b580-calcflops <path> --b580-trace <path> [--b580-unitrace <path>] --b580-t2 <ms> \
  --4080s-calcflops <path> --4080s-trace <path> --4080s-t2 <ms> \
  [--b70-calcflops <path> --b70-trace <path> --b70-unitrace <path> --b70-t2 <ms>] \
  --config config/hardware_specs.yaml \
  -o reports/oob300/per_model/eager/<model>_bs<N>_<precision>_<test>.md
```

**Why no `--4080s-unitrace`?** 4080S is CUDA — unitrace is an Intel XPU-specific
tool (`pti-gpu/unitrace`). CUDA platforms use the PyTorch profiler trace for
actual kernel times instead.

---

### Report structure (5 sections)

The script generates a deterministic 5-section report with numbered headers
(`## 1. Summary`, `## 2. Projection Quality`, etc.). All sections are fully
script-generated. AI analysis is added as a separate LLM step afterward (see
[AI Analysis](#ai-analysis-pass) below).

---

#### Section 1: Summary

Contains five sub-sections:

**1a. Model info and metrics table**

```markdown
## 1. Summary

**Model**: <model>
**Batch size**: <bs>
**Precision**: <precision>
**Mode**: <test>
**Ops per iteration**: <N>

| Metric               | B580 (XPU)   | B70 (XPU)    | 4080S (CUDA) |
|----------------------|-------------:|-------------:|-------------:|
| T2 (wall clock)      | X.XXX ms     | X.XXX ms     | X.XXX ms     |
| T1 (projection)      | X.XXX ms     | X.XXX ms     | X.XXX ms     |
| T1_compute           | X.XXX ms     | X.XXX ms     | X.XXX ms     |
| T1_memory            | X.XXX ms     | X.XXX ms     | X.XXX ms     |
| T2_device (kernel sum)| X.XXX ms    | X.XXX ms     | X.XXX ms     |
| R = T1/T2            | **X.XXX**    | **X.XXX**    | **X.XXX**    |
| Actual source        | unitrace     | unitrace     | profiler     |
| Compute-bound ops    | N            | N            | N            |
| Memory-bound ops     | N            | N            | N            |
```

**1b. Cross-Platform R Ratio**

Shows R_xpu / R_4080S for each XPU platform. >1 means XPU has better
roofline efficiency.

**1c. Hardware Specs**

Peak FP16 TFLOPS, DRAM BW, ridge point for each platform.

**1d. Action Items**

Auto-generated prioritized action table:

```markdown
### Action Items

| # | Action | Target | Op | Shape | Stride | Expected Impact | Priority |
|---|--------|--------|-----|-------|--------|----------------|----------|
| 1 | Optimize XPU kernel | torch-xpu-ops | aten::xxx | [...] | [...] | R_op ..., Save Xms (Y% T2) | High |
| 2 | Fix projection | context_func.py | aten::xxx | [...] | [...] | R_op ... | Low |
```

Action categories:
- **Optimize XPU kernel** → R_op < 0.80 on XPU but >= 0.80 on CUDA (kernel-specific problem)
- **Fix projection** → R_op < 0.80 on ALL platforms (undercounting) or R_op > 1.05 on all (overcounting)

Shape and Stride columns show the **dominant** shape/stride (the instance
contributing the most GPU time). Displayed in full — no truncation.

**1e. Overall Assessment**

One-paragraph summary: health rating (excellent/good/fair/poor based on R),
count of high-priority actions, kernel optimization targets, projection fixes,
and wall-clock comparison vs 4080S.

**Key rules**:
- R = T1 / T2 (wall clock), NOT T1 / T2_device
- T2_device is shown for reference (kernel sum from trace/unitrace)
- Cross-Platform R Ratio shows R_xpu / R_4080s — a value >1 means XPU has
  better software efficiency (closer to roofline)
- When only 2 platforms are available, omit the missing column

---

#### Section 2: Projection Quality

Combined view of ops where per-op R deviates from 1.0. Merges the old
"Projection Audit (R>1)" and "Low-R Diagnosis (R<0.8)" into one section.
Header: `## 2. Projection Quality`.

**Per-platform table** (sorted by Actual ms descending):

```markdown
| Op | R_op | Actual (ms) | % T2 | Proj (ms) | Gap (ms) | Perf | Shape | 4080S R_op | Issue |
```

Column details:
- **R_op**: Projected / Actual
- **% T2**: Actual ms / T2 — shows impact on wall clock
- **Gap**: Proj − Actual. Positive = overcounting, Negative = undercounting or slow
- **Perf**: Actual throughput — shows TFLOPS and/or BW, whatever is available
  (e.g., `2.5TFLOPS/112GB/s` for both, `159GB/s` for memory-only, `38.3TFLOPS` for compute-only).
  Vector engine ops (softmax, layer_norm, batch_norm, max_pool) have FLOPs zeroed,
  so they only show BW.
- **Shape**: Dominant input shape (full, not truncated)
- **4080S R_op**: Same op's R on 4080S — for cross-platform diagnosis
- **Issue**: Auto-classified:
  - `Overcounting` — R_op > 1.05
  - `Kernel slow` — R_op < 0.80 on this platform but >= 0.80 on CUDA
  - `Projection undercounts` — R_op < 0.80 on both this platform and CUDA
  - `Undercounts or slow` — R_op < 0.80 and no CUDA cross-reference

**T2 Coverage by T1** sub-section:

Lists ops that appear in the actual trace but have no calcflops entry (T1 = 0).
These represent uncaptured projection gaps.

```markdown
### T2 Coverage by T1
| Op | Actual (ms) | % T2 | Count |
```

---

#### Section 3: XPU vs CUDA Consistency

Header: `## 3. XPU vs CUDA Consistency`.

Model-level check: do XPU and CUDA run the same ops with the same shapes?

**Graph Consistency (calcflops-based)** — a sub-section (`### Graph Consistency`)
that compares device-independent columns (FLOPs, memory) from calcflops output
on CUDA vs XPU. Uses `compare_graphs.compare_model()` to detect dispatch path
divergence. Shows:
- Total FLOPs diff percentage
- Ops only in CUDA calcflops / only in XPU calcflops
- Ops with different FLOPs/memory values

**Trace Comparison** — for each XPU platform vs CUDA:
- **Summary counts**: common ops, platform-only ops
- **Platform-specific ops table**: ops only on one platform (fusion/dispatch differences)
- **Shape set differences**: for common ops with FLOPs > 0 (compute ops only — skip
  pure data-movement ops like clone/copy_ whose shape differences reflect dispatch
  path differences, not model behavior differences)

Shape set comparison shows per-op:
- Common shapes (present on both)
- CUDA-only shapes
- XPU-only shapes

---

#### Section 4: XPU vs 4080S Per-Op Efficiency

Header: `## 4. XPU vs 4080S: Per-Op Efficiency`.

Per-op R comparison between each XPU platform and 4080S.
Sorted by % T2 (most impactful ops first).

```markdown
| Op | R_xpu | R_4080S | R_diff | XPU (ms) | 4080S (ms) | % T2 | Verdict |
```

- **R_diff** = R_xpu − R_4080S
- **Verdict**: `XPU wins` (R_diff > +0.05), `XPU behind` (R_diff < −0.05), `~tie`
- Filters out tiny ops (|R_diff| < 0.03 AND %T2 < 1%)

---

#### Section 5: Optimization Targets

Header: `## 5. Optimization Targets`.

Ranks ops by potential T2 saving if XPU kernel matched 4080S roofline efficiency.
Only includes ops where R_xpu < R_4080S.

```markdown
| # | Op | R_xpu | R_4080S | Actual (ms) | Target (ms) | Saving (ms) | % T2 | Action |
```

- **Target (ms)** = Proj / R_4080S (what actual would be if matching 4080S efficiency)
- **Saving** = Actual − Target
- **Action**: `Optimize kernel` (XPU-specific problem) or `Fix projection` (low R on all platforms)
- **TOTAL row**: sum of all potential savings
- Footer note: kernel optimization potential vs projection fixes

---

## AI Analysis Pass

After the script generates the 5-section report, an LLM adds AI insights to
each section. For 154 models, the LLM writes individualized analysis (not
template-based — the LLM is expected to "hand-write" each one).

**AI analysis per section**:

| Section | AI adds | Source data |
|---------|---------|-------------|
| Summary | Refine action item priorities, add context for overall assessment | All sections |
| Projection Quality | Explain likely root cause for each flagged op (SDPA fusion, cache effect, gather pattern, etc.) | R_op values, op semantics, cross-platform comparison |
| XPU vs CUDA Consistency | Explain why platform-specific ops exist (fusion strategy, decomposition), assess impact | Platform-only ops, shape diffs |
| XPU vs 4080S | For each weakness: root cause hypothesis (kernel impl gap, memory subsystem, projection). For strengths: what XPU does well | R_diff, bound type, actual times |
| Optimization Targets | T2 contribution %, priority ranking, cross-section connections | Saving table + all above |

---

## Batch Report Generation

**Script**: `scripts/oob300/generate_all_reports.py`

```bash
# Inference (default: fp16, eval)
python scripts/oob300/generate_all_reports.py \
  --b580-dir <b580_results_dir> \
  --4080s-dir <4080s_results_dir> \
  [--b70-dir <b70_results_dir>] \
  --config config/hardware_specs.yaml \
  --output-dir reports/oob300/per_model/eager \
  [--models model1,model2] \
  [--skip-existing]

# Training (bf16, train)
python scripts/oob300/generate_all_reports.py \
  --b580-dir <b580_results_dir> \
  --4080s-dir <4080s_results_dir> \
  [--b70-dir <b70_results_dir>] \
  --config config/hardware_specs.yaml \
  --output-dir reports/oob300/per_model/eager \
  --precision bf16 --test train
```

Discovers all models, pairs across platforms, generates per-model reports.
Reports go to `reports/oob300/per_model/eager/<model>_bs<N>_<precision>_<test>.md`.

---

## Fleet-Level Summary Report

**Script**: `scripts/oob300/generate_fleet_summary.py`
**Issue #29 Outputs**: #2 (Fleet Geomean), #6 (Worst Models), #11 (Op Priority Ranking)

```bash
# Inference (default: fp16, eval)
python scripts/oob300/generate_fleet_summary.py \
  --b580-dir <b580_results_dir> \
  --4080s-dir <4080s_results_dir> \
  [--b70-dir <b70_results_dir>] \
  --config config/hardware_specs.yaml \
  --suite-dir benchmark/oob300 \
  -o reports/oob300/summary_eager_inference.md

# Training (bf16, train)
python scripts/oob300/generate_fleet_summary.py \
  --b580-dir <b580_results_dir> \
  --4080s-dir <4080s_results_dir> \
  [--b70-dir <b70_results_dir>] \
  --config config/hardware_specs.yaml \
  --suite-dir benchmark/oob300 \
  --precision bf16 --test train \
  -o reports/oob300/summary_eager_training.md
```

Reports go to `reports/oob300/summary_eager_inference.md` or `summary_eager_training.md`.

### Report structure (7 sections)

**Section 1: Overall** (`## 1. Overall`) — model counts (total, paired, failed), geomean R per
platform, geomean T2 ratio (XPU / CUDA).

**Section 2: Per-Suite Geomean** (`## 2. Per-Suite Geomean`) — breakdown by suite (torchbench / timm /
huggingface). Shows # models, geomean R per platform, T2 ratio per suite.

**Section 3: Model Scorecard** (`## 3. Model Scorecard`) — one row per model sorted by R_G31/R_4080S
ascending (worst first). Columns: Model, BS, R per platform, R ratio columns,
T2 per platform, T2 ratio, top gap op.

**Section 4: Worst 10 Models** (`## 4. Worst 10 Models by R`) — subset of scorecard (bottom 10 by
primary XPU R), with brief diagnosis per model: the top gap op and its R_op.

**Section 5: Op Priority Ranking** (`## 5. Op Priority Ranking`) — across all models, ranks ops by potential
fleet geomean R improvement. Shows: op name, # models it appears in, total
actual time, total saving if improved to match 4080S R_op, and estimated new fleet geomean R.

**Section 6: Projection Accuracy** (`## 6. Projection Accuracy`) — per-platform analysis of projection
quality across the fleet. Three sub-sections:

- **Overcounting (R_op > 1.05)** — ops where projection consistently exceeds
  actual GPU time (FLOPs formula overestimates or peak specs too low).
  Table: `| # | Op | Models (%) | Median R_op | Total Excess (ms) | Top Models |`
  Top Models = 3 models with the largest excess for that op.

- **Undercounting (R_op < 0.5)** — ops where actual GPU time far exceeds
  projection (missing FLOPs, memory accounting incomplete).
  Table: `| # | Op | Models (%) | Median R_op | Total Deficit (ms) | Top Models |`
  Top Models = 3 models with the largest deficit for that op.

- **Uncovered Ops** — ops with actual GPU time but zero projection (not in
  calcflops output). Includes a note about optimizer ops (fill_, add_, etc.)
  being expected uncovered in training mode.
  Table: `| # | Op | Models (%) | Total Actual (ms) | Avg per Model (ms) | Top Models |`
  Top Models = 3 models with the most actual time for that op.

Thresholds: overcounting needs R_op > 1.05 and actual > 0.01ms in >=2 models;
undercounting needs R_op < 0.5 and actual > 0.1ms in >=2 models. Top 15 ops
shown per sub-section, sorted by total excess/deficit/actual.

**Section 7: Graph Consistency** (`## 7. Graph Consistency`) — fleet-wide
calcflops graph comparison between CUDA and each XPU platform. Uses
`compare_graphs.compare_model()` to compare device-independent FLOPs/memory
columns. For each XPU platform vs CUDA, shows:

- Fleet summary: # models compared, # identical, # different
- **Difference Categories**: SDPA-only (only SDPA dispatch differs),
  Significant (>1% total FLOPs diff), Minor (small numerical differences)
- **Significant Divergences** table: models with >1% FLOPs diff, showing
  CUDA-only ops, XPU-only ops, mismatched ops counts
- **Op Differences Across Fleet**: aggregated op-level differences showing
  which ops differ most frequently across models

### Suite classification

Models are classified into suites via the YAML files in `benchmark/oob300/`.
If `--suite-dir` is not provided, models are classified as "unknown".

---

## Standalone Graph Consistency Report

**Script**: `scripts/oob300/compare_graphs.py`

Compares calcflops outputs from two devices to detect computational graph divergence.
This produces a standalone report separate from the fleet summary's Section 7 (which
embeds a subset of the same analysis).

The script compares device-independent columns (cum_flops, cum_memory) from calcflops
output. If the model's computational graph is identical on both devices, these columns
match exactly. Differences indicate dispatch path divergence (e.g., SDPA fusion,
XPU-specific overrideable ops, decomposition differences).

```bash
# Inference: CUDA vs XPU (all models with calcflops on both platforms)
python scripts/oob300/compare_graphs.py \
  --dir-a $OOB_RESULTS_INFERENCE_4080S \
  --dir-b $OOB_RESULTS_INFERENCE_B70 \
  --label-a CUDA --label-b XPU \
  -o reports/oob300/graph_consistency_eager_inference.md

# Training: CUDA vs XPU
python scripts/oob300/compare_graphs.py \
  --dir-a $OOB_RESULTS_TRAINING_4080S \
  --dir-b $OOB_RESULTS_TRAINING_B70 \
  --label-a CUDA --label-b XPU \
  --precision bf16 --test train \
  -o reports/oob300/graph_consistency_eager_training.md

# Specific models only
python scripts/oob300/compare_graphs.py \
  --dir-a $OOB_RESULTS_INFERENCE_4080S \
  --dir-b $OOB_RESULTS_INFERENCE_B70 \
  --label-a CUDA --label-b XPU \
  --models resnet50,BERT_pytorch,hf_GPT2
```

### Report structure

- **Fleet Summary**: # models compared, # identical, # different
- **Difference Categories**: SDPA-only (only SDPA dispatch differs),
  Significant (max(flops_diff, mem_diff) > 1%), Minor (small numerical differences)
- **Significant Divergences** table: models with >1% FLOPs diff
- **Per-Model Details**: for each model with differences, shows CUDA-only ops,
  XPU-only ops, ops with FLOPs/memory mismatches, and total FLOPs/memory diff %
- **Op Differences Across Fleet**: aggregated op-level differences showing which
  ops differ most frequently across models

### Notes

- Only compares models that have `*_calcflops.txt` in **both** directories
- Uses `--precision` and `--test` to match the correct file pattern
  (`*_<precision>_<test>_calcflops.txt` for training; `*_calcflops.txt` for inference)
- The comparison is device-independent: it compares FLOPs and memory columns which
  should be identical regardless of device. Differences reveal model behavior
  divergence, not hardware performance differences

---

## Interpretation Guide

### Reading R values

| R Range | Interpretation |
|---------|---------------|
| 0.95+ | Excellent — kernel nearly matches roofline |
| 0.85-0.95 | Good — minor optimization opportunities |
| 0.70-0.85 | Fair — some ops significantly below roofline |
| < 0.70 | Poor — major kernel inefficiency or projection error |

### Reading per-op R_op (Proj/Actual)

| R_op | Meaning |
|------|---------|
| > 1.05 | Projection overestimates (Section 2 flags these) |
| 0.80-1.05 | Good projection accuracy |
| < 0.80 | Projection underestimates or kernel slow (Section 2 diagnoses) |
| < 0.50 | Severe — likely missing FLOPs or memory |

### Reading Perf column

The Perf column shows actual throughput, not projected. Format depends on
what data is available:
- `38.3TFLOPS` — compute-bound op, 38.3 TFLOPS achieved
- `159GB/s` — memory-bound op, 159 GB/s achieved
- `2.5TFLOPS/112GB/s` — mixed or op with both FLOPs and memory, shows both

Compare against HW specs to judge efficiency:
- B580: 93T peak, 410 GB/s BW
- B70: 154T peak, 532 GB/s BW
- 4080S: 100.96T peak, 716.8 GB/s BW

**Vector engine ops** (softmax, native_layer_norm, native_batch_norm, max_pool2d):
FLOPs are zeroed in the parsing pipeline (`VECTOR_ENGINE_OPS` in
`compare_projection_vs_actual.py`) regardless of what calcflops data contains.
These ops run on the vector engine, not the matrix engine, so counting their
FLOPs against matrix engine peak would misclassify them as compute-bound.
The Perf column will only show BW for these ops.

### Common patterns

1. **CNN models** (resnet, vgg, inception): dominated by `aten::convolution`, `aten::addmm`.
   Often compute-bound. B580 should be competitive due to similar peak FP16.

2. **Transformer models** (BERT, GPT, LLM): dominated by `aten::addmm` (attention GEMM) +
   memory-bound layer_norm/softmax/elementwise. B580 is disadvantaged on memory-bound ops
   due to lower BW (410 vs 716.8 GB/s).

3. **SDPA fusion**: if model uses attention, check if runtime fuses SDPA. XPU uses
   `_scaled_dot_product_fused_attention_overrideable`, CUDA uses `_flash_attention`.
   DispatchLog may overcount unfused components.

4. **Vector engine ops** (max_pool, batch_norm, layer_norm, softmax): FLOPs zeroed in
   context_func.py because they use vector engine, not matrix engine. Roofline uses
   matrix engine peak, so counting their FLOPs would misclassify them as compute-bound.

### Op normalization rules

The pipeline normalizes op names before matching across platforms and data sources:

| Raw op | Normalized to | Reason |
|--------|--------------|--------|
| `aten::copy_` | `aten::clone` | `clone` dispatches to `copy_` which has the actual kernels |
| `aten::convolution_overrideable` | `aten::convolution` | XPU dispatches convolution to this op |
| `aten::convolution_backward_overrideable` | `aten::convolution_backward` | XPU dispatches backward convolution to this op |
| `aten::reshape`, `aten::contiguous` | `__view_noop__` | View/metadata ops — no GPU kernel, no calcflops entry |
| `aten::unbind` | `__view_noop__` | View op — no GPU kernel |

**Important**: `reshape` and `contiguous` are NOT normalized to `clone`. They are
view operations with no GPU kernels and no calcflops records. Normalizing them to
`clone` would pollute shape set comparisons.

---

## Current Data Locations

Configure data directories in `tools/agentic_xpu/.env` (see `.env.example` for variable names).

| Dataset | B580 env var | 4080S env var | B70 env var |
|---------|-------------|--------------|------------|
| Inference (fp16/eval) | `$OOB_RESULTS_INFERENCE_B580` | `$OOB_RESULTS_INFERENCE_4080S` | `$OOB_RESULTS_INFERENCE_B70` |
| Training (bf16/train) | `$OOB_RESULTS_TRAINING_B580` | `$OOB_RESULTS_TRAINING_4080S` | `$OOB_RESULTS_TRAINING_B70` |

Config file: `config/hardware_specs.yaml` (relative to repo root)
Suite YAML dir: `benchmark/oob300/` (relative to repo root)

### Quick reference: full commands

```bash
# Generate ALL inference per-model reports (154 models)
python scripts/oob300/generate_all_reports.py \
  --b580-dir $OOB_RESULTS_INFERENCE_B580 \
  --4080s-dir $OOB_RESULTS_INFERENCE_4080S \
  --b70-dir $OOB_RESULTS_INFERENCE_B70 \
  --config config/hardware_specs.yaml \
  --output-dir reports/oob300/per_model/eager

# Generate ALL training per-model reports (137 models)
python scripts/oob300/generate_all_reports.py \
  --b580-dir $OOB_RESULTS_TRAINING_B580 \
  --4080s-dir $OOB_RESULTS_TRAINING_4080S \
  --b70-dir $OOB_RESULTS_TRAINING_B70 \
  --config config/hardware_specs.yaml \
  --output-dir reports/oob300/per_model/eager \
  --precision bf16 --test train

# Generate inference fleet summary
python scripts/oob300/generate_fleet_summary.py \
  --b580-dir $OOB_RESULTS_INFERENCE_B580 \
  --4080s-dir $OOB_RESULTS_INFERENCE_4080S \
  --b70-dir $OOB_RESULTS_INFERENCE_B70 \
  --config config/hardware_specs.yaml \
  --suite-dir benchmark/oob300 \
  -o reports/oob300/summary_eager_inference.md

# Generate training fleet summary
python scripts/oob300/generate_fleet_summary.py \
  --b580-dir $OOB_RESULTS_TRAINING_B580 \
  --4080s-dir $OOB_RESULTS_TRAINING_4080S \
  --b70-dir $OOB_RESULTS_TRAINING_B70 \
  --config config/hardware_specs.yaml \
  --suite-dir benchmark/oob300 \
  --precision bf16 --test train \
  -o reports/oob300/summary_eager_training.md

# Generate standalone graph consistency report (inference)
python scripts/oob300/compare_graphs.py \
  --dir-a $OOB_RESULTS_INFERENCE_4080S \
  --dir-b $OOB_RESULTS_INFERENCE_B70 \
  --label-a CUDA --label-b XPU \
  -o reports/oob300/graph_consistency_eager_inference.md
```
