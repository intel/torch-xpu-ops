# Public E2E Plan — intel/torch-xpu-ops Issues

## Roles

- **Pipeline** (`run_pipeline.py`): runs the full process (format → triage → fix → PR), manages labels, posts comments, handles retries. All behavior is internal to the pipeline.
- **Hermes**: launches the pipeline, monitors output, records results, and **fixes bugs** found during execution. Hermes does NOT manually manage labels, comments, or issue bodies — that's the pipeline's job.

## Config Change

Switch `agent_config.yml` to target public repo:
```yaml
repos:
  xpu_ops_issue: "intel/torch-xpu-ops"     # was ZhaoqiongZ/torch-xpu-ops-exp
```

## Issues

### Batch 1 — High Confidence Bugs
| # | Title | Assignee |
|---|-------|----------|
| #2795 | Histc deterministic error with integer input | Tong |
| #2560 | iter.device(arg).is_xpu() RuntimeError | Tong |
| #3361 | test_dropout "CUDA not available" | YuZhuo |
| #3388 | stream_index None in dynamo | YuZhuo |

### Batch 2 — Medium Confidence Bugs
| # | Title | Assignee |
|---|-------|----------|
| #1856 | hardswish_ extra copy (channel last) | Liangang |
| #1969 | weak reference to torch.Event | YuZhuo |
| #2715 | Unsupported: skipped function inline | Zhaoqiong |

### Batch 3 — Format + Triage Only (expect NEEDS_HUMAN)
| # | Title | Assignee |
|---|-------|----------|
| #3390 | Mixed non-atomic load in Atomics.h | Liangang |
| #3150 | Align XPU kernel to stock PyTorch | Liangang |
| #3080 | cudagraph feature gap | YuZhuo |
| #2207 | Enable FP8/MXFP8 Ops | Yifeng |
| #2140 | Avoid copy in FFT kernels | Yifeng |

## Execution

1. Apply config change
2. Run `python scripts/run_pipeline.py --issues <batch>` for each batch
3. Hermes monitors output, records results
4. If pipeline hits a bug → Hermes fixes the code, re-runs
5. After all batches: review E2E report dashboard

## Cost Estimate
~$3-10 total across all 12 issues
