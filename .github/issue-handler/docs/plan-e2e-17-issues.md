# Pipeline E2E Run Plan — 17 Upstream Issues

> **Goal:** Run the full pipeline (format → verify → triage → route) on 17 issues on `intel/torch-xpu-ops`. Fix pipeline bugs found during iteration.

## Pipeline Flow (as implemented)

```
No status marker → format_agent (sets DISCOVERED)
DISCOVERED → verify_existence (test locally)
  ├── passes → agent:close label, DONE
  └── fails → triage_agent (sets IMPLEMENTING)
IMPLEMENTING → route by target_repo metadata
  ├── torch-xpu-ops → assign Copilot, done (hands off)
  └── pytorch → code_fix agent → PR to chuanqi129/pytorch
```

## Issues (upstream numbers on intel/torch-xpu-ops)

| # | Title | Core Problem |
|---|-------|-------------|
| 1951 | test_out failures | Framework regression |
| 1963 | MetadataMismatchError FakeTensor | Metadata inconsistency |
| 2015 | inf in TransformerEncoderLayer | Numerical instability |
| 2253 | dtypes not aligned with cuda | Dispatch logic |
| 2359 | Backward not reentrant | Autograd logic |
| 2436 | NoneType has no attribute clone | Logic bug |
| 2512 | _histc_xpu non-deterministic | Operator determinism |
| 2518 | Tensor subclass TypeError | Core compatibility |
| 2615 | Unsupported dtype Half | Dispatch regression |
| 2295 | embedding_bag accuracy fail | Kernel accuracy |
| 2554 | AssertionError not raised | Test logic error |
| 2609 | Inductor CppCompileError | Codegen bug |
| 2693 | Scalars not equal | Scalar logic |
| 2712 | Cannot swap t2 weakref | Memory management |
| 2800 | XpuDeviceProperties no 'major' | Python/C++ mapping |
| 2891 | Inductor runtime log pattern | Logging logic |
| 2953 | overflow in conversion | Cast logic |

## Execution Plan

### Phase 1: Single Issue Iteration (issue #1951)

1. Source env, run format on #1951
2. Verify issue body is properly formatted
3. Run verify_existence on #1951 (test locally)
4. If still fails → run triage on #1951
5. Check triage output (target_repo, verdict)
6. If IMPLEMENTING → check routing (copilot vs code_fix)
7. Fix any pipeline bugs found

### Phase 2: Fix Pipeline Bugs

Based on Phase 1 findings, patch code and re-test.

### Phase 3: Batch Run (remaining 16 issues)

Run pipeline on all remaining issues. Use `run_pipeline.py --once --issues ...` for each stage advancement.

## Pre-flight Checks

- [ ] `set -a && source .env && set +a`
- [ ] Verify `GH_TOKEN` works: `gh issue view 1951 --repo intel/torch-xpu-ops --json state`
- [ ] PyTorch build working: `source ~/intel/oneapi/setvars.sh && cd ~/pytorch && source .venv/bin/activate && python -c "import torch; print(torch.xpu.is_available())"`
- [ ] No orphan opencode processes: `ps aux | grep 'opencode run' | grep -v grep`
