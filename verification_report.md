# Pipeline Pre-flight Verification Report

**Date**: 2026-05-14
**Environment**: Intel XPU (DC GPU Max 1550), PyTorch 2.13.0a0+git12bca9d
**Baseline submodule**: `d67a87783002c17786e9501c8c67b360479f6bb7`

## Upstream Issue Assignment

| Assignee | Issue | Title | Category |
|----------|-------|-------|----------|
| Liangang | #1856 | [upstream #1856] channel last aten::hardswish_ will call extra copy | eltwise |
| Liangang | #3390 | [upstream #3390] Clarification requested on mixed non-atomic load and atomic CAS in Atomics.h | reduction |
| Liangang | #3150 | [upstream #3150] [Task] Align XPU kernel's implementation to stock PyTorch | reduction |
| Tong | #2795 | [upstream #2795] Histc raises error with integer input when deterministic algorithm is enabled | other |
| Tong | #2560 | [upstream #2560] [UT] "RuntimeError: iter.device(arg).is_xpu()" in test_torch_xpu.py | other |
| Yifeng | #2207 | [upstream #2207] Enable FP8/MXFP8 Ops with requests and CUDA alignment | other |
| Yifeng | #2140 | [upstream #2140] Consider how to avoid copy in FFT kernels | other |
| YuZhuo | #3361 | [upstream #3361] [upstream_ut] test/dynamo/test_higher_order_ops.py test_dropout failed with RuntimeError: CUDA not available | dynamo |
| YuZhuo | #3080 | [upstream #3080] cudagraph tests blocked by feature gap | dynamo |
| YuZhuo | #1969 | [upstream #1969] torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot create weak reference to 'torch.Event' object | dynamo |
| YuZhuo | #3388 | [upstream #3388] [Bug Skip] XPU Dynamo Graph Lowering - stream_index None | dynamo |
| Zhaoqiong | #2715 | [upstream #2715] [upstream_ut] torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped | dynamo |

## Verification Summary

| Repo where fix should be in | Category | Total | Reasonable PRs | PR could pass | Accepted by Reviewer | Already Fixed |
|-----------------------------|----------|:-----:|:--------------:|:-------------:|:--------------------:|:-------------:|
| torch-xpu-ops | torch-ops | 11 | 6 | 6 | 3 | 0 |
| | upstream_ut | 1 | 1 | 0 | 0 | 0 |
| | inductor | 4 | 0 | 0 | 0 | 2 |
| | task | 1 | 0 | 0 | 0 | 0 |
| | | | | | | |
| pytorch | torch-ops | 2 | 0 | 0 | 0 | 0 |
| | upstream_ut | 3 | 2 | 2 | 0 | 1 |
| | inductor | 7 | 4 | 3 | 2 | 3 |
| **TOTAL** | | **29** | **13** | **11** | **5** | **6** |

> - **Reasonable PRs** = PR with a reasonable fix approach
> - **PR could pass** = reproducer passes locally with the fix applied
> - **Accepted by Reviewer** = PR approved by human reviewer
> - **Already Fixed** = bug resolved in upstream/baseline, issue not yet closed

## Full Issue Detail

### torch-xpu-ops — torch-ops (11)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 1 | [#1856](https://github.com/intel/torch-xpu-ops/issues/1856) | channel last aten::hardswish_ extra copy | — | ✅ | ✅ | | | |
| 2 | [#1951](https://github.com/intel/torch-xpu-ops/issues/1951) | TestCommon.test_out BatchNorm non-contiguous | [txo#3665](https://github.com/intel/torch-xpu-ops/pull/3665) | ✅ | ✅ | | | |
| 3 | [#2015](https://github.com/intel/torch-xpu-ops/issues/2015) | TransformerEncoderLayer NaN/inf | [txo#3666](https://github.com/intel/torch-xpu-ops/pull/3666) | ❌ | | | | |
| 4 | [#2140](https://github.com/intel/torch-xpu-ops/issues/2140) | Consider how to avoid copy in FFT kernels | [txo#3636](https://github.com/intel/torch-xpu-ops/pull/3636) | ❌ | | | | |
| 5 | [#2207](https://github.com/intel/torch-xpu-ops/issues/2207) | Enable FP8/MXFP8 Ops | — | ❌ | | | | |
| 6 | [#2253](https://github.com/intel/torch-xpu-ops/issues/2253) | dtypes not align with CUDA | — | ✅ | ✅ | ✅ | | |
| 7 | [#2512](https://github.com/intel/torch-xpu-ops/issues/2512) | histc deterministic RuntimeError | [txo#3607](https://github.com/intel/torch-xpu-ops/pull/3607), [cp#15](https://github.com/chuanqi129/pytorch/pull/15) | ✅ | ✅ | | | |
| 8 | [#2518](https://github.com/intel/torch-xpu-ops/issues/2518) | Tensor subclass TypeError | [txo#3667](https://github.com/intel/torch-xpu-ops/pull/3667) | ❌ | | | | |
| 9 | [#2615](https://github.com/intel/torch-xpu-ops/issues/2615) | Unsupported dtype Half FFT | [txo#3668](https://github.com/intel/torch-xpu-ops/pull/3668) | ❌ | | | | |
| 10 | [#2953](https://github.com/intel/torch-xpu-ops/issues/2953) | fill overflow RuntimeError | [txo#3669](https://github.com/intel/torch-xpu-ops/pull/3669) | ✅ | ✅ | ✅ | | |
| 11 | [#3390](https://github.com/intel/torch-xpu-ops/issues/3390) | mixed non-atomic load / atomic CAS in Atomics.h | [txo#3635](https://github.com/intel/torch-xpu-ops/pull/3635), [txo#3603](https://github.com/intel/torch-xpu-ops/pull/3603) | ✅ | ✅ | ✅ | | |

### torch-xpu-ops — upstream_ut (1)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 12 | [#2436](https://github.com/intel/torch-xpu-ops/issues/2436) | NoneType.clone AttributeError | — | ✅ | | | | |

### torch-xpu-ops — inductor (4)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 13 | [#2554](https://github.com/intel/torch-xpu-ops/issues/2554) | AssertionError not raised | — | ❌ | | | | |
| 14 | [#2693](https://github.com/intel/torch-xpu-ops/issues/2693) | Scalars not equal | — | ❌ | | | | |
| 15 | [#2800](https://github.com/intel/torch-xpu-ops/issues/2800) | XpuDeviceProperties.major | — | ❌ | | | ✅ | |
| 16 | [#2891](https://github.com/intel/torch-xpu-ops/issues/2891) | Expected strides mismatch | — | ❌ | | | ✅ | |

### torch-xpu-ops — task (1)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 17 | [#3150](https://github.com/intel/torch-xpu-ops/issues/3150) | [Task] Align XPU kernel's implementation to stock PyTorch | — | | | | | |

### pytorch — torch-ops (2)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 18 | [#2560](https://github.com/intel/torch-xpu-ops/issues/2560) | iter.device XPU ASSERT | [cp#11](https://github.com/chuanqi129/pytorch/pull/11) | ❌ | | | | |
| 19 | [#2795](https://github.com/intel/torch-xpu-ops/issues/2795) | histc integer determinism | [cp#8](https://github.com/chuanqi129/pytorch/pull/8) | ❌ | | | | |

### pytorch — upstream_ut (3)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 20 | [#1963](https://github.com/intel/torch-xpu-ops/issues/1963) | MetadataMismatchError TestFakeTensor | [cp#14](https://github.com/chuanqi129/pytorch/pull/14) | ✅ | ✅ | | ✅ | |
| 21 | [#2359](https://github.com/intel/torch-xpu-ops/issues/2359) | GradcheckError backward not reentrant | — | ❌ | | | | |
| 22 | [#2712](https://github.com/intel/torch-xpu-ops/issues/2712) | weakref swap RuntimeError | [cp#17](https://github.com/chuanqi129/pytorch/pull/17) | ✅ | ✅ | | | |

### pytorch — inductor (7)

| # | Issue | Title | Fix PR | PR Reasonable | PR could pass | Accepted | Already Fixed | Notes |
|---|-------|-------|--------|:---:|:---:|:---:|:---:|-------|
| 23 | [#1969](https://github.com/intel/torch-xpu-ops/issues/1969) | InternalTorchDynamoError weakref Event | [cp#13](https://github.com/chuanqi129/pytorch/pull/13) | ✅ | ✅ | ✅ | | |
| 24 | [#2295](https://github.com/intel/torch-xpu-ops/issues/2295) | test_embedding_bag AssertionError | — | ❌ | ❌ | | ✅ | |
| 25 | [#2609](https://github.com/intel/torch-xpu-ops/issues/2609) | Inductor CppCompileError | [cp#16](https://github.com/chuanqi129/pytorch/pull/16) | ✅ | ✅ | ✅ | | |
| 26 | [#2715](https://github.com/intel/torch-xpu-ops/issues/2715) | Unsupported inline function | [cp#12](https://github.com/chuanqi129/pytorch/pull/12) | ✅ | ✅ | | ✅ | |
| 27 | [#3080](https://github.com/intel/torch-xpu-ops/issues/3080) | cudagraph tests blocked by feature gap | — | ❌ | | | | |
| 28 | [#3361](https://github.com/intel/torch-xpu-ops/issues/3361) | test_dropout CUDA not available | [cp#9](https://github.com/chuanqi129/pytorch/pull/9) | ❌ | ❌ | | ✅ | |
| 29 | [#3388](https://github.com/intel/torch-xpu-ops/issues/3388) | stream_index None | [cp#10](https://github.com/chuanqi129/pytorch/pull/10) | ✅ | ❌ | | | |

## Key Observations

1. **13 reasonable PRs** out of 29 issues — agents produced viable fixes for ~45% of issues
2. **11 PRs pass locally** — reproducer confirms the fix works
3. **5 PRs accepted by reviewer** — human-approved and ready to merge
4. **6 issues already fixed** in upstream/baseline — can be closed without further work
5. **#2512/#2795** overlap — same histc determinism bug tracked in both repos

## Logs

- Phase 1 log: `verification_phase1.log`
- Phase 2 log: `verification_phase2.log`
- Per-issue logs: `verification_results/issue_<N>.log`, `verification_results/round2/issue_<N>.log`
