# Priority Rules

## Priority labels

- `P0`: Critical
- `P1`: High
- `P2`: Medium
- `P3`: Low

## P0 - Critical

Assign `P0` when either condition holds:

1. A measured, quantified performance regression between releases is greater
   than 7%.
2. A crash or segfault occurs on a Core API listed below.

### Core API list

- **Module loading:** `import torch`, `import torch.nn`.
- **Tensor lifecycle:** `torch.tensor()`, `torch.zeros()`, `torch.ones()`,
  `torch.empty()`, `torch.rand()`, `.clone()`, `.contiguous()`.
- **Device transfer:** `.to(device)`, `.cuda()`, `.cpu()`, `.xpu()`.
- **Basic arithmetic:** `+`, `-`, `*`, `/`, `@` (matmul) on standard dtypes
  (`float32`, `float64`, `int64`).
- **Autograd core:** `.backward()`, `torch.autograd.grad()` on a <=3-op graph.
- **Key module forward/backward:** `nn.Linear` and `nn.Conv2d` forward or
  backward.
- **Serialization:** `torch.save()`, `torch.load()`.
- **Basic indexing:** `tensor[idx]`, `.view()`, `.reshape()`.

| Condition | Evidence | Examples |
|---|---|---|
| Performance regression >7% | Issue body or comments cite a measured percentage and confirm a regression between releases. The drop must be >7%, not <=7%. | 15% slower on 2.12 than 2.11: yes. 5% slower: no. |
| Crash on a Core API | Stack trace shows SIGSEGV or an access violation on a Core API, or a segfault during module import, tensor creation, `.backward()`, device transfer, matmul, `nn.Linear` or `nn.Conv2d` forward, serialization, or indexing. | SIGSEGV in `torch.tensor()`: yes. Crash in a custom op: no. |
| Legacy Torch build failure | CI log shows compilation or linker errors that prevent a build. | Compilation failures, linker errors. |

## P1 - High

| Condition | Evidence | Examples |
|---|---|---|
| More than 6 failed UT cases | Count distinct test-case names in the issue body or title. Meta-tracking issues may list many cases. | Large test-class failures. |
| Regression | The issue cites a version where it passed and a current version where it fails. | Passed on 2.10, fails on 2.11. |
| Hang or timeout | The process remains alive but is stuck, such as after a 300-second timeout, infinite wait, or deadlock. | Distributed test hangs. |
| Benchmark accuracy regression | Benchmark accuracy passed in a prior release and now fails without a crash. | `fail_accuracy` on an E2E model. |

## P2 - Medium

| Condition | Evidence | Examples |
|---|---|---|
| Benchmark performance regression <=7% | A measured performance drop is cited and is no greater than 7%. | Minor throughput decrease. |
| 1-6 failed UT cases | A small number of related test failures. | A few operator tests failing. |
| Functional error without a crash | RuntimeError, AssertionError, or an incorrect result while the process continues. | Wrong output, type errors. |
| Feature gap blocking tests | Tests fail because an API is not implemented, without a crash. | Not implemented errors. |

## P3 - Low

| Condition | Evidence | Examples |
|---|---|---|
| Enhancement or feature request | The title contains `implement`, `enable`, `support`, `RFC`, `consider`, or `investigate`, or the body describes desired new functionality. | Feature requests. |
| Validation or error-message difference | XPU raises a different error from CPU or CUDA, or does not raise when it should, without incorrect computation. | Error-message mismatch. |
| Minor, cosmetic, or warning issue | Warning mismatch, deprecated API usage, or documentation gap. | Warning mismatches. |
| CUDA alignment without a functional break | XPU behavior differs from CUDA but is not incorrect. | Dtype support alignment. |
