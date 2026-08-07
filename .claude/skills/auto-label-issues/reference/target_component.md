# Target Component Rules

The fix location determines ownership. Do not use labels, domains, or the
location where the failure surfaces as a substitute for tracing the code path.

| Path that must change | `target_component` |
|---|---|
| The **test file itself**, such as a stale assertion or tolerance, missing `sys.path` or import fix, an un-generalized CUDA-only call, a missing-attribute skip guard, or a syntax error, while product code is correct | `test-case` |
| `pytorch/aten/`, `pytorch/torch/`, or `pytorch/c10/`, excluding `third_party/torch-xpu-ops/` and the test file | `pytorch` |
| `third_party/torch-xpu-ops/`, or `src/` in a standalone torch-xpu-ops checkout | `torch-xpu-ops` |
| A driver, compiler, library, already-tracked upstream issue, or external package must change | `third-party` |
| No clean path resolves to product code, the test case, or a third-party owner | `N/A` |

If the test correctly calls the API and the underlying dispatch or kernel is
wrong, ownership is `pytorch` or `torch-xpu-ops`, not `test-case`.

## Required evidence

The delegated trace must cite file and line references, relevant symbols, and
the call path from the test or API to the failure. If pytorch_folder is not blank, it must inspect the local
checkout using `pytorch_folder` and check `git log` for an upstream or local
commit that already fixes the exact root cause. Record the commit when found.

Skip and xfail decorators are not fixes. Their presence confirms that the
failure remains relevant and cannot support an already-fixed conclusion.

## Canonical verdicts

| Ownership or condition | `need_action` |
|---|---|
| A third party must fix the cause | `NEED_FIX_3RDPARTY` |
| The issue is a feature request or a task or numerical accuracy issue or performance issue | `NEED_HUMAN` |
| `os` is Windows | `NEED_HUMAN` |
| Test file itself | `NEED_FIX_CASE` |
| `pytorch` product code | `NEED_FIX` |
| `torch-xpu-ops` product code | `NEED_FIX` |
| Test file itself | `NEED_FIX_CASE` |
| Inconclusive trace | `NEED_FIX` |
| No resolvable path, or human planning required for another reason | `NEED_HUMAN` |

An existing upstream fix may be reported in `evidence.upstream_fix`, but do
not call a skip or xfail an upstream fix.
