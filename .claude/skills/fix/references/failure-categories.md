# Failure Categories

Shared reference for `fix/triage` and `xpu-nightly-ci-fix`. Both orchestrators
use this taxonomy to classify root cause and select a fix strategy.

| Category | Description | Typical fix location |
|----------|-------------|---------------------|
| **XPU backend bug** | Bug in XPU kernel or backend code | `torch/_inductor/` or `third_party/torch-xpu-ops/` |
| **Tolerance too tight** | Numerical precision mismatch vs CUDA | Adjust `atol`/`rtol` to match CUDA |
| **Edge case / numerical accuracy** | NaN/Inf from extreme inputs, CPU-vs-XPU or fp32-vs-fp16 divergence, values near `finfo.max`/`min` | Compare against CUDA/CPU reference; confirm it is a real bug, not expected precision behavior |
| **Skip decorator stale** | `@skipIfXpu`/`@expectedFailure` but test now passes | Remove decorator — see `fix/implement` UT Skip Removal section |
| **Upstream regression** | New upstream code broke XPU; needs XPU-side workaround | `torch/`, `aten/`, `test/`, or `third_party/torch-xpu-ops/` |
| **Test infrastructure** | Environment, import, or setup issue | Test file or CI config |

When the failure involves a newly added test, check the commit/PR that
introduced it to confirm whether XPU support is expected — this affects the
fix strategy.
