# Local XPU Validation Reference

## Goal
Take a reproducer derived from an upstream issue/PR/commit (CUDA, ROCm, or any backend) and run it on the **latest XPU torch nightly** to determine whether XPU shares the same bug.

## Environment setup
Always use the latest XPU nightly before running reproducers:
```bash
python -m pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/xpu
```
Use a dedicated virtualenv/conda env. Add `torchvision`/`torchaudio` only if the repro needs them.

If the correct interpreter is unclear, ask the user.

## CUDA → XPU device mapping
When adapting upstream CUDA reproducers, apply these substitutions:
- `.cuda()` → `.xpu()`, `.to('cuda')` → `.to('xpu')`
- `torch.cuda.synchronize()` → `torch.xpu.synchronize()`
- `torch.cuda.is_available()` → `torch.xpu.is_available()`
- `torch.cuda.memory_allocated()` → `torch.xpu.memory_allocated()`
- `torch.cuda.current_device()` → `torch.xpu.current_device()`
- `torch.cuda.device_count()` → `torch.xpu.device_count()`
- `CUDA_VISIBLE_DEVICES` → `ZE_AFFINITY_MASK`
- Remove or replace `@skipIfNoCUDA` decorators
- Replace `torch.cuda.amp` with `torch.xpu` equivalents if needed

## Repro script requirements
- Print environment metadata (`torch.__version__`, `torch.xpu.is_available()`)
- Prefer extracting the upstream regression test or reproducer and adapting it to `torch.xpu` using the mapping above
- Print mismatch details or exception info
- Verify the op actually ran on XPU — check output `.device` is `xpu` or use `TORCH_SHOW_DISPATCH_TRACE=1`. If the op fell back to CPU, the result does not count as XPU-validated

## Confirmed bug criteria
Two categories:

1. **Functionality** — crash, segfault, hang, wrong numerical result, wrong shape/stride/dtype, unsupported-path error for an operator that should work
2. **Performance** — flag for benchmark validation; do not attempt manual timing comparison in this skill

**Not** a bug by itself: tiny float noise, documented unsupported paths, or failures from an invalid repro.

## Evidence to capture before escalation
- Exact command and output
- `python -W ignore::RuntimeWarning -m torch.utils.collect_env`
- Full exception text or mismatch summary
- Minimal repro script
- Upstream issue/PR/commit links
