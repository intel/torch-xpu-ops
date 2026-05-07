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
General rule: replace `torch.cuda.*` with `torch.xpu.*` and device strings `'cuda'` / `'cuda:0'` with `'xpu'` / `'xpu:0'`. Most APIs have a 1:1 counterpart.

Non-obvious exceptions:
- `CUDA_VISIBLE_DEVICES` → `ZE_AFFINITY_MASK`
- `torch.cuda.amp.*` → `torch.amp.*(device_type='xpu')`
- `torch.backends.cudnn.*` — no XPU equivalent; remove or skip the related block
- `@skipIfNoCUDA` / `@onlyCUDA` — remove or replace with XPU availability check

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
