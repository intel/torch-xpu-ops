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

## Repro script requirements
- Print environment metadata (`torch.__version__`, `torch.xpu.is_available()`)
- Prefer extracting the upstream regression test or reproducer and adapting it to `torch.xpu`
- Print mismatch details or exception info

## Confirmed bug criteria
Two categories:

1. **Functionality** — crash, segfault, hang, wrong numerical result, wrong shape/stride/dtype, unsupported-path error for an operator that should work
2. **Performance** — measurable regression compared to CPU or prior nightly

**Not** a bug by itself: tiny float noise, documented unsupported paths, or failures from an invalid repro.

## Evidence to capture before escalation
- Exact command and output
- `python -W ignore::RuntimeWarning -m torch.utils.collect_env`
- Full exception text or mismatch summary
- Minimal repro script
- Upstream issue/PR/commit links
