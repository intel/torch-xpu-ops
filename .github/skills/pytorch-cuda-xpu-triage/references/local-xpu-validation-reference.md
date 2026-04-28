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
- Use the smallest tensors that hit the edge case
- Compare CPU vs XPU output
- Print mismatch details or exception info

## Confirmed bug criteria
One of the following on XPU, when CPU semantics disagree:
- Crash or internal error
- Wrong numerical result
- Wrong shape/stride/dtype behavior
- Unsupported-path error for an operator that should work
- Backward mismatch where forward is expected to work

**Not** a bug by itself: tiny float noise, documented unsupported paths, or failures from an invalid repro.

## Evidence to capture before filing
- Exact command and output
- `python -W ignore::RuntimeWarning -m torch.utils.collect_env`
- Full exception text or mismatch summary
- Minimal repro script
- Upstream issue/PR/commit links
