# Local XPU validation reference

## Goal
Run a generated reproducer against a local torch nightly with XPU enabled and
decide whether the behavior is a real XPU backend bug.

## Environment assumptions
- Do not auto-detect Python inside this skill.
- If the correct interpreter is unclear, ask the user which Python environment contains `torch.xpu`.
- If the nightly build needs to be refreshed, give the user this command:

```bash
pip3 install --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/xpu
```

## Minimal repro script checklist
- imports only what is necessary
- prints environment metadata
- checks whether torch.xpu exists and is available
- creates the smallest tensors that still hit the edge case
- compares CPU and XPU behavior when possible
- prints mismatch details, exception type, and message

## Suggested environment print block
```python
import platform
import sys
import torch

print({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "git_version": getattr(torch.version, "git_version", None),
    "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
})
```

## Suggested run command
```bash
python <repro_script.py>
```

If the user relies on a non-default interpreter, replace `python` with the exact interpreter path or environment-specific launcher.

## Suggested environment collection command
```bash
python -W ignore::RuntimeWarning -m torch.utils.collect_env
```

## What counts as a confirmed bug
One of the following on XPU, when CPU and current PyTorch semantics indicate otherwise:
- crash or internal error
- wrong numerical result
- wrong shape, stride-sensitive behavior, or dtype behavior
- unexpected unsupported-path error for an operator/path that should be supported
- backward mismatch where forward is expected to work

## What does not count by itself
- tiny floating-point noise without semantic consequence
- unsupported behavior that is clearly documented or already tracked
- failures caused by an invalid repro unrelated to the upstream CUDA fix

## Before filing the issue
Capture:
- exact command used and its output
- full exception text or mismatch summary
- minimal repro script
- upstream issue, PR, and commit links
- any quick reduction that narrows the issue to one op family
