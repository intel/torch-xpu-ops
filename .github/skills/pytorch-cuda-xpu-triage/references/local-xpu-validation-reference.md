# Local XPU validation reference

## Goal
Run a generated reproducer against a local torch nightly with XPU enabled and
decide whether the behavior is a real XPU backend bug.

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
scripts/run_with_xpu_python.sh <repro_script.py>
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
- exact command used
- full exception text or mismatch summary
- minimal repro script
- upstream issue, PR, and commit links
- any quick reduction that narrows the issue to one op family
