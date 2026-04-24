# Local XPU validation reference

## Goal
Confirm that a reproducer demonstrates a real XPU backend bug before filing an issue.

## Environment assumptions
- If the correct Python environment is unclear, ask the user which interpreter contains `torch.xpu`.
- When shell access is available, collect environment details with:

```bash
python -W ignore::RuntimeWarning -m torch.utils.collect_env
```

- When shell access is not available, ask the user to run the same command locally and paste the output.

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

## Capture before filing
- exact reproducer command used and its output
- collect_env command and its output: `python -W ignore::RuntimeWarning -m torch.utils.collect_env`
- full exception text or mismatch summary
- minimal repro script
- upstream issue, PR, and commit links
- any quick reduction that narrows the issue to one op family