# Issue validation criteria reference

## Goal
Decide whether the available evidence is strong enough to file an XPU issue.

## Evidence modes
Use one of these evidence modes before filing:
- Manual validation: a local reproducer run on XPU with environment details.
- CI UT failure: a failing CI job tied to an upstream PyTorch change, plus the test name, rerun command, and relevant log output.

## Manual validation assumptions
- If the correct Python environment is unclear, ask the user which interpreter contains `torch.xpu`.
- When shell access is available, collect environment details with:

```bash
python -W ignore::RuntimeWarning -m torch.utils.collect_env
```

- When shell access is not available, ask the user to run the same command locally and paste the output.

## What counts as a confirmed issue
One of the following on XPU, when CPU, CI expectations, or current PyTorch semantics indicate otherwise:
- crash or internal error
- wrong numerical result
- wrong shape, stride-sensitive behavior, or dtype behavior
- unexpected unsupported-path error for an operator/path that should be supported
- backward mismatch where forward is expected to work
- a new or updated upstream test that now fails on XPU in CI

## What does not count by itself
- tiny floating-point noise without semantic consequence
- unsupported behavior that is clearly documented or already tracked
- failures caused by an invalid repro unrelated to the upstream CUDA fix

## Capture before filing
Capture these common fields:
- short bug statement and affected op, module, or test area
- full exception text, assertion failure, or mismatch summary
- upstream issue, PR, and commit links
- one-line summary of what the upstream change did

For manual validation also capture:
- exact reproducer command used and its output
- minimal repro script
- collect_env command and its output: `python -W ignore::RuntimeWarning -m torch.utils.collect_env`

For CI UT failures also capture:
- CI job link
- failing test identifier
- exact or reconstructed rerun command
- the relevant traceback or log excerpt