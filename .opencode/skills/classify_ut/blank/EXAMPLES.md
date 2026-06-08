# classify_ut blank — Known Classification Examples

Concrete `Reason` + `DetailReason` examples that have been validated by prior classification
sessions. Use as a template, NOT a substitute for source inspection. Re-read the cited source
state before applying — every cited file:line, issue URL, and commit hash must match the
**current** state of the configured checkout (`$PYTORCH_SRC`).

Read this file only when you need a `DetailReason` template for a specific verdict category.
Do NOT load it for every blank-Reason row; the canonical labels and rules already live in
`classify_ut/RULES.md` (the per-row deep analysis does not change because of examples here).

## Remote distributed file enabled

- `Reason`: `To be enabled`.
- `DetailReason`: `Distributed file enabled in remote distributed skip list: <file>`.
- Should name the remote skip-list files read and say that the file is registered for XPU
  through `run_distributed.py`.

## Remote distributed file missing

- `Reason`: `To be enabled`.
- `DetailReason`: `Distributed file missing from remote distributed skip list: <file>`.
- Should name checked dictionaries, `release/2.12` file presence, and enabled sibling files
  when useful.

## CUDA graph / cudagraph coverage

- `Reason`: `To be enabled`.
- `DetailReason`: `XPU graph coverage missing` or a similarly specific XPU graph gap.
- Should mention that XPU graph APIs exist and identify the missing XPU test coverage.

## Jiterator blank-status rows

- `Reason`: `Not applicable`.
- `DetailReason`: `CUDA-specific API: torch.cuda.jiterator`.
- Should mention the concrete `torch.cuda.jiterator` APIs used.

## cuBLAS deterministic blank-status rows

- `Reason`: `Not applicable`.
- `DetailReason`: `CUDA-specific API: cuBLAS`.
- Should mention the cuBLAS determinism behavior and any `@onlyCUDA` evidence.

## TensorExpr CUDA fuser rows

- `Reason`: `Not applicable`.
- `DetailReason`: `CUDA-specific API: TensorExpr CUDA fuser`.

## Existing XPU wrapper/direct file with generated XPU test but no XPU workbook result

- `Reason`: `To be enabled`.
- `DetailReason`: `Test exists but blank: <XPU class or file>`.
- Should name the exact XPU source file/class/function and expected XPU test name.

## Direct PyTorch test file with case-level collection evidence

- If the exact case is collected or generated from `pytorch/test`, classify using that direct
  source evidence. Do not require a `third_party/torch-xpu-ops/test/xpu` wrapper.
- If the file exists but the exact method/case is absent and targeted collection runs zero
  tests, classify `Community Change` with the source and collection evidence.
- Example: `test/dynamo/test_modules.py` exists, but `OptimizedModuleTest.test_assign_does_not_exist`
  was absent from local source, absent from `origin/release/2.12`, and
  `pytest --collect-only -k assign_does_not_exist` collected zero tests; this is NOT
  `XPU test file missing`.

## Local base test missing or method removed/refactored for a non-distributed row

- `Reason`: `Community Change`.
- `DetailReason` should name the source evidence, e.g. `Base function not found in upstream
  <testfile>; function removed, renamed, or refactored`.
