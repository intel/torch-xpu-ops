---
name: classify-test-files
description: Use when classifying PyTorch test files as device_unrelated, device_agnostic, or device_specific, or when asked to categorize/scan test files by device dependency
---

# Classify Test Files

Classify PyTorch test files into three categories based on device dependency. Runs a Python script that scans test files, detects device patterns, and builds a symlink tree in `agent_space/classified_test_files/`.

## Classification Rules

Classification is hierarchical: **device_specific > device_agnostic > device_unrelated**.

If a file has even one device-specific pattern, it is device_specific — regardless of how many device-agnostic or device-unrelated tests it contains. Same precedence for device_agnostic over device_unrelated.

### device_specific (any match wins)

- File named `test_cuda.py`, `test_mps.py`, `test_xpu.py`, `test_rocm.py`, `test_mtia.py`, `test_tpu.py`, `test_lazy.py`
- Imports `TEST_CUDA`, `TEST_MPS`, `TEST_XPU`, `TEST_ROCM`, `TEST_MTIA` from `common_{device}` modules
- References `torch.cuda.`, `torch.mps.`, `torch.xpu.` runtime APIs

### device_agnostic (any match, unless already device_specific)

- Imports `instantiate_device_type_tests` from `common_device_type`
- Uses `@ops(...)`, `@dtypes(...)`, `@dtypesIf*` decorators
- Calls `instantiate_device_type_tests(...)`
- Imports from `common_methods_invocations` or `common_dtype`
- References `TEST_PRIVATEUSE1`, `TEST_ACCELERATOR`, or `torch.accelerator.` (generic out-of-tree integration point)

### device_unrelated (default)

- CPU-only tests with no device imports, no device decorators, no device-specific APIs

## Usage

Run the classification script:

```bash
python .claude/skills/classify-test-files/classify_test_files.py [optional_test_dir]
```

Without arguments, it scans `test/` in the repo root. With an argument, scans that directory.

## Output

```
agent_space/classified_test_files/
    device_unrelated/    -> symlinks to test files
    device_agnostic/     -> symlinks to test files
    device_specific/     -> symlinks to test files
    _report.json         -> summary counts + per-file paths
```

If the output directory already exists, it is removed and recreated to keep classifications fresh.
