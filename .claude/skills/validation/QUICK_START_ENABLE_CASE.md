# Quick Start — Enable Test Case

Steps to enable a PyTorch test case for XPU.

## 1. Clone the Tools

```bash
git clone https://github.com/daisyden/torch-xpu-ops/blob/opencode/classify_ut
```

## 2. Prepare Environment

Use `setup_env.sh` under `torch-xpu-ops/.opencode/skills/validation/scripts` at `opencode/classify_ut`:

```bash
bash setup_env.sh nightly <conda_env> <pytorch_folder>
```

Ensure the torch version installed in the conda environment is aligned with the source code in `<pytorch_folder>`.

## 3. Classify the Test Case

Refer to the `test_class_classification_merged.xlsx` spreadsheet to determine whether the case is **device-specific** or **device-agnostic**.

---

## 4a. Device-Specific Case

Run the refactoring pipeline to decouple the test from specific hardware accelerators:

```bash
/refactor-test-decoupling <test_file>       # absolute path in pytorch folder
/review-test-refactoring <test_file>
/submit-refactoring-pr <test_file>
```

- **`/refactor-test-decoupling`** — Reads the file, classifies every test method as S1/S2/S3, creates correctly-named classes, replaces device-specific APIs with generic equivalents.
- **`/review-test-refactoring`** — Audits the refactored file for correctness (classification, naming, instantiation, imports, external references).
- **`/submit-refactoring-pr`** — Rebases onto pytorch/pytorch `viable/strict`, commits, pushes to fork, and opens a draft PR with test count evidence.

## 4b. Device-Agnostic Case

Run the XPU enablement pipeline to add XPU coverage to an already device-generic test:

```bash
/enable-xpu-test <test_file> <test_class> \
    conda_env=<conda_env> \
    pytorch_folder=<pytorch_folder>
```

- **`/enable-xpu-test`** — Extends `instantiate_device_type_tests` to include `"xpu"`, widens `DecorateInfo` device_type entries, mirrors `largeTensorTest` and dtype decorators for XPU, runs verification via pytest, and submits a draft PR for passing classes.

---

## Quick Reference

| Case Type | Command | Outcome |
|-----------|---------|---------|
| Device-specific | `/refactor-test-decoupling` → `/review-test-refactoring` → `/submit-refactoring-pr` | PR against pytorch/pytorch with S1/S2/S3 decoupling |
| Device-agnostic | `/enable-xpu-test <file> <class> conda_env=... pytorch_folder=...` | PR adding XPU to `instantiate_device_type_tests` |
