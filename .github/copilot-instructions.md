# Copilot Instructions — Torch XPU Ops Repository

This repository implements Torch XPU Operators for Intel GPU support.

## Repository purpose
- Provide XPU backend operator implementations.
- Maintain consistency across operator definitions, implementations, and tests.
- Ensure correctness, regression safety, and backend-specific behavior.

## Repository layout
- `src/` — XPU operator implementations and backend-specific logic.
- `yaml/` — operator definitions, schemas, and configuration sources.
- `test/` — regression tests and behavior validation.
- `.github/workflows/` and `.ci/` — CI workflows and related validation logic.
- `tools/` — helper scripts and utilities.

## General engineering priorities
- Correctness and regression safety over micro-optimizations.
- Prefer focused, targeted tests over broad or generic coverage.
- Keep `yaml/`, `src/`, and `test/` consistent.
- Be explicit about behavior changes, especially dtype, shape, indexing, or dispatch behavior.

## Review and change guidance
- Operator or kernel changes should be accompanied by targeted tests.
- YAML or schema changes should be reflected in implementation and tests.
- CI changes should clearly explain impact on validation coverage.

## Skills index

| Task | Skill file |
|------|-----------|
| Create a PR | `.github/skills/xpu-ops-pr-creation/SKILL.md` |
| Review a PR | `.github/skills/xpu-ops-pr-review/SKILL.md` |

Domain-specific coding rules:
- XPU kernels: `.github/instructions/xpu-kernels.instructions.md`
- XPU tests: `.github/instructions/xpu-tests.instructions.md`
- YAML ops: `.github/instructions/xpu-yaml.instructions.md`

## Pull request requirements

Every PR must include validation so CI can verify the fix:

1. **Include a reproducer test** under `test/regressions/test_<description>.py` that:
   - Contains `def test_...()` functions or `class Test...` classes
   - Is runnable via `pytest test/regressions/test_<description>.py`
   - Imports `torch` and targets `xpu` device

2. **If no reproducer is possible** (CI-only, docs, build fixes), state in the PR body which existing tests validate the change:
   ```
   Test: test/regressions/test_foo.py
   ```
   Or if none apply:
   ```
   Test: none (reason)
   ```

This file provides repository-wide context and applies to all Copilot interactions
within this repository.