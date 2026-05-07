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

## Required reading (mandatory)

Before performing any of the following tasks, you MUST read the linked file in
full before proceeding. Do not skip this step. Do not paraphrase from
memory. The contents of these files are authoritative.

| When you are about to... | Read this file first |
|--------------------------|---------------------|
| Open a pull request, push a branch, or write a PR body | `.github/skills/xpu-ops-pr-creation/SKILL.md` |
| Review a pull request | `.github/skills/xpu-ops-pr-review/SKILL.md` |
| Align upstream CUDA/backend fixes for XPU | `.github/skills/pytorch-cuda-fix-xpu-alignment/SKILL.md` |

Path-specific coding rules are auto-loaded by the agent based on the files you
edit (via `applyTo` globs), so you do not need to read them manually:

- `src/**` → `.github/instructions/xpu-kernels.instructions.md`
- `test/**` → `.github/instructions/xpu-tests.instructions.md`
- `yaml/**` → `.github/instructions/xpu-yaml.instructions.md`

State explicitly in your response which skill file(s) you read.

This file provides repository-wide context and applies to all Copilot
interactions within this repository.
