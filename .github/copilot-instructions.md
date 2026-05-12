---
name: copilot-instructions
description: The overall hot memory that should be loaded in all LLM agents when developing on torch-xpu-ops.
license: MIT
---

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
| Fix or triaging an issue | `.github/skills/xpu-issues-triaging/SKILL.md` |
| Open a pull request, push a branch, or write a PR body | `.github/skills/xpu-ops-pr-creation/SKILL.md` |
| Review a pull request | `.github/skills/xpu-ops-pr-review/SKILL.md` |

Path-specific coding rules are auto-loaded by the agent based on the files you
edit (via `applyTo` globs), so you do not need to read them manually:

- `src/**` → `.github/instructions/xpu-kernels.instructions.md`
- `test/**` → `.github/instructions/xpu-tests.instructions.md`
- `yaml/**` → `.github/instructions/xpu-yaml.instructions.md`

State explicitly in your response which skill file(s) you read.

This file provides repository-wide context and applies to all Copilot
interactions within this repository.

<!-- Guidelines below adapted from https://github.com/forrestchang/andrej-karpathy-skills (MIT License) -->

# Guidelines

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


