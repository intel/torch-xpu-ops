# Copilot Instructions — torch-xpu-ops

## Required reading (mandatory)

All agent instructions live in `CLAUDE.md` at the repository root. You MUST read the linked file in
full before proceeding. Do not skip this step. Do not paraphrase from
memory. The contents of these files are authoritative.

| When you are about to... | Read this file first |
|--------------------------|---------------------|
| All agent instructions | `CLAUDE.md` |
| Analyze XPU operator coverage, parity, defects, or run full/daily scan | `.github/skills/xpu-backend-scan/SKILL.md` |
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
