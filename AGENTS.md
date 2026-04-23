# AGENTS.md — torch-xpu-ops

Quick reference for agents working in this repository.
For full repo context, read `.github/copilot-instructions.md` first — it defines
the repo layout, engineering priorities, and PR requirements.

---

## Skills index

| Task | Skill file |
|------|-----------|
| Create a PR | `.github/skills/xpu-ops-pr-creation/SKILL.md` |
| Review a PR | `.github/skills/xpu-ops-pr-review/SKILL.md` |

Domain-specific coding rules (kernels, tests, YAML):
- XPU kernels: `.github/instructions/xpu-kernels.instructions.md`
- XPU tests: `.github/instructions/xpu-tests.instructions.md`
- YAML ops: `.github/instructions/xpu-yaml.instructions.md`

---

## Shared conventions

**This repo is `intel/torch-xpu-ops` — no auto-PR.** Agents commit and push a branch;
a human opens or reviews the PR.

**Branch format:** `agent/<slug>` — lowercase, spaces→hyphens, max 50 chars.

**No direct push to `main` or `upstream`.** Always work on a feature branch.

**Every PR must satisfy the test requirements in `.github/copilot-instructions.md`.**
