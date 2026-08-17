# fix/domains — Domain Registry

Authoritative mapping from `fix/root-cause` domain values to loadable
domain skills. This file is the single source of truth for:

1. **Which `domain` values `fix/root-cause` is allowed to emit** — the JSON
   array below is the closed set.
2. **Which skill each `domain` value maps to** — orchestrators load
   the `skill_path` after triage.
3. **Which `target_repo` each `domain` implies** — used to sanity-check
   `fix/root-cause`'s independent `target_repo` output.

## Registry

| domain | skill_path | target_repo | applies_when | test_locations | fix_locations |
|---|---|---|---|---|---|
| `xpu-kernel` | `fix/domains/xpu-kernel` | `torch-xpu-ops` | root cause in XPU backend kernels, dispatch, or SYCL code | `test/xpu/`, `test/inductor/` (XPU re-enabled tests) | `src/ATen/native/xpu/` |
| `inductor` | `fix/domains/inductor` | `pytorch` | root cause in `torch._inductor` or `torch._dynamo` (device-agnostic) | `test/inductor/` | `torch/_inductor/`, `torch/_dynamo/` |
| `upstream-pytorch` | `fix/domains/upstream-pytorch` | `pytorch` | root cause in device-agnostic pytorch core (framework regressions, test infra) | anywhere in `test/` | `torch/`, `aten/` (non-CUDA/XPU dirs) |

Machine-readable list of valid `domain` values (must match the table above):

```json
["xpu-kernel", "inductor", "upstream-pytorch"]
```

## Contracts

**`fix/root-cause`:**
- MUST emit `domain` from the JSON list above; if none applies, emit
  `NEEDS_HUMAN` instead of inventing a new value.
- MUST emit `target_repo` matching the registry entry for the chosen
  `domain`; a mismatch is a bug in triage's reasoning.

**Orchestrators (`issue-handler`, `xpu-nightly-ci-fix`):**
- MUST look up the emitted `domain` in this registry before loading a
  domain skill.
- MUST fail loudly (`NEEDS_HUMAN`, reason cites this file) on:
  - `domain` value not in the JSON list
  - `skill_path` directory does not exist
  - `target_repo` in triage output differs from the registry entry
- MUST NOT proceed with "no domain skill" as a fallback — that was the
  silent-no-op bug this registry fixes.

## Adding a new domain

1. Create `.claude/skills/fix/domains/<name>/SKILL.md`.
2. Add one row to the table above.
3. Add `<name>` to the JSON list.
4. Update `fix/root-cause` Step 1 to describe when to emit `<name>`.
5. Optionally add `<name>` to `.claude/skills/fix/skip-management/`'s
   caller list if the domain interacts with skip decorators.
