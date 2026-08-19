# Domain Registry (co-located with fix-root-cause)

Authoritative mapping from `fix-root-cause` domain values to
domain reference files. This file is the single source of truth for:

1. **Which `domain` values `fix-root-cause` is allowed to emit** — the
   JSON array below is the closed set.
2. **Which reference file each `domain` value maps to** — the sibling
   markdown loaded by `fix-root-cause` after it emits `domain`.
3. **Which `target_repo` each `domain` implies** — used to sanity-check
   `fix-root-cause`'s independent `target_repo` output.

The registry and the `domain-*.md` files live with `fix-root-cause`
(their sole producer/consumer) rather than under a separate
`fix-domains/` directory, mirroring pytorch/pytorch's sibling-markdown
pattern for knowledge bases (see `.claude/skills/aoti-debug/` and
`.claude/skills/fix-issue/` in pytorch/pytorch).

`fix-root-cause` loads **only** the `reference_file` for its emitted
`domain`; the other two stay off-context. That is the progressive
disclosure this layout enables — a single closed-set lookup (this
file) up front, deep knowledge (`domain-*.md`) only when it matches.

## Registry

| domain | reference_file | target_repo | applies_when | test_locations | fix_locations |
|---|---|---|---|---|---|
| `xpu-kernel` | `domain-xpu-kernel.md` | `torch-xpu-ops` | root cause in XPU backend kernels, dispatch, or SYCL code | `test/xpu/`, `test/inductor/` (XPU re-enabled tests) | `src/ATen/native/xpu/` |
| `inductor` | `domain-inductor.md` | `pytorch` | root cause in `torch._inductor` or `torch._dynamo` (device-agnostic) | `test/inductor/` | `torch/_inductor/`, `torch/_dynamo/` |
| `upstream-pytorch` | `domain-upstream-pytorch.md` | `pytorch` | root cause in device-agnostic pytorch core (framework regressions, test infra) | anywhere in `test/` | `torch/`, `aten/` (non-CUDA/XPU dirs) |

Machine-readable list of valid `domain` values (must match the table above):

```json
["xpu-kernel", "inductor", "upstream-pytorch"]
```

## Contracts

**`fix-root-cause`:**
- MUST emit `domain` from the JSON list above; if none applies, emit
  `NEEDS_HUMAN` instead of inventing a new value.
- MUST emit `target_repo` matching the registry entry for the chosen
  `domain`; a mismatch is a bug in triage's reasoning.
- MUST load only the `reference_file` matching the emitted `domain`.
  Do not load the other two.

**Orchestrators (`issue-handler`, `xpu-nightly-ci-fix`):**
- MUST look up the emitted `domain` in this registry before consuming
  triage output.
- MUST fail loudly (`NEEDS_HUMAN`, reason cites this file) on:
  - `domain` value not in the JSON list
  - `reference_file` does not exist in this directory
  - `target_repo` in triage output differs from the registry entry
- MUST NOT proceed with "no domain match" as a fallback — that was the
  silent-no-op bug this registry fixes.

## Adding a new domain

1. Create `.claude/skills/fix-root-cause/domain-<name>.md`.
2. Add one row to the table above.
3. Add `<name>` to the JSON list.
4. Update `fix-root-cause`'s SKILL.md Step 1 to describe when to emit
   `<name>` and to route to `domain-<name>.md`.
5. Optionally add `<name>` to `fix-skip-management`'s caller list if
   the domain interacts with skip decorators.
