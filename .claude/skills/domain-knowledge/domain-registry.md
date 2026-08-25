# Domain Registry (shared knowledge base)

Shared, separately-maintained knowledge base for the fix pipeline.
Both `fix-root-cause` and `fix-implement` load this registry to
resolve which domain knowledge file(s) apply to a failure. It lives
in its own folder (`.claude/skills/domain-knowledge/`) rather than
co-located under any one skill, because the same domain knowledge is
consumed by more than one skill: `fix-root-cause` uses it to route
`target_repo` / locate code, and `fix-implement` reuses the same
path conventions and fix recipes.

This file is the single source of truth for:

1. **Which `domain` values the pipeline is allowed to emit** — the
   JSON array below is the closed set.
2. **Which reference file each `domain` value maps to** — the
   knowledge file loaded on demand.
3. **Which `target_repo` each `domain` implies** — used to
   sanity-check the independently-derived `target_repo`.

## Loading contract (may load MORE THAN ONE domain)

Loading is **need-driven**, not one-per-run. A single failure can
span multiple domains and the caller loads every reference file that
applies:

1. Always read this registry first (the closed-set lookup). It is
   small and cheap.
2. Match the failure against the `applies_when` column. A failure
   may match **several** rows (e.g. an Inductor UT that fails
   because of a missing XPU kernel matches both `inductor` and
   `xpu-kernel`).
3. Load **every** matching `reference_file` — not just the first.
   Emit them in a `domains` array with the domain that owns the
   *root cause* first; that first entry drives `target_repo`. The
   rest are loaded for their path conventions / recipes but do not
   change `target_repo`.
4. If nothing matches, emit `NEEDS_HUMAN(reason=no_registered_domain)` —
   never invent a value, never silently no-op.

Progressive disclosure is preserved: the closed set (this file) is
always in context; the deep per-domain knowledge only enters context
for the domains that actually matched.

## Registry

| domain | reference_file | target_repo | applies_when | test_locations | fix_locations | module_labels |
|---|---|---|---|---|---|---|
| `xpu-kernel` | `domain-xpu-kernel.md` | `torch-xpu-ops` | root cause in XPU backend kernels, dispatch, or SYCL code | `test/xpu/`, `test/inductor/` (XPU re-enabled tests) | `src/ATen/native/xpu/` | `module: op impl`, `module: torch-ops-eltwise`, `module: torch-ops-gemm`, `module: torch-ops-reduction`, `module: torch-ops-others`, `module: sdpa`, `module: quant`, `module: sparse` |
| `inductor` | `domain-inductor.md` | `pytorch` | root cause in `torch._inductor` or `torch._dynamo` (device-agnostic) | `test/inductor/` | `torch/_inductor/`, `torch/_dynamo/` | `module: inductor`, `module: dynamo`, `module: fx` |
| `upstream-pytorch` | `domain-upstream-pytorch.md` | `pytorch` | root cause in device-agnostic pytorch core (framework regressions, test infra) | anywhere in `test/` | `torch/`, `aten/` (non-CUDA/XPU dirs) | `module: core`, `module: distributed`, `module: infra`, `module: build` |

Machine-readable list of valid `domain` values (must match the table above):

```json
["xpu-kernel", "inductor", "upstream-pytorch"]
```

## `domain` vs the repo's `module:` labels

`domain` is a coarse 3-value **routing key** internal to the fix
pipeline: it selects the `target_repo`, the `reference_file` to load,
and the code directories to inspect. It is deliberately a small closed
set so each value maps to exactly one `domain-*.md` and one repo.

The repo already has ~25 fine-grained `module: xxx` GitHub labels for
human issue triage (`module: op impl`, `module: inductor`,
`module: dynamo`, `module: core`, ...). These are a *different, finer*
taxonomy — one `domain` spans many `module:` labels. So we do **not**
rename `domain` to `module`; that would either explode the routing
closed set to 25 values or make many labels point at one reference
file, defeating the registry.

Instead the `module_labels` column reuses the existing repo labels: it
lists the candidate `module:` labels an issue in that domain should
carry. Pipeline mode picks the most specific matching label from this
column (after root-cause has located the actual code) rather than
inventing a new label. Routing stays on `domain`; issue labelling
reuses `module:`.

## Contracts

**Consumers (`fix-root-cause`, `fix-implement`):**
- MUST read this registry before consuming or producing domain info.
- MUST load the `reference_file` for **every** matching domain, not
  just one — see the loading contract above.
- MUST derive `target_repo` from the first entry in `domains` (the
  root-cause domain).
- MUST emit every applied domain in a `domains` array, root-cause
  domain first. All values MUST come from the JSON list above; if
  none applies, emit `NEEDS_HUMAN(reason=no_registered_domain)`
  instead of inventing one.
- MUST NOT emit a `target_repo` that disagrees with the first
  `domains` entry's registry row — a mismatch is a bug in the
  reasoning.

**Orchestrators (`issue-handler`, `xpu-nightly-ci-fix`):**
- MUST look up every emitted domain in this registry before consuming
  pipeline output.
- MUST fail loudly (`NEEDS_HUMAN`, reason cites this file) on:
  - any emitted domain not in the JSON list
  - a `reference_file` that does not exist in this directory
  - `target_repo` differing from the first `domains` entry's registry row
- MUST NOT proceed with "no domain match" as a fallback — that was the
  silent-no-op bug this registry fixes.
- In pipeline mode, when labelling an issue, MUST pick `module:`
  labels from the applied domains' `module_labels` columns rather than
  inventing new ones.

## Adding a new domain

1. Create `.claude/skills/domain-knowledge/domain-<name>.md`.
2. Add one row to the table above, including a `module_labels` cell
   listing the existing repo `module:` labels that apply to the domain.
3. Add `<name>` to the JSON list.
4. Update `fix-root-cause` Step 1 and `fix-implement`'s domain-load
   step to describe when `<name>` applies.
5. Optionally add `<name>` to `fix-skip-management`'s caller list if
   the domain interacts with skip decorators.
