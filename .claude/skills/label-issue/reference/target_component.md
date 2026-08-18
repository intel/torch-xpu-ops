# Target Component Rules

The fix location determines ownership. Do not use labels, domains, or the
location where the failure surfaces as a substitute for tracing the code path.

## Dependency precedence

This axis is decided AFTER the dependency axis, and consumes its result.

When the dependency axis returned a **taxonomy value**, that component owns the
fix, and it becomes the `target_component` verbatim:

- `target_component` -> the dependency value itself (`driver`, `IGC`,
  `Level_Zero`, `oneMKL`, `oneDNN`, `oneCCL`, `oneAPI`, `MSVC`, `Triton`,
  `community`, `third_party_packages`)
- `need_action` -> `NEED_FIX_3RDPARTY`

Name the component, never the generic `third-party` bucket: a reader must be
able to tell WHICH external component owns the fix from `target_component`
alone. `target_component: oneDNN`, not `target_component: third-party`. Spell
the value exactly as the dependency taxonomy spells it, so the two axes agree
(`Level_Zero`, `third_party_packages`, `oneDNN` — matching case and
underscores).

This overrides the fix-location table below, including a traced path inside
`pytorch/` or `third_party/torch-xpu-ops/` — a confirmed dependency means the
product frame is the caller, not the defect. Cite the dependency value as the
deciding signal in the reason.

`none` and `null` are NOT dependencies. Neither names a component and neither
returns `NEED_FIX_3RDPARTY`; decide ownership from the fix-location table
alone. A `null` means the dependency axis was inconclusive, never that a third
party was confirmed.

The single exception to the override is a Windows `os`, which stays
`NEED_HUMAN`. Its `target_component` still names the component when the
dependency axis confirmed one (typically `MSVC`).

## Fix location

Used only when the dependency axis returned `none` or `null`.

| Path that must change | `target_component` |
|---|---|
| The **test file itself**, such as a stale assertion or tolerance, missing `sys.path` or import fix, an un-generalized CUDA-only call, a missing-attribute skip guard, or a syntax error, while product code is correct | `test-case` |
| `pytorch/aten/`, `pytorch/torch/`, or `pytorch/c10/`, excluding `third_party/torch-xpu-ops/` and the test file | `pytorch` |
| `third_party/torch-xpu-ops/`, or `src/` in a standalone torch-xpu-ops checkout | `torch-xpu-ops` |
| No clean path resolves to product code or the test case | `N/A` |

If the test correctly calls the API and the underlying dispatch or kernel is
wrong, ownership is `pytorch` or `torch-xpu-ops`, not `test-case`.

An external owner never reaches this table — an external owner means the
dependency axis confirmed a taxonomy value, which the precedence section
already resolved. If you believe a third party owns the fix but the dependency
axis returned `none` or `null`, the dependency axis was decided wrong: revisit
Step 3 rather than writing an unnamed external owner here.

## Required evidence

The delegated trace must cite file and line references, relevant symbols, and
the call path from the test or API to the failure. If pytorch_folder is not blank, it must inspect the local
checkout using `pytorch_folder` and check `git log` for an upstream or local
commit that already fixes the exact root cause. Record the commit when found.

Skip and xfail decorators are not fixes. Their presence confirms that the
failure remains relevant and cannot support an already-fixed conclusion.

## Canonical verdicts

Apply in order; the first matching row wins.

| Ownership or condition | `need_action` |
|---|---|
| `os` is Windows | `NEED_HUMAN` |
| The dependency axis returned a taxonomy value | `NEED_FIX_3RDPARTY` |
| Test file itself | `NEED_FIX_CASE` |
| `pytorch` product code | `NEED_FIX` |
| `torch-xpu-ops` product code | `NEED_FIX` |
| Inconclusive trace | `NEED_FIX` |
| No resolvable path, or human planning required for another reason | `NEED_HUMAN` |

An existing upstream fix may be reported in `evidence.upstream_fix`, but do
not call a skip or xfail an upstream fix.
