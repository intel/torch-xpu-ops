# Priority Rules

Priority is the PyTorchXPU project's `Priority` field (Fields -> Priority), not a
GitHub label. The four options and their canonical tier names live in the
top-level `priority_field` section of `proposed_labels.json`; read the tier ->
option mapping and the per-tier `keywords` from there rather than hard-coding
them here. The keywords are only a fallback hint — the decision tree below is
primary.

## Priority tiers

Tier names (and their project field option, from `priority_field` in
`proposed_labels.json`): `Urgent` (`P0`), `High` (`P1`), `Medium` (`P2`),
`Low` (`P3`).

## Decision Priority Order

When `extract.json` already carries a non-empty `priority` from the PyTorchXPU
project field, preserve it verbatim and skip this section entirely — a human
already set it. The order below derives a priority only when that field is `""`.

An issue often matches rows in more than one tier. Evaluate the tiers in
severity order and stop at the first tier with a matching row:

1. `Urgent`
2. `High`
3. `Medium`
4. `Low`

Within a tier, any single matching row is enough; the rows inside a tier are
alternatives, not requirements. Emit exactly one tier.

Two specificity exceptions override that order, because a quantified rule beats
the generic `Regression` row it would otherwise be swallowed by:

- A **performance** regression is scored only by its measured percentage:
  `Urgent` when >7%, `Medium` when <=7%. Do not score it as High `Regression`.
- A **benchmark accuracy** regression is High `Benchmark accuracy regression`,
  not Medium `Functional error without a crash`.

Everything else follows the plain severity order. So a feature gap that also
regressed between versions is High (`Regression`) rather than Medium
(`Feature gap blocking tests`), and an enhancement request that also reports a
current functional error is Medium (`Functional error without a crash`) rather
than Low (`Enhancement or feature request`).

When no row in any tier matches, emit `Medium`.

Case-count rows (`>6 failed UT cases`, `1-6 failed UT cases`) are the one
exception to Step 1.5's per-case scoping: count every `test_cases[]` entry in
the whole issue, not just the analyzed case. Severity is a property of the
issue as a whole, so a 4-case issue is Medium even though only case 1 was
root-caused.

## Urgent

Assign `Urgent` when any of these hold:

1. A measured, quantified performance regression between releases is greater
   than 7%.
2. A crash or segfault occurs on a Core API listed below.
3. A legacy Torch build failure prevents a build.

### Core API list

- **Module loading:** `import torch`, `import torch.nn`.
- **Tensor lifecycle:** `torch.tensor()`, `torch.zeros()`, `torch.ones()`,
  `torch.empty()`, `torch.rand()`, `.clone()`, `.contiguous()`.
- **Device transfer:** `.to(device)`, `.cuda()`, `.cpu()`, `.xpu()`.
- **Basic arithmetic:** `+`, `-`, `*`, `/`, `@` (matmul) on standard dtypes
  (`float32`, `float64`, `int64`).
- **Autograd core:** `.backward()`, `torch.autograd.grad()` on a <=3-op graph.
- **Key module forward/backward:** `nn.Linear` and `nn.Conv2d` forward or
  backward.
- **Serialization:** `torch.save()`, `torch.load()`.
- **Basic indexing:** `tensor[idx]`, `.view()`, `.reshape()`.

| Condition | Evidence | Examples |
|---|---|---|
| Performance regression >7% | Issue body or comments cite a measured percentage and confirm a regression between releases. The drop must be >7%, not <=7%. | 15% slower on 2.12 than 2.11: yes. 5% slower: no. |
| Crash on a Core API | Stack trace shows SIGSEGV or an access violation on a Core API, or a segfault during module import, tensor creation, `.backward()`, device transfer, matmul, `nn.Linear` or `nn.Conv2d` forward, serialization, or indexing. | SIGSEGV in `torch.tensor()`: yes. Crash in a custom op: no. |
| Legacy Torch build failure | CI log shows compilation or linker errors that prevent a build. | Compilation failures, linker errors. |

## High

| Condition | Evidence | Examples |
|---|---|---|
| More than 6 failed UT cases | Count distinct test-case names across the whole issue (`extract.json`'s `test_cases[]`), not just the Step 1.5 analyzed case. Meta-tracking issues may list many cases. | Large test-class failures. |
| Regression | The issue cites a version where it passed and a current version where it fails. | Passed on 2.10, fails on 2.11. |
| Hang or timeout | The process remains alive but is stuck, such as after a 300-second timeout, infinite wait, or deadlock. | Distributed test hangs. |
| Benchmark accuracy regression | Benchmark accuracy passed in a prior release and now fails without a crash. | `fail_accuracy` on an E2E model. |

## Medium

| Condition | Evidence | Examples |
|---|---|---|
| Benchmark performance regression <=7% | A measured performance drop is cited and is no greater than 7%. | Minor throughput decrease. |
| 1-6 failed UT cases | A small number of related test failures, counted across the whole issue, not just the analyzed case. | A few operator tests failing. |
| Functional error without a crash | RuntimeError, AssertionError, or an incorrect result while the process continues. | Wrong output, type errors. |
| Feature gap blocking tests | Tests fail because an API is not implemented, without a crash. | Not implemented errors. |

## Low

| Condition | Evidence | Examples |
|---|---|---|
| Enhancement or feature request | The title asks for new functionality — `implement`, `enable`, `support`, `RFC`, `consider`, `investigate` — or the body describes desired new functionality. A failure reporting `not implemented` / `NotImplementedError` is NOT this row; it is Medium `Feature gap blocking tests`. | Feature requests. |
| Validation or error-message difference | XPU raises a different error from CPU or CUDA, or does not raise when it should, without incorrect computation. | Error-message mismatch. |
| Minor, cosmetic, or warning issue | Warning mismatch, deprecated API usage, or documentation gap. | Warning mismatches. |
| CUDA alignment without a functional break | XPU behavior differs from CUDA but is not incorrect. | Dtype support alignment. |
