# Dependency Rules

The label list, per-value `evidence`, `code`, and `exists_in_repo` are
authoritative in `categories.dependency` of `proposed_labels.json`. This file
adds only the reasoning the JSON cannot carry. Do not restate label names here.

## Deciding the value

Resolve to exactly one taxonomy value, `none`, or `null`:

- A taxonomy value requires **direct evidence** — traceback, source, or operator —
  that the failure *originates* in that component. A library merely appearing in
  the call path never qualifies, and a keyword hit alone never qualifies (this
  axis is evidence-only, no `keywords`).
- `none` — evidence directly establishes no dependency.
- `null` — evidence is missing, ambiguous, or leaves two values equally supported.
- Runtime and build failures are distinct; never infer one from the other.
- For `oneMKL` / `oneDNN`, confirm the operator against `dependency_info.md`
  (its label `evidence` points to sections 1.1 / 1.2). Being *listed* there is not
  sufficient: the failure must originate in that library, and any Condition
  (section 1.1) or flip condition (section 2) must be confirmed, else `null`.

`AO` is NOT a dependency value: torchao is owned by the module axis
(`module: ao`). A transformers/huggingface failure is `third_party_packages`.

## Emitting the label

Match your decided value to a label by its `code`, then emit that label's `name`
verbatim (never the `code`). Every label uses the `dependency component: <...>`
prefix, but read `name` directly rather than templating it from `code`, since the
two are not always a mechanical match (e.g. `third_party_packages` ->
`dependency component: third_party`).

When a label's `exists_in_repo` is false (currently `oneCCL`, `IGC`,
`Level_Zero`), it does not exist in `intel/torch-xpu-ops` yet. `labels.md` is a
proposal, so emit it anyway and note in the reason that it must be created first.

If `extract.json`'s `dependency` is non-blank it is the issue's existing label:
preserve it, do not re-decide. Blank is not evidence of `none` — decide from the
rules above.

## Edge cases

- The dependency must be the direct cause, not incidental issue prose.
- Normalization ops (`batch_norm*`, `native_layer_norm`, `native_group_norm`),
  activations, and elementwise/reduction ops are native SYCL — never `oneMKL` or
  `oneDNN` (see `dependency_info.md` section 1.3).
- When keyword-scanning the body, ignore the `## Versions` / `Collecting
  environment` dump: it lists `onemkl`, `oneccl`, `intel-sycl-rt` for every issue.
