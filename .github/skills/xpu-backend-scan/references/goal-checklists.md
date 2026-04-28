# Goal Checklists

Use these lists when you want a fast, repeatable pass over Goal 1, Goal 2, or
Goal 3 without expanding the main skill text.

## Common Gate Before Any Goal

- Confirm the exact schema or overload in local
   `yaml/native/native_functions.yaml` first, and use upstream
   `aten/src/ATen/native/native_functions.yaml` when exact peer schema
   confirmation is still needed.
- Check waiver applicability first.
- Check fallback, composite, decomp, structured delegate, and shared-path
  coverage before claiming a gap.
- Separate runtime coverage evidence from auxiliary metadata such as
  `xpu_functions.yaml` or test annotations.
- If the op is a wrapper, identify the real implementation target before
  deciding anything.

## Goal 1 Checklist: Static XPU Defect

Ask these in order:

1. Does XPU have a real callable path from schema to implementation?
2. If fallback exists, is the problem actually a CPU fallback issue rather than
   a missing implementation claim?
3. Are registration, `TORCH_IMPL_FUNC`, or backend-YAML signals consistent with
   the source tree?
4. Are dtype, shape, stride, layout, alias, inplace, or out semantics guarded
   correctly?
5. Is there a backward or autograd path when the forward path is materially
   supported?
6. Is the evidence specific to XPU, rather than shared with CUDA or a common
   wrapper?

Strong Goal 1 keepers:
- broken dispatch to landed XPU code
- silent CPU fallback
- missing XPU validation or version-bump behavior
- clear XPU-side numeric, atomic, or alias defect

Downgrade or stop when:
- the path is actually shared or generic
- the difference is only implementation shape
- a wrapper or delegate already supplies the missing behavior

## Goal 2 Checklist: User-Visible Parity Gap

Ask these in order:

1. Do CUDA and XPU both expose relevant support surfaces for the same schema?
2. Is the difference user-visible, not merely structural?
3. Does the difference affect legal inputs, parameter semantics, dtype or layout
   support, backward behavior, result behavior, warnings, or errors?
4. Did you read helper definitions, not just call sites?
5. Did you exclude optimization-only, vendor-library, launch-shape, and test
   metadata-only differences?

Strong Goal 2 keepers:
- XPU rejects a CUDA-valid configuration
- a stateful parameter is ignored or handled differently
- backward semantics differ in a way visible to users
- a missing branch changes public behavior or exposed capability

Downgrade or stop when:
- the only evidence is file layout, helper call-site shape, or vendor library
  choice
- both sides already share a valid wrapper, delegate, or composite path
- the claim relies only on skip or xfail evidence

## Goal 3 Checklist: Missing XPU Implementation

Ask these in order:

1. Does CUDA actually have a usable implementation or dispatch path?
2. Is there truly no XPU runtime coverage through backend YAML, source-backed
   registration, fallback, structured path, composite path, decomp, or shared
   helper?
3. Is the op free of waiver or documented unsupported rationale?
4. Is the apparent absence more than a missing local file or missing hand-written
   wrapper?
5. Is the gap real for the exact variant, overload, or dtype being discussed?

Strong Goal 3 keepers:
- CUDA has real support
- XPU has no usable path
- no fallback or shared path exists
- no waiver applies

Downgrade or reroute when:
- fallback exists, making this a Goal 1 concern
- CUDA is stub, NYI, or error-only
- XPU support comes from a shared wrapper, delegate, or generic path
- the evidence is stale historical wording rather than current source-backed
  facts

## Minimum Evidence Before Emitting A Verdict

- `LIKELY_XPU_BUG`: a concrete XPU-side defect with file-grounded evidence
- `PARITY_GAP`: a concrete user-visible contract difference with peer evidence
- `MISSING_XPU_IMPL`: no usable XPU path after exclusions and coverage checks
- `LIKELY_OK`: a strong shared-path, fallback, decomp, delegate, or non-semantic
  explanation
- `NEEDS_HUMAN_REVIEW`: mixed evidence remains after the standard checks
- `WAIVED`: the op matches a justified exclusion category