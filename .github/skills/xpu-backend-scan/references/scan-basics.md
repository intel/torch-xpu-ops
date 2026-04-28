# Scan Basics

This reference condenses the default scan-time material for the
`xpu-backend-scan` skill package.

## Repository Assumptions

- Treat the current workspace root as the `torch-xpu-ops` repository root.
- Read local files relative to that root.
- Do not assume a local upstream PyTorch checkout exists. When upstream files
  are needed, inspect the matching paths in `pytorch/pytorch` through
  GitHub-aware tools or an optional local checkout.
- Treat shell snippets as examples, not mandatory commands.

## Four-Layer Model

Before issuing any verdict, keep the target at four layers:

| Layer | Meaning |
|---|---|
| op | Base operator family such as `addmm` |
| variant or overload | Specific schema such as `addmm.out`, `addmm_`, or `addmm.Scalar` |
| dtype | fp32, bf16, half, complex, and so on |
| dispatch | Runtime path such as CUDA, XPU, CompositeImplicitAutograd |

Do not compare by filename alone.

## Common Pre-Check

Before Goal 1, Goal 2, or Goal 3 analysis:

1. Confirm the exact schema in `aten/src/ATen/native/native_functions.yaml`.
2. Check whether XPU, CUDA, composite, or structured delegate coverage already
   exists.
3. Apply the waiver gate before spending time on deeper analysis.
4. If the op is a wrapper or structured delegate, follow the real target rather
   than judging the wrapper in isolation.
5. Separate runtime coverage evidence from auxiliary metadata.

## Goal Summary

### Goal 1: Static XPU Defect

Use Goal 1 when the problem is intrinsic to the XPU implementation and does not
need CUDA as the reference.

Common Goal 1 buckets:
- registration or dispatch defects
- dtype, shape, stride, layout, alias, or inplace issues
- missing input validation or incorrect error paths
- silent CPU fallback or unreachable XPU path
- backward or autograd gaps for an otherwise supported path
- obvious race, atomic, or synchronization hazards visible in XPU code

### Goal 2: User-Visible Parity Gap

Use Goal 2 only when CUDA and XPU both have relevant support surfaces but the
user-visible contract diverges.

Strong Goal 2 signals:
- valid input space differs
- parameter semantics differ
- dtype, device, or layout support differs in a user-visible way
- backward semantics differ
- result behavior, warnings, or error paths differ

Weak signals that should not become parity conclusions by themselves:
- optimization-only differences
- library choice differences such as oneDNN versus cuDNN
- launch-shape or file-layout differences
- xfail or skip evidence without source-backed semantic difference

### Goal 3: Missing XPU Implementation

Use Goal 3 when CUDA has meaningful support but XPU may have no usable path.

Strong Goal 3 evidence requires most of the following:
- CUDA has meaningful support
- XPU has no native dispatch or usable implementation
- no fallback exists that would make the op callable
- no composite, decomposition, structured delegate, or shared path already
  covers the semantics
- no waiver or justified exclusion applies

## Negative List

Do not flag the following as bugs by themselves:

- SYCL and CUDA syntax differences
- different vendor library names with equivalent semantics
- different kernel organization or helper file layout
- codegen-produced wrapper differences
- lack of a dedicated XPU kernel file when a shared or structured path exists
- absence of an XPU-specific test alone
- optimization-only differences where results remain equivalent
- fallback presence as a Goal 3 bug; fallback is a Goal 1 concern instead

## Evidence Standard

Each finding should be grounded in inspected source and answer these questions:

- What exact schema or variant is being discussed?
- What runtime coverage signals already exist?
- What concrete user-visible defect or gap remains after exclusions?
- Which files provide the positive and negative evidence?
- Is this a true operator-family issue or only a mislabeled row?

## Output Discipline

- Use only canonical verdict labels from the main skill.
- Do not call a static review runtime-confirmed.
- Downgrade confidence when only weak signals exist.
- If a local XPU repro is later obtained in the target repository workflow,
  hand off to the issue-creation skill rather than duplicating issue-filing
  instructions here.