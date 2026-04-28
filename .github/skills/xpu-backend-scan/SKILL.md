---
name: xpu-backend-scan
description: >
  Use when triaging torch-xpu-ops backend scan tasks such as parity gaps,
  missing XPU implementation, fallback misuse, dispatch coverage, waiver checks,
  and true-vs-false bug separation for XPU operator behavior.
compatibility: >
  Designed for GitHub Copilot use in the torch-xpu-ops repository.
metadata:
  audience: backend-triage
  target-repository: intel/torch-xpu-ops
  version: "1.0"
---

# XPU Backend Scan

Use this as the single entry skill for backend triage work that was previously
split across separate Goal 1, Goal 2, Goal 3, and waiver skills.

This package is intentionally text-first and minimal:
- keep scan logic in this skill and its references
- keep only essential machine-readable data in assets
- do not depend on Python helper scripts to understand the workflow

## Load Order

Read these references first:
- `references/scan-basics.md`
- `references/goal-checklists.md` when you need a step-by-step decision pass for
  Goal 1, Goal 2, or Goal 3
- `references/dispatch-and-layout.md` when dispatch, structured, delegate,
  fallback, or source layout questions matter
- `references/report-template.md` when drafting a finding or review summary

Read `references/false-positive-patterns.md` when the current evidence is driven
by file-shape differences, helper call sites, test metadata, or stale wording
rather than a clear user-visible contract gap.

Read `references/triage-governance.md` only when reviewing historical findings,
deciding whether a recurring family is a real bug family, or deciding whether a
lesson belongs in automation versus documentation.

## What This Skill Covers

- Goal 1: static defects inside the XPU implementation itself
- Goal 2: CUDA versus XPU user-visible parity gaps
- Goal 3: missing XPU implementation or unsupported coverage where support is
  expected
- Waiver gate: NVIDIA-specific or hardware-justified exclusions that should not
  be filed as XPU bugs

## Hard Rules

- If CUDA has a feature, validation path, or native backend behavior that XPU
  lacks, treat that as an XPU bug unless the limitation is explicit and
  hardware-justified.
- If CUDA runs on GPU and XPU silently falls back to CPU, treat that as an XPU
  bug unless a waiver clearly applies.
- SYCL and CUDA can differ in style; implementation shape alone is not evidence
  of a bug.
- Prefer code-grounded findings over architecture commentary.
- For parity conclusions, require a user-visible contract gap rather than an
  optimization-only difference.
- Read helper definitions, structured delegates, and fallback paths before
  claiming algorithm divergence.
- Composite, decomposition, generic autograd, structured, or shared
  TensorIterator paths can satisfy support even when an XPU-local kernel file is
  absent.
- Distinguish family-level truth from row-level truth. A real bug family can be
  attached to the wrong overload or variant.

## Waiver Gate

Before spending time on Goal 1, Goal 2, or Goal 3 analysis, check whether the
operator falls into a waived category.

Use `assets/waivers.yaml` as the authority for waived families such as:
- NVIDIA software stack dependencies such as cuDNN, cuBLAS, cuSPARSE, NCCL, or
  CUDA runtime control ops
- hardware-specific NVIDIA-only features such as Tensor Core or FP8-only paths
- documented unsupported attention families where XPU support is intentionally
  absent
- vendor-specific backend integrations with no XPU equivalent

If a waiver clearly applies, return `WAIVED` and stop.
If a waiver may apply but the justification is incomplete, return
`NEEDS_HUMAN_REVIEW` instead of forcing a bug conclusion.

## Goal Routing

### Goal 1: Static XPU Defect

Use Goal 1 when the problem is intrinsic to the XPU implementation and does not
require CUDA as the reference.

Typical Goal 1 signals:
- broken or missing XPU dispatch connection despite source-backed code
- dtype, shape, stride, layout, alias, or inplace semantics defects
- missing guards, validation, or error-path checks on XPU
- silent CPU fallback or unreachable XPU path
- backward or autograd gaps for an otherwise supported XPU path
- clear race, atomic, or synchronization hazards visible in XPU code

Focus questions:
- is there a real dispatch path from schema to callable XPU implementation
- does the XPU path validate legal inputs and reject illegal ones correctly
- do inplace and out variants preserve expected semantics
- does the backward path exist when the forward path is meaningfully supported

### Goal 2: Parity Gap

Use Goal 2 when CUDA and XPU both have relevant support surfaces but the
user-visible contract diverges.

Only preserve Goal 2 conclusions when at least one of these is true:
- valid inputs differ between CUDA and XPU
- parameter semantics differ
- dtype, device, or layout support differs in a user-visible way
- backward semantics differ or are effectively unavailable
- result behavior, warnings, or error paths differ

By default, downgrade these to weak signals rather than parity gaps:
- optimization-only differences
- different vendor libraries
- different kernel organization or launch shape
- xfail or skip evidence without source-backed semantic difference
- call-site differences without reading helper definitions

### Goal 3: Missing XPU Implementation

Use Goal 3 when CUDA has meaningful support but XPU may have no usable path.

Strong Goal 3 evidence requires most of the following:
- CUDA has a usable implementation or dispatch path
- XPU has no native dispatch or usable implementation
- no fallback exists that would make the behavior callable
- no composite, decomposition, structured delegate, or shared path already
  covers the semantics
- no waiver or documented unsupported rationale applies

Do not call something missing implementation solely because a CUDA entry exists
in native_functions.yaml. First exclude stub, NYI, error-only, composite,
delegate, and fallback cases.

## Priority Sources

Read these sources in the target repository first, then consult upstream
`pytorch/pytorch` when exact peer semantics or upstream schema confirmation is
needed:

- `yaml/native/native_functions.yaml`
- `yaml/xpu_functions.yaml`
- `src/ATen/native/xpu/XPUFallback.template`
- `src/ATen/native/xpu/`
- `src/ATen/native/xpu/sycl/`
- `aten/src/ATen/native/native_functions.yaml`
- `tools/autograd/derivatives.yaml`
- `torch/_refs/__init__.py`
- `aten/src/ATen/native/cuda/`
- `aten/src/ATen/native/transformers/cuda/`
- `aten/src/ATen/native/transformers/xpu/`
- `test/xpu/skip_list_common.py`

## Decision Labels

Use only these verdict labels:
- `LIKELY_XPU_BUG`
- `PARITY_GAP`
- `MISSING_XPU_IMPL`
- `LIKELY_OK`
- `NEEDS_HUMAN_REVIEW`
- `WAIVED`

Do not invent local aliases. Put nuance in the explanation fields instead.

## Output Discipline

- Return only findings supported by inspected code.
- If differences are stylistic, hardware-justified, or already explained by a
  valid shared path, return no issue or downgrade confidence.
- Do not call a finding runtime-confirmed from static code review alone.
- Use `references/report-template.md` for consistent finding structure.
- For issues that deserve local reproduction, hand off to the repository's
  standard issue-filing workflow or issue template only after a local XPU repro
  actually exists in the target repository workflow.