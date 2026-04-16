# Torch XPU Ops Review Notes

This file is a repository-specific overlay for `torch-xpu-ops` reviews. Use it together with the generic PyTorch review references rather than as a replacement for them.

## Repository File Map

Inspect these first when the change touches operator wiring, kernels, or tests:

- `yaml/xpu_functions.yaml`
- `yaml/native/native_functions.yaml`
- `src/ATen/native/xpu/`
- `src/ATen/native/xpu/sycl/`
- `src/ATen/native/xpu/XPUFallback.template`
- `test/xpu/`
- `test/test_ops_xpu.py`
- `.github/workflows/` when the PR changes CI, packaging, or platform coverage

## What To Verify For Operator Changes

1. Schema, dispatch, and backend wiring change together.
2. The implementation location matches the registration path.
3. The XPU path is actually exercised by tests, not just CPU or generic wrappers.
4. CUDA or CPU parity claims are checked against the exact upstream counterpart.
5. Forward, backward, and structured or generated paths still line up after the change.

## Common Torch XPU Ops Review Patterns

| Concern family | What to verify | Typical outcome once verified |
|---|---|---|
| CUDA and XPU parity complaint | Check the exact CUDA upstream file and line | Often `Not valid` if CUDA behaves the same way |
| Dtype dispatch reported as too narrow | Compare the XPU `AT_DISPATCH_*` macro against CUDA upstream | Often `Not valid` if the macro matches upstream coverage |
| Batched sparse CSR or `dim > 2` complaint | Check whether CUDA and CPU support that path at all | Often `Not valid` if the path is unsupported everywhere |
| `.wait()` blocking concern | Verify queue ordering, actual behavior change, and target execution model | Sometimes `Technically correct but not practically significant` on in-order queue paths |
| Missing registration concern | Inspect yaml, generated wiring, and actual implementation file together | Do not conclude from one file alone |

## XPU-Specific Review Checks

- Confirm XPU registration in the relevant yaml rather than inferring support from one source file.
- Check whether the PR changes generated-code expectations, not just hand-written kernels.
- Prefer XPU device-specific test coverage or existing operator coverage over isolated one-off tests.
- When reviewing SYCL code, check read-only vs writable pointer use, address-space assumptions, and synchronization behavior explicitly.
- Treat silent fallback behavior as a correctness question, not just a performance question.

## Thread Reply Conventions

- Start with a bold verdict.
- Keep each public reply to roughly 2-4 sentences unless the user explicitly asks for more detail.
- Cite an exact repo-relative or upstream file and line whenever possible.
- Do not use thanks, apologies, or generic praise.
- Add the `*[AI-assisted reply]*` footer only when the user wants repo-ready reply text.

## Bot Comment Rules

Do not agree with or duplicate automated comments from these sources:

- Any self-authored automated comment ending with `*[AI-assisted reply]*`
- Any self-authored automated comment containing `Requested in [this mention]`
- `copilot[bot]`
- `github-actions[bot]`

If the same file and line already have an equivalent self-authored automated reply, skip the duplicate unless there is materially new evidence.

## How To Use The Generic References Here

- Use `pytorch-pr-review-skill.md` for the overall review workflow and output discipline.
- Use `review-checklist.md` for CI-blind infrastructure, testing, and safety checks.
- Use `bc-guidelines.md` when a user-visible behavior change might alter public semantics, exceptions, defaults, or compatibility.
- Use this file to adapt those generic rules to `torch-xpu-ops` file layout, review patterns, and common false positives.