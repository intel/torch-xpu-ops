# XPU Ops Review Checklist

Use this checklist for `torch-xpu-ops` PR reviews. It is intentionally focused on XPU-specific correctness, backend wiring, and risks that CI may not catch rather than style or lint.

## Scope And Ownership

- [ ] The change belongs in `torch-xpu-ops` rather than upstream PyTorch XPU, oneDNN XPU, or oneMKL
- [ ] The implementation location matches the intended backend path and registration strategy
- [ ] The PR does not duplicate an existing library or backend capability without justification
- [ ] The PR remains reviewable in one pass; if it exceeds 350 changed lines, call out the scope risk unless the size is clearly justified

## Code Quality

### Abstractions and Design

- [ ] **Clear abstractions** — State management is explicit; no dynamic attribute setting/getting
- [ ] **Match existing patterns in the same file** — Before accepting new code in a file, read how similar features are already implemented in that same file
- [ ] **No over-engineering** — Only requested changes are made; no speculative features
- [ ] **No premature abstraction** — Helpers and utilities are only created when reused

### Code Clarity

- [ ] **Self-explanatory code** — Variable and function names convey intent; minimal comments needed
- [ ] **Useful comments only** — Comments explain non-obvious context that cannot be inferred locally
- [ ] **No backward-compatibility hacks** — Unused code is deleted completely, not renamed with underscores or marked with "removed" comments
- [ ] **Documentation shows correct patterns only** — Code examples must have correct indentation, names, and syntax

## Correctness And Semantics

- [ ] CPU or CUDA parity claims were checked against the exact counterpart
- [ ] Input validation, output semantics, dtype promotion, and broadcasting behavior are still correct
- [ ] Empty tensors, zero-size dims, scalar tensors, and non-contiguous cases are handled or rejected intentionally
- [ ] `functional`, `out=`, `inplace`, view, and backward behavior still line up when relevant

## Async Execution And Synchronization

- [ ] No hidden host synchronization was introduced in a hot or common path
- [ ] Cross-stream ordering is explicit where the change relies on it
- [ ] New waits or synchronizations are required for correctness rather than masking a race
- [ ] The PR does not pull device results back to host just to drive control flow or debugging

## Layout, Memory Format, And Allocation

- [ ] Non-contiguous support is real rather than a silent `.contiguous()` fallback
- [ ] Channels-last support is either genuinely optimized or explicitly treated as a conversion path
- [ ] Output and temporary allocation choices preserve the intended memory format when needed
- [ ] Extra copies, format conversions, or oversized temporaries are justified
- [ ] **Memory format propagation** — Output tensors use `input.suggest_memory_format()` to preserve ChannelsLast or other input formats rather than defaulting to contiguous

## Dtype, Precision, And Numerics

- [ ] Compute dtype and accumulation dtype are appropriate for FP32, BF16, FP16, and integer paths
- [ ] Reductions, norms, softmax-style kernels, and atomic accumulation patterns were checked for numerical stability
- [ ] Autocast behavior is covered when the operator participates in mixed precision
- [ ] Test tolerances are chosen per dtype rather than copied blindly from another backend
- [ ] **Type promotion** — Manual dtype promotion logic should use established utilities rather than hand-written if/else chains

## Large Tensor Safety

- [ ] Index, offset, stride, and `numel` math are 64-bit safe where required
- [ ] Pointer arithmetic and flattened indexing helpers cannot overflow silently
- [ ] Large-stride or large-offset behavior has evidence beyond tiny tensors

## Kernel Mapping And Performance

- [ ] Work-group or subgroup decisions look deliberate rather than arbitrary
- [ ] The hot path is not made branch-heavy just to support rare cases
- [ ] Expensive queue, context, descriptor, or host-side setup is not repeated unnecessarily
- [ ] Claimed optimizations come with benchmark evidence or at least a concrete design rationale
- [ ] **No unnecessary allocations** — Tensors are not repeatedly created in hot loops
- [ ] **Appropriate in-place operations** — Use in-place ops where possible in performance-critical paths

## Dispatch, Fallback, And Generated Wiring

- [ ] Relevant yaml, native implementation, backward logic, and generated expectations move together
- [ ] Unsupported cases fail explicitly or intentionally fall back rather than taking the wrong path silently
- [ ] The XPU path exercised by tests is the same path wired by the registration
- [ ] **Operator tags** — New operators have appropriate tags (e.g., `pointwise`, `reduction`, `pt2_compliant_tag`)
- [ ] **Meta function / Composite fallback** — New operators should either have a Composite implementation or a clear justification for why they can only work on specific backends

## Thread Safety

- [ ] **No unprotected shared mutable state** — Shared data structures accessed from multiple threads are protected by locks or are inherently thread-safe
- [ ] **RAII lock guards** — Prefer `std::lock_guard` or `std::unique_lock` over manual `lock()`/`unlock()`
- [ ] **SYCL queue/stream synchronization** — Operations across different SYCL queues require explicit synchronization; missing synchronization can cause silent data corruption
- [ ] **No print statements** — No bare `print()` in production code; use proper logging utilities

## Test Coverage

### Test Existence

- [ ] Tests exercise the actual XPU path rather than only generic wrappers
- [ ] New functionality has corresponding tests
- [ ] Bug fixes include a regression test that reproduces the bug before the fix
- [ ] Tests are in the right place (added to existing test file next to related tests)

### Test Patterns

- [ ] **Device generic** — Tests checking compute results should happen in device-generic test classes via `instantiate_device_type_tests`
- [ ] **Use assertEqual for tensors** — Tensor comparisons use `assertEqual`, not raw assertions or `torch.allclose`
- [ ] **Use make_tensor** — Test tensors use `make_tensor(shape, device=device, dtype=dtype)` rather than `torch.rand(shape)` with implicit CPU/dtype
- [ ] **Use @dtypes** — Tests use the `@dtypes(...)` decorator rather than manual `for dtype in [...]` loops
- [ ] **Use @parametrize** — Tests use `@parametrize` rather than duplicating test methods that differ only in a parameter
- [ ] **Error conditions tested** — Expected exceptions are tested with `assertRaisesRegex`, not bare `assertRaises`

### Test Quality

- [ ] Coverage exists across relevant dtypes, layouts, shapes, and API variants
- [ ] Edge cases covered — boundary conditions, empty inputs, error cases
- [ ] Async-risk changes include stream-sensitive tests when appropriate
- [ ] Performance PRs include benchmark or regression evidence
- [ ] Test tolerances are chosen per dtype using `toleranceOverride` or equivalent

## CI/CD And Workflow Security

When reviewing changes to workflows, build scripts, or CI configuration:

- [ ] **No secrets in workflow files** — Secrets should not be hardcoded or echoed in workflow steps; use GitHub secrets mechanism properly
- [ ] **No `weights_only=False`** — `torch.load` calls should not disable safe deserialization unless absolutely justified
- [ ] **Protected branch rules respected** — Changes to merge rules, release workflows, or deployment environments require extra scrutiny
- [ ] **Immutable artifact references** — Docker images use immutable tags; no overwriting of published artifacts
- [ ] **No cache-dependent binaries in sensitive contexts** — sccache-backed builds are susceptible to cache corruption; these artifacts should not access sensitive info or be published for general use
- [ ] **Workflow trigger scope** — `pull_request_target` workflows must not check out PR head code into a trusted context without proper isolation
- [ ] **Token permissions minimized** — Workflow `permissions` block should request only what is needed (e.g., `contents: read` not `contents: write` unless required)
- [ ] **No arbitrary code execution from PR inputs** — PR title, body, branch name, and commit messages must not be interpolated into shell commands without sanitization
- [ ] **Third-party action pinning** — Actions are pinned to a full commit SHA, not a mutable tag (e.g., `actions/checkout@<sha>` not `actions/checkout@v4`)
- [ ] **CI logic changes clearly explain impact** — Changes to workflow triggers, conditions, or job structure must document what validation coverage is gained or lost

## Backward Compatibility

- [ ] Any user-visible behavior change is called out explicitly
- [ ] Error type, default behavior, output semantics, and determinism changes were reviewed as BC-sensitive
- [ ] If the change intentionally alters public behavior, the rationale is stronger than "tests pass"
