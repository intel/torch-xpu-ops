# XPU Ops Review Checklist

Use this checklist for `torch-xpu-ops` PR reviews. It is intentionally focused on XPU-specific correctness, backend wiring, and risks that CI may not catch rather than style or lint.

## PR Presentation & Structure

- [ ] **Clear, descriptive PR title** — Summarizes the main purpose (e.g., "Fix memory leak in convolutional layers by optimizing tensor allocations", not "Bug fixes and improvements")
- [ ] **Description answers "Why?" and "How?"** — not "What?" (the diff already shows what changed)
- [ ] **Bug-fix PRs link to an issue** — The issue number is referenced and the PR states how it resolves the issue
- [ ] **Feature PRs reference an RFC or prior discussion** — Non-trivial features should have been discussed before implementation
- [ ] **Related PRs in a series are linked** — If the PR is part of a sequence, related PRs are referenced for context
- [ ] **Commit messages are meaningful** — Accurately describe changes; facilitate navigation through PR history
- [ ] **Draft/WIP marking** — If not ready for review, the PR is marked as draft with [WIP] in the title
- [ ] **Missing tests are justified** — If the PR has no tests, the reason is explicitly explained in the description

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

### Variable & Function Naming

- [ ] **No single-letter variables** — except trivial loop counters (`i`, `j`, `k` in short loops)
- [ ] **No ambiguous abbreviations** — `buf`, `tmp`, `val`, `res`, `ret`, `cnt` without context should use full words
- [ ] **No type-encoded names** (Hungarian notation) — `iCount`, `strName`, `pPtr`
- [ ] **Consistent casing** — no mixing `camelCase` and `snake_case` within the same scope/file
- [ ] **Bool names read as predicates** — `is_ready`, `has_data`, `should_flush`, not `ready_flag`, `data_status`
- [ ] **Name matches semantics** — a variable named `index` should not hold a count; `size` should not hold a capacity

### No Magic Numbers

- [ ] **No unexplained numeric literals** — in conditionals, arithmetic, or indexing (0, 1, -1 in common idioms are OK)
- [ ] **No repeated unnamed literals** — same magic number in multiple places must be a named constant
- [ ] **No unexplained bit operations** — bit shifts/masks must have named constants or comments
- [ ] **No inline hardware constants** — register offsets, queue sizes, tile dimensions must be named

### Comments & Readability

- [ ] **No commented-out code** — without explanation; should be deleted or have a TODO
- [ ] **No stale comments** — comments must not contradict the code
- [ ] **No parroting comments** — `i++; // increment i` adds nothing
- [ ] **Complex logic is commented** — multi-step transformations, non-trivial math, bitwise operations need WHY comments
- [ ] **Non-trivial public functions have doc comments** — purpose, params, return value

## Intel GPU Terminology — MANDATORY

- [ ] **SYCL programming terms used in code** — subgroup size (not SIMD width), work-item (not SIMD lane), work-group (not thread block/threadblock)
- [ ] **Hardware terms only in optimization comments** — XVE, XMX, XC terms appear only in comments explaining WHY an optimization was chosen
- [ ] **No deprecated hardware terms** — EU → XVE, Subslice/DSS → Xe-core (XC), Systolic → XMX, HW thread → XVE thread
- [ ] **API boundary legacy names are wrapped** — legacy API names at call site only, with modern-named helper and comment

## General Code Quality

Also flag during review:

- [ ] **Dead code** — unreachable branches, unused variables, unused `#include`
- [ ] **Error handling** — bare `catch(...)`, swallowed errors, missing return code checks
- [ ] **Resource management** — raw `new`/`delete` without RAII in C++, unclosed handles
- [ ] **Copy-paste patterns** — duplicated blocks that should be factored into a function
- [ ] **Overly long functions** — >80 lines — suggest splitting
- [ ] **Deep nesting** — >3 levels of `if/for` — suggest early returns or extraction

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
- [ ] **Math functions in SYCL device code**
  - [ ] Real types use `sycl::<fn>` (e.g., `sycl::exp`), not `std::<fn>`
  - [ ] `sycl::native::<fn>` is used only when it exists AND the matching CUDA kernel uses a fast intrinsic (e.g., `__expf`)
  - [ ] Complex types use `std::<fn>`, not `sycl::<fn>` (see `src/ATen/native/xpu/sycl/UnaryKernels.cpp` for the real/complex split pattern)
  - [ ] When `scalar_t` may be `c10::Half` and the `<fn>` has a `sycl::half` overload, arguments are cast to `at::opmath_type<scalar_t>` to avoid overload ambiguity (`c10::Half` converts to both `float` and `sycl::half`)

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
- [ ] **Write permissions justified by GITHUB_TOKEN usage** — Every write permission declared must correspond to an actual API call using `GITHUB_TOKEN`; permissions needed only by PATs or other secrets should not be granted to the default token
- [ ] **No arbitrary code execution from PR inputs** — PR title, body, branch name, and commit messages must not be interpolated into shell commands without sanitization
- [ ] **Third-party action pinning** — Actions are pinned to a full commit SHA, not a mutable tag (e.g., `actions/checkout@<sha>` not `actions/checkout@v4`)
- [ ] **CI logic changes clearly explain impact** — Changes to workflow triggers, conditions, or job structure must document what validation coverage is gained or lost

## Backward Compatibility

- [ ] Any user-visible behavior change is called out explicitly
- [ ] Error type, default behavior, output semantics, and determinism changes were reviewed as BC-sensitive
- [ ] If the change intentionally alters public behavior, the rationale is stronger than "tests pass"
