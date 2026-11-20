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
- [ ] **Useful comments only** — Comments explain non-obvious context that cannot be inferred locally. For large comments use `// Note [Good note title]` and `// See Note [Good note title]` (C++; use `# Note [...]` / `# See Note [...]` in Python) to write larger comments that can be referenced from multiple places in the codebase. Complex logic should have a comment explaining it
- [ ] **No commented-out code** — Dead code left in comments must be removed, not kept "just in case"
- [ ] **No stale comments** — Comments must match the code they describe; a comment contradicting its adjacent code is a bug signal, not just a style nit
- [ ] **No backward-compatibility hacks** — Unused code is deleted completely, not renamed with underscores or marked with "removed" comments
- [ ] **Appropriate complexity** — Solutions are as simple as possible for the current requirements
- [ ] **Documentation shows correct patterns only** — Docs and markdown files should show the right way to do things directly, not anti-patterns followed by corrections. Code examples must have correct indentation, names, and syntax
- [ ] **Non-trivial public functions have doc comments** — Public APIs with non-obvious behavior, preconditions, or side effects document them

### Naming

- [ ] **Naming conventions** — C++ functions/vars `snake_case`, types `PascalCase`, macros `UPPER_SNAKE_CASE`
- [ ] **No vague or single-letter names** — except short loop indices; no vague `buf`/`tmp`/`val`/`res`/`ret`/`cnt` without context
- [ ] **No Hungarian notation or mixed casing** — no `iCount`/`pPtr`; no mixing `camelCase` and `snake_case` in the same scope
- [ ] **Bools read as predicates** — `is_contiguous`, `has_bias`
- [ ] **Name matches meaning** — `size` vs `capacity`, `index` vs `count`

### Constants

- [ ] **No magic numbers** — unexplained numeric literals in conditionals, arithmetic, or indexing (0, 1, -1 in common idioms are OK); the same literal repeated in multiple places must become a named constant
- [ ] **No unexplained bit operations** — bit shifts/masks must have named constants or comments
- [ ] **No inline hardware constants** — register offsets, queue sizes, tile dimensions must be named

### Intel GPU Terminology

- [ ] **SYCL programming terms used in code** — variable/function names, comments, logs, and test names use subgroup size (not SIMD width/length/simd_width), work-item (not SIMD lane), work-group (not thread block/threadblock); numeric SIMD-N becomes "subgroup size N"
- [ ] **Hardware terms only in optimization comments** — EU → XVE (Xe Vector Engine), Systolic/DPAS → XMX (Xe Matrix eXtension), Subslice/DSS → XC (Xe-core), HW thread → XVE thread; these appear only in comments explaining WHY an optimization was chosen (occupancy, register pressure, alignment), never in identifiers or general code
- [ ] **No deprecated hardware terms in identifiers** — e.g. `subslice_count` → `xc_count`/`xecore_count`, `getSimdWidth()` → `getSubgroupSize()`; also applies to GRF file/count (prefer "register file/count") and spell out SLM (Shared Local Memory) on first use
- [ ] **API boundary legacy names are wrapped** — when the underlying API (Level Zero, OpenCL, SYCL extensions) still exposes old terms in function/enum names, it's acceptable to use them at the call site only, wrapped in a modern-named helper with a comment noting the legacy API name

### Code Smells

- [ ] **Dead code** — unreachable branches, unused variables, unused `#include`
- [ ] **Error handling** — bare `catch(...)`, swallowed errors, missing return code checks
- [ ] **Resource management** — raw `new`/`delete` without RAII in C++, unclosed handles
- [ ] **Copy-paste patterns** — duplicated blocks that should be factored into a function
- [ ] **Overly long functions** — functions longer than ~80 lines should be split
- [ ] **Deep nesting** — more than 3 levels of `if`/`for` should use early returns or extraction

### XPU (SYCL) Kernels

- [ ] **Kernel style** — new kernels should prefer free functions over named functor structs, except for functors used with the `gpu_kernel`/`opmath_gpu_kernel_with_scalars` pattern.
- [ ] **Kernel launch** — reuse `sycl_kernel_submit`, which supports both the functor and free-function kernel forms, rather than calling `.submit(...)` directly on a `sycl::queue` instance (e.g. `queue.submit(...)`, `q.submit(...)`).
- [ ] **SYCL math functions** — for newly added or newly touched math calls, prefer `std::<fn>`; `sycl::<fn>` is also acceptable except for half-precision (`c10::Half`) inputs, which should use `std::<fn>`; `sycl::native::<fn>` only when it exists AND the matching CUDA kernel uses a fast intrinsic (e.g., `__expf`); use `sycl::ext::intel::math::<fn>` only when the function has no `std::`, `sycl::`, or `sycl::native::` equivalent. Does not apply to existing math calls left untouched by the PR

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

- [ ] **dtype choice in SYCL device code** — mirror the matching CUDA kernel
  - [ ] CUDA uses `at::acc_type<scalar_t, true>` (or `at::acc_type_device<scalar_t, kCUDA>`) for accumulation → XPU uses `at::acc_type_device<scalar_t, kXPU>`. Example sites: reductions, normalization, softmax.
  - [ ] CUDA uses `at::opmath_type<scalar_t>` for compute → XPU does the same and casts back to `scalar_t` on store. Example sites: element-wise / pointwise.
  - [ ] No `T_ACC` -> `T` -> `T_ACC` round-trip within one op. If a `T_ACC` value is consumed by another kernel or functor in the same op, route it as `T_ACC` end-to-end — use a separate buffer when the public output must stay `T`.
- [ ] **Atomic ops go through XPU wrappers** — use `atomicAdd`, `atomicCAS`, etc. from `src/ATen/native/xpu/sycl/Atomics.h`, not raw `sycl::atomic_ref::fetch_add` / `compare_exchange_strong`.
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
- [ ] **SYCL kernel vectorization** — If the kernel uses vectorized loads/stores, verify it follows the `aligned_vector` pattern:
  - [ ] Pointer arguments are qualified with `RESTRICT` to enable aliasing optimizations (works on both Linux and Windows)
  - [ ] Vector type declared as `using vec_t = memory::aligned_vector<scalar_t, vec_size>`
  - [ ] Loads use reinterpret cast + index: `const vec_t* RESTRICT input_vec = reinterpret_cast<const vec_t*>(input);` then `vec_t data = input_vec[i]`
  - [ ] Computation operates on the loaded `vec_t` local, not on raw pointer arithmetic
- [ ] **No shift_group_left + reduction combiner** — Any `sycl::shift_group_left` combined with a reduction combiner (add, min, max, mean, product, bitwise-or/and, etc.) must use `sycl::reduce_over_group` instead; the shift pattern generates excessive integer ALU instructions, stalling the ALU-INT pipeline. The true signal is the shift+combine pair, not the loop shape — it may run in either direction, live inside a vectorized inner loop, or have the shift+combine split into a functor/lambda with the loop elsewhere, so trace callers if needed. Common locations: reduction kernels, softmax, norm, and scan ops in `src/ATen/native/xpu/sycl/`
- [ ] **Pointwise kernel iteration and launch range**
  - [ ] Internal grid-stride loops use `XPU_KERNEL_LOOP(item, index, bound)` rather than hand-written `get_global_id`/`get_global_range` iteration, unless a documented custom mapping is required
  - [ ] Grid-stride pointwise launches bound `global_range` by resident capacity with `xpuKernelLoopGroupRange(static_cast<int64_t>(tensor.numel()), work_group_size)`, rather than calculating one work-item per element with `ceil_div(numel, work_group_size)`

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
- [ ] **Read-only top-level permissions** — Top-level `permissions` must be `read-all` or `contents: read`; all write permissions must be declared at the job level
- [ ] **Job-level permissions scoped minimally** — Each job's `permissions` block requests only the specific write permissions that job actually needs
- [ ] **Write permissions justified by GITHUB_TOKEN usage** — Every write permission declared must correspond to an actual API call using `GITHUB_TOKEN`; permissions needed only by PATs or other secrets should not be granted to the default token
- [ ] **No arbitrary code execution from PR inputs** — PR title, body, branch name, and commit messages must not be interpolated into shell commands without sanitization
- [ ] **Third-party action pinning** — Actions are pinned to a full commit SHA, not a mutable tag (e.g., `actions/checkout@<sha>` not `actions/checkout@v4`)
- [ ] **CI logic changes clearly explain impact** — Changes to workflow triggers, conditions, or job structure must document what validation coverage is gained or lost

## Backward Compatibility

- [ ] Any user-visible behavior change is called out explicitly
- [ ] Error type, default behavior, output semantics, and determinism changes were reviewed as BC-sensitive
- [ ] If the change intentionally alters public behavior, the rationale is stronger than "tests pass"
