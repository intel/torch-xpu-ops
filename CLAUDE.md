# Coding Agent Guide for torch-xpu-ops

## Project Overview

torch-xpu-ops provides XPU (Intel GPU) operator implementations for PyTorch ATen.
It is built as a submodule of PyTorch (`third_party/torch-xpu-ops`), not standalone.

Before reviewing, implementing, or debugging any operator, you **MUST**
read the corresponding upstream PyTorch code to understand the expected
semantics and some patterns:

- `src/ATen/native/xpu/<Op>.cpp` → upstream: `aten/src/ATen/native/<Op>.cpp`
- `src/ATen/native/xpu/sycl/<Op>Kernels.cpp` → upstream CUDA equivalent: `aten/src/ATen/native/cuda/<Op>.cu`
- Dispatch registration → upstream: `aten/src/ATen/native/native_functions.yaml`

Use `gh` CLI, WebFetch, or sub-agents to fetch files from the `pytorch/pytorch`
repository. Do not infer upstream behavior from memory — verify from source.
This applies to code review, new implementations, bug fixes, and understanding
existing logic.

## Working Principles

### Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Repository Structure

```
src/ATen/native/xpu/          # Dispatch registration + thin wrappers (no SYCL)
src/ATen/native/xpu/sycl/     # SYCL kernel implementations (.cpp/.h)
src/ATen/native/nested/xpu/   # Nested tensor XPU implementations
src/comm/                      # Shared utility headers
src/xccl/                      # XCCL communication backend
yaml/native/native_functions.yaml  # Op dispatch registration (PyTorch schema)
test/xpu/                      # XPU device tests (pytest-based)
test/regressions/              # Regression tests
test/sycl/                     # SYCL C++ unit tests (CMake)
tools/linter/                  # Lint adapter scripts
```

## Scratch Space

Use `agent_space_xpu/` (git-ignored, at repo root) for temporary scripts, scratch files, and throwaway experiments. Do not commit files from this directory.

## PR Review

When asked to review a PR, always use the /pr-review skill.

## Environment

If any tool you're trying to use (pip, python, spin, etc) is missing, check for
a `.venv` directory in the project root or its parent directory. If found,
activate it and retry. If no `.venv` is found, stop and ask the user if an
environment is needed. Do NOT try to find alternatives or install these tools.

## CI Docker Images

The `.ci/docker/` directory is content-hashed to determine whether Docker images
need rebuilding. Any file change inside `.ci/docker/` (including the README)
changes the hash and triggers a full Docker image rebuild. Do not make changes
in this directory unless you intend to rebuild Docker images. When Docker builds
are broken (e.g., due to an upstream Ubuntu outage), avoid touching this
directory so you don't force a rebuild against the broken state.

## Commit Messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

The commit message should be clear, informative, and have a Test Plan section
that describes how you tested the change. If you are fixing a bug, the commit
message must explain the root cause of the bug and how the fix works.
If there were multiple potential paths you could have taken, please call them
out succinctly and justify the one you took.

When describing the testing strategy in a commit message, include the literal
commands that were run in fenced Markdown code blocks.

Disclose that the PR was authored with an AI assistant.

When the user asks you to amend a commit, check whether the commit message
still accurately describes the changes. If it doesn't and the commit is not a
ghstack commit, update the message. For ghstack commits, amending the message
is a no-op, so just remind the user to update the PR description if needed.

If a commit message contains `ghstack-source-id` or `Pull-Request` trailers,
you MUST preserve them when rewriting or splitting commit messages. ghstack
will update the source id automatically when needed.

## ghstack Workflow

ghstack commits follow a different workflow than the conventional GitHub branch
and PR workflow. First identify whether you're on a ghstack commit:

- If HEAD is a detached commit, you are almost certainly in a ghstack flow.
- If the commit message contains a `ghstack-source-id` trailer, it is an
  existing ghstack commit.
- If the commit is associated with a remote branch like `origin/gh/USERNAME/N`,
  it is likely a ghstack commit (imperfect signal: local amends without a push
  can desync this).

Rules for working with ghstack:

- **Don't amend unless asked.** If the user asks you to work on a ghstack
  commit, leave changes uncommitted so the user can review with `git diff`.
  Only amend into the commit if the user explicitly asks you to amend or to
  submit it directly.
- **Submitting.** Run `ghstack` to submit. When only working on a single
  commit, use `ghstack --no-stack` to avoid updating the rest of the stack and
  burning unnecessary CI. Use a full `ghstack` when you're intentionally
  updating CI for the whole stack.
- **Preserve metadata trailers.** When editing a commit message, never delete
  `Pull-Request:` or `ghstack-source-id:` trailers. Always re-read them from
  HEAD each time you compose an amend — never reuse a saved/cached message
  body, since `ghstack` rewrites `ghstack-source-id` on every push and a
  stale trailer will clobber HEAD's current one. If you modified the commit
  message, run `ghstack -u` afterwards to push the updated PR description.
- **Never push directly.** Do not `git push` to branches, and never directly
  modify the `gh/USERNAME/N` branches — ghstack manages those.
- **Finding the PR.** If the user asks to pull CI results or code review for a
  ghstack commit, get the PR URL from the `Pull-Request` trailer in the commit
  message. Use `gh` CLI to fetch status/comments from there.
- **Editing earlier commits / splitting.** Treat it like a normal stack of
  commits (use `git rebase`, etc.). Commits that keep their metadata trailers
  stay associated with their existing PRs; commits without trailers will get a
  fresh PR on submit. A full `ghstack` run is usually appropriate here.

## Dynamo Config

Use `torch._dynamo.config.patch` for temporarily changing config. It can be used as a decorator on test methods or as a context manager:

```python
# Good - use patch as decorator on test method
@torch._dynamo.config.patch(force_compile_during_fx_trace=True)
def test_my_feature(self):
    # test code here
    pass

# Good - use patch as context manager
with torch._dynamo.config.patch(force_compile_during_fx_trace=True):
    # test code here
    pass

# Bad - manual save/restore
orig = torch._dynamo.config.force_compile_during_fx_trace
try:
    torch._dynamo.config.force_compile_during_fx_trace = True
    # test code here
finally:
    torch._dynamo.config.force_compile_during_fx_trace = orig
```

## Fixing B950 Line Too Long in Multi-line String Blocks

If B950 line too long triggers on a multi-line string block, you cannot fix it by
putting # noqa: B950 on that line directly, as that would change the meaning of the
string, nor can you fix it by line breaking the string (since you need the string
to stay the same). Instead, put # noqa: B950 on the same line as the terminating
triple quote.

Example:

```python
    self.assertExpectedInline(
        foo(),
        """
this line is too long...
""",  # noqa: B950
    )
```

## Logging and Structured Tracing

When adding debug logging for errors or diagnostic info, consider two user personas:

1. **Local development**: Users run locally and can access files on disk
2. **Production jobs**: Users can only access logs via `tlparse` from structured traces

For production debugging, use `trace_structured` to log artifacts:

```python
from torch._logging import trace_structured

# Log an artifact (graph, edge list, etc.)
trace_structured(
    "artifact",
    metadata_fn=lambda: {
        "name": "my_debug_artifact",
        "encoding": "string",
    },
    payload_fn=lambda: my_content_string,
)
```

To check if structured tracing is enabled (for conditional messaging):

```python
from torch._logging._internal import trace_log

if trace_log.handlers:
    # Structured tracing is enabled, suggest tlparse in error messages
    msg += "[Use tlparse to extract debug artifacts]"
```

**Best practices for error diagnostics:**

- Always log to `trace_structured` for production (no runtime cost if disabled)
- If you're dumping debug info in the event of a true internal compiler exception,
  you can also consider writing to local files for local debugging convenience
- In error messages, tell users about both options:
  - Local files: "FX graph dump: min_cut_failed_graph.txt"
  - Production: "Use tlparse to extract artifacts" (only if tracing enabled)
- Use `_get_unique_path()` pattern to avoid overwriting existing debug files

## Build

This project builds only as part of PyTorch. No standalone build exists.

Always check local memory for build configuration (env vars, incremental-build shortcuts, etc.) before running the build, and apply what you find. If nothing applicable is in memory, ask the user.

All build (both codegen, C++ and python) is done via `pip install -e . -v --no-build-isolation`.
You should NEVER run any other command to build PyTorch.

```bash
# Full build (from PyTorch root, with torch-xpu-ops as third_party/torch-xpu-ops)
cd <pytorch_root>
pip install -e . -v --no-build-isolation
# Or: WERROR=1 python setup.py bdist_wheel

# Debug/RelWithDebInfo builds auto-enable BUILD_SEPARATE_OPS for faster iteration.
# Set BUILD_SEPARATE_OPS=1 manually to shrink translation unit scope during dev.
```

### Commit Pin & Development Override

PyTorch pins torch-xpu-ops to a specific commit via `third_party/xpu.txt`.
During build, CMake reads this file, clones `intel/torch-xpu-ops` into
`third_party/torch-xpu-ops/`, fetches, and checks out the pinned commit
(see `caffe2/CMakeLists.txt`).

When developing a torch-xpu-ops PR, you need your PR branch — not the pinned
commit. Manually clone into the expected path before building so CMake skips
its own clone, then update the pin to match your HEAD so the checkout is a
no-op:

```bash
# 1. Clone your fork into the path CMake expects
cd <pytorch_root>/third_party
git clone <your-fork-url> torch-xpu-ops
cd torch-xpu-ops
git checkout <your-pr-branch>

# 2. Update the pin to your HEAD so CMake's checkout becomes a no-op
git rev-parse HEAD > <pytorch_root>/third_party/xpu.txt
```

Do not commit the `xpu.txt` change. This is a local-only override for
development builds.

## Lint Commands

Linting uses `lintrunner` (configured in `.lintrunner.toml`):

```bash
# Initialize linters (first time)
lintrunner init

# Lint all changed files vs main
lintrunner -m origin/main

# Lint all files (skip clang-based linters for speed)
lintrunner --skip CLANGTIDY,CLANGFORMAT,MERGE_CONFLICTLESS_CSV --all-files

# Auto-apply suggested fixes
lintrunner -a -m origin/main
```

Active linters: FLAKE8, CLANGFORMAT, CLANGTIDY, MYPY, RUFF, PYFMT (usort+ruff-format),
SHELLCHECK, CMAKE, NEWLINE, SPACES, TABS, plus custom grep-based checks.

You can also use commands provided via `spin` for linting.
Use `spin help` to list available commands.
Generally, use `spin lint` to run the lint and `spin fixlint` to apply automatic fixes.

When the user asks you to commit or amend, run `lintrunner -a` before creating
the commit. Fix any lint errors it reports, then commit.

## Test Commands

Tests run via **pytest** (not `python -m pytest`). Most tests live in `test/xpu/`.

Use our test class and test runner:

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

To test Tensor equality, use assertEqual.
For tests over multiple inputs, use the `@parametrize` decorator.
For any test that checks numerics of the on-device implementation, use `instantiate_device_type_tests` to write device-generic tests.

```bash
# Run a single test file
pytest test/xpu/test_ops_xpu.py -v --timeout 600

# Run a single test case
pytest test/xpu/test_ops_xpu.py -v -k "test_some_specific_case"

# Run with skip lists (the standard CI approach)
cd test/xpu
python run_test_with_skip.py

# Reproduce a CI failure
pytest -sv test/xpu/test_ops_xpu.py::TestClassName::test_method_name

# Regression tests
pytest test/regressions/

# Run upstream PyTorch tests with XPU
python test/run_test.py --include test_torch test_type_promotion
```

CI uses these pytest options: `-v --timeout 600 --timeout_method=thread --dist worksteal`

## Code Style Guidelines

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Comments should be useful, for example, comments that remind the reader about
  some global context that is non-obvious and can't be inferred locally.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless
  it significantly improves code readability.
- Prefer clear abstractions. State management should be explicit.
  For example, if managing state in a Python class: there should be a clear
  class definition that has all of the members: don't dynamically `setattr`
  a field on an object and then dynamically `getattr` the field on the object.
- Match existing code style and architectural patterns.
- Assume the reader has familiarity with PyTorch. They may not be the expert
  on the code that is being read, but they should have some experience in the
  area.
- ASCII only in newly added code comments. Do not introduce Unicode characters
  (e.g., smart quotes, em dashes, arrows, non-ASCII letters) in new comments.
  Leave preexisting Unicode in untouched comments alone; only enforce this for
  comments you are adding or rewriting.
- If uncertain, choose the simpler, more concise implementation.
- If you write 200 lines and it could be 50, rewrite it.

### C++ Code Style

#### File Organization
- **Dispatch layer**: `src/ATen/native/xpu/<Op>.cpp` — registration only, no SYCL code
- **Kernel layer**: `src/ATen/native/xpu/sycl/<Op>Kernels.cpp` — SYCL implementations
- **Kernel headers**: `src/ATen/native/xpu/sycl/<Op>Kernels.h` — declarations

#### Formatting
- Enforced by clang-format (`.clang-format`): 2-space indent, 80-column limit
- Braces: Attach style (K&R)
- Pointer alignment: Left (`int* ptr`)
- No tabs (spaces only)

#### Includes
- **Always use angle brackets** (`#include <...>`), never quoted includes
- Ordering: ATen core → ATen ops → project SYCL headers → comm utilities
- Use `#pragma once` for header guards

#### Naming
- Functions: `snake_case` — `gelu_kernel`, `copy_kernel`, `add_kernel`
- Structs/Functors: `PascalCase` — `AddFunctor`, `EluOutFunctor`
- Private members: `snake_case_` (trailing underscore) — `alpha_`, `negcoef_`
- Macros: `UPPER_SNAKE_CASE` — `REGISTER_XPU_DISPATCH`
- Template type params: `scalar_t`, `opmath_t` (lowercase by convention)

#### Namespaces
- Dispatch files: `namespace at { namespace native { ... } }`
- Kernel files: `namespace at::native::xpu { ... }`
- Always add closing comment: `} // namespace at::native::xpu`

#### Error Handling
- Use `TORCH_CHECK(condition, "message")` for user-facing input validation
- Use `TORCH_INTERNAL_ASSERT(condition)` for internal invariants
- Do NOT use `AT_ERROR` (legacy)

#### SYCL Kernel Pattern
```cpp
// 1. Define a functor (not a lambda) for SYCL compatibility
template <typename opmath_t>
struct MyFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a + alpha_ * b;
  }
  MyFunctor(opmath_t alpha) : alpha_(alpha) {}
 private:
  opmath_t alpha_;
};

// 2. Dispatch by dtype and submit via gpu_kernel
void my_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "my_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(
            iter, MyFunctor<opmath_t>(alpha.to<opmath_t>()));
      });
}
```

#### Header API Pattern
```cpp
// In <Op>Kernels.h:
namespace at::native::xpu {
TORCH_XPU_API void my_kernel(TensorIteratorBase& iter, const Scalar& alpha);
} // namespace at::native::xpu
```

#### Dispatch Registration
```cpp
// In <Op>.cpp — register kernel stub to XPU dispatch
REGISTER_XPU_DISPATCH(my_stub, &xpu::my_kernel);
```

#### Deprecated Macros (lint-enforced)
- Use `[[maybe_unused]]` instead of `C10_UNUSED`
- Use `[[nodiscard]]` instead of `C10_NODISCARD`
- Use `c10::call_once` / `c10::once_flag` instead of `std::call_once` / `std::once_flag`

### Python Code Style

#### Formatting
- Max line length: 120 (configured in ruff/flake8)
- Formatter: ruff-format + usort (for test files)
- 4-space indentation

#### Imports
Follow PEP 8 ordering (standard lib → third-party → internal):
```python
import math
import unittest

import torch
import numpy as np

from torch.testing._internal.common_utils import run_tests, TestCase
```

#### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Module-level lists: `_leading_underscore_snake_case`
- Logger: `log = logging.getLogger(__name__)` (never use root `logging.info()`)

#### Test File Pattern
Most XPU tests wrap upstream PyTorch tests:
```python
# Owner(s): ["module: intel"]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests
try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_ops import TestCommon

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
```

#### Type Annotations
- `type: ignore` must be qualified: `# type: ignore[attr-defined]`
- `noqa` must be qualified: `# noqa: F401`
- mypy is configured via `mypy.ini` and `mypy-strict.ini`

## YAML Op Registration

Ops are registered in `yaml/native/native_functions.yaml` using PyTorch's schema:
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  structured_delegate: add.out
  dispatch:
    XPU: add_xpu
```

XPU dispatch keys: `XPU`, `SparseXPU`, `SparseCsrXPU`, `NestedTensorXPU`.

## License Header

All files must include the Intel copyright + Apache 2.0 header:
```
Copyright 2020-2025 Intel Corporation
Licensed under the Apache License, Version 2.0
```
