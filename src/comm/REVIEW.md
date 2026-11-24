# `src/comm/` review guide

Files here (e.g. `SYCLHelpers.h`, `Memory.h`, `TensorInfo.h`, `Runtime.h`, etc.) are shared low-level infrastructure consumed across the whole XPU backend. In principle, all of these files could eventually be upstreamed into PyTorch, so the bar for adding new surface area is high.

## Adding new APIs

- **No new standalone helpers/APIs by default.** Do not add a brand-new helper function, class, struct, or type alias here just because it is convenient for one call site.
- **New macros to work around a compiler bug or warning are allowed.** Any other new macro needs the same justification as a new function below.
- **Prefer upstream first.** Before adding anything new, check whether PyTorch upstream (CUDA/CPU) already has an equivalent utility and mirror that pattern instead of inventing a new one.
- **Allowed without extra justification:**
  - Refactoring an existing API (same behavior, cleaner implementation)
  - Renaming an existing API
  - Replacing an existing API with a new implementation serving the same purpose
  - Adding an overload of an existing API (same name, new parameter set)
- **Not allowed without justification:** A purely additive new helper/API that is not a refactor, rename, replace, or overload of something that already exists.
- **Default verdict:** If a PR adds a new standalone API here that is not a refactor/rename/replace/overload of an existing one, always flag it as a **Request Changes** finding and wait for the author to justify why it must live in `src/comm/` rather than upstream or a call-site-local helper — do not approve based on an assumed justification.
