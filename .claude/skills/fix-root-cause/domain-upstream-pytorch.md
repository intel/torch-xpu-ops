# Domain: Upstream PyTorch Framework

Reference material loaded by `fix-root-cause` when it emits
`domain: upstream-pytorch`. Not a standalone skill. See `SKILL.md`
and `domain-registry.md` in the same directory for the routing
contract.

- `target_repo` must be `"pytorch"`. Never `third_party/torch-xpu-ops/`.

The signature list below is a curated set of known-misleading
failure patterns — cases where the error surface looks like a
backend bug but the fix is in framework code. **It is not
exhaustive.** If the failure does not match any signature here,
fall back to the standard investigation in `SKILL.md` Step 3; do
not force-fit an unmatched failure to the closest signature.

## Common signatures that belong here

Recurring failure patterns whose root cause is in device-agnostic
pytorch code, not in any backend kernel. Match on signature; if the
failure fits, route to this domain and stop searching the backend
kernel.

- **`RuntimeError: Expected X.is_contiguous(memory_format)` under
  vmap / functorch.** Origin: `aten/src/ATen/native/*.cpp`
  (`group_norm.cpp`, `layer_norm.cpp`, similar norms) has
  `TORCH_CHECK(X.is_contiguous(memory_format))` where
  `memory_format` is `at::MemoryFormat::Contiguous` on non-CPU
  devices and `X.suggest_memory_format()` on CPU. vmap produces
  batched inputs that are not contiguous in `Contiguous` layout, so
  any non-CPU device trips the check. Fix location: relax the check
  in pytorch (allow vmap-batched inputs), add a batching rule, or
  add a decomposition. **Not fixable in the backend kernel** — the
  kernel never sees the tensor because the check fires in the
  framework wrapper.

## Adding a new signature

Add a bullet under "Common signatures that belong here" only after
`fix-root-cause` has actually landed a fix (or a NEEDS_HUMAN
verdict tied to a filed pytorch issue) that turned on this
domain's misleading-surface trap. Include:

- The exact error string or minimal signature (so future agents
  can grep-match).
- Where the check / logic actually lives in pytorch.
- Why the backend is a dead end (one sentence).
- Fix location or, if NEEDS_HUMAN, the upstream issue link.

Do not add speculative signatures ("I imagine there could be a
case where..."). This file is a debugging shortcut, not a
brainstorm — an entry the agent cannot trust is worse than no
entry, because it burns Step 3 time chasing a wrong lead.
