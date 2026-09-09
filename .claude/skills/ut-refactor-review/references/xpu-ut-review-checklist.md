# XPU UT Refactor Review Checklist

Evaluate every changed line against these patterns. Each row gives a code
pattern to look for, what it means (the defect and the correct form), and a
severity. Severities follow this scale:

- **Blocker** — XPU gets wrong or zero test execution (the test never
  instantiates, runs on the wrong device, silently runs the wrong dtype/skip
  gating, risks an unguarded OOM/crash), or the change breaks another backend.
  Must fix before merge.
- **Major** — coverage or scope is silently wrong (a dropped/weakened assertion
  or case, a mismatched or missing skip/xfail, a behavior-changing "refactor").
  Fix before merge.
- **Minor** — tuning- or convention-level issue (tolerance, over-broad skip
  scope, naming/classification mismatch, stale skip). Fix or justify.
- **Info** — no direct action required; guidance, or a pattern with no XPU
  equivalent. Note and move on.

Rows are grouped by area and roughly ordered by how often they surfaced as real
review comments across the studied PRs.

## 1. Device generalization (replacing CUDA hardcoding)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `x.cuda()` / `.cuda(rank)` inside a generalized test | Runs on CUDA (or errors on an XPU-only box) regardless of the parameterized device. Use `.to(device)` / `.to(device_type)` / `.to(rank)`. | Blocker |
| Literal `"cuda"` in `device=`, `torch.device("cuda")`, `init_device_mesh("cuda", ...)` | A hardcoded device where a `device`/`device_type` parameter is available defeats generalization. | Blocker |
| `@unittest.skipIf(not TEST_CUDA, ...)` on a test the PR enables on XPU | A CUDA-only gate keeps the test from running on XPU. Use `@onlyAccelerator` / `@onlyNativeDeviceTypesAnd([...])` when genuinely accelerator-generic. | Blocker |
| `torch.cuda.set_device` / `current_device` / `synchronize` / `current_stream` / `Event` / `Stream()` + `stream(s)` / `manual_seed`; `GradScaler(device="cuda")` | Prefer the full generic API: `torch.accelerator.set_device_index` / `current_device_index` / `synchronize` / `current_stream`, `torch.Event`, `torch.Stream()` + `with s:`, `device_module.manual_seed`, `GradScaler(device=device_type)`. Reviewers consistently push this, not just `is_available`/`device_count`/`current_accelerator`. | Minor |
| `if TEST_CUDA: mem_stats["active_bytes.all.peak"]` widened to `TEST_CUDA or TEST_XPU` | Assumes XPU exposes the same stats key as CUDA. Confirm the key exists for XPU rather than assuming parity, or it `KeyError`s. | Blocker |
