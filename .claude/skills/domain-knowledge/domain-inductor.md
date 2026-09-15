# Domain: Inductor

Shared knowledge base for `domain: inductor`.

## Path conventions

- pytorch: `torch/_inductor/`, `torch/_dynamo/`, `test/inductor/`.
  Inside a PyTorch checkout, XPU-specific overrides may live under
  `third_party/torch-xpu-ops/test/xpu/`.

## Reproduce: disable Inductor caches

Inductor caches compiled artifacts on disk
(`TORCHINDUCTOR_CACHE_DIR`, remote cache, etc.). A stale cache from
a previous run can mask a real regression or manufacture a false
pass/fail. Always set before reproducing or re-verifying an Inductor
UT:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 pytest <inductor test>
```

If a failure only reproduces without this flag, or only passes with
it, the cache itself is suspect — do not report a verdict until
confirmed with caches disabled.

## CUDA cross-reference

Compare against upstream CUDA/generic Inductor test behavior in
`agent_space_xpu/pytorch/test/inductor/` before concluding an
XPU-specific root cause.

## Common signatures that belong here

- **AOT Inductor `CppCompileError` in generated `.wrapper.cpp`.**
  Root cause is usually **codegen ordering** in `cpp_wrapper_cpu.py`
  — a function used before its definition is emitted. Check
  `write_wrapper_decl()` and `generate_input_output_runtime_checks()`
  ordering. After editing headers under
  `torch/csrc/inductor/cpp_wrapper/`, delete the stale PCH cache or
  the fix will be masked:

  ```bash
  rm -rf /tmp/torchinductor_$USER/precompiled_headers/
  ```
