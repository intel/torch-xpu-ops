---
name: xpu-yaml-registration
description: >
  Register XPU operators in YAML configuration files for the torch-xpu-ops
  repository. Use when adding operators to xpu_functions.yaml, editing
  native_functions.yaml dispatch entries, or wiring up new kernel
  implementations to the PyTorch dispatch system.
---

# XPU YAML Registration

This Skill guides you through correctly registering XPU operators in the YAML files that control PyTorch's dispatch system.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Overview: Two YAML files

| File | Purpose | When to edit |
|------|---------|-------------|
| `yaml/xpu_functions.yaml` | Lists all ops with native XPU implementations | Always — for every new native op |
| `yaml/native/native_functions.yaml` | Defines op signatures and dispatch entries | Only for Pattern B (direct registration) ops |

The relationship:
- `xpu_functions.yaml` tells the codegen "this op has a native XPU kernel"
- `native_functions.yaml` tells the codegen "route this op to this specific C++ function"

---

## Step 1: Add to `yaml/xpu_functions.yaml`

This file has a single `supported:` section listing all natively implemented ops.

### Format

```yaml
backend: XPU
cpp_namespace: at
use_out_as_primary: true
device_guard: true
supported:
  - op_name.overload
```

### Rules

1. **Add all overloads** for the operator. Most ops have three variants:
   ```yaml
   - lerp.Tensor          # functional: returns new tensor
   - lerp_.Tensor         # inplace: modifies self
   - lerp.Tensor_out      # out: writes to provided output tensor
   ```

2. **Maintain alphabetical order** within logical groups (the file is loosely grouped by operator category).

3. **Use the exact overload name** from upstream PyTorch's `aten/src/ATen/native/native_functions.yaml`. Common patterns:
   - `.Tensor` — tensor argument variant
   - `.Scalar` — scalar argument variant
   - `.out` — out= variant
   - No suffix — default overload
   - `_` prefix on the op name — inplace

4. **Include backward ops** if you implemented them:
   ```yaml
   - batch_norm_backward_elemt
   - batch_norm_backward_reduce
   ```

### Example: adding lerp

```yaml
supported:
  # ... existing entries ...
  - lerp.Tensor
  - lerp.Tensor_out
  - lerp_.Tensor
  - lerp.Scalar
  - lerp.Scalar_out
  - lerp_.Scalar
```

---

## Step 2: Determine if `native_functions.yaml` needs editing

### Pattern A: Dispatch-stub ops (NO edit needed)

If the host wrapper uses `REGISTER_XPU_DISPATCH()`:
```cpp
REGISTER_XPU_DISPATCH(lerp_kernel_tensor_weight, &xpu::lerp_tensor_kernel);
```

Then `native_functions.yaml` does **not** need editing. The dispatch stub mechanism handles routing automatically. You only need `xpu_functions.yaml`.

### Pattern B: Direct registration ops (edit needed)

If the host wrapper defines named functions like `op_xpu()` or `op_xpu_out()`:
```cpp
Tensor batch_norm_xpu(const Tensor& input, ...) { ... }
```

Then you must add XPU dispatch entries in `native_functions.yaml`.

---

## Step 3: Edit `yaml/native/native_functions.yaml` (Pattern B only)

### Format

Each entry defines a function signature with dispatch routing:

```yaml
- func: batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
  dispatch:
    XPU: batch_norm_stats_xpu
```

### Rules

1. **Match the upstream signature exactly** — copy from `pytorch/pytorch`'s `aten/src/ATen/native/native_functions.yaml`

2. **Add only the `dispatch:` → `XPU:` entry** — do not modify the function signature, tags, variants, or other dispatch keys

3. **The C++ function name** in the dispatch entry must match exactly what you defined in the host wrapper `.cpp`

4. **For structured ops** with `.out` as the primary:
   ```yaml
   - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
     structured: True
     structured_inherits: TensorIteratorBase
     dispatch:
       XPU: add_out_xpu
   ```
   Only the `.out` variant gets a dispatch entry; `.Tensor` and `_.Tensor` use `structured_delegate`.

5. **For helper functions** (internal ops):
   ```yaml
   - func: _cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
     dispatch:
       XPU: cummax_helper_xpu
   ```

6. **For sparse/nested variants**, use the specific dispatch keys:
   ```yaml
   dispatch:
     SparseXPU: add_sparse
     SparseCsrXPU: add_sparse_csr
     NestedTensorXPU: NestedTensor_add_Tensor
   ```

### Common dispatch key variants

| Key | When to use |
|-----|-------------|
| `XPU` | Standard dense tensor ops |
| `SparseXPU` | Sparse tensor ops |
| `SparseCsrXPU` | Sparse CSR tensor ops |
| `NestedTensorXPU` | Nested tensor ops |
| `CompositeExplicitAutograd` | Device-agnostic compound ops (not XPU-specific) |

---

## Step 4: Remove from fallback (if applicable)

If the op was previously falling back to CPU, remove it from the fallback list in `src/ATen/native/xpu/XPUFallback.template`:

```cpp
// Remove this line from the fallback_list vector:
"op_name",
"op_name.out",
```

---

## Step 5: Verify consistency

After editing YAML files, verify cross-file consistency:

### Check 1: All overloads present
```bash
# Find all overloads for an op in upstream
grep "your_op" aten/src/ATen/native/native_functions.yaml

# Verify they're all in xpu_functions.yaml
grep "your_op" yaml/xpu_functions.yaml
```

### Check 2: Implementation exists
For every entry in `xpu_functions.yaml`, verify:
- Dispatch-stub ops: `REGISTER_XPU_DISPATCH(...)` exists in a `src/ATen/native/xpu/*.cpp`
- Direct registration ops: the named function exists and matches the `native_functions.yaml` dispatch entry

### Check 3: No duplicate registration
An op should not appear in both:
- `xpu_functions.yaml` supported list AND
- `XPUFallback.template` fallback list

### Check 4: Codegen builds
The YAML files are consumed by `tools/codegen/` during build. After edits, a full build will validate the registration.

---

## Common mistakes

1. **Missing overloads** — adding `lerp.Tensor` but forgetting `lerp_.Tensor` and `lerp.Tensor_out`
2. **Wrong overload name** — using `lerp.tensor` (lowercase) instead of `lerp.Tensor`
3. **Editing native_functions.yaml for dispatch-stub ops** — unnecessary; the stub handles routing
4. **Not editing native_functions.yaml for direct registration ops** — the op won't dispatch
5. **Mismatched function names** — `native_functions.yaml` says `batch_norm_stats_xpu` but C++ defines `batch_norm_stats`
6. **Leaving op in fallback list** — op appears natively registered but also in `XPUFallback.template`
7. **Adding entries in the wrong section** — `xpu_functions.yaml` only has `supported:`, not `autograd:` (unlike some other backend YAML files)

---

## Checklist

- [ ] All operator overloads added to `yaml/xpu_functions.yaml`
- [ ] `yaml/native/native_functions.yaml` updated (Pattern B only)
- [ ] Function names in YAML match C++ implementation exactly
- [ ] Op removed from `XPUFallback.template` (if previously there)
- [ ] No duplicate registration (YAML + fallback)
- [ ] `src/`, `yaml/`, and `test/` changes are consistent
