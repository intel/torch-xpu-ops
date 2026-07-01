---
name: extract-xpu-kernel-asm
description: >
  Extract Intel GPU ISA (assembly) from any XPU kernel. Classifies the codegen
  path (SYCL AOT, SYCL JIT, Triton, or oneDNN ngen) and delegates to the
  matching extraction skill. Use when asked to extract ASM, disassemble XPU
  kernels, get GPU ISA for an aten op, dump shader for PyTorch XPU, or
  disassemble a standalone DPC++/Triton binary.
---

# Extract XPU Kernel ASM (Dispatcher)

Classify which codegen path produced the kernel and delegate to the matching
atomic skill to extract its Intel GPU ISA.

## Compilation Stack

All XPU kernel compilation ultimately produces the same thing: **GPU ISA bytes**.
The skills are separated not by compilation logic (which is largely identical),
but by **where the zebin lives**:

```
┌─────────────────────────────────────────────────────────────────┐
│              Shared Compilation Stack                            │
│  SYCL/C++ → LLVM IR → SPIR-V → IGC → zebin (GPU ISA)          │
│                                                                 │
│  • -g / -gline-tables-only acts at frontend (SYCL → LLVM IR)   │
│  • Debug info flows: !dbg → OpLine → DebugLoc → .debug_line    │
│  • JIT vs AOT: same IR pipeline, different WHEN it runs         │
└─────────────────────────────────────────────────────────────────┘

┌───────────────┬──────────────────────────────────────────────────┐
│ Scenario      │ Where is the zebin? How to get it?               │
├───────────────┼──────────────────────────────────────────────────┤
│ sycl-aot      │ Embedded in host binary at build time            │
│               │ → DumpZEBin=1 NEOReadDebugKeys=1 → ocloc disasm  │
├───────────────┼──────────────────────────────────────────────────┤
│ sycl-jit      │ Generated at first launch, only in memory        │
│               │ → IGC_ShaderDumpEnable=1 → dump dir → .asm/.elf  │
├───────────────┼──────────────────────────────────────────────────┤
│ triton        │ Triton compiler → SPIR-V → IGC JIT at launch     │
│               │ → IGC_ShaderDumpEnable=1 (same as sycl-jit)      │
├───────────────┼──────────────────────────────────────────────────┤
│ onednn (ngen) │ BYPASSES the entire SPIR-V/IGC stack             │
│               │ Own JIT: ngen → raw ISA bytes (not zebin ELF)    │
│               │ → ONEDNN_JIT_DUMP=1 → IGA ctypes disassembly    │
└───────────────┴──────────────────────────────────────────────────┘
```

**Key implications for downstream:**
- `sycl-aot`, `sycl-jit`, `triton` all produce zebin ELF → same
  `ocloc disasm` / `readelf` / `.debug_line` workflow applies.
- `onednn` produces raw ISA bytes (no ELF wrapper, no `.debug_line`)
  → requires IGA ctypes, pattern-recognition only for source mapping.

## When to use

- You have a hot kernel (from profiler / unitrace) and need its ISA.
- Entry points: PyTorch op, standalone `@triton.jit`, standalone DPC++.
- Hardware: Intel XPU (PVC / BMG-G31 / DG2 / LNL / ARL / MTL / PTL).

## When NOT to use

- You only want wall-clock timing — use the profiler directly.
- You only want Triton IR (not ISA) — read `~/.triton/cache/`.

## Steps

### Step 0: Locate tools

Key tools are NOT always on PATH. Probe before proceeding:

```bash
# Detect oneAPI root: check env vars first, then common install locations
ONEAPI=${ONEAPI_ROOT:-${CMPLR_ROOT:+${CMPLR_ROOT%/*}}}
if [ -z "$ONEAPI" ]; then
  for d in /opt/intel/oneapi ~/intel/oneapi /usr/local/oneapi; do
    [ -d "$d" ] && ONEAPI="$d" && break
  done
fi

# ocloc (for disassembling zebin ELFs)
command -v ocloc >/dev/null || echo "ocloc not found; source oneapi-vars.sh"

# libiga64.so (for oneDNN ngen disassembly only)
IGA_LIB=$(test -n "$ONEAPI" && find "$ONEAPI" -name 'libiga64.so' 2>/dev/null | head -1)
```

### Step 1: Pin the actually-launched kernel

You MUST identify which kernel was actually executed on the GPU. AOT bundles
embed all specializations; IGC dumps all referenced kernels. Without pinning,
you risk extracting the wrong one.

**Preferred: unitrace** (covers ALL code paths including native-handle):

```bash
unitrace -d <repro_cmd>
```

Parse the `== L0 Backend ==` table — each data row has the kernel name as the
first double-quoted field. Skip rows starting with `ze*` (those are API calls,
not GPU kernels). Extract and deduplicate kernel names.

Probe unitrace location in order: `$UNITRACE` env var → `command -v unitrace`
→ `$UNITRACE_HOME/unitrace` → `<pti-gpu-build>/tools/unitrace/build/unitrace`.

**Fallback: SYCL_UR_TRACE** (zero-dep, but has a blind spot):

```bash
SYCL_UR_TRACE=-1 <repro_cmd> 2>&1 | grep -oP 'pKernelName = 0x[0-9a-f]+ \(\K[^)]+'
```

**WARNING**: `SYCL_UR_TRACE` is BLIND to kernels created via
`urKernelCreateWithNativeHandle` (oneDNN ngen, SYCL-TLA, Triton-xpu ≥ 3.7.0).
If the list is empty but the workload clearly ran GPU kernels, you MUST install
unitrace before proceeding. Do NOT extract ASM blindly.

### Step 2: Classify scenario

| Signal | Scenario |
|---|---|
| Kernel = `gemm_kernel` / `gen_conv_kernel` / routed via `mkldnn::*` | `onednn` |
| Kernel = `triton_*` or standalone `@triton.jit` | `triton` |
| Kernel = `_ZTS…` AND AOT path active for current device | `sycl-aot` |
| Kernel = `_ZTS…` AND JIT path active (no AOT or target mismatch) | `sycl-jit` |

**For `_ZTS…` kernels, determine AOT vs JIT:**

A binary may contain `__CLANG_OFFLOAD_BUNDLE` but its AOT targets may not cover
the current device (e.g. built for PVC, running on BMG → runtime falls back to
JIT via SPIR-V). Test which path is actually active:

```bash
# Quick test: attempt DumpZEBin — if .elf files appear, AOT path is active
mkdir -p /tmp/aot_test && cd /tmp/aot_test
DumpZEBin=1 NEOReadDebugKeys=1 <repro_cmd> 2>/dev/null
if ls *.elf 2>/dev/null | grep -q .; then
  SCENARIO=sycl-aot
else
  SCENARIO=sycl-jit
fi
rm -rf /tmp/aot_test
```

If ambiguous → ask the user.

### Step 3: Dispatch to atomic skill

```
onednn   → extract-asm-onednn
triton   → extract-asm-triton
sycl-aot → extract-asm-syclkernel-aot
sycl-jit → extract-asm-syclkernel-jit
```

### Step 4: Emit normalized outputs

All fields REQUIRED:

| Field | Description |
|---|---|
| `input-kernel` | User's kernel identifier (echoed verbatim) |
| `scenario` | `onednn` / `triton` / `sycl-aot` / `sycl-jit` |
| `asm-dir` | Absolute path to output directory |
| `asm-file` | Absolute path to the chosen `.asm` file |
| `kernel-name` | Kernel name as it appears in `asm-file` |
| `launch-evidence` | ≥3 sentences explaining WHY this is the correct kernel |

## Examples

Each example shows the **dispatcher's classification** — how to identify the
scenario and what output to expect. The detailed extraction steps are in the
respective sub-skill; these examples only demonstrate the classification signal
and final result.

### Example 1 — oneDNN ngen: BF16 GEMM via `aten::matmul`

**Repro:**
```python
import torch
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='xpu')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='xpu')
c = a @ b; torch.xpu.synchronize()
```

**Classification signal:** unitrace shows `gemm_kernel` → scenario = `onednn`
→ delegate to `extract-asm-onednn`.

**Expected result:**
```
scenario:    onednn
asm-file:    <workdir>/gemm.asm
kernel-name: gemm_kernel
validation:  grep -c dpas gemm.asm → non-zero (GEMM uses dpas instructions)
```

### Example 2 — Triton: `torch.compile(softmax)`

**Repro:**
```python
import torch
@torch.compile
def fn(x): return torch.softmax(x, dim=-1)
x = torch.randn(1024, 1024, device='xpu')
fn(x); torch.xpu.synchronize()
```

**Classification signal:** `TORCH_LOGS=output_code` shows a `triton_per_fused_*softmax*`
kernel name → scenario = `triton` → delegate to `extract-asm-triton`.

**Expected result:**
```
scenario:    triton
asm-file:    <igc_dump>/OCL_asm*_simd*_entry_*.asm
kernel-name: triton_per_fused_*softmax* (exact name varies by PyTorch version)
validation:  grep 'libdevice.exp' <asm-file> confirms softmax exp computation
```

### Example 3 — SYCL AOT: standalone DPC++ binary

**Repro:**
```cpp
// vec_add.cpp
#include <sycl/sycl.hpp>
class VecAddKernel;
int main() {
    sycl::queue q;
    constexpr int N = 1 << 24;
    float *a = sycl::malloc_device<float>(N, q);
    float *c = sycl::malloc_device<float>(N, q);
    q.parallel_for<VecAddKernel>(N, [=](int i) { c[i] = a[i] + 1.0f; }).wait();
    sycl::free(a, q); sycl::free(c, q);
}
```
```bash
icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xs "-device <dev>" vec_add.cpp -o vec_add
```

**Classification signal:** `DumpZEBin=1 NEOReadDebugKeys=1 ./vec_add` produces
`.elf` files → scenario = `sycl-aot` → delegate to `extract-asm-syclkernel-aot`.

**Expected result:**
```
scenario:    sycl-aot
asm-file:    <workdir>/<name>_dump/.text._ZTS12VecAddKernel.asm
kernel-name: _ZTS12VecAddKernel
validation:  c++filt _ZTS12VecAddKernel → VecAddKernel
```

### Example 4 — SYCL AOT: `libtorch_xpu.so`

**Repro:**
```python
import torch
x = torch.randn(1024, dtype=torch.float, device='xpu')
y = x + 1.0; torch.xpu.synchronize()
```

**Classification signal:** `DumpZEBin=1 NEOReadDebugKeys=1 python test.py`
produces `.elf` files → scenario = `sycl-aot`
→ delegate to `extract-asm-syclkernel-aot`.

**Expected result:**
```
scenario:    sycl-aot
asm-file:    <workdir>/<name>_dump/.text._ZTSN2at6native3xpu...E.asm
kernel-name: matches the demangled kernel from c++filt
```

### Example 5 — SYCL JIT: standalone DPC++ without AOT target

**Repro:**
```cpp
// shift_reduce.cpp
#include <sycl/sycl.hpp>
class ShiftReduceKernel;
int main() {
    sycl::queue q;
    auto *buf = sycl::malloc_device<int>(1024, q);
    q.parallel_for<ShiftReduceKernel>(
        sycl::nd_range<1>(1024, 32), [=](sycl::nd_item<1> it) {
        int val = buf[it.get_global_id(0)];
        val += sycl::shift_group_left(it.get_sub_group(), val, 1);
        buf[it.get_global_id(0)] = val;
    }).wait();
    sycl::free(buf, q);
}
```
```bash
icpx -fsycl -O2 -g -fsycl-targets=spir64 shift_reduce.cpp -o shift_reduce
```

**Classification signal:** `DumpZEBin=1 NEOReadDebugKeys=1 ./shift_reduce`
produces NO `.elf` files → scenario = `sycl-jit`
→ delegate to `extract-asm-syclkernel-jit`.

**Expected result:**
```
scenario:    sycl-jit
asm-file:    <igc_dump>/OCL_asm*_simd*_entry_*.asm
kernel-name: _ZTS17ShiftReduceKernel
validation:  compiled with -g → grep '// Line' shows source annotations
```
