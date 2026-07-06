---
name: extract-asm-syclkernel-jit
description: >
  Extract GPU ISA from JIT-compiled SYCL kernels. The binary only contains
  SPIR-V; IGC compiles it to native ISA at first launch. The zebin only exists
  in memory at runtime, so IGC_ShaderDumpEnable=1 is required to capture it.
  Use when extracting ASM from SYCL JIT kernels, torch-xpu-ops built with
  TORCH_XPU_ARCH_LIST=none, or standalone DPC++ without AOT flags.
---

# Extract ASM from JIT-Compiled SYCL Kernels

Same compilation stack as AOT (SYCL → LLVM IR → SPIR-V → IGC → zebin), but
execution happens at first kernel launch. The zebin only exists **transiently in
memory** — you MUST re-run with `IGC_ShaderDumpEnable=1` to capture it to disk.

## How this fits the compilation stack

```
Build time:
  SYCL source → LLVM IR → SPIR-V
                              ↓
                    embedded in binary (no native code yet)

First launch (runtime):
  SPIR-V → IGC (JIT) → zebin (in memory only!)
                              ↓
            IGC_ShaderDumpEnable=1 → dumps to disk ← this is what we capture
                              ↓
                         .asm file (with // file:line comments from IGC)
```

**Why `IGC_ShaderDumpEnable=1` is required**: Unlike AOT where the zebin is
statically extractable from the binary, JIT zebin only exists in GPU driver
memory. Without the dump flag, it's never written to disk.

**Bonus**: JIT dump produces `.asm` files with inline `// file:line` comments
(IGC's `EmitPass` annotates them directly).

## When to use

- Kernel is `_ZTS…` AND one of:
  - Binary has NO `__CLANG_OFFLOAD_BUNDLE` (pure JIT build)
  - Binary has AOT bundle but **target does not match current device** (e.g.
    built for PVC, running on BMG — runtime falls back to JIT via SPIR-V)
- Common cases:
  - `torch-xpu-ops` built with `TORCH_XPU_ARCH_LIST=none`
  - Standalone DPC++ without `-fsycl-targets=spir64_gen`
  - Device not in the AOT target list (e.g. `TORCH_XPU_ARCH_LIST=pvc` on BMG)

## When NOT to use

- Binary has AOT code **matching current device** → use `extract-asm-syclkernel-aot`
- Kernel is oneDNN ngen → use `extract-asm-onednn` (different stack)
- Kernel is Triton → use `extract-asm-triton`

## Steps

**Prerequisite**: The router (`extract-xpu-kernel-asm`) has already confirmed
that the JIT path is active for the current device. This skill assumes IGC will
be invoked at runtime.

1. **Pin the actually-launched kernel.**

   Use unitrace (preferred) or SYCL_UR_TRACE (fallback):

   ```bash
   unitrace -d <repro_cmd>
   # Or fallback:
   SYCL_UR_TRACE=-1 <repro_cmd> 2>&1 | grep -oP 'pKernelName = 0x[0-9a-f]+ \(\K[^)]+' | sort -u
   ```

   Pick the hot kernel name (`<KERNEL>`).

2. **Re-run with IGC dump (cold cache).**

   ```bash
   OUT="<workdir>/$(basename "$BIN")_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT/igc"

   NEO_CACHE_PERSISTENT=0 \
   IGC_ShaderDumpEnable=1 \
   IGC_DumpToCustomDir="$OUT/igc" \
   ONEAPI_DEVICE_SELECTOR=level_zero:0 \
     <repro_cmd> 2>&1 | tee "$OUT/run.log"

   ls "$OUT/igc"/*.asm | wc -l
   ```

3. **Match kernel name to `.asm` file.**

   ```bash
   MATCH=$(grep -l "$KERNEL" "$OUT/igc"/OCL_asm*_simd*_entry_*.asm | head -1)
   test -n "$MATCH" || { echo "no asm matches $KERNEL"; exit 1; }
   echo "asm-file: $MATCH"
   ```
