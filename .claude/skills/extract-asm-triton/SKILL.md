---
name: extract-asm-triton
description: >
  Extract GPU ISA from Triton kernels on XPU. Triton compiles through the same
  IGC backend as SYCL (Triton IR → SPIR-V → IGC → zebin). Extraction is
  identical to sycl-jit: IGC_ShaderDumpEnable=1 captures the zebin at runtime.
  Use when extracting ASM from torch.compile fusions (triton_per_fused,
  triton_poi, triton_red), standalone @triton.jit kernels, or Inductor-generated
  XPU kernels.
---

# Extract ASM from Triton Kernels on XPU

Triton on XPU uses the **same IGC backend** as SYCL JIT. The path is:
`Triton IR → ttgir → SPIR-V → IGC → zebin`. Extraction is mechanically
identical to `extract-asm-syclkernel-jit` — both use `IGC_ShaderDumpEnable=1`
to capture the runtime-compiled zebin. The only difference is how the kernel
enters the pipeline (Triton Python compiler vs DPC++ clang frontend).

## How this fits the compilation stack

```
Triton path:
  @triton.jit Python → Triton IR (ttir) → Triton GPU IR (ttgir)
                                                    ↓
                                              SPIR-V (via triton-xpu backend)
                                                    ↓
                                              IGC (JIT) → zebin
                                                    ↓
                                         IGC_ShaderDumpEnable=1 → .asm

Compare with SYCL JIT:
  SYCL C++ → LLVM IR → SPIR-V → IGC (JIT) → zebin → .asm

The back half (SPIR-V → IGC → zebin) is IDENTICAL.
```

## When to use

- Hot kernel is `triton_per_fused_*` / `triton_poi_*` / `triton_red_*`
  (from `torch.compile` / Inductor)
- Or: standalone `@triton.jit` kernel in a Python script
- You need the actual GPU ISA (not Triton IR)

## When NOT to use

- Kernel is `_ZTS…` (SYCL) → use `extract-asm-syclkernel-{aot,jit}`
- Kernel is oneDNN ngen → use `extract-asm-onednn` (different stack)
- You only need Triton IR → read `${TRITON_CACHE_DIR:-~/.triton/cache}/` directly

## Steps

1. **Identify the kernel name.**

   **Inductor fusion** (from `torch.compile`):
   ```bash
   TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
   rm -rf "$TRITON_CACHE" /tmp/torchinductor_$USER
   TORCH_LOGS=output_code python <repro.py> 2>&1 \
     | grep -oE 'triton_(poi|per|red)_fused_[A-Za-z0-9_]+' | sort -u
   ```

   **Standalone `@triton.jit`**: use unitrace to pin the kernel:
   ```bash
   unitrace -d python <repro.py>
   # Or fallback:
   SYCL_UR_TRACE=-1 python <repro.py> 2>&1 \
     | grep -oP 'pKernelName = 0x[0-9a-f]+ \(\K[^)]+' | sort -u
   ```

2. **Re-run with IGC dump (cold cache).**

   ```bash
   TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
   rm -rf "$TRITON_CACHE" /tmp/torchinductor_$USER
   OUT="<workdir>/triton_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT/igc"

   IGC_ShaderDumpEnable=1 \
   IGC_DumpToCustomDir="$OUT/igc" \
   ONEAPI_DEVICE_SELECTOR=level_zero:0 \
     python <repro.py> 2>&1 | tee "$OUT/run.log"
   ```

3. **Match kernel name to `.asm`.**

   ```bash
   NAME="<fusion-or-kernel-name>"
   MATCH=$(grep -l "$NAME" "$OUT/igc"/OCL_asm*_simd*_entry_*.asm | head -1)
   echo "asm-file: $MATCH"
   ```

   Cross-check: `.zeinfo` reports `simd_size` — confirm it matches
   `num_warps * 32`.
