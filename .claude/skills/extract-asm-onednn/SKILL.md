---
name: extract-asm-onednn
description: >
  Extract GPU ISA from oneDNN ngen-JIT kernels. This is the ONLY codegen path
  that bypasses the standard SYCL/SPIR-V/IGC stack. oneDNN uses its own native
  code generator (ngen) that directly emits GPU ISA bytes — no SPIR-V, no IGC,
  no zebin ELF, no .debug_line. Use when extracting ASM from oneDNN kernels
  (gemm_kernel, gen_conv_kernel), matmul, linear, conv, or SDPA-graph ops
  dispatched via mkldnn.
---

# Extract ASM from oneDNN ngen-JIT Kernels

**This is the ONLY path that does NOT use the standard compilation stack.** All
other scenarios (SYCL AOT/JIT, Triton) go through `SPIR-V → IGC → zebin`.
oneDNN ngen bypasses all of that.

## How this differs from the standard stack

```
Standard stack (SYCL/Triton):
  Source → LLVM IR → SPIR-V → IGC → zebin ELF (.text + .debug_line)
                                         ↓
                                    ocloc disasm → .asm

oneDNN ngen (THIS skill):
  oneDNN C++ templates → ngen JIT → RAW ISA BYTES (no ELF, no DWARF)
                                         ↓
                                    IGA ctypes → .asm
```

Key consequences:
- **No zebin** — output is raw bytes, not ELF
- **No `.debug_line`** — source mapping can only use pattern recognition
- **No IGC** — `IGC_ShaderDumpEnable` is useless
- **No SPIR-V** — ngen directly emits machine instructions
- **Dump mechanism**: `ONEDNN_JIT_DUMP=1` (writes `.bin` files)
- **Disassembly**: IGA library via ctypes (not `ocloc disasm`)

## When to use

- Op dispatched through oneDNN (`mkldnn::*`): `linear` / `matmul` / `mm` /
  `bmm` / `conv*` / `_scaled_dot_product_attention` (oneDNN-graph)
- Hot kernel named `gemm_kernel` / `gen_conv_kernel`
- Hardware: Intel XPU (PVC / BMG-G31 / DG2 / LNL / ARL)

## When NOT to use

- Kernel is `triton_*` → use `extract-asm-triton`
- Kernel is `_ZTS…` (SYCL) → use `extract-asm-syclkernel-{aot,jit}`
- You only need primitive-level timing → use `benchdnn` directly

## Steps

### Step 0: Locate tools

```bash
# libiga64.so: shipped with oneAPI debugger component
# ONEAPI_ROOT may vary (e.g. /opt/intel/oneapi, ~/intel/oneapi, /usr/local/oneapi)
ONEAPI=${ONEAPI_ROOT:-${CMPLR_ROOT%/*}}
IGA_LIB=$(find ${ONEAPI:-/opt/intel/oneapi} -name 'libiga64.so' 2>/dev/null | head -1)
test -n "$IGA_LIB" || { echo "libiga64.so not found; install oneAPI debugger or set ONEAPI_ROOT"; exit 1; }
export IGA_LIB
```

1. **Dump raw ISA via `ONEDNN_JIT_DUMP`.**

   ```bash
   OUT="<workdir>/onednn_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT" && cd "$OUT"

   ONEDNN_JIT_DUMP=1 \
   ONEAPI_DEVICE_SELECTOR=level_zero:0 \
     <repro_cmd> 2>&1 | tee run.log

   ls dnnl_dump_gpu_*.bin
   ```

   These are **raw ISA bytes** (not zebin ELF). `ocloc disasm` cannot
   read them.

2. **Disassemble with IGA ctypes.**

   oneDNN `.bin` files are raw ISA bytes (not ELF). Use `libiga64.so` via
   Python ctypes to disassemble. No separate script needed — run inline:

   ```bash
   for bin in dnnl_dump_gpu_*.bin; do
     name=$(basename "$bin" .bin)
     python3 -c "
import ctypes, sys, pathlib
iga = ctypes.CDLL('$IGA_LIB')
raw = pathlib.Path('$bin').read_bytes()
buf = ctypes.create_string_buffer(1 << 20)  # 1MB output buffer
# Platform IDs: BMG/Xe2=0x2000000, PVC=0x30000, DG2=0x30004
iga.iga_disassemble(0x2000000, raw, len(raw), buf, len(buf))
sys.stdout.write(buf.value.decode())
" > "${name}.asm"
   done
   ```

   **Platform ID selection:**
   - BMG / Xe2 / LNL: `0x2000000`
   - PVC (Ponte Vecchio): `0x30000`
   - DG2 (Alchemist): `0x30004`

   If `iga_disassemble` returns 0 bytes, the platform ID likely doesn't
   match. Try the next one in the list above.

3. **Pin the actually-invoked kernel.**

   ```bash
   ONEDNN_VERBOSE=1 <repro_cmd> 2>&1 | grep -E '^onednn_verbose.*exec' \
     | nl -ba | tee dnnl_verbose.log
   # Nth (1-indexed) exec line → dnnl_dump_gpu_*_kernel.<N-1>.bin
   ```

   The largest `.bin` by size is typically the GEMM kernel.
   Cross-check with `grep -c dpas <asm>` (GEMM has high dpas density).
