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

## When NOT to use

- Kernel is `triton_*` → use `extract-asm-triton`
- Kernel is `_ZTS…` (SYCL) → use `extract-asm-syclkernel-{aot,jit}`
- You only need primitive-level timing → use `benchdnn` directly

## Steps

### Step 0: Locate tools

```bash
# libiga64.so: shipped with oneAPI debugger component
# Detect oneAPI root: check env vars first, then common install locations
# (CMPLR_ROOT is <oneapi_root>/compiler/<version>, so strip two levels)
ONEAPI=${ONEAPI_ROOT:-${CMPLR_ROOT:+${CMPLR_ROOT%/*/*}}}
if [ -z "$ONEAPI" ]; then
  for d in /opt/intel/oneapi ~/intel/oneapi /usr/local/oneapi; do
    [ -d "$d" ] && ONEAPI="$d" && break
  done
fi
IGA_LIB=$(find ${ONEAPI:?"oneAPI not found; set ONEAPI_ROOT"} -name 'libiga64.so' 2>/dev/null | head -1)
test -n "$IGA_LIB" || { echo "libiga64.so not found under $ONEAPI; install oneAPI debugger"; exit 1; }
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
PLATFORM=0x2000000  # iga_gen_t value, see the list below
for bin in dnnl_dump_gpu_*.bin; do
  name=$(basename "$bin" .bin)
  python3 - "$IGA_LIB" "$PLATFORM" "$bin" > "${name}.asm" <<'EOF'
import ctypes, pathlib, sys

lib, platform, path = sys.argv[1], int(sys.argv[2], 0), sys.argv[3]
iga = ctypes.CDLL(lib)

class ContextOptions(ctypes.Structure):  # iga_context_options_t
    _fields_ = [("cb", ctypes.c_size_t), ("gen", ctypes.c_int)]

class DisassembleOptions(ctypes.Structure):  # iga_disassemble_options_t
    _fields_ = [(f, ctypes.c_uint32) for f in (
        "cb", "formatting_opts", "_reserved0", "_reserved1",
        "decoder_opts", "base_pc_offset")]

ctx = ctypes.c_void_p()
copts = ContextOptions(ctypes.sizeof(ContextOptions), platform)
status = iga.iga_context_create(ctypes.byref(copts), ctypes.byref(ctx))
if status != 0:
    sys.exit(f"iga_context_create failed with status {status}")
raw = pathlib.Path(path).read_bytes()
dopts = DisassembleOptions(cb=ctypes.sizeof(DisassembleOptions))
text = ctypes.c_char_p()
status = iga.iga_context_disassemble(
    ctx, ctypes.byref(dopts), raw, len(raw), None, None, ctypes.byref(text))
sys.stdout.write((text.value or b"").decode())
iga.iga_context_release(ctx)
if status != 0:
    sys.exit(f"iga_context_disassemble failed with status {status}")
EOF
done
```

   **Platform ID selection** (`iga_gen_t` in `iga/iga.h`,
   `XE_VER(major, minor) = (major << 24) | minor`):
   - BMG / LNL (Xe2): `0x2000000`
   - PTL (Xe3): `0x3000000`
   - PVC (Xe-HPC): `0x1000004`
   - DG2 / MTL (Xe-HPG): `0x1000002`

   A non-zero status means the platform ID is unsupported (context
   creation) or does not match the bytes (decode error). Try the next one
   in the list above.

   If the `iga64` CLI is on PATH, `iga64 -d -p bmg "$bin" > "${name}.asm"`
   produces the same output (`iga64 -h=p` lists the platform names).

3. **Pin the actually-invoked kernel.**

   ```bash
   ONEDNN_VERBOSE=1 <repro_cmd> 2>&1 | grep -E '^onednn_verbose.*exec' \
     | nl -ba | tee dnnl_verbose.log
   # Nth (1-indexed) exec line → dnnl_dump_gpu_*_kernel.<N-1>.bin
   ```

   The largest `.bin` by size is typically the GEMM kernel.
   Cross-check with `grep -c dpas <asm>` (GEMM has high dpas density).
