---
name: extract-asm-syclkernel-aot
description: >
  Extract GPU ISA from AOT-compiled SYCL binaries. The zebin ELF is already
  embedded in the host binary's __CLANG_OFFLOAD_BUNDLE__ section. Uses
  clang-offload-extract + ocloc disasm. Use when disassembling AOT SYCL kernels,
  extracting ASM from libtorch_xpu_ops.so, libtorch_xpu.so, SYCL-TLA FMHA, or
  any DPC++ binary compiled with -fsycl-targets=spir64_gen.
---

# Extract ASM from AOT-Compiled SYCL Binaries

The zebin (GPU ISA) was compiled at build time and embedded in the host binary.
This skill's job is: **find it → extract it → disassemble it → pin the right one**.

## How this fits the compilation stack

```
Build time (already done):
  SYCL source → LLVM IR → SPIR-V → IGC → zebin
                                           ↓
                                    embedded in binary via
                                    __CLANG_OFFLOAD_BUNDLE__

This skill (runtime):
  binary → clang-offload-extract → target.bin.N → ...
```

**Two sub-formats exist** (depending on how the binary was built):

| Source | Format of `target.bin.N` | Extra steps |
|---|---|---|
| Standalone DPC++ (`icpx -fsycl-targets=spir64_gen`) | Plain zebin ELF | None — directly `ocloc disasm` |
| PyTorch (`libtorch_xpu.so`) | zstd-wrapped AR fat binary | Decompress → `ar x 64.<device>` → `ocloc disasm` |

**Why IGC_ShaderDumpEnable does NOT work**: IGC was never called at runtime —
the binary already contains pre-compiled device code.

## When to use

- Kernel is a mangled SYCL symbol (`_ZTS…`) from an AOT binary:
  - `libtorch_xpu.so` / `libtorch_xpu_ops.so` (default PyTorch XPU build)
  - SYCL-TLA FMHA `.so`
  - Standalone DPC++ executable
- Confirm: `strings <binary> | grep -q __CLANG_OFFLOAD_BUNDLE`

## When NOT to use

- No AOT bundle → use `extract-asm-syclkernel-jit` (same stack, different extraction)
- Kernel is oneDNN ngen → use `extract-asm-onednn` (different stack entirely)
- Kernel is Triton → use `extract-asm-triton`

## Steps

### Step 0: Locate tools

`ocloc` is required for disassembly. Ensure it is on PATH:

```bash
command -v ocloc >/dev/null || { echo "ocloc not found; source oneapi-vars.sh"; exit 1; }
```

### Preferred: Runtime dump via NEO driver (`DumpZEBin=1`)

This is the **recommended approach** because it only dumps kernels that are
actually called during execution — no need to sift through hundreds of unused
kernels embedded in the PyTorch binary.

1. **Run the workload with NEO debug keys to dump zebin ELFs.**

   ```bash
   OUT="<workdir>/aot_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT" && cd "$OUT"

   DumpZEBin=1 NEOReadDebugKeys=1 python test.py
   ls *.elf
   ```

   Only kernels actually dispatched to the GPU during `test.py` will appear
   as `.elf` files. Kernels that exist in the binary but were not called are
   NOT dumped.

2. **Disassemble each ELF.**

   ```bash
   for elf in *.elf; do
     name=$(basename "$elf" .elf)
     ocloc disasm -file "$elf" -dump "${name}_dump" -device bmg
   done
   ```

3. **Identify the kernel.**

   The ASM file is at `<name>_dump/.text._ZTSxxxx.asm`. Demangle the name:

   ```bash
   ls *_dump/.text.*.asm
   # For each .asm file, demangle the kernel name:
   for f in *_dump/.text._ZTS*.asm; do
     mangled=$(basename "$f" .asm | sed 's/^\.text\.//')
     echo "$f  →  $(c++filt "$mangled")"
   done
   ```

   Match the demangled name to your target kernel.

### Fallback: Static extraction via `clang-offload-extract`

Use this only when you **cannot run the workload** (e.g., analyzing a binary
on a different machine, or the binary requires hardware you don't have).
The preferred `DumpZEBin` method above handles all runtime scenarios regardless
of binary packaging format (standalone, fat binary, zstd-compressed, etc.).

```bash
# Detect oneAPI root: check env vars first, then common install locations
ONEAPI=${ONEAPI_ROOT:-${CMPLR_ROOT:+${CMPLR_ROOT%/*}}}
if [ -z "$ONEAPI" ]; then
  for d in /opt/intel/oneapi ~/intel/oneapi /usr/local/oneapi; do
    [ -d "$d" ] && ONEAPI="$d" && break
  done
fi
COE=$(command -v clang-offload-extract 2>/dev/null \
  || { test -n "$ONEAPI" && find "$ONEAPI" -name 'clang-offload-extract' 2>/dev/null | head -1; })
test -n "$COE" || { echo "clang-offload-extract not found; install oneAPI compiler or set ONEAPI_ROOT"; exit 1; }

BIN=<path to executable or .so>
OUT="<workdir>/aot_static_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT" && cd "$OUT"

# Extract all offload bundles
"$COE" "$BIN"

# For each target.bin.N:
# - If plain ELF (arch 0xcd): ocloc disasm -file target.bin.N -dump asmN -device bmg
# - If zstd-compressed: decompress → ar x 64.bmg → ocloc disasm
# - Kernel ASM is at asmN/.text._ZTSxxxx.asm; demangle with c++filt
```

**WARNING**: This extracts ALL kernels embedded in the binary, not just those
called at runtime. You must separately identify the target kernel (e.g., via
`SYCL_UR_TRACE=-1` or `unitrace`).
