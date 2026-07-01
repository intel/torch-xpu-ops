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

These tools are NOT always on PATH. Probe in order:

```bash
# clang-offload-extract: shipped with oneAPI compiler, but buried in bin/compiler/
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

# ocloc: usually on PATH after sourcing oneapi-vars.sh
command -v ocloc >/dev/null || { echo "ocloc not found; source oneapi-vars.sh"; exit 1; }
```

### Path A: Standalone DPC++ binary (plain zebin ELF)

1. **Extract and disassemble.**

   ```bash
   BIN=<path to executable or .so>
   OUT="<workdir>/aot_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT" && cd "$OUT"

   "$COE" "$BIN"
   file target.bin.0    # → "ELF 64-bit LSB relocatable, *unknown arch 0xcd*"

   ocloc disasm -file target.bin.0 -dump asm0 -device bmg
   ls asm0/.text.*.asm
   ```

2. **Pin kernel.**

   ```bash
   KERNEL=$(SYCL_UR_TRACE=-1 <repro_cmd> 2>&1 \
     | grep -oP 'pKernelName = 0x[0-9a-f]+ \(\K[^)]+' | head -1)
   # The kernel name matches the .text.<name>.asm filename:
   ls asm0/.text.*"$KERNEL"*.asm
   ```

### Path B: zstd-compressed multi-target fat binary (e.g. PyTorch XPU `.so`)

When a SYCL library is compiled with multiple `-device` targets, the compiler
produces a **zstd-compressed AR archive** per compilation unit. Each archive
contains one zebin per target device. The runtime selects the matching device
binary at load time.

1. **Extract all bundles.**

   ```bash
   LIB=<path to .so with multi-target AOT>
   OUT="<workdir>/aot_fat_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$OUT" && cd "$OUT"

   "$COE" "$LIB"
   ls target.bin.* | wc -l
   file target.bin.0   # → "Zstandard compressed data" or "current ar archive"
   ```

2. **Decompress (if zstd) and list device members.**

   If `file` reports "Zstandard compressed data" (magic bytes `28 b5 2f fd`),
   decompress first with `python3 -c "import zstandard; ..."` or `zstd -d`.
   The decompressed output is an AR archive.

   ```bash
   # Decompress if needed (no-op if already plain AR)
   python3 -c "
   import zstandard, sys
   data = open('target.bin.0','rb').read()
   out = zstandard.ZstdDecompressor().decompress(data) if data[:4]==b'\x28\xb5\x2f\xfd' else data
   open('target.bin.0.ar','wb').write(out)"

   # Discover available device targets
   ar t target.bin.0.ar | grep -v pad
   # → 64.bmg, 64.dg2, 64.12.60.7, 64.12.74.4, ..., generic_ir
   ```

3. **Extract the zebin for your device and disassemble.**

   ```bash
   DEVICE_MEMBER=64.bmg   # from ar t output, match your hardware
   DEVICE_NAME=bmg        # ocloc device name

   ar x target.bin.0.ar "$DEVICE_MEMBER"
   file "$DEVICE_MEMBER"  # → ELF 64-bit LSB relocatable, *unknown arch 0xcd*

   ocloc disasm -file "$DEVICE_MEMBER" -dump asm -device "$DEVICE_NAME"
   ls asm/.text.*.asm
   ```

4. **Find which `target.bin.N` contains your kernel.**

   With many compilation units, grep the decompressed binary data for the
   kernel name substring (kernel names are embedded as strings in the zebin):

   ```bash
   for f in target.bin.*; do
     python3 -c "
   import zstandard, sys
   data = open('$f','rb').read()
   if data[:4]==b'\x28\xb5\x2f\xfd': data=zstandard.ZstdDecompressor().decompress(data)
   if b'AddFunctor' in data: print('FOUND in $f')" 2>/dev/null
   done
   ```
