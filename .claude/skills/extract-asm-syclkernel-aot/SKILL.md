---
name: extract-asm-syclkernel-aot
description: >
  Extract GPU ISA from AOT-compiled SYCL binaries using NEO driver runtime dump.
  Use when the kernel is a SYCL symbol (_ZTS...) from an AOT binary such as
  libtorch_xpu.so, SYCL-TLA FMHA, or any DPC++ binary compiled with
  -fsycl-targets=spir64_gen -Xs "-device <gpu>".
---

# Extract ASM from AOT-Compiled SYCL Binaries

## Key facts

- GPU ISA (zebin) was already compiled at **build time** by IGC and embedded in
  the host binary.
- `IGC_ShaderDumpEnable` does **NOT** work — IGC is never invoked at runtime.
- Extraction uses NEO driver debug keys to dump the zebin ELF at runtime, then
  `ocloc disasm` to produce readable ASM.

## When to use

- Kernel is a mangled SYCL symbol (`_ZTS…`) from an AOT-compiled binary **whose
  AOT target matches the current device**:
  - `libtorch_xpu.so` / `libtorch_xpu_ops.so` (default PyTorch XPU build)
  - SYCL-TLA FMHA `.so`
  - Standalone DPC++ executable built with `-fsycl-targets=spir64_gen`

## When NOT to use

- AOT target does not match current device (e.g. binary built for PVC, running
  on BMG) → runtime falls back to JIT → use `extract-asm-syclkernel-jit`
- Binary has no AOT code at all → use `extract-asm-syclkernel-jit`
- Kernel is oneDNN ngen (`gemm_kernel`, `gen_conv_kernel`) → use `extract-asm-onednn`
- Kernel is Triton (`triton_*`) → use `extract-asm-triton`

## Steps

### Step 1: Verify tools

```bash
command -v ocloc >/dev/null || { echo "ocloc not found; source oneapi-vars.sh"; exit 1; }
```

### Step 2: Dump zebin ELFs via NEO driver

Run the workload with NEO debug keys. Only kernels **actually dispatched** to
the GPU will be dumped — no unused kernels appear.

```bash
OUT="<workdir>/aot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT" && cd "$OUT"

DumpZEBin=1 NEOReadDebugKeys=1 <repro_cmd>
ls *.elf
```

### Step 3: Disassemble

```bash
# Detect device name (bmg, dg2, pvc, etc.)
DEVICE=$(sycl-ls 2>/dev/null | grep -oP '(?<=\[)[^]]+' | head -1 | awk '{print tolower($NF)}')
DEVICE=${DEVICE:-bmg}  # fallback

for elf in *.elf; do
  name=$(basename "$elf" .elf)
  ocloc disasm -file "$elf" -dump "${name}_dump" -device "$DEVICE"
done
```

### Step 4: Identify the target kernel

ASM files are at `<name>_dump/.text._ZTSxxxx.asm`. Demangle to find yours:

```bash
for f in *_dump/.text._ZTS*.asm; do
  mangled=$(basename "$f" .asm | sed 's/^\.text\.//')
  echo "$f  →  $(c++filt "$mangled")"
done
```

Match the demangled name to the kernel you are analyzing.
