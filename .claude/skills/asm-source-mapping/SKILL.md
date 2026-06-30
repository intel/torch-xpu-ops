---
name: asm-source-mapping
description: >
  Map Intel GPU ISA instruction addresses (hot IPs from per-IP stall data) to
  precise SYCL/DPC++ source file:line numbers. Primary method reads the DWARF
  .debug_line section from the GPU zebin ELF. Fallback uses structural pattern
  recognition by opcode mix. Use when mapping ASM to source code, finding which
  source line a hot GPU instruction comes from, mapping stall IPs to SYCL source,
  or doing dwarf line table analysis on GPU binaries.
---

# ASM → SYCL Source Mapping (per hot IP)

Given a set of hot instruction addresses (IPs) from per-IP stall data and an ASM
directory produced by `extract-xpu-kernel-asm`, resolve each IP to the original
SYCL/DPC++ source file:line and a structural construct label.

## Background

Debug line info flows: `C++ source → LLVM IR !dbg → SPIR-V OpLine → IGC vISA
DebugLoc → zebin .debug_line` (byte offset → file:line). The zebin's
`.debug_line` section maps GPU ISA offsets to source, and is the primary data
source for this skill. Requires `-g` or `-gline-tables-only` at compile time.
oneDNN ngen has no `.debug_line` — pattern recognition is the only option.

## When to use

- You have hot IPs (from `collect-eu-counters` stall-sampling or profiler).
- An ASM directory from `extract-xpu-kernel-asm` is available.
- You need precise source attribution before optimization.

## When NOT to use

- ASM is not available (blocked binary, closed-source kernel).
- The hot IP is in a tiny helper kernel (< 1% of total GPU time) — skip mapping.

## Method Priority (strict waterfall)

```
Step 1: Try zebin .debug_line          → method = "debug-line"
         ↓ (not found)
Step 2: Try IGC ASM inline comments    → method = "debug-line"
         ↓ (not found)
Step 3: Fallback: pattern recognition  → method = "pattern-recognition"
```

**Rule: if Method 1 succeeds, STOP. Do NOT run pattern-recognition for line
attribution. Only add an optional construct label as enrichment.**

## Method 1 — Zebin `.debug_line` (primary)

The zebin ELF (`target.bin.N`) must contain a `.debug_line` section (present when
compiled with `-g` or `-gline-tables-only`).

1. **Locate the zebin ELF.**

   The zebin is in the **extraction working directory** (where you ran
   `clang-offload-extract` or `ar x`), NOT inside the `ocloc disasm -dump`
   output directory. Typical locations:
   - Path A (standalone AOT): `<workdir>/target.bin.0`
   - Path B (fat binary): `<workdir>/64.bmg` (after `ar x`)
   - JIT: zebin is in memory only — use Method 2 instead.

   ```bash
   ZEBIN=<path to zebin ELF from extraction step>
   file "$ZEBIN"  # must show: ELF 64-bit LSB relocatable, *unknown arch 0xcd*
   ```

2. **Verify `.debug_line` exists.**

   ```bash
   readelf -S "$ZEBIN" | grep -q .debug_line || { echo "NO debug_line"; exit 1; }
   ```

3. **Decode the line table.**

   ```bash
   readelf --debug-dump=decodedline "$ZEBIN" 2>/dev/null
   ```

   Note: `readelf` may warn about unknown reloc types for `e_machine=205`
   (Intel GPU). **Ignore it** — the decoded output is still correct.

   Output format:
   ```
   shift_reduce.cpp                  77                   0               x
   shift_reduce.cpp                  91                0xe8               x
   shift_reduce.cpp                  93               0x238               x
   ```

4. **Build a lookup table.** Parse into sorted `(offset, file, line)` tuples.
   For a given hot IP offset, find the entry with the largest `offset ≤ IP`
   (floor lookup).

5. **Correlate with ASM file.** The `.asm` file uses labels `L<N>` where `N`
   is the **decimal byte offset** in `.text`. Map IP offset → label → ASM line.

## Method 2 — IGC ASM Inline Comments (secondary)

JIT-compiled kernels (`IGC_ShaderDumpEnable=1`) produce `.asm` files with
inline source comments. Two formats exist depending on IGC version:

```asm
// Format A (newer IGC, requires -g):
// Line 8:  int val = buf[it.get_global_id(0)];
(W) send.ugm (32|M0) r16 r14 ...
// Line 9:  val += sycl::shift_group_left(...);
(W) mov (16|M0) r20.0<1>:d r16.0<1;1,0>:d

// Format B (older IGC):
(W) add (M1, 16) r14.0<1>:d r14.0<1;1,0>:d 0x40:w  // shift_reduce.cpp:93
```

Scan for `// Line <N>:` or `// <file>:<line>` patterns. For each hot IP,
walk backward to the nearest preceding annotated line.

## Method 3 — Pattern Recognition (fallback only)

**Use ONLY when Methods 1 and 2 both fail** (no `-g` at compile time, or
oneDNN ngen JIT which has no DWARF).

Classify each hot IP by surrounding instruction mix:

| Instruction pattern | Source construct |
|---|---|
| Dense `dpas` chain, 8×repeat | GEMM tile (matmul accumulate) |
| `mul :f` + broadcast src1 | Scalar rescale (softmax denominator) |
| `exp2 :f` / `log2 :f` | Softmax numerics |
| `mov :bf :f` + `store` | BF16 epilogue / output write |
| `send.slm` | SLM load/store |
| `send.slm` store + `fence.slm` + `send.slm` load + `add :f` | Reduction tree (workgroup reduce) |
| `send.ugm` | Global memory load/store |
| `send.gtwy` + `sync.bar` | Barrier / synchronization |
| VxH indirect `mov r[a0.0]` | Sub-group shuffle (`shift_group_left`) |

## Output Format

JSON array, one entry per hot IP, sorted by stall count descending:

```json
[
  {
    "ip": "0x238",
    "asm_offset": 568,
    "asm_label": "L568",
    "sycl_file": "shift_reduce.cpp",
    "sycl_line": 93,
    "source_construct": "ALU compute (value += other)",
    "method": "debug-line"
  }
]
```

Note: `asm_label` = `L<decimal byte offset>` (matching `.asm` file labels).
`asm_offset` is the same value as an integer. `ip` is the hex form.

## Examples

### Example 1 — AOT binary with `.debug_line` (Method 1)

Source (compiled with `icpx -fsycl -g -fsycl-targets=spir64_gen`):
```cpp
class ShiftReduceKernel;
q.parallel_for<ShiftReduceKernel>(nd_range<1>(1024, 32), [=](nd_item<1> it) {
    int val = buf[it.get_global_id(0)];                    // line 8
    val += sycl::shift_group_left(it.get_sub_group(), val, 1); // line 9
    buf[it.get_global_id(0)] = val;                        // line 10
});
```

```bash
$ readelf --debug-dump=decodedline target.bin.0
# Output (offsets and lines vary by compiler version):
# <source_file>         <line>    <offset>
# shift_reduce.cpp        8        0xe8
# shift_reduce.cpp        9        0x238
# ...

# Hot IPs from stall-sampling: [0x238 Dist=2890]
# Floor lookup: largest offset ≤ 0x238 → line 9
# Result:
#   ip=0x238 → shift_reduce.cpp:9
#   construct="ALU compute (shift_group_left)"
#   method=debug-line
```

### Example 2 — JIT with inline comments (Method 2)

```asm
# From IGC_ShaderDumpEnable=1 dump (compiled with -g):
// Line 8:  int val = buf[it.get_global_id(0)];
(W)     send.ugm (32|M0)  r16  r14  ...
// Line 9:  val += sycl::shift_group_left(...);
(W)     mov (16|M0)  r20.0<1>:d  r16.0<1;1,0>:d

# Hot IP is at the mov instruction → walk backward → "// Line 9"
# Result:
#   sycl_line=9  method=debug-line
```

### Example 3 — No debug info, oneDNN ngen (Method 3 fallback)

```
# No .debug_line, no // comments (oneDNN ngen JIT kernel)
# Hot IP surrounded by dense dpas instructions:
#   dpas (8) r40.0<1>:f  r36.0<8;8,1>:hf  r20.0<8;4,1>:hf
#   dpas (8) r48.0<1>:f  r44.0<8;8,1>:hf  r20.0<8;4,1>:hf
# Result:
#   sycl_file=null  construct="GEMM tile/dpas"  method=pattern-recognition
```
