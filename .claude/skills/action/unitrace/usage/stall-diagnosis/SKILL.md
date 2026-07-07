---
name: unitrace-stall-diagnosis
description: Diagnose GPU kernel performance stalls using unitrace on Intel Xe2/BMG. Use this skill when the user asks "why is my kernel slow", "where is the kernel stalling", "which instruction is the bottleneck", "stall analysis", "hot IP", "dominant stall", or wants to identify the root cause of GPU kernel latency. Covers the full 3-stage workflow from aggregate stall identification through instruction-level sampling to source code mapping.
---

# Unitrace Stall Diagnosis — 3-Stage Workflow

Systematic methodology for identifying why a GPU kernel is slow by locating the dominant stall type, the hot instruction addresses, and mapping them back to source code.

## Agent Guardrails

- Do not assume the target GPU is device 0. If multiple Level Zero GPU devices are found, use `intel-gpu-device-selection/SKILL.md` first.
- Do not assume oneAPI is already sourced. Use `source-oneapi/SKILL.md` first.
- Do not assume `unitrace` is available. Use `unitrace-setup/SKILL.md` before collecting metrics.
- Always set `ZE_AFFINITY_MASK` and `ZE_FLAT_DEVICE_HIERARCHY=FLAT` before profiling.
- Do not skip Stage 1 — always identify the dominant stall before collecting instruction-level data.

## Prerequisites

- Intel Arc B580 or compatible Xe2/BMG GPU
- Intel oneAPI environment sourced
- `unitrace` available in PATH
- Target kernel binary built and runnable
- For source mapping: access to the `.asm` dump or `_gen.cpp` source (nGEN kernels) or IGC debug info (SYCL/OpenCL kernels)

## Environment Setup

### step 1: select gpu device
### step 2: source oneAPI
### step 3: setup unitrace

---

## Stage 1: Identify Dominant Stall Type

**Goal**: Determine which stall category dominates the kernel's execution time.

### Command

```bash
unitrace -q -g VectorEngineStalls -o stalls_output ./<runner> [args...]
```

### Output Format

The metrics CSV (`stalls_output.metrics.<PID>.csv`) contains columns:

```
Kernel, GlobalInstanceId, SubDeviceId, GpuTime[ns], GpuCoreClocks[cycles],
..., XveActive[%], XveStall[%], XveStallSbid[%], XveStallSendWr[%],
XveStallInstFetch[%], XveStallOther[%], XveStallDep[%], XveStallPipe[%],
XveStallControl[%], XveStallDist[%], XveStallSync[%], ...
```

### Stall Types (Xe2/BMG)

| Stall Column | Meaning | Common Cause |
|---|---|---|
| `XveStallSbid[%]` | Waiting for a send (load/store) to complete via SBID token | Memory-latency-bound; too few in-flight requests |
| `XveStallSendWr[%]` | Send port busy; cannot issue another send | Memory-bandwidth-bound; HW send queue saturated |
| `XveStallDist[%]` | Distance dependency; prior ALU result not ready | Insufficient SWSB distance; long-latency ALU (e.g., indirect access) |
| `XveStallDep[%]` | General data dependency stall | RAW hazard; tight loop with data chain |
| `XveStallInstFetch[%]` | Instruction cache miss | Large kernel, cold start, or loop alignment |
| `XveStallControl[%]` | Control flow (branch/join) overhead | Divergent branches, loop exit |
| `XveStallPipe[%]` | Pipeline structural hazard | Back-to-back ops competing for same pipe (see `benchmarks/pipestall/`) |
| `XveStallSync[%]` | Barrier or fence synchronization | `sync.bar`, `sync.fence` |
| `XveStallOther[%]` | Unclassified | Rare; may indicate TLB miss or other micro-arch event |

### Interpretation

1. Sum all `XveStall*` percentages for the target kernel.
2. Identify the **largest** stall percentage — this is the dominant bottleneck.
3. Decision tree:
   - **SbidStall dominant** → kernel is latency-bound on memory; needs more MLP (memory-level parallelism) or prefetching
   - **SendWrStall dominant** → kernel is bandwidth-bound; memory subsystem is saturated; adding more threads won't help
   - **DistStall dominant** → ALU dependency chain too tight; SWSB distance annotation underestimates actual latency
   - **DepStall dominant** → tight data dependency loop; consider instruction reordering or loop unrolling
   - **InstFetchStall dominant** → instruction cache pressure; kernel too large or cold-start effect
   - **SyncStall dominant** → barrier/fence cost; reduce synchronization frequency or scope

### Example

```bash
unitrace -q -g VectorEngineStalls -o ves ./<runner> --iters 4096

# Parse output:
# XveActive=12%, XveStall=88%
# XveStallSbid=5%, XveStallDist=72%, XveStallControl=8%, XveStallOther=3%
# → Dominant: DistStall (72%) → dependency chain is the bottleneck
```

---

## Stage 2: Collect Hot IPs (Instruction-Level Stall Sampling)

**Goal**: Find which specific instruction addresses accumulate the most stall samples.

### Command

```bash
unitrace --stall-sampling -o hotips ./<runner> [args...]
```

For a specific kernel only (reduces noise):

```bash
unitrace --stall-sampling --include-kernels "<kernel_name>" -o hotips ./<runner> [args...]
```

### Output Format

The stall-sampling metrics CSV (`hotips.metrics.<PID>.csv`) contains per-IP rows:

```
Kernel, IP[Address], Active[Events], PSDepStall[Events], ControlStall[Events],
PipeStall[Events], SendStall[Events], DistStall[Events], SbidStall[Events],
SyncStall[Events], InstrFetchStall[Events], OtherStall[Events]
```

Each row = one instruction address in one kernel. The `IP[Address]` is a hex offset from the kernel entry point. Each stall column shows the number of **sample events** observed at that IP.

### Interpretation

1. Filter rows for the target kernel.
2. Sort by the dominant stall column identified in Stage 1 (e.g., sort by `DistStall` descending).
3. The top 1-3 IPs are the "hot" instructions — where the EU spends most time stalled.
4. Note: The stall is attributed to the instruction that is **waiting**, not the one that is slow to produce. The producing instruction is typically the one at the preceding IP.

### Example

```
Kernel,             IP,  Active, DistStall, SbidStall, ...
"my_kernel_s16",  0x360,      8,       354,         0, ...  ← HOT (waiting for r[a0.0])
"my_kernel_s16",  0x350,      9,         0,         0, ...  ← producer (indirect read)
"my_kernel_s16",  0x380,      9,       117,         0, ...
"my_kernel_s16",  0x370,      7,        10,         0, ...
```

Here IP `0x360` has 354 DistStall events — the EU stalls here waiting for the result produced by IP `0x350`.

### Key Insight

The **hot IP** is the **consumer** (the dependent instruction). To find the root cause, look at the **preceding instruction(s)** that produce the value it depends on. The actual "slow" operation is the producer, but the stall manifests at the consumer.

---

## Stage 3: Map Hot IPs to Source Code

**Goal**: Identify which source-level operation corresponds to the hot instruction addresses.

### For nGEN Kernels (manual ASM mapping)

nGEN kernels have no debug info. Use disassembly to correlate IPs.

#### Step 3a: Disassemble the kernel binary

```bash
# Extract .text section from zebin ELF (using objcopy or readelf)
objcopy -O binary --only-section=.text.kernel_name kernel.zebin kernel_text.bin

# Disassemble with IGA
iga64 -d -p xe2 kernel_text.bin > kernel.asm
```

Or if the build already produces `.asm` dumps (ngen_bmg_standalone does):
```bash
# The _gen binary produces .asm alongside .zebin
ls *.asm
```

#### Step 3b: Compute instruction offset

Each Xe2 instruction is **16 bytes (0x10)**. The IP address from stall sampling is:

```
IP = instruction_index * 0x10
```

So `IP 0x360` = instruction #54 (0x360 / 0x10 = 54) from kernel start.

#### Step 3c: Find the instruction in disassembly

Look at the disassembly output and count instructions, or search for the offset:

```asm
// At offset 0x350 (instruction 53):
(W)  mov  (16|M0)  r8.0<1>:d   r[a0.0]<1,0>:d    // VxH indirect read

// At offset 0x360 (instruction 54):            ← HOT IP
(W)  mov  (16|M0)  a0.0<1>:uw  r8.0<2;1,0>:uw    // depends on r8 from above
```

#### Step 3d: Map to nGEN source code

In the `_gen.cpp` file, find the emission call that produces this instruction. nGEN emits instructions sequentially, so the order in `_gen.cpp` matches the order in disassembly (after prologue/loadlid/loadargs).

```cpp
// In kernel_gen.cpp — this emits the instruction at IP 0x350:
gen.mov(exec, r8, indirect[a0]);       // ← producer (slow: VxH gather)

// This emits the instruction at IP 0x360:
gen.mov(exec, a0, r8.uw(0)(2,1,0));   // ← consumer (stalls here)
```

### For SYCL/OpenCL Kernels (automated source mapping)

SYCL kernels compiled with debug info get **automated IP-to-source mapping** via IGC shader dumps.

#### Step 3a: Ensure debug info is present in the GPU binary

The GPU kernel binary must contain DWARF line tables for source mapping to work.

**CMake build types** — check what you already have:

| Build Type | Compiler Flags | GPU debug info? | Action needed |
|---|---|---|---|
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | Yes | None — works out-of-the-box |
| `Debug` | `-O0 -g` | Yes | None, but code gen differs from release |
| `Release` | `-O3 -DNDEBUG` | No | Add `-gline-tables-only` (see below) |

If already using `RelWithDebInfo`, the `-g` flag propagates through the SYCL driver to IGC automatically. Skip to Step 3b.

**For Release builds**, add `-gline-tables-only` explicitly:

```bash
icpx -fsycl -gline-tables-only -O2 -o myapp myapp.cpp
```

This instructs IGC to embed DWARF line table information in the kernel binary. It has negligible runtime cost (does NOT disable optimizations, does NOT change code generation — same as `-O2` without it).

**For AOT builds targeting BMG:**

```bash
icpx -fsycl -fsycl-targets=intel_gpu_bmg -gline-tables-only -O2 -o myapp myapp.cpp
```

Note: The SYCL driver downgrades `-gline-tables-only` to `-g` when passing to the ocloc backend. The result is full debug info (a superset of line tables) embedded in the ZEBinary — functionally equivalent for stall diagnosis.

#### Step 3b: Generate IGC shader dump

Run the application once with IGC dump environment variables to produce source-annotated `.asm` files:

```bash
# JIT builds: dump happens at runtime (first kernel compilation)
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./dump ./myapp

# AOT builds: dump happens at BUILD time, set env during compilation
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./dump icpx -fsycl -fsycl-targets=intel_gpu_bmg -g -O2 -o myapp myapp.cpp

# AOT alternative: disassemble the already-built zebin post-hoc
ocloc disasm -file myapp -device bmg -dump ./dump
```

This produces files in `./dump/` including:
- `*_simd16.asm` or `*_simd32.asm` — annotated assembly with embedded source lines
- `*.spv` — SPIR-V intermediate
- `*.ll` — LLVM IR at various stages

The `.asm` files contain source annotations like:
```asm
/* [000001B8] */   sync.nop    null    {Compacted,$5.dst}     // $15
  Line 40:  c[index] = a[index] + b[index];
  File: /home/user/myapp.cpp
```

#### Step 3c: Collect stall samples

```bash
unitrace --stall-sampling --chrome-kernel-logging -o stallperf.csv ./myapp
```

#### Step 3d: Analyze with PTI tools (automated IP-to-source)

Use `analyzeperfmetrics.py` from the PTI SDK to correlate stall IPs with source lines:

```bash
python analyzeperfmetrics.py \
    -k "kernel_name" \
    -s ./dump \
    -o stallchart.pdf \
    stallperf.metrics.<PID>.csv
```

This script:
1. Parses the stall-sampling CSV for hot IPs
2. Scans the IGC `.asm` dump for `Line NN:` and `File:` annotations
3. Maps each stall IP backward to the nearest source annotation
4. Produces a PDF chart showing stall distribution by source line

Or use `uniview.py` for an interactive visual report:

```bash
python uniview.py \
    -t myapp.<PID>.json \
    -m stallperf.metrics.<PID>.csv \
    -s ./dump
```

The scripts are located at:
```
pti-gpu/tools/unitrace/scripts/metrics/analyzeperfmetrics.py
pti-gpu/tools/unitrace/scripts/uniview.py
```

#### Step 3e: Manual mapping (if tools unavailable)

If the PTI analysis scripts are not available, manually correlate:

1. Open the `.asm` file from `./dump/` for the target kernel.
2. Each instruction has a hex offset in brackets: `/* [000001B8] */`
3. Match hot IPs from Stage 2 to these offsets.
4. Look for the nearest `Line NN:` / `File:` annotation above that instruction.

Example:
```asm
/* [000001A0] */   send.dc1 (16|M0)  r20:f  ...  // load
  Line 40:  c[index] = a[index] + b[index];
  File: /home/user/myapp.cpp
/* [000001B0] */   add (16|M0) r30:f  r20:f  r22:f  // add    ← HOT IP (SbidStall)
/* [000001C0] */   send.dc1 (16|M0)  null:f  ...  // store
```

If IP `0x1B0` has high SbidStall → the `add` is waiting for the `send` at `0x1A0` to complete → the source line is `c[index] = a[index] + b[index]` → the load of `a[index]` or `b[index]` is the latency bottleneck.

#### SYCL-specific considerations

- **IGC reordering**: IGC aggressively reorders instructions. A single source line may map to multiple non-adjacent instructions. Use the `Line:` annotations (not sequential position) for mapping.
- **Inlining**: If functions are inlined, the `File:` annotation may jump between files. Track both file and line.
- **Loop unrolling**: A hot loop may be unrolled by IGC, producing multiple copies of the same source line at different IPs. Sum stall counts across all copies.
- **SIMD divergence**: IGC may generate different code paths for SIMD lanes. Check both M0 and M16 variants.

### For Large Kernels (many IPs)

When the kernel has many instructions, use this systematic approach:

1. From Stage 2, get the top 3-5 hot IPs.
2. Compute byte offsets: `IP_value * 0x10` (if not already in hex bytes).
3. In the disassembly, mark these locations.
4. Group hot IPs by proximity — clusters indicate a hot loop body.
5. If hot IPs span exactly `N` instructions in sequence, this is likely a single loop iteration.
6. Map the loop body back to the source loop.

---

## Complete Example: nGEN Kernel

```bash
# Setup
export ZE_AFFINITY_MASK=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Stage 1: What stall dominates?
unitrace -q -g VectorEngineStalls -o ves ./my_bench_run --iters 4096
# → Parse CSV: XveStallDist = 72% dominates

# Stage 2: Which instructions are hot?
unitrace --stall-sampling --include-kernels "my_kernel" -o stall ./my_bench_run --iters 100000
# → IP 0x360: DistStall=354 events (top), IP 0x380: DistStall=117

# Stage 3: What source code?
iga64 -d -p xe2 my_kernel_text.bin > my_kernel.asm
# → IP 0x350: mov r8, r[a0.0] (indirect read)
# → IP 0x360: mov a0, r8 (feedback — stalls waiting for r8)
# → Root cause: VxH indirect access latency (~30 cycles) in tight loop
```

## Complete Example: SYCL Kernel

```bash
# Setup
export ZE_AFFINITY_MASK=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Option A: RelWithDebInfo already has -g, no extra flags needed
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --parallel

# Option B: Release build — add -gline-tables-only explicitly
icpx -fsycl -fsycl-targets=intel_gpu_bmg -gline-tables-only -O2 -o myapp myapp.cpp

# Generate IGC shader dump (JIT: at runtime; AOT: at build time or use ocloc disasm)
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./dump NEO_CACHE_PERSISTENT=0 ./myapp

# Stage 1: What stall dominates?
unitrace -q -g VectorEngineStalls -o ves ./myapp
# → Parse CSV: XveStallSbid = 65% dominates → memory-latency-bound

# Stage 2: Which instructions are hot?
unitrace --stall-sampling --include-kernels "SYCL_kernel" -o stall ./myapp
# → IP 0x1B0: SbidStall=892 events (top)

# Stage 3: Automated source mapping
python analyzeperfmetrics.py -k "SYCL_kernel" -s ./dump -o report.pdf stall.metrics.*.csv
# → Output: "Line 40 of myapp.cpp: 65% of stalls (SbidStall)"
# → Root cause: c[index] = a[index] + b[index]; — loads have high latency,
#   insufficient MLP (memory-level parallelism)
```

## Stall Diagnosis Decision Table

| Dominant Stall | Root Cause | Mitigation |
|---|---|---|
| SbidStall | Insufficient memory-level parallelism | Unroll, issue more loads before consuming |
| SendWrStall | Bandwidth saturated | Reduce data volume, improve locality, coalesce |
| DistStall | ALU dependency chain, long-latency op | Break chain, add independent work, increase SWSB distance |
| DepStall | Tight data dependency | Reorder instructions, software pipeline |
| InstFetchStall | Instruction cache miss | Reduce kernel size, align loops, warmup |
| SyncStall | Barrier overhead | Reduce barrier count, use subgroup ops instead |
| ControlStall | Branch divergence | Predication, branchless algorithms |

## Tips

- **Warmup**: Always use `--warmup N` (N > 0) on the runner to avoid measuring cold instruction cache effects in steady-state analysis.
- **Iterations**: Use high iteration counts (10000+) for stall sampling to get statistically significant sample counts.
- **Multiple runs**: Stall sampling is statistical. Run 3-5 times and check that the hot IPs are consistent.
- **Cross-validate**: The dominant stall from Stage 1 (`VectorEngineStalls` percentages) should match the dominant stall column in Stage 2 (stall-sampling per-IP events). If they disagree, increase sampling duration.
- **IP arithmetic**: On Xe2, all instructions are 16 bytes (compacted instructions exist but are rare in nGEN output). If IGA shows different instruction sizes, adjust offset calculation accordingly.
- **SYCL: always use `-gline-tables-only`**: This has zero performance impact and makes source mapping trivial. There is no reason to omit it for profiling builds.
- **SYCL: disable kernel caching**: Set `NEO_CACHE_PERSISTENT=0` when generating shader dumps to ensure IGC recompiles (otherwise it may serve a cached binary without source annotations).
- **SYCL: match kernel names**: Kernel names in unitrace output include SYCL mangling. Use `unitrace -d -v` first to see exact kernel names, then pass to `--include-kernels`.

## Reference Benchmarks

Self-contained nGEN benchmarks that reproduce specific stall types.
Each lives in `benchmarks/<stall_type>/` with its own CMakeLists.txt, source, and documentation.

| Stall Type | Directory | Achieved | Mechanism |
|---|---|---|---|
| PipeStall | `benchmarks/pipestall/` | 47.30% (dominant stall, 48% XVE_STALL) | 48 back-to-back SIMD16 `add` saturating FPU issue rate |
