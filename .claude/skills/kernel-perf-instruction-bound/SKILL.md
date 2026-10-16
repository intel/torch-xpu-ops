---
name: kernel-perf-instruction-bound
description: >
  Part of the Intel XPU Performance Optimization Skills Suite. Triggered by
  kernel-perf-analysis when a kernel is compute-bound and the bottleneck is
  instruction throughput (ALU0 or ALU1 pipe). Collects VectorEngineProfile and
  ComputeBasic counters, builds per-pipe roofline (T_ALU0, T_ALU1, T_SEND,
  T_CONTROL) vs T_mem, then inspects SYCL source for repeated or redundant
  operations and proposes concrete changes such as IntDivider, dtype narrowing,
  per-row/col precomputation, vectorization, and launch-grid reparameterization.
---

# Kernel Performance Optimization — Instruction-Bound

## Where it sits in the Performance Optimization Skills Suite

Issue #4110 defines a top-down trunk and a fixed kernel-analysis pipeline. This
skill belongs to the **compute-bound (non-GEMM) branch** of that pipeline.

```text
Level 0/1: model/op split        Level 2: kernel roofline
kernel-perf-analysis  ──────────►  build-roofline + unitrace counters
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
         memory-roofline-      gemm-tiling-analysis    kernel-perf-instruction-bound
          breakdown (step 4)    (step 5)                (this skill)
                                                             │
                              ┌──────────────────────────────┘
                              ▼
                  eu-utilization-triage (optional cross-check)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   eu-stall-attribution   (skip)          eu-tlp-occupancy
   when stall dominates   eu-ilp-coissue   when occupancy
                         (not this path)   dominates
                              ▲
                              │
            asm/source evidence: extract-xpu-kernel-asm + asm-source-mapping
```

Entry points:

- **Primary**: `kernel-perf-analysis` routes here when `T_compute ≥ 0.8 × T_mem`
  (default overlap threshold), the kernel is not GEMM-like, and the dominant
  `T_compute` pipe is ALU0/ALU1.
- **Cross-check**: `eu-utilization-triage` may also point here when the
  single-pipe active share is high and the busy pipe is ALU0/ALU1 (not SEND,
  not dominated by stall).

Follow-up skills:

- `extract-xpu-kernel-asm` + `asm-source-mapping`: localize hot instructions to
  `src/ATen/native/xpu/sycl/` file:line.
- `eu-stall-attribution`: if the remaining gap after source edits is stall.
- `eu-tlp-occupancy`: if the kernel later becomes latency-bound.

This skill does **not** reduce instruction count by moving work to overlap with
other pipes (that is `eu-ilp-coissue`). It reduces the absolute number of
instructions executed per output element.

## Input contract

The skill expects these artifacts (ask the user if any are missing):

| Input | Source | Why needed |
|-------|--------|------------|
| `repro-cmd` | user / upstream skill | rerun unitrace |
| `kernel-name` | unitrace `== L0 Backend ==` | target the right kernel |
| `platform` / `device` | user | XVE count, peak DRAM BW |
| `computebasic.csv` | `unitrace -g ComputeBasic` | memory bytes, timing |
| `vectorengine.csv` | `unitrace -g VectorEngineProfile` | instruction mix per pipe |

## When to use

- `kernel-perf-analysis` reports `T_compute` is close to or above `T_mem`
  (rule of thumb: `T_compute ≥ 0.8 × T_mem`) and the dominant term is ALU0 or
  ALU1.
- `eu-utilization-triage` shows high single-pipe active fraction and the busy
  pipe is ALU0 or ALU1.
- `XVE_INST_EXECUTED_ALU0_ALL` or `XVE_INST_EXECUTED_ALU1_ALL` (plus MATH) is
  large relative to SEND and CONTROL.
- DRAM bandwidth utilization is well below platform peak.
- The kernel is hand-written SYCL with source access.

## When NOT to use

- `T_mem` strongly dominates `T_compute` (e.g., `T_compute < 0.8 × T_mem`)
  **and** DRAM BW utilization is pushing platform peak (> 80%). Use
  `memory-roofline-breakdown`.

  Compute and memory do not overlap perfectly. A small `T_mem` advantage
  (`0.8–1.0 × T_compute`) is not enough to declare pure memory-bound: reducing
  instruction time can still improve kernel time and may expose better
  memory-latency hiding.
- The kernel is GEMM-like or DPAS-heavy. Use `gemm-tiling-analysis`.
- The dominant deficit is `XVE_STALL`. Use `eu-stall-attribution` first.
- Occupancy is critically low. Use `eu-tlp-occupancy` first.

## Background: XVE five-pipe model

| Pipe | Sub-pipe | Width | Operations | Unitrace counter |
|------|----------|-------|------------|------------------|
| ALU0 | FP16 | SIMD32 | half-precision elementwise | `XVE_INST_EXECUTED_FP16` |
| ALU0 | FP32 | SIMD16 | single-precision add/mul/mad/fma | `XVE_INST_EXECUTED_FP32` |
| ALU0 | BF16 | SIMD16 | bfloat16 elementwise | `BITCONV` / tensor path |
| ALU1 | INT8 / INT16 | SIMD32 | small-integer logic | `XVE_INST_EXECUTED_INT16` |
| ALU1 | INT32 | SIMD16 | index arith / shift / mul / logic | `XVE_INST_EXECUTED_INT32`, inside `ALU1_ALL` |
| ALU1 | INT64 | SIMD8 | 64-bit address math | `XVE_INST_EXECUTED_INT64` |
| ALU1 | FP64 | SIMD1 | double precision | `XVE_INST_EXECUTED_FP64` |
| ALU1 | MATH | SIMD4 | sqrt / inv / idiv (f32) | `XVE_INST_EXECUTED_MATH` |
| SEND | - | variable | global / SLM load/store | `XVE_INST_EXECUTED_SEND_ALL` |
| CONTROL/JUMP | - | - | branch / jump | `XVE_INST_EXECUTED_CONTROL_ALL` |

ALU0 and ALU1 are two separate decoder pipes. ALU0 carries FP8/FP16/FP32/BF16;
ALU1 carries integer and MATH. FP64 and INT64 sit on ALU1 and execute at very
low SIMD widths (SIMD1/SIMD8), so even small source uses of `double` or
`int64_t` can become a bottleneck.

For the unitrace `XVE_INST_EXECUTED_*` counters used below, treat each event as
one lane-slot (per-SIMD-lane executed operation). This matches the measured
kernel durations when dividing by `XVE_count × freq_mhz`:

```
T_ALU0 = ALU0_ALL / (XVE_count * freq_mhz)
T_ALU1 = (ALU1_ALL + MATH) / (XVE_count * freq_mhz)
T_SEND = SEND_ALL / (XVE_count * freq_mhz)
T_CTRL = CONTROL_ALL / (XVE_count * freq_mhz)
T_compute = max(T_ALU0, T_ALU1, T_SEND, T_CTRL)
T_mem = (GPU_MEMORY_BYTE_READ + GPU_MEMORY_BYTE_WRITE) / peak_dram_bw
```

Use a practical overlap threshold `α` (default `0.8`) instead of an ideal
`T_compute ≥ T_mem` boundary. Compute and memory do not overlap perfectly on
real kernels, so reducing instruction time is still useful when compute is
within roughly 80% of memory time:

```
if T_compute >= 0.8 * T_mem  and  (T_ALU0 or T_ALU1 is the max):
    apply this skill
else:
    dispatch to memory-roofline-breakdown
```

## Step 0: Collect counters

```bash
source /opt/intel/oneapi/2026.0/oneapi-vars.sh
echo 0 > /proc/sys/dev/xe/observation_paranoid
# example for B580; substitute correct max clock
sudo sh -c 'echo 2850 > /sys/class/drm/card0/gt_min_freq_mhz; echo 2850 > /sys/class/drm/card0/gt_max_freq_mhz'

unitrace -q -i 20 -g ComputeBasic    <repro_cmd> > computebasic.csv 2>&1
unitrace -q -i 20 -g VectorEngineProfile <repro_cmd> > vectorengine.csv 2>&1
```

## Step 1: Per-pipe roofline

Use the helper script bundled with this skill:

```bash
python .claude/skills/kernel-perf-instruction-bound/scripts/instruction-bound-roofline.py \
    --compute-basic computebasic.csv \
    --vector-engine vectorengine.csv \
    --kernel <KernelName> \
    --xves 160 \
    --peak-bw 456
```

The script prints `T_ALU0`, `T_ALU1_serial`, `T_SEND`, `T_CONTROL`, `T_mem`,
`T_compute`, and the bound verdict.

Continue when the verdict is **ALU0 instruction-bound** or
**ALU1 instruction-bound**. The helper script uses a `0.8×` overlap threshold
(`T_compute ≥ 0.8 × T_mem`) rather than the ideal `1.0×`. If the kernel is
classified as memory-bound, dispatch to `memory-roofline-breakdown`.

## Step 2: Extract ASM and map hot IPs to source

```text
extract-xpu-kernel-asm  ──►  asm-file + kernel-name
asm-source-mapping      ──►  hot IP → file:line
```

ASM signatures and their likely source causes:

| ASM signature | Busy pipe | Likely source cause |
|---------------|-----------|---------------------|
| dense `dpas` or matrix ops | ALU0 (FP32/XMX) | GEMM-like; route to `gemm-tiling-analysis` |
| `mul/add/fma` on `:f` | ALU0 (FP32) | elementwise math; look for redundant ops |
| `math.*` / `inv` / `sqrt` | ALU0+ALU1 | transcendentals, runtime division |
| `math.inv` / `math.idiv` | ALU1 (MATH) | runtime `%` or `/` |
| long chains of `add/mul/shr/shl` on `:d` | ALU1 (INT32) | index decomposition |
| `BITCONV` before `math.inv` | ALU1 | mixed-type division |
| frequent `:q` arithmetic | ALU0 (INT64) | 64-bit pointer/index math |
| unvectorized d16 stores | memory | partial-write RFO; fix via vectorization |

## Step 3: Detect source patterns and propose levers

Open the mapped source file (usually `src/ATen/native/xpu/sycl/<Op>Kernels.cpp`).
Identify the dominant pipe from Step 1, then apply the matching sub-patterns.

### Universal patterns (reduce ALU0 + ALU1)

#### A. Multi-dimensional coordinate recovery from flat index

```cpp
int idx = item.get_global_id(0);
int n  = idx / (C * H * W);
int c  = (idx / (H * W)) % C;
int hw = idx % (H * W);
int h  = hw / W;
int w  = hw % W;
```

Each `%` and `/` by a runtime divisor becomes INT32 or MATH ops. If repeated in
the inner loop, it is kernel-scope redundant.

**Levers:**

1. Reparameterize the launch grid so work-items map directly to `(n, h, w)`,
   making `c` a vector-lane index.
2. Vectorize over `c` with `sycl::vec<half, N>`: one `n/h/w` index is reused
   across N channels.
3. Replace remaining runtime divides with PyTorch-style IntDivider:

```cpp
// host side, once per launch
uint32_t magic, shift; // or use at::native::DivMod / IntDivider utilities
// device side, branchless
div_res = (mul_hi(magic, x) + x) >> shift;
```

#### B. Address math inside the inner loop

```cpp
for (int ih = h_start; ih < h_end; ++ih) {
  for (int iw = w_start; iw < w_end; ++iw) {
    int offset = ((n * H + ih) * W + iw) * C + c;
    val += input[offset];
  }
}
```

Rewrite to a base pointer plus column/row steps:

```cpp
T* base = input + ((n * H + h_start) * W + w_start) * C + c;
for (...) {
  val += *base;
  base += C;                         // next pixel right
}
base += (W - pool_width) * C;        // next row down
```

This removes the inner-loop multiplies and divides entirely.

#### C. Launch-grid reparameterization

Whenever the kernel recovers `n/h/w` from a flat `global_id(0)`, consider
launching `{N, H, W}` work-items instead. The 2-D/3-D index is free from runtime
division.

### ALU0-specific patterns (FP32 / FP64 / INT64)

#### D. Unnecessary higher-precision arithmetic

Look for:

- `size_t` / `int64_t` indexing where all dimensions fit in 32 bits.
- Implicit `float64` or `double` temporaries.
- `Half`/`BFloat16` inputs converted to `float` for math that could stay in
  lower precision.

Example narrowing:

```cpp
// before: 64-bit multiply on ALU0
int64_t offset = ((n * H + ih) * W + iw) * C + c;

// after: 32-bit multiply; only final pointer add is 64-bit
int32_t offset = ((int32_t)n * H + ih) * W + iw;
T* ptr = base + (ptrdiff_t)(offset * C + c);
```

#### E. Redundant elementwise math

Look for code that recomputes the same scale/bias/offset per element when the
value is uniform across the work-item or work-group:

```cpp
// before: computed per element
T val = input[i] * scale + bias;

// after: if scale/bias are uniform, pass as kernel args or load once
```

Also look for duplicate calls to `std::exp`, `sycl::log`, `sycl::pow`, etc.

### ALU1-specific patterns (INT32 + MATH)

#### F. Runtime integer division / modulo

```cpp
T mean = sum / count;       // if count is runtime, this is a MATH op
int c_block = c / vec_size; // if not compile-time constant
```

- Precompute reciprocals on the host if the divisor is uniform.
- Use IntDivider for exact integer division.
- Choose grid/vector alignment so division by `vec_size` is unnecessary.

#### G. Complex bounds/index arithmetic

Repeated `(index % stride) / inner_size` or similar decompositions are common in
channels-last or strided kernels. Replace with precomputed multipliers or
coordinate arrays where legal.

## Step 4: Prioritized lever order

Apply in this order; remeasure after each change.

1. **Vectorize per work-item.** Usually the biggest win, especially for
   channels-last layouts. Reduces both ALU0 and ALU1 by amortizing index work.
2. **Hoist loop-invariant index/base-pointer math.** Removes repeated INT32/INT64
   ops.
3. **Replace runtime divides with IntDivider or host-side constants.** Targets
   `XVE_INST_EXECUTED_MATH`.
4. **Narrow 64-bit indices/pointers to 32-bit.** Targets `XVE_INST_EXECUTED_INT64`
   and indirect FP64 pressure.
5. **Reparameterize launch grid.** Removes flat-index decomposition entirely.
6. **Remove redundant elementwise math** (scale/bias, duplicated transcendentals).

## Step 5: Validate

Rebuild the kernel, then rerun:

```bash
unitrace -q -i 20 -g ComputeBasic    <repro_cmd> > cb_after.csv 2>&1
unitrace -q -i 20 -g VectorEngineProfile <repro_cmd> > ve_after.csv 2>&1

python .claude/skills/kernel-perf-instruction-bound/scripts/instruction-bound-roofline.py \
    --compute-basic cb_after.csv --vector-engine ve_after.csv \
    --kernel <KernelName> --xves 160 --peak-bw 456
```

Report a before/after table:

| Metric | Before | After | Goal |
|--------|--------|-------|------|
| T_actual | | | lower |
| T_ALU0 | | | lower if ALU0-bound |
| T_ALU1 | | | lower if ALU1-bound |
| ALU0_ALL | | | lower |
| ALU1_ALL + MATH | | | lower |
| MATH | | | ideally → 0 |
| INT64 | | | lower if safe |
| DRAM BW utilization | | | may rise when compute stops being the bottleneck |

Accept the change if:

- T_actual improves; OR
- The dominant pipe time decreases without regressing T_mem; OR
- T_actual / max(T_mem, T_compute_new) approaches ≤ 1.10×.

If the dominant pipe improves but T_actual does not, the kernel may have been
mis-classified or has another hidden bottleneck. Stop and rerun
`kernel-perf-analysis` / `eu-utilization-triage`.

## Output card

Emit a concise optimization card:

- Target kernel, repro command, platform.
- Per-pipe roofline: `T_mem`, `T_ALU0`, `T_ALU1`, `T_compute`,
  `T_actual/T_lower_bound`.
- Dominant pipe and source pattern(s) found, with file:line.
- Lever(s) applied (vectorize / IntDivider / hoist / narrow / grid / remove
  redundant math).
- Per-lever expected counter movement.
- Before/after table.
- Downstream skill if a new deficit appears.

## Example: AvgPool2d channels-last

B580, N=1024, C=768, H=W=17, 3×3 pool. The bottleneck shifts as instruction
reduction levers are applied:

| Case | T_actual | T_mem | T_ALU1_serial | Bound |
|------|----------|-------|---------------|-------|
| Scalar | 12.99 ms | 2.56 ms | 8.32 ms | ALU1 |
| Vec2 | 6.85 ms | 2.00 ms | 4.27 ms | ALU1 |
| Vec2+IntDiv | 5.75 ms | 2.00 ms | 3.30 ms | ALU1 |
| Vec8 | 2.42 ms | 2.03 ms | 1.18 ms | DRAM |
| Vec8+IntDiv | 2.36 ms | 2.03 ms | 0.93 ms | DRAM |

The same workflow applies to an ALU0-bound kernel, with the source-level focus on
dtype narrowing and redundant elementwise math instead of IntDivider.
