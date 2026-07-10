---
name: kernel-perf-analysis
description: Analyze one hotspot operator's target kernel with a source-code based roofline workflow. Use when asked for kernel performance analysis, roofline analysis, memory hierarchy analysis, unitrace counter interpretation, XPU kernel bottleneck classification, or PR-ready perf evidence for a single operator/kernel.
---

# Kernel Perf Analysis

Analyze one hotspot operator by identifying its target kernel, building a source-code based roofline model, comparing backend measurements against that model, and producing a top-down bottleneck verdict.

This Skill is intentionally scoped to **one hotspot operator and one target kernel**. If an operator dispatches multiple important kernels, run this Skill once per kernel.

## When to Use

Use this Skill when the user asks to:
- Analyze kernel performance for a hotspot operator.
- Build a source-code based roofline model.
- Explain memory hierarchy behavior from counters.
- Interpret unitrace or backend profiler metrics for one kernel.
- Produce PR-ready performance evidence for one optimized or regressed operator.

Do not use this Skill for fleet-level model performance analysis, broad benchmark triage, or multi-model CUDA-vs-XPU comparisons. Use a fleet or OOB performance workflow for those.

## Required Inputs

Before starting, identify these four items:

| Input | Meaning |
|---|---|
| Hotspot operator | Operator or fused operator selected from profiler evidence |
| Target kernel | Runtime kernel that implements the hotspot operator |
| Reproducer | Minimal command or script that launches the target kernel |
| Backend evidence | Profiler/counter artifacts for the target kernel only |

If the target kernel is ambiguous, first collect or inspect profiler output to map the hotspot operator to the runtime kernel name.

## Device-Agnostic Workflow

### Step 0: Pick One Hotspot Operator

Start from one hotspot operator. The unit of analysis is one target kernel.

Rules:
- Exclude helper kernels such as random init, copy, flush, memset, or warmup unless one of them is the selected hotspot.
- If one operator dispatches multiple performance-relevant kernels, analyze them one at a time.
- Do not widen the report into a whole workload or subgraph analysis.

### Step 1: Confirm Target Kernel and Launch Grid

Confirm that the measured kernel matches the hotspot operator and that its launch grid matches source expectations.

Collect:
- Runtime kernel name.
- Execution width, such as SIMD/subgroup/warp/wavefront width when exposed.
- Global grid.
- Local block/workgroup shape.
- Workgroup/block count.
- Kernel time from the target kernel only.

Minimum pass criteria:
- Runtime kernel name maps to the intended source or generated kernel.
- Launch grid matches the source launch construction.
- Timing and counters are for the target kernel, not helper kernels.

### Step 2: Build the Source-Code Based Roofline Model

Read the source or generated code for the matched kernel. Split it into phases, then derive per-call roofline quantities.

| Phase | What to extract |
|---|---|
| Data load / staging | Global reads, cache reads, vector width, alignment, local/shared memory use |
| Reduction / statistics | Reductions, subgroup operations, barriers, atomics, local memory use |
| Main compute | FLOPs, special math, vector/tensor instructions, integer address math |
| Writeback / epilogue | Output stores, parameter reads, partial writes, format conversions |
| Synchronization | Barriers, subgroup reductions, memory fences, atomic operations |

Derive these quantities when applicable:

| Quantity | Meaning |
|---|---|
| `input_bytes` | Required input tensor bytes |
| `output_bytes` | Required output tensor bytes |
| `parameter_bytes` | Weights, bias, scale, metadata, descriptors |
| `source_logical_read_bytes` | Element bytes read by the implementation before cacheline or transaction effects |
| `source_logical_write_bytes` | Element bytes written by the implementation before cacheline or transaction effects |
| `dram_read_bytes_optimistic` | Minimum global-memory read traffic under optimistic cache reuse |
| `dram_write_bytes_expected` | Expected global-memory writeback traffic when writeback is visible in the measured window |
| `shared_onchip_read_reuse_bytes_optimistic` | Multi-pass read bytes expected to be served by on-chip cache/storage after the first pass |
| `shared_onchip_write_bytes_expected` | Write-side on-chip traffic, if the implementation or counters expose it |
| `local_transaction_read_bytes_expected` | Expected local/on-chip read transaction bytes, if a transaction estimate is available |
| `local_transaction_write_bytes_expected` | Expected local/on-chip write transaction bytes, if a transaction estimate is available |
| `flops_or_ops` | Implementation-specific work estimate |

Keep read-side and write-side roofline quantities separate. Only compute a combined total-byte roofline after both read-side and write-side closure have been checked independently.

Use memory sharing domains first, then map them to backend-specific names:

| Abstract domain | Meaning |
|---|---|
| Per-thread / register storage | Values kept inside a lane/thread |
| Intra-core / block-local on-chip storage | Storage shared within a core, SM, CU, Xe core, workgroup, CTA, or block |
| Inter-core on-chip cache | Cache shared across cores/SMs/CUs/Xe cores within the device |
| Global memory / DRAM / HBM | Traffic reaching external or device-global memory |

### Step 3: Collect Backend Measurements

Aggregate only rows or events matching the target kernel. Weight percentage and rate counters by kernel time from the same measurement window.

Collect these groups when available:

| Measurement group | Meaning |
|---|---|
| Runtime and launch | Kernel time, call count, launch grid, frequency, device busy |
| Compute state | Stall, active, issue utilization, occupancy, pipe mix, vector/tensor-core utilization |
| Intra-core on-chip memory | Local/shared bytes or events, local-cache transactions, bank conflicts |
| Inter-core on-chip cache | Shared cache hits/misses/transactions/stalls |
| Global memory | Read bytes, write bytes, bandwidth, request queues, TLB/page faults |

Derived metrics:
- Active vs stall split.
- Co-issue or dual-issue ratio when exposed.
- Local/on-chip hit ratio when exposed.
- Global read bandwidth.
- Global write bandwidth, only after write counter semantics are understood.
- On-chip transaction bandwidth when byte counters exist.

### Step 4: Compare Roofline Model vs Counters

Use the source-code based roofline model as the expectation and backend counters as measured evidence.

| Question | Evidence |
|---|---|
| Does measured global-memory read traffic match the optimistic read-side roofline estimate? | Compare roofline global-read estimate vs backend global-read counter |
| Does measured global-memory write traffic match expected writeback? | Compare expected logical/global writes vs backend global-write counter; if it does not close, do not use total global-memory bytes for the final verdict |
| Does multi-pass read reuse look plausible? | Later-pass read bytes should not reappear as equivalent global-memory read bytes |
| Does local/on-chip transaction traffic match implementation shape? | Compare read and write transaction estimates separately vs backend local/on-chip transaction counters |
| Is global-memory saturation likely? | High global-memory bandwidth plus high request-queue pressure or high memory stalls |
| Is cache/reuse behavior dominant? | On-chip traffic much larger than global-memory traffic while global read is near the first-pass roofline estimate |
| Is compute/pipe utilization the limiter? | High active, low stall, low co-issue, or pipe imbalance |

Do not classify a kernel as memory-bound from global-read bytes alone. Do not classify from global-read plus global-write bytes unless write-side closure is understood.

### Step 5: Produce Top-Down Verdict and Evidence Level

Use this compact report structure:

| Layer | Required content |
|---|---|
| L0 Identity / launch | Target kernel, launch grid, call count, time share |
| L1 Compute state | Stall / active / co-issue / occupancy |
| L2 Pipe / execution mix | Scalar/vector/tensor pipe utilization, load-store pressure, issue profile |
| L3 Memory hierarchy | Intra-core on-chip, inter-core on-chip, global memory, queue, TLB/page behavior |
| L4 Verdict | Primary bottleneck, secondary bottleneck, next measurable action |

Assign evidence level:

| Level | Requirement | Confidence |
|---|---|---|
| L1 Name match | Runtime kernel name maps to a plausible source kernel | Low |
| L2 Launch match | Runtime launch grid matches source launch construction | Medium |
| L3 Counter match | Counter behavior matches source-code based roofline and instruction expectations | Medium-high |
| L4 ASM/source match | Hot ASM/IPs map back to expected source loops | High |

For final optimization claims, target L4. For triage reports, L2 or L3 is acceptable if limitations are explicit.

## XPU Implementation Mapping

Use this section when the backend is Intel XPU and the available profiler is unitrace `ComputeBasic`.

### XPU Target Kernel and Launch Grid

| Abstract field | XPU / ComputeBasic source |
|---|---|
| Runtime kernel name | unitrace CSV `Kernel` |
| Execution width | Kernel signature such as `SIMD32`, or ASM |
| Global grid | Kernel signature or source launch code |
| Local workgroup | Kernel signature or source launch code |
| Workgroup count | `GPGPU_THREADGROUP_COUNT` |
| Kernel time | `GpuTime[ns]` |
| Frequency / clocks | `GpuCoreClocks`, `AvgGpuCoreFrequencyMHz` |
| Device busy | `GPU_BUSY` |

### XPU Memory Hierarchy Mapping

| Device-agnostic domain | XPU interpretation | ComputeBasic visibility |
|---|---|---|
| Per-thread / register storage | GRF/register values and subgroup shuffle data | Usually not exposed as memory bytes; confirm with ASM when needed |
| Intra-Xe-core shared on-chip | Explicit SLM/local storage plus L1/LSC load-store path inside one Xe core | `SLM_*`, `LOAD_STORE_CACHE_*` |
| Inter-Xe-core shared on-chip | Cache shared across Xe cores, exposed as L3/device-cache behavior in this metric group | `L3_*` |
| Global memory / DRAM | Traffic reaching external/global memory | `GPU_MEMORY_*` |

Important interpretation rules:
- `LOAD_STORE_CACHE_*` is intra-Xe-core L1/LSC transaction behavior, not inter-Xe-core L3 bytes.
- `SLM_*` is explicit intra-Xe-core local/shared storage traffic.
- `L3_*` is the measured view of inter-Xe-core shared on-chip cache behavior.
- `GPU_MEMORY_*` is global-memory traffic. Read and write counters may have different closure quality.

### XPU ComputeBasic Counter Groups

| Analysis layer | ComputeBasic counters |
|---|---|
| Runtime | `GpuTime[ns]`, `GpuCoreClocks`, `AvgGpuCoreFrequencyMHz`, `GPU_BUSY` |
| Launch | `GPGPU_THREADGROUP_COUNT`, `ASYNC_GPGPU_THREADGROUP_COUNT`, dispatch counters |
| Compute state | `XVE_STALL`, `XVE_ACTIVE`, `XVE_MULTIPLE_PIPE_ACTIVE`, `XVE_THREADS_OCCUPANCY_ALL` |
| Pipe mix | `XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION`, `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION`, `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION`, `XVE_INST_EXECUTED_SEND_ALL`, `XVE_INST_ISSUED_ALL` |
| Intra-Xe-core SLM/local | `SLM_BYTE_READ`, `SLM_BYTE_WRITE`, `SLM_BANK_CONFLICT_COUNT` |
| Intra-Xe-core L1/LSC | `LOAD_STORE_CACHE_BYTE_READ`, `LOAD_STORE_CACHE_BYTE_WRITE`, `LOAD_STORE_CACHE_ACCESS`, `LOAD_STORE_CACHE_HIT`, `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` |
| Inter-Xe-core L3 | `L3_HIT`, `L3_MISS`, `L3_READ`, `L3_WRITE`, `L3_STALL`, `L3_ATOMIC_ACCESS` |
| DRAM | `GPU_MEMORY_BYTE_READ`, `GPU_MEMORY_BYTE_WRITE`, `GPU_MEMORY_BYTE_READ_RATE`, `GPU_MEMORY_BYTE_WRITE_RATE`, `GPU_MEMORY_REQUEST_QUEUE_FULL`, `TLB_MISS` |

Recommended XPU derived metrics:
- `single_pipe_active = XVE_ACTIVE - XVE_MULTIPLE_PIPE_ACTIVE`
- `co_issue_ratio = XVE_MULTIPLE_PIPE_ACTIVE / XVE_ACTIVE`
- `lsc_hit_ratio = LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS`
- `l3_hit_ratio = L3_HIT / (L3_HIT + L3_MISS)`
- `dram_read_bw = GPU_MEMORY_BYTE_READ / GpuTime`
- `dram_write_bw = GPU_MEMORY_BYTE_WRITE / GpuTime`, only after write-side counter semantics are understood
- `lsc_transaction_bw = (LOAD_STORE_CACHE_BYTE_READ + LOAD_STORE_CACHE_BYTE_WRITE) / GpuTime`

### XPU Evidence Closure

Map the generic evidence levels to XPU evidence:

| Evidence level | XPU-specific closure |
|---|---|
| L1 Name match | unitrace kernel name maps to the intended source functor or generated kernel |
| L2 Launch match | unitrace launch grid matches source launch construction |
| L3 Counter match | ComputeBasic behavior matches the source-code based roofline and instruction expectations |
| L4 ASM/source match | Extracted XPU ISA or stall-sampling hot IPs map back to expected source loops |

For L4 evidence, use the repository's ASM extraction and ASM/source mapping skills when available.

## Output Format

Produce a concise analysis suitable for pasting into an issue or PR description:

```markdown
## Kernel Perf Analysis

### Target
- Hotspot operator:
- Target kernel:
- Launch grid:
- Calls / time:

### Source-Code Based Roofline
| Quantity | Read/Write | Estimate | Notes |
|---|---|---:|---|

### Measurement Snapshot
| Layer | Metric | Value | Interpretation |
|---|---|---:|---|

### Roofline Model vs Measurement
| Path | Roofline estimate | Measured | Closure |
|---|---:|---:|---|

### Top-Down Verdict
| Layer | Observation | Verdict |
|---|---|---|

### Evidence Level
- Current level:
- Missing evidence:
- Next measurable action:
```

## Common Pitfalls

- Do not use operator-level bytes when the source implementation performs multiple passes.
- Do not merge read and write traffic until each side closes independently.
- Do not interpret local/on-chip transaction counters as global-memory bytes.
- Do not treat helper kernels as part of the selected hotspot kernel.
- Do not claim final optimization root cause without L4 ASM/source evidence when the conclusion depends on instruction-level behavior.

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: kernel-perf-analysis.
