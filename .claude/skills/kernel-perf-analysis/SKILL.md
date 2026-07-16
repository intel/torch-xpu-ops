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
| Level 0 Identity / launch | Target kernel, launch grid, call count, time share |
| Level 1 Compute state | Stall / active / co-issue / occupancy |
| Level 2 Pipe / execution mix | Scalar/vector/tensor pipe utilization, load-store pressure, issue profile |
| Level 3 Memory hierarchy | Intra-core on-chip, inter-core on-chip, global memory, queue, TLB/page behavior |
| Level 4 Verdict | Primary bottleneck, secondary bottleneck, next measurable action |

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

Produce a concise, metric-first analysis suitable for pasting into an issue or PR description. Every bottleneck claim must be backed by at least one numeric counter, source-derived estimate, or artifact path.

Rules:
- Prefer tables with concrete values over prose-only conclusions.
- Include units for every numeric value, such as `us`, `ns`, `%`, `GB/s`, `bytes/call`, `GFLOP/call`, `workgroups`, or `calls`.
- For byte counters, keep the raw byte value in `Metric value`, but present `Measured per call` in a consistent human-readable unit, preferably `MB/call` or `GB/call`. Use decimal units consistently inside one report: `1 MB = 1e6 bytes`, `1 GB = 1e9 bytes`, unless the source model explicitly uses `MiB`.
- For percentage or rate counters aggregated across multiple calls, state that they are time-weighted averages.
- For byte/event counters aggregated across multiple calls, include both total and per-call values when useful.
- For each primary bottleneck, include the measured metric, the comparison point or threshold, and the next measurement or experiment that can confirm it.
- If a required metric is unavailable or unreliable, write `N/A` and explain the counter limitation in one sentence.
- For achieved TFLOPS, use target-kernel profiler time, such as `GpuTime[ns]`, as the primary timing source. Wall-clock or benchmark-table time is secondary context because profiler collection and Python/benchmark overhead can make it misleading.

````markdown
## Kernel Perf Analysis

### Target
- Hotspot operator:
- Reproducer:
- Evidence artifacts:
- Target kernel(s):

| Kernel role | Runtime kernel name / signature | Source mapping | Calls | Total time | Avg time/call | Time share | Launch grid / WG count |
|---|---|---|---:|---:|---:|---:|---|

Notes:
- State whether the report covers one kernel or multiple kernels from one operator.
- If helper kernels are excluded, list the exclusion rule briefly.
- If source-derived launch grid is used because profiler WG count is unavailable or zero, say so explicitly.

### Source-Code Based Roofline
| Phase | Source location | Work / traffic formula | Estimate per call | Key implementation detail |
|---|---|---|---:|---|

| Quantity | Direction / domain | Estimate per call | Formula / assumption | Closure expectation |
|---|---|---:|---|---|
| FLOPs or ops | compute |  |  |  |
| Required input bytes | global read |  |  |  |
| Required output bytes | global write |  |  |  |
| Source logical read bytes | logical read |  |  |  |
| Source logical write bytes | logical write |  |  |  |
| Optimistic DRAM read bytes | DRAM read |  |  |  |
| Expected DRAM write bytes | DRAM write |  |  |  |
| Expected on-chip / LSC bytes | on-chip read/write |  |  |  |

Derived ratios:
- Arithmetic intensity from optimistic DRAM traffic:
- Arithmetic intensity from measured DRAM traffic:
- On-chip amplification, if visible: `measured_onchip_bytes / measured_dram_bytes`

### Measurement Snapshot
State the aggregation method first:
- Call count:
- Percent/rate aggregation: time-weighted average
- Byte/event aggregation: sum across matching target-kernel rows

| Layer | Metric | Value | Derived / comparison | Interpretation |
|---|---|---:|---:|---|
| Runtime | `GpuTime[ns]` / avg time |  |  |  |
| Runtime | `GPU_BUSY[%]` |  |  |  |
| Compute state | `XVE_ACTIVE[%]` |  |  |  |
| Compute state | `XVE_STALL[%]` |  |  |  |
| Compute state | `XVE_MULTIPLE_PIPE_ACTIVE[%]` |  | `co_issue_ratio = multiple / active` |  |
| Compute state | `XVE_THREADS_OCCUPANCY_ALL[%]` |  |  |  |
| Pipe mix | ALU0 / ALU1 / ALU2 utilization |  |  |  |
| Pipe mix | `XVE_INST_EXECUTED_SEND_ALL` / `XVE_INST_ISSUED_ALL` |  | send density |  |
| L1 / LSC | `LOAD_STORE_CACHE_BYTE_READ/WRITE` |  | per-call bytes, hit ratio |  |
| L3 | `L3_HIT`, `L3_MISS`, `L3_STALL[%]` |  | `l3_hit_ratio` |  |
| DRAM | `GPU_MEMORY_BYTE_READ/WRITE` |  | per-call bytes, GB/s |  |
| DRAM | `GPU_MEMORY_REQUEST_QUEUE_FULL[%]` |  |  |  |
| Addressing | `TLB_MISS` |  | per-call misses |  |

Optional ComputeBasic sanity metrics, when available:

| Category | Metrics | Why record it |
|---|---|---|
| Quality / frequency | `ResultUncertainty[%]`, `GpuCoreClocks[cycles]`, `AvgGpuCoreFrequencyMHz[MHz]`, `CoreFrequencyMHz[MHz]` | Validate measurement quality and detect frequency drift or throttling |
| Instruction cache | `ICACHE_HIT[events]`, `ICACHE_MISS[events]` | Support or rule out instruction-fetch pressure |
| Finer pipe overlap | `XVE_PIPE_ALU0_AND_ALU1_ACTIVE[%]`, `XVE_PIPE_ALU0_AND_ALU2_ACTIVE[%]`, ALU0/ALU1/ALU2 event counts | Explain which pipes co-issue, beyond aggregate `XVE_MULTIPLE_PIPE_ACTIVE` |
| Shared function hold | `XVE_SHARED_FUNCTION_ACCESS_HOLD[%]` | Detect shared-function or special-function unit holds |
| SLM | `SLM_BYTE_READ[bytes]`, `SLM_BYTE_WRITE[bytes]`, `SLM_BANK_CONFLICT_COUNT[events]` | Separate explicit SLM traffic and bank conflicts from LSC/cache traffic |
| L3 atomic | `L3_ATOMIC_ACCESS[events]` | Detect atomic/cache serialization |
| Dispatch / frontend | `GPGPU_DISPATCH[%]`, `COMMAND_PARSER_COMPUTE_ENGINE_BUSY[%]`, `COMMAND_PARSER_COMPUTE_ENGINE_DISPATCH_KERNEL_COUNT[events]`, `COMMAND_PARSER_FLUSH_COUNT[events]`, `ASYNC_GPGPU_THREAD_EXIT_COUNT[messages]` | Detect dispatch, queue, flush, or async-exit overhead |
| Engine interference | `COMMAND_PARSER_COPY_ENGINE_BUSY[%]`, `COMMAND_PARSER_RENDER_ENGINE_BUSY[%]`, `COMMAND_PARSER_RENDER_ENGINE_DISPATCH_KERNEL_COUNT[events]` | Rule out copy/render engine interference |
| External/system traffic | `HOST_TO_GPUMEM_TRANSACTION_READ/WRITE[events]`, `SYSMEM_TRANSACTION_READ/WRITE[events]` | Detect host or system-memory traffic contaminating the kernel window |
| Graphics/compression sanity | `IA_VERTEX[events]`, `RASTERIZER_SAMPLE_OUTPUT[events]`, `COMPRESSOR_INPUT[events]`, `COMPRESSOR_OUTPUT[events]` | Usually expected to be zero for compute kernels; nonzero values need explanation |

Do not put optional sanity metrics in the primary bottleneck claim unless they are nonzero or directly explain the observed bottleneck. Otherwise, report them in one compact optional table or state that they were checked and were not material.

### Roofline Model vs Measurement
This table must make the raw metric-to-per-call derivation explicit. `Metric value` is the raw aggregated counter or profiler value used to derive `Measured per call`.

| Path | Source estimate per call | Measured per call | Ratio | Metric value | Closure verdict |
|---|---:|---:|---:|---|---|
| FLOPs |  |  | achieved / expected or `N/A` | source FLOPs formula; target-kernel `GpuTime[ns] = <sum>; calls = <N>; kernel_time = sum / N`; wall-clock benchmark time only as secondary context if useful |  |
| LSC Read |  |  | measured / estimate or `N/A` | `LOAD_STORE_CACHE_BYTE_READ[bytes] = <sum>; calls = <N>; measured = sum / N`; include `LOAD_STORE_CACHE_ACCESS`, `LOAD_STORE_CACHE_HIT`, `lsc_hit_ratio`; include partial-read counter if backend exposes one |  |
| LSC Write |  |  | measured / estimate or `N/A` | `LOAD_STORE_CACHE_BYTE_WRITE[bytes] = <sum>; calls = <N>; measured = sum / N`; include `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` or backend partial-write counter if available |  |
| L3 Read |  |  | measured / estimate or `N/A` | `L3_READ[events]` and/or `GPU_MEMORY_L3_READ[events]`; include `L3_HIT`, `L3_MISS`, `l3_hit_ratio`, `L3_STALL[%]`; include event-to-byte assumption only if converting |  |
| L3 Write |  |  | measured / estimate or `N/A` | `L3_WRITE[events]` and/or `GPU_MEMORY_L3_WRITE[events]`; include `L3_HIT`, `L3_MISS`, `l3_hit_ratio`, `L3_STALL[%]`; include event-to-byte assumption only if converting |  |
| Global Memory Read |  |  | measured / estimate | `GPU_MEMORY_BYTE_READ[bytes] = <sum>; calls = <N>; measured = sum / N`; include `GPU_MEMORY_BYTE_READ_RATE[GBpS]`, `GPU_MEMORY_REQUEST_QUEUE_FULL[%]`, `TLB_MISS` |  |
| Global Memory Write |  |  | measured / estimate | `GPU_MEMORY_BYTE_WRITE[bytes] = <sum>; calls = <N>; measured = sum / N`; include `GPU_MEMORY_BYTE_WRITE_RATE[GBpS]`, `GPU_MEMORY_REQUEST_QUEUE_FULL[%]`, partial-write/write-combine symptoms if exposed |  |

Rules for this table:
- Do not hide the raw counter. `Metric value` must include the exact counter name, summed value, call count, and formula used for `Measured per call`.
- `Metric value` should be a compact counter bundle for that memory layer, not just the byte counter. Include hit/miss, hit ratio, partial read/write, stall, queue, and TLB counters when the backend exposes them.
- Use readable normalized units such as `MB/call` or `GB/call` for byte-counter `Measured per call`, while preserving raw `bytes` in `Metric value`. Use `events/call` for event counters unless a documented event-to-byte conversion is available.
- For TFLOPS, use target-kernel time by default: `tflops_kernel = source_flops_per_call / kernel_avg_seconds / 1e12`. Do not use wall-clock benchmark time for the primary TFLOPS when profiler collection overhead is present.
- For XPU ComputeBasic, include `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` for LSC write closure. There may be no partial-read counter; do not invent one. If another backend exposes partial reads, include it in the same row.
- For cache rows, always include hit-ratio context when available: `lsc_hit_ratio = LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS`; `l3_hit_ratio = L3_HIT / (L3_HIT + L3_MISS)`.
- For `L3_READ` and `L3_WRITE`, do not convert events to bytes unless the backend documentation or profiler output provides the transaction size. If no conversion is valid, keep `Measured per call` in `events/call` and set byte-ratio closure to `N/A`.
- For `FLOPs`, source estimate is usually source-derived work. `Metric value` must name the target-kernel timing source used for primary achieved throughput, usually profiler `GpuTime[ns]`. Wall-clock benchmark `ms` may be listed only as secondary context when it differs.
- If one operator launches multiple relevant kernels, either provide one table per kernel or add a `Kernel` column. Do not mix counters from different runtime kernels in one per-call row unless the row explicitly says it is operator-level aggregate.

Required interpretation bullets:
- DRAM-bound check: include DRAM bandwidth, request queue pressure, and whether measured DRAM bytes close with the source estimate.
- Cache/on-chip pressure check: include LSC bytes, LSC hit ratio, L3 hit ratio, and L3 stall.
- Compute/issue check: include active, stall, co-issue ratio, occupancy, and ALU/SEND mix.

### Top-Down Verdict
| Layer | Metrics that decide this layer | Threshold / comparison | Verdict | Next action |
|---|---|---|---|---|
| Level 0 Identity / launch | kernel name, calls, avg time, grid/WG count | source launch matches? |  |  |
| Level 1 Compute state | active, stall, occupancy, GPU busy | high stall? low active? low occupancy? |  |  |
| Level 2 Pipe / execution mix | co-issue ratio, ALU util, SEND/issued | pipe imbalance or low co-issue? |  |  |
| Level 3 Memory hierarchy | LSC bytes/hit, L3 hit/miss/stall, DRAM bytes/BW, queue full, TLB | DRAM saturated or on-chip/cache pressure? |  |  |
| Level 4 Root cause status | ASM/source mapping or missing evidence | ASM/source evidence available? |  |  |

### Optimization Candidates
| Priority | Candidate | Evidence | Expected headroom | Validation experiment | Stop / rollback criterion |
|---:|---|---|---:|---|---|

Guidance:
- `Evidence` must name the deciding counters or source lines.
- `Expected headroom` should be a bounded estimate, such as `up to 32% of FMHA device time` or `unknown until tile A/B`.
- `Validation experiment` must be a rerunnable command, source knob, or metric group to collect next.

### Reproducibility
```bash
# Command used to collect the evidence
```

Expected key output:
```text
# Include the key timing/counter lines that should reappear
```

### Evidence Level
- Current level:
- Evidence that supports this level:
- Missing evidence for the next level:
- Next measurable action:
````

## Common Pitfalls

- Do not use operator-level bytes when the source implementation performs multiple passes.
- Do not merge read and write traffic until each side closes independently.
- Do not interpret local/on-chip transaction counters as global-memory bytes.
- Do not treat helper kernels as part of the selected hotspot kernel.
- Do not claim final optimization root cause without L4 ASM/source evidence when the conclusion depends on instruction-level behavior.

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: kernel-perf-analysis.
