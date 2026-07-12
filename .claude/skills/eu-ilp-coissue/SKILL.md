---
name: eu-ilp-coissue
description: Analyze Intel GPU EU instruction-level parallelism and co-issue for one XPU target kernel. Use when stall is not dominant but XVE_MULTIPLE_PIPE_ACTIVE is low, ALU pipe utilization is imbalanced, or ASM evidence is needed to find interleave opportunities.
---

# EU ILP and Co-Issue Analysis

Analyze low co-issue in one XPU target kernel and identify whether instruction-level parallelism can raise execution from mostly single-pipe active cycles to multi-pipe active cycles.

This skill is for the S1-to-S2 problem: the EU is active, but too much active time uses only one physical pipe.

Co-issue is the measurable result; ILP is the source-code technique. The EU issues one instruction at a time, but multi-cycle execution windows from adjacent instructions can overlap on different execution pipes. The goal is to move independent source-level work so the hot loop has useful instructions for more than one pipe in flight.

## When to Use

Use this skill when:
- `XVE_STALL` is not the dominant deficit.
- `XVE_MULTIPLE_PIPE_ACTIVE` or co-issue share is low.
- Per-pipe ALU utilization shows an idle pipe with plausible work to overlap.
- `eu-utilization-triage` routes to ILP/co-issue.
- `eu-stall-attribution` identifies Stage-1 dependency distance and routes here with ASM/source evidence.

Do not use this skill when S0/stall dominates; run `eu-stall-attribution` first. Do not use it when the idle pipe has no independent work available.

## Inputs

| Input | Meaning |
|---|---|
| `repro-cmd` | Command that launches the target kernel |
| `kernel-filter` | Target kernel filter |
| `ComputeBasic` counters | Co-issue, active, and per-pipe utilization counters |
| `asm-dir` | Optional existing ASM artifacts |
| `ip-source-table` | Optional source mapping from `eu-stall-attribution` |

## Workflow

### Step 1: Measure Co-Issue and Pipe Utilization

Collect or read `ComputeBasic` for the target kernel.

Key metrics:
- `XVE_ACTIVE`
- `XVE_MULTIPLE_PIPE_ACTIVE`
- `XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION`
- `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION`
- `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION`
- SEND and issue counters when available

Derived values:

```text
single_pipe_active = XVE_ACTIVE - XVE_MULTIPLE_PIPE_ACTIVE
coissue_share = XVE_MULTIPLE_PIPE_ACTIVE / XVE_ACTIVE
```

Identify the idle or underused pipe.

### Step 2: Decide Whether ILP Headroom Exists

| Observation | Meaning |
|---|---|
| S0 high | Not an ILP-first case; run stall attribution |
| S1 high and S2 low | ILP/co-issue headroom likely exists |
| One pipe busy, another pipe low | Candidate for interleaving independent work |
| All useful pipes already balanced | ILP headroom may be small |

Stop if co-issue is already healthy or if the candidate work is truly serial.

### Step 3: Inspect ASM for Interleave Opportunities

If an `ip-source-table` from `eu-stall-attribution` exists, reuse it. Otherwise extract ASM with `extract-xpu-kernel-asm` and map hot regions with `asm-source-mapping` when source-level evidence is needed.

Classify hot-loop instruction regions:

| Region | Typical pipe | What to look for |
|---|---|---|
| FP ALU | ALU0 | `mul`, `mad`, FP convert, special math |
| INT / address | ALU1 | address math, loop/index arithmetic, send setup |
| DPAS / tensor | ALU2 | `dpas` or matrix instructions |
| SEND | load/store path | memory operations and latency windows |

Look for serialization patterns:
- Long DPAS island with no independent FP/INT work interleaved.
- Long FP burst followed by DPAS burst, with no overlap.
- Address math performed just-in-time before SEND instead of ahead of time.
- Epilogue conversion/store chain that leaves tensor/INT pipes idle.

### Step 4: Recommend a Concrete Interleave Pattern

Choose a source-code pattern from the ASM evidence. The fix must name the source loop or epilogue to change, not only the pipe that is idle.

| Evidence | Source-code pattern | Before | After | Expected counter movement |
|---|---|---|---|---|
| Long FP `mul/add/math.*` burst, then matrix/tensor work | Rescale-GEMM interleave | Rescale a whole tile, then run the matrix tile | Split rescale into smaller blocks and run each block next to the corresponding matrix tile | Co-issue share rises because FP work overlaps matrix-pipe work |
| Long matrix/tensor chain with no independent FP or INT work nearby | Deferred epilogue/address preparation | Finish the whole matrix chain before preparing epilogue, addresses, or next-tile metadata | Move independent epilogue prep, address prep, or next-tile metadata between matrix tiles without breaking true dependencies | Idle FP/INT pipe utilization improves while matrix utilization stays stable |
| `mov :bf` / convert plus store chain at loop end | Epilogue split store | Convert all outputs, then store, or store rows serially with no overlapped prep | Split convert/store into chunks and overlap next address/prep work with current convert/store sequence | Epilogue single-pipe region shrinks; check SEND pressure and write-side counters |
| Address math appears immediately before each load/store SEND | Address pre-computation | Compute `ptr + offset` just before SEND, then wait for data | Precompute next addresses during current independent math or matrix work; issue SEND earlier when legal | ALU1 address work and SEND latency overlap with useful work; Stage-1/SBID waits may fall |
| Math function loop has repeated same-operation waits | Loop unroll for cross-iteration ILP | Issue one expensive math op, then wait for its result before the next independent element | Unroll independent elements so several math ops are in flight on different registers | Distance stalls fall; math pipe utilization and co-issue opportunity improve |
| Softmax/math phase and following matrix phase are strictly sequential | Math-matrix interleave | Finish all `exp/mul/reduce` work before starting the next matrix accumulation | Start matrix work for completed rows/blocks while later rows/blocks continue math work | MATH/FP work overlaps matrix-pipe work; total phase time can shrink |

Do not break a hardware-optimized matrix chain blindly. The goal is to place independent work around it, not to destroy useful tensor-pipe scheduling. If the candidate work has a true data dependency on the chain result, report that ILP is not available and route to algorithm restructuring or TLP analysis instead.

### Step 5: Emit ILP Card

```markdown
## ILP / Co-Issue Analysis: <kernel>

### Problem
- coissue-share:
- single-pipe active:
- idle pipe:
- busy pipe:

### Counter Evidence
| Metric | Value | Interpretation |
|---|---:|---|

### ASM Evidence
| Region | ASM lines | Instruction pattern | Pipe |
|---|---:|---|---|

### Interleave Opportunity
- pattern:
- source location:
- why it can overlap:
- risk: register pressure / spill / changed memory behavior

### Recommendation
- source change:
- expected counter movement:
- validation command:
```

### Step 6: Validate

After a source change:
- Re-run the benchmark and compare kernel time.
- Re-collect `ComputeBasic`.
- Confirm `XVE_MULTIPLE_PIPE_ACTIVE` increased.
- Confirm the idle pipe utilization improved.
- Re-extract ASM when needed to verify the intended interleaving is present.
- Check that GRF spill, SEND pressure, or memory traffic did not regress enough to erase the gain.

## Pitfalls

- Low co-issue is not automatically bad if the kernel is memory-bound or stall-dominated.
- Do not infer source cause from counters alone; use ASM for instruction scheduling claims.
- Do not claim DPAS chains are wrong without identifying independent work that can safely overlap.
- Watch register pressure: added ILP can increase GRF use and reduce occupancy.

Custom skills applied: eu-ilp-coissue.
