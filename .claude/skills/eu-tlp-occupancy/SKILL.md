---
name: eu-tlp-occupancy
description: Decide whether more Xe2 thread-level parallelism will help one XPU target kernel. Use when occupancy is low or moderate, Stage-1 stalls suggest candidate starvation, or a proposed work-group/vectorization change may trade occupancy against register pressure and Stage-2 contention.
---

# EU TLP and Occupancy Analysis

Decide whether increasing thread-level parallelism can improve a target XPU kernel, and identify when it would instead worsen resource contention.

This skill answers one question: will more resident work hide latency, or will it add pressure to already-contended pipes, SENDs, GRF, SLM, or memory hierarchy?

## When to Use

Use this skill when:
- `XVE_THREADS_OCCUPANCY_ALL` is low or moderate.
- `eu-utilization-triage` routes to occupancy/TLP.
- `eu-stall-attribution` identifies Stage-1 candidate starvation.
- A kernel tuning proposal changes work-group size, SIMD width, tile shape, unroll, vector width, or register pressure.
- You need to decide whether adding threads is likely to help before editing code.

Do not use this skill as a generic fix for high stall. If the dominant stall is Stage-2 Pipe or Send contention, more TLP can make the bottleneck worse.

## Inputs

| Input | Meaning |
|---|---|
| `repro-cmd` | Command that launches the target kernel |
| `kernel-filter` | Target kernel filter |
| `ComputeBasic` counters | Occupancy, S0/S1/S2, per-pipe utilization |
| Stall attribution | Stage-1 vs Stage-2 classification when available |
| Kernel config | Work-group size, SIMD width, tile shape, local memory use, vectorization |
| ASM summary | GRF pressure, spills, long dependency chains, SEND density when available |

## Workflow

### Step 1: Record Current Occupancy and EU State

Collect or reuse:
- `XVE_THREADS_OCCUPANCY_ALL`
- `XVE_STALL`
- `XVE_ACTIVE`
- `XVE_MULTIPLE_PIPE_ACTIVE`
- Per-pipe ALU utilization
- SEND or memory pressure counters when available

Emit the current portrait:

| Metric | Value |
|---|---:|
| occupancy | |
| S0 stall | |
| S1 single-pipe | |
| S2 co-issue | |
| dominant stall class | |

### Step 2: Classify the Occupancy Regime

Use platform-specific thresholds when available. Otherwise use qualitative buckets:

| Regime | Meaning |
|---|---|
| Low occupancy | Many EUs have too few resident threads; TLP may hide latency |
| Moderate occupancy | TLP may help if stalls are Stage-1 and resource pressure is not high |
| High occupancy | More threads are unlikely to help; focus on ILP, memory, or algorithmic changes |

### Step 3: Classify Stalls by Stage

Use `eu-stall-attribution` output when available.

| Stall stage | TLP implication |
|---|---|
| Stage-1 candidate starvation | More resident independent work may help |
| Stage-2 Pipe contention | More resident work may worsen pipe contention |
| Stage-2 Send contention | More resident work may worsen memory/SEND contention |
| Sync or Control | More resident work may not fix the root cause |

If stall stage is unknown, do not recommend a TLP change as final. First collect stall reason or per-IP evidence.

### Step 4: Evaluate Candidate TLP Levers

Common XPU levers:

| Lever | Expected occupancy effect | Risk |
|---|---|---|
| Reduce per-thread registers | Higher resident threads | More instructions or recomputation |
| Reduce unroll or tile size | Higher occupancy | Lower locality or more loop overhead |
| Change work-group size | More scheduling candidates | Worse locality or synchronization |
| Change SIMD/vector width | Different thread count and memory shape | Coalescing or instruction mix changes |
| Reduce SLM usage | More resident work-groups | More global/L3 traffic |

Always pair a TLP lever with a risk check: GRF spills, SLM pressure, SEND pressure, memory hierarchy bytes, and co-issue.

### Step 5: Decide

Produce one of these verdicts:

| Verdict | Meaning |
|---|---|
| `add-threads-helps` | Occupancy is low/moderate, stalls are Stage-1, and resource pressure is not already saturated |
| `occupancy-saturated` | Occupancy is already high enough; extra threads are unlikely to move time |
| `tlp-would-worsen-stage2` | Dominant issue is Pipe/Send contention; more threads likely increase contention |
| `need-stall-attribution-first` | Occupancy is suspicious but stall stage is unknown |

### Step 6: Emit TLP Card

```markdown
## TLP / Occupancy Analysis: <kernel>

### Current State
| Metric | Value | Interpretation |
|---|---:|---|
| occupancy | | |
| S0 stall | | |
| S1 single-pipe | | |
| S2 co-issue | | |
| dominant stall stage | | |

### Candidate Change
- lever:
- expected occupancy movement:
- expected risk:

### Verdict
- verdict:
- why:
- validation counters:
```

### Step 7: Validate a TLP Change

After applying a candidate change:
- Compare kernel time.
- Re-collect `ComputeBasic`.
- Confirm occupancy moved in the expected direction.
- Confirm Stage-1 stalls decreased if the verdict was `add-threads-helps`.
- Confirm Stage-2 Pipe/Send did not rise enough to erase the gain.
- Check memory hierarchy counters and ASM/GRF spill if register pressure changed.

## Pitfalls

- More threads hide latency only when the main loss is lack of ready candidates.
- More TLP can convert Stage-1 distance bubbles into Stage-2 pipe or SEND contention.
- Occupancy alone is not a performance metric; it must be interpreted with stall stage and instruction mix.
- Reducing tile size to improve occupancy can hurt locality or increase global traffic.

Custom skills applied: eu-tlp-occupancy.
