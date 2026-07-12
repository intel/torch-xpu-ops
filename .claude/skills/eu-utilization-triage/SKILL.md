---
name: eu-utilization-triage
description: Lightweight entry workflow for EU efficiency triage. Use when asked whether an XPU kernel is EU-bound, where EU time is going, how to interpret stall/single-pipe/co-issue, or which focused EU analysis skill to run next.
---

# EU Utilization Triage

Use one cheap counter pass to classify where loaded EU time goes and route to the right focused workflow.

This skill is an entry point. It does not replace source-code roofline analysis. Use `kernel-perf-analysis` first when you still need to decide whether the kernel is memory-hierarchy, compute-pipe, or cache-traffic limited.

## When to Use

Use this skill when:
- A target XPU kernel is already identified.
- You have or can collect `ComputeBasic` counters.
- The question is whether EU time is dominated by stall, single-pipe execution, low co-issue, or low occupancy.
- You need to choose between stall attribution, ILP/co-issue analysis, and TLP/occupancy analysis.

Do not use this as the first step for broad workload triage, multi-kernel analysis, or roofline memory-hierarchy closure.

## Inputs

| Input | Meaning |
|---|---|
| `repro-cmd` | Command that launches the target kernel |
| `kernel-filter` | Optional substring/regex for the target kernel |
| `ComputeBasic` counters | At minimum `XVE_STALL`, `XVE_ACTIVE`, `XVE_MULTIPLE_PIPE_ACTIVE`, `XVE_THREADS_OCCUPANCY_ALL` |

## Workflow

### Step 1: Collect the Three-State Portrait

Collect `ComputeBasic` for the target kernel and compute:

| State | Meaning | XPU counter basis |
|---|---|---|
| S0 stall | Loaded EU cycles that are stalled | `XVE_STALL` |
| S1 single-pipe | Active cycles without co-issue | `XVE_ACTIVE - XVE_MULTIPLE_PIPE_ACTIVE` |
| S2 co-issue | Multiple pipes active in the same cycle | `XVE_MULTIPLE_PIPE_ACTIVE` |

Also record `XVE_THREADS_OCCUPANCY_ALL`.

### Step 2: Check Occupancy First

If `XVE_THREADS_OCCUPANCY_ALL` is critically low, route to `eu-tlp-occupancy` before interpreting S0/S1/S2 too aggressively. Low occupancy means many EUs have too few loaded threads, so optimizing ILP or stall on the few active threads may have poor whole-device impact.

### Step 3: Compute Efficiency

Use the two-factor view:

```text
efficiency = (1 - stall_fraction) * coissue / (single_pipe + coissue)
```

Keep both factors visible:
- `(1 - stall_fraction)` says whether S0 is the main loss.
- `coissue / (single_pipe + coissue)` says whether active cycles mostly co-issue or stay single-pipe.

### Step 4: Route

| Observation | Route |
|---|---|
| Stall high, typically about 30% or higher | `eu-stall-attribution` |
| Stall low but co-issue share low | `eu-ilp-coissue` |
| Occupancy low or moderate and stall appears candidate-starved | `eu-tlp-occupancy` |
| Stall low, co-issue healthy, occupancy healthy | Stop; EU utilization is not the main issue |

### Step 5: Emit Triage Card

```markdown
## EU Utilization Triage

### Target
- kernel:
- repro:

### Three-State Portrait
| Metric | Value |
|---|---:|
| S0 stall | |
| S1 single-pipe | |
| S2 co-issue | |
| occupancy | |
| efficiency | |

### Route
- routed workflow:
- reason:
- evidence needed next:
```

## Pitfalls

- Do not route to ILP just because co-issue is low if stall is dominant.
- Do not route to TLP when stall is clearly Stage-2 resource contention; more threads may worsen resource pressure.
- Do not add S0/S1/S2 to idle time. The portrait is for loaded EU cycles.
- Do not use this skill to explain exact stall cause; use `eu-stall-attribution` with stall reason and ASM evidence.

Custom skills applied: eu-utilization-triage.
