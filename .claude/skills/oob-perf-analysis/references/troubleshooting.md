# Troubleshooting

Use this reference when session data, per-model reports, or fleet summaries are incomplete or suspicious.

This document is the canonical source of truth for manual inspection and debugging of traces, unitrace output, T2 extraction, and report anomalies.

## First Triage Order

When results look wrong, check in this order:

1. file existence and layout correctness
2. T2 extraction correctness
3. trace and unitrace presence
4. kernel-sum sanity vs T2 or T2_device
5. projection coverage gaps
6. graph divergence between CUDA and XPU

This order helps avoid spending time on interpretation before confirming the raw inputs are valid.

## Basic File Checks

Confirm that the canonical session layout exists and that the expected files are present for each model.

Minimum useful check list:

1. `t1/calcflops.txt`
2. `xpu_profiler/trace.json`
3. `cuda_profiler/trace.json`
4. `xpu_t2/rcpi1-ins0.log`
5. `cuda_t2/rcpi1-ins0.log`
6. `unitrace/python.<pid>.json` when XPU unitrace is expected

If any of these are missing:

1. mark the model incomplete
2. skip cross-platform reporting for that model
3. do not fabricate placeholder values

## Symptom: No T2 Value Found

Typical causes:

1. runtime failed before printing the final latency line
2. wrong log file downloaded
3. log format changed unexpectedly

Check:

1. whether `rcpi1-ins0.log` exists
2. whether it contains `GPU Time per batch:`
3. whether the model actually finished successfully on that pass

Meaning:

1. without T2, R cannot be computed correctly
2. the model should be treated as incomplete for per-model and fleet summary sections that require R

## Symptom: R Much Greater Than 1

Common causes:

1. projection overcounting
2. hardware peak or bandwidth assumptions too conservative
3. fusion at runtime invalidates the naive per-op projection sum
4. calcflops includes work that no longer exists as standalone runtime kernels

Specific patterns to check:

1. SDPA fusion
   - dispatch log may see unfused attention pieces
   - runtime may execute a fused SDPA kernel
2. dtype cast or copy fusion
3. outdated hardware-spec assumptions

Useful question:

1. Is `R_op > 1.05` happening on many platforms for the same op?

If yes, it is more likely a projection issue than an XPU kernel issue.

## Symptom: R Much Smaller Than Expected

Common causes:

1. real XPU kernel inefficiency
2. projection undercounting
3. profiler overhead inflation
4. significant host overhead between kernels

How to separate them:

1. if XPU `R_op < 0.80` but CUDA `R_op >= 0.80`, suspect XPU kernel inefficiency
2. if all platforms are low on the same op, suspect projection undercounting
3. if `T2 >> T2_device`, suspect host overhead or launch overhead

## Symptom: Unitrace And Profiler Disagree Strongly

This is a high-signal debugging case for XPU.

Use unitrace as the preferred ground truth for XPU per-op actual timing, but first validate the collection.

### Check 1: Unitrace kernel sum vs T2

Expectation:

```text
sum(unitrace kernels) ~= T2
```

Interpretation:

1. if kernel sum is significantly greater than T2, the collection window is wrong or multiple iterations leaked in
2. if kernel sum is moderately smaller than T2, there may be host overhead between kernel launches

### Check 2: Unitrace kernel sum vs profiler kernel sum

Expectation:

```text
sum(unitrace kernels) ~= sum(profiler device kernels)
```

Interpretation:

1. if profiler kernel sum is much larger than unitrace, profiler overhead is inflating the trace
2. in that case, keep unitrace as ground truth for XPU per-op timing

### Common root cause

Without conditional unitrace collection, multiple iterations can leak into one measurement window.

## Symptom: Trace Kernel Sum Looks Inflated

Common causes:

1. profiler instrumentation overhead
2. remote profiling artifacts
3. bad trace generation environment

Check:

1. whether `sum(kernel durations)` exceeds expected GPU time by more than about 10 percent
2. whether the trace was generated locally or through remote SSH

If inflated:

1. treat trace-based per-op timing with caution
2. regenerate trace if possible
3. prefer unitrace for XPU

## Symptom: T2 Coverage By T1 Is Poor

Meaning:

1. actual trace includes significant ops that do not appear in calcflops output
2. projection is missing work and will systematically understate T1

Check the uncovered-op table for:

1. ops that consume meaningful `% T2`
2. repeated uncovered ops across many models

Interpretation:

1. if the same uncovered op appears across many models, it is likely a projection-model gap
2. if uncovered ops are mostly optimizer-style ops in training, some may be expected

## Symptom: XPU And CUDA Graph Mismatch

Common causes:

1. SDPA dispatch difference
2. XPU-specific overrideable op path
3. decomposition differences
4. shape-dependent path divergence

How to interpret:

1. SDPA-only differences are often expected and should be separated from more serious divergence
2. significant FLOPs or memory differences usually mean true graph divergence, not just kernel efficiency differences
3. platform-only ops in trace comparison may indicate fusion or decomposition differences

Do not conclude kernel inefficiency until graph consistency has been checked.

## Symptom: `aten::copy_` Or Data Movement Dominates XPU Trace

Possible explanation:

1. XPU trace may attribute overlapped memory movement in ways that do not contribute directly to wall time
2. this can make profiler-based actual time misleading for those ops

Interpretation rule:

1. do not immediately treat large `copy_` device time as the top optimization target
2. check whether it overlaps with compute and whether it is absent from calcflops modeling

## Symptom: Session Has Partial Pass Success

This is expected in real Jenkins usage.

Recommended handling:

1. preserve all partial artifacts
2. record pass-level incompleteness in metadata or workflow notes
3. allow report generation to skip incomplete models rather than failing the whole session

## Practical Sanity Checks

Use these rules before trusting the final report:

1. `R` must be computed with T2, not T2_device
2. unitrace is XPU-only
3. CUDA uses profiler trace, not unitrace
4. calcflops should use the last benchmark iteration
5. graph consistency should be checked before claiming kernel-only root cause
6. if many models show the same suspicious pattern, suspect the methodology before blaming one kernel

## When To Escalate To Workflow Re-Run

Re-run or recollect data when one of these is true:

1. T2 log is missing or malformed
2. trace is missing
3. unitrace collection clearly spans multiple iterations
4. profiler inflation is severe enough to invalidate per-op timing
5. artifact download is incomplete or corrupted

## When Not To Escalate

Do not force a recollection immediately when:

1. only one platform is missing for one model and the rest of the session is usable
2. graph mismatch is explainable by known SDPA path differences
3. an op looks bad only because projection undercounts consistently on all platforms
