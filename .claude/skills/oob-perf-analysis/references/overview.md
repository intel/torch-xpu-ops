# OOB Perf Analysis Overview

This skill package handles OOB eager-mode performance analysis based on T1/T2/R roofline methodology.

Primary user intents:

1. Start from Jenkins session inputs and produce reports
2. Use existing raw artifacts to generate per-model reports
3. Generate a fleet summary across models
4. Compare graph consistency between CUDA and XPU
5. Produce a concise insights summary for XPU developers
6. Diagnose abnormal R values, missing artifacts, or trace mismatches

Core outputs:

1. Raw session layout under `raw_logs/<session>/`
2. Per-model markdown reports under `reports/<session>/models/`
3. Fleet summary markdown under `reports/<session>/`
4. Optional graph consistency and insights outputs

Use the other reference files for workflow details.
