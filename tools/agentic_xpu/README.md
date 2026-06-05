<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# Agentic XPU

AI agent workflows for Intel XPU engineering on PyTorch. Each scenario uses an
LLM-powered coding agent to automate a multi-step workflow that would otherwise
require expert manual intervention.

This page is an overview. For setup and usage, follow the README inside each
scenario folder.

## Scenarios

### 1. Nightly CI UT Fix

Given an Intel XPU nightly CI failure report, the agent triages, reproduces,
categorizes the root cause, applies a fix, and writes a structured summary for
each failing test case.

![Nightly CI UT Fix Workflow](assets/nightly_ci_ut_fix_workflow.png)

Starting point: [nightly_ci_fix/README.md](nightly_ci_fix/README.md)

### 2. CUDA-XPU Alignment

Scans `pytorch/pytorch` for issues, PRs, and bug-fix commits across other
backends (CUDA, ROCm, CPU, MPS) that may also affect Intel XPU through shared
code paths, then drives them through formatting, verification, triage, and fix
as a full loop.

![CUDA-XPU Alignment Workflow](assets/cuda_xpu_alignment.png)

Starting point: [xpu_alignment/README.md](xpu_alignment/README.md)

### 3. OOB Model Performance Analysis

Analyzes out-of-box model performance on Intel XPU (with NVIDIA CUDA for
comparison) from Jenkins profiling artifacts, producing per-model and fleet-level
roofline and per-op breakdowns.

![OOB Model Performance Analysis Workflow](assets/model_analysis_workflow.png)

Starting point: [oob_perf_analysis/README.md](oob_perf_analysis/README.md)

### 4. Scalable UT Issue Fix

Handles UT-related issues surfaced while scaling XPU test coverage. The issue
handler validates that an issue still exists, triages it as device-agnostic or
device-specific, and loops on fix-and-verify until the issue is resolved or the
attempt budget is reached.

![Scalable UT Issue Fix Workflow](assets/issue_handler.png)

Starting point: [issue_handler/README.md](issue_handler/README.md)
