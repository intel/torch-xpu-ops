# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import time

import torch
from torch.profiler import profile, ProfilerActivity

# Dtype alias mapping for CLI convenience
DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


def normalize_dtype(dtype):
    if isinstance(dtype, str):
        return DTYPE_MAP.get(dtype, dtype)
    return dtype


def run_profile(op_run, config, device, num_iter):
    activity = (
        ProfilerActivity.XPU
        if device == "xpu"
        else ProfilerActivity.CUDA if device == "cuda" else ProfilerActivity.CPU
    )
    with profile(
        activities=[ProfilerActivity.CPU, activity], record_shapes=True
    ) as prof:
        for _ in range(num_iter):
            op_run(config, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(op_run, config, device, num_iter):
    sync = (
        torch.xpu.synchronize
        if device == "xpu"
        else torch.cuda.synchronize if device == "cuda" else lambda: None
    )
    sync()
    t0 = time.time()
    for _ in range(num_iter):
        op_run(config, device)
    sync()
    return (time.time() - t0) / num_iter


def run_case(op_run, config, args):
    """Run one case with warmup, profile, and E2E timing."""
    device = args.device
    # Warm-up
    op_run(config, device)

    # Profile
    if not args.e2e_only:
        run_profile(op_run, config, device, args.num_iter)

    # E2E
    if not args.profile_only:
        e2e = run_e2e(op_run, config, device, args.num_iter)
        print("E2E total time:", f"{float(e2e):.20f}\n")
