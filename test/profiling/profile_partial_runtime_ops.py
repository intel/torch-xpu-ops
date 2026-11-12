# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch

SORT_BY = "self_xpu_time_total"

# The runtime ops the XPU profiler is expected to report for this workload.
EXPECTED_RUNTIME_OPS = {"urEnqueueKernelLaunchWithArgsExp", "urEnqueueUSMMemcpy"}


def compute(input1, input2):
    input1 = input1.to(device="xpu")
    return input1 + 1.0


def run_profile():
    input1 = torch.randn(3, 3, device="cpu")
    input2 = torch.randn(3, 3, device="cpu")

    # warm up
    compute(input1, input2)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as p:
        compute(input1, input2)
    return p


if __name__ == "__main__":
    print(run_profile().key_averages().table(sort_by=SORT_BY, row_limit=-1))
