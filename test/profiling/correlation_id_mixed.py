# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch

SORT_BY = "xpu_time_total"


def run_profile():
    input1 = torch.randn(3, 3, device="xpu")
    input2 = torch.randn(3, 3, device="xpu")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as prof:
        output1 = input1 + 1.0
        output2 = input2 + 2.0
        output = output1 + output2
    return prof


if __name__ == "__main__":
    print(run_profile().key_averages().table(sort_by=SORT_BY))
