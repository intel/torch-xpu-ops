# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch

SORT_BY = "self_xpu_time_total"

# The timestamp precision bug only shows up over many short profiling windows.
DEFAULT_ITERS = 1000


def compute(input1, input2):
    input1 = input1.to(device="xpu")
    return input1 + 1.0


def run_profile(iters=DEFAULT_ITERS):
    """Yield (iteration, prof) so callers can stream results instead of
    keeping ``iters`` profiler objects alive at once."""
    input1 = torch.randn(3, 3, device="cpu")
    input2 = torch.randn(3, 3, device="cpu")

    # warm up
    compute(input1, input2)

    for i in range(iters):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as p:
            compute(input1, input2)
        yield i, p


if __name__ == "__main__":
    for _, prof in run_profile():
        print(prof.key_averages().table(sort_by=SORT_BY, row_limit=-1))
