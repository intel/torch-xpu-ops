# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch

device = "xpu"

SORT_BY = "self_xpu_time_total"


@torch.compile
def compiled_fn(x):
    x = x + 1.0
    x = x * x
    x = x + 2.0
    return x


def run_profile():
    input = torch.randn(128, 128, device=device)

    # warm
    compiled_fn(input)
    print("[info] finish warm up")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as p:
        print("[info] start running")
        compiled_fn(input)
    return p


if __name__ == "__main__":
    print(run_profile().key_averages().table(sort_by=SORT_BY, row_limit=-1))
