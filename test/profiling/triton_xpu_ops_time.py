# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch

device = "xpu"


@torch.compile
def test(x):
    x = x + 1.0
    x = x * x
    x = x + 2.0
    return x


input = torch.randn(128, 128, device=device)

# warm
output = test(input)
print("[info] finish warm up")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ]
) as p:
    print("[info] start running")
    output = test(input)
print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
