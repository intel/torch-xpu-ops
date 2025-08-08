import time

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
shape_list = [(8192, 8192)]
num_iter = 20

cache_r = torch.randn((1024 * 1024 * 1024), device="xpu")
cache_w = torch.randn((1024 * 1024 * 1024), device="xpu")


def _test_dpcpp(input, target, reduce, dtype):
    loss = nn.MultiLabelMarginLoss(reduction=reduce)
    input.requires_grad = True

    if reduce == "none":
        # warm up
        output = loss(input, target)
        output.backward(torch.ones_like(output, dtype=dtype).to("xpu"))

        # go
        print(
            "shape:",
            (shape),
            "; datatype:",
            dtype,
            "; backward:",
            backward,
            "; reduce: 0" if (reduce == "none") else "; reduce: 1",
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                cache_r = cache_w
                output = loss(input, target)
                cache_r = cache_w
                output.backward(torch.ones_like(output, dtype=dtype).to("xpu"))
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            cache_r = cache_w
            output = loss(input, target)
            cache_r = cache_w
            output.backward(torch.ones_like(output, dtype=dtype).to("xpu"))
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")

    else:
        # warm up
        output = loss(input, target)
        output.backward(torch.tensor((1.0), dtype=dtype).to("xpu"))

        # go
        print(
            "shape:",
            (shape),
            "; datatype:",
            dtype,
            "; backward:",
            backward,
            "; reduce: 0" if (reduce == "none") else "; reduce: 1",
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(num_iter):
                cache_r = cache_w
                output = loss(input, target)
                cache_r = cache_w
                output.backward(torch.tensor((1.0), dtype=dtype).to("xpu"))
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            cache_r = cache_w
            output = loss(input, target)
            cache_r = cache_w
            output.backward(torch.tensor((1.0), dtype=dtype).to("xpu"))
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")


for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for reduce in ["none", "mean"]:
            input = torch.randn(shape, dtype=dtype)
            target = torch.randn(shape, dtype=dtype).long()
            input_dpcpp = input.to("xpu")
            target_dpcpp = target.to("xpu")
            _test_dpcpp(input_dpcpp, target_dpcpp, reduce, dtype)
