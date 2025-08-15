import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = False
num_iter = 20
shape_list = [((25, 300), (22, 300), (15, 300)), ((2, 1000), (100, 1000), (8192, 1000))]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for batch_first in [False, True]:
            for padding_value in [0.0, 1.0, 2.0]:
                a = torch.randn(shape[0], device=device, dtype=dtype)
                b = torch.randn(shape[1], device=device, dtype=dtype)
                c = torch.randn(shape[2], device=device, dtype=dtype)

                if backward:
                    a.requires_grad_(True)
                    b.requires_grad_(True)
                    c.requires_grad_(True)

                # warm up
                output = torch.nn.utils.rnn.pad_sequence(
                    ([a, b, c]), batch_first, padding_value
                )
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; batch_first:",
                    batch_first,
                    "; padding_value:",
                    padding_value,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        output = torch.nn.utils.rnn.pad_sequence(
                            ([a, b, c]), batch_first, padding_value
                        )
                        if backward:
                            gy = torch.empty_like(output)
                            output.backward(gy)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    output = torch.nn.utils.rnn.pad_sequence(
                        ([a, b, c]), batch_first, padding_value
                    )
                    if backward:
                        gy = torch.empty_like(output)
                        output.backward(gy)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
