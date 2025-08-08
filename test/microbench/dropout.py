import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192), (16, 1024)]
num_iter = 20

if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            H, W = (shape[0], shape[1])
            input = torch.randn((H, W)).to(dtype=dtype, device="xpu")

            dropout = torch.nn.Dropout(p=0.5)
            dropout.to(device="xpu", dtype=dtype)
            grad_dpcpp = torch.randn((H, W)).to(device="xpu", dtype=dtype)
            input.requires_grad_(True)

            # warm up
            output = dropout(input)
            if backward:
                output.backward(grad_dpcpp)

            # go
            print(
                "shape:",
                (shape),
                "; datatype:",
                dtype,
                "; P:",
                0.5,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    output = dropout(input)
                    if backward:
                        output.backward(grad_dpcpp)
            print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                output = dropout(input)
                if backward:
                    output.backward(grad_dpcpp)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
