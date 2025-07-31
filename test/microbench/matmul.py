import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
shape_list = [
    (4, 4096, 50400),
    (4, 2048, 32000),
    (4, 4096, 128256),
    (4, 5120, 32000),
    (4, 3072, 32064),
    (4, 4096, 50272),
    (4, 4096, 250880),
    (4, 2560, 32000),
    (4, 2048, 50272),
    (4, 1792, 250880),
]


def matmul(m, n, k, dtype, backward):
    m1 = torch.rand(2, m, k).type(dtype).to(device)
    m2 = torch.rand(k, n).type(dtype).to(device)
    if backward:
        m1.requires_grad_(True)
        m2.requires_grad_(True)
    output = torch.matmul(m1, m2)

    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            # warm up
            matmul(shape[0], shape[2], shape[1], dtype, backward)

            # go
            print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    matmul(shape[0], shape[2], shape[1], dtype, backward)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                matmul(shape[0], shape[2], shape[1], dtype, backward)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
