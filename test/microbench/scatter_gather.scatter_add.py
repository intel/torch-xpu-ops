import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    # shape, dim
    ((28, 4096, 9, 1), 2),  # LQCD shape
    ((512, 36, 4, 1), 2),
    ((4, 4096, 4096), 0),  # big shape
    ((2048, 4, 4096), 0),
    ((2048, 4096, 4), 0),
    ((2048, 4096, 4096), 0),
    ((4096, 8192, 8192), 0),
    ((4097, 8193, 8193), 0),
    ((4, 4096, 4096), 1),  # big shape
    ((2048, 4, 4096), 1),
    ((2048, 4096, 4), 1),
    ((2048, 4096, 4096), 1),
    ((4096, 8192, 8192), 1),
    ((4097, 8193, 8193), 1),
]

device = "xpu"
backward = False
num_iter = 20

g_xpu = torch.Generator(device=device)
g_xpu.manual_seed(25)
torch.manual_seed(25)


def Scatter_add(shape, dtype, dim, device):
    if dim == 2:
        m, n, k1, k2 = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
        src = torch.ones((m, n, k1), dtype=dtype, device=device)
        index = torch.randint(0, k2, (m, n, k1), generator=g_xpu, device=device)
        zeros = torch.zeros(m, n, k2, dtype=dtype, device=device)
    else:
        if dim == 0:
            m1, m2, n = shape[0][0], shape[0][1], shape[0][2]
            src = torch.ones((m1, n), dtype=dtype, device=device)
            index = torch.randint(0, m2, (m1, n), generator=g_xpu, device=device)
            zeros = torch.zeros(m2, n, dtype=src.dtype, device=device)
        else:
            m, n1, n2 = shape[0][0], shape[0][1], shape[0][2]
            src = torch.ones((m, n1), dtype=dtype, device=device)
            index = torch.randint(0, n2, (m, n1), generator=g_xpu, device=device)
            zeros = torch.zeros(m, n2, dtype=src.dtype, device=device)

    dst = zeros.scatter_add_(dim, index, src)


if __name__ == "__main__":
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            dim = shape[1]
            # warm up
            Scatter_add(shape, dtype, dim, device)

            # go
            print(
                "shape:",
                shape[0],
                "; datatype:",
                dtype,
                "; dim:",
                dim,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    Scatter_add(shape, dtype, dim, device)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # E2E time
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                Scatter_add(shape, dtype, dim, device)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            print("E2E total time:", f"{float(e2e_time):.20f}")
